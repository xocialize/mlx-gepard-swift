// GepardDecoder — the streaming AR generation loop tying backbone + audio interface + heads.
//
// Mirrors gepard-inference GepardRunner.generate (argmax / single-pass, no CFG):
//   prefill([prefix | embed(cond_ids)]) → frame 0 from last position (NO stop check)
//   loop: embed prev frame → 1-token decode step (KV cache) → stop-check BEFORE sampling
//         → argmax 32 codes. Stop when sigmoid(stop_logit) > 0.5.
//
// Positions advance via the KV cache offset (prefill fills K+T; each frame is the next
// position), which the backbone's mRoPE reads from `cache.offset`.

import Foundation
import MLX
import MLXNN
import MLXLMCommon

public struct GepardDecoder {
    public let backbone: GepardBackbone
    public let audio: AudioInterface
    public let heads: CodebookHeads

    public init(backbone: GepardBackbone, audio: AudioInterface, heads: CodebookHeads) {
        self.backbone = backbone
        self.audio = audio
        self.heads = heads
    }

    private func lastPos(_ h: MLXArray) -> MLXArray { h[0..., (h.dim(1) - 1)..., 0...] }

    /// inputsEmbeds = [ prefix (K speaker tokens) | embed(cond_ids) ]; returns last hidden + cache.
    public func prefill(prefix: MLXArray, condIds: MLXArray) -> (MLXArray, [KVCache]) {
        let text = backbone.embed(condIds.asType(.int32).reshaped(1, -1))
        let inputs = concatenated([prefix, text], axis: 1)
        let cache = backbone.newCache()
        let (h, _) = backbone(inputsEmbeds: inputs, cache: cache, capture: false)
        return (lastPos(h), cache)
    }

    public func decodeStep(frameEmbed: MLXArray, cache: [KVCache]) -> MLXArray {
        let (h, _) = backbone(inputsEmbeds: frameEmbed, cache: cache, capture: false)
        return lastPos(h)
    }

    /// 32 argmax codes from a [1,1,d] hidden state (head order == codebook order).
    /// Codebooks have different vocab sizes, so argmax each, then concat into ONE [32]
    /// tensor and read it back with a single GPU→CPU sync (vs 32 per-code `.item()` stalls —
    /// the dominant per-frame cost since each frame's tensor work is tiny).
    public func argmaxFrame(logits: [MLXArray]) -> [Int] {
        let idx = concatenated(logits.map { argMax($0, axis: -1).reshaped([1]) }, axis: 0)  // [32]
        return idx.asArray(Int32.self).map { Int($0) }
    }

    public struct Rollout {
        public let codes: [[Int]]       // T frames × 32 codes
        public let stopProbs: [Float]   // one per decode step (1..)
    }

    /// Runtime generation knobs (runner.py's inference options that survive into the engine).
    /// CFG is the onset-only text classifier-free-guidance path (tech report §5.3): the
    /// short-utterance rescue. `cfgScale == nil` ⇒ single-pass (the P6-validated greedy path).
    public struct Options: Sendable {
        public var maxFrames: Int
        /// Text-CFG weight `w` in `guided = uncond + w·(cond − uncond)`. nil ⇒ CFG off.
        public var cfgScale: Float?
        /// Onset window — CFG is applied only to the first `cfgFrames` frames.
        public var cfgFrames: Int
        /// The uncond text ids (`[SOT, EOT, SOS]`); required when `cfgScale != nil`.
        public var uncondIds: [Int]?
        /// Sigmoid threshold on the stop head (oracle runner.py `stop_threshold`, default 0.5).
        /// The stop head is conditioned on the speaker prefix, so some reference clips push it
        /// past 0.5 at a mid-utterance sentence pause — raising the threshold rescues those
        /// premature stops (at the cost of occasional trailing babble at 0.9+).
        public var stopThreshold: Float
        /// Earliest step at which a stop-head crossing is honored (0 = oracle behavior).
        /// With some (reference clip, text) pairs the prefix-conditioned head crosses within
        /// the first frames — before any speech (AB-L-0075). The floor ignores those
        /// crossings; whether the decode then produced actual SPEECH is judged above this
        /// layer, by audio energy (the stop-head trajectory alone cannot tell: onset-high
        /// heads occur in healthy runs, near-threshold wiggles occur in stillborn ones).
        public var minFrames: Int

        public init(maxFrames: Int = 2000, cfgScale: Float? = nil,
                    cfgFrames: Int = 20, uncondIds: [Int]? = nil,
                    stopThreshold: Float = 0.5, minFrames: Int = 0) {
            self.maxFrames = maxFrames
            self.cfgScale = cfgScale
            self.cfgFrames = cfgFrames
            self.uncondIds = uncondIds
            self.stopThreshold = stopThreshold
            self.minFrames = minFrames
        }
    }

    /// Per-head CFG combine: `uncond + w·(cond − uncond)` on fp32 logits (runner.py).
    private func guided(cond: [MLXArray], uncond: [MLXArray], w: Float) -> [MLXArray] {
        zip(cond, uncond).map { lc, lu in lu + w * (lc - lu) }
    }

    /// Backward-compatible greedy rollout (single-pass, no CFG, no cancellation) — the exact
    /// loop the P6/P8 gates validate. Delegates to `rollout` with default options; the throwing
    /// path never throws when `cancelCheck` is nil, so the `try!` is total.
    public func greedyRollout(prefix: MLXArray, condIds: MLXArray, maxFrames: Int) -> Rollout {
        try! rollout(prefix: prefix, condIds: condIds,
                     options: Options(maxFrames: maxFrames), cancelCheck: nil, onFrame: nil)
    }

    /// The engine generation loop: greedy per-codebook argmax with optional onset text-CFG,
    /// cooperative cancellation, and a per-frame progress hook.
    ///
    /// `cancelCheck` is invoked once per generated frame (the natural yield point, contract's
    /// CAN cadence); it rethrows its `CancellationError` UNCHANGED so the engine can classify
    /// user-cancel vs governor-preempt. `onFrame(count)` fires after each frame is committed
    /// (1-based count), for `RunProgress` reporting from the wrapper. `onFrameCodes` receives
    /// each committed frame's 32 codebook indices in order (frame 0 included) — the streaming
    /// emit seam (contract 1.25.0): a chunked codec decode hooks here without disturbing the
    /// AR/KV state.
    public func rollout(
        prefix: MLXArray, condIds: MLXArray, options: Options,
        cancelCheck: (() throws -> Void)?, onFrame: ((Int) -> Void)?,
        onFrameCodes: (([Int]) -> Void)? = nil
    ) rethrows -> Rollout {
        // Cond stream.
        var (hidden, cache) = prefill(prefix: prefix, condIds: condIds)

        // Uncond stream (onset-CFG only): a parallel KV cache primed on the empty-text prompt,
        // fed the SAME sampled audio frames as the cond stream. Dropped once the onset closes.
        let cfgOn = options.cfgScale != nil && (options.uncondIds?.isEmpty == false) && options.cfgFrames > 0
        let w = options.cfgScale ?? 0
        var uCache: [KVCache]? = nil
        var uHidden: MLXArray? = nil
        if cfgOn, let uncondIds = options.uncondIds {
            let idsArr = MLXArray(uncondIds.map { Int32($0) })
            let (uh, uc) = prefill(prefix: prefix, condIds: idsArr)
            uHidden = uh; uCache = uc
        }

        // Frame 0 — sampled from the prefill's last position (NO stop check).
        let (l0, _) = heads(hidden)
        let frame0: [Int]
        if cfgOn, let uh = uHidden {
            frame0 = argmaxFrame(logits: guided(cond: l0, uncond: heads(uh).logits, w: w))
        } else {
            frame0 = argmaxFrame(logits: l0)
        }
        var frames: [[Int]] = [frame0]
        var stops: [Float] = []
        onFrame?(1)
        onFrameCodes?(frame0)

        for step in 1 ..< options.maxFrames {
            try cancelCheck?()
            let fe = audio.frameEmbed(codes: frames[frames.count - 1])
            hidden = decodeStep(frameEmbed: fe, cache: cache)
            let (logits, stop) = heads(hidden)
            // Stop is read from the COND stream (runner.py) — forces eval, bounds the graph.
            let p = sigmoid(stop).item(Float.self)
            stops.append(p)
            if p > options.stopThreshold, step >= options.minFrames { break }

            if cfgOn, step < options.cfgFrames, let uc = uCache {
                let uh = decodeStep(frameEmbed: fe, cache: uc)   // same audio frame, empty text ctx
                uHidden = uh
                frames.append(argmaxFrame(logits: guided(cond: logits, uncond: heads(uh).logits, w: w)))
            } else {
                frames.append(argmaxFrame(logits: logits))
            }
            onFrame?(frames.count)
            onFrameCodes?(frames[frames.count - 1])
        }
        _ = uHidden
        return Rollout(codes: frames, stopProbs: stops)
    }
}
