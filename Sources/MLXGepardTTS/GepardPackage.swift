import Foundation
import GepardCore
import MLX
import MLXRandom
import MLXToolKit
import OSLog

/// Gepard-1.0 on the canonical `tts` surface: zero-shot voice cloning from a reference clip,
/// with a streaming-first architecture (5.3× realtime, 22 ms TTFA — a companion/realtime
/// voice). Returns the canonical `Audio` (.wav, 22.05 kHz mono).
///
/// Engine-owned lifecycle (C13): the engine constructs from a `GepardConfiguration`, pages
/// weights in with `load()` (auto-materializing the two declared sources under the engine's
/// models root when set), drives `run(_:)`, and reclaims with `unload()`.
///
/// Voice: `.referenceAudio` only (Gepard is a zero-shot cloner — no preset voices; `.auto` /
/// `.named` reject legibly via `unsupportedRequestFeature`). `referenceTranscript` is not
/// consumed (conditioning is transcript-free — the Q-Former compresses codec codes, not text).
///
/// `metaData` keys (package-specific, C5):
/// - `cfgScale` (double): onset text-CFG weight `w` (tech report §5.3, w≈2.6 lifts 1–2-word
///   success 7.5%→48.8%). Absent ⇒ CFG off (the P6-validated greedy path). The companion's
///   auto-on-for-short-text policy lives app-side; the capability just exposes the knob.
/// - `cfgFrames` (int, default 20): onset window CFG is applied over.
/// - `maxFrames` (int, default 2000 ≈ 93 s): hard generation cap.
/// - `stopThreshold` (double, default 0.5): sigmoid threshold on the stop head (oracle
///   `stop_threshold`). The head is prefix-conditioned — clips whose prefix pushes it past
///   0.5 at a sentence pause truncate multi-sentence text; 0.7–0.9 rescues those.
/// - `seed` (int): reproducible sampling, clamped to 32 bits. NOTE: V1 decoding is deterministic
///   argmax, so the seed is currently a no-op — accepted for forward-compat when a temperature
///   sampling path lands.
@InferenceActor
public final class GepardPackage: ModelPackage, StreamEmitting {
    public typealias Configuration = GepardConfiguration

    /// Split footprints — MEASURED via the headless GEPARD_VALIDATE harness (gepard-gates
    /// --validate, 2026-07-10, M5 Max, roxy_a1 ref + 12.8 s paragraph). bf16 LM (native) + fp32
    /// NanoCodec (weight-norm fused).
    ///   • resident floor (MLX-active, post-load): 1060 MB · phys_footprint 1157 MB
    ///   • run-phase peak (MLX high-water, 275-frame utterance): 4856 MB ⇒ transient ≈ 3.8 GB
    /// The transient DOMINATES the residents and scales with utterance length because V1 decodes
    /// the WHOLE utterance through the deep NanoCodec conv stack in one pass (the plan's V2
    /// frame-streaming decode would shrink it). Declared with margin for longer single sentences;
    /// operationally the app sentence-chunks (tts-orchestrator-kit), so one run() ≈ one sentence.
    /// The reactive R-MEM-1 trigger covers any overflow past the declared peak.
    nonisolated static let bf16ResidentBytes: UInt64 = 1_200_000_000
    nonisolated static let peakActivationBytes: UInt64 = 4_000_000_000

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // C7: two weight roles — LM Apache-2.0, NanoCodec NVIDIA Open Model License. Declare
            // the MORE-RESTRICTIVE of the two (the codec's .nvidiaOpenModel); it is on the
            // permissive allowlist (engine ≥0.28.0), so the package admits under the default
            // `.permissiveOnly`. C8: the port code (this repo) is Apache-2.0.
            license: LicenseDeclaration(weightLicense: .nvidiaOpenModel, portCodeLicense: .apache2),
            provenance: Provenance(
                sourceRepo: "nineninesix/gepard-1.0",
                revision: "63ff207b93bf3a590018dd5e897e02e19fdd4601", tier: 2),
            requirements: RequirementsManifest(
                footprints: [
                    QuantFootprint(quant: .bf16, residentBytes: bf16ResidentBytes,
                                   peakActivationBytes: peakActivationBytes),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [
                // Gepard has no emotion/duration control (no E12 lane). Its selection signal is
                // zero-shot voice cloning + a realtime/streaming architecture — the companion voice.
                SpecialtyWeight(.voiceClone, strength: 1.0),
                SpecialtyWeight(.realtimeStreaming, strength: 1.0),
                SpecialtyWeight(.companion, strength: 0.8),
            ],
            surfaces: [
                TTSContract.descriptor(
                    name: "gepard",
                    summary: "Gepard-1.0 zero-shot voice-cloning streaming TTS (.wav, 22.05 kHz "
                        + "mono). Realtime companion voice (~5× realtime, ~22 ms TTFA) cloned from "
                        + "a reference clip. Requires voice.referenceAudio (no preset voices). "
                        + "Onset text-CFG (metaData.cfgScale) rescues 1–2-word utterances. "
                        + "Streams PCM chunks (contract 1.25.0): first audio ≈ 6 frames / 280 ms "
                        + "of speech, exact windowed causal decode.",
                    modes: [.neutral, .expressive],
                    streaming: .audioChunk
                )
            ]
        )
    }

    private let configuration: Configuration
    private var model: GepardModel?
    // Reference-conditioning reuse (the IndexTTS2/Qwen3 E1 pattern): long-form synthesis sends
    // the SAME reference for every line; preparing it re-runs the codec encoder + Q-Former.
    // Memoize the frozen speaker prefix keyed by the reference clip bytes. Safe to hold:
    // InferenceActor serializes run(), and the prefix is read-only once built.
    /// Memoized speaker prefixes, keyed by (reference bytes, rescue nudge). Nudge 0 is the
    /// clip as given; nudge k re-encodes with a small deterministic gain change (AB-L-0075
    /// stillborn-decode rescue). Bounded — cleared on unload, and tiny in practice.
    private var cachedPrefixes: [String: MLXArray] = [:]
    private static let rescueLog = Logger(subsystem: "MLXGepardTTS", category: "rescue")

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    // MARK: - Lifecycle

    public func load() async throws {
        guard model == nil else { return }

        // Materialization is ENGINE-EXECUTED since contract 1.24 (engine ≥ 0.32.0): the engine
        // downloads the declared missing sources into the store BEFORE load(). This guard is the
        // offline backstop only — reaching it means no engine materialization ran (no store set,
        // or a non-engine caller) and the sources genuinely aren't on disk.
        let storeRoot = configuration.modelsRootDirectory
        let missing = configuration.missingWeightSources(storeRoot: storeRoot)
        guard missing.isEmpty else {
            throw GepardError.missingWeights(
                "sources not materialized: \(missing.map(\.role).joined(separator: ", ")) "
                + (storeRoot.map { "(store: \($0.path))" } ?? "(no models root set)"))
        }
        try Task.checkCancellation()

        let resolved = configuration.resolved(storeRoot: storeRoot)
        guard let modelDir = resolved.modelDirectory, let codecDir = resolved.codecDirectory else {
            throw GepardError.missingWeights("unresolved weight directories (no store root)")
        }
        let modelURL = modelDir.appending(path: GepardConfiguration.mainProbeFile)
        let codecURL = codecDir.appending(path: GepardConfiguration.codecProbeFile)

        // Heavy: pages ~1.1 GB LM (bf16) + the fp32-fused codec; tokenizer loads off the same dir.
        model = try await GepardModel.load(
            modelURL: modelURL, codecURL: codecURL, tokenizerFolder: modelDir)
    }

    public func unload() async {
        model = nil
        cachedPrefixes.removeAll()
        MLX.Memory.clearCache()   // release the retained MLX pool so eviction frees RSS
    }

    // MARK: - Run

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: the entry checkpoint is the FIRST act of run() — before notLoaded validation
        // (engine ≥ 0.27.0). Mid-run cadence: the AR rollout bails per generated frame via the
        // throwing `cancelCheck` closure threaded into GepardModel.synthesize, rethrowing the
        // CancellationError unchanged so the engine can classify user-cancel vs governor-preempt.
        try Task.checkCancellation()
        let (model, prefix, condIds, options, clip, rescueAttempts) = try conditioning(for: request)
        try Task.checkCancellation()

        // Rollout → dequant → vocode, with the stillborn-decode rescue loop (AB-L-0075):
        // a saturated attempt produced nothing; re-encode the reference with a tiny gain
        // nudge (deterministic, identity-preserving) and try again.
        var samples: [Float] = []
        for attempt in 0 ... rescueAttempts {
            let attemptPrefix = attempt == 0 ? prefix
                : try speakerPrefix(for: clip, nudge: attempt)
            let synthesis = try model.synthesize(
                prefix: attemptPrefix, condIds: condIds, options: options,
                cancelCheck: { try Task.checkCancellation() },
                onFrame: { count in RunProgress.report(.generate, step: count) })
            if !synthesis.saturated {
                if attempt > 0 {
                    Self.rescueLog.info("stillborn decode rescued at nudge \(attempt)")
                }
                samples = synthesis.samples
                break
            }
            if attempt == rescueAttempts {
                Self.rescueLog.error("stillborn decode persisted through \(rescueAttempts) nudges — emitting silence")
            } else {
                Self.rescueLog.info("stillborn decode (attempt \(attempt)) — nudging reference")
            }
        }

        try Task.checkCancellation()
        RunProgress.report(.decode)
        let wav = AudioSupport.encodeWAV16(samples: samples, sampleRate: GepardModel.sampleRate)
        return TTSResponse(audio: Audio(
            format: .wav, data: wav, sampleRate: GepardModel.sampleRate, channels: 1))
    }

    // MARK: - Streaming (contract 1.25.0, StreamEmitting)

    /// Streaming twin of `run()`: same conditioning, same AR rollout, but the NanoCodec decode
    /// is windowed + incremental (exact — the decoder stack is causal), with each chunk handed
    /// to `emit` synchronously from the run loop. Returns the same aggregated `.wav` response
    /// `run()` would have produced (STR-5 parity).
    public func runStream(_ request: any CapabilityRequest,
                          emit: @escaping @Sendable (TTSStreamChunk) -> Void)
        async throws -> any CapabilityResponse {
        try Task.checkCancellation()   // STR-2: entry checkpoint precedes the first emit
        let (model, prefix, condIds, options, clip, rescueAttempts) = try conditioning(for: request)
        try Task.checkCancellation()

        // Streaming cadence knobs (metaData plane, all optional): chunk sizes trade emit
        // overhead vs latency; `streamContextFrames` is the windowed-decode context override
        // (diagnostic — the --stream gate's full-prefix probe; default = computed receptive
        // field).
        var chunking = GepardModel.StreamingChunkOptions()
        if let v = tts(request)?.metaData.intValue("streamFirstChunkFrames") {
            chunking.firstChunkFrames = max(1, v)
        }
        if let v = tts(request)?.metaData.intValue("streamChunkFrames") {
            chunking.chunkFrames = max(1, v)
        }
        if let v = tts(request)?.metaData.intValue("streamContextFrames") {
            chunking.contextFrames = max(0, v)
        }

        var chunkIndex = 0
        // Stillborn-decode rescue (AB-L-0075): a saturated attempt emits NOTHING (the model
        // layer gates emission past the probe window), so retrying is invisible downstream.
        var samples: [Float] = []
        for attempt in 0 ... rescueAttempts {
            let attemptPrefix = attempt == 0 ? prefix
                : try speakerPrefix(for: clip, nudge: attempt)
            let synthesis = try model.synthesizeStreaming(
                prefix: attemptPrefix, condIds: condIds, options: options,
                chunking: chunking,
                onChunk: { chunkSamples, isFinal in
                    emit(TTSStreamChunk(samples: chunkSamples,
                                        sampleRate: GepardModel.sampleRate,
                                        index: chunkIndex, isFinal: isFinal))
                    chunkIndex += 1
                },
                cancelCheck: { try Task.checkCancellation() },
                onFrame: { count in RunProgress.report(.generate, step: count) })
            if !synthesis.saturated {
                if attempt > 0 {
                    Self.rescueLog.info("stillborn decode rescued at nudge \(attempt)")
                }
                samples = synthesis.samples
                break
            }
            if attempt == rescueAttempts {
                Self.rescueLog.error("stillborn decode persisted through \(rescueAttempts) nudges — emitting silence")
                emit(TTSStreamChunk(samples: [], sampleRate: GepardModel.sampleRate,
                                    index: chunkIndex, isFinal: true))
            } else {
                Self.rescueLog.info("stillborn decode (attempt \(attempt)) — nudging reference")
            }
        }

        try Task.checkCancellation()
        RunProgress.report(.postprocess)
        let wav = AudioSupport.encodeWAV16(samples: samples, sampleRate: GepardModel.sampleRate)
        return TTSResponse(audio: Audio(
            format: .wav, data: wav, sampleRate: GepardModel.sampleRate, channels: 1))
    }

    /// Typed view of a request for the metaData plane (nil for non-TTS requests, which
    /// `conditioning` already rejected).
    private func tts(_ request: any CapabilityRequest) -> TTSRequest? {
        request as? TTSRequest
    }

    // MARK: - Shared conditioning (run + runStream)

    /// Voice guard → memoized speaker prefix → cond_ids → metaData-plane options. Shared by
    /// `run()` and `runStream()` so the two paths cannot drift.
    private func conditioning(for request: any CapabilityRequest) throws
        -> (model: GepardModel, prefix: MLXArray, condIds: [Int], options: GepardDecoder.Options,
            clip: Audio, rescueAttempts: Int) {
        guard let model else { throw PackageError.notLoaded }
        guard request.capability == .tts, let tts = request as? TTSRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }

        // Voice: zero-shot cloning only.
        guard case .referenceAudio(let referenceClip) = tts.voice.selection else {
            throw PackageError.unsupportedRequestFeature(
                "voice — Gepard has no preset voices; provide voice.referenceAudio")
        }

        // Reference conditioning → frozen speaker prefix (memoized per clip bytes + nudge).
        let prefix = try speakerPrefix(for: referenceClip, nudge: 0)

        // Text → cond_ids (P10-validated conditioner).
        let condIds = model.conditioner.condIds(for: tts.text)

        // metaData plane.
        if let seed = tts.metaData.intValue("seed") {
            // Clamp to 32 bits (Qwen3 precedent — large 64-bit seeds break EOS). No-op under
            // greedy decoding, wired for a future temperature path.
            MLXRandom.seed(UInt64(bitPattern: Int64(seed)) & 0xFFFF_FFFF)
        }
        // Default frame budget scales with the text: typical speech runs ~1.4–2.2 frames
        // per character, so ~2.5×chars + 60 is a generous ceiling that still stops a
        // runaway decode (no honored stop-head crossing — e.g. trailing babble at high
        // thresholds) from filling the full 2000-frame cap (~93 s). Explicit "maxFrames"
        // always wins.
        let derivedCap = min(2000, Int(Double(tts.text.count) * 2.5) + 60)
        let maxFrames = tts.metaData.intValue("maxFrames") ?? derivedCap
        var options = GepardDecoder.Options(maxFrames: max(1, maxFrames))
        if let stopThreshold = tts.metaData.doubleValue("stopThreshold") {
            // The stop head is prefix-conditioned: some reference clips cross 0.5 at a
            // mid-utterance sentence pause and truncate multi-sentence text. Clamp to a sane
            // sigmoid band; the oracle exposes the same knob (runner.py stop_threshold).
            options.stopThreshold = Float(min(max(stopThreshold, 0.05), 0.99))
        }
        // Stop-head floor (AB-L-0075): crossings inside the first frames are never a real
        // end of speech — ignore them, and let the energy verdict above the decoder decide
        // whether the decode was stillborn. "minFrames" overrides.
        let minFrames = tts.metaData.intValue("minFrames") ?? max(8, condIds.count)
        options.minFrames = min(max(0, minFrames), options.maxFrames - 1)
        if let cfgScale = tts.metaData.doubleValue("cfgScale") {
            options.cfgScale = Float(cfgScale)
            options.cfgFrames = tts.metaData.intValue("cfgFrames") ?? 20
            options.uncondIds = [GepardConditioner.sot, GepardConditioner.eot, GepardConditioner.sos]
        }
        // Rescue budget for stillborn decodes: re-encode the reference with a tiny gain
        // nudge and retry, deterministically. "stopRescueAttempts" overrides (0 disables).
        let rescueAttempts = tts.metaData.intValue("stopRescueAttempts") ?? 3
        return (model, prefix, condIds, options, referenceClip,
                min(max(0, rescueAttempts), 6))
    }

    /// Speaker prefix for a reference clip at a given rescue nudge. Nudge k scales the
    /// decoded reference by 0.977^k (≈ −0.2 dB per step) before encoding — a tiny,
    /// identity-preserving, DETERMINISTIC perturbation that relands the prefix-conditioned
    /// stop head when a (clip, text) pair decodes stillborn (AB-L-0075: greedy decode makes
    /// stillbirth reproducible per prefix, and a small level change reliably reshuffles it).
    private func speakerPrefix(for clip: Audio, nudge: Int) throws -> MLXArray {
        guard let model else { throw PackageError.notLoaded }
        let key = "\(Self.referenceKey(clip.data))#\(nudge)"
        if let cached = cachedPrefixes[key] { return cached }
        RunProgress.report(.encode)
        let (mono, sourceRate) = try AudioSupport.decodeToMono(clip)
        var samples22k = SincResampler.resample(
            audio: mono, from: sourceRate, to: GepardModel.sampleRate)
        if nudge > 0 {
            let gain = pow(Float(0.977), Float(nudge))
            for i in samples22k.indices { samples22k[i] *= gain }
        }
        let refCodes = model.encodeReference(samples: samples22k)
        let p = model.voicePrefix(refCodes: refCodes)
        eval(p)
        cachedPrefixes[key] = p
        return p
    }

    /// In-memory cache key for a reference clip (Hasher is per-process seeded, which is all we
    /// need — reuse happens within one long-form run).
    nonisolated static func referenceKey(_ data: Data) -> Int {
        var hasher = Hasher()
        hasher.combine(data)
        return hasher.finalize()
    }
}

/// Wrapper-level errors (weight resolution). Runtime request errors use `PackageError`.
public enum GepardError: Error, CustomStringConvertible {
    case missingWeights(String)
    public var description: String {
        switch self {
        case .missingWeights(let why): return "Gepard weights unavailable: \(why)"
        }
    }
}

extension MetaData {
    /// Convenience: read an int-valued metaData key.
    func intValue(_ key: String) -> Int? {
        if case .int(let value)? = self[key] { return value }
        return nil
    }

    /// Convenience: read a double-valued key, accepting ints (JSON 1 vs 1.0).
    func doubleValue(_ key: String) -> Double? {
        switch self[key] {
        case .double(let value)?: return value
        case .int(let value)?: return Double(value)
        default: return nil
        }
    }
}
