// Backbone — Gepard's Qwen3.5 text transformer, lifted from mlx-swift-lm's Qwen35.swift.
//
// Gepard removed the GatedDeltaNet (linear) layers, so EVERY layer is full softmax
// attention with the Qwen3.5 output gate. This is the Qwen3 block + attn_output_gate:
// q_proj emits heads*headDim*2, split into queries + per-head gate, and the attention
// output is `oProj(sigmoid(gate) * out)`. q/k RMSNorm, interleaved mRoPE (mrope_section
// [11,11,10]), GQA (2 kv heads), head_dim 256. Config values from gepard_config.json.
//
// Module keys mirror the checkpoint's `model.*` subtree so weights load 1:1:
//   embed_tokens · layers.{i}.self_attn.{q,k,v,o}_proj · .self_attn.{q,k}_norm
//   .input_layernorm · .post_attention_layernorm · .mlp.{gate,up,down}_proj · norm

import Foundation
import MLX
import MLXNN
import MLXFast
import MLXLMCommon

public struct GepardBackboneConfig: Sendable {
    public var hiddenSize = 1024
    public var numLayers = 14
    public var intermediateSize = 3584
    public var attentionHeads = 8
    public var kvHeads = 2
    public var headDim = 256
    public var rmsNormEps: Float = 1e-6
    public var ropeTheta: Float = 10_000_000
    public var partialRotaryFactor: Float = 1.0
    public var mropeSection = [11, 11, 10]
    public var maxPositionEmbeddings = 262_144
    public var vocabSize = 248_320
    public init() {}
}

// MARK: - RMSNorm ((1 + weight) / Gemma convention)

/// Gepard's checkpoint stores RMSNorm gains centered on 0 (range ~±0.14), so the effective
/// scale is `(1 + weight)`. Plain `weight * normed` collapses activations ~20x — this is the
/// P3 parity fix. Applies to every norm: input/post_attention layernorm, q/k_norm, final norm.
final class GepardRMSNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float
    init(_ dim: Int, eps: Float) {
        self.eps = eps
        _weight.wrappedValue = MLXArray.zeros([dim])  // stored ~0 ⇒ effective (1+0)=1 unloaded
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: 1.0 + weight, eps: eps)
    }
}

// MARK: - Interleaved mRoPE (replicated 1:1 from mlx-swift-lm Qwen35Language)

/// Qwen3.5's interleaved multimodal RoPE. For a pure 1-D text/audio stream (all three
/// position dims equal), this reduces to standard NeoX rope — but we replicate the exact
/// interleave so parity holds if positions ever become multi-dim.
final class GepardRotaryEmbedding {
    private let invFreq: MLXArray
    private let mropeSection: [Int]

    init(dim: Int, base: Float, mropeSection: [Int]) {
        let safeDim = max(1, dim)
        var freq = MLXArray(stride(from: 0, to: safeDim, by: 2)).asType(.float32)
        freq = freq / Float(safeDim)
        self.invFreq = 1.0 / pow(MLXArray(base), freq)
        self.mropeSection = mropeSection.count >= 3 ? mropeSection : [11, 11, 10]
    }

    private func applyInterleavedMRope(_ freqs: MLXArray) -> MLXArray {
        let freqsT = freqs[0, 0..., 0..., 0...]
        let dims = freqsT.dim(-1)
        var slices: [MLXArray] = []
        slices.reserveCapacity(dims)
        for idx in 0 ..< dims {
            var slice = freqsT[0..., 0..., idx]
            for (dim, offset) in [(1, 1), (2, 2)] {
                let length = min(mropeSection[dim] * 3, dims)
                if idx >= offset && idx < length && ((idx - offset) % 3 == 0) {
                    slice = freqs[dim, 0..., 0..., idx]
                    break
                }
            }
            slices.append(slice)
        }
        return stacked(slices, axis: -1)
    }

    /// -> (cos, sin) each [B, L, dim]. `positionIds` is [3, B, L] (or [B, L], broadcast).
    func callAsFunction(dtype: DType, positionIds: MLXArray) -> (MLXArray, MLXArray) {
        var positionIds = positionIds
        if positionIds.ndim == 2 {
            positionIds = broadcast(
                positionIds[.newAxis, 0..., 0...],
                to: [3, positionIds.dim(0), positionIds.dim(1)])
        }
        let pos = positionIds.asType(.float32)
        var inv = invFreq.asType(.float32)
        inv = inv[.newAxis, .newAxis, .newAxis, 0...]
        var freqs = pos[0..., 0..., 0..., .newAxis] * inv
        freqs = applyInterleavedMRope(freqs)
        let emb = concatenated([freqs, freqs], axis: -1)
        return (cos(emb).asType(dtype), sin(emb).asType(dtype))
    }
}

private func rotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    return concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
}

/// Apply interleaved-mRoPE cos/sin to q,k ([B, H, L, D]); cos/sin are [B, L, D].
private func applyMRoPE(
    q: MLXArray, k: MLXArray, cos: MLXArray, sin: MLXArray
) -> (MLXArray, MLXArray) {
    let cos = expandedDimensions(cos, axis: 1)  // [B,1,L,D]
    let sin = expandedDimensions(sin, axis: 1)
    let qOut = (q * cos) + (rotateHalf(q) * sin)
    let kOut = (k * cos) + (rotateHalf(k) * sin)
    return (qOut, kOut)
}

// MARK: - Attention (full, with Qwen3.5 output gate)

final class GepardAttention: Module {
    let attentionHeads: Int
    let kvHeads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: GepardRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: GepardRMSNorm

    let rotary: GepardRotaryEmbedding

    init(_ c: GepardBackboneConfig) {
        let headDim = c.headDim
        self.attentionHeads = c.attentionHeads
        self.kvHeads = c.kvHeads
        self.scale = pow(Float(headDim), -0.5)

        _qProj.wrappedValue = Linear(c.hiddenSize, c.attentionHeads * headDim * 2, bias: false)
        _kProj.wrappedValue = Linear(c.hiddenSize, c.kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(c.hiddenSize, c.kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(c.attentionHeads * headDim, c.hiddenSize, bias: false)
        _qNorm.wrappedValue = GepardRMSNorm(headDim, eps: c.rmsNormEps)
        _kNorm.wrappedValue = GepardRMSNorm(headDim, eps: c.rmsNormEps)

        let ropeDim = Int(Float(headDim) * c.partialRotaryFactor)
        self.rotary = GepardRotaryEmbedding(
            dim: ropeDim, base: c.ropeTheta, mropeSection: c.mropeSection)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        let qParts = qProj(x).reshaped(B, L, attentionHeads, -1).split(parts: 2, axis: -1)
        var queries = qParts[0]
        let gate = qParts[1].reshaped(B, L, -1)

        var keys = kProj(x)
        var values = vProj(x)

        queries = qNorm(queries).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        // positionIds = arange(offset, offset+L) tiled to [3, B, L]
        let offset = cache?.offset ?? 0
        var base = MLXArray(stride(from: offset, to: offset + L, by: 1)).asType(.int32)
        base = tiled(base[.newAxis, 0...], repetitions: [B, 1])          // [B,L]
        let positionIds = tiled(base[.newAxis, 0..., 0...], repetitions: [3, 1, 1])  // [3,B,L]

        let (cosV, sinV) = rotary(dtype: queries.dtype, positionIds: positionIds)
        (queries, keys) = applyMRoPE(q: queries, k: keys, cos: cosV, sin: sinV)

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(output * sigmoid(gate))
    }
}

// MARK: - SwiGLU MLP (lifted from Qwen3NextMLP)

final class GepardMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ dim: Int, _ hidden: Int) {
        _gate.wrappedValue = Linear(dim, hidden, bias: false)
        _up.wrappedValue = Linear(dim, hidden, bias: false)
        _down.wrappedValue = Linear(hidden, dim, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

// MARK: - Decoder layer

final class GepardDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: GepardAttention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: GepardRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: GepardRMSNorm
    @ModuleInfo(key: "mlp") var mlp: GepardMLP

    init(_ c: GepardBackboneConfig) {
        _selfAttn.wrappedValue = GepardAttention(c)
        _inputLayerNorm.wrappedValue = GepardRMSNorm(c.hiddenSize, eps: c.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = GepardRMSNorm(c.hiddenSize, eps: c.rmsNormEps)
        _mlp.wrappedValue = GepardMLP(c.hiddenSize, c.intermediateSize)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

// MARK: - Backbone (= Qwen3_5TextModel inner; keys under `model.*`)

public final class GepardBackbone: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    fileprivate let layers: [GepardDecoderLayer]
    @ModuleInfo(key: "norm") var norm: GepardRMSNorm

    public init(_ c: GepardBackboneConfig) {
        _embedTokens.wrappedValue = Embedding(embeddingCount: c.vocabSize, dimensions: c.hiddenSize)
        self.layers = (0 ..< c.numLayers).map { _ in GepardDecoderLayer(c) }
        _norm.wrappedValue = GepardRMSNorm(c.hiddenSize, eps: c.rmsNormEps)
        super.init()
    }

    /// Embed text token ids -> [B, T, d] (Gepard: `model.embed_tokens`).
    public func embed(_ ids: MLXArray) -> MLXArray { embedTokens(ids) }

    /// Step-by-step layer-0 probe (parity debugging): prints the std of each sub-step.
    public func debugLayer0(inputsEmbeds x: MLXArray) {
        func s(_ n: String, _ a: MLXArray) {
            let f = a.asType(.float32)
            let std = f.variance().item(Float.self).squareRoot()
            let mx = abs(f).max().item(Float.self)
            print("    \(n.padding(toLength: 16, withPad: " ", startingAt: 0)) std \(String(format: "%.4f", std)) absmax \(String(format: "%.3f", mx))")
        }
        let l = layers[0]
        let normed = l.inputLayerNorm(x)
        s("inLN(x)", normed)
        let mask = createAttentionMask(h: x, cache: nil)
        let attn = l.selfAttn(normed, mask: mask, cache: nil)
        s("attn(normed)", attn)
        let h = x + attn
        s("x+attn", h)
        let pn = l.postAttentionLayerNorm(h)
        s("postLN(h)", pn)
        let m = l.mlp(pn)
        s("mlp(postLN)", m)
        s("h+mlp", h + m)
    }

    /// One fresh KV cache per layer (all full-attention).
    public func newCache() -> [KVCache] { layers.map { _ in KVCacheSimple() } }

    /// Prefill/step from `inputsEmbeds` [B, T, d]. Returns the post-norm last hidden
    /// state and, if `capture`, the per-layer outputs (for P3 parity).
    public func callAsFunction(
        inputsEmbeds: MLXArray, cache: [KVCache]?, capture: Bool = false
    ) -> (MLXArray, [MLXArray]) {
        var h = inputsEmbeds
        let mask = createAttentionMask(h: h, cache: cache?.first)
        var caps = [MLXArray]()
        if capture { caps.reserveCapacity(layers.count) }
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
            if capture { caps.append(h) }
        }
        return (norm(h), caps)
    }
}
