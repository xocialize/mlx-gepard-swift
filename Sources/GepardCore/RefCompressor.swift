// RefCompressor — Gepard's Q-Former voice-cloning compressor (ref codes -> K=8 speaker tokens).
//
// Ported 1:1 from gepard-inference `ref_compressor.py`. Pipeline (do_unfold_in_forward=False,
// input already unfolded [1, T_ref, 32]):
//   dequantize_codes -> Linear(32->1024) -> + sinusoidal PE -> ref_feats
//   K=8 learnable queries; L=2 pre-norm blocks: q += SelfAttn(RMSNorm q);
//   q += CrossAttn(RMSNorm q, kv=ref_feats); q += SwiGLU(RMSNorm q)
//   prefix_out = output_scale · final_RMSNorm(q)   (also returns q_normed)
//
// NOTE: these RMSNorms are PLAIN `weight * normed` (weights ~1.0), UNLIKE the backbone's
// Gemma-style (1+weight). Attention: 8 heads × 128, standard SDPA, no mask (ref all-valid).

import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - Multi-head attention (self or cross)

final class QFormerMHA: Module {
    let heads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(dModel: Int, heads: Int) {
        self.heads = heads
        self.headDim = dModel / heads
        self.scale = pow(Float(dModel / heads), -0.5)
        _qProj.wrappedValue = Linear(dModel, dModel, bias: false)
        _kProj.wrappedValue = Linear(dModel, dModel, bias: false)
        _vProj.wrappedValue = Linear(dModel, dModel, bias: false)
        _outProj.wrappedValue = Linear(dModel, dModel, bias: false)
        super.init()
    }

    func callAsFunction(_ qIn: MLXArray, _ kvIn: MLXArray) -> MLXArray {
        let B = qIn.dim(0), Tq = qIn.dim(1), Tkv = kvIn.dim(1)
        let q = qProj(qIn).reshaped(B, Tq, heads, headDim).transposed(0, 2, 1, 3)
        let k = kProj(kvIn).reshaped(B, Tkv, heads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(kvIn).reshaped(B, Tkv, heads, headDim).transposed(0, 2, 1, 3)
        let out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .none)
            .transposed(0, 2, 1, 3).reshaped(B, Tq, -1)
        return outProj(out)
    }
}

// MARK: - SwiGLU FFN

final class QFormerSwiGLU: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    init(_ d: Int, _ hidden: Int) {
        _gate.wrappedValue = Linear(d, hidden, bias: false)
        _up.wrappedValue = Linear(d, hidden, bias: false)
        _down.wrappedValue = Linear(hidden, d, bias: false)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

// MARK: - One Q-Former block (pre-norm)

final class QFormerBlock: Module {
    @ModuleInfo(key: "norm_self") var normSelf: RMSNorm
    @ModuleInfo(key: "self_attn") var selfAttn: QFormerMHA
    @ModuleInfo(key: "norm_cross") var normCross: RMSNorm
    @ModuleInfo(key: "cross_attn") var crossAttn: QFormerMHA
    @ModuleInfo(key: "norm_ffn") var normFfn: RMSNorm
    @ModuleInfo(key: "ffn") var ffn: QFormerSwiGLU

    init(dModel: Int, heads: Int, ffnHidden: Int, eps: Float) {
        _normSelf.wrappedValue = RMSNorm(dimensions: dModel, eps: eps)
        _selfAttn.wrappedValue = QFormerMHA(dModel: dModel, heads: heads)
        _normCross.wrappedValue = RMSNorm(dimensions: dModel, eps: eps)
        _crossAttn.wrappedValue = QFormerMHA(dModel: dModel, heads: heads)
        _normFfn.wrappedValue = RMSNorm(dimensions: dModel, eps: eps)
        _ffn.wrappedValue = QFormerSwiGLU(dModel, ffnHidden)
        super.init()
    }

    func callAsFunction(_ qIn: MLXArray, refFeats: MLXArray) -> MLXArray {
        var q = qIn
        let hs = normSelf(q); q = q + selfAttn(hs, hs)
        let hc = normCross(q); q = q + crossAttn(hc, refFeats)
        q = q + ffn(normFfn(q))
        return q
    }
}

// MARK: - RefCompressor

public final class RefCompressor: Module {
    @ModuleInfo(key: "input_proj") var inputProj: Linear
    @ParameterInfo(key: "queries") var queries: MLXArray         // [K, d]
    @ModuleInfo(key: "blocks") var blocks: [QFormerBlock]
    @ModuleInfo(key: "final_norm") var finalNorm: RMSNorm
    @ParameterInfo(key: "output_scale") var outputScale: MLXArray

    let fsqLevels: [Int]
    let numLayersCodec: Int
    let dModel: Int
    // Stored as [Float], NOT MLXArray — a bare MLXArray property would be collected as a
    // trainable parameter and fail the .noUnusedKeys weight bind.
    private let dequantScaleValues: [Float]                       // [C_total]

    public init(fsqLevels: [Int] = [8, 7, 6, 6], numLayersCodec: Int = 8,
                dModel: Int = 1024, numQueries: Int = 8, numBlocks: Int = 2,
                numHeads: Int = 8, ffnMult: Int = 4, eps: Float = 1e-6) {
        self.fsqLevels = fsqLevels
        self.numLayersCodec = numLayersCodec
        self.dModel = dModel
        let cTotal = numLayersCodec * fsqLevels.count
        // scale[c] = max(fsqLevels[c % L] // 2, 1)  (matches codec_ops.dequantize_codes)
        self.dequantScaleValues = (0 ..< cTotal).map { Float(max(fsqLevels[$0 % fsqLevels.count] / 2, 1)) }

        _inputProj.wrappedValue = Linear(cTotal, dModel, bias: true)
        _queries.wrappedValue = MLXArray.zeros([numQueries, dModel])
        _blocks.wrappedValue = (0 ..< numBlocks).map { _ in
            QFormerBlock(dModel: dModel, heads: numHeads, ffnHidden: dModel * ffnMult, eps: eps)
        }
        _finalNorm.wrappedValue = RMSNorm(dimensions: dModel, eps: eps)
        _outputScale.wrappedValue = MLXArray(1.0 / Float(dModel).squareRoot())
        super.init()
    }

    /// Per-channel symmetric dequant: (x - scale)/scale, scale = max(L//2, 1). -> float [B,T,C].
    func dequantize(_ codes: MLXArray) -> MLXArray {
        let scale = MLXArray(dequantScaleValues)                  // [C_total]
        return (codes.asType(.float32) - scale) / scale
    }

    /// Standard sinusoidal PE [1, T, dim]; even dims = sin, odd = cos (interleaved).
    func sinusoidalPE(_ T: Int, _ dim: Int) -> MLXArray {
        let pos = MLXArray(0 ..< T).asType(.float32).reshaped(T, 1)                 // [T,1]
        let idx = MLXArray(stride(from: 0, to: dim, by: 2)).asType(.float32)        // [dim/2]
        let div = MLX.exp(idx * Float(-Foundation.log(10000.0) / Double(dim)))      // [dim/2]
        let ang = pos * div                                                         // [T,dim/2]
        let pe = stacked([MLX.sin(ang), MLX.cos(ang)], axis: -1).reshaped(T, dim)   // interleave
        return pe.reshaped(1, T, dim)
    }

    public struct Ladder {
        public let inputProjOut: MLXArray
        public let posEncOut: MLXArray
        public let blockOuts: [MLXArray]
        public let qNormed: MLXArray
        public let prefix: MLXArray
    }

    /// Full ladder from unfolded ref codes [1, T_ref, 32] (ref_mask assumed all-valid).
    public func ladder(refCodes: MLXArray) -> Ladder {
        let deq = dequantize(refCodes)                    // [1,T,32]
        let proj = inputProj(deq)                         // [1,T,1024]
        let feats = proj + sinusoidalPE(proj.dim(1), dModel)
        var q = expandedDimensions(queries, axis: 0)      // [1,K,d]
        var outs: [MLXArray] = []
        for b in blocks { q = b(q, refFeats: feats); outs.append(q) }
        let qn = finalNorm(q)
        return Ladder(inputProjOut: proj, posEncOut: feats, blockOuts: outs,
                      qNormed: qn, prefix: outputScale * qn)
    }

    /// (prefix_out, q_normed) — the decoder consumes prefix_out.
    public func callAsFunction(refCodes: MLXArray) -> (prefix: MLXArray, qNormed: MLXArray) {
        let l = ladder(refCodes: refCodes)
        return (l.prefix, l.qNormed)
    }
}
