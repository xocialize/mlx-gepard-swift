// AudioInterface + CodebookHeads — Gepard's audio embedding front-end and prediction heads.
//
// Interface (previous frame's 32 codes -> next-frame input embedding):
//   32× Embedding(L_k, 32) -> concat[·,1024] -> Linear -> GELU -> Linear
//   -> LayerNorm(elementwise_affine=false) -> × audio_embed_scale  (frozen [1] param)
// Heads (last_hidden -> per-codebook logits): 32× Linear(1024 -> L_k) + stop Linear(1024 -> 1).
// L_k is cyclic {8,7,6,6}×8. null_prefix / supcon_head exist only for state-dict compat.

import Foundation
import MLX
import MLXNN

/// Parameter-free LayerNorm over the last dim (torch default eps 1e-5, population variance).
func layerNormNoAffine(_ x: MLXArray, eps: Float = 1e-5) -> MLXArray {
    let mean = x.mean(axis: -1, keepDims: true)
    let v = variance(x, axis: -1, keepDims: true, ddof: 0)
    return (x - mean) * rsqrt(v + eps)
}

public final class AudioInterface: Module {
    @ModuleInfo(key: "audio_embeddings") var audioEmbeddings: [Embedding]
    // audio_embed_proj = nn.Sequential(Linear, GELU, Linear); the numeric checkpoint keys
    // ".0"/".2" unflatten as an array (gap at GELU), so we bind them as two named linears and
    // remap the keys in the loader (audio_embed_proj.0 -> audioProjL0, .2 -> audioProjL2).
    @ModuleInfo(key: "audioProjL0") var l0: Linear
    @ModuleInfo(key: "audioProjL2") var l2: Linear
    @ParameterInfo(key: "audio_embed_scale") var scale: MLXArray

    public let numCodebooks: Int

    /// The loader must remap `audio_embed_proj.{0,2}.*` to `audioProjL{0,2}.*`.
    public static func remapProjKey(_ k: String) -> String {
        k.replacingOccurrences(of: "audio_embed_proj.0.", with: "audioProjL0.")
         .replacingOccurrences(of: "audio_embed_proj.2.", with: "audioProjL2.")
    }

    /// `levels` = per-codebook vocab sizes (the unfolded FSQ radices, length 32).
    public init(levels: [Int], dim: Int = 1024) {
        self.numCodebooks = levels.count
        _audioEmbeddings.wrappedValue = levels.map { Embedding(embeddingCount: $0, dimensions: dim / levels.count) }
        _l0.wrappedValue = Linear(dim, dim)
        _l2.wrappedValue = Linear(dim, dim)
        _scale.wrappedValue = MLXArray(1.0)
        super.init()
    }

    /// 32 integer codes -> frame input embedding [1, 1, dim].
    public func frameEmbed(codes: [Int]) -> MLXArray {
        var parts: [MLXArray] = []
        parts.reserveCapacity(numCodebooks)
        for i in 0 ..< numCodebooks {
            parts.append(audioEmbeddings[i](MLXArray(Int32(codes[i]))))  // [dim/32]
        }
        var x = concatenated(parts, axis: -1).reshaped(1, 1, -1)          // [1,1,dim]
        x = l2(gelu(l0(x)))                                               // exact (erf) GELU
        x = layerNormNoAffine(x)
        return scale * x
    }
}

public final class CodebookHeads: Module {
    @ModuleInfo(key: "codebook_heads") var heads: [Linear]
    @ModuleInfo(key: "stop_head") var stopHead: Linear

    public init(levels: [Int], dim: Int = 1024) {
        _heads.wrappedValue = levels.map { Linear(dim, $0) }
        _stopHead.wrappedValue = Linear(dim, 1)
        super.init()
    }

    /// hidden [.., dim] -> (32 per-codebook logits, stop logit).
    public func callAsFunction(_ hidden: MLXArray) -> (logits: [MLXArray], stop: MLXArray) {
        (heads.map { $0(hidden) }, stopHead(hidden))
    }
}
