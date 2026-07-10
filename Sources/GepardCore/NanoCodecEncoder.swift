// NanoCodecEncoder.swift — the HiFiGAN encoder + GroupFSQ of nvidia's nemo-nano-codec-22khz
// (Gepard 21.5fps / 1.89kbps), ported 1:1 from the adapted+validated MLX-Python reference
// (nanocodec-mlx: models/encoder.py, quantizer_fixed.py, layers/{conv,activations}).
//
// Path: waveform [1,T_samples] -> HiFiGAN encoder -> 32-ch continuous features [1,32,T]
//       -> GroupFSQ (8 groups × [8,7,6,6]) -> packed group tokens [1, 8, T].
// The Q-Former (RefCompressor) consumes the UNFOLDED [1,T,32] view, obtained from the packed
// tokens via the P2-validated `CodecOps.unfoldTokens`. This is Gepard P9 — the reference-audio
// encode path the engine `run()` needs for zero-shot cloning from an arbitrary clip.
//
// Same weight-norm fusion + functional (non-Module) style as NanoCodec.swift (the decoder);
// the encoder differs by REPLICATE padding (edge-repeat), strided down-convs, and LeakyReLU
// (vs the decoder's zeros padding / half-snake). Weights live under `audio_encoder.*`.

import Foundation
import MLX
import MLXNN

public final class NanoCodecEncoder {

    // --- Gepard 21.5fps encoder config (phase1_encode_parity.py ENC) ---
    public static let downSampleRates = [2, 2, 4, 8, 8]
    public static let baseChannels = 24
    public static let encodedDim = 32
    // rate -> kernel: 2->4, 4->8, 8->16 (donor: rate==2?4 : rate*2)
    static let downKernels = [4, 4, 8, 16, 16]
    // 21.5fps down-pad fix: (k - s + 1)//2  (donor hardcoded 12.5fps [1,2,3,4,4])
    static let downPads = [1, 1, 2, 4, 4]
    let resKernelSizes = [3, 7, 11]
    let resDilations = [1, 3, 5]

    // --- GroupFSQ config ---
    static let fsqLevels = [8, 7, 6, 6]
    static let numGroups = 8
    static let fsqEps: Float = 1e-3

    private var convW: [String: MLXArray] = [:]   // "<name>": weight [out,kernel,in]
    private var convB: [String: MLXArray] = [:]   // "<name>": bias   [out]

    /// Build the encoder from the raw codec safetensors dict (PyTorch weight-norm keys).
    public init(weights raw: [String: MLXArray]) {
        func fuseConv(_ prefix: String) -> (MLXArray, MLXArray) {
            guard let g = raw["\(prefix).parametrizations.weight.original0"],
                  let v = raw["\(prefix).parametrizations.weight.original1"],
                  let b = raw["\(prefix).bias"]
            else { fatalError("NanoCodecEncoder: missing conv weights for \(prefix)") }
            let gf = g.asType(.float32)                                   // [out,1,1]
            let vf = v.asType(.float32)                                   // [out,in,kernel]
            let norm = sqrt((vf * vf).sum(axis: 2, keepDims: true).sum(axis: 1, keepDims: true))
            let w = gf * vf / maximum(norm, MLXArray(Float(1e-8)))         // [out,in,kernel]
            return (w.transposed(0, 2, 1), b.asType(.float32))            // -> MLX [out,kernel,in]
        }
        func loadConv(_ key: String, _ prefix: String) {
            let (w, b) = fuseConv(prefix); convW[key] = w; convB[key] = b
        }

        loadConv("pre", "audio_encoder.pre_conv.conv")
        loadConv("post", "audio_encoder.post_conv.conv")
        for i in 0 ..< Self.downSampleRates.count {
            loadConv("down.\(i)", "audio_encoder.down_sample_conv_layers.\(i).conv")
            for j in 0 ..< resKernelSizes.count {
                for k in 0 ..< resDilations.count {
                    let p = "audio_encoder.res_layers.\(i).res_blocks.\(j).res_blocks.\(k)"
                    loadConv("res.\(i).\(j).\(k).in", "\(p).input_conv.conv")
                    loadConv("res.\(i).\(j).\(k).skip", "\(p).skip_conv.conv")
                }
            }
        }
    }

    // MARK: - primitive ops (Conv1dNorm replicate, lrelu) — 1:1 with the reference

    /// Conv1dNorm with replicate (edge-repeat) padding padL/padR, valid conv, stride + dilation.
    private func convNorm(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray,
                          padL: Int, padR: Int, stride: Int, dilation: Int) -> MLXArray {
        var xp = x
        if padL > 0 {
            xp = concatenated([repeated(xp[0..., 0..., 0 ..< 1], count: padL, axis: 2), xp], axis: 2)
        }
        if padR > 0 {
            let t = xp.dim(2)
            xp = concatenated([xp, repeated(xp[0..., 0..., (t - 1) ..< t], count: padR, axis: 2)], axis: 2)
        }
        let nlc = xp.transposed(0, 2, 1)                                  // [1, T', Cin]
        var y = conv1d(nlc, w, stride: stride, padding: 0, dilation: dilation, groups: 1)
        y = y.transposed(0, 2, 1)                                         // [1, Cout, T'']
        return y + b.reshaped([1, b.dim(0), 1])
    }

    /// Res-conv default (padding=None → symmetric (k-1)*dilation split, replicate), stride 1.
    private func symConv(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray, dilation: Int) -> MLXArray {
        let k = w.dim(1)
        let total = (k - 1) * dilation
        return convNorm(x, w, b, padL: total / 2, padR: total - total / 2, stride: 1, dilation: dilation)
    }

    private func lrelu(_ x: MLXArray) -> MLXArray { leakyRelu(x, negativeSlope: 0.01) }

    // MARK: - HiFiGAN residual stack (lrelu / replicate)

    /// ResidualBlock: x + skip_conv(lrelu(in_conv(lrelu(x)))); crop/pad back to original T.
    private func residualBlock(_ x: MLXArray, _ i: Int, _ j: Int, _ k: Int, dilation: Int) -> MLXArray {
        let orig = x.dim(2)
        var out = lrelu(x)
        out = symConv(out, convW["res.\(i).\(j).\(k).in"]!, convB["res.\(i).\(j).\(k).in"]!, dilation: dilation)
        out = lrelu(out)
        out = symConv(out, convW["res.\(i).\(j).\(k).skip"]!, convB["res.\(i).\(j).\(k).skip"]!, dilation: 1)
        if out.dim(2) != orig {
            if out.dim(2) > orig { out = out[0..., 0..., 0 ..< orig] }
            else { out = padded(out, widths: [[0, 0], [0, 0], [0, orig - out.dim(2)]]) }
        }
        return x + out
    }

    private func resBlock(_ x: MLXArray, _ i: Int, _ j: Int) -> MLXArray {
        var h = x
        for k in 0 ..< resDilations.count { h = residualBlock(h, i, j, k, dilation: resDilations[k]) }
        return h
    }

    /// HiFiGANResLayer: run each kernel branch on x, then average.
    private func resLayer(_ x: MLXArray, _ i: Int) -> MLXArray {
        var residuals: [MLXArray] = []
        for j in 0 ..< resKernelSizes.count { residuals.append(resBlock(x, i, j)) }
        var acc = residuals[0]
        for r in residuals.dropFirst() { acc = acc + r }
        return acc / MLXArray(Float(residuals.count))
    }

    // MARK: - encoder net

    /// wave [1,T] or [1,1,T] -> 32-ch continuous features [1, 32, T/1024].
    private func encodeFeatures(_ wave: MLXArray) -> MLXArray {
        var x = wave
        if x.ndim == 2 { x = x.expandedDimensions(axis: 1) }              // [1,1,T]
        x = convNorm(x, convW["pre"]!, convB["pre"]!, padL: 3, padR: 3, stride: 1, dilation: 1)  // k7
        for i in 0 ..< Self.downSampleRates.count {
            x = resLayer(x, i)
            x = lrelu(x)
            x = convNorm(x, convW["down.\(i)"]!, convB["down.\(i)"]!,
                         padL: Self.downPads[i], padR: Self.downPads[i],
                         stride: Self.downSampleRates[i], dilation: 1)
            eval(x)                                                       // per-down eval boundary
        }
        x = lrelu(x)
        x = convNorm(x, convW["post"]!, convB["post"]!, padL: 3, padR: 3, stride: 1, dilation: 1)  // k7
        return x                                                          // [1,32,T]
    }

    // MARK: - GroupFSQ.encode

    /// features [1,32,T] -> packed group tokens [1, 8, T] (Int32). Replicates the reference's
    /// exact FSQ arithmetic (tanh-compress → round → nonnegative per-dim → base-mix → int32 trunc).
    static func groupFSQEncode(_ feats: MLXArray) -> MLXArray {
        let cpg = feats.dim(1) / numGroups                               // 32/8 = 4
        let numLevels = MLXArray(fsqLevels.map { Float($0) }).reshaped([1, cpg, 1])
        let outScale = (numLevels - 1) / 2 * (1 - fsqEps)
        let outOffset = MLXArray(fsqLevels.map { $0 % 2 == 0 ? Float(0.5) : Float(0.0) }).reshaped([1, cpg, 1])
        let inputShift = MLX.tan(outOffset / outScale)
        let halfScale = MLXArray(fsqLevels.map { Float($0 / 2) }).reshaped([1, cpg, 1])  // floor(L/2)
        var base = [Float(1)]
        for i in 0 ..< (fsqLevels.count - 1) { base.append(base.last! * Float(fsqLevels[i])) }
        let dimBase = MLXArray(base).reshaped([1, cpg, 1])               // [1, 8, 56, 336]

        var groups: [MLXArray] = []
        for gi in 0 ..< numGroups {
            let gx = feats[0..., (gi * cpg) ..< ((gi + 1) * cpg), 0...]  // [1,cpg,T]
            let compressed = outScale * MLX.tanh(gx + inputShift) - outOffset
            let codesInt = round(compressed)
            let codes = codesInt / halfScale                             // [-1,1]  (exact reference path)
            let idxPerDim = halfScale * codes + halfScale                // nonnegative per-dim [0,L-1]
            let flat = (idxPerDim * dimBase).sum(axis: 1)                // [1,T]
            groups.append(flat)
        }
        return stacked(groups, axis: 1).asType(.int32)                   // [1, 8, T]
    }

    /// Encode a mono waveform [1,T] (or [1,1,T]) -> packed group tokens [1, 8, T_frames] (Int32).
    /// For the Q-Former, unfold with `CodecOps.unfoldTokens` to get [32, T].
    public func encode(_ wave: MLXArray) -> MLXArray {
        Self.groupFSQEncode(encodeFeatures(wave))
    }
}
