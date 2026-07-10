// NanoCodec.swift — the Causal HiFiGAN decoder of nvidia's nemo-nano-codec-22khz
// (Gepard 21.5fps / 1.89kbps config), ported 1:1 from the adapted+validated MLX-Python
// reference (nanocodec-mlx: models/decoder.py, layers/conv.py, layers/activations.py).
//
// Path: 32 dequantized FSQ code channels -> decoder net -> waveform.
// This is Gepard P7: the audio-output critical path (the AR model already produces the
// 32 unfolded code channels; this turns them into sound).
//
// Weight handling mirrors utils/weight_loader_v2.py exactly:
//   - convs use PyTorch weight-norm, stored UNFUSED as
//       <prefix>.parametrizations.weight.original0  (g, shape [out,1,1])
//       <prefix>.parametrizations.weight.original1  (v, shape [out,in,kernel])
//     fuse: w = g * v / max(||v||_{axes 1,2}, 1e-8), then transpose (0,2,1) -> MLX [out,kernel,in]
//   - half-snake alpha stored at <prefix>.activation.snake_act.alpha  (shape [1,split,1])
//
// Not a Module: the checkpoint keys are PyTorch weight-norm keys (not MLX layer keys) and the
// upsampler is a custom grouped transpose-conv, so — like the Python donor — we process the
// weights into plain MLXArrays and run the forward functionally. Keeps Module reflection out of
// the way and maps 1:1 to the reference forward.

import Foundation
import MLX
import MLXNN

public final class NanoCodecDecoder {

    // --- Gepard 21.5fps decoder config (from phase1_decode_parity.py DEC) ---
    public static let upSampleRates = [8, 8, 4, 2, 2]
    public static let inputDim = 32
    public static let baseChannels = 864
    // channel ladder: pre_conv 32->864, then each up-layer halves: 864,432,216,108,54,27
    let chans: [Int]                       // length upSampleRates.count + 1

    // processed weights (fused + MLX layout)
    private var convW: [String: MLXArray] = [:]   // "<name>": weight [out,kernel,in]
    private var convB: [String: MLXArray] = [:]   // "<name>": bias   [out]
    private var alpha: [String: MLXArray] = [:]   // "<name>": half-snake alpha [1,split,1]

    let resKernelSizes = [3, 7, 11]
    let resDilations = [1, 3, 5]

    /// Build the decoder from the raw codec safetensors dict (PyTorch keys, fp32).
    public init(weights raw: [String: MLXArray]) {
        var c = [NanoCodecDecoder.baseChannels]
        for _ in NanoCodecDecoder.upSampleRates { c.append(c.last! / 2) }
        self.chans = c

        func fuseConv(_ prefix: String) -> (MLXArray, MLXArray) {
            guard let g = raw["\(prefix).parametrizations.weight.original0"],
                  let v = raw["\(prefix).parametrizations.weight.original1"],
                  let b = raw["\(prefix).bias"]
            else { fatalError("NanoCodec: missing conv weights for \(prefix)") }
            let gf = g.asType(.float32)                       // [out,1,1]
            let vf = v.asType(.float32)                       // [out,in,kernel]
            // ||v|| over axes (1,2), keepdims  (matches np.linalg.norm(v, axis=(1,2)))
            let norm = sqrt((vf * vf).sum(axis: 2, keepDims: true).sum(axis: 1, keepDims: true))
            let w = gf * vf / maximum(norm, MLXArray(Float(1e-8)))   // [out,in,kernel]
            let wmlx = w.transposed(0, 2, 1)                  // -> MLX [out,kernel,in]
            return (wmlx, b.asType(.float32))
        }
        func loadConv(_ key: String, _ prefix: String) {
            let (w, b) = fuseConv(prefix)
            convW[key] = w; convB[key] = b
        }
        func loadAlpha(_ key: String, _ prefix: String) {
            guard let a = raw["\(prefix).activation.snake_act.alpha"] else {
                fatalError("NanoCodec: missing alpha for \(prefix)")
            }
            alpha[key] = a.asType(.float32)
        }

        // pre / post conv (CausalConv1d), post activation
        loadConv("pre", "audio_decoder.pre_conv.conv")
        loadConv("post", "audio_decoder.post_conv.conv")
        loadAlpha("post_act", "audio_decoder.post_activation")

        for i in 0 ..< NanoCodecDecoder.upSampleRates.count {
            loadAlpha("act.\(i)", "audio_decoder.activations.\(i)")
            loadConv("up.\(i)", "audio_decoder.up_sample_conv_layers.\(i).conv")
            // res_layers[i].res_blocks[j (kernel)].res_blocks[k (dilation)]
            for j in 0 ..< resKernelSizes.count {
                for k in 0 ..< resDilations.count {
                    let p = "audio_decoder.res_layers.\(i).res_blocks.\(j).res_blocks.\(k)"
                    loadAlpha("res.\(i).\(j).\(k).in_act", "\(p).input_activation")
                    loadAlpha("res.\(i).\(j).\(k).skip_act", "\(p).skip_activation")
                    loadConv("res.\(i).\(j).\(k).in", "\(p).input_conv.conv")
                    loadConv("res.\(i).\(j).\(k).skip", "\(p).skip_conv.conv")
                }
            }
        }
    }

    // MARK: - primitive ops (1:1 with the reference)

    /// Causal Conv1d / Conv1dNorm(zeros): left-only pad by (kernel-1)*dilation, valid conv, stride 1.
    private func causalConv(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray, dilation: Int = 1) -> MLXArray {
        let kernel = w.dim(1)
        let padL = (kernel - 1) * dilation
        var xp = x
        if padL > 0 { xp = padded(x, widths: [[0, 0], [0, 0], [padL, 0]]) }
        let nlc = xp.transposed(0, 2, 1)                        // [1, T+padL, Cin]
        var y = conv1d(nlc, w, stride: 1, padding: 0, dilation: dilation, groups: 1)
        y = y.transposed(0, 2, 1)                               // [1, Cout, T]
        return y + b.reshaped([1, b.dim(0), 1])
    }

    /// Half-Snake: first half snake `x + (1/a) sin(a x)^2`, second half LeakyReLU(0.01), concat.
    private func halfSnake(_ x: MLXArray, _ a: MLXArray) -> MLXArray {
        let split = x.dim(1) / 2
        let x1 = x[0..., 0 ..< split, 0...]
        let x2 = x[0..., split..., 0...]
        let s = MLX.sin(a * x1)
        let x1o = x1 + (s * s) / a                              // (1/a) sin^2  ==  sin^2 / a
        let x2o = leakyRelu(x2, negativeSlope: 0.01)
        return concatenated([x1o, x2o], axis: 1)
    }

    /// Grouped causal transposed conv (NanoCodec pattern: groups = out_channels).
    /// Implemented as zero-upsample + kernel-flip + grouped valid conv, exactly as the donor.
    private func transposeConv(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray,
                               stride: Int, outCh: Int) -> MLXArray {
        let inCh = x.dim(1)
        let kernel = w.dim(1)                                   // w is [in_ch, kernel, 1]
        let groups = outCh
        let cpg = inCh / outCh                                  // channels per group

        // upsample by inserting (stride-1) zeros between samples, trailing zeros trimmed
        var xu = x
        if stride > 1 {
            let t = x.dim(2)
            let e = x.expandedDimensions(axis: -1)              // [1,in,t,1]
            let ep = padded(e, widths: [[0, 0], [0, 0], [0, 0], [0, stride - 1]])  // [1,in,t,stride]
            let flat = ep.reshaped([1, inCh, t * stride])
            let upT = t + (t - 1) * (stride - 1)
            xu = flat[0..., 0..., 0 ..< upT]
        }
        // pad kernel-1 both sides
        let pad = kernel - 1
        if pad > 0 { xu = padded(xu, widths: [[0, 0], [0, 0], [pad, pad]]) }
        let nlc = xu.transposed(0, 2, 1)                        // [1, upT+2pad, in]

        // weight -> grouped, flip kernel, -> MLX grouped-conv layout [groups, kernel, cpg]
        var wg = w.reshaped([groups, cpg, kernel, 1])
        let ridx = MLXArray((0 ..< kernel).reversed().map { Int32($0) })
        wg = wg.take(ridx, axis: 2)                             // flip kernel axis
        wg = wg.reshaped([groups, cpg, kernel]).transposed(0, 2, 1)  // [groups, kernel, cpg]

        var y = conv1d(nlc, wg, stride: 1, padding: 0, dilation: 1, groups: groups)  // [1, outT, outCh]
        y = y.transposed(0, 2, 1)                              // [1, outCh, outT]
        y = y + b.reshaped([1, outCh, 1])
        let trim = kernel - stride
        if trim > 0 && y.dim(2) > trim { y = y[0..., 0..., 0 ..< (y.dim(2) - trim)] }
        return y
    }

    /// One ResidualBlock (decoder = zeros pad / half-snake): x + skip_conv(skip_act(in_conv(in_act(x)))).
    private func residualBlock(_ x: MLXArray, _ i: Int, _ j: Int, _ k: Int, dilation: Int) -> MLXArray {
        let orig = x.dim(2)
        var out = halfSnake(x, alpha["res.\(i).\(j).\(k).in_act"]!)
        out = causalConv(out, convW["res.\(i).\(j).\(k).in"]!, convB["res.\(i).\(j).\(k).in"]!, dilation: dilation)
        out = halfSnake(out, alpha["res.\(i).\(j).\(k).skip_act"]!)
        out = causalConv(out, convW["res.\(i).\(j).\(k).skip"]!, convB["res.\(i).\(j).\(k).skip"]!, dilation: 1)
        if out.dim(2) != orig {
            if out.dim(2) > orig {
                out = out[0..., 0..., 0 ..< orig]
            } else {
                out = padded(out, widths: [[0, 0], [0, 0], [0, orig - out.dim(2)]])
            }
        }
        return x + out
    }

    /// HiFiGANResBlock: sequential residual blocks over the dilation list (fixed kernel).
    private func resBlock(_ x: MLXArray, _ i: Int, _ j: Int) -> MLXArray {
        var h = x
        for k in 0 ..< resDilations.count { h = residualBlock(h, i, j, k, dilation: resDilations[k]) }
        return h
    }

    /// HiFiGANResLayer: run each kernel-branch independently on x, then average.
    private func resLayer(_ x: MLXArray, _ i: Int) -> MLXArray {
        var residuals: [MLXArray] = []
        for j in 0 ..< resKernelSizes.count { residuals.append(resBlock(x, i, j)) }
        var acc = residuals[0]
        for r in residuals.dropFirst() { acc = acc + r }
        return acc / MLXArray(Float(residuals.count))
    }

    // MARK: - forward

    /// Decode 32-channel dequantized codes [1, 32, T] -> waveform [1, 1, T*1024].
    public func decode(_ x: MLXArray) -> MLXArray {
        var h = causalConv(x, convW["pre"]!, convB["pre"]!, dilation: 1)   // kernel 7
        for i in 0 ..< NanoCodecDecoder.upSampleRates.count {
            h = halfSnake(h, alpha["act.\(i)"]!)
            h = transposeConv(h, convW["up.\(i)"]!, convB["up.\(i)"]!,
                              stride: NanoCodecDecoder.upSampleRates[i], outCh: chans[i + 1])
            h = resLayer(h, i)
            eval(h)   // mirror the reference's per-up-layer eval boundary; keeps the graph shallow
        }
        h = halfSnake(h, alpha["post_act"]!)
        h = causalConv(h, convW["post"]!, convB["post"]!, dilation: 1)     // kernel 3
        return clip(h, min: -1.0, max: 1.0)                                // output_activation = clamp
    }

    // MARK: - dequantize (for the full codes->wave path; the gate feeds decoder_input directly)

    /// Per-channel symmetric FSQ dequant of unfolded codes [1, C, T] -> floats in [-1,1].
    /// scale[c] = max(levels[c % L] // 2, 1); (code - scale)/scale.
    public static func dequantize(_ codes: MLXArray, levels: [Int] = [8, 7, 6, 6],
                                  numLayers: Int = 8) -> MLXArray {
        let cTotal = numLayers * levels.count
        let scaleVals = (0 ..< cTotal).map { Float(max(levels[$0 % levels.count] / 2, 1)) }
        let scale = MLXArray(scaleVals).reshaped([1, cTotal, 1])
        return (codes.asType(.float32) - scale) / scale
    }
}
