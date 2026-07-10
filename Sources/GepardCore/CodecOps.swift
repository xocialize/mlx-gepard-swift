// CodecOps — mixed-radix unfold + symmetric FSQ dequantization.
// Isomorphic to gepard_inference/codec_ops.py. Pure integer/float math, bit-exact.
//
// Mixed-radix decomposition (inverse of FSQ packing), little-endian:
//   packed token k encodes len(levels) per-dim codes as
//     k = code_0 + code_1*L_0 + code_2*L_0*L_1 + ...
//   so   code_d = (k / prod(L_0..L_{d-1})) % L_d
// Channel order for codebook c and FSQ dim d: output channel index = c*D + d.

import Foundation

public enum CodecOps {

    /// Cumulative-product bases for the little-endian mixed radix: [1, L0, L0*L1, ...].
    @inline(__always)
    static func bases(_ levels: [Int]) -> [Int] {
        var out = [Int](repeating: 1, count: levels.count)
        for i in 1..<levels.count { out[i] = out[i - 1] * levels[i - 1] }
        return out
    }

    /// Mixed-radix decomposition of packed codec token indices.
    /// - packed: `[C][T]` packed indices (channels-major, batch=1 implicit).
    /// - levels: FSQ levels per dimension within each codebook (e.g. [8,7,6,6]).
    /// - returns: `[C*D][T]` per-dimension discrete codes; channel `c*D + d`.
    public static func unfoldTokens(_ packed: [[Int]], levels: [Int]) -> [[Int]] {
        let D = levels.count
        let base = bases(levels)
        var out = [[Int]]()
        out.reserveCapacity(packed.count * D)
        for chan in packed {                      // codebook c
            for d in 0..<D {                      // FSQ dim d
                var row = [Int](repeating: 0, count: chan.count)
                let L = levels[d], b = base[d]
                for t in 0..<chan.count { row[t] = (chan[t] / b) % L }
                out.append(row)
            }
        }
        return out
    }

    /// Per-dimension symmetric dequantization of unfolded FSQ codes: `(x - L/2) / (L/2)`.
    /// - unfolded: `[C_total][T]` integer codes, `C_total = numLayers * levels.count`.
    /// - returns: floats in `[-1, 1]`. Per-channel level tiles `levels` `numLayers` times.
    public static func dequantize(_ unfolded: [[Int]], levels: [Int], numLayers: Int) -> [[Float]] {
        let cTotal = unfolded.count
        precondition(cTotal == numLayers * levels.count,
                     "dequantize: channels \(cTotal) != numLayers*len(levels) \(numLayers * levels.count)")
        let D = levels.count
        var out = [[Float]]()
        out.reserveCapacity(cTotal)
        for c in 0..<cTotal {
            let L = levels[c % D]
            let scale = Float(max(L / 2, 1))       // clamp_min(1) as in Python
            let inv = 1.0 / scale
            var row = [Float](repeating: 0, count: unfolded[c].count)
            for t in 0..<unfolded[c].count { row[t] = (Float(unfolded[c][t]) - scale) * inv }
            out.append(row)
        }
        return out
    }
}
