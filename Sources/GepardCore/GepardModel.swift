// GepardModel — the assembled, engine-facing inference unit: backbone + audio interface +
// heads (as a GepardDecoder), the Q-Former voice cloner, the NanoCodec encoder/decoder, and
// the text conditioner. This is what the MLXGepardTTS wrapper's load() builds and run() drives.
//
// The weight-binding here mirrors the gepard-gates loaders EXACTLY (same key filters / remaps
// the P1–P9 gates validated), with one deliberate difference: the LM weights are bound at their
// native bf16 (the as-shipped tier — ~1.1 GB resident, the plan's target) rather than upcast to
// fp32 the way the CPU-stream parity gates do. The NanoCodec weights stay fp32 (the codec loader
// fuses weight-norm in fp32, the P7-validated deep-conv path). GPU-stream forward, per P8.

import Foundation
import MLX
import MLXNN
import MLXLMCommon

public final class GepardModel {

    // FSQ / codec geometry (gepard_config.json + codec_config.yaml, verified).
    public static let fsqLevels = [8, 7, 6, 6]
    public static let numCodecGroups = 8
    /// Per-channel FSQ levels for the 32 unfolded code channels (group-major: g·4 + d).
    public static let levels32 = (0 ..< 32).map { fsqLevels[$0 % 4] }
    /// Codec I/O sample rate — 22.05 kHz mono (NOT 24 kHz; the provider carries its own rate).
    public static let sampleRate = 22050
    /// Samples per codec frame (down_sample_rates product) → 21.5 fps.
    public static let samplesPerFrame = 1024

    public let decoder: GepardDecoder
    public let refCompressor: RefCompressor
    public let codecEncoder: NanoCodecEncoder
    public let codecDecoder: NanoCodecDecoder
    public let conditioner: GepardConditioner

    init(decoder: GepardDecoder, refCompressor: RefCompressor,
         codecEncoder: NanoCodecEncoder, codecDecoder: NanoCodecDecoder,
         conditioner: GepardConditioner) {
        self.decoder = decoder
        self.refCompressor = refCompressor
        self.codecEncoder = codecEncoder
        self.codecDecoder = codecDecoder
        self.conditioner = conditioner
    }

    // MARK: - Loading

    public enum LoadError: Error, CustomStringConvertible {
        case missingBackboneKeys, missingRefCompressorKeys
        case noCodecKeys(String)
        case loadFailed(String)
        public var description: String {
            switch self {
            case .missingBackboneKeys: return "model.safetensors: model.* backbone keys missing"
            case .missingRefCompressorKeys: return "model.safetensors: ref_compressor.* keys missing"
            case .noCodecKeys(let s): return "codec.safetensors: no \(s) keys"
            case .loadFailed(let s): return "weight load failed: \(s)"
            }
        }
    }

    /// Build the full model from the two safetensors + the tokenizer folder.
    /// - modelURL: nineninesix/gepard-1.0 `model.safetensors` (the AR LM + Q-Former).
    /// - codecURL: nemo-nano-codec 21.5fps safetensors (encoder + decoder).
    /// - tokenizerFolder: folder holding `tokenizer.json` + `tokenizer_config.json`.
    public static func load(modelURL: URL, codecURL: URL, tokenizerFolder: URL) async throws -> GepardModel {
        // LM weights (bf16 native).
        let raw: [String: MLXArray]
        do { raw = try loadArrays(url: modelURL) } catch { throw LoadError.loadFailed("\(error)") }

        let backbone = try bindBackbone(raw)
        let audio = try bindAudio(raw)
        let heads = try bindHeads(raw)
        let refCompressor = try bindRefCompressor(raw)
        let decoder = GepardDecoder(backbone: backbone, audio: audio, heads: heads)
        eval(backbone, audio, heads, refCompressor)

        // Codec weights (fp32 — the loaders fuse weight-norm in fp32 internally).
        let craw: [String: MLXArray]
        do { craw = try loadArrays(url: codecURL) } catch { throw LoadError.loadFailed("\(error)") }
        guard craw.keys.contains(where: { $0.hasPrefix("audio_decoder.") }) else {
            throw LoadError.noCodecKeys("audio_decoder.*")
        }
        guard craw.keys.contains(where: { $0.hasPrefix("audio_encoder.") }) else {
            throw LoadError.noCodecKeys("audio_encoder.*")
        }
        let codecDecoder = NanoCodecDecoder(weights: craw)
        let codecEncoder = NanoCodecEncoder(weights: craw)

        // Tokenizer (async).
        let conditioner = try await GepardConditioner.load(modelFolder: tokenizerFolder)

        return GepardModel(decoder: decoder, refCompressor: refCompressor,
                           codecEncoder: codecEncoder, codecDecoder: codecDecoder,
                           conditioner: conditioner)
    }

    // Native-dtype binders (mirror the gate loaders' key filters; no fp32 upcast).

    private static func bindBackbone(_ raw: [String: MLXArray]) throws -> GepardBackbone {
        var w: [String: MLXArray] = [:]
        for (k, v) in raw where k.hasPrefix("model.") { w[String(k.dropFirst("model.".count))] = v }
        guard w["embed_tokens.weight"] != nil, w["norm.weight"] != nil else {
            throw LoadError.missingBackboneKeys
        }
        let m = GepardBackbone(GepardBackboneConfig())
        do { try m.update(parameters: ModuleParameters.unflattened(w), verify: [.all]) }
        catch { throw LoadError.loadFailed("backbone: \(error)") }
        return m
    }

    private static func bindAudio(_ raw: [String: MLXArray]) throws -> AudioInterface {
        var w: [String: MLXArray] = [:]
        for (k, v) in raw where k.hasPrefix("audio_embeddings.")
            || k.hasPrefix("audio_embed_proj.") || k == "audio_embed_scale" {
            w[AudioInterface.remapProjKey(k)] = v
        }
        let m = AudioInterface(levels: levels32)
        do { try m.update(parameters: ModuleParameters.unflattened(w), verify: [.all]) }
        catch { throw LoadError.loadFailed("audio: \(error)") }
        return m
    }

    private static func bindHeads(_ raw: [String: MLXArray]) throws -> CodebookHeads {
        var w: [String: MLXArray] = [:]
        for (k, v) in raw where k.hasPrefix("codebook_heads.") || k.hasPrefix("stop_head.") { w[k] = v }
        let m = CodebookHeads(levels: levels32)
        do { try m.update(parameters: ModuleParameters.unflattened(w), verify: [.all]) }
        catch { throw LoadError.loadFailed("heads: \(error)") }
        return m
    }

    private static func bindRefCompressor(_ raw: [String: MLXArray]) throws -> RefCompressor {
        var w: [String: MLXArray] = [:]
        for (k, v) in raw where k.hasPrefix("ref_compressor.") {
            w[String(k.dropFirst("ref_compressor.".count))] = v
        }
        guard w["queries"] != nil, w["output_scale"] != nil else {
            throw LoadError.missingRefCompressorKeys
        }
        let m = RefCompressor()
        do { try m.update(parameters: ModuleParameters.unflattened(w), verify: [.all]) }
        catch { throw LoadError.loadFailed("ref_compressor: \(error)") }
        return m
    }

    // MARK: - Reference conditioning (once per voice clip)

    /// mono 22.05 kHz samples → unfolded ref codes `[1, T, 32]` (the Q-Former's input).
    /// Encoder (packed `[1, 8, T]`) → CodecOps mixed-radix unfold → `[1, T, 32]`.
    public func encodeReference(samples: [Float]) -> MLXArray {
        let wave = MLXArray(samples).reshaped([1, samples.count])
        let packed = codecEncoder.encode(wave)                       // [1, 8, T]  Int32
        eval(packed)
        let g = packed.dim(1), T = packed.dim(2)
        let flat = packed.asArray(Int32.self)                        // row-major [g*T]
        let packed2D: [[Int]] = (0 ..< g).map { grp in (0 ..< T).map { t in Int(flat[grp * T + t]) } }
        let unfolded = CodecOps.unfoldTokens(packed2D, levels: Self.fsqLevels)  // [32][T]
        // -> [1, T, 32]
        var out = [Int32](repeating: 0, count: T * 32)
        for c in 0 ..< 32 { for t in 0 ..< T { out[t * 32 + c] = Int32(unfolded[c][t]) } }
        return MLXArray(out, [1, T, 32])
    }

    /// The frozen speaker prefix `[1, 8, d]` from unfolded ref codes (deterministic — cacheable).
    public func voicePrefix(refCodes: MLXArray) -> MLXArray {
        refCompressor(refCodes: refCodes).prefix
    }

    // MARK: - Synthesis

    /// Rollout codes `[T][32]` → decoder input `[1, 32, T]` (dequantized) — the P8 `codesToInput`.
    private func codesToDecoderInput(_ frames: [[Int]]) -> MLXArray {
        let T = frames.count
        var flat = [Int32](repeating: 0, count: 32 * T)
        for t in 0 ..< T { for c in 0 ..< 32 { flat[c * T + t] = Int32(frames[t][c]) } }
        return NanoCodecDecoder.dequantize(MLXArray(flat, [1, 32, T]))
    }

    /// Full synthesis from a prepared speaker prefix + cond ids → mono 22.05 kHz samples.
    /// `cancelCheck` fires per generated frame (rethrown unchanged); `onFrame(count)` reports
    /// progress. `options` carries maxFrames + the onset-CFG knob.
    public func synthesize(
        prefix: MLXArray, condIds: [Int], options: GepardDecoder.Options,
        cancelCheck: (() throws -> Void)? = nil, onFrame: ((Int) -> Void)? = nil
    ) rethrows -> [Float] {
        let ids = MLXArray(condIds.map { Int32($0) })
        let rollout = try decoder.rollout(
            prefix: prefix, condIds: ids, options: options,
            cancelCheck: cancelCheck, onFrame: onFrame)
        let decoderInput = codesToDecoderInput(rollout.codes)        // [1, 32, T]
        let wave = codecDecoder.decode(decoderInput).reshaped([-1])  // [T*1024]
        eval(wave)
        return wave.asArray(Float.self)
    }
}
