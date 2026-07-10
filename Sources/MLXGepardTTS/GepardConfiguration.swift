import Foundation
import MLXToolKit

/// Init-time configuration for `GepardPackage` (C9): where the Gepard weights live and which
/// quant tier the model runs at. Per-request text/voice/metaData ride the canonical `TTSRequest`.
///
/// Two weight sources back one loaded model (tier-2 pipeline):
/// - `repo` — the public Gepard-1.0 checkpoint (the AR LM + Q-Former in `model.safetensors`,
///   plus `config.json` and the Qwen3.5 `tokenizer.json` / `tokenizer_config.json` the text
///   conditioner loads). Apache-2.0 weights.
/// - `codecRepo` — the MLX-converted NVIDIA NanoCodec (22.05 kHz / 1.89 kbps / 21.5 fps),
///   NVIDIA-Open-Model-licensed; ships a `model.safetensors` carrying both `audio_encoder.*`
///   (reference encode) and `audio_decoder.*` (vocode).
///
/// `quant` is `.bf16` (as-shipped): the 555.7M-param model is small enough that V1 ships a
/// single tier — the LM binds at native bf16 (~1.1 GB), the codec fuses weight-norm in fp32
/// (the P7-validated path). int8/int4 are a V2 nice-to-have and would quantize the LM Linears
/// in-memory at load without changing the materialization set.
public struct GepardConfiguration: PackageConfiguration, ModelStorable, QuantConfigured {
    /// The public Gepard-1.0 checkpoint repo (weights + tokenizer).
    public var repo: String
    /// Pinned revision; nil = main.
    public var revision: String?
    /// The MLX-converted NanoCodec repo.
    public var codecRepo: String
    /// Quant tier: `.bf16` as-shipped (int8/int4 are future).
    public var quant: Quant
    /// Explicit checkpoint directory (dev escape hatch — never touches the network). Expected to
    /// hold `model.safetensors` + `tokenizer.json` + `tokenizer_config.json`.
    public var modelDirectory: URL?
    /// Explicit codec directory (expects `model.safetensors` inside).
    public var codecDirectory: URL?
    /// Engine-chosen models root (auto-materialization target). Environment-specific.
    public var modelsRootDirectory: URL?

    public init(
        repo: String = "nineninesix/gepard-1.0",
        revision: String? = nil,
        codecRepo: String = "mlx-community/nemo-nano-codec-22khz-1.89kbps-21.5fps",
        quant: Quant = .bf16,
        modelDirectory: URL? = nil,
        codecDirectory: URL? = nil,
        modelsRootDirectory: URL? = nil
    ) {
        self.repo = repo
        self.revision = revision
        self.codecRepo = codecRepo
        self.quant = quant
        self.modelDirectory = modelDirectory
        self.codecDirectory = codecDirectory
        self.modelsRootDirectory = modelsRootDirectory
    }

    // Environment-specific URLs are excluded from Codable.
    private enum CodingKeys: String, CodingKey {
        case repo, revision, codecRepo, quant
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        repo = try c.decode(String.self, forKey: .repo)
        revision = try c.decodeIfPresent(String.self, forKey: .revision)
        codecRepo = try c.decodeIfPresent(String.self, forKey: .codecRepo)
            ?? "mlx-community/nemo-nano-codec-22khz-1.89kbps-21.5fps"
        quant = try c.decode(Quant.self, forKey: .quant)
    }
}

// MARK: - Weight sources (auto-materialization, engine MAT gate)

extension GepardConfiguration: WeightSourcing {
    /// Files the "main" source fetches: the LM weights + config + the Qwen3.5 tokenizer pair.
    static let mainFiles = [
        "model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json",
    ]
    /// Representative main file for the missing-probe (the biggest, always present).
    static let mainProbeFile = "model.safetensors"
    /// Codec glob + probe file (published as a single `model.safetensors`).
    static let codecMatching = ["*.safetensors"]
    static let codecProbeFile = "model.safetensors"

    public var weightSources: [WeightSource] {
        [
            WeightSource(role: "main", repo: repo, revision: revision, matching: Self.mainFiles),
            WeightSource(role: "codec", repo: codecRepo, matching: Self.codecMatching),
        ]
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        let fm = FileManager.default
        func storeHas(_ repo: String, file: String) -> Bool {
            guard let dir = ModelStore(root: storeRoot).directory(for: repo) else { return false }
            return fm.fileExists(atPath: dir.appending(path: file).path)
        }
        return weightSources.filter { source in
            switch source.role {
            case "main":
                if let dir = modelDirectory,
                   fm.fileExists(atPath: dir.appending(path: Self.mainProbeFile).path) { return false }
                return !storeHas(source.repo, file: Self.mainProbeFile)
            default:  // codec
                if let dir = codecDirectory,
                   fm.fileExists(atPath: dir.appending(path: Self.codecProbeFile).path) { return false }
                return !storeHas(source.repo, file: Self.codecProbeFile)
            }
        }
    }

    /// The configuration with nil directories resolved to the store layout — what `load()` uses
    /// AFTER materialization. Explicit directories always win.
    public func resolved(storeRoot: URL?) -> GepardConfiguration {
        let store = ModelStore(root: storeRoot)
        var cfg = self
        if cfg.modelDirectory == nil { cfg.modelDirectory = store.directory(for: repo) }
        if cfg.codecDirectory == nil { cfg.codecDirectory = store.directory(for: codecRepo) }
        return cfg
    }
}

// MARK: - Cold-start prewarm

extension GepardConfiguration: WeightPrewarming {
    public var prewarmPaths: [URL] {
        // Store-resolved view so auto-materialize (nil-dir) configs prewarm the downloaded layout
        // on later cold launches; missing paths are skipped (best-effort prewarmer).
        let r = resolved(storeRoot: modelsRootDirectory)
        var paths: [URL] = []
        if let dir = r.modelDirectory { paths.append(dir.appending(path: Self.mainProbeFile)) }
        if let dir = r.codecDirectory { paths.append(dir.appending(path: Self.codecProbeFile)) }
        return paths
    }
}
