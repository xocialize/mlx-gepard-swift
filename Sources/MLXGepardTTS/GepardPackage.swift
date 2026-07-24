import Foundation
import GepardCore
import MLX
import MLXRandom
import MLXToolKit

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
public final class GepardPackage: ModelPackage {
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
                        + "Onset text-CFG (metaData.cfgScale) rescues 1–2-word utterances.",
                    modes: [.neutral, .expressive]
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
    private var cachedReference: (key: Int, prefix: MLXArray)?

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
        cachedReference = nil
        MLX.Memory.clearCache()   // release the retained MLX pool so eviction frees RSS
    }

    // MARK: - Run

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: the entry checkpoint is the FIRST act of run() — before notLoaded validation
        // (engine ≥ 0.27.0). Mid-run cadence: the AR rollout bails per generated frame via the
        // throwing `cancelCheck` closure threaded into GepardModel.synthesize, rethrowing the
        // CancellationError unchanged so the engine can classify user-cancel vs governor-preempt.
        try Task.checkCancellation()
        guard let model else { throw PackageError.notLoaded }
        guard request.capability == .tts, let tts = request as? TTSRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }

        // Voice: zero-shot cloning only.
        guard case .referenceAudio(let referenceClip) = tts.voice.selection else {
            throw PackageError.unsupportedRequestFeature(
                "voice — Gepard has no preset voices; provide voice.referenceAudio")
        }

        // Reference conditioning → frozen speaker prefix (memoized per reference clip bytes).
        let key = Self.referenceKey(referenceClip.data)
        let prefix: MLXArray
        if let cached = cachedReference, cached.key == key {
            prefix = cached.prefix
        } else {
            RunProgress.report(.encode)
            let (mono, sourceRate) = try AudioSupport.decodeToMono(referenceClip)
            let samples22k = SincResampler.resample(
                audio: mono, from: sourceRate, to: GepardModel.sampleRate)
            let refCodes = model.encodeReference(samples: samples22k)
            let p = model.voicePrefix(refCodes: refCodes)
            eval(p)
            prefix = p
            cachedReference = (key, p)
        }
        try Task.checkCancellation()

        // Text → cond_ids (P10-validated conditioner).
        let condIds = model.conditioner.condIds(for: tts.text)

        // metaData plane.
        if let seed = tts.metaData.intValue("seed") {
            // Clamp to 32 bits (Qwen3 precedent — large 64-bit seeds break EOS). No-op under
            // greedy decoding, wired for a future temperature path.
            MLXRandom.seed(UInt64(bitPattern: Int64(seed)) & 0xFFFF_FFFF)
        }
        let maxFrames = tts.metaData.intValue("maxFrames") ?? 2000
        var options = GepardDecoder.Options(maxFrames: max(1, maxFrames))
        if let stopThreshold = tts.metaData.doubleValue("stopThreshold") {
            // The stop head is prefix-conditioned: some reference clips cross 0.5 at a
            // mid-utterance sentence pause and truncate multi-sentence text. Clamp to a sane
            // sigmoid band; the oracle exposes the same knob (runner.py stop_threshold).
            options.stopThreshold = Float(min(max(stopThreshold, 0.05), 0.99))
        }
        if let cfgScale = tts.metaData.doubleValue("cfgScale") {
            options.cfgScale = Float(cfgScale)
            options.cfgFrames = tts.metaData.intValue("cfgFrames") ?? 20
            options.uncondIds = [GepardConditioner.sot, GepardConditioner.eot, GepardConditioner.sos]
        }

        // Rollout → dequant → vocode. Per-frame cancellation + progress at the generation seam.
        let samples = try model.synthesize(
            prefix: prefix, condIds: condIds, options: options,
            cancelCheck: { try Task.checkCancellation() },
            onFrame: { count in RunProgress.report(.generate, step: count) })

        try Task.checkCancellation()
        RunProgress.report(.decode)
        let wav = AudioSupport.encodeWAV16(samples: samples, sampleRate: GepardModel.sampleRate)
        return TTSResponse(audio: Audio(
            format: .wav, data: wav, sampleRate: GepardModel.sampleRate, channels: 1))
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
