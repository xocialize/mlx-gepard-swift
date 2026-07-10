# mlx-gepard-swift

Swift-MLX port of **[Gepard-1.0](https://huggingface.co/nineninesix/gepard-1.0)** — a
realtime, streaming-first, **zero-shot voice-cloning** text-to-speech model — wrapped as a
conformant [MLXEngine](https://github.com/xocialize/mlx-engine-swift) `tts` `ModelPackage` for
Apple Silicon.

Clone a voice from a single short reference clip and stream speech at **~4× realtime** with a
**~17 ms time-to-first-audio**. 22.05 kHz mono `.wav` out.

```
reference clip ──▶ NanoCodec encode ──▶ Q-Former ──▶ speaker prefix ┐
                                                                    ├─▶ AR decode ─▶ NanoCodec decode ─▶ waveform
text ──▶ Qwen3.5 tokenizer ──▶ cond_ids ────────────────────────────┘
```

## Highlights

- **Zero-shot cloning** from a frozen reference clip (no per-voice training) — the voice is
  *deterministic and consistent* per clip.
- **Realtime + streaming-first**: measured gen-RTF **0.24 (4.2× realtime)**, TTFA **17 ms** on an
  M5 Max (fp32 unoptimized).
- **Small footprint**: **~1.2 GB resident** (bf16 language model + fp32 codec) — fits every tier.
- **Two permissive weight layers** ⇒ ships under MLXEngine's default `.permissiveOnly` policy
  (no acknowledgement flow): language model **Apache-2.0**, NanoCodec **NVIDIA Open Model License**.
- **Onset text-CFG** short-utterance rescue exposed as a per-request knob (default off).

## Architecture

Two modules (mirrors `mlx-indextts2-swift`'s core/wrapper split):

| Module | Role |
|---|---|
| **`GepardCore`** | The inference port — a Qwen3.5 full-attention backbone (lifted from `mlx-swift-lm`'s `Qwen35`), the Q-Former voice cloner, the audio interface + 32 codebook heads + stop head, the greedy AR decode loop (with optional onset-CFG), the NanoCodec encoder/decoder, and the text→`cond_ids` conditioner. Depends on MLX + swift-transformers only — **MLXToolKit-free**. |
| **`MLXGepardTTS`** | The engine-facing wrapper — `GepardConfiguration` + `GepardPackage` (the `tts` surface), the two-layer license gate, `WeightSourcing` auto-materialization, and cooperative-cancellation seams. |

## Install

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/xocialize/mlx-gepard-swift", from: "0.1.0"),
]
```

Products: `GepardCore` (the model), `MLXGepardTTS` (the engine package). Requires **macOS 26+**
and Apple Silicon (Metal GPU).

## Usage (via MLXEngine)

Gepard is engine-owned — you register it with an `MLXServeEngine`, which constructs, loads,
drives, and evicts it. Both weight layers are permissive, so it admits under the **default**
policy:

```swift
import MLXGepardTTS
import MLXServeCore
import MLXToolKit

let engine = MLXServeEngine()                       // default .permissiveOnly
await engine.useModelStore(ModelStore(root: modelsFolder))   // where weights materialize

let packageID = try await engine.register(
    PackageRegistration.of(GepardPackage.self),
    configuration: GepardConfiguration())           // auto-materializes both sources on first run
try await engine.prepare(.tts, package: packageID)

let referenceClip = Audio(format: .wav, data: try Data(contentsOf: refURL), channels: 1)
let request = TTSRequest(
    text: "Hello there — it's good to see you again.",
    voice: VoiceSelector(.referenceAudio(referenceClip)))   // .referenceAudio is REQUIRED

let response = try await engine.run(request, package: packageID) as! TTSResponse
try response.audio.data.write(to: outURL)           // 22.05 kHz mono 16-bit WAV
```

### Voice

`voice.referenceAudio` only — Gepard is a pure zero-shot cloner with no preset voices; `.named` /
`.auto` reject legibly. The reference clip is any sample rate/channel count (resampled to 22.05 kHz
internally) and its prepared speaker prefix is memoized per clip for long-form/multi-line synthesis.
`referenceTranscript` is not consumed (conditioning is transcript-free).

### `metaData` knobs

| key | type | default | effect |
|---|---|---|---|
| `cfgScale` | double | *(off)* | Onset text-CFG weight `w` (tech-report §5.3 short-utterance rescue). Absent ⇒ the plain greedy path. |
| `cfgFrames` | int | 20 | Onset window CFG is applied over. |
| `maxFrames` | int | 2000 (~93 s) | Hard generation cap. |
| `seed` | int | — | Reserved (V1 decoding is deterministic greedy — currently a no-op). |

> **On CFG:** onset-CFG rescues many short utterances but is **not universally safe** per-clip
> (e.g. some 1–2-word inputs collapse to silence at `w=2.6`, consistent with the tech report's
> <50 % success). It ships **default-off**; enable it per-request via the app's short-word policy
> and audition per reference clip.

## Weights & licensing

Two weight sources, both fetched automatically on first `prepare()` into the engine's model store:

| role | repo | license |
|---|---|---|
| `main` | [`nineninesix/gepard-1.0`](https://huggingface.co/nineninesix/gepard-1.0) | Apache-2.0 |
| `codec` | [`mlx-community/nemo-nano-codec-22khz-1.89kbps-21.5fps`](https://huggingface.co/mlx-community/nemo-nano-codec-22khz-1.89kbps-21.5fps) | NVIDIA Open Model License |

The port **code** (this repo) is **Apache-2.0**. The NanoCodec weights carry the **NVIDIA Open
Model License Agreement** — commercially usable and redistributable; the codec repo ships the
Agreement and the required §3.1 Notice. Both licenses are on MLXEngine's permissive allowlist, so
the package admits under the default product policy.

## Performance & footprint

Measured on an M5 Max (`gepard-gates --validate`, and confirmed in-app through the full
`register → prepare → run → evict` path):

| metric | value |
|---|---|
| resident (post-load) | ~1.06 GB MLX-active / ~1.2 GB engine-charged |
| run-phase peak | ~4.8 GB for a 12.8 s utterance (whole-utterance codec decode dominates) |
| time-to-first-audio | ~17 ms |
| generation RTF | ~0.24 (**4.2× realtime**) |
| output | valid cloned speech, −25 dBFS, 22.05 kHz mono |

> The transient scales with utterance length because V1 decodes the whole utterance through the
> NanoCodec conv stack in one pass. In practice the consuming app sentence-chunks (so one `run()`
> ≈ one sentence); a future frame-streaming decode would bound it to a fixed window.

## Parity gates

The port was built phase-by-phase against PyTorch/NeMo goldens (the `mlx-porting` parity
doctrine). Gates live in the `gepard-gates` CLI lane (not XCTest — kernel gates need a reliable
metallib):

```
swift run -c release gepard-gates --p2      # CodecOps + text-repetition, bit-exact
                                  --p3..p6   # backbone / audio interface / Q-Former / AR rollout
                                  --p7 --p9  # NanoCodec decoder / encoder
                                  --p8       # full-pipeline GPU smoke (RTF/TTFA)
                                  --p10      # tokenizer → cond_ids, INTEGER-EXACT
                                  --validate # headless engine run() harness (footprint/dBFS/RTF)
                                  --ab-dtype # bf16-vs-fp32 rollout A/B
```

Offline Stage-2 conformance (manifest C0–C13, `MAT-1..5` materialization, `CAN-1..3`
cancellation, MLXServeEngine registration) runs under `swift test` — 12 tests.

## Credits

- **Gepard-1.0** — [nineninesix.ai](https://huggingface.co/nineninesix/gepard-1.0) (Apache-2.0
  language model).
- **NeMo NanoCodec** — [NVIDIA](https://huggingface.co/nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps)
  (NVIDIA Open Model License); the MLX-Python donor
  [`nineninesix-ai/nanocodec-mlx`](https://github.com/nineninesix-ai/nanocodec-mlx).
- Backbone block lifted from [`mlx-swift-lm`](https://github.com/ml-explore/mlx-swift-lm);
  tokenization via [`swift-transformers`](https://github.com/huggingface/swift-transformers).

## License

Port code: **Apache-2.0**. Model weights are governed by their respective licenses (see above).
