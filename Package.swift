// swift-tools-version: 6.2
// mlx-gepard-swift — Gepard-1.0 realtime conversational TTS ported to Swift-MLX.
//
// Two-module split (mirrors mlx-indextts2-swift's MLXIndexTTS2 / MLXIndexTTS2TTS):
//   • GepardCore   — the inference port (Qwen3.5 full-attn backbone, Q-Former voice cloner,
//                    audio interface + heads, AR decode loop, NanoCodec enc/dec) + the
//                    text→cond_ids conditioner. Depends on MLX + swift-transformers only —
//                    MLXToolKit-free, so the port stays engine-agnostic.
//   • MLXGepardTTS — the engine-facing wrapper: GepardConfiguration + GepardPackage (the
//                    `tts` ModelPackage), the two-layer license gate, WeightSourcing
//                    auto-materialization, and the CAN cancellation seams.
//
// Parity gates live in the `gepard-gates` CLI lane (NOT XCTest — the SPM test product's
// metallib is unreliable; `swift run` is the doctrine for gates that touch kernels). P1–P9
// cover the inference port; P10 gates the tokenizer→cond_ids path added at engine-wrap.
// XCTest carries the offline Stage-2 checks (manifest + MAT-1..5 + CAN-1..3).
//
// Engine contract pinned ≥0.28.0 (ships SPDXLicense.nvidiaOpenModel on the permissive
// allowlist — both Gepard weight layers admit under the default `.permissiveOnly`).
// Deliberately NO mlx-audio-dsp: Gepard has no mel/STFT (codec codes are the only audio
// features); the wrapper's only DSP need is a windowed-sinc resampler, which is inline
// pure-Swift in AudioSupport.

import PackageDescription

let package = Package(
    name: "mlx-gepard-swift",
    platforms: [
        .macOS(.v26)
    ],
    products: [
        .library(name: "GepardCore", targets: ["GepardCore"]),
        .library(name: "MLXGepardTTS", targets: ["MLXGepardTTS"]),
        .executable(name: "gepard-gates", targets: ["gepard-gates"]),
    ],
    dependencies: [
        // Qwen3.5 full-attn block lift + the MLXLMCommon rope/attention/KVCache helpers.
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMinor(from: "3.31.4")),
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.5"),
        // Faithful Qwen byte-level BPE (tokenizer.json) for text → cond_ids.
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        // Engine contract — 0.28.0 brought SPDXLicense.nvidiaOpenModel (permissive allowlist);
        // MLXServeConformance is the MAT/CAN offline gate harness.
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.28.0"),
        // Native downloader for WeightSourcing auto-materialization.
        .package(url: "https://github.com/huggingface/swift-huggingface.git", from: "0.9.0"),
    ],
    targets: [
        .target(
            name: "GepardCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/GepardCore",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .executableTarget(
            name: "gepard-gates",
            dependencies: [
                "GepardCore",
                // --validate exercises the real engine run() path (GepardPackage) headless.
                "MLXGepardTTS",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ],
            path: "Sources/gepard-gates",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .target(
            name: "MLXGepardTTS",
            dependencies: [
                "GepardCore",
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
            ],
            path: "Sources/MLXGepardTTS",
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "MLXGepardTTSTests",
            dependencies: [
                "GepardCore",
                "MLXGepardTTS",
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
            ],
            path: "Tests/MLXGepardTTSTests"
        ),
    ]
)
