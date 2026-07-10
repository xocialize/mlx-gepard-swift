// MaterializationTests.swift — Gepard through the engine's MAT gate (offline, no network):
// the WeightSourcing declaration, fresh-machine honesty, explicit-path satisfaction, and the
// store-layout probe/resolution. Two sources (main LM+tokenizer, converted NanoCodec).

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import MLXGepardTTS

final class MaterializationTests: XCTestCase {

    /// Temp dirs holding probe files that make an explicit-dir config read as satisfied.
    private func satisfiedDirs() throws -> (main: URL, codec: URL, cleanup: () -> Void) {
        let base = FileManager.default.temporaryDirectory
            .appending(path: "gepard-mat-\(UUID().uuidString)")
        let main = base.appending(path: "main")
        let codec = base.appending(path: "codec")
        try FileManager.default.createDirectory(at: main, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: codec, withIntermediateDirectories: true)
        FileManager.default.createFile(
            atPath: main.appending(path: GepardConfiguration.mainProbeFile).path, contents: Data([0]))
        FileManager.default.createFile(
            atPath: codec.appending(path: GepardConfiguration.codecProbeFile).path, contents: Data([0]))
        return (main, codec, { try? FileManager.default.removeItem(at: base) })
    }

    // MARK: - Engine MAT gate

    func testMATGate() throws {
        let (main, codec, cleanup) = try satisfiedDirs()
        defer { cleanup() }
        let report = MaterializationConformance.check(
            freshConfiguration: GepardConfiguration(),
            satisfiedConfiguration: GepardConfiguration(
                modelDirectory: main, codecDirectory: codec))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - Source declaration shape

    func testDeclaresTwoSources() {
        let sources = GepardConfiguration().weightSources
        XCTAssertEqual(sources.map(\.role), ["main", "codec"])
        XCTAssertEqual(sources[0].repo, "nineninesix/gepard-1.0")
        XCTAssertEqual(sources[0].matching, GepardConfiguration.mainFiles)
        XCTAssertEqual(sources[1].repo, "mlx-community/nemo-nano-codec-22khz-1.89kbps-21.5fps")
        XCTAssertEqual(sources[1].matching, GepardConfiguration.codecMatching)
    }

    // MARK: - Store-layout probe + resolution

    func testStoreLayoutSatisfiesAndResolves() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(path: "gepard-store-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let cfg = GepardConfiguration()
        // Empty store: both sources missing.
        XCTAssertEqual(cfg.missingWeightSources(storeRoot: root).count, 2)

        // Materialize the probe files into the store layout → nothing missing.
        let store = ModelStore(root: root)
        for (repo, file) in [(cfg.repo, GepardConfiguration.mainProbeFile),
                             (cfg.codecRepo, GepardConfiguration.codecProbeFile)] {
            let dir = store.directory(for: repo)!
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            FileManager.default.createFile(atPath: dir.appending(path: file).path, contents: Data([0]))
        }
        XCTAssertEqual(cfg.missingWeightSources(storeRoot: root).count, 0)

        // Resolution points nil dirs at the store layout.
        let resolved = cfg.resolved(storeRoot: root)
        XCTAssertEqual(resolved.modelDirectory, store.directory(for: cfg.repo))
        XCTAssertEqual(resolved.codecDirectory, store.directory(for: cfg.codecRepo))
    }
}
