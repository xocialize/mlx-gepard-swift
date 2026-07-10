// ManifestTests.swift — offline conformance checks on the Stage-2 contract surface:
// two-layer license gate behavior (C7/C8), requirements shape (C10), specialty declaration
// (C6), surface/capability derivation (C1/C2). No MLX kernels, no weights.

import Foundation
import MLXToolKit
import XCTest
@testable import MLXGepardTTS

final class ManifestTests: XCTestCase {

    let manifest = GepardPackage.manifest

    func testLicenseGateTwoLayer() {
        // C7: BOTH weight layers permissive — Gepard admits under the DEFAULT product policy.
        // (LM Apache; codec NVIDIA-Open-Model, which 0.28.0 added to the permissive allowlist —
        // we declare the more-restrictive codec license as the weight license.)
        XCTAssertTrue(LicensePolicy.permissiveOnly.evaluate(manifest.license).isAdmitted)
        XCTAssertEqual(manifest.license.weightLicense, .nvidiaOpenModel)
        XCTAssertTrue(manifest.license.weightLicense.isPermissive)
        // C8: the port code is Apache-2.0.
        XCTAssertEqual(manifest.license.portCodeLicense, .apache2)
        XCTAssertTrue(manifest.license.portCodeLicense.isPermissive)
    }

    func testRequirementsAndSurfaces() {
        XCTAssertEqual(manifest.contractVersion, ContractVersion.current)   // C0
        XCTAssertEqual(manifest.capabilities, [.tts])                       // C1 (derived)
        XCTAssertEqual(Set(manifest.requirements.footprints.map(\.quant)), [.bf16])  // C10
        for footprint in manifest.requirements.footprints {
            XCTAssertGreaterThan(footprint.residentBytes, 0)
            XCTAssertGreaterThan(footprint.peakActivationBytes, 0)          // split declared
        }
        XCTAssertEqual(manifest.requirements.requiredBackends, [.metalGPU])
        XCTAssertEqual(manifest.requirements.os.minMacOS,
                       SemanticVersion(major: 26, minor: 0, patch: 0))
        // C6: voice-clone + realtime selection signals (no E12 emotion/duration lane).
        let specialties = Set(manifest.specialties.map(\.specialty.rawValue))
        XCTAssertTrue(specialties.contains("voiceClone"))
        XCTAssertTrue(specialties.contains("realtimeStreaming"))
        // C11: descriptor is well-formed.
        let surface = manifest.surfaces[0]
        XCTAssertEqual(surface.capability, .tts)
        XCTAssertFalse(surface.summary.isEmpty)
        XCTAssertFalse(surface.parameters.isEmpty)
        XCTAssertEqual(Set(surface.supportedModes), [.neutral, .expressive])
    }
}
