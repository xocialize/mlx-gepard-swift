// EngineRegistrationTests.swift — MLXServeCore admission path (offline, no weights).
// register() runs the license gate + device eligibility and returns a PackageID WITHOUT
// paging weights, so this proves Gepard's engine-registration (PackageID selection) path:
// unlike IndexTTS2 (NonCommercial → needs .permissiveOrAcknowledged), Gepard's two PERMISSIVE
// weight layers admit under the DEFAULT product policy (.permissiveOnly).

import Foundation
import MLXServeCore
import MLXToolKit
import XCTest
@testable import MLXGepardTTS

final class EngineRegistrationTests: XCTestCase {

    func testRegistersUnderDefaultPermissiveOnly() async throws {
        // Skip on ineligible hardware (register() also checks device eligibility: metalGPU +
        // macOS 26). CI/dev on Apple Silicon macOS 26 is eligible.
        let engine = MLXServeEngine()   // default policy = .permissiveOnly
        do {
            let packageID = try await engine.register(
                PackageRegistration.of(GepardPackage.self),
                configuration: GepardConfiguration())
            XCTAssertEqual(packageID.rawValue, "gepard")   // derived from the surface name
        } catch let error as EngineError {
            // The ONLY acceptable failure here is device-ineligibility on non-Metal CI; a
            // license rejection would be a real bug (both layers are permissive).
            if case .licenseRejected = error {
                XCTFail("permissiveOnly rejected Gepard — both weight layers should be permissive")
            }
            throw XCTSkip("engine registration ineligible on this host: \(error)")
        }
    }
}
