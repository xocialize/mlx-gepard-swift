// CancellationTests.swift — Gepard through the engine's CAN gate (offline, no MLX kernels).
// CAN-1/2 drive the real run() pre-cancelled (the entry checkpoint fires before notLoaded
// validation or weights); CAN-3 is the document of record for the checkpoint cadence: the AR
// rollout bails per generated frame via the throwing cancelCheck threaded into
// GepardModel.synthesize → GepardDecoder.rollout, rethrowing CancellationError unchanged, and
// the wrapper checkpoints again after ref-encode and before vocode.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import MLXGepardTTS

final class CancellationTests: XCTestCase {

    // MARK: - CAN-1 / CAN-2 — pre-cancelled run() propagation + classification

    func testCANGatePreCancelledRun() async {
        // Construction is cheap (C13) and the entry checkpoint throws before validation
        // (including the referenceAudio-only voice gate) or weights are touched — offline-safe.
        let package = GepardPackage(configuration: GepardConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: TTSRequest(text: "probe"))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - CAN-3 — checkpoint-cadence declaration (the document of record)

    func testCANCadenceDeclaration() {
        // tts is a long-run capability (and the declared transient independently implies long
        // runs) — the sub-second exemption is not available.
        XCTAssertTrue(CancellationConformance.longRunImplied(by: GepardPackage.manifest))

        let report = CancellationConformance.checkCadence(
            manifest: GepardPackage.manifest,
            posture: .cadence([
                // The AR rollout checks cancellation once per generated frame (GepardDecoder
                // .rollout's `try cancelCheck?()` at the top of the loop), and reports
                // RunProgress(.generate, step:) at that same seam — observable evidence.
                .init(phase: .generate, unit: .frame, reportsRunProgress: true),
                // The wrapper also checkpoints after ref-encode and before the codec vocode.
                .init(phase: .decode, unit: .chunk),
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }
}
