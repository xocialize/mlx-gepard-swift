// StreamingTests.swift — Gepard through the engine's STR gate (offline, no network, no GPU):
// STR-1 advertisement⇔conformance coherence, STR-2 pre-cancelled runStream (zero chunks +
// CancellationError unchanged), STR-3 posture declaration. The live half (STR-4/5/7 sequence /
// parity / mid-stream cancel + TTFA/cadence) runs in the `gepard-gates --stream` CLI lane.

import Foundation
import GepardCore
import MLXServeConformance
import MLXToolKit
import XCTest
@testable import MLXGepardTTS

final class StreamingTests: XCTestCase {

    /// STR-1: the `gepard` surface declares `.audioChunk` AND the package conforms.
    func testSTR1AdvertisementCoherence() {
        let report = StreamingConformance.checkAdvertisement(GepardPackage.self)
        XCTAssertTrue(report.passed, report.summary)
    }

    /// STR-2: pre-cancelled `runStream` surfaces CancellationError unchanged with zero chunks
    /// (the entry checkpoint precedes the first emit).
    func testSTR2PreCancelledStream() async {
        let package = GepardPackage(configuration: GepardConfiguration())
        let report = await StreamingConformance.checkPreCancelledStream(
            package: package,
            request: TTSRequest(text: "hi",
                                voice: VoiceSelector(.referenceAudio(
                                    Audio(format: .wav, data: Data())))))
        XCTAssertTrue(report.passed, report.summary)
    }

    /// STR-3: the posture of record. Chunk cadence 6-then-12 frames; left context is the
    /// computed decoder receptive field (≈26, bound 32 — see NanoCodecDecoder
    /// `leftReceptiveFieldFrames`); 22.05 kHz; RunProgress reported per FRAME (the CAN-3
    /// generate/frame cadence — referenced, not duplicated, per the STR-3 convention).
    func testSTR3PostureDeclaration() {
        let posture = StreamingConformance.StreamPosture(
            chunkFrames: 12, leftContextFrames: 32, sampleRate: 22_050,
            reportsRunProgress: true)
        let report = StreamingConformance.checkPosture(GepardPackage.self, posture: posture)
        XCTAssertTrue(report.passed, report.summary)
    }

    /// The receptive-field computation itself (pure arithmetic — the exactness bound):
    /// NeMo-standard kernels [16,16,8,4,4] over strides [8,8,4,2,2] ⇒ 26 input frames.
    func testReceptiveFieldComputation() {
        let frames = NanoCodecDecoder.computeLeftReceptiveField(
            preKernel: 7, postKernel: 3,
            upKernels: [16, 16, 8, 4, 4], upRates: [8, 8, 4, 2, 2],
            resKernelSizes: [3, 7, 11], resDilations: [1, 3, 5])
        XCTAssertEqual(frames, 26)
    }
}
