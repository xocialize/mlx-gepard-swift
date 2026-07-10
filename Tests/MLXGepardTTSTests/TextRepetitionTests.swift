// TextRepetitionTests.swift — offline half of the P10 tokenizer gate: the TextRepeater framing
// (SOT/EOT/SOS + adaptive repetition) that turns tokenizer ids into cond_ids. The BPE half
// (tokenizer.json → ids) needs the 20 MB tokenizer, so it lives in the `gepard-gates --p10`
// CLI lane; this pins the framing bit-exact against the Stage-0 goldens with baked ids.

import Foundation
import GepardCore
import XCTest

final class TextRepetitionTests: XCTestCase {

    private let repeater = TextRepeater(
        config: GepardConditioner.textRepetition,
        startOfText: GepardConditioner.sot,
        endOfText: GepardConditioner.eot,
        startOfSpeech: GepardConditioner.sos)

    func testSpecialIdsMatchConfig() {
        XCTAssertEqual(GepardConditioner.sot, 248073)
        XCTAssertEqual(GepardConditioner.eot, 248074)
        XCTAssertEqual(GepardConditioner.sos, 248070)
        XCTAssertTrue(GepardConditioner.textRepetition.enabled)
        XCTAssertEqual(GepardConditioner.textRepetition.targetTextTokens, 16)
        XCTAssertEqual(GepardConditioner.textRepetition.applyBelow, 13)
        XCTAssertEqual(GepardConditioner.textRepetition.maxRepeats, 8)
    }

    // one_word "One." → ids [3833, 13] (2 tokens < applyBelow 13) → R = ceil(16/2) = 8.
    func testShortTextRepeatsToBudget() {
        let ids = [3833, 13]
        XCTAssertEqual(repeater.targetR(ids.count), 8)
        let cond = repeater.expand(ids)
        // 7 blocks [SOT,3833,13,EOT] + canonical [SOT,3833,13,EOT,SOS] = 7*4 + 5 = 33
        var expected: [Int] = []
        for _ in 0 ..< 7 { expected += [248073, 3833, 13, 248074] }
        expected += [248073, 3833, 13, 248074, 248070]
        XCTAssertEqual(cond, expected)
        XCTAssertEqual(cond.count, 33)
    }

    // canonical (20 tokens ≥ applyBelow 13) → R = 1 → single [SOT, …ids, EOT, SOS].
    func testLongTextNoRepeat() {
        let ids = [9419, 1017, 0, 2921, 803, 369, 11491, 3994, 11, 321,
                   353, 2688, 2098, 6051, 310, 1436, 488, 1495, 3242, 13]
        XCTAssertEqual(repeater.targetR(ids.count), 1)
        let cond = repeater.expand(ids)
        XCTAssertEqual(cond, [248073] + ids + [248074, 248070])
        XCTAssertEqual(cond.count, 23)
    }

    // greeting "Hello there." → 3 tokens → R = ceil(16/3) = 6.
    func testMidLengthRepeat() {
        XCTAssertEqual(repeater.targetR(3), 6)
        XCTAssertEqual(repeater.targetR(6), 3)   // "Let's try this now." (6 tokens)
        XCTAssertEqual(repeater.targetR(10), 2)  // pangram (10 tokens)
    }
}
