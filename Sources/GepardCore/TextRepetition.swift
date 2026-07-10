// TextRepetition — adaptive text-repetition for short-utterance conditioning.
// Isomorphic to gepard_inference/text_repetition.py. Bit-exact layout logic; a
// mismatch vs training-time data prep collapses WER (per the reference docstring).
//
// On short inputs the K=8 speaker-prefix tokens dominate the hidden state and the
// 1–2 text tokens drown, so the model never latches -> runaway. The fix repeats the
// text block R-1 extra times before the canonical copy so the text region carries
// ~target_text_tokens of mass:
//   [ (SOT text EOT) x (R-1) | SOT text EOT SOS | audio ... ]
// Only the canonical (last) copy carries SOS (the "audio starts now" trigger).

import Foundation

public struct TextRepetitionConfig: Sendable, Equatable {
    /// Master switch. When false, `targetR` always returns 1 (legacy single-copy).
    public var enabled: Bool
    /// Repeat a short text until its region holds ~this many text tokens.
    public var targetTextTokens: Int
    /// Only texts with `nTextTokens < applyBelow` are repeated.
    public var applyBelow: Int
    /// Hard cap on R (bounds prefill blow-up on 1-token inputs).
    public var maxRepeats: Int

    public init(enabled: Bool = false, targetTextTokens: Int = 16,
                applyBelow: Int = 13, maxRepeats: Int = 8) {
        precondition(targetTextTokens >= 1 && applyBelow >= 1 && maxRepeats >= 1)
        self.enabled = enabled
        self.targetTextTokens = targetTextTokens
        self.applyBelow = applyBelow
        self.maxRepeats = maxRepeats
    }
}

public struct TextRepeater: Sendable {
    public let config: TextRepetitionConfig
    public let sot: Int   // start_of_text
    public let eot: Int   // end_of_text
    public let sos: Int   // start_of_speech

    public init(config: TextRepetitionConfig, startOfText: Int, endOfText: Int, startOfSpeech: Int) {
        self.config = config
        self.sot = startOfText
        self.eot = endOfText
        self.sos = startOfSpeech
    }

    /// Deterministic repeat count from the text-token count. Returns 1 when disabled,
    /// already long enough (`>= applyBelow`), or already meeting the budget.
    public func targetR(_ nTextTokens: Int) -> Int {
        let c = config
        if !c.enabled { return 1 }
        if nTextTokens <= 0 || nTextTokens >= c.applyBelow { return 1 }
        if nTextTokens >= c.targetTextTokens { return 1 }
        // ceil(target / n)
        let R = (c.targetTextTokens + nTextTokens - 1) / nTextTokens
        return max(1, min(R, c.maxRepeats))
    }

    /// Assemble the text-id sequence for a given R.
    ///   R == 1 -> [SOT, *text, EOT, SOS]                          (legacy)
    ///   R >  1 -> [SOT, *text, EOT] * (R-1) + [SOT, *text, EOT, SOS]
    public func buildInputIds(_ text: [Int], R: Int) -> [Int] {
        precondition(R >= 1, "R must be >= 1, got \(R)")
        let block = [sot] + text + [eot]
        let canonical = block + [sos]
        if R == 1 { return canonical }
        var out = [Int]()
        out.reserveCapacity(block.count * (R - 1) + canonical.count)
        for _ in 0..<(R - 1) { out.append(contentsOf: block) }
        out.append(contentsOf: canonical)
        return out
    }

    /// Convenience: pick the deterministic R then build the layout.
    public func expand(_ text: [Int]) -> [Int] {
        buildInputIds(text, R: targetR(text.count))
    }
}
