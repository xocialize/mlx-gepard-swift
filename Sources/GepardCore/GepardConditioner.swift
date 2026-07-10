// GepardConditioner — text → cond_ids, the last functional gap in the engine run().
//
// Isomorphic to gepard-inference's `runner.tokenizer.encode(text, add_special_tokens=False)`
// then `runner.repeater.expand(ids)`:
//   ids       = Qwen3.5 byte-level BPE (tokenizer.json), NO special tokens auto-added
//   cond_ids  = TextRepeater layout  [(SOT text EOT)×(R−1) | SOT text EOT SOS | audio…]
//
// The tokenizer is loaded faithfully via swift-transformers' PreTrainedTokenizer (the same
// engine that HuggingFace `transformers` uses), so the byte-level pre-tokenizer, BPE merges,
// and byte-encoding all match the oracle exactly — P10 asserts integer-exact cond_ids vs the
// Stage-0 goldens (a naive whitespace-split BPE fails on contractions / punctuation).
//
// Special ids are FIXED by gepard_config.json (never inferred from the tokenizer):
//   SOT (start_of_text)   248073
//   EOT (end_of_text)     248074
//   SOS (start_of_speech) 248070   (== BOS_AUDIO, the "audio starts now" trigger)

import Foundation
import Tokenizers

public struct GepardConditioner: Sendable {
    /// gepard_config.json special_tokens (verified).
    public static let sot = 248073
    public static let eot = 248074
    public static let sos = 248070

    /// gepard_config.json text_repetition (verified): enabled at inference.
    public static let textRepetition = TextRepetitionConfig(
        enabled: true, targetTextTokens: 16, applyBelow: 13, maxRepeats: 8)

    private let tokenizer: any Tokenizer
    public let repeater: TextRepeater

    public init(tokenizer: any Tokenizer,
                repeater: TextRepeater = TextRepeater(
                    config: GepardConditioner.textRepetition,
                    startOfText: GepardConditioner.sot,
                    endOfText: GepardConditioner.eot,
                    startOfSpeech: GepardConditioner.sos)) {
        self.tokenizer = tokenizer
        self.repeater = repeater
    }

    /// Load the tokenizer from a local model folder (expects `tokenizer.json` +
    /// `tokenizer_config.json`, as materialized by the "main" weight source). `strict: false`
    /// tolerates Gepard's custom `tokenizer_class: "TokenizersBackend"` (from(modelFolder:)
    /// builds a PreTrainedTokenizer directly off tokenizer.json, so class dispatch is bypassed).
    public static func load(modelFolder: URL) async throws -> GepardConditioner {
        let tokenizer = try await AutoTokenizer.from(modelFolder: modelFolder, strict: false)
        return GepardConditioner(tokenizer: tokenizer)
    }

    /// Raw text token ids — `encode(add_special_tokens=False)`.
    public func encode(_ text: String) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: false)
    }

    /// text → cond_ids (ids + text-repetition framing). This is what prefill consumes.
    public func condIds(for text: String) -> [Int] {
        repeater.expand(encode(text))
    }
}
