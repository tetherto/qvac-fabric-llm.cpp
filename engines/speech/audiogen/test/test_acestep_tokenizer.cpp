// Model-free ACE-Step tokenizer regression tests.

#include "acestep_test_support.h"
#include "acestep_test_tokenizer.h"
#include "lm_pipeline.h"

namespace {

void test_understand_prompt() {
    using tts_cpp::acestep::AUDIO_CODE_BASE;
    using tts_cpp::acestep::lm_understand_prompt;
    using tts_cpp::acestep::lm_understand_unconditional_prompt;
    using tts_cpp::acestep::TOKEN_IM_END;
    using tts_cpp::acestep::TOKEN_IM_START;

    const tts_cpp::acestep::BpeTokenizer tok = make_test_bpe_tokenizer();
    const int codes[3] = { 7, 0, 63999 };

    const std::vector<int> prompt = lm_understand_prompt(tok, codes, 3);
    CHECK(prompt.front() == TOKEN_IM_START);
    CHECK(std::count(prompt.begin(), prompt.end(), TOKEN_IM_START) == 3);
    CHECK(std::count(prompt.begin(), prompt.end(), TOKEN_IM_END) == 2);
    const auto first_code = std::find(prompt.begin(), prompt.end(), AUDIO_CODE_BASE + 7);
    CHECK(first_code != prompt.end());
    CHECK(*(first_code + 1) == AUDIO_CODE_BASE + 0);
    CHECK(*(first_code + 2) == AUDIO_CODE_BASE + 63999);
    CHECK(*(first_code + 3) == TOKEN_IM_END);

    const std::vector<int> unconditional = lm_understand_unconditional_prompt(tok);
    CHECK(unconditional.front() == TOKEN_IM_START);
    CHECK(std::count(unconditional.begin(), unconditional.end(), TOKEN_IM_START) == 3);
    for (int id : unconditional) CHECK(id < AUDIO_CODE_BASE);

    // Identical framing up to the user turn's content.
    CHECK(std::equal(prompt.begin(), first_code, unconditional.begin()));
}

// Understand decode budget: capped by the KV room left after the prompt, so a
// long clip's code prompt (the historical 4096 default vs the LM's 2048
// max_seq) can never overflow the context mid-decode.
void test_understand_token_budget() {
    using tts_cpp::acestep::lm_understand_token_budget;

    CHECK(lm_understand_token_budget(2048, 100, 0) == 1948);
    CHECK(lm_understand_token_budget(2048, 1220, 0) == 828);
    CHECK(lm_understand_token_budget(2048, 2047, 0) == 1);
    CHECK(lm_understand_token_budget(2048, 2048, 0) == 0);
    CHECK(lm_understand_token_budget(2048, 3000, 0) == 0);
    CHECK(lm_understand_token_budget(2048, 100, 4096) == 1948);
    CHECK(lm_understand_token_budget(2048, 100, 500) == 500);
}

void test_bpe_utf8_codepoint() {
    using tts_cpp::acestep::bpe_utf8_codepoint;

    int adv = 0;
    CHECK(bpe_utf8_codepoint("A", &adv) == 0x41 && adv == 1);
    CHECK(bpe_utf8_codepoint("~", &adv) == 0x7E && adv == 1);
    CHECK(bpe_utf8_codepoint("\xC3\xA9", &adv) == 0xE9 && adv == 2);      // é
    CHECK(bpe_utf8_codepoint("\xE4\xB8\xAD", &adv) == 0x4E2D && adv == 3);  // 中
    CHECK(bpe_utf8_codepoint("\xF0\x90\x90\x80", &adv) == 0x10400 && adv == 4);

    // Malformed lead bytes fall back to advance-by-1 with the raw byte value.
    const char lone_continuation[] = { (char) 0x80, 0, 0, 0, 0 };
    CHECK(bpe_utf8_codepoint(lone_continuation, &adv) == 0x80 && adv == 1);
    const char invalid_lead[] = { (char) 0xFF, 0, 0, 0, 0 };
    CHECK(bpe_utf8_codepoint(invalid_lead, &adv) == 0xFF && adv == 1);

    // A sequence that runs into the terminator decodes as its raw lead byte
    // rather than consuming the declared width and reading past the end.
    const char truncated_two[] = { (char) 0xC3, 0, 0, 0, 0 };
    CHECK(bpe_utf8_codepoint(truncated_two, &adv) == 0xC3 && adv == 1);
    const char truncated_three[] = { (char) 0xE4, 0, 0, 0, 0 };
    CHECK(bpe_utf8_codepoint(truncated_three, &adv) == 0xE4 && adv == 1);
    const char truncated_three_partial[] = { (char) 0xE4, (char) 0xB8, 0, 0, 0 };
    CHECK(bpe_utf8_codepoint(truncated_three_partial, &adv) == 0xE4 && adv == 1);
    const char truncated_four[] = { (char) 0xF0, (char) 0x90, (char) 0x90, 0, 0 };
    CHECK(bpe_utf8_codepoint(truncated_four, &adv) == 0xF0 && adv == 1);

    // A lead byte at the very end of a buffer must not be read past: this is
    // the case that motivated the clamp.
    const char lead_at_end[] = { (char) 0xF0, 0 };
    CHECK(bpe_utf8_codepoint(lead_at_end, &adv) == 0xF0 && adv == 1);
}

void test_bpe_encode_merges() {
    using tts_cpp::acestep::bpe_encode;

    const auto tok = make_test_bpe_tokenizer();

    // Full merge chain: h+e, l+l, he+ll, hell+o -> "hello".
    CHECK(bpe_encode(tok, "hello") == std::vector<int>({ 10 }));
    // Partial merge: only "h e" applies, the rest stays single symbols.
    CHECK(bpe_encode(tok, "helo") == std::vector<int>({ 8, 2, 3 }));
    // The GPT-2 pre-tokenizer folds the leading space into the word; the
    // byte-level 'Ġ' symbol then merges into a single vocab entry.
    CHECK(bpe_encode(tok, " hello") == std::vector<int>({ 11 }));
    CHECK(bpe_encode(tok, "hello hello") == std::vector<int>({ 10, 11 }));
    // Digits are pre-tokenized one at a time.
    CHECK(bpe_encode(tok, "12") == std::vector<int>({ 6, 7 }));

    CHECK(bpe_encode(tok, "").empty());
    CHECK(bpe_encode(tok, "", /*add_eos=*/true) == std::vector<int>({ tok.eos_id }));
    CHECK(bpe_encode(tok, "hello<|endoftext|>helo") == std::vector<int>({ 10, tok.eos_id, 8, 2, 3 }));
    CHECK(bpe_encode(tok, "<|endoftext|>") == std::vector<int>({ tok.eos_id }));
}

void test_bpe_encode_byte_fallback() {
    using tts_cpp::acestep::bpe_encode;

    const auto tok = make_test_bpe_tokenizer();

    // "a b" merges into "ab", which is not in the vocab: the encoder falls
    // back to the per-byte tokens.
    CHECK(bpe_encode(tok, "ab") == std::vector<int>({ 4, 5 }));
    // Bytes with no single-byte vocab entry are dropped silently.
    CHECK(bpe_encode(tok, "q").empty());
    CHECK(bpe_encode(tok, "abq") == std::vector<int>({ 4, 5 }));
    // A raw non-UTF-8 byte maps to a byte-level symbol outside the vocab whose
    // fallback bytes are unknown too, so it encodes to nothing.
    CHECK(bpe_encode(tok, "\x80").empty());
}

void test_bpe_decode_roundtrip() {
    using tts_cpp::acestep::AUDIO_CODE_BASE;
    using tts_cpp::acestep::TOKEN_IM_END;
    using tts_cpp::acestep::TOKEN_IM_START;
    using tts_cpp::acestep::TOKEN_THINK;
    using tts_cpp::acestep::TOKEN_THINK_END;
    using tts_cpp::acestep::bpe_decode;
    using tts_cpp::acestep::bpe_encode;
    using tts_cpp::acestep::bpe_utf8_codepoint;

    auto tok = make_test_bpe_tokenizer();

    // byte_dec is empty on the hand-built tokenizer, so this exercises the
    // local-fallback byte decoder.
    CHECK(bpe_decode(tok, bpe_encode(tok, "hello")) == "hello");
    CHECK(bpe_decode(tok, bpe_encode(tok, "helo")) == "helo");
    CHECK(bpe_decode(tok, bpe_encode(tok, " hello")) == " hello");
    CHECK(bpe_decode(tok, {}).empty());

    // eos decodes to nothing (outside id_to_str), as do negative and audio ids.
    CHECK(bpe_decode(tok, bpe_encode(tok, "", /*add_eos=*/true)).empty());
    CHECK(bpe_decode(tok, { -1, tok.n_vocab, AUDIO_CODE_BASE, AUDIO_CODE_BASE + 5, 10 }) == "hello");
    CHECK(bpe_decode(tok, { TOKEN_IM_START, 10, TOKEN_IM_END }) == "hello");
    CHECK(bpe_decode(tok, { TOKEN_THINK, 10, TOKEN_THINK_END }) == "<think>hello</think>");

    // Same results through the cached byte_dec branch bpe_load_from_gguf fills.
    for (int b = 0; b < 256; ++b) {
        int adv;
        tok.byte_dec[bpe_utf8_codepoint(tok.byte2str[b].c_str(), &adv)] = (uint8_t) b;
    }
    CHECK(bpe_decode(tok, bpe_encode(tok, " hello")) == " hello");
    CHECK(bpe_decode(tok, { TOKEN_THINK, 10, TOKEN_THINK_END }) == "<think>hello</think>");
}


}  // namespace

int main() {
    test_understand_prompt();
    test_understand_token_budget();
    test_bpe_utf8_codepoint();
    test_bpe_encode_merges();
    test_bpe_encode_byte_fallback();
    test_bpe_decode_roundtrip();
    std::fprintf(stderr, "[test-acestep-tokenizer] %d/%d checks passed\n", g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
