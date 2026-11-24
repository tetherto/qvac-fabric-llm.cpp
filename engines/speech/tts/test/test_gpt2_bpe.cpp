#include "gpt2_bpe.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

constexpr int32_t k_added_token_base = 50257;

gpt2_bpe make_tokenizer(bool with_added_token) {
    std::vector<std::string> tokens = {
        "h", "e", "l", "o", "\xC4\xA0",
        "he", "hel", "hell", "hello", "\xC4\xA0hello",
        "!", ".",
    };
    if (with_added_token) {
        const size_t base = tokens.size();
        tokens.resize(k_added_token_base + 1);
        for (size_t i = base; i < tokens.size(); ++i) {
            tokens[i] = "<unused" + std::to_string(i) + ">";
        }
        tokens[k_added_token_base] = "[laugh]";
    }
    const std::vector<std::string> merges = {
        "h e", "he l", "hel l", "hell o", "\xC4\xA0 hello",
    };
    gpt2_bpe bpe;
    bpe.load_from_arrays(tokens, merges);
    return bpe;
}

bool expect_ids(const gpt2_bpe & bpe, const std::string & text,
                const std::vector<int32_t> & expected,
                const char * scenario) {
    const auto got = bpe.tokenize(text);
    if (got == expected) return true;
    std::fprintf(stderr, "%s: got [", scenario);
    for (auto id : got) std::fprintf(stderr, " %d", id);
    std::fprintf(stderr, " ], expected [");
    for (auto id : expected) std::fprintf(stderr, " %d", id);
    std::fprintf(stderr, " ]\n");
    return false;
}

bool expect_norm(const std::string & in, const std::string & expected,
                 const char * scenario) {
    const std::string got = gpt2_bpe::punc_norm(in);
    if (got == expected) return true;
    std::fprintf(stderr, "%s: got \"%s\", expected \"%s\"\n", scenario,
                 got.c_str(), expected.c_str());
    return false;
}

bool check_load_rejects_empty_vocab() {
    gpt2_bpe bpe;
    return !bpe.load_from_arrays({}, {});
}

bool check_full_merge_chain() {
    const auto bpe = make_tokenizer(false);
    return expect_ids(bpe, "hello", {8}, "full merge");
}

bool check_partial_merge_stops_at_known_rank() {
    const auto bpe = make_tokenizer(false);
    return expect_ids(bpe, "hell", {7}, "partial merge");
}

bool check_leading_space_folds_into_word_token() {
    const auto bpe = make_tokenizer(false);
    return expect_ids(bpe, "hello hello", {8, 9}, "space fold");
}

bool check_punctuation_splits_from_word() {
    const auto bpe = make_tokenizer(false);
    return expect_ids(bpe, "hello!", {8, 10}, "punctuation split");
}

bool check_unknown_word_without_byte_tokens_is_dropped() {
    const auto bpe = make_tokenizer(false);
    return expect_ids(bpe, "z", {}, "unknown dropped");
}

bool check_empty_input() {
    const auto bpe = make_tokenizer(false);
    return expect_ids(bpe, "", {}, "empty input");
}

bool check_added_token_span() {
    const auto bpe = make_tokenizer(true);
    return expect_ids(bpe, "hello [laugh] hello",
                      {8, 4, k_added_token_base, 9}, "added token span");
}

bool check_punc_norm() {
    return expect_norm("", "You need to add some text for me to talk.",
                       "empty text") &&
           expect_norm("hello world", "Hello world.", "capitalize + period") &&
           expect_norm("a  b", "A b.", "space collapse") &&
           expect_norm("wait: no", "Wait, no.", "colon to comma") &&
           expect_norm("done!", "Done!", "existing punctuation kept") &&
           expect_norm("end \n", "End.", "trailing whitespace stripped") &&
           expect_norm("a \xE2\x80\x94 b", "A - b.", "em dash") &&
           expect_norm("said \xE2\x80\x9Chi\xE2\x80\x9D",
                       "Said \"hi\".", "curly quotes");
}

}

int main() {
    int failures = 0;

    failures += check_load_rejects_empty_vocab() ? 0 : 1;
    failures += check_full_merge_chain() ? 0 : 1;
    failures += check_partial_merge_stops_at_known_rank() ? 0 : 1;
    failures += check_leading_space_folds_into_word_token() ? 0 : 1;
    failures += check_punctuation_splits_from_word() ? 0 : 1;
    failures += check_unknown_word_without_byte_tokens_is_dropped() ? 0 : 1;
    failures += check_empty_input() ? 0 : 1;
    failures += check_added_token_span() ? 0 : 1;
    failures += check_punc_norm() ? 0 : 1;

    if (failures > 0) {
        std::fprintf(stderr, "test_gpt2_bpe: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_gpt2_bpe: all checks passed\n");
    return 0;
}
