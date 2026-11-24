// Unit test for tts_cpp::supertonic::detail::split_for_streaming and
// is_sentence_end_cp (src/supertonic_chunker.cpp).
//
// Pure text->chunks policy; needs no model or fixture.  Pins the boundary
// priority (sentence > clause > whitespace > hard cut), the tolerance
// windows, the min-chunk floor, the first-chunk latency knob, the tail
// merge, and the reassembly invariant (chunks concatenated back together
// reproduce the input modulo trimmed/merge-inserted whitespace).

#include "supertonic_chunker.h"

#include <cstdio>
#include <string>
#include <vector>

using tts_cpp::supertonic::detail::split_for_streaming;
using tts_cpp::supertonic::detail::is_sentence_end_cp;

namespace {

int g_failures = 0;

#define CHECK(cond, msg)                                                          \
    do {                                                                          \
        if (!(cond)) {                                                            \
            ++g_failures;                                                         \
            std::fprintf(stderr, "FAIL: %s  (%s:%d)\n", msg, __FILE__, __LINE__); \
        }                                                                         \
    } while (0)

size_t count_cps(const std::string & s) {
    size_t n = 0;
    for (unsigned char c : s) {
        if ((c & 0xC0) != 0x80) ++n;
    }
    return n;
}

std::string strip_ws(const std::string & s) {
    std::string out;
    for (char c : s) {
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') out += c;
    }
    return out;
}

std::string join(const std::vector<std::string> & chunks) {
    std::string out;
    for (const auto & c : chunks) out += c;
    return out;
}

// Ordering + content invariant: the chunks, in order, carry exactly the
// input's non-whitespace bytes (trimming and the tail-merge separator only
// touch whitespace).
void check_reassembly(const std::string & input,
                      const std::vector<std::string> & chunks,
                      const char * what) {
    if (strip_ws(join(chunks)) != strip_ws(input)) {
        ++g_failures;
        std::fprintf(stderr, "FAIL: reassembly broken: %s\n", what);
    }
}

bool chunks_trimmed(const std::vector<std::string> & chunks) {
    for (const auto & c : chunks) {
        if (c.empty()) return false;
        if (c.front() == ' ' || c.back() == ' ') return false;
    }
    return true;
}

std::string repeat(const std::string & unit, int times) {
    std::string out;
    for (int i = 0; i < times; ++i) out += unit;
    return out;
}

std::string sentences(const std::string & sentence, int times) {
    std::string out;
    for (int i = 0; i < times; ++i) {
        if (i > 0) out += ' ';
        out += sentence;
    }
    return out;
}

void test_empty_and_whitespace_only_input() {
    CHECK(split_for_streaming("", 50).empty(), "empty input -> no chunks");
    CHECK(split_for_streaming("   ", 50).empty(), "whitespace-only input -> no chunks");
}

void test_non_positive_target_passes_through() {
    const std::string text = "Hello there.";
    auto out = split_for_streaming(text, 0);
    CHECK(out.size() == 1 && out[0] == text, "target 0 -> input unchanged");
    out = split_for_streaming(text, -7);
    CHECK(out.size() == 1 && out[0] == text, "negative target -> input unchanged");
}

void test_short_input_passes_through() {
    const std::string text = "Hello world, this is a short test.";
    const auto out = split_for_streaming(text, 50);
    CHECK(out.size() == 1, "short input -> single chunk");
    CHECK(!out.empty() && out[0] == text, "short input -> content unchanged");
}

// target 50, tolerance 20% -> upper bound 60: a boundary-free input of
// exactly 60 code points is one chunk, not a 60/0 split.
void test_input_exactly_at_the_chunk_limit() {
    const std::string text(60, 'a');
    const auto out = split_for_streaming(text, 50);
    CHECK(out.size() == 1 && out[0] == text, "input at the upper bound -> one chunk");
}

// No sentence / clause / whitespace boundary anywhere: hard cut at the
// upper tolerance bound, and the sub-threshold remainder stays a chunk
// (only a tail below max(6, target/3) is merged).
void test_no_split_point_hard_cuts_at_the_bound() {
    const std::string text(200, 'a');
    const auto out = split_for_streaming(text, 50);
    CHECK(out.size() == 4, "200 unbreakable cps at target 50 -> 4 chunks");
    if (out.size() == 4) {
        CHECK(count_cps(out[0]) == 60 && count_cps(out[1]) == 60 && count_cps(out[2]) == 60,
              "hard cut lands on the upper tolerance bound (60)");
        CHECK(count_cps(out[3]) == 20, "remainder stays as the last chunk");
        CHECK(join(out) == text, "hard cuts reassemble byte-exactly");
    }
}

void test_sentence_boundaries_win() {
    const std::string text = sentences("abcdefghijklmnopqr.", 4);  // 79 cps
    const auto out = split_for_streaming(text, 40);
    CHECK(out.size() == 2, "4 short sentences at target 40 -> 2 chunks");
    for (const auto & c : out) {
        CHECK(!c.empty() && c.back() == '.', "every chunk ends on a sentence terminator");
    }
    CHECK(chunks_trimmed(out), "sentence chunks carry no leading/trailing whitespace");
    check_reassembly(text, out, "sentence-boundary split");
}

void test_clause_fallback_when_no_sentence_in_reach() {
    const std::string text = std::string(40, 'a') + "," + std::string(30, 'b');  // 71 cps
    const auto out = split_for_streaming(text, 50);
    CHECK(out.size() == 2, "one clause boundary -> 2 chunks");
    if (out.size() == 2) {
        CHECK(out[0].back() == ',', "first chunk ends on the clause boundary");
        CHECK(out[1] == std::string(30, 'b'), "second chunk is the clause remainder");
    }
    check_reassembly(text, out, "clause-boundary split");
}

void test_whitespace_fallback_never_cuts_words() {
    std::string text = repeat("words ", 20);
    text.pop_back();  // 119 cps, no punctuation
    const auto out = split_for_streaming(text, 50);
    CHECK(out.size() == 2, "space-only boundaries at target 50 -> 2 chunks (tiny tail merged)");
    for (const auto & c : out) {
        CHECK(!c.empty() && c.front() == 'w' && c.back() == 's',
              "whitespace fallback splits between words, never inside one");
    }
    CHECK(chunks_trimmed(out), "whitespace chunks are trimmed");
    check_reassembly(text, out, "whitespace split");
}

void test_min_chunk_floor_is_respected() {
    const std::string text = sentences("Hi.", 10);  // 39 cps of 3-cp sentences
    const int min_chunk = 15;
    const auto out = split_for_streaming(text, 10, 0, 20, min_chunk);
    CHECK(out.size() == 2, "min-chunk floor groups tiny sentences");
    for (const auto & c : out) {
        CHECK(count_cps(c) >= (size_t) min_chunk, "no chunk below the min-chunk floor");
    }
    check_reassembly(text, out, "min-chunk floor");
}

void test_first_chunk_tokens_shrinks_only_the_first_chunk() {
    const std::string text = sentences("abcdefghijklmnopqr.", 8);  // 159 cps
    const auto out = split_for_streaming(text, 100, 30);
    CHECK(out.size() == 2, "first-chunk knob -> small opener + large remainder");
    if (out.size() == 2) {
        CHECK(count_cps(out[0]) == 39, "first chunk sized by first_chunk_tokens (2 sentences)");
        CHECK(out[0].back() == '.', "first chunk still ends on a sentence terminator");
        CHECK(count_cps(out[0]) < count_cps(out[1]), "later chunks keep the full target");
    }
    check_reassembly(text, out, "first-chunk knob");
}

void test_tiny_tail_is_merged_into_the_previous_chunk() {
    const std::string text(61, 'a');  // one cp past the bound -> 60 + 1, tail merged
    const auto out = split_for_streaming(text, 50);
    CHECK(out.size() == 1, "sub-threshold tail is folded into the previous chunk");
    check_reassembly(text, out, "tail merge");
}

void test_cjk_text_splits_at_ideographic_full_stop() {
    const std::string unit = repeat("\xE3\x81\x82", 11) + "\xE3\x80\x82";  // 11x HIRAGANA A + IDEOGRAPHIC FULL STOP
    const std::string text = repeat(unit, 5);  // 60 cps, no whitespace anywhere
    const auto out = split_for_streaming(text, 20);
    CHECK(out.size() == 2, "CJK text at target 20 -> 2 chunks");
    for (const auto & c : out) {
        CHECK(c.size() >= 3 && c.compare(c.size() - 3, 3, "\xE3\x80\x82") == 0,
              "CJK chunks end on the ideographic full stop");
    }
    CHECK(join(out) == text, "whitespace-free text reassembles byte-exactly");
}

void test_sentence_end_predicate() {
    CHECK(is_sentence_end_cp('.'), ". is a sentence end");
    CHECK(is_sentence_end_cp('?'), "? is a sentence end");
    CHECK(is_sentence_end_cp('!'), "! is a sentence end");
    CHECK(is_sentence_end_cp(0x3002), "ideographic full stop is a sentence end");
    CHECK(is_sentence_end_cp(0x0964), "Devanagari danda is a sentence end");
    CHECK(is_sentence_end_cp(0x06D4), "Urdu full stop is a sentence end");
    CHECK(!is_sentence_end_cp(','), ", is not a sentence end");
    CHECK(!is_sentence_end_cp(' '), "space is not a sentence end");
    CHECK(!is_sentence_end_cp('a'), "letter is not a sentence end");
    CHECK(!is_sentence_end_cp(0x2026), "ellipsis is intentionally not a sentence end");
}

}  // namespace

int main() {
    test_empty_and_whitespace_only_input();
    test_non_positive_target_passes_through();
    test_short_input_passes_through();
    test_input_exactly_at_the_chunk_limit();
    test_no_split_point_hard_cuts_at_the_bound();
    test_sentence_boundaries_win();
    test_clause_fallback_when_no_sentence_in_reach();
    test_whitespace_fallback_never_cuts_words();
    test_min_chunk_floor_is_respected();
    test_first_chunk_tokens_shrinks_only_the_first_chunk();
    test_tiny_tail_is_merged_into_the_previous_chunk();
    test_cjk_text_splits_at_ideographic_full_stop();
    test_sentence_end_predicate();

    if (g_failures) {
        std::fprintf(stderr, "test-supertonic-chunker: %d FAILURE(S)\n", g_failures);
        return 1;
    }
    std::printf("test-supertonic-chunker: all checks passed\n");
    return 0;
}
