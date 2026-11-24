// Unit test for tts_cpp::chatterbox::text_preprocess::split_text_for_tts.
//
// Pure text->text; needs no model or fixture, so it runs everywhere (CI
// included).  Pins the byte-oriented sentence-splitter behaviour that both
// the CLI (--max-sentence-chars) and the Engine (EngineOptions::
// max_sentence_chars) rely on after the splitter was lifted into the shared
// text_preprocess unit.

#include "text_preprocess.h"

#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

using tts_cpp::chatterbox::text_preprocess::split_text_for_tts;

static int g_failures = 0;
#define CHECK(cond, msg)                                                  \
    do {                                                                  \
        if (!(cond)) {                                                    \
            ++g_failures;                                                 \
            std::fprintf(stderr, "FAIL: %s  (%s:%d)\n", msg, __FILE__, __LINE__); \
        }                                                                 \
    } while (0)

static std::string strip_spaces(const std::string & s) {
    std::string out;
    for (char c : s) if (c != ' ' && c != '\t' && c != '\n' && c != '\r') out += c;
    return out;
}

int main() {
    struct Case {
        const char * text;
        int budget;
        std::vector<std::string> expected;
    };
    const Case cases[] = {
        {"", 180, {""}},
        {"One. Two. Three.", 0, {"One. Two. Three."}},
        {"One. Two. Three.", -5, {"One. Two. Three."}},
        {"Hi there. How are you?", 200, {"Hi there. How are you?"}},
        {"One. Two.", 9, {"One. Two."}},
        {"One. Two.", 8, {"One.", "Two."}},
        {"One! Two? Three.", 8, {"One!", "Two?", "Three."}},
        {"one two three", 8, {"one two", "three"}},
        {"abcde, fghij; klmno", 10, {"abcde,", "fghij;", "klmno"}},
        {"Hi.\tBye.\n", 8, {"Hi.", "Bye."}},
    };
    for (const auto & c : cases) {
        const auto got = split_text_for_tts(c.text, c.budget);
        CHECK(got == c.expected, "exact sentence boundaries and content");
        if (got != c.expected) {
            std::fprintf(stderr, "  input='%s' budget=%d\n", c.text, c.budget);
            for (const auto & segment : got) std::fprintf(stderr, "  got='%s'\n", segment.c_str());
        }
    }

    // 4) Multi-sentence text with a small budget -> multiple segments, each
    //    within budget and non-empty, content preserved (modulo whitespace).
    {
        const std::string t =
            "The morning was cold and grey. The travellers gathered their "
            "belongings slowly. They set out along the winding river path.";
        const int max_chars = 40;
        auto segs = split_text_for_tts(t, max_chars);
        CHECK(segs.size() > 1, "long text with small budget -> splits");
        for (const auto & s : segs) {
            CHECK(!s.empty(), "no empty segment");
            CHECK((int)s.size() <= max_chars, "segment within max_chars");
        }
        // Non-whitespace content is preserved across the split (the splitter
        // only strips trailing whitespace per segment, never drops content).
        std::string joined =
            std::accumulate(segs.begin(), segs.end(), std::string{});
        CHECK(strip_spaces(joined) == strip_spaces(t),
              "split preserves non-whitespace content");
    }

    // 5) A single over-long run with no sentence terminators still gets
    //    hard-broken to within budget (last-resort path).
    {
        std::string t(500, 'x');  // 500 chars, no spaces or terminators
        const int max_chars = 60;
        auto segs = split_text_for_tts(t, max_chars);
        CHECK(segs.size() > 1, "over-long unbroken run -> hard-broken");
        for (const auto & s : segs) {
            CHECK(!s.empty(), "hard-break produces no empty segments");
            CHECK((int)s.size() <= max_chars, "hard-break within max_chars");
        }
        CHECK(std::accumulate(segs.begin(), segs.end(), std::string{}) == t,
              "hard-break preserves every character");
    }

    if (g_failures == 0) {
        std::printf("test_text_split: OK\n");
        return 0;
    }
    std::fprintf(stderr, "test_text_split: %d failure(s)\n", g_failures);
    return 1;
}
