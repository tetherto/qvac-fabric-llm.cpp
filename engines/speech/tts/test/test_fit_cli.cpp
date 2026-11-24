// Model-free unit tests for the memory-fit CLI helpers and exit-code
// contract: the strict numeric parsers and saturating conversions in
// src/fit_util.h, JSON escaping of untrusted metadata strings, and the
// audio8-fit-params / chatterbox-fit-params argument surfaces (a parse error
// or unreadable model must exit with FitStatus::Error == 2, never coerce).

#include "fit_util.h"
#include "tts-cpp/audio8/fit.h"
#include "tts-cpp/chatterbox/fit.h"
#include "tts-cpp/cosyvoice/fit.h"
#include "tts-cpp/parler/fit.h"
#include "tts-cpp/supertonic/fit.h"

#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void fail(const std::string & what) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
}

void expect(bool cond, const std::string & what) {
    if (!cond) fail(what);
}

using namespace tts_cpp::fitutil;

void test_sat_math() {
    const uint64_t maxv = std::numeric_limits<uint64_t>::max();
    expect(sat_add(1, 2) == 3, "sat_add small");
    expect(sat_add(maxv, 1) == maxv, "sat_add saturates");
    expect(sat_mul(3, 4) == 12, "sat_mul small");
    expect(sat_mul(maxv / 2, 3) == maxv, "sat_mul saturates");
    expect(sat_mul(0, maxv) == 0, "sat_mul zero");
    expect(sat_u64_from_double(-1.0) == 0, "negative reads as zero");
    expect(sat_u64_from_double(1e30) == maxv, "huge double saturates");
    expect(sat_u64_from_double(std::numeric_limits<double>::quiet_NaN()) == maxv,
           "NaN saturates (unrepresentable must read as does-not-fit)");
    expect(sat_u64_from_double(std::numeric_limits<double>::infinity()) == maxv,
           "inf saturates");
    expect(margin_mib_to_bytes(1) == 1024ull * 1024, "margin MiB -> bytes");
    expect(margin_mib_to_bytes(maxv) == maxv, "absurd margin saturates (stricter)");
}

void test_strict_parsers() {
    float f = -1.0f;
    expect(parse_f32_positive("2.5", f) && f == 2.5f, "parse_f32_positive accepts 2.5");
    expect(!parse_f32_positive("0", f),    "zero is not positive");
    expect(!parse_f32_positive("-3", f),   "negative rejected");
    expect(!parse_f32_positive("abc", f),  "junk rejected");
    expect(!parse_f32_positive("1.5x", f), "trailing junk rejected");
    expect(!parse_f32_positive("inf", f),  "inf rejected");
    expect(!parse_f32_positive("nan", f),  "nan rejected");
    expect(!parse_f32_positive(nullptr, f), "null rejected");

    int i = -1;
    expect(parse_i32("42", i) && i == 42,   "parse_i32 accepts 42");
    expect(parse_i32("-7", i) && i == -7,   "parse_i32 accepts negatives");
    expect(!parse_i32("1x", i),             "parse_i32 rejects trailing junk");
    expect(!parse_i32("", i),               "parse_i32 rejects empty");
    expect(!parse_i32("99999999999999", i), "parse_i32 rejects overflow");

    // ERANGE coverage: strtol/strtoull clamp instead of failing, and on
    // LLP64 (Windows) long is 32-bit, so a pure range comparison passes
    // exactly-clamped values. These must be rejected on EVERY platform.
    expect(!parse_i32("3000000000", i),
           "parse_i32 rejects a value that clamps to LONG_MAX on LLP64");
    expect(!parse_i32("-3000000000", i),
           "parse_i32 rejects a value that clamps to LONG_MIN on LLP64");

    uint64_t u = 1;
    expect(parse_u64("0", u) && u == 0,     "parse_u64 accepts 0");
    expect(parse_u64("123456789012", u) && u == 123456789012ull, "parse_u64 big value");
    expect(!parse_u64("-1", u),             "parse_u64 rejects negatives");
    expect(!parse_u64("12a", u),            "parse_u64 rejects trailing junk");
    expect(!parse_u64("99999999999999999999", u),
           "parse_u64 rejects an out-of-range value (ERANGE clamp)");
}

void test_json_escape() {
    expect(json_escape("plain") == "plain", "plain passthrough");
    expect(json_escape("a\"b") == "a\\\"b", "quote escaped");
    expect(json_escape("a\\b") == "a\\\\b", "backslash escaped");
    expect(json_escape("a\nb") == "a\\nb",  "newline escaped");
    expect(json_escape(std::string(1, '\x01')) == "\\u0001", "control chars escaped");
    expect(json_escape("naïve") == "naïve", "UTF-8 bytes pass through");

    // Malformed UTF-8 (GGUF metadata / driver strings are untrusted) must
    // come out as U+FFFD, never as raw bytes a strict JSON parser rejects.
    const std::string fffd = "\xEF\xBF\xBD";
    expect(json_escape("a\xffz") == "a" + fffd + "z",
           "json_escape replaces an invalid byte");
    expect(json_escape("caf\xc3") == "caf" + fffd,
           "json_escape replaces a truncated tail sequence");
    expect(json_escape("\xc0\xaf") == fffd + fffd,
           "json_escape replaces an overlong encoding byte-by-byte");
    expect(json_escape("\xed\xa0\x80") == fffd + fffd + fffd,
           "json_escape replaces a surrogate encoding");
    expect(json_escape("\xf0\x9f\x8e\xb5") == "\xf0\x9f\x8e\xb5",
           "json_escape passes a valid 4-byte sequence through");
}

// Both CLI mains live in the library; drive them with crafted argv and check
// the exit-code contract (0 fits / 1 does not fit / 2 error).  No model on
// disk, so every model-touching invocation must be 2 -- and a malformed
// numeric flag must be 2 WITHOUT touching the model at all.
int run_cli(int (*main_fn)(int, char **), std::vector<const char *> args) {
    std::vector<char *> argv;
    argv.reserve(args.size());
    for (const char * a : args) argv.push_back(const_cast<char *>(a));
    return main_fn((int) argv.size(), argv.data());
}

void test_audio8_cli_contract() {
    expect(run_cli(audio8_fit_cli_main, {"audio8-fit-params", "--help"}) == 0,
           "audio8 --help exits 0");
    expect(run_cli(audio8_fit_cli_main, {"audio8-fit-params"}) == 2,
           "audio8 missing required flags exits 2");
    expect(run_cli(audio8_fit_cli_main,
                   {"audio8-fit-params", "--lm", "x.gguf"}) == 2,
           "audio8 missing decoder exits 2");
    expect(run_cli(audio8_fit_cli_main,
                   {"audio8-fit-params", "--lm", "a", "--codec-decoder", "b",
                    "--prompt-tokens", "12x"}) == 2,
           "audio8 junk --prompt-tokens exits 2 (never coerced)");
    expect(run_cli(audio8_fit_cli_main,
                   {"audio8-fit-params", "--lm", "a", "--codec-decoder", "b",
                    "--reference-seconds", "-2"}) == 2,
           "audio8 negative --reference-seconds exits 2");
    expect(run_cli(audio8_fit_cli_main,
                   {"audio8-fit-params", "--lm", "a", "--codec-decoder", "b",
                    "--margin-mib", "abc"}) == 2,
           "audio8 junk --margin-mib exits 2");
    expect(run_cli(audio8_fit_cli_main,
                   {"audio8-fit-params", "--not-a-flag"}) == 2,
           "audio8 unknown flag exits 2");
    expect(run_cli(audio8_fit_cli_main,
                   {"audio8-fit-params", "--lm", "/nonexistent-fit-test.gguf",
                    "--codec-decoder", "/nonexistent-fit-test-2.gguf", "--json"}) == 2,
           "audio8 unreadable model exits 2");
}

void test_chatterbox_cli_contract() {
    expect(run_cli(chatterbox_fit_cli_main, {"chatterbox-fit-params", "--help"}) == 0,
           "chatterbox --help exits 0");
    expect(run_cli(chatterbox_fit_cli_main, {"chatterbox-fit-params"}) == 2,
           "chatterbox missing required flags exits 2");
    expect(run_cli(chatterbox_fit_cli_main,
                   {"chatterbox-fit-params", "--t3", "a", "--s3gen", "b",
                    "--text-tokens", "0"}) == 2,
           "chatterbox non-positive --text-tokens exits 2");
    expect(run_cli(chatterbox_fit_cli_main,
                   {"chatterbox-fit-params", "--t3", "a", "--s3gen", "b",
                    "--n-predict", "1e3"}) == 2,
           "chatterbox junk --n-predict exits 2 (never coerced)");
    expect(run_cli(chatterbox_fit_cli_main,
                   {"chatterbox-fit-params", "--t3", "a", "--s3gen", "b",
                    "--n-ctx", "-5"}) == 2,
           "chatterbox negative --n-ctx exits 2");
    expect(run_cli(chatterbox_fit_cli_main,
                   {"chatterbox-fit-params", "--t3", "/nonexistent-fit-test.gguf",
                    "--s3gen", "/nonexistent-fit-test-2.gguf", "--json"}) == 2,
           "chatterbox unreadable model exits 2");
}

void test_parler_cli_contract() {
    expect(run_cli(parler_fit_cli_main, {"parler-fit-params", "--help"}) == 0,
           "parler --help exits 0");
    expect(run_cli(parler_fit_cli_main, {"parler-fit-params"}) == 2,
           "parler missing required flags exits 2");
    expect(run_cli(parler_fit_cli_main,
                   {"parler-fit-params", "--model", "a",
                    "--description-tokens", "0"}) == 2,
           "parler non-positive --description-tokens exits 2");
    expect(run_cli(parler_fit_cli_main,
                   {"parler-fit-params", "--model", "a",
                    "--prompt-tokens", "12x"}) == 2,
           "parler junk --prompt-tokens exits 2 (never coerced)");
    expect(run_cli(parler_fit_cli_main,
                   {"parler-fit-params", "--model", "a",
                    "--max-frames", "-3"}) == 2,
           "parler negative --max-frames exits 2");
    expect(run_cli(parler_fit_cli_main,
                   {"parler-fit-params", "--model", "a",
                    "--margin-mib", "abc"}) == 2,
           "parler junk --margin-mib exits 2");
    expect(run_cli(parler_fit_cli_main, {"parler-fit-params", "--not-a-flag"}) == 2,
           "parler unknown flag exits 2");
    expect(run_cli(parler_fit_cli_main,
                   {"parler-fit-params", "--model", "/nonexistent-fit-test.gguf",
                    "--json"}) == 2,
           "parler unreadable model exits 2");
}

void test_cosyvoice_cli_contract() {
    expect(run_cli(cosyvoice_fit_cli_main, {"cosyvoice-fit-params", "--help"}) == 0,
           "cosyvoice --help exits 0");
    expect(run_cli(cosyvoice_fit_cli_main, {"cosyvoice-fit-params"}) == 2,
           "cosyvoice missing required flags exits 2");
    expect(run_cli(cosyvoice_fit_cli_main,
                   {"cosyvoice-fit-params", "--llm", "a", "--flow", "b",
                    "--hift", "c"}) == 2,
           "cosyvoice missing --voice exits 2");
    expect(run_cli(cosyvoice_fit_cli_main,
                   {"cosyvoice-fit-params", "--llm", "a", "--flow", "b",
                    "--hift", "c", "--voice", "d", "--text-tokens", "0"}) == 2,
           "cosyvoice non-positive --text-tokens exits 2");
    expect(run_cli(cosyvoice_fit_cli_main,
                   {"cosyvoice-fit-params", "--llm", "a", "--flow", "b",
                    "--hift", "c", "--voice", "d", "--speech-tokens", "1e3"}) == 2,
           "cosyvoice junk --speech-tokens exits 2 (never coerced)");
    expect(run_cli(cosyvoice_fit_cli_main,
                   {"cosyvoice-fit-params", "--llm", "a", "--flow", "b",
                    "--hift", "c", "--voice", "d", "--margin-mib", "abc"}) == 2,
           "cosyvoice junk --margin-mib exits 2");
    expect(run_cli(cosyvoice_fit_cli_main, {"cosyvoice-fit-params", "--not-a-flag"}) == 2,
           "cosyvoice unknown flag exits 2");
    expect(run_cli(cosyvoice_fit_cli_main,
                   {"cosyvoice-fit-params", "--llm", "/nonexistent-fit-1.gguf",
                    "--flow", "/nonexistent-fit-2.gguf", "--hift", "/nonexistent-fit-3.gguf",
                    "--voice", "/nonexistent-fit-4.gguf", "--json"}) == 2,
           "cosyvoice unreadable models exit 2");
}

void test_supertonic_cli_contract() {
    expect(run_cli(supertonic_fit_cli_main, {"supertonic-fit-params", "--help"}) == 0,
           "supertonic --help exits 0");
    expect(run_cli(supertonic_fit_cli_main, {"supertonic-fit-params"}) == 2,
           "supertonic missing required flags exits 2");
    expect(run_cli(supertonic_fit_cli_main,
                   {"supertonic-fit-params", "--model", "a", "--text-tokens", "0"}) == 2,
           "supertonic non-positive --text-tokens exits 2");
    expect(run_cli(supertonic_fit_cli_main,
                   {"supertonic-fit-params", "--model", "a", "--audio-seconds", "-1"}) == 2,
           "supertonic negative --audio-seconds exits 2");
    expect(run_cli(supertonic_fit_cli_main,
                   {"supertonic-fit-params", "--model", "a", "--precision", "int4"}) == 2,
           "supertonic junk --precision exits 2");
    expect(run_cli(supertonic_fit_cli_main,
                   {"supertonic-fit-params", "--model", "a", "--steps", "1x"}) == 2,
           "supertonic junk --steps exits 2 (never coerced)");
    expect(run_cli(supertonic_fit_cli_main,
                   {"supertonic-fit-params", "--model", "a", "--margin-mib", "abc"}) == 2,
           "supertonic junk --margin-mib exits 2");
    expect(run_cli(supertonic_fit_cli_main, {"supertonic-fit-params", "--not-a-flag"}) == 2,
           "supertonic unknown flag exits 2");
    expect(run_cli(supertonic_fit_cli_main,
                   {"supertonic-fit-params", "--model", "/nonexistent-fit-test.gguf",
                    "--json"}) == 2,
           "supertonic unreadable model exits 2");
}

}  // namespace

int main() {
    test_sat_math();
    test_strict_parsers();
    test_json_escape();
    test_audio8_cli_contract();
    test_chatterbox_cli_contract();
    test_parler_cli_contract();
    test_cosyvoice_cli_contract();
    test_supertonic_cli_contract();

    if (g_failures == 0) {
        std::printf("test-fit-cli: all checks passed\n");
    }
    return g_failures;
}
