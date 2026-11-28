// Model-free unit tests for the acestep-fit-params CLI layer
// (src/acestep/fit_util.h + acestep_fit_cli_main argument handling). These
// cover the pieces the fixture-gated parity test (test_fit_params.cpp) cannot:
// the strict numeric parsers, the saturating arithmetic the verdict depends
// on, the JSON string escaper the @qvac/model-fit SDK will JSON.parse, and the
// CLI's exit-code contract for garbage input (exit 2 = Error, never a
// coerced-to-0 flag).
//
// Exit 0 on success; non-zero (with a FAIL line per broken invariant) otherwise.

#include "audiogen-cpp/acestep/fit.h"

#include "fit_util.h"

#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#ifdef _WIN32
#include <io.h>
#define dup _dup
#define dup2 _dup2
#define close _close
#define fileno _fileno
#else
#include <unistd.h>
#endif

extern "C" int acestep_fit_cli_main(int argc, char ** argv);

using namespace tts_cpp::acestep::fitutil;

namespace {

int g_failures = 0;

void fail(const std::string & what) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
}

void expect(bool cond, const std::string & what) {
    if (!cond) fail(what);
}

// argv-level driver: acestep_fit_cli_main takes mutable char**.
int run_cli(std::vector<std::string> args) {
    args.insert(args.begin(), "acestep-fit-params");
    std::vector<char *> argv;
    argv.reserve(args.size());
    for (std::string & a : args) argv.push_back(a.data());
    return acestep_fit_cli_main((int) argv.size(), argv.data());
}

void expect_cli_error(const std::vector<std::string> & args, const char * diagnostic,
                      bool argument_error = true) {
    FILE * captured = std::tmpfile();
    expect(captured != nullptr, "create stderr capture");
    if (!captured) return;
    std::fflush(stderr);
    const int saved = dup(fileno(stderr));
    if (saved < 0 || dup2(fileno(captured), fileno(stderr)) < 0) {
        fail("redirect stderr");
        if (saved >= 0) close(saved);
        std::fclose(captured);
        return;
    }
    const int status = run_cli(args);
    std::fflush(stderr);
    dup2(saved, fileno(stderr));
    close(saved);
    std::rewind(captured);
    std::string output;
    constexpr size_t kCaptureBufferBytes = 512;
    char buffer[kCaptureBufferBytes];
    while (std::fgets(buffer, sizeof(buffer), captured)) output += buffer;
    std::fclose(captured);
    expect(status == static_cast<int>(tts_cpp::acestep::FitStatus::Error), "CLI exits Error");
    expect(output.find(diagnostic) != std::string::npos,
           std::string("expected argument diagnostic: ") + diagnostic + "; got: " + output);
    if (argument_error) {
        expect(output.find("acestep-fit-params: error:") == std::string::npos,
               "invalid argument reached model preflight");
    }
}

constexpr uint64_t U64_MAX = std::numeric_limits<uint64_t>::max();

}  // namespace

int main() {
    // ── Strict parsers: garbage fails, never coerces ────────────────────────
    {
        int i = 7;
        expect(parse_i32("42", i) && i == 42,   "parse_i32('42')");
        expect(parse_i32("-3", i) && i == -3,   "parse_i32('-3')");
        expect(!parse_i32("1x", i),             "parse_i32 accepted trailing garbage '1x'");
        expect(!parse_i32("", i),               "parse_i32 accepted empty string");
        expect(!parse_i32("99999999999999", i), "parse_i32 accepted out-of-range value");

        // ERANGE coverage: strtol/strtoull clamp instead of failing, and on
        // LLP64 (Windows) long is 32-bit, so a pure range comparison passes
        // exactly-clamped values. These must be rejected on EVERY platform.
        expect(!parse_i32("3000000000", i),
               "parse_i32 accepted a value that clamps to LONG_MAX on LLP64");
        expect(!parse_i32("-3000000000", i),
               "parse_i32 accepted a value that clamps to LONG_MIN on LLP64");

        uint64_t u = 7;
        expect(parse_u64("0", u) && u == 0,     "parse_u64('0')");
        expect(parse_u64("18446744073709551615", u) && u == U64_MAX,
               "parse_u64(UINT64_MAX)");
        expect(!parse_u64("-1", u),             "parse_u64 accepted '-1'");
        expect(!parse_u64("12abc", u),          "parse_u64 accepted trailing garbage");
        expect(!parse_u64("99999999999999999999", u),
               "parse_u64 accepted an out-of-range value (ERANGE clamp)");

        float f = 7.0f;
        expect(parse_f32_positive("2.5", f) && f == 2.5f, "parse_f32_positive('2.5')");
        expect(!parse_f32_positive("abc", f),  "parse_f32_positive accepted 'abc'");
        expect(!parse_f32_positive("0", f),    "parse_f32_positive accepted '0'");
        expect(!parse_f32_positive("-5", f),   "parse_f32_positive accepted '-5'");
        expect(!parse_f32_positive("inf", f),  "parse_f32_positive accepted 'inf'");
        expect(!parse_f32_positive("1e9999", f),
               "parse_f32_positive accepted an overflowing literal");
        expect(!parse_f32_positive("10x", f),  "parse_f32_positive accepted trailing garbage");

        expect(parse_f32_nonnegative("0", f) && f == 0.0f, "parse_f32_nonnegative('0')");
        expect(parse_f32_nonnegative("7.5", f) && f == 7.5f, "parse_f32_nonnegative('7.5')");
        expect(!parse_f32_nonnegative("-0.5", f), "parse_f32_nonnegative accepted '-0.5'");
        expect(!parse_f32_nonnegative("nan", f),  "parse_f32_nonnegative accepted 'nan'");
    }

    // ── Saturating arithmetic: overflow must widen, never wrap ─────────────
    {
        expect(sat_add(1, 2) == 3,                       "sat_add(1,2)");
        expect(sat_add(U64_MAX, 1) == U64_MAX,           "sat_add saturates");
        expect(sat_add(U64_MAX, U64_MAX) == U64_MAX,     "sat_add(max,max) saturates");
        expect(sat_mul(3, 4) == 12,                      "sat_mul(3,4)");
        expect(sat_mul(0, U64_MAX) == 0,                 "sat_mul(0,max)");
        expect(sat_mul(U64_MAX, 2) == U64_MAX,           "sat_mul saturates");
        expect(sat_mul(1ull << 33, 1ull << 33) == U64_MAX, "sat_mul(2^33,2^33) saturates");

        expect(sat_u64_from_double(-1.0) == 0,           "sat_u64_from_double(-1)");
        expect(sat_u64_from_double(0.0) == 0,            "sat_u64_from_double(0)");
        expect(sat_u64_from_double(1.5) == 1,            "sat_u64_from_double(1.5)");
        expect(sat_u64_from_double(1e40) == U64_MAX,     "sat_u64_from_double(1e40) saturates");
        expect(sat_u64_from_double(std::numeric_limits<double>::infinity()) == U64_MAX,
               "sat_u64_from_double(inf) saturates");
        expect(sat_u64_from_double(std::numeric_limits<double>::quiet_NaN()) == U64_MAX,
               "sat_u64_from_double(nan) saturates (strict direction)");

        expect(margin_mib_to_bytes(256) == 256ull * 1024 * 1024, "margin_mib_to_bytes(256)");
        expect(margin_mib_to_bytes(U64_MAX) == U64_MAX,          "margin saturates at max");
        expect(margin_mib_to_bytes(U64_MAX / (1024 * 1024) + 1) == U64_MAX,
               "margin saturates just past the clamp");
    }

    // ── JSON escaping: what @qvac/model-fit will JSON.parse ────────────────
    {
        expect(json_escape("plain") == "plain",              "json_escape passthrough");
        expect(json_escape("a\"b") == "a\\\"b",              "json_escape quote");
        expect(json_escape("a\\b") == "a\\\\b",              "json_escape backslash");
        expect(json_escape("a\nb\tc") == "a\\nb\\tc",        "json_escape newline/tab");
        expect(json_escape(std::string("a\x01") + "b") == "a\\u0001b",
               "json_escape control char");
        expect(json_escape("caf\xc3\xa9") == "caf\xc3\xa9",  "json_escape UTF-8 passthrough");
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

    // ── CLI exit-code contract for garbage input (all fail before any model
    //    or backend work; the missing-model path is covered by the
    //    fixture-gated test_fit_params.cpp) ───────────────────────────────
    {
        expect(run_cli({"--help"}) == 0,                            "--help exits 0");
        expect_cli_error({}, "--models-dir (or all four stage paths) is required");
        expect_cli_error({"--bogus-flag"}, "unknown or incomplete argument: --bogus-flag");
        expect_cli_error({"--models-dir"}, "unknown or incomplete argument: --models-dir");
        expect_cli_error({"--models-dir", "x", "--duration", "abc"}, "--duration:");
        expect_cli_error({"--models-dir", "x", "--duration", "0"}, "--duration:");
        expect_cli_error({"--models-dir", "x", "--n-gpu-layers", "1x"}, "--n-gpu-layers:");
        expect_cli_error({"--models-dir", "x", "--threads", "two"}, "--threads:");
        expect_cli_error({"--models-dir", "x", "--margin-mib", "-1"}, "--margin-mib:");
        expect_cli_error({"--models-dir", "x", "--text-tokens", "1.5"}, "--text-tokens:");
        expect_cli_error({"--models-dir", "x", "--lyric-tokens", "x"}, "--lyric-tokens:");
        expect_cli_error({"--models-dir", "x", "--lm-cfg", "-2"}, "--lm-cfg:");
        expect_cli_error({"--models-dir", "x", "--guidance", "no"}, "--guidance:");
        expect_cli_error({"--models-dir", "x", "--keep-stages", "2"}, "--keep-stages:");
        expect_cli_error({"--models-dir", "x", "--lm-max-new-tokens", "many"}, "--lm-max-new-tokens:");
        expect_cli_error({"--models-dir", "x", "--lm-prompt-tokens", "1e3"}, "--lm-prompt-tokens:");
        expect_cli_error({"--dit", "x.gguf"}, "--models-dir (or all four stage paths) is required");
    }

    // ── Library-level argument validation (no models touched) ──────────────
    {
        tts_cpp::acestep::FitOptions o;  // no paths at all
        const tts_cpp::acestep::FitResult res = tts_cpp::acestep::fit_params(o);
        expect(res.status == tts_cpp::acestep::FitStatus::Error,
               "fit_params without models was not Error");
        expect(res.reason == "invalid-arguments",
               "fit_params without models reason was '" + res.reason + "'");
        expect(!res.fits, "fit_params without models reported fits");
    }

    // ── Near-INT_MAX token counts: the token-derived shapes (enc_S, the LM
    //    prompt) must be refused as workload-too-large BEFORE any model or
    //    backend work -- summing them in int is UB and previously handed the
    //    graph builders a negative dimension. The guard is model-independent,
    //    so fake paths never get opened. ─────────────────────────────────────
    {

        tts_cpp::acestep::FitOptions o;
        o.text_enc_model_path = "x.gguf";
        o.lm_model_path       = "x.gguf";
        o.dit_model_path      = "x.gguf";
        o.vae_model_path      = "x.gguf";
        o.text_tokens         = 1500000000;
        o.lyric_tokens        = 1000000000;
        o.lm_prompt_tokens    = 100;
        const tts_cpp::acestep::FitResult res = tts_cpp::acestep::fit_params(o);
        expect(res.status == tts_cpp::acestep::FitStatus::Error,
               "near-INT_MAX token counts were not Error");
        expect(res.reason == "workload-too-large",
               "near-INT_MAX token counts reason was '" + res.reason + "'");
        expect(!res.fits, "near-INT_MAX token counts reported fits");

        // Each count is also individually bounded, INT_MAX included.
        tts_cpp::acestep::FitOptions one = o;
        one.text_tokens  = std::numeric_limits<int>::max();
        one.lyric_tokens = 1;
        expect(tts_cpp::acestep::fit_params(one).reason == "workload-too-large",
               "INT_MAX text_tokens was not workload-too-large");
        one = o;
        one.text_tokens       = 1;
        one.lyric_tokens      = 1;
        one.lm_prompt_tokens  = std::numeric_limits<int>::max();
        expect(tts_cpp::acestep::fit_params(one).reason == "workload-too-large",
               "INT_MAX lm_prompt_tokens was not workload-too-large");
        one.lm_prompt_tokens  = 0;
        one.lm_max_new_tokens = std::numeric_limits<int>::max();
        expect(tts_cpp::acestep::fit_params(one).reason == "workload-too-large",
               "INT_MAX lm_max_new_tokens was not workload-too-large");

        // Oversized but syntactically valid integers must reach the workload guard.
        expect_cli_error({"--text-enc", "x.gguf", "--lm", "x.gguf", "--dit", "x.gguf",
                        "--vae", "x.gguf", "--lm-prompt-tokens", "100",
                        "--text-tokens", "1500000000", "--lyric-tokens", "1000000000"},
                         "acestep-fit-params: error: workload-too-large", false);
    }

    if (g_failures == 0) {
        std::printf("test-fit-cli: all checks passed\n");
    }
    return g_failures;
}
