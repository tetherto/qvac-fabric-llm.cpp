// QVAC (see PATCHES.md): model-free unit tests for the whisper memory-fit
// preflight surface -- the strict numeric parsers and JSON escaper of the
// whisper-fit-params CLI (examples/whisper-fit-params/whisper-fit-util.h),
// plus whisper_fit_params' argument/status contract for inputs that must fail
// before any backend or model work (Error = 2, never a coerced projection).
//
// The fixture-gated parity test (test-whisper-fit-params.cpp) covers the
// projection itself; this one needs no model and runs everywhere.
//
// Exit 0 on success; non-zero (with a FAIL line per broken invariant) otherwise.

#include "whisper.h"

#include "whisper-fit-params/whisper-fit-util.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>

using namespace whisper_fit_util;

namespace {

int g_failures = 0;

void fail(const std::string & what) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
}

void expect(bool cond, const std::string & what) {
    if (!cond) fail(what);
}

void log_callback_null(ggml_log_level, const char *, void *) {}

constexpr uint64_t U64_MAX = std::numeric_limits<uint64_t>::max();

}  // namespace

int main() {
    whisper_log_set(log_callback_null, nullptr);

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
    }

    // ── Saturating margin: an absurd value must widen, never wrap ───────────
    {
        expect(sat_mul(3, 4) == 12,                        "sat_mul(3,4)");
        expect(sat_mul(0, U64_MAX) == 0,                   "sat_mul(0,max)");
        expect(sat_mul(U64_MAX, 2) == U64_MAX,             "sat_mul saturates");
        expect(sat_mul(1ull << 33, 1ull << 33) == U64_MAX, "sat_mul(2^33,2^33) saturates");

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
        // Malformed UTF-8 (model metadata / driver strings are untrusted) must
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

    // ── whisper_fit_params argument/status contract (no model, no backend
    //    work needed; the readable-model paths live in the fixture test) ────
    {
        const whisper_fit_options defaults = whisper_fit_default_options();
        expect(defaults.n_decoders == 5,
               "default n_decoders mirrors whisper_full_default_params (5)");
        expect(defaults.audio_seconds == 300.0f,       "default audio_seconds == 300");
        expect(defaults.margin_bytes == 256ull*1024*1024, "default margin == 256 MiB");

        whisper_fit_result r;

        expect(whisper_fit_params(nullptr, &r) == (int) WHISPER_FIT_ERROR,
               "null opts is Error");
        expect(std::strcmp(r.reason, "invalid-arguments") == 0,
               "null opts reason is invalid-arguments");

        whisper_fit_options o = defaults;
        expect(whisper_fit_params(&o, &r) == (int) WHISPER_FIT_ERROR &&
               std::strcmp(r.reason, "invalid-arguments") == 0,
               "missing model_path is Error/invalid-arguments");

        o = defaults;
        o.model_path = "x.bin";
        o.audio_seconds = 0.0f;
        expect(whisper_fit_params(&o, &r) == (int) WHISPER_FIT_ERROR &&
               std::strcmp(r.reason, "invalid-arguments") == 0,
               "audio_seconds == 0 is Error/invalid-arguments");

        o = defaults;
        o.model_path = "x.bin";
        o.n_decoders = 0;
        expect(whisper_fit_params(&o, &r) == (int) WHISPER_FIT_ERROR &&
               std::strcmp(r.reason, "invalid-arguments") == 0,
               "n_decoders == 0 is Error/invalid-arguments");

        o = defaults;
        o.model_path = "x.bin";
        o.n_decoders = 9; // > WHISPER_MAX_DECODERS
        expect(whisper_fit_params(&o, &r) == (int) WHISPER_FIT_ERROR &&
               std::strcmp(r.reason, "invalid-arguments") == 0,
               "n_decoders > WHISPER_MAX_DECODERS is Error/invalid-arguments");

        // a missing model file must be Error, never Success -- and the return
        // value must equal the status field (the exit-code contract)
        o = defaults;
        o.model_path = "/nonexistent/whisper-fit-test-model.bin";
        o.use_gpu    = false;
        const int rc = whisper_fit_params(&o, &r);
        expect(rc == (int) WHISPER_FIT_ERROR && rc == (int) r.status,
               "missing model file is Error and return == status");
        expect(std::strcmp(r.reason, "model-unreadable") == 0,
               "missing model reason is model-unreadable");
        expect(!r.fits, "missing model must not report fits");
    }

    if (g_failures == 0) {
        std::printf("test-whisper-fit-cli: all checks passed\n");
    }
    return g_failures;
}
