#pragma once

// Pure helpers shared by the fit projector (parakeet_fit.cpp) and the
// parakeet-fit-params CLI (fit_main.cpp), kept header-only and dependency-free
// so test/test_fit_cli.cpp can exercise them model-free.
//
// The saturating arithmetic exists for one reason: a preflight's verdict must
// only ever be wrong in the strict direction. Any overflow surfaces as a
// larger requirement (DOES-NOT-FIT), never wraps into a false FITS.

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>

namespace parakeet {
namespace fitutil {

inline uint64_t sat_add(uint64_t a, uint64_t b) {
    const uint64_t s = a + b;
    return s < a ? std::numeric_limits<uint64_t>::max() : s;
}

inline uint64_t sat_mul(uint64_t a, uint64_t b) {
    if (a == 0 || b == 0) return 0;
    if (a > std::numeric_limits<uint64_t>::max() / b) {
        return std::numeric_limits<uint64_t>::max();
    }
    return a * b;
}

// Negative inputs collapse to 0; too-large, infinite, and NaN inputs all
// saturate to max -- an unrepresentable requirement must read as "does not
// fit", never as "needs nothing".
inline uint64_t sat_u64_from_double(double d) {
    if (std::isnan(d)) return std::numeric_limits<uint64_t>::max();
    if (d <= 0.0) return 0;
    if (d >= (double) std::numeric_limits<uint64_t>::max()) {
        return std::numeric_limits<uint64_t>::max();
    }
    return (uint64_t) d;
}

// ── Strict CLI parsers ─────────────────────────────────────────────────────
// A preflight must fail loudly on garbage input, not coerce it (atoi/atof
// return 0 on junk, silently changing which device or workload is projected).

inline bool parse_f32_positive(const char * s, float & out) {
    if (!s) return false;
    char * end = nullptr;
    const double v = std::strtod(s, &end);
    if (end == s || *end != '\0' || !std::isfinite(v) || v <= 0.0) return false;
    out = (float) v;
    return true;
}

// ERANGE checks throughout: strtol/strtoull CLAMP on overflow instead of
// failing, and on LLP64 platforms (Windows) `long` is 32-bit, so without the
// errno check strtol("3000000000") saturates to exactly LONG_MAX == INT_MAX
// and slips through a pure range comparison -- silently accepting garbage on
// one platform that every other platform rejects.

inline bool parse_u64(const char * s, uint64_t & out) {
    if (!s || *s == '-') return false;
    char * end = nullptr;
    errno = 0;
    const unsigned long long v = std::strtoull(s, &end, 10);
    if (end == s || *end != '\0' || errno == ERANGE) return false;
    out = (uint64_t) v;
    return true;
}

inline bool parse_i32(const char * s, int & out) {
    if (!s) return false;
    char * end = nullptr;
    errno = 0;
    const long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0' || errno == ERANGE ||
        v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max()) {
        return false;
    }
    out = (int) v;
    return true;
}

// Saturating MiB -> bytes: an absurd margin makes the verdict stricter,
// never overflows into a false FITS.
inline uint64_t margin_mib_to_bytes(uint64_t mib) {
    return sat_mul(mib, 1024ull * 1024ull);
}

// Length (2-4) of a well-formed UTF-8 sequence starting at s[i], or 0 when the
// lead/continuation bytes are invalid (overlongs, surrogates, > U+10FFFF, and
// truncated tails all return 0). Standard RFC 3629 lead/continuation table.
inline size_t utf8_seq_len(const std::string & s, size_t i) {
    const unsigned char b0 = (unsigned char) s[i];
    size_t n;            // continuation bytes expected
    unsigned char lo = 0x80, hi = 0xBF;  // bounds for the FIRST continuation
    if      (b0 >= 0xC2 && b0 <= 0xDF) { n = 1; }
    else if (b0 == 0xE0)               { n = 2; lo = 0xA0; }
    else if (b0 >= 0xE1 && b0 <= 0xEC) { n = 2; }
    else if (b0 == 0xED)               { n = 2; hi = 0x9F; }  // no surrogates
    else if (b0 >= 0xEE && b0 <= 0xEF) { n = 2; }
    else if (b0 == 0xF0)               { n = 3; lo = 0x90; }
    else if (b0 >= 0xF1 && b0 <= 0xF3) { n = 3; }
    else if (b0 == 0xF4)               { n = 3; hi = 0x8F; }  // <= U+10FFFF
    else return 0;  // 0x80-0xC1 (bare continuation / overlong lead), 0xF5+
    if (i + n >= s.size()) return 0;  // truncated tail
    for (size_t k = 1; k <= n; ++k) {
        const unsigned char b = (unsigned char) s[i + k];
        const unsigned char klo = (k == 1) ? lo : 0x80;
        const unsigned char khi = (k == 1) ? hi : 0xBF;
        if (b < klo || b > khi) return 0;
    }
    return n + 1;
}

// Escape a string for embedding in a JSON string literal. model_variant comes
// from GGUF metadata and device_name from the backend/driver, so neither is
// trusted to be JSON-clean; @qvac/model-fit JSON.parses this output. Valid
// UTF-8 passes through untouched; malformed sequences become U+FFFD so the
// emitted JSON is always valid UTF-8, never a strict-parser failure.
inline std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (size_t i = 0; i < s.size(); ) {
        const unsigned char c = (unsigned char) s[i];
        if (c < 0x80) {
            switch (c) {
                case '"':  out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\b': out += "\\b";  break;
                case '\f': out += "\\f";  break;
                case '\n': out += "\\n";  break;
                case '\r': out += "\\r";  break;
                case '\t': out += "\\t";  break;
                default:
                    if (c < 0x20) {
                        char buf[8];
                        std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                        out += buf;
                    } else {
                        out += (char) c;
                    }
            }
            ++i;
            continue;
        }
        const size_t len = utf8_seq_len(s, i);
        if (len == 0) {
            out += "\xEF\xBF\xBD";  // U+FFFD replacement character
            ++i;                    // resync one byte at a time
        } else {
            out.append(s, i, len);
            i += len;
        }
    }
    return out;
}

}  // namespace fitutil
}  // namespace parakeet
