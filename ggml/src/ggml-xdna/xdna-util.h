#pragma once

// Small helpers shared by more than one translation unit. Anything used by a
// single one stays local to it.

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

// f32 -> bf16, round to nearest even. The device side is bf16 throughout.
//
// The integer form, not ggml_fp32_to_bf16_row: that is scalar unless the
// translation unit is built with AVX512-BF16 and this backend is -O3 only, and
// the weight pack calls it once per element. The same round-to-nearest-even on
// the top 16 bits, and the compiler vectorises it.
static inline uint16_t xdna_bf16(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    return (uint16_t) ((b + 0x7FFF + ((b >> 16) & 1)) >> 16);
}

// 2^f on |f| <= 1/2, degree 6 (relative error ~2e-7): the exp2 tail of the
// host sigmoid approximations.
static inline float xdna_exp2_poly(float f) {
    float p = 1.535336188e-4f;
    p       = p * f + 1.339887440e-3f;
    p       = p * f + 9.618437357e-3f;
    p       = p * f + 5.550332471e-2f;
    p       = p * f + 2.402264791e-1f;
    p       = p * f + 6.931472028e-1f;
    return p * f + 1.0f;
}

// Threads for the host layout and quantize passes: half the hardware threads,
// at most 16, at least 1. Sizing the CPU backend's team to the same number
// keeps it from rebuilding the team per region.
static inline int xdna_host_threads() {
    static const int n = std::max(1, std::min(16, (int) std::thread::hardware_concurrency() / 2));
    return n;
}

// Round a size up to a multiple of `a`, the 4 KB device buffer granularity by
// default.
static inline size_t xdna_align_up(size_t v, size_t a = 4096) {
    return (v + a - 1) / a * a;
}

// Value of an integer environment variable, or `def` when unset. strtol stops
// at the first non-digit exactly as atoi did (a non-numeric value parses as 0),
// and an out-of-range value clamps to INT_MIN/INT_MAX instead of overflowing.
static inline int xdna_env_int(const char * name, int def) {
    const char * v = getenv(name);
    if (v == nullptr) {
        return def;
    }
    const long x = strtol(v, nullptr, 10);
    return (int) std::max<long>(std::min<long>(x, INT_MAX), INT_MIN);
}
