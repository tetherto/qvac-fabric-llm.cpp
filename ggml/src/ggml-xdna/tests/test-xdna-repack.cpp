// Which (ggml type, NPU weight format) pairs the repacker accepts, and that it
// writes exactly the bytes the caller sized for the format it asked for
// (#292). Host-only: no device is opened.
//
// Every pair is tried on a row of 256 values - one super-block, the smallest
// a repacker takes - into a buffer of xdna_wfmt_row_bytes(fmt) bytes followed
// by a guard zone. A refused pair must return false and leave the buffer and
// the guard untouched; an accepted one must fill the buffer and nothing past
// it. The reviewer's case is Q6_K into Q4G32: 152 bytes sized, 296 written.

#include "../xdna-quant.h"

#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int g_failures = 0;

static void expect(bool ok, const std::string & what) {
    std::printf("%s: %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) {
        g_failures++;
    }
}

static const char * fmt_name(xdna_wfmt f) {
    switch (f) {
        case XDNA_WFMT_NONE:  return "NONE";
        case XDNA_WFMT_Q4G32: return "Q4G32";
        case XDNA_WFMT_Q8G16: return "Q8G16";
        default:              return "?";
    }
}

// The pairs the backend asks for: each type's own GEMV format, Q4_K widened
// into the 8-bit form so one format can cover a whole decode, Q5_K/Q6_K
// requantized into the 4-bit form (GGML_XDNA_W4), and Q8_0 into the 8-bit one.
static bool supported(ggml_type t, xdna_wfmt f) {
    return (f != XDNA_WFMT_NONE && f == xdna_wfmt_gemv_for(t)) || (t == GGML_TYPE_Q4_K && f == XDNA_WFMT_Q8G16) ||
           ((t == GGML_TYPE_Q5_K || t == GGML_TYPE_Q6_K) && f == XDNA_WFMT_Q4G32) ||
           (t == GGML_TYPE_Q8_0 && f == XDNA_WFMT_Q8G16);
}

int main(void) {
    constexpr int64_t K     = 256;
    constexpr size_t  GUARD = 1024;
    constexpr uint8_t FILL  = 0xA5;

    const ggml_type types[] = { GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0,
                                GGML_TYPE_Q4_0, GGML_TYPE_F16,  GGML_TYPE_F32 };
    const xdna_wfmt fmts[]  = { XDNA_WFMT_NONE, XDNA_WFMT_Q4G32, XDNA_WFMT_Q8G16 };

    // A source row with real content for every type, so an accepted repack
    // has something to write.
    std::vector<float> row(K);
    for (int64_t i = 0; i < K; i++) {
        row[i] = (float) ((i * 37) % 23 - 11) / 7.0f;
    }

    for (ggml_type t : types) {
        std::vector<uint8_t> src(ggml_row_size(t, K));
        ggml_quantize_chunk(t, row.data(), src.data(), 0, 1, K, nullptr);
        for (xdna_wfmt f : fmts) {
            const size_t         want = xdna_wfmt_row_bytes(f, K);
            std::vector<uint8_t> dst(want + GUARD, FILL);
            const bool           ok   = xdna_wfmt_repack_row_as(t, f, src.data(), K, dst.data());
            const std::string    pair = std::string(ggml_type_name(t)) + " -> " + fmt_name(f);

            bool guard_intact = true;
            for (size_t i = want; i < dst.size(); i++) {
                guard_intact &= dst[i] == FILL;
            }
            if (supported(t, f)) {
                bool written = false;
                for (size_t i = 0; i < want; i++) {
                    written |= dst[i] != FILL;
                }
                expect(ok && written && guard_intact,
                       pair + ": accepted, " + std::to_string(want) + " bytes written, none past them");
            } else {
                bool untouched = guard_intact;
                for (size_t i = 0; i < want; i++) {
                    untouched &= dst[i] == FILL;
                }
                expect(!ok && untouched, pair + ": refused before writing");
            }
        }
    }

    // A row that is not whole super-blocks is refused for every pair.
    std::vector<uint8_t> src(ggml_row_size(GGML_TYPE_Q4_K, 512));
    std::vector<uint8_t> dst(4096, FILL);
    expect(!xdna_wfmt_repack_row_as(GGML_TYPE_Q4_K, XDNA_WFMT_Q4G32, src.data(), K + 32, dst.data()),
           "a row that is not whole super-blocks is refused");

    if (g_failures) {
        std::printf("test-xdna-repack: %d failures\n", g_failures);
        return 1;
    }
    std::printf("test-xdna-repack: all passed\n");
    return 0;
}
