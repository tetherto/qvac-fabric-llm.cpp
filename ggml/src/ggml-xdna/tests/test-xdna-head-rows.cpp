// How many leading rows the decode head keeps exact for a requested count
// (GGML_XDNA_HEAD_EXACT_ROWS): never negative, never past the head, whole
// chunks only. A negative count once packed rows from before the weight.
// Host-only: no device is opened.

#include "../xdna-head.h"

#include <cstdio>

static constexpr int64_t N_ROWS = 248320;  // the Qwen3.5 head, Q6_K [1024 x 248320]
static constexpr int64_t CHUNK  = 1024;    // its exact part's chunk on the pool

struct split_case {
    int64_t requested, n, chunk, want;
};

static const split_case CASES[] = {
    { 65536, N_ROWS, CHUNK, 65536 },     // the default
    { 0, N_ROWS, CHUNK, 0 },             // all 4-bit
    { -1024, N_ROWS, CHUNK, 0 },         // negative: all 4-bit, not rows before the weight
    { -100000, N_ROWS, CHUNK, 0 },
    { 1000, N_ROWS, CHUNK, 0 },          // under one chunk
    { 300000, N_ROWS, CHUNK, 247808 },   // past the head: its whole chunks
    { 65536, N_ROWS, 0, 0 },             // no exact geometry
};

static int count_failures(void) {
    int failures = 0;
    for (const split_case & c : CASES) {
        const int64_t got = xdna_head_exact_rows(c.requested, c.n, c.chunk);
        const bool    ok  = got == c.want;
        std::printf("%s: requested %lld of %lld (chunk %lld) -> %lld, want %lld\n", ok ? "ok  " : "FAIL",
                    (long long) c.requested, (long long) c.n, (long long) c.chunk, (long long) got,
                    (long long) c.want);
        failures += ok ? 0 : 1;
    }
    return failures;
}

int main(void) {
    const int failures = count_failures();
    if (failures) {
        std::printf("test-xdna-head-rows: %d failures\n", failures);
        return 1;
    }
    std::printf("test-xdna-head-rows: all passed\n");
    return 0;
}
