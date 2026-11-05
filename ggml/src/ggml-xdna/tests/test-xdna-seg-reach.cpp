// Whether a grouped projection's output rows, from its byte offset into the BO, stay within what the 32-bit offset of
// its C segment reaches (xdna_pgemm_seg_in_reach): an offset past 4 GiB is out of reach, not its value modulo 4 GiB.
// Host-only: no device is opened.

#include "../xdna-pgemm.h"

#include <cstdio>

static constexpr size_t ROWS      = 128;   // one block of the array's output
static constexpr size_t LDC       = 1024;  // floats a row
static constexpr size_t OUT_BYTES = ROWS * LDC * sizeof(float);
static constexpr size_t GIB4      = (size_t) 1 << 32;

struct reach_case {
    const char * name;
    size_t       off;
    bool         want;
};

static const reach_case CASES[] = {
    { "at the start",                   0,                       true },
    { "ends a float short of 4 GiB",    GIB4 - OUT_BYTES - 4,    true },
    { "ends at 4 GiB",                  GIB4 - OUT_BYTES,        false },
    { "starts 4 KiB past 4 GiB",        GIB4 + 4096,             false },
};

static int count_failures(void) {
    int failures = 0;
    for (const reach_case & c : CASES) {
        const bool got = xdna_pgemm_seg_in_reach(c.off, ROWS, LDC);
        const bool ok  = got == c.want;
        std::printf("%s: %-28s offset %zu: %s, want %s\n", ok ? "ok  " : "FAIL", c.name, c.off, got ? "in reach" : "out",
                    c.want ? "in reach" : "out");
        failures += ok ? 0 : 1;
    }
    return failures;
}

int main(void) {
    const int failures = count_failures();
    if (failures) {
        std::printf("test-xdna-seg-reach: %d failures\n", failures);
        return 1;
    }
    std::printf("test-xdna-seg-reach: all passed\n");
    return 0;
}
