// QVAC-21914: unit test for the clip flash-attention AUTO budget arithmetic
// (clip_fa_effective_min_kv in tools/mtmd/clip.cpp — pure function, exposed
// via clip.h). Locks in both halves of the design:
//   - the fast explicit path survives momentary memory pressure (the cutoff
//     is clamped by STABLE total memory, and free memory may only lower it
//     by the hard-fit requirement — never the old free/2 heuristic), and
//   - the OOM guard fails SAFE: unknown memory info caps the cutoff at a
//     conservative constant instead of trusting the raw default.

#include "clip.h"

#include <cstdio>
#include <cstdlib>

static int g_failures = 0;

static void expect_eq(const char * what, int got, int expected) {
    if (got != expected) {
        std::printf("FAIL %s: got %d, expected %d\n", what, got, expected);
        g_failures++;
    } else {
        std::printf("ok   %s: %d\n", what, got);
    }
}

static void expect_range(const char * what, int got, int lo, int hi) {
    if (got < lo || got > hi) {
        std::printf("FAIL %s: got %d, expected in [%d, %d]\n", what, got, lo, hi);
        g_failures++;
    } else {
        std::printf("ok   %s: %d in [%d, %d]\n", what, got, lo, hi);
    }
}

int main() {
    const size_t GB = 1000ull * 1000ull * 1000ull;

    // Cutoff disabled (or negative) passes through untouched — AUTO stays in
    // legacy "FA whenever supported" mode regardless of memory info.
    expect_eq("disabled cutoff, no mem",        clip_fa_effective_min_kv(0,    0,       0,      16), 0);
    expect_eq("disabled cutoff, with mem",      clip_fa_effective_min_kv(0,    16 * GB, 8 * GB, 16), 0);
    expect_eq("negative cutoff passes through", clip_fa_effective_min_kv(-1,   16 * GB, 0,      16), -1);

    // Plentiful total memory: the 4096 default survives (16 GB, n_head=16:
    // total clamp = sqrt((16e9/2)/(24*16)) ~= 4564 > 4096).
    expect_eq("16GB total keeps default", clip_fa_effective_min_kv(4096, 16 * GB, 0, 16), 4096);

    // 12 GB-class device (Pixel 9-class): total clamp ~= sqrt((12e9/2)/384)
    // ~= 3952 — trims only the topmost band of the default.
    expect_range("12GB total trims to ~3950", clip_fa_effective_min_kv(4096, 12 * GB, 0, 16), 3900, 4000);

    // P2 regression guard: high total + momentarily low free must NOT fall
    // back to the old volatile free/2 heuristic (~1976 at 1.5 GB free). The
    // hard-fit bound sqrt(1.5e9/(12*16)) ~= 2795 is the correct floor.
    expect_range("1.5GB free -> hard-fit, not free/2",
                 clip_fa_effective_min_kv(4096, 16 * GB, (size_t)(1.5 * (double) GB), 16), 2700, 2900);

    // Ample free memory does not restrict below the total clamp...
    expect_eq("8GB free does not restrict", clip_fa_effective_min_kv(4096, 16 * GB, 8 * GB, 16), 4096);
    // ...and free memory can never RAISE the cutoff past the total clamp.
    expect_range("huge free cannot raise past total clamp",
                 clip_fa_effective_min_kv(4096, 2 * GB, 64 * GB, 16), 1550, 1700);

    // No memory info at all: fail SAFE at the conservative constant cap
    // (2048), never the raw 4096 default (~3.2 GB scratch at n_head=16).
    expect_eq("no meminfo caps at 2048",        clip_fa_effective_min_kv(4096, 0, 0, 16), 2048);
    expect_eq("no meminfo keeps smaller cutoff", clip_fa_effective_min_kv(1000, 0, 0, 16), 1000);

    // Degenerate n_head is guarded (treated as 1), not a crash/div-by-zero.
    // Must pass memory info so a sqrt(.../n_head) branch actually runs — with
    // total_mem==free_mem==0 the NO_MEMINFO short-circuit returns before the
    // guard is reached, leaving it untested. At 16 GB total, n_head=1:
    // total clamp = sqrt((16e9/2)/24) ~= 18257 > 4096, so the default 4096
    // survives (and the run completing at all proves no div-by-zero).
    expect_eq("n_head=0 guarded (total mem path)", clip_fa_effective_min_kv(4096, 16 * GB, 0, 0), 4096);

    // Fewer heads => larger permissible n (scratch is linear in n_head):
    // same 1.5 GB free, n_head=4 -> sqrt(1.5e9/48) ~= 5590, so the 4096
    // default survives untouched.
    expect_eq("fewer heads relax the fit bound", clip_fa_effective_min_kv(4096, 16 * GB, (size_t)(1.5 * (double) GB), 4), 4096);

    if (g_failures) {
        std::printf("%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("all ok\n");
    return 0;
}
