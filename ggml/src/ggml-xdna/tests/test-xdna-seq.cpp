// The GEMM instruction stream's sweep window (xdna_gemm_seq_build): how many sweeps of a block are issued before the
// first one's tokens are waited, which descriptor ids they take, and that the transfers stay the same.
// Host-only: no device is opened.

#include "../xdna-seq.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <map>
#include <set>
#include <vector>

static constexpr int HEADER_WORDS = 4;
static constexpr int SHIM_ROW     = 0;
static constexpr int BD_IDS       = 16;

namespace {

struct stream_ops {
    int                   tcts = 0;
    int                   shim_pushes_before_first_tct = 0;
    std::vector<uint32_t> patch_offsets;                    // the DDR patches' offsets, in stream order
    std::map<int, std::set<uint32_t>> ids_before_first_tct; // shim column -> BD ids written before the first wait
    int                   shim_pushes = 0;
    std::vector<int>      shim_pushes_at_tct;               // shim pushes issued when each wait was reached
    int                   max_id = -1;
    size_t                words  = 0;
};

uint32_t op_words(uint32_t opcode) {
    switch (opcode) {
        case xdna_txn::OP_WRITE:      return xdna_txn::SZ_WRITE;
        case xdna_txn::OP_BLOCKWRITE: return xdna_txn::SZ_BLOCKWRITE;
        case xdna_txn::OP_MASKWRITE:  return xdna_txn::SZ_MASKWRITE;
        case xdna_txn::OP_TCT:        return xdna_txn::SZ_TCT;
        default:                      return xdna_txn::SZ_DDR_PATCH;
    }
}

xdna_gemm_tiles bf16_prefill_tiles(int M) {
    xdna_gemm_tiles t;
    t.M              = M;
    t.tile_m         = 32;
    t.rtp_base       = XDNA_RTP_BASE_TILE_M32;
    t.elem_bytes     = 2;
    return t;
}

int reg_row(uint32_t addr_word) {
    return (addr_word >> 20) & 0x1F;
}

int reg_col(uint32_t addr_word) {
    return (addr_word >> 25) & 0x7F;
}

void read_op(const std::vector<uint32_t> & w, size_t at, stream_ops & out) {
    const uint32_t opcode = w[at];
    const bool     before = out.tcts == 0;
    if (opcode == xdna_txn::OP_TCT) {
        out.tcts++;
        out.shim_pushes_at_tct.push_back(out.shim_pushes);
    } else if (opcode == xdna_txn::OP_DDR_PATCH) {
        out.patch_offsets.push_back(w[at + 10]);
    } else if (opcode == xdna_txn::OP_WRITE && reg_row(w[at + 2]) == SHIM_ROW) {
        out.shim_pushes++;
        out.shim_pushes_before_first_tct += before ? 1 : 0;
    } else if (opcode == xdna_txn::OP_BLOCKWRITE) {
        const uint32_t reg = w[at + 2] & 0xFFFFF;
        const int      id  = (int) ((reg - xdna_txn::SHIM_BD_BASE) / xdna_txn::SHIM_BD_STRIDE);
        out.max_id         = std::max(out.max_id, id);
        if (before) {
            out.ids_before_first_tct[reg_col(w[at + 2])].insert((uint32_t) id);
        }
    }
}

bool read_stream(int M, int window, stream_ops & out) {
    xdna_seq        seq;
    xdna_gemm_tiles t = bf16_prefill_tiles(M);
    if (!xdna_gemm_seq_build(&seq, &t, M, 512, 512, 0, 4096, window)) {
        return false;
    }
    const std::vector<uint32_t> w = xdna_seq_build(&seq);
    out.words                     = w.size();
    for (size_t at = HEADER_WORDS; at < w.size(); at += op_words(w[at])) {
        read_op(w, at, out);
    }
    return true;
}

// A 512-row block is four sweeps of 128 rows on eight columns.
constexpr int SWEEPS  = 4;
constexpr int COLS    = 8;
constexpr int A_SHIMS = 4;

int check(bool ok, const char * what) {
    std::printf("%s: %s\n", ok ? "ok  " : "FAIL", what);
    return ok ? 0 : 1;
}

int count_ids(const stream_ops & s) {
    int dup = 0;
    for (const auto & shim : s.ids_before_first_tct) {
        dup += (int) shim.second.size();
    }
    return dup;
}

// Every wait of sweep k is reached with min(k + window, SWEEPS) sweeps issued: never more in flight than the
// window, which keeps the ids of sweeps in flight apart, and never fewer.
bool waits_keep_window(const stream_ops & s, int window) {
    const int pushes_per_sweep = COLS + A_SHIMS + COLS;
    for (size_t t = 0; t < s.shim_pushes_at_tct.size(); t++) {
        const int sweep = (int) t / COLS;
        if (s.shim_pushes_at_tct[t] != std::min(sweep + window, SWEEPS) * pushes_per_sweep) {
            return false;
        }
    }
    return true;
}

int check_window(int window, const stream_ops & base) {
    stream_ops s;
    char       what[96];
    if (!read_stream(SWEEPS * 128, window, s)) {
        return check(false, "stream builds");
    }
    int failures = 0;
    std::snprintf(what, sizeof(what), "window %d: one token a column and sweep", window);
    failures += check(s.tcts == SWEEPS * COLS, what);
    std::snprintf(what, sizeof(what), "window %d: %d sweeps issued before the first wait", window, window);
    failures += check(s.shim_pushes_before_first_tct == window * (COLS + A_SHIMS + COLS), what);
    std::snprintf(what, sizeof(what), "window %d: the same transfers and stream length", window);
    failures += check(s.patch_offsets == base.patch_offsets && s.words == base.words, what);
    std::snprintf(what, sizeof(what), "window %d: ids below %d, none shared by sweeps in flight", window, BD_IDS);
    failures += check(s.max_id < BD_IDS && count_ids(s) == window * (COLS * 2 + A_SHIMS), what);
    std::snprintf(what, sizeof(what), "window %d: every wait keeps %d sweeps in flight, and window %d's do not", window,
                  window, window + 1);
    failures += check(waits_keep_window(s, window) && (window == SWEEPS || !waits_keep_window(s, window + 1)), what);
    return failures;
}

// More sweeps than the block has, or ids the shim has, are clamped.
int check_clamps() {
    stream_ops one, wide, single_sweep, single_sweep_wide;
    int        failures = 0;
    failures += check(read_stream(128, 1, single_sweep) && read_stream(128, 4, single_sweep_wide) &&
                          single_sweep.patch_offsets == single_sweep_wide.patch_offsets &&
                          single_sweep.words == single_sweep_wide.words &&
                          single_sweep.shim_pushes_before_first_tct == single_sweep_wide.shim_pushes_before_first_tct,
                      "one sweep: any window is the same stream");
    failures += check(read_stream(SWEEPS * 128, 1, one) && read_stream(SWEEPS * 128, 99, wide) &&
                          wide.max_id < BD_IDS && wide.tcts == SWEEPS * COLS,
                      "a window past the block or the ids is clamped");
    stream_ops four, five;
    failures += check(read_stream(SWEEPS * 2 * 128, 4, four) && read_stream(SWEEPS * 2 * 128, 5, five) &&
                          four.shim_pushes_before_first_tct == five.shim_pushes_before_first_tct &&
                          four.patch_offsets == five.patch_offsets && four.words == five.words,
                      "a window past the widest measured (4) is clamped to it");
    return failures;
}

}  // namespace

int main(void) {
    stream_ops base;
    if (!read_stream(SWEEPS * 128, 1, base)) {
        std::printf("FAIL: window 1 stream does not build\n");
        return 1;
    }
    int failures = check(base.shim_pushes_before_first_tct == COLS + A_SHIMS + COLS, "window 1: a sweep before the first wait");
    for (int window : { 1, 2, 3, 4 }) {
        failures += check_window(window, base);
    }
    failures += check_clamps();
    if (failures) {
        std::printf("test-xdna-seq: %d failures\n", failures);
        return 1;
    }
    std::printf("test-xdna-seq: all passed\n");
    return 0;
}
