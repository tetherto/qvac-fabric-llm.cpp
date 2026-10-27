#include "xdna-pgemm.h"

#include "ggml-impl.h"
#include "xdna-attn-mm.h"
#include "xdna-gdn-mm.h"
#include "xdna-gemv.h"
#include "xdna-norm.h"
#include "xdna-quant.h"
#include "xdna-runtime.h"
#include "xdna-seq.h"
#include "xdna-util.h"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

// Host side of kernels/pgemm.py. Four arguments:
//   0  W  the decode GEMV's packed tiles of the fused split, regrouped per
//         column as [chunk][group][GEMV tile r][K tile] (w_plan)
//   1  A  per slab two header objects (one per M-half stream), then per half
//         the slab's M blocks, each its K steps of (64 x 64) bf16 row-major
//         (the MemTile tiles them) - the odd half's with each pair of steps
//         swapped
//   2  C  the output rows
//   3  a sink for the objects past the output's columns
// Core (c, row) is GEMV core (c, r = (row - 2) / 2) - the output columns
// ch * 1024 + (c * 2 + r) * 64 of chunk ch - for rows mb = (row - 2) % 2 of
// the M block. A run goes chunk by chunk and in a chunk block by block. The
// weights travel shim MM2S 1 -> MemTile S2MM 5 -> MemTile MM2S 4 + r -> the
// two cores of tile r, the MemTile's side programmed here: a chunk's tiles
// sit in a slot of the MemTile's buffer and are sent once per M block, so
// DDR gives each weight once a run instead of once a block (the GEMM is
// bound by the array's DDR traffic, ~57 GB/s). The A halves enter on MM2S 0
// of columns 6 and 7, every column's C leaves on S2MM 0.

namespace {

constexpr const char * STEM        = "pgemm_c8";
constexpr int          COLS        = 8;
constexpr int          NC          = 64;             // output columns a core
constexpr int          KS          = 64;             // K a step
constexpr int          MB          = 64;             // M rows a core
constexpr int          MBLK        = 2 * MB;         // M rows a block
constexpr int          CHUNK       = COLS * 2 * NC;  // 1024
constexpr int          HDR_BYTES   = MB * KS * 2;
constexpr int          A_COL0      = 6;              // the two A streams' shim columns
// M blocks a run: the descriptor repeats are eight bits, and a slab bounds
// the staging buffers. Slabs are as large as that allows: most calls find A
// laid out already (norm, gate, GLU into A), so cutting a call finer to
// overlap the host's layout with the array only adds commands, and the
// command rate costs the package (~1.5 W, 3% of pp4096 at four slabs a
// call). Below two blocks a run's fixed cost shows.
constexpr int          MAX_BLOCKS  = 16;
constexpr int          SLAB_MIN    = 2;
// Longest K the gated passes take, and so the f32 row they lay out in one.
constexpr int          MAX_K       = 4096;
// C blocks in flight per column, two descriptors a block (target, sink).
constexpr int          C_RING      = 3;
// The weights' path, fixed by kernels/pgemm.py
constexpr uint32_t     W_SHIM_CH   = 1;        // shim MM2S channel
constexpr uint32_t     W_MT_S2MM   = 5;        // MemTile S2MM channel
constexpr uint32_t     W_MT_MM2S   = 4;        // MemTile MM2S channel of GEMV tile r: 4 + r
constexpr uint32_t     W_LOCK0     = 48;       // slot s: (empty, full) locks 48 + 2 s, + 1
constexpr uint32_t     WS_ADDR     = 0x24000;  // the slot buffer, bytes into the MemTile
constexpr size_t       WS_BYTES    = (size_t) 33 * 10752;
constexpr uint32_t     MT_CTRL_PKT = 26;       // the MemTiles' controller packet id
// MemTile descriptors the static program leaves free (it uses 0-5, 24-29):
// S2MM 5's ring, MM2S 4's, MM2S 5's
constexpr uint32_t     MT_BD_S2MM  = 40;
constexpr uint32_t     MT_BD_MM2S0 = 16;
constexpr uint32_t     MT_BD_MM2S1 = 44;
// MemTile tasks in flight a channel: its queue holds about five and drops the
// rest without a word
constexpr int          MT_Q        = 4;
// shim descriptors: 0 the A header, 1 the A body (columns 6, 7), 2-3 W, 4.. C
constexpr uint32_t     SH_BD_W     = 2;
constexpr uint32_t     SH_BD_C     = 4;

// Where a weight's chunks sit in the MemTile: `G` tiles a group (of each
// GEMV tile r), `ng` groups a chunk, `nslots` slots. When a whole chunk fits
// (ng == 1) it is sent once per run and replayed for every M block; when not,
// its groups stream once per block.
struct w_plan {
    int G = 0, ng = 0, nslots = 0;

    bool replay() const { return ng == 1; }
};

w_plan plan_of(const xdna_gemv_geom & g) {
    const size_t pair = 2 * g.tile_bytes();  // a K tile of both r
    const int    nt   = g.n_tiles();
    w_plan       p;
    if ((size_t) 2 * nt * pair <= WS_BYTES) {
        p = { nt, 1, 2 };
    } else if (nt * pair <= WS_BYTES) {
        p = { nt, 1, 1 };
    } else {
        p.G      = (int) (WS_BYTES / (2 * pair));
        p.ng     = (nt + p.G - 1) / p.G;
        p.nslots = 2;
    }
    return p;
}

struct weights {
    xdna_gemv_geom geom;
    w_plan         plan;
    xdna_buffer *  bo = nullptr;
};

// A call's activation as the array reads it: every slab's two headers, then
// every M block. Projections of one input (Q, K and V; the gate and up) read
// the same A, so the last two layouts are kept for the rest of the graph and
// the next such call rewrites only the headers.
struct a_layout {
    const ggml_tensor * src   = nullptr;
    const void *        data  = nullptr;
    int                 M     = 0;
    int                 K     = 0;
    uint64_t            epoch = 0;
    xdna_buffer *       bo    = nullptr;
    size_t              cap   = 0;
    // A one-slab layout's rows from z_rows on are zero in z_bo, for K z_K and
    // z_nmb blocks: the next call of that geometry clears only the rows the
    // last one wrote past its own M, not the whole slab again.
    const xdna_buffer * z_bo   = nullptr;
    int                 z_K    = 0;
    int                 z_nmb  = 0;
    int                 z_rows = 0;
};

// A's layout for M rows of K: whole pairs of 128-row blocks in slabs of an
// even count, per slab its two headers' place, then per half the slab's
// blocks and one spare (a fused SwiGLU's output lets each block's second
// half fall on the next block, the last one's on the spare).
struct a_geom {
    int    n_blk = 0, slab = 0, n_sl = 0;
    size_t a_blk   = 0;  // bytes, a block half
    size_t h_total = 0;  // bytes, every slab's headers

    a_geom(int M, int K) {
        n_blk   = ((M + MBLK - 1) / MBLK + 1) / 2 * 2;
        slab    = std::max(SLAB_MIN, std::min(MAX_BLOCKS, n_blk)) / 2 * 2;
        n_sl    = (n_blk + slab - 1) / slab;
        a_blk   = (size_t) MB * K * 2;
        h_total = (size_t) n_sl * 2 * HDR_BYTES;
    }

    int blocks(int sl) const { return std::min(slab, n_blk - sl * slab); }

    size_t base(int sl) const { return h_total + (size_t) sl * 2 * (slab + 1) * a_blk; }

    size_t half(int sl, int h) const { return base(sl) + (size_t) h * (blocks(sl) + 1) * a_blk; }

    size_t slab_bytes(int sl) const { return 2 * (size_t) (blocks(sl) + 1) * a_blk; }

    size_t bytes() const { return base(n_sl - 1) + slab_bytes(n_sl - 1); }
};

// Slab sl's rows of A into bo, bf16, the rows past M zero: per half the
// slab's blocks, each its K steps of (64 x 64) row-major (the MemTile tiles
// them); in an odd half's rows each pair of K steps swapped, the order its
// cores take them in. `row(m, o, sw)` writes row m, its K step st ^ sw at
// o + st * MB * KS.
// `upto` rows of the slab are written (the rest are known to be zero
// already); -1 is all of them.
template <class Row> bool lay_slab(const a_geom & ag, xdna_buffer * bo, int sl, int M, int K, Row row, int upto = -1) {
    const int  nstep = K / KS;
    const int  m0 = sl * ag.slab * MBLK, nmb = ag.blocks(sl);
    const int  nr = upto < 0 ? nmb * MBLK : std::min(upto, nmb * MBLK);
    uint16_t * ab = (uint16_t *) ((char *) bo->data + ag.base(sl));
    // a few rows are cheaper than waking the team
#pragma omp parallel for num_threads(xdna_host_threads()) if (nr >= 32)
    for (int rr = 0; rr < nr; rr++) {
        const int  m = m0 + rr;
        const int  b = rr / MBLK, hf = (rr % MBLK) / MB, r = rr % MB;
        uint16_t * o = ab + ((size_t) (hf * (nmb + 1) + b) * MB * K + (size_t) r * KS);
        if (m >= M) {
            for (int st = 0; st < nstep; st++) {
                std::memset(o + (size_t) st * MB * KS, 0, (size_t) KS * 2);
            }
            continue;
        }
        row(m, o, hf);
    }
    return xdna_buffer_sync_to_device_range(bo, ag.slab_bytes(sl), ag.base(sl));
}

struct runner {
    std::mutex                                                      mtx;
    xdna_kernel *                                                   kern = nullptr;
    // (weight, second weight, concatenated): the second is a GLU's up or
    // the columns that follow the first's
    std::map<std::tuple<const void *, const void *, bool>, weights> w;
    a_layout                                                        a[2];
    int                                                             a_next = 0;
    // the next projection's A a fused SwiGLU writes (xdna_pgemm_run_glu_fused)
    a_layout                                                        fused;
    // an RMS_NORM's A laid out at its MUL (xdna_pgemm_run_norm)
    a_layout                                                        pre;
    // this graph's activations laid out without their f32 node written: a
    // call that finds one of them but not its layout must not read the node
    std::vector<const ggml_tensor *>                                unwritten;
    uint64_t                                                        epoch = 1;
};

runner g_pg;

bool artifact_present() {
    static const bool present = !xdna_artifact_find(STEM, false).xclbin.empty();
    return present;
}

// Each weight in its own format: the decode's 4-bit re-quantization of some
// Q5_K/Q6_K tensors (xdna_gemv_type_of) buys the decode bandwidth, which the
// prefill does not need - here it would cost KLD 0.0041 -> 0.0105.
// Threads for the activation and output layouts: the CPU backend's team
// size. A region of any other size makes the runtime rebuild its team, which
// costs more than the work (a 1 MB activation: 33 us at 16, 380 at 32).
// `n` (a multiple of 16) f32 to bf16, rounded to nearest even as xdna_bf16
// does. The backend builds for plain x86-64; XDNA ships only with Zen 4/5,
// so AVX2 is there, and the activation layouts run on 16 threads while the
// array works - scalar, they cost the package ~6 W through the GEMM.
__attribute__((target("avx2"))) void bf16_row(const float * src, uint16_t * dst, int n) {
    for (int j = 0; j < n; j += 16) {
        const __m256i a = _mm256_loadu_si256((const __m256i *) (src + j));
        const __m256i b = _mm256_loadu_si256((const __m256i *) (src + j + 8));
        _mm256_storeu_si256((__m256i *) (dst + j), xdna_bf16x16(a, b));
    }
}

// bf16_row of (src * s) * g: an RMS_NORM's row times its weight, each product
// rounded to f32 in the order ggml's two ops round them
__attribute__((target("avx2"))) void bf16_row_norm(const float * src, const float * g, float s, uint16_t * dst, int n) {
    const __m256 vs = _mm256_set1_ps(s);
    for (int j = 0; j < n; j += 16) {
        const __m256 a = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(src + j), vs), _mm256_loadu_ps(g + j));
        const __m256 b = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(src + j + 8), vs), _mm256_loadu_ps(g + j + 8));
        _mm256_storeu_si256((__m256i *) (dst + j), xdna_bf16x16(_mm256_castps_si256(a), _mm256_castps_si256(b)));
    }
}

// The geometry of `n` columns of w. Q8_0 (the recurrent layers' alpha and
// beta) packs into the 8-bit form as Q6_K does; the decode never streams it,
// so only here does it map.
xdna_gemv_geom geom_of(const ggml_tensor * w, int64_t n) {
    const ggml_type t = w->type == GGML_TYPE_Q8_0 ? GGML_TYPE_Q6_K : w->type;
    return xdna_gemv_variant(t, w->ne[0], n, false, XDNA_GEMV_SPLIT_FUSED, /* uncapped */ true);
}

xdna_gemv_geom geom_of(const ggml_tensor * w) {
    return geom_of(w, w->ne[1]);
}

// A linear transfer of `words`, encoded as IRON writes one: unit strides
// (a zero stride encodes as 0xFFFFF) and the AXI cache bits.
xdna_bd linear_bd(uint32_t words) {
    xdna_bd bd;
    bd.buf_len   = words;
    bd.d0_stride = bd.d1_stride = bd.d2_stride = 1;
    bd.ax_cache                                = 2;
    return bd;
}

// Where a call's C goes: a row-major (rows x ldc) f32 target, argument 2 at
// byte offset `off`, and per column the chunks whose columns lie inside it
// (the rest of a column's chunks go to the sink, argument 3).
struct c_dest {
    uint32_t ldc      = 0;   // floats a row
    uint32_t off      = 0;   // bytes into argument 2
    int      objs     = 0;   // C objects a column's block
    int      valid[8] = {};  // objects a column keeps
    // a fused SwiGLU: argument 2 is the next projection's A, K rows of the
    // slab's two halves at bytes half0 / half1 (a_geom)
    bool     fused    = false;
    uint32_t half[2]  = {};
    uint32_t a_blk    = 0;
    // blocks of the slab with rows of the output; the rest (the pad to an
    // even count) go to the sink
    int      nreal    = 0;
};

// MemTile DMA (xdna_seq's descriptors are the shim's): a MemTile's own locks
// and memory as its DMA addresses them, locks 64 on and words 0x20000 on.
void mt_set_lock(xdna_seq * seq, uint32_t col, uint32_t id, uint32_t value) {
    xdna_seq_write(seq, col, 1, 0xC0000 + id * 0x10, value);
}

// A linear descriptor of `words` at byte `addr` of the MemTile, taking lock
// `acq` at `acq_val` or more and releasing `rel` by `rel_val`
void mt_bd(xdna_seq * seq,
           uint32_t   col,
           uint32_t   id,
           uint32_t   words,
           uint32_t   addr,
           uint32_t   acq,
           uint32_t   acq_val,
           uint32_t   rel,
           uint32_t   rel_val) {
    seq->n_instr++;
    seq->ops.push_back(xdna_txn::OP_BLOCKWRITE);
    seq->ops.push_back(0);
    seq->ops.push_back(xdna_txn::tile_reg(col, 1, 0xA0000 + id * 0x20));
    seq->ops.push_back(xdna_txn::SZ_BLOCKWRITE * 4);
    seq->ops.push_back(words);
    seq->ops.push_back(0x20000 + addr / 4);
    for (int k = 0; k < 5; k++) {
        seq->ops.push_back(0);
    }
    seq->ops.push_back(1u << 31 | (rel_val & 0x7F) << 24 | (64 + rel) << 16 | 1u << 15 | ((0u - acq_val) & 0x7F) << 8 |
                       (64 + acq));
}

// Start MemTile task `id` on a channel, `repeat` more times, with a token
void mt_push(xdna_seq * seq, uint32_t col, uint32_t id, xdna_dma_dir dir, uint32_t ch, uint32_t repeat) {
    const uint32_t ctrl = (dir == xdna_dma_dir::MM2S ? 0xA0630 : 0xA0600) + ch * 8;
    xdna_seq_maskwrite(seq, col, 1, ctrl, MT_CTRL_PKT << 8, 0x1F00);
    xdna_seq_write(seq, col, 1, ctrl + 4, id | repeat << 16 | 1u << 31);
}

// The stream for `nmb` M blocks of a weight of geometry `g` and plan `wp`,
// the slab's A headers at byte `h_off` of argument 1 and its halves at
// `a_off` (each nmb blocks).
std::vector<uint32_t> build_seq(const xdna_gemv_geom & g,
                                const w_plan &         wp,
                                int                    nmb,
                                const c_dest &         cd,
                                uint32_t               h_off,
                                uint32_t               a_off) {
    const int      nch   = g.n_out();
    const int      nt    = g.n_tiles();
    const uint32_t tb    = (uint32_t) g.tile_bytes();
    const uint32_t nstep = (uint32_t) (g.K / KS);
    const uint32_t a_blk = nstep * MB * KS * 2;               // bytes, a block half
    const uint32_t w_col = (uint32_t) g.col_weight_bytes();
    const uint32_t w_ch  = (uint32_t) nt * 2 * tb;            // a chunk of a column
    const uint32_t slot  = (uint32_t) wp.G * 2 * tb;          // bytes, a slot
    const uint32_t R     = wp.replay() ? (uint32_t) nmb : 1;  // sends a slot's fill
    const int      opc   = cd.objs / nch;                     // C objects a chunk's block

    xdna_seq seq;
    for (int c = 0; c < COLS; c++) {
        for (int k = 0; k < wp.nslots; k++) {
            mt_set_lock(&seq, c, W_LOCK0 + 2 * k, 2 * R);  // empty: both tiles' sends to go
            mt_set_lock(&seq, c, W_LOCK0 + 2 * k + 1, 0);
        }
    }
    // A: the header, then the slab's half, once a chunk
    for (int h = 0; h < 2; h++) {
        xdna_bd bd = linear_bd(HDR_BYTES / 4);
        xdna_seq_blockwrite(&seq, A_COL0 + h, 0, 0, &bd);
        xdna_seq_ddr_patch(&seq, A_COL0 + h, 0, 0, 1, h_off + (uint32_t) h * HDR_BYTES);
        xdna_seq_push_queue(&seq, A_COL0 + h, 0, 0, xdna_dma_dir::MM2S, 0, false, 0);
        bd = linear_bd((uint32_t) nmb * a_blk / 4);
        xdna_seq_blockwrite(&seq, A_COL0 + h, 0, 1, &bd);
        xdna_seq_ddr_patch(&seq, A_COL0 + h, 0, 1, 1, a_off + (uint32_t) h * (nmb + 1) * a_blk);
        xdna_seq_push_queue(&seq, A_COL0 + h, 0, 1, xdna_dma_dir::MM2S, 0, false, (uint32_t) nch - 1);
    }
    // W from DDR: replayed, each column's whole stream once; streamed, a
    // chunk once per block
    int  w_pushed = 0;
    auto w_chunk  = [&](int ch) {
        for (int c = 0; c < COLS; c++) {
            if (w_pushed >= 2) {
                xdna_seq_wait_token(&seq, c, 0, xdna_dma_dir::MM2S, W_SHIM_CH);
            }
            const uint32_t id = SH_BD_W + (uint32_t) (w_pushed % 2);
            xdna_bd        bd = linear_bd((ch < 0 ? w_col : w_ch) / 4);
            xdna_seq_blockwrite(&seq, c, 0, id, &bd);
            xdna_seq_ddr_patch(&seq, c, 0, id, 0, (uint32_t) c * w_col + (ch < 0 ? 0 : (uint32_t) ch * w_ch));
            xdna_seq_issue_token(&seq, c, 0, xdna_dma_dir::MM2S, W_SHIM_CH, 0xF);
            xdna_seq_push_queue(&seq, c, 0, id, xdna_dma_dir::MM2S, W_SHIM_CH, true, ch < 0 ? 0 : (uint32_t) nmb - 1);
        }
        w_pushed++;
    };
    // a slot fill of `gt` tiles of each r at byte `src` of the chunk order:
    // the MemTile takes it in, then sends each r's part R times
    int  inst   = 0;
    auto w_inst = [&](uint32_t gt) {
        const uint32_t k = (uint32_t) (inst % wp.nslots), q = (uint32_t) (inst % MT_Q);
        const uint32_t e = W_LOCK0 + 2 * k, f = e + 1, at = WS_ADDR + k * slot;
        for (int c = 0; c < COLS; c++) {
            if (inst >= MT_Q) {
                xdna_seq_wait_token(&seq, c, 1, xdna_dma_dir::S2MM, W_MT_S2MM);
                xdna_seq_wait_token(&seq, c, 1, xdna_dma_dir::MM2S, W_MT_MM2S);
                xdna_seq_wait_token(&seq, c, 1, xdna_dma_dir::MM2S, W_MT_MM2S + 1);
            }
            mt_bd(&seq, c, MT_BD_S2MM + q, gt * 2 * tb / 4, at, e, 2 * R, f, 2 * R);
            mt_push(&seq, c, MT_BD_S2MM + q, xdna_dma_dir::S2MM, W_MT_S2MM, 0);
            for (uint32_t r = 0; r < 2; r++) {
                const uint32_t id = (r ? MT_BD_MM2S1 : MT_BD_MM2S0) + q;
                mt_bd(&seq, c, id, gt * tb / 4, at + r * gt * tb, f, 1, e, 1);
                mt_push(&seq, c, id, xdna_dma_dir::MM2S, W_MT_MM2S + r, R - 1);
            }
        }
        inst++;
    };
    // C of chunk ch's block b: object h [GEMV tile r][128 rows][32 columns]
    // for columns (ch opc + h) 512 + column * 64 + r * 32 - a descriptor puts
    // one into the target (32 floats, 128 rows ldc apart, the two parts 32
    // apart) and repeats it 512 columns on; the objects past the output go to
    // the sink. The ring reuses a descriptor only once its transfer is done.
    int  c_e     = 0;
    auto c_block = [&](int ch, int b) {
        if (c_e >= C_RING - 1) {
            for (int c = 0; c < COLS; c++) {
                xdna_seq_wait_token(&seq, c, 0, xdna_dma_dir::S2MM, 0);
            }
        }
        if (cd.fused) {
            // a core's object is its 64 rows x 32 bf16 of columns ch 512 +
            // c 64 + r 32 (K step ch 8 + c, pair-swapped in the odd half),
            // then 4 KB that land on the next block's rows; the column's four
            // in the join's order (r, mb) = (i / 2, i % 2), a chain of four
            for (int c = 0; c < COLS; c++) {
                const uint32_t id0 = SH_BD_C + 4 * (uint32_t) (c_e % C_RING);
                for (uint32_t i = 0; i < 4; i++) {
                    const uint32_t r = i / 2, mb = i % 2;
                    const uint32_t st = (uint32_t) (ch * 8 + c) ^ mb;
                    xdna_bd        bd;
                    bd.buf_len   = 2048;
                    bd.d0_size   = 16;
                    bd.d0_stride = 1;
                    bd.d1_size   = MB;
                    bd.d1_stride = KS / 2;
                    bd.d2_stride = cd.a_blk / 4;
                    bd.ax_cache  = 2;
                    bd.next_bd   = id0 + i + 1;
                    bd.use_next  = i < 3;
                    xdna_seq_blockwrite(&seq, c, 0, id0 + i, &bd);
                    xdna_seq_ddr_patch(&seq, c, 0, id0 + i, 2,
                                       cd.half[mb] + (uint32_t) b * cd.a_blk + (st * MB * KS + r * 32) * 2);
                }
                xdna_seq_issue_token(&seq, c, 0, xdna_dma_dir::S2MM, 0, 0xF);
                xdna_seq_push_queue(&seq, c, 0, id0, xdna_dma_dir::S2MM, 0, true, 0);
            }
            c_e++;
            return;
        }
        for (int c = 0; c < COLS; c++) {
            const uint32_t id = SH_BD_C + 2 * (uint32_t) (c_e % C_RING);
            const int      nv = b < cd.nreal ? std::max(0, std::min(opc, cd.valid[c] - ch * opc)) : 0;
            const int      nx = opc - nv;
            if (nv > 0) {
                xdna_bd bd;
                bd.buf_len     = 2 * MBLK * NC / 2;  // an object, words
                bd.d0_size     = NC / 2;
                bd.d0_stride   = 1;
                bd.d1_size     = MBLK;
                bd.d1_stride   = cd.ldc;
                bd.d2_stride   = NC / 2;
                bd.iter_size   = (uint32_t) nv;
                bd.iter_stride = CHUNK / 2;
                bd.ax_cache    = 2;
                xdna_seq_blockwrite(&seq, c, 0, id, &bd);
                xdna_seq_ddr_patch(
                    &seq, c, 0, id, 2,
                    cd.off +
                        ((uint32_t) b * MBLK * cd.ldc + (uint32_t) (ch * opc) * CHUNK / 2 + (uint32_t) c * NC) * 4);
                if (nx == 0) {
                    xdna_seq_issue_token(&seq, c, 0, xdna_dma_dir::S2MM, 0, 0xF);
                }
                xdna_seq_push_queue(&seq, c, 0, id, xdna_dma_dir::S2MM, 0, nx == 0, (uint32_t) nv - 1);
            }
            if (nx > 0) {
                xdna_bd bd = linear_bd(2 * MBLK * NC / 2);
                xdna_seq_blockwrite(&seq, c, 0, id + 1, &bd);
                xdna_seq_ddr_patch(&seq, c, 0, id + 1, 3, 0);
                xdna_seq_issue_token(&seq, c, 0, xdna_dma_dir::S2MM, 0, 0xF);
                xdna_seq_push_queue(&seq, c, 0, id + 1, xdna_dma_dir::S2MM, 0, true, (uint32_t) nx - 1);
            }
        }
        c_e++;
    };
    if (wp.replay()) {
        w_chunk(-1);
        for (int ch = 0; ch < nch; ch++) {
            w_inst((uint32_t) nt);
            for (int b = 0; b < nmb; b++) {
                c_block(ch, b);
            }
        }
    } else {
        for (int ch = 0; ch < nch; ch++) {
            w_chunk(ch);
            for (int b = 0; b < nmb; b++) {
                for (int gi = 0; gi < wp.ng; gi++) {
                    w_inst((uint32_t) std::min(wp.G, nt - gi * wp.G));
                }
                c_block(ch, b);
            }
        }
    }
    // every token issued is taken before the run ends
    for (int c = 0; c < COLS; c++) {
        for (int k = std::max(0, inst - MT_Q); k < inst; k++) {
            xdna_seq_wait_token(&seq, c, 1, xdna_dma_dir::S2MM, W_MT_S2MM);
            xdna_seq_wait_token(&seq, c, 1, xdna_dma_dir::MM2S, W_MT_MM2S);
            xdna_seq_wait_token(&seq, c, 1, xdna_dma_dir::MM2S, W_MT_MM2S + 1);
        }
        for (int k = std::max(0, w_pushed - 2); k < w_pushed; k++) {
            xdna_seq_wait_token(&seq, c, 0, xdna_dma_dir::MM2S, W_SHIM_CH);
        }
    }
    for (int k = std::max(0, c_e - (C_RING - 1)); k < c_e; k++) {
        for (int c = 0; c < COLS; c++) {
            xdna_seq_wait_token(&seq, c, 0, xdna_dma_dir::S2MM, 0);
        }
    }
    return xdna_seq_build(&seq);
}

// The weight in the array's column order (kernels/pgemm.py): core (c, r)'s
// 64 columns of chunk ch are two parts h of 32, and part h is output columns
// (2 ch + h) * 512 + (c * 2 + r) * 32 - so a column's C objects follow each
// other 512 columns apart. A gate/up pair (`up` given) packs as one weight
// whose parts are the gate's and the up's columns ch * 512 + (c * 2 + r) *
// 32, the cores' SwiGLU of them the output.
// With `cat` the second weight is no GLU's up but more columns: output
// columns N .. N + its own of one plain weight.
weights * weights_for(xdna_kernel_pool *  pool,
                      const ggml_tensor * w,
                      const ggml_tensor * up  = nullptr,
                      bool                cat = false) {
    const std::tuple<const void *, const void *, bool> key{ w->data, up ? up->data : nullptr, cat };
    auto                                               it = g_pg.w.find(key);
    if (it != g_pg.w.end()) {
        return &it->second;
    }
    const int N = (int) w->ne[1];
    weights   ws;
    ws.geom = cat ? geom_of(w, N + up->ne[1]) :
              up  ? xdna_gemv_variant(w->type, w->ne[0], (int64_t) 2 * N, false, XDNA_GEMV_SPLIT_FUSED, true) :
                    geom_of(w);
    if (!ws.geom.valid()) {
        return nullptr;
    }
    std::vector<int32_t> colmap((size_t) ws.geom.N, -1);
    for (int p = 0; p < ws.geom.N; p++) {
        const int ch = p / CHUNK, core = (p % CHUNK) / NC, h = (p % NC) / 32, x = p % 32;
        if (up && !cat) {
            const int n        = ch * CHUNK / 2 + core * 32 + x;
            colmap[(size_t) p] = n < N ? h * N + n : -1;
        } else {
            const int n        = (2 * ch + h) * CHUNK / 2 + core * 32 + x;
            colmap[(size_t) p] = n < ws.geom.n_real ? n : -1;
        }
    }
    const ggml_tensor *  src[2] = { w, up };
    std::vector<uint8_t> packed;
    if (!xdna_gemv_pack_weights(ws.geom, src, up ? 2 : 1, colmap.data(), packed)) {
        return nullptr;
    }
    ws.bo = xdna_buffer_alloc(pool->device, packed.size());
    if (!ws.bo || !ws.bo->data) {
        xdna_buffer_free(ws.bo);
        return nullptr;
    }
    // [column][chunk][K tile][r] -> [column][chunk][group][r][K tile of the group]
    ws.plan = plan_of(ws.geom);
    {
        const size_t tb = ws.geom.tile_bytes();
        const int    nt = ws.geom.n_tiles(), nch = ws.geom.n_out(), G = ws.plan.G;
        uint8_t *    dst = (uint8_t *) ws.bo->data;
        for (int c = 0; c < COLS; c++) {
            for (int ch = 0; ch < nch; ch++) {
                const size_t base = ((size_t) c * nch + ch) * nt * 2 * tb;
                for (int t = 0; t < nt; t++) {
                    const int gi = t / G, tg = t % G, gt = std::min(G, nt - gi * G);
                    for (int r = 0; r < 2; r++) {
                        std::memcpy(dst + base + ((size_t) gi * G * 2 + (size_t) r * gt + tg) * tb,
                                    packed.data() + base + ((size_t) t * 2 + r) * tb, tb);
                    }
                }
            }
        }
    }
    if (!xdna_buffer_sync_to_device(ws.bo)) {
        xdna_buffer_free(ws.bo);
        return nullptr;
    }
    return &g_pg.w.emplace(key, ws).first->second;
}

}  // namespace

// ggml-xdna.cpp: the BO a host pointer of the backend's buffers lies in
xdna_buffer * xdna_host_bo_of(const void * p, size_t * offset);

bool xdna_pgemm_supported(const ggml_tensor * op) {
    if (!op || op->op != GGML_OP_MUL_MAT || xdna_env_int("GGML_XDNA_PGEMM", 1) == 0 || !artifact_present()) {
        return false;
    }
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * a = op->src[1];
    // Any M, not just a whole tile. A short prompt used to fall to the bf16
    // GEMM, which packs every weight it touches into bf16 (two bytes a
    // parameter) and keeps it for the life of the context: 9 GiB for a 9B on
    // a 15-token prompt, which is what stopped a full context from fitting the
    // host BOs. The kernel below handles M under its tile already - the rows
    // past M are zeroed in A's layout, and the output of a call whose M is not
    // a whole block leaves through the staging buffers, where the host copies
    // only the M real rows.
    if (!w || !a || a->ne[1] <= 0 || a->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return false;
    }
    // A contiguous activation of more than two dimensions is its rows one
    // after another: ssm_out reads [d_inner, n_seq_tokens, n_seqs]. Taking it
    // only when n_seqs is 1 put a request's ssm_out on the array alone and on
    // the host beside other requests, so its logits changed with the batch.
    if (w->ne[2] * w->ne[3] != 1 || w->ne[0] != a->ne[0] || ggml_nrows(op) != ggml_nrows(a)) {
        return false;
    }
    if (!ggml_is_contiguous(w) || !ggml_is_contiguous(a) || !ggml_is_contiguous(op)) {
        return false;
    }
    if (!ggml_is_quantized(w->type)) {
        return false;
    }
    // The vocabulary projection would pack a few hundred MB for a prefill
    // that only reads its last row.
    if (w->ne[1] > 16384) {
        return false;
    }
    // C is unblocked eight columns at a time
    if (w->ne[1] % 8) {
        return false;
    }
    const xdna_gemv_geom g = geom_of(w);
    return g.valid() && g.K % KS == 0 && g.n_out() <= 256;
}

void xdna_pgemm_graph_begin(void) {
    std::lock_guard<std::mutex> lock(g_pg.mtx);
    g_pg.epoch++;
    g_pg.unwritten.clear();
}

void xdna_pgemm_release(void) {
    std::lock_guard<std::mutex> lock(g_pg.mtx);
    for (auto & kv : g_pg.w) {
        xdna_buffer_free(kv.second.bo);
        kv.second.bo = nullptr;
    }
    g_pg.w.clear();
}

namespace {

// One call: C (M x N, f32, into `node`) = A (a, M x K) times the packed
// weight, or with `glu` the SwiGLU of a gate/up pair's, N columns of each.
// A destination of the backend's own instead of the node's data: rows ldc =
// N floats apart from byte `off` of `bo`, room for M rounded up to 128.
struct pg_dst {
    xdna_buffer * bo  = nullptr;
    size_t        off = 0;
};

// With `fuse` (a SwiGLU call) the output goes as bf16 into that layout, the
// next projection's A (K = N): nothing is written to the node.
bool pgemm_call(xdna_kernel_pool *  pool,
                const ggml_tensor * a,
                weights *           ws,
                const ggml_tensor * node,
                int                 N,
                bool                glu,
                const pg_dst *      ov   = nullptr,
                a_layout *          fuse = nullptr) {
    const xdna_gemv_geom & g     = ws->geom;
    const int              M     = (int) ggml_nrows(a);  // contiguous, so a row is nb[1] after the last
    const int              K     = g.K;
    const int              nch   = g.n_out();
    const int              objs  = nch * (glu ? 1 : 2);  // C objects a column's block
    const int              ldc_s = objs * CHUNK / 2;     // a staging row
    const int              nstep = K / KS;
    const bool             q8    = g.fmt != XDNA_WFMT_Q4G32;
    const int              steps = g.k_tile() / KS;

    // The rows go in slabs, so the host lays out the next slab's A while the
    // array runs this one. A is laid out once for the whole call - every
    // slab's two headers, then every M block - and kept for the next call
    // that reads the same activation in this graph, which rewrites only the
    // headers (they name the call's chunks and format).
    const a_geom ag(M, K);
    const int    n_blk     = ag.n_blk;
    const int    slab      = ag.slab;
    const int    n_sl      = ag.n_sl;
    const size_t h_total   = ag.h_total;
    auto         blocks_of = [&](int sl) {
        return ag.blocks(sl);
    };

    a_layout * al = nullptr;
    for (a_layout * e : { &g_pg.a[0], &g_pg.a[1], &g_pg.fused, &g_pg.pre }) {
        if (e->bo && e->src == a && e->data == a->data && e->M == M && e->K == K && e->epoch == g_pg.epoch) {
            al = e;
        }
    }
    const bool a_ready = al != nullptr;
    if (!a_ready && std::find(g_pg.unwritten.begin(), g_pg.unwritten.end(), a) != g_pg.unwritten.end()) {
        GGML_LOG_ERROR("%s: %s's A was laid out without its node and is gone\n", "xdna-pgemm", a->name);
        return false;
    }
    if (!al) {
        al = &g_pg.a[g_pg.a_next];
        g_pg.a_next ^= 1;
        const size_t need = ag.bytes();
        if (al->cap < need) {
            if (al->bo) {
                xdna_buffer_free(al->bo);
            }
            al->bo   = xdna_buffer_alloc(pool->device, need);
            al->cap  = al->bo ? need : 0;
            al->z_bo = nullptr;  // new memory, and maybe the old pointer
        }
        al->src = nullptr;  // valid once every slab is laid out
    }
    if (!al->bo || !al->bo->data) {
        GGML_LOG_ERROR("%s: the activation layout buffer is null or unmapped\n", "xdna-pgemm");
        return false;
    }
    // every slab's headers: its blocks times the chunks, the K tiles a block,
    // the tile format, the K steps a tile and the pairs of them
    {
        uint16_t * ah = (uint16_t *) al->bo->data;
        std::memset(ah, 0, h_total);
        for (int sl = 0; sl < n_sl; sl++) {
            for (int h = 0; h < 2; h++) {
                int32_t  words[8] = { blocks_of(sl) * nch,    g.n_tiles(),  q8 ? 1 : 0, steps, steps / 2, glu ? 1 : 2,
                                     fuse ? 2 : glu ? 1 : 0, steps / 2 - 1 };
                uint16_t hb[16];                     // (a memcpy: read through a uint16_t
                std::memcpy(hb, words, sizeof(hb));  // pointer, the ints are dead stores)
                // the header goes through the MemTile's tiling too: its bf16
                // j (row j / 8 of the first 8 x 8 sub-tile) is sent from
                // row j / 8, column j % 8 of the (64 x 64) object
                uint16_t * hdr = ah + (size_t) (sl * 2 + h) * HDR_BYTES / 2;
                for (int j = 0; j < 16; j++) {
                    hdr[(j / 8) * KS + j % 8] = hb[j];
                }
            }
        }
        if (!xdna_buffer_sync_to_device_range(al->bo, h_total, 0)) {
            return false;
        }
    }
    // slab sl's rows (lay_slab)
    auto pack = [&](int sl) {
        const auto row = [&](int m, uint16_t * o, int sw) {
            const float * r = (const float *) ((const char *) a->data + (size_t) m * a->nb[1]);
            for (int st = 0; st < nstep; st++) {
                bf16_row(r + (size_t) (st ^ sw) * KS, o + (size_t) st * MB * KS, KS);
            }
        };
        if (n_sl != 1) {
            al->z_bo = nullptr;
            return lay_slab(ag, al->bo, sl, M, K, row);
        }
        // A short M lays out a few rows of a 128-row block: the zero rows past
        // it are left from the last call of this geometry, and only the rows
        // it wrote past this M are cleared again.
        const int  nmb   = blocks_of(0);
        const bool known = al->z_bo == al->bo && al->z_K == K && al->z_nmb == nmb;
        const bool ok    = lay_slab(ag, al->bo, sl, M, K, row, known ? std::max(M, al->z_rows) : -1);
        al->z_bo         = ok ? al->bo : nullptr;
        al->z_K          = K;
        al->z_nmb        = nmb;
        al->z_rows       = M;
        return ok;
    };
    // C from a staging buffer (row-major, ldc_s floats a row) into dst, when
    // the array could not write dst itself
    auto unpack = [&](xdna_buffer * bo, int m0, int rows) {
        // only the real rows: a short M would otherwise drop the cache over
        // a whole 128-row block of staging (3 MB for a 6144-wide projection)
        const int real = std::max(0, std::min(rows, M - m0));
        if (real == 0) {
            return true;
        }
        if (!xdna_buffer_sync_from_device_range(bo, (size_t) real * ldc_s * 4, 0)) {
            return false;
        }
        const float * c = (const float *) bo->data;
#pragma omp parallel for num_threads(xdna_host_threads()) if (real >= 32)
        for (int r = 0; r < real; r++) {
            std::memcpy((char *) node->data + (size_t) (m0 + r) * node->nb[1], c + (size_t) r * ldc_s,
                        (size_t) N * sizeof(float));
        }
        return true;
    };

    // The array writes C into dst itself - rows, through the MemTile - when
    // dst is in one of this backend's BOs and whole blocks of 128 rows and 128
    // columns fit it; otherwise into staging buffers, two deep, the host
    // copies rows out of.
    const a_geom fg(M, N);  // a fused output's layout
    if (fuse) {
        if (!glu || N % (CHUNK / 2) != 0 || fg.bytes() >= (1ull << 32)) {
            GGML_LOG_ERROR("%s: the fused output layout does not fit the call\n", "xdna-pgemm");
            return false;
        }
        if (fuse->cap < fg.bytes()) {
            if (fuse->bo) {
                xdna_buffer_free(fuse->bo);
            }
            fuse->bo  = xdna_buffer_alloc(pool->device, fg.bytes());
            fuse->cap = fuse->bo ? fg.bytes() : 0;
        }
        if (!fuse->bo) {
            return false;
        }
        fuse->src = nullptr;
    }
    size_t        dst_off = 0;
    xdna_buffer * dst_bo  = fuse ? fuse->bo : ov ? ov->bo : xdna_host_bo_of(node->data, &dst_off);
    if (ov) {
        dst_off = ov->off;
    }
    const size_t m_room = ov ? (size_t) n_blk * MBLK : (size_t) M;
    const bool   direct = fuse || (dst_bo && (ov || M % MBLK == 0) && N % (2 * NC) == 0 &&
                                 (ov || node->nb[1] == (size_t) N * sizeof(float)) &&
                                 dst_off + m_room * N * 4 <= dst_bo->bytes && dst_off + m_room * N * 4 < (1ull << 32));
    if (ov && !direct) {
        GGML_LOG_ERROR("%s: the destination cannot take the output directly\n", "xdna-pgemm");
        return false;
    }
    c_dest cd;
    cd.fused = fuse != nullptr;
    cd.a_blk = (uint32_t) fg.a_blk;
    cd.ldc   = direct ? (uint32_t) N : (uint32_t) ldc_s;
    cd.objs  = objs;
    for (int c = 0; c < COLS; c++) {
        int v = 0;
        for (int o = 0; o < objs; o++) {
            v += o * CHUNK / 2 + c * NC < (direct ? N : ldc_s);
        }
        cd.valid[c] = v;
    }
    const size_t  c_bytes = (size_t) slab * MBLK * ldc_s * 4;
    xdna_buffer * bo_c[2] = {};
    xdna_buffer * sink    = xdna_kernel_pool_acquire_buffer(pool, (size_t) 2 * MBLK * NC * 4);
    xrt::run      runs[2];
    bool          ok = sink != nullptr;
    for (int i = 0; ok && !direct && i < 2 && i < n_sl; i++) {
        bo_c[i] = xdna_kernel_pool_acquire_buffer(pool, c_bytes);
        ok      = bo_c[i] != nullptr;
    }
    if (ok && !g_pg.kern) {
        g_pg.kern = xdna_kernel_find(pool->device, STEM);
        ok        = g_pg.kern != nullptr;
    }
    if (ok && direct && !fuse && !xdna_buffer_sync_to_device_range(dst_bo, (size_t) M * N * 4, dst_off)) {
        // no dirty line of dst may land over what the array writes
        ok = false;
    }
    for (int sl = 0; ok && sl <= n_sl; sl++) {
        if (sl < n_sl) {
            const int nmb = blocks_of(sl);
            c_dest    cs  = cd;
            cs.off        = direct ? (uint32_t) (dst_off + (size_t) sl * slab * MBLK * N * 4) : 0;
            cs.nreal      = std::min(nmb, (M - sl * slab * MBLK + MBLK - 1) / MBLK);
            cs.half[0]    = (uint32_t) fg.half(sl, 0);
            cs.half[1]    = (uint32_t) fg.half(sl, 1);
            // the stream names this slab's place in dst, so it is bound per
            // slab; a started run keeps the instruction buffer it was given
            const std::vector<uint32_t> insts =
                build_seq(g, ws->plan, nmb, cs, (uint32_t) (sl * 2 * HDR_BYTES), (uint32_t) ag.base(sl));
            if (!xdna_kernel_bind_insts(pool->device, g_pg.kern, insts.data(), insts.size())) {
                // xdna-runtime names the stream and the reason
                if (sl > 0) {
                    (void) xdna_run_wait(runs[(sl - 1) % 2]);
                }
                ok = false;
                break;
            }
            if (!a_ready && !pack(sl)) {
                if (sl > 0) {
                    xdna_run_wait(runs[(sl - 1) % 2]);
                }
                ok = false;
                break;
            }
            xdna_buffer * args[4] = { ws->bo, al->bo, direct ? dst_bo : bo_c[sl % 2], sink };
            runs[sl % 2]          = xdna_kernel_run_start(g_pg.kern, args, 4);
        }
        if (sl > 0) {
            const int p = sl - 1;
            if (!xdna_run_wait(runs[p % 2])) {
                ok = false;
                if (sl < n_sl) {
                    (void) xdna_run_wait(runs[sl % 2]);
                }
                break;
            }
            if (!direct && !unpack(bo_c[p % 2], p * slab * MBLK, blocks_of(p) * MBLK)) {
                ok = false;
                if (sl < n_sl) {
                    xdna_run_wait(runs[sl % 2]);
                }
                break;
            }
        }
    }
    if (ok && direct && !ov && !fuse && !xdna_buffer_sync_from_device_range(dst_bo, (size_t) M * N * 4, dst_off)) {
        ok = false;
    }
    if (ok && fuse) {
        fuse->src   = node;
        fuse->data  = node->data;
        fuse->M     = M;
        fuse->K     = N;
        fuse->epoch = g_pg.epoch;
    }
    if (ok && !a_ready) {
        al->src   = a;
        al->data  = a->data;
        al->M     = M;
        al->K     = K;
        al->epoch = g_pg.epoch;
    }
    for (int i = 0; i < 2; i++) {
        if (bo_c[i]) {
            xdna_kernel_pool_release_buffer(pool, bo_c[i]);
        }
    }
    if (sink) {
        xdna_kernel_pool_release_buffer(pool, sink);
    }
    return ok;
}

}  // namespace

bool xdna_pgemm_run(xdna_kernel_pool * pool, ggml_tensor * node) {
    std::lock_guard<std::mutex> lock(g_pg.mtx);
    const ggml_tensor *         w  = node->src[0];
    weights *                   ws = weights_for(pool, w);
    if (!ws) {
        GGML_LOG_ERROR("%s: cannot pack %s\n", "xdna-pgemm", w->name);
        return false;
    }
    return pgemm_call(pool, node->src[1], ws, node, (int) w->ne[1], false);
}

size_t xdna_pgemm_into_rows(int M) {
    return (size_t) a_geom(M, KS).n_blk * MBLK;
}

bool xdna_pgemm_run_into(xdna_kernel_pool * pool, ggml_tensor * node, xdna_buffer * bo, size_t off) {
    std::lock_guard<std::mutex> lock(g_pg.mtx);
    const ggml_tensor *         w  = node->src[0];
    weights *                   ws = weights_for(pool, w);
    if (!ws) {
        GGML_LOG_ERROR("%s: cannot pack %s\n", "xdna-pgemm", w->name);
        return false;
    }
    pg_dst ov;
    ov.bo  = bo;
    ov.off = off;
    return pgemm_call(pool, node->src[1], ws, node, (int) w->ne[1], false, &ov);
}

bool xdna_pgemm_glu_supported(const ggml_tensor * glu) {
    if (!glu || glu->op != GGML_OP_GLU || ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU ||
        xdna_env_int("GGML_XDNA_PGEMM_GLU", 1) == 0 || glu->type != GGML_TYPE_F32 ||
        ggml_get_op_params_i32(glu, 1) != 0 || !ggml_is_contiguous(glu)) {
        return false;
    }
    const ggml_tensor * gate = glu->src[0];
    const ggml_tensor * up   = glu->src[1];
    if (!gate || !up || !xdna_pgemm_supported(gate) || !xdna_pgemm_supported(up) || gate->src[1] != up->src[1] ||
        gate->src[0]->type != up->src[0]->type || !ggml_are_same_shape(gate->src[0], up->src[0]) ||
        !ggml_are_same_shape(gate, glu)) {
        return false;
    }
    const xdna_gemv_geom g = xdna_gemv_variant(gate->src[0]->type, gate->src[0]->ne[0], 2 * gate->src[0]->ne[1], false,
                                               XDNA_GEMV_SPLIT_FUSED, true);
    return g.valid() && g.n_out() <= 256;
}

bool xdna_pgemm_run_glu(xdna_kernel_pool * pool, ggml_tensor * glu) {
    std::lock_guard<std::mutex> lock(g_pg.mtx);
    const ggml_tensor *         gate = glu->src[0];
    const ggml_tensor *         up   = glu->src[1];
    weights *                   ws   = weights_for(pool, gate->src[0], up->src[0]);
    if (!ws) {
        GGML_LOG_ERROR("%s: cannot pack %s + %s\n", "xdna-pgemm", gate->src[0]->name, up->src[0]->name);
        return false;
    }
    return pgemm_call(pool, gate->src[1], ws, glu, (int) gate->src[0]->ne[1], true);
}

bool xdna_pgemm_glu_fused_supported(const ggml_tensor * glu, const ggml_tensor * down) {
    return xdna_env_int("GGML_XDNA_PGEMM_GLU_A", 1) != 0 && xdna_pgemm_glu_supported(glu) && down &&
           down->op == GGML_OP_MUL_MAT && down->src[1] == glu && xdna_pgemm_supported(down) &&
           glu->ne[0] % (CHUNK / 2) == 0 && glu->ne[0] == down->src[0]->ne[0];
}

bool xdna_pgemm_run_glu_fused(xdna_kernel_pool * pool, ggml_tensor * glu) {
    std::lock_guard<std::mutex> lock(g_pg.mtx);
    const ggml_tensor *         gate = glu->src[0];
    const ggml_tensor *         up   = glu->src[1];
    weights *                   ws   = weights_for(pool, gate->src[0], up->src[0]);
    if (!ws) {
        GGML_LOG_ERROR("%s: cannot pack %s + %s\n", "xdna-pgemm", gate->src[0]->name, up->src[0]->name);
        return false;
    }
    return pgemm_call(pool, gate->src[1], ws, glu, (int) gate->src[0]->ne[1], true, nullptr, &g_pg.fused);
}

bool xdna_pgemm_norm_supported(const ggml_tensor * mul) {
    if (!mul || mul->op != GGML_OP_MUL || mul->type != GGML_TYPE_F32 ||
        xdna_env_int("GGML_XDNA_PGEMM_NORM_A", 1) == 0) {
        return false;
    }
    const ggml_tensor * r = mul->src[0];
    const ggml_tensor * g = mul->src[1];
    if (!r || !g || r->op != GGML_OP_RMS_NORM || r->type != GGML_TYPE_F32 || g->type != GGML_TYPE_F32) {
        return false;
    }
    const ggml_tensor * x = r->src[0];
    const int64_t       K = mul->ne[0];
    return x && x->type == GGML_TYPE_F32 && ggml_are_same_shape(x, mul) && ggml_are_same_shape(r, mul) &&
           x->nb[0] == sizeof(float) && ggml_is_contiguous(mul) && ggml_is_contiguous(g) && ggml_nelements(g) == K &&
           mul->ne[2] * mul->ne[3] == 1 && K % KS == 0;
}

bool xdna_pgemm_add_supported(const ggml_tensor * add, const ggml_tensor * mul) {
    if (!add || add->op != GGML_OP_ADD || add->type != GGML_TYPE_F32 || !ggml_is_contiguous(add) ||
        !ggml_are_same_shape(add, mul) || xdna_env_int("GGML_XDNA_PGEMM_ADD", 1) == 0) {
        return false;
    }
    for (const ggml_tensor * t : { add->src[0], add->src[1] }) {
        if (!t || t->type != GGML_TYPE_F32 || !ggml_are_same_shape(t, add) || t->nb[0] != sizeof(float)) {
            return false;
        }
    }
    return true;
}

bool xdna_pgemm_run_norm(xdna_kernel_pool * pool, ggml_tensor * mul, ggml_tensor * add) {
    std::lock_guard<std::mutex> lock(g_pg.mtx);
    const ggml_tensor *         r = mul->src[0];
    const ggml_tensor *         x = r->src[0];
    const float *               g = (const float *) mul->src[1]->data;
    float                       eps;
    std::memcpy(&eps, r->op_params, sizeof(float));
    const int    M = (int) mul->ne[1], K = (int) mul->ne[0], nstep = K / KS;
    const a_geom ag(M, K);
    a_layout &   al = g_pg.pre;
    if (al.cap < ag.bytes()) {
        if (al.bo) {
            xdna_buffer_free(al.bo);
        }
        al.bo  = xdna_buffer_alloc(pool->device, ag.bytes());
        al.cap = al.bo ? ag.bytes() : 0;
    }
    al.src = nullptr;
    if (!al.bo || !al.bo->data) {
        return false;
    }
    for (int sl = 0; sl < ag.n_sl; sl++) {
        if (!lay_slab(ag, al.bo, sl, M, K, [&](int m, uint16_t * o, int sw) {
                const float * row = (const float *) ((const char *) x->data + (size_t) m * x->nb[1]);
                if (add) {
                    // the residual sum the norm reads, written for its other
                    // readers and normed from there
                    const float * a0 =
                        (const float *) ((const char *) add->src[0]->data + (size_t) m * add->src[0]->nb[1]);
                    const float * src1 =
                        (const float *) ((const char *) add->src[1]->data + (size_t) m * add->src[1]->nb[1]);
                    float * sum = (float *) ((char *) add->data + (size_t) m * add->nb[1]);
                    xdna_add_row(a0, src1, sum, K);
                    row = sum;
                }
                const float sc = xdna_rms_scale(row, K, eps);
                for (int st = 0; st < nstep; st++) {
                    const size_t k0 = (size_t) (st ^ sw) * KS;
                    bf16_row_norm(row + k0, g + k0, sc, o + (size_t) st * MB * KS, KS);
                }
            })) {
            return false;
        }
    }
    al.src   = mul;
    al.data  = mul->data;
    al.M     = M;
    al.K     = K;
    al.epoch = g_pg.epoch;
    g_pg.unwritten.push_back(mul);
    return true;
}

bool xdna_pgemm_gated_supported(const ggml_tensor * g, const ggml_tensor * v) {
    if (!g || !v || g->op != GGML_OP_MUL || g->type != GGML_TYPE_F32 || !ggml_is_contiguous(g) ||
        xdna_env_int("GGML_XDNA_PGEMM_GATED_A", 1) == 0) {
        return false;
    }
    const ggml_tensor * m  = g->src[0];
    const ggml_tensor * sl = g->src[1];
    if (!m || !sl || m->op != GGML_OP_MUL || sl->op != GGML_OP_UNARY || ggml_get_unary_op(sl) != GGML_UNARY_OP_SILU) {
        return false;
    }
    const ggml_tensor * r = m->src[0];
    const ggml_tensor * w = m->src[1];
    const ggml_tensor * z = sl->src[0];
    if (!r || !w || !z || r->op != GGML_OP_RMS_NORM) {
        return false;
    }
    const ggml_tensor * x = r->src[0];
    const int64_t       D = g->ne[0], K = g->ne[0] * g->ne[1];
    return x && x->type == GGML_TYPE_F32 && z->type == GGML_TYPE_F32 && w->type == GGML_TYPE_F32 &&
           m->type == GGML_TYPE_F32 && sl->type == GGML_TYPE_F32 && ggml_are_same_shape(x, g) &&
           ggml_are_same_shape(z, g) && ggml_are_same_shape(m, g) && x->nb[0] == sizeof(float) &&
           z->nb[0] == sizeof(float) && ggml_is_contiguous(w) && w->ne[0] == D && ggml_nrows(w) == 1 && g->ne[3] == 1 &&
           D % 16 == 0 && K % KS == 0 && K <= MAX_K && v->op == GGML_OP_RESHAPE && v->src[0] == g && v->ne[0] == K &&
           v->ne[1] == g->ne[2] && ggml_nelements(v) == ggml_nelements(g) && v->ne[1] >= MB;
}

bool xdna_pgemm_run_gated(xdna_kernel_pool *    pool,
                          const ggml_tensor *   g,
                          const ggml_tensor *   v,
                          const xdna_gdn_rows * xr) {
    std::lock_guard<std::mutex> lock(g_pg.mtx);
    const ggml_tensor *         m = g->src[0];
    const ggml_tensor *         r = m->src[0];
    const ggml_tensor *         x = r->src[0];
    const ggml_tensor *         z = g->src[1]->src[0];
    const float *               w = (const float *) m->src[1]->data;
    float                       eps;
    std::memcpy(&eps, r->op_params, sizeof(float));
    const int    D = (int) g->ne[0], H = (int) g->ne[1];
    const int    M = (int) g->ne[2], K = D * H, nstep = K / KS;
    const a_geom ag(M, K);
    a_layout &   al = g_pg.pre;
    if (al.cap < ag.bytes()) {
        if (al.bo) {
            xdna_buffer_free(al.bo);
        }
        al.bo  = xdna_buffer_alloc(pool->device, ag.bytes());
        al.cap = al.bo ? ag.bytes() : 0;
    }
    al.src = nullptr;
    if (!al.bo || !al.bo->data) {
        return false;
    }
    for (int sl = 0; sl < ag.n_sl; sl++) {
        if (!lay_slab(ag, al.bo, sl, M, K, [&](int t, uint16_t * o, int sw) {
                alignas(64) float row[MAX_K];
                for (int h = 0; h < H; h++) {
                    const float * zr =
                        (const float *) ((const char *) z->data + (size_t) h * z->nb[1] + (size_t) t * z->nb[2]);
                    if (xr) {
                        // the rows where the GDN run left them (xdna_gdn_rows): a
                        // head's two value halves in two objects
                        const float * p0 = xr->base + (size_t) (h / 2) * xr->o_col +
                                           ((size_t) (t / 16) * 4 + (size_t) 2 * (h % 2)) * 1024 +
                                           (size_t) (t % 16) * 64;
                        const float * p1 = p0 + 1024;
                        const float   sc = xdna_rms_scale2(p0, p1, D / 2, eps);
                        xdna_gated_row(p0, w, sc, zr, row + (size_t) h * D, D / 2);
                        xdna_gated_row(p1, w + D / 2, sc, zr + D / 2, row + (size_t) h * D + D / 2, D / 2);
                    } else {
                        const float * xp =
                            (const float *) ((const char *) x->data + (size_t) h * x->nb[1] + (size_t) t * x->nb[2]);
                        xdna_gated_row(xp, w, xdna_rms_scale(xp, D, eps), zr, row + (size_t) h * D, D);
                    }
                }
                for (int st = 0; st < nstep; st++) {
                    bf16_row(row + (size_t) (st ^ sw) * KS, o + (size_t) st * MB * KS, KS);
                }
            })) {
            return false;
        }
    }
    al.src   = v;
    al.data  = v->data;
    al.M     = M;
    al.K     = K;
    al.epoch = g_pg.epoch;
    g_pg.unwritten.push_back(v);
    return true;
}

namespace {

// the gate a SIGMOID reads: its CONT's source when the CONT only copies it
const ggml_tensor * gate_src(const ggml_tensor * sg) {
    const ggml_tensor * c = sg ? sg->src[0] : nullptr;
    return c && c->op == GGML_OP_CONT ? c->src[0] : c;
}

}  // namespace

bool xdna_pgemm_gate_supported(const ggml_tensor * g) {
    if (!g || g->op != GGML_OP_MUL || g->type != GGML_TYPE_F32 || !ggml_is_contiguous(g) ||
        xdna_env_int("GGML_XDNA_PGEMM_GATE_A", 1) == 0) {
        return false;
    }
    const ggml_tensor * a  = g->src[0];
    const ggml_tensor * sg = g->src[1];
    if (!a || !sg || sg->op != GGML_OP_UNARY || ggml_get_unary_op(sg) != GGML_UNARY_OP_SIGMOID ||
        sg->type != GGML_TYPE_F32 || !sg->src[0] || sg->src[0]->type != GGML_TYPE_F32) {
        return false;
    }
    const ggml_tensor * gv = gate_src(sg);
    const int64_t       K = g->ne[0], M = g->ne[1];
    if (!gv || gv->type != GGML_TYPE_F32 || gv->nb[0] != sizeof(float) || gv->ne[3] != 1 ||
        ggml_nelements(gv) != K * M || gv->ne[0] % 8 != 0) {
        return false;
    }
    // the gate's rows: [K, M], or [K / H, H, M] (a view that skips the Q
    // columns between the heads)
    const bool rows2 = gv->ne[0] == K && gv->ne[1] == M && gv->ne[2] == 1;
    const bool rows3 = gv->ne[0] * gv->ne[1] == K && gv->ne[2] == M;
    return (rows2 || rows3) && a->type == GGML_TYPE_F32 && ggml_are_same_shape(a, g) && a->nb[0] == sizeof(float) &&
           ggml_are_same_shape(sg, g) && g->ne[2] * g->ne[3] == 1 && K % KS == 0 && K <= MAX_K && M >= MB;
}

bool xdna_pgemm_run_gate(xdna_kernel_pool * pool, const ggml_tensor * g, const xdna_attn_rows * ar_rows) {
    std::lock_guard<std::mutex> lock(g_pg.mtx);
    const ggml_tensor *         a  = g ? g->src[0] : nullptr;
    const ggml_tensor *         gv = a ? gate_src(g->src[1]) : nullptr;
    if (!a || !gv) {
        return false;
    }
    const int    M = (int) g->ne[1], K = (int) g->ne[0], nstep = K / KS;
    const bool   rows2 = gv->ne[0] == K && gv->ne[1] == M && gv->ne[2] == 1;
    const int    seg   = rows2 ? K : (int) gv->ne[0];
    const a_geom ag(M, K);
    a_layout &   al = g_pg.pre;
    if (al.cap < ag.bytes()) {
        if (al.bo) {
            xdna_buffer_free(al.bo);
        }
        al.bo  = xdna_buffer_alloc(pool->device, ag.bytes());
        al.cap = al.bo ? ag.bytes() : 0;
    }
    al.src = nullptr;
    if (!al.bo || !al.bo->data) {
        return false;
    }
    for (int sl = 0; sl < ag.n_sl; sl++) {
        if (!lay_slab(ag, al.bo, sl, M, K, [&](int t, uint16_t * o, int sw) {
                alignas(64) float row[MAX_K];
                const float *     ar = (const float *) ((const char *) a->data + (size_t) t * a->nb[1]);
                for (int e = 0; e < K; e += seg) {
                    const float * gr = rows2 ?
                                           (const float *) ((const char *) gv->data + (size_t) t * gv->nb[1]) + e :
                                           (const float *) ((const char *) gv->data + (size_t) (e / seg) * gv->nb[1] +
                                                            (size_t) t * gv->nb[2]);
                    if (ar_rows) {
                        // the attention output where its run left it: a head's
                        // 256 as two halves (xdna_attn_mm_row)
                        for (int e2 = e; e2 < e + seg; e2 += 128) {
                            const float * h0 = xdna_attn_mm_row(ar_rows, t, e2 / 256);
                            xdna_gate_row(h0 + (e2 % 256 ? 4096 : 0), gr + (e2 - e), row + e2, 128);
                        }
                    } else {
                        xdna_gate_row(ar + e, gr, row + e, seg);
                    }
                }
                for (int st = 0; st < nstep; st++) {
                    bf16_row(row + (size_t) (st ^ sw) * KS, o + (size_t) st * MB * KS, KS);
                }
            })) {
            return false;
        }
    }
    al.src   = g;
    al.data  = g->data;
    al.M     = M;
    al.K     = K;
    al.epoch = g_pg.epoch;
    g_pg.unwritten.push_back(g);
    return true;
}

bool xdna_pgemm_pair_supported(const ggml_tensor * a, const ggml_tensor * b) {
    if (!a || !b || a == b || !xdna_pgemm_supported(a) || !xdna_pgemm_supported(b) || a->src[1] != b->src[1] ||
        a->src[0]->type != b->src[0]->type || a->src[1]->ne[2] * a->src[1]->ne[3] != 1) {
        return false;  // the pair counts its rows in ne[1]
    }
    // one chunk for both, where each would stream all of A for its own; a
    // wider pair would give up C written straight into its nodes
    return a->src[0]->ne[1] + b->src[0]->ne[1] <= CHUNK;
}

bool xdna_pgemm_run_pair(xdna_kernel_pool *  pool,
                         const ggml_tensor * a,
                         const ggml_tensor * b,
                         float *             a_out,
                         float *             b_out) {
    std::lock_guard<std::mutex> lock(g_pg.mtx);
    weights *                   ws = weights_for(pool, a->src[0], b->src[0], true);
    if (!ws) {
        GGML_LOG_ERROR("%s: cannot pack %s + %s\n", "xdna-pgemm", a->src[0]->name, b->src[0]->name);
        return false;
    }
    // C straight into a buffer of rows only as wide as the pair, 128 columns:
    // the array writes the rest of the chunk to the sink
    const int     na = (int) a->src[0]->ne[1], nb = (int) b->src[0]->ne[1];
    const int     ldc  = (na + nb + 2 * NC - 1) / (2 * NC) * (2 * NC);
    const int     M    = (int) a->src[1]->ne[1];
    // as many rows as the call writes: an even count of blocks (a_geom). A
    // whole block only, as this was, refused every M under 129 - the warm-up
    // and every short prompt - with "the destination cannot take the output
    // directly", 18 times a graph on the 0.8B, and ran the pair one by one.
    const size_t  rows = xdna_pgemm_into_rows(M);
    xdna_buffer * bo   = xdna_kernel_pool_acquire_buffer(pool, rows * ldc * 4);
    if (!bo || !bo->data) {
        return false;
    }
    pg_dst ov;
    ov.bo   = bo;
    bool ok = pgemm_call(pool, a->src[1], ws, a, ldc, false, &ov);
    if (ok) {
        ok = xdna_buffer_sync_from_device_range(bo, (size_t) M * ldc * 4, 0);
    }
    if (ok) {
        const float * c = (const float *) bo->data;
        for (int m = 0; m < M; m++) {
            std::memcpy(a_out + (size_t) m * na, c + (size_t) m * ldc, (size_t) na * sizeof(float));
            std::memcpy(b_out + (size_t) m * nb, c + (size_t) m * ldc + na, (size_t) nb * sizeof(float));
        }
    }
    xdna_kernel_pool_release_buffer(pool, bo);
    return ok;
}
