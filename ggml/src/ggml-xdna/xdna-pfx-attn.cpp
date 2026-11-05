#include "xdna-pfx-attn.h"

#include "ggml-impl.h"
#include "xdna-norm.h"
#include "xdna-pgemm-path.h"
#include "xdna-prof.h"
#include "xdna-runtime.h"
#include "xdna-seq.h"
#include "xdna-util.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

// Host side of kernels/pfx.py's attention mode. Four arguments, pgemm's:
//   0  W  per attention column, per pass, its four cores' Q half^T
//         (d/8, row/8, 8, 8) bf16 and trailer, TB bytes each
//   1  A  per D half (stream A0 half a, A1 half b): pgemm's header (word 8
//         the attention on, word 10 the A objects a row-op core lets pass),
//         the attention header (word 0 the passes), then per KV head and
//         32-key tile its K and V^T, each packed so that the A MemTile's
//         tiling hands the cores attn_mm's (key/8, d/8, 8, 8) and
//         (d/8, key/8, 8, 8)
//   2  C  per attention column, per pass, the C join's two objects of its
//         four cores' O half^T
//   3  unused (pgemm's sink)
// Core (c, 2 + i) of attention column ci is pair 2 ci + i / 2, D half i % 2;
// the pair owns 32 rows: row = head * pp + position, pp = 32 / hg, the
// positions pidx * pp .. + pp - 1 of its pass.

namespace {

using namespace xdna_pgemm_path;

constexpr int          ATTN_COLS[] = { 0, 1, 2, 4, 5, 6 };  // kernels/pfx.py: row ops on 3 and 7
constexpr int          N_ACOLS     = 6;
constexpr int          PAIRS       = 2 * N_ACOLS;
constexpr int          KVH         = 2;
constexpr int          DH          = 128;           // a core's half of D
constexpr int          R           = 32;            // rows a pair
constexpr int          NK          = 32;            // keys a tile
constexpr int          A_ELEMS     = 64 * 64;       // bf16 an A object: one K or V^T tile half
constexpr int          TRAILER_POS = R * DH;        // bf16 offset of a Q object's trailer
constexpr uint32_t     TB_BYTES    = 10752;         // a core's W buffer, the length of every W send
constexpr int          C_F32       = 64 * 32;       // f32 a C object
constexpr int          O_OBJS      = 2;             // C objects a core's O half
constexpr int          HDR_WORDS   = 16;            // kernels/pfx.py HDR_WORDS
constexpr int          HW_ATTN_ON  = 8;
constexpr int          HW_A_DRAIN  = 10;
constexpr uint32_t     W_PKT_CORE0 = 2;             // kernels/pgemm.py W_PKT_CORE: core i's id is 2 + i
constexpr int          W_SLOTS     = 2;             // MemTile slots of a pass's four Q objects
constexpr uint32_t     SH_BD_A_HDR = 0;
constexpr uint32_t     SH_BD_KV    = 10;            // the A shims' K/V descriptors, a ring
constexpr int          KV_RING     = 3;
constexpr int          W_RING      = 2;
constexpr int          N_ARGS      = 4;
constexpr size_t       SINK_BYTES  = 4096;

struct pass {
    int     g       = 0;  // KV head
    int64_t t0      = 0;  // first position
    int64_t tile_lo = 0, tile_hi = 0;
};

struct runner {
    std::mutex    mtx;
    xdna_kernel * kern = nullptr;
};

runner g_pfx;

int64_t positions_a_pass(const xdna_pfx_attn_job & job) {
    return (int64_t) PAIRS * (R / job.hg);
}

int64_t tiles_of(const xdna_pfx_attn_job & job) {
    return (job.n_kv + NK - 1) / NK;
}

pass pass_of(const xdna_pfx_attn_job & job, int g, int64_t t0) {
    const int64_t t1 = std::min(job.n_tokens, t0 + positions_a_pass(job));
    int32_t       lo = job.lo[t0], hi = job.hi[t0];
    for (int64_t t = t0 + 1; t < t1; t++) {
        lo = std::min(lo, job.lo[t]);
        hi = std::max(hi, job.hi[t]);
    }
    return { g, t0, lo / NK, hi / NK + 1 };
}

std::vector<pass> plan_passes(const xdna_pfx_attn_job & job) {
    std::vector<pass> ps;
    for (int g = 0; g < KVH; g++) {
        for (int64_t t0 = 0; t0 < job.n_tokens; t0 += positions_a_pass(job)) {
            ps.push_back(pass_of(job, g, t0));
        }
    }
    return ps;
}

// A stream: object 0 pgemm's header, 1 the attention header, then per KV head and tile its K, V^T.
size_t kv_obj(int64_t n_tiles, int g, int64_t t) {
    return 2 + ((size_t) g * (size_t) n_tiles + (size_t) t) * 2;
}

size_t a_stream_objs(int64_t n_tiles) {
    return kv_obj(n_tiles, KVH, 0);
}

// A header object goes through the A MemTile's tiling like the tiles: the core's bf16 j is sent from row j / 8,
// column j % 8 of the 64 x 64 object (as xdna-pgemm.cpp writes pgemm's).
void write_tiled_words(const int32_t * words, int n_words, uint16_t * obj) {
    uint16_t hb[2 * HDR_WORDS];
    std::memcpy(hb, words, (size_t) n_words * sizeof(int32_t));
    for (int j = 0; j < 2 * n_words; j++) {
        obj[(j / 8) * 64 + j % 8] = hb[j];
    }
}

int32_t drained_objects(const std::vector<pass> & ps) {
    int32_t drain = 1;  // the attention header
    for (const pass & p : ps) {
        drain += (int32_t) (2 * (p.tile_hi - p.tile_lo));
    }
    return drain;
}

void pack_headers(const std::vector<pass> & ps, uint16_t * stream) {
    int32_t pg[HDR_WORDS] = {};
    pg[HW_ATTN_ON]        = 1;
    pg[HW_A_DRAIN]        = drained_objects(ps);
    write_tiled_words(pg, HDR_WORDS, stream);
    const int32_t fa[1] = { (int32_t) ps.size() };
    write_tiled_words(fa, 1, stream + A_ELEMS);
}

// Tile t of KV head g, D half h, as f16 rows converted to bf16 (zero past n_kv).
void read_tile(const ggml_tensor * x, const xdna_pfx_attn_job & job, int g, int h, int64_t t, uint16_t (*rows)[DH]) {
    for (int key = 0; key < NK; key++) {
        const int64_t j = t * NK + key;
        if (j >= job.n_kv) {
            std::memset(rows[key], 0, sizeof(rows[key]));
            continue;
        }
        const uint16_t * src = (const uint16_t *) ((const char *) x->data + j * x->nb[1] + g * x->nb[2]);
        xdna_f16_to_bf16_row(src + (size_t) h * DH, rows[key], DH);
    }
}

// The host object the A path delivers as attn_mm's K tile (key/8, d/8, 8, 8): row 16 (key / 8) + 8 (d / 64) +
// key % 8, column d % 64.
void scatter_k(const uint16_t (*kb)[DH], uint16_t * o) {
    for (int key = 0; key < NK; key++) {
        for (int d = 0; d < DH; d++) {
            o[(16 * (key / 8) + 8 * (d / 64) + key % 8) * 64 + d % 64] = kb[key][d];
        }
    }
}

// The same for V^T (d/8, key/8, 8, 8): row 8 (d / 16) + d % 8, column 32 ((d / 8) % 2) + key.
void scatter_v(const uint16_t (*vb)[DH], uint16_t * o) {
    for (int key = 0; key < NK; key++) {
        for (int d = 0; d < DH; d++) {
            o[(8 * (d / 16) + d % 8) * 64 + 32 * ((d / 8) % 2) + key] = vb[key][d];
        }
    }
}

void pack_kv(const xdna_pfx_attn_job & job, uint16_t * a, size_t stream_elems) {
    const int64_t n_tiles = tiles_of(job);
#pragma omp parallel for collapse(3) num_threads(xdna_chunk_threads())
    for (int h = 0; h < 2; h++) {
        for (int g = 0; g < KVH; g++) {
            for (int64_t t = 0; t < n_tiles; t++) {
                alignas(64) uint16_t rows[NK][DH];
                uint16_t *           s = a + (size_t) h * stream_elems;
                read_tile(job.k, job, g, h, t, rows);
                scatter_k(rows, s + kv_obj(n_tiles, g, t) * A_ELEMS);
                read_tile(job.v, job, g, h, t, rows);
                scatter_v(rows, s + (kv_obj(n_tiles, g, t) + 1) * A_ELEMS);
            }
        }
    }
}

void pack_a(const xdna_pfx_attn_job & job, const std::vector<pass> & ps, uint16_t * a) {
    const size_t stream_elems = a_stream_objs(tiles_of(job)) * A_ELEMS;
    std::memset(a, 0, 2 * stream_elems * sizeof(uint16_t));
    pack_headers(ps, a);
    pack_headers(ps, a + stream_elems);
    pack_kv(job, a, stream_elems);
}

int64_t row_pos(const xdna_pfx_attn_job & job, const pass & p, int pidx, int row) {
    const int pp = R / job.hg;
    return p.t0 + (int64_t) pidx * pp + row % pp;
}

int head_of(const xdna_pfx_attn_job & job, const pass & p, int row) {
    return job.hg * p.g + row / (R / job.hg);
}

// Core i's Q half^T and trailer: the first key the pass streams, its tiles, then each row's first and last
// visible key; past the ubatch the rows are zero and see the first key.
void pack_q_obj(const xdna_pfx_attn_job & job, const pass & p, int pidx, int h, uint16_t * o) {
    const float   qs        = (float) (M_LOG2E * job.q_scale);
    int32_t *     tr        = (int32_t *) (o + TRAILER_POS);
    const int32_t first_key = (int32_t) (p.tile_lo * NK);
    tr[0]                   = first_key;
    tr[1]                   = (int32_t) (p.tile_hi - p.tile_lo);
    alignas(64) uint16_t qb[R][DH];
    for (int row = 0; row < R; row++) {
        const int64_t t    = row_pos(job, p, pidx, row);
        const bool    real = t < job.n_tokens;
        tr[2 + row]        = real ? job.lo[t] : first_key;
        tr[2 + R + row]    = real ? job.hi[t] : first_key;
        if (!real) {
            std::memset(qb[row], 0, sizeof(qb[row]));
            continue;
        }
        const float * src = (const float *) ((const char *) job.q->data + t * job.q->nb[1] +
                                             head_of(job, p, row) * job.q->nb[2]);
        xdna_scaled_bf16_row(src + (size_t) h * DH, qs, qb[row], DH);
    }
    for (int d = 0; d < DH; d++) {
        for (int row = 0; row < R; row++) {
            o[((d / 8) * (R / 8) + row / 8) * 64 + (d % 8) * 8 + row % 8] = qb[row][d];
        }
    }
}

size_t w_obj_off(size_t npass, int ci, size_t p, int i) {
    return (((size_t) ci * npass + p) * 4 + (size_t) i) * TB_BYTES;
}

void pack_w(const xdna_pfx_attn_job & job, const std::vector<pass> & ps, uint8_t * w) {
    std::memset(w, 0, w_obj_off(ps.size(), N_ACOLS, 0, 0));
#pragma omp parallel for collapse(2) num_threads(xdna_chunk_threads())
    for (int ci = 0; ci < N_ACOLS; ci++) {
        for (size_t p = 0; p < ps.size(); p++) {
            for (int i = 0; i < 4; i++) {
                pack_q_obj(job, ps[p], 2 * ci + i / 2, i % 2, (uint16_t *) (w + w_obj_off(ps.size(), ci, p, i)));
            }
        }
    }
}

size_t c_col_f32(size_t npass) {
    return npass * O_OBJS * 4 * C_F32;
}

// C object h of a core holds its O half^T's d = 16 (j / 8) + 8 h + j % 8 at row j, the pair's rows as columns.
void unpack_core(const xdna_pfx_attn_job & job, const pass & p, int pidx, int half, const float * objs) {
    for (int h = 0; h < O_OBJS; h++) {
        const float * o = objs + (size_t) h * 4 * C_F32;
        for (int row = 0; row < R; row++) {
            const int64_t t = row_pos(job, p, pidx, row);
            if (t >= job.n_tokens) {
                continue;
            }
            float * dst = (float *) ((char *) job.out->data + head_of(job, p, row) * job.out->nb[1] +
                                     t * job.out->nb[2]) +
                          (size_t) half * DH;
            for (int j = 0; j < 64; j++) {
                dst[16 * (j / 8) + 8 * h + j % 8] = o[j * 32 + row];
            }
        }
    }
}

void unpack_o(const xdna_pfx_attn_job & job, const std::vector<pass> & ps, const float * c) {
#pragma omp parallel for collapse(2) num_threads(xdna_chunk_threads())
    for (int ci = 0; ci < N_ACOLS; ci++) {
        for (size_t p = 0; p < ps.size(); p++) {
            const float * step = c + ci * c_col_f32(ps.size()) + p * O_OBJS * 4 * C_F32;
            for (int i = 0; i < 4; i++) {
                unpack_core(job, ps[p], 2 * ci + i / 2, i % 2, step + (size_t) i * C_F32);
            }
        }
    }
}

// The stream's W part for one pass on attention column ci: the four Q objects into a MemTile slot, then each to
// its core under its own packet id, pair r's two on MM2S 4 + r. The rings wait on the task of two (shim) or
// MT_Q (MemTile) pushes before, which only needs data queued earlier.
struct w_state {
    int shim_pushed = 0;
    int mt_s2mm     = 0;
    int mt_mm2s[2]  = { 0, 0 };
};

void wait_mt_queue(xdna_seq * seq, uint32_t c, xdna_dma_dir dir, uint32_t ch, int pushed) {
    if (pushed >= MT_Q) {
        xdna_seq_wait_token(seq, c, 1, dir, ch);
    }
}

void w_shim_send(xdna_seq * seq, uint32_t c, uint32_t ddr_off, w_state & ws) {
    if (ws.shim_pushed >= W_RING) {
        xdna_seq_wait_token(seq, c, 0, xdna_dma_dir::MM2S, W_SHIM_CH);
    }
    const uint32_t sid = SH_BD_W + (uint32_t) (ws.shim_pushed % W_RING);
    xdna_bd        bd  = linear_bd(4 * TB_BYTES / 4);
    xdna_seq_blockwrite(seq, c, 0, sid, &bd);
    xdna_seq_ddr_patch(seq, c, 0, sid, 0, ddr_off);
    xdna_seq_issue_token(seq, c, 0, xdna_dma_dir::MM2S, W_SHIM_CH, 0xF);
    xdna_seq_push_queue(seq, c, 0, sid, xdna_dma_dir::MM2S, W_SHIM_CH, true, 0);
    ws.shim_pushed++;
}

void w_core_sends(xdna_seq * seq, uint32_t c, uint32_t at, uint32_t e, uint32_t f, w_state & ws) {
    for (uint32_t i = 0; i < 4; i++) {
        const uint32_t r = i / 2;
        wait_mt_queue(seq, c, xdna_dma_dir::MM2S, W_MT_MM2S + r, ws.mt_mm2s[r]);
        const uint32_t id  = (r ? MT_BD_MM2S1 : MT_BD_MM2S0) + (uint32_t) (ws.mt_mm2s[r] % MT_Q);
        const uint32_t pkt = MT_BD_PKT | (W_PKT_CORE0 + i) << MT_BD_PKT_ID_SHIFT;
        mt_bd(seq, c, id, TB_BYTES / 4, at + i * TB_BYTES, f, 1, e, 1, pkt);
        mt_push(seq, c, id, xdna_dma_dir::MM2S, W_MT_MM2S + r, 0);
        ws.mt_mm2s[r]++;
    }
}

void w_pass(xdna_seq * seq, size_t npass, int ci, size_t p, w_state & ws) {
    const uint32_t c = (uint32_t) ATTN_COLS[ci], k = (uint32_t) (p % W_SLOTS);
    const uint32_t e = W_LOCK0 + 2 * k, f = e + 1, at = WS_ADDR + k * 4 * TB_BYTES;
    w_shim_send(seq, c, (uint32_t) w_obj_off(npass, ci, p, 0), ws);
    wait_mt_queue(seq, c, xdna_dma_dir::S2MM, W_MT_S2MM, ws.mt_s2mm);
    const uint32_t s2 = MT_BD_S2MM + (uint32_t) (ws.mt_s2mm % MT_Q);
    mt_bd(seq, c, s2, 4 * TB_BYTES / 4, at, e, 4, f, 4);
    mt_push(seq, c, s2, xdna_dma_dir::S2MM, W_MT_S2MM, 0);
    ws.mt_s2mm++;
    w_core_sends(seq, c, at, e, f, ws);
}

void wait_last(xdna_seq * seq, uint32_t c, uint32_t row, xdna_dma_dir dir, uint32_t ch, int pushed, int ring) {
    for (int k = std::max(0, pushed - ring); k < pushed; k++) {
        xdna_seq_wait_token(seq, c, row, dir, ch);
    }
}

void w_finish(xdna_seq * seq, int ci, const w_state & ws) {
    const uint32_t c = (uint32_t) ATTN_COLS[ci];
    wait_last(seq, c, 1, xdna_dma_dir::S2MM, W_MT_S2MM, ws.mt_s2mm, MT_Q);
    for (uint32_t r = 0; r < 2; r++) {
        wait_last(seq, c, 1, xdna_dma_dir::MM2S, W_MT_MM2S + r, ws.mt_mm2s[r], MT_Q);
    }
    wait_last(seq, c, 0, xdna_dma_dir::MM2S, W_SHIM_CH, ws.shim_pushed, W_RING);
}

void a_headers(xdna_seq * seq, size_t stream_bytes) {
    for (uint32_t h = 0; h < 2; h++) {
        xdna_bd bd = linear_bd(2 * A_ELEMS * 2 / 4);
        xdna_seq_blockwrite(seq, A_COL0 + h, 0, SH_BD_A_HDR, &bd);
        xdna_seq_ddr_patch(seq, A_COL0 + h, 0, SH_BD_A_HDR, 1, (uint32_t) (h * stream_bytes));
        xdna_seq_push_queue(seq, A_COL0 + h, 0, SH_BD_A_HDR, xdna_dma_dir::MM2S, 0, false, 0);
    }
}

void a_pass(xdna_seq * seq, int64_t n_tiles, const pass & p, size_t stream_bytes, int n) {
    for (uint32_t h = 0; h < 2; h++) {
        if (n >= KV_RING) {
            xdna_seq_wait_token(seq, A_COL0 + h, 0, xdna_dma_dir::MM2S, 0);
        }
        const uint32_t id = SH_BD_KV + (uint32_t) (n % KV_RING);
        xdna_bd        bd = linear_bd((uint32_t) (2 * (p.tile_hi - p.tile_lo) * A_ELEMS * 2 / 4));
        xdna_seq_blockwrite(seq, A_COL0 + h, 0, id, &bd);
        xdna_seq_ddr_patch(seq, A_COL0 + h, 0, id, 1,
                           (uint32_t) (h * stream_bytes + kv_obj(n_tiles, p.g, p.tile_lo) * A_ELEMS * 2));
        xdna_seq_issue_token(seq, A_COL0 + h, 0, xdna_dma_dir::MM2S, 0, 0xF);
        xdna_seq_push_queue(seq, A_COL0 + h, 0, id, xdna_dma_dir::MM2S, 0, true, 0);
    }
}

void c_drains(xdna_seq * seq, size_t npass) {
    for (int ci = 0; ci < N_ACOLS; ci++) {
        const uint32_t c  = (uint32_t) ATTN_COLS[ci];
        xdna_bd        bd = linear_bd((uint32_t) c_col_f32(npass));
        xdna_seq_blockwrite(seq, c, 0, SH_BD_C, &bd);
        xdna_seq_ddr_patch(seq, c, 0, SH_BD_C, 2, (uint32_t) (ci * c_col_f32(npass) * 4));
        xdna_seq_issue_token(seq, c, 0, xdna_dma_dir::S2MM, 0, 0xF);
        xdna_seq_push_queue(seq, c, 0, SH_BD_C, xdna_dma_dir::S2MM, 0, true, 0);
    }
}

void init_slot_locks(xdna_seq * seq) {
    for (int ci = 0; ci < N_ACOLS; ci++) {
        for (uint32_t k = 0; k < W_SLOTS; k++) {
            mt_set_lock(seq, (uint32_t) ATTN_COLS[ci], W_LOCK0 + 2 * k, 4);
            mt_set_lock(seq, (uint32_t) ATTN_COLS[ci], W_LOCK0 + 2 * k + 1, 0);
        }
    }
}

void stream_passes(xdna_seq * seq, int64_t n_tiles, const std::vector<pass> & ps, size_t stream_bytes,
                   std::vector<w_state> & ws) {
    for (size_t p = 0; p < ps.size(); p++) {
        for (int ci = 0; ci < N_ACOLS; ci++) {
            w_pass(seq, ps.size(), ci, p, ws[ci]);
        }
        a_pass(seq, n_tiles, ps[p], stream_bytes, (int) p);
    }
}

void finish_stream(xdna_seq * seq, const std::vector<pass> & ps, const std::vector<w_state> & ws) {
    for (int ci = 0; ci < N_ACOLS; ci++) {
        w_finish(seq, ci, ws[ci]);
    }
    for (uint32_t h = 0; h < 2; h++) {
        wait_last(seq, A_COL0 + h, 0, xdna_dma_dir::MM2S, 0, (int) ps.size(), KV_RING);
    }
    for (int ci = 0; ci < N_ACOLS; ci++) {
        xdna_seq_wait_token(seq, (uint32_t) ATTN_COLS[ci], 0, xdna_dma_dir::S2MM, 0);
    }
}

std::vector<uint32_t> build_seq(int64_t n_tiles, const std::vector<pass> & ps) {
    xdna_seq             seq;
    const size_t         stream_bytes = a_stream_objs(n_tiles) * A_ELEMS * 2;
    std::vector<w_state> ws(N_ACOLS);
    init_slot_locks(&seq);
    a_headers(&seq, stream_bytes);
    c_drains(&seq, ps.size());
    stream_passes(&seq, n_tiles, ps, stream_bytes, ws);
    finish_stream(&seq, ps, ws);
    return xdna_seq_build(&seq);
}

xdna_kernel * kernel_of(xdna_device * dev) {
    if (!g_pfx.kern) {
        xdna_kernel * base = xdna_kernel_find(dev, PFX_STEM);
        g_pfx.kern         = base ? xdna_kernel_clone(base) : nullptr;
    }
    return g_pfx.kern;
}

struct buffers {
    xdna_buffer * w = nullptr, *a = nullptr, *c = nullptr, *sink = nullptr;
    size_t        w_bytes = 0, a_bytes = 0, c_bytes = 0;
};

bool acquire(xdna_kernel_pool * pool, const xdna_pfx_attn_job & job, const std::vector<pass> & ps, buffers & b) {
    b.w_bytes = w_obj_off(ps.size(), N_ACOLS, 0, 0);
    b.a_bytes = 2 * a_stream_objs(tiles_of(job)) * A_ELEMS * 2;
    b.c_bytes = N_ACOLS * c_col_f32(ps.size()) * 4;
    b.w       = xdna_kernel_pool_acquire_buffer(pool, b.w_bytes);
    b.a       = xdna_kernel_pool_acquire_buffer(pool, b.a_bytes);
    b.c       = xdna_kernel_pool_acquire_buffer(pool, b.c_bytes);
    b.sink    = xdna_kernel_pool_acquire_buffer(pool, SINK_BYTES);
    return b.w && b.a && b.c && b.sink && b.w->data && b.a->data && b.c->data;
}

void release(xdna_kernel_pool * pool, buffers & b) {
    for (xdna_buffer * x : { b.w, b.a, b.c, b.sink }) {
        if (x) {
            xdna_kernel_pool_release_buffer(pool, x);
        }
    }
}

bool pack_inputs(const xdna_pfx_attn_job & job, const std::vector<pass> & ps, const buffers & b) {
    xdna_prof::section_timer st("prefill: pfx attn pack");
    pack_w(job, ps, (uint8_t *) b.w->data);
    pack_a(job, ps, (uint16_t *) b.a->data);
    return xdna_buffer_sync_to_device_range(b.w, b.w_bytes, 0) && xdna_buffer_sync_to_device_range(b.a, b.a_bytes, 0);
}

// C comes from the pool, and a host line its last holder left dirty would be written back over what the array writes
// by the readback's flush: C is flushed before the dispatch.
bool run_stream(xdna_kernel * kern, const buffers & b) {
    xdna_prof::section_timer st("prefill: pfx attn run");
    xdna_buffer *            args[N_ARGS] = { b.w, b.a, b.c, b.sink };
    xdna_test_dirty_if_armed(XDNA_OUT_PFX_ATTN, b.c, b.c_bytes, 0);
    if (!xdna_buffer_sync_to_device_range(b.c, b.c_bytes, 0)) {
        return false;
    }
    xrt::run run = xdna_kernel_run_start(kern, args, N_ARGS);
    return xdna_run_wait(run) && xdna_buffer_sync_from_device_range(b.c, b.c_bytes, 0);
}

}  // namespace

bool xdna_pfx_attn_active(void) {
    return pfx_loaded();
}

bool xdna_pfx_attn_run(xdna_kernel_pool * pool, const xdna_pfx_attn_job & job) {
    std::lock_guard<std::mutex> lock(g_pfx.mtx);
    if (job.hg < 1 || R % job.hg != 0 || (R / job.hg) % 8 != 0) {
        return false;
    }
    xdna_kernel * kern = kernel_of(pool->device);
    if (!kern) {
        return false;
    }
    const std::vector<pass>     ps    = plan_passes(job);
    const std::vector<uint32_t> insts = build_seq(tiles_of(job), ps);
    if (!xdna_kernel_bind_insts(pool->device, kern, insts.data(), insts.size())) {
        return false;  // xdna-runtime names the stream and the reason
    }
    buffers b;
    bool    ok = acquire(pool, job, ps, b) && pack_inputs(job, ps, b) && run_stream(kern, b);
    if (ok) {
        xdna_prof::section_timer st("prefill: pfx attn unpack");
        unpack_o(job, ps, (const float *) b.c->data);
    }
    release(pool, b);
    return ok;
}
