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
constexpr int          N_ACOLS     = xdna_pfx_layout::ATTN_COLS;
constexpr int          PAIRS       = xdna_pfx_layout::PAIRS;
constexpr int          KVH         = xdna_pfx_layout::KVH;
constexpr int          DH          = 128;           // a core's half of D
constexpr int          R           = xdna_pfx_layout::R;
constexpr int          NK          = xdna_pfx_layout::NK;
constexpr int          A_ELEMS     = 64 * 64;       // bf16 an A object: one K or V^T tile half
constexpr int          TRAILER_POS = R * DH;        // bf16 offset of a Q object's trailer
constexpr int          TRAILER_I32 = 2 + 2 * R;     // first key, tiles, each row's first and last visible key
constexpr uint32_t     TB_BYTES    = (uint32_t) xdna_pfx_layout::TB_BYTES;
constexpr int          C_F32       = 64 * 32;       // f32 a C object
constexpr int          O_OBJS      = 2;             // C objects a core's O half
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
using xdna_pfx_layout::a_stream_objs;
using xdna_pfx_layout::kv_obj;
using xdna_pfx_layout::w_obj_off;

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

// Both A streams' header objects
void a_headers_host(const std::vector<pass> & ps, uint16_t * a, size_t stream_elems) {
    for (int h = 0; h < 2; h++) {
        uint16_t * s = a + (size_t) h * stream_elems;
        std::memset(s, 0, xdna_pfx_layout::HDR_OBJS * A_ELEMS * sizeof(uint16_t));
        pack_headers(ps, s);
    }
}

int64_t row_pos(const xdna_pfx_attn_job & job, const pass & p, int pidx, int row) {
    const int pp = R / job.hg;
    return p.t0 + (int64_t) pidx * pp + row % pp;
}

int head_of(const xdna_pfx_attn_job & job, const pass & p, int row) {
    return job.hg * p.g + row / (R / job.hg);
}

// The trailer after a pair's Q rows: the first key the pass streams, its tiles, then each row's first and last
// visible key; past the ubatch the rows see the first key.
void write_trailer(const xdna_pfx_attn_job & job, const pass & p, int pidx, uint16_t * o) {
    int32_t       tr[TRAILER_I32];
    const int32_t first_key = (int32_t) (p.tile_lo * NK);
    tr[0]                   = first_key;
    tr[1]                   = (int32_t) (p.tile_hi - p.tile_lo);
    for (int row = 0; row < R; row++) {
        const int64_t t = row_pos(job, p, pidx, row);
        const bool    real = t < job.n_tokens;
        tr[2 + row]     = real ? job.lo[t] : first_key;
        tr[2 + R + row] = real ? job.hi[t] : first_key;
    }
    std::memcpy(o + TRAILER_POS, tr, sizeof(tr));
}

// Core i's Q half^T, scaled for exp2; past the ubatch the rows are zero.
void pack_q_rows(const xdna_pfx_attn_job & job, const pass & p, int pidx, int h, uint16_t * o) {
    const float          qs = (float) (M_LOG2E * job.q_scale);
    alignas(64) uint16_t qb[R][DH];
    for (int row = 0; row < R; row++) {
        const int64_t t = row_pos(job, p, pidx, row);
        if (t >= job.n_tokens) {
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

uint16_t * w_obj(uint8_t * w, size_t npass, int ci, size_t p, int i) {
    return (uint16_t *) (w + w_obj_off(npass, ci, p, i));
}

void pack_w_rows(const xdna_pfx_attn_job & job, const std::vector<pass> & ps, uint8_t * w) {
#pragma omp parallel for collapse(2) num_threads(xdna_chunk_threads())
    for (int ci = 0; ci < N_ACOLS; ci++) {
        for (size_t p = 0; p < ps.size(); p++) {
            for (int i = 0; i < 4; i++) {
                pack_q_rows(job, ps[p], 2 * ci + i / 2, i % 2, w_obj(w, ps.size(), ci, p, i));
            }
        }
    }
}

// Pass p's four W objects on attention column ci: trailers, and zero Q where the pair starts at or past `rows`
void w_host_objs(const xdna_pfx_attn_job & job, const std::vector<pass> & ps, int ci, size_t p, int64_t rows,
                 uint8_t * w) {
    const int pidx = 2 * ci;
    for (int i = 0; i < 4; i++) {
        uint16_t * o = w_obj(w, ps.size(), ci, p, i);
        write_trailer(job, ps[p], pidx + i / 2, o);
        if (row_pos(job, ps[p], pidx + i / 2, 0) >= rows) {
            std::memset(o, 0, TRAILER_POS * sizeof(uint16_t));
        }
    }
}

void w_host(const xdna_pfx_attn_job & job, const std::vector<pass> & ps, int64_t rows, uint8_t * w) {
#pragma omp parallel for collapse(2) num_threads(xdna_chunk_threads())
    for (int ci = 0; ci < N_ACOLS; ci++) {
        for (size_t p = 0; p < ps.size(); p++) {
            w_host_objs(job, ps, ci, p, rows, w);
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

bool hg_ok(int hg) {
    return hg >= 1 && R % hg == 0 && (R / hg) % 8 == 0;
}

void host_objects(const xdna_pfx_attn_job & job, const std::vector<pass> & ps, const xdna_pfx_attn_dst & d,
                  uint8_t * parent) {
    w_host(job, ps, d.n_tiles * NK, parent);
    a_headers_host(ps, (uint16_t *) (parent + d.a_off), d.stream_bytes / sizeof(uint16_t));
}

}  // namespace

// The W objects and A streams in one parent buffer (dst.bo), argument 0 and 1 as views of it
struct xdna_pfx_attn_io {
    xdna_pfx_attn_job job;
    std::vector<pass> ps;
    xdna_pfx_attn_dst dst;
    xdna_buffer *     w = nullptr, *a = nullptr, *c = nullptr, *sink = nullptr;
    size_t            c_bytes = 0;
};

namespace {

bool acquire(xdna_kernel_pool * pool, xdna_pfx_attn_io & io) {
    const xdna_pfx_attn_dst & d = io.dst;
    io.c_bytes                  = N_ACOLS * c_col_f32(io.ps.size()) * 4;
    io.dst.bo                   = xdna_kernel_pool_acquire_buffer(pool, d.a_off + d.a_bytes);
    io.c                        = xdna_kernel_pool_acquire_buffer(pool, io.c_bytes);
    io.sink                     = xdna_kernel_pool_acquire_buffer(pool, SINK_BYTES);
    if (!io.dst.bo || !io.dst.bo->data || !io.c || !io.c->data || !io.sink) {
        return false;
    }
    io.w = xdna_buffer_sub(io.dst.bo, 0, d.w_bytes);
    io.a = xdna_buffer_sub(io.dst.bo, d.a_off, d.a_bytes);
    return io.w && io.a;
}

void release_buffers(xdna_kernel_pool * pool, xdna_pfx_attn_io & io) {
    xdna_buffer_free(io.w);
    xdna_buffer_free(io.a);
    for (xdna_buffer * x : { io.dst.bo, io.c, io.sink }) {
        if (x) {
            xdna_kernel_pool_release_buffer(pool, x);
        }
    }
}

bool pack_inputs(const xdna_pfx_attn_io & io) {
    xdna_prof::section_timer st("prefill: pfx attn pack");
    pack_w_rows(io.job, io.ps, (uint8_t *) io.w->data);
    pack_kv(io.job, (uint16_t *) io.a->data, io.dst.stream_bytes / sizeof(uint16_t));
    return xdna_buffer_sync_to_device_range(io.w, io.dst.w_bytes, 0) &&
           xdna_buffer_sync_to_device_range(io.a, io.dst.a_bytes, 0);
}

// C comes from the pool, and a host line its last holder left dirty would be written back over what the array writes
// by the readback's flush: C is flushed before the dispatch.
bool run_stream(xdna_kernel * kern, const xdna_pfx_attn_io & io) {
    xdna_prof::section_timer st("prefill: pfx attn run");
    xdna_buffer *            args[N_ARGS] = { io.w, io.a, io.c, io.sink };
    xdna_test_dirty_if_armed(XDNA_OUT_PFX_ATTN, io.c, io.c_bytes, 0);
    if (!xdna_buffer_sync_to_device_range(io.c, io.c_bytes, 0)) {
        return false;
    }
    xrt::run run = xdna_kernel_run_start(kern, args, N_ARGS);
    return xdna_run_wait(run) && xdna_buffer_sync_from_device_range(io.c, io.c_bytes, 0);
}

}  // namespace

bool xdna_pfx_attn_active(void) {
    return pfx_loaded();
}

void xdna_pfx_attn_host_objects(const xdna_pfx_attn_job & job, uint8_t * parent) {
    host_objects(job, plan_passes(job), xdna_pfx_attn_geometry(job.n_tokens, job.n_kv, job.hg), parent);
}

xdna_pfx_attn_io * xdna_pfx_attn_prepare(xdna_kernel_pool * pool, const xdna_pfx_attn_job & job) {
    if (!hg_ok(job.hg) || job.n_tokens <= 0 || job.n_kv <= 0) {
        return nullptr;
    }
    xdna_pfx_attn_io * io = new xdna_pfx_attn_io;
    io->job               = job;
    io->ps                = plan_passes(job);
    io->dst               = xdna_pfx_attn_geometry(job.n_tokens, job.n_kv, job.hg);
    if (io->ps.size() != (size_t) (KVH * io->dst.npg) || !acquire(pool, *io)) {
        xdna_pfx_attn_release(pool, io);
        return nullptr;
    }
    xdna_prof::section_timer st("prefill: pfx attn prepare");
    host_objects(io->job, io->ps, io->dst, (uint8_t *) io->dst.bo->data);
    if (!xdna_buffer_sync_to_device_range(io->dst.bo, io->dst.a_off + io->dst.a_bytes, 0)) {
        xdna_pfx_attn_release(pool, io);
        return nullptr;
    }
    return io;
}

const xdna_pfx_attn_dst & xdna_pfx_attn_dst_of(const xdna_pfx_attn_io * io) {
    return io->dst;
}

bool xdna_pfx_attn_pack(xdna_pfx_attn_io * io) {
    return io && io->job.q && io->job.k && io->job.v && pack_inputs(*io);
}

bool xdna_pfx_attn_run_io(xdna_kernel_pool * pool, xdna_pfx_attn_io * io) {
    std::lock_guard<std::mutex> lock(g_pfx.mtx);
    xdna_kernel *               kern = io && io->job.out ? kernel_of(pool->device) : nullptr;
    if (!kern) {
        return false;
    }
    const std::vector<uint32_t> insts = build_seq(io->dst.n_tiles, io->ps);
    if (!xdna_kernel_bind_insts(pool->device, kern, insts.data(), insts.size()) || !run_stream(kern, *io)) {
        return false;  // xdna-runtime names the stream and the reason
    }
    xdna_prof::section_timer st("prefill: pfx attn unpack");
    unpack_o(io->job, io->ps, (const float *) io->c->data);
    return true;
}

void xdna_pfx_attn_release(xdna_kernel_pool * pool, xdna_pfx_attn_io * io) {
    if (io) {
        release_buffers(pool, *io);
        delete io;
    }
}

bool xdna_pfx_attn_run(xdna_kernel_pool * pool, const xdna_pfx_attn_job & job) {
    xdna_pfx_attn_io * io = xdna_pfx_attn_prepare(pool, job);
    const bool         ok = io && xdna_pfx_attn_pack(io) && xdna_pfx_attn_run_io(pool, io);
    xdna_pfx_attn_release(pool, io);
    return ok;
}
