#include "xdna-pfx-row.h"

#include "ggml.h"
#include "xdna-pgemm-path.h"
#include "xdna-prof.h"
#include "xdna-runtime.h"
#include "xdna-seq.h"
#include "xdna-util.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <vector>

// Host side of kernels/pfx.py's row command. Four arguments:
//   0  W  the projections' BO, read in place: per block, column 3 takes the four Q heads' 8 rows (one shim
//         descriptor iterated over the blocks), column 7 the two K heads' and then the two V heads' (a chain of two)
//   1  A  pgemm's header object (word 9 a row-op core's C objects, 11 the A objects an attention core lets pass, 12
//         a row-op core's setup objects), then a block's cos and sin a block; both A streams read it
//   2  C  per row-op column, its C join's objects; with dst, dst's buffer: a drain task a key tile and column scatters
//         its two C objects into the attention's W, K and V^T objects
//   3  per row-op column, a setup object a core: kernels/rowop.cc's parameters and the norm weight's bf16 split
// The W path is the attention command's: a block's four x objects into a MemTile slot, then each to its core.

namespace {

using namespace xdna_pgemm_path;

constexpr uint32_t ROW_COLS[]  = { 3, 7 };  // kernels/pfx.py ROW_COLS
constexpr int      N_RCOLS     = 2;
constexpr int      CORES       = 4;         // a column's cores, rows 2-5
constexpr int      D           = 256;       // dims a head
constexpr int      Q_HEADS     = 4;
constexpr int      KV_HEADS    = 2;
constexpr int      BLOCK_ROWS  = 8;         // positions a block
constexpr int      C_BLOCKS    = 2;         // blocks a core's C object (kernels/pfx.py ROW_BLOCKS)
constexpr int      C_F32       = 64 * 32;   // f32 a core's C object
constexpr int      A_STREAMS   = 2;
constexpr size_t   A_OBJ_BYTES = A_OBJ_ELEMS * sizeof(uint16_t);
constexpr uint32_t TB_BYTES    = 10752;     // a core's W buffer, the length of every W send
constexpr uint32_t X_BYTES     = BLOCK_ROWS * D * sizeof(float);
// A block's four x objects; the last core's send reads a whole W buffer
constexpr uint32_t SLOT_BYTES  = CORES * X_BYTES + TB_BYTES - X_BYTES;
constexpr int      W_SLOTS     = 2;
// The setup objects' MemTile region, taken under slot 0's locks
constexpr uint32_t SETUP_ADDR  = WS_ADDR + W_SLOTS * SLOT_BYTES;
constexpr uint32_t W_PKT_CORE0 = 2;         // kernels/pgemm.py W_PKT_CORE: core i's id is 2 + i
constexpr int      W_RING      = 2;         // shim W tasks in flight
constexpr uint32_t SH_BD_A_HDR = 0;
constexpr uint32_t SH_BD_A_TAB = 1;
constexpr uint32_t SH_BD_X     = 6;         // the x tasks' descriptors, X_PARTS_MAX a ring entry
constexpr int      X_PARTS_MAX = 2;
constexpr int64_t  ITER_MAX    = 64;        // blocks a shim descriptor iterates over (its wrap)
constexpr uint32_t AX_CACHE    = 2;
constexpr int      ARG_W = 0, ARG_A = 1, ARG_C = 2, ARG_SETUP = 3, N_ARGS = 4;
constexpr uint64_t DDR_PATCH_MAX = 1ull << 32;
// kernels/rowop.cc: the mode, the setup object's parameter words, and the rotated dims its loops step over
constexpr int32_t  RO_R1_BLOCK = 5;
constexpr int      P_MODE = 0, P_D = 1, P_N_ROT = 2, P_SCALE = 3, P_EPS = 4, P_LAYOUT = 5, PARAM_WORDS = 16;
constexpr size_t   GS_BYTE     = PARAM_WORDS * sizeof(int32_t);
constexpr int      ROT_STEP    = 128;
// The attention's objects (xdna-pfx-layout.h): a drain task takes a 32-key tile's two C objects on a column, each core
// part (C_F32 words) through one descriptor of a 4-chain, from a ring of two tasks
constexpr int      DST_PP       = C_BLOCKS * BLOCK_ROWS;  // a C object's positions: one W object's pair
constexpr int      HG           = Q_HEADS / KV_HEADS;
constexpr int      TILE_BLOCKS  = xdna_pfx_layout::NK / BLOCK_ROWS;
constexpr int      PAIR_OBJS    = TILE_BLOCKS / C_BLOCKS;
constexpr int      DRAIN_RING   = 2;
constexpr uint32_t DRAIN_BDS[DRAIN_RING][CORES] = { { 4, 5, 10, 11 }, { 12, 13, 14, 15 } };
constexpr int      HALF_D       = D / 2;                  // an attention core's D half
constexpr int      HALF_WORDS   = C_F32 / 2;              // a core part's words of one D half
// The Q, K and V^T layouts' word positions in a core part and the shim's runs, in words
constexpr int      Q_CHUNK      = 64;  // Q: a head's 16 rows of one 8-dim chunk (two row tiles)
constexpr int      Q_CHUNKS     = HALF_D / 8;
constexpr int      Q_CHUNK_STEP = 2 * Q_CHUNK;  // the two heads' chunks of a W object alternate
constexpr int      Q_HEAD_BYTES = Q_CHUNK * 4;
constexpr int      Q_TILE       = 32;           // a block's 8 rows of an 8-dim chunk
constexpr int      Q_DIM        = 4;            // a dim's 8 rows
constexpr int      K_BLOCK      = 512;          // K: a block's 8 keys x 128 dims
constexpr int      K_DHALF      = 256;          // K: dims 64-127 of the 8 keys
constexpr int      K_ROW        = 32;
constexpr int      K_OBJ_HALF   = 2 * K_BLOCK;  // keys 16-31 of a K object
constexpr int      VT_ROW       = 8;            // V^T: a dim's 16 keys of a C object
constexpr int      VT_ROW_STEP  = 16;           // a dim's 32 keys of the tile
constexpr int      VT_BLOCK     = 4;

static_assert(W_SLOTS * SLOT_BYTES + CORES * TB_BYTES <= WS_BYTES, "the slots and the setup region fit the buffer");
static_assert(GS_BYTE + 2 * D * sizeof(uint16_t) <= TB_BYTES, "the setup object fits a W buffer");

// `heads` heads of D at byte `off` of argument 0, rows `ldc` floats apart
struct x_part {
    uint32_t off   = 0;
    uint32_t ldc   = 0;
    int      heads = 0;
};

struct x_parts {
    x_part p[X_PARTS_MAX];
    int    n = 0;
};

// Column ci's x: Q's heads, or K's and then V's
x_parts parts_of(const xdna_pfx_row_job & job, int ci) {
    x_parts xp;
    if (ci == 0) {
        xp.p[xp.n++] = { (uint32_t) job.q_off, Q_HEADS * D, Q_HEADS };
        return xp;
    }
    xp.p[xp.n++] = { (uint32_t) job.k_off, KV_HEADS * D, KV_HEADS };
    xp.p[xp.n++] = { (uint32_t) job.v_off, KV_HEADS * D, KV_HEADS };
    return xp;
}

int32_t bits_of(float f) {
    int32_t w;
    std::memcpy(&w, &f, sizeof(w));
    return w;
}

float f32_of_bf16(uint16_t h) {
    const uint32_t b = (uint32_t) h << 16;
    float          f;
    std::memcpy(&f, &b, sizeof(f));
    return f;
}

// --- the A objects ---

void pack_header(int64_t n_blocks, uint16_t * obj) {
    int32_t w[HDR_WORDS]  = {};
    w[HW_ROW_ON]          = (int32_t) (n_blocks / C_BLOCKS);
    w[HW_ATTN_DRAIN]      = (int32_t) n_blocks;
    w[HW_ROW_SETUP]       = 1;
    write_tiled_words(w, HDR_WORDS, obj);
}

// ggml-cpu's rope cache (rope_yarn_ramp, rope_yarn, ggml_rope_cache_init) in its own float arithmetic, so the tables
// hold the CPU ROPE's own cos and sin
struct rope_consts {
    xdna_pfx_row_rope rp;
    float             theta_scale = 1.0f;
    float             corr[2]     = { 0.0f, 0.0f };
};

rope_consts rope_consts_of(const xdna_pfx_row_rope & rp) {
    rope_consts rc;
    rc.rp          = rp;
    rc.theta_scale = powf(rp.freq_base, -2.0f / (float) rp.n_dims);
    ggml_rope_yarn_corr_dims(rp.n_dims, rp.n_ctx_orig, rp.freq_base, rp.beta_fast, rp.beta_slow, rc.corr);
    return rc;
}

float yarn_ramp(float low, float high, int i0) {
    const float y = (i0 / 2 - low) / std::max(0.001f, high - low);
    return 1 - std::min(1.0f, std::max(0.0f, y));
}

void rope_yarn(const rope_consts & rc, float theta_extrap, int i0, float * cs, float * sn) {
    const float theta_interp = rc.rp.freq_scale * theta_extrap;
    float       theta        = theta_interp;
    float       mscale       = rc.rp.attn_factor;
    if (rc.rp.ext_factor != 0.0f) {
        const float ramp_mix = yarn_ramp(rc.corr[0], rc.corr[1], i0) * rc.rp.ext_factor;
        theta                = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / rc.rp.freq_scale);
    }
    *cs = cosf(theta) * mscale;
    *sn = sinf(theta) * mscale;
}

// Position p's cos and sin at the n_dims / 2 frequencies
void rope_row(const rope_consts & rc, float p, float * cs, float * sn) {
    float theta = p;
    for (int i = 0; i < rc.rp.n_dims / 2; i++) {
        rope_yarn(rc, theta, 2 * i, &cs[i], &sn[i]);
        theta *= rc.theta_scale;
    }
}

// Block b's table as the cores read it: cos [8][n_dims / 2], then sin; zero past n_tokens
void table_rows(const xdna_pfx_row_job & job, const rope_consts & rc, int64_t b, float * tab) {
    const int half = rc.rp.n_dims / 2;
    for (int j = 0; j < BLOCK_ROWS; j++) {
        const int64_t t = b * BLOCK_ROWS + j;
        if (t < job.n_tokens) {
            rope_row(rc, (float) job.pos[t], tab + j * half, tab + (BLOCK_ROWS + j) * half);
        }
    }
}

void pack_table(const xdna_pfx_row_job & job, const rope_consts & rc, int64_t b, uint16_t * obj) {
    float    tab[A_OBJ_ELEMS / 2] = {};
    uint16_t core[A_OBJ_ELEMS];
    table_rows(job, rc, b, tab);
    std::memcpy(core, tab, sizeof(tab));
    scatter_a_object(core, obj);
}

void pack_tables(const xdna_pfx_row_job & job, uint16_t * objs) {
    if (job.rope.n_dims == 0) {
        return;
    }
    const rope_consts rc       = rope_consts_of(job.rope);
    const int64_t     n_blocks = xdna_pfx_row_job_blocks(job);
#pragma omp parallel for num_threads(xdna_chunk_threads())
    for (int64_t b = 0; b < n_blocks; b++) {
        pack_table(job, rc, b, objs + (size_t) b * A_OBJ_ELEMS);
    }
}

// --- the setup objects ---

// Core i of row-op column ci: its norm weight (null: none, V's), rotated dims, scale and output layout
struct core_role {
    const float * gamma  = nullptr;
    int           n_rot  = 0;
    float         scale  = 1.0f;
    int32_t       layout = XDNA_PFX_ROW_LAYOUT_C;
};

core_role role_of(const xdna_pfx_row_job & job, int ci, int i) {
    if (ci == 0) {
        return { job.q_norm, job.rope.n_dims, job.q_scale, XDNA_PFX_ROW_LAYOUT_Q };
    }
    if (i < KV_HEADS) {
        return { job.k_norm, job.rope.n_dims, 1.0f, XDNA_PFX_ROW_LAYOUT_K };
    }
    return { nullptr, 0, 1.0f, XDNA_PFX_ROW_LAYOUT_VT };
}

// gamma as kernels/rowop.cc's split<3> makes it: hi the nearest bf16, lo the nearest bf16 of the rest
void split_weight(const float * gamma, uint16_t * hi, uint16_t * lo) {
    for (int d = 0; d < D; d++) {
        const float g = gamma ? gamma[d] : 1.0f;
        hi[d]         = xdna_bf16(g);
        lo[d]         = xdna_bf16(g - f32_of_bf16(hi[d]));
    }
}

void pack_setup_obj(const xdna_pfx_row_job & job, const core_role & role, uint8_t * obj) {
    int32_t prm[PARAM_WORDS] = {};
    prm[P_MODE]              = RO_R1_BLOCK;
    prm[P_D]                 = D;
    prm[P_N_ROT]             = role.n_rot;
    prm[P_SCALE]             = bits_of(role.scale);
    prm[P_EPS]               = bits_of(job.eps);
    prm[P_LAYOUT]            = role.layout;
    uint16_t gs[2 * D];
    split_weight(role.gamma, gs, gs + D);
    std::memcpy(obj, prm, sizeof(prm));
    std::memcpy(obj + GS_BYTE, gs, sizeof(gs));
}

void pack_setup_col(const xdna_pfx_row_job & job, int ci, uint8_t * objs) {
    for (int i = 0; i < CORES; i++) {
        pack_setup_obj(job, role_of(job, ci, i), objs + (size_t) i * TB_BYTES);
    }
}

// --- the instruction stream ---

struct col_state {
    int shim_pushed = 0;  // W tasks on the shim: the setup and the x tasks
    int x_tasks     = 0;
    int drains      = 0;  // C tasks with dst
    int mt_s2mm     = 0;
    int mt_mm2s[2]  = { 0, 0 };
};

void init_col_locks(xdna_seq * seq, uint32_t c) {
    for (uint32_t k = 0; k < W_SLOTS; k++) {
        mt_set_lock(seq, c, W_LOCK0 + 2 * k, CORES);
        mt_set_lock(seq, c, W_LOCK0 + 2 * k + 1, 0);
    }
}

void init_slot_locks(xdna_seq * seq) {
    for (uint32_t c : ROW_COLS) {
        init_col_locks(seq, c);
    }
}

// Both A streams: the header, then every block's table
void a_stream(xdna_seq * seq, int64_t n_blocks) {
    for (uint32_t h = 0; h < A_STREAMS; h++) {
        const uint32_t c  = A_COL0 + h;
        xdna_bd        hb = linear_bd((uint32_t) (A_OBJ_BYTES / 4));
        xdna_seq_blockwrite(seq, c, 0, SH_BD_A_HDR, &hb);
        xdna_seq_ddr_patch(seq, c, 0, SH_BD_A_HDR, ARG_A, 0);
        xdna_seq_push_queue(seq, c, 0, SH_BD_A_HDR, xdna_dma_dir::MM2S, 0, false, 0);
        xdna_bd tb = linear_bd((uint32_t) (n_blocks * A_OBJ_BYTES / 4));
        xdna_seq_blockwrite(seq, c, 0, SH_BD_A_TAB, &tb);
        xdna_seq_ddr_patch(seq, c, 0, SH_BD_A_TAB, ARG_A, (uint32_t) A_OBJ_BYTES);
        xdna_seq_push_queue(seq, c, 0, SH_BD_A_TAB, xdna_dma_dir::MM2S, 0, false, 0);
    }
}

// Core i's part of row-op column ci's C objects 2 m and 2 m + 1 (key tile m) as dst's layout takes it: from byte
// off, d0 x d1 runs, the step to its other D half, and the iteration's step to the second object, in words.
struct drain_part {
    size_t   off = 0;
    uint32_t d0 = 0, d1 = 0, d1_step = 0, d2_step = 0, iter_step = 0;
};

uint32_t words_of(size_t bytes) {
    return (uint32_t) (bytes / sizeof(uint32_t));
}

// Q head i: its 16 rows of the W objects of KV head i / HG at positions pos and pos + 16, each D half a W object,
// chunks of 8 dims beside the other head's
drain_part q_part(const xdna_pfx_attn_dst & d, int64_t pos, int i) {
    const int    g = i / HG, hh = i % HG;
    const size_t o0 = xdna_pfx_attn_q_obj(d, g, pos, 0);
    return { o0 + (size_t) hh * Q_HEAD_BYTES,
             Q_CHUNK,
             Q_CHUNKS,
             Q_CHUNK_STEP,
             words_of(xdna_pfx_attn_q_obj(d, g, pos, 1) - o0),
             words_of(xdna_pfx_attn_q_obj(d, g, pos + DST_PP, 0) - o0) };
}

// K head g: keys 0-15, then 16-31 of its K objects of tile m; V head g: the same keys of each dim's row of V^T
drain_part kv_part(const xdna_pfx_attn_dst & d, int64_t m, int i) {
    const bool     vt = i >= KV_HEADS;
    const size_t   o0 = xdna_pfx_attn_kv_obj(d, i % KV_HEADS, m, 0, vt);
    const uint32_t d2 = words_of(xdna_pfx_attn_kv_obj(d, i % KV_HEADS, m, 1, vt) - o0);
    if (!vt) {
        return { o0, K_BLOCK, C_BLOCKS, K_BLOCK, d2, K_OBJ_HALF };
    }
    return { o0, VT_ROW, HALF_D, VT_ROW_STEP, d2, VT_ROW };
}

drain_part drain_part_of(const xdna_pfx_attn_dst & d, int ci, int64_t m, int i) {
    return ci == 0 ? q_part(d, m * xdna_pfx_layout::NK, i) : kv_part(d, m, i);
}

size_t plan_at(int64_t n_tiles, int ci, int64_t m) {
    return ((size_t) ci * (size_t) n_tiles + (size_t) m) * CORES;
}

// Every core part of every tile on both row-op columns
std::vector<drain_part> drain_plan(const xdna_pfx_attn_dst & d) {
    const size_t            per_col = (size_t) d.n_tiles * CORES;
    std::vector<drain_part> plan(N_RCOLS * per_col);
    for (size_t k = 0; k < plan.size(); k++) {
        plan[k] = drain_part_of(d, (int) (k / per_col), (int64_t) ((k % per_col) / CORES), (int) (k % CORES));
    }
    return plan;
}

void drain_bd(xdna_seq * seq, uint32_t c, uint32_t id, const drain_part & p, uint32_t next, bool chained) {
    xdna_bd bd;
    bd.buf_len     = C_F32;
    bd.d0_size     = p.d0;
    bd.d0_stride   = 1;
    bd.d1_size     = p.d1;
    bd.d1_stride   = p.d1_step;
    bd.d2_stride   = p.d2_step;
    bd.iter_size   = PAIR_OBJS;
    bd.iter_stride = p.iter_step;
    bd.ax_cache    = AX_CACHE;
    bd.next_bd     = next;
    bd.use_next    = chained;
    xdna_seq_blockwrite(seq, c, 0, id, &bd);
    xdna_seq_ddr_patch(seq, c, 0, id, ARG_C, (uint32_t) p.off);
}

void drain_bds(xdna_seq * seq, uint32_t c, const uint32_t * ids, const drain_part * parts) {
    for (int i = 0; i < CORES; i++) {
        drain_bd(seq, c, ids[i], parts[i], ids[(i + 1) % CORES], i + 1 < CORES);
    }
}

// A tile's drain on row-op column ci: the four core parts chained, iterated over its two C objects
void drain_tile(xdna_seq * seq, int ci, const drain_part * parts, col_state & cs) {
    const uint32_t c = ROW_COLS[ci];
    if (cs.drains >= DRAIN_RING) {
        xdna_seq_wait_token(seq, c, 0, xdna_dma_dir::S2MM, 0);
    }
    const uint32_t * ids = DRAIN_BDS[cs.drains % DRAIN_RING];
    drain_bds(seq, c, ids, parts);
    xdna_seq_issue_token(seq, c, 0, xdna_dma_dir::S2MM, 0, 0xF);
    xdna_seq_push_queue(seq, c, 0, ids[0], xdna_dma_dir::S2MM, 0, true, PAIR_OBJS - 1);
    cs.drains++;
}

// Room for one more W task on the shim: the ring's oldest done
void shim_ring_wait(xdna_seq * seq, uint32_t c, const col_state & cs) {
    if (cs.shim_pushed >= W_RING) {
        xdna_seq_wait_token(seq, c, 0, xdna_dma_dir::MM2S, W_SHIM_CH);
    }
}

void shim_push(xdna_seq * seq, uint32_t c, uint32_t id, uint32_t repeat, col_state & cs) {
    xdna_seq_issue_token(seq, c, 0, xdna_dma_dir::MM2S, W_SHIM_CH, 0xF);
    xdna_seq_push_queue(seq, c, 0, id, xdna_dma_dir::MM2S, W_SHIM_CH, true, repeat);
    cs.shim_pushed++;
}

// `bytes` from the shim into the MemTile at `at`, once the slot of locks (e, f) is empty
void mt_fill(xdna_seq * seq, uint32_t c, uint32_t at, uint32_t bytes, uint32_t e, uint32_t f, col_state & cs) {
    wait_mt_queue(seq, c, xdna_dma_dir::S2MM, W_MT_S2MM, cs.mt_s2mm);
    const uint32_t id = MT_BD_S2MM + (uint32_t) (cs.mt_s2mm % MT_Q);
    mt_bd(seq, c, id, bytes / 4, at, e, CORES, f, CORES);
    mt_push(seq, c, id, xdna_dma_dir::S2MM, W_MT_S2MM, 0);
    cs.mt_s2mm++;
}

// A W buffer to each core from `at` + i `stride`, under its own packet id, pair r's on MM2S 4 + r
void mt_sends(xdna_seq * seq, uint32_t c, uint32_t at, uint32_t stride, uint32_t e, uint32_t f, col_state & cs) {
    for (uint32_t i = 0; i < CORES; i++) {
        const uint32_t r = i / 2;
        wait_mt_queue(seq, c, xdna_dma_dir::MM2S, W_MT_MM2S + r, cs.mt_mm2s[r]);
        const uint32_t id  = (r ? MT_BD_MM2S1 : MT_BD_MM2S0) + (uint32_t) (cs.mt_mm2s[r] % MT_Q);
        const uint32_t pkt = MT_BD_PKT | (W_PKT_CORE0 + i) << MT_BD_PKT_ID_SHIFT;
        mt_bd(seq, c, id, TB_BYTES / 4, at + i * stride, f, 1, e, 1, pkt);
        mt_push(seq, c, id, xdna_dma_dir::MM2S, W_MT_MM2S + r, 0);
        cs.mt_mm2s[r]++;
    }
}

void setup_sends(xdna_seq * seq, int ci, col_state & cs) {
    const uint32_t c = ROW_COLS[ci];
    shim_ring_wait(seq, c, cs);
    xdna_bd bd = linear_bd(CORES * TB_BYTES / 4);
    xdna_seq_blockwrite(seq, c, 0, SH_BD_W, &bd);
    xdna_seq_ddr_patch(seq, c, 0, SH_BD_W, ARG_SETUP, (uint32_t) ci * CORES * TB_BYTES);
    shim_push(seq, c, SH_BD_W, 0, cs);
    mt_fill(seq, c, SETUP_ADDR, CORES * TB_BYTES, W_LOCK0, W_LOCK0 + 1, cs);
    mt_sends(seq, c, SETUP_ADDR, TB_BYTES, W_LOCK0, W_LOCK0 + 1, cs);
}

void setup_cols(xdna_seq * seq, col_state * cs) {
    for (int ci = 0; ci < N_RCOLS; ci++) {
        setup_sends(seq, ci, cs[ci]);
    }
}

// Part p's rows of `nb` blocks from block b0 on: a head's row of D, its 8 rows, its heads, the blocks iterated
void x_bd(xdna_seq * seq, uint32_t c, uint32_t id, const x_part & p, int64_t b0, int64_t nb, bool chained) {
    xdna_bd bd;
    bd.buf_len     = (uint32_t) (p.heads * BLOCK_ROWS * D);
    bd.d0_size     = D;
    bd.d0_stride   = 1;
    bd.d1_size     = BLOCK_ROWS;
    bd.d1_stride   = p.ldc;
    bd.d2_stride   = D;
    bd.iter_size   = (uint32_t) nb;
    bd.iter_stride = BLOCK_ROWS * p.ldc;
    bd.ax_cache    = AX_CACHE;
    bd.next_bd     = id + 1;
    bd.use_next    = chained;
    xdna_seq_blockwrite(seq, c, 0, id, &bd);
    xdna_seq_ddr_patch(seq, c, 0, id, ARG_W, p.off + (uint32_t) (b0 * BLOCK_ROWS * p.ldc * sizeof(float)));
}

void x_bds(xdna_seq * seq, uint32_t c, uint32_t id0, const x_parts & xp, int64_t b0, int64_t nb) {
    for (int k = 0; k < xp.n; k++) {
        x_bd(seq, c, id0 + (uint32_t) k, xp.p[k], b0, nb, k + 1 < xp.n);
    }
}

// One shim task for blocks [b0, b0 + nb): its descriptors walk them once a repeat
void x_task(xdna_seq * seq, const xdna_pfx_row_job & job, int ci, int64_t b0, int64_t nb, col_state & cs) {
    const uint32_t c   = ROW_COLS[ci];
    const uint32_t id0 = SH_BD_X + (uint32_t) (X_PARTS_MAX * (cs.x_tasks % W_RING));
    shim_ring_wait(seq, c, cs);
    x_bds(seq, c, id0, parts_of(job, ci), b0, nb);
    shim_push(seq, c, id0, (uint32_t) (nb - 1), cs);
    cs.x_tasks++;
}

void x_block(xdna_seq * seq, const xdna_pfx_row_job & job, int ci, int64_t b, int64_t n_blocks, col_state & cs) {
    const uint32_t c = ROW_COLS[ci];
    if (b % ITER_MAX == 0) {
        x_task(seq, job, ci, b, std::min(ITER_MAX, n_blocks - b), cs);
    }
    const uint32_t k = (uint32_t) (b % W_SLOTS), e = W_LOCK0 + 2 * k, f = e + 1, at = WS_ADDR + k * SLOT_BYTES;
    mt_fill(seq, c, at, CORES * X_BYTES, e, f, cs);
    mt_sends(seq, c, at, X_BYTES, e, f, cs);
}

// Block b on both row-op columns, a tile's drain queued before its first block
void blocks_of_cols(xdna_seq * seq, const xdna_pfx_row_job & job, int64_t b, int64_t n_blocks,
                    const std::vector<drain_part> & plan, col_state * cs) {
    for (int ci = 0; ci < N_RCOLS; ci++) {
        if (b % TILE_BLOCKS == 0) {
            drain_tile(seq, ci, &plan[plan_at(n_blocks / TILE_BLOCKS, ci, b / TILE_BLOCKS)], cs[ci]);
        }
        x_block(seq, job, ci, b, n_blocks, cs[ci]);
    }
}

void stream_blocks(xdna_seq * seq, const xdna_pfx_row_job & job, int64_t n_blocks, const std::vector<drain_part> & plan,
                   col_state * cs) {
    for (int64_t b = 0; b < n_blocks; b++) {
        blocks_of_cols(seq, job, b, n_blocks, plan, cs);
    }
}

void finish_col(xdna_seq * seq, int ci, const col_state & cs) {
    const uint32_t c = ROW_COLS[ci];
    wait_last(seq, c, 1, xdna_dma_dir::S2MM, W_MT_S2MM, cs.mt_s2mm, MT_Q);
    for (uint32_t r = 0; r < 2; r++) {
        wait_last(seq, c, 1, xdna_dma_dir::MM2S, W_MT_MM2S + r, cs.mt_mm2s[r], MT_Q);
    }
    wait_last(seq, c, 0, xdna_dma_dir::MM2S, W_SHIM_CH, cs.shim_pushed, W_RING);
}

void finish_cols(xdna_seq * seq, const col_state * cs) {
    for (int ci = 0; ci < N_RCOLS; ci++) {
        finish_col(seq, ci, cs[ci]);
    }
}

// The C tasks' tokens: the drain ring's last tasks
void wait_c(xdna_seq * seq, const col_state * cs) {
    for (int ci = 0; ci < N_RCOLS; ci++) {
        wait_last(seq, ROW_COLS[ci], 0, xdna_dma_dir::S2MM, 0, cs[ci].drains, DRAIN_RING);
    }
}

std::vector<uint32_t> stream_of(const xdna_pfx_row_job & job, const std::vector<drain_part> & plan) {
    xdna_seq      seq;
    col_state     cs[N_RCOLS];
    const int64_t n_blocks = xdna_pfx_row_job_blocks(job);
    init_slot_locks(&seq);
    a_stream(&seq, n_blocks);
    setup_cols(&seq, cs);
    stream_blocks(&seq, job, n_blocks, plan, cs);
    finish_cols(&seq, cs);
    wait_c(&seq, cs);
    return xdna_seq_build(&seq);
}

// --- the run ---

struct runner {
    std::mutex    mtx;
    xdna_kernel * kern = nullptr;
};

runner g_row;

xdna_kernel * kernel_of(xdna_device * dev) {
    if (!g_row.kern) {
        xdna_kernel * base = xdna_kernel_find(dev, PFX_STEM);
        g_row.kern         = base ? xdna_kernel_clone(base) : nullptr;
    }
    return g_row.kern;
}

bool part_fits(const xdna_buffer * src, const x_part & p, int64_t rows) {
    const uint64_t end = (uint64_t) p.off + (uint64_t) rows * p.ldc * sizeof(float);
    return end <= src->bytes && end < DDR_PATCH_MAX;
}

bool parts_fit(const xdna_pfx_row_job & job, int ci) {
    const x_parts xp = parts_of(job, ci);
    for (int k = 0; k < xp.n; k++) {
        if (!part_fits(job.src, xp.p[k], xdna_pfx_row_job_blocks(job) * BLOCK_ROWS)) {
            return false;
        }
    }
    return true;
}

// dst's layout as the drains assume it: a C object a W object's pair, every key a row of this command
bool dst_ok(const xdna_pfx_row_job & job) {
    const xdna_pfx_attn_dst & d   = *job.dst;
    const size_t              end = d.a_off + d.a_bytes;
    return d.bo && d.pp == DST_PP && d.n_tiles == (job.n_tokens + xdna_pfx_layout::NK - 1) / xdna_pfx_layout::NK &&
           end <= d.bo->bytes && end < DDR_PATCH_MAX;
}

bool job_ok(const xdna_pfx_row_job & job) {
    const int n_dims = job.rope.n_dims;
    if (!job.src || !job.pos || !job.q_norm || !job.k_norm || job.n_tokens <= 0 || n_dims < 0 || n_dims > D ||
        n_dims % ROT_STEP != 0) {
        return false;
    }
    return job.dst && dst_ok(job) && parts_fit(job, 0) && parts_fit(job, 1);
}

bool rope_equal(const xdna_pfx_row_rope & a, const xdna_pfx_row_rope & b) {
    return a.n_dims == b.n_dims && a.n_ctx_orig == b.n_ctx_orig && a.freq_base == b.freq_base &&
           a.freq_scale == b.freq_scale && a.ext_factor == b.ext_factor && a.attn_factor == b.attn_factor &&
           a.beta_fast == b.beta_fast && a.beta_slow == b.beta_slow;
}

bool tables_hold(const xdna_pfx_row_tables & t, const xdna_pfx_row_job & job) {
    return t.a && t.pos == job.pos && t.n_tokens == job.n_tokens && t.n_blocks == xdna_pfx_row_job_blocks(job) &&
           rope_equal(t.rope, job.rope);
}

void tables_free(xdna_pfx_row_tables & t) {
    if (t.a) {
        xdna_kernel_pool_release_buffer(t.pool, t.a);
    }
    t.a   = nullptr;
    t.pos = nullptr;
}

bool tables_pack(xdna_kernel_pool * pool, const xdna_pfx_row_job & job, xdna_pfx_row_tables & t) {
    xdna_prof::section_timer st("prefill: pfx row tables");
    const size_t             bytes = xdna_pfx_row_a_bytes(job);
    if (t.a && (t.pool != pool || t.a->bytes < bytes)) {
        tables_free(t);
    }
    t.pos = nullptr;
    if (!t.a) {
        t.pool = pool;
        t.a    = xdna_kernel_pool_acquire_buffer(pool, bytes);
    }
    if (!t.a || !t.a->data) {
        return false;
    }
    xdna_pfx_row_pack_a(job, (uint16_t *) t.a->data);
    if (!xdna_buffer_sync_to_device_range(t.a, bytes, 0)) {
        return false;
    }
    t.pos      = job.pos;
    t.n_tokens = job.n_tokens;
    t.n_blocks = xdna_pfx_row_job_blocks(job);
    t.rope     = job.rope;
    return true;
}

bool tables_take(xdna_kernel_pool * pool, const xdna_pfx_row_job & job, xdna_pfx_row_tables & t) {
    return (t.pool == pool && tables_hold(t, job)) || tables_pack(pool, job, t);
}

xdna_buffer * setup_of(xdna_kernel_pool * pool, const xdna_pfx_row_job & job) {
    xdna_buffer * setup = xdna_kernel_pool_acquire_buffer(pool, xdna_pfx_row_setup_bytes());
    if (setup && setup->data) {
        xdna_pfx_row_pack_setup(job, (uint8_t *) setup->data);
    }
    return setup;
}

bool run_stream(xdna_kernel * kern, const xdna_pfx_row_job & job, xdna_buffer * a, xdna_buffer * setup) {
    xdna_buffer * args[N_ARGS] = {};
    args[ARG_W]                = job.src;
    args[ARG_A]                = a;
    args[ARG_C]                = job.dst->bo;
    args[ARG_SETUP]            = setup;
    xrt::run run               = xdna_kernel_run_start(kern, args, N_ARGS);
    // dst's objects stay on the device for the attention command
    return xdna_run_wait(run);
}

}  // namespace

int64_t xdna_pfx_row_job_blocks(const xdna_pfx_row_job & job) {
    return job.dst ? TILE_BLOCKS * job.dst->n_tiles : 0;
}

int xdna_pfx_row_stream_word(int layout, int bp, int j, int d, int * half) {
    const int h = d / HALF_D, dl = d % HALF_D;
    if (layout == XDNA_PFX_ROW_LAYOUT_Q) {
        *half = j % 2;
        return HALF_WORDS * h + Q_CHUNK * (dl / 8) + Q_TILE * bp + Q_DIM * (dl % 8) + j / 2;
    }
    if (layout == XDNA_PFX_ROW_LAYOUT_K) {
        *half = d % 2;
        return HALF_WORDS * h + K_BLOCK * bp + K_DHALF * (dl / 64) + K_ROW * j + (dl % 64) / 2;
    }
    const int q = 16 * (dl / 16) + 2 * (dl % 8) + (dl / 8) % 2;  // V^T: the dim's 64 B run of the object
    *half       = j % 2;
    return HALF_WORDS * h + VT_ROW * q + VT_BLOCK * bp + j / 2;
}

size_t xdna_pfx_row_dst_byte(const xdna_pfx_attn_dst & dst, int ci, int64_t o, int i, int s) {
    const drain_part p  = drain_part_of(dst, ci, o / PAIR_OBJS, i);
    const uint32_t   us = (uint32_t) s, i0 = us % p.d0, i1 = (us / p.d0) % p.d1, i2 = us / (p.d0 * p.d1);
    const uint64_t   w  = i0 + (uint64_t) i1 * p.d1_step + (uint64_t) i2 * p.d2_step + (uint64_t) (o % PAIR_OBJS) * p.iter_step;
    return p.off + (size_t) w * sizeof(uint32_t);
}

size_t xdna_pfx_row_a_bytes(const xdna_pfx_row_job & job) {
    return (size_t) (1 + xdna_pfx_row_job_blocks(job)) * A_OBJ_BYTES;
}

size_t xdna_pfx_row_setup_bytes(void) {
    return (size_t) N_RCOLS * CORES * TB_BYTES;
}

void xdna_pfx_row_pack_a(const xdna_pfx_row_job & job, uint16_t * a) {
    std::memset(a, 0, xdna_pfx_row_a_bytes(job));
    pack_header(xdna_pfx_row_job_blocks(job), a);
    pack_tables(job, a + A_OBJ_ELEMS);
}

void xdna_pfx_row_pack_setup(const xdna_pfx_row_job & job, uint8_t * setup) {
    std::memset(setup, 0, xdna_pfx_row_setup_bytes());
    for (int ci = 0; ci < N_RCOLS; ci++) {
        pack_setup_col(job, ci, setup + (size_t) ci * CORES * TB_BYTES);
    }
}

std::vector<uint32_t> xdna_pfx_row_stream(const xdna_pfx_row_job & job) {
    return stream_of(job, drain_plan(*job.dst));
}

xdna_pfx_row_tables::~xdna_pfx_row_tables() {
    tables_free(*this);
}

bool xdna_pfx_row_run(xdna_kernel_pool * pool, const xdna_pfx_row_job & job) {
    xdna_pfx_row_tables tables;
    return xdna_pfx_row_run_tables(pool, job, tables);
}

bool xdna_pfx_row_run_tables(xdna_kernel_pool * pool, const xdna_pfx_row_job & job, xdna_pfx_row_tables & tables) {
    std::lock_guard<std::mutex> lock(g_row.mtx);
    if (!job_ok(job)) {
        return false;
    }
    xdna_kernel * kern = kernel_of(pool->device);
    if (!kern || !tables_take(pool, job, tables)) {
        return false;
    }
    xdna_prof::section_timer    st("prefill: pfx row");
    const std::vector<uint32_t> insts = xdna_pfx_row_stream(job);
    if (!xdna_kernel_bind_insts(pool->device, kern, insts.data(), insts.size())) {
        return false;  // xdna-runtime names the stream and the reason
    }
    xdna_buffer * setup = setup_of(pool, job);
    const bool    ok    = setup && setup->data && xdna_buffer_sync_to_device_range(setup, xdna_pfx_row_setup_bytes(), 0) &&
                          run_stream(kern, job, tables.a, setup);
    if (setup) {
        xdna_kernel_pool_release_buffer(pool, setup);
    }
    return ok;
}
