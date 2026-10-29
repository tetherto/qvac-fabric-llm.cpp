#include "xdna-attn-mm.h"

#include "ggml-impl.h"
#include "xdna-attn-mask.h"
#include "xdna-norm.h"
#include "xdna-prof.h"
#include "xdna-runtime.h"
#include "xdna-seq.h"
#include "xdna-util.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// Host side of kernels/attn_mm.py. Three arguments:
//   0  Q   per column: the header object, then per pass the column's four
//          cores' Q half^T, (d/8, row/8, 8, 8) bf16, Q scaled by log2(e) * the
//          attention scale, each followed by its trailer: the int32 words
//          first key of the pass's first tile, its tile count, then each of
//          the 32 rows' first and last visible key
//   1  KV  per stream (KV head g, D half h, stream 2g + h), per 32-key tile:
//          K half (key/8, d/8, 8, 8) then V half^T (d/8, key/8, 8, 8), bf16
//   2  O   per column, per pass, the four cores' O half as rows (32 x 128)
//          f32
// Core (c, 2 + i) is of group g = c / 4, pair pidx = (c - 4g) * 2 + i / 2,
// half h = i % 2; the pair owns 32 rows of the group's query heads: row =
// head * pp + position, pp = 32 / (query heads of a group), the positions
// pidx * pp .. + pp - 1 of each pass (a pass is 8 pairs of positions). Where
// the streams meet the shim was read off the compiled sample stream: the K/V
// streams on MM2S channel 0 of columns 4g + h, the Q streams on MM2S channel 1
// there and channel 0 elsewhere, every column's O on S2MM channel 0.
//
// Which keys a row sees comes from the mask, row by row (scan_mask): a run of
// keys, so a causal, a bidirectional or a banded mask, packed sequences and
// the cache rows before a ubatch are all the same to the design. A pass streams
// the tiles that hold any key one of its rows sees.

namespace {

using namespace xdna_attn_mask;

constexpr const char * STEM    = "attn_mm_c8";
constexpr int          COLS    = 8;
constexpr int          KVH     = 2;            // KV heads
constexpr int          D       = 256;
constexpr int          DH      = 128;          // a core's half of D
constexpr int          R       = 32;           // rows a pair
constexpr int          NK      = 32;           // keys a tile
constexpr int          PAIRS   = 8;            // pairs a KV head group
constexpr int          TRAILER = 256;          // bf16 after a pass's Q rows
constexpr int          Q_ROWS  = R * DH;       // bf16 a core's Q rows
constexpr int          Q_OBJ   = Q_ROWS + TRAILER;
constexpr int          KV_OBJ  = 2 * NK * DH;  // bf16 a tile object
constexpr int          O_OBJ   = R * DH;       // f32 a core's O object
constexpr int          KV_RING = 3;            // K/V descriptors in flight a stream

// A call pays two hardware-context switches, one into the design and one back
// to the projections around it (~2.5 ms each: probes/attn_probe.cpp --proj 4
// measures 5.0 ms a layer), and its own time is ~0.3 ns a visible (query, key,
// head) against ~1.0 ns on the host. They meet at about 8M of them: Qwen3.5's
// pp512 on the 0.8B at depth 256 .. 1024 (2.6M .. 5.2M) is 14% .. 5% slower on
// the array, at depth 2048 (9.4M) 3% quicker, at 4096 8%.
constexpr int64_t FA_MIN_PAIR_HEADS = 8 << 20;
constexpr int64_t FA_MIN_TOKENS     = 64;  // below it a call's fixed cost is the whole of it

// How the rows of a pair are laid out: the query heads of a KV head group
// share its K/V, and a pair holds all of them at pp positions each.
struct geometry {
    int hg   = 0;  // query heads a KV head group
    int pp   = 0;  // positions a pair
    int pass = 0;  // positions a pass
};

bool geometry_of(int64_t n_head, geometry * g) {
    if (n_head % KVH != 0) {
        return false;
    }
    g->hg = (int) (n_head / KVH);
    if (g->hg < 1 || g->hg > R / 8 || R % g->hg != 0 || (R / g->hg) % 8 != 0) {
        return false;
    }
    g->pp   = R / g->hg;
    g->pass = PAIRS * g->pp;
    return true;
}

int kv_col(int s) {
    return 4 * (s / 2) + s % 2;
}

// A layer's K/V in the tile layout, kept across ubatches: within a sequence
// the cache only grows at the end, so a ubatch packs only its own rows.
struct kv_cache {
    xdna_buffer * bo        = nullptr;
    int64_t       cap_tiles = 0;  // tiles a stream holds
    int64_t       packed    = 0;  // keys packed
    uint64_t      print     = 0;  // of the keys packed (kv_print)
};

// A fingerprint of keys [0, n): the rows 0 and n - 1 of both KV heads, so a
// copy packed for another sequence - one whose later ubatches the host ran -
// is not taken for this one's.
uint64_t kv_print(const ggml_tensor * k, int64_t n) {
    uint64_t h = 1469598103934665603ull;
    for (int64_t j : { (int64_t) 0, n - 1 }) {
        for (int g = 0; g < 2; g++) {
            const uint16_t * r = (const uint16_t *) ((const char *) k->data + j * k->nb[1] + g * k->nb[2]);
            for (int d = 0; d < 64; d++) {
                h = (h ^ r[d]) * 1099511628211ull;
            }
        }
    }
    return h;
}

struct runner {
    std::mutex                                 mtx;
    xdna_kernel *                              kern = nullptr;
    std::unordered_map<const void *, kv_cache> kv;
    // the nodes whose output stays in their output buffer (xdna_attn_mm_keep),
    // and the last such run's
    std::vector<const ggml_tensor *>           keep;
    const ggml_tensor *                        kept      = nullptr;
    xdna_buffer *                              kept_bo   = nullptr;
    xdna_kernel_pool *                         kept_pool = nullptr;
    size_t                                     kept_col  = 0;
    int64_t                                    kept_tok  = 0;
};

runner g_am;

bool artifact_present() {
    static const bool present = !xdna_artifact_find(STEM, false).xclbin.empty();
    return present;
}

// What a mask says about each row: the first and last key it sees. A mask
// that hides a key between two it sees, or shows one with anything but 0, is
// not one the design can follow.
struct mask_rows {
    std::vector<int32_t> lo, hi;
    int64_t              n_kv    = 0;  // keys up to the last one any row sees
    int64_t              visible = 0;  // (row, key) pairs seen
};

// Row t of the mask: the keys it sees, or false.
bool scan_row(const uint16_t * row, int64_t n_cols, int64_t n_kv, int32_t * lo, int32_t * hi) {
    const int64_t first = first_not_hidden(row, n_cols);
    const int64_t last  = last_not_hidden(row, n_cols);
    if (first > last || last >= n_kv || !all_visible(row, first, last)) {
        return false;
    }
    *lo = (int32_t) first;
    *hi = (int32_t) last;
    return true;
}

// m->data is null while the scheduler is still deciding: supports_op runs
// before graph_reserve allocates, so reading the mask here would fault.
bool scan_mask(const ggml_tensor * m, int64_t n_kv, int64_t n_tokens, mask_rows * out) {
    if (!m || !m->data || m->type != GGML_TYPE_F16 || m->ne[0] < n_kv || m->ne[1] < n_tokens || m->ne[2] != 1 ||
        m->ne[3] != 1 || m->nb[0] != sizeof(uint16_t)) {
        return false;
    }
    out->lo.assign((size_t) n_tokens, 0);
    out->hi.assign((size_t) n_tokens, 0);
    int64_t bad = 0, visible = 0, last = 0;
#pragma omp parallel for num_threads(xdna_chunk_threads()) reduction(+ : bad, visible) reduction(max : last)
    for (int64_t t = 0; t < n_tokens; t++) {
        const uint16_t * row = (const uint16_t *) ((const char *) m->data + t * m->nb[1]);
        if (!scan_row(row, m->ne[0], n_kv, &out->lo[t], &out->hi[t])) {
            bad++;
            continue;
        }
        visible += out->hi[t] - out->lo[t] + 1;
        last = std::max<int64_t>(last, out->hi[t]);
    }
    out->n_kv    = last + 1;
    out->visible = visible;
    return bad == 0;
}

// Every attention layer of a graph reads one of two masks (the full one and
// the sliding window's), and each asks twice (supported, then run): scanning
// it each time was 12 whole-mask passes an ubatch on the 0.8B, n_tokens x n_kv
// values each - 8% of a 16k prompt's prefill energy. The answer is kept for
// the graph (xdna_attn_mm_graph_begin starts a new one: a new ubatch rewrites
// the mask in place).
struct mask_memo {
    static constexpr int SLOTS = 4;
    struct slot {
        uint64_t                         at = 0;
        const void *                     data = nullptr;
        int64_t                          n_kv = 0, n_tokens = 0, ne0 = 0;
        size_t                           nb1 = 0;
        std::shared_ptr<const mask_rows> rows;  // null: not a mask the design follows
    };
    std::mutex mtx;
    uint64_t   epoch = 1;
    int        next  = 0;
    slot       slots[SLOTS];
} g_mask;

std::shared_ptr<const mask_rows> mask_of(const ggml_tensor * m, int64_t n_kv, int64_t n_tokens) {
    if (!m || !m->data) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_mask.mtx);
    for (const auto & sl : g_mask.slots) {
        if (sl.at == g_mask.epoch && sl.data == m->data && sl.n_kv == n_kv && sl.n_tokens == n_tokens &&
            sl.ne0 == m->ne[0] && sl.nb1 == m->nb[1]) {
            return sl.rows;
        }
    }
    auto rows = std::make_shared<mask_rows>();
    if (!scan_mask(m, n_kv, n_tokens, rows.get())) {
        rows = nullptr;
    }
    auto & sl    = g_mask.slots[g_mask.next];
    g_mask.next  = (g_mask.next + 1) % mask_memo::SLOTS;
    sl.at        = g_mask.epoch;
    sl.data      = m->data;
    sl.n_kv      = n_kv;
    sl.n_tokens  = n_tokens;
    sl.ne0       = m->ne[0];
    sl.nb1       = m->nb[1];
    sl.rows      = rows;
    return rows;
}

// The tiles [tile_lo, tile_hi) of a pass: those that hold any key one of its
// rows sees.
struct pass_tiles {
    int64_t tile_lo = 0, tile_hi = 0;
};

pass_tiles tiles_of_pass(const mask_rows & mr, const geometry & geo, int64_t n_tokens, int p) {
    const int64_t t0 = (int64_t) p * geo.pass, t1 = std::min<int64_t>(n_tokens, t0 + geo.pass);
    int32_t       lo = mr.lo[t0], hi = mr.hi[t0];
    for (int64_t t = t0 + 1; t < t1; t++) {
        lo = std::min(lo, mr.lo[t]);
        hi = std::max(hi, mr.hi[t]);
    }
    return { lo / NK, hi / NK + 1 };
}

std::vector<pass_tiles> tiles_of_passes(const mask_rows & mr, const geometry & geo, int64_t n_tokens, int npass) {
    std::vector<pass_tiles> passes((size_t) npass);
    for (int p = 0; p < npass; p++) {
        passes[p] = tiles_of_pass(mr, geo, n_tokens, p);
    }
    return passes;
}

// One past the last tile any pass streams: what a layer's K/V copy must hold.
int64_t tiles_reached(const std::vector<pass_tiles> & passes) {
    int64_t reached = 0;
    for (const pass_tiles & pt : passes) {
        reached = std::max(reached, pt.tile_hi);
    }
    return reached;
}

xdna_bd linear_bd(uint32_t words) {
    xdna_bd bd;
    bd.buf_len   = words;
    bd.d0_stride = bd.d1_stride = bd.d2_stride = 1;
    bd.ax_cache                                = 2;
    return bd;
}

std::vector<uint32_t> build_seq(const std::vector<pass_tiles> & passes, int64_t cap_tiles) {
    xdna_seq       seq;
    const int      npass     = (int) passes.size();
    const uint32_t q_col     = (uint32_t) (1 + npass) * 4 * Q_OBJ * 2;  // bytes a column
    const uint32_t o_col     = (uint32_t) npass * 4 * O_OBJ * 4;
    const uint32_t kv_stream = (uint32_t) cap_tiles * KV_OBJ * 2;
    auto           is_kv_col = [](int c) {
        return c == 0 || c == 1 || c == 4 || c == 5;
    };
    for (int c = 0; c < COLS; c++) {
        xdna_bd bd = linear_bd(q_col / 4);
        xdna_seq_blockwrite(&seq, c, 0, 0, &bd);
        xdna_seq_ddr_patch(&seq, c, 0, 0, 0, (uint32_t) c * q_col);
        xdna_seq_push_queue(&seq, c, 0, 0, xdna_dma_dir::MM2S, is_kv_col(c) ? 1 : 0, false, 0);
    }
    for (int c = 0; c < COLS; c++) {
        const uint32_t id = is_kv_col(c) ? 1 + KV_RING : 1;
        xdna_bd        bd = linear_bd(o_col / 4);
        xdna_seq_blockwrite(&seq, c, 0, id, &bd);
        xdna_seq_ddr_patch(&seq, c, 0, id, 2, (uint32_t) c * o_col);
        xdna_seq_issue_token(&seq, c, 0, xdna_dma_dir::S2MM, 0, 0xF);
        xdna_seq_push_queue(&seq, c, 0, id, xdna_dma_dir::S2MM, 0, true, 0);
    }
    // K/V: a pass streams the tiles it needs, a descriptor a pass per stream,
    // round a ring whose descriptors are reused only once their transfer is done
    for (int p = 0; p < npass; p++) {
        if (p >= KV_RING - 1) {
            for (int s = 0; s < 4; s++) {
                xdna_seq_wait_token(&seq, kv_col(s), 0, xdna_dma_dir::MM2S, 0);
            }
        }
        const uint32_t id = 1 + (uint32_t) (p % KV_RING);
        for (int s = 0; s < 4; s++) {
            const pass_tiles & pt = passes[p];
            xdna_bd bd = linear_bd((uint32_t) (pt.tile_hi - pt.tile_lo) * KV_OBJ * 2 / 4);
            xdna_seq_blockwrite(&seq, kv_col(s), 0, id, &bd);
            xdna_seq_ddr_patch(&seq, kv_col(s), 0, id, 1,
                               (uint32_t) ((uint64_t) s * kv_stream + (uint64_t) pt.tile_lo * KV_OBJ * 2));
            xdna_seq_issue_token(&seq, kv_col(s), 0, xdna_dma_dir::MM2S, 0, 0xF);
            xdna_seq_push_queue(&seq, kv_col(s), 0, id, xdna_dma_dir::MM2S, 0, true, 0);
        }
    }
    for (int p = std::max(0, npass - (KV_RING - 1)); p < npass; p++) {
        for (int s = 0; s < 4; s++) {
            xdna_seq_wait_token(&seq, kv_col(s), 0, xdna_dma_dir::MM2S, 0);
        }
    }
    for (int c = 0; c < COLS; c++) {
        xdna_seq_wait_token(&seq, c, 0, xdna_dma_dir::S2MM, 0);
    }
    return xdna_seq_build(&seq);
}

// Keys [j0, n_kv) of both KV heads into the tile layout, whole tiles (the
// keys of the first one before j0 again, the same values); the rest of the
// last tile zero (finite: every query before them masks them). Rows convert
// in vectors, then scatter into the tiles.
void pack_kv(const ggml_tensor * k,
             const ggml_tensor * v,
             uint16_t *          dst,
             int64_t             cap_tiles,
             int64_t             j0,
             int64_t             n_kv) {
    const int64_t t0 = j0 / NK, t1 = (n_kv + NK - 1) / NK;
#pragma omp parallel for collapse(2) num_threads(xdna_chunk_threads())
    for (int64_t t = t0; t < t1; t++) {
        for (int s = 0; s < 4; s++) {
            const int            g = s / 2, h = s % 2;
            uint16_t *           o  = dst + ((size_t) s * cap_tiles + t) * KV_OBJ;
            uint16_t *           ov = o + (size_t) NK * DH;
            alignas(64) uint16_t kb[NK][DH];
            alignas(64) uint16_t vb[NK][DH];
            for (int jj = 0; jj < NK; jj++) {
                const int64_t j = t * NK + jj;
                if (j < n_kv) {
                    xdna_f16_to_bf16_row(
                        (const uint16_t *) ((const char *) k->data + j * k->nb[1] + g * k->nb[2]) + (size_t) h * DH,
                        kb[jj], DH);
                    xdna_f16_to_bf16_row(
                        (const uint16_t *) ((const char *) v->data + j * v->nb[1] + g * v->nb[2]) + (size_t) h * DH,
                        vb[jj], DH);
                } else {
                    std::memset(kb[jj], 0, sizeof(kb[jj]));
                    std::memset(vb[jj], 0, sizeof(vb[jj]));
                }
            }
            // K (key/8, d/8, 8, 8): a key's 8 d a run
            for (int jj = 0; jj < NK; jj++) {
                for (int d8 = 0; d8 < DH / 8; d8++) {
                    std::memcpy(o + (size_t) ((jj / 8) * (DH / 8) + d8) * 64 + (size_t) (jj % 8) * 8,
                                &kb[jj][(size_t) d8 * 8], 16);
                }
            }
            // V^T (d/8, key/8, 8, 8): a d's 8 keys a run
            for (int d = 0; d < DH; d++) {
                for (int jj = 0; jj < NK; jj++) {
                    ov[((d / 8) * (NK / 8) + jj / 8) * 64 + (d % 8) * 8 + jj % 8] = vb[jj][d];
                }
            }
        }
    }
}

// Tiles [t0, t1) of each of the 4 K/V streams to the device.
bool sync_kv_tiles(xdna_buffer * bo, int64_t cap_tiles, int64_t t0, int64_t t1) {
    for (int s = 0; s < 4; s++) {
        if (!xdna_buffer_sync_to_device_range(bo, (size_t) (t1 - t0) * KV_OBJ * 2,
                                              ((size_t) s * cap_tiles + t0) * KV_OBJ * 2)) {
            return false;
        }
    }
    return true;
}

// The scale the host puts into Q, as the exponent of 2 a score is: 1 / sqrt(D)
// for Qwen3.5, 1 for a model that normalises q and k.
float q_scale_of(const ggml_tensor * node) {
    float params[3] = { 0.0f, 0.0f, 0.0f };
    std::memcpy(params, node->op_params, sizeof(params));
    return params[0];
}

bool shapes_supported(const ggml_tensor * node, geometry * geo, int64_t * n_tokens, int64_t * n_kv) {
    const ggml_tensor * q = node->src[0];
    const ggml_tensor * k = node->src[1];
    const ggml_tensor * v = node->src[2];
    if (!q || !k || !v || node->src[4]) {
        return false;  // sinks
    }
    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16 ||
        node->type != GGML_TYPE_F32) {
        return false;
    }
    // q is [D, n_tokens, n_head], k/v are [D, n_kv, n_head_kv]
    if (q->ne[0] != D || q->ne[3] != 1 || k->ne[0] != D || v->ne[0] != D || k->ne[2] != KVH || v->ne[2] != KVH ||
        k->ne[1] != v->ne[1] || k->ne[3] != 1 || v->ne[3] != 1) {
        return false;
    }
    if (q->nb[0] != sizeof(float) || k->nb[0] != sizeof(uint16_t) || v->nb[0] != sizeof(uint16_t)) {
        return false;
    }
    *n_tokens = q->ne[1];
    *n_kv     = k->ne[1];
    return geometry_of(q->ne[2], geo) && *n_tokens >= FA_MIN_TOKENS && *n_kv >= 1;
}

}  // namespace

// GGML_XDNA_FA_MM: 0 the host, 1 the array from FA_MIN_PAIR_HEADS visible
// (query, key, head) triples a call, 2 the array whatever the size (tests,
// bisection).
bool xdna_attn_mm_supported(const ggml_tensor * node) {
    const int mode = xdna_env_int("GGML_XDNA_FA_MM", 1);
    if (!node || node->op != GGML_OP_FLASH_ATTN_EXT || mode == 0 || !artifact_present()) {
        return false;
    }
    geometry geo;
    int64_t  n_tokens, n_kv;
    if (!shapes_supported(node, &geo, &n_tokens, &n_kv)) {
        return false;
    }
    const int64_t n_head = (int64_t) geo.hg * KVH;
    // the most it can see, before the mask is scanned
    if (mode != 2 && n_tokens * n_kv * n_head < FA_MIN_PAIR_HEADS) {
        return false;
    }
    float params[3] = { 0.0f, 0.0f, 0.0f };
    std::memcpy(params, node->op_params, sizeof(params));
    if (!(params[0] > 0.0f) || params[1] != 0.0f || params[2] != 0.0f) {
        return false;  // the scale is in the host's Q; no ALiBi, no softcap
    }
    const std::shared_ptr<const mask_rows> mr = mask_of(node->src[3], n_kv, n_tokens);
    return mr && (mode == 2 || mr->visible * n_head >= FA_MIN_PAIR_HEADS);
}

namespace {

// Q for the columns: per column the header, then per pass and core Q half^T of
// its 32 rows and the pass trailer; past the ubatch the rows are zero and
// see the first key of the pass.
void pack_q(const ggml_tensor *            q,
            uint16_t *                     qh,
            const geometry &               geo,
            const mask_rows &              mr,
            const std::vector<pass_tiles> & passes,
            float                          qs,
            int64_t                        n_tokens) {
    const int     npass = (int) passes.size();
    const size_t  q_col = (size_t) (1 + npass) * 4 * Q_OBJ;  // bf16 a column
#pragma omp parallel for num_threads(xdna_chunk_threads())
    for (int cp = 0; cp < COLS * (1 + npass); cp++) {
        const int  c = cp / (1 + npass), p = cp % (1 + npass) - 1;
        uint16_t * col = qh + (size_t) c * q_col + (size_t) (p + 1) * 4 * Q_OBJ;
        if (p < 0) {
            std::memset(col, 0, (size_t) 4 * Q_OBJ * 2);
            for (int i = 0; i < 4; i++) {
                int32_t * hw = (int32_t *) (col + (size_t) i * Q_OBJ);
                hw[0]        = npass;
            }
            continue;
        }
        const int g = c / 4;
        for (int i = 0; i < 4; i++) {
            const int            pidx = (c - 4 * g) * 2 + i / 2, h = i % 2;
            uint16_t *           o = col + (size_t) i * Q_OBJ;
            int32_t *            tr = (int32_t *) (o + Q_ROWS);
            const int32_t        first_key = (int32_t) (passes[p].tile_lo * NK);
            alignas(64) uint16_t qb[R][DH];
            tr[0] = first_key;
            tr[1] = (int32_t) (passes[p].tile_hi - passes[p].tile_lo);
            for (int row = 0; row < R; row++) {
                const int     hh = row / geo.pp;
                const int64_t t  = (int64_t) p * geo.pass + (int64_t) pidx * geo.pp + row % geo.pp;
                if (t < n_tokens) {
                    xdna_scaled_bf16_row(
                        (const float *) ((const char *) q->data + t * q->nb[1] + (geo.hg * g + hh) * q->nb[2]) +
                            (size_t) h * DH,
                        qs, qb[row], DH);
                    tr[2 + row]     = mr.lo[t];
                    tr[2 + R + row] = mr.hi[t];
                } else {
                    std::memset(qb[row], 0, sizeof(qb[row]));
                    tr[2 + row]     = first_key;
                    tr[2 + R + row] = first_key;
                }
            }
            for (int d = 0; d < DH; d++) {
                for (int row = 0; row < R; row++) {
                    o[((d / 8) * (R / 8) + row / 8) * 64 + (d % 8) * 8 + row % 8] = qb[row][d];
                }
            }
        }
    }
}

// O into dst, [D, n_head, n_tokens]: the MemTile leaves each core's rows as
// rows.
void unpack_o(ggml_tensor *    node,
              const float *    ob,
              const geometry & geo,
              int              npass,
              int64_t          n_tokens) {
    const size_t o_col = (size_t) npass * 4 * O_OBJ;  // f32 a column
#pragma omp parallel for num_threads(xdna_chunk_threads())
    for (int cp = 0; cp < COLS * npass; cp++) {
        const int c = cp / npass, p = cp % npass, g = c / 4;
        for (int i = 0; i < 4; i++) {
            const int     pidx = (c - 4 * g) * 2 + i / 2, h = i % 2;
            const float * o = ob + (size_t) c * o_col + ((size_t) p * 4 + i) * O_OBJ;
            for (int row = 0; row < R; row++) {
                const int64_t t = (int64_t) p * geo.pass + (int64_t) pidx * geo.pp + row % geo.pp;
                if (t >= n_tokens) {
                    continue;
                }
                float * dst = (float *) ((char *) node->data + (geo.hg * g + row / geo.pp) * node->nb[1] +
                                         t * node->nb[2]) +
                              (size_t) h * DH;
                std::memcpy(dst, o + (size_t) row * DH, DH * sizeof(float));
            }
        }
    }
}

}  // namespace

bool xdna_attn_mm_run(xdna_kernel_pool * pool, ggml_tensor * node) {
    std::lock_guard<std::mutex> lock(g_am.mtx);
    const ggml_tensor *         q = node->src[0];
    const ggml_tensor *         k = node->src[1];
    const ggml_tensor *         v = node->src[2];
    geometry                    geo;
    int64_t                     n_tokens, n_kv_view;
    if (!shapes_supported(node, &geo, &n_tokens, &n_kv_view)) {
        return false;
    }
    const std::shared_ptr<const mask_rows> mrp = mask_of(node->src[3], n_kv_view, n_tokens);
    if (!mrp) {
        GGML_LOG_ERROR("%s: the mask is not one the design follows\n", "xdna-attn-mm");
        return false;
    }
    const mask_rows & mr = *mrp;
    // the keys up to the last any row sees; past them llama's padding
    const int64_t     n_kv  = mr.n_kv;
    // the keys before the ubatch's own: the cache's context
    const int64_t     p0    = std::max<int64_t>(0, n_kv - n_tokens);
    const int         npass = (int) ((n_tokens + geo.pass - 1) / geo.pass);

    const std::vector<pass_tiles> passes = tiles_of_passes(mr, geo, n_tokens, npass);
    const int64_t                 t_need = tiles_reached(passes);

    if (!g_am.kern) {
        g_am.kern = xdna_kernel_find(pool->device, STEM);
        if (!g_am.kern) {
            return false;
        }
    }

    // K/V: this layer's copy, grown to the last tile a pass needs; a ubatch
    // packs its own rows, and all of them when the sequence starts over
    kv_cache & kc = g_am.kv[k->data];
    // a copy that ends where this ubatch starts, of this sequence's keys
    int64_t    j0 = (p0 > 0 && kc.packed == p0 && kc.print == kv_print(k, p0)) ? p0 : 0;
    if (!kc.bo || kc.cap_tiles < t_need) {
        if (kc.bo) {
            xdna_buffer_free(kc.bo);
        }
        kc.cap_tiles = (t_need + 255) / 256 * 256;
        kc.bo        = xdna_buffer_alloc(pool->device, (size_t) 4 * kc.cap_tiles * KV_OBJ * 2);
        if (!kc.bo) {
            kc.cap_tiles = 0;
            return false;
        }
        j0 = 0;
    }
    if (!kc.bo->data) {
        return false;
    }
    uint16_t * kvh = (uint16_t *) kc.bo->data;
    // the tiles past n_kv that a pass streams are zero from the allocation
    // or a longer earlier sequence - finite either way, and masked
    const int64_t t0 = j0 / NK, t1 = (n_kv + NK - 1) / NK;
    {
        xdna_prof::section_timer st("prefill: attn kv pack");
        pack_kv(k, v, kvh, kc.cap_tiles, j0, n_kv);
        if (!sync_kv_tiles(kc.bo, kc.cap_tiles, t0, t1)) {
            return false;
        }
    }
    kc.packed = n_kv;
    kc.print  = kv_print(k, n_kv);

    const std::vector<uint32_t> insts = build_seq(passes, kc.cap_tiles);
    if (!xdna_kernel_bind_insts(pool->device, g_am.kern, insts.data(), insts.size())) {
        return false;  // xdna-runtime names the stream and the reason
    }

    if (g_am.kept_bo) {
        xdna_kernel_pool_release_buffer(g_am.kept_pool, g_am.kept_bo);
        g_am.kept_bo = nullptr;
        g_am.kept    = nullptr;
    }
    // the kept rows are read in Qwen3.5's layout of 8 heads over 2 groups
    const bool    keep  = geo.hg == 4 && std::find(g_am.keep.begin(), g_am.keep.end(), node) != g_am.keep.end();
    const size_t  q_col = (size_t) (1 + npass) * 4 * Q_OBJ;  // bf16 a column
    const size_t  o_col = (size_t) npass * 4 * O_OBJ;        // f32 a column
    xdna_buffer * bo_q  = xdna_kernel_pool_acquire_buffer(pool, COLS * q_col * 2);
    xdna_buffer * bo_o  = xdna_kernel_pool_acquire_buffer(pool, COLS * o_col * 4);
    if (!bo_q || !bo_o || !bo_q->data || !bo_o->data) {
        return false;
    }

    {
        xdna_prof::section_timer q_st("prefill: attn q pack");
        pack_q(q, (uint16_t *) bo_q->data, geo, mr, passes, (float) (M_LOG2E * q_scale_of(node)), n_tokens);
        // the pool's BO may be larger
        if (!xdna_buffer_sync_to_device_range(bo_q, COLS * q_col * 2, 0)) {
            return false;
        }
    }

    xdna_buffer * args[3] = { bo_q, kc.bo, bo_o };
    bool          ok;
    {
        xdna_prof::section_timer run_st("prefill: attn run");
        xrt::run                 run    = xdna_kernel_run_start(g_am.kern, args, 3);
        const bool               run_ok = xdna_run_wait(run);
        ok                              = run_ok && xdna_buffer_sync_from_device_range(bo_o, COLS * o_col * 4, 0);
    }
    if (ok && !keep) {
        xdna_prof::section_timer o_st("prefill: attn unpack");
        unpack_o(node, (const float *) bo_o->data, geo, npass, n_tokens);
    }
    xdna_kernel_pool_release_buffer(pool, bo_q);
    if (ok && keep) {
        g_am.kept      = node;
        g_am.kept_bo   = bo_o;
        g_am.kept_pool = pool;
        g_am.kept_col  = o_col;
        g_am.kept_tok  = n_tokens;
    } else {
        xdna_kernel_pool_release_buffer(pool, bo_o);
    }
    return ok;
}

void xdna_attn_mm_graph_begin(void) {
    std::lock_guard<std::mutex> lock(g_mask.mtx);
    g_mask.epoch++;
}

void xdna_attn_mm_release(void) {
    std::lock_guard<std::mutex> lock(g_am.mtx);
    for (auto & kv : g_am.kv) {
        xdna_buffer_free(kv.second.bo);
    }
    g_am.kv.clear();
    if (g_am.kept_bo) {
        xdna_kernel_pool_release_buffer(g_am.kept_pool, g_am.kept_bo);
        g_am.kept_bo = nullptr;
        g_am.kept    = nullptr;
    }
}

void xdna_attn_mm_keep_clear(void) {
    std::lock_guard<std::mutex> lock(g_am.mtx);
    g_am.keep.clear();
}

void xdna_attn_mm_keep(const ggml_tensor * node) {
    std::lock_guard<std::mutex> lock(g_am.mtx);
    g_am.keep.push_back(node);
}

bool xdna_attn_mm_rows(const ggml_tensor * node, xdna_attn_rows * rows) {
    std::lock_guard<std::mutex> lock(g_am.mtx);
    if (!node || g_am.kept != node || !g_am.kept_bo) {
        return false;
    }
    rows->base     = (const float *) g_am.kept_bo->data;
    rows->o_col    = g_am.kept_col;
    rows->n_tokens = g_am.kept_tok;
    return true;
}

bool xdna_attn_mm_materialize(const ggml_tensor * node) {
    std::lock_guard<std::mutex> lock(g_am.mtx);
    if (!node || g_am.kept != node || !g_am.kept_bo) {
        return false;
    }
    xdna_attn_rows r;
    r.base     = (const float *) g_am.kept_bo->data;
    r.o_col    = g_am.kept_col;
    r.n_tokens = g_am.kept_tok;
#pragma omp parallel for num_threads(xdna_chunk_threads())
    for (int64_t t = 0; t < r.n_tokens; t++) {
        for (int h = 0; h < (int) node->ne[1]; h++) {
            const float * p0  = xdna_attn_mm_row(&r, t, h);
            float *       dst = (float *) ((char *) node->data + h * node->nb[1] + t * node->nb[2]);
            std::memcpy(dst, p0, DH * sizeof(float));
            std::memcpy(dst + DH, p0 + O_OBJ, DH * sizeof(float));
        }
    }
    return true;
}
