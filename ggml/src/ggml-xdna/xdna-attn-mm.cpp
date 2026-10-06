#include "xdna-attn-mm.h"

#include "ggml-impl.h"
#include "xdna-runtime.h"
#include "xdna-seq.h"
#include "xdna-util.h"
#include "xdna-norm.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// Host side of kernels/attn_mm.py. Three arguments:
//   0  Q   per column: the header object, then per pass the column's four
//          cores' Q half^T, (d/8, row/8, 8, 8) bf16, Q scaled by log2(e) / 16
//   1  KV  per stream (KV head g, D half h, stream 2g + h), per 32-key tile:
//          K half (key/8, d/8, 8, 8) then V half^T (d/8, key/8, 8, 8), bf16
//   2  O   per column, per pass, the four cores' O half as rows (32 x 128)
//          f32
// Core (c, 2 + i) is of group g = c / 4, pair pidx = (c - 4g) * 2 + i / 2,
// half h = i % 2; the pair owns the group's 4 query heads at positions pidx *
// 8 .. + 7 of each 64-position pass (row = head * 8 + position). Where the
// streams meet the shim was read off the compiled sample stream: the K/V
// streams on MM2S channel 0 of columns 4g + h, the Q streams on MM2S channel 1
// there and channel 0 elsewhere, every column's O on S2MM channel 0.

namespace {

constexpr const char * STEM = "attn_mm_c8";
constexpr int COLS = 8;
constexpr int H    = 8;                 // query heads
constexpr int KVH  = 2;                 // KV heads
constexpr int D    = 256;
constexpr int DH   = 128;               // a core's half of D
constexpr int R    = 32;                // rows a pair
constexpr int NK   = 32;                // keys a tile
constexpr int PASS = 64;                // positions a pass
constexpr int Q_OBJ  = R * DH;          // bf16 a core's Q object
constexpr int KV_OBJ = 2 * NK * DH;     // bf16 a tile object
constexpr int O_OBJ  = R * DH;          // f32 a core's O object
constexpr int KV_RING = 3;              // K/V descriptors in flight a stream
constexpr float SCALE = 0.0625f;

int kv_col(int s) { return 4 * (s / 2) + s % 2; }

int tiles_of(int64_t p0, int p) { return (int) ((p0 + (int64_t) p * PASS + PASS + NK - 1) / NK); }

int host_threads() {
    static const int n = std::max(1, std::min(16, (int) std::thread::hardware_concurrency() / 2));
    return n;
}

// A layer's K/V in the tile layout, kept across ubatches: within a sequence
// the cache only grows at the end, so a ubatch packs only its own rows.
struct kv_cache {
    xdna_buffer * bo = nullptr;
    int64_t cap_tiles = 0;              // tiles a stream holds
    int64_t packed = 0;                 // keys packed
    uint64_t print = 0;                 // of the keys packed (kv_print)
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
    std::mutex mtx;
    xdna_kernel * kern = nullptr;
    std::unordered_map<const void *, kv_cache> kv;
    // the nodes whose output stays in their output buffer (xdna_attn_mm_keep),
    // and the last such run's
    std::vector<const ggml_tensor *> keep;
    const ggml_tensor * kept      = nullptr;
    xdna_buffer *       kept_bo   = nullptr;
    xdna_kernel_pool *  kept_pool = nullptr;
    size_t              kept_col  = 0;
    int64_t             kept_tok  = 0;
};

runner g_am;

bool artifact_present() {
    static const bool present = [] {
        for (const auto & dir : xdna_kernel_search_dirs()) {
            std::error_code ec;
            if (std::filesystem::exists(dir / (std::string(STEM) + ".xclbin"), ec)) {
                return true;
            }
        }
        return false;
    }();
    return present;
}

// The design derives its mask from the positions, so anything but a plain
// causal mask would be silently wrong: the positions before the ubatch are
// read off its first row (llama pads n_kv past them, the padding masked),
// and every row is checked all the way across. -1: not causal.
int64_t mask_past(const ggml_tensor * m, int64_t n_kv, int64_t n_tokens) {
    // m->data is null while the scheduler is still deciding: supports_op runs
    // before graph_reserve allocates, so reading the mask here would fault.
    if (!m || !m->data || m->type != GGML_TYPE_F16 || m->ne[0] < n_kv || m->ne[1] < n_tokens ||
        m->ne[2] != 1 || m->ne[3] != 1) {
        return -1;
    }
    const ggml_fp16_t * row0 = (const ggml_fp16_t *) m->data;
    int64_t npast = -1;
    while (npast + 1 < n_kv && ggml_fp16_to_fp32(row0[npast + 1]) == 0.0f) {
        npast++;
    }
    if (npast < 0 || npast + n_tokens > n_kv) {
        return -1;
    }
    // Every row, not a sample of three: the design derives its own causal mask
    // from the positions, so a row it would hide a key for has to be rejected.
    for (int64_t t = 0; t < n_tokens; t++) {
        const ggml_fp16_t * row = (const ggml_fp16_t *) ((const char *) m->data + t * m->nb[1]);
        for (int64_t j = 0; j < n_kv; j++) {
            const float v = ggml_fp16_to_fp32(row[j]);
            const bool visible = j <= npast + t;
            if (visible ? v != 0.0f : !(v < -1.0e30f)) {
                return -1;
            }
        }
    }
    return npast;
}

xdna_bd linear_bd(uint32_t words) {
    xdna_bd bd;
    bd.buf_len   = words;
    bd.d0_stride = bd.d1_stride = bd.d2_stride = 1;
    bd.ax_cache  = 2;
    return bd;
}

std::vector<uint32_t> build_seq(int npass, int64_t p0, int64_t cap_tiles) {
    xdna_seq seq;
    const uint32_t q_col = (uint32_t) (1 + npass) * 4 * Q_OBJ * 2;     // bytes a column
    const uint32_t o_col = (uint32_t) npass * 4 * O_OBJ * 4;
    const uint32_t kv_stream = (uint32_t) cap_tiles * KV_OBJ * 2;
    auto is_kv_col = [](int c) { return c == 0 || c == 1 || c == 4 || c == 5; };
    for (int c = 0; c < COLS; c++) {
        xdna_bd bd = linear_bd(q_col / 4);
        xdna_seq_blockwrite(&seq, c, 0, 0, &bd);
        xdna_seq_ddr_patch(&seq, c, 0, 0, 0, (uint32_t) c * q_col);
        xdna_seq_push_queue(&seq, c, 0, 0, xdna_dma_dir::MM2S, is_kv_col(c) ? 1 : 0, false, 0);
    }
    for (int c = 0; c < COLS; c++) {
        const uint32_t id = is_kv_col(c) ? 1 + KV_RING : 1;
        xdna_bd bd = linear_bd(o_col / 4);
        xdna_seq_blockwrite(&seq, c, 0, id, &bd);
        xdna_seq_ddr_patch(&seq, c, 0, id, 2, (uint32_t) c * o_col);
        xdna_seq_issue_token(&seq, c, 0, xdna_dma_dir::S2MM, 0, 0xF);
        xdna_seq_push_queue(&seq, c, 0, id, xdna_dma_dir::S2MM, 0, true, 0);
    }
    // K/V: a pass streams tiles 0 .. its last, a descriptor a pass per stream,
    // round a ring whose descriptors are reused only once their transfer is done
    for (int p = 0; p < npass; p++) {
        if (p >= KV_RING - 1) {
            for (int s = 0; s < 4; s++) {
                xdna_seq_wait_token(&seq, kv_col(s), 0, xdna_dma_dir::MM2S, 0);
            }
        }
        const uint32_t id = 1 + (uint32_t) (p % KV_RING);
        for (int s = 0; s < 4; s++) {
            xdna_bd bd = linear_bd((uint32_t) tiles_of(p0, p) * KV_OBJ * 2 / 4);
            xdna_seq_blockwrite(&seq, kv_col(s), 0, id, &bd);
            xdna_seq_ddr_patch(&seq, kv_col(s), 0, id, 1, (uint32_t) s * kv_stream);
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
void pack_kv(const ggml_tensor * k, const ggml_tensor * v, uint16_t * dst, int64_t cap_tiles,
             int64_t j0, int64_t n_kv) {
    const int64_t t0 = j0 / NK, t1 = (n_kv + NK - 1) / NK;
#pragma omp parallel for collapse(2) num_threads(host_threads())
    for (int64_t t = t0; t < t1; t++) {
        for (int s = 0; s < 4; s++) {
            const int g = s / 2, h = s % 2;
            uint16_t * o = dst + ((size_t) s * cap_tiles + t) * KV_OBJ;
            uint16_t * ov = o + NK * DH;
            alignas(64) uint16_t kb[NK][DH];
            alignas(64) uint16_t vb[NK][DH];
            for (int jj = 0; jj < NK; jj++) {
                const int64_t j = t * NK + jj;
                if (j < n_kv) {
                    xdna_f16_to_bf16_row((const uint16_t *) ((const char *) k->data + j * k->nb[1] + g * k->nb[2]) + h * DH,
                                         kb[jj], DH);
                    xdna_f16_to_bf16_row((const uint16_t *) ((const char *) v->data + j * v->nb[1] + g * v->nb[2]) + h * DH,
                                         vb[jj], DH);
                } else {
                    std::memset(kb[jj], 0, sizeof(kb[jj]));
                    std::memset(vb[jj], 0, sizeof(vb[jj]));
                }
            }
            // K (key/8, d/8, 8, 8): a key's 8 d a run
            for (int jj = 0; jj < NK; jj++) {
                for (int d8 = 0; d8 < DH / 8; d8++) {
                    std::memcpy(o + ((jj / 8) * (DH / 8) + d8) * 64 + (jj % 8) * 8, &kb[jj][d8 * 8], 16);
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

} // namespace

bool xdna_attn_mm_supported(const ggml_tensor * node) {
    if (!node || node->op != GGML_OP_FLASH_ATTN_EXT || xdna_env_int("GGML_XDNA_FA_MM", 1) == 0 ||
        !artifact_present()) {
        return false;
    }
    const ggml_tensor * q = node->src[0];
    const ggml_tensor * k = node->src[1];
    const ggml_tensor * v = node->src[2];
    if (!q || !k || !v || node->src[4]) {
        return false;   // sinks
    }
    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16 ||
        node->type != GGML_TYPE_F32) {
        return false;
    }
    // q is [D, n_tokens, n_head], k/v are [D, n_kv, n_head_kv]
    if (q->ne[0] != D || q->ne[2] != H || q->ne[3] != 1 || k->ne[0] != D || v->ne[0] != D ||
        k->ne[2] != KVH || v->ne[2] != KVH || k->ne[1] != v->ne[1]) {
        return false;
    }
    const int64_t n_tokens = q->ne[1];
    const int64_t n_kv     = k->ne[1];
    // Below a pass (a prompt's last few tokens, the server's small batches)
    // a call's fixed cost is the whole of it.
    if (n_tokens < PASS || n_kv < n_tokens) {
        return false;
    }
    // Everything else runs here: the prefill belongs on the array. Below ~1M
    // (token, key) pairs the host would be quicker (it takes ~11 ns a pair,
    // the array ~3 plus a few ms a call - the layouts and the stream), which
    // is the call's own cost to bring down; GGML_XDNA_FA_MM_MIN measures it.
    static const int64_t min_pairs = xdna_env_int("GGML_XDNA_FA_MM_MIN", 0);
    if (n_tokens * (n_kv - n_tokens / 2) < min_pairs) {
        return false;
    }
    float params[3] = { 0.0f, 0.0f, 0.0f };
    std::memcpy(params, node->op_params, sizeof(params));
    if (params[0] != SCALE || params[1] != 0.0f || params[2] != 0.0f) {
        return false;   // the scale is in the host's Q; no ALiBi, no softcap
    }
    return mask_past(node->src[3], n_kv, n_tokens) >= 0;
}

bool xdna_attn_mm_run(xdna_kernel_pool * pool, ggml_tensor * node) {
    std::lock_guard<std::mutex> lock(g_am.mtx);
    const ggml_tensor * q = node->src[0];
    const ggml_tensor * k = node->src[1];
    const ggml_tensor * v = node->src[2];
    const int64_t n_tokens = q->ne[1];
    // the keys up to the ubatch's last position; past them llama's padding
    const int64_t p0       = mask_past(node->src[3], k->ne[1], n_tokens);
    const int64_t n_kv     = p0 + n_tokens;
    const int     npass    = (int) ((n_tokens + PASS - 1) / PASS);
    if (p0 < 0) {
        return false;
    }
    const int64_t t_need   = tiles_of(p0, npass - 1);

    if (!g_am.kern) {
        for (const auto & dir : xdna_kernel_search_dirs()) {
            const std::filesystem::path x = dir / (std::string(STEM) + ".xclbin");
            std::error_code ec;
            if (std::filesystem::exists(x, ec)) {
                g_am.kern = xdna_kernel_load_hw(pool->device, x.c_str());
                break;
            }
        }
        if (!g_am.kern) {
            return false;
        }
    }

    // K/V: this layer's copy, grown to the pass's last tile; a ubatch packs
    // its own rows, and all of them when the sequence starts over
    kv_cache & kc = g_am.kv[k->data];
    // a copy that ends where this ubatch starts, of this sequence's keys
    int64_t j0 = (p0 > 0 && kc.packed == p0 && kc.print == kv_print(k, p0)) ? p0 : 0;
    if (!kc.bo || kc.cap_tiles < t_need) {
        if (kc.bo) {
            xdna_buffer_free(kc.bo);
        }
        kc.cap_tiles = (t_need + 255) / 256 * 256;
        kc.bo = xdna_buffer_alloc(pool->device, (size_t) 4 * kc.cap_tiles * KV_OBJ * 2);
        if (!kc.bo) {
            kc.cap_tiles = 0;
            return false;
        }
        j0 = 0;
    }
    uint16_t * kvh = (uint16_t *) kc.bo->bo.map();
    pack_kv(k, v, kvh, kc.cap_tiles, j0, n_kv);
    // the tiles past n_kv that a pass streams are zero from the allocation
    // or a longer earlier sequence - finite either way, and masked
    const int64_t t0 = j0 / NK, t1 = (n_kv + NK - 1) / NK;
    for (int st = 0; st < 4; st++) {
        xdna_buffer_sync_to_device_range(kc.bo, (size_t) (t1 - t0) * KV_OBJ * 2,
                                         ((size_t) st * kc.cap_tiles + t0) * KV_OBJ * 2);
    }
    kc.packed = n_kv;
    kc.print  = kv_print(k, n_kv);

    const std::vector<uint32_t> insts = build_seq(npass, p0, kc.cap_tiles);
    if (!xdna_kernel_bind_insts(pool->device, g_am.kern, insts.data(), insts.size())) {
        return false;
    }

    if (g_am.kept_bo) {
        xdna_kernel_pool_release_buffer(g_am.kept_pool, g_am.kept_bo);
        g_am.kept_bo = nullptr;
        g_am.kept    = nullptr;
    }
    const bool keep = std::find(g_am.keep.begin(), g_am.keep.end(), node) != g_am.keep.end();
    const size_t q_col = (size_t) (1 + npass) * 4 * Q_OBJ;      // bf16 a column
    const size_t o_col = (size_t) npass * 4 * O_OBJ;            // f32 a column
    xdna_buffer * bo_q = xdna_kernel_pool_acquire_buffer(pool, COLS * q_col * 2);
    xdna_buffer * bo_o = xdna_kernel_pool_acquire_buffer(pool, COLS * o_col * 4);
    if (!bo_q || !bo_o) {
        return false;
    }

    // Q: per column the header, then per pass and core Q half^T of its 32
    // rows (the group's 4 heads x 8 positions); past the ubatch zero
    uint16_t * qh = (uint16_t *) bo_q->bo.map();
    const float qs = (float) (M_LOG2E / 16.0);
#pragma omp parallel for num_threads(host_threads())
    for (int cp = 0; cp < COLS * (1 + npass); cp++) {
        const int c = cp / (1 + npass), p = cp % (1 + npass) - 1;
        uint16_t * col = qh + (size_t) c * q_col + (size_t) (p + 1) * 4 * Q_OBJ;
        if (p < 0) {
            std::memset(col, 0, (size_t) 4 * Q_OBJ * 2);
            for (int i = 0; i < 4; i++) {
                int32_t * hw = (int32_t *) (col + (size_t) i * Q_OBJ);
                hw[0] = npass;
                hw[1] = (int32_t) p0;
            }
            continue;
        }
        const int g = c / 4;
        for (int i = 0; i < 4; i++) {
            const int pidx = (c - 4 * g) * 2 + i / 2, h = i % 2;
            uint16_t * o = col + (size_t) i * Q_OBJ;
            alignas(64) uint16_t qb[R][DH];
            for (int row = 0; row < R; row++) {
                const int hh = row / 8;
                const int64_t t = (int64_t) p * PASS + pidx * 8 + row % 8;
                if (t < n_tokens) {
                    xdna_scaled_bf16_row((const float *) ((const char *) q->data + t * q->nb[1] + (4 * g + hh) * q->nb[2]) + h * DH,
                                         qs, qb[row], DH);
                } else {
                    std::memset(qb[row], 0, sizeof(qb[row]));
                }
            }
            for (int d = 0; d < DH; d++) {
                for (int row = 0; row < R; row++) {
                    o[((d / 8) * (R / 8) + row / 8) * 64 + (d % 8) * 8 + row % 8] = qb[row][d];
                }
            }
        }
    }
    xdna_buffer_sync_to_device_range(bo_q, COLS * q_col * 2, 0);   // the pool's BO may be larger

    xdna_buffer * args[3] = { bo_q, kc.bo, bo_o };
    xrt::run run = xdna_kernel_run_start(g_am.kern, args, 3);
    const bool ok = xdna_run_wait(run);
    if (ok) {
        // O: dst is [D, n_head, n_tokens]
        xdna_buffer_sync_from_device_range(bo_o, COLS * o_col * 4, 0);
        const float * ob = (const float *) bo_o->bo.map();
#pragma omp parallel for num_threads(host_threads())
        for (int cp = 0; cp < (keep ? 0 : COLS * npass); cp++) {
            const int c = cp / npass, p = cp % npass, g = c / 4;
            for (int i = 0; i < 4; i++) {
                const int pidx = (c - 4 * g) * 2 + i / 2, h = i % 2;
                const float * o = ob + (size_t) c * o_col + ((size_t) p * 4 + i) * O_OBJ;
                for (int row = 0; row < R; row++) {
                    const int64_t t = (int64_t) p * PASS + pidx * 8 + row % 8;
                    if (t >= n_tokens) {
                        continue;
                    }
                    float * dst = (float *) ((char *) node->data + (4 * g + row / 8) * node->nb[1] +
                                             t * node->nb[2]) + h * DH;
                    // the MemTile leaves the core's rows as rows
                    std::memcpy(dst, o + (size_t) row * DH, DH * sizeof(float));
                }
            }
        }
    } else {
        GGML_LOG_ERROR("%s: run failed n_tokens=%lld n_kv=%lld\n", "xdna-attn-mm",
                       (long long) n_tokens, (long long) n_kv);
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
    rows->base     = (const float *) g_am.kept_bo->bo.map();
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
    r.base = (const float *) g_am.kept_bo->bo.map();
    r.o_col = g_am.kept_col;
    r.n_tokens = g_am.kept_tok;
#pragma omp parallel for num_threads(host_threads())
    for (int64_t t = 0; t < r.n_tokens; t++) {
        for (int h = 0; h < H; h++) {
            const float * p0 = xdna_attn_mm_row(&r, t, h);
            float * dst = (float *) ((char *) node->data + h * node->nb[1] + t * node->nb[2]);
            std::memcpy(dst, p0, DH * sizeof(float));
            std::memcpy(dst + DH, p0 + O_OBJ, DH * sizeof(float));
        }
    }
    return true;
}
