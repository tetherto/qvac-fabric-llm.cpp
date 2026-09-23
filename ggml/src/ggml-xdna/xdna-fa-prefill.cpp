#include "xdna-fa-prefill.h"
#include "ggml-impl.h"
#include "xdna-util.h"
#include "xdna-runtime.h"
#include "xdna-types.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

// Runner for the causal flash attention kernel (kernels/fa.py). Host ABI of
// the design, three BOs:
//
//   q  : [col][row][hdr QHDR x i32 | D*MT bf16]  query block TRANSPOSED
//   kv : [kvhead][tile][K JT*D | V JT*D] bf16    keys/values as the cache has
//   o  : [col][row][(D+2)*MT] f32                O transposed, then m and l
//
// One dispatch is 8 heads x ROWS*MT = 128 queries x NJ*JT = 512 keys. The
// accumulator stays in the core's L1 between dispatches, so the key range is
// walked in chunks and only the first zeroes / the last normalises - the
// header in front of each query block says which. A layer therefore takes
// ceil(n_tokens/128) * ceil(n_kv/512) dispatches, and the rounds must not
// interleave: a round's chunks have to run back to back or the next one
// resumes from the wrong accumulator.

namespace fa_pf {

constexpr int D    = 256;   // head dim
constexpr int H    = 8;     // query heads (== columns)
constexpr int KVH  = 2;     // key/value heads
constexpr int MT   = 32;    // queries per core = vector lanes
constexpr int JT   = 8;     // keys per tile
constexpr int ROWS = 4;     // cores per column
constexpr int NJ   = 64;    // key tiles per dispatch
constexpr int QHDR = 16;    // int32 of header ahead of a query block

constexpr int MBLK   = ROWS * MT;          // 128 queries per dispatch
constexpr int KCHUNK = NJ * JT;            // 512 keys per dispatch
constexpr int Q_N    = QHDR + D * MT / 2;  // int32 per query object
constexpr int ACC_N  = (D + 2) * MT;       // f32 per output object
constexpr int KV_N   = 2 * JT * D;         // bf16 per key/value tile

constexpr float SCALE = 0.0625f;           // 1/sqrt(256), baked into the kernel

constexpr const char * STEM = "fa_prefill_bf16_D256_MT32_JT8_NJ64_c8";

static size_t q_bytes()  { return (size_t) H * ROWS * Q_N * sizeof(int32_t); }
static size_t o_bytes()  { return (size_t) H * ROWS * ACC_N * sizeof(float); }
static size_t kv_chunk_bytes() { return (size_t) KVH * NJ * KV_N * sizeof(uint16_t); }

} // namespace fa_pf

struct fa_runner {
    std::mutex     mtx;
    xdna_device *  dev  = nullptr;
    xdna_kernel *  kern = nullptr;
    xdna_buffer *  q    = nullptr;
    xdna_buffer *  o    = nullptr;
    xdna_buffer *  kv   = nullptr;          // all chunks of one layer
    size_t         kv_chunks = 0;           // chunks the kv BO holds
    std::vector<xdna_buffer *> kv_view;     // one window per chunk
};

static fa_runner g_fa;

static inline uint16_t f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    // round to nearest even; the state is bf16 throughout, so a floor here
    // biases every key and value the same way
    const uint32_t r = (u + 0x7fff + ((u >> 16) & 1)) >> 16;
    return (uint16_t) r;
}

// f16 -> bf16 by table. The cache is f16 and the kernel is bf16, so this runs
// over every key and value of every attention layer of every ubatch - tens of
// millions of conversions a prompt. Going through ggml_fp16_to_fp32 and back
// measured as the single largest cost of this path; 128 KB of table removes it.
static const uint16_t * f16_bf16_table(void) {
    static const std::vector<uint16_t> t = []() {
        std::vector<uint16_t> v(1 << 16);
        for (int i = 0; i < (1 << 16); i++) {
            v[i] = f32_to_bf16(ggml_fp16_to_fp32((ggml_fp16_t) (uint16_t) i));
        }
        return v;
    }();
    return t.data();
}

static bool fa_pf_enabled(void) {
    // Opt-in (GGML_XDNA_FA=1). The kernel runs the full-attention prefill
    // layers on the array, but its per-chunk dispatch and the scores round trip
    // cost more than the host attention for this model: ~19% of prefill
    // throughput. Off by default for speed; set it for full-array coverage.
    if (xdna_env_int("GGML_XDNA_FA", 0) == 0) {
        return false;
    }
    for (const auto & dir : xdna_kernel_search_dirs()) {
        std::error_code ec;
        if (std::filesystem::exists(dir / (std::string(fa_pf::STEM) + ".xclbin"), ec)) {
            return true;
        }
    }
    return false;
}

// The kernel derives its mask from the key and query positions, so a mask that
// is anything but plain causal would be silently ignored. Every row has to be
// checked: a packed-sequence mask can differ in one row and nowhere else.
//
// The check needs the mask contents, and a graph node has no data until the
// scheduler allocates it. Declining then is the only safe answer, because a
// node this backend accepts it must also be able to run - xdna_ops_compute
// returning false fails the whole graph rather than falling back to the host.
static bool mask_is_causal(const ggml_tensor * m, int64_t n_kv, int64_t n_tokens) {
    if (!m || !m->data) {
        return false;
    }
    if (m->type != GGML_TYPE_F16 || m->ne[0] < n_kv || m->ne[1] < n_tokens) {
        return false;
    }
    if (m->ne[2] != 1 || m->ne[3] != 1) {
        return false;
    }
    const int64_t npast = n_kv - n_tokens;
    for (int64_t t = 0; t < n_tokens; t++) {
        const ggml_fp16_t * row =
            (const ggml_fp16_t *) ((const char *) m->data + t * m->nb[1]);
        for (int64_t j = 0; j < n_kv; j++) {
            const float v = ggml_fp16_to_fp32(row[j]);
            const bool visible = j <= npast + t;
            if (visible ? v != 0.0f : !(v < -1.0e30f)) {
                return false;
            }
        }
    }
    return true;
}

bool xdna_fa_prefill_supported(const struct ggml_tensor * node) {
    using namespace fa_pf;
    if (!fa_pf_enabled() || !node || node->op != GGML_OP_FLASH_ATTN_EXT) {
        return false;
    }
    const ggml_tensor * q = node->src[0];
    const ggml_tensor * k = node->src[1];
    const ggml_tensor * v = node->src[2];
    const ggml_tensor * m = node->src[3];
    if (!q || !k || !v) {
        return false;
    }
    if (node->src[4]) {
        return false;   // sinks
    }
    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 ||
        v->type != GGML_TYPE_F16 || node->type != GGML_TYPE_F32) {
        return false;
    }
    // q is [D, n_tokens, n_head], k/v are [D, n_kv, n_head_kv].
    if (q->ne[0] != D || q->ne[2] != H || q->ne[3] != 1) {
        return false;
    }
    if (k->ne[0] != D || v->ne[0] != D || k->ne[2] != KVH || v->ne[2] != KVH) {
        return false;
    }
    if (k->ne[1] != v->ne[1] || k->ne[1] <= 0) {
        return false;
    }
    const int64_t n_tokens = q->ne[1];
    const int64_t n_kv     = k->ne[1];
    // A partial round would need the design to know how many queries are real;
    // the tail ubatches are small anyway, so they stay on the host.
    if (n_tokens < MBLK || n_tokens % MBLK || n_kv < n_tokens) {
        return false;
    }
    float params[3] = { 0.0f, 0.0f, 0.0f };
    std::memcpy(params, node->op_params, sizeof(params));
    if (params[0] != SCALE || params[1] != 0.0f || params[2] != 0.0f) {
        return false;   // scale baked into the kernel; no ALiBi, no softcap
    }
    return mask_is_causal(m, n_kv, n_tokens);
}

// Load the kernel once: xclbin, the compiled .insts.bin (the schedule is fixed
// - the chunking is host-side), and the query/output BOs.
static bool fa_load(xdna_device * dev) {
    using namespace fa_pf;
    std::lock_guard<std::mutex> lock(g_fa.mtx);
    if (g_fa.kern) {
        return true;
    }
    std::string xclbin, insts;
    for (const auto & dir : xdna_kernel_search_dirs()) {
        std::error_code ec;
        const std::string fx = (dir / (std::string(STEM) + ".xclbin")).string();
        const std::string fi = (dir / (std::string(STEM) + ".insts.bin")).string();
        if (std::filesystem::exists(fx, ec) && std::filesystem::exists(fi, ec)) {
            xclbin = fx;
            insts  = fi;
            break;
        }
    }
    if (xclbin.empty()) {
        return false;
    }
    std::ifstream f(insts, std::ios::binary);
    std::vector<char> words((std::istreambuf_iterator<char>(f)),
                            std::istreambuf_iterator<char>());
    if (words.empty() || words.size() % sizeof(uint32_t)) {
        return false;
    }
    g_fa.dev  = dev;
    g_fa.kern = xdna_kernel_load_hw(dev, xclbin.c_str());
    if (!g_fa.kern ||
        !xdna_kernel_bind_insts(dev, g_fa.kern, (const uint32_t *) words.data(),
                                words.size() / sizeof(uint32_t))) {
        xdna_kernel_free(g_fa.kern);
        g_fa.kern = nullptr;
        return false;
    }
    g_fa.q = xdna_buffer_alloc(dev, q_bytes());
    g_fa.o = xdna_buffer_alloc(dev, o_bytes());
    if (!g_fa.q || !g_fa.o) {
        xdna_kernel_free(g_fa.kern);
        g_fa.kern = nullptr;
        return false;
    }
    return true;
}

// Grow the key/value BO to hold `chunks` chunks and rebuild the per-chunk
// windows. The design's descriptors read from offset zero, so a chunk is
// handed over as its own view rather than as an offset.
static bool fa_kv_reserve(xdna_device * dev, size_t chunks) {
    using namespace fa_pf;
    if (g_fa.kv && g_fa.kv_chunks >= chunks) {
        return true;
    }
    for (auto * v : g_fa.kv_view) {
        xdna_buffer_free(v);
    }
    g_fa.kv_view.clear();
    xdna_buffer_free(g_fa.kv);
    g_fa.kv = xdna_buffer_alloc(dev, chunks * kv_chunk_bytes());
    if (!g_fa.kv) {
        g_fa.kv_chunks = 0;
        return false;
    }
    g_fa.kv_chunks = chunks;
    for (size_t c = 0; c < chunks; c++) {
        xdna_buffer * v = xdna_buffer_sub(g_fa.kv, c * kv_chunk_bytes(),
                                          kv_chunk_bytes());
        if (!v) {
            return false;
        }
        g_fa.kv_view.push_back(v);
    }
    return true;
}

// Pack keys [j0, n_kv) into [chunk][kvhead][tile][K|V] bf16. Slots past n_kv
// stay zero: they are never visible - every query sits before them - but they
// still multiply a weight of exp(-huge), so they have to be finite.
//
// j0 is where packing starts. The caller passes 0: see xdna_fa_prefill_run for
// why resuming from a previous call is not safe without a cache identity.
static void pack_kv(const ggml_tensor * k, const ggml_tensor * v,
                    uint16_t * dst, int64_t j0, int64_t n_kv) {
    using namespace fa_pf;
    const uint16_t * tbl = f16_bf16_table();
    for (int64_t j = j0; j < n_kv; j++) {
        const int64_t c  = j / KCHUNK;
        const int64_t jl = j % KCHUNK;
        uint16_t * cb = dst + (size_t) c * (KVH * NJ * KV_N);
        for (int g = 0; g < KVH; g++) {
            uint16_t * tb = cb + (size_t) g * NJ * KV_N + (size_t) (jl / JT) * KV_N;
            const uint16_t * kr =
                (const uint16_t *) ((const char *) k->data + j * k->nb[1] + g * k->nb[2]);
            const uint16_t * vr =
                (const uint16_t *) ((const char *) v->data + j * v->nb[1] + g * v->nb[2]);
            uint16_t * kd = tb + (size_t) (jl % JT) * D;
            uint16_t * vd = tb + (size_t) (JT + jl % JT) * D;
            for (int d = 0; d < D; d++) {
                kd[d] = tbl[kr[d]];
                vd[d] = tbl[vr[d]];
            }
        }
    }
}

// One round of queries, transposed into the query objects: core (h, r) gets
// rows [m0 + r*MT, +MT) of head h as [D][MT].
static void pack_q(const ggml_tensor * q, int32_t * dst, int64_t m0) {
    using namespace fa_pf;
    for (int h = 0; h < H; h++) {
        for (int r = 0; r < ROWS; r++) {
            uint16_t * qb = (uint16_t *) (dst + ((size_t) h * ROWS + r) * Q_N + QHDR);
            for (int i = 0; i < MT; i++) {
                const char * src = (const char *) q->data +
                                   (m0 + r * MT + i) * q->nb[1] + h * q->nb[2];
                for (int d = 0; d < D; d++) {
                    qb[(size_t) d * MT + i] = f32_to_bf16(((const float *) src)[d]);
                }
            }
        }
    }
}

bool xdna_fa_prefill_run(struct xdna_device * dev, struct ggml_tensor * node) {
    using namespace fa_pf;
    if (!xdna_fa_prefill_supported(node)) {
        return false;
    }
    if (!fa_load(dev)) {
        return false;
    }

    const ggml_tensor * q = node->src[0];
    const ggml_tensor * k = node->src[1];
    const ggml_tensor * v = node->src[2];

    const int64_t n_tokens = q->ne[1];
    const int64_t n_kv     = k->ne[1];
    const int64_t npast    = n_kv - n_tokens;
    const size_t  chunks   = (size_t) ((n_kv + KCHUNK - 1) / KCHUNK);
    const int64_t rounds   = n_tokens / MBLK;

    // Pack the whole window every time. The previous version kept what it had
    // already packed and appended, keyed on the cache tensors' data pointers
    // and a nondecreasing n_kv. That does not hold: a new prompt reuses the
    // same cache storage, so the pointers match and n_kv can reach the old
    // length again while the keys underneath are different ones, and the
    // retained prefix is then the previous prompt's. A context shift rewrites
    // the cache in place the same way. Restoring the incremental path needs an
    // identity for the cache contents - a generation counter, or a sequence id
    // - which this interface does not carry today.
    if (!fa_kv_reserve(dev, chunks)) {
        return false;
    }
    {
        uint16_t * kvh = (uint16_t *) g_fa.kv->bo.map();
        std::memset(kvh, 0, g_fa.kv_chunks * kv_chunk_bytes());
        pack_kv(k, v, kvh, 0, n_kv);
        xdna_buffer_sync_to_device_range(g_fa.kv, chunks * kv_chunk_bytes(), 0);
    }

    int32_t * qh = (int32_t *) g_fa.q->bo.map();
    for (int64_t rd = 0; rd < rounds; rd++) {
        const int64_t m0 = rd * MBLK;
        {
            pack_q(q, qh, m0);
        }
        for (size_t c = 0; c < chunks; c++) {
            for (int h = 0; h < H; h++) {
                for (int r = 0; r < ROWS; r++) {
                    int32_t * hdr = qh + ((size_t) h * ROWS + r) * Q_N;
                    hdr[0] = c == 0;
                    hdr[1] = c == chunks - 1;
                    hdr[2] = (int32_t) (c * NJ);
                    hdr[3] = (int32_t) (npast + m0);
                }
            }
            {
                xdna_buffer_sync_to_device(g_fa.q);
            }
            xdna_buffer * args[3] = { g_fa.q, g_fa.kv_view[c], g_fa.o };
            xrt::run run = xdna_kernel_run_start(g_fa.kern, args, 3);
            if (!xdna_run_wait(run)) {
                return false;
            }
        }
        xdna_buffer_sync_from_device(g_fa.o);

        // dst is [D, n_head, n_tokens]; a core's object holds O transposed.
        const float * o = (const float *) g_fa.o->bo.map();
        for (int h = 0; h < H; h++) {
            for (int r = 0; r < ROWS; r++) {
                const float * ob = o + ((size_t) h * ROWS + r) * ACC_N;
                for (int i = 0; i < MT; i++) {
                    float * out = (float *) ((char *) node->data +
                                             h * node->nb[1] +
                                             (m0 + r * MT + i) * node->nb[2]);
                    for (int d = 0; d < D; d++) {
                        out[d] = ob[(size_t) d * MT + i];
                    }
                }
            }
        }
    }
    return true;
}
