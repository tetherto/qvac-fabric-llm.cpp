#include "ggml-xdna.h"

#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "xdna-att-layer.h"
#include "xdna-att.h"
#include "xdna-attn-mm.h"
#include "xdna-conv-prefill.h"
#include "xdna-design-tag.h"
#include "xdna-gdn-mm.h"
#include "xdna-head.h"
#include "xdna-norm.h"
#include "xdna-ops.h"
#include "xdna-pgemm.h"
#include "xdna-prof.h"
#include "xdna-rec-gemv.h"
#include "xdna-rec.h"
#include "xdna-runtime.h"
#include "xdna-types.h"
#include "xdna-util.h"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef _WIN32
#    include <windows.h>
#else
#    include <unistd.h>
#endif

static bool ggml_xdna_is_view_op(enum ggml_op op) {
    switch (op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            return false;
    }
}

// ---------------------------------------------------------------------------
// Graph glue (default on). The XDNA backend claims the cheap CPU ops of the
// fused decode layer so the scheduler keeps the recurrent subgraph inside one
// NPU region instead of splitting a chunk at every backend change. Claimed ops
// are not executed on the NPU; they run on the host (CPU backend) from within
// the XDNA graph chunk, after any pending NPU work they depend on is finalized.
// Set GGML_XDNA_GLUE=0 to disable (then only the native NPU ops are claimed).
// ---------------------------------------------------------------------------

static bool xdna_glue_active(void) {
    return xdna_env_int("GGML_XDNA_GLUE", 1) != 0;
}

// True when `op` is claimed by the backend and should be glued (CPU-run) when
// the native NPU dispatch does not take it. The backend claims the cheap CPU
// ops of the decode graph so the scheduler keeps the fused recurrent layer in
// one NPU region instead of splitting a chunk at every backend change; claimed
// ops run on the host (CPU backend) from within the XDNA chunk.
static bool xdna_glue_claims(const struct ggml_tensor * op) {
    if (!xdna_glue_active()) {
        return false;
    }
    if (op->op == GGML_OP_NONE || ggml_xdna_is_view_op(op->op)) {
        return false;
    }
    // The KV append writes llama's f16 cache, which this backend's buffers
    // hold. Left to the CPU backend it split the graph at every attention
    // layer - seven graph_compute calls a token - and an attention layer
    // could never be one region. The CPU's own set_rows runs it here.
    if (op->op == GGML_OP_SET_ROWS) {
        return op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
               (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32);
    }
    // Host-fallback writes op->data; only claim plain contiguous f32 tensors.
    if (op->type != GGML_TYPE_F32 || !ggml_is_contiguous(op)) {
        return false;
    }
    return true;
}

// Fast path for the concat the glue claims. ggml's generic concat copies one
// element at a time, and the GDN conv input is [n_tokens + KW - 1, 6144]: at a
// 512-token chunk that is 3.2M separate 4-byte memcpys per call, measured at
// 14.9 ms, which is close to half of the whole prefill. The tensors the glue
// claims are contiguous F32, and there the run along the concat dimension is a
// contiguous block for both sources, so the same copy is two memcpys per block
// of the remaining dimensions. Anything not contiguous falls through to the
// batch (and so to ggml).
static int xdna_glue_threads(bool prompt);

static bool xdna_concat_fast(struct ggml_tensor * dst) {
    if (dst == nullptr || dst->op != GGML_OP_CONCAT || dst->type != GGML_TYPE_F32) {
        return false;
    }
    struct ggml_tensor * a = dst->src[0];
    struct ggml_tensor * b = dst->src[1];
    if (a == nullptr || b == nullptr || a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) {
        return false;
    }
    // The GDN conv input concatenates the cached tokens with the new chunk's
    // projection along the token dimension: the cache is token-major, the
    // projection is channel-major, so one side of the copy is a transpose.
    // ggml's per-element version reads the projection with a 24 KiB stride
    // between tokens, which at a 512-token chunk is 3.2M separate 4-byte
    // accesses, measured at 14.9 ms per call - close to half of the whole
    // prefill. Tiling over channels and tokens turns the reads into runs of 16
    // real floats and keeps the strided writes inside a few cache lines.
    if (ggml_get_op_params_i32(dst, 0) == 0 && ggml_is_contiguous(dst) && b->nb[1] == sizeof(float) &&
        a->nb[0] == sizeof(float)) {
        bool dims_ok = true;
        for (int d = 1; d < 4; d++) {
            if (a->ne[d] != dst->ne[d] || b->ne[d] != dst->ne[d]) {
                dims_ok = false;
            }
        }
        if (dims_ok && a->ne[0] + b->ne[0] == dst->ne[0]) {
            const int64_t na  = a->ne[0];
            const int64_t nb  = b->ne[0];
            const int64_t ne0 = dst->ne[0];
            const int64_t n_c = dst->ne[1];
            constexpr int TC  = 16;
            constexpr int TT  = 16;
            // One slice per sequence (dims 2 and 3) with its own strides: b
            // is a transposed view, so its slices are not n_c rows apart.
            for (int64_t i3 = 0; i3 < dst->ne[3]; i3++) {
                for (int64_t i2 = 0; i2 < dst->ne[2]; i2++) {
                    const char * ad = (const char *) a->data + i2 * a->nb[2] + i3 * a->nb[3];
                    const char * bd = (const char *) b->data + i2 * b->nb[2] + i3 * b->nb[3];
                    float *      dd = (float *) ((char *) dst->data + i2 * dst->nb[2] + i3 * dst->nb[3]);
#pragma omp parallel for num_threads(xdna_glue_threads(true))
                    for (int64_t c0 = 0; c0 < n_c; c0 += TC) {
                        const int nc = (int) std::min<int64_t>(TC, n_c - c0);
                        for (int c = 0; c < nc; c++) {
                            std::memcpy(dd + (c0 + c) * ne0, ad + (size_t) (c0 + c) * a->nb[1],
                                        (size_t) na * sizeof(float));
                        }
                        for (int64_t t0 = 0; t0 < nb; t0 += TT) {
                            const int nt = (int) std::min<int64_t>(TT, nb - t0);
                            for (int t = 0; t < nt; t++) {
                                const float * srow =
                                    (const float *) (bd + (size_t) (t0 + t) * b->nb[0] + (size_t) c0 * b->nb[1]);
                                const int64_t drow = na + t0 + t;
                                for (int c = 0; c < nc; c++) {
                                    dd[(c0 + c) * ne0 + drow] = srow[c];
                                }
                            }
                        }
                    }
                }
            }
            return true;
        }
    }

    if (!ggml_is_contiguous(a) || !ggml_is_contiguous(b) || !ggml_is_contiguous(dst)) {
        return false;
    }
    const int dim = ggml_get_op_params_i32(dst, 0);
    if (dim < 0 || dim > 3) {
        return false;
    }
    for (int d = 0; d < 4; d++) {
        if (d != dim && (a->ne[d] != dst->ne[d] || b->ne[d] != dst->ne[d])) {
            return false;
        }
    }
    int64_t inner = 1;
    int64_t outer = 1;
    for (int d = 0; d < dim; d++) {
        inner *= dst->ne[d];
    }
    for (int d = dim + 1; d < 4; d++) {
        outer *= dst->ne[d];
    }
    const size_t ea = (size_t) inner * (size_t) a->ne[dim] * sizeof(float);
    const size_t eb = (size_t) inner * (size_t) b->ne[dim] * sizeof(float);
    const char * pa = (const char *) a->data;
    const char * pb = (const char *) b->data;
    char *       pd = (char *) dst->data;
    for (int64_t o = 0; o < outer; o++) {
        if (ea) {
            std::memcpy(pd, pa, ea);
        }
        if (eb) {
            std::memcpy(pd + ea, pb, eb);
        }
        pa += ea;
        pb += eb;
        pd += ea + eb;
    }
    return true;
}

// Threads for the host-fallback work. The two workloads want opposite sizes.
// Decode glue is a handful of tiny nodes between NPU dispatches, about 160
// batches a token; a full pool there only adds barrier cost (16 threads
// measured 52.4 ms/token against 46.0 at four). A prefill chunk's glue is the
// opposite: elementwise passes over every token row, the whole GDN recurrence
// and the attention softmax, all of it real parallel work - and it runs on
// this backend, so the small pool is what limits prefill. The size comes from
// the chunk's token count (xdna_glue_set_n_tokens), because no single node
// shape carries it: the conv input is [n_tokens, 6144], the elementwise ops
// are [n_embd, n_tokens], and the GDN state is [S, S, H] on both passes.
// SSM_CONV on the host (GGML_XDNA_CONV=0), in place of ggml's. ggml walks a
// token at a time across every channel, and the conv input is channel-major,
// n_t + 3 floats a channel: each of the 6144 channels of each token is a
// separate cache line, 16.5 ms a layer at 4096 tokens (with the silu and the
// norms around it) and 1.2 s of a 16k prefill. Tiled over channels and
// tokens - the reads runs of a channel's tokens, the writes runs of a
// token's channels - and on the glue's threads. The sum is ggml's, in f32,
// in the same order.
static bool xdna_ssm_conv_fast(struct ggml_tensor * dst) {
    if (dst == nullptr || dst->op != GGML_OP_SSM_CONV || dst->type != GGML_TYPE_F32) {
        return false;
    }
    const struct ggml_tensor * sx = dst->src[0];
    const struct ggml_tensor * w  = dst->src[1];
    if (sx == nullptr || w == nullptr || sx->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 ||
        sx->nb[0] != sizeof(float) || w->nb[0] != sizeof(float) || sx->nb[1] != sx->ne[0] * sizeof(float) ||
        !ggml_is_contiguous(dst)) {
        return false;
    }
    const int64_t nc  = w->ne[0];   // taps
    const int64_t nr  = sx->ne[1];  // channels
    const int64_t n_t = dst->ne[1];
    const int64_t n_s = dst->ne[2];
    if (dst->ne[0] != nr || sx->ne[0] != nc - 1 + n_t) {
        return false;
    }
    constexpr int TC = 64, TT = 64;
    const int64_t nbc = (nr + TC - 1) / TC, nbt = (n_t + TT - 1) / TT;
#pragma omp parallel for collapse(2) num_threads(xdna_glue_threads(true))
    for (int64_t i3 = 0; i3 < n_s; i3++) {
        for (int64_t bt = 0; bt < nbc * nbt; bt++) {
            const int64_t c0 = (bt / nbt) * TC, t0 = (bt % nbt) * TT;
            const int64_t cn = std::min<int64_t>(TC, nr - c0), tn = std::min<int64_t>(TT, n_t - t0);
            float         tile[TT][TC];
            for (int64_t c = 0; c < cn; c++) {
                const float * srow =
                    (const float *) ((const char *) sx->data + (c0 + c) * sx->nb[1] + i3 * sx->nb[2]) + t0;
                const float * wr = (const float *) ((const char *) w->data + (c0 + c) * w->nb[1]);
                for (int64_t t = 0; t < tn; t++) {
                    float sumf = 0.0f;
                    for (int64_t i0 = 0; i0 < nc; i0++) {
                        sumf += srow[t + i0] * wr[i0];
                    }
                    tile[t][c] = sumf;
                }
            }
            for (int64_t t = 0; t < tn; t++) {
                float * x = (float *) ((char *) dst->data + (t0 + t) * dst->nb[1] + i3 * dst->nb[2]) + c0;
                std::memcpy(x, tile[t], (size_t) cn * sizeof(float));
            }
        }
    }
    return true;
}

// Sigmoid on the host, in place of ggml's scalar expf loop (the attention's
// output gate, [2048, n_tokens]: 11.6 ms a call at 4096 tokens, 70 ms of a
// ubatch). exp(-x) = 2^n 2^f with n the nearest integer to -x log2(e) and
// 2^f, |f| <= 1/2, a degree-6 polynomial (relative error ~2e-7), plain
// arithmetic the compiler vectorizes.
static inline float xdna_fast_sigmoid(float x) {
    float t       = -x * 1.44269504f;
    t             = t < -126.0f ? -126.0f : (t > 126.0f ? 126.0f : t);
    // nearest integer by the 1.5 * 2^23 add, which vectorizes where roundf may not
    const float n = (t + 12582912.0f) - 12582912.0f;
    const float p = xdna_exp2_poly(t - n);
    int32_t     bits;
    std::memcpy(&bits, &p, 4);
    bits += (int32_t) ((uint32_t) n << 23);
    float e;
    std::memcpy(&e, &bits, 4);
    return 1.0f / (1.0f + e);
}

static bool xdna_sigmoid_fast(struct ggml_tensor * dst) {
    if (dst == nullptr || dst->op != GGML_OP_UNARY || ggml_get_unary_op(dst) != GGML_UNARY_OP_SIGMOID ||
        dst->type != GGML_TYPE_F32) {
        return false;
    }
    const struct ggml_tensor * x = dst->src[0];
    if (x == nullptr || x->type != GGML_TYPE_F32 || !ggml_are_same_shape(x, dst) || x->nb[0] != sizeof(float) ||
        dst->nb[0] != sizeof(float)) {
        return false;
    }
    const int64_t n0 = dst->ne[0], n1 = dst->ne[1], n2 = dst->ne[2], n3 = dst->ne[3];
#pragma omp parallel for collapse(3) num_threads(xdna_glue_threads(true))
    for (int64_t i3 = 0; i3 < n3; i3++) {
        for (int64_t i2 = 0; i2 < n2; i2++) {
            for (int64_t i1 = 0; i1 < n1; i1++) {
                const float * xs =
                    (const float *) ((const char *) x->data + i1 * x->nb[1] + i2 * x->nb[2] + i3 * x->nb[3]);
                float * ys = (float *) ((char *) dst->data + i1 * dst->nb[1] + i2 * dst->nb[2] + i3 * dst->nb[3]);
#pragma omp simd
                for (int64_t i0 = 0; i0 < n0; i0++) {
                    ys[i0] = xdna_fast_sigmoid(xs[i0]);
                }
            }
        }
    }
    return true;
}

// The norms of this graph whose one reader is the MUL by their weight that
// follows them (planned in graph_compute): the pair runs as one pass.
static std::unordered_set<const struct ggml_tensor *> g_norm_mul;

// RMS_NORM `r` and the MUL `m` of it by a weight row, as one pass writing
// m's output (xdna-norm.h: ggml's rounding); r's own output is not written.
static bool xdna_norm_mul_fast(const struct ggml_tensor * r, struct ggml_tensor * m) {
    if (r == nullptr || m == nullptr || !g_norm_mul.count(r) || m->op != GGML_OP_MUL || m->src[0] != r) {
        return false;
    }
    const struct ggml_tensor * x = r->src[0];
    const struct ggml_tensor * w = m->src[1];
    if (x == nullptr || w == nullptr || x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 ||
        m->type != GGML_TYPE_F32 || x->nb[0] != sizeof(float) || m->nb[0] != sizeof(float) ||
        !ggml_are_same_shape(x, m) || !ggml_is_contiguous(w) || w->ne[0] != m->ne[0] || ggml_nrows(w) != 1) {
        return false;
    }
    float eps;
    memcpy(&eps, r->op_params, sizeof(float));
    const float * wd = (const float *) w->data;
    const int64_t n0 = m->ne[0], n1 = m->ne[1], n2 = m->ne[2], n3 = m->ne[3];
#pragma omp parallel for collapse(3) num_threads(xdna_glue_threads(true))
    for (int64_t i3 = 0; i3 < n3; i3++) {
        for (int64_t i2 = 0; i2 < n2; i2++) {
            for (int64_t i1 = 0; i1 < n1; i1++) {
                const float * xs =
                    (const float *) ((const char *) x->data + i1 * x->nb[1] + i2 * x->nb[2] + i3 * x->nb[3]);
                float * ys = (float *) ((char *) m->data + i1 * m->nb[1] + i2 * m->nb[2] + i3 * m->nb[3]);
                xdna_norm_mul_row(xs, wd, xdna_rms_scale(xs, n0, eps), ys, n0);
            }
        }
    }
    return true;
}

// atoi, but clamped to int instead of overflowing into UB; trailing garbage is
// still ignored.
static int xdna_atoi(const char * s) {
    return (int) std::clamp(strtol(s, nullptr, 10), (long) INT_MIN, (long) INT_MAX);
}

static int xdna_glue_threads(bool prompt) {
    static const int n_small = []() {
        const char * v = getenv("GGML_XDNA_GLUE_THREADS");
        return v ? std::max(1, xdna_atoi(v)) : 4;
    }();
    static const int n_big = []() {
        const char * v = getenv("GGML_XDNA_GLUE_THREADS_BIG");
        return v ? std::max(1, xdna_atoi(v)) : 16;
    }();
    return prompt ? n_big : n_small;
}

// Token count of the chunk being computed, read off the projections: every
// MUL_MAT activation is M rows of K, and every decoder layer has at least one.
// This is the ubatch size, 1 on a decode step.
static int g_glue_n_tokens = 1;

static void xdna_glue_set_n_tokens(const struct ggml_cgraph * cgraph) {
    int n = 1;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const struct ggml_tensor * node = cgraph->nodes[i];
        if (node->op == GGML_OP_MUL_MAT && node->src[1] != nullptr) {
            n = std::max(n, (int) node->src[1]->ne[1]);
        }
    }
    g_glue_n_tokens = n;
}

// Lazily created CPU backend + a scratch cgraph for the host-fallback compute.
static ggml_backend_t xdna_glue_cpu(void) {
    static ggml_backend_t cpu = []() {
        ggml_backend_t b = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (b == nullptr) {
            GGML_LOG_WARN("%s: glue: no CPU backend for host fallback\n", "ggml-xdna");
            return b;
        }
        ggml_backend_cpu_set_n_threads(b, xdna_glue_threads(false));
        return b;
    }();
    return cpu;
}

static ggml_cgraph * xdna_glue_cgraph(void) {
    static ggml_cgraph * cg = []() {
        struct ggml_init_params params = {
            /* .mem_size   = */ (size_t) 512 * 1024,
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ false,
        };
        ggml_context * ctx = ggml_init(params);
        if (ctx == nullptr) {
            return (ggml_cgraph *) nullptr;
        }
        return ggml_new_graph_custom(ctx, 64, false);
    }();
    return cg;
}

// Run `node` on the host from within the XDNA chunk. Flushes (waits) all
// pending NPU submissions first, so the op sees the results of the NPU ops it
// depends on. Returns false on failure or when the op is not glue-claimed.
static bool xdna_glue_host_run(xdna_ops * ops, const struct ggml_tensor * node) {
    if (!xdna_glue_claims(node)) {
        return false;
    }
    // Native NPU ops must go to the device, not the host fallback.
    if (xdna_ops_supported(ops, node)) {
        return false;
    }
    ggml_backend_t cpu = xdna_glue_cpu();
    ggml_cgraph *  cg  = xdna_glue_cgraph();
    if (cpu == nullptr || cg == nullptr) {
        GGML_LOG_ERROR("%s: glue: no CPU backend/graph for %s\n", "ggml-xdna", ggml_op_name(node->op));
        return false;
    }
    if (!xdna_ops_finalize(ops)) {
        return false;
    }
    ggml_backend_cpu_set_n_threads(cpu, xdna_glue_threads(g_glue_n_tokens >= 32));
    cg->n_nodes  = 1;
    cg->n_leafs  = 0;
    cg->nodes[0] = const_cast<struct ggml_tensor *>(node);
    bool ok;
    {
        const std::vector<std::string> op_names = { ggml_op_name(node->op) };
        xdna_prof::glue_timer          xdna_prof_glue(xdna_prof::glue_signature(op_names), op_names);
        ok = ggml_backend_graph_compute(cpu, cg) == GGML_STATUS_SUCCESS;
    }
    if (!ok) {
        GGML_LOG_ERROR("%s: glue: host compute failed for %s\n", "ggml-xdna", ggml_op_name(node->op));
        return false;
    }
    return true;
}

// Flush a batch of consecutive glue (host) ops as one CPU cgraph compute after
// any pending NPU work is finalized. The scratch cgraph holds up to 64 nodes;
// longer runs are split.
static bool xdna_glue_flush(xdna_ops * ops, std::vector<struct ggml_tensor *> & batch) {
    if (batch.empty()) {
        return true;
    }
    if (!xdna_ops_finalize(ops)) {
        batch.clear();
        return false;
    }
    ggml_backend_t cpu = xdna_glue_cpu();
    ggml_cgraph *  cg  = xdna_glue_cgraph();
    if (cpu == nullptr || cg == nullptr) {
        batch.clear();
        return false;
    }
    ggml_backend_cpu_set_n_threads(cpu, xdna_glue_threads(g_glue_n_tokens >= 32));
    size_t off = 0;
    while (off < batch.size()) {
        // a prompt's ops this backend runs faster itself go on their own, in
        // order: the CPU graph takes the run of others before them
        if (g_glue_n_tokens >= 32 && xdna_sigmoid_fast(batch[off])) {
            off++;
            continue;
        }
        if (g_glue_n_tokens >= 32 && off + 1 < batch.size() && xdna_norm_mul_fast(batch[off], batch[off + 1])) {
            off += 2;
            continue;
        }
        size_t n = std::min<size_t>(64, batch.size() - off);
        if (g_glue_n_tokens >= 32) {
            for (size_t j = 1; j < n; j++) {
                const ggml_tensor * t = batch[off + j];
                if ((t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == GGML_UNARY_OP_SIGMOID) || g_norm_mul.count(t)) {
                    n = j;
                    break;
                }
            }
        }
        cg->n_nodes = (int) n;
        cg->n_leafs = 0;
        std::vector<std::string> op_names;
        op_names.reserve(n);
        for (size_t j = 0; j < n; j++) {
            cg->nodes[j] = batch[off + j];
            op_names.push_back(ggml_op_name(batch[off + j]->op));
        }
        xdna_prof::glue_names(batch.data() + off, n, g_glue_n_tokens);
        bool ok;
        {
            xdna_prof::glue_timer xdna_prof_glue(xdna_prof::glue_signature(op_names), op_names);
            ok = ggml_backend_graph_compute(cpu, cg) == GGML_STATUS_SUCCESS;
        }
        if (!ok) {
            batch.clear();
            return false;
        }
        off += n;
    }
    batch.clear();
    return true;
}

// ---------------------------------------------------------------------------
// Fused decode recurrent layer.
//
// A single-token decode replaces every recurrent (gated-delta-net) layer by
// one xdna_rec_* persistent run on fused_layer.xclbin: the layer's projections
// (qkv/z/gate/beta) keep their graph nodes but, in isolation mode, run on the
// host glue; their outputs feed the device kernels; the recurrent conv and ssm
// state is seeded once from the llama recurrent cache and then carried on the
// device. Prefill (M > 1) and MHA layers are untouched.
// ---------------------------------------------------------------------------

struct xdna_rec_session;

namespace {

// Process-wide context shared by all backend instances.
struct ggml_backend_xdna_context {
    xdna_device *      device = nullptr;
    xdna_kernel_pool * pool   = nullptr;  // lazily scanned kernel pool

    xdna_ops ops;                         // operator-specific dispatch (GEMM variants, helpers)

    // Fused decode layer sessions: one per recurrent layer of the model,
    // created on the first single-token decode and kept for the process
    // lifetime. llama-server runs several sequences through one context, and
    // a session holds one of them at a time: the row its recurrent-memory
    // cell occupies in the cache tensor. Another row's decode writes the
    // state back into the cell it came from and reseeds from the new one
    // (xdna_rec_run). A session a sequence rebuilt every layer's weights,
    // buffers and stream - ~5 s the first token of each new server slot paid,
    // and a copy of the layers' weights per slot. Keyed (layer, 0).
    std::map<std::pair<int, int64_t>, struct xdna_rec_session *> rec;

    // Backends (llama contexts) alive on this context. The sessions and the
    // packed weights above belong to a model; when the last backend is freed
    // they are released with it (xdna_release_model_state), so a model loaded
    // afterwards - possibly at the same addresses - starts from nothing.
    std::mutex life_mutex;
    int        n_backends = 0;
    // One graph compute at a time. Every backend instance is this one context,
    // and what a compute uses - the runtime's batch (g_batch), the fused-layer
    // sessions and their buffers, the prefill runners - is per process: two
    // llama contexts computing at once raced on all of it (one found the
    // other's in-projection buffer gone). The array runs one command stream at
    // a time anyway, so this costs nothing a single context had.
    std::mutex compute_mutex;
    // Attention-layer tails (attn_output + the FFN, one dispatch), per layer.
    // Stateless, so one per layer serves every sequence.
    std::map<int, struct xdna_rec_tail *>                        tails;
    // Decode attention on the pool, and how far each layer's K and V rows
    // have been flushed to the device (reset by any multi-token graph).
    struct xdna_att *                                            att = nullptr;
    std::map<const void *, int>                                  att_flushed;
    // Whole attention layers as one dispatch, per layer; null once one
    // could not be built, so it is not retried.
    std::map<int, struct xdna_att_layer *>                       att_layers;
    // Kernels loaded up front by xdna_kernels_warm and held for the process:
    // being loaded is the point, so they stay resident here and are released
    // with the context.
    std::vector<struct xdna_kernel *>                            warm_kerns;

    ~ggml_backend_xdna_context() {
        for (auto & kv : att_layers) {
            xdna_att_layer_free(kv.second);
        }
        for (struct xdna_kernel * kern : warm_kerns) {
            xdna_kernel_free(kern);
        }
    }
    // The residual rows the fused layers hand each other (XDNA_RES_*).
    struct xdna_buffer * res       = nullptr;
    // The last fused layer's outputs while they are only in the rows: l_out
    // = F + A and attn_residual = A, written to the tensors only when a node
    // outside the fused layers reads them (xdna_res_touch).
    const ggml_tensor *  res_lout  = nullptr;
    const ggml_tensor *  res_resid = nullptr;
    bool                 res_dirty = false;  // not yet written to the tensors

    // Fused layers started and not yet waited for: they only hand each other
    // the rows (and the attention layers the cache), so a token's layers go
    // into the context's queue back to back and the host waits once, where
    // it reads something they wrote. With what each needs doing after.
    struct pending_run {
        xrt::run *         run    = nullptr;
        bool               att    = false;
        float *            logits = nullptr;  // the head's: where its logits go
        xdna_att_layer_in  in;                // an attention layer's, for its finish
        std::vector<float> cosv, sinv;
    };

    std::vector<pending_run>   pending;
    bool                       pending_failed = false;
    // Give-up reasons already reported: the fused and tail runs are retried
    // every token, so each distinct cause gets one line.
    std::set<std::string>      gave_up;
    std::mutex                 gave_up_mutex;
    // The vocabulary projection on the NPU (xdna-head.h), built at its first
    // decode; `head_tried` so a failure is not retried every token.
    struct xdna_head *         head       = nullptr;
    bool                       head_tried = false;
    // The node the dispatch loop is on, so the exception boundary in
    // ggml_backend_xdna_graph_compute can name it. Null outside a graph.
    const struct ggml_tensor * cur_node   = nullptr;
    int                        cur_index  = -1;
};

}  // namespace

// The residual rows: every queued design names them, so they are the
// arena's (xdna_arena_scope) like the designs' own buffers.
static xdna_buffer * xdna_res_alloc(xdna_device * dev) {
    xdna_arena_scope arena;
    return xdna_buffer_alloc(dev, XDNA_RES_BYTES);
}

// GGML_XDNA_BATCH=k sends queued runs to the device k to a runlist rather
// than one command each (xdna_batch_begin). The SoC's power follows the
// command rate, not the bytes: synthetic loads on the array measured 16 W
// package at 89 commands/s streaming 48 GB/s and 20 W at 1000 commands/s
// moving 1 GB/s, and the decode's 25 a token (1275/s) holds 23.5 W. At 1k,
// whole-token lists (k=32) are 18.3 W at 46.9 tok/s against 23.5 W at 50.8 -
// 0.390 J a token against 0.463, 16% less - but a command inside a list
// costs ~65 us more than one queued alone, so it is off by default until
// the token is one command of its own. 0 or 1: a command each.
static int xdna_batch_k(void) {
    // GGML_XDNA_TOKEN (1, the default): the token's queued runs as one joined
    // command, the first on its own (xdna_run_submit's token mode); t > 1: at
    // most t runs a command; 0: a command each. At 1k it is 18.0 W at ~48
    // tok/s against 24.0 W at 50.9 - 0.375 J a token against 0.471. Every
    // size from 3 runs a command up loses the same ~6%: with more than one
    // layer in a command a layer boundary costs ~65 us that separate commands
    // do not pay (one layer a command through the same join is full speed).
    if (const int t = xdna_env_int("GGML_XDNA_TOKEN", 1)) {
        return t == 1 ? -1 : -t;
    }
    static const int v = xdna_env_int("GGML_XDNA_BATCH", 0);
    return v > 1 ? v : 0;
}

static void xdna_batch_open(void) {
    if (!xdna_batch_active() && xdna_batch_k() != 0) {
        xdna_batch_begin(xdna_batch_k());
    }
}

// Wait for every started fused layer, in order.
static bool xdna_pending_wait(ggml_backend_xdna_context * ctx) {
    const bool batched = xdna_batch_active();
    if (batched && !xdna_batch_wait()) {
        ctx->pending_failed = true;
    }
    for (auto & p : ctx->pending) {
        if (!batched && !xdna_run_wait(*p.run)) {
            ctx->pending_failed = true;
        } else if (p.att) {
            if (!xdna_att_layer_finish(p.in)) {
                ctx->pending_failed = true;
            }
        } else if (p.logits) {
            if (!xdna_head_read(ctx->head, p.logits)) {
                ctx->pending_failed = true;
            }
        }
    }
    ctx->pending.clear();
    return !ctx->pending_failed;
}

// The decode's vocabulary projection on the NPU (GGML_XDNA_HEAD=0: host).
static bool xdna_head_on(void) {
    static const bool v = [] {
        return (xdna_env_int("GGML_XDNA_HEAD", 1) != 0);
    }();
    return v;
}

// The head's MUL_MAT: the tied embedding (or the output weight) against one
// normed row.
static bool xdna_head_node(const ggml_tensor * n) {
    if (n->op != GGML_OP_MUL_MAT || !n->src[0] || !n->src[1] || n->src[0]->op != GGML_OP_NONE) {
        return false;
    }
    const char *        nm = ggml_get_name(n->src[0]);
    const ggml_tensor * x  = n->src[1];
    return (strcmp(nm, "token_embd.weight") == 0 || strcmp(nm, "output.weight") == 0) && x->type == GGML_TYPE_F32 &&
           ggml_nelements(x) == x->ne[0] && ggml_is_contiguous(x) && x->ne[0] == n->src[0]->ne[0] &&
           n->type == GGML_TYPE_F32 && ggml_is_contiguous(n) && ggml_nelements(n) == n->src[0]->ne[1];
}

namespace {

// The head's input chain: [GET_ROWS of one row] <- MUL by the output norm's
// gamma <- RMS_NORM of the last layer's output. Null members when it is not.
struct xdna_head_chain {
    ggml_tensor *       rows = nullptr, *mul = nullptr, *rms = nullptr;
    const ggml_tensor * src   = nullptr;  // what the norm reads
    const ggml_tensor * gamma = nullptr;
    float               eps   = 0.0f;
};

}  // namespace

static bool xdna_head_chain_of(const ggml_tensor * head, xdna_head_chain & c) {
    c               = xdna_head_chain{};
    ggml_tensor * t = head->src[1];
    if (t && t->op == GGML_OP_GET_ROWS) {
        if (ggml_nelements(t->src[1]) != 1 || t->src[0]->ne[1] != 1) {
            return false;  // more than the one row
        }
        c.rows = t;
        t      = t->src[0];
    }
    if (!t || t->op != GGML_OP_MUL || !t->src[0] || t->src[0]->op != GGML_OP_RMS_NORM || !t->src[1] ||
        t->src[1]->op != GGML_OP_NONE || t->src[1]->type != GGML_TYPE_F32 || ggml_nelements(t->src[1]) != XDNA_RES_D) {
        return false;
    }
    c.mul   = t;
    c.rms   = t->src[0];
    c.src   = c.rms->src[0];
    c.gamma = t->src[1];
    memcpy(&c.eps, c.rms->op_params, sizeof(float));
    return true;
}

// Queue fused layers instead of waiting for each (GGML_XDNA_QUEUE=0: wait).
static bool xdna_queue_on(void) {
    static const bool v = [] {
        return (xdna_env_int("GGML_XDNA_QUEUE", 1) != 0);
    }();
    return v;
}

// Write the rows' pending outputs to their tensors. The rows keep holding
// them: the next fused layer still takes its input from there.
static bool xdna_res_materialize(ggml_backend_xdna_context * ctx) {
    if (!ctx->res_lout || !ctx->res_dirty) {
        return true;
    }
    xdna_pending_wait(ctx);
    std::vector<float> r2((size_t) 2 * XDNA_RES_D);
    if (!xdna_buffer_read_settled(ctx->res, r2.data(), r2.size() * sizeof(float), XDNA_RES_A)) {
        return false;
    }
    float * lo = (float *) ctx->res_lout->data;
    for (int i = 0; i < XDNA_RES_D; i++) {
        lo[i] = r2[(size_t) i] + r2[(size_t) XDNA_RES_D + (size_t) i];
    }
    if (ctx->res_resid) {
        std::memcpy(ctx->res_resid->data, r2.data(), (size_t) XDNA_RES_D * sizeof(float));
    }
    ctx->res_dirty = false;
    return true;
}

// The rows are about to hold something else: write what they hold first.
static bool xdna_res_release(ggml_backend_xdna_context * ctx) {
    xdna_pending_wait(ctx);
    if (!xdna_res_materialize(ctx)) {
        return false;
    }
    ctx->res_lout = ctx->res_resid = nullptr;
    return true;
}

// Before `node` runs: if it reads the rows' pending outputs, write them.
static bool xdna_res_touch(ggml_backend_xdna_context * ctx, const ggml_tensor * node) {
    if (!ctx->res_lout || !ctx->res_dirty) {
        return true;
    }
    for (int k = 0; k < GGML_MAX_SRC && node->src[k]; k++) {
        for (const ggml_tensor * t = node->src[k]; t; t = t->view_src) {
            if (t == ctx->res_lout || t == ctx->res_resid) {
                return xdna_res_materialize(ctx);
            }
        }
    }
    return true;
}

// One fused-layer session (per recurrent block). Owns the three runners and
// the host scratch buffers it reuses every token.
struct xdna_rec_session {
    int                  il   = -1;
    int64_t              row  = -1;       // the sequence whose state it holds
    xdna_rec_core *      core = nullptr;  // fused_layer.xclbin (conv+norm+gdn+gated)
    // The layer's projections, on the decode GEMV of the same design
    // (xdna-rec-gemv.h), so the layer never changes hardware context.
    xdna_rec_gemv *      gv   = nullptr;
    xdna_rec_layer_host  host;
    bool                 seeded = false;
    std::vector<uint8_t> feed0;             // one-time feed host buffer (hist+qkv+conv W)
    std::vector<uint8_t> x;                 // per-token host buffer (eg/beta/scale tails)
    std::vector<float>   hattn, hout, hff;
    std::vector<int8_t>  aq;                // 2048 gated int8 codes (fused core)
    float                d_a      = 1.0f;   // int8 scale of the fused core gated
    float                eps_post = 1e-6f;  // the post norm's epsilon, off the graph
    // Input snapshots: qkv/conv/ssm seeds and the residual h are transient
    // graph tensors whose buffers llama recycles before the fused run fires,
    // so each is copied here right after its producer node executes.
    std::vector<float>   cap_qkv, cap_conv, cap_ss, cap_hres, cap_act;
};

namespace {

// Per-graph plan of one fused layer: the node indices/tensors the fused run
// reads or writes.
struct xdna_rec_plan {
    int           il         = -1;
    int           i_conv     = -1;       // SSM_CONV node (layer body start)
    int           i_lout     = -1;       // l_out ADD (fused h_out dst)
    int           i_fire     = -1;       // first consumed node index
    ggml_tensor * n_qkv      = nullptr;  // MUL_MAT blk.N.attn_qkv (qkv data)
    ggml_tensor * n_z        = nullptr;  // MUL_MAT blk.N.attn_gate (z data)
    ggml_tensor * n_gate     = nullptr;  // MUL gate-N dst
    ggml_tensor * n_beta     = nullptr;  // UNARY beta_sigmoid-N dst
    ggml_tensor * n_resid    = nullptr;  // ADD attn_residual-N (h_attn dst)
    ggml_tensor * n_lout     = nullptr;  // ADD l_out-N (h_out dst)
    ggml_tensor * n_convst   = nullptr;  // GET_ROWS conv_states-N (conv seed)
    ggml_tensor * n_sstate   = nullptr;  // GET_ROWS cache_s read (ssm seed)
    ggml_tensor * n_hres     = nullptr;  // residual input (attn_residual src[1])
    // weight leaves
    ggml_tensor * w_conv     = nullptr;
    ggml_tensor * w_gamma    = nullptr;
    ggml_tensor * w_post     = nullptr;
    ggml_tensor * w_so       = nullptr;
    ggml_tensor * w_gate     = nullptr;
    ggml_tensor * w_up       = nullptr;
    ggml_tensor * w_down     = nullptr;
    // producer node indices of the input snapshots above (-1 = not in chunk)
    int           i_cap_qkv  = -1;  // MUL_MAT qkv node
    int           i_cap_conv = -1;  // GET_ROWS conv-state seed node
    int           i_cap_ss   = -1;  // GET_ROWS cache_s seed node
    int           i_cap_hres = -1;  // residual producer node
    // The in-projection's input (attn_norm-N): with the projection in the
    // fused stream (xdna_rec_inproj_on), this is what the fire reads in place
    // of qkv and z, and the two MUL_MATs never run.
    ggml_tensor * n_act      = nullptr;
    int           i_cap_act  = -1;
};

}  // namespace

// The recurrent layer's in-projection (attn_qkv + attn_gate) runs at the head
// of the fused dispatch and drains qkv and z straight into the core's inputs.
// GGML_XDNA_INPROJ=0 keeps it a GEMV dispatch of its own, whose outputs the
// host then copies into the core's buffers.
static bool xdna_rec_inproj_on(void) {
    static const bool v = [] {
        return (xdna_env_int("GGML_XDNA_INPROJ", 1) != 0);
    }();
    return v;
}

// Fused recurrent-layer path is the DEFAULT NPU decode mode. GGML_XDNA_FUSED_LAYER=0
// restores the per-op decode path. Requires the fused xclbin to exist
// (xdna_rec_active); otherwise the backend falls back to per-op.
//
// The artifact (kernels/fused_layer.py) holds the recurrent core and the decode
// GEMV in one array configuration, so a layer's dispatches share one hardware
// context instead of reconfiguring the array between them. It is looked up
// under its design-tagged name only (kernels/design_tag.py): an artifact from
// other sources is invisible rather than driven, because running this stream
// against a stale design reads as garbage output with no error at all. The tag
// covers the design sources and the build knobs, not this stream builder - a
// change here needs no artifact rebuild, and no artifact mismatch can hide it.
static const std::string & xdna_fused_xclbin(void) {
    static const std::string path = []() {
        const std::string xclbin = xdna_artifact_find("fused_layer_" XDNA_DESIGN_TAG, false).xclbin;
        if (xclbin.empty()) {
            if (!xdna_artifact_find("fused_layer", false).xclbin.empty()) {
                GGML_LOG_ERROR(
                    "%s: fused_layer.xclbin is present but not built for design "
                    "tag %s; the kernel artifacts are stale. Rebuild them and "
                    "the backend together:\n    cmake --build build --target "
                    "ggml-xdna-kernels\n",
                    "ggml-xdna", XDNA_DESIGN_TAG);
            } else if (xdna_env_int("GGML_XDNA_FUSED_LAYER", 1) != 0) {
                // The fused path is the default decode route and the artifact is
                // what selects it, so a missing one is worth one line. Asking for
                // the per-op path is a choice, not a fault, and stays quiet.
                GGML_LOG_WARN(
                    "%s: no fused_layer artifact for design %s; the decode runs "
                    "the per-op path\n",
                    "ggml-xdna", XDNA_DESIGN_TAG);
            }
        }
        return xclbin;
    }();
    return path;
}

// Fused layer path is active (default): enabled and the fused xclbin is
// present in the kernel search dirs. When it is missing the backend silently
// runs the per-op decode path instead.
static bool xdna_rec_active(void) {
    // GGML_XDNA_FUSED_LAYER=0 forces the per-op decode path. The fused design
    // only covers the quantized weight sets its kernels were built for, so a
    // model it does not cover needs the fallback.
    if (xdna_env_int("GGML_XDNA_FUSED_LAYER", 1) == 0) {
        return false;
    }
    return !xdna_fused_xclbin().empty();
}

// Parse "blk.N." prefixes from weight tensor names; returns false when absent.
static bool xdna_rec_blk(const char * name, int * il, const char ** rest) {
    if (!name || strncmp(name, "blk.", 4) != 0) {
        return false;
    }
    name += 4;
    char * end = nullptr;
    long   v   = strtol(name, &end, 10);
    if (end == name || *end != '.') {
        return false;
    }
    *il   = (int) v;
    *rest = end + 1;
    return true;
}

static bool xdna_rec_is_leaf(const ggml_tensor * t) {
    return t != nullptr && t->op == GGML_OP_NONE;
}

static bool xdna_norm_eps(const ggml_cgraph * cgraph, const ggml_tensor * w, float * eps);

namespace {

// An attention layer's tail - attn_output, the residual add, the
// post-attention norm and the FFN up to l_out - as one dispatch
// (xdna_rec_tail). GGML_XDNA_ATTN_TAIL=0 leaves it to the per-op path.
struct xdna_tail_plan {
    int           il      = -1;
    int           i_fire  = -1;       // the attn_output MUL_MAT
    int           i_lout  = -1;
    ggml_tensor * n_o     = nullptr;  // MUL_MAT attn_output
    ggml_tensor * n_resid = nullptr;  // ADD attn_residual-N
    ggml_tensor * n_lout  = nullptr;  // ADD l_out-N
    ggml_tensor * n_hres  = nullptr;  // the residual attn_residual adds
    ggml_tensor * w_post  = nullptr;
    ggml_tensor * w_gate  = nullptr;
    ggml_tensor * w_up    = nullptr;
    ggml_tensor * w_down  = nullptr;
};

}  // namespace

static bool xdna_tail_on(void) {
    static const bool v = [] {
        return (xdna_env_int("GGML_XDNA_ATTN_TAIL", 1) != 0);
    }();
    return v;
}

static void xdna_tail_scan(const ggml_cgraph * cgraph, std::vector<xdna_tail_plan> & plans) {
    plans.clear();
    std::map<int, xdna_tail_plan> m;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * n = cgraph->nodes[i];
        for (int k = 0; k < GGML_MAX_SRC && n->src[k]; k++) {
            ggml_tensor * w    = n->src[k];
            int           il   = -1;
            const char *  rest = nullptr;
            if (!xdna_rec_is_leaf(w) || !xdna_rec_blk(ggml_get_name(w), &il, &rest)) {
                continue;
            }
            xdna_tail_plan & p = m[il];
            p.il               = il;
            if (strcmp(rest, "attn_output.weight") == 0 && n->op == GGML_OP_MUL_MAT) {
                p.n_o    = n;
                p.i_fire = i;
            } else if (strcmp(rest, "post_attention_norm.weight") == 0) {
                p.w_post = w;
            } else if (strcmp(rest, "ffn_gate.weight") == 0) {
                p.w_gate = w;
            } else if (strcmp(rest, "ffn_up.weight") == 0) {
                p.w_up = w;
            } else if (strcmp(rest, "ffn_down.weight") == 0) {
                p.w_down = w;
            }
        }
        const char * nn = ggml_get_name(n);
        if (strncmp(nn, "attn_residual-", 14) == 0 && n->op == GGML_OP_ADD) {
            m[xdna_atoi(nn + 14)].n_resid = n;
        } else if (strncmp(nn, "l_out-", 6) == 0 && n->op == GGML_OP_ADD) {
            xdna_tail_plan & p = m[xdna_atoi(nn + 6)];
            p.n_lout           = n;
            p.i_lout           = i;
        }
    }
    for (auto & kv : m) {
        xdna_tail_plan & p = kv.second;
        p.il               = kv.first;
        if (!p.n_o || !p.n_resid || !p.n_lout || !p.w_post || !p.w_gate || !p.w_up || !p.w_down || p.i_fire < 0 ||
            p.i_lout <= p.i_fire) {
            continue;
        }
        const ggml_tensor * act = p.n_o->src[1];
        if (p.n_o->ne[1] != 1 || ggml_nelements(p.n_o) != p.n_o->ne[0] || !act || act->type != GGML_TYPE_F32 ||
            ggml_nelements(act) != act->ne[0] || !ggml_is_contiguous(act)) {
            continue;  // single-token decode only
        }
        if (p.n_resid->src[0] == p.n_o) {
            p.n_hres = p.n_resid->src[1];
        } else if (p.n_resid->src[1] == p.n_o) {
            p.n_hres = p.n_resid->src[0];
        }
        if (!p.n_hres || p.n_hres->type != GGML_TYPE_F32 || ggml_nelements(p.n_hres) != p.n_o->ne[0] ||
            p.n_lout->type != GGML_TYPE_F32 || p.n_resid->type != GGML_TYPE_F32) {
            continue;
        }
        plans.push_back(p);
    }
}

xdna_buffer * xdna_host_bo_of(const void * p, size_t * offset);

// Decode flash attention on the pool (xdna-att.h). GGML_XDNA_ATTN=0 leaves
// it to the host.
static bool xdna_att_on(void) {
    static const bool v = [] {
        return (xdna_env_int("GGML_XDNA_ATTN", 1) != 0);
    }();
    return v;
}

// The shapes the pool's attention takes: Qwen3.5's full-attention decode.
static bool xdna_att_eligible(const ggml_tensor * n) {
    const ggml_tensor *q = n->src[0], *k = n->src[1], *v = n->src[2], *m = n->src[3];
    if (n->op != GGML_OP_FLASH_ATTN_EXT || !q || !k || !v || !m || n->src[4]) {
        return false;
    }
    float par[3];
    memcpy(par, n->op_params, sizeof(par));
    return par[1] == 0.0f && par[2] == 0.0f && q->type == GGML_TYPE_F32 && q->ne[0] == 256 && q->ne[1] == 1 &&
           q->ne[2] == 8 && q->ne[3] == 1 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16 && k->ne[0] == 256 &&
           k->ne[2] == 2 && v->ne[0] == 256 && v->ne[2] == 2 && k->ne[3] == 1 && v->ne[3] == 1 &&
           k->ne[1] == v->ne[1] && k->nb[0] == 2 && k->nb[1] == 1024 && k->nb[2] == 512 && v->nb[0] == 2 &&
           v->nb[1] == 1024 && v->nb[2] == 512 && m->type == GGML_TYPE_F16 && m->ne[0] >= k->ne[1] &&
           n->type == GGML_TYPE_F32 && ggml_is_contiguous(n) && ggml_nelements(n) == (int64_t) 256 * 8;
}

static bool xdna_att_try(ggml_backend_xdna_context * ctx, ggml_tensor * n) {
    const ggml_tensor * q = n->src[0], *k = n->src[1], *v = n->src[2], *m = n->src[3];
    // the visible cells: a prefix of the view, as a single sequence's are
    const int64_t       n_kv    = k->ne[1];
    const ggml_fp16_t * mr      = (const ggml_fp16_t *) m->data;
    int                 n_valid = 0;
    while (n_valid < n_kv && ggml_fp16_to_fp32(mr[n_valid]) == 0.0f) {
        n_valid++;
    }
    for (int64_t j = n_valid; j < n_kv; j++) {
        if (ggml_fp16_to_fp32(mr[j]) != -INFINITY) {
            return false;
        }
    }
    if (n_valid <= 0 || n_valid > xdna_att_max_positions()) {
        return false;
    }
    size_t        koff = 0, voff = 0;
    xdna_buffer * kbo = xdna_host_bo_of(k->data, &koff);
    xdna_buffer * vbo = xdna_host_bo_of(v->data, &voff);
    if (!kbo || !vbo) {
        return false;
    }
    if (!ctx->att) {
        ctx->att = xdna_att_create(ctx->pool, ctx->device);
        if (!ctx->att) {
            return false;
        }
    }
    // The rows the host wrote since the last flush of this layer: one a
    // token, or all of them after a prefill.
    int & fl = ctx->att_flushed[k->data];
    if (fl > n_valid) {
        fl = 0;
    }
    if (n_valid > fl) {
        const size_t b0 = (size_t) fl * 1024, nb = (size_t) (n_valid - fl) * 1024;
        if (!xdna_buffer_sync_to_device_range(kbo, nb, koff + b0) ||
            !xdna_buffer_sync_to_device_range(vbo, nb, voff + b0)) {
            return false;
        }
        fl = n_valid;
    }
    float qc[8 * 256];
    for (int h = 0; h < 8; h++) {
        memcpy(qc + (size_t) h * 256, (const char *) q->data + h * q->nb[2], 256 * sizeof(float));
    }
    float scale;
    memcpy(&scale, n->op_params, sizeof(float));
    xdna_prof::section_timer st_att(n_valid <= 1024 ? "att: pool run (<=1k)" :
                                    n_valid <= 4096 ? "att: pool run (<=4k)" :
                                                      "att: pool run (>4k)");
    return xdna_att_run(ctx->att, kbo, koff, vbo, voff, qc, scale, n_valid, (float *) n->data);
}

// Run one attention tail and write attn_residual and l_out.
static bool xdna_tail_run(ggml_backend_xdna_context *               ctx,
                          xdna_tail_plan &                          p,
                          std::unordered_set<const ggml_tensor *> & consumed,
                          const ggml_cgraph *                       cgraph) {
    // The run's silent give-ups: one line per cause.
    const auto give_up = [&](const char * why) {
        std::lock_guard<std::mutex> lk(ctx->gave_up_mutex);
        if (ctx->gave_up.insert(why).second) {
            GGML_LOG_ERROR("%s: attention tail %d: %s\n", "ggml-xdna", p.il, why);
        }
    };
    xdna_rec_tail *& t = ctx->tails[p.il];
    if (!t) {
        float eps = 0.0f;
        if (!ctx->res) {
            ctx->res = xdna_res_alloc(ctx->device);
        }
        if (!ctx->res || !xdna_norm_eps(cgraph, p.w_post, &eps) || ggml_nelements(p.w_post) != XDNA_RES_D) {
            give_up("the residual rows or the post norm are unusable");
            return false;
        }
        t = xdna_rec_tail_create(ctx->pool, p.n_o->src[0], p.w_gate, p.w_up, p.w_down, ctx->res,
                                 (const float *) p.w_post->data, eps);
        if (!t) {
            GGML_LOG_ERROR(
                "%s: attention tail %d: cannot build it (%s, %s/%s/%s); "
                "GGML_XDNA_ATTN_TAIL=0 runs it per op\n",
                "ggml-xdna", p.il, ggml_type_name(p.n_o->src[0]->type), ggml_type_name(p.w_gate->type),
                ggml_type_name(p.w_up->type), ggml_type_name(p.w_down->type));
            return false;
        }
    }
    const int64_t d = p.n_o->ne[0];
    // Both outputs may share memory with the residual input: ggml reuses a
    // dead tensor's buffer, so read it before writing either.
    if (!xdna_res_release(ctx)) {  // it reads the residual and rewrites the rows
        return false;
    }
    std::vector<float> hres((const float *) p.n_hres->data, (const float *) p.n_hres->data + d);
    std::vector<float> h_attn((size_t) d), h_out((size_t) d);
    if (!xdna_rec_tail_run(t, (const float *) p.n_o->src[1]->data, hres.data(), h_attn.data(), h_out.data())) {
        give_up("the dispatch failed");
        return false;
    }
    std::memcpy(p.n_resid->data, h_attn.data(), (size_t) d * sizeof(float));
    std::memcpy(p.n_lout->data, h_out.data(), (size_t) d * sizeof(float));
    // its rows hold these too: the next fused layer or the head takes them
    ctx->res_lout  = p.n_lout;
    ctx->res_resid = p.n_resid;
    ctx->res_dirty = false;
    for (int j = p.i_fire; j <= p.i_lout && j < cgraph->n_nodes; j++) {
        consumed.insert(cgraph->nodes[j]);
    }
    return true;
}

namespace {

// A whole attention layer - from the q/k/v projections to l_out - as one
// dispatch (xdna-att-layer.h). Built on a tail plan, which already holds the
// attention output projection, the residuals and the FFN.
// GGML_XDNA_ATTN_LAYER=0 leaves it to the tail and the pool attention.
struct xdna_attl_plan {
    xdna_tail_plan tail;
    int            i_first  = -1;       // the layer's first projection
    ggml_tensor *  n_q      = nullptr;  // MUL_MAT attn_q (q | gate)
    ggml_tensor *  n_k      = nullptr;
    ggml_tensor *  n_v      = nullptr;
    ggml_tensor *  w_qn     = nullptr;  // attn_q_norm
    ggml_tensor *  w_kn     = nullptr;  // attn_k_norm
    ggml_tensor *  n_fa     = nullptr;
    ggml_tensor *  n_rope   = nullptr;  // q's rope: parameters and positions
    ggml_tensor *  n_rms    = nullptr;  // q's norm: epsilon
    ggml_tensor *  n_srk    = nullptr;  // the KV append
    ggml_tensor *  n_srv    = nullptr;
    ggml_tensor *  w_an     = nullptr;  // attn_norm: the input's norm
    float          eps_attn = 0.0f, eps_post = 0.0f;
    bool           pre = false;         // its input's norm marked consumed up front
};

}  // namespace

static bool xdna_attl_on(void) {
    static const bool v = [] {
        return (xdna_env_int("GGML_XDNA_ATTN_LAYER", 1) != 0);
    }();
    return v;
}

static void xdna_attl_scan(const ggml_cgraph *                 cgraph,
                           const std::vector<xdna_tail_plan> & tails,
                           std::vector<xdna_attl_plan> &       plans) {
    plans.clear();
    for (const xdna_tail_plan & tp : tails) {
        xdna_attl_plan p;
        p.tail  = tp;
        int i_q = -1;
        for (int i = 0; i < tp.i_fire; i++) {
            ggml_tensor * n = cgraph->nodes[i];
            for (int k = 0; k < GGML_MAX_SRC && n->src[k]; k++) {
                ggml_tensor * w    = n->src[k];
                int           il   = -1;
                const char *  rest = nullptr;
                if (!xdna_rec_is_leaf(w) || !xdna_rec_blk(ggml_get_name(w), &il, &rest) || il != tp.il) {
                    continue;
                }
                if (n->op == GGML_OP_MUL_MAT && k == 0) {
                    if (strcmp(rest, "attn_q.weight") == 0) {
                        p.n_q = n;
                        i_q   = i;
                    } else if (strcmp(rest, "attn_k.weight") == 0) {
                        p.n_k = n;
                    } else if (strcmp(rest, "attn_v.weight") == 0) {
                        p.n_v = n;
                    }
                } else if (n->op == GGML_OP_MUL && strcmp(rest, "attn_q_norm.weight") == 0) {
                    p.w_qn                = w;
                    const ggml_tensor * r = n->src[k == 0 ? 1 : 0];
                    if (r && r->op == GGML_OP_RMS_NORM) {
                        p.n_rms = const_cast<ggml_tensor *>(r);
                    }
                } else if (n->op == GGML_OP_MUL && strcmp(rest, "attn_k_norm.weight") == 0) {
                    p.w_kn = w;
                }
            }
        }
        if (i_q < 0 || !p.n_k || !p.n_v || !p.w_qn || !p.w_kn || !p.n_rms) {
            continue;
        }
        p.i_first = i_q;
        for (int i = i_q; i < tp.i_fire; i++) {
            ggml_tensor * n = cgraph->nodes[i];
            if (ggml_is_empty(n)) {
                continue;
            }
            if (n->op == GGML_OP_FLASH_ATTN_EXT && !p.n_fa) {
                p.n_fa = n;
            } else if (n->op == GGML_OP_ROPE && !p.n_rope) {
                p.n_rope = n;
            }
        }
        for (int i = 0; i < tp.i_fire; i++) {
            if (cgraph->nodes[i] == p.n_k || cgraph->nodes[i] == p.n_v) {
                p.i_first = std::min(p.i_first, i);
            }
        }
        if (!p.n_fa || !p.n_rope) {
            // Without flash attention llama keeps V transposed in its cache,
            // which the layer's KV append cannot write (a 2-byte column is
            // below the DMA's 4-byte word), so the layer runs on the host -
            // ~3 ms of CPU a token.
            continue;
        }
        for (int i = p.i_first; i < tp.i_fire; i++) {
            ggml_tensor * n = cgraph->nodes[i];
            if (n->op != GGML_OP_SET_ROWS) {
                continue;
            }
            // The append writes the whole cache (with several streams, one
            // per server slot, reshaped to 2D across them, its indices
            // global), and attention reads one stream's view of it - so the
            // view lies inside the append's target rather than at its start.
            const auto inside = [n](const ggml_tensor * view) {
                const char * b = (const char *) n->data;
                const char * d = (const char *) view->data;
                return d >= b && d < b + ggml_nbytes(n);
            };
            if (inside(p.n_fa->src[1])) {
                p.n_srk = n;
            } else if (inside(p.n_fa->src[2])) {
                p.n_srv = n;
            }
        }
        if (!p.n_srk || !p.n_srv) {
            continue;
        }
        // the input's norm (attn_norm) and the post-attention norm: gamma and
        // epsilon, off the MUL and the RMS_NORM under it
        const ggml_tensor * xn = p.n_q->src[1];
        if (!xn || xn->op != GGML_OP_MUL || !xn->src[0] || xn->src[0]->op != GGML_OP_RMS_NORM ||
            !xdna_rec_is_leaf(xn->src[1])) {
            continue;
        }
        p.w_an = xn->src[1];
        memcpy(&p.eps_attn, xn->src[0]->op_params, sizeof(float));
        bool post_ok = false;
        for (int i = tp.i_fire; i <= tp.i_lout; i++) {
            const ggml_tensor * n = cgraph->nodes[i];
            if (n->op == GGML_OP_MUL && n->src[1] == tp.w_post && n->src[0] && n->src[0]->op == GGML_OP_RMS_NORM) {
                memcpy(&p.eps_post, n->src[0]->op_params, sizeof(float));
                post_ok = true;
            }
        }
        if (!post_ok) {
            continue;
        }
        plans.push_back(p);
    }
}

// cos/sin of the rotated pairs for the one position of a single-token rope,
// the way ggml's CPU rope computes them (ggml_mrope_cache_init, rope_yarn).
static bool xdna_rope_table(const ggml_tensor *  rope,
                            std::vector<float> & cosv,
                            std::vector<float> & sinv,
                            int *                n_rot) {
    const int32_t * op     = (const int32_t *) rope->op_params;
    const int       n_dims = op[1], mode = op[2], n_ctx_orig = op[4];
    float           freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&freq_base, op + 5, sizeof(float));
    memcpy(&freq_scale, op + 6, sizeof(float));
    memcpy(&ext_factor, op + 7, sizeof(float));
    memcpy(&attn_factor, op + 8, sizeof(float));
    memcpy(&beta_fast, op + 9, sizeof(float));
    memcpy(&beta_slow, op + 10, sizeof(float));
    int sections[4];
    memcpy(sections, op + 11, sizeof(sections));
    const bool is_imrope  = mode == GGML_ROPE_TYPE_IMROPE;
    const bool mrope_used = (mode & GGML_ROPE_TYPE_MROPE) != 0;
    const bool neox_pairs =
        mode == GGML_ROPE_TYPE_NEOX || mode == GGML_ROPE_TYPE_MROPE || mode == GGML_ROPE_TYPE_IMROPE;
    const ggml_tensor * pos_t = rope->src[1];
    if (!neox_pairs || n_dims <= 0 || n_dims % 32 != 0 || n_dims > 256 || rope->src[2] || !pos_t ||
        pos_t->type != GGML_TYPE_I32 || rope->src[0]->ne[2] != 1) {
        return false;
    }
    const int32_t * pos         = (const int32_t *) pos_t->data;
    const float     theta_scale = powf(freq_base, -2.0f / (float) n_dims);
    float           corr_dims[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims);
    const auto yarn = [&](float theta_extrap, int i0, float * c, float * s) {
        const float theta_interp = freq_scale * theta_extrap;
        float       theta        = theta_interp;
        float       mscale       = attn_factor;
        if (ext_factor != 0.0f) {
            // NOLINTNEXTLINE(bugprone-integer-division): i0 is the even rope dimension index
            const float y        = ((float) (i0 / 2) - corr_dims[0]) / std::max(0.001f, corr_dims[1] - corr_dims[0]);
            const float ramp_mix = (1 - std::min(1.0f, std::max(0.0f, y))) * ext_factor;
            theta                = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }
        *c = cosf(theta) * mscale;
        *s = sinf(theta) * mscale;
    };
    cosv.assign((size_t) n_dims / 2, 0.0f);
    sinv.assign((size_t) n_dims / 2, 0.0f);
    if (!mrope_used) {
        float theta = (float) pos[0];
        for (int i0 = 0; i0 < n_dims; i0 += 2) {
            yarn(theta, i0, &cosv[i0 / 2], &sinv[i0 / 2]);
            theta *= theta_scale;
        }
    } else {
        const int sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
        const int sec_w     = sections[1] + sections[0];
        if (sect_dims <= 0) {
            return false;
        }
        // one token: ne2 = 1, so its four position components are pos[0..3]
        float theta_t = (float) pos[0], theta_h = (float) pos[1];
        float theta_w = (float) pos[2], theta_e = (float) pos[3];
        for (int i0 = 0; i0 < n_dims; i0 += 2) {
            const int sector = (i0 / 2) % sect_dims;
            float     theta  = theta_t;
            if (is_imrope) {
                if (sector % 3 == 1 && sector < 3 * sections[1]) {
                    theta = theta_h;
                } else if (sector % 3 == 2 && sector < 3 * sections[2]) {
                    theta = theta_w;
                } else if (sector % 3 == 0 && sector < 3 * sections[0]) {
                    theta = theta_t;
                } else {
                    theta = theta_e;
                }
            } else {
                if (sector >= sections[0] && sector < sec_w) {
                    theta = theta_h;
                } else if (sector >= sec_w && sector < sec_w + sections[2]) {
                    theta = theta_w;
                } else if (sector >= sec_w + sections[2]) {
                    theta = theta_e;
                }
            }
            yarn(theta, i0, &cosv[i0 / 2], &sinv[i0 / 2]);
            theta_t *= theta_scale;
            theta_w *= theta_scale;
            theta_h *= theta_scale;
            theta_e *= theta_scale;
        }
    }
    *n_rot = n_dims;
    return true;
}

// The cache positions a single-token attention reads, off its mask: a
// prefix of the view, -1 when the mask is not one.
static int xdna_attl_valid(const ggml_tensor * fa) {
    const ggml_tensor * k = fa->src[1], *m = fa->src[3];
    const int64_t       n_kv    = k->ne[1];
    const ggml_fp16_t * mr      = (const ggml_fp16_t *) m->data;
    int                 n_valid = 0;
    while (n_valid < n_kv && ggml_fp16_to_fp32(mr[n_valid]) == 0.0f) {
        n_valid++;
    }
    for (int64_t j = n_valid; j < n_kv; j++) {
        if (ggml_fp16_to_fp32(mr[j]) != -INFINITY) {
            return -1;
        }
    }
    return n_valid;
}

// Whether the layer will run as one dispatch this token, from what the graph
// holds before any of it runs: the layer is built, the token fits it.
static bool xdna_attl_ready(ggml_backend_xdna_context * ctx, const xdna_attl_plan & p) {
    const auto it = ctx->att_layers.find(p.tail.il);
    if (it == ctx->att_layers.end() || !it->second || !xdna_att_eligible(p.n_fa)) {
        return false;
    }
    const int     n_valid = xdna_attl_valid(p.n_fa);
    size_t        koff = 0, voff = 0;
    xdna_buffer * kbo = xdna_host_bo_of(p.n_fa->src[1]->data, &koff);
    xdna_buffer * vbo = xdna_host_bo_of(p.n_fa->src[2]->data, &voff);
    if (n_valid <= 0 || !kbo || !vbo || !xdna_att_layer_fits(kbo, koff, vbo, voff, n_valid)) {
        return false;
    }
    // the token's row is the last visible cell of its stream: the view's
    // offset into the append's target, in rows, plus n_valid - 1
    const ggml_tensor * views[2] = { p.n_fa->src[1], p.n_fa->src[2] };
    const ggml_tensor * srs[2]   = { p.n_srk, p.n_srv };
    for (int i = 0; i < 2; i++) {
        const ggml_tensor *sr = srs[i], *idx = sr->src[1];
        if (!idx || ggml_nelements(idx) != 1) {
            return false;
        }
        const size_t off = (size_t) ((const char *) views[i]->data - (const char *) sr->data);
        if (off % sr->nb[1]) {
            return false;
        }
        const int64_t r = idx->type == GGML_TYPE_I64 ? ((const int64_t *) idx->data)[0] :
                          idx->type == GGML_TYPE_I32 ? ((const int32_t *) idx->data)[0] :
                                                       -1;
        if (r != (int64_t) (off / sr->nb[1]) + n_valid - 1) {
            return false;
        }
    }
    return true;
}

// Run one attention layer as one dispatch. False without having changed
// anything when this token does not fit it - the caller then runs the layer
// the other way.
static bool xdna_attl_try(ggml_backend_xdna_context *               ctx,
                          xdna_attl_plan &                          p,
                          std::unordered_set<const ggml_tensor *> & consumed,
                          const ggml_cgraph *                       cgraph,
                          bool *                                    failed) {
    *failed          = false;
    ggml_tensor * fa = p.n_fa;
    if (!xdna_att_eligible(fa)) {
        return false;
    }
    const ggml_tensor *k = fa->src[1], *v = fa->src[2];
    const int          n_valid = xdna_attl_valid(fa);
    if (n_valid <= 0 || n_valid > xdna_att_layer_max_positions()) {
        return false;
    }
    // this token's K and V go to row n_valid - 1 of its stream: the one row
    // the mask opened last, a single sequence's contiguous cache (the append's
    // indices are global across streams, the view starts at its stream)
    const ggml_tensor * views[2] = { k, v };
    const ggml_tensor * srs[2]   = { p.n_srk, p.n_srv };
    for (int i = 0; i < 2; i++) {
        const ggml_tensor *sr = srs[i], *idx = sr->src[1];
        if (!idx || ggml_nelements(idx) != 1 || sr->src[0]->ne[1] != 1 ||
            sr->src[0]->ne[0] * sr->src[0]->ne[2] != 512 || sr->nb[1] != 1024) {
            return false;
        }
        const size_t  off = (size_t) ((const char *) views[i]->data - (const char *) sr->data);
        const int64_t r   = idx->type == GGML_TYPE_I64 ? ((const int64_t *) idx->data)[0] :
                            idx->type == GGML_TYPE_I32 ? ((const int32_t *) idx->data)[0] :
                                                         -1;
        if (off % sr->nb[1] || r != (int64_t) (off / sr->nb[1]) + n_valid - 1) {
            return false;
        }
    }
    const ggml_tensor * x = p.n_q->src[1];
    if (!x || x->type != GGML_TYPE_F32 || ggml_nelements(x) != x->ne[0] || !ggml_is_contiguous(x) ||
        p.n_k->src[1] != x || p.n_v->src[1] != x) {
        return false;
    }
    size_t        koff = 0, voff = 0;
    xdna_buffer * kbo = xdna_host_bo_of(k->data, &koff);
    xdna_buffer * vbo = xdna_host_bo_of(v->data, &voff);
    if (!kbo || !vbo || !xdna_att_layer_fits(kbo, koff, vbo, voff, n_valid)) {
        return false;
    }
    std::vector<float> cosv, sinv;
    int                n_rot = 0;
    if (!xdna_rope_table(p.n_rope, cosv, sinv, &n_rot)) {
        return false;
    }
    auto it = ctx->att_layers.find(p.tail.il);
    if (it == ctx->att_layers.end()) {
        xdna_att_layer_w w;
        w.wq               = p.n_q->src[0];
        w.wk               = p.n_k->src[0];
        w.wv               = p.n_v->src[0];
        w.gq               = p.w_qn;
        w.gk               = p.w_kn;
        w.attn_norm        = p.w_an;
        w.wo               = p.tail.n_o->src[0];
        w.post             = p.tail.w_post;
        w.gate             = p.tail.w_gate;
        w.up               = p.tail.w_up;
        w.down             = p.tail.w_down;
        xdna_att_layer * l = nullptr;
        {
            xdna_arena_scope arena;  // its buffers join the token's command
            l = xdna_att_layer_create(ctx->pool, w);
        }
        it = ctx->att_layers.emplace(p.tail.il, l).first;
    }
    if (!it->second) {
        return false;
    }
    // The rows the host wrote since the last flush (a prefill), as the pool
    // attention does.
    int & fl = ctx->att_flushed[k->data];
    if (fl > n_valid - 1) {
        fl = 0;
    }
    if (n_valid - 1 > fl) {
        const size_t b0 = (size_t) fl * 1024, nb = (size_t) (n_valid - 1 - fl) * 1024;
        if (!xdna_buffer_sync_to_device_range(kbo, nb, koff + b0) ||
            !xdna_buffer_sync_to_device_range(vbo, nb, voff + b0)) {
            return false;
        }
    }
    const int64_t d = p.tail.n_o->ne[0];
    if (d != XDNA_RES_D) {
        return false;
    }
    if (!ctx->res) {
        ctx->res = xdna_res_alloc(ctx->device);
        if (!ctx->res) {
            return false;
        }
    }
    // The layer's input as the rows it reads, h = F + A: already there when
    // the layer before it was a fused one, the host's copy otherwise.
    if (p.tail.n_hres != ctx->res_lout) {
        if (!xdna_res_release(ctx)) {
            return false;
        }
        float * r = (float *) ctx->res->data;
        std::memcpy(r + XDNA_RES_F / 4, p.tail.n_hres->data, (size_t) d * sizeof(float));
        std::memset(r + XDNA_RES_A / 4, 0, (size_t) d * sizeof(float));
        if (!xdna_buffer_sync_to_device(ctx->res)) {
            return false;
        }
    }
    xdna_att_layer_in in;
    in.res      = ctx->res;
    in.eps_attn = p.eps_attn;
    in.eps_post = p.eps_post;
    in.kbo      = kbo;
    in.koff     = koff;
    in.vbo      = vbo;
    in.voff     = voff;
    in.n_valid  = n_valid;
    in.n_rot    = n_rot;
    in.cosv     = cosv.data();
    in.sinv     = sinv.data();
    memcpy(&in.scale, fa->op_params, sizeof(float));
    memcpy(&in.eps, p.n_rms->op_params, sizeof(float));
    if (xdna_queue_on()) {
        ggml_backend_xdna_context::pending_run pr;
        pr.cosv = std::move(cosv);
        pr.sinv = std::move(sinv);
        in.cosv = pr.cosv.data();
        in.sinv = pr.sinv.data();
        xdna_batch_open();
        pr.run = xdna_att_layer_start(it->second, in);
        if (!pr.run) {
            *failed = true;
            return false;
        }
        pr.att = true;
        pr.in  = in;
        ctx->pending.push_back(std::move(pr));
        ctx->pending.back().in.cosv = ctx->pending.back().cosv.data();
        ctx->pending.back().in.sinv = ctx->pending.back().sinv.data();
    } else if (!xdna_att_layer_run(it->second, in)) {
        *failed = true;
        return false;
    }
    fl             = n_valid;
    // The outputs stay in the rows until something outside the fused layers
    // reads them.
    ctx->res_lout  = p.tail.n_lout;
    ctx->res_resid = p.tail.n_resid;
    ctx->res_dirty = true;
    for (int j = p.i_first; j <= p.tail.i_lout && j < cgraph->n_nodes; j++) {
        consumed.insert(cgraph->nodes[j]);
    }
    return true;
}

// One pre-scan pass over a graph_compute node list; fills plans for every
// complete recurrent layer found. A plan is complete when all data nodes the
// fused run needs (and every weight leaf) are present in this chunk.
static bool xdna_rec_scan(ggml_backend_xdna_context *  ctx,
                          const ggml_cgraph *          cgraph,
                          std::vector<xdna_rec_plan> & plans) {
    GGML_UNUSED(ctx);
    plans.clear();
    std::map<int, xdna_rec_plan> m;
    const int                    n_nodes = cgraph->n_nodes;

    // First pass: find the per-layer weight leaves and the layer data nodes by
    // name / src-weight structure.
    for (int i = 0; i < n_nodes; i++) {
        const ggml_tensor * n = cgraph->nodes[i];
        for (int s = 0; s < GGML_MAX_SRC && n->src[s]; s++) {
            const ggml_tensor * w = n->src[s];
            if (!xdna_rec_is_leaf(w)) {
                continue;
            }
            int          il   = -1;
            const char * rest = nullptr;
            if (!xdna_rec_blk(ggml_get_name(w), &il, &rest)) {
                continue;
            }
            xdna_rec_plan & p = m[il];
            p.il              = il;
            if (strcmp(rest, "ssm_conv1d.weight") == 0) {
                p.w_conv = const_cast<ggml_tensor *>(w);
                p.i_conv = n->op == GGML_OP_SSM_CONV ? i : p.i_conv;
            } else if (strcmp(rest, "ssm_norm.weight") == 0) {
                p.w_gamma = const_cast<ggml_tensor *>(w);
            } else if (strcmp(rest, "post_attention_norm.weight") == 0) {
                p.w_post = const_cast<ggml_tensor *>(w);
            } else if (strcmp(rest, "ssm_out.weight") == 0) {
                p.w_so = const_cast<ggml_tensor *>(w);
            } else if (strcmp(rest, "ffn_gate.weight") == 0) {
                p.w_gate = const_cast<ggml_tensor *>(w);
            } else if (strcmp(rest, "ffn_up.weight") == 0) {
                p.w_up = const_cast<ggml_tensor *>(w);
            } else if (strcmp(rest, "ffn_down.weight") == 0) {
                p.w_down = const_cast<ggml_tensor *>(w);
            } else if (strcmp(rest, "attn_qkv.weight") == 0 && n->op == GGML_OP_MUL_MAT) {
                p.n_qkv = const_cast<ggml_tensor *>(n);
            } else if (strcmp(rest, "attn_gate.weight") == 0 && n->op == GGML_OP_MUL_MAT) {
                p.n_z = const_cast<ggml_tensor *>(n);
            }
        }
        // Named data nodes of a recurrent layer (name suffix "-<il>", e.g.
        // "gate-0", "l_out-0"); weight leaves were handled above.
        const char * nn = ggml_get_name(n);
        if (strncmp(nn, "gate-", 5) == 0 && n->op == GGML_OP_MUL) {
            const int il = xdna_atoi(nn + 5);
            m[il].il     = il;
            m[il].n_gate = const_cast<ggml_tensor *>(n);
        } else if (strncmp(nn, "beta_sigmoid-", 13) == 0) {
            const int il = xdna_atoi(nn + 13);
            m[il].il     = il;
            m[il].n_beta = const_cast<ggml_tensor *>(n);
        } else if (strncmp(nn, "conv_states-", 12) == 0 && n->op == GGML_OP_GET_ROWS) {
            const int il   = xdna_atoi(nn + 12);
            m[il].il       = il;
            m[il].n_convst = const_cast<ggml_tensor *>(n);
        } else if (strncmp(nn, "l_out-", 6) == 0 && n->op == GGML_OP_ADD) {
            const int il = xdna_atoi(nn + 6);
            m[il].il     = il;
            m[il].n_lout = const_cast<ggml_tensor *>(n);
            m[il].i_lout = i;
        } else if (strncmp(nn, "attn_residual-", 14) == 0 && n->op == GGML_OP_ADD) {
            const int il  = xdna_atoi(nn + 14);
            m[il].il      = il;
            m[il].n_resid = const_cast<ggml_tensor *>(n);
        }
        // cache_s read (GET_ROWS over the state cell), the ssm seed.
        if (n->op == GGML_OP_GET_ROWS && n->ne[0] == xdna_rec_pack::state_floats() && n->ne[1] == 1) {
            const ggml_tensor * r = n->src[0];
            if (r && ggml_get_name(r)) {
                const char * src_name = ggml_get_name(r);
                if (strncmp(src_name, "cache_s_l", 9) == 0) {
                    const int il   = xdna_atoi(src_name + 9);
                    m[il].il       = il;
                    m[il].n_sstate = const_cast<ggml_tensor *>(n);
                }
            }
        }
    }

    // Decide the fire index and drop layers that are not single-token decodes
    // or miss a needed data node in this chunk.
    for (auto & kv : m) {
        xdna_rec_plan & p = kv.second;
        if (!p.w_conv || !p.w_so || !p.w_gamma || !p.w_post || !p.w_gate || !p.w_up || !p.w_down || !p.n_qkv ||
            !p.n_z || !p.n_gate || !p.n_beta || !p.n_resid || !p.n_lout || !p.n_convst || !p.n_sstate || p.i_conv < 0 ||
            p.i_lout < 0) {
            continue;  // not a complete recurrent layer in this chunk
        }
        // Single-token, single-sequence decode only (qkv rows == 1).
        if (p.n_qkv->ne[1] != 1 || p.n_qkv->ne[2] != 1 || p.n_qkv->ne[3] != 1) {
            continue;
        }
        int maxi          = 0;
        // Both projections read the same normed row; it is the fire's input
        // when they are in its stream.
        p.n_act           = p.n_qkv->src[1];
        const bool inproj = xdna_rec_inproj_on() && p.n_act && p.n_z->src[1] == p.n_act &&
                            p.n_act->type == GGML_TYPE_F32 && p.n_act->ne[0] == p.n_qkv->src[0]->ne[0] &&
                            ggml_nelements(p.n_act) == p.n_act->ne[0];
        if (!inproj) {
            p.n_act = nullptr;
        }
        const ggml_tensor * need[] = {
            inproj ? p.n_act : p.n_qkv, inproj ? p.n_act : p.n_z, p.n_gate, p.n_beta, p.n_convst, p.n_sstate
        };
        for (const ggml_tensor * t : need) {
            for (int j = 0; j < n_nodes; j++) {
                if (cgraph->nodes[j] == t) {
                    if (j > maxi) {
                        maxi = j;
                    }
                    break;
                }
            }
        }
        if (maxi + 1 > p.i_lout) {
            continue;
        }
        p.i_fire   = maxi + 1;
        p.n_hres   = p.n_resid->src[1];
        auto idxof = [&](const ggml_tensor * t) {
            if (!t) {
                return -1;
            }
            for (int j = 0; j < n_nodes; j++) {
                if (cgraph->nodes[j] == t) {
                    return j;
                }
            }
            return -1;
        };
        // The fused run reads qkv/conv/ssm seeds long after llama recycled
        // their transient buffers, so their producer nodes are snapshotted
        // right when they execute (i_cap_*). hres may be a graph leaf (then
        // i_cap_hres stays -1 and the live tensor is read instead).
        p.i_cap_qkv  = p.n_act ? -1 : idxof(p.n_qkv);
        p.i_cap_act  = p.n_act ? idxof(p.n_act) : -1;
        p.i_cap_conv = idxof(p.n_convst);
        p.i_cap_ss   = idxof(p.n_sstate);
        p.i_cap_hres = idxof(p.n_hres);
        if ((p.n_act ? p.i_cap_act : p.i_cap_qkv) < 0 || p.i_cap_conv < 0 || p.i_cap_ss < 0) {
            continue;  // producers not in this chunk: nothing to snapshot
        }
        plans.push_back(p);
    }
    return !plans.empty();
}

// The recurrent-memory row the layer's state read points at: the sequence's
// cell slot, which is what tells two sequences sharing one context apart.
static int64_t xdna_rec_seq_row(const xdna_rec_plan & p) {
    const ggml_tensor * idx = p.n_sstate ? p.n_sstate->src[1] : nullptr;
    if (idx && idx->data && idx->type == GGML_TYPE_I32 && idx->ne[0] >= 1) {
        return ((const int32_t *) idx->data)[0];
    }
    return 0;
}

// Find (or create) the fused session for a recurrent block; which sequence it
// holds is its own `row`.
static xdna_rec_session * xdna_rec_session_get(ggml_backend_xdna_context * ctx, int il) {
    const auto key = std::make_pair(il, (int64_t) 0);
    auto       it  = ctx->rec.find(key);
    if (it != ctx->rec.end()) {
        return it->second;
    }
    xdna_rec_session * s = new xdna_rec_session;
    s->il                = il;
    ctx->rec[key]        = s;
    return s;
}

// The fused sessions carry the recurrent state on the device, and llama's
// cache only sees it again here. Two things in a graph mean that copy is no
// longer the sequence's state:
//
// - A graph of more than one token (a prefill) reads the state from llama's
//   cache and writes its result there. Whatever the device carries is written
//   back into the cache first - a prefill that continues the sequence starts
//   where the decode left it, and one that starts a new sequence zeroes the
//   cell itself - and the session reseeds from the cache on its next decode.
// - A graph that zeroes a session's cell (llama's rs_z: a new sequence in that
//   cell) resets it: the session reseeds from the zeroed cache.
//
// Without this, a second request on the same slot decoded on top of the
// first one's state.
static void xdna_rec_follow_cache(ggml_backend_xdna_context * ctx, const ggml_cgraph * cgraph) {
    if (ctx->rec.empty()) {
        return;
    }
    std::map<int, ggml_tensor *>      cache_s, cache_r;
    std::set<std::pair<int, int64_t>> zeroed;
    bool                              multi    = false;
    const auto                        cache_of = [](const ggml_tensor * t, int & il) -> int {
        const ggml_tensor * b = t && t->view_src ? t->view_src : t;
        const char *        n = b ? ggml_get_name(b) : nullptr;
        if (!n) {
            return 0;
        }
        if (strncmp(n, "cache_s_l", 9) == 0) {
            il = xdna_atoi(n + 9);
            return 1;
        }
        if (strncmp(n, "cache_r_l", 9) == 0) {
            il = xdna_atoi(n + 9);
            return 2;
        }
        return 0;
    };
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * n = cgraph->nodes[i];
        if (n->op == GGML_OP_SSM_CONV && n->ne[1] > 1) {
            multi = true;
        }
        for (int k = -1; k < GGML_MAX_SRC; k++) {
            const ggml_tensor * t    = k < 0 ? n : n->src[k];
            int                 il   = -1;
            const int           kind = t ? cache_of(t, il) : 0;
            if (kind) {
                ggml_tensor * b                     = t->view_src ? t->view_src : const_cast<ggml_tensor *>(t);
                (kind == 1 ? cache_s : cache_r)[il] = b;
            }
        }
        // rs_z: an in-place scale by zero of one whole cell, empty when no
        // cell is cleared.
        int il = -1;
        if (n->op == GGML_OP_SCALE && n->view_src && cache_of(n, il) && ggml_nelements(n) > 0) {
            const size_t row_b = (size_t) ggml_nelements(n) * sizeof(float);
            const size_t off   = (size_t) ((const char *) n->data - (const char *) n->view_src->data);
            zeroed.insert(std::make_pair(il, (int64_t) (off / row_b)));
        }
    }
    for (auto & kv : ctx->rec) {
        xdna_rec_session * s   = kv.second;
        const int          il  = kv.first.first;
        const int64_t      row = s ? s->row : -1;
        if (!s || !s->seeded) {
            continue;
        }
        if (zeroed.count(std::make_pair(il, row))) {
            s->seeded = false;
            continue;
        }
        if (!multi || !cache_s.count(il) || !cache_r.count(il)) {
            continue;
        }
        ggml_tensor * ts = cache_s[il];
        ggml_tensor * tr = cache_r[il];
        const size_t  ns = (size_t) xdna_rec_pack::state_floats();
        const size_t  nr = (size_t) 3 * xdna_rec_pack::CH;
        if (ts->type != GGML_TYPE_F32 || tr->type != GGML_TYPE_F32 || !ts->data || !tr->data ||
            ggml_nbytes(ts) < (size_t) (row + 1) * ns * sizeof(float) ||
            ggml_nbytes(tr) < (size_t) (row + 1) * nr * sizeof(float)) {
            GGML_LOG_WARN(
                "%s: fused layer %d: cannot write its state back to "
                "llama's cache; it reseeds from the cache as is\n",
                "ggml-xdna", il);
            s->seeded = false;
            continue;
        }
        if (!xdna_rec_core_read_state(s->core, (float *) tr->data + (size_t) row * nr,
                                      (float *) ts->data + (size_t) row * ns)) {
            GGML_LOG_WARN("%s: fused layer %d: state readback failed\n", "ggml-xdna", il);
        }
        s->seeded = false;
    }
}

static const float * xdna_rec_tdata(const ggml_tensor * t, int64_t want) {
    if (!t || !t->data || t->type != GGML_TYPE_F32 || t->ne[0] * t->ne[1] < want) {
        return nullptr;
    }
    return (const float *) t->data;
}

// Run one fused layer for the current decode token and write h_attn / h_out
// into the layer's add tensors. Requires pending NPU work and the glue batch
// to be flushed by the caller. Returns false when the device run fails.
// The epsilon of the RMS norm whose gamma is `w`: the MUL by it and the
// RMS_NORM under that.
static bool xdna_norm_eps(const ggml_cgraph * cgraph, const ggml_tensor * w, float * eps) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * n = cgraph->nodes[i];
        if (n->op == GGML_OP_MUL && n->src[1] == w && n->src[0] && n->src[0]->op == GGML_OP_RMS_NORM) {
            memcpy(eps, n->src[0]->op_params, sizeof(float));
            return true;
        }
    }
    return false;
}

// Through reshapes and views to the op that made a tensor.
static const ggml_tensor * xdna_under_views(const ggml_tensor * t) {
    while (t && (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW) && t->src[0]) {
        t = t->src[0];
    }
    return t;
}

// A weight's rows as f32.
static bool xdna_rows_f32(const ggml_tensor * w, std::vector<float> & out) {
    const ggml_type_traits * tt = ggml_get_type_traits(w->type);
    if (!w->data || (w->type != GGML_TYPE_F32 && !tt->to_float)) {
        return false;
    }
    out.resize((size_t) ggml_nelements(w));
    for (int64_t r = 0; r < w->ne[1]; r++) {
        const char * src = (const char *) w->data + r * w->nb[1];
        float *      dst = out.data() + r * w->ne[0];
        if (w->type == GGML_TYPE_F32) {
            memcpy(dst, src, (size_t) w->ne[0] * sizeof(float));
        } else {
            tt->to_float(src, dst, w->ne[0]);
        }
    }
    return true;
}

// The GDN gates' pieces off the graph: gate = softplus(W_a x + dt) * a and
// beta = sigmoid(W_b x), the weights with NVH rows of the row length each.
static bool xdna_gates_of(const xdna_rec_plan & p,
                          std::vector<float> &  wa,
                          std::vector<float> &  wb,
                          std::vector<float> &  dt,
                          std::vector<float> &  a) {
    using namespace xdna_rec_pack;
    const ggml_tensor * g = p.n_gate;
    if (!g || g->op != GGML_OP_MUL || !xdna_rec_is_leaf(g->src[1]) || !p.n_beta) {
        return false;
    }
    const ggml_tensor * sp = xdna_under_views(g->src[0]);
    if (!sp || sp->op != GGML_OP_UNARY || ggml_get_unary_op(sp) != GGML_UNARY_OP_SOFTPLUS) {
        return false;
    }
    const ggml_tensor * add = xdna_under_views(sp->src[0]);
    if (!add || add->op != GGML_OP_ADD || !xdna_rec_is_leaf(add->src[1])) {
        return false;
    }
    const ggml_tensor * ma = xdna_under_views(add->src[0]);
    const ggml_tensor * sg = p.n_beta;
    if (sg->op != GGML_OP_UNARY || ggml_get_unary_op(sg) != GGML_UNARY_OP_SIGMOID) {
        return false;
    }
    const ggml_tensor * mb = xdna_under_views(sg->src[0]);
    if (!ma || !mb || ma->op != GGML_OP_MUL_MAT || mb->op != GGML_OP_MUL_MAT || ma->src[0]->ne[0] != XDNA_RES_D ||
        ma->src[0]->ne[1] != NVH || mb->src[0]->ne[0] != XDNA_RES_D || mb->src[0]->ne[1] != NVH ||
        ma->src[1] != p.n_act || mb->src[1] != p.n_act || ggml_nelements(add->src[1]) != NVH ||
        ggml_nelements(g->src[1]) != NVH) {
        return false;
    }
    return xdna_rows_f32(ma->src[0], wa) && xdna_rows_f32(mb->src[0], wb) && xdna_rows_f32(add->src[1], dt) &&
           xdna_rows_f32(g->src[1], a);
}

static bool xdna_rec_run(ggml_backend_xdna_context *               ctx,
                         xdna_rec_plan &                           p,
                         std::unordered_set<const ggml_tensor *> & consumed,
                         const ggml_cgraph *                       cgraph) {
    using namespace xdna_rec_pack;
    // Ways the run gives up before or at a dispatch, which the caller only sees
    // as false: one line per cause, naming the layer.
    const auto give_up = [&](const char * why) {
        std::lock_guard<std::mutex> lk(ctx->gave_up_mutex);
        if (ctx->gave_up.insert(why).second) {
            GGML_LOG_ERROR("%s: fused layer %d: %s\n", "ggml-xdna", p.il, why);
        }
    };
    xdna_rec_session * s = xdna_rec_session_get(ctx, p.il);
    if (!s) {
        return false;
    }
    // Another sequence's token: the state on the device is the one the
    // session last decoded, so it goes back into that sequence's cell and the
    // session reseeds from this one's (the graph read it, since the session
    // did not count as seeded for this row, see the node skip in the graph
    // walk).
    const int64_t row = xdna_rec_seq_row(p);
    if (s->seeded && s->row != row) {
        xdna_pending_wait(ctx);
        const ggml_tensor * ts = p.n_sstate->src[0];
        const ggml_tensor * tr = p.n_convst->src[0];
        const size_t        ns = (size_t) state_floats();
        const size_t        nr = (size_t) 3 * CH;
        if (!ts || !tr || ts->type != GGML_TYPE_F32 || tr->type != GGML_TYPE_F32 || !ts->data || !tr->data ||
            s->row < 0 || ggml_nbytes(ts) < (size_t) (s->row + 1) * ns * sizeof(float) ||
            ggml_nbytes(tr) < (size_t) (s->row + 1) * nr * sizeof(float) ||
            !xdna_rec_core_read_state(s->core, (float *) tr->data + (size_t) s->row * nr,
                                      (float *) ts->data + (size_t) s->row * ns)) {
            GGML_LOG_WARN(
                "%s: fused layer %d: cannot write sequence row %lld's state "
                "back to llama's cache\n",
                "ggml-xdna", p.il, (long long) s->row);
        }
        s->seeded = false;
    }
    s->row = row;

    // qkv / conv / ssm seed / residual are transient graph tensors that llama
    // recycled by the time this fire runs; the dispatch loop snapshotted their
    // producer outputs into s->cap_* right after each executed. gate/z/beta are
    // produced at the very end of the layer (immediately before fire) and stay
    // valid, so they are read live from their tensors.
    const bool    inproj = p.n_act != nullptr;
    const float * act    = inproj && !s->cap_act.empty() ? s->cap_act.data() : nullptr;
    const float * qkv    = inproj ? nullptr : !s->cap_qkv.empty() ? s->cap_qkv.data() : xdna_rec_tdata(p.n_qkv, CH);
    const float * z      = inproj ? nullptr : xdna_rec_tdata(p.n_z, (int64_t) NVH * SV);
    const float * gate   = xdna_rec_tdata(p.n_gate, NVH);
    const float * beta   = xdna_rec_tdata(p.n_beta, NVH);
    if ((inproj ? !act : (!qkv || !z)) || !gate || !beta) {
        GGML_LOG_ERROR("%s: fused layer %d: bad input tensors\n", "ggml-xdna", p.il);
        return false;
    }

    if (!s->seeded) {
        const float * ch = !s->cap_conv.empty() ? s->cap_conv.data() : xdna_rec_tdata(p.n_convst, (int64_t) 3 * CH);
        const float * st = !s->cap_ss.empty() ? s->cap_ss.data() : xdna_rec_tdata(p.n_sstate, state_floats());
        if (!ch || !st) {
            GGML_LOG_ERROR("%s: fused layer %d: no recurrent state to seed\n", "ggml-xdna", p.il);
            return false;
        }
        // A reseed (the state was reset or written back for a prefill)
        // keeps the session's device objects and only reloads its state.
        const bool create   = s->core == nullptr;
        s->host.il          = p.il;
        s->host.w_conv      = (const float *) p.w_conv->data;
        s->host.w_ssm_norm  = (const float *) p.w_gamma->data;
        s->host.w_post_norm = (const float *) p.w_post->data;
        // The post norm's epsilon is the graph's, as on the other norm paths
        // (xdna_tail_run, the fused prologue); the default stands only when the
        // graph does not carry the norm.
        if (!xdna_norm_eps(cgraph, p.w_post, &s->eps_post)) {
            s->eps_post = 1e-6f;
        }

        // The whole recurrent layer runs on the fused design only when the
        // quantized weight set matches its native layouts (Q4_K/Q5_K/Q6_K
        // ssm_out, Q4_K gate/up, Q4_K/Q6_K down). A mismatch is a hard error
        // here: the fused path is the only NPU route for the recurrent layers,
        // there is no scalar fused fallback.
        if ((p.w_so->type != GGML_TYPE_Q4_K && p.w_so->type != GGML_TYPE_Q5_K && p.w_so->type != GGML_TYPE_Q6_K) ||
            p.w_gate->type != GGML_TYPE_Q4_K || p.w_up->type != GGML_TYPE_Q4_K ||
            (p.w_down->type != GGML_TYPE_Q4_K && p.w_down->type != GGML_TYPE_Q6_K)) {
            GGML_LOG_ERROR(
                "%s: fused layer %d: unsupported fused weight set "
                "(so=%s gate=%s up=%s down=%s)\n",
                "ggml-xdna", p.il, ggml_type_name(p.w_so->type), ggml_type_name(p.w_gate->type),
                ggml_type_name(p.w_up->type), ggml_type_name(p.w_down->type));
            return false;
        }
        // The gated stage writes its activation in the layout the artifact was
        // built with (-DGATED_FMT), and the model's ssm_out decides which one
        // it needs: Q4_K is the 4-bit form, Q5_K/Q6_K the 8-bit. Nothing can
        // adapt at run time - the kernel's layout is fixed - so a model that
        // needs the other form stops here, before any buffer is sized or any
        // dispatch runs, instead of writing past the end of a buffer sized for
        // this one.
        const int want_fmt = p.w_so->type == GGML_TYPE_Q4_K ? 0 : 1;
        if (want_fmt != xdna_rec_pack::GATED_FMT) {
            GGML_LOG_ERROR(
                "%s: fused layer %d: ssm_out is %s, which needs the "
                "%d-bit activation layout, but the kernels were built "
                "with GATED_FMT=%d; rebuild them and the backend with "
                "-DGGML_XDNA_GATED_FMT=%d\n",
                "ggml-xdna", p.il, ggml_type_name(p.w_so->type), want_fmt == 0 ? 4 : 8, xdna_rec_pack::GATED_FMT,
                want_fmt);
            return false;
        }

        xdna_rec_pack_begin(s->host, ch, qkv, s->feed0);
        xdna_rec_geom                     g = xdna_rec_pack_geom();
        // the layer's buffers join the token's command (xdna_run_submit)
        std::unique_ptr<xdna_arena_scope> arena;
        if (create) {
            arena = std::make_unique<xdna_arena_scope>();
        }
        if (create) {
            const std::string & gx = xdna_fused_xclbin();
            if (gx.empty()) {
                return false;
            }
            // The gated out object's size follows the compiled layout (xdna-rec.h).
            g.out_bytes = (int64_t) xdna_rec_pack::OUTN;

            s->core = xdna_rec_core_create(ctx->device, g, gx.c_str());
            if (!s->core) {
                return false;
            }
            s->gv = xdna_rec_gemv_create(ctx->pool, p.w_so, p.w_gate, p.w_up, p.w_down);
            if (!s->gv) {
                GGML_LOG_ERROR(
                    "%s: fused layer %d: the GEMV route does not cover "
                    "this weight set\n",
                    "ggml-xdna", p.il);
                return false;
            }
            // One dispatch for the core and the projection that reads what its
            // gated stage writes. Both are already the same array configuration
            // and the same hardware context; this makes them one stream, so the
            // array pays its fixed per-phase cost once instead of twice.
            if (inproj) {
                xdna_rec_inproj ip;
                if (!xdna_rec_inproj_pack(p.n_qkv->src[0], p.n_z->src[0], ip) || ip.n_qkv != CH || ip.n_z != NVH * SV ||
                    !xdna_rec_core_set_inproj(s->core, ip.geom, ip.packed)) {
                    GGML_LOG_ERROR(
                        "%s: fused layer %d: the in-projection (%s, %s) "
                        "does not split into the core's inputs; "
                        "GGML_XDNA_INPROJ=0 runs it on its own\n",
                        "ggml-xdna", p.il, ggml_type_name(p.n_qkv->src[0]->type), ggml_type_name(p.n_z->src[0]->type));
                    return false;
                }
            }
            // The layer boundary on the prologue, when every piece of it is
            // at hand: the input's norm (the MUL the in-projection reads)
            // and the post-attention one.
            float               eps_a = 0.0f, eps_p = 0.0f;
            const ggml_tensor * xn = p.n_act;
            if (inproj && xn && xn->op == GGML_OP_MUL && xn->src[0] && xn->src[0]->op == GGML_OP_RMS_NORM &&
                xn->src[1] && xn->src[1]->type == GGML_TYPE_F32 && ggml_nelements(xn->src[1]) == XDNA_RES_D &&
                xdna_norm_eps(cgraph, p.w_post, &eps_p)) {
                memcpy(&eps_a, xn->src[0]->op_params, sizeof(float));
                if (!ctx->res) {
                    ctx->res = xdna_res_alloc(ctx->device);
                }
                if (ctx->res && xdna_rec_core_set_rows(s->core, ctx->res, (const float *) xn->src[1]->data, eps_a,
                                                       (const float *) p.w_post->data, eps_p)) {
                    // and the gates, when the graph has them as expected
                    std::vector<float> wa, wb, dt, aa;
                    if (xdna_gates_of(p, wa, wb, dt, aa)) {
                        (void) xdna_rec_core_set_gates(s->core, wa.data(), wb.data(), dt.data(), aa.data());
                    }
                }
            }
            // The prologue builds the layer's boundaries from the residual
            // rows and nothing else: there is no other way into the FFN.
            if (!s->core->rows) {
                GGML_LOG_ERROR(
                    "%s: fused layer %d: the layer boundary cannot go through the "
                    "residual rows (in-projection %s, input norm %s)\n",
                    "ggml-xdna", p.il, inproj ? "fused" : "off", xn ? ggml_op_name(xn->op) : "none");
                return false;
            }
            if (!xdna_rec_core_fuse_so(s->core, ctx->pool, xdna_rec_gemv_so(s->gv), s->gv->ffn) && inproj) {
                // Only the fused stream carries the in-projection.
                give_up("the fused stream does not carry the in-projection");
                return false;
            }
        }
        if (!xdna_rec_core_begin(s->core, s->feed0.data(), s->host.w_ssm_norm) || !xdna_rec_core_seed(s->core, st)) {
            give_up("the core could not begin or seed");
            return false;
        }
        s->seeded = true;
        s->x.resize((size_t) g.x_bytes);
        s->hattn.resize((size_t) g.d_out);
        s->hout.resize((size_t) g.d_out);
        s->hff.resize((size_t) g.d_out);
        s->aq.resize(KGATE);
    }

    const float * hres = !s->cap_hres.empty() ? s->cap_hres.data() : xdna_rec_tdata(p.n_hres, D_OUT);
    if (!hres) {
        GGML_LOG_ERROR("%s: fused layer %d: bad residual input\n", "ggml-xdna", p.il);
        return false;
    }

    xdna_prof::section_timer st_pre("layer: host pre (pack_x + ffn prep_raw)");
    xdna_rec_pack_x(gate, beta, s->x);
    const int t = s->core->token;

    // With the projection draining into the FFN's tiles, the host's half of
    // that activation - the residual and gamma - is written before the core
    // runs, not between the two dispatches. Both are known by now, and this is
    // what leaves nothing between them.
    const bool rows       = s->core->rows && xdna_rec_core_ffn_fused(s->core);
    const bool to_act_pre = s->gv->ffn && xdna_rec_core_so_to_act(s->core);
    if (rows && p.n_hres == ctx->res_lout) {
        // The layer before handed its output over in the rows.
    } else if (rows) {
        // The layer's input as the rows it reads, h = F + A: the host's copy.
        if (!xdna_res_release(ctx)) {
            return false;
        }
        float * r = (float *) ctx->res->data;
        std::memcpy(r + XDNA_RES_F / 4, hres, D_OUT * sizeof(float));
        std::memset(r + XDNA_RES_A / 4, 0, D_OUT * sizeof(float));
        if (!xdna_buffer_sync_to_device(ctx->res)) {
            return false;
        }
    } else if (to_act_pre && xdna_rec_core_ffn_fused(s->core) &&
               !xdna_gemv_pair_prep_raw(s->gv->ffn, hres, s->host.w_post_norm)) {
        give_up("the FFN activation could not be prepared");
        return false;
    }
    st_pre.stop();
    if (inproj && !rows && !xdna_rec_core_inproj_act(s->core, act)) {
        give_up("the in-projection activation could not be set");
        return false;
    }
    // This run drains the projection and the FFN into the pair's activation
    // buffer. Mark those ranges before it starts so the reads below can tell a
    // drain from the previous token's contents. Both flags, not just the one
    // guarding the prep above: a fuse_so that failed after setting so_to_act
    // leaves nothing to drain, and a mark nobody overwrites would spin the
    // read out.
    if (s->gv->ffn && xdna_rec_core_so_fused(s->core) &&
        xdna_rec_core_so_to_act(s->core)) {
        xdna_gemv_pair_mark_out(s->gv->ffn);
    }
    xrt::run * launched = nullptr;
    if (rows && xdna_queue_on()) {
        xdna_batch_open();
        launched = xdna_rec_core_launch(s->core, t);
    }
    if (launched) {
        ggml_backend_xdna_context::pending_run pr;
        pr.run = launched;
        ctx->pending.push_back(std::move(pr));
    } else if (!xdna_rec_core_run(s->core, t, t == 0 ? nullptr : qkv, s->x.data(), z, s->aq.data(), &s->d_a)) {
        give_up("the core dispatch failed");
        return false;
    }
    xdna_prof::section_timer st_post("layer: host post (ffn out + acc + adds)");
    const bool               so_in_core = xdna_rec_core_so_fused(s->core);
    // When the projection drained into the FFN's activation tiles there is
    // nothing to collect: its result is already where the next dispatch reads
    // it, and the host only needs it again for the layer's last residual add,
    // which happens after both dispatches rather than between them.
    const bool               so_to_act  = so_in_core && xdna_rec_core_so_to_act(s->core);
    if (rows || so_to_act) {
        // Nothing to collect: the outputs stay in the rows (xdna_res_touch) or
        // are already where the next dispatch reads them.
    } else if (so_in_core ? !xdna_rec_gemv_so_collect(s->gv, xdna_rec_core_out(s->core), xdna_rec_core_out_off(s->core),
                                                      xdna_rec_core_act(s->core), hres, s->hattn.data()) :
                            !xdna_rec_gemv_so_run(s->gv, xdna_rec_core_act(s->core), s->aq.data(), s->d_a, hres,
                                                  s->hattn.data())) {
        give_up("the ssm-out projection failed");
        return false;
    }
    if (rows) {
        // Both outputs came back as rows above.
    } else if (s->gv->ffn) {
        // The transition is the array's: the activation tiles carry the
        // projection's output, the residual and gamma as numbers and the
        // design's prologue tile turns them into codes. Nothing of the layer
        // is left on the host between its two dispatches. A null acc says the
        // projection already drained its own into the tiles.
        if (so_to_act) {
            std::vector<float> ffn_out(D_OUT);
            // Fused: the FFN already ran, in the core's own dispatch, and left
            // its result in the tail of the activation buffer. Otherwise it is
            // a dispatch of its own and its activation is prepared here.
            const bool         ffn_fused = xdna_rec_core_ffn_fused(s->core);
            if (ffn_fused) {
                if (!xdna_rec_gemv_fused_results(s->gv, s->gv->acc.data(), ffn_out.data())) {
                    give_up("the fused FFN results could not be read");
                    return false;
                }
            } else {
                if (!xdna_gemv_pair_prep_raw(s->gv->ffn, hres, s->host.w_post_norm)) {
                    give_up("the FFN activation could not be prepared");
                    return false;
                }
                if (!xdna_gemv_pair_dispatch(s->gv->ffn, ffn_out.data())) {
                    give_up("the FFN dispatch failed");
                    return false;
                }
            }
            if (!ffn_fused && !xdna_rec_gemv_acc_from_tiles(s->gv, s->gv->acc.data())) {
                give_up("the FFN accumulator could not be read");
                return false;
            }
            for (int i = 0; i < D_OUT; i++) {
                s->hattn[i] = hres[i] + s->gv->acc[i];
                s->hout[i]  = s->hattn[i] + ffn_out[i];
            }
        } else if (!xdna_rec_gemv_ffn_run_raw(s->gv, s->gv->acc.data(), hres, s->host.w_post_norm, s->hattn.data(),
                                              s->hout.data())) {
            give_up("the FFN dispatch failed");
            return false;
        }
    } else {
        xdna_rec_rms_norm(s->hattn.data(), s->host.w_post_norm, D_OUT, s->eps_post, s->hff.data());
        if (!xdna_rec_gemv_ffn_run(s->gv, s->hff.data(), s->hattn.data(), s->hout.data())) {
            give_up("the FFN dispatch failed");
            return false;
        }
    }

    st_post.stop();
    // Write the fused outputs into the layer add tensors every downstream op
    // reads (l_out feeds the next layer/MHA, attn_residual its own FFN path).
    if (rows) {
        ctx->res_lout  = p.n_lout;
        ctx->res_resid = p.n_resid;
        ctx->res_dirty = true;
    } else {
        std::memcpy(p.n_resid->data, s->hattn.data(), D_OUT * sizeof(float));
        std::memcpy(p.n_lout->data, s->hout.data(), D_OUT * sizeof(float));
    }

    // Mark the whole conv/gdn/ffn subgraph consumed (fire .. l_out).
    for (int j = p.i_fire; j <= p.i_lout && j < cgraph->n_nodes; j++) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (!ggml_xdna_is_view_op(n->op) && n->op != GGML_OP_NONE) {
            consumed.insert(n);
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Kernel warm / context registration.
// ---------------------------------------------------------------------------

// Eagerly register the xclbins this backend will actually run, so their
// hw_contexts exist at process start while the NPU is idle. Lazy registration
// in the middle of active NPU work fails DRM_IOCTL_AMDXDNA_CREATE_HWCTX with
// EINVAL, and the device caps concurrent contexts at 16.
//
// The list is exactly what the dispatch paths of the active mode can submit.
// Warming an artifact no dispatch reaches costs a hardware context that the
// 16-context budget then cannot spend on one that is used.
static void xdna_kernels_warm(xdna_kernel_pool * pool, const xdna_ops * ops, std::vector<struct xdna_kernel *> & keep) {
    if (!pool || !pool->device) {
        return;
    }
    const auto warm_stem = [&](const std::string & stem) {
        if (stem.empty()) {
            return;
        }
        if (struct xdna_kernel * kern = xdna_kernel_find(pool->device, stem.c_str())) {
            keep.push_back(kern);
        }
    };
    const auto warm_names = [&](const std::string & prefix) {
        for (const std::string & name : pool->names) {
            if (name.compare(0, prefix.size(), prefix) == 0) {
                warm_stem(name);
            }
        }
    };

    if (xdna_rec_active()) {
        // Fused decode: one design carries both the core and the decode GEMV
        // its projections run on, so warming it covers the split the mode
        // uses. The prefill kernels are warmed alongside it - a prompt chunk
        // still dispatches those.
        warm_stem(std::filesystem::path(xdna_fused_xclbin()).stem().string());
        warm_names("conv_prefill");
        warm_names("fa_prefill");
        return;
    }
    // Per-op decode: the standalone GEMV split and the GEMM variants xdna-ops
    // resolves.
    warm_stem(xdna_gemv_stem(XDNA_GEMV_SPLIT_DEFAULT));
    warm_stem(ops->gemm_xclbin_decode);
    warm_names("gemm_int8_int32");
    warm_names("gemm_bf16_f32");
}

static ggml_backend_xdna_context * ggml_xdna_device_context(void) {
    static ggml_backend_xdna_context ctx;
    static std::once_flag            once;

    std::call_once(once, [&]() {
        ctx.device = xdna_device_open();
        if (ctx.device) {
            ctx.pool         = new xdna_kernel_pool;
            ctx.pool->device = ctx.device;
            xdna_kernel_pool_scan(ctx.pool);
            xdna_ops_init(&ctx.ops, ctx.pool);
            if (ctx.ops.gemm_xclbin_decode.empty() && ctx.ops.pref_xclbin.empty()) {
                GGML_LOG_WARN("%s: no GEMM kernels found (build with GGML_XDNA=ON)\n", "ggml-xdna");
            }
            // Register the kernels this backend will run up front, while the
            // NPU is idle (see xdna_kernels_warm).
            xdna_kernels_warm(ctx.pool, &ctx.ops, ctx.warm_kerns);
        }
    });

    return &ctx;
}

// backend interface

static const char * ggml_backend_xdna_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return "XDNA";
}

static void xdna_rec_session_free(xdna_rec_session * s) {
    if (!s) {
        return;
    }
    // The core's fused projection run binds the GEMV's buffers: core first.
    xdna_rec_core_free(s->core);
    xdna_rec_gemv_free(s->gv);
    delete s;
}

// Release what was built from a model: the fused-layer sessions (their device
// state, their packed weights, and host pointers into the model's tensors),
// the packed weights of the per-op kernels and the prefill attention's K/V
// copies. The kernel pool and the other prefill runners hold no model data
// and stay.
static void xdna_release_model_state(ggml_backend_xdna_context * ctx) {
    const size_t n = ctx->rec.size();
    xdna_pending_wait(ctx);
    for (auto & kv : ctx->rec) {
        xdna_rec_session_free(kv.second);
    }
    ctx->rec.clear();
    for (auto & kv : ctx->tails) {
        xdna_rec_tail_free(kv.second);
    }
    ctx->tails.clear();
    for (auto & kv : ctx->att_layers) {
        xdna_att_layer_free(kv.second);
    }
    ctx->att_layers.clear();
    xdna_att_free(ctx->att);
    ctx->att = nullptr;
    ctx->att_flushed.clear();
    xdna_head_free(ctx->head);
    ctx->head       = nullptr;
    ctx->head_tried = false;
    xdna_buffer_free(ctx->res);
    ctx->res            = nullptr;
    ctx->res_lout       = nullptr;
    ctx->res_resid      = nullptr;
    ctx->res_dirty      = false;
    ctx->pending_failed = false;
    xdna_ops_release_weights(&ctx->ops);
    xdna_attn_mm_release();
    if (!xdna_arena_release()) {
        GGML_LOG_WARN("%s: the decode arena is still in use; kept\n", "ggml-xdna");
    }
    GGML_LOG_DEBUG("%s: released %zu fused-layer sessions and the packed weights\n", "ggml-xdna", n);
}

static void ggml_backend_xdna_free(ggml_backend_t backend) {
    // The context is a process-wide singleton; it is not freed here, but what
    // it built from the model goes with the last backend.
    ggml_backend_xdna_context * ctx = (ggml_backend_xdna_context *) backend->context;
    {
        std::lock_guard<std::mutex> lock(ctx->life_mutex);
        if (--ctx->n_backends == 0) {
            std::lock_guard<std::mutex> compute(ctx->compute_mutex);
            xdna_release_model_state(ctx);
        }
    }
    delete backend;
}

// CONCAT nodes the conv prefill reads through instead of building (see
// xdna_conv_prefill_direct_add): the conv input is the cached tokens with the
// chunk's projection, and materialising it costs an element-at-a-time copy of
// the whole thing.
static void xdna_concat_tail_plan(ggml_backend_xdna_context *                                     ctx,
                                  struct ggml_cgraph *                                            cgraph,
                                  std::unordered_map<const ggml_tensor *, struct ggml_tensor *> & out) {
    // No do/while wrapper: the `continue` has to leave the node loop.
#define CF_NO(why)        \
    {                     \
        GGML_UNUSED(why); \
        continue;         \
    }
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * n = cgraph->nodes[i];
        if (n->op != GGML_OP_CONCAT || n->type != GGML_TYPE_F32) {
            continue;
        }
        if (ggml_get_op_params_i32(n, 0) != 0 || n->ne[2] != 1 || n->ne[3] != 1) {
            CF_NO("not a dim-0 single-sequence concat");
        }
        if ((n->flags & GGML_TENSOR_FLAG_OUTPUT) || !ggml_is_contiguous(n)) {
            CF_NO("graph output or not contiguous");
        }
        const ggml_tensor * a = n->src[0];
        const ggml_tensor * b = n->src[1];
        if (!a || !b || a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) {
            CF_NO("sources are not both f32");
        }
        // Find the conv this concat feeds, and run it here rather than where
        // ggml put it: `b` is the projection, the concat is its only consumer,
        // so ggml-alloc hands its bytes to the very next node. Reading it
        // later is not an option and copying it is 12 MB a layer.
        //
        // Running the conv early is safe as long as nothing scheduled in
        // between writes over the conv's own destination, and as long as that
        // destination does not overlap the projection it is still reading.
        struct ggml_tensor * conv = nullptr;
        int                  j    = -1;
        for (int u = i + 1; u < cgraph->n_nodes; u++) {
            if (cgraph->nodes[u]->op == GGML_OP_SSM_CONV && cgraph->nodes[u]->src[0] == n) {
                conv = cgraph->nodes[u];
                j    = u;
                break;
            }
        }
        if (!conv) {
            CF_NO("no conv reads it");
        }
        if (!xdna_ops_supported(&ctx->ops, conv) || !xdna_conv_prefill_supported(conv)) {
            CF_NO("the array does not claim the conv");
        }
        const ggml_tensor * bsrc = b->view_src ? b->view_src : b;
        const char * const  blo  = (const char *) bsrc->data;
        const char * const  dlo  = (const char *) conv->data;
        if (!blo || !dlo) {
            CF_NO("unallocated source or destination");
        }
        if (dlo < blo + ggml_nbytes(bsrc) && blo < dlo + ggml_nbytes(conv)) {
            CF_NO("the conv writes over the projection it reads");
        }
        bool clobbered = false;
        for (int u = i + 1; u < j && !clobbered; u++) {
            struct ggml_tensor * m = cgraph->nodes[u];
            if (ggml_xdna_is_view_op(m->op) || m->data == nullptr) {
                continue;  // writes nothing of its own
            }
            const char * const mlo = (const char *) m->data;
            clobbered              = mlo < dlo + ggml_nbytes(conv) && dlo < mlo + ggml_nbytes(m);
        }
        if (clobbered) {
            CF_NO("the conv result is overwritten before its node is reached");
        }
        const int64_t keep     = a->ne[0];  // conv history rows
        const size_t  tail_off = (size_t) (n->ne[0] - keep) * n->nb[0];
        bool          ok       = keep > 0 && n->ne[0] > keep;
        for (int u = 0; u < cgraph->n_nodes && ok; u++) {
            struct ggml_tensor * c = cgraph->nodes[u];
            if (c == n || c == conv) {
                continue;
            }
            bool reads = c->view_src == n;
            for (int k = 0; k < GGML_MAX_SRC && !reads; k++) {
                reads = c->src[k] == n;
            }
            if (!reads) {
                continue;
            }
            // Only a view of the tail rows may read it; whatever reads that
            // view can see no further than the view does.
            const size_t off = (const char *) c->data - (const char *) n->data;
            ok               = ggml_xdna_is_view_op(c->op) && c->data != nullptr && c->ne[0] <= keep && off >= tail_off;
        }
        if (ok) {
            out[n] = conv;
        }
    }
#undef CF_NO
}

// Write the rows of a concat that xdna_concat_tail_plan left to be written:
// the last `src[0]->ne[0]` rows of ne0, which the graph views as the next
// conv state. Everything else in the destination is dead.
//
// Also snapshot the conv history for the conv to read. It cannot read src[0]
// where it sits: that is the conv state cache, and the CPY between this node
// and the conv overwrites it with the tail written just below. The snapshot
// is `src[0]->ne[0]` rows - three - so it costs nothing.
static void xdna_concat_tail_fill(struct ggml_tensor * n, std::vector<float> & hist) {
    const ggml_tensor * a  = n->src[0];
    const ggml_tensor * b  = n->src[1];
    const int64_t       t0 = n->ne[0] - a->ne[0];
    hist.resize((size_t) a->ne[0] * n->ne[1]);
    for (int64_t i0 = 0; i0 < a->ne[0]; i0++) {
        for (int64_t i1 = 0; i1 < n->ne[1]; i1++) {
            hist[(size_t) i0 * n->ne[1] + i1] =
                *(const float *) ((const char *) a->data + i0 * a->nb[0] + i1 * a->nb[1]);
        }
    }
    xdna_conv_prefill_direct_add(n, hist.data(), a->ne[0]);
    for (int64_t i1 = 0; i1 < n->ne[1]; i1++) {
        for (int64_t i0 = t0; i0 < n->ne[0]; i0++) {
            const ggml_tensor * s  = i0 < a->ne[0] ? a : b;
            const int64_t       si = i0 < a->ne[0] ? i0 : i0 - a->ne[0];
            const float         v  = *(const float *) ((const char *) s->data + si * s->nb[0] + i1 * s->nb[1]);
            *(float *) ((char *) n->data + i0 * n->nb[0] + i1 * n->nb[1]) = v;
        }
    }
}

// With flash attention off llama builds attention from MUL_MAT and SOFT_MAX,
// and every attention route here - the decode pool, the attention layer, the
// prefill attention - takes FLASH_ATTN_EXT: the attention then runs on the
// host, the decode at about half the speed (50 against 29 t/s on the 0.8B)
// with ten times the host work. Nothing fails, so say it, once. Graph
// computes take the context's lock, so the flag needs none of its own.
static void xdna_warn_host_attention(const struct ggml_cgraph * cgraph) {
    static bool said = false;
    if (said) {
        return;
    }
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const struct ggml_tensor * n = cgraph->nodes[i];
        if (n->op == GGML_OP_SOFT_MAX && strncmp(n->name, "kq_soft_max", 11) == 0) {
            GGML_LOG_WARN("%s: flash attention is off, so attention runs on the host; "
                          "the NPU takes it with flash attention on (-fa on or auto)\n",
                          "ggml-xdna");
            said = true;
            return;
        }
    }
}

static enum ggml_status xdna_graph_compute_impl(ggml_backend_xdna_context * ctx, struct ggml_cgraph * cgraph) {
    if (!ctx->device) {
        return GGML_STATUS_SUCCESS;
    }

    xdna_warn_host_attention(cgraph);

    // Size the host-fallback pool for this chunk (xdna_glue_threads).
    xdna_glue_set_n_tokens(cgraph);
    if (g_glue_n_tokens > 1) {
        // a prefill rewrites cache rows: flush them all again before the pool reads them
        ctx->att_flushed.clear();
        // and the prefill GEMM's activation layouts were the last graph's
        xdna_pgemm_graph_begin();
        // as is the attention mask the prefill attention checked
        xdna_attn_mm_graph_begin();
    }

    // Whole-call timer (GGML_XDNA_PROF=1): must be constructed right after
    // g_glue_n_tokens is known, before anything below runs, because every
    // fused_layer/gemv/glue timer nested in this call reads the decode/other
    // and steady/warm-up classification it sets. It wraps the entire rest of
    // this function via RAII (destructor fires on every return path).
    xdna_prof::call_timer xdna_prof_call(g_glue_n_tokens == 1);

    // Per-op NPU kernels are suppressed while the fused layer path is active
    // (default when the fused xclbins are present): the fused run owns the
    // recurrent layers and everything else falls back to the CPU glue.
    ctx->ops.isolation  = xdna_rec_active();
    // The decode projections outside the fused layers run on the array, on the
    // context the fused layers leave resident. Their results are read back
    // against a mark (xdna_buffer_mark): the array can report completion
    // before its last writes are visible to the host, and a read taken in that
    // window used to hand back the previous token's contents, which made the
    // whole decode irreproducible. It costs throughput against leaving those
    // projections on the host (27.4 against 37.0 t/s on the 0.8B) and buys the
    // host CPU the route exists for. GGML_XDNA_GEMV_GROUP=0 keeps them there.
    ctx->ops.fused_gemv = ctx->ops.isolation && !xdna_fused_xclbin().empty();
    if (xdna_env_int("GGML_XDNA_GEMV_GROUP", 1) == 0) {
        ctx->ops.fused_gemv = false;
    }

    // Fused decode linear layers: scan this chunk for complete recurrent
    // layers of a single-token decode; each fires once in the dispatch loop
    // below and replaces its conv/gdn/ffn subgraph with one persistent device
    // run.
    xdna_rec_follow_cache(ctx, cgraph);
    std::vector<xdna_rec_plan> rec_plans;
    if (xdna_rec_active()) {
        xdna_rec_scan(ctx, cgraph, rec_plans);
    }

    // Prefill conv inputs whose concat the array reads through instead of
    // building.
    std::unordered_map<const ggml_tensor *, struct ggml_tensor *> concat_tail;
    // Owns the conv-history snapshots for the length of the graph; deque so
    // the pointers handed to the conv runner survive later pushes.
    std::deque<std::vector<float>>                                concat_hist;
    xdna_conv_prefill_direct_reset();
    xdna_concat_tail_plan(ctx, cgraph, concat_tail);

    // Input snapshot points of the fused plans: when the loop reaches one of
    // these producer nodes it runs the node (if not already consumed by an
    // earlier fused run) and copies the output into the layer session, so the
    // fused fire later reads valid data instead of llama's recycled buffers.
    // kind: 0 = qkv, 1 = conv seed, 2 = ssm seed, 3 = residual h, 4 = the
    // in-projection's input.
    struct rec_capture {
        int             i;
        xdna_rec_plan * p;
        int             kind;
    };

    std::vector<rec_capture> rec_caps;
    for (auto & pl : rec_plans) {
        if (pl.i_cap_qkv >= 0) {
            rec_caps.push_back({ pl.i_cap_qkv, &pl, 0 });
        }
        if (pl.i_cap_conv >= 0) {
            rec_caps.push_back({ pl.i_cap_conv, &pl, 1 });
        }
        if (pl.i_cap_ss >= 0) {
            rec_caps.push_back({ pl.i_cap_ss, &pl, 2 });
        }
        if (pl.i_cap_hres >= 0) {
            rec_caps.push_back({ pl.i_cap_hres, &pl, 3 });
        }
        if (pl.i_cap_act >= 0) {
            rec_caps.push_back({ pl.i_cap_act, &pl, 4 });
        }
    }

    std::unordered_set<const ggml_tensor *> consumed;

    // The fused run recomputes the whole layer body, but it fires only after
    // the last of its inputs, and ggml's order puts part of the body - the
    // recurrence itself among it - before that. Those nodes would run on the
    // host and then be overwritten, so they are marked consumed up front.
    // Everything the fused run actually reads is spared: its six inputs and
    // whatever produces them.
    for (auto & pl : rec_plans) {
        if (pl.i_conv < 0 || pl.i_fire <= pl.i_conv) {
            continue;
        }
        // Once the session carries the state on the device, the reads of
        // llama's recurrent cache only ever fed its seed, and the writes back
        // into it feed nothing this graph reads: from the layer's first node
        // on, everything the fire does not read is the fire's.
        const auto it      = ctx->rec.find(std::make_pair(pl.il, (int64_t) 0));
        const bool seeded  = it != ctx->rec.end() && it->second->seeded && it->second->row == xdna_rec_seq_row(pl);
        int        i_first = pl.i_conv;
        if (seeded && pl.n_act) {
            for (int j = 0; j < pl.i_conv; j++) {
                if (cgraph->nodes[j] == pl.n_act->src[0] || cgraph->nodes[j] == pl.n_act) {
                    i_first = j;
                    break;
                }
            }
        }
        std::unordered_set<const ggml_tensor *> needed;
        std::vector<const ggml_tensor *>        stack = {
            pl.n_act ? pl.n_act : pl.n_qkv,
            pl.n_act ? pl.n_act : pl.n_z,
            pl.n_gate,
            pl.n_beta,
        };
        // With the boundary and the gates on the prologue the fire reads none
        // of them: the input's norm and the gates' projections are its own.
        if (seeded && it->second->core && it->second->core->rows && it->second->core->gates) {
            stack.clear();
        }
        if (!seeded) {
            stack.push_back(pl.n_convst);
            stack.push_back(pl.n_sstate);
        }
        // With the in-projection in the stream its two MUL_MATs are the
        // fire's, wherever ggml put them.
        if (pl.n_act) {
            consumed.insert(pl.n_qkv);
            consumed.insert(pl.n_z);
        }
        while (!stack.empty()) {
            const ggml_tensor * t = stack.back();
            stack.pop_back();
            if (!t || !needed.insert(t).second) {
                continue;
            }
            for (int k = 0; k < GGML_MAX_SRC; k++) {
                if (t->src[k]) {
                    stack.push_back(t->src[k]);
                }
            }
        }
        for (int j = i_first; j < pl.i_fire; j++) {
            struct ggml_tensor * n = cgraph->nodes[j];
            if (!ggml_xdna_is_view_op(n->op) && n->op != GGML_OP_NONE && needed.count(n) == 0) {
                consumed.insert(n);
            }
        }
    }

    // The scheduler routes ops accepted by supports_op plus the view ops that
    // alias their data; views are no-ops here. Kernels are submitted (started)
    // per op and finalized once at the end, so all runs of the graph overlap.
    // Consecutive glue (host) ops accumulate into one batch and flush as a
    // single CPU cgraph compute before the next NPU op or the final finalize.
    // Decide which projections share a dispatch before running anything.
    // Everything the fused layers own: their bodies, already in `consumed`,
    // plus the tail they mark consumed only when they fire and the inputs the
    // dispatch loop has to run on the host so it can snapshot them.
    std::unordered_set<const ggml_tensor *> gemv_skip = consumed;
    for (const auto & pl : rec_plans) {
        for (int j = std::max(0, pl.i_fire); j <= pl.i_lout && j < cgraph->n_nodes; j++) {
            gemv_skip.insert(cgraph->nodes[j]);
        }
    }
    // The snapshot producers are deliberately *not* skipped: a recurrent
    // layer's in_proj is the biggest projection in the model and the only
    // reason it was on the host is that the fused run reads its output. The
    // dispatch loop runs it through whatever route claims it and snapshots
    // the result either way.
    std::vector<xdna_tail_plan> tail_plans;
    if (xdna_tail_on() && xdna_rec_active()) {
        xdna_tail_scan(cgraph, tail_plans);
        for (const auto & tp : tail_plans) {
            for (int j = tp.i_fire; j <= tp.i_lout && j < cgraph->n_nodes; j++) {
                gemv_skip.insert(cgraph->nodes[j]);
            }
        }
    }
    std::vector<xdna_attl_plan> attl_plans;
    if (xdna_attl_on() && !tail_plans.empty() && g_glue_n_tokens == 1) {
        xdna_attl_scan(cgraph, tail_plans, attl_plans);
        // The input's norm is the layer's own when it will run as one
        // dispatch: left to the host it would read l_out, which may be only
        // in the rows.
        for (auto & ap : attl_plans) {
            if (xdna_attl_ready(ctx, ap)) {
                consumed.insert(ap.n_q->src[1]);
                consumed.insert(ap.n_q->src[1]->src[0]);
                ap.pre = true;
            }
        }
    }
    xdna_ops_plan_gemv(&ctx->ops, cgraph, &gemv_skip);

    // The head's norm is the array's when the head takes its input from the
    // rows (every token after the first, when the last layer is a fused one).
    bool head_pre = false;
    if (xdna_head_on() && g_glue_n_tokens == 1 && xdna_head_rows(ctx->head) && xdna_queue_on()) {
        for (int i = cgraph->n_nodes - 1; i >= 0 && !head_pre; i--) {
            xdna_head_chain hc;
            if (xdna_head_node(cgraph->nodes[i]) && xdna_head_chain_of(cgraph->nodes[i], hc)) {
                for (ggml_tensor * t : { hc.rms, hc.mul, hc.rows }) {
                    if (t) {
                        consumed.insert(t);
                    }
                }
                head_pre = true;
            }
        }
    }

    // A prompt's FFN activation on the array (xdna_pgemm_glu_supported): the
    // SWIGLU of the gate and up MUL_MATs runs as one prefill GEMM at the GLU,
    // the cores applying it, and the two MUL_MATs are skipped - which needs
    // their outputs to have no other reader.
    // And when the SWIGLU's one reader is the next projection on the array
    // (xdna_pgemm_glu_fused_supported), the cores write it as that
    // projection's A directly (glu_to_a).
    std::unordered_set<const struct ggml_tensor *> glu_fused, glu_to_a;
    if (g_glue_n_tokens > 1) {
        std::unordered_map<const struct ggml_tensor *, int>                        uses;
        std::unordered_map<const struct ggml_tensor *, const struct ggml_tensor *> reader;
        for (int i = 0; i < cgraph->n_nodes; i++) {
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                if (cgraph->nodes[i]->src[j]) {
                    uses[cgraph->nodes[i]->src[j]]++;
                    reader[cgraph->nodes[i]->src[j]] = cgraph->nodes[i];
                }
            }
        }
        for (int i = 0; i < cgraph->n_nodes; i++) {
            struct ggml_tensor * node = cgraph->nodes[i];
            if (!xdna_pgemm_glu_supported(node)) {
                continue;
            }
            struct ggml_tensor * gate = node->src[0];
            struct ggml_tensor * up   = node->src[1];
            if (uses[gate] != 1 || uses[up] != 1 || (gate->flags & GGML_TENSOR_FLAG_OUTPUT) ||
                (up->flags & GGML_TENSOR_FLAG_OUTPUT) || consumed.count(gate) || consumed.count(up)) {
                continue;
            }
            consumed.insert(gate);
            consumed.insert(up);
            glu_fused.insert(node);
            if (uses[node] == 1 && !(node->flags & GGML_TENSOR_FLAG_OUTPUT) &&
                xdna_pgemm_glu_fused_supported(node, reader[node])) {
                glu_to_a.insert(node);
            }
        }
    }

    // A prompt's recurrent conv input goes straight into the array's GDN
    // input (xdna_gdn_mm_prepare) at the concat, while the projection is
    // live: the conv, silu, norms and state copy are skipped - which needs
    // their outputs to have no reader but the GATED_DELTA_NET the array runs.
    struct gdn_in_plan {
        struct ggml_tensor * gdn = nullptr;
        xdna_gdn_conv_in     in;
        struct ggml_tensor * skip[5] = {};
        // on the array: the in-projection, run into the conv's buffer at its
        // own place in the graph (qkv_ok once it did)
        struct ggml_tensor * qkv     = nullptr;
        bool                 qkv_ok  = false;
    };
    std::unordered_map<const struct ggml_tensor *, gdn_in_plan>                gdn_in;
    std::unordered_map<const struct ggml_tensor *, const struct ggml_tensor *> qkv_of;  // qkv -> its concat
    if (g_glue_n_tokens >= 32 && xdna_env_int("GGML_XDNA_GDN_IN", 1) != 0) {
        std::unordered_map<const struct ggml_tensor *, std::vector<struct ggml_tensor *>> readers;
        for (int i = 0; i < cgraph->n_nodes; i++) {
            struct ggml_tensor * t = cgraph->nodes[i];
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                if (t->src[j]) {
                    readers[t->src[j]].push_back(t);
                }
            }
        }
        // the non-view nodes reading t or a view of it
        auto real_readers = [&](const struct ggml_tensor * t) {
            std::vector<struct ggml_tensor *> out, todo = { const_cast<struct ggml_tensor *>(t) };
            while (!todo.empty()) {
                struct ggml_tensor * u = todo.back();
                todo.pop_back();
                for (struct ggml_tensor * r : readers[u]) {
                    if (ggml_xdna_is_view_op(r->op)) {
                        todo.push_back(r);
                    } else {
                        out.push_back(r);
                    }
                }
            }
            return out;
        };
        auto f32 = [](const struct ggml_tensor * t) {
            return t && t->type == GGML_TYPE_F32;
        };
        for (int i = 0; i < cgraph->n_nodes; i++) {
            struct ggml_tensor * gd = cgraph->nodes[i];
            if (gd->op != GGML_OP_GATED_DELTA_NET || !xdna_gdn_mm_supported(gd)) {
                continue;
            }
            struct ggml_tensor *l2q = gd->src[0], *l2k = gd->src[1], *vv = gd->src[2];
            if (l2q->op != GGML_OP_L2_NORM || l2k->op != GGML_OP_L2_NORM || !vv->view_src || !l2q->src[0] ||
                !l2k->src[0]) {
                continue;
            }
            struct ggml_tensor * silu = vv->view_src;
            if (l2q->src[0]->view_src != silu || l2k->src[0]->view_src != silu || silu->op != GGML_OP_UNARY ||
                ggml_get_unary_op(silu) != GGML_UNARY_OP_SILU || !f32(silu) || !ggml_is_contiguous(silu)) {
                continue;
            }
            struct ggml_tensor * conv = silu->src[0];
            if (!conv || conv->op != GGML_OP_SSM_CONV || !f32(conv)) {
                continue;
            }
            struct ggml_tensor *       cat = conv->src[0];
            const struct ggml_tensor * w   = conv->src[1];
            if (!cat || cat->op != GGML_OP_CONCAT || ggml_get_op_params_i32(cat, 0) != 0 || !f32(w) ||
                !f32(cat->src[0]) || !f32(cat->src[1]) || cat->src[1]->nb[1] != sizeof(float) ||
                cat->src[0]->ne[0] != w->ne[0] - 1 || cat->src[0]->ne[2] != 1 || cat->src[1]->ne[2] != 1) {
                continue;
            }
            // every reader: the chain, the GDN, and one CPY of the concat
            struct ggml_tensor * cpy = nullptr;
            bool                 ok  = true;
            for (struct ggml_tensor * r : real_readers(cat)) {
                if (r == conv) {
                    continue;
                }
                if (r->op == GGML_OP_CPY && !cpy && ggml_is_contiguous(r) && f32(r) &&
                    ggml_nelements(r) == (w->ne[0] - 1) * w->ne[1]) {
                    cpy = r;
                } else {
                    ok = false;
                }
            }
            for (struct ggml_tensor * r : real_readers(conv)) {
                ok = ok && r == silu;
            }
            for (struct ggml_tensor * r : real_readers(silu)) {
                ok = ok && (r == l2q || r == l2k || r == gd);
            }
            for (struct ggml_tensor * r : real_readers(l2q)) {
                ok = ok && r == gd;
            }
            for (struct ggml_tensor * r : real_readers(l2k)) {
                ok = ok && r == gd;
            }
            if (!ok || !cpy) {
                continue;
            }
            gdn_in_plan p;
            p.gdn          = gd;
            p.in.x         = cat->src[1];
            p.in.state     = cat->src[0];
            p.in.w         = w;
            p.in.state_out = cpy;
            std::memcpy(&p.in.eps_q, l2q->op_params, sizeof(float));
            std::memcpy(&p.in.eps_k, l2k->op_params, sizeof(float));
            p.in.q_off = (long long) (l2q->src[0]->view_offs / sizeof(float));
            p.in.k_off = (long long) (l2k->src[0]->view_offs / sizeof(float));
            p.in.v_off = (long long) (vv->view_offs / sizeof(float));
            p.skip[0]  = conv;
            p.skip[1]  = silu;
            p.skip[2]  = l2q;
            p.skip[3]  = l2k;
            p.skip[4]  = cpy;
            for (struct ggml_tensor * t : p.skip) {
                consumed.insert(t);
            }
            // the projection read only through the concat can go straight
            // to the array's conv
            struct ggml_tensor * qkv = cat->src[1]->view_src;
            if (xdna_gdn_conv_supported() && qkv && qkv->op == GGML_OP_MUL_MAT && xdna_pgemm_supported(qkv) &&
                qkv->ne[0] == w->ne[1]) {
                bool only = true;
                for (struct ggml_tensor * r : real_readers(qkv)) {
                    only = only && r == cat;
                }
                if (only) {
                    p.qkv = qkv;
                    consumed.insert(qkv);
                    qkv_of[qkv] = cat;
                }
            }
            gdn_in[cat] = p;
        }
    }

    // Two small projections of one activation (a recurrent layer's alpha and
    // beta) as one prefill GEMM (xdna_pgemm_pair_supported). It runs before
    // the first projection on the array that reads the same activation (the
    // in-projection), so it adds no design change between the conv and the
    // GDN it sits between; the outputs wait until the graph reaches their
    // nodes, whose memory may hold a live tensor until then.
    struct mm_pair {
        struct ggml_tensor * a = nullptr;
        struct ggml_tensor * b = nullptr;
        std::vector<float>   out_a, out_b;
        bool                 ok = false;
    };

    std::vector<mm_pair>                                   pairs;
    std::unordered_map<const struct ggml_tensor *, size_t> pair_of, pair_at;
    if (g_glue_n_tokens > 1) {
        auto free_mm = [&](const struct ggml_tensor * t) {
            return t->op == GGML_OP_MUL_MAT && !consumed.count(t) && !qkv_of.count(t) && !pair_of.count(t) &&
                   !(t->flags & GGML_TENSOR_FLAG_OUTPUT);
        };
        for (int i = 0; i < cgraph->n_nodes; i++) {
            struct ggml_tensor * a = cgraph->nodes[i];
            if (!free_mm(a)) {
                continue;
            }
            for (int j = i + 1; j < cgraph->n_nodes; j++) {
                struct ggml_tensor * b = cgraph->nodes[j];
                if (!free_mm(b) || !xdna_pgemm_pair_supported(a, b)) {
                    continue;
                }
                struct ggml_tensor * at = a;
                for (int k = 0; k < i; k++) {
                    struct ggml_tensor * t = cgraph->nodes[k];
                    // (the in-projection counts as consumed: its conv is)
                    if (t->op == GGML_OP_MUL_MAT && t->src[1] == a->src[1] &&
                        (qkv_of.count(t) || (!consumed.count(t) && xdna_pgemm_supported(t)))) {
                        at = t;
                        break;
                    }
                }
                pair_of[a] = pair_of[b] = pair_at[at] = pairs.size();
                pairs.push_back({ a, b, {}, {}, false });
                break;
            }
        }
    }

    // An RMS_NORM times its weight whose only readers are prefill GEMMs
    // (xdna_pgemm_norm_supported): laid out as their A at the MUL, one pass
    // over the norm's input instead of the norm, the MUL and the GEMM's own
    // layout each going over the rows - host memory traffic is what the
    // package pays for through the GEMM. The norm is skipped and neither node
    // is written, which needs every reader to be such a MUL_MAT.
    std::unordered_set<const struct ggml_tensor *>                             norm_a;
    std::unordered_map<const struct ggml_tensor *, struct ggml_tensor *>       norm_add;
    std::unordered_map<const struct ggml_tensor *, int>                        node_at;  // node index in graph order
    std::unordered_map<const struct ggml_tensor *, const struct ggml_tensor *> gated_a;
    std::unordered_set<const struct ggml_tensor *>                             gate_a;
    g_norm_mul.clear();
    xdna_gdn_mm_keep_clear();
    xdna_attn_mm_keep_clear();
    if (g_glue_n_tokens > 1) {
        std::unordered_map<const struct ggml_tensor *, std::vector<const struct ggml_tensor *>> readers;
        for (int i = 0; i < cgraph->n_nodes; i++) {
            const struct ggml_tensor * t = cgraph->nodes[i];
            node_at[t] = i;
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                if (t->src[j] && (j == 0 || t->src[j] != t->src[j - 1])) {
                    readers[t->src[j]].push_back(t);
                }
            }
        }
        for (int i = 0; i < cgraph->n_nodes; i++) {
            struct ggml_tensor * m = cgraph->nodes[i];
            if (!xdna_pgemm_norm_supported(m) || consumed.count(m) || (m->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                continue;
            }
            struct ggml_tensor * r = m->src[0];
            if (consumed.count(r) || (r->flags & GGML_TENSOR_FLAG_OUTPUT) || readers[r].size() != 1) {
                continue;
            }
            const auto & rd = readers[m];
            bool         ok = !rd.empty();
            for (const struct ggml_tensor * t : rd) {
                ok = ok && t->op == GGML_OP_MUL_MAT && t->src[1] == m && t->src[0] != m && xdna_pgemm_supported(t);
            }
            if (ok) {
                norm_a.insert(m);
                consumed.insert(r);
                // the residual sum the norm reads, in the same pass
                struct ggml_tensor * ad    = r->src[0];
                bool                 ad_ok = xdna_pgemm_add_supported(ad, m) && !consumed.count(ad);
                // the sum is written at this MUL and not where the ADD sits: a
                // reader that runs earlier would see the ADD's buffer unwritten
                for (const struct ggml_tensor * t : readers[ad]) {
                    ad_ok = ad_ok && (t == r || node_at[t] > i);
                }
                if (ad_ok) {
                    norm_add[m] = ad;
                    consumed.insert(ad);
                }
            }
        }
        // a recurrent layer's output, (norm * weight) * silu(z), whose one
        // reader is a RESHAPE the output projection reads: into its A in one
        // pass (xdna_pgemm_run_gated) at the gating MUL
        for (int i = 0; i < cgraph->n_nodes; i++) {
            struct ggml_tensor * g = cgraph->nodes[i];
            if (g->op != GGML_OP_MUL || consumed.count(g) || (g->flags & GGML_TENSOR_FLAG_OUTPUT) ||
                readers[g].size() != 1) {
                continue;
            }
            const struct ggml_tensor * v = readers[g][0];
            if (!xdna_pgemm_gated_supported(g, v) || (v->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                continue;
            }
            struct ggml_tensor * m  = g->src[0];
            struct ggml_tensor * sl = g->src[1];
            struct ggml_tensor * r  = m->src[0];
            bool                 ok = readers[r].size() == 1 && readers[m].size() == 1 && readers[sl].size() == 1 &&
                      !readers[v].empty() && !consumed.count(r) && !consumed.count(m) && !consumed.count(sl);
            for (const struct ggml_tensor * t :
                 { (const struct ggml_tensor *) r, (const struct ggml_tensor *) m, (const struct ggml_tensor *) sl }) {
                ok = ok && !(t->flags & GGML_TENSOR_FLAG_OUTPUT);
            }
            for (const struct ggml_tensor * t : readers[v]) {
                ok = ok && t->op == GGML_OP_MUL_MAT && t->src[1] == v && xdna_pgemm_supported(t);
            }
            if (ok) {
                gated_a[g] = v;
                consumed.insert(r);
                consumed.insert(m);
                consumed.insert(sl);
                // the GDN output the norm reads: left where the array wrote
                // it when nothing else reads its rows (xdna_gdn_mm_keep)
                const struct ggml_tensor * x      = r->src[0];
                const struct ggml_tensor * gdn    = x->view_src;
                const int64_t              attn_b = (int64_t) 128 * 16 * g->ne[2] * (int64_t) sizeof(float);
                bool keep = x->op == GGML_OP_VIEW && gdn && gdn->op == GGML_OP_GATED_DELTA_NET && x->view_offs == 0 &&
                            x->ne[0] == 128 && x->ne[1] == 16 && x->nb[1] == 128 * sizeof(float) &&
                            x->nb[2] == (size_t) 128 * 16 * sizeof(float) && readers[x].size() == 1 &&
                            xdna_gdn_mm_supported(gdn) && !(gdn->flags & GGML_TENSOR_FLAG_OUTPUT);
                for (const struct ggml_tensor * u : readers[gdn]) {
                    keep = keep && u->op == GGML_OP_VIEW && (u == x || (int64_t) u->view_offs >= attn_b);
                }
                if (keep) {
                    xdna_gdn_mm_keep(gdn);
                }
            }
        }
        // an attention output times its gate whose only readers are
        // prefill GEMMs: into their A in one pass (xdna_pgemm_run_gate) at
        // the MUL, the SIGMOID and its CONT skipped
        for (int i = 0; i < cgraph->n_nodes; i++) {
            struct ggml_tensor * g = cgraph->nodes[i];
            if (g->op != GGML_OP_MUL || consumed.count(g) || gated_a.count(g) || (g->flags & GGML_TENSOR_FLAG_OUTPUT) ||
                !xdna_pgemm_gate_supported(g)) {
                continue;
            }
            struct ggml_tensor * sg = g->src[1];
            struct ggml_tensor * c  = sg->src[0];
            bool ok = readers[sg].size() == 1 && !consumed.count(sg) && !(sg->flags & GGML_TENSOR_FLAG_OUTPUT) &&
                      !readers[g].empty();
            const bool skip_c = c->op == GGML_OP_CONT;
            if (skip_c) {
                ok = ok && readers[c].size() == 1 && !consumed.count(c) && !(c->flags & GGML_TENSOR_FLAG_OUTPUT);
            }
            for (const struct ggml_tensor * t : readers[g]) {
                ok = ok && t->op == GGML_OP_MUL_MAT && t->src[1] == g && xdna_pgemm_supported(t);
            }
            if (ok) {
                gate_a.insert(g);
                consumed.insert(sg);
                if (skip_c) {
                    consumed.insert(c);
                }
                // the attention output the gate reads: left where the array
                // wrote it when nothing else reads it (xdna_attn_mm_keep)
                const struct ggml_tensor * a  = g->src[0];
                const struct ggml_tensor * fa = a->view_src;
                if (a->op == GGML_OP_RESHAPE && fa && fa->op == GGML_OP_FLASH_ATTN_EXT && a->src[0] == fa &&
                    fa->ne[0] == 256 && fa->ne[1] == 8 && ggml_is_contiguous(fa) && a->ne[0] == 2048 &&
                    readers[fa].size() == 1 && readers[a].size() == 1 && !(fa->flags & GGML_TENSOR_FLAG_OUTPUT) &&
                    xdna_attn_mm_supported(fa)) {
                    xdna_attn_mm_keep(fa);
                }
            }
        }
        // the rest of the norms: the MUL by the weight in the same pass
        // (xdna_norm_mul_fast, in the glue batch)
        for (int i = 0; i < cgraph->n_nodes; i++) {
            const struct ggml_tensor * r = cgraph->nodes[i];
            if (r->op != GGML_OP_RMS_NORM || consumed.count(r) || (r->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                continue;
            }
            const auto & rd = readers[r];
            if (rd.size() == 1 && rd[0]->op == GGML_OP_MUL && rd[0]->src[0] == r && !consumed.count(rd[0])) {
                g_norm_mul.insert(r);
            }
        }
    }

    std::vector<struct ggml_tensor *> glue_batch;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        ctx->cur_node             = node;
        ctx->cur_index            = i;
        // Fused-layer input snapshot: right when a producer node runs, copy its
        // output into the layer session before llama recycles the buffer. A
        // consumed node was already produced (e.g. the fused write of a
        // previous layer) and is only copied. Views need no snapshot.
        for (auto & c : rec_caps) {
            if (c.i != i || ggml_xdna_is_view_op(node->op)) {
                continue;
            }
            xdna_rec_session * s = xdna_rec_session_get(ctx, c.p->il);
            if (s && consumed.count(node) == 0) {
                if (!xdna_res_touch(ctx, node)) {
                    return GGML_STATUS_FAILED;
                }
                // A snapshot input the GEMV claimed runs there; the copy below
                // reads its output the same way either way. One the host runs
                // closes the pending glue batch instead of running after it as
                // a batch of its own.
                const bool on_npu  = xdna_ops_supported(&ctx->ops, node);
                const bool batched = !on_npu && xdna_glue_claims(node);
                if (batched) {
                    glue_batch.push_back(node);
                }
                if (!glue_batch.empty() && !xdna_glue_flush(&ctx->ops, glue_batch)) {
                    return GGML_STATUS_FAILED;
                }
                if (!batched && !(on_npu ? xdna_ops_compute(&ctx->ops, node) : xdna_glue_host_run(&ctx->ops, node))) {
                    GGML_LOG_ERROR("%s: fused layer %d: cannot snapshot %s\n", "ggml-xdna", c.p->il,
                                   node->name ? node->name : "?");
                    return GGML_STATUS_FAILED;
                }
                if (on_npu && !xdna_ops_finalize(&ctx->ops)) {
                    return GGML_STATUS_FAILED;
                }
            }
            // a seed is taken when the session will reseed: not seeded, or
            // seeded with another sequence's state (xdna_rec_run switches)
            const bool reseed = s && (!s->seeded || s->row != xdna_rec_seq_row(*c.p));
            if (s && node->type == GGML_TYPE_F32 && node->data) {
                switch (c.kind) {
                    case 0:  // qkv
                        s->cap_qkv.resize(xdna_rec_pack::CH);
                        std::memcpy(s->cap_qkv.data(), node->data, xdna_rec_pack::CH * sizeof(float));
                        break;
                    case 1:  // conv seed (only consumed at the one-time seed)
                        if (reseed) {
                            s->cap_conv.resize((size_t) 3 * xdna_rec_pack::CH);
                            std::memcpy(s->cap_conv.data(), node->data, (size_t) 3 * xdna_rec_pack::CH * sizeof(float));
                        }
                        break;
                    case 2:
                        {  // ssm seed (only consumed at the one-time seed)
                            const int64_t n = xdna_rec_pack::state_floats();
                            if (reseed) {
                                s->cap_ss.resize((size_t) n);
                                std::memcpy(s->cap_ss.data(), node->data, (size_t) n * sizeof(float));
                            }
                            break;
                        }
                    case 3:  // residual h
                        s->cap_hres.resize(xdna_rec_pack::D_OUT);
                        std::memcpy(s->cap_hres.data(), node->data, xdna_rec_pack::D_OUT * sizeof(float));
                        break;
                    case 4:  // the in-projection's input
                        s->cap_act.resize((size_t) node->ne[0]);
                        std::memcpy(s->cap_act.data(), node->data, (size_t) node->ne[0] * sizeof(float));
                        break;
                    default:
                        break;
                }
            }
            consumed.insert(node);
            break;
        }
        // Fused decode layer firing point: everything the layer reads has been
        // executed by now, so flush the pending NPU/glue work and run the fused
        // layer (writes h_attn/h_out and consumes the subgraph).
        for (auto & ap : attl_plans) {
            if (ap.i_first != i || consumed.count(node)) {
                continue;
            }
            if (!glue_batch.empty() && !xdna_glue_flush(&ctx->ops, glue_batch)) {
                return GGML_STATUS_FAILED;
            }
            if (!xdna_ops_finalize(&ctx->ops)) {
                return GGML_STATUS_FAILED;
            }
            bool failed = false, ran;
            {
                xdna_prof::fused_layer_timer xdna_prof_fl(ap.tail.il);
                ran = xdna_attl_try(ctx, ap, consumed, cgraph, &failed);
            }
            if (failed) {
                GGML_LOG_ERROR("%s: attention layer %d: the dispatch failed\n", "ggml-xdna", ap.tail.il);
                return GGML_STATUS_FAILED;
            }
            if (!ran && ap.pre) {
                // the other way after all: the input's norm on the host now
                if (!xdna_res_materialize(ctx)) {
                    return GGML_STATUS_FAILED;
                }
                const ggml_tensor * xn = ap.n_q->src[1];
                if (!xdna_glue_host_run(&ctx->ops, xn->src[0]) || !xdna_glue_host_run(&ctx->ops, xn)) {
                    return GGML_STATUS_FAILED;
                }
            }
            break;
        }
        for (auto & tp : tail_plans) {
            if (tp.i_fire != i || consumed.count(node)) {
                continue;
            }
            if (!glue_batch.empty() && !xdna_glue_flush(&ctx->ops, glue_batch)) {
                return GGML_STATUS_FAILED;
            }
            if (!xdna_ops_finalize(&ctx->ops)) {
                return GGML_STATUS_FAILED;
            }
            bool ok;
            {
                xdna_prof::fused_layer_timer xdna_prof_fl(tp.il);
                ok = xdna_tail_run(ctx, tp, consumed, cgraph);
            }
            if (!ok) {
                xdna_ops_finalize(&ctx->ops);
                return GGML_STATUS_FAILED;
            }
            break;
        }
        if (!rec_plans.empty()) {
            for (auto & pl : rec_plans) {
                if (pl.i_fire == i) {
                    if (!glue_batch.empty()) {
                        if (!xdna_glue_flush(&ctx->ops, glue_batch)) {
                            return GGML_STATUS_FAILED;
                        }
                    }
                    if (!xdna_ops_finalize(&ctx->ops)) {
                        return GGML_STATUS_FAILED;
                    }
                    bool rec_ok;
                    {
                        xdna_prof::fused_layer_timer xdna_prof_fl(pl.il);
                        rec_ok = xdna_rec_run(ctx, pl, consumed, cgraph);
                    }
                    if (!rec_ok) {
                        xdna_ops_finalize(&ctx->ops);
                        return GGML_STATUS_FAILED;
                    }
                    break;
                }
            }
        }
        if (ggml_xdna_is_view_op(node->op)) {
            continue;
        }
        auto pa = pair_at.find(node);
        if (pa != pair_at.end()) {
            mm_pair & p = pairs[pa->second];
            if (!glue_batch.empty() && !xdna_glue_flush(&ctx->ops, glue_batch)) {
                return GGML_STATUS_FAILED;
            }
            if (!xdna_ops_finalize(&ctx->ops)) {
                return GGML_STATUS_FAILED;
            }
            xdna_pending_wait(ctx);
            p.out_a.resize((size_t) ggml_nelements(p.a));
            p.out_b.resize((size_t) ggml_nelements(p.b));
            xdna_prof::section_timer st("prefill: pgemm pair");
            p.ok = xdna_pgemm_run_pair(ctx->ops.pool, p.a, p.b, p.out_a.data(), p.out_b.data());
        }
        auto qo = qkv_of.find(node);
        if (qo != qkv_of.end()) {
            // the in-projection into the conv's buffer, not its node
            gdn_in_plan & p = gdn_in[qo->second];
            if (!glue_batch.empty() && !xdna_glue_flush(&ctx->ops, glue_batch)) {
                return GGML_STATUS_FAILED;
            }
            if (!xdna_ops_finalize(&ctx->ops)) {
                return GGML_STATUS_FAILED;
            }
            size_t        off = 0;
            xdna_buffer * bo  = xdna_gdn_conv_input(ctx->ops.pool, (int) node->ne[1], (int) node->ne[0], &off);
            {
                xdna_prof::section_timer st("prefill: pgemm");
                p.qkv_ok = bo && xdna_pgemm_run_into(ctx->ops.pool, node, bo, off);
            }
            if (!p.qkv_ok && !xdna_ops_compute(&ctx->ops, node)) {
                return GGML_STATUS_FAILED;
            }
            continue;
        }
        if (consumed.count(node)) {
            continue;
        }
        if (gate_a.count(node)) {
            if (!glue_batch.empty() && !xdna_glue_flush(&ctx->ops, glue_batch)) {
                return GGML_STATUS_FAILED;
            }
            if (!xdna_ops_finalize(&ctx->ops)) {
                return GGML_STATUS_FAILED;
            }
            xdna_pending_wait(ctx);
            bool                       ok;
            const struct ggml_tensor * fa = node->src[0]->view_src;
            xdna_attn_rows             arows;
            const bool                 akept = fa && xdna_attn_mm_rows(fa, &arows);
            {
                xdna_prof::section_timer st("prefill: gate into A");
                ok = xdna_pgemm_run_gate(ctx->ops.pool, node, akept ? &arows : nullptr);
            }
            if (!ok) {
                if (akept && !xdna_attn_mm_materialize(fa)) {
                    return GGML_STATUS_FAILED;
                }
                const struct ggml_tensor * sg = node->src[1];
                if ((sg->src[0]->op == GGML_OP_CONT && !xdna_glue_host_run(&ctx->ops, sg->src[0])) ||
                    !xdna_glue_host_run(&ctx->ops, sg) || !xdna_glue_host_run(&ctx->ops, node)) {
                    return GGML_STATUS_FAILED;
                }
            }
            continue;
        }
        auto ga = gated_a.find(node);
        if (ga != gated_a.end()) {
            if (!glue_batch.empty() && !xdna_glue_flush(&ctx->ops, glue_batch)) {
                return GGML_STATUS_FAILED;
            }
            if (!xdna_ops_finalize(&ctx->ops)) {
                return GGML_STATUS_FAILED;
            }
            xdna_pending_wait(ctx);
            bool                       ok;
            const struct ggml_tensor * gdn = node->src[0]->src[0]->src[0]->view_src;
            xdna_gdn_rows              rows;
            const bool                 kept = xdna_gdn_mm_rows(gdn, &rows);
            {
                xdna_prof::section_timer st("prefill: gated output into A");
                ok = xdna_pgemm_run_gated(ctx->ops.pool, node, ga->second, kept ? &rows : nullptr);
            }
            if (!ok) {
                if (kept && !xdna_gdn_mm_materialize(gdn)) {
                    return GGML_STATUS_FAILED;
                }
                const struct ggml_tensor * m = node->src[0];
                if (!xdna_glue_host_run(&ctx->ops, m->src[0]) || !xdna_glue_host_run(&ctx->ops, m) ||
                    !xdna_glue_host_run(&ctx->ops, node->src[1]) || !xdna_glue_host_run(&ctx->ops, node)) {
                    return GGML_STATUS_FAILED;
                }
            }
            continue;
        }
        if (norm_a.count(node)) {
            if (!glue_batch.empty() && !xdna_glue_flush(&ctx->ops, glue_batch)) {
                return GGML_STATUS_FAILED;
            }
            if (!xdna_ops_finalize(&ctx->ops)) {
                return GGML_STATUS_FAILED;
            }
            xdna_pending_wait(ctx);
            bool ok;
            {
                xdna_prof::section_timer st("prefill: norm into A");
                auto                     na = norm_add.find(node);
                ok = xdna_pgemm_run_norm(ctx->ops.pool, node, na != norm_add.end() ? na->second : nullptr);
            }
            if (!ok) {
                auto na = norm_add.find(node);
                if ((na != norm_add.end() && !xdna_glue_host_run(&ctx->ops, na->second)) ||
                    !xdna_glue_host_run(&ctx->ops, node->src[0]) || !xdna_glue_host_run(&ctx->ops, node)) {
                    return GGML_STATUS_FAILED;
                }
            }
            continue;
        }
        auto po = pair_of.find(node);
        if (po != pair_of.end() && pairs[po->second].ok) {
            // the node's memory may be a dead input a pending host op reads
            if (!glue_batch.empty() && !xdna_glue_flush(&ctx->ops, glue_batch)) {
                return GGML_STATUS_FAILED;
            }
            const mm_pair &            p   = pairs[po->second];
            const std::vector<float> & out = node == p.a ? p.out_a : p.out_b;
            const size_t               row = (size_t) node->ne[0] * sizeof(float);
            for (int64_t r = 0; r < node->ne[1]; r++) {
                std::memcpy((char *) node->data + r * node->nb[1], out.data() + r * node->ne[0], row);
            }
            continue;
        }
        if (glu_fused.count(node)) {
            if (!glue_batch.empty() && !xdna_glue_flush(&ctx->ops, glue_batch)) {
                return GGML_STATUS_FAILED;
            }
            if (!xdna_ops_finalize(&ctx->ops)) {
                return GGML_STATUS_FAILED;
            }
            xdna_pending_wait(ctx);
            bool ok;
            {
                xdna_prof::section_timer st("prefill: pgemm swiglu");
                ok = (glu_to_a.count(node) && xdna_pgemm_run_glu_fused(ctx->ops.pool, node)) ||
                     xdna_pgemm_run_glu(ctx->ops.pool, node);
            }
            if (!ok) {
                // the two projections one at a time and the host's SwiGLU
                if (!xdna_ops_compute(&ctx->ops, node->src[0]) || !xdna_ops_compute(&ctx->ops, node->src[1]) ||
                    !xdna_ops_finalize(&ctx->ops) || !xdna_glue_host_run(&ctx->ops, node)) {
                    return GGML_STATUS_FAILED;
                }
            }
            continue;
        }
        if (!xdna_res_touch(ctx, node)) {
            return GGML_STATUS_FAILED;
        }
        if (xdna_head_on() && g_glue_n_tokens == 1 && xdna_head_node(node)) {
            xdna_head_chain hc;
            const bool      chain = xdna_head_chain_of(node, hc);
            if (!ctx->head && !ctx->head_tried) {
                ctx->head_tried = true;
                if (!ctx->res) {
                    ctx->res = xdna_res_alloc(ctx->device);
                }
                xdna_arena_scope arena;
                ctx->head = xdna_head_create(ctx->pool, node->src[0], chain ? ctx->res : nullptr,
                                             chain ? (const float *) hc.gamma->data : nullptr, hc.eps);
            }
            // From the rows, queued with the layers, when the norm reads what
            // they hold: the norm and the head are the array's.
            bool from_rows = false;
            if (ctx->head && head_pre && ctx->res_lout && xdna_queue_on()) {
                for (const ggml_tensor * t = hc.src; t; t = t->view_src) {
                    from_rows = from_rows || t == ctx->res_lout;
                }
            }
            if (from_rows) {
                ggml_backend_xdna_context::pending_run pr;
                xdna_batch_open();
                pr.run = xdna_head_start_rows(ctx->head);
                if (!pr.run) {
                    return GGML_STATUS_FAILED;
                }
                pr.logits = (float *) node->data;
                ctx->pending.push_back(std::move(pr));
                continue;
            }
            if (ctx->head && head_pre) {
                // the other way after all: the norm on the host now
                if (!xdna_res_materialize(ctx)) {
                    return GGML_STATUS_FAILED;
                }
                for (ggml_tensor * t : { hc.rms, hc.mul, hc.rows }) {
                    if (t && !xdna_glue_host_run(&ctx->ops, t)) {
                        return GGML_STATUS_FAILED;
                    }
                }
            }
            if (ctx->head) {
                if (!glue_batch.empty() && !xdna_glue_flush(&ctx->ops, glue_batch)) {
                    return GGML_STATUS_FAILED;
                }
                if (!xdna_ops_finalize(&ctx->ops)) {
                    return GGML_STATUS_FAILED;
                }
                xdna_pending_wait(ctx);
                bool ok;
                {
                    xdna_prof::section_timer st_head("head: NPU");
                    ok = xdna_head_run(ctx->head, (const float *) node->src[1]->data, (float *) node->data);
                }
                if (!ok) {
                    return GGML_STATUS_FAILED;
                }
                continue;
            }
        }
        // A concat the conv reads through: write the live tail, then run the
        // conv here, while the projection behind the concat is still alive.
        // Both read sources the glue batch may still be producing, so flush.
        {
            auto cf = concat_tail.find(node);
            if (cf != concat_tail.end()) {
                if (!glue_batch.empty()) {
                    if (!xdna_glue_flush(&ctx->ops, glue_batch)) {
                        return GGML_STATUS_FAILED;
                    }
                }
                concat_hist.emplace_back();
                xdna_concat_tail_fill(node, concat_hist.back());
                if (!xdna_ops_compute(&ctx->ops, cf->second) || !xdna_ops_finalize(&ctx->ops)) {
                    xdna_ops_finalize(&ctx->ops);
                    return GGML_STATUS_FAILED;
                }
                consumed.insert(cf->second);
                continue;
            }
        }
        // Decode attention on the pool; anything it does not take stays on
        // the host below.
        if (node->op == GGML_OP_FLASH_ATTN_EXT && xdna_att_on() && xdna_rec_active() && xdna_att_eligible(node)) {
            if (!glue_batch.empty() && !xdna_glue_flush(&ctx->ops, glue_batch)) {
                return GGML_STATUS_FAILED;
            }
            if (!xdna_ops_finalize(&ctx->ops)) {
                return GGML_STATUS_FAILED;
            }
            xdna_pending_wait(ctx);
            if (xdna_att_try(ctx, node)) {
                if (getenv("GGML_XDNA_ATTN_CHECK")) {
                    // Diagnostic: the host's flash attention on the same node,
                    // against the pool's; the host's result is kept unless
                    // GGML_XDNA_ATTN_CHECK=npu.
                    std::vector<float> npu((const float *) node->data, (const float *) node->data + 2048);
                    xdna_glue_host_run(&ctx->ops, node);
                    double e2 = 0, r2 = 0, mx = 0;
                    for (int q = 0; q < 2048; q++) {
                        const double d = npu[q] - ((const float *) node->data)[q];
                        e2 += d * d;
                        r2 += (double) ((const float *) node->data)[q] * (double) ((const float *) node->data)[q];
                        mx = std::max(mx, fabs(d));
                    }
                    static int shown      = 0;
                    const int  shown_prev = shown++;
                    if (shown_prev < 40 || (shown % 200) == 0) {
                        fprintf(stderr, "[attcheck] %s nkv=%lld rel=%.3e max=%.3e\n", node->name,
                                (long long) node->src[1]->ne[1], sqrt(e2 / std::max(r2, 1e-30)), mx);
                    }
                    if (strcmp(getenv("GGML_XDNA_ATTN_CHECK"), "npu") == 0) {
                        memcpy(node->data, npu.data(), 2048 * sizeof(float));
                    }
                }
                continue;
            }
        }
        // Glue-claimed cheap ops accumulate into one CPU batch (flushed before
        // the next NPU op or at the graph end).
        if (xdna_glue_claims(node) && !xdna_ops_supported(&ctx->ops, node)) {
            // A concat this backend can copy itself does not go to the batch:
            // it is pure data movement with no cgraph amortization to gain, and
            // ggml's element-at-a-time version is the most expensive host op in
            // the graph. Flush first so its inputs are ready.
            if (node->op == GGML_OP_CONCAT || node->op == GGML_OP_SSM_CONV) {
                if (!glue_batch.empty()) {
                    if (!xdna_glue_flush(&ctx->ops, glue_batch)) {
                        return GGML_STATUS_FAILED;
                    }
                }
                if (!xdna_ops_finalize(&ctx->ops)) {
                    return GGML_STATUS_FAILED;
                }
                auto gi = gdn_in.find(node);
                if (gi != gdn_in.end()) {
                    bool done = false;
                    if (gi->second.qkv_ok) {
                        {
                            xdna_prof::section_timer st("prefill: gdn conv");
                            done = xdna_gdn_mm_prepare_npu(ctx->ops.pool, gi->second.gdn, gi->second.in);
                        }
                        if (!done && !xdna_ops_compute(&ctx->ops, gi->second.qkv)) {
                            // the projection where llama reads it, for the host
                            return GGML_STATUS_FAILED;
                        }
                    }
                    if (!done) {
                        xdna_prof::section_timer st("prefill: gdn input");
                        done = xdna_gdn_mm_prepare(ctx->ops.pool, gi->second.gdn, gi->second.in);
                    }
                    if (done) {
                        continue;
                    }
                    // the chain as llama built it
                    for (struct ggml_tensor * t : gi->second.skip) {
                        consumed.erase(t);
                    }
                }
                if (node->op == GGML_OP_CONCAT ? xdna_concat_fast(node) : xdna_ssm_conv_fast(node)) {
                    continue;
                }
            }
            glue_batch.push_back(node);
            continue;
        }
        if (!glue_batch.empty()) {
            if (!xdna_glue_flush(&ctx->ops, glue_batch)) {
                return GGML_STATUS_FAILED;
            }
        }
        xdna_pending_wait(ctx);
        if (!xdna_ops_compute(&ctx->ops, node)) {
            xdna_ops_finalize(&ctx->ops);
            return GGML_STATUS_FAILED;
        }
    }
    if (!glue_batch.empty()) {
        if (!xdna_glue_flush(&ctx->ops, glue_batch)) {
            return GGML_STATUS_FAILED;
        }
    }
    if (!xdna_ops_finalize(&ctx->ops)) {
        return GGML_STATUS_FAILED;
    }
    const bool queued_ok = xdna_pending_wait(ctx);
    ctx->pending_failed  = false;
    // Whatever the rows still hold nothing read: the tensors are this graph's,
    // and their memory may already be another's.
    ctx->res_lout = ctx->res_resid = nullptr;
    if (!queued_ok) {
        ctx->res_dirty = false;
        return GGML_STATUS_FAILED;
    }
    ctx->res_dirty = false;
    return GGML_STATUS_SUCCESS;
}

// The boundary between llama.cpp and this backend: XRT throws on device
// failures, and an exception crossing into llama.cpp names nothing. The
// impl's early returns and RAII timers keep their semantics inside.
static enum ggml_status ggml_backend_xdna_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_xdna_context * ctx = (ggml_backend_xdna_context *) backend->context;
    std::lock_guard<std::mutex>  compute(ctx->compute_mutex);
    try {
        const enum ggml_status status = xdna_graph_compute_impl(ctx, cgraph);
        ctx->cur_node                 = nullptr;
        ctx->cur_index                = -1;
        return status;
    } catch (const std::exception & e) {
        if (ctx->cur_node) {
            GGML_LOG_ERROR("%s: graph compute: node %d %s (%s): %s\n", "ggml-xdna", ctx->cur_index,
                           ggml_op_name(ctx->cur_node->op), ctx->cur_node->name[0] ? ctx->cur_node->name : "?",
                           e.what());
        } else {
            GGML_LOG_ERROR("%s: graph compute: %s\n", "ggml-xdna", e.what());
        }
        ctx->cur_node  = nullptr;
        ctx->cur_index = -1;
        return GGML_STATUS_FAILED;
    } catch (...) {
        if (ctx->cur_node) {
            GGML_LOG_ERROR("%s: graph compute: node %d %s (%s): unknown exception\n", "ggml-xdna", ctx->cur_index,
                           ggml_op_name(ctx->cur_node->op), ctx->cur_node->name[0] ? ctx->cur_node->name : "?");
        } else {
            GGML_LOG_ERROR("%s: graph compute: unknown exception\n", "ggml-xdna");
        }
        ctx->cur_node  = nullptr;
        ctx->cur_index = -1;
        return GGML_STATUS_FAILED;
    }
}

static struct ggml_backend_i xdna_backend_i = {
    /* .get_name                = */ ggml_backend_xdna_get_name,
    /* .free                    = */ ggml_backend_xdna_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_xdna_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_xdna_guid(void) {
    static ggml_guid guid = { 0x7a, 0xff, 0x0d, 0x7e, 0x5a, 0x1b, 0x4c, 0xf6,
                              0xbd, 0x94, 0x6e, 0xc1, 0xf1, 0xcb, 0x3f, 0xbd };
    return &guid;
}

ggml_backend_t ggml_backend_xdna_init(void) {
    ggml_backend_xdna_context * ctx = ggml_xdna_device_context();

    // When the NPU is absent the backend is still registered (llama.cpp
    // expects every ACCEL device to yield a backend), but supports_op()
    // rejects everything so all work stays on the CPU.
    if (!ctx->device) {
        GGML_LOG_INFO("%s: XDNA backend init: disabled (no NPU)\n", __func__);
    } else {
        GGML_LOG_INFO("%s: XDNA backend init: device=%s\n", __func__, ctx->device->name.c_str());
    }

    ggml_backend_t backend = new ggml_backend{
        /* .guid    = */ ggml_backend_xdna_guid(),
        /* .iface   = */ xdna_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_xdna_reg(), 0),
        /* .context = */ ctx,
    };
    {
        std::lock_guard<std::mutex> lock(ctx->life_mutex);
        ctx->n_backends++;
    }

    return backend;
}

bool ggml_backend_is_xdna(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_xdna_guid());
}

// device interface

static const char * ggml_backend_xdna_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_xdna_context * ctx = (ggml_backend_xdna_context *) dev->context;
    return ctx->device ? ctx->device->name.c_str() : "XDNA";
}

static const char * ggml_backend_xdna_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_xdna_context * ctx = (ggml_backend_xdna_context *) dev->context;
    return ctx->device ? ctx->device->description.c_str() : "AMD XDNA (no NPU device)";
}

static void ggml_backend_xdna_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_UNUSED(dev);
    // The NPU reads/writes host-visible BOs, so report the host memory the
    // device can consume (fit params divides by the free size, so it must be
    // non-zero for the XDNA device to be used with --fit).
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    *total = status.ullTotalPhys;
    *free  = status.ullAvailPhys;
#else
    long pages     = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    *total         = (size_t) pages * page_size;
    // "free" system memory is ill-defined, for practical purposes assume all of it is free:
    *free          = *total;
#endif  // _WIN32
}

static enum ggml_backend_dev_type ggml_backend_xdna_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_xdna_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_xdna_device_get_name(dev);
    props->description = ggml_backend_xdna_device_get_description(dev);
    props->type        = ggml_backend_xdna_device_get_type(dev);
    ggml_backend_xdna_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_xdna_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(dev);
    GGML_UNUSED(params);
    return ggml_backend_xdna_init();
}

// ---------------------------------------------------------------------------
// The backend's buffer type: host memory the array can read.
//
// Tensors llama allocates on this device - the KV cache among them - used to
// live in the CPU buffer type, plain host memory no DMA can address. They now
// live in XRT host BOs, mapped, so the host glue works on them exactly as
// before (the buffer type is host) and a design can bind the BO a tensor sits
// in as an argument (xdna_host_bo_of). Model weights loaded by mmap still come
// through buffer_from_host_ptr and are repacked into BOs of their own as
// before.
// ---------------------------------------------------------------------------

namespace {

struct xdna_host_bo_entry {
    xdna_buffer * bo;
    size_t        bytes;
};

}  // namespace

static std::mutex & xdna_host_bo_mu(void) {
    static std::mutex m;
    return m;
}

static std::map<uintptr_t, xdna_host_bo_entry> & xdna_host_bos(void) {
    static std::map<uintptr_t, xdna_host_bo_entry> m;
    return m;
}

// The BO a host pointer of this buffer type lies in, and its byte offset
// there; null when the pointer is not in one.
xdna_buffer * xdna_host_bo_of(const void * p, size_t * offset) {
    std::lock_guard<std::mutex> lk(xdna_host_bo_mu());
    auto &                      m  = xdna_host_bos();
    auto                        it = m.upper_bound((uintptr_t) p);
    if (it == m.begin()) {
        return nullptr;
    }
    --it;
    const uintptr_t off = (uintptr_t) p - it->first;
    if (off >= it->second.bytes) {
        return nullptr;
    }
    if (offset) {
        *offset = (size_t) off;
    }
    return it->second.bo;
}

static void xdna_hb_free(ggml_backend_buffer_t buffer) {
    xdna_buffer * bo = (xdna_buffer *) buffer->context;
    {
        std::lock_guard<std::mutex> lk(xdna_host_bo_mu());
        xdna_host_bos().erase((uintptr_t) bo->data);
    }
    xdna_buffer_free(bo);
}

static void * xdna_hb_base(ggml_backend_buffer_t buffer) {
    return ((xdna_buffer *) buffer->context)->data;
}

static void xdna_hb_memset(ggml_backend_buffer_t buffer,
                           struct ggml_tensor *  tensor,
                           uint8_t               value,
                           size_t                offset,
                           size_t                size) {
    GGML_UNUSED(buffer);
    memset((char *) tensor->data + offset, value, size);
}

static void xdna_hb_set(ggml_backend_buffer_t buffer,
                        struct ggml_tensor *  tensor,
                        const void *          data,
                        size_t                offset,
                        size_t                size) {
    GGML_UNUSED(buffer);
    memcpy((char *) tensor->data + offset, data, size);
}

static void xdna_hb_get(ggml_backend_buffer_t      buffer,
                        const struct ggml_tensor * tensor,
                        void *                     data,
                        size_t                     offset,
                        size_t                     size) {
    GGML_UNUSED(buffer);
    memcpy(data, (const char *) tensor->data + offset, size);
}

static bool xdna_hb_cpy(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_UNUSED(buffer);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;
}

static void xdna_hb_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(xdna_hb_base(buffer), value, buffer->size);
}

static const struct ggml_backend_buffer_i xdna_hb_iface = {
    /* .free_buffer   = */ xdna_hb_free,
    /* .get_base      = */ xdna_hb_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ xdna_hb_memset,
    /* .set_tensor    = */ xdna_hb_set,
    /* .get_tensor    = */ xdna_hb_get,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ xdna_hb_cpy,
    /* .clear         = */ xdna_hb_clear,
    /* .reset         = */ nullptr,
};

static const char * xdna_hbt_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return "XDNA_Host";
}

static ggml_backend_buffer_t xdna_hbt_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_xdna_context * ctx = ggml_xdna_device_context();
    // A zero-size buffer is legal in ggml; a BO is not.
    xdna_buffer * bo = ctx && ctx->device ? xdna_buffer_alloc(ctx->device, std::max<size_t>(size, 4096)) : nullptr;
    if (!bo) {
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }
    {
        std::lock_guard<std::mutex> lk(xdna_host_bo_mu());
        xdna_host_bos()[(uintptr_t) bo->data] = { bo, bo->bytes };
    }
    return ggml_backend_buffer_init(buft, xdna_hb_iface, bo, size);
}

static size_t xdna_hbt_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 64;
}

static bool xdna_hbt_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return true;
}

static ggml_backend_buffer_type_t xdna_host_buffer_type(void) {
    // The device is filled in on first use: the registry is built after this
    // file's statics.
    static struct ggml_backend_buffer_type buft = {
        /* .iface   = */ {
                          /* .get_name       = */ xdna_hbt_name,
                          /* .alloc_buffer   = */ xdna_hbt_alloc,
                          /* .get_alignment  = */ xdna_hbt_alignment,
                          /* .get_max_size   = */ nullptr,
                          /* .get_alloc_size = */ nullptr,
                          /* .is_host        = */ xdna_hbt_is_host,
                          },
        /* .device  = */
        nullptr,
        /* .context = */ nullptr,
    };
    if (!buft.device) {
        buft.device = ggml_backend_reg_dev_get(ggml_backend_xdna_reg(), 0);
    }
    return &buft;
}

static ggml_backend_buffer_type_t ggml_backend_xdna_device_get_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    // GGML_XDNA_HOST_BO=0 keeps the plain CPU buffer type.
    static const bool bo = xdna_env_int("GGML_XDNA_HOST_BO", 1) != 0;
    return bo ? xdna_host_buffer_type() : ggml_backend_cpu_buffer_type();
}

static ggml_backend_buffer_t ggml_backend_xdna_device_buffer_from_host_ptr(ggml_backend_dev_t dev,
                                                                           void *             ptr,
                                                                           size_t             size,
                                                                           size_t             max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);
}

static bool ggml_backend_xdna_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    ggml_backend_xdna_context * ctx = (ggml_backend_xdna_context *) dev->context;

    if (!ctx->device || !ctx->pool) {
        return false;
    }

    // Claim the native NPU ops plus the whitelisted cheap ops of the fused
    // recurrent layer, so the decode chunk stays whole and the glue can host
    // the rest.
    return xdna_glue_claims(op) || xdna_ops_supported(&ctx->ops, op);
}

static bool ggml_backend_xdna_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return ggml_backend_buft_is_host(buft);
}

static const struct ggml_backend_device_i ggml_backend_xdna_device_i = {
    /* .get_name             = */ ggml_backend_xdna_device_get_name,
    /* .get_description      = */ ggml_backend_xdna_device_get_description,
    /* .get_memory           = */ ggml_backend_xdna_device_get_memory,
    /* .get_type             = */ ggml_backend_xdna_device_get_type,
    /* .get_props            = */ ggml_backend_xdna_device_get_props,
    /* .init_backend         = */ ggml_backend_xdna_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_xdna_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_xdna_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_xdna_device_supports_op,
    /* .supports_buft        = */ ggml_backend_xdna_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// backend reg interface

static const char * ggml_backend_xdna_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "XDNA";
}

static size_t ggml_backend_xdna_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return ggml_xdna_device_context()->device ? 1 : 0;
}

static ggml_backend_dev_t ggml_backend_xdna_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    GGML_UNUSED(index);

    static ggml_backend_device ggml_backend_xdna_device = {
        /* .iface   = */ ggml_backend_xdna_device_i,
        /* .reg     = */ reg,
        /* .context = */ ggml_xdna_device_context(),
    };

    return &ggml_backend_xdna_device;
}

static void * ggml_backend_xdna_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
    return NULL;
}

static const struct ggml_backend_reg_i ggml_backend_xdna_reg_i = {
    /* .get_name         = */ ggml_backend_xdna_reg_get_name,
    /* .get_device_count = */ ggml_backend_xdna_reg_get_device_count,
    /* .get_device       = */ ggml_backend_xdna_reg_get_device,
    /* .get_proc_address = */ ggml_backend_xdna_get_proc_address,
};

ggml_backend_reg_t ggml_backend_xdna_reg(void) {
    static struct ggml_backend_reg ggml_backend_xdna_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_xdna_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_xdna_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_xdna_reg)
