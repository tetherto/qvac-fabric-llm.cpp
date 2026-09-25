#include "ggml-impl.h"
#include "ggml-xdna.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "xdna-util.h"

#include "xdna-types.h"
#include "xdna-runtime.h"
#include "xdna-ops.h"
#include "xdna-design-tag.h"
#include "xdna-rec.h"
#include "xdna-rec-gemv.h"
#include "xdna-conv-prefill.h"

#include <algorithm>
#include <chrono>
#include <deque>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

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
static bool xdna_concat_fast(struct ggml_tensor * dst) {
    if (dst == nullptr || dst->op != GGML_OP_CONCAT || dst->type != GGML_TYPE_F32) {
        return false;
    }
    struct ggml_tensor * a = dst->src[0];
    struct ggml_tensor * b = dst->src[1];
    if (a == nullptr || b == nullptr ||
        a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) {
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
    if (ggml_get_op_params_i32(dst, 0) == 0 && ggml_is_contiguous(dst) &&
        b->nb[1] == sizeof(float) && a->nb[0] == sizeof(float)) {
        bool dims_ok = true;
        for (int d = 1; d < 4; d++) {
            if (a->ne[d] != dst->ne[d] || b->ne[d] != dst->ne[d]) {
                dims_ok = false;
            }
        }
        if (dims_ok && a->ne[0] + b->ne[0] == dst->ne[0]) {
            const int64_t na   = a->ne[0];
            const int64_t nb   = b->ne[0];
            const int64_t ne0  = dst->ne[0];
            const int64_t n_c  = dst->ne[1] * dst->ne[2] * dst->ne[3];
            const float * ad = (const float *) a->data;
            const float * bd = (const float *) b->data;
            float * dd = (float *) dst->data;
            constexpr int TC = 16;
            constexpr int TT = 16;
            for (int64_t c0 = 0; c0 < n_c; c0 += TC) {
                const int nc = (int) std::min<int64_t>(TC, n_c - c0);
                for (int c = 0; c < nc; c++) {
                    std::memcpy(dd + (c0 + c) * ne0,
                                (const char *) ad + (size_t) (c0 + c) * a->nb[1],
                                (size_t) na * sizeof(float));
                }
                for (int64_t t0 = 0; t0 < nb; t0 += TT) {
                    const int nt = (int) std::min<int64_t>(TT, nb - t0);
                    for (int t = 0; t < nt; t++) {
                        const float * srow = (const float *)
                            ((const char *) bd + (size_t) (t0 + t) * b->nb[0] +
                             (size_t) c0 * b->nb[1]);
                        const int64_t drow = na + t0 + t;
                        for (int c = 0; c < nc; c++) {
                            dd[(c0 + c) * ne0 + drow] = srow[c];
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
    char * pd = (char *) dst->data;
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
static int xdna_glue_threads(bool prompt) {
    static const int n_small = []() {
        const char * v = getenv("GGML_XDNA_GLUE_THREADS");
        return v ? std::max(1, atoi(v)) : 4;
    }();
    static const int n_big = []() {
        const char * v = getenv("GGML_XDNA_GLUE_THREADS_BIG");
        return v ? std::max(1, atoi(v)) : 16;
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
            /* .mem_size   = */ 512 * 1024,
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
    ggml_cgraph * cg = xdna_glue_cgraph();
    if (cpu == nullptr || cg == nullptr) {
        GGML_LOG_ERROR("%s: glue: no CPU backend/graph for %s\n", "ggml-xdna", ggml_op_name(node->op));
        return false;
    }
    if (!xdna_ops_finalize(ops)) {
        return false;
    }
    ggml_backend_cpu_set_n_threads(cpu, xdna_glue_threads(g_glue_n_tokens >= 32));
    cg->n_nodes = 1;
    cg->n_leafs = 0;
    cg->nodes[0] = const_cast<struct ggml_tensor *>(node);
    if (ggml_backend_graph_compute(cpu, cg) != GGML_STATUS_SUCCESS) {
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
    ggml_cgraph * cg = xdna_glue_cgraph();
    if (cpu == nullptr || cg == nullptr) {
        GGML_LOG_ERROR("%s: glue: no CPU backend/graph for batch\n", "ggml-xdna");
        batch.clear();
        return false;
    }
    ggml_backend_cpu_set_n_threads(cpu, xdna_glue_threads(g_glue_n_tokens >= 32));
    size_t off = 0;
    while (off < batch.size()) {
        const size_t n = std::min<size_t>(64, batch.size() - off);
        cg->n_nodes = (int) n;
        cg->n_leafs = 0;
        for (size_t j = 0; j < n; j++) {
            cg->nodes[j] = batch[off + j];
        }
        if (ggml_backend_graph_compute(cpu, cg) != GGML_STATUS_SUCCESS) {
            GGML_LOG_ERROR("%s: glue: host compute failed for batch\n", "ggml-xdna");
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

// Process-wide context shared by all backend instances.
struct ggml_backend_xdna_context {
    xdna_device * device = nullptr;
    xdna_kernel_pool * pool = nullptr;   // lazily scanned kernel pool

    xdna_ops ops;   // operator-specific dispatch (GEMM variants, helpers)

    // Fused decode layer sessions: one per recurrent layer of the model,
    // created on the first single-token decode and kept for the process
    // lifetime.
    // Fused decode layer sessions: one per (recurrent block, sequence). The
    // recurrent state lives on the device per session, and llama-server runs
    // several sequences through one context - keying by the block alone made
    // the slots share the state, which read as instantly-broken logits. The
    // sequence is identified by the row its recurrent-memory cell occupies in
    // the cache tensor.
    std::map<std::pair<int, int64_t>, struct xdna_rec_session *> rec;
};

// One fused-layer session (per recurrent block). Owns the three runners and
// the host scratch buffers it reuses every token.
struct xdna_rec_session {
    int il = -1;
    xdna_rec_core * core = nullptr;    // fused_layer.xclbin (conv+norm+gdn+gated)
    // The layer's projections, on the decode GEMV of the same design
    // (xdna-rec-gemv.h), so the layer never changes hardware context.
    xdna_rec_gemv * gv = nullptr;
    xdna_rec_layer_host host;
    bool seeded = false;          // core/gemv built and state seeded once
    // Set when a prefill ubatch made the llama recurrent cache authoritative
    // again. The session key tells concurrent sequences apart by their cell
    // row, but a later request reuses that row, so without re-seeding the
    // device would carry the previous sequence's conv history and ssm state.
    bool reseed = false;
    std::vector<uint8_t> feed0;        // one-time feed host buffer (hist+qkv+conv W)
    std::vector<uint8_t> x;            // per-token host buffer (eg/beta/scale tails)
    std::vector<float> hattn, hout, hff;
    std::vector<int8_t> aq;            // 2048 gated int8 codes (fused core)
    float d_a = 1.0f;                  // int8 scale of the fused core gated
    // Input snapshots: qkv/conv/ssm seeds and the residual h are transient
    // graph tensors whose buffers llama recycles before the fused run fires,
    // so each is copied here right after its producer node executes.
    std::vector<float> cap_qkv, cap_conv, cap_ss, cap_hres;
};

// Per-graph plan of one fused layer: the node indices/tensors the fused run
// reads or writes.
struct xdna_rec_plan {
    int il = -1;
    int i_conv = -1;                 // SSM_CONV node (layer body start)
    int i_lout = -1;                 // l_out ADD (fused h_out dst)
    int i_fire = -1;                 // first consumed node index
    ggml_tensor * n_qkv = nullptr;   // MUL_MAT blk.N.attn_qkv (qkv data)
    ggml_tensor * n_z = nullptr;     // MUL_MAT blk.N.attn_gate (z data)
    ggml_tensor * n_gate = nullptr;  // MUL gate-N dst
    ggml_tensor * n_beta = nullptr;  // UNARY beta_sigmoid-N dst
    ggml_tensor * n_resid = nullptr; // ADD attn_residual-N (h_attn dst)
    ggml_tensor * n_lout = nullptr;  // ADD l_out-N (h_out dst)
    ggml_tensor * n_convst = nullptr;// GET_ROWS conv_states-N (conv seed)
    ggml_tensor * n_sstate = nullptr;// GET_ROWS cache_s read (ssm seed)
    ggml_tensor * n_hres = nullptr;  // residual input (attn_residual src[1])
    // weight leaves
    ggml_tensor * w_conv = nullptr;
    ggml_tensor * w_gamma = nullptr;
    ggml_tensor * w_post = nullptr;
    ggml_tensor * w_so = nullptr;
    ggml_tensor * w_gate = nullptr;
    ggml_tensor * w_up = nullptr;
    ggml_tensor * w_down = nullptr;
    // producer node indices of the input snapshots above (-1 = not in chunk)
    int i_cap_qkv = -1;              // MUL_MAT qkv node
    int i_cap_conv = -1;             // GET_ROWS conv-state seed node
    int i_cap_ss = -1;               // GET_ROWS cache_s seed node
    int i_cap_hres = -1;             // residual producer node
};

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
        for (const auto & dir : xdna_kernel_search_dirs()) {
            std::error_code ec;
            const std::string f =
                (dir / ("fused_layer_" XDNA_DESIGN_TAG ".xclbin")).string();
            if (std::filesystem::exists(f, ec)) {
                return f;
            }
        }
        for (const auto & dir : xdna_kernel_search_dirs()) {
            std::error_code ec;
            if (std::filesystem::exists(dir / "fused_layer.xclbin", ec)) {
                GGML_LOG_ERROR(
                    "%s: fused_layer.xclbin is present but not built for design "
                    "tag %s; the kernel artifacts are stale. Rebuild them and "
                    "the backend together:\n    cmake --build build --target "
                    "ggml-xdna-kernels\n", "ggml-xdna", XDNA_DESIGN_TAG);
            }
        }
        return std::string();
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
    long v = strtol(name, &end, 10);
    if (end == name || *end != '.') {
        return false;
    }
    *il = (int) v;
    *rest = end + 1;
    return true;
}

static bool xdna_rec_is_leaf(const ggml_tensor * t) {
    return t != nullptr && t->op == GGML_OP_NONE;
}

// One pre-scan pass over a graph_compute node list; fills plans for every
// complete recurrent layer found. A plan is complete when all data nodes the
// fused run needs (and every weight leaf) are present in this chunk.
static bool xdna_rec_scan(ggml_backend_xdna_context * ctx,
                          const ggml_cgraph * cgraph,
                          std::vector<xdna_rec_plan> & plans) {
    GGML_UNUSED(ctx);
    plans.clear();
    std::map<int, xdna_rec_plan> m;
    const int n_nodes = cgraph->n_nodes;

    // First pass: find the per-layer weight leaves and the layer data nodes by
    // name / src-weight structure.
    for (int i = 0; i < n_nodes; i++) {
        const ggml_tensor * n = cgraph->nodes[i];
        for (int s = 0; s < GGML_MAX_SRC && n->src[s]; s++) {
            const ggml_tensor * w = n->src[s];
            if (!xdna_rec_is_leaf(w)) {
                continue;
            }
            int il = -1;
            const char * rest = nullptr;
            if (!xdna_rec_blk(ggml_get_name(w), &il, &rest)) {
                continue;
            }
            xdna_rec_plan & p = m[il];
            p.il = il;
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
            const int il = atoi(nn + 5);
            m[il].il = il;
            m[il].n_gate = const_cast<ggml_tensor *>(n);
        } else if (strncmp(nn, "beta_sigmoid-", 13) == 0) {
            const int il = atoi(nn + 13);
            m[il].il = il;
            m[il].n_beta = const_cast<ggml_tensor *>(n);
        } else if (strncmp(nn, "conv_states-", 12) == 0 && n->op == GGML_OP_GET_ROWS) {
            const int il = atoi(nn + 12);
            m[il].il = il;
            m[il].n_convst = const_cast<ggml_tensor *>(n);
        } else if (strncmp(nn, "l_out-", 6) == 0 && n->op == GGML_OP_ADD) {
            const int il = atoi(nn + 6);
            m[il].il = il;
            m[il].n_lout = const_cast<ggml_tensor *>(n);
            m[il].i_lout = i;
        } else if (strncmp(nn, "attn_residual-", 14) == 0 && n->op == GGML_OP_ADD) {
            const int il = atoi(nn + 14);
            m[il].il = il;
            m[il].n_resid = const_cast<ggml_tensor *>(n);
        }
        // cache_s read (GET_ROWS over the 262144-float cell), the ssm seed.
        if (n->op == GGML_OP_GET_ROWS && n->ne[0] == 262144 && n->ne[1] == 1) {
            const ggml_tensor * r = n->src[0];
            if (r && ggml_get_name(r)) {
                const char * rn = ggml_get_name(r);
                if (strncmp(rn, "cache_s_l", 9) == 0) {
                    const int il = atoi(rn + 9);
                    m[il].il = il;
                    m[il].n_sstate = const_cast<ggml_tensor *>(n);
                }
            }
        }
    }

    // Decide the fire index and drop layers that are not single-token decodes
    // or miss a needed data node in this chunk.
    for (auto & kv : m) {
        xdna_rec_plan & p = kv.second;
        if (!p.w_conv || !p.w_so || !p.w_gamma || !p.w_post ||
            !p.w_gate || !p.w_up || !p.w_down || !p.n_qkv || !p.n_z ||
            !p.n_gate || !p.n_beta || !p.n_resid || !p.n_lout ||
            !p.n_convst || !p.n_sstate || p.i_conv < 0 || p.i_lout < 0) {
            continue;   // not a complete recurrent layer in this chunk
        }
        // Single-token, single-sequence decode only (qkv rows == 1).
        if (p.n_qkv->ne[1] != 1 || p.n_qkv->ne[2] != 1 || p.n_qkv->ne[3] != 1) {
            continue;
        }
        int maxi = 0;
        const ggml_tensor * need[] = { p.n_qkv, p.n_z, p.n_gate, p.n_beta,
                                       p.n_convst, p.n_sstate };
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
        p.i_fire = maxi + 1;
        p.n_hres = p.n_resid->src[1];
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
        p.i_cap_qkv  = idxof(p.n_qkv);
        p.i_cap_conv = idxof(p.n_convst);
        p.i_cap_ss   = idxof(p.n_sstate);
        p.i_cap_hres = idxof(p.n_hres);
        if (p.i_cap_qkv < 0 || p.i_cap_conv < 0 || p.i_cap_ss < 0) {
            continue;   // producers not in this chunk: nothing to snapshot
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

// Find (or create) the fused session for a recurrent block and a sequence.
static xdna_rec_session * xdna_rec_session_get(ggml_backend_xdna_context * ctx,
                                               int il, int64_t row) {
    const auto key = std::make_pair(il, row);
    auto it = ctx->rec.find(key);
    if (it != ctx->rec.end()) {
        return it->second;
    }
    xdna_rec_session * s = new xdna_rec_session;
    s->il = il;
    ctx->rec[key] = s;
    return s;
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
static bool xdna_rec_run(ggml_backend_xdna_context * ctx, xdna_rec_plan & p,
                         std::unordered_set<const ggml_tensor *> & consumed,
                         const ggml_cgraph * cgraph) {
    using namespace xdna_rec_pack;
    xdna_rec_session * s = xdna_rec_session_get(ctx, p.il, xdna_rec_seq_row(p));
    if (!s) {
        return false;
    }

    // qkv / conv / ssm seed / residual are transient graph tensors that llama
    // recycled by the time this fire runs; the dispatch loop snapshotted their
    // producer outputs into s->cap_* right after each executed. gate/z/beta are
    // produced at the very end of the layer (immediately before fire) and stay
    // valid, so they are read live from their tensors.
    const float * qkv = !s->cap_qkv.empty() ? s->cap_qkv.data()
                                            : xdna_rec_tdata(p.n_qkv, CH);
    const float * z   = xdna_rec_tdata(p.n_z, NVH * SV);
    const float * gate = xdna_rec_tdata(p.n_gate, NVH);
    const float * beta = xdna_rec_tdata(p.n_beta, NVH);
    if (!qkv || !z || !gate || !beta) {
        GGML_LOG_ERROR("%s: fused layer %d: bad input tensors\n", "ggml-xdna", p.il);
        return false;
    }

    if (!s->seeded) {
        const float * ch = !s->cap_conv.empty() ? s->cap_conv.data()
                                                : xdna_rec_tdata(p.n_convst, 3 * CH);
        const float * st = !s->cap_ss.empty() ? s->cap_ss.data()
                                              : xdna_rec_tdata(p.n_sstate, state_floats());
        if (!ch || !st) {
            GGML_LOG_ERROR("%s: fused layer %d: no recurrent state to seed\n", "ggml-xdna", p.il);
            return false;
        }
        s->host.il = p.il;
        s->host.w_conv      = (const float *) p.w_conv->data;
        s->host.w_ssm_norm  = (const float *) p.w_gamma->data;
        s->host.w_post_norm = (const float *) p.w_post->data;

        // The whole recurrent layer runs on the fused design only when the
        // quantized weight set matches its native layouts (Q4_K/Q5_K/Q6_K
        // ssm_out, Q4_K gate/up, Q4_K/Q6_K down). A mismatch is a hard error
        // here: the fused path is the only NPU route for the recurrent layers,
        // there is no scalar fused fallback.
        if ((p.w_so->type != GGML_TYPE_Q4_K && p.w_so->type != GGML_TYPE_Q5_K &&
             p.w_so->type != GGML_TYPE_Q6_K) ||
            p.w_gate->type != GGML_TYPE_Q4_K ||
            p.w_up->type   != GGML_TYPE_Q4_K ||
            (p.w_down->type != GGML_TYPE_Q4_K && p.w_down->type != GGML_TYPE_Q6_K)) {
            GGML_LOG_ERROR("%s: fused layer %d: unsupported fused weight set "
                           "(so=%s gate=%s up=%s down=%s)\n", "ggml-xdna", p.il,
                           ggml_type_name(p.w_so->type), ggml_type_name(p.w_gate->type),
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
            GGML_LOG_ERROR("%s: fused layer %d: ssm_out is %s, which needs the "
                           "%d-bit activation layout, but the kernels were built "
                           "with GATED_FMT=%d; rebuild them and the backend with "
                           "-DGGML_XDNA_GATED_FMT=%d\n", "ggml-xdna", p.il,
                           ggml_type_name(p.w_so->type), want_fmt == 0 ? 4 : 8,
                           xdna_rec_pack::GATED_FMT, want_fmt);
            return false;
        }

        xdna_rec_pack_begin(s->host, ch, qkv, s->feed0);

        const std::string & gx = xdna_fused_xclbin();
        if (gx.empty()) {
            GGML_LOG_ERROR("%s: fused layer %d: no fused_layer artifact for this "
                           "design\n", "ggml-xdna", p.il);
            return false;
        }
        xdna_rec_geom g = xdna_rec_pack_geom();
        // The gated out object's size follows the compiled layout (xdna-rec.h).
        g.out_bytes = (int64_t) xdna_rec_pack::OUTN;

        s->core = xdna_rec_core_create(ctx->device, g, gx.c_str());
        if (!s->core) {
            GGML_LOG_ERROR("%s: fused layer %d: fused_layer load failed\n", "ggml-xdna", p.il);
            return false;
        }
        s->gv = xdna_rec_gemv_create(ctx->pool, p.w_so, p.w_gate, p.w_up,
                                     p.w_down);
        if (!s->gv) {
            GGML_LOG_ERROR("%s: fused layer %d: the GEMV route does not cover "
                           "this weight set\n", "ggml-xdna", p.il);
            return false;
        }
        if (!xdna_rec_core_begin(s->core, s->feed0.data(), s->host.w_ssm_norm)) {
            return false;
        }
        // One dispatch for the core and the projection that reads what its
        // gated stage writes. Both are already the same array configuration
        // and the same hardware context; this makes them one stream, so the
        // array pays its fixed per-phase cost once instead of twice.
        xdna_rec_core_fuse_so(s->core, ctx->pool, xdna_rec_gemv_so(s->gv),
                              s->gv->ffn);
        if (!xdna_rec_core_seed(s->core, st)) {
            return false;
        }
        s->seeded = true;
        s->x.resize((size_t) g.x_bytes);
        s->hattn.resize((size_t) g.d_out);
        s->hout.resize((size_t) g.d_out);
        s->hff.resize((size_t) g.d_out);
        s->aq.resize(KGATE);
    } else if (s->reseed) {
        const float * ch = !s->cap_conv.empty() ? s->cap_conv.data()
                                                : xdna_rec_tdata(p.n_convst, 3 * CH);
        const float * st = !s->cap_ss.empty() ? s->cap_ss.data()
                                              : xdna_rec_tdata(p.n_sstate, state_floats());
        if (!ch || !st) {
            GGML_LOG_ERROR("%s: fused layer %d: no recurrent state to re-seed\n",
                           "ggml-xdna", p.il);
            return false;
        }
        // Only the state: the core, the GEMV and the fused stream are the same
        // design and stay as they are. xdna_rec_core_begin re-uploads the feed
        // object and rewinds the token counter, it allocates nothing.
        xdna_rec_pack_begin(s->host, ch, qkv, s->feed0);
        if (!xdna_rec_core_begin(s->core, s->feed0.data(), s->host.w_ssm_norm)) {
            return false;
        }
        if (!xdna_rec_core_seed(s->core, st)) {
            return false;
        }
        s->reseed = false;
    }

    const float * hres = !s->cap_hres.empty() ? s->cap_hres.data()
                                              : xdna_rec_tdata(p.n_hres, D_OUT);
    if (!hres) {
        GGML_LOG_ERROR("%s: fused layer %d: bad residual input\n", "ggml-xdna", p.il);
        return false;
    }

    xdna_rec_pack_x(gate, beta, s->x);
    const int t = s->core->token;

    // With the projection draining into the FFN's tiles, the host's half of
    // that activation - the residual and gamma - is written before the core
    // runs, not between the two dispatches. Both are known by now, and this is
    // what leaves nothing between them.
    const bool to_act_pre = s->gv->ffn && xdna_rec_core_so_to_act(s->core);
    if (to_act_pre && xdna_rec_core_ffn_fused(s->core) &&
        !xdna_gemv_pair_prep_raw(s->gv->ffn, hres, s->host.w_post_norm)) {
        return false;
    }
    if (!xdna_rec_core_run(s->core, t, t == 0 ? nullptr : qkv,
                           s->x.data(), z, s->aq.data(), &s->d_a)) {
        return false;
    }
    const bool so_in_core = xdna_rec_core_so_fused(s->core);
    // When the projection drained into the FFN's activation tiles there is
    // nothing to collect: its result is already where the next dispatch reads
    // it, and the host only needs it again for the layer's last residual add,
    // which happens after both dispatches rather than between them.
    const bool so_to_act = so_in_core && xdna_rec_core_so_to_act(s->core);
    if (so_to_act) {
        // Nothing: as above.
    } else if (so_in_core ? !xdna_rec_gemv_so_collect(s->gv, xdna_rec_core_out(s->core),
                                                      xdna_rec_core_out_off(s->core),
                                                      xdna_rec_core_act(s->core),
                                                      hres, s->hattn.data())
                          : !xdna_rec_gemv_so_run(s->gv, xdna_rec_core_act(s->core),
                                                  s->aq.data(), s->d_a, hres,
                                                  s->hattn.data())) {
        return false;
    }
    if (s->gv->ffn) {
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
            if (xdna_rec_core_ffn_fused(s->core)) {
                if (!xdna_gemv_pair_out_from_tail(s->gv->ffn, ffn_out.data())) {
                    return false;
                }
            } else {
                if (!xdna_gemv_pair_prep_raw(s->gv->ffn, hres,
                                             s->host.w_post_norm)) {
                    return false;
                }
                if (!xdna_gemv_pair_dispatch(s->gv->ffn, ffn_out.data())) {
                    return false;
                }
            }
            if (!xdna_rec_gemv_acc_from_tiles(s->gv, s->gv->acc.data())) {
                return false;
            }
            for (int i = 0; i < D_OUT; i++) {
                s->hattn[i] = hres[i] + s->gv->acc[i];
                s->hout[i]  = s->hattn[i] + ffn_out[i];
            }
        } else if (!xdna_rec_gemv_ffn_run_raw(s->gv, s->gv->acc.data(), hres,
                                              s->host.w_post_norm,
                                              s->hattn.data(), s->hout.data())) {
            return false;
        }
    } else {
        xdna_rec_rms_norm(s->hattn.data(), s->host.w_post_norm, D_OUT, 1e-6f, s->hff.data());
        if (!xdna_rec_gemv_ffn_run(s->gv, s->hff.data(), s->hattn.data(),
                                   s->hout.data())) {
            return false;
        }
    }

    // Write the fused outputs into the layer add tensors every downstream op
    // reads (l_out feeds the next layer/MHA, attn_residual its own FFN path).
    std::memcpy(p.n_resid->data, s->hattn.data(), D_OUT * sizeof(float));
    std::memcpy(p.n_lout->data,  s->hout.data(),  D_OUT * sizeof(float));

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
static void xdna_kernels_warm(xdna_kernel_pool * pool, const xdna_ops * ops) {
    if (!pool || !pool->device) {
        return;
    }
    const auto warm_stem = [&](const std::string & stem) {
        if (stem.empty()) {
            return;
        }
        for (const auto & dir : xdna_kernel_search_dirs()) {
            std::error_code ec;
            const std::string x = (dir / (stem + ".xclbin")).string();
            if (std::filesystem::exists(x, ec)) {
                xdna_kernel_load_hw(pool->device, x.c_str());
                return;
            }
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
    static std::once_flag once;

    std::call_once(once, [&]() {
        ctx.device = xdna_device_open();
        if (ctx.device) {
            ctx.pool = new xdna_kernel_pool;
            ctx.pool->device = ctx.device;
            xdna_kernel_pool_scan(ctx.pool);
            xdna_ops_init(&ctx.ops, ctx.pool);
            if (ctx.ops.gemm_xclbin_decode.empty() && ctx.ops.pref_xclbin.empty()) {
                GGML_LOG_WARN("%s: no GEMM kernels found (build with GGML_XDNA=ON)\n", "ggml-xdna");
            }
            // Register the kernels this backend will run up front, while the
            // NPU is idle (see xdna_kernels_warm).
            xdna_kernels_warm(ctx.pool, &ctx.ops);
        }
    });

    return &ctx;
}

// backend interface

static const char * ggml_backend_xdna_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return "XDNA";
}

static void ggml_backend_xdna_free(ggml_backend_t backend) {
    // The context is a process-wide singleton; it is not freed here.
    GGML_UNUSED(backend);
    delete backend;
}

// CONCAT nodes the conv prefill reads through instead of building (see
// xdna_conv_prefill_direct_add): the conv input is the cached tokens with the
// chunk's projection, and materialising it costs an element-at-a-time copy of
// the whole thing.
static void xdna_concat_tail_plan(ggml_backend_xdna_context * ctx,
                                  struct ggml_cgraph * cgraph,
                                  std::unordered_map<const ggml_tensor *,
                                                     struct ggml_tensor *> & out) {
    // No do/while wrapper: the `continue` has to leave the node loop.
#define CF_NO(why) { GGML_UNUSED(why); continue; }
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
        int j = -1;
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
        const char * const blo = (const char *) bsrc->data;
        const char * const dlo = (const char *) conv->data;
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
                continue;   // writes nothing of its own
            }
            const char * const mlo = (const char *) m->data;
            clobbered = mlo < dlo + ggml_nbytes(conv) && dlo < mlo + ggml_nbytes(m);
        }
        if (clobbered) {
            CF_NO("the conv result is overwritten before its node is reached");
        }
        const int64_t keep     = a->ne[0];             // conv history rows
        const size_t  tail_off = (size_t) (n->ne[0] - keep) * n->nb[0];
        bool ok = keep > 0 && n->ne[0] > keep;
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
            ok = ggml_xdna_is_view_op(c->op) && c->data != nullptr &&
                 c->ne[0] <= keep && off >= tail_off;
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
static void xdna_concat_tail_fill(struct ggml_tensor * n,
                                  std::vector<float> & hist) {
    const ggml_tensor * a = n->src[0];
    const ggml_tensor * b = n->src[1];
    const int64_t t0 = n->ne[0] - a->ne[0];
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
            const ggml_tensor * s = i0 < a->ne[0] ? a : b;
            const int64_t si = i0 < a->ne[0] ? i0 : i0 - a->ne[0];
            const float v = *(const float *) ((const char *) s->data +
                                              si * s->nb[0] + i1 * s->nb[1]);
            *(float *) ((char *) n->data + i0 * n->nb[0] + i1 * n->nb[1]) = v;
        }
    }
}

static enum ggml_status ggml_backend_xdna_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_xdna_context * ctx = (ggml_backend_xdna_context *) backend->context;

    if (!ctx->device) {
        return GGML_STATUS_SUCCESS;
    }

    // Size the host-fallback pool for this chunk (xdna_glue_threads).
    xdna_glue_set_n_tokens(cgraph);

    // A prefill ubatch rewrites the recurrent cache, so every live session has
    // to take its state from there again on the next token (xdna_rec_session).
    if (g_glue_n_tokens > 1) {
        for (auto & kv : ctx->rec) {
            kv.second->reseed = true;
        }
    }

    // Per-op NPU kernels are suppressed while the fused layer path is active
    // (default when the fused xclbins are present): the fused run owns the
    // recurrent layers and everything else falls back to the CPU glue.
    ctx->ops.isolation = xdna_rec_active();
    // Projections outside the fused layers can go through the decode GEMV on
    // the merged artifact, which is the context the layers leave resident.
    // They stay on the host by default: that route is both faster (about 38
    // against 30 t/s on the 0.8B, because the packed format moves more bytes
    // at a lower rate) and the only one that has not been caught diverging.
    // Over 120 runs the array route still produced two whose logits differ
    // from the rest, against none in 48 on the host route. The switch keeps
    // the array route reachable for anyone measuring NPU residency or power,
    // where the extra work belongs on the device; that arm is not reproducible
    // today.
    ctx->ops.fused_gemv = false;
    if (xdna_env_int("GGML_XDNA_GEMV_GROUP", 0) != 0) {
        ctx->ops.fused_gemv = ctx->ops.isolation && !xdna_fused_xclbin().empty();
    }

    // Fused decode linear layers: scan this chunk for complete recurrent
    // layers of a single-token decode; each fires once in the dispatch loop
    // below and replaces its conv/gdn/ffn subgraph with one persistent device
    // run.
    std::vector<xdna_rec_plan> rec_plans;
    if (xdna_rec_active()) {
        xdna_rec_scan(ctx, cgraph, rec_plans);
    }

    // Prefill conv inputs whose concat the array reads through instead of
    // building.
    std::unordered_map<const ggml_tensor *, struct ggml_tensor *> concat_tail;
    // Owns the conv-history snapshots for the length of the graph; deque so
    // the pointers handed to the conv runner survive later pushes.
    std::deque<std::vector<float>> concat_hist;
    xdna_conv_prefill_direct_reset();
    xdna_concat_tail_plan(ctx, cgraph, concat_tail);

    // Input snapshot points of the fused plans: when the loop reaches one of
    // these producer nodes it runs the node (if not already consumed by an
    // earlier fused run) and copies the output into the layer session, so the
    // fused fire later reads valid data instead of llama's recycled buffers.
    // kind: 0 = qkv, 1 = conv seed, 2 = ssm seed, 3 = residual h.
    struct rec_capture {
        int i;
        xdna_rec_plan * p;
        int kind;
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
        std::unordered_set<const ggml_tensor *> needed;
        std::vector<const ggml_tensor *> stack = {
            pl.n_qkv, pl.n_z, pl.n_gate, pl.n_beta, pl.n_convst, pl.n_sstate,
        };
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
        for (int j = pl.i_conv; j < pl.i_fire; j++) {
            struct ggml_tensor * n = cgraph->nodes[j];
            if (!ggml_xdna_is_view_op(n->op) && n->op != GGML_OP_NONE &&
                needed.count(n) == 0) {
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
    xdna_ops_plan_gemv(&ctx->ops, cgraph, &gemv_skip);

    std::vector<struct ggml_tensor *> glue_batch;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        // Fused-layer input snapshot: right when a producer node runs, copy its
        // output into the layer session before llama recycles the buffer. A
        // consumed node was already produced (e.g. the fused write of a
        // previous layer) and is only copied. Views need no snapshot.
        for (auto & c : rec_caps) {
            if (c.i != i || ggml_xdna_is_view_op(node->op)) {
                continue;
            }
            xdna_rec_session * s =
                xdna_rec_session_get(ctx, c.p->il, xdna_rec_seq_row(*c.p));
            if (s && consumed.count(node) == 0) {
                if (!glue_batch.empty() && !xdna_glue_flush(&ctx->ops, glue_batch)) {
                    return GGML_STATUS_FAILED;
                }
                // A snapshot input the GEMV claimed runs there; the copy below
                // reads its output the same way either way.
                const bool on_npu = xdna_ops_supported(&ctx->ops, node);
                if (!(on_npu ? xdna_ops_compute(&ctx->ops, node)
                             : xdna_glue_host_run(&ctx->ops, node))) {
                    GGML_LOG_ERROR("%s: fused layer %d: cannot snapshot %s\n",
                                   "ggml-xdna", c.p->il, node->name ? node->name : "?");
                    return GGML_STATUS_FAILED;
                }
                if (on_npu && !xdna_ops_finalize(&ctx->ops)) {
                    return GGML_STATUS_FAILED;
                }
            }
            if (s && node->type == GGML_TYPE_F32 && node->data) {
                switch (c.kind) {
                    case 0: // qkv
                        s->cap_qkv.resize(xdna_rec_pack::CH);
                        std::memcpy(s->cap_qkv.data(), node->data,
                                    xdna_rec_pack::CH * sizeof(float));
                        break;
                    case 1: // conv seed (consumed when seeding or re-seeding)
                        if (!s->seeded || s->reseed) {
                            s->cap_conv.resize(3 * xdna_rec_pack::CH);
                            std::memcpy(s->cap_conv.data(), node->data,
                                        3 * xdna_rec_pack::CH * sizeof(float));
                        }
                        break;
                    case 2: { // ssm seed (only consumed at the one-time seed)
                        const int64_t n = xdna_rec_pack::state_floats();
                        if (!s->seeded || s->reseed) {
                            s->cap_ss.resize((size_t) n);
                            std::memcpy(s->cap_ss.data(), node->data, (size_t) n * sizeof(float));
                        }
                        break;
                    }
                    case 3: // residual h
                        s->cap_hres.resize(xdna_rec_pack::D_OUT);
                        std::memcpy(s->cap_hres.data(), node->data,
                                    xdna_rec_pack::D_OUT * sizeof(float));
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
                    if (!xdna_rec_run(ctx, pl, consumed, cgraph)) {
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
        if (consumed.count(node)) {
            continue;
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
                if (!xdna_ops_compute(&ctx->ops, cf->second) ||
                    !xdna_ops_finalize(&ctx->ops)) {
                    xdna_ops_finalize(&ctx->ops);
                    return GGML_STATUS_FAILED;
                }
                consumed.insert(cf->second);
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
            if (node->op == GGML_OP_CONCAT) {
                if (!glue_batch.empty()) {
                    if (!xdna_glue_flush(&ctx->ops, glue_batch)) {
                        return GGML_STATUS_FAILED;
                    }
                }
                if (!xdna_ops_finalize(&ctx->ops)) {
                    return GGML_STATUS_FAILED;
                }
                if (xdna_concat_fast(node)) {
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
    return GGML_STATUS_SUCCESS;
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
    static ggml_guid guid = { 0x7a, 0xff, 0x0d, 0x7e, 0x5a, 0x1b, 0x4c, 0xf6, 0xbd, 0x94, 0x6e, 0xc1, 0xf1, 0xcb, 0x3f, 0xbd };
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

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_xdna_guid(),
        /* .iface   = */ xdna_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_xdna_reg(), 0),
        /* .context = */ ctx,
    };

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
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    *total = (size_t) pages * page_size;
    // "free" system memory is ill-defined, for practical purposes assume all of it is free:
    *free = *total;
#endif // _WIN32
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

static ggml_backend_buffer_type_t ggml_backend_xdna_device_get_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cpu_buffer_type();
}

static ggml_backend_buffer_t ggml_backend_xdna_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
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
