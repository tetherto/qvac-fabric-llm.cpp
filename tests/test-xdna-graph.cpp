// Graphs the XDNA backend rewrites, computed on the NPU and on the CPU backend
// and compared node by node.
//
//   test-xdna-graph
//
// A prefill layer's residual sum into its normalized projection, ADD,
// RMS_NORM, MUL, MUL_MAT: the backend lays the norm out as the projection's A
// at the MUL and can compute the ADD in the same pass, skipping the ADD where
// it stands. With a second reader of the sum ordered before the norm, ADD,
// SQR, RMS_NORM, MUL, MUL_MAT, that reader must still see the sum - it saw
// whatever the sum's memory held before.
//
// A one-token projection computed again with new activations, on the fused
// route (the decode GEMV) and the per-op one (GGML_XDNA_FUSED_LAYER=0, the
// prefill GEMM): a later graph must not reuse the A an earlier one laid out
// for the same tensor.
//
// A prefill FLASH_ATTN_EXT in the shapes llama builds - the 8 query heads of
// Qwen3.5 over a causal cache, and a bidirectional or banded attention of 4
// query heads with packed sequences, K and V cast to f16 - computed on the
// array and on the CPU backend. The array answers to ~2e-4 NMSE and the host
// path bit for bit, so the two are told apart: a mask the design cannot follow,
// or a call too small (the size rule, tried either side of both its bounds),
// has to come out equal.
//
// BF16 and F16 weight projections, which no quantized route carries, through
// the bf16 GEMM: an M that fits the decode block, one that takes several
// prefill blocks, one whose last block is partial and one whose last block
// holds fewer rows than a sweep of its kernel (it runs a shorter stream); a K
// of one run, of several and one that ends in a partial run; and, in one
// graph, projections of different K and N one after the other (the cores hang
// when consecutive runs differ in their K-loop count).
//
// A prefill projection of a row count that is not a whole 128-row block, its
// tensors in the backend's own buffers: the array writes the output in place,
// in whole blocks, into rows the allocation reserves for it, and the tensor
// that follows it is untouched.
//
// Projections of one activation that follow each other (Q, K and V), in the
// backend's own buffers: one call writes each output into its own tensor, in
// whole blocks of 128 rows, and the tensor that follows them is untouched.
//
// Projections of one activation that ggml orders apart, each followed by its
// norm (a layer's Q, K and V): the backend, as a scheduler's graph_optimize,
// moves them next to each other so that one call can run them. The scheduler
// computes the graph, and so does the CPU. A projection stays where it is when
// a node between writes into its activation in place, or when its weight is
// a node of the graph computed after the first projection.
//
// A prompt's FFN, down(GLU(gate(x), up(x))) with SWIGLU and with GEGLU: the
// backend runs the gate and up projections and the activation as one call,
// writing the product straight into the down projection's input; with the
// GLU a graph output it writes the product itself. Activations are scaled
// to put the gate on the activation's curve and, larger, in its tails. Two
// sizes: EmbeddingGemma 2's (512, 2048) and Qwen3.5-4B's (2560, 9216).
//
// Every intermediate is cleared before each backend runs, so a node read
// before it was written shows as zeros instead of as the other backend's
// result. Skips (exit 0) when no XDNA device is registered.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

static ggml_backend_dev_t xdna_device(void) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "XDNA") == 0) {
            return dev;
        }
    }
    return nullptr;
}

static std::vector<float> get_f32(const ggml_tensor * t) {
    std::vector<float> v(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    return v;
}

// normalized mean squared error, as test-backend-ops measures it
static double nmse(const std::vector<float> & ref, const std::vector<float> & got) {
    double err = 0, sq = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        if (!std::isfinite(got[i])) {
            return INFINITY;
        }
        err += ((double) ref[i] - got[i]) * ((double) ref[i] - got[i]);
        sq += (double) ref[i] * ref[i];
    }
    return err / sq;
}

struct norm_graph {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_cgraph *         gf  = nullptr;
    std::vector<ggml_tensor *> inputs;
    std::vector<ggml_tensor *> checked;  // compared, in graph order
    float                      act_scale = 1.0f;  // of the f32 inputs
};

// K x M activations, an N x K Q4_K projection; side_reader adds SQR(sum)
// through a second projection, built first so it is ordered before the norm
static norm_graph build(ggml_backend_t cpu, bool side_reader, int64_t K, int64_t N, int64_t M) {
    norm_graph g;
    ggml_init_params ip = { ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true };
    g.ctx               = ggml_init(ip);

    ggml_tensor * a  = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * b  = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * w  = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, K);
    ggml_tensor * p  = ggml_new_tensor_2d(g.ctx, GGML_TYPE_Q4_K, K, N);
    ggml_tensor * p2 = ggml_new_tensor_2d(g.ctx, GGML_TYPE_Q4_K, K, N);
    g.inputs         = { a, b, w, p, p2 };

    ggml_tensor * sum = ggml_add(g.ctx, a, b);
    ggml_set_name(sum, "sum");
    ggml_tensor * side = nullptr;
    ggml_tensor * sqr  = nullptr;
    if (side_reader) {
        sqr = ggml_sqr(g.ctx, sum);
        ggml_set_name(sqr, "sqr");
        side = ggml_mul_mat(g.ctx, p2, sqr);
        ggml_set_name(side, "side");
    }
    ggml_tensor * norm = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, sum, 1e-6f), w);
    ggml_set_name(norm, "norm");
    ggml_tensor * out = ggml_mul_mat(g.ctx, p, norm);
    ggml_set_name(out, "proj");
    if (side) {
        out = ggml_add(g.ctx, side, out);
        ggml_set_name(out, "out");
    }

    g.gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, out);
    g.checked = { sum };
    if (side_reader) {
        g.checked.push_back(sqr);
        g.checked.push_back(side);
    }
    g.checked.push_back(out);

    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, cpu);
    return g;
}

// the activations from `seed`, the weights always the same: a model's weights
// do not change under the address they were packed from
static void set_inputs(norm_graph & g, uint32_t seed) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (size_t i = 0; i < g.inputs.size(); i++) {
        ggml_tensor *      t = g.inputs[i];
        std::mt19937       rng((t->type == GGML_TYPE_F32 ? seed : 1234) * 16 + (uint32_t) i);
        std::vector<float> f(ggml_nelements(t));
        for (float & x : f) {
            x = u(rng) * (t->type == GGML_TYPE_F32 ? g.act_scale : 1.0f);
        }
        if (t->type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(t, f.data(), 0, ggml_nbytes(t));
        } else {
            std::vector<uint8_t> q(ggml_nbytes(t));
            ggml_quantize_chunk(t->type, f.data(), q.data(), 0, t->ne[1], t->ne[0], nullptr);
            ggml_backend_tensor_set(t, q.data(), 0, q.size());
        }
    }
}

// clear everything, then the inputs, and compute on `be`
static bool run(norm_graph & g, ggml_backend_t be, std::vector<std::vector<float>> & res, uint32_t seed = 1234) {
    ggml_backend_buffer_clear(g.buf, 0);
    set_inputs(g, seed);
    if (ggml_backend_graph_compute(be, g.gf) != GGML_STATUS_SUCCESS) {
        return false;
    }
    res.clear();
    for (const ggml_tensor * t : g.checked) {
        res.push_back(get_f32(t));
    }
    return true;
}

// one Q4_K projection of a single token, the same graph object every run
static norm_graph build_one_token(ggml_backend_t cpu, int64_t K, int64_t N) {
    norm_graph g;
    ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
    g.ctx               = ggml_init(ip);
    ggml_tensor * x     = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, K, 1);
    ggml_tensor * p     = ggml_new_tensor_2d(g.ctx, GGML_TYPE_Q4_K, K, N);
    g.inputs            = { x, p };
    ggml_tensor * out   = ggml_mul_mat(g.ctx, p, x);
    ggml_set_name(out, "proj");
    g.gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, out);
    g.checked = { out };
    g.buf     = ggml_backend_alloc_ctx_tensors(g.ctx, cpu);
    return g;
}

enum fa_mask { FA_CAUSAL, FA_FULL, FA_BAND, FA_HOLES };

// GGML_XDNA_FA_MM: the host, the array from the size rule's bound, the array whatever the size
enum fa_route { FA_ROUTE_HOST = 0, FA_ROUTE_BY_SIZE = 1, FA_ROUTE_ARRAY = 2 };

struct fa_case {
    const char * name;
    fa_route     route;
    int          heads;
    int          n_tokens;
    int          n_ctx;       // keys of earlier context before the tokens (causal)
    int          pad;         // the keys round up to a multiple of it, hidden
    int          seqs;        // equal sequences packed in the ubatch
    fa_mask      mask;
    int          window;      // FA_BAND: keys on each side
    float        scale;
    bool         cache_view;  // K and V as views of a cache, not a cast
    bool         on_array;    // expected to run on the array
};

static const int FA_KV_HEADS = 2;
static const int FA_HEAD_DIM = 256;

// The graph the model builds around one attention: a projection of n_tokens
// beside it, so the backend sees a prefill graph and starts a new mask epoch.
static norm_graph build_fa(ggml_backend_t cpu, const fa_case & c) {
    norm_graph g;
    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    g.ctx               = ggml_init(ip);
    const int64_t n_kv  = ((int64_t) c.n_ctx + c.n_tokens + c.pad - 1) / c.pad * c.pad;
    const int64_t D     = FA_HEAD_DIM;

    ggml_tensor * qr = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, D, c.heads, c.n_tokens);
    ggml_tensor * q  = ggml_permute(g.ctx, qr, 0, 2, 1, 3);
    ggml_tensor * kr, *vr, *k, *v;
    if (c.cache_view) {
        kr = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F16, D, FA_KV_HEADS, n_kv);
        vr = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F16, D, FA_KV_HEADS, n_kv);
        k  = ggml_permute(g.ctx, kr, 0, 2, 1, 3);
        v  = ggml_permute(g.ctx, vr, 0, 2, 1, 3);
    } else {
        kr = k = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F16, D, n_kv, FA_KV_HEADS);
        vr = v = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F16, D, n_kv, FA_KV_HEADS);
    }
    ggml_tensor * m  = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F16, n_kv, c.n_tokens);
    ggml_tensor * dw = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, 64, 64);
    ggml_tensor * dx = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, 64, c.n_tokens);
    g.inputs         = { qr, kr, vr, m, dw, dx };

    ggml_tensor * out = ggml_flash_attn_ext(g.ctx, q, k, v, m, c.scale, 0.0f, 0.0f);
    ggml_prec_set_acc(out, GGML_PREC_F32);
    ggml_set_name(out, "attn");
    g.gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, out);
    ggml_build_forward_expand(g.gf, ggml_mul_mat(g.ctx, dw, dx));
    g.checked = { out };
    g.buf     = ggml_backend_alloc_ctx_tensors(g.ctx, cpu);
    return g;
}

// which keys a token sees: the same sequence, and the mask's own rule
static bool fa_visible(const fa_case & c, int t, int j) {
    const int len = c.n_tokens / c.seqs;
    const int s_t = std::min(t / len, c.seqs - 1);
    // keys before n_ctx are the first sequence's earlier positions
    const bool ctx_key = j < c.n_ctx;
    const int  jt      = j - c.n_ctx;
    if (!ctx_key && jt >= c.n_tokens) {
        return false;
    }
    const int s_j  = ctx_key ? 0 : std::min(jt / len, c.seqs - 1);
    const int p_t  = t - s_t * len + (s_t == 0 ? c.n_ctx : 0);
    const int p_j  = ctx_key ? j : jt - s_j * len + (s_j == 0 ? c.n_ctx : 0);
    if (s_t != s_j) {
        return false;
    }
    switch (c.mask) {
        case FA_CAUSAL: return p_j <= p_t;
        case FA_FULL:   return true;
        case FA_BAND:   return std::abs(p_j - p_t) <= c.window;
        case FA_HOLES:  return j % 2 == 0;
    }
    return false;
}

static void set_fa_inputs(norm_graph & g, const fa_case & c) {
    std::mt19937 rng(77);
    // q and k are small, as a model that normalises them leaves them: a score of a few
    std::normal_distribution<float> nd(0.0f, c.scale < 0.5f ? 1.0f : 0.25f);
    for (size_t i = 0; i < 3; i++) {
        ggml_tensor *      t = g.inputs[i];
        std::vector<float> f(ggml_nelements(t));
        for (float & x : f) {
            x = nd(rng);
        }
        if (t->type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(t, f.data(), 0, ggml_nbytes(t));
        } else {
            std::vector<ggml_fp16_t> h(f.size());
            ggml_fp32_to_fp16_row(f.data(), h.data(), (int64_t) f.size());
            ggml_backend_tensor_set(t, h.data(), 0, ggml_nbytes(t));
        }
    }
    ggml_tensor *            m = g.inputs[3];
    std::vector<ggml_fp16_t> mk((size_t) ggml_nelements(m), ggml_fp32_to_fp16(-INFINITY));
    for (int t = 0; t < c.n_tokens; t++) {
        for (int64_t j = 0; j < m->ne[0]; j++) {
            if (fa_visible(c, t, (int) j)) {
                mk[(size_t) t * m->ne[0] + j] = ggml_fp32_to_fp16(0.0f);
            }
        }
    }
    ggml_backend_tensor_set(m, mk.data(), 0, ggml_nbytes(m));
    for (size_t i = 4; i < g.inputs.size(); i++) {
        std::vector<float> f(ggml_nelements(g.inputs[i]), 0.5f);
        ggml_backend_tensor_set(g.inputs[i], f.data(), 0, ggml_nbytes(g.inputs[i]));
    }
}

static void set_fa_route(fa_route route) {
    setenv("GGML_XDNA_FA_MM", std::to_string((int) route).c_str(), 1);
}

static bool run_fa(norm_graph & g, const fa_case & c, ggml_backend_t be, std::vector<float> & res) {
    ggml_backend_buffer_clear(g.buf, 0);
    set_fa_inputs(g, c);
    if (ggml_backend_graph_compute(be, g.gf) != GGML_STATUS_SUCCESS) {
        return false;
    }
    res = get_f32(g.checked[0]);
    return true;
}

struct gemm_shape {
    int64_t K;
    int64_t N;
};

// The projections of `shapes` on M activation rows, in this order and with no
// other op between them. The activations are one tensor per distinct K.
static norm_graph build_gemms(ggml_backend_t cpu, enum ggml_type wtype, int64_t M, const std::vector<gemm_shape> & shapes) {
    norm_graph g;
    ggml_init_params ip = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true };
    g.ctx               = ggml_init(ip);
    g.gf                = ggml_new_graph(g.ctx);

    std::map<int64_t, ggml_tensor *> acts;
    for (const gemm_shape & sh : shapes) {
        ggml_tensor *& a = acts[sh.K];
        if (!a) {
            a = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, sh.K, M);
            g.inputs.push_back(a);
        }
        ggml_tensor * w = ggml_new_tensor_2d(g.ctx, wtype, sh.K, sh.N);
        g.inputs.push_back(w);
        ggml_tensor * out = ggml_mul_mat(g.ctx, w, a);
        ggml_set_name(out, (std::string("K") + std::to_string(sh.K) + " N" + std::to_string(sh.N)).c_str());
        ggml_build_forward_expand(g.gf, out);
        g.checked.push_back(out);
    }
    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, cpu);
    return g;
}

struct gemm_case {
    const char *            name;
    enum ggml_type          wtype;
    int64_t                 M;
    std::vector<gemm_shape> shapes;
};

// An N x K Q4_K projection of M tokens followed by a tensor of its own to guard,
// allocated in `be`'s buffers
static norm_graph build_projection(ggml_backend_t be, int64_t K, int64_t N, int64_t M, ggml_tensor ** guard) {
    norm_graph g;
    ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
    g.ctx               = ggml_init(ip);
    ggml_tensor * a     = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * p     = ggml_new_tensor_2d(g.ctx, GGML_TYPE_Q4_K, K, N);
    g.inputs            = { a, p };
    ggml_tensor * out   = ggml_mul_mat(g.ctx, p, a);
    ggml_set_name(out, "proj");
    *guard = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, 2 * 1024);
    g.gf   = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, out);
    g.checked = { out };
    g.buf     = ggml_backend_alloc_ctx_tensors(g.ctx, be);
    return g;
}

static bool test_output_in_place(ggml_backend_t npu, ggml_backend_t cpu, int64_t M, double tol) {
    const int64_t K = 1024, N = 1024;
    ggml_tensor * cpu_guard = nullptr;
    ggml_tensor * npu_guard = nullptr;
    norm_graph    want_g    = build_projection(cpu, K, N, M, &cpu_guard);
    norm_graph    got_g     = build_projection(npu, K, N, M, &npu_guard);
    ggml_tensor * out       = got_g.checked[0];

    const size_t room     = ggml_backend_buffer_get_alloc_size(got_g.buf, out);
    const bool   reserved = room >= (size_t) ((M + 127) / 128 * 128) * N * sizeof(float);

    std::vector<float> pattern(ggml_nelements(npu_guard), 3.0f);
    std::vector<std::vector<float>> want, got;
    bool ok = run(want_g, cpu, want);
    ggml_backend_buffer_clear(got_g.buf, 0);
    set_inputs(got_g, 1234);
    ggml_backend_tensor_set(npu_guard, pattern.data(), 0, ggml_nbytes(npu_guard));
    ok = ok && ggml_backend_graph_compute(npu, got_g.gf) == GGML_STATUS_SUCCESS;
    std::vector<float> after(pattern.size());
    ggml_backend_tensor_get(npu_guard, after.data(), 0, ggml_nbytes(npu_guard));
    if (ok) {
        got = { get_f32(out) };
    }
    const bool   intact = after == pattern;
    const double e      = ok ? nmse(want[0], got[0]) : INFINITY;
    const bool   passed = ok && reserved && intact && e < tol;
    std::printf("MUL_MAT in place, M %-4lld alloc %zu of %zu bytes %s, guard %s, nmse %.3g %s\n", (long long) M, room,
                ggml_nbytes(out), reserved ? "reserved" : "NOT RESERVED", intact ? "intact" : "OVERWRITTEN", e,
                passed ? "ok" : "FAIL");
    ggml_backend_buffer_free(want_g.buf);
    ggml_backend_buffer_free(got_g.buf);
    ggml_free(want_g.ctx);
    ggml_free(got_g.ctx);
    return passed;
}

// One Q4_K weight applied to `count` activations of M tokens, each product in a node of its own
static norm_graph build_repeated(ggml_backend_t be, int64_t K, int64_t N, int64_t M, int count) {
    norm_graph g;
    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    g.ctx               = ggml_init(ip);
    ggml_tensor * p     = ggml_new_tensor_2d(g.ctx, GGML_TYPE_Q4_K, K, N);
    g.inputs            = { p };
    g.gf                = ggml_new_graph(g.ctx);
    for (int i = 0; i < count; i++) {
        ggml_tensor * a = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, K, M);
        g.inputs.push_back(a);
        ggml_tensor * out = ggml_mul_mat(g.ctx, p, a);
        ggml_build_forward_expand(g.gf, out);
        g.checked.push_back(out);
    }
    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, be);
    return g;
}

// The streams of the same call are kept; a repeat of its shape with another
// activation and another output place, in one graph and in the next pass,
// has to land in its own node
static bool test_repeated_projection(ggml_backend_t npu, ggml_backend_t cpu, int64_t M, int count, double tol) {
    const int64_t K = 1024, N = 1024;
    norm_graph    want_g = build_repeated(cpu, K, N, M, count);
    norm_graph    got_g  = build_repeated(npu, K, N, M, count);
    std::vector<std::vector<float>> want, got;
    bool ok = run(want_g, cpu, want);
    double worst = 0;
    for (int pass = 0; ok && pass < 2; pass++) {
        ok = run(got_g, npu, got);
        for (int i = 0; ok && i < count; i++) {
            worst = std::max(worst, nmse(want[i], got[i]));
        }
    }
    const bool passed = ok && worst < tol;
    std::printf("MUL_MAT x%d of one weight, M %-4lld two passes, worst nmse %.3g %s\n", count, (long long) M, worst,
                passed ? "ok" : "FAIL");
    ggml_backend_buffer_free(want_g.buf);
    ggml_backend_buffer_free(got_g.buf);
    ggml_free(want_g.ctx);
    ggml_free(got_g.ctx);
    return passed;
}

// x (K x M) through a gate and an up projection of F rows, joined by `op`
static norm_graph build_ffn(ggml_backend_t be, ggml_glu_op op, bool glu_is_output, int64_t K, int64_t F, int64_t M,
                            float scale) {
    norm_graph    g;
    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    g.ctx               = ggml_init(ip);
    g.act_scale         = scale;
    ggml_tensor * x     = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * wg    = ggml_new_tensor_2d(g.ctx, GGML_TYPE_Q8_0, K, F);
    ggml_tensor * wu    = ggml_new_tensor_2d(g.ctx, GGML_TYPE_Q8_0, K, F);
    ggml_tensor * wd    = ggml_new_tensor_2d(g.ctx, GGML_TYPE_Q8_0, F, K);
    g.inputs            = { x, wg, wu };
    ggml_tensor * gate  = ggml_mul_mat(g.ctx, wg, x);
    ggml_tensor * up    = ggml_mul_mat(g.ctx, wu, x);
    ggml_tensor * glu =
        op == GGML_GLU_OP_GEGLU ? ggml_geglu_split(g.ctx, gate, up) : ggml_swiglu_split(g.ctx, gate, up);
    ggml_set_name(glu, "glu");
    ggml_tensor * out = glu;
    if (glu_is_output) {
        ggml_set_output(glu);
    } else {
        g.inputs.push_back(wd);
        out = ggml_mul_mat(g.ctx, wd, glu);
        ggml_set_name(out, "down");
    }
    g.gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, out);
    g.checked = { out };
    g.buf     = ggml_backend_alloc_ctx_tensors(g.ctx, be);
    return g;
}

static bool test_ffn(ggml_backend_t npu, ggml_backend_t cpu, ggml_glu_op op, bool glu_is_output, int64_t K, int64_t F,
                     int64_t M, float scale, double tol) {
    norm_graph                      want_g = build_ffn(cpu, op, glu_is_output, K, F, M, scale);
    norm_graph                      got_g  = build_ffn(npu, op, glu_is_output, K, F, M, scale);
    std::vector<std::vector<float>> want, got;
    const bool                      ok = run(want_g, cpu, want) && run(got_g, npu, got);
    const double                    e  = ok ? nmse(want[0], got[0]) : INFINITY;
    const bool                      passed = ok && e < tol;
    std::printf("%-6s %-4s K %-5lld F %-5lld M %-4lld scale %-5.2f nmse %.3g %s\n",
                op == GGML_GLU_OP_GEGLU ? "GEGLU" : "SWIGLU", glu_is_output ? "glu" : "down", (long long) K,
                (long long) F, (long long) M, scale, e, passed ? "ok" : "FAIL");
    ggml_backend_buffer_free(want_g.buf);
    ggml_backend_buffer_free(got_g.buf);
    ggml_free(want_g.ctx);
    ggml_free(got_g.ctx);
    return passed;
}

// x (K x M) through Q8_0 projections of `ns[i]` rows each, then a tensor to guard
static norm_graph build_group(ggml_backend_t be, const std::vector<int64_t> & ns, int64_t M, ggml_tensor ** guard) {
    const int64_t K = 512;
    norm_graph    g;
    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    g.ctx               = ggml_init(ip);
    ggml_tensor * x     = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, K, M);
    g.inputs            = { x };
    std::vector<ggml_tensor *> mm;
    for (int64_t n : ns) {
        ggml_tensor * w = ggml_new_tensor_2d(g.ctx, GGML_TYPE_Q8_0, K, n);
        g.inputs.push_back(w);
        mm.push_back(ggml_mul_mat(g.ctx, w, x));
    }
    *guard = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, 2 * 1024);
    g.gf   = ggml_new_graph(g.ctx);
    for (ggml_tensor * t : mm) {
        ggml_build_forward_expand(g.gf, t);
    }
    g.checked = mm;
    g.buf     = ggml_backend_alloc_ctx_tensors(g.ctx, be);
    return g;
}

static bool test_group(ggml_backend_t npu, ggml_backend_t cpu, const std::vector<int64_t> & ns, int64_t M, double tol) {
    ggml_tensor *                   cpu_guard = nullptr;
    ggml_tensor *                   npu_guard = nullptr;
    norm_graph                      want_g    = build_group(cpu, ns, M, &cpu_guard);
    norm_graph                      got_g     = build_group(npu, ns, M, &npu_guard);
    std::vector<std::vector<float>> want, got;
    std::vector<float>              pattern(ggml_nelements(npu_guard), 3.0f);
    bool                            ok = run(want_g, cpu, want);
    ggml_backend_buffer_clear(got_g.buf, 0);
    set_inputs(got_g, 1234);
    ggml_backend_tensor_set(npu_guard, pattern.data(), 0, ggml_nbytes(npu_guard));
    ok = ok && ggml_backend_graph_compute(npu, got_g.gf) == GGML_STATUS_SUCCESS;
    std::vector<float> after(pattern.size());
    ggml_backend_tensor_get(npu_guard, after.data(), 0, ggml_nbytes(npu_guard));
    double worst = 0;
    for (size_t i = 0; ok && i < got_g.checked.size(); i++) {
        worst = std::max(worst, nmse(want[i], get_f32(got_g.checked[i])));
    }
    const bool intact = after == pattern;
    const bool passed = ok && intact && worst < tol;
    std::printf("MUL_MAT group of %zu, M %-4lld guard %s, worst nmse %.3g %s\n", ns.size(), (long long) M,
                intact ? "intact" : "OVERWRITTEN", ok ? worst : INFINITY, passed ? "ok" : "FAIL");
    ggml_backend_buffer_free(want_g.buf);
    ggml_backend_buffer_free(got_g.buf);
    ggml_free(want_g.ctx);
    ggml_free(got_g.ctx);
    return passed;
}

// x (K x M) through Q8_0 projections of `ns[i]` rows each, an RMS_NORM after every one and the
// results joined in that order, as a graph expanded from its output places them: projection,
// norm, next projection. Nodes are allocated in `ops_be`, or left to the scheduler when null.
enum class between { nothing, writer, weight_view };

struct ordered_graph {
    ggml_context *             leaf = nullptr;
    ggml_context *             ops  = nullptr;
    ggml_backend_buffer_t      leaf_buf = nullptr;
    ggml_backend_buffer_t      ops_buf  = nullptr;
    ggml_cgraph *              gf   = nullptr;
    std::vector<ggml_tensor *> mm;
    ggml_tensor *              out = nullptr;
    norm_graph                 inputs;
};

// the second projection's weight: a view of a larger tensor, a node of its own
static ggml_tensor * weight_view(ordered_graph & g, int64_t K, int64_t n) {
    ggml_tensor * full = ggml_new_tensor_2d(g.leaf, GGML_TYPE_Q8_0, K, n + 128);
    g.inputs.inputs.push_back(full);
    return ggml_view_2d(g.ops, full, K, n, full->nb[1], 0);
}

static ordered_graph build_ordered(ggml_backend_t leaf_be, ggml_backend_t ops_be, const std::vector<int64_t> & ns,
                                   int64_t M, between what) {
    const int64_t K = 512;
    ordered_graph g;
    ggml_init_params lp = { ggml_tensor_overhead() * 8, nullptr, true };
    ggml_init_params op = { ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true };
    g.leaf              = ggml_init(lp);
    g.ops               = ggml_init(op);
    ggml_tensor * x     = ggml_new_tensor_2d(g.leaf, GGML_TYPE_F32, K, M);
    g.inputs.inputs     = { x };
    g.gf                = ggml_new_graph(g.ops);
    for (size_t i = 0; i < ns.size(); i++) {
        ggml_tensor * w = nullptr;
        if (i == 1 && what == between::weight_view) {
            w = weight_view(g, K, ns[i]);
        } else {
            w = ggml_new_tensor_2d(g.leaf, GGML_TYPE_Q8_0, K, ns[i]);
            g.inputs.inputs.push_back(w);
        }
        ggml_tensor * mm = ggml_mul_mat(g.ops, w, x);
        g.mm.push_back(mm);
        ggml_tensor * norm = ggml_rms_norm(g.ops, mm, 1e-6f);
        g.out              = g.out ? ggml_concat(g.ops, g.out, norm, 0) : norm;
        if (i == 0) {
            ggml_build_forward_expand(g.gf, g.out);
            if (what == between::writer) {
                ggml_build_forward_expand(g.gf, ggml_scale_inplace(g.ops, x, 1.0f));
            }
        }
    }
    ggml_build_forward_expand(g.gf, g.out);
    g.leaf_buf = ggml_backend_alloc_ctx_tensors(g.leaf, leaf_be);
    if (ops_be) {
        g.ops_buf = ggml_backend_alloc_ctx_tensors(g.ops, ops_be);
    }
    return g;
}

static void free_ordered(ordered_graph & g) {
    ggml_backend_buffer_free(g.leaf_buf);
    ggml_backend_buffer_free(g.ops_buf);
    ggml_free(g.leaf);
    ggml_free(g.ops);
}

static int node_index(const ggml_cgraph * gf, const ggml_tensor * t) {
    for (int i = 0; i < ggml_graph_n_nodes(const_cast<ggml_cgraph *>(gf)); i++) {
        if (ggml_graph_node(const_cast<ggml_cgraph *>(gf), i) == t) {
            return i;
        }
    }
    return -1;
}

static bool next_to(const ordered_graph & g, size_t i) {
    return node_index(g.gf, g.mm[i]) == node_index(g.gf, g.mm[i - 1]) + 1;
}

// whether every projection follows the one before it in the scheduler's graph
static bool projections_adjacent(const ordered_graph & g) {
    for (size_t i = 1; i < g.mm.size(); i++) {
        if (!next_to(g, i)) {
            return false;
        }
    }
    return true;
}

// Q K V behind their norms through the scheduler against the CPU. `what` is a node between the
// first and second projection; `group_on` false (GGML_XDNA_PGEMM_GROUP=0) leaves the order as
// the graph was expanded.
static bool test_group_order(ggml_backend_t npu, ggml_backend_t cpu, between what, bool group_on, int64_t M,
                             double tol) {
    setenv("GGML_XDNA_PGEMM_GROUP", group_on ? "1" : "0", 1);
    const std::vector<int64_t> ns = { 1024, 512, 512 };
    ordered_graph              want_g = build_ordered(cpu, cpu, ns, M, what);
    ordered_graph              got_g  = build_ordered(npu, nullptr, ns, M, what);
    set_inputs(want_g.inputs, 1234);
    set_inputs(got_g.inputs, 1234);
    bool ok = ggml_backend_graph_compute(cpu, want_g.gf) == GGML_STATUS_SUCCESS;
    ggml_backend_t        backends[2] = { npu, cpu };
    ggml_backend_sched_t  sched       = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);
    ok = ok && ggml_backend_sched_alloc_graph(sched, got_g.gf);
    const bool moved = what == between::nothing ? projections_adjacent(got_g) : next_to(got_g, 1);
    ok = ok && ggml_backend_sched_graph_compute(sched, got_g.gf) == GGML_STATUS_SUCCESS;
    const double e      = ok ? nmse(get_f32(want_g.out), get_f32(got_g.out)) : INFINITY;
    const bool   passed = ok && moved == (group_on && what == between::nothing) && e < tol;
    const char * names[] = { "norms", "a writer of the activation", "a weight computed after Q" };
    std::printf("MUL_MAT Q K V behind %-27s M %-4lld GROUP=%d: projections %s, nmse %.3g %s\n",
                names[(int) what], (long long) M, group_on ? 1 : 0, moved ? "moved together" : "left apart", e,
                passed ? "ok" : "FAIL");
    ggml_backend_sched_free(sched);
    free_ordered(want_g);
    free_ordered(got_g);
    unsetenv("GGML_XDNA_PGEMM_GROUP");
    return passed;
}

int main(void) {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = xdna_device();
    if (!dev) {
        std::printf("test-xdna-graph: no XDNA device, skipping\n");
        return 0;
    }
    ggml_backend_t npu = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!npu || !cpu) {
        std::printf("test-xdna-graph: backend init failed\n");
        return 1;
    }

    // the projection's activations are rounded to bf16 on the NPU and to
    // Q8_K on the CPU, which leaves them 1-2e-4 apart
    const double tol    = 1e-3;
    bool         passed = true;
    for (bool side_reader : { false, true }) {
        norm_graph                      g = build(cpu, side_reader, 1024, 1024, 128);
        std::vector<std::vector<float>> want, got;
        if (!run(g, cpu, want) || !run(g, npu, got)) {
            std::printf("test-xdna-graph: compute failed\n");
            return 1;
        }
        for (size_t i = 0; i < g.checked.size(); i++) {
            const double e  = nmse(want[i], got[i]);
            const bool   ok = e < tol;
            passed          = passed && ok;
            std::printf("%-26s %-5s nmse %.3g %s\n", side_reader ? "ADD SQR RMS_NORM MUL_MAT" : "ADD RMS_NORM MUL_MAT",
                        ggml_get_name(g.checked[i]), e, ok ? "ok" : "FAIL");
        }
        ggml_backend_buffer_free(g.buf);
        ggml_free(g.ctx);
    }

    for (const char * route : { "fused", "per-op" }) {
        // the NPU computes the graph again with new activations; the CPU's
        // answer for each set is computed right before it
        setenv("GGML_XDNA_FUSED_LAYER", std::strcmp(route, "fused") == 0 ? "1" : "0", 1);
        norm_graph g = build_one_token(cpu, 1024, 1024);
        for (uint32_t seed : { 1u, 2u, 3u }) {
            std::vector<std::vector<float>> want, got;
            if (!run(g, cpu, want, seed) || !run(g, npu, got, seed)) {
                std::printf("test-xdna-graph: compute failed\n");
                return 1;
            }
            const double e  = nmse(want[0], got[0]);
            const bool   ok = e < tol;
            passed          = passed && ok;
            std::printf("MUL_MAT one token, %-6s run %u nmse %.3g %s\n", route, seed, e, ok ? "ok" : "FAIL");
        }
        ggml_backend_buffer_free(g.buf);
        ggml_free(g.ctx);
    }
    unsetenv("GGML_XDNA_FUSED_LAYER");

    // the cases that are not about the size rule force the array (or the host), whatever the size; the
    // by-size cases sit either side of its two bounds: n_tokens x n_kv x n_head, then the visible keys x n_head
    const fa_case fa_cases[] = {
        // name                                       route               heads tokens   ctx  pad seqs mask       window scale   cache  array
        { "causal 8 heads, cache view",               FA_ROUTE_ARRAY,        8, 1536,    0, 256,  1, FA_CAUSAL,   0, 0.0625f, true,  true },
        { "causal with 1100 keys of context",         FA_ROUTE_ARRAY,        8,  600, 1100, 256,  1, FA_CAUSAL,   0, 0.0625f, true,  true },
        { "bidirectional, 3 sequences",               FA_ROUTE_ARRAY,        4,  777,    0,   1,  3, FA_FULL,     0, 1.0f,    false, true },
        { "band 512, 2 sequences",                    FA_ROUTE_ARRAY,        4, 1300,    0,   1,  2, FA_BAND,   512, 1.0f,    false, true },
        { "band 40, one sequence",                    FA_ROUTE_ARRAY,        4,  900,    0,   1,  1, FA_BAND,    40, 1.0f,    false, true },
        { "40 sequences of 16 tokens",                FA_ROUTE_ARRAY,        4,  640,    0,   1, 40, FA_FULL,     0, 1.0f,    false, true },
        { "1 head per KV head group, causal",         FA_ROUTE_ARRAY,        2,  700,    0,   1,  1, FA_CAUSAL,   0, 0.0625f, false, true },
        { "below 64 tokens: the host",                FA_ROUTE_ARRAY,        4,   40,    0,   1,  1, FA_FULL,     0, 1.0f,    false, false },
        { "a mask with holes: the host",              FA_ROUTE_ARRAY,        4,  700,    0,   1,  1, FA_HOLES,    0, 1.0f,    false, false },
        { "FA_MM=0: the host at any size",            FA_ROUTE_HOST,         4, 1600,    0,   1,  1, FA_FULL,     0, 1.0f,    false, false },
        { "by size: under the shape bound",           FA_ROUTE_BY_SIZE,      4,  700,    0,   1,  1, FA_FULL,     0, 1.0f,    false, false },
        { "by size: over it, few keys visible",       FA_ROUTE_BY_SIZE,      4, 1600,    0,   1,  1, FA_BAND,    40, 1.0f,    false, false },
        { "by size: over both bounds",                FA_ROUTE_BY_SIZE,      4, 1600,    0,   1,  1, FA_FULL,     0, 1.0f,    false, true },
        { "by size: 8 heads, 1100 keys of context",   FA_ROUTE_BY_SIZE,      8,  600, 1100, 256,  1, FA_CAUSAL,   0, 0.0625f, true,  false },
        { "by size: 8 heads, 1500 keys of context",   FA_ROUTE_BY_SIZE,      8,  600, 1500, 256,  1, FA_CAUSAL,   0, 0.0625f, true,  true },
    };
    for (const fa_case & c : fa_cases) {
        set_fa_route(c.route);
        norm_graph               g = build_fa(cpu, c);
        std::vector<float>       want, got;
        if (!run_fa(g, c, cpu, want) || !run_fa(g, c, npu, got)) {
            std::printf("test-xdna-graph: compute failed\n");
            return 1;
        }
        const double e  = nmse(want, got);
        const bool   ok = c.on_array ? e > 0.0 && e < 1e-3 : e == 0.0;
        passed          = passed && ok;
        std::printf("FLASH_ATTN_EXT %-42s nmse %.3g %s\n", c.name, e, ok ? "ok" : "FAIL");
        ggml_backend_buffer_free(g.buf);
        ggml_free(g.ctx);
    }
    unsetenv("GGML_XDNA_FA_MM");

    const gemm_case gemm_cases[] = {
        { "decode block",     GGML_TYPE_BF16, 15,  { { 512, 1024 }, { 1024, 512 } } },
        { "two blocks",       GGML_TYPE_BF16, 300, { { 512, 1024 }, { 512, 512 } } },
        { "partial block",    GGML_TYPE_BF16, 912, { { 512, 2048 }, { 512, 512 } } },
        { "short tail",       GGML_TYPE_BF16, 1100, { { 512, 512 }, { 1024, 512 } } },
        { "mixed K and N",    GGML_TYPE_BF16, 912, { { 512, 1024 }, { 1024, 512 }, { 2048, 512 }, { 576, 1024 }, { 512, 512 } } },
        { "f16 weights",      GGML_TYPE_F16,  130, { { 1536, 512 }, { 512, 2048 } } },
    };
    for (const gemm_case & c : gemm_cases) {
        norm_graph                      g = build_gemms(cpu, c.wtype, c.M, c.shapes);
        std::vector<std::vector<float>> want, got;
        if (!run(g, cpu, want) || !run(g, npu, got)) {
            std::printf("test-xdna-graph: compute failed\n");
            return 1;
        }
        for (size_t i = 0; i < g.checked.size(); i++) {
            const double e  = nmse(want[i], got[i]);
            const bool   ok = e < tol;
            passed          = passed && ok;
            std::printf("MUL_MAT %-4s M%-4lld %-14s %-12s nmse %.3g %s\n", ggml_type_name(c.wtype), (long long) c.M, c.name,
                        ggml_get_name(g.checked[i]), e, ok ? "ok" : "FAIL");
        }
        ggml_backend_buffer_free(g.buf);
        ggml_free(g.ctx);
    }

    // a call runs the 128-row blocks its rows fill: one up to 128 rows, an odd count at 300 and 520, and 2100
    // rows are 17 blocks, a 16-block slab and a last one of one block
    for (int64_t M : { 15, 100, 128, 129, 300, 520, 2100 }) {
        passed = test_output_in_place(npu, cpu, M, tol) && passed;
    }
    for (int64_t M : { 15, 200 }) {
        passed = test_repeated_projection(npu, cpu, M, 3, tol) && passed;
    }

    const std::vector<std::vector<int64_t>> groups = { { 1024, 512, 512 }, { 2048, 512, 512 }, { 1024, 640 } };
    for (const std::vector<int64_t> & ns : groups) {
        for (int64_t M : { 15, 100, 128, 200, 300, 2100 }) {
            passed = test_group(npu, cpu, ns, M, tol) && passed;
        }
    }

    for (bool group_on : { true, false }) {
        passed = test_group_order(npu, cpu, between::nothing, group_on, 200, tol) && passed;
    }
    for (between what : { between::writer, between::weight_view }) {
        passed = test_group_order(npu, cpu, what, true, 200, tol) && passed;
    }

    for (ggml_glu_op op : { GGML_GLU_OP_SWIGLU, GGML_GLU_OP_GEGLU }) {
        for (bool glu_is_output : { false, true }) {
            for (float scale : { 0.05f, 0.3f }) {
                for (int64_t M : { 15, 128, 200, 2100 }) {
                    passed = test_ffn(npu, cpu, op, glu_is_output, 512, 2048, M, scale, 3e-3) && passed;
                }
                passed = test_ffn(npu, cpu, op, glu_is_output, 2560, 9216, 300, scale, 3e-3) && passed;
                passed = test_ffn(npu, cpu, op, glu_is_output, 2560, 9216, 100, scale, 3e-3) && passed;
            }
        }
    }

    ggml_backend_free(npu);
    ggml_backend_free(cpu);
    std::printf("test-xdna-graph: %s\n", passed ? "passed" : "FAILED");
    return passed ? 0 : 1;
}
