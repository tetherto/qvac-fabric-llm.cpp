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
// A prompt's last layer cropped to its output row, as llama builds it when
// only the last token's logits are asked for: the FFN of that one row in a
// prefill graph, on both routes, with the SwiGLU folded into the GEMM and not.
// The fused route runs one-row projections off the prefill GEMM, so no fusion
// may leave the norm or the SwiGLU only in such a projection's A.
// Qwen3.5's attention at one and two head groups (eight query heads over two
// kv heads a group: the 0.8B/2B, the 4B/9B): prefill on the NPU's mmul, the
// decode on its pool, reading the cache where llama keeps it (the XDNA
// device's host buffers). And prefill attention over a K/V row list
// (ggml_flash_attn_ext_rows): the mmul reads K/V in mask order, so the backend
// has to leave it to the host, which follows the list.
//
// A prompt's gated delta net at the 4B/9B's width (32 value heads over 16 q/k
// groups, two runs of the array), twice: with one head decaying ~1e-10 a token,
// as the 4B's layer 8 head 15 does (its chunk factors must stay in range,
// xdna-gdn-mm.cpp G_MIN), and with one that turns that fast after a chunk that
// leaves it just above the renormalization point (its shift, SHIFT_MAX).
//
// Every intermediate is cleared before each backend runs, so a node read
// before it was written shows as zeros instead of as the other backend's
// result. Skips (exit 0) when no XDNA device is registered.

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

// Qwen3.5's full-attention layers (xdna-attn-mm.cpp, xdna-att.h): prefill
// past the pair count up to which the host runs it, decode over a part of the
// cache that ends inside a chunk of the pool's
static constexpr int64_t FA_HEAD_DIM  = 256;
static constexpr int64_t FA_GROUP     = 8;  // query heads a head group
static constexpr int64_t FA_GQA       = 4;  // query heads a kv head serves
static constexpr int64_t FA_TOKENS    = 512;
static constexpr int64_t FA_KV        = 1024;
static constexpr int64_t FA_DEC_CTX   = 512;
static constexpr int64_t FA_DEC_VALID = 300;
static constexpr float   FA_SCALE     = 0.0625f;

// The gated delta net case: a prompt over the array's minimum, the 4B's heads,
// one head's gates (natural log) far under the floor and the others' range
static constexpr int64_t GDN_S         = 128;
static constexpr int64_t GDN_GROUPS    = 16;
static constexpr int64_t GDN_V_HEADS   = 32;
static constexpr int64_t GDN_TOKENS    = 163;
static constexpr int64_t GDN_FAST_HEAD = 15;
static constexpr float   GDN_FAST_GATE = -23.0f;
static constexpr float   GDN_GATE_LO   = -2.0f;
static constexpr float   GDN_GATE_HI   = -0.05f;
static constexpr float   GDN_BETA_LO   = 0.1f;
static constexpr float   GDN_BETA_HI   = 0.9f;
static constexpr int64_t GDN_CHUNK     = 16;      // tokens a chunk on the array
static constexpr int64_t GDN_EDGE_HEAD = 23;
static constexpr float   GDN_EDGE_GATE = -1.36f;  // a chunk of it sums to -31.4 log2, just above LS_MIN

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
    int32_t                    out_row = 0;  // what an I32 input (the output-row crop) selects
    std::vector<std::pair<ggml_tensor *, std::vector<uint8_t>>> fixed;  // set as given, after the inputs
};

template <typename T> static std::vector<uint8_t> to_bytes(const std::vector<T> & v) {
    const uint8_t * b = (const uint8_t *) v.data();
    return std::vector<uint8_t>(b, b + v.size() * sizeof(T));
}

// causal over n_kv keys: every key before the ubatch, then each token's own prefix
static std::vector<uint8_t> causal_mask(int64_t n_kv, int64_t n_tokens) {
    const int64_t            n_past = n_kv - n_tokens;
    std::vector<ggml_fp16_t> m((size_t) (n_kv * n_tokens));
    for (int64_t t = 0; t < n_tokens; t++) {
        for (int64_t j = 0; j < n_kv; j++) {
            m[(size_t) (t * n_kv + j)] = ggml_fp32_to_fp16(j <= n_past + t ? 0.0f : -INFINITY);
        }
    }
    return to_bytes(m);
}

// the first n_valid of n_kv keys, for one query
static std::vector<uint8_t> prefix_mask(int64_t n_kv, int64_t n_valid) {
    std::vector<ggml_fp16_t> m((size_t) n_kv);
    for (int64_t j = 0; j < n_kv; j++) {
        m[(size_t) j] = ggml_fp32_to_fp16(j < n_valid ? 0.0f : -INFINITY);
    }
    return to_bytes(m);
}

static std::vector<uint8_t> reversed_rows(int64_t n) {
    std::vector<int32_t> r((size_t) n);
    for (int64_t i = 0; i < n; i++) {
        r[(size_t) i] = (int32_t) (n - 1 - i);
    }
    return to_bytes(r);
}

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
        if (t->type == GGML_TYPE_I32) {
            std::vector<int32_t> rows(ggml_nelements(t), g.out_row);
            ggml_backend_tensor_set(t, rows.data(), 0, ggml_nbytes(t));
            continue;
        }
        std::mt19937       rng((t->type == GGML_TYPE_F32 ? seed : 1234) * 16 + (uint32_t) i);
        std::vector<float> f(ggml_nelements(t));
        for (float & x : f) {
            x = u(rng);
        }
        if (t->type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(t, f.data(), 0, ggml_nbytes(t));
        } else {
            std::vector<uint8_t> q(ggml_nbytes(t));
            ggml_quantize_chunk(t->type, f.data(), q.data(), 0, t->ne[1], t->ne[0], nullptr);
            ggml_backend_tensor_set(t, q.data(), 0, q.size());
        }
    }
    for (const auto & f : g.fixed) {
        ggml_backend_tensor_set(f.first, f.second.data(), 0, f.second.size());
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

// Qwen3.5-0.8B's residual and FFN widths, and a short prompt's rows
static constexpr int64_t CROP_EMBD   = 1024;
static constexpr int64_t CROP_FFN    = 3584;
static constexpr int64_t CROP_PROMPT = 16;

// An M-row Q4_K projection, the GET_ROWS of its last row, then RMS_NORM, MUL
// and a SwiGLU FFN of that one row, in the device's buffers as an offloaded
// layer's are (the norm is laid out as an A only there)
static norm_graph build_cropped_ffn(ggml_backend_buffer_type_t buft, int64_t K, int64_t F, int64_t M) {
    norm_graph g;
    ggml_init_params ip = { ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true };
    g.ctx               = ggml_init(ip);
    ggml_tensor * x     = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * p     = ggml_new_tensor_2d(g.ctx, GGML_TYPE_Q4_K, K, K);
    ggml_tensor * rows  = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, 1);
    ggml_tensor * w     = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, K);
    ggml_tensor * wg    = ggml_new_tensor_2d(g.ctx, GGML_TYPE_Q4_K, K, F);
    ggml_tensor * wu    = ggml_new_tensor_2d(g.ctx, GGML_TYPE_Q4_K, K, F);
    ggml_tensor * wd    = ggml_new_tensor_2d(g.ctx, GGML_TYPE_Q4_K, F, K);
    g.inputs            = { x, p, rows, w, wg, wu, wd };
    g.out_row           = (int32_t) M - 1;
    ggml_tensor * last  = ggml_get_rows(g.ctx, ggml_mul_mat(g.ctx, p, x), rows);
    ggml_tensor * norm  = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, last, 1e-6f), w);
    ggml_tensor * glu   = ggml_swiglu_split(g.ctx, ggml_mul_mat(g.ctx, wg, norm), ggml_mul_mat(g.ctx, wu, norm));
    ggml_tensor * out   = ggml_mul_mat(g.ctx, wd, glu);
    ggml_set_name(out, "ffn_out");
    g.gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, out);
    g.checked = { out };
    g.buf     = ggml_backend_alloc_ctx_tensors_from_buft(g.ctx, buft);
    return g;
}

// the cropped FFN on both routes, the SwiGLU folded into the GEMM and not
static bool check_cropped_ffn(ggml_backend_t cpu, ggml_backend_t npu, ggml_backend_buffer_type_t buft, double tol) {
    bool passed = true;
    for (const char * route : { "fused", "per-op" }) {
        setenv("GGML_XDNA_FUSED_LAYER", std::strcmp(route, "fused") == 0 ? "1" : "0", 1);
        for (const char * fold : { "1", "0" }) {
            setenv("GGML_XDNA_PGEMM_GLU", fold, 1);
            norm_graph                      g = build_cropped_ffn(buft, CROP_EMBD, CROP_FFN, CROP_PROMPT);
            std::vector<std::vector<float>> want, got;
            if (!run(g, cpu, want) || !run(g, npu, got)) {
                std::printf("test-xdna-graph: compute failed\n");
                return false;
            }
            const double e  = nmse(want[0], got[0]);
            const bool   ok = e < tol;
            passed          = passed && ok;
            std::printf("FFN of a cropped row, %-6s GLU fold %s nmse %.3g %s\n", route, fold, e, ok ? "ok" : "FAIL");
            ggml_backend_buffer_free(g.buf);
            ggml_free(g.ctx);
        }
    }
    unsetenv("GGML_XDNA_PGEMM_GLU");
    unsetenv("GGML_XDNA_FUSED_LAYER");
    return passed;
}

// Prefill attention of `heads` query heads; with `rows`, over the K/V rows in
// reverse, so attention that ignores the list sees other keys
static norm_graph build_attention(ggml_backend_t be, int64_t heads, int64_t n_tokens, int64_t n_kv, bool rows) {
    norm_graph       g;
    ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
    g.ctx               = ggml_init(ip);
    ggml_tensor * q     = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, FA_HEAD_DIM, n_tokens, heads, 1);
    ggml_tensor * k     = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F16, FA_HEAD_DIM, n_kv, heads / FA_GQA, 1);
    ggml_tensor * v     = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F16, FA_HEAD_DIM, n_kv, heads / FA_GQA, 1);
    ggml_tensor * mask  = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F16, n_kv, n_tokens, 1, 1);
    ggml_tensor * list  = ggml_new_tensor_2d(g.ctx, GGML_TYPE_I32, n_kv, 1);
    g.inputs            = { q, k, v };
    ggml_tensor * out   = rows ? ggml_flash_attn_ext_rows(g.ctx, q, k, v, mask, list, FA_SCALE, 0.0f, 0.0f) :
                                 ggml_flash_attn_ext(g.ctx, q, k, v, mask, FA_SCALE, 0.0f, 0.0f);
    ggml_set_name(out, "attn");
    g.gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, out);
    g.checked = { out };
    g.fixed   = {
        { mask, causal_mask(n_kv, n_tokens) }
    };
    if (rows) {
        g.fixed.push_back({ list, reversed_rows(n_kv) });
    }
    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, be);
    return g;
}

// Decode attention of `heads` query heads on llama's cache layout: a position
// holds every kv head's row, the view steps over them
static norm_graph build_decode_attention(ggml_backend_t be, int64_t heads, int64_t n_ctx, int64_t n_valid) {
    norm_graph       g;
    const int64_t    kvh = heads / FA_GQA;
    ggml_init_params ip  = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
    g.ctx                = ggml_init(ip);
    ggml_tensor * q      = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, FA_HEAD_DIM, 1, heads, 1);
    ggml_tensor * kc     = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F16, FA_HEAD_DIM * kvh, n_ctx);
    ggml_tensor * vc     = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F16, FA_HEAD_DIM * kvh, n_ctx);
    const size_t  hb     = FA_HEAD_DIM * sizeof(ggml_fp16_t);
    ggml_tensor * k      = ggml_view_3d(g.ctx, kc, FA_HEAD_DIM, n_ctx, kvh, kc->nb[1], hb, 0);
    ggml_tensor * v      = ggml_view_3d(g.ctx, vc, FA_HEAD_DIM, n_ctx, kvh, vc->nb[1], hb, 0);
    ggml_tensor * mask   = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F16, n_ctx, 1, 1, 1);
    g.inputs             = { q, kc, vc };
    ggml_tensor * out    = ggml_flash_attn_ext(g.ctx, q, k, v, mask, FA_SCALE, 0.0f, 0.0f);
    ggml_set_name(out, "attn");
    g.gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, out);
    g.checked = { out };
    g.fixed   = {
        { mask, prefix_mask(n_ctx, n_valid) }
    };
    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, be);
    return g;
}

// n rows of GDN_S values, uniform in [-1, 1), each scaled to unit length (the
// model's L2 norm of q and k) when `unit`
static std::vector<float> gdn_rows(std::mt19937 & rng, int64_t n, bool unit) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float>                    v((size_t) (n * GDN_S));
    for (float & x : v) {
        x = u(rng);
    }
    for (int64_t r = 0; unit && r < n; r++) {
        double ss = 0;
        for (int64_t i = 0; i < GDN_S; i++) {
            ss += (double) v[(size_t) (r * GDN_S + i)] * v[(size_t) (r * GDN_S + i)];
        }
        for (int64_t i = 0; i < GDN_S; i++) {
            v[(size_t) (r * GDN_S + i)] /= (float) std::sqrt(ss);
        }
    }
    return v;
}

// a value a token and value head, uniform in [lo, hi)
static std::vector<float> gdn_uniform(std::mt19937 & rng, float lo, float hi) {
    std::uniform_real_distribution<float> u(lo, hi);
    std::vector<float>                    v((size_t) (GDN_TOKENS * GDN_V_HEADS));
    for (float & x : v) {
        x = u(rng);
    }
    return v;
}

// the gates (natural log) in [GDN_GATE_LO, GDN_GATE_HI), but `head`'s: `first`
// over its first `n_first` tokens, GDN_FAST_GATE after
static std::vector<float> gdn_gates(std::mt19937 & rng, int64_t head, int64_t n_first, float first) {
    std::vector<float> v = gdn_uniform(rng, GDN_GATE_LO, GDN_GATE_HI);
    for (int64_t t = 0; t < GDN_TOKENS; t++) {
        v[(size_t) (t * GDN_V_HEADS + head)] = t < n_first ? first : GDN_FAST_GATE;
    }
    return v;
}

// One prefill GATED_DELTA_NET from a zero state, its gates from gdn_gates(head,
// n_first, first); its inputs fixed, so both backends see the same gates
static norm_graph build_gdn(ggml_backend_t be, int64_t head, int64_t n_first, float first) {
    norm_graph       g;
    ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
    g.ctx               = ggml_init(ip);
    ggml_tensor * q     = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, GDN_S, GDN_GROUPS, GDN_TOKENS, 1);
    ggml_tensor * k     = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, GDN_S, GDN_GROUPS, GDN_TOKENS, 1);
    ggml_tensor * v     = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, GDN_S, GDN_V_HEADS, GDN_TOKENS, 1);
    ggml_tensor * gate  = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, 1, GDN_V_HEADS, GDN_TOKENS, 1);
    ggml_tensor * beta  = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, 1, GDN_V_HEADS, GDN_TOKENS, 1);
    ggml_tensor * s     = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, GDN_S, GDN_S, GDN_V_HEADS, 1);
    ggml_tensor * out   = ggml_gated_delta_net(g.ctx, q, k, v, gate, beta, s, 1);
    ggml_set_name(out, "gdn");
    g.gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, out);
    g.checked = { out };
    std::mt19937 rng(1234);
    g.fixed = {
        { q, to_bytes(gdn_rows(rng, GDN_GROUPS * GDN_TOKENS, true)) },
        { k, to_bytes(gdn_rows(rng, GDN_GROUPS * GDN_TOKENS, true)) },
        { v, to_bytes(gdn_rows(rng, GDN_V_HEADS * GDN_TOKENS, false)) },
        { gate, to_bytes(gdn_gates(rng, head, n_first, first)) },
        { beta, to_bytes(gdn_uniform(rng, GDN_BETA_LO, GDN_BETA_HI)) },
        { s, to_bytes(std::vector<float>((size_t) (GDN_S * GDN_S * GDN_V_HEADS), 0.0f)) },
    };
    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, be);
    return g;
}

static void free_graph(norm_graph & g) {
    ggml_backend_buffer_free(g.buf);
    ggml_free(g.ctx);
}

// The graph on the CPU and on the NPU, its one checked node compared; false
// when either compute fails.
static bool check_one(norm_graph &   g,
                      ggml_backend_t cpu,
                      ggml_backend_t npu,
                      const char *   what,
                      double         tol,
                      bool *         passed) {
    std::vector<std::vector<float>> want, got;
    if (!run(g, cpu, want) || !run(g, npu, got)) {
        return false;
    }
    const double e  = nmse(want[0], got[0]);
    const bool   ok = e < tol;
    *passed         = *passed && ok;
    std::printf("%-26s %-5s nmse %.3g %s\n", what, ggml_get_name(g.checked[0]), e, ok ? "ok" : "FAIL");
    return true;
}

// value head h's nmse over its columns of every token's output and its state
static double gdn_head_nmse(const std::vector<float> & ref, const std::vector<float> & got, int64_t h) {
    std::vector<float> r, o;
    for (int64_t t = 0; t < GDN_TOKENS; t++) {
        const size_t at = (size_t) ((t * GDN_V_HEADS + h) * GDN_S);
        r.insert(r.end(), ref.begin() + at, ref.begin() + at + GDN_S);
        o.insert(o.end(), got.begin() + at, got.begin() + at + GDN_S);
    }
    const size_t st = (size_t) (GDN_TOKENS * GDN_V_HEADS * GDN_S + h * GDN_S * GDN_S);
    r.insert(r.end(), ref.begin() + st, ref.begin() + st + GDN_S * GDN_S);
    o.insert(o.end(), got.begin() + st, got.begin() + st + GDN_S * GDN_S);
    return nmse(r, o);
}

// The GDN graph on the CPU and on the NPU, head by head: a head decaying fast
// carries a small share of the whole output, which one nmse would not see
static bool check_gdn(norm_graph &   g,
                      ggml_backend_t cpu,
                      ggml_backend_t npu,
                      const char *   what,
                      double         tol,
                      bool *         passed) {
    std::vector<std::vector<float>> want, got;
    if (!run(g, cpu, want) || !run(g, npu, got)) {
        return false;
    }
    bool    ok    = true;
    int64_t worst = 0;
    double  e_max = 0;
    for (int64_t h = 0; h < GDN_V_HEADS; h++) {
        const double e = gdn_head_nmse(want[0], got[0], h);
        ok             = ok && e < tol;
        worst          = e > e_max ? h : worst;
        e_max          = std::fmax(e_max, e);
    }
    *passed = *passed && ok;
    std::printf("%-26s %-5s worst head %lld nmse %.3g %s\n", what, ggml_get_name(g.checked[0]), (long long) worst,
                e_max, ok ? "ok" : "FAIL");
    return true;
}

// Every attention case; the decode graphs all stay allocated until each has
// run, so none reuses another's cache memory (the pool keeps which rows of a
// cache it has flushed, by address).
static bool check_attention(ggml_backend_t cpu, ggml_backend_t npu, double tol, bool * passed) {
    bool       ran = true;
    norm_graph rl  = build_attention(cpu, FA_GROUP, FA_TOKENS, FA_KV, true);
    ran            = ran && check_one(rl, cpu, npu, "FLASH_ATTN_EXT kv_rows", tol, passed);
    free_graph(rl);
    norm_graph p1 = build_attention(cpu, FA_GROUP, FA_TOKENS, FA_KV, false);
    ran           = ran && check_one(p1, cpu, npu, "FLASH_ATTN_EXT prefill 8/2", tol, passed);
    free_graph(p1);
    norm_graph p2 = build_attention(cpu, 2 * FA_GROUP, FA_TOKENS, FA_KV, false);
    ran           = ran && check_one(p2, cpu, npu, "FLASH_ATTN_EXT prefill 16/4", tol, passed);
    free_graph(p2);
    norm_graph d1 = build_decode_attention(npu, FA_GROUP, FA_DEC_CTX, FA_DEC_VALID);
    norm_graph d2 = build_decode_attention(npu, 2 * FA_GROUP, FA_DEC_CTX, FA_DEC_VALID);
    ran           = ran && check_one(d1, cpu, npu, "FLASH_ATTN_EXT decode 8/2", tol, passed);
    ran           = ran && check_one(d2, cpu, npu, "FLASH_ATTN_EXT decode 16/4", tol, passed);
    free_graph(d1);
    free_graph(d2);
    return ran;
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

    passed = check_cropped_ffn(cpu, npu, ggml_backend_dev_buffer_type(dev), tol) && passed;
    if (!check_attention(cpu, npu, tol, &passed)) {
        std::printf("test-xdna-graph: compute failed\n");
        return 1;
    }
    norm_graph gdn = build_gdn(cpu, GDN_FAST_HEAD, 0, GDN_FAST_GATE);
    if (!check_gdn(gdn, cpu, npu, "GATED_DELTA_NET fast head", tol, &passed)) {
        std::printf("test-xdna-graph: compute failed\n");
        return 1;
    }
    free_graph(gdn);
    gdn = build_gdn(cpu, GDN_EDGE_HEAD, GDN_CHUNK, GDN_EDGE_GATE);
    if (!check_gdn(gdn, cpu, npu, "GATED_DELTA_NET shift edge", tol, &passed)) {
        std::printf("test-xdna-graph: compute failed\n");
        return 1;
    }
    free_graph(gdn);

    ggml_backend_free(npu);
    ggml_backend_free(cpu);
    std::printf("test-xdna-graph: %s\n", passed ? "passed" : "FAILED");
    return passed ? 0 : 1;
}
