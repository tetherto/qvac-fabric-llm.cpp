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
//
// Every intermediate is cleared before each backend runs, so a node read
// before it was written shows as zeros instead of as the other backend's
// result. Skips (exit 0) when no XDNA device is registered.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
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
    int32_t                    out_row = 0;  // what an I32 input (the output-row crop) selects
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

    ggml_backend_free(npu);
    ggml_backend_free(cpu);
    std::printf("test-xdna-graph: %s\n", passed ? "passed" : "FAILED");
    return passed ? 0 : 1;
}
