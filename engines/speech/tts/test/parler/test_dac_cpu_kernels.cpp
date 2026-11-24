// Model-free coverage for the CPU DAC kernels against naive double-precision
// references.  Every build: the DAC snake (fused ggml_snake over precomputed
// 1 / (alpha + eps), and the broadcasting chain for a scalar alpha), the
// ggml "same" conv, and the Hexagon transposed-conv lowering (kernel
// rearranged at load, one GEMM into columns, ggml_col2im_1d) at every DAC
// stride.  Accelerate builds additionally pin the cblas kernels --
// the conv1d (every DAC dilation, the k=1 residual form, row tiles, one- and
// two-frame inputs, a scalar bias routed back to ggml, one and several
// threads), the transposed conv (sgemm columns + ggml_col2im_1d, every DAC
// stride) and the vForce snake -- plus the switches that route the DAC onto
// them (CPU only, PARLER_DAC_NO_ACCEL, shape exactness).

#include "parler/internal.h"
#include "backend_selection.h"
#include "sched_dispatch.h"
#include "../test_env_portable.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>

using namespace tts_cpp::parler::detail;

static int g_failures = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            ++g_failures;                                                      \
            fprintf(stderr, "FAIL: %s  (%s:%d)\n", msg, __FILE__, __LINE__);   \
        }                                                                      \
    } while (0)

namespace {

constexpr int    GRAPH_NODES       = 64;
constexpr double REL_TOLERANCE     = 1e-5;
constexpr double F16_REL_TOLERANCE = 2e-3;

struct det_rng {
    uint32_t state = 0x2545F491u;
    float next() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return (float) (state & 0xFFFFFF) / (float) 0xFFFFFF - 0.5f;
    }
};

std::vector<float> random_values(size_t n, det_rng & rng, float scale, float bias) {
    std::vector<float> v(n);
    for (float & x : v) x = bias + scale * rng.next();
    return v;
}

struct graph_input {
    ggml_tensor *              tensor = nullptr;
    const std::vector<float> * values = nullptr;
};

// Inputs a graph does not consume (the snake epsilon on the fused route) get
// no buffer and are skipped.
void set_inputs(const std::vector<graph_input> & inputs) {
    for (const graph_input & in : inputs) {
        if (!in.tensor->buffer) continue;
        ggml_backend_tensor_set(in.tensor, in.values->data(), 0, in.values->size() * sizeof(float));
    }
}

// Builds a graph through `build` over the declared inputs, runs it on the CPU
// backend with n_threads, and returns the output tensor's values.
std::vector<float> run_graph(ggml_backend_t backend, int n_threads,
                             const std::function<ggml_tensor *(ggml_context *, std::vector<graph_input> &)> & build) {
    const size_t ctx_size = ggml_tensor_overhead() * GRAPH_NODES +
                            ggml_graph_overhead_custom(GRAPH_NODES, false);
    ggml_init_params ip = { ctx_size, nullptr, /*no_alloc=*/ true };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, GRAPH_NODES, false);
    std::vector<graph_input> inputs;
    ggml_tensor * out = build(ctx, inputs);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    std::vector<float> result;
    if (ggml_gallocr_alloc_graph(allocr, gf)) {
        set_inputs(inputs);
        if (tts_cpp::detail::direct_compute(backend, gf, n_threads) == GGML_STATUS_SUCCESS) {
            result.resize((size_t) ggml_nelements(out));
            ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
        }
    }
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    return result;
}

ggml_tensor * input_2d(ggml_context * ctx, std::vector<graph_input> & inputs,
                       const std::vector<float> & values, int64_t ne0, int64_t ne1) {
    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    inputs.push_back({ t, &values });
    return t;
}

ggml_tensor * input_3d(ggml_context * ctx, std::vector<graph_input> & inputs,
                       const std::vector<float> & values, int64_t ne0, int64_t ne1, int64_t ne2) {
    ggml_tensor * t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, ne2);
    inputs.push_back({ t, &values });
    return t;
}

double max_rel_error(const std::vector<float> & got, const std::vector<double> & ref) {
    double max_ref = 0.0, max_err = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        max_ref = std::max(max_ref, std::fabs(ref[i]));
        max_err = std::max(max_err, std::fabs((double) got[i] - ref[i]));
    }
    return max_err / std::max(max_ref, 1e-30);
}

bool matches(const char * what, const std::vector<float> & got, const std::vector<double> & ref,
             double tolerance = REL_TOLERANCE) {
    if (got.size() != ref.size()) {
        fprintf(stderr, "  %s: size %zu != reference %zu\n", what, got.size(), ref.size());
        return false;
    }
    const double rel = max_rel_error(got, ref);
    fprintf(stderr, "  %s: max relative error %.3g\n", what, rel);
    return rel <= tolerance;
}

// ---- references (x is [L, C], L fastest; ggml weight layouts) -------------

double snake_one(double x, double alpha) {
    const double s = std::sin(alpha * x);
    return x + s * s / (alpha + (double) PARLER_DAC_SNAKE_EPS);
}

// A one-element parameter broadcasts over every channel, as ggml_add does.
float channel_value(const std::vector<float> & v, size_t channel) {
    return v[v.size() == 1 ? 0 : channel];
}

std::vector<double> snake_ref(const std::vector<float> & x, const std::vector<float> & alpha, int64_t T) {
    std::vector<double> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) y[i] = snake_one(x[i], channel_value(alpha, i / T));
    return y;
}

struct conv_case {
    int64_t L, IC, OC, K;
    int     dilation;
    bool    residual;
    bool    scalar_bias;
};

double conv1d_channel_taps(const std::vector<float> & x, const std::vector<float> & w,
                           const conv_case & c, int64_t t, int64_t oc, int64_t ic) {
    const int64_t pad = (c.K - 1) / 2 * c.dilation;
    double acc = 0.0;
    for (int64_t k = 0; k < c.K; ++k) {
        const int64_t src = t + k * c.dilation - pad;
        if (src < 0 || src >= c.L) continue;
        acc += (double) x[ic * c.L + src] * w[k + c.K * ic + c.K * c.IC * oc];
    }
    return acc;
}

double conv1d_tap_sum(const std::vector<float> & x, const std::vector<float> & w,
                      const conv_case & c, int64_t t, int64_t oc) {
    double acc = 0.0;
    for (int64_t ic = 0; ic < c.IC; ++ic) acc += conv1d_channel_taps(x, w, c, t, oc, ic);
    return acc;
}

void conv1d_ref_channel(const std::vector<float> & x, const std::vector<float> & w,
                        const std::vector<float> & b, const std::vector<float> & res,
                        const conv_case & c, int64_t oc, std::vector<double> & y) {
    for (int64_t t = 0; t < c.L; ++t) {
        const double r = c.residual ? res[oc * c.L + t] : 0.0;
        y[oc * c.L + t] = conv1d_tap_sum(x, w, c, t, oc) + channel_value(b, oc) + r;
    }
}

std::vector<double> conv1d_ref(const std::vector<float> & x, const std::vector<float> & w,
                               const std::vector<float> & b, const std::vector<float> & res,
                               const conv_case & c) {
    std::vector<double> y((size_t) (c.L * c.OC));
    for (int64_t oc = 0; oc < c.OC; ++oc) conv1d_ref_channel(x, w, b, res, c, oc, y);
    return y;
}

struct convt_case {
    int64_t IL, IC, OC;
    int     stride;
};

double convt_tap(const std::vector<float> & x, const std::vector<float> & w,
                 const convt_case & c, int64_t i, int64_t oc, int64_t k) {
    const int64_t K = 2 * c.stride;
    double acc = 0.0;
    for (int64_t ic = 0; ic < c.IC; ++ic) acc += (double) x[ic * c.IL + i] * w[k + K * oc + K * c.OC * ic];
    return acc;
}

void convt_scatter_channel(const std::vector<float> & x, const std::vector<float> & w,
                           const convt_case & c, int64_t i, int64_t oc, std::vector<double> & y) {
    const int64_t L_out = c.IL * c.stride, trim = c.stride / 2;
    for (int64_t k = 0; k < 2 * c.stride; ++k) {
        const int64_t o = i * c.stride + k - trim;
        if (o >= 0 && o < L_out) y[oc * L_out + o] += convt_tap(x, w, c, i, oc, k);
    }
}

void convt_scatter(const std::vector<float> & x, const std::vector<float> & w,
                   const convt_case & c, int64_t i, std::vector<double> & y) {
    for (int64_t oc = 0; oc < c.OC; ++oc) convt_scatter_channel(x, w, c, i, oc, y);
}

std::vector<double> bias_rows(const std::vector<float> & b, int64_t len) {
    std::vector<double> y((size_t) (len * b.size()));
    for (size_t i = 0; i < y.size(); ++i) y[i] = b[i / len];
    return y;
}

std::vector<double> convt_ref(const std::vector<float> & x, const std::vector<float> & w,
                              const std::vector<float> & b, const convt_case & c) {
    std::vector<double> y = bias_rows(b, c.IL * c.stride);
    for (int64_t i = 0; i < c.IL; ++i) convt_scatter(x, w, c, i, y);
    return y;
}

// ---- cases -----------------------------------------------------------------

struct snake_data {
    int64_t T = 0, C = 0;
    std::vector<float> x, alpha, inv, eps;
};

snake_data make_snake_data(det_rng & rng, bool per_channel) {
    snake_data d;
    d.T = 1500;
    d.C = 6;
    d.x     = random_values((size_t) (d.T * d.C), rng, 6.0f, 0.0f);
    d.alpha = random_values(per_channel ? (size_t) d.C : 1, rng, 1.5f, 1.0f);
    d.inv.resize(d.alpha.size());
    std::transform(d.alpha.begin(), d.alpha.end(), d.inv.begin(),
                   [](float a) { return 1.0f / (a + PARLER_DAC_SNAKE_EPS); });
    d.eps = { PARLER_DAC_SNAKE_EPS };
    return d;
}

std::vector<float> run_snake(ggml_backend_t backend, const snake_data & d, bool accel) {
    const int64_t n_alpha = (int64_t) d.alpha.size();
    return run_graph(backend, 4, [&](ggml_context * ctx, std::vector<graph_input> & in) {
        ggml_tensor * x = input_2d(ctx, in, d.x, d.T, d.C);
        ggml_tensor * a = input_3d(ctx, in, d.alpha, 1, n_alpha, 1);
        ggml_tensor * v = input_3d(ctx, in, d.inv, 1, n_alpha, 1);
        ggml_tensor * e = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        in.push_back({ e, &d.eps });
        return parler_dac_snake(ctx, accel, x, a, v, e);
    });
}

void run_snake_case(ggml_backend_t backend, bool per_channel, bool accel) {
    det_rng rng;
    const snake_data d = make_snake_data(rng, per_channel);
    char what[96];
    snprintf(what, sizeof(what), "snake, %s alpha, %s", per_channel ? "per-channel" : "scalar",
             accel ? "Accelerate route" : "ggml route");
    CHECK(matches(what, run_snake(backend, d, accel), snake_ref(d.x, d.alpha, d.T)),
          "the DAC snake matches x + sin^2(a x) / (a + eps), broadcasting a scalar alpha");
}

void test_snake(ggml_backend_t backend) {
    const bool per_channel[] = { true, false };
    for (bool pc : per_channel) {
        run_snake_case(backend, pc, false);
        if (parler_dac_accel_compiled()) run_snake_case(backend, pc, true);
    }
}

void run_conv_case(ggml_backend_t backend, const conv_case & c, bool accel, int n_threads,
                   bool f16_gemm = false) {
    det_rng rng;
    const int64_t n_bias = c.scalar_bias ? 1 : c.OC;
    const std::vector<float> x   = random_values((size_t) (c.L * c.IC), rng, 1.0f, 0.0f);
    const std::vector<float> w   = random_values((size_t) (c.K * c.IC * c.OC), rng, 0.2f, 0.0f);
    const std::vector<float> b   = random_values((size_t) n_bias, rng, 0.5f, 0.0f);
    const std::vector<float> res = random_values((size_t) (c.L * c.OC), rng, 1.0f, 0.0f);
    const std::vector<float> got = run_graph(backend, n_threads,
        [&](ggml_context * ctx, std::vector<graph_input> & in) {
            ggml_tensor * xt = input_3d(ctx, in, x, c.L, c.IC, 1);
            ggml_tensor * wt = input_3d(ctx, in, w, c.K, c.IC, c.OC);
            ggml_tensor * bt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_bias);
            in.push_back({ bt, &b });
            ggml_tensor * rt = c.residual ? input_3d(ctx, in, res, c.L, c.OC, 1) : nullptr;
            return parler_dac_conv_same(ctx, accel, xt, wt, bt, c.dilation, rt, f16_gemm);
        });
    char what[176];
    snprintf(what, sizeof(what), "conv1d L=%lld IC=%lld OC=%lld K=%lld d=%d%s%s, %s, %d threads",
             (long long) c.L, (long long) c.IC, (long long) c.OC, (long long) c.K, c.dilation,
             c.residual ? " +residual" : "", c.scalar_bias ? " scalar-bias" : "",
             accel ? "Accelerate route" : f16_gemm ? "F16 GEMM route" : "ggml route", n_threads);
    CHECK(matches(what, got, conv1d_ref(x, w, b, res, c), f16_gemm ? F16_REL_TOLERANCE : REL_TOLERANCE),
          "the DAC conv matches the reference");
}

void test_conv_same_f16_gemm(ggml_backend_t backend) {
    const conv_case cases[] = {
        { 300, 200, 24, 7, 1, false, false },
        { 190,  32,  8, 7, 9, false, false },
        { 333,  96, 40, 1, 1, true,  false },
        {  50, 128,  1, 7, 1, false, false },
    };
    for (const conv_case & c : cases) run_conv_case(backend, c, false, 4, true);
}

void test_conv_same(ggml_backend_t backend, bool accel) {
    const conv_case cases[] = {
        { 300, 200, 24, 7, 1, false, false },   // several im2col row tiles
        { 257,  48, 16, 7, 3, false, false },
        { 190,  32,  8, 7, 9, false, false },   // padding wider than a tap stride
        { 333,  96, 40, 1, 1, true,  false },   // the k=1 residual form
        {  50, 128,  1, 7, 1, false, false },   // conv_out: a single output channel
        {   1,  32,  8, 7, 9, false, false },   // one frame: every tap but one is padding
        {   2,  32,  8, 7, 3, false, false },
        { 120,  64, 16, 7, 1, false, true  },   // a scalar bias broadcasts
        {  90,  48, 12, 1, 1, true,  true  },
    };
    const int threads[] = { 1, 4 };
    for (const conv_case & c : cases) {
        for (int n : threads) run_conv_case(backend, c, accel, n);
    }
}

void run_convt_case(ggml_backend_t backend, const convt_case & c) {
    det_rng rng;
    const int64_t K = 2 * c.stride;
    const std::vector<float> x = random_values((size_t) (c.IL * c.IC), rng, 1.0f, 0.0f);
    const std::vector<float> w = random_values((size_t) (K * c.OC * c.IC), rng, 0.2f, 0.0f);
    const std::vector<float> b = random_values((size_t) c.OC, rng, 0.5f, 0.0f);
    const std::vector<float> got = run_graph(backend, 4,
        [&](ggml_context * ctx, std::vector<graph_input> & in) {
            ggml_tensor * xt = input_3d(ctx, in, x, c.IL, c.IC, 1);
            ggml_tensor * wt = input_3d(ctx, in, w, K, c.OC, c.IC);
            ggml_tensor * bt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, c.OC);
            in.push_back({ bt, &b });
            ggml_tensor * cols = parler_dac_accel_convt_columns(ctx, xt, wt);
            return ggml_add(ctx, ggml_col2im_1d(ctx, cols, c.stride, (int) c.OC, c.stride / 2), bt);
        });
    char what[96];
    snprintf(what, sizeof(what), "conv_transpose IL=%lld IC=%lld OC=%lld s=%d",
             (long long) c.IL, (long long) c.IC, (long long) c.OC, c.stride);
    CHECK(matches(what, got, convt_ref(x, w, b, c)), "Accelerate transposed conv matches the reference");
}

std::vector<float> convt_cols_weight(const std::vector<float> & w, const convt_case & c) {
    const int64_t K = 2 * c.stride;
    std::vector<float> cols(w.size());
    parler_dac_convt_cols_from_kernel(w.data(), K, c.OC, c.IC, cols.data());
    return cols;
}

void run_convt_columns_case(ggml_backend_t backend, const convt_case & c) {
    det_rng rng;
    const int64_t K = 2 * c.stride;
    const std::vector<float> x    = random_values((size_t) (c.IL * c.IC), rng, 1.0f, 0.0f);
    const std::vector<float> w    = random_values((size_t) (K * c.OC * c.IC), rng, 0.2f, 0.0f);
    const std::vector<float> b    = random_values((size_t) c.OC, rng, 0.5f, 0.0f);
    const std::vector<float> cols = convt_cols_weight(w, c);
    const std::vector<float> got  = run_graph(backend, 4,
        [&](ggml_context * ctx, std::vector<graph_input> & in) {
            ggml_tensor * xt = input_3d(ctx, in, x, c.IL, c.IC, 1);
            ggml_tensor * wt = input_2d(ctx, in, cols, c.IC, K * c.OC);
            ggml_tensor * bt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, c.OC);
            in.push_back({ bt, &b });
            return ggml_add(ctx, parler_dac_convt_columns(ctx, xt, wt, c.stride, true), bt);
        });
    char what[112];
    snprintf(what, sizeof(what), "conv_transpose columns IL=%lld IC=%lld OC=%lld s=%d",
             (long long) c.IL, (long long) c.IC, (long long) c.OC, c.stride);
    CHECK(matches(what, got, convt_ref(x, w, b, c)),
          "the columns-GEMM transposed conv matches the reference");
}

void test_convt_columns(ggml_backend_t backend) {
    const convt_case cases[] = {
        { 141, 64, 24, 8 },
        {  70, 40, 16, 4 },
        {  33, 24,  8, 2 },
        {   1, 32,  8, 8 },
        {   2, 16,  4, 2 },
    };
    for (const convt_case & c : cases) run_convt_columns_case(backend, c);
}

void test_accel_convt(ggml_backend_t backend) {
    const convt_case cases[] = {
        { 141, 64, 24, 8 },   // spreads over several column spans
        {  70, 40, 16, 4 },
        {  33, 24,  8, 2 },
    };
    for (const convt_case & c : cases) run_convt_case(backend, c);
}

// Shapes of the mini/large DAC convs that bound the im2col tile: conv_in is
// the widest row (7 taps x 1024 latent channels), so it sets the bound at the
// 64-row floor; conv_out and a k=1 conv2 must not raise it.
void test_scratch_bound() {
    ggml_init_params ip = { 8 * ggml_tensor_overhead(), nullptr, /*no_alloc=*/ true };
    ggml_context * ctx = ggml_init(ip);
    parler_model model;
    model.dac_conv_in_w  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 7, 1024, 1536);
    model.dac_conv_out_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 7, 96, 1);
    model.dac_blocks.resize(1);
    for (parler_dac_residual & r : model.dac_blocks[0].res) {
        r.conv1_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 7, 768, 768);
        r.conv2_w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 768, 768);
    }
    const size_t conv_in_tile = (size_t) 64 * 7 * 1024 * sizeof(float);
    const size_t want = parler_dac_accel_compiled() ? 3 * conv_in_tile : 0;
    CHECK(parler_dac_accel_scratch_bytes(model, 3) == want,
          "the im2col scratch bound is the widest conv's tile per thread");
    model.on_gpu = true;
    CHECK(parler_dac_accel_scratch_bytes(model, 3) == 0, "GPU models price no im2col scratch");
    ggml_free(ctx);
}

void test_accel_switches() {
    parler_model gpu_model;
    gpu_model.on_gpu = true;
    CHECK(!parler_dac_accel_enabled(gpu_model), "GPU models never take the Accelerate DAC");
    CHECK(!parler_dac_uses_f16_gemm(gpu_model), "GPU models keep F32 DAC GEMMs");

    parler_model hexagon_model;
    hexagon_model.on_gpu     = true;
    hexagon_model.on_hexagon = true;
    CHECK(parler_dac_uses_f16_gemm(hexagon_model), "Hexagon models run the DAC GEMMs in F16 on HMX");
    CHECK(!parler_dac_accel_enabled(hexagon_model), "Hexagon models never take the Accelerate DAC");

    parler_model cpu_model;
    CHECK(parler_dac_accel_enabled(cpu_model) == parler_dac_accel_compiled(),
          "CPU models take the Accelerate DAC exactly when it is compiled in");
    CHECK(!parler_dac_uses_f16_gemm(cpu_model), "CPU models keep F32 DAC GEMMs");
    if (parler_dac_accel_compiled()) {
        CHECK(!parler_dac_cpu_is_shape_exact(), "the Accelerate DAC is not shape-exact");
    }
    setenv("PARLER_DAC_NO_ACCEL", "1", 1);
    CHECK(!parler_dac_accel_enabled(cpu_model), "PARLER_DAC_NO_ACCEL keeps the ggml kernels");
    unsetenv("PARLER_DAC_NO_ACCEL");
}

} // namespace

int main() {
    ::tts_cpp::detail::ensure_backends_loaded();
    ggml_backend_t backend = ::tts_cpp::detail::init_cpu_backend();
    if (!backend) {
        fprintf(stderr, "parler dac cpu kernels: no CPU backend\n");
        return 1;
    }
    test_snake(backend);
    test_conv_same(backend, false);
    test_conv_same_f16_gemm(backend);
    test_convt_columns(backend);
    test_accel_switches();
    test_scratch_bound();
    if (parler_dac_accel_compiled()) {
        test_conv_same(backend, true);
        test_accel_convt(backend);
    } else {
        fprintf(stderr, "parler dac cpu kernels: Accelerate kernels not compiled in; cblas cases skipped\n");
    }
    ggml_backend_free(backend);

    if (g_failures == 0) {
        fprintf(stderr, "parler dac cpu kernels: PASS\n");
        return 0;
    }
    fprintf(stderr, "parler dac cpu kernels: %d failure(s)\n", g_failures);
    return 1;
}
