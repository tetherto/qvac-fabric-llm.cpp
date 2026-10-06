// Unit test for ggml_snake (ACE-Step Oobleck VAE op).
//
// Snake activation: y = x + sin^2(a*x) * inv_b, with per-channel a and inv_b.
// Checked against a straightforward host reference over the same [T, C] layout.

#include <ggml.h>
#include <ggml-cpu.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>

#include <cmath>
#include <cstdio>
#include <vector>

// y[c*T + t] = x[c*T + t] + sin(a[c] * x[c*T + t])^2 * inv_b[c]
static std::vector<float> snake_reference(
        const std::vector<float> & x,
        const std::vector<float> & a,
        const std::vector<float> & inv_b,
        int64_t T, int64_t C) {

    std::vector<float> dst(T * C);
    for (int64_t c = 0; c < C; ++c) {
        for (int64_t t = 0; t < T; ++t) {
            const float xi = x[c * T + t];
            const float si = std::sin(a[c] * xi);
            dst[c * T + t] = xi + si * si * inv_b[c];
        }
    }
    return dst;
}

static bool check_equal(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        printf("  size mismatch: %d != %d\n", (int) a.size(), (int) b.size());
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > 1e-5f) {
            printf("  [%d] %f != %f\n", (int) i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

static bool test_snake(int64_t T, int64_t C) {
    ggml_init_params params {
        /*.mem_size   =*/ 16 * ggml_tensor_overhead() + ggml_graph_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr{ggml_init(params)};
    ggml_context * ctx = ctx_ptr.get();
    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * x     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, C);
    ggml_tensor * a     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
    ggml_tensor * inv_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
    ggml_tensor * res   = ggml_snake(ctx, x, a, inv_b);
    ggml_build_forward_expand(gf, res);

    ggml_backend_ptr backend_ptr{ggml_backend_cpu_init()};
    ggml_backend_t backend = backend_ptr.get();
    ggml_backend_cpu_set_n_threads(backend, 2);
    ggml_backend_buffer_ptr buffer{ggml_backend_alloc_ctx_tensors(ctx, backend)};

    std::vector<float> x_values(T * C);
    for (size_t i = 0; i < x_values.size(); ++i) {
        x_values[i] = -1.5f + 0.037f * (float) i; // spread across sign changes
    }
    std::vector<float> a_values(C), b_values(C);
    for (int64_t c = 0; c < C; ++c) {
        a_values[c] = 0.5f + 0.25f * (float) c;
        b_values[c] = 1.0f / (1.0f + (float) c); // inv_b
    }
    ggml_backend_tensor_set(x,     x_values.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(a,     a_values.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(inv_b, b_values.data(), 0, ggml_nbytes(inv_b));

    ggml_backend_graph_compute(backend, gf);

    std::vector<float> res_values(ggml_nelements(res));
    ggml_backend_tensor_get(res, res_values.data(), 0, ggml_nbytes(res));

    const std::vector<float> expected = snake_reference(x_values, a_values, b_values, T, C);

    const bool passed = check_equal(res_values, expected);
    printf("ggml_snake(T=%d, C=%d): %s\n", (int) T, (int) C,
        passed ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m");
    return passed;
}

int main() {
    bool ok = true;
    ok &= test_snake(32, 8);
    ok &= test_snake(127, 3);
    ok &= test_snake(1, 16); // single timestep
    ok &= test_snake(64, 1); // mono channel
    return ok ? 0 : 1;
}
