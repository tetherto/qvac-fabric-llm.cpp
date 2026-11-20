#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

static std::vector<float> make_angles() {
    std::vector<float> angles = {0.0f, -0.0f, std::numeric_limits<float>::min(),
                               std::numeric_limits<float>::max(), -std::numeric_limits<float>::max()};
    constexpr int samples = 257;
    for (float range : {1.0f, 1e3f, 1e5f, 1e8f, 1e20f, 1e30f}) {
        for (int i = 0; i < samples; ++i) {
            angles.push_back(float(double(range) * (2.0 * i / (samples - 1) - 1.0)));
        }
    }
    return angles;
}

static bool compare(const std::vector<float> & angles, const std::vector<float> & values, bool cosine) {
    constexpr double tolerance = 1e-6;
    double worst = 0;
    for (size_t i = 0; i < angles.size(); ++i) {
        const double expected = cosine ? std::cos(double(angles[i])) : std::sin(double(angles[i]));
        const double error = std::abs(double(values[i]) - expected);
        if (!std::isfinite(values[i]) || error > tolerance) {
            std::fprintf(stderr, "%s(%g): got %g, expected %.9g\n", cosine ? "cos" : "sin", angles[i], values[i], expected);
            return false;
        }
        worst = std::max(worst, error);
    }
    std::printf("%s: %zu angles, max absolute error %.3e\n", cosine ? "cos" : "sin", angles.size(), worst);
    return true;
}

static bool check(ggml_backend_t backend, bool cosine) {
    const std::vector<float> angles = make_angles();
    const ggml_init_params params = {ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true};
    ggml_context * ctx = ggml_init(params);
    if (!ctx) { return false; }
    ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, angles.size());
    ggml_tensor * output = cosine ? ggml_cos(ctx, input) : ggml_sin(ctx, input);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { ggml_free(ctx); return false; }
    ggml_backend_tensor_set(input, angles.data(), 0, angles.size()*sizeof(float));
    bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    if (ok) {
        std::vector<float> values(angles.size());
        ggml_backend_tensor_get(output, values.data(), 0, values.size()*sizeof(float));
        ok = compare(angles, values, cosine);
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_reg_t reg = ggml_backend_reg_by_name("WebGPU");
    if (!reg || ggml_backend_reg_dev_count(reg) == 0) { return 77; }
    ggml_backend_t backend = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
    if (!backend) { return 1; }
    const bool sin_ok = check(backend, false);
    const bool cos_ok = check(backend, true);
    ggml_backend_free(backend);
    return sin_ok && cos_ok ? 0 : 1;
}
