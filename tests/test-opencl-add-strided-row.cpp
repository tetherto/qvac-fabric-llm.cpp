#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>

static bool run(ggml_backend_t backend, std::array<float, 32> & result) {
    ggml_context * ctx = ggml_init({ ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(8, false), nullptr, true });
    if (!ctx) return false;

    ggml_tensor * base = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 4);
    ggml_tensor * row  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_tensor * view = ggml_transpose(ctx, base);
    ggml_tensor * sum  = ggml_add(ctx, view, row);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(graph, sum);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        return false;
    }

    std::array<float, 32> values{};
    const std::array<float, 4> bias = { 100.0f, 101.0f, 102.0f, 103.0f };
    for (size_t i = 0; i < values.size(); ++i) values[i] = float(i);
    ggml_backend_tensor_set(base, values.data(), 0, sizeof(values));
    ggml_backend_tensor_set(row, bias.data(), 0, sizeof(bias));
    const bool passed = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    if (passed) ggml_backend_tensor_get(sum, result.data(), 0, sizeof(result));

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return passed;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t opencl_dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "OpenCL") == 0) {
            opencl_dev = dev;
            break;
        }
    }
    if (!opencl_dev) {
        std::fprintf(stderr, "No OpenCL device available\n");
        return 77;
    }

    ggml_backend_t opencl = ggml_backend_dev_init(opencl_dev, nullptr);
    if (!opencl) {
        return 1;
    }

    std::array<float, 32> opencl_result{};
    bool passed = run(opencl, opencl_result);
    for (size_t j = 0; passed && j < 8; ++j) {
        for (size_t i = 0; i < 4; ++i) {
            const float expected = float(j + 8*i) + float(100 + i);
            passed = std::fabs(opencl_result[j*4 + i] - expected) <= 1e-6f;
        }
    }
    ggml_backend_free(opencl);
    if (!passed) {
        std::fprintf(stderr, "Strided row-broadcast ADD differs from the reference values\n");
        return 1;
    }
    return 0;
}
