#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>

static bool compute(ggml_backend_t backend) {
    constexpr size_t context_bytes = 1024 * 1024;
    const ggml_init_params params = {context_bytes, nullptr, true};
    auto * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }
    auto * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    auto * output = ggml_scale(ctx, input, 2.0f);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        ggml_free(ctx);
        return false;
    }
    ggml_set_f32(input, 3.0f);
    const bool computed = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    float values[4] = {};
    ggml_backend_tensor_get(output, values, 0, sizeof(values));
    const bool ok = computed && values[0] == 6.0f && values[1] == 6.0f && values[2] == 6.0f && values[3] == 6.0f;
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    auto * cpu = ggml_backend_reg_by_name("CPU");
    if (!cpu || ggml_backend_dev_count() != 1) {
        std::fputs("CPU must be registered before loading any module\n", stderr);
        return 1;
    }
    auto * direct = ggml_backend_cpu_init();
    if (!direct) {
        return 1;
    }
    ggml_backend_free(direct);
    if (argc > 1 && !ggml_backend_load(argv[1])) {
        return 1;
    }
    if (ggml_backend_load("missing-hybrid-backend") != nullptr || ggml_backend_reg_by_name("CPU") != cpu) {
        return 1;
    }
    auto * backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend) {
        return 1;
    }
    const bool ok = compute(backend);
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
