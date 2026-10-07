#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr int64_t KERNEL_SIZE       = 16;
constexpr int64_t STRIDE            = 4;
constexpr int     PADDING           = 0;
constexpr int     DILATION          = 1;
constexpr int64_t CHANNELS_IN       = 18;
constexpr int64_t CHANNELS_OUT      = 1;
constexpr int64_t SATURATING_INPUT  = 16384;
constexpr int64_t LENGTH_RATIO      = 16;
constexpr int64_t SCALED_INPUT      = SATURATING_INPUT * LENGTH_RATIO;
constexpr double  MAX_COST_GROWTH   = 4.0;
constexpr int     TIMED_RUNS        = 5;
constexpr size_t  CONTEXT_BYTES     = size_t(32) * 1024 * 1024;
constexpr float   KERNEL_FILL       = 0.01f;
constexpr float   INPUT_FILL        = 0.02f;
constexpr int     CTEST_SKIP_STATUS = 77;
constexpr const char * GGML_CUDA_REGISTRIES[] = { "CUDA", "ROCm", "MUSA" };

enum class outcome { passed, failed, skipped };

struct timing {
    bool   ok;
    double nanoseconds;
};

struct measurement {
    bool    ok;
    double  nanoseconds_per_output;
    int64_t output_elements;
};

ggml_tensor * build_conv_transpose_1d(ggml_context * ctx, int64_t input_length) {
    ggml_tensor * kernel = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, KERNEL_SIZE, CHANNELS_OUT, CHANNELS_IN);
    ggml_set_name(kernel, "kernel");

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_length, CHANNELS_IN);
    ggml_set_name(input, "input");

    return ggml_conv_transpose_1d(ctx, kernel, input, STRIDE, PADDING, DILATION);
}

bool backend_supports_conv_transpose_1d(ggml_backend_t backend) {
    ggml_init_params params = { CONTEXT_BYTES, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(params);

    const bool supported = ggml_backend_supports_op(backend, build_conv_transpose_1d(ctx, SATURATING_INPUT));

    ggml_free(ctx);
    return supported;
}

timing time_one_run(ggml_backend_t backend, ggml_cgraph * graph) {
    const auto        start  = std::chrono::steady_clock::now();
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);
    const auto        end    = std::chrono::steady_clock::now();

    return { status == GGML_STATUS_SUCCESS,
             std::chrono::duration<double, std::nano>(end - start).count() };
}

timing fastest_of_timed_runs(ggml_backend_t backend, ggml_cgraph * graph) {
    timing fastest = time_one_run(backend, graph);
    if (!fastest.ok) {
        return fastest;
    }
    for (int run = 1; run < TIMED_RUNS; run++) {
        const timing current = time_one_run(backend, graph);
        if (!current.ok) {
            return current;
        }
        if (current.nanoseconds < fastest.nanoseconds) {
            fastest = current;
        }
    }
    return fastest;
}

timing warm_up_then_time(ggml_backend_t backend, ggml_cgraph * graph) {
    const timing warm_up = time_one_run(backend, graph);
    if (!warm_up.ok) {
        return warm_up;
    }
    return fastest_of_timed_runs(backend, graph);
}

void fill(ggml_tensor * tensor, size_t elements, float value) {
    std::vector<float> data(elements, value);
    ggml_backend_tensor_set(tensor, data.data(), 0, data.size() * sizeof(float));
}

measurement measure(ggml_backend_t backend, int64_t input_length) {
    ggml_init_params params = { CONTEXT_BYTES, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * output = build_conv_transpose_1d(ctx, input_length);
    ggml_cgraph * graph  = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        printf("  %s: FAIL could not allocate tensors for input length %lld\n",
               ggml_backend_name(backend), (long long) input_length);
        ggml_free(ctx);
        return { false, 0.0, 0 };
    }

    fill(ggml_graph_get_tensor(graph, "kernel"), size_t(KERNEL_SIZE) * CHANNELS_OUT * CHANNELS_IN, KERNEL_FILL);
    fill(ggml_graph_get_tensor(graph, "input"), size_t(input_length) * CHANNELS_IN, INPUT_FILL);

    const timing  elapsed  = warm_up_then_time(backend, graph);
    const int64_t elements = ggml_nelements(output);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);

    if (!elapsed.ok) {
        printf("  %s: FAIL compute did not succeed for input length %lld\n",
               ggml_backend_name(backend), (long long) input_length);
        return { false, 0.0, 0 };
    }
    return { true, elapsed.nanoseconds / double(elements), elements };
}

outcome check_backend(ggml_backend_t backend) {
    const char * name = ggml_backend_name(backend);

    if (!backend_supports_conv_transpose_1d(backend)) {
        printf("  %s: skipped, does not support CONV_TRANSPOSE_1D\n", name);
        return outcome::skipped;
    }

    const measurement saturating = measure(backend, SATURATING_INPUT);
    const measurement scaled     = measure(backend, SCALED_INPUT);
    if (!saturating.ok || !scaled.ok) {
        return outcome::failed;
    }

    const double growth = scaled.nanoseconds_per_output / saturating.nanoseconds_per_output;
    printf("  %s: %lld outputs at %.4f ns each, %lld outputs at %.4f ns each -> growth %.2fx\n",
           name,
           (long long) saturating.output_elements, saturating.nanoseconds_per_output,
           (long long) scaled.output_elements, scaled.nanoseconds_per_output,
           growth);

    if (growth > MAX_COST_GROWTH) {
        printf("  %s: FAIL cost per output element grew %.2fx when the input grew %lldx; "
               "the work is not bounded by the contributing taps\n",
               name, growth, (long long) LENGTH_RATIO);
        return outcome::failed;
    }
    return outcome::passed;
}

bool registry_compiles_ggml_cuda_kernels(const char * name) {
    for (const char * candidate : GGML_CUDA_REGISTRIES) {
        if (name && std::strcmp(name, candidate) == 0) {
            return true;
        }
    }
    return false;
}

bool device_compiles_ggml_cuda_kernels(ggml_backend_dev_t device) {
    ggml_backend_reg_t registry = ggml_backend_dev_backend_reg(device);
    return registry && registry_compiles_ggml_cuda_kernels(ggml_backend_reg_name(registry));
}

outcome check_one_device(ggml_backend_dev_t device) {
    if (!device_compiles_ggml_cuda_kernels(device)) {
        return outcome::skipped;
    }

    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    if (!backend) {
        printf("  %s: FAIL device did not initialise, so the cost model went unchecked\n",
               ggml_backend_dev_name(device));
        return outcome::failed;
    }

    const outcome result = check_backend(backend);
    ggml_backend_free(backend);
    return result;
}

outcome check_every_device() {
    outcome overall = outcome::skipped;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        const outcome result = check_one_device(ggml_backend_dev_get(i));
        if (result == outcome::failed) {
            return outcome::failed;
        }
        if (result == outcome::passed) {
            overall = outcome::passed;
        }
    }
    return overall;
}

} // namespace

int main() {
    ggml_backend_load_all();

    switch (check_every_device()) {
        case outcome::failed:
            return 1;
        case outcome::skipped:
            printf("no ggml-cuda backend supporting CONV_TRANSPOSE_1D; nothing to check\n");
            return CTEST_SKIP_STATUS;
        case outcome::passed:
            printf("OK\n");
            return 0;
    }
    return 1;
}
