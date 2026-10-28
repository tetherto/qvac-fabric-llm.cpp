#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static const char * const OPENCL_REGISTRY_NAME = "OpenCL";

static const int64_t K = 512;
static const int64_t M = 512;

static const float WEIGHT_MAX     = 1.0f;
static const float ACTIVATION_MAX = 1000.0f;
static const float FP16_MAX       = 65504.0f;
static const double MAX_NMSE      = 5e-4;

static const size_t CONTEXT_BYTES = 16 * 1024 * 1024;

static const uint32_t LCG_MULTIPLIER  = 1664525u;
static const uint32_t LCG_INCREMENT   = 1013904223u;
static const int      RANDOM_SHIFT    = 8;
static const uint32_t RANDOM_MASK     = 0xffffff;
static const float    RANDOM_RANGE    = (float) 0x1000000;
static const uint32_t BASE_SEED       = 0x5EED;
static const uint32_t SEED_STRIDE     = 101;

struct graph_inputs {
    std::vector<uint8_t> weights_q8_0;
    std::vector<float>   activations;
};

static uint32_t next_random(uint32_t & state) {
    state = state * LCG_MULTIPLIER + LCG_INCREMENT;
    return state;
}

static float uniform_unit(uint32_t & state) {
    return ((next_random(state) >> RANDOM_SHIFT) & RANDOM_MASK) / RANDOM_RANGE;
}

static std::vector<float> fill_uniform(size_t count, float max_value, uint32_t & state) {
    std::vector<float> values(count);
    for (float & value : values) {
        value = uniform_unit(state) * max_value;
    }
    return values;
}

static std::vector<uint8_t> quantize_q8_0(const std::vector<float> & values) {
    std::vector<uint8_t> quantized(ggml_row_size(GGML_TYPE_Q8_0, K) * M);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, values.data(), quantized.data(), 0, M, K, nullptr);
    return quantized;
}

static graph_inputs make_inputs(int64_t n, uint32_t seed) {
    uint32_t state = seed;
    graph_inputs inputs;
    inputs.weights_q8_0 = quantize_q8_0(fill_uniform((size_t) (K * M), WEIGHT_MAX, state));
    inputs.activations  = fill_uniform((size_t) (K * n), ACTIVATION_MAX, state);
    return inputs;
}

static std::vector<float> run_mul_mat(ggml_backend_t backend, const graph_inputs & inputs, int64_t n) {
    ggml_init_params params = { CONTEXT_BYTES, nullptr, true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * weights     = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, K, M);
    ggml_tensor * activations = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, n);
    ggml_tensor * out         = ggml_mul_mat(ctx, weights, activations);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer != nullptr);

    ggml_backend_tensor_set(weights, inputs.weights_q8_0.data(), 0, inputs.weights_q8_0.size());
    ggml_backend_tensor_set(activations, inputs.activations.data(), 0, inputs.activations.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> result(ggml_nelements(out));
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return result;
}

static double nmse(const std::vector<float> & reference, const std::vector<float> & actual) {
    double squared_error = 0.0;
    double squared_reference = 1e-12;
    for (size_t i = 0; i < reference.size(); ++i) {
        const double difference = (double) reference[i] - (double) actual[i];
        squared_error     += difference * difference;
        squared_reference += (double) reference[i] * (double) reference[i];
    }
    return squared_error / squared_reference;
}

static float max_abs(const std::vector<float> & values) {
    float result = 0.0f;
    for (float value : values) {
        result = std::fmax(result, std::fabs(value));
    }
    return result;
}

static bool all_finite(const std::vector<float> & values) {
    for (float value : values) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

static ggml_backend_dev_t find_opencl_device() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), OPENCL_REGISTRY_NAME) == 0) {
            return dev;
        }
    }
    return nullptr;
}

static bool check_batch(ggml_backend_t cpu, ggml_backend_t opencl, int64_t n, uint32_t seed) {
    const graph_inputs inputs = make_inputs(n, seed);
    const std::vector<float> reference = run_mul_mat(cpu, inputs, n);
    const std::vector<float> actual    = run_mul_mat(opencl, inputs, n);

    const float  reference_max = max_abs(reference);
    const bool   finite        = all_finite(actual);
    const double error         = nmse(reference, actual);
    const bool   ok = reference_max > FP16_MAX && finite && error < MAX_NMSE;

    std::printf("%s n=%3lld reference_max=%.0f finite=%d nmse=%.3e\n",
                ok ? "ok  " : "FAIL", (long long) n, reference_max, finite ? 1 : 0, error);
    return ok;
}

static int count_failures(ggml_backend_t cpu, ggml_backend_t opencl) {
    const int64_t batches[] = { 1, 8, 63, 64, 750 };
    uint32_t seed = BASE_SEED;
    int failures = 0;
    for (int64_t n : batches) {
        seed += SEED_STRIDE;
        failures += check_batch(cpu, opencl, n, seed) ? 0 : 1;
    }
    return failures;
}

int main() {
    ggml_backend_load_all();

    ggml_backend_dev_t opencl_device = find_opencl_device();
    if (opencl_device == nullptr) {
        std::printf("no OpenCL device found - test skipped\n");
        return 77;
    }

    ggml_backend_t cpu    = ggml_backend_dev_init(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr);
    ggml_backend_t opencl = ggml_backend_dev_init(opencl_device, nullptr);
    GGML_ASSERT(cpu != nullptr && opencl != nullptr);

    const int failures = count_failures(cpu, opencl);

    ggml_backend_free(opencl);
    ggml_backend_free(cpu);

    if (failures > 0) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("all ok\n");
    return 0;
}
