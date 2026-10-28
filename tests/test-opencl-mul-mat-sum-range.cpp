#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
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

static const float   CANCELLING_WEIGHT   = 2.0f;
static const float   CANCELLING_NEGATIVE = -500.0f;
static const float   CANCELLING_POSITIVE = 1500.0f;
static const int64_t CANCELLING_RUN      = 32;

static const size_t CONTEXT_BYTES = 16 * 1024 * 1024;

static const uint32_t LCG_MULTIPLIER  = 1664525u;
static const uint32_t LCG_INCREMENT   = 1013904223u;
static const int      RANDOM_SHIFT    = 8;
static const uint32_t RANDOM_MASK     = 0xffffff;
static const float    RANDOM_RANGE    = (float) 0x1000000;
static const uint32_t BASE_SEED       = 0x5EED;
static const uint32_t SEED_STRIDE     = 101;

static const ggml_type WEIGHT_TYPES[] = {
    GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
    GGML_TYPE_Q1_0, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ4_NL,
};
static const int64_t BATCHES[] = { 1, 8, 63, 64, 750 };

struct graph_inputs {
    ggml_type            weight_type;
    std::vector<uint8_t> weights;
    std::vector<float>   activations;
};

struct batch_result {
    float  reference_max;
    bool   finite;
    double error;
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

static std::vector<uint8_t> quantize(ggml_type type, const std::vector<float> & values) {
    std::vector<uint8_t> quantized(ggml_row_size(type, K) * M);
    ggml_quantize_chunk(type, values.data(), quantized.data(), 0, M, K, nullptr);
    return quantized;
}

static graph_inputs make_inputs(ggml_type type, int64_t n, uint32_t seed) {
    uint32_t state = seed;
    graph_inputs inputs;
    inputs.weight_type = type;
    inputs.weights     = quantize(type, fill_uniform((size_t) (K * M), WEIGHT_MAX, state));
    inputs.activations = fill_uniform((size_t) (K * n), ACTIVATION_MAX, state);
    return inputs;
}

static void fill_cancelling_column(float * column) {
    std::fill_n(column, CANCELLING_RUN, CANCELLING_NEGATIVE);
    std::fill_n(column + CANCELLING_RUN, CANCELLING_RUN, CANCELLING_POSITIVE);
}

static std::vector<float> fill_cancelling_columns(int64_t n) {
    std::vector<float> values((size_t) (K * n), 0.0f);
    for (int64_t column = 0; column < n; ++column) {
        fill_cancelling_column(values.data() + column * K);
    }
    return values;
}

static graph_inputs make_cancelling_inputs(ggml_type type, int64_t n) {
    graph_inputs inputs;
    inputs.weight_type = type;
    inputs.weights     = quantize(type, std::vector<float>((size_t) (K * M), CANCELLING_WEIGHT));
    inputs.activations = fill_cancelling_columns(n);
    return inputs;
}

static std::vector<float> run_mul_mat(ggml_backend_t backend, const graph_inputs & inputs, int64_t n) {
    ggml_init_params params = { CONTEXT_BYTES, nullptr, true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * weights     = ggml_new_tensor_2d(ctx, inputs.weight_type, K, M);
    ggml_tensor * activations = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, n);
    ggml_tensor * out         = ggml_mul_mat(ctx, weights, activations);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer != nullptr);

    ggml_backend_tensor_set(weights, inputs.weights.data(), 0, inputs.weights.size());
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

static batch_result compare_backends(ggml_backend_t cpu, ggml_backend_t opencl, const graph_inputs & inputs, int64_t n) {
    const std::vector<float> reference = run_mul_mat(cpu, inputs, n);
    const std::vector<float> actual    = run_mul_mat(opencl, inputs, n);
    return { max_abs(reference), all_finite(actual), nmse(reference, actual) };
}

static bool report_batch(const char * kind, ggml_type type, int64_t n, const batch_result & result, bool ok) {
    std::printf("%s %-6s %-6s n=%3lld reference_max=%.0f finite=%d nmse=%.3e\n",
                ok ? "ok  " : "FAIL", kind, ggml_type_name(type), (long long) n,
                result.reference_max, result.finite ? 1 : 0, result.error);
    return ok;
}

static bool check_sum_range_batch(ggml_backend_t cpu, ggml_backend_t opencl, ggml_type type, int64_t n, uint32_t seed) {
    const batch_result result = compare_backends(cpu, opencl, make_inputs(type, n, seed), n);
    const bool ok = result.reference_max > FP16_MAX && result.finite && result.error < MAX_NMSE;
    return report_batch("range", type, n, result, ok);
}

static bool check_cancelling_batch(ggml_backend_t cpu, ggml_backend_t opencl, ggml_type type, int64_t n) {
    const batch_result result = compare_backends(cpu, opencl, make_cancelling_inputs(type, n), n);
    const bool ok = result.finite && result.error < MAX_NMSE;
    return report_batch("cancel", type, n, result, ok);
}

static int count_type_failures(ggml_backend_t cpu, ggml_backend_t opencl, ggml_type type, uint32_t & seed) {
    int failures = 0;
    for (int64_t n : BATCHES) {
        seed += SEED_STRIDE;
        failures += check_sum_range_batch(cpu, opencl, type, n, seed) ? 0 : 1;
        failures += check_cancelling_batch(cpu, opencl, type, n) ? 0 : 1;
    }
    return failures;
}

static int count_failures(ggml_backend_t cpu, ggml_backend_t opencl) {
    uint32_t seed = BASE_SEED;
    int failures = 0;
    for (ggml_type type : WEIGHT_TYPES) {
        failures += count_type_failures(cpu, opencl, type, seed);
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
