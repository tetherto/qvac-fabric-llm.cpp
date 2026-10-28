#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static const char * const WEBGPU_REGISTRY_NAME = "WebGPU";

static const int64_t M = 96;

static const float  UNIT_MAX           = 1.0f;
static const float  F32_ACTIVATION_MAX = 1.0e5f;
static const float  F16_OPERAND_MAX    = 256.0f;
static const float  FP16_MAX           = 65504.0f;
static const float  ANY_REFERENCE_MAX  = 0.0f;
static const double MAX_NMSE           = 1e-9;

static const size_t CONTEXT_BYTES = 16 * 1024 * 1024;

static const uint32_t LCG_MULTIPLIER = 1664525u;
static const uint32_t LCG_INCREMENT  = 1013904223u;
static const int      RANDOM_SHIFT   = 8;
static const uint32_t RANDOM_MASK    = 0xffffff;
static const float    RANDOM_RANGE   = (float) 0x1000000;
static const uint32_t BASE_SEED      = 0x5EED;
static const uint32_t SEED_STRIDE    = 101;

struct graph_inputs {
    std::vector<float> weights;
    std::vector<float> activations;
};

struct mul_mat_shape {
    int64_t   k;
    int64_t   n;
    ggml_type weight_type;
    ggml_type activation_type;
    float     weight_max;
    float     activation_max;
    ggml_prec prec;
    float     min_reference_max;
};

static uint32_t next_random(uint32_t & state) {
    state = state * LCG_MULTIPLIER + LCG_INCREMENT;
    return state;
}

static float uniform_symmetric(uint32_t & state, float max_value) {
    const float unit = ((next_random(state) >> RANDOM_SHIFT) & RANDOM_MASK) / RANDOM_RANGE;
    return (2.0f * unit - 1.0f) * max_value;
}

static std::vector<float> fill_uniform(size_t count, float max_value, uint32_t & state) {
    std::vector<float> values(count);
    for (float & value : values) {
        value = uniform_symmetric(state, max_value);
    }
    return values;
}

static std::vector<ggml_fp16_t> to_f16(const std::vector<float> & values) {
    std::vector<ggml_fp16_t> converted(values.size());
    ggml_fp32_to_fp16_row(values.data(), converted.data(), (int64_t) values.size());
    return converted;
}

static std::vector<float> round_to_f16(const std::vector<float> & values) {
    const std::vector<ggml_fp16_t> converted = to_f16(values);
    std::vector<float>             rounded(values.size());
    ggml_fp16_to_fp32_row(converted.data(), rounded.data(), (int64_t) values.size());
    return rounded;
}

static std::vector<float> fill_operand(size_t count, float max_value, ggml_type type, uint32_t & state) {
    const std::vector<float> values = fill_uniform(count, max_value, state);
    return type == GGML_TYPE_F16 ? round_to_f16(values) : values;
}

static graph_inputs make_inputs(const mul_mat_shape & shape, uint32_t seed) {
    uint32_t     state = seed;
    graph_inputs inputs;
    inputs.weights     = fill_operand((size_t) (shape.k * M), shape.weight_max, shape.weight_type, state);
    inputs.activations = fill_operand((size_t) (shape.k * shape.n), shape.activation_max, shape.activation_type, state);
    return inputs;
}

static void set_values(ggml_tensor * tensor, const std::vector<float> & values) {
    if (tensor->type == GGML_TYPE_F16) {
        const std::vector<ggml_fp16_t> converted = to_f16(values);
        ggml_backend_tensor_set(tensor, converted.data(), 0, converted.size() * sizeof(ggml_fp16_t));
        return;
    }
    ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
}

static double dot_row(const graph_inputs & inputs, const mul_mat_shape & shape, int64_t row, int64_t col) {
    double sum = 0.0;
    for (int64_t i = 0; i < shape.k; ++i) {
        const double weight     = inputs.weights[row * shape.k + i];
        const double activation = inputs.activations[col * shape.k + i];
        sum += weight * activation;
    }
    return sum;
}

static void fill_reference_column(const graph_inputs &  inputs,
                                  const mul_mat_shape & shape,
                                  int64_t               col,
                                  std::vector<float> &  result) {
    for (int64_t row = 0; row < M; ++row) {
        result[col * M + row] = (float) dot_row(inputs, shape, row, col);
    }
}

static std::vector<float> reference_mul_mat(const graph_inputs & inputs, const mul_mat_shape & shape) {
    std::vector<float> result((size_t) (M * shape.n));
    for (int64_t col = 0; col < shape.n; ++col) {
        fill_reference_column(inputs, shape, col, result);
    }
    return result;
}

static std::vector<float> run_mul_mat(ggml_backend_t        backend,
                                      const graph_inputs &  inputs,
                                      const mul_mat_shape & shape) {
    ggml_init_params params = { CONTEXT_BYTES, nullptr, true };
    ggml_context *   ctx    = ggml_init(params);

    ggml_tensor * weights     = ggml_new_tensor_2d(ctx, shape.weight_type, shape.k, M);
    ggml_tensor * activations = ggml_new_tensor_2d(ctx, shape.activation_type, shape.k, shape.n);
    ggml_tensor * out         = ggml_mul_mat(ctx, weights, activations);
    ggml_mul_mat_set_prec(out, shape.prec);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer != nullptr);

    set_values(weights, inputs.weights);
    set_values(activations, inputs.activations);
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> result(ggml_nelements(out));
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return result;
}

static double nmse(const std::vector<float> & reference, const std::vector<float> & actual) {
    double squared_error     = 0.0;
    double squared_reference = 1e-12;
    for (size_t i = 0; i < reference.size(); ++i) {
        const double difference = (double) reference[i] - (double) actual[i];
        squared_error += difference * difference;
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

static ggml_backend_dev_t find_webgpu_device() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), WEBGPU_REGISTRY_NAME) == 0) {
            return dev;
        }
    }
    return nullptr;
}

static bool check_shape(ggml_backend_t webgpu, const mul_mat_shape & shape, uint32_t seed) {
    const graph_inputs       inputs    = make_inputs(shape, seed);
    const std::vector<float> reference = reference_mul_mat(inputs, shape);
    const std::vector<float> actual    = run_mul_mat(webgpu, inputs, shape);

    const float  reference_max = max_abs(reference);
    const bool   finite        = all_finite(actual);
    const double error         = nmse(reference, actual);
    const bool   ok            = reference_max > shape.min_reference_max && finite && error < MAX_NMSE;

    std::printf("%s %s x %s prec=%s k=%3lld n=%3lld reference_max=%.0f finite=%d nmse=%.3e\n", ok ? "ok  " : "FAIL",
                ggml_type_name(shape.weight_type), ggml_type_name(shape.activation_type),
                shape.prec == GGML_PREC_F32 ? "f32" : "default", (long long) shape.k, (long long) shape.n,
                reference_max, finite ? 1 : 0, error);
    return ok;
}

static int count_failures(ggml_backend_t webgpu) {
    const mul_mat_shape shapes[] = {
        { 512, 1, GGML_TYPE_F16, GGML_TYPE_F32, UNIT_MAX, F32_ACTIVATION_MAX, GGML_PREC_F32, FP16_MAX },
        { 512, 4, GGML_TYPE_F16, GGML_TYPE_F32, UNIT_MAX, F32_ACTIVATION_MAX, GGML_PREC_F32, FP16_MAX },
        { 512, 5, GGML_TYPE_F16, GGML_TYPE_F32, UNIT_MAX, F32_ACTIVATION_MAX, GGML_PREC_F32, FP16_MAX },
        { 512, 63, GGML_TYPE_F16, GGML_TYPE_F32, UNIT_MAX, F32_ACTIVATION_MAX, GGML_PREC_F32, FP16_MAX },
        { 512, 129, GGML_TYPE_F16, GGML_TYPE_F32, UNIT_MAX, F32_ACTIVATION_MAX, GGML_PREC_F32, FP16_MAX },
        { 513, 1, GGML_TYPE_F16, GGML_TYPE_F32, UNIT_MAX, F32_ACTIVATION_MAX, GGML_PREC_F32, FP16_MAX },
        { 513, 63, GGML_TYPE_F16, GGML_TYPE_F32, UNIT_MAX, F32_ACTIVATION_MAX, GGML_PREC_F32, FP16_MAX },
        { 512, 1, GGML_TYPE_F16, GGML_TYPE_F16, F16_OPERAND_MAX, F16_OPERAND_MAX, GGML_PREC_F32, FP16_MAX },
        { 512, 4, GGML_TYPE_F16, GGML_TYPE_F16, F16_OPERAND_MAX, F16_OPERAND_MAX, GGML_PREC_F32, FP16_MAX },
        { 512, 63, GGML_TYPE_F16, GGML_TYPE_F16, F16_OPERAND_MAX, F16_OPERAND_MAX, GGML_PREC_F32, FP16_MAX },
        { 512, 1, GGML_TYPE_F32, GGML_TYPE_F32, UNIT_MAX, UNIT_MAX, GGML_PREC_DEFAULT, ANY_REFERENCE_MAX },
        { 512, 63, GGML_TYPE_F32, GGML_TYPE_F32, UNIT_MAX, UNIT_MAX, GGML_PREC_DEFAULT, ANY_REFERENCE_MAX },
        { 513, 63, GGML_TYPE_F32, GGML_TYPE_F32, UNIT_MAX, UNIT_MAX, GGML_PREC_DEFAULT, ANY_REFERENCE_MAX },
        { 512, 129, GGML_TYPE_F32, GGML_TYPE_F32, UNIT_MAX, UNIT_MAX, GGML_PREC_DEFAULT, ANY_REFERENCE_MAX },
    };
    uint32_t seed     = BASE_SEED;
    int      failures = 0;
    for (const mul_mat_shape & shape : shapes) {
        seed += SEED_STRIDE;
        failures += check_shape(webgpu, shape, seed) ? 0 : 1;
    }
    return failures;
}

int main() {
    ggml_backend_load_all();

    ggml_backend_dev_t webgpu_device = find_webgpu_device();
    if (webgpu_device == nullptr) {
        std::printf("no WebGPU device found - test skipped\n");
        return 77;
    }

    ggml_backend_t webgpu = ggml_backend_dev_init(webgpu_device, nullptr);
    GGML_ASSERT(webgpu != nullptr);

    const int failures = count_failures(webgpu);

    ggml_backend_free(webgpu);

    if (failures > 0) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("all ok\n");
    return 0;
}
