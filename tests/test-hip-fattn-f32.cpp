#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-cuda/fattn-tile-precision.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
constexpr int64_t head_size = 128;
constexpr size_t context_bytes = 2 * 1024 * 1024;
constexpr int skip = 77;
constexpr double tolerance = 2e-5;

bool check_dispatch() {
    ggml_tensor q = {}, k = {}, other = {}, out = {};
    q.ne[1] = 33;
    q.ne[2] = k.ne[2] = 2;
    out.op = GGML_OP_FLASH_ATTN_EXT;
    out.src[0] = &q;
    out.src[1] = &k;
    if (!ggml_prec_set_acc(&out, GGML_PREC_F32)) { return false; }
    const auto selected = [&](bool device = true) {
        return ggml_cuda_fattn_tile_use_f32_pv<128, 128, 1, false>(device, &out);
    };
    if (!selected() || selected(false) ||
        ggml_cuda_fattn_tile_use_f32_pv<64, 64, 1, false>(true, &out) ||
        ggml_cuda_fattn_tile_use_f32_pv<128, 64, 1, false>(true, &out) ||
        ggml_cuda_fattn_tile_use_f32_pv<128, 128, 2, false>(true, &out) ||
        ggml_cuda_fattn_tile_use_f32_pv<128, 128, 1, true>(true, &out) ||
        ggml_cuda_fattn_tile_use_f32_pv<128, 128, 1, false, true>(true, &out)) {
        return false;
    }
    q.ne[1] = 32;
    if (selected()) { return false; }
    q.ne[1] = 33;
    if (!ggml_prec_set_acc(&out, GGML_PREC_UNDEFINED)) { return false; }
    if (selected()) { return false; }
    if (!ggml_prec_set_acc(&out, GGML_PREC_F32)) { return false; }
    out.src[3] = &other;
    if (selected()) { return false; }
    out.src[3] = nullptr;
    out.src[4] = &other;
    if (selected()) { return false; }
    out.src[4] = nullptr;
    k.ne[2] = 1;
    return !selected();
}

struct test_shape {
    int64_t queries;
    int64_t keys;
    int64_t heads;
    int64_t batches;
    float amplitude;
    ggml_prec precision;
    bool weighted = false;
};

float value_at(int64_t channel, int64_t token, int64_t head, int64_t batch, float amplitude) {
    const float sign = channel % 2 == 0 ? 1.0f : -1.0f;
    return sign * amplitude + float(token % 4) * 0.25f + float(head + batch) * 0.5f;
}

std::vector<float> make_values(const test_shape & shape) {
    std::vector<float> values(head_size * shape.keys * shape.heads * shape.batches);
    for (size_t i = 0; i < values.size(); ++i) {
        const int64_t channel = i % head_size;
        const int64_t token = (i / head_size) % shape.keys;
        const int64_t head = (i / (head_size * shape.keys)) % shape.heads;
        const int64_t batch = i / (head_size * shape.keys * shape.heads);
        values[i] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(value_at(channel, token, head, batch, shape.amplitude)));
    }
    return values;
}

float key_logit(const test_shape & shape, int64_t token) {
    constexpr int segments = 4;
    return shape.weighted ? float(token / (shape.keys / segments)) * 0.5f : 0.0f;
}

void fill_queries(ggml_tensor * q) {
    std::vector<float> values(ggml_nelements(q), 0.0f);
    for (size_t i = 0; i < values.size(); i += head_size) {
        values[i] = 1.0f;
    }
    ggml_backend_tensor_set(q, values.data(), 0, values.size() * sizeof(float));
}

void fill_keys(ggml_tensor * k, const test_shape & shape) {
    std::vector<float> values(ggml_nelements(k), 0.0f);
    for (size_t i = 0; i < values.size(); i += head_size) {
        values[i] = key_logit(shape, (i / head_size) % shape.keys);
    }
    ggml_backend_tensor_set(k, values.data(), 0, values.size() * sizeof(float));
}

double reference_denominator(const test_shape & shape) {
    double denominator = 0.0;
    for (int64_t token = 0; token < shape.keys; ++token) {
        denominator += std::exp(double(key_logit(shape, token)));
    }
    return denominator;
}

void normalize_means(std::vector<double> & means, double denominator) {
    for (double & mean : means) {
        mean /= denominator;
    }
}

std::vector<double> reference_means(const test_shape & shape, const std::vector<float> & values) {
    std::vector<double> means(head_size * shape.heads * shape.batches, 0.0);
    for (size_t i = 0; i < values.size(); ++i) {
        const size_t channel = i % head_size;
        const size_t plane = i / (head_size * shape.keys);
        const size_t token = (i / head_size) % shape.keys;
        means[plane * head_size + channel] += double(values[i]) * std::exp(double(key_logit(shape, token)));
    }
    normalize_means(means, reference_denominator(shape));
    return means;
}

bool compare_output(const test_shape & shape, ggml_tensor * output, const std::vector<double> & means) {
    constexpr double weighted_tolerance = 1e-3;
    const double limit = shape.weighted ? weighted_tolerance : tolerance;
    std::vector<float> actual(ggml_nelements(output));
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    for (size_t i = 0; i < actual.size(); ++i) {
        const size_t channel = i % head_size;
        const size_t head = (i / head_size) % shape.heads;
        const size_t batch = i / (head_size * shape.heads * shape.queries);
        const double expected = means[(batch * shape.heads + head) * head_size + channel];
        if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected) > limit * (1.0 + std::abs(expected))) {
            std::fprintf(stderr, "output[%zu]: expected %.9g, got %.9g\n", i, expected, actual[i]);
            return false;
        }
    }
    return true;
}

bool run_graph(ggml_backend_t backend, ggml_context * ctx, const test_shape & shape) {
    auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_size, shape.queries, shape.heads, shape.batches);
    auto * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_size, shape.keys, shape.heads, shape.batches);
    auto * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_size, shape.keys, shape.heads, shape.batches);
    auto * out = ggml_flash_attn_ext(ctx, q, k, v, nullptr, 1.0f, 0.0f, 0.0f);
    if (!ggml_prec_set_acc(out, shape.precision)) { return false; }
    if (!ggml_backend_supports_op(backend, out)) {
        std::fprintf(stderr, "Selected backend does not support the test operation\n");
        return false;
    }
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { return false; }
    ggml_backend_buffer_clear(buffer, 0);
    fill_queries(q);
    fill_keys(k, shape);
    const auto values = make_values(shape);
    ggml_backend_tensor_set(v, values.data(), 0, values.size() * sizeof(float));
    const bool computed = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    const bool passed = computed && compare_output(shape, out, reference_means(shape, values));
    ggml_backend_buffer_free(buffer);
    return passed;
}

bool run_case(ggml_backend_t backend, const test_shape & shape) {
    ggml_context * ctx = ggml_init({context_bytes, nullptr, true});
    if (!ctx) { return false; }
    const bool passed = run_graph(backend, ctx, shape);
    ggml_free(ctx);
    std::printf("%s: queries=%lld keys=%lld heads=%lld batches=%lld amplitude=%g precision=%d weighted=%d: %s\n",
                ggml_backend_name(backend), (long long) shape.queries, (long long) shape.keys,
                (long long) shape.heads, (long long) shape.batches, shape.amplitude, shape.precision, shape.weighted,
                passed ? "PASS" : "FAIL");
    return passed;
}

bool run_cases(ggml_backend_t backend) {
    const test_shape cases[] = {
        {33, 4096, 1, 1, 256.0f, GGML_PREC_F32},
        {65, 8192, 2, 2, 256.0f, GGML_PREC_F32},
        {33, 4096, 2, 1, 0.0f, GGML_PREC_F32},
        {65, 4096, 2, 1, 0.0f, GGML_PREC_F32, true},
        {32, 256,  1, 1, 0.0f, GGML_PREC_F32},
        {33, 256,  1, 1, 0.0f, GGML_PREC_UNDEFINED},
    };
    bool passed = true;
    for (const auto & shape : cases) {
        passed = run_case(backend, shape) && passed;
    }
    return passed;
}
}

int main(int argc, char ** argv) {
    if (!check_dispatch()) {
        std::fprintf(stderr, "FP32 tile dispatch guards failed\n");
        return 1;
    }
    if (argc == 2 && std::strcmp(argv[1], "dispatch") == 0) { return 0; }
    ggml_backend_load_all();
    const char * name = argc == 2 ? argv[1] : "CPU";
    auto * device = ggml_backend_dev_by_name(name);
    if (!device) {
        std::printf("SKIP: backend %s unavailable\n", name);
        return skip;
    }
    auto * backend = ggml_backend_dev_init(device, nullptr);
    if (!backend) { return 1; }
    const bool passed = run_cases(backend);
    ggml_backend_free(backend);
    return passed ? 0 : 1;
}
