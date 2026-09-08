#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-xdna.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using backend_ptr = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
using buffer_ptr  = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;

using get_npu_call_count_fn = uint64_t (*)(ggml_backend_t backend);

static constexpr int SKIP = 77;

struct matrix_data {
    std::vector<uint8_t> weights;
    std::vector<float>   activations;
};

static matrix_data make_data(ggml_type type, int64_t k, int64_t n, int64_t m) {
    std::vector<float> weights_f32((size_t) k * n);
    std::vector<float> activations((size_t) k * m);

    for (size_t i = 0; i < weights_f32.size(); ++i) {
        weights_f32[i] = 0.125f * std::sin((float) (i % 251) * 0.071f);
    }
    for (size_t i = 0; i < activations.size(); ++i) {
        activations[i] = 0.25f * std::cos((float) (i % 127) * 0.053f);
    }

    std::vector<uint8_t> weights(ggml_row_size(type, k) * n);
    if (type == GGML_TYPE_F32) {
        std::memcpy(weights.data(), weights_f32.data(), weights.size());
    } else {
        const size_t written = ggml_quantize_chunk(
            type, weights_f32.data(), weights.data(), 0, n, k, nullptr);
        if (written != weights.size()) {
            throw std::runtime_error("weight quantization size mismatch");
        }
    }

    return { std::move(weights), std::move(activations) };
}

static std::vector<float> run_standard(
        ggml_backend_t backend, ggml_type type,
        int64_t k, int64_t n, int64_t m, const matrix_data & data,
        bool * supported = nullptr) {
    context_ptr ctx(ggml_init({
        /* .mem_size   = */ 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    }), ggml_free);
    if (!ctx) {
        throw std::runtime_error("ggml_init failed");
    }

    ggml_tensor * weights = ggml_new_tensor_2d(ctx.get(), type, k, n);
    ggml_tensor * acts    = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, m);
    ggml_tensor * output  = ggml_mul_mat(ctx.get(), weights, acts);
    ggml_cgraph * graph   = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);

    buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
    if (!buffer) {
        throw std::runtime_error("standard tensor allocation failed");
    }

    ggml_backend_tensor_set(weights, data.weights.data(), 0, data.weights.size());
    ggml_backend_tensor_set(acts, data.activations.data(), 0, data.activations.size() * sizeof(float));
    if (supported) {
        *supported = ggml_backend_supports_op(backend, output);
    }

    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("standard graph compute failed");
    }

    std::vector<float> result(ggml_nelements(output));
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
    return result;
}

static std::vector<float> run_repacked(
        ggml_backend_t backend, ggml_backend_buffer_type_t repack_buft,
        int64_t k, int64_t n, int64_t m, const matrix_data & data) {
    context_ptr ctx(ggml_init({
        /* .mem_size   = */ 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    }), ggml_free);
    if (!ctx) {
        throw std::runtime_error("ggml_init failed");
    }

    ggml_tensor * weights = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q4_K, k, n);
    ggml_set_name(weights, "phase1.weight");

    buffer_ptr weight_buffer(
        ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), repack_buft),
        ggml_backend_buffer_free);
    if (!weight_buffer) {
        throw std::runtime_error("XDNA_REPACK allocation failed");
    }
    if (std::strcmp(ggml_backend_buffer_name(weight_buffer.get()), "XDNA_REPACK") != 0) {
        throw std::runtime_error("weight did not use XDNA_REPACK");
    }
    ggml_backend_tensor_set(weights, data.weights.data(), 0, data.weights.size());

    ggml_tensor * acts   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, m);
    ggml_tensor * output = ggml_mul_mat(ctx.get(), weights, acts);
    ggml_set_name(output, "phase1.mul_mat");
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, output);

    buffer_ptr compute_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
    if (!compute_buffer) {
        throw std::runtime_error("XDNA compute tensor allocation failed");
    }
    ggml_backend_tensor_set(acts, data.activations.data(), 0, data.activations.size() * sizeof(float));

    if (!ggml_backend_supports_op(backend, output)) {
        throw std::runtime_error("XDNA does not support the repacked MUL_MAT");
    }
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("XDNA graph compute failed");
    }

    std::vector<float> result(ggml_nelements(output));
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
    return result;
}

static double nmse(const std::vector<float> & ref, const std::vector<float> & got) {
    double square_error = 0.0;
    double square_ref   = 1e-30;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(got[i])) {
            return INFINITY;
        }
        const double diff = (double) got[i] - ref[i];
        square_error += diff * diff;
        square_ref   += (double) ref[i] * ref[i];
    }
    return square_error / square_ref;
}

static get_npu_call_count_fn get_call_counter(ggml_backend_reg_t reg) {
    return (get_npu_call_count_fn) ggml_backend_reg_get_proc_address(
        reg, "ggml_backend_xdna_get_npu_call_count");
}

static int test_micro() {
    constexpr int64_t K = 256;
    constexpr int64_t N = 512;
    constexpr int64_t M = 128;
    constexpr double TOL = 5e-4;

    ggml_backend_reg_t reg = ggml_backend_xdna_reg();
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    backend_ptr xdna(ggml_backend_xdna_init(), ggml_backend_free);
    backend_ptr cpu(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr), ggml_backend_free);
    if (!xdna || !cpu) {
        throw std::runtime_error("backend initialization failed");
    }

    auto get_extra = (ggml_backend_dev_get_extra_bufts_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    get_npu_call_count_fn get_calls = get_call_counter(reg);
    ggml_backend_buffer_type_t * extra = get_extra ? get_extra(dev) : nullptr;
    if (!extra || !extra[0]) {
        std::printf("SKIP: XDNA NPU or GEMM artifact is unavailable\n");
        return SKIP;
    }
    if (!get_calls) {
        throw std::runtime_error("XDNA NPU call counter is unavailable");
    }

    matrix_data data = make_data(GGML_TYPE_Q4_K, K, N, M);
    const std::vector<float> ref = run_standard(cpu.get(), GGML_TYPE_Q4_K, K, N, M, data);
    const uint64_t calls_before = get_calls(xdna.get());
    const std::vector<float> got = run_repacked(xdna.get(), extra[0], K, N, M, data);
    const uint64_t calls_after = get_calls(xdna.get());
    const double error = nmse(ref, got);

    std::printf("micro: Q4_K K=%lld N=%lld M=%lld buffer=XDNA_REPACK calls=%llu->%llu nmse=%.3e tol=%.1e\n",
        (long long) K, (long long) N, (long long) M,
        (unsigned long long) calls_before, (unsigned long long) calls_after, error, TOL);
    if (calls_after <= calls_before) {
        std::fprintf(stderr, "FAIL: supported MUL_MAT did not submit to the NPU\n");
        return 1;
    }
    if (error > TOL) {
        std::fprintf(stderr, "FAIL: micro NMSE exceeds tolerance\n");
        return 1;
    }
    return 0;
}

static int test_fallback(bool npu_disabled) {
    const int64_t K = npu_disabled ? 16 : 256;
    const int64_t N = npu_disabled ? 17 : 513;
    const int64_t M = npu_disabled ? 4 : 128;
    const ggml_type type = npu_disabled ? GGML_TYPE_F32 : GGML_TYPE_Q4_K;
    constexpr double TOL = 1e-12;

    ggml_backend_reg_t reg = ggml_backend_xdna_reg();
    backend_ptr xdna(ggml_backend_xdna_init(), ggml_backend_free);
    backend_ptr cpu(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr), ggml_backend_free);
    if (!xdna || !cpu) {
        throw std::runtime_error("backend initialization failed");
    }
    get_npu_call_count_fn get_calls = get_call_counter(reg);
    if (!get_calls) {
        throw std::runtime_error("XDNA NPU call counter is unavailable");
    }

    matrix_data data = make_data(type, K, N, M);
    const std::vector<float> ref = run_standard(cpu.get(), type, K, N, M, data);
    const uint64_t calls_before = get_calls(xdna.get());
    bool claimed = false;
    const std::vector<float> got = run_standard(xdna.get(), type, K, N, M, data, &claimed);
    const uint64_t calls_after = get_calls(xdna.get());
    const double error = nmse(ref, got);

    std::printf("%s: %s K=%lld N=%lld M=%lld claimed=%s calls=%llu->%llu nmse=%.3e tol=%.1e\n",
        npu_disabled ? "no-npu" : "fallback", ggml_type_name(type),
        (long long) K, (long long) N, (long long) M, claimed ? "yes" : "no",
        (unsigned long long) calls_before, (unsigned long long) calls_after, error, TOL);
    if (!npu_disabled && !claimed) {
        std::fprintf(stderr, "FAIL: own_graph did not claim the unsupported shape\n");
        return 1;
    }
    if (calls_after != calls_before) {
        std::fprintf(stderr, "FAIL: CPU fallback incremented the NPU call count\n");
        return 1;
    }
    if (error > TOL) {
        std::fprintf(stderr, "FAIL: CPU fallback differs from the CPU reference\n");
        return 1;
    }
    return 0;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s micro|fallback|no-npu\n", argv[0]);
        return 2;
    }

    try {
        const std::string mode = argv[1];
        int result;
        if (mode == "micro") {
            result = test_micro();
        } else if (mode == "fallback") {
            result = test_fallback(false);
        } else if (mode == "no-npu") {
            result = test_fallback(true);
        } else {
            std::fprintf(stderr, "unknown mode: %s\n", argv[1]);
            return 2;
        }
        ggml_quantize_free();
        return result;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        ggml_quantize_free();
        return 1;
    }
}
