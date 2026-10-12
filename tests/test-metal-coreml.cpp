#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "../ggml/src/ggml-metal/ggml-metal-coreml-policy.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>

static void check(bool value, const char * message) {
    if (!value) { throw std::runtime_error(message); }
}

using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
using backend_ptr = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;

static int parse_int(const char * text, int limit) {
    char * end = nullptr;
    errno = 0;
    const long value = std::strtol(text, &end, 10);
    check(errno == 0 && end != text && *end == '\0' && value > 0 && value <= limit, "invalid integer argument");
    return (int) value;
}

static void policy_tests() {
    context_ptr owned_ctx(ggml_init({1024*1024, nullptr, true}), ggml_free);
    ggml_context * ctx = owned_ctx.get();
    check(ctx != nullptr, "context");
    auto * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 512, 512);
    auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 64);
    auto * y = ggml_mul_mat(ctx, w, x);
    check(ggml_metal_coreml_shape_supported(y), "large Q8/F32 should qualify");
    check(!ggml_metal_coreml_shape_supported(nullptr), "null op must fall back");
    auto * h = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 512, 64);
    check(ggml_metal_coreml_shape_supported(ggml_mul_mat(ctx, w, h)), "Q8/F16 should qualify");
    ggml_tensor altered = *y;
    altered.op = GGML_OP_MUL_MAT_ID;
    check(!ggml_metal_coreml_shape_supported(&altered), "MoE must fall back");
    altered = *y; altered.nb[1] += 4;
    check(!ggml_metal_coreml_shape_supported(&altered), "output stride must fall back");
    ggml_tensor source = *w;
    altered = *y; altered.src[0] = &source;
    source.type = GGML_TYPE_Q4_0;
    check(!ggml_metal_coreml_shape_supported(&altered), "other quantization must fall back");
    source = *w; source.ne[2] = 2;
    check(!ggml_metal_coreml_shape_supported(&altered), "batch must fall back");
    source = *w; source.nb[1] += 34;
    check(!ggml_metal_coreml_shape_supported(&altered), "weight stride must fall back");
    ggml_tensor activation = *x;
    ggml_tensor layout = *y; layout.src[1] = &activation;
    activation.nb[1] += sizeof(float);
    check(!ggml_metal_coreml_shape_supported(&layout), "activation stride must fall back");
    activation = *x; activation.ne[3] = 2;
    check(!ggml_metal_coreml_shape_supported(&layout), "activation batch must fall back");
    source = *w; source.ne[0] = INT64_MAX;
    check(!ggml_metal_coreml_shape_supported(&altered), "huge dimensions must fall back");
    source = *w; source.ne[0] = -32;
    check(!ggml_metal_coreml_shape_supported(&altered), "negative dimension must fall back");
    altered = *y; altered.src[1] = nullptr;
    check(!ggml_metal_coreml_shape_supported(&altered), "missing input must fall back");
    altered = *y; altered.ne[1] += 32;
    check(!ggml_metal_coreml_shape_supported(&altered), "output shape mismatch must fall back");
    altered = *y; altered.type = GGML_TYPE_F16;
    check(!ggml_metal_coreml_shape_supported(&altered), "F16 output must fall back");
    altered = *y;
    const int32_t hint = GGML_HINT_SRC0_IS_HADAMARD;
    std::memcpy((char *) altered.op_params + sizeof(int32_t), &hint, sizeof(hint));
    check(!ggml_metal_coreml_shape_supported(&altered), "hint must fall back");
    auto * huge_w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 8192, 8192);
    auto * huge_x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8192, 2048);
    check(!ggml_metal_coreml_shape_supported(ggml_mul_mat(ctx, huge_w, huge_x)), "scratch budget must fall back");
    auto * small = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 1);
    check(!ggml_metal_coreml_shape_supported(ggml_mul_mat(ctx, w, small)), "decode must fall back");
    ggml_mul_mat_set_prec(y, GGML_PREC_F32);
    check(!ggml_metal_coreml_shape_supported(y), "explicit F32 precision must fall back");
    auto * selected = ggml_mul_mat(ctx, w, x);
    auto * skipped = ggml_mul_mat(ctx, w, x);
    ggml_tensor * choices[] = { selected, skipped };
    auto * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_select(graph, choices, 2, 0);
    check(ggml_metal_coreml_node_supported(selected), "selected matmul should qualify");
    check(ggml_metal_coreml_shape_supported(skipped), "skipped matmul has eligible shape");
    check(!ggml_metal_coreml_node_supported(skipped), "unselected matmul must not run");
    check(!ggml_metal_coreml_node_supported(nullptr), "null node must not run");
    std::puts("policy: PASS");
}

static double milliseconds(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

static void run(ggml_backend_t backend, int m, int n, int k, ggml_type type, int repeats, const std::string & scenario) {
    const bool bare = scenario == "bare";
    const bool chain = scenario == "chain";
    check(!chain || m == k, "chain requires M=K");
    check(scenario != "overflow" || type == GGML_TYPE_F32, "overflow requires f32");
    context_ptr owned_ctx(ggml_init({4*1024*1024, nullptr, true}), ggml_free);
    ggml_context * ctx = owned_ctx.get();
    check(ctx != nullptr, "context");
    auto * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, k, m);
    // Nonzero view offset exercises the Metal buffer API.
    auto * storage = ggml_new_tensor_2d(ctx, type, k, n + 1);
    auto * x = ggml_view_2d(ctx, storage, k, n, storage->nb[1], storage->nb[1]);
    auto * packed = bare ? x : ggml_cont(ctx, x);
    auto * y = ggml_mul_mat(ctx, w, packed);
    check(ggml_backend_supports_op(backend, y), "backend does not support this matmul type");
    ggml_set_name(y, "coreml-test-matmul");
    if (scenario == "precision") { ggml_mul_mat_set_prec(y, GGML_PREC_F32); }
    if (chain) {
        y = ggml_mul_mat(ctx, w, y);
        ggml_set_name(y, "coreml-test-matmul-chain");
    }
    auto * out = bare ? y : ggml_scale(ctx, y, 0.5f);
    auto * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, out);
    buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx, backend), ggml_backend_buffer_free);
    check(buffer != nullptr, "backend allocation");
    std::vector<float> weights(size_t(m)*k), activations(size_t(n)*k), reference(size_t(m)*n), result(reference.size());
    std::vector<unsigned char> quant(ggml_nbytes(w));
    std::vector<ggml_fp16_t> half(activations.size());
    double cold = 0, warm = 0, nmse = 0, max_error = 0;
    for (int mutation = 0; mutation < 2; ++mutation) {
        for (size_t i = 0; i < weights.size(); ++i) {
            // Different scales per 32-element block catch accidental rowwise scale reuse.
            uint32_t bits = uint32_t(i) + 0x9e3779b9u*uint32_t(mutation + 1);
            bits = (bits ^ (bits >> 16))*0x7feb352du;
            bits = (bits ^ (bits >> 15))*0x846ca68bu;
            bits ^= bits >> 16;
            weights[i] = (float(bits & 65535)/32768.0f - 1.0f) * (0.01f + float((i/32)%17)*0.015f);
        }
        check(ggml_quantize_chunk(GGML_TYPE_Q8_0, weights.data(), quant.data(), 0, m, k, nullptr) == quant.size(), "quantized size");
        ggml_backend_tensor_set(w, quant.data(), 0, quant.size());
        for (size_t i = 0; i < activations.size(); ++i) {
            activations[i] = std::cos(float(i)*0.017f + mutation) * 0.25f;
            if (scenario == "overflow") { activations[i] *= 1000000.0f; }
            if (scenario == "zero") { activations[i] = 0.0f; }
            if (type == GGML_TYPE_F16) {
                half[i] = ggml_fp32_to_fp16(activations[i]);
                activations[i] = ggml_fp16_to_fp32(half[i]);
            }
        }
        ggml_backend_tensor_set(x, type == GGML_TYPE_F16 ? (void *) half.data() : (void *) activations.data(), 0, ggml_nbytes(x));
        // Decode once, then accumulate independently of the backend in double precision.
        for (size_t i = 0; i < weights.size(); ++i) {
            ggml_fp16_t scale;
            std::memcpy(&scale, &quant[34*(i/32)], sizeof(scale));
            int8_t q;
            std::memcpy(&q, &quant[34*(i/32) + 2 + i%32], sizeof(q));
            weights[i] = ggml_fp16_to_fp32(scale)*q;
        }
        for (int row = 0; row < m; ++row) {
            for (int token = 0; token < n; ++token) {
                double sum = 0;
                for (int col = 0; col < k; ++col) {
                    sum += double(weights[size_t(row)*k + col])*activations[size_t(token)*k + col];
                }
                reference[size_t(token)*m + row] = float(sum);
            }
        }
        if (chain) {
            const auto first = reference;
            for (int token = 0; token < n; ++token) {
                for (int row = 0; row < m; ++row) {
                    double sum = 0;
                    for (int col = 0; col < k; ++col) { sum += double(weights[size_t(row)*k + col])*first[size_t(token)*k + col]; }
                    reference[size_t(token)*m + row] = float(sum);
                }
            }
        }
        if (!bare) { for (float & value : reference) { value *= 0.5f; } }
        for (int iteration = 0; iteration <= repeats; ++iteration) {
            const auto start = std::chrono::steady_clock::now();
            check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph compute");
            ggml_backend_synchronize(backend);
            const double elapsed = milliseconds(start);
            if (iteration == 0 && mutation == 0) { cold = elapsed; }
            if (iteration > 0) { warm += elapsed; }
            ggml_backend_tensor_get(out, result.data(), 0, ggml_nbytes(out));
            double error = 0, norm = 0;
            for (size_t i = 0; i < result.size(); ++i) {
                check(std::isfinite(result[i]), "nonfinite result");
                const double delta = double(result[i]) - reference[i];
                error += delta*delta;
                norm += double(reference[i])*reference[i];
                max_error = std::max(max_error, std::abs(delta));
            }
            nmse = std::max(nmse, error/std::max(norm, 1e-20));
            if (nmse >= 5e-4) { std::fprintf(stderr, "scenario=%s mutation=%d iteration=%d nmse=%.9g max_abs=%.9g\n", scenario.c_str(), mutation, iteration, nmse, max_error); }
            check(nmse < 5e-4, "Q8 matmul correctness NMSE");
        }
    }
    std::printf("{\"backend\":\"%s\",\"m\":%d,\"n\":%d,\"k\":%d,\"activation\":\"%s\",\"scenario\":\"%s\",\"cold_ms\":%.3f,\"warm_ms\":%.3f,\"nmse\":%.9g,\"max_abs\":%.6g}\n",
                ggml_backend_name(backend), m, n, k, ggml_type_name(type), scenario.c_str(), cold, warm/(2*repeats), nmse, max_error);
}

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && strcmp(argv[1], "--policy") == 0) { policy_tests(); return 0; }
        if (argc != 7 && argc != 8) {
            std::fprintf(stderr, "usage: %s CPU|MTL M N K f16|f32 REPEATS [mixed|bare|chain|overflow|zero|precision|cache|instances] (or --policy)\n", argv[0]);
            return 1;
        }
        const int m = parse_int(argv[2], 8192), n = parse_int(argv[3], 2048), k = parse_int(argv[4], 8192), repeats = parse_int(argv[6], 10000);
        check(k%32 == 0, "K must be divisible by 32");
        const std::string scenario = argc == 8 ? argv[7] : "mixed";
        check(scenario == "mixed" || scenario == "bare" || scenario == "chain" || scenario == "overflow" ||
              scenario == "zero" || scenario == "precision" || scenario == "cache" || scenario == "instances", "invalid scenario");
        check(strcmp(argv[5], "f16") == 0 || strcmp(argv[5], "f32") == 0, "activation type");
        ggml_backend_load_all();
        auto reg = ggml_backend_reg_by_name(argv[1]);
        check(reg != nullptr, "requested backend unavailable");
        check(ggml_backend_reg_dev_count(reg) > 0, "requested backend has no devices");
        backend_ptr backend(ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr), ggml_backend_free);
        check(backend != nullptr, "backend init");
        const ggml_type type = strcmp(argv[5], "f16") == 0 ? GGML_TYPE_F16 : GGML_TYPE_F32;
        if (scenario == "instances") {
            backend_ptr second(ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr), ggml_backend_free);
            check(second != nullptr, "second backend init");
            for (int round = 0; round < 2; ++round) {
                run(backend.get(), m, n, k, type, repeats, "mixed");
                run(second.get(), m, n, k, type, repeats, "mixed");
            }
        } else if (scenario == "cache") {
            check(n <= 1920, "cache sweep needs N <= 1920");
            for (int round = 0; round < 2; ++round) {
                for (int offset = 0; offset < 5; ++offset) { run(backend.get(), m, n + 32*offset, k, type, repeats, "mixed"); }
            }
        } else { run(backend.get(), m, n, k, type, repeats, scenario); }
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
