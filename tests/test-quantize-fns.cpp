// Unit tests for quantization specific functions - quantize, dequantize and dot product

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-quants.h"

#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

constexpr float MAX_QUANTIZATION_REFERENCE_ERROR = 0.0001f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR = 0.002f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_BINARY = 0.025f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_TERNARY = 0.01f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_2BITS = 0.0075f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS = 0.0040f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS = 0.0050f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_FP4 = 0.0030f;
constexpr float MAX_DOT_PRODUCT_ERROR = 0.02f;
constexpr float MAX_DOT_PRODUCT_ERROR_LOWBIT = 0.04f;
constexpr float MAX_DOT_PRODUCT_ERROR_FP4 = 0.03f;
constexpr float MAX_DOT_PRODUCT_ERROR_BINARY = 0.40f;
constexpr float MAX_DOT_PRODUCT_ERROR_TERNARY = 0.15f;
constexpr float MAX_QUANTIZATION_TOTAL_ERROR_TURBOQUANT = 0.005f;
constexpr float MAX_DOT_PRODUCT_ERROR_TURBOQUANT = 0.05f;

static const char* RESULT_STR[] = {"ok", "FAILED"};

static float max_quantization_error_for(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:    return MAX_QUANTIZATION_TOTAL_ERROR_BINARY;
        case GGML_TYPE_TQ1_0:
        case GGML_TYPE_TQ2_0:
        case GGML_TYPE_Q2_0:    return MAX_QUANTIZATION_TOTAL_ERROR_TERNARY;
        case GGML_TYPE_TBQ3_0:
        case GGML_TYPE_TBQ4_0:
        case GGML_TYPE_TBQ3_0_64:
        case GGML_TYPE_TBQ4_0_64:
        case GGML_TYPE_PQ3_0:
        case GGML_TYPE_PQ3_0_64:
        case GGML_TYPE_PQ4_0:
        case GGML_TYPE_PQ4_0_64: return MAX_QUANTIZATION_TOTAL_ERROR_TURBOQUANT;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_IQ2_S:   return MAX_QUANTIZATION_TOTAL_ERROR_2BITS;
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_IQ3_S:   return MAX_QUANTIZATION_TOTAL_ERROR_3BITS;
        case GGML_TYPE_IQ3_XXS: return MAX_QUANTIZATION_TOTAL_ERROR_3BITS_XXS;
        case GGML_TYPE_NVFP4:   return MAX_QUANTIZATION_TOTAL_ERROR_FP4;
        default:                return MAX_QUANTIZATION_TOTAL_ERROR;
    }
}

static float max_dot_product_error_for(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_S:   return MAX_DOT_PRODUCT_ERROR_LOWBIT;
        case GGML_TYPE_Q1_0:    return MAX_DOT_PRODUCT_ERROR_BINARY;
        case GGML_TYPE_TQ1_0:
        case GGML_TYPE_TQ2_0:
        case GGML_TYPE_Q2_0:    return MAX_DOT_PRODUCT_ERROR_TERNARY;
        case GGML_TYPE_TBQ3_0:
        case GGML_TYPE_TBQ4_0:
        case GGML_TYPE_TBQ3_0_64:
        case GGML_TYPE_TBQ4_0_64:
        case GGML_TYPE_PQ3_0:
        case GGML_TYPE_PQ3_0_64:
        case GGML_TYPE_PQ4_0:
        case GGML_TYPE_PQ4_0_64: return MAX_DOT_PRODUCT_ERROR_TURBOQUANT;
        case GGML_TYPE_NVFP4:   return MAX_DOT_PRODUCT_ERROR_FP4;
        default:                return MAX_DOT_PRODUCT_ERROR;
    }
}


// Generate synthetic data
static void generate_data(float offset, size_t n, float * dst) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = 0.1 + 2*cosf(i + offset);
    }
}

// Calculate RMSE between two float arrays
static float array_rmse(const float * a1, const float * a2, size_t n) {
    double sum = 0;
    for (size_t i = 0; i < n; i++) {
        double diff = a1[i] - a2[i];
        sum += diff * diff;
    }
    return sqrtf(sum) / n;
}

static float dot_product(const float * a1, const float * a2, size_t test_size) {
    double sum = 0;
    for (size_t i = 0; i < test_size; i++) {
        sum += a1[i] * a2[i];
    }
    return sum;
}

static float total_quantization_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data) {
    std::vector<uint8_t> tmp_q(2*test_size);
    std::vector<float> tmp_out(test_size);

    qfns_cpu->from_float(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out.data(), test_size);
    return array_rmse(test_data, tmp_out.data(), test_size);
}

static float reference_quantization_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data) {
    std::vector<uint8_t> tmp_q(2*test_size);
    std::vector<float> tmp_out(test_size);
    std::vector<float> tmp_out_ref(test_size);

    qfns_cpu->from_float(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out.data(), test_size);

    qfns->from_float_ref(test_data, tmp_q.data(), test_size);
    qfns->to_float(tmp_q.data(), tmp_out_ref.data(), test_size);

    return array_rmse(tmp_out.data(), tmp_out_ref.data(), test_size);
}

static float dot_product_error(const ggml_type_traits * qfns, const ggml_type_traits_cpu * qfns_cpu, size_t test_size, const float * test_data1, const float * test_data2) {
    GGML_UNUSED(qfns);

    std::vector<uint8_t> tmp_q1(2*test_size);
    std::vector<uint8_t> tmp_q2(2*test_size);

    const auto * vdot = ggml_get_type_traits_cpu(qfns_cpu->vec_dot_type);

    qfns_cpu->from_float(test_data1, tmp_q1.data(), test_size);
    vdot->from_float(test_data2, tmp_q2.data(), test_size);

    float result = INFINITY;
    qfns_cpu->vec_dot(test_size, &result, 0, tmp_q1.data(), 0, tmp_q2.data(), 0, 1);

    const float dot_ref = dot_product(test_data1, test_data2, test_size);

    return fabsf(result - dot_ref) / test_size;
}

// Backend path: builds ggml compute graphs and runs them through the scheduler.
struct backend_context {
    ggml_backend_t backend;
    ggml_backend_t cpu_backend;

    backend_context(ggml_backend_t b, ggml_backend_t cpu) : backend(b), cpu_backend(cpu) {}

    ~backend_context() {
        ggml_backend_free(backend);
        ggml_backend_free(cpu_backend);
    }
};

static constexpr size_t BACKEND_TEST_CTX_MEM_SIZE = 1 << 20; // 1 MiB

static bool backend_supports_cpy(ggml_backend_t backend, ggml_type qtype, int64_t test_size, bool verbose = false) {
    ggml_init_params params = { BACKEND_TEST_CTX_MEM_SIZE, nullptr, true };
    ggml_context *   ctx    = ggml_init(params);

    ggml_tensor * f32_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, test_size);
    ggml_tensor * q_buf  = ggml_new_tensor_1d(ctx, qtype, test_size);
    ggml_tensor * result = ggml_cpy(ctx, f32_in, q_buf);

    bool supported = ggml_backend_supports_op(backend, result);

    if (verbose) {
        fprintf(stderr, "[backend-debug] cpy op supported: type=%s test_size=%lld -> %s\n",
                ggml_type_name(qtype), (long long) test_size, supported ? "true" : "false");
    }

    ggml_free(ctx);
    return supported;
}

static float backend_quantization_error(backend_context & bctx,
                                        ggml_type         qtype,
                                        size_t            test_size,
                                        const float *     test_data) {
    const int64_t n = (int64_t) test_size;

    ggml_init_params params = { BACKEND_TEST_CTX_MEM_SIZE, nullptr, true };
    ggml_context *   ctx    = ggml_init(params);

    // f32 input -> quantized -> f32 output
    ggml_tensor * f32_src = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * q_tmp   = ggml_new_tensor_1d(ctx, qtype, n);
    ggml_tensor * f32_dst = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);

    ggml_tensor * cpy_to_q   = ggml_cpy(ctx, f32_src, q_tmp);
    ggml_tensor * cpy_to_f32 = ggml_cpy(ctx, cpy_to_q, f32_dst);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, cpy_to_f32);

    ggml_backend_t backends[2] = { bctx.backend, bctx.cpu_backend };
    ggml_backend_sched_t sched =
        ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    ggml_backend_sched_alloc_graph(sched, graph);

    ggml_backend_tensor_set(f32_src, test_data, 0, n * sizeof(float));

    ggml_backend_sched_graph_compute(sched, graph);

    std::vector<float> out(test_size);
    ggml_backend_tensor_get(f32_dst, out.data(), 0, n * sizeof(float));

    ggml_backend_sched_free(sched);
    ggml_free(ctx);

    return array_rmse(test_data, out.data(), test_size);
}

static float backend_dot_product_error(backend_context & bctx,
                                       ggml_type         qtype,
                                       size_t            test_size,
                                       const float *     test_data1,
                                       const float *     test_data2,
                                       bool              verbose = false) {
    const bool print_debug = verbose;
    const int64_t n = (int64_t) test_size;

    // mul_mat: A is [n, 1] quantized, B is [n, 1] f32 => result is [1, 1]
    // mul_mat computes A^T * B, so with A=[n,1] and B=[n,1] we get a [1,1] dot product
    ggml_init_params params = { BACKEND_TEST_CTX_MEM_SIZE, nullptr, true };
    ggml_context *   ctx    = ggml_init(params);

    ggml_tensor * f32_a  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * q_a    = ggml_new_tensor_1d(ctx, qtype, n);
    ggml_tensor * f32_b  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, 1);
    ggml_tensor * cpy_a  = ggml_cpy(ctx, f32_a, q_a);
    // reshape to [n, 1] for mul_mat
    ggml_tensor * q_a_2d = ggml_reshape_2d(ctx, cpy_a, n, 1);

    ggml_tensor * mm = ggml_mul_mat(ctx, q_a_2d, f32_b);

    if (print_debug) {
        fprintf(stderr,
                "[backend-debug] dot op details: type=%s n=%lld src0=%llux%llux%llux%llu src1=%llux%llux%llu%llu dst=%llux%llux%llux%llu\n",
                ggml_type_name(qtype),
                (long long) n,
                (unsigned long long) mm->src[0]->ne[0], (unsigned long long) mm->src[0]->ne[1],
                (unsigned long long) mm->src[0]->ne[2], (unsigned long long) mm->src[0]->ne[3],
                (unsigned long long) mm->src[1]->ne[0], (unsigned long long) mm->src[1]->ne[1],
                (unsigned long long) mm->src[1]->ne[2], (unsigned long long) mm->src[1]->ne[3],
                (unsigned long long) mm->ne[0], (unsigned long long) mm->ne[1],
                (unsigned long long) mm->ne[2], (unsigned long long) mm->ne[3]);
    }

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, mm);

    ggml_backend_t backends[2] = { bctx.backend, bctx.cpu_backend };
    const bool supports_mul = ggml_backend_supports_op(bctx.backend, mm);
    if (print_debug) {
        fprintf(stderr, "[backend-debug] ggml_backend_supports_op(mm)=%s for type=%s\n", supports_mul ? "true" : "false", ggml_type_name(qtype));
    }

    if (!supports_mul) {
        ggml_free(ctx);
        return -1.0f;
    }

    ggml_backend_sched_t sched =
        ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);

    if (!ggml_backend_sched_alloc_graph(sched, graph)) {
        if (print_debug) {
            fprintf(stderr, "[backend-debug] ggml_backend_sched_alloc_graph(mm) failed for type=%s\n", ggml_type_name(qtype));
        }
        ggml_backend_sched_free(sched);
        ggml_free(ctx);
        return -1.0f;
    }

    ggml_backend_tensor_set(f32_a, test_data1, 0, n * sizeof(float));
    ggml_backend_tensor_set(f32_b, test_data2, 0, n * sizeof(float));

    ggml_backend_sched_graph_compute(sched, graph);

    float gpu_dot = 0.0f;
    ggml_backend_tensor_get(mm, &gpu_dot, 0, sizeof(float));

    if (print_debug) {
        fprintf(stderr, "[backend-debug] mm result from backend=%f\n", gpu_dot);
    }

    ggml_backend_sched_free(sched);
    ggml_free(ctx);

    const float dot_ref = dot_product(test_data1, test_data2, test_size);
    const float err = fabsf(gpu_dot - dot_ref) / test_size;
    if (print_debug) {
        fprintf(stderr, "[backend-debug] dot ref=%f err=%f\n", dot_ref, err);
    }
    return err;
}

struct test_results {
    float quant_errors[GGML_TYPE_COUNT] = {};
    float dot_errors[GGML_TYPE_COUNT]   = {};
    bool  tested[GGML_TYPE_COUNT]       = {};
    int   num_failed                    = 0;
};

// Run all quantization tests via the backend compute graph API
static int run_backend_tests(const char *   backend_name,
                             size_t         test_size,
                             bool           verbose,
                             const float *  test_data,
                             const float *  test_data2,
                             test_results & res) {
    printf("=== Backend mode: %s ===\n\n", backend_name);

    ggml_backend_load_all();

    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d    = ggml_backend_dev_get(i);
        const char *       name = ggml_backend_dev_name(d);
        std::string        dev_name_lower(name);
        std::string        filter_lower(backend_name);
        for (auto & c : dev_name_lower) {
            c = tolower(c);
        }
        for (auto & c : filter_lower) {
            c = tolower(c);
        }
        if (dev_name_lower.find(filter_lower) != std::string::npos) {
            dev = d;
            break;
        }
    }

    if (!dev) {
        fprintf(stderr, "Backend '%s' not found. Available backends:\n", backend_name);
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            fprintf(stderr, "  %s (%s)\n", ggml_backend_dev_name(ggml_backend_dev_get(i)),
                    ggml_backend_dev_description(ggml_backend_dev_get(i)));
        }
        return 1;
    }

    printf("Using device: %s (%s)\n\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    assert(backend);

    ggml_backend_t cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    assert(cpu_backend);

    backend_context bctx(backend, cpu_backend);

    bool failed = false;

    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        ggml_type    type = (ggml_type) i;
        const auto * qfns = ggml_get_type_traits(type);

        if (qfns->blck_size == 0 || !ggml_is_quantized(type)) {
            continue;
        }

        if ((int64_t) test_size % qfns->blck_size != 0) {
            continue;
        }

        printf("Testing %s (backend)\n", ggml_type_name(type));

        if (!backend_supports_cpy(bctx.backend, type, (int64_t) test_size, verbose)) {
            printf("  %s: cpy not supported on this backend, skipping\n", ggml_type_name(type));
            continue;
        }

        res.tested[i] = true;

        const float total_error   = backend_quantization_error(bctx, type, test_size, test_data);
        const float max_quant_err = max_quantization_error_for(type);
        failed                    = !(total_error < max_quant_err);
        res.num_failed += failed;
        if (failed || verbose) {
            printf("%5s absolute quantization error:    %s (%f)\n", ggml_type_name(type), RESULT_STR[failed],
                   total_error);
        }
        res.quant_errors[i] = total_error;

        const float vec_dot_error = backend_dot_product_error(bctx, type, test_size, test_data, test_data2, verbose);
        if (vec_dot_error < 0.0f) {
            res.dot_errors[i] = vec_dot_error;
            if (verbose) {
                printf("%5s dot product: mul_mat not supported, skipping\n", ggml_type_name(type));
            }
        } else {
            const float max_allowed_error = max_dot_product_error_for(type);
            failed                        = !(vec_dot_error < max_allowed_error);
            res.num_failed += failed;
            if (failed || verbose) {
                printf("%5s dot product error:              %s (%f)\n", ggml_type_name(type), RESULT_STR[failed],
                       vec_dot_error);
            }
            res.dot_errors[i] = vec_dot_error;
        }
    }

    return 0;
}

static void run_cpu_tests(size_t         test_size,
                          bool           verbose,
                          const float *  test_data,
                          const float *  test_data2,
                          test_results & res) {
    printf("=== CPU mode ===\n\n");

    ggml_cpu_init();

    bool failed = false;

    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        ggml_type    type     = (ggml_type) i;
        const auto * qfns = ggml_get_type_traits(type);
        const auto * qfns_cpu = ggml_get_type_traits_cpu(type);

        if (qfns->blck_size == 0) {
            continue;
        }

        printf("Testing %s\n", ggml_type_name(type));
        ggml_quantize_init(type);

        if (qfns_cpu->from_float && qfns->to_float) {
            res.tested[i] = true;

            const float total_error = total_quantization_error(qfns, qfns_cpu, test_size, test_data);
            const float max_quantization_error = max_quantization_error_for(type);
            failed = !(total_error < max_quantization_error);
            res.num_failed += failed;
            if (failed || verbose) {
                printf("%5s absolute quantization error:    %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], total_error);
            }
            res.quant_errors[i] = total_error;

            const float reference_error = reference_quantization_error(qfns, qfns_cpu, test_size, test_data);
            failed = !(reference_error < MAX_QUANTIZATION_REFERENCE_ERROR);
            res.num_failed += failed;
            if (failed || verbose) {
                printf("%5s reference implementation error: %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], reference_error);
            }

            const float vec_dot_error = dot_product_error(qfns, qfns_cpu, test_size, test_data, test_data2);
            const float max_allowed_error = max_dot_product_error_for(type);
            failed = !(vec_dot_error < max_allowed_error);
            res.num_failed += failed;
            if (failed || verbose) {
                printf("%5s dot product error:              %s (%f)\n", ggml_type_name(type), RESULT_STR[failed], vec_dot_error);
            }
            res.dot_errors[i] = vec_dot_error;
        }
    }
}

static void run_cross_type_checks(bool verbose, test_results & res) {
    printf("\nCross-type checks\n");

    bool failed = false;

    auto check_lower = [&](ggml_type better, ggml_type worse, const char * metric, const float * errors) {
        if (!res.tested[better] || !res.tested[worse]) {
            return;
        }
        if (errors[better] < 0.0f || errors[worse] < 0.0f) {
            return;
        }

        failed = !(errors[better] < errors[worse]);
        res.num_failed += failed;
        if (failed || verbose) {
            printf("%s %s should be lower than %s: %s (%f vs %f)\n", ggml_type_name(better), metric,
                   ggml_type_name(worse), RESULT_STR[failed], errors[better], errors[worse]);
        }
    };

    // Quant error: deterministic — higher bitwidth always wins.
    // TBQ and PQ share the same Stage 1 codebook, so quant errors are identical
    // within a bitwidth. We use PQ for cross-bitwidth quant checks and TBQ for
    // "better than ternary" checks.

    // block=128
    check_lower(GGML_TYPE_PQ4_0, GGML_TYPE_PQ3_0, "quant error", res.quant_errors);
    check_lower(GGML_TYPE_PQ4_0, GGML_TYPE_TQ1_0, "quant error", res.quant_errors);
    check_lower(GGML_TYPE_PQ4_0, GGML_TYPE_TQ2_0, "quant error", res.quant_errors);
    check_lower(GGML_TYPE_PQ3_0, GGML_TYPE_TQ1_0, "quant error", res.quant_errors);
    check_lower(GGML_TYPE_PQ3_0, GGML_TYPE_TQ2_0, "quant error", res.quant_errors);
    check_lower(GGML_TYPE_TBQ4_0, GGML_TYPE_TBQ3_0, "quant error", res.quant_errors);
    check_lower(GGML_TYPE_TBQ3_0, GGML_TYPE_TQ1_0, "quant error", res.quant_errors);
    check_lower(GGML_TYPE_TBQ3_0, GGML_TYPE_TQ2_0, "quant error", res.quant_errors);
    check_lower(GGML_TYPE_TBQ4_0, GGML_TYPE_TQ1_0, "quant error", res.quant_errors);
    check_lower(GGML_TYPE_TBQ4_0, GGML_TYPE_TQ2_0, "quant error", res.quant_errors);

    // Dot error for PQ (no QJL): deterministic, higher bitwidth always wins.
    check_lower(GGML_TYPE_PQ4_0, GGML_TYPE_PQ3_0, "dot error", res.dot_errors);

    // TBQ vs PQ dot error is NOT checked here. The QJL correction improves
    // dot products on real data (verified via perplexity in test-kv-cache-quantization.sh),
    // but this test can't reliably validate it: on CPU the synthetic dataset is
    // too small for the 1-bit sketch variance to average out, and on GPU the
    // cpy_f32_quant shader doesn't compute QJL (fields are zeroed).

    // block=64
    check_lower(GGML_TYPE_PQ4_0_64, GGML_TYPE_PQ3_0_64, "quant error", res.quant_errors);
    check_lower(GGML_TYPE_PQ4_0_64, GGML_TYPE_TQ1_0, "quant error", res.quant_errors);
    check_lower(GGML_TYPE_PQ4_0_64, GGML_TYPE_TQ2_0, "quant error", res.quant_errors);
    check_lower(GGML_TYPE_PQ3_0_64, GGML_TYPE_TQ1_0, "quant error", res.quant_errors);
    check_lower(GGML_TYPE_PQ3_0_64, GGML_TYPE_TQ2_0, "quant error", res.quant_errors);
    check_lower(GGML_TYPE_TBQ4_0_64, GGML_TYPE_TBQ3_0_64, "quant error", res.quant_errors);
    check_lower(GGML_TYPE_TBQ3_0_64, GGML_TYPE_TQ1_0, "quant error", res.quant_errors);
    check_lower(GGML_TYPE_TBQ3_0_64, GGML_TYPE_TQ2_0, "quant error", res.quant_errors);
    check_lower(GGML_TYPE_TBQ4_0_64, GGML_TYPE_TQ1_0, "quant error", res.quant_errors);
    check_lower(GGML_TYPE_TBQ4_0_64, GGML_TYPE_TQ2_0, "quant error", res.quant_errors);

    check_lower(GGML_TYPE_PQ4_0_64, GGML_TYPE_PQ3_0_64, "dot error", res.dot_errors);
}

// ===================== TurboQuant-specific unit tests =====================

static void test_tq_forward_inverse_roundtrip(void) {
    printf("\nTurboQuant forward/inverse roundtrip test\n");

    // Hardcoded sign tables matching tq_utils.comp / ggml-quants.c (seed 42)
    static const uint32_t TQ_SIGN_BITS[4] = { 0x40f54e8cu, 0x6587b7b0u, 0xc31220eau, 0x32f6449bu };

    for (int d : { 64, 128 }) {
        std::vector<float> signs(d);
        for (int i = 0; i < d; i++) {
            signs[i] = ((TQ_SIGN_BITS[i / 32] >> (i % 32)) & 1u) ? 1.0f : -1.0f;
        }

        std::vector<float> original(d);
        std::vector<float> buf(d);
        for (int i = 0; i < d; i++) {
            original[i] = (float)(i + 1) / d;
        }

        memcpy(buf.data(), original.data(), d * sizeof(float));
        tq_forward_inplace(buf.data(), d, signs.data());
        tq_inverse_inplace(buf.data(), d, signs.data());

        double max_err = 0.0;
        for (int i = 0; i < d; i++) {
            double err = fabs((double)buf[i] - (double)original[i]);
            if (err > max_err) { max_err = err; }
        }

        bool ok = max_err < 1e-5;
        printf("  d=%3d: max_err=%.2e %s\n", d, max_err, ok ? "OK" : "FAILED");
        assert(ok);
    }
}

static void test_tq_quantize_val_boundaries(void) {
    printf("\nTurboQuant quantize_val boundary tests\n");

    bool failed = false;

    for (int d : { 64, 128 }) {
        const float * cb3 = tq3_codebook_for(d);
        const float * cb4 = tq4_codebook_for(d);

        // TQ3: each centroid value should quantize to itself
        {
            float b[7];
            tq_compute_boundaries(cb3, b, 8);
            printf("  TQ3 d=%d centroid self-mapping: ", d);
            bool ok = true;
            for (int i = 0; i < 8; i++) {
                uint8_t idx = tq3_quantize_val(cb3[i], b);
                if (idx != i) {
                    printf("FAILED (centroid %d -> bucket %d)\n", i, idx);
                    ok = false;
                    failed = true;
                    break;
                }
            }
            if (ok) { printf("OK\n"); }
        }

        // TQ3: values just inside each boundary should map to correct neighbor
        {
            float b[7];
            tq_compute_boundaries(cb3, b, 8);
            printf("  TQ3 d=%d boundary neighbors:   ", d);
            bool ok = true;
            for (int i = 0; i < 7; i++) {
                float eps = 1e-7f;
                uint8_t lo = tq3_quantize_val(b[i] - eps, b);
                uint8_t hi = tq3_quantize_val(b[i] + eps, b);
                if (lo != (uint8_t)i || hi != (uint8_t)(i + 1)) {
                    printf("FAILED at boundary %d (lo=%d expected %d, hi=%d expected %d)\n", i, lo, i, hi, i+1);
                    ok = false;
                    failed = true;
                    break;
                }
            }
            if (ok) { printf("OK\n"); }
        }

        // TQ4: each centroid value should quantize to itself
        {
            float b[15];
            tq_compute_boundaries(cb4, b, 16);
            printf("  TQ4 d=%d centroid self-mapping: ", d);
            bool ok = true;
            for (int i = 0; i < 16; i++) {
                uint8_t idx = tq4_quantize_val(cb4[i], b);
                if (idx != i) {
                    printf("FAILED (centroid %d -> bucket %d)\n", i, idx);
                    ok = false;
                    failed = true;
                    break;
                }
            }
            if (ok) { printf("OK\n"); }
        }

        // TQ4: boundary neighbors
        {
            float b[15];
            tq_compute_boundaries(cb4, b, 16);
            printf("  TQ4 d=%d boundary neighbors:   ", d);
            bool ok = true;
            for (int i = 0; i < 15; i++) {
                float eps = 1e-7f;
                uint8_t lo = tq4_quantize_val(b[i] - eps, b);
                uint8_t hi = tq4_quantize_val(b[i] + eps, b);
                if (lo != (uint8_t)i || hi != (uint8_t)(i + 1)) {
                    printf("FAILED at boundary %d (lo=%d expected %d, hi=%d expected %d)\n", i, lo, i, hi, i+1);
                    ok = false;
                    failed = true;
                    break;
                }
            }
            if (ok) { printf("OK\n"); }
        }
    }

    assert(!failed);
}

int main(int argc, char * argv[]) {
    bool verbose = false;
    size_t test_size = 32 * 128;
    const char * backend_env = getenv("GGML_TEST_BACKEND");
    bool use_backend = (backend_env != nullptr && strlen(backend_env) > 0 && strcmp(backend_env, "cpu") != 0);

    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        if (arg == "-v") {
            verbose = true;
        } else if (arg == "-b" && i + 1 < argc) {
            backend_env = argv[++i];
            use_backend = true;
        } else if (arg == "-s" && i + 1 < argc) {
            test_size = std::stoul(argv[++i]);
            if (test_size % 128 != 0) {
                fprintf(stderr, "error: test size must be a multiple of 128\n");
                return 1;
            }
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            fprintf(stderr, "usage: %s [-v] [-b backend_name] [-s test_size]\n", argv[0]);
            fprintf(stderr, "  -s  number of floats to test (must be multiple of 128, default: 4096)\n");
            fprintf(stderr, "  or set GGML_TEST_BACKEND=vulkan (or cuda, etc.)\n");
            return 1;
        }
    }

    std::vector<float> test_data(test_size);
    std::vector<float> test_data2(test_size);

    generate_data(0.0, test_data.size(), test_data.data());
    generate_data(1.0, test_data2.size(), test_data2.data());

    test_results res;

    if (use_backend) {
        int err = run_backend_tests(backend_env, test_size, verbose, test_data.data(), test_data2.data(), res);
        if (err) return err;
    } else {
        run_cpu_tests(test_size, verbose, test_data.data(), test_data2.data(), res);
    }

    run_cross_type_checks(verbose, res);

    test_tq_forward_inverse_roundtrip();
    test_tq_quantize_val_boundaries();

    if (res.num_failed || verbose) {
        printf("%d tests failed\n", res.num_failed);
    }

    return res.num_failed > 0;
}
