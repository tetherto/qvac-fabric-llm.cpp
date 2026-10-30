// Benchmark quantization specific functions on synthetic data

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#undef NDEBUG
#include <algorithm>
#include <assert.h>
#include <cctype>
#include <functional>
#include <map>
#include <math.h>
#include <memory>
#include <stdio.h>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#define MAX_ALIGNMENT 64
#define QK 32
#define WARMUP 5
#define ITERATIONS 10
#define MAX_ITERATIONS 100000000

#define L1_SIZE      32*128
#define L2_SIZE     32*2048
#define L3_SIZE    32*20480
#define MEM_SIZE 32*2048000

struct quantize_perf_params {
    std::vector<std::string> include_types;
    std::vector<size_t> test_sizes;
    size_t alignment_offset = 0;
    bool op_quantize_row_q_reference = false;
    bool op_quantize_row_q = false;
    bool op_dequantize_row_q = false;
    bool op_quantize_row_q_dot = false;
    bool op_vec_dot_q = false;
    int64_t iterations = ITERATIONS;
    std::string backend_name;
};

#if defined(__x86_64__) || defined(__i386__)

#include <x86intrin.h>
inline int64_t cpu_cycles() {
// Rough way to detect new-ish CPUs
#ifdef __POPCNT__
    unsigned int dummy;
    return __rdtscp(&dummy);
#else
    return __rdtsc();
#endif
}

#else

#define cpu_cycles() 0

#endif


// Generate synthetic data
static void generate_data(float offset, size_t n, float * dst) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = 0.1 + 2*cosf(i + offset);
    }
}

static float gigabytes_per_second(size_t bytes, int64_t usecs) {
    return bytes / (float) usecs * 1000000 / (1024*1024*1024);
}

static void * align_with_offset(void * ptr, int offset) {
    size_t dummy_size = MAX_ALIGNMENT * 4;
    return (char *) std::align(MAX_ALIGNMENT, MAX_ALIGNMENT, ptr, dummy_size) + offset;
}

struct bench_result {
    double avg_time_us;
    double min_time_us;
};

static bench_result benchmark_function(size_t size, size_t q_size, int64_t iterations, const std::function<float(void)> & func) {
    int64_t min_time_us = INT64_MAX;
    int64_t total_time_us = 0;
    int64_t min_time_cycles = INT64_MAX;
    int64_t total_time_cycles = 0;

    for (int i = 0; i < WARMUP; i++) {
        func();
    }

    for (int i = 0; i < iterations; i++) {
        const int64_t start_time = ggml_time_us();
        const int64_t start_cycles = cpu_cycles();

        func();

        const int64_t end_cycles = cpu_cycles();
        const int64_t end_time = ggml_time_us();

        total_time_cycles += end_cycles - start_cycles;
        min_time_cycles = std::min(min_time_cycles, end_cycles - start_cycles);
        total_time_us += end_time - start_time;
        min_time_us = std::min(min_time_us, end_time - start_time);
    }

    printf("      min cycles/%d vals   : %9.2f\n",  QK, QK * min_time_cycles / (float) size);
    printf("      avg cycles/%d vals   : %9.2f\n",  QK, QK * total_time_cycles / (float) (size * iterations));
    printf("      float32 throughput   : %9.2f GB/s\n",  gigabytes_per_second(4 * size * iterations, total_time_us));
    printf("      quantized throughput : %9.2f GB/s\n",  gigabytes_per_second(q_size * iterations, total_time_us));

    return { (double) total_time_us / iterations, (double) min_time_us };
}

static void usage(char * argv[]) {
    printf("Benchmark quantization specific functions on synthetic data\n");
    printf("\n");
    printf("usage: %s [options]\n", argv[0]);
    printf("\n");
    printf("options: (default)\n");
    printf("  -h, --help            show this help message and exit\n");
    printf("  --size SIZE           set test size, divisible by 32 (L1_SIZE:%d)\n", L1_SIZE);
    printf("  -3                    use size as L1, L2, L3 sizes (L1:%d L2:%d L3:%d)\n", L1_SIZE, L2_SIZE, L3_SIZE);
    printf("  -4                    use size as L1, L2, L3, MEM sizes (L1:%d L2:%d L3:%d MEM:%d)\n", L1_SIZE, L2_SIZE, L3_SIZE, MEM_SIZE);
    printf("  --op OP               set test operation as quantize_row_q_reference, quantize_row_q, dequantize_row_q,\n");
    printf("                        quantize_row_q_dot, vec_dot_q (all)\n");
    printf("  --type TYPE           set test type as");
    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        ggml_type type = (ggml_type) i;
        const auto * qfns     = ggml_get_type_traits(type);
        const auto * qfns_cpu = ggml_get_type_traits_cpu(type);
        if (ggml_type_name(type) != NULL) {
            if (qfns_cpu->from_float && qfns->to_float) {
                printf(" %s", ggml_type_name(type));
            }
        }
    }
    printf(" (all)\n");
    printf("  --alignment-offset OFFSET\n");
    printf("                        set alignment offset as OFFSET (0)\n");
    printf("  -b BACKEND            run benchmarks on a backend (e.g. vulkan, cuda) instead of CPU\n");
    printf("  -i NUM, --iterations NUM\n");
    printf("                        set test iteration number (%d)\n", ITERATIONS);
}

// Backend path: build ggml compute graphs and time them through the scheduler.
static constexpr size_t BACKEND_PERF_CTX_MEM_SIZE = 1 << 20;

struct backend_perf_context {
    ggml_backend_t backend;
    ggml_backend_t cpu_backend;

    backend_perf_context(ggml_backend_t b, ggml_backend_t cpu) : backend(b), cpu_backend(cpu) {}

    ~backend_perf_context() {
        ggml_backend_free(backend);
        ggml_backend_free(cpu_backend);
    }
};

static backend_perf_context * init_backend(const std::string & backend_name) {
    ggml_backend_load_all();

    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        std::string dev_name(ggml_backend_dev_name(d));
        std::string filter(backend_name);
        for (auto & c : dev_name)  { c = tolower(c); }
        for (auto & c : filter)    { c = tolower(c); }
        if (dev_name.find(filter) != std::string::npos) {
            dev = d;
            break;
        }
    }

    if (!dev) {
        fprintf(stderr, "Backend '%s' not found. Available backends:\n", backend_name.c_str());
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            fprintf(stderr, "  %s (%s)\n", ggml_backend_dev_name(ggml_backend_dev_get(i)),
                    ggml_backend_dev_description(ggml_backend_dev_get(i)));
        }
        return nullptr;
    }

    printf("Using device: %s (%s)\n\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));

    ggml_backend_t backend     = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    assert(backend && cpu_backend);

    return new backend_perf_context(backend, cpu_backend);
}

static bench_result benchmark_backend_quantize(backend_perf_context & bctx, ggml_type type,
                                       size_t size, int64_t iterations, const float * src_data) {
    const int64_t n = (int64_t) size;

    ggml_init_params params = { BACKEND_PERF_CTX_MEM_SIZE, nullptr, true };
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * f32_src = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * q_dst   = ggml_new_tensor_1d(ctx, type, n);
    ggml_tensor * cpy     = ggml_cpy(ctx, f32_src, q_dst);

    if (!ggml_backend_supports_op(bctx.backend, cpy)) {
        printf("      (cpy f32->%s not supported, skipping)\n", ggml_type_name(type));
        ggml_free(ctx);
        return { -1, -1 };
    }

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, cpy);

    ggml_backend_t backends[] = { bctx.backend, bctx.cpu_backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    ggml_backend_sched_alloc_graph(sched, graph);
    ggml_backend_tensor_set(f32_src, src_data, 0, n * sizeof(float));

    for (int i = 0; i < WARMUP; i++) {
        ggml_backend_sched_graph_compute(sched, graph);
    }

    int64_t total_time_us = 0;
    int64_t min_time_us = INT64_MAX;
    for (int64_t i = 0; i < iterations; i++) {
        const int64_t t0 = ggml_time_us();
        ggml_backend_sched_graph_compute(sched, graph);
        const int64_t dt = ggml_time_us() - t0;
        total_time_us += dt;
        min_time_us = std::min(min_time_us, dt);
    }

    size_t quantized_size = ggml_row_size(type, size);
    printf("      min time             : %9.2f us\n", (float) min_time_us);
    printf("      avg time             : %9.2f us\n", (float) total_time_us / iterations);
    printf("      float32 throughput   : %9.2f GB/s\n", gigabytes_per_second(4 * size * iterations, total_time_us));
    printf("      quantized throughput : %9.2f GB/s\n", gigabytes_per_second(quantized_size * iterations, total_time_us));

    bench_result res = { (double) total_time_us / iterations, (double) min_time_us };
    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    return res;
}

static bench_result benchmark_backend_dequantize(backend_perf_context & bctx, ggml_type type,
                                         size_t size, int64_t iterations, const float * src_data) {
    const int64_t n = (int64_t) size;

    ggml_init_params params = { BACKEND_PERF_CTX_MEM_SIZE, nullptr, true };
    ggml_context * ctx = ggml_init(params);

    // f32 -> quant -> f32: single graph so the quantized buffer stays alive
    ggml_tensor * f32_src    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * q_tmp      = ggml_new_tensor_1d(ctx, type, n);
    ggml_tensor * cpy_to_q   = ggml_cpy(ctx, f32_src, q_tmp);
    ggml_tensor * f32_dst    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * cpy_to_f32 = ggml_cpy(ctx, cpy_to_q, f32_dst);

    if (!ggml_backend_supports_op(bctx.backend, cpy_to_q) ||
        !ggml_backend_supports_op(bctx.backend, cpy_to_f32)) {
        printf("      (cpy for %s not supported, skipping)\n", ggml_type_name(type));
        ggml_free(ctx);
        return { -1, -1 };
    }

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, cpy_to_f32);

    ggml_backend_t backends[] = { bctx.backend, bctx.cpu_backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    ggml_backend_sched_alloc_graph(sched, graph);
    ggml_backend_tensor_set(f32_src, src_data, 0, n * sizeof(float));

    for (int i = 0; i < WARMUP; i++) {
        ggml_backend_sched_graph_compute(sched, graph);
    }

    int64_t total_time_us = 0;
    int64_t min_time_us = INT64_MAX;
    for (int64_t i = 0; i < iterations; i++) {
        const int64_t t0 = ggml_time_us();
        ggml_backend_sched_graph_compute(sched, graph);
        const int64_t dt = ggml_time_us() - t0;
        total_time_us += dt;
        min_time_us = std::min(min_time_us, dt);
    }

    size_t quantized_size = ggml_row_size(type, size);
    printf("      min time             : %9.2f us\n", (float) min_time_us);
    printf("      avg time             : %9.2f us\n", (float) total_time_us / iterations);
    printf("      float32 throughput   : %9.2f GB/s\n", gigabytes_per_second(4 * size * iterations, total_time_us));
    printf("      quantized throughput : %9.2f GB/s\n", gigabytes_per_second(quantized_size * iterations, total_time_us));

    bench_result res = { (double) total_time_us / iterations, (double) min_time_us };
    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    return res;
}

static bench_result benchmark_backend_mul_mat(backend_perf_context & bctx, ggml_type type,
                                      size_t size, int64_t iterations,
                                      const float * src_data1, const float * src_data2) {
    const int64_t n = (int64_t) size;

    ggml_init_params params = { BACKEND_PERF_CTX_MEM_SIZE, nullptr, true };
    ggml_context * ctx = ggml_init(params);

    // quantize src1 via cpy, keep src2 as f32
    ggml_tensor * f32_a  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * q_a    = ggml_new_tensor_1d(ctx, type, n);
    ggml_tensor * cpy_a  = ggml_cpy(ctx, f32_a, q_a);
    ggml_tensor * q_a_2d = ggml_reshape_2d(ctx, cpy_a, n, 1);
    ggml_tensor * f32_b  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, 1);

    ggml_tensor * mm = ggml_mul_mat(ctx, q_a_2d, f32_b);

    if (!ggml_backend_supports_op(bctx.backend, mm)) {
        printf("      (mul_mat for %s not supported, skipping)\n", ggml_type_name(type));
        ggml_free(ctx);
        return { -1, -1 };
    }

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, mm);

    ggml_backend_t backends[] = { bctx.backend, bctx.cpu_backend };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
    ggml_backend_sched_alloc_graph(sched, graph);
    ggml_backend_tensor_set(f32_a, src_data1, 0, n * sizeof(float));
    ggml_backend_tensor_set(f32_b, src_data2, 0, n * sizeof(float));

    for (int i = 0; i < WARMUP; i++) {
        ggml_backend_sched_graph_compute(sched, graph);
    }

    int64_t total_time_us = 0;
    int64_t min_time_us = INT64_MAX;
    for (int64_t i = 0; i < iterations; i++) {
        const int64_t t0 = ggml_time_us();
        ggml_backend_sched_graph_compute(sched, graph);
        const int64_t dt = ggml_time_us() - t0;
        total_time_us += dt;
        min_time_us = std::min(min_time_us, dt);
    }

    size_t quantized_size = ggml_row_size(type, size);
    printf("      min time             : %9.2f us\n", (float) min_time_us);
    printf("      avg time             : %9.2f us\n", (float) total_time_us / iterations);
    printf("      float32 throughput   : %9.2f GB/s\n", gigabytes_per_second(4 * size * iterations, total_time_us));
    printf("      quantized throughput : %9.2f GB/s\n", gigabytes_per_second(quantized_size * iterations, total_time_us));

    bench_result res = { (double) total_time_us / iterations, (double) min_time_us };
    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    return res;
}

int main(int argc, char * argv[]) {
    quantize_perf_params params {};

    // read command line

    bool invalid_param = false;
    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];

        if (arg == "--size") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            size_t size = std::stoi(argv[i]);
            if (size % 32 != 0) {
                fprintf(stderr, "error: size %zu not divisible by 32\n", size);
                invalid_param = true;
                break;
            }
            params.test_sizes.push_back(size);
        } else if (arg == "-3") {
            // quick select sizes that probably fit in CPU caches
            params.test_sizes.push_back(L1_SIZE);
            params.test_sizes.push_back(L2_SIZE);
            params.test_sizes.push_back(L3_SIZE);
        } else if (arg == "-4") {
            // quick select cache sizes + memory
            params.test_sizes.push_back(L1_SIZE);
            params.test_sizes.push_back(L2_SIZE);
            params.test_sizes.push_back(L3_SIZE);
            params.test_sizes.push_back(MEM_SIZE);
        } else if (arg == "--op") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            std::string op {argv[i]};
            if (op == "quantize_row_q_reference") {
                params.op_quantize_row_q_reference = true;
            } else if (op == "quantize_row_q") {
                params.op_quantize_row_q = true;
            } else if (op == "dequantize_row_q") {
                params.op_dequantize_row_q = true;
            } else if (op == "quantize_row_q_dot") {
                params.op_quantize_row_q_dot = true;
            } else if (op == "vec_dot_q") {
                params.op_vec_dot_q = true;
            } else {
                invalid_param = true;
                break;
            }
        } else if (arg == "--type") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.include_types.push_back(argv[i]);
        } else if (arg == "--alignment-offset") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            int alignment = std::stoi(argv[i]);
            if (alignment < 0 || alignment > MAX_ALIGNMENT) {
            fprintf(stderr, "error: alignment-offset must be less than %d\n", MAX_ALIGNMENT);
                invalid_param = true;
                break;
            }
            params.alignment_offset = alignment;
        } else if (arg == "-b") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.backend_name = argv[i];
        } else if ((arg == "-i") || (arg == "--iterations")) {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            int number = std::stoi(argv[i]);
            if (number < 0 || number > MAX_ITERATIONS) {
            fprintf(stderr, "error: iterations must be less than %d\n", MAX_ITERATIONS);
                invalid_param = true;
                break;
            }
            params.iterations = number;
        } else if ((arg == "-h") || (arg == "--help")) {
            usage(argv);
            return 1;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }
    if (invalid_param) {
        fprintf(stderr, "error: invalid parameter for argument: %s\n", arg.c_str());
        return 1;
    }

    if (params.test_sizes.empty()) {
        params.test_sizes.push_back(L1_SIZE);
    }
    if (!(params.op_quantize_row_q_reference || params.op_quantize_row_q || params.op_dequantize_row_q || params.op_quantize_row_q_dot || params.op_vec_dot_q)) {
        params.op_quantize_row_q_reference = params.op_quantize_row_q = params.op_dequantize_row_q = params.op_quantize_row_q_dot = params.op_vec_dot_q = true;
    }

    std::sort(params.test_sizes.begin(), params.test_sizes.end());
    size_t largest = params.test_sizes.back();

    std::vector<uint8_t> test_data1_v(largest*4 + MAX_ALIGNMENT*2);
    std::vector<uint8_t> test_data2_v(largest*4 + MAX_ALIGNMENT*2);
    std::vector<uint8_t> test_q1_v   (largest*4 + MAX_ALIGNMENT*2);
    std::vector<uint8_t> test_q2_v   (largest*4 + MAX_ALIGNMENT*2);
    std::vector<uint8_t> test_out_v  (largest*4 + MAX_ALIGNMENT*2);

    float * test_data1 = (float *) align_with_offset(test_data1_v.data(), params.alignment_offset);
    float * test_data2 = (float *) align_with_offset(test_data2_v.data(), params.alignment_offset);
    float * test_q1    = (float *) align_with_offset(test_q1_v.data(),    params.alignment_offset);
    float * test_q2    = (float *) align_with_offset(test_q2_v.data(),    params.alignment_offset);
    float * test_out   = (float *) align_with_offset(test_out_v.data(),   params.alignment_offset);

    generate_data(0, largest, test_data1);
    generate_data(1, largest, test_data2);

    int64_t iterations = params.iterations;

    const bool use_backend = !params.backend_name.empty();

    backend_perf_context * bctx = nullptr;
    if (use_backend) {
        bctx = init_backend(params.backend_name);
        if (!bctx) {
            return 1;
        }
        printf("=== Backend mode: %s ===\n\n", params.backend_name.c_str());
    } else {
        ggml_cpu_init();
        printf("=== CPU mode ===\n\n");
    }

    std::map<ggml_type, double> quantize_times;
    std::map<ggml_type, double> dequantize_times;
    std::map<ggml_type, double> mul_mat_times;

    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        ggml_type type = (ggml_type) i;
        const auto * qfns = ggml_get_type_traits(type);
        const auto * qfns_cpu = ggml_get_type_traits_cpu(type);
        if (!params.include_types.empty() && ggml_type_name(type) && std::find(params.include_types.begin(), params.include_types.end(), ggml_type_name(type)) == params.include_types.end()) {
            continue;
        }

        if (qfns_cpu->from_float && qfns->to_float) {
            printf("%s\n", ggml_type_name(type));

            ggml_quantize_init(type);

            if (use_backend) {
                if (params.op_quantize_row_q || params.op_quantize_row_q_reference) {
                    printf("  quantize (cpy f32->quant)\n");
                    for (size_t size : params.test_sizes) {
                        printf("    %zu values (%.2f MB)\n", size, 4*size/(float)(1024*1024));
                        auto r = benchmark_backend_quantize(*bctx, type, size, iterations, test_data1);
                        if (r.avg_time_us > 0) { quantize_times[type] = r.avg_time_us; }
                    }
                    printf("\n");
                }

                if (params.op_dequantize_row_q) {
                    printf("  dequantize (cpy quant->f32)\n");
                    for (size_t size : params.test_sizes) {
                        printf("    %zu values (%.2f MB)\n", size, 4*size/(float)(1024*1024));
                        auto r = benchmark_backend_dequantize(*bctx, type, size, iterations, test_data1);
                        if (r.avg_time_us > 0) { dequantize_times[type] = r.avg_time_us; }
                    }
                    printf("\n");
                }

                if (params.op_vec_dot_q || params.op_quantize_row_q_dot) {
                    printf("  mul_mat (vec_dot equivalent)\n");
                    for (size_t size : params.test_sizes) {
                        printf("    %zu values (%.2f MB)\n", size, 4*size/(float)(1024*1024));
                        auto r = benchmark_backend_mul_mat(*bctx, type, size, iterations, test_data1, test_data2);
                        if (r.avg_time_us > 0) { mul_mat_times[type] = r.avg_time_us; }
                    }
                    printf("\n");
                }
            } else {
                if (params.op_quantize_row_q_reference) {
                    printf("  quantize_row_q_reference\n");
                    for (size_t size : params.test_sizes) {
                        printf("    %zu values (%.2f MB)\n", size, 4*size/(float)(1024*1024));
                        auto quantize_fn = [&](void) -> float {
                            qfns->from_float_ref(test_data1, test_q1, size);
                            return test_q1[0];
                        };
                        size_t quantized_size = ggml_row_size(type, size);
                        benchmark_function(size, quantized_size, iterations, quantize_fn);
                    }
                    printf("\n");
                }

                if (params.op_quantize_row_q) {
                    printf("  quantize_row_q\n");
                    for (size_t size : params.test_sizes) {
                        printf("    %zu values (%.2f MB)\n", size, 4*size/(float)(1024*1024));
                        auto quantize_fn = [&](void) -> float {
                            qfns_cpu->from_float(test_data1, test_q1, size);
                            return test_q1[0];
                        };
                        size_t quantized_size = ggml_row_size(type, size);
                        auto r = benchmark_function(size, quantized_size, iterations, quantize_fn);
                        quantize_times[type] = r.avg_time_us;
                    }
                    printf("\n");
                }

                if (params.op_dequantize_row_q) {
                    printf("  dequantize_row_q\n");
                    qfns_cpu->from_float(test_data1, test_q1, largest);
                    for (size_t size : params.test_sizes) {
                        printf("    %zu values (%.2f MB)\n", size, 4*size/(float)(1024*1024));
                        auto quantize_fn = [&](void) -> float {
                            qfns->to_float(test_q1, test_out, size);
                            return test_out[0];
                        };
                        size_t quantized_size = ggml_row_size(type, size);
                        benchmark_function(size, quantized_size, iterations, quantize_fn);
                    }
                    printf("\n");
                }

                if (params.op_quantize_row_q_dot) {
                    printf("  quantize_row_q_dot\n");
                    for (size_t size : params.test_sizes) {
                        printf("    %zu values (%.2f MB)\n", size, 4*size/(float)(1024*1024));
                        auto quantize_fn = [&](void) -> float {
                            const auto * vdot = ggml_get_type_traits_cpu(qfns_cpu->vec_dot_type);
                            vdot->from_float(test_data1, test_q1, size);
                            return test_q1[0];
                        };
                        size_t quantized_size = ggml_row_size(type, size);
                        benchmark_function(size, quantized_size, iterations, quantize_fn);
                    }
                    printf("\n");
                }

                if (params.op_vec_dot_q) {
                    printf("  vec_dot_q\n");
                    qfns_cpu->from_float(test_data1, test_q1, largest);
                    qfns_cpu->from_float(test_data2, test_q2, largest);
                    for (size_t size : params.test_sizes) {
                        printf("    %zu values (%.2f MB)\n", size, 4*size/(float)(1024*1024));
                        auto quantize_fn = [&](void) -> float {
                            float result;
                            qfns_cpu->vec_dot(size, &result, 0, test_q1, 0, test_q2, 0, 1);
                            return result;
                        };
                        size_t quantized_size = ggml_row_size(type, size);
                        auto r = benchmark_function(size, quantized_size, iterations, quantize_fn);
                        mul_mat_times[type] = r.avg_time_us;
                    }
                    printf("\n");
                }
            }
        }
    }

    // TurboQuant perf sanity checks (soft warnings, not hard failures).
    // PQ should be faster than TBQ (no QJL overhead in quantize).
    // 4-bit dequant should be faster than 3-bit (simpler nibble extraction vs bit-spanning).
    auto check_faster = [](const std::map<ggml_type, double> & times,
                           ggml_type faster, ggml_type slower, const char * op) {
        auto it_f = times.find(faster);
        auto it_s = times.find(slower);
        if (it_f == times.end() || it_s == times.end()) return;
        if (it_f->second <= 0 || it_s->second <= 0) return;
        const char * name_f = ggml_type_name(faster);
        const char * name_s = ggml_type_name(slower);
        if (it_f->second <= it_s->second) {
            printf("  PERF OK:      %s %s (%.1f us) <= %s (%.1f us)\n",
                   op, name_f, it_f->second, name_s, it_s->second);
        } else {
            printf("  PERF WARNING: %s %s (%.1f us) > %s (%.1f us) — expected %s to be faster\n",
                   op, name_f, it_f->second, name_s, it_s->second, name_f);
        }
    };

    if (!quantize_times.empty() || !dequantize_times.empty() || !mul_mat_times.empty()) {
        printf("\n=== TurboQuant perf sanity checks ===\n");

        // PQ quantize should be faster than TBQ (no QJL residual computation)
        check_faster(quantize_times, GGML_TYPE_PQ3_0,  GGML_TYPE_TBQ3_0,  "quantize");
        check_faster(quantize_times, GGML_TYPE_PQ4_0,  GGML_TYPE_TBQ4_0,  "quantize");

        // 4-bit dequant should be comparable or faster than 3-bit (nibble vs bit-spanning)
        check_faster(dequantize_times, GGML_TYPE_PQ4_0,  GGML_TYPE_PQ3_0,  "dequantize");
        check_faster(dequantize_times, GGML_TYPE_TBQ4_0, GGML_TYPE_TBQ3_0, "dequantize");

        // TBQ/PQ dequant should be in same ballpark as q4_0 (not orders of magnitude slower)
        auto it_q4 = dequantize_times.find(GGML_TYPE_Q4_0);
        for (ggml_type t : { GGML_TYPE_TBQ3_0, GGML_TYPE_TBQ4_0, GGML_TYPE_PQ3_0, GGML_TYPE_PQ4_0 }) {
            auto it = dequantize_times.find(t);
            if (it != dequantize_times.end() && it_q4 != dequantize_times.end() && it_q4->second > 0) {
                double ratio = it->second / it_q4->second;
                if (ratio > 5.0) {
                    printf("  PERF WARNING: dequantize %s is %.1fx slower than q4_0\n",
                           ggml_type_name(t), ratio);
                } else {
                    printf("  PERF OK:      dequantize %s is %.1fx vs q4_0\n",
                           ggml_type_name(t), ratio);
                }
            }
        }

        printf("=== end perf checks ===\n\n");
    }

    delete bctx;
    return 0;
}
