#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

constexpr size_t  GRAPH_NODES       = 32768;
constexpr size_t  SPLIT_INPUTS      = 30;
constexpr size_t  COPY_NODES        = 4;
constexpr int64_t WIDTH             = 128;
constexpr int64_t ROWS              = 64;
constexpr int64_t BATCH             = 32;
constexpr int     EXECUTION_THREADS = 4;
constexpr int     MANY_THREADS      = 1024;
constexpr size_t  SLOT_POINTERS     = 3;

void check(bool passed, const char * message) {
    if (!passed) {
        throw std::runtime_error(message);
    }
}

void set_threads(ggml_backend_t backend, int threads) {
    auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    auto   set = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    check(set != nullptr, "CPU thread setter unavailable");
    set(backend, threads);
}

struct Probe {
    ggml_context * ctx     = nullptr;
    ggml_cgraph *  graph   = nullptr;
    ggml_tensor *  weights = nullptr;
    ggml_tensor *  input   = nullptr;
    ggml_tensor *  product = nullptr;

    Probe() {
        ctx = ggml_init({ ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(16, false), nullptr, true });
        check(ctx != nullptr, "probe context allocation failed");
        weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, WIDTH, ROWS);
        input   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, WIDTH, BATCH);
        ggml_set_input(weights);
        ggml_set_input(input);
        product = ggml_mul_mat(ctx, weights, input);
        ggml_set_output(product);
        graph = ggml_new_graph_custom(ctx, 16, false);
        ggml_build_forward_expand(graph, product);
    }

    ~Probe() { ggml_free(ctx); }
};

size_t scheduler_minimum() {
    return GRAPH_NODES * SPLIT_INPUTS * COPY_NODES * sizeof(ggml_tensor) +
           (GRAPH_NODES + GRAPH_NODES * SPLIT_INPUTS * COPY_NODES) * sizeof(int) * COPY_NODES +
           ggml_graph_overhead_custom(GRAPH_NODES, false);
}

void check_host(ggml_backend_t cpu) {
    auto *     sched = ggml_backend_sched_new(&cpu, nullptr, 1, GRAPH_NODES, false, false);
    const auto empty = ggml_backend_sched_get_host_size(sched);
    check(empty >= scheduler_minimum(), "scheduler host query omits allocated descriptor storage");
    check(empty > (size_t(1) << 30), "scheduler host storage incorrectly fits within 1 GiB");
    check(ggml_backend_get_work_size(cpu) == 0, "host query allocated CPU scratch");
    Probe  probe;
    size_t sizes[1] = { 0 };
    check(ggml_backend_sched_reserve_size(sched, probe.graph, sizes), "size-only reservation failed");
    const auto measured = ggml_backend_sched_get_host_size(sched);
    check(measured > empty, "host query omits graph allocator metadata");
    check(ggml_backend_sched_get_buffer_size(sched, cpu) == 0, "size-only reservation allocated tensor buffers");
    ggml_backend_sched_reset(sched);
    check(ggml_backend_sched_get_host_size(sched) == measured, "reset lost retained host allocations");
    ggml_backend_sched_free(sched);
}

size_t reserved_host_size(ggml_backend_buffer_type_t * bufts, int count, Probe & probe) {
    auto * galloc = ggml_gallocr_new_n(bufts, count);
    check(ggml_gallocr_reserve(galloc, probe.graph), "graph allocator reservation failed");
    const size_t size = ggml_gallocr_get_host_size(galloc);
    ggml_gallocr_free(galloc);
    return size;
}

void check_shared_buffers(ggml_backend_t cpu, Probe & probe) {
    ggml_backend_buffer_type_t single[1]   = { ggml_backend_get_default_buffer_type(cpu) };
    ggml_backend_buffer_type_t repeated[2] = { single[0], single[0] };
    const size_t               one         = reserved_host_size(single, 1, probe);
    const size_t               two         = reserved_host_size(repeated, 2, probe);
    check(two == one + SLOT_POINTERS * sizeof(void *), "repeated buffer type counts its shared buffer twice");
}

void check_plan(ggml_backend_t cpu, Probe & probe) {
    auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(cpu));
    auto plan = reinterpret_cast<decltype(&ggml_graph_plan)>(ggml_backend_reg_get_proc_address(reg, "ggml_graph_plan"));
    check(plan != nullptr, "CPU graph planner unavailable");
    set_threads(cpu, EXECUTION_THREADS);
    const size_t ordinary = ggml_backend_graph_get_work_size(cpu, probe.graph);
    check(ordinary == plan(probe.graph, EXECUTION_THREADS, nullptr).work_size,
          "graph workspace query differs from the CPU plan");
    set_threads(cpu, MANY_THREADS);
    const size_t many = ggml_backend_graph_get_work_size(cpu, probe.graph);
    check(many == plan(probe.graph, MANY_THREADS, nullptr).work_size,
          "graph workspace query ignores configured threads");
    check(many > ordinary, "probe does not exercise thread-dependent workspace");
    check(ggml_backend_get_work_size(cpu) == 0, "workspace query allocated scratch");
}

void check_execution(ggml_backend_t primary, ggml_backend_t cpu) {
    ggml_backend_t backends[2] = { primary, cpu };
    const int      count       = primary == cpu ? 1 : 2;
    auto *         sched       = ggml_backend_sched_new(backends, nullptr, count, 32, false, false);
    Probe          probe;
    set_threads(cpu, EXECUTION_THREADS);
    if (count == 2) {
        auto * output = ggml_scale(probe.ctx, probe.product, 1.0f);
        ggml_set_output(output);
        ggml_build_forward_expand(probe.graph, output);
        ggml_backend_sched_set_tensor_backend(sched, probe.product, cpu);
        ggml_backend_sched_set_tensor_backend(sched, output, primary);
    }
    size_t sizes[2] = { 0, 0 };
    check(ggml_backend_sched_reserve_size(sched, probe.graph, sizes), "split reservation failed");
    const size_t projected = ggml_backend_sched_get_work_size(sched, cpu);
    check(projected > 0, "CPU split workspace omitted");
    if (count == 2) {
        check(ggml_backend_sched_get_work_size(sched, primary) == 0, "unused backend charged workspace");
    }
    check(ggml_backend_sched_reserve(sched, probe.graph), "probe tensor buffer reservation failed");
    check(ggml_backend_sched_alloc_graph(sched, probe.graph), "probe graph allocation failed");
    const auto host = ggml_backend_sched_get_host_size(sched);
    ggml_backend_tensor_memset(probe.weights, 0, 0, ggml_nbytes(probe.weights));
    ggml_backend_tensor_memset(probe.input, 0, 0, ggml_nbytes(probe.input));
    check(ggml_backend_sched_graph_compute(sched, probe.graph) == GGML_STATUS_SUCCESS, "probe execution failed");
    check(ggml_backend_get_work_size(cpu) == projected, "allocated CPU scratch differs from split projection");
    check(ggml_backend_sched_get_host_size(sched) == host, "workspace counted twice in scheduler metadata");
    ggml_backend_sched_free(sched);
}

void check_gpu(ggml_backend_t cpu) {
    if (!std::getenv("GGML_MEMORY_TEST_GPU")) {
        return;
    }
    auto * gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!gpu) {
        gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
    }
    check(gpu != nullptr, "GPU workspace test has no GPU backend");
    check_execution(gpu, cpu);
    ggml_backend_free(gpu);
}

void check_repack_matmul(ggml_backend_t backend, ggml_tensor * weights,
                         const std::vector<uint8_t> & expected, const char * name, int & checks) {
    const int64_t width = weights->ne[0];
    const int64_t rows = weights->ne[1];
    auto * ctx = ggml_init({ggml_tensor_overhead() * 4 + ggml_graph_overhead_custom(8, false), nullptr, true});
    check(ctx != nullptr, "REPACK matmul context allocation failed");
    auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, 1);
    auto * output = ggml_mul_mat(ctx, weights, input);
    check(ggml_backend_supports_op(backend, output), "REPACK matmul unsupported");
    auto * graph = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(graph, output);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    check(buffer != nullptr, "REPACK matmul buffer allocation failed");

    std::vector<float> activation(width), canonical(width * rows), actual(rows);
    for (int64_t i = 0; i < width; ++i) {
        activation[i] = float(int((i * 13) % 71) - 35) / 37.0f;
    }
    const auto * traits = ggml_get_type_traits(weights->type);
    check(traits->to_float != nullptr, "REPACK reference dequantizer unavailable");
    traits->to_float(expected.data(), canonical.data(), width * rows);
    ggml_backend_tensor_set(input, activation.data(), 0, activation.size() * sizeof(float));
    const auto status = ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);
    check(status == GGML_STATUS_SUCCESS, "REPACK matmul execution failed");
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));

    double squared_error = 0.0;
    double squared_reference = 0.0;
    bool finite = true;
    for (int64_t row = 0; row < rows; ++row) {
        double reference = 0.0;
        for (int64_t col = 0; col < width; ++col) {
            reference += double(canonical[row * width + col]) * activation[col];
        }
        finite = finite && std::isfinite(reference) && std::isfinite(actual[row]);
        const double difference = reference - actual[row];
        squared_error += difference * difference;
        squared_reference += reference * reference;
    }
    const double nmse = squared_reference > 0.0 ? squared_error / squared_reference : INFINITY;
    const bool passed = finite && std::isfinite(nmse) && nmse <= 5e-4;
    std::printf("%s %s matmul: %s (NMSE %.12g, limit 0.0005)\n",
                ggml_type_name(weights->type), name, passed ? "PASS" : "FAIL", nmse);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    check(passed, "REPACK matmul differs from canonical reference");
    ++checks;
}

void check_foreign_set_rows(ggml_backend_t backend, int & checks) {
    auto * cpu = ggml_backend_cpu_init();
    check(cpu != nullptr, "foreign-buffer CPU backend unavailable");
    auto * ctx = ggml_init({ggml_tensor_overhead() * 8, nullptr, true});
    check(ctx != nullptr, "foreign-buffer context allocation failed");
    auto * destination = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 256, 32);
    auto * alias = ggml_view_tensor(ctx, destination);
    auto * values = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 1);
    auto * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
    auto * write = ggml_set_rows(ctx, destination, values, indices);
    auto * alias_write = ggml_set_rows(ctx, alias, values, indices);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
    check(buffer != nullptr, "foreign-buffer allocation failed");
    check(ggml_backend_supports_op(backend, write), "SET_ROWS rejected a canonical CPU destination");
    ++checks;

    // A foreign backend's opaque metadata must not affect Hexagon layout detection.
    std::vector<uint8_t> foreign_extra(ggml_nbytes(destination), 0xff);
    void * original_extra = destination->extra;
    destination->extra = foreign_extra.data();
    const bool direct_supported = ggml_backend_supports_op(backend, write);
    const bool alias_supported = ggml_backend_supports_op(backend, alias_write);
    destination->extra = original_extra;
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(cpu);
    check(direct_supported, "SET_ROWS inspected foreign CPU metadata");
    check(alias_supported, "SET_ROWS inspected foreign CPU alias metadata");
    checks += 2;
    std::puts("foreign CPU SET_ROWS queries: 3 PASS");
}

void check_runtime_repack(ggml_backend_t backend, ggml_type type,
                          const std::vector<uint8_t> & a, const std::vector<uint8_t> & b, int & checks) {
    auto * ctx = ggml_init({ggml_tensor_overhead() * 4, nullptr, true});
    check(ctx != nullptr, "runtime REPACK context allocation failed");
    auto * weights = ggml_new_tensor_2d(ctx, type, 256, 32);
    auto * alias = ggml_view_tensor(ctx, weights);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    check(buffer != nullptr, "runtime REPACK buffer allocation failed");
    // Keep the default buffer usage so Hexagon tiles these weights during compute.
    ggml_backend_tensor_set(weights, a.data(), 0, a.size());
    check_repack_matmul(backend, weights, a, "runtime-full", checks);
    std::vector<uint8_t> actual(a.size());
    ggml_backend_tensor_get(alias, actual.data(), 0, actual.size());
    check(actual == a, "runtime REPACK lost canonical bytes");
    ++checks;

    const size_t half = a.size() / 2;
    std::vector<uint8_t> expected = a;
    std::copy(b.begin() + half, b.end(), expected.begin() + half);
    ggml_backend_tensor_set(alias, b.data() + half, half, half);
    // Hexagon rejects quantized views as matmul inputs; compute through their canonical owner.
    check_repack_matmul(backend, weights, expected, "runtime-alias-partial", checks);
    ggml_backend_tensor_get(weights, actual.data(), 0, actual.size());
    check(actual == expected, "runtime REPACK partial update lost untouched bytes");
    ++checks;
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

void check_repack_type(ggml_backend_t backend, ggml_type type, int & checks) {
    auto * ctx = ggml_init({ggml_tensor_overhead() * 16, nullptr, true});
    check(ctx != nullptr, "REPACK context allocation failed");
    auto * overwrite = ggml_new_tensor_2d(ctx, type, 256, 32);
    auto * streamed = ggml_new_tensor_2d(ctx, type, 256, 32);
    auto * strided = ggml_new_tensor_2d(ctx, type, 256, 32);
    auto * copied = ggml_new_tensor_2d(ctx, type, 256, 32);
    auto * alias = ggml_view_tensor(ctx, overwrite);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    check(buffer != nullptr, "REPACK buffer allocation failed");
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    const size_t bytes = ggml_nbytes(overwrite);
    const size_t half = bytes / 2;
    const size_t row = ggml_row_size(type, 256);
    if (std::strncmp(ggml_backend_name(backend), "HTP", 3) == 0) {
        const size_t allocated = ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), overwrite);
        const bool needs_canonical = type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K;
        check(needs_canonical ? allocated >= 2 * bytes : allocated < 2 * bytes,
              "REPACK canonical storage is missing from the buffer allocation or duplicated for a reversible type");
        ++checks;
    }
    std::vector<float> values(256 * 32);
    std::vector<uint8_t> a(bytes), b(bytes), actual(bytes), expected(bytes);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = float(int(i % 97) - 48) / 31.0f;
    }
    check(ggml_quantize_chunk(type, values.data(), a.data(), 0, 32, 256, nullptr) == bytes, "REPACK quantization failed");
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = float(int(i % 67) - 33) / 19.0f;
    }
    check(ggml_quantize_chunk(type, values.data(), b.data(), 0, 32, 256, nullptr) == bytes, "REPACK quantization failed");
    check_runtime_repack(backend, type, a, b, checks);
    auto verify = [&](ggml_tensor * tensor, const char * name, bool compute = false) {
        ggml_backend_tensor_get(tensor, actual.data(), 0, bytes);
        const bool passed = actual == expected;
        std::printf("%s %s: %s\n", ggml_type_name(type), name, passed ? "PASS" : "FAIL");
        check(passed, "REPACK canonical bytes changed");
        ++checks;
        if (compute) {
            check_repack_matmul(backend, tensor, expected, name, checks);
        }
    };

    expected = a;
    ggml_backend_tensor_set(overwrite, a.data(), 0, bytes);
    verify(overwrite, "full");
    std::vector<uint8_t> slice(row + 3);
    ggml_backend_tensor_get(overwrite, slice.data(), row / 2 + 1, slice.size());
    check(std::equal(slice.begin(), slice.end(), a.begin() + row / 2 + 1), "REPACK partial byte readback changed");
    ++checks;
    std::copy(b.begin() + half, b.end(), expected.begin() + half);
    for (int repeat = 0; repeat < 2; ++repeat) {
        ggml_backend_tensor_set(overwrite, b.data() + half, half, half);
        verify(overwrite, repeat == 0 ? "partial" : "repeat", repeat == 1);
    }

    ggml_backend_tensor_set(streamed, a.data(), 0, half);
    ggml_backend_tensor_set(streamed, a.data(), 0, half);
    ggml_backend_tensor_set(streamed, b.data() + half, half, half);
    verify(streamed, "streamed-overlap", true);

    ggml_backend_tensor_set(strided, a.data(), 0, bytes);
    for (int repeat = 0; repeat < 2; ++repeat) {
        ggml_backend_tensor_set_2d(strided, b.data() + half, half, row, 16, row, row);
        std::vector<uint8_t> upper(half);
        ggml_backend_tensor_get_2d(strided, upper.data(), half, row, 16, row, row);
        check(std::equal(upper.begin(), upper.end(), b.begin() + half), "REPACK strided readback changed");
        verify(strided, repeat == 0 ? "strided" : "strided-repeat", repeat == 1);
    }

    ggml_backend_tensor_copy_async(backend, backend, overwrite, copied);
    ggml_backend_synchronize(backend);
    verify(copied, "copy");
    std::copy(b.begin(), b.begin() + row, expected.begin());
    ggml_backend_tensor_set(copied, b.data(), 0, row);
    verify(copied, "copy-partial", true);

    ggml_backend_buffer_clear(buffer, 0);
    ggml_backend_tensor_get(overwrite, expected.data(), 0, bytes);
    std::copy(b.begin() + half, b.end(), expected.begin() + half);
    ggml_backend_tensor_set(overwrite, b.data() + half, half, half);
    verify(overwrite, "clear-partial");
    expected = a;
    ggml_backend_tensor_set(overwrite, a.data(), 0, bytes);
    verify(overwrite, "replace-full");
    verify(alias, "alias-read");
    expected = b;
    ggml_backend_tensor_set(alias, b.data(), 0, bytes);
    verify(overwrite, "alias-full-write");
    ggml_backend_tensor_set(overwrite, a.data(), 0, bytes);
    expected = a;
    std::copy(b.begin() + half, b.end(), expected.begin() + half);
    ggml_backend_tensor_set(alias, b.data() + half, half, half);
    verify(overwrite, "alias-partial-write", true);
    expected = b;
    ggml_backend_tensor_set(overwrite, b.data(), 0, half);
    verify(alias, "alias-base-update");
    check_repack_matmul(backend, overwrite, expected, "alias-base-update", checks);
    ggml_backend_buffer_clear(buffer, 0);
    ggml_backend_tensor_get(overwrite, expected.data(), 0, bytes);
    verify(alias, "alias-clear");

    if (type == GGML_TYPE_Q8_0 && std::strncmp(ggml_backend_name(backend), "HTP", 3) == 0) {
        auto * values = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 1);
        auto * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
        auto * write = ggml_set_rows(ctx, alias, values, indices);
        check(!ggml_backend_supports_op(backend, write), "SET_ROWS accepted a packed destination");
        std::puts("q8_0 packed SET_ROWS rejection: PASS");
        ++checks;
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

void check_repack(const char * name) {
    auto * dev = ggml_backend_dev_by_name(name);
    check(dev != nullptr, "REPACK backend unavailable");
    auto * backend = ggml_backend_dev_init(dev, nullptr);
    check(backend != nullptr, "REPACK backend initialization failed");
    int checks = 0;
    for (ggml_type type : {GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL,
                          GGML_TYPE_MXFP4, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K}) {
        check_repack_type(backend, type, checks);
    }
    if (std::strncmp(ggml_backend_name(backend), "HTP", 3) == 0) {
        Probe probe;
        auto * buft = ggml_backend_dev_buffer_type(dev);
        check(ggml_backend_buft_get_alloc_size(buft, probe.product) == ggml_nbytes(probe.product),
              "REPACK expanded a computed F32 tensor");
        ++checks;
        check_foreign_set_rows(backend, checks);
    }
    ggml_backend_free(backend);
    std::printf("REPACK checks: %d passed, 0 failed, 0 unsupported, 0 skipped\n", checks);
}

}  // namespace

int main(int argc, char ** argv) try {
    ggml_backend_load_all();
    if (argc == 3 && std::strcmp(argv[1], "--repack") == 0) {
        check_repack(argv[2]);
        return 0;
    }
    auto * cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    check(cpu != nullptr, "CPU backend unavailable");
    check_host(cpu);
    Probe probe;
    check_shared_buffers(cpu, probe);
    check_plan(cpu, probe);
    check_execution(cpu, cpu);
    ggml_backend_free(cpu);
    cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    check_gpu(cpu);
    ggml_backend_free(cpu);
    std::puts("test-backend-memory: all checks passed");
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
