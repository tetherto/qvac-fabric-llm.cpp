#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
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
    return (GRAPH_NODES * SPLIT_INPUTS * COPY_NODES + GRAPH_NODES) * sizeof(ggml_tensor) +
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

}  // namespace

int main() try {
    ggml_backend_load_all();
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
