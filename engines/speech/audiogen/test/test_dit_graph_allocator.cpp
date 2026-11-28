#include "acestep/dit_graph_allocator.h"

#include <array>
#include <cstdio>
#include <cstdint>
#include <stdexcept>

namespace {

constexpr size_t kGraphSlots = GGML_DEFAULT_GRAPH_SIZE;
constexpr int64_t kElements = 4;
constexpr int kReuses = 5;

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

struct Graph {
    ggml_context * context = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * factor = nullptr;
    ggml_tensor * sum = nullptr;
    ggml_tensor * output = nullptr;

    explicit Graph(bool metadata_only = false, bool external = true) {
        const ggml_init_params params{ggml_tensor_overhead() * kGraphSlots +
            ggml_graph_overhead_custom(kGraphSlots, false), nullptr, true};
        context = ggml_init(params);
        require(context != nullptr, "graph context allocation failed");
        graph = ggml_new_graph_custom(context, kGraphSlots, false);
        input = ggml_new_tensor_1d(context, GGML_TYPE_F32, kElements);
        factor = ggml_new_tensor_1d(context, GGML_TYPE_F32, kElements);
        ggml_set_input(input);
        ggml_set_input(factor);
        if (metadata_only) {
            factor->flags = 0;
            if (external) factor->data = reinterpret_cast<void *>(static_cast<uintptr_t>(1));
        }
        auto * factor_view = ggml_view_1d(context, factor, kElements, 0);
        sum = ggml_add(context, input, factor_view);
        output = ggml_mul(context, sum, factor_view);
        ggml_set_output(output);
        ggml_build_forward_expand(graph, output);
    }

    ~Graph() { ggml_free(context); }
};

void assign_splits(tts_cpp::acestep::DitGraphAllocator & allocator, const Graph & graph,
                   ggml_backend_t active, ggml_backend_t cpu) {
    if (!allocator.scheduler()) return;
    ggml_backend_sched_set_tensor_backend(allocator.scheduler(), graph.sum, active);
    ggml_backend_sched_set_tensor_backend(allocator.scheduler(), graph.output, cpu);
}

void run_reused(tts_cpp::acestep::DitGraphAllocator & allocator, Graph & graph) {
    const std::array<float, kElements> factors{4, 3, 2, 1};
    for (int iteration = 0; iteration < kReuses; ++iteration) {
        const std::array<float, kElements> input{1.0f + iteration, 2, 3, 4};
        require(allocator.allocate(graph.graph), "cached graph allocation failed");
        ggml_backend_tensor_set(graph.input, input.data(), 0, sizeof(input));
        ggml_backend_tensor_set(graph.factor, factors.data(), 0, sizeof(factors));
        require(allocator.compute(graph.graph) == GGML_STATUS_SUCCESS, "cached graph computation failed");
        std::array<float, kElements> actual{};
        ggml_backend_tensor_get(graph.output, actual.data(), 0, sizeof(actual));
        require(actual == std::array<float, kElements>{(input[0] + factors[0]) * factors[0], 15, 10, 5},
                "cached graph reused stale inputs or invalid source pointers");
    }
}

void test_allocator(ggml_backend_t active, ggml_backend_t cpu, bool fallback) {
    Graph projection;
    tts_cpp::acestep::DitGraphAllocator measure;
    require(measure.initialize(active, cpu, kGraphSlots, fallback), "measurement allocator initialization failed");
    size_t estimated = 0;
    require(measure.reserve(projection.graph, &estimated), "graph measurement failed");
    require(estimated > 0 && measure.bytes() == 0, "measurement allocated a compute buffer");

    Graph graph;
    tts_cpp::acestep::DitGraphAllocator allocator;
    require(allocator.initialize(active, cpu, kGraphSlots, fallback), "allocator initialization failed");
    require(allocator.reserve(graph.graph, nullptr), "graph reservation failed");
    if (allocator.bytes() != estimated) {
        std::fprintf(stderr, "fallback=%d measured=%zu reserved=%zu\n", fallback, estimated, allocator.bytes());
    }
    require(allocator.bytes() == estimated, "graph measurement differs from reserved compute buffers");
    run_reused(allocator, graph);
    allocator.release();
    require(allocator.bytes() == 0, "released allocator retained compute buffers");
}

void test_split_reuse(ggml_backend_t active, ggml_backend_t cpu) {
    Graph graph;
    tts_cpp::acestep::DitGraphAllocator allocator;
    require(allocator.initialize(active, cpu, kGraphSlots, true), "split allocator initialization failed");
    assign_splits(allocator, graph, active, cpu);
    require(allocator.reserve(graph.graph, nullptr), "split graph reservation failed");
    require(ggml_backend_sched_get_n_splits(allocator.scheduler()) == 2, "expected two backend splits");
    run_reused(allocator, graph);
}

void test_metadata_views(ggml_backend_t active, ggml_backend_t cpu, bool external) {
    Graph graph(true, external);
    tts_cpp::acestep::DitGraphAllocator allocator;
    require(allocator.initialize(active, cpu, kGraphSlots, true), "metadata allocator initialization failed");
    size_t estimated = 0;
    require(allocator.reserve(graph.graph, &estimated), "external weight measurement failed");
    require(estimated > 0 && allocator.bytes() == 0, "metadata projection allocated a compute buffer");
    require(graph.factor->flags == 0, "projection changed external weight flags");
    const auto expected_data = external ? reinterpret_cast<void *>(static_cast<uintptr_t>(1)) : nullptr;
    require(graph.factor->data == expected_data &&
            graph.factor->buffer == nullptr, "projection allocated external weights");
}

}

int main() {
    ggml_backend_load_all();
    ggml_backend_t active = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_t gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!gpu) gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
    int result = 0;
    try {
        require(active && cpu, "CPU backends unavailable");
        test_allocator(active, cpu, false);
        test_allocator(active, cpu, true);
        test_metadata_views(active, cpu, true);
        test_metadata_views(active, cpu, false);
        if (gpu) test_split_reuse(gpu, cpu);
        else std::puts("Split reuse requires an accelerator; CPU scheduler reuse was tested");
        std::puts("DiT allocator reuse and memory projection passed");
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        result = 1;
    }
    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    ggml_backend_free(active);
    return result;
}
