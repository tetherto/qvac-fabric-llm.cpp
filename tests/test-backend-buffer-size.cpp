#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void check(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// Host-portable backend with a deliberately small payload cap and a non-page
// guard. Like the Hexagon backend it reports the payload size on the buffer
// and keeps the guard inside the physical backing allocation. The allocator's
// own payload chunks are the oracle, not a second implementation of its
// packing algorithm.
struct fixture {
    ggml_backend_reg reg = {};
    ggml_backend_device dev = {};
    ggml_backend_buffer_type buft = {};
    size_t guard = 37;
    size_t max_size = 256;
    bool saturate = false;
    size_t tensor_size_override = 0;
    size_t freed_buffers = 0;
    std::vector<size_t> payloads;

    static fixture & get(ggml_backend_buffer_type_t type) {
        return *static_cast<fixture *>(type->context);
    }
    static size_t predict(ggml_backend_buffer_type_t type, size_t size) {
        const auto & f = get(type);
        return f.saturate || size > SIZE_MAX - f.guard ? SIZE_MAX : size + f.guard;
    }
    static void * proc(ggml_backend_reg_t, const char * name) {
        return std::strcmp(name, "ggml_backend_buft_get_buffer_alloc_size") == 0 ?
            reinterpret_cast<void *>(predict) : nullptr;
    }
    static const char * name(ggml_backend_buffer_type_t) { return "guard-probe"; }
    static size_t alignment(ggml_backend_buffer_type_t) { return 16; }
    static size_t maximum(ggml_backend_buffer_type_t type) { return get(type).max_size; }
    static size_t tensor_size(ggml_backend_buffer_type_t type, const ggml_tensor * tensor) {
        return get(type).tensor_size_override ? get(type).tensor_size_override : ggml_nbytes(tensor);
    }
    static bool is_host(ggml_backend_buffer_type_t) { return true; }
    static void free_buffer(ggml_backend_buffer_t buffer) {
        ++get(ggml_backend_buffer_get_type(buffer)).freed_buffers;
        ggml_backend_buffer_free(static_cast<ggml_backend_buffer_t>(buffer->context));
    }
    static void * base(ggml_backend_buffer_t buffer) {
        return ggml_backend_buffer_get_base(static_cast<ggml_backend_buffer_t>(buffer->context));
    }
    static ggml_backend_buffer_t alloc(ggml_backend_buffer_type_t type, size_t size) {
        get(type).payloads.push_back(size);
        const size_t physical = predict(type, size);
        auto * backing = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), physical);
        if (backing == nullptr) {
            return nullptr;
        }
        ggml_backend_buffer_i iface = {};
        iface.free_buffer = free_buffer;
        iface.get_base = base;
        return ggml_backend_buffer_init(type, iface, backing, size);
    }
    size_t physical_total() {
        size_t total = 0;
        for (size_t payload : payloads) {
            total += predict(&buft, payload);
        }
        return total;
    }
    fixture() {
        reg.iface.get_proc_address = proc;
        dev.reg = &reg;
        buft.device = &dev;
        buft.context = this;
        buft.iface.get_name = name;
        buft.iface.alloc_buffer = alloc;
        buft.iface.get_alignment = alignment;
        buft.iface.get_max_size = maximum;
        buft.iface.get_alloc_size = tensor_size;
        buft.iface.is_host = is_host;
    }
};

struct context {
    ggml_context * ctx = nullptr;
    context() {
        ctx = ggml_init({64 * ggml_tensor_overhead() + ggml_graph_overhead_custom(64, false), nullptr, true});
        check(ctx != nullptr, "context allocation failed");
    }
    ~context() { ggml_free(ctx); }
};

void check_queries() {
    fixture f;
    check(ggml_backend_buft_get_buffer_alloc_size(&f.buft, 160) == 197, "callback missing guard");
    check(ggml_backend_buft_get_buffer_alloc_size(&f.buft, 0) == 0, "zero-size query called callback");
    auto * empty = ggml_backend_buft_alloc_buffer(&f.buft, 0);
    check(empty && ggml_backend_buffer_get_size(empty) == 0 && f.payloads.empty(), "zero-size dummy allocated payload");
    ggml_backend_buffer_free(empty);
    check(ggml_backend_buft_get_buffer_alloc_size(&f.buft, SIZE_MAX - 20) == SIZE_MAX, "guard addition wrapped");
    f.reg.iface.get_proc_address = nullptr;
    check(ggml_backend_buft_get_buffer_alloc_size(&f.buft, 160) == 160, "absent callback is not identity");
    f.buft.device = nullptr;
    check(ggml_backend_buft_get_buffer_alloc_size(&f.buft, 160) == 160, "device-less buffer is not identity");
    check(ggml_backend_buft_get_buffer_alloc_size(ggml_backend_cpu_buffer_type(), 160) == 160, "CPU size changed");
}

void check_context() {
    fixture f;
    context c;
    // Three 160-byte tensors cannot fit into two 256-byte buffers, although
    // ceil(total/cap) is two. Views and empty tensors add no physical buffers.
    auto * first = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, 40);
    auto * view = ggml_view_1d(c.ctx, first, 16, 0);
    ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, 40);
    ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, 0);
    ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, 40);
    const size_t projected = ggml_backend_alloc_ctx_tensors_from_buft_size(c.ctx, &f.buft);
    check(projected == 3 * (160 + f.guard), "context missed a physical chunk guard");
    check(f.payloads.empty() && first->buffer == nullptr && view->buffer == nullptr, "measurement allocated or modified tensors");
    auto * buffer = ggml_backend_alloc_ctx_tensors_from_buft(c.ctx, &f.buft);
    check(buffer && f.payloads.size() == 3 && f.physical_total() == projected,
        "context projection differs from actual multi-buffer allocation");
    check(ggml_backend_buffer_get_size(buffer) == 3 * 160, "buffer reports guards as payload");
    check(view->buffer == first->buffer, "view initialization changed");
    check(ggml_backend_alloc_ctx_tensors_from_buft_size(c.ctx, &f.buft) == 0, "allocated context charged twice");
    ggml_backend_buffer_free(buffer);
    context empty;
    check(ggml_backend_alloc_ctx_tensors_from_buft_size(empty.ctx, &f.buft) == 0, "empty context has a guard");
}

ggml_cgraph * graph(context & c, int inputs, int width) {
    auto * result = ggml_new_graph_custom(c.ctx, 64, false);
    for (int i = 0; i < inputs; ++i) {
        auto * input = ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, width);
        ggml_set_input(input);
        auto * output = ggml_dup(c.ctx, input);
        ggml_set_output(output);
        ggml_build_forward_expand(result, output);
    }
    return result;
}

void check_graph() {
    fixture f;
    context c;
    auto * g = graph(c, 3, 40);
    ggml_backend_buffer_type_t types[] = {&f.buft, &f.buft};
    auto * allocator = ggml_gallocr_new_n(types, 2);
    size_t sizes[2] = {};
    ggml_gallocr_reserve_n_size(allocator, g, nullptr, nullptr, sizes);
    check(sizes[0] > 0 && sizes[1] == 0, "aliased buffer types counted twice");
    check(f.payloads.empty() && ggml_gallocr_get_buffer_size(allocator, 0) == 0, "graph measurement allocated payloads");
    check(ggml_gallocr_reserve(allocator, g), "graph reserve failed");
    check(f.payloads.size() > 1, "graph did not exercise multiple physical chunks");
    check(f.physical_total() == sizes[0] && ggml_gallocr_get_buffer_size(allocator, 1) == 0,
        "graph projection differs from physical buffers");
    check(ggml_gallocr_get_buffer_size(allocator, 0) < sizes[0], "graph buffers report guards as payload");
    ggml_gallocr_free(allocator);
}

void check_growth() {
    fixture f;
    f.max_size = 4096;
    context small, large;
    auto * small_graph = graph(small, 1, 32);
    auto * large_graph = graph(large, 1, 36);
    auto * allocator = ggml_gallocr_new(&f.buft);
    check(ggml_gallocr_reserve(allocator, small_graph), "small graph reserve failed");
    const size_t old_size = ggml_gallocr_get_buffer_size(allocator, 0);
    const size_t old_allocations = f.payloads.size();
    check(ggml_gallocr_reserve(allocator, large_graph), "larger graph reserve failed");
    check(f.payloads.size() > old_allocations && ggml_gallocr_get_buffer_size(allocator, 0) > old_size,
        "graph growth did not allocate a larger buffer");
    ggml_gallocr_free(allocator);
}

void check_shared_growth() {
    fixture f;
    f.max_size = 4096;
    context small, large;
    auto * small_graph = graph(small, 1, 32);
    auto * large_graph = graph(large, 1, 36);
    auto * cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    check(cpu != nullptr, "shared-growth CPU backend unavailable");
    ggml_backend_buffer_type_t type = &f.buft;
    auto * source = ggml_backend_sched_new(&cpu, &type, 1, 64, false, false);
    auto * borrower = ggml_backend_sched_new(&cpu, &type, 1, 64, false, false);
    check(ggml_backend_sched_reserve(source, small_graph) &&
        ggml_backend_sched_alloc_graph(source, small_graph), "source shared graph allocation failed");
    const size_t source_bytes = ggml_backend_sched_get_buffer_size(source, cpu);
    auto * source_input = ggml_graph_node(small_graph, 0)->src[0];
    check(source_input->data != nullptr, "shared source input was not allocated");
    std::memset(source_input->data, 0x5a, ggml_nbytes(source_input));
    check(ggml_backend_sched_share_compute_buffers(borrower, source), "compute buffer sharing failed");
    check(ggml_backend_sched_reserve(borrower, large_graph), "borrower growth failed");
    // Growth is only 32 payload bytes. The borrower must detach from the
    // shared storage and leave the source's payload untouched.
    check(f.freed_buffers == 0, "borrower growth freed storage still referenced by the source");
    check(ggml_backend_sched_get_buffer_size(source, cpu) == source_bytes &&
        ggml_backend_sched_get_buffer_size(borrower, cpu) > source_bytes,
        "borrower growth replaced the source scheduler's shared storage");
    const auto * data = static_cast<const unsigned char *>(source_input->data);
    for (size_t i = 0; i < ggml_nbytes(source_input); ++i) {
        check(data[i] == 0x5a, "source payload changed after borrower detachment");
    }
    ggml_backend_sched_free(borrower);
    check(f.freed_buffers == 1, "borrower cleanup did not free exactly its detached buffer");
    check(ggml_backend_sched_get_buffer_size(source, cpu) == source_bytes,
        "borrower cleanup invalidated the source buffer");
    ggml_backend_sched_free(source);
    check(f.freed_buffers == 2, "shared source buffer leaked or was freed twice");
    ggml_backend_free(cpu);
}

void check_saturation() {
    fixture f;
    f.saturate = true;
    context c;
    ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, 40);
    ggml_new_tensor_1d(c.ctx, GGML_TYPE_F32, 40);
    check(ggml_backend_alloc_ctx_tensors_from_buft_size(c.ctx, &f.buft) == SIZE_MAX,
        "context physical-size sum wrapped");
    auto * g = graph(c, 2, 40);
    auto * allocator = ggml_gallocr_new(&f.buft);
    size_t size = 0;
    ggml_gallocr_reserve_n_size(allocator, g, nullptr, nullptr, &size);
    check(size == SIZE_MAX && f.payloads.empty(), "graph physical-size sum wrapped or allocated");
    ggml_gallocr_free(allocator);

    // Oversized payloads are only legal on leaf tensors: the backend layer
    // refuses a per-tensor size above ggml_nbytes for ordinary ops.
    context leaves;
    ggml_new_tensor_1d(leaves.ctx, GGML_TYPE_F32, 40);
    ggml_new_tensor_1d(leaves.ctx, GGML_TYPE_F32, 40);
    fixture payload;
    payload.max_size = SIZE_MAX;
    payload.tensor_size_override = SIZE_MAX / 2 + 1;
    check(ggml_backend_alloc_ctx_tensors_from_buft_size(leaves.ctx, &payload.buft) == SIZE_MAX,
        "context payload sum wrapped before chunk accounting");
    payload.tensor_size_override = SIZE_MAX - 7;
    check(ggml_backend_alloc_ctx_tensors_from_buft_size(leaves.ctx, &payload.buft) == SIZE_MAX && payload.payloads.empty(),
        "unalignable tensor size wrapped or allocated payloads");
}

} // namespace

int main() try {
    ggml_backend_load_all();
    check_queries();
    check_context();
    check_graph();
    check_growth();
    check_shared_growth();
    check_saturation();
    std::puts("test-backend-buffer-size: all checks passed");
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
}
