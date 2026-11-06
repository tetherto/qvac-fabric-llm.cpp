#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

void check(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

constexpr size_t GUARD_BYTES   = 37;
constexpr size_t PAYLOAD_BYTES = 160;

// Host-portable buffer type whose physical allocation carries a guard that the
// buffer does not report, like the Hexagon shared buffer. Its allocator is the
// oracle the registry procedure has to agree with.
struct fixture {
    ggml_backend_reg reg = {};
    ggml_backend_device dev = {};
    ggml_backend_buffer_type buft = {};
    std::vector<size_t> physical_sizes;

    static fixture & get(ggml_backend_buffer_type_t type) {
        return *static_cast<fixture *>(type->context);
    }
    static size_t physical(size_t size) {
        return size > SIZE_MAX - GUARD_BYTES ? SIZE_MAX : size + GUARD_BYTES;
    }
    static size_t predict(ggml_backend_buffer_type_t, size_t size) {
        return physical(size);
    }
    static void * proc(ggml_backend_reg_t, const char * name) {
        return std::strcmp(name, "ggml_backend_buft_get_buffer_alloc_size") == 0 ?
            reinterpret_cast<void *>(predict) : nullptr;
    }
    static const char * name(ggml_backend_buffer_type_t) { return "guard-probe"; }
    static size_t alignment(ggml_backend_buffer_type_t) { return 16; }
    static size_t maximum(ggml_backend_buffer_type_t) { return SIZE_MAX; }
    static bool is_host(ggml_backend_buffer_type_t) { return true; }
    static void free_buffer(ggml_backend_buffer_t buffer) {
        ggml_backend_buffer_free(static_cast<ggml_backend_buffer_t>(buffer->context));
    }
    static void * base(ggml_backend_buffer_t buffer) {
        return ggml_backend_buffer_get_base(static_cast<ggml_backend_buffer_t>(buffer->context));
    }
    static ggml_backend_buffer_t alloc(ggml_backend_buffer_type_t type, size_t size) {
        const size_t total = physical(size);
        get(type).physical_sizes.push_back(total);
        auto * backing = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), total);
        if (backing == nullptr) {
            return nullptr;
        }
        ggml_backend_buffer_i iface = {};
        iface.free_buffer = free_buffer;
        iface.get_base = base;
        return ggml_backend_buffer_init(type, iface, backing, size);
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
        buft.iface.is_host = is_host;
    }
};

void check_query_matches_allocator() {
    fixture f;
    const size_t predicted = ggml_backend_buft_get_buffer_alloc_size(&f.buft, PAYLOAD_BYTES);
    check(predicted == PAYLOAD_BYTES + GUARD_BYTES, "query missed the guard");
    check(f.physical_sizes.empty(), "query allocated a buffer");
    auto * buffer = ggml_backend_buft_alloc_buffer(&f.buft, PAYLOAD_BYTES);
    check(buffer != nullptr && f.physical_sizes.size() == 1 && f.physical_sizes[0] == predicted,
        "query differs from the physical allocation");
    check(ggml_backend_buffer_get_size(buffer) == PAYLOAD_BYTES, "buffer reports the guard as payload");
    ggml_backend_buffer_free(buffer);
}

void check_zero_and_overflow() {
    fixture f;
    check(ggml_backend_buft_get_buffer_alloc_size(&f.buft, 0) == 0, "zero-size query called the procedure");
    auto * empty = ggml_backend_buft_alloc_buffer(&f.buft, 0);
    check(empty != nullptr && ggml_backend_buffer_get_size(empty) == 0 && f.physical_sizes.empty(),
        "zero-size dummy allocated a payload");
    ggml_backend_buffer_free(empty);
    check(ggml_backend_buft_get_buffer_alloc_size(&f.buft, SIZE_MAX - 20) == SIZE_MAX, "guard addition wrapped");
}

void check_identity_without_procedure() {
    fixture f;
    f.reg.iface.get_proc_address = nullptr;
    check(ggml_backend_buft_get_buffer_alloc_size(&f.buft, PAYLOAD_BYTES) == PAYLOAD_BYTES,
        "absent procedure is not the identity");
    f.buft.device = nullptr;
    check(ggml_backend_buft_get_buffer_alloc_size(&f.buft, PAYLOAD_BYTES) == PAYLOAD_BYTES,
        "device-less buffer type is not the identity");
    check(ggml_backend_buft_get_buffer_alloc_size(ggml_backend_cpu_buffer_type(), PAYLOAD_BYTES) == PAYLOAD_BYTES,
        "CPU buffer size changed");
}

} // namespace

int main() try {
    ggml_backend_load_all();
    check_query_matches_allocator();
    check_zero_and_overflow();
    check_identity_without_procedure();
    std::puts("test-backend-buffer-size: all checks passed");
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
}
