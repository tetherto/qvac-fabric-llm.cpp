#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-vulkan/ggml-vulkan-capacity.h"

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
constexpr size_t CHUNK_BYTES = 64;
constexpr size_t CAPACITY_BYTES = 256;

// Host-portable buffer type whose physical allocation carries a guard that the
// buffer does not report, like the Hexagon shared buffer. Its allocator is the
// oracle the registry procedure has to agree with.
struct fixture {
    ggml_backend_reg reg = {};
    ggml_backend_device dev = {};
    ggml_backend_buffer_type buft = {};
    std::vector<size_t> physical_sizes;
    ggml_backend backend = {};
    size_t chunk_bytes = SIZE_MAX;
    size_t capacity_bytes = SIZE_MAX;

    static fixture & get(ggml_backend_buffer_type_t type) {
        return *static_cast<fixture *>(type->context);
    }
    static size_t physical(size_t size) {
        return size > SIZE_MAX - GUARD_BYTES ? SIZE_MAX : size + GUARD_BYTES;
    }
    static size_t predict(ggml_backend_buffer_type_t, size_t size) {
        return physical(size);
    }
    static size_t capacity(ggml_backend_buffer_type_t type) {
        return get(type).capacity_bytes;
    }
    static ggml_backend_buffer_type_t buffer_type(ggml_backend_dev_t device) {
        return &static_cast<fixture *>(device->context)->buft;
    }
    static void * proc(ggml_backend_reg_t, const char * name) {
        if (std::strcmp(name, "ggml_backend_get_buffer_capacity") == 0) {
            return reinterpret_cast<void *>(capacity);
        }
        return std::strcmp(name, "ggml_backend_buft_get_buffer_alloc_size") == 0 ?
            reinterpret_cast<void *>(predict) : nullptr;
    }
    static const char * name(ggml_backend_buffer_type_t) { return "guard-probe"; }
    static size_t alignment(ggml_backend_buffer_type_t) { return 16; }
    static size_t maximum(ggml_backend_buffer_type_t type) { return get(type).chunk_bytes; }
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
        dev.context = this;
        dev.iface.get_buffer_type = buffer_type;
        backend.device = &dev;
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

void check_capacity_extension() {
    fixture f;
    f.chunk_bytes = CHUNK_BYTES;
    f.capacity_bytes = CAPACITY_BYTES;
    check(ggml_backend_buft_get_max_capacity(&f.buft) == CAPACITY_BYTES, "logical capacity query");
    check(ggml_backend_get_max_buffer_capacity(&f.backend) == CAPACITY_BYTES, "backend capacity query");
    check(ggml_backend_buft_get_max_size(&f.buft) == CHUNK_BYTES, "capacity query changed chunk limit");
    check(ggml_backend_get_max_size(&f.backend) == CHUNK_BYTES, "backend chunk limit changed");
    check(ggml_backend_is_buffer_size_supported(&f.backend, CHUNK_BYTES + 1), "capacity capped by chunk size");
    check(ggml_backend_buft_is_size_supported(&f.buft, CAPACITY_BYTES), "capacity boundary rejected");
    check(ggml_backend_is_buffer_size_supported(&f.backend, CAPACITY_BYTES), "backend capacity boundary rejected");
    check(!ggml_backend_buft_is_size_supported(&f.buft, CAPACITY_BYTES + 1), "capacity overflow accepted");
    check(!ggml_backend_is_buffer_size_supported(&f.backend, SIZE_MAX), "oversize request accepted");
    check(f.physical_sizes.empty(), "capacity preflight allocated a buffer");

    fixture other;
    other.capacity_bytes = CHUNK_BYTES;
    other.buft.device = &f.dev;
    check(ggml_backend_buft_get_max_capacity(&other.buft) == CHUNK_BYTES, "extension received wrong buffer type");
}

void check_capacity_boundaries() {
    fixture f;
    f.chunk_bytes = CHUNK_BYTES;
    f.capacity_bytes = 0;
    check(ggml_backend_buft_is_size_supported(&f.buft, 0), "zero capacity rejected empty buffer");
    check(ggml_backend_is_buffer_size_supported(&f.backend, 0), "backend rejected empty buffer");
    check(!ggml_backend_buft_is_size_supported(&f.buft, 1), "zero capacity ignored");
    check(!ggml_backend_is_buffer_size_supported(&f.backend, 1), "backend ignored zero capacity");
    f.capacity_bytes = SIZE_MAX;
    check(ggml_backend_buft_is_size_supported(&f.buft, SIZE_MAX), "unspecified capacity rejected");
    check(ggml_backend_is_buffer_size_supported(&f.backend, SIZE_MAX), "backend unspecified capacity rejected");
    check(f.physical_sizes.empty(), "boundary preflight allocated a buffer");
}

void * missing_capacity_proc(ggml_backend_reg_t, const char *) {
    return nullptr;
}

void check_capacity_fallback() {
    fixture f;
    f.chunk_bytes = CHUNK_BYTES;
    f.reg.iface.get_proc_address = missing_capacity_proc;
    check(ggml_backend_buft_get_max_capacity(&f.buft) == CHUNK_BYTES, "missing extension fallback");
    check(ggml_backend_is_buffer_size_supported(&f.backend, CHUNK_BYTES), "fallback boundary rejected");
    check(!ggml_backend_is_buffer_size_supported(&f.backend, CHUNK_BYTES + 1), "fallback overflow accepted");
    f.reg.iface.get_proc_address = nullptr;
    check(ggml_backend_get_max_buffer_capacity(&f.backend) == CHUNK_BYTES, "missing procedure lookup fallback");
    f.dev.reg = nullptr;
    check(ggml_backend_buft_get_max_capacity(&f.buft) == CHUNK_BYTES, "missing registry fallback");
    f.buft.device = nullptr;
    check(ggml_backend_buft_get_max_capacity(&f.buft) == CHUNK_BYTES, "device-less buffer fallback");
    f.buft.iface.get_max_size = nullptr;
    check(ggml_backend_buft_get_max_capacity(&f.buft) == SIZE_MAX, "unspecified chunk limit fallback");
    check(ggml_backend_buft_is_size_supported(&f.buft, SIZE_MAX), "unspecified fallback preflight");
    check(f.physical_sizes.empty(), "fallback preflight allocated a buffer");
    auto * cpu = ggml_backend_cpu_buffer_type();
    check(ggml_backend_buft_get_max_capacity(cpu) == ggml_backend_buft_get_max_size(cpu), "CPU limit changed");
}

void check_vulkan_capacity_limits() {
    constexpr uint64_t GIB = uint64_t{1} << 30;
    const size_t expected = static_cast<size_t>(std::min(uint64_t{3} * GIB, uint64_t{SIZE_MAX}));
    check(ggml_vk_buffer_capacity(4 * GIB, 3 * GIB, GIB, false, false) == expected,
          "Vulkan allocation limit confused with descriptor range");
    check(ggml_vk_buffer_capacity(3 * GIB, 4 * GIB, GIB, true, true) == expected,
          "Vulkan buffer limit ignored");
    check(ggml_vk_buffer_capacity(4 * GIB, 3 * GIB, GIB, true, false) == expected,
          "Vulkan indexing feature changed allocation capacity");
    check(ggml_vk_buffer_capacity(4 * GIB, 3 * GIB, GIB, false, true) == expected,
          "Vulkan device address feature changed allocation capacity");
    check(ggml_vk_buffer_capacity(0, 4 * GIB, GIB, true, true) == 0, "Vulkan zero buffer limit");
    check(ggml_vk_buffer_capacity(4 * GIB, 0, GIB, true, true) == 0, "Vulkan zero allocation limit");
    check(ggml_vk_buffer_capacity(UINT64_MAX, UINT64_MAX, GIB, false, false) == SIZE_MAX,
          "Vulkan capacity exceeds size_t");
}

void check_vulkan_device_capacity(ggml_backend_dev_t device) {
    auto * buft = ggml_backend_dev_buffer_type(device);
    const size_t capacity = ggml_backend_buft_get_max_capacity(buft);
    check(ggml_backend_buft_is_size_supported(buft, capacity), "Vulkan capacity boundary rejected");
    if (capacity < SIZE_MAX) {
        check(!ggml_backend_buft_is_size_supported(buft, capacity + 1), "Vulkan capacity overflow accepted");
        check(ggml_backend_buft_alloc_buffer(buft, capacity + 1) == nullptr, "Vulkan oversized allocation accepted");
    }
    auto * host = ggml_backend_dev_host_buffer_type(device);
    if (host != nullptr) {
        check(ggml_backend_buft_get_max_capacity(host) == SIZE_MAX, "Vulkan host fallback has a finite capacity");
    }
}

void check_vulkan_capacity_provider() {
    auto * reg = ggml_backend_reg_by_name("Vulkan");
    if (reg == nullptr) {
        return;
    }
    check(ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_buffer_capacity") != nullptr,
          "Vulkan capacity provider missing");
    for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); ++i) {
        check_vulkan_device_capacity(ggml_backend_reg_dev_get(reg, i));
    }
}

} // namespace

int main() try {
    ggml_backend_load_all();
    check_query_matches_allocator();
    check_zero_and_overflow();
    check_identity_without_procedure();
    check_capacity_extension();
    check_capacity_boundaries();
    check_capacity_fallback();
    check_vulkan_capacity_limits();
    check_vulkan_capacity_provider();
    std::puts("test-backend-buffer-size: all checks passed");
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
}
