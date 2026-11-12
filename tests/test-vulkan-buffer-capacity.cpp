#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-vulkan/ggml-vulkan-capacity.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace {

constexpr size_t MIB = size_t{1} << 20;
constexpr size_t DEVICE_CAPACITY = 4 * MIB;
constexpr size_t CHUNK_SIZE = MIB;

void check(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void set_test_limit(const char * name, size_t value) {
    const std::string text = std::to_string(value);
#ifdef _WIN32
    const int result = _putenv_s(name, text.c_str());
#else
    const int result = setenv(name, text.c_str(), 1);
#endif
    check(result == 0, "could not configure Vulkan test limits");
}

void configure_test_limits() {
    constexpr size_t BUFFER_LIMIT = 2 * DEVICE_CAPACITY;
    set_test_limit("GGML_VK_FORCE_MAX_ALLOCATION_SIZE", DEVICE_CAPACITY);
    set_test_limit("GGML_VK_FORCE_MAX_BUFFER_SIZE", BUFFER_LIMIT);
    set_test_limit("GGML_VK_SUBALLOCATION_BLOCK_SIZE", CHUNK_SIZE);
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

struct warning_capture {
    std::string text;

    static void record(enum ggml_log_level level, const char * text, void * user_data) {
        if (level == GGML_LOG_LEVEL_WARN) {
            static_cast<warning_capture *>(user_data)->text += text;
        }
    }
};

void check_allocation(ggml_backend_buffer_type_t buft) {
    auto * buffer = ggml_backend_buft_alloc_buffer(buft, 2 * CHUNK_SIZE);
    check(buffer != nullptr, "allocation above chunk size failed");
    ggml_backend_buffer_free(buffer);

    warning_capture warning;
    ggml_log_set(warning_capture::record, &warning);
    buffer = ggml_backend_buft_alloc_buffer(buft, DEVICE_CAPACITY + 1);
    ggml_log_set(nullptr, nullptr);
    const bool rejected = buffer == nullptr;
    ggml_backend_buffer_free(buffer);
    check(rejected, "oversized Vulkan allocation accepted");
    check(warning.text.find(std::to_string(DEVICE_CAPACITY + 1)) != std::string::npos,
          "allocation warning omitted requested size");
    check(warning.text.find(std::to_string(DEVICE_CAPACITY)) != std::string::npos,
          "allocation warning omitted capacity");
}

void check_operation_capacity(ggml_backend_dev_t device) {
    constexpr size_t TENSOR_COUNT = 8;
    ggml_init_params params = {TENSOR_COUNT * ggml_tensor_overhead(), nullptr, true};
    auto * ctx = ggml_init(params);
    check(ctx != nullptr, "tensor context allocation failed");
    constexpr int64_t ELEMENTS = DEVICE_CAPACITY / sizeof(float);
    auto * boundary = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ELEMENTS);
    auto * oversized = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ELEMENTS + 1);
    auto * small = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    const bool boundary_supported = ggml_backend_dev_supports_op(device, ggml_relu(ctx, boundary));
    const bool input_supported = ggml_backend_dev_supports_op(device, ggml_relu(ctx, oversized));
    const bool output_supported = ggml_backend_dev_supports_op(device, ggml_pad(ctx, small, ELEMENTS, 0, 0, 0));
    ggml_free(ctx);
    check(boundary_supported, "Vulkan rejected an operation at capacity");
    check(!input_supported, "Vulkan scheduled an input above allocation capacity");
    check(!output_supported, "Vulkan scheduled an output above allocation capacity");
}

void check_host_capacity(ggml_backend_dev_t device) {
    auto * host = ggml_backend_dev_host_buffer_type(device);
    if (host == nullptr) {
        return;
    }
    check(ggml_backend_buft_get_max_capacity(host) == SIZE_MAX, "Vulkan host capacity is finite");
    auto host_with_context = *host;
    int context = 0;
    host_with_context.context = &context;
    check(ggml_backend_buft_get_max_capacity(&host_with_context) == SIZE_MAX,
          "Vulkan host identification depends on null context");
}

void check_vulkan_device(ggml_backend_dev_t device) {
    auto * buft = ggml_backend_dev_buffer_type(device);
    check(ggml_backend_buft_get_max_size(buft) == CHUNK_SIZE, "unexpected Vulkan chunk size");
    check(ggml_backend_buft_get_max_capacity(buft) == DEVICE_CAPACITY, "unexpected Vulkan capacity");
    check_allocation(buft);
    check_operation_capacity(device);
    check_host_capacity(device);
}

void check_vulkan_devices(ggml_backend_reg_t reg) {
    for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); ++i) {
        check_vulkan_device(ggml_backend_reg_dev_get(reg, i));
    }
}

void check_vulkan_provider() {
    ggml_backend_load_all();
    auto * reg = ggml_backend_reg_by_name("Vulkan");
    if (reg == nullptr || ggml_backend_reg_dev_count(reg) == 0) {
        std::puts("Vulkan runtime unavailable; portable capacity tests only");
        return;
    }
    check(ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_buffer_capacity") != nullptr,
          "Vulkan capacity provider missing");
    check_vulkan_devices(reg);
}

}

int main() try {
    configure_test_limits();
    check_vulkan_capacity_limits();
    check_vulkan_provider();
    std::puts("test-vulkan-buffer-capacity: all checks passed");
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
}
