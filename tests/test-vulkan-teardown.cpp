#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static constexpr size_t buffer_bytes = 1 << 20;
static constexpr const char * device_name = "Vulkan0";
static constexpr const char * default_mode = "";
static constexpr const char * keep_buffer_flag = "--keep-buffer";
static constexpr const char * free_after_teardown_flag = "--free-after-teardown";
static constexpr const char * no_device_notice = "no Vulkan device available; teardown test skipped";

struct retained_resources {
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t device_buffer = nullptr;
    ggml_backend_buffer_t host_buffer = nullptr;
    ggml_backend_event_t event = nullptr;
};

static retained_resources retained;

static void free_retained() {
    ggml_backend_event_free(retained.event);
    ggml_backend_buffer_free(retained.host_buffer);
    ggml_backend_buffer_free(retained.device_buffer);
    ggml_backend_free(retained.backend);
}

static ggml_backend_buffer_t alloc_from(ggml_backend_buffer_type_t type) {
    return ggml_backend_buft_alloc_buffer(type, buffer_bytes);
}

static ggml_backend_event_t record_event(ggml_backend_dev_t device, ggml_backend_t backend) {
    ggml_backend_event_t event = ggml_backend_event_new(device);
    if (event != nullptr) {
        ggml_backend_event_record(event, backend);
    }
    return event;
}

static bool retain_on(ggml_backend_dev_t device) {
    retained.backend = ggml_backend_dev_init(device, nullptr);
    if (retained.backend == nullptr) {
        std::fprintf(stderr, "Vulkan backend init failed\n");
        return false;
    }
    retained.device_buffer = alloc_from(ggml_backend_get_default_buffer_type(retained.backend));
    retained.host_buffer = alloc_from(ggml_backend_dev_host_buffer_type(device));
    retained.event = record_event(device, retained.backend);
    return retained.device_buffer != nullptr && retained.host_buffer != nullptr && retained.event != nullptr;
}

static bool run_on(ggml_backend_dev_t device, bool keep_buffer) {
    ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
    if (backend == nullptr) {
        std::fprintf(stderr, "Vulkan backend init failed\n");
        return false;
    }
    ggml_backend_buffer_t buffer = alloc_from(ggml_backend_get_default_buffer_type(backend));
    const bool allocated = buffer != nullptr;
    if (keep_buffer) {
        retained.device_buffer = buffer;
    } else {
        ggml_backend_buffer_free(buffer);
    }
    ggml_backend_free(backend);
    return allocated;
}

static bool is_mode(const char * mode, const char * flag) {
    return std::strcmp(mode, flag) == 0;
}

static bool frees_at_exit(const char * mode) {
    return is_mode(mode, keep_buffer_flag) || is_mode(mode, free_after_teardown_flag);
}

static bool run_mode(ggml_backend_dev_t device, const char * mode) {
    if (is_mode(mode, free_after_teardown_flag)) {
        return retain_on(device);
    }
    return run_on(device, is_mode(mode, keep_buffer_flag));
}

int main(int argc, char ** argv) {
    const char * mode = argc > 1 ? argv[1] : default_mode;
    if (frees_at_exit(mode)) {
        std::atexit(free_retained);
    }
    ggml_backend_load_all();
    ggml_backend_dev_t device = ggml_backend_dev_by_name(device_name);
    if (device == nullptr) {
        std::puts(no_device_notice);
        return 0;
    }
    return run_mode(device, mode) ? 0 : 1;
}
