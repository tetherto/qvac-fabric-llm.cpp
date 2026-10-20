#include "ggml.h"
#include "ggml-backend.h"

#ifndef TEST_WEBGPU_DL
#include "ggml-webgpu.h"
#endif

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static std::atomic<int> disabled_warnings{0};
static std::atomic<int> device_initializations{0};
static std::atomic<bool> null_adapter{false};

static void log_callback(ggml_log_level level, const char * text, void *) {
    if (std::strstr(text, "ggml_webgpu: adapter_info:") != nullptr) {
        device_initializations.fetch_add(1);
        if (std::strstr(text, "name: Null backend") != nullptr) {
            null_adapter.store(true);
        }
    }
    if (level == GGML_LOG_LEVEL_WARN &&
        std::strstr(text, "ggml_webgpu:") != nullptr &&
        std::strstr(text, "backend disabled") != nullptr) {
        disabled_warnings.fetch_add(1);
    }
    std::fputs(text, stderr);
}

int main(int argc, char ** argv) {
#ifdef TEST_WEBGPU_DL
    const bool no_autoload = argc == 4 && std::strcmp(argv[3], "--no-autoload") == 0;
    if (argc != 3 && !no_autoload) {
        std::fprintf(stderr, "usage: %s <backend-directory> <webgpu-module> [--no-autoload]\n", argv[0]);
        return 2;
    }
#else
    if (argc != 1) {
        return 2;
    }
    (void) argv;
#endif

    ggml_log_set(log_callback, nullptr);

#ifdef TEST_WEBGPU_DL
    ggml_backend_load_all_from_path(argv[1]);
    if (no_autoload) {
        if (ggml_backend_reg_by_name("WebGPU") != nullptr ||
            device_initializations.load() != 0 || disabled_warnings.load() != 0) {
            std::fprintf(stderr, "FAIL: WebGPU loaded without opt-in\n");
            return 1;
        }
        std::printf("OK: WebGPU autoload is disabled\n");
        return 0;
    }
#else
    // Keep resources alive across global registry setup.
    ggml_backend_t initial_backend = ggml_backend_webgpu_init();
    ggml_backend_buffer_t initial_buffer = initial_backend != nullptr && !null_adapter.load()
        ? ggml_backend_alloc_buffer(initial_backend, 256) : nullptr;
    if (initial_backend != nullptr && !null_adapter.load() && initial_buffer == nullptr) {
        ggml_backend_free(initial_backend);
        return 1;
    }
    ggml_backend_load_all();
#endif

    ggml_backend_reg_t reg = ggml_backend_reg_by_name("WebGPU");
    if (reg == nullptr) {
        std::fprintf(stderr, "FAIL: WebGPU registry was not discovered\n");
        return 1;
    }

    const size_t count = ggml_backend_reg_dev_count(reg);
    const size_t total = ggml_backend_dev_count();

    for (int i = 0; i < 2; ++i) {
#ifdef TEST_WEBGPU_DL
        ggml_backend_reg_t repeated = ggml_backend_load(argv[2]);
#else
        ggml_backend_reg_t repeated = ggml_backend_webgpu_reg();
#endif
        if (repeated != reg ||
            ggml_backend_reg_dev_count(reg) != count ||
            ggml_backend_dev_count() != total) {
            std::fprintf(stderr, "FAIL: repeated registration changed registry state\n");
            return 1;
        }
    }

    if (count > 1 || disabled_warnings.load() != (count == 0 ? 1 : 0)) {
        std::fprintf(stderr, "FAIL: unexpected device count or disabled diagnostics\n");
        return 1;
    }

    if (count == 1 && !null_adapter.load()) {
        ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        ggml_backend_buffer_t buffer = backend ? ggml_backend_alloc_buffer(backend, 256) : nullptr;
        if (buffer == nullptr) {
            ggml_backend_free(backend);
            std::fprintf(stderr, "FAIL: WebGPU buffer allocation failed\n");
            return 1;
        }
        ggml_backend_buffer_clear(buffer, 0);
        if (ggml_backend_reg_dev_get(reg, 0) != dev || device_initializations.load() != 1) {
            std::fprintf(stderr, "FAIL: WebGPU device was recreated\n");
            return 1;
        }
        ggml_backend_buffer_clear(buffer, 0x5a);
        ggml_backend_synchronize(backend);
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
    }
#ifndef TEST_WEBGPU_DL
    if (initial_buffer != nullptr) {
        ggml_backend_buffer_clear(initial_buffer, 0x5a);
        ggml_backend_synchronize(initial_backend);
        ggml_backend_buffer_free(initial_buffer);
    }
    ggml_backend_free(initial_backend);
#endif

    const char * disabled = std::getenv("TEST_WEBGPU_EXPECT_DISABLED");
    if (disabled != nullptr && std::strcmp(disabled, "1") == 0) {
        if (count != 0) {
            std::fprintf(stderr, "FAIL: expected WebGPU to be disabled\n");
            return 1;
        }
        std::printf("OK: WebGPU is disabled with a diagnostic\n");
        return 0;
    }

    if (count == 0 || null_adapter.load()) {
        const char * required = std::getenv("TEST_WEBGPU_REQUIRE_DEVICE");
        std::fprintf(stderr, "No usable WebGPU device%s\n", null_adapter.load() ? " (Dawn Null backend)" : "");
        return required != nullptr && std::strcmp(required, "1") == 0 ? 1 : 77;
    }

    std::printf("OK: WebGPU registration is stable\n");
    return 0;
}
