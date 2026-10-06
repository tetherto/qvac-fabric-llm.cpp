#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>

#ifndef GGML_TEST_BACKEND_COUNT
#error "GGML_TEST_BACKEND_COUNT must be the number of built backends"
#endif

static bool check(bool ok, const char * what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
    }
    return ok;
}

static size_t expected_backend_count(bool vulkan_disabled) {
#ifdef GGML_TEST_HAS_VULKAN
    return vulkan_disabled ? GGML_TEST_BACKEND_COUNT - 1 : GGML_TEST_BACKEND_COUNT;
#else
    (void) vulkan_disabled;
    return GGML_TEST_BACKEND_COUNT;
#endif
}

static bool every_built_backend_registered(size_t expected) {
    const size_t registered = ggml_backend_reg_count();
    printf("registered %zu backends, expected at least %zu\n", registered, expected);
    return check(registered >= expected, "the CPU and RPC backends register through the configured library prefix");
}

static bool vulkan_stays_unloaded() {
    return check(ggml_backend_reg_by_name("Vulkan") == nullptr, "Vulkan must not register when GGML_DISABLE_VULKAN is set");
}

static bool opencl_still_loads() {
#ifdef GGML_TEST_HAS_OPENCL
    return check(ggml_backend_reg_by_name("OpenCL") != nullptr, "OpenCL must register although Vulkan is disabled");
#else
    return true;
#endif
}

int main() {
    const bool vulkan_disabled = getenv("GGML_DISABLE_VULKAN") != nullptr;

    ggml_backend_load_all();

    bool ok = every_built_backend_registered(expected_backend_count(vulkan_disabled));
    if (vulkan_disabled) {
        ok = vulkan_stays_unloaded() && ok;
        ok = opencl_still_loads() && ok;
    }
    return ok ? 0 : 1;
}
