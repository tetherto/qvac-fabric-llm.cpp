#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>

#ifndef GGML_TEST_REQUIRED_BACKENDS
#error "GGML_TEST_REQUIRED_BACKENDS must list the backends that register on any host"
#endif

static bool check(bool ok, const char * what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
    }
    return ok;
}

static bool required_backend_registered(const std::string & name) {
    const bool ok = ggml_backend_reg_by_name(name.c_str()) != nullptr;
    printf("%s backend %s\n", name.c_str(), ok ? "registered" : "missing");
    return check(ok, "every required backend registers through the configured library prefix");
}

static bool every_required_backend_registered() {
    std::istringstream names(GGML_TEST_REQUIRED_BACKENDS);
    std::string name;
    bool ok = true;
    while (std::getline(names, name, ',')) {
        ok = required_backend_registered(name) && ok;
    }
    return ok;
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

    bool ok = every_required_backend_registered();
    if (vulkan_disabled) {
        ok = vulkan_stays_unloaded() && ok;
        ok = opencl_still_loads() && ok;
    }
    return ok ? 0 : 1;
}
