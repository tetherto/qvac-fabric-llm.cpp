#include "ggml-backend.h"

#include <cstdio>
#include <sstream>
#include <string>

#ifndef GGML_TEST_REQUIRED_BACKENDS
#    error "GGML_TEST_REQUIRED_BACKENDS must list the backends that register on any host"
#endif

static bool required_backend_registered(const std::string & name) {
    const bool ok = ggml_backend_reg_by_name(name.c_str()) != nullptr;
    printf("%s backend %s\n", name.c_str(), ok ? "registered" : "missing");
    if (!ok) {
        fprintf(stderr, "FAIL: %s must register through the configured library prefix\n", name.c_str());
    }
    return ok;
}

static bool every_required_backend_registered() {
    std::istringstream names(GGML_TEST_REQUIRED_BACKENDS);
    std::string        name;
    bool               ok = true;
    while (std::getline(names, name, ',')) {
        ok = required_backend_registered(name) && ok;
    }
    return ok;
}

int main() {
    ggml_backend_load_all();
    return every_required_backend_registered() ? 0 : 1;
}
