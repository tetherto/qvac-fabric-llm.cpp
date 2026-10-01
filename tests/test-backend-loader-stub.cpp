// Stub backend module for test-backend-loader. Built once per case, with STUB_NAME
// and STUB_SCORE set per copy. It registers no devices and appends one line to the
// file named by GGML_LOADER_TEST_LOG each time the loader scores it. With
// STUB_SCORE_FROM_DEP the score comes from an imported DLL, so the module only
// loads when that DLL is found.

#include "ggml-backend-impl.h"

#include <cstdio>
#include <cstdlib>

#ifdef _WIN32
#    define STUB_EXPORT extern "C" __declspec(dllexport)
#else
#    define STUB_EXPORT extern "C" __attribute__((visibility("default")))
#endif

#ifdef STUB_SCORE_FROM_DEP
extern "C" __declspec(dllimport) int loader_test_dep_score(void);
#    define STUB_SCORE loader_test_dep_score()
#endif

static const char * stub_get_name(ggml_backend_reg_t) {
    return STUB_NAME;
}

static size_t stub_get_device_count(ggml_backend_reg_t) {
    return 0;
}

static ggml_backend_dev_t stub_get_device(ggml_backend_reg_t, size_t) {
    return nullptr;
}

static void * stub_get_proc_address(ggml_backend_reg_t, const char *) {
    return nullptr;
}

STUB_EXPORT ggml_backend_reg_t ggml_backend_init(void) {
    static ggml_backend_reg reg = {};
    reg.api_version             = GGML_BACKEND_API_VERSION;
    reg.iface.get_name          = stub_get_name;
    reg.iface.get_device_count  = stub_get_device_count;
    reg.iface.get_device        = stub_get_device;
    reg.iface.get_proc_address  = stub_get_proc_address;
    return &reg;
}

STUB_EXPORT int ggml_backend_score(void) {
    const char * log_path = getenv("GGML_LOADER_TEST_LOG");
    if (log_path != nullptr) {
        FILE * f = fopen(log_path, "a");
        if (f != nullptr) {
            fprintf(f, "%s\n", STUB_NAME);
            fclose(f);
        }
    }
    return STUB_SCORE;
}
