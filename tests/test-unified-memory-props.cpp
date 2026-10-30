// Asserts that backends whose device memory shares the host's physical pool
// report it through ggml_backend_dev_props.memory_unified, and that the
// public wrapper zero-initializes the field for backends that do not set it.
//
// Regression fixture for the fit host+shared-memory fold: a name-based check
// in common/fit.cpp once matched a registry name that did not exist, so the
// fold was dead code and no fixture could tell "fold works" from "fold never
// runs" (qvac-fabric-llm.cpp#214 review).

#include "ggml-backend.h"

#include <cstdio>
#include <cstring>

int main() {
    ggml_backend_load_all();

    const size_t n_dev = ggml_backend_dev_count();
    bool saw_metal = false;

    for (size_t i = 0; i < n_dev; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);

        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);

        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        const char * reg_name = reg ? ggml_backend_reg_name(reg) : "";

        printf("device %zu: reg=%s name=%s type=%d memory_unified=%d\n",
            i, reg_name, props.name, (int) props.type, (int) props.memory_unified);

        if (strcmp(reg_name, "MTL") == 0) {
            saw_metal = true;
#if defined(__APPLE__) && defined(__aarch64__)
            // Every Apple silicon Metal device is unified with system memory.
            if (!props.memory_unified) {
                fprintf(stderr, "FAIL: Metal device does not report memory_unified on Apple silicon\n");
                return 1;
            }
#endif
        }

        // The CPU device is host memory by definition but is not a shared
        // *device* pool; it must not claim the flag.
        if (props.type == GGML_BACKEND_DEVICE_TYPE_CPU && props.memory_unified) {
            fprintf(stderr, "FAIL: CPU device claims memory_unified\n");
            return 1;
        }
    }

    // LLAMA_TEST_EXPECT_METAL is set only when GGML_METAL is ON. Apple silicon
    // CI also builds Vulkan/WebGPU with Metal off, where no Metal device exists.
#if defined(__APPLE__) && defined(__aarch64__) && defined(LLAMA_TEST_EXPECT_METAL)
    if (!saw_metal) {
        fprintf(stderr, "FAIL: no Metal device enumerated on Apple silicon\n");
        return 1;
    }
#else
    (void) saw_metal;
#endif

    printf("OK\n");
    return 0;
}
