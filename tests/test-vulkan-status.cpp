#include "ggml-backend.h"

#include <cstdio>

extern "C" {
GGML_BACKEND_API bool ggml_backend_vk_test_sticky_status(void);
}

int main() {
    if (!ggml_backend_vk_test_sticky_status()) {
        std::fprintf(stderr, "Vulkan sticky-status short-circuit test failed\n");
        return 1;
    }

    return 0;
}
