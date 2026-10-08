#include "ggml.h"
#include "ggml-backend.h"

#ifdef GGML_CONSUMER_EXPECT_CPU
#include "ggml-cpu.h"
#endif

#ifdef GGML_CONSUMER_HAS_VECTOR_INDEX
#include "ggml-vector-index.h"
#endif

#include <cstdio>

static bool check_core() {
    ggml_init_params params = { 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "FAIL: ggml_init returned null\n");
        return false;
    }
    ggml_free(ctx);
    return true;
}

static bool check_static_cpu() {
#ifdef GGML_CONSUMER_EXPECT_CPU
    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (cpu == nullptr) {
        std::fprintf(stderr, "FAIL: ggml_backend_cpu_init returned null\n");
        return false;
    }
    ggml_backend_free(cpu);
#endif
    return true;
}

static bool check_vector_index() {
#ifdef GGML_CONSUMER_HAS_VECTOR_INDEX
    ggml_vec_index_t * idx = ggml_vec_index_create(16, 32);
    if (idx == nullptr) {
        std::fprintf(stderr, "FAIL: ggml_vec_index_create returned null\n");
        return false;
    }
    ggml_vec_index_free(idx);
#endif
    return true;
}

int main() {
    if (!check_core() || !check_static_cpu() || !check_vector_index()) {
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
