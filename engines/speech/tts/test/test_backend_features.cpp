// Registry feature lookup (backend_util.h): lists what the CPU registry
// advertises and, when TTS_CPP_EXPECT_LLAMAFILE is set, asserts that the
// tinyBLAS detection agrees with how ggml was built. Model-free.

#include "backend_util.h"

#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

int main() {
    ggml_backend_load_all();
    ggml_backend_reg_t cpu = ggml_backend_reg_by_name("CPU");
    if (!cpu) {
        std::fprintf(stderr, "[backend-features] no CPU registry\n");
        return 1;
    }
    auto get_features = (ggml_backend_get_features_t)
        ggml_backend_reg_get_proc_address(cpu, "ggml_backend_get_features");
    if (!get_features) {
        std::fprintf(stderr, "[backend-features] CPU registry exposes no ggml_backend_get_features\n");
        return 1;
    }
    std::printf("[backend-features] CPU:");
    for (const ggml_backend_feature * f = get_features(cpu); f && f->name; ++f) {
        std::printf(" %s=%s", f->name, f->value);
    }
    std::printf("\n");

    const bool llamafile = tts_cpp::detail::cpu_backend_has_feature("LLAMAFILE");
    if (llamafile == tts_cpp::detail::cpu_matmul_is_shape_exact()) {
        std::fprintf(stderr, "[backend-features] cpu_matmul_is_shape_exact disagrees with LLAMAFILE\n");
        return 1;
    }
    std::printf("[backend-features] LLAMAFILE=%d cpu_matmul_is_shape_exact=%d\n",
                (int) llamafile, (int) !llamafile);

    if (const char * expect = std::getenv("TTS_CPP_EXPECT_LLAMAFILE"); expect && *expect) {
        const bool want = std::strcmp(expect, "1") == 0 || std::strcmp(expect, "ON") == 0 ||
                          std::strcmp(expect, "on") == 0 || std::strcmp(expect, "TRUE") == 0;
        if (llamafile != want) {
            std::fprintf(stderr, "[backend-features] FAIL: LLAMAFILE detected=%d, build says %s\n",
                         (int) llamafile, expect);
            return 1;
        }
        std::printf("[backend-features] matches the build (TTS_CPP_EXPECT_LLAMAFILE=%s)\n", expect);
    }
    std::printf("[backend-features] PASS\n");
    return 0;
}
