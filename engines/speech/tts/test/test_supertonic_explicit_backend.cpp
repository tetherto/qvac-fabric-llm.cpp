#include "supertonic_internal.h"

#include "backend_util.h"

#include <cstdio>

using namespace tts_cpp::supertonic::detail;

namespace {

constexpr const char * CPU_REQUEST     = "cpu";
constexpr const char * MISSING_REQUEST = "no-such-device";

int failures = 0;

void check(bool condition, const char * message) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
}

bool load_with_backend(const char * path, const char * backend, bool & on_cpu) {
    supertonic_model model;
    const bool loaded = load_supertonic_gguf(path, model, /*n_gpu_layers=*/0, /*verbose=*/false, /*f16_weights=*/-1,
                                             supertonic_precision::Auto, /*vulkan_device=*/0,
                                             /*f16_weights_deny_list=*/{}, backend);
    on_cpu = loaded && ::tts_cpp::detail::backend_is_cpu(model.backend);
    if (loaded) free_supertonic_model(model);
    return loaded;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]);
        return 2;
    }
    bool on_cpu = false;
    check(load_with_backend(argv[1], CPU_REQUEST, on_cpu), "an explicit cpu request must load");
    check(on_cpu, "an explicit cpu request must place the model on the CPU");
    check(!load_with_backend(argv[1], MISSING_REQUEST, on_cpu),
          "a request for a missing device must fail instead of falling back");
    return failures == 0 ? 0 : 1;
}
