// QVAC: an unavailable CPU backend must fail fit measurement safely, then
// recover when backend initialization becomes available again. No inference.
#include "whisper.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__GLIBC__)
#include <malloc.h>

namespace {
bool inject_cpu_failure = false;
unsigned injected_failures = 0;
unsigned recovered_cpu_initializations = 0;

bool expect(bool condition, const char * message) {
    if (!condition) std::fprintf(stderr, "FAIL: %s\n", message);
    return condition;
}
}

extern "C" ggml_backend_t __real_ggml_backend_init_by_type(enum ggml_backend_dev_type type, const char * params);
extern "C" ggml_backend_t __wrap_ggml_backend_init_by_type(enum ggml_backend_dev_type type, const char * params) {
    if (type == GGML_BACKEND_DEVICE_TYPE_CPU) {
        if (inject_cpu_failure) {
            ++injected_failures;
            return nullptr;
        }
        ++recovered_cpu_initializations;
    }
    return __real_ggml_backend_init_by_type(type, params);
}

extern "C" void __real_free(void * pointer);
extern "C" void __wrap_free(void * pointer) {
    // glibc fills new allocations with the complement of M_PERTURB. Catch
    // the known uninitialized batch-pointer cleanup without dereferencing it.
    const uintptr_t poison = sizeof(uintptr_t) == 8 ? UINT64_C(0x5a5a5a5a5a5a5a5a) : UINT32_C(0x5a5a5a5a);
    if (injected_failures && reinterpret_cast<uintptr_t>(pointer) == poison) {
        std::fprintf(stderr, "FAIL: cleanup attempted to free an uninitialized batch pointer\n");
        std::fflush(stderr);
        std::_Exit(86);
    }
    __real_free(pointer);
}
#endif

int main(int argc, char ** argv) {
#if !defined(__GLIBC__)
    (void) argc;
    (void) argv;
    std::printf("SKIP: heap-poison fault injection requires glibc M_PERTURB\n");
    return 77;
#else
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <model>\n", argv[0]);
        return 2;
    }
    if (!expect(mallopt(M_PERTURB, 0xa5) == 1, "enable glibc heap perturbation")) return 1;
    ggml_backend_load_all();
    auto options = whisper_fit_default_options();
    options.model_path = argv[1];
    options.use_gpu = false;
    options.n_decoders = 1;
    options.audio_seconds = 1.0f;

    for (unsigned attempt = 1; attempt <= 2; ++attempt) {
        inject_cpu_failure = true;
        whisper_fit_result result = {};
        const int status = whisper_fit_params(&options, &result);
        if (!expect(injected_failures == attempt, "exactly one CPU failure injected per measurement") ||
            !expect(status == WHISPER_FIT_ERROR && result.status == WHISPER_FIT_ERROR,
                    "CPU initialization failure returns WHISPER_FIT_ERROR") ||
            !expect(!result.fits, "failed measurement cannot claim the model fits") ||
            !expect(std::strcmp(result.reason, "measurement-failed") == 0,
                    "CPU initialization failure reports measurement-failed")) return 1;
    }

    inject_cpu_failure = false;
    const unsigned before_recovery = recovered_cpu_initializations;
    whisper_fit_result recovered = {};
    const int status = whisper_fit_params(&options, &recovered);
    if (!expect(recovered_cpu_initializations > before_recovery, "recovery initializes the actual CPU backend") ||
        !expect(status == WHISPER_FIT_SUCCESS || status == WHISPER_FIT_FAILURE,
                "restored CPU backend completes measurement") ||
        !expect(status == recovered.status && recovered.fits == (status == WHISPER_FIT_SUCCESS),
                "recovery result and fit status agree") ||
        !expect(recovered.device_is_cpu, "recovery measures the CPU backend") ||
        !expect(recovered.device.total_bytes + recovered.host_bytes > 0, "recovery reports measured memory") ||
        !expect(std::strcmp(recovered.reason, "measurement-failed") != 0,
                "recovery does not retain measurement-failed")) return 1;
    std::printf("PASS: two CPU-init failures returned measurement-failed safely; restored backend measured memory\n");
    return 0;
#endif
}
