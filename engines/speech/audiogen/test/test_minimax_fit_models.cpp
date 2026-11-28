#include "minimax_fit_parity.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

using tts_cpp::minimax::EngineOptions;
using tts_cpp::minimax::FitResult;
using tts_cpp::minimax::FitStatus;
using tts_cpp::minimax::FitWorkload;

constexpr int         kSkip          = 77;
constexpr int64_t     kFrames        = 300;
constexpr int64_t     kPrompt        = 19;
constexpr int         kThreads       = 4;
constexpr const char * kDefaultDevice = "cpu";

int failures = 0;

void expect_eq(uint64_t projected, uint64_t real, const char * what) {
    if (projected != real) {
        ++failures;
        std::fprintf(stderr, "FAIL %s: projected %llu != real %llu\n", what, (unsigned long long) projected,
                     (unsigned long long) real);
    }
}

uint64_t stage_value(const FitResult & fit, const char * name, uint64_t tts_cpp::minimax::FitStageProjection::*field) {
    const auto * stage = minimax_fit_parity::find_stage(fit, name);
    return stage ? stage->*field : 0;
}

void compare(const FitResult & fit, const minimax_fit_parity::RealBytes & real) {
    using Row = tts_cpp::minimax::FitStageProjection;
    expect_eq(minimax_fit_parity::projected_weights(fit), real.weights, "weights");
    expect_eq(stage_value(fit, "lm", &Row::state_bytes), real.lm_state, "lm KV cache");
    expect_eq(stage_value(fit, "depth", &Row::state_bytes), real.depth_state, "depth KV cache");
    expect_eq(stage_value(fit, "lm", &Row::compute_bytes), real.lm_compute, "lm compute");
    expect_eq(stage_value(fit, "depth", &Row::compute_bytes), real.depth_compute, "depth compute");
    expect_eq(stage_value(fit, "depth", &Row::host_bytes), real.depth_host, "depth host");
    expect_eq(stage_value(fit, "cond", &Row::compute_bytes), real.cond_compute, "cond compute");
    expect_eq(stage_value(fit, "cond", &Row::host_bytes), real.cond_host, "cond host");
    expect_eq(stage_value(fit, "dit", &Row::compute_bytes), real.dit_compute, "dit compute");
    expect_eq(stage_value(fit, "vocoder", &Row::compute_bytes), real.vocoder_compute, "vocoder compute");
}

EngineOptions fixture_options(const char * models_dir) {
    EngineOptions options;
    options.model_dir = models_dir;
    options.n_threads = kThreads;
    const char * device = std::getenv("MM3_DEVICE");
    options.device      = device && *device ? device : kDefaultDevice;
    if (const char * backends_dir = std::getenv("AUDIOGEN_TEST_BACKENDS_DIR")) {
        options.backends_dir = backends_dir;
    }
    return options;
}

}

int main() {
    const char * models_dir = std::getenv("AUDIOGEN_TEST_MINIMAX_MODELS_DIR");
    if (!models_dir || !*models_dir) {
        std::fprintf(stderr, "[test-minimax-fit-models] skipped: AUDIOGEN_TEST_MINIMAX_MODELS_DIR is unset\n");
        return kSkip;
    }
    const EngineOptions options = fixture_options(models_dir);
    FitWorkload         request;
    request.max_frames    = kFrames;
    request.prompt_tokens = kPrompt;
    const FitResult fit   = tts_cpp::minimax::fit_params(options, request);
    std::printf("%s", fit.report.c_str());
    if (fit.status == FitStatus::Error) {
        std::fprintf(stderr, "FAIL fit_params: %s\n", fit.reason.c_str());
        return 1;
    }
    minimax_fit_parity::RealBytes real;
    std::string                   error;
    if (!minimax_fit_parity::measure_real(options, kPrompt, kFrames, real, error)) {
        std::fprintf(stderr, "FAIL real allocation: %s\n", error.c_str());
        return 1;
    }
    compare(fit, real);
    std::printf("[test-minimax-fit-models] %s\n", failures == 0 ? "projection matches the runtime" : "mismatch");
    return failures == 0 ? 0 : 1;
}
