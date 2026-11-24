#include "audiogen-cpp/acestep/engine.h"
#include "acestep/engine_backends.h"
#include "ggml-backend.h"
#include "parity_checks.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace {

constexpr int kThreads = 4;
constexpr int kSingleThread = 1;
constexpr int kSteps = 2;
constexpr int kReuses = 2;
constexpr int kSampleRate = 48000;
constexpr int kChannels = 2;
constexpr float kDuration = 2.0f;
constexpr long long kSeed = 42;
constexpr int kNoHardware = 77;
constexpr double kMaxCpuXdnaRelativeL2 = 0.1;

using namespace tts_cpp::acestep;

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

void check_placement() {
    BackendRequest request;
    request.backend = BACKEND_REQUEST_XDNA;
    request.n_threads = kThreads;
    AcestepBackends backends;
    require(resolve_acestep_backends(request, backends), "XDNA backend resolution failed");
    const bool matched = backend_name_is_xdna(backend_reg_name(backends.backend)) &&
                         backends.enc == backends.backend_cpu &&
                         backends.lm == backends.backend_cpu && backends.detok == backends.backend_cpu &&
                         vae_backend_request(request).backend == BACKEND_REQUEST_CPU;
    free_acestep_backends(backends);
    require(matched, "XDNA stage placement differs from the public contract");
}

EngineOptions options(const char * models, const char * backend) {
    EngineOptions value;
    value.models_dir = models;
    value.backend = backend;
    value.n_threads = kThreads;
    return value;
}

GenerateParams parameters() {
    GenerateParams value;
    value.caption = "Warm piano with a steady acoustic rhythm";
    value.duration = kDuration;
    value.inference_steps = kSteps;
    value.seed = kSeed;
    value.lm_phase1 = false;
    value.normalize_loudness = false;
    return value;
}

void check_audio(const GenerateResult & result) {
    require(result.sample_rate == kSampleRate && !result.pcm.empty() && result.pcm.size() % kChannels == 0,
            "Generation did not produce stereo 48 kHz audio");
    require(std::all_of(result.pcm.begin(), result.pcm.end(), [](float sample) {
        return std::isfinite(sample);
    }), "Generation produced non-finite audio");
}

GenerateResult generate(const EngineOptions & config) {
    auto engine = Engine::create(config);
    const auto result = engine->generate(parameters());
    check_audio(result);
    return result;
}

void check_reuse(const EngineOptions & config, const GenerateResult & expected) {
    auto engine = Engine::create(config);
    for (int reuse = 0; reuse < kReuses; ++reuse) {
        const auto actual = engine->generate(parameters());
        check_audio(actual);
        require(actual.pcm == expected.pcm, "Repeated XDNA generation changed its seeded output");
    }
}

void check_cpu_parity(const GenerateResult & expected, const GenerateResult & actual) {
    require(actual.pcm.size() == expected.pcm.size(), "CPU and XDNA output lengths differ");
    double error = 0, norm = 0;
    for (size_t index = 0; index < actual.pcm.size(); ++index) {
        const double delta = actual.pcm[index] - expected.pcm[index];
        error += delta * delta;
        norm += static_cast<double>(expected.pcm[index]) * expected.pcm[index];
    }
    require(norm > 0, "CPU fixture produced silent audio");
    const double relative_l2 = std::sqrt(error / norm);
    std::printf("Seeded CPU/XDNA PCM relative L2: %.8f (maximum %.8f)\n", relative_l2,
                kMaxCpuXdnaRelativeL2);
    require(audiogen_test::relative_error_passes(relative_l2, kMaxCpuXdnaRelativeL2),
            "XDNA generation exceeds the CPU parity tolerance");
}

void check_lyric_alignment(EngineOptions config) {
    config.n_threads = kSingleThread;
    auto engine = Engine::create(config);
    auto params = parameters();
    params.lyrics = "Shine a little light";
    params.generate_lrc = true;
    const auto result = engine->generate(params);
    check_audio(result);
    require(!result.metadata.lrc.empty() && std::isfinite(result.metadata.lyrics_score),
            "XDNA lyric alignment did not return timestamps");
}

}

int main(int argc, char ** argv) {
    if (argc > 2) {
        std::puts("Usage: test-acestep-xdna-generation models-directory");
        return 1;
    }
    const char * models = argc == 2 ? argv[1] : std::getenv("AUDIOGEN_TEST_MODELS_DIR");
    if (!models || !*models) return kNoHardware;
    ggml_backend_load_all();
    if (!ggml_backend_dev_by_name("XDNA0")) {
        std::puts("XDNA0 is unavailable");
        return kNoHardware;
    }
    try {
        check_placement();
        const auto cpu = generate(options(models, "cpu"));
        const auto xdna = generate(options(models, "xdna"));
        check_cpu_parity(cpu, xdna);
        check_reuse(options(models, "XDNA0"), xdna);
        check_lyric_alignment(options(models, "xdna"));
        std::puts("XDNA stage placement, generation, repeated calls and engine recreation passed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
