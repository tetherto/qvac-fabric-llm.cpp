#include "tts-cpp/moss/fit.h"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace {
constexpr int PROMPT_ROWS = 128;
constexpr int REFERENCE_SAMPLES = 24000;
constexpr int SKIP = 77;

bool check_workloads(const std::filesystem::path & directory, const char * model, bool gpu) {
    tts_cpp::moss::EngineOptions options;
    options.backbone_path = (directory / model).string();
    options.decoder_path = (directory / "moss-codec-decoder-f16.gguf").string();
    options.encoder_path = (directory / "moss-codec-encoder-f16.gguf").string();
    options.use_gpu = gpu;
    bool ok = true;
    for (int mode = 0; mode < 3; ++mode) {
        const tts_cpp::moss::FitWorkload workload(
            PROMPT_ROWS, mode == 2 ? REFERENCE_SAMPLES : 0, mode == 1, 0);
        const auto result = tts_cpp::moss::fit_params(options, workload);
        std::printf("%s workload %d\n%s\n", model, mode, result.report.c_str());
        if (result.status == tts_cpp::FitStatus::Error ||
            result.device.total_bytes == 0 || (gpu && result.device_is_cpu)) ok = false;
    }
    return ok;
}
}

int main() {
    const char * directory = std::getenv("MOSS_FIT_MODEL_DIR");
    if (!directory || !*directory) return SKIP;
    const char * gpu_value = std::getenv("MOSS_FIT_GPU");
    const bool gpu = gpu_value && std::string(gpu_value) == "1";
    const bool tts = check_workloads(directory, "moss-tts-delay-f16.gguf", gpu);
    const bool ttsd = check_workloads(directory, "moss-ttsd-f16.gguf", gpu);
    return tts && ttsd ? 0 : 1;
}
