// Per-call Core ML routing of Nemotron 3 Diarization through the public Engine
// API. DiarizationResult::encoder_used_coreml is true only when every network
// pass of a call ran on a diarizer sidecar: offline inputs, the long-form
// blocks, and calls above the staged capacities, which run on ggml. Stages the
// GGUF beside subsets of its bundles in temporary directories; exits 77 when
// the smallest bundle is missing.

#include "coreml_test_options.h"
#include "parakeet/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#include <unistd.h>

namespace {

namespace fs = std::filesystem;

constexpr int kSkip = 77;
constexpr int kSampleRate = 16000;
constexpr int kGpuLayers = 99;
constexpr int kSmallestCapacity = 358;
constexpr int kMelFramesPerSecond = 100;
constexpr int kSubsampling = 8;
constexpr int kLongFormBlockFrames = 745;
constexpr int kOfflineSeconds = 20;
constexpr int kAboveSmallestSeconds = 40;
constexpr int kLongFormSeconds = 100;
constexpr double kToneSeconds = 1.5;
constexpr double kLowHz = 220.0;
constexpr double kHighHz = 330.0;
constexpr double kAmplitude = 0.25;
constexpr double kPi = 3.14159265358979323846;
constexpr const char * kDiarizerInfix = "-diarizer-";
constexpr const char * kSidecarExtension = ".mlmodelc";
constexpr const char * kQuantTags[] = {
    "f32", "f16", "bf16", "q8_0", "q5_0", "q4_0", "q6_k", "q5_k", "q4_k", "mxfp4", "nvfp4",
};

struct Case {
    const char * name;
    int seconds;
    bool expects_coreml;
};

struct TemporaryDirectory {
    fs::path path;
    ~TemporaryDirectory() {
        std::error_code error;
        if (!path.empty()) fs::remove_all(path, error);
    }
};

std::string sidecar_stem(const fs::path & gguf) {
    const std::string stem = gguf.stem().string();
    const size_t separator = stem.find_last_of(".-");
    if (separator == std::string::npos) return stem;
    const std::string tag = stem.substr(separator + 1);
    for (const char * known : kQuantTags) {
        if (tag == known) return stem.substr(0, separator);
    }
    return stem;
}

int encoder_frames(int seconds) {
    return (1 + seconds * kMelFramesPerSecond + kSubsampling - 1) / kSubsampling;
}

fs::path bundle_path(const fs::path & gguf, int capacity) {
    return gguf.parent_path() /
        (sidecar_stem(gguf) + kDiarizerInfix + std::to_string(capacity) + kSidecarExtension);
}

int largest_staged_capacity(const fs::path & gguf) {
    const std::string prefix = sidecar_stem(gguf) + kDiarizerInfix;
    const std::string extension = kSidecarExtension;
    int largest = 0;
    std::error_code error;
    for (const auto & entry : fs::directory_iterator(gguf.parent_path(), error)) {
        const std::string name = entry.path().filename().string();
        if (name.size() <= prefix.size() + extension.size() ||
            name.compare(0, prefix.size(), prefix) != 0 ||
            name.compare(name.size() - extension.size(), extension.size(), extension) != 0) {
            continue;
        }
        const std::string digits =
            name.substr(prefix.size(), name.size() - prefix.size() - extension.size());
        if (digits.find_first_not_of("0123456789") != std::string::npos) continue;
        largest = std::max(largest, std::atoi(digits.c_str()));
    }
    return largest;
}

bool stage(const fs::path & gguf, const std::vector<int> & capacities,
           TemporaryDirectory & directory, fs::path & staged_gguf) {
    std::string pattern = (fs::temp_directory_path() / "parakeet-diarize-routing-XXXXXX").string();
    if (!mkdtemp(pattern.data())) return false;
    directory.path = pattern;
    staged_gguf = directory.path / gguf.filename();
    std::error_code error;
    fs::create_symlink(fs::absolute(gguf), staged_gguf, error);
    for (int capacity : capacities) {
        if (error) break;
        fs::create_directory_symlink(fs::absolute(bundle_path(gguf, capacity)),
            bundle_path(staged_gguf, capacity), error);
    }
    return !error;
}

std::vector<float> tones(int seconds) {
    std::vector<float> samples(static_cast<size_t>(seconds) * kSampleRate);
    for (size_t index = 0; index < samples.size(); ++index) {
        const double time = static_cast<double>(index) / kSampleRate;
        const bool low = static_cast<long long>(time / kToneSeconds) % 2 == 0;
        samples[index] = static_cast<float>(
            kAmplitude * std::sin(2.0 * kPi * (low ? kLowHz : kHighHz) * time));
    }
    return samples;
}

bool run(const fs::path & gguf, const char * staging, bool expects_sidecar,
         const std::vector<Case> & cases) {
    parakeet::EngineOptions options;
    options.coreml = coreml_ready_options();
    options.model_gguf_path = gguf.string();
    options.n_gpu_layers = kGpuLayers;
    parakeet::Engine engine(options);
    bool passed = engine.encoder_on_coreml() == expects_sidecar;
    std::printf("[%s] %s: sidecar=%d\n", passed ? "ok  " : "FAIL", staging,
        engine.encoder_on_coreml() ? 1 : 0);
    for (const Case & test : cases) {
        const std::vector<float> samples = tones(test.seconds);
        const parakeet::DiarizationResult result = engine.diarize_samples(
            samples.data(), static_cast<int>(samples.size()), kSampleRate);
        const bool routed = result.encoder_used_coreml == test.expects_coreml;
        std::printf("[%s] %s %s: %d s coreml=%d\n", routed ? "ok  " : "FAIL", staging,
            test.name, test.seconds, result.encoder_used_coreml ? 1 : 0);
        passed = routed && passed;
    }
    return passed;
}

bool run_all(const fs::path & gguf) {
    const int largest = largest_staged_capacity(gguf);
    bool passed = run(gguf, "staged", true, {
        {"offline", kOfflineSeconds, true},
        {"offline-mid", kAboveSmallestSeconds,
            largest >= encoder_frames(kAboveSmallestSeconds)},
        {"long-form", kLongFormSeconds, largest >= kLongFormBlockFrames},
    });

    TemporaryDirectory smallest;
    fs::path smallest_gguf;
    if (!stage(gguf, {kSmallestCapacity}, smallest, smallest_gguf)) return false;
    passed = run(smallest_gguf, "smallest-only", true, {
        {"offline", kOfflineSeconds, true},
        {"above-capacity", kAboveSmallestSeconds, false},
        {"long-form", kLongFormSeconds, false},
    }) && passed;

    TemporaryDirectory none;
    fs::path none_gguf;
    if (!stage(gguf, {}, none, none_gguf)) return false;
    return run(none_gguf, "none", false, {{"offline", kOfflineSeconds, false}}) && passed;
}

}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]);
        return 2;
    }
    const fs::path gguf = argv[1];
    if (!fs::exists(bundle_path(gguf, kSmallestCapacity))) {
        std::printf("SKIP: no %s\n", bundle_path(gguf, kSmallestCapacity).c_str());
        return kSkip;
    }
    try {
        const bool passed = run_all(gguf);
        std::printf("[diarize-coreml-routing] %s\n", passed ? "PASS" : "FAIL");
        return passed ? 0 : 1;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "[diarize-coreml-routing] FAIL: %s\n", error.what());
        return 1;
    }
}
