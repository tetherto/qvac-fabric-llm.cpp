// Valid TDT encoder frames must not depend on zero-filled right padding.
#include "long_form.h"
#include "parakeet_ctc.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

namespace {

constexpr int kThreads = 8;
constexpr int kGpuLayers = 99;
// Both runs take a ggml encoder route and capture intermediates, so a staged
// Core ML sidecar is never used.
constexpr bool kCaptureIntermediates = true;
constexpr float kFrameStep = 0.17f;
constexpr float kChannelStep = 0.11f;
// GPU kernels can change F16 reduction rounding when the input shape changes.
constexpr float kMaximumError = 1.0e-3f;

struct PaddingCase {
    int frames;
    int padding;
};

constexpr PaddingCase kCases[] = {{32, 1}, {32, 9}, {33, 1}, {4000, 1}};

std::vector<float> make_mel(int frames, int channels) {
    std::vector<float> mel((size_t)frames * channels);
    for (int frame = 0; frame < frames; ++frame) {
        for (int channel = 0; channel < channels; ++channel) {
            mel[(size_t)frame * channels + channel] =
                std::sin((frame + 1) * kFrameStep + channel * kChannelStep);
        }
    }
    return mel;
}

bool check_padding(parakeet::ParakeetCtcModel &model, const PaddingCase &test) {
    const int channels = model.mel_cfg.n_mels;
    auto mel = make_mel(test.frames, channels);
    parakeet::EncoderOutputs reference;
    if (parakeet::run_encoder(model, mel.data(), test.frames, channels, reference,
                              parakeet::EncoderRoute::ggml(), -1, kCaptureIntermediates) != 0) {
        std::fprintf(stderr, "FAIL: encoding the valid input\n");
        return false;
    }

    mel.resize((size_t)(test.frames + test.padding) * channels, 0.0f);
    parakeet::EncoderOutputs padded;
    if (parakeet::run_encoder(model, mel.data(), test.frames + test.padding, channels, padded,
                              parakeet::EncoderRoute::ggml(), -1, kCaptureIntermediates) != 0) {
        std::fprintf(stderr, "FAIL: encoding the padded input\n");
        return false;
    }

    if (reference.used_coreml || padded.used_coreml) {
        std::fprintf(stderr, "FAIL: padding comparison requires ggml for both encoder runs\n");
        return false;
    }

    const size_t valid_values = (size_t)reference.n_enc_frames * reference.d_model;
    if (reference.encoder_out.size() != valid_values || padded.encoder_out.size() < valid_values ||
        padded.d_model != reference.d_model) {
        std::fprintf(stderr, "FAIL: inconsistent encoder output shape\n");
        return false;
    }
    float maximum_error = 0.0f;
    for (size_t index = 0; index < valid_values; ++index) {
        const float actual = padded.encoder_out[index];
        const float expected = reference.encoder_out[index];
        if (!std::isfinite(actual) || !std::isfinite(expected)) {
            std::fprintf(stderr, "FAIL: nonfinite encoder output\n");
            return false;
        }
        maximum_error = std::max(maximum_error, std::fabs(actual - expected));
    }
    const bool passed = maximum_error <= kMaximumError;
    std::fprintf(stderr, "%s: frames=%d padding=%d max_abs=%g tolerance=%g\n", passed ? "PASS" : "FAIL",
                 test.frames, test.padding, maximum_error, kMaximumError);
    return passed;
}

int run(const std::string &model_path, const std::string &backend) {
    parakeet::ParakeetCtcModel model;
    const bool cpu = backend == "cpu";
    if (parakeet::load_from_gguf(model_path, model, kThreads, cpu ? 0 : kGpuLayers, false, backend) != 0) {
        std::fprintf(stderr, "FAIL: loading TDT model on %s\n", backend.c_str());
        return 1;
    }
    if (model.model_type != parakeet::ParakeetModelType::TDT ||
        model.model_variant == parakeet::kTranscribePsyEuVariant) {
        std::fprintf(stderr, "FAIL: expected a standard offline TDT model\n");
        return 1;
    }
    bool passed = true;
    for (const auto &test : kCases)
        passed = check_padding(model, test) && passed;
    return passed ? 0 : 1;
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        std::fprintf(stderr, "usage: %s <tdt.gguf> [backend=cpu]\n", argv[0]);
        return 2;
    }
    try {
        return run(argv[1], argc == 3 ? argv[2] : "cpu");
    } catch (const std::exception &error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
