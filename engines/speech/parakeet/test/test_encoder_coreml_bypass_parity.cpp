// Sortformer v2.1 AOSC Core ML block-stack parity against the ggml bypass path.

#include "coreml_test_options.h"
#include "parakeet_ctc.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

constexpr int kShortInputDeltaFrames = 7;
constexpr int kMinimumShortInputCapacity = kShortInputDeltaFrames + 1;

std::vector<float> create_input(int frames, int d_model) {
    std::vector<float> input((size_t) frames * d_model);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = 1.0e-3f * (float) ((int) (i * 2654435761u) % 1000 - 500);
    }
    return input;
}

double calculate_cosine(const std::vector<float> & ggml,
                        const std::vector<float> & coreml) {
    double dot = 0.0;
    double na = 0.0;
    double nb = 0.0;
    for (size_t i = 0; i < ggml.size(); ++i) {
        const double a = ggml[i];
        const double b = coreml[i];
        if (!std::isfinite(b)) return -1.0;
        dot += a * b;
        na += a * a;
        nb += b * b;
    }
    const double denom = std::sqrt(na) * std::sqrt(nb);
    return denom > 0.0 ? dot / denom : -1.0;
}

bool run_case(parakeet::ParakeetCtcModel & model, int frames, int capacity,
              int d_model, double min_cos) {
    const std::vector<float> input = create_input(frames, d_model);

    parakeet::EncoderOutputs ggml;
    if (parakeet::run_encoder_bypass_pre_encode(
            model, input.data(), frames, d_model, ggml,
            /*max_layers=*/model.encoder_cfg.n_layers) != 0 || ggml.used_coreml) {
        return false;
    }

    parakeet::EncoderOutputs coreml;
    if (parakeet::run_encoder_bypass_pre_encode(
            model, input.data(), frames, d_model, coreml) != 0 ||
        !coreml.used_coreml || coreml.encoder_out.size() != ggml.encoder_out.size()) {
        return false;
    }

    const double cosine = calculate_cosine(ggml.encoder_out, coreml.encoder_out);
    std::fprintf(stderr, "[coreml-bypass-parity] T=%d/%d D=%d cosine=%.6f\n",
                 frames, capacity, d_model, cosine);
    return cosine >= min_cos;
}

bool run_cases(parakeet::ParakeetCtcModel & model, int capacity,
               int d_model, double min_cos) {
    const int short_frames = capacity >= kMinimumShortInputCapacity
        ? capacity - kShortInputDeltaFrames
        : capacity;
    return run_case(model, capacity, capacity, d_model, min_cos) &&
           run_case(model, short_frames, capacity, d_model, min_cos);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <sortformer-v2.1.gguf> [min-cos]\n", argv[0]);
        return 2;
    }
    const double min_cos = argc > 2 ? std::atof(argv[2]) : 0.99;

    parakeet::ParakeetCtcModel model;
    if (parakeet::load_from_gguf(argv[1], model, 0, 999, true, "auto", coreml_ready_options()) != 0) return 3;
    if (!parakeet::model_bypass_encoder_on_coreml(model)) {
        std::fprintf(stderr, "[coreml-bypass-parity] SKIP: bypass sidecar inactive\n");
        return 0;
    }
    const int capacity = parakeet::model_coreml_bypass_fixed_frames(model);
    const int d_model = model.encoder_cfg.d_model;
    if (capacity <= 0 || d_model <= 0) return 4;
    return run_cases(model, capacity, d_model, min_cos) ? 0 : 1;
}
