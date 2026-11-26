#include "parakeet_ctc.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>

static bool rejects_non_float_silence_embedding() {
    ggml_tensor tensor{};
    tensor.type = GGML_TYPE_F16;
    try {
        parakeet::validate_nemotron_silence_embedding_type(&tensor);
        return false;
    } catch (const std::runtime_error &) {
        tensor.type = GGML_TYPE_F32;
        parakeet::validate_nemotron_silence_embedding_type(&tensor);
        return true;
    }
}

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    if (!rejects_non_float_silence_embedding()) return 1;
    parakeet::ParakeetCtcModel model;
    const int result = parakeet::load_from_gguf(argv[1], model, 1, 0, false);
    if (result != 0) {
        std::fprintf(stderr, "Nemotron diarization load failed: %d\n", result);
        return 1;
    }
    const auto & config = model.nemotron_diarization_cfg;
    const auto & weights = model.nemotron_diarization;
    if (model.model_type != parakeet::ParakeetModelType::NEMOTRON_DIARIZATION ||
        config.speakers != 8 || config.output_stride != 1 ||
        weights.layers.size() != static_cast<size_t>(config.encoder_layers) ||
        !weights.feature_projection || !weights.upsample_w ||
        weights.upsample_w->ne[2] !=
            static_cast<int64_t>(config.output_width) * config.subsampling_factor ||
        !weights.silence_embedding || !weights.speakers_w ||
        model.mel_cfg.filterbank.size() != 128 * 257 ||
        model.mel_cfg.window.size() != 400 ||
        std::fabs(model.mel_cfg.window.front()) > 1.0e-6f ||
        std::fabs(model.mel_cfg.window.back()) > 1.0e-6f) {
        std::fprintf(stderr, "Nemotron diarization model contract mismatch\n");
        return 1;
    }
    return 0;
}
