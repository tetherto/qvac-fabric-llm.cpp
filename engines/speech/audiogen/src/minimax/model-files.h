#pragma once

#include "audiogen-cpp/minimax/engine.h"
#include "minimax/logic.h"
#include "minimax/mm3-model.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace tts_cpp::minimax::detail {

constexpr const char * kModelLicense = "MiniMax-Music3 Community License";
constexpr size_t kSynthComponentCount = 4;

static EngineOptions resolve_model_paths(const EngineOptions & input) {
    EngineOptions options = input;
    const ModelPair pair = resolve_model_pair(options.model_dir, options.lm_model_path, options.synth_model_path);
    options.lm_model_path = pair.lm;
    options.synth_model_path = pair.synth;
    return options;
}

static ModelCompatibility model_compatibility(const MM3Model & model) {
    const MM3LmConfig & lm = model.lm_cfg;
    const MM3SynthConfig & synth = model.synth_cfg;
    ModelCompatibility result;
    result.lm_embedding = lm.embedding_length;
    result.lm_codebooks = lm.num_codebooks;
    result.lm_acoustic_vocab = lm.acoustic_vocab_size;
    result.frame_rate = lm.frame_rate;
    result.max_audio_frames = lm.max_audio_frames;
    result.max_prompt_tokens = lm.max_prompt_tokens;
    result.depth_embedding = synth.depth.embedding_length;
    result.depth_codebooks = synth.depth.num_codebooks;
    result.depth_acoustic_vocab = synth.depth.audio_vocab_size;
    result.condition_layers = synth.cond.num_layers;
    result.condition_hidden = synth.cond.hidden_dim;
    result.condition_out = synth.cond.out_dim;
    result.condition_rate = {static_cast<int>(synth.cond.input_sampling_rate),
                             static_cast<int>(synth.cond.input_hop_length),
                             static_cast<int>(synth.cond.output_sampling_rate),
                             static_cast<int>(synth.cond.output_hop_length)};
    result.dit_condition = synth.dit.condition_dim;
    result.dit_channels = synth.dit.in_channels;
    result.window_frames = synth.dit.window_frames;
    result.hop_frames = synth.dit.hop_frames;
    result.window_latents = synth.dit.window_latents;
    result.hop_latents = synth.dit.hop_latents;
    result.vocoder_latent_channels = synth.voc.latent_channels;
    result.vocoder_sampling_rate = synth.voc.sampling_rate;
    result.vocoder_channels = synth.voc.channels;
    result.vocoder_upsample = synth.voc.total_upsample;
    result.components = synth.components;
    return result;
}

static void probe_model_files(MM3Model & model, const EngineOptions & options, MM3GgufOpen open_gguf = gf_load) {
    model.models_dir = options.model_dir;
    mm3_probe_file(options.lm_model_path, &model.lm_file, &model.lm_cfg, nullptr, &model.meta_errors, open_gguf);
    mm3_probe_file(options.synth_model_path, &model.synth_file, nullptr, &model.synth_cfg, &model.meta_errors,
                   open_gguf);
    if (model.lm_file.arch != "qwen3") {
        model.meta_errors.push_back("LM general.architecture must be qwen3");
    }
    if (model.synth_file.arch != "mm3") {
        model.meta_errors.push_back("synth general.architecture must be mm3");
    }
    if (model.lm_file.license != kModelLicense || model.synth_file.license != kModelLicense) {
        model.meta_errors.push_back("both GGUF files must declare the MiniMax-Music3 Community License");
    }
    if (model.synth_cfg.components.size() != kSynthComponentCount) {
        model.meta_errors.push_back("synth GGUF must contain depth, cond, dit, and vocoder components");
    }
    const std::vector<std::string> compatibility_errors = validate_model_compatibility(model_compatibility(model));
    model.meta_errors.insert(model.meta_errors.end(), compatibility_errors.begin(), compatibility_errors.end());
    if (!model.meta_errors.empty()) {
        throw std::runtime_error("minimax engine: " + model.meta_errors.front());
    }
}

}
