#pragma once

#include "ggml.h"
#include "gguf.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <string>
#include <vector>

namespace minimax_tiny {

constexpr const char * kLicense          = "MiniMax-Music3 Community License";
constexpr const char * kModel            = "MiniMax-Music3";
constexpr uint32_t     kHidden           = 32;
constexpr uint32_t     kFeedForward      = 64;
constexpr uint32_t     kHeads            = 4;
constexpr uint32_t     kKvHeads          = 2;
constexpr uint32_t     kHeadDim          = 8;
constexpr uint32_t     kVocab            = 64;
constexpr uint32_t     kSemanticOffset   = 32;
constexpr uint32_t     kSemanticVocab    = 16;
constexpr uint32_t     kAcousticVocab    = 8;
constexpr uint32_t     kCodebooks        = 3;
constexpr uint32_t     kEos              = 20;
constexpr uint32_t     kFrameRate        = 25;
constexpr uint32_t     kMaxFrames        = 9000;
constexpr uint32_t     kContext          = 10240;
constexpr uint32_t     kMaxPromptTokens  = 2048;
constexpr uint32_t     kDepthPositions   = 4;
constexpr uint32_t     kCondOut          = 16;
constexpr uint32_t     kCondKernel       = 3;
constexpr uint32_t     kCondInputRate    = 24000;
constexpr uint32_t     kCondInputHop     = 960;
constexpr uint32_t     kSampleRate       = 44100;
constexpr uint32_t     kUpsample         = 512;
constexpr uint32_t     kDitHidden        = 16;
constexpr uint32_t     kDitHeads         = 2;
constexpr uint32_t     kDitFeedForward   = 32;
constexpr uint32_t     kLatentChannels   = 8;
constexpr uint32_t     kFoldChannels     = 4;
constexpr uint32_t     kRopeDim          = 8;
constexpr uint32_t     kFourierDim       = 8;
constexpr uint32_t     kWindowFrames     = 200;
constexpr uint32_t     kHopFrames        = 100;
constexpr uint32_t     kWindowLatents    = 689;
constexpr uint32_t     kHopLatents       = 344;
constexpr uint32_t     kFlowSteps        = 30;
constexpr uint32_t     kVocInput         = 8;
constexpr uint32_t     kVocHidden        = 16;
constexpr uint32_t     kVocInputKernel   = 7;
constexpr uint32_t     kResidualKernel   = 7;
constexpr size_t       kContextBytes     = 64ull * 1024 * 1024;
constexpr float        kWeightScale      = 0.02f;
constexpr float        kWeightFrequency  = 0.37f;
constexpr float        kRmsEps           = 1e-6f;
constexpr float        kLayerNormEps     = 1e-5f;
constexpr float        kRopeBase         = 1000000.0f;
constexpr float        kDitRopeTheta     = 10000.0f;
constexpr float        kArCfgScale       = 1.5f;
constexpr uint32_t     kArTopK           = 50;
constexpr float        kEmbeddingScale   = 0.35f;
constexpr float        kFlowCfgScale     = 1.7f;
constexpr float        kSnakeEps         = 1e-9f;
constexpr int32_t      kUpsampleRates[]  = { 8, 8, 4, 2 };
constexpr int32_t      kResDilations[]   = { 1, 3, 9 };

struct Writer {
    gguf_context * gguf = gguf_init_empty();
    ggml_context * ctx  = ggml_init({ kContextBytes, nullptr, false });
    float          seed = 0.0f;

    ~Writer() {
        gguf_free(gguf);
        ggml_free(ctx);
    }
};

inline void fill_pattern(ggml_tensor * tensor, float seed) {
    float * data = static_cast<float *>(tensor->data);
    for (int64_t i = 0; i < ggml_nelements(tensor); i++) {
        data[i] = kWeightScale * std::sin(static_cast<float>(i) * kWeightFrequency + seed);
    }
}

inline void fill_value(ggml_tensor * tensor, float value) {
    float * data = static_cast<float *>(tensor->data);
    for (int64_t i = 0; i < ggml_nelements(tensor); i++) {
        data[i] = value;
    }
}

inline ggml_tensor * new_tensor(Writer & writer, const std::string & name, std::initializer_list<int64_t> shape) {
    std::vector<int64_t> ne(shape);
    ggml_tensor *        tensor = ggml_new_tensor(writer.ctx, GGML_TYPE_F32, static_cast<int>(ne.size()), ne.data());
    ggml_set_name(tensor, name.c_str());
    gguf_add_tensor(writer.gguf, tensor);
    return tensor;
}

inline void add_weight(Writer & writer, const std::string & name, std::initializer_list<int64_t> shape) {
    writer.seed += 1.0f;
    fill_pattern(new_tensor(writer, name, shape), writer.seed);
}

inline void add_ones(Writer & writer, const std::string & name, std::initializer_list<int64_t> shape) {
    fill_value(new_tensor(writer, name, shape), 1.0f);
}

inline void set_identity(Writer & writer, const char * architecture, const char * name) {
    gguf_set_val_str(writer.gguf, "general.architecture", architecture);
    gguf_set_val_str(writer.gguf, "general.name", name);
    gguf_set_val_str(writer.gguf, "general.license", kLicense);
    gguf_set_val_str(writer.gguf, "mm3.model", kModel);
}

inline void set_lm_shape(gguf_context * gguf) {
    gguf_set_val_u32(gguf, "qwen3.block_count", 1);
    gguf_set_val_u32(gguf, "qwen3.context_length", kContext);
    gguf_set_val_u32(gguf, "qwen3.embedding_length", kHidden);
    gguf_set_val_u32(gguf, "qwen3.feed_forward_length", kFeedForward);
    gguf_set_val_u32(gguf, "qwen3.attention.head_count", kHeads);
    gguf_set_val_u32(gguf, "qwen3.attention.head_count_kv", kKvHeads);
    gguf_set_val_u32(gguf, "qwen3.attention.key_length", kHeadDim);
    gguf_set_val_u32(gguf, "qwen3.attention.value_length", kHeadDim);
    gguf_set_val_f32(gguf, "qwen3.attention.layer_norm_rms_epsilon", kRmsEps);
    gguf_set_val_f32(gguf, "qwen3.rope.freq_base", kRopeBase);
    gguf_set_val_u32(gguf, "qwen3.vocab_size", kVocab);
}

inline void set_lm_audio(gguf_context * gguf) {
    gguf_set_val_u32(gguf, "mm3.semantic_vocab_offset", kSemanticOffset);
    gguf_set_val_u32(gguf, "mm3.semantic_vocab_size", kSemanticVocab);
    gguf_set_val_u32(gguf, "mm3.acoustic_vocab_size", kAcousticVocab);
    gguf_set_val_u32(gguf, "mm3.num_codebooks", kCodebooks);
    gguf_set_val_u32(gguf, "mm3.eos_audio", kEos);
    gguf_set_val_u32(gguf, "mm3.frame_rate", kFrameRate);
    gguf_set_val_u32(gguf, "mm3.max_audio_frames", kMaxFrames);
    gguf_set_val_u32(gguf, "mm3.max_prompt_tokens", kMaxPromptTokens);
    gguf_set_val_f32(gguf, "mm3.ar.cfg_scale", kArCfgScale);
    gguf_set_val_u32(gguf, "mm3.ar.top_k", kArTopK);
    gguf_set_val_f32(gguf, "mm3.ar.embedding_scale", kEmbeddingScale);
}

inline void add_lm_tensors(Writer & writer) {
    const int64_t queries = kHeads * kHeadDim;
    const int64_t keys    = kKvHeads * kHeadDim;
    add_weight(writer, "token_embd.weight", { kHidden, kVocab });
    add_ones(writer, "output_norm.weight", { kHidden });
    add_weight(writer, "output.weight", { kHidden, kVocab });
    add_ones(writer, "blk.0.attn_norm.weight", { kHidden });
    add_weight(writer, "blk.0.attn_q.weight", { kHidden, queries });
    add_weight(writer, "blk.0.attn_k.weight", { kHidden, keys });
    add_weight(writer, "blk.0.attn_v.weight", { kHidden, keys });
    add_weight(writer, "blk.0.attn_output.weight", { queries, kHidden });
    add_ones(writer, "blk.0.attn_q_norm.weight", { kHeadDim });
    add_ones(writer, "blk.0.attn_k_norm.weight", { kHeadDim });
    add_ones(writer, "blk.0.ffn_norm.weight", { kHidden });
    add_weight(writer, "blk.0.ffn_gate.weight", { kHidden, kFeedForward });
    add_weight(writer, "blk.0.ffn_up.weight", { kHidden, kFeedForward });
    add_weight(writer, "blk.0.ffn_down.weight", { kFeedForward, kHidden });
}

inline void set_depth(gguf_context * gguf) {
    gguf_set_val_u32(gguf, "mm3.depth.block_count", 1);
    gguf_set_val_u32(gguf, "mm3.depth.embedding_length", kHidden);
    gguf_set_val_u32(gguf, "mm3.depth.feed_forward_length", kFeedForward);
    gguf_set_val_u32(gguf, "mm3.depth.head_count", kHeads);
    gguf_set_val_u32(gguf, "mm3.depth.head_dim", kHeadDim);
    gguf_set_val_u32(gguf, "mm3.depth.max_position", kDepthPositions);
    gguf_set_val_f32(gguf, "mm3.depth.rms_eps", kRmsEps);
    gguf_set_val_u32(gguf, "mm3.depth.num_codebooks", kCodebooks);
    gguf_set_val_u32(gguf, "mm3.depth.audio_vocab_size", kAcousticVocab);
    gguf_set_val_u32(gguf, "mm3.depth.audio_embd_rows", (kCodebooks - 1) * kAcousticVocab);
    gguf_set_val_bool(gguf, "mm3.depth.causal", true);
    gguf_set_val_bool(gguf, "mm3.depth.rope", false);
}

inline void add_depth_tensors(Writer & writer) {
    add_weight(writer, "depth.proj.weight", { kHidden, kHidden });
    add_weight(writer, "depth.pos_embd.weight", { kHidden, kDepthPositions });
    add_weight(writer, "depth.audio_embd.weight", { kHidden, (kCodebooks - 1) * kAcousticVocab });
    add_ones(writer, "depth.output_norm.weight", { kHidden });
    add_weight(writer, "depth.head.0.weight", { kHidden, kAcousticVocab });
    add_weight(writer, "depth.head.1.weight", { kHidden, kAcousticVocab });
    add_ones(writer, "depth.blk.0.attn_norm.weight", { kHidden });
    add_weight(writer, "depth.blk.0.attn_q.weight", { kHidden, kHidden });
    add_weight(writer, "depth.blk.0.attn_k.weight", { kHidden, kHidden });
    add_weight(writer, "depth.blk.0.attn_v.weight", { kHidden, kHidden });
    add_weight(writer, "depth.blk.0.attn_output.weight", { kHidden, kHidden });
    add_ones(writer, "depth.blk.0.ffn_norm.weight", { kHidden });
    add_weight(writer, "depth.blk.0.ffn_gate.weight", { kHidden, kFeedForward });
    add_weight(writer, "depth.blk.0.ffn_up.weight", { kHidden, kFeedForward });
    add_weight(writer, "depth.blk.0.ffn_down.weight", { kFeedForward, kHidden });
}

inline void set_cond(gguf_context * gguf) {
    gguf_set_val_u32(gguf, "mm3.cond.num_layers", kCodebooks);
    gguf_set_val_u32(gguf, "mm3.cond.hidden_dim", kHidden);
    gguf_set_val_u32(gguf, "mm3.cond.out_dim", kCondOut);
    gguf_set_val_u32(gguf, "mm3.cond.kernel_size", kCondKernel);
    gguf_set_val_u32(gguf, "mm3.cond.padding", 1);
    gguf_set_val_u32(gguf, "mm3.cond.input_sampling_rate", kCondInputRate);
    gguf_set_val_u32(gguf, "mm3.cond.input_hop_length", kCondInputHop);
    gguf_set_val_u32(gguf, "mm3.cond.output_sampling_rate", kSampleRate);
    gguf_set_val_u32(gguf, "mm3.cond.output_hop_length", kUpsample);
    gguf_set_val_str(gguf, "mm3.cond.interpolation", "nearest");
    gguf_set_val_str(gguf, "mm3.cond.layer_mix", "softmax");
}

inline void add_cond_tensors(Writer & writer) {
    add_weight(writer, "cond.layer_logits", { kCodebooks });
    add_ones(writer, "cond.layer_scale", { 1 });
    add_weight(writer, "cond.proj.weight", { kCondKernel, kHidden, kCondOut });
    add_weight(writer, "cond.proj.bias", { kCondOut });
}

inline void set_dit(gguf_context * gguf) {
    gguf_set_val_u32(gguf, "mm3.dit.block_count", 1);
    gguf_set_val_u32(gguf, "mm3.dit.embedding_length", kDitHidden);
    gguf_set_val_u32(gguf, "mm3.dit.head_count", kDitHeads);
    gguf_set_val_u32(gguf, "mm3.dit.head_dim", kDitHidden / kDitHeads);
    gguf_set_val_u32(gguf, "mm3.dit.ff_inner", kDitFeedForward);
    gguf_set_val_u32(gguf, "mm3.dit.in_channels", kLatentChannels);
    gguf_set_val_u32(gguf, "mm3.dit.condition_dim", kCondOut);
    gguf_set_val_u32(gguf, "mm3.dit.concat_channels", 2 * kLatentChannels + kCondOut);
    gguf_set_val_f32(gguf, "mm3.dit.layer_norm_eps", kLayerNormEps);
    gguf_set_val_u32(gguf, "mm3.dit.rope_dim", kRopeDim);
    gguf_set_val_f32(gguf, "mm3.dit.rope_theta", kDitRopeTheta);
    gguf_set_val_str(gguf, "mm3.dit.rope_type", "neox");
    gguf_set_val_u32(gguf, "mm3.dit.fourier_dim", kFourierDim);
    gguf_set_val_str(gguf, "mm3.dit.glu_order", "value_gate");
    gguf_set_val_bool(gguf, "mm3.dit.timestep_token_prepended", true);
    gguf_set_val_bool(gguf, "mm3.dit.pre_post_conv_residual", true);
    gguf_set_val_bool(gguf, "mm3.dit.attn_bias", false);
    gguf_set_val_u32(gguf, "mm3.dit.window_frames", kWindowFrames);
    gguf_set_val_u32(gguf, "mm3.dit.hop_frames", kHopFrames);
    gguf_set_val_u32(gguf, "mm3.dit.window_latents", kWindowLatents);
    gguf_set_val_u32(gguf, "mm3.dit.hop_latents", kHopLatents);
}

inline void add_dit_tensors(Writer & writer) {
    const int64_t concat = 2 * kLatentChannels + kCondOut;
    add_weight(writer, "dit.preprocess_conv.weight", { 1, concat, concat });
    add_weight(writer, "dit.postprocess_conv.weight", { 1, kLatentChannels, kLatentChannels });
    add_weight(writer, "dit.time_fourier.weight", { 1, kFourierDim / 2 });
    add_weight(writer, "dit.time_embd.0.weight", { kFourierDim, kDitHidden });
    add_weight(writer, "dit.time_embd.0.bias", { kDitHidden });
    add_weight(writer, "dit.time_embd.1.weight", { kDitHidden, kDitHidden });
    add_weight(writer, "dit.time_embd.1.bias", { kDitHidden });
    add_weight(writer, "dit.proj_in.weight", { concat, kDitHidden });
    add_weight(writer, "dit.proj_out.weight", { kDitHidden, kLatentChannels });
    add_weight(writer, "dit.rope_inv_freq", { kRopeDim / 2 });
    add_ones(writer, "dit.blk.0.attn_norm.weight", { kDitHidden });
    add_weight(writer, "dit.blk.0.attn_norm.bias", { kDitHidden });
    add_weight(writer, "dit.blk.0.attn_qkv.weight", { kDitHidden, 3 * kDitHidden });
    add_weight(writer, "dit.blk.0.attn_output.weight", { kDitHidden, kDitHidden });
    add_ones(writer, "dit.blk.0.ffn_norm.weight", { kDitHidden });
    add_weight(writer, "dit.blk.0.ffn_norm.bias", { kDitHidden });
    add_weight(writer, "dit.blk.0.ffn_in.weight", { kDitHidden, 2 * kDitFeedForward });
    add_weight(writer, "dit.blk.0.ffn_in.bias", { 2 * kDitFeedForward });
    add_weight(writer, "dit.blk.0.ffn_out.weight", { kDitFeedForward, kDitHidden });
    add_weight(writer, "dit.blk.0.ffn_out.bias", { kDitHidden });
}

inline void set_flow(gguf_context * gguf) {
    gguf_set_val_str(gguf, "mm3.flow.scheduler", "FlowMatchEulerDiscrete");
    gguf_set_val_u32(gguf, "mm3.flow.steps", kFlowSteps);
    gguf_set_val_f32(gguf, "mm3.flow.cfg_scale", kFlowCfgScale);
    gguf_set_val_bool(gguf, "mm3.flow.invert_sigmas", true);
    gguf_set_val_f32(gguf, "mm3.flow.shift", 1.0f);
    gguf_set_val_u32(gguf, "mm3.flow.num_train_timesteps", 1);
}

inline void set_vocoder(gguf_context * gguf) {
    gguf_set_val_u32(gguf, "mm3.voc.latent_channels", kLatentChannels);
    gguf_set_val_u32(gguf, "mm3.voc.fold_channels", kFoldChannels);
    gguf_set_val_u32(gguf, "mm3.voc.dec_in_dim", kVocInput);
    gguf_set_val_u32(gguf, "mm3.voc.hidden_dim", kVocHidden);
    gguf_set_arr_data(gguf, "mm3.voc.upsample_rates", GGUF_TYPE_INT32, kUpsampleRates, std::size(kUpsampleRates));
    gguf_set_arr_data(gguf, "mm3.voc.res_dilations", GGUF_TYPE_INT32, kResDilations, std::size(kResDilations));
    gguf_set_val_u32(gguf, "mm3.voc.total_upsample", kUpsample);
    gguf_set_val_u32(gguf, "mm3.voc.sampling_rate", kSampleRate);
    gguf_set_val_u32(gguf, "mm3.voc.channels", 2);
    gguf_set_val_f32(gguf, "mm3.voc.snake_eps", kSnakeEps);
    gguf_set_val_bool(gguf, "mm3.voc.final_tanh", true);
    gguf_set_val_bool(gguf, "mm3.voc.weight_norm_folded", true);
}

inline void add_residual_tensors(Writer & writer, const std::string & prefix, int64_t channels) {
    add_ones(writer, prefix + "snake1.alpha", { 1, channels, 1 });
    add_weight(writer, prefix + "conv1.weight", { kResidualKernel, channels, channels });
    add_weight(writer, prefix + "conv1.bias", { channels });
    add_ones(writer, prefix + "snake2.alpha", { 1, channels, 1 });
    add_weight(writer, prefix + "conv2.weight", { 1, channels, channels });
    add_weight(writer, prefix + "conv2.bias", { channels });
}

inline void add_vocoder_block(Writer & writer, size_t block, int64_t channels) {
    const std::string prefix = "voc.blk." + std::to_string(block) + ".";
    const int64_t     output = channels / 2;
    add_ones(writer, prefix + "snake.alpha", { 1, channels, 1 });
    add_weight(writer, prefix + "convt.weight", { 2 * kUpsampleRates[block], output, channels });
    add_weight(writer, prefix + "convt.bias", { output });
    for (size_t residual = 0; residual < std::size(kResDilations); residual++) {
        add_residual_tensors(writer, prefix + "res." + std::to_string(residual) + ".", output);
    }
}

inline void add_vocoder_tensors(Writer & writer) {
    add_weight(writer, "voc.dec_in.weight", { 1, kFoldChannels, kVocInput });
    add_weight(writer, "voc.dec_in.bias", { kVocInput });
    add_weight(writer, "voc.conv_in.weight", { kVocInputKernel, kVocInput, kVocHidden });
    add_weight(writer, "voc.conv_in.bias", { kVocHidden });
    int64_t channels = kVocHidden;
    for (size_t block = 0; block < std::size(kUpsampleRates); block++) {
        add_vocoder_block(writer, block, channels);
        channels /= 2;
    }
    add_ones(writer, "voc.snake_out.alpha", { 1, channels, 1 });
    add_weight(writer, "voc.conv_out.weight", { kResidualKernel, channels, 1 });
    add_weight(writer, "voc.conv_out.bias", { 1 });
}

inline void set_components(gguf_context * gguf) {
    const char * components[] = { "depth", "cond", "dit", "vocoder" };
    gguf_set_arr_str(gguf, "mm3.synth.components", components, std::size(components));
}

inline bool write_lm(const std::filesystem::path & path) {
    Writer writer;
    set_identity(writer, "qwen3", "tiny MiniMax-Music3 LM");
    set_lm_shape(writer.gguf);
    set_lm_audio(writer.gguf);
    add_lm_tensors(writer);
    return gguf_write_to_file(writer.gguf, path.string().c_str(), false);
}

inline bool write_synth(const std::filesystem::path & path) {
    Writer writer;
    set_identity(writer, "mm3", "tiny MiniMax-Music3 synth");
    set_components(writer.gguf);
    set_depth(writer.gguf);
    set_cond(writer.gguf);
    set_dit(writer.gguf);
    set_flow(writer.gguf);
    set_vocoder(writer.gguf);
    add_depth_tensors(writer);
    add_cond_tensors(writer);
    add_dit_tensors(writer);
    add_vocoder_tensors(writer);
    return gguf_write_to_file(writer.gguf, path.string().c_str(), false);
}

inline void strip_payload(const std::filesystem::path & path) {
    ggml_context * metadata = nullptr;
    gguf_context * gguf     = gguf_init_from_file(path.string().c_str(), { true, &metadata });
    const size_t   bytes    = gguf_get_meta_size(gguf);
    ggml_free(metadata);
    gguf_free(gguf);
    std::filesystem::resize_file(path, bytes);
}

struct Pair {
    std::filesystem::path directory;
    std::filesystem::path lm;
    std::filesystem::path synth;
    bool                  written = false;

    explicit Pair(const std::string & tag) {
        const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
        directory = std::filesystem::temp_directory_path() / ("minimax-" + tag + "-" + std::to_string(unique));
        std::filesystem::create_directories(directory);
        lm      = directory / "mm3-lm-f16.gguf";
        synth   = directory / "mm3-synth-f16.gguf";
        written = write_lm(lm) && write_synth(synth);
    }

    ~Pair() {
        std::error_code error;
        std::filesystem::remove_all(directory, error);
    }
};

}
