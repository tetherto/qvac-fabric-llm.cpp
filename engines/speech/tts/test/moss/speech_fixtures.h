#pragma once

#include "gguf_fixtures.h"

#include <string>
#include <vector>

namespace moss_speech_fixtures {

using moss_fixtures::GgufBuilder;

constexpr int EMBD = 16;
constexpr int FF = 32;
constexpr int HEADS = 2;
constexpr int KV_HEADS = 1;
constexpr int HEAD_DIM = 8;
constexpr int CONTEXT = 1024;
constexpr int AUDIO_VOCAB = 24;
constexpr int TOKEN_PAD = 0;
constexpr int TOKEN_IM_START = 1;
constexpr int TOKEN_IM_END = 2;
constexpr int TOKEN_SPEECH_START = 3;
constexpr int TOKEN_TEXT_PLACEHOLDER = 4;
constexpr int TOKEN_AUDIO_PAD = 5;
constexpr int TOKEN_SPEECH_END = 16;
constexpr int TOKEN_TYPE_NORMAL = 1;
constexpr int TOKEN_TYPE_CONTROL = 3;
const char * const AUDIO_SYSTEM_PROMPT = "say";
const char * const TEXT_SYSTEM_PROMPT = "write";

constexpr int VQ_SAMPLE_RATE = 8000;
constexpr int VQ_N_FFT = 16;
constexpr int VQ_HOP = 4;
constexpr int VQ_MELS = 4;
constexpr int VQ_CHUNK = 256;
constexpr int VQ_EMBD = 8;
constexpr int VQ_FF = 16;
constexpr int VQ_HEADS = 2;
constexpr int VQ_CTX = VQ_CHUNK / (VQ_HOP * 2);
constexpr int VQ_POOL = 2;
constexpr int VQ_CODES = 6;
constexpr int VQ_KERNEL = 3;
constexpr float VQ_MEL_FILTER = 0.1f;

inline int32_t byte_token(unsigned char byte) {
    return moss_fixtures::SPECIALS + byte;
}

inline std::vector<int32_t> token_types() {
    std::vector<int32_t> types(moss_fixtures::TEXT_VOCAB, TOKEN_TYPE_NORMAL);
    for (int id = 0; id < moss_fixtures::SPECIALS; ++id) {
        types[(size_t) id] = TOKEN_TYPE_CONTROL;
    }
    return types;
}

inline void add_tokenizer_meta(gguf_context * f) {
    const std::vector<std::string> tokens = moss_fixtures::byte_complete_vocab();
    std::vector<const char *> ptrs;
    for (const std::string & token : tokens) {
        ptrs.push_back(token.c_str());
    }
    gguf_set_arr_str(f, "tokenizer.ggml.tokens", ptrs.data(), ptrs.size());
    const char * merges[] = {"a b"};
    gguf_set_arr_str(f, "tokenizer.ggml.merges", merges, 1);
    const std::vector<int32_t> types = token_types();
    gguf_set_arr_data(f, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
}

inline void add_lm_meta(gguf_context * f) {
    gguf_set_val_str(f, "general.architecture", "moss-speech");
    gguf_set_val_u32(f, "moss-speech.block_count", 1);
    gguf_set_val_u32(f, "moss-speech.modality_block_count", 1);
    gguf_set_val_u32(f, "moss-speech.context_length", CONTEXT);
    gguf_set_val_u32(f, "moss-speech.embedding_length", EMBD);
    gguf_set_val_u32(f, "moss-speech.feed_forward_length", FF);
    gguf_set_val_u32(f, "moss-speech.attention.head_count", HEADS);
    gguf_set_val_u32(f, "moss-speech.attention.head_count_kv", KV_HEADS);
    gguf_set_val_u32(f, "moss-speech.attention.key_length", HEAD_DIM);
    gguf_set_val_f32(f, "moss-speech.rope.freq_base", 1000000.0f);
    gguf_set_val_f32(f, "moss-speech.attention.layer_norm_rms_epsilon", 1e-6f);
    gguf_set_val_u32(f, "moss-speech.text_vocab_size", moss_fixtures::TEXT_VOCAB);
    gguf_set_val_u32(f, "moss-speech.audio_vocab_size", AUDIO_VOCAB);
    gguf_set_val_u32(f, "moss-speech.token.text_placeholder", TOKEN_TEXT_PLACEHOLDER);
    gguf_set_val_u32(f, "moss-speech.token.audio_pad", TOKEN_AUDIO_PAD);
    gguf_set_val_u32(f, "moss-speech.token.speech_start", TOKEN_SPEECH_START);
    gguf_set_val_u32(f, "moss-speech.token.speech_end", TOKEN_SPEECH_END);
    gguf_set_val_u32(f, "moss-speech.token.im_start", TOKEN_IM_START);
    gguf_set_val_u32(f, "moss-speech.token.im_end", TOKEN_IM_END);
    gguf_set_val_u32(f, "moss-speech.token.pad", TOKEN_PAD);
    gguf_set_val_str(f, "moss-speech.audio_output_system_prompt", AUDIO_SYSTEM_PROMPT);
    gguf_set_val_str(f, "moss-speech.text_output_system_prompt", TEXT_SYSTEM_PROMPT);
    add_tokenizer_meta(f);
}

inline void speech_add_lm_layer(GgufBuilder & b, const std::string & prefix) {
    const int64_t q_dim = (int64_t) HEADS * HEAD_DIM;
    const int64_t kv_dim = (int64_t) KV_HEADS * HEAD_DIM;
    b.add1(prefix + "attn_norm.weight", EMBD);
    b.add2(prefix + "attn_q.weight", EMBD, q_dim);
    b.add2(prefix + "attn_k.weight", EMBD, kv_dim);
    b.add2(prefix + "attn_v.weight", EMBD, kv_dim);
    b.add2(prefix + "attn_output.weight", q_dim, EMBD);
    b.add1(prefix + "attn_q_norm.weight", HEAD_DIM);
    b.add1(prefix + "attn_k_norm.weight", HEAD_DIM);
    b.add1(prefix + "ffn_norm.weight", EMBD);
    b.add2(prefix + "ffn_gate.weight", EMBD, FF);
    b.add2(prefix + "ffn_up.weight", EMBD, FF);
    b.add2(prefix + "ffn_down.weight", FF, EMBD);
}

inline void speech_add_lm_tensors(GgufBuilder & b) {
    b.add2("text.token_embd.weight", EMBD, moss_fixtures::TEXT_VOCAB);
    b.add2("audio.token_embd.weight", EMBD, AUDIO_VOCAB);
    b.add1("text.output_norm.weight", EMBD);
    b.add1("audio.output_norm.weight", EMBD);
    b.add2("text.output.weight", EMBD, moss_fixtures::TEXT_VOCAB);
    b.add2("audio.output.weight", EMBD, AUDIO_VOCAB);
    speech_add_lm_layer(b, "blk.0.");
    speech_add_lm_layer(b, "text.blk.0.");
    speech_add_lm_layer(b, "audio.blk.0.");
}

inline std::filesystem::path write_lm(const char * tag, uint32_t seed,
        const std::function<void(gguf_context *)> & mutate = [](gguf_context *) {},
        const moss_fixtures::TensorShapes & shapes = {}) {
    std::mt19937 rng(seed);
    GgufBuilder b;
    b.random = &rng;
    b.shapes = shapes;
    add_lm_meta(b.file);
    mutate(b.file);
    speech_add_lm_tensors(b);
    const auto path = moss_fixtures::temp_gguf(tag);
    b.write(path);
    return path;
}

inline void add_vq_meta(gguf_context * f) {
    gguf_set_val_str(f, "general.architecture", "moss-speech-codec");
    gguf_set_val_u32(f, "moss-speech-codec.vq.sample_rate", VQ_SAMPLE_RATE);
    gguf_set_val_u32(f, "moss-speech-codec.vq.n_fft", VQ_N_FFT);
    gguf_set_val_u32(f, "moss-speech-codec.vq.hop_length", VQ_HOP);
    gguf_set_val_u32(f, "moss-speech-codec.vq.n_mels", VQ_MELS);
    gguf_set_val_u32(f, "moss-speech-codec.vq.chunk_samples", VQ_CHUNK);
    gguf_set_val_u32(f, "moss-speech-codec.vq.block_count", 1);
    gguf_set_val_u32(f, "moss-speech-codec.vq.embedding_length", VQ_EMBD);
    gguf_set_val_u32(f, "moss-speech-codec.vq.feed_forward_length", VQ_FF);
    gguf_set_val_u32(f, "moss-speech-codec.vq.attention.head_count", VQ_HEADS);
    gguf_set_val_u32(f, "moss-speech-codec.vq.context_length", VQ_CTX);
    gguf_set_val_u32(f, "moss-speech-codec.vq.pooling_kernel", VQ_POOL);
    gguf_set_val_u32(f, "moss-speech-codec.vq.codebook_size", VQ_CODES);
}

inline void speech_add_f16_kernel(GgufBuilder & b, const std::string & name, int64_t in, int64_t out) {
    std::uniform_real_distribution<float> draw(-moss_fixtures::RANDOM_WEIGHT_RANGE, moss_fixtures::RANDOM_WEIGHT_RANGE);
    ggml_tensor * tensor = ggml_new_tensor_3d(b.tensors, GGML_TYPE_F16, VQ_KERNEL, in, out);
    auto * data = (ggml_fp16_t *) tensor->data;
    for (int64_t i = 0; i < ggml_nelements(tensor); ++i) {
        data[i] = ggml_fp32_to_fp16(draw(*b.random));
    }
    ggml_set_name(tensor, name.c_str());
    gguf_add_tensor(b.file, tensor);
}

inline void speech_add_vq_linear(GgufBuilder & b, const std::string & name, int64_t in, int64_t out, bool bias) {
    b.add2(name + ".weight", in, out);
    if (bias) {
        b.add1(name + ".bias", out);
    }
}

inline void speech_add_vq_tensors(GgufBuilder & b) {
    const std::string blk = "whispervq.blk.0.";
    speech_add_f16_kernel(b, "whispervq.conv1.weight", VQ_MELS, VQ_EMBD);
    b.add2("whispervq.conv1.bias", 1, VQ_EMBD);
    speech_add_f16_kernel(b, "whispervq.conv2.weight", VQ_EMBD, VQ_EMBD);
    b.add2("whispervq.conv2.bias", 1, VQ_EMBD);
    b.add2("whispervq.pos_embd", VQ_EMBD, VQ_CTX);
    b.add2("whispervq.codebook", VQ_EMBD, VQ_CODES);
    b.add2("whispervq.mel_filters", VQ_N_FFT / 2 + 1, VQ_MELS);
    b.add1(blk + "attn_norm.weight", VQ_EMBD);
    b.add1(blk + "attn_norm.bias", VQ_EMBD);
    speech_add_vq_linear(b, blk + "attn_q", VQ_EMBD, VQ_EMBD, true);
    speech_add_vq_linear(b, blk + "attn_k", VQ_EMBD, VQ_EMBD, false);
    speech_add_vq_linear(b, blk + "attn_v", VQ_EMBD, VQ_EMBD, true);
    speech_add_vq_linear(b, blk + "attn_output", VQ_EMBD, VQ_EMBD, true);
    b.add1(blk + "ffn_norm.weight", VQ_EMBD);
    b.add1(blk + "ffn_norm.bias", VQ_EMBD);
    speech_add_vq_linear(b, blk + "ffn_up", VQ_EMBD, VQ_FF, true);
    speech_add_vq_linear(b, blk + "ffn_down", VQ_FF, VQ_EMBD, true);
}

inline std::filesystem::path write_vq(const char * tag, uint32_t seed,
        const std::function<void(gguf_context *)> & mutate = [](gguf_context *) {}) {
    std::mt19937 rng(seed);
    GgufBuilder b;
    b.random = &rng;
    b.constants["whispervq.mel_filters"] = {GGML_TYPE_F32, VQ_MEL_FILTER};
    add_vq_meta(b.file);
    mutate(b.file);
    speech_add_vq_tensors(b);
    const auto path = moss_fixtures::temp_gguf(tag);
    b.write(path);
    return path;
}

} // namespace moss_speech_fixtures
