#pragma once

#include "moss/transcribe_bpe.h"

#include "ggml.h"
#include "gguf.h"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <functional>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace moss_transcribe_fixtures {

constexpr int SPECIALS = 8;
constexpr int TEXT_VOCAB = SPECIALS + 256;
constexpr float RANDOM_WEIGHT_RANGE = 0.5f;
constexpr size_t FIXTURE_ARENA_BYTES = 16 * 1024 * 1024;

using TensorShapes = std::map<std::string, std::pair<int64_t, int64_t>>;
using ConstantTensors = std::map<std::string, float>;

inline std::filesystem::path temp_gguf(const char * tag) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
        ("moss-" + std::string(tag) + "-" + std::to_string(stamp) + ".gguf");
}

struct GgufBuilder {
    gguf_context * file;
    ggml_context * tensors;
    ConstantTensors constants;
    TensorShapes shapes;
    std::mt19937 * random = nullptr;

    GgufBuilder() : file(gguf_init_empty()), tensors(ggml_init({FIXTURE_ARENA_BYTES, nullptr, false})) {}

    ~GgufBuilder() {
        ggml_free(tensors);
        gguf_free(file);
    }

    void fill_constant(ggml_tensor * tensor, float value) {
        for (int64_t i = 0; i < ggml_nelements(tensor); ++i) {
            ((float *) tensor->data)[i] = value;
        }
    }

    void fill_random(ggml_tensor * tensor) {
        std::uniform_real_distribution<float> draw(-RANDOM_WEIGHT_RANGE, RANDOM_WEIGHT_RANGE);
        for (int64_t i = 0; i < ggml_nelements(tensor); ++i) {
            ((float *) tensor->data)[i] = draw(*random);
        }
    }

    void fill(ggml_tensor * tensor, const std::string & name) {
        const auto found = constants.find(name);
        if (found != constants.end()) {
            fill_constant(tensor, found->second);
        } else {
            fill_random(tensor);
        }
    }

    void register_tensor(ggml_tensor * tensor, const std::string & name) {
        fill(tensor, name);
        ggml_set_name(tensor, name.c_str());
        gguf_add_tensor(file, tensor);
    }

    void add1(const std::string & name, int64_t n) {
        register_tensor(ggml_new_tensor_1d(tensors, GGML_TYPE_F32, n), name);
    }

    void add2(const std::string & name, int64_t ne0, int64_t ne1) {
        const auto shaped = shapes.find(name);
        if (shaped != shapes.end()) {
            ne0 = shaped->second.first;
            ne1 = shaped->second.second;
        }
        register_tensor(ggml_new_tensor_2d(tensors, GGML_TYPE_F32, ne0, ne1), name);
    }

    void write(const std::filesystem::path & path) {
        if (!gguf_write_to_file(file, path.string().c_str(), false)) {
            throw std::runtime_error("cannot write test GGUF");
        }
    }
};

inline std::vector<std::string> byte_complete_vocab() {
    std::vector<std::string> tokens(TEXT_VOCAB);
    tokens[0] = "<|endoftext|>";
    tokens[1] = "<|im_start|>";
    tokens[2] = "<|im_end|>";
    for (int i = 3; i < SPECIALS; ++i) {
        tokens[(size_t) i] = "tok" + std::to_string(i);
    }
    const parakeet::moss::detail::QwenByteBpe byte_map({}, {});
    for (int b = 0; b < 256; ++b) {
        tokens[(size_t) (SPECIALS + b)] = byte_map.byte_symbols()[(size_t) b];
    }
    return tokens;
}

constexpr int SAMPLE_RATE = 8000;
constexpr int N_FFT = 16;
constexpr int HOP = 4;
constexpr int N_MELS = 4;
constexpr int CHUNK_SAMPLES = 256;
constexpr int CHUNK_FRAMES = CHUNK_SAMPLES / HOP;
constexpr int ENC_EMBD = 8;
constexpr int ENC_FF = 16;
constexpr int ENC_HEADS = 2;
constexpr int ENC_CTX = CHUNK_FRAMES / 2;
constexpr int MERGE = 4;
constexpr int CHUNK_TOKENS = ENC_CTX / MERGE;
constexpr int TEXT_EMBD = 16;
constexpr int TEXT_FF = 32;
constexpr int TEXT_HEADS = 2;
constexpr int TEXT_KV_HEADS = 1;
constexpr int TEXT_HEAD_DIM = 8;
constexpr int TEXT_CTX = 4096;
constexpr int TOKEN_END_OF_TEXT = 0;
constexpr int TOKEN_IM_START = 1;
constexpr int TOKEN_IM_END = 2;
constexpr int TOKEN_AUDIO_START = 3;
constexpr int TOKEN_AUDIO_END = 4;
constexpr int TOKEN_AUDIO_PAD = 5;
constexpr int TOKEN_USER_DEFINED = 6;
constexpr int TOKEN_TYPE_NORMAL = 1;
constexpr int TOKEN_TYPE_CONTROL = 3;
constexpr int TOKEN_TYPE_USER_DEFINED = 4;
constexpr float TOKENS_PER_SECOND = 4.0f;
constexpr int MARKER_SECONDS = 1;
constexpr int DEFAULT_MAX_NEW_TOKENS = 6;
constexpr int CONV_KERNEL = 3;
constexpr float MEL_FILTER_WEIGHT = 0.1f;
const char * const SYSTEM_PROMPT = "sys";
const char * const DEFAULT_PROMPT = "hi";
const char * const HOTWORD_PREFIX = " Hotwords: ";
const char * const HOTWORD_SEPARATOR = ", ";

inline std::vector<std::string> transcribe_vocab() {
    std::vector<std::string> tokens = byte_complete_vocab();
    tokens[TOKEN_AUDIO_START] = "<|audio_start|>";
    tokens[TOKEN_AUDIO_END] = "<|audio_end|>";
    tokens[TOKEN_AUDIO_PAD] = "<|audio_pad|>";
    tokens[TOKEN_USER_DEFINED] = "<think>";
    return tokens;
}

inline std::vector<int32_t> transcribe_token_types() {
    std::vector<int32_t> types(TEXT_VOCAB, TOKEN_TYPE_NORMAL);
    for (int id = 0; id < SPECIALS; ++id) {
        types[(size_t) id] = TOKEN_TYPE_CONTROL;
    }
    types[TOKEN_USER_DEFINED] = TOKEN_TYPE_USER_DEFINED;
    return types;
}

inline int32_t byte_token(unsigned char byte) {
    return SPECIALS + byte;
}

inline void add_audio_meta(gguf_context * f) {
    gguf_set_val_u32(f, "moss-transcribe.audio.sample_rate", SAMPLE_RATE);
    gguf_set_val_u32(f, "moss-transcribe.audio.n_fft", N_FFT);
    gguf_set_val_u32(f, "moss-transcribe.audio.hop_length", HOP);
    gguf_set_val_u32(f, "moss-transcribe.audio.n_mels", N_MELS);
    gguf_set_val_u32(f, "moss-transcribe.audio.chunk_samples", CHUNK_SAMPLES);
    gguf_set_val_u32(f, "moss-transcribe.audio.chunk_frames", CHUNK_FRAMES);
}

inline void add_encoder_meta(gguf_context * f) {
    gguf_set_val_u32(f, "moss-transcribe.encoder.block_count", 1);
    gguf_set_val_u32(f, "moss-transcribe.encoder.embedding_length", ENC_EMBD);
    gguf_set_val_u32(f, "moss-transcribe.encoder.feed_forward_length", ENC_FF);
    gguf_set_val_u32(f, "moss-transcribe.encoder.attention.head_count", ENC_HEADS);
    gguf_set_val_u32(f, "moss-transcribe.encoder.context_length", ENC_CTX);
    gguf_set_val_f32(f, "moss-transcribe.encoder.attention.layer_norm_epsilon", 1e-5f);
    gguf_set_val_u32(f, "moss-transcribe.adaptor.merge_size", MERGE);
    gguf_set_val_f32(f, "moss-transcribe.adaptor.layer_norm_epsilon", 1e-6f);
}

inline void add_text_meta(gguf_context * f) {
    gguf_set_val_u32(f, "moss-transcribe.text.block_count", 1);
    gguf_set_val_u32(f, "moss-transcribe.text.embedding_length", TEXT_EMBD);
    gguf_set_val_u32(f, "moss-transcribe.text.feed_forward_length", TEXT_FF);
    gguf_set_val_u32(f, "moss-transcribe.text.attention.head_count", TEXT_HEADS);
    gguf_set_val_u32(f, "moss-transcribe.text.attention.head_count_kv", TEXT_KV_HEADS);
    gguf_set_val_u32(f, "moss-transcribe.text.attention.key_length", TEXT_HEAD_DIM);
    gguf_set_val_u32(f, "moss-transcribe.text.context_length", TEXT_CTX);
    gguf_set_val_f32(f, "moss-transcribe.text.rope.freq_base", 1000000.0f);
    gguf_set_val_f32(f, "moss-transcribe.text.attention.layer_norm_rms_epsilon", 1e-6f);
}

inline void add_token_meta(gguf_context * f) {
    gguf_set_val_u32(f, "moss-transcribe.token.audio_start", TOKEN_AUDIO_START);
    gguf_set_val_u32(f, "moss-transcribe.token.audio_end", TOKEN_AUDIO_END);
    gguf_set_val_u32(f, "moss-transcribe.token.audio_pad", TOKEN_AUDIO_PAD);
    gguf_set_val_u32(f, "moss-transcribe.token.im_start", TOKEN_IM_START);
    gguf_set_val_u32(f, "moss-transcribe.token.im_end", TOKEN_IM_END);
    gguf_set_val_u32(f, "moss-transcribe.token.pad", TOKEN_END_OF_TEXT);
}

inline void add_prompt_meta(gguf_context * f) {
    gguf_set_val_f32(f, "moss-transcribe.audio_tokens_per_second", TOKENS_PER_SECOND);
    gguf_set_val_u32(f, "moss-transcribe.time_marker_every_seconds", MARKER_SECONDS);
    gguf_set_val_bool(f, "moss-transcribe.time_markers", true);
    gguf_set_val_str(f, "moss-transcribe.system_prompt", SYSTEM_PROMPT);
    gguf_set_val_str(f, "moss-transcribe.default_prompt", DEFAULT_PROMPT);
    gguf_set_val_str(f, "moss-transcribe.hotword_prefix", HOTWORD_PREFIX);
    gguf_set_val_str(f, "moss-transcribe.hotword_separator", HOTWORD_SEPARATOR);
    const int32_t prompt_ids[] = {byte_token('h'), byte_token('i')};
    gguf_set_arr_data(f, "moss-transcribe.default_prompt_ids", GGUF_TYPE_INT32, prompt_ids, 2);
    gguf_set_val_u32(f, "moss-transcribe.default_max_new_tokens", DEFAULT_MAX_NEW_TOKENS);
}

inline void add_tokenizer_meta(gguf_context * f) {
    const std::vector<std::string> tokens = transcribe_vocab();
    std::vector<const char *> token_ptrs;
    for (const std::string & token : tokens) {
        token_ptrs.push_back(token.c_str());
    }
    gguf_set_arr_str(f, "tokenizer.ggml.tokens", token_ptrs.data(), token_ptrs.size());
    const char * merges[] = {"a b"};
    gguf_set_arr_str(f, "tokenizer.ggml.merges", merges, 1);
    const std::vector<int32_t> types = transcribe_token_types();
    gguf_set_arr_data(f, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
}

inline void add_meta(gguf_context * f) {
    gguf_set_val_str(f, "general.architecture", "moss-transcribe");
    add_audio_meta(f);
    add_encoder_meta(f);
    add_text_meta(f);
    add_token_meta(f);
    add_prompt_meta(f);
    add_tokenizer_meta(f);
}

inline void transcribe_add_f16_kernel(GgufBuilder & b, const std::string & name, int64_t in, int64_t out) {
    std::uniform_real_distribution<float> draw(-RANDOM_WEIGHT_RANGE, RANDOM_WEIGHT_RANGE);
    ggml_tensor * tensor = ggml_new_tensor_3d(b.tensors, GGML_TYPE_F16, CONV_KERNEL, in, out);
    auto * data = (ggml_fp16_t *) tensor->data;
    for (int64_t i = 0; i < ggml_nelements(tensor); ++i) {
        data[i] = ggml_fp32_to_fp16(draw(*b.random));
    }
    ggml_set_name(tensor, name.c_str());
    gguf_add_tensor(b.file, tensor);
}

inline void transcribe_add_linear(GgufBuilder & b, const std::string & name, int64_t in, int64_t out) {
    b.add2(name + ".weight", in, out);
    b.add1(name + ".bias", out);
}

inline void transcribe_add_norm(GgufBuilder & b, const std::string & name, int64_t width) {
    b.add1(name + ".weight", width);
    b.add1(name + ".bias", width);
}

inline void transcribe_add_encoder_tensors(GgufBuilder & b) {
    b.add2("audio.mel_filters", N_FFT / 2 + 1, N_MELS);
    transcribe_add_f16_kernel(b, "enc.conv1.weight", N_MELS, ENC_EMBD);
    b.add2("enc.conv1.bias", 1, ENC_EMBD);
    transcribe_add_f16_kernel(b, "enc.conv2.weight", ENC_EMBD, ENC_EMBD);
    b.add2("enc.conv2.bias", 1, ENC_EMBD);
    b.add2("enc.pos_embd", ENC_EMBD, ENC_CTX);
    transcribe_add_norm(b, "enc.blk.0.attn_norm", ENC_EMBD);
    transcribe_add_linear(b, "enc.blk.0.attn_q", ENC_EMBD, ENC_EMBD);
    b.add2("enc.blk.0.attn_k.weight", ENC_EMBD, ENC_EMBD);
    transcribe_add_linear(b, "enc.blk.0.attn_v", ENC_EMBD, ENC_EMBD);
    transcribe_add_linear(b, "enc.blk.0.attn_output", ENC_EMBD, ENC_EMBD);
    transcribe_add_norm(b, "enc.blk.0.ffn_norm", ENC_EMBD);
    transcribe_add_linear(b, "enc.blk.0.ffn_up", ENC_EMBD, ENC_FF);
    transcribe_add_linear(b, "enc.blk.0.ffn_down", ENC_FF, ENC_EMBD);
    transcribe_add_norm(b, "enc.output_norm", ENC_EMBD);
    transcribe_add_linear(b, "adaptor.fc1", (int64_t) ENC_EMBD * MERGE, TEXT_EMBD);
    transcribe_add_linear(b, "adaptor.fc2", TEXT_EMBD, TEXT_EMBD);
    transcribe_add_norm(b, "adaptor.norm", TEXT_EMBD);
}

inline void transcribe_add_text_tensors(GgufBuilder & b) {
    const int64_t q_dim = (int64_t) TEXT_HEADS * TEXT_HEAD_DIM;
    const int64_t kv_dim = (int64_t) TEXT_KV_HEADS * TEXT_HEAD_DIM;
    b.add2("text.token_embd.weight", TEXT_EMBD, TEXT_VOCAB);
    b.add1("text.output_norm.weight", TEXT_EMBD);
    b.add1("text.blk.0.attn_norm.weight", TEXT_EMBD);
    b.add2("text.blk.0.attn_q.weight", TEXT_EMBD, q_dim);
    b.add2("text.blk.0.attn_k.weight", TEXT_EMBD, kv_dim);
    b.add2("text.blk.0.attn_v.weight", TEXT_EMBD, kv_dim);
    b.add2("text.blk.0.attn_output.weight", q_dim, TEXT_EMBD);
    b.add1("text.blk.0.attn_q_norm.weight", TEXT_HEAD_DIM);
    b.add1("text.blk.0.attn_k_norm.weight", TEXT_HEAD_DIM);
    b.add1("text.blk.0.ffn_norm.weight", TEXT_EMBD);
    b.add2("text.blk.0.ffn_gate.weight", TEXT_EMBD, TEXT_FF);
    b.add2("text.blk.0.ffn_up.weight", TEXT_EMBD, TEXT_FF);
    b.add2("text.blk.0.ffn_down.weight", TEXT_FF, TEXT_EMBD);
}

inline std::filesystem::path write_transcribe_model(const char * tag, uint32_t seed,
        const std::function<void(gguf_context *)> & mutate_meta = [](gguf_context *) {},
        const TensorShapes & shapes = {}) {
    std::mt19937 rng(seed);
    GgufBuilder b;
    b.random = &rng;
    b.shapes = shapes;
    b.constants["audio.mel_filters"] = MEL_FILTER_WEIGHT;
    add_meta(b.file);
    mutate_meta(b.file);
    transcribe_add_encoder_tensors(b);
    transcribe_add_text_tensors(b);
    const auto path = temp_gguf(tag);
    b.write(path);
    return path;
}

} // namespace moss_transcribe_fixtures
