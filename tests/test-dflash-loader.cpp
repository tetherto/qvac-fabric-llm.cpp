#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <numeric>
#include <vector>

static constexpr const char * PATH = "test-dflash-loader.gguf";
static constexpr int64_t N_EMBD = 16;
static constexpr int32_t N_BLOCK = 4;
static constexpr int64_t N_VOCAB = 16;
static constexpr int64_t N_FF = 32;
static constexpr size_t TENSOR_DATA_BYTES = 1024 * 1024;
static constexpr int32_t SELECTOR_TOP_K = 2;
static constexpr float REL_TOLERANCE = 1e-3f;
static constexpr float RMS_NORM_EPS = 1e-5f;
// the fused-encoder check extracts two target layers, so its features are wider than the decoder input
static constexpr int32_t N_FUSED_TARGET_LAYERS = 2;
static constexpr uint32_t N_SWA = 8;

struct tensor_info {
    const char * name;
    int64_t ne[3];
};

static const tensor_info BASE_TENSORS[] = {
    { "fc.weight",                   { N_EMBD, N_EMBD, 0 } },
    { "enc.output_norm.weight",      { N_EMBD, 0,      0 } },
    { "output_norm.weight",          { N_EMBD, 0,      0 } },
    { "blk.0.attn_norm.weight",      { N_EMBD, 0,      0 } },
    { "blk.0.attn_q.weight",         { N_EMBD, N_EMBD, 0 } },
    { "blk.0.attn_k.weight",         { N_EMBD, N_EMBD, 0 } },
    { "blk.0.attn_v.weight",         { N_EMBD, N_EMBD, 0 } },
    { "blk.0.attn_output.weight",    { N_EMBD, N_EMBD, 0 } },
    { "blk.0.attn_q_norm.weight",    { N_EMBD / 2, 0,  0 } },
    { "blk.0.attn_k_norm.weight",    { N_EMBD / 2, 0,  0 } },
    { "blk.0.ffn_norm.weight",       { N_EMBD, 0,      0 } },
    { "blk.0.ffn_gate.weight",       { N_EMBD, N_FF,   0 } },
    { "blk.0.ffn_down.weight",       { N_FF,   N_EMBD, 0 } },
    { "blk.0.ffn_up.weight",         { N_EMBD, N_FF,   0 } },
};

static const tensor_info SELECTOR_TENSORS[] = {
    { "selector_predecessor.weight", { 4, N_VOCAB, 0 } },
    { "selector_successor.weight",   { 4, N_VOCAB, 0 } },
    { "selector_hidden.weight",      { N_EMBD, 4, 0 } },
    { "blk.0.attn_conv_base",        { N_EMBD, 2, 2 } },
    { "blk.0.attn_conv_proj.weight", { N_EMBD, 16, 0 } },
    { "blk.0.ffn_conv_base",         { N_EMBD, 2, 2 } },
    { "blk.0.ffn_conv_proj.weight",  { N_EMBD, 16, 0 } },
};

static const tensor_info DSPARK_TENSORS[] = {
    { "markov_w1.weight",           { 2, N_VOCAB, 0 } },
    { "markov_w2.weight",           { 2, N_VOCAB, 0 } },
    { "conf_proj.weight",           { N_EMBD + 2, 1, 0 } },
};

enum class case_type {
    legacy,
    dspark,
    valid_selector,
    oversized_top_k,
    missing_rank,
    missing_predecessor,
    missing_conv_projection,
    sliding_window,
    full_attention_window,
};

// the encoder fc takes n_target_layers concatenated features
static void set_fc_width(std::vector<tensor_info> & tensors, int32_t n_target_layers) {
    for (auto & ti : tensors) {
        if (std::strcmp(ti.name, "fc.weight") == 0) {
            ti.ne[0] = N_EMBD * n_target_layers;
        }
    }
}

// fc passes the first target layer's features through unchanged
static void fill_identity_fc(float * data, int64_t n_in) {
    for (int64_t o = 0; o < N_EMBD; ++o) {
        data[o*n_in + o] = 1.0f;
    }
}

static bool write_model(case_type kind, bool tensor_backed = false, int32_t n_target_layers = 1) {
    std::vector<tensor_info> tensors(std::begin(BASE_TENSORS), std::end(BASE_TENSORS));
    if (kind == case_type::dspark) {
        tensors.insert(tensors.end(), std::begin(DSPARK_TENSORS), std::end(DSPARK_TENSORS));
    }
    if (tensor_backed) {
        tensors.push_back({ "token_embd.weight", { N_EMBD, N_VOCAB, 0 } });
        tensors.push_back({ "output.weight", { N_EMBD, N_VOCAB, 0 } });
    }
    set_fc_width(tensors, n_target_layers);
    if (kind != case_type::legacy && kind != case_type::dspark) {
        for (size_t i = 0; i < std::size(SELECTOR_TENSORS); ++i) {
            if ((kind == case_type::missing_predecessor && i == 0) ||
                (kind == case_type::missing_conv_projection && i == 4)) {
                continue;
            }
            tensors.push_back(SELECTOR_TENSORS[i]);
        }
    }

    const size_t mem_size = ggml_tensor_overhead() * tensors.size() + (tensor_backed ? TENSOR_DATA_BYTES : 0);
    std::vector<uint8_t> mem(mem_size);
    ggml_init_params ip = { mem_size, mem.data(), !tensor_backed };
    ggml_context * ctx = ggml_init(ip);
    gguf_context * gguf = gguf_init_empty();

    gguf_set_val_str(gguf, "general.architecture", "dflash");
    gguf_set_val_u32(gguf, "dflash.block_count", 1);
    gguf_set_val_u32(gguf, "dflash.context_length", 16);
    gguf_set_val_u32(gguf, "dflash.embedding_length", N_EMBD);
    gguf_set_val_u32(gguf, "dflash.feed_forward_length", N_FF);
    gguf_set_val_u32(gguf, "dflash.attention.head_count", 2);
    gguf_set_val_f32(gguf, "dflash.attention.layer_norm_rms_epsilon", RMS_NORM_EPS);
    gguf_set_val_u32(gguf, "dflash.vocab_size", N_VOCAB);
    gguf_set_val_str(gguf, "tokenizer.ggml.model", "no_vocab");
    std::vector<int32_t> target_layers(n_target_layers);
    std::iota(target_layers.begin(), target_layers.end(), 1);
    gguf_set_arr_data(gguf, "dflash.target_layers", GGUF_TYPE_INT32, target_layers.data(), n_target_layers);

    gguf_set_val_u32(gguf, "dflash.block_size", 4);
    if (kind == case_type::sliding_window || kind == case_type::full_attention_window) {
        const bool is_swa = kind == case_type::sliding_window;
        gguf_set_val_u32(gguf, "dflash.attention.sliding_window", N_SWA);
        gguf_set_arr_data(gguf, "dflash.attention.sliding_window_pattern", GGUF_TYPE_BOOL, &is_swa, 1);
    }
    if (kind != case_type::legacy && kind != case_type::dspark) {
        gguf_set_val_u32(gguf, "dflash.conv_kernel_size", 2);
        gguf_set_val_u32(gguf, "dflash.conv_group_size", 4);
        if (kind != case_type::missing_rank) {
            gguf_set_val_u32(gguf, "dflash.selector_rank", 4);
        }
        gguf_set_val_u32(gguf, "dflash.selector_top_k", kind == case_type::oversized_top_k ? UINT32_MAX : SELECTOR_TOP_K);
    }

    for (const auto & ti : tensors) {
        ggml_tensor * t = ti.ne[2] > 0 ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ti.ne[0], ti.ne[1], ti.ne[2]) :
                ti.ne[1] > 0 ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ti.ne[0], ti.ne[1]) :
                ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ti.ne[0]);
        ggml_set_name(t, ti.name);
        if (tensor_backed) {
            auto * data = static_cast<float *>(t->data);
            std::fill_n(data, ggml_nelements(t), 0.0f);
            if (std::strcmp(ti.name, "output_norm.weight") == 0 || std::strcmp(ti.name, "enc.output_norm.weight") == 0) {
                std::fill_n(data, N_EMBD, 1.0f);
            } else if (std::strcmp(ti.name, "fc.weight") == 0) {
                fill_identity_fc(data, ti.ne[0]);
            }
        }
        gguf_add_tensor(gguf, t);
    }
    const bool ok = gguf_write_to_file(gguf, PATH, !tensor_backed);
    gguf_free(gguf);
    ggml_free(ctx);
    return ok;
}

// the values of the graph tensor whose name starts with name
struct observed_tensor {
    const char *       name;
    std::vector<float> values;
};

static bool observe_tensor(ggml_tensor * tensor, bool ask, void * user_data) {
    auto & observed = *static_cast<observed_tensor *>(user_data);
    if (std::strncmp(tensor->name, observed.name, std::strlen(observed.name)) != 0) {
        return false;
    }
    if (!ask) {
        observed.values.resize(ggml_nelements(tensor));
        ggml_backend_tensor_get(tensor, observed.values.data(), 0, ggml_nbytes(tensor));
    }
    return true;
}

// target features whose rows differ in scale, so the rms norm changes every row
static std::vector<float> make_target_features(int64_t n_embd_enc) {
    std::vector<float> features(N_BLOCK * n_embd_enc);
    for (size_t i = 0; i < features.size(); ++i) {
        features[i] = float(i % n_embd_enc + 1) * float(i / n_embd_enc + 1);
    }
    return features;
}

static void append_rms_norm(const float * row, std::vector<float> & out) {
    const float scale = 1.0f / std::sqrt(std::inner_product(row, row + N_EMBD, row, 0.0f) / N_EMBD + RMS_NORM_EPS);
    for (int64_t i = 0; i < N_EMBD; ++i) {
        out.push_back(row[i] * scale);
    }
}

// the encoder output with the identity fc: the rms norm of each row's first target layer features
static std::vector<float> expected_encoder_output(const std::vector<float> & features, int64_t n_embd_enc) {
    std::vector<float> out;
    for (int64_t r = 0; r < N_BLOCK; ++r) {
        append_rms_norm(features.data() + r * n_embd_enc, out);
    }
    return out;
}

static bool values_match(const std::vector<float> & values, const std::vector<float> & expected) {
    if (values.size() != expected.size()) {
        return false;
    }
    for (size_t i = 0; i < values.size(); ++i) {
        if (std::fabs(values[i] - expected[i]) > REL_TOLERANCE * std::max(1.0f, std::fabs(expected[i]))) {
            return false;
        }
    }
    return true;
}

static bool decode_target_features(llama_context * ctx, const std::vector<float> & features, int64_t n_embd_enc) {
    llama_batch batch = llama_batch_init(N_BLOCK, (int32_t) n_embd_enc, 1);
    batch.n_tokens = N_BLOCK;
    std::copy(features.begin(), features.end(), batch.embd);
    for (int32_t i = 0; i < N_BLOCK; ++i) {
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = false;
    }
    const bool decoded = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return decoded;
}

// an injection batch carries the target features at the encoder input width, and the decode graph encodes them
static bool check_fused_encoder() {
    if (!write_model(case_type::valid_selector, true, N_FUSED_TARGET_LAYERS)) {
        return false;
    }
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(PATH, mparams);
    if (!model) {
        return false;
    }
    observed_tensor observed = { "inp_g_embeddings", {} };
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = N_EMBD;
    cparams.n_batch = N_BLOCK;
    cparams.n_ubatch = N_BLOCK;
    cparams.cb_eval = observe_tensor;
    cparams.cb_eval_user_data = &observed;
    llama_context * ctx = llama_init_from_model(model, cparams);

    const int64_t n_embd_enc = N_EMBD * N_FUSED_TARGET_LAYERS;
    const std::vector<float> features = make_target_features(n_embd_enc);
    const bool ok = ctx && decode_target_features(ctx, features, n_embd_enc) &&
        values_match(observed.values, expected_encoder_output(features, n_embd_enc));
    llama_free(ctx);
    llama_model_free(model);
    return ok;
}

// the attention window bounds prompt rows a draft can attend: only a drafter whose every layer is SWA has one
static bool check_attn_window(case_type kind, int32_t expected) {
    if (!write_model(kind)) {
        return false;
    }
    llama_model_params params = llama_model_default_params();
    params.no_alloc = true;
    params.load_mode = LLAMA_LOAD_MODE_NONE;
    llama_model * model = llama_model_load_from_file(PATH, params);
    const bool ok = model && llama_model_attn_window(model) == expected;
    llama_model_free(model);
    return ok;
}

int main() {
    llama_backend_init();
    const struct {
        case_type kind;
        const char * name;
        bool expected;
    } cases[] = {
        { case_type::legacy,                  "legacy DFlash",                true  },
        { case_type::dspark,                  "legacy DSpark",                true  },
        { case_type::valid_selector,          "complete DFlash2",             true  },
        { case_type::oversized_top_k,         "UINT32_MAX top-k",             false },
        { case_type::missing_rank,            "incomplete selector metadata", false },
        { case_type::missing_predecessor,     "missing selector codebook",    false },
        { case_type::missing_conv_projection, "missing conv projection",      false },
    };

    int failures = 0;
    for (const auto & test : cases) {
        if (!write_model(test.kind)) {
            fprintf(stderr, "FAIL: could not write %s\n", test.name);
            ++failures;
            continue;
        }
        llama_model_params params = llama_model_default_params();
        params.no_alloc = true;
        params.load_mode = LLAMA_LOAD_MODE_NONE;
        llama_model * model = llama_model_load_from_file(PATH, params);
        const bool loaded = model != nullptr;
        if (loaded != test.expected) {
            fprintf(stderr, "FAIL: %s: expected %s, got %s\n", test.name,
                    test.expected ? "load" : "reject", loaded ? "load" : "reject");
            ++failures;
        }
        llama_model_free(model);
    }
    if (!check_fused_encoder()) {
        fprintf(stderr, "FAIL: an injection batch at the encoder input width was not encoded by fc and the encoder norm\n");
        ++failures;
    }
    if (!check_attn_window(case_type::sliding_window, (int32_t) N_SWA) ||
        !check_attn_window(case_type::full_attention_window, 0) ||
        !check_attn_window(case_type::valid_selector, 0)) {
        fprintf(stderr, "FAIL: the attention window must equal the sliding window only when every layer uses it\n");
        ++failures;
    }
    std::remove(PATH);
    llama_backend_free();
    return failures == 0 ? 0 : 1;
}
