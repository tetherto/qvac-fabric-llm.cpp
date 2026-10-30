#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-ext.h"
#include "speculative.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <vector>

static constexpr const char * PATH = "test-dflash-loader.gguf";
// a whole Q4_0 block per head row, so the CPU backend can repack a quantized head
static constexpr int64_t N_EMBD = 32;
static constexpr int32_t N_BLOCK = 4;
// room for one-token draft vocabulary ranges whose graph nodes outgrow the DFlash2 base budget of 1024
static constexpr int64_t N_VOCAB = 512;
static constexpr int64_t N_FF = 32;
static constexpr size_t TENSOR_DATA_BYTES = 1024 * 1024;
static constexpr int32_t SELECTOR_TOP_K = 2;
// two [begin, end) ranges, so the head views are concatenated
static constexpr int32_t DRAFT_VOCAB_RANGES[] = { 2, 4, 10, 14 };
static constexpr int32_t N_DRAFT_VOCAB_RANGES = 2;
// one range with fewer ids than the selector top-k
static constexpr int32_t UNDERSIZED_DRAFT_VOCAB_RANGE[] = { 2, 2 + SELECTOR_TOP_K - 1 };
static constexpr float REL_TOLERANCE = 1e-3f;
static constexpr float RMS_NORM_EPS = 1e-5f;
// the fused-encoder check extracts two target layers, so its features are wider than the decoder input
static constexpr int32_t N_FUSED_TARGET_LAYERS = 2;
// rows of a reduced (d2t) draft head
static constexpr int64_t N_REDUCED_HEAD = 8;
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
    // 2 * kernel * (N_EMBD / group) projected values, N_EMBD for kernel 2 and group 4
    { "blk.0.attn_conv_proj.weight", { N_EMBD, N_EMBD, 0 } },
    { "blk.0.ffn_conv_base",         { N_EMBD, 2, 2 } },
    { "blk.0.ffn_conv_proj.weight",  { N_EMBD, N_EMBD, 0 } },
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
    reduced_head,
    quantized_head,
    sliding_window,
    full_attention_window,
};

static bool in_draft_vocab(int64_t id) {
    for (int32_t r = 0; r < N_DRAFT_VOCAB_RANGES; ++r) {
        if (id >= DRAFT_VOCAB_RANGES[2*r] && id < DRAFT_VOCAB_RANGES[2*r + 1]) {
            return true;
        }
    }
    return false;
}

// head rows with distinct logits that favor the ids outside the draft vocabulary; Q4_0 stores the constant rows exactly
static void fill_output_head(ggml_tensor * head) {
    std::vector<float> rows(N_VOCAB * N_EMBD);
    for (int64_t r = 0; r < N_VOCAB; ++r) {
        std::fill_n(rows.data() + r*N_EMBD, N_EMBD, (in_draft_vocab(r) ? -1.0f : 1.0f) * float(r + 1));
    }
    ggml_quantize_chunk(head->type, rows.data(), head->data, 0, N_VOCAB, N_EMBD, nullptr);
}

static ggml_type tensor_type(case_type kind, const char * name) {
    return kind == case_type::quantized_head && std::strcmp(name, "output.weight") == 0 ? GGML_TYPE_Q4_0 : GGML_TYPE_F32;
}

// the encoder fc takes n_target_layers concatenated features and the head has n_head rows
static void set_io_widths(std::vector<tensor_info> & tensors, int32_t n_target_layers, int64_t n_head) {
    for (auto & ti : tensors) {
        if (std::strcmp(ti.name, "fc.weight") == 0) {
            ti.ne[0] = N_EMBD * n_target_layers;
        } else if (std::strcmp(ti.name, "output.weight") == 0) {
            ti.ne[1] = n_head;
        }
    }
}

// fc passes the first target layer's features through unchanged
static void fill_identity_fc(float * data, int64_t n_in) {
    for (int64_t o = 0; o < N_EMBD; ++o) {
        data[o*n_in + o] = 1.0f;
    }
}

static void add_d2t(ggml_context * ctx, gguf_context * gguf, bool tensor_backed) {
    ggml_tensor * d2t = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, N_REDUCED_HEAD);
    ggml_set_name(d2t, "d2t");
    if (tensor_backed) {
        auto * ids = static_cast<int64_t *>(d2t->data);
        std::iota(ids, ids + N_REDUCED_HEAD, 0);
    }
    gguf_add_tensor(gguf, d2t);
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
    set_io_widths(tensors, n_target_layers, kind == case_type::reduced_head ? N_REDUCED_HEAD : N_VOCAB);
    if (kind != case_type::legacy && kind != case_type::dspark) {
        for (size_t i = 0; i < std::size(SELECTOR_TENSORS); ++i) {
            if ((kind == case_type::missing_predecessor && i == 0) ||
                (kind == case_type::missing_conv_projection && i == 4)) {
                continue;
            }
            tensors.push_back(SELECTOR_TENSORS[i]);
        }
    }

    // one more tensor for d2t
    const size_t mem_size = ggml_tensor_overhead() * (tensors.size() + 1) + (tensor_backed ? TENSOR_DATA_BYTES : 0);
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
        const ggml_type type = tensor_type(kind, ti.name);
        ggml_tensor * t = ti.ne[2] > 0 ? ggml_new_tensor_3d(ctx, type, ti.ne[0], ti.ne[1], ti.ne[2]) :
                ti.ne[1] > 0 ? ggml_new_tensor_2d(ctx, type, ti.ne[0], ti.ne[1]) :
                ggml_new_tensor_1d(ctx, type, ti.ne[0]);
        ggml_set_name(t, ti.name);
        if (tensor_backed) {
            auto * data = static_cast<float *>(t->data);
            std::memset(t->data, 0, ggml_nbytes(t));
            // token 0 embeds as 1..N_EMBD, so the decoder output and the head logits are not zero
            if (std::strcmp(ti.name, "token_embd.weight") == 0) {
                for (int64_t i = 0; i < N_EMBD; ++i) {
                    data[i] = float(i + 1);
                }
            } else if (std::strcmp(ti.name, "output_norm.weight") == 0 || std::strcmp(ti.name, "enc.output_norm.weight") == 0) {
                std::fill_n(data, N_EMBD, 1.0f);
            } else if (std::strcmp(ti.name, "fc.weight") == 0) {
                fill_identity_fc(data, ti.ne[0]);
            } else if (std::strcmp(ti.name, "output.weight") == 0 && ti.ne[1] == N_VOCAB) {
                fill_output_head(t);
            }
        }
        gguf_add_tensor(gguf, t);
    }
    if (kind == case_type::reduced_head) {
        add_d2t(ctx, gguf, tensor_backed);
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

// one block of token 0 at positions 0..N_BLOCK-1 on an empty cache, with logits for every token when asked
static bool decode_block(llama_context * ctx, bool logits) {
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_batch batch = llama_batch_init(N_BLOCK, 0, 1);
    batch.n_tokens = N_BLOCK;
    for (int32_t i = 0; i < N_BLOCK; ++i) {
        batch.token[i] = 0;
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = logits;
    }
    const bool decoded = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return decoded;
}

static bool read_last_logits(llama_context * ctx, std::vector<float> & logits) {
    const float * row = llama_get_logits_ith(ctx, N_BLOCK - 1);
    if (!row) {
        return false;
    }
    logits.assign(row, row + N_VOCAB);
    return true;
}

// every selector candidate after the anchor lies in the draft vocabulary
static bool candidates_in_draft_vocab(llama_context * ctx) {
    const float * lattice = llama_get_embeddings_nextn(ctx);
    if (!lattice) {
        return false;
    }
    for (int32_t i = 1; i < N_BLOCK; ++i) {
        for (int32_t k = 0; k < SELECTOR_TOP_K; ++k) {
            if (!in_draft_vocab((int64_t) lattice[i*N_EMBD + k])) {
                return false;
            }
        }
    }
    return true;
}

// the restricted row keeps the full width: -inf outside the draft vocabulary, the full-head logit inside
static bool logits_match_draft_vocab(const std::vector<float> & restricted, const std::vector<float> & full) {
    for (int64_t id = 0; id < N_VOCAB; ++id) {
        const bool ok = in_draft_vocab(id)
            ? std::fabs(restricted[id] - full[id]) <= REL_TOLERANCE * std::max(1.0f, std::fabs(full[id]))
            : restricted[id] == -INFINITY;
        if (!ok) {
            return false;
        }
    }
    return true;
}

static bool rejects_undersized_draft_vocab(llama_context * ctx) {
    return !llama_set_draft_vocab(ctx, UNDERSIZED_DRAFT_VOCAB_RANGE, 1) && ctx->get_cparams().draft_vocab.empty();
}

static bool run_draft_vocab_decodes(llama_context * ctx) {
    std::vector<float> full;
    std::vector<float> restricted;
    return decode_block(ctx, true) && read_last_logits(ctx, full) && rejects_undersized_draft_vocab(ctx) &&
        llama_set_draft_vocab(ctx, DRAFT_VOCAB_RANGES, N_DRAFT_VOCAB_RANGES) &&
        decode_block(ctx, false) && candidates_in_draft_vocab(ctx) &&
        decode_block(ctx, true) && read_last_logits(ctx, restricted) && logits_match_draft_vocab(restricted, full);
}

// one-token ranges over the whole vocabulary
static std::vector<int32_t> one_token_ranges() {
    std::vector<int32_t> ranges;
    for (int32_t id = 0; id < N_VOCAB; ++id) {
        ranges.push_back(id);
        ranges.push_back(id + 1);
    }
    return ranges;
}

static bool run_many_ranges_decode(llama_context * ctx) {
    const std::vector<int32_t> ranges = one_token_ranges();
    return llama_set_draft_vocab(ctx, ranges.data(), (int32_t) ranges.size()/2) && decode_block(ctx, false);
}

static bool rejects_draft_vocab(llama_context * ctx) {
    return !llama_set_draft_vocab(ctx, DRAFT_VOCAB_RANGES, N_DRAFT_VOCAB_RANGES);
}

// runs run on a draft context of the tensor-backed DFlash2 model of kind
static bool run_on_dflash2_context(case_type kind, bool (*run)(llama_context *)) {
    if (!write_model(kind, true)) {
        return false;
    }
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(PATH, mparams);
    if (!model) {
        return false;
    }
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = N_EMBD;
    cparams.n_batch = N_BLOCK;
    cparams.n_ubatch = N_BLOCK;
    llama_context * ctx = llama_init_from_model(model, cparams);

    bool ok = false;
    if (ctx) {
        llama_set_causal_attn(ctx, false);
        llama_set_embeddings_nextn(ctx, true, false);
        ok = run(ctx);
    }
    llama_free(ctx);
    llama_model_free(model);
    return ok;
}

// a DFlash2 drafter with a draft vocabulary proposes only ids in the ranges and still returns full-width logits
static bool check_draft_vocab_graph() {
    return run_on_dflash2_context(case_type::valid_selector, run_draft_vocab_decodes);
}

// every range adds graph nodes, so a vocabulary of one-token ranges needs a larger graph
static bool check_many_draft_vocab_ranges() {
    return run_on_dflash2_context(case_type::valid_selector, run_many_ranges_decode);
}

// the ranges are target token ids, which do not index the rows of a reduced (d2t) head
static bool check_reduced_head_draft_vocab() {
    return run_on_dflash2_context(case_type::reduced_head, rejects_draft_vocab);
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

// a head the loader stored repacked or split has no row views, so the drafter ignores the ranges and returns the full-head logits
static bool run_ignored_draft_vocab_decodes(llama_context * ctx) {
    std::vector<float> full;
    std::vector<float> ignored;
    return decode_block(ctx, true) && read_last_logits(ctx, full) &&
        llama_set_draft_vocab(ctx, DRAFT_VOCAB_RANGES, N_DRAFT_VOCAB_RANGES) &&
        decode_block(ctx, true) && read_last_logits(ctx, ignored) && values_match(ignored, full);
}

static bool run_quantized_head_decodes(llama_context * ctx) {
    const llama_model * model = llama_get_model(ctx);
    return model->has_backend_layout(model->output) ? run_ignored_draft_vocab_decodes(ctx) : run_draft_vocab_decodes(ctx);
}

// the CPU backend may repack a Q4_0 head, interleaving rows that a view at an arbitrary row does not address
static bool check_quantized_head_draft_vocab() {
    return run_on_dflash2_context(case_type::quantized_head, run_quantized_head_decodes);
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

static bool extraction_enabled(const llama_context * ctx) {
    const std::vector<bool> & layers = ctx->get_cparams().embeddings_layer_inp;
    return std::find(layers.begin(), layers.end(), true) != layers.end();
}

static bool spec_init_rejects(common_params_speculative & params) {
    try {
        common_speculative_free(common_speculative_init(params, 1));
    } catch (const std::runtime_error &) {
        return true;
    }
    return false;
}

// a draft vocabulary for a drafter without a selector stops the speculative setup and leaves the target context unchanged
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

static bool check_rejected_draft_vocab() {
    if (!write_model(case_type::legacy, true)) {
        return false;
    }
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(PATH, mparams);
    if (!model) {
        return false;
    }
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = N_EMBD;
    cparams.n_batch = N_BLOCK;
    cparams.n_ubatch = N_BLOCK;
    llama_context * ctx_tgt = llama_init_from_model(model, cparams);
    llama_context * ctx_dft = llama_init_from_model(model, cparams);

    bool ok = false;
    if (ctx_tgt && ctx_dft) {
        common_params_speculative params;
        params.types = { COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH };
        params.draft.ctx_tgt = ctx_tgt;
        params.draft.ctx_dft = ctx_dft;
        params.draft.vocab_ranges = { 0, (int32_t) N_VOCAB };
        ok = spec_init_rejects(params) && !extraction_enabled(ctx_tgt);
    }
    llama_free(ctx_dft);
    llama_free(ctx_tgt);
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
    if (!check_draft_vocab_graph()) {
        fprintf(stderr, "FAIL: DFlash2 draft vocabulary: candidate outside the ranges or logits not restricted to them\n");
        ++failures;
    }
    if (!check_quantized_head_draft_vocab()) {
        fprintf(stderr, "FAIL: DFlash2 draft vocabulary with a Q4_0 head: the logits are wrong, or the ranges were kept for a repacked head\n");
        ++failures;
    }
    if (!check_rejected_draft_vocab()) {
        fprintf(stderr, "FAIL: a rejected draft vocabulary did not stop the setup or left target extraction on\n");
        ++failures;
    }
    if (!check_many_draft_vocab_ranges()) {
        fprintf(stderr, "FAIL: a draft vocabulary of one-token ranges did not decode\n");
        ++failures;
    }
    if (!check_reduced_head_draft_vocab()) {
        fprintf(stderr, "FAIL: a DFlash2 drafter with a reduced (d2t) head accepted a draft vocabulary\n");
        ++failures;
    }
    if (!check_attn_window(case_type::sliding_window, (int32_t) N_SWA) ||
        !check_attn_window(case_type::full_attention_window, 0) ||
        !check_attn_window(case_type::valid_selector, 0)) {
        fprintf(stderr, "FAIL: the attention window must equal the sliding window only when every layer uses it\n");
        ++failures;
    }
    if (!check_fused_encoder()) {
        fprintf(stderr, "FAIL: an injection batch at the encoder input width was not encoded by fc and the encoder norm\n");
        ++failures;
    }
    std::remove(PATH);
    llama_backend_free();
    return failures == 0 ? 0 : 1;
}
