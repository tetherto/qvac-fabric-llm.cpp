// Tests the laya decision graph (src/models/laya.cpp) on tiny models with random weights:
//
// - the output row layout and its zero padding
// - per-sequence results that do not depend on what else shares the batch
// - a sequence without a known question type still runs (as choice)
// - more option markers than the output row holds, without aborting
// - the encoder is ModernBert's: with no decision blocks and a zero type embedding, the per-token
//   output of a laya model equals the one of a modern-bert model with the same weights
// - models with invalid decision metadata fail to load instead of aborting
//
// usage: test-laya [directory for the generated models]

#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

// tiny model
static const int n_vocab       = 40;
static const int n_embd        = 64;
static const int n_head        = 2;
static const int n_layer       = 3;
static const int n_ff          = 64;
static const int n_act         = 2;
static const int n_act_hidden  = 16;
static const int n_max_options = 8;

// vocabulary: specials, the three question-type tokens, byte for '\n', then filler
enum : llama_token { T_PAD = 0, T_SEP = 1, T_CLS = 2, T_UNK = 3, T_MASK = 4, T_CHOICE = 5, T_SCORE = 6, T_NOUL = 7, T_NL = 8, T_WORD = 9 };

static int n_failed = 0;

#define CHECK(cond, ...) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); n_failed++; } } while (0)

struct model_desc {
    std::string arch;
    int         n_decision_layer;
    bool        zero_type_embd;

    // decision metadata, for the invalid models
    uint32_t max_options   = n_max_options;
    uint32_t act_count     = n_act;
    int      n_qtype_tokens = 3;
};

static void write_model(const std::string & path, const model_desc & desc) {
    const char * arch = desc.arch.c_str();
    auto key = [&](const char * k) { return desc.arch + "." + k; };

    gguf_context * gguf = gguf_init_empty();

    gguf_set_val_str (gguf, "general.architecture", arch);
    gguf_set_val_u32 (gguf, key("context_length").c_str(),      512);
    gguf_set_val_u32 (gguf, key("embedding_length").c_str(),    n_embd);
    gguf_set_val_u32 (gguf, key("feed_forward_length").c_str(), n_ff);
    gguf_set_val_u32 (gguf, key("block_count").c_str(),         n_layer);
    gguf_set_val_u32 (gguf, key("attention.head_count").c_str(), n_head);
    gguf_set_val_f32 (gguf, key("attention.layer_norm_epsilon").c_str(), 1e-5f);
    gguf_set_val_bool(gguf, key("attention.causal").c_str(), false);
    gguf_set_val_u32 (gguf, key("attention.sliding_window").c_str(), 8);
    gguf_set_val_u32 (gguf, key("attention.sliding_window_pattern").c_str(), 3);
    gguf_set_val_f32 (gguf, key("rope.freq_base").c_str(), 160000.0f);
    gguf_set_val_f32 (gguf, key("rope.freq_base_swa").c_str(), 10000.0f);
    // SwiGLU keeps the laya and modern-bert encoders on the same activation (laya switches GeGLU to its erf form)
    gguf_set_val_str (gguf, key("hidden_activation").c_str(), "silu");
    gguf_set_val_u32 (gguf, key("pooling_type").c_str(), desc.arch == "laya" ? LLAMA_POOLING_TYPE_RANK : LLAMA_POOLING_TYPE_NONE);

    if (desc.arch == "laya") {
        const int32_t qtype_tokens[3] = { T_CHOICE, T_SCORE, T_NOUL };
        gguf_set_val_u32 (gguf, "laya.decision.block_count", desc.n_decision_layer);
        gguf_set_val_u32 (gguf, "laya.decision.act_count",   desc.act_count);
        gguf_set_val_u32 (gguf, "laya.decision.max_options", desc.max_options);
        gguf_set_arr_data(gguf, "laya.decision.qtype_tokens", GGUF_TYPE_INT32, qtype_tokens, desc.n_qtype_tokens);
        gguf_set_val_str (gguf, "laya.decision.config", R"({"max_len": 128, "head_max_len": 64, "temperature": [1.0, 1.0, 1.0]})");
    }

    std::vector<std::string> tokens = { "<pad>", "<eos>", "<bos>", "<unk>", "<mask>", "\xe2\x96\x81" "choice", "\xe2\x96\x81" "score", "\xe2\x96\x81" "noul", "<0x0A>" };
    std::vector<float>   scores;
    std::vector<int32_t> types;
    for (int i = (int) tokens.size(); i < n_vocab; ++i) {
        tokens.push_back("\xe2\x96\x81" "w" + std::to_string(i));
    }
    for (int i = 0; i < n_vocab; ++i) {
        scores.push_back(-(float) i);
        types.push_back(i <= T_MASK ? 3 /* control */ : i == T_NL ? 6 /* byte */ : 1 /* normal */);
    }
    std::vector<const char *> token_ptrs;
    for (const auto & t : tokens) {
        token_ptrs.push_back(t.c_str());
    }
    gguf_set_val_str (gguf, "tokenizer.ggml.model", "llama");
    gguf_set_val_str (gguf, "tokenizer.ggml.pre",   "default");
    gguf_set_arr_str (gguf, "tokenizer.ggml.tokens", token_ptrs.data(), token_ptrs.size());
    gguf_set_arr_data(gguf, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), scores.size());
    gguf_set_arr_data(gguf, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
    gguf_set_val_bool(gguf, "tokenizer.ggml.add_space_prefix", false);
    gguf_set_val_u32 (gguf, "tokenizer.ggml.bos_token_id",       T_CLS);
    gguf_set_val_u32 (gguf, "tokenizer.ggml.eos_token_id",       T_SEP);
    gguf_set_val_u32 (gguf, "tokenizer.ggml.seperator_token_id", T_SEP);
    gguf_set_val_u32 (gguf, "tokenizer.ggml.unknown_token_id",   T_UNK);
    gguf_set_val_u32 (gguf, "tokenizer.ggml.padding_token_id",   T_PAD);
    gguf_set_val_u32 (gguf, "tokenizer.ggml.mask_token_id",      T_MASK);

    ggml_init_params params = { /*.mem_size =*/ 64u*1024*1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false };
    ggml_context * ctx = ggml_init(params);

    // the same seed gives laya and modern-bert models the same encoder weights
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.2f);

    auto add = [&](const std::string & name, std::vector<int64_t> ne, float offset = 0.0f, bool zero = false) {
        ggml_tensor * t = ne.size() == 1 ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne[0]) : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne[0], ne[1]);
        ggml_set_name(t, name.c_str());
        float * data = (float *) t->data;
        for (int64_t i = 0; i < ggml_nelements(t); ++i) {
            data[i] = zero ? 0.0f : offset + dist(rng);
        }
        gguf_add_tensor(gguf, t);
    };

    add("token_embd.weight",      { n_embd, n_vocab });
    add("token_embd_norm.weight", { n_embd }, 1.0f);
    add("output_norm.weight",     { n_embd }, 1.0f);
    for (int il = 0; il < n_layer; ++il) {
        const std::string blk = "blk." + std::to_string(il) + ".";
        if (il != 0) {
            add(blk + "attn_norm.weight", { n_embd }, 1.0f);
        }
        add(blk + "attn_qkv.weight",    { n_embd, 3*n_embd });
        add(blk + "attn_output.weight", { n_embd, n_embd });
        add(blk + "ffn_up.weight",      { n_embd, 2*n_ff });
        add(blk + "ffn_down.weight",    { n_ff, n_embd });
        add(blk + "ffn_norm.weight",    { n_embd }, 1.0f);
    }

    if (desc.arch == "laya") {
        add("decision.type_embd.weight", { n_embd, 3 }, 0.0f, desc.zero_type_embd);
        for (int il = 0; il < desc.n_decision_layer; ++il) {
            const std::string blk = "decision.blk." + std::to_string(il) + ".";
            add(blk + "attn_norm.weight",   { n_embd }, 1.0f);
            add(blk + "attn_norm.bias",     { n_embd });
            add(blk + "attn_qkv.weight",    { n_embd, 3*n_embd });
            add(blk + "attn_qkv.bias",      { 3*n_embd });
            add(blk + "attn_output.weight", { n_embd, n_embd });
            add(blk + "attn_output.bias",   { n_embd });
            add(blk + "ffn_norm.weight",    { n_embd }, 1.0f);
            add(blk + "ffn_norm.bias",      { n_embd });
            add(blk + "ffn_up.weight",      { n_embd, 4*n_embd });
            add(blk + "ffn_up.bias",        { 4*n_embd });
            add(blk + "ffn_down.weight",    { 4*n_embd, n_embd });
            add(blk + "ffn_down.bias",      { n_embd });
        }
        add("decision.scorer_norm.weight", { n_embd }, 1.0f);
        add("decision.scorer_norm.bias",   { n_embd });
        add("decision.scorer.weight",      { n_embd, n_embd });
        add("decision.scorer.bias",        { n_embd });
        add("decision.scorer_out.weight",  { n_embd, 1 });
        add("decision.scorer_out.bias",    { 1 });
        add("decision.act.weight",         { n_embd + 4, n_act_hidden });
        add("decision.act.bias",           { n_act_hidden });
        // the tensors keep a valid shape when only the metadata is wrong
        const int64_t n_act_out = desc.act_count > 0 ? desc.act_count : n_act;
        add("decision.act_out.weight",     { n_act_hidden, n_act_out });
        add("decision.act_out.bias",       { n_act_out });
    }

    if (!gguf_write_to_file(gguf, path.c_str(), false)) {
        fprintf(stderr, "failed to write %s\n", path.c_str());
        exit(1);
    }

    ggml_free(ctx);
    gguf_free(gguf);
}

// [CLS] <type> w w [SEP] ([MASK] w)*n_options [SEP] w w w [SEP]
static std::vector<llama_token> sequence(llama_token qtype, int n_options, int salt) {
    std::vector<llama_token> ids = { T_CLS, qtype, T_WORD + salt % 7, T_WORD + 1, T_SEP };
    for (int i = 0; i < n_options; ++i) {
        ids.push_back(T_MASK);
        ids.push_back(T_WORD + (3*i + salt) % 20);
    }
    ids.push_back(T_SEP);
    for (int i = 0; i < 3; ++i) {
        ids.push_back(T_WORD + (5*i + salt) % 25);
    }
    ids.push_back(T_SEP);
    return ids;
}

struct runner {
    llama_model   * model = nullptr;
    llama_context * ctx   = nullptr;
    int n_out = 0;

    runner(const std::string & path, enum llama_pooling_type pooling) {
        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = 0;
        model = llama_model_load_from_file(path.c_str(), mparams);
        if (!model) {
            fprintf(stderr, "failed to load %s\n", path.c_str());
            exit(1);
        }
        llama_context_params cparams = llama_context_default_params();
        cparams.embeddings   = true;
        cparams.pooling_type = pooling;
        cparams.n_ctx        = 512;
        cparams.n_batch      = 512;
        cparams.n_ubatch     = 512;
        cparams.n_seq_max    = 4;
        cparams.n_threads    = 4;
        // keep every op in f32 on the CPU: flash attention casts K and V to f16, which turns the tiny
        // differences between BLAS and ggml kernels (chosen by batch size) into f16 rounding steps
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cparams.op_offload      = false;
        ctx = llama_init_from_model(model, cparams);
        if (!ctx) {
            fprintf(stderr, "failed to create a context for %s\n", path.c_str());
            exit(1);
        }
        n_out = pooling == LLAMA_POOLING_TYPE_RANK ? (int) llama_model_n_cls_out(model) : llama_model_n_embd(model);
    }

    ~runner() {
        llama_free(ctx);
        llama_model_free(model);
    }

    // one output row per sequence (RANK) or per token (NONE)
    std::vector<std::vector<float>> run(const std::vector<std::vector<llama_token>> & seqs) {
        int n_tokens = 0;
        for (const auto & s : seqs) {
            n_tokens += s.size();
        }
        llama_batch batch = llama_batch_init(n_tokens, 0, 1);
        for (size_t s = 0; s < seqs.size(); ++s) {
            for (size_t i = 0; i < seqs[s].size(); ++i) {
                const int j = batch.n_tokens++;
                batch.token[j]     = seqs[s][i];
                batch.pos[j]       = i;
                batch.n_seq_id[j]  = 1;
                batch.seq_id[j][0] = s;
                batch.logits[j]    = true;
            }
        }
        llama_memory_clear(llama_get_memory(ctx), true);
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "llama_decode failed\n");
            exit(1);
        }

        std::vector<std::vector<float>> out;
        if (llama_pooling_type(ctx) == LLAMA_POOLING_TYPE_RANK) {
            for (size_t s = 0; s < seqs.size(); ++s) {
                const float * row = llama_get_embeddings_seq(ctx, s);
                out.emplace_back(row, row + n_out);
            }
        } else {
            for (int j = 0; j < n_tokens; ++j) {
                const float * row = llama_get_embeddings_ith(ctx, j);
                out.emplace_back(row, row + n_out);
            }
        }
        llama_batch_free(batch);
        return out;
    }
};

static float max_diff(const std::vector<float> & a, const std::vector<float> & b, size_t n) {
    float d = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        d = std::max(d, std::fabs(a[i] - b[i]));
    }
    return d;
}

static bool all_finite(const std::vector<float> & v) {
    for (float x : v) {
        if (!std::isfinite(x)) {
            return false;
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    const std::string dir = argc > 1 ? argv[1] : ".";
    const std::string laya_path   = dir + "/test-laya.gguf";
    const std::string laya0_path  = dir + "/test-laya-noblocks.gguf";
    const std::string mbert_path  = dir + "/test-laya-modern-bert.gguf";

    write_model(laya_path,  { "laya",        2, false });
    write_model(laya0_path, { "laya",        0, true  });
    write_model(mbert_path, { "modern-bert", 0, false });

    llama_backend_init();

    {
        runner r(laya_path, LLAMA_POOLING_TYPE_RANK);
        CHECK(r.n_out == n_act + n_max_options, "output row has %d floats, expected %d", r.n_out, n_act + n_max_options);

        const auto a = sequence(T_CHOICE, 3, 1);
        const auto b = sequence(T_SCORE,  5, 2);

        // row layout: act logits, one logit per option, zero after the sequence's options
        const auto alone = r.run({ a });
        CHECK(all_finite(alone[0]), "non-finite output");
        for (int i = n_act + 3; i < r.n_out; ++i) {
            CHECK(alone[0][i] == 0.0f, "slot %d after the last option is %g, expected 0", i - n_act, alone[0][i]);
        }

        // packed with a longer sequence, in either order: same results, same zero padding
        const auto packed   = r.run({ a, b });
        const auto reversed = r.run({ b, a });
        CHECK(max_diff(alone[0], packed[0], r.n_out)   < 1e-3f, "sequence depends on its batch: max diff %g", max_diff(alone[0], packed[0], r.n_out));
        CHECK(max_diff(alone[0], reversed[1], r.n_out) < 1e-3f, "sequence depends on its position in the batch: max diff %g", max_diff(alone[0], reversed[1], r.n_out));
        CHECK(max_diff(packed[1], reversed[0], r.n_out) < 1e-3f, "second sequence depends on the batch order");
        for (int i = n_act + 5; i < r.n_out; ++i) {
            CHECK(packed[1][i] == 0.0f, "slot %d after the last option is %g, expected 0", i - n_act, packed[1][i]);
        }

        // no known question type after [CLS]: evaluated as choice, with a warning
        auto a_other = a; a_other[1] = T_WORD;
        const auto untyped = r.run({ a_other });
        CHECK(all_finite(untyped[0]), "non-finite output without a question type");

        // more markers than the row holds: the first ones are scored, nothing aborts
        const auto many = r.run({ sequence(T_CHOICE, n_max_options + 4, 3) });
        CHECK(all_finite(many[0]), "non-finite output with too many markers");
    }

    {
        // per-token output of the decision blocks
        runner r(laya_path, LLAMA_POOLING_TYPE_NONE);
        const auto a = sequence(T_NOUL, 2, 4);
        const auto rows = r.run({ a });
        CHECK(rows.size() == a.size(), "expected one row per token");
        for (const auto & row : rows) {
            CHECK(all_finite(row), "non-finite per-token output");
        }
    }

    {
        // the laya encoder is modern-bert's
        runner laya (laya0_path, LLAMA_POOLING_TYPE_NONE);
        runner mbert(mbert_path, LLAMA_POOLING_TYPE_NONE);
        const auto a = sequence(T_CHOICE, 3, 5);
        const auto x = laya.run({ a });
        const auto y = mbert.run({ a });
        float d = 0.0f;
        for (size_t j = 0; j < x.size(); ++j) {
            d = std::max(d, max_diff(x[j], y[j], n_embd));
        }
        CHECK(d < 1e-5f, "laya encoder output differs from modern-bert: max diff %g", d);
    }

    {
        // invalid decision metadata: the load fails instead of aborting or sizing buffers from it
        const model_desc invalid[] = {
            { "laya", 2, false, 0xFFFFFFFFu, n_act, 3 }, // n_act + max_options wraps around
            { "laya", 2, false, 0,           n_act, 3 },
            { "laya", 2, false, 1,           n_act, 3 }, // the top-2 features need 2 option slots
            { "laya", 2, false, 256,         n_act, 3 },
            { "laya", n_layer + 1, false, n_max_options, n_act, 3 }, // more decision blocks than encoder layers
            { "laya", 2, false, n_max_options, n_act, 2 },           // not the 3 question types
            { "laya", 2, false, n_max_options, 0,     3 },           // no act outputs
        };
        const std::string path = dir + "/test-laya-invalid.gguf";
        for (size_t i = 0; i < sizeof(invalid)/sizeof(invalid[0]); ++i) {
            write_model(path, invalid[i]);
            llama_model_params mparams = llama_model_default_params();
            mparams.n_gpu_layers = 0;
            llama_model * model = llama_model_load_from_file(path.c_str(), mparams);
            CHECK(model == nullptr, "invalid model %zu loaded", i);
            if (model) {
                llama_model_free(model);
            }
        }
        std::remove(path.c_str());
    }

    llama_backend_free();

    if (argc <= 1) {
        std::remove(laya_path.c_str());
        std::remove(laya0_path.c_str());
        std::remove(mbert_path.c_str());
    }

    if (n_failed > 0) {
        fprintf(stderr, "%d checks failed\n", n_failed);
        return 1;
    }
    printf("all laya checks passed\n");
    return 0;
}
