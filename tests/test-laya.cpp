// Tests the laya decision graph (src/models/laya.cpp) on tiny models with random weights:
//
// - the output row layout and its zero padding
// - per-sequence results that do not depend on what else shares the batch
// - a sequence without a known question type still runs (as choice)
// - more option markers than the output row holds, without aborting
// - the encoder is ModernBert's: with no decision blocks and a zero type embedding, the per-token
//   output of a laya model equals the one of a modern-bert model with the same weights
// - models with invalid decision metadata fail to load instead of aborting
// - requests through common_laya_predict (common/laya.h): the response structure, the sequence layout, the
//   calibration and option_order decoding against the raw outputs, batch packing, truncation, and
//   malformed requests
//
// usage: test-laya [directory for the generated models]

#include "ggml.h"
#include "gguf.h"
#include "laya.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// tiny model
static const int n_embd        = 64;
static const int n_head        = 2;
static const int n_layer       = 3;
static const int n_ff          = 64;
static const int n_act         = 2;
static const int n_act_hidden  = 16;
static const int n_max_options = 8;

// vocabulary: specials, the three question-type tokens, then the byte tokens (T_WORD on, also used as filler)
enum : llama_token { T_PAD = 0, T_SEP = 1, T_CLS = 2, T_UNK = 3, T_MASK = 4, T_CHOICE = 5, T_SCORE = 6, T_NOUL = 7, T_WORD = 8 };

#define SP "\xe2\x96\x81" // U+2581, the SPM space

struct vocab_entry {
    std::string text;
    float       score;
    int32_t     type; // 1 normal, 3 control, 6 byte
};

// a byte-fallback SPM vocabulary like mmBERT's: every byte, the letters, and merges that form
// "\u2581choice", "\u2581score" and "\u2581noul", so that real text tokenizes and each question starts
// with its type token
static std::vector<vocab_entry> make_vocab() {
    std::vector<vocab_entry> v = {
        { "<pad>", 0, 3 }, { "<eos>", 0, 3 }, { "<bos>", 0, 3 }, { "<unk>", 0, 3 }, { "<mask>", 0, 3 },
        { SP "choice", 10, 1 }, { SP "score", 10, 1 }, { SP "noul", 10, 1 },
    };
    for (int b = 0; b < 256; ++b) {
        char buf[8];
        snprintf(buf, sizeof(buf), "<0x%02X>", b);
        v.push_back({ buf, 0, 6 });
    }
    v.push_back({ SP, -1, 1 });
    for (char c = 'a'; c <= 'z'; ++c) {
        v.push_back({ std::string(1, c), -1, 1 });
    }
    for (const char * word : { "choice", "score", "noul" }) {
        const std::string w = word;
        for (size_t n = 1; n < w.size(); ++n) {
            v.push_back({ SP + w.substr(0, n), 10, 1 });
        }
    }
    return v;
}

static const int n_vocab = (int) make_vocab().size();

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
        gguf_set_val_str (gguf, "laya.decision.config", R"({"max_len": 512, "head_max_len": 128, "temperature": [1.0, 1.0, 1.0]})");
    }

    const auto vocab = make_vocab();
    std::vector<const char *> token_ptrs;
    std::vector<float>        scores;
    std::vector<int32_t>      types;
    for (const auto & e : vocab) {
        token_ptrs.push_back(e.text.c_str());
        scores.push_back(e.score);
        types.push_back(e.type);
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

// the tiny models stay on the CPU: an explicit, empty device list keeps every GPU backend out of
// the context. n_gpu_layers = 0 alone is not enough: the weights then sit in the first device's
// host buffer type, and a backend that accepts host buffers (OpenVINO) computes the graph anyway
// although its decoder assumes KV-cache-backed attention and throws on the decision blocks
static llama_model_params cpu_model_params() {
    static ggml_backend_dev_t no_devices[] = { nullptr };
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.devices      = no_devices;
    return mparams;
}

struct runner {
    llama_model   * model = nullptr;
    llama_context * ctx   = nullptr;
    int n_out = 0;

    runner(const std::string & path, enum llama_pooling_type pooling) {
        llama_model_params mparams = cpu_model_params();
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

//
// requests through common_laya_predict
//

using json = nlohmann::ordered_json;

struct laya_env {
    llama_model   * model = nullptr;
    llama_context * ctx   = nullptr;

    laya_env(const std::string & path, uint32_t n_batch, enum llama_pooling_type pooling = LLAMA_POOLING_TYPE_RANK) {
        llama_model_params mparams = cpu_model_params();
        model = llama_model_load_from_file(path.c_str(), mparams);
        if (!model) {
            fprintf(stderr, "failed to load %s\n", path.c_str());
            exit(1);
        }
        llama_context_params cparams = llama_context_default_params();
        common_laya_context_params(cparams, n_batch);
        cparams.pooling_type    = pooling;
        cparams.n_threads       = 4;
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cparams.op_offload      = false;
        ctx = llama_init_from_model(model, cparams);
        if (!ctx) {
            fprintf(stderr, "failed to create a context for %s\n", path.c_str());
            exit(1);
        }
    }

    ~laya_env() {
        llama_free(ctx);
        llama_model_free(model);
    }
};

static std::vector<double> softmax(const std::vector<float> & z) {
    double zmax = -INFINITY;
    for (float x : z) {
        zmax = std::max<double>(zmax, x);
    }
    std::vector<double> p;
    double sum = 0.0;
    for (float x : z) {
        p.push_back(std::exp(x - zmax));
        sum += p.back();
    }
    for (double & x : p) {
        x /= sum;
    }
    return p;
}

// numbers within tol, everything else equal
static bool json_close(const json & a, const json & b, double tol) {
    if (a.is_number() && b.is_number()) {
        return std::fabs(a.get<double>() - b.get<double>()) <= tol;
    }
    if (a.type() != b.type() || a.size() != b.size()) {
        return false;
    }
    if (a.is_object()) {
        auto ib = b.begin();
        for (auto ia = a.begin(); ia != a.end(); ++ia, ++ib) {
            if (ia.key() != ib.key() || !json_close(ia.value(), ib.value(), tol)) {
                return false;
            }
        }
        return true;
    }
    if (a.is_array()) {
        for (size_t i = 0; i < a.size(); ++i) {
            if (!json_close(a[i], b[i], tol)) {
                return false;
            }
        }
        return true;
    }
    return a == b;
}

static void test_requests(const std::string & laya_path, const std::string & mbert_path) {
    laya_env env(laya_path, 512);
    const common_laya_ptr laya = common_laya_init(env.ctx);
    common_laya_warmup(laya.get());

    // a valid request: every question type, typed labels, structured criteria, option_order, three kinds of state
    const json request = json::parse(R"({
        "states": [
            "my payment failed twice, please refund the duplicate",
            {"subject": "invoice", "amount": 129.5, "items": [1, 2.0, true, null]},
            [{"role": "user", "content": "hello"}, {"role": "user", "content": "cancel my plan"}]
        ],
        "questions": {
            "pick":    {"type": "choice", "instructions": "which team", "criteria": {"alpha": "billing", "beta": {"n": 2}, "gamma": ""}},
            "labels":  {"type": "choice", "instructions": "which label", "criteria": ["a", 2, true]},
            "rate":    {"type": "score",  "instructions": "how urgent", "criteria": ["low", {"desc": "mid"}, 3]},
            "flag":    {"type": "noul",   "instructions": "a refund is requested", "labels": {"false": "no", "true": "yes"}},
            "ordered": {"type": "choice", "instructions": "which queue", "criteria": ["x", "y", "z"], "option_order": [2, 0, 1]}
        }
    })");

    const common_laya_result res = common_laya_predict(laya.get(), request);

    const json & response = res.response;
    CHECK(response.is_array() && response.size() == 3, "expected one result per state");
    CHECK(res.sequences.size() == 3*5, "expected one sequence per state and question, got %zu", res.sequences.size());

    struct expected_q { const char * id; llama_token type_token; size_t n_options; std::vector<int> order; };
    const std::vector<expected_q> expected = {
        { "pick",    T_CHOICE, 3, {} },
        { "labels",  T_CHOICE, 3, {} },
        { "rate",    T_SCORE,  3, {} },
        { "flag",    T_NOUL,   2, {} },
        { "ordered", T_CHOICE, 3, { 2, 0, 1 } },
    };

    for (size_t r = 0; r < res.sequences.size() && response.is_array() && response.size() == 3; ++r) {
        const auto & seq = res.sequences[r];
        const auto & q   = expected[r % expected.size()];
        const json & ans = response[seq.state]["answers"][q.id];

        CHECK(seq.question == q.id, "sequence %zu is for question %s, expected %s", r, seq.question.c_str(), q.id);

        // [CLS] <type> ... [SEP], a [MASK] at every marker
        CHECK(seq.tokens.size() > 2 && seq.tokens.front() == T_CLS && seq.tokens.back() == T_SEP, "%s: sequence does not start with [CLS] and end with [SEP]", q.id);
        CHECK(seq.tokens.size() > 1 && seq.tokens[1] == q.type_token, "%s: token after [CLS] is %d, expected the question type %d", q.id, seq.tokens[1], q.type_token);
        CHECK(seq.markers.size() == q.n_options, "%s: %zu markers, expected %zu", q.id, seq.markers.size(), q.n_options);
        for (int32_t m : seq.markers) {
            CHECK(m >= 0 && m < (int32_t) seq.tokens.size() && seq.tokens[m] == T_MASK, "%s: marker %d is not a [MASK]", q.id, m);
        }
        CHECK(seq.logits.size() == seq.markers.size() && seq.act.size() == (size_t) n_act, "%s: unexpected raw output sizes", q.id);

        // the answer follows from the raw logits: temperature 1, softmax, back to option order
        std::vector<double> p = softmax(seq.logits);
        if (!q.order.empty()) {
            std::vector<double> canonical(p.size());
            for (size_t s = 0; s < p.size(); ++s) {
                canonical[q.order[s]] = p[s];
            }
            p = canonical;
        }
        const size_t argmax = std::max_element(p.begin(), p.end()) - p.begin();

        const std::string type = ans.value("type", "");
        if (type == "choice") {
            const json & probs = ans["probabilities"];
            CHECK(probs.size() == p.size(), "%s: %zu probabilities, expected %zu", q.id, probs.size(), p.size());
            size_t i = 0;
            for (auto it = probs.begin(); it != probs.end() && i < p.size(); ++it, ++i) {
                CHECK(std::fabs(it.value().get<double>() - p[i]) < 1e-4, "%s: probability %zu is %g, raw logits give %g", q.id, i, it.value().get<double>(), p[i]);
            }
        } else if (type == "score") {
            double score = 0.0;
            for (size_t i = 0; i < p.size(); ++i) {
                score += i*p[i];
            }
            CHECK(std::fabs(ans["score"].get<double>() - score) < 1e-4, "%s: score %g, raw logits give %g", q.id, ans["score"].get<double>(), score);
        } else if (type == "noul") {
            CHECK(std::fabs(ans["noul"].get<double>() - p[1]) < 1e-4, "%s: noul %g, raw logits give %g", q.id, ans["noul"].get<double>(), p[1]);
        } else {
            CHECK(false, "%s: unexpected answer type '%s'", q.id, type.c_str());
        }
        CHECK(std::fabs(ans["answer_confidence"].get<double>() - p[argmax]) < 1e-4, "%s: answer_confidence does not match the raw logits", q.id);
        CHECK(std::fabs(ans["action"]["act_probability"].get<double>() - softmax(seq.act)[0]) < 1e-4, "%s: act_probability does not match the raw act logits", q.id);

        if (std::string(q.id) == "pick") {
            const char * labels[3] = { "alpha", "beta", "gamma" };
            CHECK(ans["choice"] == labels[argmax], "pick: choice %s, expected %s", ans["choice"].dump().c_str(), labels[argmax]);
        }
        if (std::string(q.id) == "labels") {
            // labels keep their JSON types, probability keys are the JSON key strings
            const json labels = json::array({ "a", 2, true });
            CHECK(ans["choice"] == labels[argmax], "labels: choice %s, expected %s", ans["choice"].dump().c_str(), labels[argmax].dump().c_str());
            std::vector<std::string> keys;
            for (auto it = ans["probabilities"].begin(); it != ans["probabilities"].end(); ++it) {
                keys.push_back(it.key());
            }
            CHECK((keys == std::vector<std::string>{ "a", "2", "true" }), "labels: unexpected probability keys");
        }
        if (std::string(q.id) == "rate") {
            CHECK(ans["legend"] == json::parse(R"({"0": "low", "1": "{\"desc\": \"mid\"}", "2": "3"})"), "rate: legend %s", ans["legend"].dump().c_str());
        }
    }

    // usage: the tokens of the state's sequences, no truncation
    for (size_t st = 0; st < 3 && response.is_array() && response.size() == 3; ++st) {
        size_t n_tokens = 0;
        for (const auto & seq : res.sequences) {
            if (seq.state == st) {
                n_tokens += seq.tokens.size();
            }
        }
        const json & usage = response[st]["usage"];
        CHECK(usage["input_tokens"].get<size_t>() == n_tokens, "state %zu: input_tokens %zu, sequences hold %zu", st, usage["input_tokens"].get<size_t>(), n_tokens);
        CHECK(usage["state_tokens"].get<size_t>() > 0 && usage["truncated"] == false, "state %zu: unexpected state usage %s", st, usage.dump().c_str());
    }

    // batch packing: the same answers from one forward pass or from many small ones
    {
        const json packed_request = json::parse(R"({
            "states": ["first ticket about a refund", "second ticket about a crash", "third ticket", "fourth"],
            "questions": {
                "pick": {"type": "choice", "instructions": "which team", "criteria": ["billing", "technical"]},
                "flag": {"type": "noul", "instructions": "urgent"}
            },
            "max_len": 96
        })");

        laya_env small(laya_path, 128);
        const common_laya_ptr laya_small = common_laya_init(small.ctx);

        const common_laya_result one  = common_laya_predict(laya.get(),       packed_request);
        const common_laya_result many = common_laya_predict(laya_small.get(), packed_request);
        CHECK(one.n_passes < many.n_passes, "expected fewer forward passes with the larger batch, got %d and %d", one.n_passes, many.n_passes);
        CHECK(json_close(one.response, many.response, 2e-4), "answers depend on the batch size:\n%s\n%s", one.response.dump().c_str(), many.response.dump().c_str());

        // a sequence that does not fit the batch is a request error
        json too_long = packed_request;
        too_long["max_len"] = 1000;
        too_long["states"] = json::array({ std::string(600, 'a') + " b" });
        bool threw = false;
        try {
            common_laya_predict(laya_small.get(), too_long);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        CHECK(threw, "a sequence longer than the batch did not throw std::invalid_argument");
    }

    // truncation: a long state is cut to max_len and reported
    {
        std::string long_state;
        for (int i = 0; i < 100; ++i) {
            long_state += "word ";
        }
        const json truncated_request = {
            { "state", long_state },
            { "questions", { { "flag", { { "type", "noul" }, { "instructions", "x" } } }, { "rate", { { "type", "score" }, { "instructions", "y" }, { "criteria", { "a", "b" } } } } } },
            { "max_len", 160 }, // longer than the heads (character-level tokens), shorter than head + state
        };
        const common_laya_result tr = common_laya_predict(laya.get(), truncated_request);
        const json & usage = tr.response["usage"];
        CHECK(usage["truncated"] == true && usage["state_tokens_dropped"].get<size_t>() > 0, "truncation not reported: %s", usage.dump().c_str());
        CHECK(usage["truncated_questions"].size() == 2, "expected both questions truncated: %s", usage.dump().c_str());
        for (const auto & seq : tr.sequences) {
            CHECK(seq.tokens.size() <= 160 && seq.tokens.back() == T_SEP, "sequence of %zu tokens, max_len 160", seq.tokens.size());
        }
    }

    // malformed requests throw std::invalid_argument
    {
        const char * malformed[] = {
            R"([])",
            R"({"state": "x", "questions": []})",
            R"({"states": "x", "questions": {}})",
            R"({"state": null, "questions": {"q": {"type": "noul", "instructions": "x"}}})",
            R"({"state": "x", "questions": {"q": {"type": "maybe", "instructions": "x"}}})",
            R"({"state": "x", "questions": {"q": {"type": "noul"}}})",
            R"({"state": "x", "questions": {"q": {"type": "noul", "instructions": null}}})",
            R"({"state": "x", "questions": {"": {"type": "noul", "instructions": "x"}}})",
            R"({"state": "x", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": []}}})",
            R"({"state": "x", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": [1, 1.0]}}})",
            R"({"state": "x", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": ["a", "b"], "option_order": [0, 0]}}})",
            R"({"state": "x", "questions": {"q": {"type": "score", "instructions": "x", "criteria": ["a", null]}}})",
            R"({"state": "x", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": ["a", "b", "c", "d", "e", "f", "g", "h", "i"]}}})",
            R"({"state": "x", "questions": {"q": {"type": "noul", "instructions": "x"}}, "max_len": -1})",
            R"({"state": "x", "questions": {"q": {"type": "noul", "instructions": "x"}}, "head_max_len": 0})",
            R"({"state": "x", "questions": {"q": {"type": "noul", "instructions": "x"}}, "max_len": "big"})",
        };
        for (const char * m : malformed) {
            bool invalid = false;
            try {
                common_laya_predict(laya.get(), json::parse(m));
            } catch (const std::invalid_argument &) {
                invalid = true;
            } catch (const std::exception & e) {
                fprintf(stderr, "  %s threw %s\n", m, e.what());
            }
            CHECK(invalid, "malformed request did not throw std::invalid_argument: %s", m);
        }
    }

    // contexts common_laya cannot use
    {
        laya_env mbert(mbert_path, 512, LLAMA_POOLING_TYPE_NONE);
        laya_env none(laya_path, 512, LLAMA_POOLING_TYPE_NONE);
        for (llama_context * ctx : { mbert.ctx, none.ctx }) {
            bool invalid = false;
            try {
                common_laya_init(ctx);
            } catch (const std::invalid_argument &) {
                invalid = true;
            }
            CHECK(invalid, "common_laya_init accepted a context it cannot use");
        }
    }
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

    test_requests(laya_path, mbert_path);

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
            llama_model_params mparams = cpu_model_params();
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
