// checks DSpine draft loading, the packed candidate lattice of the code bridge against a host reference, and the setup checks
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"
#include "llama-ext.h"
#include "llama-model.h"
#include "speculative.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr const char * DRAFT_PATH  = "test-dspine-draft.gguf";
static constexpr const char * TARGET_PATH = "test-dspine-target.gguf";

// a whole Q4_0 block per head row, so the CPU backend can repack a quantized head
static constexpr int64_t  N_EMBD       = 32;
static constexpr int64_t  N_VOCAB      = 64;
static constexpr int64_t  N_VOCAB_WIDE = N_VOCAB + 32; // a target head wider than the draft codebook
static constexpr int64_t  N_FF         = 32;
static constexpr int32_t  N_LAYER      = 2;
static constexpr int32_t  N_HEAD       = 2;
static constexpr int64_t  CODE_SIZE    = 8;
static constexpr int64_t  RANK         = 4;
static constexpr int32_t  TOP_K        = 4;
static constexpr int32_t  N_ROWS       = 4; // block size: the anchor row and three draft rows
static constexpr float    DSPINE_EPS   = 0.25f; // large, so the eps placement changes the result
static constexpr float    RMS_EPS      = 0.05f;
static constexpr int64_t  MASK_ID      = N_VOCAB - 1;
static constexpr float    WEIGHT_AMP   = 0.5f;
static constexpr double   REL_TOL      = 1e-4;
static constexpr size_t   DATA_BYTES   = 1024 * 1024;
static constexpr int32_t  ANCHORS[]    = { 5, 17 };
static constexpr int32_t  MAX_BLOCKS   = 2;
static constexpr uint32_t SEED         = 12345;

enum class draft_kind {
    valid,
    zero_eps,
    top_k_not_pow2,
    top_k_too_wide,
    not_causal,
    missing_codebook,
    with_markov,
    with_selector,
};

struct tensor_spec {
    std::string name;
    int64_t     ne0;
    int64_t     ne1;
};

static std::vector<tensor_spec> layer_tensors(int32_t il, bool with_bridge) {
    const std::string p = "blk." + std::to_string(il) + ".";
    std::vector<tensor_spec> tensors = {
        { p + "attn_norm.weight",   N_EMBD,          0 },
        { p + "attn_q.weight",      N_EMBD,          N_EMBD },
        { p + "attn_k.weight",      N_EMBD,          N_EMBD },
        { p + "attn_v.weight",      N_EMBD,          N_EMBD },
        { p + "attn_output.weight", N_EMBD,          N_EMBD },
        { p + "attn_q_norm.weight", N_EMBD / N_HEAD, 0 },
        { p + "attn_k_norm.weight", N_EMBD / N_HEAD, 0 },
        { p + "ffn_norm.weight",    N_EMBD,          0 },
        { p + "ffn_gate.weight",    N_EMBD,          N_FF },
        { p + "ffn_down.weight",    N_FF,            N_EMBD },
        { p + "ffn_up.weight",      N_EMBD,          N_FF },
    };
    if (with_bridge) {
        tensors.push_back({ p + "dspine_read.weight",  N_EMBD, CODE_SIZE });
        tensors.push_back({ p + "dspine_write.weight", RANK,   N_EMBD });
    }
    return tensors;
}

static void append_layer_tensors(std::vector<tensor_spec> & tensors, bool with_bridge) {
    for (int32_t il = 0; il < N_LAYER; ++il) {
        const auto layer = layer_tensors(il, with_bridge);
        tensors.insert(tensors.end(), layer.begin(), layer.end());
    }
}

// the draft has no token embeddings and no head: like the released checkpoints, it borrows the target's
static std::vector<tensor_spec> draft_tensors(draft_kind kind) {
    std::vector<tensor_spec> tensors = {
        { "fc.weight",               N_EMBD,    N_EMBD },
        { "enc.output_norm.weight",  N_EMBD,    0 },
        { "output_norm.weight",      N_EMBD,    0 },
        { "dspine_gate.weight",      CODE_SIZE, RANK },
        { "dspine_gate.bias",        RANK,      0 },
        { "dspine_pred_gate.weight", CODE_SIZE, RANK },
        { "dspine_prev.weight",      CODE_SIZE, RANK },
    };
    if (kind != draft_kind::missing_codebook) {
        tensors.push_back({ "dspine_codebook.weight", CODE_SIZE, N_VOCAB });
    }
    if (kind == draft_kind::with_markov) {
        tensors.push_back({ "markov_w1.weight", 2, N_VOCAB });
        tensors.push_back({ "markov_w2.weight", 2, N_VOCAB });
    }
    append_layer_tensors(tensors, true);
    return tensors;
}

// a plain DFlash model that owns the token embeddings and the head the draft borrows
static std::vector<tensor_spec> target_tensors(int64_t n_vocab) {
    std::vector<tensor_spec> tensors = {
        { "fc.weight",              N_EMBD, N_EMBD },
        { "enc.output_norm.weight", N_EMBD, 0 },
        { "output_norm.weight",     N_EMBD, 0 },
        { "token_embd.weight",      N_EMBD, n_vocab },
        { "output.weight",          N_EMBD, n_vocab },
    };
    append_layer_tensors(tensors, false);
    return tensors;
}

static void set_base_metadata(gguf_context * gguf, int64_t n_vocab) {
    gguf_set_val_str(gguf, "general.architecture", "dflash");
    gguf_set_val_u32(gguf, "dflash.block_count", N_LAYER);
    gguf_set_val_u32(gguf, "dflash.context_length", 16);
    gguf_set_val_u32(gguf, "dflash.embedding_length", N_EMBD);
    gguf_set_val_u32(gguf, "dflash.feed_forward_length", N_FF);
    gguf_set_val_u32(gguf, "dflash.attention.head_count", N_HEAD);
    gguf_set_val_f32(gguf, "dflash.attention.layer_norm_rms_epsilon", RMS_EPS);
    gguf_set_val_u32(gguf, "dflash.vocab_size", (uint32_t) n_vocab);
    gguf_set_val_str(gguf, "tokenizer.ggml.model", "no_vocab");
    const int32_t target_layers[] = { 1 };
    gguf_set_arr_data(gguf, "dflash.target_layers", GGUF_TYPE_INT32, target_layers, 1);
    gguf_set_val_u32(gguf, "dflash.block_size", N_ROWS);
}

static void set_dspine_metadata(gguf_context * gguf, draft_kind kind) {
    set_base_metadata(gguf, N_VOCAB);
    gguf_set_val_bool(gguf, "dflash.attention.causal", kind != draft_kind::not_causal);
    const uint32_t top_k = kind == draft_kind::top_k_not_pow2 ? 3 : kind == draft_kind::top_k_too_wide ? 8 : TOP_K;
    gguf_set_val_u32(gguf, "dflash.dspine.head_topk", top_k);
    gguf_set_val_f32(gguf, "dflash.dspine.eps", kind == draft_kind::zero_eps ? 0.0f : DSPINE_EPS);
    if (kind == draft_kind::with_selector) {
        gguf_set_val_u32(gguf, "dflash.selector_top_k", TOP_K);
    }
}

// a deterministic value in [-amp, amp]
static float next_value(uint32_t & state, float amp) {
    state = state * 1664525u + 1013904223u;
    return amp * (float(state >> 8) / float(1u << 24) * 2.0f - 1.0f);
}

static bool ends_with(const std::string & s, const char * suffix) {
    const size_t n = std::strlen(suffix);
    return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

// zero attention and FFN outputs make every decoder layer the identity, so only the DSpine writes change the hidden state
static bool is_zero_tensor(const std::string & name) {
    return ends_with(name, "attn_output.weight") || ends_with(name, "ffn_down.weight") || name == "fc.weight";
}

static float initial_value(const std::string & name, uint32_t & state) {
    if (is_zero_tensor(name)) {
        return 0.0f;
    }
    if (name == "output_norm.weight") {
        return 1.0f + next_value(state, WEIGHT_AMP);
    }
    return ends_with(name, "norm.weight") ? 1.0f : next_value(state, WEIGHT_AMP);
}

static void fill_values(std::vector<float> & values, const std::string & name, uint32_t & state) {
    for (auto & v : values) {
        v = initial_value(name, state);
    }
}

static ggml_tensor * new_tensor(ggml_context * ctx, ggml_type type, const tensor_spec & spec) {
    ggml_tensor * t = spec.ne1 > 0 ? ggml_new_tensor_2d(ctx, type, spec.ne0, spec.ne1) : ggml_new_tensor_1d(ctx, type, spec.ne0);
    ggml_set_name(t, spec.name.c_str());
    return t;
}

static void store_values(ggml_tensor * t, const std::vector<float> & values) {
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(t->data, values.data(), ggml_nbytes(t));
        return;
    }
    ggml_quantize_chunk(t->type, values.data(), t->data, 0, t->ne[1], t->ne[0], nullptr);
}

static void add_tensors(ggml_context * ctx, gguf_context * gguf, const std::vector<tensor_spec> & specs, ggml_type head_type, bool with_data) {
    uint32_t state = SEED;
    for (const auto & spec : specs) {
        ggml_tensor * t = new_tensor(ctx, spec.name == "output.weight" ? head_type : GGML_TYPE_F32, spec);
        if (with_data) {
            std::vector<float> values(ggml_nelements(t));
            fill_values(values, spec.name, state);
            store_values(t, values);
        }
        gguf_add_tensor(gguf, t);
    }
}

static bool write_gguf(const char * path, gguf_context * gguf, const std::vector<tensor_spec> & specs, ggml_type head_type, bool with_data) {
    const size_t mem_size = ggml_tensor_overhead() * specs.size() + (with_data ? DATA_BYTES : 0);
    std::vector<uint8_t> mem(mem_size);
    ggml_init_params ip = { mem_size, mem.data(), !with_data };
    ggml_context * ctx = ggml_init(ip);
    add_tensors(ctx, gguf, specs, head_type, with_data);
    const bool ok = gguf_write_to_file(gguf, path, !with_data);
    gguf_free(gguf);
    ggml_free(ctx);
    return ok;
}

static bool write_draft(draft_kind kind, bool with_data) {
    gguf_context * gguf = gguf_init_empty();
    set_dspine_metadata(gguf, kind);
    return write_gguf(DRAFT_PATH, gguf, draft_tensors(kind), GGML_TYPE_F32, with_data);
}

static bool write_target(int64_t n_vocab, ggml_type head_type) {
    gguf_context * gguf = gguf_init_empty();
    set_base_metadata(gguf, n_vocab);
    return write_gguf(TARGET_PATH, gguf, target_tensors(n_vocab), head_type, true);
}

static bool header_loads(draft_kind kind) {
    if (!write_draft(kind, false)) {
        return false;
    }
    llama_model_params params = llama_model_default_params();
    params.no_alloc = true;
    params.load_mode = LLAMA_LOAD_MODE_NONE;
    llama_model * model = llama_model_load_from_file(DRAFT_PATH, params);
    const bool loaded = model != nullptr;
    llama_model_free(model);
    return loaded;
}

// host reference of the DSpine bridge and lattice, in double

using vec = std::vector<double>;

struct host_draft {
    vec tok_embd, head, codebook, gate, gate_b, pred_gate, prev, out_norm;
    std::vector<vec> read, write;
};

static vec tensor_values(const llama_model * model, const char * name) {
    const ggml_tensor * t = model->get_tensor(name);
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    std::vector<float> f(ggml_nelements(t));
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(f.data(), raw.data(), raw.size());
    } else {
        ggml_get_type_traits(t->type)->to_float(raw.data(), f.data(), ggml_nelements(t));
    }
    return vec(f.begin(), f.end());
}

static void read_layer_values(const llama_model * draft, host_draft & d) {
    for (int32_t il = 0; il < N_LAYER; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        d.read.push_back(tensor_values(draft, (p + "dspine_read.weight").c_str()));
        d.write.push_back(tensor_values(draft, (p + "dspine_write.weight").c_str()));
    }
}

// target must keep the head in its stored row layout (loaded without repacking)
static host_draft read_host_draft(const llama_model * draft, const llama_model * target) {
    host_draft d;
    d.tok_embd  = tensor_values(target, "token_embd.weight");
    d.head      = tensor_values(target, "output.weight");
    d.codebook  = tensor_values(draft,  "dspine_codebook.weight");
    d.gate      = tensor_values(draft,  "dspine_gate.weight");
    d.gate_b    = tensor_values(draft,  "dspine_gate.bias");
    d.pred_gate = tensor_values(draft,  "dspine_pred_gate.weight");
    d.prev      = tensor_values(draft,  "dspine_prev.weight");
    d.out_norm  = tensor_values(draft,  "output_norm.weight");
    read_layer_values(draft, d);
    return d;
}

// y = W x for a ggml weight with ne0 = n_in, ne1 = n_out
static vec matvec(const vec & w, const vec & x, int64_t n_out) {
    const int64_t n_in = (int64_t) x.size();
    vec y(n_out, 0.0);
    for (int64_t o = 0; o < n_out; ++o) {
        y[o] = std::inner_product(x.begin(), x.end(), w.begin() + o * n_in, 0.0);
    }
    return y;
}

static vec row_of(const vec & table, int64_t row, int64_t width) {
    return vec(table.begin() + row * width, table.begin() + (row + 1) * width);
}

static double l2(const vec & x) {
    return std::sqrt(std::inner_product(x.begin(), x.end(), x.begin(), 0.0));
}

static double rms(const vec & x) {
    return l2(x) / std::sqrt((double) x.size());
}

static vec scaled(const vec & x, double s) {
    vec y(x);
    for (auto & v : y) {
        v *= s;
    }
    return y;
}

static vec added(const vec & a, const vec & b) {
    vec y(a);
    for (size_t i = 0; i < y.size(); ++i) {
        y[i] += b[i];
    }
    return y;
}

static vec multiplied(const vec & a, const vec & b) {
    vec y(a);
    for (size_t i = 0; i < y.size(); ++i) {
        y[i] *= b[i];
    }
    return y;
}

static vec sigmoid(const vec & x) {
    vec y(x);
    for (auto & v : y) {
        v = 1.0 / (1.0 + std::exp(-v));
    }
    return y;
}

static double dot(const vec & a, const vec & b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

static vec token_code(const host_draft & d, int64_t token) {
    return scaled(row_of(d.codebook, token, CODE_SIZE), std::sqrt((double) CODE_SIZE));
}

// eps is added after the square roots, as in the reference bridge
static vec soft_code(const host_draft & d, const vec & raw, int32_t il) {
    const vec proj = matvec(d.read[il], scaled(raw, 1.0 / (rms(raw) + DSPINE_EPS)), CODE_SIZE);
    return scaled(proj, std::sqrt((double) CODE_SIZE) / (l2(proj) + DSPINE_EPS));
}

// a row's state before the last write, which the refinement reuses
struct host_row_state {
    vec    raw;
    double rms = 0.0;
    vec    query;
};

// the message a row with query receives from the code of its predecessor row
static vec bridge_message(const host_draft & d, const vec & query, const vec & sender_code) {
    const vec gate = sigmoid(added(added(query, matvec(d.pred_gate, sender_code, RANK)), d.gate_b));
    return multiplied(gate, matvec(d.prev, sender_code, RANK));
}

static vec written(const vec & write, const host_row_state & s, const vec & msg) {
    return added(s.raw, scaled(matvec(write, msg, N_EMBD), s.rms));
}

static std::vector<vec> block_codes(const host_draft & d, int32_t il, const std::vector<vec> & h, int64_t anchor) {
    std::vector<vec> codes(N_ROWS);
    codes[0] = token_code(d, anchor);
    for (int32_t t = 1; t < N_ROWS; ++t) {
        codes[t] = soft_code(d, h[t], il);
    }
    return codes;
}

static void record_states(const host_draft & d, const std::vector<vec> & h, const std::vector<vec> & codes, std::vector<host_row_state> & state) {
    for (int32_t t = 0; t < N_ROWS; ++t) {
        state[t] = { h[t], rms(h[t]), matvec(d.gate, codes[t], RANK) };
    }
}

// one DSpine write over the rows of a block: row 0 carries the anchor code and gets no message
static std::vector<vec> bridge_layer(const host_draft & d, int32_t il, const std::vector<vec> & h, int64_t anchor, std::vector<host_row_state> & state) {
    const std::vector<vec> codes = block_codes(d, il, h, anchor);
    record_states(d, h, codes, state);
    std::vector<vec> out(h);
    for (int32_t t = 1; t < N_ROWS; ++t) {
        out[t] = written(d.write[il], state[t], bridge_message(d, state[t].query, codes[t - 1]));
    }
    return out;
}

static vec output_norm(const host_draft & d, const vec & x) {
    return multiplied(scaled(x, 1.0 / std::sqrt(rms(x) * rms(x) + RMS_EPS)), d.out_norm);
}

static std::vector<vec> embed_block(const host_draft & d, int64_t anchor) {
    std::vector<vec> h(N_ROWS);
    for (int32_t t = 0; t < N_ROWS; ++t) {
        h[t] = row_of(d.tok_embd, t == 0 ? anchor : MASK_ID, N_EMBD);
    }
    return h;
}

static std::vector<vec> run_bridge_layers(const host_draft & d, int64_t anchor, std::vector<host_row_state> & last) {
    std::vector<vec> h = embed_block(d, anchor);
    for (int32_t il = 0; il < N_LAYER; ++il) {
        h = bridge_layer(d, il, h, anchor, last);
    }
    return h;
}

struct host_block {
    std::vector<vec>            logits;
    std::vector<host_row_state> last;
};

static host_block forward_block(const host_draft & d, int64_t anchor) {
    host_block b;
    b.last.resize(N_ROWS);
    const std::vector<vec> h = run_bridge_layers(d, anchor, b.last);
    const int64_t n_vocab = (int64_t) d.head.size() / N_EMBD;
    for (int32_t t = 0; t < N_ROWS; ++t) {
        b.logits.push_back(matvec(d.head, output_norm(d, h[t]), n_vocab));
    }
    return b;
}

static std::set<int64_t> top_k_ids(const vec & logits) {
    std::vector<int64_t> ids(logits.size());
    std::iota(ids.begin(), ids.end(), 0);
    std::partial_sort(ids.begin(), ids.begin() + TOP_K, ids.end(), [&](int64_t a, int64_t b) { return logits[a] > logits[b]; });
    return std::set<int64_t>(ids.begin(), ids.begin() + TOP_K);
}

// score of candidate cand when the predecessor row holds token pred: the last write redone with the exact code of pred
static double refined_score(const host_draft & d, const host_row_state & s, int64_t cand, int64_t pred) {
    const vec refined = written(d.write[N_LAYER - 1], s, bridge_message(d, s.query, token_code(d, pred)));
    return dot(row_of(d.head, cand, N_EMBD), output_norm(d, refined));
}

static bool close(double value, double expected) {
    return std::fabs(value - expected) <= REL_TOL * std::max(1.0, std::fabs(expected));
}

static std::vector<int64_t> row_candidates(const float * row) {
    std::vector<int64_t> cand(TOP_K);
    for (int32_t k = 0; k < TOP_K; ++k) {
        cand[k] = (int64_t) row[k];
    }
    return cand;
}

// scores[k + j*TOP_K] is candidate k of the row under candidate j of the predecessor row
static bool check_row_scores(const host_draft & d, const host_row_state & s, const float * row, const std::vector<int64_t> & preds) {
    const std::vector<int64_t> cand = row_candidates(row);
    for (int32_t j = 0; j < TOP_K; ++j) {
        for (int32_t k = 0; k < TOP_K; ++k) {
            const double expected = refined_score(d, s, cand[k], preds[j]);
            if (!close(row[TOP_K + j * TOP_K + k], expected)) {
                fprintf(stderr, "  candidate %lld under predecessor %lld: %.7f, expected %.7f\n", (long long) cand[k], (long long) preds[j], row[TOP_K + j * TOP_K + k], expected);
                return false;
            }
        }
    }
    return true;
}

static bool row_is_zero(const float * row) {
    return std::all_of(row, row + N_EMBD, [](float v) { return v == 0.0f; });
}

static bool check_row(const host_draft & d, const host_block & b, const float * row, int32_t t, const std::vector<int64_t> & preds, bool check_candidates) {
    const std::vector<int64_t> cand = row_candidates(row);
    if (check_candidates && std::set<int64_t>(cand.begin(), cand.end()) != top_k_ids(b.logits[t])) {
        fprintf(stderr, "  row %d: candidates are not the reference top-%d\n", t, TOP_K);
        return false;
    }
    return check_row_scores(d, b.last[t], row, preds);
}

// row 0 is empty; row t holds the top-k ids of row t and their scores under every candidate of row t - 1 (the anchor for row 1)
static bool check_block(const host_draft & d, const float * lattice, int64_t anchor, bool check_candidates) {
    const host_block b = forward_block(d, anchor);
    if (!row_is_zero(lattice)) {
        fprintf(stderr, "  anchor row is not empty\n");
        return false;
    }
    std::vector<int64_t> preds(TOP_K, anchor);
    for (int32_t t = 1; t < N_ROWS; ++t) {
        const float * row = lattice + t * N_EMBD;
        if (!check_row(d, b, row, t, preds, check_candidates)) {
            return false;
        }
        preds = row_candidates(row);
    }
    return true;
}

static bool check_blocks(const host_draft & d, const float * lattice, int32_t n_blocks, bool check_candidates) {
    for (int32_t b = 0; b < n_blocks; ++b) {
        if (!check_block(d, lattice + (size_t) b * N_ROWS * N_EMBD, ANCHORS[b], check_candidates)) {
            return false;
        }
    }
    return true;
}

static void add_block(llama_batch & batch, int32_t b) {
    for (int32_t t = 0; t < N_ROWS; ++t) {
        const int32_t i = b * N_ROWS + t;
        batch.token[i] = t == 0 ? ANCHORS[b] : (llama_token) MASK_ID;
        batch.pos[i] = t;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = b;
        batch.logits[i] = false;
    }
}

// one [anchor, mask...] block per sequence in a single decode, as the DFlash drafter submits them
static bool decode_blocks(llama_context * ctx, int32_t n_blocks) {
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_batch batch = llama_batch_init(n_blocks * N_ROWS, 0, 1);
    batch.n_tokens = n_blocks * N_ROWS;
    for (int32_t b = 0; b < n_blocks; ++b) {
        add_block(batch, b);
    }
    const bool decoded = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return decoded;
}

static llama_context_params block_context_params(int32_t n_ubatch, llama_context * ctx_other) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2 * N_EMBD;
    cparams.n_batch = n_ubatch;
    cparams.n_ubatch = n_ubatch;
    cparams.n_seq_max = MAX_BLOCKS;
    cparams.ctx_other = ctx_other;
    return cparams;
}

static llama_model * load_model(const char * path, bool use_extra_bufts) {
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.use_extra_bufts = use_extra_bufts;
    return llama_model_load_from_file(path, mparams);
}

// a target model, its context, and the DSpine draft that borrows its embeddings and head
struct dspine_setup {
    llama_model   * target  = nullptr;
    llama_model   * draft   = nullptr;
    llama_context * ctx_tgt = nullptr;
    llama_context * ctx_dft = nullptr;

    ~dspine_setup() {
        llama_free(ctx_dft);
        llama_free(ctx_tgt);
        llama_model_free(draft);
        llama_model_free(target);
    }
};

static bool open_setup(dspine_setup & s, int64_t n_vocab_target, ggml_type head_type, bool use_extra_bufts, int32_t n_ubatch_dft) {
    if (!write_target(n_vocab_target, head_type) || !write_draft(draft_kind::valid, true)) {
        return false;
    }
    s.target  = load_model(TARGET_PATH, use_extra_bufts);
    s.draft   = load_model(DRAFT_PATH, true);
    s.ctx_tgt = s.target ? llama_init_from_model(s.target, block_context_params(MAX_BLOCKS * N_ROWS, nullptr)) : nullptr;
    s.ctx_dft = s.draft && s.ctx_tgt ? llama_init_from_model(s.draft, block_context_params(n_ubatch_dft, s.ctx_tgt)) : nullptr;
    return s.ctx_dft != nullptr;
}

static bool decode_and_check(dspine_setup & s, int32_t n_blocks, bool check_candidates) {
    llama_set_causal_attn(s.ctx_dft, true);
    llama_set_embeddings_nextn(s.ctx_dft, true, false);
    if (!decode_blocks(s.ctx_dft, n_blocks) || !llama_get_embeddings_nextn(s.ctx_dft)) {
        return false;
    }
    llama_model * plain_target = load_model(TARGET_PATH, false);
    const bool ok = plain_target && check_blocks(read_host_draft(s.draft, plain_target), llama_get_embeddings_nextn(s.ctx_dft), n_blocks, check_candidates);
    llama_model_free(plain_target);
    return ok;
}

static bool check_lattice(int32_t n_blocks) {
    dspine_setup s;
    return open_setup(s, N_VOCAB, GGML_TYPE_F32, true, n_blocks * N_ROWS) && decode_and_check(s, n_blocks, true);
}

// the codebook rows are indexed by the head's candidate ids, so a target head of another width stops the draft context
static bool check_head_width_rejected() {
    dspine_setup s;
    return !open_setup(s, N_VOCAB_WIDE, GGML_TYPE_F32, true, N_ROWS) && s.ctx_tgt != nullptr;
}

// get_rows on a repacked head reads interleaved rows, so a repacked target head stops the draft context; a plain quantized head
// works, and its candidates come from quantized activations, so only the scores are compared
static bool check_quantized_head(bool use_extra_bufts) {
    dspine_setup s;
    const bool opened = open_setup(s, N_VOCAB, GGML_TYPE_Q4_0, use_extra_bufts, N_ROWS);
    if (s.target && s.target->has_backend_layout(s.target->output)) {
        return !opened && s.ctx_tgt != nullptr;
    }
    return opened && decode_and_check(s, 1, false);
}

static bool spec_init_rejects(common_params_speculative & params, uint32_t n_seq) {
    try {
        common_speculative_free(common_speculative_init(params, n_seq));
    } catch (const std::runtime_error &) {
        return true;
    }
    return false;
}

// the bridge needs whole blocks, so the blocks of all sequences must fit one draft ubatch
static bool check_spec_ubatch() {
    dspine_setup s;
    if (!open_setup(s, N_VOCAB, GGML_TYPE_F32, true, N_ROWS)) {
        return false;
    }
    common_params_speculative params;
    params.types = { COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH };
    params.draft.ctx_tgt = s.ctx_tgt;
    params.draft.ctx_dft = s.ctx_dft;
    params.draft.n_max = N_ROWS - 1;
    const bool rejected_two = spec_init_rejects(params, MAX_BLOCKS);
    const bool rejected_one = spec_init_rejects(params, 1);
    printf("%s: two sequences rejected %d, one sequence rejected %d\n", __func__, rejected_two, rejected_one);
    return rejected_two && !rejected_one;
}

int main() {
    llama_backend_init();
    const struct {
        draft_kind   kind;
        const char * name;
        bool         loads;
    } load_cases[] = {
        { draft_kind::valid,            "valid DSpine draft",          true  },
        { draft_kind::zero_eps,         "zero bridge eps",             false },
        { draft_kind::top_k_not_pow2,   "head top-k not a power of 2", false },
        { draft_kind::top_k_too_wide,   "lattice wider than n_embd",   false },
        { draft_kind::not_causal,       "non-causal draft",            false },
        { draft_kind::missing_codebook, "missing codebook",            false },
        { draft_kind::with_markov,      "DSpark Markov head",          false },
        { draft_kind::with_selector,    "DFlash2 selector metadata",   false },
    };

    int failures = 0;
    for (const auto & c : load_cases) {
        const bool loaded = header_loads(c.kind);
        if (loaded != c.loads) {
            fprintf(stderr, "FAIL: %s: expected %s, got %s\n", c.name, c.loads ? "load" : "reject", loaded ? "load" : "reject");
            ++failures;
        }
    }
    if (!check_lattice(1)) {
        fprintf(stderr, "FAIL: DSpine lattice of one block does not match the reference\n");
        ++failures;
    }
    if (!check_lattice(MAX_BLOCKS)) {
        fprintf(stderr, "FAIL: DSpine lattice of two blocks in one ubatch does not match the reference\n");
        ++failures;
    }
    if (!check_head_width_rejected()) {
        fprintf(stderr, "FAIL: a target head wider than the DSpine codebook did not stop the draft context\n");
        ++failures;
    }
    if (!check_spec_ubatch()) {
        fprintf(stderr, "FAIL: the speculative setup did not check that the DSpine blocks fit one draft ubatch\n");
        ++failures;
    }
    if (!check_quantized_head(false) || !check_quantized_head(true)) {
        fprintf(stderr, "FAIL: a quantized target head gave wrong scores, or a repacked one did not stop the draft context\n");
        ++failures;
    }

    printf("%s: %d failures\n", __func__, failures);
    return failures == 0 ? 0 : 1;
}
