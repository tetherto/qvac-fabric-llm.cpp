#pragma once
// A tiny but complete audio8-lm GGUF, synthesized on the fly for tests that
// need a loadable language model without a fixture.

#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <string>
#include <vector>

namespace audio8_test {

// ── Tiny synthetic audio8-lm GGUF ───────────────────────────────────────────
// The smallest hparam set the loader, the graph builders, and the RoPE tables
// accept: every dimension real models scale up, none of the structure changed.

struct tiny_lm {
    int hidden = 32, depth = 2, n_head = 4, n_kv = 2, head_dim = 8, inter = 48;
    int vocab = 96;
    int fast_depth = 1, fast_n_head = 4, fast_n_kv = 2, fast_head_dim = 8, fast_inter = 48;
    int num_codebooks = 4, codebook_size = 24;
    int semantic_begin = 8, semantic_end = 31, eos = 32, pad = 0;
    int max_seq_len = 48, ras_window = 6;
    // Projection weights and biases get distinct values instead of 0.01, so a
    // test can see a row land in the wrong place; matrices whose rows are whole
    // q8_0 blocks can also be stored in q8_0.
    bool varied = false;
    bool q8_matrices = false;
    // Keeps every wk in f32 while the other matrices go to q8_0, so q, k and v
    // cannot be stacked.
    bool f32_keys = false;
    // Rows added to every wk, so the file disagrees with its head geometry.
    int extra_key_rows = 0;
    // A one-row tensor written under this name, when set.
    std::string stray_tensor;
};

inline float varied_value(const char * name, int64_t index) {
    uint32_t h = 2166136261u;
    for (const char * c = name; *c; ++c) h = (h ^ (uint8_t) *c) * 16777619u;
    h = (h ^ (uint32_t) index) * 2654435761u;
    return (float) ((int) ((h >> 8) % 2001u) - 1000) * 1e-4f;
}

inline void fill(ggml_tensor * t, const char * name, bool varied) {
    float * d = (float *) t->data;
    for (int64_t i = 0; i < ggml_nelements(t); ++i) d[i] = varied ? varied_value(name, i) : 0.01f;
}

inline void add_f32(gguf_context * g, ggml_context * ctx, const char * name,
             std::initializer_list<int64_t> ne, bool varied = false) {
    ggml_tensor * t = ggml_new_tensor(ctx, GGML_TYPE_F32, (int) ne.size(),
                                      std::vector<int64_t>(ne).data());
    ggml_set_name(t, name);
    fill(t, name, varied);
    gguf_add_tensor(g, t);
}

inline void add_q8(gguf_context * g, ggml_context * ctx, const char * name,
                   int64_t cols, int64_t rows) {
    std::vector<float> values((size_t) (cols * rows));
    for (size_t i = 0; i < values.size(); ++i) values[i] = varied_value(name, (int64_t) i);
    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, cols, rows);
    ggml_set_name(t, name);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, values.data(), t->data, 0, rows, cols, nullptr);
    gguf_add_tensor(g, t);
}

inline void add_ones(gguf_context * g, ggml_context * ctx, const char * name, int64_t n) {
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_set_name(t, name);
    float * d = (float *) t->data;
    for (int64_t i = 0; i < n; ++i) d[i] = 1.0f;
    gguf_add_tensor(g, t);
}

// A norm weight: 1 when p.varied, so activations keep their spread.
inline void add_norm(gguf_context * g, ggml_context * ctx, const tiny_lm & p,
                     const char * name, int64_t n) {
    if (p.varied) {
        add_ones(g, ctx, name, n);
        return;
    }
    add_f32(g, ctx, name, {n});
}

constexpr float TINY_ROPE_THETA = 10000.0f;

// Real RoPE planes when p.varied, so attention scores depend on the keys.
inline void add_rope(gguf_context * g, ggml_context * ctx, const tiny_lm & p,
                     const char * name, int64_t half, int64_t positions, bool sine) {
    if (!p.varied) {
        add_f32(g, ctx, name, {half, positions});
        return;
    }
    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, half, positions);
    ggml_set_name(t, name);
    float * d = (float *) t->data;
    for (int64_t pos = 0; pos < positions; ++pos) {
        for (int64_t i = 0; i < half; ++i) {
            const double angle = pos * std::pow(TINY_ROPE_THETA, -(double) i / (double) half);
            d[pos * half + i] = (float) (sine ? std::sin(angle) : std::cos(angle));
        }
    }
    gguf_add_tensor(g, t);
}

// A projection: its own values when p.varied, q8_0 when asked and the rows are
// whole blocks.
inline bool is_key_weight(const char * name) {
    const std::string text(name);
    return text.size() >= 2 && text.compare(text.size() - 2, 2, "wk") == 0;
}

inline void add_weight(gguf_context * g, ggml_context * ctx, const tiny_lm & p,
                       const char * name, std::initializer_list<int64_t> ne) {
    const std::vector<int64_t> dims(ne);
    const bool keep_f32 = p.f32_keys && is_key_weight(name);
    if (p.q8_matrices && !keep_f32 && dims.size() == 2 &&
        dims[0] % ggml_blck_size(GGML_TYPE_Q8_0) == 0) {
        add_q8(g, ctx, name, dims[0], dims[1]);
        return;
    }
    add_f32(g, ctx, name, ne, p.varied);
}

inline std::string write_tiny_lm_gguf(const tiny_lm & p, const std::string & path,
                                      bool with_vocab = true, bool only_meta = false) {
    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", "audio8-lm");
    auto u32 = [&](const char * k, int v) {
        gguf_set_val_u32(g, (std::string("audio8.lm.") + k).c_str(), (uint32_t) v);
    };
    auto f32 = [&](const char * k, float v) {
        gguf_set_val_f32(g, (std::string("audio8.lm.") + k).c_str(), v);
    };
    auto b = [&](const char * k, bool v) {
        gguf_set_val_bool(g, (std::string("audio8.lm.") + k).c_str(), v);
    };
    u32("depth", p.depth);            u32("hidden", p.hidden);
    u32("n_head", p.n_head);          u32("n_kv", p.n_kv);
    u32("head_dim", p.head_dim);      u32("inter", p.inter);
    u32("vocab", p.vocab);
    u32("fast_depth", p.fast_depth);  u32("fast_hidden", p.hidden);
    u32("fast_n_head", p.fast_n_head); u32("fast_n_kv", p.fast_n_kv);
    u32("fast_head_dim", p.fast_head_dim); u32("fast_inter", p.fast_inter);
    u32("num_codebooks", p.num_codebooks); u32("codebook_size", p.codebook_size);
    u32("semantic_begin", p.semantic_begin); u32("semantic_end", p.semantic_end);
    u32("eos", p.eos);                u32("pad", p.pad);
    u32("max_seq_len", p.max_seq_len); u32("ras_window", p.ras_window);
    f32("rope_theta", 10000.0f);      f32("rms_eps", 1e-5f);
    f32("ras_top_p", 0.9f);           f32("ras_temperature", 0.7f);
    b("norm_fast_input", true);       b("qkv_bias", true);
    b("fast_qkv_bias", false);

    if (with_vocab) {
        const char * toks[] = {"<pad>", "a", "b", "c"};
        gguf_set_arr_str(g, "tokenizer.ggml.tokens", toks, 4);
        const char * merges[] = {"a b"};
        gguf_set_arr_str(g, "tokenizer.ggml.merges", merges, 1);
        const int32_t added[] = {0};
        gguf_set_arr_data(g, "tokenizer.ggml.added_token_ids", GGUF_TYPE_INT32, added, 1);
    }

    ggml_init_params ip = { 16u * 1024 * 1024, nullptr, /*no_alloc=*/false };
    ggml_context * ctx = ggml_init(ip);

    add_f32(g, ctx, "lm/tok_emb",      {p.hidden, p.vocab}, p.varied);
    add_f32(g, ctx, "lm/codebook_emb", {p.hidden, (int64_t) p.num_codebooks * p.codebook_size}, p.varied);
    add_norm(g, ctx, p, "lm/norm", p.hidden);
    add_f32(g, ctx, "lm/sem_head",     {p.hidden, p.codebook_size + 1}, p.varied);
    add_rope(g, ctx, p, "lm/rope_cos", p.head_dim / 2, p.max_seq_len, false);
    add_rope(g, ctx, p, "lm/rope_sin", p.head_dim / 2, p.max_seq_len, true);
    for (int i = 0; i < p.depth; ++i) {
        const std::string pre = "lm/blk/" + std::to_string(i) + "/";
        add_weight(g, ctx, p, (pre + "wq").c_str(),        {p.hidden, p.n_head * p.head_dim});
        add_weight(g, ctx, p, (pre + "wk").c_str(),        {p.hidden, p.n_kv * p.head_dim + p.extra_key_rows});
        add_weight(g, ctx, p, (pre + "wv").c_str(),        {p.hidden, p.n_kv * p.head_dim});
        add_weight(g, ctx, p, (pre + "wo").c_str(),        {p.n_head * p.head_dim, p.hidden});
        add_f32(g, ctx, (pre + "wq_b").c_str(),      {p.n_head * p.head_dim}, p.varied);
        add_f32(g, ctx, (pre + "wk_b").c_str(),      {p.n_kv * p.head_dim}, p.varied);
        add_f32(g, ctx, (pre + "wv_b").c_str(),      {p.n_kv * p.head_dim}, p.varied);
        add_norm(g, ctx, p, (pre + "attn_norm").c_str(), p.hidden);
        add_weight(g, ctx, p, (pre + "w1").c_str(),        {p.hidden, p.inter});
        add_weight(g, ctx, p, (pre + "w2").c_str(),        {p.inter, p.hidden});
        add_weight(g, ctx, p, (pre + "w3").c_str(),        {p.hidden, p.inter});
        add_norm(g, ctx, p, (pre + "ffn_norm").c_str(), p.hidden);
    }
    add_f32(g, ctx, "fast/emb",      {p.hidden, p.codebook_size}, p.varied);
    add_norm(g, ctx, p, "fast/norm", p.hidden);
    add_f32(g, ctx, "fast/out",      {p.hidden, p.codebook_size}, p.varied);
    add_rope(g, ctx, p, "fast/rope_cos", p.fast_head_dim / 2, p.num_codebooks, false);
    add_rope(g, ctx, p, "fast/rope_sin", p.fast_head_dim / 2, p.num_codebooks, true);
    for (int i = 0; i < p.fast_depth; ++i) {
        const std::string pre = "fast/blk/" + std::to_string(i) + "/";
        add_weight(g, ctx, p, (pre + "wq").c_str(),        {p.hidden, p.fast_n_head * p.fast_head_dim});
        add_weight(g, ctx, p, (pre + "wk").c_str(),        {p.hidden, p.fast_n_kv * p.fast_head_dim});
        add_weight(g, ctx, p, (pre + "wv").c_str(),        {p.hidden, p.fast_n_kv * p.fast_head_dim});
        add_weight(g, ctx, p, (pre + "wo").c_str(),        {p.fast_n_head * p.fast_head_dim, p.hidden});
        add_norm(g, ctx, p, (pre + "attn_norm").c_str(), p.hidden);
        add_weight(g, ctx, p, (pre + "w1").c_str(),        {p.hidden, p.fast_inter});
        add_weight(g, ctx, p, (pre + "w2").c_str(),        {p.fast_inter, p.hidden});
        add_weight(g, ctx, p, (pre + "w3").c_str(),        {p.hidden, p.fast_inter});
        add_norm(g, ctx, p, (pre + "ffn_norm").c_str(), p.hidden);
    }

    if (!p.stray_tensor.empty()) {
        add_weight(g, ctx, p, p.stray_tensor.c_str(), {ggml_blck_size(GGML_TYPE_Q8_0), 1});
    }

    if (!gguf_write_to_file(g, path.c_str(), only_meta)) {
        std::fprintf(stderr, "FATAL: cannot write %s\n", path.c_str());
        std::exit(2);
    }
    ggml_free(ctx);
    gguf_free(g);
    return path;
}

}  // namespace audio8_test
