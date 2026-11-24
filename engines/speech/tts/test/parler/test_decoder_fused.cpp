// Model-free decoder coverage for the CPU decode optimizations, run once per
// supported projection quantization (f32, f16, q8_0, q6_k): fused q|k|v +
// stacked LM-head projections must reproduce the separate-weight logits
// byte-for-byte on the decode step (and within float tolerance on prefill,
// where the wider GEMM may tile differently), and on a host backend the
// prefill/step logits hand out a zero-copy view into the graph buffer.
// The CPU flash-attention path (F32 KV, non-transposed cross V) must match
// the manual softmax(QK^T)V path within float tolerance on the same weights.
// Also pins the fusion gate: mmap-backed CPU, GPU, and measure loads fuse;
// the CPU allocate-and-stream fallback and PARLER_NO_FUSED do not.
// Runs on a synthetic two-layer model, so it needs no GGUF fixture.

#include "parler/internal.h"
#include "backend_selection.h"
#include "../test_env_portable.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace tts_cpp::parler::detail;

static int g_failures = 0;

#define CHECK(cond, msg)                                                       \
    do {                                                                       \
        if (!(cond)) {                                                         \
            ++g_failures;                                                      \
            fprintf(stderr, "FAIL: %s  (%s:%d)\n", msg, __FILE__, __LINE__);   \
        }                                                                      \
    } while (0)

namespace {

// q6_k superblocks span 256 values, so the fused dimension must be a multiple.
constexpr int D          = 256;
constexpr int N_HEAD     = 4;
constexpr int N_LAYER    = 2;
constexpr int D_FF       = 256;
constexpr int N_CB       = 3;
constexpr int VOCAB      = 32;
constexpr int CROSS_LEN  = 5;
constexpr int N_CTX      = 64;
constexpr int MAX_POS    = 64;
constexpr int BOS_ID     = 30;
constexpr int PROMPT_VOCAB = 8;
constexpr int N_THREADS  = 2;
constexpr int N_STEPS    = 3;
// FA reorders the softmax reduction (online, per KV row) against the manual
// path; logits are O(1) here, so rounding stays orders of magnitude below this.
constexpr float FA_TOLERANCE = 1e-4f;
// f16 projections round their activations to half precision, so a last-bit
// difference upstream can flip one rounding and move a logit by ~one f16 ulp.
constexpr float FA_TOLERANCE_F16 = 4e-3f;

// xorshift-based deterministic fill: the values only need to be finite,
// spread out, and identical across the two runs.
struct det_rng {
    uint32_t state = 0x9E3779B9u;
    float next() {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return ((float) (state & 0xFFFFFF) / (float) 0xFFFFFF - 0.5f) * 0.2f;
    }
};

std::vector<float> random_host_values(ggml_tensor * t, det_rng & rng, float bias) {
    std::vector<float> host((size_t) ggml_nelements(t));
    for (float & v : host) v = bias + rng.next();
    return host;
}

void fill_f32_tensor(ggml_tensor * t, det_rng & rng, float bias) {
    const std::vector<float> host = random_host_values(t, rng, bias);
    ggml_backend_tensor_set(t, host.data(), 0, host.size() * sizeof(float));
}

// Projection weights carry the type under test; quantization runs on the host
// values so both decode runs see identical bytes.
void fill_typed_tensor(ggml_tensor * t, det_rng & rng) {
    const std::vector<float> host = random_host_values(t, rng, 0.0f);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, host.data(), 0, host.size() * sizeof(float));
        return;
    }
    if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> h16(host.size());
        ggml_fp32_to_fp16_row(host.data(), h16.data(), (int64_t) host.size());
        ggml_backend_tensor_set(t, h16.data(), 0, h16.size() * sizeof(ggml_fp16_t));
        return;
    }
    std::vector<uint8_t> quantized(ggml_nbytes(t));
    ggml_quantize_chunk(t->type, host.data(), quantized.data(), 0,
                        t->ne[1], t->ne[0], nullptr);
    ggml_backend_tensor_set(t, quantized.data(), 0, quantized.size());
}

ggml_tensor * new_f32(ggml_context * ctx, int64_t ne0, int64_t ne1) {
    return ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
}

// [D, CROSS_LEN] token-major -> [CROSS_LEN, D]
std::vector<float> transpose_cross(const std::vector<float> & v) {
    std::vector<float> t(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        const size_t tok = i / D, ch = i % D;
        t[ch * CROSS_LEN + tok] = v[i];
    }
    return t;
}

// The same cross V values in either layout: non-transposed [D, CROSS_LEN]
// for flash attention, transposed [CROSS_LEN, D] for the manual path.
void fill_cross_v(ggml_tensor * t, det_rng & rng, bool transposed) {
    const std::vector<float> v = random_host_values(t, rng, 0.0f);
    const std::vector<float> host = transposed ? transpose_cross(v) : v;
    ggml_backend_tensor_set(t, host.data(), 0, host.size() * sizeof(float));
}

bool build_model(parler_model & model, ggml_type wtype, bool fa = false) {
    parler_hparams & hp = model.hparams;
    hp.dec_n_layer    = N_LAYER;
    hp.dec_d_model    = D;
    hp.dec_n_head     = N_HEAD;
    hp.dec_d_ff       = D_FF;
    hp.n_codebooks    = N_CB;
    hp.dec_vocab      = VOCAB;
    hp.dec_ln_eps     = 1e-5f;
    hp.bos_id         = BOS_ID;
    hp.max_position   = MAX_POS;
    hp.n_ctx          = N_CTX;
    hp.gen_max_length = 8;

    ::tts_cpp::detail::ensure_backends_loaded();
    model.backend = ::tts_cpp::detail::init_cpu_backend();
    if (!model.backend) return false;
    model.on_gpu  = false;
    model.use_fa  = fa;
    model.kv_type = fa ? parler_fa_kv_type(false) : GGML_TYPE_F32;

    {
        ggml_init_params ip = { 64 * ggml_tensor_overhead(), nullptr, /*no_alloc=*/ true };
        model.ctx_w = ggml_init(ip);
        if (!model.ctx_w) return false;
    }
    model.dec_layers.resize(N_LAYER);
    for (auto & l : model.dec_layers) {
        l.attn_norm_w  = new_f32(model.ctx_w, D, 1);
        l.attn_norm_b  = new_f32(model.ctx_w, D, 1);
        l.q            = ggml_new_tensor_2d(model.ctx_w, wtype, D, D);
        l.k            = ggml_new_tensor_2d(model.ctx_w, wtype, D, D);
        l.v            = ggml_new_tensor_2d(model.ctx_w, wtype, D, D);
        l.o            = new_f32(model.ctx_w, D, D);
        l.cross_norm_w = new_f32(model.ctx_w, D, 1);
        l.cross_norm_b = new_f32(model.ctx_w, D, 1);
        l.cq           = new_f32(model.ctx_w, D, D);
        l.co           = new_f32(model.ctx_w, D, D);
        l.ffn_norm_w   = new_f32(model.ctx_w, D, 1);
        l.ffn_norm_b   = new_f32(model.ctx_w, D, 1);
        l.up           = new_f32(model.ctx_w, D, D_FF);
        l.down         = new_f32(model.ctx_w, D_FF, D);
    }
    model.embed_positions   = new_f32(model.ctx_w, D, MAX_POS);
    model.embed_prompts     = new_f32(model.ctx_w, D, PROMPT_VOCAB);
    model.dec_output_norm_w = new_f32(model.ctx_w, D, 1);
    model.dec_output_norm_b = new_f32(model.ctx_w, D, 1);
    model.dec_embed.resize(N_CB);
    model.lm_heads.resize(N_CB);
    for (int k = 0; k < N_CB; ++k) {
        model.dec_embed[k] = new_f32(model.ctx_w, D, BOS_ID + 2);
        model.lm_heads[k]  = ggml_new_tensor_2d(model.ctx_w, wtype, D, VOCAB);
    }
    model.buffer_w = ggml_backend_alloc_ctx_tensors(model.ctx_w, model.backend);
    if (!model.buffer_w) return false;

    det_rng rng;
    for (ggml_tensor * t = ggml_get_first_tensor(model.ctx_w); t;
         t = ggml_get_next_tensor(model.ctx_w, t)) {
        if (t->type != GGML_TYPE_F32) {
            fill_typed_tensor(t, rng);
        } else {
            fill_f32_tensor(t, rng, t->ne[1] == 1 ? 1.0f : 0.0f);
        }
    }

    {
        ggml_init_params ip = { 8 * ggml_tensor_overhead(), nullptr, /*no_alloc=*/ true };
        model.ctx_kv = ggml_init(ip);
        if (!model.ctx_kv) return false;
        const int64_t rows = (int64_t) N_CTX * N_LAYER;
        model.memory_k = ggml_new_tensor_2d(model.ctx_kv, model.kv_type, D, rows);
        model.memory_v = ggml_new_tensor_2d(model.ctx_kv, model.kv_type, D, rows);
        model.buffer_kv = ggml_backend_alloc_ctx_tensors(model.ctx_kv, model.backend);
        if (!model.buffer_kv) return false;
    }

    {
        ggml_init_params ip = { (size_t) (2 * N_LAYER + 2) * ggml_tensor_overhead(),
                                nullptr, /*no_alloc=*/ true };
        model.ctx_cross = ggml_init(ip);
        if (!model.ctx_cross) return false;
        model.cross_k.resize(N_LAYER);
        model.cross_v_t.resize(N_LAYER);
        for (int l = 0; l < N_LAYER; ++l) {
            model.cross_k[l]   = new_f32(model.ctx_cross, D, CROSS_LEN);
            model.cross_v_t[l] = fa ? new_f32(model.ctx_cross, D, CROSS_LEN)
                                    : new_f32(model.ctx_cross, CROSS_LEN, D);
        }
        model.buffer_cross = ggml_backend_alloc_ctx_tensors(model.ctx_cross, model.backend);
        if (!model.buffer_cross) return false;
        for (int l = 0; l < N_LAYER; ++l) {
            fill_f32_tensor(model.cross_k[l], rng, 0.0f);
            fill_cross_v(model.cross_v_t[l], rng, !fa);
        }
        model.cross_len = CROSS_LEN;
    }
    return true;
}

// prefill + N_STEPS teacher-forced steps; returns one logits block per call.
bool run_decode(const parler_model & model, ggml_gallocr_t allocr,
                std::vector<std::vector<float>> & out, bool & zero_copy) {
    const size_t n = (size_t) VOCAB * N_CB;
    out.clear();
    zero_copy = true;

    const std::vector<int32_t> prompt_ids  = { 0, 1, 2 };
    const std::vector<int32_t> start_frame((size_t) N_CB, BOS_ID);
    parler_step_logits logits;
    int n_past = 0;
    if (!parler_dec_prefill(model, prompt_ids, start_frame, allocr, N_THREADS,
                            logits, n_past)) return false;
    zero_copy = zero_copy && logits.copy.empty();
    out.emplace_back(logits.view, logits.view + n);

    for (int s = 0; s < N_STEPS; ++s) {
        const std::vector<int32_t> frame = { (int32_t) (s * 3 + 1) % VOCAB,
                                             (int32_t) (s * 3 + 2) % VOCAB,
                                             (int32_t) (s * 3 + 3) % VOCAB };
        if (!parler_dec_step(model, frame, n_past, allocr, N_THREADS, logits)) return false;
        n_past++;
        zero_copy = zero_copy && logits.copy.empty();
        out.emplace_back(logits.view, logits.view + n);
    }
    return true;
}

float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

void run_type_case(ggml_type wtype) {
    const char * tn = ggml_type_name(wtype);
    fprintf(stderr, "parler decoder fused: projections %s\n", tn);

    parler_model model;
    if (!build_model(model, wtype)) {
        ++g_failures;
        fprintf(stderr, "FAIL: %s model construction\n", tn);
        parler_free_model(model);
        return;
    }
    ggml_gallocr_t allocr = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(model.backend));
    if (!allocr) {
        ++g_failures;
        fprintf(stderr, "FAIL: %s allocator creation\n", tn);
        parler_free_model(model);
        return;
    }

    std::vector<std::vector<float>> separate, fused;
    bool zero_copy = false;
    CHECK(run_decode(model, allocr, separate, zero_copy), "separate-weight decode runs");
    CHECK(zero_copy, "host backend hands out zero-copy logits views");

    CHECK(parler_fuse_decode_weights(model), "fusing the decode weights succeeds");
    bool wired = model.lm_head_stacked != nullptr;
    for (const auto & l : model.dec_layers) wired = wired && l.qkv != nullptr;
    CHECK(wired, "fusion wires qkv on every layer and the stacked LM head");

    CHECK(run_decode(model, allocr, fused, zero_copy), "fused-weight decode runs");
    if (separate.size() == fused.size() && !separate.empty()) {
        // prefill (N > 1) may tile the wider GEMM differently; the N=1 decode
        // steps are per-row dot products and must not move a single bit
        CHECK(max_abs_diff(separate[0], fused[0]) <= 1e-5f,
              "fused prefill logits match within float tolerance");
        for (size_t s = 1; s < separate.size(); ++s) {
            CHECK(std::memcmp(separate[s].data(), fused[s].data(),
                              separate[s].size() * sizeof(float)) == 0,
                  "fused step logits are byte-identical to separate weights");
        }
    } else {
        CHECK(false, "both decodes produced the same number of logits blocks");
    }

    ggml_gallocr_free(allocr);
    parler_free_model(model);
}

float max_block_diff(const std::vector<std::vector<float>> & a,
                     const std::vector<std::vector<float>> & b) {
    float m = 0.0f;
    for (size_t s = 0; s < a.size(); ++s) m = std::max(m, max_abs_diff(a[s], b[s]));
    return m;
}

bool decode_model(ggml_type wtype, bool fa, std::vector<std::vector<float>> & out) {
    parler_model model;
    bool ok = build_model(model, wtype, fa);
    ggml_gallocr_t allocr = ok ? ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(model.backend)) : nullptr;
    bool zero_copy = false;
    ok = ok && allocr && run_decode(model, allocr, out, zero_copy);
    if (allocr) ggml_gallocr_free(allocr);
    parler_free_model(model);
    return ok;
}

void run_fa_case(ggml_type wtype) {
    fprintf(stderr, "parler decoder cpu flash attention: projections %s\n", ggml_type_name(wtype));
    std::vector<std::vector<float>> manual, fa;
    CHECK(decode_model(wtype, false, manual), "manual-attention decode runs");
    CHECK(decode_model(wtype, true, fa), "flash-attention decode runs");
    if (manual.size() != fa.size() || manual.empty()) {
        CHECK(false, "both attention paths produced the same number of logits blocks");
        return;
    }
    const float diff = max_block_diff(manual, fa);
    const float bar  = wtype == GGML_TYPE_F16 ? FA_TOLERANCE_F16 : FA_TOLERANCE;
    fprintf(stderr, "  max |manual - fa| = %.3g (bar %.0e)\n", diff, bar);
    CHECK(diff <= bar, "CPU flash attention matches the manual path within float tolerance");
}

void test_fa_kv_type() {
    CHECK(parler_fa_kv_type(false) == GGML_TYPE_F32, "CPU flash attention keeps an F32 KV cache");
    CHECK(parler_fa_kv_type(true) == GGML_TYPE_F16, "GPU flash attention stores an F16 KV cache");
}

void test_fusion_gate() {
    parler_model m;
    m.on_gpu  = false;
    m.map_buf = nullptr;
    CHECK(!parler_should_fuse_decode_weights(m, false),
          "CPU allocate-and-stream load does not fuse");
    CHECK(parler_should_fuse_decode_weights(m, true), "measure-mode load fuses");
    m.on_gpu = true;
    CHECK(parler_should_fuse_decode_weights(m, false), "GPU load fuses");

    m.on_gpu = false;
    // ggml_backend_cpu_buffer_from_ptr asserts TENSOR_ALIGNMENT (32 bytes)
    alignas(64) static uint8_t backing[64];
    m.map_buf = ggml_backend_cpu_buffer_from_ptr(backing, sizeof(backing));
    CHECK(m.map_buf != nullptr && parler_should_fuse_decode_weights(m, false),
          "mmap-backed CPU load fuses");

    setenv("PARLER_NO_FUSED", "1", 1);
    CHECK(!parler_should_fuse_decode_weights(m, false) &&
              !parler_should_fuse_decode_weights(m, true),
          "PARLER_NO_FUSED disables fusion on every path");
    unsetenv("PARLER_NO_FUSED");

    ggml_backend_buffer_free(m.map_buf);
    m.map_buf = nullptr;
}

} // namespace

int main() {
    test_fusion_gate();
    test_fa_kv_type();
    const ggml_type types[] = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q6_K };
    for (ggml_type t : types) run_type_case(t);
    for (ggml_type t : types) run_fa_case(t);

    if (g_failures == 0) {
        fprintf(stderr, "parler decoder fused: PASS\n");
        return 0;
    }
    fprintf(stderr, "parler decoder fused: %d failure(s)\n", g_failures);
    return 1;
}
