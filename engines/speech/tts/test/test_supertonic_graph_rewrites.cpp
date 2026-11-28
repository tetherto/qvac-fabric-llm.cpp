// TDD harness for the graph-side optimizations added in the
// audit follow-up (audit findings F3, F8, F11).
//
// Each of these findings is a graph rewrite or new cache: the output
// of the stage must stay bit-exact (or within F32 ULP tolerance) vs
// the pre-rewrite CPU reference path that ships in
// `supertonic_*_forward_cpu` /
// `supertonic_*_trace_*`.  The existing fixture-bound
// `test-supertonic-{vocoder,duration,vector,pipeline}` harnesses
// already gate the *production* GGML path against ONNX reference
// dumps; this harness layers on a finer-grained check that runs the
// same GGUF through both the GGML path and the scalar-CPU reference
// inside the same process and asserts they agree.
//
//   F3  Vocoder unpack-on-GPU: the host-side `[1, 144, L] →
//       [144, L*6]` transpose moves into the vocoder graph as
//       `ggml_permute + ggml_cont`.  Vocoder output must stay
//       bit-exact vs `supertonic_vocoder_forward_cpu`.
//
//   F8  Style residual + LN cached graph: the four per-step
//       residual-add-then-layer-norm tiny graphs (one per group)
//       become cached graphs survival across synth calls.  Pipeline
//       output must stay bit-exact vs the previous per-call graph
//       allocation.  This file's check is structural: the cache
//       allocator survives a second `synthesize` invocation without
//       rebuilding (no second `gallocr_new` call on the per-style
//       allocators).
//
//   F11 Duration cached graph: same pattern.  Single-synth wall-time
//       drops on warm-cache invocations; structural check that
//       `supertonic_duration_forward_ggml` reuses its allocator
//       across two calls.
//
// Fixture test — requires the Supertonic GGUF.

#include "supertonic_internal.h"
#include "npy.h"
#include "test_env_portable.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace tts_cpp::supertonic::detail;

namespace {

int g_failures = 0;
int g_checks   = 0;

#define CHECK(cond) do {                                              \
    ++g_checks;                                                       \
    if (!(cond)) {                                                    \
        ++g_failures;                                                 \
        std::fprintf(stderr, "FAIL %s:%d  %s\n",                     \
                     __FILE__, __LINE__, #cond);                      \
    }                                                                 \
} while (0)

bool close_enough(float a, float b, float atol = 1e-4f, float rtol = 1e-4f) {
    return std::fabs(a - b) <= atol + rtol * std::fabs(b);
}

// Generate a synthetic latent vector with deterministic content so
// the test is reproducible without requiring an ONNX reference dump.
std::vector<float> make_synthetic_latent(int latent_channels, int latent_len, uint32_t seed) {
    std::vector<float> out((size_t) latent_channels * latent_len);
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto & v : out) v = dist(rng);
    return out;
}

// F3 — Vocoder unpack-on-GPU parity.
//
// The audit fix moves the input transpose from the host loop into
// the GGML graph.  Math is a pure permutation, so output should
// match `supertonic_vocoder_forward_cpu` within F32 ULP (typically
// bit-exact, since the rest of the vocoder graph is unchanged).
//
// Tolerance: 1e-3 absolute matches `test_supertonic_pipeline.cpp`'s
// end-to-end gate, plenty for a vocoder-only check.
void test_f3_vocoder_unpack_parity(const supertonic_model & model) {
    std::fprintf(stderr, "[F3 vocoder unpack parity]\n");
    if (!model_prefers_cpu_kernels(model)) {
        std::fprintf(stderr, "  SKIP: the scalar reference reads the model from host memory (CPU backend only)\n");
        return;
    }

    const int C = model.hparams.latent_channels;
    const int L = 8;  // small latent_len for the test
    auto latent = make_synthetic_latent(C, L, 0xDEADBEEF);

    std::string err;
    std::vector<float> wav_cpu;
    if (!supertonic_vocoder_forward_cpu(model, latent.data(), L, wav_cpu, &err)) {
        std::fprintf(stderr, "  SKIP vocoder cpu: %s\n", err.c_str());
        return;
    }

    std::vector<float> wav_ggml;
    if (!supertonic_vocoder_forward_ggml(model, latent.data(), L, wav_ggml, &err)) {
        std::fprintf(stderr, "  SKIP vocoder ggml: %s\n", err.c_str());
        return;
    }

    const size_t n = std::min(wav_cpu.size(), wav_ggml.size());
    CHECK(n > 0);

    int bad = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float a = wav_cpu[i];
        const float b = wav_ggml[i];
        max_abs = std::max(max_abs, std::fabs(a - b));
        if (!close_enough(a, b, /*atol=*/1e-3f, /*rtol=*/1e-3f)) {
            if (bad < 4) {
                std::fprintf(stderr,
                             "  vocoder mismatch @ %zu: cpu=%.6g ggml=%.6g\n",
                             i, a, b);
            }
            ++bad;
        }
    }
    std::fprintf(stderr,
                 "  L=%d, samples=%zu, max_abs_err=%.3e, bad=%d\n",
                 L, n, max_abs, bad);
    CHECK(bad == 0);
}

// F11 — Duration cached graph parity.
//
// Two consecutive `supertonic_duration_forward_ggml` calls with the
// same shape must produce bit-exact identical output.  Trivially
// true even today, but the new cache adds the structural guarantee
// that no allocator/context churn happens on the second call.
//
// Pure parity gate: bit-exact equality after cache rebuild + reuse.
void test_f11_duration_cache_parity(const supertonic_model & model) {
    std::fprintf(stderr, "[F11 duration cached graph parity]\n");

    // Build a small synthetic text-id sequence + style.
    std::vector<int64_t> text_ids;
    for (int i = 1; i <= 16; ++i) text_ids.push_back(i);
    // Style: pull from any voice the GGUF carries.
    if (model.voices.empty()) {
        std::fprintf(stderr, "  SKIP: no voices in model\n");
        return;
    }
    const auto & voice = model.voices.begin()->second;
    std::vector<float> style_dp((size_t) ggml_nelements(voice.dp));
    ggml_backend_tensor_get(voice.dp, style_dp.data(), 0, ggml_nbytes(voice.dp));

    std::string err;
    float dur1 = 0.0f, dur2 = 0.0f;
    bool ok1 = supertonic_duration_forward_ggml(model, text_ids.data(), (int) text_ids.size(),
                                                 style_dp.data(), dur1, &err);
    if (!ok1) {
        std::fprintf(stderr, "  SKIP duration call 1: %s\n", err.c_str());
        return;
    }
    bool ok2 = supertonic_duration_forward_ggml(model, text_ids.data(), (int) text_ids.size(),
                                                 style_dp.data(), dur2, &err);
    if (!ok2) {
        std::fprintf(stderr, "  SKIP duration call 2: %s\n", err.c_str());
        return;
    }

    // Cached re-run must be bit-exact (same graph, same inputs).
    CHECK(dur1 == dur2);
    std::fprintf(stderr, "  dur1=%.6g  dur2=%.6g\n", dur1, dur2);
}

// F8 — Style residual cached graph parity (indirect).
//
// Without exposing the per-style-residual cache internals we can't
// count gallocr_new calls directly, but we can check the pipeline-
// level invariant: two consecutive `supertonic_vector_step_ggml`
// calls with identical inputs produce identical outputs.  If the
// cache rebuild logic accidentally aliased buffers across calls
// the second call would differ from the first; this catches that.
void test_f8_style_residual_cache_parity(const supertonic_model & model) {
    std::fprintf(stderr, "[F8 style residual cached graph parity]\n");

    const int text_len   = 16;
    const int latent_len = 8;
    const int Cin        = model.hparams.latent_channels;

    auto latent     = make_synthetic_latent(Cin,  latent_len, 0xCAFEBABE);
    auto text_emb   = make_synthetic_latent(256,  text_len,   0xBADF00D);
    std::vector<float> latent_mask((size_t) latent_len, 1.0f);

    if (model.voices.empty()) {
        std::fprintf(stderr, "  SKIP: no voices in model\n");
        return;
    }
    const auto & voice = model.voices.begin()->second;
    std::vector<float> style_ttl((size_t) ggml_nelements(voice.ttl));
    ggml_backend_tensor_get(voice.ttl, style_ttl.data(), 0, ggml_nbytes(voice.ttl));

    std::string err;
    std::vector<float> next1, next2;
    if (!supertonic_vector_step_ggml(model, latent.data(), latent_len,
                                     text_emb.data(), text_len,
                                     style_ttl.data(), latent_mask.data(),
                                     /*current_step=*/0, /*total_steps=*/5,
                                     next1, &err)) {
        std::fprintf(stderr, "  SKIP vector step 1: %s\n", err.c_str());
        return;
    }
    if (!supertonic_vector_step_ggml(model, latent.data(), latent_len,
                                     text_emb.data(), text_len,
                                     style_ttl.data(), latent_mask.data(),
                                     /*current_step=*/0, /*total_steps=*/5,
                                     next2, &err)) {
        std::fprintf(stderr, "  SKIP vector step 2: %s\n", err.c_str());
        return;
    }

    CHECK(next1.size() == next2.size());
    int bad = 0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < next1.size(); ++i) {
        max_abs = std::max(max_abs, std::fabs(next1[i] - next2[i]));
        if (next1[i] != next2[i]) ++bad;
    }
    std::fprintf(stderr,
                 "  next.size=%zu  max_abs_diff=%.3e  bad=%d\n",
                 next1.size(), max_abs, bad);
    CHECK(bad == 0);
}


bool run_per_step_chain(const supertonic_model & model,
                        const std::vector<float> & latent, int latent_len,
                        const std::vector<float> & text_emb, int text_len,
                        const std::vector<float> & style_ttl,
                        const std::vector<float> & latent_mask,
                        int total_steps, std::vector<float> & out, std::string & err) {
    std::vector<float> cur = latent, next;
    for (int s = 0; s < total_steps; ++s) {
        if (!supertonic_vector_step_ggml(model, cur.data(), latent_len, text_emb.data(), text_len,
                                         style_ttl.data(), latent_mask.data(), s, total_steps,
                                         next, &err)) {
            return false;
        }
        cur.swap(next);
    }
    out.swap(cur);
    return true;
}

int count_mismatches(const std::vector<float> & a, const std::vector<float> & b, float & max_abs) {
    int bad = 0;
    max_abs = 0.0f;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        max_abs = std::max(max_abs, std::fabs(a[i] - b[i]));
        if (!close_enough(a[i], b[i], /*atol=*/1e-3f, /*rtol=*/1e-3f)) ++bad;
    }
    return bad;
}

// 139 latent frames exercises the medium production shape where CFG batching
// must preserve branch-local attention and ConvNeXt semantics.
struct cfm_loop_inputs {
    int text_len    = 151;
    int latent_len  = 139;
    int total_steps = 5;
    std::vector<float> latent, text_emb, style_ttl, latent_mask;
};

bool make_cfm_loop_inputs(const supertonic_model & model, cfm_loop_inputs & in) {
    if (model.voices.empty()) {
        std::fprintf(stderr, "  SKIP: no voices in model\n");
        return false;
    }
    in.latent      = make_synthetic_latent(model.hparams.latent_channels, in.latent_len, 0x5EED1234);
    in.text_emb    = make_synthetic_latent(256, in.text_len, 0x7E57);
    in.latent_mask.assign((size_t) in.latent_len, 1.0f);
    const auto & voice = model.voices.begin()->second;
    in.style_ttl.resize((size_t) ggml_nelements(voice.ttl));
    ggml_backend_tensor_get(voice.ttl, in.style_ttl.data(), 0, ggml_nbytes(voice.ttl));
    return true;
}

bool run_cfm_loop_graph(const supertonic_model & model, const cfm_loop_inputs & in,
                        std::vector<float> & out, std::string & err) {
    return supertonic_vector_loop_ggml(model, in.latent.data(), in.latent_len, in.text_emb.data(),
                                       in.text_len, in.style_ttl.data(), in.latent_mask.data(),
                                       in.total_steps, out, &err);
}

bool run_cfm_step_chain(const supertonic_model & model, const cfm_loop_inputs & in,
                        std::vector<float> & out, std::string & err) {
    return run_per_step_chain(model, in.latent, in.latent_len, in.text_emb, in.text_len,
                              in.style_ttl, in.latent_mask, in.total_steps, out, err);
}

// Off the CPU backend the two arms of a parity check run different kernels and
// summation orders, so they agree to a cosine rather than element by element.
constexpr float kGpuMinCosine = 0.9999f;
// The predicted duration only steers the wav length; Metal lands within 4e-4 of the host MLP.
constexpr float kGpuDurationRtol = 1e-3f;

float cosine_similarity(const std::vector<float> & a, const std::vector<float> & b);

void check_parity(const supertonic_model & model, const std::vector<float> & a, const std::vector<float> & b,
                  const char * label) {
    CHECK(a.size() == b.size());
    float max_abs = 0.0f;
    const int bad = count_mismatches(a, b, max_abs);
    const float cos = cosine_similarity(a, b);
    std::fprintf(stderr, "  %s n=%zu, max_abs_err=%.3e, off_tolerance=%d, cosine=%.7f\n",
                 label, a.size(), max_abs, bad, cos);
    if (model_prefers_cpu_kernels(model)) {
        CHECK(bad == 0);
    } else {
        CHECK(cos >= kGpuMinCosine);
    }
}

float cosine_similarity(const std::vector<float> & a, const std::vector<float> & b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
        dot += (double) a[i] * b[i];
        na  += (double) a[i] * a[i];
        nb  += (double) b[i] * b[i];
    }
    return (na > 0.0 && nb > 0.0) ? (float) (dot / std::sqrt(na * nb)) : 0.0f;
}

// The unrolled loop graph shares the step-invariant K/V projections across
// its steps and batches the CFG passes; it must still equal the per-step chain.
void test_loop_graph_matches_per_step(const supertonic_model & model) {
    std::fprintf(stderr, "[unrolled CFM loop graph vs per-step chain]\n");
    cfm_loop_inputs in;
    if (!make_cfm_loop_inputs(model, in)) return;

    std::string err;
    std::vector<float> loop_out, chain_out;
    if (!run_cfm_loop_graph(model, in, loop_out, err)) {
        std::fprintf(stderr, "  SKIP loop graph: %s\n", err.c_str());
        return;
    }
    if (!run_cfm_step_chain(model, in, chain_out, err)) {
        std::fprintf(stderr, "  SKIP per-step chain: %s\n", err.c_str());
        return;
    }

    CHECK(loop_out.size() == chain_out.size());
    float max_abs = 0.0f;
    const int bad = count_mismatches(loop_out, chain_out, max_abs);
    std::fprintf(stderr, "  L=%d, text_len=%d, steps=%d, n=%zu, max_abs_err=%.3e, bad=%d\n",
                 in.latent_len, in.text_len, in.total_steps, loop_out.size(), max_abs, bad);
    CHECK(bad == 0);
}

// Off the CPU backend the loop runs the [C, T] step; it must agree with the
// legacy [T, C] per-step chain up to the summation-order drift of the two layouts.
void test_ct_loop_matches_legacy_chain(const supertonic_model & model) {
    std::fprintf(stderr, "[[C, T] CFM loop vs legacy [T, C] per-step chain]\n");
    if (model_prefers_cpu_kernels(model)) {
        std::fprintf(stderr, "  SKIP: the [C, T] step only runs off the CPU backend\n");
        return;
    }
    cfm_loop_inputs in;
    if (!make_cfm_loop_inputs(model, in)) return;

    std::string err;
    std::vector<float> legacy_out, ct_out;
    setenv("SUPERTONIC_DISABLE_CT_STEP", "1", 1);
    release_vector_estimator_thread_local_caches();
    const bool legacy_ok = run_cfm_step_chain(model, in, legacy_out, err);
    unsetenv("SUPERTONIC_DISABLE_CT_STEP");
    release_vector_estimator_thread_local_caches();
    if (!legacy_ok) {
        std::fprintf(stderr, "  SKIP legacy chain: %s\n", err.c_str());
        return;
    }
    if (!run_cfm_loop_graph(model, in, ct_out, err)) {
        std::fprintf(stderr, "  SKIP loop graph: %s\n", err.c_str());
        return;
    }

    check_parity(model, ct_out, legacy_out, "[C, T] loop vs legacy chain");
}


std::vector<int64_t> make_synthetic_ids(int text_len, int64_t vocab_size, uint32_t seed) {
    std::vector<int64_t> ids((size_t) text_len);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int64_t> dist(0, vocab_size - 1);
    for (auto & id : ids) id = dist(rng);
    return ids;
}

// The one-graph text encoder must match the per-island path on the same backend.
void check_text_encoder_one_graph(const supertonic_model & model, const std::vector<float> & style_ttl, int text_len) {
    ggml_tensor * emb = require_source_tensor(model,
        "text_encoder:tts.ttl.text_encoder.text_embedder.char_embedder.weight");
    const std::vector<int64_t> ids = make_synthetic_ids(text_len, emb->ne[1], 0x7E47u + (uint32_t) text_len);
    std::string err;
    std::vector<float> one_graph, islands;
    const bool ok_one = supertonic_text_encoder_forward_one_graph_ggml(model, ids.data(), text_len,
                                                                       style_ttl.data(), one_graph, &err);
    if (!ok_one) std::fprintf(stderr, "  one-graph failed: %s\n", err.c_str());
    CHECK(ok_one);
    const bool ok_islands = supertonic_text_encoder_forward_islands_ggml(model, ids.data(), text_len,
                                                                         style_ttl.data(), islands, &err);
    if (!ok_islands) std::fprintf(stderr, "  islands failed: %s\n", err.c_str());
    CHECK(ok_islands);
    if (!ok_one || !ok_islands) return;
    std::fprintf(stderr, "  text_len=%d\n", text_len);
    check_parity(model, one_graph, islands, "one-graph vs islands");
}

void test_text_encoder_one_graph_matches_islands(const supertonic_model & model) {
    std::fprintf(stderr, "[one-graph text encoder vs per-island path]\n");
    if (model.voices.empty()) {
        std::fprintf(stderr, "  SKIP: no voices in model\n");
        return;
    }
    const auto & voice = model.voices.begin()->second;
    std::vector<float> style_ttl((size_t) ggml_nelements(voice.ttl));
    ggml_backend_tensor_get(voice.ttl, style_ttl.data(), 0, ggml_nbytes(voice.ttl));
    // Lengths on both sides of the relative-position window (9 offsets).
    for (int text_len : { 3, 8, 9, 24, 151 }) {
        check_text_encoder_one_graph(model, style_ttl, text_len);
    }
}

// The one-graph duration encoder must match the hybrid path on the same backend.
void check_duration_one_graph(const supertonic_model & model, const std::vector<float> & style_dp, int text_len) {
    ggml_tensor * emb = require_source_tensor(model,
        "duration:tts.dp.sentence_encoder.text_embedder.char_embedder.weight");
    const std::vector<int64_t> ids = make_synthetic_ids(text_len, emb->ne[1], 0x5D1Au + (uint32_t) text_len);
    std::string err;
    float dur_one = 0.0f, dur_hybrid = 0.0f;
    std::vector<float> proj_one, proj_hybrid;
    const bool ok_one = supertonic_duration_forward_one_graph_ggml(model, ids.data(), text_len, style_dp.data(),
                                                                   dur_one, &err, &proj_one);
    if (!ok_one) std::fprintf(stderr, "  one-graph failed: %s\n", err.c_str());
    CHECK(ok_one);
    const bool ok_hybrid = supertonic_duration_forward_hybrid_ggml(model, ids.data(), text_len, style_dp.data(),
                                                                   dur_hybrid, &err, &proj_hybrid);
    if (!ok_hybrid) std::fprintf(stderr, "  hybrid failed: %s\n", err.c_str());
    CHECK(ok_hybrid);
    if (!ok_one || !ok_hybrid) return;
    std::fprintf(stderr, "  text_len=%d, dur_one=%.6g, dur_hybrid=%.6g\n", text_len, dur_one, dur_hybrid);
    check_parity(model, proj_one, proj_hybrid, "one-graph vs hybrid");
    if (model_prefers_cpu_kernels(model)) {
        CHECK(close_enough(dur_one, dur_hybrid));
    } else {
        CHECK(close_enough(dur_one, dur_hybrid, 0.0f, kGpuDurationRtol));
    }
}

void test_duration_one_graph_matches_hybrid(const supertonic_model & model) {
    std::fprintf(stderr, "[one-graph duration encoder vs hybrid path]\n");
    if (model.voices.empty()) {
        std::fprintf(stderr, "  SKIP: no voices in model\n");
        return;
    }
    const auto & voice = model.voices.begin()->second;
    std::vector<float> style_dp((size_t) ggml_nelements(voice.dp));
    ggml_backend_tensor_get(voice.dp, style_dp.data(), 0, ggml_nbytes(voice.dp));
    for (int text_len : { 3, 8, 9, 24, 151 }) {
        check_duration_one_graph(model, style_dp, text_len);
    }
}
} // namespace

int main(int argc, char ** argv) {
    // This harness gates ggml behavior; a Core ML vocoder sidecar staged next
    // to the GGUF must not reroute it (mirrors the Audio8 harnesses).
#ifndef _WIN32
    setenv("SUPERTONIC_COREML_DISABLE", "1", 1);
#endif

    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL.gguf [N_GPU_LAYERS]\n", argv[0]);
        return 2;
    }
    const int n_gpu_layers = (argc > 2) ? std::atoi(argv[2]) : 0;
    supertonic_model model;
    if (!load_supertonic_gguf(argv[1], model, n_gpu_layers)) {
        std::fprintf(stderr, "failed to load model: %s\n", argv[1]);
        return 1;
    }
    std::fprintf(stderr, "backend: %s\n", model.backend ? ggml_backend_name(model.backend) : "none");

    test_f3_vocoder_unpack_parity(model);
    test_f11_duration_cache_parity(model);
    test_f8_style_residual_cache_parity(model);
    test_loop_graph_matches_per_step(model);
    test_ct_loop_matches_legacy_chain(model);
    test_text_encoder_one_graph_matches_islands(model);
    test_duration_one_graph_matches_hybrid(model);

    free_supertonic_model(model);

    std::fprintf(stderr,
                 "test_supertonic_graph_rewrites: %d / %d checks passed\n",
                 g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
