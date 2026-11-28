// Encoder backend parity: locate the first encoder stage/layer where the
// Hexagon (HTP0) backend diverges from the CPU reference.
//
// Backend parity diagnostic. Per-op test-backend-ops passes 152/152 on HTP0 but
// end-to-end Parakeet inference produces garbage. This harness loads the same
// GGUF twice (backend A = "cpu" by default, backend B = "hexagon" by default),
// runs the same mel through run_encoder() on both, and prints per-stage
// cosine similarity, max_abs difference, and relative L2 for every captured
// tensor. It also sweeps max_layers to pinpoint the first Conformer block
// that diverges.
// For Sortformer, also compares head probabilities on identical CPU encoder
// output. PARAKEET_TRACE_SORTFORMER_HEAD=1 prints per-node head differences;
// observation disables fusion, so use the default run for acceptance.
//
// Usage:
//   test-encoder-backend-parity --model <gguf> --wav <wav>
//                               [--backend-a cpu] [--backend-b hexagon]
//                               [--min-cos 0.99] [--layer-sweep]
//
// Exit 0 on success (all stages meet --min-cos), non-zero otherwise. The
// per-stage table is always printed so a failure is actionable.

#include "parakeet_ctc.h"
#include "parakeet_sortformer.h"
#include "mel_preprocess.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model <gguf> --wav <wav>\n"
        "           [--backend-a <name>] [--backend-b <name>]\n"
        "           [--min-cos <float>] [--layer-sweep]\n"
        "\n"
        "Loads <gguf> under two backends and compares per-stage encoder outputs\n"
        "for the same mel input. Defaults: backend-a=cpu, backend-b=hexagon,\n"
        "min-cos=0.99. With --layer-sweep, additionally sweeps max_layers over\n"
        "{0, 1, 2, 4, 8, and every 4th layer} to identify the first diverging\n"
        "Conformer block. Sortformer also checks head probability parity\n"
        "(max absolute error <= 0.05). Set PARAKEET_TRACE_SORTFORMER_HEAD=1\n"
        "for a per-node head diagnostic with fusion disabled by observation.\n",
        argv0);
}

struct Metrics {
    double cosine  = 0.0;
    double max_abs = 0.0;
    double rel_l2  = 0.0;
    bool   nonfinite = false;
};

Metrics compute_metrics(const std::vector<float> & a, const std::vector<float> & b) {
    Metrics m;
    const size_t n = std::min(a.size(), b.size());
    double dot = 0.0, na = 0.0, nb = 0.0, diff_sq = 0.0, ref_sq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double av = a[i];
        const double bv = b[i];
        if (!std::isfinite(av) || !std::isfinite(bv)) {
            m.nonfinite = true;
        }
        const double d = av - bv;
        if (std::fabs(d) > m.max_abs) m.max_abs = std::fabs(d);
        dot     += av * bv;
        na      += av * av;
        nb      += bv * bv;
        diff_sq += d * d;
        ref_sq  += av * av;
    }
    const double denom = std::sqrt(na) * std::sqrt(nb);
    m.cosine = denom > 0.0 ? dot / denom : (na == 0.0 && nb == 0.0 ? 1.0 : 0.0);
    m.rel_l2 = ref_sq > 0.0 ? std::sqrt(diff_sq / ref_sq) : std::sqrt(diff_sq);
    return m;
}

// Reconcile a `block_0_attn_qkv` capture pair when the two backends took
// different loader paths. Some backends load a pre-stacked
// `encoder.blk.*.attn.qkv.weight` (single Q8_0 mat-mul, capture is the full
// 3*d_model x T post-projection tensor) while others load split
// attn_q / attn_k / attn_v weights (capture is the Q projection alone,
// d_model x T). Compared naively the shared prefix of the two vectors mixes
// Q with K / V per frame on the stacked side, and cosine collapses to ~0.04
// despite the underlying Q values matching bit-for-bit.
//
// The stacked layout is [Q_dmodel | K_dmodel | V_dmodel] per T frame in ggml
// row-major order (Q at offset 0 within each 3*d_model stride, matching the
// view offsets used to build q / k / v in rel_pos_mha_graph). Extract the Q
// slice from the stacked side into a fresh buffer so the caller can compare
// like-for-like. Same-size captures are returned untouched; unrelated size
// mismatches (not a 3:1 ratio) are also left alone.
void align_qkv_captures(std::vector<float> & a, std::vector<float> & b, int d_model) {
    if (a.size() == b.size()) return;
    const bool a_stacked = (a.size() == 3 * b.size());
    const bool b_stacked = (b.size() == 3 * a.size());
    if (!a_stacked && !b_stacked) return;
    if (d_model <= 0) return;
    std::vector<float>       & wide   = a_stacked ? a : b;
    const std::vector<float> & narrow = a_stacked ? b : a;
    const size_t n_frames = narrow.size() / (size_t) d_model;
    std::vector<float> q_slice;
    q_slice.reserve(narrow.size());
    for (size_t f = 0; f < n_frames; ++f) {
        const size_t base = f * 3 * (size_t) d_model;
        for (int i = 0; i < d_model; ++i) {
            q_slice.push_back(wide[base + i]);
        }
    }
    wide.swap(q_slice);
}

struct StageRef {
    const char * name;
    const std::vector<float> parakeet::EncoderOutputs::* ptr;
};

// Optional per-node head trace. Requesting observations prevents fusion, so
// normal head parity below also runs without a callback by default.
struct HeadTrace {
    std::map<std::string, std::vector<float>> reference;
    bool compare = false;
    size_t index = 0;
};

bool trace_head(ggml_tensor * t, bool ask, void * user_data) {
    if (ask) return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t);
    auto & trace = *static_cast<HeadTrace *>(user_data);
    std::vector<float> values(ggml_nelements(t));
    ggml_backend_tensor_get(t, values.data(), 0, values.size() * sizeof(float));
    const std::string key = std::to_string(trace.index++) + ":" + ggml_op_name(t->op) + ":" + t->name;
    if (!trace.compare) {
        trace.reference[key] = std::move(values);
    } else {
        const auto found = trace.reference.find(key);
        if (found != trace.reference.end() && found->second.size() == values.size()) {
            const Metrics m = compute_metrics(found->second, values);
            std::fprintf(stderr, "[head-trace] %-32s cos=%.6f max_abs=%.3e rel_l2=%.3e%s\n",
                         key.c_str(), m.cosine, m.max_abs, m.rel_l2, m.nonfinite ? " NONFINITE" : "");
        }
    }
    return true;
}

const StageRef kStages[] = {
    {"subsampling_out",      &parakeet::EncoderOutputs::subsampling_out},
    {"block_0_post_ff1",     &parakeet::EncoderOutputs::block_0_post_ff1},
    {"block_0_attn_qkv",     &parakeet::EncoderOutputs::block_0_attn_qkv},
    {"block_0_attn_k_raw",   &parakeet::EncoderOutputs::block_0_attn_k_raw},
    {"block_0_attn_v_raw",   &parakeet::EncoderOutputs::block_0_attn_v_raw},
    {"block_0_attn_q_perm",  &parakeet::EncoderOutputs::block_0_attn_q_perm},
    {"block_0_attn_k_perm",  &parakeet::EncoderOutputs::block_0_attn_k_perm},
    {"block_0_attn_q_u",     &parakeet::EncoderOutputs::block_0_attn_q_u},
    {"block_0_attn_ac",      &parakeet::EncoderOutputs::block_0_attn_ac},
    {"block_0_attn_bd",      &parakeet::EncoderOutputs::block_0_attn_bd},
    {"block_0_attn_scores",  &parakeet::EncoderOutputs::block_0_attn_scores},
    {"block_0_attn_softmax", &parakeet::EncoderOutputs::block_0_attn_softmax},
    {"block_0_attn_out",     &parakeet::EncoderOutputs::block_0_attn_out},
    {"block_0_post_attn",    &parakeet::EncoderOutputs::block_0_post_attn},
    {"block_0_post_conv",    &parakeet::EncoderOutputs::block_0_post_conv},
    {"block_0_post_ff2",     &parakeet::EncoderOutputs::block_0_post_ff2},
    {"block_0_out",          &parakeet::EncoderOutputs::block_0_out},
    {"block_last_out",       &parakeet::EncoderOutputs::block_last_out},
    {"encoder_out",          &parakeet::EncoderOutputs::encoder_out},
};

int load_model(const std::string & gguf, const std::string & backend,
               parakeet::ParakeetCtcModel & out) {
    // Explicit-backend overload: pass n_gpu_layers=999 so backends that pay
    // attention to the flag (opencl / hexagon) actually get the encoder.
    int rc = parakeet::load_from_gguf(gguf, out,
                                      /*n_threads=*/0,
                                      /*n_gpu_layers=*/999,
                                      /*verbose=*/false,
                                      backend);
    return rc;
}

struct Options {
    std::string model_path;
    std::string wav_path;
    std::string backend_a = "cpu";
    std::string backend_b = "hexagon";
    double      min_cos = 0.99;
    bool        layer_sweep = false;
};

// Parse the CLI. Returns 0 on success, 2 on unknown/missing args, and sets
// help_requested = true when --help was passed so main can exit 0.
int parse_args(int argc, char ** argv, Options & opts, bool & help_requested) {
    help_requested = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if      (a == "--model"     && i + 1 < argc) opts.model_path = argv[++i];
        else if (a == "--wav"       && i + 1 < argc) opts.wav_path   = argv[++i];
        else if (a == "--backend-a" && i + 1 < argc) opts.backend_a  = argv[++i];
        else if (a == "--backend-b" && i + 1 < argc) opts.backend_b  = argv[++i];
        else if (a == "--min-cos"   && i + 1 < argc) opts.min_cos    = std::atof(argv[++i]);
        else if (a == "--layer-sweep")               opts.layer_sweep = true;
        else if (a == "-h" || a == "--help")         { help_requested = true; return 0; }
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }
    if (opts.model_path.empty() || opts.wav_path.empty()) return 2;
    return 0;
}

// Load the same GGUF on two backends. Returns 0 on success, 3 on either load
// failure. Emits the active backend name for each so cache misses / auto
// resolution are visible.
int load_both_models(const std::string & model_path,
                     const std::string & backend_a, const std::string & backend_b,
                     parakeet::ParakeetCtcModel & model_a,
                     parakeet::ParakeetCtcModel & model_b) {
    std::fprintf(stderr, "[backend-parity] loading %s on backend A=%s\n",
                 model_path.c_str(), backend_a.c_str());
    if (int rc = load_model(model_path, backend_a, model_a); rc != 0) {
        std::fprintf(stderr, "[backend-parity] load(A=%s) rc=%d\n", backend_a.c_str(), rc);
        return 3;
    }
    std::fprintf(stderr, "[backend-parity]   active backend A: %s\n",
                 parakeet::model_active_backend_name(model_a).c_str());

    std::fprintf(stderr, "[backend-parity] loading %s on backend B=%s\n",
                 model_path.c_str(), backend_b.c_str());
    if (int rc = load_model(model_path, backend_b, model_b); rc != 0) {
        std::fprintf(stderr, "[backend-parity] load(B=%s) rc=%d\n", backend_b.c_str(), rc);
        return 3;
    }
    std::fprintf(stderr, "[backend-parity]   active backend B: %s\n",
                 parakeet::model_active_backend_name(model_b).c_str());
    return 0;
}

// Decode the wav and compute the log-mel spectrogram on host so both backends
// receive an identical float32 buffer. Returns 0 on success, 4 on IO / config
// mismatch.
int compute_mel_from_wav(const std::string & wav_path,
                         const parakeet::ParakeetCtcModel & model_a,
                         std::vector<float> & mel, int & n_mel_frames) {
    std::vector<float> samples;
    int sr = 0;
    if (int rc = parakeet::load_wav_mono_f32(wav_path, samples, sr); rc != 0) {
        std::fprintf(stderr, "[backend-parity] load_wav rc=%d\n", rc);
        return 4;
    }
    if (sr != model_a.mel_cfg.sample_rate) {
        std::fprintf(stderr, "[backend-parity] FAIL: wav sr %d != model sr %d\n",
                     sr, model_a.mel_cfg.sample_rate);
        return 4;
    }
    if (int rc = parakeet::compute_log_mel(samples.data(), (int) samples.size(),
                                           model_a.mel_cfg, mel, n_mel_frames); rc != 0) {
        std::fprintf(stderr, "[backend-parity] compute_log_mel rc=%d\n", rc);
        return 4;
    }
    std::fprintf(stderr, "[backend-parity] wav=%zu samples @ %d Hz, mel=%d frames x %d mels\n",
                 samples.size(), sr, n_mel_frames, model_a.mel_cfg.n_mels);
    return 0;
}

// Run the full encoder on both backends with capture_intermediates so per-
// stage comparison is possible. Returns 0 on success, 5 on either run failure,
// 6 when the two runs disagree on output shape metadata.
int run_both_encoders(parakeet::ParakeetCtcModel & model_a,
                      parakeet::ParakeetCtcModel & model_b,
                      const float * mel_data, int n_mel_frames,
                      parakeet::EncoderOutputs & out_a,
                      parakeet::EncoderOutputs & out_b) {
    if (int rc = run_encoder(model_a, mel_data, n_mel_frames, model_a.mel_cfg.n_mels,
                             out_a, parakeet::EncoderRoute::ggml(), /*max_layers=*/-1, /*capture_intermediates=*/true); rc != 0) {
        std::fprintf(stderr, "[backend-parity] run_encoder(A) rc=%d\n", rc);
        return 5;
    }
    if (int rc = run_encoder(model_b, mel_data, n_mel_frames, model_b.mel_cfg.n_mels,
                             out_b, parakeet::EncoderRoute::ggml(), /*max_layers=*/-1, /*capture_intermediates=*/true); rc != 0) {
        std::fprintf(stderr, "[backend-parity] run_encoder(B) rc=%d\n", rc);
        return 5;
    }
    if (out_a.n_enc_frames != out_b.n_enc_frames || out_a.d_model != out_b.d_model) {
        std::fprintf(stderr,
            "[backend-parity] FAIL: shape metadata drift  n_enc=%d/%d  d_model=%d/%d\n",
            out_a.n_enc_frames, out_b.n_enc_frames, out_a.d_model, out_b.d_model);
        return 6;
    }
    return 0;
}

// Walk kStages and score each intermediate capture. Returns the number of
// stages that fell below min_cos (or size-mismatched after QKV normalization,
// or produced non-finite values).
int compare_stages(const parakeet::EncoderOutputs & out_a,
                   const parakeet::EncoderOutputs & out_b,
                   int d_model, double min_cos,
                   const std::string & backend_a, const std::string & backend_b) {
    std::fprintf(stderr,
        "\n[backend-parity] per-stage comparison  A=%s  B=%s\n"
        "  %-20s %10s %14s %14s %14s\n",
        backend_a.c_str(), backend_b.c_str(),
        "stage", "floats", "cosine", "max_abs", "rel_l2");

    bool first_fail_reported = false;
    int failures = 0;
    for (const auto & stage : kStages) {
        auto va = out_a.*(stage.ptr);
        auto vb = out_b.*(stage.ptr);
        if (va.empty() || vb.empty()) {
            std::fprintf(stderr, "  %-20s %10s %14s %14s %14s\n",
                         stage.name, "(empty)", "-", "-", "-");
            continue;
        }
        // The `block_0_attn_qkv` capture is (n_embd, T) on backends that
        // took the split-Q/K/V loader path and (3*n_embd, T) on backends
        // that took the pre-stacked qkv path. Extract the Q slice from the
        // stacked side so we compare like-for-like. No-op when sizes match.
        const bool is_qkv_stage = std::strcmp(stage.name, "block_0_attn_qkv") == 0;
        const bool sized_mismatch_pre = va.size() != vb.size();
        if (is_qkv_stage && sized_mismatch_pre) {
            align_qkv_captures(va, vb, d_model);
        }
        // Any residual size mismatch after the permitted QKV normalization
        // means one backend produced a shorter capture (e.g. dropped a
        // tail). compute_metrics only walks min(a, b), so a matching
        // prefix would otherwise print a spurious cosine == 1 for
        // unequal captures. Fail the stage before scoring cosine.
        const bool size_mismatch = va.size() != vb.size();
        Metrics m = compute_metrics(va, vb);
        const bool ok = !size_mismatch && !m.nonfinite && m.cosine >= min_cos;
        std::fprintf(stderr, "  %-20s %10zu %14.6f %14.3e %14.3e  %s%s%s%s\n",
                     stage.name, va.size(), m.cosine, m.max_abs, m.rel_l2,
                     ok ? "OK" : "FAIL",
                     size_mismatch ? " (size mismatch)" : "",
                     m.nonfinite ? " (non-finite!)" : "",
                     (is_qkv_stage && sized_mismatch_pre && !size_mismatch) ? " (Q slice)" : "");
        if (!ok) {
            failures++;
            if (!first_fail_reported) {
                if (size_mismatch) {
                    std::fprintf(stderr,
                        "\n  >>> first divergence at stage `%s` (size %zu vs %zu) <<<\n\n",
                        stage.name, va.size(), vb.size());
                } else {
                    std::fprintf(stderr,
                        "\n  >>> first divergence at stage `%s` (cos=%.6f < %.6f) <<<\n\n",
                        stage.name, m.cosine, min_cos);
                }
                first_fail_reported = true;
            }
        }
    }
    return failures;
}

// Compare Sortformer speaker probabilities driven from the SAME encoder
// output on both models, so a head-only regression can be pinpointed. Returns
// 0 when the head matched (or was not applicable), 1 when the head diverged,
// and 7 on infrastructure failure (a diarization call failed, or the two
// heads produced different-sized outputs).
int compare_sortformer_head(parakeet::ParakeetCtcModel & model_a,
                            parakeet::ParakeetCtcModel & model_b,
                            const parakeet::EncoderOutputs & out_a,
                            double min_cos) {
    using namespace parakeet;
    if (model_a.model_type != ParakeetModelType::SORTFORMER) return 0;

    // Give both heads identical encoder output to isolate head errors.
    const char * trace_env = std::getenv("PARAKEET_TRACE_SORTFORMER_HEAD");
    const bool trace_enabled = trace_env && std::strcmp(trace_env, "1") == 0;
    HeadTrace trace;
    if (trace_enabled) ggml_backend_sched_set_eval_callback(model_sched(model_a), trace_head, &trace);
    SortformerDiarizationResult head_a, head_b;
    int rc = sortformer_diarize_ggml(model_a, out_a.encoder_out.data(), out_a.n_enc_frames,
                                    out_a.d_model, {}, head_a);
    ggml_backend_sched_set_eval_callback(model_sched(model_a), nullptr, nullptr);
    if (rc != 0) return 7;
    trace.compare = true;
    trace.index = 0;
    if (trace_enabled) ggml_backend_sched_set_eval_callback(model_sched(model_b), trace_head, &trace);
    rc = sortformer_diarize_ggml(model_b, out_a.encoder_out.data(), out_a.n_enc_frames,
                                out_a.d_model, {}, head_b);
    ggml_backend_sched_set_eval_callback(model_sched(model_b), nullptr, nullptr);
    if (rc != 0 || head_a.speaker_probs.size() != head_b.speaker_probs.size()) return 7;
    const Metrics m = compute_metrics(head_a.speaker_probs, head_b.speaker_probs);
    const bool ok = !m.nonfinite && m.cosine >= min_cos && m.max_abs <= 0.05;
    std::fprintf(stderr, "[head-parity] cos=%.6f max_abs=%.3e rel_l2=%.3e %s\n",
                 m.cosine, m.max_abs, m.rel_l2, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

// Re-run the encoder at growing depths (1, 2, 4, 8, 12, 16, ...) to localize
// which block first diverges. This is diagnostic; it does not contribute to
// the failure count.
void run_layer_sweep(parakeet::ParakeetCtcModel & model_a,
                     parakeet::ParakeetCtcModel & model_b,
                     const float * mel_data, int n_mel_frames,
                     double min_cos) {
    const int n_layers = model_a.encoder_cfg.n_layers;
    std::vector<int> plan;
    for (int L = 0; L <= n_layers; L = (L == 0 ? 1 : (L < 8 ? L * 2 : L + 4))) {
        plan.push_back(std::min(L, n_layers));
        if (L >= n_layers) break;
    }
    // Ensure the final layer is compared.
    if (plan.empty() || plan.back() != n_layers) plan.push_back(n_layers);

    std::fprintf(stderr,
        "\n[backend-parity] layer sweep  (encoder has %d Conformer blocks)\n"
        "  %-12s %10s %14s %14s %14s\n",
        n_layers, "max_layers", "floats", "cosine", "max_abs", "rel_l2");

    for (int L : plan) {
        parakeet::EncoderOutputs pa, pb;
        if (run_encoder(model_a, mel_data, n_mel_frames, model_a.mel_cfg.n_mels,
                        pa, parakeet::EncoderRoute::ggml(), L, /*capture_intermediates=*/true) != 0) continue;
        if (run_encoder(model_b, mel_data, n_mel_frames, model_b.mel_cfg.n_mels,
                        pb, parakeet::EncoderRoute::ggml(), L, /*capture_intermediates=*/true) != 0) continue;
        if (pa.encoder_out.empty() || pb.encoder_out.empty()) continue;
        const bool size_mismatch = pa.encoder_out.size() != pb.encoder_out.size();
        Metrics m = compute_metrics(pa.encoder_out, pb.encoder_out);
        const bool ok = !size_mismatch && m.cosine >= min_cos;
        std::fprintf(stderr, "  %-12d %10zu %14.6f %14.3e %14.3e  %s%s\n",
                     L, pa.encoder_out.size(), m.cosine, m.max_abs, m.rel_l2,
                     ok ? "OK" : "FAIL",
                     size_mismatch ? " (size mismatch)" : "");
    }
}

}  // namespace

int main(int argc, char ** argv) {
    Options opts;
    bool help_requested = false;
    if (int rc = parse_args(argc, argv, opts, help_requested); rc != 0) {
        usage(argv[0]);
        return rc;
    }
    if (help_requested) { usage(argv[0]); return 0; }

    using namespace parakeet;
    ParakeetCtcModel model_a, model_b;
    if (int rc = load_both_models(opts.model_path, opts.backend_a, opts.backend_b,
                                  model_a, model_b); rc != 0) return rc;

    std::vector<float> mel;
    int n_mel_frames = 0;
    if (int rc = compute_mel_from_wav(opts.wav_path, model_a, mel, n_mel_frames); rc != 0) return rc;

    EncoderOutputs out_a, out_b;
    if (int rc = run_both_encoders(model_a, model_b, mel.data(), n_mel_frames,
                                   out_a, out_b); rc != 0) return rc;

    int failures = compare_stages(out_a, out_b, model_a.encoder_cfg.d_model, opts.min_cos,
                                  opts.backend_a, opts.backend_b);

    const int head_rc = compare_sortformer_head(model_a, model_b, out_a, opts.min_cos);
    if (head_rc == 7) return 7;
    if (head_rc != 0) ++failures;

    if (opts.layer_sweep) {
        run_layer_sweep(model_a, model_b, mel.data(), n_mel_frames, opts.min_cos);
    }

    if (failures > 0) {
        std::fprintf(stderr,
            "\n[backend-parity] FAIL: %d stage(s) below cosine %.4f\n", failures, opts.min_cos);
        return 1;
    }
    std::fprintf(stderr, "\n[backend-parity] PASS: all stages >= cosine %.4f\n", opts.min_cos);
    return 0;
}
