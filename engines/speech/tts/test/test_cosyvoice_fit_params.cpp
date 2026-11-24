// Fit-projection parity tests (include/tts-cpp/cosyvoice/fit.h): assert that
// the metadata-only memory projection matches what a REAL load and REAL
// dispatches actually allocate, byte for byte where the projection is exact
// by construction.
//
// Modes:
//
//   * No arguments (the always-on CI form): a synthetic CosyVoice3 set is
//     written on the fly -- a tiny LM and a tiny flow (both hyper-parameter-
//     driven from GGUF metadata: cosyvoice3.llm.* / cosyvoice3.flow.*), a
//     tiny voice GGUF, and a FULL-GEOMETRY HiFT (its channel plan is
//     hardcoded in the vocoder builder, so ~70 MB of random weights is the
//     smallest model the real graphs accept).  Gates:
//       1. projected weight bytes == a real allocation of the same tensor
//          set, per GGUF (the real CPU load maps the file in place, so the
//          anchor is a genuine ggml_backend_alloc_ctx_tensors of identical
//          headers);
//       2. projected LM KV bytes == the cache a real decode allocates, and
//          the projected LM arena (max of prefill and deepest step) == the
//          gallocr a real prefill + decode steps settle on;
//       3. projected flow arena (max of front-end and DiT) and HiFT arena
//          (max of f0 / STFT / decode) == the real dispatch allocations of
//          the same graphs;
//       4. fit_params end-to-end: staged peak-phase projection, growth with
//          speech_tokens, the baked-noise length gate, near-INT_MAX
//          rejection, and Error (never Success) on unreadable models.
//
//   * --synthetic-gpu: the same set with n_gpu_layers=99 (exercises the
//     host-resident embedding split and the GPU dispatch; falls back to CPU
//     parity when no validated GPU is present).
//
//   * <llm> <flow> <hift> <voice> [n_gpu_layers]: the same gates on real
//     fixtures.
//
// Exit 0 on success; non-zero with a FAIL line per broken invariant.

#include "tts-cpp/cosyvoice/fit.h"

#include "cosyvoice_fit_internal.h"
#include "test_env_portable.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

int g_failures = 0;

void fail(const std::string & what) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
}

void expect(bool cond, const std::string & what) {
    if (!cond) fail(what);
}

void expect_eq(uint64_t projected, uint64_t real, const std::string & what) {
    if (projected != real) {
        fail(what + ": projected " + std::to_string(projected) +
             " != allocated " + std::to_string(real));
    }
}

// ── Synthetic GGUF writers ──────────────────────────────────────────────────

struct gguf_writer {
    gguf_context * g   = nullptr;
    ggml_context * ctx = nullptr;
    explicit gguf_writer(size_t mem_bytes) {
        g = gguf_init_empty();
        ggml_init_params ip = { mem_bytes, nullptr, /*no_alloc=*/false };
        ctx = ggml_init(ip);
    }
    ~gguf_writer() {
        if (ctx) ggml_free(ctx);
        if (g) gguf_free(g);
    }
    void f32(const std::string & name, std::initializer_list<int64_t> ne) {
        ggml_tensor * t = ggml_new_tensor(ctx, GGML_TYPE_F32, (int) ne.size(),
                                          std::vector<int64_t>(ne).data());
        ggml_set_name(t, name.c_str());
        float * d = (float *) t->data;
        for (int64_t i = 0; i < ggml_nelements(t); ++i) d[i] = 0.01f;
        gguf_add_tensor(g, t);
    }
    void i32(const std::string & name, int64_t n) {
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
        ggml_set_name(t, name.c_str());
        int32_t * d = (int32_t *) t->data;
        for (int64_t i = 0; i < n; ++i) d[i] = (int32_t) (i % 8);
        gguf_add_tensor(g, t);
    }
    void write(const std::string & path) {
        if (!gguf_write_to_file(g, path.c_str(), /*only_meta=*/false)) {
            std::fprintf(stderr, "FATAL: cannot write %s\n", path.c_str());
            std::exit(2);
        }
    }
};

// Tiny LM: hyper-parameters carried in cosyvoice3.llm.* metadata.
constexpr const char * kFixtureDirPrefix = "test-cosyvoice-fit-params-";

fs::path make_fixture_dir() {
    const fs::path dir = fs::temp_directory_path() / (kFixtureDirPrefix + test_process_tag());
    fs::create_directories(dir);
    return dir;
}

std::string write_tiny_llm(const fs::path & dir) {
    const int depth = 2, hidden = 64, n_head = 4, n_kv = 2, head_dim = 16, inter = 128;
    const int VS = 70, VOCAB = 100;
    gguf_writer w(8u << 20);
    gguf_set_val_u32(w.g, "cosyvoice3.llm.depth", depth);
    gguf_set_val_u32(w.g, "cosyvoice3.llm.hidden", hidden);
    gguf_set_val_u32(w.g, "cosyvoice3.llm.n_head", n_head);
    gguf_set_val_u32(w.g, "cosyvoice3.llm.n_kv", n_kv);
    gguf_set_val_u32(w.g, "cosyvoice3.llm.head_dim", head_dim);
    gguf_set_val_u32(w.g, "cosyvoice3.llm.inter", inter);
    w.f32("lm/embed_tokens/weight", { hidden, VOCAB });
    w.f32("lm/speech_embedding/weight", { hidden, VS });
    w.f32("lm/norm/weight", { hidden });
    w.f32("lm/llm_decoder/weight", { hidden, VS });
    for (int i = 0; i < depth; ++i) {
        const std::string p = "lm/blk/" + std::to_string(i) + "/";
        w.f32(p + "in_ln/weight", { hidden });
        w.f32(p + "q_proj/weight", { hidden, n_head * head_dim });
        w.f32(p + "q_proj/bias", { n_head * head_dim });
        w.f32(p + "k_proj/weight", { hidden, n_kv * head_dim });
        w.f32(p + "k_proj/bias", { n_kv * head_dim });
        w.f32(p + "v_proj/weight", { hidden, n_kv * head_dim });
        w.f32(p + "v_proj/bias", { n_kv * head_dim });
        w.f32(p + "o_proj/weight", { n_head * head_dim, hidden });
        w.f32(p + "post_ln/weight", { hidden });
        w.f32(p + "gate/weight", { hidden, inter });
        w.f32(p + "up/weight", { hidden, inter });
        w.f32(p + "down/weight", { inter, hidden });
    }
    const std::string path = (dir / "fit-tiny-cosy-llm.gguf").string();
    w.write(path);
    return path;
}

// Tiny flow: DiT hyper-parameters carried in cosyvoice3.flow.* metadata (the
// front-end's 1024-wide lookahead convolution is hardcoded, so that one stays
// full width).  rand_noise is 200 frames so the length gate is testable.
constexpr int kTinyFlowRNW = 200;
std::string write_tiny_flow(const fs::path & dir) {
    const int dim = 64, depth = 2, groups = 4, conv_k = 7;
    gguf_writer w(24u << 20);
    gguf_set_val_u32(w.g, "cosyvoice3.flow.depth", depth);
    gguf_set_val_u32(w.g, "cosyvoice3.flow.dim", dim);
    gguf_set_val_u32(w.g, "cosyvoice3.flow.heads", 2);
    gguf_set_val_u32(w.g, "cosyvoice3.flow.dim_head", 32);
    gguf_set_val_u32(w.g, "cosyvoice3.flow.conv_k", conv_k);
    gguf_set_val_u32(w.g, "cosyvoice3.flow.conv_groups", groups);
    w.f32("flow/input_embedding/weight", { 80, 32 });
    w.f32("flow/pre_lookahead_layer/conv1/weight", { 4, 80, 1024 });
    w.f32("flow/pre_lookahead_layer/conv1/bias", { 1024 });
    w.f32("flow/pre_lookahead_layer/conv2/weight", { 3, 1024, 80 });
    w.f32("flow/pre_lookahead_layer/conv2/bias", { 80 });
    w.f32("flow/spk_embed_affine_layer/weight", { 16, 80 });
    w.f32("flow/spk_embed_affine_layer/bias", { 80 });
    w.f32("flow/rand_noise", { kTinyFlowRNW, 80 });
    w.f32("flow/time_embed/time_mlp/0/weight", { 256, dim });
    w.f32("flow/time_embed/time_mlp/0/bias", { dim });
    w.f32("flow/time_embed/time_mlp/2/weight", { dim, dim });
    w.f32("flow/time_embed/time_mlp/2/bias", { dim });
    w.f32("flow/input_embed/proj/weight", { 320, dim });
    w.f32("flow/input_embed/proj/bias", { dim });
    for (int l = 1; l <= 2; ++l) {
        const std::string p = "flow/input_embed/conv_pos_embed/conv" + std::to_string(l) + "/0/";
        w.f32(p + "weight", { conv_k, dim / groups, dim });
        w.f32(p + "bias", { dim });
    }
    for (int i = 0; i < depth; ++i) {
        const std::string p = "flow/blk/" + std::to_string(i) + "/";
        w.f32(p + "attn_norm/linear/weight", { dim, 6 * dim });
        w.f32(p + "attn_norm/linear/bias", { 6 * dim });
        for (const char * q : { "to_q", "to_k", "to_v" }) {
            w.f32(p + "attn/" + std::string(q) + "/weight", { dim, dim });
            w.f32(p + "attn/" + std::string(q) + "/bias", { dim });
        }
        w.f32(p + "attn/to_out/0/weight", { dim, dim });
        w.f32(p + "attn/to_out/0/bias", { dim });
        w.f32(p + "ff/ff/0/0/weight", { dim, 2 * dim });
        w.f32(p + "ff/ff/0/0/bias", { 2 * dim });
        w.f32(p + "ff/ff/2/weight", { 2 * dim, dim });
        w.f32(p + "ff/ff/2/bias", { dim });
    }
    w.f32("flow/norm_out/linear/weight", { dim, 2 * dim });
    w.f32("flow/norm_out/linear/bias", { 2 * dim });
    w.f32("flow/proj_out/weight", { dim, 80 });
    w.f32("flow/proj_out/bias", { 80 });
    const std::string path = (dir / "fit-tiny-cosy-flow.gguf").string();
    w.write(path);
    return path;
}

// Full-geometry HiFT: the vocoder's channel plan (512 -> 256/128/64, 18
// STFT channels, 9 harmonics) is hardcoded in the graph builder, so the
// synthetic model carries the real shapes with random weights (~70 MB).
std::string write_hift(const fs::path & dir) {
    gguf_writer w(160u << 20);
    // f0 predictor: shapes are tensor-driven, so these CAN be tiny.
    for (int i = 0; i < 5; ++i) {
        const std::string p = "hift/f0_predictor/condnet/" + std::to_string(i * 2);
        w.f32(p + "/weight", { 3, i == 0 ? 80 : 32, 32 });
        w.f32(p + "/bias", { 32 });
    }
    w.f32("hift/f0_predictor/classifier/weight", { 32, 1 });
    w.f32("hift/f0_predictor/classifier/bias", { 1 });
    // decode: hardcoded channel plan.
    const int ups_ch[3]    = { 256, 128, 64 };
    const int ups_k[3]     = { 16, 11, 7 };
    const int src_k[3]     = { 7, 7, 11 };
    const int src_stride_k[3] = { 15, 3, 1 };
    const int rb_k[3]      = { 3, 7, 11 };
    w.f32("hift/conv_pre/weight", { 5, 80, 512 });
    w.f32("hift/conv_pre/bias", { 512 });
    auto resblock = [&](const std::string & prefix, int C, int k) {
        for (int j = 0; j < 3; ++j) {
            w.f32(prefix + "/activations1/" + std::to_string(j) + "/alpha", { C });
            w.f32(prefix + "/convs1/" + std::to_string(j) + "/weight", { k, C, C });
            w.f32(prefix + "/convs1/" + std::to_string(j) + "/bias", { C });
            w.f32(prefix + "/activations2/" + std::to_string(j) + "/alpha", { C });
            w.f32(prefix + "/convs2/" + std::to_string(j) + "/weight", { k, C, C });
            w.f32(prefix + "/convs2/" + std::to_string(j) + "/bias", { C });
        }
    };
    int C_in = 512;
    for (int i = 0; i < 3; ++i) {
        const int C = ups_ch[i];
        w.f32("hift/ups/" + std::to_string(i) + "/weight", { ups_k[i], C_in, C });
        w.f32("hift/ups/" + std::to_string(i) + "/bias", { C });
        w.f32("hift/source_downs/" + std::to_string(i) + "/weight", { src_stride_k[i], 18, C });
        w.f32("hift/source_downs/" + std::to_string(i) + "/bias", { C });
        resblock("hift/source_resblocks/" + std::to_string(i), C, src_k[i]);
        for (int j = 0; j < 3; ++j) {
            resblock("hift/resblocks/" + std::to_string(i * 3 + j), C, rb_k[j]);
        }
        C_in = C;
    }
    w.f32("hift/conv_post/weight", { 7, 64, 18 });
    w.f32("hift/conv_post/bias", { 18 });
    w.f32("hift/m_source/l_linear/weight", { 9 });
    w.f32("hift/m_source/l_linear/bias", { 1 });
    const std::string path = (dir / "fit-cosy-hift.gguf").string();
    w.write(path);
    return path;
}

constexpr int kVoicePromptStok = 12, kVoicePromptTok = 10, kVoiceMel1 = 6, kVoiceSpk = 16;
std::string write_tiny_voice(const fs::path & dir) {
    gguf_writer w(4u << 20);
    w.i32("voice/prompt_stok", kVoicePromptStok);
    w.i32("voice/prompt_token", kVoicePromptTok);
    w.f32("voice/prompt_feat", { 80, kVoiceMel1 });
    w.f32("voice/embedding", { kVoiceSpk });
    const std::string path = (dir / "fit-tiny-cosy-voice.gguf").string();
    w.write(path);
    return path;
}

// ── Gates ───────────────────────────────────────────────────────────────────

// Weight-sizing parity: the metadata-only size vs a genuine allocation of
// identical tensor headers on the same backend (the real CPU load maps the
// file in place, so the real buffer is null there).
void check_weights_parity(const std::string & path, ggml_backend_t backend,
                          const char * what) {
    cosyvoice_fit_load_measure meas;
    model_ctx mm = cosyvoice_load_gguf_metadata_only(path, backend, meas);
    model_ctx real = cosyvoice_load_gguf(path, backend);
    uint64_t real_dev = 0, real_host = 0;
    if (real.buffer_w) {
        real_dev = ggml_backend_buffer_get_size(real.buffer_w);
    } else {
        size_t n = 8;
        for (ggml_tensor * t = ggml_get_first_tensor(real.ctx_w); t;
             t = ggml_get_next_tensor(real.ctx_w, t)) {
            ++n;
        }
        ggml_init_params wp = { n * ggml_tensor_overhead(), nullptr, /*no_alloc=*/true };
        ggml_context * dup = ggml_init(wp);
        for (ggml_tensor * t = ggml_get_first_tensor(real.ctx_w); t;
             t = ggml_get_next_tensor(real.ctx_w, t)) {
            ggml_set_name(ggml_dup_tensor(dup, t), ggml_get_name(t));
        }
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(dup, real.backend);
        if (!buf) {
            fail(std::string(what) + ": anchor allocation failed");
        } else {
            real_dev = ggml_backend_buffer_get_size(buf);
            ggml_backend_buffer_free(buf);
        }
        ggml_free(dup);
    }
    if (real.buffer_h) real_host = ggml_backend_buffer_get_size(real.buffer_h);
    expect_eq(meas.device_bytes, real_dev, std::string(what) + " device weights parity");
    if (real.buffer_h) {
        expect_eq(meas.host_bytes, real_host, std::string(what) + " host weights parity");
    }
    cosyvoice_free(real);
    cosyvoice_free(mm);
}

void run_parity_gates(const std::string & llm_path, const std::string & flow_path,
                      const std::string & hift_path, ggml_backend_t backend) {
    check_weights_parity(llm_path, backend, "llm");
    check_weights_parity(flow_path, backend, "flow");
    check_weights_parity(hift_path, backend, "hift");

    std::string error;

    // LM: KV + arena vs a real prefill + decode steps at the same shapes.
    {
        cosyvoice_fit_load_measure meas;
        model_ctx mm = cosyvoice_load_gguf_metadata_only(llm_path, backend, meas);
        model_ctx real = cosyvoice_load_gguf(llm_path, backend);
        const qwen_hp hp = cosyvoice_qwen_hp(real);
        const int L0 = 18, n_steps = 3;
        uint64_t kv_meas = 0, kv_real = 0, arena_real = 0;
        cosyvoice_fit_price arena_meas;
        if (!cosyvoice_fit_measure_llm(mm, hp, L0, n_steps, kv_meas, arena_meas, &error)) {
            fail("cosyvoice_fit_measure_llm failed: " + error);
        } else if (!cosyvoice_fit_llm_parity_probe(real, hp, L0, n_steps,
                                                   kv_real, arena_real, &error)) {
            fail("LM parity probe failed: " + error);
        } else {
            expect_eq(kv_meas, kv_real, "LM KV parity");
            if (arena_meas.host_bytes == 0 && arena_real > 0) {
                expect_eq(arena_meas.device_bytes, arena_real, "LM arena parity");
            }
        }
        cosyvoice_free(real);
        cosyvoice_free(mm);
    }

    // Flow: front-end + DiT arenas vs the real dispatch of the same graphs.
    {
        cosyvoice_fit_load_measure meas;
        model_ctx mm = cosyvoice_load_gguf_metadata_only(flow_path, backend, meas);
        model_ctx real = cosyvoice_load_gguf(flow_path, backend);
        const int T_tok = 16, TM = 32, SPK = kVoiceSpk;
        cosyvoice_fit_price arena_meas;
        uint64_t fe_real = 0, dit_real = 0;
        if (!cosyvoice_fit_measure_flow(mm, T_tok, TM, SPK, arena_meas, &error)) {
            fail("cosyvoice_fit_measure_flow failed: " + error);
        } else if (!cosyvoice_fit_flow_parity_probe(real, T_tok, TM, SPK,
                                                    fe_real, dit_real, &error)) {
            fail("flow parity probe failed: " + error);
        } else if (arena_meas.host_bytes == 0 && fe_real > 0 && dit_real > 0) {
            expect_eq(arena_meas.device_bytes, std::max(fe_real, dit_real),
                      "flow arena parity");
        }
        cosyvoice_free(real);
        cosyvoice_free(mm);
    }

    // HiFT: f0 / STFT / decode arenas vs the real dispatch of the same graphs.
    {
        cosyvoice_fit_load_measure meas;
        model_ctx mm = cosyvoice_load_gguf_metadata_only(hift_path, backend, meas);
        model_ctx real = cosyvoice_load_gguf(hift_path, backend);
        const int T_mel = 26;
        cosyvoice_fit_price arena_meas;
        uint64_t f0_real = 0, stft_real = 0, dec_real = 0;
        if (!cosyvoice_fit_measure_hift(mm, T_mel, arena_meas, &error)) {
            fail("cosyvoice_fit_measure_hift failed: " + error);
        } else if (!cosyvoice_fit_hift_parity_probe(real, T_mel, f0_real, stft_real,
                                                    dec_real, &error)) {
            fail("hift parity probe failed: " + error);
        } else if (arena_meas.host_bytes == 0 && f0_real > 0 && stft_real > 0 &&
                   dec_real > 0) {
            expect_eq(arena_meas.device_bytes,
                      std::max(f0_real, std::max(stft_real, dec_real)),
                      "hift arena parity");
        }
        cosyvoice_free(real);
        cosyvoice_free(mm);
    }
}

void run_fit_gates(const std::string & llm_path, const std::string & flow_path,
                   const std::string & hift_path, const std::string & voice_path,
                   int n_gpu_layers, bool tiny_rand_noise) {
    tts_cpp::cosyvoice::FitOptions fopts;
    fopts.llm_gguf_path   = llm_path;
    fopts.flow_gguf_path  = flow_path;
    fopts.hift_gguf_path  = hift_path;
    fopts.voice_gguf_path = voice_path;
    fopts.n_gpu_layers    = n_gpu_layers;
    fopts.text_tokens     = 4;
    fopts.speech_tokens   = 40;

    const tts_cpp::FitResult fit = tts_cpp::cosyvoice::fit_params(fopts);
    expect(fit.status != tts_cpp::FitStatus::Error,
           "fit_params returned Error (" + fit.reason + ") for readable models");
    expect(fit.device.weights_bytes > 0, "projected weights_bytes == 0");
    expect(fit.device.total_bytes > fit.device.weights_bytes,
           "projected total does not exceed the peak phase's weights");
    expect(fit.host_bytes > 0, "projected host_bytes == 0");
    expect(!fit.report.empty(), "empty report");
    if (g_failures == 0) std::printf("%s", fit.report.c_str());

    // A longer utterance grows the flow/hift phases.
    {
        tts_cpp::cosyvoice::FitOptions longer = fopts;
        longer.speech_tokens = 80;
        const tts_cpp::FitResult big = tts_cpp::cosyvoice::fit_params(longer);
        expect(big.status != tts_cpp::FitStatus::Error,
               "longer utterance was Error (" + big.reason + ")");
        const uint64_t sum_a = fit.device.total_bytes + fit.host_bytes;
        const uint64_t sum_b = big.device.total_bytes + big.host_bytes;
        expect(sum_b > sum_a, "projection did not grow with speech_tokens");
    }

    // The baked-noise gate: a default (cap-length) generation exceeds the
    // tiny fixture's 200 rand_noise frames and must refuse, exactly as
    // cosyvoice_flow_run would.
    if (tiny_rand_noise) {
        tts_cpp::cosyvoice::FitOptions capped = fopts;
        capped.speech_tokens = 0;  // the 50*(text_tokens+1) cap
        const tts_cpp::FitResult fr = tts_cpp::cosyvoice::fit_params(capped);
        expect(fr.status == tts_cpp::FitStatus::Error &&
                   fr.reason == "workload-too-large",
               "over-rand_noise workload was not workload-too-large (" + fr.reason + ")");
    }

    // Near-INT_MAX workloads are rejected strictly, never priced through
    // wrapped-int graph shapes.
    {
        tts_cpp::cosyvoice::FitOptions huge = fopts;
        huge.text_tokens = std::numeric_limits<int>::max();
        const tts_cpp::FitResult fr = tts_cpp::cosyvoice::fit_params(huge);
        expect(fr.status == tts_cpp::FitStatus::Error &&
                   fr.reason == "workload-too-large",
               "near-INT_MAX text_tokens was not workload-too-large");
    }

    {
        tts_cpp::cosyvoice::FitOptions missing = fopts;
        missing.backend = "no-such-device";
        const tts_cpp::FitResult fr = tts_cpp::cosyvoice::fit_params(missing);
        expect(fr.status == tts_cpp::FitStatus::Error && fr.reason == "no-backend-device",
               "unavailable explicit backend reason was '" + fr.reason + "'");
    }
    {
        tts_cpp::cosyvoice::FitOptions cpu = fopts;
        cpu.backend = "cpu";
        const tts_cpp::FitResult fr = tts_cpp::cosyvoice::fit_params(cpu);
        expect(fr.status != tts_cpp::FitStatus::Error && fr.device_is_cpu,
               "explicit cpu backend did not project the CPU (" + fr.reason + ")");
    }

    // Errors surface as Error, never Success.
    {
        tts_cpp::cosyvoice::FitOptions bad = fopts;
        bad.flow_gguf_path = flow_path + ".does-not-exist";
        const tts_cpp::FitResult fr = tts_cpp::cosyvoice::fit_params(bad);
        expect(fr.status == tts_cpp::FitStatus::Error, "missing flow was not Error");
        expect(fr.reason == "model-unreadable",
               "missing flow reason was '" + fr.reason + "'");
    }
}

void run_all(const std::string & llm, const std::string & flow, const std::string & hift,
             const std::string & voice, int n_gpu_layers, bool tiny_rand_noise) {
    namespace det = ::tts_cpp::detail;
    ggml_backend_t backend = det::init_gpu_backend(
        n_gpu_layers, /*verbose=*/false, "cosyvoice", /*vulkan_device=*/0,
        /*allow_arm_mali=*/false, nullptr, cosyvoice_gpu_requirement());
    if (!backend) backend = det::init_cpu_backend();
    if (!backend) {
        fail("no backend");
        return;
    }
    run_parity_gates(llm, flow, hift, backend);
    run_fit_gates(llm, flow, hift, voice, n_gpu_layers, tiny_rand_noise);
    ggml_backend_free(backend);
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc >= 5) {
        const int n_gpu_layers = argc > 5 ? std::atoi(argv[5]) : 0;
        run_all(argv[1], argv[2], argv[3], argv[4], n_gpu_layers,
                /*tiny_rand_noise=*/false);
    } else {
        const int n_gpu_layers =
            (argc >= 2 && std::string(argv[1]) == "--synthetic-gpu") ? 99 : 0;
        const fs::path dir = make_fixture_dir();
        const std::string llm   = write_tiny_llm(dir);
        const std::string flow  = write_tiny_flow(dir);
        const std::string hift  = write_hift(dir);
        const std::string voice = write_tiny_voice(dir);
        run_all(llm, flow, hift, voice, n_gpu_layers, /*tiny_rand_noise=*/true);
        fs::remove_all(dir);
    }
    if (g_failures == 0) {
        std::printf("test-cosyvoice-fit-params: all checks passed\n");
    }
    return g_failures;
}
