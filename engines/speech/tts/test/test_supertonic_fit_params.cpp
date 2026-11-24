// Fit-projection parity tests (include/tts-cpp/supertonic/fit.h): assert that
// the metadata-only memory projection matches what a REAL load and REAL
// reservations actually allocate, byte for byte where the projection is exact
// by construction.
//
// Usage: test-supertonic-fit-params [<supertonic.gguf> [n_gpu_layers=99]]
//
// Two modes:
//
//   * No arguments (the always-on CI form): a metadata-only synthetic
//     supertonic GGUF is synthesized on the fly -- the exact load-time tensor
//     contract (unicode indexer, voices, the full vocoder bind roster with
//     real-width final_norm, RoPE theta, tensor/source name maps) at tiny
//     dims everywhere the loader does not pin a width.  The stage graph
//     builders hardcode the real channel plans, so stage-arena parity still
//     needs a fixture; this mode pins weight-sizing parity (buffer_w +
//     the GPU pre-transpose extra buffer, byte for byte vs a real load) and
//     the refusal paths (CPU pointwise-path refusal, invalid arguments,
//     unreadable model, oversized workload, and Error -- never a guessed
//     FITS -- when a stage measure cannot run).
//
//   * With arguments <supertonic.gguf> [n_gpu_layers]: the full gates on a
//     real fixture.
//
// Gates (all against the real GGUF):
//   1. projected weight bytes (buffer_w + buffer_w_extra) == the buffers a
//      real load allocates, byte for byte -- including the per-tensor
//      storage-type decisions and the GPU pre-transpose roster;
//   2. every stage measure (text encoder cache set / duration / vector
//      estimator loop graph / vocoder) == the real reservation of the same
//      graphs (the parity probes), byte for byte on the direct path;
//   3. fit_params end-to-end: quadratic growth with text_tokens (the relpos
//      masks), growth with audio_seconds, the CPU pointwise-path refusal
//      (compute-path-not-supported -- never a guessed FITS), near-INT_MAX
//      rejection, and Error (never Success) on unreadable models.
//
// The vector-estimator gates require the fused graph path: a GPU or a CPU
// build without Accelerate/CBLAS pointwise kernels. The CPU pointwise
// multi-cache path is refused by design, so those builds check the refusal.
//
// Exit 0 on success; non-zero with a FAIL line per broken invariant.

#include "tts-cpp/supertonic/fit.h"

#include "supertonic_internal.h"

#include "ggml-backend.h"
#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <string>
#include <vector>

using namespace tts_cpp::supertonic::detail;

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

// ── Metadata-only synthetic supertonic GGUF ─────────────────────────────────
// The exact load-time contract of load_supertonic_gguf (hparams keys, the
// unicode indexer, one voice, the full bind_vocoder_weights roster, RoPE
// theta, the tensor_names/source_names map), at tiny dims everywhere the
// loader does not pin a width.  The two pinned widths are honored: the
// vocoder final_norm.* quartet must be exactly 512 (the F2 BN pre-bake
// checks), and the four F6 t_proj sources are [512, 64] so the GPU
// pre-transpose roster (and with it buffer_w_extra parity) is exercised.
// Stage graph tensors (text encoder / duration / vector estimator bodies)
// are intentionally absent -- their channel plans are hardcoded to the real
// architecture, and this mode asserts that fit_params answers Error (never a
// guessed FITS) when a stage measure cannot run.

struct synthetic_writer {
    gguf_context * g = nullptr;
    ggml_context * ctx = nullptr;
    std::vector<std::string> tensor_names;
    std::vector<std::string> source_names;

    void add_f32(const std::string & name, std::initializer_list<int64_t> ne,
                 const std::string & source = {}) {
        ggml_tensor * t = ggml_new_tensor(ctx, GGML_TYPE_F32, (int) ne.size(),
                                          std::vector<int64_t>(ne).data());
        ggml_set_name(t, name.c_str());
        float * d = (float *) t->data;
        for (int64_t i = 0; i < ggml_nelements(t); ++i) d[i] = 0.5f;
        gguf_add_tensor(g, t);
        if (!source.empty()) {
            tensor_names.push_back(name);
            source_names.push_back(source);
        }
    }
};

std::string write_tiny_supertonic_gguf() {
    namespace fs = std::filesystem;
    const std::string path =
        (fs::temp_directory_path() / "test-supertonic-fit-tiny.gguf").string();
    synthetic_writer w;
    w.g = gguf_init_empty();
    gguf_set_val_str(w.g, "supertonic.arch", "supertonic");
    gguf_set_val_u32(w.g, "supertonic.sample_rate", 44100);
    gguf_set_val_u32(w.g, "supertonic.base_chunk_size", 128);
    gguf_set_val_u32(w.g, "supertonic.ttl_chunk_compress_factor", 6);
    gguf_set_val_u32(w.g, "supertonic.latent_dim", 24);
    gguf_set_val_u32(w.g, "supertonic.latent_channels", 24);
    gguf_set_val_u32(w.g, "supertonic.default_steps", 5);
    gguf_set_val_f32(w.g, "supertonic.default_speed", 1.05f);
    {
        const char * voices[] = { "F1" };
        gguf_set_arr_str(w.g, "supertonic.voice_names", voices, 1);
        const char * langs[] = { "en" };
        gguf_set_arr_str(w.g, "supertonic.languages", langs, 1);
    }

    ggml_init_params ip = { 8u * 1024 * 1024, nullptr, /*no_alloc=*/false };
    w.ctx = ggml_init(ip);

    // Tensors the loader requires by GGUF name (no source map entry).
    {
        ggml_tensor * t = ggml_new_tensor_1d(w.ctx, GGML_TYPE_I32, 16);
        ggml_set_name(t, "supertonic/unicode_indexer");
        int32_t * d = (int32_t *) t->data;
        for (int i = 0; i < 16; ++i) d[i] = i;
        gguf_add_tensor(w.g, t);
    }
    w.add_f32("supertonic/voices/F1/ttl", { 256, 50 });
    w.add_f32("supertonic/voices/F1/dp",  { 192 });

    // bind_vocoder_weights roster (tiny except the two pinned widths).
    int n = 0;
    auto voc = [&](const std::string & source, std::initializer_list<int64_t> ne) {
        w.add_f32("syn.voc." + std::to_string(n++), ne, source);
    };
    voc("vocoder:tts.ttl.normalizer.scale", { 8 });
    voc("vocoder:tts.ae.latent_mean", { 24 });
    voc("vocoder:tts.ae.latent_std", { 24 });
    voc("vocoder:node:/decoder/embed/net/Conv#1", { 8 });
    voc("vocoder:node:/decoder/embed/net/Conv#2", { 8 });
    for (int i = 0; i < 10; ++i) {
        const std::string p = "vocoder:tts.ae.decoder.convnext." + std::to_string(i);
        voc(p + ".dwconv.net.weight", { 8 });
        voc(p + ".dwconv.net.bias", { 8 });
        voc(p + ".norm.norm.weight", { 8 });
        voc(p + ".norm.norm.bias", { 8 });
        voc(p + ".pwconv1.weight", { 8 });
        voc(p + ".pwconv1.bias", { 8 });
        voc(p + ".pwconv2.weight", { 8 });
        voc(p + ".pwconv2.bias", { 8 });
        voc(p + ".gamma", { 8 });
    }
    // F2 BN pre-bake: exactly 512 wide, and running_var must keep
    // sqrt(var + eps) real -- the writer fills 0.5f, which is fine.
    voc("vocoder:tts.ae.decoder.final_norm.norm.weight", { 512 });
    voc("vocoder:tts.ae.decoder.final_norm.norm.bias", { 512 });
    voc("vocoder:tts.ae.decoder.final_norm.norm.running_mean", { 512 });
    voc("vocoder:tts.ae.decoder.final_norm.norm.running_var", { 512 });
    voc("vocoder:tts.ae.decoder.head.layer1.net.weight", { 8 });
    voc("vocoder:tts.ae.decoder.head.layer1.net.bias", { 8 });
    voc("vocoder:node:/decoder/head/act/PRelu#1", { 8 });
    voc("vocoder:tts.ae.decoder.head.layer2.weight", { 8 });

    // Load-time-mandatory RoPE theta.
    voc("vector_estimator:tts.ttl.vector_field.main_blocks.3.attn.theta", { 64 });

    // The four F6 t_proj sources at their audited [512, 64] shape: on a
    // non-CPU backend these join the pre-transpose roster (F6 on the F32
    // path, the generic ":onnx::MatMul_" scan otherwise), so the synthetic
    // exercises buffer_w_extra sizing parity too.
    voc("vector_estimator:onnx::MatMul_3095", { 512, 64 });
    voc("vector_estimator:onnx::MatMul_3140", { 512, 64 });
    voc("vector_estimator:onnx::MatMul_3185", { 512, 64 });
    voc("vector_estimator:onnx::MatMul_3230", { 512, 64 });

    {
        std::vector<const char *> tn, sn;
        for (const auto & s : w.tensor_names) tn.push_back(s.c_str());
        for (const auto & s : w.source_names) sn.push_back(s.c_str());
        gguf_set_arr_str(w.g, "supertonic.tensor_names", tn.data(), (int) tn.size());
        gguf_set_arr_str(w.g, "supertonic.source_names", sn.data(), (int) sn.size());
    }

    if (!gguf_write_to_file(w.g, path.c_str(), /*only_meta=*/false)) {
        std::fprintf(stderr, "FATAL: cannot write %s\n", path.c_str());
        std::exit(2);
    }
    ggml_free(w.ctx);
    gguf_free(w.g);
    return path;
}

void run_synthetic_gates() {
    const std::string path = write_tiny_supertonic_gguf();

    // 1. Weight-sizing parity: metadata-only twin vs a real load of the same
    //    file, byte for byte per buffer (buffer_w, and buffer_w_extra when
    //    the backend pre-transposes).
    {
        supertonic_model mm, real;
        supertonic_fit_load_measure fm;
        if (!load_supertonic_gguf_metadata_only(path, mm, /*n_gpu_layers=*/99,
                                                /*f16_weights=*/-1,
                                                supertonic_precision::Auto,
                                                /*vulkan_device=*/0, {}, fm)) {
            fail("synthetic: load_supertonic_gguf_metadata_only failed");
            return;
        }
        if (!load_supertonic_gguf(path, real, /*n_gpu_layers=*/99)) {
            fail("synthetic: real load_supertonic_gguf failed");
            free_supertonic_model(mm);
            return;
        }
        expect(real.buffer_w != nullptr, "synthetic: real load produced no weight buffer");
        if (real.buffer_w) {
            expect_eq(fm.weights_bytes, ggml_backend_buffer_get_size(real.buffer_w),
                      "synthetic buffer_w parity");
        }
        if (real.buffer_w_extra) {
            expect_eq(fm.extra_bytes, ggml_backend_buffer_get_size(real.buffer_w_extra),
                      "synthetic buffer_w_extra parity");
        } else {
            expect_eq(fm.extra_bytes, 0, "synthetic extra bytes with no extra buffer");
        }
        free_supertonic_model(real);
        free_supertonic_model(mm);
    }

    // 2. Refusal paths through the public fit_params entry point.
    tts_cpp::supertonic::FitOptions fopts;
    fopts.model_gguf_path = path;
    fopts.n_gpu_layers    = 99;
    fopts.text_tokens     = 32;
    fopts.audio_seconds   = 4.0f;
    {
        // The synthetic has no stage graph tensors, so a full projection can
        // never complete: Error always, FITS never. On the CPU pointwise
        // path the refusal fires first; on a fused path the stage measure fails.
        const tts_cpp::FitResult fr = tts_cpp::supertonic::fit_params(fopts);
        expect(fr.status == tts_cpp::FitStatus::Error,
               "synthetic partial model was not Error (" + fr.reason + ")");
        expect(fr.reason == "compute-path-not-supported" ||
                   fr.reason == "measurement-failed",
               "synthetic partial model: unexpected reason (" + fr.reason + ")");
    }
    {
        // CPU is not synonymous with the unmodelled pointwise path. Without
        // Accelerate/CBLAS the fused graph path reaches the missing stage
        // tensors, so it must fail measurement rather than refuse dispatch.
        tts_cpp::supertonic::FitOptions cpu = fopts;
        cpu.n_gpu_layers = 0;
        const tts_cpp::FitResult fr = tts_cpp::supertonic::fit_params(cpu);
        const std::string expected_reason = cpu_pointwise_accel_compiled()
            ? "compute-path-not-supported" : "measurement-failed";
        expect(fr.device_is_cpu, "synthetic explicit CPU request selected a non-CPU device");
        expect(fr.status == tts_cpp::FitStatus::Error &&
                   !fr.fits && fr.reason == expected_reason,
               "synthetic CPU request: expected " + expected_reason + ", got " + fr.reason);
    }
    {
        tts_cpp::supertonic::FitOptions huge = fopts;
        huge.text_tokens = std::numeric_limits<int>::max();
        const tts_cpp::FitResult fr = tts_cpp::supertonic::fit_params(huge);
        expect(fr.status == tts_cpp::FitStatus::Error &&
                   fr.reason == "workload-too-large",
               "synthetic near-INT_MAX text_tokens was not workload-too-large");
    }
    {
        tts_cpp::supertonic::FitOptions bad = fopts;
        bad.precision = "int4";
        const tts_cpp::FitResult fr = tts_cpp::supertonic::fit_params(bad);
        expect(fr.status == tts_cpp::FitStatus::Error &&
                   fr.reason == "invalid-arguments",
               "synthetic junk precision was not invalid-arguments");
    }
    {
        tts_cpp::supertonic::FitOptions bad = fopts;
        bad.model_gguf_path = path + ".does-not-exist";
        const tts_cpp::FitResult fr = tts_cpp::supertonic::fit_params(bad);
        expect(fr.status == tts_cpp::FitStatus::Error &&
                   fr.reason == "model-unreadable",
               "synthetic missing model was not model-unreadable (" + fr.reason + ")");
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

void run_gates(const std::string & path, int n_gpu_layers) {
    // ── Loads: metadata-only twin vs real ───────────────────────────────────
    supertonic_model mm, real;
    supertonic_fit_load_measure fm;
    if (!load_supertonic_gguf_metadata_only(path, mm, n_gpu_layers, /*f16_weights=*/-1,
                                            supertonic_precision::F32, /*vulkan_device=*/0,
                                            {}, fm)) {
        fail("load_supertonic_gguf_metadata_only failed");
        return;
    }
    if (!load_supertonic_gguf(path, real, n_gpu_layers)) {
        fail("real load_supertonic_gguf failed");
        free_supertonic_model(mm);
        return;
    }
    const bool cpu_pointwise_path = model_prefers_cpu_kernels(real);

    // 1. Weight parity, byte for byte, per buffer.
    expect(real.buffer_w != nullptr, "real load produced no weight buffer");
    if (real.buffer_w) {
        expect_eq(fm.weights_bytes, ggml_backend_buffer_get_size(real.buffer_w),
                  "buffer_w parity");
    }
    if (real.buffer_w_extra) {
        expect_eq(fm.extra_bytes, ggml_backend_buffer_get_size(real.buffer_w_extra),
                  "buffer_w_extra parity");
    } else {
        expect_eq(fm.extra_bytes, 0, "extra bytes on a path with no extra buffer");
    }

    // 2. Stage arena parity: size-only measure (metadata model) vs real
    //    reservation (real model) of the same graphs.
    const int L = 32;
    const int latent_len = 64;
    const int steps = real.hparams.default_steps;
    std::string error;
    {
        uint64_t meas = 0, probe = 0;
        if (!supertonic_fit_measure_text_encoder(mm, L, meas, &error)) {
            fail("text encoder measure failed: " + error);
        } else if (!supertonic_fit_parity_probe_text_encoder(real, L, probe, &error)) {
            fail("text encoder probe failed: " + error);
        } else if (probe > 0) {
            expect_eq(meas, probe, "text encoder cache-set parity");
        }
    }
    {
        uint64_t meas = 0, probe = 0;
        if (!supertonic_fit_measure_duration(mm, L + 1, meas, &error)) {
            fail("duration measure failed: " + error);
        } else if (!supertonic_fit_parity_probe_duration(real, L + 1, probe, &error)) {
            fail("duration probe failed: " + error);
        } else if (probe > 0) {
            expect_eq(meas, probe, "duration cache parity");
        }
    }
    if (!cpu_pointwise_path) {
        uint64_t meas = 0, probe = 0;
        if (!supertonic_fit_measure_vector(mm, latent_len, L, steps, meas, &error)) {
            fail("vector estimator measure failed: " + error);
        } else if (!supertonic_fit_parity_probe_vector(real, latent_len, L, steps,
                                                       probe, &error)) {
            fail("vector estimator probe failed: " + error);
        } else if (probe > 0) {
            expect_eq(meas, probe, "vector estimator loop-graph parity");
        }
    } else {
        uint64_t meas = 0;
        expect(!supertonic_fit_measure_vector(mm, latent_len, L, steps, meas, &error),
               "CPU vector-estimator path was not refused by the measure");
    }
    // CFG arena delta (supertonic3 only): the [C, T] CFM loop batches
    // cond | uncond along time (batch=2), so the loop arena on a CFG model
    // must strictly exceed the same shapes priced with CFG off.  Pins the
    // CFG multiplier's presence and sign directly on the metadata model,
    // independent of which other fixture lanes are registered.
    if (!cpu_pointwise_path && mm.hparams.cfg_enabled()) {
        uint64_t cfg_bytes = 0, nocfg_bytes = 0;
        if (!supertonic_fit_measure_vector(mm, latent_len, L, steps, cfg_bytes, &error)) {
            fail("CFG vector measure failed: " + error);
        } else {
            const float saved = mm.hparams.cfg_uncond_scale;
            mm.hparams.cfg_uncond_scale = 0.0f;  // cfg_enabled() -> false
            const bool ok = supertonic_fit_measure_vector(mm, latent_len, L, steps,
                                                          nocfg_bytes, &error);
            mm.hparams.cfg_uncond_scale = saved;
            if (!ok) {
                fail("CFG-off vector measure failed: " + error);
            } else if (cfg_bytes <= nocfg_bytes) {
                fail("CFG loop arena (" + std::to_string(cfg_bytes) +
                     ") does not exceed the CFG-off arena (" +
                     std::to_string(nocfg_bytes) + ")");
            }
        }
    }
    {
        uint64_t meas_dev = 0, meas_host = 0, probe = 0;
        if (!supertonic_fit_measure_vocoder(mm, latent_len, meas_dev, meas_host, &error)) {
            fail("vocoder measure failed: " + error);
        } else if (!supertonic_fit_parity_probe_vocoder(real, latent_len, probe, &error)) {
            fail("vocoder probe failed: " + error);
        } else if (probe > 0 && meas_host == 0) {
            expect_eq(meas_dev, probe, "vocoder cache parity");
        }
    }

    free_supertonic_model(real);
    free_supertonic_model(mm);

    // ── fit_params end-to-end ────────────────────────────────────────────────
    tts_cpp::supertonic::FitOptions fopts;
    fopts.model_gguf_path = path;
    fopts.n_gpu_layers    = n_gpu_layers;
    fopts.text_tokens     = 32;
    fopts.audio_seconds   = 4.0f;

    const tts_cpp::FitResult fit = tts_cpp::supertonic::fit_params(fopts);
    if (cpu_pointwise_path) {
        // The CPU pointwise multi-cache path is not modelled yet.
        expect(fit.status == tts_cpp::FitStatus::Error &&
                   fit.reason == "compute-path-not-supported",
               "CPU pointwise projection was not refused (" + fit.reason + ")");
    } else {
        expect(fit.status != tts_cpp::FitStatus::Error,
               "fit_params returned Error (" + fit.reason + ") for a readable model");
        expect(fit.device.weights_bytes > 0,       "projected weights_bytes == 0");
        expect(fit.device.lm_compute_bytes > 0,    "projected stage-cache bytes == 0");
        expect(fit.device.codec_compute_bytes > 0, "projected vocoder bytes == 0");
        expect(fit.host_bytes > 0,                 "projected host_bytes == 0");
        expect(!fit.report.empty(),                "empty report");
        if (g_failures == 0) std::printf("%s", fit.report.c_str());

        // The relpos masks make the text-encoder caches quadratic in the
        // text length; a longer utterance grows the estimator/vocoder.
        tts_cpp::supertonic::FitOptions more = fopts;
        more.text_tokens = 96;
        const tts_cpp::FitResult big_text = tts_cpp::supertonic::fit_params(more);
        expect(big_text.device.lm_compute_bytes > fit.device.lm_compute_bytes,
               "stage caches did not grow with text_tokens");
        more = fopts;
        more.audio_seconds = 12.0f;
        const tts_cpp::FitResult big_audio = tts_cpp::supertonic::fit_params(more);
        expect(big_audio.device.total_bytes > fit.device.total_bytes,
               "projection did not grow with audio_seconds");
    }

    // An explicit CPU request must follow the same dispatch contract even
    // when the fixture gates above ran on a GPU.
    {
        tts_cpp::supertonic::FitOptions cpu = fopts;
        cpu.n_gpu_layers = 0;
        const tts_cpp::FitResult fr = tts_cpp::supertonic::fit_params(cpu);
        expect(fr.device_is_cpu, "explicit CPU request selected a non-CPU device");
        if (cpu_pointwise_accel_compiled()) {
            expect(fr.status == tts_cpp::FitStatus::Error &&
                       !fr.fits && fr.reason == "compute-path-not-supported",
                   "CPU pointwise request was not refused (" + fr.reason + ")");
        } else {
            expect(fr.status != tts_cpp::FitStatus::Error,
                   "fused CPU projection failed for a readable model (" + fr.reason + ")");
            expect(fr.device.weights_bytes > 0 && fr.device.lm_compute_bytes > 0 &&
                       fr.device.codec_compute_bytes > 0 && fr.host_bytes > 0,
                   "fused CPU projection omitted memory allocations");
        }
    }

    // Strict workload / argument rejection.
    {
        tts_cpp::supertonic::FitOptions huge = fopts;
        huge.text_tokens = std::numeric_limits<int>::max();
        const tts_cpp::FitResult fr = tts_cpp::supertonic::fit_params(huge);
        expect(fr.status == tts_cpp::FitStatus::Error &&
                   fr.reason == "workload-too-large",
               "near-INT_MAX text_tokens was not workload-too-large");
    }
    {
        tts_cpp::supertonic::FitOptions bad = fopts;
        bad.precision = "int4";
        const tts_cpp::FitResult fr = tts_cpp::supertonic::fit_params(bad);
        expect(fr.status == tts_cpp::FitStatus::Error &&
                   fr.reason == "invalid-arguments",
               "junk precision was not invalid-arguments");
    }
    {
        tts_cpp::supertonic::FitOptions bad = fopts;
        bad.model_gguf_path = path + ".does-not-exist";
        const tts_cpp::FitResult fr = tts_cpp::supertonic::fit_params(bad);
        expect(fr.status == tts_cpp::FitStatus::Error &&
                   fr.reason == "model-unreadable",
               "missing model was not model-unreadable (" + fr.reason + ")");
    }
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        // No-argument (always-on CI) form: metadata-only synthetic GGUF.
        // Weight-sizing parity + the refusal paths need no fixture; the
        // stage-arena parity gates still do (the vocoder/estimator channel
        // plans are hardcoded to the real architecture).
        run_synthetic_gates();
        if (g_failures == 0) {
            std::printf("test-supertonic-fit-params (synthetic): all checks passed\n");
        }
        return g_failures;
    }
    const int n_gpu_layers = argc > 2 ? std::atoi(argv[2]) : 99;
    run_gates(argv[1], n_gpu_layers);
    if (g_failures == 0) {
        std::printf("test-supertonic-fit-params: all checks passed\n");
    }
    return g_failures;
}
