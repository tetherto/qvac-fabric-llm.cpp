// Fit-projection parity tests (include/audiogen-cpp/acestep/fit.h): assert
// that the metadata-only memory projection matches what a REAL load actually
// allocates, byte for byte where the projection is exact by construction:
//
//   1. fit_params returns a projection (never Error) for a readable model set,
//      with non-zero per-stage weight and compute figures and a report;
//   2. weights parity per stage -- the metadata-only sizing (alloc + mmap, and
//      the LM KV cache) equals what a real loader allocates on the same
//      backend (both run the same tensor wiring and buffer-type sizing);
//   3. compute parity per stage -- the size-only graph measurement equals the
//      buffer a real forward/decode actually allocates for the same shape
//      (ggml_gallocr_reserve_n_size is the size-only twin of the real
//      allocations); the VAE gate is exact on CPU and bounded by one
//      graph-input slot against a GPU sched (see vae_measure_graph);
//   4. the projection grows with the workload (a longer generation must never
//      project smaller), and a missing model file is Error, never Success.
//
// Usage: test-fit-params [models_dir] [n_gpu_layers]
// models_dir defaults to $AUDIOGEN_TEST_MODELS_DIR (the directory the
// integration test uses); absent -> exit 77 (ctest SKIP_RETURN_CODE), so the
// test self-disables where the fixture is not staged. n_gpu_layers defaults to
// 0 (the only backend every CI lane has); pass e.g. 99 manually to check
// parity on a GPU backend.

#include "audiogen-cpp/acestep/fit.h"
#include "audiogen-cpp/acestep/engine.h"

#include "cond_ggml.h"
#include "detok_ggml.h"
#include "dit_ggml.h"
#include "engine_backends.h"
#include "engine_paths.h"
#include "fit_measure.h"
#include "lm_ggml.h"
#include "textenc_ggml.h"
#include "vae_ggml.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace tts_cpp::acestep;

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
             " != real " + std::to_string(real));
    }
}

}  // namespace

int main(int argc, char ** argv) {
    const char * dir_env    = std::getenv("AUDIOGEN_TEST_MODELS_DIR");
    const char * models_dir = argc > 1 ? argv[1] : dir_env;
    if (!models_dir || !*models_dir) {
        std::fprintf(stderr,
                     "test-fit-params: AUDIOGEN_TEST_MODELS_DIR not set and no dir argument; skipping\n");
        return 77;
    }
    const int n_gpu_layers = argc > 2 ? std::atoi(argv[2]) : 0;

    EngineOptions paths;
    paths.models_dir = models_dir;
    resolve_stage_paths(paths);
    if (paths.text_enc_model_path.empty() || paths.lm_model_path.empty() ||
        paths.dit_model_path.empty() || paths.vae_model_path.empty()) {
        std::fprintf(stderr, "test-fit-params: '%s' lacks the four stage GGUFs; skipping\n", models_dir);
        return 77;
    }

    // ── 1. A readable model set always yields a projection ──────────────────
    FitOptions fopts;
    fopts.models_dir       = models_dir;
    fopts.n_gpu_layers     = n_gpu_layers;
    fopts.duration_seconds = 10.0f;
    fopts.text_tokens      = 96;
    fopts.lyric_tokens     = 128;

    const FitResult fit = fit_params(fopts);
    expect(fit.status != FitStatus::Error,
           "fit_params returned Error (" + fit.reason + ") for a readable model set");
    expect(fit.stages.size() == 6, "expected 6 stage rows");
    for (const FitStageProjection & s : fit.stages) {
        expect(s.weights_bytes + s.weights_mmap_bytes > 0, "stage " + s.name + " projected 0 weight bytes");
        expect(s.compute_bytes > 0, "stage " + s.name + " projected 0 compute bytes");
    }
    expect(!fit.report.empty(), "empty report");
    expect(fit.device_total_bytes > 0, "device_total_bytes == 0");
    expect(fit.host_total_bytes > 0, "host_total_bytes == 0");
    expect(fit.extra_devices.empty(), "automatic resolution placed a stage on a third device");
    if (g_failures) {
        return g_failures;  // nothing below is meaningful without a projection
    }
    std::printf("%s", fit.report.c_str());

    // Backends resolved exactly as fit_params / Engine::create resolve them.
    BackendRequest request;
    request.n_gpu_layers = n_gpu_layers;
    AcestepBackends rb;
    if (!resolve_acestep_backends(request, rb)) {
        fail("resolve_acestep_backends failed");
        return g_failures;
    }

    // ── 2 + 3. Per-stage parity: metadata-only sizing vs a real load, and
    //           size-only graph measurement vs a real forward's allocation ───

    {  // text encoder
        AcestepStageMeasure w{};
        TextEncModel * meta = textenc_model_load_metadata_only(paths.text_enc_model_path, rb.enc, false, w);
        TextEncModel * real = textenc_model_load(paths.text_enc_model_path, rb.enc, false);
        if (!meta || !real) {
            fail("textenc loads failed");
        } else {
            expect_eq(w.weights_alloc_bytes + w.weights_mapped_bytes,
                      textenc_model_weight_bytes(real), "textenc weights parity");

            const int S = 96;
            size_t projected = 0;
            std::vector<float> out;
            std::vector<int32_t> ids((size_t) S, 0);
            expect(textenc_model_forward(meta, nullptr, S, out, &projected), "textenc measure forward");
            expect(textenc_model_forward(real, ids.data(), S, out), "textenc real forward");
            expect_eq(projected, textenc_model_compute_buffer_bytes(real), "textenc compute parity");

            size_t projected_lookup = 0;
            expect(textenc_model_embed_lookup(meta, nullptr, S, out, &projected_lookup),
                   "textenc measure embed lookup");
            expect(textenc_model_embed_lookup(real, ids.data(), S, out), "textenc real embed lookup");
            expect_eq(projected_lookup, textenc_model_compute_buffer_bytes(real),
                      "textenc embed-lookup compute parity");
        }
        if (meta) textenc_model_free(meta);
        if (real) textenc_model_free(real);
    }

    {  // condition encoder (weights live in the DiT GGUF)
        AcestepStageMeasure w{};
        CondModel * meta = cond_model_load_metadata_only(paths.dit_model_path, rb.enc, false, w);
        CondModel * real = cond_model_load(paths.dit_model_path, rb.enc, false);
        if (!meta || !real) {
            fail("cond loads failed");
        } else {
            expect_eq(w.weights_alloc_bytes + w.weights_mapped_bytes,
                      cond_model_weight_bytes(real), "cond weights parity");

            const int S_text = 96, S_lyric = 128;
            size_t projected = 0;
            std::vector<float> out;
            int    out_S = 0;
            // Timbre-less shape on both sides (a timbre-carrying graph is the
            // same builder with one more branch; the fit projection uses it,
            // this gate pins the shared shape byte-for-byte).
            expect(cond_model_forward(meta, nullptr, S_text, nullptr, S_lyric,
                                      nullptr, 0, out, &out_S, &projected),
                   "cond measure forward");
            std::vector<float> text((size_t) 1024 * S_text, 0.0f);
            std::vector<float> lyric((size_t) 1024 * S_lyric, 0.0f);
            expect(cond_model_forward(real, text.data(), S_text, lyric.data(), S_lyric,
                                      nullptr, 0, out, &out_S),
                   "cond real forward");
            expect_eq(projected, cond_model_compute_buffer_bytes(real), "cond compute parity");
        }
        if (meta) cond_model_free(meta);
        if (real) cond_model_free(real);
    }

    {  // FSQ detokenizer (weights live in the DiT GGUF)
        AcestepStageMeasure w{};
        DetokModel * meta = detok_model_load_metadata_only(paths.dit_model_path, rb.detok, false, w);
        DetokModel * real = detok_model_load(paths.dit_model_path, rb.detok, false);
        if (!meta || !real) {
            fail("detok loads failed");
        } else {
            expect_eq(w.weights_alloc_bytes + w.weights_mapped_bytes,
                      detok_model_weight_bytes(real), "detok weights parity");

            size_t projected = 0;
            expect(detok_model_decode(meta, nullptr, 1, nullptr, &projected) >= 0,
                   "detok measure decode");
            const int          code = 0;
            std::vector<float> ctx_out((size_t) 64 * 5, 0.0f);
            expect(detok_model_decode(real, &code, 1, ctx_out.data()) == 5, "detok real decode");
            expect_eq(projected, detok_model_compute_buffer_bytes(real), "detok compute parity");
        }
        if (meta) detok_model_free(meta);
        if (real) detok_model_free(real);
    }

    {  // LM
        AcestepStageMeasure w{};
        LMModel * meta = lm_model_load_metadata_only(paths.lm_model_path, rb.lm, 2048, false, 2, w);
        LMModel * real = lm_model_load(paths.lm_model_path, rb.lm, 2048, false, 2);
        if (!meta || !real) {
            fail("lm loads failed");
        } else {
            expect_eq(w.weights_alloc_bytes + w.weights_mapped_bytes,
                      lm_model_weight_bytes(real), "lm weights parity");
            expect_eq(w.kv_bytes, lm_model_kv_bytes(real), "lm KV parity");

            const int S = 16;
            size_t projected = 0;
            expect(lm_model_measure_prefill(meta, S, 0, projected), "lm measure prefill");
            std::vector<int32_t> ids((size_t) S, 0);
            std::vector<float>   logits;
            expect(lm_model_forward(real, ids.data(), S, logits), "lm real prefill");
            expect_eq(projected, lm_model_compute_buffer_bytes(real), "lm prefill compute parity");

            // The decode step right after the prefill (same padded KV window a
            // real decode allocates next).
            size_t projected_decode = 0;
            expect(lm_model_measure_decode(meta, S + 1, 0, projected_decode), "lm measure decode");
            const int32_t tok = 0;
            expect(lm_model_forward(real, &tok, 1, logits), "lm real decode");
            expect_eq(projected_decode, lm_model_compute_buffer_bytes(real), "lm decode compute parity");
        }
        if (meta) lm_model_free(meta);
        if (real) lm_model_free(real);
    }

    {  // DiT
        AcestepStageMeasure w{};
        DitModel * meta = dit_model_load_metadata_only(paths.dit_model_path, rb.backend, rb.nth, false, w);
        DitModel * real = dit_model_load(paths.dit_model_path, rb.backend, rb.nth, false);
        if (!meta || !real) {
            fail("dit loads failed");
        } else {
            expect_eq(w.weights_alloc_bytes + w.weights_mapped_bytes,
                      dit_model_weight_bytes(real), "dit weights parity");

            const DitConfig & c = dit_model_config(real);
            const int T = c.patch_size * 8;
            const int enc_S = 32;
            const int S = T / c.patch_size;

            DitForwardInputs fin;
            fin.T          = T;
            fin.N          = 1;
            fin.enc_S      = enc_S;
            fin.H_enc      = c.enc_hidden_size;
            fin.sa_mask_sw = reinterpret_cast<const void *>(&fin);
            fin.ca_mask    = reinterpret_cast<const void *>(&fin);
            size_t projected = 0;
            std::vector<float> out;
            expect(dit_model_forward(meta, fin, out, &projected), "dit measure forward");

            std::vector<float>    input((size_t) c.in_channels * T, 0.0f);
            std::vector<float>    hidden((size_t) c.enc_hidden_size * enc_S, 0.0f);
            std::vector<uint16_t> sa((size_t) S * S, 0);
            std::vector<uint16_t> ca((size_t) enc_S * S, 0);
            DitForwardInputs real_in = fin;
            real_in.input_latents = input.data();
            real_in.enc_hidden    = hidden.data();
            real_in.sa_mask_sw    = sa.data();
            real_in.ca_mask       = ca.data();
            expect(dit_model_forward(real, real_in, out), "dit real forward");
            expect_eq(projected, dit_model_compute_buffer_bytes(real), "dit compute parity");
        }
        if (meta) dit_model_free(meta);
        if (real) dit_model_free(real);
    }

    {  // VAE (decode-only load, like the engine's synthesis path)
        AcestepStageMeasure w{};
        VaeModel * meta = vae_model_load_metadata_only(paths.vae_model_path, rb.backend, false, false, w);
        VaeModel * real = vae_model_load(paths.vae_model_path, rb.backend, false, false);
        if (!meta || !real) {
            fail("vae loads failed");
        } else {
            expect_eq(w.weights_alloc_bytes, vae_model_weight_bytes(real), "vae weights parity");

            const int T_latent = 8;  // below the window core: a single-window decode
            size_t backend_b = 0, host_in_b = 0;
            expect(vae_model_measure_decode(meta, T_latent, backend_b, host_in_b), "vae measure decode");
            std::vector<float> latent((size_t) T_latent * 64, 0.0f);
            std::vector<float> pcm;
            expect(vae_model_decode(real, latent.data(), T_latent, pcm) == T_latent * 1920,
                   "vae real decode");
            const uint64_t projected = (uint64_t) backend_b + host_in_b;
            const uint64_t reserved  = vae_model_compute_buffer_bytes(real);
            const bool on_cpu = ggml_backend_dev_type(ggml_backend_get_device(rb.backend)) ==
                                GGML_BACKEND_DEVICE_TYPE_CPU;
            if (on_cpu) {
                // CPU-only sched: the reservation IS the gallocr layout.
                expect_eq(projected, reserved, "vae compute parity");
            } else {
                // GPU sched: the device-side input copy is a reusable node, so
                // the real reservation may come in below the measure by at
                // most one input slot -- never above (strict direction). The
                // slack term allows for device-side alignment padding.
                const uint64_t input_slack = (uint64_t) T_latent * 64 * sizeof(float) + 1024;
                expect(reserved <= projected,
                       "vae UNDER-projection: projected " + std::to_string(projected) +
                       " < sched reserved " + std::to_string(reserved));
                expect(projected <= reserved + input_slack,
                       "vae over-projection beyond the input slot: projected " +
                       std::to_string(projected) + " > reserved " + std::to_string(reserved) +
                       " + input slack " + std::to_string(input_slack));
            }
        }
        if (meta) vae_model_free(meta);
        if (real) vae_model_free(real);
    }

    free_acestep_backends(rb);

    // ── 4. Monotonicity + error paths ───────────────────────────────────────
    {
        FitOptions longer = fopts;
        longer.duration_seconds = 120.0f;
        const FitResult flong = fit_params(longer);
        expect(flong.status != FitStatus::Error, "120 s projection errored");
        expect(flong.peak_device_bytes + flong.peak_host_bytes >=
                   fit.peak_device_bytes + fit.peak_host_bytes,
               "a longer generation projected smaller than a shorter one");
    }
    {
        FitOptions bad = fopts;
        bad.models_dir = "";
        bad.text_enc_model_path = paths.text_enc_model_path;
        bad.lm_model_path       = paths.lm_model_path;
        bad.dit_model_path      = paths.dit_model_path + ".does-not-exist";
        bad.vae_model_path      = paths.vae_model_path;
        const FitResult fr = fit_params(bad);
        expect(fr.status == FitStatus::Error, "missing model was not Error");
        expect(fr.reason == "model-unreadable", "missing model reason was '" + fr.reason + "'");
        expect(!fr.fits, "missing model reported fits");
    }

    if (g_failures == 0) {
        std::printf("test-fit-params: all checks passed\n");
    }
    return g_failures;
}
