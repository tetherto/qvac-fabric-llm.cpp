#define _USE_MATH_DEFINES  // For M_PI on MSVC

// vae-decode: load the REAL ACE-Step Oobleck VAE decoder weights (GGUF) and run a
// full decode on the CPU backend, writing a 48 kHz stereo WAV.
//
// Exercises the two custom ops we ported into ggml-speech (ggml_col2im_1d +
// ggml_snake) end-to-end on real bf16 weights, with weight_norm fusion at load.
//
// Usage:  vae-decode [model.gguf] [out.wav] [T_latent] [threads]
//
// NOTE: without the DiT/LM stages we have no real latent, so we feed a structured
// synthetic latent (per-channel low-frequency sinusoids). Output is a deterministic
// texture, not music -- the point is that real weights load + fuse and the full
// decode graph produces finite, coherent 48 kHz stereo audio on CPU. For an audible
// reconstruction of real audio use vae-roundtrip.

#include "vae-common.h"

#include <thread>

int main(int argc, char ** argv) {
    const char * gguf_path = (argc > 1) ? argv[1] : "models/vae-BF16.gguf";
    const char * out_path  = (argc > 2) ? argv[2] : "vae_out.wav";
    const int    T_latent  = (argc > 3) ? atoi(argv[3]) : 32;
    int          nthreads  = (argc > 4) ? atoi(argv[4]) : (int) std::thread::hardware_concurrency();
    if (nthreads <= 0) nthreads = 4;

    GGUF g;
    if (!gguf_open(g, gguf_path)) return 1;

    ggml_init_params wp{ ggml_tensor_overhead() * 512, nullptr, /*no_alloc=*/true };
    ggml_context_ptr wctx_ptr{ ggml_init(wp) };
    ggml_context *   wctx = wctx_ptr.get();

    VAE m = {};
    decoder_create(m, wctx);

    ggml_backend_ptr backend_ptr{ ggml_backend_cpu_init() };
    ggml_backend_t   backend = backend_ptr.get();
    ggml_backend_cpu_set_n_threads(backend, nthreads);

    ggml_backend_buffer_ptr wbuf{ ggml_backend_alloc_ctx_tensors(wctx, backend) };
    if (!wbuf) { fprintf(stderr, "[VAE] failed to alloc weight buffer\n"); return 1; }
    fprintf(stderr, "[VAE] weight buffer: %.1f MB, threads=%d\n",
            (float) ggml_backend_buffer_get_size(wbuf.get()) / (1024 * 1024), nthreads);

    decoder_load(m, g);
    gguf_close(g);
    fprintf(stderr, "[VAE] weights loaded + weight_norm fused (5 blocks, 1920x upsample)\n");

    ggml_init_params gp{ ggml_tensor_overhead() * 1024 + ggml_graph_overhead_custom(8192, false),
                         nullptr, /*no_alloc=*/true };
    ggml_context_ptr gctx_ptr{ ggml_init(gp) };
    ggml_context *   gctx = gctx_ptr.get();

    ggml_tensor * latent = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, T_latent, 64);
    ggml_set_name(latent, "latent");
    ggml_set_input(latent);
    ggml_tensor * out = build_decode(gctx, &m, latent);
    ggml_set_name(out, "audio");
    ggml_set_output(out);

    ggml_cgraph * gf = ggml_new_graph_custom(gctx, 8192, false);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_ptr galloc{ ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)) };
    if (!ggml_gallocr_alloc_graph(galloc.get(), gf)) {
        fprintf(stderr, "[VAE] gallocr alloc failed (T_latent=%d)\n", T_latent);
        return 1;
    }

    // structured synthetic latent: per-channel low-frequency sinusoids
    // ggml input is [T (ne0), 64 (ne1)] channel-major: idx = c*T + t.
    std::vector<float> lat((size_t) T_latent * 64);
    for (int c = 0; c < 64; ++c) {
        float freq = 0.5f + 0.15f * (float) c, phase = 0.37f * (float) c, amp = 3.0f * expf(-(float) c / 48.0f);
        for (int t = 0; t < T_latent; ++t) {
            float u = (float) t / (float) T_latent;
            lat[(size_t) c * T_latent + t] = amp * sinf(2.0f * (float) M_PI * freq * u + phase);
        }
    }
    ggml_backend_tensor_set(latent, lat.data(), 0, ggml_nbytes(latent));

    fprintf(stderr, "[VAE] decoding T_latent=%d -> T_audio=%d (%.2fs), graph nodes=%d ...\n",
            T_latent, T_latent * UPSAMPLE, (float) (T_latent * UPSAMPLE) / 48000.0f, ggml_graph_n_nodes(gf));
    int64_t t0 = ggml_time_us();
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[VAE] graph compute failed\n");
        return 1;
    }
    int64_t t1 = ggml_time_us();

    const int T_audio = (int) out->ne[0], OC = (int) out->ne[1];
    std::vector<float> audio((size_t) T_audio * OC);
    ggml_backend_tensor_get(out, audio.data(), 0, ggml_nbytes(out));

    bool finite = true; float amax = 0.0f;
    for (float v : audio) { if (!std::isfinite(v)) { finite = false; break; } amax = std::max(amax, std::abs(v)); }

    const bool shape_ok = (T_audio == T_latent * UPSAMPLE) && (OC == 2);
    fprintf(stderr, "[VAE] done in %.2fs  out=[%d,%d] (expected [%d,2])  |max|=%.4f  finite=%s\n",
            (t1 - t0) / 1e6, T_audio, OC, T_latent * UPSAMPLE, amax, finite ? "yes" : "no");
    if (!shape_ok || !finite) { fprintf(stderr, "\033[31mFAILED\033[0m\n"); return 1; }

    wav_write(out_path, audio.data(), audio.data() + T_audio, T_audio, 48000, /*normalize=*/true);
    fprintf(stderr, "\033[32mOK: real-weight VAE decode -> %s\033[0m\n", out_path);
    return 0;
}
