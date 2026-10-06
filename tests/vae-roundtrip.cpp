// vae-roundtrip: REAL end-to-end audio reconstruction through the ACE-Step Oobleck
// VAE on CPU. Reads a 48 kHz WAV, encodes to the 64-ch latent, decodes
// back to audio, writes the reconstruction and reports how close it is to the input.
//
// This is the audible "it works": a real waveform in -> real waveform out, using the
// encoder (strided conv1d + snake) and decoder (col2im_1d + snake) graphs on real
// bf16 weights. No DiT/LM needed -- decode(encode(x).mean) is the raw VAE roundtrip.
//
// Usage:  vae-roundtrip [model.gguf] [in.wav] [out.wav] [seconds] [threads]

#include "vae-common.h"

#include <thread>

// Pearson correlation between a and b over n samples at a given integer lag
// (b shifted by `lag`). Used to score reconstruction quality, tolerant to the
// small group delay the conv stack introduces.
static double corr_at_lag(const float * a, const float * b, int n, int lag) {
    int i0 = std::max(0, -lag);
    int i1 = std::min(n, n - lag);
    int m  = i1 - i0;
    if (m <= 1) return 0.0;
    double sa = 0, sb = 0;
    for (int i = i0; i < i1; i++) { sa += a[i]; sb += b[i + lag]; }
    double ma = sa / m, mb = sb / m, cov = 0, va = 0, vb = 0;
    for (int i = i0; i < i1; i++) {
        double da = a[i] - ma, db = b[i + lag] - mb;
        cov += da * db; va += da * da; vb += db * db;
    }
    if (va <= 0 || vb <= 0) return 0.0;
    return cov / std::sqrt(va * vb);
}

static double rms(const float * x, int n) {
    double s = 0; for (int i = 0; i < n; i++) s += (double) x[i] * x[i];
    return std::sqrt(s / std::max(1, n));
}

// Build + alloc + compute a graph with a single input tensor set from `input`.
// Returns the output tensor's data copied to `out_host` and its ne[0]/ne[1].
struct RunResult { int ne0, ne1; std::vector<float> data; };

int main(int argc, char ** argv) {
    const char * gguf_path = (argc > 1) ? argv[1] : "models/vae-BF16.gguf";
    const char * in_path   = (argc > 2) ? argv[2] : "in.wav";
    const char * out_path  = (argc > 3) ? argv[3] : "vae_roundtrip.wav";
    const float  seconds   = (argc > 4) ? (float) atof(argv[4]) : 2.56f;
    int          nthreads  = (argc > 5) ? atoi(argv[5]) : (int) std::thread::hardware_concurrency();
    if (nthreads <= 0) nthreads = 4;

    // ---- read input WAV ----
    int in_T = 0, in_rate = 0;
    std::vector<float> in_audio = wav_read(in_path, &in_T, &in_rate);  // [T,2] interleaved
    if (in_audio.empty()) return 1;
    fprintf(stderr, "[RT] input: %s  %d frames @ %d Hz\n", in_path, in_T, in_rate);
    if (in_rate != 48000) {
        fprintf(stderr, "[RT] WARNING: VAE expects 48 kHz; input is %d Hz (feeding as-is)\n", in_rate);
    }

    // trim to a latent-aligned window, capped at `seconds`
    int cap   = (int) (seconds * 48000.0f);
    int T_aud = std::min(in_T, cap);
    T_aud     = (T_aud / UPSAMPLE) * UPSAMPLE;  // multiple of 1920
    if (T_aud < UPSAMPLE) { fprintf(stderr, "[RT] input too short\n"); return 1; }
    fprintf(stderr, "[RT] using %d frames (%.2fs), threads=%d\n", T_aud, (float) T_aud / 48000.0f, nthreads);

    // ---- load encoder + decoder weights into one buffer ----
    GGUF g;
    if (!gguf_open(g, gguf_path)) return 1;

    ggml_init_params wp{ ggml_tensor_overhead() * 1024, nullptr, /*no_alloc=*/true };
    ggml_context_ptr wctx_ptr{ ggml_init(wp) };
    ggml_context *   wctx = wctx_ptr.get();

    Encoder enc = {};
    VAE     dec = {};
    encoder_create(enc, wctx);
    decoder_create(dec, wctx);

    ggml_backend_ptr backend_ptr{ ggml_backend_cpu_init() };
    ggml_backend_t   backend = backend_ptr.get();
    ggml_backend_cpu_set_n_threads(backend, nthreads);

    ggml_backend_buffer_ptr wbuf{ ggml_backend_alloc_ctx_tensors(wctx, backend) };
    if (!wbuf) { fprintf(stderr, "[RT] failed to alloc weight buffer\n"); return 1; }
    fprintf(stderr, "[RT] weight buffer (enc+dec): %.1f MB\n",
            (float) ggml_backend_buffer_get_size(wbuf.get()) / (1024 * 1024));

    encoder_load(enc, g);
    decoder_load(dec, g);
    gguf_close(g);
    fprintf(stderr, "[RT] weights loaded + fused (encoder + decoder)\n");

    // =================== ENCODE: audio [T_aud,2] -> latent mean [T_lat,64] ===================
    std::vector<float> latent;  // [T_lat, 64] time-major
    int T_lat = 0;
    {
        ggml_init_params gp{ ggml_tensor_overhead() * 1024 + ggml_graph_overhead_custom(8192, false),
                             nullptr, true };
        ggml_context_ptr gctx_ptr{ ggml_init(gp) };
        ggml_context *   gctx = gctx_ptr.get();

        ggml_tensor * a = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, T_aud, 2);
        ggml_set_input(a);
        ggml_tensor * z = build_encode(gctx, &enc, a);  // [T_lat, 128]
        ggml_set_output(z);

        ggml_cgraph * gf = ggml_new_graph_custom(gctx, 8192, false);
        ggml_build_forward_expand(gf, z);
        ggml_gallocr_ptr ga{ ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)) };
        if (!ggml_gallocr_alloc_graph(ga.get(), gf)) { fprintf(stderr, "[RT] encode alloc failed\n"); return 1; }

        // input ggml [T_aud(ne0), 2(ne1)] channel-major: idx = c*T_aud + t
        std::vector<float> ain((size_t) T_aud * 2);
        for (int c = 0; c < 2; c++)
            for (int t = 0; t < T_aud; t++) ain[(size_t) c * T_aud + t] = in_audio[(size_t) t * 2 + c];
        ggml_backend_tensor_set(a, ain.data(), 0, ain.size() * sizeof(float));

        fprintf(stderr, "[RT] encoding (nodes=%d) ...\n", ggml_graph_n_nodes(gf));
        int64_t t0 = ggml_time_us();
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "[RT] encode failed\n"); return 1; }
        int64_t t1 = ggml_time_us();

        T_lat = (int) z->ne[0];
        int ZC = (int) z->ne[1];
        std::vector<float> raw((size_t) T_lat * ZC);
        ggml_backend_tensor_get(z, raw.data(), 0, ggml_nbytes(z));
        // extract mean channels 0..63, store time-major [T_lat,64]
        latent.resize((size_t) T_lat * 64);
        for (int t = 0; t < T_lat; t++)
            for (int c = 0; c < 64; c++) latent[(size_t) t * 64 + c] = raw[(size_t) c * T_lat + t];

        double lmin = 1e30, lmax = -1e30, lsq = 0;
        for (float v : latent) { lmin = std::min(lmin, (double) v); lmax = std::max(lmax, (double) v); lsq += (double) v * v; }
        fprintf(stderr, "[RT] encoded in %.2fs -> latent [%d,64]  (mean range [%.3f, %.3f], rms %.3f)\n",
                (t1 - t0) / 1e6, T_lat, lmin, lmax, std::sqrt(lsq / latent.size()));
    }

    // =================== DECODE: latent [T_lat,64] -> audio [T_out,2] ===================
    std::vector<float> rec;  // planar: ch0 [T_out], ch1 [T_out]
    int T_out = 0;
    {
        ggml_init_params gp{ ggml_tensor_overhead() * 1024 + ggml_graph_overhead_custom(8192, false),
                             nullptr, true };
        ggml_context_ptr gctx_ptr{ ggml_init(gp) };
        ggml_context *   gctx = gctx_ptr.get();

        ggml_tensor * lat = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, T_lat, 64);
        ggml_set_input(lat);
        ggml_tensor * out = build_decode(gctx, &dec, lat);  // [T_out, 2]
        ggml_set_output(out);

        ggml_cgraph * gf = ggml_new_graph_custom(gctx, 8192, false);
        ggml_build_forward_expand(gf, out);
        ggml_gallocr_ptr ga{ ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)) };
        if (!ggml_gallocr_alloc_graph(ga.get(), gf)) { fprintf(stderr, "[RT] decode alloc failed\n"); return 1; }

        // decode input ggml [T_lat(ne0), 64(ne1)] channel-major: idx = c*T_lat + t
        std::vector<float> lin((size_t) T_lat * 64);
        for (int c = 0; c < 64; c++)
            for (int t = 0; t < T_lat; t++) lin[(size_t) c * T_lat + t] = latent[(size_t) t * 64 + c];
        ggml_backend_tensor_set(lat, lin.data(), 0, lin.size() * sizeof(float));

        fprintf(stderr, "[RT] decoding (nodes=%d) ...\n", ggml_graph_n_nodes(gf));
        int64_t t0 = ggml_time_us();
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "[RT] decode failed\n"); return 1; }
        int64_t t1 = ggml_time_us();

        T_out = (int) out->ne[0];
        rec.resize((size_t) T_out * 2);
        ggml_backend_tensor_get(out, rec.data(), 0, ggml_nbytes(out));
        fprintf(stderr, "[RT] decoded in %.2fs -> audio [%d,2] (%.2fs)\n",
                (t1 - t0) / 1e6, T_out, (float) T_out / 48000.0f);
    }

    // =================== compare input vs reconstruction ===================
    const float * rec_l = rec.data();
    const float * rec_r = rec.data() + T_out;
    int N = std::min(T_aud, T_out);

    // deinterleave input to planar for correlation
    std::vector<float> in_l(N), in_r(N);
    for (int t = 0; t < N; t++) { in_l[t] = in_audio[(size_t) t * 2 + 0]; in_r[t] = in_audio[(size_t) t * 2 + 1]; }

    // best lag from channel-0 (conv stack introduces a small group delay)
    int best_lag = 0; double best = -2;
    for (int lag = -512; lag <= 512; lag++) {
        double c = corr_at_lag(in_l.data(), rec_l, N, lag);
        if (c > best) { best = c; best_lag = lag; }
    }
    double cL = corr_at_lag(in_l.data(), rec_l, N, best_lag);
    double cR = corr_at_lag(in_r.data(), rec_r, N, best_lag);

    fprintf(stderr, "\n[RT] ===== reconstruction quality =====\n");
    fprintf(stderr, "[RT] compared %d frames, best lag %d samples\n", N, best_lag);
    fprintf(stderr, "[RT] correlation  L=%.4f  R=%.4f  (1.0 = identical shape)\n", cL, cR);
    fprintf(stderr, "[RT] RMS  in: L=%.4f R=%.4f   out: L=%.4f R=%.4f\n",
            rms(in_l.data(), N), rms(in_r.data(), N), rms(rec_l, N), rms(rec_r, N));

    // write reconstruction (peak-normalized so it's comfortably audible)
    wav_write(out_path, rec_l, rec_r, T_out, 48000, /*normalize=*/true);

    // also write the trimmed input we actually compared against, for A/B listening
    std::string ref_path = std::string(out_path) + ".ref.wav";
    wav_write(ref_path.c_str(), in_l.data(), in_r.data(), N, 48000, /*normalize=*/false);

    fprintf(stderr, "\033[32m[RT] OK: roundtrip done -> %s (ref: %s)\033[0m\n", out_path, ref_path.c_str());
    return 0;
}
