// vae-common.h: shared helpers for the ACE-Step Oobleck VAE bring-up on ggml-speech.
// GGUF loader + weight_norm fusion + graph ops (conv1d, snake,
// col2im-based conv_t1d, res_unit) + encoder/decoder builders + WAV I/O.
//
// Logic ported from acestep.cpp src/vae.h and src/vae-enc.h. The two custom ops
// (ggml_col2im_1d + ggml_snake) live in our ggml-speech fork; everything else is
// stock ggml. F32 activations, bf16 weights fused to F16 at load.
#pragma once

#include <ggml.h>
#include <ggml-cpu.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <gguf.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ------------------------------------------------------------------ GGUF loader
struct GGUF {
    gguf_context * ctx      = nullptr;
    ggml_context * meta     = nullptr;
    uint8_t *      map      = nullptr;  // whole-file buffer (portable fread, not mmap)
    size_t         fsize    = 0;
    size_t         data_off = 0;
};

// Read the whole file into a heap buffer. This deliberately avoids POSIX mmap
// (open/mmap/munmap in <sys/mman.h> etc.): ggml-speech is a cross-platform
// vcpkg port and these test targets must also compile on MSVC/Windows, which
// have no <sys/mman.h>. The test GGUFs fit comfortably in RAM, so a plain read
// is equivalent to the previous read-only mapping for `gdata()`'s pointer math.
static bool gguf_open(GGUF & g, const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[GGUF] cannot open %s\n", path); return false; }
    if (fseek(f, 0, SEEK_END) != 0) { fprintf(stderr, "[GGUF] seek failed\n"); fclose(f); return false; }
    const long sz = ftell(f);
    if (sz < 0) { fprintf(stderr, "[GGUF] ftell failed\n"); fclose(f); return false; }
    rewind(f);
    g.fsize = (size_t) sz;
    g.map   = (uint8_t *) malloc(g.fsize);
    if (!g.map) { fprintf(stderr, "[GGUF] alloc %zu failed\n", g.fsize); fclose(f); return false; }
    if (fread(g.map, 1, g.fsize, f) != g.fsize) {
        fprintf(stderr, "[GGUF] read failed\n");
        free(g.map); g.map = nullptr; fclose(f); return false;
    }
    fclose(f);
    struct gguf_init_params p = { /*no_alloc=*/true, /*ctx=*/&g.meta };
    g.ctx = gguf_init_from_file(path, p);
    if (!g.ctx) { fprintf(stderr, "[GGUF] failed to parse %s\n", path); free(g.map); g.map = nullptr; return false; }
    g.data_off = gguf_get_data_offset(g.ctx);
    fprintf(stderr, "[GGUF] %s: %lld tensors, data at offset %zu\n", path,
            (long long) gguf_get_n_tensors(g.ctx), g.data_off);
    return true;
}

static void gguf_close(GGUF & g) {
    if (g.ctx) gguf_free(g.ctx);
    if (g.meta) ggml_free(g.meta);
    if (g.map) free(g.map);
    g = {};
}

static const void * gdata(const GGUF & g, const std::string & name) {
    int64_t idx = gguf_find_tensor(g.ctx, name.c_str());
    if (idx < 0) return nullptr;
    return g.map + g.data_off + gguf_get_tensor_offset(g.ctx, idx);
}

static ggml_tensor * gmeta(const GGUF & g, const std::string & name) {
    return ggml_get_tensor(g.meta, name.c_str());
}

static float bf16_to_f32(uint16_t v) {
    ggml_bf16_t b; b.bits = v; return ggml_bf16_to_fp32(b);
}

// ------------------------------------------------------- weight_norm fusion
static void upload_f32_as(ggml_tensor * dst, const std::vector<float> & w) {
    if (dst->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> h(w.size());
        ggml_fp32_to_fp16_row(w.data(), h.data(), (int) w.size());
        ggml_backend_tensor_set(dst, h.data(), 0, h.size() * sizeof(ggml_fp16_t));
    } else {
        ggml_backend_tensor_set(dst, w.data(), 0, w.size() * sizeof(float));
    }
}

// Conv1d weight_norm: w = g*v/||v||, normalized over PyTorch dim0 (= ggml ne[n-1]).
static void fuse_wn(ggml_tensor * dst, const GGUF & g, const std::string & pfx) {
    ggml_tensor *    mv   = gmeta(g, pfx + ".weight_v");
    const uint16_t * gp   = (const uint16_t *) gdata(g, pfx + ".weight_g");
    const uint16_t * vp   = (const uint16_t *) gdata(g, pfx + ".weight_v");
    const int        nd   = ggml_n_dims(mv);
    const int        dim0 = (int) mv->ne[nd - 1];
    const int        fan  = (int) (ggml_nelements(mv) / dim0);
    std::vector<float> w((size_t) dim0 * fan);
    for (int d = 0; d < dim0; d++) {
        float gv = bf16_to_f32(gp[d]), nsq = 0.0f;
        for (int i = 0; i < fan; i++) { float vv = bf16_to_f32(vp[(size_t) d * fan + i]); nsq += vv * vv; }
        float s = gv / (sqrtf(nsq) + 1e-12f);
        for (int i = 0; i < fan; i++) w[(size_t) d * fan + i] = bf16_to_f32(vp[(size_t) d * fan + i]) * s;
    }
    upload_f32_as(dst, w);
}

// ConvTranspose1d weight_norm + transpose to [IC, K*OC] for the mul_mat path.
static void fuse_wn_ct(ggml_tensor * dst, const GGUF & g, const std::string & pfx) {
    ggml_tensor *    mv   = gmeta(g, pfx + ".weight_v");
    const uint16_t * gp   = (const uint16_t *) gdata(g, pfx + ".weight_g");
    const uint16_t * vp   = (const uint16_t *) gdata(g, pfx + ".weight_v");
    const int        nd   = ggml_n_dims(mv);
    const int        dim0 = (int) mv->ne[nd - 1];              // IC
    const int        fan  = (int) (ggml_nelements(mv) / dim0); // K*OC
    std::vector<float> w((size_t) dim0 * fan);
    for (int d = 0; d < dim0; d++) {
        float gv = bf16_to_f32(gp[d]), nsq = 0.0f;
        for (int i = 0; i < fan; i++) { float vv = bf16_to_f32(vp[(size_t) d * fan + i]); nsq += vv * vv; }
        float s = gv / (sqrtf(nsq) + 1e-12f);
        for (int i = 0; i < fan; i++) w[(size_t) i * dim0 + d] = bf16_to_f32(vp[(size_t) d * fan + i]) * s;
    }
    upload_f32_as(dst, w);
}

// snake alpha/beta bf16 [1,C,1]; decoder/encoder use exp(a) and 1/exp(b).
static void load_snake(ggml_tensor * dst, const GGUF & g, const std::string & name, bool inv) {
    ggml_tensor *    mt  = gmeta(g, name);
    const int        C   = (int) mt->ne[1];
    const uint16_t * raw = (const uint16_t *) gdata(g, name);
    std::vector<float> d(C);
    for (int i = 0; i < C; i++) { float e = expf(bf16_to_f32(raw[i])); d[i] = inv ? 1.0f / e : e; }
    ggml_backend_tensor_set(dst, d.data(), 0, C * sizeof(float));
}

static void load_bias(ggml_tensor * dst, const GGUF & g, const std::string & name) {
    ggml_tensor *    mt  = gmeta(g, name);
    const int        C   = (int) mt->ne[0];
    const uint16_t * raw = (const uint16_t *) gdata(g, name);
    std::vector<float> d(C);
    for (int i = 0; i < C; i++) d[i] = bf16_to_f32(raw[i]);
    ggml_backend_tensor_set(dst, d.data(), 0, C * sizeof(float));
}

// ------------------------------------------------------------- graph ops
static const int UPSAMPLE = 10 * 6 * 4 * 4 * 2;  // 1920

static ggml_tensor * vae_snake(ggml_context * ctx, ggml_tensor * x, ggml_tensor * a, ggml_tensor * inv_b) {
    return ggml_snake(ctx, x, a, inv_b);
}

static ggml_tensor * vae_conv1d(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b,
                                ggml_tensor * x, int stride, int pad, int dil) {
    ggml_tensor * y = ggml_conv_1d(ctx, w, x, stride, pad, dil);
    y = ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1]);
    if (b) y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
    return y;
}

static ggml_tensor * vae_conv_t1d(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b,
                                  ggml_tensor * x, int stride, int pad, int oc) {
    ggml_tensor * xt  = ggml_cont(ctx, ggml_transpose(ctx, x));  // [IC, T_in]
    ggml_tensor * col = ggml_mul_mat(ctx, w, xt);                // [K*OC, T_in]
    ggml_tensor * y   = ggml_col2im_1d(ctx, col, stride, oc, pad);
    if (b) y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
    return y;
}

// ------------------------------------------------------------- decoder
struct ResUnit { ggml_tensor *s1a, *s1b, *c1w, *c1b, *s2a, *s2b, *c2w, *c2b; int dilation; };
struct Block   { ggml_tensor *sa, *sb, *ctw, *ctb; int in_ch, out_ch, stride, kernel; ResUnit ru[3]; };
struct VAE     { ggml_tensor *c1w, *c1b; Block blk[5]; ggml_tensor *sa, *sb, *c2w; };

static ggml_tensor * res_unit(ggml_context * ctx, ResUnit * ru, ggml_tensor * x) {
    ggml_tensor * skip = x;
    const int pad = 3 * ru->dilation;  // (k-1)*dil/2 with k=7
    x = vae_snake(ctx, x, ru->s1a, ru->s1b);
    x = vae_conv1d(ctx, ru->c1w, ru->c1b, x, 1, pad, ru->dilation);
    x = vae_snake(ctx, x, ru->s2a, ru->s2b);
    x = vae_conv1d(ctx, ru->c2w, ru->c2b, x, 1, 0, 1);
    return ggml_add(ctx, skip, x);
}

static void decoder_create(VAE & m, ggml_context * ctx) {
    static const int STRIDES[5] = { 10, 6, 4, 4, 2 };
    static const int IN_CH[5]   = { 2048, 1024, 512, 256, 128 };
    static const int OUT_CH[5]  = { 1024, 512, 256, 128, 128 };
    static const int DIL[3]     = { 1, 3, 9 };
    m.c1w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 7, 64, 2048);
    m.c1b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2048);
    for (int i = 0; i < 5; ++i) {
        Block & b = m.blk[i];
        b.in_ch = IN_CH[i]; b.out_ch = OUT_CH[i]; b.stride = STRIDES[i]; b.kernel = STRIDES[i] * 2;
        const int C = b.out_ch;
        b.sa  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, b.in_ch);
        b.sb  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, b.in_ch);
        b.ctw = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, b.in_ch, b.kernel * b.out_ch);
        b.ctb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, b.out_ch);
        for (int r = 0; r < 3; ++r) {
            ResUnit & ru = b.ru[r];
            ru.dilation = DIL[r];
            ru.s1a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
            ru.s1b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
            ru.c1w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 7, C, C);
            ru.c1b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
            ru.s2a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
            ru.s2b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
            ru.c2w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 1, C, C);
            ru.c2b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
        }
    }
    m.sa  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 128);
    m.sb  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 128);
    m.c2w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 7, 128, 2);
}

static void decoder_load(VAE & m, const GGUF & g) {
    fuse_wn(m.c1w, g, "decoder.conv1");
    load_bias(m.c1b, g, "decoder.conv1.bias");
    for (int i = 0; i < 5; ++i) {
        Block &     b  = m.blk[i];
        std::string bp = "decoder.block." + std::to_string(i);
        load_snake(b.sa, g, bp + ".snake1.alpha", false);
        load_snake(b.sb, g, bp + ".snake1.beta",  true);
        fuse_wn_ct(b.ctw, g, bp + ".conv_t1");
        load_bias(b.ctb, g, bp + ".conv_t1.bias");
        for (int r = 0; r < 3; ++r) {
            ResUnit &   ru = b.ru[r];
            std::string rp = bp + ".res_unit" + std::to_string(r + 1);
            load_snake(ru.s1a, g, rp + ".snake1.alpha", false);
            load_snake(ru.s1b, g, rp + ".snake1.beta",  true);
            fuse_wn(ru.c1w, g, rp + ".conv1");
            load_bias(ru.c1b, g, rp + ".conv1.bias");
            load_snake(ru.s2a, g, rp + ".snake2.alpha", false);
            load_snake(ru.s2b, g, rp + ".snake2.beta",  true);
            fuse_wn(ru.c2w, g, rp + ".conv2");
            load_bias(ru.c2b, g, rp + ".conv2.bias");
        }
    }
    load_snake(m.sa, g, "decoder.snake1.alpha", false);
    load_snake(m.sb, g, "decoder.snake1.beta",  true);
    fuse_wn(m.c2w, g, "decoder.conv2");
}

// latent [T_latent, 64] -> audio [T_audio, 2]
static ggml_tensor * build_decode(ggml_context * ctx, VAE * m, ggml_tensor * latent) {
    ggml_tensor * x = vae_conv1d(ctx, m->c1w, m->c1b, latent, 1, 3, 1);  // [T, 2048]
    for (int i = 0; i < 5; ++i) {
        Block & b = m->blk[i];
        x = vae_snake(ctx, x, b.sa, b.sb);
        const int pad = (b.kernel - b.stride) / 2;
        x = vae_conv_t1d(ctx, b.ctw, b.ctb, x, b.stride, pad, b.out_ch);
        for (int r = 0; r < 3; ++r) x = res_unit(ctx, &b.ru[r], x);
    }
    x = vae_snake(ctx, x, m->sa, m->sb);
    x = vae_conv1d(ctx, m->c2w, nullptr, x, 1, 3, 1);  // [T_audio, 2]
    return x;
}

// ------------------------------------------------------------- encoder
struct EncBlock { ResUnit ru[3]; ggml_tensor *sa, *sb, *dw, *db; int in_ch, out_ch, stride, kernel, padding; };
struct Encoder  { ggml_tensor *c1w, *c1b; EncBlock blk[5]; ggml_tensor *sa, *sb, *c2w, *c2b; };

static void encoder_create(Encoder & m, ggml_context * ctx) {
    static const int IN_CH[5]  = { 128, 128, 256, 512, 1024 };
    static const int OUT_CH[5] = { 128, 256, 512, 1024, 2048 };
    static const int STR[5]    = { 2, 4, 4, 6, 10 };
    static const int DIL[3]    = { 1, 3, 9 };
    m.c1w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 7, 2, 128);
    m.c1b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
    for (int i = 0; i < 5; ++i) {
        EncBlock & b = m.blk[i];
        b.in_ch = IN_CH[i]; b.out_ch = OUT_CH[i]; b.stride = STR[i];
        b.kernel = STR[i] * 2; b.padding = (STR[i] + 1) / 2;
        const int C = b.in_ch;
        for (int r = 0; r < 3; ++r) {
            ResUnit & ru = b.ru[r];
            ru.dilation = DIL[r];
            ru.s1a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
            ru.s1b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
            ru.c1w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 7, C, C);
            ru.c1b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
            ru.s2a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
            ru.s2b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
            ru.c2w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 1, C, C);
            ru.c2b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
        }
        b.sa = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
        b.sb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, C);
        b.dw = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, b.kernel, b.in_ch, b.out_ch);
        b.db = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, b.out_ch);
    }
    m.sa  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 2048);
    m.sb  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 2048);
    m.c2w = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 3, 2048, 128);
    m.c2b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
}

static void encoder_load(Encoder & m, const GGUF & g) {
    fuse_wn(m.c1w, g, "encoder.conv1");
    load_bias(m.c1b, g, "encoder.conv1.bias");
    for (int i = 0; i < 5; ++i) {
        EncBlock &  b  = m.blk[i];
        std::string bp = "encoder.block." + std::to_string(i);
        for (int r = 0; r < 3; ++r) {
            ResUnit &   ru = b.ru[r];
            std::string rp = bp + ".res_unit" + std::to_string(r + 1);
            load_snake(ru.s1a, g, rp + ".snake1.alpha", false);
            load_snake(ru.s1b, g, rp + ".snake1.beta",  true);
            fuse_wn(ru.c1w, g, rp + ".conv1");
            load_bias(ru.c1b, g, rp + ".conv1.bias");
            load_snake(ru.s2a, g, rp + ".snake2.alpha", false);
            load_snake(ru.s2b, g, rp + ".snake2.beta",  true);
            fuse_wn(ru.c2w, g, rp + ".conv2");
            load_bias(ru.c2b, g, rp + ".conv2.bias");
        }
        load_snake(b.sa, g, bp + ".snake1.alpha", false);
        load_snake(b.sb, g, bp + ".snake1.beta",  true);
        fuse_wn(b.dw, g, bp + ".conv1");
        load_bias(b.db, g, bp + ".conv1.bias");
    }
    load_snake(m.sa, g, "encoder.snake1.alpha", false);
    load_snake(m.sb, g, "encoder.snake1.beta",  true);
    fuse_wn(m.c2w, g, "encoder.conv2");
    load_bias(m.c2b, g, "encoder.conv2.bias");
}

// audio [T_audio, 2] -> [T_latent, 128] (mean = ch 0..63)
static ggml_tensor * build_encode(ggml_context * ctx, Encoder * m, ggml_tensor * audio) {
    ggml_tensor * x = vae_conv1d(ctx, m->c1w, m->c1b, audio, 1, 3, 1);  // [T, 128]
    for (int i = 0; i < 5; ++i) {
        EncBlock & b = m->blk[i];
        for (int r = 0; r < 3; ++r) x = res_unit(ctx, &b.ru[r], x);
        x = vae_snake(ctx, x, b.sa, b.sb);
        x = vae_conv1d(ctx, b.dw, b.db, x, b.stride, b.padding, 1);  // downsample
    }
    x = vae_snake(ctx, x, m->sa, m->sb);
    x = vae_conv1d(ctx, m->c2w, m->c2b, x, 1, 1, 1);  // [T_latent, 128]
    return x;
}

// ------------------------------------------------------------- WAV I/O
// Read a 16-bit PCM WAV (mono or stereo). Returns interleaved f32 [T, 2] (mono
// duplicated to stereo), sets *T and *rate. Returns empty on failure.
static std::vector<float> wav_read(const char * path, int * T, int * rate) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[WAV] cannot open %s\n", path); return {}; }
    char riff[4]; fread(riff, 1, 4, f);
    uint32_t rsz; fread(&rsz, 4, 1, f);
    char wave[4]; fread(wave, 1, 4, f);
    if (memcmp(riff, "RIFF", 4) || memcmp(wave, "WAVE", 4)) { fprintf(stderr, "[WAV] not RIFF/WAVE\n"); fclose(f); return {}; }

    uint16_t channels = 0, bits = 0; uint32_t srate = 0;
    std::vector<float> out;
    while (!feof(f)) {
        char id[4]; if (fread(id, 1, 4, f) != 4) break;
        uint32_t sz; if (fread(&sz, 4, 1, f) != 1) break;
        if (!memcmp(id, "fmt ", 4)) {
            uint16_t fmt; fread(&fmt, 2, 1, f); fread(&channels, 2, 1, f);
            fread(&srate, 4, 1, f);
            uint32_t byte_rate; fread(&byte_rate, 4, 1, f);
            uint16_t block_align; fread(&block_align, 2, 1, f);
            fread(&bits, 2, 1, f);
            if (sz > 16) fseek(f, sz - 16, SEEK_CUR);
        } else if (!memcmp(id, "data", 4)) {
            if (bits != 16) { fprintf(stderr, "[WAV] only 16-bit PCM supported (got %d)\n", bits); fclose(f); return {}; }
            int n_samp = (int) (sz / 2);            // total int16 samples (all channels)
            std::vector<int16_t> pcm(n_samp);
            fread(pcm.data(), 2, n_samp, f);
            int frames = n_samp / channels;
            out.resize((size_t) frames * 2);
            for (int t = 0; t < frames; t++) {
                float l = pcm[(size_t) t * channels + 0] / 32768.0f;
                float r = (channels >= 2) ? pcm[(size_t) t * channels + 1] / 32768.0f : l;
                out[(size_t) t * 2 + 0] = l;
                out[(size_t) t * 2 + 1] = r;
            }
            *T = frames; *rate = (int) srate;
            fclose(f);
            return out;
        } else {
            fseek(f, sz, SEEK_CUR);
        }
    }
    fclose(f);
    fprintf(stderr, "[WAV] no data chunk\n");
    return {};
}

// Write 16-bit PCM stereo WAV from planar float ch0/ch1. If normalize, peak-scale to 0.9.
static void wav_write(const char * path, const float * l, const float * r, int n, int rate, bool normalize) {
    float gain = 1.0f;
    if (normalize) {
        float peak = 1e-9f;
        for (int i = 0; i < n; i++) { peak = std::max(peak, std::abs(l[i])); peak = std::max(peak, std::abs(r[i])); }
        gain = 0.9f / peak;
    }
    FILE * f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[WAV] cannot write %s\n", path); return; }
    const int channels = 2, bits = 16;
    const uint32_t data_bytes = (uint32_t) n * channels * (bits / 8);
    auto w32 = [&](uint32_t v) { fwrite(&v, 4, 1, f); };
    auto w16 = [&](uint16_t v) { fwrite(&v, 2, 1, f); };
    fwrite("RIFF", 1, 4, f); w32(36 + data_bytes); fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); w32(16); w16(1); w16(channels);
    w32((uint32_t) rate); w32((uint32_t) rate * channels * (bits / 8)); w16(channels * (bits / 8)); w16(bits);
    fwrite("data", 1, 4, f); w32(data_bytes);
    for (int i = 0; i < n; i++) {
        auto c16 = [&](float x) -> int16_t {
            float v = x * gain * 32767.0f;
            if (v > 32767.0f) v = 32767.0f; if (v < -32768.0f) v = -32768.0f;
            return (int16_t) lrintf(v);
        };
        w16((uint16_t) c16(l[i])); w16((uint16_t) c16(r[i]));
    }
    fclose(f);
    fprintf(stderr, "[WAV] wrote %s: %d frames, %.2fs @ %d Hz stereo (gain %.3f)\n",
            path, n, (float) n / rate, rate, gain);
}
