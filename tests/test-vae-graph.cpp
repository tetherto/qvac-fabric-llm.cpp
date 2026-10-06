// Smoke test: build the ACE-Step Oobleck VAE decoder graph on CPU and run it
// end-to-end with synthetic weights.
//
// Purpose: prove our two custom ops (ggml_col2im_1d + ggml_snake) carry the real
// VAE decode topology on the CPU backend -- graph builds, allocates and computes
// to a finite [T_audio, 2] output with the expected 1920x upsample.  This is a
// topology / integration check, NOT numerical parity (that needs real weights).
//
// Mirrors src/vae.h from acestep.cpp:
//   conv1(64->2048,k7) -> 5x[snake -> convT(upsample) -> 3x resunit] -> snake -> conv2(128->2,k7)
//   resunit: skip + conv1(k1)(snake2(conv1(k7,dil)(snake1(x))))
//   convT via GEMM + col2im_1d;  snake via the fused ggml_snake op.

#include <ggml.h>
#include <ggml-cpu.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static const int STRIDES[5] = { 10, 6, 4, 4, 2 };
static const int IN_CH[5]   = { 2048, 1024, 512, 256, 128 };
static const int OUT_CH[5]  = { 1024, 512, 256, 128, 128 };
static const int DILATION[3] = { 1, 3, 9 };
static const int UPSAMPLE   = 10 * 6 * 4 * 4 * 2; // 1920

struct ResUnit { ggml_tensor *s1a, *s1b, *c1w, *c1b, *s2a, *s2b, *c2w, *c2b; int dilation; };
struct Block   { ggml_tensor *sa, *sb, *ctw, *ctb; int in_ch, out_ch, stride, kernel; ResUnit ru[3]; };
struct VAE     { ggml_tensor *c1w, *c1b; Block blk[5]; ggml_tensor *sa, *sb, *c2w; };

static std::mt19937 g_rng(1234);

static void upload(ggml_tensor * t, std::vector<float> & v) {
    const int64_t n = ggml_nelements(t);
    if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> h(n);
        ggml_fp32_to_fp16_row(v.data(), h.data(), (int) n);
        ggml_backend_tensor_set(t, h.data(), 0, n * sizeof(ggml_fp16_t));
    } else {
        ggml_backend_tensor_set(t, v.data(), 0, n * sizeof(float));
    }
}

// Conv weights: Xavier-ish N(0, 1/sqrt(fan_in)) so activations stay O(1) through
// the deep 1920x cascade -- bounded but non-trivial (no total cancellation).
//   conv1d kernel [K, IC, OC] -> fan = K*IC   |   conv_t [IC, K*OC] -> fan = IC
static void fill_conv(ggml_tensor * t) {
    const int64_t n   = ggml_nelements(t);
    const int64_t fan = (ggml_n_dims(t) == 3) ? (t->ne[0] * t->ne[1]) : t->ne[0];
    std::normal_distribution<float> nd(0.0f, 1.0f / std::sqrt((float) fan));
    std::vector<float> v(n);
    for (int64_t i = 0; i < n; ++i) v[i] = nd(g_rng);
    upload(t, v);
}

// Snake exp(a)/inv(exp(b)) params: close to 1 (their bf16 alpha/beta ~ 0).
static void fill_snake_param(ggml_tensor * t) {
    const int64_t n = ggml_nelements(t);
    std::normal_distribution<float> nd(1.0f, 0.05f);
    std::vector<float> v(n);
    for (int64_t i = 0; i < n; ++i) v[i] = nd(g_rng);
    upload(t, v);
}

static void fill_zero(ggml_tensor * t) {
    std::vector<float> v(ggml_nelements(t), 0.0f);
    upload(t, v);
}

static ggml_tensor * snake(ggml_context * ctx, ggml_tensor * x, ggml_tensor * a, ggml_tensor * inv_b) {
    return ggml_snake(ctx, x, a, inv_b); // fused op (replaces the naive mul->sin->sqr->mul->add chain)
}

// Conv1d + optional bias: x [T, IC] -> [T_out, OC]
static ggml_tensor * conv1d(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b,
                            ggml_tensor * x, int stride, int pad, int dil) {
    ggml_tensor * y = ggml_conv_1d(ctx, w, x, stride, pad, dil);
    y = ggml_reshape_2d(ctx, y, y->ne[0], y->ne[1]);
    if (b) {
        y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
    }
    return y;
}

// ConvTranspose1d via GEMM + col2im_1d: x [T_in, IC] -> [T_out, OC]
static ggml_tensor * conv_t1d(ggml_context * ctx, ggml_tensor * w, ggml_tensor * b,
                              ggml_tensor * x, int stride, int pad, int oc) {
    ggml_tensor * xt  = ggml_cont(ctx, ggml_transpose(ctx, x)); // [IC, T_in]
    ggml_tensor * col = ggml_mul_mat(ctx, w, xt);               // [K*OC, T_in]
    ggml_tensor * y   = ggml_col2im_1d(ctx, col, stride, oc, pad);
    if (b) {
        y = ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
    }
    return y;
}

static ggml_tensor * res_unit(ggml_context * ctx, ResUnit * ru, ggml_tensor * x) {
    ggml_tensor * skip = x;
    const int pad = 3 * ru->dilation; // (k-1)*dil/2 with k=7
    x = snake(ctx, x, ru->s1a, ru->s1b);
    x = conv1d(ctx, ru->c1w, ru->c1b, x, 1, pad, ru->dilation);
    x = snake(ctx, x, ru->s2a, ru->s2b);
    x = conv1d(ctx, ru->c2w, ru->c2b, x, 1, 0, 1);
    return ggml_add(ctx, skip, x);
}

static ggml_tensor * build_decode(ggml_context * ctx, VAE * m, ggml_tensor * latent) {
    ggml_tensor * x = conv1d(ctx, m->c1w, m->c1b, latent, 1, 3, 1); // [T, 2048]
    for (int i = 0; i < 5; ++i) {
        Block & b = m->blk[i];
        x = snake(ctx, x, b.sa, b.sb);
        const int pad = (b.kernel - b.stride) / 2;
        x = conv_t1d(ctx, b.ctw, b.ctb, x, b.stride, pad, b.out_ch);
        for (int r = 0; r < 3; ++r) {
            x = res_unit(ctx, &b.ru[r], x);
        }
    }
    x = snake(ctx, x, m->sa, m->sb);
    x = conv1d(ctx, m->c2w, nullptr, x, 1, 3, 1); // [T_audio, 2]
    return x;
}

int main() {
    const int T_latent = 4; // tiny window; T_audio = T_latent * 1920

    ggml_init_params wp {
        /*.mem_size   =*/ ggml_tensor_overhead() * 2048 + ggml_graph_overhead_custom(8192, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr wctx_ptr{ggml_init(wp)};
    ggml_context * wctx = wctx_ptr.get();

    VAE m = {};
    m.c1w = ggml_new_tensor_3d(wctx, GGML_TYPE_F16, 7, 64, 2048);
    m.c1b = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, 2048);
    for (int i = 0; i < 5; ++i) {
        Block & b = m.blk[i];
        b.in_ch = IN_CH[i]; b.out_ch = OUT_CH[i]; b.stride = STRIDES[i]; b.kernel = STRIDES[i] * 2;
        const int C = b.out_ch;
        b.sa  = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, 1, b.in_ch);
        b.sb  = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, 1, b.in_ch);
        b.ctw = ggml_new_tensor_2d(wctx, GGML_TYPE_F16, b.in_ch, b.kernel * b.out_ch);
        b.ctb = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, b.out_ch);
        for (int r = 0; r < 3; ++r) {
            ResUnit & ru = b.ru[r];
            ru.dilation = DILATION[r];
            ru.s1a = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, 1, C);
            ru.s1b = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, 1, C);
            ru.c1w = ggml_new_tensor_3d(wctx, GGML_TYPE_F16, 7, C, C);
            ru.c1b = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, C);
            ru.s2a = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, 1, C);
            ru.s2b = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, 1, C);
            ru.c2w = ggml_new_tensor_3d(wctx, GGML_TYPE_F16, 1, C, C);
            ru.c2b = ggml_new_tensor_1d(wctx, GGML_TYPE_F32, C);
        }
    }
    m.sa  = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, 1, 128);
    m.sb  = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, 1, 128);
    m.c2w = ggml_new_tensor_3d(wctx, GGML_TYPE_F16, 7, 128, 2);

    ggml_tensor * latent = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, T_latent, 64);
    ggml_set_input(latent);

    // Build the graph in the same context so one alloc covers weights + intermediates.
    ggml_tensor * out = build_decode(wctx, &m, latent);
    ggml_set_output(out);

    ggml_backend_ptr backend_ptr{ggml_backend_cpu_init()};
    ggml_backend_t backend = backend_ptr.get();
    ggml_backend_cpu_set_n_threads(backend, 4);

    ggml_cgraph * gf = ggml_new_graph_custom(wctx, 8192, false);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_ptr buffer{ggml_backend_alloc_ctx_tensors(wctx, backend)};
    if (!buffer) {
        printf("FAILED: could not allocate tensors\n");
        return 1;
    }

    // Synthetic weights (Xavier-ish -> bounded but non-trivial through 1920x).
    fill_conv(m.c1w); fill_zero(m.c1b);
    for (int i = 0; i < 5; ++i) {
        Block & b = m.blk[i];
        fill_snake_param(b.sa); fill_snake_param(b.sb);
        fill_conv(b.ctw); fill_zero(b.ctb);
        for (int r = 0; r < 3; ++r) {
            ResUnit & ru = b.ru[r];
            fill_snake_param(ru.s1a); fill_snake_param(ru.s1b);
            fill_conv(ru.c1w); fill_zero(ru.c1b);
            fill_snake_param(ru.s2a); fill_snake_param(ru.s2b);
            fill_conv(ru.c2w); fill_zero(ru.c2b);
        }
    }
    fill_snake_param(m.sa); fill_snake_param(m.sb);
    fill_conv(m.c2w);

    std::vector<float> lat(ggml_nelements(latent));
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (size_t i = 0; i < lat.size(); ++i) lat[i] = nd(g_rng);
    ggml_backend_tensor_set(latent, lat.data(), 0, ggml_nbytes(latent));

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        printf("FAILED: graph compute error\n");
        return 1;
    }

    const int64_t T_audio = out->ne[0];
    const int64_t OC      = out->ne[1];
    std::vector<float> audio(ggml_nelements(out));
    ggml_backend_tensor_get(out, audio.data(), 0, ggml_nbytes(out));

    bool finite = true;
    float amax = 0.0f;
    for (float v : audio) {
        if (!std::isfinite(v)) { finite = false; break; }
        amax = std::max(amax, std::abs(v));
    }

    const bool shape_ok = (T_audio == (int64_t) T_latent * UPSAMPLE) && (OC == 2);
    const bool passed   = shape_ok && finite;

    printf("VAE decode graph: nodes=%d  in=[%d,64] -> out=[%d,%d] (expected [%d,2])  |max|=%.4f  finite=%s\n",
        ggml_graph_n_nodes(gf), T_latent, (int) T_audio, (int) OC, T_latent * UPSAMPLE, amax,
        finite ? "yes" : "no");
    printf("%s\n", passed ? "\033[32mPASSED\033[0m" : "\033[31mFAILED\033[0m");
    return passed ? 0 : 1;
}
