#include "internal.h"

#include "backend_util.h"

#if defined(TTS_CPP_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#endif

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace tts_cpp {
namespace parler {
namespace detail {

#if defined(TTS_CPP_USE_ACCELERATE)

namespace {

constexpr int64_t IM2COL_TILE_FLOATS = (int64_t) 1 << 18;
constexpr int64_t MIN_TILE_ROWS      = 64;
constexpr int64_t MIN_CONVT_COLUMNS  = 32;
constexpr int64_t SNAKE_CHUNK        = 1024;

struct span {
    int64_t begin = 0;
    int64_t end   = 0;
    int64_t size() const { return end - begin; }
};

struct conv1d_geometry {
    const float * x        = nullptr;  // [L, IC]
    const float * w        = nullptr;  // [K, IC, OC] == [K*IC, OC]
    const float * bias     = nullptr;  // [OC]
    const float * residual = nullptr;  // [L, OC] or null
    float *       dst      = nullptr;  // [L, OC]
    int64_t L = 0, IC = 0, K = 0, OC = 0;
    int64_t dilation = 1, pad = 0;
};

void sgemm(bool trans_b, int64_t M, int64_t N, int64_t K,
           const float * A, int64_t lda, const float * B, int64_t ldb,
           float beta, float * C, int64_t ldc) {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
    cblas_sgemm(CblasColMajor, CblasNoTrans, trans_b ? CblasTrans : CblasNoTrans,
                (int) M, (int) N, (int) K, 1.0f, A, (int) lda, B, (int) ldb,
                beta, C, (int) ldc);
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
}

void * dilation_to_userdata(int dilation) {
    return reinterpret_cast<void *>(static_cast<intptr_t>(dilation));
}

int64_t userdata_to_dilation(void * userdata) {
    return (int64_t) reinterpret_cast<intptr_t>(userdata);
}

conv1d_geometry conv1d_geometry_of(ggml_tensor * dst, void * userdata) {
    conv1d_geometry g;
    const ggml_tensor * x = dst->src[0];
    const ggml_tensor * w = dst->src[1];
    g.x        = (const float *) x->data;
    g.w        = (const float *) w->data;
    g.bias     = (const float *) dst->src[2]->data;
    g.residual = dst->src[3] ? (const float *) dst->src[3]->data : nullptr;
    g.dst      = (float *) dst->data;
    g.L        = x->ne[0];
    g.IC       = x->ne[1];
    g.K        = w->ne[0];
    g.OC       = w->ne[2];
    g.dilation = userdata_to_dilation(userdata);
    g.pad      = (g.K - 1) / 2 * g.dilation;
    return g;
}

int64_t tile_rows_for(int64_t row_floats, int64_t L) {
    return std::min(L, std::max(MIN_TILE_ROWS, IM2COL_TILE_FLOATS / row_floats));
}

int64_t conv1d_tile_rows(const conv1d_geometry & g) {
    return tile_rows_for(g.K == 1 ? g.IC : g.K * g.IC, g.L);
}

int64_t tile_floats_of(const ggml_tensor * w) {
    const int64_t row_floats = w->ne[0] * w->ne[1];
    return w->ne[0] == 1 ? 0 : row_floats * tile_rows_for(row_floats, INT64_MAX);
}

int64_t max_res_tile_floats(const parler_model & model) {
    int64_t m = 0;
    for (const parler_dac_block & b : model.dac_blocks) {
        for (const parler_dac_residual & r : b.res) m = std::max(m, tile_floats_of(r.conv1_w));
    }
    return m;
}

int64_t ceil_div(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

span nth_span(int64_t n, int64_t tile, int64_t index) {
    span s;
    s.begin = index * tile;
    s.end   = std::min(n, s.begin + tile);
    return s;
}

void fill_channel_rows(float * out, span rows, float bias, const float * residual) {
    for (int64_t t = rows.begin; t < rows.end; ++t) {
        out[t] = residual ? residual[t] + bias : bias;
    }
}

void init_conv1d_rows(const conv1d_geometry & g, span rows) {
    for (int64_t oc = 0; oc < g.OC; ++oc) {
        fill_channel_rows(g.dst + oc * g.L, rows, g.bias[oc],
                          g.residual ? g.residual + oc * g.L : nullptr);
    }
}

void gather_shifted(const float * src, int64_t len, int64_t start, int64_t n, float * out) {
    const int64_t lo = std::clamp<int64_t>(-start, 0, n);
    const int64_t hi = std::clamp<int64_t>(len - start, lo, n);
    std::memset(out, 0, (size_t) lo * sizeof(float));
    if (hi > lo) std::memcpy(out + lo, src + start + lo, (size_t) (hi - lo) * sizeof(float));
    std::memset(out + hi, 0, (size_t) (n - hi) * sizeof(float));
}

void gather_channel_taps(const conv1d_geometry & g, int64_t ic, span rows, float * tile) {
    const float * src = g.x + ic * g.L;
    for (int64_t k = 0; k < g.K; ++k) {
        gather_shifted(src, g.L, rows.begin + k * g.dilation - g.pad, rows.size(),
                       tile + (ic * g.K + k) * rows.size());
    }
}

void build_im2col_tile(const conv1d_geometry & g, span rows, float * tile) {
    for (int64_t ic = 0; ic < g.IC; ++ic) {
        gather_channel_taps(g, ic, rows, tile);
    }
}

std::vector<float> & im2col_scratch() {
    thread_local std::vector<float> scratch;
    return scratch;
}

void run_conv1d_tile(const conv1d_geometry & g, span rows) {
    init_conv1d_rows(g, rows);
    if (g.K == 1) {
        sgemm(false, rows.size(), g.OC, g.IC, g.x + rows.begin, g.L, g.w, g.IC,
              1.0f, g.dst + rows.begin, g.L);
        return;
    }
    std::vector<float> & tile = im2col_scratch();
    tile.resize((size_t) (rows.size() * g.K * g.IC));
    build_im2col_tile(g, rows, tile.data());
    sgemm(false, rows.size(), g.OC, g.K * g.IC, tile.data(), rows.size(), g.w, g.K * g.IC,
          1.0f, g.dst + rows.begin, g.L);
}

void conv1d_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const conv1d_geometry g = conv1d_geometry_of(dst, userdata);
    const int64_t rows    = conv1d_tile_rows(g);
    const int64_t n_tiles = ceil_div(g.L, rows);
    for (int64_t tile = ith; tile < n_tiles; tile += nth) {
        run_conv1d_tile(g, nth_span(g.L, rows, tile));
    }
}

void convt_columns_op(ggml_tensor * dst, int ith, int nth, void * /*userdata*/) {
    const ggml_tensor * x = dst->src[0];  // [IL, IC]
    const ggml_tensor * w = dst->src[1];  // [K, OC, IC] == [K*OC, IC]
    const int64_t IL = x->ne[0];
    const int64_t IC = x->ne[1];
    const int64_t KOC = w->ne[0] * w->ne[1];
    const int64_t cols = std::max(MIN_CONVT_COLUMNS, ceil_div(IL, nth));
    const span s = nth_span(IL, cols, ith);
    if (s.size() <= 0) return;
    sgemm(true, KOC, s.size(), IC, (const float *) w->data, KOC,
          (const float *) x->data + s.begin, IL, 0.0f,
          (float *) dst->data + s.begin * KOC, KOC);
}

void snake_channel(const float * x, float * y, int64_t n, float alpha, float inv) {
    float buf[SNAKE_CHUNK];
    for (int64_t t0 = 0; t0 < n; t0 += SNAKE_CHUNK) {
        const int len = (int) std::min(SNAKE_CHUNK, n - t0);
        vDSP_vsmul(x + t0, 1, &alpha, buf, 1, len);
        vvsinf(buf, buf, &len);
        vDSP_vsq(buf, 1, buf, 1, len);
        vDSP_vsma(buf, 1, &inv, x + t0, 1, y + t0, 1, len);
    }
}

void snake_op(ggml_tensor * dst, int ith, int nth, void * /*userdata*/) {
    const ggml_tensor * x = dst->src[0];
    const float * alpha = (const float *) dst->src[1]->data;
    const float * inv   = (const float *) dst->src[2]->data;
    const int64_t T = x->ne[0];
    const int64_t C = x->ne[1];
    for (int64_t c = ith; c < C; c += nth) {
        snake_channel((const float *) x->data + c * T, (float *) dst->data + c * T, T, alpha[c], inv[c]);
    }
}

} // namespace

bool parler_dac_accel_compiled() {
    return true;
}

ggml_tensor * parler_dac_accel_snake(ggml_context * ctx, ggml_tensor * x, ggml_tensor * alpha,
                                     ggml_tensor * inv) {
    GGML_ASSERT(ggml_is_contiguous(x) && ggml_is_contiguous(alpha) && ggml_is_contiguous(inv));
    GGML_ASSERT(x->ne[2] == 1 && x->ne[3] == 1);
    GGML_ASSERT(ggml_nelements(alpha) == x->ne[1] && ggml_nelements(inv) == x->ne[1]);
    ggml_tensor * args[] = { x, alpha, inv };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, x->ne[0], x->ne[1], 1, 1,
                          args, 3, snake_op, GGML_N_TASKS_MAX, nullptr);
}

ggml_tensor * parler_dac_accel_conv1d(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w,
                                      ggml_tensor * bias, int dilation, ggml_tensor * residual) {
    GGML_ASSERT(ggml_is_contiguous(x) && ggml_is_contiguous(w) && ggml_is_contiguous(bias));
    GGML_ASSERT(!residual || ggml_is_contiguous(residual));
    GGML_ASSERT(ggml_nelements(bias) == w->ne[2]);
    ggml_tensor * args[] = { x, w, bias, residual };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, x->ne[0], w->ne[2], 1, 1,
                          args, residual ? 4 : 3, conv1d_op, GGML_N_TASKS_MAX,
                          dilation_to_userdata(dilation));
}

ggml_tensor * parler_dac_accel_convt_columns(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w) {
    GGML_ASSERT(ggml_is_contiguous(x) && ggml_is_contiguous(w));
    ggml_tensor * args[] = { x, w };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, w->ne[0] * w->ne[1], x->ne[0], 1, 1,
                          args, 2, convt_columns_op, GGML_N_TASKS_MAX, nullptr);
}

#else

bool parler_dac_accel_compiled() {
    return false;
}

ggml_tensor * parler_dac_accel_snake(ggml_context *, ggml_tensor *, ggml_tensor *, ggml_tensor *) {
    return nullptr;
}

ggml_tensor * parler_dac_accel_conv1d(ggml_context *, ggml_tensor *, ggml_tensor *,
                                      ggml_tensor *, int, ggml_tensor *) {
    return nullptr;
}

ggml_tensor * parler_dac_accel_convt_columns(ggml_context *, ggml_tensor *, ggml_tensor *) {
    return nullptr;
}

#endif

namespace {

bool accel_requested() {
    return parler_dac_accel_compiled() && std::getenv("PARLER_DAC_NO_ACCEL") == nullptr;
}

} // namespace

bool parler_dac_accel_enabled(const parler_model & model) {
    return !model.on_gpu && accel_requested();
}

size_t parler_dac_accel_scratch_bytes(const parler_model & model, int n_threads) {
#if defined(TTS_CPP_USE_ACCELERATE)
    if (!parler_dac_accel_enabled(model) || !model.dac_conv_in_w || !model.dac_conv_out_w) return 0;
    const int64_t per_thread = std::max({ tile_floats_of(model.dac_conv_in_w),
                                          tile_floats_of(model.dac_conv_out_w),
                                          max_res_tile_floats(model) });
    return (size_t) per_thread * sizeof(float) * (size_t) std::max(1, n_threads);
#else
    (void) model;
    (void) n_threads;
    return 0;
#endif
}

bool parler_dac_cpu_is_shape_exact() {
    return !accel_requested() && ::tts_cpp::detail::cpu_matmul_is_shape_exact();
}

} // namespace detail
} // namespace parler
} // namespace tts_cpp
