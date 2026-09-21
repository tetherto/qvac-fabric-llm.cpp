#include "xdna-ops.h"
#include "xdna-runtime.h"
#include "xdna-util.h"
#include "xdna-quant.h"
#include "xdna-gemv.h"
#include "xdna-gdn-prefill.h"
#include "xdna-conv-prefill.h"
#include "xdna-fa-prefill.h"

#include "ggml-impl.h"
#include "ggml-quants.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// --- GEMM internals (static) ------------------------------------------------

// K is split into blocks of at most GEMM_K_MAX: a per-core K-loop count above
// 64 (K > 1024 with tile_k=16) wedges the shared hw_context. GEMM_N_MAX is a
// practical cap on N: wide projections (e.g. the vocabulary output) need a
// huge B transpose and run too long on the NPU.
static const int GEMM_K_MAX = 1024;
static const int GEMM_N_MAX = 16384;
// Rows at or above this use the prefill geometry; below it the decode M32 one.
static const int GEMM_M_BIG_MIN = 64;

// --- int8 activation packing (SIMD) -----------------------------------------
//
// The int8 prefill route quantizes the f32 activation rows twice per op: once
// to scan each row's scale (d_scan) and once to emit the int8 codes (a_pack).
// Both are straight-line f32 work over M x K, so they vectorize well. On x86
// the scans use AVX2 and the quantizer AVX512 (Zen4/5); every other target
// keeps the scalar fallback, which is bit-identical apart from the fused
// multiply (x * (1/d) instead of x / d).

#if defined(__x86_64__)
#include <immintrin.h>

static bool cpu_has_avx2(void) {
    static const bool v = __builtin_cpu_supports("avx2");
    return v;
}
static bool cpu_has_avx512(void) {
    static const bool v = __builtin_cpu_supports("avx512f");
    return v;
}

// Max |x| over one row (AVX2 path).
__attribute__((target("avx2")))
static float row_amax_avx2(const float * __restrict row, int n) {
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    __m256 vmax = _mm256_setzero_ps();
    int k = 0;
    for (; k + 8 <= n; k += 8) {
        vmax = _mm256_max_ps(vmax, _mm256_and_ps(_mm256_loadu_ps(row + k), absmask));
    }
    __m128 lo = _mm256_castps256_ps128(vmax);
    __m128 hi = _mm256_extractf128_ps(vmax, 1);
    __m128 m  = _mm_max_ps(lo, hi);
    m = _mm_max_ps(m, _mm_shuffle_ps(m, m, 0x4E));   // swap 64-bit halves
    m = _mm_max_ps(m, _mm_shuffle_ps(m, m, 0xB1));   // swap 32-bit halves
    float amax = _mm_cvtss_f32(m);
    for (; k < n; k++) {
        const float a = fabsf(row[k]);
        amax = a > amax ? a : amax;
    }
    return amax;
}

// Quantize one row to saturating int8: dst[k] = round(x[k] * recip), clamped.
// AVX512 path converts 16 f32 at a time and packs straight to int8.
__attribute__((target("avx512f")))
static void quantize_row_avx512(int8_t * __restrict dst, const float * __restrict row,
                                int n, float recip) {
    const __m512 rv = _mm512_set1_ps(recip);
    int k = 0;
    for (; k + 16 <= n; k += 16) {
        const __m512 x = _mm512_mul_ps(_mm512_loadu_ps(row + k), rv);
        const __m512i i32 = _mm512_cvtps_epi32(x);       // rounds per MXCSR (nearest)
        const __m128i i8  = _mm512_cvtsepi32_epi8(i32);  // saturates to int8
        _mm_storeu_si128((__m128i *) (dst + k), i8);
    }
    for (; k < n; k++) {
        const int q = (int) lrintf(row[k] * recip);
        dst[k] = (int8_t) std::max(-128, std::min(127, q));
    }
}
#endif   // __x86_64__

static float xdna_row_amax(const float * row, int n) {
#if defined(__x86_64__)
    if (cpu_has_avx2()) {
        return row_amax_avx2(row, n);
    }
#endif
    float amax = 0.0f;
    for (int k = 0; k < n; k++) {
        const float a = fabsf(row[k]);
        amax = a > amax ? a : amax;
    }
    return amax;
}

static void xdna_quantize_row(int8_t * dst, const float * row, int n, float recip) {
#if defined(__x86_64__)
    if (cpu_has_avx512()) {
        quantize_row_avx512(dst, row, n, recip);
        return;
    }
#endif
    for (int k = 0; k < n; k++) {
        const int q = (int) lrintf(row[k] * recip);
        dst[k] = (int8_t) std::max(-128, std::min(127, q));
    }
}

// RTP base for a per-core M tile. The address is fixed by the compiled L1
// layout, so each tile size the build bakes needs its own constant, and an
// unknown tile writes every RTP somewhere wrong. 0 says "no such tile": the
// geometry is then refused instead of driven with the wrong address.
static int rtp_base_for_tile_m(int tile_m) {
    switch (tile_m) {
        case 8:  return XDNA_RTP_BASE_TILE_M8;
        case 16: return XDNA_RTP_BASE_TILE_M16;
        case 32: return XDNA_RTP_BASE_TILE_M32;
        case 64: return XDNA_RTP_BASE_TILE_M64;
        default: return 0;
    }
}

// Prefill tile layout for a baked M block. bf16 runs the bfp16 r=8 mmul with
// the narrower tile its double-width banks allow; int8 keeps 1-byte elements
// and doubles the per-core M tile, which halves the number of times the
// weights cross DDR per op. Only the baked M block differs between kernels.
static xdna_gemm_tiles prefill_tiles_for(int Mk, int elem_bytes) {
    xdna_gemm_tiles t;   // rtp_base 0 on an unbaked tile: xdna_pick_geom refuses it
    t.M              = Mk;
    t.tile_m         = elem_bytes == 1 ? GGML_XDNA_TILE_M_BIG_I8 : GGML_XDNA_TILE_M_BIG;
    t.tile_k         = GGML_XDNA_TILE_K;
    t.tile_n         = GGML_XDNA_TILE_N;
    t.n_cols         = GGML_XDNA_N_COLS;
    t.n_compute_rows = GGML_XDNA_N_COMPUTE_ROWS;
    t.rtp_base       = rtp_base_for_tile_m(t.tile_m);
    t.elem_bytes     = elem_bytes;
    return t;
}

// A resolved GEMM geometry: the tiles (decode M32 or a chosen prefill block)
// plus the xclbin stem to load the kernel from.
struct xdna_gemm_geom {
    xdna_gemm_tiles tiles;
    const char *    xclbin = nullptr;
};

// Pick the prefill M block for an op: the available block that minimizes the
// byte cost of re-streaming B per submission vs. padding the batch to whole
// blocks (A fill + accumulator work on the padded rows). Returns 0 when
// nothing fits.
static int pick_prefill_block(const xdna_ops * ops, int M, int K, int N, int eb,
                              const std::unordered_map<int, std::string> & route) {
    int64_t best_b = 0, best_cost = 0;
    for (int Mk : ops->pref_blocks) {           // descending
        if (route.find(Mk) == route.end()) {
            continue;
        }
        const int64_t nblk = ((int64_t) M + Mk - 1) / Mk;
        const int64_t pad  = nblk * Mk - M;
        const int64_t cost = nblk * (int64_t) K * N * eb + pad * ((int64_t) K * eb + (int64_t) N * 4);
        if (best_b == 0 || cost < best_cost) {
            best_b = Mk;
            best_cost = cost;
        }
    }
    return (int) best_b;
}

// Pick the geometry for an op: the decode M32 geometry below GEMM_M_BIG_MIN,
// otherwise the cheapest prefill M block <= M that has an artifact for the
// chosen route (bf16 or native int8). Falls back to the decode geometry when
// no prefill block is available.
static xdna_gemm_geom xdna_pick_geom(const xdna_ops * ops, int M, int K, int N, bool want_i8) {
    xdna_gemm_geom g;
    g.tiles  = ops->gemm_tiles;
    g.xclbin = ops->gemm_xclbin_decode.c_str();
    if (M < GEMM_M_BIG_MIN) {
        return g;
    }
    const std::unordered_map<int, std::string> & route = want_i8 ? ops->i8_xclbin
                                                                 : ops->pref_xclbin;
    const int eb = want_i8 ? 1 : 2;
    const int Mk = pick_prefill_block(ops, M, K, N, eb, route);
    if (Mk > 0) {
        const xdna_gemm_tiles t = prefill_tiles_for(Mk, eb);
        if (t.rtp_base > 0) {
            g.tiles  = t;
            g.xclbin = route.at(Mk).c_str();
        }
    }
    return g;
}

// Return the device BO holding the packed [K_pad x N] bf16 weight for `src0`,
// built and cached on first use. K_pad rounds K up to the GEMM block, so every
// K-block reads a valid slice. Weights are immutable, so the transpose + bf16
// conversion runs once.
// ggml_fp32_to_bf16_row is scalar unless the translation unit is built with
// AVX512-BF16, and this backend is -O3 only. The weight pack calls it once per
// element, so at 750M weights that is 750M calls; the integer form below is the
// same round-to-nearest-even on the top 16 bits and the compiler vectorises it.
static inline uint16_t f32_to_bf16_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    return (uint16_t) ((x + 0x7fffu + ((x >> 16) & 1u)) >> 16);
}

static void f32_to_bf16_run(const float * src, ggml_bf16_t * dst, int64_t n) {
    uint16_t * d = (uint16_t *) dst;
    for (int64_t i = 0; i < n; i++) {
        d[i] = f32_to_bf16_bits(src[i]);
    }
}

static xdna_buffer * gemm_weight_bo(xdna_ops * ops, const struct ggml_tensor * src0, int K, int N) {
    const xdna_ops::weight_key key = { src0->data, K, N };
    {
        std::lock_guard<std::mutex> lock(ops->weight_mutex);
        auto it = ops->weight_bo.find(key);
        if (it != ops->weight_bo.end()) {
            return it->second;
        }
    }

    const int K_pad = (K + GEMM_K_MAX - 1) / GEMM_K_MAX * GEMM_K_MAX;
    xdna_buffer * bo_w = xdna_buffer_alloc(ops->pool->device, (size_t) K_pad * N * sizeof(ggml_bf16_t));
    if (!bo_w) {
        GGML_LOG_ERROR("%s: failed to allocate weight BO\n", "xdna-ops");
        return nullptr;
    }

    ggml_bf16_t * w = (ggml_bf16_t *) bo_w->bo.map();
    std::memset(w, 0, (size_t) K_pad * N * sizeof(ggml_bf16_t));
    if (src0->type == GGML_TYPE_BF16) {
        const auto * d = (const ggml_bf16_t *) src0->data;
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k++) {
                w[(size_t) k * N + n] = d[(size_t) n * K + k];
            }
        }
    } else if (src0->type == GGML_TYPE_F16) {
        std::vector<float> row(K);
        for (int n = 0; n < N; n++) {
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) src0->data + (size_t) n * K, row.data(), K);
            for (int k = 0; k < K; k++) {
                w[(size_t) k * N + n].bits = f32_to_bf16_bits(row[k]);
            }
        }
    } else if (src0->type == GGML_TYPE_Q4_K) {
        // Quantized weights are dequantized once at pack time and stored as
        // bf16, so the NPU kernel stays unchanged.
        std::vector<float> row(K);
        const int nb = K / QK_K;
        for (int n = 0; n < N; n++) {
            dequantize_row_q4_K((const block_q4_K *) src0->data + (size_t) n * nb, row.data(), K);
            for (int k = 0; k < K; k++) {
                w[(size_t) k * N + n].bits = f32_to_bf16_bits(row[k]);
            }
        }
    }
    xdna_buffer_sync_to_device(bo_w);

    std::lock_guard<std::mutex> lock(ops->weight_mutex);
    ops->weight_bo.emplace(key, bo_w);
    return bo_w;
}

// --- int8 GEMM (native int8 x int8 -> int32, quantized prefill weights) -----

// True when `node` routes to the native int8 GEMM: a quantized (Q4_K/Q5_K/Q6_K)
// weight used at prefill batch size with the int8 artifact present.
static bool xdna_int8_eligible(const xdna_ops * ops, const struct ggml_tensor * node) {
    if (ops->i8_xclbin.empty()) {
        return false;
    }
    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1];
    if (!src0 || !src1) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_K &&
        src0->type != GGML_TYPE_Q6_K) {
        return false;
    }
    if (src1->ne[1] < (int64_t) GEMM_M_BIG_MIN) {
        return false;
    }
    // The chosen block has to exist. Without this the picker would fall back
    // to the decode geometry and run the int8 stream through the bf16 kernel.
    if (pick_prefill_block(ops, (int) src1->ne[1], (int) src1->ne[0],
                           (int) src0->ne[1], 1, ops->i8_xclbin) <= 0) {
        return false;
    }
    return true;
}

// Pack the [K_pad x N] int8 codes + per-column scales for a quantized tensor,
// dequantizing each column once and re-quantizing symmetrically to int8.
static void gemm_int8_pack_col(const struct ggml_tensor * src0, int n, int K, float * row) {
    const size_t stride = (size_t) (K / QK_K);
    const uint8_t * base = (const uint8_t *) src0->data;
    switch (src0->type) {
        case GGML_TYPE_Q4_K:
            dequantize_row_q4_K((const block_q4_K *) (base + n * stride * sizeof(block_q4_K)), row, K);
            break;
        case GGML_TYPE_Q5_K:
            dequantize_row_q5_K((const block_q5_K *) (base + n * stride * sizeof(block_q5_K)), row, K);
            break;
        case GGML_TYPE_Q6_K:
            dequantize_row_q6_K((const block_q6_K *) (base + n * stride * sizeof(block_q6_K)), row, K);
            break;
        default:
            break;
    }
}

static xdna_ops::int8_wbo * gemm_int8_weight_bo(xdna_ops * ops, const struct ggml_tensor * src0, int K, int N) {
    const xdna_ops::weight_key key = { src0->data, K, N };
    {
        std::lock_guard<std::mutex> lock(ops->int8_mutex);
        auto it = ops->int8_wbo_map.find(key);
        if (it != ops->int8_wbo_map.end()) {
            return it->second;
        }
    }

    const int K_pad = (K + GEMM_K_MAX - 1) / GEMM_K_MAX * GEMM_K_MAX;
    xdna_ops::int8_wbo * wb = new xdna_ops::int8_wbo;
    wb->bo = xdna_buffer_alloc(ops->pool->device, (size_t) K_pad * N);
    wb->K = K;
    wb->N = N;
    wb->dw.assign(N, 1.0f);
    if (!wb->bo) {
        GGML_LOG_ERROR("%s: failed to allocate int8 weight BO\n", "xdna-ops");
        delete wb;
        return nullptr;
    }

    int8_t * w = (int8_t *) wb->bo->bo.map();
    std::memset(w, 0, (size_t) K_pad * N);
    // Dequantising the tensor and re-quantising it to int8 is once per weight,
    // but it measured 1638 ms of a 6.8 s prompt - more than the array spent on
    // the GEMMs themselves.
    //
    // Two things made it slow. The columns are independent, so this is a
    // parallel loop; and the destination is transposed, w[k*N + n], so writing
    // a column wrote K single bytes 8 KB apart - a cache miss each. Doing a
    // strip of columns at a time turns those into contiguous runs of NSTRIP
    // bytes, at the price of a strip's worth of dequantised floats in L2.
    constexpr int NSTRIP = 64;
#ifdef GGML_USE_OPENMP
#   pragma omp parallel
#endif
    {
        std::vector<float> cols((size_t) NSTRIP * K);
#ifdef GGML_USE_OPENMP
#       pragma omp for schedule(static)
#endif
        for (int n0 = 0; n0 < N; n0 += NSTRIP) {
            const int nb = std::min(NSTRIP, N - n0);
            for (int j = 0; j < nb; j++) {
                float * col = cols.data() + (size_t) j * K;
                gemm_int8_pack_col(src0, n0 + j, K, col);
                float amax = 0.0f;
                for (int k = 0; k < K; k++) {
                    amax = std::max(amax, std::fabs(col[k]));
                }
                wb->dw[n0 + j] = amax > 0.0f ? amax / 127.0f : 1.0f;
            }
            for (int k = 0; k < K; k++) {
                int8_t * wrow = w + (size_t) k * N + n0;
                for (int j = 0; j < nb; j++) {
                    const int q = (int) lrintf(cols[(size_t) j * K + k] /
                                               wb->dw[n0 + j]);
                    wrow[j] = (int8_t) std::max(-128, std::min(127, q));
                }
            }
        }
    }
    xdna_buffer_sync_to_device(wb->bo);

    std::lock_guard<std::mutex> lock(ops->int8_mutex);
    ops->int8_wbo_map.emplace(key, wb);
    return wb;
}

// Native int8 M64 GEMM for one quantized prefill MUL_MAT. Runs synchronously
// (each op completes before the next submission), mirroring gemm_compute's
// ping-pong but reading back raw int32 C: the per-M-block accumulator is
// rescaled once by d_a[m] * dw[n] and written to dst as f32.
static bool gemm_compute_i8(xdna_ops * ops, struct ggml_tensor * node) {
    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1];
    const int M = (int) src1->ne[1];
    const int K = (int) src1->ne[0];
    const int N = (int) src0->ne[1];


    xdna_ops::int8_wbo * wb = gemm_int8_weight_bo(ops, src0, K, N);
    if (!wb) {
        return false;
    }
    const xdna_gemm_geom geom = xdna_pick_geom(ops, M, K, N, /* want_i8 */ true);
    const xdna_gemm_tiles & tiles = geom.tiles;
    const int Mk = tiles.M;

    const int n_mb = (M + Mk - 1) / Mk;
    // The raw int32 C is folded straight into dst by d_a * d_w. A K that spans
    // several kernel blocks therefore accumulates the scaled partials in dst
    // and needs no per-block int32 accumulator: one full pass over M x N less
    // per op, and no second buffer. dst is zeroed first in that case; a single
    // K block is the whole reduction and assigns instead.
    const bool single_k = K <= GEMM_K_MAX;
    std::vector<std::vector<float>> d_a((size_t) n_mb);
    {
        for (int b = 0; b < n_mb; b++) {
            d_a[b].assign((size_t) Mk, 1.0f);
        }
        if (!single_k) {
            for (int m = 0; m < M; m++) {
                std::memset((char *) node->data + (size_t) m * node->nb[1], 0,
                            (size_t) N * sizeof(float));
            }
        }
    }

    // Per-row activation scale over the full K: every K-block of a row is
    // quantized with the same d_a, so the per-block results sum into one
    // consistent total before the d_a * d_w rescale.
    {
        for (int b = 0; b < n_mb; b++) {
            const int mc = std::min(Mk, M - b * Mk);
            for (int m = 0; m < mc; m++) {
                const float * row = (const float *) src1->data + (size_t) (b * Mk + m) * K;
                const float amax = xdna_row_amax(row, K);
                d_a[b][m] = amax > 0.0f ? amax / 127.0f : 1.0f;
            }
        }
    }

    // Fold one read-back C block into the destination. The row scale is per
    // row and the weight scale is per column, so both apply once per element.
    // This is the largest host-side cost of the int8 route: it moves M x N
    // elements twice (read the raw int32, write f32) and, run on one core, sits
    // at single-core DRAM bandwidth.
    const float * dw = wb->dw.data();
    auto fold_block = [&](int mb, const int32_t * c) {
        const int pmc = std::min(Mk, M - mb * Mk);
        for (int m = 0; m < pmc; m++) {
            const int32_t * crow = c + (size_t) m * N;
            const float da = d_a[mb][m];
            float * dst = (float *) node->data + (size_t) (mb * Mk + m) * (node->nb[1] / 4);
            if (single_k) {
                for (int n = 0; n < N; n++) {
                    dst[n] = (float) crow[n] * da * dw[n];
                }
            } else {
                for (int n = 0; n < N; n++) {
                    dst[n] += (float) crow[n] * da * dw[n];
                }
            }
        }
    };

    xdna_buffer * bo_a[2] = { nullptr, nullptr };
    xdna_buffer * bo_c[2] = { nullptr, nullptr };
    xrt::run runs[2];
    int bank = 0;
    int pend = -1;
    int pend_mb = 0;

    for (int k0 = 0; k0 < K; k0 += GEMM_K_MAX) {
        const int Kb = std::min(GEMM_K_MAX, K - k0);
        const uint32_t b_offset = (uint32_t) k0 * N;   // int8 bytes into B
        xdna_seq seq;
        if (!xdna_gemm_seq_build(&seq, &tiles, Mk, GEMM_K_MAX, N, b_offset)) {
            GGML_LOG_ERROR("%s: failed to build int8 GEMM stream M=%d K=%d N=%d\n", "xdna-ops", M, K, N);
            return false;
        }
        std::vector<uint32_t> insts = xdna_seq_build(&seq);
        char name[96];
        snprintf(name, sizeof(name), "gemm_i8_K%d_N%d_b%d_m%d", GEMM_K_MAX, N, k0, Mk);
        xdna_kernel * kern = xdna_kernel_pool_get_built(ops->pool, name,
                                                        geom.xclbin,
                                                        insts.data(), insts.size());
        if (!kern) {
            GGML_LOG_ERROR("%s: failed to load int8 GEMM kernel %s\n", "xdna-ops", name);
            return false;
        }

        for (int b = 0; b < n_mb; b++) {
            const int mc = std::min(Mk, M - b * Mk);
            if (bo_a[bank] == nullptr) {
                bo_a[bank] = xdna_kernel_pool_acquire_buffer(ops->pool, (size_t) Mk * GEMM_K_MAX);
                bo_c[bank] = xdna_kernel_pool_acquire_buffer(ops->pool, (size_t) Mk * N * sizeof(int32_t));
                if (!bo_a[bank] || !bo_c[bank]) {
                    GGML_LOG_ERROR("%s: failed to allocate int8 GEMM buffers\n", "xdna-ops");
                    return false;
                }
            }
            {
                int8_t * a = (int8_t *) bo_a[bank]->bo.map();
                // A full block (every row, every K word written below) needs no
                // zeroing; only a partial K block or a padded M tail does.
                if (Kb < GEMM_K_MAX || mc < Mk) {
                    std::memset(a, 0, (size_t) Mk * GEMM_K_MAX);
                }
                for (int m = 0; m < mc; m++) {
                    const float * row = (const float *) src1->data + (size_t) (b * Mk + m) * K + k0;
                    const float d = d_a[b][m];   // full-K row scale (see above)
                    int8_t * dst = a + (size_t) m * GEMM_K_MAX;
                    xdna_quantize_row(dst, row, Kb, 1.0f / d);
                }
            }
            {
                xdna_buffer_sync_to_device(bo_a[bank]);
            }
            {
                xdna_buffer * args[3] = { bo_a[bank], wb->bo, bo_c[bank] };
                runs[bank] = xdna_kernel_run_start(kern, args, 3);
                if (!runs[bank]) {
                    GGML_LOG_ERROR("%s: int8 GEMM submit failed M=%d K=%d N=%d\n", "xdna-ops", M, K, N);
                    return false;
                }
            }

            if (pend >= 0) {
                if (!xdna_run_wait(runs[pend])) {
                    GGML_LOG_ERROR("%s: int8 GEMM wait failed M=%d K=%d N=%d node=%s\n", "xdna-ops", M, K, N, node->name ? node->name : "?");
                    return false;
                }
                {
                    xdna_buffer_sync_from_device(bo_c[pend]);
                }
                {
                    fold_block(pend_mb, (const int32_t *) bo_c[pend]->bo.map());
                }
                {
                    xdna_kernel_pool_release_buffer(ops->pool, bo_a[pend]);
                    xdna_kernel_pool_release_buffer(ops->pool, bo_c[pend]);
                }
                bo_a[pend] = nullptr;
                bo_c[pend] = nullptr;
            }
            pend = bank;
            pend_mb = b;
            bank = 1 - bank;
        }
    }
    if (pend >= 0) {
        if (!xdna_run_wait(runs[pend])) {
            GGML_LOG_ERROR("%s: int8 GEMM final wait failed M=%d K=%d N=%d node=%s\n", "xdna-ops", M, K, N, node->name ? node->name : "?");
            return false;
        }
        {
            xdna_buffer_sync_from_device(bo_c[pend]);
        }
        {
            fold_block(pend_mb, (const int32_t *) bo_c[pend]->bo.map());
        }
        {
            xdna_kernel_pool_release_buffer(ops->pool, bo_a[pend]);
            xdna_kernel_pool_release_buffer(ops->pool, bo_c[pend]);
        }
        bo_a[pend] = nullptr;
        bo_c[pend] = nullptr;
    }
    return true;
}


// MUL_MAT follows ggml's convention: src0 = weights [K, N] (stored as N rows
// of K), src1 = activations [K, M] (stored as M rows of K), so dst[M, N] =
// src1^T @ src0. src1's memory feeds the kernel directly as A [M, K]; src0's
// memory is transposed to B [K, N]. The stream is built per shape via
// xdna_gemm_seq_build, so only A's rows are padded to the baked M block.
//
// Submission is pipelined: two A/C buffer banks are ping-ponged. The run for
// block (m0, k0) is submitted without waiting, then the previous block's run
// is waited and its C read back while the NPU processes the new one, so the
// C readback overlaps the GEMM. The last few runs are finished in
// xdna_ops_finalize().
static bool gemm_compute(xdna_ops * ops, struct ggml_tensor * node) {
    const struct ggml_tensor * src0 = node->src[0];
    const struct ggml_tensor * src1 = node->src[1];

    const int M = (int) src1->ne[1];
    const int K = (int) src1->ne[0];
    const int N = (int) src0->ne[1];

    // Guaranteed to be supported: compute is only reached for ops accepted by
    // xdna_ops_supported.
    const xdna_gemm_geom geom = xdna_pick_geom(ops, M, K, N, /* want_i8 */ false);
    const xdna_gemm_tiles & tiles = geom.tiles;
    const int Mk = tiles.M;   // baked M block of the selected geometry


    // B is a persistent BO with the packed [K_pad x N] weights; each K-block's
    // stream points into it via its K offset, so no per-call weight copy.
    xdna_buffer * bo_w = gemm_weight_bo(ops, src0, K, N);
    if (!bo_w) {
        return false;
    }

    // All blocks use the full GEMM_K_MAX kernel: a partial K block (K < 1024)
    // after a run of full blocks wedges the shared hw_context, so the last
    // block is zero-padded in K and runs the same K1024 kernel as the rest.
    xdna_ops::pending_op op;
    op.node = node;
    op.M = M;
    op.N = N;
    op.Mk = Mk;
    op.m_off = 0;
    for (int m0 = 0; m0 < M; m0 += Mk) {
        xdna_ops::pending_m_block mb;
        mb.m0 = m0;
        op.m_blocks.push_back(std::move(mb));
    }

    // Zero the f32 dst once so each K-block's C partial sum can be accumulated
    // straight into it; no per-op host accumulator or staging copy.
    for (int m = 0; m < M; m++) {
        std::memset((char *) node->data + (size_t) (op.m_off + m) * node->nb[1], 0,
                    (size_t) N * sizeof(float));
    }

    // Ping-pong A/C banks. Bank b is reused only after its in-flight run has
    // been waited and its C read back, so the readback overlaps the NPU run
    // that follows.
    xdna_ops::pending_run banks[2];
    int bank = 0;
    int pending_bank = -1;   // bank with a submitted, un-waited run

    for (int k0 = 0; k0 < K; k0 += GEMM_K_MAX) {
        const int Kb = std::min(GEMM_K_MAX, K - k0);
        const uint32_t b_offset = (uint32_t) k0 * N * 2;

        // Build the instruction stream for (GEMM_K_MAX, N) with this block's
        // B offset and cache the bound kernel under the shape key.
        xdna_seq seq;
        if (!xdna_gemm_seq_build(&seq, &tiles, Mk, GEMM_K_MAX, N, b_offset)) {
            GGML_LOG_ERROR("%s: failed to build GEMM stream M=%d K=%d N=%d\n", "xdna-ops", M, K, N);
            return false;
        }
        std::vector<uint32_t> insts = xdna_seq_build(&seq);

        char name[64];
        snprintf(name, sizeof(name), "gemm_K%d_N%d_b%d_m%d", GEMM_K_MAX, N, k0, Mk);
        xdna_kernel * kern = xdna_kernel_pool_get_built(ops->pool, name, geom.xclbin,
                                                        insts.data(), insts.size());
        if (!kern) {
            GGML_LOG_ERROR("%s: failed to load GEMM kernel %s\n", "xdna-ops", name);
            return false;
        }

        for (auto & mb : op.m_blocks) {
            const int mc = std::min(Mk, M - mb.m0);

            // Pack A into the current bank and submit. The NPU starts on this
            // run right away; the previous bank's run (already queued ahead of
            // it) is then waited and read back while this one runs.
            xdna_ops::pending_run & pr = banks[bank];
            if (pr.bo_a == nullptr) {
                pr.bo_a = xdna_kernel_pool_acquire_buffer(ops->pool, (size_t) Mk * GEMM_K_MAX * sizeof(ggml_bf16_t));
                pr.bo_c = xdna_kernel_pool_acquire_buffer(ops->pool, (size_t) Mk * N * sizeof(float));
                if (!pr.bo_a || !pr.bo_c) {
                    GGML_LOG_ERROR("%s: failed to allocate GEMM buffers\n", "xdna-ops");
                    return false;
                }
            }
            pr.mb_idx = (int) (&mb - op.m_blocks.data());
            {
                ggml_bf16_t * a_map = (ggml_bf16_t *) pr.bo_a->bo.map();
                std::memset(a_map, 0, (size_t) Mk * GEMM_K_MAX * sizeof(ggml_bf16_t));
                for (int m = 0; m < mc; m++) {
                    f32_to_bf16_run((const float *) src1->data + (size_t) (mb.m0 + m) * K + k0,
                                    a_map + (size_t) m * GEMM_K_MAX, Kb);
                }
            }

            {
                xdna_buffer_sync_to_device(pr.bo_a);
            }

            xdna_buffer * args[3] = { pr.bo_a, bo_w, pr.bo_c };
            {
                pr.run = xdna_kernel_run_start(kern, args, 3);
                if (!pr.run) {
                    GGML_LOG_ERROR("%s: GEMM submit failed M=%d K=%d N=%d kernel=gemm_K%d_N%d_b%d\n",
                                   "xdna-ops", M, K, N, GEMM_K_MAX, N, k0);
                    return false;
                }
            }

            // The NPU is now working on `bank`; wait + read back the previous
            // bank's run (queued ahead of this one) in parallel with it.
            if (pending_bank >= 0) {
                xdna_ops::pending_run & prev = banks[pending_bank];
                {
                    if (!xdna_run_wait(prev.run)) {
                        GGML_LOG_ERROR("%s: GEMM wait failed M=%d K=%d N=%d\n", "xdna-ops", M, K, N);
                        return false;
                    }
                }
                // Read back only the valid rows: rows [mc, Mk) of a padded
                // block are zero (the padded A rows are zeroed), so the prefix
                // is exact and the readback drops to mc*N elements.
                const xdna_ops::pending_m_block & pmb = op.m_blocks[prev.mb_idx];
                const int pmc = std::min(Mk, M - pmb.m0);
                {
                    xdna_buffer_sync_from_device(prev.bo_c);
                }
                {
                    const float * c = (const float *) prev.bo_c->bo.map();
                    for (int m = 0; m < pmc; m++) {
                        float * dst = (float *) ((char *) node->data +
                                                 (size_t) (op.m_off + pmb.m0 + m) * node->nb[1]);
                        const float * crow = c + (size_t) m * N;
                        for (int n = 0; n < N; n++) {
                            dst[n] += crow[n];
                        }
                    }
                }
                xdna_kernel_pool_release_buffer(ops->pool, prev.bo_a);
                xdna_kernel_pool_release_buffer(ops->pool, prev.bo_c);
                banks[pending_bank].bo_a = nullptr;
                banks[pending_bank].bo_c = nullptr;
            }

            pending_bank = bank;
            bank = 1 - bank;
        }
    }

    // Flush the trailing banks into the op so finalize waits+reads them.
    if (pending_bank >= 0) {
        op.m_blocks[banks[pending_bank].mb_idx].runs.push_back(std::move(banks[pending_bank]));
        banks[pending_bank] = xdna_ops::pending_run();
        pending_bank = -1;
    }

    ops->pending.push_back(std::move(op));
    return true;
}


bool xdna_ops_finalize(xdna_ops * ops) {
    if (ops->pending.empty()) {
        return true;
    }


    // Wait all runs first so all kernels of the layer complete together.
    for (auto & op : ops->pending) {
        for (auto & mb : op.m_blocks) {
            for (auto & pr : mb.runs) {
                if (!xdna_run_wait(pr.run)) {
                    GGML_LOG_ERROR("%s: finalize: kernel wait failed\n", "xdna-ops");
                    for (auto & fop : ops->pending) {
                        for (auto & fmb : fop.m_blocks) {
                            for (auto & fpr : fmb.runs) {
                                xdna_kernel_pool_release_buffer(ops->pool, fpr.bo_a);
                                xdna_kernel_pool_release_buffer(ops->pool, fpr.bo_c);
                            }
                        }
                    }
                    ops->pending.clear();
                    return false;
                }
            }
        }
    }

    // Read the trailing banks' C straight from their BOs into the (already
    // zeroed and partially accumulated) f32 dst, then release the buffers.
    for (auto & op : ops->pending) {
        const int Mk = op.Mk;   // per-op geometry M block
        for (auto & mb : op.m_blocks) {
            const int mc = std::min(Mk, op.M - mb.m0);
            for (auto & pr : mb.runs) {
                xdna_buffer_sync_from_device(pr.bo_c);
                const float * c = (const float *) pr.bo_c->bo.map();
                for (int m = 0; m < mc; m++) {
                    float * dst = (float *) ((char *) op.node->data +
                                             (size_t) (op.m_off + mb.m0 + m) * op.node->nb[1]);
                    const float * crow = c + (size_t) m * op.N;
                    for (int n = 0; n < op.N; n++) {
                        dst[n] += crow[n];
                    }
                }
            }
            for (auto & pr : mb.runs) {
                xdna_kernel_pool_release_buffer(ops->pool, pr.bo_a);
                xdna_kernel_pool_release_buffer(ops->pool, pr.bo_c);
            }
        }
    }

    ops->pending.clear();
    return true;
}

// --- decode GEMV ------------------------------------------------------------
//
// A single-row MUL_MAT padded up to the GEMM's baked 32-row block wastes 32x
// the work, so decode can go to the GEMV design instead (xdna-gemv.h): the
// weights stay packed and the group parameters are applied to the accumulator.
//
// Off by default: the route is correct for any one shape but not yet across
// them. A core picks up the previous dispatch's tile count, because the
// barrier that should gate the per-core counts is not re-armed between runs -
// a dispatch with K=2048 after one with K=1024 sums half of K, which measures
// as 0.78 relative. Set GGML_XDNA_GEMV=1 to exercise it.

// Ops that only alias their source; the grouping walk steps over them.
static bool xdna_ops_is_view(enum ggml_op op) {
    switch (op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            return false;
    }
}


// The geometry serving this node, or an invalid one. Decode only: ne[1] is the
// number of activation rows.
// The decode GEMV gate. `why`, when given, is set to the first condition that
// failed, so a caller can say which one a graph is losing nodes to instead of
// reporting that nothing was claimed.
static xdna_gemv_geom xdna_gemv_geom_for(const struct ggml_tensor * op,
                                         const char ** why = nullptr,
                                         xdna_gemv_split split = XDNA_GEMV_SPLIT_DEFAULT) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    const auto no = [&](const char * reason) {
        if (why) {
            *why = reason;
        }
        return xdna_gemv_geom{};
    };
    if (op->op != GGML_OP_MUL_MAT || !src0 || !src1) {
        return no("not a mul_mat");
    }
    if (src1->ne[1] != 1) {
        return no("batch > 1 row");
    }
    if (src1->ne[2] * src1->ne[3] != 1 || src0->ne[2] * src0->ne[3] != 1) {
        return no("broadcast dims");
    }
    if (src1->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return no("not f32 in/out");
    }
    if (!ggml_is_contiguous(op) || !ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
        return no("not contiguous");
    }
    const xdna_gemv_geom g = xdna_gemv_variant(src0->type, src0->ne[0], src0->ne[1],
                                               false, split);
    if (!g.valid()) {
        return no("no geometry for the shape");
    }
    return g;
}

// Fetch (or build) the runner for a dispatch, keyed by its first weight.
static xdna_gemv * xdna_gemv_get(xdna_ops * ops,
                                 const struct ggml_tensor * const * ws, int n_w,
                                 const xdna_gemv_geom & geom) {
    {
        std::lock_guard<std::mutex> lock(ops->gemv_mutex);
        auto it = ops->gemv.find(ws[0]->data);
        if (it != ops->gemv.end()) {
            return it->second;
        }
    }
    std::vector<uint8_t> packed;
    if (!xdna_gemv_pack_weights(geom, ws, n_w, nullptr, packed)) {
        GGML_LOG_ERROR("%s: gemv: cannot pack %s\n", "xdna-ops", ggml_get_name(ws[0]));
        return nullptr;
    }
    xdna_gemv * g = xdna_gemv_create(ops->pool, geom, packed);
    if (!g) {
        GGML_LOG_ERROR("%s: gemv: no artifact for %s\n", "xdna-ops", geom.stem().c_str());
    }
    std::lock_guard<std::mutex> lock(ops->gemv_mutex);
    ops->gemv.emplace(ws[0]->data, g);
    return g;
}

// Run one dispatch and hand `node` its slice of the result. The dispatch fires
// on the group's first node, where every member's activation is ready; the
// members' outputs are copied when the graph reaches each of them, because a
// group spans whatever distance ggml put between the projections and writing
// a node's buffer early would land on whatever the allocator still has live
// there.
static bool gemv_compute_group(xdna_ops * ops, xdna_ops::gemv_group & grp,
                               struct ggml_tensor * node) {
    if (!grp.ran) {
        std::vector<const struct ggml_tensor *> ws;
        ws.reserve(grp.nodes.size());
        for (const struct ggml_tensor * n : grp.nodes) {
            ws.push_back(n->src[0]);
        }
        xdna_gemv * g = xdna_gemv_get(ops, ws.data(), (int) ws.size(), grp.geom);
        if (!g) {
            return false;
        }

        grp.buf.resize((size_t) grp.geom.n_real);
        if (!xdna_gemv_run(g, (const float *) grp.nodes[0]->src[1]->data,
                           grp.buf.data())) {
            return false;
        }
        grp.ran = true;
    }

    for (size_t k = 0; k < grp.nodes.size(); k++) {
        if (grp.nodes[k] == node) {
            std::memcpy(node->data, grp.buf.data() + grp.off[k],
                        (size_t) node->ne[0] * sizeof(float));
            return true;
        }
    }
    return true;
}

void xdna_ops_plan_gemv(xdna_ops * ops, const struct ggml_cgraph * cgraph,
                        const std::unordered_set<const struct ggml_tensor *> * skip) {
    // Rebuilt per graph, so every group starts unrun.
    ops->gemv_groups.clear();
    ops->gemv_group_of.clear();
    // The per-op route needs the merged artifact: on any other one a
    // projection outside the fused layers reconfigures the array for its own
    // xclbin, which costs more than the projection.
    if (!ops->fused_gemv) {
        return;
    }

    // The split decides the artifact, and therefore the hardware context. On
    // the merged layer artifact a projection outside the fused layers costs
    // only its own work; on the standalone GEMV xclbin it costs a
    // reconfiguration of the array as well, which is what made this route
    // measure 4x worse than leaving those projections on the host.
    const xdna_gemv_split split = ops->fused_gemv ? XDNA_GEMV_SPLIT_FUSED
                                                  : XDNA_GEMV_SPLIT_DEFAULT;
    const auto claimed = [&](const struct ggml_tensor * n) {
        return skip && skip->count(n) != 0;
    };

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        const bool ok = !claimed(node) && xdna_gemv_geom_for(node, nullptr, split).valid();
        if (!ok || ops->gemv_group_of.count(node)) {
            continue;
        }
        // Every projection that reads the same activation, however far apart
        // they sit: they cannot depend on one another, so running them together
        // only changes how many launches the layer costs. Walking the whole
        // graph rather than the adjacent nodes is what collects a recurrent
        // layer's in_proj, z, beta and gate - which ggml emits twenty nodes
        // apart - into one dispatch instead of four.
        xdna_ops::gemv_group grp;
        grp.nodes.push_back(node);
        grp.off.push_back(0);
        int64_t n_total = node->ne[0];
        // A group has one weight format, and a projection whose natural format
        // is the narrow one used to be left out of the group for that reason -
        // in a recurrent layer that is attn_gate against a Q5_K attn_qkv, two
        // dispatches where the array could run one. q8g16 represents Q4_K
        // exactly, so the group can take the wider format and everything in it
        // repacks. The wider codes cost bytes and the merge saves a dispatch;
        // at 56 GB/s and ~150 us a dispatch that is worth it up to about 8 MB,
        // and the projections this joins are far under.
        // GGML_XDNA_GEMV_PROMOTE=0 keeps a group to one natural format.
        static const bool promote = xdna_env_on("GGML_XDNA_GEMV_PROMOTE");
        enum ggml_type gtype = node->src[0]->type;
        for (int j = i + 1; j < cgraph->n_nodes; j++) {
            struct ggml_tensor * cand = cgraph->nodes[j];
            const xdna_wfmt cf = xdna_wfmt_gemv_for(cand->src[0]->type);
            const bool same_fmt = cf == xdna_wfmt_gemv_for(gtype);
            if (cand->op != GGML_OP_MUL_MAT || cand->src[1] != node->src[1] ||
                cand->src[0]->ne[0] != node->src[0]->ne[0] ||
                cand->ne[1] != 1 ||
                (!same_fmt && !(promote && cf != XDNA_WFMT_NONE)) ||
                claimed(cand) || ops->gemv_group_of.count(cand) ||
                !xdna_gemv_geom_for(cand, nullptr, split).valid()) {
                continue;
            }
            // The group's type is whichever of the two packs into q8g16, so
            // the geometry and the packer both see the wider format.
            const enum ggml_type mtype =
                same_fmt ? gtype
                         : (cf == XDNA_WFMT_Q8G16 ? cand->src[0]->type : gtype);
            const xdna_gemv_geom merged =
                xdna_gemv_variant(mtype, node->src[0]->ne[0],
                                  n_total + cand->ne[0], false, split);
            if (!merged.valid()) {
                break;   // past what one dispatch's descriptors can carry
            }
            gtype = mtype;
            grp.nodes.push_back(cand);
            grp.off.push_back(n_total);
            n_total += cand->ne[0];
        }
        grp.geom = xdna_gemv_variant(gtype, node->src[0]->ne[0], n_total,
                                     false, split);
        if (!grp.geom.valid()) {
            continue;
        }
        const int idx = (int) ops->gemv_groups.size();
        for (struct ggml_tensor * n : grp.nodes) {
            ops->gemv_group_of[n] = idx;
        }
        ops->gemv_groups.push_back(std::move(grp));
    }

}

static bool gemm_supported(const xdna_ops * ops, const struct ggml_tensor * op) {
    if (op->op != GGML_OP_MUL_MAT) {
        return false;
    }

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    if (!src0 || !src1) {
        return false;
    }

    const int K = (int) src1->ne[0];
    const int N = (int) src0->ne[1];
    const int M = (int) src1->ne[1];

    // Decode shapes with a GEMV artifact take that route regardless of the
    // GEMM geometry.
    if (xdna_gemv_geom_for(op).valid()) {
        return true;
    }

    if (ops->gemm_xclbin_decode.empty()) {
        return false;
    }
    if (M <= 0) {
        return false;
    }
    const bool prefill_m = M >= GEMM_M_BIG_MIN;
    const bool is_q = src0->type == GGML_TYPE_Q4_K ||
                      src0->type == GGML_TYPE_Q5_K ||
                      src0->type == GGML_TYPE_Q6_K;
    // Native int8 prefill route for quantized weights when the artifact exists.
    const bool use_i8 = is_q && prefill_m && !ops->i8_xclbin.empty();
    // bf16 route: BF16/F16 at any M, Q4_K falls back to bf16 when the int8
    // artifact is absent or for decode-sized batches. Q5_K/Q6_K have no bf16
    // dequant path here, so only the int8 prefill route covers them.
    const bool use_bf16 = !use_i8 &&
        (src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_F16 ||
         src0->type == GGML_TYPE_Q4_K);
    if (!use_i8 && !use_bf16) {
        return false;
    }
    // The stream is always built for the baked M block, so the geometry check
    // is on the block M; op M is tiled into blocks.
    const xdna_gemm_geom geom = xdna_pick_geom(ops, M, K, N, use_i8);
    if (!geom.xclbin) {
        return false;
    }
    if (!xdna_gemm_seq_supported(&geom.tiles, geom.tiles.M, K, N)) {
        return false;
    }
    // Practical cap on N: wide projections (e.g. the vocabulary output) need a
    // huge B transpose and run too long on the NPU; those stay on the CPU.
    if (N > GEMM_N_MAX) {
        return false;
    }
    if (src0->ne[0] != K) {
        return false;
    }
    if (src0->ne[2] * src0->ne[3] != 1 || src1->ne[2] * src1->ne[3] != 1) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return false;
    }
    // Quantized rows are packed in 256-element blocks (QK_K); dequantization
    // needs whole blocks.
    if (is_q && K % QK_K != 0) {
        return false;
    }
    if (!ggml_is_contiguous(op) || !ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
        return false;
    }

    return true;
}

// --- public API -------------------------------------------------------------

void xdna_ops_init(xdna_ops * ops, xdna_kernel_pool * pool) {
    ops->pool = pool;
    ops->gemm_tiles = xdna_gemm_tiles{};
    ops->gemm_xclbin_decode.clear();
    ops->pref_blocks.clear();
    ops->pref_xclbin.clear();
    ops->i8_xclbin.clear();

    // Stems are gemm_bf16_f32_M%d_K%d_N%d_c%d and gemm_int8_int32_M%d_K%d_N%d_c%d
    // (see CMakeLists). The decode M block (GGML_XDNA_GEMM_M) is the native bf16
    // M32 geometry; every larger bf16 M block is a prefill geometry, as is each
    // int8 M block.
    for (const std::string & name : pool->names) {
        int M = 0, K = 0, N = 0, C = 0;
        if (sscanf(name.c_str(), "gemm_bf16_f32_M%d_K%d_N%d_c%d", &M, &K, &N, &C) == 4) {
            if (M == GGML_XDNA_GEMM_M) {
                if (ops->gemm_xclbin_decode.empty()) {
                    ops->gemm_xclbin_decode = name;
                }
            } else if (M > GGML_XDNA_GEMM_M) {
                ops->pref_xclbin[M] = name;
            }
            continue;
        }
        if (sscanf(name.c_str(), "gemm_int8_int32_M%d_K%d_N%d_c%d", &M, &K, &N, &C) == 4) {
            ops->i8_xclbin[M] = name;
        }
    }

    // Descending: the geometry picker walks the list looking for the cheapest
    // block <= the op's row count.
    ops->pref_blocks.reserve(ops->pref_xclbin.size());
    for (const auto & kv : ops->pref_xclbin) {
        ops->pref_blocks.push_back(kv.first);
    }
    std::sort(ops->pref_blocks.begin(), ops->pref_blocks.end(), std::greater<int>());

    // Native int8 prefill route for quantized weights when the artifact exists
    // (GGML_XDNA_INT8=0 restores the bf16-dequant/CPU behaviour).
    if (xdna_env_int("GGML_XDNA_INT8", 1) == 0) {
        ops->i8_xclbin.clear();
    }

    std::string blocks;
    for (size_t i = 0; i < ops->pref_blocks.size(); i++) {
        if (i > 0) {
            blocks += ",";
        }
        blocks += std::to_string(ops->pref_blocks[i]);
    }
    std::string i8_blocks;
    for (const auto & kv : ops->i8_xclbin) {
        if (!i8_blocks.empty()) {
            i8_blocks += ",";
        }
        i8_blocks += std::to_string(kv.first);
    }
    GGML_LOG_INFO("%s: GEMM geometries: decode=%s prefill blocks=[%s] int8 blocks=[%s] (big-M threshold %d)\n",
                  "xdna-ops", ops->gemm_xclbin_decode.c_str(), blocks.c_str(),
                  i8_blocks.empty() ? "-" : i8_blocks.c_str(), GEMM_M_BIG_MIN);
}

bool xdna_ops_supported(const xdna_ops * ops, const struct ggml_tensor * op) {
    // Isolation mode (see xdna_ops::isolation): the per-op NPU kernels are off
    // for the single-token decode, whose projections feed the fused layer from
    // host-visible buffers. Prefill batches (M > 1) keep their per-op kernels.
    //
    // A projection the GEMV planner grouped runs on the merged layer artifact
    // - the context the fused layers leave resident, so no reconfiguration of
    // the array. With that and the grouping (every projection reading one
    // activation shares a dispatch, however far apart ggml put them) every
    // MUL_MAT of the decode is on the array: MUL_MAT host reads zero.
    //
    // It costs 39.7 -> 31.7 t/s today, and the reason is the weight stream,
    // not the launches: the packed format is 0.656 B/value against ggml's
    // 0.5625 on Q4_K and 1.312 against 0.8203 on Q6_K, so the array moves
    // about 1.3x the bytes at 26 GB/s where the CPU moves them at 28. The
    // in_proj of a recurrent layer is Q6_K and is the single biggest
    // projection in the model, which is most of the difference.
    if (ops->isolation && op->ne[1] <= 1 && !ops->gemv_group_of.count(op)) {
        return false;
    }
    switch (op->op) {
        case GGML_OP_MUL_MAT:
            return gemm_supported(ops, op);
        case GGML_OP_GATED_DELTA_NET:
            // Prefill recurrent body offloaded to the GDN prefill kernel
            // (kernels/gdn_prefill.py) when the artifact is present.
            return xdna_gdn_prefill_supported(op);
        case GGML_OP_SSM_CONV:
            // Depthwise conv1d offloaded to the conv kernel when present.
            return xdna_conv_prefill_supported(op);
        case GGML_OP_FLASH_ATTN_EXT:
            // The six full-attention prefill layers, on the flash attention
            // kernel (kernels/fa.py) when the artifact is present.
            return xdna_fa_prefill_supported(op);
        default:
            return false;
    }
}

bool xdna_ops_compute(xdna_ops * ops, struct ggml_tensor * node) {
    switch (node->op) {
        case GGML_OP_MUL_MAT: {
            // The GEMV runs to completion rather than joining the batch, so
            // pending GEMM submissions are flushed first: it reads and writes
            // host-visible tensors the batch may still be producing.
            auto it = ops->gemv_group_of.find(node);
            if (it != ops->gemv_group_of.end()) {
                if (!xdna_ops_finalize(ops)) {
                    return false;
                }
                return gemv_compute_group(ops, ops->gemv_groups[it->second], node);
            }
            return xdna_int8_eligible(ops, node) ? gemm_compute_i8(ops, node)
                                                 : gemm_compute(ops, node);
        }
        case GGML_OP_GATED_DELTA_NET:
            // The GDN run reads its src tensors' data synchronously, so flush
            // any pending (unfinalized) GEMM submissions first - the scheduler
            // may have placed them in the same NPU chunk ahead of this node.
            if (!xdna_ops_finalize(ops)) {
                return false;
            }
            return xdna_gdn_prefill_run(ops->pool->device, node);
        case GGML_OP_SSM_CONV:
            if (!xdna_ops_finalize(ops)) {
                return false;
            }
            return xdna_conv_prefill_run(ops->pool->device, node);
        case GGML_OP_FLASH_ATTN_EXT:
            if (!xdna_ops_finalize(ops)) {
                return false;
            }
            return xdna_fa_prefill_run(ops->pool->device, node);
        default:
            GGML_LOG_ERROR("%s: unsupported op %d in XDNA graph\n",
                           "xdna-ops", (int) node->op);
            return false;
    }
}
