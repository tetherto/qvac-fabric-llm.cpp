#include "ggml.h"
#include "common.cuh"
#include "convert.cuh"
#include "mmf8.cuh"
#include "unary.cuh"
#ifdef GGML_CUDA_CUTLASS
#include "mmf8-cutlass.cuh"
#endif // GGML_CUDA_CUTLASS

#include <cfloat>

// two packed e4m3 -> two floats; e4m3 values are exact in f16
static __device__ __forceinline__ float2 ggml_cuda_e4m3x2_to_float2(const uint16_t v) {
#if defined(FP8_AVAILABLE) && !defined(GGML_USE_HIP)
    const __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t) v, __NV_E4M3);
    return __half22float2(*reinterpret_cast<const half2 *>(&hr));
#else
    return make_float2(ggml_cuda_e4m3_to_fp32((uint8_t) (v & 0xFF)), ggml_cuda_e4m3_to_fp32((uint8_t) (v >> 8)));
#endif // defined(FP8_AVAILABLE) && !defined(GGML_USE_HIP)
}

#define MMF8_GEMV_NWARPS 4
#define MMF8_GEMV_MAX_NCOLS GGML_CUDA_MMF8_GEMV_MAX_NCOLS

// one trip of weights (16 bytes per row) and the block scale of each matrix at column k
template <int nmat, int nrows_w>
static __device__ __forceinline__ void mmf8_gemv_load_w(
        const uint8_t * xr, const float * sr, const uint8_t * xgr, const float * sgr, const int ncols, const int nblk_n, const int k,
        uint4 (&w4)[nmat][nrows_w], float (&s)[nmat]) {
#pragma unroll
    for (int r = 0; r < nrows_w; ++r) {
        w4[0][r] = *(const uint4 *) (xr + (size_t) r*ncols + k);
        if constexpr (nmat == 2) {
            w4[1][r] = *(const uint4 *) (xgr + (size_t) r*ncols + k);
        }
    }
    s[0] = sr[(k/GGML_F8_E4M3_SCALE_BLOCK)*nblk_n];
    if constexpr (nmat == 2) {
        s[1] = sgr[(k/GGML_F8_E4M3_SCALE_BLOCK)*nblk_n];
    }
}

// a block owns nrows_w consecutive rows and its warps split k in trips of 512 columns (16 bytes per lane, 4 scale blocks)
// one activation slice per trip serves all nrows_w rows: at one row per warp the activation loads were the limiter.
// has_glu: xg/sxg are the gate matrix (same shape and scales layout as x); the activation is read once for both and
// the epilogue writes silu(gate) * up from the same per-row sums the separate launches would produce
// PDL: the grid dependency is waited at entry and the launch completion signalled before the reduction (as mmvf), so
// the next kernel's launch overlaps the epilogue; no __restrict__ on the parameters (PDL rule)
template <int ncols_dst, int nrows_w, bool y_vec, bool has_glu>
static __device__ __forceinline__ void mmf8_gemv_block(
        const uint8_t * x, const float * sx, const uint8_t * xg, const float * sxg,
        const float * y, float * dst,
        const int ncols, const int nblk_n, const int stride_col_y, const int stride_col_dst, const int row0,
        float (&red)[has_glu ? 2 : 1][MMF8_GEMV_NWARPS][nrows_w][ncols_dst]) {
    constexpr int warp_size = 32;
    constexpr int trip      = warp_size*16;
    constexpr int nmat      = has_glu ? 2 : 1;

    const int warp = threadIdx.y;
    const int lane = threadIdx.x;

    const uint8_t * xr  = x  + (size_t) row0*ncols;
    const float   * sr  = sx + row0/GGML_F8_E4M3_SCALE_BLOCK; // nrows_w divides the scale block, so the rows share it
    const uint8_t * xgr = has_glu ? xg  + (size_t) row0*ncols : nullptr;
    const float   * sgr = has_glu ? sxg + row0/GGML_F8_E4M3_SCALE_BLOCK : nullptr;

    float sumf[nmat][nrows_w][ncols_dst];
#pragma unroll
    for (int g = 0; g < nmat; ++g) {
#pragma unroll
        for (int r = 0; r < nrows_w; ++r) {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                sumf[g][r][j] = 0.0f;
            }
        }
    }

    ggml_cuda_pdl_sync();
    for (int k = warp*trip + lane*16; k < ncols; k += MMF8_GEMV_NWARPS*trip) {
        uint4 w4[nmat][nrows_w];
        float s[nmat];
        mmf8_gemv_load_w<nmat, nrows_w>(xr, sr, xgr, sgr, ncols, nblk_n, k, w4, s);

        float v[ncols_dst][16];
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            const float * yj = y + (size_t) j*stride_col_y + k;
            if constexpr (y_vec) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float4 t = ((const float4 *) yj)[i];
                    v[j][4*i + 0] = t.x;
                    v[j][4*i + 1] = t.y;
                    v[j][4*i + 2] = t.z;
                    v[j][4*i + 3] = t.w;
                }
            } else {
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    v[j][i] = yj[i];
                }
            }
        }

#pragma unroll
        for (int g = 0; g < nmat; ++g) {
#pragma unroll
            for (int r = 0; r < nrows_w; ++r) {
                const uint16_t * w2 = (const uint16_t *) &w4[g][r];
                float w[16];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float2 f = ggml_cuda_e4m3x2_to_float2(w2[i]);
                    w[2*i + 0] = f.x;
                    w[2*i + 1] = f.y;
                }
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    float part = 0.0f;
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        part += w[i]*v[j][i];
                    }
                    sumf[g][r][j] += s[g]*part;
                }
            }
        }
    }

    ggml_cuda_pdl_lc();
#pragma unroll
    for (int g = 0; g < nmat; ++g) {
#pragma unroll
        for (int r = 0; r < nrows_w; ++r) {
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                sumf[g][r][j] = warp_reduce_sum<warp_size>(sumf[g][r][j]);
                if (lane == 0) {
                    red[g][warp][r][j] = sumf[g][r][j];
                }
            }
        }
    }
    __syncthreads();

    if (warp == 0 && lane < nrows_w*ncols_dst) {
        const int r = lane / ncols_dst;
        const int j = lane % ncols_dst;
        float t[nmat];
#pragma unroll
        for (int g = 0; g < nmat; ++g) {
            t[g] = 0.0f;
#pragma unroll
            for (int w = 0; w < MMF8_GEMV_NWARPS; ++w) {
                t[g] += red[g][w][r][j];
            }
        }
        if constexpr (has_glu) {
            dst[(size_t) j*stride_col_dst + row0 + r] = ggml_cuda_op_silu_single(t[1])*t[0]; // silu(gate) * up, as the GLU kernel
        } else {
            dst[(size_t) j*stride_col_dst + row0 + r] = t[0];
        }
    }
}

template <int ncols_dst, int nrows_w, bool y_vec, bool has_glu>
static __global__ void mul_mat_vec_f8_e4m3(
        const uint8_t * x, const float * sx, const uint8_t * xg, const float * sxg,
        const float * y, float * dst,
        const int ncols, const int nblk_n, const int stride_col_y, const int stride_col_dst) {
    __shared__ float red[has_glu ? 2 : 1][MMF8_GEMV_NWARPS][nrows_w][ncols_dst];
    mmf8_gemv_block<ncols_dst, nrows_w, y_vec, has_glu>(x, sx, xg, sxg, y, dst, ncols, nblk_n, stride_col_y, stride_col_dst, blockIdx.x*nrows_w, red);
}

// up to three F8 matrices sharing one activation in one launch (q/k/v, the delta-net qkv/z): block b computes rows of
// the matrix whose block range holds b, with the plain kernel's per-row arithmetic (bit-identical per output)
#define MMF8_MULTI_MAX 3
struct mmf8_multi_args {
    const uint8_t * x[MMF8_MULTI_MAX];
    const float   * sx[MMF8_MULTI_MAX];
    float         * dst[MMF8_MULTI_MAX];
    int nrows[MMF8_MULTI_MAX];
    int block0[MMF8_MULTI_MAX + 1]; // first block of each matrix; block0[n] = the grid size
    int n;
};

template <int ncols_dst, int nrows_w, bool y_vec>
static __global__ void mul_mat_vec_f8_e4m3_multi(const mmf8_multi_args a, const float * y, const int ncols, const int stride_col_y) {
    __shared__ float red[1][MMF8_GEMV_NWARPS][nrows_w][ncols_dst];
    // constant indices only: a dynamic index into the parameter struct would copy it to local memory per thread
    const uint8_t * x     = a.x[0];
    const float   * sx    = a.sx[0];
    float         * dst   = a.dst[0];
    int             nrows = a.nrows[0];
    int             b0    = a.block0[0];
#pragma unroll
    for (int i = 1; i < MMF8_MULTI_MAX; ++i) {
        if (i < a.n && (int) blockIdx.x >= a.block0[i]) {
            x     = a.x[i];
            sx    = a.sx[i];
            dst   = a.dst[i];
            nrows = a.nrows[i];
            b0    = a.block0[i];
        }
    }
    const int row0 = (blockIdx.x - b0)*nrows_w;
    mmf8_gemv_block<ncols_dst, nrows_w, y_vec, false>(x, sx, nullptr, nullptr, y, dst, ncols, nrows/GGML_F8_E4M3_SCALE_BLOCK, stride_col_y, nrows, row0, red);
}

template <int ncols_dst>
static void launch_mul_mat_vec_f8_e4m3_multi(const mmf8_multi_args & a, const float * y, const int ncols, const int stride_col_y, cudaStream_t stream) {
    constexpr int nrows_w = ncols_dst == 1 ? 4 : (ncols_dst == 2 ? 2 : 1);
    mmf8_multi_args b = a;
    b.block0[0] = 0;
    for (int i = 0; i < a.n; ++i) {
        GGML_ASSERT(a.nrows[i] % nrows_w == 0);
        b.block0[i + 1] = b.block0[i] + a.nrows[i]/nrows_w;
    }
    const dim3 block_dims(32, MMF8_GEMV_NWARPS, 1);
    const dim3 block_nums(b.block0[a.n], 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
    const bool y_vec = ((uintptr_t) y % sizeof(float4) == 0) && stride_col_y % 4 == 0;
    if (y_vec) {
        ggml_cuda_kernel_launch(mul_mat_vec_f8_e4m3_multi<ncols_dst, nrows_w, true>, launch_params, b, y, ncols, stride_col_y);
    } else {
        ggml_cuda_kernel_launch(mul_mat_vec_f8_e4m3_multi<ncols_dst, nrows_w, false>, launch_params, b, y, ncols, stride_col_y);
    }
}

static void mul_mat_vec_f8_e4m3_multi_cuda(const mmf8_multi_args & a, const float * y, const int ncols, const int ncols_dst, const int stride_col_y, cudaStream_t stream) {
    switch (ncols_dst) {
        case 1: launch_mul_mat_vec_f8_e4m3_multi<1>(a, y, ncols, stride_col_y, stream); break;
        case 2: launch_mul_mat_vec_f8_e4m3_multi<2>(a, y, ncols, stride_col_y, stream); break;
        case 3: launch_mul_mat_vec_f8_e4m3_multi<3>(a, y, ncols, stride_col_y, stream); break;
        case 4: launch_mul_mat_vec_f8_e4m3_multi<4>(a, y, ncols, stride_col_y, stream); break;
        case 5: launch_mul_mat_vec_f8_e4m3_multi<5>(a, y, ncols, stride_col_y, stream); break;
        case 6: launch_mul_mat_vec_f8_e4m3_multi<6>(a, y, ncols, stride_col_y, stream); break;
        case 7: launch_mul_mat_vec_f8_e4m3_multi<7>(a, y, ncols, stride_col_y, stream); break;
        case 8: launch_mul_mat_vec_f8_e4m3_multi<8>(a, y, ncols, stride_col_y, stream); break;
        default: GGML_ABORT("fatal error");
    }
}

template <int ncols_dst, int nrows_w>
static void launch_mul_mat_vec_f8_e4m3_rows(
        const uint8_t * x, const float * sx, const uint8_t * xg, const float * sxg, const float * y, float * dst,
        const int ncols, const int nrows, const int nblk_n, const int stride_col_y, const int stride_col_dst, cudaStream_t stream) {
    GGML_ASSERT(nrows % nrows_w == 0);
    const dim3 block_dims(32, MMF8_GEMV_NWARPS, 1);
    const dim3 block_nums(nrows/nrows_w, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
    const bool y_vec = ((uintptr_t) y % sizeof(float4) == 0) && stride_col_y % 4 == 0;
    if (xg != nullptr) {
        // the fused up/gate epilogue keeps two accumulator sets: batch <= 4 only
        if constexpr (ncols_dst <= 4) {
            if (y_vec) {
                ggml_cuda_kernel_launch(mul_mat_vec_f8_e4m3<ncols_dst, nrows_w, true, true>, launch_params, x, sx, xg, sxg, y, dst, ncols, nblk_n, stride_col_y, stride_col_dst);
            } else {
                ggml_cuda_kernel_launch(mul_mat_vec_f8_e4m3<ncols_dst, nrows_w, false, true>, launch_params, x, sx, xg, sxg, y, dst, ncols, nblk_n, stride_col_y, stride_col_dst);
            }
            return;
        }
        GGML_ABORT("fused F8 up/gate GEMV: batch above 4");
    }
    const uint8_t * no_xg  = nullptr;
    const float   * no_sxg = nullptr;
    if (y_vec) {
        ggml_cuda_kernel_launch(mul_mat_vec_f8_e4m3<ncols_dst, nrows_w, true, false>, launch_params, x, sx, no_xg, no_sxg, y, dst, ncols, nblk_n, stride_col_y, stride_col_dst);
    } else {
        ggml_cuda_kernel_launch(mul_mat_vec_f8_e4m3<ncols_dst, nrows_w, false, false>, launch_params, x, sx, no_xg, no_sxg, y, dst, ncols, nblk_n, stride_col_y, stride_col_dst);
    }
}

template <int ncols_dst>
static void launch_mul_mat_vec_f8_e4m3(
        const uint8_t * x, const float * sx, const uint8_t * xg, const float * sxg, const float * y, float * dst,
        const int ncols, const int nrows, const int nblk_n, const int stride_col_y, const int stride_col_dst, cudaStream_t stream) {
    // 4 rows per warp at batch 1, fewer when more activation columns have to stay in registers
    constexpr int nrows_w = ncols_dst == 1 ? 4 : (ncols_dst == 2 ? 2 : 1);
    launch_mul_mat_vec_f8_e4m3_rows<ncols_dst, nrows_w>(x, sx, xg, sxg, y, dst, ncols, nrows, nblk_n, stride_col_y, stride_col_dst, stream);
}

// xg/sxg: the gate matrix of a fused up/gate GEMV (nullptr for a plain one)
static void mul_mat_vec_f8_e4m3_cuda(
        const uint8_t * x, const float * sx, const uint8_t * xg, const float * sxg, const float * y, float * dst,
        const int ncols, const int nrows, const int nblk_n, const int ncols_dst, const int stride_col_y, const int stride_col_dst, cudaStream_t stream) {
    switch (ncols_dst) {
        case 1: launch_mul_mat_vec_f8_e4m3<1>(x, sx, xg, sxg, y, dst, ncols, nrows, nblk_n, stride_col_y, stride_col_dst, stream); break;
        case 2: launch_mul_mat_vec_f8_e4m3<2>(x, sx, xg, sxg, y, dst, ncols, nrows, nblk_n, stride_col_y, stride_col_dst, stream); break;
        case 3: launch_mul_mat_vec_f8_e4m3<3>(x, sx, xg, sxg, y, dst, ncols, nrows, nblk_n, stride_col_y, stride_col_dst, stream); break;
        case 4: launch_mul_mat_vec_f8_e4m3<4>(x, sx, xg, sxg, y, dst, ncols, nrows, nblk_n, stride_col_y, stride_col_dst, stream); break;
        case 5: launch_mul_mat_vec_f8_e4m3<5>(x, sx, xg, sxg, y, dst, ncols, nrows, nblk_n, stride_col_y, stride_col_dst, stream); break;
        case 6: launch_mul_mat_vec_f8_e4m3<6>(x, sx, xg, sxg, y, dst, ncols, nrows, nblk_n, stride_col_y, stride_col_dst, stream); break;
        case 7: launch_mul_mat_vec_f8_e4m3<7>(x, sx, xg, sxg, y, dst, ncols, nrows, nblk_n, stride_col_y, stride_col_dst, stream); break;
        case 8: launch_mul_mat_vec_f8_e4m3<8>(x, sx, xg, sxg, y, dst, ncols, nrows, nblk_n, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("fatal error");
    }
}

// y[n*ncols + k] = e4m3(x[n*ncols + k]) * scale[(k/128)*nblk_n + n/128], 8 elements per thread; one flat grid over
// (row, 8-column group) so that weights with more than 65535 rows (the output head) fit the launch limits
static __global__ void dequant_f8_e4m3_blockscaled_f16(
        const uint8_t * __restrict__ x, const float * __restrict__ sx, half * __restrict__ y, const int ncols, const int nblk_n, const int64_t n_groups) {
    const int64_t g = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (g >= n_groups) {
        return;
    }
    const int n8  = ncols/8;
    const int row = g / n8;
    const int k   = (g % n8)*8;
    const size_t i = (size_t) row*ncols + k;
    const uint2 w8 = *(const uint2 *) (x + i);
    const uint16_t * w2 = (const uint16_t *) &w8;
    const float s = sx[(k/GGML_F8_E4M3_SCALE_BLOCK)*nblk_n + row/GGML_F8_E4M3_SCALE_BLOCK];
    half2 out[4];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const float2 f = ggml_cuda_e4m3x2_to_float2(w2[p]);
        out[p] = __floats2half2_rn(f.x*s, f.y*s);
    }
    *(uint4 *) (y + i) = *(const uint4 *) out;
}

static void dequant_f8_e4m3_blockscaled_f16_cuda(
        const uint8_t * x, const float * sx, half * y, const int ncols, const int nrows, const int nblk_n, cudaStream_t stream) {
    const int64_t n_groups = (int64_t) nrows*(ncols/8);
    const dim3 block_nums((n_groups + CUDA_DEQUANTIZE_BLOCK_SIZE - 1)/CUDA_DEQUANTIZE_BLOCK_SIZE, 1, 1);
    dequant_f8_e4m3_blockscaled_f16<<<block_nums, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(x, sx, y, ncols, nblk_n, n_groups);
}

// scaled dequant to F16 followed by cuBLAS with F32 accumulation
static void mul_mat_f8_e4m3_cublas(
        ggml_backend_cuda_context & ctx, const uint8_t * x, const float * sx, const float * y, float * dst,
        const int ncols, const int nrows, const int nblk_n, const int64_t ntokens, cudaStream_t stream) {
    ggml_cuda_pool_alloc<half> x16(ctx.pool(), (size_t) nrows*ncols);
    dequant_f8_e4m3_blockscaled_f16_cuda(x, sx, x16.get(), ncols, nrows, nblk_n, stream);

    ggml_cuda_pool_alloc<half> y16(ctx.pool(), (size_t) ntokens*ncols);
    const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
    to_fp16(y, y16.get(), ntokens*ncols, stream);

    static const float alpha = 1.0f;
    static const float beta  = 0.0f;
    CUBLAS_CHECK(
        cublasGemmEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                nrows, ntokens, ncols,
                &alpha, x16.get(), CUDA_R_16F, ncols,
                        y16.get(), CUDA_R_16F, ncols,
                &beta,  dst,       CUDA_R_32F, nrows,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

#ifdef GGML_CUDA_CUTLASS
// the quantizer of one lane's 4 values of a (token, 128-k group) pair: warp amax, scale = amax/448 floored at FLT_MIN,
// two packed e4m3 conversions; scales stored [K/128][M_pad] (tokens contiguous) as CUTLASS expects
static __device__ __forceinline__ void quantize_group128_lane(
        const float4 v, uint8_t * q, float * sfa, const size_t i, const size_t s_idx, const int lane) {
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    amax = warp_reduce_max<32>(amax);

    const float scale = fmaxf(amax/448.0f, FLT_MIN); // keeps 1/scale finite for zero or subnormal groups
    const float inv   = 1.0f/scale;
    const __nv_fp8x2_storage_t lo = __nv_cvt_float2_to_fp8x2(make_float2(v.x*inv, v.y*inv), __NV_SATFINITE, __NV_E4M3);
    const __nv_fp8x2_storage_t hi = __nv_cvt_float2_to_fp8x2(make_float2(v.z*inv, v.w*inv), __NV_SATFINITE, __NV_E4M3);
    *(uint32_t *) (q + i) = (uint32_t) lo | ((uint32_t) hi << 16);
    if (lane == 0) {
        sfa[s_idx] = scale;
    }
}

// one warp per (token, 128-k group), 4 floats per lane
static __global__ void quantize_f8_e4m3_group128(
        const float * __restrict__ x, uint8_t * __restrict__ q, float * __restrict__ sfa, const int stride_sfa, const int ntokens, const int ncols) {
    const int nblk_k = ncols/GGML_F8_E4M3_SCALE_BLOCK;
    const int pair   = blockIdx.x*(blockDim.x/32) + threadIdx.x/32;
    const int lane   = threadIdx.x % 32;
    if (pair >= ntokens*nblk_k) {
        return;
    }
    const int m  = pair % ntokens;
    const int kb = pair / ntokens;

    const size_t i = (size_t) m*ncols + (size_t) kb*GGML_F8_E4M3_SCALE_BLOCK + lane*4;
    const float4 v = *(const float4 *) (x + i);
    quantize_group128_lane(v, q, sfa, i, (size_t) kb*stride_sfa + m, lane);
}

// the same quantizer fed by silu(gate) * up computed in fp32 exactly as the GLU kernel does, so the e4m3 input of the
// down projection is bit-identical to quantizing the materialized GLU output; gate and up rows have their own strides
static __global__ void glu_quantize_f8_e4m3_group128(
        const float * __restrict__ g, const float * __restrict__ u, uint8_t * __restrict__ q, float * __restrict__ sfa,
        const int stride_sfa, const int ntokens, const int ncols, const int64_t o_g, const int64_t o_u) {
    const int nblk_k = ncols/GGML_F8_E4M3_SCALE_BLOCK;
    const int pair   = blockIdx.x*(blockDim.x/32) + threadIdx.x/32;
    const int lane   = threadIdx.x % 32;
    if (pair >= ntokens*nblk_k) {
        return;
    }
    const int m  = pair % ntokens;
    const int kb = pair / ntokens;

    const size_t col = (size_t) kb*GGML_F8_E4M3_SCALE_BLOCK + lane*4;
    const float4 gg = *(const float4 *) (g + (size_t) m*o_g + col);
    const float4 uu = *(const float4 *) (u + (size_t) m*o_u + col);
    const float4 v  = make_float4(ggml_cuda_op_silu_single(gg.x)*uu.x, ggml_cuda_op_silu_single(gg.y)*uu.y,
                                  ggml_cuda_op_silu_single(gg.z)*uu.z, ggml_cuda_op_silu_single(gg.w)*uu.w);
    quantize_group128_lane(v, q, sfa, (size_t) m*ncols + col, (size_t) kb*stride_sfa + m, lane);
}

// the per-token e4m3 activation with [K/128][M_pad] scales: pool buffers, M padded to a multiple of 4 (the scale
// tensor's TMA row stride must be 16 bytes), quantized from y or from silu(glu_gate) * glu_up
struct mmf8_quantized_act {
    ggml_cuda_pool_alloc<uint8_t> yq;
    ggml_cuda_pool_alloc<float>   ys;
    int64_t ntokens_pad;
    mmf8_quantized_act(ggml_cuda_pool & pool) : yq(pool), ys(pool), ntokens_pad(0) {}
};

static void mmf8_quantize_activations(
        mmf8_quantized_act & q, const float * y, const int ncols, const int64_t ntokens, cudaStream_t stream,
        const float * glu_gate = nullptr, const float * glu_up = nullptr, const int64_t o_g = 0, const int64_t o_u = 0) {
    const int nblk_k = ncols/GGML_F8_E4M3_SCALE_BLOCK;
    q.ntokens_pad = (ntokens + 3) & ~(int64_t) 3;
    q.yq.alloc((size_t) q.ntokens_pad*ncols);
    q.ys.alloc((size_t) q.ntokens_pad*nblk_k);
    if (q.ntokens_pad != ntokens) {
        CUDA_CHECK(cudaMemsetAsync(q.yq.get() + (size_t) ntokens*ncols, 0, (size_t) (q.ntokens_pad - ntokens)*ncols, stream));
        CUDA_CHECK(cudaMemsetAsync(q.ys.get(), 0, (size_t) q.ntokens_pad*nblk_k*sizeof(float), stream));
    }
    const int64_t npairs = ntokens*nblk_k;
    const dim3 block_nums((npairs + 7)/8, 1, 1);
    if (glu_gate != nullptr) {
        glu_quantize_f8_e4m3_group128<<<block_nums, 256, 0, stream>>>(glu_gate, glu_up, q.yq.get(), q.ys.get(), q.ntokens_pad, ntokens, ncols, o_g, o_u);
    } else {
        quantize_f8_e4m3_group128<<<block_nums, 256, 0, stream>>>(y, q.yq.get(), q.ys.get(), q.ntokens_pad, ntokens, ncols);
    }
}

// the GEMM of one weight against a quantized activation; a padded M runs into a pool buffer and is copied out
static void mmf8_gemm_cutlass_quantized(
        ggml_backend_cuda_context & ctx, mmf8_quantized_act & q, const uint8_t * x, const float * sx, float * dst,
        const int ncols, const int nrows, const int64_t ntokens, cudaStream_t stream) {
    const size_t ws_size = ggml_cuda_mmf8_cutlass_workspace_size(q.ntokens_pad, nrows, ncols);
    ggml_cuda_pool_alloc<uint8_t> ws(ctx.pool(), ws_size > 0 ? ws_size : 16);
    if (q.ntokens_pad == ntokens) {
        ggml_cuda_mmf8_cutlass(q.yq.get(), q.ys.get(), x, sx, dst, ntokens, nrows, ncols, ws.get(), ws_size, stream);
        return;
    }
    ggml_cuda_pool_alloc<float> dst_pad(ctx.pool(), (size_t) q.ntokens_pad*nrows);
    ggml_cuda_mmf8_cutlass(q.yq.get(), q.ys.get(), x, sx, dst_pad.get(), q.ntokens_pad, nrows, ncols, ws.get(), ws_size, stream);
    CUDA_CHECK(cudaMemcpyAsync(dst, dst_pad.get(), (size_t) ntokens*nrows*sizeof(float), cudaMemcpyDeviceToDevice, stream));
}

// y is the F32 activation; with glu_gate set the activation is silu(glu_gate) * glu_up instead (never materialized)
static void mul_mat_f8_e4m3_cutlass(
        ggml_backend_cuda_context & ctx, const uint8_t * x, const float * sx, const float * y, float * dst,
        const int ncols, const int nrows, const int64_t ntokens, cudaStream_t stream,
        const float * glu_gate = nullptr, const float * glu_up = nullptr, const int64_t o_g = 0, const int64_t o_u = 0) {
    mmf8_quantized_act q(ctx.pool());
    mmf8_quantize_activations(q, y, ncols, ntokens, stream, glu_gate, glu_up, o_g, o_u);
    mmf8_gemm_cutlass_quantized(ctx, q, x, sx, dst, ncols, nrows, ntokens, stream);
}
#endif // GGML_CUDA_CUTLASS

bool ggml_cuda_mul_mat_f8_uses_cutlass(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1) {
#ifdef GGML_CUDA_CUTLASS
    // GGML_CUDA_DISABLE_MMF8_CUTLASS: run the F16 fallback GEMM instead (A/B and equivalence gates)
    static const bool cutlass_disabled = getenv("GGML_CUDA_DISABLE_MMF8_CUTLASS") != nullptr;
    return src0->type == GGML_TYPE_F8_E4M3 && ggml_nrows(src1) > MMF8_GEMV_MAX_NCOLS && !cutlass_disabled &&
        ggml_cuda_info().devices[ctx.device].cc == GGML_CUDA_CC_HOPPER;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    return false;
#endif // GGML_CUDA_CUTLASS
}

static void ggml_cuda_mul_mat_f8_check(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    const ggml_tensor * src0_s = dst->src[2];

    GGML_ASSERT(src0->type == GGML_TYPE_F8_E4M3);
    GGML_ASSERT(src0_s && src0_s->type == GGML_TYPE_F32 && "F8_E4M3 mul_mat needs the block scale tensor as src[2]");
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0) && ggml_is_contiguous(src0_s) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst));
    GGML_ASSERT(src0->ne[2] == 1 && src0->ne[3] == 1);
    GGML_ASSERT(src0->ne[0] % GGML_F8_E4M3_SCALE_BLOCK == 0 && src0->ne[1] % GGML_F8_E4M3_SCALE_BLOCK == 0);
    GGML_ASSERT(src0_s->ne[0] == src0->ne[1]/GGML_F8_E4M3_SCALE_BLOCK && src0_s->ne[1] == src0->ne[0]/GGML_F8_E4M3_SCALE_BLOCK);
    GGML_ASSERT(src1->ne[0] == src0->ne[0]);
}

void ggml_cuda_mul_mat_f8_glu_cutlass(ggml_backend_cuda_context & ctx, const ggml_tensor * glu, ggml_tensor * dst) {
#ifdef GGML_CUDA_CUTLASS
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * gate = glu->src[0];
    const ggml_tensor * up   = glu->src[1];
    ggml_cuda_mul_mat_f8_check(src0, glu, dst);
    GGML_ASSERT(ggml_cuda_mul_mat_f8_uses_cutlass(ctx, src0, glu));
    GGML_ASSERT(gate->type == GGML_TYPE_F32 && up->type == GGML_TYPE_F32);
    GGML_ASSERT(gate->ne[0] == src0->ne[0] && up->ne[0] == src0->ne[0]);
    GGML_ASSERT(ggml_nrows(gate) == ggml_nrows(glu) && ggml_nrows(up) == ggml_nrows(glu));
    GGML_ASSERT((uintptr_t) gate->data % 16 == 0 && (uintptr_t) up->data % 16 == 0 && gate->nb[1] % 16 == 0 && up->nb[1] % 16 == 0);

    const int     ncols   = src0->ne[0];
    const int     nrows   = src0->ne[1];
    const int64_t ntokens = ggml_nrows(glu);

    mul_mat_f8_e4m3_cutlass(ctx, (const uint8_t *) src0->data, (const float *) dst->src[2]->data, nullptr, (float *) dst->data,
                            ncols, nrows, ntokens, ctx.stream(),
                            (const float *) gate->data, (const float *) up->data, gate->nb[1]/sizeof(float), up->nb[1]/sizeof(float));
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(glu);
    GGML_UNUSED(dst);
    GGML_ABORT("F8 GLU fusion needs the CUTLASS build");
#endif // GGML_CUDA_CUTLASS
}

void ggml_cuda_mul_mat_f8(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    ggml_cuda_mul_mat_f8_check(src0, src1, dst);

    const int     ncols   = src0->ne[0];
    const int     nrows   = src0->ne[1];
    const int     nblk_n  = nrows/GGML_F8_E4M3_SCALE_BLOCK;
    const int64_t ntokens = ggml_nrows(src1);

    const uint8_t * x  = (const uint8_t *) src0->data;
    const float   * sx = (const float *)   dst->src[2]->data;
    const float   * y  = (const float *)   src1->data;
    float         * d  = (float *)         dst->data;

    cudaStream_t stream = ctx.stream();

    if (ntokens <= MMF8_GEMV_MAX_NCOLS) {
        mul_mat_vec_f8_e4m3_cuda(x, sx, nullptr, nullptr, y, d, ncols, nrows, nblk_n, ntokens, ncols, nrows, stream);
        return;
    }

#ifdef GGML_CUDA_CUTLASS
    if (ggml_cuda_mul_mat_f8_uses_cutlass(ctx, src0, src1) && ((uintptr_t) y % 16 == 0)) {
        mul_mat_f8_e4m3_cutlass(ctx, x, sx, y, d, ncols, nrows, ntokens, stream);
        return;
    }
#endif // GGML_CUDA_CUTLASS

    mul_mat_f8_e4m3_cublas(ctx, x, sx, y, d, ncols, nrows, nblk_n, ntokens, stream);
}

void ggml_cuda_mul_mat_f8_gemv_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * up, const ggml_tensor * gate, ggml_tensor * glu) {
    const ggml_tensor * src0  = up->src[0];
    const ggml_tensor * src0g = gate->src[0];
    const ggml_tensor * src1  = up->src[1];
    ggml_cuda_mul_mat_f8_check(src0, src1, up);
    ggml_cuda_mul_mat_f8_check(src0g, src1, gate);
    GGML_ASSERT(gate->src[1] == src1);
    GGML_ASSERT(ggml_are_same_shape(src0, src0g) && ggml_are_same_shape(up->src[2], gate->src[2]));
    GGML_ASSERT(glu->type == GGML_TYPE_F32 && ggml_is_contiguous(glu));
    GGML_ASSERT(glu->ne[0] == src0->ne[1] && ggml_nrows(glu) == ggml_nrows(src1));

    const int     ncols   = src0->ne[0];
    const int     nrows   = src0->ne[1];
    const int     nblk_n  = nrows/GGML_F8_E4M3_SCALE_BLOCK;
    const int64_t ntokens = ggml_nrows(src1);
    GGML_ASSERT(ntokens <= 4);

    mul_mat_vec_f8_e4m3_cuda((const uint8_t *) src0->data, (const float *) up->src[2]->data,
                             (const uint8_t *) src0g->data, (const float *) gate->src[2]->data,
                             (const float *) src1->data, (float *) glu->data, ncols, nrows, nblk_n, ntokens, ncols, nrows, ctx.stream());
}

// n F8 mul_mats (2 or 3) on the same F32 activation at the CUTLASS batch: the activation is quantized once, then one
// GEMM per weight; the bytes and scales are what each unfused call would have produced
void ggml_cuda_mul_mat_f8_shared_cutlass(ggml_backend_cuda_context & ctx, ggml_tensor ** dsts, const int n) {
#ifdef GGML_CUDA_CUTLASS
    GGML_ASSERT(n >= 2 && n <= MMF8_MULTI_MAX);
    const ggml_tensor * src1 = dsts[0]->src[1];
    for (int i = 0; i < n; ++i) {
        ggml_cuda_mul_mat_f8_check(dsts[i]->src[0], src1, dsts[i]);
        GGML_ASSERT(dsts[i]->src[1] == src1);
        GGML_ASSERT((uintptr_t) dsts[i]->data % 16 == 0);
    }
    GGML_ASSERT(ggml_cuda_mul_mat_f8_uses_cutlass(ctx, dsts[0]->src[0], src1) && (uintptr_t) src1->data % 16 == 0);

    const int     ncols   = src1->ne[0];
    const int64_t ntokens = ggml_nrows(src1);
    cudaStream_t  stream  = ctx.stream();

    mmf8_quantized_act q(ctx.pool());
    mmf8_quantize_activations(q, (const float *) src1->data, ncols, ntokens, stream);
    for (int i = 0; i < n; ++i) {
        const ggml_tensor * src0 = dsts[i]->src[0];
        mmf8_gemm_cutlass_quantized(ctx, q, (const uint8_t *) src0->data, (const float *) dsts[i]->src[2]->data, (float *) dsts[i]->data,
                                    ncols, src0->ne[1], ntokens, stream);
    }
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(dsts);
    GGML_UNUSED(n);
    GGML_ABORT("F8 shared-activation GEMM needs the CUTLASS build");
#endif // GGML_CUDA_CUTLASS
}

// the same at the GEMV batch (<= MMF8_GEMV_MAX_NCOLS tokens): one launch over all matrices
void ggml_cuda_mul_mat_f8_shared_gemv(ggml_backend_cuda_context & ctx, ggml_tensor ** dsts, const int n) {
    GGML_ASSERT(n >= 2 && n <= MMF8_MULTI_MAX);
    const ggml_tensor * src1 = dsts[0]->src[1];
    const int     ncols   = src1->ne[0];
    const int64_t ntokens = ggml_nrows(src1);
    GGML_ASSERT(ntokens <= MMF8_GEMV_MAX_NCOLS);

    mmf8_multi_args a = {};
    a.n = n;
    for (int i = 0; i < n; ++i) {
        const ggml_tensor * src0 = dsts[i]->src[0];
        ggml_cuda_mul_mat_f8_check(src0, src1, dsts[i]);
        GGML_ASSERT(dsts[i]->src[1] == src1);
        a.x[i]     = (const uint8_t *) src0->data;
        a.sx[i]    = (const float *) dsts[i]->src[2]->data;
        a.dst[i]   = (float *) dsts[i]->data;
        a.nrows[i] = src0->ne[1];
    }
    mul_mat_vec_f8_e4m3_multi_cuda(a, (const float *) src1->data, ncols, ntokens, ncols, ctx.stream());
}
