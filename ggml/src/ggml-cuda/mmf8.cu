#include "ggml.h"
#include "common.cuh"
#include "convert.cuh"
#include "mmf8.cuh"
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
#define MMF8_GEMV_MAX_NCOLS 8

// a block owns nrows_w consecutive rows and its warps split k in trips of 512 columns (16 bytes per lane, 4 scale blocks)
// one activation slice per trip serves all nrows_w rows: at one row per warp the activation loads were the limiter
template <int ncols_dst, int nrows_w, bool y_vec>
static __global__ void mul_mat_vec_f8_e4m3(
        const uint8_t * __restrict__ x, const float * __restrict__ sx, const float * __restrict__ y, float * __restrict__ dst,
        const int ncols, const int nblk_n, const int stride_col_y, const int stride_col_dst) {
    constexpr int warp_size = 32;
    constexpr int trip      = warp_size*16;
    __shared__ float red[MMF8_GEMV_NWARPS][nrows_w][ncols_dst];

    const int row0 = blockIdx.x*nrows_w;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;

    const uint8_t * xr = x  + (size_t) row0*ncols;
    const float   * sr = sx + row0/GGML_F8_E4M3_SCALE_BLOCK; // nrows_w divides the scale block, so the rows share it

    float sumf[nrows_w][ncols_dst];
#pragma unroll
    for (int r = 0; r < nrows_w; ++r) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            sumf[r][j] = 0.0f;
        }
    }

    for (int k = warp*trip + lane*16; k < ncols; k += MMF8_GEMV_NWARPS*trip) {
        uint4 w4[nrows_w];
#pragma unroll
        for (int r = 0; r < nrows_w; ++r) {
            w4[r] = *(const uint4 *) (xr + (size_t) r*ncols + k);
        }
        const float s = sr[(k/GGML_F8_E4M3_SCALE_BLOCK)*nblk_n];

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
        for (int r = 0; r < nrows_w; ++r) {
            const uint16_t * w2 = (const uint16_t *) &w4[r];
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
                sumf[r][j] += s*part;
            }
        }
    }

#pragma unroll
    for (int r = 0; r < nrows_w; ++r) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            sumf[r][j] = warp_reduce_sum<warp_size>(sumf[r][j]);
            if (lane == 0) {
                red[warp][r][j] = sumf[r][j];
            }
        }
    }
    __syncthreads();

    if (warp == 0 && lane < nrows_w*ncols_dst) {
        const int r = lane / ncols_dst;
        const int j = lane % ncols_dst;
        float t = 0.0f;
#pragma unroll
        for (int w = 0; w < MMF8_GEMV_NWARPS; ++w) {
            t += red[w][r][j];
        }
        dst[(size_t) j*stride_col_dst + row0 + r] = t;
    }
}

template <int ncols_dst>
static void launch_mul_mat_vec_f8_e4m3(
        const uint8_t * x, const float * sx, const float * y, float * dst,
        const int ncols, const int nrows, const int nblk_n, const int stride_col_y, const int stride_col_dst, cudaStream_t stream) {
    // 4 rows per warp at batch 1, fewer when more activation columns have to stay in registers
    constexpr int nrows_w = ncols_dst == 1 ? 4 : (ncols_dst == 2 ? 2 : 1);
    GGML_ASSERT(nrows % nrows_w == 0);
    const dim3 block_dims(32, MMF8_GEMV_NWARPS, 1);
    const dim3 block_nums(nrows/nrows_w, 1, 1);
    const bool y_vec = ((uintptr_t) y % sizeof(float4) == 0) && stride_col_y % 4 == 0;
    if (y_vec) {
        mul_mat_vec_f8_e4m3<ncols_dst, nrows_w, true><<<block_nums, block_dims, 0, stream>>>(x, sx, y, dst, ncols, nblk_n, stride_col_y, stride_col_dst);
    } else {
        mul_mat_vec_f8_e4m3<ncols_dst, nrows_w, false><<<block_nums, block_dims, 0, stream>>>(x, sx, y, dst, ncols, nblk_n, stride_col_y, stride_col_dst);
    }
}

static void mul_mat_vec_f8_e4m3_cuda(
        const uint8_t * x, const float * sx, const float * y, float * dst,
        const int ncols, const int nrows, const int nblk_n, const int ncols_dst, const int stride_col_y, const int stride_col_dst, cudaStream_t stream) {
    switch (ncols_dst) {
        case 1: launch_mul_mat_vec_f8_e4m3<1>(x, sx, y, dst, ncols, nrows, nblk_n, stride_col_y, stride_col_dst, stream); break;
        case 2: launch_mul_mat_vec_f8_e4m3<2>(x, sx, y, dst, ncols, nrows, nblk_n, stride_col_y, stride_col_dst, stream); break;
        case 3: launch_mul_mat_vec_f8_e4m3<3>(x, sx, y, dst, ncols, nrows, nblk_n, stride_col_y, stride_col_dst, stream); break;
        case 4: launch_mul_mat_vec_f8_e4m3<4>(x, sx, y, dst, ncols, nrows, nblk_n, stride_col_y, stride_col_dst, stream); break;
        case 5: launch_mul_mat_vec_f8_e4m3<5>(x, sx, y, dst, ncols, nrows, nblk_n, stride_col_y, stride_col_dst, stream); break;
        case 6: launch_mul_mat_vec_f8_e4m3<6>(x, sx, y, dst, ncols, nrows, nblk_n, stride_col_y, stride_col_dst, stream); break;
        case 7: launch_mul_mat_vec_f8_e4m3<7>(x, sx, y, dst, ncols, nrows, nblk_n, stride_col_y, stride_col_dst, stream); break;
        case 8: launch_mul_mat_vec_f8_e4m3<8>(x, sx, y, dst, ncols, nrows, nblk_n, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("fatal error");
    }
}

// y[n*ncols + k] = e4m3(x[n*ncols + k]) * scale[(k/128)*nblk_n + n/128], 8 elements per thread
static __global__ void dequant_f8_e4m3_blockscaled_f16(
        const uint8_t * __restrict__ x, const float * __restrict__ sx, half * __restrict__ y, const int ncols, const int nblk_n) {
    const int row = blockIdx.y;
    const int k   = (blockIdx.x*blockDim.x + threadIdx.x)*8;
    if (k >= ncols) {
        return;
    }
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
    const int n8 = ncols/8;
    const dim3 block_nums((n8 + CUDA_DEQUANTIZE_BLOCK_SIZE - 1)/CUDA_DEQUANTIZE_BLOCK_SIZE, nrows, 1);
    dequant_f8_e4m3_blockscaled_f16<<<block_nums, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>(x, sx, y, ncols, nblk_n);
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
// one warp per (token, 128-k group), 4 floats per lane; scales stored [K/128][M_pad] (tokens contiguous) as CUTLASS expects
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
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    amax = warp_reduce_max<32>(amax);

    const float scale = fmaxf(amax/448.0f, FLT_MIN); // keeps 1/scale finite for zero or subnormal groups
    const float inv   = 1.0f/scale;
    const __nv_fp8x2_storage_t lo = __nv_cvt_float2_to_fp8x2(make_float2(v.x*inv, v.y*inv), __NV_SATFINITE, __NV_E4M3);
    const __nv_fp8x2_storage_t hi = __nv_cvt_float2_to_fp8x2(make_float2(v.z*inv, v.w*inv), __NV_SATFINITE, __NV_E4M3);
    *(uint32_t *) (q + i) = (uint32_t) lo | ((uint32_t) hi << 16);
    if (lane == 0) {
        sfa[(size_t) kb*stride_sfa + m] = scale;
    }
}

static void mul_mat_f8_e4m3_cutlass(
        ggml_backend_cuda_context & ctx, const uint8_t * x, const float * sx, const float * y, float * dst,
        const int ncols, const int nrows, const int64_t ntokens, cudaStream_t stream) {
    const int nblk_k = ncols/GGML_F8_E4M3_SCALE_BLOCK;

    // the activation scale tensor is [K/128][M] and TMA needs a 16-byte row stride, so M is padded to a multiple of 4
    const int64_t ntokens_pad = (ntokens + 3) & ~(int64_t) 3;

    ggml_cuda_pool_alloc<uint8_t> yq(ctx.pool(), (size_t) ntokens_pad*ncols);
    ggml_cuda_pool_alloc<float>   ys(ctx.pool(), (size_t) ntokens_pad*nblk_k);
    if (ntokens_pad != ntokens) {
        CUDA_CHECK(cudaMemsetAsync(yq.get() + (size_t) ntokens*ncols, 0, (size_t) (ntokens_pad - ntokens)*ncols, stream));
        CUDA_CHECK(cudaMemsetAsync(ys.get(), 0, (size_t) ntokens_pad*nblk_k*sizeof(float), stream));
    }
    {
        const int64_t npairs = ntokens*nblk_k;
        const dim3 block_nums((npairs + 7)/8, 1, 1);
        quantize_f8_e4m3_group128<<<block_nums, 256, 0, stream>>>(y, yq.get(), ys.get(), ntokens_pad, ntokens, ncols);
    }

    const size_t ws_size = ggml_cuda_mmf8_cutlass_workspace_size(ntokens_pad, nrows, ncols);
    ggml_cuda_pool_alloc<uint8_t> ws(ctx.pool(), ws_size > 0 ? ws_size : 16);
    if (ntokens_pad == ntokens) {
        ggml_cuda_mmf8_cutlass(yq.get(), ys.get(), x, sx, dst, ntokens, nrows, ncols, ws.get(), ws_size, stream);
        return;
    }
    ggml_cuda_pool_alloc<float> dst_pad(ctx.pool(), (size_t) ntokens_pad*nrows);
    ggml_cuda_mmf8_cutlass(yq.get(), ys.get(), x, sx, dst_pad.get(), ntokens_pad, nrows, ncols, ws.get(), ws_size, stream);
    CUDA_CHECK(cudaMemcpyAsync(dst, dst_pad.get(), (size_t) ntokens*nrows*sizeof(float), cudaMemcpyDeviceToDevice, stream));
}
#endif // GGML_CUDA_CUTLASS

void ggml_cuda_mul_mat_f8(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
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

    const int     ncols   = src0->ne[0];
    const int     nrows   = src0->ne[1];
    const int     nblk_n  = nrows/GGML_F8_E4M3_SCALE_BLOCK;
    const int64_t ntokens = ggml_nrows(src1);

    const uint8_t * x  = (const uint8_t *) src0->data;
    const float   * sx = (const float *)   src0_s->data;
    const float   * y  = (const float *)   src1->data;
    float         * d  = (float *)         dst->data;

    cudaStream_t stream = ctx.stream();

    if (ntokens <= MMF8_GEMV_MAX_NCOLS) {
        mul_mat_vec_f8_e4m3_cuda(x, sx, y, d, ncols, nrows, nblk_n, ntokens, ncols, nrows, stream);
        return;
    }

#ifdef GGML_CUDA_CUTLASS
    if (ggml_cuda_info().devices[ctx.device].cc == GGML_CUDA_CC_HOPPER && ((uintptr_t) y % 16 == 0)) {
        mul_mat_f8_e4m3_cutlass(ctx, x, sx, y, d, ncols, nrows, ntokens, stream);
        return;
    }
#endif // GGML_CUDA_CUTLASS

    mul_mat_f8_e4m3_cublas(ctx, x, sx, y, d, ncols, nrows, nblk_n, ntokens, stream);
}
