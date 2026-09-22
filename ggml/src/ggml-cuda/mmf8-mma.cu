#include "ggml.h"
#include "common.cuh"
#include "convert.cuh"
#include "cp-async.cuh"
#include "mma.cuh"
#include "mmf8.cuh"
#include "unary.cuh"

using namespace ggml_cuda_mma;

// Tensor-core decode matmul for the F8 route. The scalar GEMV pays one e4m3-to-float conversion and one FP32 FMA
// chain per activation column chunk, which at batch 8 is about 27 ms per decode step above the weight-traffic floor.
// Hopper has no FP8 MMA operand type, but every e4m3 value is exact in f16, so converting each weight byte once into
// an f16 A tile and accumulating in FP32 removes both costs and leaves the kernel bandwidth bound.
// A block owns 32 weight rows and its warps split k into slabs of 128 columns, which is exactly one weight scale
// block, so the scale is applied once per slab to the FP32 accumulator and no accumulator alignment is needed.
// Measured 2026-09-18: 64 rows per block halves the activation restaging but costs more than it saves (S_TG 352
// against 367 at batch 8, 182 against 215 at batch 4), because the row blocks are what fills the GPU.
// The weight path is double-buffered: each A tile is copied as raw e4m3 bytes into one of two shared slots with
// cp.async while the previous tile's conversion and mma run, so a load is always in flight.
#define MMF8_MMA_NWARPS 4
#define MMF8_MMA_ROWS   32
#define MMF8_MMA_KSLAB  GGML_F8_E4M3_SCALE_BLOCK        // scalar k per warp trip
#define MMF8_MMA_TILE_I 16                              // rows of one A tile
#define MMF8_MMA_KPAD   (MMF8_MMA_KSLAB/2 + 4)          // half2 per staged row, padded against bank conflicts
#define MMF8_MMA_SMEM   (MMF8_MMA_NWARPS*MMF8_MMA_TILE_I*MMF8_MMA_KPAD*(int) sizeof(half2))
// blocks to aim for; measured at batch 8: 1280 gives S_TG 359, 2560 gives 367, 5120 gives 363
#define MMF8_MMA_TARGET_BLOCKS 2560
#define MMF8_MMA_STAGES 2                                    // raw weight slots per warp: one in flight, one in use
#define MMF8_MMA_RAW    (MMF8_MMA_TILE_I*MMF8_MMA_KSLAB)      // bytes of one raw A tile: 16 rows of 128 columns
#define MMF8_MMA_RAWMEM (MMF8_MMA_NWARPS*MMF8_MMA_STAGES*MMF8_MMA_RAW)

// two packed e4m3 -> half2; e4m3 is exact in f16, so this is lossless and costs one instruction per two weights
static __device__ __forceinline__ half2 ggml_cuda_e4m3x2_to_half2(const uint16_t v) {
#if defined(FP8_AVAILABLE) && !defined(GGML_USE_HIP)
    const __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t) v, __NV_E4M3);
    return *reinterpret_cast<const half2 *>(&hr);
#else
    return __floats2half2_rn(ggml_cuda_e4m3_to_fp32((uint8_t) (v & 0xFF)), ggml_cuda_e4m3_to_fp32((uint8_t) (v >> 8)));
#endif // defined(FP8_AVAILABLE) && !defined(GGML_USE_HIP)
}

// Stages one raw A tile into a slot with cp.async, in the layout the conversion reads back: lane l owns row
// l/8 of each group of four rows and bytes 16*(l%8) of that row, which is the 16 bytes it converts itself.
static __device__ __forceinline__ void mmf8_mma_issue_tile(
        char * slot, const uint8_t * xw, const int row_off, const int ncols, const int k, const int lane) {
#pragma unroll
    for (int it = 0; it < MMF8_MMA_TILE_I/4; ++it) {
        const int          i   = it*4 + lane/8;
        const void *       src = xw + (size_t) (row_off + i)*ncols + k + 16*(lane % 8);
        const unsigned int dst = ggml_cuda_cvta_generic_to_shared(slot + (i*8 + (lane % 8))*16);
        cp_async_cg_16<128>(dst, src);
    }
    cp_async_commit_group();
}

// has_glu: xg/sxg are the gate matrix of a fused up/gate matmul; the activation is staged once for both and the
// epilogue writes silu(gate) * up, as the GEMV does
// split/ksplit: one A tile is 16 rows, so row parallelism alone leaves a narrow matrix with too few blocks to fill
// the GPU (N=5120 is 160 blocks against 132 SMs). Blocks of the same rows split k instead and add their partial
// sums into the output, which needs it pre-zeroed; ksplit == 1 stores and is what the fused up/gate path uses,
// since silu needs the complete sum
// PDL: the grid dependency is waited at entry and launch completion signalled before the reduction; no __restrict__
// on the parameters (PDL rule)
template <int ncols_dst, bool has_glu>
static __device__ __forceinline__ void mmf8_mma_block(
        const uint8_t * x, const float * sx, const uint8_t * xg, const float * sxg,
        const half * y, float * dst,
        const int ncols, const int nblk_n, const int stride_col_y, const int stride_col_dst, const int row0,
        const int split, const int ksplit, char * smem, char * raw) {
#ifdef TURING_MMA_AVAILABLE
    typedef tile<16, 8, half2> tile_A;
    typedef tile< 8, 8, half2> tile_B;
    typedef tile<16, 8, float> tile_C;

    constexpr int warp_size = 32;
    constexpr int nmat      = has_glu ? 2 : 1;
    constexpr int ntA       = MMF8_MMA_ROWS / tile_A::I;         // A tiles per block
    constexpr int nfrag     = (MMF8_MMA_KSLAB/2) / tile_A::J;    // mma steps per slab
    constexpr int kiw       = MMF8_MMA_NWARPS*MMF8_MMA_ROWS + 4; // stride of the cross-warp reduction buffer
    static_assert(tile_A::I == MMF8_MMA_TILE_I, "MMF8_MMA_SMEM assumes 16-row A tiles");
    static_assert(MMF8_MMA_ROWS % warp_size == 0, "the epilogue maps whole warps of output rows");
    static_assert(nmat*ncols_dst*kiw*sizeof(float) <= MMF8_MMA_SMEM, "the reduction buffer must fit the staging area");

    const int warp = threadIdx.y;
    const int lane = threadIdx.x;

    half2 * tile_xy = (half2 *) smem + warp*(tile_A::I*MMF8_MMA_KPAD);

    const uint8_t * xr  = x  + (size_t) row0*ncols;
    const float   * sr  = sx + row0/GGML_F8_E4M3_SCALE_BLOCK; // the rows of a block divide the scale block
    const uint8_t * xgr = has_glu ? xg  + (size_t) row0*ncols : xr;
    const float   * sgr = has_glu ? sxg + row0/GGML_F8_E4M3_SCALE_BLOCK : sr;

    float sums[nmat][ntA][tile_C::ne] = {};

    constexpr int ntiles = nmat*ntA;                                 // A tiles consumed per slab

    const int kstride  = ksplit*MMF8_MMA_NWARPS*MMF8_MMA_KSLAB;
    char *    raw_warp = raw + warp*(MMF8_MMA_STAGES*MMF8_MMA_RAW);
    int       k0       = (split*MMF8_MMA_NWARPS + warp)*MMF8_MMA_KSLAB;
    int       stage    = 0;

    // the first copy has to follow the grid dependency: PDL forbids touching producer memory before it is waited
    ggml_cuda_pdl_sync();
    if (k0 < ncols) {
        mmf8_mma_issue_tile(raw_warp, xr, 0, ncols, k0, lane);
    }
    for (; k0 < ncols; k0 += kstride) {
        // the activation slab: 8 columns of 128 k as half2, zero-filled above ncols_dst
        __syncwarp();
#pragma unroll
        for (int j0 = 0; j0 < tile_B::I; ++j0) {
            half2 a0 = __floats2half2_rn(0.0f, 0.0f);
            half2 a1 = a0;
            if (j0 < ncols_dst) {
                const half2 * yj = (const half2 *) (y + (size_t) j0*stride_col_y + k0) + 2*lane;
                a0 = yj[0];
                a1 = yj[1];
            }
            tile_xy[j0*MMF8_MMA_KPAD + 2*lane + 0] = a0;
            tile_xy[j0*MMF8_MMA_KPAD + 2*lane + 1] = a1;
        }
        __syncwarp();

        tile_B B[nfrag];
#pragma unroll
        for (int i = 0; i < nfrag; ++i) {
            load_ldmatrix(B[i], tile_xy + i*tile_B::J, MMF8_MMA_KPAD);
        }

#pragma unroll
        for (int g = 0; g < nmat; ++g) {
            const float s = (g == 0 ? sr : sgr)[(k0/GGML_F8_E4M3_SCALE_BLOCK)*nblk_n];
#pragma unroll
            for (int itA = 0; itA < ntA; ++itA) {
                // issue the next tile before consuming this one, so its copy overlaps the conversion and the mma;
                // the last tile of a slab prefetches the first tile of the next slab
                const int t  = g*ntA + itA;
                const int tn = (t + 1) % ntiles;
                const int kn = t + 1 == ntiles ? k0 + kstride : k0;
                if (kn < ncols) {
                    mmf8_mma_issue_tile(raw_warp + ((stage + 1) % MMF8_MMA_STAGES)*MMF8_MMA_RAW,
                                        tn < ntA ? xr : xgr, (tn % ntA)*tile_A::I, ncols, kn, lane);
                    cp_async_wait_group<1>();
                } else {
                    cp_async_wait_group<0>();   // nothing younger is in flight, so wait for this tile itself
                }
                const char * slot = raw_warp + stage*MMF8_MMA_RAW;

                __syncwarp();
                // every lane converts exactly the 16 bytes it copied, so only the cross-lane ldmatrix below
                // needs the warp barrier
#pragma unroll
                for (int it = 0; it < tile_A::I/4; ++it) {
                    const int      i = it*4 + lane/8;
                    const uint4    w = *(const uint4 *) (slot + (i*8 + (lane % 8))*16);
                    const uint32_t v[4] = {w.x, w.y, w.z, w.w};
                    half2 *        p = tile_xy + i*MMF8_MMA_KPAD + (lane % 8)*8;
#pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        p[2*u + 0] = ggml_cuda_e4m3x2_to_half2((uint16_t) (v[u] & 0xFFFF));
                        p[2*u + 1] = ggml_cuda_e4m3x2_to_half2((uint16_t) (v[u] >> 16));
                    }
                }
                __syncwarp();

                tile_C C;
#pragma unroll
                for (int i = 0; i < nfrag; ++i) {
                    tile_A A;
                    load_ldmatrix(A, tile_xy + i*tile_A::J, MMF8_MMA_KPAD);
                    mma(C, A, B[i]);
                }
#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sums[g][itA][l] += s*C.x[l];
                }
                stage = (stage + 1) % MMF8_MMA_STAGES;
            }
        }
    }
    ggml_cuda_pdl_lc();

    // the reduction buffer reuses the staging area, so every warp must be done with it first
    float * buf_iw = (float *) smem;
    __syncthreads();
#pragma unroll
    for (int g = 0; g < nmat; ++g) {
#pragma unroll
        for (int itA = 0; itA < ntA; ++itA) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                const int i = warp*MMF8_MMA_ROWS + itA*tile_C::I + tile_C::get_i(l);
                const int j = tile_C::get_j(l);
                buf_iw[(g*ncols_dst + j)*kiw + i] = sums[g][itA][l];
            }
        }
    }
    __syncthreads();

#pragma unroll
    for (int j0 = 0; j0 < ncols_dst; j0 += MMF8_MMA_NWARPS) {
        const int j = j0 + warp;
        if (j >= ncols_dst) {
            break;
        }
        // a block owns more rows than a warp has lanes, so each lane finishes one row per pass
#pragma unroll
        for (int r = 0; r < MMF8_MMA_ROWS/warp_size; ++r) {
            const int row = r*warp_size + lane;
            float     t[nmat];
#pragma unroll
            for (int g = 0; g < nmat; ++g) {
                t[g] = 0.0f;
#pragma unroll
                for (int w = 0; w < MMF8_MMA_NWARPS; ++w) {
                    t[g] += buf_iw[(g*ncols_dst + j)*kiw + w*MMF8_MMA_ROWS + row];
                }
            }
            if constexpr (has_glu) {
                dst[(size_t) j*stride_col_dst + row0 + row] = ggml_cuda_op_silu_single(t[1])*t[0]; // silu(gate) * up
            } else if (ksplit > 1) {
                atomicAdd(&dst[(size_t) j*stride_col_dst + row0 + row], t[0]);
            } else {
                dst[(size_t) j*stride_col_dst + row0 + row] = t[0];
            }
        }
    }
#else
    GGML_UNUSED_VARS(x, sx, xg, sxg, y, dst, ncols, nblk_n, stride_col_y, stride_col_dst, row0, split, ksplit,
                     smem, raw);
    NO_DEVICE_CODE;
#endif // TURING_MMA_AVAILABLE
}

template <int ncols_dst, bool has_glu>
static __global__ void mul_mat_f8_e4m3_mma(
        const uint8_t * x, const float * sx, const uint8_t * xg, const float * sxg,
        const half * y, float * dst,
        const int ncols, const int nblk_n, const int stride_col_y, const int stride_col_dst, const int ksplit) {
    __shared__ __align__(16) char smem[MMF8_MMA_SMEM];
    __shared__ __align__(16) char raw[MMF8_MMA_RAWMEM];
    mmf8_mma_block<ncols_dst, has_glu>(x, sx, xg, sxg, y, dst, ncols, nblk_n, stride_col_y, stride_col_dst,
                                       blockIdx.x*MMF8_MMA_ROWS, blockIdx.y, ksplit, smem, raw);
}

// up to three F8 matrices sharing one activation in one launch, with the same block mapping as the multi GEMV
template <int ncols_dst>
static __global__ void mul_mat_f8_e4m3_mma_multi(const mmf8_multi_args a, const half * y, const int ncols, const int stride_col_y, const int ksplit) {
    __shared__ __align__(16) char smem[MMF8_MMA_SMEM];
    __shared__ __align__(16) char raw[MMF8_MMA_RAWMEM];
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
    const int row0 = (blockIdx.x - b0)*MMF8_MMA_ROWS;
    mmf8_mma_block<ncols_dst, false>(x, sx, nullptr, nullptr, y, dst, ncols, nrows/GGML_F8_E4M3_SCALE_BLOCK,
                                     stride_col_y, nrows, row0, blockIdx.y, ksplit, smem, raw);
}

// How many blocks split k so the grid fills the GPU: one wave is 132 blocks on an H100 and the GEMV this replaces
// ran thousands, so the target is a few thousand warps. A split needs a whole slab per warp, and the fused up/gate
// kernel cannot split at all.
static int mmf8_mma_ksplit(const int nrows, const int ncols, const bool has_glu) {
    if (has_glu) {
        return 1;
    }
    const int row_blocks = nrows/MMF8_MMA_ROWS;
    const int max_split  = ncols/(MMF8_MMA_NWARPS*MMF8_MMA_KSLAB);
    const int want       = (MMF8_MMA_TARGET_BLOCKS + row_blocks - 1)/row_blocks;
    return std::max(1, std::min(want, std::max(1, max_split)));
}

template <int ncols_dst>
static void launch_mul_mat_f8_e4m3_mma(
        const uint8_t * x, const float * sx, const uint8_t * xg, const float * sxg, const half * y, float * dst,
        const int ncols, const int nrows, const int nblk_n, const int stride_col_y, const int stride_col_dst, cudaStream_t stream) {
    GGML_ASSERT(nrows % MMF8_MMA_ROWS == 0);
    GGML_ASSERT(ncols % MMF8_MMA_KSLAB == 0);
    const int  ksplit = mmf8_mma_ksplit(nrows, ncols, xg != nullptr);
    const dim3 block_dims(32, MMF8_MMA_NWARPS, 1);
    const dim3 block_nums(nrows/MMF8_MMA_ROWS, ksplit, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
    if (xg != nullptr) {
        ggml_cuda_kernel_launch(mul_mat_f8_e4m3_mma<ncols_dst, true>, launch_params,
                                x, sx, xg, sxg, y, dst, ncols, nblk_n, stride_col_y, stride_col_dst, ksplit);
        return;
    }
    if (ksplit > 1) {
        // the split blocks add into the output, so it starts at zero; the columns are contiguous here
        GGML_ASSERT(stride_col_dst == nrows);
        CUDA_CHECK(cudaMemsetAsync(dst, 0, (size_t) ncols_dst*nrows*sizeof(float), stream));
    }
    const uint8_t * no_xg  = nullptr;
    const float   * no_sxg = nullptr;
    ggml_cuda_kernel_launch(mul_mat_f8_e4m3_mma<ncols_dst, false>, launch_params,
                            x, sx, no_xg, no_sxg, y, dst, ncols, nblk_n, stride_col_y, stride_col_dst, ksplit);
}

// The activation is converted to f16 once per launch: the B tile is f16 either way, and every row block of every
// launch reads the whole activation, so this halves those bytes and drops one conversion per staged element.
// xg/sxg: the gate matrix of a fused up/gate matmul (nullptr for a plain one)
void mul_mat_f8_e4m3_mma_cuda(
        ggml_backend_cuda_context & ctx,
        const uint8_t * x, const float * sx, const uint8_t * xg, const float * sxg, const float * yf, float * dst,
        const int ncols, const int nrows, const int nblk_n, const int ncols_dst, const int stride_col_y, const int stride_col_dst, cudaStream_t stream) {
    GGML_ASSERT(stride_col_y == ncols);
    ggml_cuda_pool_alloc<half> y16(ctx.pool(), (size_t) ncols_dst*ncols);
    const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
    to_fp16(yf, y16.get(), (int64_t) ncols_dst*ncols, stream);
    const half * y = y16.get();
    switch (ncols_dst) {
        case 2: launch_mul_mat_f8_e4m3_mma<2>(x, sx, xg, sxg, y, dst, ncols, nrows, nblk_n, stride_col_y, stride_col_dst, stream); break;
        case 3: launch_mul_mat_f8_e4m3_mma<3>(x, sx, xg, sxg, y, dst, ncols, nrows, nblk_n, stride_col_y, stride_col_dst, stream); break;
        case 4: launch_mul_mat_f8_e4m3_mma<4>(x, sx, xg, sxg, y, dst, ncols, nrows, nblk_n, stride_col_y, stride_col_dst, stream); break;
        case 5: launch_mul_mat_f8_e4m3_mma<5>(x, sx, xg, sxg, y, dst, ncols, nrows, nblk_n, stride_col_y, stride_col_dst, stream); break;
        case 6: launch_mul_mat_f8_e4m3_mma<6>(x, sx, xg, sxg, y, dst, ncols, nrows, nblk_n, stride_col_y, stride_col_dst, stream); break;
        case 7: launch_mul_mat_f8_e4m3_mma<7>(x, sx, xg, sxg, y, dst, ncols, nrows, nblk_n, stride_col_y, stride_col_dst, stream); break;
        case 8: launch_mul_mat_f8_e4m3_mma<8>(x, sx, xg, sxg, y, dst, ncols, nrows, nblk_n, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("F8 tensor-core matmul: batch outside 2..8");
    }
}

template <int ncols_dst>
static void launch_mul_mat_f8_e4m3_mma_multi(const mmf8_multi_args & a, const half * y, const int ncols, const int stride_col_y, cudaStream_t stream) {
    mmf8_multi_args b = a;
    b.block0[0] = 0;
    for (int i = 0; i < a.n; ++i) {
        GGML_ASSERT(a.nrows[i] % MMF8_MMA_ROWS == 0);
        b.block0[i + 1] = b.block0[i] + a.nrows[i]/MMF8_MMA_ROWS;
    }
    GGML_ASSERT(ncols % MMF8_MMA_KSLAB == 0);
    // the split count is shared by every matrix in the launch, so it comes from the whole grid
    const int  ksplit = mmf8_mma_ksplit(b.block0[a.n]*MMF8_MMA_ROWS, ncols, false);
    const dim3 block_dims(32, MMF8_MMA_NWARPS, 1);
    const dim3 block_nums(b.block0[a.n], ksplit, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
    if (ksplit > 1) {
        for (int i = 0; i < a.n; ++i) {
            CUDA_CHECK(cudaMemsetAsync(a.dst[i], 0, (size_t) ncols_dst*a.nrows[i]*sizeof(float), stream));
        }
    }
    ggml_cuda_kernel_launch(mul_mat_f8_e4m3_mma_multi<ncols_dst>, launch_params, b, y, ncols, stride_col_y, ksplit);
}

void mul_mat_f8_e4m3_mma_multi_cuda(
        ggml_backend_cuda_context & ctx,
        const mmf8_multi_args & a, const float * yf, const int ncols, const int ncols_dst, const int stride_col_y, cudaStream_t stream) {
    GGML_ASSERT(stride_col_y == ncols);
    ggml_cuda_pool_alloc<half> y16(ctx.pool(), (size_t) ncols_dst*ncols);
    const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
    to_fp16(yf, y16.get(), (int64_t) ncols_dst*ncols, stream);
    const half * y = y16.get();
    switch (ncols_dst) {
        case 2: launch_mul_mat_f8_e4m3_mma_multi<2>(a, y, ncols, stride_col_y, stream); break;
        case 3: launch_mul_mat_f8_e4m3_mma_multi<3>(a, y, ncols, stride_col_y, stream); break;
        case 4: launch_mul_mat_f8_e4m3_mma_multi<4>(a, y, ncols, stride_col_y, stream); break;
        case 5: launch_mul_mat_f8_e4m3_mma_multi<5>(a, y, ncols, stride_col_y, stream); break;
        case 6: launch_mul_mat_f8_e4m3_mma_multi<6>(a, y, ncols, stride_col_y, stream); break;
        case 7: launch_mul_mat_f8_e4m3_mma_multi<7>(a, y, ncols, stride_col_y, stream); break;
        case 8: launch_mul_mat_f8_e4m3_mma_multi<8>(a, y, ncols, stride_col_y, stream); break;
        default: GGML_ABORT("F8 tensor-core matmul: batch outside 2..8");
    }
}

// GGML_CUDA_MMF8_MMA_MIN: smallest batch that takes the tensor-core matmul instead of the scalar GEMV; the cap + 1
// disables it. Batch 1 stays on the GEMV, which already streams the weights at 80% of peak, and an 8-wide B tile
// would waste seven of its columns there.
int ggml_cuda_mmf8_mma_min_ncols() {
    static const int v = []() {
        const char * s = getenv("GGML_CUDA_MMF8_MMA_MIN");
        const int    n = s ? atoi(s) : GGML_CUDA_MMF8_MMA_MIN_DEFAULT;
        return n < 2 ? 2 : (n > GGML_CUDA_MMF8_GEMV_MAX_NCOLS + 1 ? GGML_CUDA_MMF8_GEMV_MAX_NCOLS + 1 : n);
    }();
    return v;
}

bool ggml_cuda_mmf8_use_mma(ggml_backend_cuda_context & ctx, const int64_t ntokens) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    return ntokens >= ggml_cuda_mmf8_mma_min_ncols() && GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING;
}
