#include "w4a4.cuh"

// From mmq.cuh, only used by GGML_CUDA_W4A4_CHECK=2 (including the header here would pull in all MMQ templates).
void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);

#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Activations are quantized in chunks of W4A4_CHUNK_K values (W4A4_NBLOCKS q4_0 blocks). For each chunk and column the
// quantized buffer holds 32 ints of packed nibbles followed by 8 float scales, columns of the same chunk are
// contiguous so that one tile of columns can be copied to shared memory with plain 16 byte loads.
#ifndef W4A4_CHUNK_K
#define W4A4_CHUNK_K   256                             // K values per shared memory tile, 128 or 256
#endif
#define W4A4_NBLOCKS   (W4A4_CHUNK_K / QK4_0)          // q4_0 blocks per chunk
#define W4A4_Y_INTS    (W4A4_NBLOCKS*4 + W4A4_NBLOCKS) // 32 nibble ints + 8 scales per column and chunk
#define W4A4_I         128                             // weight rows per CUDA block
#define W4A4_NWARPS    8
#define W4A4_NTHREADS  (W4A4_NWARPS*32)
#define W4A4_X_STRIDE  (W4A4_NBLOCKS*4)                // shared memory row stride in ints, XOR swizzled against bank conflicts

static_assert(W4A4_I == 128, "the thread mapping assumes 128 rows per tile");

// Tuning switches (compile time).
#ifndef W4A4_PREFETCH
#define W4A4_PREFETCH 0    // load the next chunk into registers while the current one is processed (slower on gfx1151)
#endif
#ifndef W4A4_INTERLEAVE
#define W4A4_INTERLEAVE 1  // breadth-first dot8 chains over the 8 rows of a thread
#endif
#ifndef W4A4_LDS_EXTRA
#define W4A4_LDS_EXTRA 0   // extra shared memory in bytes per block, to cap the number of blocks per WGP
#endif

// Dot product of 8 signed 4 bit values packed into a and b.
static __device__ __forceinline__ int w4a4_dot8(const int a, const int b, const int c) {
#if defined(GGML_USE_HIP) && (defined(RDNA3) || defined(RDNA4))
    return __builtin_amdgcn_sudot8(true, a, true, b, c, false);
#else
    int s = c;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        s += ((a << (28 - 4*k)) >> 28) * ((b << (28 - 4*k)) >> 28);
    }
    return s;
#endif // defined(GGML_USE_HIP) && (defined(RDNA3) || defined(RDNA4))
}

// Position of int k of q4_0 block b of row i in the shared memory weight tile. The 16 byte groups of a row are
// permuted by the row index so that the 16 rows read by a half-warp start in different banks.
static __device__ __forceinline__ int w4a4_x_idx(const int i, const int b, const int k) {
    constexpr int rows_per_bank_span = 32 / W4A4_X_STRIDE; // rows sharing one pass over the 32 banks
    return i*W4A4_X_STRIDE + 4*(b ^ ((i / rows_per_bank_span) % W4A4_NBLOCKS)) + k;
}

// The weight tile of one chunk is loaded as pairs of q4_0 blocks: 36 bytes = 9 aligned ints per pair (K % 256 == 0
// keeps the rows 4 byte aligned). 128 rows x 4 pairs = 512 pairs, 2 per thread for 256 K values per chunk.
#define W4A4_X_PAIRS_PER_THREAD (W4A4_I*W4A4_NBLOCKS/2/W4A4_NTHREADS)

static __device__ __forceinline__ void w4a4_load_x(
        const char * __restrict__ x, int (&xr)[W4A4_X_PAIRS_PER_THREAD][9], const int row0, const int nrows_x,
        const int64_t stride_row_x, const int c, const int tid) {
#pragma unroll
    for (int n = 0; n < W4A4_X_PAIRS_PER_THREAD; ++n) {
        const int t    = n*W4A4_NTHREADS + tid;
        const int i    = t / (W4A4_NBLOCKS/2);
        const int p    = t % (W4A4_NBLOCKS/2);
        const int irow = min(row0 + i, nrows_x - 1);
        const int * src = (const int *) (x + irow*stride_row_x + ((int64_t) c*W4A4_NBLOCKS + 2*p)*sizeof(block_q4_0));
#pragma unroll
        for (int w = 0; w < 9; ++w) {
            xr[n][w] = src[w];
        }
    }
}

static __device__ __forceinline__ void w4a4_store_x(
        const int (&xr)[W4A4_X_PAIRS_PER_THREAD][9], int * __restrict__ x_qs, float * __restrict__ x_d, const int tid) {
#pragma unroll
    for (int n = 0; n < W4A4_X_PAIRS_PER_THREAD; ++n) {
        const int t  = n*W4A4_NTHREADS + tid;
        const int i  = t / (W4A4_NBLOCKS/2);
        const int b0 = 2*(t % (W4A4_NBLOCKS/2));
        const int b1 = b0 + 1;

        // block b0: d in bytes 0-1, qs in bytes 2-17; block b1: d in bytes 18-19, qs in bytes 20-35
        x_d[b0*W4A4_I + i] = __half2float(__ushort_as_half((unsigned short) (xr[n][0] & 0xFFFF)));
        x_d[b1*W4A4_I + i] = __half2float(__ushort_as_half((unsigned short) ((unsigned int) xr[n][4] >> 16)));
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int q0 = (int) (((unsigned int) xr[n][k] >> 16) | ((unsigned int) xr[n][k + 1] << 16));
            x_qs[w4a4_x_idx(i, b0, k)] = q0 ^ 0x88888888;
            x_qs[w4a4_x_idx(i, b1, k)] = xr[n][5 + k] ^ 0x88888888;
        }
    }
}

// Same rule as quantize_row_q4_0_ref: the value with the largest magnitude maps to -8 (ties prefer the larger value
// so that all lanes of a block agree).
static __device__ __forceinline__ void w4a4_amax_update(float & amax, float & vmax, const float a, const float v) {
    if (a > amax || (a == amax && v > vmax)) {
        amax = a;
        vmax = v;
    }
}

static __device__ __forceinline__ int w4a4_quantize_q(const float v, const float id) {
    return min(15, (int) (v*id + 8.5f));
}

// One warp per (column, 256 values). Lane l = 4*b + k produces int k of block b: the low nibbles hold the values 4k..4k+3
// of the block and the high nibbles the values 16+4k..16+4k+3, the same order as block_q4_0::qs. The nibbles are
// stored as signed values q - 8 (q ^ 8) so both operands of v_dot8_i32_iu4 are signed.
static __global__ void quantize_w4a4(
        const float * __restrict__ x, int * __restrict__ y, const int64_t s11, const int ncols, const int ncols_pad) {
    const int lane = threadIdx.x;
    const int j    = blockIdx.x*blockDim.y + threadIdx.y;
    const int gb   = blockIdx.y*8 + lane/4;     // q4_0 block index within the column
    const int c    = gb / W4A4_NBLOCKS;         // chunk
    const int b    = gb % W4A4_NBLOCKS;         // block within the chunk
    const int k    = lane % 4;

    if (j >= ncols_pad) {
        return;
    }

    int * yc = y + ((int64_t) c*ncols_pad + j)*W4A4_Y_INTS;

    if (j >= ncols) {
        yc[4*b + k] = 0;
        if (k == 0) {
            yc[W4A4_NBLOCKS*4 + b] = 0;
        }
        return;
    }

    const float * xb = x + j*s11 + gb*QK4_0;
    const float4 lo = *((const float4 *) (xb +      4*k));
    const float4 hi = *((const float4 *) (xb + 16 + 4*k));
    const float v[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};

    float amax = 0.0f;
    float vmax = 0.0f;
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        w4a4_amax_update(amax, vmax, fabsf(v[l]), v[l]);
    }
#pragma unroll
    for (int offset = 1; offset < 4; offset <<= 1) {
        const float oa = __shfl_xor_sync(0xffffffff, amax, offset, 32);
        const float ov = __shfl_xor_sync(0xffffffff, vmax, offset, 32);
        w4a4_amax_update(amax, vmax, oa, ov);
    }

    const float d  = vmax / -8.0f;
    const float id = d != 0.0f ? 1.0f/d : 0.0f;

    int packed = 0;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        packed |= (w4a4_quantize_q(v[l + 0], id) ^ 8) << (8*l + 0);
        packed |= (w4a4_quantize_q(v[l + 4], id) ^ 8) << (8*l + 4);
    }
    yc[4*b + k] = packed;
    if (k == 0) {
        yc[W4A4_NBLOCKS*4 + b] = __float_as_int(__half2float(__float2half(d)));
    }
}

// Each thread computes 8 rows x J/16 columns: rows (lane % 16) + 16*a, columns 2*warp + lane/16 + 16*jb.
template <int J>
__launch_bounds__(W4A4_NTHREADS, 1)
static __global__ void mul_mat_w4a4(
        const char * __restrict__ x, const int * __restrict__ y, float * __restrict__ dst,
        const int nrows_x, const int64_t stride_row_x, const int ncols, const int ncols_pad, const int nchunks,
        const int64_t stride_col_dst) {
    constexpr int RJ = J/16;
    static_assert(J % 16 == 0 && J <= 128, "bad J");

    __shared__ __align__(16) int   x_qs[W4A4_I*W4A4_X_STRIDE];
    __shared__ __align__(16) float x_d[W4A4_NBLOCKS*W4A4_I];
    __shared__ __align__(16) int   y_tile[J*W4A4_Y_INTS];
#if W4A4_LDS_EXTRA > 0
    __shared__ int lds_extra[W4A4_LDS_EXTRA/4];
    if (blockIdx.x == 0xFFFFFFFF) { lds_extra[threadIdx.x] = 0; dst[0] = lds_extra[threadIdx.y]; }
#endif

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid  = warp*32 + lane;

    const int row0 = blockIdx.x*W4A4_I;
    const int col0 = blockIdx.y*J;

    float sum[8][RJ];
#pragma unroll
    for (int a = 0; a < 8; ++a) {
#pragma unroll
        for (int jb = 0; jb < RJ; ++jb) {
            sum[a][jb] = 0.0f;
        }
    }

    constexpr int n_int4 = J*W4A4_Y_INTS/4;
    constexpr int NY     = (n_int4 + W4A4_NTHREADS - 1)/W4A4_NTHREADS;

    int  xr[W4A4_X_PAIRS_PER_THREAD][9];
    int4 yr[NY];

    auto load_y = [&](const int c) {
        const int4 * src = (const int4 *) (y + ((int64_t) c*ncols_pad + col0)*W4A4_Y_INTS);
#pragma unroll
        for (int n = 0; n < NY; ++n) {
            const int idx = n*W4A4_NTHREADS + tid;
            if (n_int4 % W4A4_NTHREADS == 0 || idx < n_int4) {
                yr[n] = src[idx];
            }
        }
    };

#if W4A4_PREFETCH
    w4a4_load_x(x, xr, row0, nrows_x, stride_row_x, 0, tid);
    load_y(0);
#endif

    for (int c = 0; c < nchunks; ++c) {
#if !W4A4_PREFETCH
        w4a4_load_x(x, xr, row0, nrows_x, stride_row_x, c, tid);
        load_y(c);
#endif
        w4a4_store_x(xr, x_qs, x_d, tid);
#pragma unroll
        for (int n = 0; n < NY; ++n) {
            const int idx = n*W4A4_NTHREADS + tid;
            if (n_int4 % W4A4_NTHREADS == 0 || idx < n_int4) {
                ((int4 *) y_tile)[idx] = yr[n];
            }
        }

        __syncthreads();

#if W4A4_PREFETCH
        // Prefetch the next chunk into registers while this one is being processed.
        if (c + 1 < nchunks) {
            w4a4_load_x(x, xr, row0, nrows_x, stride_row_x, c + 1, tid);
            load_y(c + 1);
        }
#endif

#pragma unroll 1
        for (int b = 0; b < W4A4_NBLOCKS; ++b) {
            int4  xq[8];
            float dx[8];
#pragma unroll
            for (int a = 0; a < 8; ++a) {
                const int i = (lane % 16) + 16*a;
                xq[a] = *((const int4 *) &x_qs[w4a4_x_idx(i, b, 0)]);
                dx[a] = x_d[b*W4A4_I + i];
            }
#pragma unroll
            for (int jb = 0; jb < RJ; ++jb) {
                const int j = 2*warp + lane/16 + 16*jb;
                const int4  yq = *((const int4 *) &y_tile[j*W4A4_Y_INTS + 4*b]);
                const float dy = __int_as_float(y_tile[j*W4A4_Y_INTS + W4A4_NBLOCKS*4 + b]);
#if W4A4_INTERLEAVE
                // Breadth-first over the 8 rows so that the dependent v_dot8 chains interleave.
                int isum[8];
#pragma unroll
                for (int a = 0; a < 8; ++a) {
                    isum[a] = w4a4_dot8(xq[a].x, yq.x, 0);
                }
#pragma unroll
                for (int a = 0; a < 8; ++a) {
                    isum[a] = w4a4_dot8(xq[a].y, yq.y, isum[a]);
                }
#pragma unroll
                for (int a = 0; a < 8; ++a) {
                    isum[a] = w4a4_dot8(xq[a].z, yq.z, isum[a]);
                }
#pragma unroll
                for (int a = 0; a < 8; ++a) {
                    isum[a] = w4a4_dot8(xq[a].w, yq.w, isum[a]);
                }
#pragma unroll
                for (int a = 0; a < 8; ++a) {
                    sum[a][jb] += (dx[a]*dy) * (float) isum[a];
                }
#else
#pragma unroll
                for (int a = 0; a < 8; ++a) {
                    int is = w4a4_dot8(xq[a].x, yq.x, 0);
                    is     = w4a4_dot8(xq[a].y, yq.y, is);
                    is     = w4a4_dot8(xq[a].z, yq.z, is);
                    is     = w4a4_dot8(xq[a].w, yq.w, is);
                    sum[a][jb] += (dx[a]*dy) * (float) is;
                }
#endif // W4A4_INTERLEAVE
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int jb = 0; jb < RJ; ++jb) {
        const int j = col0 + 2*warp + lane/16 + 16*jb;
        if (j >= ncols) {
            continue;
        }
#pragma unroll
        for (int a = 0; a < 8; ++a) {
            const int i = row0 + (lane % 16) + 16*a;
            if (i < nrows_x) {
                dst[j*stride_col_dst + i] = sum[a][jb];
            }
        }
    }
}

// Variant with V_WMMA_I32_16X16X16_IU4 instead of v_dot8_i32_iu4 (RDNA3 only, GGML_CUDA_W4A4=2).
// The activations are matrix A (M = columns of src1/dst) and the weights matrix B (N = rows of src0/dst). On RDNA3 wave32
// lane l holds row/column l % 16 of A/B with all 16 K values (16 nibbles = 2 ints), lanes 16-31 duplicate lanes 0-15,
// and element r of D in lane l is (M = 2*r + l/16, N = l % 16). The order of the K values within the 16 nibbles does not
// matter as long as A and B agree, so ints 0-1 and 2-3 of a q4_0 block are fed as they are to two WMMAs that share one
// accumulator. The accumulator starts at the bit pattern of 1.5*2^23 so that converting the integer block sum to float is
// a single exact subtraction (|sum| <= 2048).
#define W4A4W_Y_STRIDE (W4A4_NBLOCKS*4 + 4) // shared memory column stride of the activation nibbles in ints, padded
#define W4A4W_MAGIC    0x4B400000            // 1.5*2^23

// Position of the scale of q4_0 block b of tile column j: the 8 columns 2*r + h (r = 0..7) of each 16 column fragment are
// contiguous so that a lane reads the scales of its 8 accumulator columns with two 16 byte loads.
template <int J>
static __device__ __forceinline__ int w4a4w_yd_idx(const int b, const int j) {
    return (b*(J/16) + j/16)*16 + (j % 2)*8 + (j % 16)/2;
}

template <int J>
__launch_bounds__(W4A4_NTHREADS, 1)
static __global__ void mul_mat_w4a4_wmma(
        const char * __restrict__ x, const int * __restrict__ y, float * __restrict__ dst,
        const int nrows_x, const int64_t stride_row_x, const int ncols, const int ncols_pad, const int nchunks,
        const int64_t stride_col_dst, const int cols_first) {
#if defined(GGML_USE_HIP) && defined(RDNA3)
    typedef int v2i __attribute__((ext_vector_type(2)));
    typedef int v8i __attribute__((ext_vector_type(8)));

    constexpr int AF  = J/16;                                       // 16 column fragments per tile
    constexpr int WR  = AF % 4 == 0 ? 4 : (AF % 2 == 0 ? 2 : 1);    // 16 row fragments per warp
    constexpr int AC  = AF/WR;                                      // 16 column fragments per warp
    constexpr int NWR = W4A4_NWARPS/WR;                             // warps along the rows
    static_assert(J % 16 == 0 && J <= 128, "bad J");
    static_assert(NWR*WR*16 == W4A4_I && (AF/AC)*NWR == W4A4_NWARPS, "bad warp tiling");

    __shared__ __align__(16) int   x_qs[W4A4_I*W4A4_X_STRIDE];
    __shared__ __align__(16) float x_d[W4A4_NBLOCKS*W4A4_I];
    __shared__ __align__(16) int   y_qs[J*W4A4W_Y_STRIDE];
    __shared__ __align__(16) float y_d[W4A4_NBLOCKS*J];

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid  = warp*32 + lane;

    const int row0 = (cols_first ? blockIdx.y : blockIdx.x)*W4A4_I;
    const int col0 = (cols_first ? blockIdx.x : blockIdx.y)*J;

    const int iw = 16*WR*(warp % NWR); // first row of the warp within the tile
    const int jw = 16*AC*(warp / NWR); // first column of the warp within the tile

    float sum[AC][WR][8];
#pragma unroll
    for (int g = 0; g < AC; ++g) {
#pragma unroll
        for (int f = 0; f < WR; ++f) {
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                sum[g][f][r] = 0.0f;
            }
        }
    }

    v8i magic;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        magic[r] = W4A4W_MAGIC;
    }

    constexpr int y_int4_per_col = W4A4_Y_INTS/4;
    constexpr int n_int4         = J*y_int4_per_col;
    constexpr int NY             = (n_int4 + W4A4_NTHREADS - 1)/W4A4_NTHREADS;
    static_assert(W4A4_NBLOCKS % 4 == 0, "the scales of a column must fill whole int4");

    int  xr[W4A4_X_PAIRS_PER_THREAD][9];
    int4 yr[NY];

    for (int c = 0; c < nchunks; ++c) {
        w4a4_load_x(x, xr, row0, nrows_x, stride_row_x, c, tid);
        {
            const int4 * src = (const int4 *) (y + ((int64_t) c*ncols_pad + col0)*W4A4_Y_INTS);
#pragma unroll
            for (int n = 0; n < NY; ++n) {
                const int idx = n*W4A4_NTHREADS + tid;
                if (n_int4 % W4A4_NTHREADS == 0 || idx < n_int4) {
                    yr[n] = src[idx];
                }
            }
        }

        w4a4_store_x(xr, x_qs, x_d, tid);
#pragma unroll
        for (int n = 0; n < NY; ++n) {
            const int idx = n*W4A4_NTHREADS + tid;
            if (n_int4 % W4A4_NTHREADS == 0 || idx < n_int4) {
                const int j = idx / y_int4_per_col;
                const int q = idx % y_int4_per_col;
                if (q < W4A4_NBLOCKS) {
                    *((int4 *) &y_qs[j*W4A4W_Y_STRIDE + 4*q]) = yr[n];
                } else {
                    const int b0 = 4*(q - W4A4_NBLOCKS);
                    y_d[w4a4w_yd_idx<J>(b0 + 0, j)] = __int_as_float(yr[n].x);
                    y_d[w4a4w_yd_idx<J>(b0 + 1, j)] = __int_as_float(yr[n].y);
                    y_d[w4a4w_yd_idx<J>(b0 + 2, j)] = __int_as_float(yr[n].z);
                    y_d[w4a4w_yd_idx<J>(b0 + 3, j)] = __int_as_float(yr[n].w);
                }
            }
        }

        __syncthreads();

#pragma unroll 1
        for (int b = 0; b < W4A4_NBLOCKS; ++b) {
            // The swizzle of w4a4_x_idx is the same for rows that differ by a multiple of 16.
            const int * xb = &x_qs[w4a4_x_idx(iw + lane % 16, b, 0)];
            int4  xq[WR];
            float dx[WR];
#pragma unroll
            for (int f = 0; f < WR; ++f) {
                xq[f] = *((const int4 *) &xb[16*f*W4A4_X_STRIDE]);
                dx[f] = x_d[b*W4A4_I + iw + 16*f + lane % 16];
            }
            int4  yq[AC];
            float dy[AC][8];
#pragma unroll
            for (int g = 0; g < AC; ++g) {
                const int j = jw + 16*g + lane % 16;
                yq[g] = *((const int4 *) &y_qs[j*W4A4W_Y_STRIDE + 4*b]);
                const float4 * yd = (const float4 *) &y_d[w4a4w_yd_idx<J>(b, jw + 16*g + lane/16)];
                const float4 d0 = yd[0];
                const float4 d1 = yd[1];
                dy[g][0] = d0.x; dy[g][1] = d0.y; dy[g][2] = d0.z; dy[g][3] = d0.w;
                dy[g][4] = d1.x; dy[g][5] = d1.y; dy[g][6] = d1.z; dy[g][7] = d1.w;
            }

            v8i acc[AC][WR];
#pragma unroll
            for (int g = 0; g < AC; ++g) {
#pragma unroll
                for (int f = 0; f < WR; ++f) {
                    const v2i a = {yq[g].x, yq[g].y};
                    const v2i w = {xq[f].x, xq[f].y};
                    acc[g][f] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, w, magic, false);
                }
            }
#pragma unroll
            for (int g = 0; g < AC; ++g) {
#pragma unroll
                for (int f = 0; f < WR; ++f) {
                    const v2i a = {yq[g].z, yq[g].w};
                    const v2i w = {xq[f].z, xq[f].w};
                    acc[g][f] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, w, acc[g][f], false);
                }
            }
#pragma unroll
            for (int g = 0; g < AC; ++g) {
#pragma unroll
                for (int f = 0; f < WR; ++f) {
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        const float is = __int_as_float(acc[g][f][r]) - 12582912.0f;
                        sum[g][f][r] += (dx[f]*dy[g][r]) * is;
                    }
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int g = 0; g < AC; ++g) {
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int j = col0 + jw + 16*g + 2*r + lane/16;
            if (j >= ncols) {
                continue;
            }
#pragma unroll
            for (int f = 0; f < WR; ++f) {
                const int i = row0 + iw + 16*f + lane % 16;
                if (i < nrows_x) {
                    dst[j*stride_col_dst + i] = sum[g][f][r];
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(x, y, dst, nrows_x, stride_row_x, ncols, ncols_pad, nchunks, stride_col_dst, cols_first);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(RDNA3)
}

// Naive reference with the same numerics, one thread per output. Only used by GGML_CUDA_W4A4_CHECK.
static __global__ void mul_mat_w4a4_ref(
        const char * __restrict__ x, const float * __restrict__ yf, float * __restrict__ dst,
        const int nrows_x, const int64_t stride_row_x, const int ne00, const int64_t s11, const int64_t stride_col_dst) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y;
    if (i >= nrows_x) {
        return;
    }

    const block_q4_0 * bx = (const block_q4_0 *) (x + i*stride_row_x);
    float sum = 0.0f;
    for (int b = 0; b < ne00/QK4_0; ++b) {
        const float * v = yf + j*s11 + b*QK4_0;
        float amax = 0.0f;
        float vmax = 0.0f;
        for (int t = 0; t < QK4_0; ++t) {
            w4a4_amax_update(amax, vmax, fabsf(v[t]), v[t]);
        }
        const float d  = vmax / -8.0f;
        const float id = d != 0.0f ? 1.0f/d : 0.0f;

        int s = 0;
        for (int t = 0; t < QK4_0/2; ++t) {
            const int wlo = (bx[b].qs[t] & 0x0F) - 8;
            const int whi = (bx[b].qs[t] >>   4) - 8;
            const int alo = w4a4_quantize_q(v[t + 0],       id) - 8;
            const int ahi = w4a4_quantize_q(v[t + QK4_0/2], id) - 8;
            s += wlo*alo + whi*ahi;
        }
        const float dx = __half2float(bx[b].d);
        const float dy = __half2float(__float2half(d));
        sum += (dx*dy) * (float) s;
    }
    dst[j*stride_col_dst + i] = sum;
}

static int w4a4_env_int(const char * name, const int def) {
    const char * env = getenv(name);
    return env != nullptr ? atoi(env) : def;
}

// 0: off, 1: v_dot8_i32_iu4, 2: V_WMMA_I32_16X16X16_IU4 (RDNA3 only).
static int w4a4_mode() {
    static const int mode = w4a4_env_int("GGML_CUDA_W4A4", 0);
    return mode;
}

bool ggml_cuda_should_use_w4a4(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, const int cc) {
    const int mode = w4a4_mode();
    static const int64_t min_batch = [] {
        const char * env = getenv("GGML_CUDA_W4A4_MIN_BATCH");
        return env ? (int64_t) atoll(env) : (int64_t) 16;
    }();

    // Comma separated substrings of weight names that keep the default path, e.g. "ffn_down,attn_output".
    static const std::vector<std::string> exclude = [] {
        std::vector<std::string> ret;
        const char * env = getenv("GGML_CUDA_W4A4_EXCLUDE");
        std::string s = env ? env : "";
        size_t pos = 0;
        while (pos <= s.size()) {
            const size_t end = s.find(',', pos);
            const std::string item = s.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
            if (!item.empty()) {
                ret.push_back(item);
            }
            if (end == std::string::npos) {
                break;
            }
            pos = end + 1;
        }
        return ret;
    }();

    if (!((mode == 1 && (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc))) || (mode == 2 && GGML_CUDA_CC_IS_RDNA3(cc)))) {
        return false;
    }

    for (const std::string & item : exclude) {
        if (strstr(src0->name, item.c_str()) != nullptr) {
            return false;
        }
    }

    const int64_t ncols = src1->ne[1]*src1->ne[2]*src1->ne[3];

    return src0->type == GGML_TYPE_Q4_0 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        src0->ne[2] == 1 && src0->ne[3] == 1 && src0->ne[0] % 256 == 0 &&
        src0->nb[0] == ggml_type_size(GGML_TYPE_Q4_0) && src0->nb[1] % 4 == 0 && (uintptr_t) src0->data % 4 == 0 &&
        ggml_is_contiguous(src1) && ggml_is_contiguous(dst) && (uintptr_t) src1->data % 16 == 0 &&
        ncols >= min_batch && ncols <= INT_MAX/4 && src0->ne[1] <= INT_MAX;
}

template <int J>
static void launch_mul_mat_w4a4(
        const char * x, const int * y, float * dst, const int nrows_x, const int64_t stride_row_x, const int ncols,
        const int ncols_pad, const int nchunks, const int64_t stride_col_dst, cudaStream_t stream) {
    const dim3 block_dims(32, W4A4_NWARPS, 1);
    if (w4a4_mode() == 2) {
        // Consecutive blocks share the weight tile (GGML_CUDA_W4A4_COLS_FIRST=0: share the activation tile instead).
        static const int cols_first = w4a4_env_int("GGML_CUDA_W4A4_COLS_FIRST", 1);
        const int nr = (nrows_x + W4A4_I - 1)/W4A4_I;
        const int nc = (ncols + J - 1)/J;
        const dim3 block_nums(cols_first ? nc : nr, cols_first ? nr : nc, 1);
        mul_mat_w4a4_wmma<J><<<block_nums, block_dims, 0, stream>>>(
            x, y, dst, nrows_x, stride_row_x, ncols, ncols_pad, nchunks, stride_col_dst, cols_first);
        return;
    }
    const dim3 block_nums((nrows_x + W4A4_I - 1)/W4A4_I, (ncols + J - 1)/J, 1);
    mul_mat_w4a4<J><<<block_nums, block_dims, 0, stream>>>(
        x, y, dst, nrows_x, stride_row_x, ncols, ncols_pad, nchunks, stride_col_dst);
}

void ggml_cuda_mul_mat_w4a4(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    cudaStream_t stream = ctx.stream();

    const int     ne00    = src0->ne[0];
    const int     nrows_x = src0->ne[1];
    const int64_t s01     = src0->nb[1];
    const int     ncols   = src1->ne[1]*src1->ne[2]*src1->ne[3];
    const int64_t s11     = src1->nb[1] / sizeof(float);
    const int64_t s1      = dst->nb[1]  / sizeof(float);
    const int     nchunks = ne00 / W4A4_CHUNK_K;

    // Use as few column tiles as possible, then the narrowest tile width that still covers all columns.
    const int ntiles    = (ncols + 127) / 128;
    const int J         = GGML_PAD((ncols + ntiles - 1) / ntiles, 16);
    const int ncols_pad = ntiles*J;

    ggml_cuda_pool_alloc<int> src1_q(ctx.pool(), (size_t) nchunks*ncols_pad*W4A4_Y_INTS);
    {
        const dim3 block_dims(32, 8, 1);
        const dim3 block_nums((ncols_pad + 7)/8, ne00/256, 1);
        quantize_w4a4<<<block_nums, block_dims, 0, stream>>>((const float *) src1->data, src1_q.get(), s11, ncols, ncols_pad);
        CUDA_CHECK(cudaGetLastError());
    }

    const char * x = (const char *) src0->data;
    float * dst_d  = (float *) dst->data;

    switch (J) {
        case  16: launch_mul_mat_w4a4< 16>(x, src1_q.get(), dst_d, nrows_x, s01, ncols, ncols_pad, nchunks, s1, stream); break;
        case  32: launch_mul_mat_w4a4< 32>(x, src1_q.get(), dst_d, nrows_x, s01, ncols, ncols_pad, nchunks, s1, stream); break;
        case  48: launch_mul_mat_w4a4< 48>(x, src1_q.get(), dst_d, nrows_x, s01, ncols, ncols_pad, nchunks, s1, stream); break;
        case  64: launch_mul_mat_w4a4< 64>(x, src1_q.get(), dst_d, nrows_x, s01, ncols, ncols_pad, nchunks, s1, stream); break;
        case  80: launch_mul_mat_w4a4< 80>(x, src1_q.get(), dst_d, nrows_x, s01, ncols, ncols_pad, nchunks, s1, stream); break;
        case  96: launch_mul_mat_w4a4< 96>(x, src1_q.get(), dst_d, nrows_x, s01, ncols, ncols_pad, nchunks, s1, stream); break;
        case 112: launch_mul_mat_w4a4<112>(x, src1_q.get(), dst_d, nrows_x, s01, ncols, ncols_pad, nchunks, s1, stream); break;
        case 128: launch_mul_mat_w4a4<128>(x, src1_q.get(), dst_d, nrows_x, s01, ncols, ncols_pad, nchunks, s1, stream); break;
        default: GGML_ABORT("fatal error");
    }
    CUDA_CHECK(cudaGetLastError());

    // 1: compare against the naive kernel with identical numerics, 2: report the error relative to the default W4A8 path.
    static const int check = w4a4_env_int("GGML_CUDA_W4A4_CHECK", 0);
    if (check == 0) {
        return;
    }

    ggml_cuda_pool_alloc<float> ref(ctx.pool(), (size_t) ncols*nrows_x);
    if (check == 1) {
        mul_mat_w4a4_ref<<<dim3((nrows_x + 127)/128, ncols, 1), dim3(128, 1, 1), 0, stream>>>(
            x, (const float *) src1->data, ref.get(), nrows_x, s01, ne00, s11, nrows_x);
        CUDA_CHECK(cudaGetLastError());
    } else {
        ggml_tensor dst_ref = *dst;
        dst_ref.data  = ref.get();
        dst_ref.nb[1] = nrows_x*sizeof(float);
        dst_ref.nb[2] = dst_ref.nb[1]*dst->ne[1];
        dst_ref.nb[3] = dst_ref.nb[2]*dst->ne[2];
        ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, &dst_ref);
    }

    std::vector<float> h_ref((size_t) ncols*nrows_x);
    std::vector<float> h_dst((size_t) ncols*nrows_x);
    CUDA_CHECK(cudaMemcpyAsync(h_ref.data(), ref.get(), h_ref.size()*sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpy2DAsync(h_dst.data(), nrows_x*sizeof(float), dst_d, s1*sizeof(float), nrows_x*sizeof(float), ncols,
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    double max_err = 0.0;
    double max_ref = 0.0;
    double sq_err  = 0.0;
    double sq_ref  = 0.0;
    for (size_t l = 0; l < h_ref.size(); ++l) {
        const double e = (double) h_dst[l] - (double) h_ref[l];
        max_err = std::max(max_err, fabs(e));
        max_ref = std::max(max_ref, (double) fabsf(h_ref[l]));
        sq_err += e*e;
        sq_ref += (double) h_ref[l]*h_ref[l];
    }
    GGML_LOG_INFO("%s: mode=%d check=%d %s M=%d N=%d K=%d J=%d max_abs_err=%.3e max_abs_ref=%.3e rel=%.3e nmse=%.3e\n", __func__,
        w4a4_mode(), check, src0->name, nrows_x, ncols, ne00, J, max_err, max_ref, max_ref > 0.0 ? max_err/max_ref : 0.0,
        sq_ref > 0.0 ? sq_err/sq_ref : 0.0);
}
