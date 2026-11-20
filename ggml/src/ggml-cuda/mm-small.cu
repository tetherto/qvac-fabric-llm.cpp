#include "mm-small.cuh"

#include <cstdint>

// not tuned
#define CUDA_MM_SMALL_BLOCK_SIZE 128
// floats in one 16-byte load or store
static constexpr int MM_SMALL_VEC = sizeof(float4) / sizeof(float);

// dst[m, n, b] = sum over k of src0[k, m, b] * src1[k, n, b], all three contiguous.
//
// One thread per R rows of one matrix, K fixed at compile time. The R rows of src0 stay in registers, every src1
// column is read once with 16-byte loads and gives R dst values. Global loads are what limits a product this small,
// so the kernel is laid out to issue as few as possible. The sum runs over k in order.
template <int K, int R>
static __global__ void k_mul_mat_small(
        const float * __restrict__ src0, const float * __restrict__ src1, float * __restrict__ dst,
        const int ne01, const int ne11, const int64_t n_groups) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_groups) {
        return;
    }
    const int     groups_per_mat = ne01 / R;
    const int     group          = i % groups_per_mat;
    const int64_t mat            = i / groups_per_mat;

    const float4 * x = (const float4 *) (src0 + (mat*ne01 + (int64_t) group*R)*K);
    const float4 * y = (const float4 *) (src1 + mat*ne11*K);
    float        * d = dst + mat*ne01*ne11 + group*R;

    float4 rows[R][K/4];
    for (int r = 0; r < R; ++r) {
        for (int k = 0; k < K/4; ++k) {
            rows[r][k] = x[r*(K/4) + k];
        }
    }

    for (int i11 = 0; i11 < ne11; ++i11) {
        float4 col[K/4];
        for (int k = 0; k < K/4; ++k) {
            col[k] = y[i11*(K/4) + k];
        }
        float sum[R];
        for (int r = 0; r < R; ++r) {
            float acc = 0.0f;
            for (int k = 0; k < K/4; ++k) {
                acc += rows[r][k].x * col[k].x;
                acc += rows[r][k].y * col[k].y;
                acc += rows[r][k].z * col[k].z;
                acc += rows[r][k].w * col[k].w;
            }
            sum[r] = acc;
        }
        if (R == 4) {
            *(float4 *) (d + (int64_t) i11*ne01) = make_float4(sum[0], sum[1], sum[2], sum[3]);
        } else {
            for (int r = 0; r < R; ++r) {
                d[(int64_t) i11*ne01 + r] = sum[r];
            }
        }
    }
}

// K below one 16-byte load: one thread per src0 row, four src1 columns at a time.
static __global__ void k_mul_mat_small_scalar(
        const float * __restrict__ src0, const float * __restrict__ src1, float * __restrict__ dst,
        const int ne00, const int ne01, const int ne11, const int64_t n_rows) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_rows) {
        return;
    }
    const int     i01 = i % ne01;
    const int64_t mat = i / ne01;

    const float * x = src0 + i*ne00;
    const float * y = src1 + mat*ne11*ne00;
    float       * d = dst + mat*ne01*ne11 + i01;

    int i11 = 0;
    for (; i11 + 4 <= ne11; i11 += 4) {
        const float * y0 = y + i11*ne00;
        float s0 = 0.0f;
        float s1 = 0.0f;
        float s2 = 0.0f;
        float s3 = 0.0f;
        for (int k = 0; k < ne00; ++k) {
            const float xv = x[k];
            s0 += xv * y0[k];
            s1 += xv * y0[k +   ne00];
            s2 += xv * y0[k + 2*ne00];
            s3 += xv * y0[k + 3*ne00];
        }
        d[(int64_t) (i11 + 0)*ne01] = s0;
        d[(int64_t) (i11 + 1)*ne01] = s1;
        d[(int64_t) (i11 + 2)*ne01] = s2;
        d[(int64_t) (i11 + 3)*ne01] = s3;
    }
    for (; i11 < ne11; ++i11) {
        const float * y0 = y + i11*ne00;
        float s = 0.0f;
        for (int k = 0; k < ne00; ++k) {
            s += x[k] * y0[k];
        }
        d[(int64_t) i11*ne01] = s;
    }
}

static bool mm_small_is_aligned(const void * ptr) {
    return (uintptr_t) ptr % sizeof(float4) == 0;
}

// rows per thread: 4 where the dst rows can be written with one 16-byte store, single rows for very short matrices
static int mm_small_rows_per_thread(const ggml_tensor * src0, const ggml_tensor * dst) {
    const int64_t ne01 = src0->ne[1];
    if (ne01 <= MM_SMALL_VEC) {
        return 1;
    }
    if (ne01 % MM_SMALL_VEC == 0 && mm_small_is_aligned(dst->data)) {
        return MM_SMALL_VEC;
    }
    return ne01 % 2 == 0 ? 2 : 1;
}

static bool ggml_cuda_mm_small_forced() {
    static const bool forced = ggml_cuda_env_flag("GGML_CUDA_FORCE_MM_SMALL");
    return forced;
}

bool ggml_cuda_should_use_mm_small(const ggml_tensor * src0,
                                   const ggml_tensor * src1,
                                   const ggml_tensor * dst,
                                   const int           cc) {
    if (!ggml_cuda_mm_small_forced() && !ggml_cuda_mm_small_measured_cc(cc)) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    // no broadcast: one src0 matrix per src1 matrix
    if (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
        return false;
    }
    const int64_t ne00 = src0->ne[0];
    if (ne00 > CUDA_MM_SMALL_MAX_K || src1->ne[1] > CUDA_MM_SMALL_MAX_COLS || src0->ne[1] > INT32_MAX/CUDA_MM_SMALL_MAX_K) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }
    const int64_t n_mats = src1->ne[2]*src1->ne[3];
    if (ne00 < MM_SMALL_VEC) {
        // k_mul_mat_small_scalar: one thread per src0 row
        return src0->ne[1]*n_mats >= CUDA_MM_SMALL_MIN_THREADS;
    }
    // the 16-byte loads need whole float4 rows
    if (ne00 % MM_SMALL_VEC != 0 || !mm_small_is_aligned(src0->data) || !mm_small_is_aligned(src1->data)) {
        return false;
    }
    return src0->ne[1]/mm_small_rows_per_thread(src0, dst)*n_mats >= CUDA_MM_SMALL_MIN_THREADS;
}

template <int R>
static void mm_small_launch(
        const float * src0_d, const float * src1_d, float * dst_d,
        const int ne00, const int ne01, const int ne11, const int64_t n_mats, cudaStream_t stream) {
    const int64_t n_groups = (ne01 / R)*n_mats;
    const int64_t n_blocks = (n_groups + CUDA_MM_SMALL_BLOCK_SIZE - 1) / CUDA_MM_SMALL_BLOCK_SIZE;
    GGML_ASSERT(n_blocks <= INT32_MAX);
    const dim3 grid((unsigned int) n_blocks);
    const dim3 block(CUDA_MM_SMALL_BLOCK_SIZE);
    switch (ne00) {
        case  4: k_mul_mat_small< 4, R><<<grid, block, 0, stream>>>(src0_d, src1_d, dst_d, ne01, ne11, n_groups); break;
        case  8: k_mul_mat_small< 8, R><<<grid, block, 0, stream>>>(src0_d, src1_d, dst_d, ne01, ne11, n_groups); break;
        case 12: k_mul_mat_small<12, R><<<grid, block, 0, stream>>>(src0_d, src1_d, dst_d, ne01, ne11, n_groups); break;
        case 16: k_mul_mat_small<16, R><<<grid, block, 0, stream>>>(src0_d, src1_d, dst_d, ne01, ne11, n_groups); break;
        case 20: k_mul_mat_small<20, R><<<grid, block, 0, stream>>>(src0_d, src1_d, dst_d, ne01, ne11, n_groups); break;
        case 24: k_mul_mat_small<24, R><<<grid, block, 0, stream>>>(src0_d, src1_d, dst_d, ne01, ne11, n_groups); break;
        case 28: k_mul_mat_small<28, R><<<grid, block, 0, stream>>>(src0_d, src1_d, dst_d, ne01, ne11, n_groups); break;
        case 32: k_mul_mat_small<32, R><<<grid, block, 0, stream>>>(src0_d, src1_d, dst_d, ne01, ne11, n_groups); break;
        default: GGML_ABORT("fatal error");
    }
}

void ggml_cuda_mul_mat_small(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_should_use_mm_small(src0, src1, dst, ggml_cuda_info().devices[ctx.device].cc));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float *) dst->data;

    const int     ne00   = src0->ne[0];
    const int     ne01   = src0->ne[1];
    const int     ne11   = src1->ne[1];
    const int64_t n_mats = src1->ne[2]*src1->ne[3];
    cudaStream_t  stream = ctx.stream();

    if (ne00 < MM_SMALL_VEC) {
        const int64_t n_rows   = ne01*n_mats;
        const int64_t n_blocks = (n_rows + CUDA_MM_SMALL_BLOCK_SIZE - 1) / CUDA_MM_SMALL_BLOCK_SIZE;
        GGML_ASSERT(n_blocks <= INT32_MAX);
        const dim3 grid((unsigned int) n_blocks);
        const dim3 block(CUDA_MM_SMALL_BLOCK_SIZE);
        k_mul_mat_small_scalar<<<grid, block, 0, stream>>>(src0_d, src1_d, dst_d, ne00, ne01, ne11, n_rows);
        return;
    }

    switch (mm_small_rows_per_thread(src0, dst)) {
        case 4:  mm_small_launch<4>(src0_d, src1_d, dst_d, ne00, ne01, ne11, n_mats, stream); break;
        case 2:  mm_small_launch<2>(src0_d, src1_d, dst_d, ne00, ne01, ne11, n_mats, stream); break;
        default: mm_small_launch<1>(src0_d, src1_d, dst_d, ne00, ne01, ne11, n_mats, stream); break;
    }
}
