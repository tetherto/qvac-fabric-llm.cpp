#include "mmvf-act.cuh"
#include "mmf.cuh"
#include "mmvf.cuh"
#include "unary.cuh"

// one block per weight row; each thread keeps up to MMVF_ACT_MAX_ITERS float4 of the row in registers
static __device__ __forceinline__ void mmvf_act_load_row(
        const float4 * x4, const int ncols_x4, float4 (&x)[MMVF_ACT_MAX_ITERS]) {
#pragma unroll
    for (int it = 0; it < MMVF_ACT_MAX_ITERS; ++it) {
        const int i = it*blockDim.x + threadIdx.x;
        x[it] = i < ncols_x4 ? x4[i] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
}

template <int ncols_dst>
static __device__ __forceinline__ void mmvf_act_dot(
        const float4 (&x)[MMVF_ACT_MAX_ITERS], const float4 * y4, const int ncols_x4, const int64_t stride_col_y4,
        float (&sum)[ncols_dst]) {
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        sum[j] = 0.0f;
#pragma unroll
        for (int it = 0; it < MMVF_ACT_MAX_ITERS; ++it) {
            const int i = it*blockDim.x + threadIdx.x;
            if (i < ncols_x4) {
                const float4 y = y4[j*stride_col_y4 + i];
                sum[j] += x[it].x*y.x;
                sum[j] += x[it].y*y.y;
                sum[j] += x[it].z*y.z;
                sum[j] += x[it].w*y.w;
            }
        }
    }
}

// block-wide sums in a fixed order, valid in thread 0
template <int ncols_dst>
static __device__ __forceinline__ void mmvf_act_block_sum(float (&sum)[ncols_dst]) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    __shared__ float buf[ncols_dst][MMVF_ACT_BLOCK_SIZE/WARP_SIZE];

    const int lane   = threadIdx.x % warp_size;
    const int warp   = threadIdx.x / warp_size;
    const int nwarps = blockDim.x / warp_size;

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        sum[j] = warp_reduce_sum<warp_size>(sum[j]);
        if (lane == 0) {
            buf[j][warp] = sum[j];
        }
    }
    __syncthreads();

    if (warp == 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            sum[j] = warp_reduce_sum<warp_size>(lane < nwarps ? buf[j][lane] : 0.0f);
        }
    }
}

template <ggml_cuda_mmvf_act act>
static __device__ __forceinline__ float mmvf_act_apply(const float v, const float * bias, const float * scale, const int row) {
    if constexpr (act == GGML_CUDA_MMVF_ACT_SOFTPLUS_GATE) {
        return ggml_cuda_op_softplus_single(v + bias[row]) * scale[row];
    } else {
        GGML_UNUSED(bias);
        GGML_UNUSED(scale);
        GGML_UNUSED(row);
        return ggml_cuda_op_sigmoid_single(v);
    }
}

template <int ncols_dst, ggml_cuda_mmvf_act act>
static __device__ __forceinline__ void mmvf_act_store(
        const float (&sum)[ncols_dst], const float * bias, const float * scale, float * dst, const int64_t stride_col_dst) {
    const int row = blockIdx.x;
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
        dst[j*stride_col_dst + row] = mmvf_act_apply<act>(sum[j], bias, scale, row);
    }
}

template <int ncols_dst, ggml_cuda_mmvf_act act>
static __global__ void __launch_bounds__(MMVF_ACT_BLOCK_SIZE, 1) mul_mat_vec_f32_act(
        const float * x, const float * y, const float * bias, const float * scale, float * dst,
        const int ncols_x4, const int64_t stride_row_x4, const int64_t stride_col_y4, const int64_t stride_col_dst,
        const bool x_is_constant) {
    ggml_cuda_pdl_lc();

    // constant weights can be loaded before waiting for the producer of y
    float4 xr[MMVF_ACT_MAX_ITERS];
    const float4 * x_row = (const float4 *) x + blockIdx.x*stride_row_x4;
    if (x_is_constant) {
        mmvf_act_load_row(x_row, ncols_x4, xr);
        ggml_cuda_pdl_sync();
    } else {
        ggml_cuda_pdl_sync();
        mmvf_act_load_row(x_row, ncols_x4, xr);
    }

    float sum[ncols_dst];
    mmvf_act_dot<ncols_dst>(xr, (const float4 *) y, ncols_x4, stride_col_y4, sum);
    mmvf_act_block_sum<ncols_dst>(sum);

    if (threadIdx.x == 0) {
        mmvf_act_store<ncols_dst, act>(sum, bias, scale, dst, stride_col_dst);
    }
}

bool ggml_cuda_should_use_mmvf_act(const ggml_tensor * mul_mat) {
    const ggml_tensor * src0 = mul_mat->src[0];
    const ggml_tensor * src1 = mul_mat->src[1];

    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || mul_mat->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(mul_mat) ||
        src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (src0->ne[0] <= 0 || src0->ne[0] % MMVF_ACT_VEC_WIDTH != 0 || src0->ne[0] > MMVF_ACT_MAX_K ||
        src1->ne[1] > MMVF_ACT_MAX_NCOLS ||
        (uintptr_t) src0->data % sizeof(float4) != 0 || (uintptr_t) src1->data % sizeof(float4) != 0) {
        return false;
    }

    // only replace the cuBLAS path: mmvf and mmf keep their column counts
    const int device    = ggml_cuda_get_device();
    const int cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    return !ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, src1->ne[1]) &&
           !ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, src1->ne[1], /*mul_mat_id =*/ false);
}

template <ggml_cuda_mmvf_act act>
static void mul_mat_vec_f32_act_cuda(
        const float * x, const float * y, const float * bias, const float * scale, float * dst,
        const int64_t ncols_x, const int64_t nrows_x, const int64_t ncols_dst,
        const int64_t stride_row_x, const int64_t stride_col_y, const int64_t stride_col_dst, const bool x_is_constant,
        cudaStream_t stream) {
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int ncols_x4  = (int) (ncols_x / MMVF_ACT_VEC_WIDTH);

    // smallest block that covers the row in MMVF_ACT_MAX_ITERS float4 per thread
    const int nthreads   = (ncols_x4 + MMVF_ACT_MAX_ITERS - 1) / MMVF_ACT_MAX_ITERS;
    const int block_size = std::min(MMVF_ACT_BLOCK_SIZE, (nthreads + warp_size - 1) / warp_size * warp_size);

    const ggml_cuda_kernel_launch_params launch_params = { dim3(nrows_x, 1, 1), dim3(block_size, 1, 1), 0, stream };
    const int64_t stride_row_x4 = stride_row_x / MMVF_ACT_VEC_WIDTH;
    const int64_t stride_col_y4 = stride_col_y / MMVF_ACT_VEC_WIDTH;

    switch (ncols_dst) {
        case 1: ggml_cuda_kernel_launch(mul_mat_vec_f32_act<1, act>, launch_params, x, y, bias, scale, dst, ncols_x4, stride_row_x4, stride_col_y4, stride_col_dst, x_is_constant); break;
        case 2: ggml_cuda_kernel_launch(mul_mat_vec_f32_act<2, act>, launch_params, x, y, bias, scale, dst, ncols_x4, stride_row_x4, stride_col_y4, stride_col_dst, x_is_constant); break;
        case 3: ggml_cuda_kernel_launch(mul_mat_vec_f32_act<3, act>, launch_params, x, y, bias, scale, dst, ncols_x4, stride_row_x4, stride_col_y4, stride_col_dst, x_is_constant); break;
        case 4: ggml_cuda_kernel_launch(mul_mat_vec_f32_act<4, act>, launch_params, x, y, bias, scale, dst, ncols_x4, stride_row_x4, stride_col_y4, stride_col_dst, x_is_constant); break;
        case 5: ggml_cuda_kernel_launch(mul_mat_vec_f32_act<5, act>, launch_params, x, y, bias, scale, dst, ncols_x4, stride_row_x4, stride_col_y4, stride_col_dst, x_is_constant); break;
        case 6: ggml_cuda_kernel_launch(mul_mat_vec_f32_act<6, act>, launch_params, x, y, bias, scale, dst, ncols_x4, stride_row_x4, stride_col_y4, stride_col_dst, x_is_constant); break;
        case 7: ggml_cuda_kernel_launch(mul_mat_vec_f32_act<7, act>, launch_params, x, y, bias, scale, dst, ncols_x4, stride_row_x4, stride_col_y4, stride_col_dst, x_is_constant); break;
        case 8: ggml_cuda_kernel_launch(mul_mat_vec_f32_act<8, act>, launch_params, x, y, bias, scale, dst, ncols_x4, stride_row_x4, stride_col_y4, stride_col_dst, x_is_constant); break;
        default: GGML_ABORT("unsupported ncols_dst %d", (int) ncols_dst);
    }
}

void ggml_cuda_mul_mat_vec_f32_act(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                   const ggml_tensor * bias, const ggml_tensor * scale, ggml_cuda_mmvf_act act,
                                   ggml_tensor * dst) {
    GGML_ASSERT(ggml_nelements(dst) == src0->ne[1]*src1->ne[1]);
    GGML_ASSERT(act != GGML_CUDA_MMVF_ACT_SOFTPLUS_GATE || (bias != nullptr && scale != nullptr));

    const float * x_d     = (const float *) src0->data;
    const float * y_d     = (const float *) src1->data;
    const float * bias_d  = bias  ? (const float *) bias->data  : nullptr;
    const float * scale_d = scale ? (const float *) scale->data : nullptr;
    float *       dst_d   = (float *) dst->data;

    const int64_t stride_row_x   = src0->nb[1] / sizeof(float);
    const int64_t stride_col_y   = src1->nb[1] / sizeof(float);
    const int64_t stride_col_dst = src0->ne[1];

    // same rule as ggml_is_constant: weights that no op of the graph writes
    const bool x_is_constant = src0->buffer != nullptr &&
        ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        (src0->flags & GGML_TENSOR_FLAG_PARAM) == 0;

    cudaStream_t stream = ctx.stream();
    if (act == GGML_CUDA_MMVF_ACT_SOFTPLUS_GATE) {
        mul_mat_vec_f32_act_cuda<GGML_CUDA_MMVF_ACT_SOFTPLUS_GATE>(x_d, y_d, bias_d, scale_d, dst_d, src0->ne[0],
            src0->ne[1], src1->ne[1], stride_row_x, stride_col_y, stride_col_dst, x_is_constant, stream);
    } else {
        mul_mat_vec_f32_act_cuda<GGML_CUDA_MMVF_ACT_SIGMOID>(x_d, y_d, bias_d, scale_d, dst_d, src0->ne[0],
            src0->ne[1], src1->ne[1], stride_row_x, stride_col_y, stride_col_dst, x_is_constant, stream);
    }
}
