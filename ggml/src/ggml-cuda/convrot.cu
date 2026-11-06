#include "convrot.cuh"
#include "convert.cuh"

#include <climits>
#include <cstdlib>
#include <cstring>

#define CONVROT_TILE 256

static __device__ __forceinline__ float convrot_butterfly(const float * v, int tid, int stride) {
    const int base = (tid / (4*stride)) * (4*stride) + tid % stride;
    const int lane = (tid / stride) % 4;
    const float a = v[base];
    const float b = v[base + 1*stride];
    const float c = v[base + 2*stride];
    const float d = v[base + 3*stride];
    return lane == 0 ? ( a + b + c - d)*0.5f :
           lane == 1 ? ( a + b - c + d)*0.5f :
           lane == 2 ? ( a - b + c + d)*0.5f :
                       (-a + b + c + d)*0.5f;
}

// Each of the CONVROT_TILE threads owns values[tid], which must be published before the call.
static __device__ __forceinline__ void convrot_rotate_tile(float * values, int tid) {
#pragma unroll
    for (int stride = 1; stride < CONVROT_TILE; stride *= 4) {
        const float value = convrot_butterfly(values, tid, stride);
        __syncthreads();
        values[tid] = value;
        __syncthreads();
    }
}

template <typename T>
static __global__ void convrot_rotate_rows(const T * src, float * dst) {
    __shared__ float values[CONVROT_TILE];
    const int tid = threadIdx.x;
    ggml_cuda_pdl_sync();
    const size_t offset = size_t(blockIdx.x) * CONVROT_TILE;
    values[tid] = float(src[offset + tid]);
    __syncthreads();
    convrot_rotate_tile(values, tid);
    dst[offset + tid] = values[tid];
}

static __global__ void convrot_reconstruct_f16(const int8_t * weights, const float * scales, half * reconstructed, int k) {
    __shared__ float values[CONVROT_TILE];
    const int tid = threadIdx.x;
    ggml_cuda_pdl_sync();
    const int row = blockIdx.x;
    const size_t offset = size_t(row) * k + size_t(blockIdx.y) * CONVROT_TILE;
    values[tid] = float(weights[offset + tid]) * scales[row];
    __syncthreads();
    convrot_rotate_tile(values, tid);
    reconstructed[offset + tid] = __float2half_rn(values[tid]);
}

template <int Columns>
static __global__ void convrot_i8_matvec(const float * x, const int8_t * w, const float * scales,
                                         float * dst, int k, int n, int columns) {
    const int lane = threadIdx.x;
    ggml_cuda_pdl_sync();
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= n) {
        return;
    }
    const int first_column = blockIdx.y * Columns;
    x += size_t(first_column) * k;
    const int * packed = reinterpret_cast<const int *>(w + size_t(row) * k);

    float sums[Columns][4] = {};
    for (int i = lane; i < k / 4; i += WARP_SIZE) {
        const int q = packed[i];
        const float w0 = float(int8_t(q));
        const float w1 = float(int8_t(q >> 8));
        const float w2 = float(int8_t(q >> 16));
        const float w3 = float(int8_t(q >> 24));
#pragma unroll
        for (int c = 0; c < Columns; ++c) {
            if (first_column + c < columns) {
                const float4 a = reinterpret_cast<const float4 *>(x + size_t(c) * k)[i];
                sums[c][0] = fmaf(w0, a.x, sums[c][0]);
                sums[c][1] = fmaf(w1, a.y, sums[c][1]);
                sums[c][2] = fmaf(w2, a.z, sums[c][2]);
                sums[c][3] = fmaf(w3, a.w, sums[c][3]);
            }
        }
    }
#pragma unroll
    for (int c = 0; c < Columns; ++c) {
        const float sum = warp_reduce_sum((sums[c][0] + sums[c][1]) + (sums[c][2] + sums[c][3]));
        if (lane == 0 && first_column + c < columns) {
            dst[row + size_t(first_column + c) * n] = sum * scales[row];
        }
    }
}

static __device__ __forceinline__ float convrot_block_sum(float * values, int tid, float sum) {
    values[tid] = sum;
    __syncthreads();
#pragma unroll
    for (int stride = CONVROT_TILE / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            values[tid] += values[tid + stride];
        }
        __syncthreads();
    }
    return values[0];
}

template <typename T>
static __global__ void mul_mat_convrot_strided(
        const char * activations, const char * weights, const char * scales, char * dst,
        int k, int ne02,
        size_t nb00, size_t nb01, size_t nb02, size_t nb03,
        size_t nb10, size_t nb11, size_t nb20,
        size_t nb0, size_t nb1, size_t nb2, size_t nb3) {
    __shared__ float values[CONVROT_TILE];

    const int tid = threadIdx.x;
    ggml_cuda_pdl_sync();
    const int row = blockIdx.x;
    const int i1 = blockIdx.y;
    const int i2 = blockIdx.z % ne02;
    const int i3 = blockIdx.z / ne02;
    const float scale = *reinterpret_cast<const float *>(scales + row*nb20);
    const char * activation = activations + i1*nb01 + i2*nb02 + i3*nb03;
    const char * weight_row = weights + row*nb11;

    float sum = 0.0f;
    for (int k0 = 0; k0 < k; k0 += CONVROT_TILE) {
        values[tid] = float(*reinterpret_cast<const int8_t *>(weight_row + (k0 + tid)*nb10)) * scale;
        __syncthreads();
        convrot_rotate_tile(values, tid);
        sum += values[tid] * float(*reinterpret_cast<const T *>(activation + (k0 + tid)*nb00));
        __syncthreads();
    }
    const float total = convrot_block_sum(values, tid, sum);
    if (tid == 0) {
        *reinterpret_cast<float *>(dst + row*nb0 + i1*nb1 + i2*nb2 + i3*nb3) = total;
    }
}

static __global__ void convrot_h256_rows(const float * src, float * dst, int64_t rows) {
    __shared__ float values[CONVROT_TILE];
    const int tid = threadIdx.x;
    const int64_t row = blockIdx.x;
    ggml_cuda_pdl_sync();
    if (row >= rows) {
        return;
    }
    values[tid] = src[row * CONVROT_TILE + tid];
    __syncthreads();
    convrot_rotate_tile(values, tid);
    dst[row * CONVROT_TILE + tid] = values[tid];
}

static bool convrot_f16_compat_forced() {
    static const bool forced = [] {
        const char * env = std::getenv("GGML_CUDA_CONVROT_F16_MATMUL");
        return env != nullptr && std::strcmp(env, "0") != 0;
    }();
    return forced;
}

static bool convrot_can_use_f16_compat(const ggml_tensor * dst) {
    const ggml_tensor * activations = dst->src[0];
    const ggml_tensor * weights     = dst->src[1];
    return ggml_is_contiguous(activations) && ggml_is_contiguous(weights) && ggml_is_contiguous(dst->src[2]) &&
           ggml_is_contiguous(dst) && activations->ne[2] == 1 && activations->ne[3] == 1 &&
           weights->ne[0] <= INT_MAX && weights->ne[1] <= INT_MAX && activations->ne[1] <= INT_MAX &&
           ggml_get_to_fp16_cuda(activations->type) != nullptr;
}

// Rebuilds the F16 weights the compatibility loaders produce, one tile at a time,
// and runs them through F16 cuBLAS so operand, accumulation and output rounding match.
static void convrot_mul_mat_f16_compat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * activations = dst->src[0];
    const ggml_tensor * weights     = dst->src[1];
    const ggml_tensor * scales      = dst->src[2];

    const int k = int(weights->ne[0]);
    const int n = int(weights->ne[1]);
    const int columns = int(activations->ne[1]);
    ggml_cuda_pool_alloc<half> reconstructed(ctx.pool(), size_t(k) * n);
    ggml_cuda_pool_alloc<half> activation_f16(ctx.pool(), ggml_nelements(activations));
    ggml_cuda_pool_alloc<half> output_f16(ctx.pool(), ggml_nelements(dst));

    const auto launch = ggml_cuda_kernel_launch_params(dim3(n, k / CONVROT_TILE), dim3(CONVROT_TILE), 0, ctx.stream());
    ggml_cuda_kernel_launch(convrot_reconstruct_f16, launch,
                            (const int8_t *) weights->data, (const float *) scales->data, reconstructed.get(), k);
    ggml_get_to_fp16_cuda(activations->type)(activations->data, activation_f16.get(), ggml_nelements(activations), ctx.stream());

    const half alpha = 1.0f;
    const half beta  = 0.0f;
    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), ctx.stream()));
    CUBLAS_CHECK(cublasGemmEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                              n, columns, k,
                              &alpha, reconstructed.get(),  CUDA_R_16F, k,
                                      activation_f16.get(), CUDA_R_16F, k,
                              &beta,  output_f16.get(),     CUDA_R_16F, n,
                              CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    ggml_get_to_fp32_cuda(GGML_TYPE_F16)(output_f16.get(), (float *) dst->data, ggml_nelements(dst), ctx.stream());
}

static bool convrot_can_use_matvec(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    const ggml_tensor * activations = dst->src[0];
    const ggml_tensor * weights     = dst->src[1];
    return ggml_is_contiguous(activations) && ggml_is_contiguous(weights) &&
           ggml_is_contiguous(dst->src[2]) && ggml_is_contiguous(dst) &&
           uintptr_t(weights->data) % alignof(int) == 0 &&
           weights->ne[0] <= INT_MAX && weights->ne[1] <= INT_MAX - 3 &&
           ggml_nelements(activations) / CONVROT_TILE <= INT_MAX &&
           ggml_nrows(activations) <= 65535 && ggml_cuda_info().devices[ctx.device].warp_size == 32;
}

// The rotation is symmetric, so dot(H*(s*w), x) == s*dot(w, H*x): rotate the
// activations once instead of every weight row, with K*columns floats of scratch.
static void convrot_mul_mat_rotated_input(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * activations = dst->src[0];
    const ggml_tensor * weights     = dst->src[1];
    const ggml_tensor * scales      = dst->src[2];

    const int k = int(weights->ne[0]);
    const int n = int(weights->ne[1]);
    const int columns = int(ggml_nrows(activations));
    ggml_cuda_pool_alloc<float> rotated(ctx.pool(), ggml_nelements(activations));

    const auto rotate_launch = ggml_cuda_kernel_launch_params(dim3(ggml_nelements(activations) / CONVROT_TILE), dim3(CONVROT_TILE), 0, ctx.stream());
    if (activations->type == GGML_TYPE_F32) {
        ggml_cuda_kernel_launch(convrot_rotate_rows<float>, rotate_launch, (const float *) activations->data, rotated.get());
    } else {
        ggml_cuda_kernel_launch(convrot_rotate_rows<half>, rotate_launch, (const half *) activations->data, rotated.get());
    }

    constexpr int rows_per_block = 4;
    const int columns_per_block = columns > 1 ? 4 : 1;
    const auto dot_launch = ggml_cuda_kernel_launch_params(
        dim3((n + rows_per_block - 1) / rows_per_block, (columns + columns_per_block - 1) / columns_per_block),
        dim3(WARP_SIZE, rows_per_block), 0, ctx.stream());
    if (columns > 1) {
        ggml_cuda_kernel_launch(convrot_i8_matvec<4>, dot_launch, rotated.get(), (const int8_t *) weights->data,
                                (const float *) scales->data, (float *) dst->data, k, n, columns);
    } else {
        ggml_cuda_kernel_launch(convrot_i8_matvec<1>, dot_launch, rotated.get(), (const int8_t *) weights->data,
                                (const float *) scales->data, (float *) dst->data, k, n, columns);
    }
}

template <typename T>
static void convrot_mul_mat_strided_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * a = dst->src[0];
    const ggml_tensor * w = dst->src[1];
    const ggml_tensor * s = dst->src[2];

    const dim3 grid(w->ne[1], a->ne[1], a->ne[2] * a->ne[3]);
    const auto launch = ggml_cuda_kernel_launch_params(grid, dim3(CONVROT_TILE), 0, ctx.stream());
    ggml_cuda_kernel_launch(mul_mat_convrot_strided<T>, launch,
        (const char *) a->data, (const char *) w->data, (const char *) s->data, (char *) dst->data,
        (int) a->ne[0], (int) a->ne[2], a->nb[0], a->nb[1], a->nb[2], a->nb[3],
        w->nb[0], w->nb[1], s->nb[0], dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
}

void ggml_cuda_op_mul_mat_convrot(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * activations = dst->src[0];
    GGML_ASSERT(ggml_get_op_params_i32(dst, 0) == CONVROT_TILE);
    GGML_ASSERT(activations->ne[0] == dst->src[1]->ne[0] && activations->ne[0] % CONVROT_TILE == 0);

    const bool f16_compat = ggml_get_op_params_i32(dst, 1) != 0 || convrot_f16_compat_forced();
    if (f16_compat && convrot_can_use_f16_compat(dst)) {
        convrot_mul_mat_f16_compat(ctx, dst);
        return;
    }
    if (convrot_can_use_matvec(ctx, dst)) {
        convrot_mul_mat_rotated_input(ctx, dst);
        return;
    }
    if (activations->type == GGML_TYPE_F32) {
        convrot_mul_mat_strided_launch<float>(ctx, dst);
    } else {
        convrot_mul_mat_strided_launch<half>(ctx, dst);
    }
}

bool ggml_cuda_op_mul_mat_use_convrot_h256(const ggml_tensor * op) {
    return op->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_CONVROT_H256;
}

bool ggml_cuda_op_convrot_h256(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    if (src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src->ne[0] != CONVROT_TILE ||
        !ggml_are_same_shape(src, dst) || !ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const int64_t rows = ggml_nrows(src);
    GGML_ASSERT(rows <= INT_MAX);
    const auto launch = ggml_cuda_kernel_launch_params(dim3(rows), dim3(CONVROT_TILE), 0, ctx.stream());
    ggml_cuda_kernel_launch(convrot_h256_rows, launch, (const float *) src->data, (float *) dst->data, rows);
    return true;
}
