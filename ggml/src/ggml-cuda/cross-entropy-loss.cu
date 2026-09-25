#include "common.cuh"
#include "cross-entropy-loss.cuh"
#include "sum.cuh"

#include <cmath>
#include <cstdint>

static __device__ __forceinline__ int cross_entropy_loss_nvalid(
        const float * __restrict__ mask, const int nclasses, const int nrows) {
    int nvalid = 0;
    for (int r = threadIdx.x; r < nrows; r += WARP_SIZE) {
        nvalid += mask[int64_t(r)*nclasses] > 0.5f;
    }
    return warp_reduce_sum(nvalid);
}

template <bool use_shared, bool masked>
static __global__ void cross_entropy_loss_f32(
        const float * __restrict__ logits, const float * __restrict__ labels, const float * __restrict__ mask,
        float * __restrict__ dst, const int nclasses, const int k) {
    extern __shared__ float tmp[];

    logits += int64_t(blockIdx.x)*nclasses;
    labels += int64_t(blockIdx.x)*nclasses;

    if (masked && mask[int64_t(blockIdx.x)*nclasses] <= 0.5f) {
        if (threadIdx.x == 0) {
            dst[blockIdx.x] = 0.0f;
        }
        return;
    }

    // Find maximum for softmax:
    float max_logit = -INFINITY;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        const float val = logits[i];
        max_logit = fmaxf(max_logit, val);

        if (use_shared) {
            tmp[i] = val;
        }
    }
    max_logit = warp_reduce_max(max_logit);

    // Calculate log(softmax(logits)) which is just logits - max:
    float sum = 0.0f;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        const float logit_i = use_shared ? tmp[i] : logits[i];
        sum += expf(logit_i - max_logit);
    }
    sum = warp_reduce_sum(sum);
    sum = logf(sum);

    // log(exp(logits - max) / sum) = (logits - max) - log(sum)
    float loss = 0.0f;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        const float logit_i = use_shared ? tmp[i] : logits[i];
        loss += (logit_i - max_logit - sum) * labels[i];
    }

    const float nrows_eff = masked ? (float) cross_entropy_loss_nvalid(mask, nclasses, k) : (float) k;
    loss = -warp_reduce_sum(loss) / nrows_eff;

    if (threadIdx.x != 0) {
        return;
    }

    dst[blockIdx.x] = loss;
}

template <bool use_shared, bool masked>
static __global__ void cross_entropy_loss_back_f32(
        const float * __restrict__ grad, const float * __restrict__ logits, const float * __restrict__ labels,
        const float * __restrict__ mask, float * __restrict__ dst, const int nclasses) {
    extern __shared__ float tmp[];

    logits += int64_t(blockIdx.x)*nclasses;
    labels += int64_t(blockIdx.x)*nclasses;
    dst    += int64_t(blockIdx.x)*nclasses;

    if (masked && mask[int64_t(blockIdx.x)*nclasses] <= 0.5f) {
        for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
            dst[i] = 0.0f;
        }
        return;
    }

    float maxval = -INFINITY;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        const float val = logits[i];
        maxval = fmaxf(maxval, val);

        if (use_shared) {
            tmp[i] = val;
        }
    }
    maxval = warp_reduce_max(maxval);

    float sum = 0.0f;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        const float val = expf((use_shared ? tmp[i] : logits[i]) - maxval);
        sum += val;

        if (use_shared) {
            tmp[i] = val;
        } else {
            dst[i] = val;
        }
    }
    sum = warp_reduce_sum(sum);
    const float sm_scale = 1.0f/sum;

    const float nrows_eff = masked ? (float) cross_entropy_loss_nvalid(mask, nclasses, gridDim.x) : (float) gridDim.x;
    const float d_by_nrows = *grad/nrows_eff;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        const float val = use_shared ? tmp[i] : dst[i];
        dst[i] = (val*sm_scale - labels[i])*d_by_nrows;
    }
}

void ggml_cuda_cross_entropy_loss(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float       *) dst->data;

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();

    const dim3 blocks_dim(WARP_SIZE, 1, 1);
    const dim3 blocks_num(nrows, 1, 1);
    const size_t nbytes_shared = ne00*sizeof(float);

    const int    id    = ggml_cuda_get_device();
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

    ggml_cuda_pool_alloc<float> dst_tmp(pool, blocks_num.x);

    if (nbytes_shared <= smpbo) {
        CUDA_SET_SHARED_MEMORY_LIMIT((cross_entropy_loss_f32<true, false>), smpbo);
        cross_entropy_loss_f32<true, false><<<blocks_num, blocks_dim, nbytes_shared, stream>>>(src0_d, src1_d, nullptr, dst_tmp.ptr, ne00, nrows);
    } else {
        cross_entropy_loss_f32<false, false><<<blocks_num, blocks_dim, 0, stream>>>(src0_d, src1_d, nullptr, dst_tmp.ptr, ne00, nrows);
    }
    CUDA_CHECK(cudaGetLastError());

    // Combine results from individual blocks:
    sum_f32_cuda(pool, dst_tmp.ptr, dst_d, blocks_num.x, stream);
}

void ggml_cuda_cross_entropy_loss_masked(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src2->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(src2));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_are_same_shape(src0, src2));

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * src2_d = (const float *) src2->data;
    float       * dst_d  = (float       *) dst->data;

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();

    const dim3 blocks_dim(WARP_SIZE, 1, 1);
    const dim3 blocks_num(nrows, 1, 1);
    const size_t nbytes_shared = ne00*sizeof(float);

    const int    id    = ggml_cuda_get_device();
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

    ggml_cuda_pool_alloc<float> dst_tmp(pool, blocks_num.x);

    if (nbytes_shared <= smpbo) {
        CUDA_SET_SHARED_MEMORY_LIMIT((cross_entropy_loss_f32<true, true>), smpbo);
        cross_entropy_loss_f32<true, true><<<blocks_num, blocks_dim, nbytes_shared, stream>>>(src0_d, src1_d, src2_d, dst_tmp.ptr, ne00, nrows);
    } else {
        cross_entropy_loss_f32<false, true><<<blocks_num, blocks_dim, 0, stream>>>(src0_d, src1_d, src2_d, dst_tmp.ptr, ne00, nrows);
    }
    CUDA_CHECK(cudaGetLastError());

    // Combine results from individual blocks:
    sum_f32_cuda(pool, dst_tmp.ptr, dst_d, blocks_num.x, stream);
}

void ggml_cuda_cross_entropy_loss_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * grad  = dst->src[0];
    const ggml_tensor * src0f = dst->src[1];
    const ggml_tensor * src1f = dst->src[2];

    GGML_ASSERT(src0f->type == GGML_TYPE_F32);
    GGML_ASSERT(src1f->type == GGML_TYPE_F32);
    GGML_ASSERT( grad->type == GGML_TYPE_F32);
    GGML_ASSERT(  dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_scalar(grad));
    GGML_ASSERT(ggml_is_contiguous(src0f));
    GGML_ASSERT(ggml_is_contiguous(src1f));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0f, src1f));
    GGML_ASSERT(ggml_are_same_shape(src0f, dst));

    const int64_t ne00  = src0f->ne[0];
    const int64_t nrows = ggml_nrows(src0f);

    const float * grad_d  = (const float *) grad->data;
    const float * src0f_d = (const float *) src0f->data;
    const float * src1f_d = (const float *) src1f->data;
    float       * dst_d   = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    const dim3 blocks_dim(WARP_SIZE, 1, 1);
    const dim3 blocks_num(nrows, 1, 1);
    const size_t nbytes_shared = ne00*sizeof(float);

    const int    id    = ggml_cuda_get_device();
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

    if (nbytes_shared <= smpbo) {
        CUDA_SET_SHARED_MEMORY_LIMIT((cross_entropy_loss_back_f32<true, false>), smpbo);
        cross_entropy_loss_back_f32<true, false><<<blocks_num, blocks_dim, nbytes_shared, stream>>>(grad_d, src0f_d, src1f_d, nullptr, dst_d, ne00);
    } else {
        cross_entropy_loss_back_f32<false, false><<<blocks_num, blocks_dim, 0, stream>>>(grad_d, src0f_d, src1f_d, nullptr, dst_d, ne00);
    }
}

void ggml_cuda_cross_entropy_loss_masked_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * grad  = dst->src[0];
    const ggml_tensor * src0f = dst->src[1];
    const ggml_tensor * src1f = dst->src[2];
    const ggml_tensor * src2f = dst->src[3];

    GGML_ASSERT(src0f->type == GGML_TYPE_F32);
    GGML_ASSERT(src1f->type == GGML_TYPE_F32);
    GGML_ASSERT(src2f->type == GGML_TYPE_F32);
    GGML_ASSERT( grad->type == GGML_TYPE_F32);
    GGML_ASSERT(  dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_scalar(grad));
    GGML_ASSERT(ggml_is_contiguous(src0f));
    GGML_ASSERT(ggml_is_contiguous(src1f));
    GGML_ASSERT(ggml_is_contiguous(src2f));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0f, src1f));
    GGML_ASSERT(ggml_are_same_shape(src0f, src2f));
    GGML_ASSERT(ggml_are_same_shape(src0f, dst));

    const int64_t ne00  = src0f->ne[0];
    const int64_t nrows = ggml_nrows(src0f);

    const float * grad_d  = (const float *) grad->data;
    const float * src0f_d = (const float *) src0f->data;
    const float * src1f_d = (const float *) src1f->data;
    const float * src2f_d = (const float *) src2f->data;
    float       * dst_d   = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    const dim3 blocks_dim(WARP_SIZE, 1, 1);
    const dim3 blocks_num(nrows, 1, 1);
    const size_t nbytes_shared = ne00*sizeof(float);

    const int    id    = ggml_cuda_get_device();
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;

    if (nbytes_shared <= smpbo) {
        CUDA_SET_SHARED_MEMORY_LIMIT((cross_entropy_loss_back_f32<true, true>), smpbo);
        cross_entropy_loss_back_f32<true, true><<<blocks_num, blocks_dim, nbytes_shared, stream>>>(grad_d, src0f_d, src1f_d, src2f_d, dst_d, ne00);
    } else {
        cross_entropy_loss_back_f32<false, true><<<blocks_num, blocks_dim, 0, stream>>>(grad_d, src0f_d, src1f_d, src2f_d, dst_d, ne00);
    }
    CUDA_CHECK(cudaGetLastError());
}
