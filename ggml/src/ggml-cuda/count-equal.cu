#include "common.cuh"
#include "count-equal.cuh"

#include <cstdint>

template <typename T, bool has_mask>
static __global__ void count_equal(const T * __restrict__ x, const T * __restrict__ y, const float * __restrict__ mask,
        int64_t * __restrict__ dst, const int64_t mask_stride, const int64_t dk, const int64_t k) {
    const int64_t i0 = (int64_t) blockIdx.x*dk;
    const int64_t i1 = min(i0 + dk, k);

    int nequal = 0;

    for (int64_t i = i0 + threadIdx.x; i < i1; i += WARP_SIZE) {
        if (has_mask && mask[i*mask_stride] <= 0.5f) {
            continue;
        }
        const T xi = x[i];
        const T yi = y[i];
        nequal += xi == yi;
    }

    nequal = warp_reduce_sum(nequal);

    if (threadIdx.x != 0) {
        return;
    }

    atomicAdd((int *) dst, nequal);
}

void ggml_cuda_count_equal(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT( dst->type == GGML_TYPE_I64);

    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    int64_t * dst_d  = (int64_t *) dst->data;

    cudaStream_t stream = ctx.stream();
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;

    const int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne < (1 << 30) && "atomicAdd implementation only supports int");
    const int64_t dne = GGML_PAD((ne + 4*nsm - 1) / (4*nsm), CUDA_COUNT_EQUAL_CHUNK_SIZE);

    CUDA_CHECK(cudaMemsetAsync(dst_d, 0, ggml_nbytes(dst), stream));

    const dim3 blocks_dim(WARP_SIZE, 1, 1);
    const dim3 blocks_num(std::min((int64_t)4*nsm, (ne + CUDA_COUNT_EQUAL_CHUNK_SIZE - 1)/CUDA_COUNT_EQUAL_CHUNK_SIZE), 1, 1);

    switch (src0->type) {
        case GGML_TYPE_I32: {
            const int * src0_d = (const int *) src0->data;
            const int * src1_d = (const int *) src1->data;
            count_equal<int, false><<<blocks_num, blocks_dim, 0, stream>>>(src0_d, src1_d, nullptr, dst_d, 0, dne, ne);
        } break;
        default:
            GGML_ASSERT(false);
            break;
    }
}

void ggml_cuda_count_equal_masked(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT(src2->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I64);

    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(src2));
    GGML_ASSERT(ggml_is_contiguous(dst));

    // Either the mask is elementwise, or it is one row per element of src0 and only the
    // first element of each row carries the flag.
    const bool mask_same_shape = ggml_are_same_shape(src0, src2);
    GGML_ASSERT(mask_same_shape || src2->ne[1] == src0->ne[0]);
    const int64_t mask_stride = mask_same_shape ? 1 : src2->ne[0];

    int64_t * dst_d  = (int64_t *) dst->data;
    const float * src2_d = (const float *) src2->data;

    cudaStream_t stream = ctx.stream();
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;

    const int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne < (1 << 30) && "atomicAdd implementation only supports int");
    const int64_t dne = GGML_PAD((ne + 4*nsm - 1) / (4*nsm), CUDA_COUNT_EQUAL_CHUNK_SIZE);

    CUDA_CHECK(cudaMemsetAsync(dst_d, 0, ggml_nbytes(dst), stream));

    const dim3 blocks_dim(WARP_SIZE, 1, 1);
    const dim3 blocks_num(std::min((int64_t)4*nsm, (ne + CUDA_COUNT_EQUAL_CHUNK_SIZE - 1)/CUDA_COUNT_EQUAL_CHUNK_SIZE), 1, 1);

    switch (src0->type) {
        case GGML_TYPE_I32: {
            const int * src0_d = (const int *) src0->data;
            const int * src1_d = (const int *) src1->data;
            count_equal<int, true><<<blocks_num, blocks_dim, 0, stream>>>(src0_d, src1_d, src2_d, dst_d, mask_stride, dne, ne);
        } break;
        default:
            GGML_ASSERT(false);
            break;
    }
    CUDA_CHECK(cudaGetLastError());
}
