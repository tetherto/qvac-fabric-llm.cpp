#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#else  // CUB_TOP_K_AVAILABLE

// rows up to this length are sorted in shared memory by the bitonic sort
static constexpr int TOP_K_BITONIC_MAX_NCOLS = 1024;

#ifdef GGML_CUDA_TOP_K_LONG_ROWS

static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

struct top_k_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
};

static __global__ void top_k_radix_init(top_k_radix_state * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k};
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    __shared__ int histogram[NBINS];

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state state = states[row];
    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_select(
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    static_assert(BLOCK_SIZE == NBINS, "one thread per bin");

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int suffix[NBINS];

    // read before the scan barriers so no thread sees the state written below
    const top_k_radix_state state = states[row];

    int count = 0;
    for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
        const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
        count += block_histograms[offset + tid];
    }
    suffix[tid] = count;
    __syncthreads();

    // suffix[b] = number of keys in bins >= b
    for (int stride = 1; stride < NBINS; stride *= 2) {
        const int add = tid + stride < NBINS ? suffix[tid + stride] : 0;
        __syncthreads();
        suffix[tid] += add;
        __syncthreads();
    }

    // the selected bin is the highest one whose suffix reaches rank; bin 0 is the fallback
    const int above = tid + 1 < NBINS ? suffix[tid + 1] : 0;
    if (above < state.rank && (tid == 0 || suffix[tid] >= state.rank)) {
        top_k_radix_state next = state;
        next.rank -= above;
        next.prefix |= (uint32_t) tid << shift;
        next.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = next;
    }
}

static __device__ __forceinline__ int top_k_sum_warp_totals(const int * warp_totals, int n_warps, int warp, int & total) {
    int before = 0;
    total = 0;
    for (int w = 0; w < n_warps; ++w) {
        before += w < warp ? warp_totals[w] : 0;
        total  += warp_totals[w];
    }
    return before;
}

// Exclusive prefix sum of one flag per thread across the block, with the block total in total.
template<int BLOCK_SIZE>
static __device__ __forceinline__ int top_k_block_exclusive_scan(int flag, int * warp_totals, int & total) {
    constexpr int n_warps = BLOCK_SIZE / WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int inclusive = warp_prefix_inclusive_sum<int, WARP_SIZE>(flag);
    if (threadIdx.x % WARP_SIZE == WARP_SIZE - 1) {
        warp_totals[warp] = inclusive;
    }
    __syncthreads();
    const int before = top_k_sum_warp_totals(warp_totals, n_warps, warp, total);
    __syncthreads();
    return before + inclusive - flag;
}

// Keys above and equal to the threshold in each block's contiguous column range.
template<int BLOCK_SIZE>
static __global__ void top_k_radix_count(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int2 * __restrict__ block_counts,
        int ncols,
        int blocks_per_row,
        int cols_per_block) {
    const int row = blockIdx.x / blocks_per_row;
    const int begin = (blockIdx.x % blocks_per_row) * cols_per_block;
    const int end = min(begin + cols_per_block, ncols);
    const float * row_src = src + (size_t) row * ncols;
    const uint32_t prefix = states[row].prefix;
    __shared__ int counts[2];

    if (threadIdx.x == 0) {
        counts[0] = 0;
        counts[1] = 0;
    }
    __syncthreads();

    for (int col = begin + threadIdx.x; col < end; col += BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if (key > prefix) {
            atomicAdd(&counts[0], 1);
        } else if (key == prefix) {
            atomicAdd(&counts[1], 1);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        block_counts[blockIdx.x] = make_int2(counts[0], counts[1]);
    }
}

static __device__ __forceinline__ int2 top_k_radix_counts_before(const int2 * row_counts, int row_block) {
    int2 before = make_int2(0, 0);
    for (int b = 0; b < row_block; ++b) {
        before.x += row_counts[b].x;
        before.y += row_counts[b].y;
    }
    return before;
}

// Writes the selected columns in ascending order, so ties keep the lowest indices and the result is deterministic.
template<int BLOCK_SIZE>
static __global__ void top_k_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        const top_k_radix_state * __restrict__ states,
        const int2 * __restrict__ block_counts,
        int ncols,
        int k,
        int blocks_per_row,
        int cols_per_block) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int begin = row_block * cols_per_block;
    const int end = min(begin + cols_per_block, ncols);
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    const top_k_radix_state state = states[row];
    __shared__ int scratch[BLOCK_SIZE / WARP_SIZE];

    const int2 before = top_k_radix_counts_before(block_counts + (size_t) row * blocks_per_row, row_block);
    int n_equal_before = before.y;
    int out = before.x + min(n_equal_before, state.rank);

    for (int col0 = begin; col0 < end; col0 += BLOCK_SIZE) {
        const int col = col0 + threadIdx.x;
        const uint32_t key = col < end ? top_k_float_to_ordered(row_src[col]) : 0;
        const bool is_equal = col < end && key == state.prefix;

        int n_equal;
        const int equal_rank = n_equal_before + top_k_block_exclusive_scan<BLOCK_SIZE>(is_equal, scratch, n_equal);
        const bool selected = col < end && (key > state.prefix || (is_equal && equal_rank < state.rank));

        int n_selected;
        const int pos = out + top_k_block_exclusive_scan<BLOCK_SIZE>(selected, scratch, n_selected);
        if (selected) {
            row_dst[pos] = col;
        }
        out += n_selected;
        n_equal_before += n_equal;
    }
}

static void top_k_radix_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = std::min((ncols + 1023) / 1024, 64);
    const int cols_per_block = GGML_PAD((ncols + blocks_per_row - 1) / blocks_per_row, BLOCK_SIZE);

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) nrows * blocks_per_row * NBINS);
    ggml_cuda_pool_alloc<int2> counts_alloc(pool, (size_t) nrows * blocks_per_row);
    top_k_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();
    int2 * counts = counts_alloc.get();

    top_k_radix_init<<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows, k);

    const dim3 row_grid(blocks_per_row * nrows);
    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
        top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, states, histograms, ncols, blocks_per_row, shift);
        top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
            <<<nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
    }

    top_k_radix_count<BLOCK_SIZE>
        <<<row_grid, BLOCK_SIZE, 0, stream>>>(src, states, counts, ncols, blocks_per_row, cols_per_block);
    top_k_radix_gather<BLOCK_SIZE>
        <<<row_grid, BLOCK_SIZE, 0, stream>>>(
            src, dst, states, counts, ncols, k, blocks_per_row, cols_per_block);
}

#ifdef GGML_CUDA_USE_CUB
// cub sorts rows of up to this many columns faster than radix selection
static constexpr int64_t TOP_K_SORT_MAX_NCOLS = 4096;
// below cub's 300-segment partition threshold, whose host sync stalls pipelined GPUs
static constexpr int64_t TOP_K_SORT_MAX_NROWS = 256;

static bool top_k_stream_is_capturing(cudaStream_t stream) {
#ifdef USE_CUDA_GRAPH
    cudaStreamCaptureStatus status;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
    return status != cudaStreamCaptureStatusNone;
#else
    GGML_UNUSED(stream);
    return false;
#endif  // USE_CUDA_GRAPH
}
#endif  // GGML_CUDA_USE_CUB

static bool top_k_use_radix(int64_t ncols, int64_t nrows, cudaStream_t stream) {
    if (ncols <= TOP_K_BITONIC_MAX_NCOLS) {
        return false;
    }
#ifdef GGML_CUDA_USE_CUB
    if (ncols > TOP_K_SORT_MAX_NCOLS || nrows > TOP_K_SORT_MAX_NROWS) {
        return true;
    }
    // under stream capture cub sorts several rows by segmented radix sort, which is slower than radix selection
    return nrows > 1 && top_k_stream_is_capturing(stream);
#else
    GGML_UNUSED(nrows);
    GGML_UNUSED(stream);
    return true;
#endif  // GGML_CUDA_USE_CUB
}

#endif  // GGML_CUDA_TOP_K_LONG_ROWS

// Sorts each row, by bitonic sort or by cub for long rows, and keeps its first k columns.
static void top_k_argsort_chunk(
        ggml_cuda_pool & pool, const float * src, int * tmp, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
#ifdef GGML_CUDA_USE_CUB
    if (ncols > TOP_K_BITONIC_MAX_NCOLS) {
        argsort_f32_i32_cuda_cub(pool, src, tmp, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    } else {
        argsort_f32_i32_cuda_bitonic(src, tmp, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    }
#else
    GGML_ASSERT(ncols <= TOP_K_BITONIC_MAX_NCOLS);
    argsort_f32_i32_cuda_bitonic(src, tmp, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
#endif  // GGML_CUDA_USE_CUB
    CUDA_CHECK(cudaMemcpy2DAsync(dst, k * sizeof(int), tmp, ncols * sizeof(int), k * sizeof(int), nrows,
                                 cudaMemcpyDeviceToDevice, stream));
}

// Sorts the rows in chunks, which bounds the temporary indices and keeps the row offsets of the sorts in int range.
static void top_k_argsort_cuda(
        ggml_cuda_pool & pool, const float * src, int * dst, int64_t ncols, int64_t nrows, int64_t k, size_t nb01,
        cudaStream_t stream) {
    const int64_t chunk_nrows = argsort_f32_i32_cuda_cub_chunk_nrows(nb01, nrows);
    ggml_cuda_pool_alloc<int> tmp_alloc(pool, ncols * chunk_nrows);
    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        const int iter_nrows = (int) std::min(chunk_nrows, nrows - i);
        top_k_argsort_chunk(pool, src + i * ncols, tmp_alloc.get(), dst + i * k, ncols, iter_nrows, k, stream);
    }
}

#endif // CUB_TOP_K_AVAILABLE

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#else  // CUB_TOP_K_AVAILABLE
#ifdef GGML_CUDA_TOP_K_LONG_ROWS
    if (top_k_use_radix(ncols, nrows, stream)) {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
        return;
    }
#endif  // GGML_CUDA_TOP_K_LONG_ROWS
    top_k_argsort_cuda(pool, src0_d, dst_d, ncols, nrows, k, src0->nb[1], stream);
#endif  // CUB_TOP_K_AVAILABLE
}
