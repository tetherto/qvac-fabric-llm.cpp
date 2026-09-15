#include "argsort.cuh"
#include "top-k.cuh"

#include <cstdlib>
#include <cstring>

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

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

namespace {

constexpr int CUDA_RADIX_TOP_K_THREADS = 1024;
constexpr int CUDA_RADIX_TOP_K_BINS    = 256;

// Map IEEE-754 values to monotonically ordered unsigned keys. QSA scores are
// finite or +/- infinity; TOP_K does not define a useful ordering for NaNs.
static __device__ __forceinline__ uint32_t radix_top_k_key(float value) {
    const uint32_t bits = __float_as_uint(value);
    return bits & 0x80000000U ? ~bits : bits | 0x80000000U;
}

// Fixed-width radix selection for the Qwen4Exp QSA block selector. A single
// block rescans the input for each byte of the exact F32 threshold, then emits
// every greater key and enough threshold ties to produce exactly k values.
// This trades five coalesced reads of a small score row for the global scratch
// and repeated launches of a full device sort.
template<int k>
__global__ __launch_bounds__(CUDA_RADIX_TOP_K_THREADS, 1) void radix_top_k_f32(
        const float * src,
        int32_t * dst,
        int ncols) {
    const int row = blockIdx.x;
    src += static_cast<size_t>(row) * ncols;
    dst += static_cast<size_t>(row) * k;

    if (ncols <= k) {
        for (int col = threadIdx.x; col < k; col += blockDim.x) {
            dst[col] = col < ncols ? col : -1;
        }
        return;
    }

    __shared__ int histogram[2][CUDA_RADIX_TOP_K_BINS];
    __shared__ uint32_t prefix;
    __shared__ uint32_t prefix_mask;
    __shared__ int target_rank;
    __shared__ int output_count;
    __shared__ int tie_count;

    if (threadIdx.x == 0) {
        prefix        = 0;
        prefix_mask   = 0;
        target_rank   = k;
    }
    __syncthreads();

#pragma unroll
    for (int pass = 0; pass < 4; ++pass) {
        if (threadIdx.x < CUDA_RADIX_TOP_K_BINS) {
            histogram[0][threadIdx.x] = 0;
        }
        __syncthreads();

        const int shift = 24 - 8*pass;
        const uint32_t current_prefix = prefix;
        const uint32_t current_mask   = prefix_mask;
        for (int col = threadIdx.x; col < ncols; col += blockDim.x) {
            const uint32_t key = radix_top_k_key(src[col]);
            if ((key & current_mask) == current_prefix) {
                atomicAdd(&histogram[0][(key >> shift) & 0xffU], 1);
            }
        }
        __syncthreads();

        // Inclusive suffix sum: histogram[0][bin] becomes the number of
        // candidates whose current byte is >= bin.
        int ping = 0;
#pragma unroll
        for (int offset = 1; offset < CUDA_RADIX_TOP_K_BINS; offset *= 2) {
            if (threadIdx.x < CUDA_RADIX_TOP_K_BINS) {
                int count = histogram[ping][threadIdx.x];
                if (threadIdx.x + offset < CUDA_RADIX_TOP_K_BINS) {
                    count += histogram[ping][threadIdx.x + offset];
                }
                histogram[ping ^ 1][threadIdx.x] = count;
            }
            __syncthreads();
            ping ^= 1;
        }

        if (threadIdx.x == 0) {
            const int rank = target_rank;
            for (int bin = CUDA_RADIX_TOP_K_BINS - 1; bin >= 0; --bin) {
                const int inclusive = histogram[ping][bin];
                const int greater = bin + 1 < CUDA_RADIX_TOP_K_BINS
                    ? histogram[ping][bin + 1]
                    : 0;
                if (inclusive >= rank && greater < rank) {
                    target_rank  = rank - greater;
                    prefix      |= static_cast<uint32_t>(bin) << shift;
                    prefix_mask |= 0xffU << shift;
                    break;
                }
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output_count = 0;
        tie_count    = 0;
    }
    __syncthreads();

    const uint32_t threshold = prefix;
    const int ties_needed = target_rank;
    for (int col = threadIdx.x; col < ncols; col += blockDim.x) {
        const uint32_t key = radix_top_k_key(src[col]);
        bool selected = key > threshold;
        if (key == threshold) {
            selected = atomicAdd(&tie_count, 1) < ties_needed;
        }
        if (selected) {
            const int out = atomicAdd(&output_count, 1);
            if (out < k) {
                dst[out] = col;
            }
        }
    }
}

static bool use_radix_top_k(int device, int64_t ncols, int64_t k) {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_CUDA_RADIX_TOP_K");
        return value == nullptr || value[0] == '\0' || std::strcmp(value, "0") != 0;
    }();

    // One CTA is faster than the device-wide sort for QSA-sized block rows.
    // Keep very large rows on CUB, which can exploit multiple SMs.
    return enabled && ggml_cuda_info().devices[device].cc == GGML_CUDA_CC_HOPPER &&
        (k == 512 || k == 513) && ncols > k && ncols <= 65536;
}

} // namespace

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
    if (use_radix_top_k(ctx.device, ncols, k)) {
        if (k == 512) {
            radix_top_k_f32<512><<<nrows, CUDA_RADIX_TOP_K_THREADS, 0, stream>>>(src0_d, dst_d, ncols);
        } else {
            radix_top_k_f32<513><<<nrows, CUDA_RADIX_TOP_K_THREADS, 0, stream>>>(src0_d, dst_d, ncols);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
    int *                     tmp_dst = temp_dst_alloc.get();
    argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                 cudaMemcpyDeviceToDevice, stream));
#endif
}
