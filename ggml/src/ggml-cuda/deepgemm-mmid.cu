#include "common.cuh"
#include "deepgemm-mmid.cuh"
#include "mmid.cuh"

#include <deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh>

#include <cmath>
#include <cstring>

namespace {

constexpr int DG_M_BLOCKED_MIN = 128;
constexpr int DG_M_BLOCKED_MID = 512;
constexpr int DG_M_BLOCKED_MAX = 2048;
constexpr int DG_TEST_GROUPS = 16;
constexpr int DG_EP_GROUPS   = 256;
constexpr int DG_QWEN_GROUPS = 512;
constexpr int DG_SCALE_K   = 128;
constexpr int DG_NUM_SMS   = 128;
constexpr int DG_CONTIG_ALIGNMENT = 128;

template<int n_threads>
__device__ float block_abs_max(float value) {
    constexpr int n_warps = n_threads / 32;
    __shared__ float warp_max[n_warps];
    __shared__ float result;

    value = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }

    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    if (lane == 0) {
        warp_max[warp] = value;
    }
    __syncthreads();

    if (warp == 0) {
        value = lane < n_warps ? warp_max[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
        }
        if (lane == 0) {
            result = value;
        }
    }
    __syncthreads();
    return result;
}

__device__ uint8_t fp32_to_e4m3(float value) {
    const __nv_fp8_e4m3 fp8(value);
    return fp8.__x;
}

__global__ void pack_quantize_activations(
        const float * x,
        const int32_t * ids,
        int32_t * counts,
        int32_t * assignment_rows,
        uint8_t * a_fp8,
        float * sfa,
        int k,
        int top_k,
        int n_tokens,
        int m_blocked,
        int64_t x_stride_slot,
        int64_t x_stride_token,
        int64_t ids_stride_token,
        int expert_offset,
        int n_groups) {
    const int assignment = blockIdx.x;
    if (assignment >= n_tokens * top_k) {
        return;
    }

    const int token = assignment / top_k;
    const int slot  = assignment % top_k;
    const int global_expert = ids[token * ids_stride_token + slot];
    const int routed_expert = global_expert - expert_offset;

    if (routed_expert < 0 || routed_expert >= n_groups) {
        if (threadIdx.x == 0) assignment_rows[assignment] = -1;
        return;
    }

    __shared__ int packed_row;
    if (threadIdx.x == 0) {
        const int row = atomicAdd(counts + routed_expert, 1);
        packed_row = routed_expert * m_blocked + row;
        assignment_rows[assignment] = packed_row;
    }
    __syncthreads();

    const int source_slot = x_stride_slot == 0 ? 0 : slot;
    const float * x_row = x + token * x_stride_token + source_slot * x_stride_slot;
    const int scale_k = k / DG_SCALE_K;
    const int expert = packed_row / m_blocked;
    const int row = packed_row % m_blocked;

    for (int kb = 0; kb < scale_k; ++kb) {
        const int col = kb * DG_SCALE_K + threadIdx.x;
        const float value = x_row[col];
        const float amax = fmaxf(block_abs_max<128>(value), 1.0e-4f);
        const float scale = amax / 448.0f;

        if (threadIdx.x == 0) {
            // DeepGEMM requires MN-major, TMA-aligned activation scales:
            // [group, scale_k, M] in physical memory.
            sfa[(expert * scale_k + kb) * m_blocked + row] = scale;
        }
        a_fp8[static_cast<size_t>(packed_row) * k + col] = fp32_to_e4m3(value / scale);
    }
}

__global__ void localize_expert_ids(
        const int32_t * ids,
        int32_t * local_ids,
        int assignments,
        int top_k,
        int64_t ids_stride_token,
        int expert_offset,
        int n_groups) {
    const int assignment = blockIdx.x*blockDim.x + threadIdx.x;
    if (assignment >= assignments) {
        return;
    }

    const int token = assignment/top_k;
    const int slot  = assignment % top_k;
    const int expert = ids[token*ids_stride_token + slot] - expert_offset;
    local_ids[assignment] = expert >= 0 && expert < n_groups ? expert : INT_MAX;
}

// DeepGEMM's contiguous grouped layout requires each expert segment to start
// on a 128-row boundary. The unpadded expert bounds come from GGML's existing
// MUL_MAT_ID sorting helper; this tiny serial scan only covers local experts.
__global__ void make_padded_expert_bounds(
        const int32_t * expert_bounds,
        int32_t * padded_bounds,
        int n_groups) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    int padded = 0;
    for (int expert = 0; expert < n_groups; ++expert) {
        padded_bounds[expert] = padded;
        const int count = expert_bounds[expert + 1] - expert_bounds[expert];
        padded += (count + DG_CONTIG_ALIGNMENT - 1) & -DG_CONTIG_ALIGNMENT;
    }
    padded_bounds[n_groups] = padded;
}

__global__ void expert_bounds_to_counts(
        const int32_t * expert_bounds,
        int32_t * counts,
        int n_groups) {
    const int expert = blockIdx.x*blockDim.x + threadIdx.x;
    if (expert >= n_groups) {
        return;
    }
    counts[expert] = expert_bounds[expert + 1] - expert_bounds[expert];
}

__global__ void make_padded_assignment_map(
        const int32_t * ids_dst,
        const int32_t * expert_bounds,
        const int32_t * padded_bounds,
        int32_t * assignment_rows,
        int32_t * m_indices,
        int n_groups) {
    const int expert = blockIdx.x;
    if (expert >= n_groups) {
        return;
    }

    const int src_begin = expert_bounds[expert];
    const int count = expert_bounds[expert + 1] - src_begin;
    const int dst_begin = padded_bounds[expert];
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        const int assignment = ids_dst[src_begin + i];
        const int row = dst_begin + i;
        assignment_rows[assignment] = row;
        m_indices[row] = expert;
    }
}

__global__ void pack_quantize_activations_contiguous(
        const float * x,
        const int32_t * assignment_rows,
        uint8_t * a_fp8,
        float * sfa,
        int k,
        int top_k,
        int assignments,
        int m_capacity,
        int64_t x_stride_token) {
    const int assignment = blockIdx.x;
    if (assignment >= assignments) {
        return;
    }

    const int packed_row = assignment_rows[assignment];
    if (packed_row < 0) {
        return;
    }

    const int token = assignment/top_k;
    const float * x_row = x + token*x_stride_token;
    const int scale_k = k/DG_SCALE_K;
    for (int kb = 0; kb < scale_k; ++kb) {
        const int col = kb*DG_SCALE_K + threadIdx.x;
        const float value = x_row[col];
        const float amax = fmaxf(block_abs_max<128>(value), 1.0e-4f);
        const float scale = amax/448.0f;

        if (threadIdx.x == 0) {
            // MN-major activation scales: [K/128, compact M].
            sfa[kb*m_capacity + packed_row] = scale;
        }
        a_fp8[static_cast<size_t>(packed_row)*k + col] = fp32_to_e4m3(value/scale);
    }
}

__global__ void swiglu_quantize_contiguous(
        const __nv_bfloat16 * gate_up,
        const int32_t * assignment_rows,
        uint8_t * activated_fp8,
        float * activated_scales,
        int n_ff,
        int assignments,
        int m_capacity) {
    const int assignment = blockIdx.x;
    if (assignment >= assignments) {
        return;
    }

    const int packed_row = assignment_rows[assignment];
    if (packed_row < 0) {
        return;
    }

    const int scale_k = n_ff/DG_SCALE_K;
    const __nv_bfloat16 * gate_up_row = gate_up + static_cast<size_t>(packed_row)*(2*n_ff);
    for (int kb = 0; kb < scale_k; ++kb) {
        const int col = kb*DG_SCALE_K + threadIdx.x;
        const float gate = static_cast<float>(gate_up_row[col]);
        const float up   = static_cast<float>(gate_up_row[n_ff + col]);
        const float value = gate/(1.0f + expf(-gate))*up;
        const float amax = fmaxf(block_abs_max<128>(value), 1.0e-4f);
        const float scale = amax/448.0f;

        if (threadIdx.x == 0) {
            activated_scales[kb*m_capacity + packed_row] = scale;
        }
        activated_fp8[static_cast<size_t>(packed_row)*n_ff + col] = fp32_to_e4m3(value/scale);
    }
}

__global__ void quantize_weights(
        const __nv_bfloat16 * weights,
        const int32_t * counts,
        uint8_t * b_fp8,
        float * sfb,
        int n,
        int k) {
    constexpr int n_threads = 256;
    constexpr int block_elems = DG_SCALE_K * DG_SCALE_K;

    const int scale_k = k / DG_SCALE_K;
    const int scale_n = n / DG_SCALE_K;
    int block = blockIdx.x;
    const int kb = block % scale_k;
    block /= scale_k;
    const int nb = block % scale_n;
    const int expert = block / scale_n;

    if (counts[expert] == 0) {
        return;
    }

    float local_max = 0.0f;
    for (int i = threadIdx.x; i < block_elems; i += n_threads) {
        const int row = nb * DG_SCALE_K + i / DG_SCALE_K;
        const int col = kb * DG_SCALE_K + i % DG_SCALE_K;
        const size_t index = (static_cast<size_t>(expert) * n + row) * k + col;
        local_max = fmaxf(local_max, fabsf(static_cast<float>(weights[index])));
    }

    const float amax = fmaxf(block_abs_max<n_threads>(local_max), 1.0e-4f);
    const float scale = amax / 448.0f;
    if (threadIdx.x == 0) {
        // SFB stays in DeepGEMM's contiguous [group, N/128, K/128]
        // block-scale layout.
        sfb[(expert * scale_n + nb) * scale_k + kb] = scale;
    }

    for (int i = threadIdx.x; i < block_elems; i += n_threads) {
        const int row = nb * DG_SCALE_K + i / DG_SCALE_K;
        const int col = kb * DG_SCALE_K + i % DG_SCALE_K;
        const size_t index = (static_cast<size_t>(expert) * n + row) * k + col;
        b_fp8[index] = fp32_to_e4m3(static_cast<float>(weights[index]) / scale);
    }
}

__global__ void scatter_bf16_output(
        const __nv_bfloat16 * packed,
        const int32_t * assignment_rows,
        float * dst,
        int n,
        int top_k,
        int n_tokens,
        int64_t dst_stride_slot,
        int64_t dst_stride_token) {
    const int assignment = blockIdx.x;
    if (assignment >= n_tokens * top_k) {
        return;
    }

    const int token = assignment / top_k;
    const int slot  = assignment % top_k;
    const int packed_row = assignment_rows[assignment];

    for (int col = threadIdx.x; col < n; col += blockDim.x) {
        dst[token * dst_stride_token + slot * dst_stride_slot + col] =
            static_cast<float>(packed[static_cast<size_t>(packed_row) * n + col]);
    }
}
__global__ void swiglu_quantize_packed(
        const __nv_bfloat16 * gate_up,
        const int32_t * assignment_rows,
        uint8_t * activated_fp8,
        float * activated_scales,
        int n_ff,
        int assignments,
        int m_blocked) {
    const int assignment = blockIdx.x;
    if (assignment >= assignments) {
        return;
    }

    const int packed_row = assignment_rows[assignment];
    if (packed_row < 0) {
        return;
    }

    const int scale_k = n_ff / DG_SCALE_K;
    const int expert = packed_row / m_blocked;
    const int row = packed_row % m_blocked;
    const __nv_bfloat16 * gate_up_row = gate_up + static_cast<size_t>(packed_row) * (2*n_ff);

    for (int kb = 0; kb < scale_k; ++kb) {
        const int col = kb*DG_SCALE_K + threadIdx.x;
        const float gate = static_cast<float>(gate_up_row[col]);
        const float up   = static_cast<float>(gate_up_row[n_ff + col]);
        const float value = gate/(1.0f + expf(-gate))*up;
        const float amax = fmaxf(block_abs_max<128>(value), 1.0e-4f);
        const float scale = amax/448.0f;

        if (threadIdx.x == 0) {
            activated_scales[(expert*scale_k + kb)*m_blocked + row] = scale;
        }
        activated_fp8[static_cast<size_t>(packed_row)*n_ff + col] = fp32_to_e4m3(value/scale);
    }
}

__global__ void reduce_weighted_packed(
        const __nv_bfloat16 * packed,
        const int32_t * assignment_rows,
        const float * weights,
        float * dst,
        int n_embd,
        int top_k,
        int n_tokens,
        int64_t weights_stride_slot,
        int64_t weights_stride_token) {
    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= n_embd*n_tokens) {
        return;
    }

    const int token = index/n_embd;
    const int col = index % n_embd;
    float sum = 0.0f;
    for (int slot = 0; slot < top_k; ++slot) {
        const int assignment = token*top_k + slot;
        const int packed_row = assignment_rows[assignment];
        if (packed_row >= 0) {
            const float route_weight = weights[token*weights_stride_token + slot*weights_stride_slot];
            sum += route_weight*static_cast<float>(packed[static_cast<size_t>(packed_row)*n_embd + col]);
        }
    }
    dst[index] = sum;
}


CUtensorMap make_tma_2d_desc(
        void * data,
        CUtensorMapDataType dtype,
        int element_size,
        int gmem_inner,
        int gmem_outer,
        int smem_inner,
        int smem_outer,
        int gmem_outer_stride,
        CUtensorMapSwizzle swizzle) {
    if (swizzle != CU_TENSOR_MAP_SWIZZLE_NONE) {
        const int swizzle_bytes =
            swizzle == CU_TENSOR_MAP_SWIZZLE_128B ? 128 :
            swizzle == CU_TENSOR_MAP_SWIZZLE_64B  ?  64 : 32;
        smem_inner = swizzle_bytes / element_size;
    }

    CUtensorMap map;
    const cuuint64_t gmem_dims[2] = {
        static_cast<cuuint64_t>(gmem_inner),
        static_cast<cuuint64_t>(gmem_outer),
    };
    const cuuint64_t gmem_strides[1] = {
        static_cast<cuuint64_t>(gmem_outer_stride * element_size),
    };
    const cuuint32_t smem_dims[2] = {
        static_cast<cuuint32_t>(smem_inner),
        static_cast<cuuint32_t>(smem_outer),
    };
    const cuuint32_t elem_strides[2] = { 1, 1 };

    CU_CHECK(cuTensorMapEncodeTiled(
        &map,
        dtype,
        2,
        data,
        gmem_dims,
        gmem_strides,
        smem_dims,
        elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return map;
}

template<int n, int k, int n_last_stages, int n_groups, int block_n = 160>
void launch_deepgemm(
        uint8_t * a_fp8,
        float * sfa,
        uint8_t * b_fp8,
        float * sfb,
        __nv_bfloat16 * packed_dst,
        int32_t * counts,
        int m_blocked,
        cudaStream_t stream) {
    constexpr int block_m = 64;
    constexpr int block_k = 128;
    constexpr int n_stages = 4;
    constexpr int swizzle_d = 64;
    constexpr int n_threads = 256;
    constexpr int sfb_rows = k / block_k;
    constexpr int sfb_factor = block_k % block_n == 0 ? 1 : 2;
    constexpr int smem_size =
        block_m * block_n * static_cast<int>(sizeof(__nv_bfloat16)) +
        n_stages * (block_m * block_k + block_n * block_k + block_m * static_cast<int>(sizeof(float))) +
        ((sfb_rows * sfb_factor * static_cast<int>(sizeof(float)) + 7) / 8) * 8 +
        n_stages * 2 * 8;
    static_assert(smem_size <= 232448);

    auto kernel = &deep_gemm::sm90_fp8_gemm_1d2d_impl<
        0, n, k,
        n_groups,
        block_m, block_n, block_k,
        swizzle_d,
        n_stages, n_last_stages,
        128, 128,
        1, false,
        DG_NUM_SMS, deep_gemm::GemmType::MGroupedMasked, deep_gemm::EpilogueIdentity>;

    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    CUtensorMap map_a = make_tma_2d_desc(
        a_fp8, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1,
        k, m_blocked * n_groups,
        block_k, block_m, k,
        CU_TENSOR_MAP_SWIZZLE_128B);
    CUtensorMap map_b = make_tma_2d_desc(
        b_fp8, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1,
        k, n * n_groups,
        block_k, block_n, k,
        CU_TENSOR_MAP_SWIZZLE_128B);
    CUtensorMap map_d = make_tma_2d_desc(
        packed_dst, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, sizeof(__nv_bfloat16),
        n, m_blocked * n_groups,
        block_n, block_m, n,
        CU_TENSOR_MAP_SWIZZLE_64B);
    CUtensorMap map_sfa = make_tma_2d_desc(
        sfa, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, sizeof(float),
        m_blocked, (k / DG_SCALE_K) * n_groups,
        block_m, 1, m_blocked,
        CU_TENSOR_MAP_SWIZZLE_NONE);

    uint32_t shape_m = m_blocked;
    uint32_t shape_n = n;
    uint32_t shape_k = k;
    float * kernel_sfb = sfb;
    int32_t * grouped_layout = counts;
    void * args[] = {
        &kernel_sfb,
        &grouped_layout,
        &shape_m,
        &shape_n,
        &shape_k,
        &map_a,
        &map_b,
        &map_d,
        &map_sfa,
    };

    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(DG_NUM_SMS, 1, 1);
    config.blockDim = dim3(n_threads, 1, 1);
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;
    CUDA_CHECK(cudaLaunchKernelExC(&config, reinterpret_cast<const void *>(kernel), args));
}

template<int n, int k, int n_last_stages, int n_groups, int block_n = 160>
void launch_deepgemm_contiguous(
        uint8_t * a_fp8,
        float * sfa,
        uint8_t * b_fp8,
        float * sfb,
        __nv_bfloat16 * packed_dst,
        int32_t * m_indices,
        int m,
        cudaStream_t stream) {
    constexpr int block_m = DG_CONTIG_ALIGNMENT;
    constexpr int block_k = 128;
    constexpr int n_stages = 4;
    constexpr int swizzle_d = 64;
    constexpr int n_threads = 256;
    constexpr int sfb_rows = k/block_k;
    constexpr int sfb_factor = block_k % block_n == 0 ? 1 : 2;
    constexpr int smem_size =
        block_m*block_n*static_cast<int>(sizeof(__nv_bfloat16)) +
        n_stages*(block_m*block_k + block_n*block_k + block_m*static_cast<int>(sizeof(float))) +
        ((sfb_rows*sfb_factor*static_cast<int>(sizeof(float)) + 7)/8)*8 +
        n_stages*2*8;
    static_assert(smem_size <= 232448);

    auto kernel = &deep_gemm::sm90_fp8_gemm_1d2d_impl<
        0, n, k,
        n_groups,
        block_m, block_n, block_k,
        swizzle_d,
        n_stages, n_last_stages,
        128, 128,
        1, false,
        DG_NUM_SMS, deep_gemm::GemmType::MGroupedContiguous, deep_gemm::EpilogueIdentity>;

    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    CUtensorMap map_a = make_tma_2d_desc(
        a_fp8, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1,
        k, m,
        block_k, block_m, k,
        CU_TENSOR_MAP_SWIZZLE_128B);
    CUtensorMap map_b = make_tma_2d_desc(
        b_fp8, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1,
        k, n*n_groups,
        block_k, block_n, k,
        CU_TENSOR_MAP_SWIZZLE_128B);
    CUtensorMap map_d = make_tma_2d_desc(
        packed_dst, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, sizeof(__nv_bfloat16),
        n, m,
        block_n, block_m, n,
        CU_TENSOR_MAP_SWIZZLE_64B);
    CUtensorMap map_sfa = make_tma_2d_desc(
        sfa, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, sizeof(float),
        m, k/DG_SCALE_K,
        block_m, 1, m,
        CU_TENSOR_MAP_SWIZZLE_NONE);

    uint32_t shape_m = m;
    uint32_t shape_n = n;
    uint32_t shape_k = k;
    float * kernel_sfb = sfb;
    int32_t * grouped_layout = m_indices;
    void * args[] = {
        &kernel_sfb,
        &grouped_layout,
        &shape_m,
        &shape_n,
        &shape_k,
        &map_a,
        &map_b,
        &map_d,
        &map_sfa,
    };

    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(DG_NUM_SMS, 1, 1);
    config.blockDim = dim3(n_threads, 1, 1);
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;
    CUDA_CHECK(cudaLaunchKernelExC(&config, reinterpret_cast<const void *>(kernel), args));
}

template<int n_groups>
void launch_qwen_gate_up_contiguous(
        uint8_t * a_fp8, float * sfa, uint8_t * b_fp8, float * sfb,
        __nv_bfloat16 * packed_dst, int32_t * m_indices,
        int n, int m, cudaStream_t stream) {
    if (n == 1280) {
        launch_deepgemm_contiguous<1280, 2560, 0, n_groups>(
            a_fp8, sfa, b_fp8, sfb, packed_dst, m_indices, m, stream);
    } else if (n == 512) {
        launch_deepgemm_contiguous<512, 2560, 0, n_groups, 128>(
            a_fp8, sfa, b_fp8, sfb, packed_dst, m_indices, m, stream);
    } else {
        launch_deepgemm_contiguous<768, 2560, 0, n_groups, 128>(
            a_fp8, sfa, b_fp8, sfb, packed_dst, m_indices, m, stream);
    }
}

template<int n_groups>
void launch_qwen_down_contiguous(
        uint8_t * a_fp8, float * sfa, uint8_t * b_fp8, float * sfb,
        __nv_bfloat16 * packed_dst, int32_t * m_indices,
        int k, int m, cudaStream_t stream) {
    if (k == 640) {
        launch_deepgemm_contiguous<2560, 640, 1, n_groups>(
            a_fp8, sfa, b_fp8, sfb, packed_dst, m_indices, m, stream);
    } else if (k == 256) {
        launch_deepgemm_contiguous<2560, 256, 2, n_groups>(
            a_fp8, sfa, b_fp8, sfb, packed_dst, m_indices, m, stream);
    } else {
        launch_deepgemm_contiguous<2560, 384, 3, n_groups>(
            a_fp8, sfa, b_fp8, sfb, packed_dst, m_indices, m, stream);
    }
}

template<int n_groups>
void launch_qwen_gate_up(
        uint8_t * a_fp8,
        float * sfa,
        uint8_t * b_fp8,
        float * sfb,
        __nv_bfloat16 * packed_dst,
        int32_t * counts,
        int n,
        int m_blocked,
        cudaStream_t stream) {
    if (n == 1280) {
        launch_deepgemm<1280, 2560, 0, n_groups>(
            a_fp8, sfa, b_fp8, sfb, packed_dst, counts, m_blocked, stream);
    } else if (n == 512) {
        launch_deepgemm<512, 2560, 0, n_groups, 128>(
            a_fp8, sfa, b_fp8, sfb, packed_dst, counts, m_blocked, stream);
    } else {
        launch_deepgemm<768, 2560, 0, n_groups, 128>(
            a_fp8, sfa, b_fp8, sfb, packed_dst, counts, m_blocked, stream);
    }
}

template<int n_groups>
void launch_qwen_down(
        uint8_t * a_fp8,
        float * sfa,
        uint8_t * b_fp8,
        float * sfb,
        __nv_bfloat16 * packed_dst,
        int32_t * counts,
        int k,
        int m_blocked,
        cudaStream_t stream) {
    if (k == 640) {
        launch_deepgemm<2560, 640, 1, n_groups>(
            a_fp8, sfa, b_fp8, sfb, packed_dst, counts, m_blocked, stream);
    } else if (k == 256) {
        launch_deepgemm<2560, 256, 2, n_groups>(
            a_fp8, sfa, b_fp8, sfb, packed_dst, counts, m_blocked, stream);
    } else {
        launch_deepgemm<2560, 384, 3, n_groups>(
            a_fp8, sfa, b_fp8, sfb, packed_dst, counts, m_blocked, stream);
    }
}

template<int n_groups>
void launch_qwen(
        uint8_t * a_fp8,
        float * sfa,
        uint8_t * b_fp8,
        float * sfb,
        __nv_bfloat16 * packed_dst,
        int32_t * counts,
        bool gate_up,
        int n,
        int k,
        int m_blocked,
        cudaStream_t stream) {
    if (gate_up) {
        launch_qwen_gate_up<n_groups>(
            a_fp8, sfa, b_fp8, sfb, packed_dst, counts, n, m_blocked, stream);
    } else {
        launch_qwen_down<n_groups>(
            a_fp8, sfa, b_fp8, sfb, packed_dst, counts, k, m_blocked, stream);
    }
}

bool runtime_enabled() {
    const char * value = std::getenv("GGML_CUDA_DEEPGEMM_MUL_MAT_ID");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

bool moe_ffn_runtime_enabled() {
    const char * value = std::getenv("GGML_CUDA_DEEPGEMM_MOE_FFN");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

bool moe_ffn_debug_enabled() {
    const char * value = std::getenv("GGML_CUDA_DEEPGEMM_DEBUG");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

bool mul_mat_id_supported(const ggml_tensor * dst, int device) {
    if (dst == nullptr || dst->src[0] == nullptr || dst->src[1] == nullptr || dst->src[2] == nullptr) {
        return false;
    }

    const ggml_tensor * weights = dst->src[0];
    const ggml_tensor * x       = dst->src[1];
    const ggml_tensor * ids     = dst->src[2];
    const ggml_tensor * scale   = dst->src[3];
    const bool native_fp8 = weights->type == GGML_TYPE_F8_E4M3;

    const int cc = ggml_cuda_info().devices[device].cc;
    if (cc != GGML_CUDA_CC_HOPPER ||
        (!native_fp8 && weights->type != GGML_TYPE_BF16) ||
        x->type != GGML_TYPE_F32 ||
        ids->type != GGML_TYPE_I32 ||
        dst->type != GGML_TYPE_F32) {
        return false;
    }

    const int k         = static_cast<int>(weights->ne[0]);
    const int n         = static_cast<int>(weights->ne[1]);
    const int n_groups  = static_cast<int>(weights->ne[2]);
    const int top_k     = static_cast<int>(ids->ne[0]);
    const int n_tokens  = static_cast<int>(ids->ne[1]);
    const int scale_k   = k / DG_SCALE_K;
    const int scale_n   = n / DG_SCALE_K;
    const int m_blocked = n_tokens <= DG_M_BLOCKED_MIN ? DG_M_BLOCKED_MIN :
                          n_tokens <= DG_M_BLOCKED_MID ? DG_M_BLOCKED_MID :
                          n_tokens <= DG_M_BLOCKED_MAX ? DG_M_BLOCKED_MAX : 0;

    const bool qwen_gate_up = k == 2560 && (n == 1280 || n == 512 || n == 768);
    const bool qwen_down    = n == 2560 && (k == 640 || k == 256 || k == 384);
    if ((!qwen_gate_up && !qwen_down) ||
        (n_groups != DG_TEST_GROUPS && n_groups != DG_EP_GROUPS && n_groups != DG_QWEN_GROUPS) ||
        top_k != 10 ||
        m_blocked == 0 ||
        weights->ne[3] != 1 ||
        x->ne[0] != k ||
        (x->ne[1] != 1 && x->ne[1] != top_k) ||
        x->ne[2] != n_tokens ||
        dst->ne[0] != n ||
        dst->ne[1] != top_k ||
        dst->ne[2] != n_tokens ||
        weights->nb[0] != ggml_type_size(weights->type) ||
        x->nb[0] != sizeof(float) ||
        ids->nb[0] != sizeof(int32_t) ||
        dst->nb[0] != sizeof(float) ||
        !ggml_is_contiguous(weights) ||
        !ggml_is_contiguous(x) ||
        !ggml_is_contiguous(dst)) {
        return false;
    }

    if (native_fp8 && (
            scale == nullptr ||
            scale->type != GGML_TYPE_F32 ||
            scale->ne[0] != scale_k ||
            scale->ne[1] != scale_n ||
            scale->ne[2] != n_groups ||
            scale->ne[3] != 1 ||
            !ggml_is_contiguous(scale))) {
        return false;
    }

    return true;
}

bool moe_ffn_supported(const ggml_tensor * dst, int device) {
    if (dst == nullptr) {
        return false;
    }
    for (int i = 0; i < 5; ++i) {
        if (dst->src[i] == nullptr) {
            return false;
        }
    }

    const ggml_tensor * gate_up       = dst->src[0];
    const ggml_tensor * down          = dst->src[1];
    const ggml_tensor * x             = dst->src[2];
    const ggml_tensor * ids           = dst->src[3];
    const ggml_tensor * weights       = dst->src[4];
    const ggml_tensor * gate_up_scale = dst->src[5];
    const ggml_tensor * down_scale    = dst->src[6];
    const bool native_fp8 = gate_up->type == GGML_TYPE_F8_E4M3;

    const auto reject = [&](const char * reason) {
        if (moe_ffn_debug_enabled()) {
            GGML_LOG_INFO(
                "%s: reject %-14s dev=%d gate_up=%s[%lld,%lld,%lld,%lld] down=%s[%lld,%lld,%lld,%lld] "
                "x=[%lld,%lld,%lld,%lld] ids=[%lld,%lld,%lld,%lld] weights=[%lld,%lld,%lld,%lld] "
                "dst=[%lld,%lld,%lld,%lld] contiguous=%d/%d/%d/%d/%d/%d offset=%d name=%s\n",
                __func__, reason, device,
                ggml_type_name(gate_up->type),
                (long long) gate_up->ne[0], (long long) gate_up->ne[1],
                (long long) gate_up->ne[2], (long long) gate_up->ne[3],
                ggml_type_name(down->type),
                (long long) down->ne[0], (long long) down->ne[1],
                (long long) down->ne[2], (long long) down->ne[3],
                (long long) x->ne[0], (long long) x->ne[1],
                (long long) x->ne[2], (long long) x->ne[3],
                (long long) ids->ne[0], (long long) ids->ne[1],
                (long long) ids->ne[2], (long long) ids->ne[3],
                (long long) weights->ne[0], (long long) weights->ne[1],
                (long long) weights->ne[2], (long long) weights->ne[3],
                (long long) dst->ne[0], (long long) dst->ne[1],
                (long long) dst->ne[2], (long long) dst->ne[3],
                ggml_is_contiguous(gate_up), ggml_is_contiguous(down),
                ggml_is_contiguous(x), ggml_is_contiguous(ids),
                ggml_is_contiguous(weights), ggml_is_contiguous(dst),
                ggml_get_op_params_i32(dst, 0), gate_up->name);
        }
        return false;
    };

    const int cc = ggml_cuda_info().devices[device].cc;
    if (cc != GGML_CUDA_CC_HOPPER ||
        gate_up->type != down->type ||
        (!native_fp8 && gate_up->type != GGML_TYPE_BF16) ||
        x->type != GGML_TYPE_F32 ||
        ids->type != GGML_TYPE_I32 ||
        weights->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32) {
        return reject("types/cc");
    }

    const int n_embd        = static_cast<int>(gate_up->ne[0]);
    const int gate_up_n     = static_cast<int>(gate_up->ne[1]);
    const int n_ff          = gate_up_n/2;
    const int n_groups      = static_cast<int>(gate_up->ne[2]);
    const int top_k         = static_cast<int>(ids->ne[0]);
    const int n_tokens      = static_cast<int>(ids->ne[1]);
    const int m_blocked     = n_tokens <= DG_M_BLOCKED_MIN ? DG_M_BLOCKED_MIN :
                              n_tokens <= DG_M_BLOCKED_MID ? DG_M_BLOCKED_MID :
                              n_tokens <= DG_M_BLOCKED_MAX ? DG_M_BLOCKED_MAX : 0;

    if (n_embd != 2560 ||
        (gate_up_n != 1280 && gate_up_n != 512 && gate_up_n != 768) ||
        (n_groups != DG_TEST_GROUPS && n_groups != DG_EP_GROUPS && n_groups != DG_QWEN_GROUPS) ||
        top_k != 10 || m_blocked == 0 ||
        gate_up->ne[3] != 1) {
        return reject("base shape");
    }
    if (down->ne[0] != n_ff || down->ne[1] != n_embd ||
        down->ne[2] != n_groups || down->ne[3] != 1) {
        return reject("down shape");
    }
    if (x->ne[0] != n_embd || x->ne[1] != 1 || x->ne[2] != n_tokens || x->ne[3] != 1) {
        return reject("x shape");
    }
    if (ids->ne[2] != 1 || ids->ne[3] != 1) {
        return reject("ids shape");
    }
    if (weights->ne[0] != 1 || weights->ne[1] != top_k ||
        weights->ne[2] != n_tokens || weights->ne[3] != 1) {
        return reject("weights shape");
    }
    if (dst->ne[0] != n_embd || dst->ne[1] != n_tokens) {
        return reject("dst shape");
    }
    if (gate_up->nb[0] != ggml_type_size(gate_up->type) ||
        down->nb[0] != ggml_type_size(down->type) ||
        x->nb[0] != sizeof(float) || ids->nb[0] != sizeof(int32_t) ||
        weights->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        return reject("element stride");
    }
    if (!ggml_is_contiguous(gate_up) || !ggml_is_contiguous(down) ||
        !ggml_is_contiguous(x) || !ggml_is_contiguous(ids) ||
        !ggml_is_contiguous(weights) || !ggml_is_contiguous(dst)) {
        return reject("contiguity");
    }

    if (native_fp8) {
        if (gate_up_scale == nullptr || down_scale == nullptr ||
            gate_up_scale->type != GGML_TYPE_F32 || down_scale->type != GGML_TYPE_F32 ||
            gate_up_scale->ne[0] != n_embd/DG_SCALE_K ||
            gate_up_scale->ne[1] != gate_up_n/DG_SCALE_K ||
            gate_up_scale->ne[2] != n_groups || gate_up_scale->ne[3] != 1 ||
            down_scale->ne[0] != n_ff/DG_SCALE_K ||
            down_scale->ne[1] != n_embd/DG_SCALE_K ||
            down_scale->ne[2] != n_groups || down_scale->ne[3] != 1 ||
            !ggml_is_contiguous(gate_up_scale) || !ggml_is_contiguous(down_scale)) {
            return reject("fp8 scales");
        }
    } else if (gate_up_scale != nullptr || down_scale != nullptr) {
        return reject("bf16 scales");
    }

    if (ggml_get_op_params_i32(dst, 0) < 0) {
        return reject("expert offset");
    }
    if (moe_ffn_debug_enabled()) {
        GGML_LOG_INFO("%s: accept dev=%d experts=%d tokens=%d offset=%d name=%s\n",
            __func__, device, n_groups, n_tokens, ggml_get_op_params_i32(dst, 0), gate_up->name);
    }
    return true;
}

} // namespace

bool ggml_cuda_deepgemm_mul_mat_id_graph_compatible(const ggml_tensor * dst, int device) {
    return dst != nullptr &&
           dst->src[0] != nullptr &&
           dst->src[0]->type == GGML_TYPE_F8_E4M3 &&
           mul_mat_id_supported(dst, device);
}

bool ggml_cuda_deepgemm_moe_ffn_supported(const ggml_tensor * dst, int device) {
    return moe_ffn_supported(dst, device) &&
        (dst->src[0]->type == GGML_TYPE_F8_E4M3 || moe_ffn_runtime_enabled());
}

bool ggml_cuda_deepgemm_moe_ffn_graph_compatible(const ggml_tensor * dst, int device) {
    return dst != nullptr &&
           dst->src[0] != nullptr &&
           dst->src[0]->type == GGML_TYPE_F8_E4M3 &&
           moe_ffn_supported(dst, device);
}

bool ggml_cuda_deepgemm_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * weights = dst->src[0];
    const ggml_tensor * x       = dst->src[1];
    const ggml_tensor * ids     = dst->src[2];
    const ggml_tensor * scale   = dst->src[3];
    const bool native_fp8 = weights->type == GGML_TYPE_F8_E4M3;

    // The BF16 path is retained as an opt-in correctness test. Native FP8
    // weights have no fallback implementation, so selecting their GGUF type
    // opts into DeepGEMM automatically.
    if (!native_fp8 && !runtime_enabled()) {
        return false;
    }

    if (!mul_mat_id_supported(dst, ctx.device)) {
        return false;
    }

    const int k         = static_cast<int>(weights->ne[0]);
    const int n         = static_cast<int>(weights->ne[1]);
    const int n_groups  = static_cast<int>(weights->ne[2]);
    const int top_k     = static_cast<int>(ids->ne[0]);
    const int n_tokens  = static_cast<int>(ids->ne[1]);
    const int scale_k   = k / DG_SCALE_K;
    const int scale_n   = n / DG_SCALE_K;
    const int m_blocked = n_tokens <= DG_M_BLOCKED_MIN ? DG_M_BLOCKED_MIN :
                          n_tokens <= DG_M_BLOCKED_MID ? DG_M_BLOCKED_MID : DG_M_BLOCKED_MAX;
    const bool qwen_gate_up = k == 2560;

    static bool logged = false;
    if (!logged) {
        GGML_LOG_INFO("%s: using experimental DeepGEMM SM90 MUL_MAT_ID path (%s weights, %d experts)\n",
            __func__, native_fp8 ? "persistent FP8" : "dynamically quantized BF16", n_groups);
        logged = true;
    }

    const int assignments = n_tokens * top_k;
    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool & pool = ctx.pool();

    ggml_cuda_pool_alloc<uint8_t> a_fp8(pool, static_cast<size_t>(n_groups) * m_blocked * k);
    ggml_cuda_pool_alloc<float> sfa(pool, static_cast<size_t>(n_groups) * m_blocked * scale_k);
    ggml_cuda_pool_alloc<uint8_t> quantized_weights(pool);
    ggml_cuda_pool_alloc<float> quantized_weight_scales(pool);

    uint8_t * b_fp8 = native_fp8
        ? static_cast<uint8_t *>(weights->data)
        : quantized_weights.alloc(static_cast<size_t>(n_groups) * n * k);
    float * sfb = native_fp8
        ? static_cast<float *>(scale->data)
        : quantized_weight_scales.alloc(static_cast<size_t>(n_groups) * scale_n * scale_k);

    ggml_cuda_pool_alloc<__nv_bfloat16> packed_dst(pool, static_cast<size_t>(n_groups) * m_blocked * n);
    ggml_cuda_pool_alloc<int32_t> counts(pool, n_groups);
    ggml_cuda_pool_alloc<int32_t> assignment_rows(pool, assignments);

    CUDA_CHECK(cudaMemsetAsync(counts.ptr, 0, n_groups * sizeof(int32_t), stream));

    const int64_t x_stride_slot =
        x->ne[1] == 1 ? 0 : static_cast<int64_t>(x->nb[1] / sizeof(float));
    pack_quantize_activations<<<assignments, 128, 0, stream>>>(
        static_cast<const float *>(x->data),
        static_cast<const int32_t *>(ids->data),
        counts.ptr,
        assignment_rows.ptr,
        a_fp8.ptr,
        sfa.ptr,
        k,
        top_k,
        n_tokens,
        m_blocked,
        x_stride_slot,
        static_cast<int64_t>(x->nb[2] / sizeof(float)),
        static_cast<int64_t>(ids->nb[1] / sizeof(int32_t)),
        0,
        n_groups);
    CUDA_CHECK(cudaGetLastError());

    if (!native_fp8) {
        quantize_weights<<<n_groups * scale_n * scale_k, 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16 *>(weights->data),
            counts.ptr,
            b_fp8,
            sfb,
            n,
            k);
        CUDA_CHECK(cudaGetLastError());
    }

    if (n_groups == DG_TEST_GROUPS) {
        launch_qwen<DG_TEST_GROUPS>(
            a_fp8.ptr, sfa.ptr, b_fp8, sfb, packed_dst.ptr, counts.ptr,
            qwen_gate_up, n, k, m_blocked, stream);
    } else if (n_groups == DG_EP_GROUPS) {
        launch_qwen<DG_EP_GROUPS>(
            a_fp8.ptr, sfa.ptr, b_fp8, sfb, packed_dst.ptr, counts.ptr,
            qwen_gate_up, n, k, m_blocked, stream);
    } else {
        launch_qwen<DG_QWEN_GROUPS>(
            a_fp8.ptr, sfa.ptr, b_fp8, sfb, packed_dst.ptr, counts.ptr,
            qwen_gate_up, n, k, m_blocked, stream);
    }

    scatter_bf16_output<<<assignments, 256, 0, stream>>>(
        packed_dst.ptr,
        assignment_rows.ptr,
        static_cast<float *>(dst->data),
        n,
        top_k,
        n_tokens,
        static_cast<int64_t>(dst->nb[1] / sizeof(float)),
        static_cast<int64_t>(dst->nb[2] / sizeof(float)));
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_cuda_deepgemm_moe_ffn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * gate_up       = dst->src[0];
    const ggml_tensor * down          = dst->src[1];
    const ggml_tensor * x             = dst->src[2];
    const ggml_tensor * ids           = dst->src[3];
    const ggml_tensor * weights       = dst->src[4];
    const ggml_tensor * gate_up_scale = dst->src[5];
    const ggml_tensor * down_scale    = dst->src[6];
    const bool native_fp8 = gate_up->type == GGML_TYPE_F8_E4M3;

    if (!native_fp8 && !moe_ffn_runtime_enabled()) {
        return false;
    }
    if (!moe_ffn_supported(dst, ctx.device)) {
        return false;
    }
    if (moe_ffn_debug_enabled()) {
        GGML_LOG_INFO("%s: CUDA execute dev=%d tokens=%lld experts=%lld offset=%d name=%s\n",
            __func__, ctx.device, (long long) dst->src[3]->ne[1], (long long) dst->src[0]->ne[2],
            ggml_get_op_params_i32(dst, 0), dst->src[0]->name);
    }

    const int n_embd        = static_cast<int>(gate_up->ne[0]);
    const int gate_up_n     = static_cast<int>(gate_up->ne[1]);
    const int n_ff          = gate_up_n/2;
    const int n_groups      = static_cast<int>(gate_up->ne[2]);
    const int top_k         = static_cast<int>(ids->ne[0]);
    const int n_tokens      = static_cast<int>(ids->ne[1]);
    const int assignments   = n_tokens*top_k;
    const int m_blocked     = n_tokens <= DG_M_BLOCKED_MIN ? DG_M_BLOCKED_MIN :
                              n_tokens <= DG_M_BLOCKED_MID ? DG_M_BLOCKED_MID : DG_M_BLOCKED_MAX;
    const int expert_offset = ggml_get_op_params_i32(dst, 0);

    static bool logged = false;
    if (!logged) {
        GGML_LOG_INFO("%s: using experimental fused DeepGEMM SM90 MoE FFN path (%s weights, %d experts)\n",
            __func__, native_fp8 ? "persistent FP8" : "dynamically quantized BF16", n_groups);
        logged = true;
    }

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool & pool = ctx.pool();

    // For prefill, use DeepGEMM's M-grouped contiguous layout. This mirrors
    // upstream GGML_MOE_FFN's expert-sorted dataflow and avoids reserving
    // n_groups*n_tokens activation and output rows. Decode keeps the masked
    // layout below because it schedules only populated experts under graphs.
    if (n_tokens >= DG_CONTIG_ALIGNMENT) {
        const int m_capacity =
            ((assignments + n_groups*(DG_CONTIG_ALIGNMENT - 1) + DG_CONTIG_ALIGNMENT - 1)/
             DG_CONTIG_ALIGNMENT)*DG_CONTIG_ALIGNMENT;

        ggml_cuda_pool_alloc<uint8_t> activations_fp8(
            pool, static_cast<size_t>(m_capacity)*n_embd);
        ggml_cuda_pool_alloc<float> activation_scales(
            pool, static_cast<size_t>(m_capacity)*(n_embd/DG_SCALE_K));
        ggml_cuda_pool_alloc<__nv_bfloat16> packed(
            pool, static_cast<size_t>(m_capacity)*n_embd);

        ggml_cuda_pool_alloc<uint8_t> quantized_weights(pool);
        ggml_cuda_pool_alloc<float> quantized_weight_scales(pool);
        uint8_t * gate_up_fp8 = native_fp8
            ? static_cast<uint8_t *>(gate_up->data)
            : quantized_weights.alloc(static_cast<size_t>(n_groups)*gate_up_n*n_embd);
        float * gate_up_sfb = native_fp8
            ? static_cast<float *>(gate_up_scale->data)
            : quantized_weight_scales.alloc(static_cast<size_t>(n_groups)*
                (gate_up_n/DG_SCALE_K)*(n_embd/DG_SCALE_K));

        ggml_cuda_pool_alloc<int32_t> local_ids(pool, assignments);
        ggml_cuda_pool_alloc<int32_t> ids_src1(pool, assignments);
        ggml_cuda_pool_alloc<int32_t> ids_dst(pool, assignments);
        ggml_cuda_pool_alloc<int32_t> expert_bounds(pool, n_groups + 1);
        ggml_cuda_pool_alloc<int32_t> padded_bounds(pool, n_groups + 1);
        ggml_cuda_pool_alloc<int32_t> assignment_rows(pool, assignments);
        ggml_cuda_pool_alloc<int32_t> m_indices(pool, m_capacity);
        ggml_cuda_pool_alloc<int32_t> counts(pool, n_groups);

        CUDA_CHECK(cudaMemsetAsync(assignment_rows.ptr, 0xFF,
            assignments*sizeof(int32_t), stream));
        CUDA_CHECK(cudaMemsetAsync(m_indices.ptr, 0xFF,
            m_capacity*sizeof(int32_t), stream));

        localize_expert_ids<<<(assignments + 255)/256, 256, 0, stream>>>(
            static_cast<const int32_t *>(ids->data), local_ids.ptr,
            assignments, top_k, static_cast<int64_t>(ids->nb[1]/sizeof(int32_t)),
            expert_offset, n_groups);
        CUDA_CHECK(cudaGetLastError());

        ggml_cuda_launch_mm_ids_helper(
            local_ids.ptr, ids_src1.ptr, ids_dst.ptr, expert_bounds.ptr,
            n_groups, n_tokens, top_k, 1, top_k, 1, false, stream);
        CUDA_CHECK(cudaGetLastError());

        expert_bounds_to_counts<<<(n_groups + 255)/256, 256, 0, stream>>>(
            expert_bounds.ptr, counts.ptr, n_groups);
        make_padded_expert_bounds<<<1, 1, 0, stream>>>(
            expert_bounds.ptr, padded_bounds.ptr, n_groups);
        make_padded_assignment_map<<<n_groups, 128, 0, stream>>>(
            ids_dst.ptr, expert_bounds.ptr, padded_bounds.ptr,
            assignment_rows.ptr, m_indices.ptr, n_groups);
        CUDA_CHECK(cudaGetLastError());

        pack_quantize_activations_contiguous<<<assignments, 128, 0, stream>>>(
            static_cast<const float *>(x->data), assignment_rows.ptr,
            activations_fp8.ptr, activation_scales.ptr,
            n_embd, top_k, assignments, m_capacity,
            static_cast<int64_t>(x->nb[2]/sizeof(float)));
        CUDA_CHECK(cudaGetLastError());

        if (!native_fp8) {
            quantize_weights<<<n_groups*(gate_up_n/DG_SCALE_K)*(n_embd/DG_SCALE_K), 256, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16 *>(gate_up->data), counts.ptr,
                gate_up_fp8, gate_up_sfb, gate_up_n, n_embd);
            CUDA_CHECK(cudaGetLastError());
        }

        if (n_groups == DG_TEST_GROUPS) {
            launch_qwen_gate_up_contiguous<DG_TEST_GROUPS>(
                activations_fp8.ptr, activation_scales.ptr,
                gate_up_fp8, gate_up_sfb,
                packed.ptr, m_indices.ptr, gate_up_n, m_capacity, stream);
        } else if (n_groups == DG_EP_GROUPS) {
            launch_qwen_gate_up_contiguous<DG_EP_GROUPS>(
                activations_fp8.ptr, activation_scales.ptr,
                gate_up_fp8, gate_up_sfb,
                packed.ptr, m_indices.ptr, gate_up_n, m_capacity, stream);
        } else {
            launch_qwen_gate_up_contiguous<DG_QWEN_GROUPS>(
                activations_fp8.ptr, activation_scales.ptr,
                gate_up_fp8, gate_up_sfb,
                packed.ptr, m_indices.ptr, gate_up_n, m_capacity, stream);
        }

        swiglu_quantize_contiguous<<<assignments, 128, 0, stream>>>(
            packed.ptr, assignment_rows.ptr, activations_fp8.ptr, activation_scales.ptr,
            n_ff, assignments, m_capacity);
        CUDA_CHECK(cudaGetLastError());

        uint8_t * down_fp8 = native_fp8
            ? static_cast<uint8_t *>(down->data)
            : quantized_weights.ptr;
        float * down_sfb = native_fp8
            ? static_cast<float *>(down_scale->data)
            : quantized_weight_scales.ptr;
        if (!native_fp8) {
            quantize_weights<<<n_groups*(n_embd/DG_SCALE_K)*(n_ff/DG_SCALE_K), 256, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat16 *>(down->data), counts.ptr,
                down_fp8, down_sfb, n_embd, n_ff);
            CUDA_CHECK(cudaGetLastError());
        }

        if (n_groups == DG_TEST_GROUPS) {
            launch_qwen_down_contiguous<DG_TEST_GROUPS>(
                activations_fp8.ptr, activation_scales.ptr,
                down_fp8, down_sfb,
                packed.ptr, m_indices.ptr, n_ff, m_capacity, stream);
        } else if (n_groups == DG_EP_GROUPS) {
            launch_qwen_down_contiguous<DG_EP_GROUPS>(
                activations_fp8.ptr, activation_scales.ptr,
                down_fp8, down_sfb,
                packed.ptr, m_indices.ptr, n_ff, m_capacity, stream);
        } else {
            launch_qwen_down_contiguous<DG_QWEN_GROUPS>(
                activations_fp8.ptr, activation_scales.ptr,
                down_fp8, down_sfb,
                packed.ptr, m_indices.ptr, n_ff, m_capacity, stream);
        }

        reduce_weighted_packed<<<(n_embd*n_tokens + 255)/256, 256, 0, stream>>>(
            packed.ptr, assignment_rows.ptr, static_cast<const float *>(weights->data),
            static_cast<float *>(dst->data), n_embd, top_k, n_tokens,
            static_cast<int64_t>(weights->nb[1]/sizeof(float)),
            static_cast<int64_t>(weights->nb[2]/sizeof(float)));
        CUDA_CHECK(cudaGetLastError());
        return true;
    }

    // These buffers are deliberately reused between the two GEMMs. CUDA stream
    // ordering makes the gate/up result available to SwiGLU before the input is
    // overwritten with the activated FP8 values, and likewise permits the down
    // weights to replace dynamically quantized gate/up weights.
    ggml_cuda_pool_alloc<uint8_t> activations_fp8(
        pool, static_cast<size_t>(n_groups)*m_blocked*n_embd);
    ggml_cuda_pool_alloc<float> activation_scales(
        pool, static_cast<size_t>(n_groups)*m_blocked*(n_embd/DG_SCALE_K));
    ggml_cuda_pool_alloc<__nv_bfloat16> packed(
        pool, static_cast<size_t>(n_groups)*m_blocked*n_embd);
    ggml_cuda_pool_alloc<uint8_t> quantized_weights(pool);
    ggml_cuda_pool_alloc<float> quantized_weight_scales(pool);

    uint8_t * gate_up_fp8 = native_fp8
        ? static_cast<uint8_t *>(gate_up->data)
        : quantized_weights.alloc(static_cast<size_t>(n_groups)*gate_up_n*n_embd);
    float * gate_up_sfb = native_fp8
        ? static_cast<float *>(gate_up_scale->data)
        : quantized_weight_scales.alloc(static_cast<size_t>(n_groups)*
            (gate_up_n/DG_SCALE_K)*(n_embd/DG_SCALE_K));
    ggml_cuda_pool_alloc<int32_t> counts(pool, n_groups);
    ggml_cuda_pool_alloc<int32_t> assignment_rows(pool, assignments);

    CUDA_CHECK(cudaMemsetAsync(counts.ptr, 0, n_groups*sizeof(int32_t), stream));
    pack_quantize_activations<<<assignments, 128, 0, stream>>>(
        static_cast<const float *>(x->data),
        static_cast<const int32_t *>(ids->data),
        counts.ptr,
        assignment_rows.ptr,
        activations_fp8.ptr,
        activation_scales.ptr,
        n_embd,
        top_k,
        n_tokens,
        m_blocked,
        0,
        static_cast<int64_t>(x->nb[2]/sizeof(float)),
        static_cast<int64_t>(ids->nb[1]/sizeof(int32_t)),
        expert_offset,
        n_groups);
    CUDA_CHECK(cudaGetLastError());

    if (!native_fp8) {
        quantize_weights<<<n_groups*(gate_up_n/DG_SCALE_K)*(n_embd/DG_SCALE_K), 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16 *>(gate_up->data), counts.ptr,
            gate_up_fp8, gate_up_sfb, gate_up_n, n_embd);
        CUDA_CHECK(cudaGetLastError());
    }

    if (n_groups == DG_TEST_GROUPS) {
        launch_qwen_gate_up<DG_TEST_GROUPS>(activations_fp8.ptr, activation_scales.ptr,
            gate_up_fp8, gate_up_sfb, packed.ptr, counts.ptr, gate_up_n, m_blocked, stream);
    } else if (n_groups == DG_EP_GROUPS) {
        launch_qwen_gate_up<DG_EP_GROUPS>(activations_fp8.ptr, activation_scales.ptr,
            gate_up_fp8, gate_up_sfb, packed.ptr, counts.ptr, gate_up_n, m_blocked, stream);
    } else {
        launch_qwen_gate_up<DG_QWEN_GROUPS>(activations_fp8.ptr, activation_scales.ptr,
            gate_up_fp8, gate_up_sfb, packed.ptr, counts.ptr, gate_up_n, m_blocked, stream);
    }

    swiglu_quantize_packed<<<assignments, 128, 0, stream>>>(
        packed.ptr, assignment_rows.ptr, activations_fp8.ptr, activation_scales.ptr,
        n_ff, assignments, m_blocked);
    CUDA_CHECK(cudaGetLastError());

    uint8_t * down_fp8 = native_fp8
        ? static_cast<uint8_t *>(down->data)
        : quantized_weights.ptr;
    float * down_sfb = native_fp8
        ? static_cast<float *>(down_scale->data)
        : quantized_weight_scales.ptr;

    if (!native_fp8) {
        quantize_weights<<<n_groups*(n_embd/DG_SCALE_K)*(n_ff/DG_SCALE_K), 256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16 *>(down->data), counts.ptr,
            down_fp8, down_sfb, n_embd, n_ff);
        CUDA_CHECK(cudaGetLastError());
    }

    if (n_groups == DG_TEST_GROUPS) {
        launch_qwen_down<DG_TEST_GROUPS>(activations_fp8.ptr, activation_scales.ptr,
            down_fp8, down_sfb, packed.ptr, counts.ptr, n_ff, m_blocked, stream);
    } else if (n_groups == DG_EP_GROUPS) {
        launch_qwen_down<DG_EP_GROUPS>(activations_fp8.ptr, activation_scales.ptr,
            down_fp8, down_sfb, packed.ptr, counts.ptr, n_ff, m_blocked, stream);
    } else {
        launch_qwen_down<DG_QWEN_GROUPS>(activations_fp8.ptr, activation_scales.ptr,
            down_fp8, down_sfb, packed.ptr, counts.ptr, n_ff, m_blocked, stream);
    }

    reduce_weighted_packed<<<(n_embd*n_tokens + 255)/256, 256, 0, stream>>>(
        packed.ptr, assignment_rows.ptr, static_cast<const float *>(weights->data),
        static_cast<float *>(dst->data), n_embd, top_k, n_tokens,
        static_cast<int64_t>(weights->nb[1]/sizeof(float)),
        static_cast<int64_t>(weights->nb[2]/sizeof(float)));
    CUDA_CHECK(cudaGetLastError());
    return true;
}
