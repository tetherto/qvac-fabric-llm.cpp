#include "common.cuh"
#include "deepgemm-mmid.cuh"
#include "mmid.cuh"

#include <deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh>
#if __has_include(<deep_gemm/epilogue/transform.cuh>)
#include <deep_gemm/epilogue/transform.cuh>
#endif
#include <cmath>
#include <cstring>

namespace {

#if __has_include(<deep_gemm/epilogue/transform.cuh>)
using dg_epilogue_identity = deep_gemm::epilogue::transform::EpilogueIdentity;
#else
using dg_epilogue_identity = deep_gemm::EpilogueIdentity;
#endif

constexpr int DG_M_BLOCKED_MIN = 128;
constexpr int DG_M_BLOCKED_MID = 512;
constexpr int DG_M_BLOCKED_MAX = 2048;
constexpr int DG_TEST_GROUPS = 16;
constexpr int DG_EP_GROUPS   = 256;
constexpr int DG_QWEN_GROUPS = 512;
constexpr int DG_SCALE_K   = 128;
constexpr int DG_NUM_SMS   = 128;

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

__device__ float2 e4m3x2_to_fp32x2(uint16_t value) {
    __nv_fp8x2_e4m3 fp8;
    fp8.__x = value;
    return static_cast<float2>(fp8);
}

__device__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

constexpr int DG_GEMV_WARPS = 8;

// DeepGEMM's persistent grouped kernel is designed for substantially more M
// than batch-one expert routing provides. In the hybrid TP/EP graph each rank
// normally owns only about five of the ten selected routes, and each local
// expert has a single row. Read the persistent native-FP8 weights directly as
// coalesced GEMVs instead of packing those rows into an M=128 grouped layout.
__global__ void fp8_moe_gate_up_swiglu_b1(
        const uint8_t * __restrict__ weights,
        const float * __restrict__ weight_scales,
        const float * __restrict__ x,
        const int32_t * __restrict__ ids,
        float * __restrict__ activated,
        int n_embd,
        int n_ff,
        int top_k,
        int expert_offset,
        int n_groups) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row  = blockIdx.x * DG_GEMV_WARPS + warp;
    const int slot = blockIdx.y;
    if (row >= n_ff || slot >= top_k) {
        return;
    }

    const int expert = ids[slot] - expert_offset;
    if (expert < 0 || expert >= n_groups) {
        if (lane == 0) {
            activated[slot * n_ff + row] = 0.0f;
        }
        return;
    }

    const int scale_k = n_embd / DG_SCALE_K;
    const int scale_n = (2 * n_ff) / DG_SCALE_K;
    const size_t expert_base = static_cast<size_t>(expert) * (2 * n_ff) * n_embd;
    const size_t gate_base   = expert_base + static_cast<size_t>(row) * n_embd;
    const size_t up_base     = expert_base + static_cast<size_t>(n_ff + row) * n_embd;

    float gate = 0.0f;
    float up   = 0.0f;
#pragma unroll
    for (int kb = 0; kb < 20; ++kb) {
        if (kb >= scale_k) {
            break;
        }
        const float gate_scale = weight_scales[
            (static_cast<size_t>(expert) * scale_n + row / DG_SCALE_K) * scale_k + kb];
        const float up_scale = weight_scales[
            (static_cast<size_t>(expert) * scale_n + (n_ff + row) / DG_SCALE_K) * scale_k + kb];
#pragma unroll
        for (int i = 0; i < DG_SCALE_K / 64; ++i) {
            const int col = kb * DG_SCALE_K + i * 64 + 2 * lane;
            const float2 xv = *reinterpret_cast<const float2 *>(x + col);
            const float2 gate_weight = e4m3x2_to_fp32x2(
                *reinterpret_cast<const uint16_t *>(weights + gate_base + col));
            const float2 up_weight = e4m3x2_to_fp32x2(
                *reinterpret_cast<const uint16_t *>(weights + up_base + col));
            gate = fmaf(gate_weight.x * gate_scale, xv.x, gate);
            gate = fmaf(gate_weight.y * gate_scale, xv.y, gate);
            up   = fmaf(up_weight.x * up_scale, xv.x, up);
            up   = fmaf(up_weight.y * up_scale, xv.y, up);
        }
    }

    gate = warp_sum(gate);
    up   = warp_sum(up);
    if (lane == 0) {
        activated[slot * n_ff + row] = (gate / (1.0f + expf(-gate))) * up;
    }
}

// Fuse the down GEMVs across all locally-owned routes with router weighting.
__global__ void fp8_moe_down_reduce_b1(
        const uint8_t * __restrict__ weights,
        const float * __restrict__ weight_scales,
        const float * __restrict__ activated,
        const int32_t * __restrict__ ids,
        const float * __restrict__ route_weights,
        float * __restrict__ dst,
        int n_embd,
        int n_ff,
        int top_k,
        int expert_offset,
        int n_groups) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row  = blockIdx.x * DG_GEMV_WARPS + warp;
    if (row >= n_embd) {
        return;
    }

    const int scale_k = n_ff / DG_SCALE_K;
    const int scale_n = n_embd / DG_SCALE_K;
    float result = 0.0f;
    for (int slot = 0; slot < top_k; ++slot) {
        const int expert = ids[slot] - expert_offset;
        if (expert < 0 || expert >= n_groups) {
            continue;
        }

        const size_t row_base = (static_cast<size_t>(expert) * n_embd + row) * n_ff;
        float dot = 0.0f;
#pragma unroll
        for (int kb = 0; kb < 5; ++kb) {
            if (kb >= scale_k) {
                break;
            }
            const float weight_scale = weight_scales[
                (static_cast<size_t>(expert) * scale_n + row / DG_SCALE_K) * scale_k + kb];
#pragma unroll
            for (int i = 0; i < DG_SCALE_K / 64; ++i) {
                const int col = kb * DG_SCALE_K + i * 64 + 2 * lane;
                const float2 weight = e4m3x2_to_fp32x2(
                    *reinterpret_cast<const uint16_t *>(weights + row_base + col));
                const float2 value = *reinterpret_cast<const float2 *>(
                    activated + slot * n_ff + col);
                dot = fmaf(weight.x * weight_scale, value.x, dot);
                dot = fmaf(weight.y * weight_scale, value.y, dot);
            }
        }
        dot = warp_sum(dot);
        if (lane == 0) {
            result = fmaf(route_weights[slot], dot, result);
        }
    }

    if (lane == 0) {
        dst[row] = result;
    }
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

// Batch-1 gate/up receives one activation row broadcast over all selected
// experts. Quantize each 128-element K block once, then fan it out to the
// expert-local rows expected by DeepGEMM. The generic kernel above instead
// repeats the quantization once per assignment.
__global__ void pack_quantize_broadcast_b1(
        const float * x,
        const int32_t * ids,
        int32_t * counts,
        int32_t * assignment_rows,
        uint8_t * a_fp8,
        float * sfa,
        int k,
        int top_k,
        int m_blocked,
        int expert_offset,
        int n_groups,
        const float * route_weights,
        float * route_weights_copy) {
    const int kb = blockIdx.x;
    const int col = kb * DG_SCALE_K + threadIdx.x;
    if (kb == 0 && threadIdx.x < top_k && route_weights != nullptr) {
        route_weights_copy[threadIdx.x] = route_weights[threadIdx.x];
    }
    const float value = x[col];
    const float amax = fmaxf(block_abs_max<128>(value), 1.0e-4f);
    const float scale = amax / 448.0f;
    const uint8_t quantized = fp32_to_e4m3(value / scale);
    const int scale_k = k / DG_SCALE_K;

    for (int slot = 0; slot < top_k; ++slot) {
        const int routed_expert = ids[slot] - expert_offset;
        if (routed_expert < 0 || routed_expert >= n_groups) {
            if (kb == 0 && threadIdx.x == 0) {
                assignment_rows[slot] = -1;
            }
            continue;
        }

        // TOP_K normally produces unique IDs. Retain correct MUL_MAT_ID
        // semantics for duplicate IDs by assigning successive rows.
        int row = 0;
        for (int previous = 0; previous < slot; ++previous) {
            row += ids[previous] - expert_offset == routed_expert;
        }
        const int packed_row = routed_expert * m_blocked + row;

        if (threadIdx.x == 0) {
            sfa[(routed_expert * scale_k + kb) * m_blocked + row] = scale;
            if (kb == 0) {
                assignment_rows[slot] = packed_row;
                counts[routed_expert] = row + 1;
            }
        }
        a_fp8[static_cast<size_t>(packed_row) * k + col] = quantized;
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
        dst[token * dst_stride_token + slot * dst_stride_slot + col] = packed_row >= 0
            ? static_cast<float>(packed[static_cast<size_t>(packed_row) * n + col])
            : 0.0f;
    }
}

__global__ void weighted_reduce_bf16_output_b1(
        const __nv_bfloat16 * packed,
        const int32_t * assignment_rows,
        const float * weights,
        float * dst,
        int n,
        int top_k) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) {
        return;
    }

    float sum = 0.0f;
    for (int slot = 0; slot < top_k; ++slot) {
        const int packed_row = assignment_rows[slot];
        if (packed_row >= 0) {
            sum += static_cast<float>(packed[static_cast<size_t>(packed_row) * n + col]) * weights[slot];
        }
    }
    dst[col] = sum;
}

// Consume the packed BF16 gate/up result in expert-row order and produce the
// FP8 activation/scales expected by the down projection. This avoids gate/up
// scatter, a separate F32 SwiGLU intermediate, and a second route/pack pass.
__global__ void swiglu_quantize_packed_b1(
        const __nv_bfloat16 * gate_up,
        const int32_t * assignment_rows,
        uint8_t * activated_fp8,
        float * activated_scales,
        int n_ff,
        int top_k,
        int m_blocked) {
    const int slot = blockIdx.x;
    if (slot >= top_k) {
        return;
    }

    const int packed_row = assignment_rows[slot];
    if (packed_row < 0) {
        return;
    }

    const int scale_k = n_ff / DG_SCALE_K;
    const int expert = packed_row / m_blocked;
    const int row = packed_row % m_blocked;
    const __nv_bfloat16 * gate_up_row =
        gate_up + static_cast<size_t>(packed_row) * (2*n_ff);

    for (int kb = 0; kb < scale_k; ++kb) {
        const int col = kb * DG_SCALE_K + threadIdx.x;
        const float gate = static_cast<float>(gate_up_row[col]);
        const float up   = static_cast<float>(gate_up_row[n_ff + col]);
        const float value = gate / (1.0f + expf(-gate)) * up;
        const float amax = fmaxf(block_abs_max<128>(value), 1.0e-4f);
        const float scale = amax / 448.0f;

        if (threadIdx.x == 0) {
            activated_scales[(expert * scale_k + kb) * m_blocked + row] = scale;
        }
        activated_fp8[static_cast<size_t>(packed_row) * n_ff + col] =
            fp32_to_e4m3(value / scale);
    }
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

#if __has_include(<deep_gemm/epilogue/transform.cuh>)
    auto kernel = &deep_gemm::sm90_fp8_gemm_1d2d_impl<
        cute::UMMA::Major::K,
        0, n, k,
        n_groups,
        block_m, block_n, block_k,
        128, 128, swizzle_d,
        n_stages,
        128, 128,
        1, false,
        DG_NUM_SMS, deep_gemm::GemmType::MGroupedMasked,
        cutlass::bfloat16_t, dg_epilogue_identity>;
#else
    auto kernel = &deep_gemm::sm90_fp8_gemm_1d2d_impl<
        0, n, k,
        n_groups,
        block_m, block_n, block_k,
        swizzle_d,
        n_stages, n_last_stages,
        128, 128,
        1, false,
        DG_NUM_SMS, deep_gemm::GemmType::MGroupedMasked, dg_epilogue_identity>;
#endif

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

bool broadcast_b1_pack_enabled() {
    const char * value = std::getenv("GGML_CUDA_DEEPGEMM_B1_PACK");
    return value == nullptr || value[0] == '\0' || std::strcmp(value, "0") != 0;
}

bool moe_ffn_b1_enabled() {
    const char * value = std::getenv("GGML_CUDA_DEEPGEMM_B1_FFN");
    return value == nullptr || value[0] == '\0' || std::strcmp(value, "0") != 0;
}

bool moe_ffn_b1_gemv_enabled() {
    const char * value = std::getenv("GGML_CUDA_FP8_MOE_GEMV");
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
    const int expert_offset = ggml_get_op_params_i32(dst, 2);
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
        !ggml_is_contiguous(dst) ||
        expert_offset < 0 || expert_offset + n_groups > DG_QWEN_GROUPS) {
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

} // namespace

bool ggml_cuda_deepgemm_mul_mat_id_graph_compatible(const ggml_tensor * dst, int device) {
    return dst != nullptr &&
           dst->src[0] != nullptr &&
           dst->src[0]->type == GGML_TYPE_F8_E4M3 &&
           mul_mat_id_supported(dst, device);
}

static bool ggml_cuda_deepgemm_mul_mat_id_impl(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        const ggml_tensor * reduction_weights,
        ggml_tensor * reduced_dst) {
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
    const int expert_offset = ggml_get_op_params_i32(dst, 2);
    const int top_k     = static_cast<int>(ids->ne[0]);
    const int n_tokens  = static_cast<int>(ids->ne[1]);
    const int scale_k   = k / DG_SCALE_K;
    const int scale_n   = n / DG_SCALE_K;
    const int m_blocked = n_tokens <= DG_M_BLOCKED_MIN ? DG_M_BLOCKED_MIN :
                          n_tokens <= DG_M_BLOCKED_MID ? DG_M_BLOCKED_MID : DG_M_BLOCKED_MAX;
    const bool qwen_gate_up = k == 2560;

    const bool fused_reduction = reduction_weights != nullptr || reduced_dst != nullptr;
    if (fused_reduction && (
            reduction_weights == nullptr || reduced_dst == nullptr || qwen_gate_up || n_tokens != 1 ||
            reduction_weights->type != GGML_TYPE_F32 || !ggml_is_contiguous(reduction_weights) ||
            reduction_weights->ne[0] != 1 || reduction_weights->ne[1] != top_k ||
            reduction_weights->ne[2] != 1 || reduction_weights->ne[3] != 1 ||
            reduced_dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(reduced_dst) ||
            reduced_dst->ne[0] != n || reduced_dst->ne[1] != 1 ||
            reduced_dst->ne[2] != 1 || reduced_dst->ne[3] != 1)) {
        return false;
    }

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
    if (n_tokens == 1 && x_stride_slot == 0 && broadcast_b1_pack_enabled()) {
        pack_quantize_broadcast_b1<<<scale_k, 128, 0, stream>>>(
            static_cast<const float *>(x->data),
            static_cast<const int32_t *>(ids->data),
            counts.ptr,
            assignment_rows.ptr,
            a_fp8.ptr,
            sfa.ptr,
            k,
            top_k,
            m_blocked,
            expert_offset,
            n_groups,
            nullptr,
            nullptr);
    } else {
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
            expert_offset,
            n_groups);
    }
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

    if (fused_reduction) {
        weighted_reduce_bf16_output_b1<<<(n + 255) / 256, 256, 0, stream>>>(
            packed_dst.ptr,
            assignment_rows.ptr,
            static_cast<const float *>(reduction_weights->data),
            static_cast<float *>(reduced_dst->data),
            n,
            top_k);
    } else {
        scatter_bf16_output<<<assignments, 256, 0, stream>>>(
            packed_dst.ptr,
            assignment_rows.ptr,
            static_cast<float *>(dst->data),
            n,
            top_k,
            n_tokens,
            static_cast<int64_t>(dst->nb[1] / sizeof(float)),
            static_cast<int64_t>(dst->nb[2] / sizeof(float)));
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_cuda_deepgemm_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    return ggml_cuda_deepgemm_mul_mat_id_impl(ctx, dst, nullptr, nullptr);
}

bool ggml_cuda_deepgemm_mul_mat_id_weighted_reduction(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        const ggml_tensor * weights,
        ggml_tensor * reduced_dst) {
    const char * value = std::getenv("GGML_CUDA_DEEPGEMM_B1_REDUCE");
    if (value != nullptr && value[0] != '\0' && std::strcmp(value, "0") == 0) {
        return false;
    }

    static bool logged = false;
    const bool handled = ggml_cuda_deepgemm_mul_mat_id_impl(ctx, dst, weights, reduced_dst);
    if (handled && !logged) {
        GGML_LOG_INFO("%s: fusing batch-1 down scatter with weighted expert reduction\n", __func__);
        logged = true;
    }
    return handled;
}

bool ggml_cuda_deepgemm_moe_ffn_b1(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * gate_up_dst,
        ggml_tensor * down_dst,
        const ggml_tensor * reduction_weights,
        ggml_tensor * reduced_dst) {
    if (!moe_ffn_b1_enabled() ||
            !mul_mat_id_supported(gate_up_dst, ctx.device) ||
            !mul_mat_id_supported(down_dst, ctx.device)) {
        return false;
    }

    const ggml_tensor * gate_up       = gate_up_dst->src[0];
    const ggml_tensor * x             = gate_up_dst->src[1];
    const ggml_tensor * ids           = gate_up_dst->src[2];
    const ggml_tensor * gate_up_scale = gate_up_dst->src[3];
    const ggml_tensor * down          = down_dst->src[0];
    const ggml_tensor * down_scale    = down_dst->src[3];
    const bool native_fp8 = gate_up->type == GGML_TYPE_F8_E4M3;

    if (!native_fp8 && !runtime_enabled()) {
        return false;
    }

    const int n_embd        = static_cast<int>(gate_up->ne[0]);
    const int gate_up_n     = static_cast<int>(gate_up->ne[1]);
    const int n_ff          = gate_up_n / 2;
    const int n_groups      = static_cast<int>(gate_up->ne[2]);
    const int top_k         = static_cast<int>(ids->ne[0]);
    const int n_tokens      = static_cast<int>(ids->ne[1]);
    const int m_blocked     = DG_M_BLOCKED_MIN;
    const int expert_offset = ggml_get_op_params_i32(gate_up_dst, 2);

    if (n_tokens != 1 || x->ne[1] != 1 || gate_up_n != 2*n_ff ||
            down->type != gate_up->type || down->ne[0] != n_ff ||
            down->ne[1] != n_embd || down->ne[2] != n_groups || down->ne[3] != 1 ||
            down_dst->src[2] != ids ||
            ggml_get_op_params_i32(down_dst, 2) != expert_offset ||
            reduction_weights == nullptr || reduction_weights->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(reduction_weights) ||
            reduction_weights->ne[0] != 1 || reduction_weights->ne[1] != top_k ||
            reduction_weights->ne[2] != 1 || reduction_weights->ne[3] != 1 ||
            reduced_dst == nullptr || reduced_dst->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(reduced_dst) ||
            reduced_dst->ne[0] != n_embd || reduced_dst->ne[1] != 1 ||
            reduced_dst->ne[2] != 1 || reduced_dst->ne[3] != 1) {
        return false;
    }

    if (native_fp8 && (gate_up_scale == nullptr || down_scale == nullptr)) {
        return false;
    }

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool & pool = ctx.pool();

    if (native_fp8 && moe_ffn_b1_gemv_enabled() &&
            n_embd == 2560 && n_ff == 640 && top_k == 10 &&
            (n_groups == DG_EP_GROUPS || n_groups == DG_QWEN_GROUPS)) {
        ggml_cuda_pool_alloc<float> activated(pool, static_cast<size_t>(top_k) * n_ff);
        const dim3 gate_up_grid((n_ff + DG_GEMV_WARPS - 1) / DG_GEMV_WARPS, top_k, 1);
        fp8_moe_gate_up_swiglu_b1<<<gate_up_grid, DG_GEMV_WARPS * 32, 0, stream>>>(
            static_cast<const uint8_t *>(gate_up->data),
            static_cast<const float *>(gate_up_scale->data),
            static_cast<const float *>(x->data),
            static_cast<const int32_t *>(ids->data),
            activated.ptr,
            n_embd, n_ff, top_k, expert_offset, n_groups);
        CUDA_CHECK(cudaGetLastError());

        const dim3 down_grid((n_embd + DG_GEMV_WARPS - 1) / DG_GEMV_WARPS, 1, 1);
        fp8_moe_down_reduce_b1<<<down_grid, DG_GEMV_WARPS * 32, 0, stream>>>(
            static_cast<const uint8_t *>(down->data),
            static_cast<const float *>(down_scale->data),
            activated.ptr,
            static_cast<const int32_t *>(ids->data),
            static_cast<const float *>(reduction_weights->data),
            static_cast<float *>(reduced_dst->data),
            n_embd, n_ff, top_k, expert_offset, n_groups);
        CUDA_CHECK(cudaGetLastError());

        static bool gemv_logged = false;
        if (!gemv_logged) {
            GGML_LOG_INFO(
                "%s: using native-FP8 batch-1 gate/up+SwiGLU and down+reduce GEMVs (%d experts)\n",
                __func__, n_groups);
            gemv_logged = true;
        }
        return true;
    }

    const size_t activation_elems = static_cast<size_t>(n_groups) * m_blocked * n_embd;
    const size_t activation_scale_elems =
        static_cast<size_t>(n_groups) * m_blocked * (n_embd / DG_SCALE_K);
    ggml_cuda_pool_alloc<uint8_t> activations_fp8(pool, activation_elems);
    ggml_cuda_pool_alloc<float> activation_scales(pool, activation_scale_elems);
    ggml_cuda_pool_alloc<__nv_bfloat16> packed(
        pool, static_cast<size_t>(n_groups) * m_blocked * n_embd);
    ggml_cuda_pool_alloc<int32_t> counts(pool, n_groups);
    ggml_cuda_pool_alloc<int32_t> assignment_rows(pool, top_k);
    ggml_cuda_pool_alloc<float> route_weights_copy(pool, top_k);

    ggml_cuda_pool_alloc<uint8_t> quantized_weights(pool);
    ggml_cuda_pool_alloc<float> quantized_weight_scales(pool);
    uint8_t * gate_up_fp8 = native_fp8
        ? static_cast<uint8_t *>(gate_up->data)
        : quantized_weights.alloc(static_cast<size_t>(n_groups) * gate_up_n * n_embd);
    float * gate_up_sfb = native_fp8
        ? static_cast<float *>(gate_up_scale->data)
        : quantized_weight_scales.alloc(static_cast<size_t>(n_groups) *
            (gate_up_n / DG_SCALE_K) * (n_embd / DG_SCALE_K));

    CUDA_CHECK(cudaMemsetAsync(counts.ptr, 0, n_groups * sizeof(int32_t), stream));
    pack_quantize_broadcast_b1<<<n_embd / DG_SCALE_K, 128, 0, stream>>>(
        static_cast<const float *>(x->data),
        static_cast<const int32_t *>(ids->data),
        counts.ptr,
        assignment_rows.ptr,
        activations_fp8.ptr,
        activation_scales.ptr,
        n_embd,
        top_k,
        m_blocked,
        expert_offset,
        n_groups,
        static_cast<const float *>(reduction_weights->data),
        route_weights_copy.ptr);
    CUDA_CHECK(cudaGetLastError());

    if (!native_fp8) {
        quantize_weights<<<n_groups * (gate_up_n / DG_SCALE_K) * (n_embd / DG_SCALE_K),
                256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16 *>(gate_up->data), counts.ptr,
            gate_up_fp8, gate_up_sfb, gate_up_n, n_embd);
        CUDA_CHECK(cudaGetLastError());
    }

    if (n_groups == DG_TEST_GROUPS) {
        launch_qwen_gate_up<DG_TEST_GROUPS>(
            activations_fp8.ptr, activation_scales.ptr, gate_up_fp8, gate_up_sfb,
            packed.ptr, counts.ptr, gate_up_n, m_blocked, stream);
    } else if (n_groups == DG_EP_GROUPS) {
        launch_qwen_gate_up<DG_EP_GROUPS>(
            activations_fp8.ptr, activation_scales.ptr, gate_up_fp8, gate_up_sfb,
            packed.ptr, counts.ptr, gate_up_n, m_blocked, stream);
    } else {
        launch_qwen_gate_up<DG_QWEN_GROUPS>(
            activations_fp8.ptr, activation_scales.ptr, gate_up_fp8, gate_up_sfb,
            packed.ptr, counts.ptr, gate_up_n, m_blocked, stream);
    }

    swiglu_quantize_packed_b1<<<top_k, 128, 0, stream>>>(
        packed.ptr, assignment_rows.ptr, activations_fp8.ptr, activation_scales.ptr,
        n_ff, top_k, m_blocked);
    CUDA_CHECK(cudaGetLastError());

    uint8_t * down_fp8 = native_fp8
        ? static_cast<uint8_t *>(down->data)
        : quantized_weights.ptr;
    float * down_sfb = native_fp8
        ? static_cast<float *>(down_scale->data)
        : quantized_weight_scales.ptr;
    if (!native_fp8) {
        quantize_weights<<<n_groups * (n_embd / DG_SCALE_K) * (n_ff / DG_SCALE_K),
                256, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16 *>(down->data), counts.ptr,
            down_fp8, down_sfb, n_embd, n_ff);
        CUDA_CHECK(cudaGetLastError());
    }

    if (n_groups == DG_TEST_GROUPS) {
        launch_qwen_down<DG_TEST_GROUPS>(
            activations_fp8.ptr, activation_scales.ptr, down_fp8, down_sfb,
            packed.ptr, counts.ptr, n_ff, m_blocked, stream);
    } else if (n_groups == DG_EP_GROUPS) {
        launch_qwen_down<DG_EP_GROUPS>(
            activations_fp8.ptr, activation_scales.ptr, down_fp8, down_sfb,
            packed.ptr, counts.ptr, n_ff, m_blocked, stream);
    } else {
        launch_qwen_down<DG_QWEN_GROUPS>(
            activations_fp8.ptr, activation_scales.ptr, down_fp8, down_sfb,
            packed.ptr, counts.ptr, n_ff, m_blocked, stream);
    }

    weighted_reduce_bf16_output_b1<<<(n_embd + 255) / 256, 256, 0, stream>>>(
        packed.ptr, assignment_rows.ptr,
        route_weights_copy.ptr,
        static_cast<float *>(reduced_dst->data), n_embd, top_k);
    CUDA_CHECK(cudaGetLastError());

    static bool logged = false;
    if (!logged) {
        GGML_LOG_INFO(
            "%s: fusing batch-1 gate/up, SwiGLU, down, and weighted reduction (%d experts)\n",
            __func__, n_groups);
        logged = true;
    }
    return true;
}
