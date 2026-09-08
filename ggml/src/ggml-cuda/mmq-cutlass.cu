#include "mmq-cutlass.cuh"

#include <climits>

static_assert(ggml_cuda_cutlass_result_can_fallback(ggml_cuda_cutlass_result::fallback));
static_assert(!ggml_cuda_cutlass_result_can_fallback(ggml_cuda_cutlass_result::success));
static_assert(!ggml_cuda_cutlass_result_can_fallback(ggml_cuda_cutlass_result::failure));

bool ggml_cuda_repacked_mul_mat_supported(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (!ggml_cuda_cutlass_weight_supported(src0) || src1 == nullptr || dst == nullptr ||
        src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        src0->ne[2] != 1 || src0->ne[3] != 1 ||
        src0->ne[0] <= 0 || src0->ne[0] > INT_MAX - 127 ||
        src0->ne[1] <= 0 || src0->ne[1] > INT_MAX ||
        src1->ne[0] != src0->ne[0] ||
        !ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const int64_t n_elements = ggml_nelements(src1);
    const int64_t n_rows     = n_elements / src0->ne[0];
    return n_elements == n_rows * src0->ne[0] &&
        n_rows > 0 && n_rows <= INT_MAX &&
        dst->ne[0] == src0->ne[1] &&
        ggml_nelements(dst) == n_rows * src0->ne[1];
}

#ifdef GGML_CUDA_CUTLASS

#    include <cuda_fp8.h>

#    include "quantize.cuh"

#    include "cute/tensor.hpp"
#    include "cutlass/cutlass.h"
#    include "cutlass/detail/sm100_blockscaled_layout.hpp"
#    include "cutlass/epilogue/collective/collective_builder.hpp"
#    include "cutlass/gemm/collective/collective_builder.hpp"
#    include "cutlass/gemm/device/gemm_universal_adapter.h"
#    include "cutlass/gemm/kernel/gemm_universal.hpp"
#    include "cutlass/util/packed_stride.hpp"

static __device__ __forceinline__ uint8_t cutlass_mxfp8_scale(float amax) {
    if (!(amax > 0.0f)) {
        return 0;
    }

    constexpr float e4m3_max = 448.0f;
    const int exponent = __float2int_ru(log2f(amax / e4m3_max));
    return (uint8_t) max(0, min(254, exponent + 127));
}

static __device__ __forceinline__ float cutlass_half_warp_amax(float value0, float value1) {
    float amax = fmaxf(fabsf(value0), fabsf(value1));
#    pragma unroll
    for (int mask = 8; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, 16));
    }
    return amax;
}

static __global__ void cutlass_quantize_mxfp8(
        const float * __restrict__ src,
        uint8_t * __restrict__ dst,
        uint8_t * __restrict__ scales,
        int64_t n_cols,
        int64_t n_cols_padded,
        int64_t stride_row) {
    constexpr int warps = 8;
    const int row       = blockIdx.x;
    const int warp      = threadIdx.x / WARP_SIZE;
    const int lane      = threadIdx.x % WARP_SIZE;
    const int half      = lane / 16;
    const int pair_lane = lane % 16;
    const int scale_blocks      = n_cols_padded / WARP_SIZE;
    const int scale_block_pairs = scale_blocks / 2;

    for (int pair = warp; pair < scale_block_pairs; pair += warps) {
        const int scale_block = 2 * pair + half;
        const int64_t k       = (int64_t) scale_block * WARP_SIZE + pair_lane * 2;
        float2 value          = { 0.0f, 0.0f };
        if (k + 1 < n_cols) {
            value = *reinterpret_cast<const float2 *>(src + (int64_t) row * stride_row + k);
        } else {
            if (k < n_cols) {
                value.x = src[(int64_t) row * stride_row + k];
            }
        }

        const float amax         = cutlass_half_warp_amax(value.x, value.y);
        const uint8_t scale      = cutlass_mxfp8_scale(amax);
        const float inv_scale    = amax == 0.0f ? 0.0f : __frcp_rn(ggml_cuda_e8m0_to_fp32(scale));
        const __nv_fp8_e4m3 q0(value.x * inv_scale);
        const __nv_fp8_e4m3 q1(value.y * inv_scale);
        *reinterpret_cast<uint16_t *>(dst + (int64_t) row * n_cols_padded + k) =
            (uint16_t) q0.__x | ((uint16_t) q1.__x << 8);
        if (pair_lane == 0) {
            scales[ggml_cuda_cutlass_blockscaled_scale_offset(row, scale_block, scale_blocks)] = scale;
        }
    }
}

static __global__ void cutlass_repack_nvfp4_activation(
        const block_fp4_mmq * __restrict__ src,
        uint8_t * __restrict__ dst,
        uint8_t * __restrict__ scales,
        int64_t n_rows,
        int64_t n_cols_padded) {
    const int64_t index = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t scale_blocks = n_cols_padded / QK_NVFP4_SUB;
    if (index >= n_rows * scale_blocks) {
        return;
    }

    const int64_t row = index / scale_blocks;
    const int64_t subblock = index % scale_blocks;
    const int64_t k = subblock * QK_NVFP4_SUB;
    const block_fp4_mmq & block = src[(k / QK_FP4_MMQ) * n_rows + row];
    const int sub = (k % QK_FP4_MMQ) / QK_NVFP4_SUB;
    const uint8_t * values = reinterpret_cast<const uint8_t *>(block.qs) + sub * QK_NVFP4_SUB / 2;
    uint8_t * output = dst + row * (n_cols_padded / 2) + k / 2;
#    pragma unroll
    for (int i = 0; i < QK_NVFP4_SUB / 2; ++i) {
        const int element = 2 * i;
        const uint8_t lo = (values[element % 8] >> (4 * (element / 8))) & 0x0F;
        const uint8_t hi = (values[(element + 1) % 8] >> (4 * ((element + 1) / 8))) & 0x0F;
        output[i] = lo | (hi << 4);
    }
    scales[ggml_cuda_cutlass_blockscaled_scale_offset(row, subblock, scale_blocks)] =
        reinterpret_cast<const uint8_t *>(block.d4)[sub];
}

static __global__ void cutlass_scale_rows(
        float * __restrict__ dst, const float * __restrict__ row_scales, int64_t n_rows, int64_t n_cols) {
    const int64_t index = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n_rows * n_cols) {
        dst[index] *= row_scales[index / n_cols];
    }
}

static size_t cutlass_activation_size(ggml_type type, int64_t n_rows, int64_t n_cols) {
    GGML_ASSERT(type == GGML_TYPE_MXFP4 || type == GGML_TYPE_NVFP4);
    GGML_ASSERT(n_rows > 0 && n_cols > 0 && n_cols % 128 == 0);
    return type == GGML_TYPE_NVFP4 ? (size_t) n_rows * n_cols / 2 : (size_t) n_rows * n_cols;
}

static size_t cutlass_scale_size(ggml_type type, int64_t n_rows, int64_t n_cols) {
    GGML_ASSERT(type == GGML_TYPE_MXFP4 || type == GGML_TYPE_NVFP4);
    GGML_ASSERT(n_rows > 0 && n_cols > 0 && n_cols % 128 == 0);
    const int scale_values = type == GGML_TYPE_NVFP4 ? QK_NVFP4_SUB : QK_MXFP4;
    return (size_t) GGML_PAD(n_rows, 128) * (n_cols / scale_values);
}

static bool cutlass_quantize(
        ggml_backend_cuda_context & ctx,
        const float * src,
        uint8_t * dst,
        uint8_t * scales,
        float * row_scales,
        bool use_aligned_float8,
        ggml_type type,
        int64_t n_cols,
        int64_t n_cols_padded,
        int64_t stride_row,
        int64_t n_rows,
        cudaStream_t stream) {
    if ((type != GGML_TYPE_MXFP4 && type != GGML_TYPE_NVFP4) ||
        n_cols <= 0 || n_cols % 2 != 0 ||
        n_cols_padded < n_cols || n_cols_padded % 128 != 0 ||
        stride_row < n_cols || stride_row % 2 != 0 ||
        n_rows <= 0 || n_rows > UINT_MAX) {
        return false;
    }

    constexpr int threads = 256;
    CUDA_CHECK(cudaMemsetAsync(scales, 0, cutlass_scale_size(type, n_rows, n_cols_padded), stream));
    if (type == GGML_TYPE_NVFP4) {
        const int64_t blocks_per_row = (n_cols_padded + QK_FP4_MMQ - 1) / QK_FP4_MMQ;
        ggml_cuda_pool_alloc<block_fp4_mmq> quantized(ctx.pool(), blocks_per_row * n_rows);
        quantize_mmq_fp4_cuda(src, nullptr, quantized.get(), row_scales, type, use_aligned_float8,
            n_cols, stride_row, stride_row * n_rows, stride_row * n_rows,
            n_cols_padded, n_rows, 1, 1, stream);
        const int64_t subblocks = n_rows * (n_cols_padded / QK_NVFP4_SUB);
        cutlass_repack_nvfp4_activation<<<(unsigned) ((subblocks + threads - 1) / threads), threads, 0, stream>>>(
            quantized.get(), dst, scales, n_rows, n_cols_padded);
    } else {
        cutlass_quantize_mxfp8<<<(unsigned) n_rows, threads, 0, stream>>>(
            src, dst, scales, n_cols, n_cols_padded, stride_row);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

namespace ggml_cutlass_sm120 {

using namespace cute;

struct mxfp_format_traits {
    static constexpr int scale_granularity = QK_MXFP4;

    using Scale      = cutlass::float_ue8m0_t;
    using Activation = cutlass::float_e4m3_t;
    using Weight     = cutlass::float_e2m1_t;

    template <typename Element>
    using MainloopElement = cute::tuple<Element, Scale>;

    template <typename Element>
    static constexpr int alignment = 128 / cutlass::sizeof_bits<Element>::value;
};

struct nvfp4_format_traits {
    static constexpr int scale_granularity = QK_NVFP4_SUB;

    using Scale      = cutlass::float_ue4m3_t;
    using Activation = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using Weight     = Activation;

    template <typename Element>
    using MainloopElement = Element;

    template <typename Element>
    static constexpr int alignment = 32;
};

template <typename Format>
struct blockscaled_kernel_traits {
    using Output           = float;
    using ElementA         = typename Format::Activation;
    using ElementB         = typename Format::Weight;
    using ElementAMainloop = typename Format::template MainloopElement<ElementA>;
    using ElementBMainloop = typename Format::template MainloopElement<ElementB>;
    using LayoutA          = cutlass::layout::RowMajor;
    using LayoutB          = cutlass::layout::ColumnMajor;
    using LayoutD          = cutlass::layout::RowMajor;
    using TileShape        = Shape<_128, _128, _128>;
    using ClusterShape     = Shape<_1, _1, _1>;

    static constexpr int alignment_a = Format::template alignment<ElementA>;
    static constexpr int alignment_b = Format::template alignment<ElementB>;
    static constexpr int alignment_d = 128 / cutlass::sizeof_bits<Output>::value;

    using EpilogueBuilder = cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120,
        cutlass::arch::OpClassBlockScaledTensorOp,
        TileShape,
        ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        float,
        float,
        void,
        LayoutD,
        alignment_d,
        Output,
        LayoutD,
        alignment_d,
        cutlass::epilogue::collective::EpilogueScheduleAuto>;
    using CollectiveEpilogue = typename EpilogueBuilder::CollectiveOp;
    using StageCount = cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue::SharedStorage))>;
    using MainloopBuilder = cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120,
        cutlass::arch::OpClassBlockScaledTensorOp,
        ElementAMainloop,
        LayoutA,
        alignment_a,
        ElementBMainloop,
        LayoutB,
        alignment_b,
        float,
        TileShape,
        ClusterShape,
        StageCount,
        cutlass::gemm::collective::KernelScheduleAuto>;
    using CollectiveMainloop = typename MainloopBuilder::CollectiveOp;
    using ProblemShape = Shape<int, int, int, int>;
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        ProblemShape, CollectiveMainloop, CollectiveEpilogue, void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideC = typename GemmKernel::StrideC;
    using StrideD = typename GemmKernel::StrideD;
    using Scale = typename Format::Scale;
    using BlockScaleConfig = typename CollectiveMainloop::Sm1xxBlkScaledConfig;

    static_assert(CollectiveMainloop::TiledMma::SFVecSize == Format::scale_granularity);
};

template <typename Traits>
static ggml_cuda_cutlass_result run_dense_gemm(
        ggml_backend_cuda_context & ctx,
        const ggml_cuda_cutlass_weight & weight,
        const uint8_t * activation,
        const uint8_t * activation_scales,
        void * dst,
        int m,
        int n,
        int k,
        cudaStream_t stream) {
    using Gemm             = typename Traits::Gemm;
    using BlockScaleConfig = typename Traits::BlockScaleConfig;

    const auto stride_a   = cutlass::make_cute_packed_stride(typename Traits::StrideA{}, make_shape(m, k, 1));
    const auto stride_b   = cutlass::make_cute_packed_stride(typename Traits::StrideB{}, make_shape(n, k, 1));
    const auto stride_c   = cutlass::make_cute_packed_stride(typename Traits::StrideC{}, make_shape(m, n, 1));
    const auto stride_d   = cutlass::make_cute_packed_stride(typename Traits::StrideD{}, make_shape(m, n, 1));
    const auto shape      = make_shape(m, n, k, 1);
    const auto layout_sfa = BlockScaleConfig::tile_atom_to_shape_SFA(shape);
    const auto layout_sfb = BlockScaleConfig::tile_atom_to_shape_SFB(shape);

    typename Gemm::Arguments arguments = {
        cutlass::gemm::GemmUniversalMode::kGemm,
        shape,
        {
            reinterpret_cast<const typename Gemm::ElementA *>(activation),
            stride_a,
            reinterpret_cast<const typename Gemm::ElementB *>(weight.values),
            stride_b,
            reinterpret_cast<const typename Traits::Scale *>(activation_scales),
            layout_sfa,
            reinterpret_cast<const typename Traits::Scale *>(weight.scales),
            layout_sfb,
        },
        {
            {},
            nullptr,
            stride_c,
            reinterpret_cast<typename Traits::Output *>(dst),
            stride_d,
        },
    };
    arguments.epilogue.thread.alpha = 1.0f;
    arguments.epilogue.thread.beta  = 0.0f;

    Gemm gemm;
    const cutlass::Status can_implement = gemm.can_implement(arguments);
    if (can_implement != cutlass::Status::kSuccess) {
        GGML_LOG_WARN("%s: can_implement failed: %s\n", __func__, cutlassGetStatusString(can_implement));
        return ggml_cuda_cutlass_result::fallback;
    }

    const size_t workspace_size = Gemm::get_workspace_size(arguments);
    ggml_cuda_pool_alloc<char> workspace(ctx.pool());
    if (workspace_size != 0) {
        workspace.alloc(workspace_size);
    }
    const cutlass::Status initialize = gemm.initialize(arguments, workspace.get(), stream);
    if (initialize != cutlass::Status::kSuccess) {
        GGML_LOG_WARN("%s: initialize failed: %s\n", __func__, cutlassGetStatusString(initialize));
        return ggml_cuda_cutlass_result::fallback;
    }

    const cutlass::Status run = gemm.run(stream);
    if (run != cutlass::Status::kSuccess) {
        GGML_LOG_ERROR("%s: run failed: %s\n", __func__, cutlassGetStatusString(run));
        return ggml_cuda_cutlass_result::failure;
    }
    return ggml_cuda_cutlass_result::success;
}

static ggml_cuda_cutlass_result dispatch_dense_gemm(
        ggml_backend_cuda_context & ctx,
        const ggml_cuda_cutlass_weight & weight,
        const uint8_t * activation,
        const uint8_t * activation_scales,
        void * dst,
        int m,
        int n,
        int k,
        cudaStream_t stream) {
    if (weight.type == GGML_TYPE_MXFP4) {
        return run_dense_gemm<blockscaled_kernel_traits<mxfp_format_traits>>(
            ctx, weight, activation, activation_scales, dst, m, n, k, stream);
    }
    if (weight.type == GGML_TYPE_NVFP4) {
        return run_dense_gemm<blockscaled_kernel_traits<nvfp4_format_traits>>(
            ctx, weight, activation, activation_scales, dst, m, n, k, stream);
    }
    return ggml_cuda_cutlass_result::fallback;
}

} // namespace ggml_cutlass_sm120

bool ggml_cuda_cutlass_compiled() {
    return true;
}

ggml_cuda_cutlass_result ggml_cuda_cutlass_mul_mat(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
    using namespace ggml_cutlass_sm120;

    ggml_cuda_cutlass_weight weight;
    if (!ggml_cuda_repacked_mul_mat_supported(src0, src1, dst) ||
        !ggml_cuda_cutlass_weight_from_tensor(src0, weight) ||
        src1->buffer == nullptr || dst->buffer == nullptr) {
        return ggml_cuda_cutlass_result::fallback;
    }
    if (src0->ne[1] % 4 != 0) {
        return ggml_cuda_cutlass_result::fallback;
    }

    const auto & device_info = ggml_cuda_info().devices[ctx.device];
    if (!blackwell_mma_available(device_info.cc)) {
        return ggml_cuda_cutlass_result::fallback;
    }

    const ggml_backend_buffer_type_t buffer_type = ggml_backend_cuda_buffer_type(ctx.device);
    if (ggml_backend_buffer_get_type(src1->buffer) != buffer_type ||
        ggml_backend_buffer_get_type(dst->buffer) != buffer_type) {
        return ggml_cuda_cutlass_result::fallback;
    }

    const int64_t k = src0->ne[0];
    const int64_t n = src0->ne[1];
    const int64_t m = ggml_nelements(src1) / k;
    const int64_t k_padded = weight.k;
    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<uint8_t> activation(ctx.pool(), cutlass_activation_size(src0->type, m, k_padded));
    ggml_cuda_pool_alloc<uint8_t> scales(ctx.pool(), cutlass_scale_size(src0->type, m, k_padded));
    ggml_cuda_pool_alloc<float> row_scales(ctx.pool());
    if (src0->type == GGML_TYPE_NVFP4) {
        row_scales.alloc(m);
    }
    if (!cutlass_quantize(
            ctx, (const float *) src1->data,
            activation.get(),
            scales.get(),
            row_scales.get(),
            ggml_cuda_is_aligned(src1, 32),
            src0->type,
            k,
            k_padded,
            src1->nb[1] / sizeof(float),
            m,
            stream)) {
        return ggml_cuda_cutlass_result::fallback;
    }

    const auto result = dispatch_dense_gemm(
        ctx, weight, activation.get(), scales.get(), dst->data,
        (int) m, (int) n, (int) k_padded, stream);
    if (result == ggml_cuda_cutlass_result::success && row_scales.get() != nullptr) {
        constexpr int threads = 256;
        cutlass_scale_rows<<<(unsigned) ((m * n + threads - 1) / threads), threads, 0, stream>>>(
            (float *) dst->data, row_scales.get(), m, n);
        CUDA_CHECK(cudaGetLastError());
    }
    return result;
}

#else

bool ggml_cuda_cutlass_compiled() {
    return false;
}

ggml_cuda_cutlass_result ggml_cuda_cutlass_mul_mat(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
    GGML_UNUSED_VARS(ctx, src0, src1, dst);
    return ggml_cuda_cutlass_result::fallback;
}

#endif
