// SM90 block-scaled FP8 GEMM for F8_E4M3 weights; compiled for sm_90a only (see GGML_CUDA_CUTLASS in CMakeLists.txt)
#include "mmf8-cutlass.cuh"

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/util/packed_stride.hpp"

#include <cstdio>
#include <cstdlib>

namespace {

using namespace cute;

// A [M][K] row-major (per-token activations), B [N][K] (weights, k contiguous), D [M][N] row-major f32
// tile 128x128x128, cluster 1x1x1, persistent scheduler: 1094 / 1136 TFLOPS at M=4096 on the FFN shapes (F0)
using ElementA = cutlass::float_e4m3_t;
using LayoutA  = cutlass::layout::RowMajor;
using ElementB = cutlass::float_e4m3_t;
using LayoutB  = cutlass::layout::ColumnMajor;
using ElementD = float;
using LayoutD  = cutlass::layout::RowMajor;
using ElementAccumulator = float;
using ElementCompute     = float;
constexpr int AlignA = 16;
constexpr int AlignB = 16;
constexpr int AlignD = 4;

using TileShape    = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

using ScaleConfig = cutlass::detail::Sm90BlockwiseScaleConfig<1, 128, 128>;
using LayoutSFA   = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB   = decltype(ScaleConfig::deduce_layoutSFB());

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    void, LayoutD, AlignD,
    ElementD, LayoutD, AlignD,
    cutlass::epilogue::TmaWarpSpecializedCooperative
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignA,
    ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8Blockwise
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, cutlass::gemm::PersistentScheduler>;
using Gemm       = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA    = typename Gemm::GemmKernel::StrideA;
using StrideB    = typename Gemm::GemmKernel::StrideB;
using StrideD    = typename Gemm::GemmKernel::StrideD;

constexpr int MAX_SWIZZLE = 4;

typename Gemm::Arguments make_args(const uint8_t * a, const float * sfa, const uint8_t * b, const float * sfb, float * d, int M, int N, int K) {
    static cutlass::KernelHardwareInfo hw_info = [] {
        cutlass::KernelHardwareInfo info;
        cudaGetDevice(&info.device_id);
        info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(info.device_id);
        return info;
    }();

    const StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, make_shape(M, K, 1));
    const StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, make_shape(N, K, 1));
    const StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, make_shape(M, N, 1));
    const LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
    const LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {(const ElementA *) a, stride_A, (const ElementB *) b, stride_B, sfa, layout_SFA, sfb, layout_SFB},
        {{}, nullptr, stride_D, d, stride_D},
        hw_info
    };
    args.scheduler.max_swizzle_size = MAX_SWIZZLE;
    return args;
}

} // namespace

size_t ggml_cuda_mmf8_cutlass_workspace_size(int M, int N, int K) {
    return Gemm::get_workspace_size(make_args(nullptr, nullptr, nullptr, nullptr, nullptr, M, N, K));
}

void ggml_cuda_mmf8_cutlass(
        const uint8_t * a, const float * sfa, const uint8_t * b, const float * sfb, float * d,
        int M, int N, int K, void * workspace, size_t workspace_size, cudaStream_t stream) {
    const typename Gemm::Arguments args = make_args(a, sfa, b, sfb, d, M, N, K);

    Gemm gemm;
    cutlass::Status status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "%s: CUTLASS cannot implement the FP8 GEMM for M=%d N=%d K=%d: %s\n", __func__, M, N, K, cutlass::cutlassGetStatusString(status));
        abort();
    }
    if (Gemm::get_workspace_size(args) > workspace_size) {
        fprintf(stderr, "%s: workspace too small for M=%d N=%d K=%d\n", __func__, M, N, K);
        abort();
    }
    status = gemm.initialize(args, workspace, stream);
    if (status == cutlass::Status::kSuccess) {
        status = gemm.run(stream);
    }
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "%s: CUTLASS FP8 GEMM failed for M=%d N=%d K=%d: %s\n", __func__, M, N, K, cutlass::cutlassGetStatusString(status));
        abort();
    }
}
