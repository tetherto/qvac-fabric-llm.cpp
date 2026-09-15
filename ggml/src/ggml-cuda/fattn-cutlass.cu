// SM90 fused multi-head attention for causal prefill (CUTLASS example 88 kernels); compiled for sm_90a only
#include "fattn-cutlass.cuh"

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "collective/fmha_fusion.hpp"
#include "device/device_universal.hpp"
#include "kernel/fmha_kernel_builder.hpp"
#include "cutlass-hw.cuh"

#include <cstdio>
#include <cstdlib>

namespace {

using namespace cute;

// Bottom-right causal mask over the used cells: the n_q queries are the last n_q positions of a sequence that
// occupies cells [0, K), so query q attends cell k iff k < K && k <= q + (K - Q). Stateless like the example's
// fusions (the mainloop default-constructs it); everything derives from the problem size (B, H, Q, K, D).
struct CausalOffsetFusion : cutlass::fmha::collective::DefaultFusion {
    template<class BlkCoord, class TileShape, class ProblemSize>
    CUTLASS_DEVICE int get_trip_count(BlkCoord const & blk_coord, TileShape const & tile_shape, ProblemSize const & ps) {
        const int Q = get<2>(ps);
        const int K = get<3>(ps);
        // exclusive bound of the keys the last row of this query block may attend
        const int k_end = (get<0>(blk_coord) + 1) * (int) get<0>(tile_shape) + (K - Q);
        return ceil_div(std::min(K, k_end), (int) get<1>(tile_shape));
    }

    // the diagonal spans at most TILE_Q/TILE_K + 1 key tiles; the residual key tile (K % TILE_K != 0) is the last one
    template<class BlkCoord, class TileShape, class ProblemSize>
    CUTLASS_DEVICE int get_masked_trip_count(BlkCoord const & blk_coord, TileShape const & tile_shape, ProblemSize const & ps) {
        const int n_diag = ceil_div((int) get<0>(tile_shape), (int) get<1>(tile_shape)) + 1;
        return std::min(get_trip_count(blk_coord, tile_shape, ps), n_diag);
    }

    template<class BlkCoord, class TileShape, class ProblemSize>
    CUTLASS_DEVICE int get_unmasked_trip_count(BlkCoord const & blk_coord, TileShape const & tile_shape, ProblemSize const & ps) {
        return get_trip_count(blk_coord, tile_shape, ps) - get_masked_trip_count(blk_coord, tile_shape, ps);
    }

    template<class AccQK, class IndexQK, class ProblemSize>
    CUTLASS_DEVICE void before_softmax(AccQK & acc_qk, IndexQK const & index_qk, ProblemSize const & ps) {
        const int Q = get<2>(ps);
        const int K = get<3>(ps);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(acc_qk); i++) {
            auto pos = index_qk(i);
            const int q = get<0>(pos);
            const int k = get<1>(pos);
            if (k >= K || k > q + (K - Q)) {
                acc_qk(i) = -INFINITY;
            }
        }
    }
};

using Element      = cutlass::half_t;
using ElementAcc   = float;
using TileShape    = Shape<Int<GGML_FATTN_CUTLASS_TILE_Q>, Int<GGML_FATTN_CUTLASS_TILE_K>, Int<GGML_FATTN_CUTLASS_D>>;
using StrideQ      = cute::tuple<int, _1, cute::tuple<int, int>>; // Q D (B H)
using StrideK      = cute::tuple<int, _1, cute::tuple<int, int>>; // K D (B H)
using StrideV      = cute::tuple<int, _1, cute::tuple<int, int>>; // K D (B H)
using StrideO      = cute::tuple<int, _1, cute::tuple<int, int>>; // Q D (B H)
using StrideLSE    = cute::tuple<_1, cute::tuple<int, int>>;      // Q (B H)
using ProblemShape = cute::tuple<int, int, int, int, int>;        // B H Q K D

using Operation = cutlass::device::Universal<
    typename cutlass::fmha::kernel::FmhaBuilder<
        Element, ElementAcc, ElementAcc, TileShape, StrideQ, StrideK, StrideV,
        CausalOffsetFusion, cutlass::gemm::KernelTmaWarpSpecializedCooperative
    >::Kernel>;

// B = kv head (group), H = query head within the group: q/o heads are adjacent (stride D), k/v repeat (stride 0)
typename Operation::Arguments make_args(
        const half * q16, const half * k, const half * v, half * o16, float * lse,
        int n_head_kv, int gqa, int n_q, int n_kv_used,
        int64_t k_nb1, int64_t k_nb2, int64_t v_nb1, int64_t v_nb2) {
    const int D      = GGML_FATTN_CUTLASS_D;
    const int n_head = n_head_kv * gqa;

    const StrideQ   stride_Q   = make_stride(n_head*D, _1{}, make_stride(gqa*D, D));
    const StrideK   stride_K   = make_stride((int) k_nb1, _1{}, make_stride((int) k_nb2, 0));
    const StrideV   stride_V   = make_stride((int) v_nb1, _1{}, make_stride((int) v_nb2, 0));
    const StrideO   stride_O   = stride_Q;
    const StrideLSE stride_LSE = make_stride(_1{}, make_stride(gqa*n_q, n_q));

    return typename Operation::Arguments{
        ProblemShape{n_head_kv, gqa, n_q, n_kv_used, D},
        { (const Element *) q16, stride_Q, (const Element *) k, stride_K, (const Element *) v, stride_V },
        { (Element *) o16, stride_O, lse, stride_LSE },
        ggml_cutlass_hw_info_for_current_device()
    };
}

} // namespace

size_t ggml_cuda_fattn_cutlass_workspace_size(int n_head_kv, int gqa, int n_q, int n_kv_used) {
    return Operation::get_workspace_size(make_args(nullptr, nullptr, nullptr, nullptr, nullptr,
        n_head_kv, gqa, n_q, n_kv_used, 0, 0, 0, 0));
}

void ggml_cuda_fattn_cutlass(
        const half * q16, const half * k, const half * v, half * o16, float * lse,
        int n_head_kv, int gqa, int n_q, int n_kv_used,
        int64_t k_nb1, int64_t k_nb2, int64_t v_nb1, int64_t v_nb2,
        void * workspace, size_t workspace_size, cudaStream_t stream) {
    const typename Operation::Arguments args = make_args(q16, k, v, o16, lse,
        n_head_kv, gqa, n_q, n_kv_used, k_nb1, k_nb2, v_nb1, v_nb2);

    Operation op;
    cutlass::Status status = op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "%s: CUTLASS cannot implement the FMHA for n_q=%d n_kv=%d heads=%dx%d: %s\n", __func__,
            n_q, n_kv_used, n_head_kv, gqa, cutlass::cutlassGetStatusString(status));
        abort();
    }
    if (Operation::get_workspace_size(args) > workspace_size) {
        fprintf(stderr, "%s: workspace too small for n_q=%d n_kv=%d\n", __func__, n_q, n_kv_used);
        abort();
    }
    status = op.initialize(args, workspace, stream);
    if (status == cutlass::Status::kSuccess) {
        status = op.run(stream);
    }
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "%s: CUTLASS FMHA failed for n_q=%d n_kv=%d: %s\n", __func__, n_q, n_kv_used, cutlass::cutlassGetStatusString(status));
        abort();
    }
}
