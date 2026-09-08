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

bool ggml_cuda_cutlass_compiled() {
#ifdef GGML_CUDA_CUTLASS
    return true;
#else
    return false;
#endif
}

ggml_cuda_cutlass_result ggml_cuda_cutlass_mul_mat(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
    GGML_UNUSED_VARS(ctx, src0, src1, dst);
    return ggml_cuda_cutlass_result::fallback;
}
