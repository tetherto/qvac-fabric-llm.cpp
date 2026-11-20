#include "common.cuh"

#define MMVQ_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMVQ kernels.
#define MMVQ_MAX_BATCH_SIZE_WIDE 16 // Max. batch size of the types that ggml_cuda_mmvq_wide accepts.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// MMVQ up to MMVQ_MAX_BATCH_SIZE_WIDE columns for this type and device
bool ggml_cuda_mmvq_wide(enum ggml_type type, int cc);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);
