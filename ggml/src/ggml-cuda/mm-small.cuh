#include "common.cuh"

// Batched F32 products with a short contraction and few columns, one direct kernel instead of cuBLAS:
// windowed attention over many small windows gives tens of thousands of 16x16 products per call.

// longest contraction (src0->ne[0]): the rows of src0 a thread holds must fit in registers
#define MM_SMALL_MAX_K 32
// most src1 columns: above this the strided batched cuBLAS GEMM is as fast or faster
#define MM_SMALL_MAX_COLS 32
// fewest matrices per call: below this the call is too small to matter and the other paths are kept
#define MM_SMALL_MIN_BATCH 16

bool ggml_cuda_should_use_mm_small(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

void ggml_cuda_mul_mat_small(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
