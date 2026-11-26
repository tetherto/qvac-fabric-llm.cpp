#pragma once

#include "common.cuh"

// ids != nullptr: MUL_MAT_ID
bool ggml_cuda_should_use_mmq_few(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, int cc);

// min. number of src1 columns (MUL_MAT_ID: tokens) from which the few-row MMQ takes this type, 0 if never
int ggml_cuda_mmq_few_min_ncols(ggml_type type, int cc, int64_t ne00, bool has_ids = false);

// NVFP4 weights in the CUTLASS repacked buffer
bool ggml_cuda_should_use_mmq_few_repacked(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);

void ggml_cuda_mul_mat_q_few_repacked(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

void ggml_cuda_mul_mat_q_few(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst);
