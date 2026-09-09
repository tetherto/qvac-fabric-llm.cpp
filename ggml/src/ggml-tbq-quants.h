#pragma once

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml.h"

// GGML internal header

#ifdef __cplusplus
extern "C" {
#endif

GGML_API void quantize_row_tbq3_0_ref(const float * GGML_RESTRICT x, block_tbq3_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_tbq4_0_ref(const float * GGML_RESTRICT x, block_tbq4_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_tbq3_0_64_ref(const float * GGML_RESTRICT x, block_tbq3_0_64 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_tbq4_0_64_ref(const float * GGML_RESTRICT x, block_tbq4_0_64 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_pq3_0_ref(const float * GGML_RESTRICT x, block_pq3_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_pq3_0_64_ref(const float * GGML_RESTRICT x, block_pq3_0_64 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_pq4_0_ref(const float * GGML_RESTRICT x, block_pq4_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_pq4_0_64_ref(const float * GGML_RESTRICT x, block_pq4_0_64 * GGML_RESTRICT y, int64_t k);

GGML_API void dequantize_row_tbq3_0(const block_tbq3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_tbq4_0(const block_tbq4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_tbq3_0_64(const block_tbq3_0_64 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_tbq4_0_64(const block_tbq4_0_64 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_pq3_0(const block_pq3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_pq3_0_64(const block_pq3_0_64 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_pq4_0(const block_pq4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_pq4_0_64(const block_pq4_0_64 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API size_t quantize_tbq3_0   (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_tbq4_0   (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_tbq3_0_64(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_tbq4_0_64(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_pq3_0    (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_pq3_0_64 (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_pq4_0    (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_pq4_0_64 (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API float qjl_dot_correction(const uint8_t * qjl_bits, float d_r, const float * b, int d);

// TurboQuant internal helpers (exposed for testing)
GGML_API void    tq_fht(float * x, int d);
GGML_API void    tq_forward_inplace(float * buf, int d, const float * signs);
GGML_API void    tq_inverse_inplace(float * buf, int d, const float * signs);
GGML_API uint8_t tq3_quantize_val(float val, const float * boundaries);
GGML_API uint8_t tq4_quantize_val(float val, const float * boundaries);
GGML_API void    tq_compute_boundaries(const float * cb, float * boundaries, int n);
GGML_API const float * tq3_codebook_for(int d);
GGML_API const float * tq4_codebook_for(int d);

#ifdef __cplusplus
}
#endif
