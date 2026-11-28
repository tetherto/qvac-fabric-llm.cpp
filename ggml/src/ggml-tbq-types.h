//
// TurboQuant / PolarQuant quantization (Zandieh et al., ICLR 2026)
//
// PolarQuant (pq3_0):  Stage 1 only  — rotation + Lloyd-Max scalar quantization + bit-packing.
//                      The original "tbq3_0" implementation, renamed to free up the tq3 name.
// TurboQuant (tbq3_0):  Stage 1 + QJL Stage 2 — adds 1-bit residual sketch for unbiased inner products.
//
// Both share identical codebooks, sign arrays, Hadamard transforms, and Stage 1 quantization logic.
// TQ3 simply appends a QJL sidecar (sign bits of projected residual + residual norm) to each block.
//
// Two block sizes: 128 (head_dim is a multiple of 128) and 64 (head_dim=64).
// The user specifies "tbq3_0" (with QJL) or "pq3_0" (without) on the CLI;
// the KV cache init selects the _64 variant automatically when head_dim=64
// and packs wider heads as consecutive 128-element blocks.
//
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#define QK_TQ 128

// QJL sketch dimensions per block size (= number of random projections for Stage 2)
#define QJL_SKETCH_DIM_128   128
#define QJL_SKETCH_DIM_64    64
#define QJL_SKETCH_BYTES_128 (QJL_SKETCH_DIM_128 / 8)  // 16 bytes
#define QJL_SKETCH_BYTES_64  (QJL_SKETCH_DIM_64 / 8)   //  8 bytes

// --- PolarQuant 3-bit: Stage 1 only (identical to the former "tbq3_0" layout) ---
// block=128: 48 index bytes + 2 norm bytes = 50 bytes (3.125 bpw)
// block=64:  24 index bytes + 2 norm bytes = 26 bytes (3.25 bpw)

#define PQ3_0_INDEX_BYTES ((QK_TQ * 3 + 7) / 8)  // 48

typedef struct {
    uint8_t   qs[PQ3_0_INDEX_BYTES];  // bit-packed 3-bit codebook indices
    ggml_half d;                      // L2 norm of original vector
} block_pq3_0;

static_assert(sizeof(block_pq3_0) == sizeof(ggml_half) + PQ3_0_INDEX_BYTES, "wrong pq3_0 block size/padding");

// --- TurboQuant 3-bit: Stage 1 + QJL Stage 2 ---
// block=128: 48 + 2 + 16 + 2 = 68 bytes (4.25 bpw)
// block=64:  24 + 2 +  8 + 2 = 36 bytes (4.5 bpw)
// The first two fields (qs, d) are identical to PQ3 for codebook compatibility.
// The QJL sidecar stores sign(R * residual) and ||residual||.

#define TBQ3_0_INDEX_BYTES PQ3_0_INDEX_BYTES  // same codebook indices as PQ3

typedef struct {
    uint8_t   qs[TBQ3_0_INDEX_BYTES];     // bit-packed 3-bit codebook indices (Stage 1)
    ggml_half d;                          // L2 norm of original vector
    uint8_t   qjl[QJL_SKETCH_BYTES_128];  // Stage 2: sign bits of R * residual
    ggml_half d_r;                        // Stage 2: L2 norm of residual
} block_tbq3_0;

static_assert(sizeof(block_tbq3_0) == 2 * sizeof(ggml_half) + TBQ3_0_INDEX_BYTES + QJL_SKETCH_BYTES_128,
              "wrong tbq3_0 block size/padding");

// --- PolarQuant 4-bit: Stage 1 only ---
// block=128: 64 index bytes + 2 norm bytes = 66 bytes (4.125 bpw)
// block=64:  32 index bytes + 2 norm bytes = 34 bytes (4.25 bpw)

#define PQ4_0_INDEX_BYTES (QK_TQ / 2)  // 64

typedef struct {
    uint8_t   qs[PQ4_0_INDEX_BYTES];  // packed 4-bit codebook indices (2 per byte)
    ggml_half d;                      // L2 norm of original vector
} block_pq4_0;

static_assert(sizeof(block_pq4_0) == sizeof(ggml_half) + PQ4_0_INDEX_BYTES, "wrong pq4_0 block size/padding");

// --- TurboQuant 4-bit: Stage 1 + QJL Stage 2 ---
// block=128: 64 + 2 + 16 + 2 = 84 bytes (5.25 bpw)
// block=64:  32 + 2 +  8 + 2 = 44 bytes (5.5 bpw)

#define TBQ4_0_INDEX_BYTES PQ4_0_INDEX_BYTES

typedef struct {
    uint8_t   qs[TBQ4_0_INDEX_BYTES];     // packed 4-bit codebook indices (Stage 1)
    ggml_half d;                          // L2 norm of original vector
    uint8_t   qjl[QJL_SKETCH_BYTES_128];  // Stage 2: sign bits of R * residual
    ggml_half d_r;                        // Stage 2: L2 norm of residual
} block_tbq4_0;

static_assert(sizeof(block_tbq4_0) == 2 * sizeof(ggml_half) + TBQ4_0_INDEX_BYTES + QJL_SKETCH_BYTES_128,
              "wrong tbq4_0 block size/padding");

// --- block size 64 (head_dim=64: Llama-3.2-1B/3B) ---

#define QK_TQ_64 64

#define PQ3_0_64_INDEX_BYTES ((QK_TQ_64 * 3 + 7) / 8)  // 24

typedef struct {
    uint8_t   qs[PQ3_0_64_INDEX_BYTES];
    ggml_half d;
} block_pq3_0_64;

static_assert(sizeof(block_pq3_0_64) == sizeof(ggml_half) + PQ3_0_64_INDEX_BYTES, "wrong pq3_0_64 block size/padding");

#define TBQ3_0_64_INDEX_BYTES PQ3_0_64_INDEX_BYTES

typedef struct {
    uint8_t   qs[TBQ3_0_64_INDEX_BYTES];
    ggml_half d;
    uint8_t   qjl[QJL_SKETCH_BYTES_64];
    ggml_half d_r;
} block_tbq3_0_64;

static_assert(sizeof(block_tbq3_0_64) == 2 * sizeof(ggml_half) + TBQ3_0_64_INDEX_BYTES + QJL_SKETCH_BYTES_64,
              "wrong tbq3_0_64 block size/padding");

#define PQ4_0_64_INDEX_BYTES (QK_TQ_64 / 2)  // 32

typedef struct {
    uint8_t   qs[PQ4_0_64_INDEX_BYTES];
    ggml_half d;
} block_pq4_0_64;

static_assert(sizeof(block_pq4_0_64) == sizeof(ggml_half) + PQ4_0_64_INDEX_BYTES, "wrong pq4_0_64 block size/padding");

#define TBQ4_0_64_INDEX_BYTES PQ4_0_64_INDEX_BYTES

typedef struct {
    uint8_t   qs[TBQ4_0_64_INDEX_BYTES];
    ggml_half d;
    uint8_t   qjl[QJL_SKETCH_BYTES_64];
    ggml_half d_r;
} block_tbq4_0_64;

static_assert(sizeof(block_tbq4_0_64) == 2 * sizeof(ggml_half) + TBQ4_0_64_INDEX_BYTES + QJL_SKETCH_BYTES_64,
              "wrong tbq4_0_64 block size/padding");

#ifdef __cplusplus
}
#endif
