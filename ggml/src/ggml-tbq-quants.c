#include "ggml-impl.h"
#include "ggml-tbq-quants.h"

// ====================== TurboQuant (Zandieh et al., ICLR 2026) ======================
//
// Algorithm 1 (TurboQuant_mse): rotation + Lloyd-Max scalar quantization + bit-packing
// Adapted from community CPU implementation by veritatisquaesitoressumus

// Lloyd-Max codebooks for the Beta distribution induced by random rotation of unit
// vectors in R^d. Pre-computed via Lloyd-Max algorithm per Theorem 1 of the paper.
// Recompute with: scripts/compute_tq_codebooks.py --dims 64 128 --bits 3 4 --c-code

// d=128 codebooks
static const float TQ3_CODEBOOK_128[8] = {
    -0.18839718597003241f, -0.11813976699668613f,
    -0.06658560804735174f, -0.02160431064212660f,
     0.02160431064212660f,  0.06658560804735174f,
     0.11813976699668613f,  0.18839718597003241f,
};

static const float TQ4_CODEBOOK_128[16] = {
    -0.23762692286887249f, -0.18079342531272283f,
    -0.14176134070424901f, -0.11024676790280842f,
    -0.08279230816984559f, -0.05774433563409530f,
    -0.03413390187425037f, -0.01129645493594766f,
     0.01129645493594766f,  0.03413390187425037f,
     0.05774433563409530f,  0.08279230816984559f,
     0.11024676790280842f,  0.14176134070424901f,
     0.18079342531272283f,  0.23762692286887249f,
};

// d=64 codebooks (wider spread since sigma = 1/sqrt(d) is larger)
static const float TQ3_CODEBOOK_64[8] = {
    -0.26391393084454512f, -0.16616785892516461f,
    -0.09383226321833739f, -0.03046917893115905f,
     0.03046917893115905f,  0.09383226321833739f,
     0.16616785892516461f,  0.26391393084454512f,
};

static const float TQ4_CODEBOOK_64[16] = {
    -0.33074821159014389f, -0.25285715281341298f,
    -0.19879720552558833f, -0.15486925951295250f,
    -0.11643764752566743f, -0.08127367507061777f,
    -0.04806567112944460f, -0.01591077077846402f,
     0.01591077077846402f,  0.04806567112944460f,
     0.08127367507061777f,  0.11643764752566743f,
     0.15486925951295250f,  0.19879720552558833f,
     0.25285715281341298f,  0.33074821159014389f,
};

// xoshiro256** PRNG for deterministic rotation matrix generation
typedef struct { uint64_t s[4]; } tq_rng_t;

static inline uint64_t tq_rng_rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t tq_rng_next(tq_rng_t * rng) {
    const uint64_t result = tq_rng_rotl(rng->s[1] * 5, 7) * 9;
    const uint64_t t = rng->s[1] << 17;
    rng->s[2] ^= rng->s[0];
    rng->s[3] ^= rng->s[1];
    rng->s[1] ^= rng->s[2];
    rng->s[0] ^= rng->s[3];
    rng->s[2] ^= t;
    rng->s[3] = tq_rng_rotl(rng->s[3], 45);
    return result;
}

static void tq_rng_seed(tq_rng_t * rng, uint64_t seed) {
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        rng->s[i] = z ^ (z >> 31);
    }
}

// Not needed for Hadamard transform — kept for potential future use
// static float tq_rng_normal(tq_rng_t * rng) { ... }

// ====================== Randomized Hadamard Transform ======================
// Replaces dense O(d²) rotation with O(d log d) butterfly transform.
// R = (1/√d) · H · D where H is Walsh-Hadamard, D is random ±1 diagonal.
// R is orthogonal: R^T = (1/√d) · D · H (since H^T=H, D^T=D, H·H=d·I).

// Random sign arrays (±1) for the diagonal D, one per block size, from fixed seeds.
#define TQ_SIGN_SEED_128 42
#define TQ_SIGN_SEED_64  43
static float   tq_signs_128[QK_TQ];
static float   tq_signs_64[QK_TQ_64];
static int32_t tq_signs_128_ready = 0;
static int32_t tq_signs_64_ready  = 0;

static void tq_generate_signs(float * signs, int d, uint64_t seed) {
    tq_rng_t rng;
    tq_rng_seed(&rng, seed);
    for (int i = 0; i < d; i++) {
        signs[i] = (tq_rng_next(&rng) & 1) ? 1.0f : -1.0f;
    }
}

static const float * tq_get_signs(int d) {
    if (d == QK_TQ) {
        if (!tq_signs_128_ready) { tq_generate_signs(tq_signs_128, QK_TQ, TQ_SIGN_SEED_128); tq_signs_128_ready = 1; }
        return tq_signs_128;
    }
    if (!tq_signs_64_ready) { tq_generate_signs(tq_signs_64, QK_TQ_64, TQ_SIGN_SEED_64); tq_signs_64_ready = 1; }
    return tq_signs_64;
}

const float * tq3_codebook_for(int d) {
    GGML_ASSERT(d == QK_TQ || d == QK_TQ_64);
    return d == QK_TQ ? TQ3_CODEBOOK_128 : TQ3_CODEBOOK_64;
}
const float * tq4_codebook_for(int d) {
    GGML_ASSERT(d == QK_TQ || d == QK_TQ_64);
    return d == QK_TQ ? TQ4_CODEBOOK_128 : TQ4_CODEBOOK_64;
}

// In-place Fast Walsh-Hadamard Transform (FHT) via iterative butterfly pattern.
// Equivalent to multiplying x by the unnormalized d×d Hadamard matrix H_d.
// Complexity: O(d log d) using log2(d) passes of d/2 butterfly pairs.
// Reference: https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform
// d must be a power of 2.
void tq_fht(float * x, int d) {
    for (int half = 1; half < d; half <<= 1) {
        for (int i = 0; i < d; i += half << 1) {
            for (int j = i; j < i + half; j++) {
                float a = x[j];
                float b = x[j + half];
                x[j]        = a + b;
                x[j + half] = a - b;
            }
        }
    }
}

// Forward transform (in-place): buf = (1/√d) · H · D · buf
void tq_forward_inplace(float * buf, int d, const float * signs) {
    for (int i = 0; i < d; i++) buf[i] *= signs[i];
    tq_fht(buf, d);
    float inv_sqrt_d = 1.0f / sqrtf((float)d);
    for (int i = 0; i < d; i++) buf[i] *= inv_sqrt_d;
}

// Inverse transform (in-place): buf = D · H · buf · (1/√d)
void tq_inverse_inplace(float * buf, int d, const float * signs) {
    tq_fht(buf, d);
    float inv_sqrt_d = 1.0f / sqrtf((float)d);
    for (int i = 0; i < d; i++) buf[i] *= signs[i] * inv_sqrt_d;
}


// Binary search quantize: 3 comparisons for 8 sorted centroids
uint8_t tq3_quantize_val(float val, const float * b) {
    if (val < b[3]) {
        if (val < b[1]) { return val < b[0] ? 0 : 1; }
        else            { return val < b[2] ? 2 : 3; }
    } else {
        if (val < b[5]) { return val < b[4] ? 4 : 5; }
        else            { return val < b[6] ? 6 : 7; }
    }
}

// Binary search quantize: 4 comparisons for 16 sorted centroids
uint8_t tq4_quantize_val(float val, const float * b) {
    if (val < b[7]) {
        if (val < b[3]) {
            if (val < b[1]) { return val < b[0] ? 0 : 1; }
            else            { return val < b[2] ? 2 : 3; }
        } else {
            if (val < b[5]) { return val < b[4] ? 4 : 5; }
            else            { return val < b[6] ? 6 : 7; }
        }
    } else {
        if (val < b[11]) {
            if (val < b[9])  { return val < b[8]  ? 8  : 9;  }
            else             { return val < b[10] ? 10 : 11; }
        } else {
            if (val < b[13]) { return val < b[12] ? 12 : 13; }
            else             { return val < b[14] ? 14 : 15; }
        }
    }
}

// Compute decision boundaries as midpoints between adjacent codebook centroids.
// Used for nearest-centroid quantization: a value falling between cb[i] and cb[i+1]
// is assigned to whichever centroid is closer (i.e. the boundary is their average).
// n = number of centroids (8 for TQ3, 16 for TQ4), outputs n-1 boundaries.
void tq_compute_boundaries(const float * cb, float * boundaries, int n) {
    for (int i = 0; i < n - 1; i++) {
        boundaries[i] = (cb[i] + cb[i + 1]) * 0.5f;
    }
}

// Norm correction: store MSE-optimal scale alpha = <x, c> / <c, c> instead of
// ||x||, where c is the codebook reconstruction direction (cb[idx] values).
// This minimizes ||x - alpha*c||^2 and corrects quantization's norm shrinkage.
// Controlled by GGML_TQ_NORM_CORRECTION env var (checked once, cached).
static int tq_norm_correction_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char * env = getenv("GGML_TQ_NORM_CORRECTION");
        cached = (env && env[0] == '1') ? 1 : 0;
    }
    return cached;
}

// Shared TQ3 quantize: normalize + binary-search + packed 3-bit write
// Rotation is handled at graph level by optRot (ggml_rotate_hadamard)
static void tq3_quantize_block(const float * src, uint8_t * qs, ggml_half * norm_out,
                                int d, int index_bytes, const float * signs, const float * cb) {
    GGML_UNUSED(signs);
    float norm = 0.0f;
    for (int j = 0; j < d; j++) norm += src[j] * src[j];
    norm = sqrtf(norm);
    if (norm < 1e-15f) { *norm_out = GGML_FP32_TO_FP16(0.0f); memset(qs, 0, index_bytes); return; }

    float boundaries[7];
    tq_compute_boundaries(cb, boundaries, 8);

    float inv_norm = 1.0f / norm;

    // Pack 8 indices (24 bits = 3 bytes) at a time
    for (int g = 0; g < d / 8; g++) {
        uint32_t accum = 0;
        for (int i = 0; i < 8; i++) {
            uint8_t idx = tq3_quantize_val(src[g * 8 + i] * inv_norm, boundaries);
            accum |= (uint32_t)idx << (i * 3);
        }
        int base = g * 3;
        qs[base + 0] = (uint8_t)(accum & 0xFF);
        qs[base + 1] = (uint8_t)((accum >> 8) & 0xFF);
        qs[base + 2] = (uint8_t)((accum >> 16) & 0xFF);
    }

    if (tq_norm_correction_enabled()) {
        // MSE-optimal scale: alpha = <x/||x||, c> / <c, c> * ||x|| = <x, c> / <c, c>
        // where c is the vector of cb[idx] values (unit-norm codebook reconstruction)
        float dot_xc = 0.0f, dot_cc = 0.0f;
        int bit_pos = 0;
        for (int r = 0; r < d; r++) {
            uint8_t idx = 0;
            for (int b = 0; b < 3; b++) {
                if (qs[bit_pos / 8] & (1 << (bit_pos % 8))) idx |= (1 << b);
                bit_pos++;
            }
            float cv = cb[idx];
            dot_xc += src[r] * cv;
            dot_cc += cv * cv;
        }
        if (dot_cc > 1e-15f) {
            norm = dot_xc / dot_cc;
        }
    }

    *norm_out = GGML_FP32_TO_FP16(norm);
}

// Shared TQ4 quantize: normalize + binary-search + nibble pack
// Rotation is handled at graph level by optRot (ggml_rotate_hadamard)
static void tq4_quantize_block(const float * src, uint8_t * qs, ggml_half * norm_out,
                                int d, int index_bytes, const float * signs, const float * cb) {
    GGML_UNUSED(signs);
    float norm = 0.0f;
    for (int j = 0; j < d; j++) norm += src[j] * src[j];
    norm = sqrtf(norm);
    if (norm < 1e-15f) { *norm_out = GGML_FP32_TO_FP16(0.0f); memset(qs, 0, index_bytes); return; }

    float boundaries[15];
    tq_compute_boundaries(cb, boundaries, 16);

    float inv_norm = 1.0f / norm;

    for (int r = 0; r < d; r += 2) {
        uint8_t idx0 = tq4_quantize_val(src[r]     * inv_norm, boundaries);
        uint8_t idx1 = tq4_quantize_val(src[r + 1] * inv_norm, boundaries);
        qs[r / 2] = idx0 | (idx1 << 4);
    }

    if (tq_norm_correction_enabled()) {
        float dot_xc = 0.0f, dot_cc = 0.0f;
        for (int r = 0; r < d; r += 2) {
            uint8_t byte = qs[r / 2];
            float cv0 = cb[byte & 0x0F];
            float cv1 = cb[byte >> 4];
            dot_xc += src[r] * cv0 + src[r + 1] * cv1;
            dot_cc += cv0 * cv0 + cv1 * cv1;
        }
        if (dot_cc > 1e-15f) {
            norm = dot_xc / dot_cc;
        }
    }

    *norm_out = GGML_FP32_TO_FP16(norm);
}

// Shared TQ3 dequantize: unpack + codebook lookup + scale
// Inverse rotation is handled at graph level by optRot
static void tq3_dequantize_block(const uint8_t * qs, ggml_half norm_h,
                                  float * dst, int d, const float * signs, const float * cb) {
    GGML_UNUSED(signs);
    float norm = GGML_FP16_TO_FP32(norm_h);
    if (fabsf(norm) < 1e-15f) { memset(dst, 0, d * sizeof(float)); return; }

    int bit_pos = 0;
    for (int r = 0; r < d; r++) {
        uint8_t idx = 0;
        for (int b = 0; b < 3; b++) {
            if (qs[bit_pos / 8] & (1 << (bit_pos % 8))) idx |= (1 << b);
            bit_pos++;
        }
        dst[r] = cb[idx] * norm;
    }
}

// Shared TQ4 dequantize: unpack nibbles + codebook lookup + scale
// Inverse rotation is handled at graph level by optRot
static void tq4_dequantize_block(const uint8_t * qs, ggml_half norm_h,
                                  float * dst, int d, const float * signs, const float * cb) {
    GGML_UNUSED(signs);
    float norm = GGML_FP16_TO_FP32(norm_h);
    if (fabsf(norm) < 1e-15f) { memset(dst, 0, d * sizeof(float)); return; }

    for (int r = 0; r < d; r += 2) {
        uint8_t byte = qs[r / 2];
        dst[r    ] = cb[byte & 0x0F] * norm;
        dst[r + 1] = cb[byte >> 4]  * norm;
    }
}

static void qjl_encode_residual(const float * residual, int d, uint8_t * qjl_out, int qjl_bytes, ggml_half * d_r_out);

void quantize_row_tbq4_0_ref(const float * GGML_RESTRICT x, block_tbq4_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ == 0);
    const float * signs = tq_get_signs(QK_TQ);
    const float * cb = tq4_codebook_for(QK_TQ);

    float dequant_buf[QK_TQ];
    float residual[QK_TQ];

    for (int64_t i = 0; i < k / QK_TQ; i++) {
        const float * src = x + i * QK_TQ;

        tq4_quantize_block(src, y[i].qs, &y[i].d, QK_TQ, TBQ4_0_INDEX_BYTES, signs, cb);

        tq4_dequantize_block(y[i].qs, y[i].d, dequant_buf, QK_TQ, signs, cb);
        for (int j = 0; j < QK_TQ; j++) residual[j] = src[j] - dequant_buf[j];
        qjl_encode_residual(residual, QK_TQ, y[i].qjl, QJL_SKETCH_BYTES_128, &y[i].d_r);
    }
}

void dequantize_row_tbq4_0(const block_tbq4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ == 0);
    const float * signs = tq_get_signs(QK_TQ);
    const float * cb = tq4_codebook_for(QK_TQ);
    for (int64_t i = 0; i < k / QK_TQ; i++)
        tq4_dequantize_block(x[i].qs, x[i].d, y + i*QK_TQ, QK_TQ, signs, cb);
}

size_t quantize_tbq4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void)quant_weights;
    quantize_row_tbq4_0_ref(src, dst, nrow*n_per_row);
    return nrow * ggml_row_size(GGML_TYPE_TBQ4_0, n_per_row);
}

// --- block=64 public API ---

void quantize_row_tbq4_0_64_ref(const float * GGML_RESTRICT x, block_tbq4_0_64 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ_64 == 0);
    const float * signs = tq_get_signs(QK_TQ_64);
    const float * cb = tq4_codebook_for(QK_TQ_64);

    float dequant_buf[QK_TQ_64];
    float residual[QK_TQ_64];

    for (int64_t i = 0; i < k / QK_TQ_64; i++) {
        const float * src = x + i * QK_TQ_64;

        tq4_quantize_block(src, y[i].qs, &y[i].d, QK_TQ_64, TBQ4_0_64_INDEX_BYTES, signs, cb);

        tq4_dequantize_block(y[i].qs, y[i].d, dequant_buf, QK_TQ_64, signs, cb);
        for (int j = 0; j < QK_TQ_64; j++) residual[j] = src[j] - dequant_buf[j];
        qjl_encode_residual(residual, QK_TQ_64, y[i].qjl, QJL_SKETCH_BYTES_64, &y[i].d_r);
    }
}

void dequantize_row_tbq4_0_64(const block_tbq4_0_64 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ_64 == 0);
    const float * signs = tq_get_signs(QK_TQ_64);
    const float * cb = tq4_codebook_for(QK_TQ_64);
    for (int64_t i = 0; i < k / QK_TQ_64; i++)
        tq4_dequantize_block(x[i].qs, x[i].d, y + i*QK_TQ_64, QK_TQ_64, signs, cb);
}

size_t quantize_tbq4_0_64(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void)quant_weights;
    quantize_row_tbq4_0_64_ref(src, dst, nrow*n_per_row);
    return nrow * ggml_row_size(GGML_TYPE_TBQ4_0_64, n_per_row);
}

// ====================== PolarQuant (Stage 1 only, no QJL) ======================
// PQ3 uses identical codebook logic to TQ3 Stage 1 — thin wrappers over shared helpers.

void quantize_row_pq3_0_ref(const float * GGML_RESTRICT x, block_pq3_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ == 0);
    const float * signs = tq_get_signs(QK_TQ);
    const float * cb = tq3_codebook_for(QK_TQ);
    for (int64_t i = 0; i < k / QK_TQ; i++)
        tq3_quantize_block(x + i*QK_TQ, y[i].qs, &y[i].d, QK_TQ, PQ3_0_INDEX_BYTES, signs, cb);
}

void dequantize_row_pq3_0(const block_pq3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ == 0);
    const float * signs = tq_get_signs(QK_TQ);
    const float * cb = tq3_codebook_for(QK_TQ);
    for (int64_t i = 0; i < k / QK_TQ; i++)
        tq3_dequantize_block(x[i].qs, x[i].d, y + i*QK_TQ, QK_TQ, signs, cb);
}

size_t quantize_pq3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void)quant_weights;
    quantize_row_pq3_0_ref(src, dst, nrow*n_per_row);
    return nrow * ggml_row_size(GGML_TYPE_PQ3_0, n_per_row);
}

void quantize_row_pq3_0_64_ref(const float * GGML_RESTRICT x, block_pq3_0_64 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ_64 == 0);
    const float * signs = tq_get_signs(QK_TQ_64);
    const float * cb = tq3_codebook_for(QK_TQ_64);
    for (int64_t i = 0; i < k / QK_TQ_64; i++)
        tq3_quantize_block(x + i*QK_TQ_64, y[i].qs, &y[i].d, QK_TQ_64, PQ3_0_64_INDEX_BYTES, signs, cb);
}

void dequantize_row_pq3_0_64(const block_pq3_0_64 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ_64 == 0);
    const float * signs = tq_get_signs(QK_TQ_64);
    const float * cb = tq3_codebook_for(QK_TQ_64);
    for (int64_t i = 0; i < k / QK_TQ_64; i++)
        tq3_dequantize_block(x[i].qs, x[i].d, y + i*QK_TQ_64, QK_TQ_64, signs, cb);
}

size_t quantize_pq3_0_64(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void)quant_weights;
    quantize_row_pq3_0_64_ref(src, dst, nrow*n_per_row);
    return nrow * ggml_row_size(GGML_TYPE_PQ3_0_64, n_per_row);
}

// ====================== PQ4 (Stage 1 only, no QJL) ======================

void quantize_row_pq4_0_ref(const float * GGML_RESTRICT x, block_pq4_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ == 0);
    const float * signs = tq_get_signs(QK_TQ);
    const float * cb = tq4_codebook_for(QK_TQ);
    for (int64_t i = 0; i < k / QK_TQ; i++)
        tq4_quantize_block(x + i*QK_TQ, y[i].qs, &y[i].d, QK_TQ, PQ4_0_INDEX_BYTES, signs, cb);
}

void dequantize_row_pq4_0(const block_pq4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ == 0);
    const float * signs = tq_get_signs(QK_TQ);
    const float * cb = tq4_codebook_for(QK_TQ);
    for (int64_t i = 0; i < k / QK_TQ; i++)
        tq4_dequantize_block(x[i].qs, x[i].d, y + i*QK_TQ, QK_TQ, signs, cb);
}

size_t quantize_pq4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void)quant_weights;
    quantize_row_pq4_0_ref(src, dst, nrow*n_per_row);
    return nrow * ggml_row_size(GGML_TYPE_PQ4_0, n_per_row);
}

void quantize_row_pq4_0_64_ref(const float * GGML_RESTRICT x, block_pq4_0_64 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ_64 == 0);
    const float * signs = tq_get_signs(QK_TQ_64);
    const float * cb = tq4_codebook_for(QK_TQ_64);
    for (int64_t i = 0; i < k / QK_TQ_64; i++)
        tq4_quantize_block(x + i*QK_TQ_64, y[i].qs, &y[i].d, QK_TQ_64, PQ4_0_64_INDEX_BYTES, signs, cb);
}

void dequantize_row_pq4_0_64(const block_pq4_0_64 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ_64 == 0);
    const float * signs = tq_get_signs(QK_TQ_64);
    const float * cb = tq4_codebook_for(QK_TQ_64);
    for (int64_t i = 0; i < k / QK_TQ_64; i++)
        tq4_dequantize_block(x[i].qs, x[i].d, y + i*QK_TQ_64, QK_TQ_64, signs, cb);
}

size_t quantize_pq4_0_64(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void)quant_weights;
    quantize_row_pq4_0_64_ref(src, dst, nrow*n_per_row);
    return nrow * ggml_row_size(GGML_TYPE_PQ4_0_64, n_per_row);
}

// ====================== QJL Stage 2 helpers ======================
// The QJL sketch uses m independent randomized Hadamard projections applied to
// the quantization residual.  For d=m (sketch_dim == block_size), a single
// Hadamard transform with an independent sign array suffices: each coordinate
// of the transformed residual is one projection.  We store sign(transformed[j])
// as packed bits and ||residual|| as a scalar, giving an unbiased inner-product
// estimator via the asymmetric formula:
//   correction = sqrt(pi/2) / m * ||r_k|| * sum_j( sign_k_j * (R * q)_j )
// where R is the same structured random projection applied on-the-fly to the query.

// Use distinct seeds from the main Hadamard signs to get independent projections.
#define QJL_SIGN_SEED_128 137
#define QJL_SIGN_SEED_64  139

static float   qjl_signs_128[QK_TQ];
static float   qjl_signs_64[QK_TQ_64];
static int32_t qjl_signs_128_ready = 0;
static int32_t qjl_signs_64_ready  = 0;

static const float * qjl_get_signs(int d) {
    if (d == QK_TQ) {
        if (!qjl_signs_128_ready) { tq_generate_signs(qjl_signs_128, QK_TQ, QJL_SIGN_SEED_128); qjl_signs_128_ready = 1; }
        return qjl_signs_128;
    }
    if (!qjl_signs_64_ready) { tq_generate_signs(qjl_signs_64, QK_TQ_64, QJL_SIGN_SEED_64); qjl_signs_64_ready = 1; }
    return qjl_signs_64;
}

// Apply QJL projection in-place: buf = H * D_qjl * buf
// This is a randomized Hadamard with a *different* sign diagonal than Stage 1.
// No 1/sqrt(d) normalization — the scale factor sqrt(pi/2)/d in qjl_dot_correction
// expects unnormalized H*D, matching the QJL paper (Zandieh et al., 2024).
static void qjl_project_inplace(float * buf, int d, const float * qjl_signs_arr) {
    for (int i = 0; i < d; i++) buf[i] *= qjl_signs_arr[i];
    tq_fht(buf, d);
}

// Compute QJL sketch: project residual, take sign bits, store packed + norm.
static void qjl_encode_residual(const float * residual, int d,
                                 uint8_t * qjl_out, int qjl_bytes,
                                 ggml_half * d_r_out) {
    float r_norm = 0.0f;
    for (int j = 0; j < d; j++) r_norm += residual[j] * residual[j];
    r_norm = sqrtf(r_norm);
    *d_r_out = GGML_FP32_TO_FP16(r_norm);

    if (r_norm < 1e-15f) { memset(qjl_out, 0, qjl_bytes); return; }

    float tmp[128]; // max block size
    memcpy(tmp, residual, d * sizeof(float));

    const float * qjl_signs_arr = qjl_get_signs(d);
    qjl_project_inplace(tmp, d, qjl_signs_arr);

    // Pack projected residual signs into a bitfield: bit j=1 means the j-th
    // projected component is positive. j/8 selects the byte, 1<<(j%8) selects
    // the bit within that byte. This 1-bit sketch is used during attention to
    // approximate the residual dot product via the QJL estimator.
    memset(qjl_out, 0, qjl_bytes);
    for (int j = 0; j < d; j++) {
        if (tmp[j] > 0.0f) {
            qjl_out[j / 8] |= (1 << (j % 8));
        }
    }
}

// Compute QJL dot product correction: estimate <residual, b>
// QJL paper (Zandieh et al., 2024), Eq. 4:
//   score = √(π/2) / m * ||r|| * Σ_j sign((S r)_j) * (S b)_j
// where S has rows of norm ~√d. Our qjl_project_inplace uses R = H*D
// (unnormalized, rows of norm √d), so scale = √(π/2) / d matches directly.
float qjl_dot_correction(const uint8_t * qjl_bits, float d_r,
                          const float * b, int d) {
    if (d_r < 1e-15f) return 0.0f;

    float proj_b[128];
    memcpy(proj_b, b, d * sizeof(float));
    const float * qjl_signs_arr = qjl_get_signs(d);
    qjl_project_inplace(proj_b, d, qjl_signs_arr);

    float sum = 0.0f;
    for (int j = 0; j < d; j++) {
        float sign_j = ((qjl_bits[j / 8] >> (j % 8)) & 1) ? 1.0f : -1.0f;
        sum += sign_j * proj_b[j];
    }
    // Reference (Zandieh et al.): scale = √(π/2) / sketch_dim.
    // Our qjl_project_inplace normalizes by 1/√d on both encode and decode sides,
    // so the combined projection is (1/d) H D, matching the reference's 1/d factor.
    // The remaining correction is √(π/2) for the 1-bit sign quantization.
    const float scale = sqrtf(1.5707963f) / (float)d;  // √(π/2) / d
    return d_r * scale * sum;
}

// ====================== TQ3 (Stage 1 + QJL Stage 2) quantize/dequantize ======================

void quantize_row_tbq3_0_ref(const float * GGML_RESTRICT x, block_tbq3_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ == 0);
    const float * signs = tq_get_signs(QK_TQ);
    const float * cb = tq3_codebook_for(QK_TQ);

    float dequant_buf[QK_TQ];
    float residual[QK_TQ];

    for (int64_t i = 0; i < k / QK_TQ; i++) {
        const float * src = x + i * QK_TQ;

        // Stage 1: codebook quantize (identical to PQ3)
        tq3_quantize_block(src, y[i].qs, &y[i].d, QK_TQ, TBQ3_0_INDEX_BYTES, signs, cb);

        // Stage 2: compute residual and QJL sketch
        tq3_dequantize_block(y[i].qs, y[i].d, dequant_buf, QK_TQ, signs, cb);
        for (int j = 0; j < QK_TQ; j++) residual[j] = src[j] - dequant_buf[j];
        qjl_encode_residual(residual, QK_TQ, y[i].qjl, QJL_SKETCH_BYTES_128, &y[i].d_r);
    }
}

void dequantize_row_tbq3_0(const block_tbq3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ == 0);
    const float * signs = tq_get_signs(QK_TQ);
    const float * cb = tq3_codebook_for(QK_TQ);
    // Dequantize uses Stage 1 only — QJL correction is applied during dot product
    for (int64_t i = 0; i < k / QK_TQ; i++)
        tq3_dequantize_block(x[i].qs, x[i].d, y + i*QK_TQ, QK_TQ, signs, cb);
}

size_t quantize_tbq3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void)quant_weights;
    quantize_row_tbq3_0_ref(src, dst, nrow*n_per_row);
    return nrow * ggml_row_size(GGML_TYPE_TBQ3_0, n_per_row);
}

// --- block=64 ---

void quantize_row_tbq3_0_64_ref(const float * GGML_RESTRICT x, block_tbq3_0_64 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ_64 == 0);
    const float * signs = tq_get_signs(QK_TQ_64);
    const float * cb = tq3_codebook_for(QK_TQ_64);

    float dequant_buf[QK_TQ_64];
    float residual[QK_TQ_64];

    for (int64_t i = 0; i < k / QK_TQ_64; i++) {
        const float * src = x + i * QK_TQ_64;
        tq3_quantize_block(src, y[i].qs, &y[i].d, QK_TQ_64, TBQ3_0_64_INDEX_BYTES, signs, cb);

        tq3_dequantize_block(y[i].qs, y[i].d, dequant_buf, QK_TQ_64, signs, cb);
        for (int j = 0; j < QK_TQ_64; j++) residual[j] = src[j] - dequant_buf[j];
        qjl_encode_residual(residual, QK_TQ_64, y[i].qjl, QJL_SKETCH_BYTES_64, &y[i].d_r);
    }
}

void dequantize_row_tbq3_0_64(const block_tbq3_0_64 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ_64 == 0);
    const float * signs = tq_get_signs(QK_TQ_64);
    const float * cb = tq3_codebook_for(QK_TQ_64);
    for (int64_t i = 0; i < k / QK_TQ_64; i++)
        tq3_dequantize_block(x[i].qs, x[i].d, y + i*QK_TQ_64, QK_TQ_64, signs, cb);
}

size_t quantize_tbq3_0_64(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void)quant_weights;
    quantize_row_tbq3_0_64_ref(src, dst, nrow*n_per_row);
    return nrow * ggml_row_size(GGML_TYPE_TBQ3_0_64, n_per_row);
}
