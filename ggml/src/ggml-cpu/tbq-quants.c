#include "ggml-cpu-impl.h"
#include "ggml-quants.h"
#include "tbq-quants.h"

#define UNUSED GGML_UNUSED

void quantize_row_tbq3_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_TQ == 0);
    block_tbq3_0 * GGML_RESTRICT y = vy;
    quantize_row_tbq3_0_ref(x, y, k);
}

void quantize_row_tbq4_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_TQ == 0);
    block_tbq4_0 * GGML_RESTRICT y = vy;
    quantize_row_tbq4_0_ref(x, y, k);
}

// TurboQuant vec_dot: Stage 1 dot + QJL Stage 2 correction
void ggml_vec_dot_tbq3_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_TQ == 0);
    assert(nrc == 1);
    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);

    const block_tbq3_0 * GGML_RESTRICT x = (const block_tbq3_0 *)vx;
    const block_q8_0   * GGML_RESTRICT y = (const block_q8_0 *)vy;
    const int nb_tq = n / QK_TQ;
    const int M     = QK_TQ / QK8_0;
    float tmp_x[QK_TQ];
    float tmp_y_full[QK_TQ];
    float sumf = 0.0f;
    for (int i = 0; i < nb_tq; i++) {
        GGML_CPU_PREFETCH(&x[i + 1], 0, 0);
        GGML_CPU_PREFETCH(&y[(i + 1) * M], 0, 0);
        dequantize_row_tbq3_0(x + i, tmp_x, QK_TQ);
        float base = 0.0f;
        for (int j = 0; j < M; j++) {
            float       * GGML_RESTRICT yb = &tmp_y_full[j * QK8_0];
            const float * GGML_RESTRICT xb = &tmp_x[j * QK8_0];
            dequantize_row_q8_0(&y[i * M + j], yb, QK8_0);
            float acc = 0.0f;
            for (int k = 0; k < QK8_0; k++) {
                acc += xb[k] * yb[k];
            }
            base += acc;
        }
        const float d_r = GGML_FP16_TO_FP32(x[i].d_r);
        sumf += base + qjl_dot_correction(x[i].qjl, d_r, tmp_y_full, QK_TQ);
    }
    *s = sumf;
}

void ggml_vec_dot_tbq3_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    ggml_vec_dot_tbq3_0_q8_0(n, s, bs, vx, bx, vy, by, nrc);
}

void ggml_vec_dot_tbq4_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_TQ == 0);
    assert(nrc == 1);
    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);

    const block_tbq4_0 * GGML_RESTRICT x = (const block_tbq4_0 *)vx;
    const block_q8_0   * GGML_RESTRICT y = (const block_q8_0 *)vy;
    const int nb_tq = n / QK_TQ;
    const int M     = QK_TQ / QK8_0;
    float tmp_x[QK_TQ];
    float tmp_y_full[QK_TQ];
    float sumf = 0.0f;
    for (int i = 0; i < nb_tq; i++) {
        GGML_CPU_PREFETCH(&x[i + 1], 0, 0);
        GGML_CPU_PREFETCH(&y[(i + 1) * M], 0, 0);
        dequantize_row_tbq4_0(x + i, tmp_x, QK_TQ);
        float base = 0.0f;
        for (int j = 0; j < M; j++) {
            float       * GGML_RESTRICT yb = &tmp_y_full[j * QK8_0];
            const float * GGML_RESTRICT xb = &tmp_x[j * QK8_0];
            dequantize_row_q8_0(&y[i * M + j], yb, QK8_0);
            float acc = 0.0f;
            for (int k = 0; k < QK8_0; k++) {
                acc += xb[k] * yb[k];
            }
            base += acc;
        }
        const float d_r = GGML_FP16_TO_FP32(x[i].d_r);
        sumf += base + qjl_dot_correction(x[i].qjl, d_r, tmp_y_full, QK_TQ);
    }
    *s = sumf;
}

void ggml_vec_dot_tbq4_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    ggml_vec_dot_tbq4_0_q8_0(n, s, bs, vx, bx, vy, by, nrc);
}

// --- block=64 CPU wrappers ---

void quantize_row_tbq3_0_64(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_TQ_64 == 0);
    block_tbq3_0_64 * GGML_RESTRICT y = vy;
    quantize_row_tbq3_0_64_ref(x, y, k);
}

void quantize_row_tbq4_0_64(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_TQ_64 == 0);
    block_tbq4_0_64 * GGML_RESTRICT y = vy;
    quantize_row_tbq4_0_64_ref(x, y, k);
}

void ggml_vec_dot_tbq3_0_64_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_TQ_64 == 0);
    assert(nrc == 1);
    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);

    float tmp_x[QK_TQ_64];
    float tmp_y_full[QK_TQ_64];
    float tmp_y[QK8_0];
    float sumf = 0.0f;
    int pos = 0;

    const block_tbq3_0_64 * GGML_RESTRICT x = (const block_tbq3_0_64 *)vx;
    const block_q8_0     * GGML_RESTRICT y = (const block_q8_0 *)vy;

    const int nb_tq = n / QK_TQ_64;

    for (int i = 0; i < nb_tq; i++) {
        dequantize_row_tbq3_0_64(x + i, tmp_x, QK_TQ_64);

        for (int j = 0; j < QK_TQ_64 / QK8_0; j++) {
            dequantize_row_q8_0(y + pos, tmp_y, QK8_0);
            for (int k = 0; k < QK8_0; k++) {
                tmp_y_full[j * QK8_0 + k] = tmp_y[k];
                sumf += tmp_x[j * QK8_0 + k] * tmp_y[k];
            }
            pos++;
        }

        float d_r = GGML_FP16_TO_FP32(x[i].d_r);
        sumf += qjl_dot_correction(x[i].qjl, d_r, tmp_y_full, QK_TQ_64);
    }
    *s = sumf;
}

void ggml_vec_dot_tbq3_0_64_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    ggml_vec_dot_tbq3_0_64_q8_0(n, s, bs, vx, bx, vy, by, nrc);
}

void ggml_vec_dot_tbq4_0_64_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_TQ_64 == 0);
    assert(nrc == 1);
    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);

    float tmp_x[QK_TQ_64];
    float tmp_y_full[QK_TQ_64];
    float tmp_y[QK8_0];
    float sumf = 0.0f;
    int pos = 0;

    const block_tbq4_0_64 * GGML_RESTRICT x = (const block_tbq4_0_64 *)vx;
    const block_q8_0     * GGML_RESTRICT y = (const block_q8_0 *)vy;

    const int nb_tq = n / QK_TQ_64;

    for (int i = 0; i < nb_tq; i++) {
        dequantize_row_tbq4_0_64(x + i, tmp_x, QK_TQ_64);

        for (int j = 0; j < QK_TQ_64 / QK8_0; j++) {
            dequantize_row_q8_0(y + pos, tmp_y, QK8_0);
            for (int k = 0; k < QK8_0; k++) {
                tmp_y_full[j * QK8_0 + k] = tmp_y[k];
                sumf += tmp_x[j * QK8_0 + k] * tmp_y[k];
            }
            pos++;
        }

        float d_r = GGML_FP16_TO_FP32(x[i].d_r);
        sumf += qjl_dot_correction(x[i].qjl, d_r, tmp_y_full, QK_TQ_64);
    }
    *s = sumf;
}

void ggml_vec_dot_tbq4_0_64_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    ggml_vec_dot_tbq4_0_64_q8_0(n, s, bs, vx, bx, vy, by, nrc);
}

// ====================== PolarQuant CPU wrappers ======================
// PQ3 is Stage 1 only (no QJL). Uses identical codebook logic to TQ3's Stage 1.

void quantize_row_pq3_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_TQ == 0);
    block_pq3_0 * GGML_RESTRICT y = vy;
    quantize_row_pq3_0_ref(x, y, k);
}

void quantize_row_pq3_0_64(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_TQ_64 == 0);
    block_pq3_0_64 * GGML_RESTRICT y = vy;
    quantize_row_pq3_0_64_ref(x, y, k);
}

void ggml_vec_dot_pq3_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_TQ == 0);
    assert(nrc == 1);
    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);

    const block_pq3_0 * GGML_RESTRICT x = (const block_pq3_0 *)vx;
    const block_q8_0  * GGML_RESTRICT y = (const block_q8_0 *)vy;
    const int nb = n / QK_TQ;
    const int M  = QK_TQ / QK8_0;
    float tmp_x[QK_TQ];
    float tmp_y[QK8_0];
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        GGML_CPU_PREFETCH(&x[i + 1], 0, 0);
        GGML_CPU_PREFETCH(&y[(i + 1) * M], 0, 0);
        dequantize_row_pq3_0(x + i, tmp_x, QK_TQ);
        float base = 0.0f;
        for (int j = 0; j < M; j++) {
            const float * GGML_RESTRICT xb = &tmp_x[j * QK8_0];
            dequantize_row_q8_0(&y[i * M + j], tmp_y, QK8_0);
            float acc = 0.0f;
            for (int k = 0; k < QK8_0; k++) {
                acc += xb[k] * tmp_y[k];
            }
            base += acc;
        }
        sumf += base;
    }
    *s = sumf;
}

void ggml_vec_dot_pq3_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    ggml_vec_dot_pq3_0_q8_0(n, s, bs, vx, bx, vy, by, nrc);
}

void ggml_vec_dot_pq3_0_64_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_TQ_64 == 0);
    assert(nrc == 1);
    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);

    const block_pq3_0_64 * GGML_RESTRICT x = (const block_pq3_0_64 *)vx;
    const block_q8_0     * GGML_RESTRICT y = (const block_q8_0 *)vy;
    const int nb = n / QK_TQ_64;
    const int M  = QK_TQ_64 / QK8_0;
    float tmp_x[QK_TQ_64];
    float tmp_y[QK8_0];
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        GGML_CPU_PREFETCH(&x[i + 1], 0, 0);
        GGML_CPU_PREFETCH(&y[(i + 1) * M], 0, 0);
        dequantize_row_pq3_0_64(x + i, tmp_x, QK_TQ_64);
        float base = 0.0f;
        for (int j = 0; j < M; j++) {
            const float * GGML_RESTRICT xb = &tmp_x[j * QK8_0];
            dequantize_row_q8_0(&y[i * M + j], tmp_y, QK8_0);
            float acc = 0.0f;
            for (int k = 0; k < QK8_0; k++) {
                acc += xb[k] * tmp_y[k];
            }
            base += acc;
        }
        sumf += base;
    }
    *s = sumf;
}

void ggml_vec_dot_pq3_0_64_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    ggml_vec_dot_pq3_0_64_q8_0(n, s, bs, vx, bx, vy, by, nrc);
}

// ====================== PQ4 (Stage 1 only) CPU wrappers ======================

void quantize_row_pq4_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_TQ == 0);
    block_pq4_0 * GGML_RESTRICT y = vy;
    quantize_row_pq4_0_ref(x, y, k);
}

void quantize_row_pq4_0_64(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_TQ_64 == 0);
    block_pq4_0_64 * GGML_RESTRICT y = vy;
    quantize_row_pq4_0_64_ref(x, y, k);
}

void ggml_vec_dot_pq4_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_TQ == 0);
    assert(nrc == 1);
    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);

    const block_pq4_0 * GGML_RESTRICT x = (const block_pq4_0 *)vx;
    const block_q8_0  * GGML_RESTRICT y = (const block_q8_0 *)vy;
    const int nb = n / QK_TQ;
    const int M  = QK_TQ / QK8_0;
    float tmp_x[QK_TQ];
    float tmp_y[QK8_0];
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        GGML_CPU_PREFETCH(&x[i + 1], 0, 0);
        GGML_CPU_PREFETCH(&y[(i + 1) * M], 0, 0);
        dequantize_row_pq4_0(x + i, tmp_x, QK_TQ);
        float base = 0.0f;
        for (int j = 0; j < M; j++) {
            const float * GGML_RESTRICT xb = &tmp_x[j * QK8_0];
            dequantize_row_q8_0(&y[i * M + j], tmp_y, QK8_0);
            float acc = 0.0f;
            for (int k = 0; k < QK8_0; k++) {
                acc += xb[k] * tmp_y[k];
            }
            base += acc;
        }
        sumf += base;
    }
    *s = sumf;
}

void ggml_vec_dot_pq4_0_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    ggml_vec_dot_pq4_0_q8_0(n, s, bs, vx, bx, vy, by, nrc);
}

void ggml_vec_dot_pq4_0_64_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_TQ_64 == 0);
    assert(nrc == 1);
    UNUSED(nrc); UNUSED(bx); UNUSED(by); UNUSED(bs);

    const block_pq4_0_64 * GGML_RESTRICT x = (const block_pq4_0_64 *)vx;
    const block_q8_0     * GGML_RESTRICT y = (const block_q8_0 *)vy;
    const int nb = n / QK_TQ_64;
    const int M  = QK_TQ_64 / QK8_0;
    float tmp_x[QK_TQ_64];
    float tmp_y[QK8_0];
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        GGML_CPU_PREFETCH(&x[i + 1], 0, 0);
        GGML_CPU_PREFETCH(&y[(i + 1) * M], 0, 0);
        dequantize_row_pq4_0_64(x + i, tmp_x, QK_TQ_64);
        float base = 0.0f;
        for (int j = 0; j < M; j++) {
            const float * GGML_RESTRICT xb = &tmp_x[j * QK8_0];
            dequantize_row_q8_0(&y[i * M + j], tmp_y, QK8_0);
            float acc = 0.0f;
            for (int k = 0; k < QK8_0; k++) {
                acc += xb[k] * tmp_y[k];
            }
            base += acc;
        }
        sumf += base;
    }
    *s = sumf;
}

void ggml_vec_dot_pq4_0_64_q8_0_generic(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    ggml_vec_dot_pq4_0_64_q8_0(n, s, bs, vx, bx, vy, by, nrc);
}
