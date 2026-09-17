#include "xdna-quant.h"

#include "ggml-impl.h"
#include "ggml-quants.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <set>
#include <vector>

namespace {

// Split an f32 into two bf16 values that sum to it: the rounded value and the
// residual. Together they carry ~16 mantissa bits, and the device expands them
// with two native bf16 multiply-accumulates.
inline void split_bf16(float v, uint16_t * hi, uint16_t * lo) {
    ggml_bf16_t h;
    ggml_fp32_to_bf16_row(&v, &h, 1);
    const float r = v - ggml_bf16_to_fp32(h);
    ggml_bf16_t l;
    ggml_fp32_to_bf16_row(&r, &l, 1);
    *hi = h.bits;
    *lo = l.bits;
}

inline float join_bf16(uint16_t hi, uint16_t lo) {
    ggml_bf16_t h;
    ggml_bf16_t l;
    h.bits = hi;
    l.bits = lo;
    return ggml_bf16_to_fp32(h) + ggml_bf16_to_fp32(l);
}

// Q4_K packs the 6-bit sub-block scale and min of eight 32-value groups into
// 12 bytes; this is the ggml unpacking (ggml-quants.c, get_scale_min_k4).
inline void q4k_scale_min(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>   4) | ((q[j - 0] >> 6) << 4);
    }
}

// Repack one Q4_K super-block (256 values) into eight q4g32 groups.
//
// ggml stores a group's nibbles interleaved: within each 64-value half, the
// even group is the low nibbles of 32 bytes and the odd group the high
// nibbles of the same bytes. q4g32 stores each group densely instead, value i
// in the low nibble of byte i/2 for even i and the high nibble for odd i,
// which is the order the AIE unpack consumes.
void repack_q4k_block(const block_q4_K * b, uint8_t * dst) {
    const float d    = GGML_FP16_TO_FP32(b->d);
    const float dmin = GGML_FP16_TO_FP32(b->dmin);
    constexpr int NG = XDNA_Q4G32_SB_GROUPS;

    std::memset(dst, 0, XDNA_Q4G32_SB_BYTES);
    int8_t * d8 = (int8_t *) (dst + NG * XDNA_Q4G32_CODE);
    int8_t * m8 = d8 + NG;
    for (int g = 0; g < NG; g++) {
        uint8_t sc = 0;
        uint8_t mn = 0;
        q4k_scale_min(g, b->scales, &sc, &mn);

        const uint8_t * q = b->qs + (g / 2) * 32;   // the 32 bytes of this half
        const bool high  = (g & 1) != 0;            // odd group = high nibbles

        uint8_t * out = dst + (size_t) g * XDNA_Q4G32_CODE;
        for (int i = 0; i < XDNA_Q4G32_GROUP; i++) {
            const uint8_t v = high ? (uint8_t) (q[i] >> 4) : (uint8_t) (q[i] & 0x0F);
            out[i >> 1] |= (uint8_t) (i & 1 ? (v << 4) : v);
        }
        // w = q * (d * sc) + (-dmin * mn): the 6-bit sc and mn are the record's
        // int8s and d, -dmin the pair they scale, so this is exact.
        d8[g] = (int8_t) sc;
        m8[g] = (int8_t) mn;
    }
    uint16_t p[4];
    split_bf16(d,     &p[0], &p[1]);
    split_bf16(-dmin, &p[2], &p[3]);
    std::memcpy(dst + NG * XDNA_Q4G32_CODE + 2 * NG, p, sizeof(p));
}

// Repack one Q6_K super-block (256 values) into sixteen q8g16 groups. Q6_K is
// symmetric with one int8 scale per 16 values, so the mapping is 1:1 and the
// 6-bit code (-32..31) fits an int8 exactly.
void repack_q6k_block(const block_q6_K * b, uint8_t * dst) {
    const float d = GGML_FP16_TO_FP32(b->d);

    int8_t q[QK_K];
    const uint8_t * ql = b->ql;
    const uint8_t * qh = b->qh;
    int8_t * y = q;
    for (int n = 0; n < QK_K; n += 128) {
        for (int l = 0; l < 32; l++) {
            y[l +  0] = (int8_t) ((ql[l +  0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
            y[l + 32] = (int8_t) ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
            y[l + 64] = (int8_t) ((ql[l +  0] >>   4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            y[l + 96] = (int8_t) ((ql[l + 32] >>   4) | (((qh[l] >> 6) & 3) << 4)) - 32;
        }
        ql += 64;
        qh += 32;
        y  += 128;
    }

    constexpr int NG = XDNA_Q8G16_SB_GROUPS;
    std::memset(dst, 0, XDNA_Q8G16_SB_BYTES);
    int8_t * d8 = (int8_t *) (dst + NG * XDNA_Q8G16_CODE);
    for (int g = 0; g < NG; g++) {
        std::memcpy(dst + (size_t) g * XDNA_Q8G16_CODE,
                    q + g * XDNA_Q8G16_GROUP, XDNA_Q8G16_GROUP);
        d8[g] = b->scales[g];   // already the int8 the f16 d scales
    }
    // Q6_K is symmetric: the min is zero, and so is every m8.
    uint16_t p[4] = { 0, 0, 0, 0 };
    split_bf16(d, &p[0], &p[1]);
    std::memcpy(dst + NG * XDNA_Q8G16_CODE + 2 * NG, p, sizeof(p));
}

// Q4_K into q8g16: the same affine parameters as q4g32 carries, but with the
// codes widened to int8 and written to both 16-halves of each 32-group.
void repack_q4k_block_g16(const block_q4_K * b, uint8_t * dst) {
    const float d    = GGML_FP16_TO_FP32(b->d);
    const float dmin = GGML_FP16_TO_FP32(b->dmin);

    constexpr int NG = XDNA_Q8G16_SB_GROUPS;
    std::memset(dst, 0, XDNA_Q8G16_SB_BYTES);
    int8_t * d8 = (int8_t *) (dst + NG * XDNA_Q8G16_CODE);
    int8_t * m8 = d8 + NG;
    for (int g = 0; g < 8; g++) {
        uint8_t sc = 0;
        uint8_t mn = 0;
        q4k_scale_min(g, b->scales, &sc, &mn);
        const uint8_t * q = b->qs + (g / 2) * 32;
        const bool high = (g & 1) != 0;

        for (int h = 0; h < 2; h++) {
            uint8_t * out = dst + (size_t) (2 * g + h) * XDNA_Q8G16_CODE;
            for (int i = 0; i < XDNA_Q8G16_GROUP; i++) {
                const int k = h * XDNA_Q8G16_GROUP + i;
                out[i] = high ? (uint8_t) (q[k] >> 4) : (uint8_t) (q[k] & 0x0F);
            }
            d8[2 * g + h] = (int8_t) sc;
            m8[2 * g + h] = (int8_t) mn;
        }
    }
    uint16_t p[4];
    split_bf16(d,     &p[0], &p[1]);
    split_bf16(-dmin, &p[2], &p[3]);
    std::memcpy(dst + NG * XDNA_Q8G16_CODE + 2 * NG, p, sizeof(p));
}

// Q5_K has Q4_K's affine per-32 structure with a fifth bit in a separate
// plane, so the code is 0..31 and the parameters are shared by both halves.
void repack_q5k_block(const block_q5_K * b, uint8_t * dst) {
    const float d    = GGML_FP16_TO_FP32(b->d);
    const float dmin = GGML_FP16_TO_FP32(b->dmin);

    constexpr int NG = XDNA_Q8G16_SB_GROUPS;
    std::memset(dst, 0, XDNA_Q8G16_SB_BYTES);
    int8_t * d8 = (int8_t *) (dst + NG * XDNA_Q8G16_CODE);
    int8_t * m8 = d8 + NG;
    for (int g = 0; g < 8; g++) {
        uint8_t sc = 0;
        uint8_t mn = 0;
        q4k_scale_min(g, b->scales, &sc, &mn);
        const uint8_t * ql = b->qs + (g / 2) * 32;
        const uint8_t * qh = b->qh;
        const bool high = (g & 1) != 0;

        for (int h = 0; h < 2; h++) {
            uint8_t * out = dst + (size_t) (2 * g + h) * XDNA_Q8G16_CODE;
            for (int i = 0; i < XDNA_Q8G16_GROUP; i++) {
                const int k  = h * XDNA_Q8G16_GROUP + i;
                const int lo = high ? (ql[k] >> 4) : (ql[k] & 0x0F);
                const int hi = (qh[k] >> g) & 1;
                out[i] = (uint8_t) (lo | (hi << 4));
            }
            d8[2 * g + h] = (int8_t) sc;
            m8[2 * g + h] = (int8_t) mn;
        }
    }
    uint16_t p[4];
    split_bf16(d,     &p[0], &p[1]);
    split_bf16(-dmin, &p[2], &p[3]);
    std::memcpy(dst + NG * XDNA_Q8G16_CODE + 2 * NG, p, sizeof(p));
}

} // namespace

xdna_wfmt xdna_wfmt_for(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_K: return XDNA_WFMT_Q4G32;
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K: return XDNA_WFMT_Q8G16;
        default:             return XDNA_WFMT_NONE;
    }
}

size_t xdna_wfmt_row_bytes(xdna_wfmt fmt, int64_t k) {
    // A record is a super-block either way, so a row is whole records.
    if (k % XDNA_SB_VALUES) {
        return 0;
    }
    const int64_t nsb = k / XDNA_SB_VALUES;
    switch (fmt) {
        case XDNA_WFMT_Q4G32: return (size_t) nsb * XDNA_Q4G32_SB_BYTES;
        case XDNA_WFMT_Q8G16: return (size_t) nsb * XDNA_Q8G16_SB_BYTES;
        default:              return 0;
    }
}

xdna_wfmt xdna_wfmt_gemv_for(enum ggml_type type) {
    switch (type) {
        // Q4_K keeps its 4-bit codes: both widths are the same tile size, so
        // one artifact streams either and the denser form costs nothing.
        case GGML_TYPE_Q4_K: return XDNA_WFMT_Q4G32;
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K: return XDNA_WFMT_Q8G16;
        default:             return XDNA_WFMT_NONE;
    }
}

bool xdna_wfmt_repack_row_as(enum ggml_type type, xdna_wfmt fmt, const void * src,
                             int64_t k, void * dst) {
    if (!src || !dst || k <= 0 || k % QK_K) {
        return false;
    }
    if (fmt != XDNA_WFMT_Q8G16 || type != GGML_TYPE_Q4_K) {
        return xdna_wfmt_repack_row(type, src, k, dst);
    }
    // Q4_K into the 8-bit affine form, so one format covers the whole decode.
    const block_q4_K * b = (const block_q4_K *) src;
    uint8_t * out = (uint8_t *) dst;
    for (int64_t i = 0; i < k / QK_K; i++) {
        repack_q4k_block_g16(b + i, out + (size_t) i * XDNA_Q8G16_SB_BYTES);
    }
    return true;
}

bool xdna_wfmt_repack_row(enum ggml_type type, const void * src, int64_t k, void * dst) {
    if (!src || !dst || k <= 0 || k % QK_K) {
        return false;
    }
    const int64_t nb = k / QK_K;
    uint8_t * out = (uint8_t *) dst;

    switch (type) {
        case GGML_TYPE_Q4_K: {
            const block_q4_K * b = (const block_q4_K *) src;
            for (int64_t i = 0; i < nb; i++) {
                repack_q4k_block(b + i, out + (size_t) i * XDNA_Q4G32_SB_BYTES);
            }
            return true;
        }
        case GGML_TYPE_Q6_K: {
            const block_q6_K * b = (const block_q6_K *) src;
            for (int64_t i = 0; i < nb; i++) {
                repack_q6k_block(b + i, out + (size_t) i * XDNA_Q8G16_SB_BYTES);
            }
            return true;
        }
        case GGML_TYPE_Q5_K: {
            const block_q5_K * b = (const block_q5_K *) src;
            for (int64_t i = 0; i < nb; i++) {
                repack_q5k_block(b + i, out + (size_t) i * XDNA_Q8G16_SB_BYTES);
            }
            return true;
        }
        default:
            return false;
    }
}

bool xdna_wfmt_decode_row(xdna_wfmt fmt, const void * src, int64_t k, float * dst) {
    if (!src || !dst || k <= 0) {
        return false;
    }
    const uint8_t * in = (const uint8_t *) src;

    if (k % XDNA_SB_VALUES) {
        return false;
    }
    const int64_t nsb = k / XDNA_SB_VALUES;

    if (fmt == XDNA_WFMT_Q4G32) {
        constexpr int NG = XDNA_Q4G32_SB_GROUPS;
        for (int64_t sb = 0; sb < nsb; sb++) {
            const uint8_t * rec = in + (size_t) sb * XDNA_Q4G32_SB_BYTES;
            const int8_t * d8 = (const int8_t *) (rec + NG * XDNA_Q4G32_CODE);
            const int8_t * m8 = d8 + NG;
            uint16_t q[4];
            std::memcpy(q, rec + NG * XDNA_Q4G32_CODE + 2 * NG, sizeof(q));
            const float dS = join_bf16(q[0], q[1]);
            const float mS = join_bf16(q[2], q[3]);
            for (int g = 0; g < NG; g++) {
                const uint8_t * p = rec + (size_t) g * XDNA_Q4G32_CODE;
                const float d = dS * (float) d8[g];
                const float m = mS * (float) m8[g];
                float * y = dst + (sb * NG + g) * XDNA_Q4G32_GROUP;
                for (int i = 0; i < XDNA_Q4G32_GROUP; i++) {
                    const uint8_t v = (i & 1) ? (uint8_t) (p[i >> 1] >> 4)
                                              : (uint8_t) (p[i >> 1] & 0x0F);
                    y[i] = (float) v * d + m;
                }
            }
        }
        return true;
    }

    if (fmt == XDNA_WFMT_Q8G16) {
        constexpr int NG = XDNA_Q8G16_SB_GROUPS;
        for (int64_t sb = 0; sb < nsb; sb++) {
            const uint8_t * rec = in + (size_t) sb * XDNA_Q8G16_SB_BYTES;
            const int8_t * d8 = (const int8_t *) (rec + NG * XDNA_Q8G16_CODE);
            const int8_t * m8 = d8 + NG;
            uint16_t q[4];
            std::memcpy(q, rec + NG * XDNA_Q8G16_CODE + 2 * NG, sizeof(q));
            const float dS = join_bf16(q[0], q[1]);
            const float mS = join_bf16(q[2], q[3]);
            for (int g = 0; g < NG; g++) {
                const uint8_t * p = rec + (size_t) g * XDNA_Q8G16_CODE;
                const float d = dS * (float) d8[g];
                const float m = mS * (float) m8[g];
                float * y = dst + (sb * NG + g) * XDNA_Q8G16_GROUP;
                for (int i = 0; i < XDNA_Q8G16_GROUP; i++) {
                    y[i] = (float) (int8_t) p[i] * d + m;
                }
            }
        }
        return true;
    }

    return false;
}


