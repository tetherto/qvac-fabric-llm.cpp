#include "xdna-quant.h"

#include "ggml-impl.h"
#include "ggml-quants.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
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
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
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
    const float   d    = GGML_FP16_TO_FP32(b->d);
    const float   dmin = GGML_FP16_TO_FP32(b->dmin);
    constexpr int NG   = XDNA_Q4G32_SB_GROUPS;

    std::memset(dst, 0, XDNA_Q4G32_SB_BYTES);
    int8_t * d8 = (int8_t *) (dst + (size_t) NG * XDNA_Q4G32_CODE);
    int8_t * m8 = d8 + NG;
    for (int g = 0; g < NG; g++) {
        uint8_t sc = 0;
        uint8_t mn = 0;
        q4k_scale_min(g, b->scales, &sc, &mn);

        const uint8_t * q    = b->qs + (size_t) (g / 2) * 32;  // the 32 bytes of this half
        const bool      high = (g & 1) != 0;                   // odd group = high nibbles

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
    split_bf16(d, &p[0], &p[1]);
    split_bf16(-dmin, &p[2], &p[3]);
    std::memcpy(dst + (size_t) NG * XDNA_Q4G32_CODE + (size_t) 2 * NG, p, sizeof(p));
}

// Repack one Q6_K super-block (256 values) into eight 32-value groups. Q6_K
// is symmetric with one int8 scale per 16 values, so a group holds two of
// them: the group scale is the larger (by magnitude), the super-block's d is
// quartered, and each code becomes 4 * q * s / S, rounded - exact for the
// half whose scale is S (4 * q fits an int8 for q in -32..31), and two bits
// finer than the source's own step for the other.
void repack_q6k_block(const block_q6_K * b, uint8_t * dst) {
    const float d = GGML_FP16_TO_FP32(b->d);

    int8_t          q[QK_K];
    const uint8_t * ql = b->ql;
    const uint8_t * qh = b->qh;
    int8_t *        y  = q;
    for (int n = 0; n < QK_K; n += 128) {
        for (int l = 0; l < 32; l++) {
            y[l + 0]  = (int8_t) (((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32);
            y[l + 32] = (int8_t) (((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32);
            y[l + 64] = (int8_t) (((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
            y[l + 96] = (int8_t) (((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
        }
        ql += 64;
        qh += 32;
        y += 128;
    }

    constexpr int NG = XDNA_Q8G16_SB_GROUPS;
    constexpr int H  = XDNA_Q8G16_GROUP / 2;  // Q6_K's 16
    std::memset(dst, 0, XDNA_Q8G16_SB_BYTES);
    int8_t * d8 = (int8_t *) (dst + (size_t) NG * XDNA_Q8G16_CODE);
    for (int g = 0; g < NG; g++) {
        // NOLINTNEXTLINE(bugprone-signed-char-misuse, cert-str34-c): the quantization scales are signed
        const int s0 = b->scales[(size_t) 2 * g], s1 = b->scales[(size_t) 2 * g + 1];
        const int S   = std::abs(s0) >= std::abs(s1) ? s0 : s1;
        int8_t *  out = (int8_t *) (dst + (size_t) g * XDNA_Q8G16_CODE);
        for (int i = 0; S != 0 && i < XDNA_Q8G16_GROUP; i++) {
            const int sh = i < H ? s0 : s1;
            const int v  = (int) std::lrint(4.0 * q[g * XDNA_Q8G16_GROUP + i] * sh / S);
            out[i]       = (int8_t) std::min(127, std::max(-128, v));
        }
        d8[g] = (int8_t) S;
    }
    // Q6_K is symmetric: the min is zero, and so is every m8.
    uint16_t p[4] = { 0, 0, 0, 0 };
    split_bf16(d * 0.25f, &p[0], &p[1]);
    std::memcpy(dst + (size_t) NG * XDNA_Q8G16_CODE + (size_t) 2 * NG, p, sizeof(p));
}

// A super-block of values (256 f32) into the 8-bit form, symmetric: a group's
// scale is its step amax / 127 as an int8 multiple of the super-block's (the
// largest step / 127), rounded up, and the codes are the values in that
// step, rounded.
// For Q8_0, whose 32-value blocks are the groups, that is one extra rounding
// of the source's codes.
void pack_f32_block_q8g16(const float * v, uint8_t * dst) {
    constexpr int NG = XDNA_Q8G16_SB_GROUPS;
    float         step[NG];
    float         smax = 0.0f;
    for (int g = 0; g < NG; g++) {
        float amax = 0.0f;
        for (int i = 0; i < XDNA_Q8G16_GROUP; i++) {
            amax = std::max(amax, std::fabs(v[g * XDNA_Q8G16_GROUP + i]));
        }
        step[g] = amax / 127.0f;
        smax    = std::max(smax, step[g]);
    }
    const float D = smax / 127.0f;
    std::memset(dst, 0, XDNA_Q8G16_SB_BYTES);
    int8_t * d8 = (int8_t *) (dst + (size_t) NG * XDNA_Q8G16_CODE);
    for (int g = 0; D > 0.0f && g < NG; g++) {
        // rounded up, so no code is clipped
        const int   s   = std::min(127, std::max(1, (int) std::ceil(step[g] / D * (1.0f - 1e-6f))));
        const float inv = 1.0f / (D * (float) s);
        int8_t *    out = (int8_t *) (dst + (size_t) g * XDNA_Q8G16_CODE);
        for (int i = 0; i < XDNA_Q8G16_GROUP; i++) {
            const int q = (int) std::lrint(v[g * XDNA_Q8G16_GROUP + i] * inv);
            out[i]      = (int8_t) std::min(127, std::max(-127, q));
        }
        d8[g] = (int8_t) s;
    }
    uint16_t p[4] = { 0, 0, 0, 0 };
    split_bf16(D, &p[0], &p[1]);
    std::memcpy(dst + (size_t) NG * XDNA_Q8G16_CODE + (size_t) 2 * NG, p, sizeof(p));
}

// Q4_K into the 8-bit form: the same affine parameters as q4g32 carries, with
// the codes widened to int8.
void repack_q4k_block_g16(const block_q4_K * b, uint8_t * dst) {
    const float d    = GGML_FP16_TO_FP32(b->d);
    const float dmin = GGML_FP16_TO_FP32(b->dmin);

    constexpr int NG = XDNA_Q8G16_SB_GROUPS;
    std::memset(dst, 0, XDNA_Q8G16_SB_BYTES);
    int8_t * d8 = (int8_t *) (dst + (size_t) NG * XDNA_Q8G16_CODE);
    int8_t * m8 = d8 + NG;
    for (int g = 0; g < NG; g++) {
        uint8_t sc = 0;
        uint8_t mn = 0;
        q4k_scale_min(g, b->scales, &sc, &mn);
        const uint8_t * q    = b->qs + (size_t) (g / 2) * 32;
        const bool      high = (g & 1) != 0;
        uint8_t *       out  = dst + (size_t) g * XDNA_Q8G16_CODE;
        for (int i = 0; i < XDNA_Q8G16_GROUP; i++) {
            out[i] = high ? (uint8_t) (q[i] >> 4) : (uint8_t) (q[i] & 0x0F);
        }
        d8[g] = (int8_t) sc;
        m8[g] = (int8_t) mn;
    }
    uint16_t p[4];
    split_bf16(d, &p[0], &p[1]);
    split_bf16(-dmin, &p[2], &p[3]);
    std::memcpy(dst + (size_t) NG * XDNA_Q8G16_CODE + (size_t) 2 * NG, p, sizeof(p));
}

// Q5_K has Q4_K's affine per-32 structure with a fifth bit in a separate
// plane, so the code is 0..31 and a group is one of its 32-value blocks.
void repack_q5k_block(const block_q5_K * b, uint8_t * dst) {
    const float d    = GGML_FP16_TO_FP32(b->d);
    const float dmin = GGML_FP16_TO_FP32(b->dmin);

    constexpr int NG = XDNA_Q8G16_SB_GROUPS;
    std::memset(dst, 0, XDNA_Q8G16_SB_BYTES);
    int8_t * d8 = (int8_t *) (dst + (size_t) NG * XDNA_Q8G16_CODE);
    int8_t * m8 = d8 + NG;
    for (int g = 0; g < NG; g++) {
        uint8_t sc = 0;
        uint8_t mn = 0;
        q4k_scale_min(g, b->scales, &sc, &mn);
        const uint8_t * ql   = b->qs + (size_t) (g / 2) * 32;
        const uint8_t * qh   = b->qh;
        const bool      high = (g & 1) != 0;
        uint8_t *       out  = dst + (size_t) g * XDNA_Q8G16_CODE;
        for (int i = 0; i < XDNA_Q8G16_GROUP; i++) {
            const int lo = high ? (ql[i] >> 4) : (ql[i] & 0x0F);
            const int hi = (qh[i] >> g) & 1;
            out[i]       = (uint8_t) (lo | (hi << 4));
        }
        d8[g] = (int8_t) sc;
        m8[g] = (int8_t) mn;
    }
    uint16_t p[4];
    split_bf16(d, &p[0], &p[1]);
    split_bf16(-dmin, &p[2], &p[3]);
    std::memcpy(dst + (size_t) NG * XDNA_Q8G16_CODE + (size_t) 2 * NG, p, sizeof(p));
}

}  // namespace

size_t xdna_wfmt_row_bytes(xdna_wfmt fmt, int64_t k) {
    // A record is a super-block either way, so a row is whole records.
    if (k % XDNA_SB_VALUES) {
        return 0;
    }
    const int64_t nsb = k / XDNA_SB_VALUES;
    switch (fmt) {
        case XDNA_WFMT_Q4G32:
            return (size_t) nsb * XDNA_Q4G32_SB_BYTES;
        case XDNA_WFMT_Q8G16:
            return (size_t) nsb * XDNA_Q8G16_SB_BYTES;
        default:
            return 0;
    }
}

xdna_wfmt xdna_wfmt_gemv_for(enum ggml_type type) {
    switch (type) {
        // Q4_K keeps its 4-bit codes: both widths are the same tile size, so
        // one artifact streams either and the denser form costs nothing.
        case GGML_TYPE_Q4_K:
            return XDNA_WFMT_Q4G32;
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return XDNA_WFMT_Q8G16;
        default:
            return XDNA_WFMT_NONE;
    }
}

enum ggml_type xdna_gemv_type_of(const struct ggml_tensor * w) {
    static const std::vector<std::string> sel = [] {
        std::vector<std::string> v;
        // Every attn_qkv and attn_v by default: measured one tensor at a time
        // the KLD adds up, and these are the best tok/s per unit of it -
        // 47.5 -> 52.4 tok/s at 1k for decode KLD 0.0051 -> 0.0122. ffn_down
        // pays about half as much speed for the same accuracy.
        const char *             e = getenv("GGML_XDNA_W4");
        if (!e) {
            e = "attn_qkv,attn_v";
        }
        std::string cur;
        for (const char * p = e ? e : "";; p++) {
            if (*p == ',' || *p == '\0') {
                if (!cur.empty()) {
                    v.push_back(cur);
                }
                cur.clear();
                if (*p == '\0') {
                    break;
                }
            } else if (*p != ' ') {
                cur += *p;
            }
        }
        return v;
    }();
    if (!w || (w->type != GGML_TYPE_Q5_K && w->type != GGML_TYPE_Q6_K) || sel.empty()) {
        return w ? w->type : GGML_TYPE_COUNT;
    }
    std::string       name = w->name;
    const std::string sfx  = ".weight";
    if (name.size() > sfx.size() && name.compare(name.size() - sfx.size(), sfx.size(), sfx) == 0) {
        name.resize(name.size() - sfx.size());
    }
    if (name.find("ssm_out") != std::string::npos) {
        return w->type;
    }
    for (const std::string & e : sel) {
        if (name == e || (name.size() > e.size() && name.compare(name.size() - e.size(), e.size(), e) == 0 &&
                          name[name.size() - e.size() - 1] == '.')) {
            return GGML_TYPE_Q4_K;
        }
    }
    return w->type;
}

bool xdna_wfmt_repack_row_as(enum ggml_type type, xdna_wfmt fmt, const void * src, int64_t k, void * dst) {
    if (!src || !dst || k <= 0 || k % QK_K) {
        return false;
    }
    // The write below is sized by the requested format, not by the source type,
    // so a pair this function does not carry has to be refused here: forwarding
    // it to xdna_wfmt_repack_row would size the output by the source and write
    // past a buffer the caller sized for the format it asked for.
    const bool four_bit = type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K;
    const bool eight_bit = four_bit || type == GGML_TYPE_Q8_0;
    if (!((fmt == XDNA_WFMT_Q4G32 && four_bit) || (fmt == XDNA_WFMT_Q8G16 && eight_bit))) {
        return false;
    }
    if (fmt == XDNA_WFMT_Q4G32 && (type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K)) {
        // Chosen for the 4-bit form (xdna_gemv_type_of): re-quantized to Q4_K
        // from its dequantized values, then packed as one.
        std::vector<float>      f((size_t) k);
        std::vector<block_q4_K> q((size_t) (k / QK_K));
        ggml_get_type_traits(type)->to_float(src, f.data(), k);
        quantize_row_q4_K_ref(f.data(), q.data(), k);
        return xdna_wfmt_repack_row(GGML_TYPE_Q4_K, q.data(), k, dst);
    }
    if (fmt == XDNA_WFMT_Q8G16 && type == GGML_TYPE_Q8_0) {
        std::vector<float> f((size_t) k);
        ggml_get_type_traits(type)->to_float(src, f.data(), k);
        for (int64_t i = 0; i < k / QK_K; i++) {
            pack_f32_block_q8g16(f.data() + i * QK_K, (uint8_t *) dst + (size_t) i * XDNA_Q8G16_SB_BYTES);
        }
        return true;
    }
    if (fmt != XDNA_WFMT_Q8G16 || type != GGML_TYPE_Q4_K) {
        return xdna_wfmt_repack_row(type, src, k, dst);
    }
    // Q4_K into the 8-bit affine form, so one format covers the whole decode.
    const block_q4_K * b   = (const block_q4_K *) src;
    uint8_t *          out = (uint8_t *) dst;
    for (int64_t i = 0; i < k / QK_K; i++) {
        repack_q4k_block_g16(b + i, out + (size_t) i * XDNA_Q8G16_SB_BYTES);
    }
    return true;
}

bool xdna_wfmt_repack_row(enum ggml_type type, const void * src, int64_t k, void * dst) {
    if (!src || !dst || k <= 0 || k % QK_K) {
        return false;
    }
    const int64_t nb  = k / QK_K;
    uint8_t *     out = (uint8_t *) dst;

    switch (type) {
        case GGML_TYPE_Q4_K:
            {
                const block_q4_K * b = (const block_q4_K *) src;
                for (int64_t i = 0; i < nb; i++) {
                    repack_q4k_block(b + i, out + (size_t) i * XDNA_Q4G32_SB_BYTES);
                }
                return true;
            }
        case GGML_TYPE_Q6_K:
            {
                const block_q6_K * b = (const block_q6_K *) src;
                for (int64_t i = 0; i < nb; i++) {
                    repack_q6k_block(b + i, out + (size_t) i * XDNA_Q8G16_SB_BYTES);
                }
                return true;
            }
        case GGML_TYPE_Q5_K:
            {
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
