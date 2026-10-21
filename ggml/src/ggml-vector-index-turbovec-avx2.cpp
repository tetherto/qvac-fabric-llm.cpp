// ggml-vector-index-turbovec-avx2.cpp - AVX2 TurboVec rotation, encode and
// layout kernels, ported from upstream turbovec 1.1.2 (rotation.rs,
// encode.rs, pack.rs). Built with AVX2 + FMA enabled and called only after a
// runtime check; every kernel performs the scalar reference's IEEE
// operations in the same order (no fused multiply-add), so its output is
// bit-identical.

#include "ggml-vector-index-x86.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

namespace ggml_vec_index_detail {

namespace {

// Upstream rotation.rs `permute_gather_avx2`; mode 0 moves, mode 1 applies
// the sign, mode 2 applies inv and then the sign.
template <int MODE>
void permute_gather_avx2_impl(
        const float * src, const uint32_t * perm, const float * signs, float inv, float * dst, size_t n) {
    const __m256 invv = _mm256_set1_ps(inv);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256i idx = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(perm + i));
        __m256 v = _mm256_i32gather_ps(src, idx, 4);
        if (MODE == 2) {
            v = _mm256_mul_ps(v, invv);
        }
        if (MODE != 0) {
            v = _mm256_mul_ps(v, _mm256_loadu_ps(signs + i));
        }
        _mm256_storeu_ps(dst + i, v);
    }
    // Dead for every rotation (dim is a multiple of 8); keeps any length correct.
    for (; i < n; ++i) {
        const float v = src[perm[i]];
        dst[i] = MODE == 0 ? v : (MODE == 1 ? v * signs[i] : (v * inv) * signs[i]);
    }
}

} // namespace

void turbovec_permute_gather_avx2(
        int mode, const float * src, const uint32_t * perm, const float * signs, float inv, float * dst, size_t n) {
    if (mode == 0) {
        permute_gather_avx2_impl<0>(src, perm, signs, inv, dst, n);
    } else if (mode == 1) {
        permute_gather_avx2_impl<1>(src, perm, signs, inv, dst, n);
    } else {
        permute_gather_avx2_impl<2>(src, perm, signs, inv, dst, n);
    }
}

// Upstream rotation.rs `wht_block_avx2`: stages len = 1, 2, 4 resolved in one
// register per 8-float group, then radix-8 passes (three stages each) with
// radix-4 / radix-2 tails. Same operand pairs, same expression trees, same
// f32 roundings as the scalar ladder.
void turbovec_wht_block_avx2(float * p, int block, float inv_sqrt_block) {
    for (int j = 0; j < block; j += 8) {
        const __m256 v = _mm256_loadu_ps(p + j);
        // len = 1: pairs (0,1) (2,3) (4,5) (6,7).
        __m256 a = _mm256_shuffle_ps(v, v, 0xA0);  // 0b10_10_00_00
        __m256 b = _mm256_shuffle_ps(v, v, 0xF5);  // 0b11_11_01_01
        const __m256 r1 = _mm256_blend_ps(_mm256_add_ps(a, b), _mm256_sub_ps(a, b), 0xAA);
        // len = 2: pairs (0,2) (1,3) (4,6) (5,7).
        a = _mm256_shuffle_ps(r1, r1, 0x44);  // 0b01_00_01_00
        b = _mm256_shuffle_ps(r1, r1, 0xEE);  // 0b11_10_11_10
        const __m256 r2 = _mm256_blend_ps(_mm256_add_ps(a, b), _mm256_sub_ps(a, b), 0xCC);
        // len = 4: pairs (k, k+4), across the two 128-bit halves.
        a = _mm256_permute2f128_ps(r2, r2, 0x00);
        b = _mm256_permute2f128_ps(r2, r2, 0x11);
        const __m256 r4 = _mm256_blend_ps(_mm256_add_ps(a, b), _mm256_sub_ps(a, b), 0xF0);
        _mm256_storeu_ps(p + j, r4);
    }

    int len = 8;
    for (; 4 * len < block; len <<= 3) {
        for (int i = 0; i < block; i += 8 * len) {
            for (int j = i; j < i + len; j += 8) {
                const __m256 a = _mm256_loadu_ps(p + j);
                const __m256 b = _mm256_loadu_ps(p + j + len);
                const __m256 c = _mm256_loadu_ps(p + j + 2 * len);
                const __m256 d = _mm256_loadu_ps(p + j + 3 * len);
                const __m256 e = _mm256_loadu_ps(p + j + 4 * len);
                const __m256 f = _mm256_loadu_ps(p + j + 5 * len);
                const __m256 g = _mm256_loadu_ps(p + j + 6 * len);
                const __m256 h = _mm256_loadu_ps(p + j + 7 * len);
                const __m256 apb = _mm256_add_ps(a, b);
                const __m256 amb = _mm256_sub_ps(a, b);
                const __m256 cpd = _mm256_add_ps(c, d);
                const __m256 cmd = _mm256_sub_ps(c, d);
                const __m256 epf = _mm256_add_ps(e, f);
                const __m256 emf = _mm256_sub_ps(e, f);
                const __m256 gph = _mm256_add_ps(g, h);
                const __m256 gmh = _mm256_sub_ps(g, h);
                const __m256 s0 = _mm256_add_ps(apb, cpd);
                const __m256 s1 = _mm256_add_ps(amb, cmd);
                const __m256 s2 = _mm256_sub_ps(apb, cpd);
                const __m256 s3 = _mm256_sub_ps(amb, cmd);
                const __m256 s4 = _mm256_add_ps(epf, gph);
                const __m256 s5 = _mm256_add_ps(emf, gmh);
                const __m256 s6 = _mm256_sub_ps(epf, gph);
                const __m256 s7 = _mm256_sub_ps(emf, gmh);
                _mm256_storeu_ps(p + j, _mm256_add_ps(s0, s4));
                _mm256_storeu_ps(p + j + len, _mm256_add_ps(s1, s5));
                _mm256_storeu_ps(p + j + 2 * len, _mm256_add_ps(s2, s6));
                _mm256_storeu_ps(p + j + 3 * len, _mm256_add_ps(s3, s7));
                _mm256_storeu_ps(p + j + 4 * len, _mm256_sub_ps(s0, s4));
                _mm256_storeu_ps(p + j + 5 * len, _mm256_sub_ps(s1, s5));
                _mm256_storeu_ps(p + j + 6 * len, _mm256_sub_ps(s2, s6));
                _mm256_storeu_ps(p + j + 7 * len, _mm256_sub_ps(s3, s7));
            }
        }
    }
    if (2 * len < block) {
        for (int i = 0; i < block; i += 4 * len) {
            for (int j = i; j < i + len; j += 8) {
                const __m256 a = _mm256_loadu_ps(p + j);
                const __m256 b = _mm256_loadu_ps(p + j + len);
                const __m256 c = _mm256_loadu_ps(p + j + 2 * len);
                const __m256 d = _mm256_loadu_ps(p + j + 3 * len);
                const __m256 apb = _mm256_add_ps(a, b);
                const __m256 amb = _mm256_sub_ps(a, b);
                const __m256 cpd = _mm256_add_ps(c, d);
                const __m256 cmd = _mm256_sub_ps(c, d);
                _mm256_storeu_ps(p + j, _mm256_add_ps(apb, cpd));
                _mm256_storeu_ps(p + j + len, _mm256_add_ps(amb, cmd));
                _mm256_storeu_ps(p + j + 2 * len, _mm256_sub_ps(apb, cpd));
                _mm256_storeu_ps(p + j + 3 * len, _mm256_sub_ps(amb, cmd));
            }
        }
        len <<= 2;
    }
    if (len < block) {
        for (int i = 0; i < block; i += 2 * len) {
            for (int j = i; j < i + len; j += 8) {
                const __m256 a = _mm256_loadu_ps(p + j);
                const __m256 b = _mm256_loadu_ps(p + j + len);
                _mm256_storeu_ps(p + j, _mm256_add_ps(a, b));
                _mm256_storeu_ps(p + j + len, _mm256_sub_ps(a, b));
            }
        }
    }

    const __m256 sv = _mm256_set1_ps(inv_sqrt_block);
    for (int j = 0; j < block; j += 8) {
        _mm256_storeu_ps(p + j, _mm256_mul_ps(_mm256_loadu_ps(p + j), sv));
    }
}

// Upstream encode.rs `norm_sq_avx`: the eight chains are the eight lanes of
// one register, multiplied and added as separate instructions.
float turbovec_norm_sq_avx(const float * row, int dim) {
    constexpr int chains_n = 8;
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + chains_n <= dim; i += chains_n) {
        const __m256 v = _mm256_loadu_ps(row + i);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(v, v));
    }
    float chains[chains_n];
    _mm256_storeu_ps(chains, acc);
    for (; i < dim; ++i) {
        chains[i % chains_n] += row[i] * row[i];
    }
    return ((chains[0] + chains[1]) + (chains[2] + chains[3])) +
        ((chains[4] + chains[5]) + (chains[6] + chains[7]));
}

// Upstream encode.rs `fused_quantize_scale_pack_avx2`: calibration and the
// boundary counts eight lanes at a time, bit planes gathered with a reversed
// lane order and movemask, and the reconstruction `centroid * inv_scale -
// shift` evaluated in f64 lanes, accumulating term t into chain t % 4 in the
// scalar order. Returns the inner product before the degeneracy check.
double turbovec_quantize_row_avx2(
        const float * rotated,
        const float * shift,
        const float * scale_tq,
        const float * inv_scale_tq,
        const float * boundaries,
        const float * centroids,
        int bits,
        int dim,
        uint8_t * packed_row) {
    const size_t bytes_per_plane = static_cast<size_t>(dim) / 8;
    const int n_codes = 1 << bits;
    __m256d acc4 = _mm256_setzero_pd();
    // A lane permute needs an 8-entry source: pad below 3 bits, split at 4.
    float cpad[8];
    float cpad_hi[8] = {};
    for (int i = 0; i < 8; ++i) {
        cpad[i] = centroids[i < n_codes - 1 ? i : n_codes - 1];
    }
    if (bits == 4) {
        for (int i = 0; i < 8; ++i) {
            cpad_hi[i] = centroids[8 + i];
        }
    }
    const __m256 cvec = _mm256_loadu_ps(cpad);
    const __m256 cvec_hi = _mm256_loadu_ps(cpad_hi);
    const __m256i reverse = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);

    for (int chunk = 0; chunk < dim / 8; ++chunk) {
        const int offset = chunk * 8;
        const __m256 vals = _mm256_mul_ps(
            _mm256_add_ps(_mm256_loadu_ps(rotated + offset), _mm256_loadu_ps(shift + offset)),
            _mm256_loadu_ps(scale_tq + offset));

        // Boundary count per lane (acc -= cmp adds 1 where val > b).
        __m256i acc;
        if (bits == 4) {
            const __m256 m = _mm256_cmp_ps(vals, _mm256_set1_ps(boundaries[7]), _CMP_GT_OQ);
            acc = _mm256_slli_epi32(_mm256_srli_epi32(_mm256_castps_si256(m), 31), 3);
            for (int k = 0; k < 7; ++k) {
                const __m256 bv = _mm256_blendv_ps(
                    _mm256_set1_ps(boundaries[k]), _mm256_set1_ps(boundaries[8 + k]), m);
                const __m256 gt = _mm256_cmp_ps(vals, bv, _CMP_GT_OQ);
                acc = _mm256_sub_epi32(acc, _mm256_castps_si256(gt));
            }
        } else if (bits == 2) {
            const __m256 m = _mm256_cmp_ps(vals, _mm256_set1_ps(boundaries[1]), _CMP_GT_OQ);
            acc = _mm256_slli_epi32(_mm256_srli_epi32(_mm256_castps_si256(m), 31), 1);
            const __m256 bv = _mm256_blendv_ps(
                _mm256_set1_ps(boundaries[0]), _mm256_set1_ps(boundaries[2]), m);
            const __m256 gt = _mm256_cmp_ps(vals, bv, _CMP_GT_OQ);
            acc = _mm256_sub_epi32(acc, _mm256_castps_si256(gt));
        } else {
            acc = _mm256_setzero_si256();
            for (int b = 0; b < n_codes - 1; ++b) {
                const __m256 gt = _mm256_cmp_ps(vals, _mm256_set1_ps(boundaries[b]), _CMP_GT_OQ);
                acc = _mm256_sub_epi32(acc, _mm256_castps_si256(gt));
            }
        }

        // Bit plane p of the chunk: bit (7 - k) is bit p of code k. movemask
        // puts lane k in bit k, so the lanes are reversed once up front.
        const __m256i rev = _mm256_permutevar8x32_epi32(acc, reverse);
        for (int plane = 0; plane < bits; ++plane) {
            const __m256i bit = _mm256_sll_epi32(rev, _mm_cvtsi32_si128(31 - plane));
            packed_row[static_cast<size_t>(plane) * bytes_per_plane + static_cast<size_t>(chunk)] =
                static_cast<uint8_t>(_mm256_movemask_ps(_mm256_castsi256_ps(bit)));
        }

        __m256 sel;
        if (bits == 4) {
            const __m256i low3 = _mm256_and_si256(acc, _mm256_set1_epi32(7));
            const __m256 lo = _mm256_permutevar8x32_ps(cvec, low3);
            const __m256 hi = _mm256_permutevar8x32_ps(cvec_hi, low3);
            const __m256i use_hi = _mm256_cmpgt_epi32(acc, _mm256_set1_epi32(7));
            sel = _mm256_blendv_ps(lo, hi, _mm256_castsi256_ps(use_hi));
        } else {
            sel = _mm256_permutevar8x32_ps(cvec, acc);
        }
        const __m256d x_lo = _mm256_sub_pd(
            _mm256_mul_pd(
                _mm256_cvtps_pd(_mm256_castps256_ps128(sel)),
                _mm256_cvtps_pd(_mm_loadu_ps(inv_scale_tq + offset))),
            _mm256_cvtps_pd(_mm_loadu_ps(shift + offset)));
        const __m256d x_hi = _mm256_sub_pd(
            _mm256_mul_pd(
                _mm256_cvtps_pd(_mm256_extractf128_ps(sel, 1)),
                _mm256_cvtps_pd(_mm_loadu_ps(inv_scale_tq + offset + 4))),
            _mm256_cvtps_pd(_mm_loadu_ps(shift + offset + 4)));
        const __m256d rot_lo = _mm256_cvtps_pd(_mm_loadu_ps(rotated + offset));
        const __m256d rot_hi = _mm256_cvtps_pd(_mm_loadu_ps(rotated + offset + 4));
        acc4 = _mm256_add_pd(acc4, _mm256_mul_pd(rot_lo, x_lo));
        acc4 = _mm256_add_pd(acc4, _mm256_mul_pd(rot_hi, x_hi));
    }

    double chains[4];
    _mm256_storeu_pd(chains, acc4);
    return (chains[0] + chains[1]) + (chains[2] + chains[3]);
}

// Upstream encode.rs `first_invalid_in_chunk_avx2`: an all-clean fast path
// testing |x| < max_abs eight lanes at a time (NaN, Inf and over-magnitude
// values all fail), with the exact scalar predicate pinpointing the first
// invalid element of a flagged group. Returns n when every element is valid.
void permute_dot_finish_avx2(
        const int32_t * sums, float scale, float bias, const float * vector_scales, size_t valid, float * out_scores) {
    const __m256 scale_v = _mm256_set1_ps(scale);
    const __m256 bias_v = _mm256_set1_ps(bias);
    uint32_t neg_inf_bits = 0xff800000u;
    float neg_inf = 0.0f;
    std::memcpy(&neg_inf, &neg_inf_bits, sizeof(neg_inf));
    // Eight lanes at a time over all 32 (sums always holds 32); scales are
    // read only for valid lanes, through a padded copy in a partial group.
    for (size_t lane = 0; lane < 32; lane += 8) {
        if (lane >= valid) {
            for (size_t i = lane; i < lane + 8; ++i) {
                out_scores[i] = neg_inf;
            }
            continue;
        }
        const size_t live = valid - lane < 8 ? valid - lane : 8;
        float scales[8] = {};
        std::memcpy(scales, vector_scales + lane, live * sizeof(float));
        const __m256 sum = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(sums + lane)));
        const __m256 fused = _mm256_fmadd_ps(sum, scale_v, bias_v);
        float result[8];
        _mm256_storeu_ps(result, _mm256_mul_ps(fused, _mm256_loadu_ps(scales)));
        for (size_t i = 0; i < 8; ++i) {
            out_scores[lane + i] = i < live ? result[i] : neg_inf;
        }
    }
}

// The scalar predicate !isfinite(x) || fabs(x) >= max_abs on the bits, so
// this object instantiates no <cmath> inline function: an AVX2-built copy
// of one could be the copy baseline code links against.
static bool invalid_abs(float x, float max_abs) {
    uint32_t bits = 0;
    std::memcpy(&bits, &x, sizeof(bits));
    bits &= 0x7fffffffu;
    if (bits >= 0x7f800000u) {
        return true;
    }
    float magnitude = 0.0f;
    std::memcpy(&magnitude, &bits, sizeof(magnitude));
    return magnitude >= max_abs;
}

size_t first_invalid_abs_avx2(const float * values, size_t n, float max_abs) {
    const __m256 bound = _mm256_set1_ps(max_abs);
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 x = _mm256_loadu_ps(values + i);
        const __m256 ok = _mm256_cmp_ps(_mm256_and_ps(x, abs_mask), bound, _CMP_LT_OQ);
        if (_mm256_movemask_ps(ok) != 0xff) {
            for (size_t j = i; j < i + 8; ++j) {
                if (invalid_abs(values[j], max_abs)) {
                    return j;
                }
            }
        }
    }
    for (; i < n; ++i) {
        if (invalid_abs(values[i], max_abs)) {
            return i;
        }
    }
    return n;
}

// Upstream pack.rs `interleave_chunk_avx2`: the SSSE3 perm0 nibble
// interleave on two 32-byte groups per iteration. `len` is a multiple of 32;
// both groups are loaded before either is stored, so it runs in place.
void turbovec_interleave_avx2(uint8_t * buf, size_t len) {
    static const uint8_t kPerm0[16] = { 0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15 };
    const __m256i permv = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i *>(kPerm0)));
    const __m256i lo_mask = _mm256_set1_epi8(0x0f);
    const __m256i hi_mask = _mm256_set1_epi8(static_cast<char>(0xf0));
    constexpr size_t block = 32;
    const size_t n = len / block;
    const size_t pairs = n / 2;
    for (size_t i = 0; i < pairs; ++i) {
        uint8_t * p = buf + i * 2 * block;
        const __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p));
        const __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p + block));
        // [b0.lo | b1.lo] and [b0.hi | b1.hi].
        const __m256i a = _mm256_shuffle_epi8(_mm256_permute2x128_si256(v0, v1, 0x20), permv);
        const __m256i b = _mm256_shuffle_epi8(_mm256_permute2x128_si256(v0, v1, 0x31), permv);
        const __m256i a_hi = _mm256_and_si256(_mm256_srli_epi16(a, 4), lo_mask);
        const __m256i out_hi = _mm256_or_si256(a_hi, _mm256_and_si256(b, hi_mask));
        const __m256i b_lo4 = _mm256_and_si256(b, lo_mask);
        const __m256i out_lo = _mm256_or_si256(_mm256_and_si256(a, lo_mask), _mm256_slli_epi16(b_lo4, 4));
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(p), _mm256_permute2x128_si256(out_hi, out_lo, 0x20));
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(p + block), _mm256_permute2x128_si256(out_hi, out_lo, 0x31));
    }
    if (n % 2 == 1) {
        // AVX2 implies SSSE3; the final odd group takes the 16-byte kernel.
        uint8_t * p = buf + pairs * 2 * block;
        const __m128i perm128 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(kPerm0));
        const __m128i lo128 = _mm_set1_epi8(0x0f);
        const __m128i hi128 = _mm_set1_epi8(static_cast<char>(0xf0));
        const __m128i a = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p)), perm128);
        const __m128i b = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(p + 16)), perm128);
        const __m128i out_hi = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(a, 4), lo128), _mm_and_si128(b, hi128));
        const __m128i out_lo = _mm_or_si128(_mm_and_si128(a, lo128), _mm_slli_epi16(_mm_and_si128(b, lo128), 4));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(p), out_hi);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(p + 16), out_lo);
    }
}

} // namespace ggml_vec_index_detail
