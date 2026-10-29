// ggml-vector-index-turbovec-vnni.cpp - AVX-512 VBMI/VNNI TurboVec search
// kernels over upstream's vector-major layout, ported from upstream turbovec
// 1.1.2 search.rs. Built with AVX-512 F, BW, VBMI and VNNI enabled and
// called only after a runtime check for all four (turbovec_cpu_dot_product).
// Every kernel returns exact integer sums, equal to its scalar model's in
// ggml-vector-index-turbovec.cpp whatever the order of accumulation; the
// float epilogue is shared with that model.

#include "ggml-vector-index-x86.h"
#include "ggml-vector-index-turbovec-pd-x86.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <limits>

namespace ggml_vec_index_detail {

namespace {

constexpr size_t kBlock = 32;

// Upstream `search_multi_query_vnni::<PF, NQ>`: per four byte groups,
// `vpermb` picks each byte's entry from the four groups' 16-entry tables
// (the index is (j << 4) | code for byte j of a vector's dword) and
// `vpdpbusd` adds the four entries into that vector's u32 lane.
template <int NQ, bool PF>
inline void vnni_lut_acc_nq(
        const uint8_t * const * split_luts,
        const uint8_t * block,
        size_t n_byte_groups,
        size_t bytes_after,
        __m512i acc[NQ][2]) {
    const __m512i m0f = _mm512_set1_epi8(0x0f);
    const __m512i kpos = _mm512_set1_epi32(0x30201000);
    const __m512i ones = _mm512_set1_epi8(1);
    const size_t quads = n_byte_groups / 4;
    for (int q = 0; q < NQ; ++q) {
        acc[q][0] = _mm512_setzero_si512();
        acc[q][1] = _mm512_setzero_si512();
    }
    for (size_t q4 = 0; q4 < quads; ++q4) {
        for (size_t h = 0; h < 2; ++h) {
            if (PF) {
                const size_t pf = (q4 + 8) * 128 + h * 64;
                if (pf + 64 <= bytes_after) {
                    _mm_prefetch(reinterpret_cast<const char *>(block + pf), _MM_HINT_T0);
                }
            }
            const __m512i c = _mm512_loadu_si512(reinterpret_cast<const void *>(block + q4 * 128 + h * 64));
            const __m512i ilo = _mm512_or_si512(_mm512_and_si512(c, m0f), kpos);
            const __m512i ihi = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(c, 4), m0f), kpos);
            for (int q = 0; q < NQ; ++q) {
                const uint8_t * tp = split_luts[q] + q4 * 128;
                const __m512i tlo = _mm512_loadu_si512(reinterpret_cast<const void *>(tp));
                const __m512i thi = _mm512_loadu_si512(reinterpret_cast<const void *>(tp + 64));
                acc[q][h] = _mm512_dpbusd_epi32(acc[q][h], _mm512_permutexvar_epi8(ilo, tlo), ones);
                acc[q][h] = _mm512_dpbusd_epi32(acc[q][h], _mm512_permutexvar_epi8(ihi, thi), ones);
            }
        }
    }
}

template <int NQ, bool PF>
void vnni_lut_sums_nq(
        const uint8_t * const * split_luts,
        const uint8_t * block,
        size_t n_byte_groups,
        size_t bytes_after,
        int32_t * sums) {
    __m512i acc[NQ][2];
    vnni_lut_acc_nq<NQ, PF>(split_luts, block, n_byte_groups, bytes_after, acc);
    for (int q = 0; q < NQ; ++q) {
        _mm512_storeu_si512(reinterpret_cast<void *>(sums + static_cast<size_t>(q) * kBlock), acc[q][0]);
        _mm512_storeu_si512(reinterpret_cast<void *>(sums + static_cast<size_t>(q) * kBlock + 16), acc[q][1]);
    }
}

// Upstream `search_single_query_vnni_blk2`: one query over two blocks in
// flight, sharing each quad's table loads; prefetches the first block's
// stream eight quads ahead. The sums stay in a0 and a1 (16 lanes each).
inline void vnni_lut_acc_blk2(
        const uint8_t * split_lut,
        const uint8_t * block0,
        size_t n_byte_groups,
        size_t bytes_after,
        __m512i a0[2],
        __m512i a1[2]) {
    const __m512i m0f = _mm512_set1_epi8(0x0f);
    const __m512i kpos = _mm512_set1_epi32(0x30201000);
    const __m512i ones = _mm512_set1_epi8(1);
    const size_t quads = n_byte_groups / 4;
    const uint8_t * block1 = block0 + n_byte_groups * kBlock;
    a0[0] = _mm512_setzero_si512();
    a0[1] = _mm512_setzero_si512();
    a1[0] = _mm512_setzero_si512();
    a1[1] = _mm512_setzero_si512();
    for (size_t q4 = 0; q4 < quads; ++q4) {
        for (size_t h = 0; h < 2; ++h) {
            const size_t pf = (q4 + 8) * 128 + h * 64;
            if (pf + 64 <= bytes_after) {
                _mm_prefetch(reinterpret_cast<const char *>(block0 + pf), _MM_HINT_T0);
            }
            const uint8_t * tp = split_lut + q4 * 128;
            const __m512i tlo = _mm512_loadu_si512(reinterpret_cast<const void *>(tp));
            const __m512i thi = _mm512_loadu_si512(reinterpret_cast<const void *>(tp + 64));
            const __m512i c0 = _mm512_loadu_si512(reinterpret_cast<const void *>(block0 + q4 * 128 + h * 64));
            const __m512i c1 = _mm512_loadu_si512(reinterpret_cast<const void *>(block1 + q4 * 128 + h * 64));
            const __m512i i0lo = _mm512_or_si512(_mm512_and_si512(c0, m0f), kpos);
            const __m512i i0hi = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(c0, 4), m0f), kpos);
            const __m512i i1lo = _mm512_or_si512(_mm512_and_si512(c1, m0f), kpos);
            const __m512i i1hi = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(c1, 4), m0f), kpos);
            a0[h] = _mm512_dpbusd_epi32(a0[h], _mm512_permutexvar_epi8(i0lo, tlo), ones);
            a0[h] = _mm512_dpbusd_epi32(a0[h], _mm512_permutexvar_epi8(i0hi, thi), ones);
            a1[h] = _mm512_dpbusd_epi32(a1[h], _mm512_permutexvar_epi8(i1lo, tlo), ones);
            a1[h] = _mm512_dpbusd_epi32(a1[h], _mm512_permutexvar_epi8(i1hi, thi), ones);
        }
    }
}

void vnni_lut_sums_blk2(
        const uint8_t * split_lut,
        const uint8_t * block0,
        size_t n_byte_groups,
        size_t bytes_after,
        int32_t * sums) {
    __m512i a0[2];
    __m512i a1[2];
    vnni_lut_acc_blk2(split_lut, block0, n_byte_groups, bytes_after, a0, a1);
    _mm512_storeu_si512(reinterpret_cast<void *>(sums), a0[0]);
    _mm512_storeu_si512(reinterpret_cast<void *>(sums + 16), a0[1]);
    _mm512_storeu_si512(reinterpret_cast<void *>(sums + kBlock), a1[0]);
    _mm512_storeu_si512(reinterpret_cast<void *>(sums + kBlock + 16), a1[1]);
}

// One block of the fused lone-query scan: the `vpermb` epilogue of
// vnni_lut_finish (ggml-vector-index-turbovec.cpp) in registers, sum *
// scale + bias as two roundings, times the vector's scale, -inf past the
// valid lanes; then the per-block tests of the staged Collector's SSE2
// add_block, which must keep the same rows: lanes above the seed counted
// (ordered, so a NaN lane does not count), a block without a NaN lane
// skipped when its best lane cannot enter, and under a threshold only the
// lanes above it visited, NaN lanes among them (cmp NLE, unordered). The
// lanes that remain go to the sink compressed. The threshold is a double
// the collector made from a float, so its float form compares the same;
// so without a floor the block max test says exactly what the lane test
// says (no lane above the threshold) and is left out.
inline void vnni_fused_block(
        const __m512i a[2],
        __m512 vs,
        __m512 vb,
        const float * vector_scales,
        size_t block,
        size_t n_vectors,
        TurboVecVnniScanState & state,
        TurboVecVnniScanSink sink,
        void * ctx,
        size_t & above_seed) {
    const size_t base = block * kBlock;
    const size_t valid = n_vectors - base < kBlock ? n_vectors - base : kBlock;
    const uint32_t valid_mask = valid >= 32 ? 0xffffffffu : (1u << valid) - 1u;
    const __mmask16 v0 = static_cast<__mmask16>(valid_mask);
    const __mmask16 v1 = static_cast<__mmask16>(valid_mask >> 16);
    const __m512 neg_inf = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
    const __m512 f0 = _mm512_add_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(a[0]), vs), vb);
    const __m512 f1 = _mm512_add_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(a[1]), vs), vb);
    const __m512 s0 = _mm512_mask_mul_ps(neg_inf, v0, f0, _mm512_maskz_loadu_ps(v0, vector_scales + base));
    const __m512 s1 = _mm512_mask_mul_ps(neg_inf, v1, f1, _mm512_maskz_loadu_ps(v1, vector_scales + base + 16));
    if (state.seed_active) {
        const __m512 seed = _mm512_set1_ps(state.seed);
        const uint32_t above = static_cast<uint32_t>(_mm512_cmp_ps_mask(s0, seed, _CMP_GT_OQ)) |
            (static_cast<uint32_t>(_mm512_cmp_ps_mask(s1, seed, _CMP_GT_OQ)) << 16);
        above_seed += static_cast<size_t>(_mm_popcnt_u32(above));
    }
    double floor = -std::numeric_limits<double>::infinity();
    if (state.floor_score != nullptr) {
        floor = state.floor_score->load(std::memory_order_relaxed);
        const uint32_t unord = static_cast<uint32_t>(_mm512_cmp_ps_mask(s0, s0, _CMP_UNORD_Q)) |
            (static_cast<uint32_t>(_mm512_cmp_ps_mask(s1, s1, _CMP_UNORD_Q)) << 16);
        if (unord == 0) {
            const double block_max = static_cast<double>(_mm512_reduce_max_ps(_mm512_max_ps(s0, s1)));
            if ((state.cut && block_max <= state.threshold) || block_max < floor) {
                return;
            }
        }
    }
    uint32_t lanes = valid_mask;
    if (state.cut) {
        const __m512 thr = _mm512_set1_ps(static_cast<float>(state.threshold));
        lanes &= static_cast<uint32_t>(_mm512_cmp_ps_mask(s0, thr, _CMP_NLE_UQ)) |
            (static_cast<uint32_t>(_mm512_cmp_ps_mask(s1, thr, _CMP_NLE_UQ)) << 16);
        if (lanes == 0) {
            return;
        }
    }
    // Compressed in registers, then stored whole: the memory form of
    // vcompressps is microcoded on some cores.
    const __mmask16 l0 = static_cast<__mmask16>(lanes);
    const __mmask16 l1 = static_cast<__mmask16>(lanes >> 16);
    const size_t n0 = static_cast<size_t>(_mm_popcnt_u32(lanes & 0xffffu));
    const size_t n1 = static_cast<size_t>(_mm_popcnt_u32(lanes >> 16));
    const __m512i iota = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    alignas(64) float scores[kBlock + 16];
    alignas(64) int32_t lane_ids[kBlock + 16];
    _mm512_storeu_ps(scores, _mm512_maskz_compress_ps(l0, s0));
    _mm512_storeu_ps(scores + n0, _mm512_maskz_compress_ps(l1, s1));
    _mm512_storeu_si512(reinterpret_cast<void *>(lane_ids), _mm512_maskz_compress_epi32(l0, iota));
    _mm512_storeu_si512(reinterpret_cast<void *>(lane_ids + n0),
        _mm512_maskz_compress_epi32(l1, _mm512_add_epi32(iota, _mm512_set1_epi32(16))));
    sink(ctx, block, floor, scores, lane_ids, n0 + n1, state);
}

struct HiNibbleShift {
    __m512i operator()(__m512i c, __m512i m0f) const {
        return _mm512_and_si512(_mm512_srli_epi16(c, 4), m0f);
    }
};

} // namespace

void turbovec_vnni_lut_sums(
        const uint8_t * const * split_luts,
        int nq,
        const uint8_t * blocks,
        size_t n_blocks,
        size_t n_byte_groups,
        size_t bytes_after,
        int32_t * sums) {
    const size_t block_bytes = n_byte_groups * kBlock;
    if (nq == 1) {
        size_t b = 0;
        for (; b + 2 <= n_blocks; b += 2) {
            vnni_lut_sums_blk2(split_luts[0], blocks + b * block_bytes, n_byte_groups, bytes_after - b * block_bytes,
                sums + b * kBlock);
        }
        if (b < n_blocks) {
            // The odd tail block, single stream, without prefetch.
            vnni_lut_sums_nq<1, false>(split_luts, blocks + b * block_bytes, n_byte_groups, 0, sums + b * kBlock);
        }
        return;
    }
    // H53: one instantiation per batch width, so the accumulators stay in
    // registers; the batch re-reads the codes and does not prefetch.
    for (size_t b = 0; b < n_blocks; ++b) {
        const uint8_t * block = blocks + b * block_bytes;
        int32_t * out = sums + b * static_cast<size_t>(nq) * kBlock;
        // Callers batch at most 6; a wider batch runs in groups of 8, so
        // every query's sums are written.
        for (int q0 = 0; q0 < nq; q0 += 8) {
            const uint8_t * const * luts = split_luts + q0;
            int32_t * group_out = out + static_cast<size_t>(q0) * kBlock;
            // No std::min: this object pulls in no <algorithm> inline function.
            switch (nq - q0 < 8 ? nq - q0 : 8) {
                case 1: vnni_lut_sums_nq<1, false>(luts, block, n_byte_groups, 0, group_out); break;
                case 2: vnni_lut_sums_nq<2, false>(luts, block, n_byte_groups, 0, group_out); break;
                case 3: vnni_lut_sums_nq<3, false>(luts, block, n_byte_groups, 0, group_out); break;
                case 4: vnni_lut_sums_nq<4, false>(luts, block, n_byte_groups, 0, group_out); break;
                case 5: vnni_lut_sums_nq<5, false>(luts, block, n_byte_groups, 0, group_out); break;
                case 6: vnni_lut_sums_nq<6, false>(luts, block, n_byte_groups, 0, group_out); break;
                case 7: vnni_lut_sums_nq<7, false>(luts, block, n_byte_groups, 0, group_out); break;
                default: vnni_lut_sums_nq<8, false>(luts, block, n_byte_groups, 0, group_out); break;
            }
        }
    }
}

void turbovec_vnni_lut_scan_blk2(
        const uint8_t * split_lut,
        float scale,
        float bias,
        const float * vector_scales,
        const uint8_t * blocks,
        size_t block_begin,
        size_t block_end,
        size_t n_vectors,
        size_t n_byte_groups,
        TurboVecVnniScanState & state,
        TurboVecVnniScanSink sink,
        void * ctx) {
    const size_t block_bytes = n_byte_groups * kBlock;
    const size_t n_blocks = (n_vectors + kBlock - 1) / kBlock;
    const __m512 vs = _mm512_set1_ps(scale);
    const __m512 vb = _mm512_set1_ps(bias);
    size_t above_seed = state.above_seed;
    size_t b = block_begin;
    for (; b + 2 <= block_end; b += 2) {
        __m512i a0[2];
        __m512i a1[2];
        vnni_lut_acc_blk2(split_lut, blocks + b * block_bytes, n_byte_groups, (n_blocks - b) * block_bytes, a0, a1);
        vnni_fused_block(a0, vs, vb, vector_scales, b, n_vectors, state, sink, ctx, above_seed);
        vnni_fused_block(a1, vs, vb, vector_scales, b + 1, n_vectors, state, sink, ctx, above_seed);
    }
    if (b < block_end) {
        // The odd tail block, single stream, without prefetch.
        alignas(64) int32_t sums[kBlock];
        vnni_lut_sums_nq<1, false>(&split_lut, blocks + b * block_bytes, n_byte_groups, 0, sums);
        const __m512i a[2] = { _mm512_load_si512(reinterpret_cast<const void *>(sums)),
            _mm512_load_si512(reinterpret_cast<const void *>(sums + 16)) };
        vnni_fused_block(a, vs, vb, vector_scales, b, n_vectors, state, sink, ctx, above_seed);
    }
    state.above_seed = above_seed;
}

namespace {

// Upstream `search_multi_query_vnni::<false, NQ>` with its per-query
// epilogue: every block of the range through the NQ-query kernel, then each
// query's sums finished and tested in registers against its own collector
// state. The scale and bias are broadcast per block so no zmm stays live
// across the quad loop beyond the accumulators and the table operands.
template <int NQ>
void vnni_lut_scan_batch_nq(
        const uint8_t * const * split_luts,
        const float * scales,
        const float * biases,
        const float * vector_scales,
        const uint8_t * blocks,
        size_t block_begin,
        size_t block_end,
        size_t n_vectors,
        size_t n_byte_groups,
        TurboVecVnniScanState * states,
        TurboVecVnniScanSink sink,
        void * const * ctxs) {
    const size_t block_bytes = n_byte_groups * kBlock;
    size_t above_seed[NQ];
    for (int q = 0; q < NQ; ++q) {
        above_seed[q] = states[q].above_seed;
    }
    for (size_t b = block_begin; b < block_end; ++b) {
        __m512i acc[NQ][2];
        vnni_lut_acc_nq<NQ, false>(split_luts, blocks + b * block_bytes, n_byte_groups, 0, acc);
        for (int q = 0; q < NQ; ++q) {
            vnni_fused_block(acc[q], _mm512_set1_ps(scales[q]), _mm512_set1_ps(biases[q]), vector_scales, b, n_vectors,
                states[q], sink, ctxs[q], above_seed[q]);
        }
    }
    for (int q = 0; q < NQ; ++q) {
        states[q].above_seed = above_seed[q];
    }
}

} // namespace

void turbovec_vnni_lut_scan_batch(
        const uint8_t * const * split_luts,
        const float * scales,
        const float * biases,
        int nq,
        const float * vector_scales,
        const uint8_t * blocks,
        size_t block_begin,
        size_t block_end,
        size_t n_vectors,
        size_t n_byte_groups,
        TurboVecVnniScanState * states,
        TurboVecVnniScanSink sink,
        void * const * ctxs) {
    // Eight queries a pass, as upstream's `acc` array; the callers batch
    // six. A lone query takes the two-block kernel, as upstream's dispatch
    // sends nq == 1 to search_single_query_vnni_blk2.
    if (nq > 8) {
        for (int q0 = 0; q0 < nq; q0 += 8) {
            turbovec_vnni_lut_scan_batch(split_luts + q0, scales + q0, biases + q0, nq - q0 < 8 ? nq - q0 : 8,
                vector_scales, blocks, block_begin, block_end, n_vectors, n_byte_groups, states + q0, sink, ctxs + q0);
        }
        return;
    }
    switch (nq) {
        case 1:
            turbovec_vnni_lut_scan_blk2(split_luts[0], scales[0], biases[0], vector_scales, blocks, block_begin,
                block_end, n_vectors, n_byte_groups, states[0], sink, ctxs[0]);
            break;
        case 2:
            vnni_lut_scan_batch_nq<2>(split_luts, scales, biases, vector_scales, blocks, block_begin, block_end,
                n_vectors, n_byte_groups, states, sink, ctxs);
            break;
        case 3:
            vnni_lut_scan_batch_nq<3>(split_luts, scales, biases, vector_scales, blocks, block_begin, block_end,
                n_vectors, n_byte_groups, states, sink, ctxs);
            break;
        case 4:
            vnni_lut_scan_batch_nq<4>(split_luts, scales, biases, vector_scales, blocks, block_begin, block_end,
                n_vectors, n_byte_groups, states, sink, ctxs);
            break;
        case 5:
            vnni_lut_scan_batch_nq<5>(split_luts, scales, biases, vector_scales, blocks, block_begin, block_end,
                n_vectors, n_byte_groups, states, sink, ctxs);
            break;
        case 6:
            vnni_lut_scan_batch_nq<6>(split_luts, scales, biases, vector_scales, blocks, block_begin, block_end,
                n_vectors, n_byte_groups, states, sink, ctxs);
            break;
        case 7:
            vnni_lut_scan_batch_nq<7>(split_luts, scales, biases, vector_scales, blocks, block_begin, block_end,
                n_vectors, n_byte_groups, states, sink, ctxs);
            break;
        default:
            vnni_lut_scan_batch_nq<8>(split_luts, scales, biases, vector_scales, blocks, block_begin, block_end,
                n_vectors, n_byte_groups, states, sink, ctxs);
            break;
    }
}

void turbovec_permute_dot_sums_vnni(
        const int8_t * levels16,
        const int8_t * const * weights,
        const int32_t * zeros,
        int nq,
        const uint8_t * blocks,
        size_t n_blocks,
        size_t n_byte_groups,
        size_t bytes_after,
        int32_t * sums) {
    permute_dot_sums_dispatch(
        levels16, weights, zeros, nq, blocks, n_blocks, n_byte_groups, bytes_after, sums, HiNibbleShift{});
}

// Upstream `exact4_sum_avx512`: one vector's 4-bit permute-dot sum read
// straight from a planes cache, its sign plane from the vector-major sign
// region and its three low planes from the vector's packed row. `w` is
// laid out the way a plane byte is read (position 8c + b is dim
// 8c + 7 - b) and zero-padded to whole 64s.
int32_t turbovec_exact4_sum_avx512(
        const int8_t * w,
        const int8_t * levels16,
        int32_t zero,
        const uint8_t * sign_block,
        const uint8_t * low_row,
        size_t nsg,
        size_t lane) {
    auto quad = [&](size_t q) -> uint32_t {
        uint32_t v;
        std::memcpy(&v, sign_block + q * 128 + (lane / 16) * 64 + (lane % 16) * 4, 4);
        return v;
    };
    auto rd = [](const uint8_t * p, bool wide) -> uint64_t {
        if (wide) {
            uint64_t v;
            std::memcpy(&v, p, 8);
            return v;
        }
        uint32_t v;
        std::memcpy(&v, p, 4);
        return v;
    };
    const __m512i levels = _mm512_xor_si512(
        _mm512_broadcast_i32x4(_mm_loadu_si128(reinterpret_cast<const __m128i *>(levels16))),
        _mm512_set1_epi8(static_cast<char>(-128)));
    const __m512i b8 = _mm512_set1_epi8(8);
    const __m512i b4 = _mm512_set1_epi8(4);
    const __m512i b2 = _mm512_set1_epi8(2);
    const __m512i b1 = _mm512_set1_epi8(1);
    __m512i acc = _mm512_setzero_si512();
    for (size_t c = 0; c < nsg; c += 8) {
        // Eight bytes of each plane, or the last four.
        const bool wide = c + 8 <= nsg;
        const uint64_t ks = wide ? (static_cast<uint64_t>(quad(c / 4)) | (static_cast<uint64_t>(quad(c / 4 + 1)) << 32)) :
                                   static_cast<uint64_t>(quad(c / 4));
        const uint64_t k2 = rd(low_row + 2 * nsg + c, wide);
        const uint64_t k1 = rd(low_row + nsg + c, wide);
        const uint64_t k0 = rd(low_row + c, wide);
        const __m512i codes = _mm512_or_si512(
            _mm512_or_si512(_mm512_maskz_mov_epi8(ks, b8), _mm512_maskz_mov_epi8(k2, b4)),
            _mm512_or_si512(_mm512_maskz_mov_epi8(k1, b2), _mm512_maskz_mov_epi8(k0, b1)));
        const __m512i lv = _mm512_shuffle_epi8(levels, codes);
        const __m512i wv = _mm512_loadu_si512(reinterpret_cast<const void *>(w + c * 8));
        acc = _mm512_dpbusd_epi32(acc, lv, wv);
    }
    return static_cast<int32_t>(static_cast<uint32_t>(_mm512_reduce_add_epi32(acc)) + static_cast<uint32_t>(zero));
}

} // namespace ggml_vec_index_detail
