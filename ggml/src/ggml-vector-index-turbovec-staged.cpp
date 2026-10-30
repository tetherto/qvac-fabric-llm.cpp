// ggml-vector-index-turbovec-staged.cpp - staged search over the sign planes
// of a TurboVec 1.x index (upstream turbovec 1.1.2 "planes" search).
//
// A search scans the sign planes alone for a shortlist, ranks the shortlist
// with estimates that add the lower planes, and rescores the best few with
// the full scan's own arithmetic, so returned scores are exact while the
// candidate set is approximate. This follows upstream's single-threaded
// path (search.rs `search` with a `PlanesRef`, `rerank_legacy`, `Refine`).
// Upstream's seeded collectors only speed the shortlist up, except that on
// aarch64 the one query of a batch whose seed runs short is rescanned with
// single-query tables; the port seeds the same way and reports that query.
// Upstream's in-scan refinement on a thread pool, which can return other
// rows than its single-threaded path, is not reproduced: a lone query here
// splits the shortlist scan, the ranking and the rescore across threads and
// still returns the single-threaded results.

#include "ggml-vector-index-impl.h"

#include <bitset>

#if defined(GGML_VEC_INDEX_HAVE_AVX512BW_KERNEL) || defined(GGML_VEC_INDEX_HAVE_VPOPCNT_KERNEL) || \
    defined(GGML_VEC_INDEX_HAVE_VNNI_KERNEL)
#include "ggml-vector-index-x86.h"
#endif

#if GGML_VEC_INDEX_USE_NEON
#include <arm_neon.h>
#elif defined(__SSE2__) || defined(_M_X64)
#include <emmintrin.h>
#endif

namespace {

constexpr size_t kBlock = 32;
// How far the sign stage's query is pulled toward the coordinates' centre,
// as a fraction of the TQ+ shift (upstream SIGN_CENTRE).
constexpr float kSignCentre = 0.5f;
// Bits of a query coordinate's magnitude the refine pass keeps, and the
// low-row bytes one group of its masks covers.
constexpr int kLowBits = 6;
constexpr size_t kLowChunk = 64;
#if GGML_VEC_INDEX_USE_NEON
// Entry cap of the sign tables a single query scans with on aarch64.
constexpr float kSignLutCapSingle = 31.0f;
#endif

std::atomic<int> g_planes_override{ -1 };
std::atomic<size_t> g_planes_min_vectors{ kTurboVecPlanesMinVectors };
#if defined(GGML_VEC_INDEX_HAVE_AVX512BW_KERNEL) || defined(GGML_VEC_INDEX_HAVE_VPOPCNT_KERNEL)
// Off keeps the bit-mask build and count on their scalar loops (tests).
std::atomic<bool> g_low_kernels_enabled{ true };
#endif
#if defined(GGML_VEC_INDEX_HAVE_VNNI_KERNEL)
// Off scores a lone query's vector-major sign region block by block
// through the Collector (tests); the batch flag does the same for a
// batch's six queries.
std::atomic<bool> g_vnni_fused_scan_enabled{ true };
std::atomic<bool> g_vnni_fused_batch_enabled{ true };
#endif
#if defined(GGML_VEC_INDEX_HAVE_VNNI_KERNEL) && defined(GGML_VEC_INDEX_TEST_HOOKS)
// Fused scans and fused batch tasks run, for the same tests.
std::atomic<int64_t> g_vnni_fused_scan_count{ 0 };
std::atomic<int64_t> g_vnni_fused_batch_count{ 0 };
#endif
void count_vnni_fused_scan() {
#if defined(GGML_VEC_INDEX_HAVE_VNNI_KERNEL) && defined(GGML_VEC_INDEX_TEST_HOOKS)
    g_vnni_fused_scan_count.fetch_add(1, std::memory_order_relaxed);
#endif
}
void count_vnni_fused_batch() {
#if defined(GGML_VEC_INDEX_HAVE_VNNI_KERNEL) && defined(GGML_VEC_INDEX_TEST_HOOKS)
    g_vnni_fused_batch_count.fetch_add(1, std::memory_order_relaxed);
#endif
}

// Set bits of a word; MSVC has no __builtin_popcount.
inline int popcount64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(x);
#else
    return static_cast<int>(std::bitset<64>(x).count());
#endif
}

// Hint that the cache line at p is about to be read.
inline void prefetch_read(const void * p) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(p, 0, 3);
#else
    (void) p;
#endif
}

// Hints that every cache line holding [p, p + len) is about to be read,
// the line of the last byte included. The rows live in a std::vector,
// and glibc hands a large allocation out 16 bytes past a page boundary,
// so a 128-byte plane row spans three lines there, not two; a prefetch
// of offsets 0 and 64 left the third line to miss at full memory latency
// on every ranked row. Upstream prefetch_low covers the last byte for the
// same reason.
inline void prefetch_span(const void * p, size_t len) {
    const uintptr_t first = reinterpret_cast<uintptr_t>(p) & ~static_cast<uintptr_t>(63);
    const uintptr_t last = reinterpret_cast<uintptr_t>(p) + len - 1;
    for (uintptr_t a = first; a <= last; a += 64) {
        prefetch_read(reinterpret_cast<const void *>(a));
    }
}

// Trailing zero bits of a non-zero word (callers check; the loop below
// would never end on zero).
inline int ctz64(uint64_t x) {
    assert(x != 0);
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(x);
#else
    int n = 0;
    while ((x & 1u) == 0) {
        x >>= 1;
        ++n;
    }
    return n;
#endif
}

// Writes src[0..n) to dst[0..n) with the elements pred() accepts first, in
// their order, and the rest after them, in reverse order; returns how many
// were accepted. Every element is stored at both ends and the two cursors
// move by pred's 0 or 1, so the loop has no data-dependent branch: a
// quickselect over a shortlist's scores mispredicts about every other
// compare in std::nth_element's partition (upstream's select_nth_unstable
// partitions this way as well).
template <class T, class Pred>
size_t partition_branchless(const T * src, size_t n, T * dst, Pred pred) {
    size_t l = 0;
    size_t r = n;
    for (size_t i = 0; i < n; ++i) {
        const T x = src[i];
        const size_t accept = static_cast<size_t>(pred(x));
        dst[l] = x;
        dst[r - 1] = x;
        l += accept;
        r -= 1 - accept;
    }
    return l;
}

// The k-th best of a[0..n) (1 <= k <= n) under `better`, which returns 1
// when its first argument ranks first and 0 otherwise, as an integer so the
// partitions above stay branch-free; `better` must be a strict weak order
// and a class type (see ScoreIdBetterI), never a function pointer.
// Quickselect with two scratch buffers in place of std::nth_element, which
// left the lone and batch shortlist cuts, the ranking's cuts and the seed
// at several microseconds each on x86. `a` is read only, so a caller keeps
// its order and filters by the result (see cut_to_best).
template <class T, class Better>
T kth_best(const T * a, size_t n, size_t k, Better better, std::vector<T> & scratch) {
    assert(k >= 1 && k <= n);
    scratch.resize(2 * n);
    T * cur = scratch.data();
    T * other = scratch.data() + n;
    const T * src = a;
    size_t lo = 0;
    size_t hi = n;
    // Beyond the depth a balanced run needs the rest goes to introselect.
    int budget = 0;
    for (size_t m = n; m > 0; m >>= 1) {
        budget += 3;
    }
    while (hi - lo > 16 && budget-- > 0) {
        const T & p0 = src[lo];
        const T & p1 = src[lo + (hi - lo) / 2];
        const T & p2 = src[hi - 1];
        // Median of three.
        const T pivot = better(p0, p1) != 0 ? (better(p1, p2) != 0 ? p1 : (better(p0, p2) != 0 ? p2 : p0)) :
                                              (better(p0, p2) != 0 ? p0 : (better(p1, p2) != 0 ? p2 : p1));
        // Those strictly better than the pivot first; the pivot is not among
        // them, so a step into them shrinks the range.
        size_t l = lo + partition_branchless(src + lo, hi - lo, cur + lo, [&](const T & x) { return better(x, pivot); });
        src = cur;
        std::swap(cur, other);
        if (k <= l - lo) {
            hi = l;
            continue;
        }
        k -= l - lo;
        lo = l;
        // Then those the pivot does not beat (at least the pivot itself): if
        // the k-th lies among them it ranks with the pivot.
        l = lo + partition_branchless(src + lo, hi - lo, cur + lo, [&](const T & x) { return 1 - better(pivot, x); });
        src = cur;
        std::swap(cur, other);
        if (k <= l - lo) {
            return pivot;
        }
        k -= l - lo;
        lo = l;
    }
    if (src != cur) {
        // The remaining range is in a read-only or the other buffer.
        std::copy(src + lo, src + hi, cur + lo);
    }
    std::nth_element(cur + lo, cur + lo + static_cast<std::ptrdiff_t>(k - 1), cur + hi,
        [&](const T & x, const T & y) { return better(x, y) != 0; });
    return cur[lo + k - 1];
}

// Keeps, in place and in order, the elements of v for which keep() returns
// 1, without a data-dependent branch; returns their count.
template <class T, class Keep>
size_t keep_in_order(std::vector<T> & v, Keep keep) {
    size_t w = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        const T x = v[i];
        v[w] = x;
        w += static_cast<size_t>(keep(x));
    }
    return w;
}

// Cuts v (longer than `keep`) to its best `keep` elements under `better`
// (see kth_best), in the order they had, so a shortlist collected in slot
// order ranks its rows in slot order; returns the k-th best. The cut is
// its: whatever it does not beat stays, which is exactly `keep` elements
// under a strict total order, and ties for the last place are settled by
// std::nth_element as before.
template <class T, class Better>
T cut_to_best(std::vector<T> & v, size_t keep, Better better, std::vector<T> & scratch) {
    assert(v.size() > keep);
    const T kth = kth_best(v.data(), v.size(), keep, better, scratch);
    const size_t kept = keep_in_order(v, [&](const T & x) { return 1 - better(kth, x); });
    if (kept > keep) {
        std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(keep - 1),
            v.begin() + static_cast<std::ptrdiff_t>(kept), [&](const T & x, const T & y) { return better(x, y) != 0; });
    }
    v.resize(std::min(kept, keep));
    return kth;
}

// score_id_better as an integer without a branch, for the cuts above. A
// function object, not a function: handed over as a pointer, the compare
// is an indirect call per element unless the compiler clones the select
// for the constant pointer, and GCC does not (every compare in
// cut_to_best and keep_best was an indirect call), which left the cuts
// slower than std::nth_element on the x86 Ubuntu host while clang inlined
// them. A class type instantiates the select on the comparator itself.
struct ScoreIdBetterI {
    int operator()(const ScoreId & a, const ScoreId & b) const {
        return static_cast<int>(a.score > b.score) | (static_cast<int>(a.score == b.score) & static_cast<int>(a.id < b.id));
    }
};

// Shortlist, middle and rescore lengths for a top-k request (upstream
// planes_shortlist_len, planes_mid_len, planes_rescore_len).
size_t shortlist_len(size_t k, int bits) {
    if (bits == 4) {
        return std::max<size_t>(k >= 64 ? k * 16 : k * 20, 256);
    }
    return std::max<size_t>(k * 16, 384);
}

size_t mid_len(size_t k) {
    return std::max<size_t>(6 * k, 256);
}

size_t rescore_len(size_t k, int bits) {
    return bits == 4 ? std::max<size_t>(3 * k / 2, 64) : std::max<size_t>(2 * k, 64);
}

struct SignLut {
    std::vector<uint8_t> values;
    // The `vpermb` tables, for a vector-major sign region.
    std::vector<uint8_t> split;
    float scale = 1.0f;
    float bias = 0.0f;
    // Entries capped at kSignLutCapSingle, so eight lookups fit a byte and
    // the scan widens once per four groups (upstream score_sign_block_neon).
    bool deferred = false;
};

// Per-query nibble tables over a sign plane (upstream build_sign_lut): the
// high nibble of byte group g covers dims 8g..8g+3, the low nibble the next
// four, and entry p is sum_j (+-m) * q[d + j]. `single_query` caps the
// entries for aarch64's deferred-widening single-query scan.
SignLut build_sign_lut(const float * q, float m, int dim, bool single_query) {
    const size_t n_groups = static_cast<size_t>(dim) / 8;
    SignLut out;
    std::vector<float> values(n_groups * 32);
    std::vector<float> mins(n_groups * 2);
    float max_span = 0.0f;
    float bias = 0.0f;
    for (size_t g = 0; g < n_groups; ++g) {
        for (size_t half = 0; half < 2; ++half) {
            const size_t d = g * 8 + half * 4;
            const float p[4] = { q[d] * m, q[d + 1] * m, q[d + 2] * m, q[d + 3] * m };
            const float a[4] = { -p[0] - p[1], p[1] - p[0], p[0] - p[1], p[0] + p[1] };
            const float b[4] = { -p[2] - p[3], p[3] - p[2], p[2] - p[3], p[2] + p[3] };
            float * entries = values.data() + g * 32 + half * 16;
            for (size_t pat = 0; pat < 16; ++pat) {
                entries[pat] = a[pat >> 2] + b[pat & 3];
            }
            const float m01 = std::fabs(p[0]) + std::fabs(p[1]);
            const float m23 = std::fabs(p[2]) + std::fabs(p[3]);
            const float mn = -m01 - m23;
            const float mx = m01 + m23;
            mins[g * 2 + half] = mn;
            bias += mn;
            if (mx - mn > max_span) {
                max_span = mx - mn;
            }
        }
    }
    // A single query on aarch64 scans with a kernel that pre-adds eight
    // lookups in u8, so its entries are capped at 31.
#if GGML_VEC_INDEX_USE_NEON
    const float max_lut = single_query ? kSignLutCapSingle : 127.0f;
    out.deferred = single_query;
#else
    (void) single_query;
    const float max_lut = 127.0f;
#endif
    float scale = max_span > 0.0f ? max_span / max_lut : 1.0f;
    float inv_scale = 1.0f;
    if (scale >= FLT_MIN) {
        inv_scale = 1.0f / scale;
    } else {
        scale = 1.0f;
    }
    out.values.resize(n_groups * 32);
    // Sixteen entries share a minimum (upstream quantise_tables).
    for (size_t s = 0; s < n_groups * 2; ++s) {
        quantise_table16(values.data() + s * 16, mins[s], inv_scale, max_lut, out.values.data() + s * 16);
    }
    out.scale = scale;
    out.bias = bias;
    return out;
}

// The x86 sign scan's `vpermb` tables for a sign region in the vector-major
// layout (upstream builds them with every sign table there).
void split_sign_lut(const ggml_vec_index & idx, SignLut & lut) {
    if (turbovec_sign_layout(idx) == kTurboVecLayoutVm) {
        lut.split.resize(lut.values.size());
        turbovec_split_lut_for_vnni(lut.values.data(), lut.values.size() / 32, lut.split.data());
    }
}

#if GGML_VEC_INDEX_USE_NEON
// score_turbovec_lut_block for a block of the sign region under deferred
// tables (upstream score_lut_block_neon<DEFER = true>): four byte groups,
// eight lookups of at most 31, are summed in u8 and widened once instead
// of once per group. Each flush's u16 sums are the same integers as the
// per-group kernel's, so the scores match it bit for bit.
void score_sign_block_deferred(const uint8_t * lut, float lut_scale, float lut_bias, const uint8_t * blocked_codes,
        const float * vector_scales, size_t block, size_t n, size_t n_groups, float * out) {
    const size_t base = block * kBlock;
    const size_t valid = std::min(kBlock, n - base);
    const uint8_t * codes = blocked_codes + block * n_groups * kBlock;
    const uint8x16_t nib = vdupq_n_u8(0x0f);
    const float32x4_t scale_v = vdupq_n_f32(lut_scale);
    float32x4_t facc[8];
    for (float32x4_t & f : facc) {
        f = vdupq_n_f32(lut_bias);
    }
    const size_t flush = static_cast<size_t>(kTurboVecFlushEvery);
    for (size_t g_begin = 0; g_begin < n_groups; g_begin += flush) {
        const size_t g_end = std::min(g_begin + flush, n_groups);
        uint16x8_t acc[4] = { vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0) };
        size_t g = g_begin;
        for (; g + 3 < g_end; g += 4) {
            const uint8_t * lp = lut + g * 32;
            const uint8_t * cp = codes + g * kBlock;
            uint8x16_t t0 = vdupq_n_u8(0);
            uint8x16_t t1 = vdupq_n_u8(0);
            for (size_t j = 0; j < 4; ++j) {
                const uint8x16_t lut_hi = vld1q_u8(lp + j * 32);
                const uint8x16_t lut_lo = vld1q_u8(lp + j * 32 + 16);
                const uint8x16_t c0 = vld1q_u8(cp + j * kBlock);
                const uint8x16_t c1 = vld1q_u8(cp + j * kBlock + 16);
                t0 = vaddq_u8(t0, vaddq_u8(vqtbl1q_u8(lut_lo, vandq_u8(c0, nib)), vqtbl1q_u8(lut_hi, vshrq_n_u8(c0, 4))));
                t1 = vaddq_u8(t1, vaddq_u8(vqtbl1q_u8(lut_lo, vandq_u8(c1, nib)), vqtbl1q_u8(lut_hi, vshrq_n_u8(c1, 4))));
            }
            acc[0] = vaddw_u8(acc[0], vget_low_u8(t0));
            acc[1] = vaddw_high_u8(acc[1], t0);
            acc[2] = vaddw_u8(acc[2], vget_low_u8(t1));
            acc[3] = vaddw_high_u8(acc[3], t1);
        }
        for (; g < g_end; ++g) {
            const uint8_t * lp = lut + g * 32;
            const uint8_t * cp = codes + g * kBlock;
            const uint8x16_t lut_hi = vld1q_u8(lp);
            const uint8x16_t lut_lo = vld1q_u8(lp + 16);
            const uint8x16_t c0 = vld1q_u8(cp);
            const uint8x16_t c1 = vld1q_u8(cp + 16);
            const uint8x16_t s0 = vaddq_u8(vqtbl1q_u8(lut_lo, vandq_u8(c0, nib)), vqtbl1q_u8(lut_hi, vshrq_n_u8(c0, 4)));
            const uint8x16_t s1 = vaddq_u8(vqtbl1q_u8(lut_lo, vandq_u8(c1, nib)), vqtbl1q_u8(lut_hi, vshrq_n_u8(c1, 4)));
            acc[0] = vaddw_u8(acc[0], vget_low_u8(s0));
            acc[1] = vaddw_high_u8(acc[1], s0);
            acc[2] = vaddw_u8(acc[2], vget_low_u8(s1));
            acc[3] = vaddw_high_u8(acc[3], s1);
        }
        for (int i = 0; i < 4; ++i) {
            const float32x4_t lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(acc[i])));
            const float32x4_t hi = vcvtq_f32_u32(vmovl_high_u16(acc[i]));
            facc[i * 2] = vfmaq_f32(facc[i * 2], scale_v, lo);
            facc[i * 2 + 1] = vfmaq_f32(facc[i * 2 + 1], scale_v, hi);
        }
    }
    if (valid == kBlock) {
        for (int i = 0; i < 8; ++i) {
            vst1q_f32(out + i * 4, vmulq_f32(facc[i], vld1q_f32(vector_scales + base + static_cast<size_t>(i) * 4)));
        }
        return;
    }
    float decoded[kBlock];
    for (int i = 0; i < 8; ++i) {
        vst1q_f32(decoded + i * 4, facc[i]);
    }
    for (size_t lane = 0; lane < kBlock; ++lane) {
        out[lane] = lane < valid ? decoded[lane] * vector_scales[base + lane] : -std::numeric_limits<float>::infinity();
    }
}

// Sixteen 16-byte rows to sixteen columns (upstream transpose16_neon):
// out[j] holds byte j of each row, in row order.
void transpose16(const uint8x16_t r[16], uint8x16_t out[16]) {
    uint8x16_t a[8];
    uint8x16_t b[8];
    for (size_t i = 0; i < 8; ++i) {
        a[i] = vzip1q_u8(r[2 * i], r[2 * i + 1]);
        b[i] = vzip2q_u8(r[2 * i], r[2 * i + 1]);
    }
    uint8x16_t q[4][4];
    for (size_t i = 0; i < 4; ++i) {
        const uint16x8_t a0 = vreinterpretq_u16_u8(a[2 * i]);
        const uint16x8_t a1 = vreinterpretq_u16_u8(a[2 * i + 1]);
        const uint16x8_t b0 = vreinterpretq_u16_u8(b[2 * i]);
        const uint16x8_t b1 = vreinterpretq_u16_u8(b[2 * i + 1]);
        q[0][i] = vreinterpretq_u8_u16(vzip1q_u16(a0, a1));
        q[1][i] = vreinterpretq_u8_u16(vzip2q_u16(a0, a1));
        q[2][i] = vreinterpretq_u8_u16(vzip1q_u16(b0, b1));
        q[3][i] = vreinterpretq_u8_u16(vzip2q_u16(b0, b1));
    }
    for (size_t c = 0; c < 4; ++c) {
        // q[c][i]: columns 4c..4c+4 of rows 4i..4i+4.
        uint8x16_t o[2][2];
        for (size_t i = 0; i < 2; ++i) {
            const uint32x4_t x0 = vreinterpretq_u32_u8(q[c][2 * i]);
            const uint32x4_t x1 = vreinterpretq_u32_u8(q[c][2 * i + 1]);
            o[0][i] = vreinterpretq_u8_u32(vzip1q_u32(x0, x1));
            o[1][i] = vreinterpretq_u8_u32(vzip2q_u32(x0, x1));
        }
        for (size_t h = 0; h < 2; ++h) {
            // o[h][i]: columns 4c+2h, 4c+2h+1 of rows 8i..8i+8.
            const uint64x2_t y0 = vreinterpretq_u64_u8(o[h][0]);
            const uint64x2_t y1 = vreinterpretq_u64_u8(o[h][1]);
            out[4 * c + 2 * h] = vreinterpretq_u8_u64(vzip1q_u64(y0, y1));
            out[4 * c + 2 * h + 1] = vreinterpretq_u8_u64(vzip2q_u64(y0, y1));
        }
    }
}

// For up to kRankBatch rows of nsg bytes, the sum over byte positions of
// the two nibble lookups in the sign tables t: the sign scan's arithmetic,
// on rows gathered from anywhere (upstream table_sums_neon). Sixteen rows
// are transposed so one position's lookups serve them all.
void table_sums(const uint8_t * t, const uint8_t * const * rows, size_t n_rows, size_t nsg, uint32_t * out) {
    const uint8x16_t zero = vdupq_n_u8(0);
    const uint8x16_t nib = vdupq_n_u8(0x0f);
    uint32x4_t acc32[8];
    for (uint32x4_t & a : acc32) {
        a = vdupq_n_u32(0);
    }
    uint16x8_t acc16[4] = { vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0) };
    for (size_t c = 0; c < nsg;) {
        const size_t width = std::min<size_t>(nsg - c, 16);
        uint8x16_t r[2][16];
        for (size_t i = 0; i < 2 * 16; ++i) {
            if (i >= n_rows) {
                r[i / 16][i % 16] = zero;
            } else if (width == 16) {
                r[i / 16][i % 16] = vld1q_u8(rows[i] + c);
            } else {
                uint8_t tmp[16] = {};
                std::memcpy(tmp, rows[i] + c, width);
                r[i / 16][i % 16] = vld1q_u8(tmp);
            }
        }
        uint8x16_t cols[2][16];
        transpose16(r[0], cols[0]);
        transpose16(r[1], cols[1]);
        for (size_t j = 0; j < width; ++j) {
            const uint8_t * tp = t + (c + j) * 32;
            const uint8x16_t thi = vld1q_u8(tp);
            const uint8x16_t tlo = vld1q_u8(tp + 16);
            for (size_t h = 0; h < 2; ++h) {
                const uint8x16_t x = cols[h][j];
                // Each entry is at most 127, so the pair fits a byte.
                const uint8x16_t sum = vaddq_u8(vqtbl1q_u8(thi, vshrq_n_u8(x, 4)), vqtbl1q_u8(tlo, vandq_u8(x, nib)));
                acc16[2 * h] = vaddw_u8(acc16[2 * h], vget_low_u8(sum));
                acc16[2 * h + 1] = vaddw_high_u8(acc16[2 * h + 1], sum);
            }
        }
        c += 16;
        // 16 positions add at most 16 * 254 to a u16 lane: widen every 256
        // positions, and at the end.
        if (c % 256 == 0 || c >= nsg) {
            for (size_t k = 0; k < 4; ++k) {
                acc32[2 * k] = vaddw_u16(acc32[2 * k], vget_low_u16(acc16[k]));
                acc32[2 * k + 1] = vaddw_high_u16(acc32[2 * k + 1], acc16[k]);
                acc16[k] = vdupq_n_u16(0);
            }
        }
    }
    for (size_t k = 0; k < 8; ++k) {
        vst1q_u32(out + 4 * k, acc32[k]);
    }
}
#endif

// One block of sign scores, with the kernel upstream scans the sign region
// with on this index's host: the `vpermb` scan over a vector-major region
// (x86), the deferred nibble LUT scan for a lone query on aarch64, the
// nibble LUT scan otherwise.
void score_sign_block(const ggml_vec_index & idx, const SignLut & lut, const uint8_t * sign, size_t block, size_t n,
        float * out) {
    const float * vscales = idx.turbovec_scale.data();
    if (turbovec_sign_layout(idx) == kTurboVecLayoutVm) {
        score_turbovec_lut_block_vnni(lut.values.data(), lut.split.empty() ? nullptr : lut.split.data(), lut.scale,
            lut.bias, sign, vscales, block, n, 2, idx.dim / 2, out);
        return;
    }
#if GGML_VEC_INDEX_USE_NEON
    if (lut.deferred) {
        score_sign_block_deferred(
            lut.values.data(), lut.scale, lut.bias, sign, vscales, block, n, static_cast<size_t>(idx.dim) / 8, out);
        return;
    }
#endif
    score_turbovec_lut_block(lut.values.data(), lut.scale, lut.bias, sign, vscales, block, n, 2, idx.dim / 2, out);
}

// One query as bit masks over a low plane row (upstream LowPlanes): each
// coordinate rounded to a signed kLowBits-bit weight, the row's dot product
// counted with AND and popcount per weight bit.
struct LowPlanes {
    std::vector<uint64_t> masks;
    float unit = 0.0f;
    int32_t sum_w = 0;
    size_t row_len = 0;
};

LowPlanes build_low_planes(const float * q, float m, int dim) {
    LowPlanes out;
    const size_t n_bytes = static_cast<size_t>(dim) / 8;
    const size_t n_chunks = (n_bytes + kLowChunk - 1) / kLowChunk;
    std::vector<uint8_t> masks(n_chunks * 2 * kLowBits * kLowChunk, 0);
    const float top = static_cast<float>((1u << kLowBits) - 1);
    float q_max = 0.0f;
    for (int i = 0; i < dim; ++i) {
        q_max = std::max(q_max, std::fabs(q[i]));
    }
    // For a tiny query top / q_max rounds to infinity. Upstream's Rust casts
    // saturate there (infinity to the top weight, the NaN of a zero
    // coordinate to 0) and unit becomes m / inf = 0; a C++ cast of either is
    // undefined, so that case is spelled out. The double quotient is exact
    // enough to tell, and raises no floating-point flag.
    const bool valid = q_max > 0.0f && std::isfinite(q_max);
    const bool overflow = valid && static_cast<double>(top) / q_max >= 0x1.fffffep127 + 0x1p103;
    const float inv = valid && !overflow ? top / q_max : 0.0f;
    // A non-finite q_max (an infinite coordinate) would make inf * 0 a NaN,
    // which Rust casts to 0 and C++ may not cast: every weight stays 0.
    bool built = !valid;
#if defined(GGML_VEC_INDEX_HAVE_AVX512BW_KERNEL)
    // Upstream's AVX-512 build (sixteen coordinates a step), where dim
    // allows it; the same weights as the loop below. Its conversions are
    // not saturating, so an overflowing query takes the loop.
    const VecIndexCpuFeatures & features = vec_index_cpu_features();
    if (valid && !overflow && dim % 16 == 0 && features.avx512bw && features.avx2 &&
        g_low_kernels_enabled.load(std::memory_order_relaxed)) {
        out.sum_w = ggml_vec_index_detail::turbovec_build_low_masks_avx512(
            q, static_cast<size_t>(dim), inv, static_cast<uint32_t>(top), masks.data());
        built = true;
    }
#endif
    for (int i = 0; i < dim && !built; ++i) {
        const uint32_t w = overflow ? (q[i] != 0.0f ? static_cast<uint32_t>(top) : 0u) :
            std::min(static_cast<uint32_t>(std::fabs(q[i]) * inv + 0.5f), static_cast<uint32_t>(top));
        if (w == 0) {
            continue;
        }
        const bool neg = q[i] < 0.0f;
        out.sum_w += neg ? -static_cast<int32_t>(w) : static_cast<int32_t>(w);
        const size_t byte = static_cast<size_t>(i) / 8;
        const uint8_t bit = static_cast<uint8_t>(0x80u >> (i % 8));
        const size_t base = (byte / kLowChunk) * 2 * kLowBits * kLowChunk +
            (neg ? kLowBits * kLowChunk : 0) + byte % kLowChunk;
        for (int b = 0; b < kLowBits; ++b) {
            if (((w >> b) & 1u) != 0) {
                masks[base + static_cast<size_t>(b) * kLowChunk] |= bit;
            }
        }
    }
    // Stored as little-endian 64-bit words, as upstream reads them.
    out.masks.resize(masks.size() / 8);
    for (size_t i = 0; i < out.masks.size(); ++i) {
        out.masks[i] = get_u64_le(masks.data() + i * 8);
    }
    out.unit = inv > 0.0f ? m / inv : 0.0f;
    out.row_len = n_bytes;
    return out;
}

uint64_t load_word(const uint8_t * bytes, size_t n) {
    uint8_t w[8] = {};
    std::memcpy(w, bytes, n);
    return get_u64_le(w);
}

#if GGML_VEC_INDEX_USE_NEON
// Upstream low_dot_neon: the bit counts a byte at a time, each plane's
// difference (-8..8) weighted into two i8 accumulators, so the result is
// the scalar loop's integer.
int64_t low_dot_neon(const uint8_t * masks, const uint8_t * row, size_t row_len) {
    static_assert(kLowBits == 6, "the byte accumulators hold three bits' worth each");
    int32x4_t acc_lo = vdupq_n_s32(0);
    int32x4_t acc_hi = vdupq_n_s32(0);
    for (size_t c = 0; c * kLowChunk < row_len; ++c) {
        const size_t chunk_len = std::min(kLowChunk, row_len - c * kLowChunk);
        const uint8_t * group = masks + c * 2 * kLowBits * kLowChunk;
        const uint8_t * chunk = row + c * kLowChunk;
        for (size_t j = 0; j < chunk_len; j += 16) {
            uint8x16_t l;
            if (chunk_len - j >= 16) {
                l = vld1q_u8(chunk + j);
            } else {
                uint8_t t[16] = {};
                std::memcpy(t, chunk + j, chunk_len - j);
                l = vld1q_u8(t);
            }
            int8x16_t lo = vdupq_n_s8(0);
            int8x16_t hi = vdupq_n_s8(0);
            for (int b = 0; b < kLowBits; ++b) {
                const uint8x16_t pos = vcntq_u8(vandq_u8(l, vld1q_u8(group + static_cast<size_t>(b) * kLowChunk + j)));
                const uint8x16_t neg =
                    vcntq_u8(vandq_u8(l, vld1q_u8(group + static_cast<size_t>(kLowBits + b) * kLowChunk + j)));
                const int8x16_t d = vreinterpretq_s8_u8(vsubq_u8(pos, neg));
                if (b < 3) {
                    lo = vaddq_s8(lo, vshlq_s8(d, vdupq_n_s8(static_cast<int8_t>(b))));
                } else {
                    hi = vaddq_s8(hi, vshlq_s8(d, vdupq_n_s8(static_cast<int8_t>(b - 3))));
                }
            }
            acc_lo = vpadalq_s16(acc_lo, vpaddlq_s8(lo));
            acc_hi = vpadalq_s16(acc_hi, vpaddlq_s8(hi));
        }
    }
    return vaddlvq_s32(acc_lo) + 8 * vaddlvq_s32(acc_hi);
}

// Upstream exact4_sum_neon: one row's 4-bit permute-dot integer sum from
// its planes, the sign plane read from the lane-blocked sign region
// (aarch64's layout), so the row's exact score needs no block scan.
int32_t exact4_sum_blocked(const int8_t * w, const int8_t * levels, const uint8_t * sign_block,
        const uint8_t * low_row, size_t nsg, size_t lane) {
    // spread8[b]: byte i is bit i of b.
    static const std::array<uint64_t, 256> spread8 = [] {
        std::array<uint64_t, 256> t{};
        for (size_t b = 0; b < 256; ++b) {
            for (int i = 0; i < 8; ++i) {
                t[b] |= static_cast<uint64_t>((b >> i) & 1u) << (8 * i);
            }
        }
        return t;
    }();
    auto codes_of = [&](size_t g) {
        return (spread8[sign_block[g * kBlock + lane]] << 3) | (spread8[low_row[2 * nsg + g]] << 2) |
            (spread8[low_row[nsg + g]] << 1) | spread8[low_row[g]];
    };
    const int8x16_t level_table = vld1q_s8(levels);
    int32x4_t acc = vdupq_n_s32(0);
    size_t c = 0;
    for (; c + 2 <= nsg; c += 2) {
        const uint64_t codes[2] = { codes_of(c), codes_of(c + 1) };
        const int8x16_t lv = vqtbl1q_s8(level_table, vreinterpretq_u8_u64(vld1q_u64(codes)));
        const int8x16_t wv = vld1q_s8(w + c * 8);
        acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(lv), vget_low_s8(wv)));
        acc = vpadalq_s16(acc, vmull_high_s8(lv, wv));
    }
    int32_t sum = vaddvq_s32(acc);
    for (; c < nsg; ++c) {
        const uint64_t codes = codes_of(c);
        for (int b = 0; b < 8; ++b) {
            sum += static_cast<int32_t>(w[c * 8 + static_cast<size_t>(b)]) * levels[(codes >> (8 * b)) & 15];
        }
    }
    return sum;
}
#endif

// Upstream pack.rs PLANES_COMB: entry (sign_nibble << 4) | low_nibble is
// the 2-bit code byte of the four dims those nibbles cover.
struct PlanesComb {
    uint8_t t[256];
    constexpr PlanesComb() : t{} {
        for (unsigned i = 0; i < 256; ++i) {
            t[i] = static_cast<uint8_t>((spread(i >> 4) << 1) | spread(i & 15));
        }
    }
    // A nibble's bits at the even positions of a byte.
    static constexpr unsigned spread(unsigned n) {
        return (((n >> 3) & 1) << 6) | (((n >> 2) & 1) << 4) | (((n >> 1) & 1) << 2) | (n & 1);
    }
};
constexpr PlanesComb kPlanesComb;

// Whether a 2-bit index's shortlist is rescored row by row from the
// planes (upstream legacy_score), and with which epilogue: the `vpermb`
// scan's (sum * scale + bias, unfused) or the LUT kernels' fused flush
// every kTurboVecFlushEvery byte groups (NEON, and the vm8 kernel's scalar
// model). An x86 sequential layout scores with the AVX2 kernel or the
// scalar loop depending on the CPU, so it keeps the block scan.
bool exact2_rows(const ggml_vec_index & idx) {
    if (idx.turbovec_bits != 2) {
        return false;
    }
#if GGML_VEC_INDEX_USE_NEON
    return true;
#else
    return turbovec_scores_vnni_lut(idx) || idx.turbovec_blocked_vm8;
#endif
}

// Upstream legacy_score: one row's exact 2-bit score from its packed
// planes row (low plane, then sign plane), the full scan's u8 table
// entries summed exactly and its epilogue applied in its order, so the
// result is the block scan's score bit for bit.
// The epilogue is a template parameter so the `vpermb` form's loop carries
// no flush test per byte group (upstream's x86 legacy_score has none).
template <bool FMA_FLUSH>
float exact2_row_score_t(const uint8_t * lut, float scale, float bias, const uint8_t * row, size_t nsg, float vscale) {
    const uint8_t * low = row;
    const uint8_t * sign = row + nsg;
    constexpr size_t flush = static_cast<size_t>(kTurboVecFlushEvery) / 2;
    uint32_t sum = 0;
    float fa = bias;
    for (size_t g = 0; g < nsg; ++g) {
        const uint8_t sb = sign[g];
        const uint8_t lb = low[g];
        const uint8_t c0 = kPlanesComb.t[(sb & 0xf0) | (lb >> 4)];
        const uint8_t c1 = kPlanesComb.t[((sb & 0x0f) << 4) | (lb & 0x0f)];
        const uint8_t * t = lut + g * 64;
        sum += static_cast<uint32_t>(t[c0 >> 4]) + t[16 + (c0 & 0x0f)] + t[32 + (c1 >> 4)] + t[48 + (c1 & 0x0f)];
        if (FMA_FLUSH && (g + 1) % flush == 0) {
            fa = std::fma(scale, static_cast<float>(sum), fa);
            sum = 0;
        }
    }
    if (!FMA_FLUSH) {
        const float product = static_cast<float>(sum) * scale;
        const float f = product + bias;
        return f * vscale;
    }
    if (nsg % flush != 0) {
        fa = std::fma(scale, static_cast<float>(sum), fa);
    }
    return fa * vscale;
}

float exact2_row_score(const uint8_t * lut, float scale, float bias, const uint8_t * row, size_t nsg, float vscale,
        bool fma_flush) {
    return fma_flush ? exact2_row_score_t<true>(lut, scale, bias, row, nsg, vscale) :
                       exact2_row_score_t<false>(lut, scale, bias, row, nsg, vscale);
}

int64_t low_dot(const LowPlanes & planes, const uint8_t * row) {
#if GGML_VEC_INDEX_USE_NEON
    return low_dot_neon(reinterpret_cast<const uint8_t *>(planes.masks.data()), row, planes.row_len);
#endif
#if defined(GGML_VEC_INDEX_HAVE_VPOPCNT_KERNEL)
    // Upstream's VPOPCNTDQ form, counting the same bits.
    static const bool has_vpopcnt = [] {
        const VecIndexCpuFeatures & f = vec_index_cpu_features();
        return f.avx512bw && f.avx512vpopcntdq;
    }();
    if (has_vpopcnt && g_low_kernels_enabled.load(std::memory_order_relaxed)) {
        return ggml_vec_index_detail::turbovec_low_dot_avx512(
            reinterpret_cast<const uint8_t *>(planes.masks.data()), row, planes.row_len);
    }
#endif
    int64_t acc = 0;
    const size_t words_per_chunk = kLowChunk / 8;
    for (size_t c = 0; c * kLowChunk < planes.row_len; ++c) {
        const size_t chunk_len = std::min(kLowChunk, planes.row_len - c * kLowChunk);
        const uint64_t * group = planes.masks.data() + c * 2 * kLowBits * words_per_chunk;
        for (size_t j = 0; j * 8 < chunk_len; ++j) {
            const uint64_t l = load_word(row + c * kLowChunk + j * 8, std::min<size_t>(8, chunk_len - j * 8));
            for (int b = 0; b < kLowBits; ++b) {
                const uint64_t p = group[static_cast<size_t>(b) * words_per_chunk + j];
                const uint64_t n = group[static_cast<size_t>(kLowBits + b) * words_per_chunk + j];
                // A multiply, not a shift: the difference can be negative.
                acc += (static_cast<int64_t>(popcount64(l & p)) - static_cast<int64_t>(popcount64(l & n))) *
                    (int64_t{ 1 } << b);
            }
        }
    }
    return acc;
}

}  // namespace

bool turbovec_planes_for(int bits, int dim) {
    const int forced = g_planes_override.load(std::memory_order_relaxed);
    if (forced >= 0) {
        return forced != 0 && (bits == 2 || bits == 4) && dim % 32 == 0;
    }
    // Upstream pack.rs `planes_for`: on x86 both widths need the
    // vector-major kernels (the `vpermb` scan reads the sign region);
    // aarch64 searches 2-bit planes always and 4-bit planes with its
    // dot-product kernels. dim % 32 == 0 is a whole sign unit at either
    // width.
    if (dim % 32 != 0 || (bits != 2 && bits != 4)) {
        return false;
    }
    if (turbovec_x86_rules()) {
        return turbovec_cpu_vector_major();
    }
#if GGML_VEC_INDEX_USE_NEON
    // TURBOVEC_2BIT_VM8 replaces the classic NEON scan the sign region
    // needs, so upstream drops the planes layout at both widths.
    if (turbovec_vm8_2bit_enabled()) {
        return false;
    }
    return bits == 2 || turbovec_cpu_vector_major();
#else
    return false;
#endif
}

bool turbovec_planes_wanted(int bits, int dim, size_t n_vectors) {
    return n_vectors > 0 && n_vectors >= g_planes_min_vectors.load(std::memory_order_relaxed) &&
        turbovec_planes_for(bits, dim);
}

void turbovec_planes_refresh(ggml_vec_index & idx, size_t first_slot) {
    if (!is_turbovec(idx) || idx.turbovec_legacy) {
        return;
    }
    const size_t n = idx.slot_to_id.size();
    std::lock_guard<std::mutex> lock(idx.turbovec_planes_mutex);
    if (!idx.turbovec_planes) {
        // An index with staged search switched off never needs the planes.
        if (!idx.turbovec_staged_enabled || !turbovec_planes_wanted(idx.turbovec_bits, idx.dim, n)) {
            return;
        }
        // Promotion: the statistics are fitted afresh from the new cache,
        // and the staged search reads the bit planes from packed rows.
        try {
            turbovec_materialize_packed_rows(idx);
        } catch (const std::bad_alloc &) {
            // Not promoted yet; the next add tries again.
            return;
        }
        idx.turbovec_planes = true;
        idx.turbovec_stats_ready = false;
        idx.turbovec_sign_blocked.clear();
        first_slot = 0;
    }
    try {
        repack_turbovec_sign_codes(
            idx.turbovec_data.data(), n, idx.turbovec_bits, idx.dim, first_slot, idx.turbovec_sign_blocked,
            idx.n_threads, turbovec_sign_layout(idx));
    } catch (const std::bad_alloc &) {
        // Without memory for the sign planes the index drops back to full
        // scans, so a mutation never fails after its rows are committed;
        // the next add promotes it again.
        idx.turbovec_planes = false;
        idx.turbovec_stats_ready = false;
        std::vector<uint8_t>().swap(idx.turbovec_sign_blocked);
        std::vector<float>().swap(idx.turbovec_centre);
        turbovec_settle_rows(idx);
    }
    idx.turbovec_centre_ready = false;
}

void turbovec_planes_reset(ggml_vec_index & idx) {
    std::lock_guard<std::mutex> lock(idx.turbovec_planes_mutex);
    idx.turbovec_planes = false;
    idx.turbovec_stats_ready = false;
    idx.turbovec_centre_ready = false;
    idx.turbovec_sign_blocked.clear();
    idx.turbovec_centre.clear();
}

void turbovec_planes_swap_remove(ggml_vec_index & idx, size_t slot, size_t last) {
    std::lock_guard<std::mutex> lock(idx.turbovec_planes_mutex);
    if (!idx.turbovec_planes) {
        return;
    }
    turbovec_sign_blocked_swap_remove(idx, slot, last);
    idx.turbovec_centre_ready = false;
}

namespace {

// Upstream seeds its buffered collectors for an index of at least 1024 full
// blocks (pack.rs planes_sample) wherever the kernel that scans the sign
// tables implements the collector (search.rs planes_buffered_supported):
// always under its aarch64 rules, under its x86 rules when the tables carry
// `vpermb` split tables, so over a vector-major sign region. A seed changes
// how the shortlist is collected, not which rows it holds (see Collector),
// except that on aarch64 the one query of a batch whose seed runs short is
// rescanned with single-query tables.
constexpr size_t kSeedMinBlocks = 1024;
#ifdef GGML_VEC_INDEX_TEST_HOOKS
// Queries whose seed left them short, and batch scans run, for a test that
// must know the path it checks was taken.
std::atomic<int64_t> g_seed_short_count{ 0 };
std::atomic<int64_t> g_batch_scan_count{ 0 };
#endif
void count_seed_short() {
#ifdef GGML_VEC_INDEX_TEST_HOOKS
    g_seed_short_count.fetch_add(1, std::memory_order_relaxed);
#endif
}
void count_batch_scan() {
#ifdef GGML_VEC_INDEX_TEST_HOOKS
    g_batch_scan_count.fetch_add(1, std::memory_order_relaxed);
#endif
}

// Upstream's collector seed (search.rs sample_seeds over pack.rs
// planes_sample): the r-th best sign score among 48 blocks taken at a
// stride, with r set so a little more than a shortlist's worth of the
// whole index should score above it.
float sample_seed(const ggml_vec_index & idx, const SignLut & lut, size_t s_len, int bits) {
    constexpr size_t kSampleBlocks = 48;
    const size_t n = idx.slot_to_id.size();
    const size_t stride = (n / kBlock) / kSampleBlocks;
    const size_t n_s = kSampleBlocks * kBlock;
    const float r_s = static_cast<float>(s_len * n_s) / static_cast<float>(n);
    const float width = bits == 4 ? 4.0f : 6.0f;
    const float over = std::min(1.0f + width / std::sqrt(std::max(r_s, 1.0f)), 4.0f);
    const size_t r = std::min(std::max(static_cast<size_t>(std::ceil(over * r_s)), size_t{ 6 }), n_s);
    std::vector<float> scores(n_s);
    for (size_t i = 0; i < kSampleBlocks; ++i) {
        // Offset by half a stride so the sample is not the index's head.
        const size_t block = i * stride + stride / 2;
        score_sign_block(idx, lut, idx.turbovec_sign_blocked.data(), block, n, scores.data() + i * kBlock);
    }
    std::vector<float> scratch;
    return kth_best(scores.data(), n_s, r, [](float a, float b) { return static_cast<int>(a > b); }, scratch);
}

// Upstream pack.rs `planes_stats`, from the first 64 blocks of the cache.
TurboVecPlanesStats compute_stats(const ggml_vec_index & idx) {
    const int bits = idx.turbovec_bits;
    const int dim = idx.dim;
    const size_t n = std::min(idx.slot_to_id.size(), static_cast<size_t>(64 * kBlock));
    const size_t row_bytes = turbovec_row_bytes(bits, static_cast<size_t>(dim));
    const size_t nsg = static_cast<size_t>(dim) / 8;
    std::vector<float> boundaries;
    std::vector<float> centroids;
    turbovec_codebook_values(bits, dim, false, boundaries, centroids);
    TurboVecPlanesStats stats;
    if (bits == 2) {
        // Fraction of codes on an outer level (sign bit == low bit).
        uint64_t outer = 0;
        uint64_t total = 0;
        for (size_t v = 0; v < n; ++v) {
            const uint8_t * row = idx.turbovec_data.data() + v * row_bytes;
            for (size_t g = 0; g < nsg; ++g) {
                outer += static_cast<uint64_t>(popcount64(static_cast<uint8_t>(~(row[nsg + g] ^ row[g]))));
                total += 8;
            }
        }
        const float f = total == 0 ? 0.5f : static_cast<float>(outer) / static_cast<float>(total);
        stats.m = centroids[2] * (1.0f - f) + centroids[3] * f;
        stats.alpha = (centroids[3] + centroids[2]) * 0.5f;
        stats.beta[0] = (centroids[3] - centroids[2]) * 0.5f;
        stats.alpha1 = stats.alpha;
        stats.beta1 = stats.beta[0];
        return stats;
    }
    // 4 bits: a least-squares fit of the levels on the bit signs, weighted
    // by how often each level occurs in the sample.
    const size_t levels = size_t{ 1 } << bits;
    std::vector<uint64_t> hist(levels, 0);
    const size_t n_byte_groups = static_cast<size_t>(dim) / static_cast<size_t>(8 / bits);
    std::vector<uint8_t> seq(n_byte_groups);
    for (size_t v = 0; v < n; ++v) {
        turbovec_packed_row_to_seq(idx.turbovec_data.data() + v * row_bytes, bits, dim, seq.data());
        for (uint8_t byte : seq) {
            for (int c = 0; c < 8 / bits; ++c) {
                ++hist[(byte >> (c * bits)) & (levels - 1)];
            }
        }
    }
    uint64_t total_count = 0;
    for (uint64_t h : hist) {
        total_count += h;
    }
    const double total = static_cast<double>(std::max<uint64_t>(total_count, 1));
    const int nf = bits;
    double ata[4][4] = {};
    double atb[4] = {};
    double m = 0.0;
    double w_all = 0.0;
    double w_sr = 0.0;
    double b_s = 0.0;
    double b_r = 0.0;
    for (size_t code = 0; code < levels; ++code) {
        const double w = (static_cast<double>(hist[code]) + 0.5) / total;
        const double c = static_cast<double>(centroids[code]);
        m += (static_cast<double>(hist[code]) / total) * std::fabs(c);
        double x[4] = {};
        x[0] = (code >> (bits - 1)) != 0 ? 1.0 : -1.0;
        for (int j = 0; j < bits - 1; ++j) {
            x[1 + j] = ((code >> j) & 1u) != 0 ? 1.0 : -1.0;
        }
        const double sg = x[0];
        const double top = x[bits - 1];
        w_all += w;
        w_sr += w * sg * top;
        b_s += w * sg * c;
        b_r += w * top * c;
        for (int a = 0; a < nf; ++a) {
            atb[a] += w * x[a] * c;
            for (int b = 0; b < nf; ++b) {
                ata[a][b] += w * x[a] * x[b];
            }
        }
    }
    // Gaussian elimination with partial pivoting.
    double sol[4] = { atb[0], atb[1], atb[2], atb[3] };
    for (int i = 0; i < nf; ++i) {
        int piv = i;
        for (int r = i + 1; r < nf; ++r) {
            // Rust max_by keeps the last of equal maxima.
            if (std::fabs(ata[r][i]) >= std::fabs(ata[piv][i])) {
                piv = r;
            }
        }
        std::swap(ata[i], ata[piv]);
        std::swap(sol[i], sol[piv]);
        const double d = ata[i][i];
        if (std::fabs(d) < 1e-12) {
            continue;
        }
        for (int r = 0; r < nf; ++r) {
            if (r != i) {
                const double f = ata[r][i] / d;
                for (int c = i; c < nf; ++c) {
                    ata[r][c] -= f * ata[i][c];
                }
                sol[r] -= f * sol[i];
            }
        }
    }
    auto coef = [&](int i) {
        return std::fabs(ata[i][i]) < 1e-12 ? 0.0f : static_cast<float>(sol[i] / ata[i][i]);
    };
    for (int j = 0; j < bits - 1; ++j) {
        stats.beta[j] = coef(1 + j);
    }
    const double det = w_all * w_all - w_sr * w_sr;
    if (std::fabs(det) < 1e-12) {
        stats.alpha1 = coef(0);
        stats.beta1 = stats.beta[bits - 2];
    } else {
        stats.alpha1 = static_cast<float>((b_s * w_all - b_r * w_sr) / det);
        stats.beta1 = static_cast<float>((b_r * w_all - b_s * w_sr) / det);
    }
    stats.m = static_cast<float>(m);
    stats.alpha = coef(0);
    return stats;
}

// Rows per task of the centre term and candidates per task of the rank
// and rescore stages: enough to amortize the hand-off.
constexpr size_t kCentreRowsPerTask = 4096;
constexpr size_t kRankPerTask = 64;
// Candidates one ranking batch gathers (upstream RANK_BATCH).
constexpr size_t kRankBatch = 32;
// Ranking steps the row prefetch runs ahead of the count.
constexpr size_t kRankAhead = 2;
// Shortlist ranges per thread of a split scan: each range fills a shortlist
// of its own before it can prune, so fewer ranges than the full scan's.
constexpr size_t kRangesPerThread = 2;

// Upstream pack.rs `planes_centre`: C(v) = sum_d w_d * sign_d(v).
std::vector<float> compute_centre(const ggml_vec_index & idx, float m) {
    const int dim = idx.dim;
    const size_t nsg = static_cast<size_t>(dim) / 8;
    const size_t row_bytes = turbovec_row_bytes(idx.turbovec_bits, static_cast<size_t>(dim));
    const size_t sign_at = static_cast<size_t>(idx.turbovec_bits - 1) * nsg;
    std::vector<float> w(static_cast<size_t>(dim));
    for (int d = 0; d < dim; ++d) {
        w[static_cast<size_t>(d)] =
            m * idx.turbovec_tqplus_shift[static_cast<size_t>(d)] / idx.turbovec_tqplus_scale[static_cast<size_t>(d)];
    }
    const size_t n = idx.slot_to_id.size();
    std::vector<float> centre(n);
    // Each row's sum is its own, so rows may spread over threads.
    parallel_for((n + kCentreRowsPerTask - 1) / kCentreRowsPerTask, idx.n_threads, [&](size_t task, size_t) {
        const size_t end = std::min(n, (task + 1) * kCentreRowsPerTask);
        for (size_t v = task * kCentreRowsPerTask; v < end; ++v) {
            const uint8_t * sign = idx.turbovec_data.data() + v * row_bytes + sign_at;
            float acc = 0.0f;
            for (size_t g = 0; g < nsg; ++g) {
                for (int j = 0; j < 8; ++j) {
                    const float wj = w[g * 8 + static_cast<size_t>(j)];
                    acc += ((sign[g] >> (7 - j)) & 1u) != 0 ? wj : -wj;
                }
            }
            centre[v] = acc;
        }
    });
    return centre;
}

struct Candidate {
    size_t slot;
    float score;
};

// Best first by score, then by slot (upstream breaks score ties by index).
bool candidate_better(const Candidate & a, const Candidate & b) {
    if (a.score != b.score) {
        return a.score > b.score;
    }
    return a.slot < b.slot;
}

// candidate_better as an integer without a branch, for the cuts (a
// function object for the reason ScoreIdBetterI is).
struct CandidateBetterI {
    int operator()(const Candidate & a, const Candidate & b) const {
        return static_cast<int>(a.score > b.score) | (static_cast<int>(a.score == b.score) & static_cast<int>(a.slot < b.slot));
    }
};

// Cuts cands to its best `keep` under the strict (score, slot) order, in
// the order they had, and `tops` (aligned with cands) with them.
void keep_best(std::vector<Candidate> & cands, size_t keep, std::vector<float> * tops = nullptr) {
    if (cands.size() <= keep) {
        return;
    }
    std::vector<Candidate> scratch;
    const CandidateBetterI better;
    const Candidate kth = kth_best(cands.data(), cands.size(), keep, better, scratch);
    size_t w = 0;
    for (size_t i = 0; i < cands.size(); ++i) {
        const Candidate c = cands[i];
        cands[w] = c;
        if (tops != nullptr) {
            (*tops)[w] = (*tops)[i];
        }
        w += static_cast<size_t>(1 - better(kth, c));
    }
    // Exactly `keep` under the strict order; only a NaN score, which it
    // does not cover, can leave more, and then the tail goes.
    cands.resize(std::min(w, keep));
    if (tops != nullptr) {
        tops->resize(cands.size());
    }
}

}  // namespace

namespace {

// The planes cache's statistics and centre, computed on first use. False
// without a valid cache: the caller's full scan (and its per-row fallback)
// answers then.
bool planes_ready(const ggml_vec_index & idx, TurboVecPlanesStats & stats, const std::vector<float> *& centre) {
    const int bits = idx.turbovec_bits;
    const size_t n = idx.slot_to_id.size();
    const size_t nsg = static_cast<size_t>(idx.dim) / 8;
    std::lock_guard<std::mutex> lock(idx.turbovec_planes_mutex);
    const size_t n_blocks = (n + kBlock - 1) / kBlock;
    const size_t n_byte_groups = static_cast<size_t>(idx.dim) / static_cast<size_t>(8 / bits);
    if (!idx.turbovec_planes || idx.turbovec_sign_blocked.size() != n_blocks * nsg * kBlock ||
        idx.turbovec_blocked_n_blocks != n_blocks ||
        idx.turbovec_blocked_data.size() != n_blocks * n_byte_groups * kBlock) {
        return false;
    }
    if (!idx.turbovec_stats_ready) {
        idx.turbovec_stats = compute_stats(idx);
        idx.turbovec_stats_ready = true;
    }
    stats = idx.turbovec_stats;
    centre = nullptr;
    if (!idx.turbovec_tqplus_shift.empty()) {
        if (!idx.turbovec_centre_ready) {
            idx.turbovec_centre = compute_centre(idx, stats.m);
            idx.turbovec_centre_ready = true;
        }
        centre = &idx.turbovec_centre;
    }
    return true;
}

// Collects one query's rows above a threshold from its sign scores
// (upstream's buffered collector): rows are appended, and at twice s_len
// the buffer is cut back to its best s_len, whose worst score becomes the
// threshold. A seed is the threshold until the first cut-back. Whenever at
// least s_len rows score above the seed, the rows kept are exactly the top
// s_len under the strict (score, slot) order, whatever order they came in.
struct Collector {
    std::vector<ScoreId> top;
    size_t s_len = 0;
    float seed = -std::numeric_limits<float>::infinity();
    bool seed_active = false;
    bool cut = false;
    double threshold = 0.0;
    // Rows that scored above the seed, to tell a short query.
    size_t above_seed = 0;
    // For one range of a split query: the best s_len-th score any range has
    // kept so far. Every row of the shortlist scores at least that, so a
    // range may drop a row that scores strictly less (a tie may still win
    // on its slot). Null for a lone range.
    std::atomic<double> * floor_score = nullptr;
    // The VNNI fused scan reads the floor as a bare double (vnni.cpp
    // load_floor_relaxed), so it needs the atomic to be one.
    static_assert(sizeof(std::atomic<double>) == sizeof(double) && std::atomic<double>::is_always_lock_free,
        "std::atomic<double> must be a lock-free bare double");

    void start(size_t s_len_, float seed_, size_t reserve, std::atomic<double> * floor) {
        top.clear();
        top.reserve(reserve);
        s_len = s_len_;
        seed = seed_;
        seed_active = seed > -std::numeric_limits<float>::infinity();
        cut = seed_active;
        threshold = seed_active ? static_cast<double>(seed) : 0.0;
        above_seed = 0;
        floor_score = floor;
    }

    // Scratch for the cuts, kept across them.
    std::vector<ScoreId> scratch;

    void cut_back() {
        cut = true;
        threshold = cut_to_best(top, s_len, ScoreIdBetterI{},scratch).score;
        if (floor_score != nullptr) {
            double seen = floor_score->load(std::memory_order_relaxed);
            while (threshold > seen && !floor_score->compare_exchange_weak(seen, threshold, std::memory_order_relaxed)) {
            }
        }
    }

    // One scored block: lanes [0, count) are rows block * kBlock + lane,
    // the rest -inf padding. Rows arrive in slot order, so once s_len rows
    // are kept a row scoring no better than the worst of them cannot enter.
    void add_block(const float * block_scores, size_t block, size_t count, const uint8_t * allowed_mask) {
        const double floor = floor_score != nullptr ? floor_score->load(std::memory_order_relaxed) :
                                                      -std::numeric_limits<double>::infinity();
#if GGML_VEC_INDEX_USE_NEON
        // Upstream neon_block_topk_update: a block whose best lane cannot
        // enter is skipped whole, and under a threshold only the lanes above
        // it are visited, taken as a mask so the walk does not branch on
        // every lane. Padding lanes are -inf, so they never count or enter.
        float32x4_t v[8];
        for (size_t i = 0; i < 8; ++i) {
            v[i] = vld1q_f32(block_scores + i * 4);
        }
        if (seed_active) {
            const float32x4_t seed_v = vdupq_n_f32(seed);
            uint32x4_t above = vdupq_n_u32(0);
            for (const float32x4_t & x : v) {
                above = vsubq_u32(above, vcgtq_f32(x, seed_v));
            }
            above_seed += vaddvq_u32(above);
        }
        const float32x4_t m0 = vmaxq_f32(vmaxq_f32(v[0], v[1]), vmaxq_f32(v[2], v[3]));
        const float32x4_t m1 = vmaxq_f32(vmaxq_f32(v[4], v[5]), vmaxq_f32(v[6], v[7]));
        const double block_max = static_cast<double>(vmaxvq_f32(vmaxq_f32(m0, m1)));
        if ((cut && block_max <= threshold) || block_max < floor) {
            return;
        }
        if (cut) {
            // Unlike the SSE2 walk, vcgtq drops a NaN lane; the API's finite,
            // bounded inputs keep sign scores finite, so the two agree.
            const float32x4_t thr = vdupq_n_f32(static_cast<float>(threshold));
            for (size_t half = 0; half < 2; ++half) {
                const uint16x4_t c0 = vmovn_u32(vcgtq_f32(v[half * 4], thr));
                const uint16x4_t c1 = vmovn_u32(vcgtq_f32(v[half * 4 + 1], thr));
                const uint16x4_t c2 = vmovn_u32(vcgtq_f32(v[half * 4 + 2], thr));
                const uint16x4_t c3 = vmovn_u32(vcgtq_f32(v[half * 4 + 3], thr));
                const uint8x16_t b = vcombine_u8(vmovn_u16(vcombine_u16(c0, c1)), vmovn_u16(vcombine_u16(c2, c3)));
                // Four bits a lane (the narrowing-shift idiom).
                uint64_t bits = vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(b), 4)), 0) &
                    0x1111111111111111ull;
                while (bits != 0) {
                    const size_t lane = half * 16 + static_cast<size_t>(ctz64(bits)) / 4;
                    bits &= bits - 1;
                    const double score = static_cast<double>(block_scores[lane]);
                    if (lane >= count || score < floor) {
                        continue;
                    }
                    const size_t v_slot = block * kBlock + lane;
                    if (allowed_mask != nullptr && allowed_mask[v_slot] == 0) {
                        continue;
                    }
                    top.push_back({ score, static_cast<uint64_t>(v_slot) });
                }
            }
            if (top.size() >= 2 * s_len) {
                cut_back();
            }
            return;
        }
#elif defined(__SSE2__) || defined(_M_X64)
        // The NEON branch with SSE2, the x86_64 baseline, so no dispatch.
        // _mm_max_ps does not propagate a NaN the way vmaxq_f32 does, so a
        // NaN lane is found with cmpunord and keeps its block; in the walk
        // cmpnle is true for a NaN lane, which the plain loop below pushes
        // as well. Padding lanes are -inf, so they never count or enter.
        __m128 v[8];
        for (size_t i = 0; i < 8; ++i) {
            v[i] = _mm_loadu_ps(block_scores + i * 4);
        }
        if (seed_active) {
            const __m128 seed_v = _mm_set1_ps(seed);
            uint32_t above = 0;
            for (size_t i = 0; i < 8; ++i) {
                above |= static_cast<uint32_t>(_mm_movemask_ps(_mm_cmpgt_ps(v[i], seed_v))) << (4 * i);
            }
            above_seed += static_cast<size_t>(popcount64(above));
        }
        const __m128 m0 = _mm_max_ps(_mm_max_ps(v[0], v[1]), _mm_max_ps(v[2], v[3]));
        const __m128 m1 = _mm_max_ps(_mm_max_ps(v[4], v[5]), _mm_max_ps(v[6], v[7]));
        __m128 m = _mm_max_ps(m0, m1);
        m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(1, 0, 3, 2)));
        m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(2, 3, 0, 1)));
        const double block_max = static_cast<double>(_mm_cvtss_f32(m));
        const __m128 unord = _mm_or_ps(_mm_or_ps(_mm_cmpunord_ps(v[0], v[1]), _mm_cmpunord_ps(v[2], v[3])),
            _mm_or_ps(_mm_cmpunord_ps(v[4], v[5]), _mm_cmpunord_ps(v[6], v[7])));
        if (_mm_movemask_ps(unord) == 0 && ((cut && block_max <= threshold) || block_max < floor)) {
            return;
        }
        if (cut) {
            const __m128 thr = _mm_set1_ps(static_cast<float>(threshold));
            uint64_t bits = 0;
            for (size_t i = 0; i < 8; ++i) {
                bits |= static_cast<uint64_t>(_mm_movemask_ps(_mm_cmpnle_ps(v[i], thr))) << (4 * i);
            }
            while (bits != 0) {
                const size_t lane = static_cast<size_t>(ctz64(bits));
                bits &= bits - 1;
                const double score = static_cast<double>(block_scores[lane]);
                if (lane >= count || score < floor) {
                    continue;
                }
                const size_t v_slot = block * kBlock + lane;
                if (allowed_mask != nullptr && allowed_mask[v_slot] == 0) {
                    continue;
                }
                top.push_back({ score, static_cast<uint64_t>(v_slot) });
            }
            if (top.size() >= 2 * s_len) {
                cut_back();
            }
            return;
        }
#else
        if (seed_active) {
            for (size_t lane = 0; lane < count; ++lane) {
                above_seed += block_scores[lane] > seed ? 1 : 0;
            }
        }
#endif
        for (size_t lane = 0; lane < count; ++lane) {
            const double score = static_cast<double>(block_scores[lane]);
            if ((cut && score <= threshold) || score < floor) {
                continue;
            }
            const size_t v = block * kBlock + lane;
            if (allowed_mask != nullptr && allowed_mask[v] == 0) {
                continue;
            }
            top.push_back({ score, static_cast<uint64_t>(v) });
        }
        if (top.size() >= 2 * s_len) {
            cut_back();
        }
    }

#if defined(GGML_VEC_INDEX_HAVE_VNNI_KERNEL)
    // The lanes the fused x86 scan hands over (turbovec_vnni_lut_scan_blk2):
    // the SSE2 add_block's walk over them, the seed count and the block
    // tests having run in the kernel. No allowed mask: that scan is only
    // run unfiltered.
    void add_lanes(size_t block, double floor, const float * scores, const int32_t * lanes, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            const double score = static_cast<double>(scores[i]);
            if (score < floor) {
                continue;
            }
            top.push_back({ score, static_cast<uint64_t>(block * kBlock + static_cast<size_t>(lanes[i])) });
        }
        if (top.size() >= 2 * s_len) {
            cut_back();
        }
    }

    static void vnni_sink(
            void * ctx,
            size_t block,
            double floor,
            const float * scores,
            const int32_t * lanes,
            size_t n,
            ggml_vec_index_detail::TurboVecVnniScanState & state) {
        Collector & col = *static_cast<Collector *>(ctx);
        col.add_lanes(block, floor, scores, lanes, n);
        state.threshold = col.threshold;
        state.cut = col.cut;
    }

    // The state the fused kernel reads before each block; the sink keeps
    // the threshold current and the caller takes the seed count back.
    ggml_vec_index_detail::TurboVecVnniScanState vnni_state() const {
        ggml_vec_index_detail::TurboVecVnniScanState state;
        state.threshold = threshold;
        state.cut = cut;
        state.seed_active = seed_active;
        state.seed = seed;
        state.floor_score = floor_score;
        state.above_seed = above_seed;
        return state;
    }

    // Blocks [block_begin, block_end) of a vector-major sign region through
    // the fused kernel: the same rows as add_block over its scores.
    void scan_vnni(const SignLut & lut, const uint8_t * blocks, const float * vscales, size_t block_begin,
            size_t block_end, size_t n, size_t n_byte_groups) {
        ggml_vec_index_detail::TurboVecVnniScanState state = vnni_state();
        ggml_vec_index_detail::turbovec_vnni_lut_scan_blk2(lut.split.data(), lut.scale, lut.bias, vscales, blocks,
            block_begin, block_end, n, n_byte_groups, state, &Collector::vnni_sink, this);
        above_seed = state.above_seed;
    }
#endif

    // A range of a split query hands back at most s_len rows.
    void finish() {
        if (floor_score != nullptr && top.size() > s_len) {
            cut_back();
        }
    }
};

#if defined(GGML_VEC_INDEX_HAVE_VNNI_KERNEL)
// Whether a lone query's vector-major sign region is scanned by the fused
// AVX-512 kernel: the `vpermb` kernel's own gate (turbovec_cpu_dot_product)
// and split tables, as score_turbovec_lut_blocks_vnni requires.
bool vnni_fused_scan_ready(const SignLut & lut) {
    return !lut.split.empty() && turbovec_cpu_dot_product() &&
        g_vnni_fused_scan_enabled.load(std::memory_order_relaxed);
}

// The same for a batch task's queries, every one with split tables.
bool vnni_fused_batch_ready(const uint8_t * const * splits, size_t n) {
    if (!turbovec_cpu_dot_product() || !g_vnni_fused_batch_enabled.load(std::memory_order_relaxed)) {
        return false;
    }
    for (size_t i = 0; i < n; ++i) {
        if (splits[i] == nullptr) {
            return false;
        }
    }
    return true;
}
#endif

}  // namespace

bool turbovec_staged_search(
        const ggml_vec_index & idx,
        const float * query,
        float bias_corr,
        int k,
        const uint8_t * allowed_mask,
        const TurboVecStagedOpts & opts,
        std::vector<ScoreId> & heap) {
    if (opts.seed_short != nullptr) {
        *opts.seed_short = false;
    }
    const bool single_query = opts.single_query;
    if (!is_turbovec(idx) || idx.turbovec_legacy || !idx.turbovec_staged_enabled) {
        return false;
    }
    const int bits = idx.turbovec_bits;
    const int dim = idx.dim;
    const size_t n = idx.slot_to_id.size();
    const size_t nsg = static_cast<size_t>(dim) / 8;
    const size_t n_low = static_cast<size_t>(bits - 1);
    const size_t row_bytes = turbovec_row_bytes(bits, static_cast<size_t>(dim));
    const float * vscales = idx.turbovec_scale.data();
    const bool calibrated = !idx.turbovec_tqplus_shift.empty();

    TurboVecPlanesStats stats;
    const std::vector<float> * centre = nullptr;
    if (!planes_ready(idx, stats, centre)) {
        return false;
    }
    const float m = stats.m;

    size_t n_allowed = n;
    if (allowed_mask != nullptr) {
        n_allowed = 0;
        for (size_t v = 0; v < n; ++v) {
            n_allowed += allowed_mask[v] != 0 ? 1 : 0;
        }
    }
    const size_t k_sz = static_cast<size_t>(k);
    const size_t s_len = shortlist_len(k_sz, bits);
    const size_t t_len = rescore_len(k_sz, bits);
    const bool refines = s_len < n_allowed && t_len < s_len;
    // Upstream's aarch64 rules: 4-bit ranking reads the sign tables, and a
    // lone query's sign tables are capped for its deferred-widening scan.
    // Its x86 rules count bit masks and keep 7-bit tables throughout.
#if GGML_VEC_INDEX_USE_NEON
    const bool arm_rules = !idx.turbovec_x86;
#else
    const bool arm_rules = false;
#endif
    const bool table_rank = arm_rules && bits == 4;

    // The sign stage ranks on the query pulled toward the centre; later
    // stages read the plain query.
    std::vector<float> q_sign(query, query + dim);
    if (calibrated) {
        for (int d = 0; d < dim; ++d) {
            q_sign[static_cast<size_t>(d)] +=
                kSignCentre * idx.turbovec_tqplus_shift[static_cast<size_t>(d)] / idx.turbovec_tqplus_scale[static_cast<size_t>(d)];
        }
    }
    // A batch query arrives with its shortlist scanned, so its sign tables
    // are needed only by the NEON table ranking.
    SignLut sign_lut;
    if (opts.shortlist == nullptr || table_rank) {
        sign_lut = build_sign_lut(q_sign.data(), m, dim, arm_rules && (single_query || opts.sign_single));
        sign_lut.bias += bias_corr;
        split_sign_lut(idx, sign_lut);
    }
    SignLut plain_lut;
    if (refines && table_rank && calibrated) {
        plain_lut = build_sign_lut(query, m, dim, single_query);
        plain_lut.bias += bias_corr;
    } else if (refines && table_rank && opts.sign_single && !single_query) {
        // The ranking keeps the batch's sign tables (upstream ranks on
        // sign_luts, not the rescan's).
        plain_lut = build_sign_lut(q_sign.data(), m, dim, false);
        plain_lut.bias += bias_corr;
    }
    // Read by the NEON table ranking only.
    [[maybe_unused]] const SignLut & rank_lut = plain_lut.values.empty() ? sign_lut : plain_lut;
    // The query's tables for the ranking and the exact rescore. A split
    // query builds them while its blocks are scanned.
    LowPlanes low_planes;
    auto build_low = [&]() {
        if (refines && !table_rank) {
            low_planes = build_low_planes(query, m, dim);
        }
    };
    std::vector<uint8_t> exact_lut;
    std::vector<uint8_t> exact_split;
    float exact_scale = 1.0f;
    float exact_bias = 0.0f;
    TurboVecPermuteDot pd;
    const bool permute_dot = turbovec_uses_permute_dot(idx);
    // Upstream rescores a 4-bit shortlist from the planes themselves
    // (`exact4_sum`), its sign plane read from the vector-major region on
    // x86 and from the lane-blocked one on aarch64.
    const bool vm_sign = turbovec_sign_layout(idx) == kTurboVecLayoutVm;
    const bool exact4 = permute_dot && (vm_sign || arm_rules);
    // A 2-bit shortlist likewise (`legacy_score`), from the packed rows.
    const bool exact2 = exact2_rows(idx);
    const bool exact2_fma = !turbovec_scores_vnni_lut(idx);
    std::vector<int8_t> exact4_w;
    bool exact_built = false;
    auto build_exact = [&]() {
        if (permute_dot) {
            build_turbovec_permute_dot(query, dim, pd, turbovec_layout(idx) == kTurboVecLayoutVm8);
            pd.bias = bias_corr;
            if (exact4) {
                // Position 8c + b is dim 8c + 7 - b, zero-padded to 64s.
                exact4_w.assign((static_cast<size_t>(dim) + 63) / 64 * 64, 0);
                for (int d = 0; d < dim; ++d) {
                    exact4_w[static_cast<size_t>(d / 8) * 8 + 7 - static_cast<size_t>(d % 8)] =
                        pd.weights[static_cast<size_t>(d)];
                }
                int32_t wsum = 0;
                for (int8_t w : pd.weights) {
                    wsum += w;
                }
                pd.zero = -128 * wsum;
            }
        } else {
            build_turbovec_lut(bits, false, query, dim, exact_lut, exact_scale, exact_bias);
            exact_bias += bias_corr;
            if (turbovec_scores_vnni_lut(idx) && !exact2) {
                exact_split.resize(exact_lut.size());
                turbovec_split_lut_for_vnni(exact_lut.data(), exact_lut.size() / 32, exact_split.data());
            }
        }
        exact_built = true;
    };
    // One candidate's exact 4-bit score from the planes: the permute-dot
    // integer sum and its epilogue, the full scan's score bit for bit.
    auto exact4_score = [&](size_t v) {
        const uint8_t * sign_block = idx.turbovec_sign_blocked.data() + (v / kBlock) * nsg * kBlock;
        const uint8_t * low_row = idx.turbovec_data.data() + v * row_bytes;
#if GGML_VEC_INDEX_USE_NEON
        const int32_t sum = vm_sign ?
            turbovec_exact4_sum(exact4_w.data(), pd.levels, pd.zero, sign_block, low_row, nsg, v % kBlock) :
            exact4_sum_blocked(exact4_w.data(), pd.levels, sign_block, low_row, nsg, v % kBlock);
#else
        const int32_t sum = turbovec_exact4_sum(exact4_w.data(), pd.levels, pd.zero, sign_block, low_row, nsg, v % kBlock);
#endif
        return std::fma(static_cast<float>(sum), pd.scale, pd.bias) * vscales[v];
    };
    auto exact2_score = [&](size_t v) {
        return exact2_row_score(
            exact_lut.data(), exact_scale, exact_bias, idx.turbovec_data.data() + v * row_bytes, nsg, vscales[v],
            exact2_fma);
    };
    auto plain_score = [&](size_t v, float s) {
        return centre != nullptr ? s - kSignCentre * vscales[v] * (*centre)[v] : s;
    };

    // Stage one: the shortlist, the top s_len allowed rows by sign score
    // under the strict (score, slot) order, so which rows it holds does not
    // depend on the order they are collected in. A lone query over enough
    // blocks splits them into ranges, as the full scan does: each range
    // collects its own top s_len, and the global top s_len rows are each in
    // their range's, so the best s_len of all ranges together is the
    // single-threaded shortlist.
    const size_t n_blocks = (n + kBlock - 1) / kBlock;
    std::vector<uint8_t> scan_blocks;
    size_t n_scan_blocks = n_blocks;
    if (allowed_mask != nullptr) {
        // A block without an allowed row adds nothing to the shortlist.
        scan_blocks.assign(n_blocks, 0);
        for (size_t v = 0; v < n; ++v) {
            scan_blocks[v / kBlock] |= allowed_mask[v] != 0 ? 1 : 0;
        }
        n_scan_blocks = static_cast<size_t>(std::count(scan_blocks.begin(), scan_blocks.end(), uint8_t{ 1 }));
    }
    const int n_threads = std::max(1, opts.n_threads);
    const bool split = n_threads > 1 && n_scan_blocks >= kTurboVecParallelMinBlocks;
    std::vector<Candidate> list;
    if (opts.shortlist != nullptr) {
        // The caller scanned the sign region for this query.
        build_low();
        list.reserve(opts.shortlist->size());
        for (const ScoreId & c : *opts.shortlist) {
            const size_t v = static_cast<size_t>(c.id);
            list.push_back({ v, plain_score(v, static_cast<float>(c.score)) });
        }
    } else if (s_len >= n_allowed || !split) {
        build_low();
    }
    if (opts.shortlist != nullptr) {
    } else if (s_len >= n_allowed) {
        for (size_t v = 0; v < n; ++v) {
            if (allowed_mask == nullptr || allowed_mask[v] != 0) {
                list.push_back({ v, 0.0f });
            }
        }
    } else {
        // Upstream seeds an unmasked scan's collector with the r-th best
        // score of a 48-block sample, so it admits a few hundred rows in
        // all instead of ratcheting a top s_len, and rescans a query that
        // comes back with fewer than s_len rows above the seed. With s_len
        // rows above it the rows kept are exactly the unseeded top s_len.
        const bool seeded = !opts.unseeded && (arm_rules || vm_sign) && allowed_mask == nullptr && 2 * s_len < n_allowed &&
            n / kBlock >= kSeedMinBlocks;
        float seed = -std::numeric_limits<float>::infinity();
        if (seeded) {
            seed = sample_seed(idx, sign_lut, s_len, bits);
        }
        // Shared by the ranges of a split query (see Collector).
        std::atomic<double> floor_score{ -std::numeric_limits<double>::infinity() };
        auto scan = [&](size_t block_begin, size_t block_end, Collector & col) {
            // A short range never needs the full 2 * s_len.
            col.start(s_len, seed, std::min(2 * s_len, (block_end - block_begin) * kBlock + 1),
                split ? &floor_score : nullptr);
            if (vm_sign && scan_blocks.empty()) {
                // The `vpermb` scan over several blocks per call, so a lone
                // query walks them in pairs (upstream
                // search_single_query_vnni_blk2); the scores are the
                // one-block call's.
                constexpr size_t kGroup = 8;
                float group_scores[kGroup * kBlock];
                float * out = group_scores;
                const uint8_t * lut = sign_lut.values.data();
                const uint8_t * split_lut = sign_lut.split.empty() ? nullptr : sign_lut.split.data();
#if defined(GGML_VEC_INDEX_HAVE_VNNI_KERNEL)
                if (vnni_fused_scan_ready(sign_lut)) {
                    // The scores finished and tested in the kernel's
                    // registers (upstream search_single_query_vnni_blk2).
                    count_vnni_fused_scan();
                    col.scan_vnni(sign_lut, idx.turbovec_sign_blocked.data(), vscales, block_begin, block_end, n, nsg);
                    col.finish();
                    return;
                }
#endif
                for (size_t block = block_begin; block < block_end; block += kGroup) {
                    const size_t n_group = std::min(kGroup, block_end - block);
                    score_turbovec_lut_blocks_vnni(&lut, split_lut != nullptr ? &split_lut : nullptr, &sign_lut.scale,
                        &sign_lut.bias, 1, idx.turbovec_sign_blocked.data(), vscales, block, n_group, n, 2, dim / 2,
                        &out);
                    for (size_t b = 0; b < n_group; ++b) {
                        col.add_block(group_scores + b * kBlock, block + b, std::min(kBlock, n - (block + b) * kBlock),
                            allowed_mask);
                    }
                }
                col.finish();
                return;
            }
            float block_scores[kBlock];
            for (size_t block = block_begin; block < block_end; ++block) {
                if (!scan_blocks.empty() && scan_blocks[block] == 0) {
                    continue;
                }
                score_sign_block(idx, sign_lut, idx.turbovec_sign_blocked.data(), block, n, block_scores);
                col.add_block(block_scores, block, std::min(kBlock, n - block * kBlock), allowed_mask);
            }
            col.finish();
        };
        std::vector<ScoreId> top;
        bool seed_short = false;
        // A seeded scan that comes back short runs once more unseeded; the
        // later tables are built on the first pass only.
        for (int pass = 0; pass < 2; ++pass) {
            size_t above_seed = 0;
            top.clear();
            floor_score.store(-std::numeric_limits<double>::infinity(), std::memory_order_relaxed);
            if (split) {
                const size_t n_ranges = static_cast<size_t>(n_threads) * kRangesPerThread;
                const size_t per_range = (n_blocks + n_ranges - 1) / n_ranges;
                std::vector<Collector> ranges(n_ranges);
                // Two more tasks build the query's later tables meanwhile.
                parallel_for(n_ranges + 2, n_threads, [&](size_t range, size_t) {
                    if (range == n_ranges) {
                        if (pass == 0) {
                            build_low();
                        }
                        return;
                    }
                    if (range == n_ranges + 1) {
                        if (pass == 0) {
                            build_exact();
                        }
                        return;
                    }
                    const size_t begin = range * per_range;
                    const size_t end = std::min(n_blocks, begin + per_range);
                    if (begin < end) {
                        scan(begin, end, ranges[range]);
                    }
                });
                // A row below the final floor is below some range's s_len rows.
                const double floor = floor_score.load(std::memory_order_relaxed);
                size_t total = 0;
                for (const Collector & range : ranges) {
                    above_seed += range.above_seed;
                    total += range.top.size();
                }
                top.reserve(total);
                for (const Collector & range : ranges) {
                    for (const ScoreId & c : range.top) {
                        if (c.score >= floor) {
                            top.push_back(c);
                        }
                    }
                }
            } else {
                Collector col;
                scan(0, n_blocks, col);
                top.swap(col.top);
                above_seed = col.above_seed;
            }
            if (!seeded || seed == -std::numeric_limits<float>::infinity() || above_seed >= s_len) {
                break;
            }
            seed_short = true;
            count_seed_short();
            seed = -std::numeric_limits<float>::infinity();
        }
        if (top.size() > s_len) {
            std::vector<ScoreId> scratch;
            cut_to_best(top, s_len, ScoreIdBetterI{},scratch);
        }
        list.reserve(top.size());
        for (const ScoreId & c : top) {
            const size_t v = static_cast<size_t>(c.id);
            list.push_back({ v, plain_score(v, static_cast<float>(c.score)) });
        }
        // Reported so a batch's lone short query is rescanned with
        // single-query tables; those differ only under the aarch64 rules
        // (upstream search.rs `search`: `deferred = aarch64 && short_q.len()
        // == 1`), so under the x86 rules the unseeded pass above is final.
        if (seeded && arm_rules && opts.seed_short != nullptr) {
            *opts.seed_short = seed_short;
        }
    }
    // Runs fn(i) for every candidate i in [0, count): each candidate's
    // arithmetic is its own, so a split query spreads them over threads.
    auto for_each_candidate = [&](size_t count, auto && fn) {
        if (!split) {
            for (size_t i = 0; i < count; ++i) {
                fn(i);
            }
            return;
        }
        parallel_for((count + kRankPerTask - 1) / kRankPerTask, n_threads, [&](size_t task, size_t) {
            const size_t end = std::min(count, (task + 1) * kRankPerTask);
            for (size_t i = task * kRankPerTask; i < end; ++i) {
                fn(i);
            }
        });
    };

    // Stages two and three: rank by estimates from the lower planes and
    // keep the best t_len (upstream rank_first_keep / rank_full_known).
    if (refines && list.size() > t_len) {
        const float a_over_m = stats.alpha / m;
        const float b_over_m[3] = { stats.beta[0] / m, stats.beta[1] / m, stats.beta[2] / m };
        const float a1_over_m = stats.alpha1 / m;
        const float b1_over_m = stats.beta1 / m;
        // m * sum(+-q) over one low plane for the candidates in [begin, end),
        // at most kRankBatch of them (upstream plane_terms): the sign tables
        // over their rows, undone to m * sum(+-q), or the bit masks counted.
        auto plane_terms = [&](size_t begin, size_t end, size_t plane, float * out) {
            // The rows are scattered over the index, so the lines of the
            // batch kRankAhead steps on are asked for while this one is
            // counted; asked for just before the read, each row still waited
            // on its miss. Upstream (rank_first_keep / rank_full_known) goes
            // one step ahead; two measured 9 us faster at q4 k64 on Zen 5
            // and the same on Apple M5. The first batches are asked for here
            // as well.
            const size_t span = end - begin;
            const size_t ahead_end = std::min(list.size(), end + kRankAhead * span);
            for (size_t i = begin == 0 ? begin : end + (kRankAhead - 1) * span; i < ahead_end; ++i) {
                prefetch_span(idx.turbovec_data.data() + list[i].slot * row_bytes + plane * nsg, nsg);
            }
            if (table_rank) {
#if GGML_VEC_INDEX_USE_NEON
                const uint8_t * rows[kRankBatch] = {};
                for (size_t i = begin; i < end; ++i) {
                    rows[i - begin] = idx.turbovec_data.data() + list[i].slot * row_bytes + plane * nsg;
                }
                uint32_t sums[kRankBatch];
                table_sums(rank_lut.values.data(), rows, end - begin, nsg, sums);
                const float b = rank_lut.bias - bias_corr;
                for (size_t i = begin; i < end; ++i) {
                    out[i - begin] = rank_lut.scale * static_cast<float>(sums[i - begin]) + b;
                }
#endif
                return;
            }
            for (size_t i = begin; i < end; ++i) {
                const uint8_t * low = idx.turbovec_data.data() + list[i].slot * row_bytes + plane * nsg;
                out[i - begin] = low_planes.unit * static_cast<float>(2 * low_dot(low_planes, low) - low_planes.sum_w);
            }
        };
        // Runs fn(begin, end) over the list in steps; a split query spreads
        // the steps over threads. The table kernel wants its kRankBatch lanes
        // full; the mask count takes a row at a time, and a short step keeps
        // the prefetch a few rows ahead instead of a burst of kRankBatch rows
        // (upstream rank_step).
        const size_t step = table_rank ? kRankBatch : 8;
        auto for_each_batch = [&](auto && fn) {
            auto run = [&](size_t from, size_t to) {
                for (size_t b = from; b < to; b += step) {
                    fn(b, std::min(to, b + step));
                }
            };
            const size_t count = list.size();
            if (!split) {
                run(0, count);
                return;
            }
            parallel_for((count + kRankPerTask - 1) / kRankPerTask, n_threads, [&](size_t task, size_t) {
                run(task * kRankPerTask, std::min(count, (task + 1) * kRankPerTask));
            });
        };
        const size_t top_plane = n_low - 1;
        std::vector<float> tops(list.size());
        for_each_batch([&](size_t begin, size_t end) {
            plane_terms(begin, end, top_plane, tops.data() + begin);
            for (size_t i = begin; i < end; ++i) {
                Candidate & c = list[i];
                const float vscale = vscales[c.slot];
                if (vscale == 0.0f) {
                    c.score = 0.0f;
                } else {
                    const float m_s = c.score / vscale - bias_corr;
                    c.score = vscale * (a1_over_m * m_s + b1_over_m * tops[i] + bias_corr);
                }
            }
        });
        if (n_low > 1) {
            // keep_best's choice follows the strict (score, slot) order, so
            // it does not depend on the order the shortlist arrived in.
            keep_best(list, mid_len(k_sz), &tops);
            for_each_batch([&](size_t begin, size_t end) {
                float l[3][kRankBatch];
                for (size_t j = 0; j + 1 < n_low; ++j) {
                    plane_terms(begin, end, j, l[j]);
                }
                for (size_t i = begin; i < end; ++i) {
                    Candidate & c = list[i];
                    const float vscale = vscales[c.slot];
                    if (vscale == 0.0f) {
                        c.score = 0.0f;
                        continue;
                    }
                    l[n_low - 1][i - begin] = tops[i];
                    const float m_s = ((c.score / vscale - bias_corr) - b1_over_m * tops[i]) / a1_over_m;
                    float est = a_over_m * m_s;
                    for (size_t j = 0; j < n_low; ++j) {
                        est += b_over_m[j] * l[j][i - begin];
                    }
                    c.score = vscale * (est + bias_corr);
                }
            });
        }
        keep_best(list, t_len);
    }

    // Stage four: the exact score for what is left, row by row from the
    // planes where the arithmetic is known, else from the full scan's own
    // block kernel; either way a returned score equals the full scan's.
    if (!exact_built) {
        build_exact();
    }
    std::sort(list.begin(), list.end(), [](const Candidate & a, const Candidate & b) { return a.slot < b.slot; });
    if (exact4 || exact2) {
        std::vector<float> exact(list.size());
        // The rows are scattered, so each is fetched a few candidates ahead
        // (upstream rerank_legacy AHEAD).
        constexpr size_t kAhead = 4;
        for_each_candidate(list.size(), [&](size_t i) {
            if (exact2 && i + kAhead < list.size()) {
                prefetch_span(idx.turbovec_data.data() + list[i + kAhead].slot * row_bytes, row_bytes);
            }
            exact[i] = exact4 ? exact4_score(list[i].slot) : exact2_score(list[i].slot);
        });
        for (size_t i = 0; i < list.size(); ++i) {
            update_topk_heap(heap, k, { exact[i], topk_key(idx, list[i].slot) });
        }
        return true;
    }
    const uint8_t * split_ptr = exact_split.empty() ? nullptr : exact_split.data();
    if (!split) {
        float block_scores[kBlock];
        size_t scored_block = SIZE_MAX;
        for (const Candidate & c : list) {
            const size_t block = c.slot / kBlock;
            if (block != scored_block) {
                score_turbovec_block(
                    idx, exact_lut.data(), exact_scale, exact_bias, permute_dot ? &pd : nullptr, block, block_scores,
                    split_ptr);
                scored_block = block;
            }
            update_topk_heap(heap, k, { block_scores[c.slot % kBlock], topk_key(idx, c.slot) });
        }
        return true;
    }
    // A split query scores the candidates' blocks across threads, each block
    // once, then fills the heap in slot order as the loop above does.
    std::vector<size_t> block_starts;
    for (size_t i = 0; i < list.size(); ++i) {
        if (i == 0 || list[i].slot / kBlock != list[i - 1].slot / kBlock) {
            block_starts.push_back(i);
        }
    }
    block_starts.push_back(list.size());
    const size_t n_cand_blocks = block_starts.size() - 1;
    std::vector<float> exact(list.size());
    constexpr size_t kBlocksPerTask = 4;
    parallel_for((n_cand_blocks + kBlocksPerTask - 1) / kBlocksPerTask, n_threads, [&](size_t task, size_t) {
        float block_scores[kBlock];
        const size_t end = std::min(n_cand_blocks, (task + 1) * kBlocksPerTask);
        for (size_t b = task * kBlocksPerTask; b < end; ++b) {
            score_turbovec_block(
                idx, exact_lut.data(), exact_scale, exact_bias, permute_dot ? &pd : nullptr,
                list[block_starts[b]].slot / kBlock, block_scores, split_ptr);
            for (size_t i = block_starts[b]; i < block_starts[b + 1]; ++i) {
                exact[i] = block_scores[list[i].slot % kBlock];
            }
        }
    });
    for (size_t i = 0; i < list.size(); ++i) {
        update_topk_heap(heap, k, { exact[i], topk_key(idx, list[i].slot) });
    }
    return true;
}

bool turbovec_staged_search_batch(
        const ggml_vec_index & idx,
        const float * rotated_queries,
        int n_q,
        int k,
        int n_threads,
        std::vector<std::vector<ScoreId>> & heaps) {
    if (!is_turbovec(idx) || idx.turbovec_legacy || !idx.turbovec_staged_enabled) {
        return false;
    }
    // Upstream scans a batch's sign region several queries at a time, a
    // seeded collector per query: under its aarch64 rules four with the
    // fused LUT kernel over the lane-blocked region (one code load and
    // nibble split for all four), under its x86 rules six with the
    // `vpermb` kernel over the vector-major region (search.rs VNNI_BATCH,
    // where that kernel's cost per query is lowest; the tail takes its own
    // width, a lone one two blocks a call).
    constexpr int kQuad = 4;
    constexpr int kVnniBatch = 6;
    constexpr int kBatchMax = 6;
    static_assert(kQuad <= kBatchMax && kVnniBatch <= kBatchMax, "the per-task arrays hold a batch");
    const bool vm_sign = turbovec_sign_layout(idx) == kTurboVecLayoutVm;
#if GGML_VEC_INDEX_USE_NEON
    const bool quad_scan = !idx.turbovec_x86 && !vm_sign;
#else
    // The aarch64 rules without their NEON kernel (forced in a test on an
    // x86 host) run query by query.
    const bool quad_scan = false;
#endif
    if (!quad_scan && !vm_sign) {
        return false;
    }
    // The quad kernel wants four queries; the `vpermb` scan has a form for
    // every width, so any batch takes it.
    if (n_q < (quad_scan ? kQuad : 2)) {
        return false;
    }
    const size_t width = static_cast<size_t>(quad_scan ? kQuad : kVnniBatch);
    TurboVecPlanesStats stats;
    const std::vector<float> * centre = nullptr;
    if (!planes_ready(idx, stats, centre)) {
        return false;
    }
    const int bits = idx.turbovec_bits;
    const int dim = idx.dim;
    const size_t dim_sz = static_cast<size_t>(dim);
    const size_t n = idx.slot_to_id.size();
    const size_t n_blocks = (n + kBlock - 1) / kBlock;
    const size_t s_len = shortlist_len(static_cast<size_t>(k), bits);
    if (s_len >= n) {
        // Every row is rescored; nothing to shortlist.
        return false;
    }
    n_threads = std::max(1, n_threads);
    const size_t n_tasks = (static_cast<size_t>(n_q) + width - 1) / width;
    if (n_threads > 1 && n_tasks < static_cast<size_t>(n_threads)) {
        // Fewer batches than threads would leave threads idle; query by
        // query each query splits its blocks instead. (Upstream tiles the
        // block axis as well.)
        return false;
    }
    const float m = stats.m;
    const float * vscales = idx.turbovec_scale.data();
    const bool calibrated = !idx.turbovec_tqplus_shift.empty();
    // Both kernels implement the collector (upstream planes_buffered_supported
    // holds for either), so a large enough index seeds every query.
    const bool seeded = 2 * s_len < n && n / kBlock >= kSeedMinBlocks;
    const size_t n_q_sz = static_cast<size_t>(n_q);

    // Each query's calibrated query and TQ+ bias, as search_one computes
    // them, and its sign-stage query pulled toward the centre.
    std::vector<float> score_queries(n_q_sz * dim_sz);
    std::vector<float> biases(n_q_sz, 0.0f);
    for (size_t q = 0; q < n_q_sz; ++q) {
        const float * rq = rotated_queries + q * dim_sz;
        float * sq = score_queries.data() + q * dim_sz;
        if (calibrated) {
            double bias_correction = 0.0;
            for (size_t d = 0; d < dim_sz; ++d) {
                sq[d] = rq[d] / idx.turbovec_tqplus_scale[d];
                bias_correction -= static_cast<double>(rq[d]) * static_cast<double>(idx.turbovec_tqplus_shift[d]);
            }
            biases[q] = static_cast<float>(bias_correction);
        } else {
            std::copy(rq, rq + dim_sz, sq);
        }
    }
    heaps.assign(n_q_sz, {});
    auto run_query = [&](size_t q, const std::vector<ScoreId> * shortlist, bool sign_single, bool rescan,
                         int query_threads) {
        TurboVecStagedOpts o;
        o.single_query = false;
        o.sign_single = sign_single;
        o.n_threads = query_threads;
        o.shortlist = shortlist;
        o.unseeded = rescan;
        turbovec_staged_search(idx, score_queries.data() + q * dim_sz, biases[q], k, nullptr, o, heaps[q]);
    };

    // The scan, a batch per task. A quad path's short tail scans query by
    // query with the same tables so its queries count toward the batch's
    // short ones; the `vpermb` kernel takes a tail at its own width.
    std::vector<uint8_t> seed_short(n_q_sz, 0);
    parallel_for(n_tasks, n_threads, [&](size_t task, size_t) {
        count_batch_scan();
        const size_t q0 = task * width;
        const size_t n_in = std::min(width, n_q_sz - q0);
        SignLut luts[kBatchMax];
        const uint8_t * lut_ptrs[kBatchMax] = {};
        const uint8_t * split_ptrs[kBatchMax] = {};
        float lut_scales[kBatchMax] = {};
        float lut_biases[kBatchMax] = {};
        Collector cols[kBatchMax];
        std::vector<float> q_sign(dim_sz);
        for (size_t i = 0; i < n_in; ++i) {
            const size_t q = q0 + i;
            const float * sq = score_queries.data() + q * dim_sz;
            for (size_t d = 0; d < dim_sz; ++d) {
                q_sign[d] = sq[d];
                if (calibrated) {
                    q_sign[d] += kSignCentre * idx.turbovec_tqplus_shift[d] / idx.turbovec_tqplus_scale[d];
                }
            }
            luts[i] = build_sign_lut(q_sign.data(), m, dim, false);
            luts[i].bias += biases[q];
            split_sign_lut(idx, luts[i]);
            lut_ptrs[i] = luts[i].values.data();
            split_ptrs[i] = luts[i].split.empty() ? nullptr : luts[i].split.data();
            lut_scales[i] = luts[i].scale;
            lut_biases[i] = luts[i].bias;
            const float seed = seeded ? sample_seed(idx, luts[i], s_len, bits) : -std::numeric_limits<float>::infinity();
            cols[i].start(s_len, seed, 2 * s_len, nullptr);
        }
        if (quad_scan) {
#if GGML_VEC_INDEX_USE_NEON
            float block_scores[kQuad][kBlock];
            float * const outs[kQuad] = { block_scores[0], block_scores[1], block_scores[2], block_scores[3] };
            for (size_t block = 0; block < n_blocks; ++block) {
                const size_t count = std::min(kBlock, n - block * kBlock);
                if (n_in == kQuad) {
                    score_turbovec_lut_block_4(
                        lut_ptrs, lut_scales, lut_biases, idx.turbovec_sign_blocked.data(), vscales, block, n, 2,
                        dim / 2, outs);
                } else {
                    for (size_t i = 0; i < n_in; ++i) {
                        score_sign_block(idx, luts[i], idx.turbovec_sign_blocked.data(), block, n, block_scores[i]);
                    }
                }
                for (size_t i = 0; i < n_in; ++i) {
                    cols[i].add_block(block_scores[i], block, count, nullptr);
                }
            }
#endif
        } else {
#if defined(GGML_VEC_INDEX_HAVE_VNNI_KERNEL)
            if (vnni_fused_batch_ready(split_ptrs, n_in)) {
                // Every block through the batch kernel, each query's scores
                // finished and tested in its registers (upstream
                // search_multi_query_vnni and its epilogue).
                count_vnni_fused_batch();
                ggml_vec_index_detail::TurboVecVnniScanState states[kBatchMax];
                void * ctxs[kBatchMax] = {};
                for (size_t i = 0; i < n_in; ++i) {
                    states[i] = cols[i].vnni_state();
                    ctxs[i] = &cols[i];
                }
                ggml_vec_index_detail::turbovec_vnni_lut_scan_batch(split_ptrs, lut_scales, lut_biases,
                    static_cast<int>(n_in), vscales, idx.turbovec_sign_blocked.data(), 0, n_blocks, n, dim_sz / 8,
                    states, &Collector::vnni_sink, ctxs);
                for (size_t i = 0; i < n_in; ++i) {
                    cols[i].above_seed = states[i].above_seed;
                }
            } else
#endif
            {
                // The kernel's scores block by block through the Collector
                // (a build or host without the fused kernel). Blocks per
                // `vpermb` call: enough that a lone tail query's kernel walks
                // its block pairs and the per-call setup is shared.
                constexpr size_t kVnniGroup = 8;
                std::vector<float> scores(n_in * kVnniGroup * kBlock);
                float * outs[kBatchMax] = {};
                for (size_t i = 0; i < n_in; ++i) {
                    outs[i] = scores.data() + i * kVnniGroup * kBlock;
                }
                for (size_t block = 0; block < n_blocks; block += kVnniGroup) {
                    const size_t n_group = std::min(kVnniGroup, n_blocks - block);
                    score_turbovec_lut_blocks_vnni(
                        lut_ptrs, split_ptrs, lut_scales, lut_biases, static_cast<int>(n_in),
                        idx.turbovec_sign_blocked.data(), vscales, block, n_group, n, 2, dim / 2, outs);
                    for (size_t b = 0; b < n_group; ++b) {
                        const size_t count = std::min(kBlock, n - (block + b) * kBlock);
                        for (size_t i = 0; i < n_in; ++i) {
                            cols[i].add_block(outs[i] + b * kBlock, block + b, count, nullptr);
                        }
                    }
                }
            }
        }
        for (size_t i = 0; i < n_in; ++i) {
            const size_t q = q0 + i;
            if (seeded && cols[i].above_seed < s_len) {
                seed_short[q] = 1;
                count_seed_short();
                continue;
            }
            std::vector<ScoreId> & top = cols[i].top;
            if (top.size() > s_len) {
                cut_to_best(top, s_len, ScoreIdBetterI{},cols[i].scratch);
            }
            run_query(q, &top, false, false, 1);
        }
    });

    // Upstream rescans a batch's short queries unseeded: one alone through
    // the single-query scan and, under the aarch64 rules, its narrower
    // tables, several with the batch tables. A lone one may split its
    // blocks over every thread, as a lone search does.
    std::vector<size_t> short_queries;
    for (size_t q = 0; q < n_q_sz; ++q) {
        if (seed_short[q] != 0) {
            short_queries.push_back(q);
        }
    }
    const bool lone_short = short_queries.size() == 1;
    parallel_for(short_queries.size(), n_threads, [&](size_t i, size_t) {
        run_query(short_queries[i], nullptr, lone_short, true, lone_short ? n_threads : 1);
    });
    return true;
}

#ifdef GGML_VEC_INDEX_TEST_HOOKS
int turbovec_low_kernel_parity_for_test(int kernel) {
#if defined(GGML_VEC_INDEX_HAVE_AVX512BW_KERNEL) || defined(GGML_VEC_INDEX_HAVE_VPOPCNT_KERNEL)
    const VecIndexCpuFeatures & f = vec_index_cpu_features();
    bool available = false;
#if defined(GGML_VEC_INDEX_HAVE_AVX512BW_KERNEL)
    available = available || (kernel == 0 && f.avx512bw && f.avx2);
#endif
#if defined(GGML_VEC_INDEX_HAVE_VPOPCNT_KERNEL)
    available = available || (kernel == 1 && f.avx512bw && f.avx512vpopcntdq);
#endif
    if (!available) {
        return -1;
    }
    uint32_t state = 0x6a09e667u + static_cast<uint32_t>(kernel);
    auto next = [&]() {
        state = state * 1664525u + 1013904223u;
        return state >> 8;
    };
    bool ok = true;
    // A row shorter than one mask group, one across groups with a short
    // tail, and whole groups.
    for (const int dim : { 16, 128, 784, 1536 }) {
        std::vector<float> q(static_cast<size_t>(dim));
        for (int trial = 0; trial < 6 && ok; ++trial) {
            for (float & x : q) {
                x = static_cast<float>(static_cast<int>(next() % 20001) - 10000) / 4093.0f;
            }
            if (trial == 1) {
                // Ties at the rounding boundary and zero weights.
                for (size_t i = 0; i < q.size(); i += 3) {
                    q[i] = 0.0f;
                }
            }
            g_low_kernels_enabled.store(kernel != 0, std::memory_order_relaxed);
            const LowPlanes scalar = build_low_planes(q.data(), 1.5f, dim);
            g_low_kernels_enabled.store(true, std::memory_order_relaxed);
            const LowPlanes simd = build_low_planes(q.data(), 1.5f, dim);
            if (kernel == 0) {
                ok = scalar.masks == simd.masks && scalar.sum_w == simd.sum_w && scalar.unit == simd.unit;
                continue;
            }
            std::vector<uint8_t> row(static_cast<size_t>(dim) / 8);
            for (uint8_t & b : row) {
                b = static_cast<uint8_t>(next());
            }
            const int64_t got = low_dot(simd, row.data());
            g_low_kernels_enabled.store(false, std::memory_order_relaxed);
            const int64_t want = low_dot(simd, row.data());
            g_low_kernels_enabled.store(true, std::memory_order_relaxed);
            ok = got == want;
        }
    }
    return ok ? 1 : 0;
#else
    (void) kernel;
    return -1;
#endif
}

int turbovec_exact2_row_parity_for_test(const ggml_vec_index_t * idx, const float * query) {
    std::shared_lock<std::shared_mutex> lock(idx->mutex);
    if (!is_turbovec(*idx) || idx->turbovec_legacy || !exact2_rows(*idx)) {
        return -1;
    }
    TurboVecPlanesStats stats;
    const std::vector<float> * centre = nullptr;
    if (!planes_ready(*idx, stats, centre)) {
        return -1;
    }
    const int dim = idx->dim;
    const size_t nsg = static_cast<size_t>(dim) / 8;
    const size_t n = idx->slot_to_id.size();
    // The query as the search scores it: rotated, then calibrated.
    std::vector<float> q(static_cast<size_t>(dim));
    rotate_turbovec_query(false, query, q.data(), dim);
    float bias_corr = 0.0f;
    if (!idx->turbovec_tqplus_shift.empty()) {
        double b = 0.0;
        for (size_t i = 0; i < q.size(); ++i) {
            b -= static_cast<double>(q[i]) * static_cast<double>(idx->turbovec_tqplus_shift[i]);
            q[i] /= idx->turbovec_tqplus_scale[i];
        }
        bias_corr = static_cast<float>(b);
    }
    std::vector<uint8_t> lut;
    float scale = 1.0f;
    float bias = 0.0f;
    build_turbovec_lut(2, false, q.data(), dim, lut, scale, bias);
    bias += bias_corr;
    std::vector<uint8_t> split;
    if (turbovec_scores_vnni_lut(*idx)) {
        split.resize(lut.size());
        turbovec_split_lut_for_vnni(lut.data(), lut.size() / 32, split.data());
    }
    const bool fma_flush = !turbovec_scores_vnni_lut(*idx);
    const size_t row_bytes = turbovec_row_bytes(2, static_cast<size_t>(dim));
    float block_scores[kBlock];
    for (size_t block = 0; block * kBlock < n; ++block) {
        score_turbovec_block(*idx, lut.data(), scale, bias, nullptr, block, block_scores, split.empty() ? nullptr : split.data());
        for (size_t lane = 0; lane < std::min(kBlock, n - block * kBlock); ++lane) {
            const size_t v = block * kBlock + lane;
            const float got = exact2_row_score(
                lut.data(), scale, bias, idx->turbovec_data.data() + v * row_bytes, nsg, idx->turbovec_scale[v], fma_flush);
            if (std::memcmp(&got, &block_scores[lane], sizeof(float)) != 0) {
                return 0;
            }
        }
    }
    return 1;
}

int32_t turbovec_low_planes_for_test(const float * q, int dim, float m, float * unit, std::vector<uint64_t> * masks) {
    const LowPlanes planes = build_low_planes(q, m, dim);
    *unit = planes.unit;
    *masks = planes.masks;
    return planes.sum_w;
}

int turbovec_rounding_parity_for_test() {
    // Ties both ways, the largest float below a half, the 2^23 edge where a
    // float is its own rounding, values past the table's max, NaN and inf.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const float edges[] = { 0.0f, -0.0f, 0.5f, -0.5f, 1.5f, -1.5f, 2.5f, -2.5f, 0.49999997f, -0.49999997f,
        14.5f, 15.5f, 254.5f, 255.5f, 256.0f, 8388607.5f, -8388607.5f, 8388608.0f, -8388608.0f, 8388609.0f,
        1e30f, -1e30f, nan, inf, -inf };
    int case_no = 0;
    for (const float x : edges) {
        ++case_no;
        const float want = std::round(x);
        const float got = round_half_away_f32(x);
        if (!(got == want || (std::isnan(got) && std::isnan(want)))) {
            return case_no;
        }
    }
    // Each table entry is (in - mn) * inv_scale; mn 0 and inv_scale 1 put
    // the edges above in exactly. NaN and inf are left out: a table entry
    // is always finite.
    std::vector<float> in;
    for (const float x : edges) {
        if (std::isfinite(x)) {
            in.push_back(x);
        }
    }
    while (in.size() % 16 != 0) {
        in.push_back(in[in.size() % 7]);
    }
    for (const float max_lut : { 15.0f, 255.0f }) {
        for (size_t g = 0; g < in.size(); g += 16) {
            uint8_t out[16];
            quantise_table16(in.data() + g, 0.0f, 1.0f, max_lut, out);
            for (size_t i = 0; i < 16; ++i) {
                ++case_no;
                const float want = std::max(0.0f, std::min(max_lut, std::round(in[g + i])));
                if (static_cast<float>(out[i]) != want) {
                    return case_no;
                }
            }
        }
    }
    return 0;
}

void turbovec_force_planes_for_test(int mode, size_t min_vectors) {
    g_planes_override.store(mode, std::memory_order_relaxed);
    g_planes_min_vectors.store(min_vectors == 0 ? kTurboVecPlanesMinVectors : min_vectors, std::memory_order_relaxed);
}

int64_t turbovec_seed_short_count_for_test() {
    return g_seed_short_count.load(std::memory_order_relaxed);
}

int64_t turbovec_batch_scan_count_for_test() {
    return g_batch_scan_count.load(std::memory_order_relaxed);
}

int turbovec_vnni_fused_scan_parity_for_test(ggml_vec_index_t * idx, const float * queries, int n_q, int k) {
    // Below zero: why the parity could not run (-1 kernel not built, -2 not
    // a staged 1.x index, -3 sign region not vector-major, -4 no dot product,
    // -5 planes cache not ready), so a skip on a host that should run it
    // says what to fix. The host checks run everywhere.
    {
        std::shared_lock<std::shared_mutex> lock(idx->mutex);
        if (!is_turbovec(*idx) || idx->turbovec_legacy || !idx->turbovec_staged_enabled) {
            return -2;
        }
        if (turbovec_sign_layout(*idx) != kTurboVecLayoutVm) {
            return -3;
        }
        if (!turbovec_cpu_dot_product()) {
            return -4;
        }
    }
    // A planes cache that is not ready makes the staged search fall back to
    // the full scan: check before and after one search (a cache built lazily
    // by the first search shows here), and return -5 when it never is.
    for (int attempt = 0; attempt < 2; ++attempt) {
        bool ready = false;
        {
            std::shared_lock<std::shared_mutex> lock(idx->mutex);
            TurboVecPlanesStats stats;
            const std::vector<float> * centre = nullptr;
            ready = planes_ready(*idx, stats, centre);
        }
        if (ready) {
            break;
        }
        if (attempt == 1) {
            return -5;
        }
        std::vector<float> warm_scores(static_cast<size_t>(k));
        std::vector<uint64_t> warm_ids(static_cast<size_t>(k));
        ggml_vec_index_search(idx, queries, 1, k, warm_scores.data(), warm_ids.data());
    }
#if defined(GGML_VEC_INDEX_HAVE_VNNI_KERNEL)
    const size_t dim = static_cast<size_t>(idx->dim);
    const size_t k_sz = static_cast<size_t>(k);
    std::vector<float> scores[2];
    std::vector<uint64_t> ids[2];
    // Each query searched alone, so the lone scan runs; the fused run must
    // have taken the kernel, the other must not have.
    for (int fused = 0; fused < 2; ++fused) {
        scores[fused].assign(static_cast<size_t>(n_q) * k_sz, 0.0f);
        ids[fused].assign(static_cast<size_t>(n_q) * k_sz, 0);
        g_vnni_fused_scan_enabled.store(fused != 0, std::memory_order_relaxed);
        const int64_t before = g_vnni_fused_scan_count.load(std::memory_order_relaxed);
        bool ok = true;
        for (int q = 0; q < n_q && ok; ++q) {
            ok = ggml_vec_index_search(idx, queries + static_cast<size_t>(q) * dim, 1, k,
                     scores[fused].data() + static_cast<size_t>(q) * k_sz,
                     ids[fused].data() + static_cast<size_t>(q) * k_sz) == GGML_VEC_INDEX_OK;
        }
        g_vnni_fused_scan_enabled.store(true, std::memory_order_relaxed);
        const int64_t ran = g_vnni_fused_scan_count.load(std::memory_order_relaxed) - before;
        if (!ok || (fused != 0 ? ran < n_q : ran != 0)) {
            return 0;
        }
    }
    return ids[0] == ids[1] &&
        std::memcmp(scores[0].data(), scores[1].data(), scores[0].size() * sizeof(float)) == 0 ? 1 : 0;
#else
    (void) idx;
    (void) queries;
    (void) n_q;
    (void) k;
    return -1;
#endif
}

int turbovec_vnni_fused_batch_parity_for_test(ggml_vec_index_t * idx, const float * queries, int n_q, int k) {
    // The host checks of the lone-scan parity, then -6 when the queries did
    // not take the staged batch at all (too few for the threads, or no
    // shortlist), so a cell that proves nothing is not reported as passing.
    {
        const int status = turbovec_vnni_fused_scan_parity_for_test(idx, queries, 1, k);
        if (status < 0) {
            return status;
        }
    }
#if defined(GGML_VEC_INDEX_HAVE_VNNI_KERNEL)
    const size_t k_sz = static_cast<size_t>(k);
    std::vector<float> scores[2];
    std::vector<uint64_t> ids[2];
    for (int fused = 0; fused < 2; ++fused) {
        scores[fused].assign(static_cast<size_t>(n_q) * k_sz, 0.0f);
        ids[fused].assign(static_cast<size_t>(n_q) * k_sz, 0);
        g_vnni_fused_batch_enabled.store(fused != 0, std::memory_order_relaxed);
        const int64_t before = g_vnni_fused_batch_count.load(std::memory_order_relaxed);
        const int64_t batches_before = g_batch_scan_count.load(std::memory_order_relaxed);
        const bool ok = ggml_vec_index_search(idx, queries, n_q, k, scores[fused].data(), ids[fused].data()) ==
            GGML_VEC_INDEX_OK;
        g_vnni_fused_batch_enabled.store(true, std::memory_order_relaxed);
        const int64_t ran = g_vnni_fused_batch_count.load(std::memory_order_relaxed) - before;
        const int64_t batches = g_batch_scan_count.load(std::memory_order_relaxed) - batches_before;
        if (!ok) {
            return 0;
        }
        if (batches == 0) {
            return -6;
        }
        // Every task of the fused run must have taken the kernel.
        if (fused != 0 ? ran != batches : ran != 0) {
            return 0;
        }
    }
    return ids[0] == ids[1] &&
        std::memcmp(scores[0].data(), scores[1].data(), scores[0].size() * sizeof(float)) == 0 ? 1 : 0;
#else
    (void) n_q;
    return -1;
#endif
}
#endif
