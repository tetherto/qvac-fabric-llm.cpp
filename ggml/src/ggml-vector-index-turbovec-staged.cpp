// ggml-vector-index-turbovec-staged.cpp - staged search over the sign planes
// of a TurboVec 1.x index (upstream turbovec 1.1.2 "planes" search).
//
// A search scans the sign planes alone for a shortlist, ranks the shortlist
// with estimates that add the lower planes, and rescores the best few with
// the full scan's own arithmetic, so returned scores are exact while the
// candidate set is approximate. This follows upstream's single-threaded
// path (search.rs `search` with a `PlanesRef`, `rerank_legacy`, `Refine`).
// Upstream's seeded collectors only speed the shortlist up, except that the
// one query of a batch whose seed runs short is rescanned with single-query
// tables; the port computes the seed to reproduce that. Upstream's in-scan
// refinement on a thread pool, which can return other rows than its
// single-threaded path, is not reproduced: a lone query here splits the
// shortlist scan, the ranking and the rescore across threads and still
// returns the single-threaded results.

#include "ggml-vector-index-impl.h"

#include <bitset>

#if defined(GGML_VEC_INDEX_HAVE_AVX512BW_KERNEL) || defined(GGML_VEC_INDEX_HAVE_VPOPCNT_KERNEL)
#include "ggml-vector-index-x86.h"
#endif

#if GGML_VEC_INDEX_USE_NEON
#include <arm_neon.h>
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

// Set bits of a word; MSVC has no __builtin_popcount.
inline int popcount64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcountll(x);
#else
    return static_cast<int>(std::bitset<64>(x).count());
#endif
}

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
    for (size_t i = 0; i < out.values.size(); ++i) {
        const float r = std::max(0.0f, std::min(max_lut, std::round((values[i] - mins[i / 16]) * inv_scale)));
        out.values[i] = static_cast<uint8_t>(r);
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

// One block of sign scores, with the kernel upstream scans the sign region
// with on this index's host: the `vpermb` scan over a vector-major region
// (x86), the nibble LUT scan otherwise.
void score_sign_block(const ggml_vec_index & idx, const SignLut & lut, const uint8_t * sign, size_t block, size_t n,
        float * out) {
    const float * vscales = idx.turbovec_scale.data();
    if (turbovec_sign_layout(idx) == kTurboVecLayoutVm) {
        score_turbovec_lut_block_vnni(lut.values.data(), lut.split.empty() ? nullptr : lut.split.data(), lut.scale,
            lut.bias, sign, vscales, block, n, 2, idx.dim / 2, out);
        return;
    }
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
    bool built = false;
#if defined(GGML_VEC_INDEX_HAVE_AVX512BW_KERNEL)
    // Upstream's AVX-512 build (sixteen coordinates a step), where dim
    // allows it; the same weights as the loop below. Its conversions are
    // not saturating, so an overflowing query takes the loop.
    const VecIndexCpuFeatures & features = vec_index_cpu_features();
    if (!overflow && dim % 16 == 0 && features.avx512bw && features.avx2 &&
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

int64_t low_dot(const LowPlanes & planes, const uint8_t * row) {
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
// blocks (pack.rs planes_sample). What a seed changes is which query of a
// batch is rescanned with single-query tables, and only aarch64's differ
// (their entries are capped at 31), so the port computes it there alone.
#if GGML_VEC_INDEX_USE_NEON
constexpr bool kSeedSupported = true;
#else
constexpr bool kSeedSupported = false;
#endif
constexpr size_t kSeedMinBlocks = 1024;

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
    std::nth_element(scores.begin(), scores.begin() + static_cast<std::ptrdiff_t>(r - 1), scores.end(),
        [](float a, float b) { return a > b; });
    return scores[r - 1];
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

void keep_best(std::vector<Candidate> & cands, size_t keep, std::vector<float> * tops = nullptr) {
    if (cands.size() <= keep) {
        return;
    }
    std::vector<size_t> order(cands.size());
    for (size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::nth_element(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(keep - 1), order.end(),
        [&](size_t a, size_t b) { return candidate_better(cands[a], cands[b]); });
    order.resize(keep);
    std::vector<Candidate> kept(keep);
    std::vector<float> kept_tops;
    for (size_t i = 0; i < keep; ++i) {
        kept[i] = cands[order[i]];
        if (tops != nullptr) {
            kept_tops.push_back((*tops)[order[i]]);
        }
    }
    cands.swap(kept);
    if (tops != nullptr) {
        tops->swap(kept_tops);
    }
}

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
    {
        std::lock_guard<std::mutex> lock(idx.turbovec_planes_mutex);
        // The exact rescore reads the blocked cache; without a valid one
        // the caller's full scan (and its per-row fallback) answers.
        const size_t n_blocks = (n + kBlock - 1) / kBlock;
        const size_t n_byte_groups = static_cast<size_t>(dim) / static_cast<size_t>(8 / bits);
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
        if (calibrated) {
            if (!idx.turbovec_centre_ready) {
                idx.turbovec_centre = compute_centre(idx, stats.m);
                idx.turbovec_centre_ready = true;
            }
            centre = &idx.turbovec_centre;
        }
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
    SignLut sign_lut = build_sign_lut(q_sign.data(), m, dim, arm_rules && (single_query || opts.sign_single));
    sign_lut.bias += bias_corr;
    split_sign_lut(idx, sign_lut);
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
    const SignLut & rank_lut = plain_lut.values.empty() ? sign_lut : plain_lut;
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
    // Upstream x86 rescores a 4-bit shortlist from the planes themselves
    // (`exact4_sum`), its sign plane read from the vector-major region.
    const bool exact4 = permute_dot && turbovec_sign_layout(idx) == kTurboVecLayoutVm;
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
            if (turbovec_scores_vnni_lut(idx)) {
                exact_split.resize(exact_lut.size());
                turbovec_split_lut_for_vnni(exact_lut.data(), exact_lut.size() / 32, exact_split.data());
            }
        }
        exact_built = true;
    };
    // One candidate's exact 4-bit score from the planes: the permute-dot
    // integer sum and its epilogue, the full scan's score bit for bit.
    auto exact4_score = [&](size_t v) {
        const int32_t sum = turbovec_exact4_sum(exact4_w.data(), pd.levels, pd.zero,
            idx.turbovec_sign_blocked.data() + (v / kBlock) * nsg * kBlock, idx.turbovec_data.data() + v * row_bytes,
            nsg, v % kBlock);
        return std::fma(static_cast<float>(sum), pd.scale, pd.bias) * vscales[v];
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
    if (s_len >= n_allowed || !split) {
        build_low();
    }
    if (s_len >= n_allowed) {
        for (size_t v = 0; v < n; ++v) {
            if (allowed_mask == nullptr || allowed_mask[v] != 0) {
                list.push_back({ v, 0.0f });
            }
        }
    } else {
        // Upstream seeds an unmasked scan's collector with the r-th best
        // score of a 48-block sample and rescans a query that comes back
        // with fewer than s_len rows above it. The rows kept are the same
        // either way; only which query is short matters, to the caller.
        const bool seeded = kSeedSupported && arm_rules && opts.seed_short != nullptr && allowed_mask == nullptr &&
            2 * s_len < n_allowed && n / kBlock >= kSeedMinBlocks;
        float seed = -std::numeric_limits<float>::infinity();
        if (seeded) {
            seed = sample_seed(idx, sign_lut, s_len, bits);
        }
        // The best s_len-th score any range has kept so far: every row of
        // the shortlist scores at least that, so a range may drop a row that
        // scores strictly less (a tie may still win on its slot).
        std::atomic<double> floor_score{ -std::numeric_limits<double>::infinity() };
        // Collects a range's rows in a buffer cut back to its best s_len
        // whenever it doubles; the s_len-th best kept is the cut-off.
        auto scan = [&](size_t block_begin, size_t block_end, std::vector<ScoreId> & top, size_t & above_seed) {
            // A short range never needs the full 2 * s_len.
            top.reserve(std::min(2 * s_len, (block_end - block_begin) * kBlock + 1));
            bool cut = false;
            double threshold = 0.0;
            auto cut_back = [&]() {
                std::nth_element(top.begin(), top.begin() + static_cast<std::ptrdiff_t>(s_len - 1), top.end(), score_id_better);
                top.resize(s_len);
                cut = true;
                threshold = top.back().score;
                if (split) {
                    double seen = floor_score.load(std::memory_order_relaxed);
                    while (threshold > seen &&
                           !floor_score.compare_exchange_weak(seen, threshold, std::memory_order_relaxed)) {
                    }
                }
            };
            float block_scores[kBlock];
            for (size_t block = block_begin; block < block_end; ++block) {
                if (!scan_blocks.empty() && scan_blocks[block] == 0) {
                    continue;
                }
                score_sign_block(idx, sign_lut, idx.turbovec_sign_blocked.data(), block, n, block_scores);
                const size_t count = std::min(kBlock, n - block * kBlock);
                if (seeded) {
                    for (size_t lane = 0; lane < count; ++lane) {
                        above_seed += block_scores[lane] > seed ? 1 : 0;
                    }
                }
                // Rows arrive in slot order, so once s_len rows are kept a
                // row scoring no better than the worst of them cannot enter.
                const double floor = split ? floor_score.load(std::memory_order_relaxed) :
                                             -std::numeric_limits<double>::infinity();
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
            if (split && top.size() > s_len) {
                cut_back();
            }
        };
        std::vector<ScoreId> top;
        size_t above_seed = 0;
        if (split) {
            const size_t n_ranges = static_cast<size_t>(n_threads) * kRangesPerThread;
            const size_t per_range = (n_blocks + n_ranges - 1) / n_ranges;
            std::vector<std::vector<ScoreId>> range_tops(n_ranges);
            std::vector<size_t> range_above(n_ranges, 0);
            // Two more tasks build the query's later tables meanwhile.
            parallel_for(n_ranges + 2, n_threads, [&](size_t range, size_t) {
                if (range == n_ranges) {
                    build_low();
                    return;
                }
                if (range == n_ranges + 1) {
                    build_exact();
                    return;
                }
                const size_t begin = range * per_range;
                const size_t end = std::min(n_blocks, begin + per_range);
                if (begin < end) {
                    scan(begin, end, range_tops[range], range_above[range]);
                }
            });
            // A row below the final floor is below some range's s_len rows.
            const double floor = floor_score.load(std::memory_order_relaxed);
            size_t total = 0;
            for (size_t range = 0; range < n_ranges; ++range) {
                above_seed += range_above[range];
                total += range_tops[range].size();
            }
            top.reserve(total);
            for (const std::vector<ScoreId> & range_top : range_tops) {
                for (const ScoreId & c : range_top) {
                    if (c.score >= floor) {
                        top.push_back(c);
                    }
                }
            }
        } else {
            scan(0, n_blocks, top, above_seed);
        }
        if (top.size() > s_len) {
            std::nth_element(top.begin(), top.begin() + static_cast<std::ptrdiff_t>(s_len - 1), top.end(), score_id_better);
            top.resize(s_len);
        }
        list.reserve(top.size());
        for (const ScoreId & c : top) {
            const size_t v = static_cast<size_t>(c.id);
            list.push_back({ v, plain_score(v, static_cast<float>(c.score)) });
        }
        if (seeded) {
            *opts.seed_short = above_seed < s_len;
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
        auto plane_term = [&](size_t v, size_t plane) -> float {
            const uint8_t * low = idx.turbovec_data.data() + v * row_bytes + plane * nsg;
            if (table_rank) {
                // The sign tables over a low plane row (upstream table_sums),
                // undone to m * sum(+-q).
                uint32_t sum = 0;
                for (size_t g = 0; g < nsg; ++g) {
                    sum += static_cast<uint32_t>(rank_lut.values[g * 32 + (low[g] >> 4)]) +
                        rank_lut.values[g * 32 + 16 + (low[g] & 0x0f)];
                }
                const float b = rank_lut.bias - bias_corr;
                return rank_lut.scale * static_cast<float>(sum) + b;
            }
            return low_planes.unit * static_cast<float>(2 * low_dot(low_planes, low) - low_planes.sum_w);
        };
        const size_t top_plane = n_low - 1;
        std::vector<float> tops(list.size());
        for_each_candidate(list.size(), [&](size_t i) {
            Candidate & c = list[i];
            const float l = plane_term(c.slot, top_plane);
            tops[i] = l;
            const float vscale = vscales[c.slot];
            if (vscale == 0.0f) {
                c.score = 0.0f;
            } else {
                const float m_s = c.score / vscale - bias_corr;
                c.score = vscale * (a1_over_m * m_s + b1_over_m * l + bias_corr);
            }
        });
        if (n_low > 1) {
            // keep_best's choice follows the strict (score, slot) order, so
            // it does not depend on the order the shortlist arrived in.
            keep_best(list, mid_len(k_sz), &tops);
            for_each_candidate(list.size(), [&](size_t i) {
                Candidate & c = list[i];
                const float vscale = vscales[c.slot];
                if (vscale == 0.0f) {
                    c.score = 0.0f;
                    return;
                }
                float l[3] = {};
                for (size_t j = 0; j + 1 < n_low; ++j) {
                    l[j] = plane_term(c.slot, j);
                }
                l[n_low - 1] = tops[i];
                const float m_s = ((c.score / vscale - bias_corr) - b1_over_m * l[n_low - 1]) / a1_over_m;
                float est = a_over_m * m_s;
                for (size_t j = 0; j < n_low; ++j) {
                    est += b_over_m[j] * l[j];
                }
                c.score = vscale * (est + bias_corr);
            });
        }
        keep_best(list, t_len);
    }

    // Stage four: the exact score for what is left, from the full scan's own
    // block kernel, so a returned score equals the full scan's.
    if (!exact_built) {
        build_exact();
    }
    std::sort(list.begin(), list.end(), [](const Candidate & a, const Candidate & b) { return a.slot < b.slot; });
    if (exact4) {
        std::vector<float> exact(list.size());
        for_each_candidate(list.size(), [&](size_t i) { exact[i] = exact4_score(list[i].slot); });
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

int32_t turbovec_low_planes_for_test(const float * q, int dim, float m, float * unit, std::vector<uint64_t> * masks) {
    const LowPlanes planes = build_low_planes(q, m, dim);
    *unit = planes.unit;
    *masks = planes.masks;
    return planes.sum_w;
}

void turbovec_force_planes_for_test(int mode, size_t min_vectors) {
    g_planes_override.store(mode, std::memory_order_relaxed);
    g_planes_min_vectors.store(min_vectors == 0 ? kTurboVecPlanesMinVectors : min_vectors, std::memory_order_relaxed);
}
#endif
