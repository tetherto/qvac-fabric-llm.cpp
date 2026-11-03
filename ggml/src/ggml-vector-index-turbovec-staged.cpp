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
// refinement on a thread pool is not reproduced.

#include "ggml-vector-index-impl.h"

#include <bitset>

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
    float scale = 1.0f;
    float bias = 0.0f;
};

// Per-query nibble tables over a sign plane (upstream build_sign_lut): the
// high nibble of byte group g covers dims 8g..8g+3, the low nibble the next
// four, and entry p is sum_j (+-m) * q[d + j].
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
    for (int i = 0; i < dim; ++i) {
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
                acc += (static_cast<int64_t>(popcount64(l & p)) - static_cast<int64_t>(popcount64(l & n))) << b;
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
    // Upstream pack.rs `planes_for`: aarch64 searches 2-bit planes always
    // and 4-bit planes with its dot-product kernels; x86 needs AVX-512
    // kernels this port does not have yet.
#if GGML_VEC_INDEX_USE_NEON
    if (dim % 32 != 0) {
        return false;
    }
    return bits == 2 || (bits == 4 && turbovec_cpu_vector_major());
#else
    (void) bits;
    (void) dim;
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
            idx.turbovec_data.data(), n, idx.turbovec_bits, idx.dim, first_slot, idx.turbovec_sign_blocked);
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

// Upstream seeds its buffered collectors on aarch64 only, and only for an
// index of at least 1024 full blocks (pack.rs planes_sample).
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
        score_turbovec_lut_block(
            lut.values.data(), lut.scale, lut.bias, idx.turbovec_sign_blocked.data(), idx.turbovec_scale.data(),
            block, n, 2, idx.dim / 2, scores.data() + i * kBlock);
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
    for (size_t v = 0; v < n; ++v) {
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
#if GGML_VEC_INDEX_USE_NEON
    const bool table_rank = bits == 4;
#else
    const bool table_rank = false;
#endif

    // The sign stage ranks on the query pulled toward the centre; later
    // stages read the plain query.
    std::vector<float> q_sign(query, query + dim);
    if (calibrated) {
        for (int d = 0; d < dim; ++d) {
            q_sign[static_cast<size_t>(d)] +=
                kSignCentre * idx.turbovec_tqplus_shift[static_cast<size_t>(d)] / idx.turbovec_tqplus_scale[static_cast<size_t>(d)];
        }
    }
    SignLut sign_lut = build_sign_lut(q_sign.data(), m, dim, single_query || opts.sign_single);
    sign_lut.bias += bias_corr;
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
    LowPlanes low_planes;
    if (refines && !table_rank) {
        low_planes = build_low_planes(query, m, dim);
    }
    auto plain_score = [&](size_t v, float s) {
        return centre != nullptr ? s - kSignCentre * vscales[v] * (*centre)[v] : s;
    };

    // Stage one: the shortlist, the top s_len allowed rows by sign score.
    std::vector<Candidate> list;
    if (s_len >= n_allowed) {
        for (size_t v = 0; v < n; ++v) {
            if (allowed_mask == nullptr || allowed_mask[v] != 0) {
                list.push_back({ v, 0.0f });
            }
        }
    } else {
        std::vector<ScoreId> top;
        top.reserve(s_len);
        float block_scores[kBlock];
        const size_t n_blocks = (n + kBlock - 1) / kBlock;
        // Upstream seeds an unmasked scan's collector with the r-th best
        // score of a 48-block sample and rescans a query that comes back
        // with fewer than s_len rows above it. The rows kept are the same
        // either way; only which query is short matters, to the caller.
        const bool seeded = kSeedSupported && opts.seed_short != nullptr && allowed_mask == nullptr &&
            2 * s_len < n_allowed && n / kBlock >= kSeedMinBlocks;
        float seed = -std::numeric_limits<float>::infinity();
        if (seeded) {
            seed = sample_seed(idx, sign_lut, s_len, bits);
        }
        size_t above_seed = 0;
        for (size_t block = 0; block < n_blocks; ++block) {
            score_turbovec_lut_block(
                sign_lut.values.data(), sign_lut.scale, sign_lut.bias, idx.turbovec_sign_blocked.data(), vscales,
                block, n, 2, dim / 2, block_scores);
            const size_t count = std::min(kBlock, n - block * kBlock);
            if (seeded) {
                for (size_t lane = 0; lane < count; ++lane) {
                    above_seed += block_scores[lane] > seed ? 1 : 0;
                }
            }
            // Rows arrive in slot order, so once the shortlist is full a row
            // scoring no better than its worst cannot displace it.
            const bool full = top.size() == s_len;
            const double threshold = full ? top.front().score : 0.0;
            for (size_t lane = 0; lane < count; ++lane) {
                if (full && static_cast<double>(block_scores[lane]) <= threshold) {
                    continue;
                }
                const size_t v = block * kBlock + lane;
                if (allowed_mask != nullptr && allowed_mask[v] == 0) {
                    continue;
                }
                update_topk_heap(top, static_cast<int>(s_len), { block_scores[lane], static_cast<uint64_t>(v) });
            }
        }
        for (const ScoreId & c : top) {
            const size_t v = static_cast<size_t>(c.id);
            list.push_back({ v, plain_score(v, static_cast<float>(c.score)) });
        }
        if (seeded) {
            *opts.seed_short = above_seed < s_len;
        }
    }

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
        for (size_t i = 0; i < list.size(); ++i) {
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
        }
        if (n_low > 1) {
            keep_best(list, mid_len(k_sz), &tops);
            for (size_t i = 0; i < list.size(); ++i) {
                Candidate & c = list[i];
                const float vscale = vscales[c.slot];
                if (vscale == 0.0f) {
                    c.score = 0.0f;
                    continue;
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
            }
        }
        keep_best(list, t_len);
    }

    // Stage four: the exact score for what is left, from the full scan's own
    // block kernel, so a returned score equals the full scan's.
    std::vector<uint8_t> exact_lut;
    float exact_scale = 1.0f;
    float exact_bias = 0.0f;
    TurboVecPermuteDot pd;
    const bool permute_dot = turbovec_uses_permute_dot(idx);
    if (permute_dot) {
        build_turbovec_permute_dot(query, dim, pd);
        pd.bias = bias_corr;
    } else {
        build_turbovec_lut(bits, false, query, dim, exact_lut, exact_scale, exact_bias);
        exact_bias += bias_corr;
    }
    std::sort(list.begin(), list.end(), [](const Candidate & a, const Candidate & b) { return a.slot < b.slot; });
    float block_scores[kBlock];
    size_t scored_block = SIZE_MAX;
    for (const Candidate & c : list) {
        const size_t block = c.slot / kBlock;
        if (block != scored_block) {
            score_turbovec_block(
                idx, exact_lut.data(), exact_scale, exact_bias, permute_dot ? &pd : nullptr, block, block_scores);
            scored_block = block;
        }
        update_topk_heap(heap, k, { block_scores[c.slot % kBlock], topk_key(idx, c.slot) });
    }
    return true;
}

#ifdef GGML_VEC_INDEX_TEST_HOOKS
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
