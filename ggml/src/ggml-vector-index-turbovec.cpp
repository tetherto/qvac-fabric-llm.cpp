// ggml-vector-index-turbovec.cpp - TurboQuant q2/q4 index helpers.

#include "ggml-vector-index-impl.h"

#if defined(GGML_VEC_INDEX_USE_ACCELERATE_QUERY)
#define ACCELERATE_NEW_LAPACK
#if defined(__LP64__)
#define ACCELERATE_LAPACK_ILP64
#endif
#include <Accelerate/Accelerate.h>
#endif

#if GGML_VEC_INDEX_USE_NEON
#include <arm_neon.h>
#endif

#if defined(GGML_VEC_INDEX_HAVE_AVX2_KERNEL) || defined(GGML_VEC_INDEX_HAVE_AVX512_KERNEL) || \
    defined(GGML_VEC_INDEX_HAVE_SSE42_KERNEL)
#include "ggml-vector-index-x86.h"
#endif

namespace {

constexpr int kTurboVecDimMultiple = 8;
constexpr int kTurboVecScoreBlock = 32;
// 32-row blocks per parallel task of the cache repacks.
constexpr size_t kTurboVecBlocksPerTask = 64;
#if GGML_VEC_INDEX_USE_NEON
constexpr int kTurboVecFlushEvery = 256;
#endif
constexpr uint64_t kTurboVecRotationSeed = 42;
constexpr int kTurboVecHadamardRounds = 2;
// rand_core 0.6 `seed_from_u64(42)`, frozen by upstream turbovec 1.x
// (rotation.rs ROTATION_SEED) so the rotation depends only on ChaCha8.
constexpr uint8_t kTurboVecHadamardSeed[32] = {
    164, 143, 161, 123, 88, 50, 61, 10, 234, 184, 161, 204, 105, 1, 20, 184,
    43, 140, 200, 117, 24, 180, 247, 84, 141, 68, 110, 161, 228, 223, 32, 242,
};
constexpr int kTurboVecNormChains = 8;
// Fewest rows a TQ+ fit can use: two distinct order statistics per coordinate.
constexpr int kTurboVecMinCalibrationRows = 2;
// Reconstructions whose unit-space inner product is at or below this are
// degenerate and store scale 0 (upstream #116).
constexpr double kTurboVecDegenerateInner = 0.1;
// Norms at or below this have no direction and store scale 0 (upstream
// MIN_INPUT_NORM).
constexpr float kTurboVecMinInputNorm = 1e-10f;
constexpr int kTurboVecZigguratLayers = 256;
constexpr int kTurboVecZigguratTableSize = kTurboVecZigguratLayers + 1;
constexpr uint64_t kTurboVecZigguratLayerMask = 0xff;
constexpr int kTurboVecZigguratFloatShift = 12;
constexpr int kTurboVecF64MantissaBits = 52;
constexpr uint64_t kTurboVecF64ExponentBias = 1023;
constexpr uint64_t kTurboVecF64Open01Exponent = kTurboVecF64ExponentBias;
constexpr uint64_t kTurboVecF64SymmetricExponent = kTurboVecF64ExponentBias + 1;
constexpr double kTurboVecNormalPdfExponent = -0.5;
constexpr double kTurboVecZigNormX0 = 3.910757959537090045;
constexpr double kTurboVecZigNormR = 3.654152885361008796;
constexpr int kStatrsRegularizedBetaIterations = 141;
constexpr int kStatrsContinuousCdfInverseIterations = 16;

struct TurboVecCodebook {
    std::vector<float> boundaries;
    std::vector<float> centroids;
};

struct TurboVecRotationCacheEntry {
    size_t ref_count = 0;
    std::weak_ptr<const std::vector<float>> weak;
    std::shared_ptr<const std::vector<float>> strong;
};

class ScopedNearestRounding {
public:
    ScopedNearestRounding() : saved_rounding(std::fegetround()) {
        if (saved_rounding != FE_TONEAREST && saved_rounding != -1) {
            std::fesetround(FE_TONEAREST);
        }
    }

    ~ScopedNearestRounding() {
        if (saved_rounding != FE_TONEAREST && saved_rounding != -1) {
            std::fesetround(saved_rounding);
        }
    }

    ScopedNearestRounding(const ScopedNearestRounding &) = delete;
    ScopedNearestRounding & operator=(const ScopedNearestRounding &) = delete;

private:
    int saved_rounding = FE_TONEAREST;
};

struct ChaCha8 {
    uint32_t key[8] = {};
    uint64_t counter = 0;
    uint32_t words[16] = {};
    int next_word = 16;
};

static uint32_t rotl32_local(uint32_t value, int shift) {
    return (value << shift) | (value >> (32 - shift));
}

static void chacha_quarter_round(uint32_t & a, uint32_t & b, uint32_t & c, uint32_t & d) {
    a += b;
    d = rotl32_local(d ^ a, 16);
    c += d;
    b = rotl32_local(b ^ c, 12);
    a += b;
    d = rotl32_local(d ^ a, 8);
    c += d;
    b = rotl32_local(b ^ c, 7);
}

static uint32_t load_u32_le_local(const uint8_t * src) {
    return static_cast<uint32_t>(src[0]) |
        (static_cast<uint32_t>(src[1]) << 8) |
        (static_cast<uint32_t>(src[2]) << 16) |
        (static_cast<uint32_t>(src[3]) << 24);
}

static void chacha8_seed(ChaCha8 & rng, uint64_t seed) {
    uint8_t key_bytes[32];
    for (int i = 0; i < 8; ++i) {
        seed = seed * 6364136223846793005ull + 11634580027462260723ull;
        const uint32_t xorshifted = static_cast<uint32_t>(((seed >> 18) ^ seed) >> 27);
        const uint32_t rotation = static_cast<uint32_t>(seed >> 59);
        const uint32_t value =
            (xorshifted >> rotation) | (xorshifted << ((32 - rotation) & 31));
        key_bytes[i * 4 + 0] = static_cast<uint8_t>(value);
        key_bytes[i * 4 + 1] = static_cast<uint8_t>(value >> 8);
        key_bytes[i * 4 + 2] = static_cast<uint8_t>(value >> 16);
        key_bytes[i * 4 + 3] = static_cast<uint8_t>(value >> 24);
    }
    for (int i = 0; i < 8; ++i) {
        rng.key[i] = load_u32_le_local(key_bytes + i * 4);
    }
    rng.counter = 0;
    rng.next_word = 16;
}

static void chacha8_refill(ChaCha8 & rng) {
    const uint32_t initial[16] = {
        0x61707865u, 0x3320646eu, 0x79622d32u, 0x6b206574u,
        rng.key[0], rng.key[1], rng.key[2], rng.key[3],
        rng.key[4], rng.key[5], rng.key[6], rng.key[7],
        static_cast<uint32_t>(rng.counter),
        static_cast<uint32_t>(rng.counter >> 32),
        0,
        0,
    };
    std::memcpy(rng.words, initial, sizeof(initial));
    for (int round = 0; round < 4; ++round) {
        chacha_quarter_round(rng.words[0], rng.words[4], rng.words[8], rng.words[12]);
        chacha_quarter_round(rng.words[1], rng.words[5], rng.words[9], rng.words[13]);
        chacha_quarter_round(rng.words[2], rng.words[6], rng.words[10], rng.words[14]);
        chacha_quarter_round(rng.words[3], rng.words[7], rng.words[11], rng.words[15]);
        chacha_quarter_round(rng.words[0], rng.words[5], rng.words[10], rng.words[15]);
        chacha_quarter_round(rng.words[1], rng.words[6], rng.words[11], rng.words[12]);
        chacha_quarter_round(rng.words[2], rng.words[7], rng.words[8], rng.words[13]);
        chacha_quarter_round(rng.words[3], rng.words[4], rng.words[9], rng.words[14]);
    }
    for (int i = 0; i < 16; ++i) {
        rng.words[i] += initial[i];
    }
    ++rng.counter;
    rng.next_word = 0;
}

static uint32_t chacha8_next_u32(ChaCha8 & rng) {
    if (rng.next_word == 16) {
        chacha8_refill(rng);
    }
    return rng.words[rng.next_word++];
}

static uint64_t chacha8_next_u64(ChaCha8 & rng) {
    const uint64_t lo = chacha8_next_u32(rng);
    const uint64_t hi = chacha8_next_u32(rng);
    return lo | (hi << 32);
}

static double chacha8_open01(ChaCha8 & rng) {
    // Use 52 high bits as an f64 mantissa, set exponent 1023 to build [1, 2),
    // then subtract 1 - eps/2 to match rand::distributions::Open01.
    const uint64_t fraction = chacha8_next_u64(rng) >> kTurboVecZigguratFloatShift;
    const uint64_t bits = fraction | (kTurboVecF64Open01Exponent << kTurboVecF64MantissaBits);
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value - (1.0 - std::numeric_limits<double>::epsilon() * 0.5);
}

static double chacha8_standard_f64(ChaCha8 & rng) {
    return static_cast<double>(chacha8_next_u64(rng) >> 11) *
        (1.0 / static_cast<double>(UINT64_C(1) << 53));
}

struct TurboVecNormalTables {
    double x[kTurboVecZigguratTableSize] = {};
    double f[kTurboVecZigguratTableSize] = {};
};

// Rust turbovec v0.9.0 builds the dense QR rotation from
// rand_distr::StandardNormal. rand_distr 0.4.3 implements StandardNormal with
// the ZIGNOR variant of the Ziggurat method from Doornik 2005:
// https://docs.rs/rand_distr/0.4.3/src/rand_distr/normal.rs.html
// https://docs.rs/rand_distr/0.4.3/src/rand_distr/utils.rs.html
// https://docs.rs/rand_distr/0.4.3/src/rand_distr/ziggurat_tables.rs.html
static const TurboVecNormalTables & ziggurat_standard_normal_tables() {
    static const TurboVecNormalTables tables = []() {
        TurboVecNormalTables out;
        const double area =
            kTurboVecZigNormX0 * std::exp(kTurboVecNormalPdfExponent * kTurboVecZigNormR * kTurboVecZigNormR);
        out.x[0] = kTurboVecZigNormX0;
        out.x[1] = kTurboVecZigNormR;
        for (int i = 1; i < kTurboVecZigguratLayers; ++i) {
            const double density = std::exp(kTurboVecNormalPdfExponent * out.x[i] * out.x[i]);
            out.x[i + 1] = std::sqrt(-2.0 * std::log(area / out.x[i] + density));
        }
        out.x[kTurboVecZigguratLayers] = 0.0;
        for (int i = 0; i < kTurboVecZigguratTableSize; ++i) {
            out.f[i] = std::exp(kTurboVecNormalPdfExponent * out.x[i] * out.x[i]);
        }
        return out;
    }();
    return tables;
}

static double sample_standard_normal_ziggurat_tail(ChaCha8 & rng, double u) {
    double x = 1.0;
    double y = 0.0;
    while (-2.0 * y < x * x) {
        x = std::log(chacha8_open01(rng)) / kTurboVecZigNormR;
        y = std::log(chacha8_open01(rng));
    }
    return u < 0.0 ? x - kTurboVecZigNormR : kTurboVecZigNormR - x;
}

static double sample_standard_normal_ziggurat(ChaCha8 & rng) {
    const TurboVecNormalTables & tables = ziggurat_standard_normal_tables();
    for (;;) {
        const uint64_t bits = chacha8_next_u64(rng);
        const int index = static_cast<int>(bits & kTurboVecZigguratLayerMask);
        // The low 8 bits choose one of 256 Ziggurat layers. Dropping 12 bits
        // leaves 52 mantissa bits; exponent 1024 builds [2, 4), minus 3.0
        // gives the symmetric [-1, 1) variate used by rand_distr::ziggurat.
        const uint64_t fraction = bits >> kTurboVecZigguratFloatShift;
        const uint64_t float_bits =
            fraction | (kTurboVecF64SymmetricExponent << kTurboVecF64MantissaBits);
        double u = 0.0;
        std::memcpy(&u, &float_bits, sizeof(u));
        u -= 3.0;
        const double x = u * tables.x[index];
        if (std::fabs(x) < tables.x[index + 1]) {
            return x;
        }
        if (index == 0) {
            return sample_standard_normal_ziggurat_tail(rng, u);
        }
        const double density = std::exp(kTurboVecNormalPdfExponent * x * x);
        if (tables.f[index + 1] +
                (tables.f[index] - tables.f[index + 1]) * chacha8_standard_f64(rng) <
            density) {
            return x;
        }
    }
}

static std::vector<float> make_turbovec_rotation(int dim) {
    const ScopedNearestRounding rounding_guard;
    const size_t dim_sz = static_cast<size_t>(dim);
    std::vector<double> matrix(dim_sz * dim_sz);
    ChaCha8 rng;
    chacha8_seed(rng, kTurboVecRotationSeed);

    for (int column = 0; column < dim; ++column) {
        for (int row = 0; row < dim; ++row) {
            matrix[static_cast<size_t>(row) * dim_sz + static_cast<size_t>(column)] =
                sample_standard_normal_ziggurat(rng);
        }
    }

    std::vector<double> q(dim_sz * dim_sz, 0.0);
    for (int i = 0; i < dim; ++i) {
        q[static_cast<size_t>(i) * dim_sz + static_cast<size_t>(i)] = 1.0;
    }
    std::vector<double> reflector(dim_sz, 0.0);

    for (int column = 0; column < dim; ++column) {
        double norm = 0.0;
        for (int row = column; row < dim; ++row) {
            const double value = matrix[static_cast<size_t>(row) * dim_sz + static_cast<size_t>(column)];
            norm = std::hypot(norm, value);
        }
        if (norm == 0.0) {
            continue;
        }
        const size_t diagonal = static_cast<size_t>(column) * dim_sz + static_cast<size_t>(column);
        const double alpha = matrix[diagonal] >= 0.0 ? -norm : norm;
        double reflector_norm = 0.0;
        for (int row = column; row < dim; ++row) {
            const double value =
                matrix[static_cast<size_t>(row) * dim_sz + static_cast<size_t>(column)] -
                (row == column ? alpha : 0.0);
            reflector[static_cast<size_t>(row)] = value;
            reflector_norm += value * value;
        }
        if (reflector_norm == 0.0) {
            continue;
        }
        const double beta = 2.0 / reflector_norm;

        for (int j = column; j < dim; ++j) {
            double projection = 0.0;
            for (int row = column; row < dim; ++row) {
                projection += reflector[static_cast<size_t>(row)] *
                    matrix[static_cast<size_t>(row) * dim_sz + static_cast<size_t>(j)];
            }
            projection *= beta;
            for (int row = column; row < dim; ++row) {
                matrix[static_cast<size_t>(row) * dim_sz + static_cast<size_t>(j)] -=
                    projection * reflector[static_cast<size_t>(row)];
            }
        }

        for (int row = 0; row < dim; ++row) {
            double projection = 0.0;
            for (int j = column; j < dim; ++j) {
                projection += q[static_cast<size_t>(row) * dim_sz + static_cast<size_t>(j)] *
                    reflector[static_cast<size_t>(j)];
            }
            projection *= beta;
            for (int j = column; j < dim; ++j) {
                q[static_cast<size_t>(row) * dim_sz + static_cast<size_t>(j)] -=
                    projection * reflector[static_cast<size_t>(j)];
            }
        }
    }

    for (int column = 0; column < dim; ++column) {
        const double sign =
            matrix[static_cast<size_t>(column) * dim_sz + static_cast<size_t>(column)] >= 0.0 ?
                1.0 : -1.0;
        if (sign < 0.0) {
            for (int row = 0; row < dim; ++row) {
                q[static_cast<size_t>(row) * dim_sz + static_cast<size_t>(column)] *= sign;
            }
        }
    }

    std::vector<float> result(dim_sz * dim_sz);
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = static_cast<float>(q[i]);
    }
    return result;
}

static std::mutex & turbovec_rotation_cache_mutex() {
    static std::mutex mutex;
    return mutex;
}

static std::unordered_map<int, TurboVecRotationCacheEntry> & turbovec_rotation_cache() {
    static std::unordered_map<int, TurboVecRotationCacheEntry> cache;
    return cache;
}

static std::shared_ptr<const std::vector<float>> turbovec_rotation(int dim) {
    std::lock_guard<std::mutex> lock(turbovec_rotation_cache_mutex());
    auto & cache = turbovec_rotation_cache();
    auto it = cache.find(dim);
    if (it == cache.end()) {
        it = cache.emplace(dim, TurboVecRotationCacheEntry{}).first;
    }
    std::shared_ptr<const std::vector<float>> rotation =
        it->second.strong != nullptr ? it->second.strong : it->second.weak.lock();
    if (rotation == nullptr) {
        rotation = std::make_shared<const std::vector<float>>(make_turbovec_rotation(dim));
        it->second.weak = rotation;
        if (it->second.ref_count != 0) {
            it->second.strong = rotation;
        }
    }
    return rotation;
}

static void apply_turbovec_rotation_with_matrix(
        const float * matrix,
        const float * src,
        float * dst,
        int dim,
        bool transpose) {
    const size_t dim_sz = static_cast<size_t>(dim);
    for (int row = 0; row < dim; ++row) {
        float sum = 0.0f;
        for (int column = 0; column < dim; ++column) {
            const size_t matrix_index = transpose ?
                static_cast<size_t>(column) * dim_sz + static_cast<size_t>(row) :
                static_cast<size_t>(row) * dim_sz + static_cast<size_t>(column);
            sum = std::fma(matrix[matrix_index], src[static_cast<size_t>(column)], sum);
        }
        dst[static_cast<size_t>(row)] = sum;
    }
}

static void apply_turbovec_rotation(const float * src, float * dst, int dim, bool transpose) {
    const std::shared_ptr<const std::vector<float>> rotation = turbovec_rotation(dim);
    apply_turbovec_rotation_with_matrix(rotation->data(), src, dst, dim, transpose);
}

static void apply_turbovec_rotation_batch(const float * src, float * dst, int n, int dim) {
    const std::shared_ptr<const std::vector<float>> rotation = turbovec_rotation(dim);
    // TQ+ calibration is persisted; keep this reduction order backend-independent.
    for (int row = 0; row < n; ++row) {
        apply_turbovec_rotation_with_matrix(
            rotation->data(),
            src + static_cast<size_t>(row) * static_cast<size_t>(dim),
            dst + static_cast<size_t>(row) * static_cast<size_t>(dim),
            dim,
            false);
    }
}

static void apply_turbovec_query_rotation_batch(
        const float * src,
        float * dst,
        int n,
        int dim) {
#if defined(GGML_VEC_INDEX_USE_ACCELERATE_QUERY)
    if (n <= 0) {
        return;
    }
    if (__builtin_available(
            macOS 13.3,
            iOS 16.4,
            tvOS 16.4,
            watchOS 9.4,
            *)) {
        const std::shared_ptr<const std::vector<float>> rotation = turbovec_rotation(dim);
        constexpr int batch_size = 16;
        std::vector<float> padded;
        for (int row = 0; row < n; row += batch_size) {
            const int count = std::min(batch_size, n - row);
            const float * batch_src =
                src + static_cast<size_t>(row) * static_cast<size_t>(dim);
            float * batch_dst =
                dst + static_cast<size_t>(row) * static_cast<size_t>(dim);
            if (count < batch_size) {
                const size_t padded_count =
                    static_cast<size_t>(batch_size) * static_cast<size_t>(dim);
                padded.assign(2 * padded_count, 0.0f);
                std::memcpy(
                    padded.data(),
                    batch_src,
                    static_cast<size_t>(count) * static_cast<size_t>(dim) * sizeof(float));
                batch_src = padded.data();
                batch_dst = padded.data() + padded_count;
            }
            cblas_sgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                batch_size,
                dim,
                dim,
                1.0f,
                batch_src,
                dim,
                rotation->data(),
                dim,
                0.0f,
                batch_dst,
                dim);
            if (count < batch_size) {
                std::memcpy(
                    dst + static_cast<size_t>(row) * static_cast<size_t>(dim),
                    batch_dst,
                    static_cast<size_t>(count) * static_cast<size_t>(dim) * sizeof(float));
            }
        }
        return;
    }
#endif
    apply_turbovec_rotation_batch(src, dst, n, dim);
}

// TurboVec 1.x rotation: a globally permuted block-Hadamard transform, ported
// from upstream turbovec 1.1.2 `rotation.rs`. Every multiply and add happens in
// the upstream order with no fused multiply-add, so the rotated coordinates,
// and therefore the encoded bytes, are identical on every target.
struct TurboVecHadamardRotation {
    int dim = 0;
    int block = 0;
    float inv_sqrt_block = 1.0f;
    std::array<std::vector<float>, kTurboVecHadamardRounds> signs;
    std::array<std::vector<uint32_t>, kTurboVecHadamardRounds> perms;
};

struct TurboVecHadamardCacheEntry {
    size_t ref_count = 0;
    std::weak_ptr<const TurboVecHadamardRotation> weak;
    std::shared_ptr<const TurboVecHadamardRotation> strong;
};

static void chacha8_seed_bytes(ChaCha8 & rng, const uint8_t (&seed)[32]) {
    for (int i = 0; i < 8; ++i) {
        rng.key[i] = load_u32_le_local(seed + i * 4);
    }
    rng.counter = 0;
    rng.next_word = 16;
}

static TurboVecHadamardRotation make_turbovec_hadamard_rotation(int dim) {
    TurboVecHadamardRotation rotation;
    rotation.dim = dim;
    // Largest power-of-two divisor of dim; at least 8 because dim % 8 == 0.
    rotation.block = dim & -dim;
    rotation.inv_sqrt_block = 1.0f / std::sqrt(static_cast<float>(rotation.block));

    // One ChaCha8 stream drives the construction. Per round: dim sign draws,
    // then a Fisher-Yates permutation. The draw order is part of the format.
    ChaCha8 rng;
    chacha8_seed_bytes(rng, kTurboVecHadamardSeed);
    const size_t dim_sz = static_cast<size_t>(dim);
    for (int round = 0; round < kTurboVecHadamardRounds; ++round) {
        std::vector<float> & signs = rotation.signs[static_cast<size_t>(round)];
        signs.resize(dim_sz);
        for (size_t i = 0; i < dim_sz; ++i) {
            signs[i] = (chacha8_next_u32(rng) & 1u) != 0 ? -1.0f : 1.0f;
        }
        std::vector<uint32_t> & perm = rotation.perms[static_cast<size_t>(round)];
        perm.resize(dim_sz);
        for (size_t i = 0; i < dim_sz; ++i) {
            perm[i] = static_cast<uint32_t>(i);
        }
        for (size_t i = dim_sz - 1; i >= 1; --i) {
            const size_t j = static_cast<size_t>(
                chacha8_next_u64(rng) % (static_cast<uint64_t>(i) + 1));
            std::swap(perm[i], perm[j]);
        }
    }
    return rotation;
}

static std::mutex & turbovec_hadamard_cache_mutex() {
    static std::mutex mutex;
    return mutex;
}

static std::unordered_map<int, TurboVecHadamardCacheEntry> & turbovec_hadamard_cache() {
    static std::unordered_map<int, TurboVecHadamardCacheEntry> cache;
    return cache;
}

static std::shared_ptr<const TurboVecHadamardRotation> turbovec_hadamard_rotation(int dim) {
    std::lock_guard<std::mutex> lock(turbovec_hadamard_cache_mutex());
    auto & cache = turbovec_hadamard_cache();
    auto it = cache.find(dim);
    if (it == cache.end()) {
        it = cache.emplace(dim, TurboVecHadamardCacheEntry{}).first;
    }
    std::shared_ptr<const TurboVecHadamardRotation> rotation =
        it->second.strong != nullptr ? it->second.strong : it->second.weak.lock();
    if (rotation == nullptr) {
        rotation = std::make_shared<const TurboVecHadamardRotation>(
            make_turbovec_hadamard_rotation(dim));
        it->second.weak = rotation;
        if (it->second.ref_count != 0) {
            it->second.strong = rotation;
        }
    }
    return rotation;
}

// SIMD kernel family a rotation or encode helper runs with. Upstream
// turbovec 1.1.2 has NEON, AVX2 and AVX-512 forms of these kernels and
// documents each as bit-identical to its scalar reference; the parity test
// hooks below hold every form this host can run to that.
enum class TurboVecSimd { scalar, neon, avx2, avx512 };

// The widest kernel family this host runs, picked once (upstream's
// is_x86_feature_detected! dispatch: AVX-512 F, then AVX2; NEON is baseline
// on aarch64). Kernels without an AVX-512 form use their AVX2 one.
static TurboVecSimd turbovec_host_simd() {
#if GGML_VEC_INDEX_USE_NEON
    return TurboVecSimd::neon;
#else
    static const TurboVecSimd simd = [] {
        const VecIndexCpuFeatures & features = vec_index_cpu_features();
#if defined(GGML_VEC_INDEX_HAVE_AVX512_KERNEL)
        if (features.avx512f) {
            return TurboVecSimd::avx512;
        }
#endif
#if defined(GGML_VEC_INDEX_HAVE_AVX2_KERNEL)
        if (features.avx2) {
            return TurboVecSimd::avx2;
        }
#endif
        (void) features;
        return TurboVecSimd::scalar;
    }();
    return simd;
#endif
}

// Unnormalized in-place Walsh-Hadamard butterfly on one block, then the
// orthonormal 1/sqrt(block) scale (upstream `wht_block_scalar`).
static void turbovec_wht_block_scalar(float * blk, int block, float inv_sqrt_block) {
    for (int len = 1; len < block; len <<= 1) {
        for (int i = 0; i < block; i += 2 * len) {
            for (int j = i; j < i + len; ++j) {
                const float a = blk[j];
                const float b = blk[j + len];
                blk[j] = a + b;
                blk[j + len] = a - b;
            }
        }
    }
    for (int i = 0; i < block; ++i) {
        blk[i] *= inv_sqrt_block;
    }
}

#if GGML_VEC_INDEX_USE_NEON
// Upstream rotation.rs aarch64 `wht_block`: stages len = 1 and 2 fused in
// registers, then radix-8 passes (three stages each), a radix-4 pass if two
// stages remain and a radix-2 pass if one does. Butterflies within a stage
// touch disjoint elements, and every output is the same (a +- b) expression
// tree with the same f32 roundings as the scalar ladder, so the result is
// bit-identical to turbovec_wht_block_scalar.
static void turbovec_wht_block_neon(float * p, int block, float inv_sqrt_block) {
    for (int j = 0; j < block; j += 8) {
        // len = 1: adjacent pairs.
        const float32x4x2_t ab = vld2q_f32(p + j);
        const float32x4_t s1 = vaddq_f32(ab.val[0], ab.val[1]);  // [s0 s1 s2 s3]
        const float32x4_t d1 = vsubq_f32(ab.val[0], ab.val[1]);  // [d0 d1 d2 d3]
        // len = 2 pairs (s0, d0) with (s1, d1) and (s2, d2) with (s3, d3).
        const float32x4_t s_even = vtrn1q_f32(s1, d1);  // [s0 d0 s2 d2]
        const float32x4_t s_odd = vtrn2q_f32(s1, d1);   // [s1 d1 s3 d3]
        const float32x4_t sum2 = vaddq_f32(s_even, s_odd);
        const float32x4_t dif2 = vsubq_f32(s_even, s_odd);
        vst1q_f32(p + j, vcombine_f32(vget_low_f32(sum2), vget_low_f32(dif2)));
        vst1q_f32(p + j + 4, vcombine_f32(vget_high_f32(sum2), vget_high_f32(dif2)));
    }

    int len = 4;
    for (; 4 * len < block; len <<= 3) {
        for (int i = 0; i < block; i += 8 * len) {
            for (int j = i; j < i + len; j += 4) {
                const float32x4_t a = vld1q_f32(p + j);
                const float32x4_t b = vld1q_f32(p + j + len);
                const float32x4_t c = vld1q_f32(p + j + 2 * len);
                const float32x4_t d = vld1q_f32(p + j + 3 * len);
                const float32x4_t e = vld1q_f32(p + j + 4 * len);
                const float32x4_t f = vld1q_f32(p + j + 5 * len);
                const float32x4_t g = vld1q_f32(p + j + 6 * len);
                const float32x4_t h = vld1q_f32(p + j + 7 * len);
                const float32x4_t apb = vaddq_f32(a, b);
                const float32x4_t amb = vsubq_f32(a, b);
                const float32x4_t cpd = vaddq_f32(c, d);
                const float32x4_t cmd = vsubq_f32(c, d);
                const float32x4_t epf = vaddq_f32(e, f);
                const float32x4_t emf = vsubq_f32(e, f);
                const float32x4_t gph = vaddq_f32(g, h);
                const float32x4_t gmh = vsubq_f32(g, h);
                const float32x4_t s0 = vaddq_f32(apb, cpd);
                const float32x4_t s1 = vaddq_f32(amb, cmd);
                const float32x4_t s2 = vsubq_f32(apb, cpd);
                const float32x4_t s3 = vsubq_f32(amb, cmd);
                const float32x4_t s4 = vaddq_f32(epf, gph);
                const float32x4_t s5 = vaddq_f32(emf, gmh);
                const float32x4_t s6 = vsubq_f32(epf, gph);
                const float32x4_t s7 = vsubq_f32(emf, gmh);
                vst1q_f32(p + j, vaddq_f32(s0, s4));
                vst1q_f32(p + j + len, vaddq_f32(s1, s5));
                vst1q_f32(p + j + 2 * len, vaddq_f32(s2, s6));
                vst1q_f32(p + j + 3 * len, vaddq_f32(s3, s7));
                vst1q_f32(p + j + 4 * len, vsubq_f32(s0, s4));
                vst1q_f32(p + j + 5 * len, vsubq_f32(s1, s5));
                vst1q_f32(p + j + 6 * len, vsubq_f32(s2, s6));
                vst1q_f32(p + j + 7 * len, vsubq_f32(s3, s7));
            }
        }
    }
    if (2 * len < block) {
        for (int i = 0; i < block; i += 4 * len) {
            for (int j = i; j < i + len; j += 4) {
                const float32x4_t a = vld1q_f32(p + j);
                const float32x4_t b = vld1q_f32(p + j + len);
                const float32x4_t c = vld1q_f32(p + j + 2 * len);
                const float32x4_t d = vld1q_f32(p + j + 3 * len);
                const float32x4_t apb = vaddq_f32(a, b);
                const float32x4_t amb = vsubq_f32(a, b);
                const float32x4_t cpd = vaddq_f32(c, d);
                const float32x4_t cmd = vsubq_f32(c, d);
                vst1q_f32(p + j, vaddq_f32(apb, cpd));
                vst1q_f32(p + j + len, vaddq_f32(amb, cmd));
                vst1q_f32(p + j + 2 * len, vsubq_f32(apb, cpd));
                vst1q_f32(p + j + 3 * len, vsubq_f32(amb, cmd));
            }
        }
        len <<= 2;
    }
    if (len < block) {
        for (int i = 0; i < block; i += 2 * len) {
            for (int j = i; j < i + len; j += 4) {
                const float32x4_t a = vld1q_f32(p + j);
                const float32x4_t b = vld1q_f32(p + j + len);
                vst1q_f32(p + j, vaddq_f32(a, b));
                vst1q_f32(p + j + len, vsubq_f32(a, b));
            }
        }
    }

    const float32x4_t sv = vdupq_n_f32(inv_sqrt_block);
    for (int j = 0; j < block; j += 4) {
        vst1q_f32(p + j, vmulq_f32(vld1q_f32(p + j), sv));
    }
}
#endif

static void turbovec_wht_block_with(TurboVecSimd simd, float * blk, int block, float inv_sqrt_block) {
    switch (simd) {
#if GGML_VEC_INDEX_USE_NEON
        case TurboVecSimd::neon:
            turbovec_wht_block_neon(blk, block, inv_sqrt_block);
            return;
#endif
#if defined(GGML_VEC_INDEX_HAVE_AVX512_KERNEL)
        case TurboVecSimd::avx512:
            ggml_vec_index_detail::turbovec_wht_block_avx512(blk, block, inv_sqrt_block);
            return;
#endif
#if defined(GGML_VEC_INDEX_HAVE_AVX2_KERNEL)
        case TurboVecSimd::avx2:
            ggml_vec_index_detail::turbovec_wht_block_avx2(blk, block, inv_sqrt_block);
            return;
#endif
        default:
            break;
    }
    turbovec_wht_block_scalar(blk, block, inv_sqrt_block);
}

// One permutation round (upstream rotation.rs `permute_gather`):
// dst[i] = src[perm[i]], times signs[i] when mode is 1, and times inv and
// then signs[i] when mode is 2. The x86 forms replace the per-element index
// and value loads with a hardware gather; the multiplies are the same IEEE
// operations in the same order, so the output is bit-identical.
static void turbovec_permute_gather_scalar(
        int mode, const float * src, const uint32_t * perm, const float * signs, float inv, float * dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        const float v = src[perm[i]];
        dst[i] = mode == 0 ? v : (mode == 1 ? v * signs[i] : (v * inv) * signs[i]);
    }
}

static void turbovec_permute_gather_with(
        TurboVecSimd simd,
        int mode,
        const float * src,
        const uint32_t * perm,
        const float * signs,
        float inv,
        float * dst,
        size_t n) {
    switch (simd) {
#if defined(GGML_VEC_INDEX_HAVE_AVX512_KERNEL)
        case TurboVecSimd::avx512:
            ggml_vec_index_detail::turbovec_permute_gather_avx512(mode, src, perm, signs, inv, dst, n);
            return;
#endif
#if defined(GGML_VEC_INDEX_HAVE_AVX2_KERNEL)
        case TurboVecSimd::avx2:
            ggml_vec_index_detail::turbovec_permute_gather_avx2(mode, src, perm, signs, inv, dst, n);
            return;
#endif
        default:
            break;
    }
    turbovec_permute_gather_scalar(mode, src, perm, signs, inv, dst, n);
}

// Rotate src scaled by inv into dst (upstream `apply_scaled_into`; with
// inv == 1 it is `apply`, since x * 1.0f is exact). scratch holds dim floats.
static void apply_turbovec_hadamard_with(
        TurboVecSimd simd,
        const TurboVecHadamardRotation & rotation,
        const float * src,
        float inv,
        float * dst,
        float * scratch) {
    static_assert(kTurboVecHadamardRounds == 2, "the buffer schedule is written for two rounds");
    const int dim = rotation.dim;
    const int block = rotation.block;
    const size_t dim_sz = static_cast<size_t>(dim);
    turbovec_permute_gather_with(
        simd, 2, src, rotation.perms[0].data(), rotation.signs[0].data(), inv, scratch, dim_sz);
    for (int offset = 0; offset < dim; offset += block) {
        turbovec_wht_block_with(simd, scratch + offset, block, rotation.inv_sqrt_block);
    }
    turbovec_permute_gather_with(
        simd, 1, scratch, rotation.perms[1].data(), rotation.signs[1].data(), 1.0f, dst, dim_sz);
    for (int offset = 0; offset < dim; offset += block) {
        turbovec_wht_block_with(simd, dst + offset, block, rotation.inv_sqrt_block);
    }
}

static void apply_turbovec_hadamard(
        const TurboVecHadamardRotation & rotation,
        const float * src,
        float inv,
        float * dst,
        float * scratch) {
    apply_turbovec_hadamard_with(turbovec_host_simd(), rotation, src, inv, dst, scratch);
}

// Inverse of apply_turbovec_hadamard with inv == 1. The normalized
// Walsh-Hadamard block is its own inverse, so each round is undone in reverse.
static void apply_turbovec_hadamard_inverse_with(
        TurboVecSimd simd,
        const TurboVecHadamardRotation & rotation,
        const float * src,
        float * dst,
        float * scratch) {
    const int dim = rotation.dim;
    const int block = rotation.block;
    std::memcpy(scratch, src, static_cast<size_t>(dim) * sizeof(float));
    for (int round = kTurboVecHadamardRounds - 1; round >= 0; --round) {
        for (int offset = 0; offset < dim; offset += block) {
            turbovec_wht_block_with(simd, scratch + offset, block, rotation.inv_sqrt_block);
        }
        const uint32_t * perm = rotation.perms[static_cast<size_t>(round)].data();
        const float * signs = rotation.signs[static_cast<size_t>(round)].data();
        for (int i = 0; i < dim; ++i) {
            dst[perm[i]] = scratch[i] * signs[i];
        }
        if (round != 0) {
            std::memcpy(scratch, dst, static_cast<size_t>(dim) * sizeof(float));
        }
    }
}

static void apply_turbovec_hadamard_inverse(
        const TurboVecHadamardRotation & rotation,
        const float * src,
        float * dst,
        float * scratch) {
    apply_turbovec_hadamard_inverse_with(turbovec_host_simd(), rotation, src, dst, scratch);
}

// Squared L2 norm with upstream's eight fixed accumulation chains (encode.rs
// `norm_sq_scalar`): chain j % 8, combined pairwise.
static float turbovec_norm_sq_scalar(const float * row, int dim) {
    float chains[kTurboVecNormChains] = {};
    for (int j = 0; j < dim; ++j) {
        chains[j % kTurboVecNormChains] += row[j] * row[j];
    }
    return ((chains[0] + chains[1]) + (chains[2] + chains[3])) +
        ((chains[4] + chains[5]) + (chains[6] + chains[7]));
}

#if GGML_VEC_INDEX_USE_NEON
// Upstream encode.rs `norm_sq_neon`: the eight chains are two float32x4
// accumulators, multiplied and added as separate instructions (never a
// fused multiply-add), so each term rounds twice as in the scalar chains.
static float turbovec_norm_sq_neon(const float * row, int dim) {
    float32x4_t acc_lo = vdupq_n_f32(0.0f);
    float32x4_t acc_hi = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + kTurboVecNormChains <= dim; i += kTurboVecNormChains) {
        const float32x4_t lo = vld1q_f32(row + i);
        const float32x4_t hi = vld1q_f32(row + i + 4);
        acc_lo = vaddq_f32(acc_lo, vmulq_f32(lo, lo));
        acc_hi = vaddq_f32(acc_hi, vmulq_f32(hi, hi));
    }
    float chains[kTurboVecNormChains];
    vst1q_f32(chains, acc_lo);
    vst1q_f32(chains + 4, acc_hi);
    for (; i < dim; ++i) {
        chains[i % kTurboVecNormChains] += row[i] * row[i];
    }
    return ((chains[0] + chains[1]) + (chains[2] + chains[3])) +
        ((chains[4] + chains[5]) + (chains[6] + chains[7]));
}
#endif

static float turbovec_norm_sq_with(TurboVecSimd simd, const float * row, int dim) {
    switch (simd) {
#if GGML_VEC_INDEX_USE_NEON
        case TurboVecSimd::neon:
            return turbovec_norm_sq_neon(row, dim);
#endif
#if defined(GGML_VEC_INDEX_HAVE_AVX2_KERNEL)
        case TurboVecSimd::avx2:
        case TurboVecSimd::avx512:
            return ggml_vec_index_detail::turbovec_norm_sq_avx(row, dim);
#endif
        default:
            break;
    }
    return turbovec_norm_sq_scalar(row, dim);
}

// L2 norm (upstream encode.rs `simd_norm`): the fixed-chain squared norm,
// then one correctly rounded f32 sqrt.
static float turbovec_hadamard_norm(const float * row, int dim) {
    return std::sqrt(turbovec_norm_sq_with(turbovec_host_simd(), row, dim));
}

static double vector_norm(const float * values, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim; ++i) {
        const double value = static_cast<double>(values[i]);
        sum += value * value;
    }
    return std::sqrt(sum);
}

static float float_score_from_double_local(double score) {
    if (score > static_cast<double>(FLT_MAX)) {
        return FLT_MAX;
    }
    if (score < -static_cast<double>(FLT_MAX)) {
        return -FLT_MAX;
    }
    return static_cast<float>(score);
}

/*
 * Local subset derived from statrs 0.17.1, used by Rust turbovec v0.9.0
 * through statrs::distribution::Beta. These helpers derive TQ+ theoretical
 * quantiles and Lloyd-Max codebooks; they do not sample rotation vectors.
 *
 * Upstream sources:
 * https://docs.rs/statrs/0.17.1/src/statrs/function/gamma.rs.html
 * https://docs.rs/statrs/0.17.1/src/statrs/function/beta.rs.html
 * https://docs.rs/statrs/0.17.1/src/statrs/distribution/mod.rs.html
 *
 * MIT License
 *
 * Copyright (c) 2016 Michael Ma
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
static double statrs_ln_gamma(double x) {
    static constexpr double coefficients[] = {
        2.48574089138753565546e-5,
        1.05142378581721974210,
        -3.45687097222016235469,
        4.51227709466894823700,
        -2.98285225323576655721,
        1.05639711577126713077,
        -1.95428773191645869583e-1,
        1.70970543404441224307e-2,
        -5.71926117404305781283e-4,
        4.63399473359905636708e-6,
        -2.71994908488607703910e-9,
    };
    constexpr double gamma_r = 10.900511;
    constexpr double ln_two_sqrt_e_over_pi =
        0.6207822376352452223455184457816472122518527279025978;
    constexpr double e = 2.71828182845904523536028747135266249775724709369995;
    double sum = coefficients[0];
    for (size_t i = 1; i < std::size(coefficients); ++i) {
        sum += coefficients[i] / (x + static_cast<double>(i) - 1.0);
    }
    return std::log(sum) + ln_two_sqrt_e_over_pi +
        (x - 0.5) * std::log((x - 0.5 + gamma_r) / e);
}

static double statrs_gamma(double x) {
    static constexpr double coefficients[] = {
        2.48574089138753565546e-5,
        1.05142378581721974210,
        -3.45687097222016235469,
        4.51227709466894823700,
        -2.98285225323576655721,
        1.05639711577126713077,
        -1.95428773191645869583e-1,
        1.70970543404441224307e-2,
        -5.71926117404305781283e-4,
        4.63399473359905636708e-6,
        -2.71994908488607703910e-9,
    };
    constexpr double gamma_r = 10.900511;
    constexpr double two_sqrt_e_over_pi =
        1.8603827342052657173362492472666631120594218414085755;
    constexpr double e = 2.71828182845904523536028747135266249775724709369995;
    double sum = coefficients[0];
    for (size_t i = 1; i < std::size(coefficients); ++i) {
        sum += coefficients[i] / (x + static_cast<double>(i) - 1.0);
    }
    return sum * two_sqrt_e_over_pi *
        std::pow((x - 0.5 + gamma_r) / e, x - 0.5);
}

static double regularized_beta(double x, double a, double b) {
    if (x <= 0.0) {
        return 0.0;
    }
    if (x >= 1.0) {
        return 1.0;
    }

    // Matches statrs::function::beta::checked_beta_reg: modified Lentz
    // continued fraction with the statrs f64 precision and 1..141 cap.
    constexpr double eps = 0.00000000000000011102230246251565;
    constexpr double fpmin = std::numeric_limits<double>::min() / eps;
    const double bt = std::exp(
        statrs_ln_gamma(a + b) - statrs_ln_gamma(a) - statrs_ln_gamma(b) +
        a * std::log(x) + b * std::log(1.0 - x));
    const bool symmetric_transform = x >= (a + 1.0) / (a + b + 2.0);
    if (symmetric_transform) {
        std::swap(a, b);
        x = 1.0 - x;
    }

    const double qab = a + b;
    const double qap = a + 1.0;
    const double qam = a - 1.0;
    double c = 1.0;
    double d = 1.0 - qab * x / qap;
    if (std::fabs(d) < fpmin) {
        d = fpmin;
    }
    d = 1.0 / d;
    double h = d;

    for (int m_integer = 1; m_integer < kStatrsRegularizedBetaIterations; ++m_integer) {
        const double m = static_cast<double>(m_integer);
        const double m2 = m * 2.0;
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if (std::fabs(d) < fpmin) {
            d = fpmin;
        }
        c = 1.0 + aa / c;
        if (std::fabs(c) < fpmin) {
            c = fpmin;
        }
        d = 1.0 / d;
        h *= d * c;

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (std::fabs(d) < fpmin) {
            d = fpmin;
        }
        c = 1.0 + aa / c;
        if (std::fabs(c) < fpmin) {
            c = fpmin;
        }
        d = 1.0 / d;
        const double del = d * c;
        h *= del;
        if (std::fabs(del - 1.0) <= eps) {
            return symmetric_transform ? 1.0 - bt * h / a : bt * h / a;
        }
    }
    return symmetric_transform ? 1.0 - bt * h / a : bt * h / a;
}

static double inverse_regularized_beta(double probability, double a) {
    double lo = -2.0;
    double hi = 2.0;
    // Rust's Beta distribution does not specialize inverse_cdf in statrs 0.17.1,
    // so ContinuousCDF::inverse_cdf uses this 16-step domain bisection.
    // https://docs.rs/statrs/0.17.1/src/statrs/distribution/mod.rs.html
    for (int iteration = 0; iteration < kStatrsContinuousCdfInverseIterations; ++iteration) {
        const double mid = (lo + hi) * 0.5;
        const double cdf = mid <= 0.0 ? 0.0 :
            mid >= 1.0 ? 1.0 :
            regularized_beta(mid, a, a);
        if (cdf >= probability) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    return (lo + hi) * 0.5;
}

static double beta_pdf_01(double x, double a) {
    if (x <= 0.0 || x >= 1.0) {
        return 0.0;
    }
    if (a > 80.0) {
        const double aa = statrs_ln_gamma(2.0 * a) - statrs_ln_gamma(a) - statrs_ln_gamma(a);
        const double bb = (a - 1.0) * std::log(x);
        const double cc = (a - 1.0) * std::log(1.0 - x);
        return std::exp(aa + bb + cc);
    }
    const double beta_normalizer =
        statrs_gamma(2.0 * a) / (statrs_gamma(a) * statrs_gamma(a));
    return beta_normalizer *
        std::pow(x, a - 1.0) *
        std::pow(1.0 - x, a - 1.0);
}

template <typename Fn>
static double adaptive_simpson_rec(
        const Fn & f,
        double a,
        double b,
        double fa,
        double fb,
        double fm,
        double whole,
        double tol,
        int depth) {
    const double mid = (a + b) * 0.5;
    const double m1 = (a + mid) * 0.5;
    const double m2 = (mid + b) * 0.5;
    const double fm1 = f(m1);
    const double fm2 = f(m2);
    const double left = (mid - a) / 6.0 * (fa + 4.0 * fm1 + fm);
    const double right = (b - mid) / 6.0 * (fm + 4.0 * fm2 + fb);
    const double refined = left + right;
    if (depth == 0 || std::fabs(refined - whole) < 15.0 * tol) {
        return refined + (refined - whole) / 15.0;
    }
    return adaptive_simpson_rec(f, a, mid, fa, fm, fm1, left, tol * 0.5, depth - 1) +
        adaptive_simpson_rec(f, mid, b, fm, fb, fm2, right, tol * 0.5, depth - 1);
}

template <typename Fn>
static double adaptive_simpson(const Fn & f, double a, double b, double tol, int max_depth) {
    const double mid = (a + b) * 0.5;
    const double fa = f(a);
    const double fb = f(b);
    const double fm = f(mid);
    const double whole = (b - a) / 6.0 * (fa + 4.0 * fm + fb);
    return adaptive_simpson_rec(f, a, b, fa, fb, fm, whole, tol, max_depth);
}

// legacy selects turbovec 0.9 boundaries (f64 midpoints cast to f32); 1.x
// uses f32 midpoints of the f32 centroids.
static TurboVecCodebook make_turbovec_codebook(int bits, int dim, bool legacy) {
    const ScopedNearestRounding rounding_guard;
    const double a = (static_cast<double>(dim) - 1.0) * 0.5;
    const int n_levels = 1 << bits;
    const double std_dev = std::sqrt(2.0 * a / ((2.0 * a + 1.0) * 4.0 * a));
    const double spread = 3.0 * std_dev;
    std::vector<double> centroids(static_cast<size_t>(n_levels));
    for (int i = 0; i < n_levels; ++i) {
        centroids[static_cast<size_t>(i)] =
            -spread + 2.0 * spread * static_cast<double>(i) / static_cast<double>(n_levels - 1);
    }

    for (int iter = 0; iter < 200; ++iter) {
        std::vector<double> edges(static_cast<size_t>(n_levels) + 1);
        edges[0] = -1.0;
        for (int i = 0; i < n_levels - 1; ++i) {
            edges[static_cast<size_t>(i) + 1] =
                (centroids[static_cast<size_t>(i)] + centroids[static_cast<size_t>(i) + 1]) * 0.5;
        }
        edges[static_cast<size_t>(n_levels)] = 1.0;

        std::vector<double> next(static_cast<size_t>(n_levels), 0.0);
        for (int i = 0; i < n_levels; ++i) {
            const double lo = edges[static_cast<size_t>(i)];
            const double hi = edges[static_cast<size_t>(i) + 1];
            const double cdf_lo = regularized_beta((lo + 1.0) * 0.5, a, a);
            const double cdf_hi = regularized_beta((hi + 1.0) * 0.5, a, a);
            const double prob = cdf_hi - cdf_lo;
            if (prob < 1e-15) {
                next[static_cast<size_t>(i)] = centroids[static_cast<size_t>(i)];
            } else {
                const double mean = adaptive_simpson(
                    [a](double x) {
                        const double t = (x + 1.0) * 0.5;
                        return x * beta_pdf_01(t, a) * 0.5;
                    },
                    lo,
                    hi,
                    1e-14,
                    50);
                next[static_cast<size_t>(i)] = mean / prob;
            }
        }

        double max_change = 0.0;
        for (int i = 0; i < n_levels; ++i) {
            max_change = std::max(
                max_change,
                std::fabs(centroids[static_cast<size_t>(i)] - next[static_cast<size_t>(i)]));
        }
        centroids = std::move(next);
        if (max_change < 1e-12) {
            break;
        }
    }

    TurboVecCodebook out;
    out.boundaries.resize(static_cast<size_t>(n_levels - 1));
    out.centroids.resize(static_cast<size_t>(n_levels));
    for (int i = 0; i < n_levels; ++i) {
        out.centroids[static_cast<size_t>(i)] = static_cast<float>(centroids[static_cast<size_t>(i)]);
    }
    for (int i = 0; i < n_levels - 1; ++i) {
        const size_t level = static_cast<size_t>(i);
        out.boundaries[level] = legacy ?
            static_cast<float>((centroids[level] + centroids[level + 1]) * 0.5) :
            (out.centroids[level] + out.centroids[level + 1]) * 0.5f;
    }
    return out;
}

static const TurboVecCodebook & turbovec_codebook(int bits, int dim, bool legacy) {
    static std::mutex mutex;
    static std::unordered_map<int, TurboVecCodebook> cache;
    const int key = (legacy ? 10000000 : 0) + bits * 1000000 + dim;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(key);
    if (it == cache.end()) {
        it = cache.emplace(key, make_turbovec_codebook(bits, dim, legacy)).first;
    }
    return it->second;
}

static uint8_t turbovec_quantize_val(float value, const float * boundaries, int bits) {
    uint8_t code = 0;
    const int n_boundaries = (1 << bits) - 1;
    for (int i = 0; i < n_boundaries; ++i) {
        if (value > boundaries[i]) {
            ++code;
        }
    }
    return code;
}

static uint8_t turbovec_get_bitplane_code(const uint8_t * row, int coord, int bits, int dim) {
    const size_t bytes_per_plane = static_cast<size_t>(dim) / 8;
    const size_t byte_pos = static_cast<size_t>(coord) / 8;
    const uint8_t mask = static_cast<uint8_t>(1u << (7 - (coord & 7)));
    uint8_t code = 0;
    for (int p = 0; p < bits; ++p) {
        if ((row[static_cast<size_t>(p) * bytes_per_plane + byte_pos] & mask) != 0) {
            code |= static_cast<uint8_t>(1u << p);
        }
    }
    return code;
}

static void turbovec_set_bitplane_code(uint8_t * row, int coord, int bits, int dim, uint8_t code) {
    const size_t bytes_per_plane = static_cast<size_t>(dim) / 8;
    const size_t byte_pos = static_cast<size_t>(coord) / 8;
    const uint8_t mask = static_cast<uint8_t>(1u << (7 - (coord & 7)));
    for (int p = 0; p < bits; ++p) {
        if ((code & (1u << p)) != 0) {
            row[static_cast<size_t>(p) * bytes_per_plane + byte_pos] |= mask;
        }
    }
}

static uint8_t turbovec_group_code_byte(const uint8_t * row, int group, int bits, int dim) {
    const int codes_per_byte = 8 / bits;
    // 3-bit codes take a whole nibble each, two per byte (upstream pack.rs).
    const int field = bits == 3 ? 4 : bits;
    const int dim_start = group * codes_per_byte;
    uint8_t byte = 0;
    for (int c = 0; c < codes_per_byte; ++c) {
        const uint8_t code = turbovec_get_bitplane_code(row, dim_start + c, bits, dim);
        const int shift = (codes_per_byte - 1 - c) * field;
        byte |= static_cast<uint8_t>(code << shift);
    }
    return byte;
}

// Byte-level conversion tables between packed bit-plane rows and
// sequential rows (one byte per group of 8 / bits codes; 3-bit codes in
// 4-bit fields), from upstream pack.rs `build_extract_lut` and
// `build_unpack_lut`.
struct TurboVecPackTables {
    // extract[p][b]: plane p's byte b spread into the sequential bytes of
    // its eight coordinates, little-endian in a u32.
    uint32_t extract[4][256];
    // unpack[b]: for each plane p, a (8 / bits)-bit field at offset
    // p * (8 / bits) holding plane p of the byte's codes, in dim order.
    uint16_t unpack[256];
};

static TurboVecPackTables make_turbovec_pack_tables(int bits) {
    TurboVecPackTables t{};
    const int codes_per_byte = 8 / bits;
    const int field = bits == 3 ? 4 : bits;
    for (int p = 0; p < bits; ++p) {
        for (int b = 0; b < 256; ++b) {
            uint32_t acc = 0;
            for (int j = 0; j < 8; ++j) {
                if ((b & (1 << (7 - j))) != 0) {
                    const int out_byte = j / codes_per_byte;
                    const int shift = (codes_per_byte - 1 - j % codes_per_byte) * field;
                    acc |= UINT32_C(1) << (out_byte * 8 + shift + p);
                }
            }
            t.extract[p][b] = acc;
        }
    }
    for (int b = 0; b < 256; ++b) {
        uint16_t entry = 0;
        for (int c = 0; c < codes_per_byte; ++c) {
            const int code = (b >> ((codes_per_byte - 1 - c) * field)) & ((1 << bits) - 1);
            for (int p = 0; p < bits; ++p) {
                if ((code & (1 << p)) != 0) {
                    entry |= static_cast<uint16_t>(1u << (p * codes_per_byte + (codes_per_byte - 1 - c)));
                }
            }
        }
        t.unpack[b] = entry;
    }
    return t;
}

static const TurboVecPackTables & turbovec_pack_tables(int bits) {
    static const std::array<TurboVecPackTables, 3> tables = {
        make_turbovec_pack_tables(2),
        make_turbovec_pack_tables(3),
        make_turbovec_pack_tables(4),
    };
    return tables[static_cast<size_t>(bits - 2)];
}

static void packed_row_to_seq_fast(const uint8_t * packed, int bits, int dim, uint8_t * seq) {
    const TurboVecPackTables & t = turbovec_pack_tables(bits);
    const size_t bytes_per_plane = static_cast<size_t>(dim) / 8;
    const size_t n_out = static_cast<size_t>(8 / (8 / bits));
    for (size_t c = 0; c < bytes_per_plane; ++c) {
        uint32_t acc = 0;
        for (int p = 0; p < bits; ++p) {
            acc |= t.extract[p][packed[static_cast<size_t>(p) * bytes_per_plane + c]];
        }
        for (size_t o = 0; o < n_out; ++o) {
            seq[c * n_out + o] = static_cast<uint8_t>(acc >> (8 * o));
        }
    }
}

static void seq_row_to_packed_fast(const uint8_t * seq, int bits, int dim, uint8_t * packed) {
    const TurboVecPackTables & t = turbovec_pack_tables(bits);
    const int codes_per_byte = 8 / bits;
    const int groups_per_out = 8 / codes_per_byte;
    const uint16_t field_mask = static_cast<uint16_t>((1u << codes_per_byte) - 1u);
    const size_t bytes_per_plane = static_cast<size_t>(dim) / 8;
    for (size_t ob = 0; ob < bytes_per_plane; ++ob) {
        uint8_t acc[4] = {};
        for (int q = 0; q < groups_per_out; ++q) {
            const uint16_t e = t.unpack[seq[ob * static_cast<size_t>(groups_per_out) + static_cast<size_t>(q)]];
            const int shift = 8 - codes_per_byte * (q + 1);
            for (int p = 0; p < bits; ++p) {
                acc[p] |= static_cast<uint8_t>(((e >> (p * codes_per_byte)) & field_mask) << shift);
            }
        }
        for (int p = 0; p < bits; ++p) {
            packed[static_cast<size_t>(p) * bytes_per_plane + ob] = acc[p];
        }
    }
}

#if defined(_MSC_VER)
__declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
static float turbovec_lut_product(float lhs, float rhs) {
    return lhs * rhs;
}

static void build_turbovec_query_lut(
        const float * rotated_query,
        const float * centroids,
        int bits,
        int dim,
        std::vector<uint8_t> & out_values,
        float & out_scale,
        float & out_bias) {
    const int codes_per_byte = 8 / bits;
    const int codes_per_nibble = codes_per_byte / 2;
    const int n_byte_groups = dim / codes_per_byte;
    const uint16_t code_mask = static_cast<uint16_t>((1u << bits) - 1u);
    constexpr float max_lut = 127.0f;

    out_values.assign(static_cast<size_t>(n_byte_groups) * 32, 0);
    out_scale = 1.0f;
    out_bias = 0.0f;
    std::vector<float> float_vals(static_cast<size_t>(n_byte_groups) * 32, 0.0f);
    std::vector<float> mins(static_cast<size_t>(n_byte_groups) * 2, 0.0f);
    float max_span = 0.0f;

    for (int g = 0; g < n_byte_groups; ++g) {
        const int dim_start = g * codes_per_byte;
        float lo_min = FLT_MAX;
        float lo_max = -FLT_MAX;
        for (uint16_t nibble = 0; nibble < 16; ++nibble) {
            float s = 0.0f;
            for (int c = 0; c < codes_per_nibble; ++c) {
                const int shift = (codes_per_nibble - 1 - c) * bits;
                const uint16_t code = static_cast<uint16_t>((nibble >> shift) & code_mask);
                s += turbovec_lut_product(
                    rotated_query[static_cast<size_t>(dim_start + c)],
                    centroids[code]);
            }
            float_vals[static_cast<size_t>(g) * 32 + nibble] = s;
            lo_min = std::min(lo_min, s);
            lo_max = std::max(lo_max, s);
        }

        float hi_min = FLT_MAX;
        float hi_max = -FLT_MAX;
        for (uint16_t nibble = 0; nibble < 16; ++nibble) {
            float s = 0.0f;
            for (int c = 0; c < codes_per_nibble; ++c) {
                const int shift = (codes_per_nibble - 1 - c) * bits;
                const uint16_t code = static_cast<uint16_t>((nibble >> shift) & code_mask);
                s += turbovec_lut_product(
                    rotated_query[static_cast<size_t>(dim_start + codes_per_nibble + c)],
                    centroids[code]);
            }
            float_vals[static_cast<size_t>(g) * 32 + 16 + nibble] = s;
            hi_min = std::min(hi_min, s);
            hi_max = std::max(hi_max, s);
        }

        mins[static_cast<size_t>(g) * 2] = lo_min;
        mins[static_cast<size_t>(g) * 2 + 1] = hi_min;
        out_bias += lo_min + hi_min;
        max_span = std::max(max_span, lo_max - lo_min);
        max_span = std::max(max_span, hi_max - hi_min);
    }

    // The u8 table is query-magnitude free and the scale carries the
    // magnitude, so any positive span is usable; an absolute floor broke
    // ranking for small queries (upstream #335).
    out_scale = max_span > 0.0f ? max_span / max_lut : 1.0f;
    if (!(out_scale >= FLT_MIN)) {
        out_scale = 1.0f;
    }
    const float inv_scale = 1.0f / out_scale;
    for (int g = 0; g < n_byte_groups; ++g) {
        const float lo_min = mins[static_cast<size_t>(g) * 2];
        const float hi_min = mins[static_cast<size_t>(g) * 2 + 1];
        for (int i = 0; i < 16; ++i) {
            const size_t lo = static_cast<size_t>(g) * 32 + static_cast<size_t>(i);
            const size_t hi = static_cast<size_t>(g) * 32 + 16 + static_cast<size_t>(i);
            out_values[lo] = static_cast<uint8_t>(
                std::max(0.0f, std::min(max_lut, std::round((float_vals[lo] - lo_min) * inv_scale))));
            out_values[hi] = static_cast<uint8_t>(
                std::max(0.0f, std::min(max_lut, std::round((float_vals[hi] - hi_min) * inv_scale))));
        }
    }
}

static float dot_turbovec_lut_row(
        const uint8_t * lut,
        float lut_scale,
        float lut_bias,
        const uint8_t * codes,
        const float * scales,
        int bits,
        int dim) {
    const int codes_per_byte = 8 / bits;
    const int n_byte_groups = dim / codes_per_byte;
    double score = static_cast<double>(lut_bias);
    for (int g = 0; g < n_byte_groups; ++g) {
        const uint8_t byte = turbovec_group_code_byte(codes, g, bits, dim);
        const uint8_t hi = static_cast<uint8_t>(byte >> 4);
        const uint8_t lo = static_cast<uint8_t>(byte & 0x0f);
        score += static_cast<double>(lut_scale) *
            static_cast<double>(lut[static_cast<size_t>(g) * 32 + hi]);
        score += static_cast<double>(lut_scale) *
            static_cast<double>(lut[static_cast<size_t>(g) * 32 + 16 + lo]);
    }
    score *= static_cast<double>(scales[0]);
    return float_score_from_double_local(score);
}

#if GGML_VEC_INDEX_USE_NEON
static float horizontal_sum_local(float32x4_t v) {
#if defined(__aarch64__)
    return vaddvq_f32(v);
#else
    const float32x2_t sum2 = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    return vget_lane_f32(vpadd_f32(sum2, sum2), 0);
#endif
}
#endif

} // namespace

bool turbovec_supported_dim(int dim, bool legacy) {
    return sizeof(size_t) >= 8 &&
        dim > 0 && dim <= (legacy ? kTurboVecLegacyMaxDim : kTurboVecMaxDim) &&
        dim % kTurboVecDimMultiple == 0;
}

bool turbovec_serialized_dim_supported(int dim) {
    return sizeof(size_t) >= 8 &&
        dim > 0 && dim <= kTurboVecMaxSerializedDim && dim % kTurboVecDimMultiple == 0;
}

void turbovec_retain_rotation(int dim, bool legacy) {
    if (legacy) {
        std::lock_guard<std::mutex> lock(turbovec_rotation_cache_mutex());
        TurboVecRotationCacheEntry & entry = turbovec_rotation_cache()[dim];
        ++entry.ref_count;
        if (entry.strong == nullptr) {
            entry.strong = entry.weak.lock();
        }
        return;
    }
    std::lock_guard<std::mutex> lock(turbovec_hadamard_cache_mutex());
    TurboVecHadamardCacheEntry & entry = turbovec_hadamard_cache()[dim];
    ++entry.ref_count;
    if (entry.strong == nullptr) {
        entry.strong = entry.weak.lock();
    }
}

template <typename Entry>
static void turbovec_release_cache_entry(std::unordered_map<int, Entry> & cache, int dim) {
    const auto it = cache.find(dim);
    if (it == cache.end() || it->second.ref_count == 0) {
        return;
    }
    --it->second.ref_count;
    if (it->second.ref_count == 0) {
        it->second.strong.reset();
        if (it->second.weak.expired()) {
            cache.erase(it);
        }
    }
}

void turbovec_release_rotation(int dim, bool legacy) noexcept {
    try {
        if (legacy) {
            std::lock_guard<std::mutex> lock(turbovec_rotation_cache_mutex());
            turbovec_release_cache_entry(turbovec_rotation_cache(), dim);
        } else {
            std::lock_guard<std::mutex> lock(turbovec_hadamard_cache_mutex());
            turbovec_release_cache_entry(turbovec_hadamard_cache(), dim);
        }
    } catch (...) {
    }
}

#ifdef GGML_VEC_INDEX_TEST_HOOKS
static uint64_t turbovec_fnv1a_quantized_double_for_test(uint64_t hash, double value) {
    const int64_t quantized = static_cast<int64_t>(std::llround(value * 1.0e12));
    uint64_t bits = 0;
    std::memcpy(&bits, &quantized, sizeof(bits));
    for (int byte = 0; byte < 8; ++byte) {
        hash ^= static_cast<uint8_t>(bits >> (byte * 8));
        hash *= UINT64_C(0x100000001b3);
    }
    return hash;
}

uint64_t turbovec_ziggurat_table_hash_for_test(void) {
    uint64_t hash = UINT64_C(0xcbf29ce484222325);
    const TurboVecNormalTables & tables = ziggurat_standard_normal_tables();
    for (int i = 0; i < kTurboVecZigguratTableSize; ++i) {
        hash = turbovec_fnv1a_quantized_double_for_test(hash, tables.x[i]);
    }
    for (int i = 0; i < kTurboVecZigguratTableSize; ++i) {
        hash = turbovec_fnv1a_quantized_double_for_test(hash, tables.f[i]);
    }
    return hash;
}

double turbovec_ziggurat_x_for_test(int index) {
    if (index < 0 || index >= kTurboVecZigguratTableSize) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return ziggurat_standard_normal_tables().x[index];
}

double turbovec_ziggurat_f_for_test(int index) {
    if (index < 0 || index >= kTurboVecZigguratTableSize) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return ziggurat_standard_normal_tables().f[index];
}

double turbovec_regularized_beta_for_test(double x, double a, double b) {
    return regularized_beta(x, a, b);
}

double turbovec_inverse_regularized_beta_for_test(double probability, double a) {
    return inverse_regularized_beta(probability, a);
}

size_t turbovec_rotation_cache_bytes_for_test(void) {
    try {
        std::lock_guard<std::mutex> lock(turbovec_rotation_cache_mutex());
        size_t bytes = 0;
        for (const auto & entry : turbovec_rotation_cache()) {
            if (entry.second.strong != nullptr) {
                bytes += entry.second.strong->size() * sizeof(float);
            }
        }
        return bytes;
    } catch (...) {
        return 0;
    }
}

uint64_t turbovec_rotation_hash_for_test(int dim) {
    if (!turbovec_supported_dim(dim, true)) {
        return 0;
    }
    uint64_t hash = UINT64_C(0xcbf29ce484222325);
    const std::shared_ptr<const std::vector<float>> rotation = turbovec_rotation(dim);
    for (float value : *rotation) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        for (int byte = 0; byte < 4; ++byte) {
            hash ^= static_cast<uint8_t>(bits >> (byte * 8));
            hash *= UINT64_C(0x100000001b3);
        }
    }
    return hash;
}

uint64_t turbovec_query_rotation_hash_for_test(
        const float * queries,
        int n_queries,
        int dim) {
    if (queries == nullptr || n_queries < 0 || !turbovec_supported_dim(dim, true)) {
        return 0;
    }
    std::vector<float> rotated(static_cast<size_t>(n_queries) * static_cast<size_t>(dim));
    apply_turbovec_rotation_batch(queries, rotated.data(), n_queries, dim);
    uint64_t hash = UINT64_C(0xcbf29ce484222325);
    for (float value : rotated) {
        uint32_t bits_value = 0;
        std::memcpy(&bits_value, &value, sizeof(bits_value));
        for (int byte = 0; byte < 4; ++byte) {
            hash ^= static_cast<uint8_t>(bits_value >> (byte * 8));
            hash *= UINT64_C(0x100000001b3);
        }
    }
    return hash;
}

double turbovec_query_rotation_max_abs_diff_for_test(
        const float * queries,
        int n_queries,
        int dim) {
    if (queries == nullptr || n_queries < 0 || !turbovec_supported_dim(dim, true)) {
        return std::numeric_limits<double>::infinity();
    }
    const size_t value_count =
        static_cast<size_t>(n_queries) * static_cast<size_t>(dim);
    std::vector<float> fast(value_count);
    std::vector<float> reference(value_count);
    apply_turbovec_query_rotation_batch(queries, fast.data(), n_queries, dim);
    apply_turbovec_rotation_batch(queries, reference.data(), n_queries, dim);
    double max_diff = 0.0;
    for (size_t i = 0; i < value_count; ++i) {
        max_diff = std::max(
            max_diff,
            std::fabs(
                static_cast<double>(fast[i]) -
                static_cast<double>(reference[i])));
    }
    return max_diff;
}

uint64_t turbovec_lut_hash_for_test(
        const float * query,
        const float * tqplus_shift,
        const float * tqplus_scale,
        int bits,
        int n_queries,
        int dim,
        uint32_t * lut_scale_bits,
        uint32_t * lut_bias_bits) {
    if (query == nullptr || tqplus_shift == nullptr || tqplus_scale == nullptr ||
        lut_scale_bits == nullptr || lut_bias_bits == nullptr ||
        (bits != 2 && bits != 4) || n_queries <= 0 || !turbovec_supported_dim(dim, true)) {
        return 0;
    }
    std::vector<float> rotated(static_cast<size_t>(n_queries) * static_cast<size_t>(dim));
    std::vector<float> calibrated(static_cast<size_t>(dim));
    // Parity fixtures use scalar rotation hashes, independent of backend reduction order.
    apply_turbovec_rotation_batch(query, rotated.data(), n_queries, dim);
    double bias_correction = 0.0;
    for (int coordinate = 0; coordinate < dim; ++coordinate) {
        const size_t i = static_cast<size_t>(coordinate);
        calibrated[i] = rotated[i] / tqplus_scale[i];
        bias_correction -=
            static_cast<double>(rotated[i]) *
            static_cast<double>(tqplus_shift[i]);
    }
    std::vector<uint8_t> lut;
    float lut_scale = 1.0f;
    float lut_bias = 0.0f;
    build_turbovec_query_lut(
        calibrated.data(),
        turbovec_codebook(bits, dim, true).centroids.data(),
        bits,
        dim,
        lut,
        lut_scale,
        lut_bias);
    const float bias_correction_f32 = static_cast<float>(bias_correction);
    lut_bias += bias_correction_f32;
    std::memcpy(lut_scale_bits, &lut_scale, sizeof(lut_scale));
    std::memcpy(lut_bias_bits, &lut_bias, sizeof(lut_bias));
    uint64_t hash = UINT64_C(0xcbf29ce484222325);
    for (uint8_t value : lut) {
        hash ^= value;
        hash *= UINT64_C(0x100000001b3);
    }
    return hash;
}

uint64_t turbovec_codebook_hash_for_test(int bits, int dim) {
    if ((bits != 2 && bits != 4) || !turbovec_supported_dim(dim, true)) {
        return 0;
    }
    uint64_t hash = UINT64_C(0xcbf29ce484222325);
    for (float value : turbovec_codebook(bits, dim, true).centroids) {
        uint32_t bits_value = 0;
        std::memcpy(&bits_value, &value, sizeof(bits_value));
        for (int byte = 0; byte < 4; ++byte) {
            hash ^= static_cast<uint8_t>(bits_value >> (byte * 8));
            hash *= UINT64_C(0x100000001b3);
        }
    }
    return hash;
}
#endif

#ifdef GGML_VEC_INDEX_TEST_HOOKS
void turbovec_hadamard_rotate_for_test(const float * src, float * dst, int dim) {
    const std::shared_ptr<const TurboVecHadamardRotation> rotation = turbovec_hadamard_rotation(dim);
    std::vector<float> scratch(static_cast<size_t>(dim));
    apply_turbovec_hadamard(*rotation, src, 1.0f, dst, scratch.data());
}

// The SIMD family a parity hook checks (0: NEON, 1: AVX2, 2: AVX-512), or
// false when this build or CPU cannot run it.
static bool turbovec_simd_for_test(int kernel, TurboVecSimd & simd) {
    const VecIndexCpuFeatures & features = vec_index_cpu_features();
    (void) features;
    switch (kernel) {
#if GGML_VEC_INDEX_USE_NEON
        case 0:
            simd = TurboVecSimd::neon;
            return true;
#endif
#if defined(GGML_VEC_INDEX_HAVE_AVX2_KERNEL)
        case 1:
            simd = TurboVecSimd::avx2;
            return features.avx2;
#endif
#if defined(GGML_VEC_INDEX_HAVE_AVX512_KERNEL)
        case 2:
            simd = TurboVecSimd::avx512;
            return features.avx512f;
#endif
        default:
            return false;
    }
}

static bool turbovec_same_bits_for_test(const float * a, const float * b, size_t n) {
    return std::memcmp(a, b, n * sizeof(float)) == 0;
}

// Deterministic pseudo-random values in [-0.5, 0.5) (upstream's xorshift).
struct TurboVecTestRng {
    uint64_t state;
    uint64_t next() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    }
    float uniform() {
        return static_cast<float>(static_cast<double>(next() >> 11) / 9007199254740992.0) - 0.5f;
    }
};

// 1 when the rotation kernels of `kernel` (Walsh-Hadamard butterfly at every
// block size up to 16384, permutation gather in every multiply mode, the
// fixed-chain norm, and whole forward and inverse rotations) reproduce the
// scalar ones bit for bit; 0 on a mismatch; -1 when unavailable.
int turbovec_rotation_kernel_parity_for_test(int kernel) {
    TurboVecSimd simd = TurboVecSimd::scalar;
    if (!turbovec_simd_for_test(kernel, simd)) {
        return -1;
    }
    const ScopedNearestRounding rounding_guard;
    TurboVecTestRng rng{ UINT64_C(0x9e3779b97f4a7c15) };
    for (int block = 8; block <= 16384; block *= 2) {
        const float inv = 1.0f / std::sqrt(static_cast<float>(block));
        std::vector<float> expect(static_cast<size_t>(block));
        for (float & v : expect) {
            v = rng.uniform();
        }
        std::vector<float> got = expect;
        turbovec_wht_block_scalar(expect.data(), block, inv);
        turbovec_wht_block_with(simd, got.data(), block, inv);
        if (!turbovec_same_bits_for_test(got.data(), expect.data(), got.size())) {
            return 0;
        }
    }
    // Lengths below and between the 8 and 16 lane widths reach every tail.
    for (size_t n : { size_t{ 8 }, size_t{ 12 }, size_t{ 13 }, size_t{ 16 }, size_t{ 24 }, size_t{ 29 },
                      size_t{ 64 }, size_t{ 200 }, size_t{ 1536 } }) {
        std::vector<float> src(n);
        std::vector<float> signs(n);
        std::vector<uint32_t> perm(n);
        for (size_t i = 0; i < n; ++i) {
            src[i] = rng.uniform();
            signs[i] = (rng.next() & 1) != 0 ? -1.0f : 1.0f;
            perm[i] = static_cast<uint32_t>(i);
        }
        for (size_t i = n - 1; i >= 1; --i) {
            std::swap(perm[i], perm[static_cast<size_t>(rng.next() % (i + 1))]);
        }
        for (int mode = 0; mode <= 2; ++mode) {
            std::vector<float> expect(n);
            std::vector<float> got(n, std::numeric_limits<float>::quiet_NaN());
            turbovec_permute_gather_scalar(mode, src.data(), perm.data(), signs.data(), 0.8125f, expect.data(), n);
            turbovec_permute_gather_with(simd, mode, src.data(), perm.data(), signs.data(), 0.8125f, got.data(), n);
            if (!turbovec_same_bits_for_test(got.data(), expect.data(), n)) {
                return 0;
            }
        }
    }
    for (int n = 0; n <= 3072; n = n < 40 ? n + 1 : n * 2) {
        std::vector<float> row(static_cast<size_t>(n));
        for (float & v : row) {
            v = rng.uniform() * 3.0f;
        }
        const float expect = turbovec_norm_sq_scalar(row.data(), n);
        const float got = turbovec_norm_sq_with(simd, row.data(), n);
        if (std::memcmp(&expect, &got, sizeof(float)) != 0) {
            return 0;
        }
    }
    for (int dim : { 8, 24, 64, 128, 200, 768, 1000, 1024, 1536, 3072 }) {
        const std::shared_ptr<const TurboVecHadamardRotation> rotation = turbovec_hadamard_rotation(dim);
        const size_t dim_sz = static_cast<size_t>(dim);
        std::vector<float> src(dim_sz);
        for (float & v : src) {
            v = rng.uniform();
        }
        const float norm = std::sqrt(turbovec_norm_sq_scalar(src.data(), dim));
        std::vector<float> expect(dim_sz);
        std::vector<float> got(dim_sz);
        std::vector<float> scratch(dim_sz);
        // 0 is the scale encode uses for a zero-norm row: it is the input
        // that would expose a signed-zero difference.
        for (float inv : { 1.0f, 1.0f / norm, 0.0f, 0.8125f, -1.0f }) {
            std::fill(scratch.begin(), scratch.end(), std::numeric_limits<float>::quiet_NaN());
            apply_turbovec_hadamard_with(TurboVecSimd::scalar, *rotation, src.data(), inv, expect.data(), scratch.data());
            std::fill(scratch.begin(), scratch.end(), std::numeric_limits<float>::quiet_NaN());
            std::fill(got.begin(), got.end(), std::numeric_limits<float>::quiet_NaN());
            apply_turbovec_hadamard_with(simd, *rotation, src.data(), inv, got.data(), scratch.data());
            if (!turbovec_same_bits_for_test(got.data(), expect.data(), dim_sz)) {
                return 0;
            }
        }
        apply_turbovec_hadamard_inverse_with(TurboVecSimd::scalar, *rotation, src.data(), expect.data(), scratch.data());
        apply_turbovec_hadamard_inverse_with(simd, *rotation, src.data(), got.data(), scratch.data());
        if (!turbovec_same_bits_for_test(got.data(), expect.data(), dim_sz)) {
            return 0;
        }
    }
    return 1;
}

static uint64_t turbovec_fnv1a_for_test(uint64_t hash, const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= UINT64_C(0x100000001b3);
    }
    return hash;
}

int turbovec_index_hashes_for_test(
        const ggml_vec_index_t * idx,
        uint64_t * codes_hash,
        uint64_t * scales_hash,
        uint64_t * shift_hash,
        uint64_t * scale_hash) {
    if (idx == nullptr || !is_turbovec(*idx)) {
        return 0;
    }
    std::shared_lock<std::shared_mutex> lock(idx->mutex);
    constexpr uint64_t basis = UINT64_C(0xcbf29ce484222325);
    std::vector<uint8_t> packed;
    turbovec_rows_packed(*idx, packed);
    *codes_hash = turbovec_fnv1a_for_test(basis, packed.data(), packed.size());
    *scales_hash = turbovec_fnv1a_for_test(
        basis, idx->turbovec_scale.data(), idx->turbovec_scale.size() * sizeof(float));
    *shift_hash = turbovec_fnv1a_for_test(
        basis, idx->turbovec_tqplus_shift.data(), idx->turbovec_tqplus_shift.size() * sizeof(float));
    *scale_hash = turbovec_fnv1a_for_test(
        basis, idx->turbovec_tqplus_scale.data(), idx->turbovec_tqplus_scale.size() * sizeof(float));
    return 1;
}
#endif

void prepare_turbovec(int bits, int dim, bool legacy) {
    if (bits < 2 || bits > 4 || !turbovec_supported_dim(dim, legacy)) {
        return;
    }
    if (legacy) {
        (void) turbovec_rotation(dim);
    } else {
        (void) turbovec_hadamard_rotation(dim);
    }
    (void) turbovec_codebook(bits, dim, legacy);
}

#ifdef GGML_VEC_INDEX_TEST_HOOKS
uint64_t turbovec_blocked_hash_for_test(const ggml_vec_index_t * idx) {
    if (idx == nullptr) {
        return 0;
    }
    try {
        std::shared_lock<std::shared_mutex> lock(idx->mutex);
        if (idx->turbovec_blocked_data.empty()) {
            return 0;
        }
        uint64_t hash = UINT64_C(0xcbf29ce484222325);
        for (uint8_t value : idx->turbovec_blocked_data) {
            hash ^= value;
            hash *= UINT64_C(0x100000001b3);
        }
        return hash;
    } catch (...) {
        return 0;
    }
}

void turbovec_code_bytes_for_test(
        const ggml_vec_index_t * idx, size_t * packed, size_t * blocked, size_t * sign) {
    std::shared_lock<std::shared_mutex> lock(idx->mutex);
    *packed = idx->turbovec_data.capacity();
    *blocked = idx->turbovec_blocked_data.size();
    *sign = idx->turbovec_sign_blocked.size();
}

void turbovec_clear_blocked_for_test(ggml_vec_index_t * idx) {
    if (idx == nullptr) {
        return;
    }
    try {
        std::unique_lock<std::shared_mutex> lock(idx->mutex);
        // The blocked cache may hold the only copy of the codes.
        turbovec_materialize_packed_rows(*idx);
        idx->turbovec_blocked_data.clear();
        idx->turbovec_blocked_n_blocks = 0;
    } catch (...) {
    }
}

int turbovec_layout_for_test(const ggml_vec_index_t * idx) {
    return idx == nullptr ? -1 : static_cast<int>(turbovec_layout(*idx));
}
#endif

size_t turbovec_row_bytes(int bits, size_t dim) {
    return static_cast<size_t>(bits) * (dim / 8);
}

// Byte offset of (group g, lane) inside a block in upstream's vector-major
// layout (pack.rs `vm_byte_index`): four groups of one vector are adjacent,
// so a dot-product kernel loads them as one 32-bit lane.
static size_t vector_major_offset(size_t g, size_t lane) {
    return (g / 4) * 128 + (lane / 16) * 64 + (lane % 16) * 4 + g % 4;
}

// Byte offset of (group g, lane) inside a block in upstream's `vm8` layout
// (pack.rs `vm8_byte_index`): 16 bytes hold two vectors' eight groups, so
// the register is an SMMLA operand as loaded.
static size_t vm8_offset(size_t g, size_t lane) {
    return (g / 8) * 256 + (lane / 2) * 16 + (lane % 2) * 8 + g % 8;
}

bool turbovec_wants_vector_major(int bits, int dim) {
    // dim % 8 == 0 for every 1.x index, so a 4-bit group count is a
    // multiple of four (upstream `vector_major_for` on aarch64).
    (void) dim;
    return bits == 4 && turbovec_cpu_vector_major();
}

bool turbovec_wants_vm8(int bits, int dim) {
    // Upstream `vm8_for`: 8 groups per unit, 4-bit always and 2-bit under
    // the TURBOVEC_2BIT_VM8 opt-in.
    if ((bits != 2 && bits != 4) || !turbovec_cpu_vm8()) {
        return false;
    }
    const size_t n_byte_groups = static_cast<size_t>(dim) / static_cast<size_t>(8 / bits);
    return n_byte_groups % 8 == 0 && (bits == 4 || turbovec_vm8_2bit_enabled());
}

static void write_blocked_block(const uint8_t * seq, size_t n_byte_groups, uint8_t * block, TurboVecLayout layout);

static void repack_turbovec_code_block(
        const uint8_t * packed_codes,
        size_t n_vectors,
        int bits,
        int dim,
        size_t block,
        std::vector<uint8_t> & blocked_codes,
        TurboVecLayout layout) {
    const size_t dim_sz = static_cast<size_t>(dim);
    const size_t codes_per_byte = static_cast<size_t>(8 / bits);
    const size_t n_byte_groups = dim_sz / codes_per_byte;
    const size_t row_bytes = static_cast<size_t>(bits) * (dim_sz / 8);
    const size_t base_vector = block * kTurboVecScoreBlock;
    const size_t block_offset =
        block * n_byte_groups * static_cast<size_t>(kTurboVecScoreBlock);
    // Each lane's sequential row, converted once: rows past the last
    // vector stay zero.
    thread_local std::vector<uint8_t> seq;
    seq.assign(static_cast<size_t>(kTurboVecScoreBlock) * n_byte_groups, 0);
    for (size_t lane = 0; lane < static_cast<size_t>(kTurboVecScoreBlock); ++lane) {
        const size_t vector = base_vector + lane;
        if (vector < n_vectors) {
            packed_row_to_seq_fast(packed_codes + vector * row_bytes, bits, dim, seq.data() + lane * n_byte_groups);
        }
    }
    write_blocked_block(seq.data(), n_byte_groups, blocked_codes.data() + block_offset, layout);
}

// Blocks [first_block, end_block); each writes only its own bytes.
static void repack_turbovec_code_blocks(
        const uint8_t * packed_codes,
        size_t n_vectors,
        int bits,
        int dim,
        size_t first_block,
        size_t end_block,
        std::vector<uint8_t> & blocked_codes,
        TurboVecLayout layout,
        int n_threads) {
    const size_t n_repack = end_block > first_block ? end_block - first_block : 0;
    parallel_for((n_repack + kTurboVecBlocksPerTask - 1) / kTurboVecBlocksPerTask, n_threads, [&](size_t task, size_t) {
        const size_t begin = first_block + task * kTurboVecBlocksPerTask;
        const size_t end = std::min(end_block, begin + kTurboVecBlocksPerTask);
        for (size_t block = begin; block < end; ++block) {
            repack_turbovec_code_block(packed_codes, n_vectors, bits, dim, block, blocked_codes, layout);
        }
    });
}

// Writes one block of 32 sequential rows (lane-major, n_byte_groups bytes
// each) in the blocked layout the scan kernels read.
static void write_blocked_block(const uint8_t * seq, size_t n_byte_groups, uint8_t * block, TurboVecLayout layout) {
    if (layout == kTurboVecLayoutVm8) {
        // Each lane's eight groups of a unit are contiguous in both.
        for (size_t group = 0; group < n_byte_groups; group += 8) {
            for (size_t lane = 0; lane < static_cast<size_t>(kTurboVecScoreBlock); ++lane) {
                std::memcpy(block + vm8_offset(group, lane), seq + lane * n_byte_groups + group, 8);
            }
        }
        return;
    }
    if (layout == kTurboVecLayoutVm) {
        for (size_t group = 0; group < n_byte_groups; ++group) {
            for (size_t lane = 0; lane < static_cast<size_t>(kTurboVecScoreBlock); ++lane) {
                block[vector_major_offset(group, lane)] = seq[lane * n_byte_groups + group];
            }
        }
        return;
    }
    for (size_t group = 0; group < n_byte_groups; ++group) {
        uint8_t * out = block + group * kTurboVecScoreBlock;
#if defined(__x86_64__) || defined(_M_X64)
        static constexpr size_t perm[16] = {
            0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15,
        };
        for (size_t j = 0; j < 16; ++j) {
            const uint8_t byte_a = seq[perm[j] * n_byte_groups + group];
            const uint8_t byte_b = seq[(perm[j] + 16) * n_byte_groups + group];
            out[j] = static_cast<uint8_t>((byte_a >> 4) | ((byte_b >> 4) << 4));
            out[16 + j] = static_cast<uint8_t>((byte_a & 0x0f) | ((byte_b & 0x0f) << 4));
        }
#else
        for (size_t lane = 0; lane < static_cast<size_t>(kTurboVecScoreBlock); ++lane) {
            out[lane] = seq[lane * n_byte_groups + group];
        }
#endif
    }
}

// One lane's code byte for a byte group of a blocked block, in either
// in-memory layout: sequential lanes, or x86's perm0 nibble interleave
// (see repack_turbovec_code_block).
static uint8_t blocked_lane_byte(const uint8_t * group_bytes, size_t lane);
static void set_blocked_lane_byte(uint8_t * group_bytes, size_t lane, uint8_t value);

// (group, lane) of a whole block, in the block's layout.
static uint8_t block_lane_byte(const uint8_t * block, size_t g, size_t lane, TurboVecLayout layout) {
    switch (layout) {
        case kTurboVecLayoutVm8:
            return block[vm8_offset(g, lane)];
        case kTurboVecLayoutVm:
            return block[vector_major_offset(g, lane)];
        case kTurboVecLayoutSeq:
            break;
    }
    return blocked_lane_byte(block + g * static_cast<size_t>(kTurboVecScoreBlock), lane);
}

static void set_block_lane_byte(uint8_t * block, size_t g, size_t lane, TurboVecLayout layout, uint8_t value) {
    switch (layout) {
        case kTurboVecLayoutVm8:
            block[vm8_offset(g, lane)] = value;
            return;
        case kTurboVecLayoutVm:
            block[vector_major_offset(g, lane)] = value;
            return;
        case kTurboVecLayoutSeq:
            break;
    }
    set_blocked_lane_byte(block + g * static_cast<size_t>(kTurboVecScoreBlock), lane, value);
}

// One lane's sequential row out of a block, in any layout.
static void block_lane_row(const uint8_t * block, size_t n_byte_groups, size_t lane, TurboVecLayout layout, uint8_t * seq) {
    if (layout == kTurboVecLayoutVm8) {
        for (size_t g = 0; g < n_byte_groups; g += 8) {
            std::memcpy(seq + g, block + vm8_offset(g, lane), 8);
        }
        return;
    }
    for (size_t g = 0; g < n_byte_groups; ++g) {
        seq[g] = block_lane_byte(block, g, lane, layout);
    }
}

// The inverse: one lane's row into a block (zeros clear the lane).
static void set_block_lane_row(uint8_t * block, size_t n_byte_groups, size_t lane, TurboVecLayout layout, const uint8_t * seq) {
    if (layout == kTurboVecLayoutVm8) {
        for (size_t g = 0; g < n_byte_groups; g += 8) {
            std::memcpy(block + vm8_offset(g, lane), seq + g, 8);
        }
        return;
    }
    for (size_t g = 0; g < n_byte_groups; ++g) {
        set_block_lane_byte(block, g, lane, layout, seq[g]);
    }
}

static uint8_t blocked_lane_byte(const uint8_t * group_bytes, size_t lane) {
#if defined(__x86_64__) || defined(_M_X64)
    const size_t p = lane % 16;
    const size_t j = p < 8 ? 2 * p : 2 * (p - 8) + 1;
    const int shift = lane < 16 ? 0 : 4;
    return static_cast<uint8_t>(
        (((group_bytes[j] >> shift) & 0x0f) << 4) | ((group_bytes[16 + j] >> shift) & 0x0f));
#else
    return group_bytes[lane];
#endif
}

static void set_blocked_lane_byte(uint8_t * group_bytes, size_t lane, uint8_t value) {
#if defined(__x86_64__) || defined(_M_X64)
    const size_t p = lane % 16;
    const size_t j = p < 8 ? 2 * p : 2 * (p - 8) + 1;
    const int shift = lane < 16 ? 0 : 4;
    const uint8_t keep = static_cast<uint8_t>(0xf0u >> shift);
    group_bytes[j] = static_cast<uint8_t>((group_bytes[j] & keep) | ((value >> 4) << shift));
    group_bytes[16 + j] = static_cast<uint8_t>((group_bytes[16 + j] & keep) | ((value & 0x0f) << shift));
#else
    group_bytes[lane] = value;
#endif
}

static int8_t turbovec_round_i8(float value) {
    // Rust `(x).round().clamp(-127.0, 127.0) as i8`: half away from zero.
    return static_cast<int8_t>(std::max(-127.0f, std::min(127.0f, std::round(value))));
}

// The query rounded to int8 under one scale (upstream `build_permute_dot`).
static float turbovec_query_weights(const float * query, int dim, std::vector<int8_t> & weights) {
    float qmax = 0.0f;
    for (int d = 0; d < dim; ++d) {
        qmax = std::max(qmax, std::fabs(query[d]));
    }
    float qs = qmax / 127.0f;
    float inv_qs = 1.0f;
    if (qs >= FLT_MIN) {
        inv_qs = 1.0f / qs;
    } else {
        qs = 1.0f;
    }
    weights.resize(static_cast<size_t>(dim));
    for (int d = 0; d < dim; ++d) {
        weights[static_cast<size_t>(d)] = turbovec_round_i8(query[d] * inv_qs);
    }
    return qs;
}

void build_turbovec_permute_dot(const float * query, int dim, TurboVecPermuteDot & out, bool vm8) {
    // Upstream search.rs `build_permute_dot`: the codebook and the query
    // each rounded to int8 under one scale (round is half away from zero).
    const TurboVecCodebook & codebook = turbovec_codebook(4, dim, false);
    float cmax = 0.0f;
    for (float c : codebook.centroids) {
        cmax = std::max(cmax, std::fabs(c));
    }
    const float cs = cmax > 0.0f ? cmax / 127.0f : 1.0f;
    for (int i = 0; i < 16; ++i) {
        out.levels[i] = turbovec_round_i8(codebook.centroids[i] / cs);
    }
    std::fill(std::begin(out.levels2), std::end(out.levels2), int8_t{ 0 });
    const float qs = turbovec_query_weights(query, dim, out.weights);
    out.scale = cs * qs;
    out.bias = 0.0f;
    // The lone-query weights over the vm8 layout (upstream's
    // `build_smmla_a_vm8::<2>(&[pd, pd])`): per octet of byte groups, the
    // eight even (high-nibble) dims twice, then the eight odd ones twice.
    const size_t n_byte_groups = static_cast<size_t>(dim) / 2;
    out.vm8_weights.clear();
    if (vm8 && n_byte_groups % 8 == 0) {
        out.vm8_weights.resize(n_byte_groups * 4);
        for (size_t oct = 0; oct < n_byte_groups / 8; ++oct) {
            int8_t * a = out.vm8_weights.data() + oct * 32;
            for (size_t j = 0; j < 8; ++j) {
                const size_t g = oct * 8 + j;
                a[j] = a[8 + j] = out.weights[2 * g];
                a[16 + j] = a[24 + j] = out.weights[2 * g + 1];
            }
        }
    }
}

void build_turbovec_permute_dot_2bit(const float * query, int dim, TurboVecPermuteDot & out) {
    // Upstream search.rs `build_permute_dot_2bit` (H72): the four levels
    // as int8 in two 16-entry tables, one per nibble field pair, and the
    // query rounded to int8 in plain dimension order.
    const TurboVecCodebook & codebook = turbovec_codebook(2, dim, false);
    float cmax = 0.0f;
    for (int i = 0; i < 4; ++i) {
        cmax = std::max(cmax, std::fabs(codebook.centroids[static_cast<size_t>(i)]));
    }
    const float cs = cmax > 0.0f ? cmax / 127.0f : 1.0f;
    int8_t lv[4];
    for (int i = 0; i < 4; ++i) {
        lv[i] = turbovec_round_i8(codebook.centroids[static_cast<size_t>(i)] / cs);
    }
    for (int i = 0; i < 16; ++i) {
        out.levels[i] = lv[i & 3];
        out.levels2[i] = lv[(i >> 2) & 3];
    }
    const float qs = turbovec_query_weights(query, dim, out.weights);
    out.scale = cs * qs;
    out.bias = 0.0f;
    out.vm8_weights.clear();
}

#if GGML_VEC_INDEX_USE_NEON && (defined(__clang__) || defined(__GNUC__))
// Clang spells the AArch64 extensions "dotprod" and "i8mm". GCC must match
// the target its arm_neon.h declares each intrinsic under,
// "arch=armv8.2-a+dotprod" and "arch=armv8.2-a+i8mm": GCC 12 rejects a bare
// "+dotprod" or "+i8mm" on an armv8-a baseline as a target mismatch when it
// inlines vdotq_s32 or vmmlaq_s32.
#if defined(__clang__)
#define GGML_VEC_INDEX_TARGET_DOTPROD __attribute__((target("dotprod")))
#define GGML_VEC_INDEX_TARGET_I8MM __attribute__((target("i8mm")))
#else
#define GGML_VEC_INDEX_TARGET_DOTPROD __attribute__((target("arch=armv8.2-a+dotprod")))
#define GGML_VEC_INDEX_TARGET_I8MM __attribute__((target("arch=armv8.2-a+i8mm")))
#endif

// SDOT over the sequential blocked layout: four byte groups of sixteen
// lanes are transposed in registers so each 32-bit lane holds one vector's
// four codes, and one SDOT adds four weight * level products per vector.
// Integer sums, so bit-identical to the scalar kernel below.
GGML_VEC_INDEX_TARGET_DOTPROD
static void permute_dot_block_sums_sdot(
        const TurboVecPermuteDot & pd,
        const uint8_t * block,
        size_t n_byte_groups,
        bool vector_major,
        int32_t * sums) {
    const int8x16_t levels = vld1q_s8(pd.levels);
    const uint8x16_t nibble = vdupq_n_u8(0x0f);
    int32x4_t acc[8];
    for (int32x4_t & a : acc) {
        a = vdupq_n_s32(0);
    }
    const int8_t * w = pd.weights.data();
    for (size_t g = 0; g < n_byte_groups; g += 4) {
        const int8_t even[4] = { w[2 * g], w[2 * g + 2], w[2 * g + 4], w[2 * g + 6] };
        const int8_t odd[4] = { w[2 * g + 1], w[2 * g + 3], w[2 * g + 5], w[2 * g + 7] };
        int32_t even_word = 0;
        int32_t odd_word = 0;
        std::memcpy(&even_word, even, 4);
        std::memcpy(&odd_word, odd, 4);
        const int8x16_t w_even = vreinterpretq_s8_s32(vdupq_n_s32(even_word));
        const int8x16_t w_odd = vreinterpretq_s8_s32(vdupq_n_s32(odd_word));
        if (vector_major) {
            // The block already holds each vector's four codes together.
            const uint8_t * unit = block + (g / 4) * 128;
            for (int i = 0; i < 8; ++i) {
                const uint8x16_t quad = vld1q_u8(unit + 16 * i);
                acc[i] = vdotq_s32(acc[i], vqtbl1q_s8(levels, vshrq_n_u8(quad, 4)), w_even);
                acc[i] = vdotq_s32(acc[i], vqtbl1q_s8(levels, vandq_u8(quad, nibble)), w_odd);
            }
            continue;
        }
        for (size_t half = 0; half < 2; ++half) {
            const uint8_t * base = block + g * 32 + half * 16;
            const uint8x16_t x0 = vld1q_u8(base);
            const uint8x16_t x1 = vld1q_u8(base + 32);
            const uint8x16_t x2 = vld1q_u8(base + 64);
            const uint8x16_t x3 = vld1q_u8(base + 96);
            const uint8x16x2_t z01 = vzipq_u8(x0, x1);
            const uint8x16x2_t z23 = vzipq_u8(x2, x3);
            const uint16x8x2_t q0 = vzipq_u16(vreinterpretq_u16_u8(z01.val[0]), vreinterpretq_u16_u8(z23.val[0]));
            const uint16x8x2_t q1 = vzipq_u16(vreinterpretq_u16_u8(z01.val[1]), vreinterpretq_u16_u8(z23.val[1]));
            const uint8x16_t quads[4] = {
                vreinterpretq_u8_u16(q0.val[0]), vreinterpretq_u8_u16(q0.val[1]),
                vreinterpretq_u8_u16(q1.val[0]), vreinterpretq_u8_u16(q1.val[1]),
            };
            for (int i = 0; i < 4; ++i) {
                int32x4_t & a = acc[half * 4 + static_cast<size_t>(i)];
                a = vdotq_s32(a, vqtbl1q_s8(levels, vshrq_n_u8(quads[i], 4)), w_even);
                a = vdotq_s32(a, vqtbl1q_s8(levels, vandq_u8(quads[i], nibble)), w_odd);
            }
        }
    }
    for (int i = 0; i < 8; ++i) {
        vst1q_s32(sums + 4 * i, acc[i]);
    }
}

GGML_VEC_INDEX_TARGET_DOTPROD
static void permute_dot_block_sums_sdot_4(
        const TurboVecPermuteDot * const * pds,
        const uint8_t * block,
        size_t n_byte_groups,
        int32_t (*sums)[kTurboVecScoreBlock]) {
    const int8x16_t levels = vld1q_s8(pds[0]->levels);
    const uint8x16_t nibble = vdupq_n_u8(0x0f);
    int32x4_t acc[4][8];
    for (auto & query_acc : acc) {
        for (int32x4_t & a : query_acc) {
            a = vdupq_n_s32(0);
        }
    }
    for (size_t g = 0; g < n_byte_groups; g += 4) {
        int8x16_t w_even[4];
        int8x16_t w_odd[4];
        for (int q = 0; q < 4; ++q) {
            const int8_t * w = pds[q]->weights.data();
            const int8_t even[4] = { w[2 * g], w[2 * g + 2], w[2 * g + 4], w[2 * g + 6] };
            const int8_t odd[4] = { w[2 * g + 1], w[2 * g + 3], w[2 * g + 5], w[2 * g + 7] };
            int32_t even_word = 0;
            int32_t odd_word = 0;
            std::memcpy(&even_word, even, 4);
            std::memcpy(&odd_word, odd, 4);
            w_even[q] = vreinterpretq_s8_s32(vdupq_n_s32(even_word));
            w_odd[q] = vreinterpretq_s8_s32(vdupq_n_s32(odd_word));
        }
        const uint8_t * unit = block + (g / 4) * 128;
        for (int i = 0; i < 8; ++i) {
            const uint8x16_t quad = vld1q_u8(unit + 16 * i);
            const int8x16_t hi = vqtbl1q_s8(levels, vshrq_n_u8(quad, 4));
            const int8x16_t lo = vqtbl1q_s8(levels, vandq_u8(quad, nibble));
            for (int q = 0; q < 4; ++q) {
                acc[q][i] = vdotq_s32(acc[q][i], hi, w_even[q]);
                acc[q][i] = vdotq_s32(acc[q][i], lo, w_odd[q]);
            }
        }
    }
    for (int q = 0; q < 4; ++q) {
        for (int i = 0; i < 8; ++i) {
            vst1q_s32(sums[q] + 4 * i, acc[q][i]);
        }
    }
}

// A batch reads each block eight times over (once per part), so the
// stream is prefetched 32 units (8 KB) ahead, as upstream's vm8 kernel does
// (H67); measured 0.78 -> 0.71 ms per query at 100k x 768. Never past the
// cache's end.
static inline void turbovec_prefetch_ahead(const uint8_t * block, size_t offset, size_t bytes_after) {
    constexpr size_t ahead = 32 * 256;
    if (offset + ahead + 64 <= bytes_after) {
        __builtin_prefetch(block + offset + ahead);
    }
}

// SMMLA computes a 2x2 tile: lanes 0/1 are row 0 (the even query of a pair)
// against columns 0/1 (two vectors), lanes 2/3 row 1. Four consecutive
// block lanes of one query come from the matching halves of two tiles.
GGML_VEC_INDEX_TARGET_I8MM
static inline void smmla_store_pair(int32x4_t x, int32x4_t y, int32_t * even_query, int32_t * odd_query) {
    vst1q_s32(even_query, vcombine_s32(vget_low_s32(x), vget_low_s32(y)));
    vst1q_s32(odd_query, vcombine_s32(vget_high_s32(x), vget_high_s32(y)));
}

// One query over a vm8 block. Upstream's `score_block_vm8_single` feeds
// the query to SMMLA as a duplicated pair, so half of every SMMLA's
// products are thrown away; SDOT issues the same number of instructions
// over the same registers with none wasted, and measured 1.21 ms against
// SMMLA's 1.50 per query on an Apple M4 (100k x 768). The integers are the
// same. Each 32-bit lane sums four groups of one vector, so two lanes per
// vector are added at the end; all 16 vector pairs keep their own
// accumulator, and codes are loaded four at a time to cap register
// pressure at 16 + 4.
GGML_VEC_INDEX_TARGET_DOTPROD
static void permute_dot_block_sums_sdot_vm8_single(
        const TurboVecPermuteDot & pd,
        const uint8_t * block,
        size_t n_byte_groups,
        int32_t * sums) {
    const int8x16_t levels = vld1q_s8(pd.levels);
    const uint8x16_t nibble = vdupq_n_u8(0x0f);
    const int8_t * a = pd.vm8_weights.data();
    int32x4_t acc[16];
    for (int32x4_t & v : acc) {
        v = vdupq_n_s32(0);
    }
    for (size_t oct = 0; oct < n_byte_groups / 8; ++oct) {
        const int8x16_t ae = vld1q_s8(a + oct * 32);
        const int8x16_t ao = vld1q_s8(a + oct * 32 + 16);
        const uint8_t * unit = block + oct * 256;
        for (int g = 0; g < 4; ++g) {
            uint8x16_t c[4];
            for (int j = 0; j < 4; ++j) {
                c[j] = vld1q_u8(unit + (g * 4 + j) * 16);
            }
            for (int j = 0; j < 4; ++j) {
                const int r = g * 4 + j;
                const int8x16_t bo = vqtbl1q_s8(levels, vandq_u8(c[j], nibble));
                const int8x16_t be = vqtbl1q_s8(levels, vshrq_n_u8(c[j], 4));
                acc[r] = vdotq_s32(acc[r], be, ae);
                acc[r] = vdotq_s32(acc[r], bo, ao);
            }
        }
    }
    for (int i = 0; i < 8; ++i) {
        vst1q_s32(sums + 4 * i, vpaddq_s32(acc[2 * i], acc[2 * i + 1]));
    }
}

// NP query pairs over a vm8 block (upstream `score_block_smmla_vm8`): the
// TBL output is the B operand as loaded. Eighth blocks keep 2 * NP
// accumulators and 2 * NP A operands in registers.
template <int NP>
GGML_VEC_INDEX_TARGET_I8MM
static void permute_dot_block_sums_smmla_vm8(
        const int8_t * levels_table,
        const int8_t * a_buf,
        const uint8_t * block,
        size_t n_byte_groups,
        size_t bytes_after,
        int32_t (*sums)[kTurboVecScoreBlock]) {
    const int8x16_t levels = vld1q_s8(levels_table);
    const uint8x16_t nibble = vdupq_n_u8(0x0f);
    const size_t octs = n_byte_groups / 8;
    for (int part = 0; part < 8; ++part) {
        int32x4_t acc[NP][2];
        for (int p = 0; p < NP; ++p) {
            acc[p][0] = vdupq_n_s32(0);
            acc[p][1] = vdupq_n_s32(0);
        }
        for (size_t oct = 0; oct < octs; ++oct) {
            const int8_t * ap = a_buf + oct * NP * 32;
            turbovec_prefetch_ahead(block, oct * 256, bytes_after);
            int8x16_t ae[NP];
            int8x16_t ao[NP];
            for (int p = 0; p < NP; ++p) {
                ae[p] = vld1q_s8(ap + p * 32);
                ao[p] = vld1q_s8(ap + p * 32 + 16);
            }
            for (int r = 0; r < 2; ++r) {
                const uint8x16_t c = vld1q_u8(block + oct * 256 + (part * 2 + r) * 16);
                const int8x16_t bo = vqtbl1q_s8(levels, vandq_u8(c, nibble));
                const int8x16_t be = vqtbl1q_s8(levels, vshrq_n_u8(c, 4));
                for (int p = 0; p < NP; ++p) {
                    acc[p][r] = vmmlaq_s32(acc[p][r], ae[p], be);
                    acc[p][r] = vmmlaq_s32(acc[p][r], ao[p], bo);
                }
            }
        }
        for (int p = 0; p < NP; ++p) {
            smmla_store_pair(acc[p][0], acc[p][1], sums[2 * p] + part * 4, sums[2 * p + 1] + part * 4);
        }
    }
}

// NP query pairs over a vector-major (4-group) block (upstream
// `score_block_permute_smmla_neon`): two ZIPs per code register regroup
// the TBL outputs into two-vector, eight-dimension B operands.
template <int NP>
GGML_VEC_INDEX_TARGET_I8MM
static void permute_dot_block_sums_smmla_vm(
        const int8_t * levels_table,
        const int8_t * a_buf,
        const uint8_t * block,
        size_t n_byte_groups,
        size_t bytes_after,
        int32_t (*sums)[kTurboVecScoreBlock]) {
    const int8x16_t levels = vld1q_s8(levels_table);
    const uint8x16_t nibble = vdupq_n_u8(0x0f);
    const size_t quads = n_byte_groups / 4;
    for (int part = 0; part < 4; ++part) {
        int32x4_t acc[NP][4];
        for (int p = 0; p < NP; ++p) {
            for (int w = 0; w < 4; ++w) {
                acc[p][w] = vdupq_n_s32(0);
            }
        }
        for (size_t q4 = 0; q4 < quads; ++q4) {
            const int8_t * ap = a_buf + q4 * NP * 16;
            turbovec_prefetch_ahead(block, q4 * 128, bytes_after);
            int8x16_t a[NP];
            for (int p = 0; p < NP; ++p) {
                a[p] = vld1q_s8(ap + p * 16);
            }
            for (int i = 0; i < 2; ++i) {
                const uint8x16_t c = vld1q_u8(block + q4 * 128 + (part * 2 + i) * 16);
                const int8x16_t vlo = vqtbl1q_s8(levels, vandq_u8(c, nibble));
                const int8x16_t vhi = vqtbl1q_s8(levels, vshrq_n_u8(c, 4));
                const int8x16_t b0 = vzip1q_s8(vhi, vlo);
                const int8x16_t b1 = vzip2q_s8(vhi, vlo);
                for (int p = 0; p < NP; ++p) {
                    acc[p][i * 2] = vmmlaq_s32(acc[p][i * 2], a[p], b0);
                    acc[p][i * 2 + 1] = vmmlaq_s32(acc[p][i * 2 + 1], a[p], b1);
                }
            }
        }
        for (int p = 0; p < NP; ++p) {
            for (int h = 0; h < 2; ++h) {
                smmla_store_pair(
                    acc[p][h * 2], acc[p][h * 2 + 1], sums[2 * p] + part * 8 + h * 4, sums[2 * p + 1] + part * 8 + h * 4);
            }
        }
    }
}

// NP query pairs over a 2-bit vm8 block (upstream
// `score_block_smmla_vm8_2bit`, H72). A byte holds dims 4g..4g+3 in bits
// 7:6, 5:4, 3:2, 1:0; two tables turn each nibble into two level
// registers, each already a B operand.
template <int NP>
GGML_VEC_INDEX_TARGET_I8MM
static void permute_dot_block_sums_smmla_vm8_2bit(
        const int8_t * levels_table,
        const int8_t * levels2_table,
        const int8_t * a_buf,
        const uint8_t * block,
        size_t n_byte_groups,
        size_t bytes_after,
        int32_t (*sums)[kTurboVecScoreBlock]) {
    const int8x16_t ta = vld1q_s8(levels_table);
    const int8x16_t tb = vld1q_s8(levels2_table);
    const uint8x16_t nibble = vdupq_n_u8(0x0f);
    const size_t octs = n_byte_groups / 8;
    for (int part = 0; part < 8; ++part) {
        int32x4_t acc[NP][2];
        for (int p = 0; p < NP; ++p) {
            acc[p][0] = vdupq_n_s32(0);
            acc[p][1] = vdupq_n_s32(0);
        }
        for (size_t oct = 0; oct < octs; ++oct) {
            const int8_t * ap = a_buf + oct * NP * 64;
            turbovec_prefetch_ahead(block, oct * 256, bytes_after);
            for (int r = 0; r < 2; ++r) {
                const uint8x16_t c = vld1q_u8(block + oct * 256 + (part * 2 + r) * 16);
                const uint8x16_t lo = vandq_u8(c, nibble);
                const uint8x16_t hi = vshrq_n_u8(c, 4);
                const int8x16_t f0 = vqtbl1q_s8(tb, hi);
                const int8x16_t f1 = vqtbl1q_s8(ta, hi);
                const int8x16_t f2 = vqtbl1q_s8(tb, lo);
                const int8x16_t f3 = vqtbl1q_s8(ta, lo);
                for (int p = 0; p < NP; ++p) {
                    const int8_t * a = ap + p * 64;
                    acc[p][r] = vmmlaq_s32(acc[p][r], vld1q_s8(a), f0);
                    acc[p][r] = vmmlaq_s32(acc[p][r], vld1q_s8(a + 16), f1);
                    acc[p][r] = vmmlaq_s32(acc[p][r], vld1q_s8(a + 32), f2);
                    acc[p][r] = vmmlaq_s32(acc[p][r], vld1q_s8(a + 48), f3);
                }
            }
        }
        for (int p = 0; p < NP; ++p) {
            smmla_store_pair(acc[p][0], acc[p][1], sums[2 * p] + part * 4, sums[2 * p + 1] + part * 4);
        }
    }
}
#endif

// Integer permute-dot sums of one query over one block, in any layout:
// the reference the SIMD kernels must match exactly.
static void permute_dot_block_sums_scalar(
        const TurboVecPermuteDot & pd,
        const uint8_t * block,
        size_t n_byte_groups,
        int bits,
        TurboVecLayout layout,
        size_t valid,
        int32_t * sums) {
    const int8_t * w = pd.weights.data();
    for (size_t lane = 0; lane < static_cast<size_t>(kTurboVecScoreBlock); ++lane) {
        sums[lane] = 0;
    }
    for (size_t g = 0; g < n_byte_groups; ++g) {
        for (size_t lane = 0; lane < valid; ++lane) {
            const uint8_t byte = block_lane_byte(block, g, lane, layout);
            if (bits == 4) {
                sums[lane] += w[2 * g] * pd.levels[byte >> 4] + w[2 * g + 1] * pd.levels[byte & 0x0f];
            } else {
                // 2 bits: levels[c] is level c for c < 4; dims 4g.. high first.
                sums[lane] += w[4 * g] * pd.levels[(byte >> 6) & 3] + w[4 * g + 1] * pd.levels[(byte >> 4) & 3] +
                              w[4 * g + 2] * pd.levels[(byte >> 2) & 3] + w[4 * g + 3] * pd.levels[byte & 3];
            }
        }
    }
}

// fma(sum, scale, bias) times the vector's scale, as every upstream
// permute-dot kernel finishes; padding lanes are -inf.
static void permute_dot_finish(
        const TurboVecPermuteDot & pd,
        const int32_t * sums,
        const float * vector_scales,
        size_t base,
        size_t valid,
        float * out_scores) {
    for (size_t lane = 0; lane < static_cast<size_t>(kTurboVecScoreBlock); ++lane) {
        out_scores[lane] = lane < valid ?
            std::fma(static_cast<float>(sums[lane]), pd.scale, pd.bias) * vector_scales[base + lane] :
            -std::numeric_limits<float>::infinity();
    }
}

void score_turbovec_permute_dot_block(
        const TurboVecPermuteDot & pd,
        const uint8_t * blocked_codes,
        const float * vector_scales,
        size_t block_index,
        size_t n_vectors,
        int dim,
        TurboVecLayout layout,
        float * out_scores) {
    // A 4-bit byte group holds dim 2g in its high nibble and 2g + 1 in its
    // low one. The sums are exact integers, so they match upstream's SDOT,
    // SMMLA and VNNI kernels whatever the order of accumulation.
    const size_t n_byte_groups = static_cast<size_t>(dim) / 2;
    const size_t lane_size = static_cast<size_t>(kTurboVecScoreBlock);
    const size_t base = block_index * lane_size;
    const size_t valid = std::min(lane_size, n_vectors - base);
    const uint8_t * block = blocked_codes + block_index * n_byte_groups * lane_size;
    int32_t sums[kTurboVecScoreBlock] = {};
#if GGML_VEC_INDEX_USE_NEON && (defined(__clang__) || defined(__GNUC__))
    // SDOT for one query in every layout (see the vm8 kernel for why not
    // SMMLA). dim % 8 == 0 makes the group count a multiple of four.
    const bool has_dot_product = turbovec_cpu_dot_product();
    if (layout == kTurboVecLayoutVm8 && has_dot_product && !pd.vm8_weights.empty()) {
        permute_dot_block_sums_sdot_vm8_single(pd, block, n_byte_groups, sums);
    } else if (layout != kTurboVecLayoutVm8 && has_dot_product) {
        permute_dot_block_sums_sdot(pd, block, n_byte_groups, layout == kTurboVecLayoutVm, sums);
    } else
#endif
    {
        permute_dot_block_sums_scalar(pd, block, n_byte_groups, 4, layout, valid, sums);
    }
    permute_dot_finish(pd, sums, vector_scales, base, valid, out_scores);
}

void prepare_turbovec_permute_dot_batch(TurboVecPermuteDotBatch & batch, int dim) {
    // The batch's A operands, reshaped once and reused by every block
    // (upstream `build_smmla_a`, `build_smmla_a_vm8`,
    // `build_smmla_a_vm8_2bit`). Only the SMMLA kernels read them.
    batch.smmla_a.clear();
    const int n = batch.n;
    const int pairs = n / 2;
    const size_t n_byte_groups = static_cast<size_t>(dim) / static_cast<size_t>(8 / batch.bits);
    if (!turbovec_cpu_i8mm() || (n != 4 && n != 8 && n != 12)) {
        return;
    }
    if (batch.layout == kTurboVecLayoutVm8 && n_byte_groups % 8 == 0) {
        const size_t octs = n_byte_groups / 8;
        if (batch.bits == 2) {
            batch.smmla_a.resize(octs * static_cast<size_t>(pairs) * 64);
            for (size_t oct = 0; oct < octs; ++oct) {
                for (int p = 0; p < pairs; ++p) {
                    int8_t * dst = batch.smmla_a.data() + (oct * static_cast<size_t>(pairs) + static_cast<size_t>(p)) * 64;
                    for (int r = 0; r < 2; ++r) {
                        const int8_t * w = batch.pds[2 * p + r]->weights.data();
                        for (size_t k = 0; k < 4; ++k) {
                            for (size_t j = 0; j < 8; ++j) {
                                dst[k * 16 + static_cast<size_t>(r) * 8 + j] = w[4 * (8 * oct + j) + k];
                            }
                        }
                    }
                }
            }
        } else {
            batch.smmla_a.resize(octs * static_cast<size_t>(pairs) * 32);
            for (size_t oct = 0; oct < octs; ++oct) {
                for (int p = 0; p < pairs; ++p) {
                    int8_t * dst = batch.smmla_a.data() + (oct * static_cast<size_t>(pairs) + static_cast<size_t>(p)) * 32;
                    for (int r = 0; r < 2; ++r) {
                        const int8_t * w = batch.pds[2 * p + r]->weights.data();
                        for (size_t j = 0; j < 8; ++j) {
                            const size_t g = 8 * oct + j;
                            dst[static_cast<size_t>(r) * 8 + j] = w[2 * g];
                            dst[16 + static_cast<size_t>(r) * 8 + j] = w[2 * g + 1];
                        }
                    }
                }
            }
        }
        return;
    }
    if (batch.layout == kTurboVecLayoutVm && batch.bits == 4) {
        // Two queries' eight consecutive dims per byte-group quad.
        const size_t quads = n_byte_groups / 4;
        batch.smmla_a.resize(quads * static_cast<size_t>(pairs) * 16);
        for (size_t q4 = 0; q4 < quads; ++q4) {
            for (int p = 0; p < pairs; ++p) {
                int8_t * dst = batch.smmla_a.data() + (q4 * static_cast<size_t>(pairs) + static_cast<size_t>(p)) * 16;
                for (int r = 0; r < 2; ++r) {
                    std::memcpy(dst + r * 8, batch.pds[2 * p + r]->weights.data() + 8 * q4, 8);
                }
            }
        }
    }
}

void score_turbovec_permute_dot_batch_block(
        const TurboVecPermuteDotBatch & batch,
        const uint8_t * blocked_codes,
        const float * vector_scales,
        size_t block_index,
        size_t n_vectors,
        int dim,
        float * const * out_scores) {
    const int n = batch.n;
    const size_t n_byte_groups = static_cast<size_t>(dim) / static_cast<size_t>(8 / batch.bits);
    const size_t lane_size = static_cast<size_t>(kTurboVecScoreBlock);
    const size_t base = block_index * lane_size;
    const size_t valid = std::min(lane_size, n_vectors - base);
    const uint8_t * block = blocked_codes + block_index * n_byte_groups * lane_size;
    int32_t sums[kTurboVecMaxBatch][kTurboVecScoreBlock];
    bool done = false;
#if GGML_VEC_INDEX_USE_NEON && (defined(__clang__) || defined(__GNUC__))
    if (!batch.smmla_a.empty() && turbovec_cpu_i8mm()) {
        const int8_t * levels = batch.pds[0]->levels;
        const int8_t * a = batch.smmla_a.data();
        const size_t n_blocks = (n_vectors + lane_size - 1) / lane_size;
        const size_t bytes_after = (n_blocks - block_index) * n_byte_groups * lane_size;
        done = true;
#define GGML_VEC_INDEX_SMMLA_BATCH(NP)                                                                    \
        if (batch.layout == kTurboVecLayoutVm8 && batch.bits == 2) {                                      \
            permute_dot_block_sums_smmla_vm8_2bit<NP>(levels, batch.pds[0]->levels2, a, block, n_byte_groups, bytes_after, sums); \
        } else if (batch.layout == kTurboVecLayoutVm8) {                                                  \
            permute_dot_block_sums_smmla_vm8<NP>(levels, a, block, n_byte_groups, bytes_after, sums);     \
        } else {                                                                                          \
            permute_dot_block_sums_smmla_vm<NP>(levels, a, block, n_byte_groups, bytes_after, sums);      \
        }
        switch (n) {
            case 12: GGML_VEC_INDEX_SMMLA_BATCH(6) break;
            case 8: GGML_VEC_INDEX_SMMLA_BATCH(4) break;
            default: GGML_VEC_INDEX_SMMLA_BATCH(2) break;
        }
#undef GGML_VEC_INDEX_SMMLA_BATCH
    } else if (batch.bits == 4 && batch.layout == kTurboVecLayoutVm && turbovec_cpu_dot_product() && n % 4 == 0) {
        // Four queries at a time share the code loads and level lookups.
        for (int q = 0; q < n; q += 4) {
            permute_dot_block_sums_sdot_4(batch.pds + q, block, n_byte_groups, sums + q);
        }
        done = true;
    } else if (batch.bits == 4 && turbovec_cpu_dot_product()) {
        for (int q = 0; q < n; ++q) {
            if (batch.layout == kTurboVecLayoutVm8) {
                permute_dot_block_sums_sdot_vm8_single(*batch.pds[q], block, n_byte_groups, sums[q]);
            } else {
                permute_dot_block_sums_sdot(*batch.pds[q], block, n_byte_groups, batch.layout == kTurboVecLayoutVm, sums[q]);
            }
        }
        done = true;
    }
#endif
    if (!done) {
        for (int q = 0; q < n; ++q) {
            permute_dot_block_sums_scalar(*batch.pds[q], block, n_byte_groups, batch.bits, batch.layout, valid, sums[q]);
        }
    }
    for (int q = 0; q < n; ++q) {
        permute_dot_finish(*batch.pds[q], sums[q], vector_scales, base, valid, out_scores[q]);
    }
}

bool turbovec_rows_in_blocked(const ggml_vec_index & idx) {
    return is_turbovec(idx) && !idx.turbovec_packed_rows;
}

void turbovec_materialize_packed_rows(ggml_vec_index & idx) {
    if (idx.turbovec_packed_rows) {
        return;
    }
    std::vector<uint8_t> packed;
    turbovec_rows_packed(idx, packed);
    idx.turbovec_data.swap(packed);
    idx.turbovec_packed_rows = true;
}

void turbovec_settle_rows(ggml_vec_index & idx) {
    if (!is_turbovec(idx) || idx.turbovec_legacy || idx.turbovec_planes || !idx.turbovec_packed_rows) {
        return;
    }
    const size_t n = idx.slot_to_id.size();
    const size_t n_byte_groups = static_cast<size_t>(idx.dim) / static_cast<size_t>(8 / idx.turbovec_bits);
    const size_t lane_size = static_cast<size_t>(kTurboVecScoreBlock);
    const size_t n_blocks = (n + lane_size - 1) / lane_size;
    if (idx.turbovec_blocked_n_blocks != n_blocks || idx.turbovec_blocked_data.size() != n_blocks * n_byte_groups * lane_size) {
        return;
    }
    std::vector<uint8_t>().swap(idx.turbovec_data);
    idx.turbovec_packed_rows = false;
}

void turbovec_row_seq(const ggml_vec_index & idx, size_t slot, uint8_t * seq) {
    const int bits = idx.turbovec_bits;
    const size_t n_byte_groups = static_cast<size_t>(idx.dim) / static_cast<size_t>(8 / bits);
    if (!turbovec_rows_in_blocked(idx)) {
        packed_row_to_seq_fast(
            idx.turbovec_data.data() + slot * turbovec_row_bytes(bits, static_cast<size_t>(idx.dim)), bits, idx.dim, seq);
        return;
    }
    const size_t lane_size = static_cast<size_t>(kTurboVecScoreBlock);
    const uint8_t * block = idx.turbovec_blocked_data.data() + (slot / lane_size) * n_byte_groups * lane_size;
    block_lane_row(block, n_byte_groups, slot % lane_size, turbovec_layout(idx), seq);
}

const uint8_t * turbovec_row_packed_ptr(const ggml_vec_index & idx, size_t slot, uint8_t * scratch) {
    if (!turbovec_rows_in_blocked(idx)) {
        return idx.turbovec_data.data() + slot * turbovec_row_bytes(idx.turbovec_bits, static_cast<size_t>(idx.dim));
    }
    turbovec_row_packed(idx, slot, scratch);
    return scratch;
}

void turbovec_row_packed(const ggml_vec_index & idx, size_t slot, uint8_t * packed) {
    const int bits = idx.turbovec_bits;
    if (!turbovec_rows_in_blocked(idx)) {
        const size_t row_bytes = turbovec_row_bytes(bits, static_cast<size_t>(idx.dim));
        std::memcpy(packed, idx.turbovec_data.data() + slot * row_bytes, row_bytes);
        return;
    }
    uint8_t seq[kTurboVecMaxDim / 2];
    turbovec_row_seq(idx, slot, seq);
    seq_row_to_packed_fast(seq, bits, idx.dim, packed);
}

size_t turbovec_block_scratch_bytes(const ggml_vec_index & idx) {
    return static_cast<size_t>(kTurboVecScoreBlock) * (static_cast<size_t>(idx.dim) / static_cast<size_t>(8 / idx.turbovec_bits));
}

void turbovec_block_rows_packed(const ggml_vec_index & idx, size_t block, uint8_t * packed, uint8_t * scratch) {
    const int bits = idx.turbovec_bits;
    const size_t lane_size = static_cast<size_t>(kTurboVecScoreBlock);
    const size_t n = idx.slot_to_id.size();
    const size_t first = block * lane_size;
    const size_t count = std::min(lane_size, n - first);
    const size_t row_bytes = turbovec_row_bytes(bits, static_cast<size_t>(idx.dim));
    if (!turbovec_rows_in_blocked(idx)) {
        std::memcpy(packed, idx.turbovec_data.data() + first * row_bytes, count * row_bytes);
        return;
    }
    // One pass over the block, group by group, then each row repacked.
    const size_t n_byte_groups = static_cast<size_t>(idx.dim) / static_cast<size_t>(8 / bits);
    const uint8_t * src = idx.turbovec_blocked_data.data() + block * n_byte_groups * lane_size;
    const TurboVecLayout layout = turbovec_layout(idx);
    if (layout == kTurboVecLayoutVm8) {
        for (size_t lane = 0; lane < count; ++lane) {
            block_lane_row(src, n_byte_groups, lane, layout, scratch + lane * n_byte_groups);
        }
    } else {
        for (size_t g = 0; g < n_byte_groups; ++g) {
            for (size_t lane = 0; lane < count; ++lane) {
                scratch[lane * n_byte_groups + g] = block_lane_byte(src, g, lane, layout);
            }
        }
    }
    for (size_t lane = 0; lane < count; ++lane) {
        seq_row_to_packed_fast(scratch + lane * n_byte_groups, bits, idx.dim, packed + lane * row_bytes);
    }
}

void turbovec_rows_packed(const ggml_vec_index & idx, std::vector<uint8_t> & packed) {
    const size_t row_bytes = turbovec_row_bytes(idx.turbovec_bits, static_cast<size_t>(idx.dim));
    const size_t n = idx.slot_to_id.size();
    const size_t lane_size = static_cast<size_t>(kTurboVecScoreBlock);
    packed.resize(n * row_bytes);
    const size_t n_blocks = (n + lane_size - 1) / lane_size;
    // Each block writes its own rows.
    parallel_for((n_blocks + kTurboVecBlocksPerTask - 1) / kTurboVecBlocksPerTask, idx.n_threads, [&](size_t task, size_t) {
        std::vector<uint8_t> scratch(turbovec_rows_in_blocked(idx) ? turbovec_block_scratch_bytes(idx) : 0);
        const size_t end = std::min(n_blocks, (task + 1) * kTurboVecBlocksPerTask);
        for (size_t block = task * kTurboVecBlocksPerTask; block < end; ++block) {
            turbovec_block_rows_packed(idx, block, packed.data() + block * lane_size * row_bytes, scratch.data());
        }
    });
}

void turbovec_blocked_set_row_seq(ggml_vec_index & idx, size_t slot, const uint8_t * seq) {
    const size_t n_byte_groups = static_cast<size_t>(idx.dim) / static_cast<size_t>(8 / idx.turbovec_bits);
    const size_t lane_size = static_cast<size_t>(kTurboVecScoreBlock);
    uint8_t * block = idx.turbovec_blocked_data.data() + (slot / lane_size) * n_byte_groups * lane_size;
    set_block_lane_row(block, n_byte_groups, slot % lane_size, turbovec_layout(idx), seq);
}

// Upstream pack.rs x86 perm0 nibble interleave of 32-byte groups of
// sequential lane bytes, in place (the scalar arm of `interleave_chunk_x86`),
// and its inverse (`deinterleave_x86_code_byte` per lane).
[[maybe_unused]] static void turbovec_interleave_scalar(uint8_t * buf, size_t len) {
    static constexpr size_t perm0[16] = { 0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15 };
    const size_t lane_size = static_cast<size_t>(kTurboVecScoreBlock);
    uint8_t tmp[kTurboVecScoreBlock];
    for (size_t at = 0; at + lane_size <= len; at += lane_size) {
        uint8_t * group = buf + at;
        std::memcpy(tmp, group, lane_size);
        for (size_t j = 0; j < 16; ++j) {
            const uint8_t ba = tmp[perm0[j]];
            const uint8_t bb = tmp[perm0[j] + 16];
            group[j] = static_cast<uint8_t>((ba >> 4) | (bb & 0xf0));
            group[16 + j] = static_cast<uint8_t>((ba & 0x0f) | ((bb & 0x0f) << 4));
        }
    }
}

[[maybe_unused]] static void turbovec_deinterleave_scalar(const uint8_t * blocked, uint8_t * out, size_t len) {
    static constexpr size_t inv_perm0[16] = { 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15 };
    const size_t lane_size = static_cast<size_t>(kTurboVecScoreBlock);
    for (size_t at = 0; at + lane_size <= len; at += lane_size) {
        for (size_t lane = 0; lane < lane_size; ++lane) {
            const size_t j = inv_perm0[lane & 15];
            const uint8_t hi_plane = blocked[at + j];
            const uint8_t lo_plane = blocked[at + 16 + j];
            const int shift = lane < 16 ? 0 : 4;
            out[at + lane] = static_cast<uint8_t>(
                (((hi_plane >> shift) & 0x0f) << 4) | ((lo_plane >> shift) & 0x0f));
        }
    }
}

#if defined(__x86_64__) || defined(_M_X64)
// Upstream `interleave_chunk_x86` / `deinterleave_chunk_x86` dispatch: AVX2,
// then SSSE3, then the scalar loop.
static void turbovec_interleave_native(uint8_t * buf, size_t len) {
#if defined(GGML_VEC_INDEX_HAVE_AVX2_KERNEL)
    if (vec_index_cpu_features().avx2) {
        ggml_vec_index_detail::turbovec_interleave_avx2(buf, len);
        return;
    }
#endif
#if defined(GGML_VEC_INDEX_HAVE_SSE42_KERNEL)
    if (vec_index_cpu_features().ssse3) {
        ggml_vec_index_detail::turbovec_interleave_ssse3(buf, len);
        return;
    }
#endif
    turbovec_interleave_scalar(buf, len);
}

static void turbovec_deinterleave_native(const uint8_t * blocked, uint8_t * out, size_t len) {
#if defined(GGML_VEC_INDEX_HAVE_SSE42_KERNEL)
    if (vec_index_cpu_features().ssse3) {
        ggml_vec_index_detail::turbovec_deinterleave_ssse3(blocked, out, len);
        return;
    }
#endif
    turbovec_deinterleave_scalar(blocked, out, len);
}
#endif

void turbovec_block_seq(const ggml_vec_index & idx, size_t block, uint8_t * seq_blocked) {
    const size_t n_byte_groups = static_cast<size_t>(idx.dim) / static_cast<size_t>(8 / idx.turbovec_bits);
    const size_t lane_size = static_cast<size_t>(kTurboVecScoreBlock);
    const size_t block_bytes = n_byte_groups * lane_size;
    const uint8_t * src = idx.turbovec_blocked_data.data() + block * block_bytes;
    const TurboVecLayout layout = turbovec_layout(idx);
    if (layout != kTurboVecLayoutSeq) {
        for (size_t g = 0; g < n_byte_groups; ++g) {
            for (size_t lane = 0; lane < lane_size; ++lane) {
                seq_blocked[g * lane_size + lane] = block_lane_byte(src, g, lane, layout);
            }
        }
        return;
    }
#if defined(__x86_64__) || defined(_M_X64)
    turbovec_deinterleave_native(src, seq_blocked, block_bytes);
#else
    std::memcpy(seq_blocked, src, block_bytes);
#endif
}

void turbovec_blocked_set_block_seq(ggml_vec_index & idx, size_t block, const uint8_t * seq_blocked) {
    const size_t n_byte_groups = static_cast<size_t>(idx.dim) / static_cast<size_t>(8 / idx.turbovec_bits);
    const size_t lane_size = static_cast<size_t>(kTurboVecScoreBlock);
    const size_t block_bytes = n_byte_groups * lane_size;
    uint8_t * dst = idx.turbovec_blocked_data.data() + block * block_bytes;
    const TurboVecLayout layout = turbovec_layout(idx);
    if (layout != kTurboVecLayoutSeq) {
        for (size_t g = 0; g < n_byte_groups; ++g) {
            for (size_t lane = 0; lane < lane_size; ++lane) {
                set_block_lane_byte(dst, g, lane, layout, seq_blocked[g * lane_size + lane]);
            }
        }
        return;
    }
    std::memcpy(dst, seq_blocked, block_bytes);
#if defined(__x86_64__) || defined(_M_X64)
    turbovec_interleave_native(dst, block_bytes);
#endif
}

#ifdef GGML_VEC_INDEX_TEST_HOOKS
// Kernel 0: SSSE3 interleave and deinterleave; 1: AVX2 interleave, each
// against the scalar loops; 2: the whole-block helpers in both blocked
// layouts against the per-lane accessors (every host). 1 on a match, 0 on a
// mismatch, -1 when this build or CPU lacks the kernel.
int turbovec_layout_kernel_parity_for_test(int kernel) {
    const size_t lane_size = static_cast<size_t>(kTurboVecScoreBlock);
    // An odd group count covers the AVX2 kernel's single-group tail.
    constexpr size_t n_groups = 101;
    std::vector<uint8_t> src(n_groups * lane_size);
    uint32_t state = 0x9e3779b9u;
    for (uint8_t & b : src) {
        state = state * 1664525u + 1013904223u;
        b = static_cast<uint8_t>(state >> 24);
    }
    if (kernel == 0 || kernel == 1) {
        void (*interleave)(uint8_t *, size_t) = nullptr;
        void (*deinterleave)(const uint8_t *, uint8_t *, size_t) = nullptr;
#if (defined(__x86_64__) || defined(_M_X64)) && defined(GGML_VEC_INDEX_HAVE_SSE42_KERNEL)
        if (kernel == 0 && vec_index_cpu_features().ssse3) {
            interleave = ggml_vec_index_detail::turbovec_interleave_ssse3;
            deinterleave = ggml_vec_index_detail::turbovec_deinterleave_ssse3;
        }
#endif
#if (defined(__x86_64__) || defined(_M_X64)) && defined(GGML_VEC_INDEX_HAVE_AVX2_KERNEL)
        if (kernel == 1 && vec_index_cpu_features().avx2) {
            interleave = ggml_vec_index_detail::turbovec_interleave_avx2;
        }
#endif
        if (interleave == nullptr) {
            return -1;
        }
        std::vector<uint8_t> expect = src;
        turbovec_interleave_scalar(expect.data(), expect.size());
        for (size_t groups : { n_groups, size_t{ 1 }, size_t{ 2 }, size_t{ 0 } }) {
            std::vector<uint8_t> got(src.begin(), src.begin() + static_cast<std::ptrdiff_t>(groups * lane_size));
            interleave(got.data(), got.size());
            if (!std::equal(got.begin(), got.end(), expect.begin())) {
                return 0;
            }
        }
        if (expect == src) {
            return 0;
        }
        std::vector<uint8_t> back(src.size());
        turbovec_deinterleave_scalar(expect.data(), back.data(), back.size());
        if (back != src) {
            return 0;
        }
        if (deinterleave != nullptr) {
            std::fill(back.begin(), back.end(), 0);
            deinterleave(expect.data(), back.data(), back.size());
            if (back != src) {
                return 0;
            }
        }
        return 1;
    }
    if (kernel != 2) {
        return -1;
    }
    // dim 192 at 2 bits: 48 byte groups, a multiple of the vector-major unit.
    constexpr int dim = 192;
    ggml_vec_index_t * idx = ggml_vec_index_create_turbovec_q2(dim);
    if (idx == nullptr) {
        return 0;
    }
    const size_t n_byte_groups = static_cast<size_t>(dim) / 4;
    const size_t block_bytes = n_byte_groups * lane_size;
    bool ok = true;
    for (const TurboVecLayout layout : { kTurboVecLayoutSeq, kTurboVecLayoutVm, kTurboVecLayoutVm8 }) {
        idx->turbovec_blocked_vm = layout == kTurboVecLayoutVm;
        idx->turbovec_blocked_vm8 = layout == kTurboVecLayoutVm8;
        idx->turbovec_blocked_data.assign(2 * block_bytes, 0);
        const uint8_t * seq = src.data();
        turbovec_blocked_set_block_seq(*idx, 1, seq);
        const uint8_t * block = idx->turbovec_blocked_data.data() + block_bytes;
        for (size_t g = 0; g < n_byte_groups && ok; ++g) {
            for (size_t lane = 0; lane < lane_size; ++lane) {
                ok = ok && block_lane_byte(block, g, lane, layout) == seq[g * lane_size + lane];
            }
        }
        ok = ok && std::all_of(
            idx->turbovec_blocked_data.begin(),
            idx->turbovec_blocked_data.begin() + static_cast<std::ptrdiff_t>(block_bytes),
            [](uint8_t b) { return b == 0; });
        std::vector<uint8_t> back(block_bytes);
        turbovec_block_seq(*idx, 1, back.data());
        ok = ok && std::equal(back.begin(), back.end(), seq);
    }
    idx->turbovec_blocked_data.clear();
    ggml_vec_index_free(idx);
    return ok ? 1 : 0;
}
#endif

void turbovec_blocked_append(ggml_vec_index & idx, const uint8_t * packed, size_t first_slot, size_t n_new) {
    // New rows land lane by lane: lanes already in the partial tail block
    // keep their bytes.
    const int bits = idx.turbovec_bits;
    const size_t n_byte_groups = static_cast<size_t>(idx.dim) / static_cast<size_t>(8 / bits);
    const size_t lane_size = static_cast<size_t>(kTurboVecScoreBlock);
    const size_t block_bytes = n_byte_groups * lane_size;
    const size_t row_bytes = turbovec_row_bytes(bits, static_cast<size_t>(idx.dim));
    const size_t n_total = first_slot + n_new;
    const size_t n_blocks = (n_total + lane_size - 1) / lane_size;
    std::vector<uint8_t> & blocked = idx.turbovec_blocked_data;
    reserve_mostly_exact(blocked, n_blocks * block_bytes > blocked.size() ? n_blocks * block_bytes - blocked.size() : 0);
    blocked.resize(n_blocks * block_bytes, 0);
    idx.turbovec_blocked_n_blocks = n_blocks;
    const TurboVecLayout layout = turbovec_layout(idx);
    // A task covers whole blocks, as lanes of one block can share bytes.
    const size_t first_block = first_slot / lane_size;
    const size_t n_tasks = (n_blocks - first_block + kTurboVecBlocksPerTask - 1) / kTurboVecBlocksPerTask;
    parallel_for(n_new == 0 ? 0 : n_tasks, idx.n_threads, [&](size_t task, size_t) {
        uint8_t seq[kTurboVecMaxDim / 2];
        const size_t begin = std::max(first_slot, (first_block + task * kTurboVecBlocksPerTask) * lane_size);
        const size_t end = std::min(n_total, (first_block + (task + 1) * kTurboVecBlocksPerTask) * lane_size);
        for (size_t slot = begin; slot < end; ++slot) {
            packed_row_to_seq_fast(packed + (slot - first_slot) * row_bytes, bits, idx.dim, seq);
            uint8_t * block = blocked.data() + (slot / lane_size) * block_bytes;
            set_block_lane_row(block, n_byte_groups, slot % lane_size, layout, seq);
        }
    });
}

void turbovec_blocked_truncate(ggml_vec_index & idx, size_t n) {
    const size_t n_byte_groups = static_cast<size_t>(idx.dim) / static_cast<size_t>(8 / idx.turbovec_bits);
    const size_t lane_size = static_cast<size_t>(kTurboVecScoreBlock);
    const size_t block_bytes = n_byte_groups * lane_size;
    const size_t n_blocks = (n + lane_size - 1) / lane_size;
    std::vector<uint8_t> & blocked = idx.turbovec_blocked_data;
    if (blocked.size() < n_blocks * block_bytes) {
        // The cache does not cover these rows; with packed rows it is
        // rebuilt on the next add.
        if (idx.turbovec_packed_rows) {
            blocked.clear();
            idx.turbovec_blocked_n_blocks = 0;
        }
        return;
    }
    blocked.resize(n_blocks * block_bytes);
    idx.turbovec_blocked_n_blocks = n_blocks;
    if (n % lane_size != 0) {
        // Lanes past the last row stay zero, as a fresh repack leaves them.
        uint8_t * tail = blocked.data() + (n_blocks - 1) * block_bytes;
        const uint8_t zeros[kTurboVecMaxDim / 2] = {};
        for (size_t lane = n % lane_size; lane < lane_size; ++lane) {
            set_block_lane_row(tail, n_byte_groups, lane, turbovec_layout(idx), zeros);
        }
    }
}

void turbovec_blocked_swap_remove(ggml_vec_index & idx, size_t slot, size_t last) {
    const size_t n_byte_groups = static_cast<size_t>(idx.dim) / static_cast<size_t>(8 / idx.turbovec_bits);
    const size_t block_bytes = n_byte_groups * static_cast<size_t>(kTurboVecScoreBlock);
    std::vector<uint8_t> & blocked = idx.turbovec_blocked_data;
    if (blocked.size() != idx.turbovec_blocked_n_blocks * block_bytes ||
        idx.turbovec_blocked_n_blocks != (last + kTurboVecScoreBlock) / kTurboVecScoreBlock) {
        // No valid cache to patch; the next add or prepare rebuilds it.
        blocked.clear();
        idx.turbovec_blocked_n_blocks = 0;
        return;
    }
    // Move the last lane into the hole, zero the vacated lane (a stale lane
    // would break byte determinism), then drop a now-empty block.
    const size_t lane_size = static_cast<size_t>(kTurboVecScoreBlock);
    const TurboVecLayout layout = turbovec_layout(idx);
    uint8_t * src = blocked.data() + (last / lane_size) * block_bytes;
    uint8_t * dst = blocked.data() + (slot / lane_size) * block_bytes;
    uint8_t row[kTurboVecMaxDim / 2];
    const uint8_t zeros[kTurboVecMaxDim / 2] = {};
    if (slot != last) {
        block_lane_row(src, n_byte_groups, last % lane_size, layout, row);
        set_block_lane_row(dst, n_byte_groups, slot % lane_size, layout, row);
    }
    set_block_lane_row(src, n_byte_groups, last % lane_size, layout, zeros);
    idx.turbovec_blocked_n_blocks = (last + lane_size - 1) / lane_size;
    blocked.resize(idx.turbovec_blocked_n_blocks * block_bytes);
}

void repack_turbovec_sign_codes(
        const uint8_t * packed_codes,
        size_t n_vectors,
        int bits,
        int dim,
        size_t first_slot,
        std::vector<uint8_t> & sign_blocked,
        int n_threads) {
    // The sign plane of a bit-plane row is its top plane: one byte per
    // eight coordinates, scanned as an index with dim / 8 byte groups.
    const size_t n_sign_groups = static_cast<size_t>(dim) / 8;
    const size_t row_bytes = turbovec_row_bytes(bits, static_cast<size_t>(dim));
    const size_t sign_at = static_cast<size_t>(bits - 1) * n_sign_groups;
    const size_t lane_size = static_cast<size_t>(kTurboVecScoreBlock);
    const size_t block_bytes = n_sign_groups * lane_size;
    const size_t n_blocks = (n_vectors + lane_size - 1) / lane_size;
    const size_t first_block = std::min(first_slot / lane_size, sign_blocked.size() / block_bytes);
    reserve_mostly_exact(sign_blocked, n_blocks * block_bytes > sign_blocked.size() ?
        n_blocks * block_bytes - sign_blocked.size() : 0);
    sign_blocked.resize(n_blocks * block_bytes, 0);
    const size_t n_repack = n_blocks > first_block ? n_blocks - first_block : 0;
    parallel_for((n_repack + kTurboVecBlocksPerTask - 1) / kTurboVecBlocksPerTask, n_threads, [&](size_t task, size_t) {
        std::vector<uint8_t> seq(lane_size * n_sign_groups);
        const size_t begin = first_block + task * kTurboVecBlocksPerTask;
        const size_t end = std::min(n_blocks, begin + kTurboVecBlocksPerTask);
        for (size_t block = begin; block < end; ++block) {
            std::fill(seq.begin(), seq.end(), 0);
            for (size_t lane = 0; lane < lane_size; ++lane) {
                const size_t vector = block * lane_size + lane;
                if (vector < n_vectors) {
                    std::memcpy(seq.data() + lane * n_sign_groups, packed_codes + vector * row_bytes + sign_at, n_sign_groups);
                }
            }
            write_blocked_block(seq.data(), n_sign_groups, sign_blocked.data() + block * block_bytes, kTurboVecLayoutSeq);
        }
    });
}

void turbovec_sign_blocked_swap_remove(ggml_vec_index & idx, size_t slot, size_t last) {
    const size_t n_sign_groups = static_cast<size_t>(idx.dim) / 8;
    const size_t lane_size = static_cast<size_t>(kTurboVecScoreBlock);
    const size_t block_bytes = n_sign_groups * lane_size;
    std::vector<uint8_t> & sign = idx.turbovec_sign_blocked;
    if (sign.size() != ((last + lane_size) / lane_size) * block_bytes) {
        sign.clear();
        return;
    }
    uint8_t * src = sign.data() + (last / lane_size) * block_bytes;
    uint8_t * dst = sign.data() + (slot / lane_size) * block_bytes;
    for (size_t g = 0; g < n_sign_groups; ++g) {
        if (slot != last) {
            set_block_lane_byte(
                dst, g, slot % lane_size, kTurboVecLayoutSeq, block_lane_byte(src, g, last % lane_size, kTurboVecLayoutSeq));
        }
        set_block_lane_byte(src, g, last % lane_size, kTurboVecLayoutSeq, 0);
    }
    sign.resize(((last + lane_size - 1) / lane_size) * block_bytes);
}

void repack_turbovec_codes(
        const uint8_t * packed_codes,
        size_t n_vectors,
        int bits,
        int dim,
        std::vector<uint8_t> & blocked_codes,
        size_t & n_blocks,
        TurboVecLayout layout,
        int n_threads) {
    const size_t dim_sz = static_cast<size_t>(dim);
    const size_t codes_per_byte = static_cast<size_t>(8 / bits);
    const size_t n_byte_groups = dim_sz / codes_per_byte;
    // n_blocks changes only once the cache is resized, so a failed
    // allocation leaves the old cache and its block count in step.
    const size_t new_blocks = (n_vectors + kTurboVecScoreBlock - 1) / kTurboVecScoreBlock;
    blocked_codes.assign(
        new_blocks * n_byte_groups * static_cast<size_t>(kTurboVecScoreBlock),
        0);
    n_blocks = new_blocks;
    repack_turbovec_code_blocks(packed_codes, n_vectors, bits, dim, 0, n_blocks, blocked_codes, layout, n_threads);
}

void repack_turbovec_codes_from_slot(
        const uint8_t * packed_codes,
        size_t n_vectors,
        int bits,
        int dim,
        size_t first_slot,
        std::vector<uint8_t> & blocked_codes,
        size_t & n_blocks,
        TurboVecLayout layout,
        int n_threads) {
    const size_t dim_sz = static_cast<size_t>(dim);
    const size_t codes_per_byte = static_cast<size_t>(8 / bits);
    const size_t n_byte_groups = dim_sz / codes_per_byte;
    // Patching from first_slot needs a cache that covers exactly the rows
    // before it; anything else (an invalidated, empty cache included) is
    // rebuilt in full.
    const size_t old_blocks = (first_slot + kTurboVecScoreBlock - 1) / kTurboVecScoreBlock;
    const size_t old_expected_size =
        old_blocks * n_byte_groups * static_cast<size_t>(kTurboVecScoreBlock);
    if (first_slot == 0 || first_slot > n_vectors || n_blocks != old_blocks ||
        blocked_codes.size() != old_expected_size) {
        repack_turbovec_codes(packed_codes, n_vectors, bits, dim, blocked_codes, n_blocks, layout, n_threads);
        return;
    }

    const size_t first_block = first_slot / kTurboVecScoreBlock;
    n_blocks = (n_vectors + kTurboVecScoreBlock - 1) / kTurboVecScoreBlock;
    reserve_mostly_exact(
        blocked_codes,
        n_blocks * n_byte_groups * static_cast<size_t>(kTurboVecScoreBlock) - blocked_codes.size());
    blocked_codes.resize(
        n_blocks * n_byte_groups * static_cast<size_t>(kTurboVecScoreBlock),
        0);
    repack_turbovec_code_blocks(
        packed_codes, n_vectors, bits, dim, first_block, n_blocks, blocked_codes, layout, n_threads);
}

void score_turbovec_lut_block(
        const uint8_t * lut,
        float lut_scale,
        float lut_bias,
        const uint8_t * blocked_codes,
        const float * vector_scales,
        size_t block_index,
        size_t n_vectors,
        int bits,
        int dim,
        float * out_scores) {
    const size_t n_byte_groups =
        static_cast<size_t>(dim) / static_cast<size_t>(8 / bits);
    const size_t base_vector = block_index * kTurboVecScoreBlock;
    const size_t block_offset =
        block_index * n_byte_groups * static_cast<size_t>(kTurboVecScoreBlock);
    const size_t valid_lanes =
        std::min(static_cast<size_t>(kTurboVecScoreBlock), n_vectors - base_vector);

#if GGML_VEC_INDEX_USE_NEON
    const uint8x16_t nibble_mask = vdupq_n_u8(0x0f);
    const float32x4_t scale_v = vdupq_n_f32(lut_scale);
    float32x4_t float_accum[8];
    for (float32x4_t & value : float_accum) {
        value = vdupq_n_f32(lut_bias);
    }

    const size_t n_batches =
        (n_byte_groups + kTurboVecFlushEvery - 1) / kTurboVecFlushEvery;
    for (size_t batch = 0; batch < n_batches; ++batch) {
        const size_t group_begin = batch * kTurboVecFlushEvery;
        const size_t group_end =
            std::min(group_begin + static_cast<size_t>(kTurboVecFlushEvery), n_byte_groups);
        uint16x8_t accum[4] = {
            vdupq_n_u16(0),
            vdupq_n_u16(0),
            vdupq_n_u16(0),
            vdupq_n_u16(0),
        };
        for (size_t group = group_begin; group < group_end; ++group) {
            const uint8_t * lut_group = lut + group * 32;
            const uint8_t * code_group =
                blocked_codes + block_offset + group * kTurboVecScoreBlock;
            const uint8x16_t lut_hi = vld1q_u8(lut_group);
            const uint8x16_t lut_lo = vld1q_u8(lut_group + 16);
            const uint8x16_t codes0 = vld1q_u8(code_group);
            const uint8x16_t codes1 = vld1q_u8(code_group + 16);
            const uint8x16_t values0 = vaddq_u8(
                vqtbl1q_u8(lut_lo, vandq_u8(codes0, nibble_mask)),
                vqtbl1q_u8(lut_hi, vshrq_n_u8(codes0, 4)));
            const uint8x16_t values1 = vaddq_u8(
                vqtbl1q_u8(lut_lo, vandq_u8(codes1, nibble_mask)),
                vqtbl1q_u8(lut_hi, vshrq_n_u8(codes1, 4)));
            accum[0] = vaddw_u8(accum[0], vget_low_u8(values0));
            accum[1] = vaddw_u8(accum[1], vget_high_u8(values0));
            accum[2] = vaddw_u8(accum[2], vget_low_u8(values1));
            accum[3] = vaddw_u8(accum[3], vget_high_u8(values1));
        }
        for (int i = 0; i < 4; ++i) {
            const float32x4_t lo =
                vcvtq_f32_u32(vmovl_u16(vget_low_u16(accum[i])));
            const float32x4_t hi =
                vcvtq_f32_u32(vmovl_u16(vget_high_u16(accum[i])));
            float_accum[i * 2] = vfmaq_f32(float_accum[i * 2], scale_v, lo);
            float_accum[i * 2 + 1] =
                vfmaq_f32(float_accum[i * 2 + 1], scale_v, hi);
        }
    }

    if (valid_lanes == kTurboVecScoreBlock) {
        for (int i = 0; i < 8; ++i) {
            const float32x4_t vector_scale =
                vld1q_f32(vector_scales + base_vector + static_cast<size_t>(i) * 4);
            vst1q_f32(
                out_scores + static_cast<size_t>(i) * 4,
                vmulq_f32(float_accum[i], vector_scale));
        }
    } else {
        float decoded[kTurboVecScoreBlock];
        for (int i = 0; i < 8; ++i) {
            vst1q_f32(decoded + static_cast<size_t>(i) * 4, float_accum[i]);
        }
        for (size_t lane = 0; lane < kTurboVecScoreBlock; ++lane) {
            out_scores[lane] = lane < valid_lanes ?
                decoded[lane] * vector_scales[base_vector + lane] :
                -std::numeric_limits<float>::infinity();
        }
    }
#else
#if defined(__x86_64__) || defined(_M_X64)
    static constexpr size_t inverse_perm[16] = {
        0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15,
    };
#endif
    for (size_t lane = 0; lane < kTurboVecScoreBlock; ++lane) {
        if (lane >= valid_lanes) {
            out_scores[lane] = -std::numeric_limits<float>::infinity();
            continue;
        }
        float score = lut_bias;
        for (size_t group = 0; group < n_byte_groups; ++group) {
            const size_t group_offset =
                block_offset + group * static_cast<size_t>(kTurboVecScoreBlock);
#if defined(__x86_64__) || defined(_M_X64)
            const size_t j = inverse_perm[lane & 15];
            const uint8_t hi_plane = blocked_codes[group_offset + j];
            const uint8_t lo_plane = blocked_codes[group_offset + 16 + j];
            const uint8_t hi = lane < 16 ?
                static_cast<uint8_t>(hi_plane & 0x0f) :
                static_cast<uint8_t>(hi_plane >> 4);
            const uint8_t lo = lane < 16 ?
                static_cast<uint8_t>(lo_plane & 0x0f) :
                static_cast<uint8_t>(lo_plane >> 4);
#else
            const uint8_t code_byte = blocked_codes[group_offset + lane];
            const uint8_t hi = static_cast<uint8_t>(code_byte >> 4);
            const uint8_t lo = static_cast<uint8_t>(code_byte & 0x0f);
#endif
            score += lut_scale * static_cast<float>(lut[group * 32 + hi]);
            score += lut_scale * static_cast<float>(lut[group * 32 + 16 + lo]);
        }
        out_scores[lane] = score * vector_scales[base_vector + lane];
    }
#endif
}

#if GGML_VEC_INDEX_USE_NEON
// The eight group registers (16 lanes each) of half a vm8 octet: a
// three-level UZP tree over eight 16-byte loads (upstream
// `score_2bit_block_vm8_neon`'s `groups_of`).
static inline void turbovec_vm8_groups(const uint8_t * p, uint8x16_t * out) {
    uint8x16_t r[8];
    for (int i = 0; i < 8; ++i) {
        r[i] = vld1q_u8(p + i * 16);
    }
    const uint8x16_t t[8] = {
        vuzp1q_u8(r[0], r[1]), vuzp1q_u8(r[2], r[3]), vuzp1q_u8(r[4], r[5]), vuzp1q_u8(r[6], r[7]),
        vuzp2q_u8(r[0], r[1]), vuzp2q_u8(r[2], r[3]), vuzp2q_u8(r[4], r[5]), vuzp2q_u8(r[6], r[7]),
    };
    const uint8x16_t u[8] = {
        vuzp1q_u8(t[0], t[1]), vuzp1q_u8(t[2], t[3]), vuzp2q_u8(t[0], t[1]), vuzp2q_u8(t[2], t[3]),
        vuzp1q_u8(t[4], t[5]), vuzp1q_u8(t[6], t[7]), vuzp2q_u8(t[4], t[5]), vuzp2q_u8(t[6], t[7]),
    };
    out[0] = vuzp1q_u8(u[0], u[1]);
    out[1] = vuzp1q_u8(u[4], u[5]);
    out[2] = vuzp1q_u8(u[2], u[3]);
    out[3] = vuzp1q_u8(u[6], u[7]);
    out[4] = vuzp2q_u8(u[0], u[1]);
    out[5] = vuzp2q_u8(u[4], u[5]);
    out[6] = vuzp2q_u8(u[2], u[3]);
    out[7] = vuzp2q_u8(u[6], u[7]);
}
#endif

void score_turbovec_lut_block_vm8(
        const uint8_t * lut,
        float lut_scale,
        float lut_bias,
        const uint8_t * blocked_codes,
        const float * vector_scales,
        size_t block_index,
        size_t n_vectors,
        int bits,
        int dim,
        float * out_scores) {
    // The nibble-LUT scan over a vm8 block (upstream
    // `score_2bit_block_vm8_neon`): the same tables, u8 pre-add, u16
    // accumulators and flush cadence as score_turbovec_lut_block, so the
    // scores are bit-identical to the sequential layout's.
    const size_t n_byte_groups = static_cast<size_t>(dim) / static_cast<size_t>(8 / bits);
    const size_t base_vector = block_index * kTurboVecScoreBlock;
    const uint8_t * block = blocked_codes + block_index * n_byte_groups * static_cast<size_t>(kTurboVecScoreBlock);
    const size_t valid_lanes = std::min(static_cast<size_t>(kTurboVecScoreBlock), n_vectors - base_vector);
#if GGML_VEC_INDEX_USE_NEON
    const uint8x16_t nibble_mask = vdupq_n_u8(0x0f);
    const float32x4_t scale_v = vdupq_n_f32(lut_scale);
    float32x4_t float_accum[8];
    for (float32x4_t & value : float_accum) {
        value = vdupq_n_f32(lut_bias);
    }
    uint16x8_t accum[4] = { vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0), vdupq_n_u16(0) };
    const size_t octs = n_byte_groups / 8;
    for (size_t oct = 0; oct < octs; ++oct) {
        uint8x16_t ga[8];
        uint8x16_t gb[8];
        turbovec_vm8_groups(block + oct * 256, ga);
        turbovec_vm8_groups(block + oct * 256 + 128, gb);
        for (size_t j = 0; j < 8; ++j) {
            const uint8_t * lut_group = lut + (oct * 8 + j) * 32;
            const uint8x16_t lut_hi = vld1q_u8(lut_group);
            const uint8x16_t lut_lo = vld1q_u8(lut_group + 16);
            const uint8x16_t values0 = vaddq_u8(
                vqtbl1q_u8(lut_lo, vandq_u8(ga[j], nibble_mask)), vqtbl1q_u8(lut_hi, vshrq_n_u8(ga[j], 4)));
            const uint8x16_t values1 = vaddq_u8(
                vqtbl1q_u8(lut_lo, vandq_u8(gb[j], nibble_mask)), vqtbl1q_u8(lut_hi, vshrq_n_u8(gb[j], 4)));
            accum[0] = vaddw_u8(accum[0], vget_low_u8(values0));
            accum[1] = vaddw_u8(accum[1], vget_high_u8(values0));
            accum[2] = vaddw_u8(accum[2], vget_low_u8(values1));
            accum[3] = vaddw_u8(accum[3], vget_high_u8(values1));
        }
        // kTurboVecFlushEvery is a multiple of eight, so an octet never
        // straddles a flush.
        if ((oct + 1) * 8 % kTurboVecFlushEvery == 0 || oct + 1 == octs) {
            for (int i = 0; i < 4; ++i) {
                const float32x4_t lo = vcvtq_f32_u32(vmovl_u16(vget_low_u16(accum[i])));
                const float32x4_t hi = vcvtq_f32_u32(vmovl_u16(vget_high_u16(accum[i])));
                float_accum[i * 2] = vfmaq_f32(float_accum[i * 2], scale_v, lo);
                float_accum[i * 2 + 1] = vfmaq_f32(float_accum[i * 2 + 1], scale_v, hi);
                accum[i] = vdupq_n_u16(0);
            }
        }
    }
    float decoded[kTurboVecScoreBlock];
    for (int i = 0; i < 8; ++i) {
        vst1q_f32(decoded + static_cast<size_t>(i) * 4, float_accum[i]);
    }
    for (size_t lane = 0; lane < kTurboVecScoreBlock; ++lane) {
        out_scores[lane] = lane < valid_lanes ?
            decoded[lane] * vector_scales[base_vector + lane] :
            -std::numeric_limits<float>::infinity();
    }
#else
    // The NEON arithmetic exactly: integer sums per flush of 256 groups
    // (entries are at most 127, so neither the pair nor the sum wraps),
    // folded in with one fma each.
    constexpr size_t flush_every = 256;
    for (size_t lane = 0; lane < kTurboVecScoreBlock; ++lane) {
        if (lane >= valid_lanes) {
            out_scores[lane] = -std::numeric_limits<float>::infinity();
            continue;
        }
        float score = lut_bias;
        uint32_t sum = 0;
        for (size_t group = 0; group < n_byte_groups; ++group) {
            const uint8_t code_byte = block[vm8_offset(group, lane)];
            sum += static_cast<uint32_t>(lut[group * 32 + (code_byte >> 4)]) +
                   static_cast<uint32_t>(lut[group * 32 + 16 + (code_byte & 0x0f)]);
            if ((group + 1) % flush_every == 0 || group + 1 == n_byte_groups) {
                score = std::fma(lut_scale, static_cast<float>(sum), score);
                sum = 0;
            }
        }
        out_scores[lane] = score * vector_scales[base_vector + lane];
    }
#endif
}

void score_turbovec_lut_block_4(
        const uint8_t * const luts[4],
        const float lut_scales[4],
        const float lut_biases[4],
        const uint8_t * blocked_codes,
        const float * vector_scales,
        size_t block_index,
        size_t n_vectors,
        int bits,
        int dim,
        float * const out_scores[4]) {
#if GGML_VEC_INDEX_USE_NEON
    const size_t n_byte_groups =
        static_cast<size_t>(dim) / static_cast<size_t>(8 / bits);
    const size_t base_vector = block_index * kTurboVecScoreBlock;
    const size_t block_offset =
        block_index * n_byte_groups * static_cast<size_t>(kTurboVecScoreBlock);
    const size_t valid_lanes =
        std::min(static_cast<size_t>(kTurboVecScoreBlock), n_vectors - base_vector);
    const uint8x16_t nibble_mask = vdupq_n_u8(0x0f);
    float32x4_t float_accum[4][8];
    for (int query = 0; query < 4; ++query) {
        for (float32x4_t & value : float_accum[query]) {
            value = vdupq_n_f32(lut_biases[query]);
        }
    }

    const size_t n_batches =
        (n_byte_groups + kTurboVecFlushEvery - 1) / kTurboVecFlushEvery;
    for (size_t batch = 0; batch < n_batches; ++batch) {
        const size_t group_begin = batch * kTurboVecFlushEvery;
        const size_t group_end =
            std::min(group_begin + static_cast<size_t>(kTurboVecFlushEvery), n_byte_groups);
        uint16x8_t accum[4][4];
        for (auto & query_accum : accum) {
            for (uint16x8_t & value : query_accum) {
                value = vdupq_n_u16(0);
            }
        }

        for (size_t group = group_begin; group < group_end; ++group) {
            const uint8_t * code_group =
                blocked_codes + block_offset + group * kTurboVecScoreBlock;
            const uint8x16_t codes0 = vld1q_u8(code_group);
            const uint8x16_t codes1 = vld1q_u8(code_group + 16);
            const uint8x16_t lo0 = vandq_u8(codes0, nibble_mask);
            const uint8x16_t lo1 = vandq_u8(codes1, nibble_mask);
            const uint8x16_t hi0 = vshrq_n_u8(codes0, 4);
            const uint8x16_t hi1 = vshrq_n_u8(codes1, 4);

            for (int query = 0; query < 4; ++query) {
                const uint8_t * lut_group = luts[query] + group * 32;
                const uint8x16_t lut_hi = vld1q_u8(lut_group);
                const uint8x16_t lut_lo = vld1q_u8(lut_group + 16);
                const uint8x16_t values0 = vaddq_u8(
                    vqtbl1q_u8(lut_lo, lo0),
                    vqtbl1q_u8(lut_hi, hi0));
                const uint8x16_t values1 = vaddq_u8(
                    vqtbl1q_u8(lut_lo, lo1),
                    vqtbl1q_u8(lut_hi, hi1));
                accum[query][0] = vaddw_u8(accum[query][0], vget_low_u8(values0));
                accum[query][1] = vaddw_u8(accum[query][1], vget_high_u8(values0));
                accum[query][2] = vaddw_u8(accum[query][2], vget_low_u8(values1));
                accum[query][3] = vaddw_u8(accum[query][3], vget_high_u8(values1));
            }
        }

        for (int query = 0; query < 4; ++query) {
            const float32x4_t scale_v = vdupq_n_f32(lut_scales[query]);
            for (int i = 0; i < 4; ++i) {
                const float32x4_t lo =
                    vcvtq_f32_u32(vmovl_u16(vget_low_u16(accum[query][i])));
                const float32x4_t hi =
                    vcvtq_f32_u32(vmovl_u16(vget_high_u16(accum[query][i])));
                float_accum[query][i * 2] =
                    vfmaq_f32(float_accum[query][i * 2], scale_v, lo);
                float_accum[query][i * 2 + 1] =
                    vfmaq_f32(float_accum[query][i * 2 + 1], scale_v, hi);
            }
        }
    }

    for (int query = 0; query < 4; ++query) {
        if (valid_lanes == kTurboVecScoreBlock) {
            for (int i = 0; i < 8; ++i) {
                const float32x4_t vector_scale =
                    vld1q_f32(vector_scales + base_vector + static_cast<size_t>(i) * 4);
                vst1q_f32(
                    out_scores[query] + static_cast<size_t>(i) * 4,
                    vmulq_f32(float_accum[query][i], vector_scale));
            }
        } else {
            float decoded[kTurboVecScoreBlock];
            for (int i = 0; i < 8; ++i) {
                vst1q_f32(decoded + static_cast<size_t>(i) * 4, float_accum[query][i]);
            }
            for (size_t lane = 0; lane < kTurboVecScoreBlock; ++lane) {
                out_scores[query][lane] = lane < valid_lanes ?
                    decoded[lane] * vector_scales[base_vector + lane] :
                    -std::numeric_limits<float>::infinity();
            }
        }
    }
#else
    for (int query = 0; query < 4; ++query) {
        score_turbovec_lut_block(
            luts[query],
            lut_scales[query],
            lut_biases[query],
            blocked_codes,
            vector_scales,
            block_index,
            n_vectors,
            bits,
            dim,
            out_scores[query]);
    }
#endif
}

void rotate_turbovec_query(bool legacy, const float * src, float * dst, int dim) {
    rotate_turbovec_queries(legacy, src, dst, 1, dim);
}

void rotate_turbovec_queries(
        bool legacy,
        const float * src,
        float * dst,
        int n_queries,
        int dim) {
    if (legacy) {
        apply_turbovec_query_rotation_batch(src, dst, n_queries, dim);
        return;
    }
    const ScopedNearestRounding rounding_guard;
    const std::shared_ptr<const TurboVecHadamardRotation> rotation = turbovec_hadamard_rotation(dim);
    std::vector<float> scratch(static_cast<size_t>(dim));
    for (int query = 0; query < n_queries; ++query) {
        const size_t offset = static_cast<size_t>(query) * static_cast<size_t>(dim);
        apply_turbovec_hadamard(*rotation, src + offset, 1.0f, dst + offset, scratch.data());
    }
}

void quantize_turbovec_batch(
        const float * src,
        int n,
        int bits,
        uint8_t * dst,
        float * scales,
        int dim,
        std::vector<float> & tqplus_shift,
        std::vector<float> & tqplus_scale) {
    const ScopedNearestRounding rounding_guard;
    const size_t dim_sz = static_cast<size_t>(dim);
    const size_t n_sz = static_cast<size_t>(n);
    const size_t row_bytes = static_cast<size_t>(bits) * (dim_sz / 8);
    const TurboVecCodebook & codebook = turbovec_codebook(bits, dim, true);

    std::vector<double> norms(n_sz);
    std::vector<float> unit(n_sz * dim_sz);
    for (int row = 0; row < n; ++row) {
        const float * input = src + static_cast<size_t>(row) * dim_sz;
        float * unit_row = unit.data() + static_cast<size_t>(row) * dim_sz;
        const double norm = vector_norm(input, dim);
        norms[static_cast<size_t>(row)] = norm;
        const float inv_norm =
            norm > 1e-10 && norm <= static_cast<double>(FLT_MAX) ?
                1.0f / static_cast<float>(norm) : 0.0f;
        for (int column = 0; column < dim; ++column) {
            unit_row[static_cast<size_t>(column)] =
                input[static_cast<size_t>(column)] * inv_norm;
        }
    }

    std::vector<float> rotated(n_sz * dim_sz);
    apply_turbovec_rotation_batch(unit.data(), rotated.data(), n, dim);

    if (tqplus_shift.empty() != tqplus_scale.empty()) {
        throw std::invalid_argument("invalid TurboVec TQ+ calibration");
    }
    if (tqplus_shift.empty()) {
        tqplus_shift.assign(dim_sz, 0.0f);
        tqplus_scale.assign(dim_sz, 1.0f);
        if (n >= 1000) {
            constexpr double p_lo = 0.05;
            constexpr double p_hi = 0.95;
            // TQ+ maps empirical per-coordinate 5/95% quantiles onto the
            // canonical Beta((dim - 1) / 2, (dim - 1) / 2) marginal used by
            // Rust turbovec v0.9.0 before Lloyd-Max quantization.
            const double beta_a = (static_cast<double>(dim) - 1.0) * 0.5;
            const float canonical_lo =
                static_cast<float>(2.0 * inverse_regularized_beta(p_lo, beta_a) - 1.0);
            const float canonical_hi =
                static_cast<float>(2.0 * inverse_regularized_beta(p_hi, beta_a) - 1.0);
            const float canonical_span = canonical_hi - canonical_lo;
            const size_t lo_index = static_cast<size_t>(static_cast<double>(n) * p_lo);
            const size_t hi_index = std::min(
                static_cast<size_t>(static_cast<double>(n) * p_hi),
                n_sz - 1);
            std::vector<float> coordinate(n_sz);
            for (int column = 0; column < dim; ++column) {
                for (int row = 0; row < n; ++row) {
                    coordinate[static_cast<size_t>(row)] =
                        rotated[static_cast<size_t>(row) * dim_sz + static_cast<size_t>(column)];
                }
                std::sort(coordinate.begin(), coordinate.end());
                const float empirical_lo = coordinate[lo_index];
                const float empirical_hi = coordinate[hi_index];
                const float empirical_span = empirical_hi - empirical_lo;
                if (empirical_span > 1e-6f) {
                    const float scale = canonical_span / empirical_span;
                    tqplus_scale[static_cast<size_t>(column)] = scale;
                    tqplus_shift[static_cast<size_t>(column)] =
                        canonical_lo / scale - empirical_lo;
                }
            }
        }
    }

    if (tqplus_shift.size() != dim_sz || tqplus_scale.size() != dim_sz) {
        throw std::invalid_argument("invalid TurboVec TQ+ calibration");
    }
    const float * shift = tqplus_shift.data();
    const float * calibration_scale = tqplus_scale.data();

    std::memset(dst, 0, n_sz * row_bytes);
    for (int row = 0; row < n; ++row) {
        const float * rotated_row = rotated.data() + static_cast<size_t>(row) * dim_sz;
        uint8_t * packed_row = dst + static_cast<size_t>(row) * row_bytes;
        double inner = 0.0;
        for (int column = 0; column < dim; ++column) {
            const size_t coordinate = static_cast<size_t>(column);
            const float calibrated =
                (rotated_row[coordinate] + shift[coordinate]) *
                calibration_scale[coordinate];
            const uint8_t code =
                turbovec_quantize_val(calibrated, codebook.boundaries.data(), bits);
            turbovec_set_bitplane_code(packed_row, column, bits, dim, code);
            const float inverse_calibration_scale =
                1.0f / calibration_scale[coordinate];
            const double centroid_original =
                static_cast<double>(codebook.centroids[code]) *
                    static_cast<double>(inverse_calibration_scale) -
                static_cast<double>(shift[coordinate]);
            inner += static_cast<double>(rotated_row[coordinate]) * centroid_original;
        }
        // Degenerate reconstructions store scale 0 so one out-of-distribution
        // vector cannot win every query (upstream #116).
        if (!(inner > kTurboVecDegenerateInner)) {
            scales[static_cast<size_t>(row)] = 0.0f;
            continue;
        }
        const float scale =
            static_cast<float>(norms[static_cast<size_t>(row)]) /
            static_cast<float>(inner);
        if (std::isfinite(scale)) {
            scales[static_cast<size_t>(row)] = scale;
        } else {
            const double scale_f64 = norms[static_cast<size_t>(row)] / inner;
            if (!std::isfinite(scale_f64) || scale_f64 > static_cast<double>(FLT_MAX)) {
                throw std::invalid_argument("invalid TurboVec vector scale");
            }
            scales[static_cast<size_t>(row)] = static_cast<float>(scale_f64);
        }
    }
}

void turbovec_codebook_values(
        int bits,
        int dim,
        bool legacy,
        std::vector<float> & boundaries,
        std::vector<float> & centroids) {
    const TurboVecCodebook & codebook = turbovec_codebook(bits, dim, legacy);
    boundaries = codebook.boundaries;
    centroids = codebook.centroids;
}

void turbovec_packed_row_to_seq(const uint8_t * packed, int bits, int dim, uint8_t * seq) {
    packed_row_to_seq_fast(packed, bits, dim, seq);
}

void turbovec_seq_row_to_packed(const uint8_t * seq, int bits, int dim, uint8_t * packed) {
    seq_row_to_packed_fast(seq, bits, dim, packed);
}

void decode_turbovec_row_calibrated(
        int bits,
        bool legacy,
        const uint8_t * codes,
        const float * scales,
        const float * tqplus_shift,
        const float * tqplus_scale,
        float * dst,
        int dim) {
    const TurboVecCodebook & codebook = turbovec_codebook(bits, dim, legacy);
    const float * centroids = codebook.centroids.data();
    std::vector<float> rotated(static_cast<size_t>(dim));
    for (int i = 0; i < dim; ++i) {
        const uint8_t code = turbovec_get_bitplane_code(codes, i, bits, dim);
        const size_t coordinate = static_cast<size_t>(i);
        const float shift = tqplus_shift != nullptr ? tqplus_shift[coordinate] : 0.0f;
        const float calibration_scale = tqplus_scale != nullptr ? tqplus_scale[coordinate] : 1.0f;
        rotated[coordinate] =
            (centroids[code] / calibration_scale - shift) * scales[0];
    }
    if (legacy) {
        apply_turbovec_rotation(rotated.data(), dst, dim, true);
        return;
    }
    const std::shared_ptr<const TurboVecHadamardRotation> rotation = turbovec_hadamard_rotation(dim);
    std::vector<float> scratch(static_cast<size_t>(dim));
    apply_turbovec_hadamard_inverse(*rotation, rotated.data(), dst, scratch.data());
}

void build_turbovec_lut(
        int bits,
        bool legacy,
        const float * rotated_query,
        int dim,
        std::vector<uint8_t> & lut,
        float & scale,
        float & bias) {
    const TurboVecCodebook & codebook = turbovec_codebook(bits, dim, legacy);
    build_turbovec_query_lut(rotated_query, codebook.centroids.data(), bits, dim, lut, scale, bias);
}

float dot_turbovec_bits_lut_row(
        const uint8_t * lut,
        float lut_scale,
        float lut_bias,
        const uint8_t * codes,
        const float * scales,
        int bits,
        int dim) {
    return dot_turbovec_lut_row(lut, lut_scale, lut_bias, codes, scales, bits, dim);
}

// Quantize one already-rotated unit row under (shift, scale_tq) and pack one
// bit-plane row, returning the reconstruction inner product <u_rot, x_hat>
// in original space (upstream encode.rs `fused_quantize_scale_pack_scalar`).
static double quantize_turbovec_hadamard_row_scalar(
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
    const int n_boundaries = (1 << bits) - 1;
    double chains[4] = {};
    for (int chunk = 0; chunk < dim / 8; ++chunk) {
        uint8_t codes[8];
        for (int k = 0; k < 8; ++k) {
            const int j = chunk * 8 + k;
            const float calibrated = (rotated[j] + shift[j]) * scale_tq[j];
            uint8_t code = 0;
            for (int b = 0; b < n_boundaries; ++b) {
                if (calibrated > boundaries[b]) {
                    ++code;
                }
            }
            codes[k] = code;
            const double reconstructed =
                static_cast<double>(centroids[code]) * static_cast<double>(inv_scale_tq[j]) -
                static_cast<double>(shift[j]);
            chains[j % 4] += static_cast<double>(rotated[j]) * reconstructed;
        }
        for (int plane = 0; plane < bits; ++plane) {
            uint8_t byte = 0;
            for (int k = 0; k < 8; ++k) {
                byte |= static_cast<uint8_t>(((codes[k] >> plane) & 1u) << (7 - k));
            }
            packed_row[static_cast<size_t>(plane) * bytes_per_plane + static_cast<size_t>(chunk)] = byte;
        }
    }
    return (chains[0] + chains[1]) + (chains[2] + chains[3]);
}

#if GGML_VEC_INDEX_USE_NEON
// Upstream encode.rs aarch64 `fused_quantize_scale_pack` with the
// reconstruction computed inline (its `None` arm): calibration and boundary
// counts eight lanes at a time, the reconstruction terms in the scalar
// chains' order (term j joins chain j % 4 through two float64x2
// accumulators) and the bit planes gathered with a weighted horizontal add.
// Packed bytes and the inner product are bit-identical to the scalar kernel.
template <int BITS>
static double quantize_turbovec_hadamard_row_neon(
        const float * rotated,
        const float * shift,
        const float * scale_tq,
        const float * inv_scale_tq,
        const float * boundaries,
        const float * centroids,
        int dim,
        uint8_t * packed_row) {
    const size_t bytes_per_plane = static_cast<size_t>(dim) / 8;
    float64x2_t acc_a = vdupq_n_f64(0.0);
    float64x2_t acc_b = vdupq_n_f64(0.0);
    static const uint8_t kWeights[8] = { 128, 64, 32, 16, 8, 4, 2, 1 };
    const uint8x8_t weights = vld1_u8(kWeights);
    for (int chunk = 0; chunk < dim / 8; ++chunk) {
        const int offset = chunk * 8;
        const float32x4_t vals_lo = vmulq_f32(
            vaddq_f32(vld1q_f32(rotated + offset), vld1q_f32(shift + offset)), vld1q_f32(scale_tq + offset));
        const float32x4_t vals_hi = vmulq_f32(
            vaddq_f32(vld1q_f32(rotated + offset + 4), vld1q_f32(shift + offset + 4)),
            vld1q_f32(scale_tq + offset + 4));
        uint32x4_t acc_lo = vdupq_n_u32(0);
        uint32x4_t acc_hi = vdupq_n_u32(0);
        if (BITS == 4) {
            // Two-level scan: one compare against the median boundary picks
            // the 7-boundary half to count; x > b[7] implies x exceeds all of
            // b[0..7], so the count equals the flat scan's.
            const float32x4_t mid = vdupq_n_f32(boundaries[7]);
            const uint32x4_t m_lo = vcgtq_f32(vals_lo, mid);
            const uint32x4_t m_hi = vcgtq_f32(vals_hi, mid);
            acc_lo = vshlq_n_u32(vshrq_n_u32(m_lo, 31), 3);
            acc_hi = vshlq_n_u32(vshrq_n_u32(m_hi, 31), 3);
            for (int k = 0; k < 7; ++k) {
                const float32x4_t b_low = vdupq_n_f32(boundaries[k]);
                const float32x4_t b_high = vdupq_n_f32(boundaries[8 + k]);
                const float32x4_t bv_lo = vbslq_f32(m_lo, b_high, b_low);
                const float32x4_t bv_hi = vbslq_f32(m_hi, b_high, b_low);
                acc_lo = vaddq_u32(acc_lo, vshrq_n_u32(vcgtq_f32(vals_lo, bv_lo), 31));
                acc_hi = vaddq_u32(acc_hi, vshrq_n_u32(vcgtq_f32(vals_hi, bv_hi), 31));
            }
        } else {
            for (int b = 0; b < (1 << BITS) - 1; ++b) {
                const float32x4_t bv = vdupq_n_f32(boundaries[b]);
                acc_lo = vaddq_u32(acc_lo, vshrq_n_u32(vcgtq_f32(vals_lo, bv), 31));
                acc_hi = vaddq_u32(acc_hi, vshrq_n_u32(vcgtq_f32(vals_hi, bv), 31));
            }
        }
        uint8_t counts[8];
        counts[0] = static_cast<uint8_t>(vgetq_lane_u32(acc_lo, 0));
        counts[1] = static_cast<uint8_t>(vgetq_lane_u32(acc_lo, 1));
        counts[2] = static_cast<uint8_t>(vgetq_lane_u32(acc_lo, 2));
        counts[3] = static_cast<uint8_t>(vgetq_lane_u32(acc_lo, 3));
        counts[4] = static_cast<uint8_t>(vgetq_lane_u32(acc_hi, 0));
        counts[5] = static_cast<uint8_t>(vgetq_lane_u32(acc_hi, 1));
        counts[6] = static_cast<uint8_t>(vgetq_lane_u32(acc_hi, 2));
        counts[7] = static_cast<uint8_t>(vgetq_lane_u32(acc_hi, 3));

        double terms[8];
        for (int k = 0; k < 8; ++k) {
            const int d = offset + k;
            const double reconstructed =
                static_cast<double>(centroids[counts[k]]) * static_cast<double>(inv_scale_tq[d]) -
                static_cast<double>(shift[d]);
            terms[k] = static_cast<double>(rotated[d]) * reconstructed;
        }
        acc_a = vaddq_f64(acc_a, vld1q_f64(terms));
        acc_b = vaddq_f64(acc_b, vld1q_f64(terms + 2));
        acc_a = vaddq_f64(acc_a, vld1q_f64(terms + 4));
        acc_b = vaddq_f64(acc_b, vld1q_f64(terms + 6));

        const uint8x8_t codes = vld1_u8(counts);
        for (int plane = 0; plane < BITS; ++plane) {
            const uint8x8_t hit = vtst_u8(codes, vdup_n_u8(static_cast<uint8_t>(1u << plane)));
            packed_row[static_cast<size_t>(plane) * bytes_per_plane + static_cast<size_t>(chunk)] =
                vaddv_u8(vand_u8(hit, weights));
        }
    }
    return (vgetq_lane_f64(acc_a, 0) + vgetq_lane_f64(acc_a, 1)) +
        (vgetq_lane_f64(acc_b, 0) + vgetq_lane_f64(acc_b, 1));
}
#endif

static double quantize_turbovec_hadamard_row_with(
        TurboVecSimd simd,
        const float * rotated,
        const float * shift,
        const float * scale_tq,
        const float * inv_scale_tq,
        const float * boundaries,
        const float * centroids,
        int bits,
        int dim,
        uint8_t * packed_row) {
    switch (simd) {
#if GGML_VEC_INDEX_USE_NEON
        case TurboVecSimd::neon:
            if (bits == 2) {
                return quantize_turbovec_hadamard_row_neon<2>(
                    rotated, shift, scale_tq, inv_scale_tq, boundaries, centroids, dim, packed_row);
            }
            if (bits == 3) {
                return quantize_turbovec_hadamard_row_neon<3>(
                    rotated, shift, scale_tq, inv_scale_tq, boundaries, centroids, dim, packed_row);
            }
            if (bits == 4) {
                return quantize_turbovec_hadamard_row_neon<4>(
                    rotated, shift, scale_tq, inv_scale_tq, boundaries, centroids, dim, packed_row);
            }
            break;
#endif
#if defined(GGML_VEC_INDEX_HAVE_AVX2_KERNEL)
        case TurboVecSimd::avx2:
        case TurboVecSimd::avx512:
            if (bits >= 2 && bits <= 4) {
                return ggml_vec_index_detail::turbovec_quantize_row_avx2(
                    rotated, shift, scale_tq, inv_scale_tq, boundaries, centroids, bits, dim, packed_row);
            }
            break;
#endif
        default:
            break;
    }
    return quantize_turbovec_hadamard_row_scalar(
        rotated, shift, scale_tq, inv_scale_tq, boundaries, centroids, bits, dim, packed_row);
}

// Quantize already-rotated unit rows under (shift, scale_tq) and pack one
// bit-plane row each, returning the per-vector correction scale (upstream
// encode.rs `fused_quantize_scale_pack` + `scale_from_inner`).
static float quantize_turbovec_hadamard_row(
        const float * rotated,
        float norm,
        const float * shift,
        const float * scale_tq,
        const float * inv_scale_tq,
        const TurboVecCodebook & codebook,
        int bits,
        int dim,
        uint8_t * packed_row) {
    const double inner = quantize_turbovec_hadamard_row_with(
        turbovec_host_simd(),
        rotated,
        shift,
        scale_tq,
        inv_scale_tq,
        codebook.boundaries.data(),
        codebook.centroids.data(),
        bits,
        dim,
        packed_row);
    // Written positively so a NaN inner lands in the degenerate branch.
    return inner > kTurboVecDegenerateInner ? norm / static_cast<float>(inner) : 0.0f;
}

#ifdef GGML_VEC_INDEX_TEST_HOOKS
// 1 when the fused quantize-and-pack kernel of `kernel` (0: NEON, 1: AVX2)
// writes the scalar kernel's packed bytes and inner product bit for bit at
// 2, 3 and 4 bits, with and without a calibration, including values that
// land exactly on boundaries and zero rows; 0 on a mismatch; -1 when
// unavailable.
int turbovec_encode_kernel_parity_for_test(int kernel) {
    TurboVecSimd simd = TurboVecSimd::scalar;
    if (kernel == 2 || !turbovec_simd_for_test(kernel, simd)) {
        return -1;
    }
    const ScopedNearestRounding rounding_guard;
    TurboVecTestRng rng{ UINT64_C(0x243f6a8885a308d3) };
    for (int bits = 2; bits <= 4; ++bits) {
        for (int dim : { 8, 64, 200, 768, 1024 }) {
            const TurboVecCodebook & codebook = turbovec_codebook(bits, dim, false);
            const size_t dim_sz = static_cast<size_t>(dim);
            const size_t row_bytes = turbovec_row_bytes(bits, dim_sz);
            const float spread = 3.0f / std::sqrt(static_cast<float>(dim));
            for (int calibrated = 0; calibrated <= 1; ++calibrated) {
                std::vector<float> shift(dim_sz, 0.0f);
                std::vector<float> scale(dim_sz, 1.0f);
                std::vector<float> inv_scale(dim_sz, 1.0f);
                if (calibrated != 0) {
                    for (size_t d = 0; d < dim_sz; ++d) {
                        shift[d] = rng.uniform() * spread * 0.2f;
                        scale[d] = 0.75f + rng.uniform();
                        inv_scale[d] = 1.0f / scale[d];
                    }
                }
                for (int trial = 0; trial < 6; ++trial) {
                    std::vector<float> row(dim_sz);
                    for (size_t d = 0; d < dim_sz; ++d) {
                        row[d] = trial == 0 ? 0.0f : rng.uniform() * spread;
                    }
                    if (trial == 1 && calibrated == 0) {
                        // Exact boundary hits: a strict compare keeps them low.
                        for (size_t d = 0; d < dim_sz; ++d) {
                            row[d] = codebook.boundaries[d % codebook.boundaries.size()];
                        }
                    }
                    std::vector<uint8_t> expect(row_bytes, 0xa5);
                    std::vector<uint8_t> got(row_bytes, 0x5a);
                    const double expect_inner = quantize_turbovec_hadamard_row_scalar(
                        row.data(), shift.data(), scale.data(), inv_scale.data(), codebook.boundaries.data(),
                        codebook.centroids.data(), bits, dim, expect.data());
                    const double got_inner = quantize_turbovec_hadamard_row_with(
                        simd, row.data(), shift.data(), scale.data(), inv_scale.data(), codebook.boundaries.data(),
                        codebook.centroids.data(), bits, dim, got.data());
                    if (got != expect || std::memcmp(&got_inner, &expect_inner, sizeof(double)) != 0) {
                        return 0;
                    }
                }
            }
        }
    }
    return 1;
}
#endif

struct TurboVecCalibrationView {
    std::vector<float> identity_shift;
    std::vector<float> identity_scale;
    const float * shift = nullptr;
    const float * scale = nullptr;
    std::vector<float> inverse_scale;

    TurboVecCalibrationView(const std::vector<float> & tq_shift, const std::vector<float> & tq_scale, int dim) {
        const size_t dim_sz = static_cast<size_t>(dim);
        if (tq_shift.empty()) {
            identity_shift.assign(dim_sz, 0.0f);
            identity_scale.assign(dim_sz, 1.0f);
            shift = identity_shift.data();
            scale = identity_scale.data();
        } else {
            if (tq_shift.size() != dim_sz || tq_scale.size() != dim_sz) {
                throw std::invalid_argument("invalid TurboVec TQ+ calibration");
            }
            shift = tq_shift.data();
            scale = tq_scale.data();
        }
        inverse_scale.resize(dim_sz);
        for (size_t i = 0; i < dim_sz; ++i) {
            inverse_scale[i] = 1.0f / scale[i];
        }
    }
};

// Rows per parallel task: large enough to amortize the hand-off.
constexpr size_t kTurboVecRowsPerTask = 256;

void quantize_turbovec_hadamard_batch(
        const float * src,
        int n,
        int bits,
        uint8_t * dst,
        float * scales,
        int dim,
        const std::vector<float> & tqplus_shift,
        const std::vector<float> & tqplus_scale,
        int n_threads) {
    const size_t dim_sz = static_cast<size_t>(dim);
    const size_t row_bytes = turbovec_row_bytes(bits, dim_sz);
    const TurboVecCodebook & codebook = turbovec_codebook(bits, dim, false);
    const std::shared_ptr<const TurboVecHadamardRotation> rotation = turbovec_hadamard_rotation(dim);
    const TurboVecCalibrationView calibration(tqplus_shift, tqplus_scale, dim);
    // Rows are independent and every row's arithmetic is fixed, so the
    // bytes do not depend on how rows are spread over threads.
    const size_t n_sz = static_cast<size_t>(n);
    const size_t n_tasks = (n_sz + kTurboVecRowsPerTask - 1) / kTurboVecRowsPerTask;
    parallel_for(n_tasks, n_threads, [&](size_t task, size_t) {
        const ScopedNearestRounding rounding_guard;
        std::vector<float> rotated(dim_sz);
        std::vector<float> scratch(dim_sz);
        const size_t end = std::min(n_sz, (task + 1) * kTurboVecRowsPerTask);
        for (size_t row = task * kTurboVecRowsPerTask; row < end; ++row) {
            const float * input = src + row * dim_sz;
            const float norm = turbovec_hadamard_norm(input, dim);
            const float inv = norm > kTurboVecMinInputNorm ? 1.0f / norm : 0.0f;
            apply_turbovec_hadamard(*rotation, input, inv, rotated.data(), scratch.data());
            scales[row] = quantize_turbovec_hadamard_row(
                rotated.data(),
                norm,
                calibration.shift,
                calibration.scale,
                calibration.inverse_scale.data(),
                codebook,
                bits,
                dim,
                dst + row * row_bytes);
        }
    });
}

// Total-order sort key for f32 (upstream encode.rs `f32_sort_key`): ranking
// keys and mapping back gives the same order statistic as ranking values.
static uint32_t turbovec_f32_sort_key(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits ^ (static_cast<uint32_t>(static_cast<int32_t>(bits) >> 31) | 0x80000000u);
}

static float turbovec_f32_from_sort_key(uint32_t key) {
    const uint32_t mask = (key & 0x80000000u) != 0 ? 0x80000000u : 0xffffffffu;
    const uint32_t bits = key ^ mask;
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

int fit_turbovec_calibration(
        const float * sample,
        int n,
        int bits,
        int dim,
        std::vector<float> & out_shift,
        std::vector<float> & out_scale,
        int n_threads) {
    const ScopedNearestRounding rounding_guard;
    if (n < kTurboVecMinCalibrationRows) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    const size_t dim_sz = static_cast<size_t>(dim);
    const size_t n_sz = static_cast<size_t>(n);
    const std::shared_ptr<const TurboVecHadamardRotation> rotation = turbovec_hadamard_rotation(dim);
    std::vector<float> rotated(n_sz * dim_sz);
    parallel_for((n_sz + kTurboVecRowsPerTask - 1) / kTurboVecRowsPerTask, n_threads, [&](size_t task, size_t) {
        const ScopedNearestRounding task_rounding;
        std::vector<float> scratch(dim_sz);
        const size_t end = std::min(n_sz, (task + 1) * kTurboVecRowsPerTask);
        for (size_t row = task * kTurboVecRowsPerTask; row < end; ++row) {
            const float * input = sample + row * dim_sz;
            const float norm = turbovec_hadamard_norm(input, dim);
            const float inv = norm > kTurboVecMinInputNorm ? 1.0f / norm : 0.0f;
            apply_turbovec_hadamard(*rotation, input, inv, rotated.data() + row * dim_sz, scratch.data());
        }
    });

    // Anchor on the codebook's outermost centroid (upstream #454): the
    // probability level P(|x| <= c_outer) moves with bit width.
    const TurboVecCodebook & codebook = turbovec_codebook(bits, dim, false);
    float c_outer = 0.0f;
    for (float centroid : codebook.centroids) {
        c_outer = std::max(c_outer, std::fabs(centroid));
    }
    const double beta_a = (static_cast<double>(dim) - 1.0) * 0.5;
    const double p_hi = regularized_beta((static_cast<double>(c_outer) + 1.0) / 2.0, beta_a, beta_a);
    const double p_lo = 1.0 - p_hi;
    const float qc_lo = -c_outer;
    const float qc_hi = c_outer;
    const float qc_span = qc_hi - qc_lo;
    const size_t lo_index = static_cast<size_t>(static_cast<double>(n_sz) * p_lo);
    const size_t hi_index = std::max(
        std::min(static_cast<size_t>(static_cast<double>(n_sz) * p_hi), n_sz - 1),
        lo_index + 1);

    std::vector<float> shift(dim_sz, 0.0f);
    std::vector<float> scale(dim_sz, 1.0f);
    // Each coordinate's fit is independent: an order statistic of its column.
    constexpr size_t kColumnsPerTask = 16;
    parallel_for((dim_sz + kColumnsPerTask - 1) / kColumnsPerTask, n_threads, [&](size_t task, size_t) {
        const ScopedNearestRounding task_rounding;
        std::vector<uint32_t> keys(n_sz);
        const size_t column_end = std::min(dim_sz, (task + 1) * kColumnsPerTask);
        for (size_t column = task * kColumnsPerTask; column < column_end; ++column) {
            for (size_t row = 0; row < n_sz; ++row) {
                keys[row] = turbovec_f32_sort_key(rotated[row * dim_sz + column]);
            }
            std::nth_element(keys.begin(), keys.begin() + static_cast<std::ptrdiff_t>(lo_index), keys.end());
            const float qe_lo = turbovec_f32_from_sort_key(keys[lo_index]);
            std::nth_element(
                keys.begin() + static_cast<std::ptrdiff_t>(lo_index) + 1,
                keys.begin() + static_cast<std::ptrdiff_t>(hi_index),
                keys.end());
            const float qe_hi = turbovec_f32_from_sort_key(keys[hi_index]);
            const float qe_span = qe_hi - qe_lo;
            if (qe_span > 1e-6f) {
                scale[column] = qc_span / qe_span;
                shift[column] = qc_lo / scale[column] - qe_lo;
            }
        }
    });

    bool identity = true;
    for (size_t i = 0; i < dim_sz && identity; ++i) {
        identity = shift[i] == 0.0f && scale[i] == 1.0f;
    }
    if (identity) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    out_shift.swap(shift);
    out_scale.swap(scale);
    return GGML_VEC_INDEX_OK;
}

void reencode_turbovec_rows(
        int bits,
        int dim,
        size_t n,
        const uint8_t * codes,
        const float * scales,
        const std::vector<float> & old_shift,
        const std::vector<float> & old_scale,
        const std::vector<float> & new_shift,
        const std::vector<float> & new_scale,
        std::vector<uint8_t> & out_codes,
        std::vector<float> & out_scales,
        int n_threads) {
    const size_t dim_sz = static_cast<size_t>(dim);
    const size_t row_bytes = turbovec_row_bytes(bits, dim_sz);
    const TurboVecCodebook & codebook = turbovec_codebook(bits, dim, false);
    const float * centroids = codebook.centroids.data();
    const TurboVecCalibrationView old_calibration(old_shift, old_scale, dim);
    const TurboVecCalibrationView new_calibration(new_shift, new_scale, dim);
    std::vector<uint8_t> next_codes(n * row_bytes);
    std::vector<float> next_scales(n);
    parallel_for((n + kTurboVecRowsPerTask - 1) / kTurboVecRowsPerTask, n_threads, [&](size_t task, size_t) {
        const ScopedNearestRounding rounding_guard;
        std::vector<float> reconstructed(dim_sz);
        const size_t row_end = std::min(n, (task + 1) * kTurboVecRowsPerTask);
        for (size_t row = task * kTurboVecRowsPerTask; row < row_end; ++row) {
            // Rebuild the rotated row from its codes under the old pair, then
            // encode it under the new one (upstream lib.rs `reencode_stored_rows`).
            const uint8_t * packed = codes + row * row_bytes;
            double sum_squares = 0.0;
            for (int d = 0; d < dim; ++d) {
                const uint8_t code = turbovec_get_bitplane_code(packed, d, bits, dim);
                const float x =
                    centroids[code] * old_calibration.inverse_scale[static_cast<size_t>(d)] -
                    old_calibration.shift[d];
                reconstructed[static_cast<size_t>(d)] = x;
                sum_squares += static_cast<double>(x) * static_cast<double>(x);
            }
            const float norm = static_cast<float>(static_cast<double>(scales[row]) * sum_squares);
            next_scales[row] = quantize_turbovec_hadamard_row(
                reconstructed.data(),
                norm,
                new_calibration.shift,
                new_calibration.scale,
                new_calibration.inverse_scale.data(),
                codebook,
                bits,
                dim,
                next_codes.data() + row * row_bytes);
        }
    });
    out_codes.swap(next_codes);
    out_scales.swap(next_scales);
}

void quantize_turbovec_q2_row(const float * src, uint8_t * dst, float * scales, int dim) {
    std::vector<float> shift;
    std::vector<float> scale;
    quantize_turbovec_batch(src, 1, 2, dst, scales, dim, shift, scale);
}

void decode_turbovec_q2_row(const uint8_t * codes, const float * scales, float * dst, int dim) {
    decode_turbovec_q2_row_calibrated(codes, scales, nullptr, nullptr, dst, dim);
}

void decode_turbovec_q2_row_calibrated(
        const uint8_t * codes,
        const float * scales,
        const float * tqplus_shift,
        const float * tqplus_scale,
        float * dst,
        int dim) {
    decode_turbovec_row_calibrated(2, true, codes, scales, tqplus_shift, tqplus_scale, dst, dim);
}

float dot_turbovec_q2_rotated_row(const float * rotated_query, const uint8_t * codes, const float * scales, int dim) {
    std::vector<uint8_t> lut;
    float lut_scale = 1.0f;
    float lut_bias = 0.0f;
    build_turbovec_q2_lut(rotated_query, dim, lut, lut_scale, lut_bias);
    return dot_turbovec_q2_lut_row(lut.data(), lut_scale, lut_bias, codes, scales, dim);
}

float dot_turbovec_q2_row(const float * query, const uint8_t * codes, const float * scales, int dim) {
    std::vector<float> rotated(static_cast<size_t>(dim));
    rotate_turbovec_query(true, query, rotated.data(), dim);
    return dot_turbovec_q2_rotated_row(rotated.data(), codes, scales, dim);
}

void quantize_turbovec_q4_row(const float * src, uint8_t * dst, float * scales, int dim) {
    std::vector<float> shift;
    std::vector<float> scale;
    quantize_turbovec_batch(src, 1, 4, dst, scales, dim, shift, scale);
}

void build_turbovec_q2_lut(const float * rotated_query, int dim, std::vector<uint8_t> & lut, float & scale, float & bias) {
    build_turbovec_lut(2, true, rotated_query, dim, lut, scale, bias);
}

float dot_turbovec_q2_lut_row(
        const uint8_t * lut,
        float lut_scale,
        float lut_bias,
        const uint8_t * codes,
        const float * scales,
        int dim) {
    return dot_turbovec_lut_row(lut, lut_scale, lut_bias, codes, scales, 2, dim);
}

void decode_turbovec_q4_row(const uint8_t * codes, const float * scales, float * dst, int dim) {
    decode_turbovec_q4_row_calibrated(codes, scales, nullptr, nullptr, dst, dim);
}

void decode_turbovec_q4_row_calibrated(
        const uint8_t * codes,
        const float * scales,
        const float * tqplus_shift,
        const float * tqplus_scale,
        float * dst,
        int dim) {
    decode_turbovec_row_calibrated(4, true, codes, scales, tqplus_shift, tqplus_scale, dst, dim);
}

void build_turbovec_q4_lut(const float * rotated_query, int dim, std::vector<uint8_t> & lut, float & scale, float & bias) {
    build_turbovec_lut(4, true, rotated_query, dim, lut, scale, bias);
}

float dot_turbovec_q4_lut_row(
        const uint8_t * lut,
        float lut_scale,
        float lut_bias,
        const uint8_t * codes,
        const float * scales,
        int dim) {
    return dot_turbovec_lut_row(lut, lut_scale, lut_bias, codes, scales, 4, dim);
}

float dot_turbovec_q4_rotated_row(const float * rotated_query, const uint8_t * codes, const float * scales, int dim) {
    std::vector<uint8_t> lut;
    float lut_scale = 1.0f;
    float lut_bias = 0.0f;
    build_turbovec_q4_lut(rotated_query, dim, lut, lut_scale, lut_bias);
    return dot_turbovec_q4_lut_row(lut.data(), lut_scale, lut_bias, codes, scales, dim);
}

float dot_turbovec_q4_row(const float * query, const uint8_t * codes, const float * scales, int dim) {
    std::vector<float> rotated(static_cast<size_t>(dim));
    rotate_turbovec_query(true, query, rotated.data(), dim);
    return dot_turbovec_q4_rotated_row(rotated.data(), codes, scales, dim);
}
