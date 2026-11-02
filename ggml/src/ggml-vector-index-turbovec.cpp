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

namespace {

constexpr int kTurboVecDimMultiple = 8;
constexpr int kTurboVecScoreBlock = 32;
#if GGML_VEC_INDEX_USE_NEON
constexpr int kTurboVecFlushEvery = 256;
#endif
constexpr uint64_t kTurboVecRotationSeed = 42;
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

static TurboVecCodebook make_turbovec_codebook(int bits, int dim) {
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
    for (int i = 0; i < n_levels - 1; ++i) {
        out.boundaries[static_cast<size_t>(i)] =
            static_cast<float>((centroids[static_cast<size_t>(i)] + centroids[static_cast<size_t>(i) + 1]) * 0.5);
    }
    for (int i = 0; i < n_levels; ++i) {
        out.centroids[static_cast<size_t>(i)] = static_cast<float>(centroids[static_cast<size_t>(i)]);
    }
    return out;
}

static const TurboVecCodebook & turbovec_codebook(int bits, int dim) {
    static std::mutex mutex;
    static std::unordered_map<int, TurboVecCodebook> cache;
    const int key = bits * 1000000 + dim;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(key);
    if (it == cache.end()) {
        it = cache.emplace(key, make_turbovec_codebook(bits, dim)).first;
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
    const int dim_start = group * codes_per_byte;
    uint8_t byte = 0;
    for (int c = 0; c < codes_per_byte; ++c) {
        const uint8_t code = turbovec_get_bitplane_code(row, dim_start + c, bits, dim);
        const int shift = (codes_per_byte - 1 - c) * bits;
        byte |= static_cast<uint8_t>(code << shift);
    }
    return byte;
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

    out_scale = max_span > 1e-10f ? max_span / max_lut : 1.0f;
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

bool turbovec_q2_supported_dim(int dim) {
    return sizeof(size_t) >= 8 &&
        dim > 0 && dim <= kTurboVecMaxDim && dim % kTurboVecDimMultiple == 0;
}

bool turbovec_q4_supported_dim(int dim) {
    return sizeof(size_t) >= 8 &&
        dim > 0 && dim <= kTurboVecMaxDim && dim % kTurboVecDimMultiple == 0;
}

bool turbovec_serialized_dim_supported(int dim) {
    return sizeof(size_t) >= 8 &&
        dim > 0 && dim <= kTurboVecMaxSerializedDim && dim % kTurboVecDimMultiple == 0;
}

void turbovec_retain_rotation(int dim) {
    std::lock_guard<std::mutex> lock(turbovec_rotation_cache_mutex());
    TurboVecRotationCacheEntry & entry = turbovec_rotation_cache()[dim];
    ++entry.ref_count;
    if (entry.strong == nullptr) {
        entry.strong = entry.weak.lock();
    }
}

void turbovec_release_rotation(int dim) noexcept {
    try {
        std::lock_guard<std::mutex> lock(turbovec_rotation_cache_mutex());
        auto & cache = turbovec_rotation_cache();
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
    if (!turbovec_q4_supported_dim(dim)) {
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
    if (queries == nullptr || n_queries < 0 || !turbovec_q4_supported_dim(dim)) {
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
    if (queries == nullptr || n_queries < 0 || !turbovec_q4_supported_dim(dim)) {
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
        (bits != 2 && bits != 4) || n_queries <= 0 || !turbovec_q4_supported_dim(dim)) {
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
        turbovec_codebook(bits, dim).centroids.data(),
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
    if ((bits != 2 && bits != 4) || !turbovec_q4_supported_dim(dim)) {
        return 0;
    }
    uint64_t hash = UINT64_C(0xcbf29ce484222325);
    for (float value : turbovec_codebook(bits, dim).centroids) {
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

void prepare_turbovec(int bits, int dim) {
    if ((bits != 2 && bits != 4) || !turbovec_q4_supported_dim(dim)) {
        return;
    }
    (void) turbovec_rotation(dim);
    (void) turbovec_codebook(bits, dim);
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

void turbovec_clear_blocked_for_test(ggml_vec_index_t * idx) {
    if (idx == nullptr) {
        return;
    }
    try {
        std::unique_lock<std::shared_mutex> lock(idx->mutex);
        idx->turbovec_blocked_data.clear();
        idx->turbovec_blocked_n_blocks = 0;
    } catch (...) {
    }
}
#endif

size_t turbovec_q2_row_bytes(size_t dim) {
    return dim / 4;
}

size_t turbovec_q2_scale_count(size_t dim) {
    (void) dim;
    return 1;
}

size_t turbovec_q4_row_bytes(size_t dim) {
    return dim / 2;
}

size_t turbovec_q4_scale_count(size_t dim) {
    (void) dim;
    return 1;
}

static void repack_turbovec_code_block(
        const uint8_t * packed_codes,
        size_t n_vectors,
        int bits,
        int dim,
        size_t block,
        std::vector<uint8_t> & blocked_codes) {
    const size_t dim_sz = static_cast<size_t>(dim);
    const size_t codes_per_byte = static_cast<size_t>(8 / bits);
    const size_t n_byte_groups = dim_sz / codes_per_byte;
    const size_t row_bytes = static_cast<size_t>(bits) * (dim_sz / 8);
    const size_t base_vector = block * kTurboVecScoreBlock;
    const size_t block_offset =
        block * n_byte_groups * static_cast<size_t>(kTurboVecScoreBlock);
    std::fill_n(
        blocked_codes.data() + block_offset,
        n_byte_groups * static_cast<size_t>(kTurboVecScoreBlock),
        0);
    for (size_t group = 0; group < n_byte_groups; ++group) {
        const size_t output_offset =
            (block * n_byte_groups + group) * kTurboVecScoreBlock;
#if defined(__x86_64__) || defined(_M_X64)
        static constexpr size_t perm[16] = {
            0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15,
        };
        for (size_t j = 0; j < 16; ++j) {
            const size_t vector_a = base_vector + perm[j];
            const size_t vector_b = vector_a + 16;
            const uint8_t byte_a = vector_a < n_vectors ?
                turbovec_group_code_byte(
                    packed_codes + vector_a * row_bytes,
                    static_cast<int>(group),
                    bits,
                    dim) :
                0;
            const uint8_t byte_b = vector_b < n_vectors ?
                turbovec_group_code_byte(
                    packed_codes + vector_b * row_bytes,
                    static_cast<int>(group),
                    bits,
                    dim) :
                0;
            blocked_codes[output_offset + j] =
                static_cast<uint8_t>((byte_a >> 4) | ((byte_b >> 4) << 4));
            blocked_codes[output_offset + 16 + j] =
                static_cast<uint8_t>((byte_a & 0x0f) | ((byte_b & 0x0f) << 4));
        }
#else
        for (size_t lane = 0; lane < kTurboVecScoreBlock; ++lane) {
            const size_t vector = base_vector + lane;
            if (vector < n_vectors) {
                blocked_codes[output_offset + lane] =
                    turbovec_group_code_byte(
                        packed_codes + vector * row_bytes,
                        static_cast<int>(group),
                        bits,
                        dim);
            }
        }
#endif
    }
}

void repack_turbovec_codes(
        const uint8_t * packed_codes,
        size_t n_vectors,
        int bits,
        int dim,
        std::vector<uint8_t> & blocked_codes,
        size_t & n_blocks) {
    const size_t dim_sz = static_cast<size_t>(dim);
    const size_t codes_per_byte = static_cast<size_t>(8 / bits);
    const size_t n_byte_groups = dim_sz / codes_per_byte;
    n_blocks = (n_vectors + kTurboVecScoreBlock - 1) / kTurboVecScoreBlock;
    blocked_codes.assign(
        n_blocks * n_byte_groups * static_cast<size_t>(kTurboVecScoreBlock),
        0);

    for (size_t block = 0; block < n_blocks; ++block) {
        repack_turbovec_code_block(packed_codes, n_vectors, bits, dim, block, blocked_codes);
    }
}

void repack_turbovec_codes_from_slot(
        const uint8_t * packed_codes,
        size_t n_vectors,
        int bits,
        int dim,
        size_t first_slot,
        std::vector<uint8_t> & blocked_codes,
        size_t & n_blocks) {
    const size_t dim_sz = static_cast<size_t>(dim);
    const size_t codes_per_byte = static_cast<size_t>(8 / bits);
    const size_t n_byte_groups = dim_sz / codes_per_byte;
    const size_t old_expected_size =
        n_blocks * n_byte_groups * static_cast<size_t>(kTurboVecScoreBlock);
    if (first_slot == 0 || first_slot > n_vectors ||
        blocked_codes.size() != old_expected_size) {
        repack_turbovec_codes(packed_codes, n_vectors, bits, dim, blocked_codes, n_blocks);
        return;
    }

    const size_t first_block = first_slot / kTurboVecScoreBlock;
    n_blocks = (n_vectors + kTurboVecScoreBlock - 1) / kTurboVecScoreBlock;
    blocked_codes.resize(
        n_blocks * n_byte_groups * static_cast<size_t>(kTurboVecScoreBlock),
        0);
    for (size_t block = first_block; block < n_blocks; ++block) {
        repack_turbovec_code_block(packed_codes, n_vectors, bits, dim, block, blocked_codes);
    }
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

void rotate_turbovec_query(const float * src, float * dst, int dim) {
    rotate_turbovec_queries(src, dst, 1, dim);
}

void rotate_turbovec_queries(
        const float * src,
        float * dst,
        int n_queries,
        int dim) {
    apply_turbovec_query_rotation_batch(src, dst, n_queries, dim);
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
    const TurboVecCodebook & codebook = turbovec_codebook(bits, dim);

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
        if (!(inner > 1e-10)) {
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
    const TurboVecCodebook & codebook = turbovec_codebook(2, dim);
    const float * centroids = codebook.centroids.data();
    std::vector<float> rotated(static_cast<size_t>(dim));
    for (int i = 0; i < dim; ++i) {
        const uint8_t code = turbovec_get_bitplane_code(codes, i, 2, dim);
        const size_t coordinate = static_cast<size_t>(i);
        const float shift = tqplus_shift != nullptr ? tqplus_shift[coordinate] : 0.0f;
        const float calibration_scale = tqplus_scale != nullptr ? tqplus_scale[coordinate] : 1.0f;
        rotated[coordinate] =
            (centroids[code] / calibration_scale - shift) * scales[0];
    }
    apply_turbovec_rotation(rotated.data(), dst, dim, true);
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
    rotate_turbovec_query(query, rotated.data(), dim);
    return dot_turbovec_q2_rotated_row(rotated.data(), codes, scales, dim);
}

void quantize_turbovec_q4_row(const float * src, uint8_t * dst, float * scales, int dim) {
    std::vector<float> shift;
    std::vector<float> scale;
    quantize_turbovec_batch(src, 1, 4, dst, scales, dim, shift, scale);
}

void build_turbovec_q2_lut(const float * rotated_query, int dim, std::vector<uint8_t> & lut, float & scale, float & bias) {
    const TurboVecCodebook & codebook = turbovec_codebook(2, dim);
    build_turbovec_query_lut(rotated_query, codebook.centroids.data(), 2, dim, lut, scale, bias);
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
    const TurboVecCodebook & codebook = turbovec_codebook(4, dim);
    const float * centroids = codebook.centroids.data();
    std::vector<float> rotated(static_cast<size_t>(dim));
    for (int i = 0; i < dim; ++i) {
        const uint8_t code = turbovec_get_bitplane_code(codes, i, 4, dim);
        const size_t coordinate = static_cast<size_t>(i);
        const float shift = tqplus_shift != nullptr ? tqplus_shift[coordinate] : 0.0f;
        const float calibration_scale = tqplus_scale != nullptr ? tqplus_scale[coordinate] : 1.0f;
        rotated[coordinate] =
            (centroids[code] / calibration_scale - shift) * scales[0];
    }
    apply_turbovec_rotation(rotated.data(), dst, dim, true);
}

void build_turbovec_q4_lut(const float * rotated_query, int dim, std::vector<uint8_t> & lut, float & scale, float & bias) {
    const TurboVecCodebook & codebook = turbovec_codebook(4, dim);
    build_turbovec_query_lut(rotated_query, codebook.centroids.data(), 4, dim, lut, scale, bias);
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
    rotate_turbovec_query(query, rotated.data(), dim);
    return dot_turbovec_q4_rotated_row(rotated.data(), codes, scales, dim);
}
