// bench-vector-index.cpp - f32 vs q8 vector-index quality and latency smoke bench.

#include "ggml-vector-index.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <system_error>
#include <unordered_set>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

namespace {

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);\
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

constexpr int kModeTurboVecQ2 = -2;
constexpr int kModeTurboVecQ4 = -4;

struct BenchConfig {
    int n_vec   = 2048;
    int dim     = 256;
    int n_query = 64;
    int k       = 10;
    int warmups = 2;
    int repeats = 7;
    int delta_ops = 256;
    int ivf_lists = 64;
    int ivf_iters = 4;
    int ivf_nprobe = 4;
    int delete_stride = 2;
};

enum class ParseResult {
    ok,
    help,
    error,
};

struct TimedSearch {
    double ms = 0.0;
    std::vector<float> scores;
    std::vector<uint64_t> ids;
};

struct QualityMetrics {
    double recall_at_k = 0.0;
    double mean_abs_score_drift = 0.0;
    double max_abs_score_drift = 0.0;
};

struct QualityBenchResult {
    const char * name = "";
    QualityMetrics q8;
    QualityMetrics q4;
    QualityMetrics tvq2;
    QualityMetrics tvq4;
    std::vector<QualityMetrics> q4_calibrated;
};

struct Q4CalibrationMode {
    const char * name = "";
    float percentile = 1.0f;
    float rms_factor = 0.0f;
};

struct Q4SimulatedIndex {
    int dim = 0;
    std::vector<int8_t> codes;
    std::vector<float> scales;
};

struct ScoreId {
    float score = 0.0f;
    uint64_t id = 0;
};

int round_nearest_even(float value) {
    const float lower_f = std::floor(value);
    const float upper_f = lower_f + 1.0f;
    const float lower_dist = value - lower_f;
    const float upper_dist = upper_f - value;
    if (lower_dist < upper_dist) {
        return static_cast<int>(lower_f);
    }
    if (upper_dist < lower_dist) {
        return static_cast<int>(upper_f);
    }

    const int lower = static_cast<int>(lower_f);
    return (lower % 2) == 0 ? lower : static_cast<int>(upper_f);
}

template <typename Fn>
double median_time_ms(int warmups, int repeats, Fn fn) {
    for (int i = 0; i < warmups; ++i) {
        fn();
    }

    std::vector<double> times;
    times.reserve(static_cast<size_t>(repeats));
    for (int i = 0; i < repeats; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        const auto t1 = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

std::filesystem::path make_temp_dir() {
    const auto base = std::filesystem::temp_directory_path();
    std::random_device rd;
    for (int attempt = 0; attempt < 32; ++attempt) {
        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        const std::string name =
            "ggml-vector-index-bench-" +
            std::to_string(static_cast<unsigned long long>(now)) + "-" +
            std::to_string(static_cast<unsigned long long>(rd()));
        std::error_code ec;
        const auto path = base / name;
        if (std::filesystem::create_directory(path, ec)) {
            return path;
        }
    }
    CHECK(false);
    return base / "ggml-vector-index-bench";
}

void remove_delta_artifacts(const std::filesystem::path & delta_path) {
    std::filesystem::remove(delta_path);
    std::filesystem::remove(delta_path.string() + ".lock");
}

bool x86_cpu_has_avx2() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int regs[4] = {};
    __cpuid(regs, 0);
    if (regs[0] < 7) {
        return false;
    }
    __cpuidex(regs, 1, 0);
    constexpr int kOsxsave = 1 << 27;
    constexpr int kAvx = 1 << 28;
    constexpr int kFma = 1 << 12;
    if ((regs[2] & (kOsxsave | kAvx)) != (kOsxsave | kAvx) ||
        (regs[2] & kFma) == 0 ||
        (_xgetbv(0) & 0x6) != 0x6) {
        return false;
    }
    __cpuidex(regs, 7, 0);
    return (regs[1] & (1 << 5)) != 0;
#elif (defined(__GNUC__) || defined(__clang__)) && \
      (defined(__x86_64__) || defined(__i386__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

const char * quantized_kernel_name() {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    return "arm-neon";
#elif defined(GGML_VEC_INDEX_HAVE_AVX2_KERNEL) && \
      (defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__))
    return x86_cpu_has_avx2() ? "avx2" : "scalar";
#else
    return "scalar";
#endif
}

const char * q8_kernel_name() {
    return quantized_kernel_name();
}

const char * q4_kernel_name() {
    return quantized_kernel_name();
}

const char * turbovec_kernel_name() {
#if defined(GGML_VEC_INDEX_HAVE_AVX2_KERNEL) && \
      (defined(_M_X64) || defined(__x86_64__))
    return x86_cpu_has_avx2() ? "avx2-lut" : "scalar-lut";
#else
    return "scalar-lut";
#endif
}

ggml_vec_index_t * create_index_mode(int mode, int dim) {
    if (mode == kModeTurboVecQ2) {
        return ggml_vec_index_create_turbovec_q2(dim);
    }
    if (mode == kModeTurboVecQ4) {
        return ggml_vec_index_create_turbovec_q4(dim);
    }
    return ggml_vec_index_create(dim, mode);
}

std::vector<float> make_normalized_vectors(int n, int dim, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> vectors(static_cast<size_t>(n) * static_cast<size_t>(dim));
    for (int row = 0; row < n; ++row) {
        float norm2 = 0.0f;
        float * v = vectors.data() + static_cast<size_t>(row) * static_cast<size_t>(dim);
        for (int i = 0; i < dim; ++i) {
            v[i] = dist(rng);
            norm2 += v[i] * v[i];
        }
        const float inv_norm = norm2 > 0.0f ? 1.0f / std::sqrt(norm2) : 1.0f;
        for (int i = 0; i < dim; ++i) {
            v[i] *= inv_norm;
        }
    }
    return vectors;
}

std::vector<float> make_gaussian_vectors(int n, int dim, uint32_t seed, bool normalize_rows) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> vectors(static_cast<size_t>(n) * static_cast<size_t>(dim));
    for (int row = 0; row < n; ++row) {
        float norm2 = 0.0f;
        float * v = vectors.data() + static_cast<size_t>(row) * static_cast<size_t>(dim);
        for (int i = 0; i < dim; ++i) {
            v[i] = dist(rng);
            norm2 += v[i] * v[i];
        }
        if (normalize_rows) {
            const float inv_norm = norm2 > 0.0f ? 1.0f / std::sqrt(norm2) : 1.0f;
            for (int i = 0; i < dim; ++i) {
                v[i] *= inv_norm;
            }
        }
    }
    return vectors;
}

std::vector<float> make_sparse_vectors(int n, int dim, uint32_t seed, int nnz) {
    CHECK(nnz > 0 && nnz <= dim);
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::uniform_int_distribution<int> coord_dist(0, dim - 1);

    std::vector<float> vectors(static_cast<size_t>(n) * static_cast<size_t>(dim), 0.0f);
    for (int row = 0; row < n; ++row) {
        float * v = vectors.data() + static_cast<size_t>(row) * static_cast<size_t>(dim);
        for (int j = 0; j < nnz; ++j) {
            v[coord_dist(rng)] += dist(rng);
        }
        float norm2 = 0.0f;
        for (int i = 0; i < dim; ++i) {
            norm2 += v[i] * v[i];
        }
        const float inv_norm = norm2 > 0.0f ? 1.0f / std::sqrt(norm2) : 1.0f;
        for (int i = 0; i < dim; ++i) {
            v[i] *= inv_norm;
        }
    }
    return vectors;
}

std::vector<float> make_cluster_centers(int dim, uint32_t seed, int n_clusters) {
    CHECK(n_clusters > 0);
    std::mt19937 rng(seed);
    std::normal_distribution<float> center_dist(0.0f, 1.0f);
    std::vector<float> centers(static_cast<size_t>(n_clusters) * static_cast<size_t>(dim));
    for (int cluster = 0; cluster < n_clusters; ++cluster) {
        float norm2 = 0.0f;
        float * center = centers.data() + static_cast<size_t>(cluster) * static_cast<size_t>(dim);
        for (int i = 0; i < dim; ++i) {
            center[i] = center_dist(rng);
            norm2 += center[i] * center[i];
        }
        const float inv_norm = norm2 > 0.0f ? 1.0f / std::sqrt(norm2) : 1.0f;
        for (int i = 0; i < dim; ++i) {
            center[i] *= inv_norm;
        }
    }
    return centers;
}

std::vector<float> make_clustered_vectors(
        int n,
        int dim,
        uint32_t seed,
        const std::vector<float> & centers,
        int n_clusters) {
    CHECK(n_clusters > 0);
    CHECK(centers.size() == static_cast<size_t>(n_clusters) * static_cast<size_t>(dim));
    std::mt19937 rng(seed);
    std::normal_distribution<float> noise_dist(0.0f, 0.05f);
    std::vector<float> vectors(static_cast<size_t>(n) * static_cast<size_t>(dim));
    for (int row = 0; row < n; ++row) {
        const float * center =
            centers.data() + static_cast<size_t>(row % n_clusters) * static_cast<size_t>(dim);
        float * v = vectors.data() + static_cast<size_t>(row) * static_cast<size_t>(dim);
        float norm2 = 0.0f;
        for (int i = 0; i < dim; ++i) {
            v[i] = center[i] + noise_dist(rng);
            norm2 += v[i] * v[i];
        }
        const float inv_norm = norm2 > 0.0f ? 1.0f / std::sqrt(norm2) : 1.0f;
        for (int i = 0; i < dim; ++i) {
            v[i] *= inv_norm;
        }
    }
    return vectors;
}

TimedSearch run_search(
        const ggml_vec_index_t * idx,
        const std::vector<float> & queries,
        int n_query,
        int k,
        int warmups,
        int repeats) {
    TimedSearch result;
    result.scores.resize(static_cast<size_t>(n_query) * static_cast<size_t>(k));
    result.ids.resize(static_cast<size_t>(n_query) * static_cast<size_t>(k));

    result.ms = median_time_ms(warmups, repeats, [&]() {
        CHECK(ggml_vec_index_search(
            idx,
            queries.data(),
            n_query,
            k,
            result.scores.data(),
            result.ids.data()) == GGML_VEC_INDEX_OK);
    });
    return result;
}

TimedSearch run_ivf_search(
        const ggml_vec_index_t * idx,
        const std::vector<float> & queries,
        int n_query,
        int k,
        int nprobe,
        int warmups,
        int repeats) {
    TimedSearch result;
    result.scores.resize(static_cast<size_t>(n_query) * static_cast<size_t>(k));
    result.ids.resize(static_cast<size_t>(n_query) * static_cast<size_t>(k));

    result.ms = median_time_ms(warmups, repeats, [&]() {
        CHECK(ggml_vec_index_search_ivf(
            idx,
            queries.data(),
            n_query,
            k,
            nprobe,
            result.scores.data(),
            result.ids.data()) == GGML_VEC_INDEX_OK);
    });
    return result;
}

TimedSearch run_filtered_search(
        const ggml_vec_index_t * idx,
        const std::vector<float> & queries,
        int n_query,
        int k,
        const std::vector<uint64_t> & allowed_ids,
        int warmups,
        int repeats) {
    TimedSearch result;
    result.scores.resize(static_cast<size_t>(n_query) * static_cast<size_t>(k));
    result.ids.resize(static_cast<size_t>(n_query) * static_cast<size_t>(k));

    result.ms = median_time_ms(warmups, repeats, [&]() {
        CHECK(ggml_vec_index_search_filtered(
            idx,
            queries.data(),
            n_query,
            k,
            allowed_ids.data(),
            static_cast<int>(allowed_ids.size()),
            result.scores.data(),
            result.ids.data()) == GGML_VEC_INDEX_OK);
    });
    return result;
}

TimedSearch run_prepared_filtered_search(
        const ggml_vec_index_t * idx,
        const ggml_vec_index_filter_t * filter,
        const std::vector<float> & queries,
        int n_query,
        int k,
        int warmups,
        int repeats) {
    TimedSearch result;
    result.scores.resize(static_cast<size_t>(n_query) * static_cast<size_t>(k));
    result.ids.resize(static_cast<size_t>(n_query) * static_cast<size_t>(k));

    result.ms = median_time_ms(warmups, repeats, [&]() {
        CHECK(ggml_vec_index_search_prepared_filtered(
            idx,
            filter,
            queries.data(),
            n_query,
            k,
            result.scores.data(),
            result.ids.data()) == GGML_VEC_INDEX_OK);
    });
    return result;
}

float dot_exact(
        const std::vector<float> & vectors,
        const float * query,
        uint64_t id,
        int dim) {
    const size_t row = static_cast<size_t>(id - 1);
    const float * v = vectors.data() + row * static_cast<size_t>(dim);
    float acc = 0.0f;
    for (int i = 0; i < dim; ++i) {
        acc += query[i] * v[i];
    }
    return acc;
}

float q4_reference_abs_for_mode(const float * row, int dim, const Q4CalibrationMode & mode) {
    float max_abs = 0.0f;
    float sumsq = 0.0f;
    std::vector<float> abs_values;
    abs_values.reserve(static_cast<size_t>(dim));
    for (int i = 0; i < dim; ++i) {
        const float a = std::fabs(row[i]);
        max_abs = std::max(max_abs, a);
        sumsq += row[i] * row[i];
        abs_values.push_back(a);
    }
    if (max_abs == 0.0f) {
        return 0.0f;
    }

    if (mode.rms_factor > 0.0f) {
        const float rms = std::sqrt(sumsq / static_cast<float>(dim));
        return std::min(max_abs, mode.rms_factor * rms);
    }

    if (mode.percentile < 1.0f) {
        std::sort(abs_values.begin(), abs_values.end());
        const size_t rank = static_cast<size_t>(
            std::floor(mode.percentile * static_cast<float>(dim - 1)));
        return std::max(abs_values[rank], std::numeric_limits<float>::min());
    }

    return max_abs;
}

Q4SimulatedIndex build_simulated_q4(
        const std::vector<float> & vectors,
        int n,
        int dim,
        const Q4CalibrationMode & mode) {
    Q4SimulatedIndex sim;
    sim.dim = dim;
    sim.codes.resize(static_cast<size_t>(n) * static_cast<size_t>(dim));
    sim.scales.resize(static_cast<size_t>(n));
    for (int row = 0; row < n; ++row) {
        const float * src = vectors.data() + static_cast<size_t>(row) * static_cast<size_t>(dim);
        const float ref_abs = q4_reference_abs_for_mode(src, dim, mode);
        float scale = ref_abs == 0.0f ? 1.0f : ref_abs / 7.0f;
        if (scale == 0.0f) {
            scale = ref_abs;
        }
        sim.scales[static_cast<size_t>(row)] = scale;
        int8_t * dst = sim.codes.data() + static_cast<size_t>(row) * static_cast<size_t>(dim);
        for (int i = 0; i < dim; ++i) {
            const float scaled = src[i] / scale;
            int q = round_nearest_even(scaled);
            q = std::max(-7, std::min(7, q));
            dst[i] = static_cast<int8_t>(q);
        }
    }
    return sim;
}

float dot_simulated_q4(const Q4SimulatedIndex & sim, const float * query, int row) {
    const int8_t * codes =
        sim.codes.data() + static_cast<size_t>(row) * static_cast<size_t>(sim.dim);
    const float scale = sim.scales[static_cast<size_t>(row)];
    float acc = 0.0f;
    for (int i = 0; i < sim.dim; ++i) {
        acc += query[i] * (static_cast<float>(codes[i]) * scale);
    }
    return acc;
}

TimedSearch run_simulated_q4_search(
        const Q4SimulatedIndex & sim,
        const std::vector<uint64_t> & ids,
        const std::vector<float> & queries,
        int n_query,
        int k) {
    TimedSearch result;
    result.scores.resize(static_cast<size_t>(n_query) * static_cast<size_t>(k));
    result.ids.resize(static_cast<size_t>(n_query) * static_cast<size_t>(k));
    std::vector<ScoreId> candidates(ids.size());
    for (int q = 0; q < n_query; ++q) {
        const float * query = queries.data() + static_cast<size_t>(q) * static_cast<size_t>(sim.dim);
        for (size_t row = 0; row < ids.size(); ++row) {
            candidates[row] = {
                dot_simulated_q4(sim, query, static_cast<int>(row)),
                ids[row],
            };
        }
        std::partial_sort(
            candidates.begin(),
            candidates.begin() + k,
            candidates.end(),
            [](const ScoreId & a, const ScoreId & b) {
                return a.score > b.score;
            });
        for (int i = 0; i < k; ++i) {
            const size_t out = static_cast<size_t>(q) * static_cast<size_t>(k) +
                static_cast<size_t>(i);
            result.scores[out] = candidates[static_cast<size_t>(i)].score;
            result.ids[out] = candidates[static_cast<size_t>(i)].id;
        }
    }
    return result;
}

std::filesystem::path write_index_file(
        ggml_vec_index_t * idx,
        const std::filesystem::path & temp_dir,
        const char * name) {
    const auto path = temp_dir / name;
    CHECK(ggml_vec_index_write(idx, path.string().c_str()) == GGML_VEC_INDEX_OK);
    return path;
}

std::vector<uint8_t> read_file_bytes(const std::filesystem::path & path) {
    std::ifstream f(path, std::ios::binary);
    CHECK(f.is_open());
    const auto size = std::filesystem::file_size(path);
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    if (!bytes.empty()) {
        f.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        CHECK(f.gcount() == static_cast<std::streamsize>(bytes.size()));
    }
    return bytes;
}

void write_file_bytes(const std::filesystem::path & path, const std::vector<uint8_t> & bytes) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    CHECK(f.is_open());
    if (!bytes.empty()) {
        f.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    CHECK(static_cast<bool>(f));
}

std::vector<uint64_t> make_allowlist(int n_vec, int n_allowed) {
    std::vector<uint64_t> allowed;
    allowed.reserve(static_cast<size_t>(n_allowed));
    for (int i = 0; i < n_allowed; ++i) {
        const int row = (i * 37) % n_vec;
        allowed.push_back(static_cast<uint64_t>(row) + 1);
    }
    std::sort(allowed.begin(), allowed.end());
    allowed.erase(std::unique(allowed.begin(), allowed.end()), allowed.end());
    return allowed;
}

double recall_against(const TimedSearch & exact, const TimedSearch & candidate, int n_query, int k) {
    int overlap = 0;
    for (int q = 0; q < n_query; ++q) {
        std::unordered_set<uint64_t> exact_topk;
        for (int j = 0; j < k; ++j) {
            exact_topk.insert(exact.ids[static_cast<size_t>(q) * k + j]);
        }
        for (int j = 0; j < k; ++j) {
            const uint64_t id = candidate.ids[static_cast<size_t>(q) * k + j];
            if (exact_topk.count(id) != 0) {
                ++overlap;
            }
        }
    }
    return static_cast<double>(overlap) / static_cast<double>(n_query * k);
}

QualityMetrics quality_against(
        const TimedSearch & exact,
        const TimedSearch & candidate,
        const std::vector<float> & vectors,
        const std::vector<float> & queries,
        int n_query,
        int k,
        int dim) {
    QualityMetrics metrics;
    int overlap = 0;
    int drift_count = 0;
    for (int q = 0; q < n_query; ++q) {
        std::unordered_set<uint64_t> exact_topk;
        for (int j = 0; j < k; ++j) {
            exact_topk.insert(exact.ids[static_cast<size_t>(q) * static_cast<size_t>(k) + j]);
        }

        const float * query = queries.data() + static_cast<size_t>(q) * static_cast<size_t>(dim);
        for (int j = 0; j < k; ++j) {
            const size_t pos = static_cast<size_t>(q) * static_cast<size_t>(k) +
                static_cast<size_t>(j);
            const uint64_t id = candidate.ids[pos];
            if (exact_topk.count(id) != 0) {
                ++overlap;
            }
            const float exact_score = dot_exact(vectors, query, id, dim);
            const double drift = std::fabs(static_cast<double>(exact_score) - candidate.scores[pos]);
            metrics.mean_abs_score_drift += drift;
            metrics.max_abs_score_drift = std::max(metrics.max_abs_score_drift, drift);
            ++drift_count;
        }
    }
    metrics.recall_at_k = static_cast<double>(overlap) /
        static_cast<double>(n_query * k);
    metrics.mean_abs_score_drift /= static_cast<double>(drift_count);
    return metrics;
}

QualityBenchResult run_quality_bench(
        const char * name,
        const std::vector<float> & vectors,
        const std::vector<float> & queries,
        const std::vector<Q4CalibrationMode> & q4_calibration_modes,
        const BenchConfig & cfg) {
    std::vector<uint64_t> ids(static_cast<size_t>(cfg.n_vec));
    for (int i = 0; i < cfg.n_vec; ++i) {
        ids[static_cast<size_t>(i)] = static_cast<uint64_t>(i) + 1;
    }

    ggml_vec_index_t * f32 = ggml_vec_index_create(cfg.dim, 32);
    ggml_vec_index_t * q8 = ggml_vec_index_create(cfg.dim, 8);
    ggml_vec_index_t * q4 = ggml_vec_index_create(cfg.dim, 4);
    ggml_vec_index_t * tvq2 = ggml_vec_index_create_turbovec_q2(cfg.dim);
    ggml_vec_index_t * tvq4 = ggml_vec_index_create_turbovec_q4(cfg.dim);
    CHECK(f32 != nullptr);
    CHECK(q8 != nullptr);
    CHECK(q4 != nullptr);
    CHECK(tvq2 != nullptr);
    CHECK(tvq4 != nullptr);
    CHECK(ggml_vec_index_add(f32, vectors.data(), cfg.n_vec, ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_add(q8, vectors.data(), cfg.n_vec, ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_add(q4, vectors.data(), cfg.n_vec, ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_add(tvq2, vectors.data(), cfg.n_vec, ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_add(tvq4, vectors.data(), cfg.n_vec, ids.data()) == GGML_VEC_INDEX_OK);

    const TimedSearch f32_res = run_search(
        f32, queries, cfg.n_query, cfg.k, cfg.warmups, cfg.repeats);
    const TimedSearch q8_res = run_search(
        q8, queries, cfg.n_query, cfg.k, cfg.warmups, cfg.repeats);
    const TimedSearch q4_res = run_search(
        q4, queries, cfg.n_query, cfg.k, cfg.warmups, cfg.repeats);
    const TimedSearch tvq2_res = run_search(
        tvq2, queries, cfg.n_query, cfg.k, cfg.warmups, cfg.repeats);
    const TimedSearch tvq4_res = run_search(
        tvq4, queries, cfg.n_query, cfg.k, cfg.warmups, cfg.repeats);

    QualityBenchResult result;
    result.name = name;
    result.q8 = quality_against(f32_res, q8_res, vectors, queries, cfg.n_query, cfg.k, cfg.dim);
    result.q4 = quality_against(f32_res, q4_res, vectors, queries, cfg.n_query, cfg.k, cfg.dim);
    result.tvq2 = quality_against(f32_res, tvq2_res, vectors, queries, cfg.n_query, cfg.k, cfg.dim);
    result.tvq4 = quality_against(f32_res, tvq4_res, vectors, queries, cfg.n_query, cfg.k, cfg.dim);
    result.q4_calibrated.reserve(q4_calibration_modes.size());
    for (const Q4CalibrationMode & mode : q4_calibration_modes) {
        const Q4SimulatedIndex sim = build_simulated_q4(vectors, cfg.n_vec, cfg.dim, mode);
        const TimedSearch sim_res = run_simulated_q4_search(
            sim, ids, queries, cfg.n_query, cfg.k);
        result.q4_calibrated.push_back(
            quality_against(f32_res, sim_res, vectors, queries, cfg.n_query, cfg.k, cfg.dim));
    }

    ggml_vec_index_free(f32);
    ggml_vec_index_free(q8);
    ggml_vec_index_free(q4);
    ggml_vec_index_free(tvq2);
    ggml_vec_index_free(tvq4);
    return result;
}

struct DeltaBenchResult {
    double add_batch_ms = 0.0;
    double add_batch_us_per_vector = 0.0;
    double add_single_us_per_op = 0.0;
    double remove_us_per_op = 0.0;
    double snapshot_load_ms = 0.0;
    double replay_load_ms = 0.0;
    double compact_ms = 0.0;
    double post_compact_load_ms = 0.0;
    uintmax_t snapshot_bytes_before = 0;
    uintmax_t delta_bytes_before = 0;
    uintmax_t snapshot_bytes_after = 0;
    uintmax_t delta_bytes_after = 0;
};

struct DeleteBenchResult {
    int deleted = 0;
    int live = 0;
    double full_search_ms = 0.0;
    double tombstone_search_ms = 0.0;
    double compact_ms = 0.0;
    double compacted_search_ms = 0.0;
};

DeleteBenchResult run_delete_bench(
        int bit_width,
        const std::vector<float> & vectors,
        const std::vector<uint64_t> & ids,
        const std::vector<float> & queries,
        const BenchConfig & cfg) {
    CHECK(cfg.delete_stride > 0);

    ggml_vec_index_t * idx = create_index_mode(bit_width, cfg.dim);
    CHECK(idx != nullptr);
    CHECK(ggml_vec_index_add(idx, vectors.data(), cfg.n_vec, ids.data()) == GGML_VEC_INDEX_OK);

    DeleteBenchResult result;
    result.full_search_ms = run_search(
        idx, queries, cfg.n_query, cfg.k, cfg.warmups, cfg.repeats).ms;

    for (int row = 0; row < cfg.n_vec; row += cfg.delete_stride) {
        CHECK(ggml_vec_index_remove(idx, ids[static_cast<size_t>(row)]) == GGML_VEC_INDEX_OK);
        ++result.deleted;
    }
    result.live = ggml_vec_index_len(idx);
    CHECK(result.live == cfg.n_vec - result.deleted);

    result.tombstone_search_ms = run_search(
        idx, queries, cfg.n_query, cfg.k, cfg.warmups, cfg.repeats).ms;

    const auto t0 = std::chrono::steady_clock::now();
    CHECK(ggml_vec_index_compact(idx) == GGML_VEC_INDEX_OK);
    const auto t1 = std::chrono::steady_clock::now();
    result.compact_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    CHECK(ggml_vec_index_len(idx) == result.live);

    result.compacted_search_ms = run_search(
        idx, queries, cfg.n_query, cfg.k, cfg.warmups, cfg.repeats).ms;

    ggml_vec_index_free(idx);
    return result;
}

DeltaBenchResult run_delta_bench(
        int bit_width,
        const std::vector<float> & vectors,
        const std::vector<uint64_t> & ids,
        const BenchConfig & cfg,
        const std::filesystem::path & temp_dir,
        const char * snapshot_name,
        const char * delta_name) {
    CHECK(cfg.delta_ops > 0 && cfg.delta_ops <= (cfg.n_vec - 1) / 2);
    const int base_n = cfg.n_vec - cfg.delta_ops;
    const std::filesystem::path snapshot_path = temp_dir / snapshot_name;
    const std::filesystem::path delta_path = temp_dir / delta_name;
    const std::filesystem::path single_delta_path = delta_path.string() + ".single";
    std::filesystem::remove(snapshot_path);
    remove_delta_artifacts(delta_path);
    remove_delta_artifacts(single_delta_path);

    ggml_vec_index_t * idx = create_index_mode(bit_width, cfg.dim);
    CHECK(idx != nullptr);
    CHECK(ggml_vec_index_add(idx, vectors.data(), base_n, ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(idx, snapshot_path.string().c_str()) == GGML_VEC_INDEX_OK);

    DeltaBenchResult result;
    {
        ggml_vec_index_t * single_idx = ggml_vec_index_load(snapshot_path.string().c_str());
        CHECK(single_idx != nullptr);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < cfg.delta_ops; ++i) {
            const int row = base_n + i;
            CHECK(ggml_vec_index_add_logged(
                single_idx,
                vectors.data() + static_cast<size_t>(row) * static_cast<size_t>(cfg.dim),
                1,
                ids.data() + row,
                single_delta_path.string().c_str()) == GGML_VEC_INDEX_OK);
        }
        const auto t1 = std::chrono::steady_clock::now();
        result.add_single_us_per_op =
            std::chrono::duration<double, std::micro>(t1 - t0).count() / cfg.delta_ops;
        ggml_vec_index_free(single_idx);
        remove_delta_artifacts(single_delta_path);
    }
    {
        const auto t0 = std::chrono::steady_clock::now();
        CHECK(ggml_vec_index_add_logged(
            idx,
            vectors.data() + static_cast<size_t>(base_n) * static_cast<size_t>(cfg.dim),
            cfg.delta_ops,
            ids.data() + base_n,
            delta_path.string().c_str()) == GGML_VEC_INDEX_OK);
        const auto t1 = std::chrono::steady_clock::now();
        result.add_batch_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        result.add_batch_us_per_vector = result.add_batch_ms * 1000.0 / cfg.delta_ops;
    }
    const int remove_ops = cfg.delta_ops / 2;
    const auto remove_t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < remove_ops; ++i) {
        CHECK(ggml_vec_index_remove_logged(
            idx,
            ids[static_cast<size_t>(i)],
            delta_path.string().c_str()) == GGML_VEC_INDEX_OK);
    }
    const auto remove_t1 = std::chrono::steady_clock::now();
    if (remove_ops > 0) {
        result.remove_us_per_op =
            std::chrono::duration<double, std::micro>(remove_t1 - remove_t0).count() / remove_ops;
    }

    result.snapshot_bytes_before = std::filesystem::file_size(snapshot_path);
    result.delta_bytes_before = std::filesystem::file_size(delta_path);
    const std::vector<uint8_t> dirty_snapshot = read_file_bytes(snapshot_path);
    const std::vector<uint8_t> dirty_delta = read_file_bytes(delta_path);

    result.snapshot_load_ms = median_time_ms(cfg.warmups, cfg.repeats, [&]() {
        ggml_vec_index_t * loaded = ggml_vec_index_load(snapshot_path.string().c_str());
        CHECK(loaded != nullptr);
        ggml_vec_index_free(loaded);
    });
    result.replay_load_ms = median_time_ms(cfg.warmups, cfg.repeats, [&]() {
        ggml_vec_index_t * loaded = ggml_vec_index_load_with_delta(
            snapshot_path.string().c_str(),
            delta_path.string().c_str());
        CHECK(loaded != nullptr);
        ggml_vec_index_free(loaded);
    });
    std::vector<double> compact_times;
    compact_times.reserve(static_cast<size_t>(cfg.repeats));
    for (int i = 0; i < cfg.repeats; ++i) {
        write_file_bytes(snapshot_path, dirty_snapshot);
        write_file_bytes(delta_path, dirty_delta);
        const auto t0 = std::chrono::steady_clock::now();
        CHECK(ggml_vec_index_compact_delta(
            idx,
            snapshot_path.string().c_str(),
            delta_path.string().c_str()) == GGML_VEC_INDEX_OK);
        const auto t1 = std::chrono::steady_clock::now();
        compact_times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(compact_times.begin(), compact_times.end());
    result.compact_ms = compact_times[compact_times.size() / 2];
    result.snapshot_bytes_after = std::filesystem::file_size(snapshot_path);
    result.delta_bytes_after = std::filesystem::file_size(delta_path);
    result.post_compact_load_ms = median_time_ms(cfg.warmups, cfg.repeats, [&]() {
        ggml_vec_index_t * loaded = ggml_vec_index_load_with_delta(
            snapshot_path.string().c_str(),
            delta_path.string().c_str());
        CHECK(loaded != nullptr);
        ggml_vec_index_free(loaded);
    });

    ggml_vec_index_free(idx);
    std::filesystem::remove(snapshot_path);
    remove_delta_artifacts(delta_path);
    return result;
}

} // namespace

static void print_usage(const char * argv0) {
    std::printf(
        "usage: %s [--n-vec N] [--dim N] [--n-query N] [--k N]\n"
        "       [--warmups N] [--repeats N] [--delta-ops N]\n"
        "       [--ivf-lists N] [--ivf-iters N] [--ivf-nprobe N]\n"
        "       [--delete-stride N]\n",
        argv0);
}

static bool parse_int_arg(const char * value, int min_value, int & out) {
    char * end = nullptr;
    errno = 0;
    const long parsed = std::strtol(value, &end, 10);
    if (errno == ERANGE || end == value || *end != '\0' ||
        parsed < min_value || parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    out = static_cast<int>(parsed);
    return true;
}

static ParseResult parse_args(int argc, char ** argv, BenchConfig & cfg) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            return ParseResult::help;
        }
        auto read_int = [&](int min_value, int & out) {
            if (i + 1 >= argc || !parse_int_arg(argv[++i], min_value, out)) {
                std::fprintf(stderr, "invalid value for %s\n", arg.c_str());
                return false;
            }
            return true;
        };
        if (arg == "--n-vec") {
            if (!read_int(1, cfg.n_vec)) { return ParseResult::error; }
        } else if (arg == "--dim") {
            if (!read_int(1, cfg.dim)) { return ParseResult::error; }
        } else if (arg == "--n-query") {
            if (!read_int(1, cfg.n_query)) { return ParseResult::error; }
        } else if (arg == "--k") {
            if (!read_int(1, cfg.k)) { return ParseResult::error; }
        } else if (arg == "--warmups") {
            if (!read_int(0, cfg.warmups)) { return ParseResult::error; }
        } else if (arg == "--repeats") {
            if (!read_int(1, cfg.repeats)) { return ParseResult::error; }
        } else if (arg == "--delta-ops") {
            if (!read_int(1, cfg.delta_ops)) { return ParseResult::error; }
        } else if (arg == "--ivf-lists") {
            if (!read_int(1, cfg.ivf_lists)) { return ParseResult::error; }
        } else if (arg == "--ivf-iters") {
            if (!read_int(0, cfg.ivf_iters)) { return ParseResult::error; }
        } else if (arg == "--ivf-nprobe") {
            if (!read_int(1, cfg.ivf_nprobe)) { return ParseResult::error; }
        } else if (arg == "--delete-stride") {
            if (!read_int(1, cfg.delete_stride)) { return ParseResult::error; }
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return ParseResult::error;
        }
    }
    if (cfg.delta_ops > (cfg.n_vec - 1) / 2) {
        std::fprintf(stderr, "--delta-ops must be less than half of --n-vec\n");
        return ParseResult::error;
    }
    return ParseResult::ok;
}

int main(int argc, char ** argv) {
    BenchConfig cfg;
    const ParseResult parse_result = parse_args(argc, argv, cfg);
    if (parse_result == ParseResult::help) {
        print_usage(argv[0]);
        return 0;
    }
    if (parse_result == ParseResult::error) {
        print_usage(argv[0]);
        return 1;
    }
    const std::filesystem::path temp_dir = make_temp_dir();

    std::vector<float> vectors = make_normalized_vectors(cfg.n_vec, cfg.dim, 0xdeadbeef);
    std::vector<float> queries = make_normalized_vectors(cfg.n_query, cfg.dim, 0xc001d00d);
    std::vector<uint64_t> ids(static_cast<size_t>(cfg.n_vec));
    for (int i = 0; i < cfg.n_vec; ++i) {
        ids[static_cast<size_t>(i)] = static_cast<uint64_t>(i) + 1;
    }

    ggml_vec_index_t * f32 = ggml_vec_index_create(cfg.dim, 32);
    ggml_vec_index_t * q8  = ggml_vec_index_create(cfg.dim, 8);
    ggml_vec_index_t * q4  = ggml_vec_index_create(cfg.dim, 4);
    ggml_vec_index_t * tvq2 = ggml_vec_index_create_turbovec_q2(cfg.dim);
    ggml_vec_index_t * tvq4 = ggml_vec_index_create_turbovec_q4(cfg.dim);
    CHECK(f32 != nullptr);
    CHECK(q8 != nullptr);
    CHECK(q4 != nullptr);
    CHECK(tvq2 != nullptr);
    CHECK(tvq4 != nullptr);
    CHECK(ggml_vec_index_add(f32, vectors.data(), cfg.n_vec, ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_add(q8,  vectors.data(), cfg.n_vec, ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_add(q4,  vectors.data(), cfg.n_vec, ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_add(tvq2, vectors.data(), cfg.n_vec, ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_add(tvq4, vectors.data(), cfg.n_vec, ids.data()) == GGML_VEC_INDEX_OK);

    const double f32_ivf_build_ms = median_time_ms(cfg.warmups, cfg.repeats, [&]() {
        CHECK(ggml_vec_index_build_ivf(f32, cfg.ivf_lists, cfg.ivf_iters) == GGML_VEC_INDEX_OK);
    });
    const double q8_ivf_build_ms = median_time_ms(cfg.warmups, cfg.repeats, [&]() {
        CHECK(ggml_vec_index_build_ivf(q8, cfg.ivf_lists, cfg.ivf_iters) == GGML_VEC_INDEX_OK);
    });
    const double q4_ivf_build_ms = median_time_ms(cfg.warmups, cfg.repeats, [&]() {
        CHECK(ggml_vec_index_build_ivf(q4, cfg.ivf_lists, cfg.ivf_iters) == GGML_VEC_INDEX_OK);
    });
    const double tvq2_ivf_build_ms = median_time_ms(cfg.warmups, cfg.repeats, [&]() {
        CHECK(ggml_vec_index_build_ivf(tvq2, cfg.ivf_lists, cfg.ivf_iters) == GGML_VEC_INDEX_OK);
    });
    const double tvq4_ivf_build_ms = median_time_ms(cfg.warmups, cfg.repeats, [&]() {
        CHECK(ggml_vec_index_build_ivf(tvq4, cfg.ivf_lists, cfg.ivf_iters) == GGML_VEC_INDEX_OK);
    });

    const TimedSearch f32_res = run_search(
        f32, queries, cfg.n_query, cfg.k, cfg.warmups, cfg.repeats);
    const TimedSearch q8_res = run_search(
        q8, queries, cfg.n_query, cfg.k, cfg.warmups, cfg.repeats);
    const TimedSearch q4_res = run_search(
        q4, queries, cfg.n_query, cfg.k, cfg.warmups, cfg.repeats);
    const TimedSearch tvq2_res = run_search(
        tvq2, queries, cfg.n_query, cfg.k, cfg.warmups, cfg.repeats);
    const TimedSearch tvq4_res = run_search(
        tvq4, queries, cfg.n_query, cfg.k, cfg.warmups, cfg.repeats);
    const TimedSearch f32_ivf_res = run_ivf_search(
        f32, queries, cfg.n_query, cfg.k, cfg.ivf_nprobe, cfg.warmups, cfg.repeats);
    const TimedSearch q8_ivf_res = run_ivf_search(
        q8, queries, cfg.n_query, cfg.k, cfg.ivf_nprobe, cfg.warmups, cfg.repeats);
    const TimedSearch q4_ivf_res = run_ivf_search(
        q4, queries, cfg.n_query, cfg.k, cfg.ivf_nprobe, cfg.warmups, cfg.repeats);
    const TimedSearch tvq2_ivf_res = run_ivf_search(
        tvq2, queries, cfg.n_query, cfg.k, cfg.ivf_nprobe, cfg.warmups, cfg.repeats);
    const TimedSearch tvq4_ivf_res = run_ivf_search(
        tvq4, queries, cfg.n_query, cfg.k, cfg.ivf_nprobe, cfg.warmups, cfg.repeats);

    const std::vector<int> filter_sizes = {
        32,
        128,
        512,
        cfg.n_vec / 2,
    };
    std::vector<std::vector<uint64_t>> allowlists;
    std::vector<TimedSearch> f32_filtered;
    std::vector<TimedSearch> q8_filtered;
    std::vector<TimedSearch> q4_filtered;
    std::vector<TimedSearch> tvq2_filtered;
    std::vector<TimedSearch> tvq4_filtered;
    std::vector<TimedSearch> f32_prepared_filtered;
    std::vector<TimedSearch> q8_prepared_filtered;
    std::vector<TimedSearch> q4_prepared_filtered;
    std::vector<TimedSearch> tvq2_prepared_filtered;
    std::vector<TimedSearch> tvq4_prepared_filtered;
    for (int requested : filter_sizes) {
        const int n_allowed = std::min(requested, cfg.n_vec);
        allowlists.push_back(make_allowlist(cfg.n_vec, n_allowed));
        f32_filtered.push_back(run_filtered_search(
            f32,
            queries,
            cfg.n_query,
            cfg.k,
            allowlists.back(),
            cfg.warmups,
            cfg.repeats));
        q8_filtered.push_back(run_filtered_search(
            q8,
            queries,
            cfg.n_query,
            cfg.k,
            allowlists.back(),
            cfg.warmups,
            cfg.repeats));
        q4_filtered.push_back(run_filtered_search(
            q4,
            queries,
            cfg.n_query,
            cfg.k,
            allowlists.back(),
            cfg.warmups,
            cfg.repeats));
        tvq2_filtered.push_back(run_filtered_search(
            tvq2,
            queries,
            cfg.n_query,
            cfg.k,
            allowlists.back(),
            cfg.warmups,
            cfg.repeats));
        tvq4_filtered.push_back(run_filtered_search(
            tvq4,
            queries,
            cfg.n_query,
            cfg.k,
            allowlists.back(),
            cfg.warmups,
            cfg.repeats));
        ggml_vec_index_filter_t * f32_filter = ggml_vec_index_filter_create(
            f32, allowlists.back().data(), static_cast<int>(allowlists.back().size()));
        ggml_vec_index_filter_t * q8_filter = ggml_vec_index_filter_create(
            q8, allowlists.back().data(), static_cast<int>(allowlists.back().size()));
        ggml_vec_index_filter_t * q4_filter = ggml_vec_index_filter_create(
            q4, allowlists.back().data(), static_cast<int>(allowlists.back().size()));
        ggml_vec_index_filter_t * tvq2_filter = ggml_vec_index_filter_create(
            tvq2, allowlists.back().data(), static_cast<int>(allowlists.back().size()));
        ggml_vec_index_filter_t * tvq4_filter = ggml_vec_index_filter_create(
            tvq4, allowlists.back().data(), static_cast<int>(allowlists.back().size()));
        CHECK(f32_filter != nullptr);
        CHECK(q8_filter != nullptr);
        CHECK(q4_filter != nullptr);
        CHECK(tvq2_filter != nullptr);
        CHECK(tvq4_filter != nullptr);
        f32_prepared_filtered.push_back(run_prepared_filtered_search(
            f32,
            f32_filter,
            queries,
            cfg.n_query,
            cfg.k,
            cfg.warmups,
            cfg.repeats));
        q8_prepared_filtered.push_back(run_prepared_filtered_search(
            q8,
            q8_filter,
            queries,
            cfg.n_query,
            cfg.k,
            cfg.warmups,
            cfg.repeats));
        q4_prepared_filtered.push_back(run_prepared_filtered_search(
            q4,
            q4_filter,
            queries,
            cfg.n_query,
            cfg.k,
            cfg.warmups,
            cfg.repeats));
        tvq2_prepared_filtered.push_back(run_prepared_filtered_search(
            tvq2,
            tvq2_filter,
            queries,
            cfg.n_query,
            cfg.k,
            cfg.warmups,
            cfg.repeats));
        tvq4_prepared_filtered.push_back(run_prepared_filtered_search(
            tvq4,
            tvq4_filter,
            queries,
            cfg.n_query,
            cfg.k,
            cfg.warmups,
            cfg.repeats));
        ggml_vec_index_filter_free(f32_filter);
        ggml_vec_index_filter_free(q8_filter);
        ggml_vec_index_filter_free(q4_filter);
        ggml_vec_index_filter_free(tvq2_filter);
        ggml_vec_index_filter_free(tvq4_filter);
    }

    const double f32_ivf_recall_at_k =
        recall_against(f32_res, f32_ivf_res, cfg.n_query, cfg.k);
    const double q8_ivf_recall_at_k =
        recall_against(f32_res, q8_ivf_res, cfg.n_query, cfg.k);
    const double q4_ivf_recall_at_k =
        recall_against(f32_res, q4_ivf_res, cfg.n_query, cfg.k);
    const double tvq2_ivf_recall_at_k =
        recall_against(f32_res, tvq2_ivf_res, cfg.n_query, cfg.k);
    const double tvq4_ivf_recall_at_k =
        recall_against(f32_res, tvq4_ivf_res, cfg.n_query, cfg.k);

    const auto f32_path = write_index_file(f32, temp_dir, "ggml-vector-index-bench-f32.tvim");
    const auto q8_path  = write_index_file(q8,  temp_dir, "ggml-vector-index-bench-q8.tvim");
    const auto q4_path  = write_index_file(q4,  temp_dir, "ggml-vector-index-bench-q4.tvim");
    const auto tvq2_path = write_index_file(tvq2, temp_dir, "ggml-vector-index-bench-tvq2.tvim");
    const auto tvq4_path = write_index_file(tvq4, temp_dir, "ggml-vector-index-bench-tvq4.tvim");
    const uintmax_t f32_file_size = std::filesystem::file_size(f32_path);
    const uintmax_t q8_file_size  = std::filesystem::file_size(q8_path);
    const uintmax_t q4_file_size  = std::filesystem::file_size(q4_path);
    const uintmax_t tvq2_file_size = std::filesystem::file_size(tvq2_path);
    const uintmax_t tvq4_file_size = std::filesystem::file_size(tvq4_path);
    const double f32_mmap_load_ms = median_time_ms(cfg.warmups, cfg.repeats, [&]() {
        ggml_vec_index_t * loaded = ggml_vec_index_load_mmap(f32_path.string().c_str());
        CHECK(loaded != nullptr);
        ggml_vec_index_free(loaded);
    });
    const double q8_mmap_load_ms = median_time_ms(cfg.warmups, cfg.repeats, [&]() {
        ggml_vec_index_t * loaded = ggml_vec_index_load_mmap(q8_path.string().c_str());
        CHECK(loaded != nullptr);
        ggml_vec_index_free(loaded);
    });
    const double q4_mmap_load_ms = median_time_ms(cfg.warmups, cfg.repeats, [&]() {
        ggml_vec_index_t * loaded = ggml_vec_index_load_mmap(q4_path.string().c_str());
        CHECK(loaded != nullptr);
        ggml_vec_index_free(loaded);
    });
    const double tvq2_load_ms = median_time_ms(cfg.warmups, cfg.repeats, [&]() {
        ggml_vec_index_t * loaded = ggml_vec_index_load(tvq2_path.string().c_str());
        CHECK(loaded != nullptr);
        ggml_vec_index_free(loaded);
    });
    const double tvq4_load_ms = median_time_ms(cfg.warmups, cfg.repeats, [&]() {
        ggml_vec_index_t * loaded = ggml_vec_index_load(tvq4_path.string().c_str());
        CHECK(loaded != nullptr);
        ggml_vec_index_free(loaded);
    });
    std::filesystem::remove(f32_path);
    std::filesystem::remove(q8_path);
    std::filesystem::remove(q4_path);
    std::filesystem::remove(tvq2_path);
    std::filesystem::remove(tvq4_path);

    const DeltaBenchResult f32_delta = run_delta_bench(
        32,
        vectors,
        ids,
        cfg,
        temp_dir,
        "ggml-vector-index-bench-delta-f32.tvim",
        "ggml-vector-index-bench-delta-f32.tvid");
    const DeltaBenchResult q8_delta = run_delta_bench(
        8,
        vectors,
        ids,
        cfg,
        temp_dir,
        "ggml-vector-index-bench-delta-q8.tvim",
        "ggml-vector-index-bench-delta-q8.tvid");
    const DeltaBenchResult q4_delta = run_delta_bench(
        4,
        vectors,
        ids,
        cfg,
        temp_dir,
        "ggml-vector-index-bench-delta-q4.tvim",
        "ggml-vector-index-bench-delta-q4.tvid");
    const DeleteBenchResult f32_delete = run_delete_bench(32, vectors, ids, queries, cfg);
    const DeleteBenchResult q8_delete = run_delete_bench(8, vectors, ids, queries, cfg);
    const DeleteBenchResult q4_delete = run_delete_bench(4, vectors, ids, queries, cfg);
    const std::vector<Q4CalibrationMode> q4_calibration_modes = {
        { "p99_abs", 0.99f, 0.0f },
        { "p95_abs", 0.95f, 0.0f },
        { "rms_3", 1.0f, 3.0f },
    };
    std::vector<QualityBenchResult> quality_suite;
    quality_suite.push_back(run_quality_bench(
        "normalized_gaussian", vectors, queries, q4_calibration_modes, cfg));
    quality_suite.push_back(run_quality_bench(
        "raw_gaussian",
        make_gaussian_vectors(cfg.n_vec, cfg.dim, 0x12345678, false),
        make_gaussian_vectors(cfg.n_query, cfg.dim, 0x87654321, false),
        q4_calibration_modes,
        cfg));
    quality_suite.push_back(run_quality_bench(
        "sparse_16",
        make_sparse_vectors(cfg.n_vec, cfg.dim, 0x51a2b3c4, 16),
        make_sparse_vectors(cfg.n_query, cfg.dim, 0x15a2b3c4, 16),
        q4_calibration_modes,
        cfg));
    const std::vector<float> cluster_centers = make_cluster_centers(cfg.dim, 0x0ddc0ffe, 64);
    quality_suite.push_back(run_quality_bench(
        "clustered_64",
        make_clustered_vectors(cfg.n_vec, cfg.dim, 0x0ddc0ffd, cluster_centers, 64),
        make_clustered_vectors(cfg.n_query, cfg.dim, 0x0ddc0fff, cluster_centers, 64),
        q4_calibration_modes,
        cfg));

    const size_t f32_memory_bytes =
        static_cast<size_t>(cfg.n_vec) * static_cast<size_t>(cfg.dim) * sizeof(float) +
        static_cast<size_t>(cfg.n_vec) * sizeof(uint64_t);
    const size_t q8_memory_bytes =
        static_cast<size_t>(cfg.n_vec) * static_cast<size_t>(cfg.dim) * sizeof(int8_t) +
        static_cast<size_t>(cfg.n_vec) * sizeof(float) +
        static_cast<size_t>(cfg.n_vec) * sizeof(uint64_t);
    const size_t q4_memory_bytes =
        static_cast<size_t>(cfg.n_vec) * ((static_cast<size_t>(cfg.dim) + 1) / 2) +
        static_cast<size_t>(cfg.n_vec) * sizeof(float) +
        static_cast<size_t>(cfg.n_vec) * sizeof(uint64_t);
    const size_t tv_scale_bytes =
        static_cast<size_t>(cfg.n_vec) * sizeof(float);
    const size_t tv_calibration_bytes =
        2 * static_cast<size_t>(cfg.dim) * sizeof(float);
    const size_t tv_block_slots =
        ((static_cast<size_t>(cfg.n_vec) + 31) / 32) * 32;
    const size_t tvq2_memory_bytes =
        static_cast<size_t>(cfg.n_vec) * (static_cast<size_t>(cfg.dim) / 4) +
        tv_block_slots * (static_cast<size_t>(cfg.dim) / 4) +
        tv_scale_bytes +
        tv_calibration_bytes +
        static_cast<size_t>(cfg.n_vec) * sizeof(uint64_t);
    const size_t tvq4_memory_bytes =
        static_cast<size_t>(cfg.n_vec) * (static_cast<size_t>(cfg.dim) / 2) +
        tv_block_slots * (static_cast<size_t>(cfg.dim) / 2) +
        tv_scale_bytes +
        tv_calibration_bytes +
        static_cast<size_t>(cfg.n_vec) * sizeof(uint64_t);

    std::printf("llama-vector-index-bench\n");
    std::printf("  q8 kernel=%s\n", q8_kernel_name());
    std::printf("  q4 kernel=%s\n", q4_kernel_name());
    std::printf("  turbovec kernel=%s\n", turbovec_kernel_name());
    std::printf("  n_vec=%d dim=%d n_query=%d k=%d warmups=%d repeats=%d\n",
        cfg.n_vec, cfg.dim, cfg.n_query, cfg.k, cfg.warmups, cfg.repeats);
    std::printf("  estimated memory: f32=%zu bytes q8=%zu bytes q4=%zu bytes tvq2=%zu bytes tvq4=%zu bytes q8/f32=%.3f q4/f32=%.3f tvq2/f32=%.3f tvq4/f32=%.3f\n",
        f32_memory_bytes, q8_memory_bytes, q4_memory_bytes, tvq2_memory_bytes, tvq4_memory_bytes,
        static_cast<double>(q8_memory_bytes) / static_cast<double>(f32_memory_bytes),
        static_cast<double>(q4_memory_bytes) / static_cast<double>(f32_memory_bytes),
        static_cast<double>(tvq2_memory_bytes) / static_cast<double>(f32_memory_bytes),
        static_cast<double>(tvq4_memory_bytes) / static_cast<double>(f32_memory_bytes));
    std::printf("  file size:        f32=%llu bytes q8=%llu bytes q4=%llu bytes tvq2=%llu bytes tvq4=%llu bytes q8/f32=%.3f q4/f32=%.3f tvq2/f32=%.3f tvq4/f32=%.3f\n",
        static_cast<unsigned long long>(f32_file_size),
        static_cast<unsigned long long>(q8_file_size),
        static_cast<unsigned long long>(q4_file_size),
        static_cast<unsigned long long>(tvq2_file_size),
        static_cast<unsigned long long>(tvq4_file_size),
        static_cast<double>(q8_file_size) / static_cast<double>(f32_file_size),
        static_cast<double>(q4_file_size) / static_cast<double>(f32_file_size),
        static_cast<double>(tvq2_file_size) / static_cast<double>(f32_file_size),
        static_cast<double>(tvq4_file_size) / static_cast<double>(f32_file_size));
    std::printf("  mmap load:        f32=%.3f ms q8=%.3f ms q4=%.3f ms\n",
        f32_mmap_load_ms, q8_mmap_load_ms, q4_mmap_load_ms);
    std::printf("  snapshot load:    tvq2=%.3f ms tvq4=%.3f ms (mmap unsupported)\n",
        tvq2_load_ms, tvq4_load_ms);
    std::printf("  median latency:   f32=%.3f ms q8=%.3f ms q4=%.3f ms tvq2=%.3f ms tvq4=%.3f ms q8/f32=%.3f q4/f32=%.3f tvq2/f32=%.3f tvq4/f32=%.3f\n",
        f32_res.ms, q8_res.ms, q4_res.ms, tvq2_res.ms, tvq4_res.ms,
        q8_res.ms / f32_res.ms, q4_res.ms / f32_res.ms,
        tvq2_res.ms / f32_res.ms, tvq4_res.ms / f32_res.ms);
    std::printf(
        "  ivf build:        lists=%d iters=%d f32=%.3f ms q8=%.3f ms q4=%.3f ms tvq2=%.3f ms tvq4=%.3f ms\n",
        cfg.ivf_lists,
        cfg.ivf_iters,
        f32_ivf_build_ms,
        q8_ivf_build_ms,
        q4_ivf_build_ms,
        tvq2_ivf_build_ms,
        tvq4_ivf_build_ms);
    std::printf(
        "  ivf latency:      nprobe=%d f32=%.3f ms q8=%.3f ms q4=%.3f ms tvq2=%.3f ms tvq4=%.3f ms f32/full=%.3f q8/full=%.3f q4/full=%.3f tvq2/full=%.3f tvq4/full=%.3f\n",
        cfg.ivf_nprobe,
        f32_ivf_res.ms,
        q8_ivf_res.ms,
        q4_ivf_res.ms,
        tvq2_ivf_res.ms,
        tvq4_ivf_res.ms,
        f32_ivf_res.ms / f32_res.ms,
        q8_ivf_res.ms / q8_res.ms,
        q4_ivf_res.ms / q4_res.ms,
        tvq2_ivf_res.ms / tvq2_res.ms,
        tvq4_ivf_res.ms / tvq4_res.ms);
    std::printf(
        "  ivf recall:       f32@%d=%.4f q8@%d=%.4f q4@%d=%.4f tvq2@%d=%.4f tvq4@%d=%.4f against exact f32\n",
        cfg.k,
        f32_ivf_recall_at_k,
        cfg.k,
        q8_ivf_recall_at_k,
        cfg.k,
        q4_ivf_recall_at_k,
        cfg.k,
        tvq2_ivf_recall_at_k,
        cfg.k,
        tvq4_ivf_recall_at_k);
    for (size_t i = 0; i < allowlists.size(); ++i) {
        std::printf(
            "  filtered latency: allowed=%zu f32=%.3f ms q8=%.3f ms q4=%.3f ms tvq2=%.3f ms tvq4=%.3f ms f32/prepared=%.3f ms q8/prepared=%.3f ms q4/prepared=%.3f ms tvq2/prepared=%.3f ms tvq4/prepared=%.3f ms\n",
            allowlists[i].size(),
            f32_filtered[i].ms,
            q8_filtered[i].ms,
            q4_filtered[i].ms,
            tvq2_filtered[i].ms,
            tvq4_filtered[i].ms,
            f32_prepared_filtered[i].ms,
            q8_prepared_filtered[i].ms,
            q4_prepared_filtered[i].ms,
            tvq2_prepared_filtered[i].ms,
            tvq4_prepared_filtered[i].ms);
        std::printf(
            "  filtered ratio:   allowed=%zu f32/full=%.3f q8/full=%.3f q4/full=%.3f tvq2/full=%.3f tvq4/full=%.3f f32/prep_speedup=%.3f q8/prep_speedup=%.3f q4/prep_speedup=%.3f tvq2/prep_speedup=%.3f tvq4/prep_speedup=%.3f\n",
            allowlists[i].size(),
            f32_filtered[i].ms / f32_res.ms,
            q8_filtered[i].ms / q8_res.ms,
            q4_filtered[i].ms / q4_res.ms,
            tvq2_filtered[i].ms / tvq2_res.ms,
            tvq4_filtered[i].ms / tvq4_res.ms,
            f32_filtered[i].ms / f32_prepared_filtered[i].ms,
            q8_filtered[i].ms / q8_prepared_filtered[i].ms,
            q4_filtered[i].ms / q4_prepared_filtered[i].ms,
            tvq2_filtered[i].ms / tvq2_prepared_filtered[i].ms,
            tvq4_filtered[i].ms / tvq4_prepared_filtered[i].ms);
    }
    std::printf(
        "  delta append f32:  batch=%.3f ms batch=%.3f us/vector single_add=%.3f us/op remove=%.3f us/op\n",
        f32_delta.add_batch_ms,
        f32_delta.add_batch_us_per_vector,
        f32_delta.add_single_us_per_op,
        f32_delta.remove_us_per_op);
    std::printf(
        "  delta load f32:    snapshot=%.3f ms replay=%.3f ms compact=%.3f ms post_compact=%.3f ms\n",
        f32_delta.snapshot_load_ms,
        f32_delta.replay_load_ms,
        f32_delta.compact_ms,
        f32_delta.post_compact_load_ms);
    std::printf(
        "  delta bytes f32:   snapshot_before=%llu delta_before=%llu snapshot_after=%llu delta_after=%llu\n",
        static_cast<unsigned long long>(f32_delta.snapshot_bytes_before),
        static_cast<unsigned long long>(f32_delta.delta_bytes_before),
        static_cast<unsigned long long>(f32_delta.snapshot_bytes_after),
        static_cast<unsigned long long>(f32_delta.delta_bytes_after));
    std::printf(
        "  delta append q8:   batch=%.3f ms batch=%.3f us/vector single_add=%.3f us/op remove=%.3f us/op\n",
        q8_delta.add_batch_ms,
        q8_delta.add_batch_us_per_vector,
        q8_delta.add_single_us_per_op,
        q8_delta.remove_us_per_op);
    std::printf(
        "  delta load q8:     snapshot=%.3f ms replay=%.3f ms compact=%.3f ms post_compact=%.3f ms\n",
        q8_delta.snapshot_load_ms,
        q8_delta.replay_load_ms,
        q8_delta.compact_ms,
        q8_delta.post_compact_load_ms);
    std::printf(
        "  delta bytes q8:    snapshot_before=%llu delta_before=%llu snapshot_after=%llu delta_after=%llu\n",
        static_cast<unsigned long long>(q8_delta.snapshot_bytes_before),
        static_cast<unsigned long long>(q8_delta.delta_bytes_before),
        static_cast<unsigned long long>(q8_delta.snapshot_bytes_after),
        static_cast<unsigned long long>(q8_delta.delta_bytes_after));
    std::printf(
        "  delta append q4:   batch=%.3f ms batch=%.3f us/vector single_add=%.3f us/op remove=%.3f us/op\n",
        q4_delta.add_batch_ms,
        q4_delta.add_batch_us_per_vector,
        q4_delta.add_single_us_per_op,
        q4_delta.remove_us_per_op);
    std::printf(
        "  delta load q4:     snapshot=%.3f ms replay=%.3f ms compact=%.3f ms post_compact=%.3f ms\n",
        q4_delta.snapshot_load_ms,
        q4_delta.replay_load_ms,
        q4_delta.compact_ms,
        q4_delta.post_compact_load_ms);
    std::printf(
        "  delta bytes q4:    snapshot_before=%llu delta_before=%llu snapshot_after=%llu delta_after=%llu\n",
        static_cast<unsigned long long>(q4_delta.snapshot_bytes_before),
        static_cast<unsigned long long>(q4_delta.delta_bytes_before),
        static_cast<unsigned long long>(q4_delta.snapshot_bytes_after),
        static_cast<unsigned long long>(q4_delta.delta_bytes_after));
    std::printf(
        "  delete-heavy:      deleted=%d live=%d stride=%d\n",
        f32_delete.deleted,
        f32_delete.live,
        cfg.delete_stride);
    std::printf(
        "  delete f32:        full_before=%.3f ms tombstoned=%.3f ms compact=%.3f ms compacted=%.3f ms tomb/full_before=%.3f compacted/tomb=%.3f\n",
        f32_delete.full_search_ms,
        f32_delete.tombstone_search_ms,
        f32_delete.compact_ms,
        f32_delete.compacted_search_ms,
        f32_delete.tombstone_search_ms / f32_delete.full_search_ms,
        f32_delete.compacted_search_ms / f32_delete.tombstone_search_ms);
    std::printf(
        "  delete q8:         full_before=%.3f ms tombstoned=%.3f ms compact=%.3f ms compacted=%.3f ms tomb/full_before=%.3f compacted/tomb=%.3f\n",
        q8_delete.full_search_ms,
        q8_delete.tombstone_search_ms,
        q8_delete.compact_ms,
        q8_delete.compacted_search_ms,
        q8_delete.tombstone_search_ms / q8_delete.full_search_ms,
        q8_delete.compacted_search_ms / q8_delete.tombstone_search_ms);
    std::printf(
        "  delete q4:         full_before=%.3f ms tombstoned=%.3f ms compact=%.3f ms compacted=%.3f ms tomb/full_before=%.3f compacted/tomb=%.3f\n",
        q4_delete.full_search_ms,
        q4_delete.tombstone_search_ms,
        q4_delete.compact_ms,
        q4_delete.compacted_search_ms,
        q4_delete.tombstone_search_ms / q4_delete.full_search_ms,
        q4_delete.compacted_search_ms / q4_delete.tombstone_search_ms);
    for (const QualityBenchResult & quality : quality_suite) {
        std::printf(
            "  quality %-19s q8_recall=%.4f q8_mean_drift=%.6f q8_max_drift=%.6f q4_recall=%.4f q4_mean_drift=%.6f q4_max_drift=%.6f tvq2_recall=%.4f tvq2_mean_drift=%.6f tvq2_max_drift=%.6f tvq4_recall=%.4f tvq4_mean_drift=%.6f tvq4_max_drift=%.6f\n",
            quality.name,
            quality.q8.recall_at_k,
            quality.q8.mean_abs_score_drift,
            quality.q8.max_abs_score_drift,
            quality.q4.recall_at_k,
            quality.q4.mean_abs_score_drift,
            quality.q4.max_abs_score_drift,
            quality.tvq2.recall_at_k,
            quality.tvq2.mean_abs_score_drift,
            quality.tvq2.max_abs_score_drift,
            quality.tvq4.recall_at_k,
            quality.tvq4.mean_abs_score_drift,
            quality.tvq4.max_abs_score_drift);
        for (size_t i = 0; i < q4_calibration_modes.size(); ++i) {
            const QualityMetrics & metrics = quality.q4_calibrated[i];
            std::printf(
                "  q4cal   %-19s mode=%-7s recall=%.4f mean_drift=%.6f max_drift=%.6f\n",
                quality.name,
                q4_calibration_modes[i].name,
                metrics.recall_at_k,
                metrics.mean_abs_score_drift,
                metrics.max_abs_score_drift);
        }
    }
    ggml_vec_index_free(f32);
    ggml_vec_index_free(q8);
    ggml_vec_index_free(q4);
    ggml_vec_index_free(tvq2);
    ggml_vec_index_free(tvq4);
    std::error_code cleanup_ec;
    std::filesystem::remove_all(temp_dir, cleanup_ec);
    return 0;
}
