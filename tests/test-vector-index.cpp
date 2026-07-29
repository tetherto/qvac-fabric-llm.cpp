// test-vector-index.cpp - standalone C-API smoke test for the POC vector
// index. Exercises lifecycle, add, search, remove, contains, write, load,
// search-after-load, quantized storage, filtered/prepared-filter search, and
// IVF-flat search. No model, no llama; only the new ggml-vector-index public
// C API.

#include "ggml-vector-index.h"
#ifdef GGML_VEC_INDEX_TEST_HOOKS
#include "ggml-vector-index-impl.h"
#endif

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#ifdef _WIN32
#    include <process.h>
#else
#    include <sys/stat.h>
#    include <unistd.h>
#endif

#ifdef GGML_VEC_INDEX_TEST_HOOKS
extern "C" void ggml_vec_index_test_set_write_fail_after(int64_t bytes);
#endif

namespace {

constexpr int kDim = 4;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

std::vector<float> normalize(std::vector<float> v) {
    double sumsq = 0.0;
    for (float x : v) {
        sumsq += static_cast<double>(x) * x;
    }
    const float n = static_cast<float>(std::sqrt(sumsq));
    if (n > 0.0f) {
        for (float & x : v) {
            x /= n;
        }
    }
    return v;
}

int test_pid() {
#ifdef _WIN32
    return _getpid();
#else
    return getpid();
#endif
}

std::filesystem::path temp_path(const char * suffix) {
    static int counter = 0;
    return std::filesystem::temp_directory_path() /
           ("ggml-vector-index-test-" + std::to_string(test_pid()) + "-" + std::to_string(counter++) + suffix);
}

struct temp_file {
    explicit temp_file(const char * suffix) : path(temp_path(suffix)) { std::filesystem::remove(path); }

    ~temp_file() { std::filesystem::remove(path); }

    std::filesystem::path path;
};

void append_u32_le(std::vector<uint8_t> & bytes, uint32_t value) {
    for (int i = 0; i < 4; ++i) {
        bytes.push_back(static_cast<uint8_t>(value >> (8 * i)));
    }
}

void append_u64_le(std::vector<uint8_t> & bytes, uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        bytes.push_back(static_cast<uint8_t>(value >> (8 * i)));
    }
}

void append_f32_le(std::vector<uint8_t> & bytes, float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    append_u32_le(bytes, bits);
}

void write_bytes(const std::filesystem::path & path, const std::vector<uint8_t> & bytes) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    CHECK(f.is_open());
    if (!bytes.empty()) {
        f.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    CHECK(static_cast<bool>(f));
}

std::vector<uint8_t> read_bytes(const std::filesystem::path & path) {
    std::ifstream f(path, std::ios::binary);
    CHECK(f.is_open());
    return {
        std::istreambuf_iterator<char>(f),
        std::istreambuf_iterator<char>(),
    };
}

bool has_snapshot_tmp(const std::filesystem::path & path) {
    const std::filesystem::path dir    = path.parent_path().empty() ? std::filesystem::path(".") : path.parent_path();
    const std::string           prefix = ".ggml-vi-" + std::to_string(std::filesystem::hash_value(path)) + "-";

    std::error_code                           ec;
    std::filesystem::directory_iterator       it(dir, ec);
    const std::filesystem::directory_iterator end;
    while (!ec && it != end) {
        if (it->path().filename().string().rfind(prefix, 0) == 0) {
            return true;
        }
        it.increment(ec);
    }
    return false;
}

std::vector<uint8_t> snapshot_bytes(uint32_t                      dim,
                                    uint32_t                      n,
                                    const std::vector<float> &    values,
                                    const std::vector<uint64_t> & ids) {
    std::vector<uint8_t> bytes;
    bytes.insert(bytes.end(), { 'T', 'V', 'P', 'I' });
    bytes.push_back(1);
    bytes.push_back(32);
    bytes.push_back(0);
    bytes.push_back(0);
    append_u32_le(bytes, dim);
    append_u32_le(bytes, n);
    for (float value : values) {
        append_f32_le(bytes, value);
    }
    for (uint64_t id : ids) {
        append_u64_le(bytes, id);
    }
    return bytes;
}

struct RefScore {
    float    score = 0.0f;
    uint64_t id    = 0;
};

bool ref_score_better(const RefScore & a, const RefScore & b) {
    if (a.score != b.score) {
        return a.score > b.score;
    }
    return a.id < b.id;
}

int round_nearest_even_ref(float value) {
    const float lower_f    = std::floor(value);
    const float upper_f    = lower_f + 1.0f;
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

float quantized_reference_score(const float * vector, const float * query, int dim, int bit_width) {
    float max_abs = 0.0f;
    for (int i = 0; i < dim; ++i) {
        max_abs = std::max(max_abs, std::fabs(vector[i]));
    }

    const int   max_code = bit_width == 8 ? 127 : 7;
    const float scale    = max_abs == 0.0f ? 1.0f : max_abs / static_cast<float>(max_code);
    double      acc      = 0.0;
    for (int i = 0; i < dim; ++i) {
        int q = max_abs == 0.0f ? 0 : round_nearest_even_ref(vector[i] / scale);
        q     = std::max(-max_code, std::min(max_code, q));
        acc += static_cast<double>(query[i]) * static_cast<double>(q) * static_cast<double>(scale);
    }
    return static_cast<float>(acc);
}

std::vector<RefScore> reference_topk(const std::vector<float> &    vectors,
                                     const std::vector<uint64_t> & ids,
                                     const float *                 query,
                                     int                           dim,
                                     int                           bit_width,
                                     int                           k) {
    std::vector<RefScore> scores;
    scores.reserve(ids.size());
    for (size_t row = 0; row < ids.size(); ++row) {
        scores.push_back(
            { quantized_reference_score(vectors.data() + row * static_cast<size_t>(dim), query, dim, bit_width),
              ids[row] });
    }
    std::sort(scores.begin(), scores.end(), ref_score_better);
    scores.resize(static_cast<size_t>(k), RefScore{ -FLT_MAX, UINT64_MAX });
    return scores;
}

void check_quantized_reference(int bit_width) {
    constexpr int dim = 19;
    constexpr int n   = 6;
    constexpr int n_q = 3;
    constexpr int k   = 4;

    std::vector<float> vectors;
    vectors.reserve(static_cast<size_t>(n) * dim);
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < dim; ++col) {
            const float sign = ((row + col) & 1) == 0 ? 1.0f : -1.0f;
            vectors.push_back(sign * (0.03f * static_cast<float>((row + 1) * (col + 1)) +
                                      0.01f * static_cast<float>((row * 3 + col) % 7)));
        }
    }

    const std::vector<uint64_t> ids = { 7105ULL, 7102ULL, 7109ULL, 7101ULL, 7107ULL, 7103ULL };
    std::vector<float>          queries;
    queries.reserve(static_cast<size_t>(n_q) * dim);
    for (int row = 0; row < n_q; ++row) {
        for (int col = 0; col < dim; ++col) {
            const float sign = ((row * 2 + col) & 1) == 0 ? 1.0f : -1.0f;
            queries.push_back(sign * (0.02f * static_cast<float>((row + 2) * (col + 1)) -
                                      0.015f * static_cast<float>((row + col) % 5)));
        }
    }

    auto * idx = ggml_vec_index_create(dim, bit_width);
    CHECK(idx != nullptr);
    CHECK(ggml_vec_index_add(idx, vectors.data(), n, ids.data()) == GGML_VEC_INDEX_OK);

    std::vector<float>    out_scores(static_cast<size_t>(n_q) * k);
    std::vector<uint64_t> out_ids(static_cast<size_t>(n_q) * k);
    CHECK(ggml_vec_index_search(idx, queries.data(), n_q, k, out_scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);

    const float tolerance = bit_width == 8 ? 1e-4f : 2e-4f;
    for (int q = 0; q < n_q; ++q) {
        const auto expected =
            reference_topk(vectors, ids, queries.data() + static_cast<size_t>(q) * dim, dim, bit_width, k);
        for (int j = 0; j < k; ++j) {
            const size_t offset = static_cast<size_t>(q) * k + static_cast<size_t>(j);
            CHECK(out_ids[offset] == expected[static_cast<size_t>(j)].id);
            const float limit = tolerance * std::max(1.0f, std::fabs(expected[static_cast<size_t>(j)].score));
            CHECK(std::fabs(out_scores[offset] - expected[static_cast<size_t>(j)].score) <= limit);
        }
    }

    ggml_vec_index_free(idx);
}

void check_quantized_small_dim_tie_order(int bit_width) {
    constexpr int dim = 6;
    constexpr int k   = 2;

    const std::array<float, dim * 2> vectors = {
        1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f,
    };
    const std::array<uint64_t, 2> ids = {
        910001ULL,
        910002ULL,
    };
    const std::array<float, dim> query = {
        1.0e8f, 1.0f, -1.0e8f, 1.0f, 0.0f, 0.0f,
    };

    auto * idx = ggml_vec_index_create(dim, bit_width);
    CHECK(idx != nullptr);
    CHECK(ggml_vec_index_add(idx, vectors.data(), static_cast<int>(ids.size()), ids.data()) == GGML_VEC_INDEX_OK);

    std::array<float, k>    scores{};
    std::array<uint64_t, k> out_ids{};
    CHECK(ggml_vec_index_search(idx, query.data(), 1, k, scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(out_ids[0] == ids[0]);
    CHECK(out_ids[1] == ids[1]);
    CHECK(scores[0] == scores[1]);

    ggml_vec_index_free(idx);
}

void check_quantized_simd_near_tie_order(int bit_width) {
    constexpr int dim = 16;
    constexpr int k   = 2;

    std::array<float, dim * 2> vectors{};
    const int max_code = bit_width == 8 ? 127 : 7;
    vectors[0]       = 1.0f;
    vectors[dim]     = 1.0f;
    vectors[dim + 1] = 1.0f / static_cast<float>(max_code);

    const std::array<uint64_t, 2> ids = {
        915001ULL,
        915002ULL,
    };
    std::array<float, dim> query{};
    query[0] = 1.0f;
    query[1] = 2.0e-7f;

    auto * idx = ggml_vec_index_create(dim, bit_width);
    CHECK(idx != nullptr);
    CHECK(ggml_vec_index_add(idx, vectors.data(), static_cast<int>(ids.size()), ids.data()) == GGML_VEC_INDEX_OK);

    std::array<float, k>    scores{};
    std::array<uint64_t, k> out_ids{};
    CHECK(ggml_vec_index_search(idx, query.data(), 1, k, scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(out_ids[0] == ids[1]);
    CHECK(out_ids[1] == ids[0]);

    ggml_vec_index_free(idx);
}

void check_quantized_overflow_topk_order(int bit_width) {
    constexpr int dim = 1;
    constexpr int k   = 2;

    const std::array<float, dim * 2> vectors = {
        1.5f,
        2.0f,
    };
    const std::array<uint64_t, 2> ids = {
        920001ULL,
        920002ULL,
    };
    const std::array<float, dim> query = {
        FLT_MAX,
    };

    auto * idx = ggml_vec_index_create(dim, bit_width);
    CHECK(idx != nullptr);
    CHECK(ggml_vec_index_add(idx, vectors.data(), static_cast<int>(ids.size()), ids.data()) == GGML_VEC_INDEX_OK);

    std::array<float, k>    scores{};
    std::array<uint64_t, k> out_ids{};
    CHECK(ggml_vec_index_search(idx, query.data(), 1, k, scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(out_ids[0] == ids[1]);
    CHECK(out_ids[1] == ids[0]);
    CHECK(scores[0] == FLT_MAX);
    CHECK(scores[1] == FLT_MAX);

    ggml_vec_index_free(idx);
}

void check_ivf_full_probe_recall(int bit_width) {
    constexpr int dim     = 6;
    constexpr int n       = 12;
    constexpr int n_q     = 4;
    constexpr int k       = 3;
    constexpr int n_lists = 4;

    std::vector<float> vectors;
    vectors.reserve(static_cast<size_t>(n) * dim);
    std::vector<uint64_t> ids;
    ids.reserve(n);
    for (int row = 0; row < n; ++row) {
        const int cluster = row % n_lists;
        ids.push_back(8200ULL + static_cast<uint64_t>(row));
        for (int col = 0; col < dim; ++col) {
            const float base = col == cluster ? 1.0f : 0.05f * static_cast<float>((row + col) % 3);
            vectors.push_back(base + 0.01f * static_cast<float>(row + 1));
        }
    }

    std::vector<float> queries;
    queries.reserve(static_cast<size_t>(n_q) * dim);
    for (int q = 0; q < n_q; ++q) {
        for (int col = 0; col < dim; ++col) {
            queries.push_back(col == q ? 1.0f : 0.02f * static_cast<float>((q + col) % 4));
        }
    }

    auto * idx = ggml_vec_index_create(dim, bit_width);
    CHECK(idx != nullptr);
    CHECK(ggml_vec_index_add(idx, vectors.data(), n, ids.data()) == GGML_VEC_INDEX_OK);

    std::vector<float>    exact_scores(static_cast<size_t>(n_q) * k);
    std::vector<uint64_t> exact_ids(static_cast<size_t>(n_q) * k);
    std::vector<float>    ivf_scores(static_cast<size_t>(n_q) * k);
    std::vector<uint64_t> ivf_ids(static_cast<size_t>(n_q) * k);

    CHECK(ggml_vec_index_search(idx, queries.data(), n_q, k, exact_scores.data(), exact_ids.data()) ==
          GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_build_ivf(idx, n_lists, 2) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_search_ivf(idx, queries.data(), n_q, k, n_lists, ivf_scores.data(), ivf_ids.data()) ==
          GGML_VEC_INDEX_OK);

    for (size_t i = 0; i < exact_ids.size(); ++i) {
        CHECK(ivf_ids[i] == exact_ids[i]);
        CHECK(std::fabs(ivf_scores[i] - exact_scores[i]) <= 1e-5f * std::max(1.0f, std::fabs(exact_scores[i])));
    }

    ggml_vec_index_free(idx);
}

void check_f32_ivf_extreme_centroid_routing() {
    constexpr int dim = 1;

    const std::array<float, dim * 2> vectors = {
        1.5f,
        2.0f,
    };
    const std::array<uint64_t, 2> ids   = { 8500ULL, 8501ULL };
    const std::array<float, dim>  query = { FLT_MAX };

    auto * idx = ggml_vec_index_create(dim, 32);
    CHECK(idx != nullptr);
    CHECK(ggml_vec_index_add(idx, vectors.data(), static_cast<int>(ids.size()), ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_build_ivf(idx, /*n_lists=*/2, /*n_iter=*/0) == GGML_VEC_INDEX_OK);

    std::array<float, 1>    exact_score{};
    std::array<uint64_t, 1> exact_id{};
    std::array<float, 1>    ivf_score{};
    std::array<uint64_t, 1> ivf_id{};
    CHECK(ggml_vec_index_search(idx, query.data(), 1, 1, exact_score.data(), exact_id.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_search_ivf(idx, query.data(), 1, 1, 1, ivf_score.data(), ivf_id.data()) == GGML_VEC_INDEX_OK);
    CHECK(exact_id[0] == ids[1]);
    CHECK(ivf_id[0] == exact_id[0]);
    CHECK(exact_score[0] == FLT_MAX);
    CHECK(ivf_score[0] == FLT_MAX);

    ggml_vec_index_free(idx);
}

void check_ivf_partial_probe_routing() {
    constexpr int dim = 2;

    const std::array<float, dim * 4> vectors = {
        1.0f,   0.0f,
        2.0f,   0.0f,
        0.0f,   1.0f,
        100.0f, 101.0f,
    };
    const std::array<uint64_t, 4> ids   = { 8250ULL, 8251ULL, 8252ULL, 8253ULL };
    const std::array<float, dim>  query = { 1.0f, 0.0f };

    auto * idx = ggml_vec_index_create(dim, /*bit_width=*/32);
    CHECK(idx != nullptr);
    CHECK(ggml_vec_index_add(idx, vectors.data(), static_cast<int>(ids.size()), ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_build_ivf(idx, /*n_lists=*/2, /*n_iter=*/0) == GGML_VEC_INDEX_OK);

    std::array<float, 1>    partial_score{};
    std::array<uint64_t, 1> partial_id{};
    CHECK(ggml_vec_index_search_ivf(idx, query.data(), 1, 1, /*nprobe=*/1,
                                    partial_score.data(), partial_id.data()) == GGML_VEC_INDEX_OK);
    CHECK(partial_id[0] == ids[1]);
    CHECK(partial_score[0] == 2.0f);

    std::array<float, 1>    full_score{};
    std::array<uint64_t, 1> full_id{};
    CHECK(ggml_vec_index_search_ivf(idx, query.data(), 1, 1, /*nprobe=*/2,
                                    full_score.data(), full_id.data()) == GGML_VEC_INDEX_OK);
    CHECK(full_id[0] == ids[3]);
    CHECK(full_score[0] == 100.0f);

    ggml_vec_index_free(idx);
}

void check_ivf_centroid_overflow_fallback() {
    constexpr int dim = 4;

    const std::array<float, dim * 2> vectors = {
        1.0f, -1.0f, 0.0f, 0.0f, 1.0e30f, 1.0e30f, 0.0f, 0.0f,
    };
    const std::array<uint64_t, 2> ids   = { 8300ULL, 8301ULL };
    const std::array<float, dim>  query = { -1.0e10f, 1.0e10f, 0.0f, 0.0f };

    auto * idx = ggml_vec_index_create(dim, 32);
    CHECK(idx != nullptr);
    CHECK(ggml_vec_index_add(idx, vectors.data(), static_cast<int>(ids.size()), ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_build_ivf(idx, /*n_lists=*/2, /*n_iter=*/0) == GGML_VEC_INDEX_OK);

    std::array<float, 1>    exact_score{};
    std::array<uint64_t, 1> exact_id{};
    std::array<float, 1>    ivf_score{};
    std::array<uint64_t, 1> ivf_id{};
    CHECK(ggml_vec_index_search(idx, query.data(), 1, 1, exact_score.data(), exact_id.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_search_ivf(idx, query.data(), 1, 1, 1, ivf_score.data(), ivf_id.data()) == GGML_VEC_INDEX_OK);
    CHECK(exact_id[0] == ids[1]);
    CHECK(ivf_id[0] == exact_id[0]);
    CHECK(ivf_score[0] == exact_score[0]);

    ggml_vec_index_free(idx);
}

void check_q8_ivf_extreme_centroid_routing() {
    constexpr int dim = 2;

    const std::array<float, dim * 2> vectors = {
        FLT_MAX, 0.0f,
        0.0f,    1.0f,
    };
    const std::array<uint64_t, 2> ids   = { 8400ULL, 8401ULL };
    const std::array<float, dim>  query = { 0.0f, -1.0f };

    auto * idx = ggml_vec_index_create(dim, /*bit_width=*/8);
    CHECK(idx != nullptr);
    CHECK(ggml_vec_index_add(idx, vectors.data(), static_cast<int>(ids.size()), ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_build_ivf(idx, /*n_lists=*/2, /*n_iter=*/1) == GGML_VEC_INDEX_OK);

    std::array<float, 1>    exact_score{};
    std::array<uint64_t, 1> exact_id{};
    std::array<float, 1>    ivf_score{};
    std::array<uint64_t, 1> ivf_id{};
    CHECK(ggml_vec_index_search(idx, query.data(), 1, 1, exact_score.data(), exact_id.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_search_ivf(idx, query.data(), 1, 1, 1, ivf_score.data(), ivf_id.data()) == GGML_VEC_INDEX_OK);
    CHECK(exact_id[0] == ids[0]);
    CHECK(ivf_id[0] == exact_id[0]);
    CHECK(ivf_score[0] == exact_score[0]);

    ggml_vec_index_free(idx);
}

void check_ivf_empty_batch_state_validation() {
    constexpr int dim = 2;

    auto * idx = ggml_vec_index_create(dim, /*bit_width=*/32);
    CHECK(idx != nullptr);
    CHECK(ggml_vec_index_search_ivf(idx, nullptr, 0, 1, 1, nullptr, nullptr) ==
          GGML_VEC_INDEX_E_INVALID_ARG);

    const std::array<float, dim> first_vector = { 1.0f, 0.0f };
    const uint64_t first_id = 930001ULL;
    CHECK(ggml_vec_index_add(idx, first_vector.data(), 1, &first_id) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_search_ivf(idx, nullptr, 0, 1, 1, nullptr, nullptr) ==
          GGML_VEC_INDEX_E_INVALID_ARG);

    CHECK(ggml_vec_index_build_ivf(idx, /*n_lists=*/1, /*n_iter=*/0) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_search_ivf(idx, nullptr, 0, 1, 1, nullptr, nullptr) == GGML_VEC_INDEX_OK);

    const std::array<float, dim> second_vector = { 0.0f, 1.0f };
    const uint64_t second_id = 930002ULL;
    CHECK(ggml_vec_index_add(idx, second_vector.data(), 1, &second_id) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_search_ivf(idx, nullptr, 0, 1, 1, nullptr, nullptr) ==
          GGML_VEC_INDEX_E_INVALID_ARG);

    CHECK(ggml_vec_index_build_ivf(idx, /*n_lists=*/2, /*n_iter=*/0) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_remove(idx, second_id) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_search_ivf(idx, nullptr, 0, 1, 1, nullptr, nullptr) ==
          GGML_VEC_INDEX_E_INVALID_ARG);

    CHECK(ggml_vec_index_build_ivf(idx, /*n_lists=*/1, /*n_iter=*/0) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_compact(idx) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_search_ivf(idx, nullptr, 0, 1, 1, nullptr, nullptr) ==
          GGML_VEC_INDEX_E_INVALID_ARG);

    ggml_vec_index_free(idx);
}

void check_filtered_and_ivf_search(int bit_width) {
    constexpr int dim = 4;

    const std::array<float, dim * 4> vectors = {
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    };
    const std::array<uint64_t, 4> ids = {
        9001ULL,
        9002ULL,
        9003ULL,
        9004ULL,
    };
    const float tolerance = bit_width == 32 ? 1e-6f : (bit_width == 8 ? 1e-4f : 2e-4f);

    auto * idx = ggml_vec_index_create(dim, bit_width);
    CHECK(idx != nullptr);
    CHECK(ggml_vec_index_add(idx, vectors.data(), static_cast<int>(ids.size()), ids.data()) == GGML_VEC_INDEX_OK);

    const float *           query = vectors.data() + dim;
    std::array<float, 4>    exact_scores{};
    std::array<uint64_t, 4> exact_ids{};
    CHECK(ggml_vec_index_search(idx, query, 1, 4, exact_scores.data(), exact_ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(exact_ids[0] == ids[1]);

    const std::array<uint64_t, 3> allowed = {
        ids[2],
        ids[1],
        ids[1],
    };
    std::array<float, 2>    scores{};
    std::array<uint64_t, 2> out_ids{};
    CHECK(ggml_vec_index_search_filtered(idx, query, 1, 2, allowed.data(), static_cast<int>(allowed.size()),
                                         scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(out_ids[0] == ids[1]);
    CHECK(out_ids[1] == ids[2]);
    CHECK(scores[0] > scores[1]);
    CHECK(scores[0] > 0.95f);
    CHECK(std::fabs(scores[1]) < tolerance);
    CHECK(std::fabs(scores[0] - exact_scores[0]) < tolerance);
    const auto filtered_scores = scores;

    auto * filter = ggml_vec_index_filter_create(idx, allowed.data(), static_cast<int>(allowed.size()));
    CHECK(filter != nullptr);
    scores.fill(123.0f);
    out_ids.fill(123ULL);
    CHECK(ggml_vec_index_search_prepared_filtered(idx, filter, query, 1, 2, scores.data(), out_ids.data()) ==
          GGML_VEC_INDEX_OK);
    CHECK(out_ids[0] == ids[1]);
    CHECK(out_ids[1] == ids[2]);
    CHECK(scores == filtered_scores);

    std::array<float, 2>    repeated_scores{};
    std::array<uint64_t, 2> repeated_ids{};
    CHECK(ggml_vec_index_search_prepared_filtered(idx, filter, query, 1, 2, repeated_scores.data(),
                                                  repeated_ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(repeated_ids == out_ids);
    CHECK(repeated_scores == scores);
    ggml_vec_index_filter_free(filter);

    std::array<float, 3>    padded_scores{};
    std::array<uint64_t, 3> padded_ids{};
    CHECK(ggml_vec_index_search_filtered(idx, query, 1, 3, allowed.data(), static_cast<int>(allowed.size()),
                                         padded_scores.data(), padded_ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(padded_ids[0] == ids[1]);
    CHECK(padded_ids[1] == ids[2]);
    CHECK(padded_scores[0] >= padded_scores[1]);
    CHECK(padded_ids[2] == UINT64_MAX);
    CHECK(padded_scores[2] == -FLT_MAX);

    std::array<float, 2>    empty_scores{};
    std::array<uint64_t, 2> empty_ids{};
    CHECK(ggml_vec_index_search_filtered(idx, query, 1, 2, nullptr, 0, empty_scores.data(), empty_ids.data()) ==
          GGML_VEC_INDEX_OK);
    CHECK(empty_ids[0] == UINT64_MAX);
    CHECK(empty_ids[1] == UINT64_MAX);
    CHECK(empty_scores[0] == -FLT_MAX);
    CHECK(empty_scores[1] == -FLT_MAX);

    empty_scores.fill(123.0f);
    empty_ids.fill(123ULL);
    CHECK(ggml_vec_index_search_filtered(idx, query, 1, 2, allowed.data(), 0, empty_scores.data(), empty_ids.data()) ==
          GGML_VEC_INDEX_OK);
    CHECK(empty_ids[0] == UINT64_MAX);
    CHECK(empty_ids[1] == UINT64_MAX);
    CHECK(empty_scores[0] == -FLT_MAX);
    CHECK(empty_scores[1] == -FLT_MAX);

    const std::array<uint64_t, 2> absent = { 123456789ULL, 987654321ULL };
    empty_scores.fill(123.0f);
    empty_ids.fill(123ULL);
    CHECK(ggml_vec_index_search_filtered(idx, query, 1, 2, absent.data(), static_cast<int>(absent.size()),
                                         empty_scores.data(), empty_ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(empty_ids[0] == UINT64_MAX);
    CHECK(empty_ids[1] == UINT64_MAX);
    CHECK(empty_scores[0] == -FLT_MAX);
    CHECK(empty_scores[1] == -FLT_MAX);

    auto * other_idx = ggml_vec_index_create(dim, bit_width);
    CHECK(other_idx != nullptr);
    CHECK(ggml_vec_index_add(other_idx, vectors.data(), static_cast<int>(ids.size()), ids.data()) == GGML_VEC_INDEX_OK);
    auto * foreign = ggml_vec_index_filter_create(other_idx, allowed.data(), static_cast<int>(allowed.size()));
    CHECK(foreign != nullptr);
    CHECK(ggml_vec_index_search_prepared_filtered(idx, foreign, query, 1, 2, scores.data(), out_ids.data()) ==
          GGML_VEC_INDEX_E_INVALID_ARG);
    ggml_vec_index_filter_free(foreign);
    ggml_vec_index_free(other_idx);
    CHECK(ggml_vec_index_search_prepared_filtered(idx, nullptr, query, 1, 2, scores.data(), out_ids.data()) ==
          GGML_VEC_INDEX_E_INVALID_ARG);
    ggml_vec_index_filter_free(nullptr);

    CHECK(ggml_vec_index_search_ivf(idx, nullptr, 0, 1, 1, nullptr, nullptr) ==
          GGML_VEC_INDEX_E_INVALID_ARG);
    CHECK(ggml_vec_index_search_ivf(idx, query, 1, 1, 1, scores.data(), out_ids.data()) ==
          GGML_VEC_INDEX_E_INVALID_ARG);
    CHECK(ggml_vec_index_build_ivf(idx, /*n_lists=*/4, /*n_iter=*/4) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_search_ivf(idx, nullptr, 0, 1, 1, nullptr, nullptr) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_search_ivf(idx, query, 1, 1, 0, scores.data(), out_ids.data()) ==
          GGML_VEC_INDEX_E_INVALID_ARG);
    CHECK(ggml_vec_index_search_ivf(idx, query, 1, 1, -1, scores.data(), out_ids.data()) ==
          GGML_VEC_INDEX_E_INVALID_ARG);

    for (size_t row = 0; row < ids.size(); ++row) {
        std::array<float, 1>    exact_score{};
        std::array<uint64_t, 1> exact_id{};
        std::array<float, 1>    ivf_score{};
        std::array<uint64_t, 1> ivf_id{};
        const float *           row_query = vectors.data() + row * dim;
        CHECK(ggml_vec_index_search(idx, row_query, 1, 1, exact_score.data(), exact_id.data()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search_ivf(idx, row_query, 1, 1, 1, ivf_score.data(), ivf_id.data()) == GGML_VEC_INDEX_OK);
        CHECK(ivf_id == exact_id);
        CHECK(std::fabs(ivf_score[0] - exact_score[0]) < tolerance);
    }

    std::array<float, 4>    ivf_all_scores{};
    std::array<uint64_t, 4> ivf_all_ids{};
    CHECK(ggml_vec_index_search_ivf(idx, query, 1, 4, 99, ivf_all_scores.data(), ivf_all_ids.data()) ==
          GGML_VEC_INDEX_OK);
    CHECK(ivf_all_ids == exact_ids);
    for (size_t i = 0; i < exact_scores.size(); ++i) {
        CHECK(std::fabs(ivf_all_scores[i] - exact_scores[i]) < tolerance);
    }

    auto * stale_after_add = ggml_vec_index_filter_create(idx, allowed.data(), static_cast<int>(allowed.size()));
    CHECK(stale_after_add != nullptr);
    const std::array<float, dim> added_vector = { 0.5f, 0.5f, 0.0f, 0.0f };
    const uint64_t               added_id     = 9010ULL;
    CHECK(ggml_vec_index_add(idx, added_vector.data(), 1, &added_id) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_search_prepared_filtered(idx, stale_after_add, query, 1, 2, scores.data(), out_ids.data()) ==
          GGML_VEC_INDEX_E_INVALID_ARG);
    CHECK(ggml_vec_index_search_ivf(idx, nullptr, 0, 1, 1, nullptr, nullptr) ==
          GGML_VEC_INDEX_E_INVALID_ARG);
    CHECK(ggml_vec_index_search_ivf(idx, query, 1, 1, 1, scores.data(), out_ids.data()) ==
          GGML_VEC_INDEX_E_INVALID_ARG);
    ggml_vec_index_filter_free(stale_after_add);

    CHECK(ggml_vec_index_build_ivf(idx, /*n_lists=*/4, /*n_iter=*/1) == GGML_VEC_INDEX_OK);
    auto * stale_after_remove = ggml_vec_index_filter_create(idx, allowed.data(), static_cast<int>(allowed.size()));
    CHECK(stale_after_remove != nullptr);
    CHECK(ggml_vec_index_remove(idx, added_id) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_search_prepared_filtered(idx, stale_after_remove, query, 1, 2, scores.data(),
                                                  out_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);
    CHECK(ggml_vec_index_search_ivf(idx, nullptr, 0, 1, 1, nullptr, nullptr) ==
          GGML_VEC_INDEX_E_INVALID_ARG);
    CHECK(ggml_vec_index_search_ivf(idx, query, 1, 1, 1, scores.data(), out_ids.data()) ==
          GGML_VEC_INDEX_E_INVALID_ARG);
    ggml_vec_index_filter_free(stale_after_remove);

    const std::array<uint64_t, 2> with_removed = { added_id, ids[2] };
    empty_scores.fill(123.0f);
    empty_ids.fill(123ULL);
    CHECK(ggml_vec_index_search_filtered(idx, query, 1, 2, with_removed.data(), static_cast<int>(with_removed.size()),
                                         empty_scores.data(), empty_ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(empty_ids[0] == ids[2]);
    CHECK(empty_ids[1] == UINT64_MAX);
    CHECK(std::fabs(empty_scores[0]) < tolerance);
    CHECK(empty_scores[1] == -FLT_MAX);

    CHECK(ggml_vec_index_build_ivf(idx, /*n_lists=*/4, /*n_iter=*/1) == GGML_VEC_INDEX_OK);
    auto * stale_after_compact = ggml_vec_index_filter_create(idx, allowed.data(), static_cast<int>(allowed.size()));
    CHECK(stale_after_compact != nullptr);
    CHECK(ggml_vec_index_compact(idx) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_search_prepared_filtered(idx, stale_after_compact, query, 1, 2, scores.data(),
                                                  out_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);
    CHECK(ggml_vec_index_search_ivf(idx, nullptr, 0, 1, 1, nullptr, nullptr) ==
          GGML_VEC_INDEX_E_INVALID_ARG);
    CHECK(ggml_vec_index_search_ivf(idx, query, 1, 1, 1, scores.data(), out_ids.data()) ==
          GGML_VEC_INDEX_E_INVALID_ARG);
    ggml_vec_index_filter_free(stale_after_compact);

    ggml_vec_index_free(idx);
}

void check_ivf_state_not_persisted() {
    const std::array<float, kDim * 2> vectors = {
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
    };
    const std::array<uint64_t, 2> ids = { 9201ULL, 9202ULL };

    temp_file snapshot(".tvim");
    auto *    built = ggml_vec_index_create(kDim, /*bit_width=*/32);
    CHECK(built != nullptr);
    CHECK(ggml_vec_index_add(built, vectors.data(), static_cast<int>(ids.size()), ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_build_ivf(built, /*n_lists=*/2, /*n_iter=*/1) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(built, snapshot.path.string().c_str()) == GGML_VEC_INDEX_OK);
    ggml_vec_index_free(built);

    auto * loaded = ggml_vec_index_load(snapshot.path.string().c_str());
    CHECK(loaded != nullptr);
    std::array<float, 1>    scores{};
    std::array<uint64_t, 1> out_ids{};
    CHECK(ggml_vec_index_search_ivf(loaded, vectors.data(), 1, 1, 2, scores.data(), out_ids.data()) ==
          GGML_VEC_INDEX_E_INVALID_ARG);
    CHECK(ggml_vec_index_build_ivf(loaded, /*n_lists=*/2, /*n_iter=*/1) == GGML_VEC_INDEX_OK);
    std::array<float, 1>    exact_scores{};
    std::array<uint64_t, 1> exact_ids{};
    CHECK(ggml_vec_index_search(loaded, vectors.data(), 1, 1, exact_scores.data(), exact_ids.data()) ==
          GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_search_ivf(loaded, vectors.data(), 1, 1, 2, scores.data(), out_ids.data()) ==
          GGML_VEC_INDEX_OK);
    CHECK(exact_ids[0] == ids[0]);
    CHECK(out_ids == exact_ids);
    CHECK(scores == exact_scores);
    ggml_vec_index_free(loaded);
}

}  // namespace

int main() {
    auto * idx = ggml_vec_index_create(kDim, /*bit_width=*/32);
    CHECK(idx != nullptr);
    CHECK(ggml_vec_index_dim(idx) == kDim);
    CHECK(ggml_vec_index_len(idx) == 0);
    CHECK(ggml_vec_index_bit_width(idx) == 32);
    CHECK(ggml_vec_index_create(0, /*bit_width=*/32) == nullptr);
    CHECK(ggml_vec_index_create(kDim, /*bit_width=*/31) == nullptr);
    CHECK(std::strcmp(ggml_vec_index_error_to_string(GGML_VEC_INDEX_E_NOT_DURABLE), "not durable") == 0);

    // Public APIs reject invalid arguments without mutating the index.
    {
        const std::array<float, kDim> vector = {
            1.0f,
            0.0f,
            0.0f,
            0.0f,
        };
        const uint64_t          id = 123ULL;
        std::array<float, 1>    scores{};
        std::array<uint64_t, 1> out_ids{};

        CHECK(ggml_vec_index_add(nullptr, vector.data(), 1, &id) == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_add(idx, nullptr, 1, &id) == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_add(idx, vector.data(), 1, nullptr) == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_add(idx, vector.data(), -1, &id) == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_add(idx, nullptr, 0, nullptr) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search(nullptr, vector.data(), 1, 1, scores.data(), out_ids.data()) ==
              GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_search(idx, nullptr, 1, 1, scores.data(), out_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_search(idx, vector.data(), -1, 1, scores.data(), out_ids.data()) ==
              GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_search(idx, vector.data(), 1, 0, scores.data(), out_ids.data()) ==
              GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_search(idx, vector.data(), 1, 1, nullptr, out_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_search(idx, vector.data(), 1, 1, scores.data(), nullptr) == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_search(idx, nullptr, 0, 1, nullptr, nullptr) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search_filtered(nullptr, vector.data(), 1, 1, &id, 1, scores.data(), out_ids.data()) ==
              GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_search_filtered(idx, vector.data(), 1, 1, nullptr, 1, scores.data(), out_ids.data()) ==
              GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_search_filtered(idx, vector.data(), 1, 1, &id, -1, scores.data(), out_ids.data()) ==
              GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_filter_create(nullptr, &id, 1) == nullptr);
        CHECK(ggml_vec_index_filter_create(idx, nullptr, 1) == nullptr);
        CHECK(ggml_vec_index_filter_create(idx, &id, -1) == nullptr);
        CHECK(ggml_vec_index_search_prepared_filtered(idx, nullptr, vector.data(), 1, 1, scores.data(),
                                                      out_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_build_ivf(nullptr, 1, 1) == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_build_ivf(idx, 0, 1) == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_build_ivf(idx, 1, -1) == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_search_ivf(nullptr, vector.data(), 1, 1, 1, scores.data(), out_ids.data()) ==
              GGML_VEC_INDEX_E_INVALID_ARG);
        {
            auto * search_idx = ggml_vec_index_create(kDim, /*bit_width=*/32);
            CHECK(search_idx != nullptr);
            auto * filter = ggml_vec_index_filter_create(search_idx, &id, 1);
            CHECK(filter != nullptr);

            CHECK(ggml_vec_index_search_filtered(
                      search_idx, vector.data(), -1, 1, &id, 1, scores.data(), out_ids.data()) ==
                  GGML_VEC_INDEX_E_INVALID_ARG);
            CHECK(ggml_vec_index_search_filtered(
                      search_idx, vector.data(), 1, 0, &id, 1, scores.data(), out_ids.data()) ==
                  GGML_VEC_INDEX_E_INVALID_ARG);
            CHECK(ggml_vec_index_search_filtered(
                      search_idx, nullptr, 1, 1, &id, 1, scores.data(), out_ids.data()) ==
                  GGML_VEC_INDEX_E_INVALID_ARG);
            CHECK(ggml_vec_index_search_filtered(
                      search_idx, vector.data(), 1, 1, &id, 1, nullptr, out_ids.data()) ==
                  GGML_VEC_INDEX_E_INVALID_ARG);
            CHECK(ggml_vec_index_search_filtered(
                      search_idx, vector.data(), 1, 1, &id, 1, scores.data(), nullptr) ==
                  GGML_VEC_INDEX_E_INVALID_ARG);
            CHECK(ggml_vec_index_search_prepared_filtered(
                      search_idx, filter, vector.data(), -1, 1, scores.data(), out_ids.data()) ==
                  GGML_VEC_INDEX_E_INVALID_ARG);
            CHECK(ggml_vec_index_search_prepared_filtered(
                      search_idx, filter, vector.data(), 1, 0, scores.data(), out_ids.data()) ==
                  GGML_VEC_INDEX_E_INVALID_ARG);
            CHECK(ggml_vec_index_search_prepared_filtered(
                      search_idx, filter, nullptr, 1, 1, scores.data(), out_ids.data()) ==
                  GGML_VEC_INDEX_E_INVALID_ARG);
            CHECK(ggml_vec_index_search_prepared_filtered(
                      search_idx, filter, vector.data(), 1, 1, nullptr, out_ids.data()) ==
                  GGML_VEC_INDEX_E_INVALID_ARG);
            CHECK(ggml_vec_index_search_prepared_filtered(
                      search_idx, filter, vector.data(), 1, 1, scores.data(), nullptr) ==
                  GGML_VEC_INDEX_E_INVALID_ARG);

            CHECK(ggml_vec_index_build_ivf(search_idx, 1, 0) == GGML_VEC_INDEX_OK);
            CHECK(ggml_vec_index_search_ivf(
                      search_idx, vector.data(), -1, 1, 1, scores.data(), out_ids.data()) ==
                  GGML_VEC_INDEX_E_INVALID_ARG);
            CHECK(ggml_vec_index_search_ivf(
                      search_idx, vector.data(), 1, 0, 1, scores.data(), out_ids.data()) ==
                  GGML_VEC_INDEX_E_INVALID_ARG);
            CHECK(ggml_vec_index_search_ivf(
                      search_idx, vector.data(), 1, 1, 0, scores.data(), out_ids.data()) ==
                  GGML_VEC_INDEX_E_INVALID_ARG);
            CHECK(ggml_vec_index_search_ivf(
                      search_idx, nullptr, 1, 1, 1, scores.data(), out_ids.data()) ==
                  GGML_VEC_INDEX_E_INVALID_ARG);
            CHECK(ggml_vec_index_search_ivf(
                      search_idx, vector.data(), 1, 1, 1, nullptr, out_ids.data()) ==
                  GGML_VEC_INDEX_E_INVALID_ARG);
            CHECK(ggml_vec_index_search_ivf(
                      search_idx, vector.data(), 1, 1, 1, scores.data(), nullptr) ==
                  GGML_VEC_INDEX_E_INVALID_ARG);

            ggml_vec_index_filter_free(filter);
            ggml_vec_index_free(search_idx);
        }
        CHECK(ggml_vec_index_write(nullptr, "unused.tvim") == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_write(idx, nullptr) == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_load(nullptr) == nullptr);
        CHECK(ggml_vec_index_load(temp_path(".missing").string().c_str()) == nullptr);
        CHECK(ggml_vec_index_load_mmap(nullptr) == nullptr);
        CHECK(ggml_vec_index_load_with_delta(nullptr, nullptr) == nullptr);
        {
            ggml_vec_index_t * out = idx;
            CHECK(ggml_vec_index_load_mmap_ex("unused.tvim", &out) == GGML_VEC_INDEX_E_BAD_VERSION);
            CHECK(out == nullptr);
            out = idx;
            CHECK(ggml_vec_index_load_with_delta_ex("unused.tvim", "unused.tvid", &out) ==
                  GGML_VEC_INDEX_E_INVALID_ARG);
            CHECK(out == nullptr);
        }
        CHECK(ggml_vec_index_remove(nullptr, id) == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_add_logged(idx, vector.data(), 1, &id, "unused.tvid") ==
              GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_len(idx) == 0);
        CHECK(ggml_vec_index_contains(idx, id) == 0);
        CHECK(ggml_vec_index_remove_logged(idx, id, "unused.tvid") == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_compact_delta(idx, "unused.tvim", "unused.tvid") == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_contains(nullptr, id) == 0);
        CHECK(ggml_vec_index_len(nullptr) == 0);
        CHECK(ggml_vec_index_dim(nullptr) == 0);
        CHECK(ggml_vec_index_bit_width(nullptr) == 0);
        ggml_vec_index_prepare(nullptr);
        CHECK(ggml_vec_index_len(idx) == 0);
    }

#ifdef GGML_VEC_INDEX_TEST_HOOKS
    // Byte-span arithmetic is checked independently of the host word size so
    // 64-bit CI still covers the overflow boundary used by 32-bit builds.
    {
        constexpr size_t max_size = std::numeric_limits<size_t>::max();
        CHECK(ggml_vec_index_detail::can_address_array(max_size / sizeof(float), sizeof(float)));
        CHECK(!ggml_vec_index_detail::can_address_array(max_size / sizeof(float) + 1, sizeof(float)));
        CHECK(ggml_vec_index_detail::can_address_array(max_size / sizeof(uint64_t), sizeof(uint64_t)));
        CHECK(!ggml_vec_index_detail::can_address_array(max_size / sizeof(uint64_t) + 1, sizeof(uint64_t)));
    }
#endif

    // Reject array shapes whose byte spans cannot be represented on 32-bit targets.
    if (sizeof(size_t) == sizeof(uint32_t)) {
        constexpr int oversized_dim = 1 << 30;
        auto *        oversized_idx = ggml_vec_index_create(oversized_dim, /*bit_width=*/32);
        CHECK(oversized_idx != nullptr);

        float    value = 0.0f;
        uint64_t id    = 1;
        CHECK(ggml_vec_index_add(oversized_idx, &value, 1, &id) == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_search(oversized_idx, &value, 1, 1, &value, &id) == GGML_VEC_INDEX_E_INVALID_ARG);
        ggml_vec_index_free(oversized_idx);

        auto * output_idx = ggml_vec_index_create(1, /*bit_width=*/32);
        CHECK(output_idx != nullptr);
        CHECK(ggml_vec_index_search(output_idx, &value, 1, std::numeric_limits<int>::max(), &value, &id) ==
              GGML_VEC_INDEX_E_INVALID_ARG);
        ggml_vec_index_free(output_idx);
    }

#ifdef GGML_VEC_INDEX_TEST_HOOKS
    // The documented oversized-index example exceeds the v1 persistence limit.
    {
        constexpr size_t n   = 262144;
        constexpr size_t dim = 4096;
        CHECK(snapshot_write_v1_preflight(n, dim) == GGML_VEC_INDEX_E_INVALID_ARG);
    }
#endif

    // Zero-query prepared-filter calls still validate filter ownership and staleness.
    {
        const std::array<float, kDim> vector = {
            1.0f,
            0.0f,
            0.0f,
            0.0f,
        };
        const uint64_t id = 123ULL;

        auto * filter_idx = ggml_vec_index_create(kDim, /*bit_width=*/32);
        CHECK(filter_idx != nullptr);
        auto * valid_filter = ggml_vec_index_filter_create(filter_idx, nullptr, 0);
        CHECK(valid_filter != nullptr);
        CHECK(ggml_vec_index_search_prepared_filtered(filter_idx, valid_filter, nullptr, 0, 1, nullptr, nullptr) ==
              GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_add(filter_idx, vector.data(), 1, &id) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search_prepared_filtered(filter_idx, valid_filter, nullptr, 0, 1, nullptr, nullptr) ==
              GGML_VEC_INDEX_E_INVALID_ARG);
        ggml_vec_index_filter_free(valid_filter);

        auto * other_idx = ggml_vec_index_create(kDim, /*bit_width=*/32);
        CHECK(other_idx != nullptr);
        auto * other_filter = ggml_vec_index_filter_create(other_idx, nullptr, 0);
        CHECK(other_filter != nullptr);
        CHECK(ggml_vec_index_search_prepared_filtered(filter_idx, other_filter, nullptr, 0, 1, nullptr, nullptr) ==
              GGML_VEC_INDEX_E_INVALID_ARG);
        ggml_vec_index_filter_free(other_filter);
        ggml_vec_index_free(other_idx);
        ggml_vec_index_free(filter_idx);
        CHECK(ggml_vec_index_len(idx) == 0);
    }

    // Empty indexes return only sentinel-padded search results.
    {
        const std::array<float, kDim> query = {
            1.0f,
            0.0f,
            0.0f,
            0.0f,
        };
        std::array<float, 3>    scores{};
        std::array<uint64_t, 3> out_ids{};
        CHECK(ggml_vec_index_search(idx, query.data(), 1, static_cast<int>(scores.size()), scores.data(),
                                    out_ids.data()) == GGML_VEC_INDEX_OK);
        for (size_t i = 0; i < scores.size(); ++i) {
            CHECK(scores[i] == -FLT_MAX);
            CHECK(out_ids[i] == UINT64_MAX);
        }
    }

    // Empty indexes round-trip as valid snapshots.
    {
        temp_file empty_file(".tvim");
        CHECK(ggml_vec_index_write(idx, empty_file.path.string().c_str()) == 0);
        auto * empty = ggml_vec_index_load(empty_file.path.string().c_str());
        CHECK(empty != nullptr);
        CHECK(ggml_vec_index_dim(empty) == kDim);
        CHECK(ggml_vec_index_len(empty) == 0);

        std::array<float, kDim> query = {
            1.0f,
            0.0f,
            0.0f,
            0.0f,
        };
        std::array<float, 2>    scores{};
        std::array<uint64_t, 2> out_ids{};
        CHECK(ggml_vec_index_search(empty, query.data(), 1, static_cast<int>(scores.size()), scores.data(),
                                    out_ids.data()) == GGML_VEC_INDEX_OK);
        for (size_t i = 0; i < scores.size(); ++i) {
            CHECK(scores[i] == -FLT_MAX);
            CHECK(out_ids[i] == UINT64_MAX);
        }
        ggml_vec_index_free(empty);
    }

    // Add 4 well-separated unit vectors. IDs are non-trivial uint64 to
    // catch sign-extension or BigInt round-trip bugs at the JS boundary
    // when this codepath is later exercised from Bare.
    std::vector<float>    vecs;
    std::vector<uint64_t> ids = {
        42ULL,
        (1ULL << 40) + 7ULL,
        (1ULL << 62) + 11ULL,
        UINT64_MAX - 13ULL,
    };
    std::vector<std::vector<float>> seeds = {
        normalize({ 1.0f, 0.0f, 0.0f, 0.0f }),
        normalize({ 0.0f, 1.0f, 0.0f, 0.0f }),
        normalize({ 0.0f, 0.0f, 1.0f, 0.0f }),
        normalize({ 0.0f, 0.0f, 0.0f, 1.0f }),
    };
    for (const auto & s : seeds) {
        vecs.insert(vecs.end(), s.begin(), s.end());
    }
    CHECK(ggml_vec_index_add(idx, vecs.data(), static_cast<int>(ids.size()), ids.data()) == 0);
    CHECK(ggml_vec_index_len(idx) == 4);
    CHECK(ggml_vec_index_contains(idx, ids[0]) == 1);
    CHECK(ggml_vec_index_contains(idx, 999ULL) == 0);
    ggml_vec_index_prepare(idx);
    CHECK(ggml_vec_index_len(idx) == 4);

    // UINT64_MAX is reserved for padded search results.
    {
        const std::vector<uint64_t> reserved_ids = { UINT64_MAX };
        std::vector<float>          reserved_vec(seeds[0]);
        CHECK(ggml_vec_index_add(idx, reserved_vec.data(), 1, reserved_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_remove(idx, UINT64_MAX) == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_contains(idx, UINT64_MAX) == 0);
        CHECK(ggml_vec_index_len(idx) == 4);
    }

    // Non-finite vectors are rejected before mutating state.
    {
        const std::vector<uint64_t> bad_ids = { 777ULL };
        std::vector<float>          bad_vec(seeds[0]);
        const std::array<float, 3>  non_finite = {
            std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
        };
        for (float value : non_finite) {
            bad_vec[2] = value;
            CHECK(ggml_vec_index_add(idx, bad_vec.data(), 1, bad_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);
        }
        CHECK(ggml_vec_index_contains(idx, bad_ids[0]) == 0);
        CHECK(ggml_vec_index_len(idx) == 4);
    }

    // Duplicate add must fail without mutating state.
    {
        const std::vector<uint64_t> dup_ids = { ids[0] };
        std::vector<float>          dup_vec(seeds[0]);
        CHECK(ggml_vec_index_add(idx, dup_vec.data(), 1, dup_ids.data()) == GGML_VEC_INDEX_E_DUPLICATE);
        CHECK(ggml_vec_index_len(idx) == 4);
    }

    // In-batch duplicate ids must also fail atomically.
    {
        const uint64_t              new_id  = (1ULL << 50) + 123ULL;
        const std::vector<uint64_t> dup_ids = { new_id, new_id };
        std::vector<float>          dup_vecs;
        dup_vecs.insert(dup_vecs.end(), seeds[0].begin(), seeds[0].end());
        dup_vecs.insert(dup_vecs.end(), seeds[1].begin(), seeds[1].end());
        CHECK(ggml_vec_index_add(idx, dup_vecs.data(), 2, dup_ids.data()) == GGML_VEC_INDEX_E_DUPLICATE);
        CHECK(ggml_vec_index_len(idx) == 4);
        CHECK(ggml_vec_index_contains(idx, new_id) == 0);
    }

    // Top-1 of querying with each unit vector should retrieve itself with
    // score very close to 1.0 (full f32, no quantization noise).
    {
        std::array<float, 4>    scores{};
        std::array<uint64_t, 4> out_ids{};
        for (size_t i = 0; i < seeds.size(); ++i) {
            CHECK(ggml_vec_index_search(idx, seeds[i].data(), 1, /*k=*/1, scores.data(), out_ids.data()) == 0);
            CHECK(out_ids[0] == ids[i]);
            CHECK(std::fabs(scores[0] - 1.0f) < 1e-5f);
        }
    }

    // Batched queries write one result row per query.
    {
        std::array<float, 2 * kDim> queries{};
        std::copy(seeds[0].begin(), seeds[0].end(), queries.begin());
        std::copy(seeds[2].begin(), seeds[2].end(), queries.begin() + kDim);
        std::array<float, 2>    scores{};
        std::array<uint64_t, 2> out_ids{};
        CHECK(ggml_vec_index_search(idx, queries.data(), 2, /*k=*/1, scores.data(), out_ids.data()) == 0);
        CHECK(out_ids[0] == ids[0]);
        CHECK(out_ids[1] == ids[2]);
        CHECK(std::fabs(scores[0] - 1.0f) < 1e-5f);
        CHECK(std::fabs(scores[1] - 1.0f) < 1e-5f);
    }

    // Top-k > len returns sentinel-padded tail.
    {
        std::array<float, 8>    scores{};
        std::array<uint64_t, 8> out_ids{};
        CHECK(ggml_vec_index_search(idx, seeds[0].data(), 1, /*k=*/8, scores.data(), out_ids.data()) == 0);
        CHECK(out_ids[0] == ids[0]);
        CHECK(std::fabs(scores[0] - 1.0f) < 1e-5f);
        for (int i = 1; i < 4; ++i) {
            CHECK(std::find(ids.begin(), ids.end(), out_ids[i]) != ids.end());
            CHECK(std::fabs(scores[i]) < 1e-5f);
            for (int j = 0; j < i; ++j) {
                CHECK(out_ids[i] != out_ids[j]);
            }
        }
        // Tail entries (positions 4..7) use sentinel score/id values.
        for (int i = 4; i < 8; ++i) {
            CHECK(scores[i] == -FLT_MAX);
            CHECK(out_ids[i] == UINT64_MAX);
        }
    }

    // Results are sorted by descending score for k > 1.
    {
        const std::array<float, kDim> query = {
            0.50f,
            0.75f,
            0.25f,
            0.00f,
        };
        std::array<float, 4>    scores{};
        std::array<uint64_t, 4> out_ids{};
        CHECK(ggml_vec_index_search(idx, query.data(), 1, static_cast<int>(scores.size()), scores.data(),
                                    out_ids.data()) == 0);
        CHECK(out_ids[0] == ids[1]);
        CHECK(out_ids[1] == ids[0]);
        CHECK(out_ids[2] == ids[2]);
        CHECK(out_ids[3] == ids[3]);
        CHECK(std::fabs(scores[0] - 0.75f) < 1e-5f);
        CHECK(std::fabs(scores[1] - 0.50f) < 1e-5f);
        CHECK(std::fabs(scores[2] - 0.25f) < 1e-5f);
        CHECK(std::fabs(scores[3] - 0.00f) < 1e-5f);
        for (size_t i = 1; i < scores.size(); ++i) {
            CHECK(scores[i - 1] >= scores[i]);
        }
    }

    // Top-k eviction keeps the best rows even when they are scanned last.
    {
        const std::array<float, kDim> query = {
            0.10f,
            0.20f,
            0.70f,
            0.90f,
        };
        std::array<float, 2>    scores{};
        std::array<uint64_t, 2> out_ids{};
        CHECK(ggml_vec_index_search(idx, query.data(), 1, /*k=*/2, scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(out_ids[0] == ids[3]);
        CHECK(out_ids[1] == ids[2]);
        CHECK(std::fabs(scores[0] - 0.90f) < 1e-5f);
        CHECK(std::fabs(scores[1] - 0.70f) < 1e-5f);
    }

    // Non-finite queries are rejected.
    {
        std::array<float, 1>    scores{};
        std::array<uint64_t, 1> out_ids{};
        std::array<float, kDim> bad_query = {
            1.0f,
            0.0f,
            0.0f,
            0.0f,
        };
        const std::array<float, 3> non_finite = {
            std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
        };
        for (float value : non_finite) {
            bad_query[1] = value;
            CHECK(ggml_vec_index_search(idx, bad_query.data(), 1, /*k=*/1, scores.data(), out_ids.data()) ==
                  GGML_VEC_INDEX_E_INVALID_ARG);
        }
    }

    // Finite inputs that overflow f32 score range are clamped and still keep
    // the real result distinct from sentinel-padded ids.
    {
        auto * overflow_idx = ggml_vec_index_create(kDim, /*bit_width=*/32);
        CHECK(overflow_idx != nullptr);
        const std::array<float, kDim> overflow_vec = {
            -FLT_MAX,
            0.0f,
            0.0f,
            0.0f,
        };
        const std::array<uint64_t, 1> overflow_ids = { 1234567ULL };
        CHECK(ggml_vec_index_add(overflow_idx, overflow_vec.data(), 1, overflow_ids.data()) == 0);
        const std::array<float, kDim> overflow_query = {
            FLT_MAX,
            0.0f,
            0.0f,
            0.0f,
        };
        std::array<float, 2>    scores{};
        std::array<uint64_t, 2> out_ids{};
        CHECK(ggml_vec_index_search(overflow_idx, overflow_query.data(), 1, /*k=*/2, scores.data(), out_ids.data()) ==
              0);
        CHECK(scores[0] == -FLT_MAX);
        CHECK(out_ids[0] == overflow_ids[0]);
        CHECK(scores[1] == -FLT_MAX);
        CHECK(out_ids[1] == UINT64_MAX);
        ggml_vec_index_free(overflow_idx);
    }
    {
        auto * overflow_idx = ggml_vec_index_create(kDim, /*bit_width=*/32);
        CHECK(overflow_idx != nullptr);
        const std::array<float, 2 * kDim> overflow_vecs = {
            1.0e30f, 0.0f, 0.0f, 0.0f, FLT_MAX, 0.0f, 0.0f, 0.0f,
        };
        const std::array<uint64_t, 2> overflow_ids = {
            7654321ULL,
            7654322ULL,
        };
        CHECK(ggml_vec_index_add(overflow_idx, overflow_vecs.data(), 2, overflow_ids.data()) == 0);
        const std::array<float, kDim> overflow_query = {
            FLT_MAX,
            0.0f,
            0.0f,
            0.0f,
        };
        std::array<float, 1>    scores{};
        std::array<uint64_t, 1> out_ids{};
        CHECK(ggml_vec_index_search(overflow_idx, overflow_query.data(), 1, 1, scores.data(), out_ids.data()) == 0);
        CHECK(scores[0] == FLT_MAX);
        CHECK(out_ids[0] == overflow_ids[1]);
        ggml_vec_index_free(overflow_idx);
    }

    // Removing the last slot leaves earlier entries and their vectors intact.
    {
        auto * remove_last_idx = ggml_vec_index_create(kDim, /*bit_width=*/32);
        CHECK(remove_last_idx != nullptr);
        std::vector<float> remove_vecs;
        remove_vecs.insert(remove_vecs.end(), seeds[0].begin(), seeds[0].end());
        remove_vecs.insert(remove_vecs.end(), seeds[1].begin(), seeds[1].end());
        const std::array<uint64_t, 2> remove_ids = { 8001ULL, 8002ULL };
        CHECK(ggml_vec_index_add(remove_last_idx, remove_vecs.data(), 2, remove_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_remove(remove_last_idx, remove_ids[1]) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_len(remove_last_idx) == 1);
        CHECK(ggml_vec_index_contains(remove_last_idx, remove_ids[0]) == 1);

        std::array<float, 1>    scores{};
        std::array<uint64_t, 1> out_ids{};
        CHECK(ggml_vec_index_search(remove_last_idx, seeds[0].data(), 1, 1, scores.data(), out_ids.data()) ==
              GGML_VEC_INDEX_OK);
        CHECK(out_ids[0] == remove_ids[0]);
        CHECK(std::fabs(scores[0] - 1.0f) < 1e-5f);
        ggml_vec_index_free(remove_last_idx);
    }

    // Remove + search: the removed id must no longer surface.
    {
        CHECK(ggml_vec_index_remove(idx, ids[1]) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_remove(idx, ids[1]) == GGML_VEC_INDEX_E_NOT_FOUND);
        CHECK(ggml_vec_index_len(idx) == 3);
        CHECK(ggml_vec_index_contains(idx, ids[1]) == 0);

        std::array<float, 3>    scores{};
        std::array<uint64_t, 3> out_ids{};
        CHECK(ggml_vec_index_search(idx, seeds[1].data(), 1, /*k=*/3, scores.data(), out_ids.data()) == 0);
        for (int i = 0; i < 3; ++i) {
            CHECK(out_ids[i] != ids[1]);
        }

        CHECK(ggml_vec_index_search(idx, seeds[3].data(), 1, /*k=*/1, scores.data(), out_ids.data()) == 0);
        CHECK(out_ids[0] == ids[3]);

        const uint64_t replacement_id = 9001ULL;
        CHECK(ggml_vec_index_add(idx, seeds[1].data(), 1, &replacement_id) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_len(idx) == 4);
        CHECK(ggml_vec_index_search(idx, seeds[1].data(), 1, /*k=*/1, scores.data(), out_ids.data()) == 0);
        CHECK(out_ids[0] == replacement_id);
        CHECK(ggml_vec_index_remove(idx, replacement_id) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_len(idx) == 3);
    }

    // Persistence round-trip: write, free, load, re-query.
    temp_file         round_trip_file(".tvim");
    const std::string path = round_trip_file.path.string();
#ifndef _WIN32
    const mode_t previous_umask = umask(0027);
#endif
    const int write_result = ggml_vec_index_write(idx, path.c_str());
#ifndef _WIN32
    const mode_t write_umask = umask(previous_umask);
#endif
    CHECK(write_result == 0);
#ifndef _WIN32
    CHECK(write_umask == 0027);
#endif
    CHECK(!has_snapshot_tmp(round_trip_file.path));
#ifndef _WIN32
    {
        std::error_code ec;
        const auto      perms = std::filesystem::status(round_trip_file.path, ec).permissions();
        CHECK(!ec);
        CHECK((perms & std::filesystem::perms::owner_read) != std::filesystem::perms::none);
        CHECK((perms & std::filesystem::perms::owner_write) != std::filesystem::perms::none);
        CHECK((perms & std::filesystem::perms::group_read) != std::filesystem::perms::none);
        CHECK((perms & std::filesystem::perms::group_write) == std::filesystem::perms::none);
        CHECK((perms & std::filesystem::perms::others_read) == std::filesystem::perms::none);
    }
#endif
    {
        std::ifstream in(round_trip_file.path, std::ios::binary);
        CHECK(in.is_open());
        const std::vector<uint8_t> actual{
            std::istreambuf_iterator<char>(in),
            std::istreambuf_iterator<char>(),
        };
        std::vector<float> expected_values;
        expected_values.insert(expected_values.end(), seeds[0].begin(), seeds[0].end());
        expected_values.insert(expected_values.end(), seeds[2].begin(), seeds[2].end());
        expected_values.insert(expected_values.end(), seeds[3].begin(), seeds[3].end());
        const std::vector<uint64_t> expected_ids = {
            ids[0],
            ids[2],
            ids[3],
        };
        CHECK(actual == snapshot_bytes(kDim, 3, expected_values, expected_ids));
    }
#ifndef _WIN32
    {
        std::error_code ec;
        std::filesystem::permissions(round_trip_file.path,
                                     std::filesystem::perms::owner_read | std::filesystem::perms::owner_write,
                                     std::filesystem::perm_options::replace, ec);
        CHECK(!ec);
    }
#endif
    CHECK(ggml_vec_index_write(idx, path.c_str()) == 0);
    CHECK(!has_snapshot_tmp(round_trip_file.path));
#ifndef _WIN32
    {
        std::error_code ec;
        const auto      perms = std::filesystem::status(round_trip_file.path, ec).permissions();
        CHECK(!ec);
        CHECK((perms & std::filesystem::perms::owner_read) != std::filesystem::perms::none);
        CHECK((perms & std::filesystem::perms::owner_write) != std::filesystem::perms::none);
        CHECK((perms & std::filesystem::perms::group_read) == std::filesystem::perms::none);
        CHECK((perms & std::filesystem::perms::others_read) == std::filesystem::perms::none);
    }
#endif
    {
        const std::filesystem::path missing_parent = temp_path(".missing-dir");
        const std::filesystem::path bad_path       = missing_parent / "snapshot.tvim";
        CHECK(ggml_vec_index_write(idx, bad_path.string().c_str()) == GGML_VEC_INDEX_E_IO);
    }
#ifdef GGML_VEC_INDEX_TEST_HOOKS
    {
        auto * empty_idx = ggml_vec_index_create(kDim, /*bit_width=*/32);
        CHECK(empty_idx != nullptr);
        temp_file snapshot(".tvim");
        CHECK(ggml_vec_index_write(empty_idx, snapshot.path.string().c_str()) == GGML_VEC_INDEX_OK);
        ggml_vec_index_free(empty_idx);

        const std::vector<uint8_t> before = read_bytes(snapshot.path);
        ggml_vec_index_test_set_write_fail_after(8);
        const int rc = ggml_vec_index_write(idx, snapshot.path.string().c_str());
        ggml_vec_index_test_set_write_fail_after(-1);
        CHECK(rc == GGML_VEC_INDEX_E_IO);
        CHECK(read_bytes(snapshot.path) == before);
        CHECK(!has_snapshot_tmp(snapshot.path));
    }
    {
        auto * empty_idx = ggml_vec_index_create(kDim, /*bit_width=*/32);
        CHECK(empty_idx != nullptr);
        temp_file snapshot(".tvim");
        CHECK(ggml_vec_index_write(empty_idx, snapshot.path.string().c_str()) == GGML_VEC_INDEX_OK);
        ggml_vec_index_free(empty_idx);

        ggml_vec_index_test_set_parent_fsync_fail(1);
        const int rc = ggml_vec_index_write(idx, snapshot.path.string().c_str());
        ggml_vec_index_test_set_parent_fsync_fail(0);
        CHECK(rc == GGML_VEC_INDEX_E_NOT_DURABLE);
        CHECK(!has_snapshot_tmp(snapshot.path));

        auto * replaced = ggml_vec_index_load(snapshot.path.string().c_str());
        CHECK(replaced != nullptr);
        CHECK(ggml_vec_index_len(replaced) == ggml_vec_index_len(idx));
        CHECK(ggml_vec_index_contains(replaced, ids[0]) == 1);
        CHECK(ggml_vec_index_contains(replaced, ids[2]) == 1);
        CHECK(ggml_vec_index_contains(replaced, ids[3]) == 1);
        ggml_vec_index_free(replaced);
    }
#endif
    auto * preserved = ggml_vec_index_load(path.c_str());
    CHECK(preserved != nullptr);
    CHECK(ggml_vec_index_len(preserved) == 3);
    ggml_vec_index_free(preserved);

#ifndef _WIN32
    // A failed overwrite must leave an existing snapshot unchanged.
    if (geteuid() != 0) {
        const std::filesystem::path protected_dir  = temp_path(".protected-dir");
        const std::filesystem::path protected_path = protected_dir / "snapshot.tvim";
        CHECK(std::filesystem::create_directory(protected_dir));

        const std::vector<uint8_t> original = snapshot_bytes(kDim, 0, {}, {});
        write_bytes(protected_path, original);

        std::error_code ec;
        std::filesystem::permissions(protected_dir,
                                     std::filesystem::perms::owner_read | std::filesystem::perms::owner_exec,
                                     std::filesystem::perm_options::replace, ec);
        CHECK(!ec);
        const int protected_write_result = ggml_vec_index_write(idx, protected_path.string().c_str());
        std::filesystem::permissions(protected_dir, std::filesystem::perms::owner_all,
                                     std::filesystem::perm_options::replace, ec);
        CHECK(!ec);

        CHECK(protected_write_result == GGML_VEC_INDEX_E_IO);
        std::ifstream in(protected_path, std::ios::binary);
        CHECK(in.is_open());
        const std::vector<uint8_t> actual{
            std::istreambuf_iterator<char>(in),
            std::istreambuf_iterator<char>(),
        };
        CHECK(actual == original);
        CHECK(!has_snapshot_tmp(protected_path));
        in.close();
        CHECK(std::filesystem::remove_all(protected_dir) == 2);
    }
#endif

    // Replacement failure leaves the destination and no temporary file.
    {
        temp_file directory_path(".dir");
        CHECK(std::filesystem::create_directory(directory_path.path));
        CHECK(ggml_vec_index_write(idx, directory_path.path.string().c_str()) == GGML_VEC_INDEX_E_IO);
        CHECK(std::filesystem::is_directory(directory_path.path));
        CHECK(!has_snapshot_tmp(directory_path.path));
    }

    ggml_vec_index_free(idx);

    auto * loaded = ggml_vec_index_load(path.c_str());
    CHECK(loaded != nullptr);
    CHECK(ggml_vec_index_dim(loaded) == kDim);
    CHECK(ggml_vec_index_len(loaded) == 3);
    CHECK(ggml_vec_index_bit_width(loaded) == 32);
    CHECK(ggml_vec_index_contains(loaded, ids[0]) == 1);
    CHECK(ggml_vec_index_contains(loaded, ids[1]) == 0);  // stayed deleted
    CHECK(ggml_vec_index_contains(loaded, ids[2]) == 1);
    CHECK(ggml_vec_index_contains(loaded, ids[3]) == 1);

    // Top-1 self-match after reload.
    {
        std::array<float, 1>    scores{};
        std::array<uint64_t, 1> out_ids{};
        CHECK(ggml_vec_index_search(loaded, seeds[0].data(), 1, /*k=*/1, scores.data(), out_ids.data()) == 0);
        CHECK(out_ids[0] == ids[0]);
        CHECK(std::fabs(scores[0] - 1.0f) < 1e-5f);
    }

    // Loaded indexes remain fully mutable and can be persisted again.
    {
        const uint64_t loaded_id = 9002ULL;
        CHECK(ggml_vec_index_add(loaded, seeds[1].data(), 1, &loaded_id) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_remove(loaded, ids[2]) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_contains(loaded, loaded_id) == 1);
        CHECK(ggml_vec_index_contains(loaded, ids[2]) == 0);

        temp_file mutated_file(".tvim");
        CHECK(ggml_vec_index_write(loaded, mutated_file.path.string().c_str()) == GGML_VEC_INDEX_OK);
        auto * mutated = ggml_vec_index_load(mutated_file.path.string().c_str());
        CHECK(mutated != nullptr);
        CHECK(ggml_vec_index_len(mutated) == 3);
        CHECK(ggml_vec_index_contains(mutated, loaded_id) == 1);
        CHECK(ggml_vec_index_contains(mutated, ids[2]) == 0);

        std::array<float, 1>    scores{};
        std::array<uint64_t, 1> out_ids{};
        CHECK(ggml_vec_index_search(mutated, seeds[1].data(), 1, 1, scores.data(), out_ids.data()) ==
              GGML_VEC_INDEX_OK);
        CHECK(out_ids[0] == loaded_id);
        ggml_vec_index_free(mutated);
    }

    ggml_vec_index_free(loaded);


    // Quantized storage keeps exact-search ranking close to the f32 baseline.
    for (int bit_width : { 8, 4 }) {
        auto * qidx = ggml_vec_index_create(kDim, bit_width);
        CHECK(qidx != nullptr);
        CHECK(ggml_vec_index_bit_width(qidx) == bit_width);
        CHECK(ggml_vec_index_add(qidx, vecs.data(), static_cast<int>(ids.size()), ids.data()) == GGML_VEC_INDEX_OK);
        temp_file q_snapshot(".tvim");
        CHECK(ggml_vec_index_write(qidx, q_snapshot.path.string().c_str()) == GGML_VEC_INDEX_E_INVALID_ARG);
        std::array<float, 4> q_scores{};
        std::array<uint64_t, 4> q_ids{};
        CHECK(ggml_vec_index_search(qidx, seeds[0].data(), 1, 4, q_scores.data(), q_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(q_ids[0] == ids[0]);
        CHECK(q_scores[0] > 0.95f);
        CHECK(ggml_vec_index_remove(qidx, ids[0]) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search(qidx, seeds[0].data(), 1, 1, q_scores.data(), q_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(q_ids[0] != ids[0]);
        ggml_vec_index_free(qidx);
    }

    check_quantized_reference(8);
    check_quantized_reference(4);
    check_quantized_small_dim_tie_order(8);
    check_quantized_small_dim_tie_order(4);
    check_quantized_simd_near_tie_order(8);
    check_quantized_simd_near_tie_order(4);
    check_quantized_overflow_topk_order(8);
    check_quantized_overflow_topk_order(4);

    for (int bit_width : { 32, 8, 4 }) {
        check_ivf_full_probe_recall(bit_width);
    }
    check_f32_ivf_extreme_centroid_routing();
    check_ivf_centroid_overflow_fallback();
    check_q8_ivf_extreme_centroid_routing();
    check_ivf_empty_batch_state_validation();
    check_ivf_partial_probe_routing();
    for (int bit_width : { 32, 8, 4 }) {
        check_filtered_and_ivf_search(bit_width);
    }
    check_ivf_state_not_persisted();

    // Malformed snapshots are rejected before allocating from untrusted counts.
    {
        temp_file truncated_header(".tvim");
        write_bytes(truncated_header.path, { 'T', 'V', 'P' });
        CHECK(ggml_vec_index_load(truncated_header.path.string().c_str()) == nullptr);
    }
    {
        temp_file oversized_count(".tvim");
        write_bytes(oversized_count.path,
                    snapshot_bytes(
                        /*dim=*/1, static_cast<uint32_t>(std::numeric_limits<int>::max()) + 1u, {}, {}));
        ggml_vec_index_t * out = nullptr;
        CHECK(ggml_vec_index_load_ex(oversized_count.path.string().c_str(), &out) == GGML_VEC_INDEX_E_IO);
        CHECK(out == nullptr);
        CHECK(ggml_vec_index_load(oversized_count.path.string().c_str()) == nullptr);
    }
    {
        temp_file oversized_dim(".tvim");
        write_bytes(oversized_dim.path, snapshot_bytes(static_cast<uint32_t>(std::numeric_limits<int>::max()) + 1u,
                                                       /*n=*/0, {}, {}));
        ggml_vec_index_t * out = nullptr;
        CHECK(ggml_vec_index_load_ex(oversized_dim.path.string().c_str(), &out) == GGML_VEC_INDEX_E_IO);
        CHECK(out == nullptr);
        CHECK(ggml_vec_index_load(oversized_dim.path.string().c_str()) == nullptr);
    }
    {
        temp_file product_overflow(".tvim");
        write_bytes(product_overflow.path,
                    snapshot_bytes(
                        static_cast<uint32_t>(std::numeric_limits<int>::max()),
                        static_cast<uint32_t>(std::numeric_limits<int>::max()),
                        {}, {}));
        ggml_vec_index_t * out = nullptr;
        CHECK(ggml_vec_index_load_ex(product_overflow.path.string().c_str(), &out) == GGML_VEC_INDEX_E_IO);
        CHECK(out == nullptr);
        CHECK(ggml_vec_index_load(product_overflow.path.string().c_str()) == nullptr);
    }
    {
        temp_file truncated_payload(".tvim");
        write_bytes(truncated_payload.path, snapshot_bytes(
                                                /*dim=*/kDim,
                                                /*n=*/2, { 1.0f, 0.0f, 0.0f, 0.0f }, {}));
        ggml_vec_index_t * out = nullptr;
        CHECK(ggml_vec_index_load_ex(truncated_payload.path.string().c_str(), &out) == GGML_VEC_INDEX_E_IO);
        CHECK(out == nullptr);
        CHECK(ggml_vec_index_load(truncated_payload.path.string().c_str()) == nullptr);
    }
    {
        temp_file            trailing_bytes(".tvim");
        std::vector<uint8_t> bytes = snapshot_bytes(
            /*dim=*/kDim,
            /*n=*/1, { 1.0f, 0.0f, 0.0f, 0.0f }, { 123ULL });
        bytes.push_back(0);
        write_bytes(trailing_bytes.path, bytes);
        CHECK(ggml_vec_index_load(trailing_bytes.path.string().c_str()) == nullptr);
    }
    {
        temp_file bad_dim(".tvim");
        write_bytes(bad_dim.path, snapshot_bytes(
                                      /*dim=*/0,
                                      /*n=*/0, {}, {}));
        CHECK(ggml_vec_index_load(bad_dim.path.string().c_str()) == nullptr);
    }
    {
        temp_file            bad_magic(".tvim");
        std::vector<uint8_t> bytes = snapshot_bytes(
            /*dim=*/kDim,
            /*n=*/1, { 1.0f, 0.0f, 0.0f, 0.0f }, { 123ULL });
        bytes[0] = 'X';
        write_bytes(bad_magic.path, bytes);
        CHECK(ggml_vec_index_load(bad_magic.path.string().c_str()) == nullptr);
    }
    {
        temp_file            bad_version(".tvim");
        std::vector<uint8_t> bytes = snapshot_bytes(
            /*dim=*/kDim,
            /*n=*/1, { 1.0f, 0.0f, 0.0f, 0.0f }, { 123ULL });
        bytes[4] = 99;
        write_bytes(bad_version.path, bytes);
        CHECK(ggml_vec_index_load(bad_version.path.string().c_str()) == nullptr);
    }
    {
        temp_file            bad_bit_width(".tvim");
        std::vector<uint8_t> bytes = snapshot_bytes(
            /*dim=*/kDim,
            /*n=*/1, { 1.0f, 0.0f, 0.0f, 0.0f }, { 123ULL });
        bytes[5] = 8;
        write_bytes(bad_bit_width.path, bytes);
        CHECK(ggml_vec_index_load(bad_bit_width.path.string().c_str()) == nullptr);
    }
    {
        temp_file            reserved_header(".tvim");
        std::vector<uint8_t> bytes = snapshot_bytes(
            /*dim=*/kDim,
            /*n=*/1, { 1.0f, 0.0f, 0.0f, 0.0f }, { 123ULL });
        CHECK(bytes.size() >= 8);
        bytes[6] = 1;
        write_bytes(reserved_header.path, bytes);
        CHECK(ggml_vec_index_load(reserved_header.path.string().c_str()) == nullptr);
        bytes[6] = 0;
        bytes[7] = 1;
        write_bytes(reserved_header.path, bytes);
        CHECK(ggml_vec_index_load(reserved_header.path.string().c_str()) == nullptr);
    }
    {
        temp_file duplicate_ids(".tvim");
        write_bytes(duplicate_ids.path, snapshot_bytes(
                                            /*dim=*/kDim,
                                            /*n=*/2,
                                            {
                                                1.0f,
                                                0.0f,
                                                0.0f,
                                                0.0f,
                                                0.0f,
                                                1.0f,
                                                0.0f,
                                                0.0f,
                                            },
                                            { 321ULL, 321ULL }));
        ggml_vec_index_t * duplicate_out = nullptr;
        CHECK(ggml_vec_index_load_ex(duplicate_ids.path.string().c_str(), &duplicate_out) == GGML_VEC_INDEX_E_IO);
        CHECK(duplicate_out == nullptr);
        CHECK(ggml_vec_index_load(duplicate_ids.path.string().c_str()) == nullptr);
    }
    {
        temp_file reserved_id_file(".tvim");
        write_bytes(reserved_id_file.path, snapshot_bytes(
                                               /*dim=*/kDim,
                                               /*n=*/1, { 1.0f, 0.0f, 0.0f, 0.0f }, { UINT64_MAX }));
        CHECK(ggml_vec_index_load(reserved_id_file.path.string().c_str()) == nullptr);
    }
    {
        temp_file non_finite_file(".tvim");
        write_bytes(non_finite_file.path,
                    snapshot_bytes(
                        /*dim=*/kDim,
                        /*n=*/1, { std::numeric_limits<float>::infinity(), 0.0f, 0.0f, 0.0f }, { 321ULL }));
        CHECK(ggml_vec_index_load(non_finite_file.path.string().c_str()) == nullptr);
    }

    std::printf("test-vector-index: OK\n");
    return 0;
}
