// test-vector-index.cpp - standalone C-API smoke test for the POC vector
// index. Exercises lifecycle, add, search, remove, contains, write, load,
// search-after-load. No model, no llama; only the new ggml-vector-index
// public C API.

#include "ggml-vector-index.h"
#include "ggml-vector-index-impl.h"

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

bool has_snapshot_tmp(const std::filesystem::path & path) {
    const std::filesystem::path dir    = path.parent_path().empty() ? std::filesystem::path(".") : path.parent_path();
    const std::string           prefix = path.filename().string() + ".tmp.";

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

}  // namespace

int main() {
    auto * idx = ggml_vec_index_create(kDim, /*bit_width=*/32);
    CHECK(idx != nullptr);
    CHECK(ggml_vec_index_dim(idx) == kDim);
    CHECK(ggml_vec_index_len(idx) == 0);
    CHECK(ggml_vec_index_bit_width(idx) == 32);
    CHECK(ggml_vec_index_create(0, /*bit_width=*/32) == nullptr);
    CHECK(ggml_vec_index_create(kDim, /*bit_width=*/31) == nullptr);

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
        CHECK(ggml_vec_index_write(nullptr, "unused.tvim") == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_write(idx, nullptr) == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_load(nullptr) == nullptr);
        CHECK(ggml_vec_index_load(temp_path(".missing").string().c_str()) == nullptr);
        CHECK(ggml_vec_index_remove(nullptr, id) == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_contains(nullptr, id) == 0);
        CHECK(ggml_vec_index_len(nullptr) == 0);
        CHECK(ggml_vec_index_dim(nullptr) == 0);
        CHECK(ggml_vec_index_bit_width(nullptr) == 0);
        ggml_vec_index_prepare(nullptr);
        CHECK(ggml_vec_index_len(idx) == 0);
    }

    // Byte-span arithmetic is checked independently of the host word size so
    // 64-bit CI still covers the overflow boundary used by 32-bit builds.
    {
        constexpr size_t max_size = std::numeric_limits<size_t>::max();
        CHECK(ggml_vec_index_detail::can_address_array(max_size / sizeof(float), sizeof(float)));
        CHECK(!ggml_vec_index_detail::can_address_array(max_size / sizeof(float) + 1, sizeof(float)));
        CHECK(ggml_vec_index_detail::can_address_array(max_size / sizeof(uint64_t), sizeof(uint64_t)));
        CHECK(!ggml_vec_index_detail::can_address_array(max_size / sizeof(uint64_t) + 1, sizeof(uint64_t)));
    }

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

    // The documented oversized-index example exceeds the v1 persistence limit.
    {
        constexpr size_t n   = 262144;
        constexpr size_t dim = 4096;
        size_t           data_size = 0;
        CHECK(ggml_vec_index_detail::checked_mul_size(n, dim, data_size));
        CHECK(ggml_vec_index_detail::snapshot_write_preflight(n, dim, data_size) ==
              GGML_VEC_INDEX_E_INVALID_ARG);
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
        expected_values.insert(expected_values.end(), seeds[3].begin(), seeds[3].end());
        expected_values.insert(expected_values.end(), seeds[2].begin(), seeds[2].end());
        const std::vector<uint64_t> expected_ids = {
            ids[0],
            ids[3],
            ids[2],
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
        CHECK(ggml_vec_index_load(oversized_count.path.string().c_str()) == nullptr);
    }
    {
        temp_file oversized_dim(".tvim");
        write_bytes(oversized_dim.path, snapshot_bytes(static_cast<uint32_t>(std::numeric_limits<int>::max()) + 1u,
                                                       /*n=*/0, {}, {}));
        CHECK(ggml_vec_index_load(oversized_dim.path.string().c_str()) == nullptr);
    }
    {
        temp_file truncated_payload(".tvim");
        write_bytes(truncated_payload.path, snapshot_bytes(
                                                /*dim=*/kDim,
                                                /*n=*/2, { 1.0f, 0.0f, 0.0f, 0.0f }, {}));
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
