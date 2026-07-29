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
#include <cfenv>
#include <chrono>
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
#include <thread>
#include <vector>

#ifdef GGML_VEC_INDEX_TEST_HOOKS
extern "C" {
void    ggml_vec_index_test_set_oom_countdown(int64_t countdown);
void    ggml_vec_index_test_set_parent_fsync_fail(int fail);
void    ggml_vec_index_test_set_parent_fsync_fail_after(int64_t count);
void    ggml_vec_index_test_set_load_with_delta_block(int block);
int     ggml_vec_index_test_get_load_with_delta_waiters(void);
void    ggml_vec_index_test_reset_state_crc_scan_count(void);
int64_t ggml_vec_index_test_get_state_crc_scan_count(void);
void    ggml_vec_index_test_reset_delta_max_read_size(void);
size_t  ggml_vec_index_test_get_delta_max_read_size(void);
void    ggml_vec_index_test_reset_mmap_count_reject_count(void);
int64_t ggml_vec_index_test_get_mmap_count_reject_count(void);
void    ggml_vec_index_test_reset_load_count_reject_count(void);
int64_t ggml_vec_index_test_get_load_count_reject_count(void);
}
#endif

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

uint32_t crc32c_update(uint32_t crc, const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < size; ++i) {
        crc ^= bytes[i];
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1) ^ (0x82f63b78u & (0u - (crc & 1u)));
        }
    }
    return crc;
}

uint32_t crc32c_bytes(const std::vector<uint8_t> & bytes) {
    if (bytes.empty()) {
        return 0u;
    }
    return crc32c_update(0xffffffffu, bytes.data(), bytes.size()) ^ 0xffffffffu;
}

void put_u32_le(std::vector<uint8_t> & bytes, size_t offset, uint32_t value) {
    CHECK(offset <= bytes.size() && bytes.size() - offset >= sizeof(uint32_t));
    for (int i = 0; i < 4; ++i) {
        bytes[offset + static_cast<size_t>(i)] = static_cast<uint8_t>(value >> (8 * i));
    }
}

uint32_t crc32c_u32(uint32_t crc, uint32_t value) {
    std::array<uint8_t, 4> bytes{};
    for (int i = 0; i < 4; ++i) {
        bytes[static_cast<size_t>(i)] = static_cast<uint8_t>(value >> (8 * i));
    }
    return crc32c_update(crc, bytes.data(), bytes.size());
}

uint32_t crc32c_u64(uint32_t crc, uint64_t value) {
    std::array<uint8_t, 8> bytes{};
    for (int i = 0; i < 8; ++i) {
        bytes[static_cast<size_t>(i)] = static_cast<uint8_t>(value >> (8 * i));
    }
    return crc32c_update(crc, bytes.data(), bytes.size());
}

uint32_t f32_state_crc(uint32_t dim, const std::vector<float> & vectors, const std::vector<uint64_t> & ids) {
    CHECK(vectors.size() == ids.size() * dim);
    uint32_t crc = 0xffffffffu;
    crc          = crc32c_u32(crc, dim);
    crc          = crc32c_u32(crc, 32);
    crc          = crc32c_u32(crc, 1);
    crc          = crc32c_u64(crc, ids.size());
    for (float value : vectors) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        crc = crc32c_u32(crc, bits);
    }
    for (uint64_t id : ids) {
        crc = crc32c_u64(crc, id);
    }
    return crc ^ 0xffffffffu;
}

uint64_t rotl64_ref(uint64_t value, int shift) {
    return (value << shift) | (value >> (64 - shift));
}

uint64_t f32_slot_state_hash(uint64_t id, const float * vector, size_t dim) {
    uint32_t crc0 = 0xffffffffu;
    uint32_t crc1 = 0x82f63b78u;
    crc0          = crc32c_u64(crc0, id);
    crc1          = crc32c_u64(crc1, id ^ 0xa5a5a5a5a5a5a5a5ULL);
    for (size_t i = 0; i < dim; ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, vector + i, sizeof(bits));
        crc0 = crc32c_u32(crc0, bits);
        crc1 = crc32c_u32(crc1, bits ^ 0xa5a5a5a5u);
    }
    return (static_cast<uint64_t>(crc0 ^ 0xffffffffu) << 32) | static_cast<uint64_t>(crc1 ^ 0xffffffffu);
}

uint64_t encoded_slot_state_hash(uint64_t id, float scale, const std::vector<uint8_t> & codes) {
    uint32_t crc0       = 0xffffffffu;
    uint32_t crc1       = 0x82f63b78u;
    crc0                = crc32c_u64(crc0, id);
    crc1                = crc32c_u64(crc1, id ^ 0xa5a5a5a5a5a5a5a5ULL);
    uint32_t scale_bits = 0;
    std::memcpy(&scale_bits, &scale, sizeof(scale_bits));
    crc0 = crc32c_u32(crc0, scale_bits);
    crc1 = crc32c_u32(crc1, scale_bits ^ 0xa5a5a5a5u);
    crc0 = crc32c_update(crc0, codes.data(), codes.size());
    crc1 = crc32c_update(crc1, codes.data(), codes.size());
    return (static_cast<uint64_t>(crc0 ^ 0xffffffffu) << 32) | static_cast<uint64_t>(crc1 ^ 0xffffffffu);
}

std::array<uint64_t, 4> wide_state_from_hashes(std::initializer_list<uint64_t> hashes) {
    std::array<uint64_t, 4> state = { hashes.size(), 0, 0, 0 };
    for (uint64_t hash : hashes) {
        state[1] ^= hash;
        state[2] += hash;
        state[3] += rotl64_ref(hash, 17);
    }
    return state;
}

uint32_t encoded_state_token(const std::array<uint64_t, 4> & state, int dim, int bit_width) {
    uint32_t crc = 0xffffffffu;
    crc          = crc32c_u32(crc, static_cast<uint32_t>(dim));
    crc          = crc32c_u32(crc, static_cast<uint32_t>(bit_width));
    crc          = crc32c_u32(crc, bit_width == 4 ? 3u : 2u);
    for (uint64_t value : state) {
        crc = crc32c_u64(crc, value);
    }
    return crc ^ 0xffffffffu;
}

uint32_t encoded_state_crc(int                                dim,
                           int                                bit_width,
                           const std::vector<float> &         scales,
                           const std::vector<std::vector<uint8_t>> & rows,
                           const std::vector<uint64_t> &      ids) {
    CHECK(scales.size() == rows.size());
    CHECK(rows.size() == ids.size());
    uint32_t crc = 0xffffffffu;
    crc          = crc32c_u32(crc, static_cast<uint32_t>(dim));
    crc          = crc32c_u32(crc, static_cast<uint32_t>(bit_width));
    crc          = crc32c_u32(crc, bit_width == 4 ? 3u : 2u);
    crc          = crc32c_u64(crc, ids.size());
    for (float scale : scales) {
        uint32_t bits = 0;
        std::memcpy(&bits, &scale, sizeof(bits));
        crc = crc32c_u32(crc, bits);
    }
    for (const auto & row : rows) {
        crc = crc32c_update(crc, row.data(), row.size());
    }
    for (uint64_t id : ids) {
        crc = crc32c_u64(crc, id);
    }
    return crc ^ 0xffffffffu;
}

std::array<uint64_t, 4> f32_wide_state(const std::vector<float> &    vectors,
                                       const std::vector<uint64_t> & ids,
                                       size_t                        dim) {
    CHECK(vectors.size() == ids.size() * dim);
    std::array<uint64_t, 4> state = { ids.size(), 0, 0, 0 };
    for (size_t i = 0; i < ids.size(); ++i) {
        const uint64_t hash = f32_slot_state_hash(ids[i], vectors.data() + i * dim, dim);
        state[1] ^= hash;
        state[2] += hash;
        state[3] += rotl64_ref(hash, 17);
    }
    return state;
}

uint32_t f32_state_token(const std::vector<float> & vectors, const std::vector<uint64_t> & ids, size_t dim) {
    const auto state = f32_wide_state(vectors, ids, dim);
    uint32_t   crc   = 0xffffffffu;
    crc              = crc32c_u32(crc, static_cast<uint32_t>(dim));
    crc              = crc32c_u32(crc, 32);
    crc              = crc32c_u32(crc, 1);
    for (uint64_t value : state) {
        crc = crc32c_u64(crc, value);
    }
    return crc ^ 0xffffffffu;
}

void append_wide_state(std::vector<uint8_t> & bytes, const std::array<uint64_t, 4> & state) {
    for (uint64_t value : state) {
        append_u64_le(bytes, value);
    }
}

void append_v1_delta_record(std::vector<uint8_t> &       log,
                            uint8_t                      op,
                            uint32_t                     n,
                            const std::vector<uint8_t> & payload,
                            uint32_t                     state_crc) {
    const size_t record_offset = log.size();
    log.insert(log.end(), { op, 0, 0, 0 });
    append_u32_le(log, n);
    append_u64_le(log, payload.size());
    append_u32_le(log, 0);
    append_u32_le(log, state_crc);
    const size_t payload_offset = log.size();
    log.insert(log.end(), payload.begin(), payload.end());

    uint32_t crc = crc32c_update(0xffffffffu, log.data() + record_offset, 16);
    crc          = crc32c_update(crc, log.data() + record_offset + 20, 4);
    if (!payload.empty()) {
        crc = crc32c_update(crc, log.data() + payload_offset, payload.size());
    }
    put_u32_le(log, record_offset + 16, crc ^ 0xffffffffu);
}

void append_v4_delta_record(std::vector<uint8_t> &          log,
                            uint8_t                         op,
                            uint32_t                        n,
                            const std::vector<uint8_t> &    payload,
                            const std::array<uint64_t, 4> & state) {
    const size_t record_offset = log.size();
    log.insert(log.end(), { op, 0, 0, 0 });
    append_u32_le(log, n);
    append_u64_le(log, payload.size());
    append_u32_le(log, 0);
    append_u32_le(log, 0);
    append_wide_state(log, state);
    const size_t payload_offset = log.size();
    log.insert(log.end(), payload.begin(), payload.end());

    uint32_t crc = crc32c_update(0xffffffffu, log.data() + record_offset, 16);
    crc          = crc32c_update(crc, log.data() + record_offset + 24, 32);
    if (!payload.empty()) {
        crc = crc32c_update(crc, log.data() + payload_offset, payload.size());
    }
    put_u32_le(log, record_offset + 16, crc ^ 0xffffffffu);
}

void write_bytes(const std::filesystem::path & path, const std::vector<uint8_t> & bytes) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    CHECK(f.is_open());
    if (!bytes.empty()) {
        f.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    CHECK(static_cast<bool>(f));
}

bool try_write_sparse_bytes(const std::filesystem::path & path, const std::vector<uint8_t> & prefix, uint64_t size) {
    if (size < prefix.size() || size > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        return false;
    }
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f.is_open()) {
        return false;
    }
    if (!prefix.empty()) {
        f.write(reinterpret_cast<const char *>(prefix.data()), static_cast<std::streamsize>(prefix.size()));
    }
    if (size > prefix.size()) {
        f.seekp(static_cast<std::streamoff>(size - 1));
        const char zero = 0;
        f.write(&zero, 1);
    }
    return static_cast<bool>(f);
}

void write_sparse_bytes(const std::filesystem::path & path, const std::vector<uint8_t> & prefix, uint64_t size) {
    CHECK(try_write_sparse_bytes(path, prefix, size));
}

std::vector<uint8_t> read_bytes(const std::filesystem::path & path) {
    std::ifstream f(path, std::ios::binary);
    CHECK(f.is_open());
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
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

std::vector<uint8_t> snapshot_v2_f32_bytes(uint32_t                      dim,
                                           uint32_t                      n,
                                           const std::vector<float> &    values,
                                           const std::vector<uint64_t> & ids) {
    std::vector<uint8_t> header;
    header.reserve(32);
    header.insert(header.end(), { 'T', 'V', 'P', 'I' });
    header.push_back(2);
    header.push_back(32);
    header.push_back(1);
    header.push_back(1);
    append_u32_le(header, dim);
    append_u32_le(header, n);
    append_u32_le(header, 0);
    append_u32_le(header, 0);
    append_u32_le(header, 4);
    append_u32_le(header, 0);

    std::vector<uint8_t> vector_bytes;
    for (float value : values) {
        append_f32_le(vector_bytes, value);
    }
    std::vector<uint8_t> id_bytes;
    for (uint64_t id : ids) {
        append_u64_le(id_bytes, id);
    }

    std::vector<uint8_t> bytes = header;
    bytes.insert(bytes.end(), vector_bytes.begin(), vector_bytes.end());
    bytes.insert(bytes.end(), id_bytes.begin(), id_bytes.end());
    append_u32_le(bytes, crc32c_bytes(header));
    append_u32_le(bytes, 0);
    append_u32_le(bytes, crc32c_bytes(vector_bytes));
    append_u32_le(bytes, crc32c_bytes(id_bytes));
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
        1.0f, 0.0f, 2.0f, 0.0f, 0.0f, 1.0f, 100.0f, 101.0f,
    };
    const std::array<uint64_t, 4> ids   = { 8250ULL, 8251ULL, 8252ULL, 8253ULL };
    const std::array<float, dim>  query = { 1.0f, 0.0f };

    auto * idx = ggml_vec_index_create(dim, /*bit_width=*/32);
    CHECK(idx != nullptr);
    CHECK(ggml_vec_index_add(idx, vectors.data(), static_cast<int>(ids.size()), ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_build_ivf(idx, /*n_lists=*/2, /*n_iter=*/0) == GGML_VEC_INDEX_OK);

    std::array<float, 1>    partial_score{};
    std::array<uint64_t, 1> partial_id{};
    CHECK(ggml_vec_index_search_ivf(idx, query.data(), 1, 1, /*nprobe=*/1, partial_score.data(), partial_id.data()) ==
          GGML_VEC_INDEX_OK);
    CHECK(partial_id[0] == ids[1]);
    CHECK(partial_score[0] == 2.0f);

    std::array<float, 1>    full_score{};
    std::array<uint64_t, 1> full_id{};
    CHECK(ggml_vec_index_search_ivf(idx, query.data(), 1, 1, /*nprobe=*/2, full_score.data(), full_id.data()) ==
          GGML_VEC_INDEX_OK);
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
        FLT_MAX,
        0.0f,
        0.0f,
        1.0f,
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

void check_quantized_flt_max_persistence_round_trip(int bit_width) {
    const std::array<float, kDim> vector = {
        FLT_MAX,
        0.0f,
        0.0f,
        0.0f,
    };
    const std::array<float, kDim> query = {
        1.0f,
        0.0f,
        0.0f,
        0.0f,
    };
    const uint64_t id = 8450ULL;

    temp_file snapshot(".tvim");
    auto *    idx = ggml_vec_index_create(kDim, bit_width);
    CHECK(idx != nullptr);
    CHECK(ggml_vec_index_add(idx, vector.data(), 1, &id) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(idx, snapshot.path.string().c_str()) == GGML_VEC_INDEX_OK);
    ggml_vec_index_free(idx);

    auto * loaded = ggml_vec_index_load(snapshot.path.string().c_str());
    CHECK(loaded != nullptr);
    auto * mmap = ggml_vec_index_load_mmap(snapshot.path.string().c_str());
    CHECK(mmap != nullptr);

    for (auto * handle : { loaded, mmap }) {
        std::array<float, 1>    scores{};
        std::array<uint64_t, 1> out_ids{};
        CHECK(ggml_vec_index_search(handle, query.data(), 1, 1, scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(out_ids[0] == id);
        CHECK(scores[0] == FLT_MAX);
    }

    ggml_vec_index_free(mmap);
    ggml_vec_index_free(loaded);
}

void check_quantized_reconstruction_overflow_rejected() {
    const std::array<float, kDim> vector = {
        1.0f,
        0.0f,
        0.0f,
        0.0f,
    };
    const uint64_t id = 8460ULL;

    temp_file snapshot(".tvim");
    auto *    idx = ggml_vec_index_create(kDim, /*bit_width=*/8);
    CHECK(idx != nullptr);
    CHECK(ggml_vec_index_add(idx, vector.data(), 1, &id) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(idx, snapshot.path.string().c_str()) == GGML_VEC_INDEX_OK);
    ggml_vec_index_free(idx);

    std::vector<uint8_t> bad = read_bytes(snapshot.path);
    const size_t qparams_offset  = 32;
    const size_t vectors_offset  = qparams_offset + sizeof(uint32_t);
    const size_t ids_offset      = vectors_offset + kDim;
    const size_t checksums_offset = ids_offset + sizeof(uint64_t);
    uint32_t     max_bits = 0;
    const float  max_value = FLT_MAX;
    std::memcpy(&max_bits, &max_value, sizeof(max_bits));
    const double reconstruction_limit =
        static_cast<double>(FLT_MAX) * (1.0 + static_cast<double>(std::numeric_limits<float>::epsilon()));
    float invalid_q8_scale = static_cast<float>(reconstruction_limit / 127.0);
    while (static_cast<double>(invalid_q8_scale) * 127.0 <= reconstruction_limit) {
        invalid_q8_scale = std::nextafter(invalid_q8_scale, std::numeric_limits<float>::infinity());
    }
    uint32_t invalid_q8_scale_bits = 0;
    std::memcpy(&invalid_q8_scale_bits, &invalid_q8_scale, sizeof(invalid_q8_scale_bits));
    put_u32_le(bad, qparams_offset, invalid_q8_scale_bits);
    bad[vectors_offset] = 127;
    const std::vector<uint8_t> qparams(
        bad.begin() + static_cast<std::ptrdiff_t>(qparams_offset),
        bad.begin() + static_cast<std::ptrdiff_t>(vectors_offset));
    const std::vector<uint8_t> vectors(
        bad.begin() + static_cast<std::ptrdiff_t>(vectors_offset),
        bad.begin() + static_cast<std::ptrdiff_t>(ids_offset));
    put_u32_le(bad, checksums_offset + 4, crc32c_bytes(qparams));
    put_u32_le(bad, checksums_offset + 8, crc32c_bytes(vectors));
    write_bytes(snapshot.path, bad);
    CHECK(ggml_vec_index_load(snapshot.path.string().c_str()) == nullptr);
    CHECK(ggml_vec_index_load_mmap(snapshot.path.string().c_str()) == nullptr);

    temp_file q4_snapshot(".tvim");
    auto *    q4 = ggml_vec_index_create(kDim, /*bit_width=*/4);
    CHECK(q4 != nullptr);
    CHECK(ggml_vec_index_add(q4, vector.data(), 1, &id) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(q4, q4_snapshot.path.string().c_str()) == GGML_VEC_INDEX_OK);
    ggml_vec_index_free(q4);
    bad = read_bytes(q4_snapshot.path);
    const size_t q4_vectors_offset   = qparams_offset + sizeof(uint32_t);
    const size_t q4_ids_offset       = q4_vectors_offset + (kDim + 1) / 2;
    const size_t q4_checksums_offset = q4_ids_offset + sizeof(uint64_t);
    put_u32_le(bad, qparams_offset, max_bits);
    bad[q4_vectors_offset] = static_cast<uint8_t>((bad[q4_vectors_offset] & 0xf0u) | 0x0fu);
    const std::vector<uint8_t> q4_qparams(
        bad.begin() + static_cast<std::ptrdiff_t>(qparams_offset),
        bad.begin() + static_cast<std::ptrdiff_t>(q4_vectors_offset));
    const std::vector<uint8_t> q4_vectors(
        bad.begin() + static_cast<std::ptrdiff_t>(q4_vectors_offset),
        bad.begin() + static_cast<std::ptrdiff_t>(q4_ids_offset));
    put_u32_le(bad, q4_checksums_offset + 4, crc32c_bytes(q4_qparams));
    put_u32_le(bad, q4_checksums_offset + 8, crc32c_bytes(q4_vectors));
    write_bytes(q4_snapshot.path, bad);
    CHECK(ggml_vec_index_load(q4_snapshot.path.string().c_str()) == nullptr);
    CHECK(ggml_vec_index_load_mmap(q4_snapshot.path.string().c_str()) == nullptr);

    temp_file empty_snapshot(".tvim");
    temp_file delta(".tvid");
    auto *    empty = ggml_vec_index_create(kDim, /*bit_width=*/8);
    CHECK(empty != nullptr);
    CHECK(ggml_vec_index_write(empty, empty_snapshot.path.string().c_str()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_compact_delta(empty, empty_snapshot.path.string().c_str(), delta.path.string().c_str()) ==
          GGML_VEC_INDEX_OK);
    std::vector<uint8_t> log = read_bytes(delta.path);
    std::vector<uint8_t> payload;
    append_u64_le(payload, id);
    append_u32_le(payload, invalid_q8_scale_bits);
    const std::vector<uint8_t> bad_codes = { 127, 0, 0, 0 };
    payload.insert(payload.end(), bad_codes.begin(), bad_codes.end());
    const auto bad_state = wide_state_from_hashes({ encoded_slot_state_hash(id, invalid_q8_scale, bad_codes) });
    append_v4_delta_record(log, /*op=*/1, /*n=*/1, payload, bad_state);
    write_bytes(delta.path, log);
    ggml_vec_index_free(empty);
    CHECK(ggml_vec_index_load_with_delta(empty_snapshot.path.string().c_str(), delta.path.string().c_str()) == nullptr);

    std::error_code       ec;
    std::filesystem::path lock_path = delta.path;
    lock_path += ".lock";
    std::filesystem::remove(lock_path, ec);
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

void check_committed_delta_replay() {
    const std::vector<float> base_vector = {
        1.0f,
        0.0f,
        0.0f,
        0.0f,
    };
    const std::vector<float> added_vector = {
        0.0f,
        1.0f,
        0.0f,
        0.0f,
    };
    const uint64_t base_id  = 9251ULL;
    const uint64_t added_id = 9252ULL;

    temp_file snapshot(".tvim");
    auto *    base = ggml_vec_index_create(kDim, /*bit_width=*/32);
    CHECK(base != nullptr);
    CHECK(ggml_vec_index_add(base, base_vector.data(), 1, &base_id) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(base, snapshot.path.string().c_str()) == GGML_VEC_INDEX_OK);
    ggml_vec_index_free(base);

    const std::vector<float> after_add_vectors = {
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
    };
    const std::vector<uint64_t> after_add_ids        = { base_id, added_id };
    const std::vector<float>    after_remove_vectors = added_vector;
    const std::vector<uint64_t> after_remove_ids     = { added_id };

    std::vector<uint8_t> add_payload;
    append_u64_le(add_payload, added_id);
    for (float value : added_vector) {
        append_f32_le(add_payload, value);
    }
    std::vector<uint8_t> remove_payload;
    append_u64_le(remove_payload, base_id);

    for (int version : { 1, 2, 3, 4 }) {
        std::vector<uint8_t> log = { 'T', 'V', 'D', 'L', static_cast<uint8_t>(version), 32, 0, 0 };
        append_u32_le(log, kDim);
        if (version != 4) {
            const auto state_value = [&](const std::vector<float> & vectors, const std::vector<uint64_t> & ids) {
                return version == 1 ? f32_state_crc(kDim, vectors, ids) : f32_state_token(vectors, ids, kDim);
            };
            append_u32_le(log, state_value(base_vector, { base_id }));
            append_v1_delta_record(log,
                                   /*op=*/1,
                                   /*n=*/1, add_payload, state_value(after_add_vectors, after_add_ids));
            append_v1_delta_record(log,
                                   /*op=*/2,
                                   /*n=*/1, remove_payload, state_value(after_remove_vectors, after_remove_ids));
        } else {
            append_u32_le(log, 0);
            append_wide_state(log, f32_wide_state(base_vector, { base_id }, kDim));
            append_v4_delta_record(log,
                                   /*op=*/1,
                                   /*n=*/1, add_payload, f32_wide_state(after_add_vectors, after_add_ids, kDim));
            append_v4_delta_record(log,
                                   /*op=*/2,
                                   /*n=*/1, remove_payload,
                                   f32_wide_state(after_remove_vectors, after_remove_ids, kDim));
        }

        temp_file delta(".tvid");
        write_bytes(delta.path, log);
#ifdef GGML_VEC_INDEX_TEST_HOOKS
        if (version == 1) {
            ggml_vec_index_test_reset_state_crc_scan_count();
        }
#endif
        auto * loaded = ggml_vec_index_load_with_delta(snapshot.path.string().c_str(), delta.path.string().c_str());
        CHECK(loaded != nullptr);
#ifdef GGML_VEC_INDEX_TEST_HOOKS
        if (version == 1) {
            CHECK(ggml_vec_index_test_get_state_crc_scan_count() == 2);
        }
#endif
        CHECK(ggml_vec_index_len(loaded) == 1);
        CHECK(ggml_vec_index_contains(loaded, base_id) == 0);
        CHECK(ggml_vec_index_contains(loaded, added_id) == 1);
        std::array<float, 1>    scores{};
        std::array<uint64_t, 1> out_ids{};
        CHECK(ggml_vec_index_search(loaded, added_vector.data(), 1, 1, scores.data(), out_ids.data()) ==
              GGML_VEC_INDEX_OK);
        CHECK(out_ids[0] == added_id);

        const uint64_t rejected_id = 9253ULL;
        temp_file     rejected_snapshot(".tvim");
        CHECK(ggml_vec_index_add(loaded, base_vector.data(), 1, &rejected_id) ==
              GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_remove(loaded, added_id) == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_compact(loaded) == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_write(loaded, rejected_snapshot.path.string().c_str()) ==
              GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_len(loaded) == 1);
        CHECK(ggml_vec_index_contains(loaded, added_id) == 1);
        ggml_vec_index_free(loaded);

#ifdef GGML_VEC_INDEX_TEST_HOOKS
        if (version == 4) {
            temp_file race_delta(".tvid");
            write_bytes(race_delta.path, log);
            const std::filesystem::path moved_path = race_delta.path.string() + ".moved";
            ggml_vec_index_t *          raced      = nullptr;
            ggml_vec_index_test_set_load_with_delta_block(1);
            std::thread loader([&]() {
                raced =
                    ggml_vec_index_load_with_delta(snapshot.path.string().c_str(), race_delta.path.string().c_str());
            });
            const auto  deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (ggml_vec_index_test_get_load_with_delta_waiters() == 0 &&
                   std::chrono::steady_clock::now() < deadline) {
                std::this_thread::yield();
            }
            if (ggml_vec_index_test_get_load_with_delta_waiters() == 0) {
                ggml_vec_index_test_set_load_with_delta_block(0);
                loader.join();
                CHECK(false);
            }
            std::filesystem::rename(race_delta.path, moved_path);
            write_bytes(race_delta.path, { 'T', 'V', 'D', 'L' });
            ggml_vec_index_test_set_load_with_delta_block(0);
            loader.join();
            CHECK(raced == nullptr);
            std::error_code ec;
            std::filesystem::remove(moved_path, ec);
        }
#endif
    }
}

void check_quantized_committed_delta_replay(int bit_width) {
    const std::array<float, kDim> base_vector = {
        1.0f,
        0.25f,
        -0.5f,
        0.75f,
    };
    std::array<float, kDim> added_vector = {
        -0.25f,
        1.0f,
        0.5f,
        -0.75f,
    };
    if (bit_width == 8) {
        added_vector = {
            FLT_MAX,
            0.0f,
            0.0f,
            0.0f,
        };
    }
    const uint64_t base_id  = 9271ULL + static_cast<uint64_t>(bit_width);
    const uint64_t added_id = 9281ULL + static_cast<uint64_t>(bit_width);

    temp_file snapshot(".tvim");
    auto *    base = ggml_vec_index_create(kDim, bit_width);
    CHECK(base != nullptr);
    CHECK(ggml_vec_index_add(base, base_vector.data(), 1, &base_id) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(base, snapshot.path.string().c_str()) == GGML_VEC_INDEX_OK);
    ggml_vec_index_free(base);

    temp_file added_snapshot(".tvim");
    auto *    added = ggml_vec_index_create(kDim, bit_width);
    CHECK(added != nullptr);
    CHECK(ggml_vec_index_add(added, added_vector.data(), 1, &added_id) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(added, added_snapshot.path.string().c_str()) == GGML_VEC_INDEX_OK);
    ggml_vec_index_free(added);

    struct encoded_row {
        float                scale = 0.0f;
        std::vector<uint8_t> codes;
    };

    const auto read_row = [&](const std::filesystem::path & path) {
        const std::vector<uint8_t> bytes     = read_bytes(path);
        const size_t               row_bytes = bit_width == 4 ? (static_cast<size_t>(kDim) + 1) / 2 : kDim;
        CHECK(bytes.size() >= 36 + row_bytes);
        uint32_t scale_bits = 0;
        for (int i = 0; i < 4; ++i) {
            scale_bits |= static_cast<uint32_t>(bytes[32 + static_cast<size_t>(i)]) << (8 * i);
        }
        encoded_row row;
        std::memcpy(&row.scale, &scale_bits, sizeof(row.scale));
        row.codes.assign(bytes.begin() + 36, bytes.begin() + static_cast<std::ptrdiff_t>(36 + row_bytes));
        return row;
    };
    const encoded_row base_row    = read_row(snapshot.path);
    const encoded_row added_row   = read_row(added_snapshot.path);
    const uint64_t    base_hash   = encoded_slot_state_hash(base_id, base_row.scale, base_row.codes);
    const uint64_t    added_hash  = encoded_slot_state_hash(added_id, added_row.scale, added_row.codes);
    const auto        base_wide   = wide_state_from_hashes({ base_hash });
    const auto        add_wide    = wide_state_from_hashes({ base_hash, added_hash });
    const auto        remove_wide = wide_state_from_hashes({ added_hash });
    const auto        state_token = [&](const std::array<uint64_t, 4> & state) {
        uint32_t crc = 0xffffffffu;
        crc          = crc32c_u32(crc, kDim);
        crc          = crc32c_u32(crc, static_cast<uint32_t>(bit_width));
        crc          = crc32c_u32(crc, bit_width == 4 ? 3 : 2);
        for (uint64_t value : state) {
            crc = crc32c_u64(crc, value);
        }
        return crc ^ 0xffffffffu;
    };
    const auto state_crc = [&](const std::vector<const encoded_row *> & rows, const std::vector<uint64_t> & ids) {
        CHECK(rows.size() == ids.size());
        uint32_t crc = 0xffffffffu;
        crc          = crc32c_u32(crc, kDim);
        crc          = crc32c_u32(crc, static_cast<uint32_t>(bit_width));
        crc          = crc32c_u32(crc, bit_width == 4 ? 3 : 2);
        crc          = crc32c_u64(crc, rows.size());
        for (const encoded_row * row : rows) {
            uint32_t bits = 0;
            std::memcpy(&bits, &row->scale, sizeof(bits));
            crc = crc32c_u32(crc, bits);
        }
        for (const encoded_row * row : rows) {
            crc = crc32c_update(crc, row->codes.data(), row->codes.size());
        }
        for (uint64_t id : ids) {
            crc = crc32c_u64(crc, id);
        }
        return crc ^ 0xffffffffu;
    };
    const uint32_t base_crc   = state_crc({ &base_row }, { base_id });
    const uint32_t add_crc    = state_crc({ &base_row, &added_row }, { base_id, added_id });
    const uint32_t remove_crc = state_crc({ &added_row }, { added_id });

    for (int version : { 1, 2, 3, 4 }) {
        std::vector<uint8_t> log = {
            'T', 'V', 'D', 'L', static_cast<uint8_t>(version), static_cast<uint8_t>(bit_width), 0, 0,
        };
        append_u32_le(log, kDim);
        if (version == 4) {
            append_u32_le(log, 0);
            append_wide_state(log, base_wide);
        } else {
            append_u32_le(log, version == 1 ? base_crc : state_token(base_wide));
        }

        std::vector<uint8_t> add_payload;
        append_u64_le(add_payload, added_id);
        if (version <= 2) {
            for (float value : added_vector) {
                append_f32_le(add_payload, value);
            }
        } else {
            append_f32_le(add_payload, added_row.scale);
            add_payload.insert(add_payload.end(), added_row.codes.begin(), added_row.codes.end());
        }
        std::vector<uint8_t> remove_payload;
        append_u64_le(remove_payload, base_id);
        if (version == 4) {
            append_v4_delta_record(log,
                                   /*op=*/1,
                                   /*n=*/1, add_payload, add_wide);
            append_v4_delta_record(log,
                                   /*op=*/2,
                                   /*n=*/1, remove_payload, remove_wide);
        } else {
            append_v1_delta_record(log,
                                   /*op=*/1,
                                   /*n=*/1, add_payload, version == 1 ? add_crc : state_token(add_wide));
            append_v1_delta_record(log,
                                   /*op=*/2,
                                   /*n=*/1, remove_payload, version == 1 ? remove_crc : state_token(remove_wide));
        }

        temp_file delta(".tvid");
        write_bytes(delta.path, log);
        auto * loaded = ggml_vec_index_load_with_delta(snapshot.path.string().c_str(), delta.path.string().c_str());
        CHECK(loaded != nullptr);
        CHECK(ggml_vec_index_len(loaded) == 1);
        CHECK(ggml_vec_index_contains(loaded, base_id) == 0);
        CHECK(ggml_vec_index_contains(loaded, added_id) == 1);
        std::array<float, 1>    scores{};
        std::array<uint64_t, 1> out_ids{};
        CHECK(ggml_vec_index_search(loaded, added_vector.data(), 1, 1, scores.data(), out_ids.data()) ==
              GGML_VEC_INDEX_OK);
        CHECK(out_ids[0] == added_id);
        ggml_vec_index_free(loaded);
    }
}

void check_delta_log_tail_recovery() {
    const std::array<float, kDim> base_vec = {
        1.0f,
        0.0f,
        0.0f,
        0.0f,
    };
    const uint64_t base_id = 9301ULL;

    temp_file snapshot(".tvim");
    temp_file delta(".tvid");
    auto *    base = ggml_vec_index_create(kDim, /*bit_width=*/32);
    CHECK(base != nullptr);
    CHECK(ggml_vec_index_add(base, base_vec.data(), 1, &base_id) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(base, snapshot.path.string().c_str()) == GGML_VEC_INDEX_OK);

    auto * mmap = ggml_vec_index_load_mmap(snapshot.path.string().c_str());
    CHECK(mmap != nullptr);
    CHECK(ggml_vec_index_compact_delta(mmap, snapshot.path.string().c_str(), delta.path.string().c_str()) ==
          GGML_VEC_INDEX_E_INVALID_ARG);
    CHECK(ggml_vec_index_contains(mmap, base_id) == 1);
    ggml_vec_index_free(mmap);

    CHECK(ggml_vec_index_compact_delta(base, snapshot.path.string().c_str(), delta.path.string().c_str()) ==
          GGML_VEC_INDEX_OK);
    {
        temp_file       delta_alias(".tvid.alias");
        std::error_code ec;
        std::filesystem::create_hard_link(delta.path, delta_alias.path, ec);
        if (!ec) {
            CHECK(std::filesystem::equivalent(delta.path, delta_alias.path, ec));
            CHECK(!ec);
            CHECK(ggml_vec_index_compact_delta(base, snapshot.path.string().c_str(), delta.path.string().c_str()) ==
                  GGML_VEC_INDEX_OK);
            CHECK(std::filesystem::equivalent(delta.path, delta_alias.path, ec));
            CHECK(!ec);
            CHECK(read_bytes(delta_alias.path) == read_bytes(delta.path));
            CHECK(read_bytes(delta_alias.path).size() == 48);
            std::filesystem::remove(delta_alias.path, ec);
        } else {
            CHECK(ec == std::errc::operation_not_supported || ec == std::errc::function_not_supported ||
                  ec == std::errc::permission_denied);
        }
    }
    ggml_vec_index_free(base);

#ifndef _WIN32
    std::filesystem::path lock_path = delta.path;
    lock_path += ".lock";
    std::filesystem::permissions(lock_path, std::filesystem::perms::owner_read, std::filesystem::perm_options::replace);
    auto * read_only_lock_loaded =
        ggml_vec_index_load_with_delta(snapshot.path.string().c_str(), delta.path.string().c_str());
    CHECK(read_only_lock_loaded != nullptr);
    ggml_vec_index_free(read_only_lock_loaded);
    std::filesystem::permissions(lock_path, std::filesystem::perms::owner_read | std::filesystem::perms::owner_write,
                                 std::filesystem::perm_options::replace);
#endif

    std::vector<uint8_t> corrupted_delta = read_bytes(delta.path);
    CHECK(corrupted_delta.size() >= 48);
    corrupted_delta.resize(corrupted_delta.size() + 56, 0);
    write_bytes(delta.path, corrupted_delta);

    auto * loaded = ggml_vec_index_load_with_delta(snapshot.path.string().c_str(), delta.path.string().c_str());
    CHECK(loaded != nullptr);
    CHECK(ggml_vec_index_len(loaded) == 1);
    CHECK(ggml_vec_index_contains(loaded, base_id) == 1);
    CHECK(ggml_vec_index_compact_delta(loaded, snapshot.path.string().c_str(), delta.path.string().c_str()) ==
          GGML_VEC_INDEX_OK);
    CHECK(read_bytes(delta.path).size() == 48);
    ggml_vec_index_free(loaded);

    const std::array<float, kDim> added_vec = {
        0.0f,
        1.0f,
        0.0f,
        0.0f,
    };
    const uint64_t added_id = 9302ULL;
    std::vector<uint8_t> add_payload;
    append_u64_le(add_payload, added_id);
    for (float value : added_vec) {
        append_f32_le(add_payload, value);
    }
    std::vector<float> after_add_vectors(base_vec.begin(), base_vec.end());
    after_add_vectors.insert(after_add_vectors.end(), added_vec.begin(), added_vec.end());
    const std::vector<uint64_t> after_add_ids = { base_id, added_id };
    std::vector<uint8_t> add_log = { 'T', 'V', 'D', 'L', 4, 32, 0, 0 };
    append_u32_le(add_log, kDim);
    append_u32_le(add_log, 0);
    append_wide_state(add_log, f32_wide_state(std::vector<float>(base_vec.begin(), base_vec.end()), { base_id }, kDim));
    append_v4_delta_record(add_log,
                           /*op=*/1,
                           /*n=*/1, add_payload, f32_wide_state(after_add_vectors, after_add_ids, kDim));
    add_log.resize(add_log.size() + 56, 0);
    write_bytes(delta.path, add_log);

    loaded = ggml_vec_index_load_with_delta(snapshot.path.string().c_str(), delta.path.string().c_str());
    CHECK(loaded != nullptr);
    CHECK(ggml_vec_index_contains(loaded, base_id) == 1);
    CHECK(ggml_vec_index_contains(loaded, added_id) == 1);
    CHECK(ggml_vec_index_compact_delta(loaded, snapshot.path.string().c_str(), delta.path.string().c_str()) ==
          GGML_VEC_INDEX_OK);
    CHECK(read_bytes(delta.path).size() == 48);
    ggml_vec_index_free(loaded);

    auto * compacted = ggml_vec_index_load(snapshot.path.string().c_str());
    CHECK(compacted != nullptr);
    CHECK(ggml_vec_index_contains(compacted, base_id) == 1);
    CHECK(ggml_vec_index_contains(compacted, added_id) == 1);
    ggml_vec_index_free(compacted);

    write_bytes(delta.path, { 'T', 'V', 'D' });
    loaded = ggml_vec_index_load_with_delta(snapshot.path.string().c_str(), delta.path.string().c_str());
    CHECK(loaded != nullptr);
    CHECK(ggml_vec_index_len(loaded) == 2);
    CHECK(ggml_vec_index_contains(loaded, base_id) == 1);
    CHECK(ggml_vec_index_contains(loaded, added_id) == 1);
    CHECK(ggml_vec_index_compact_delta(loaded, snapshot.path.string().c_str(), delta.path.string().c_str()) ==
          GGML_VEC_INDEX_OK);
    CHECK(read_bytes(delta.path).size() == 48);
    ggml_vec_index_free(loaded);

    std::error_code       ec;
    std::filesystem::path lock_path_to_remove = delta.path;
    lock_path_to_remove += ".lock";
    std::filesystem::remove(lock_path_to_remove, ec);
}

void check_delta_log_rejects_oversized_payload_header() {
    const std::array<float, kDim> base_vec = {
        1.0f,
        0.0f,
        0.0f,
        0.0f,
    };
    const uint64_t base_id = 9351ULL;

    temp_file snapshot(".tvim");
    temp_file delta(".tvid");
    auto *    base = ggml_vec_index_create(kDim, /*bit_width=*/32);
    CHECK(base != nullptr);
    CHECK(ggml_vec_index_add(base, base_vec.data(), 1, &base_id) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(base, snapshot.path.string().c_str()) == GGML_VEC_INDEX_OK);

    std::vector<uint8_t> log = { 'T', 'V', 'D', 'L', 4, 32, 0, 0 };
    append_u32_le(log, kDim);
    append_u32_le(log, 0);
    append_wide_state(log, f32_wide_state(std::vector<float>(base_vec.begin(), base_vec.end()), { base_id }, kDim));

    constexpr uint64_t oversized_payload = 1ull << 20;
    const size_t       record_offset      = log.size();
    log.insert(log.end(), { 1, 0, 0, 0 });
    append_u32_le(log, 1);
    append_u64_le(log, oversized_payload);
    append_u32_le(log, 0);
    append_u32_le(log, 0);
    append_wide_state(log, f32_wide_state(std::vector<float>(base_vec.begin(), base_vec.end()), { base_id }, kDim));
    uint32_t crc = crc32c_update(0xffffffffu, log.data() + record_offset, 16);
    crc          = crc32c_update(crc, log.data() + record_offset + 24, 32);
    std::array<uint8_t, 64 * 1024> zeros{};
    for (uint64_t offset = 0; offset < oversized_payload; offset += zeros.size()) {
        const size_t chunk = static_cast<size_t>(std::min<uint64_t>(zeros.size(), oversized_payload - offset));
        crc = crc32c_update(crc, zeros.data(), chunk);
    }
    put_u32_le(log, record_offset + 16, crc ^ 0xffffffffu);
    write_sparse_bytes(delta.path, log, log.size() + oversized_payload);

    CHECK(ggml_vec_index_compact_delta(base, snapshot.path.string().c_str(), delta.path.string().c_str()) ==
          GGML_VEC_INDEX_E_IO);
    ggml_vec_index_free(base);
    CHECK(ggml_vec_index_load_with_delta(snapshot.path.string().c_str(), delta.path.string().c_str()) == nullptr);

    std::error_code       ec;
    std::filesystem::path lock_path = delta.path;
    lock_path += ".lock";
    std::filesystem::remove(lock_path, ec);
}

void check_delta_log_corrupt_final_header_recovery() {
    const std::array<float, kDim> base_vec = {
        1.0f,
        0.0f,
        0.0f,
        0.0f,
    };
    const std::array<float, kDim> added_vec = {
        0.0f,
        1.0f,
        0.0f,
        0.0f,
    };
    const uint64_t base_id  = 9361ULL;
    const uint64_t added_id = 9362ULL;

    temp_file snapshot(".tvim");
    temp_file delta(".tvid");
    auto *    base = ggml_vec_index_create(kDim, /*bit_width=*/32);
    CHECK(base != nullptr);
    CHECK(ggml_vec_index_add(base, base_vec.data(), 1, &base_id) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(base, snapshot.path.string().c_str()) == GGML_VEC_INDEX_OK);

    std::vector<uint8_t> payload;
    append_u64_le(payload, added_id);
    for (float value : added_vec) {
        append_f32_le(payload, value);
    }
    std::vector<float> after_vectors(base_vec.begin(), base_vec.end());
    after_vectors.insert(after_vectors.end(), added_vec.begin(), added_vec.end());
    const std::vector<uint64_t> after_ids = { base_id, added_id };
    std::vector<uint8_t> good = { 'T', 'V', 'D', 'L', 4, 32, 0, 0 };
    append_u32_le(good, kDim);
    append_u32_le(good, 0);
    append_wide_state(good, f32_wide_state(std::vector<float>(base_vec.begin(), base_vec.end()), { base_id }, kDim));
    const size_t record_offset = good.size();
    append_v4_delta_record(good, /*op=*/1, /*n=*/1, payload, f32_wide_state(after_vectors, after_ids, kDim));

    for (int field : { 0, 1, 2 }) {
        std::vector<uint8_t> corrupt = good;
        if (field == 0) {
            corrupt[record_offset + 4] ^= 1;
        } else if (field == 1) {
            corrupt[record_offset + 8] += 1;
        } else {
            corrupt[record_offset + 8] -= 1;
        }
        write_bytes(delta.path, corrupt);
        auto * loaded = ggml_vec_index_load_with_delta(snapshot.path.string().c_str(), delta.path.string().c_str());
        CHECK(loaded != nullptr);
        CHECK(ggml_vec_index_len(loaded) == 1);
        CHECK(ggml_vec_index_contains(loaded, base_id) == 1);
        CHECK(ggml_vec_index_contains(loaded, added_id) == 0);
        CHECK(ggml_vec_index_compact_delta(loaded, snapshot.path.string().c_str(), delta.path.string().c_str()) ==
              GGML_VEC_INDEX_OK);
        ggml_vec_index_free(loaded);
    }
    ggml_vec_index_free(base);

    std::error_code       ec;
    std::filesystem::path lock_path = delta.path;
    lock_path += ".lock";
    std::filesystem::remove(lock_path, ec);
}

#ifdef GGML_VEC_INDEX_TEST_HOOKS
void check_delta_replay_is_chunked() {
    constexpr int n = 20000;
    const std::array<float, kDim> base_vec = {
        1.0f,
        0.0f,
        0.0f,
        0.0f,
    };
    const uint64_t base_id = 9371ULL;

    temp_file snapshot(".tvim");
    temp_file delta(".tvid");
    auto *    idx = ggml_vec_index_create(kDim, /*bit_width=*/32);
    CHECK(idx != nullptr);
    CHECK(ggml_vec_index_add(idx, base_vec.data(), 1, &base_id) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(idx, snapshot.path.string().c_str()) == GGML_VEC_INDEX_OK);
    ggml_vec_index_free(idx);

    std::vector<float>    vectors(static_cast<size_t>(n) * kDim, 0.0f);
    std::vector<uint64_t> ids(n);
    for (int i = 0; i < n; ++i) {
        vectors[static_cast<size_t>(i) * kDim + static_cast<size_t>(i % kDim)] = 1.0f;
        ids[static_cast<size_t>(i)] = 100000ULL + static_cast<uint64_t>(i);
    }
    std::vector<uint8_t> payload;
    payload.reserve(ids.size() * sizeof(uint64_t) + vectors.size() * sizeof(uint32_t));
    for (uint64_t id : ids) {
        append_u64_le(payload, id);
    }
    for (float value : vectors) {
        append_f32_le(payload, value);
    }
    std::vector<float> all_vectors(base_vec.begin(), base_vec.end());
    all_vectors.insert(all_vectors.end(), vectors.begin(), vectors.end());
    std::vector<uint64_t> all_ids = { base_id };
    all_ids.insert(all_ids.end(), ids.begin(), ids.end());
    std::vector<uint8_t> log = { 'T', 'V', 'D', 'L', 4, 32, 0, 0 };
    append_u32_le(log, kDim);
    append_u32_le(log, 0);
    append_wide_state(log, f32_wide_state(std::vector<float>(base_vec.begin(), base_vec.end()), { base_id }, kDim));
    append_v4_delta_record(log, /*op=*/1, static_cast<uint32_t>(n), payload,
                           f32_wide_state(all_vectors, all_ids, kDim));
    write_bytes(delta.path, log);

    for (int64_t countdown : { int64_t{ 1 }, int64_t{ 2 } }) {
        ggml_vec_index_test_set_oom_countdown(countdown);
        ggml_vec_index_t * failed = nullptr;
        const int status =
            ggml_vec_index_load_with_delta_ex(snapshot.path.string().c_str(), delta.path.string().c_str(), &failed);
        ggml_vec_index_test_set_oom_countdown(-1);
        CHECK(status == GGML_VEC_INDEX_E_OOM);
        CHECK(failed == nullptr);
    }

    ggml_vec_index_test_reset_delta_max_read_size();
    auto * loaded = ggml_vec_index_load_with_delta(snapshot.path.string().c_str(), delta.path.string().c_str());
    CHECK(loaded != nullptr);
    CHECK(ggml_vec_index_len(loaded) == n + 1);
    CHECK(ggml_vec_index_test_get_delta_max_read_size() <= 64 * 1024);
    ggml_vec_index_free(loaded);

    std::error_code       ec;
    std::filesystem::path lock_path = delta.path;
    lock_path += ".lock";
    std::filesystem::remove(lock_path, ec);
}

void check_large_row_delta_replay_is_chunked() {
    {
        constexpr int dim = 20000;
        const uint64_t id = 9381ULL;
        std::vector<float> vector(dim, 0.0f);
        vector[0] = 1.0f;

        temp_file snapshot(".tvim");
        temp_file delta(".tvid");
        auto *    idx = ggml_vec_index_create(dim, /*bit_width=*/32);
        CHECK(idx != nullptr);
        CHECK(ggml_vec_index_write(idx, snapshot.path.string().c_str()) == GGML_VEC_INDEX_OK);
        ggml_vec_index_free(idx);

        std::vector<uint8_t> payload;
        append_u64_le(payload, id);
        for (float value : vector) {
            append_f32_le(payload, value);
        }
        std::vector<uint8_t> log = { 'T', 'V', 'D', 'L', 4, 32, 0, 0 };
        append_u32_le(log, dim);
        append_u32_le(log, 0);
        append_wide_state(log, f32_wide_state({}, {}, dim));
        append_v4_delta_record(log, /*op=*/1, /*n=*/1, payload, f32_wide_state(vector, { id }, dim));
        write_bytes(delta.path, log);

        for (int64_t countdown : { int64_t{ 1 }, int64_t{ 2 } }) {
            ggml_vec_index_test_set_oom_countdown(countdown);
            ggml_vec_index_t * failed = nullptr;
            const int status =
                ggml_vec_index_load_with_delta_ex(snapshot.path.string().c_str(), delta.path.string().c_str(), &failed);
            ggml_vec_index_test_set_oom_countdown(-1);
            CHECK(status == GGML_VEC_INDEX_E_OOM);
            CHECK(failed == nullptr);
        }

        ggml_vec_index_test_reset_delta_max_read_size();
        auto * loaded = ggml_vec_index_load_with_delta(snapshot.path.string().c_str(), delta.path.string().c_str());
        CHECK(loaded != nullptr);
        CHECK(ggml_vec_index_len(loaded) == 1);
        CHECK(ggml_vec_index_contains(loaded, id) == 1);
        CHECK(ggml_vec_index_test_get_delta_max_read_size() <= 64 * 1024);
        ggml_vec_index_free(loaded);

        std::error_code       ec;
        std::filesystem::path lock_path = delta.path;
        lock_path += ".lock";
        std::filesystem::remove(lock_path, ec);
    }

    for (int bit_width : { 4, 8 }) {
        const int      dim = bit_width == 4 ? 140000 : 70000;
        const uint64_t id  = 9390ULL + static_cast<uint64_t>(bit_width);
        std::vector<float> vector(dim, 0.0f);
        for (int i = 0; i < dim; ++i) {
            vector[static_cast<size_t>(i)] = static_cast<float>((i % 15) - 7) / 7.0f;
        }

        temp_file encoded_snapshot(".tvim");
        auto *    encoded = ggml_vec_index_create(dim, bit_width);
        CHECK(encoded != nullptr);
        CHECK(ggml_vec_index_add(encoded, vector.data(), 1, &id) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_write(encoded, encoded_snapshot.path.string().c_str()) == GGML_VEC_INDEX_OK);
        ggml_vec_index_free(encoded);
        const std::vector<uint8_t> encoded_bytes = read_bytes(encoded_snapshot.path);
        const size_t row_bytes = bit_width == 4 ? (static_cast<size_t>(dim) + 1) / 2 : static_cast<size_t>(dim);
        uint32_t     scale_bits = 0;
        for (int i = 0; i < 4; ++i) {
            scale_bits |= static_cast<uint32_t>(encoded_bytes[32 + static_cast<size_t>(i)]) << (8 * i);
        }
        float scale = 0.0f;
        std::memcpy(&scale, &scale_bits, sizeof(scale));
        const std::vector<uint8_t> codes(
            encoded_bytes.begin() + 36, encoded_bytes.begin() + static_cast<std::ptrdiff_t>(36 + row_bytes));
        const auto empty_state = wide_state_from_hashes({});
        const auto row_state = wide_state_from_hashes({ encoded_slot_state_hash(id, scale, codes) });
        const uint32_t empty_crc = encoded_state_crc(dim, bit_width, {}, {}, {});
        const uint32_t row_crc = encoded_state_crc(dim, bit_width, { scale }, { codes }, { id });

        temp_file snapshot(".tvim");
        auto *    empty = ggml_vec_index_create(dim, bit_width);
        CHECK(empty != nullptr);
        CHECK(ggml_vec_index_write(empty, snapshot.path.string().c_str()) == GGML_VEC_INDEX_OK);
        ggml_vec_index_free(empty);

        for (int version : { 1, 2, 3, 4 }) {
            temp_file delta(".tvid");
            std::vector<uint8_t> log = {
                'T', 'V', 'D', 'L', static_cast<uint8_t>(version), static_cast<uint8_t>(bit_width), 0, 0,
            };
            append_u32_le(log, static_cast<uint32_t>(dim));
            if (version == 4) {
                append_u32_le(log, 0);
                append_wide_state(log, empty_state);
            } else {
                append_u32_le(log, version == 1 ? empty_crc : encoded_state_token(empty_state, dim, bit_width));
            }
            const size_t record_offset = log.size();

            std::vector<uint8_t> payload;
            append_u64_le(payload, id);
            if (version <= 2) {
                for (float value : vector) {
                    append_f32_le(payload, value);
                }
            } else {
                append_f32_le(payload, scale);
                payload.insert(payload.end(), codes.begin(), codes.end());
            }
            if (version == 4) {
                append_v4_delta_record(log, /*op=*/1, /*n=*/1, payload, row_state);
            } else {
                append_v1_delta_record(log, /*op=*/1, /*n=*/1, payload,
                                       version == 1 ? row_crc : encoded_state_token(row_state, dim, bit_width));
            }
            write_bytes(delta.path, log);

            if (version == 1) {
                temp_file corrupt_delta(".tvid");
                std::vector<uint8_t> corrupt = log;
                corrupt[record_offset + 8] -= 1;
                write_bytes(corrupt_delta.path, corrupt);
                auto * recovered =
                    ggml_vec_index_load_with_delta(snapshot.path.string().c_str(), corrupt_delta.path.string().c_str());
                CHECK(recovered != nullptr);
                CHECK(ggml_vec_index_len(recovered) == 0);
                ggml_vec_index_free(recovered);
                std::error_code       corrupt_ec;
                std::filesystem::path corrupt_lock = corrupt_delta.path;
                corrupt_lock += ".lock";
                std::filesystem::remove(corrupt_lock, corrupt_ec);
            }

            for (int64_t countdown : { int64_t{ 1 }, int64_t{ 2 } }) {
                ggml_vec_index_test_set_oom_countdown(countdown);
                ggml_vec_index_t * failed = nullptr;
                const int status = ggml_vec_index_load_with_delta_ex(
                    snapshot.path.string().c_str(), delta.path.string().c_str(), &failed);
                ggml_vec_index_test_set_oom_countdown(-1);
                CHECK(status == GGML_VEC_INDEX_E_OOM);
                CHECK(failed == nullptr);
            }

            ggml_vec_index_test_reset_delta_max_read_size();
            auto * loaded =
                ggml_vec_index_load_with_delta(snapshot.path.string().c_str(), delta.path.string().c_str());
            CHECK(loaded != nullptr);
            CHECK(ggml_vec_index_len(loaded) == 1);
            CHECK(ggml_vec_index_contains(loaded, id) == 1);
            CHECK(ggml_vec_index_test_get_delta_max_read_size() <= 64 * 1024);
            ggml_vec_index_free(loaded);

            if (version == 2) {
                for (int rounding_mode : { FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO }) {
                    temp_file rounded_snapshot(".tvim");
                    const int saved_rounding = std::fegetround();
                    CHECK(std::fesetround(rounding_mode) == 0);
                    auto * rounded =
                        ggml_vec_index_load_with_delta(snapshot.path.string().c_str(), delta.path.string().c_str());
                    CHECK(rounded != nullptr);
                    CHECK(std::fegetround() == rounding_mode);
                    CHECK(std::fesetround(saved_rounding) == 0);
                    CHECK(ggml_vec_index_compact_delta(
                              rounded, rounded_snapshot.path.string().c_str(), delta.path.string().c_str()) ==
                          GGML_VEC_INDEX_OK);
                    CHECK(read_bytes(rounded_snapshot.path) == encoded_bytes);
                    ggml_vec_index_free(rounded);
                    write_bytes(delta.path, log);
                }
            }

            std::error_code       ec;
            std::filesystem::path lock_path = delta.path;
            lock_path += ".lock";
            std::filesystem::remove(lock_path, ec);
        }
    }
}

void check_not_durable_status() {
    const std::array<float, kDim> vector = {
        1.0f,
        0.0f,
        0.0f,
        0.0f,
    };
    const uint64_t id = 9401ULL;

    temp_file snapshot(".tvim");
    temp_file delta(".tvid");
    auto *    idx = ggml_vec_index_create(kDim, /*bit_width=*/32);
    CHECK(idx != nullptr);
    CHECK(ggml_vec_index_add(idx, vector.data(), 1, &id) == GGML_VEC_INDEX_OK);

    ggml_vec_index_test_set_parent_fsync_fail(1);
    const int write_status = ggml_vec_index_write(idx, snapshot.path.string().c_str());
    ggml_vec_index_test_set_parent_fsync_fail(0);
    CHECK(write_status == GGML_VEC_INDEX_E_NOT_DURABLE);
    CHECK(std::string(ggml_vec_index_error_to_string(write_status)) == "not durable");

    auto * loaded = ggml_vec_index_load(snapshot.path.string().c_str());
    CHECK(loaded != nullptr);
    CHECK(ggml_vec_index_contains(loaded, id) == 1);
    ggml_vec_index_free(loaded);

    CHECK(ggml_vec_index_compact_delta(idx, snapshot.path.string().c_str(), delta.path.string().c_str()) ==
          GGML_VEC_INDEX_OK);
    ggml_vec_index_test_set_parent_fsync_fail_after(1);
    const int compact_status =
        ggml_vec_index_compact_delta(idx, snapshot.path.string().c_str(), delta.path.string().c_str());
    ggml_vec_index_test_set_parent_fsync_fail_after(-1);
    CHECK(compact_status == GGML_VEC_INDEX_E_PARTIAL_COMPACT);
    ggml_vec_index_free(idx);

    std::error_code       ec;
    std::filesystem::path lock_path = delta.path;
    lock_path += ".lock";
    std::filesystem::remove(lock_path, ec);
}
#endif

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
        CHECK(ggml_vec_index_remove(nullptr, id) == GGML_VEC_INDEX_E_INVALID_ARG);
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
        CHECK((perms & std::filesystem::perms::group_read) == std::filesystem::perms::none);
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
        CHECK(actual == snapshot_v2_f32_bytes(kDim, 3, expected_values, expected_ids));
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
    auto * mapped = ggml_vec_index_load_mmap(path.c_str());
    CHECK(mapped != nullptr);
    std::array<float, 3>    preserved_scores{};
    std::array<uint64_t, 3> preserved_ids{};
    std::array<float, 3>    mapped_scores{};
    std::array<uint64_t, 3> mapped_ids{};
    CHECK(ggml_vec_index_search(
              preserved, seeds[0].data(), 1, 3, preserved_scores.data(), preserved_ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_search(mapped, seeds[0].data(), 1, 3, mapped_scores.data(), mapped_ids.data()) ==
          GGML_VEC_INDEX_OK);
    CHECK(mapped_ids == preserved_ids);
    CHECK(mapped_scores == preserved_scores);
    ggml_vec_index_free(mapped);
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
        std::array<float, 4>    q_scores{};
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
    check_quantized_flt_max_persistence_round_trip(/*bit_width=*/4);
    check_quantized_flt_max_persistence_round_trip(/*bit_width=*/8);
    check_quantized_reconstruction_overflow_rejected();
    check_ivf_partial_probe_routing();
    for (int bit_width : { 32, 8, 4 }) {
        check_filtered_and_ivf_search(bit_width);
    }
    check_ivf_state_not_persisted();
    check_committed_delta_replay();
    check_quantized_committed_delta_replay(/*bit_width=*/4);
    check_quantized_committed_delta_replay(/*bit_width=*/8);
    check_delta_log_tail_recovery();
    check_delta_log_rejects_oversized_payload_header();
    check_delta_log_corrupt_final_header_recovery();
#ifdef GGML_VEC_INDEX_TEST_HOOKS
    check_delta_replay_is_chunked();
    check_large_row_delta_replay_is_chunked();
    check_not_durable_status();
#endif

    // v2 snapshots preserve quantized storage and can be loaded through mmap
    // for read-only search.
    {
        temp_file q4_file(".tvim");
        auto *    q4_idx = ggml_vec_index_create(kDim, /*bit_width=*/4);
        CHECK(q4_idx != nullptr);
        CHECK(ggml_vec_index_add(q4_idx, vecs.data(), static_cast<int>(ids.size()), ids.data()) == GGML_VEC_INDEX_OK);
        std::array<float, 4>    before_scores{};
        std::array<uint64_t, 4> before_ids{};
        CHECK(ggml_vec_index_search(q4_idx, seeds[0].data(), 1, 4, before_scores.data(), before_ids.data()) ==
              GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_write(q4_idx, q4_file.path.string().c_str()) == GGML_VEC_INDEX_OK);

        auto * q4_loaded = ggml_vec_index_load(q4_file.path.string().c_str());
        CHECK(q4_loaded != nullptr);
        CHECK(ggml_vec_index_bit_width(q4_loaded) == 4);

        auto * q4_mmap = ggml_vec_index_load_mmap(q4_file.path.string().c_str());
        CHECK(q4_mmap != nullptr);
        std::array<float, 1>    scores{};
        std::array<uint64_t, 1> out_ids{};
        CHECK(ggml_vec_index_search(q4_mmap, seeds[0].data(), 1, 1, scores.data(), out_ids.data()) ==
              GGML_VEC_INDEX_OK);
        CHECK(out_ids[0] == ids[0]);
        CHECK(ggml_vec_index_add(q4_mmap, seeds[0].data(), 1, &ids[0]) == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_write(q4_mmap, q4_file.path.string().c_str()) == GGML_VEC_INDEX_E_INVALID_ARG);
        {
            temp_file       symlink_file(".tvim.link");
            std::error_code ec;
            std::filesystem::create_symlink(q4_file.path, symlink_file.path, ec);
            if (!ec) {
                auto * symlink_mmap = ggml_vec_index_load_mmap(symlink_file.path.string().c_str());
                CHECK(symlink_mmap != nullptr);
                CHECK(ggml_vec_index_search(
                          symlink_mmap, seeds[0].data(), 1, 1, scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
                CHECK(out_ids[0] == ids[0]);
                ggml_vec_index_free(symlink_mmap);
            } else {
                CHECK(ec == std::errc::operation_not_supported || ec == std::errc::function_not_supported ||
                      ec == std::errc::permission_denied);
            }
        }

        for (auto * handle : { q4_loaded, q4_mmap }) {
            std::array<float, 4>    after_scores{};
            std::array<uint64_t, 4> after_ids{};
            CHECK(ggml_vec_index_search(handle, seeds[0].data(), 1, 4, after_scores.data(), after_ids.data()) ==
                  GGML_VEC_INDEX_OK);
            CHECK(after_ids == before_ids);
            CHECK(after_scores == before_scores);
        }

        const std::vector<uint8_t> good = read_bytes(q4_file.path);
        CHECK(good.size() >= 48);
        const size_t                qparams_offset  = 32;
        const size_t                vectors_offset  = qparams_offset + ids.size() * sizeof(uint32_t);
        const size_t                ids_offset      = vectors_offset + ids.size() * ((kDim + 1) / 2);
        const std::array<size_t, 8> corrupt_offsets = {
            size_t{ 5 },      qparams_offset,   vectors_offset,  ids_offset,
            good.size() - 16, good.size() - 12, good.size() - 8, good.size() - 4,
        };
        for (const size_t offset : corrupt_offsets) {
            CHECK(offset < good.size());
            std::vector<uint8_t> corrupt = good;
            corrupt[offset] ^= 1;
            temp_file corrupt_file(".tvim");
            write_bytes(corrupt_file.path, corrupt);
            CHECK(ggml_vec_index_load(corrupt_file.path.string().c_str()) == nullptr);
            CHECK(ggml_vec_index_load_mmap(corrupt_file.path.string().c_str()) == nullptr);
        }
        {
            std::vector<uint8_t> truncated(good.begin(), good.end() - 4);
            temp_file            truncated_file(".tvim");
            write_bytes(truncated_file.path, truncated);
            CHECK(ggml_vec_index_load(truncated_file.path.string().c_str()) == nullptr);
            CHECK(ggml_vec_index_load_mmap(truncated_file.path.string().c_str()) == nullptr);
        }

        ggml_vec_index_free(q4_mmap);
        ggml_vec_index_free(q4_loaded);
        ggml_vec_index_free(q4_idx);
    }

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
#ifdef GGML_VEC_INDEX_TEST_HOOKS
        ggml_vec_index_test_reset_load_count_reject_count();
#endif
        CHECK(ggml_vec_index_load(oversized_count.path.string().c_str()) == nullptr);
#ifdef GGML_VEC_INDEX_TEST_HOOKS
        CHECK(ggml_vec_index_test_get_load_count_reject_count() == 1);
#endif
    }
    {
        temp_file            oversized_mmap_count(".tvim");
        std::vector<uint8_t> bytes;
        bytes.insert(bytes.end(), { 'T', 'V', 'P', 'I', 2, 32, 1, 1 });
        append_u32_le(bytes, /*dim=*/1);
        const uint32_t n = static_cast<uint32_t>(std::numeric_limits<int>::max()) + 1u;
        append_u32_le(bytes, n);
        append_u32_le(bytes, 0);
        append_u32_le(bytes, 0);
        append_u32_le(bytes, 4);
        append_u32_le(bytes, 0);
        CHECK(bytes.size() == 32);
        write_bytes(oversized_mmap_count.path, bytes);
#ifdef GGML_VEC_INDEX_TEST_HOOKS
        ggml_vec_index_test_reset_mmap_count_reject_count();
#endif
        CHECK(ggml_vec_index_load_mmap(oversized_mmap_count.path.string().c_str()) == nullptr);
#ifdef GGML_VEC_INDEX_TEST_HOOKS
        CHECK(ggml_vec_index_test_get_mmap_count_reject_count() == 1);
#endif
    }
    {
        temp_file oversized_dim(".tvim");
        write_bytes(oversized_dim.path, snapshot_bytes(static_cast<uint32_t>(std::numeric_limits<int>::max()) + 1u,
                                                       /*n=*/0, {}, {}));
        CHECK(ggml_vec_index_load(oversized_dim.path.string().c_str()) == nullptr);
    }
    {
        temp_file            empty_legacy_q8(".tvim");
        std::vector<uint8_t> bytes = snapshot_bytes(
            /*dim=*/kDim,
            /*n=*/0, {}, {});
        bytes[5] = 8;
        write_bytes(empty_legacy_q8.path, bytes);
        ggml_vec_index_t * mapped_legacy = nullptr;
        CHECK(ggml_vec_index_load_mmap_ex(empty_legacy_q8.path.string().c_str(), &mapped_legacy) ==
              GGML_VEC_INDEX_E_BAD_VERSION);
        CHECK(mapped_legacy == nullptr);
        auto * loaded_empty_q8 = ggml_vec_index_load(empty_legacy_q8.path.string().c_str());
        CHECK(loaded_empty_q8 != nullptr);
        CHECK(ggml_vec_index_bit_width(loaded_empty_q8) == 8);
        CHECK(ggml_vec_index_len(loaded_empty_q8) == 0);
        ggml_vec_index_free(loaded_empty_q8);
    }
    {
        constexpr uint32_t dim = 20000;
        temp_file             large_legacy_q8(".tvim");
        std::vector<float>    values(dim, 0.0f);
        values[0] = 1.0f;
        std::vector<uint8_t> bytes = snapshot_bytes(dim, /*n=*/1, values, { 8123ULL });
        bytes[5] = 8;
        write_bytes(large_legacy_q8.path, bytes);
#ifdef GGML_VEC_INDEX_TEST_HOOKS
        ggml_vec_index_test_set_oom_countdown(1);
        ggml_vec_index_t * failed = nullptr;
        const int oom_status = ggml_vec_index_load_ex(large_legacy_q8.path.string().c_str(), &failed);
        ggml_vec_index_test_set_oom_countdown(-1);
        CHECK(oom_status == GGML_VEC_INDEX_E_OOM);
        CHECK(failed == nullptr);
#endif
        auto * loaded_large_q8 = ggml_vec_index_load(large_legacy_q8.path.string().c_str());
        CHECK(loaded_large_q8 != nullptr);
        CHECK(ggml_vec_index_dim(loaded_large_q8) == static_cast<int>(dim));
        CHECK(ggml_vec_index_len(loaded_large_q8) == 1);
        CHECK(ggml_vec_index_contains(loaded_large_q8, 8123ULL) == 1);
        ggml_vec_index_free(loaded_large_q8);
    }
    {
        temp_file            large_empty_legacy_q8(".tvim");
        std::vector<uint8_t> bytes = snapshot_bytes(static_cast<uint32_t>(std::numeric_limits<int>::max()),
                                                    /*n=*/0, {}, {});
        bytes[5] = 8;
        write_bytes(large_empty_legacy_q8.path, bytes);
        auto * loaded_large_empty_q8 = ggml_vec_index_load(large_empty_legacy_q8.path.string().c_str());
        CHECK(loaded_large_empty_q8 != nullptr);
        CHECK(ggml_vec_index_dim(loaded_large_empty_q8) == std::numeric_limits<int>::max());
        CHECK(ggml_vec_index_len(loaded_large_empty_q8) == 0);
        ggml_vec_index_free(loaded_large_empty_q8);
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
        auto * legacy_q8 = ggml_vec_index_load(bad_bit_width.path.string().c_str());
        CHECK(legacy_q8 != nullptr);
        CHECK(ggml_vec_index_bit_width(legacy_q8) == 8);
        std::array<float, 1>    scores{};
        std::array<uint64_t, 1> out_ids{};
        CHECK(ggml_vec_index_search(legacy_q8, seeds[0].data(), 1, 1, scores.data(), out_ids.data()) ==
              GGML_VEC_INDEX_OK);
        CHECK(out_ids[0] == 123ULL);
        CHECK(scores[0] > 0.95f);
        ggml_vec_index_free(legacy_q8);
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
