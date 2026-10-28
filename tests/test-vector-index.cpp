// test-vector-index.cpp - standalone C-API smoke test for the vector
// index. Exercises lifecycle, add, search, remove, contains, write, load,
// search-after-load, quantized storage, filtered/prepared-filter search, and
// IVF-flat search. No model, no llama; only the new ggml-vector-index public
// C API.

#include "ggml-vector-index.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cerrno>
#include <atomic>
#include <bitset>
#include <cfloat>
#include <chrono>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <new>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

#ifndef _WIN32
#include <sys/stat.h>
#endif

#ifdef GGML_VEC_INDEX_TEST_HOOKS
// The largest single operator new request while tracking is on, to bound
// what a snapshot writer holds at once.
static std::atomic<bool> g_track_largest_alloc{ false };
static std::atomic<size_t> g_largest_alloc{ 0 };
// When positive, the operator new call that brings it to zero throws, so a
// test can fail every allocation an API call makes, one at a time.
static std::atomic<long> g_alloc_fail_countdown{ 0 };

// GCC can inline a replacement operator delete into its callers and then
// flags its free() of operator new's memory (-Wmismatched-new-delete),
// although these replacements pair malloc with free.
#if defined(__GNUC__) && !defined(__clang__)
#define GGML_VI_TEST_NOINLINE __attribute__((noinline))
#if __GNUC__ >= 11
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif
#else
#define GGML_VI_TEST_NOINLINE
#endif

void * operator new(size_t size) {
    if (g_alloc_fail_countdown.load(std::memory_order_relaxed) > 0 && g_alloc_fail_countdown.fetch_sub(1) == 1) {
        throw std::bad_alloc();
    }
    if (g_track_largest_alloc.load(std::memory_order_relaxed)) {
        size_t seen = g_largest_alloc.load(std::memory_order_relaxed);
        while (size > seen && !g_largest_alloc.compare_exchange_weak(seen, size)) {
        }
    }
    if (void * p = std::malloc(size == 0 ? 1 : size)) {
        return p;
    }
    throw std::bad_alloc();
}

GGML_VI_TEST_NOINLINE void operator delete(void * p) noexcept {
    std::free(p);
}

GGML_VI_TEST_NOINLINE void operator delete(void * p, size_t) noexcept {
    std::free(p);
}

// The other forms, so every allocation the program makes pairs malloc with
// free: a sanitizer's own nothrow or array operator new would otherwise
// hand our free memory it allocated differently.
void * operator new[](size_t size) {
    return ::operator new(size);
}

void * operator new(size_t size, const std::nothrow_t &) noexcept {
    try {
        return ::operator new(size);
    } catch (...) {
        return nullptr;
    }
}

void * operator new[](size_t size, const std::nothrow_t &) noexcept {
    try {
        return ::operator new(size);
    } catch (...) {
        return nullptr;
    }
}

GGML_VI_TEST_NOINLINE void operator delete[](void * p) noexcept {
    std::free(p);
}

GGML_VI_TEST_NOINLINE void operator delete[](void * p, size_t) noexcept {
    std::free(p);
}

GGML_VI_TEST_NOINLINE void operator delete(void * p, const std::nothrow_t &) noexcept {
    std::free(p);
}

GGML_VI_TEST_NOINLINE void operator delete[](void * p, const std::nothrow_t &) noexcept {
    std::free(p);
}

#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 11
#pragma GCC diagnostic pop
#endif

extern "C" {
int     ggml_vec_index_test_can_address_array(size_t count, size_t element_size);
void    ggml_vec_index_test_set_write_fail_after(int64_t bytes);
void    ggml_vec_index_test_set_oom_countdown(int64_t countdown);
void    ggml_vec_index_test_set_parent_fsync_fail(int fail);
void    ggml_vec_index_test_set_data_fsync_fail(int fail);
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
int snapshot_write_v1_preflight(size_t n, size_t dim);
uint64_t turbovec_rotation_hash_for_test(int dim);
size_t turbovec_rotation_cache_bytes_for_test(void);
uint64_t turbovec_query_rotation_hash_for_test(
    const float * queries,
    int n_queries,
    int dim);
double turbovec_query_rotation_max_abs_diff_for_test(
    const float * queries,
    int n_queries,
    int dim);
uint64_t turbovec_lut_hash_for_test(
    const float * query,
    const float * tqplus_shift,
    const float * tqplus_scale,
    int bits,
    int n_queries,
    int dim,
    uint32_t * lut_scale_bits,
    uint32_t * lut_bias_bits);
uint64_t turbovec_codebook_hash_for_test(int bits, int dim);
uint64_t turbovec_blocked_hash_for_test(const ggml_vec_index_t * idx);
void turbovec_clear_blocked_for_test(ggml_vec_index_t * idx);
void turbovec_code_bytes_for_test(
    const ggml_vec_index_t * idx, size_t * packed, size_t * blocked, size_t * sign);
int turbovec_avx2_available_for_test();
int turbovec_avx2_lut_block_matches_scalar_for_test(int bits, int dim);
void turbovec_reset_block_score_call_count_for_test(void);
int64_t turbovec_block_score_call_count_for_test(void);
uint64_t turbovec_ziggurat_table_hash_for_test(void);
void turbovec_hadamard_rotate_for_test(const float * src, float * dst, int dim);
int turbovec_rotation_kernel_parity_for_test(int kernel);
int turbovec_encode_kernel_parity_for_test(int kernel);
int turbovec_layout_kernel_parity_for_test(int kernel);
int vector_validation_kernel_parity_for_test(int kernel);
int crc32c_kernel_parity_for_test(int kernel);
void turbovec_force_vector_major_for_test(int mode);
void turbovec_disable_dot_product_for_test(int disabled);
void turbovec_force_vm8_for_test(int mode);
void turbovec_disable_i8mm_for_test(int disabled);
void turbovec_force_vm8_2bit_for_test(int mode);
void turbovec_force_batches_for_test(int enabled);
int turbovec_layout_for_test(const ggml_vec_index_t * idx);
bool turbovec_planes_for(int bits, int dim);
void turbovec_force_planes_for_test(int mode, size_t min_vectors);
size_t parallel_pool_threads_for_test(void);
void state_hash_for_test(const ggml_vec_index_t * idx, uint64_t * sums);
int parallel_pool_selftest_for_test(int n_threads);
int32_t turbovec_low_planes_for_test(const float * q, int dim, float m, float * unit, std::vector<uint64_t> * masks);
int turbovec_sync_plan_for_test(
    ggml_vec_index_t * idx,
    const char * path,
    std::vector<std::vector<std::pair<uint64_t, std::vector<uint8_t>>>> & batches);
int turbovec_index_hashes_for_test(
    const ggml_vec_index_t * idx,
    uint64_t * codes_hash,
    uint64_t * scales_hash,
    uint64_t * shift_hash,
    uint64_t * scale_hash);
void turbovec_codebook_values(
    int bits,
    int dim,
    bool legacy,
    std::vector<float> & boundaries,
    std::vector<float> & centroids);
double turbovec_ziggurat_x_for_test(int index);
double turbovec_ziggurat_f_for_test(int index);
double turbovec_regularized_beta_for_test(double x, double a, double b);
double turbovec_inverse_regularized_beta_for_test(double probability, double a);
#endif

namespace {

constexpr int kDim = 4;
constexpr uint32_t kFloatParityMaxUlpDiff = 4;
#ifdef GGML_VEC_INDEX_TEST_HOOKS
// Windows x64 static builds have shown 13 ULP of final-score drift; keep a
// small margin while also bounding relative and absolute error.
constexpr uint32_t kTqplusScoreMaxUlpDiff = 16;
constexpr float kTqplusScoreRelTolerance = 2e-6f;
constexpr float kTqplusScoreAbsTolerance = 8e-5f;

#if defined(__x86_64__) || defined(_M_X64)
constexpr bool kTurboVecInterleavedBlockLayout = true;
#else
constexpr bool kTurboVecInterleavedBlockLayout = false;
#endif
#endif
constexpr bool kTurboVecSupported = sizeof(size_t) >= 8;
constexpr int kTurboVecMaxTestDim = 16384;

#include "turbovec-golden-q2.inc"
#include "turbovec-golden-q4.inc"
#include "turbovec-golden-dim256-q2.inc"
#include "turbovec-golden-dim256-q4.inc"
#include "turbovec-1x-golden.inc"
#include "turbovec-1x-golden-vm.inc"
#include "turbovec-1x-golden-staged.inc"

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);\
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

uint32_t bit_ulp_diff(uint32_t actual, uint32_t expected);
ggml_vec_index_t * create_legacy_turbovec(int dim, int bit_width);
void check_bits_within_ulp(
        const char * label,
        uint32_t actual,
        uint32_t expected,
        uint32_t max_ulp);
#ifdef GGML_VEC_INDEX_TEST_HOOKS
void check_float_bits_close(
        const char * label,
        uint32_t actual_bits,
        uint32_t expected_bits,
        uint32_t max_ulp,
        float rel_tolerance,
        float abs_tolerance);
#endif
void check_double_close(
        const char * label,
        double actual,
        double expected,
        double tolerance);
void check_u64_equal(
        const char * label,
        uint64_t actual,
        uint64_t expected);

std::vector<float> normalize(std::vector<float> v) {
    double sumsq = 0.0;
    for (float x : v) sumsq += static_cast<double>(x) * x;
    const float n = static_cast<float>(std::sqrt(sumsq));
    if (n > 0.0f) for (float & x : v) x /= n;
    return v;
}

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

float q8_dot_reference(const std::vector<float> & vector, const std::vector<float> & query) {
    CHECK(vector.size() == query.size());

    float max_abs = 0.0f;
    for (float value : vector) {
        max_abs = std::max(max_abs, std::fabs(value));
    }
    const float scale = max_abs == 0.0f ? 1.0f : max_abs / 127.0f;

    float acc = 0.0f;
    for (size_t i = 0; i < vector.size(); ++i) {
        int code = max_abs == 0.0f ?
            0 : round_nearest_even(vector[i] / scale);
        code = std::max(-127, std::min(127, code));
        acc += query[i] * (static_cast<float>(code) * scale);
    }
    return acc;
}

float q4_dot_reference(const std::vector<float> & vector, const std::vector<float> & query) {
    CHECK(vector.size() == query.size());

    float max_abs = 0.0f;
    for (float value : vector) {
        max_abs = std::max(max_abs, std::fabs(value));
    }
    const float scale = max_abs == 0.0f ? 1.0f : max_abs / 7.0f;

    float acc = 0.0f;
    for (size_t i = 0; i < vector.size(); ++i) {
        int code = max_abs == 0.0f ?
            0 : round_nearest_even(vector[i] / scale);
        code = std::max(-7, std::min(7, code));
        acc += query[i] * (static_cast<float>(code) * scale);
    }
    return acc;
}

uint8_t read_file_byte(const std::string & path, std::streamoff offset) {
    std::ifstream f(path, std::ios::binary);
    CHECK(f.is_open());
    f.seekg(offset);
    char c = 0;
    f.read(&c, 1);
    CHECK(f.good());
    return static_cast<uint8_t>(c);
}

std::vector<uint8_t> read_file_bytes(const std::string & path) {
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

void write_file_bytes(const std::string & path, const std::vector<uint8_t> & bytes) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    CHECK(f.is_open());
    if (!bytes.empty()) {
        f.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    CHECK(static_cast<bool>(f));
}

uint32_t read_u32_le_from(const uint8_t * bytes) {
    return static_cast<uint32_t>(bytes[0]) |
        (static_cast<uint32_t>(bytes[1]) << 8) |
        (static_cast<uint32_t>(bytes[2]) << 16) |
        (static_cast<uint32_t>(bytes[3]) << 24);
}

std::vector<uint8_t> bytes_to_vector(const uint8_t * bytes, size_t len) {
    return std::vector<uint8_t>(bytes, bytes + len);
}

bool any_score_differs(const float * lhs, const float * rhs, size_t n, float eps) {
    for (size_t i = 0; i < n; ++i) {
        if (std::fabs(lhs[i] - rhs[i]) > eps) {
            return true;
        }
    }
    return false;
}

std::filesystem::path make_test_temp_dir() {
#ifdef _WIN32
    const int pid = _getpid();
#else
    const int pid = static_cast<int>(getpid());
#endif
    return std::filesystem::temp_directory_path() /
        ("ggml-vector-index-test-" + std::to_string(pid));
}

void set_test_temp_dir(const std::filesystem::path & path) {
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
    const std::string value = path.string();
#ifdef _WIN32
    _putenv_s("TMP", value.c_str());
    _putenv_s("TEMP", value.c_str());
#else
    setenv("TMPDIR", value.c_str(), 1);
#endif
}

struct temp_file {
    explicit temp_file(const char * suffix) {
        const auto base = std::filesystem::temp_directory_path() /
            ("ggml-vector-index-" + std::to_string(++counter) + suffix);
        path = base;
        std::filesystem::remove(path);
        std::filesystem::remove(path.string() + ".lock");
    }

    ~temp_file() {
        std::error_code ec;
        std::filesystem::remove(path, ec);
        std::filesystem::remove(path.string() + ".lock", ec);
    }

    std::filesystem::path path;

private:
    static inline std::atomic<uint64_t> counter{ 0 };
};

void check_rust_tv_shape(
        const uint8_t * bytes,
        size_t len,
        int bit_width,
        int dim,
        int n,
        int packed_bytes,
        int calib_count) {
    CHECK(len >= 14);
    CHECK(bytes[0] == 'T' && bytes[1] == 'V' && bytes[2] == 'P' && bytes[3] == 'I');
    CHECK(bytes[4] == 3);
    CHECK(bytes[5] == static_cast<uint8_t>(bit_width));
    CHECK(read_u32_le_from(bytes + 6) == static_cast<uint32_t>(dim));
    CHECK(read_u32_le_from(bytes + 10) == static_cast<uint32_t>(n));
    CHECK(14 + static_cast<size_t>(packed_bytes) + static_cast<size_t>(n) * 4 + 4 <= len);
    const size_t calib_offset = 14 + static_cast<size_t>(packed_bytes) + static_cast<size_t>(n) * 4;
    CHECK(read_u32_le_from(bytes + calib_offset) == static_cast<uint32_t>(calib_count));
}

void check_rust_persistence_parity(
        ggml_vec_index_t * idx,
        const char * suffix,
        int bit_width,
        int storage_kind,
        int n,
        const float * rust_scales,
        size_t rust_scale_count,
        const uint8_t * rust_tv,
        size_t rust_tv_len,
        const uint8_t * rust_codes,
        size_t rust_codes_len) {
    const std::string rust_path =
        (std::filesystem::temp_directory_path() /
         ("ggml-vector-index-rust-" + std::string(suffix) + ".tv")).string();
    const std::string qvac_path =
        (std::filesystem::temp_directory_path() /
         ("ggml-vector-index-qvac-" + std::string(suffix) + ".tvim")).string();
    std::filesystem::remove(rust_path);
    std::filesystem::remove(qvac_path);

    write_file_bytes(rust_path, bytes_to_vector(rust_tv, rust_tv_len));
    ggml_vec_index_t * rust_loaded = nullptr;
    CHECK(ggml_vec_index_load_ex(rust_path.c_str(), &rust_loaded) == GGML_VEC_INDEX_E_IO);
    CHECK(rust_loaded == nullptr);
    CHECK(ggml_vec_index_load(rust_path.c_str()) == nullptr);

    CHECK(ggml_vec_index_write(idx, qvac_path.c_str()) == GGML_VEC_INDEX_OK);
    const std::vector<uint8_t> qvac = read_file_bytes(qvac_path);
    CHECK(qvac.size() >= 32);
    CHECK(qvac[0] == 'T' && qvac[1] == 'V' && qvac[2] == 'P' && qvac[3] == 'I');
    CHECK(qvac[4] == 3);
    CHECK(qvac[5] == static_cast<uint8_t>(bit_width));
    CHECK(qvac[6] == static_cast<uint8_t>(storage_kind));

    const size_t qparam_bytes = read_u32_le_from(qvac.data() + 20);
    const size_t calibration_bytes = read_u32_le_from(qvac.data() + 28);
    const size_t dim = read_u32_le_from(qvac.data() + 8);
    CHECK(qparam_bytes == sizeof(float));
    CHECK(calibration_bytes == 2 * dim * sizeof(float));
    CHECK(rust_scale_count == static_cast<size_t>(n));
    for (size_t i = 0; i < rust_scale_count; ++i) {
        uint32_t expected = 0;
        std::memcpy(&expected, rust_scales + i, sizeof(expected));
        const uint32_t actual = read_u32_le_from(qvac.data() + 32 + i * sizeof(float));
        const uint32_t ulp_diff = bit_ulp_diff(actual, expected);
        if (ulp_diff > kFloatParityMaxUlpDiff) {
            std::fprintf(
                stderr,
                "FAIL rust scale[%zu]: actual=0x%08x expected=0x%08x ulp=%u max=%u\n",
                i,
                actual,
                expected,
                ulp_diff,
                kFloatParityMaxUlpDiff);
        }
        CHECK(ulp_diff <= kFloatParityMaxUlpDiff);
    }
    const size_t vector_offset =
        32 + static_cast<size_t>(n) * qparam_bytes + calibration_bytes;
    CHECK(qvac.size() >= vector_offset + rust_codes_len);
    const bool same_codes = std::equal(
        rust_codes,
        rust_codes + rust_codes_len,
        qvac.data() + vector_offset);
    CHECK(same_codes);
    CHECK(qvac != bytes_to_vector(rust_tv, rust_tv_len));

    std::filesystem::remove(rust_path);
    std::filesystem::remove(qvac_path);
}

void check_turbovec_rust_golden(
        const char * suffix,
        int bits,
        int storage_kind,
        int dim,
        int n_db,
        int n_query,
        int k,
        uint64_t rust_rotation_hash,
        const float * db,
        const float * queries,
        const float * rust_scores,
        const float * rust_scales,
        size_t rust_scale_count,
        const uint8_t * rust_tv_bytes,
        size_t rust_tv_bytes_len,
        const uint8_t * rust_packed_codes,
        size_t rust_packed_bytes,
        int rust_calib_count,
        const int * topk) {
#ifdef GGML_VEC_INDEX_TEST_HOOKS
    CHECK(turbovec_rotation_hash_for_test(dim) == rust_rotation_hash);
#else
    (void) rust_rotation_hash;
#endif
    auto * tv = create_legacy_turbovec(dim, bits);
    std::vector<uint64_t> ids(static_cast<size_t>(n_db));
    for (int i = 0; i < n_db; ++i) {
        ids[static_cast<size_t>(i)] = static_cast<uint64_t>(i);
    }
    CHECK(ggml_vec_index_add(tv, db, n_db, ids.data()) == GGML_VEC_INDEX_OK);

    std::vector<float> scores(static_cast<size_t>(n_query) * static_cast<size_t>(k));
    std::vector<uint64_t> out(static_cast<size_t>(n_query) * static_cast<size_t>(k));
    CHECK(ggml_vec_index_search(
        tv,
        queries,
        n_query,
        k,
        scores.data(),
        out.data()) == GGML_VEC_INDEX_OK);
    CHECK(!any_score_differs(scores.data(), rust_scores, scores.size(), 1e-5f));

    check_rust_tv_shape(
        rust_tv_bytes,
        rust_tv_bytes_len,
        bits,
        dim,
        n_db,
        static_cast<int>(rust_packed_bytes),
        rust_calib_count);
    check_rust_persistence_parity(
        tv,
        suffix,
        bits,
        storage_kind,
        n_db,
        rust_scales,
        rust_scale_count,
        rust_tv_bytes,
        rust_tv_bytes_len,
        rust_packed_codes,
        rust_packed_bytes);
    for (int i = 0; i < n_query * k; ++i) {
        CHECK(out[static_cast<size_t>(i)] == static_cast<uint64_t>(topk[i]));
    }
    ggml_vec_index_free(tv);
}

void append_file_bytes(const std::string & path, const std::vector<uint8_t> & bytes) {
    std::ofstream f(path, std::ios::binary | std::ios::app);
    CHECK(f.is_open());
    if (!bytes.empty()) {
        f.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    CHECK(static_cast<bool>(f));
}

bool has_snapshot_tmp(const std::filesystem::path & path) {
    const std::filesystem::path parent = path.parent_path();
    const std::string prefix = path.filename().string() + ".tmp.";
    std::error_code ec;
    for (const auto & entry : std::filesystem::directory_iterator(parent, ec)) {
        const std::string name = entry.path().filename().string();
        if (name.compare(0, prefix.size(), prefix) == 0) {
            return true;
        }
    }
    return false;
}

// Windows file APIs reject paths past MAX_PATH unless given in verbatim
// form (a \\?\ prefix); tests that use file names near NAME_MAX build
// their paths that way.
std::filesystem::path verbatim_path(const std::filesystem::path & path) {
#ifdef _WIN32
    const std::wstring full = std::filesystem::absolute(path).wstring();
    return full.rfind(L"\\\\?\\", 0) == 0 ? path : std::filesystem::path(L"\\\\?\\" + full);
#else
    return path;
#endif
}

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

uint64_t read_u64_le_at(const std::vector<uint8_t> & bytes, size_t offset) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(bytes[offset + static_cast<size_t>(i)]) << (8 * i);
    }
    return value;
}

void write_u32_le_at(std::vector<uint8_t> & bytes, size_t offset, uint32_t value) {
    for (int i = 0; i < 4; ++i) {
        bytes[offset + static_cast<size_t>(i)] = static_cast<uint8_t>(value >> (8 * i));
    }
}

void write_u64_le_at(std::vector<uint8_t> & bytes, size_t offset, uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        bytes[offset + static_cast<size_t>(i)] = static_cast<uint8_t>(value >> (8 * i));
    }
}

void append_f32_le(std::vector<uint8_t> & bytes, float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    append_u32_le(bytes, bits);
}

uint32_t crc32c_update(uint32_t crc, const void * data, size_t size) {
    const uint8_t * p = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < size; ++i) {
        crc ^= p[i];
        for (int bit = 0; bit < 8; ++bit) {
            const uint32_t mask = 0u - (crc & 1u);
            crc = (crc >> 1) ^ (0x82f63b78u & mask);
        }
    }
    return crc;
}

uint32_t crc32c_update_u32(uint32_t crc, uint32_t value) {
    uint8_t bytes[4];
    for (int i = 0; i < 4; ++i) {
        bytes[i] = static_cast<uint8_t>(value >> (8 * i));
    }
    return crc32c_update(crc, bytes, sizeof(bytes));
}

uint32_t crc32c_update_u64(uint32_t crc, uint64_t value) {
    uint8_t bytes[8];
    for (int i = 0; i < 8; ++i) {
        bytes[i] = static_cast<uint8_t>(value >> (8 * i));
    }
    return crc32c_update(crc, bytes, sizeof(bytes));
}

bool delta_log_is_v4(const std::vector<uint8_t> & bytes) {
    return bytes.size() > 4 && bytes[4] == 4;
}

size_t delta_log_header_size(const std::vector<uint8_t> & bytes) {
    return delta_log_is_v4(bytes) ? 48 : 16;
}

size_t delta_record_header_size(const std::vector<uint8_t> & bytes) {
    return delta_log_is_v4(bytes) ? 56 : 24;
}

size_t delta_record_state_offset(const std::vector<uint8_t> & bytes, size_t record_offset) {
    return record_offset + (delta_log_is_v4(bytes) ? 24 : 20);
}

size_t delta_record_payload_offset(const std::vector<uint8_t> & bytes, size_t record_offset) {
    return record_offset + delta_record_header_size(bytes);
}

void refresh_delta_record_crc(std::vector<uint8_t> & bytes, size_t record_offset) {
    const uint64_t payload_bytes = read_u64_le_at(bytes, record_offset + 8);
    uint32_t crc = crc32c_update(0xffffffffu, bytes.data() + record_offset, 16);
    if (delta_log_is_v4(bytes)) {
        crc = crc32c_update(crc, bytes.data() + record_offset + 24, 32);
    } else {
        crc = crc32c_update(crc, bytes.data() + record_offset + 20, 4);
    }
    if (payload_bytes != 0) {
        crc = crc32c_update(
            crc,
            bytes.data() + delta_record_payload_offset(bytes, record_offset),
            static_cast<size_t>(payload_bytes));
    }
    write_u32_le_at(bytes, record_offset + 16, crc ^ 0xffffffffu);
}

uint64_t rotl64(uint64_t value, int shift) {
    return (value << shift) | (value >> (64 - shift));
}

uint32_t float_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

#ifdef GGML_VEC_INDEX_TEST_HOOKS
float float_from_bits(uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}
#endif

uint32_t bit_ulp_diff(uint32_t actual, uint32_t expected) {
    return actual > expected ? actual - expected : expected - actual;
}

void check_bits_within_ulp(
        const char * label,
        uint32_t actual,
        uint32_t expected,
        uint32_t max_ulp) {
    const uint32_t diff = bit_ulp_diff(actual, expected);
    if (diff > max_ulp) {
        std::fprintf(
            stderr,
            "FAIL %s: actual=0x%08x expected=0x%08x ulp=%u max=%u\n",
            label,
            actual,
            expected,
            diff,
            max_ulp);
    }
    CHECK(diff <= max_ulp);
}

#ifdef GGML_VEC_INDEX_TEST_HOOKS
void check_float_bits_close(
        const char * label,
        uint32_t actual_bits,
        uint32_t expected_bits,
        uint32_t max_ulp,
        float rel_tolerance,
        float abs_tolerance) {
    const uint32_t ulp_diff = bit_ulp_diff(actual_bits, expected_bits);
    const float actual = float_from_bits(actual_bits);
    const float expected = float_from_bits(expected_bits);
    const float abs_diff = std::fabs(actual - expected);
    const float tolerance =
        std::max(abs_tolerance, rel_tolerance * std::max(std::fabs(actual), std::fabs(expected)));
    const bool close =
        ulp_diff <= max_ulp ||
        (std::isfinite(actual) && std::isfinite(expected) && abs_diff <= tolerance);
    if (!close) {
        std::fprintf(
            stderr,
            "FAIL %s: actual=0x%08x expected=0x%08x ulp=%u max=%u abs=%g tolerance=%g\n",
            label,
            actual_bits,
            expected_bits,
            ulp_diff,
            max_ulp,
            static_cast<double>(abs_diff),
            static_cast<double>(tolerance));
    }
    CHECK(close);
}
#endif

void check_double_close(
        const char * label,
        double actual,
        double expected,
        double tolerance) {
    const double diff = std::fabs(actual - expected);
    if (!(diff <= tolerance)) {
        std::fprintf(
            stderr,
            "FAIL %s: actual=%.17g expected=%.17g diff=%.17g max=%.17g\n",
            label,
            actual,
            expected,
            diff,
            tolerance);
    }
    CHECK(diff <= tolerance);
}

void check_u64_equal(
        const char * label,
        uint64_t actual,
        uint64_t expected) {
    if (actual != expected) {
        std::fprintf(
            stderr,
            "FAIL %s: actual=0x%016llx expected=0x%016llx\n",
            label,
            static_cast<unsigned long long>(actual),
            static_cast<unsigned long long>(expected));
    }
    CHECK(actual == expected);
}

uint64_t slot_state_hash_f32(uint64_t id, const std::vector<float> & vector) {
    uint32_t crc0 = 0xffffffffu;
    uint32_t crc1 = 0x82f63b78u;
    crc0 = crc32c_update_u64(crc0, id);
    crc1 = crc32c_update_u64(crc1, id ^ 0xa5a5a5a5a5a5a5a5ull);
    for (float value : vector) {
        const uint32_t bits = float_bits(value);
        crc0 = crc32c_update_u32(crc0, bits);
        crc1 = crc32c_update_u32(crc1, bits ^ 0xa5a5a5a5u);
    }
    return (static_cast<uint64_t>(crc0 ^ 0xffffffffu) << 32) |
        static_cast<uint64_t>(crc1 ^ 0xffffffffu);
}

void quantize_for_state_hash(
        int bit_width,
        const std::vector<float> & vector,
        std::vector<uint8_t> & codes,
        float & scale) {
    float max_abs = 0.0f;
    for (float value : vector) {
        max_abs = std::max(max_abs, std::fabs(value));
    }

    if (bit_width == 8) {
        codes.assign(vector.size(), 0);
        if (max_abs == 0.0f) {
            scale = 1.0f;
            return;
        }
        scale = max_abs / 127.0f;
        for (size_t i = 0; i < vector.size(); ++i) {
            int q = round_nearest_even(vector[i] / scale);
            q = std::max(-127, std::min(127, q));
            codes[i] = static_cast<uint8_t>(static_cast<int8_t>(q));
        }
        return;
    }

    codes.assign((vector.size() + 1) / 2, 0x88);
    if (max_abs == 0.0f) {
        scale = 1.0f;
        return;
    }
    scale = max_abs / 7.0f;
    for (size_t i = 0; i < vector.size(); ++i) {
        int q = round_nearest_even(vector[i] / scale);
        q = std::max(-7, std::min(7, q));
        const uint8_t code = static_cast<uint8_t>(q + 8);
        uint8_t & byte = codes[i / 2];
        if ((i & 1) == 0) {
            byte = static_cast<uint8_t>((byte & 0xf0u) | code);
        } else {
            byte = static_cast<uint8_t>((byte & 0x0fu) | (code << 4));
        }
    }
}

uint64_t slot_state_hash_quantized(
        int bit_width,
        uint64_t id,
        const std::vector<float> & vector) {
    std::vector<uint8_t> codes;
    float scale = 1.0f;
    quantize_for_state_hash(bit_width, vector, codes, scale);

    uint32_t crc0 = 0xffffffffu;
    uint32_t crc1 = 0x82f63b78u;
    crc0 = crc32c_update_u64(crc0, id);
    crc1 = crc32c_update_u64(crc1, id ^ 0xa5a5a5a5a5a5a5a5ull);
    const uint32_t scale_bits = float_bits(scale);
    crc0 = crc32c_update_u32(crc0, scale_bits);
    crc1 = crc32c_update_u32(crc1, scale_bits ^ 0xa5a5a5a5u);
    crc0 = crc32c_update(crc0, codes.data(), codes.size());
    crc1 = crc32c_update(crc1, codes.data(), codes.size());
    return (static_cast<uint64_t>(crc0 ^ 0xffffffffu) << 32) |
        static_cast<uint64_t>(crc1 ^ 0xffffffffu);
}

uint32_t f32_state_token(
        int dim,
        size_t n_active,
        uint64_t hash_xor,
        uint64_t hash_sum,
        uint64_t hash_sum_rot) {
    const uint32_t bit_width = 32;
    const uint32_t storage_kind = 1;
    uint32_t crc = 0xffffffffu;
    crc = crc32c_update_u32(crc, static_cast<uint32_t>(dim));
    crc = crc32c_update_u32(crc, bit_width);
    crc = crc32c_update_u32(crc, storage_kind);
    crc = crc32c_update_u64(crc, static_cast<uint64_t>(n_active));
    crc = crc32c_update_u64(crc, hash_xor);
    crc = crc32c_update_u64(crc, hash_sum);
    crc = crc32c_update_u64(crc, hash_sum_rot);
    return crc ^ 0xffffffffu;
}

uint32_t state_token_from_wide_log_header(const std::vector<uint8_t> & bytes, int dim) {
    CHECK(delta_log_is_v4(bytes));
    const int bit_width = static_cast<int>(bytes[5]);
    const uint32_t storage_kind =
        bit_width == 4 ? 3u : (bit_width == 8 ? 2u : 1u);
    const size_t state_offset = 16;
    uint32_t crc = 0xffffffffu;
    crc = crc32c_update_u32(crc, static_cast<uint32_t>(dim));
    crc = crc32c_update_u32(crc, static_cast<uint32_t>(bit_width));
    crc = crc32c_update_u32(crc, storage_kind);
    crc = crc32c_update_u64(crc, read_u64_le_at(bytes, state_offset + 0));
    crc = crc32c_update_u64(crc, read_u64_le_at(bytes, state_offset + 8));
    crc = crc32c_update_u64(crc, read_u64_le_at(bytes, state_offset + 16));
    crc = crc32c_update_u64(crc, read_u64_le_at(bytes, state_offset + 24));
    return crc ^ 0xffffffffu;
}

uint32_t state_token_from_wide_values(
        int bit_width,
        int dim,
        uint64_t n_active,
        uint64_t hash_xor,
        uint64_t hash_sum,
        uint64_t hash_sum_rot) {
    const uint32_t storage_kind =
        bit_width == 4 ? 3u : (bit_width == 8 ? 2u : 1u);
    uint32_t crc = 0xffffffffu;
    crc = crc32c_update_u32(crc, static_cast<uint32_t>(dim));
    crc = crc32c_update_u32(crc, static_cast<uint32_t>(bit_width));
    crc = crc32c_update_u32(crc, storage_kind);
    crc = crc32c_update_u64(crc, n_active);
    crc = crc32c_update_u64(crc, hash_xor);
    crc = crc32c_update_u64(crc, hash_sum);
    crc = crc32c_update_u64(crc, hash_sum_rot);
    return crc ^ 0xffffffffu;
}

void write_v1_index(
        const std::string & path,
        int dim,
        int bit_width,
        const std::vector<float> & vectors,
        const std::vector<uint64_t> & ids) {
    CHECK(vectors.size() == ids.size() * static_cast<size_t>(dim));

    std::vector<uint8_t> bytes = { 'T', 'V', 'P', 'I', 1,
                                   static_cast<uint8_t>(bit_width), 0, 0 };
    append_u32_le(bytes, static_cast<uint32_t>(dim));
    append_u32_le(bytes, static_cast<uint32_t>(ids.size()));
    for (float value : vectors) {
        append_f32_le(bytes, value);
    }
    for (uint64_t id : ids) {
        append_u64_le(bytes, id);
    }
    write_file_bytes(path, bytes);
}

std::vector<uint8_t> read_bytes(const std::filesystem::path & path) {
    return read_file_bytes(path.string());
}

void write_bytes(const std::filesystem::path & path, const std::vector<uint8_t> & bytes) {
    write_file_bytes(path.string(), bytes);
}

void put_u32_le(std::vector<uint8_t> & bytes, size_t offset, uint32_t value) {
    write_u32_le_at(bytes, offset, value);
}

uint32_t crc32c_bytes(const std::vector<uint8_t> & bytes) {
    return crc32c_update(0xffffffffu, bytes.data(), bytes.size()) ^ 0xffffffffu;
}

std::vector<uint8_t> snapshot_bytes(
        int dim,
        uint32_t n,
        const std::vector<float> & vectors,
        const std::vector<uint64_t> & ids) {
    CHECK(vectors.size() == static_cast<size_t>(n) * static_cast<size_t>(dim));
    CHECK(ids.size() == n);

    std::vector<uint8_t> bytes = { 'T', 'V', 'P', 'I', 1, 32, 0, 0 };
    append_u32_le(bytes, static_cast<uint32_t>(dim));
    append_u32_le(bytes, n);
    for (float value : vectors) {
        append_f32_le(bytes, value);
    }
    for (uint64_t id : ids) {
        append_u64_le(bytes, id);
    }
    return bytes;
}

uint32_t legacy_state_crc32c_f32(
        int dim,
        const std::vector<float> & vectors,
        const std::vector<uint64_t> & ids) {
    CHECK(vectors.size() == ids.size() * static_cast<size_t>(dim));

    uint32_t crc = 0xffffffffu;
    crc = crc32c_update_u32(crc, static_cast<uint32_t>(dim));
    crc = crc32c_update_u32(crc, 32);
    crc = crc32c_update_u32(crc, 1);
    crc = crc32c_update_u64(crc, static_cast<uint64_t>(ids.size()));
    for (float value : vectors) {
        crc = crc32c_update_u32(crc, float_bits(value));
    }
    for (uint64_t id : ids) {
        crc = crc32c_update_u64(crc, id);
    }
    return crc ^ 0xffffffffu;
}

uint32_t f32_state_token_for(
        int dim,
        const std::vector<float> & vectors,
        const std::vector<uint64_t> & ids) {
    CHECK(vectors.size() == ids.size() * static_cast<size_t>(dim));

    uint64_t hash_xor = 0;
    uint64_t hash_sum = 0;
    uint64_t hash_sum_rot = 0;
    const size_t dim_sz = static_cast<size_t>(dim);
    for (size_t row = 0; row < ids.size(); ++row) {
        const std::vector<float> vector(
            vectors.begin() + static_cast<std::ptrdiff_t>(row * dim_sz),
            vectors.begin() + static_cast<std::ptrdiff_t>((row + 1) * dim_sz));
        const uint64_t hash = slot_state_hash_f32(ids[row], vector);
        hash_xor ^= hash;
        hash_sum += hash;
        hash_sum_rot += rotl64(hash, 17);
    }
    return f32_state_token(dim, ids.size(), hash_xor, hash_sum, hash_sum_rot);
}

std::vector<uint8_t> build_legacy_f32_delta_log(
        uint8_t version,
        int dim,
        uint32_t base_state,
        uint32_t post_state,
        const std::vector<float> & vectors,
        const std::vector<uint64_t> & ids) {
    CHECK(version == 1 || version == 3);
    CHECK(vectors.size() == ids.size() * static_cast<size_t>(dim));

    std::vector<uint8_t> payload;
    for (uint64_t id : ids) {
        append_u64_le(payload, id);
    }
    for (float value : vectors) {
        append_f32_le(payload, value);
    }

    std::vector<uint8_t> bytes = { 'T', 'V', 'D', 'L', version, 32, 0, 0 };
    append_u32_le(bytes, static_cast<uint32_t>(dim));
    append_u32_le(bytes, base_state);
    const size_t record_offset = bytes.size();
    bytes.push_back(1); // add
    bytes.insert(bytes.end(), { 0, 0, 0 });
    append_u32_le(bytes, static_cast<uint32_t>(ids.size()));
    append_u64_le(bytes, payload.size());
    append_u32_le(bytes, 0); // record CRC placeholder
    append_u32_le(bytes, post_state);
    bytes.insert(bytes.end(), payload.begin(), payload.end());
    refresh_delta_record_crc(bytes, record_offset);
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
        scores.push_back({ quantized_reference_score(vectors.data() + row * static_cast<size_t>(dim), query, dim,
                                                     bit_width),
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
    CHECK(ggml_vec_index_search(idx, queries.data(), n_q, k, out_scores.data(), out_ids.data()) ==
          GGML_VEC_INDEX_OK);

    const float tolerance = bit_width == 8 ? 1e-4f : 2e-4f;
    for (int q = 0; q < n_q; ++q) {
        const auto expected = reference_topk(vectors, ids, queries.data() + static_cast<size_t>(q) * dim, dim,
                                            bit_width, k);
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
        1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f,
    };
    const std::array<uint64_t, 2> ids = {
        910001ULL,
        910002ULL,
    };
    const std::array<float, dim> query = {
        1.0e8f,
        1.0f,
        -1.0e8f,
        1.0f,
        0.0f,
        0.0f,
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
        1.0f, -1.0f, 0.0f, 0.0f,
        1.0e30f, 1.0e30f, 0.0f, 0.0f,
    };
    const std::array<uint64_t, 2> ids = { 8300ULL, 8301ULL };
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
    CHECK(ggml_vec_index_search_ivf(idx, query.data(), 1, 1, 1, ivf_score.data(), ivf_id.data()) ==
          GGML_VEC_INDEX_OK);
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
    const std::array<float, dim>  query = { -1.0f, 0.0f };

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
    CHECK(exact_id[0] == ids[1]);
    CHECK(ivf_id[0] == exact_id[0]);
    CHECK(ivf_score[0] == exact_score[0]);

    ggml_vec_index_free(idx);
}

uint64_t encoded_slot_state_hash(uint64_t id, float scale, const std::vector<uint8_t> & codes);
std::array<uint64_t, 4> wide_state_from_hashes(std::initializer_list<uint64_t> hashes);
void append_v4_delta_record(
        std::vector<uint8_t> & bytes,
        uint8_t op,
        uint32_t n,
        const std::vector<uint8_t> & payload,
        const std::array<uint64_t, 4> & post_state);

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

void check_writer_completes_after_read_admission_closes() {
    constexpr int dim    = 16;
    constexpr int n_rows = 512;
    constexpr int k      = 8;

    std::vector<float> rows;
    rows.reserve(static_cast<size_t>(n_rows) * dim);
    std::vector<uint64_t> row_ids;
    row_ids.reserve(n_rows);
    for (int row = 0; row < n_rows; ++row) {
        row_ids.push_back(static_cast<uint64_t>(920000 + row));
        for (int col = 0; col < dim; ++col) {
            const float x = static_cast<float>(((row + 1) * (col + 3)) % 17) - 8.0f;
            rows.push_back(x / 8.0f);
        }
    }

    auto * idx = ggml_vec_index_create(dim, /*bit_width=*/32);
    CHECK(idx != nullptr);
    CHECK(ggml_vec_index_add(idx, rows.data(), n_rows, row_ids.data()) == GGML_VEC_INDEX_OK);

    std::atomic<int>  ready{ 0 };
    std::atomic<int>  read_count{ 0 };
    std::atomic<int>  failures{ 0 };
    std::atomic<bool> start{ false };
    std::atomic<bool> writer_pending{ false };
    std::vector<std::thread> readers;
    for (int t = 0; t < 4; ++t) {
        readers.emplace_back([&, t]() {
            const float * query = rows.data() + static_cast<size_t>(t * 37 % n_rows) * dim;
            std::array<float, k>    scores{};
            std::array<uint64_t, k> out_ids{};
            ready.fetch_add(1);
            while (!start.load()) {
                std::this_thread::yield();
            }
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
            while (!writer_pending.load() && std::chrono::steady_clock::now() < deadline) {
                if (ggml_vec_index_search(idx, query, 1, k, scores.data(), out_ids.data()) != GGML_VEC_INDEX_OK) {
                    failures.fetch_add(1);
                }
                if (ggml_vec_index_len(idx) < n_rows) {
                    failures.fetch_add(1);
                }
                read_count.fetch_add(1);
            }
        });
    }

    while (ready.load() != 4) {
        std::this_thread::yield();
    }
    start.store(true);
    const auto read_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
    while (read_count.load() < 32 && std::chrono::steady_clock::now() < read_deadline) {
        std::this_thread::yield();
    }
    CHECK(read_count.load() >= 32);

    std::atomic<int>  writer_status{ GGML_VEC_INDEX_E_INTERNAL };
    std::atomic<bool> writer_done{ false };
    const std::vector<float> writer_vec = normalize({
        1.0f,
        -0.5f,
        0.25f,
        0.75f,
        -1.0f,
        0.5f,
        -0.25f,
        0.125f,
        0.0f,
        0.25f,
        -0.75f,
        1.0f,
        -0.125f,
        0.625f,
        -0.375f,
        0.875f,
    });
    const uint64_t writer_id = 930000ULL;
    std::thread writer([&]() {
        writer_pending.store(true);
        writer_status.store(ggml_vec_index_add(idx, writer_vec.data(), 1, &writer_id));
        writer_done.store(true);
    });

    const auto writer_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!writer_done.load() && std::chrono::steady_clock::now() < writer_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    CHECK(writer_done.load());
    writer.join();
    for (std::thread & reader : readers) {
        reader.join();
    }

    CHECK(failures.load() == 0);
    CHECK(writer_status.load() == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_contains(idx, writer_id) == 1);
    CHECK(ggml_vec_index_len(idx) == n_rows + 1);
    ggml_vec_index_free(idx);
}

void check_ivf_state_not_persisted() {
    const std::array<float, kDim * 2> vectors = {
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
    };
    const std::array<uint64_t, 2> ids = { 9201ULL, 9202ULL };

    const std::string snapshot_path =
        (std::filesystem::temp_directory_path() / "ggml-vector-index-ivf-state.tvim").string();
    std::filesystem::remove(snapshot_path);
    auto * built = ggml_vec_index_create(kDim, /*bit_width=*/32);
    CHECK(built != nullptr);
    CHECK(ggml_vec_index_add(built, vectors.data(), static_cast<int>(ids.size()), ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_build_ivf(built, /*n_lists=*/2, /*n_iter=*/1) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(built, snapshot_path.c_str()) == GGML_VEC_INDEX_OK);
    ggml_vec_index_free(built);

    auto * loaded = ggml_vec_index_load(snapshot_path.c_str());
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
    std::filesystem::remove(snapshot_path);
}

void write_v2_turbovec_index(
        const std::string & path,
        int dim,
        int bit_width,
        uint8_t storage_kind) {
    std::vector<uint8_t> bytes = { 'T', 'V', 'P', 'I', 2,
                                   static_cast<uint8_t>(bit_width), storage_kind, 0 };
    append_u32_le(bytes, static_cast<uint32_t>(dim));
    append_u32_le(bytes, 1);
    append_u32_le(bytes, 1);
    append_u32_le(bytes, sizeof(float));
    append_u32_le(bytes, 0);
    append_u32_le(bytes, 0);
    append_f32_le(bytes, 1.0f);
    bytes.resize(bytes.size() + static_cast<size_t>(bit_width) * (static_cast<size_t>(dim) / 8), 0);
    append_u64_le(bytes, 12345);
    write_file_bytes(path, bytes);
}

// An empty v3 snapshot of a legacy (turbovec 0.9) TurboVec index. v3 files
// are the only way to obtain a legacy index through the public API: loading
// one keeps the 0.9 rotation, codebook and first-add TQ+ fit.
void write_empty_v3_turbovec_index(const std::string & path, int dim, int bit_width) {
    std::vector<uint8_t> bytes = {
        'T', 'V', 'P', 'I', 3, static_cast<uint8_t>(bit_width),
        static_cast<uint8_t>(bit_width == 2 ? 5 : 4), 1,
    };
    append_u32_le(bytes, static_cast<uint32_t>(dim));
    append_u32_le(bytes, 0);
    append_u32_le(bytes, 1);
    append_u32_le(bytes, sizeof(float));
    append_u32_le(bytes, 0);
    append_u32_le(bytes, 0);
    const uint32_t header_crc = crc32c_update(0xffffffffu, bytes.data(), bytes.size()) ^ 0xffffffffu;
    append_u32_le(bytes, header_crc);
    append_u32_le(bytes, 0);
    append_u32_le(bytes, 0);
    append_u32_le(bytes, 0);
    write_file_bytes(path, bytes);
}

ggml_vec_index_t * create_legacy_turbovec(int dim, int bit_width) {
    const std::string path =
        (std::filesystem::temp_directory_path() /
         ("ggml-vector-index-legacy-turbovec-" + std::to_string(dim) + "-q" +
          std::to_string(bit_width) + ".tvim")).string();
    write_empty_v3_turbovec_index(path, dim, bit_width);
    ggml_vec_index_t * idx = ggml_vec_index_load(path.c_str());
    std::filesystem::remove(path);
    CHECK(idx != nullptr);
    return idx;
}

template <typename Fn>
void expect_corrupt_load_fails(
        const std::string & source_path,
        const std::string & corrupt_path,
        Fn mutate) {
    std::vector<uint8_t> bytes = read_file_bytes(source_path);
    mutate(bytes);
    write_file_bytes(corrupt_path, bytes);

    auto * bad = ggml_vec_index_load(corrupt_path.c_str());
    CHECK(bad == nullptr);
    ggml_vec_index_free(bad);
    std::filesystem::remove(corrupt_path);
}

void write_sparse_bytes(const std::filesystem::path & path, const std::vector<uint8_t> & prefix, uint64_t size) {
    CHECK(size >= prefix.size());
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    CHECK(f.is_open());
    if (!prefix.empty()) {
        f.write(reinterpret_cast<const char *>(prefix.data()), static_cast<std::streamsize>(prefix.size()));
    }
    if (size > prefix.size()) {
        f.seekp(static_cast<std::streamoff>(size - 1));
        const char zero = 0;
        f.write(&zero, 1);
    }
    CHECK(static_cast<bool>(f));
}

uint32_t crc32c_u32(uint32_t crc, uint32_t value) {
    return crc32c_update_u32(crc, value);
}

uint32_t crc32c_u64(uint32_t crc, uint64_t value) {
    return crc32c_update_u64(crc, value);
}

uint32_t f32_state_crc(int dim, const std::vector<float> & vectors, const std::vector<uint64_t> & ids) {
    return legacy_state_crc32c_f32(dim, vectors, ids);
}

uint32_t f32_state_token(const std::vector<float> & vectors, const std::vector<uint64_t> & ids, int dim) {
    return f32_state_token_for(dim, vectors, ids);
}

uint64_t encoded_slot_state_hash(uint64_t id, float scale, const std::vector<uint8_t> & codes) {
    uint32_t crc0 = 0xffffffffu;
    uint32_t crc1 = 0x82f63b78u;
    crc0 = crc32c_update_u64(crc0, id);
    crc1 = crc32c_update_u64(crc1, id ^ 0xa5a5a5a5a5a5a5a5ull);
    const uint32_t scale_bits = float_bits(scale);
    crc0 = crc32c_update_u32(crc0, scale_bits);
    crc1 = crc32c_update_u32(crc1, scale_bits ^ 0xa5a5a5a5u);
    crc0 = crc32c_update(crc0, codes.data(), codes.size());
    crc1 = crc32c_update(crc1, codes.data(), codes.size());
    return (static_cast<uint64_t>(crc0 ^ 0xffffffffu) << 32) |
        static_cast<uint64_t>(crc1 ^ 0xffffffffu);
}

std::array<uint64_t, 4> wide_state_from_hashes(std::initializer_list<uint64_t> hashes) {
    uint64_t hash_xor = 0;
    uint64_t hash_sum = 0;
    uint64_t hash_sum_rot = 0;
    for (uint64_t hash : hashes) {
        hash_xor ^= hash;
        hash_sum += hash;
        hash_sum_rot += rotl64(hash, 17);
    }
    return { static_cast<uint64_t>(hashes.size()), hash_xor, hash_sum, hash_sum_rot };
}

uint32_t encoded_state_token(const std::array<uint64_t, 4> & state, int dim, int bit_width) {
    uint32_t crc = 0xffffffffu;
    crc = crc32c_update_u32(crc, static_cast<uint32_t>(dim));
    crc = crc32c_update_u32(crc, static_cast<uint32_t>(bit_width));
    crc = crc32c_update_u32(crc, bit_width == 4 ? 3u : 2u);
    for (uint64_t value : state) {
        crc = crc32c_update_u64(crc, value);
    }
    return crc ^ 0xffffffffu;
}

uint32_t encoded_state_crc(
        int dim,
        int bit_width,
        const std::vector<float> & scales,
        const std::vector<std::vector<uint8_t>> & rows,
        const std::vector<uint64_t> & ids) {
    CHECK(scales.size() == rows.size());
    CHECK(rows.size() == ids.size());
    uint32_t crc = 0xffffffffu;
    crc = crc32c_update_u32(crc, static_cast<uint32_t>(dim));
    crc = crc32c_update_u32(crc, static_cast<uint32_t>(bit_width));
    crc = crc32c_update_u32(crc, bit_width == 4 ? 3u : 2u);
    crc = crc32c_update_u64(crc, static_cast<uint64_t>(ids.size()));
    for (float scale : scales) {
        uint32_t bits = 0;
        std::memcpy(&bits, &scale, sizeof(bits));
        crc = crc32c_update_u32(crc, bits);
    }
    for (const auto & row : rows) {
        crc = crc32c_update(crc, row.data(), row.size());
    }
    for (uint64_t id : ids) {
        crc = crc32c_update_u64(crc, id);
    }
    return crc ^ 0xffffffffu;
}

std::array<uint64_t, 4> f32_wide_state(
        const std::vector<float> & vectors,
        const std::vector<uint64_t> & ids,
        int dim) {
    CHECK(vectors.size() == ids.size() * static_cast<size_t>(dim));
    uint64_t hash_xor = 0;
    uint64_t hash_sum = 0;
    uint64_t hash_sum_rot = 0;
    const size_t dim_sz = static_cast<size_t>(dim);
    for (size_t row = 0; row < ids.size(); ++row) {
        const std::vector<float> vector(
            vectors.begin() + static_cast<std::ptrdiff_t>(row * dim_sz),
            vectors.begin() + static_cast<std::ptrdiff_t>((row + 1) * dim_sz));
        const uint64_t hash = slot_state_hash_f32(ids[row], vector);
        hash_xor ^= hash;
        hash_sum += hash;
        hash_sum_rot += rotl64(hash, 17);
    }
    return { static_cast<uint64_t>(ids.size()), hash_xor, hash_sum, hash_sum_rot };
}

void append_wide_state(std::vector<uint8_t> & bytes, const std::array<uint64_t, 4> & state) {
    for (uint64_t value : state) {
        append_u64_le(bytes, value);
    }
}

void append_v1_delta_record(
        std::vector<uint8_t> & bytes,
        uint8_t op,
        uint32_t n,
        const std::vector<uint8_t> & payload,
        uint32_t post_state) {
    const size_t record_offset = bytes.size();
    bytes.push_back(op);
    bytes.insert(bytes.end(), { 0, 0, 0 });
    append_u32_le(bytes, n);
    append_u64_le(bytes, payload.size());
    append_u32_le(bytes, 0);
    append_u32_le(bytes, post_state);
    bytes.insert(bytes.end(), payload.begin(), payload.end());
    refresh_delta_record_crc(bytes, record_offset);
}

void append_v4_delta_record(
        std::vector<uint8_t> & bytes,
        uint8_t op,
        uint32_t n,
        const std::vector<uint8_t> & payload,
        const std::array<uint64_t, 4> & post_state) {
    const size_t record_offset = bytes.size();
    bytes.push_back(op);
    bytes.insert(bytes.end(), { 0, 0, 0 });
    append_u32_le(bytes, n);
    append_u64_le(bytes, payload.size());
    append_u32_le(bytes, 0);
    append_u32_le(bytes, 0);
    append_wide_state(bytes, post_state);
    bytes.insert(bytes.end(), payload.begin(), payload.end());
    refresh_delta_record_crc(bytes, record_offset);
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
        ggml_vec_index_t * loaded = nullptr;
        const int load_status =
            ggml_vec_index_load_with_delta_ex(snapshot.path.string().c_str(), delta.path.string().c_str(), &loaded);
        if (load_status != GGML_VEC_INDEX_OK) {
            std::fprintf(stderr, "FAIL committed delta replay v%d status=%d\n", version, load_status);
        }
        CHECK(load_status == GGML_VEC_INDEX_OK);
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

void check_delta_log_tail_recovery() {
    const std::array<float, kDim> base_vec = {
        1.0f,
        0.0f,
        0.0f,
        0.0f,
    };
    const uint64_t base_id = 9301ULL;

    const std::filesystem::path snapshot_path =
        std::filesystem::temp_directory_path() / "ggml-vector-index-tail-recovery.tvim";
    const std::filesystem::path delta_path =
        std::filesystem::temp_directory_path() / "ggml-vector-index-tail-recovery.tvid";
    const std::string snapshot = snapshot_path.string();
    const std::string delta = delta_path.string();
    auto remove_delta_artifacts = [](const std::filesystem::path & path) {
        std::error_code ec;
        std::filesystem::remove(path, ec);
        std::filesystem::remove(path.string() + ".lock", ec);
    };
    std::error_code cleanup_ec;
    std::filesystem::remove(snapshot_path, cleanup_ec);
    remove_delta_artifacts(delta_path);

    auto * base = ggml_vec_index_create(kDim, /*bit_width=*/32);
    CHECK(base != nullptr);
    CHECK(ggml_vec_index_add(base, base_vec.data(), 1, &base_id) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(base, snapshot.c_str()) == GGML_VEC_INDEX_OK);

    auto * mmap = ggml_vec_index_load_mmap(snapshot.c_str());
    CHECK(mmap != nullptr);
    CHECK(ggml_vec_index_compact_delta(mmap, snapshot.c_str(), delta.c_str()) == GGML_VEC_INDEX_E_INVALID_ARG);
    CHECK(ggml_vec_index_contains(mmap, base_id) == 1);
    {
        const std::filesystem::path snapshot_alias_path = snapshot_path.string() + ".alias";
        const std::string           snapshot_alias      = snapshot_alias_path.string();
        std::error_code ec;
        std::filesystem::remove(snapshot_alias_path, ec);
        ec.clear();
        std::filesystem::create_hard_link(snapshot_path, snapshot_alias_path, ec);
        if (!ec) {
            CHECK(std::filesystem::equivalent(snapshot_path, snapshot_alias_path, ec));
            CHECK(!ec);
            CHECK(ggml_vec_index_write(mmap, snapshot_alias.c_str()) == GGML_VEC_INDEX_E_INVALID_ARG);
            CHECK(ggml_vec_index_compact_delta(mmap, snapshot_alias.c_str(), delta.c_str()) ==
                  GGML_VEC_INDEX_E_INVALID_ARG);
            std::filesystem::remove(snapshot_alias_path, ec);
        } else {
            CHECK(ec == std::errc::operation_not_supported || ec == std::errc::function_not_supported ||
                  ec == std::errc::permission_denied);
        }
    }
    ggml_vec_index_free(mmap);

    CHECK(ggml_vec_index_compact_delta(base, snapshot.c_str(), delta.c_str()) == GGML_VEC_INDEX_OK);
    {
        const std::filesystem::path delta_alias_path = delta_path.string() + ".alias";
        const std::string           delta_alias      = delta_alias_path.string();
        std::error_code ec;
        std::filesystem::remove(delta_alias_path, ec);
        ec.clear();
        std::filesystem::create_hard_link(delta_path, delta_alias_path, ec);
        if (!ec) {
            CHECK(std::filesystem::equivalent(delta_path, delta_alias_path, ec));
            CHECK(!ec);
            CHECK(ggml_vec_index_compact_delta(base, snapshot.c_str(), delta_alias.c_str()) == GGML_VEC_INDEX_OK);
            CHECK(std::filesystem::equivalent(delta_path, delta_alias_path, ec));
            CHECK(!ec);
            CHECK(read_file_bytes(delta_alias) == read_file_bytes(delta));
            CHECK(read_file_bytes(delta_alias).size() == 48);
            std::filesystem::remove(delta_alias_path, ec);
        } else {
            CHECK(ec == std::errc::operation_not_supported || ec == std::errc::function_not_supported ||
                  ec == std::errc::permission_denied);
        }
    }
    ggml_vec_index_free(base);

#ifndef _WIN32
    std::filesystem::path lock_path = delta_path;
    lock_path += ".lock";
    std::filesystem::permissions(lock_path, std::filesystem::perms::owner_read, std::filesystem::perm_options::replace);
    auto * read_only_lock_loaded =
        ggml_vec_index_load_with_delta(snapshot.c_str(), delta.c_str());
    CHECK(read_only_lock_loaded != nullptr);
    ggml_vec_index_free(read_only_lock_loaded);
    std::filesystem::permissions(lock_path, std::filesystem::perms::owner_read | std::filesystem::perms::owner_write,
                                 std::filesystem::perm_options::replace);
#endif

    std::vector<uint8_t> corrupted_delta = read_file_bytes(delta);
    CHECK(corrupted_delta.size() >= 48);
    corrupted_delta.resize(corrupted_delta.size() + 56, 0);
    write_file_bytes(delta, corrupted_delta);

    auto * loaded = ggml_vec_index_load_with_delta(snapshot.c_str(), delta.c_str());
    CHECK(loaded != nullptr);
    CHECK(ggml_vec_index_len(loaded) == 1);
    CHECK(ggml_vec_index_contains(loaded, base_id) == 1);
    CHECK(ggml_vec_index_compact_delta(loaded, snapshot.c_str(), delta.c_str()) == GGML_VEC_INDEX_OK);
    CHECK(read_file_bytes(delta).size() == 48);
    ggml_vec_index_free(loaded);

    const std::array<float, kDim> added_vec = {
        0.0f,
        1.0f,
        0.0f,
        0.0f,
    };
    const uint64_t added_id = 9302ULL;
    auto * add_idx = ggml_vec_index_load(snapshot.c_str());
    CHECK(add_idx != nullptr);
    CHECK(ggml_vec_index_add_logged(add_idx, added_vec.data(), 1, &added_id, delta.c_str()) == GGML_VEC_INDEX_OK);
    ggml_vec_index_free(add_idx);
    std::vector<uint8_t> add_log = read_file_bytes(delta);
    add_log.resize(add_log.size() + 56, 0);
    write_file_bytes(delta, add_log);

    loaded = ggml_vec_index_load_with_delta(snapshot.c_str(), delta.c_str());
    CHECK(loaded != nullptr);
    CHECK(ggml_vec_index_contains(loaded, base_id) == 1);
    CHECK(ggml_vec_index_contains(loaded, added_id) == 1);
    CHECK(ggml_vec_index_compact_delta(loaded, snapshot.c_str(), delta.c_str()) == GGML_VEC_INDEX_OK);
    CHECK(read_file_bytes(delta).size() == 48);
    ggml_vec_index_free(loaded);

    auto * compacted = ggml_vec_index_load(snapshot.c_str());
    CHECK(compacted != nullptr);
    CHECK(ggml_vec_index_contains(compacted, base_id) == 1);
    CHECK(ggml_vec_index_contains(compacted, added_id) == 1);
    ggml_vec_index_free(compacted);

    write_file_bytes(delta, { 'T', 'V', 'D' });
    loaded = ggml_vec_index_load_with_delta(snapshot.c_str(), delta.c_str());
    CHECK(loaded != nullptr);
    CHECK(ggml_vec_index_len(loaded) == 2);
    CHECK(ggml_vec_index_contains(loaded, base_id) == 1);
    CHECK(ggml_vec_index_contains(loaded, added_id) == 1);
    CHECK(ggml_vec_index_compact_delta(loaded, snapshot.c_str(), delta.c_str()) == GGML_VEC_INDEX_OK);
    CHECK(read_file_bytes(delta).size() == 48);
    ggml_vec_index_free(loaded);

    std::error_code       ec;
    std::filesystem::path lock_path_to_remove = delta_path;
    lock_path_to_remove += ".lock";
    std::filesystem::remove(lock_path_to_remove, ec);
    std::filesystem::remove(snapshot_path, ec);
    remove_delta_artifacts(delta_path);
}

void check_delta_log_alternating_writers() {
    const std::filesystem::path snapshot_path =
        std::filesystem::temp_directory_path() / "ggml-vector-index-alternating-writers.tvim";
    const std::filesystem::path delta_path =
        std::filesystem::temp_directory_path() / "ggml-vector-index-alternating-writers.tvid";
    const std::string snapshot = snapshot_path.string();
    const std::string delta = delta_path.string();
    std::error_code ec;
    std::filesystem::remove(snapshot_path, ec);
    std::filesystem::remove(delta_path, ec);
    std::filesystem::remove(delta_path.string() + ".lock", ec);

    const std::array<uint64_t, 4> ids = { 9401ULL, 9402ULL, 9403ULL, 9404ULL };
    const std::vector<float> base_vec = normalize({ 1.0f, 0.0f, 0.0f, 0.0f });
    const std::vector<float> a_first  = normalize({ 0.0f, 1.0f, 0.0f, 0.0f });
    const std::vector<float> b_first  = normalize({ 0.0f, 0.0f, 1.0f, 0.0f });
    const std::vector<float> a_second = normalize({ 0.0f, 0.0f, 0.0f, 1.0f });

    auto * base = ggml_vec_index_create(kDim, /*bit_width=*/32);
    CHECK(base != nullptr);
    CHECK(ggml_vec_index_add(base, base_vec.data(), 1, &ids[0]) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(base, snapshot.c_str()) == GGML_VEC_INDEX_OK);
    ggml_vec_index_free(base);

    auto * writer_a = ggml_vec_index_load(snapshot.c_str());
    auto * writer_b = ggml_vec_index_load(snapshot.c_str());
    auto * writer_c = ggml_vec_index_load(snapshot.c_str());
    auto * writer_d = ggml_vec_index_load(snapshot.c_str());
    CHECK(writer_a != nullptr);
    CHECK(writer_b != nullptr);
    CHECK(writer_c != nullptr);
    CHECK(writer_d != nullptr);

    CHECK(ggml_vec_index_add_logged(writer_a, a_first.data(), 1, &ids[1], delta.c_str()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_add_logged(writer_b, b_first.data(), 1, &ids[2], delta.c_str()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_contains(writer_b, ids[1]) == 1);
    CHECK(ggml_vec_index_contains(writer_b, ids[2]) == 1);

    CHECK(ggml_vec_index_add_logged(writer_c, b_first.data(), 1, &ids[2], delta.c_str()) ==
          GGML_VEC_INDEX_E_DUPLICATE);
    CHECK(ggml_vec_index_contains(writer_c, ids[1]) == 1);
    CHECK(ggml_vec_index_contains(writer_c, ids[2]) == 1);
    CHECK(ggml_vec_index_len(writer_c) == 3);

    CHECK(ggml_vec_index_remove_logged(writer_d, 9499ULL, delta.c_str()) ==
          GGML_VEC_INDEX_E_NOT_FOUND);
    CHECK(ggml_vec_index_contains(writer_d, ids[1]) == 1);
    CHECK(ggml_vec_index_contains(writer_d, ids[2]) == 1);
    CHECK(ggml_vec_index_len(writer_d) == 3);

    CHECK(ggml_vec_index_add_logged(writer_a, a_second.data(), 1, &ids[3], delta.c_str()) == GGML_VEC_INDEX_OK);
    for (uint64_t id : ids) {
        CHECK(ggml_vec_index_contains(writer_a, id) == 1);
    }
    CHECK(ggml_vec_index_len(writer_a) == static_cast<int>(ids.size()));

    auto * loaded = ggml_vec_index_load_with_delta(snapshot.c_str(), delta.c_str());
    CHECK(loaded != nullptr);
    for (uint64_t id : ids) {
        CHECK(ggml_vec_index_contains(loaded, id) == 1);
    }
    CHECK(ggml_vec_index_len(loaded) == static_cast<int>(ids.size()));

    ggml_vec_index_free(loaded);
    ggml_vec_index_free(writer_d);
    ggml_vec_index_free(writer_c);
    ggml_vec_index_free(writer_b);
    ggml_vec_index_free(writer_a);
    std::filesystem::remove(snapshot_path, ec);
    std::filesystem::remove(delta_path, ec);
    std::filesystem::remove(delta_path.string() + ".lock", ec);
}

uint64_t fnv1a_bytes(const uint8_t * values, size_t size) {
    uint64_t hash = UINT64_C(0xcbf29ce484222325);
    for (size_t i = 0; i < size; ++i) {
        hash ^= values[i];
        hash *= UINT64_C(0x100000001b3);
    }
    return hash;
}

struct TvimSection {
    size_t offset = 0;
    size_t size = 0;
};

struct TurboVecTvimLayout {
    int dim = 0;
    int n = 0;
    TvimSection qparams;
    TvimSection calibration;
    TvimSection vectors;
    TvimSection ids;
    TvimSection checksum;
};

TurboVecTvimLayout parse_turbovec_tvim_layout(const std::vector<uint8_t> & bytes, int bits) {
    CHECK(bits == 2 || bits == 4);
    CHECK(bytes.size() >= 32 + 16);
    CHECK(bytes[0] == 'T' && bytes[1] == 'V' && bytes[2] == 'P' && bytes[3] == 'I');
    CHECK(bytes[4] == 3);
    CHECK(bytes[5] == static_cast<uint8_t>(bits));
    CHECK(bytes[6] == static_cast<uint8_t>(bits == 2 ? 5 : 4));
    CHECK((bytes[7] & 1) != 0);

    TurboVecTvimLayout layout;
    layout.dim = static_cast<int>(read_u32_le_from(bytes.data() + 8));
    layout.n = static_cast<int>(read_u32_le_from(bytes.data() + 12));
    const size_t qparam_bytes = read_u32_le_from(bytes.data() + 20);
    const size_t comp_bytes = read_u32_le_from(bytes.data() + 24);
    const size_t calibration_bytes = read_u32_le_from(bytes.data() + 28);
    CHECK(layout.dim > 0);
    CHECK(layout.n >= 0);
    CHECK(qparam_bytes == sizeof(float));
    CHECK(comp_bytes == 0);
    CHECK(calibration_bytes == 0 ||
          calibration_bytes == 2 * static_cast<size_t>(layout.dim) * sizeof(float));

    const size_t row_bytes = bits == 2 ?
        static_cast<size_t>(layout.dim) / 4 :
        static_cast<size_t>(layout.dim) / 2;
    layout.qparams = { 32, static_cast<size_t>(layout.n) * qparam_bytes };
    layout.calibration = {
        layout.qparams.offset + layout.qparams.size,
        calibration_bytes,
    };
    layout.vectors = {
        layout.calibration.offset + layout.calibration.size,
        static_cast<size_t>(layout.n) * row_bytes,
    };
    layout.ids = {
        layout.vectors.offset + layout.vectors.size,
        static_cast<size_t>(layout.n) * sizeof(uint64_t),
    };
    layout.checksum = {
        layout.ids.offset + layout.ids.size,
        16,
    };
    CHECK(bytes.size() == layout.checksum.offset + layout.checksum.size);
    return layout;
}

uint64_t section_hash(const std::vector<uint8_t> & bytes, TvimSection section) {
    CHECK(section.size > 0);
    CHECK(bytes.size() >= section.offset + section.size);
    return fnv1a_bytes(bytes.data() + section.offset, section.size);
}

void refresh_turbovec_tvim_checksums(std::vector<uint8_t> & bytes, const TurboVecTvimLayout & layout) {
    CHECK(layout.checksum.offset + layout.checksum.size == bytes.size());
    write_u32_le_at(
        bytes,
        layout.checksum.offset,
        crc32c_update(0xffffffffu, bytes.data(), 32) ^ 0xffffffffu);
    write_u32_le_at(
        bytes,
        layout.checksum.offset + 4,
        crc32c_update(0xffffffffu, bytes.data() + layout.qparams.offset,
                      layout.qparams.size + layout.calibration.size) ^
            0xffffffffu);
    write_u32_le_at(
        bytes,
        layout.checksum.offset + 8,
        crc32c_update(0xffffffffu, bytes.data() + layout.vectors.offset, layout.vectors.size) ^
            0xffffffffu);
    write_u32_le_at(
        bytes,
        layout.checksum.offset + 12,
        crc32c_update(0xffffffffu, bytes.data() + layout.ids.offset, layout.ids.size) ^
            0xffffffffu);
}

// v3 snapshots load at dimensions past what legacy indexes add or search;
// at 40000 a packed row is wider than any TurboVec 1.x row.
void check_turbovec_oversized_snapshot_compatibility(int bits, int old_dim = 1032) {
    constexpr int current_dim = 16;
    constexpr int count = 2;
    CHECK(bits == 2 || bits == 4);

    const std::string suffix = std::to_string(bits) + "-" + std::to_string(old_dim);
    const std::string path =
        (std::filesystem::temp_directory_path() /
         ("ggml-vector-index-turbovec-old-dim-q" + suffix + ".tvim")).string();
    const std::string roundtrip_path =
        (std::filesystem::temp_directory_path() /
         ("ggml-vector-index-turbovec-old-dim-roundtrip-q" + suffix + ".tvim")).string();
    std::filesystem::remove(path);
    std::filesystem::remove(roundtrip_path);

    auto * current = create_legacy_turbovec(current_dim, bits);
    CHECK(current != nullptr);
    std::vector<float> current_vectors(static_cast<size_t>(count * current_dim), 0.0f);
    current_vectors[0] = 1.0f;
    current_vectors[static_cast<size_t>(current_dim) + 1] = 1.0f;
    const std::array<uint64_t, count> current_ids = {
        static_cast<uint64_t>(640000 + bits),
        static_cast<uint64_t>(640100 + bits),
    };
    CHECK(ggml_vec_index_add(current, current_vectors.data(), count, current_ids.data()) ==
          GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(current, path.c_str()) == GGML_VEC_INDEX_OK);
    ggml_vec_index_free(current);

    const std::vector<uint8_t> current_bytes = read_file_bytes(path);
    const TurboVecTvimLayout current_layout = parse_turbovec_tvim_layout(current_bytes, bits);
    CHECK(current_layout.n == count);

    TurboVecTvimLayout old_layout;
    old_layout.dim = old_dim;
    old_layout.n = count;
    old_layout.qparams = { 32, current_layout.qparams.size };
    old_layout.calibration = {
        old_layout.qparams.offset + old_layout.qparams.size,
        2 * static_cast<size_t>(old_dim) * sizeof(float),
    };
    old_layout.vectors = {
        old_layout.calibration.offset + old_layout.calibration.size,
        static_cast<size_t>(count * bits) * (static_cast<size_t>(old_dim) / 8),
    };
    old_layout.ids = {
        old_layout.vectors.offset + old_layout.vectors.size,
        count * sizeof(uint64_t),
    };
    old_layout.checksum = {
        old_layout.ids.offset + old_layout.ids.size,
        16,
    };

    std::vector<uint8_t> bytes(old_layout.checksum.offset + old_layout.checksum.size, 0);
    std::memcpy(bytes.data(), current_bytes.data(), 32);
    write_u32_le_at(bytes, 8, static_cast<uint32_t>(old_dim));
    write_u32_le_at(bytes, 28, static_cast<uint32_t>(old_layout.calibration.size));
    std::memcpy(
        bytes.data() + old_layout.qparams.offset,
        current_bytes.data() + current_layout.qparams.offset,
        current_layout.qparams.size);
    const size_t current_coordinate_bytes = static_cast<size_t>(current_dim) * sizeof(float);
    std::memcpy(
        bytes.data() + old_layout.calibration.offset,
        current_bytes.data() + current_layout.calibration.offset,
        current_coordinate_bytes);
    const size_t old_scale_offset =
        old_layout.calibration.offset + static_cast<size_t>(old_dim) * sizeof(float);
    std::memcpy(
        bytes.data() + old_scale_offset,
        current_bytes.data() + current_layout.calibration.offset + current_coordinate_bytes,
        current_coordinate_bytes);
    for (int coordinate = current_dim; coordinate < old_dim; ++coordinate) {
        write_u32_le_at(
            bytes,
            old_scale_offset + static_cast<size_t>(coordinate) * sizeof(float),
            float_bits(1.0f));
    }
    const size_t current_row_bytes = current_layout.vectors.size / count;
    const size_t old_row_bytes = old_layout.vectors.size / count;
    for (int row = 0; row < count; ++row) {
        std::memcpy(
            bytes.data() + old_layout.vectors.offset + static_cast<size_t>(row) * old_row_bytes,
            current_bytes.data() + current_layout.vectors.offset +
                static_cast<size_t>(row) * current_row_bytes,
            current_row_bytes);
    }
    std::memcpy(
        bytes.data() + old_layout.ids.offset,
        current_bytes.data() + current_layout.ids.offset,
        current_layout.ids.size);
    refresh_turbovec_tvim_checksums(bytes, old_layout);
    write_file_bytes(path, bytes);

    auto * loaded = ggml_vec_index_load(path.c_str());
    CHECK(loaded != nullptr);
    CHECK(ggml_vec_index_dim(loaded) == old_dim);
    CHECK(ggml_vec_index_bit_width(loaded) == bits);
    CHECK(ggml_vec_index_len(loaded) == count);
#ifdef GGML_VEC_INDEX_TEST_HOOKS
    CHECK(turbovec_blocked_hash_for_test(loaded) == 0);
    const size_t rotation_bytes_before_prepare = turbovec_rotation_cache_bytes_for_test();
    ggml_vec_index_prepare(loaded);
    CHECK(turbovec_rotation_cache_bytes_for_test() == rotation_bytes_before_prepare);
#endif

    std::vector<float> vector(static_cast<size_t>(old_dim), 0.0f);
    vector[0] = 1.0f;
    const uint64_t extra_id = static_cast<uint64_t>(641000 + bits);
    CHECK(ggml_vec_index_add(loaded, vector.data(), 1, &extra_id) == GGML_VEC_INDEX_E_INVALID_ARG);
    std::array<float, 1> scores{};
    std::array<uint64_t, 1> ids{};
    CHECK(ggml_vec_index_search(
        loaded, vector.data(), 1, 1, scores.data(), ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);
    CHECK(ggml_vec_index_remove(loaded, current_ids[0]) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_compact(loaded) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_len(loaded) == 1);
    CHECK(ggml_vec_index_contains(loaded, current_ids[1]) == 1);
#ifdef GGML_VEC_INDEX_TEST_HOOKS
    CHECK(turbovec_blocked_hash_for_test(loaded) == 0);
#endif

    CHECK(ggml_vec_index_write(loaded, roundtrip_path.c_str()) == GGML_VEC_INDEX_OK);
    auto * roundtrip = ggml_vec_index_load(roundtrip_path.c_str());
    CHECK(roundtrip != nullptr);
    CHECK(ggml_vec_index_dim(roundtrip) == old_dim);
    CHECK(ggml_vec_index_len(roundtrip) == 1);

    ggml_vec_index_free(roundtrip);
    ggml_vec_index_free(loaded);
    std::filesystem::remove(path);
    std::filesystem::remove(roundtrip_path);
}

float tqplus_golden_value(int row, int column);

#ifdef GGML_VEC_INDEX_TEST_HOOKS
void check_turbovec_numeric_helper_parity() {
    // The hash covers all 257 x and f table entries from Rust rand_distr 0.4.3.
    // Values are quantized in the test hook to avoid irrelevant libm bit noise.
    check_u64_equal(
        "Ziggurat x/f table hash",
        turbovec_ziggurat_table_hash_for_test(),
        UINT64_C(0xc9ae9c919c45a9e9));
    check_double_close(
        "Ziggurat x[0]",
        turbovec_ziggurat_x_for_test(0),
        3.910757959537090045,
        0.0);
    check_double_close(
        "Ziggurat x[1]",
        turbovec_ziggurat_x_for_test(1),
        3.654152885361008796,
        0.0);
    check_double_close("Ziggurat x[256]", turbovec_ziggurat_x_for_test(256), 0.0, 0.0);
    check_double_close(
        "Ziggurat f[0]",
        turbovec_ziggurat_f_for_test(0),
        0.0004774677645866553,
        1e-15);
    check_double_close(
        "Ziggurat f[1]",
        turbovec_ziggurat_f_for_test(1),
        0.001260285930498598,
        1e-15);
    check_double_close("Ziggurat f[256]", turbovec_ziggurat_f_for_test(256), 1.0, 0.0);

    // statrs 0.17.1 Beta::cdf and default ContinuousCDF::inverse_cdf behavior
    // used by Rust turbovec v0.9.0 TQ+ calibration.
    struct RegularizedBetaCase {
        const char * label;
        double x;
        double a;
        double b;
        double expected;
        double tolerance;
    };
    const RegularizedBetaCase beta_cases[] = {
        { "regularized beta symmetric", 0.5, 3.5, 3.5, 0.5, 1e-13 },
        { "regularized beta asymmetric", 0.2, 2.5, 5.0, 0.22997511934989717, 1e-12 },
        { "regularized beta dim8 lower tail", 0.25, 3.5, 3.5, 0.085235330393527292, 1e-12 },
        { "regularized beta dim128 q05", 0.427276611328125, 63.5, 63.5, 0.050022388729378635, 1e-10 },
        { "regularized beta dim1536 lower tail", 0.484405517578125, 767.5, 767.5, 0.11084377073802887, 1e-9 },
        { "regularized beta dim65536 lower tail", 0.4960784912109375, 32767.5, 32767.5, 0.022331190135124888, 1e-8 },
    };
    for (const RegularizedBetaCase & test : beta_cases) {
        check_double_close(
            test.label,
            turbovec_regularized_beta_for_test(test.x, test.a, test.b),
            test.expected,
            test.tolerance);
    }

    struct InverseBetaCase {
        int dim;
        double probability;
        double a;
        double quantile;
        double cdf;
        double cdf_tolerance;
    };
    const InverseBetaCase inverse_cases[] = {
        { 8, 0.01, 3.5, 0.125091552734375, 0.0099946943498694478, 1e-12 },
        { 8, 0.05, 3.5, 0.208892822265625, 0.049996830392321556, 1e-12 },
        { 8, 0.95, 3.5, 0.791107177734375, 0.9500031696076785, 1e-12 },
        { 8, 0.99, 3.5, 0.874908447265625, 0.99000530565013056, 1e-12 },
        { 128, 0.01, 63.5, 0.397674560546875, 0.0099974788182620299, 1e-10 },
        { 128, 0.05, 63.5, 0.427276611328125, 0.050022388729378635, 1e-10 },
        { 128, 0.95, 63.5, 0.572723388671875, 0.94997761127062141, 1e-10 },
        { 128, 0.99, 63.5, 0.602325439453125, 0.99000252118173793, 1e-10 },
        { 1536, 0.01, 767.5, 0.470306396484375, 0.00994511332916251, 1e-9 },
        { 1536, 0.05, 767.5, 0.479034423828125, 0.050162611666987017, 1e-9 },
        { 1536, 0.95, 767.5, 0.520965576171875, 0.94983738833301012, 1e-9 },
        { 1536, 0.99, 767.5, 0.529693603515625, 0.99005488667083696, 1e-9 },
        { 65536, 0.01, 32767.5, 0.495452880859375, 0.0099521630676677759, 1e-8 },
        { 65536, 0.05, 32767.5, 0.496795654296875, 0.050437841492088784, 1e-8 },
        { 65536, 0.95, 32767.5, 0.503204345703125, 0.94956215850791126, 1e-8 },
        { 65536, 0.99, 32767.5, 0.504547119140625, 0.99004783693233223, 1e-8 },
    };
    for (const InverseBetaCase & test : inverse_cases) {
        char label[96];
        std::snprintf(label, sizeof(label), "inverse beta dim%d p%.2f", test.dim, test.probability);
        const double quantile = turbovec_inverse_regularized_beta_for_test(test.probability, test.a);
        check_double_close(label, quantile, test.quantile, 0.0);
        std::snprintf(label, sizeof(label), "inverse beta dim%d p%.2f cdf", test.dim, test.probability);
        check_double_close(
            label,
            turbovec_regularized_beta_for_test(quantile, test.a, test.a),
            test.cdf,
            test.cdf_tolerance);
    }
}

void check_tqplus_deterministic_layout(
        int bits,
        uint64_t expected_codes_hash,
        uint64_t expected_scales_hash,
        uint64_t expected_shift_hash,
        uint64_t expected_tqscale_hash) {
    constexpr int dim = 128;
    constexpr int n = 1000;
    std::vector<float> vectors(static_cast<size_t>(n) * dim);
    std::vector<uint64_t> ids(n);
    for (int row = 0; row < n; ++row) {
        ids[static_cast<size_t>(row)] = static_cast<uint64_t>(row + 1);
        for (int column = 0; column < dim; ++column) {
            vectors[static_cast<size_t>(row) * dim + static_cast<size_t>(column)] =
                tqplus_golden_value(row, column);
        }
    }

    ggml_vec_index_t * index = create_legacy_turbovec(dim, bits);
    CHECK(index != nullptr);
    CHECK(ggml_vec_index_add(index, vectors.data(), n, ids.data()) == GGML_VEC_INDEX_OK);
    const uint64_t expected_blocked_hash = bits == 2 ?
        (kTurboVecInterleavedBlockLayout ? UINT64_C(0xcf52dbd26452d0c9) : UINT64_C(0x481ee4411871b4cd)) :
        (kTurboVecInterleavedBlockLayout ? UINT64_C(0x480945c9f53b88b9) : UINT64_C(0x001f1478c8b61a63));
    CHECK(turbovec_blocked_hash_for_test(index) == expected_blocked_hash);
    const std::string path =
        (std::filesystem::temp_directory_path() /
         ("ggml-vector-index-tqplus-q" + std::to_string(bits) + ".tvim")).string();
    std::filesystem::remove(path);
    CHECK(ggml_vec_index_write(index, path.c_str()) == GGML_VEC_INDEX_OK);
    const std::vector<uint8_t> bytes = read_file_bytes(path);
    CHECK(bytes.size() > 32);
    CHECK(bytes[4] == 3);
    CHECK(read_u32_le_from(bytes.data() + 28) == 2 * dim * sizeof(float));

    const size_t scales_offset = 32;
    const size_t scales_bytes = n * sizeof(float);
    const size_t shift_offset = scales_offset + scales_bytes;
    const size_t calibration_bytes = dim * sizeof(float);
    const size_t tqscale_offset = shift_offset + calibration_bytes;
    const size_t codes_offset = tqscale_offset + calibration_bytes;
    const size_t codes_bytes = static_cast<size_t>(n) * bits * (dim / 8);
    const uint64_t actual_codes_hash = fnv1a_bytes(bytes.data() + codes_offset, codes_bytes);
    const uint64_t actual_scales_hash = fnv1a_bytes(bytes.data() + scales_offset, scales_bytes);
    const uint64_t actual_shift_hash = fnv1a_bytes(bytes.data() + shift_offset, calibration_bytes);
    const uint64_t actual_tqscale_hash = fnv1a_bytes(bytes.data() + tqscale_offset, calibration_bytes);
    if (actual_codes_hash != expected_codes_hash) {
        std::fprintf(stderr, "FAIL TQ+ q%d codes hash: actual=0x%016llx expected=0x%016llx\n",
            bits,
            static_cast<unsigned long long>(actual_codes_hash),
            static_cast<unsigned long long>(expected_codes_hash));
    }
    CHECK(actual_codes_hash == expected_codes_hash);
    if (actual_scales_hash != expected_scales_hash) {
        std::fprintf(stderr, "FAIL TQ+ q%d scales hash: actual=0x%016llx expected=0x%016llx\n",
            bits,
            static_cast<unsigned long long>(actual_scales_hash),
            static_cast<unsigned long long>(expected_scales_hash));
    }
    CHECK(actual_scales_hash == expected_scales_hash);
    if (actual_shift_hash != expected_shift_hash) {
        std::fprintf(stderr, "FAIL TQ+ q%d shift hash: actual=0x%016llx expected=0x%016llx\n",
            bits,
            static_cast<unsigned long long>(actual_shift_hash),
            static_cast<unsigned long long>(expected_shift_hash));
    }
    CHECK(actual_shift_hash == expected_shift_hash);
    if (actual_tqscale_hash != expected_tqscale_hash) {
        std::fprintf(stderr, "FAIL TQ+ q%d tqscale hash: actual=0x%016llx expected=0x%016llx\n",
            bits,
            static_cast<unsigned long long>(actual_tqscale_hash),
            static_cast<unsigned long long>(expected_tqscale_hash));
    }
    CHECK(actual_tqscale_hash == expected_tqscale_hash);

    auto * loaded = ggml_vec_index_load(path.c_str());
    CHECK(loaded != nullptr);
    CHECK(turbovec_blocked_hash_for_test(loaded) == expected_blocked_hash);
    std::array<float, 3 * dim> queries{};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < dim; ++column) {
            const double x = static_cast<double>(row * 2 + 1);
            const double y = static_cast<double>(column + 1);
            queries[static_cast<size_t>(row) * dim + static_cast<size_t>(column)] =
                static_cast<float>(
                    0.63 * std::sin(0.017 * x * y + 0.29) +
                    0.31 * std::cos(0.041 * (x + 3.0) * (y + 1.0)) +
                    0.06 * std::sin(0.097 * (x + y)));
        }
    }
    CHECK(turbovec_query_rotation_hash_for_test(queries.data(), 3, dim) ==
        UINT64_C(0x22582f085fd79768));
    std::array<float, dim> tqplus_shift{};
    std::array<float, dim> tqplus_scale{};
    std::memcpy(tqplus_shift.data(), bytes.data() + shift_offset, calibration_bytes);
    std::memcpy(tqplus_scale.data(), bytes.data() + tqscale_offset, calibration_bytes);
    uint32_t lut_scale_bits = 0;
    uint32_t lut_bias_bits = 0;
    const uint64_t lut_hash = turbovec_lut_hash_for_test(
        queries.data(),
        tqplus_shift.data(),
        tqplus_scale.data(),
        bits,
        3,
        dim,
        &lut_scale_bits,
        &lut_bias_bits);
    CHECK(lut_hash == (bits == 2 ?
        UINT64_C(0x3b105f838666dbbb) :
        UINT64_C(0x9691906f2a148805)));
    check_bits_within_ulp(
        "TQ+ LUT scale",
        lut_scale_bits,
        bits == 2 ? 0x3ba9233c : 0x3bb920ca,
        kFloatParityMaxUlpDiff);
    check_bits_within_ulp(
        "TQ+ LUT bias",
        lut_bias_bits,
        bits == 2 ? 0xc0fabc4e : 0xc1606205,
        kFloatParityMaxUlpDiff);
    CHECK(turbovec_codebook_hash_for_test(bits, dim) == (bits == 2 ?
        UINT64_C(0xa37c605fe8acd601) :
        UINT64_C(0xd74197c1c7f95b91)));
    static constexpr std::array<uint32_t, 9> q2_score_bits = {
        0x4223a35c, 0x420f51fc, 0x420905a1,
        0x41f4e33a, 0x41bd8451, 0x41bd7b9f,
        0x4211fc2c, 0x41cd812e, 0x41c1e660,
    };
    static constexpr std::array<uint64_t, 9> q2_ids = {
        1, 740, 370, 3, 372, 742, 5, 375, 744,
    };
    static constexpr std::array<uint32_t, 9> q4_score_bits = {
        0x422312e8, 0x42108e97, 0x42020020,
        0x41f4f68c, 0x41c97248, 0x41b70ac3,
        0x4212e92f, 0x41cb8eec, 0x41cb0d33,
    };
    static constexpr std::array<uint64_t, 9> q4_ids = {
        1, 740, 370, 3, 742, 372, 5, 375, 744,
    };
    const auto & expected_score_bits = bits == 2 ? q2_score_bits : q4_score_bits;
    const auto & expected_ids = bits == 2 ? q2_ids : q4_ids;
    std::array<float, 9> scores{};
    std::array<uint64_t, 9> results{};
    CHECK(ggml_vec_index_search(
        loaded, queries.data(), 3, 3, scores.data(), results.data()) == GGML_VEC_INDEX_OK);
    for (size_t i = 0; i < scores.size(); ++i) {
        const uint32_t actual_score_bits = float_bits(scores[i]);
        // Final scores can differ slightly by compiler/backend; persisted
        // codebooks and calibration above remain bitwise/hash checked.
        check_float_bits_close(
            "TQ+ score",
            actual_score_bits,
            expected_score_bits[i],
            kTqplusScoreMaxUlpDiff,
            kTqplusScoreRelTolerance,
            kTqplusScoreAbsTolerance);
        CHECK(results[i] == expected_ids[i]);
    }
    ggml_vec_index_free(loaded);
    ggml_vec_index_free(index);
    std::filesystem::remove(path);
}
#endif

float tqplus_golden_value(int row, int column) {
    const double x = static_cast<double>(row + 1);
    const double y = static_cast<double>(column + 1);
    return static_cast<float>(
        0.63 * std::sin(0.017 * x * y + 0.47) +
        0.31 * std::cos(0.041 * (x + 3.0) * (y + 1.0)) +
        0.06 * std::sin(0.097 * (x + y)));
}

#ifdef GGML_VEC_INDEX_TEST_HOOKS
void check_turbovec_blocked_scalar_scores(
        int bits,
        int dim,
        int n,
        int n_queries,
        int n_scalar_queries = -1) {
    if (n_scalar_queries < 0) {
        n_scalar_queries = n_queries;
    }
    CHECK(n_scalar_queries >= 0);
    CHECK(n_scalar_queries <= n_queries);
    auto * blocked = bits == 2 ?
        ggml_vec_index_create_turbovec_q2(dim) :
        ggml_vec_index_create_turbovec_q4(dim);
    auto * scalar = bits == 2 ?
        ggml_vec_index_create_turbovec_q2(dim) :
        ggml_vec_index_create_turbovec_q4(dim);
    CHECK(blocked != nullptr);
    CHECK(scalar != nullptr);

    const uint64_t id_base = static_cast<uint64_t>(30000 + bits * 1000 + dim + n);
    std::vector<uint64_t> ids(static_cast<size_t>(n));
    std::vector<float> vectors(static_cast<size_t>(n) * dim);
    for (int row = 0; row < n; ++row) {
        ids[static_cast<size_t>(row)] = id_base + static_cast<uint64_t>(row);
        for (int col = 0; col < dim; ++col) {
            const double x = static_cast<double>(row + 1);
            const double y = static_cast<double>(col + 3);
            vectors[static_cast<size_t>(row) * dim + static_cast<size_t>(col)] =
                static_cast<float>(
                    0.55 * std::sin(0.013 * x * y + 0.17) +
                    0.35 * std::cos(0.019 * (x + 5.0) * (y + 1.0)) +
                    0.10 * std::sin(0.071 * (x + y)));
        }
    }

    CHECK(ggml_vec_index_add(blocked, vectors.data(), n, ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_add(scalar, vectors.data(), n, ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(turbovec_blocked_hash_for_test(blocked) != 0);
    turbovec_clear_blocked_for_test(scalar);
    CHECK(turbovec_blocked_hash_for_test(scalar) == 0);

    std::vector<float> queries(static_cast<size_t>(n_queries) * dim);
    for (int row = 0; row < n_queries; ++row) {
        for (int col = 0; col < dim; ++col) {
            const double x = static_cast<double>(row + 2);
            const double y = static_cast<double>(col + 7);
            queries[static_cast<size_t>(row) * dim + static_cast<size_t>(col)] =
                static_cast<float>(
                    0.48 * std::cos(0.023 * x * y + 0.31) +
                    0.41 * std::sin(0.037 * (x + 3.0) * (y + 2.0)) +
                    0.11 * std::cos(0.083 * (x + y)));
        }
    }
    CHECK(turbovec_query_rotation_max_abs_diff_for_test(
        queries.data(), n_queries, dim) <= 1e-5);

    std::vector<float> blocked_scores(static_cast<size_t>(n_queries) * n);
    std::vector<float> scalar_scores(static_cast<size_t>(n_scalar_queries) * n);
    std::vector<uint64_t> blocked_ids(static_cast<size_t>(n_queries) * n);
    std::vector<uint64_t> scalar_ids(static_cast<size_t>(n_scalar_queries) * n);
    CHECK(ggml_vec_index_search(
        blocked, queries.data(), n_queries, n, blocked_scores.data(), blocked_ids.data()) ==
        GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_search(
        scalar, queries.data(), n_scalar_queries, n, scalar_scores.data(), scalar_ids.data()) ==
        GGML_VEC_INDEX_OK);

    std::vector<float> blocked_by_row(static_cast<size_t>(n));
    std::vector<float> scalar_by_row(static_cast<size_t>(n));
    std::vector<float> single_scores(static_cast<size_t>(n));
    std::vector<uint64_t> single_ids(static_cast<size_t>(n));
    std::vector<float> single_by_row(static_cast<size_t>(n));
    for (int query = 0; query < n_queries; ++query) {
        std::fill(blocked_by_row.begin(), blocked_by_row.end(), std::numeric_limits<float>::quiet_NaN());
        std::fill(scalar_by_row.begin(), scalar_by_row.end(), std::numeric_limits<float>::quiet_NaN());
        for (int rank = 0; rank < n; ++rank) {
            const size_t offset = static_cast<size_t>(query) * n + static_cast<size_t>(rank);
            CHECK(blocked_ids[offset] >= id_base);
            CHECK(blocked_ids[offset] < id_base + static_cast<uint64_t>(n));
            blocked_by_row[static_cast<size_t>(blocked_ids[offset] - id_base)] = blocked_scores[offset];
            if (query < n_scalar_queries) {
                CHECK(scalar_ids[offset] >= id_base);
                CHECK(scalar_ids[offset] < id_base + static_cast<uint64_t>(n));
                scalar_by_row[static_cast<size_t>(scalar_ids[offset] - id_base)] = scalar_scores[offset];
            }
        }
        CHECK(ggml_vec_index_search(
            blocked,
            queries.data() + static_cast<size_t>(query) * dim,
            1,
            n,
            single_scores.data(),
            single_ids.data()) == GGML_VEC_INDEX_OK);
        std::fill(single_by_row.begin(), single_by_row.end(), std::numeric_limits<float>::quiet_NaN());
        for (int rank = 0; rank < n; ++rank) {
            CHECK(single_ids[static_cast<size_t>(rank)] >= id_base);
            CHECK(single_ids[static_cast<size_t>(rank)] < id_base + static_cast<uint64_t>(n));
            single_by_row[static_cast<size_t>(single_ids[static_cast<size_t>(rank)] - id_base)] =
                single_scores[static_cast<size_t>(rank)];
        }
        for (int row = 0; row < n; ++row) {
            const float blocked_score = blocked_by_row[static_cast<size_t>(row)];
            const float single_score = single_by_row[static_cast<size_t>(row)];
            CHECK(std::isfinite(blocked_score));
            CHECK(float_bits(blocked_score) == float_bits(single_score));
            if (query < n_scalar_queries) {
                const float scalar_score = scalar_by_row[static_cast<size_t>(row)];
                CHECK(std::isfinite(scalar_score));
                const float tolerance = std::max(
                    1e-4f * std::fabs(scalar_score),
                    1e-4f);
                const float drift = std::fabs(blocked_score - scalar_score);
                if (!(drift <= tolerance)) {
                    std::fprintf(
                        stderr,
                        "FAIL TurboVec q%d blocked/scalar drift: dim=%d n=%d query=%d row=%d drift=%g tolerance=%g\n",
                        bits,
                        dim,
                        n,
                        query,
                        row,
                        static_cast<double>(drift),
                        static_cast<double>(tolerance));
                    std::exit(1);
                }
            }
        }
    }

    ggml_vec_index_free(scalar);
    ggml_vec_index_free(blocked);
}

void check_turbovec_fused_batch_scores(int bits) {
    constexpr int dim = 256;
    constexpr int n = 2048;
    constexpr int n_queries = 17;
    constexpr int k = 10;
    auto * index = bits == 2 ?
        ggml_vec_index_create_turbovec_q2(dim) :
        ggml_vec_index_create_turbovec_q4(dim);
    CHECK(index != nullptr);

    std::vector<uint64_t> ids(static_cast<size_t>(n));
    std::vector<float> vectors(static_cast<size_t>(n) * dim);
    for (int row = 0; row < n; ++row) {
        ids[static_cast<size_t>(row)] = static_cast<uint64_t>(40000 + row);
        for (int col = 0; col < dim; ++col) {
            const double x = static_cast<double>(row + 1);
            const double y = static_cast<double>(col + 3);
            vectors[static_cast<size_t>(row) * dim + static_cast<size_t>(col)] =
                static_cast<float>(
                    0.55 * std::sin(0.013 * x * y + 0.17) +
                    0.35 * std::cos(0.019 * (x + 5.0) * (y + 1.0)) +
                    0.10 * std::sin(0.071 * (x + y)));
        }
    }
    CHECK(ggml_vec_index_add(index, vectors.data(), n, ids.data()) == GGML_VEC_INDEX_OK);

    std::vector<float> queries(static_cast<size_t>(n_queries) * dim);
    for (int row = 0; row < n_queries; ++row) {
        for (int col = 0; col < dim; ++col) {
            const double x = static_cast<double>(row + 2);
            const double y = static_cast<double>(col + 7);
            queries[static_cast<size_t>(row) * dim + static_cast<size_t>(col)] =
                static_cast<float>(
                    0.48 * std::cos(0.023 * x * y + 0.31) +
                    0.41 * std::sin(0.037 * (x + 3.0) * (y + 2.0)) +
                    0.11 * std::cos(0.083 * (x + y)));
        }
    }

    std::vector<float> batch_scores(static_cast<size_t>(n_queries) * k);
    std::vector<uint64_t> batch_ids(static_cast<size_t>(n_queries) * k);
    turbovec_reset_block_score_call_count_for_test();
    CHECK(ggml_vec_index_search(
        index,
        queries.data(),
        n_queries,
        k,
        batch_scores.data(),
        batch_ids.data()) == GGML_VEC_INDEX_OK);
#if defined(__aarch64__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
    const int64_t n_blocks = (n + 31) / 32;
    const int64_t expected_calls =
        (n_queries / 4 + n_queries % 4) * n_blocks;
    CHECK(turbovec_block_score_call_count_for_test() == expected_calls);
#endif

    std::array<float, k> single_scores{};
    std::array<uint64_t, k> single_ids{};
    for (int query = 0; query < n_queries; ++query) {
        CHECK(ggml_vec_index_search(
            index,
            queries.data() + static_cast<size_t>(query) * dim,
            1,
            k,
            single_scores.data(),
            single_ids.data()) == GGML_VEC_INDEX_OK);
        for (int rank = 0; rank < k; ++rank) {
            const size_t offset =
                static_cast<size_t>(query) * k + static_cast<size_t>(rank);
            CHECK(batch_ids[offset] == single_ids[static_cast<size_t>(rank)]);
            CHECK(float_bits(batch_scores[offset]) ==
                float_bits(single_scores[static_cast<size_t>(rank)]));
        }
    }
    ggml_vec_index_free(index);
}
#endif

void fill_turbovec_regression_vectors(std::vector<float> & vectors, int n, int dim) {
    CHECK(vectors.size() == static_cast<size_t>(n) * static_cast<size_t>(dim));
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < dim; ++col) {
            const double x = static_cast<double>(row + 1);
            const double y = static_cast<double>(col + 1);
            vectors[static_cast<size_t>(row) * dim + static_cast<size_t>(col)] =
                static_cast<float>(
                    0.51 * std::sin(0.011 * x * y + 0.37) +
                    0.34 * std::cos(0.017 * (x + 3.0) * (y + 5.0)) +
                    0.15 * std::sin(0.043 * (x + y + 7.0)));
        }
    }
}

void set_distinct_turbovec_vector(float * vector, int dim) {
    for (int col = 0; col < dim; ++col) {
        vector[col] = (col % 3 == 0) ? 1.0f : ((col % 3 == 1) ? -0.35f : 0.12f);
    }
}

void check_search_contains(
        ggml_vec_index_t * idx,
        const float * query,
        int k,
        uint64_t expected_id) {
    std::vector<float> scores(static_cast<size_t>(k));
    std::vector<uint64_t> out(static_cast<size_t>(k));
    CHECK(ggml_vec_index_search(
        idx, query, 1, k, scores.data(), out.data()) == GGML_VEC_INDEX_OK);
    CHECK(std::find(out.begin(), out.end(), expected_id) != out.end());
}

void check_ivf_contains(
        ggml_vec_index_t * idx,
        const float * query,
        int k,
        int nprobe,
        uint64_t expected_id) {
    std::vector<float> scores(static_cast<size_t>(k));
    std::vector<uint64_t> out(static_cast<size_t>(k));
    CHECK(ggml_vec_index_search_ivf(
        idx, query, 1, k, nprobe, scores.data(), out.data()) == GGML_VEC_INDEX_OK);
    CHECK(std::find(out.begin(), out.end(), expected_id) != out.end());
}

void check_turbovec_rounding_mode_persistence(int bits) {
    constexpr int dim = 136;
    constexpr int n = 1000;
    constexpr int k = 4;
    CHECK(bits == 2 || bits == 4);
    const int saved_rounding = std::fegetround();
    CHECK(saved_rounding != -1);

    std::vector<float> vectors(static_cast<size_t>(n) * dim);
    fill_turbovec_regression_vectors(vectors, n, dim);
    std::vector<uint64_t> ids(static_cast<size_t>(n));
    for (int row = 0; row < n; ++row) {
        ids[static_cast<size_t>(row)] = static_cast<uint64_t>(700000 + bits * 10000 + row);
    }

    struct RoundingSnapshot {
        std::vector<uint8_t> bytes;
        std::array<float, 4> scores;
        std::array<uint64_t, 4> ids;
    };

    auto make_snapshot = [&](int rounding_mode, const char * suffix) {
        const std::string path =
            (std::filesystem::temp_directory_path() /
             ("ggml-vector-index-turbovec-rounding-q" + std::to_string(bits) + "-" +
              suffix + ".tvim")).string();
        std::filesystem::remove(path);

        CHECK(std::fesetround(rounding_mode) == 0);
        auto * tv = create_legacy_turbovec(dim, bits);
        CHECK(tv != nullptr);
        ggml_vec_index_prepare(tv);
        CHECK(ggml_vec_index_add(tv, vectors.data(), n, ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_write(tv, path.c_str()) == GGML_VEC_INDEX_OK);
        ggml_vec_index_free(tv);
        CHECK(std::fesetround(saved_rounding) == 0);

        RoundingSnapshot snapshot;
        snapshot.bytes = read_file_bytes(path);
        auto * loaded = ggml_vec_index_load(path.c_str());
        CHECK(loaded != nullptr);
        CHECK(ggml_vec_index_search(
            loaded, vectors.data(), 1, k, snapshot.scores.data(), snapshot.ids.data()) ==
            GGML_VEC_INDEX_OK);
        ggml_vec_index_free(loaded);
        std::filesystem::remove(path);
        return snapshot;
    };

    const RoundingSnapshot downward = make_snapshot(FE_DOWNWARD, "downward");
    const RoundingSnapshot upward = make_snapshot(FE_UPWARD, "upward");
    CHECK(downward.bytes == upward.bytes);
    CHECK(downward.ids == upward.ids);
    for (size_t i = 0; i < downward.scores.size(); ++i) {
        CHECK(float_bits(downward.scores[i]) == float_bits(upward.scores[i]));
    }
    CHECK(std::fesetround(saved_rounding) == 0);
}

float score_for_id(
        const std::vector<uint64_t> & ids,
        const std::vector<float> & scores,
        uint64_t id) {
    const auto it = std::find(ids.begin(), ids.end(), id);
    CHECK(it != ids.end());
    return scores[static_cast<size_t>(it - ids.begin())];
}

#ifdef GGML_VEC_INDEX_TEST_HOOKS
void check_loaded_blocked_scores_match_scalar(
        const std::string & path,
        const std::vector<float> & queries,
        int n_queries,
        const std::vector<uint64_t> & live_ids) {
    auto * blocked = ggml_vec_index_load(path.c_str());
    auto * scalar = ggml_vec_index_load(path.c_str());
    CHECK(blocked != nullptr);
    CHECK(scalar != nullptr);
    CHECK(turbovec_blocked_hash_for_test(blocked) != 0);
    turbovec_clear_blocked_for_test(scalar);
    CHECK(turbovec_blocked_hash_for_test(scalar) == 0);

    const int n_live = static_cast<int>(live_ids.size());
    std::vector<float> blocked_scores(static_cast<size_t>(n_queries) * n_live);
    std::vector<float> scalar_scores(static_cast<size_t>(n_queries) * n_live);
    std::vector<uint64_t> blocked_ids(static_cast<size_t>(n_queries) * n_live);
    std::vector<uint64_t> scalar_ids(static_cast<size_t>(n_queries) * n_live);
    CHECK(ggml_vec_index_search(
        blocked, queries.data(), n_queries, n_live, blocked_scores.data(), blocked_ids.data()) ==
        GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_search(
        scalar, queries.data(), n_queries, n_live, scalar_scores.data(), scalar_ids.data()) ==
        GGML_VEC_INDEX_OK);

    for (int query = 0; query < n_queries; ++query) {
        const size_t base = static_cast<size_t>(query) * n_live;
        std::vector<uint64_t> blocked_row(
            blocked_ids.begin() + static_cast<std::ptrdiff_t>(base),
            blocked_ids.begin() + static_cast<std::ptrdiff_t>(base + n_live));
        std::vector<uint64_t> scalar_row(
            scalar_ids.begin() + static_cast<std::ptrdiff_t>(base),
            scalar_ids.begin() + static_cast<std::ptrdiff_t>(base + n_live));
        std::vector<float> blocked_score_row(
            blocked_scores.begin() + static_cast<std::ptrdiff_t>(base),
            blocked_scores.begin() + static_cast<std::ptrdiff_t>(base + n_live));
        std::vector<float> scalar_score_row(
            scalar_scores.begin() + static_cast<std::ptrdiff_t>(base),
            scalar_scores.begin() + static_cast<std::ptrdiff_t>(base + n_live));
        for (uint64_t id : live_ids) {
            const float blocked_score = score_for_id(blocked_row, blocked_score_row, id);
            const float scalar_score = score_for_id(scalar_row, scalar_score_row, id);
            CHECK(std::isfinite(blocked_score));
            CHECK(std::isfinite(scalar_score));
            const float tolerance = std::max(
                1e-4f * std::fabs(scalar_score),
                1e-4f);
            CHECK(std::fabs(blocked_score - scalar_score) <= tolerance);
        }
    }

    ggml_vec_index_free(scalar);
    ggml_vec_index_free(blocked);
}

void check_turbovec_mutation_cache_regression(int bits) {
    constexpr int dim = 128;
    constexpr int n_initial = 1000;
    constexpr int n_total = n_initial + 1;
    constexpr int n_lists = 16;
    constexpr int remove_row = 37;
    CHECK(bits == 2 || bits == 4);

    std::vector<float> vectors(static_cast<size_t>(n_total) * dim);
    fill_turbovec_regression_vectors(vectors, n_total, dim);
    set_distinct_turbovec_vector(vectors.data() + static_cast<size_t>(n_initial) * dim, dim);
    std::vector<uint64_t> ids(static_cast<size_t>(n_total));
    for (int row = 0; row < n_total; ++row) {
        ids[static_cast<size_t>(row)] =
            static_cast<uint64_t>(500000 + bits * 10000 + row);
    }

    auto * tv = create_legacy_turbovec(dim, bits);
    CHECK(tv != nullptr);
    CHECK(ggml_vec_index_add(tv, vectors.data(), n_initial, ids.data()) == GGML_VEC_INDEX_OK);
    const uint64_t rotation_hash = turbovec_rotation_hash_for_test(dim);
    const uint64_t initial_blocked_hash = turbovec_blocked_hash_for_test(tv);
    CHECK(rotation_hash != 0);
    CHECK(initial_blocked_hash != 0);

    const std::string initial_path =
        (std::filesystem::temp_directory_path() /
         ("ggml-vector-index-turbovec-mutation-initial-q" + std::to_string(bits) + ".tvim")).string();
    const std::string final_path =
        (std::filesystem::temp_directory_path() /
         ("ggml-vector-index-turbovec-mutation-final-q" + std::to_string(bits) + ".tvim")).string();
    std::filesystem::remove(initial_path);
    std::filesystem::remove(final_path);

    CHECK(ggml_vec_index_write(tv, initial_path.c_str()) == GGML_VEC_INDEX_OK);
    const std::vector<uint8_t> initial_bytes = read_file_bytes(initial_path);
    const TurboVecTvimLayout initial_layout =
        parse_turbovec_tvim_layout(initial_bytes, bits);
    const uint64_t initial_calibration_hash =
        section_hash(initial_bytes, initial_layout.calibration);
    CHECK(initial_calibration_hash != 0);

    CHECK(ggml_vec_index_build_ivf(tv, n_lists, 2) == GGML_VEC_INDEX_OK);
    check_ivf_contains(tv, vectors.data(), 8, n_lists, ids[0]);

    CHECK(ggml_vec_index_remove(tv, ids[remove_row]) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_contains(tv, ids[remove_row]) == 0);
    {
        std::array<float, 1> scores{};
        std::array<uint64_t, 1> out{};
        CHECK(ggml_vec_index_search_ivf(
            tv, vectors.data(), 1, 1, n_lists, scores.data(), out.data()) ==
            GGML_VEC_INDEX_E_INVALID_ARG);
    }

    CHECK(ggml_vec_index_compact(tv) == GGML_VEC_INDEX_OK);
    const uint64_t compacted_blocked_hash = turbovec_blocked_hash_for_test(tv);
    CHECK(compacted_blocked_hash != 0);
    CHECK(compacted_blocked_hash != initial_blocked_hash);
    CHECK(turbovec_rotation_hash_for_test(dim) == rotation_hash);
    CHECK(ggml_vec_index_build_ivf(tv, n_lists, 2) == GGML_VEC_INDEX_OK);
    check_ivf_contains(tv, vectors.data(), 8, n_lists, ids[0]);

    const uint64_t replacement_id = ids[static_cast<size_t>(n_initial)];
    CHECK(ggml_vec_index_add(
        tv,
        vectors.data() + static_cast<size_t>(n_initial) * dim,
        1,
        &replacement_id) == GGML_VEC_INDEX_OK);
    const uint64_t final_blocked_hash = turbovec_blocked_hash_for_test(tv);
    CHECK(final_blocked_hash != 0);
    CHECK(final_blocked_hash != compacted_blocked_hash);
    CHECK(turbovec_rotation_hash_for_test(dim) == rotation_hash);
    {
        std::array<float, 1> scores{};
        std::array<uint64_t, 1> out{};
        CHECK(ggml_vec_index_search_ivf(
            tv,
            vectors.data() + static_cast<size_t>(n_initial) * dim,
            1,
            1,
            n_lists,
            scores.data(),
            out.data()) == GGML_VEC_INDEX_E_INVALID_ARG);
    }

    CHECK(ggml_vec_index_build_ivf(tv, n_lists, 2) == GGML_VEC_INDEX_OK);
    check_search_contains(
        tv,
        vectors.data() + static_cast<size_t>(n_initial) * dim,
        8,
        replacement_id);
    check_ivf_contains(
        tv,
        vectors.data() + static_cast<size_t>(n_initial) * dim,
        8,
        n_lists,
        replacement_id);

    CHECK(ggml_vec_index_write(tv, final_path.c_str()) == GGML_VEC_INDEX_OK);
    const std::vector<uint8_t> final_bytes = read_file_bytes(final_path);
    const TurboVecTvimLayout final_layout = parse_turbovec_tvim_layout(final_bytes, bits);
    CHECK(section_hash(final_bytes, final_layout.calibration) == initial_calibration_hash);

    auto * loaded = ggml_vec_index_load(final_path.c_str());
    CHECK(loaded != nullptr);
    CHECK(turbovec_blocked_hash_for_test(loaded) == final_blocked_hash);
    CHECK(turbovec_rotation_hash_for_test(dim) == rotation_hash);
    check_search_contains(
        loaded,
        vectors.data() + static_cast<size_t>(n_initial) * dim,
        8,
        replacement_id);
    CHECK(ggml_vec_index_build_ivf(loaded, n_lists, 2) == GGML_VEC_INDEX_OK);
    check_ivf_contains(
        loaded,
        vectors.data() + static_cast<size_t>(n_initial) * dim,
        8,
        n_lists,
        replacement_id);

    std::vector<uint64_t> live_ids;
    live_ids.reserve(n_initial);
    for (int row = 0; row < n_initial; ++row) {
        if (row != remove_row) {
            live_ids.push_back(ids[static_cast<size_t>(row)]);
        }
    }
    live_ids.push_back(replacement_id);
    std::vector<float> queries(static_cast<size_t>(3) * dim);
    std::memcpy(queries.data(), vectors.data(), static_cast<size_t>(dim) * sizeof(float));
    std::memcpy(
        queries.data() + dim,
        vectors.data() + static_cast<size_t>(n_initial / 2) * dim,
        static_cast<size_t>(dim) * sizeof(float));
    std::memcpy(
        queries.data() + static_cast<size_t>(2) * dim,
        vectors.data() + static_cast<size_t>(n_initial) * dim,
        static_cast<size_t>(dim) * sizeof(float));
    check_loaded_blocked_scores_match_scalar(final_path, queries, 3, live_ids);

    ggml_vec_index_free(loaded);
    ggml_vec_index_free(tv);
    std::filesystem::remove(initial_path);
    std::filesystem::remove(final_path);
}

void check_turbovec_incremental_block_repacking() {
    constexpr int dim = 64;
    constexpr int n = 45;
    std::vector<float> vectors(static_cast<size_t>(dim) * n);
    std::vector<uint64_t> ids(static_cast<size_t>(n));
    for (int row = 0; row < n; ++row) {
        ids[static_cast<size_t>(row)] = static_cast<uint64_t>(9600 + row);
        for (int col = 0; col < dim; ++col) {
            vectors[static_cast<size_t>(row) * dim + static_cast<size_t>(col)] =
                static_cast<float>(std::sin(0.01 * static_cast<double>((row + 1) * (col + 3))));
        }
    }

    for (const int bits : { 2, 4 }) {
        auto * batch = bits == 2 ?
            ggml_vec_index_create_turbovec_q2(dim) :
            ggml_vec_index_create_turbovec_q4(dim);
        auto * incremental = bits == 2 ?
            ggml_vec_index_create_turbovec_q2(dim) :
            ggml_vec_index_create_turbovec_q4(dim);
        CHECK(batch != nullptr);
        CHECK(incremental != nullptr);
        CHECK(ggml_vec_index_add(batch, vectors.data(), n, ids.data()) == GGML_VEC_INDEX_OK);
        for (int row = 0; row < n; ++row) {
            CHECK(ggml_vec_index_add(
                incremental,
                vectors.data() + static_cast<size_t>(row) * dim,
                1,
                ids.data() + row) == GGML_VEC_INDEX_OK);
        }
        const uint64_t batch_hash = turbovec_blocked_hash_for_test(batch);
        CHECK(batch_hash != 0);
        CHECK(turbovec_blocked_hash_for_test(incremental) == batch_hash);
        ggml_vec_index_free(incremental);
        ggml_vec_index_free(batch);
    }
}

void check_turbovec_sparse_filter_block_selection() {
    constexpr int dim = 128;
    constexpr int n = 96;
    std::vector<float> vectors(static_cast<size_t>(dim) * n);
    std::vector<uint64_t> ids(static_cast<size_t>(n));
    for (int row = 0; row < n; ++row) {
        ids[static_cast<size_t>(row)] = static_cast<uint64_t>(9900 + row);
        for (int col = 0; col < dim; ++col) {
            const double x = static_cast<double>(row + 1);
            const double y = static_cast<double>(col + 5);
            vectors[static_cast<size_t>(row) * dim + static_cast<size_t>(col)] =
                static_cast<float>(
                    0.44 * std::sin(0.017 * x * y + 0.13) +
                    0.38 * std::cos(0.029 * (x + 2.0) * (y + 1.0)) +
                    0.18 * std::sin(0.061 * (x + y)));
        }
    }

    for (const int bits : { 2, 4 }) {
        auto * index = bits == 2 ?
            ggml_vec_index_create_turbovec_q2(dim) :
            ggml_vec_index_create_turbovec_q4(dim);
        CHECK(index != nullptr);
        CHECK(ggml_vec_index_add(index, vectors.data(), n, ids.data()) == GGML_VEC_INDEX_OK);

        std::vector<float> all_scores(static_cast<size_t>(n));
        std::vector<uint64_t> all_ids(static_cast<size_t>(n));
        turbovec_reset_block_score_call_count_for_test();
        CHECK(ggml_vec_index_search(
            index, vectors.data(), 1, n, all_scores.data(), all_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(turbovec_block_score_call_count_for_test() == 3);

        const std::array<uint64_t, 2> allowed = { ids[5], ids[63] };
        std::array<float, 2> scores{};
        std::array<uint64_t, 2> out{};
        turbovec_reset_block_score_call_count_for_test();
        CHECK(ggml_vec_index_search_filtered(
            index,
            vectors.data(),
            1,
            2,
            allowed.data(),
            static_cast<int>(allowed.size()),
            scores.data(),
            out.data()) == GGML_VEC_INDEX_OK);
        CHECK(turbovec_block_score_call_count_for_test() == 2);
        ggml_vec_index_free(index);
    }
}
#endif

void check_turbovec_v3_corruption_rejected(int bits) {
    constexpr int dim = 128;
    constexpr int n = 1000;
    CHECK(bits == 2 || bits == 4);

    std::vector<float> vectors(static_cast<size_t>(n) * dim);
    std::vector<uint64_t> ids(static_cast<size_t>(n));
    fill_turbovec_regression_vectors(vectors, n, dim);
    for (int row = 0; row < n; ++row) {
        ids[static_cast<size_t>(row)] =
            static_cast<uint64_t>(600000 + bits * 10000 + row);
    }

    auto * tv = create_legacy_turbovec(dim, bits);
    CHECK(tv != nullptr);
    CHECK(ggml_vec_index_add(tv, vectors.data(), n, ids.data()) == GGML_VEC_INDEX_OK);

    const std::string path =
        (std::filesystem::temp_directory_path() /
         ("ggml-vector-index-turbovec-v3-corrupt-q" + std::to_string(bits) + ".tvim")).string();
    const std::string corrupt_path =
        (std::filesystem::temp_directory_path() /
         ("ggml-vector-index-turbovec-v3-corrupt-mutated-q" + std::to_string(bits) + ".tvim")).string();
    std::filesystem::remove(path);
    std::filesystem::remove(corrupt_path);
    CHECK(ggml_vec_index_write(tv, path.c_str()) == GGML_VEC_INDEX_OK);

    const std::vector<uint8_t> bytes = read_file_bytes(path);
    const TurboVecTvimLayout layout = parse_turbovec_tvim_layout(bytes, bits);
    expect_corrupt_load_fails(path, corrupt_path, [layout](std::vector<uint8_t> & corrupt) {
        corrupt[layout.qparams.offset] ^= 0x01;
    });
    expect_corrupt_load_fails(path, corrupt_path, [layout](std::vector<uint8_t> & corrupt) {
        corrupt[layout.calibration.offset] ^= 0x01;
    });
    expect_corrupt_load_fails(path, corrupt_path, [layout](std::vector<uint8_t> & corrupt) {
        corrupt[layout.vectors.offset] ^= 0x01;
    });
    expect_corrupt_load_fails(path, corrupt_path, [layout](std::vector<uint8_t> & corrupt) {
        corrupt[layout.ids.offset] ^= 0x01;
    });
    expect_corrupt_load_fails(path, corrupt_path, [layout](std::vector<uint8_t> & corrupt) {
        corrupt[layout.checksum.offset + layout.checksum.size - 1] ^= 0x01;
    });

    ggml_vec_index_free(tv);
    std::filesystem::remove(path);
    std::filesystem::remove(corrupt_path);
}

void check_turbovec_tiny_tqplus_scale_rejected(int bits) {
    constexpr int dim = 64;
    CHECK(bits == 2 || bits == 4);

    std::vector<float> vector(static_cast<size_t>(dim), 0.0f);
    for (int i = 0; i < dim; ++i) {
        vector[static_cast<size_t>(i)] =
            static_cast<float>(0.25 * std::sin(0.11 * static_cast<double>(i + 1)));
    }
    const uint64_t id = static_cast<uint64_t>(625000 + bits);

    auto * tv = create_legacy_turbovec(dim, bits);
    CHECK(tv != nullptr);
    CHECK(ggml_vec_index_add(tv, vector.data(), 1, &id) == GGML_VEC_INDEX_OK);

    const std::string path =
        (std::filesystem::temp_directory_path() /
         ("ggml-vector-index-turbovec-tiny-scale-q" + std::to_string(bits) + ".tvim")).string();
    const std::string tiny_path =
        (std::filesystem::temp_directory_path() /
         ("ggml-vector-index-turbovec-tiny-scale-mutated-q" + std::to_string(bits) + ".tvim")).string();
    std::filesystem::remove(path);
    std::filesystem::remove(tiny_path);

    CHECK(ggml_vec_index_write(tv, path.c_str()) == GGML_VEC_INDEX_OK);
    std::vector<uint8_t> bytes = read_file_bytes(path);
    const TurboVecTvimLayout layout = parse_turbovec_tvim_layout(bytes, bits);
    const size_t tqplus_scale_offset =
        layout.calibration.offset + static_cast<size_t>(dim) * sizeof(float);
    write_u32_le_at(bytes, tqplus_scale_offset, 1u);
    refresh_turbovec_tvim_checksums(bytes, layout);
    write_file_bytes(tiny_path, bytes);

    auto * loaded = ggml_vec_index_load(tiny_path.c_str());
    CHECK(loaded == nullptr);
    ggml_vec_index_free(loaded);

    ggml_vec_index_free(tv);
    std::filesystem::remove(path);
    std::filesystem::remove(tiny_path);
}

void check_turbovec_zero_scale_and_legacy_calibration(int bits) {
    constexpr int dim = 64;
    CHECK(bits == 2 || bits == 4);

    std::vector<float> zero(static_cast<size_t>(dim), 0.0f);
    std::vector<float> query(static_cast<size_t>(dim), 0.0f);
    for (int i = 0; i < dim; ++i) {
        query[static_cast<size_t>(i)] =
            static_cast<float>(std::sin(0.07 * static_cast<double>(i + 1)));
    }
    const uint64_t zero_id = static_cast<uint64_t>(620000 + bits);
    auto * tv = create_legacy_turbovec(dim, bits);
    CHECK(tv != nullptr);
    CHECK(ggml_vec_index_add(tv, zero.data(), 1, &zero_id) == GGML_VEC_INDEX_OK);

    const std::string path =
        (std::filesystem::temp_directory_path() /
         ("ggml-vector-index-turbovec-zero-scale-q" + std::to_string(bits) + ".tvim")).string();
    const std::string legacy_path =
        (std::filesystem::temp_directory_path() /
         ("ggml-vector-index-turbovec-zero-calibration-q" + std::to_string(bits) + ".tvim")).string();
    const std::string roundtrip_path =
        (std::filesystem::temp_directory_path() /
         ("ggml-vector-index-turbovec-zero-calibration-roundtrip-q" + std::to_string(bits) + ".tvim")).string();
    std::filesystem::remove(path);
    std::filesystem::remove(legacy_path);
    std::filesystem::remove(roundtrip_path);

    CHECK(ggml_vec_index_write(tv, path.c_str()) == GGML_VEC_INDEX_OK);
    const std::vector<uint8_t> bytes = read_file_bytes(path);
    const TurboVecTvimLayout layout = parse_turbovec_tvim_layout(bytes, bits);
    CHECK(read_u32_le_from(bytes.data() + layout.qparams.offset) == 0);

    auto * loaded = ggml_vec_index_load(path.c_str());
    CHECK(loaded != nullptr);
    std::array<float, 1> scores{};
    std::array<uint64_t, 1> out{};
    CHECK(ggml_vec_index_search(loaded, query.data(), 1, 1, scores.data(), out.data()) ==
          GGML_VEC_INDEX_OK);
    CHECK(out[0] == zero_id);
    CHECK(scores[0] == 0.0f);

    std::vector<uint8_t> legacy = bytes;
    write_u32_le_at(legacy, 28, 0);
    legacy.erase(
        legacy.begin() + static_cast<std::ptrdiff_t>(layout.calibration.offset),
        legacy.begin() + static_cast<std::ptrdiff_t>(layout.calibration.offset + layout.calibration.size));
    const size_t legacy_qparams_offset = 32;
    const size_t legacy_vectors_offset = legacy_qparams_offset + layout.qparams.size;
    const size_t legacy_ids_offset = legacy_vectors_offset + layout.vectors.size;
    const size_t legacy_checksum_offset = legacy_ids_offset + layout.ids.size;
    CHECK(legacy_checksum_offset + layout.checksum.size == legacy.size());
    write_u32_le_at(
        legacy,
        legacy_checksum_offset,
        crc32c_update(0xffffffffu, legacy.data(), 32) ^ 0xffffffffu);
    write_u32_le_at(
        legacy,
        legacy_checksum_offset + 4,
        crc32c_update(0xffffffffu, legacy.data() + legacy_qparams_offset, layout.qparams.size) ^
            0xffffffffu);
    write_u32_le_at(
        legacy,
        legacy_checksum_offset + 8,
        crc32c_update(0xffffffffu, legacy.data() + legacy_vectors_offset, layout.vectors.size) ^
            0xffffffffu);
    write_u32_le_at(
        legacy,
        legacy_checksum_offset + 12,
        crc32c_update(0xffffffffu, legacy.data() + legacy_ids_offset, layout.ids.size) ^
            0xffffffffu);
    write_file_bytes(legacy_path, legacy);

    auto * legacy_loaded = ggml_vec_index_load(legacy_path.c_str());
    CHECK(legacy_loaded != nullptr);
    CHECK(ggml_vec_index_search(legacy_loaded, query.data(), 1, 1, scores.data(), out.data()) ==
          GGML_VEC_INDEX_OK);
    CHECK(out[0] == zero_id);
    CHECK(scores[0] == 0.0f);

    std::vector<float> added(static_cast<size_t>(dim), 0.0f);
    added[0] = 1.0f;
    added[1] = 0.5f;
    const uint64_t added_id = static_cast<uint64_t>(621000 + bits);
    CHECK(ggml_vec_index_add(legacy_loaded, added.data(), 1, &added_id) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(legacy_loaded, roundtrip_path.c_str()) == GGML_VEC_INDEX_OK);
    const std::vector<uint8_t> roundtrip = read_file_bytes(roundtrip_path);
    const TurboVecTvimLayout roundtrip_layout = parse_turbovec_tvim_layout(roundtrip, bits);
    CHECK(roundtrip_layout.calibration.size == 2 * static_cast<size_t>(dim) * sizeof(float));

    CHECK(ggml_vec_index_search(legacy_loaded, added.data(), 1, 1, scores.data(), out.data()) ==
          GGML_VEC_INDEX_OK);
    CHECK(out[0] == added_id);

    auto * roundtrip_loaded = ggml_vec_index_load(roundtrip_path.c_str());
    CHECK(roundtrip_loaded != nullptr);
    CHECK(ggml_vec_index_contains(roundtrip_loaded, zero_id) == 1);
    CHECK(ggml_vec_index_contains(roundtrip_loaded, added_id) == 1);
    CHECK(ggml_vec_index_search(roundtrip_loaded, added.data(), 1, 1, scores.data(), out.data()) ==
          GGML_VEC_INDEX_OK);
    CHECK(out[0] == added_id);
    std::array<float, 2> roundtrip_scores{};
    std::array<uint64_t, 2> roundtrip_ids{};
    CHECK(ggml_vec_index_search(
        roundtrip_loaded,
        query.data(),
        1,
        2,
        roundtrip_scores.data(),
        roundtrip_ids.data()) == GGML_VEC_INDEX_OK);
    const size_t zero_position = roundtrip_ids[0] == zero_id ? 0 : 1;
    CHECK(roundtrip_ids[zero_position] == zero_id);
    CHECK(roundtrip_scores[zero_position] == 0.0f);

    ggml_vec_index_free(roundtrip_loaded);
    ggml_vec_index_free(legacy_loaded);
    ggml_vec_index_free(loaded);
    ggml_vec_index_free(tv);
    std::filesystem::remove(path);
    std::filesystem::remove(legacy_path);
    std::filesystem::remove(roundtrip_path);
}

void check_quantized_partial_replay_rollback(int bit_width) {
    CHECK(bit_width == 4 || bit_width == 8);
    const std::string suffix = std::to_string(bit_width);
    const std::string snapshot_path =
        (std::filesystem::temp_directory_path() /
         ("ggml-vector-index-quantized-replay-base-q" + suffix + ".tvim")).string();
    const std::string delta_path =
        (std::filesystem::temp_directory_path() /
         ("ggml-vector-index-quantized-replay-q" + suffix + ".tvid")).string();
    const std::string rollback_path =
        (std::filesystem::temp_directory_path() /
         ("ggml-vector-index-quantized-replay-rollback-q" + suffix + ".tvim")).string();
    std::filesystem::remove(snapshot_path);
    std::filesystem::remove(delta_path);
    std::filesystem::remove(delta_path + ".lock");
    std::filesystem::remove(rollback_path);

    const std::array<float, kDim * 4> vectors = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
    const std::array<uint64_t, 4> ids = {
        static_cast<uint64_t>(630000 + bit_width * 10),
        static_cast<uint64_t>(630001 + bit_width * 10),
        static_cast<uint64_t>(630002 + bit_width * 10),
        static_cast<uint64_t>(630003 + bit_width * 10),
    };

    auto * writer = ggml_vec_index_create(kDim, bit_width);
    CHECK(writer != nullptr);
    CHECK(ggml_vec_index_add(writer, vectors.data(), 2, ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(writer, snapshot_path.c_str()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_add_logged(
        writer, vectors.data() + 2 * kDim, 1, ids.data() + 2, delta_path.c_str()) ==
        GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_remove_logged(writer, ids[0], delta_path.c_str()) == GGML_VEC_INDEX_OK);
    ggml_vec_index_free(writer);

    std::vector<uint8_t> corrupt = read_file_bytes(delta_path);
    const size_t first_record_offset = delta_log_header_size(corrupt);
    const size_t second_record_offset =
        first_record_offset +
        delta_record_header_size(corrupt) +
        static_cast<size_t>(read_u64_le_at(corrupt, first_record_offset + 8));
    CHECK(second_record_offset + delta_record_header_size(corrupt) <= corrupt.size());
    corrupt[delta_record_state_offset(corrupt, second_record_offset)] ^= 1;
    refresh_delta_record_crc(corrupt, second_record_offset);
    write_file_bytes(delta_path, corrupt);

    auto * stale = ggml_vec_index_load(snapshot_path.c_str());
    CHECK(stale != nullptr);
    CHECK(ggml_vec_index_add_logged(
        stale, vectors.data() + 3 * kDim, 1, ids.data() + 3, delta_path.c_str()) ==
        GGML_VEC_INDEX_E_IO);
    CHECK(ggml_vec_index_len(stale) == 2);
    CHECK(ggml_vec_index_contains(stale, ids[0]) == 1);
    CHECK(ggml_vec_index_contains(stale, ids[1]) == 1);
    CHECK(ggml_vec_index_contains(stale, ids[2]) == 0);
    CHECK(ggml_vec_index_contains(stale, ids[3]) == 0);
    CHECK(ggml_vec_index_write(stale, rollback_path.c_str()) == GGML_VEC_INDEX_OK);
    CHECK(read_file_bytes(rollback_path) == read_file_bytes(snapshot_path));

    ggml_vec_index_free(stale);
    std::filesystem::remove(snapshot_path);
    std::filesystem::remove(delta_path);
    std::filesystem::remove(delta_path + ".lock");
    std::filesystem::remove(rollback_path);
}

// ---------------------------------------------------------------------------
// TurboVec 1.x: parity with upstream turbovec 1.1.2 (tests/turbovec-1x-golden.inc)
// ---------------------------------------------------------------------------

uint64_t tv1x_splitmix64(uint64_t x) {
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

// Mirrors golden_value() in tests/turbovec-golden-gen-1x: integer bits and
// exact f32 steps, so both sides build identical inputs.
float tv1x_value(uint64_t seed, size_t row, size_t col, size_t dim) {
    const uint64_t r = tv1x_splitmix64((seed * UINT64_C(0x100000001b3)) ^ static_cast<uint64_t>(row * dim + col));
    const float unit = static_cast<float>(r >> 40) * (1.0f / 16777216.0f) * 2.0f - 1.0f;
    const float scale = 0.25f * static_cast<float>(1 + col % 7);
    const float offset = 0.125f * static_cast<float>(col % 5);
    return unit * scale + offset;
}

std::vector<float> tv1x_rows(uint64_t seed, size_t n, size_t dim) {
    std::vector<float> out(n * dim);
    for (size_t row = 0; row < n; ++row) {
        for (size_t col = 0; col < dim; ++col) {
            out[row * dim + col] = tv1x_value(seed, row, col, dim);
        }
    }
    return out;
}

uint64_t tv1x_fnv(const void * data, size_t size) {
    return fnv1a_bytes(static_cast<const uint8_t *>(data), size);
}

std::vector<uint64_t> tv1x_ids(size_t n) {
    std::vector<uint64_t> ids(n);
    for (size_t i = 0; i < n; ++i) {
        ids[i] = 1000 + 7 * static_cast<uint64_t>(i);
    }
    return ids;
}

std::vector<uint8_t> tv1x_snapshot_bytes(ggml_vec_index_t * idx, const char * name) {
    const std::string path = (std::filesystem::temp_directory_path() / name).string();
    std::filesystem::remove(path);
    CHECK(ggml_vec_index_write(idx, path.c_str()) == GGML_VEC_INDEX_OK);
    std::vector<uint8_t> bytes = read_file_bytes(path);
    std::filesystem::remove(path);
    return bytes;
}

bool tv1x_search_matches(
        ggml_vec_index_t * idx,
        const std::vector<float> & queries,
        const uint64_t * expected_ids,
        const float * expected_scores) {
    std::vector<float> scores(static_cast<size_t>(kTv1xNQuery * kTv1xK));
    std::vector<uint64_t> ids(scores.size());
    CHECK(ggml_vec_index_search(idx, queries.data(), kTv1xNQuery, kTv1xK, scores.data(), ids.data()) ==
          GGML_VEC_INDEX_OK);
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] != expected_ids[i] || scores[i] != expected_scores[i]) {
            return false;
        }
    }
    return true;
}

const Tv1xVmCase * tv1x_vm_case(int dim) {
    for (const Tv1xVmCase & c : kTv1xVmCases) {
        if (c.dim == dim) {
            return &c;
        }
    }
    return nullptr;
}

// Upstream scores 4-bit codes with its permute-dot kernel on hosts that run
// it and with the nibble LUT elsewhere; this port makes the same choice when
// an index is created. `kernel` names the expected one: 0 LUT, 1 permute-dot,
// -1 whichever this host picked.
void tv1x_check_search(
        ggml_vec_index_t * idx,
        int bits,
        int kernel,
        const std::vector<float> & queries,
        const uint64_t * lut_ids,
        const float * lut_scores,
        const uint64_t * vm_ids,
        const float * vm_scores) {
    if (bits != 4 || kernel == 0) {
        CHECK(tv1x_search_matches(idx, queries, lut_ids, lut_scores));
    } else if (kernel == 1) {
        CHECK(tv1x_search_matches(idx, queries, vm_ids, vm_scores));
    } else {
        CHECK(tv1x_search_matches(idx, queries, lut_ids, lut_scores) ||
              tv1x_search_matches(idx, queries, vm_ids, vm_scores));
    }
}

void check_turbovec_1x_golden_case(const Tv1xIndexCase & c, int kernel) {
    const size_t dim = static_cast<size_t>(c.dim);
    std::vector<float> db = tv1x_rows(kTv1xSeedDb, kTv1xNDb, dim);
    std::fill(db.begin(), db.begin() + static_cast<std::ptrdiff_t>(dim), 0.0f);
    const std::vector<float> queries = tv1x_rows(kTv1xSeedQuery, kTv1xNQuery, dim);
    const std::vector<float> calib = tv1x_rows(kTv1xSeedCalib, kTv1xNCalib, dim);
    const std::vector<uint64_t> ids = tv1x_ids(kTv1xNDb);

    auto * plain = ggml_vec_index_create_turbovec(c.dim, c.bits);
    CHECK(plain != nullptr);
    CHECK(ggml_vec_index_bit_width(plain) == c.bits);
    CHECK(ggml_vec_index_add(plain, db.data(), kTv1xNDb, ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_turbovec_is_calibrated(plain) == 0);
    const Tv1xVmCase * vm = c.bits == 4 ? tv1x_vm_case(c.dim) : nullptr;
    CHECK(c.bits != 4 || vm != nullptr);
    tv1x_check_search(
        plain, c.bits, kernel, queries, c.search_ids, c.search_scores,
        vm != nullptr ? vm->search_ids : nullptr, vm != nullptr ? vm->search_scores : nullptr);
    const std::vector<uint8_t> plain_bytes = tv1x_snapshot_bytes(plain, "ggml-vector-index-tv1x-plain.tvim");
    CHECK(plain_bytes.size() == c.snapshot_len);
    CHECK(tv1x_fnv(plain_bytes.data(), plain_bytes.size()) == c.snapshot_hash);

    auto * calibrated = ggml_vec_index_create_turbovec(c.dim, c.bits);
    CHECK(calibrated != nullptr);
    CHECK(ggml_vec_index_turbovec_calibrate(calibrated, calib.data(), kTv1xNCalib) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_turbovec_is_calibrated(calibrated) == 1);
    CHECK(ggml_vec_index_add(calibrated, db.data(), kTv1xNDb, ids.data()) == GGML_VEC_INDEX_OK);
    tv1x_check_search(
        calibrated, c.bits, kernel, queries, c.calibrated_search_ids, c.calibrated_search_scores,
        vm != nullptr ? vm->calibrated_search_ids : nullptr, vm != nullptr ? vm->calibrated_search_scores : nullptr);
    const std::vector<uint8_t> calibrated_bytes =
        tv1x_snapshot_bytes(calibrated, "ggml-vector-index-tv1x-calibrated.tvim");
    CHECK(calibrated_bytes.size() == c.calibrated_snapshot_len);
    CHECK(tv1x_fnv(calibrated_bytes.data(), calibrated_bytes.size()) == c.calibrated_snapshot_hash);

#ifdef GGML_VEC_INDEX_TEST_HOOKS
    uint64_t codes = 0;
    uint64_t scales = 0;
    uint64_t shift = 0;
    uint64_t scale = 0;
    CHECK(turbovec_index_hashes_for_test(plain, &codes, &scales, &shift, &scale) == 1);
    CHECK(codes == c.codes_hash);
    CHECK(scales == c.scales_hash);
    CHECK(turbovec_index_hashes_for_test(calibrated, &codes, &scales, &shift, &scale) == 1);
    CHECK(shift == c.calib_shift_hash);
    CHECK(scale == c.calib_scale_hash);
    CHECK(codes == c.calibrated_codes_hash);
    CHECK(scales == c.calibrated_scales_hash);
    // Calibrating a populated index re-encodes its rows from their codes.
    CHECK(ggml_vec_index_turbovec_calibrate(plain, calib.data(), kTv1xNCalib) == GGML_VEC_INDEX_OK);
    CHECK(turbovec_index_hashes_for_test(plain, &codes, &scales, &shift, &scale) == 1);
    CHECK(codes == c.reencoded_codes_hash);
    CHECK(scales == c.reencoded_scales_hash);
#endif
    ggml_vec_index_free(calibrated);
    ggml_vec_index_free(plain);
}

#ifdef GGML_VEC_INDEX_TEST_HOOKS
// A SIMD kernel's parity status: 1 matched its scalar reference bit for bit,
// 0 did not, -1 this build or CPU cannot run it (x86 kernels on an arm64
// host; CI's x86 runners execute them).
void check_simd_parity_status(const char * what, const char * kernel, int status, bool required) {
    if (status < 0) {
        CHECK(!required);
        std::printf("SKIP %s %s parity\n", what, kernel);
        return;
    }
    CHECK(status == 1);
}

void check_simd_kernel_parity() {
#if defined(__aarch64__) || defined(_M_ARM64)
    constexpr bool arm64 = true;
#else
    constexpr bool arm64 = false;
#endif
    // Rotation (Walsh-Hadamard, gather, norm), encode and input validation.
    static const char * const simd_names[] = { "NEON", "AVX2", "AVX-512" };
    for (int kernel = 0; kernel < 3; ++kernel) {
        const bool required = kernel == 0 && arm64;
        check_simd_parity_status(
            "TurboVec rotation", simd_names[kernel], turbovec_rotation_kernel_parity_for_test(kernel), required);
        if (kernel < 2) {
            check_simd_parity_status(
                "TurboVec encode", simd_names[kernel], turbovec_encode_kernel_parity_for_test(kernel), required);
            check_simd_parity_status(
                "vector validation", simd_names[kernel], vector_validation_kernel_parity_for_test(kernel), required);
        }
    }
    // x86 search-layout interleave; the whole-block helpers run everywhere.
    check_simd_parity_status("TurboVec layout", "SSSE3", turbovec_layout_kernel_parity_for_test(0), false);
    check_simd_parity_status("TurboVec layout", "AVX2", turbovec_layout_kernel_parity_for_test(1), false);
    check_simd_parity_status("TurboVec layout", "block helpers", turbovec_layout_kernel_parity_for_test(2), true);
    // Hardware CRC-32C against the table loop.
    check_simd_parity_status("CRC-32C", "ARMv8 CRC32", crc32c_kernel_parity_for_test(0), false);
    check_simd_parity_status("CRC-32C", "SSE4.2", crc32c_kernel_parity_for_test(1), false);
}

void check_turbovec_1x_rotation_and_codebooks() {
    for (const Tv1xRotationCase & c : kTv1xRotationCases) {
        const std::vector<float> row = tv1x_rows(kTv1xSeedRotation, 1, static_cast<size_t>(c.dim));
        std::vector<float> rotated(row.size());
        turbovec_hadamard_rotate_for_test(row.data(), rotated.data(), c.dim);
        CHECK(tv1x_fnv(rotated.data(), rotated.size() * sizeof(float)) == c.hash);
    }
    for (const Tv1xCodebookCase & c : kTv1xCodebookCases) {
        std::vector<float> boundaries;
        std::vector<float> centroids;
        turbovec_codebook_values(c.bits, c.dim, false, boundaries, centroids);
        CHECK(tv1x_fnv(boundaries.data(), boundaries.size() * sizeof(float)) == c.boundaries_hash);
        CHECK(tv1x_fnv(centroids.data(), centroids.size() * sizeof(float)) == c.centroids_hash);
    }
}
#endif

uint32_t tv1x_crc32(const std::vector<uint8_t> & bytes, size_t size) {
    // Superblocks are under 4096 bytes, where upstream's crc32 is one chain.
    CHECK(size < 4096);
    return crc32c_update(0xffffffffu, bytes.data(), size) ^ 0xffffffffu;
}

void tv1x_reseal_superblock(std::vector<uint8_t> & image, size_t superblock_len) {
    write_u32_le_at(image, superblock_len - 4, tv1x_crc32(image, superblock_len - 4));
}

std::vector<uint8_t> tv1x_upstream_image() {
    std::vector<uint8_t> image(kTv1xImageLen, 0);
    std::memcpy(image.data(), kTv1xImageSuperblock, sizeof(kTv1xImageSuperblock));
    std::memcpy(image.data() + sizeof(kTv1xImageSuperblock), kTv1xImageHeader0, sizeof(kTv1xImageHeader0));
    std::memcpy(
        image.data() + sizeof(kTv1xImageSuperblock) + 2 * kTv1xImageHeaderLen,
        kTv1xImageUnits,
        sizeof(kTv1xImageUnits));
    return image;
}

ggml_vec_index_t * tv1x_load_bytes(const std::vector<uint8_t> & bytes, int * status = nullptr) {
    const std::string path =
        (std::filesystem::temp_directory_path() / "ggml-vector-index-tv1x-load.tvim").string();
    write_file_bytes(path, bytes);
    ggml_vec_index_t * idx = nullptr;
    const int rc = ggml_vec_index_load_ex(path.c_str(), &idx);
    std::filesystem::remove(path);
    if (status != nullptr) {
        *status = rc;
    }
    return idx;
}

// A v8 unit in the planes layout holds the block's sign bytes group-major
// across its 32 lanes, then each lane's low planes.
std::vector<uint8_t> tv1x_to_planes_layout(const std::vector<uint8_t> & image, int dim, int bits) {
    std::vector<uint8_t> out = image;
    const size_t sb_len = sizeof(kTv1xImageSuperblock);
    const size_t row_bytes = static_cast<size_t>(dim) / static_cast<size_t>(8 / bits);
    const size_t nsg = static_cast<size_t>(dim) / 8;
    const size_t low_row = static_cast<size_t>(bits - 1) * nsg;
    const size_t unit_len = 32 * row_bytes + 32 * 4 + 32 * 8;
    const size_t units_at = sb_len + 2 * kTv1xImageHeaderLen;
    const size_t n_units = (image.size() - units_at) / unit_len;
    for (size_t u = 0; u < n_units; ++u) {
        uint8_t * unit = out.data() + units_at + u * unit_len;
        std::vector<uint8_t> codes(32 * row_bytes);
        for (size_t lane = 0; lane < 32; ++lane) {
            std::vector<uint8_t> packed(static_cast<size_t>(bits) * nsg, 0);
            for (int coord = 0; coord < dim; ++coord) {
                const size_t group = static_cast<size_t>(coord) / 2;
                const uint8_t byte = image[units_at + u * unit_len + group * 32 + lane];
                const uint8_t code = static_cast<uint8_t>(coord % 2 == 0 ? byte >> 4 : byte & 0x0f);
                for (int plane = 0; plane < bits; ++plane) {
                    if ((code >> plane) & 1) {
                        packed[static_cast<size_t>(plane) * nsg + static_cast<size_t>(coord) / 8] |=
                            static_cast<uint8_t>(1u << (7 - coord % 8));
                    }
                }
            }
            for (size_t g = 0; g < nsg; ++g) {
                codes[g * 32 + lane] = packed[low_row + g];
            }
            std::memcpy(codes.data() + nsg * 32 + lane * low_row, packed.data(), low_row);
        }
        std::memcpy(unit, codes.data(), codes.size());
    }
    out[7] = 1;
    tv1x_reseal_superblock(out, sb_len);
    return out;
}

void check_turbovec_1x_snapshot_interop() {
    CHECK(kTv1xImageBits == 4);
    const size_t dim = static_cast<size_t>(kTv1xImageDim);
    const std::vector<float> db = tv1x_rows(kTv1xSeedDb, kTv1xNDb, dim);
    const std::vector<float> queries = tv1x_rows(kTv1xSeedQuery, kTv1xNQuery, dim);
    const std::vector<uint64_t> ids = tv1x_ids(kTv1xNDb);

    auto * idx = ggml_vec_index_create_turbovec(kTv1xImageDim, kTv1xImageBits);
    CHECK(idx != nullptr);
    CHECK(ggml_vec_index_add(idx, db.data(), kTv1xNDb, ids.data()) == GGML_VEC_INDEX_OK);
    std::vector<float> expected_scores(static_cast<size_t>(kTv1xNQuery * kTv1xK));
    std::vector<uint64_t> expected_ids(expected_scores.size());
    CHECK(ggml_vec_index_search(
        idx, queries.data(), kTv1xNQuery, kTv1xK, expected_scores.data(), expected_ids.data()) ==
        GGML_VEC_INDEX_OK);

    // Our snapshot is upstream's image, piece by piece.
    const std::vector<uint8_t> ours = tv1x_snapshot_bytes(idx, "ggml-vector-index-tv1x-image.tvim");
    const std::vector<uint8_t> upstream = tv1x_upstream_image();
    CHECK(ours.size() == upstream.size());
    CHECK(std::memcmp(ours.data(), kTv1xImageSuperblock, sizeof(kTv1xImageSuperblock)) == 0);
    CHECK(std::memcmp(ours.data() + sizeof(kTv1xImageSuperblock), kTv1xImageHeader0, sizeof(kTv1xImageHeader0)) == 0);
    CHECK(ours == upstream);

    auto check_loaded = [&](ggml_vec_index_t * loaded) {
        CHECK(loaded != nullptr);
        CHECK(ggml_vec_index_len(loaded) == kTv1xNDb);
        CHECK(ggml_vec_index_dim(loaded) == kTv1xImageDim);
        CHECK(ggml_vec_index_bit_width(loaded) == kTv1xImageBits);
        std::vector<float> scores(expected_scores.size());
        std::vector<uint64_t> out(expected_ids.size());
        CHECK(ggml_vec_index_search(loaded, queries.data(), kTv1xNQuery, kTv1xK, scores.data(), out.data()) ==
              GGML_VEC_INDEX_OK);
        CHECK(out == expected_ids);
        CHECK(scores == expected_scores);
    };

    // In-memory snapshots are the same image.
    CHECK(ggml_vec_index_serialized_size(idx) == upstream.size());
    uint8_t * bytes = nullptr;
    size_t size = 0;
    CHECK(ggml_vec_index_to_bytes(idx, &bytes, &size) == GGML_VEC_INDEX_OK);
    CHECK(size == upstream.size());
    CHECK(std::equal(bytes, bytes + size, upstream.begin()));
    ggml_vec_index_t * from_bytes = nullptr;
    CHECK(ggml_vec_index_from_bytes(bytes, size, &from_bytes) == GGML_VEC_INDEX_OK);
    check_loaded(from_bytes);
    ggml_vec_index_free(from_bytes);
    CHECK(ggml_vec_index_from_bytes(bytes, size / 2, &from_bytes) != GGML_VEC_INDEX_OK);
    CHECK(from_bytes == nullptr);
    ggml_vec_index_bytes_free(bytes);
    const uint8_t garbage[8] = { 'n', 'o', 't', 'a', 't', 'v', '8', 0 };
    CHECK(ggml_vec_index_from_bytes(garbage, sizeof(garbage), &from_bytes) == GGML_VEC_INDEX_E_BAD_MAGIC);
    CHECK(ggml_vec_index_from_bytes(nullptr, 4, &from_bytes) == GGML_VEC_INDEX_E_INVALID_ARG);
    CHECK(ggml_vec_index_to_bytes(idx, nullptr, &size) == GGML_VEC_INDEX_E_INVALID_ARG);
    {
        auto * generic = ggml_vec_index_create(kTv1xImageDim, 8);
        CHECK(ggml_vec_index_to_bytes(generic, &bytes, &size) == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(bytes == nullptr && size == 0);
        CHECK(ggml_vec_index_serialized_size(generic) == 0);
        ggml_vec_index_free(generic);
        auto * legacy = create_legacy_turbovec(kTv1xImageDim, 4);
        CHECK(ggml_vec_index_to_bytes(legacy, &bytes, &size) == GGML_VEC_INDEX_E_INVALID_ARG);
        ggml_vec_index_free(legacy);
    }

    // Upstream's image loads, and saving it again reproduces it.
    auto * loaded = tv1x_load_bytes(upstream);
    check_loaded(loaded);
    CHECK(tv1x_snapshot_bytes(loaded, "ggml-vector-index-tv1x-resave.tvim") == upstream);
    ggml_vec_index_free(loaded);

    // The planes unit layout (written by upstream for large 2/4-bit indexes)
    // loads to the same index.
    auto * planes = tv1x_load_bytes(tv1x_to_planes_layout(upstream, kTv1xImageDim, kTv1xImageBits));
    check_loaded(planes);
    CHECK(tv1x_snapshot_bytes(planes, "ggml-vector-index-tv1x-planes-resave.tvim") == upstream);
    ggml_vec_index_free(planes);

    const size_t sb_len = sizeof(kTv1xImageSuperblock);
    int status = GGML_VEC_INDEX_OK;
    // A torn commit header fails its CRC, and the zeroed alternate slot is
    // not a commit either.
    {
        std::vector<uint8_t> bad = upstream;
        bad[sb_len + 8] ^= 0x01;
        CHECK(tv1x_load_bytes(bad, &status) == nullptr);
        CHECK(status != GGML_VEC_INDEX_OK);
    }
    // A codebook that drifted from the canonical one is refused even when
    // the superblock CRC is resealed over it.
    {
        std::vector<uint8_t> bad = upstream;
        bad[24] ^= 0x01;
        tv1x_reseal_superblock(bad, sb_len);
        CHECK(tv1x_load_bytes(bad, &status) == nullptr);
    }
    // A positional (TurboQuantIndex) file has no ids for this API.
    {
        std::vector<uint8_t> bad = upstream;
        bad[6] = 0;
        tv1x_reseal_superblock(bad, sb_len);
        CHECK(tv1x_load_bytes(bad, &status) == nullptr);
        CHECK(status == GGML_VEC_INDEX_E_INVALID_ARG);
    }
    // Truncation, an unknown revision and a negative per-vector scale.
    {
        std::vector<uint8_t> bad(upstream.begin(), upstream.begin() + static_cast<std::ptrdiff_t>(upstream.size() / 2));
        CHECK(tv1x_load_bytes(bad, &status) == nullptr);
        bad = upstream;
        bad[4] = 9;
        CHECK(tv1x_load_bytes(bad, &status) == nullptr);
        CHECK(status == GGML_VEC_INDEX_E_BAD_VERSION);
        bad = upstream;
        const size_t units_at = sb_len + 2 * kTv1xImageHeaderLen;
        const size_t row_bytes = dim / 2;
        bad[units_at + 32 * row_bytes + 3] |= 0x80;
        CHECK(tv1x_load_bytes(bad, &status) == nullptr);
    }
    CHECK(ggml_vec_index_load_mmap(
        (std::filesystem::temp_directory_path() / "ggml-vector-index-tv1x-missing.tvim").string().c_str()) == nullptr);
    ggml_vec_index_free(idx);
}

void check_turbovec_1x_calibrate_api() {
    constexpr int dim = 64;
    const std::vector<float> calib = tv1x_rows(kTv1xSeedCalib, 256, dim);
    const std::vector<float> db = tv1x_rows(kTv1xSeedDb, 40, dim);
    const std::vector<uint64_t> ids = tv1x_ids(40);

    CHECK(ggml_vec_index_turbovec_calibrate(nullptr, calib.data(), 256) == GGML_VEC_INDEX_E_INVALID_ARG);
    CHECK(ggml_vec_index_turbovec_is_calibrated(nullptr) == 0);
    auto * generic = ggml_vec_index_create(dim, 4);
    CHECK(ggml_vec_index_turbovec_calibrate(generic, calib.data(), 256) == GGML_VEC_INDEX_E_INVALID_ARG);
    ggml_vec_index_free(generic);
    auto * legacy = create_legacy_turbovec(dim, 4);
    CHECK(ggml_vec_index_turbovec_calibrate(legacy, calib.data(), 256) == GGML_VEC_INDEX_E_INVALID_ARG);
    ggml_vec_index_free(legacy);

    for (const int bits : { 2, 3, 4 }) {
        auto * tv = ggml_vec_index_create_turbovec(dim, bits);
        CHECK(tv != nullptr);
        CHECK(ggml_vec_index_turbovec_calibrate(tv, nullptr, 256) == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_turbovec_calibrate(tv, calib.data(), 1) == GGML_VEC_INDEX_E_INVALID_ARG);
        std::vector<float> bad = calib;
        bad[17] = std::numeric_limits<float>::quiet_NaN();
        CHECK(ggml_vec_index_turbovec_calibrate(tv, bad.data(), 256) == GGML_VEC_INDEX_E_INVALID_ARG);
        // Identical rows give every coordinate a zero span: the fit is the
        // identity, which is refused rather than committed.
        std::vector<float> flat(static_cast<size_t>(dim) * 8);
        for (size_t row = 0; row < 8; ++row) {
            std::copy(calib.begin(), calib.begin() + dim, flat.begin() + static_cast<std::ptrdiff_t>(row * dim));
        }
        CHECK(ggml_vec_index_turbovec_calibrate(tv, flat.data(), 8) == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_turbovec_is_calibrated(tv) == 0);

        CHECK(ggml_vec_index_add(tv, db.data(), 40, ids.data()) == GGML_VEC_INDEX_OK);
        auto * filter = ggml_vec_index_filter_create(tv, ids.data(), 4);
        CHECK(filter != nullptr);
        CHECK(ggml_vec_index_build_ivf(tv, 2, 1) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_turbovec_calibrate(tv, calib.data(), 256) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_turbovec_is_calibrated(tv) == 1);
        std::array<float, 1> score{};
        std::array<uint64_t, 1> out{};
        // A calibration re-encodes every row: prepared filters and IVF state
        // go stale.
        CHECK(ggml_vec_index_search_prepared_filtered(tv, filter, db.data(), 1, 1, score.data(), out.data()) ==
              GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_search_ivf(tv, db.data(), 1, 1, 2, score.data(), out.data()) ==
              GGML_VEC_INDEX_E_INVALID_ARG);
        ggml_vec_index_filter_free(filter);
        CHECK(ggml_vec_index_search(tv, db.data() + 5 * dim, 1, 1, score.data(), out.data()) == GGML_VEC_INDEX_OK);
        CHECK(out[0] == ids[5]);
        // Refitting is allowed, and the calibration survives a snapshot.
        CHECK(ggml_vec_index_turbovec_calibrate(tv, calib.data() + 16 * dim, 200) == GGML_VEC_INDEX_OK);
        const std::vector<uint8_t> bytes = tv1x_snapshot_bytes(tv, "ggml-vector-index-tv1x-calibrated-api.tvim");
        auto * loaded = tv1x_load_bytes(bytes);
        CHECK(loaded != nullptr);
        CHECK(ggml_vec_index_turbovec_is_calibrated(loaded) == 1);
        CHECK(ggml_vec_index_search(loaded, db.data() + 5 * dim, 1, 1, score.data(), out.data()) == GGML_VEC_INDEX_OK);
        CHECK(out[0] == ids[5]);
        CHECK(tv1x_snapshot_bytes(loaded, "ggml-vector-index-tv1x-calibrated-api-2.tvim") == bytes);
        ggml_vec_index_free(loaded);
        ggml_vec_index_free(tv);
    }
}

void check_turbovec_1x_dims() {
    CHECK(ggml_vec_index_create_turbovec(64, 1) == nullptr);
    CHECK(ggml_vec_index_create_turbovec(64, 5) == nullptr);
    CHECK(ggml_vec_index_create_turbovec(68, 2) == nullptr);
    CHECK(ggml_vec_index_create_turbovec(kTurboVecMaxTestDim + 8, 2) == nullptr);
    // The block-Hadamard rotation lifts the 0.9 dense-rotation cap.
    for (const int dim : { 1000, 3072, kTurboVecMaxTestDim }) {
        auto * tv = ggml_vec_index_create_turbovec(dim, dim == kTurboVecMaxTestDim ? 2 : 3);
        CHECK(tv != nullptr);
        const std::vector<float> rows = tv1x_rows(kTv1xSeedDb, 3, static_cast<size_t>(dim));
        const std::vector<uint64_t> ids = tv1x_ids(3);
        CHECK(ggml_vec_index_add(tv, rows.data(), 3, ids.data()) == GGML_VEC_INDEX_OK);
        std::array<float, 3> scores{};
        std::array<uint64_t, 3> out{};
        CHECK(ggml_vec_index_search(tv, rows.data() + dim, 1, 3, scores.data(), out.data()) == GGML_VEC_INDEX_OK);
        CHECK(out[0] == ids[1]);
        const std::vector<uint8_t> bytes = tv1x_snapshot_bytes(tv, "ggml-vector-index-tv1x-dims.tvim");
        auto * loaded = tv1x_load_bytes(bytes);
        CHECK(loaded != nullptr);
        CHECK(ggml_vec_index_dim(loaded) == dim);
        CHECK(ggml_vec_index_search(loaded, rows.data() + dim, 1, 3, scores.data(), out.data()) == GGML_VEC_INDEX_OK);
        CHECK(out[0] == ids[1]);
        ggml_vec_index_free(loaded);
        ggml_vec_index_free(tv);
    }
}

// A synced file hashed with its random nonce and the superblock CRC zeroed,
// as tests/turbovec-golden-gen-1x hashes upstream's.
std::pair<uint64_t, size_t> tv1x_masked_file_hash(const std::string & path, int bits, int n_calib) {
    std::vector<uint8_t> bytes = read_file_bytes(path);
    const size_t n_levels = size_t{ 1 } << bits;
    const size_t sb_len = 24 + (n_levels - 1) * 4 + n_levels * 4 + 4 + static_cast<size_t>(n_calib) * 8 + 4;
    CHECK(bytes.size() > sb_len);
    std::fill(bytes.begin() + 12, bytes.begin() + 20, 0);
    std::fill(bytes.begin() + static_cast<std::ptrdiff_t>(sb_len - 4), bytes.begin() + static_cast<std::ptrdiff_t>(sb_len), 0);
    return { tv1x_fnv(bytes.data(), bytes.size()), bytes.size() };
}

void check_turbovec_1x_sync_case(const Tv1xSyncCase & c) {
    const size_t dim = static_cast<size_t>(c.dim);
    const std::vector<float> all = tv1x_rows(kTv1xSeedDb, 160, dim);
    const std::vector<uint64_t> ids = tv1x_ids(160);
    const std::string path =
        (std::filesystem::temp_directory_path() / ("ggml-vector-index-tv1x-sync-q" + std::to_string(c.bits) + ".tvim")).string();
    std::filesystem::remove(path);
    auto add_rows = [&](ggml_vec_index_t * idx, size_t from, size_t to) {
        CHECK(ggml_vec_index_add(idx, all.data() + from * dim, static_cast<int>(to - from), ids.data() + from) ==
              GGML_VEC_INDEX_OK);
    };
    auto check_step = [&](int step, int n_calib) {
        const auto hashed = tv1x_masked_file_hash(path, c.bits, n_calib);
        CHECK(hashed.second == c.step_lens[step]);
        CHECK(hashed.first == c.step_hashes[step]);
    };

    auto * idx = ggml_vec_index_create_turbovec(c.dim, c.bits);
    CHECK(idx != nullptr);
    add_rows(idx, 0, 70);
    CHECK(ggml_vec_index_sync(idx, path.c_str()) == GGML_VEC_INDEX_OK);
    check_step(0, 0);
    add_rows(idx, 70, 120);
    CHECK(ggml_vec_index_sync(idx, path.c_str()) == GGML_VEC_INDEX_OK);
    check_step(1, 0);
    for (const size_t i : { 5, 40, 100, 119 }) {
        CHECK(ggml_vec_index_remove(idx, ids[i]) == GGML_VEC_INDEX_OK);
    }
    CHECK(ggml_vec_index_sync(idx, path.c_str()) == GGML_VEC_INDEX_OK);
    check_step(2, 0);
    const std::vector<uint8_t> after_step2 = read_file_bytes(path);
    for (const size_t i : { 6, 64, 33 }) {
        CHECK(ggml_vec_index_remove(idx, ids[i]) == GGML_VEC_INDEX_OK);
    }
    // Compaction renumbers slots but must not disturb the file's rows.
    CHECK(ggml_vec_index_compact(idx) == GGML_VEC_INDEX_OK);
    add_rows(idx, 120, 130);
    CHECK(ggml_vec_index_sync(idx, path.c_str()) == GGML_VEC_INDEX_OK);
    check_step(3, 0);
    CHECK(ggml_vec_index_sync(idx, path.c_str()) == GGML_VEC_INDEX_OK);
    check_step(4, 0);
    const std::vector<float> calib = tv1x_rows(kTv1xSeedCalib, kTv1xNCalib, dim);
    CHECK(ggml_vec_index_turbovec_calibrate(idx, calib.data(), kTv1xNCalib) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_sync(idx, path.c_str()) == GGML_VEC_INDEX_OK);
    check_step(5, c.dim);

    const std::vector<float> queries = tv1x_rows(kTv1xSeedQuery, kTv1xNQuery, dim);
    std::vector<float> scores(static_cast<size_t>(kTv1xNQuery * kTv1xK));
    std::vector<uint64_t> out(scores.size());
    auto * loaded = ggml_vec_index_load(path.c_str());
    CHECK(loaded != nullptr);
    CHECK(ggml_vec_index_len(loaded) == 120 - 7 + 10);
    CHECK(ggml_vec_index_search(loaded, queries.data(), kTv1xNQuery, kTv1xK, scores.data(), out.data()) ==
          GGML_VEC_INDEX_OK);
    const bool lut_ids = std::equal(out.begin(), out.end(), c.final_ids);
    const bool vm_ids = c.bits == 4 && std::equal(out.begin(), out.end(), kTv1xVmSyncFinalIds4);
    CHECK(lut_ids || vm_ids);

    // The loaded index is bound to the file and syncs on incrementally.
    CHECK(ggml_vec_index_remove(loaded, ids[3]) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_sync(loaded, path.c_str()) == GGML_VEC_INDEX_OK);
    auto * reloaded = ggml_vec_index_load(path.c_str());
    CHECK(reloaded != nullptr);
    CHECK(ggml_vec_index_len(reloaded) == ggml_vec_index_len(loaded));
    CHECK(ggml_vec_index_contains(reloaded, ids[3]) == 0);
    ggml_vec_index_free(reloaded);

    // The original index's commit has been overtaken by another writer.
    CHECK(ggml_vec_index_sync(idx, path.c_str()) == GGML_VEC_INDEX_E_IO);
    // A snapshot dropped over the path is nobody's sync target: rewrite it.
    CHECK(ggml_vec_index_write(idx, path.c_str()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_sync(idx, path.c_str()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_sync(loaded, path.c_str()) == GGML_VEC_INDEX_E_IO);
    ggml_vec_index_free(loaded);

    // A commit whose header tore falls back to the previous commit.
    {
        std::vector<uint8_t> torn = after_step2;
        const size_t n_levels = size_t{ 1 } << c.bits;
        const size_t sb_len = 24 + (n_levels - 1) * 4 + n_levels * 4 + 4 + 4;
        const size_t row_bytes = dim / static_cast<size_t>(8 / c.bits);
        const size_t hdr_len = 16 + 31 * (row_bytes + 12) + 4 + 1024 * (5 + 1 + row_bytes + 12) + 4 + 1024 * 4 + 12 + 4;
        // Generation 2 lives in slot 0.
        torn[sb_len + 40] ^= 0x5a;
        auto * fallback = tv1x_load_bytes(torn);
        CHECK(fallback != nullptr);
        CHECK(ggml_vec_index_len(fallback) == 120);
        ggml_vec_index_free(fallback);
        (void) hdr_len;
    }

    ggml_vec_index_free(idx);
    std::filesystem::remove(path);
}

void check_turbovec_1x_sync_api() {
    const std::string path = (std::filesystem::temp_directory_path() / "ggml-vector-index-tv1x-sync-api.tvim").string();
    CHECK(ggml_vec_index_sync(nullptr, path.c_str()) == GGML_VEC_INDEX_E_INVALID_ARG);
    auto * generic = ggml_vec_index_create(64, 8);
    CHECK(ggml_vec_index_sync(generic, path.c_str()) == GGML_VEC_INDEX_E_INVALID_ARG);
    ggml_vec_index_free(generic);
    auto * legacy = create_legacy_turbovec(64, 2);
    CHECK(ggml_vec_index_sync(legacy, path.c_str()) == GGML_VEC_INDEX_E_INVALID_ARG);
    ggml_vec_index_free(legacy);
    auto * tv = ggml_vec_index_create_turbovec(64, 2);
    CHECK(ggml_vec_index_sync(tv, nullptr) == GGML_VEC_INDEX_E_INVALID_ARG);
    // Syncing an empty index, and moving to a new path, both full-write.
    CHECK(ggml_vec_index_sync(tv, path.c_str()) == GGML_VEC_INDEX_OK);
    const std::string other = path + ".other";
    CHECK(ggml_vec_index_sync(tv, other.c_str()) == GGML_VEC_INDEX_OK);
    auto * loaded = ggml_vec_index_load(other.c_str());
    CHECK(loaded != nullptr);
    CHECK(ggml_vec_index_len(loaded) == 0);
    ggml_vec_index_free(loaded);
    ggml_vec_index_free(tv);
    std::filesystem::remove(path);
    std::filesystem::remove(other);
}

void check_turbovec_1x_history_snapshots() {
    constexpr size_t dim = 128;
    const std::vector<float> all = tv1x_rows(kTv1xSeedDb, 80, dim);
    const std::vector<uint64_t> ids = tv1x_ids(80);
    for (const Tv1xHistoryCase & c : kTv1xHistoryCases) {
        auto * idx = ggml_vec_index_create_turbovec(static_cast<int>(dim), c.bits);
        CHECK(ggml_vec_index_add(idx, all.data(), 70, ids.data()) == GGML_VEC_INDEX_OK);
        for (const size_t i : { 5, 40, 69, 0 }) {
            CHECK(ggml_vec_index_remove(idx, ids[i]) == GGML_VEC_INDEX_OK);
        }
        CHECK(ggml_vec_index_add(idx, all.data() + 70 * dim, 5, ids.data() + 70) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_remove(idx, ids[71]) == GGML_VEC_INDEX_OK);
        // Upstream's swap-remove order survives our tombstones and compaction.
        CHECK(ggml_vec_index_compact(idx) == GGML_VEC_INDEX_OK);
        const std::vector<uint8_t> bytes = tv1x_snapshot_bytes(idx, "ggml-vector-index-tv1x-history.tvim");
        CHECK(bytes.size() == c.snapshot_len);
        CHECK(tv1x_fnv(bytes.data(), bytes.size()) == c.snapshot_hash);
        ggml_vec_index_free(idx);
    }
}

#ifdef GGML_VEC_INDEX_TEST_HOOKS
// Tears a planned sync at many byte offsets (upstream
// a_sync_torn_at_any_byte_recovers_the_previous_commit): every prefix of the
// write must load as either the previous commit or the new one.
void check_turbovec_1x_torn_sync() {
    constexpr int dim = 64;
    const std::vector<float> all = tv1x_rows(kTv1xSeedDb, 200, dim);
    const std::vector<uint64_t> ids = tv1x_ids(200);
    const std::string path = (std::filesystem::temp_directory_path() / "ggml-vector-index-tv1x-torn.tvim").string();
    std::filesystem::remove(path);
    auto * idx = ggml_vec_index_create_turbovec(dim, 4);
    CHECK(ggml_vec_index_add(idx, all.data(), 100, ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_sync(idx, path.c_str()) == GGML_VEC_INDEX_OK);
    for (const size_t i : { 3, 50, 99, 64 }) {
        CHECK(ggml_vec_index_remove(idx, ids[i]) == GGML_VEC_INDEX_OK);
    }
    CHECK(ggml_vec_index_sync(idx, path.c_str()) == GGML_VEC_INDEX_OK);
    const std::vector<uint8_t> before = read_file_bytes(path);

    // The next commit: removals in committed blocks plus appended blocks.
    for (const size_t i : { 10, 20 }) {
        CHECK(ggml_vec_index_remove(idx, ids[i]) == GGML_VEC_INDEX_OK);
    }
    CHECK(ggml_vec_index_add(idx, all.data() + 100 * dim, 70, ids.data() + 100) == GGML_VEC_INDEX_OK);
    std::vector<std::vector<std::pair<uint64_t, std::vector<uint8_t>>>> batches;
    CHECK(turbovec_sync_plan_for_test(idx, path.c_str(), batches) == GGML_VEC_INDEX_OK);
    std::vector<std::pair<uint64_t, uint8_t>> writes;
    for (const auto & batch : batches) {
        for (const auto & op : batch) {
            for (size_t k = 0; k < op.second.size(); ++k) {
                writes.emplace_back(op.first + k, op.second[k]);
            }
        }
    }
    const int old_len = 96;
    const int new_len = 96 - 2 + 70;
    size_t tested = 0;
    for (size_t cut = 0; cut <= writes.size(); cut += (cut < 64 || writes.size() - cut < 64) ? 1 : 61) {
        std::vector<uint8_t> torn = before;
        for (size_t w = 0; w < cut; ++w) {
            if (writes[w].first >= torn.size()) {
                torn.resize(static_cast<size_t>(writes[w].first) + 1, 0);
            }
            torn[static_cast<size_t>(writes[w].first)] = writes[w].second;
        }
        auto * loaded = tv1x_load_bytes(torn);
        CHECK(loaded != nullptr);
        const int len = ggml_vec_index_len(loaded);
        CHECK(len == old_len || len == new_len);
        CHECK(ggml_vec_index_contains(loaded, ids[10]) == (len == old_len ? 1 : 0));
        if (cut == writes.size()) {
            CHECK(len == new_len);
        }
        ggml_vec_index_free(loaded);
        ++tested;
    }
    CHECK(tested > 128);
    ggml_vec_index_free(idx);
    std::filesystem::remove(path);
}
#endif

// Threads change how work is split, never the result: encodes, calibration,
// snapshots and every search path must match the single-threaded run.
void check_turbovec_1x_threads() {
    constexpr int dim = 64;
    constexpr int n = 40000;
    const std::vector<float> rows = tv1x_rows(kTv1xSeedDb, n, dim);
    const std::vector<float> calib = tv1x_rows(kTv1xSeedCalib, 1500, dim);
    const std::vector<float> queries = tv1x_rows(kTv1xSeedQuery, 9, dim);
    std::vector<uint64_t> ids(n);
    for (int i = 0; i < n; ++i) {
        ids[static_cast<size_t>(i)] = 5000000 + static_cast<uint64_t>(i);
    }
    CHECK(ggml_vec_index_set_threads(nullptr, 2) == GGML_VEC_INDEX_E_INVALID_ARG);
    for (const int bits : { 2, 3, 4 }) {
        ggml_vec_index_t * built[2] = {};
        std::vector<uint8_t> bytes[2];
        std::vector<float> scores[2];
        std::vector<uint64_t> out[2];
        const std::vector<uint64_t> allowed = { ids[7], ids[31000], ids[39999], ids[12345] };
        for (int run = 0; run < 2; ++run) {
            auto * tv = ggml_vec_index_create_turbovec(dim, bits);
            CHECK(ggml_vec_index_threads(tv) == 1);
            CHECK(ggml_vec_index_set_threads(tv, 0) == GGML_VEC_INDEX_E_INVALID_ARG);
            CHECK(ggml_vec_index_set_threads(tv, run == 0 ? 1 : 8) == GGML_VEC_INDEX_OK);
            CHECK(ggml_vec_index_add(tv, rows.data(), n / 2, ids.data()) == GGML_VEC_INDEX_OK);
            CHECK(ggml_vec_index_turbovec_calibrate(tv, calib.data(), 1500) == GGML_VEC_INDEX_OK);
            CHECK(ggml_vec_index_add(tv, rows.data() + static_cast<size_t>(n / 2) * dim, n / 2, ids.data() + n / 2) ==
                  GGML_VEC_INDEX_OK);
            for (const size_t i : { 3, 900, 20001, 33333 }) {
                CHECK(ggml_vec_index_remove(tv, ids[i]) == GGML_VEC_INDEX_OK);
            }
            uint8_t * data = nullptr;
            size_t size = 0;
            CHECK(ggml_vec_index_to_bytes(tv, &data, &size) == GGML_VEC_INDEX_OK);
            bytes[run].assign(data, data + size);
            ggml_vec_index_bytes_free(data);
            // A batch, a lone query (block split at 8 threads) and a filter.
            scores[run].resize(9 * 10 + 10 + 4);
            out[run].resize(scores[run].size());
            CHECK(ggml_vec_index_search(tv, queries.data(), 9, 10, scores[run].data(), out[run].data()) ==
                  GGML_VEC_INDEX_OK);
            CHECK(ggml_vec_index_search(tv, queries.data(), 1, 10, scores[run].data() + 90, out[run].data() + 90) ==
                  GGML_VEC_INDEX_OK);
            CHECK(ggml_vec_index_search_filtered(
                      tv, queries.data() + dim, 1, 4, allowed.data(), 4, scores[run].data() + 100, out[run].data() + 100) ==
                  GGML_VEC_INDEX_OK);
            built[run] = tv;
        }
        CHECK(bytes[0] == bytes[1]);
        CHECK(out[0] == out[1]);
        CHECK(scores[0] == scores[1]);
        CHECK(std::equal(out[0].begin(), out[0].begin() + 10, out[0].begin() + 90));
#ifdef GGML_VEC_INDEX_TEST_HOOKS
        // The state hash an add and a calibrate keep over 8 threads is the
        // one a single-threaded load rebuilds.
        {
            uint64_t sums[3][3] = {};
            state_hash_for_test(built[0], sums[0]);
            state_hash_for_test(built[1], sums[1]);
            ggml_vec_index_t * loaded = nullptr;
            CHECK(ggml_vec_index_from_bytes(bytes[1].data(), bytes[1].size(), &loaded) == GGML_VEC_INDEX_OK);
            state_hash_for_test(loaded, sums[2]);
            ggml_vec_index_free(loaded);
            for (int i = 0; i < 3; ++i) {
                CHECK(sums[0][i] == sums[1][i] && sums[0][i] == sums[2][i]);
            }
        }
#endif
        ggml_vec_index_free(built[0]);
        ggml_vec_index_free(built[1]);
    }
}

void check_vector_index_introspection() {
    CHECK(ggml_vec_index_storage(nullptr) == GGML_VEC_INDEX_STORAGE_NONE);
    CHECK(ggml_vec_index_turbovec_is_legacy(nullptr) == 0);
    const std::pair<ggml_vec_index_t *, int> cases[] = {
        { ggml_vec_index_create(64, 32), GGML_VEC_INDEX_STORAGE_F32 },
        { ggml_vec_index_create(64, 8), GGML_VEC_INDEX_STORAGE_Q8 },
        { ggml_vec_index_create(64, 4), GGML_VEC_INDEX_STORAGE_Q4 },
        { ggml_vec_index_create_turbovec(64, 2), GGML_VEC_INDEX_STORAGE_TURBOVEC_Q2 },
        { ggml_vec_index_create_turbovec(64, 3), GGML_VEC_INDEX_STORAGE_TURBOVEC_Q3 },
        { ggml_vec_index_create_turbovec(64, 4), GGML_VEC_INDEX_STORAGE_TURBOVEC_Q4 },
        { create_legacy_turbovec(64, 4), GGML_VEC_INDEX_STORAGE_TURBOVEC_Q4 },
    };
    const std::vector<float> rows = tv1x_rows(kTv1xSeedDb, 5, 64);
    const std::vector<uint64_t> ids = tv1x_ids(5);
    for (const auto & c : cases) {
        CHECK(ggml_vec_index_storage(c.first) == c.second);
        CHECK(ggml_vec_index_add(c.first, rows.data(), 5, ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_remove(c.first, ids[1]) == GGML_VEC_INDEX_OK);
        size_t count = 0;
        CHECK(ggml_vec_index_ids(c.first, nullptr, 0, &count) == GGML_VEC_INDEX_OK);
        CHECK(count == 4);
        std::vector<uint64_t> listed(8, 0);
        CHECK(ggml_vec_index_ids(c.first, listed.data(), listed.size(), &count) == GGML_VEC_INDEX_OK);
        CHECK(count == 4 && listed[4] == 0);
        std::vector<uint64_t> live(listed.begin(), listed.begin() + 4);
        std::sort(live.begin(), live.end());
        CHECK((live == std::vector<uint64_t>{ ids[0], ids[2], ids[3], ids[4] }));
        // A short buffer gets a prefix of the same order.
        std::vector<uint64_t> prefix(2);
        CHECK(ggml_vec_index_ids(c.first, prefix.data(), prefix.size(), &count) == GGML_VEC_INDEX_OK);
        CHECK(count == 4 && prefix[0] == listed[0] && prefix[1] == listed[1]);
        CHECK(ggml_vec_index_ids(c.first, nullptr, 1, &count) == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_ids(c.first, listed.data(), 1, nullptr) == GGML_VEC_INDEX_E_INVALID_ARG);
    }
    CHECK(ggml_vec_index_turbovec_is_legacy(cases[5].first) == 0);
    CHECK(ggml_vec_index_turbovec_is_legacy(cases[6].first) == 1);
    for (const auto & c : cases) {
        ggml_vec_index_free(c.first);
    }
}

// The upstream image with an extra block of bytes rebuilt around a new
// superblock (its header slots and units move with it).
std::vector<uint8_t> tv1x_with_superblock(const std::vector<uint8_t> & superblock) {
    std::vector<uint8_t> image(superblock.size() + 2 * kTv1xImageHeaderLen + sizeof(kTv1xImageUnits), 0);
    std::memcpy(image.data(), superblock.data(), superblock.size());
    std::memcpy(image.data() + superblock.size(), kTv1xImageHeader0, sizeof(kTv1xImageHeader0));
    std::memcpy(image.data() + superblock.size() + 2 * kTv1xImageHeaderLen, kTv1xImageUnits, sizeof(kTv1xImageUnits));
    return image;
}

std::vector<uint64_t> tv1x_top_ids(ggml_vec_index_t * idx, const float * query, int k) {
    std::vector<float> scores(static_cast<size_t>(k));
    std::vector<uint64_t> ids(static_cast<size_t>(k));
    CHECK(ggml_vec_index_search(idx, query, 1, k, scores.data(), ids.data()) == GGML_VEC_INDEX_OK);
    return ids;
}

uint64_t tv1x_nonce(const std::string & path) {
    const std::vector<uint8_t> bytes = read_file_bytes(path);
    return read_u64_le_at(bytes, 12);
}

// Upstream turbovec behaviours the golden fixtures do not reach.
void check_turbovec_1x_upstream_contracts() {
    const std::vector<uint8_t> upstream = tv1x_upstream_image();
    const size_t sb_len = sizeof(kTv1xImageSuperblock);
    const size_t dim = static_cast<size_t>(kTv1xImageDim);
    const std::vector<float> queries = tv1x_rows(kTv1xSeedQuery, kTv1xNQuery, dim);
    auto * reference = tv1x_load_bytes(upstream);
    CHECK(reference != nullptr);
    const std::vector<uint64_t> reference_ids = tv1x_top_ids(reference, queries.data(), 10);

    // A stored identity calibration declares none (upstream
    // normalize_calibration): the index is uncalibrated and writes none.
    {
        std::vector<uint8_t> sb(kTv1xImageSuperblock, kTv1xImageSuperblock + sb_len - 8);
        append_u32_le(sb, static_cast<uint32_t>(dim));
        for (size_t i = 0; i < dim; ++i) {
            append_f32_le(sb, 0.0f);
        }
        for (size_t i = 0; i < dim; ++i) {
            append_f32_le(sb, 1.0f);
        }
        append_u32_le(sb, tv1x_crc32(sb, sb.size()));
        auto * loaded = tv1x_load_bytes(tv1x_with_superblock(sb));
        CHECK(loaded != nullptr);
        CHECK(ggml_vec_index_turbovec_is_calibrated(loaded) == 0);
        CHECK(tv1x_snapshot_bytes(loaded, "ggml-vector-index-tv1x-identity.tvim") == upstream);
        ggml_vec_index_free(loaded);

        // The same file as a sync destination (a nonzero nonce): the loaded
        // index's superblock is shorter than the file's, so its next sync
        // rewrites the file rather than patching it at shifted offsets.
        std::vector<uint8_t> synced = sb;
        synced.resize(synced.size() - 4);
        write_u64_le_at(synced, 12, UINT64_C(0x5eed5eed5eed5eed));
        append_u32_le(synced, tv1x_crc32(synced, synced.size()));
        const std::string path =
            (std::filesystem::temp_directory_path() / "ggml-vector-index-tv1x-identity-sync.tvim").string();
        write_file_bytes(path, tv1x_with_superblock(synced));
        auto * bound = ggml_vec_index_load(path.c_str());
        CHECK(bound != nullptr);
        const std::vector<float> extra = tv1x_rows(kTv1xSeedDb + 7, 40, dim);
        std::vector<uint64_t> extra_ids(40);
        for (size_t i = 0; i < extra_ids.size(); ++i) {
            extra_ids[i] = UINT64_C(900000) + i;
        }
        CHECK(ggml_vec_index_add(bound, extra.data(), 40, extra_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_sync(bound, path.c_str()) == GGML_VEC_INDEX_OK);
        auto * reloaded = ggml_vec_index_load(path.c_str());
        CHECK(reloaded != nullptr);
        CHECK(ggml_vec_index_len(reloaded) == ggml_vec_index_len(bound));
        CHECK(tv1x_snapshot_bytes(reloaded, "ggml-vector-index-tv1x-identity-r.tvim") ==
              tv1x_snapshot_bytes(bound, "ggml-vector-index-tv1x-identity-b.tvim"));
        ggml_vec_index_free(reloaded);
        ggml_vec_index_free(bound);
        std::filesystem::remove(path);
    }
    // A v7 container (one superblock byte shorter, no layout byte) loads.
    {
        std::vector<uint8_t> sb(kTv1xImageSuperblock, kTv1xImageSuperblock + sb_len - 4);
        sb[2] = '7';
        sb[4] = 2;
        sb.erase(sb.begin() + 7);
        append_u32_le(sb, tv1x_crc32(sb, sb.size()));
        auto * loaded = tv1x_load_bytes(tv1x_with_superblock(sb));
        CHECK(loaded != nullptr);
        CHECK(tv1x_top_ids(loaded, queries.data(), 10) == reference_ids);
        // A v7 file this index loaded is rewritten as v8 by a write.
        CHECK(tv1x_snapshot_bytes(loaded, "ggml-vector-index-tv1x-v7.tvim") == upstream);
        ggml_vec_index_free(loaded);
    }
    // Trailing bytes past the declared image are ignored.
    {
        std::vector<uint8_t> padded = upstream;
        padded.resize(padded.size() + 65536, 0);
        auto * loaded = tv1x_load_bytes(padded);
        CHECK(loaded != nullptr);
        CHECK(ggml_vec_index_len(loaded) == kTv1xNDb);
        ggml_vec_index_free(loaded);
    }
    // Duplicate ids, an absurd row count, an oversized dim.
    {
        const size_t units_at = sb_len + 2 * kTv1xImageHeaderLen;
        const size_t ids_at = units_at + 32 * (dim / 2) + 32 * 4;
        std::vector<uint8_t> bad = upstream;
        std::memcpy(bad.data() + ids_at + 8, bad.data() + ids_at, 8);
        CHECK(tv1x_load_bytes(bad) == nullptr);
        bad = upstream;
        write_u64_le_at(bad, sb_len + 8, UINT64_C(1) << 40);
        CHECK(tv1x_load_bytes(bad) == nullptr);
        bad = upstream;
        write_u32_le_at(bad, 8, 16392);
        tv1x_reseal_superblock(bad, sb_len);
        CHECK(tv1x_load_bytes(bad) == nullptr);
    }
    // A calibration scale small enough to turn scores into Inf is refused.
    {
        auto * tv = ggml_vec_index_create_turbovec(kTv1xImageDim, 4);
        const std::vector<float> calib = tv1x_rows(kTv1xSeedCalib, 256, dim);
        CHECK(ggml_vec_index_turbovec_calibrate(tv, calib.data(), 256) == GGML_VEC_INDEX_OK);
        std::vector<uint8_t> bytes = tv1x_snapshot_bytes(tv, "ggml-vector-index-tv1x-tiny-scale.tvim");
        const size_t n_levels = 16;
        const size_t calibrated_sb = 24 + (n_levels - 1) * 4 + n_levels * 4 + 4 + dim * 8 + 4;
        const size_t scale_at = 24 + (n_levels - 1) * 4 + n_levels * 4 + 4 + dim * 4;
        std::vector<uint8_t> bad = bytes;
        uint32_t tiny = 0;
        const float tiny_value = 1e-40f;
        std::memcpy(&tiny, &tiny_value, sizeof(tiny));
        write_u32_le_at(bad, scale_at + 4 * 3, tiny);
        write_u32_le_at(bad, calibrated_sb - 4, tv1x_crc32(bad, calibrated_sb - 4));
        CHECK(tv1x_load_bytes(bad) == nullptr);
        ggml_vec_index_free(tv);
    }

    // Swap-removes patch the blocked search cache in place; it scores
    // exactly like one rebuilt from the remaining rows.
    for (const int bits : { 2, 3, 4 }) {
        constexpr int rdim = 64;
        const std::vector<float> rrows = tv1x_rows(kTv1xSeedDb, 300, rdim);
        const std::vector<uint64_t> rids = tv1x_ids(300);
        const std::vector<float> rqueries = tv1x_rows(kTv1xSeedQuery, 5, rdim);
        auto * tv = ggml_vec_index_create_turbovec(rdim, bits);
        CHECK(ggml_vec_index_add(tv, rrows.data(), 300, rids.data()) == GGML_VEC_INDEX_OK);
        for (size_t i = 0; i < 300; i += 7) {
            CHECK(ggml_vec_index_remove(tv, rids[(i * 13) % 300]) == GGML_VEC_INDEX_OK);
        }
        // The last slot itself (no move), whether or not already gone.
        (void) ggml_vec_index_remove(tv, rids[299]);
        uint8_t * data = nullptr;
        size_t size = 0;
        CHECK(ggml_vec_index_to_bytes(tv, &data, &size) == GGML_VEC_INDEX_OK);
        ggml_vec_index_t * rebuilt = nullptr;
        CHECK(ggml_vec_index_from_bytes(data, size, &rebuilt) == GGML_VEC_INDEX_OK);
        ggml_vec_index_bytes_free(data);
        std::vector<float> s1(5 * 20), s2(5 * 20);
        std::vector<uint64_t> o1(5 * 20), o2(5 * 20);
        CHECK(ggml_vec_index_search(tv, rqueries.data(), 5, 20, s1.data(), o1.data()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search(rebuilt, rqueries.data(), 5, 20, s2.data(), o2.data()) == GGML_VEC_INDEX_OK);
        CHECK(o1 == o2 && s1 == s2);
        ggml_vec_index_free(rebuilt);
        ggml_vec_index_free(tv);
    }

    // Ranking does not depend on the query's magnitude (upstream #335).
    {
        std::vector<float> tiny_query(queries.begin(), queries.begin() + static_cast<std::ptrdiff_t>(dim));
        for (float & v : tiny_query) {
            v *= 1e-20f;
        }
        CHECK(tv1x_top_ids(reference, tiny_query.data(), 10) == reference_ids);
    }

    // Calibration contracts (upstream tests/explicit_calibration.rs).
    {
        constexpr int cdim = 64;
        const std::vector<float> sample = tv1x_rows(kTv1xSeedCalib, 1024, cdim);
        const std::vector<float> rows = tv1x_rows(kTv1xSeedDb, 300, cdim);
        const std::vector<uint64_t> ids = tv1x_ids(300);
        for (const int bits : { 2, 3, 4 }) {
            // Refitting under the committed pair reproduces the codes.
            auto * tv = ggml_vec_index_create_turbovec(cdim, bits);
            CHECK(ggml_vec_index_turbovec_calibrate(tv, sample.data(), 1024) == GGML_VEC_INDEX_OK);
            CHECK(ggml_vec_index_add(tv, rows.data(), 300, ids.data()) == GGML_VEC_INDEX_OK);
            const std::vector<uint8_t> before = tv1x_snapshot_bytes(tv, "ggml-vector-index-tv1x-refit.tvim");
            CHECK(ggml_vec_index_turbovec_calibrate(tv, sample.data(), 1024) == GGML_VEC_INDEX_OK);
            const std::vector<uint8_t> after = tv1x_snapshot_bytes(tv, "ggml-vector-index-tv1x-refit.tvim");
            CHECK(after.size() == before.size());
            // Codes are byte-identical; scales may move by rounding only.
            auto * a = tv1x_load_bytes(before);
            auto * b = tv1x_load_bytes(after);
            CHECK(tv1x_top_ids(a, rows.data() + 7 * cdim, 5) == tv1x_top_ids(b, rows.data() + 7 * cdim, 5));
            ggml_vec_index_free(a);
            ggml_vec_index_free(b);
            // A calibrated empty index round-trips calibrated, and adds after
            // the reload reuse its calibration.
            auto * empty = ggml_vec_index_create_turbovec(cdim, bits);
            CHECK(ggml_vec_index_turbovec_calibrate(empty, sample.data(), 1024) == GGML_VEC_INDEX_OK);
            auto * restored = tv1x_load_bytes(tv1x_snapshot_bytes(empty, "ggml-vector-index-tv1x-empty-cal.tvim"));
            CHECK(ggml_vec_index_turbovec_is_calibrated(restored) == 1);
            CHECK(ggml_vec_index_add(restored, rows.data(), 300, ids.data()) == GGML_VEC_INDEX_OK);
            auto * fresh = ggml_vec_index_create_turbovec(cdim, bits);
            CHECK(ggml_vec_index_turbovec_calibrate(fresh, sample.data(), 1024) == GGML_VEC_INDEX_OK);
            CHECK(ggml_vec_index_add(fresh, rows.data(), 300, ids.data()) == GGML_VEC_INDEX_OK);
            CHECK(tv1x_snapshot_bytes(restored, "ggml-vector-index-tv1x-a.tvim") ==
                  tv1x_snapshot_bytes(fresh, "ggml-vector-index-tv1x-b.tvim"));
            // A drained index can be calibrated, and a loaded one refit.
            for (size_t i = 0; i < 300; ++i) {
                CHECK(ggml_vec_index_remove(fresh, ids[i]) == GGML_VEC_INDEX_OK);
            }
            CHECK(ggml_vec_index_turbovec_calibrate(fresh, sample.data() + 10 * cdim, 500) == GGML_VEC_INDEX_OK);
            CHECK(ggml_vec_index_turbovec_calibrate(restored, sample.data() + 10 * cdim, 500) == GGML_VEC_INDEX_OK);
            CHECK(tv1x_top_ids(restored, rows.data() + 9 * cdim, 1)[0] == ids[9]);
            // Exactly the two-row floor is accepted.
            CHECK(ggml_vec_index_turbovec_calibrate(fresh, sample.data(), 2) == GGML_VEC_INDEX_OK);
            for (auto * idx : { tv, empty, restored, fresh }) {
                ggml_vec_index_free(idx);
            }
        }
    }
    ggml_vec_index_free(reference);
}

// Sync contracts beyond the golden scenario (upstream lib.rs v7 crash and
// delta tests).
void check_turbovec_1x_sync_contracts() {
    constexpr int dim = 64;
    const std::vector<float> rows = tv1x_rows(kTv1xSeedDb, 2400, dim);
    const std::vector<uint64_t> ids = tv1x_ids(2400);
    const std::string path = (std::filesystem::temp_directory_path() / "ggml-vector-index-tv1x-sync-contracts.tvim").string();
    std::filesystem::remove(path);

    // Up to 1024 redo ops ride one commit header; more force a rewrite.
    // Removing rows from the low end leaves each hole inside committed
    // blocks, so every removal carries one op.
    for (const size_t n_removed : { size_t{ 1000 }, size_t{ 1100 } }) {
        std::filesystem::remove(path);
        auto * tv = ggml_vec_index_create_turbovec(dim, 2);
        CHECK(ggml_vec_index_add(tv, rows.data(), 2400, ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_sync(tv, path.c_str()) == GGML_VEC_INDEX_OK);
        const uint64_t nonce = tv1x_nonce(path);
        for (size_t i = 0; i < n_removed; ++i) {
            CHECK(ggml_vec_index_remove(tv, ids[i]) == GGML_VEC_INDEX_OK);
        }
        CHECK(ggml_vec_index_sync(tv, path.c_str()) == GGML_VEC_INDEX_OK);
        CHECK((tv1x_nonce(path) == nonce) == (n_removed <= 1024));
        auto * loaded = ggml_vec_index_load(path.c_str());
        CHECK(ggml_vec_index_len(loaded) == static_cast<int>(2400 - n_removed));
        CHECK(ggml_vec_index_contains(loaded, ids[n_removed - 1]) == 0);
        CHECK(ggml_vec_index_contains(loaded, ids[n_removed]) == 1);
        ggml_vec_index_free(loaded);
        ggml_vec_index_free(tv);
    }
    // Draining to empty and regrowing below the synced watermark.
    {
        std::filesystem::remove(path);
        auto * tv = ggml_vec_index_create_turbovec(dim, 4);
        CHECK(ggml_vec_index_add(tv, rows.data(), 100, ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_sync(tv, path.c_str()) == GGML_VEC_INDEX_OK);
        for (size_t i = 0; i < 100; ++i) {
            CHECK(ggml_vec_index_remove(tv, ids[i]) == GGML_VEC_INDEX_OK);
        }
        CHECK(ggml_vec_index_sync(tv, path.c_str()) == GGML_VEC_INDEX_OK);
        auto * empty = ggml_vec_index_load(path.c_str());
        CHECK(empty != nullptr && ggml_vec_index_len(empty) == 0);
        ggml_vec_index_free(empty);
        CHECK(ggml_vec_index_add(tv, rows.data() + 200 * dim, 40, ids.data() + 200) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_sync(tv, path.c_str()) == GGML_VEC_INDEX_OK);
        auto * regrown = ggml_vec_index_load(path.c_str());
        CHECK(ggml_vec_index_len(regrown) == 40);
        CHECK(tv1x_top_ids(regrown, rows.data() + 233 * dim, 1)[0] == ids[233]);
        CHECK(tv1x_snapshot_bytes(regrown, "ggml-vector-index-tv1x-regrown-a.tvim") ==
              tv1x_snapshot_bytes(tv, "ggml-vector-index-tv1x-regrown-b.tvim"));
        ggml_vec_index_free(regrown);
        ggml_vec_index_free(tv);
    }
    // A calibrate never licenses overwriting another writer's commits.
    {
        std::filesystem::remove(path);
        auto * a = ggml_vec_index_create_turbovec(dim, 2);
        CHECK(ggml_vec_index_add(a, rows.data(), 100, ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_sync(a, path.c_str()) == GGML_VEC_INDEX_OK);
        auto * b = ggml_vec_index_load(path.c_str());
        CHECK(ggml_vec_index_add(b, rows.data() + 100 * dim, 10, ids.data() + 100) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_sync(b, path.c_str()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_turbovec_calibrate(a, rows.data(), 100) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_sync(a, path.c_str()) == GGML_VEC_INDEX_E_IO);
        auto * check = ggml_vec_index_load(path.c_str());
        CHECK(ggml_vec_index_len(check) == 110);
        ggml_vec_index_free(check);
        ggml_vec_index_free(b);
        ggml_vec_index_free(a);
    }
    // Upstream a_queued_compaction_still_respects_the_foreign_file_guard:
    // another writer claimed the path with its own full write.
    {
        std::filesystem::remove(path);
        auto * a = ggml_vec_index_create_turbovec(dim, 4);
        CHECK(ggml_vec_index_turbovec_calibrate(a, rows.data(), 1024) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_add(a, rows.data(), 40, ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_sync(a, path.c_str()) == GGML_VEC_INDEX_OK);
        auto * b = ggml_vec_index_create_turbovec(dim, 4);
        CHECK(ggml_vec_index_turbovec_calibrate(b, rows.data() + 1024 * dim, 1024) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_add(b, rows.data() + 40 * dim, 20, ids.data() + 40) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_sync(b, path.c_str()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_turbovec_calibrate(a, rows.data() + 512 * dim, 1024) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_sync(a, path.c_str()) == GGML_VEC_INDEX_E_IO);
        auto * loaded = ggml_vec_index_load(path.c_str());
        CHECK(tv1x_snapshot_bytes(loaded, "ggml-vector-index-tv1x-guard-a.tvim") ==
              tv1x_snapshot_bytes(b, "ggml-vector-index-tv1x-guard-b.tvim"));
        ggml_vec_index_free(loaded);
        ggml_vec_index_free(b);
        ggml_vec_index_free(a);
    }
#ifndef _WIN32
    // A sync that fails partway drops the binding; the next one rewrites.
    if (::geteuid() != 0) {
        std::filesystem::remove(path);
        auto * tv = ggml_vec_index_create_turbovec(dim, 2);
        CHECK(ggml_vec_index_add(tv, rows.data(), 64, ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_sync(tv, path.c_str()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_add(tv, rows.data() + 64 * dim, 64, ids.data() + 64) == GGML_VEC_INDEX_OK);
        ::chmod(path.c_str(), 0444);
        CHECK(ggml_vec_index_sync(tv, path.c_str()) == GGML_VEC_INDEX_E_IO);
        ::chmod(path.c_str(), 0644);
        CHECK(ggml_vec_index_sync(tv, path.c_str()) == GGML_VEC_INDEX_OK);
        auto * loaded = ggml_vec_index_load(path.c_str());
        CHECK(ggml_vec_index_len(loaded) == 128);
        ggml_vec_index_free(loaded);
        ggml_vec_index_free(tv);
    }
#endif
    // Destination names near NAME_MAX still get a temp sibling.
    {
        const std::string long_path =
            (verbatim_path(std::filesystem::temp_directory_path()) / (std::string(240, 'n') + ".tvim")).string();
        auto * tv = ggml_vec_index_create_turbovec(dim, 2);
        CHECK(ggml_vec_index_add(tv, rows.data(), 10, ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_write(tv, long_path.c_str()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_sync(tv, long_path.c_str()) == GGML_VEC_INDEX_OK);
        CHECK(!has_snapshot_tmp(long_path));
        std::filesystem::remove(long_path);
        ggml_vec_index_free(tv);
    }
    std::filesystem::remove(path);
}

#ifdef GGML_VEC_INDEX_TEST_HOOKS
// An add that fails partway on a sync-bound index leaves nothing behind:
// the next sync commits exactly the rows that were added (upstream #388).
void check_turbovec_1x_failed_add_then_sync() {
    constexpr int dim = 64;
    const std::vector<float> rows = tv1x_rows(kTv1xSeedDb, 200, dim);
    const std::vector<uint64_t> ids = tv1x_ids(200);
    const std::string path = (std::filesystem::temp_directory_path() / "ggml-vector-index-tv1x-oom.tvim").string();
    std::filesystem::remove(path);
    auto * tv = ggml_vec_index_create_turbovec(dim, 2);
    CHECK(ggml_vec_index_add(tv, rows.data(), 70, ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_sync(tv, path.c_str()) == GGML_VEC_INDEX_OK);
    int failures = 0;
    for (int64_t countdown = 1; countdown < 64; ++countdown) {
        ggml_vec_index_test_set_oom_countdown(countdown);
        const int rc = ggml_vec_index_add(tv, rows.data() + 70 * dim, 50, ids.data() + 70);
        ggml_vec_index_test_set_oom_countdown(-1);
        if (rc == GGML_VEC_INDEX_OK) {
            break;
        }
        CHECK(rc == GGML_VEC_INDEX_E_OOM);
        ++failures;
        CHECK(ggml_vec_index_len(tv) == 70);
        CHECK(ggml_vec_index_sync(tv, path.c_str()) == GGML_VEC_INDEX_OK);
        auto * loaded = ggml_vec_index_load(path.c_str());
        CHECK(ggml_vec_index_len(loaded) == 70);
        ggml_vec_index_free(loaded);
    }
    CHECK(failures > 0);
    CHECK(ggml_vec_index_len(tv) == 120);
    CHECK(ggml_vec_index_sync(tv, path.c_str()) == GGML_VEC_INDEX_OK);
    auto * loaded = ggml_vec_index_load(path.c_str());
    CHECK(ggml_vec_index_len(loaded) == 120);
    CHECK(tv1x_snapshot_bytes(loaded, "ggml-vector-index-tv1x-oom-a.tvim") ==
          tv1x_snapshot_bytes(tv, "ggml-vector-index-tv1x-oom-b.tvim"));
    ggml_vec_index_free(loaded);
    ggml_vec_index_free(tv);
    std::filesystem::remove(path);
}

// A sync that fails keeps its binding, so the retry still refuses a file
// another writer committed to in the meantime instead of rewriting it with
// that writer's commits lost.
void check_turbovec_1x_failed_sync_keeps_binding() {
    constexpr int dim = 32;
    const std::vector<float> rows = tv1x_rows(kTv1xSeedDb, 128, dim);
    const std::vector<uint64_t> ids = tv1x_ids(128);
    const std::string path = (std::filesystem::temp_directory_path() / "ggml-vector-index-tv1x-failed-sync.tvim").string();
    auto add = [&](ggml_vec_index_t * idx, size_t from, size_t to) {
        CHECK(ggml_vec_index_add(idx, rows.data() + from * dim, static_cast<int>(to - from), ids.data() + from) ==
              GGML_VEC_INDEX_OK);
    };
    // Another handle loads the file, removes ids[0] and commits.
    auto commit_remove_elsewhere = [&]() {
        auto * other = ggml_vec_index_load(path.c_str());
        CHECK(other != nullptr);
        CHECK(ggml_vec_index_remove(other, ids[0]) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_sync(other, path.c_str()) == GGML_VEC_INDEX_OK);
        ggml_vec_index_free(other);
    };
    auto check_removal_kept = [&](int64_t len) {
        auto * reloaded = ggml_vec_index_load(path.c_str());
        CHECK(reloaded != nullptr);
        CHECK(ggml_vec_index_len(reloaded) == len);
        CHECK(ggml_vec_index_contains(reloaded, ids[0]) == 0);
        ggml_vec_index_free(reloaded);
    };

    // Out of memory before anything is written, after another commit.
    {
        std::filesystem::remove(path);
        auto * tv = ggml_vec_index_create_turbovec(dim, 2);
        add(tv, 0, 64);
        CHECK(ggml_vec_index_sync(tv, path.c_str()) == GGML_VEC_INDEX_OK);
        commit_remove_elsewhere();
        ggml_vec_index_test_set_oom_countdown(0);
        CHECK(ggml_vec_index_sync(tv, path.c_str()) == GGML_VEC_INDEX_E_OOM);
        ggml_vec_index_test_set_oom_countdown(-1);
        CHECK(ggml_vec_index_sync(tv, path.c_str()) == GGML_VEC_INDEX_E_IO);
        check_removal_kept(63);
        ggml_vec_index_free(tv);
    }
    // A write that fails partway, then another commit at the same
    // generation the failed sync was writing.
    {
        std::filesystem::remove(path);
        auto * tv = ggml_vec_index_create_turbovec(dim, 2);
        add(tv, 0, 64);
        CHECK(ggml_vec_index_sync(tv, path.c_str()) == GGML_VEC_INDEX_OK);
        add(tv, 64, 96);
        ggml_vec_index_test_set_write_fail_after(8);
        CHECK(ggml_vec_index_sync(tv, path.c_str()) == GGML_VEC_INDEX_E_IO);
        ggml_vec_index_test_set_write_fail_after(-1);
        commit_remove_elsewhere();
        CHECK(ggml_vec_index_sync(tv, path.c_str()) == GGML_VEC_INDEX_E_IO);
        check_removal_kept(63);
        ggml_vec_index_free(tv);
    }
    // With no other writer, the retry commits everything.
    {
        std::filesystem::remove(path);
        auto * tv = ggml_vec_index_create_turbovec(dim, 2);
        add(tv, 0, 64);
        CHECK(ggml_vec_index_sync(tv, path.c_str()) == GGML_VEC_INDEX_OK);
        add(tv, 64, 96);
        ggml_vec_index_test_set_write_fail_after(8);
        CHECK(ggml_vec_index_sync(tv, path.c_str()) == GGML_VEC_INDEX_E_IO);
        ggml_vec_index_test_set_write_fail_after(-1);
        CHECK(ggml_vec_index_sync(tv, path.c_str()) == GGML_VEC_INDEX_OK);
        auto * reloaded = ggml_vec_index_load(path.c_str());
        CHECK(reloaded != nullptr);
        CHECK(ggml_vec_index_len(reloaded) == 96);
        ggml_vec_index_free(reloaded);
        ggml_vec_index_free(tv);
    }
    std::filesystem::remove(path);
}

// A commit header that reaches disk before its data is detected and the
// load falls back; the recovered index's next commit wipes the abandoned
// header before reusing its slot.
void check_turbovec_1x_reordered_and_stale_commits() {
    constexpr int dim = 64;
    const std::vector<float> rows = tv1x_rows(kTv1xSeedDb, 300, dim);
    const std::vector<uint64_t> ids = tv1x_ids(300);
    const std::string path = (std::filesystem::temp_directory_path() / "ggml-vector-index-tv1x-reorder.tvim").string();
    std::filesystem::remove(path);
    auto * tv = ggml_vec_index_create_turbovec(dim, 4);
    CHECK(ggml_vec_index_add(tv, rows.data(), 100, ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_sync(tv, path.c_str()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_remove(tv, ids[7]) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_add(tv, rows.data() + 100 * dim, 100, ids.data() + 100) == GGML_VEC_INDEX_OK);
    std::vector<std::vector<std::pair<uint64_t, std::vector<uint8_t>>>> batches;
    CHECK(turbovec_sync_plan_for_test(tv, path.c_str(), batches) == GGML_VEC_INDEX_OK);
    CHECK(batches.size() == 1);
    auto apply = [&](const std::vector<uint8_t> & base, bool with_units, bool damage_unit) {
        std::vector<uint8_t> bytes = base;
        const auto & batch = batches.back();
        for (size_t op = 0; op < batch.size(); ++op) {
            const bool is_header = op + 1 == batch.size();
            if (!is_header && !with_units) {
                continue;
            }
            const auto & write = batch[op];
            if (bytes.size() < write.first + write.second.size()) {
                bytes.resize(static_cast<size_t>(write.first + write.second.size()), 0);
            }
            std::memcpy(bytes.data() + write.first, write.second.data(), write.second.size());
            if (!is_header && damage_unit && op == 0) {
                bytes[static_cast<size_t>(write.first) + 5] ^= 0x40;
            }
        }
        return bytes;
    };
    const std::vector<uint8_t> committed = read_file_bytes(path);
    // Only the header lands: it names units past the end of the file and
    // is not a commit.
    write_file_bytes(path, apply(committed, false, false));
    auto * header_only = ggml_vec_index_load(path.c_str());
    CHECK(header_only != nullptr && ggml_vec_index_len(header_only) == 100);
    ggml_vec_index_free(header_only);
    // Everything lands but a unit is damaged: the header parses, its delta
    // fails, and the load falls back to the previous commit.
    write_file_bytes(path, apply(committed, true, true));
    auto * recovered = ggml_vec_index_load(path.c_str());
    CHECK(recovered != nullptr);
    CHECK(ggml_vec_index_len(recovered) == 100);
    CHECK(ggml_vec_index_contains(recovered, ids[7]) == 1);
    // Generation 1 again, into the slot that still holds the abandoned one.
    CHECK(ggml_vec_index_remove(recovered, ids[9]) == GGML_VEC_INDEX_OK);
    batches.clear();
    CHECK(turbovec_sync_plan_for_test(recovered, path.c_str(), batches) == GGML_VEC_INDEX_OK);
    CHECK(batches.size() == 2);
    CHECK(ggml_vec_index_sync(recovered, path.c_str()) == GGML_VEC_INDEX_OK);
    auto * reloaded = ggml_vec_index_load(path.c_str());
    CHECK(ggml_vec_index_len(reloaded) == 99);
    CHECK(ggml_vec_index_contains(reloaded, ids[9]) == 0);
    CHECK(ggml_vec_index_contains(reloaded, ids[7]) == 1);
    ggml_vec_index_free(reloaded);
    ggml_vec_index_free(recovered);
    ggml_vec_index_free(tv);
    std::filesystem::remove(path);
}
#endif

#ifdef GGML_VEC_INDEX_TEST_HOOKS
// The portable permute-dot kernel and the SDOT one accumulate the same
// integers, in both blocked layouts, including after swap-removes.
void check_turbovec_1x_permute_dot_kernels() {
    constexpr int dim = 200;
    const std::vector<float> rows = tv1x_rows(kTv1xSeedDb, 150, dim);
    const std::vector<uint64_t> ids = tv1x_ids(150);
    const std::vector<float> queries = tv1x_rows(kTv1xSeedQuery, 6, dim);
    turbovec_force_vector_major_for_test(1);
    auto * tv = ggml_vec_index_create_turbovec(dim, 4);
    turbovec_force_vector_major_for_test(-1);
    CHECK(ggml_vec_index_add(tv, rows.data(), 150, ids.data()) == GGML_VEC_INDEX_OK);
    for (const size_t i : { 4, 77, 149, 30 }) {
        CHECK(ggml_vec_index_remove(tv, ids[i]) == GGML_VEC_INDEX_OK);
    }
    std::vector<float> scores[2];
    std::vector<uint64_t> out[2];
    for (int scalar = 0; scalar < 2; ++scalar) {
        turbovec_disable_dot_product_for_test(scalar);
        scores[scalar].resize(6 * 12 + 12);
        out[scalar].resize(scores[scalar].size());
        CHECK(ggml_vec_index_search(tv, queries.data(), 6, 12, scores[scalar].data(), out[scalar].data()) ==
              GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search(tv, queries.data(), 1, 12, scores[scalar].data() + 72, out[scalar].data() + 72) ==
              GGML_VEC_INDEX_OK);
    }
    turbovec_disable_dot_product_for_test(0);
    CHECK(scores[0] == scores[1]);
    CHECK(out[0] == out[1]);
    ggml_vec_index_free(tv);
}
#endif

#ifdef GGML_VEC_INDEX_TEST_HOOKS
// Puts every kernel and layout override back to following the host.
void tv1x_reset_kernel_overrides() {
    turbovec_force_vector_major_for_test(-1);
    turbovec_force_vm8_for_test(-1);
    turbovec_force_vm8_2bit_for_test(-1);
    turbovec_disable_i8mm_for_test(0);
    turbovec_disable_dot_product_for_test(0);
    turbovec_force_batches_for_test(0);
}

// An index created under the given layout overrides (vector-major forced on).
ggml_vec_index_t * tv1x_create_with_layout(int dim, int bits, int vm8, int vm8_2bit) {
    turbovec_force_vector_major_for_test(1);
    turbovec_force_vm8_for_test(vm8);
    turbovec_force_vm8_2bit_for_test(vm8_2bit);
    auto * idx = ggml_vec_index_create_turbovec(dim, bits);
    CHECK(idx != nullptr);
    return idx;
}

// Every query count upstream batches differently (12 + 12 + 1, 12 + 8,
// 12 + 4, 8 + 2, 4 + 1 + 1, 4, a lone query), then a filtered batch; all results
// concatenated.
std::pair<std::vector<float>, std::vector<uint64_t>> tv1x_search_batches(
        ggml_vec_index_t * idx,
        const std::vector<float> & queries,
        int dim,
        const std::vector<uint64_t> & allow) {
    constexpr int k = 7;
    std::pair<std::vector<float>, std::vector<uint64_t>> all;
    for (const int n_q : { 25, 20, 16, 10, 6, 4, 1, -20 }) {
        const int n = std::abs(n_q);
        CHECK(static_cast<size_t>(n) * static_cast<size_t>(dim) <= queries.size());
        std::vector<float> scores(static_cast<size_t>(n * k));
        std::vector<uint64_t> ids(scores.size());
        if (n_q < 0) {
            CHECK(ggml_vec_index_search_filtered(
                      idx, queries.data(), n, k, allow.data(), static_cast<int>(allow.size()), scores.data(),
                      ids.data()) == GGML_VEC_INDEX_OK);
        } else {
            CHECK(ggml_vec_index_search(idx, queries.data(), n, k, scores.data(), ids.data()) == GGML_VEC_INDEX_OK);
        }
        all.first.insert(all.first.end(), scores.begin(), scores.end());
        all.second.insert(all.second.end(), ids.begin(), ids.end());
    }
    return all;
}

// Upstream picks the vm8 layout on aarch64 with i8mm, for 4-bit indexes
// whose byte-group count is a multiple of 8 and, under TURBOVEC_2BIT_VM8,
// for such 2-bit ones; it needs the vector-major layout underneath.
void check_turbovec_1x_vm8_layout_choice() {
    struct Case { int bits; int dim; int vm; int vm8; int vm8_2bit; int layout; };
    static constexpr Case cases[] = {
        { 4, 128, 1, 1, 0, 2 }, { 4, 200, 1, 1, 0, 1 }, { 4, 136, 1, 1, 1, 1 }, { 4, 128, 1, 0, 0, 1 },
        { 4, 128, 0, 1, 1, 0 }, { 2, 128, 1, 1, 0, 0 }, { 2, 128, 1, 1, 1, 2 }, { 2, 96, 1, 1, 1, 2 },
        { 2, 200, 1, 1, 1, 0 }, { 2, 144, 1, 1, 1, 0 }, { 2, 128, 1, 0, 1, 0 }, { 2, 128, 0, 1, 1, 0 },
        { 3, 128, 1, 1, 1, 0 },
    };
    for (const Case & c : cases) {
        turbovec_force_vector_major_for_test(c.vm);
        turbovec_force_vm8_for_test(c.vm8);
        turbovec_force_vm8_2bit_for_test(c.vm8_2bit);
        auto * idx = ggml_vec_index_create_turbovec(c.dim, c.bits);
        CHECK(idx != nullptr);
        CHECK(turbovec_layout_for_test(idx) == c.layout);
        // The layout is fixed at creation: a later override does not move it.
        turbovec_force_vm8_for_test(1 - c.vm8);
        const std::vector<float> rows = tv1x_rows(kTv1xSeedDb, 40, static_cast<size_t>(c.dim));
        const std::vector<uint64_t> ids = tv1x_ids(40);
        CHECK(ggml_vec_index_add(idx, rows.data(), 40, ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(turbovec_layout_for_test(idx) == c.layout);
        ggml_vec_index_free(idx);
    }
    // The opt-in replaces the scan the planes layout needs (aarch64 only).
    turbovec_force_vm8_2bit_for_test(1);
    turbovec_force_planes_for_test(-1, 0);
    CHECK(!turbovec_planes_for(2, 128));
    CHECK(!turbovec_planes_for(4, 128));
    turbovec_force_vm8_2bit_for_test(0);
#if defined(__aarch64__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
    CHECK(turbovec_planes_for(2, 128));
#else
    CHECK(!turbovec_planes_for(2, 128));
#endif
    tv1x_reset_kernel_overrides();
}

// The SMMLA kernels (vm8 single query, vm8 and vector-major batches) and the
// SDOT and scalar ones accumulate the same integers, so every layout and
// kernel returns the same scores, at group counts with and without whole
// vm8 units, after swap-removes.
void check_turbovec_1x_vm8_kernel_parity() {
    turbovec_force_batches_for_test(1);
    for (const int dim : { 64, 136, 200, 256, 768 }) {
        const size_t d = static_cast<size_t>(dim);
        const std::vector<float> rows = tv1x_rows(kTv1xSeedDb, 150, d);
        const std::vector<uint64_t> ids = tv1x_ids(150);
        const std::vector<float> queries = tv1x_rows(kTv1xSeedQuery, 25, d);
        std::vector<uint64_t> allow;
        for (size_t i = 0; i < ids.size(); i += 3) {
            allow.push_back(ids[i]);
        }
        std::vector<std::pair<std::vector<float>, std::vector<uint64_t>>> results;
        for (const int vm8 : { 1, 0 }) {
            auto * tv = tv1x_create_with_layout(dim, 4, vm8, 0);
            CHECK(turbovec_layout_for_test(tv) == (vm8 != 0 && dim % 16 == 0 ? 2 : 1));
            CHECK(ggml_vec_index_add(tv, rows.data(), 150, ids.data()) == GGML_VEC_INDEX_OK);
            for (const size_t i : { 4, 77, 149, 30 }) {
                CHECK(ggml_vec_index_remove(tv, ids[i]) == GGML_VEC_INDEX_OK);
            }
            // SMMLA, SDOT, then scalar.
            for (const int kernel : { 0, 1, 2 }) {
                turbovec_disable_i8mm_for_test(kernel >= 1);
                turbovec_disable_dot_product_for_test(kernel >= 2);
                results.push_back(tv1x_search_batches(tv, queries, dim, allow));
            }
            turbovec_disable_i8mm_for_test(0);
            turbovec_disable_dot_product_for_test(0);
            ggml_vec_index_free(tv);
        }
        for (const auto & r : results) {
            CHECK(r.first == results[0].first);
            CHECK(r.second == results[0].second);
        }
    }
    tv1x_reset_kernel_overrides();
}

// Every mutation of a vm8 cache (append into a partial tail block,
// swap-remove including the last row, rolled-back appends, calibrate)
// leaves the bytes a fresh build of the same rows has, and snapshots stay
// independent of the layout.
void check_turbovec_1x_vm8_mutations() {
    turbovec_force_batches_for_test(1);
    constexpr int dim = 128;
    const size_t d = static_cast<size_t>(dim);
    const std::vector<float> rows = tv1x_rows(kTv1xSeedDb, 260, d);
    const std::vector<uint64_t> ids = tv1x_ids(260);
    const std::vector<float> queries = tv1x_rows(kTv1xSeedQuery, 25, d);
    const std::vector<float> calib = tv1x_rows(kTv1xSeedCalib, kTv1xNCalib, d);
    std::vector<uint64_t> allow;
    for (size_t i = 0; i < ids.size(); i += 3) {
        allow.push_back(ids[i]);
    }
    for (const int bits : { 4, 2 }) {
        auto * vm8 = tv1x_create_with_layout(dim, bits, 1, 1);
        CHECK(turbovec_layout_for_test(vm8) == 2);
        // The same mutations on the sequential layout, for the snapshots.
        turbovec_force_vector_major_for_test(0);
        auto * seq = ggml_vec_index_create_turbovec(dim, bits);
        CHECK(turbovec_layout_for_test(seq) == 0);
        turbovec_force_vector_major_for_test(1);
        // A fresh vm8 index built from vm8's snapshot holds the same cache.
        auto check_against_rebuilt = [&](const char * name) {
            const std::vector<uint8_t> bytes = tv1x_snapshot_bytes(vm8, name);
            CHECK(bytes == tv1x_snapshot_bytes(seq, name));
            const std::string path = (std::filesystem::temp_directory_path() / name).string();
            write_file_bytes(path, bytes);
            auto * rebuilt = ggml_vec_index_load(path.c_str());
            std::filesystem::remove(path);
            CHECK(rebuilt != nullptr);
            CHECK(turbovec_layout_for_test(rebuilt) == 2);
            CHECK(turbovec_blocked_hash_for_test(rebuilt) == turbovec_blocked_hash_for_test(vm8));
            CHECK(tv1x_search_batches(rebuilt, queries, dim, allow) == tv1x_search_batches(vm8, queries, dim, allow));
            ggml_vec_index_free(rebuilt);
        };
        for (auto * idx : { vm8, seq }) {
            CHECK(ggml_vec_index_add(idx, rows.data(), 70, ids.data()) == GGML_VEC_INDEX_OK);
        }
        check_against_rebuilt("ggml-vector-index-tv1x-vm8-a.tvim");
        // Appends that fail part way roll back into the partial tail block.
        const uint64_t before = turbovec_blocked_hash_for_test(vm8);
        int failures = 0;
        for (int64_t countdown = 1; countdown < 64; ++countdown) {
            ggml_vec_index_test_set_oom_countdown(countdown);
            const int rc = ggml_vec_index_add(vm8, rows.data() + 70 * d, 50, ids.data() + 70);
            ggml_vec_index_test_set_oom_countdown(-1);
            if (rc == GGML_VEC_INDEX_OK) {
                break;
            }
            CHECK(rc == GGML_VEC_INDEX_E_OOM);
            CHECK(ggml_vec_index_len(vm8) == 70);
            CHECK(turbovec_blocked_hash_for_test(vm8) == before);
            ++failures;
        }
        CHECK(failures > 0);
        CHECK(ggml_vec_index_add(seq, rows.data() + 70 * d, 50, ids.data() + 70) == GGML_VEC_INDEX_OK);
        check_against_rebuilt("ggml-vector-index-tv1x-vm8-b.tvim");
        // Swap-removes: the last row, rows in other blocks, the new tail.
        for (const size_t i : { 119, 4, 77, 31, 0, 100, 64 }) {
            for (auto * idx : { vm8, seq }) {
                CHECK(ggml_vec_index_remove(idx, ids[i]) == GGML_VEC_INDEX_OK);
            }
        }
        check_against_rebuilt("ggml-vector-index-tv1x-vm8-c.tvim");
        // Remove down to an empty block and grow again.
        for (size_t i = 120; i < 260; ++i) {
            for (auto * idx : { vm8, seq }) {
                CHECK(ggml_vec_index_add(idx, rows.data() + i * d, 1, ids.data() + i) == GGML_VEC_INDEX_OK);
            }
        }
        for (size_t i = 200; i < 260; ++i) {
            for (auto * idx : { vm8, seq }) {
                CHECK(ggml_vec_index_remove(idx, ids[i]) == GGML_VEC_INDEX_OK);
            }
        }
        check_against_rebuilt("ggml-vector-index-tv1x-vm8-d.tvim");
        // Calibrating re-encodes every row into the cache.
        for (auto * idx : { vm8, seq }) {
            CHECK(ggml_vec_index_turbovec_calibrate(idx, calib.data(), kTv1xNCalib) == GGML_VEC_INDEX_OK);
        }
        check_against_rebuilt("ggml-vector-index-tv1x-vm8-e.tvim");
        ggml_vec_index_free(seq);
        ggml_vec_index_free(vm8);
    }
    tv1x_reset_kernel_overrides();
}

// The 2-bit vm8 opt-in: a lone query (and any query outside a batch of 12,
// 8 or 4) scans the vm8 codes with the LUT kernel, bit-identical to the
// sequential layout; batches score with SMMLA, matched by the scalar
// fallback.
void check_turbovec_1x_vm8_2bit() {
    turbovec_force_batches_for_test(1);
    for (const int dim : { 96, 128, 768 }) {
        const size_t d = static_cast<size_t>(dim);
        const std::vector<float> rows = tv1x_rows(kTv1xSeedDb, 150, d);
        const std::vector<uint64_t> ids = tv1x_ids(150);
        const std::vector<float> queries = tv1x_rows(kTv1xSeedQuery, 25, d);
        std::vector<uint64_t> allow;
        for (size_t i = 0; i < ids.size(); i += 3) {
            allow.push_back(ids[i]);
        }
        auto * vm8 = tv1x_create_with_layout(dim, 2, 1, 1);
        auto * seq = tv1x_create_with_layout(dim, 2, 1, 0);
        CHECK(turbovec_layout_for_test(vm8) == 2);
        CHECK(turbovec_layout_for_test(seq) == 0);
        for (auto * idx : { vm8, seq }) {
            CHECK(ggml_vec_index_add(idx, rows.data(), 150, ids.data()) == GGML_VEC_INDEX_OK);
            CHECK(ggml_vec_index_remove(idx, ids[17]) == GGML_VEC_INDEX_OK);
        }
        constexpr int k = 9;
        // Two and three queries run one by one under upstream's aarch64
        // batching, the only place a vm8 index exists; an x86 build pads
        // such a tail into a batch, so only the lone query is compared.
#if GGML_VEC_INDEX_USE_NEON
        const std::vector<int> query_counts = { 1, 2, 3 };
#else
        const std::vector<int> query_counts = { 1 };
#endif
        for (const int n_q : query_counts) {
            std::vector<float> scores[2];
            std::vector<uint64_t> out[2];
            for (int i = 0; i < 2; ++i) {
                scores[i].resize(static_cast<size_t>(n_q * k));
                out[i].resize(scores[i].size());
                CHECK(ggml_vec_index_search(i == 0 ? vm8 : seq, queries.data(), n_q, k, scores[i].data(), out[i].data()) ==
                      GGML_VEC_INDEX_OK);
            }
            CHECK(scores[0] == scores[1]);
            CHECK(out[0] == out[1]);
        }
        const auto smmla = tv1x_search_batches(vm8, queries, dim, allow);
        turbovec_disable_i8mm_for_test(1);
        CHECK(tv1x_search_batches(vm8, queries, dim, allow) == smmla);
        turbovec_disable_i8mm_for_test(0);
        ggml_vec_index_free(seq);
        ggml_vec_index_free(vm8);
    }
    tv1x_reset_kernel_overrides();
}

// Upstream's batched searches (tests/turbovec-1x-golden-vm.inc) under every
// kernel this port has for the layout: 4-bit permute-dot, 2-bit LUT, and
// 2-bit under TURBOVEC_2BIT_VM8.
void check_turbovec_1x_batch_goldens() {
    auto run = [](const Tv1xBatchCase & c, int vm8, int vm8_2bit, int kernel, int force_batches = 1) {
        turbovec_force_batches_for_test(force_batches);
        const size_t d = static_cast<size_t>(c.dim);
        std::vector<float> db = tv1x_rows(kTv1xSeedDb, kTv1xNDb, d);
        std::fill(db.begin(), db.begin() + static_cast<std::ptrdiff_t>(d), 0.0f);
        const std::vector<float> queries = tv1x_rows(kTv1xSeedQuery, static_cast<size_t>(c.n_q), d);
        const std::vector<uint64_t> ids = tv1x_ids(kTv1xNDb);
        std::vector<uint64_t> allow;
        for (size_t i = 0; i < ids.size(); i += 3) {
            allow.push_back(ids[i]);
        }
        auto * idx = tv1x_create_with_layout(c.dim, c.bits, vm8, vm8_2bit);
        if (c.calibrated != 0) {
            const std::vector<float> calib = tv1x_rows(kTv1xSeedCalib, kTv1xNCalib, d);
            CHECK(ggml_vec_index_turbovec_calibrate(idx, calib.data(), kTv1xNCalib) == GGML_VEC_INDEX_OK);
        }
        CHECK(ggml_vec_index_add(idx, db.data(), kTv1xNDb, ids.data()) == GGML_VEC_INDEX_OK);
        turbovec_disable_i8mm_for_test(kernel >= 1);
        turbovec_disable_dot_product_for_test(kernel >= 2);
        std::vector<float> scores(static_cast<size_t>(c.n_q * kTv1xK));
        std::vector<uint64_t> out(scores.size());
        if (c.filtered != 0) {
            CHECK(ggml_vec_index_search_filtered(
                      idx, queries.data(), c.n_q, kTv1xK, allow.data(), static_cast<int>(allow.size()), scores.data(),
                      out.data()) == GGML_VEC_INDEX_OK);
        } else {
            CHECK(ggml_vec_index_search(idx, queries.data(), c.n_q, kTv1xK, scores.data(), out.data()) ==
                  GGML_VEC_INDEX_OK);
        }
        turbovec_disable_i8mm_for_test(0);
        turbovec_disable_dot_product_for_test(0);
        for (size_t i = 0; i < out.size(); ++i) {
            CHECK(out[i] == c.ids[i]);
            CHECK(scores[i] == c.scores[i]);
        }
        ggml_vec_index_free(idx);
    };
    for (const Tv1xBatchCase & c : kTv1xBatchCases) {
        for (const int vm8 : { 1, 0 }) {
            // SMMLA, SDOT, scalar; the 2-bit LUT ignores the 4-bit kernels,
            // and the opt-in leaves a geometry without whole vm8 units alone.
            for (const int kernel : { 0, 1, 2 }) {
                run(c, vm8, 0, kernel);
            }
        }
        if (c.bits == 2 && (c.dim / 4) % 8 != 0) {
            run(c, 1, 1, 0);
        }
    }
    // The fused batches are NEON-only in this port; elsewhere every query
    // runs alone, and upstream never takes the vm8 layout there.
#if defined(__aarch64__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
    for (const Tv1xBatchCase & c : kTv1xVm8TwoBitCases) {
        for (const int kernel : { 0, 1 }) {
            run(c, 1, 1, kernel);
        }
        // Batching follows upstream on its own: a 2-bit vm8 batch scores
        // differently from the single-query scan.
        run(c, 1, 1, 0, 0);
    }
#else
    (void) kTv1xVm8TwoBitCases;
#endif
    tv1x_reset_kernel_overrides();
}
#endif

#ifdef GGML_VEC_INDEX_TEST_HOOKS
// The staged ranking's low-plane weights: a query so small that its weight
// reciprocal overflows still gets upstream's saturated weights (the top
// weight for every nonzero coordinate, unit 0), with no floating-point flag.
void check_turbovec_1x_low_planes_tiny() {
    constexpr int dim = 32;
    constexpr int32_t top = 63;
    for (const float scale : { 1.0f, 1e-38f }) {
        std::vector<float> q(dim);
        int32_t want_sum = 0;
        int nonzero = 0;
        for (int i = 0; i < dim; ++i) {
            const float mag = static_cast<float>(2 + i % 4);
            q[i] = i % 5 == 0 ? 0.0f : (i % 3 == 0 ? -mag : mag) * scale;
            if (q[i] != 0.0f) {
                ++nonzero;
                // At scale 1, q_max is 5: weights trunc(mag * 63 / 5 + 0.5).
                const int32_t w = scale == 1.0f ? static_cast<int32_t>(mag * (63.0f / 5.0f) + 0.5f) : top;
                want_sum += q[i] < 0.0f ? -w : w;
            }
        }
        float unit = -1.0f;
        std::vector<uint64_t> masks;
        std::feclearexcept(FE_ALL_EXCEPT);
        const int32_t sum_w = turbovec_low_planes_for_test(q.data(), dim, 1.5f, &unit, &masks);
        CHECK(std::fetestexcept(FE_OVERFLOW | FE_INVALID) == 0);
        CHECK(sum_w == want_sum);
        if (scale == 1.0f) {
            CHECK(unit == 1.5f / (63.0f / 5.0f));
        } else {
            CHECK(unit == 0.0f);
            int bits = 0;
            for (const uint64_t word : masks) {
                bits += static_cast<int>(std::bitset<64>(word).count());
            }
            CHECK(bits == 6 * nonzero);
        }
    }
}

// Staged search (upstream planes search): returned scores are the full
// scan's for the same ids, and on aarch64 the results are upstream's.
void check_turbovec_1x_staged() {
    const int n = kTv1xStagedN;
    const size_t dim = static_cast<size_t>(kTv1xStagedDim);
    const int k = kTv1xStagedK;
    const std::vector<float> db = tv1x_rows(kTv1xSeedDb, static_cast<size_t>(n), dim);
    const std::vector<uint64_t> ids = tv1x_ids(static_cast<size_t>(n));
    const std::vector<float> queries = tv1x_rows(kTv1xSeedQuery, 3, dim);
    const std::vector<float> calib = tv1x_rows(kTv1xSeedCalib, kTv1xNCalib, dim);
    std::vector<uint64_t> allow;
    for (size_t i = 0; i < ids.size(); i += 3) {
        allow.push_back(ids[i]);
    }
    turbovec_force_planes_for_test(1, 0);
    turbovec_force_vector_major_for_test(1);
    for (const int bits : { 2, 4 }) {
        for (const int calibrated : { 0, 1 }) {
            auto * tv = ggml_vec_index_create_turbovec(static_cast<int>(dim), bits);
            if (calibrated != 0) {
                CHECK(ggml_vec_index_turbovec_calibrate(tv, calib.data(), kTv1xNCalib) == GGML_VEC_INDEX_OK);
            }
            CHECK(ggml_vec_index_add(tv, db.data(), n, ids.data()) == GGML_VEC_INDEX_OK);
            // At this size upstream stores units as bit planes.
            const std::vector<uint8_t> image = tv1x_snapshot_bytes(tv, "ggml-vector-index-tv1x-staged.tvim");
            CHECK(image[7] == 1);
            bool snapshot_found = false;
            for (const Tv1xStagedSnapshot & snap : kTv1xStagedSnapshots) {
                if (snap.bits == bits && snap.calibrated == calibrated) {
                    snapshot_found = true;
                    CHECK(image.size() == snap.len);
                    CHECK(tv1x_fnv(image.data(), image.size()) == snap.hash);
                }
            }
            CHECK(snapshot_found);
            auto * reloaded = tv1x_load_bytes(image);
            CHECK(reloaded != nullptr);
            CHECK(tv1x_snapshot_bytes(reloaded, "ggml-vector-index-tv1x-staged-2.tvim") == image);
            ggml_vec_index_free(reloaded);
            for (const int nq : { 1, 3 }) {
                for (const int filtered : { 0, 1 }) {
                    auto search = [&](std::vector<float> & scores, std::vector<uint64_t> & out) {
                        scores.assign(static_cast<size_t>(3 * k), 0.0f);
                        out.assign(scores.size(), 0);
                        for (int q = 0; q < 3; q += nq) {
                            const float * query = queries.data() + static_cast<size_t>(q) * dim;
                            float * s = scores.data() + static_cast<size_t>(q * k);
                            uint64_t * o = out.data() + static_cast<size_t>(q * k);
                            CHECK((filtered != 0 ?
                                ggml_vec_index_search_filtered(tv, query, nq, k, allow.data(), static_cast<int>(allow.size()), s, o) :
                                ggml_vec_index_search(tv, query, nq, k, s, o)) == GGML_VEC_INDEX_OK);
                        }
                    };
                    std::vector<float> staged_scores;
                    std::vector<uint64_t> staged_ids;
                    search(staged_scores, staged_ids);
                    // Every returned score is the full scan's for that id.
                    CHECK(ggml_vec_index_turbovec_set_staged(tv, 0) == GGML_VEC_INDEX_OK);
                    std::vector<float> full_scores;
                    std::vector<uint64_t> full_ids;
                    search(full_scores, full_ids);
                    CHECK(ggml_vec_index_turbovec_set_staged(tv, 1) == GGML_VEC_INDEX_OK);
                    size_t shared = 0;
                    for (size_t i = 0; i < staged_ids.size(); ++i) {
                        const size_t q = i / static_cast<size_t>(k);
                        for (size_t j = q * static_cast<size_t>(k); j < (q + 1) * static_cast<size_t>(k); ++j) {
                            if (full_ids[j] == staged_ids[i]) {
                                CHECK(full_scores[j] == staged_scores[i]);
                                ++shared;
                            }
                        }
                    }
                    CHECK(shared * 10 >= staged_ids.size() * 8);
#if defined(__aarch64__) || defined(_M_ARM64)
                    bool found = false;
                    for (const Tv1xStagedCase & c : kTv1xStagedCases) {
                        if (c.bits == bits && c.calibrated == calibrated && c.nq == nq && c.filtered == filtered) {
                            found = true;
                            for (size_t i = 0; i < staged_ids.size(); ++i) {
                                CHECK(staged_ids[i] == c.ids[i]);
                                CHECK(staged_scores[i] == c.scores[i]);
                            }
                        }
                    }
                    CHECK(found);
#endif
                }
            }
            ggml_vec_index_free(tv);
        }
    }
#if defined(__aarch64__) || defined(_M_ARM64)
    // A batch in which only query 0's sampled seed runs short: upstream
    // rescans it with the single-query sign tables, and so does the port.
    {
        const int short_k = kTv1xStagedShortK;
        std::vector<float> short_db = db;
        const std::vector<float> noise = tv1x_rows(kTv1xStagedShortNoiseSeed, 48 * 32, dim);
        const size_t stride = (static_cast<size_t>(n) / 32) / 48;
        for (size_t i = 0; i < 48; ++i) {
            const size_t block = i * stride + stride / 2;
            for (size_t lane = 0; lane < 32; ++lane) {
                const size_t row = block * 32 + lane;
                for (size_t c = 0; c < dim; ++c) {
                    short_db[row * dim + c] = queries[c] + 0.05f * noise[(i * 32 + lane) * dim + c];
                }
            }
        }
        for (const Tv1xStagedShortCase & c : kTv1xStagedShortCases) {
            auto * tv = ggml_vec_index_create_turbovec(static_cast<int>(dim), c.bits);
            if (c.calibrated != 0) {
                CHECK(ggml_vec_index_turbovec_calibrate(tv, calib.data(), kTv1xNCalib) == GGML_VEC_INDEX_OK);
            }
            CHECK(ggml_vec_index_add(tv, short_db.data(), n, ids.data()) == GGML_VEC_INDEX_OK);
            std::vector<float> scores(static_cast<size_t>(3 * short_k));
            std::vector<uint64_t> out(scores.size());
            // The rescan of the short query splits its blocks above one thread.
            for (const int threads : { 1, 2, 4, 8 }) {
                ggml_vec_index_set_threads(tv, threads);
                CHECK(ggml_vec_index_search(tv, queries.data(), 3, short_k, scores.data(), out.data()) ==
                      GGML_VEC_INDEX_OK);
                for (size_t i = 0; i < out.size(); ++i) {
                    CHECK(out[i] == c.ids[i]);
                    CHECK(scores[i] == c.scores[i]);
                }
            }
            ggml_vec_index_free(tv);
        }
    }
#endif
    turbovec_force_vector_major_for_test(-1);
    turbovec_force_planes_for_test(-1, 0);
}
#endif

#ifdef GGML_VEC_INDEX_TEST_HOOKS
// Every search returns the same ids and score bits at any thread count:
// lone queries (which split their blocks), batches (split across queries),
// wide and narrow filters, staged and full scans, and k large enough for
// the ranking to split too.
void check_turbovec_1x_thread_invariance() {
    constexpr int dim = 64;
    constexpr int n = 40000;
    constexpr int nq = 5;
    const std::vector<float> rows = tv1x_rows(kTv1xSeedDb, n, dim);
    const std::vector<uint64_t> ids = tv1x_ids(n);
    const std::vector<float> queries = tv1x_rows(kTv1xSeedQuery, nq, dim);
    const std::vector<float> calib = tv1x_rows(kTv1xSeedCalib, 1024, dim);
    std::vector<uint64_t> wide;
    for (size_t i = 0; i < ids.size(); i += 3) {
        wide.push_back(ids[i]);
    }
    std::vector<uint64_t> narrow;
    for (size_t i = 5; i < ids.size(); i += 811) {
        narrow.push_back(ids[i]);
    }
    turbovec_force_planes_for_test(1, 0);
    turbovec_force_vector_major_for_test(1);
    for (const int bits : { 2, 4 }) {
        for (const int calibrated : { 0, 1 }) {
            auto * tv = ggml_vec_index_create_turbovec(dim, bits);
            if (calibrated != 0) {
                CHECK(ggml_vec_index_turbovec_calibrate(tv, calib.data(), 1024) == GGML_VEC_INDEX_OK);
            }
            CHECK(ggml_vec_index_add(tv, rows.data(), n, ids.data()) == GGML_VEC_INDEX_OK);
            auto run = [&](int threads, std::vector<float> & scores, std::vector<uint64_t> & out) {
                CHECK(ggml_vec_index_set_threads(tv, threads) == GGML_VEC_INDEX_OK);
                scores.clear();
                out.clear();
                auto search = [&](const float * q, int n_q, int k, const std::vector<uint64_t> * allow) {
                    const size_t at = scores.size();
                    scores.resize(at + static_cast<size_t>(n_q * k));
                    out.resize(scores.size());
                    CHECK((allow != nullptr ?
                        ggml_vec_index_search_filtered(
                            tv, q, n_q, k, allow->data(), static_cast<int>(allow->size()), scores.data() + at, out.data() + at) :
                        ggml_vec_index_search(tv, q, n_q, k, scores.data() + at, out.data() + at)) == GGML_VEC_INDEX_OK);
                };
                for (const int staged : { 1, 0 }) {
                    CHECK(ggml_vec_index_turbovec_set_staged(tv, staged) == GGML_VEC_INDEX_OK);
                    for (const int k : { 10, 150 }) {
                        for (int q = 0; q < nq; ++q) {
                            search(queries.data() + static_cast<size_t>(q) * dim, 1, k, nullptr);
                        }
                        search(queries.data(), nq, k, nullptr);
                        search(queries.data(), 1, k, &wide);
                        search(queries.data() + dim, 1, k, &narrow);
                        search(queries.data(), 3, k, &wide);
                    }
                }
                CHECK(ggml_vec_index_turbovec_set_staged(tv, 1) == GGML_VEC_INDEX_OK);
            };
            std::vector<float> ref_scores;
            std::vector<uint64_t> ref_ids;
            run(1, ref_scores, ref_ids);
            for (const int threads : { 2, 8, 3 }) {
                std::vector<float> scores;
                std::vector<uint64_t> out;
                run(threads, scores, out);
                CHECK(out == ref_ids);
                CHECK(scores.size() == ref_scores.size());
                CHECK(std::memcmp(scores.data(), ref_scores.data(), scores.size() * sizeof(float)) == 0);
            }
            ggml_vec_index_free(tv);
        }
    }
    turbovec_force_vector_major_for_test(-1);
    turbovec_force_planes_for_test(-1, 0);
}

// The pool behind every multi-threaded call: tasks run once each, nested
// calls run inline, exceptions reach the caller, concurrent callers share
// the helpers instead of each starting their own, and a forked child that
// searches does not wait on its parent's helpers.
void check_vector_index_pool() {
    for (const int threads : { 1, 2, 3, 8, 64 }) {
        CHECK(parallel_pool_selftest_for_test(threads) == 0);
    }
    const size_t pool_threads = parallel_pool_threads_for_test();
    CHECK(pool_threads >= 7 && pool_threads <= 255);
    std::atomic<int> failures{ 0 };
    std::vector<std::thread> callers;
    for (int c = 0; c < 8; ++c) {
        callers.emplace_back([&] {
            for (int i = 0; i < 25; ++i) {
                if (parallel_pool_selftest_for_test(8) != 0) {
                    failures.fetch_add(1);
                }
            }
        });
    }
    for (std::thread & caller : callers) {
        caller.join();
    }
    CHECK(failures.load() == 0);
    CHECK(parallel_pool_threads_for_test() == pool_threads);

#ifndef _WIN32
    constexpr int dim = 64;
    constexpr int n = 40000;
    const std::vector<float> rows = tv1x_rows(kTv1xSeedDb, n, dim);
    const std::vector<uint64_t> ids = tv1x_ids(n);
    const std::vector<float> queries = tv1x_rows(kTv1xSeedQuery, 2, dim);
    auto * tv = ggml_vec_index_create_turbovec(dim, 2);
    CHECK(ggml_vec_index_set_threads(tv, 4) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_add(tv, rows.data(), n, ids.data()) == GGML_VEC_INDEX_OK);
    std::vector<float> scores(20);
    std::vector<uint64_t> out(20);
    CHECK(ggml_vec_index_search(tv, queries.data(), 2, 10, scores.data(), out.data()) == GGML_VEC_INDEX_OK);
    std::fflush(nullptr);
    const pid_t child = fork();
    CHECK(child >= 0);
    if (child == 0) {
        alarm(60);
        std::vector<float> child_scores(20);
        std::vector<uint64_t> child_out(20);
        const bool ok = ggml_vec_index_search(tv, queries.data(), 2, 10, child_scores.data(), child_out.data()) ==
                GGML_VEC_INDEX_OK &&
            ggml_vec_index_search(tv, queries.data(), 1, 10, child_scores.data(), child_out.data()) ==
                GGML_VEC_INDEX_OK &&
            parallel_pool_selftest_for_test(4) == 0 &&
            std::equal(child_out.begin(), child_out.begin() + 10, out.begin()) &&
            std::memcmp(child_scores.data(), scores.data(), 10 * sizeof(float)) == 0;
        _exit(ok ? 0 : 1);
    }
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        CHECK(errno == EINTR);
    }
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    ggml_vec_index_free(tv);
#endif
}

// Searches from several threads on one index while another thread adds
// and removes a row under the exclusive lock: every search sees the index
// either with the row or without it, and matches that state's results.
void check_turbovec_1x_concurrent_search() {
    constexpr int dim = 64;
    constexpr int n = 40000;
    constexpr int nq = 4;
    constexpr int k = 10;
    const std::vector<float> rows = tv1x_rows(kTv1xSeedDb, n, dim);
    const std::vector<uint64_t> ids = tv1x_ids(n);
    const std::vector<float> queries = tv1x_rows(kTv1xSeedQuery, nq, dim);
    const std::vector<float> calib = tv1x_rows(kTv1xSeedCalib, 1024, dim);
    // A row on top of query 0, so it is in that query's results while added.
    const std::vector<float> extra(queries.begin(), queries.begin() + dim);
    const uint64_t extra_id = 99999999;
    turbovec_force_planes_for_test(1, 0);
    for (const int staged : { 1, 0 }) {
        auto * tv = ggml_vec_index_create_turbovec(dim, 2);
        CHECK(ggml_vec_index_turbovec_calibrate(tv, calib.data(), 1024) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_add(tv, rows.data(), n, ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_turbovec_set_staged(tv, staged) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_set_threads(tv, 4) == GGML_VEC_INDEX_OK);
        // Per state: nq lone queries, then the batch of all of them.
        auto search_all = [&](std::vector<float> & scores, std::vector<uint64_t> & out) {
            scores.assign(static_cast<size_t>(2 * nq * k), 0.0f);
            out.assign(scores.size(), 0);
            for (int q = 0; q < nq; ++q) {
                if (ggml_vec_index_search(tv, queries.data() + static_cast<size_t>(q) * dim, 1, k,
                        scores.data() + q * k, out.data() + q * k) != GGML_VEC_INDEX_OK) {
                    return false;
                }
            }
            return ggml_vec_index_search(tv, queries.data(), nq, k, scores.data() + nq * k, out.data() + nq * k) ==
                GGML_VEC_INDEX_OK;
        };
        std::vector<float> without_scores;
        std::vector<float> with_scores;
        std::vector<uint64_t> without_ids;
        std::vector<uint64_t> with_ids;
        CHECK(search_all(without_scores, without_ids));
        CHECK(ggml_vec_index_add(tv, extra.data(), 1, &extra_id) == GGML_VEC_INDEX_OK);
        CHECK(search_all(with_scores, with_ids));
        CHECK(with_ids[0] == extra_id);
        CHECK(ggml_vec_index_remove(tv, extra_id) == GGML_VEC_INDEX_OK);
        {
            std::vector<float> again_scores;
            std::vector<uint64_t> again_ids;
            CHECK(search_all(again_scores, again_ids));
            CHECK(again_ids == without_ids && again_scores == without_scores);
        }
        auto matches = [&](const std::vector<float> & scores, const std::vector<uint64_t> & out) {
            // Each query's slice must be one state's; the state may change
            // between the calls.
            for (int part = 0; part < nq + 1; ++part) {
                const size_t begin = static_cast<size_t>(part * k);
                const size_t len = static_cast<size_t>(part == nq ? nq * k : k);
                auto same = [&](const std::vector<float> & s, const std::vector<uint64_t> & o) {
                    return std::equal(out.begin() + begin, out.begin() + begin + len, o.begin() + begin) &&
                        std::memcmp(scores.data() + begin, s.data() + begin, len * sizeof(float)) == 0;
                };
                if (!same(without_scores, without_ids) && !same(with_scores, with_ids)) {
                    return false;
                }
            }
            return true;
        };
        std::atomic<bool> stop{ false };
        std::atomic<int> mismatches{ 0 };
        std::atomic<int> readers_done{ 0 };
        std::vector<std::thread> readers;
        for (int r = 0; r < 4; ++r) {
            readers.emplace_back([&] {
                std::vector<float> scores;
                std::vector<uint64_t> out;
                // A bounded run with a pause after each search: glibc's
                // reader-preferring rwlock would let four back-to-back
                // readers keep the writer out for good.
                for (int it = 0; it < 60 && !stop.load(); ++it) {
                    if (!search_all(scores, out) || !matches(scores, out)) {
                        mismatches.fetch_add(1);
                    }
                    std::this_thread::sleep_for(std::chrono::microseconds(200));
                }
                readers_done.fetch_add(1);
            });
        }
        int toggles = 0;
        while (toggles < 40 && readers_done.load() < 4) {
            if (ggml_vec_index_add(tv, extra.data(), 1, &extra_id) != GGML_VEC_INDEX_OK ||
                ggml_vec_index_remove(tv, extra_id) != GGML_VEC_INDEX_OK ||
                ggml_vec_index_set_threads(tv, toggles % 3 == 0 ? 1 : 4) != GGML_VEC_INDEX_OK) {
                mismatches.fetch_add(1);
            }
            ++toggles;
        }
        stop.store(true);
        for (std::thread & reader : readers) {
            reader.join();
        }
        CHECK(mismatches.load() == 0);
        ggml_vec_index_free(tv);
    }
    turbovec_force_planes_for_test(-1, 0);
}
#endif

#ifdef GGML_VEC_INDEX_TEST_HOOKS
// The staged-search cache follows every mutation: an index searched after
// adds, removals, a calibrate or a reload agrees with one rebuilt from its
// bytes, and the file layout follows the index's size across syncs.
void check_turbovec_1x_staged_maintenance() {
    constexpr int dim = 64;
    constexpr int threshold = 1000;
    const std::vector<float> rows = tv1x_rows(kTv1xSeedDb, 3000, dim);
    const std::vector<uint64_t> ids = tv1x_ids(3000);
    const std::vector<float> queries = tv1x_rows(kTv1xSeedQuery, 4, dim);
    const std::vector<float> calib = tv1x_rows(kTv1xSeedCalib, 1024, dim);
    const std::string path = (std::filesystem::temp_directory_path() / "ggml-vector-index-tv1x-staged-sync.tvim").string();
    turbovec_force_planes_for_test(1, threshold);
    turbovec_force_vector_major_for_test(1);
    auto agrees_with_rebuilt = [&](ggml_vec_index_t * tv) {
        uint8_t * data = nullptr;
        size_t size = 0;
        CHECK(ggml_vec_index_to_bytes(tv, &data, &size) == GGML_VEC_INDEX_OK);
        ggml_vec_index_t * rebuilt = nullptr;
        CHECK(ggml_vec_index_from_bytes(data, size, &rebuilt) == GGML_VEC_INDEX_OK);
        ggml_vec_index_bytes_free(data);
        std::vector<float> s1(4 * 8), s2(4 * 8);
        std::vector<uint64_t> o1(4 * 8), o2(4 * 8);
        CHECK(ggml_vec_index_search(tv, queries.data(), 4, 8, s1.data(), o1.data()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search(rebuilt, queries.data(), 4, 8, s2.data(), o2.data()) == GGML_VEC_INDEX_OK);
        CHECK(o1 == o2 && s1 == s2);
        ggml_vec_index_free(rebuilt);
        return size;
    };
    auto layout_of = [&](ggml_vec_index_t * tv) {
        return tv1x_snapshot_bytes(tv, "ggml-vector-index-tv1x-staged-layout.tvim")[7];
    };
    for (const int bits : { 2, 4 }) {
        std::filesystem::remove(path);
        auto * tv = ggml_vec_index_create_turbovec(dim, bits);
        CHECK(ggml_vec_index_add(tv, rows.data(), 900, ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(layout_of(tv) == 0);
        CHECK(ggml_vec_index_sync(tv, path.c_str()) == GGML_VEC_INDEX_OK);
        // Crossing the threshold promotes the index; the next sync rewrites
        // the file in the planes layout.
        CHECK(ggml_vec_index_add(tv, rows.data() + 900 * dim, 600, ids.data() + 900) == GGML_VEC_INDEX_OK);
        CHECK(layout_of(tv) == 1);
        agrees_with_rebuilt(tv);
        const uint64_t nonce = tv1x_nonce(path);
        CHECK(ggml_vec_index_sync(tv, path.c_str()) == GGML_VEC_INDEX_OK);
        CHECK(tv1x_nonce(path) != nonce);
        CHECK(read_file_bytes(path)[7] == 1);
        // Adds and removals patch the sign planes in place.
        CHECK(ggml_vec_index_add(tv, rows.data() + 1500 * dim, 77, ids.data() + 1500) == GGML_VEC_INDEX_OK);
        for (size_t i = 0; i < 1577; i += 41) {
            CHECK(ggml_vec_index_remove(tv, ids[i]) == GGML_VEC_INDEX_OK);
        }
        agrees_with_rebuilt(tv);
        // A later sync stays incremental in the planes layout and reloads.
        const uint64_t planes_nonce = tv1x_nonce(path);
        CHECK(ggml_vec_index_sync(tv, path.c_str()) == GGML_VEC_INDEX_OK);
        CHECK(tv1x_nonce(path) == planes_nonce);
        auto * loaded = ggml_vec_index_load(path.c_str());
        CHECK(loaded != nullptr);
        CHECK(tv1x_snapshot_bytes(loaded, "ggml-vector-index-tv1x-staged-l.tvim") ==
              tv1x_snapshot_bytes(tv, "ggml-vector-index-tv1x-staged-t.tvim"));
        ggml_vec_index_free(loaded);
        // A calibrate re-encodes every row and rebuilds the cache.
        CHECK(ggml_vec_index_turbovec_calibrate(tv, calib.data(), 1024) == GGML_VEC_INDEX_OK);
        agrees_with_rebuilt(tv);
        // Shrinking below the threshold keeps the staged search but stores
        // sequential units, as upstream does.
        for (size_t i = 1; i < 1577 && ggml_vec_index_len(tv) > 800; i += 2) {
            ggml_vec_index_remove(tv, ids[i]);
        }
        CHECK(ggml_vec_index_len(tv) <= 800);
        CHECK(layout_of(tv) == 0);
        std::vector<float> scores(8);
        std::vector<uint64_t> out(8);
        CHECK(ggml_vec_index_search(tv, queries.data(), 1, 8, scores.data(), out.data()) == GGML_VEC_INDEX_OK);
        ggml_vec_index_free(tv);
    }
    std::filesystem::remove(path);
    turbovec_force_vector_major_for_test(-1);
    turbovec_force_planes_for_test(-1, 0);
}
#endif

#ifdef GGML_VEC_INDEX_TEST_HOOKS
// A 1.x index keeps its codes once, in the blocked search cache, through
// every mutation, snapshot and reload (upstream #475). Only the planes
// layout, whose staged search reads bit planes row by row, adds packed rows.
void check_turbovec_1x_single_code_copy() {
    constexpr int dim = 64;
    constexpr int n = 1000;
    const std::vector<float> rows = tv1x_rows(kTv1xSeedDb, n, dim);
    const std::vector<uint64_t> ids = tv1x_ids(n);
    const std::vector<float> queries = tv1x_rows(kTv1xSeedQuery, 4, dim);
    const std::vector<float> calib = tv1x_rows(kTv1xSeedCalib, 256, dim);
    const std::string path = (std::filesystem::temp_directory_path() / "ggml-vector-index-tv1x-one-copy.tvim").string();
    auto codes_once = [&](const ggml_vec_index_t * tv) {
        size_t packed = 0;
        size_t blocked = 0;
        size_t sign = 0;
        turbovec_code_bytes_for_test(tv, &packed, &blocked, &sign);
        const int bits = ggml_vec_index_bit_width(tv);
        const size_t n_byte_groups = static_cast<size_t>(dim) / static_cast<size_t>(8 / bits);
        const size_t n_blocks = (static_cast<size_t>(ggml_vec_index_len(tv)) + 31) / 32;
        return packed == 0 && sign == 0 && blocked == n_blocks * n_byte_groups * 32;
    };
    auto same_results = [&](ggml_vec_index_t * a, ggml_vec_index_t * b) {
        std::vector<float> s1(4 * 10), s2(4 * 10);
        std::vector<uint64_t> o1(4 * 10), o2(4 * 10);
        CHECK(ggml_vec_index_search(a, queries.data(), 4, 10, s1.data(), o1.data()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search(b, queries.data(), 4, 10, s2.data(), o2.data()) == GGML_VEC_INDEX_OK);
        return o1 == o2 && s1 == s2;
    };
    // The per-row scorer below is the LUT kernel; pin the blocked one to it.
    turbovec_force_vector_major_for_test(0);
    turbovec_force_planes_for_test(0, 0);
    for (const int bits : { 2, 3, 4 }) {
        std::filesystem::remove(path);
        auto * tv = ggml_vec_index_create_turbovec(dim, bits);
        CHECK(ggml_vec_index_add(tv, rows.data(), 700, ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(codes_once(tv));
        CHECK(ggml_vec_index_sync(tv, path.c_str()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_add(tv, rows.data() + 700 * dim, n - 700, ids.data() + 700) == GGML_VEC_INDEX_OK);
        for (int i = 0; i < n; i += 37) {
            CHECK(ggml_vec_index_remove(tv, ids[static_cast<size_t>(i)]) == GGML_VEC_INDEX_OK);
        }
        CHECK(codes_once(tv));
        CHECK(ggml_vec_index_sync(tv, path.c_str()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_compact(tv) == GGML_VEC_INDEX_OK);
        CHECK(codes_once(tv));

        // A reload, a bytes round trip and a calibrate all end with one copy,
        // and the rows they hold match the index's own.
        auto * loaded = ggml_vec_index_load(path.c_str());
        CHECK(loaded != nullptr);
        CHECK(codes_once(loaded));
        CHECK(same_results(tv, loaded));
        CHECK(tv1x_snapshot_bytes(loaded, "ggml-vector-index-tv1x-one-copy-l.tvim") ==
              tv1x_snapshot_bytes(tv, "ggml-vector-index-tv1x-one-copy-t.tvim"));
        uint8_t * data = nullptr;
        size_t size = 0;
        CHECK(ggml_vec_index_to_bytes(tv, &data, &size) == GGML_VEC_INDEX_OK);
        ggml_vec_index_t * rebuilt = nullptr;
        CHECK(ggml_vec_index_from_bytes(data, size, &rebuilt) == GGML_VEC_INDEX_OK);
        ggml_vec_index_bytes_free(data);
        CHECK(codes_once(rebuilt));
        CHECK(ggml_vec_index_turbovec_calibrate(tv, calib.data(), 256) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_turbovec_calibrate(rebuilt, calib.data(), 256) == GGML_VEC_INDEX_OK);
        CHECK(codes_once(tv));
        CHECK(same_results(tv, rebuilt));
        uint64_t h1[4] = {};
        uint64_t h2[4] = {};
        CHECK(turbovec_index_hashes_for_test(tv, &h1[0], &h1[1], &h1[2], &h1[3]) == 1);
        CHECK(turbovec_index_hashes_for_test(rebuilt, &h2[0], &h2[1], &h2[2], &h2[3]) == 1);
        CHECK(std::equal(h1, h1 + 4, h2));
        ggml_vec_index_free(rebuilt);

        // Per-row scoring reads the rows back from the blocked cache.
        turbovec_clear_blocked_for_test(loaded);
        size_t packed = 0;
        size_t blocked = 0;
        size_t sign = 0;
        turbovec_code_bytes_for_test(loaded, &packed, &blocked, &sign);
        CHECK(packed > 0 && blocked == 0);
        auto * reloaded = ggml_vec_index_load(path.c_str());
        const int live = ggml_vec_index_len(reloaded);
        std::vector<float> s1(static_cast<size_t>(live)), s2(static_cast<size_t>(live));
        std::vector<uint64_t> o1(static_cast<size_t>(live)), o2(static_cast<size_t>(live));
        CHECK(ggml_vec_index_search(loaded, queries.data(), 1, live, s1.data(), o1.data()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search(reloaded, queries.data(), 1, live, s2.data(), o2.data()) == GGML_VEC_INDEX_OK);
        std::unordered_map<uint64_t, float> by_id;
        for (int i = 0; i < live; ++i) {
            by_id[o2[static_cast<size_t>(i)]] = s2[static_cast<size_t>(i)];
        }
        for (int i = 0; i < live; ++i) {
            const auto it = by_id.find(o1[static_cast<size_t>(i)]);
            CHECK(it != by_id.end());
            CHECK(std::fabs(it->second - s1[static_cast<size_t>(i)]) <= std::max(1e-4f * std::fabs(it->second), 1e-4f));
        }
        ggml_vec_index_free(reloaded);
        ggml_vec_index_free(loaded);
        ggml_vec_index_free(tv);
    }

    // Promotion to the planes layout materializes the packed rows next to
    // the sign planes; a calibrate below the threshold drops them again.
    turbovec_force_planes_for_test(1, 500);
    for (const int bits : { 2, 4 }) {
        auto * tv = ggml_vec_index_create_turbovec(dim, bits);
        CHECK(ggml_vec_index_add(tv, rows.data(), 400, ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(codes_once(tv));
        CHECK(ggml_vec_index_add(tv, rows.data() + 400 * dim, 200, ids.data() + 400) == GGML_VEC_INDEX_OK);
        size_t packed = 0;
        size_t blocked = 0;
        size_t sign = 0;
        turbovec_code_bytes_for_test(tv, &packed, &blocked, &sign);
        CHECK(sign > 0 && packed >= static_cast<size_t>(600) * static_cast<size_t>(dim * bits / 8));
        for (int i = 0; i < 200; ++i) {
            CHECK(ggml_vec_index_remove(tv, ids[static_cast<size_t>(i)]) == GGML_VEC_INDEX_OK);
        }
        CHECK(ggml_vec_index_turbovec_calibrate(tv, calib.data(), 256) == GGML_VEC_INDEX_OK);
        CHECK(codes_once(tv));
        ggml_vec_index_free(tv);
    }
    turbovec_force_planes_for_test(-1, 0);
    turbovec_force_vector_major_for_test(-1);
    std::filesystem::remove(path);
}
#endif

#ifdef GGML_VEC_INDEX_TEST_HOOKS
// Equal scores come back in row order, as upstream turbovec returns them:
// insertion order, as reshuffled by swap-remove. Other indexes order ties
// by ascending id.
void check_turbovec_1x_tie_order() {
    constexpr int dim = 64;
    const std::vector<float> row = tv1x_rows(kTv1xSeedDb, 1, dim);
    std::vector<float> rows;
    for (int i = 0; i < 4; ++i) {
        rows.insert(rows.end(), row.begin(), row.end());
    }
    const uint64_t ids[4] = { 100, 99, 7, 50 };
    auto top = [&](ggml_vec_index_t * idx, int k) {
        std::vector<float> scores(static_cast<size_t>(k));
        std::vector<uint64_t> out(static_cast<size_t>(k));
        CHECK(ggml_vec_index_search(idx, row.data(), 1, k, scores.data(), out.data()) == GGML_VEC_INDEX_OK);
        return out;
    };
    for (const int bits : { 2, 3, 4 }) {
        auto * tv = ggml_vec_index_create_turbovec(dim, bits);
        CHECK(ggml_vec_index_add(tv, rows.data(), 4, ids) == GGML_VEC_INDEX_OK);
        CHECK((top(tv, 4) == std::vector<uint64_t>{ 100, 99, 7, 50 }));
        CHECK((top(tv, 2) == std::vector<uint64_t>{ 100, 99 }));
        // Removing row 0 moves the last row (id 50) into it.
        CHECK(ggml_vec_index_remove(tv, 100) == GGML_VEC_INDEX_OK);
        CHECK((top(tv, 3) == std::vector<uint64_t>{ 50, 99, 7 }));
        ggml_vec_index_set_threads(tv, 4);
        CHECK((top(tv, 3) == std::vector<uint64_t>{ 50, 99, 7 }));
        ggml_vec_index_free(tv);
    }
    auto * f32 = ggml_vec_index_create(dim, 32);
    CHECK(ggml_vec_index_add(f32, rows.data(), 4, ids) == GGML_VEC_INDEX_OK);
    CHECK((top(f32, 4) == std::vector<uint64_t>{ 7, 50, 99, 100 }));
    ggml_vec_index_free(f32);
}

// The stale-temp sweep reclaims only temps written for this destination in
// the `.tmp.pid.seq.hex8` shape this port and upstream turbovec write (or
// upstream's older `.tmp.pid.seq`), and leaves every other `.tmp.` file
// alone, a user's `<name>.tmp.backup` included.
void check_turbovec_stale_temp_sweep() {
    const std::filesystem::path dir = verbatim_path(std::filesystem::temp_directory_path()) / "ggml-vector-index-sweep";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const auto old = std::filesystem::file_time_type::clock::now() - std::chrono::hours(2);
    auto make = [&](const std::string & name, bool stale) {
        const std::filesystem::path path = dir / name;
        std::ofstream(path) << "x";
        if (stale) {
            std::filesystem::last_write_time(path, old);
        }
        return path;
    };
    const std::string long_base = std::string(240, 'a') + ".tvim";
    const std::string kept_long_base = std::string(240, 'b') + ".tvim";
    const std::string suffix = ".tmp.123.4.deadbeef";
    const size_t cut = 255 - suffix.size();
    const std::filesystem::path reclaimed[] = {
        make("idx.tv.tmp.123.4", true),
        make("idx.tv" + suffix, true),
        // A long destination's temp carries the writer's exact cut.
        make(long_base.substr(0, cut) + suffix, true),
    };
    const std::filesystem::path kept[] = {
        make("idx.tv.tmp.backup", true),
        make("idx.tv.tmp.Ab3dE9", true),
        make("idx.tmp.backup", true),
        make("idx.tv.tmp.notours", true),
        make("idx.tv.tmp.12.x", true),
        make("idx.tv.tmp.12.34.56", true),
        make("idx.tv.tmp.123.4.deadbee", true),
        make("idx.tv.tmp.99.7.0badf00d", false),
        make("other.tv" + suffix, true),
        // A cut stem that names an existing file may be that file's temp.
        make(kept_long_base.substr(0, cut) + suffix, true),
        make(kept_long_base.substr(0, cut), false),
    };
    constexpr int dim = 64;
    const std::vector<float> rows = tv1x_rows(kTv1xSeedDb, 4, dim);
    const std::vector<uint64_t> ids = tv1x_ids(4);
    auto * tv = ggml_vec_index_create_turbovec(dim, 2);
    CHECK(ggml_vec_index_add(tv, rows.data(), 4, ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(tv, (dir / "idx.tv").string().c_str()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(tv, (dir / long_base).string().c_str()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(tv, (dir / kept_long_base).string().c_str()) == GGML_VEC_INDEX_OK);
    ggml_vec_index_free(tv);
    for (const auto & path : reclaimed) {
        CHECK(!std::filesystem::exists(path));
    }
    for (const auto & path : kept) {
        CHECK(std::filesystem::exists(path));
    }
    std::filesystem::remove_all(dir);
}

// Every allocation an add or a calibrate makes can fail without leaving
// the index half changed: a failed call leaves the snapshot bytes as they
// were, a successful one matches the same call made with memory to spare,
// and the staged-search caches stay consistent with the rows either way.
void check_turbovec_1x_alloc_failures() {
    constexpr int dim = 64;
    const std::vector<float> rows = tv1x_rows(kTv1xSeedDb, 140, dim);
    const std::vector<uint64_t> ids = tv1x_ids(140);
    const std::vector<float> queries = tv1x_rows(kTv1xSeedQuery, 3, dim);
    const std::vector<float> calib = tv1x_rows(kTv1xSeedCalib, 64, dim);
    turbovec_force_vector_major_for_test(0);
    turbovec_force_planes_for_test(1, 96);
    auto bytes_of = [](ggml_vec_index_t * idx) {
        uint8_t * data = nullptr;
        size_t size = 0;
        CHECK(ggml_vec_index_to_bytes(idx, &data, &size) == GGML_VEC_INDEX_OK);
        std::vector<uint8_t> out(data, data + size);
        ggml_vec_index_bytes_free(data);
        return out;
    };
    auto from = [](const std::vector<uint8_t> & bytes) {
        ggml_vec_index_t * idx = nullptr;
        CHECK(ggml_vec_index_from_bytes(bytes.data(), bytes.size(), &idx) == GGML_VEC_INDEX_OK);
        return idx;
    };
    auto results = [&](ggml_vec_index_t * idx) {
        std::vector<float> scores(3 * 10);
        std::vector<uint64_t> out(3 * 10);
        CHECK(ggml_vec_index_search(idx, queries.data(), 3, 10, scores.data(), out.data()) == GGML_VEC_INDEX_OK);
        return std::make_pair(out, scores);
    };
    // Promoted exactly when it holds the sign planes, with packed rows next
    // to them; otherwise one blocked copy of the codes.
    auto consistent = [&](ggml_vec_index_t * idx, int bits) {
        size_t packed = 0;
        size_t blocked = 0;
        size_t sign = 0;
        turbovec_code_bytes_for_test(idx, &packed, &blocked, &sign);
        const size_t n = static_cast<size_t>(ggml_vec_index_len(idx));
        const size_t n_blocks = (n + 31) / 32;
        const bool blocked_ok = blocked == n_blocks * 32 * static_cast<size_t>(dim / (8 / bits));
        if (sign == 0) {
            return blocked_ok && packed == 0;
        }
        return blocked_ok && sign == n_blocks * 32 * static_cast<size_t>(dim / 8) &&
            packed >= n * static_cast<size_t>(dim * bits / 8);
    };
    // The reference searches the same rows promoted the same way.
    auto same_search = [&](ggml_vec_index_t * idx, const std::vector<uint8_t> & bytes) {
        size_t packed = 0;
        size_t blocked = 0;
        size_t sign = 0;
        turbovec_code_bytes_for_test(idx, &packed, &blocked, &sign);
        auto * ref = from(bytes);
        if (sign == 0) {
            ggml_vec_index_turbovec_set_staged(ref, 0);
        }
        const bool same = results(idx) == results(ref);
        ggml_vec_index_free(ref);
        return same;
    };
    for (const int bits : { 2, 4 }) {
        for (const int op : { 0, 1 }) {
            // op 0: an add that promotes the index; op 1: a calibrate of a
            // promoted index.
            auto * seed = ggml_vec_index_create_turbovec(dim, bits);
            CHECK(ggml_vec_index_add(seed, rows.data(), op == 0 ? 95 : 130, ids.data()) == GGML_VEC_INDEX_OK);
            const std::vector<uint8_t> before = bytes_of(seed);
            auto run = [&](ggml_vec_index_t * idx) {
                return op == 0 ? ggml_vec_index_add(idx, rows.data() + 95 * dim, 10, ids.data() + 95) :
                                 ggml_vec_index_turbovec_calibrate(idx, calib.data(), 64);
            };
            CHECK(run(seed) == GGML_VEC_INDEX_OK);
            const std::vector<uint8_t> after = bytes_of(seed);
            ggml_vec_index_free(seed);
            int failures = 0;
            for (long countdown = 1; countdown < 20000; ++countdown) {
                auto * idx = from(before);
                g_alloc_fail_countdown.store(countdown);
                const int rc = run(idx);
                const bool injected = g_alloc_fail_countdown.exchange(0) <= 0;
                if (rc == GGML_VEC_INDEX_OK) {
                    CHECK(bytes_of(idx) == after);
                    CHECK(consistent(idx, bits));
                    CHECK(same_search(idx, after));
                } else {
                    CHECK(rc == GGML_VEC_INDEX_E_OOM);
                    ++failures;
                    CHECK(bytes_of(idx) == before);
                    CHECK(consistent(idx, bits));
                    CHECK(same_search(idx, before));
                    CHECK(run(idx) == GGML_VEC_INDEX_OK);
                    CHECK(bytes_of(idx) == after);
                    CHECK(consistent(idx, bits));
                }
                ggml_vec_index_free(idx);
                if (!injected) {
                    break;
                }
            }
            CHECK(failures > 0);
        }
    }
    turbovec_force_planes_for_test(-1, 0);
    turbovec_force_vector_major_for_test(-1);
}

// A search cache that was dropped is rebuilt in full by the next add, and
// a staged search never reads it; set_staged(0) drops the planes caches.
void check_turbovec_1x_cache_recovery() {
    constexpr int dim = 64;
    const std::vector<float> rows = tv1x_rows(kTv1xSeedDb, 200, dim);
    const std::vector<uint64_t> ids = tv1x_ids(200);
    const std::vector<float> queries = tv1x_rows(kTv1xSeedQuery, 3, dim);
    turbovec_force_vector_major_for_test(0);
    auto results = [&](ggml_vec_index_t * idx) {
        std::vector<float> scores(3 * 10);
        std::vector<uint64_t> out(3 * 10);
        CHECK(ggml_vec_index_search(idx, queries.data(), 3, 10, scores.data(), out.data()) == GGML_VEC_INDEX_OK);
        return std::make_pair(out, scores);
    };
    for (const int bits : { 2, 3, 4 }) {
        auto * hooked = ggml_vec_index_create_turbovec(dim, bits);
        auto * plain = ggml_vec_index_create_turbovec(dim, bits);
        for (auto * idx : { hooked, plain }) {
            CHECK(ggml_vec_index_add(idx, rows.data(), 100, ids.data()) == GGML_VEC_INDEX_OK);
        }
        turbovec_clear_blocked_for_test(hooked);
        for (auto * idx : { hooked, plain }) {
            CHECK(ggml_vec_index_remove(idx, ids[0]) == GGML_VEC_INDEX_OK);
            CHECK(ggml_vec_index_add(idx, rows.data() + 100 * dim, 3, ids.data() + 100) == GGML_VEC_INDEX_OK);
        }
        CHECK(turbovec_blocked_hash_for_test(hooked) == turbovec_blocked_hash_for_test(plain));
        CHECK(results(hooked) == results(plain));
        size_t packed = 0;
        size_t blocked = 0;
        size_t sign = 0;
        turbovec_code_bytes_for_test(hooked, &packed, &blocked, &sign);
        CHECK(packed == 0);
        ggml_vec_index_free(hooked);
        ggml_vec_index_free(plain);
    }

    turbovec_force_planes_for_test(1, 96);
    for (const int bits : { 2, 4 }) {
        auto * tv = ggml_vec_index_create_turbovec(dim, bits);
        auto * full = ggml_vec_index_create_turbovec(dim, bits);
        CHECK(ggml_vec_index_turbovec_set_staged(full, 0) == GGML_VEC_INDEX_OK);
        for (auto * idx : { tv, full }) {
            CHECK(ggml_vec_index_add(idx, rows.data(), 150, ids.data()) == GGML_VEC_INDEX_OK);
        }
        size_t packed = 0;
        size_t blocked = 0;
        size_t sign = 0;
        turbovec_code_bytes_for_test(full, &packed, &blocked, &sign);
        CHECK(packed == 0 && sign == 0);
        turbovec_code_bytes_for_test(tv, &packed, &blocked, &sign);
        CHECK(packed > 0 && sign > 0);
        const std::vector<uint8_t> snapshot = tv1x_snapshot_bytes(tv, "ggml-vector-index-tv1x-staged-off.tvim");
        CHECK(snapshot == tv1x_snapshot_bytes(full, "ggml-vector-index-tv1x-staged-full.tvim"));
        CHECK(ggml_vec_index_turbovec_set_staged(tv, 0) == GGML_VEC_INDEX_OK);
        turbovec_code_bytes_for_test(tv, &packed, &blocked, &sign);
        CHECK(packed == 0 && sign == 0);
        CHECK(results(tv) == results(full));
        CHECK(ggml_vec_index_turbovec_set_staged(tv, 1) == GGML_VEC_INDEX_OK);
        turbovec_code_bytes_for_test(tv, &packed, &blocked, &sign);
        CHECK(packed > 0 && sign > 0);
        CHECK(tv1x_snapshot_bytes(tv, "ggml-vector-index-tv1x-staged-on.tvim") == snapshot);
        // Without its blocked cache a promoted index answers from the full
        // scan's per-row path instead of staged rescoring.
        turbovec_clear_blocked_for_test(tv);
        const auto staged_off = results(full);
        const auto recovered = results(tv);
        CHECK(recovered.first == staged_off.first);
        ggml_vec_index_free(tv);
        ggml_vec_index_free(full);
    }
    turbovec_force_planes_for_test(-1, 0);
    turbovec_force_vector_major_for_test(-1);
}

// A fast write keeps the temp file and atomic rename but skips both fsyncs
// (upstream Durability::Fast), for every index kind, and writes the same
// bytes as a durable one. It does not let a delta log start.
void check_write_with_durability() {
    constexpr int dim = 64;
    const std::vector<float> rows = tv1x_rows(kTv1xSeedDb, 100, dim);
    const std::vector<uint64_t> ids = tv1x_ids(101);
    const std::string durable_path = (std::filesystem::temp_directory_path() / "ggml-vector-index-durable.tvim").string();
    const std::string fast_path = (std::filesystem::temp_directory_path() / "ggml-vector-index-fast.tvim").string();
    const std::string delta_path = (std::filesystem::temp_directory_path() / "ggml-vector-index-fast.tvid").string();
    for (const int kind : { 0, 1 }) {
        auto * idx = kind == 0 ? ggml_vec_index_create(dim, 32) : ggml_vec_index_create_turbovec(dim, 4);
        CHECK(ggml_vec_index_add(idx, rows.data(), 100, ids.data()) == GGML_VEC_INDEX_OK);
        std::filesystem::remove(durable_path);
        std::filesystem::remove(fast_path);
        std::filesystem::remove(delta_path);
        CHECK(ggml_vec_index_write_with_durability(idx, fast_path.c_str(), static_cast<ggml_vec_index_durability>(2)) ==
              GGML_VEC_INDEX_E_INVALID_ARG);

        ggml_vec_index_test_set_data_fsync_fail(1);
        CHECK(ggml_vec_index_write(idx, durable_path.c_str()) == GGML_VEC_INDEX_E_IO);
        CHECK(!std::filesystem::exists(durable_path));
        CHECK(ggml_vec_index_write_with_durability(idx, fast_path.c_str(), GGML_VEC_INDEX_DURABILITY_FAST) ==
              GGML_VEC_INDEX_OK);
        ggml_vec_index_test_set_data_fsync_fail(0);
        ggml_vec_index_test_set_parent_fsync_fail(1);
        CHECK(ggml_vec_index_write_with_durability(idx, fast_path.c_str(), GGML_VEC_INDEX_DURABILITY_FAST) ==
              GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_write(idx, durable_path.c_str()) == GGML_VEC_INDEX_E_NOT_DURABLE);
        ggml_vec_index_test_set_parent_fsync_fail(0);
        CHECK(read_file_bytes(fast_path) == read_file_bytes(durable_path));
        auto * loaded = ggml_vec_index_load(fast_path.c_str());
        CHECK(loaded != nullptr && ggml_vec_index_len(loaded) == 100);
        ggml_vec_index_free(loaded);
        ggml_vec_index_free(idx);

        if (kind == 0) {
            // Only a durable write lets a delta log start from this state.
            auto * fresh = ggml_vec_index_create(dim, 32);
            CHECK(ggml_vec_index_add(fresh, rows.data(), 100, ids.data()) == GGML_VEC_INDEX_OK);
            CHECK(ggml_vec_index_write_with_durability(fresh, fast_path.c_str(), GGML_VEC_INDEX_DURABILITY_FAST) ==
                  GGML_VEC_INDEX_OK);
            CHECK(ggml_vec_index_add_logged(fresh, rows.data(), 1, &ids[100], delta_path.c_str()) ==
                  GGML_VEC_INDEX_E_INVALID_ARG);
            CHECK(ggml_vec_index_write_with_durability(fresh, fast_path.c_str(), GGML_VEC_INDEX_DURABILITY_DURABLE) ==
                  GGML_VEC_INDEX_OK);
            CHECK(ggml_vec_index_add_logged(fresh, rows.data(), 1, &ids[100], delta_path.c_str()) == GGML_VEC_INDEX_OK);
            ggml_vec_index_free(fresh);
        }
    }
    std::filesystem::remove(durable_path);
    std::filesystem::remove(fast_path);
    std::filesystem::remove(delta_path);
}

// to_bytes, write and a full sync build block units as they go instead of
// materializing all of them, so none holds a second copy of the payload.
void check_turbovec_1x_streamed_snapshots() {
    constexpr int dim = 256;
    constexpr int n = 40000;
    const std::vector<float> rows = tv1x_rows(kTv1xSeedDb, n, dim);
    const std::vector<uint64_t> ids = tv1x_ids(n);
    const std::string path = (std::filesystem::temp_directory_path() / "ggml-vector-index-tv1x-stream.tvim").string();
    const std::string sync_path =
        (std::filesystem::temp_directory_path() / "ggml-vector-index-tv1x-stream-sync.tvim").string();
    auto * tv = ggml_vec_index_create_turbovec(dim, 4);
    CHECK(ggml_vec_index_add(tv, rows.data(), n, ids.data()) == GGML_VEC_INDEX_OK);
    const size_t payload = ggml_vec_index_serialized_size(tv);
    CHECK(payload > static_cast<size_t>(n) * (dim / 2));
    auto largest_during = [&](auto && fn) {
        g_largest_alloc.store(0);
        g_track_largest_alloc.store(true);
        fn();
        g_track_largest_alloc.store(false);
        return g_largest_alloc.load();
    };
    uint8_t * data = nullptr;
    size_t size = 0;
    const size_t to_bytes_peak = largest_during([&] {
        CHECK(ggml_vec_index_to_bytes(tv, &data, &size) == GGML_VEC_INDEX_OK);
    });
    CHECK(size == payload);
    std::filesystem::remove(path);
    std::filesystem::remove(sync_path);
    const size_t write_peak = largest_during([&] { CHECK(ggml_vec_index_write(tv, path.c_str()) == GGML_VEC_INDEX_OK); });
    const size_t sync_peak = largest_during([&] { CHECK(ggml_vec_index_sync(tv, sync_path.c_str()) == GGML_VEC_INDEX_OK); });
    CHECK(to_bytes_peak <= (size_t{ 1 } << 20));
    CHECK(write_peak <= (size_t{ 1 } << 20) + payload / 64);
    CHECK(sync_peak <= (size_t{ 1 } << 20) + payload / 64);
    const std::vector<uint8_t> written = read_file_bytes(path);
    CHECK(written.size() == size && std::memcmp(written.data(), data, size) == 0);
    CHECK(read_file_bytes(sync_path).size() == size);
    ggml_vec_index_bytes_free(data);
    auto * loaded = ggml_vec_index_load(sync_path.c_str());
    CHECK(loaded != nullptr && ggml_vec_index_len(loaded) == n);
    ggml_vec_index_free(loaded);
    ggml_vec_index_free(tv);
    std::filesystem::remove(path);
    std::filesystem::remove(sync_path);
}
#endif

// A v8 file is read only as far as its geometry and a CRC-checked header
// declare, so a sparse file with a huge apparent size costs no memory,
// whether its metadata is valid or not.
void check_turbovec_1x_sparse_load() {
    constexpr int dim = 32;
    const std::vector<float> rows = tv1x_rows(kTv1xSeedDb, 64, dim);
    const std::vector<uint64_t> ids = tv1x_ids(64);
    const std::string path = (std::filesystem::temp_directory_path() / "ggml-vector-index-tv1x-sparse.tvim").string();
    std::filesystem::remove(path);
    auto * tv = ggml_vec_index_create_turbovec(dim, 2);
    CHECK(ggml_vec_index_add(tv, rows.data(), 64, ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_write(tv, path.c_str()) == GGML_VEC_INDEX_OK);
    ggml_vec_index_free(tv);
    const std::vector<uint8_t> image = read_file_bytes(path);
    auto write_sparse = [&](const std::vector<uint8_t> & bytes) {
        write_file_bytes(path, bytes);
        std::error_code ec;
        std::filesystem::resize_file(path, uintmax_t{ 1 } << 40, ec);
        return !ec;
    };
    if (!write_sparse(image)) {
        std::printf("SKIP TurboVec sparse load: no sparse files here\n");
        std::filesystem::remove(path);
        return;
    }
    auto * loaded = ggml_vec_index_load(path.c_str());
    CHECK(loaded != nullptr);
    CHECK(ggml_vec_index_len(loaded) == 64);
    ggml_vec_index_free(loaded);

    // An invalid dimension.
    std::vector<uint8_t> bad = image;
    put_u32_le(bad, 8, 7);
    CHECK(write_sparse(bad));
    ggml_vec_index_t * out = nullptr;
    CHECK(ggml_vec_index_load_ex(path.c_str(), &out) == GGML_VEC_INDEX_E_IO);
    CHECK(out == nullptr);
    // A huge row count in the only header, whose CRC then fails.
    bad = image;
    const size_t sb_len = 24 + 3 * 4 + 4 * 4 + 4 + 4;
    write_u64_le_at(bad, sb_len + 8, uint64_t{ 1 } << 34);
    CHECK(write_sparse(bad));
    CHECK(ggml_vec_index_load_ex(path.c_str(), &out) == GGML_VEC_INDEX_E_IO);
    CHECK(out == nullptr);
    std::filesystem::remove(path);
}

// A stale handle whose next sync would switch the file to the planes
// layout still checks the file's commits first, so it cannot rewrite over
// another writer's commit.
void check_turbovec_1x_stale_layout_change() {
    constexpr int dim = 32;
    // Upstream switches to the planes layout at 32768 rows.
    constexpr size_t n = 32768;
    const std::vector<float> rows = tv1x_rows(kTv1xSeedDb, n, dim);
    const std::vector<uint64_t> ids = tv1x_ids(n);
    const std::string path = (std::filesystem::temp_directory_path() / "ggml-vector-index-tv1x-layout.tvim").string();
    std::filesystem::remove(path);
    auto * tv = ggml_vec_index_create_turbovec(dim, 2);
    CHECK(ggml_vec_index_add(tv, rows.data(), static_cast<int>(n - 1), ids.data()) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_sync(tv, path.c_str()) == GGML_VEC_INDEX_OK);
    auto * other = ggml_vec_index_load(path.c_str());
    CHECK(other != nullptr);
    CHECK(ggml_vec_index_remove(other, ids[0]) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_sync(other, path.c_str()) == GGML_VEC_INDEX_OK);
    ggml_vec_index_free(other);
    CHECK(ggml_vec_index_add(tv, rows.data() + (n - 1) * dim, 1, ids.data() + n - 1) == GGML_VEC_INDEX_OK);
    CHECK(ggml_vec_index_sync(tv, path.c_str()) == GGML_VEC_INDEX_E_IO);
    auto * reloaded = ggml_vec_index_load(path.c_str());
    CHECK(reloaded != nullptr);
    CHECK(ggml_vec_index_len(reloaded) == static_cast<int>(n - 2));
    CHECK(ggml_vec_index_contains(reloaded, ids[0]) == 0);
    ggml_vec_index_free(reloaded);
    ggml_vec_index_free(tv);
    std::filesystem::remove(path);
}

void check_turbovec_1x() {
    for (const Tv1xIndexCase & c : kTv1xIndexCases) {
#ifdef GGML_VEC_INDEX_TEST_HOOKS
        // Each 4-bit kernel against its own fixture, on indexes built under it.
        if (c.bits == 4) {
            for (const int kernel : { 0, 1 }) {
                turbovec_force_vector_major_for_test(kernel);
                check_turbovec_1x_golden_case(c, kernel);
            }
            turbovec_force_vector_major_for_test(-1);
            continue;
        }
#endif
        check_turbovec_1x_golden_case(c, -1);
    }
    check_turbovec_1x_snapshot_interop();
    check_turbovec_1x_calibrate_api();
    check_turbovec_1x_dims();
    for (const Tv1xSyncCase & c : kTv1xSyncCases) {
        check_turbovec_1x_sync_case(c);
    }
    check_turbovec_1x_sync_api();
    check_turbovec_1x_sparse_load();
    check_turbovec_1x_stale_layout_change();
    check_turbovec_1x_history_snapshots();
    check_turbovec_1x_threads();
    check_vector_index_introspection();
    check_turbovec_1x_upstream_contracts();
    check_turbovec_1x_sync_contracts();
}

}  // namespace


int main(int argc, char ** argv) {
    const std::filesystem::path test_temp_dir = make_test_temp_dir();
    set_test_temp_dir(test_temp_dir);

#ifdef GGML_VEC_INDEX_TEST_HOOKS
    if (argc == 2 && std::string(argv[1]) == "--internal-hooks") {
        // Kernel-consistency checks below compare against the scalar LUT.
        turbovec_force_vector_major_for_test(0);
        if (kTurboVecSupported) {
            {
                constexpr int tv_cache_dim = 64;
                CHECK(turbovec_rotation_cache_bytes_for_test() == 0);
                auto * tv_cache = create_legacy_turbovec(tv_cache_dim, 2);
                ggml_vec_index_prepare(tv_cache);
                CHECK(turbovec_rotation_cache_bytes_for_test() >=
                      static_cast<size_t>(tv_cache_dim) * static_cast<size_t>(tv_cache_dim) *
                          sizeof(float));
                ggml_vec_index_free(tv_cache);
                CHECK(turbovec_rotation_cache_bytes_for_test() == 0);
            }

            for (const int bit_width : { 2, 4 }) {
                check_turbovec_blocked_scalar_scores(bit_width, 128, 17, 4);
                check_turbovec_blocked_scalar_scores(bit_width, 128, 33, 5);
                check_turbovec_blocked_scalar_scores(bit_width, 256, 65, 17, 8);
                check_turbovec_fused_batch_scores(bit_width);
                check_turbovec_oversized_snapshot_compatibility(bit_width);
            }
            check_turbovec_blocked_scalar_scores(2, 128, 1000, 2);
            check_turbovec_blocked_scalar_scores(4, 128, 1000, 2);
            for (const int bit_width : { 2, 4 }) {
                for (const int dim : { 128, 256 }) {
                    const int avx2_status =
                        turbovec_avx2_lut_block_matches_scalar_for_test(bit_width, dim);
                    if (avx2_status < 0) {
                        std::printf("SKIP TurboVec AVX2-LUT parity bits=%d dim=%d\n", bit_width, dim);
                    } else {
                        CHECK(avx2_status == 1);
                    }
                }
                check_turbovec_mutation_cache_regression(bit_width);
                check_turbovec_rounding_mode_persistence(bit_width);
            }
            check_turbovec_incremental_block_repacking();
            check_turbovec_sparse_filter_block_selection();

            check_turbovec_numeric_helper_parity();
            check_simd_kernel_parity();
            check_turbovec_1x_rotation_and_codebooks();
            check_turbovec_1x_torn_sync();
            check_turbovec_1x_failed_add_then_sync();
            check_turbovec_1x_failed_sync_keeps_binding();
            check_turbovec_1x_reordered_and_stale_commits();
            check_turbovec_1x_permute_dot_kernels();
            check_turbovec_1x_vm8_layout_choice();
            check_turbovec_1x_vm8_kernel_parity();
            check_turbovec_1x_vm8_mutations();
            check_turbovec_1x_vm8_2bit();
            check_turbovec_1x_batch_goldens();
            check_turbovec_1x_low_planes_tiny();
            check_turbovec_1x_staged();
            check_turbovec_1x_thread_invariance();
            check_vector_index_pool();
            check_turbovec_1x_concurrent_search();
            check_turbovec_1x_staged_maintenance();
            check_turbovec_1x_single_code_copy();
            check_turbovec_1x_streamed_snapshots();
            check_write_with_durability();
            check_turbovec_1x_tie_order();
            check_turbovec_stale_temp_sweep();
            check_turbovec_1x_alloc_failures();
            check_turbovec_1x_cache_recovery();
            check_tqplus_deterministic_layout(
                2,
                UINT64_C(0xc4140782241d45eb),
                UINT64_C(0xb111cd7d1dded99f),
                UINT64_C(0x3b8289e6bc6c026e),
                UINT64_C(0x09481a1adb4e3fe4));
            check_tqplus_deterministic_layout(
                4,
                UINT64_C(0x2c4e8e9e2a991e21),
                UINT64_C(0x140caf9a5c52a967),
                UINT64_C(0x3b8289e6bc6c026e),
                UINT64_C(0x09481a1adb4e3fe4));
        }

        std::filesystem::remove_all(test_temp_dir);
        std::printf("test-vector-index-hooks: OK\n");
        return 0;
    }
#else
    (void) argc;
    (void) argv;
#endif

    CHECK(ggml_vec_index_create(0, /*bit_width=*/32) == nullptr);
    CHECK(ggml_vec_index_create(-1, /*bit_width=*/32) == nullptr);
    CHECK(ggml_vec_index_create(kDim, /*bit_width=*/16) == nullptr);
    CHECK(ggml_vec_index_contains(nullptr, 123ULL) == 0);
    CHECK(ggml_vec_index_len(nullptr) == 0);
    CHECK(ggml_vec_index_dim(nullptr) == 0);
    CHECK(ggml_vec_index_bit_width(nullptr) == 0);
    ggml_vec_index_prepare(nullptr);
    CHECK(ggml_vec_index_create(kDim, 2) == nullptr);
    CHECK(ggml_vec_index_create_turbovec_q2(0) == nullptr);
    CHECK(ggml_vec_index_create_turbovec_q2(kDim) == nullptr);
    CHECK(ggml_vec_index_create_turbovec_q2(kTurboVecMaxTestDim + 8) == nullptr);
    CHECK(ggml_vec_index_create_turbovec_q2(65536) == nullptr);
    CHECK(ggml_vec_index_create_turbovec_q4(0) == nullptr);
    CHECK(ggml_vec_index_create_turbovec_q4(kDim) == nullptr);
    CHECK(ggml_vec_index_create_turbovec_q4(kTurboVecMaxTestDim + 8) == nullptr);
    CHECK(ggml_vec_index_create_turbovec_q4(65536) == nullptr);
    if (!kTurboVecSupported) {
        CHECK(ggml_vec_index_create_turbovec_q2(kTurboVecMaxTestDim) == nullptr);
        CHECK(ggml_vec_index_create_turbovec_q4(kTurboVecMaxTestDim) == nullptr);
    }

    if (kTurboVecSupported) {
        auto * max_dim_q2 = ggml_vec_index_create_turbovec_q2(kTurboVecMaxTestDim);
        CHECK(max_dim_q2 != nullptr);
        ggml_vec_index_free(max_dim_q2);
        auto * max_dim_q4 = ggml_vec_index_create_turbovec_q4(kTurboVecMaxTestDim);
        CHECK(max_dim_q4 != nullptr);
        ggml_vec_index_free(max_dim_q4);

    // The early TurboVec prototype wrote incompatible v2 snapshots. They must
    // not be silently decoded as the v3 Rust-compatible layout.
    {
        const std::string path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-v2-turbovec.tvim").string();
        for (const auto & spec : {
                 std::pair<int, uint8_t>{ 2, 5 },
                 std::pair<int, uint8_t>{ 4, 4 },
             }) {
            std::filesystem::remove(path);
            write_v2_turbovec_index(path, 128, spec.first, spec.second);
            ggml_vec_index_t * loaded = nullptr;
            CHECK(ggml_vec_index_load_ex(path.c_str(), &loaded) ==
                  GGML_VEC_INDEX_E_BAD_VERSION);
            CHECK(loaded == nullptr);
            CHECK(ggml_vec_index_load(path.c_str()) == nullptr);
        }
        std::filesystem::remove(path);
    }

    // Match Rust input validation: finite but unsafe-magnitude coordinates are
    // rejected before they can overflow TurboVec's float norm/score path.
    {
        constexpr int tv_dim = 64;
        for (const int bit_width : { 2, 4 }) {
            auto * tv = bit_width == 2 ?
                ggml_vec_index_create_turbovec_q2(tv_dim) :
                ggml_vec_index_create_turbovec_q4(tv_dim);
            CHECK(tv != nullptr);
            std::vector<float> safe(static_cast<size_t>(tv_dim), 0.25f);
            const uint64_t safe_id = static_cast<uint64_t>(9300 + bit_width);
            CHECK(ggml_vec_index_add(tv, safe.data(), 1, &safe_id) == GGML_VEC_INDEX_OK);
            std::vector<float> unsafe(static_cast<size_t>(tv_dim), 0.0f);
            unsafe[3] = 1e16f;
            const uint64_t unsafe_id = static_cast<uint64_t>(9350 + bit_width);
            CHECK(ggml_vec_index_add(tv, unsafe.data(), 1, &unsafe_id) ==
                  GGML_VEC_INDEX_E_INVALID_ARG);
            CHECK(ggml_vec_index_len(tv) == 1);
            std::array<float, 1> scores{};
            std::array<uint64_t, 1> out{};
            CHECK(ggml_vec_index_search(tv, unsafe.data(), 1, 1, scores.data(), out.data()) ==
                  GGML_VEC_INDEX_E_INVALID_ARG);
            ggml_vec_index_free(tv);
        }
    }

    // TurboVec bit-plane packing only requires dimensions to be multiples of 8.
    {
        constexpr int tv_dim = 64;
        for (const int bit_width : { 2, 4 }) {
            auto * tv = bit_width == 2 ?
                ggml_vec_index_create_turbovec_q2(tv_dim) :
                ggml_vec_index_create_turbovec_q4(tv_dim);
            CHECK(tv != nullptr);

            std::vector<float> tv_vecs(static_cast<size_t>(tv_dim) * 2);
            for (int i = 0; i < tv_dim; ++i) {
                tv_vecs[static_cast<size_t>(i)] =
                    static_cast<float>(std::sin(0.03 * static_cast<double>(i + 1)));
                tv_vecs[static_cast<size_t>(tv_dim + i)] =
                    static_cast<float>(std::cos(0.05 * static_cast<double>(i + 3)));
            }
            const std::array<uint64_t, 2> tv_ids = {
                static_cast<uint64_t>(9400 + bit_width),
                static_cast<uint64_t>(9500 + bit_width),
            };
            CHECK(ggml_vec_index_add(tv, tv_vecs.data(), 2, tv_ids.data()) ==
                  GGML_VEC_INDEX_OK);

            std::array<float, 2> scores{};
            std::array<uint64_t, 2> out{};
            CHECK(ggml_vec_index_search(tv, tv_vecs.data(), 1, 2, scores.data(), out.data()) ==
                  GGML_VEC_INDEX_OK);
            CHECK(out[0] == tv_ids[0]);

            const std::string path =
                (std::filesystem::temp_directory_path() /
                 ("ggml-vector-index-turbovec-dim64-q" + std::to_string(bit_width) + ".tvim")).string();
            std::filesystem::remove(path);
            CHECK(ggml_vec_index_write(tv, path.c_str()) == GGML_VEC_INDEX_OK);
            auto * loaded = ggml_vec_index_load(path.c_str());
            CHECK(loaded != nullptr);
            CHECK(ggml_vec_index_dim(loaded) == tv_dim);
            CHECK(ggml_vec_index_bit_width(loaded) == bit_width);
            CHECK(ggml_vec_index_search(
                loaded, tv_vecs.data(), 1, 2, scores.data(), out.data()) ==
                GGML_VEC_INDEX_OK);
            CHECK(out[0] == tv_ids[0]);
            ggml_vec_index_free(loaded);
            ggml_vec_index_free(tv);
            std::filesystem::remove(path);
        }
    }

    // Sparse filters score only touched TurboVec blocks and preserve the exact
    // scores produced by an unfiltered search.
    {
        constexpr int tv_dim = 128;
        constexpr int n_vecs = 96;
        for (const int bit_width : { 2, 4 }) {
            auto * tv = bit_width == 2 ?
                ggml_vec_index_create_turbovec_q2(tv_dim) :
                ggml_vec_index_create_turbovec_q4(tv_dim);
            CHECK(tv != nullptr);
            std::vector<float> tv_vecs(static_cast<size_t>(tv_dim) * n_vecs);
            std::vector<uint64_t> tv_ids(static_cast<size_t>(n_vecs));
            for (int row = 0; row < n_vecs; ++row) {
                tv_ids[static_cast<size_t>(row)] =
                    static_cast<uint64_t>(9700 + bit_width * 100 + row);
                for (int col = 0; col < tv_dim; ++col) {
                    const double x = static_cast<double>(row + 1);
                    const double y = static_cast<double>(col + 5);
                    tv_vecs[static_cast<size_t>(row) * tv_dim + static_cast<size_t>(col)] =
                        static_cast<float>(
                            0.44 * std::sin(0.017 * x * y + 0.13) +
                            0.38 * std::cos(0.029 * (x + 2.0) * (y + 1.0)) +
                            0.18 * std::sin(0.061 * (x + y)));
                }
            }
            CHECK(ggml_vec_index_add(tv, tv_vecs.data(), n_vecs, tv_ids.data()) ==
                  GGML_VEC_INDEX_OK);

            const std::array<uint64_t, 2> allowed = {
                tv_ids[5],
                tv_ids[63],
            };
            std::array<float, 4> scores{};
            std::array<uint64_t, 4> out{};
            std::vector<float> all_scores(n_vecs);
            std::vector<uint64_t> all_out(n_vecs);
            CHECK(ggml_vec_index_search(
                tv, tv_vecs.data(), 1, n_vecs, all_scores.data(), all_out.data()) ==
                GGML_VEC_INDEX_OK);
            CHECK(ggml_vec_index_search_filtered(
                tv,
                tv_vecs.data(),
                1,
                2,
                allowed.data(),
                static_cast<int>(allowed.size()),
                scores.data(),
                out.data()) == GGML_VEC_INDEX_OK);
            for (size_t i = 0; i < allowed.size(); ++i) {
                const auto it = std::find(all_out.begin(), all_out.end(), out[i]);
                CHECK(it != all_out.end());
                const size_t position = static_cast<size_t>(it - all_out.begin());
                CHECK(scores[i] == all_scores[position]);
            }
            ggml_vec_index_free(tv);
        }
    }

    // TurboVec q2/q4 are distinct modes. This first milestone supports
    // add/search/filter/IVF and regular snapshots; delta logs are format-gated.
    {
        constexpr int tv_dim = 128;
        auto * tv = ggml_vec_index_create_turbovec_q2(tv_dim);
        CHECK(tv != nullptr);
        CHECK(ggml_vec_index_dim(tv) == tv_dim);
        CHECK(ggml_vec_index_bit_width(tv) == 2);

        std::vector<float> tv_vecs(static_cast<size_t>(tv_dim) * 3);
        for (int i = 0; i < tv_dim; ++i) {
            const float a = static_cast<float>(std::sin(0.07 * static_cast<double>(i + 1)));
            const float b = static_cast<float>(std::cos(0.11 * static_cast<double>(i + 3)));
            tv_vecs[static_cast<size_t>(i)] = a;
            tv_vecs[static_cast<size_t>(tv_dim + i)] = -a;
            tv_vecs[static_cast<size_t>(2 * tv_dim + i)] = b;
        }
        const std::array<uint64_t, 3> tv_ids = { 9201, 9202, 9203 };
        CHECK(ggml_vec_index_add(tv, tv_vecs.data(), 3, tv_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_len(tv) == 3);

        std::array<float, 3> tv_scores{};
        std::array<uint64_t, 3> tv_out{};
        CHECK(ggml_vec_index_search(
            tv, tv_vecs.data(), 1, 3, tv_scores.data(), tv_out.data()) == GGML_VEC_INDEX_OK);
        CHECK(tv_out[0] == tv_ids[0]);

        const std::array<uint64_t, 2> allowed = { tv_ids[1], tv_ids[2] };
        CHECK(ggml_vec_index_search_filtered(
            tv, tv_vecs.data(), 1, 1, allowed.data(), static_cast<int>(allowed.size()),
            tv_scores.data(), tv_out.data()) == GGML_VEC_INDEX_OK);
        CHECK(tv_out[0] != tv_ids[0]);

        CHECK(ggml_vec_index_build_ivf(tv, /*n_lists=*/2, /*n_iter=*/1) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search_ivf(
            tv, tv_vecs.data(), 1, 1, /*nprobe=*/2,
            tv_scores.data(), tv_out.data()) == GGML_VEC_INDEX_OK);
        CHECK(tv_out[0] == tv_ids[0]);

        const std::string tv_snapshot_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-turbovec-q2.tvim").string();
        const std::string tv_delta_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-turbovec-q2.tvid").string();
        const std::string tv_unchecksummed_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-turbovec-q2-unchecksummed.tvim").string();
        std::filesystem::remove(tv_snapshot_path);
        std::filesystem::remove(tv_delta_path);
        std::filesystem::remove(tv_unchecksummed_path);
        std::filesystem::remove(tv_delta_path + ".lock");
        CHECK(ggml_vec_index_write(tv, tv_snapshot_path.c_str()) == GGML_VEC_INDEX_OK);
        auto * tv_loaded = ggml_vec_index_load(tv_snapshot_path.c_str());
        CHECK(tv_loaded != nullptr);
        CHECK(ggml_vec_index_dim(tv_loaded) == tv_dim);
        CHECK(ggml_vec_index_bit_width(tv_loaded) == 2);
        CHECK(ggml_vec_index_len(tv_loaded) == 3);
        ggml_vec_index_prepare(tv_loaded);
        CHECK(ggml_vec_index_search(
            tv_loaded, tv_vecs.data(), 1, 3, tv_scores.data(), tv_out.data()) ==
            GGML_VEC_INDEX_OK);
        CHECK(tv_out[0] == tv_ids[0]);
        CHECK(ggml_vec_index_load_mmap(tv_snapshot_path.c_str()) == nullptr);
        ggml_vec_index_free(tv_loaded);
        // TurboVec 1.x snapshots are upstream v8 containers; a damaged
        // superblock fails its CRC instead of loading.
        std::vector<uint8_t> corrupt_tv = read_file_bytes(tv_snapshot_path);
        CHECK(corrupt_tv.size() > 64);
        CHECK(std::memcmp(corrupt_tv.data(), "TV8", 4) == 0);
        corrupt_tv[40] ^= 0x01;
        write_file_bytes(tv_unchecksummed_path, corrupt_tv);
        auto * corrupt_loaded = ggml_vec_index_load(tv_unchecksummed_path.c_str());
        CHECK(corrupt_loaded == nullptr);
        ggml_vec_index_free(corrupt_loaded);
        ggml_vec_index_t * tv_delta_loaded = nullptr;
        CHECK(ggml_vec_index_load_with_delta_ex(
            tv_snapshot_path.c_str(), tv_delta_path.c_str(), &tv_delta_loaded) ==
            GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(tv_delta_loaded == nullptr);
        CHECK(ggml_vec_index_load_with_delta(
            tv_snapshot_path.c_str(), tv_delta_path.c_str()) == nullptr);
        CHECK(ggml_vec_index_compact_delta(
            tv, tv_snapshot_path.c_str(), tv_delta_path.c_str()) ==
            GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(!std::filesystem::exists(tv_delta_path));
        CHECK(ggml_vec_index_remove(tv, tv_ids[1]) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_compact(tv) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_len(tv) == 2);
        CHECK(ggml_vec_index_contains(tv, tv_ids[0]) == 1);
        CHECK(ggml_vec_index_contains(tv, tv_ids[1]) == 0);
        const uint64_t replacement_id = 9204;
        CHECK(ggml_vec_index_add(
            tv,
            tv_vecs.data() + tv_dim,
            1,
            &replacement_id) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search(
            tv,
            tv_vecs.data() + tv_dim,
            1,
            1,
            tv_scores.data(),
            tv_out.data()) == GGML_VEC_INDEX_OK);
        CHECK(tv_out[0] == replacement_id);
        CHECK(ggml_vec_index_add_logged(
            tv, tv_vecs.data(), 1, &tv_ids[0], tv_delta_path.c_str()) ==
            GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_remove_logged(tv, tv_ids[0], tv_delta_path.c_str()) ==
            GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(!std::filesystem::exists(tv_delta_path));
        std::filesystem::remove(tv_snapshot_path);
        std::filesystem::remove(tv_unchecksummed_path);
        std::filesystem::remove(tv_delta_path + ".lock");
        ggml_vec_index_free(tv);
    }

    // TurboVec q4 is a distinct mode. This first milestone supports
    // add/search/filter/IVF and regular snapshots; delta logs are format-gated.
    {
        constexpr int tv_dim = 128;
        auto * tv = ggml_vec_index_create_turbovec_q4(tv_dim);
        CHECK(tv != nullptr);
        CHECK(ggml_vec_index_dim(tv) == tv_dim);
        CHECK(ggml_vec_index_bit_width(tv) == 4);

        std::vector<float> tv_vecs(static_cast<size_t>(tv_dim) * 3);
        for (int i = 0; i < tv_dim; ++i) {
            const float a = static_cast<float>(std::sin(0.07 * static_cast<double>(i + 1)));
            const float b = static_cast<float>(std::cos(0.11 * static_cast<double>(i + 3)));
            tv_vecs[static_cast<size_t>(i)] = a;
            tv_vecs[static_cast<size_t>(tv_dim + i)] = -a;
            tv_vecs[static_cast<size_t>(2 * tv_dim + i)] = b;
        }
        const std::array<uint64_t, 3> tv_ids = { 9101, 9102, 9103 };
        CHECK(ggml_vec_index_add(tv, tv_vecs.data(), 3, tv_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_len(tv) == 3);

        std::array<float, 3> tv_scores{};
        std::array<uint64_t, 3> tv_out{};
        CHECK(ggml_vec_index_search(
            tv, tv_vecs.data(), 1, 3, tv_scores.data(), tv_out.data()) == GGML_VEC_INDEX_OK);
        CHECK(tv_out[0] == tv_ids[0]);

        const std::array<uint64_t, 2> allowed = { tv_ids[1], tv_ids[2] };
        CHECK(ggml_vec_index_search_filtered(
            tv, tv_vecs.data(), 1, 1, allowed.data(), static_cast<int>(allowed.size()),
            tv_scores.data(), tv_out.data()) == GGML_VEC_INDEX_OK);
        CHECK(tv_out[0] != tv_ids[0]);

        CHECK(ggml_vec_index_build_ivf(tv, /*n_lists=*/2, /*n_iter=*/1) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search_ivf(
            tv, tv_vecs.data(), 1, 1, /*nprobe=*/2,
            tv_scores.data(), tv_out.data()) == GGML_VEC_INDEX_OK);
        CHECK(tv_out[0] == tv_ids[0]);

        const std::string tv_snapshot_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-turbovec-q4.tvim").string();
        const std::string tv_delta_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-turbovec-q4.tvid").string();
        std::filesystem::remove(tv_snapshot_path);
        std::filesystem::remove(tv_delta_path);
        std::filesystem::remove(tv_delta_path + ".lock");
        CHECK(ggml_vec_index_write(tv, tv_snapshot_path.c_str()) == GGML_VEC_INDEX_OK);
        auto * tv_loaded = ggml_vec_index_load(tv_snapshot_path.c_str());
        CHECK(tv_loaded != nullptr);
        CHECK(ggml_vec_index_dim(tv_loaded) == tv_dim);
        CHECK(ggml_vec_index_bit_width(tv_loaded) == 4);
        CHECK(ggml_vec_index_len(tv_loaded) == 3);
        ggml_vec_index_prepare(tv_loaded);
        CHECK(ggml_vec_index_search(
            tv_loaded, tv_vecs.data(), 1, 3, tv_scores.data(), tv_out.data()) ==
            GGML_VEC_INDEX_OK);
        CHECK(tv_out[0] == tv_ids[0]);
        CHECK(ggml_vec_index_load_mmap(tv_snapshot_path.c_str()) == nullptr);
        ggml_vec_index_free(tv_loaded);
        ggml_vec_index_t * tv_delta_loaded = nullptr;
        CHECK(ggml_vec_index_load_with_delta_ex(
            tv_snapshot_path.c_str(), tv_delta_path.c_str(), &tv_delta_loaded) ==
            GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(tv_delta_loaded == nullptr);
        CHECK(ggml_vec_index_load_with_delta(
            tv_snapshot_path.c_str(), tv_delta_path.c_str()) == nullptr);
        CHECK(ggml_vec_index_compact_delta(
            tv, tv_snapshot_path.c_str(), tv_delta_path.c_str()) ==
            GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(!std::filesystem::exists(tv_delta_path));
        CHECK(ggml_vec_index_remove(tv, tv_ids[1]) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_compact(tv) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_len(tv) == 2);
        CHECK(ggml_vec_index_contains(tv, tv_ids[0]) == 1);
        CHECK(ggml_vec_index_contains(tv, tv_ids[1]) == 0);
        const uint64_t replacement_id = 9104;
        CHECK(ggml_vec_index_add(
            tv,
            tv_vecs.data() + tv_dim,
            1,
            &replacement_id) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search(
            tv,
            tv_vecs.data() + tv_dim,
            1,
            1,
            tv_scores.data(),
            tv_out.data()) == GGML_VEC_INDEX_OK);
        CHECK(tv_out[0] == replacement_id);
        CHECK(ggml_vec_index_add_logged(
            tv, tv_vecs.data(), 1, &tv_ids[0], tv_delta_path.c_str()) ==
            GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_remove_logged(tv, tv_ids[0], tv_delta_path.c_str()) ==
            GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(!std::filesystem::exists(tv_delta_path));
        std::filesystem::remove(tv_snapshot_path);
        std::filesystem::remove(tv_delta_path + ".lock");
        ggml_vec_index_free(tv);
    }

    for (const int bit_width : { 2, 4 }) {
        check_turbovec_oversized_snapshot_compatibility(bit_width);
        check_turbovec_oversized_snapshot_compatibility(bit_width, 40000);
        check_turbovec_v3_corruption_rejected(bit_width);
        check_turbovec_tiny_tqplus_scale_rejected(bit_width);
        check_turbovec_zero_scale_and_legacy_calibration(bit_width);
    }

    if (kTurboVecSupported) {
        check_turbovec_1x();
    }

    // Rust TurboVec golden parity: generated by tests/turbovec-golden-gen.
    // The small fixtures use Rust's identity TQ+ fallback.
    check_turbovec_rust_golden(
        "q2-golden",
        2,
        5,
        kTurboVecGoldenQ2Dim,
        kTurboVecGoldenQ2NDb,
        kTurboVecGoldenQ2NQuery,
        kTurboVecGoldenQ2K,
        kTurboVecGoldenQ2RustRotationHash,
        kTurboVecGoldenQ2Db,
        kTurboVecGoldenQ2Queries,
        kTurboVecGoldenQ2RustScores,
        kTurboVecGoldenQ2RustScales,
        kTurboVecGoldenQ2RustScaleCount,
        kTurboVecGoldenQ2RustTvBytes,
        kTurboVecGoldenQ2RustTvBytesLen,
        kTurboVecGoldenQ2RustPackedCodes,
        kTurboVecGoldenQ2RustPackedBytes,
        kTurboVecGoldenQ2RustCalibCount,
        kTurboVecGoldenQ2TopK);
    check_turbovec_rust_golden(
        "q4-golden",
        4,
        4,
        kTurboVecGoldenDim,
        kTurboVecGoldenNDb,
        kTurboVecGoldenNQuery,
        kTurboVecGoldenK,
        kTurboVecGoldenRustRotationHash,
        kTurboVecGoldenDb,
        kTurboVecGoldenQueries,
        kTurboVecGoldenRustScores,
        kTurboVecGoldenRustScales,
        kTurboVecGoldenRustScaleCount,
        kTurboVecGoldenRustTvBytes,
        kTurboVecGoldenRustTvBytesLen,
        kTurboVecGoldenRustPackedCodes,
        kTurboVecGoldenRustPackedBytes,
        kTurboVecGoldenRustCalibCount,
        kTurboVecGoldenTopK);
    check_turbovec_rust_golden(
        "q2-dim256-golden",
        2,
        5,
        kTurboVecGoldenDim256Q2Dim,
        kTurboVecGoldenDim256Q2NDb,
        kTurboVecGoldenDim256Q2NQuery,
        kTurboVecGoldenDim256Q2K,
        kTurboVecGoldenDim256Q2RustRotationHash,
        kTurboVecGoldenDim256Q2Db,
        kTurboVecGoldenDim256Q2Queries,
        kTurboVecGoldenDim256Q2RustScores,
        kTurboVecGoldenDim256Q2RustScales,
        kTurboVecGoldenDim256Q2RustScaleCount,
        kTurboVecGoldenDim256Q2RustTvBytes,
        kTurboVecGoldenDim256Q2RustTvBytesLen,
        kTurboVecGoldenDim256Q2RustPackedCodes,
        kTurboVecGoldenDim256Q2RustPackedBytes,
        kTurboVecGoldenDim256Q2RustCalibCount,
        kTurboVecGoldenDim256Q2TopK);
    check_turbovec_rust_golden(
        "q4-dim256-golden",
        4,
        4,
        kTurboVecGoldenDim256Q4Dim,
        kTurboVecGoldenDim256Q4NDb,
        kTurboVecGoldenDim256Q4NQuery,
        kTurboVecGoldenDim256Q4K,
        kTurboVecGoldenDim256Q4RustRotationHash,
        kTurboVecGoldenDim256Q4Db,
        kTurboVecGoldenDim256Q4Queries,
        kTurboVecGoldenDim256Q4RustScores,
        kTurboVecGoldenDim256Q4RustScales,
        kTurboVecGoldenDim256Q4RustScaleCount,
        kTurboVecGoldenDim256Q4RustTvBytes,
        kTurboVecGoldenDim256Q4RustTvBytesLen,
        kTurboVecGoldenDim256Q4RustPackedCodes,
        kTurboVecGoldenDim256Q4RustPackedBytes,
        kTurboVecGoldenDim256Q4RustCalibCount,
        kTurboVecGoldenDim256Q4TopK);

    }

    auto * idx = ggml_vec_index_create(kDim, /*bit_width=*/32);
    CHECK(idx != nullptr);
    CHECK(ggml_vec_index_dim(idx) == kDim);
    CHECK(ggml_vec_index_len(idx) == 0);
    CHECK(ggml_vec_index_bit_width(idx) == 32);
    CHECK(ggml_vec_index_create(0, /*bit_width=*/32) == nullptr);
    CHECK(ggml_vec_index_create(kDim, /*bit_width=*/31) == nullptr);
    CHECK(std::strcmp(ggml_vec_index_error_to_string(GGML_VEC_INDEX_E_NOT_DURABLE), "not durable") == 0);
    ggml_vec_index_prepare(idx);

    // Zero-row plain adds are valid no-ops. Logged zero-row adds still obey
    // the current delta start policy and must not create artifacts.
    {
        const std::array<float, kDim> vector = {
            1.0f, 0.0f, 0.0f, 0.0f,
        };
        const uint64_t id = 1234ULL;
        std::array<float, 1> scores{};
        std::array<uint64_t, 1> out_ids{};
        CHECK(ggml_vec_index_add(idx, vector.data(), /*n=*/0, &id)
              == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_add(idx, nullptr, /*n=*/0, nullptr)
              == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_len(idx) == 0);

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
        const std::string missing_path =
            (std::filesystem::temp_directory_path() / "ggml-vector-index-missing.tvim").string();
        std::filesystem::remove(missing_path);
        CHECK(ggml_vec_index_load(missing_path.c_str()) == nullptr);
        CHECK(ggml_vec_index_remove(nullptr, id) == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_contains(nullptr, id) == 0);
        CHECK(ggml_vec_index_len(nullptr) == 0);
        CHECK(ggml_vec_index_dim(nullptr) == 0);
        CHECK(ggml_vec_index_bit_width(nullptr) == 0);
        ggml_vec_index_prepare(nullptr);

        const std::string zero_delta_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-zero-add.tvid").string();
        std::filesystem::remove(zero_delta_path);
        std::filesystem::remove(zero_delta_path + ".lock");
        CHECK(ggml_vec_index_add_logged(
            idx, vector.data(), /*n=*/0, &id, zero_delta_path.c_str()) ==
            GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_add_logged(
            idx, nullptr, /*n=*/0, nullptr, zero_delta_path.c_str()) ==
            GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_len(idx) == 0);
        CHECK(!std::filesystem::exists(zero_delta_path));
        std::filesystem::remove(zero_delta_path + ".lock");
    }

    // Non-finite vectors are rejected without mutation.
    {
        const std::array<float, kDim> bad_vector = {
            1.0f, 0.0f, std::numeric_limits<float>::infinity(), 0.0f,
        };
        const uint64_t bad_id = 777ULL;
        CHECK(ggml_vec_index_add(idx, bad_vector.data(), 1, &bad_id)
              == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_len(idx) == 0);
    }

#ifdef GGML_VEC_INDEX_TEST_HOOKS
    // Byte-span arithmetic is checked independently of the host word size so
    // 64-bit CI still covers the overflow boundary used by 32-bit builds.
    {
        constexpr size_t max_size = std::numeric_limits<size_t>::max();
        CHECK(ggml_vec_index_test_can_address_array(max_size / sizeof(float), sizeof(float)) == 1);
        CHECK(ggml_vec_index_test_can_address_array(max_size / sizeof(float) + 1, sizeof(float)) == 0);
        CHECK(ggml_vec_index_test_can_address_array(max_size / sizeof(uint64_t), sizeof(uint64_t)) == 1);
        CHECK(ggml_vec_index_test_can_address_array(max_size / sizeof(uint64_t) + 1, sizeof(uint64_t)) == 0);
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

    // UINT64_MAX is reserved as the empty-result sentinel.
    {
        const std::array<float, kDim> vector = {
            1.0f, 0.0f, 0.0f, 0.0f,
        };
        const uint64_t reserved_id = UINT64_MAX;
        CHECK(ggml_vec_index_add(idx, vector.data(), 1, &reserved_id)
              == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_remove(idx, reserved_id)
              == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_len(idx) == 0);
    }

    // Padding is identified by the id sentinel, not by score alone.
    {
        const std::array<float, kDim> min_score_vector = {
            -FLT_MAX, 0.0f, 0.0f, 0.0f,
        };
        const std::array<float, kDim> min_score_query = {
            1.0f, 0.0f, 0.0f, 0.0f,
        };
        const uint64_t min_score_id = 12345ULL;
        CHECK(ggml_vec_index_add(idx, min_score_vector.data(), 1, &min_score_id) ==
              GGML_VEC_INDEX_OK);
        std::array<float, 2> scores{};
        std::array<uint64_t, 2> out_ids{};
        CHECK(ggml_vec_index_search(
            idx, min_score_query.data(), 1, /*k=*/2,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(out_ids[0] == min_score_id);
        CHECK(scores[0] == -FLT_MAX);
        CHECK(out_ids[1] == UINT64_MAX);
        CHECK(scores[1] == -FLT_MAX);
        CHECK(ggml_vec_index_remove(idx, min_score_id) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_len(idx) == 0);
    }

    // Add 4 well-separated unit vectors. IDs are non-trivial uint64 to
    // catch sign-extension bugs when this codepath is called from bindings.
    std::vector<float> vecs;
    std::vector<uint64_t> ids = {
        42ULL,
        (1ULL << 40) + 7ULL,
        (1ULL << 62) + 11ULL,
        UINT64_MAX - 13ULL,
    };
    std::vector<std::vector<float>> seeds = {
        normalize({1.0f, 0.0f, 0.0f, 0.0f}),
        normalize({0.0f, 1.0f, 0.0f, 0.0f}),
        normalize({0.0f, 0.0f, 1.0f, 0.0f}),
        normalize({0.0f, 0.0f, 0.0f, 1.0f}),
    };
    for (const auto & s : seeds) {
        vecs.insert(vecs.end(), s.begin(), s.end());
    }
    CHECK(ggml_vec_index_add(
        idx, vecs.data(), static_cast<int>(ids.size()), ids.data()) == 0);
    CHECK(ggml_vec_index_len(idx) == 4);
    CHECK(ggml_vec_index_contains(idx, ids[0]) == 1);
    CHECK(ggml_vec_index_contains(idx, 999ULL) == 0);

    // Zero-query searches are no-ops, while k=0 is invalid for every search mode.
    {
        std::array<float, 4> scores = { 123.0f, 456.0f, 789.0f, 101.0f };
        std::array<uint64_t, 4> out_ids = { 1, 2, 3, 4 };
        CHECK(ggml_vec_index_search(
            idx, seeds[0].data(), /*n_q=*/0, /*k=*/1,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search(
            idx, nullptr, /*n_q=*/0, /*k=*/1, nullptr, nullptr) ==
            GGML_VEC_INDEX_OK);
        CHECK(scores[0] == 123.0f);
        CHECK(out_ids[0] == 1);
        CHECK(ggml_vec_index_search(
            idx, seeds[0].data(), /*n_q=*/1, /*k=*/0,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);

        const std::array<uint64_t, 2> allowed = { ids[0], ids[2] };
        CHECK(ggml_vec_index_search_filtered(
            idx, seeds[0].data(), /*n_q=*/0, /*k=*/1,
            allowed.data(), static_cast<int>(allowed.size()),
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search_filtered(
            idx, nullptr, /*n_q=*/0, /*k=*/1,
            allowed.data(), static_cast<int>(allowed.size()), nullptr, nullptr) ==
            GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search_filtered(
            idx, seeds[0].data(), /*n_q=*/1, /*k=*/0,
            allowed.data(), static_cast<int>(allowed.size()),
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);

        ggml_vec_index_filter_t * filter = ggml_vec_index_filter_create(
            idx, allowed.data(), static_cast<int>(allowed.size()));
        CHECK(filter != nullptr);
        CHECK(ggml_vec_index_search_prepared_filtered(
            idx, filter, seeds[0].data(), /*n_q=*/0, /*k=*/1,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search_prepared_filtered(
            idx, filter, nullptr, /*n_q=*/0, /*k=*/1, nullptr, nullptr) ==
            GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search_prepared_filtered(
            idx, filter, seeds[0].data(), /*n_q=*/1, /*k=*/0,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);
        ggml_vec_index_filter_free(filter);

        CHECK(ggml_vec_index_build_ivf(idx, /*n_lists=*/2, /*n_iter=*/1)
              == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search_ivf(
            idx, seeds[0].data(), /*n_q=*/0, /*k=*/1, /*nprobe=*/1,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search_ivf(
            idx, nullptr, /*n_q=*/0, /*k=*/1, /*nprobe=*/1, nullptr, nullptr) ==
            GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search_ivf(
            idx, seeds[0].data(), /*n_q=*/1, /*k=*/0, /*nprobe=*/1,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);
    }

    // Multi-query searches write independent result rows for every mode.
    {
        std::vector<float> batch_queries;
        batch_queries.insert(batch_queries.end(), seeds[0].begin(), seeds[0].end());
        batch_queries.insert(batch_queries.end(), seeds[2].begin(), seeds[2].end());
        const std::vector<float> mixed_query = normalize({0.6f, 0.8f, 0.0f, 0.0f});
        batch_queries.insert(batch_queries.end(), mixed_query.begin(), mixed_query.end());

        std::array<float, 6> exact_scores{};
        std::array<float, 6> filtered_scores{};
        std::array<float, 6> prepared_scores{};
        std::array<float, 6> ivf_scores{};
        std::array<uint64_t, 6> exact_ids{};
        std::array<uint64_t, 6> filtered_ids{};
        std::array<uint64_t, 6> prepared_ids{};
        std::array<uint64_t, 6> ivf_ids{};

        CHECK(ggml_vec_index_search(
            idx, batch_queries.data(), /*n_q=*/3, /*k=*/2,
            exact_scores.data(), exact_ids.data()) == GGML_VEC_INDEX_OK);
        const std::array<uint64_t, 6> expected_exact = {
            ids[0], ids[1],
            ids[2], ids[0],
            ids[1], ids[0],
        };
        CHECK(exact_ids == expected_exact);
        CHECK(exact_scores[0] == 1.0f);
        CHECK(exact_scores[2] == 1.0f);
        CHECK(exact_scores[4] > exact_scores[5]);

        const std::array<uint64_t, 2> allowed = { ids[0], ids[2] };
        CHECK(ggml_vec_index_search_filtered(
            idx, batch_queries.data(), /*n_q=*/3, /*k=*/2,
            allowed.data(), static_cast<int>(allowed.size()),
            filtered_scores.data(), filtered_ids.data()) == GGML_VEC_INDEX_OK);
        const std::array<uint64_t, 6> expected_filtered = {
            ids[0], ids[2],
            ids[2], ids[0],
            ids[0], ids[2],
        };
        CHECK(filtered_ids == expected_filtered);

        ggml_vec_index_filter_t * filter = ggml_vec_index_filter_create(
            idx, allowed.data(), static_cast<int>(allowed.size()));
        CHECK(filter != nullptr);
        CHECK(ggml_vec_index_search_prepared_filtered(
            idx, filter, batch_queries.data(), /*n_q=*/3, /*k=*/2,
            prepared_scores.data(), prepared_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(prepared_ids == expected_filtered);
        ggml_vec_index_filter_free(filter);

        CHECK(ggml_vec_index_build_ivf(idx, /*n_lists=*/4, /*n_iter=*/2)
              == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search_ivf(
            idx, batch_queries.data(), /*n_q=*/3, /*k=*/2, /*nprobe=*/4,
            ivf_scores.data(), ivf_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(ivf_ids == expected_exact);
        for (size_t i = 0; i < exact_scores.size(); ++i) {
            CHECK(std::fabs(ivf_scores[i] - exact_scores[i]) < 1e-5f);
        }
    }

    // Non-finite queries are rejected before search.
    {
        const std::array<float, kDim> bad_query = {
            1.0f, std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f,
        };
        std::array<float, 1> scores{};
        std::array<uint64_t, 1> out_ids{};
        CHECK(ggml_vec_index_search(
            idx, bad_query.data(), 1, /*k=*/1,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);
    }

    // Duplicate add must fail without mutating state.
    {
        const std::vector<uint64_t> dup_ids = { ids[0] };
        std::vector<float> dup_vec(seeds[0]);
        CHECK(ggml_vec_index_add(idx, dup_vec.data(), 1, dup_ids.data())
              == GGML_VEC_INDEX_E_DUPLICATE);
        CHECK(ggml_vec_index_len(idx) == 4);
    }

    // In-batch duplicate ids must also fail atomically.
    {
        const uint64_t new_id = (1ULL << 50) + 123ULL;
        const std::vector<uint64_t> dup_ids = { new_id, new_id };
        std::vector<float> dup_vecs;
        dup_vecs.insert(dup_vecs.end(), seeds[0].begin(), seeds[0].end());
        dup_vecs.insert(dup_vecs.end(), seeds[1].begin(), seeds[1].end());
        CHECK(ggml_vec_index_add(idx, dup_vecs.data(), 2, dup_ids.data())
              == GGML_VEC_INDEX_E_DUPLICATE);
        CHECK(ggml_vec_index_len(idx) == 4);
        CHECK(ggml_vec_index_contains(idx, new_id) == 0);
    }

    // IVF-flat ANN search is explicit and stale builds are rejected.
    {
        auto * ann = ggml_vec_index_create(kDim, /*bit_width=*/32);
        CHECK(ann != nullptr);
        CHECK(ggml_vec_index_add(
            ann, vecs.data(), static_cast<int>(ids.size()), ids.data()) == GGML_VEC_INDEX_OK);

        const std::vector<float> query = normalize({0.9f, 0.3f, 0.1f, -0.2f});
        std::array<float, 4> exact_scores{};
        std::array<float, 4> ann_scores{};
        std::array<uint64_t, 4> exact_ids{};
        std::array<uint64_t, 4> ann_ids{};

        CHECK(ggml_vec_index_search_ivf(
            ann, query.data(), 1, /*k=*/1, /*nprobe=*/1,
            ann_scores.data(), ann_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_build_ivf(ann, /*n_lists=*/0, /*n_iter=*/1)
              == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_build_ivf(ann, /*n_lists=*/16, /*n_iter=*/3)
              == GGML_VEC_INDEX_OK);

        CHECK(ggml_vec_index_search(
            ann, query.data(), 1, /*k=*/4,
            exact_scores.data(), exact_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search_ivf(
            ann, query.data(), 1, /*k=*/4, /*nprobe=*/16,
            ann_scores.data(), ann_ids.data()) == GGML_VEC_INDEX_OK);
        for (int i = 0; i < 4; ++i) {
            CHECK(ann_ids[i] == exact_ids[i]);
            CHECK(std::fabs(ann_scores[i] - exact_scores[i]) < 1e-5f);
        }

        CHECK(ggml_vec_index_search_ivf(
            ann, seeds[0].data(), 1, /*k=*/1, /*nprobe=*/1,
            ann_scores.data(), ann_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(ann_ids[0] == ids[0]);

        const uint64_t ann_new_id = 9999991ULL;
        CHECK(ggml_vec_index_add(ann, seeds[3].data(), 1, &ann_new_id)
              == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search_ivf(
            ann, query.data(), 1, /*k=*/1, /*nprobe=*/1,
            ann_scores.data(), ann_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);

        CHECK(ggml_vec_index_build_ivf(ann, /*n_lists=*/16, /*n_iter=*/3)
              == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search_ivf(
            ann, query.data(), 1, /*k=*/1, /*nprobe=*/16,
            ann_scores.data(), ann_ids.data()) == GGML_VEC_INDEX_OK);
        ggml_vec_index_free(ann);

        auto * empty_list_ann = ggml_vec_index_create(kDim, /*bit_width=*/32);
        CHECK(empty_list_ann != nullptr);
        const std::array<float, 12> empty_list_vecs = {
             3.0f, 3.0f, 0.0f, 0.0f,
            -2.0f, 1.0f, 0.0f, 0.0f,
             0.0f, 1.0f, 0.0f, 0.0f,
        };
        const std::array<uint64_t, 3> empty_list_ids = { 7101, 7102, 7103 };
        const std::array<float, 4> empty_list_query = { 1.0f, -3.0f, 0.0f, 0.0f };
        CHECK(ggml_vec_index_add(
            empty_list_ann,
            empty_list_vecs.data(),
            static_cast<int>(empty_list_ids.size()),
            empty_list_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_build_ivf(empty_list_ann, /*n_lists=*/3, /*n_iter=*/1)
              == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search_ivf(
            empty_list_ann,
            empty_list_query.data(),
            1,
            /*k=*/1,
            /*nprobe=*/1,
            ann_scores.data(),
            ann_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(ann_ids[0] == empty_list_ids[2]);
        CHECK(std::fabs(ann_scores[0] + 3.0f) < 1e-5f);
        ggml_vec_index_free(empty_list_ann);
    }

    // Read-only APIs on one handle can run concurrently.
    {
        constexpr int n_rows = 16;
        std::vector<float> rows;
        std::vector<uint64_t> row_ids;
        rows.reserve(static_cast<size_t>(n_rows) * kDim);
        row_ids.reserve(n_rows);
        for (int row = 0; row < n_rows; ++row) {
            const std::vector<float> v = normalize({
                static_cast<float>((row % 5) - 2),
                static_cast<float>(((row + 1) % 7) - 3),
                static_cast<float>(((row * 3) % 11) - 5),
                1.0f,
            });
            rows.insert(rows.end(), v.begin(), v.end());
            row_ids.push_back(static_cast<uint64_t>(7000 + row));
        }

        auto * concurrent = ggml_vec_index_create(kDim, /*bit_width=*/8);
        CHECK(concurrent != nullptr);
        CHECK(ggml_vec_index_add(concurrent, rows.data(), n_rows, row_ids.data())
              == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_build_ivf(concurrent, /*n_lists=*/4, /*n_iter=*/2)
              == GGML_VEC_INDEX_OK);

        const std::array<uint64_t, 5> allowed = {
            row_ids[0], row_ids[2], row_ids[4], row_ids[6], row_ids[8],
        };
        ggml_vec_index_filter_t * filter = ggml_vec_index_filter_create(
            concurrent, allowed.data(), static_cast<int>(allowed.size()));
        CHECK(filter != nullptr);

        std::atomic<int> ready{ 0 };
        std::atomic<bool> start{ false };
        std::vector<std::thread> threads;
        for (int t = 0; t < 8; ++t) {
            threads.emplace_back([&, t]() {
                std::array<float, 4> query = {
                    rows[static_cast<size_t>((t % n_rows) * kDim + 0)],
                    rows[static_cast<size_t>((t % n_rows) * kDim + 1)],
                    rows[static_cast<size_t>((t % n_rows) * kDim + 2)],
                    rows[static_cast<size_t>((t % n_rows) * kDim + 3)],
                };
                std::array<float, 3> expected_scores{};
                std::array<uint64_t, 3> expected_ids{};
                std::array<float, 3> expected_filtered_scores{};
                std::array<uint64_t, 3> expected_filtered_ids{};
                CHECK(ggml_vec_index_search(
                    concurrent, query.data(), 1, /*k=*/3,
                    expected_scores.data(), expected_ids.data()) == GGML_VEC_INDEX_OK);
                CHECK(ggml_vec_index_search_filtered(
                    concurrent, query.data(), 1, /*k=*/3,
                    allowed.data(), static_cast<int>(allowed.size()),
                    expected_filtered_scores.data(), expected_filtered_ids.data()) == GGML_VEC_INDEX_OK);
                ready.fetch_add(1);
                while (!start.load()) {
                    std::this_thread::yield();
                }
                for (int iter = 0; iter < 200; ++iter) {
                    std::array<float, 3> scores{};
                    std::array<uint64_t, 3> out_ids{};
                    scores.fill(std::numeric_limits<float>::quiet_NaN());
                    out_ids.fill(UINT64_MAX);
                    CHECK(ggml_vec_index_search(
                        concurrent, query.data(), 1, /*k=*/3,
                        scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
                    CHECK(scores == expected_scores);
                    CHECK(out_ids == expected_ids);
                    scores.fill(std::numeric_limits<float>::quiet_NaN());
                    out_ids.fill(UINT64_MAX);
                    CHECK(ggml_vec_index_search_filtered(
                        concurrent, query.data(), 1, /*k=*/3,
                        allowed.data(), static_cast<int>(allowed.size()),
                        scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
                    CHECK(scores == expected_filtered_scores);
                    CHECK(out_ids == expected_filtered_ids);
                    scores.fill(std::numeric_limits<float>::quiet_NaN());
                    out_ids.fill(UINT64_MAX);
                    CHECK(ggml_vec_index_search_prepared_filtered(
                        concurrent, filter, query.data(), 1, /*k=*/3,
                        scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
                    CHECK(scores == expected_filtered_scores);
                    CHECK(out_ids == expected_filtered_ids);
                    scores.fill(std::numeric_limits<float>::quiet_NaN());
                    out_ids.fill(UINT64_MAX);
                    CHECK(ggml_vec_index_search_ivf(
                        concurrent, query.data(), 1, /*k=*/3, /*nprobe=*/4,
                        scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
                    CHECK(scores == expected_scores);
                    CHECK(out_ids == expected_ids);
                    CHECK(ggml_vec_index_contains(concurrent, row_ids[static_cast<size_t>(t % n_rows)]) == 1);
                    CHECK(ggml_vec_index_len(concurrent) == n_rows);
                    CHECK(ggml_vec_index_dim(concurrent) == kDim);
                    CHECK(ggml_vec_index_bit_width(concurrent) == 8);
                }
            });
        }
        while (ready.load() != 8) {
            std::this_thread::yield();
        }
        start.store(true);
        for (std::thread & thread : threads) {
            thread.join();
        }

        ggml_vec_index_filter_free(filter);
        ggml_vec_index_free(concurrent);
    }

    // Mutations are serialized with readers on the same handle.
    {
        constexpr int n_rows = 16;
        std::vector<float> rows;
        std::vector<uint64_t> row_ids;
        rows.reserve(static_cast<size_t>(n_rows) * kDim);
        row_ids.reserve(n_rows);
        for (int row = 0; row < n_rows; ++row) {
            const std::vector<float> v = normalize({
                1.0f,
                static_cast<float>((row % 3) - 1),
                static_cast<float>(((row + 2) % 5) - 2),
                0.5f,
            });
            rows.insert(rows.end(), v.begin(), v.end());
            row_ids.push_back(static_cast<uint64_t>(9000 + row));
        }

        auto * concurrent_mutation = ggml_vec_index_create(kDim, /*bit_width=*/32);
        CHECK(concurrent_mutation != nullptr);
        CHECK(ggml_vec_index_add(
            concurrent_mutation, rows.data(), n_rows, row_ids.data()) ==
            GGML_VEC_INDEX_OK);

        std::atomic<int>         ready{ 0 };
        std::atomic<int>         read_count{ 0 };
        std::atomic<int>         readers_waiting{ 0 };
        std::atomic<int>         post_signal_reads{ 0 };
        std::atomic<bool>        start{ false };
        std::atomic<bool>        writer_pending{ false };
        std::atomic<bool>        race_start{ false };
        std::atomic<bool>        writer_done{ false };
        std::atomic<int>         failures{ 0 };
        std::vector<std::thread> readers;
        for (int t = 0; t < 4; ++t) {
            readers.emplace_back([&, t]() {
                const float *           query = rows.data() + static_cast<size_t>(t % n_rows) * kDim;
                std::array<float, 3>    scores{};
                std::array<uint64_t, 3> out_ids{};
                ready.fetch_add(1);
                while (!start.load()) {
                    std::this_thread::yield();
                }
                while (!writer_pending.load()) {
                    if (ggml_vec_index_search(concurrent_mutation, query, 1, /*k=*/3, scores.data(), out_ids.data()) !=
                        GGML_VEC_INDEX_OK) {
                        failures.fetch_add(1);
                    }
                    if (ggml_vec_index_len(concurrent_mutation) < n_rows) {
                        failures.fetch_add(1);
                    }
                    read_count.fetch_add(1);
                }
                readers_waiting.fetch_add(1);
                while (!race_start.load()) {
                    std::this_thread::yield();
                }
                for (int iter = 0; iter < 64; ++iter) {
                    if (ggml_vec_index_search(concurrent_mutation, query, 1, /*k=*/3, scores.data(), out_ids.data()) !=
                        GGML_VEC_INDEX_OK) {
                        failures.fetch_add(1);
                    }
                    if (ggml_vec_index_len(concurrent_mutation) < n_rows) {
                        failures.fetch_add(1);
                    }
                    post_signal_reads.fetch_add(1);
                    std::this_thread::yield();
                }
            });
        }

        while (ready.load() != 4) {
            std::this_thread::yield();
        }
        start.store(true);
        const auto read_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
        while (read_count.load() < 32 && std::chrono::steady_clock::now() < read_deadline) {
            std::this_thread::yield();
        }
        CHECK(read_count.load() >= 32);

        std::thread writer([&]() {
            writer_pending.store(true);
            while (readers_waiting.load() != 4) {
                std::this_thread::yield();
            }
            race_start.store(true);
            for (int iter = 0; iter < 100; ++iter) {
                const std::vector<float> v  = normalize({
                    0.25f,
                    static_cast<float>((iter % 7) - 3),
                    1.0f,
                    -0.5f,
                });
                const uint64_t           id = static_cast<uint64_t>(10000 + iter);
                if (ggml_vec_index_add(concurrent_mutation, v.data(), 1, &id) != GGML_VEC_INDEX_OK ||
                    ggml_vec_index_remove(concurrent_mutation, id) != GGML_VEC_INDEX_OK) {
                    failures.fetch_add(1);
                    break;
                }
            }
            writer_done.store(true);
        });

        const auto writer_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (!writer_done.load() && std::chrono::steady_clock::now() < writer_deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        CHECK(writer_done.load());
        writer.join();
        for (std::thread & reader : readers) {
            reader.join();
        }
        CHECK(failures.load() == 0);
        CHECK(post_signal_reads.load() == 4 * 64);
        CHECK(ggml_vec_index_len(concurrent_mutation) == n_rows);

        ggml_vec_index_free(concurrent_mutation);
    }

    // Top-1 of querying with each unit vector should retrieve itself with
    // score very close to 1.0 (full f32, no quantization noise).
    {
        std::array<float, 4> scores{};
        std::array<uint64_t, 4> out_ids{};
        for (size_t i = 0; i < seeds.size(); ++i) {
            CHECK(ggml_vec_index_search(
                idx, seeds[i].data(), 1, /*k=*/1,
                scores.data(), out_ids.data()) == 0);
            CHECK(out_ids[0] == ids[i]);
            CHECK(std::fabs(scores[0] - 1.0f) < 1e-5f);
        }
    }

    // Top-k > len returns sentinel-padded tail.
    {
        std::array<float, 8> scores{};
        std::array<uint64_t, 8> out_ids{};
        CHECK(ggml_vec_index_search(
            idx, seeds[0].data(), 1, /*k=*/8,
            scores.data(), out_ids.data()) == 0);
        CHECK(out_ids[0] == ids[0]);
        // Tail entries (positions 4..7) use sentinel score/id values.
        for (int i = 4; i < 8; ++i) {
            CHECK(scores[i] == -FLT_MAX);
            CHECK(out_ids[i] == UINT64_MAX);
        }
    }

    // Equal scores retain the lowest ids when top-k eviction is required.
    {
        auto * tie_idx = ggml_vec_index_create(kDim, /*bit_width=*/32);
        CHECK(tie_idx != nullptr);
        const std::array<uint64_t, 4> tie_ids = { 50, 10, 30, 20 };
        std::vector<float> tie_vecs;
        for (size_t i = 0; i < tie_ids.size(); ++i) {
            tie_vecs.insert(tie_vecs.end(), seeds[0].begin(), seeds[0].end());
        }
        CHECK(ggml_vec_index_add(
            tie_idx, tie_vecs.data(), static_cast<int>(tie_ids.size()), tie_ids.data()) ==
            GGML_VEC_INDEX_OK);
        std::array<float, 2> scores{};
        std::array<uint64_t, 2> out_ids{};
        CHECK(ggml_vec_index_search(
            tie_idx, seeds[0].data(), 1, /*k=*/2,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(out_ids[0] == 10);
        CHECK(out_ids[1] == 20);
        CHECK(scores[0] == scores[1]);
        ggml_vec_index_free(tie_idx);
    }

    // Tombstone-heavy exact search visits live slots without scanning deleted rows.
    {
        auto * tombstone_idx = ggml_vec_index_create(kDim, /*bit_width=*/32);
        CHECK(tombstone_idx != nullptr);
        const std::array<uint64_t, 6> tombstone_ids = { 60, 50, 40, 30, 20, 10 };
        std::vector<float> tombstone_vecs;
        for (size_t i = 0; i < tombstone_ids.size(); ++i) {
            tombstone_vecs.insert(
                tombstone_vecs.end(), seeds[0].begin(), seeds[0].end());
        }
        CHECK(ggml_vec_index_add(
            tombstone_idx,
            tombstone_vecs.data(),
            static_cast<int>(tombstone_ids.size()),
            tombstone_ids.data()) == GGML_VEC_INDEX_OK);
        for (size_t i = 0; i < 4; ++i) {
            CHECK(ggml_vec_index_remove(tombstone_idx, tombstone_ids[i]) == GGML_VEC_INDEX_OK);
        }

        std::array<float, 2> scores{};
        std::array<uint64_t, 2> out_ids{};
        CHECK(ggml_vec_index_search(
            tombstone_idx, seeds[0].data(), 1, /*k=*/2,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(out_ids[0] == 10);
        CHECK(out_ids[1] == 20);
        ggml_vec_index_free(tombstone_idx);
    }

    // Filtered search only considers ids present in the allowlist. Missing
    // and duplicate filter ids do not produce duplicate result rows.
    {
        const uint64_t missing_id = (1ULL << 60) + 99ULL;
        const std::array<uint64_t, 4> allowed = {
            ids[2], missing_id, ids[0], ids[0],
        };
        std::array<float, 3> scores{};
        std::array<uint64_t, 3> out_ids{};
        CHECK(ggml_vec_index_search_filtered(
            idx, seeds[0].data(), 1, /*k=*/3,
            allowed.data(), static_cast<int>(allowed.size()),
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(out_ids[0] == ids[0]);
        CHECK(out_ids[1] == ids[2]);
        CHECK(out_ids[2] == UINT64_MAX);
        CHECK(scores[2] == -FLT_MAX);

        CHECK(ggml_vec_index_search_filtered(
            idx, seeds[0].data(), 1, /*k=*/2,
            nullptr, /*n_allowed=*/0,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(out_ids[0] == UINT64_MAX);
        CHECK(out_ids[1] == UINT64_MAX);
        CHECK(scores[0] == -FLT_MAX);
        CHECK(scores[1] == -FLT_MAX);

        CHECK(ggml_vec_index_search_filtered(
            idx, seeds[0].data(), 1, /*k=*/1,
            nullptr, /*n_allowed=*/1,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);

        auto * filter = ggml_vec_index_filter_create(
            idx, allowed.data(), static_cast<int>(allowed.size()));
        CHECK(filter != nullptr);
        scores = {};
        out_ids = {};
        CHECK(ggml_vec_index_search_prepared_filtered(
            idx, filter, seeds[0].data(), 1, /*k=*/3,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(out_ids[0] == ids[0]);
        CHECK(out_ids[1] == ids[2]);
        CHECK(out_ids[2] == UINT64_MAX);
        CHECK(scores[2] == -FLT_MAX);

        auto * other_idx = ggml_vec_index_create(kDim, /*bit_width=*/32);
        CHECK(other_idx != nullptr);
        CHECK(ggml_vec_index_add(
            other_idx, vecs.data(), static_cast<int>(ids.size()), ids.data()) ==
            GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search_prepared_filtered(
            other_idx, filter, seeds[0].data(), 1, /*k=*/1,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);
        ggml_vec_index_free(other_idx);

        ggml_vec_index_filter_free(filter);

        auto * empty_filter = ggml_vec_index_filter_create(
            idx, nullptr, /*n_allowed=*/0);
        CHECK(empty_filter != nullptr);
        CHECK(ggml_vec_index_search_prepared_filtered(
            idx, empty_filter, seeds[0].data(), 1, /*k=*/2,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(out_ids[0] == UINT64_MAX);
        CHECK(out_ids[1] == UINT64_MAX);
        CHECK(scores[0] == -FLT_MAX);
        CHECK(scores[1] == -FLT_MAX);
        ggml_vec_index_filter_free(empty_filter);

        CHECK(ggml_vec_index_filter_create(
            idx, nullptr, /*n_allowed=*/1) == nullptr);
        CHECK(ggml_vec_index_search_prepared_filtered(
            idx, nullptr, seeds[0].data(), 1, /*k=*/1,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);

        auto * stale_filter = ggml_vec_index_filter_create(
            idx, allowed.data(), static_cast<int>(allowed.size()));
        CHECK(stale_filter != nullptr);
        const uint64_t stale_new_id = (1ULL << 60) + 100ULL;
        CHECK(ggml_vec_index_add(
            idx, seeds[3].data(), 1, &stale_new_id) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search_prepared_filtered(
            idx, stale_filter, seeds[0].data(), 1, /*k=*/1,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_remove(idx, stale_new_id) == GGML_VEC_INDEX_OK);
        ggml_vec_index_filter_free(stale_filter);
    }

    // Remove + search: the removed id must no longer surface.
    {
        CHECK(ggml_vec_index_remove(idx, ids[1]) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_remove(idx, ids[1]) == GGML_VEC_INDEX_E_NOT_FOUND); // already gone
        CHECK(ggml_vec_index_len(idx) == 3);
        CHECK(ggml_vec_index_contains(idx, ids[1]) == 0);

        std::array<float, 3> scores{};
        std::array<uint64_t, 3> out_ids{};
        CHECK(ggml_vec_index_search(
            idx, seeds[1].data(), 1, /*k=*/3,
            scores.data(), out_ids.data()) == 0);
        for (int i = 0; i < 3; ++i) {
            CHECK(out_ids[i] != ids[1]);
        }
    }

    // Persistence round-trip: write, free, load, re-query.
    const auto tmp = std::filesystem::temp_directory_path() /
                     "ggml-vector-index-test.tvim";
    const std::string path = tmp.string();
    CHECK(ggml_vec_index_write(idx, path.c_str()) == 0);
#ifndef _WIN32
    CHECK(::chmod(path.c_str(), 0600) == 0);
    CHECK(ggml_vec_index_write(idx, path.c_str()) == 0);
    struct stat persisted_stat;
    CHECK(::stat(path.c_str(), &persisted_stat) == 0);
    CHECK((persisted_stat.st_mode & 0777) == 0600);
#endif
    {
        const std::filesystem::path missing_parent =
            std::filesystem::temp_directory_path() / "ggml-vector-index-missing-dir";
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
        const std::filesystem::path protected_dir  = test_temp_dir / "protected-dir";
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

    {
        ggml_vec_index_t * diag_loaded = nullptr;
        CHECK(ggml_vec_index_load_ex(path.c_str(), &diag_loaded) == GGML_VEC_INDEX_OK);
        CHECK(diag_loaded != nullptr);
        ggml_vec_index_free(diag_loaded);
        CHECK(ggml_vec_index_load_ex(path.c_str(), nullptr) == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(std::strcmp(
            ggml_vec_index_error_to_string(GGML_VEC_INDEX_E_BAD_MAGIC),
            "unknown error") != 0);

        const std::string bad_magic_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-bad-magic.tvim").string();
        const std::string bad_version_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-bad-version.tvim").string();
        const std::string diag_delta_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-diagnostics.tvid").string();
        std::vector<uint8_t> bytes = read_file_bytes(path);

        CHECK(bytes.size() > 4);
        bytes[0] = 'X';
        write_file_bytes(bad_magic_path, bytes);

        diag_loaded = nullptr;
        CHECK(ggml_vec_index_load_ex(bad_magic_path.c_str(), &diag_loaded) ==
              GGML_VEC_INDEX_E_BAD_MAGIC);
        CHECK(diag_loaded == nullptr);
        CHECK(ggml_vec_index_load_mmap_ex(bad_magic_path.c_str(), &diag_loaded) ==
              GGML_VEC_INDEX_E_BAD_MAGIC);
        CHECK(diag_loaded == nullptr);
        CHECK(ggml_vec_index_load_with_delta_ex(
                  bad_magic_path.c_str(), diag_delta_path.c_str(), &diag_loaded) ==
              GGML_VEC_INDEX_E_BAD_MAGIC);
        CHECK(diag_loaded == nullptr);

        bytes = read_file_bytes(path);
        CHECK(bytes.size() > 4);
        bytes[4] = 99;
        write_file_bytes(bad_version_path, bytes);

        diag_loaded = nullptr;
        CHECK(ggml_vec_index_load_ex(bad_version_path.c_str(), &diag_loaded) ==
              GGML_VEC_INDEX_E_BAD_VERSION);
        CHECK(diag_loaded == nullptr);
        CHECK(ggml_vec_index_load_mmap_ex(bad_version_path.c_str(), &diag_loaded) ==
              GGML_VEC_INDEX_E_BAD_VERSION);
        CHECK(diag_loaded == nullptr);

        std::filesystem::remove(bad_magic_path);
        std::filesystem::remove(bad_version_path);
        std::filesystem::remove(diag_delta_path);
    }

    const std::string reserved_id_path =
        (std::filesystem::temp_directory_path() /
         "ggml-vector-index-reserved-id-corrupt.tvim").string();
    expect_corrupt_load_fails(path, reserved_id_path, [](std::vector<uint8_t> & bytes) {
        constexpr size_t ids_offset = 32 + 3 * kDim * sizeof(float);
        CHECK(bytes.size() >= ids_offset + sizeof(uint64_t));
        for (size_t i = 0; i < sizeof(uint64_t); ++i) {
            bytes[ids_offset + i] = 0xff;
        }
    });

    ggml_vec_index_free(idx);

    auto * loaded = ggml_vec_index_load(path.c_str());
    CHECK(loaded != nullptr);
    CHECK(ggml_vec_index_dim(loaded) == kDim);
    CHECK(ggml_vec_index_len(loaded) == 3);
    CHECK(ggml_vec_index_bit_width(loaded) == 32);
    CHECK(ggml_vec_index_contains(loaded, ids[0]) == 1);
    CHECK(ggml_vec_index_contains(loaded, ids[1]) == 0); // stayed deleted
    CHECK(ggml_vec_index_contains(loaded, ids[2]) == 1);
    CHECK(ggml_vec_index_contains(loaded, ids[3]) == 1);

    // Top-1 self-match after reload.
    {
        std::array<float, 1> scores{};
        std::array<uint64_t, 1> out_ids{};
        CHECK(ggml_vec_index_search(
            loaded, seeds[0].data(), 1, /*k=*/1,
            scores.data(), out_ids.data()) == 0);
        CHECK(out_ids[0] == ids[0]);
        CHECK(std::fabs(scores[0] - 1.0f) < 1e-5f);
    }

    ggml_vec_index_free(loaded);
    std::filesystem::remove(path);

    // Tombstone removal: later adds append, searches skip deleted slots, IVF
    // rebuilds exclude them, and snapshots write only live rows.
    {
        for (int bit_width : { 32, 8, 4 }) {
            const std::string tombstone_path =
                (std::filesystem::temp_directory_path() /
                 ("ggml-vector-index-tombstone-" + std::to_string(bit_width) + ".tvim")).string();
            std::filesystem::remove(tombstone_path);

            auto * tombstone_idx = ggml_vec_index_create(kDim, bit_width);
            CHECK(tombstone_idx != nullptr);
            CHECK(ggml_vec_index_add(
                tombstone_idx, vecs.data(), static_cast<int>(ids.size()), ids.data()) ==
                GGML_VEC_INDEX_OK);
            CHECK(ggml_vec_index_remove(tombstone_idx, ids[1]) == GGML_VEC_INDEX_OK);
            CHECK(ggml_vec_index_len(tombstone_idx) == 3);

            const uint64_t appended_id =
                (1ULL << 40) + 123ULL + static_cast<uint64_t>(bit_width);
            CHECK(ggml_vec_index_add(
                tombstone_idx, seeds[1].data(), 1, &appended_id) == GGML_VEC_INDEX_OK);
            CHECK(ggml_vec_index_len(tombstone_idx) == 4);
            CHECK(ggml_vec_index_contains(tombstone_idx, ids[1]) == 0);
            CHECK(ggml_vec_index_contains(tombstone_idx, appended_id) == 1);

            std::array<float, 4> scores{};
            std::array<uint64_t, 4> out_ids{};
            CHECK(ggml_vec_index_search(
                tombstone_idx, seeds[1].data(), 1, /*k=*/4,
                scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
            for (uint64_t result_id : out_ids) {
                CHECK(result_id != ids[1]);
            }
            CHECK(out_ids[0] == appended_id);

            const std::array<uint64_t, 2> allowed = { ids[1], appended_id };
            scores = {};
            out_ids = {};
            CHECK(ggml_vec_index_search_filtered(
                tombstone_idx, seeds[1].data(), 1, /*k=*/2,
                allowed.data(), static_cast<int>(allowed.size()),
                scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
            CHECK(out_ids[0] == appended_id);
            CHECK(out_ids[1] == UINT64_MAX);

            auto * filter = ggml_vec_index_filter_create(
                tombstone_idx, allowed.data(), static_cast<int>(allowed.size()));
            CHECK(filter != nullptr);
            scores = {};
            out_ids = {};
            CHECK(ggml_vec_index_search_prepared_filtered(
                tombstone_idx, filter, seeds[1].data(), 1, /*k=*/2,
                scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
            CHECK(out_ids[0] == appended_id);
            CHECK(out_ids[1] == UINT64_MAX);
            ggml_vec_index_filter_free(filter);

            CHECK(ggml_vec_index_build_ivf(tombstone_idx, /*n_lists=*/8, /*n_iter=*/2) ==
                  GGML_VEC_INDEX_OK);
            scores = {};
            out_ids = {};
            CHECK(ggml_vec_index_search_ivf(
                tombstone_idx, seeds[1].data(), 1, /*k=*/4, /*nprobe=*/8,
                scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
            CHECK(out_ids[0] == appended_id);
            for (uint64_t result_id : out_ids) {
                CHECK(result_id != ids[1]);
            }

            auto * stale_filter = ggml_vec_index_filter_create(
                tombstone_idx, allowed.data(), static_cast<int>(allowed.size()));
            CHECK(stale_filter != nullptr);
            CHECK(ggml_vec_index_compact(tombstone_idx) == GGML_VEC_INDEX_OK);
            CHECK(ggml_vec_index_len(tombstone_idx) == 4);
            CHECK(ggml_vec_index_contains(tombstone_idx, ids[1]) == 0);
            CHECK(ggml_vec_index_contains(tombstone_idx, appended_id) == 1);
            CHECK(ggml_vec_index_search_prepared_filtered(
                tombstone_idx, stale_filter, seeds[1].data(), 1, /*k=*/2,
                scores.data(), out_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);
            ggml_vec_index_filter_free(stale_filter);
            CHECK(ggml_vec_index_search_ivf(
                tombstone_idx, seeds[1].data(), 1, /*k=*/1, /*nprobe=*/1,
                scores.data(), out_ids.data()) == GGML_VEC_INDEX_E_INVALID_ARG);

            scores = {};
            out_ids = {};
            CHECK(ggml_vec_index_search_filtered(
                tombstone_idx, seeds[1].data(), 1, /*k=*/2,
                allowed.data(), static_cast<int>(allowed.size()),
                scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
            CHECK(out_ids[0] == appended_id);
            CHECK(out_ids[1] == UINT64_MAX);
            CHECK(ggml_vec_index_build_ivf(tombstone_idx, /*n_lists=*/8, /*n_iter=*/2) ==
                  GGML_VEC_INDEX_OK);
            scores = {};
            out_ids = {};
            CHECK(ggml_vec_index_search_ivf(
                tombstone_idx, seeds[1].data(), 1, /*k=*/4, /*nprobe=*/8,
                scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
            CHECK(out_ids[0] == appended_id);
            for (uint64_t result_id : out_ids) {
                CHECK(result_id != ids[1]);
            }

            CHECK(ggml_vec_index_write(tombstone_idx, tombstone_path.c_str()) ==
                  GGML_VEC_INDEX_OK);
            ggml_vec_index_free(tombstone_idx);

            auto * tombstone_loaded = ggml_vec_index_load(tombstone_path.c_str());
            CHECK(tombstone_loaded != nullptr);
            CHECK(ggml_vec_index_bit_width(tombstone_loaded) == bit_width);
            CHECK(ggml_vec_index_len(tombstone_loaded) == 4);
            CHECK(ggml_vec_index_contains(tombstone_loaded, ids[1]) == 0);
            CHECK(ggml_vec_index_contains(tombstone_loaded, appended_id) == 1);
            scores = {};
            out_ids = {};
            CHECK(ggml_vec_index_search(
                tombstone_loaded, seeds[1].data(), 1, /*k=*/4,
                scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
            CHECK(out_ids[0] == appended_id);
            for (uint64_t result_id : out_ids) {
                CHECK(result_id != ids[1]);
            }
            ggml_vec_index_free(tombstone_loaded);
            std::filesystem::remove(tombstone_path);
        }
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
    check_writer_completes_after_read_admission_closes();
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

    check_quantized_partial_replay_rollback(4);
    check_quantized_partial_replay_rollback(8);
    check_delta_log_alternating_writers();

    // Incremental persistence: replay add/remove deltas on top of a snapshot.
    {
        const std::string snapshot_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-delta-base.tvim").string();
        const std::string delta_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-delta-log.tvid").string();
        const std::string missing_delta_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-missing-delta-log.tvid").string();
        const std::string mismatched_snapshot_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-delta-mismatch.tvim").string();
        const std::string corrupt_delta_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-delta-corrupt.tvid").string();
        const std::string other_delta_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-delta-other.tvid").string();
        const std::string diverged_delta_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-delta-diverged.tvid").string();
        const std::string delta_bound_write_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-delta-bound-write.tvim").string();
        const std::string replay_failure_write_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-delta-replay-failure-write.tvim").string();
        const std::string alternate_delta_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-delta-alternate.tvid").string();
        std::filesystem::remove(snapshot_path);
        std::filesystem::remove(delta_path);
        std::filesystem::remove(missing_delta_path);
        std::filesystem::remove(mismatched_snapshot_path);
        std::filesystem::remove(corrupt_delta_path);
        std::filesystem::remove(other_delta_path);
        std::filesystem::remove(other_delta_path + ".lock");
        std::filesystem::remove(diverged_delta_path);
        std::filesystem::remove(diverged_delta_path + ".lock");
        std::filesystem::remove(delta_bound_write_path);
        std::filesystem::remove(replay_failure_write_path);
        std::filesystem::remove(alternate_delta_path);
        std::filesystem::remove(alternate_delta_path + ".lock");

        auto * base = ggml_vec_index_create(kDim, /*bit_width=*/32);
        CHECK(base != nullptr);
        std::vector<float> base_vecs;
        base_vecs.insert(base_vecs.end(), seeds[0].begin(), seeds[0].end());
        base_vecs.insert(base_vecs.end(), seeds[1].begin(), seeds[1].end());
        CHECK(ggml_vec_index_add(base, base_vecs.data(), 2, ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_write(base, snapshot_path.c_str()) == GGML_VEC_INDEX_OK);

        auto * diverged = ggml_vec_index_load(snapshot_path.c_str());
        CHECK(diverged != nullptr);
        const uint64_t diverged_plain_id = (1ULL << 41) + 4ULL;
        const uint64_t diverged_logged_id = (1ULL << 41) + 5ULL;
        CHECK(ggml_vec_index_add(
            diverged, seeds[2].data(), 1, &diverged_plain_id) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_add_logged(
            diverged,
            seeds[3].data(),
            1,
            &diverged_logged_id,
            diverged_delta_path.c_str()) == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_contains(diverged, diverged_plain_id) == 1);
        CHECK(ggml_vec_index_contains(diverged, diverged_logged_id) == 0);
        CHECK(!std::filesystem::exists(diverged_delta_path));
        ggml_vec_index_free(diverged);

        auto * base_only = ggml_vec_index_load_with_delta(
            snapshot_path.c_str(), missing_delta_path.c_str());
        CHECK(base_only != nullptr);
        CHECK(ggml_vec_index_len(base_only) == 2);
        const uint64_t plain_delta_bound_id = (1ULL << 41) + 6ULL;
        CHECK(ggml_vec_index_add(
            base_only, seeds[2].data(), 1, &plain_delta_bound_id) ==
            GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_remove(base_only, ids[0]) == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_compact(base_only) == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_write(base_only, delta_bound_write_path.c_str()) ==
              GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(!std::filesystem::exists(delta_bound_write_path));
        ggml_vec_index_free(base_only);

        const uint64_t reserved_delta_id = UINT64_MAX;
        CHECK(ggml_vec_index_add_logged(
            base, seeds[2].data(), 1, &reserved_delta_id, delta_path.c_str()) ==
            GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_remove_logged(
            base, reserved_delta_id, delta_path.c_str()) ==
            GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_len(base) == 2);
        CHECK(!std::filesystem::exists(delta_path));

        const uint64_t delta_id = (1ULL << 41) + 7ULL;
        CHECK(ggml_vec_index_add_logged(
            base, seeds[2].data(), 1, &delta_id, delta_path.c_str()) == GGML_VEC_INDEX_OK);
        const uint64_t wrong_delta_id = (1ULL << 41) + 12ULL;
        CHECK(ggml_vec_index_add_logged(
            base, seeds[3].data(), 1, &wrong_delta_id, other_delta_path.c_str()) ==
            GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_contains(base, wrong_delta_id) == 0);
        CHECK(ggml_vec_index_remove_logged(
            base, ids[1], other_delta_path.c_str()) == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_contains(base, ids[1]) == 1);
        CHECK(!std::filesystem::exists(other_delta_path));
        CHECK(ggml_vec_index_remove_logged(
            base, ids[0], delta_path.c_str()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_add_logged(
            base, seeds[2].data(), 1, &delta_id, delta_path.c_str()) ==
            GGML_VEC_INDEX_E_DUPLICATE);
        CHECK(ggml_vec_index_write(base, mismatched_snapshot_path.c_str()) ==
              GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(!std::filesystem::exists(mismatched_snapshot_path));
        const uint64_t alternate_delta_id = (1ULL << 41) + 8ULL;
        CHECK(ggml_vec_index_add_logged(
            base, seeds[3].data(), 1, &alternate_delta_id, alternate_delta_path.c_str()) ==
            GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_contains(base, alternate_delta_id) == 0);
        CHECK(!std::filesystem::exists(alternate_delta_path));
        CHECK(ggml_vec_index_compact_delta(
            base, mismatched_snapshot_path.c_str(), alternate_delta_path.c_str()) ==
            GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(!std::filesystem::exists(mismatched_snapshot_path));

        auto * replayed = ggml_vec_index_load_with_delta(
            snapshot_path.c_str(), delta_path.c_str());
        CHECK(replayed != nullptr);
        CHECK(ggml_vec_index_len(replayed) == 2);
        CHECK(ggml_vec_index_contains(replayed, ids[0]) == 0);
        CHECK(ggml_vec_index_contains(replayed, ids[1]) == 1);
        CHECK(ggml_vec_index_contains(replayed, delta_id) == 1);
        CHECK(ggml_vec_index_add_logged(
            replayed,
            seeds[3].data(),
            1,
            &alternate_delta_id,
            alternate_delta_path.c_str()) == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_contains(replayed, alternate_delta_id) == 0);

        std::array<float, 2> scores{};
        std::array<uint64_t, 2> out_ids{};
        CHECK(ggml_vec_index_search(
            replayed, seeds[2].data(), 1, /*k=*/2,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(out_ids[0] == delta_id);

        std::vector<uint8_t> corrupt_delta = read_file_bytes(delta_path);
        const size_t first_record_offset = delta_log_header_size(corrupt_delta);
        corrupt_delta[first_record_offset + 16] ^= 1;
        write_file_bytes(corrupt_delta_path, corrupt_delta);

        auto * corrupt_stale_writer = ggml_vec_index_load(snapshot_path.c_str());
        CHECK(corrupt_stale_writer != nullptr);
        const uint64_t corrupt_stale_id = (1ULL << 41) + 8ULL;
        CHECK(ggml_vec_index_add_logged(
            corrupt_stale_writer,
            seeds[3].data(),
            1,
            &corrupt_stale_id,
            corrupt_delta_path.c_str()) == GGML_VEC_INDEX_E_IO);
        CHECK(ggml_vec_index_len(corrupt_stale_writer) == 2);
        CHECK(ggml_vec_index_contains(corrupt_stale_writer, delta_id) == 0);
        CHECK(ggml_vec_index_contains(corrupt_stale_writer, corrupt_stale_id) == 0);
        CHECK(ggml_vec_index_contains(corrupt_stale_writer, ids[0]) == 1);
        CHECK(ggml_vec_index_contains(corrupt_stale_writer, ids[1]) == 1);
        CHECK(ggml_vec_index_write(corrupt_stale_writer, replay_failure_write_path.c_str()) ==
              GGML_VEC_INDEX_OK);
        CHECK(read_file_bytes(replay_failure_write_path) == read_file_bytes(snapshot_path));
        std::filesystem::remove(replay_failure_write_path);
        const uint64_t replay_failure_plain_id = (1ULL << 41) + 13ULL;
        CHECK(ggml_vec_index_add(
            corrupt_stale_writer,
            seeds[3].data(),
            1,
            &replay_failure_plain_id) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_contains(corrupt_stale_writer, replay_failure_plain_id) == 1);
        CHECK(ggml_vec_index_write(corrupt_stale_writer, replay_failure_write_path.c_str()) ==
              GGML_VEC_INDEX_OK);
        CHECK(std::filesystem::exists(replay_failure_write_path));
        std::filesystem::remove(replay_failure_write_path);
        ggml_vec_index_free(corrupt_stale_writer);

        auto * corrupt_delta_loaded = ggml_vec_index_load_with_delta(
            snapshot_path.c_str(), corrupt_delta_path.c_str());
        CHECK(corrupt_delta_loaded == nullptr);
        ggml_vec_index_free(corrupt_delta_loaded);

        write_file_bytes(corrupt_delta_path, corrupt_delta);
        auto * retry_stale_writer = ggml_vec_index_load(snapshot_path.c_str());
        CHECK(retry_stale_writer != nullptr);
        const uint64_t retry_stale_id = (1ULL << 41) + 15ULL;
        CHECK(ggml_vec_index_add_logged(
            retry_stale_writer,
            seeds[3].data(),
            1,
            &retry_stale_id,
            corrupt_delta_path.c_str()) == GGML_VEC_INDEX_E_IO);
        write_file_bytes(corrupt_delta_path, read_file_bytes(delta_path));
        CHECK(ggml_vec_index_add_logged(
            retry_stale_writer,
            seeds[3].data(),
            1,
            &retry_stale_id,
            corrupt_delta_path.c_str()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_contains(retry_stale_writer, delta_id) == 1);
        CHECK(ggml_vec_index_contains(retry_stale_writer, retry_stale_id) == 1);
        ggml_vec_index_free(retry_stale_writer);

        corrupt_delta_loaded = ggml_vec_index_load_with_delta(
            snapshot_path.c_str(), corrupt_delta_path.c_str());
        CHECK(corrupt_delta_loaded != nullptr);
        CHECK(ggml_vec_index_contains(corrupt_delta_loaded, retry_stale_id) == 1);
        ggml_vec_index_free(corrupt_delta_loaded);

        std::vector<uint8_t> corrupt_middle_crc_delta = read_file_bytes(corrupt_delta_path);
        const size_t middle_second_record_offset =
            first_record_offset +
            delta_record_header_size(corrupt_middle_crc_delta) +
            static_cast<size_t>(read_u64_le_at(corrupt_middle_crc_delta, first_record_offset + 8));
        const size_t middle_third_record_offset =
            middle_second_record_offset +
            delta_record_header_size(corrupt_middle_crc_delta) +
            static_cast<size_t>(read_u64_le_at(corrupt_middle_crc_delta, middle_second_record_offset + 8));
        CHECK(middle_second_record_offset + 16 < corrupt_middle_crc_delta.size());
        CHECK(middle_third_record_offset + delta_record_header_size(corrupt_middle_crc_delta) <=
              corrupt_middle_crc_delta.size());
        corrupt_middle_crc_delta[middle_second_record_offset + 16] ^= 1;
        write_file_bytes(corrupt_delta_path, corrupt_middle_crc_delta);
        auto * middle_crc_stale_writer = ggml_vec_index_load(snapshot_path.c_str());
        CHECK(middle_crc_stale_writer != nullptr);
        const uint64_t middle_crc_stale_id = (1ULL << 41) + 16ULL;
        CHECK(ggml_vec_index_add_logged(
            middle_crc_stale_writer,
            seeds[3].data(),
            1,
            &middle_crc_stale_id,
            corrupt_delta_path.c_str()) == GGML_VEC_INDEX_E_IO);
        CHECK(ggml_vec_index_len(middle_crc_stale_writer) == 2);
        CHECK(ggml_vec_index_contains(middle_crc_stale_writer, delta_id) == 0);
        CHECK(ggml_vec_index_contains(middle_crc_stale_writer, retry_stale_id) == 0);
        CHECK(ggml_vec_index_contains(middle_crc_stale_writer, middle_crc_stale_id) == 0);
        CHECK(ggml_vec_index_contains(middle_crc_stale_writer, ids[0]) == 1);
        CHECK(ggml_vec_index_contains(middle_crc_stale_writer, ids[1]) == 1);
        CHECK(ggml_vec_index_write(middle_crc_stale_writer, replay_failure_write_path.c_str()) ==
              GGML_VEC_INDEX_OK);
        CHECK(read_file_bytes(replay_failure_write_path) == read_file_bytes(snapshot_path));
        std::filesystem::remove(replay_failure_write_path);
        ggml_vec_index_free(middle_crc_stale_writer);

        std::vector<uint8_t> corrupt_second_record_delta = read_file_bytes(delta_path);
        const size_t second_record_offset =
            first_record_offset +
            delta_record_header_size(corrupt_second_record_delta) +
            static_cast<size_t>(read_u64_le_at(corrupt_second_record_delta, first_record_offset + 8));
        CHECK(second_record_offset + 16 < corrupt_second_record_delta.size());
        corrupt_second_record_delta[
            delta_record_state_offset(corrupt_second_record_delta, second_record_offset)] ^= 1;
        refresh_delta_record_crc(corrupt_second_record_delta, second_record_offset);
        write_file_bytes(corrupt_delta_path, corrupt_second_record_delta);
        auto * partial_stale_writer = ggml_vec_index_load(snapshot_path.c_str());
        CHECK(partial_stale_writer != nullptr);
        const uint64_t partial_stale_id = (1ULL << 41) + 14ULL;
        CHECK(ggml_vec_index_add_logged(
            partial_stale_writer,
            seeds[3].data(),
            1,
            &partial_stale_id,
            corrupt_delta_path.c_str()) == GGML_VEC_INDEX_E_IO);
        CHECK(ggml_vec_index_len(partial_stale_writer) == 2);
        CHECK(ggml_vec_index_contains(partial_stale_writer, delta_id) == 0);
        CHECK(ggml_vec_index_contains(partial_stale_writer, partial_stale_id) == 0);
        CHECK(ggml_vec_index_contains(partial_stale_writer, ids[0]) == 1);
        CHECK(ggml_vec_index_contains(partial_stale_writer, ids[1]) == 1);
        CHECK(ggml_vec_index_write(partial_stale_writer, replay_failure_write_path.c_str()) ==
              GGML_VEC_INDEX_OK);
        CHECK(read_file_bytes(replay_failure_write_path) == read_file_bytes(snapshot_path));
        std::filesystem::remove(replay_failure_write_path);
        CHECK(ggml_vec_index_add(
            partial_stale_writer,
            seeds[3].data(),
            1,
            &partial_stale_id) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_write(partial_stale_writer, replay_failure_write_path.c_str()) ==
              GGML_VEC_INDEX_OK);
        CHECK(std::filesystem::exists(replay_failure_write_path));
        std::filesystem::remove(replay_failure_write_path);
        ggml_vec_index_free(partial_stale_writer);

        std::vector<uint8_t> corrupt_payload_size_delta = read_file_bytes(delta_path);
        const size_t corrupt_payload_size_record_offset =
            delta_log_header_size(corrupt_payload_size_delta);
        const uint64_t declared_payload_size =
            read_u64_le_at(corrupt_payload_size_delta, corrupt_payload_size_record_offset + 8);
        write_u64_le_at(
            corrupt_payload_size_delta,
            corrupt_payload_size_record_offset + 8,
            declared_payload_size + 1);
        write_file_bytes(corrupt_delta_path, corrupt_payload_size_delta);
        auto * corrupt_payload_size_loaded = ggml_vec_index_load_with_delta(
            snapshot_path.c_str(), corrupt_delta_path.c_str());
        CHECK(corrupt_payload_size_loaded == nullptr);
        ggml_vec_index_free(corrupt_payload_size_loaded);

        std::vector<uint8_t> forged_intermediate_delta = read_file_bytes(delta_path);
        const size_t forged_first_record_offset =
            delta_log_header_size(forged_intermediate_delta);
        forged_intermediate_delta[
            delta_record_state_offset(forged_intermediate_delta, forged_first_record_offset)] ^= 1;
        refresh_delta_record_crc(forged_intermediate_delta, forged_first_record_offset);
        write_file_bytes(corrupt_delta_path, forged_intermediate_delta);
        auto * forged_intermediate_loaded = ggml_vec_index_load_with_delta(
            snapshot_path.c_str(), corrupt_delta_path.c_str());
        CHECK(forged_intermediate_loaded == nullptr);
        ggml_vec_index_free(forged_intermediate_loaded);

        auto * mismatch = ggml_vec_index_create(kDim, /*bit_width=*/32);
        CHECK(mismatch != nullptr);
        std::vector<float> mismatch_vecs;
        mismatch_vecs.insert(mismatch_vecs.end(), seeds[0].begin(), seeds[0].end());
        mismatch_vecs.insert(mismatch_vecs.end(), seeds[3].begin(), seeds[3].end());
        CHECK(ggml_vec_index_add(
            mismatch, mismatch_vecs.data(), 2, ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_write(
            mismatch, mismatched_snapshot_path.c_str()) == GGML_VEC_INDEX_OK);
        auto * mismatch_loaded = ggml_vec_index_load_with_delta(
            mismatched_snapshot_path.c_str(), delta_path.c_str());
        CHECK(mismatch_loaded == nullptr);
        ggml_vec_index_free(mismatch_loaded);
        const uint64_t mismatch_new_id = (1ULL << 41) + 9ULL;
        CHECK(ggml_vec_index_add_logged(
            mismatch, seeds[2].data(), 1, &mismatch_new_id, delta_path.c_str()) ==
            GGML_VEC_INDEX_E_IO);
        CHECK(ggml_vec_index_contains(mismatch, mismatch_new_id) == 0);
        ggml_vec_index_free(mismatch);

        const uint64_t missing_remove_id = (1ULL << 41) + 123ULL;
        const std::string missing_remove_delta_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-missing-remove-delta.tvid").string();
        uint64_t hash_xor = 0;
        uint64_t hash_sum = 0;
        uint64_t hash_sum_rot = 0;
        for (size_t i = 0; i < 2; ++i) {
            const uint64_t hash = slot_state_hash_f32(ids[i], seeds[i]);
            hash_xor ^= hash;
            hash_sum += hash;
            hash_sum_rot += rotl64(hash, 17);
        }
        const uint32_t base_token =
            f32_state_token(kDim, 2, hash_xor, hash_sum, hash_sum_rot);
        // Claim an unchanged state so this only fails if replay rejects the
        // missing remove, not because final state validation catches it later.
        const uint32_t claimed_post_remove_token = base_token;
        std::vector<uint8_t> missing_remove_log = {
            'T', 'V', 'D', 'L',
            2, 32, 0, 0,
        };
        append_u32_le(missing_remove_log, kDim);
        append_u32_le(missing_remove_log, base_token);
        const size_t record_offset = missing_remove_log.size();
        missing_remove_log.push_back(2); // remove
        missing_remove_log.insert(missing_remove_log.end(), { 0, 0, 0 });
        append_u32_le(missing_remove_log, 1);
        append_u64_le(missing_remove_log, sizeof(uint64_t));
        append_u32_le(missing_remove_log, 0); // record CRC placeholder
        append_u32_le(missing_remove_log, claimed_post_remove_token);
        append_u64_le(missing_remove_log, missing_remove_id);
        uint32_t record_crc = crc32c_update(
            0xffffffffu,
            missing_remove_log.data() + record_offset,
            16);
        record_crc = crc32c_update_u32(record_crc, claimed_post_remove_token);
        record_crc = crc32c_update_u64(record_crc, missing_remove_id);
        record_crc ^= 0xffffffffu;
        for (int i = 0; i < 4; ++i) {
            missing_remove_log[record_offset + 16 + static_cast<size_t>(i)] =
                static_cast<uint8_t>(record_crc >> (8 * i));
        }
        write_file_bytes(missing_remove_delta_path, missing_remove_log);
        auto * missing_remove_loaded = ggml_vec_index_load_with_delta(
            snapshot_path.c_str(), missing_remove_delta_path.c_str());
        CHECK(missing_remove_loaded == nullptr);
        ggml_vec_index_free(missing_remove_loaded);
        auto * missing_remove_writer = ggml_vec_index_load(snapshot_path.c_str());
        CHECK(missing_remove_writer != nullptr);
        const uint64_t after_missing_remove_id = (1ULL << 41) + 124ULL;
        CHECK(ggml_vec_index_add_logged(
            missing_remove_writer,
            seeds[2].data(),
            1,
            &after_missing_remove_id,
            missing_remove_delta_path.c_str()) == GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(ggml_vec_index_contains(missing_remove_writer, after_missing_remove_id) == 0);
        ggml_vec_index_free(missing_remove_writer);
        std::filesystem::remove(missing_remove_delta_path);
        std::filesystem::remove(missing_remove_delta_path + ".lock");

        const std::vector<uint8_t> pre_compact_delta = read_file_bytes(delta_path);
        const std::vector<uint8_t> pre_same_path_snapshot = read_file_bytes(snapshot_path);
        CHECK(ggml_vec_index_compact_delta(
            base, snapshot_path.c_str(), snapshot_path.c_str()) ==
            GGML_VEC_INDEX_E_INVALID_ARG);
        CHECK(read_file_bytes(snapshot_path) == pre_same_path_snapshot);
        auto * same_path_snapshot = ggml_vec_index_load(snapshot_path.c_str());
        CHECK(same_path_snapshot != nullptr);
        CHECK(ggml_vec_index_len(same_path_snapshot) == 2);
        CHECK(ggml_vec_index_contains(same_path_snapshot, ids[0]) == 1);
        CHECK(ggml_vec_index_contains(same_path_snapshot, delta_id) == 0);
        ggml_vec_index_free(same_path_snapshot);

        CHECK(ggml_vec_index_compact_delta(
            base, snapshot_path.c_str(), delta_path.c_str()) == GGML_VEC_INDEX_OK);
        CHECK(std::filesystem::file_size(delta_path) == 48);

        auto * compacted = ggml_vec_index_load_with_delta(
            snapshot_path.c_str(), delta_path.c_str());
        CHECK(compacted != nullptr);
        CHECK(ggml_vec_index_len(compacted) == 2);
        CHECK(ggml_vec_index_contains(compacted, ids[0]) == 0);
        CHECK(ggml_vec_index_contains(compacted, ids[1]) == 1);
        CHECK(ggml_vec_index_contains(compacted, delta_id) == 1);
        ggml_vec_index_free(compacted);

        // Crash window: the compacted snapshot is durable but the old log
        // survived. Replay must remain idempotent.
        write_file_bytes(delta_path, pre_compact_delta);
        auto * compacted_with_old_log = ggml_vec_index_load_with_delta(
            snapshot_path.c_str(), delta_path.c_str());
        CHECK(compacted_with_old_log != nullptr);
        CHECK(ggml_vec_index_len(compacted_with_old_log) == 2);
        CHECK(ggml_vec_index_contains(compacted_with_old_log, ids[0]) == 0);
        CHECK(ggml_vec_index_contains(compacted_with_old_log, ids[1]) == 1);
        CHECK(ggml_vec_index_contains(compacted_with_old_log, delta_id) == 1);

        const uint64_t post_crash_compact_id = (1ULL << 41) + 10ULL;
        CHECK(ggml_vec_index_add_logged(
            compacted_with_old_log, seeds[3].data(), 1, &post_crash_compact_id, delta_path.c_str()) ==
            GGML_VEC_INDEX_OK);
        CHECK(std::filesystem::file_size(delta_path) < pre_compact_delta.size());
        auto * replayed_after_old_log_append = ggml_vec_index_load_with_delta(
            snapshot_path.c_str(), delta_path.c_str());
        CHECK(replayed_after_old_log_append != nullptr);
        CHECK(ggml_vec_index_len(replayed_after_old_log_append) == 3);
        CHECK(ggml_vec_index_contains(replayed_after_old_log_append, ids[0]) == 0);
        CHECK(ggml_vec_index_contains(replayed_after_old_log_append, ids[1]) == 1);
        CHECK(ggml_vec_index_contains(replayed_after_old_log_append, delta_id) == 1);
        CHECK(ggml_vec_index_contains(replayed_after_old_log_append, post_crash_compact_id) == 1);
        ggml_vec_index_free(replayed_after_old_log_append);
        ggml_vec_index_free(compacted_with_old_log);

        write_file_bytes(delta_path, pre_compact_delta);
        auto * recompact_from_old_log = ggml_vec_index_load_with_delta(
            snapshot_path.c_str(), delta_path.c_str());
        CHECK(recompact_from_old_log != nullptr);
        CHECK(ggml_vec_index_compact_delta(
            recompact_from_old_log, snapshot_path.c_str(), delta_path.c_str()) ==
            GGML_VEC_INDEX_OK);
        CHECK(std::filesystem::file_size(delta_path) == 48);
        const uint64_t post_recompact_id = (1ULL << 41) + 11ULL;
        CHECK(ggml_vec_index_add_logged(
            recompact_from_old_log, seeds[3].data(), 1, &post_recompact_id, delta_path.c_str()) ==
            GGML_VEC_INDEX_OK);
        auto * replayed_after_recompact = ggml_vec_index_load_with_delta(
            snapshot_path.c_str(), delta_path.c_str());
        CHECK(replayed_after_recompact != nullptr);
        CHECK(ggml_vec_index_len(replayed_after_recompact) == 3);
        CHECK(ggml_vec_index_contains(replayed_after_recompact, post_recompact_id) == 1);
        ggml_vec_index_free(replayed_after_recompact);
        ggml_vec_index_free(recompact_from_old_log);

        CHECK(ggml_vec_index_compact_delta(
            base, snapshot_path.c_str(), delta_path.c_str()) == GGML_VEC_INDEX_E_IO);
        auto * current_after_recompact = ggml_vec_index_load_with_delta(
            snapshot_path.c_str(), delta_path.c_str());
        CHECK(current_after_recompact != nullptr);
        CHECK(ggml_vec_index_contains(current_after_recompact, post_recompact_id) == 1);
        const uint64_t post_compact_id = (1ULL << 41) + 8ULL;
        CHECK(ggml_vec_index_add_logged(
            current_after_recompact, seeds[3].data(), 1, &post_compact_id, delta_path.c_str()) ==
            GGML_VEC_INDEX_OK);
        auto * replayed_after_compact = ggml_vec_index_load_with_delta(
            snapshot_path.c_str(), delta_path.c_str());
        CHECK(replayed_after_compact != nullptr);
        CHECK(ggml_vec_index_len(replayed_after_compact) == 4);
        CHECK(ggml_vec_index_contains(replayed_after_compact, post_recompact_id) == 1);
        CHECK(ggml_vec_index_contains(replayed_after_compact, post_compact_id) == 1);
        ggml_vec_index_free(replayed_after_compact);
        ggml_vec_index_free(current_after_recompact);

        append_file_bytes(delta_path, { 0x01, 0x00, 0x00 });
        auto * replayed_with_torn_tail = ggml_vec_index_load_with_delta(
            snapshot_path.c_str(), delta_path.c_str());
        CHECK(replayed_with_torn_tail != nullptr);
        CHECK(ggml_vec_index_contains(replayed_with_torn_tail, delta_id) == 1);
        CHECK(ggml_vec_index_contains(replayed_with_torn_tail, post_compact_id) == 1);
        const uint64_t post_torn_id = (1ULL << 41) + 13ULL;
        CHECK(ggml_vec_index_add_logged(
            replayed_with_torn_tail,
            seeds[0].data(),
            1,
            &post_torn_id,
            delta_path.c_str()) == GGML_VEC_INDEX_OK);
        auto * replayed_after_torn_append = ggml_vec_index_load_with_delta(
            snapshot_path.c_str(), delta_path.c_str());
        CHECK(replayed_after_torn_append != nullptr);
        CHECK(ggml_vec_index_contains(replayed_after_torn_append, post_torn_id) == 1);
        ggml_vec_index_free(replayed_after_torn_append);

        ggml_vec_index_free(replayed_with_torn_tail);
        ggml_vec_index_free(replayed);
        ggml_vec_index_free(base);
        std::filesystem::remove(snapshot_path);
        std::filesystem::remove(delta_path);
        std::filesystem::remove(mismatched_snapshot_path);
        std::filesystem::remove(corrupt_delta_path);
        std::filesystem::remove(other_delta_path);
        std::filesystem::remove(other_delta_path + ".lock");
        std::filesystem::remove(diverged_delta_path);
        std::filesystem::remove(diverged_delta_path + ".lock");
        std::filesystem::remove(delta_bound_write_path);
        std::filesystem::remove(alternate_delta_path);
        std::filesystem::remove(alternate_delta_path + ".lock");
    }
    check_ivf_state_not_persisted();

    // Delta replay supports tombstone delete followed by re-adding the same ID.
    {
        for (int bit_width : { 32, 8, 4 }) {
            const std::string snapshot_path =
                (std::filesystem::temp_directory_path() /
                 ("ggml-vector-index-delta-tombstone-" +
                  std::to_string(bit_width) + ".tvim")).string();
            const std::string delta_path =
                (std::filesystem::temp_directory_path() /
                 ("ggml-vector-index-delta-tombstone-" +
                  std::to_string(bit_width) + ".tvid")).string();
            std::filesystem::remove(snapshot_path);
            std::filesystem::remove(delta_path);

            auto * delta_tombstone = ggml_vec_index_create(kDim, bit_width);
            CHECK(delta_tombstone != nullptr);
            std::vector<float> base_vecs;
            base_vecs.insert(base_vecs.end(), seeds[0].begin(), seeds[0].end());
            base_vecs.insert(base_vecs.end(), seeds[1].begin(), seeds[1].end());
            CHECK(ggml_vec_index_add(
                delta_tombstone, base_vecs.data(), 2, ids.data()) == GGML_VEC_INDEX_OK);
            CHECK(ggml_vec_index_write(
                delta_tombstone, snapshot_path.c_str()) == GGML_VEC_INDEX_OK);

            CHECK(ggml_vec_index_remove_logged(
                delta_tombstone, ids[0], delta_path.c_str()) == GGML_VEC_INDEX_OK);
            CHECK(ggml_vec_index_add_logged(
                delta_tombstone, seeds[2].data(), 1, &ids[0], delta_path.c_str()) ==
                GGML_VEC_INDEX_OK);
            CHECK(ggml_vec_index_len(delta_tombstone) == 2);
            CHECK(ggml_vec_index_contains(delta_tombstone, ids[0]) == 1);

            auto * replayed_tombstone = ggml_vec_index_load_with_delta(
                snapshot_path.c_str(), delta_path.c_str());
            CHECK(replayed_tombstone != nullptr);
            CHECK(ggml_vec_index_bit_width(replayed_tombstone) == bit_width);
            CHECK(ggml_vec_index_len(replayed_tombstone) == 2);
            CHECK(ggml_vec_index_contains(replayed_tombstone, ids[0]) == 1);
            CHECK(ggml_vec_index_contains(replayed_tombstone, ids[1]) == 1);

            std::array<float, 2> scores{};
            std::array<uint64_t, 2> out_ids{};
            CHECK(ggml_vec_index_search(
                replayed_tombstone, seeds[2].data(), 1, /*k=*/2,
                scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
            CHECK(out_ids[0] == ids[0]);

            ggml_vec_index_free(replayed_tombstone);
            ggml_vec_index_free(delta_tombstone);
            std::filesystem::remove(snapshot_path);
            std::filesystem::remove(delta_path);
        }
    }

    // v1 f32 snapshots migrate to q8 only for legacy bit_width=8; other
    // legacy widths, including bit_width=4, migrate to f32.
    {
        const std::vector<uint64_t> v1_ids = {
            (1ULL << 37) + 1ULL,
            (1ULL << 37) + 2ULL,
        };
        std::vector<float> v1_vectors;
        v1_vectors.insert(v1_vectors.end(), seeds[0].begin(), seeds[0].end());
        v1_vectors.insert(v1_vectors.end(), seeds[1].begin(), seeds[1].end());

        for (int bit_width : { 32, 8, 4 }) {
            const auto v1_tmp = std::filesystem::temp_directory_path() /
                                ("ggml-vector-index-v1-" + std::to_string(bit_width) + ".tvim");
            const std::string v1_path = v1_tmp.string();
            write_v1_index(v1_path, kDim, bit_width, v1_vectors, v1_ids);

            auto * v1 = ggml_vec_index_load(v1_path.c_str());
            CHECK(v1 != nullptr);
            ggml_vec_index_t * mapped_v1 = nullptr;
            CHECK(ggml_vec_index_load_mmap_ex(v1_path.c_str(), &mapped_v1) ==
                  GGML_VEC_INDEX_E_BAD_VERSION);
            CHECK(mapped_v1 == nullptr);
            CHECK(ggml_vec_index_dim(v1) == kDim);
            CHECK(ggml_vec_index_len(v1) == 2);
            CHECK(ggml_vec_index_bit_width(v1) == (bit_width == 8 ? 8 : 32));

            std::array<float, 1> scores{};
            std::array<uint64_t, 1> out_ids{};
            CHECK(ggml_vec_index_search(
                v1, seeds[1].data(), 1, /*k=*/1,
                scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
            CHECK(out_ids[0] == v1_ids[1]);
            CHECK(std::fabs(scores[0] - 1.0f) < 1e-5f);

            ggml_vec_index_free(v1);
            std::filesystem::remove(v1_path);
        }
    }

    // mmap loading maps the vector section read-only and keeps search parity.
    {
        const std::vector<uint64_t> mmap_ids = {
            (1ULL << 38) + 1ULL,
            (1ULL << 38) + 2ULL,
            (1ULL << 38) + 3ULL,
            (1ULL << 38) + 4ULL,
        };
        std::array<float, 4> normal_scores{};
        std::array<float, 4> mmap_scores{};
        std::array<uint64_t, 4> normal_ids{};
        std::array<uint64_t, 4> mmap_out_ids{};
        const std::vector<float> query = normalize({0.5f, -0.25f, 0.75f, 0.125f});

        for (int bit_width : { 32, 8, 4 }) {
            const auto mmap_tmp = std::filesystem::temp_directory_path() /
                                  ("ggml-vector-index-mmap-" + std::to_string(bit_width) + ".tvim");
            const auto mmap_copy_tmp = std::filesystem::temp_directory_path() /
                                       ("ggml-vector-index-mmap-copy-" + std::to_string(bit_width) + ".tvim");
            const auto mmap_delta_tmp = std::filesystem::temp_directory_path() /
                                        ("ggml-vector-index-mmap-delta-" + std::to_string(bit_width) + ".tvid");
            const std::string mmap_path = mmap_tmp.string();
            const std::string mmap_copy_path = mmap_copy_tmp.string();
            const std::string mmap_delta_path = mmap_delta_tmp.string();
            std::filesystem::remove(mmap_path);
            std::filesystem::remove(mmap_copy_path);
            std::filesystem::remove(mmap_delta_path);
            std::filesystem::remove(mmap_delta_path + ".lock");

            auto * source = ggml_vec_index_create(kDim, bit_width);
            CHECK(source != nullptr);
            CHECK(ggml_vec_index_add(
                source, vecs.data(), static_cast<int>(mmap_ids.size()), mmap_ids.data()) ==
                GGML_VEC_INDEX_OK);
            CHECK(ggml_vec_index_write(source, mmap_path.c_str()) == GGML_VEC_INDEX_OK);

            auto * normal = ggml_vec_index_load(mmap_path.c_str());
            auto * mapped = ggml_vec_index_load_mmap(mmap_path.c_str());
            CHECK(normal != nullptr);
            CHECK(mapped != nullptr);
            CHECK(ggml_vec_index_bit_width(mapped) == bit_width);
            CHECK(ggml_vec_index_len(mapped) == static_cast<int>(mmap_ids.size()));

            CHECK(ggml_vec_index_search(
                normal, query.data(), 1, /*k=*/4,
                normal_scores.data(), normal_ids.data()) == GGML_VEC_INDEX_OK);
            CHECK(ggml_vec_index_search(
                mapped, query.data(), 1, /*k=*/4,
                mmap_scores.data(), mmap_out_ids.data()) == GGML_VEC_INDEX_OK);
            for (int i = 0; i < 4; ++i) {
                CHECK(mmap_out_ids[i] == normal_ids[i]);
                CHECK(std::fabs(mmap_scores[i] - normal_scores[i]) <= 1e-6f);
            }
            if (bit_width == 32) {
                temp_file       symlink_file(".tvim.link");
                std::error_code ec;
                std::filesystem::create_symlink(mmap_path, symlink_file.path, ec);
                if (!ec) {
                    auto * symlink_mmap = ggml_vec_index_load_mmap(symlink_file.path.string().c_str());
                    CHECK(symlink_mmap != nullptr);
                    mmap_scores.fill(std::numeric_limits<float>::quiet_NaN());
                    mmap_out_ids.fill(UINT64_MAX);
                    CHECK(ggml_vec_index_search(
                              symlink_mmap, query.data(), 1, 4, mmap_scores.data(), mmap_out_ids.data()) ==
                          GGML_VEC_INDEX_OK);
                    for (int i = 0; i < 4; ++i) {
                        CHECK(mmap_out_ids[i] == normal_ids[i]);
                        CHECK(std::fabs(mmap_scores[i] - normal_scores[i]) <= 1e-6f);
                    }
                    ggml_vec_index_free(symlink_mmap);
                } else {
                    CHECK(ec == std::errc::operation_not_supported || ec == std::errc::function_not_supported ||
                          ec == std::errc::permission_denied);
                }
            }

            CHECK(std::filesystem::remove(mmap_path));
            std::array<float, 4> unlinked_scores{};
            std::array<uint64_t, 4> unlinked_ids{};
            CHECK(ggml_vec_index_search(
                mapped, query.data(), 1, /*k=*/4,
                unlinked_scores.data(), unlinked_ids.data()) == GGML_VEC_INDEX_OK);
            for (int i = 0; i < 4; ++i) {
                CHECK(unlinked_ids[i] == normal_ids[i]);
                CHECK(std::fabs(unlinked_scores[i] - normal_scores[i]) <= 1e-6f);
            }

            CHECK(ggml_vec_index_build_ivf(mapped, /*n_lists=*/2, /*n_iter=*/2)
                  == GGML_VEC_INDEX_OK);
            std::array<float, 2> mmap_ivf_scores{};
            std::array<uint64_t, 2> mmap_ivf_ids{};
            mmap_ivf_scores.fill(std::numeric_limits<float>::quiet_NaN());
            mmap_ivf_ids.fill(UINT64_MAX);
            CHECK(ggml_vec_index_search_ivf(
                mapped, query.data(), 1, /*k=*/2, /*nprobe=*/2,
                mmap_ivf_scores.data(), mmap_ivf_ids.data()) == GGML_VEC_INDEX_OK);
            for (int i = 0; i < 2; ++i) {
                CHECK(mmap_ivf_ids[i] == normal_ids[i]);
                CHECK(std::fabs(mmap_ivf_scores[i] - normal_scores[i]) <= 1e-6f);
            }

            const uint64_t new_id = (1ULL << 38) + 99ULL;
            CHECK(ggml_vec_index_add(mapped, seeds[0].data(), 1, &new_id)
                  == GGML_VEC_INDEX_E_INVALID_ARG);
            CHECK(ggml_vec_index_remove(mapped, mmap_ids[0])
                  == GGML_VEC_INDEX_E_INVALID_ARG);
            CHECK(ggml_vec_index_compact(mapped) == GGML_VEC_INDEX_E_INVALID_ARG);
            CHECK(ggml_vec_index_write(mapped, mmap_copy_path.c_str()) == GGML_VEC_INDEX_OK);
            auto * copied = ggml_vec_index_load(mmap_copy_path.c_str());
            CHECK(copied != nullptr);
            CHECK(ggml_vec_index_len(copied) == static_cast<int>(mmap_ids.size()));
            ggml_vec_index_free(copied);

            CHECK(ggml_vec_index_compact_delta(
                mapped, mmap_copy_path.c_str(), mmap_delta_path.c_str()) ==
                GGML_VEC_INDEX_OK);
            auto * compacted = ggml_vec_index_load_with_delta(
                mmap_copy_path.c_str(), mmap_delta_path.c_str());
            CHECK(compacted != nullptr);
            CHECK(ggml_vec_index_len(compacted) == static_cast<int>(mmap_ids.size()));
            CHECK(ggml_vec_index_write(mapped, mmap_path.c_str())
                  == GGML_VEC_INDEX_E_INVALID_ARG);
            CHECK(ggml_vec_index_write(mapped, mmap_copy_path.c_str())
                  == GGML_VEC_INDEX_E_INVALID_ARG);

            ggml_vec_index_free(compacted);
            ggml_vec_index_free(mapped);
            ggml_vec_index_free(normal);
            ggml_vec_index_free(source);
            std::filesystem::remove(mmap_path);
            std::filesystem::remove(mmap_copy_path);
            std::filesystem::remove(mmap_delta_path);
            std::filesystem::remove(mmap_delta_path + ".lock");
        }
    }

    // q8 score parity for a dimension that exercises the SIMD tail.
    {
        constexpr int tail_dim = 13;
        const std::vector<float> tail_vector = {
            -1.0f, 0.75f, -0.5f, 0.25f, 0.125f, -0.875f, 0.625f,
            -0.375f, 0.9f, -0.7f, 0.3f, -0.2f, 0.05f,
        };
        const std::vector<float> tail_query = {
            0.2f, -0.4f, 0.6f, -0.8f, 1.0f, 0.3f, -0.5f,
            0.7f, -0.9f, 0.11f, -0.22f, 0.33f, -0.44f,
        };
        const uint64_t tail_id = (1ULL << 55) + 321ULL;

        auto * tail_idx = ggml_vec_index_create(tail_dim, /*bit_width=*/8);
        CHECK(tail_idx != nullptr);
        CHECK(ggml_vec_index_add(
            tail_idx, tail_vector.data(), 1, &tail_id) == GGML_VEC_INDEX_OK);

        std::array<float, 1> scores{};
        std::array<uint64_t, 1> out_ids{};
        CHECK(ggml_vec_index_search(
            tail_idx, tail_query.data(), 1, /*k=*/1,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(out_ids[0] == tail_id);

        const float expected = q8_dot_reference(tail_vector, tail_query);
        const float tolerance = 1e-5f * std::max(1.0f, std::fabs(expected));
        CHECK(std::fabs(scores[0] - expected) <= tolerance);

        ggml_vec_index_free(tail_idx);

        std::vector<float> zero_vector(tail_dim, 0.0f);
        auto * zero_idx = ggml_vec_index_create(tail_dim, /*bit_width=*/8);
        CHECK(zero_idx != nullptr);
        CHECK(ggml_vec_index_add(
            zero_idx, zero_vector.data(), 1, &tail_id) == GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_search(
            zero_idx, tail_query.data(), 1, /*k=*/1,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(scores[0] == 0.0f);
        ggml_vec_index_free(zero_idx);
    }

    // Applying the q8 scale after accumulation can overflow even when the
    // dequantized dot product is finite.
    {
        constexpr int overflow_dim = 8;
        const std::vector<float> small_vector(overflow_dim, 1e-30f);
        const std::vector<float> large_query(overflow_dim, 1e38f);
        const uint64_t overflow_id = 123456789ULL;

        auto * overflow_idx = ggml_vec_index_create(overflow_dim, /*bit_width=*/8);
        CHECK(overflow_idx != nullptr);
        CHECK(ggml_vec_index_add(
            overflow_idx, small_vector.data(), 1, &overflow_id) == GGML_VEC_INDEX_OK);

        std::array<float, 1> scores{};
        std::array<uint64_t, 1> out_ids{};
        CHECK(ggml_vec_index_search(
            overflow_idx, large_query.data(), 1, /*k=*/1,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        const float expected = static_cast<float>(
            overflow_dim * static_cast<double>(small_vector[0]) * large_query[0]);
        CHECK(out_ids[0] == overflow_id);
        CHECK(std::isfinite(scores[0]));
        CHECK(std::fabs(scores[0] - expected) <= std::fabs(expected) * 1e-5f);

        ggml_vec_index_free(overflow_idx);
    }

    // SIMD safety-boundary rounding must not expose non-finite scores.
    {
        constexpr int boundary_dim = 16;
        const std::vector<float> boundary_vector(boundary_dim, 1.0f);
        const std::vector<float> boundary_query(
            boundary_dim, FLT_MAX / static_cast<float>(boundary_dim));
        const uint64_t boundary_id = 123456791ULL;

        for (int bit_width : { 8, 4 }) {
            auto * boundary_idx = ggml_vec_index_create(boundary_dim, bit_width);
            CHECK(boundary_idx != nullptr);
            CHECK(ggml_vec_index_add(
                boundary_idx, boundary_vector.data(), 1, &boundary_id) ==
                GGML_VEC_INDEX_OK);

            std::array<float, 1> scores{};
            std::array<uint64_t, 1> out_ids{};
            CHECK(ggml_vec_index_search(
                boundary_idx, boundary_query.data(), 1, /*k=*/1,
                scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
            CHECK(out_ids[0] == boundary_id);
            CHECK(std::isfinite(scores[0]));

            ggml_vec_index_free(boundary_idx);
        }
    }

    // Large finite terms can overflow float intermediates even when the final
    // dot product is representable after cancellation.
    {
        constexpr int cancel_dim = 2;
        const std::array<float, cancel_dim> cancel_vector = { 1e30f, 1e30f };
        const std::array<float, cancel_dim> cancel_query = { 1e10f, -1e10f };
        const uint64_t cancel_id = 123456790ULL;

        for (int bit_width : { 32, 8, 4 }) {
            auto * cancel_idx = ggml_vec_index_create(cancel_dim, bit_width);
            CHECK(cancel_idx != nullptr);
            CHECK(ggml_vec_index_add(
                cancel_idx, cancel_vector.data(), 1, &cancel_id) == GGML_VEC_INDEX_OK);

            std::array<float, 1> scores{};
            std::array<uint64_t, 1> out_ids{};
            CHECK(ggml_vec_index_search(
                cancel_idx, cancel_query.data(), 1, /*k=*/1,
                scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
            CHECK(out_ids[0] == cancel_id);
            CHECK(scores[0] == 0.0f);

            ggml_vec_index_free(cancel_idx);
        }
    }

    // q4 path: packed nibbles with one f32 scale per vector.
    {
        constexpr int tail_dim = 13;
        const std::vector<float> tail_vector = {
            -1.0f, 0.75f, -0.5f, 0.25f, 0.125f, -0.875f, 0.625f,
            -0.375f, 0.9f, -0.7f, 0.3f, -0.2f, 0.05f,
        };
        const std::vector<float> tail_query = {
            0.2f, -0.4f, 0.6f, -0.8f, 1.0f, 0.3f, -0.5f,
            0.7f, -0.9f, 0.11f, -0.22f, 0.33f, -0.44f,
        };
        const uint64_t q4_id = (1ULL << 55) + 654ULL;

        auto * q4 = ggml_vec_index_create(tail_dim, /*bit_width=*/4);
        CHECK(q4 != nullptr);
        CHECK(ggml_vec_index_bit_width(q4) == 4);
        CHECK(ggml_vec_index_add(q4, tail_vector.data(), 1, &q4_id) == GGML_VEC_INDEX_OK);

        std::array<float, 2> scores{};
        std::array<uint64_t, 2> out_ids{};
        CHECK(ggml_vec_index_search(
            q4, tail_query.data(), 1, /*k=*/2,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(out_ids[0] == q4_id);
        CHECK(out_ids[1] == UINT64_MAX);
        CHECK(scores[1] == -FLT_MAX);

        const float expected = q4_dot_reference(tail_vector, tail_query);
        const float tolerance = 1e-5f * std::max(1.0f, std::fabs(expected));
        CHECK(std::fabs(scores[0] - expected) <= tolerance);

        const std::string q4_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-q4-test.tvim").string();
        const std::string corrupt_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-q4-corrupt-test.tvim").string();
        CHECK(ggml_vec_index_write(q4, q4_path.c_str()) == GGML_VEC_INDEX_OK);
        CHECK(read_file_byte(q4_path, 4) == 2); // .tvim v2
        CHECK(read_file_byte(q4_path, 5) == 4); // q4 bit width
        CHECK(read_file_byte(q4_path, 6) == 3); // q4 storage kind
        CHECK(read_file_byte(q4_path, 24) == 0); // packed components

        auto * q4_loaded = ggml_vec_index_load(q4_path.c_str());
        CHECK(q4_loaded != nullptr);
        CHECK(ggml_vec_index_bit_width(q4_loaded) == 4);
        CHECK(ggml_vec_index_len(q4_loaded) == 1);
        scores = {};
        out_ids = {};
        CHECK(ggml_vec_index_search(
            q4_loaded, tail_query.data(), 1, /*k=*/1,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(out_ids[0] == q4_id);
        CHECK(std::fabs(scores[0] - expected) <= tolerance);

        constexpr size_t q4_vector_offset = 32 + sizeof(float);
        constexpr size_t q4_row_bytes = (tail_dim + 1) / 2;
        expect_corrupt_load_fails(q4_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
            bytes[q4_vector_offset] =
                static_cast<uint8_t>(bytes[q4_vector_offset] & 0xf0u); // low nibble 0 is invalid
        });
        expect_corrupt_load_fails(q4_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
            const size_t tail_byte = q4_vector_offset + q4_row_bytes - 1;
            bytes[tail_byte] =
                static_cast<uint8_t>((bytes[tail_byte] & 0x0fu) | 0x90u); // odd tail high nibble must be zero-code
        });

        ggml_vec_index_free(q4_loaded);
        ggml_vec_index_free(q4);
        std::filesystem::remove(q4_path);
    }

    // q4 parity for a dimension that exercises the optimized loop and tail.
    {
        constexpr int q4_dim = 33;
        std::vector<float> q4_vector(q4_dim);
        std::vector<float> q4_query(q4_dim);
        for (int i = 0; i < q4_dim; ++i) {
            q4_vector[static_cast<size_t>(i)] =
                static_cast<float>((i % 11) - 5) / 5.0f;
            q4_query[static_cast<size_t>(i)] =
                static_cast<float>(((i * 7) % 13) - 6) / 7.0f;
        }
        const uint64_t q4_id = (1ULL << 56) + 123ULL;
        auto * q4 = ggml_vec_index_create(q4_dim, /*bit_width=*/4);
        CHECK(q4 != nullptr);
        CHECK(ggml_vec_index_add(q4, q4_vector.data(), 1, &q4_id) == GGML_VEC_INDEX_OK);

        std::array<float, 1> scores{};
        std::array<uint64_t, 1> out_ids{};
        CHECK(ggml_vec_index_search(
            q4, q4_query.data(), 1, /*k=*/1,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(out_ids[0] == q4_id);

        const float expected = q4_dot_reference(q4_vector, q4_query);
        const float tolerance = 1e-5f * std::max(1.0f, std::fabs(expected));
        CHECK(std::fabs(scores[0] - expected) <= tolerance);
        ggml_vec_index_free(q4);
    }

    // Quantized multi-query searches exercise SIMD-sized rows and top-k ranking.
    {
        constexpr int quant_dim = 17;
        const std::vector<std::vector<float>> quant_vectors = {
            normalize({ 1.0f, 0.5f, -0.25f, 0.125f, 0.0f, -0.75f, 0.625f, -0.5f,
                        0.375f, -0.25f, 0.125f, 0.0f, -0.125f, 0.25f, -0.375f, 0.5f, -0.625f }),
            normalize({ -0.5f, 1.0f, 0.75f, -0.625f, 0.5f, -0.375f, 0.25f, -0.125f,
                        0.0f, 0.125f, -0.25f, 0.375f, -0.5f, 0.625f, -0.75f, 0.875f, -1.0f }),
            normalize({ 0.25f, -0.5f, 1.0f, 0.875f, -0.75f, 0.625f, -0.5f, 0.375f,
                        -0.25f, 0.125f, 0.0f, -0.125f, 0.25f, -0.375f, 0.5f, -0.625f, 0.75f }),
        };
        const std::vector<std::vector<float>> quant_queries = {
            normalize({ 0.9f, 0.4f, -0.2f, 0.1f, 0.0f, -0.7f, 0.6f, -0.45f,
                        0.3f, -0.2f, 0.1f, 0.0f, -0.1f, 0.2f, -0.3f, 0.4f, -0.5f }),
            normalize({ 0.1f, -0.3f, 0.8f, 0.7f, -0.6f, 0.5f, -0.4f, 0.3f,
                        -0.2f, 0.1f, 0.0f, -0.1f, 0.2f, -0.3f, 0.4f, -0.5f, 0.6f }),
        };
        const std::array<uint64_t, 3> quant_ids = { 8801001ULL, 8801002ULL, 8801003ULL };
        std::vector<float> quant_rows;
        std::vector<float> quant_query_rows;
        for (const auto & vector : quant_vectors) {
            quant_rows.insert(quant_rows.end(), vector.begin(), vector.end());
        }
        for (const auto & query : quant_queries) {
            quant_query_rows.insert(quant_query_rows.end(), query.begin(), query.end());
        }

        for (int bit_width : { 8, 4 }) {
            auto * quant_idx = ggml_vec_index_create(quant_dim, bit_width);
            CHECK(quant_idx != nullptr);
            CHECK(ggml_vec_index_add(
                quant_idx,
                quant_rows.data(),
                static_cast<int>(quant_ids.size()),
                quant_ids.data()) == GGML_VEC_INDEX_OK);

            std::array<float, 4> scores{};
            std::array<uint64_t, 4> out_ids{};
            CHECK(ggml_vec_index_search(
                quant_idx, quant_query_rows.data(), /*n_q=*/2, /*k=*/2,
                scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);

            for (size_t q = 0; q < quant_queries.size(); ++q) {
                std::vector<std::pair<float, uint64_t>> expected;
                for (size_t row = 0; row < quant_vectors.size(); ++row) {
                    const float score = bit_width == 8 ?
                        q8_dot_reference(quant_vectors[row], quant_queries[q]) :
                        q4_dot_reference(quant_vectors[row], quant_queries[q]);
                    expected.push_back({ score, quant_ids[row] });
                }
                std::sort(
                    expected.begin(),
                    expected.end(),
                    [](const std::pair<float, uint64_t> & a,
                       const std::pair<float, uint64_t> & b) {
                        if (a.first != b.first) {
                            return a.first > b.first;
                        }
                        return a.second < b.second;
                    });

                for (size_t i = 0; i < 2; ++i) {
                    const size_t out = q * 2 + i;
                    CHECK(out_ids[out] == expected[i].second);
                    const float tolerance =
                        1e-5f * std::max(1.0f, std::fabs(expected[i].first));
                    CHECK(std::fabs(scores[out] - expected[i].first) <= tolerance);
                }
            }

            ggml_vec_index_free(quant_idx);
        }
    }

    // Duplicate IDs in snapshots are malformed input.
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

    // Quantization must not depend on the caller's active rounding mode.
    {
        const int saved_rounding_mode = std::fegetround();
        CHECK(saved_rounding_mode != -1);

        auto score_after_add_with_rounding = [&](int bit_width, int rounding_mode) {
            const std::array<float, kDim> rounding_vector = {
                1.0f, 0.02f, 0.0f, 0.0f,
            };
            const std::array<float, kDim> rounding_query = {
                0.0f, 1.0f, 0.0f, 0.0f,
            };
            const uint64_t rounding_id =
                (1ULL << 57) + static_cast<uint64_t>(bit_width) +
                static_cast<uint64_t>(rounding_mode);
            CHECK(std::fesetround(rounding_mode) == 0);
            auto * rounding_idx = ggml_vec_index_create(kDim, bit_width);
            CHECK(rounding_idx != nullptr);
            CHECK(ggml_vec_index_add(
                rounding_idx, rounding_vector.data(), 1, &rounding_id) ==
                GGML_VEC_INDEX_OK);
            CHECK(std::fesetround(saved_rounding_mode) == 0);

            std::array<float, 1> scores{};
            std::array<uint64_t, 1> out_ids{};
            CHECK(ggml_vec_index_search(
                rounding_idx, rounding_query.data(), 1, /*k=*/1,
                scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
            CHECK(out_ids[0] == rounding_id);
            ggml_vec_index_free(rounding_idx);
            return scores[0];
        };

        for (int bit_width : { 8, 4 }) {
            const float downward_score = score_after_add_with_rounding(bit_width, FE_DOWNWARD);
            const float upward_score = score_after_add_with_rounding(bit_width, FE_UPWARD);
            CHECK(downward_score == upward_score);

            const std::string snapshot_path =
                (std::filesystem::temp_directory_path() /
                 ("ggml-vector-index-rounding-" + std::to_string(bit_width) + ".tvim")).string();
            const std::string delta_path =
                (std::filesystem::temp_directory_path() /
                 ("ggml-vector-index-rounding-" + std::to_string(bit_width) + ".tvid")).string();
            std::filesystem::remove(snapshot_path);
            std::filesystem::remove(delta_path);
            std::filesystem::remove(delta_path + ".lock");

            const std::array<float, kDim> rounding_vector = {
                1.0f, 0.02f, 0.0f, 0.0f,
            };
            const std::array<float, kDim> rounding_query = {
                0.0f, 1.0f, 0.0f, 0.0f,
            };
            const uint64_t rounding_id = (1ULL << 58) + static_cast<uint64_t>(bit_width);
            auto * logged_rounding = ggml_vec_index_create(kDim, bit_width);
            CHECK(logged_rounding != nullptr);
            CHECK(ggml_vec_index_write(logged_rounding, snapshot_path.c_str()) ==
                  GGML_VEC_INDEX_OK);
            CHECK(std::fesetround(FE_DOWNWARD) == 0);
            CHECK(ggml_vec_index_add_logged(
                logged_rounding, rounding_vector.data(), 1, &rounding_id,
                delta_path.c_str()) == GGML_VEC_INDEX_OK);
            CHECK(std::fesetround(FE_UPWARD) == 0);
            auto * replayed_rounding = ggml_vec_index_load_with_delta(
                snapshot_path.c_str(), delta_path.c_str());
            CHECK(std::fesetround(saved_rounding_mode) == 0);
            CHECK(replayed_rounding != nullptr);

            std::array<float, 1> scores{};
            std::array<uint64_t, 1> out_ids{};
            CHECK(ggml_vec_index_search(
                replayed_rounding, rounding_query.data(), 1, /*k=*/1,
                scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
            CHECK(out_ids[0] == rounding_id);
            CHECK(scores[0] == downward_score);

            ggml_vec_index_free(replayed_rounding);
            ggml_vec_index_free(logged_rounding);
            std::filesystem::remove(snapshot_path);
            std::filesystem::remove(delta_path);
            std::filesystem::remove(delta_path + ".lock");
        }
        CHECK(std::fesetround(saved_rounding_mode) == 0);
    }

    // Legacy v1/v3 delta logs remain replayable for f32 snapshots.
    {
        const std::string snapshot_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-legacy-delta-base.tvim").string();
        std::filesystem::remove(snapshot_path);

        const std::vector<uint64_t> base_ids = {
            9001001ULL,
            9001002ULL,
        };
        std::vector<float> base_vectors;
        base_vectors.insert(base_vectors.end(), seeds[0].begin(), seeds[0].end());
        base_vectors.insert(base_vectors.end(), seeds[1].begin(), seeds[1].end());
        const uint64_t delta_id = 9001003ULL;
        std::vector<float> delta_vectors;
        delta_vectors.insert(delta_vectors.end(), seeds[2].begin(), seeds[2].end());
        std::vector<uint64_t> post_ids = base_ids;
        post_ids.push_back(delta_id);
        std::vector<float> post_vectors = base_vectors;
        post_vectors.insert(post_vectors.end(), delta_vectors.begin(), delta_vectors.end());

        auto * base = ggml_vec_index_create(kDim, /*bit_width=*/32);
        CHECK(base != nullptr);
        CHECK(ggml_vec_index_add(
            base, base_vectors.data(), static_cast<int>(base_ids.size()), base_ids.data()) ==
            GGML_VEC_INDEX_OK);
        CHECK(ggml_vec_index_write(base, snapshot_path.c_str()) == GGML_VEC_INDEX_OK);
        ggml_vec_index_free(base);

        for (uint8_t version : { static_cast<uint8_t>(1), static_cast<uint8_t>(3) }) {
            const std::string delta_path =
                (std::filesystem::temp_directory_path() /
                 ("ggml-vector-index-legacy-delta-v" + std::to_string(version) + ".tvid")).string();
            std::filesystem::remove(delta_path);

            const uint32_t base_state = version == 1 ?
                legacy_state_crc32c_f32(kDim, base_vectors, base_ids) :
                f32_state_token_for(kDim, base_vectors, base_ids);
            const uint32_t post_state = version == 1 ?
                legacy_state_crc32c_f32(kDim, post_vectors, post_ids) :
                f32_state_token_for(kDim, post_vectors, post_ids);
            const std::vector<uint8_t> delta_log = build_legacy_f32_delta_log(
                version, kDim, base_state, post_state, delta_vectors, { delta_id });
            write_file_bytes(delta_path, delta_log);

            auto * stale = ggml_vec_index_load(snapshot_path.c_str());
            CHECK(stale != nullptr);
            const uint64_t rejected_id = delta_id + 100 + version;
            CHECK(ggml_vec_index_add_logged(
                stale, seeds[3].data(), 1, &rejected_id, delta_path.c_str()) ==
                GGML_VEC_INDEX_E_INVALID_ARG);
            CHECK(ggml_vec_index_len(stale) == static_cast<int>(base_ids.size()));
            CHECK(ggml_vec_index_contains(stale, delta_id) == 0);
            CHECK(ggml_vec_index_add(stale, seeds[3].data(), 1, &rejected_id) ==
                  GGML_VEC_INDEX_OK);
            ggml_vec_index_free(stale);

            auto * replayed = ggml_vec_index_load_with_delta(
                snapshot_path.c_str(), delta_path.c_str());
            CHECK(replayed != nullptr);
            CHECK(ggml_vec_index_len(replayed) == 3);
            CHECK(ggml_vec_index_contains(replayed, delta_id) == 1);
            std::array<float, 1> scores{};
            std::array<uint64_t, 1> out_ids{};
            CHECK(ggml_vec_index_search(
                replayed, seeds[2].data(), 1, /*k=*/1,
                scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
            CHECK(out_ids[0] == delta_id);

            ggml_vec_index_free(replayed);
            std::filesystem::remove(delta_path);
        }

        std::filesystem::remove(snapshot_path);
    }

    // Delta replay keeps quantized storage quantized. New v4 logs store native
    // q4/q8 rows instead of f32 vectors, while v2 f32-payload logs remain readable.
    {
        for (int bit_width : { 8, 4 }) {
            const std::string suffix = std::to_string(bit_width);
            const std::string snapshot_path =
                (std::filesystem::temp_directory_path() /
                 ("ggml-vector-index-q" + suffix + "-delta-base.tvim")).string();
            const std::string delta_path =
                (std::filesystem::temp_directory_path() /
                 ("ggml-vector-index-q" + suffix + "-delta-log.tvid")).string();
            const std::string corrupt_delta_path =
                (std::filesystem::temp_directory_path() /
                 ("ggml-vector-index-q" + suffix + "-delta-corrupt.tvid")).string();
            const std::string v2_delta_path =
                (std::filesystem::temp_directory_path() /
                 ("ggml-vector-index-q" + suffix + "-delta-v2.tvid")).string();
            std::filesystem::remove(snapshot_path);
            std::filesystem::remove(delta_path);
            std::filesystem::remove(corrupt_delta_path);
            std::filesystem::remove(v2_delta_path);

            const uint64_t base_id = (1ULL << 42) + static_cast<uint64_t>(bit_width);
            const uint64_t delta_id = (1ULL << 42) + static_cast<uint64_t>(bit_width + 100);
            auto * quant_delta = ggml_vec_index_create(kDim, bit_width);
            CHECK(quant_delta != nullptr);
            CHECK(ggml_vec_index_add(
                quant_delta, seeds[0].data(), 1, &base_id) == GGML_VEC_INDEX_OK);
            CHECK(ggml_vec_index_write(
                quant_delta, snapshot_path.c_str()) == GGML_VEC_INDEX_OK);
            CHECK(ggml_vec_index_add_logged(
                quant_delta, seeds[3].data(), 1, &delta_id, delta_path.c_str()) ==
                GGML_VEC_INDEX_OK);

            std::vector<uint8_t> delta_bytes = read_file_bytes(delta_path);
            CHECK(delta_bytes.size() >= 48 + 56);
            CHECK(delta_bytes[4] == 4);
            const size_t old_f32_payload = sizeof(uint64_t) + kDim * sizeof(uint32_t);
            const size_t expected_native_payload =
                sizeof(uint64_t) + sizeof(uint32_t) +
                (bit_width == 4 ? (kDim + 1) / 2 : kDim);
            CHECK(read_u64_le_at(delta_bytes, delta_log_header_size(delta_bytes) + 8) ==
                  expected_native_payload);
            CHECK(expected_native_payload < old_f32_payload);

            auto * replayed_quant = ggml_vec_index_load_with_delta(
                snapshot_path.c_str(), delta_path.c_str());
            CHECK(replayed_quant != nullptr);
            CHECK(ggml_vec_index_bit_width(replayed_quant) == bit_width);
            CHECK(ggml_vec_index_len(replayed_quant) == 2);
            CHECK(ggml_vec_index_contains(replayed_quant, delta_id) == 1);

            std::array<float, 1> scores{};
            std::array<uint64_t, 1> out_ids{};
            CHECK(ggml_vec_index_search(
                replayed_quant, seeds[3].data(), 1, /*k=*/1,
                scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
            CHECK(out_ids[0] == delta_id);

            std::vector<uint8_t> corrupt_bytes = delta_bytes;
            const size_t record_offset = delta_log_header_size(corrupt_bytes);
            const size_t payload_offset = delta_record_payload_offset(corrupt_bytes, record_offset);
            const size_t scale_offset = payload_offset + sizeof(uint64_t);
            write_u32_le_at(corrupt_bytes, scale_offset, 0);
            refresh_delta_record_crc(corrupt_bytes, record_offset);
            write_file_bytes(corrupt_delta_path, corrupt_bytes);
            auto * corrupt_loaded = ggml_vec_index_load_with_delta(
                snapshot_path.c_str(), corrupt_delta_path.c_str());
            CHECK(corrupt_loaded == nullptr);
            ggml_vec_index_free(corrupt_loaded);

            CHECK(ggml_vec_index_compact_delta(
                quant_delta, snapshot_path.c_str(), delta_path.c_str()) == GGML_VEC_INDEX_OK);
            CHECK(std::filesystem::file_size(delta_path) == 48);
            auto * compacted_quant = ggml_vec_index_load_with_delta(
                snapshot_path.c_str(), delta_path.c_str());
            CHECK(compacted_quant != nullptr);
            CHECK(ggml_vec_index_bit_width(compacted_quant) == bit_width);
            CHECK(ggml_vec_index_contains(compacted_quant, delta_id) == 1);
            ggml_vec_index_free(compacted_quant);

            auto * v2_writer = ggml_vec_index_load(snapshot_path.c_str());
            CHECK(v2_writer != nullptr);
            CHECK(ggml_vec_index_compact_delta(
                v2_writer, snapshot_path.c_str(), v2_delta_path.c_str()) ==
                GGML_VEC_INDEX_OK);
            const std::vector<uint8_t> compacted_v4 = read_file_bytes(v2_delta_path);
            CHECK(compacted_v4.size() == 48);
            std::vector<uint8_t> empty_v2(16, 0);
            std::memcpy(empty_v2.data(), compacted_v4.data(), 12);
            empty_v2[4] = 2;
            write_u32_le_at(empty_v2, 12, state_token_from_wide_log_header(compacted_v4, kDim));
            write_file_bytes(v2_delta_path, empty_v2);
            const uint64_t v2_delta_id =
                (1ULL << 42) + static_cast<uint64_t>(bit_width + 200);
            CHECK(ggml_vec_index_add_logged(
                v2_writer, seeds[1].data(), 1, &v2_delta_id, v2_delta_path.c_str()) ==
                GGML_VEC_INDEX_E_INVALID_ARG);
            CHECK(ggml_vec_index_contains(v2_writer, v2_delta_id) == 0);

            std::vector<uint8_t> v2_payload;
            append_u64_le(v2_payload, v2_delta_id);
            for (float value : seeds[1]) {
                append_f32_le(v2_payload, value);
            }
            CHECK(v2_payload.size() == old_f32_payload);

            uint64_t wide_n_active = read_u64_le_at(compacted_v4, 16);
            uint64_t wide_hash_xor = read_u64_le_at(compacted_v4, 24);
            uint64_t wide_hash_sum = read_u64_le_at(compacted_v4, 32);
            uint64_t wide_hash_sum_rot = read_u64_le_at(compacted_v4, 40);
            const uint64_t v2_hash =
                slot_state_hash_quantized(bit_width, v2_delta_id, seeds[1]);
            ++wide_n_active;
            wide_hash_xor ^= v2_hash;
            wide_hash_sum += v2_hash;
            wide_hash_sum_rot += rotl64(v2_hash, 17);
            const uint32_t v2_post_token = state_token_from_wide_values(
                bit_width,
                kDim,
                wide_n_active,
                wide_hash_xor,
                wide_hash_sum,
                wide_hash_sum_rot);

            std::vector<uint8_t> v2_log = empty_v2;
            const size_t v2_record_offset = v2_log.size();
            v2_log.push_back(1); // add
            v2_log.insert(v2_log.end(), { 0, 0, 0 });
            append_u32_le(v2_log, 1);
            append_u64_le(v2_log, v2_payload.size());
            append_u32_le(v2_log, 0); // record CRC placeholder
            append_u32_le(v2_log, v2_post_token);
            v2_log.insert(v2_log.end(), v2_payload.begin(), v2_payload.end());
            uint32_t v2_record_crc = crc32c_update(
                0xffffffffu,
                v2_log.data() + v2_record_offset,
                16);
            v2_record_crc = crc32c_update_u32(v2_record_crc, v2_post_token);
            v2_record_crc = crc32c_update(v2_record_crc, v2_payload.data(), v2_payload.size());
            write_u32_le_at(v2_log, v2_record_offset + 16, v2_record_crc ^ 0xffffffffu);
            write_file_bytes(v2_delta_path, v2_log);

            auto * replayed_v2 = ggml_vec_index_load_with_delta(
                snapshot_path.c_str(), v2_delta_path.c_str());
            CHECK(replayed_v2 != nullptr);
            CHECK(ggml_vec_index_bit_width(replayed_v2) == bit_width);
            CHECK(ggml_vec_index_contains(replayed_v2, v2_delta_id) == 1);

            ggml_vec_index_free(replayed_v2);
            ggml_vec_index_free(v2_writer);
            ggml_vec_index_free(replayed_quant);
            ggml_vec_index_free(quant_delta);
            std::filesystem::remove(snapshot_path);
            std::filesystem::remove(delta_path);
            std::filesystem::remove(corrupt_delta_path);
            std::filesystem::remove(v2_delta_path);
            std::filesystem::remove(delta_path + ".lock");
            std::filesystem::remove(v2_delta_path + ".lock");
            std::filesystem::remove(corrupt_delta_path + ".lock");
        }
    }

    // Header metadata is protected even when all payload sections are empty.
    {
        auto * empty_idx = ggml_vec_index_create(kDim, /*bit_width=*/8);
        CHECK(empty_idx != nullptr);
        const std::string empty_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-empty-test.tvim").string();
        const std::string corrupt_path =
            (std::filesystem::temp_directory_path() /
             "ggml-vector-index-empty-corrupt-test.tvim").string();
        CHECK(ggml_vec_index_write(empty_idx, empty_path.c_str()) == GGML_VEC_INDEX_OK);
        expect_corrupt_load_fails(
            empty_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
                bytes[8] += 1; // valid dimension change must fail the header CRC
            });
        ggml_vec_index_free(empty_idx);
        std::filesystem::remove(empty_path);
    }

    // q8 path: stores quantized codes, searches directly against
    // q8 storage, and persists as .tvim v2 with q8 metadata.
    {
        auto * q8 = ggml_vec_index_create(kDim, /*bit_width=*/8);
        CHECK(q8 != nullptr);
        CHECK(ggml_vec_index_dim(q8) == kDim);
        CHECK(ggml_vec_index_bit_width(q8) == 8);

        const std::vector<uint64_t> q8_ids = {
            (1ULL << 33) + 99ULL,
            (1ULL << 48) + 77ULL,
        };
        std::vector<float> q8_vecs;
        q8_vecs.insert(q8_vecs.end(), seeds[0].begin(), seeds[0].end());
        q8_vecs.insert(q8_vecs.end(), seeds[2].begin(), seeds[2].end());
        CHECK(ggml_vec_index_add(q8, q8_vecs.data(), 2, q8_ids.data()) == 0);

        std::array<float, 4> scores{};
        std::array<uint64_t, 4> out_ids{};
        CHECK(ggml_vec_index_search(
            q8, seeds[2].data(), 1, /*k=*/4,
            scores.data(), out_ids.data()) == 0);
        CHECK(out_ids[0] == q8_ids[1]);
        CHECK(std::fabs(scores[0] - 1.0f) < 1e-5f);
        CHECK(scores[2] == -FLT_MAX);
        CHECK(out_ids[2] == UINT64_MAX);
        CHECK(scores[3] == -FLT_MAX);
        CHECK(out_ids[3] == UINT64_MAX);

        const uint64_t q8_missing_id = (1ULL << 59) + 17ULL;
        const std::array<uint64_t, 3> q8_allowed = {
            q8_missing_id, q8_ids[0], q8_ids[0],
        };
        scores = {};
        out_ids = {};
        CHECK(ggml_vec_index_search_filtered(
            q8, seeds[2].data(), 1, /*k=*/3,
            q8_allowed.data(), static_cast<int>(q8_allowed.size()),
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(out_ids[0] == q8_ids[0]);
        CHECK(out_ids[1] == UINT64_MAX);
        CHECK(out_ids[2] == UINT64_MAX);
        CHECK(scores[1] == -FLT_MAX);
        CHECK(scores[2] == -FLT_MAX);

        CHECK(ggml_vec_index_search_filtered(
            q8, seeds[2].data(), 1, /*k=*/2,
            nullptr, /*n_allowed=*/0,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(out_ids[0] == UINT64_MAX);
        CHECK(out_ids[1] == UINT64_MAX);
        CHECK(scores[0] == -FLT_MAX);
        CHECK(scores[1] == -FLT_MAX);

        auto * q8_filter = ggml_vec_index_filter_create(
            q8, q8_allowed.data(), static_cast<int>(q8_allowed.size()));
        CHECK(q8_filter != nullptr);
        scores = {};
        out_ids = {};
        CHECK(ggml_vec_index_search_prepared_filtered(
            q8, q8_filter, seeds[2].data(), 1, /*k=*/3,
            scores.data(), out_ids.data()) == GGML_VEC_INDEX_OK);
        CHECK(out_ids[0] == q8_ids[0]);
        CHECK(out_ids[1] == UINT64_MAX);
        CHECK(out_ids[2] == UINT64_MAX);
        CHECK(scores[1] == -FLT_MAX);
        CHECK(scores[2] == -FLT_MAX);
        ggml_vec_index_filter_free(q8_filter);

        const auto q8_tmp = std::filesystem::temp_directory_path() /
                            "ggml-vector-index-q8-test.tvim";
        const std::string q8_path = q8_tmp.string();
        CHECK(ggml_vec_index_write(q8, q8_path.c_str()) == 0);
        CHECK(read_file_byte(q8_path, 4) == 2); // .tvim v2
        CHECK(read_file_byte(q8_path, 5) == 8); // q8 bit width
        CHECK(read_file_byte(q8_path, 6) == 2); // q8 storage kind
        CHECK(read_file_byte(q8_path, 7) == 1); // checksum trailer present

        const auto corrupt_tmp = std::filesystem::temp_directory_path() /
                                 "ggml-vector-index-corrupt-test.tvim";
        const std::string corrupt_path = corrupt_tmp.string();

        // Legacy v2 files without a checksum remain readable.
        const auto legacy_v2_tmp = std::filesystem::temp_directory_path() /
                                   "ggml-vector-index-legacy-v2-test.tvim";
        const std::string legacy_v2_path = legacy_v2_tmp.string();
        std::vector<uint8_t> legacy_v2 = read_file_bytes(q8_path);
        legacy_v2[7] = 0;
        legacy_v2.resize(legacy_v2.size() - 4 * sizeof(uint32_t));
        write_file_bytes(legacy_v2_path, legacy_v2);
        auto * legacy_loaded = ggml_vec_index_load(legacy_v2_path.c_str());
        CHECK(legacy_loaded != nullptr);
        CHECK(ggml_vec_index_len(legacy_loaded) == 2);
        ggml_vec_index_free(legacy_loaded);
        expect_corrupt_load_fails(
            legacy_v2_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
                bytes[32] = 0;
                bytes[33] = 0;
                bytes[34] = 0;
                bytes[35] = 0; // q8 scale must be positive and finite
            });
        expect_corrupt_load_fails(
            legacy_v2_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
                bytes[40] = 0x80; // q8 codes are restricted to [-127, 127]
            });
        expect_corrupt_load_fails(
            legacy_v2_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
                const size_t id_offset =
                    32 + 2 * sizeof(float) + 2 * kDim * sizeof(int8_t);
                for (size_t i = 0; i < sizeof(uint64_t); ++i) {
                    bytes[id_offset + sizeof(uint64_t) + i] = bytes[id_offset + i];
                }
            });
        expect_corrupt_load_fails(
            legacy_v2_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
                const size_t id_offset =
                    32 + 2 * sizeof(float) + 2 * kDim * sizeof(int8_t);
                for (size_t i = 0; i < sizeof(uint64_t); ++i) {
                    bytes[id_offset + i] = 0xff;
                }
            });
        std::filesystem::remove(legacy_v2_path);

        expect_corrupt_load_fails(q8_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
            bytes[0] = 'X';
        });
        expect_corrupt_load_fails(q8_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
            bytes[4] = 99; // unsupported version
        });
        expect_corrupt_load_fails(q8_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
            bytes[6] = 1; // storage kind does not match bit_width=8
        });
        expect_corrupt_load_fails(q8_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
            bytes[7] |= 0x80; // unknown flags are rejected
        });
        expect_corrupt_load_fails(q8_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
            bytes[16] = 0; // qparam_type must be scale-f32 for q8
        });
        expect_corrupt_load_fails(q8_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
            bytes[28] = 1; // reserved u32 must be zero
        });
        expect_corrupt_load_fails(q8_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
            bytes[32] ^= 1; // q8 scale payload bit flip
        });
        expect_corrupt_load_fails(q8_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
            bytes[40] ^= 1; // q8 code payload bit flip
        });
        expect_corrupt_load_fails(q8_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
            bytes[48] ^= 1; // id payload bit flip
        });
        expect_corrupt_load_fails(q8_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
            bytes[bytes.size() - 16] ^= 1; // header checksum bit flip
        });
        expect_corrupt_load_fails(q8_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
            bytes[bytes.size() - 12] ^= 1; // qparams checksum bit flip
        });
        expect_corrupt_load_fails(q8_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
            bytes[bytes.size() - 8] ^= 1; // vectors checksum bit flip
        });
        expect_corrupt_load_fails(q8_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
            bytes[bytes.size() - 4] ^= 1; // ids checksum bit flip
        });
        expect_corrupt_load_fails(q8_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
            bytes.resize(35); // truncated q8 scales
        });
        expect_corrupt_load_fails(q8_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
            bytes.resize(43); // truncated q8 codes
        });
        expect_corrupt_load_fails(q8_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
            bytes.resize(bytes.size() - 1); // truncated ids
        });
        expect_corrupt_load_fails(q8_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
            bytes.push_back(0); // trailing data is not part of the declared file
        });
        expect_corrupt_load_fails(q8_path, corrupt_path, [](std::vector<uint8_t> & bytes) {
            bytes[12] = 0xff;
            bytes[13] = 0xff;
            bytes[14] = 0xff;
            bytes[15] = 0xff; // impossible vector count for this file size
        });
        ggml_vec_index_free(q8);

        auto * q8_loaded = ggml_vec_index_load(q8_path.c_str());
        CHECK(q8_loaded != nullptr);
        CHECK(ggml_vec_index_dim(q8_loaded) == kDim);
        CHECK(ggml_vec_index_len(q8_loaded) == 2);
        CHECK(ggml_vec_index_bit_width(q8_loaded) == 8);
        CHECK(ggml_vec_index_contains(q8_loaded, q8_ids[0]) == 1);
        CHECK(ggml_vec_index_contains(q8_loaded, q8_ids[1]) == 1);

        scores = {};
        out_ids = {};
        CHECK(ggml_vec_index_search(
            q8_loaded, seeds[0].data(), 1, /*k=*/1,
            scores.data(), out_ids.data()) == 0);
        CHECK(out_ids[0] == q8_ids[0]);
        CHECK(std::fabs(scores[0] - 1.0f) < 1e-5f);

        ggml_vec_index_free(q8_loaded);
        std::filesystem::remove(q8_path);
    }

    std::filesystem::remove_all(test_temp_dir);

    std::printf("test-vector-index: OK\n");
    return 0;
}
