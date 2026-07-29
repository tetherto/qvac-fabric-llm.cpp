#pragma once

#include "ggml-vector-index.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <cfloat>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cwchar>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <queue>
#include <shared_mutex>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#else
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <fcntl.h>
#include <io.h>
#include <share.h>
#include <sys/stat.h>
#include <windows.h>
#endif

inline constexpr uint8_t  kTvimMagic[4]   = { 'T', 'V', 'P', 'I' };
inline constexpr uint8_t  kTvimVersionV1  = 1;
inline constexpr uint8_t  kTvimVersion    = 2;
inline constexpr uint8_t  kStorageF32     = 1;
inline constexpr uint8_t  kStorageQ8      = 2;
inline constexpr uint8_t  kStorageQ4      = 3;
inline constexpr uint8_t  kFlagCRC32C     = 1;
inline constexpr uint32_t kQParamNone     = 0;
inline constexpr uint32_t kQParamScaleF32 = 1;
inline constexpr size_t   kTvimV1HeaderSize = 16;
inline constexpr size_t   kTvimHeaderSize = 32;
inline constexpr size_t   kTvimChecksumSize = 16;
inline constexpr uint8_t  kTvidMagic[4]   = { 'T', 'V', 'D', 'L' };
inline constexpr uint8_t  kTvidVersionV1  = 1;
inline constexpr uint8_t  kTvidVersion    = 2;
inline constexpr uint8_t  kTvidVersionV3  = 3;
inline constexpr uint8_t  kTvidVersionV4  = 4;
inline constexpr uint8_t  kTvidOpAdd      = 1;
inline constexpr uint8_t  kTvidOpRemove   = 2;
inline constexpr size_t   kTvidHeaderSize = 16;
inline constexpr size_t   kTvidHeaderSizeV4 = 48;
inline constexpr size_t   kTvidRecordHeaderSize = 24;
inline constexpr size_t   kTvidRecordHeaderSizeV4 = 56;
inline constexpr size_t   kTvidWideStateSize = 32;
inline constexpr size_t   kMaxIndexLen    = static_cast<size_t>(std::numeric_limits<int>::max());
inline constexpr uint64_t kMaxSnapshotBytes = UINT64_C(1) << 32;

static_assert(sizeof(float) == sizeof(uint32_t), "ggml-vector-index requires float32");

namespace ggml_vec_index_detail {
inline bool can_address_array(size_t count, size_t element_size) {
    return count == 0 || element_size <= std::numeric_limits<size_t>::max() / count;
}
}  // namespace ggml_vec_index_detail

using ggml_vec_index_detail::can_address_array;

struct MappedFile {
    void * data = nullptr;
    size_t size = 0;
#ifdef _WIN32
    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE mapping = nullptr;
#else
    int fd = -1;
#endif

    MappedFile() = default;
    ~MappedFile();
    MappedFile(const MappedFile &) = delete;
    MappedFile & operator=(const MappedFile &) = delete;
};

struct DeltaFileStamp {
    bool valid = false;
    uint64_t size = 0;
    int64_t write_time = 0;
#ifdef _WIN32
    uint64_t volume_serial = 0;
    uint64_t file_index = 0;
#else
    uint64_t device = 0;
    uint64_t inode = 0;
#endif
};

struct DeltaStateWide {
    uint64_t n_active = 0;
    uint64_t hash_xor = 0;
    uint64_t hash_sum = 0;
    uint64_t hash_sum_rot = 0;
};

struct DeltaTailCache {
    bool valid = false;
    std::string path_key;
    int state_kind = 0;
    uint32_t tail_crc = 0;
    DeltaStateWide tail_wide;
    uint64_t complete_size = 0;
    DeltaFileStamp stamp;
};

struct ScoreId {
    double score = 0.0;
    uint64_t id = 0;
};

inline bool score_id_better(const ScoreId & a, const ScoreId & b) {
    if (a.score != b.score) {
        return a.score > b.score;
    }
    return a.id < b.id;
}

struct MinHeapCmp {
    bool operator()(const ScoreId & a, const ScoreId & b) const { return score_id_better(a, b); }
};

struct ggml_vec_index {
    mutable std::shared_mutex mutex;

    int            dim                     = 0;
    int            bit_width               = 32;
    uint64_t       generation              = 0;
    uint64_t       filter_cookie           = 0;
    bool           read_only_mmap          = false;
    bool           delta_log_start_allowed = false;
    bool           delta_log_bound         = false;
    std::string bound_delta_log_path_key;
    bool delta_log_rebase_pending = false;
    uint32_t delta_log_rebase_crc = 0;
    DeltaStateWide delta_log_rebase_wide;
    int delta_log_rebase_state_kind = 0;
    uint64_t state_hash_xor = 0;
    uint64_t state_hash_sum = 0;
    uint64_t state_hash_sum_rot = 0;
    DeltaTailCache delta_tail_cache;

    std::unique_ptr<MappedFile> mapped_file;
    std::string mapped_source_path;
    size_t mapped_vector_bytes = 0;
    const float   * mapped_data = nullptr;
    const int8_t  * mapped_q8_data = nullptr;
    const uint8_t * mapped_q4_data = nullptr;

    std::vector<float> data;
    std::vector<int8_t> q8_data;
    std::vector<float>  q8_scale;
    std::vector<uint8_t> q4_data;
    std::vector<float>   q4_scale;

    std::vector<uint64_t> slot_to_id;
    std::vector<uint8_t>  slot_active;
    size_t n_active = 0;
    std::unordered_map<uint64_t, size_t> id_to_slot;

    uint64_t ivf_generation = std::numeric_limits<uint64_t>::max();
    int ivf_n_lists = 0;
    std::vector<float> ivf_centroids;
    std::vector<std::vector<size_t>> ivf_lists;
};

struct ggml_vec_index_filter {
    const ggml_vec_index_t * owner = nullptr;
    uint64_t owner_cookie = 0;
    int dim = 0;
    int bit_width = 32;
    uint64_t generation = 0;
    std::vector<size_t> slots;
};

enum class DeltaStateKind {
    legacy_crc,
    state_token,
    wide_state,
};

enum class DeltaLogFormat {
    v1,
    v2,
    v3,
    v4,
};

struct DeltaAppendResult {
    int status = GGML_VEC_INDEX_OK;
    bool record_complete = false;
};

class DeltaLogLock {
public:
    explicit DeltaLogLock(const char * path);
    ~DeltaLogLock();

    DeltaLogLock(const DeltaLogLock &) = delete;
    DeltaLogLock & operator=(const DeltaLogLock &) = delete;

    bool ok() const;
    bool ensure_data_file_locked(const char * path);

private:
    std::shared_ptr<std::mutex> sidecar_process_mutex;
    std::shared_ptr<std::mutex> data_process_mutex;
    std::unique_lock<std::mutex> sidecar_process_lock;
    std::unique_lock<std::mutex> data_process_lock;
    bool locked = false;
#ifdef _WIN32
    HANDLE sidecar_file = INVALID_HANDLE_VALUE;
    HANDLE data_file = INVALID_HANDLE_VALUE;
    OVERLAPPED sidecar_lock_overlapped = {};
    OVERLAPPED data_lock_overlapped = {};

    bool lock_data_file(const std::filesystem::path & path, bool create);
    static void close_file(HANDLE & file, OVERLAPPED & overlapped);
#else
    int sidecar_fd = -1;
    int data_fd = -1;

    static bool lock_fd(int fd);
    bool lock_data_file(const std::filesystem::path & path, bool create);
    static void close_fd(int & fd);
#endif
};

#ifdef GGML_VEC_INDEX_TEST_HOOKS
extern "C" {
void ggml_vec_index_test_set_oom_countdown(int64_t countdown);
void ggml_vec_index_test_set_write_fail_after(int64_t bytes);
void ggml_vec_index_test_set_truncate_fail(int fail);
void ggml_vec_index_test_set_parent_fsync_fail(int fail);
void ggml_vec_index_test_set_delta_append_wait_target(int target);
int ggml_vec_index_test_get_delta_append_waiters(void);
void ggml_vec_index_test_set_load_with_delta_pause_ms(int pause_ms);
void ggml_vec_index_test_reset_delta_tail_scan_count(void);
int64_t ggml_vec_index_test_get_delta_tail_scan_count(void);
void ggml_vec_index_test_reset_state_crc_scan_count(void);
int64_t ggml_vec_index_test_get_state_crc_scan_count(void);
int ggml_vec_index_test_get_load_with_delta_waiters(void);
}
#endif

void test_maybe_throw_bad_alloc();
bool test_consume_write_bytes(size_t n);
void test_wait_after_delta_validate();
void test_wait_after_load_with_delta_snapshot();

bool is_supported_bit_width(int bit_width);
bool is_valid_id(uint64_t id);
bool all_finite(const float * values, size_t n);
uint32_t float_to_u32(float value);
float u32_to_float(uint32_t value);
void put_u32_le(uint8_t * dst, uint32_t v);
uint32_t get_u32_le(const uint8_t * src);
void put_u64_le(uint8_t * dst, uint64_t v);
uint64_t get_u64_le(const uint8_t * src);
uint32_t crc32c_update(uint32_t crc, const void * data, size_t len);
uint32_t crc32c_update_u32(uint32_t crc, uint32_t v);
uint32_t crc32c_update_u64(uint32_t crc, uint64_t v);
bool checked_add_u64(uint64_t a, uint64_t b, uint64_t & out);
bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t & out);
bool expected_v1_snapshot_size(uint64_t n, uint64_t dim, uint64_t & expected);
bool supported_v1_snapshot_size(uint64_t n, uint64_t dim, uint64_t & expected);
int snapshot_write_v1_preflight(size_t n, size_t dim);
bool filesystem_path_from_utf8(const char * path, std::filesystem::path & out);
bool filesystem_paths_equal(const char * lhs, const char * rhs);

void invalidate_ivf(ggml_vec_index & idx);
bool is_q8(const ggml_vec_index & idx);
bool is_q4(const ggml_vec_index & idx);
bool is_quantized(const ggml_vec_index & idx);
uint8_t storage_kind(const ggml_vec_index & idx);
size_t q4_row_bytes(size_t dim);
size_t vector_bytes(const ggml_vec_index & idx);
bool slot_is_active(const ggml_vec_index & idx, size_t slot);
size_t active_count(const ggml_vec_index & idx);
const float * f32_data_ptr(const ggml_vec_index & idx);
const int8_t * q8_data_ptr(const ggml_vec_index & idx);
const uint8_t * q4_data_ptr(const ggml_vec_index & idx);
bool has_vector_storage(const ggml_vec_index & idx);
int q4_decode(uint8_t nibble);
void quantize_q8_row(const float * src, int8_t * dst, int dim, float & scale);
void quantize_q4_row(const float * src, uint8_t * dst, int dim, float & scale);
void rollback_appended_slots_unlocked(
    ggml_vec_index_t * idx,
    size_t base_slot,
    const uint64_t * ids,
    int n) noexcept;
int ggml_vec_index_add_unlocked(
    ggml_vec_index_t * idx,
    const float * vectors,
    int n,
    const uint64_t * ids,
    bool finalize);
int ggml_vec_index_remove_unlocked(
    ggml_vec_index_t * idx,
    uint64_t id,
    bool allow_delta_bound = false);

uint32_t index_state_crc32c(const ggml_vec_index & idx);
uint32_t index_state_crc32c_after_remove(const ggml_vec_index & idx, uint64_t id);
uint64_t slot_state_hash(const ggml_vec_index & idx, size_t slot);
void add_state_hash(ggml_vec_index & idx, uint64_t hash);
void remove_state_hash(ggml_vec_index & idx, uint64_t hash);
void rebuild_state_hash(ggml_vec_index & idx);
uint32_t index_state_token(const ggml_vec_index & idx);
uint32_t index_state_token_after_remove(const ggml_vec_index & idx, uint64_t id);
DeltaStateWide index_state_wide(const ggml_vec_index & idx);
DeltaStateWide index_state_wide_after_remove(const ggml_vec_index & idx, uint64_t id);

DeltaStateKind delta_state_kind_for_format(DeltaLogFormat format);
DeltaLogFormat delta_log_format_for_append(const char * path);
uint32_t current_delta_state(const ggml_vec_index & idx, DeltaStateKind state_kind);
DeltaStateWide current_delta_state_wide(const ggml_vec_index & idx);
void invalidate_delta_tail_cache(ggml_vec_index & idx);
bool bind_delta_log_path(ggml_vec_index & idx, const char * delta_path);
bool delta_log_matches_index_unlocked(const ggml_vec_index_t * idx, const char * delta_path);
bool validate_logged_add_args(
    const ggml_vec_index_t * idx,
    const float * vectors,
    int n,
    const uint64_t * ids);
int check_logged_add_duplicates(
    const ggml_vec_index_t * idx,
    int n,
    const uint64_t * ids);
bool build_add_delta_payload_f32(
    const ggml_vec_index_t * idx,
    const float * vectors,
    int n,
    const uint64_t * ids,
    std::vector<uint8_t> & payload);
bool build_add_delta_payload_from_slots(
    const ggml_vec_index_t * idx,
    size_t base_slot,
    int n,
    std::vector<uint8_t> & payload);
std::vector<uint8_t> build_remove_delta_payload(uint64_t id);
DeltaAppendResult append_delta_record_locked(
    ggml_vec_index & idx,
    const char * delta_path,
    DeltaLogFormat format,
    uint8_t op,
    uint32_t n,
    uint32_t base_crc_for_new_log,
    uint32_t state_crc,
    const DeltaStateWide & base_wide_for_new_log,
    const DeltaStateWide & state_wide,
    const std::vector<uint8_t> & payload);
