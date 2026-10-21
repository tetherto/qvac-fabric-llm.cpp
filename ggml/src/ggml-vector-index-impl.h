#pragma once

#include "ggml-vector-index.h"

#include <algorithm>
#include <array>
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
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <queue>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(__aarch64__) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
#define GGML_VEC_INDEX_USE_NEON 1
#else
#define GGML_VEC_INDEX_USE_NEON 0
#endif

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
inline constexpr uint8_t  kTvimVersionV3  = 3;
inline constexpr uint8_t  kStorageF32     = 1;
inline constexpr uint8_t  kStorageQ8      = 2;
inline constexpr uint8_t  kStorageQ4      = 3;
inline constexpr uint8_t  kStorageTurboVecQ4 = 4;
inline constexpr uint8_t  kStorageTurboVecQ2 = 5;
inline constexpr uint8_t  kFlagCRC32C     = 1;
inline constexpr uint32_t kQParamNone     = 0;
inline constexpr uint32_t kQParamScaleF32 = 1;
inline constexpr size_t   kTvimV1HeaderSize = 16;
inline constexpr size_t   kTvimHeaderSize = 32;
inline constexpr size_t   kTvimChecksumSize = 16;
inline constexpr size_t   kTvimOffMagic           = 0;
inline constexpr size_t   kTvimOffVersion         = 4;
inline constexpr size_t   kTvimOffBitWidth        = 5;
inline constexpr size_t   kTvimOffStorage         = 6;
inline constexpr size_t   kTvimOffFlags           = 7;
inline constexpr size_t   kTvimOffDim             = 8;
inline constexpr size_t   kTvimOffCount           = 12;
inline constexpr size_t   kTvimOffQParamType      = 16;
inline constexpr size_t   kTvimOffQParamSize      = 20;
inline constexpr size_t   kTvimOffCompSize        = 24;
inline constexpr size_t   kTvimOffReserved        = 28;
inline constexpr size_t   kTvimChecksumOffHeader  = 0;
inline constexpr size_t   kTvimChecksumOffQParams = 4;
inline constexpr size_t   kTvimChecksumOffVectors = 8;
inline constexpr size_t   kTvimChecksumOffIds     = 12;
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
inline constexpr size_t   kTvidOffMagic           = 0;
inline constexpr size_t   kTvidOffVersion         = 4;
inline constexpr size_t   kTvidOffBitWidth        = 5;
inline constexpr size_t   kTvidOffReserved0       = 6;
inline constexpr size_t   kTvidOffReserved1       = 7;
inline constexpr size_t   kTvidOffDim             = 8;
inline constexpr size_t   kTvidOffState           = 12;
inline constexpr size_t   kTvidOffWideState       = 16;
inline constexpr size_t   kTvidRecordOffOp        = 0;
inline constexpr size_t   kTvidRecordOffReserved  = 1;
inline constexpr size_t   kTvidRecordOffCount     = 4;
inline constexpr size_t   kTvidRecordOffPayload   = 8;
inline constexpr size_t   kTvidRecordOffCrc       = 16;
inline constexpr size_t   kTvidRecordOffState     = 20;
inline constexpr size_t   kTvidRecordOffWide      = 24;
inline constexpr size_t   kMaxIndexLen    = static_cast<size_t>(std::numeric_limits<int>::max());
inline constexpr uint64_t kMaxSnapshotBytes = UINT64_C(1) << 32;
// TurboVec 1.x (block-Hadamard rotation) accepts dims up to upstream MAX_DIM.
inline constexpr int      kTurboVecMaxDim = 16384;
// Legacy 0.9 TurboVec (v3 snapshots) builds a dense dim x dim QR rotation.
inline constexpr int      kTurboVecLegacyMaxDim = 1024;
inline constexpr int      kTurboVecMaxSerializedDim = 65536;
inline constexpr float    kTurboVecMaxInputMagnitude = 1e16f;

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
    int64_t  change_time = 0;
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
    uint32_t       tail_record_crc = 0;
    DeltaStateWide tail_wide;
    uint64_t       tail_record_offset = 0;
    uint64_t       tail_crc_offset    = 0;
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

// Reserves room for `additional` more elements without doubling a tight
// buffer for a small append (upstream lib.rs `reserve_mostly_exact`, #501):
// appends of at most an eighth of the length reserve exactly, plus an
// eighth of headroom so a run of small appends stays amortized.
template <typename T>
void reserve_mostly_exact(std::vector<T> & v, size_t additional) {
    const size_t len = v.size();
    if (v.capacity() - len >= additional) {
        return;
    }
    const size_t headroom = len / 8;
    if (additional <= headroom) {
        v.reserve(len + additional + headroom);
    } else {
        v.reserve(std::max(len + additional, v.capacity() + v.capacity() / 2));
    }
}

// Keeps the best k candidates in a min-heap under score_id_better.
inline void update_topk_heap(std::vector<ScoreId> & heap, int k, const ScoreId & candidate) {
    if (heap.size() < static_cast<size_t>(k)) {
        heap.push_back(candidate);
        std::push_heap(heap.begin(), heap.end(), MinHeapCmp());
    } else if (score_id_better(candidate, heap.front())) {
        std::pop_heap(heap.begin(), heap.end(), MinHeapCmp());
        heap.back() = candidate;
        std::push_heap(heap.begin(), heap.end(), MinHeapCmp());
    }
}

inline constexpr int kMaxThreads = 256;

// A single TurboVec query over at least this many 32-vector blocks splits
// them across threads (upstream turbovec's block-parallel threshold).
inline constexpr size_t kTurboVecParallelMinBlocks = 1024;

// One parallel_for task: ctx is the caller's functor.
using ParallelTaskFn = void (*)(void * ctx, size_t task, size_t worker);

// Runs fn(ctx, task, worker) for every task in [0, n_tasks) on the calling
// thread (worker 0) and up to workers - 1 helpers from the process-wide
// pool (ggml-vector-index-pool.cpp), each with its own worker id in
// [1, workers). Helpers run under the caller's floating-point environment.
// A call from inside a task, or after the pool shut down, runs inline.
void parallel_run(size_t n_tasks, size_t workers, ParallelTaskFn fn, void * ctx);

// Runs fn(task, worker) for every task in [0, n_tasks) on up to n_threads
// threads, the calling thread included; worker is in [0, workers). Tasks
// are handed out dynamically, so fn must not depend on which worker runs a
// task. Rethrows the first exception a task threw. One thread keeps every
// task on the calling thread without touching the pool.
template <typename Fn>
void parallel_for(size_t n_tasks, int n_threads, Fn && fn) {
    const size_t workers = std::min(n_tasks, static_cast<size_t>(std::max(1, n_threads)));
    if (workers <= 1) {
        for (size_t task = 0; task < n_tasks; ++task) {
            fn(task, size_t{ 0 });
        }
        return;
    }
    using Functor = std::remove_reference_t<Fn>;
    parallel_run(
        n_tasks,
        workers,
        [](void * ctx, size_t task, size_t worker) { (*static_cast<Functor *>(ctx))(task, worker); },
        const_cast<void *>(static_cast<const void *>(std::addressof(fn))));
}

// How a staged search weighs the sign plane and estimates a level from its
// bits (upstream pack.rs `PlanesStats`).
struct TurboVecPlanesStats {
    float m = 0.0f;
    float alpha = 0.0f;
    float beta[3] = {};
    float alpha1 = 0.0f;
    float beta1 = 0.0f;
};

// A TurboVec index's binding to the v8 file it syncs to (upstream
// `SyncCursor` plus its dirty sets). A 1.x index removes by swap-remove,
// as upstream does, so its slot order is the file's row order.
struct TurboVecSyncBinding {
    bool bound = false;
    std::string path;
    // Generation and row count of the last commit, the file's identity,
    // and the calibration generation that commit was written under.
    uint64_t gen = 0;
    uint64_t n_synced = 0;
    uint64_t nonce = 0;
    uint64_t calib_gen = 0;
    // File rows whose redo ops ride the last commit, and rows changed since.
    std::unordered_set<size_t> pending;
    std::unordered_set<size_t> fresh;
    // `fresh` lost a row, so the next sync must rewrite the file, after the
    // same generation check as an incremental one.
    bool rewrite = false;
    // A failed sync may still have landed its commit. These name it, so the
    // next sync can tell its own commit from another writer's: the nonce of
    // a full write, or the generation and header CRC of an incremental one.
    uint64_t attempt_nonce = 0;
    uint64_t attempt_gen = 0;
    uint32_t attempt_crc = 0;
};

struct ggml_vec_index {
    mutable std::shared_mutex mutex;

    int            dim                     = 0;
    int            bit_width               = 32;
    uint64_t       generation              = 0;
    // Worker threads a call may use (ggml_vec_index_set_threads); 1 keeps
    // every call on the calling thread.
    int            n_threads               = 1;
    uint64_t       filter_cookie           = 0;
    bool           read_only_mmap          = false;
    bool           delta_log_start_allowed = false;
    bool           delta_log_bound         = false;
    bool           delta_log_reload_required = false;
    std::string bound_delta_log_path_key;
    bool delta_log_rebase_pending = false;
    uint32_t delta_log_rebase_crc = 0;
    DeltaStateWide delta_log_rebase_wide;
    int delta_log_rebase_state_kind = 0;
    uint64_t state_hash_xor = 0;
    uint64_t state_hash_sum = 0;
    uint64_t state_hash_sum_rot = 0;
    bool           state_hash_valid            = true;
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
    // TurboVec code width (2, 3 or 4); 0 for every other storage mode.
    int turbovec_bits = 0;
    // True for an index loaded from a v3 snapshot: it keeps the turbovec 0.9
    // QR rotation, codebook and first-add calibration it was written with.
    bool turbovec_legacy = false;
    // The blocked cache uses upstream's vector-major layout (1.x 4-bit
    // indexes on hosts that score with permute-dot); fixed at creation.
    bool turbovec_blocked_vm = false;
    // The blocked cache uses upstream's wider `vm8` unit instead (aarch64
    // with i8mm: 4-bit indexes whose byte-group count is a multiple of
    // eight, and 2-bit ones under TURBOVEC_2BIT_VM8=1); fixed at creation
    // and never persisted, since snapshots are written from sequential rows.
    bool turbovec_blocked_vm8 = false;
    // Created under upstream's x86 rules (turbovec_x86_rules); fixed at
    // creation. With the vector-major layout, 2- and 3-bit codes score
    // with the `vpermb` scan's arithmetic, and the staged search keeps its
    // sign region vector-major, scans it the same way and ranks with bit
    // masks, as upstream does on x86.
    bool turbovec_x86 = false;
    // Bumped by every calibrate; a sync across one rewrites the whole file.
    uint64_t turbovec_calib_gen = 0;
    TurboVecSyncBinding turbovec_sync;
    // Whether turbovec_data holds the packed rows. A 1.x index keeps its
    // codes only in the blocked cache (upstream #475) unless it is in the
    // planes layout, whose staged search reads the bit planes row by row;
    // a legacy index always keeps packed rows.
    bool turbovec_packed_rows = true;
    std::vector<uint8_t> turbovec_data;
    std::vector<float>   turbovec_scale;
    std::vector<float>   turbovec_tqplus_shift;
    std::vector<float>   turbovec_tqplus_scale;
    std::vector<uint8_t> turbovec_blocked_data;
    size_t turbovec_blocked_n_blocks = 0;

    // Staged search (upstream turbovec's planes search). Once a 2- or
    // 4-bit index reaches kTurboVecPlanesMinVectors on a host that runs it
    // the index is promoted and keeps a blocked copy of its sign planes;
    // it stays promoted if it later shrinks.
    bool turbovec_staged_enabled = true;
    bool turbovec_planes = false;
    std::vector<uint8_t> turbovec_sign_blocked;
    // Derived lazily by the first staged search, as upstream does: the
    // planes statistics (from the cache head, reset only on promotion) and
    // the per-vector centre term (reset by every mutation).
    mutable std::mutex turbovec_planes_mutex;
    mutable bool turbovec_stats_ready = false;
    mutable TurboVecPlanesStats turbovec_stats;
    mutable bool turbovec_centre_ready = false;
    mutable std::vector<float> turbovec_centre;

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
    bool data_synced = false;
};

class DeltaLogLock {
public:
  // Not reentrant for the same path in one process; blocks until the OS lock
  // is acquired or returns an error.
  explicit DeltaLogLock(const char * path, bool writable = true);
  ~DeltaLogLock();

  DeltaLogLock(const DeltaLogLock &)             = delete;
  DeltaLogLock & operator=(const DeltaLogLock &) = delete;

  bool ok() const;
  bool needs_absence_recheck() const;
  bool has_data_file() const;
  bool ensure_data_file_locked(const char * path);
  bool path_matches_data_file(const char * path) const;
  bool data_file_size(uint64_t & size) const;
  bool data_file_stamp(DeltaFileStamp & stamp) const;
  bool open_read_stream(std::FILE ** out) const;
  bool open_append_stream(const char * path, std::FILE ** out);
  bool truncate_data_file(const char * path, uint64_t size);

private:
    std::shared_ptr<std::mutex> sidecar_process_mutex;
    std::shared_ptr<std::mutex> data_process_mutex;
    std::unique_lock<std::mutex> sidecar_process_lock;
    std::unique_lock<std::mutex> data_process_lock;
    bool locked = false;
    bool                         recheck_absence = false;
#ifdef _WIN32
    HANDLE sidecar_file = INVALID_HANDLE_VALUE;
    HANDLE data_file = INVALID_HANDLE_VALUE;
    OVERLAPPED sidecar_lock_overlapped = {};
    OVERLAPPED data_lock_overlapped = {};

    bool        lock_data_file(const std::filesystem::path & path, bool create, bool writable);
    static void close_file(HANDLE & file, OVERLAPPED & overlapped);
#else
    int sidecar_fd = -1;
    int data_fd = -1;

    static bool lock_fd(int fd, bool exclusive);
    bool        lock_data_file(const std::filesystem::path & path, bool create, bool writable);
    static void close_fd(int & fd);
#endif
};

#ifdef GGML_VEC_INDEX_TEST_HOOKS
extern "C" {
int     ggml_vec_index_test_can_address_array(size_t count, size_t element_size);
void    ggml_vec_index_test_set_oom_countdown(int64_t countdown);
void    ggml_vec_index_test_set_write_fail_after(int64_t bytes);
void    ggml_vec_index_test_set_truncate_fail(int fail);
void    ggml_vec_index_test_set_data_fsync_fail(int fail);
void    ggml_vec_index_test_set_parent_fsync_fail(int fail);
void    ggml_vec_index_test_set_parent_fsync_fail_after(int64_t count);
void    ggml_vec_index_test_set_delta_append_wait_target(int target);
int     ggml_vec_index_test_get_delta_append_waiters(void);
void    ggml_vec_index_test_set_delta_append_hold(int hold);
void    ggml_vec_index_test_release_delta_append(void);
int     ggml_vec_index_test_get_sidecar_lock_probe(void);
int     ggml_vec_index_test_get_delta_append_max_active_waiters(void);
void    ggml_vec_index_test_set_load_with_delta_pause_ms(int pause_ms);
void    ggml_vec_index_test_set_load_with_delta_block(int block);
void    ggml_vec_index_test_reset_delta_tail_scan_count(void);
int64_t ggml_vec_index_test_get_delta_tail_scan_count(void);
void    ggml_vec_index_test_reset_state_crc_scan_count(void);
int64_t ggml_vec_index_test_get_state_crc_scan_count(void);
void    ggml_vec_index_test_reset_delta_max_read_size(void);
size_t  ggml_vec_index_test_get_delta_max_read_size(void);
void    ggml_vec_index_test_reset_mmap_count_reject_count(void);
int64_t ggml_vec_index_test_get_mmap_count_reject_count(void);
void    ggml_vec_index_test_reset_load_count_reject_count(void);
int64_t ggml_vec_index_test_get_load_count_reject_count(void);
int     ggml_vec_index_test_get_load_with_delta_waiters(void);
}

void test_maybe_throw_bad_alloc();
bool test_consume_write_bytes(size_t n);
void test_wait_after_delta_validate();
void test_wait_after_load_with_delta_snapshot();
void test_record_delta_read_size(size_t size);
#else
inline void test_maybe_throw_bad_alloc() {}
inline bool test_consume_write_bytes(size_t) {
    return true;
}
inline void test_wait_after_delta_validate() {}
inline void test_wait_after_load_with_delta_snapshot() {}
inline void test_record_delta_read_size(size_t) {}
#endif

bool is_supported_bit_width(int bit_width);
bool is_valid_id(uint64_t id);
bool all_finite(const float * values, size_t n);
// n_threads splits the check; the answer does not depend on it.
bool all_finite_abs_less_than(const float * values, size_t n, float max_abs, int n_threads = 1);
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
bool delta_log_path_key(const char * path, std::string & out);

void invalidate_ivf(ggml_vec_index & idx);
bool is_q8(const ggml_vec_index & idx);
bool is_q4(const ggml_vec_index & idx);
bool is_turbovec(const ggml_vec_index & idx);
// Top-k heaps order equal scores by their key. A TurboVec 1.x index keys
// on the slot, which is upstream turbovec's row order, so ties come out as
// upstream returns them; other indexes key on the external id. Keys are
// mapped back to ids when results are written.
bool topk_keys_are_slots(const ggml_vec_index & idx);
uint64_t topk_key(const ggml_vec_index & idx, size_t slot);
uint64_t topk_key_id(const ggml_vec_index & idx, uint64_t key);
bool is_turbovec_q2(const ggml_vec_index & idx);
bool is_turbovec_q4(const ggml_vec_index & idx);
bool is_quantized(const ggml_vec_index & idx);
uint8_t storage_kind(const ggml_vec_index & idx);
size_t q4_row_bytes(size_t dim);
size_t turbovec_row_bytes(int bits, size_t dim);
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
float quantization_scale(float max_abs, float divisor);
void quantize_q8_values(const float * src, int8_t * dst, size_t n, float scale);
void quantize_q4_values(const float * src, uint8_t * dst, size_t offset, size_t n, float scale);
bool turbovec_supported_dim(int dim, bool legacy);
bool turbovec_serialized_dim_supported(int dim);
ggml_vec_index_t * ggml_vec_index_create_turbovec_for_load(int dim, int bit_width, bool legacy);
void turbovec_retain_rotation(int dim, bool legacy);
void turbovec_release_rotation(int dim, bool legacy) noexcept;
#ifdef GGML_VEC_INDEX_TEST_HOOKS
uint64_t turbovec_ziggurat_table_hash_for_test(void);
double turbovec_ziggurat_x_for_test(int index);
double turbovec_ziggurat_f_for_test(int index);
double turbovec_regularized_beta_for_test(double x, double a, double b);
double turbovec_inverse_regularized_beta_for_test(double probability, double a);
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
// The blocked cache layout: 0 sequential, 1 vector-major, 2 vm8.
int turbovec_layout_for_test(const ggml_vec_index_t * idx);
void turbovec_code_bytes_for_test(
    const ggml_vec_index_t * idx, size_t * packed, size_t * blocked, size_t * sign);
int turbovec_avx2_available_for_test();
int turbovec_avx2_lut_block_matches_scalar_for_test(int bits, int dim);
void turbovec_reset_block_score_call_count_for_test(void);
int64_t turbovec_block_score_call_count_for_test(void);
void turbovec_hadamard_rotate_for_test(const float * src, float * dst, int dim);
int turbovec_rotation_kernel_parity_for_test(int kernel);
int turbovec_encode_kernel_parity_for_test(int kernel);
int turbovec_layout_kernel_parity_for_test(int kernel);
int vector_validation_kernel_parity_for_test(int kernel);
int crc32c_kernel_parity_for_test(int kernel);
// The x86 search kernels against their scalar models (kernels 0-7, see
// ggml-vector-index-turbovec.cpp); 1 match, 0 mismatch, -1 unavailable.
int turbovec_search_kernel_parity_for_test(int kernel);
// The staged search's AVX-512 bit-mask build (0) and VPOPCNTDQ count (1).
int turbovec_low_kernel_parity_for_test(int kernel);
// Upstream's x86 `vpermb`-scan search computed slot by slot from the code
// bytes (ggml-vector-index-search.cpp); 0 when the index does not score so.
int turbovec_vpermb_reference_search_for_test(
    const ggml_vec_index_t * idx,
    const float * queries,
    int n_q,
    int k,
    const uint64_t * allowed,
    int n_allowed,
    float * out_scores,
    uint64_t * out_ids);
int turbovec_sync_plan_for_test(
    ggml_vec_index_t * idx,
    const char * path,
    std::vector<std::vector<std::pair<uint64_t, std::vector<uint8_t>>>> & batches);
void turbovec_force_vector_major_for_test(int mode);
void turbovec_disable_dot_product_for_test(int disabled);
int vec_index_simd_kernel_built_for_test(int id);
int turbovec_pd_finish_parity_for_test();
// -1 follows the build, 0 or 1 forces upstream's aarch64 or x86 rules for
// indexes created afterwards (turbovec_x86_rules).
void turbovec_force_x86_rules_for_test(int mode);
// -1 follows the host, 0 or 1 forces the `vm8` choice for indexes created
// afterwards (still only on top of the vector-major layout).
void turbovec_force_vm8_for_test(int mode);
// Keeps SMMLA off even on an i8mm host; kernels fall back to SDOT or scalar.
void turbovec_disable_i8mm_for_test(int disabled);
// -1 follows TURBOVEC_2BIT_VM8, 0 or 1 overrides it.
void turbovec_force_vm8_2bit_for_test(int mode);
// Fused batches for every query count upstream batches, filters and
// small indexes included, so a test reaches the batch kernels.
void turbovec_force_batches_for_test(int enabled);
void turbovec_force_planes_for_test(int mode, size_t min_vectors);
size_t parallel_pool_threads_for_test(void);
void state_hash_for_test(const ggml_vec_index_t * idx, uint64_t * sums);
int parallel_pool_selftest_for_test(int n_threads);
int32_t turbovec_low_planes_for_test(const float * q, int dim, float m, float * unit, std::vector<uint64_t> * masks);
int turbovec_index_hashes_for_test(
    const ggml_vec_index_t * idx,
    uint64_t * codes_hash,
    uint64_t * scales_hash,
    uint64_t * shift_hash,
    uint64_t * scale_hash);
#endif
// Whether this CPU runs upstream turbovec's dot-product ("vector-major")
// kernels: aarch64 with dot product, x86 with AVX-512 F/BW/VBMI/VNNI. Those
// hosts score 4-bit codes with int8 permute-dot arithmetic.
bool turbovec_cpu_vector_major();
// The CPU's own dot-product support, which a test override cannot change:
// gates the SDOT instructions themselves.
bool turbovec_cpu_dot_product();
// Runtime CPU features gating the rotation, encode, layout and checksum
// kernels, as upstream turbovec gates them with is_x86_feature_detected!.
struct VecIndexCpuFeatures {
    bool ssse3 = false;
    bool sse42 = false;
    // AVX2 with FMA: the AVX2 object is compiled with both.
    bool avx2 = false;
    // AVX-512 F, only together with AVX2.
    bool avx512f = false;
    // AVX-512 extensions, each only together with AVX-512 F. GFNI counts
    // only there too: the search kernels use its 512-bit form.
    bool avx512bw = false;
    bool avx512vbmi = false;
    bool avx512vnni = false;
    bool avx512vpopcntdq = false;
    bool gfni = false;
    bool arm_crc32 = false;
};
const VecIndexCpuFeatures & vec_index_cpu_features();
// Upstream `have_gfni`: the GFNI form of the x86 permute-dot kernel, unless
// TURBOVEC_NO_GFNI is set. Both forms sum the same integers.
bool turbovec_cpu_gfni();
// Whether new indexes follow upstream's x86 rules rather than its aarch64
// ones: on a vector-major host the 2- and 3-bit layouts are vector-major
// too (scored by the `vpermb` scan), the staged search's sign region is
// vector-major and scanned the same way, and its ranking counts bit masks.
// True on x86 builds; a test may force it anywhere, so the scalar models
// of those kernels run on any host.
bool turbovec_x86_rules();

// The CPU's int8 matrix multiply (aarch64 i8mm): gates the SMMLA
// instructions themselves.
bool turbovec_cpu_i8mm();
// Whether new indexes take upstream's `vm8` layout (pack.rs `use_vm8`).
bool turbovec_cpu_vm8();
// Upstream's TURBOVEC_2BIT_VM8=1 opt-in (pack.rs `use_vm8_2bit`), read
// once per process: 2-bit indexes take the `vm8` layout where `vm8` is
// available, batches score them with SMMLA, and on aarch64 the staged
// (planes) search is off.
bool turbovec_vm8_2bit_enabled();

// In-memory layout of a TurboVec blocked cache.
enum TurboVecLayout : uint8_t {
    // Byte group by byte group, 32 lanes each (x86: perm0 nibble interleave).
    kTurboVecLayoutSeq = 0,
    // Upstream `vm_byte_index`: four byte groups of a vector adjacent.
    kTurboVecLayoutVm = 1,
    // Upstream `vm8_byte_index`: eight byte groups of a vector adjacent, two
    // vectors per 16 bytes, so a register is an SMMLA operand as loaded.
    kTurboVecLayoutVm8 = 2,
};
inline TurboVecLayout turbovec_layout(const ggml_vec_index & idx) {
    return idx.turbovec_blocked_vm8 ? kTurboVecLayoutVm8 :
        idx.turbovec_blocked_vm ? kTurboVecLayoutVm : kTurboVecLayoutSeq;
}
// The staged search's sign region (dim / 8 byte groups): vector-major
// under upstream's x86 rules (pack.rs `planes_sign_permuted`), sequential
// otherwise.
inline TurboVecLayout turbovec_sign_layout(const ggml_vec_index & idx) {
    return idx.turbovec_x86 && idx.turbovec_blocked_vm && (idx.dim / 8) % 4 == 0 ? kTurboVecLayoutVm :
                                                                                  kTurboVecLayoutSeq;
}
// Whether this index's codes score with the `vpermb` scan's arithmetic
// (upstream x86, 2 and 3 bits in the vector-major layout): exact u32 sums
// of the nibble tables, then sum * scale + bias, unfused.
inline bool turbovec_scores_vnni_lut(const ggml_vec_index & idx) {
    return idx.turbovec_blocked_vm && !idx.turbovec_blocked_vm8 && idx.turbovec_bits != 4;
}
// The layouts upstream picks for a new 1.x index of this geometry.
bool turbovec_wants_vector_major(int bits, int dim);
bool turbovec_wants_vm8(int bits, int dim);

// One query quantized for the 4-bit permute-dot score (upstream
// `QueryPermuteDot`): sum_d weight[d] * level[code[d]] in int32, then
// fma(sum, scale, bias) times the vector's scale.
struct TurboVecPermuteDot {
    int8_t levels[16] = {};
    // 2-bit operand set only (upstream H72): levels[i] = level[i & 3] and
    // levels2[i] = level[(i >> 2) & 3].
    int8_t levels2[16] = {};
    // One int8 weight per dimension, in dimension order.
    std::vector<int8_t> weights;
    // The lone-query weights over the vm8 layout (4-bit, when the
    // byte-group count is a multiple of eight).
    std::vector<int8_t> vm8_weights;
    // Builds with the AVX-512 VNNI kernels, 4-bit: the weights in
    // upstream's own order (per four byte groups, the four odd dims then
    // the four even ones) and -128 times their sum, for those kernels.
    std::vector<int8_t> x86_weights;
    int32_t zero = 0;
    float scale = 1.0f;
    float bias = 0.0f;
};
// vm8 builds the lone-query vm8 weights, which only a vm8-layout index reads.
void build_turbovec_permute_dot(const float * query, int dim, TurboVecPermuteDot & out, bool vm8);
// The 2-bit SMMLA operand set of a 2-bit vm8 index (upstream
// `build_permute_dot_2bit`).
void build_turbovec_permute_dot_2bit(const float * query, int dim, TurboVecPermuteDot & out);
void score_turbovec_permute_dot_block(
    const TurboVecPermuteDot & pd,
    const uint8_t * blocked_codes,
    const float * vector_scales,
    size_t block_index,
    size_t n_vectors,
    int dim,
    TurboVecLayout layout,
    float * out_scores);
// Upstream's fused permute-dot batches: 4, 8 or 12 queries over one block,
// sharing the code loads (pd_scan widths). 4-bit codes in the vm or vm8
// layout, or 2-bit codes in vm8 (TURBOVEC_2BIT_VM8).
inline constexpr int kTurboVecMaxBatch = 12;
struct TurboVecPermuteDotBatch {
    int n = 0;
    int bits = 4;
    TurboVecLayout layout = kTurboVecLayoutSeq;
    const TurboVecPermuteDot * pds[kTurboVecMaxBatch] = {};
    // SMMLA A operands for the batch, built once by prepare.
    std::vector<int8_t> smmla_a;
};
void prepare_turbovec_permute_dot_batch(TurboVecPermuteDotBatch & batch, int dim);
void score_turbovec_permute_dot_batch_block(
    const TurboVecPermuteDotBatch & batch,
    const uint8_t * blocked_codes,
    const float * vector_scales,
    size_t block_index,
    size_t n_vectors,
    int dim,
    float * const * out_scores);
// score_turbovec_permute_dot_block over n_group (up to eight) consecutive
// blocks, out_scores[b * 32 + lane]: on x86 one pass interleaves them, as
// upstream's lone query does.
void score_turbovec_permute_dot_blocks(
    const TurboVecPermuteDot & pd,
    const uint8_t * blocked_codes,
    const float * vector_scales,
    size_t first_block,
    size_t n_group,
    size_t n_vectors,
    int dim,
    TurboVecLayout layout,
    float * out_scores);
// Upstream `split_lut_for_vnni`: a query's nibble tables reordered for the
// `vpermb` scan (n_byte_groups a multiple of four, n_byte_groups * 32
// bytes).
void turbovec_split_lut_for_vnni(const uint8_t * lut, size_t n_byte_groups, uint8_t * out);
// The `vpermb` scan (turbovec_scores_vnni_lut) of nq queries (up to 8)
// over n_group consecutive vector-major blocks, out_scores[q][b * 32 +
// lane]. `splits` (each query's split tables) may be null: the AVX-512
// kernel needs them, the scalar model reads `luts`.
void score_turbovec_lut_blocks_vnni(
    const uint8_t * const * luts,
    const uint8_t * const * splits,
    const float * scales,
    const float * biases,
    int nq,
    const uint8_t * blocked_codes,
    const float * vector_scales,
    size_t first_block,
    size_t n_group,
    size_t n_vectors,
    int bits,
    int dim,
    float * const * out_scores);
void score_turbovec_lut_block_vnni(
    const uint8_t * lut,
    const uint8_t * split,
    float lut_scale,
    float lut_bias,
    const uint8_t * blocked_codes,
    const float * vector_scales,
    size_t block_index,
    size_t n_vectors,
    int bits,
    int dim,
    float * out_scores);
// One vector's 4-bit permute-dot sum from a planes cache (upstream
// `exact4_sum`): its sign plane from the vector-major sign block, its low
// planes from its packed row; `w` is upstream's `Exact4` weight layout.
int32_t turbovec_exact4_sum(
    const int8_t * w,
    const int8_t * levels,
    int32_t zero,
    const uint8_t * sign_block,
    const uint8_t * low_row,
    size_t nsg,
    size_t lane);
int32_t turbovec_exact4_sum_scalar(
    const int8_t * w,
    const int8_t * levels,
    const uint8_t * sign_block,
    const uint8_t * low_row,
    size_t nsg,
    size_t lane);
bool turbovec_uses_permute_dot(const ggml_vec_index_t & idx);
void score_turbovec_block(
    const ggml_vec_index_t & idx,
    const uint8_t * lut,
    float lut_scale,
    float lut_bias,
    const TurboVecPermuteDot * permute_dot,
    size_t block,
    float * out_scores,
    const uint8_t * split = nullptr);

// Staged search (ggml-vector-index-turbovec-staged.cpp).
inline constexpr size_t kTurboVecPlanesMinVectors = 32768;
bool turbovec_planes_for(int bits, int dim);
bool turbovec_planes_wanted(int bits, int dim, size_t n_vectors);
// After an add (first new slot), a load or calibrate (0): promotes the
// index or patches its sign planes.
void turbovec_planes_refresh(ggml_vec_index & idx, size_t first_slot);
void turbovec_planes_reset(ggml_vec_index & idx);
void turbovec_planes_swap_remove(ggml_vec_index & idx, size_t slot, size_t last);
// Searches one query (rotated, TQ+-calibrated) in stages into `heap`;
// returns false when the index is not set up for it.
// How a staged search is run, following upstream's batch bookkeeping.
struct TurboVecStagedOpts {
    // The call searches one query: on aarch64 its sign tables cap at 31.
    bool single_query = false;
    // The shortlist scan alone uses a lone query's tables (upstream's
    // rescan of the one query in a batch whose sampled seed ran short).
    bool sign_single = false;
    // Set to whether upstream's sampled seed would have left this query's
    // shortlist short; false when upstream does not seed.
    bool * seed_short = nullptr;
    // Threads the one query may split its blocks over (a batch already
    // splits across queries and passes 1). The results do not depend on it.
    int n_threads = 1;
};
bool turbovec_staged_search(
    const ggml_vec_index & idx,
    const float * query,
    float bias_corr,
    int k,
    const uint8_t * allowed_mask,
    const TurboVecStagedOpts & opts,
    std::vector<ScoreId> & heap);
void prepare_turbovec(int bits, int dim, bool legacy);
void rotate_turbovec_query(bool legacy, const float * src, float * dst, int dim);
void rotate_turbovec_queries(
    bool legacy,
    const float * src,
    float * dst,
    int n_queries,
    int dim);
void decode_turbovec_row_calibrated(
    int bits,
    bool legacy,
    const uint8_t * codes,
    const float * scales,
    const float * tqplus_shift,
    const float * tqplus_scale,
    float * dst,
    int dim);
void build_turbovec_lut(
    int bits,
    bool legacy,
    const float * rotated_query,
    int dim,
    std::vector<uint8_t> & lut,
    float & scale,
    float & bias);
float dot_turbovec_bits_lut_row(
    const uint8_t * lut,
    float lut_scale,
    float lut_bias,
    const uint8_t * codes,
    const float * scales,
    int bits,
    int dim);
void quantize_turbovec_q2_row(const float * src, uint8_t * dst, float * scales, int dim);
void quantize_turbovec_batch(
    const float * src,
    int n,
    int bits,
    uint8_t * dst,
    float * scales,
    int dim,
    std::vector<float> & tqplus_shift,
    std::vector<float> & tqplus_scale);
void quantize_turbovec_hadamard_batch(
    const float * src,
    int n,
    int bits,
    uint8_t * dst,
    float * scales,
    int dim,
    const std::vector<float> & tqplus_shift,
    const std::vector<float> & tqplus_scale,
    int n_threads);
int fit_turbovec_calibration(
    const float * sample,
    int n,
    int bits,
    int dim,
    std::vector<float> & out_shift,
    std::vector<float> & out_scale,
    int n_threads);
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
    int n_threads);
void turbovec_codebook_values(
    int bits,
    int dim,
    bool legacy,
    std::vector<float> & boundaries,
    std::vector<float> & centroids);
// Packed bit-plane row <-> sequential row (one byte per group of 8 / bits
// codes, 3-bit codes in 4-bit fields), the code layout upstream files store.
void turbovec_packed_row_to_seq(const uint8_t * packed, int bits, int dim, uint8_t * seq);
void turbovec_seq_row_to_packed(const uint8_t * seq, int bits, int dim, uint8_t * packed);

// Upstream turbovec v8 snapshot pieces: the superblock, the used prefix of
// header slot 0, and the block units. The full image is superblock, two
// header_len slots (slot 0 starts with header0, the rest is zero), units.
// Units are produced on demand, so a writer never holds them all at once;
// `unit` reads the index and is only valid while its lock is held.
struct TurboVecV8Image {
    std::vector<uint8_t> superblock;
    std::vector<uint8_t> header0;
    size_t header_len = 0;
    size_t n_units = 0;
    size_t unit_len = 0;
    std::function<void(size_t block, uint8_t * out)> unit;
};
bool turbovec_v8_magic(const uint8_t * bytes, size_t size);
uint64_t turbovec_v8_declared_size(std::istream & f, uint64_t file_size);
void turbovec_v8_build_image(const ggml_vec_index & idx, TurboVecV8Image & image);
size_t turbovec_v8_image_size(const TurboVecV8Image & image);
size_t turbovec_v8_serialized_size(const ggml_vec_index & idx);
// Writes the whole image into out (turbovec_v8_image_size bytes, zeroed).
void turbovec_v8_fill_image(const TurboVecV8Image & image, uint8_t * out);
struct TurboVecV8LoadInfo {
    uint64_t gen = 0;
    uint64_t nonce = 0;
    // File rows with redo ops in the adopted commit.
    std::vector<size_t> pending;
};
ggml_vec_index_t * turbovec_v8_load_image(std::vector<uint8_t> raw, int & error, TurboVecV8LoadInfo * info);

// Positioned writes made durable together (one fsync per batch).
using TurboVecWriteBatch = std::vector<std::pair<uint64_t, std::vector<uint8_t>>>;
int turbovec_v8_write_file(const char * path, const TurboVecV8Image & image, bool durable = true);
int turbovec_v8_apply_batches(const char * path, const std::vector<TurboVecWriteBatch> & batches);
// Reads up to `size` bytes at `offset`; GGML_VEC_INDEX_E_NOT_FOUND when the
// file does not exist.
int turbovec_read_file_range(
    const char * path,
    uint64_t offset,
    size_t size,
    std::vector<uint8_t> & out,
    uint64_t & file_size);
void turbovec_sync_bind_loaded(ggml_vec_index & idx, const char * path, const TurboVecV8LoadInfo & info);
void turbovec_sync_unbind(ggml_vec_index & idx) noexcept;
// Records a swap-remove of `slot` (the last slot moves into it) for the
// next sync's redo ops.
void turbovec_sync_note_remove(ggml_vec_index & idx, size_t slot) noexcept;
int turbovec_v8_sync(ggml_vec_index & idx, const char * path);
void decode_turbovec_q2_row(const uint8_t * codes, const float * scales, float * dst, int dim);
void decode_turbovec_q2_row_calibrated(
    const uint8_t * codes,
    const float * scales,
    const float * tqplus_shift,
    const float * tqplus_scale,
    float * dst,
    int dim);
void build_turbovec_q2_lut(const float * rotated_query, int dim, std::vector<uint8_t> & lut, float & scale, float & bias);
float dot_turbovec_q2_lut_row(const uint8_t * lut, float lut_scale, float lut_bias, const uint8_t * codes, const float * scales, int dim);
float dot_turbovec_q2_rotated_row(const float * rotated_query, const uint8_t * codes, const float * scales, int dim);
float dot_turbovec_q2_row(const float * query, const uint8_t * codes, const float * scales, int dim);
void quantize_turbovec_q4_row(const float * src, uint8_t * dst, float * scales, int dim);
void decode_turbovec_q4_row(const uint8_t * codes, const float * scales, float * dst, int dim);
void decode_turbovec_q4_row_calibrated(
    const uint8_t * codes,
    const float * scales,
    const float * tqplus_shift,
    const float * tqplus_scale,
    float * dst,
    int dim);
void build_turbovec_q4_lut(const float * rotated_query, int dim, std::vector<uint8_t> & lut, float & scale, float & bias);
float dot_turbovec_q4_lut_row(const uint8_t * lut, float lut_scale, float lut_bias, const uint8_t * codes, const float * scales, int dim);
float dot_turbovec_q4_rotated_row(const float * rotated_query, const uint8_t * codes, const float * scales, int dim);
float dot_turbovec_q4_row(const float * query, const uint8_t * codes, const float * scales, int dim);
// Patches the blocked cache for a swap-remove of `slot` whose last slot
// (`last`, already removed from the rows) moved into it.
void turbovec_blocked_swap_remove(ggml_vec_index & idx, size_t slot, size_t last);
// Drops the blocked cache's rows from n on, zeroing the tail block's lanes.
void turbovec_blocked_truncate(ggml_vec_index & idx, size_t n);
// Code access that works whether the codes live in packed rows (legacy) or
// only in the blocked cache (1.x).
bool turbovec_rows_in_blocked(const ggml_vec_index & idx);
// Materializes the packed rows from the blocked cache (planes promotion),
// and drops them again once only the blocked cache is needed.
void turbovec_materialize_packed_rows(ggml_vec_index & idx);
void turbovec_settle_rows(ggml_vec_index & idx);
void turbovec_row_seq(const ggml_vec_index & idx, size_t slot, uint8_t * seq);
void turbovec_row_packed(const ggml_vec_index & idx, size_t slot, uint8_t * packed);
// The slot's packed row without copying when the index keeps packed rows (a
// legacy row can be wider than any 1.x row); a 1.x row is read back from the
// blocked cache into `scratch`, which must hold kTurboVecMaxDim / 2 bytes.
const uint8_t * turbovec_row_packed_ptr(const ggml_vec_index & idx, size_t slot, uint8_t * scratch);
void turbovec_rows_packed(const ggml_vec_index & idx, std::vector<uint8_t> & packed);
// The packed rows of one 32-row block (fewer for the tail block).
// `scratch` holds turbovec_block_scratch_bytes(idx) bytes.
size_t turbovec_block_scratch_bytes(const ggml_vec_index & idx);
void turbovec_block_rows_packed(const ggml_vec_index & idx, size_t block, uint8_t * packed, uint8_t * scratch);
// Writes one row's sequential codes into its blocked-cache lane.
void turbovec_blocked_set_row_seq(ggml_vec_index & idx, size_t slot, const uint8_t * seq);
// Whole 32-row blocks in the sequential-blocked layout (byte g of lane l at
// g * 32 + l: the v7/v8 unit codes), read from or written to the blocked
// cache in its in-memory layout (upstream pack.rs `native_to_seq` /
// `seq_into_native`). The block must lie inside the cache.
void turbovec_block_seq(const ggml_vec_index & idx, size_t block, uint8_t * seq_blocked);
void turbovec_blocked_set_block_seq(ggml_vec_index & idx, size_t block, const uint8_t * seq_blocked);
// Writes n_new packed rows into the blocked cache from first_slot on.
void turbovec_blocked_append(ggml_vec_index & idx, const uint8_t * packed, size_t first_slot, size_t n_new);
// The sign planes of rows [first_slot, n_vectors) into the sign blocked
// cache (upstream's planes sign region), growing or shrinking it to cover
// n_vectors rows.
void repack_turbovec_sign_codes(
    const uint8_t * packed_codes,
    size_t n_vectors,
    int bits,
    int dim,
    size_t first_slot,
    std::vector<uint8_t> & sign_blocked,
    int n_threads = 1,
    TurboVecLayout layout = kTurboVecLayoutSeq);
void turbovec_sign_blocked_swap_remove(ggml_vec_index & idx, size_t slot, size_t last);
// Blocks are independent, so n_threads does not change the bytes.
void repack_turbovec_codes(
    const uint8_t * packed_codes,
    size_t n_vectors,
    int bits,
    int dim,
    std::vector<uint8_t> & blocked_codes,
    size_t & n_blocks,
    TurboVecLayout layout = kTurboVecLayoutSeq,
    int n_threads = 1);
void repack_turbovec_codes_from_slot(
    const uint8_t * packed_codes,
    size_t n_vectors,
    int bits,
    int dim,
    size_t first_slot,
    std::vector<uint8_t> & blocked_codes,
    size_t & n_blocks,
    TurboVecLayout layout = kTurboVecLayoutSeq,
    int n_threads = 1);
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
    float * out_scores);
// score_turbovec_lut_block over a vm8 block (2-bit indexes under
// TURBOVEC_2BIT_VM8), bit-identical to it on the same codes.
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
    float * out_scores);
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
    float * const out_scores[4]);
// score_turbovec_lut_block_4 over blocks block_index and block_index + 1
// (upstream's AVX-512 BW kernel takes them in one pass).
void score_turbovec_lut_block_pair_4(
    const uint8_t * const luts[4],
    const float lut_scales[4],
    const float lut_biases[4],
    const uint8_t * blocked_codes,
    const float * vector_scales,
    size_t block_index,
    size_t n_vectors,
    int bits,
    int dim,
    float * const out_scores0[4],
    float * const out_scores1[4]);
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
// turbovec_row, when given, is the slot's packed TurboVec row (saves
// reading it back from the blocked cache).
uint64_t slot_state_hash(const ggml_vec_index & idx, size_t slot, const uint8_t * turbovec_row = nullptr);
void add_state_hash(ggml_vec_index & idx, uint64_t hash);
// Adds the hashes of the active slots [first, last) over idx.n_threads;
// `rows`, when given, holds their packed TurboVec rows. The hash sums are
// commutative, so the result does not depend on the split.
void add_state_hash_range(ggml_vec_index & idx, size_t first, size_t last, const uint8_t * rows);
void remove_state_hash(ggml_vec_index & idx, uint64_t hash);
void rebuild_state_hash(ggml_vec_index & idx);
uint32_t index_state_token(const ggml_vec_index & idx);
uint32_t index_state_token_after_remove(const ggml_vec_index & idx, uint64_t id);
DeltaStateWide index_state_wide(const ggml_vec_index & idx);
DeltaStateWide index_state_wide_after_remove(const ggml_vec_index & idx, uint64_t id);

DeltaStateKind delta_state_kind_for_format(DeltaLogFormat format);
DeltaLogFormat delta_log_format_for_append(const char * path, const DeltaLogLock * lock = nullptr);
bool prepare_delta_log_format_for_append_unlocked(
    ggml_vec_index_t * idx,
    const char * delta_path,
    DeltaLogLock & lock,
    DeltaLogFormat & format);
uint32_t current_delta_state(const ggml_vec_index & idx, DeltaStateKind state_kind);
DeltaStateWide current_delta_state_wide(const ggml_vec_index & idx);
void invalidate_delta_tail_cache(ggml_vec_index & idx) noexcept;
bool bind_delta_log_path(ggml_vec_index & idx, const char * delta_path);
bool delta_log_matches_index_unlocked(
    const ggml_vec_index_t * idx,
    const char * delta_path,
    DeltaLogLock * lock);
bool replay_delta_log_unlocked(ggml_vec_index_t * idx, const char * delta_path, DeltaLogLock & lock);
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
DeltaAppendResult    append_delta_record_locked(ggml_vec_index &             idx,
                                                DeltaLogLock &               lock,
                                                const char *                 delta_path,
                                                DeltaLogFormat               format,
                                                uint8_t                      op,
                                                uint32_t                     n,
                                                uint32_t                     base_crc_for_new_log,
                                                uint32_t                     state_crc,
                                                const DeltaStateWide &       base_wide_for_new_log,
                                                const DeltaStateWide &       state_wide,
                                                const std::vector<uint8_t> & payload);
