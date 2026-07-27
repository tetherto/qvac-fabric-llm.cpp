// ggml-vector-index.cpp - POC scalar implementation of the fabric vector
// index C API declared in `ggml/include/ggml-vector-index.h`.
//
// Storage: full f32 vectors as a contiguous std::vector<float>. ID map uses
// std::unordered_map<uint64_t, size_t> for lookup and a parallel vector for
// the slot->id reverse map. Remove uses swap-with-last.
//
// Search: naive scalar dot product across all slots + min-heap of size k.
// No SIMD, no GPU. Correctness over speed; the optimization phase will swap
// the storage layout and kernel without touching the C API.

#include "ggml-vector-index.h"
#include "ggml-vector-index-impl.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef _WIN32
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <fcntl.h>
#    include <io.h>
#    include <windows.h>
#else
#    include <fcntl.h>
#    include <sys/stat.h>
#    include <unistd.h>
#endif

namespace {

using ggml_vec_index_detail::checked_mul_size;
using ggml_vec_index_detail::can_address_array;
using ggml_vec_index_detail::kTvimHeaderSize;
using ggml_vec_index_detail::snapshot_write_preflight;
using ggml_vec_index_detail::supported_snapshot_size;

constexpr uint8_t  kTvimMagic[4]   = { 'T', 'V', 'P', 'I' };
constexpr uint8_t  kTvimVersion    = 1;
constexpr uint64_t kPaddingId      = UINT64_MAX;

static_assert(sizeof(float) == sizeof(uint32_t) && std::numeric_limits<float>::is_iec559,
              "ggml-vector-index requires IEEE 754 float32");

size_t grow_capacity(size_t current, size_t required, size_t max_capacity) {
    if (current >= required) {
        return current;
    }

    size_t grown = current == 0 ? required : current + current / 2;
    if (grown < current || grown < required) {
        grown = required;
    }
    return std::min(grown, max_capacity);
}

bool can_insert_without_rehash(const std::unordered_map<uint64_t, size_t> & map, size_t n) {
    const long double bucket_capacity =
        static_cast<long double>(map.bucket_count()) * static_cast<long double>(map.max_load_factor());
    return static_cast<long double>(n) <= bucket_capacity;
}

bool all_finite(const float * values, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(values[i])) {
            return false;
        }
    }
    return true;
}

float float_score_from_double(double score) {
    if (std::isnan(score)) {
        return -FLT_MAX;
    }
    if (!std::isfinite(score)) {
        return score < 0.0 ? -FLT_MAX : FLT_MAX;
    }
    if (score > static_cast<double>(FLT_MAX)) {
        return FLT_MAX;
    }
    if (score < -static_cast<double>(FLT_MAX)) {
        return -FLT_MAX;
    }
    return static_cast<float>(score);
}

double rank_score_from_double(double score) {
    return std::isnan(score) ? -std::numeric_limits<double>::infinity() : score;
}

uint64_t process_id() {
#ifdef _WIN32
    return static_cast<uint64_t>(GetCurrentProcessId());
#else
    return static_cast<uint64_t>(getpid());
#endif
}

std::filesystem::path make_tmp_path(const std::filesystem::path & dst_path) {
    static std::atomic<uint32_t> counter{ 0 };

    const uint32_t count = counter.fetch_add(1, std::memory_order_relaxed);
    const uint64_t ticks = static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());

    std::filesystem::path tmp_path = dst_path;
    tmp_path += ".tmp.";
    tmp_path += std::to_string(process_id());
    tmp_path += ".";
    tmp_path += std::to_string(count);
    tmp_path += ".";
    tmp_path += std::to_string(ticks);
    return tmp_path;
}

struct tmp_file_guard {
    explicit tmp_file_guard(const std::filesystem::path & path) : path(path) {}

    ~tmp_file_guard() {
        if (active) {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
    }

    void dismiss() { active = false; }

    std::filesystem::path path;
    bool                  active = true;
};

struct file_closer {
    void operator()(std::FILE * file) const {
        std::fclose(file);
    }
};

#ifndef _WIN32
bool sync_file(std::FILE * file) {
    const int fd = fileno(file);
#if defined(__APPLE__) && defined(F_FULLFSYNC)
    int result;
    do {
        result = fcntl(fd, F_FULLFSYNC);
    } while (result != 0 && errno == EINTR);
    if (result == 0) {
        return true;
    }
    if (errno != ENOTSUP && errno != ENOTTY && errno != EINVAL) {
        return false;
    }
#endif
    return fsync(fd) == 0;
}

bool sync_parent_directory(const std::filesystem::path & path) {
    std::filesystem::path dir_path = path.parent_path();
    if (dir_path.empty()) {
        dir_path = ".";
    }

#ifdef O_DIRECTORY
    const int fd = open(dir_path.c_str(), O_RDONLY | O_DIRECTORY);
#else
    const int fd = open(dir_path.c_str(), O_RDONLY);
#endif
    if (fd == -1) {
        return false;
    }

    const bool sync_ok = fsync(fd) == 0;
    close(fd);
    return sync_ok;
}
#endif

enum class replace_result {
    success,
    failure,
    not_durable,
};

replace_result replace_file(const std::filesystem::path & tmp_path, const std::filesystem::path & dst_path) {
#ifdef _WIN32
    const std::wstring tmp_native = tmp_path.wstring();
    const std::wstring dst_native = dst_path.wstring();
    return MoveFileExW(tmp_native.c_str(), dst_native.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0
               ? replace_result::success
               : replace_result::failure;
#else
    std::error_code ec;
    std::filesystem::rename(tmp_path, dst_path, ec);
    if (ec) {
        return replace_result::failure;
    }
    return sync_parent_directory(dst_path) ? replace_result::success : replace_result::not_durable;
#endif
}

std::FILE * open_exclusive(const std::filesystem::path & path) {
#ifdef _WIN32
    const HANDLE handle =
        CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE) {
        return nullptr;
    }
    const int fd = _open_osfhandle(reinterpret_cast<intptr_t>(handle), _O_BINARY | _O_WRONLY);
    if (fd == -1) {
        CloseHandle(handle);
        return nullptr;
    }
    std::FILE * file = _fdopen(fd, "wb");
    if (file == nullptr) {
        _close(fd);
    }
    return file;
#else
    const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd == -1) {
        return nullptr;
    }
    std::FILE * file = fdopen(fd, "wb");
    if (file == nullptr) {
        close(fd);
    }
    return file;
#endif
}

#ifdef _WIN32
bool copy_permissions_if_exists(const std::filesystem::path & src_path, const std::filesystem::path & dst_path) {
    std::error_code                    ec;
    const std::filesystem::file_status src_status = std::filesystem::status(src_path, ec);
    if (ec) {
        std::error_code exists_ec;
        const bool      src_exists = std::filesystem::exists(src_path, exists_ec);
        return !exists_ec && !src_exists;
    }
    if (!std::filesystem::exists(src_status)) {
        return true;
    }

    std::filesystem::permissions(dst_path, src_status.permissions(), std::filesystem::perm_options::replace, ec);
    return !ec;
}
#else
bool default_file_mode(const std::filesystem::path & dst_path, mode_t & mode) {
    // An empty probe observes umask and default ACLs without changing process-global state.
    std::filesystem::path probe_path = make_tmp_path(dst_path);
    probe_path += ".mode";

    const int fd = open(probe_path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0666);
    if (fd == -1) {
        return false;
    }
    tmp_file_guard probe_guard(probe_path);

    struct stat probe_stat;
    const bool  stat_ok  = fstat(fd, &probe_stat) == 0;
    const bool  close_ok = close(fd) == 0;
    if (!stat_ok || !close_ok) {
        return false;
    }
    mode = probe_stat.st_mode & 0777;
    return true;
}

bool set_snapshot_permissions(std::FILE * file, const std::filesystem::path & dst_path) {
    struct stat dst_stat;
    mode_t      mode = 0;
    if (stat(dst_path.c_str(), &dst_stat) == 0) {
        mode = dst_stat.st_mode & 07777;
    } else if (errno != ENOENT || !default_file_mode(dst_path, mode)) {
        return false;
    }
    return fchmod(fileno(file), mode) == 0;
}
#endif

void put_u32_le(uint8_t * dst, uint32_t v) {
    dst[0] = static_cast<uint8_t>(v >> 0);
    dst[1] = static_cast<uint8_t>(v >> 8);
    dst[2] = static_cast<uint8_t>(v >> 16);
    dst[3] = static_cast<uint8_t>(v >> 24);
}

void put_u64_le(uint8_t * dst, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        dst[i] = static_cast<uint8_t>(v >> (8 * i));
    }
}

uint32_t get_u32_le(const uint8_t * src) {
    return (static_cast<uint32_t>(src[0]) << 0) | (static_cast<uint32_t>(src[1]) << 8) |
           (static_cast<uint32_t>(src[2]) << 16) | (static_cast<uint32_t>(src[3]) << 24);
}

uint64_t get_u64_le(const uint8_t * src) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= static_cast<uint64_t>(src[i]) << (8 * i);
    }
    return v;
}

uint32_t float_to_u32(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    return bits;
}

float u32_to_float(uint32_t bits) {
    float v;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

bool host_is_little_endian() {
    const uint16_t value = 1;
    uint8_t        bytes[sizeof(value)];
    std::memcpy(bytes, &value, sizeof(value));
    return bytes[0] == 1;
}

bool write_f32_array_le(std::FILE * f, const std::vector<float> & values) {
    if (values.empty()) {
        return true;
    }
    if (host_is_little_endian()) {
        return std::fwrite(values.data(), sizeof(float), values.size(), f) == values.size();
    }

    constexpr size_t     kChunkElements = 4096;
    std::vector<uint8_t> bytes(kChunkElements * sizeof(uint32_t));
    for (size_t offset = 0; offset < values.size(); offset += kChunkElements) {
        const size_t count = std::min(kChunkElements, values.size() - offset);
        for (size_t i = 0; i < count; ++i) {
            put_u32_le(bytes.data() + i * sizeof(uint32_t), float_to_u32(values[offset + i]));
        }
        const size_t byte_count = count * sizeof(uint32_t);
        if (std::fwrite(bytes.data(), 1, byte_count, f) != byte_count) {
            return false;
        }
    }
    return true;
}

bool write_u64_array_le(std::FILE * f, const std::vector<uint64_t> & values) {
    if (values.empty()) {
        return true;
    }
    if (host_is_little_endian()) {
        return std::fwrite(values.data(), sizeof(uint64_t), values.size(), f) == values.size();
    }

    constexpr size_t     kChunkElements = 4096;
    std::vector<uint8_t> bytes(kChunkElements * sizeof(uint64_t));
    for (size_t offset = 0; offset < values.size(); offset += kChunkElements) {
        const size_t count = std::min(kChunkElements, values.size() - offset);
        for (size_t i = 0; i < count; ++i) {
            put_u64_le(bytes.data() + i * sizeof(uint64_t), values[offset + i]);
        }
        const size_t byte_count = count * sizeof(uint64_t);
        if (std::fwrite(bytes.data(), 1, byte_count, f) != byte_count) {
            return false;
        }
    }
    return true;
}

bool read_bytes(std::ifstream & f, void * data, size_t size) {
    constexpr size_t kMaxReadSize = 1U << 30;
    auto *           dst          = static_cast<uint8_t *>(data);
    while (size > 0) {
        const size_t chunk = std::min(size, kMaxReadSize);
        f.read(reinterpret_cast<char *>(dst), static_cast<std::streamsize>(chunk));
        if (!f) {
            return false;
        }
        dst += chunk;
        size -= chunk;
    }
    return true;
}

bool read_f32_array_le(std::ifstream & f, std::vector<float> & values) {
    if (values.empty()) {
        return true;
    }
    if (host_is_little_endian()) {
        return read_bytes(f, values.data(), values.size() * sizeof(float));
    }

    constexpr size_t     kChunkElements = 4096;
    std::vector<uint8_t> bytes(kChunkElements * sizeof(uint32_t));
    for (size_t offset = 0; offset < values.size(); offset += kChunkElements) {
        const size_t count = std::min(kChunkElements, values.size() - offset);
        if (!read_bytes(f, bytes.data(), count * sizeof(uint32_t))) {
            return false;
        }
        for (size_t i = 0; i < count; ++i) {
            values[offset + i] = u32_to_float(get_u32_le(bytes.data() + i * sizeof(uint32_t)));
        }
    }
    return true;
}

bool read_u64_array_le(std::ifstream & f, std::vector<uint64_t> & values) {
    if (values.empty()) {
        return true;
    }
    if (host_is_little_endian()) {
        return read_bytes(f, values.data(), values.size() * sizeof(uint64_t));
    }

    constexpr size_t     kChunkElements = 4096;
    std::vector<uint8_t> bytes(kChunkElements * sizeof(uint64_t));
    for (size_t offset = 0; offset < values.size(); offset += kChunkElements) {
        const size_t count = std::min(kChunkElements, values.size() - offset);
        if (!read_bytes(f, bytes.data(), count * sizeof(uint64_t))) {
            return false;
        }
        for (size_t i = 0; i < count; ++i) {
            values[offset + i] = get_u64_le(bytes.data() + i * sizeof(uint64_t));
        }
    }
    return true;
}

// Top-k via min-heap of (score, id). The heap holds at most `k` candidates;
// each new score is compared against the smallest in the heap.
struct score_id {
    double   score;
    uint64_t id;
};

struct score_id_min_heap_cmp {
    bool operator()(const score_id & a, const score_id & b) const {
        // Min-heap by score (smallest score at the top).
        return a.score > b.score;
    }
};

}  // namespace

// Lifetime-managed instance state. Lives behind the opaque
// `ggml_vec_index_t` typedef.
struct ggml_vec_index {
    int dim       = 0;
    int bit_width = 32;

    // Flat row-major storage: `data[slot * dim + i]` is component i of vec slot.
    std::vector<float> data;

    // slot -> external id (parallel to logical slot index).
    std::vector<uint64_t> slot_to_id;

    // external id -> slot.
    std::unordered_map<uint64_t, size_t> id_to_slot;
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

ggml_vec_index_t * ggml_vec_index_create(int dim, int bit_width) {
    try {
        if (dim <= 0) {
            return nullptr;
        }
        if (bit_width != 32) {
            return nullptr;
        }
        auto * idx = new (std::nothrow) ggml_vec_index();
        if (idx == nullptr) {
            return nullptr;
        }
        idx->dim       = dim;
        idx->bit_width = bit_width;
        return idx;
    } catch (...) {
        return nullptr;
    }
}

void ggml_vec_index_free(ggml_vec_index_t * idx) {
    delete idx;
}

// ---------------------------------------------------------------------------
// Mutation
// ---------------------------------------------------------------------------

int ggml_vec_index_add(ggml_vec_index_t * idx, const float * vectors, int n, const uint64_t * ids) {
    size_t base_slot = 0;
    size_t dim_sz    = 0;
    bool   resized   = false;

    try {
        if (idx == nullptr) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (n < 0) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (n == 0) {
            return GGML_VEC_INDEX_OK;
        }
        if (vectors == nullptr || ids == nullptr) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }

        dim_sz            = static_cast<size_t>(idx->dim);
        const size_t n_sz = static_cast<size_t>(n);
        if (idx->slot_to_id.size() > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            n_sz > static_cast<size_t>(std::numeric_limits<int>::max()) - idx->slot_to_id.size()) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        size_t value_count = 0;
        if (!checked_mul_size(n_sz, dim_sz, value_count) ||
            !can_address_array(value_count, sizeof(float)) ||
            !can_address_array(n_sz, sizeof(uint64_t))) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }

        base_slot = idx->slot_to_id.size();
        if (n_sz > std::numeric_limits<size_t>::max() - base_slot) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        const size_t new_slots = base_slot + n_sz;
        size_t       new_value_count = 0;
        if (!checked_mul_size(new_slots, dim_sz, new_value_count) ||
            !can_address_array(new_value_count, sizeof(float)) ||
            !can_address_array(new_slots, sizeof(uint64_t))) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (!all_finite(vectors, value_count)) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }

        // Atomic add: detect duplicates first (against existing AND in-batch),
        // bail before mutating any state.
        std::unordered_set<uint64_t> batch_ids;
        batch_ids.reserve(n_sz);
        for (int i = 0; i < n; ++i) {
            if (ids[i] == kPaddingId) {
                return GGML_VEC_INDEX_E_INVALID_ARG;
            }
            if (idx->id_to_slot.find(ids[i]) != idx->id_to_slot.end()) {
                return GGML_VEC_INDEX_E_DUPLICATE;
            }
            if (!batch_ids.insert(ids[i]).second) {
                return GGML_VEC_INDEX_E_DUPLICATE;
            }
        }

        const size_t max_data_slots = (std::numeric_limits<size_t>::max() / sizeof(float)) / dim_sz;
        const size_t max_id_slots   = std::numeric_limits<size_t>::max() / sizeof(uint64_t);
        const size_t max_slots      = std::min(static_cast<size_t>(std::numeric_limits<int>::max()),
                                              std::min(max_data_slots, max_id_slots));
        const size_t current_slots = std::min(idx->slot_to_id.capacity(), idx->data.capacity() / dim_sz);
        const size_t target_slots  = grow_capacity(current_slots, new_slots, max_slots);
        const size_t target_values = target_slots * dim_sz;

        if (idx->data.capacity() < target_values) {
            idx->data.reserve(target_values);
        }
        if (idx->slot_to_id.capacity() < target_slots) {
            idx->slot_to_id.reserve(target_slots);
        }
        if (!can_insert_without_rehash(idx->id_to_slot, new_slots)) {
            idx->id_to_slot.reserve(target_slots);
        }

        idx->data.insert(idx->data.end(), vectors, vectors + value_count);
        resized = true;

        for (int i = 0; i < n; ++i) {
            const size_t slot = base_slot + static_cast<size_t>(i);
            idx->slot_to_id.push_back(ids[i]);
            idx->id_to_slot.emplace(ids[i], slot);
        }
    } catch (const std::bad_alloc &) {
        if (idx != nullptr && resized) {
            for (int i = 0; i < n; ++i) {
                idx->id_to_slot.erase(ids[i]);
            }
            idx->data.resize(base_slot * dim_sz);
            idx->slot_to_id.resize(base_slot);
        }
        return GGML_VEC_INDEX_E_OOM;
    } catch (...) {
        if (idx != nullptr && resized) {
            for (int i = 0; i < n; ++i) {
                idx->id_to_slot.erase(ids[i]);
            }
            idx->data.resize(base_slot * dim_sz);
            idx->slot_to_id.resize(base_slot);
        }
        return GGML_VEC_INDEX_E_INTERNAL;
    }
    return GGML_VEC_INDEX_OK;
}

int ggml_vec_index_remove(ggml_vec_index_t * idx, uint64_t id) {
    try {
        if (idx == nullptr || id == kPaddingId) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        auto it = idx->id_to_slot.find(id);
        if (it == idx->id_to_slot.end()) {
            return GGML_VEC_INDEX_E_NOT_FOUND;
        }
        const size_t slot   = it->second;
        const size_t last   = idx->slot_to_id.size() - 1;
        const size_t dim_sz = static_cast<size_t>(idx->dim);

        idx->id_to_slot.erase(it);

        if (slot != last) {
            // Move last vector into the freed slot and update its id mapping.
            const uint64_t moved_id = idx->slot_to_id[last];
            auto           moved_it = idx->id_to_slot.find(moved_id);
            if (moved_it == idx->id_to_slot.end()) {
                return GGML_VEC_INDEX_E_INTERNAL;
            }
            std::memcpy(idx->data.data() + slot * dim_sz, idx->data.data() + last * dim_sz, dim_sz * sizeof(float));
            idx->slot_to_id[slot] = moved_id;
            moved_it->second      = slot;
        }

        idx->slot_to_id.pop_back();
        idx->data.resize(last * dim_sz);
        return GGML_VEC_INDEX_OK;
    } catch (...) {
        return GGML_VEC_INDEX_E_INTERNAL;
    }
}

int ggml_vec_index_contains(const ggml_vec_index_t * idx, uint64_t id) {
    if (idx == nullptr) {
        return 0;
    }
    return idx->id_to_slot.count(id) != 0 ? 1 : 0;
}

void ggml_vec_index_prepare(ggml_vec_index_t * /*idx*/) {
    // POC no-op. Future: warm caches, materialize codebooks, etc.
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

namespace {

// Scalar dot product of two `dim`-length f32 vectors.
inline double dot(const float * a, const float * b, int dim) {
    double acc = 0.0;
    for (int i = 0; i < dim; ++i) {
        acc += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return acc;
}

// Run a single query against all slots, write top-k into out_scores/out_ids.
// If the index holds fewer than k entries, pad with sentinels.
void search_one(const ggml_vec_index_t & idx,
                const float *            query,
                int                      k,
                float *                  out_scores,
                uint64_t *               out_ids,
                std::vector<score_id> &  heap,
                std::vector<score_id> &  drained) {
    const int    dim     = idx.dim;
    const size_t n_slots = idx.slot_to_id.size();

    heap.clear();
    drained.clear();

    for (size_t slot = 0; slot < n_slots; ++slot) {
        const double s = rank_score_from_double(dot(query, idx.data.data() + slot * static_cast<size_t>(dim), dim));
        if (heap.size() < static_cast<size_t>(k)) {
            heap.push_back({ s, idx.slot_to_id[slot] });
            std::push_heap(heap.begin(), heap.end(), score_id_min_heap_cmp());
        } else if (s > heap.front().score) {
            std::pop_heap(heap.begin(), heap.end(), score_id_min_heap_cmp());
            heap.back() = { s, idx.slot_to_id[slot] };
            std::push_heap(heap.begin(), heap.end(), score_id_min_heap_cmp());
        }
    }

    // Drain the heap into a temporary descending list.
    while (!heap.empty()) {
        std::pop_heap(heap.begin(), heap.end(), score_id_min_heap_cmp());
        drained.push_back(heap.back());
        heap.pop_back();
    }
    std::reverse(drained.begin(), drained.end());  // now descending by score

    for (int i = 0; i < k; ++i) {
        if (static_cast<size_t>(i) < drained.size()) {
            out_scores[i] = float_score_from_double(drained[i].score);
            out_ids[i]    = drained[i].id;
        } else {
            out_scores[i] = -FLT_MAX;
            out_ids[i]    = UINT64_MAX;
        }
    }
}

}  // namespace

int ggml_vec_index_search(const ggml_vec_index_t * idx,
                          const float *            queries,
                          int                      n_q,
                          int                      k,
                          float *                  out_scores,
                          uint64_t *               out_ids) {
    if (idx == nullptr) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    if (n_q < 0 || k <= 0) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    if (n_q == 0) {
        return GGML_VEC_INDEX_OK;
    }
    if (queries == nullptr || out_scores == nullptr || out_ids == nullptr) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }

    try {
        const int    dim    = idx->dim;
        const size_t n_q_sz = static_cast<size_t>(n_q);
        const size_t k_sz   = static_cast<size_t>(k);
        const size_t dim_sz = static_cast<size_t>(dim);
        size_t       query_count = 0;
        size_t       result_count = 0;
        if (!checked_mul_size(n_q_sz, dim_sz, query_count) ||
            !can_address_array(query_count, sizeof(float)) ||
            !checked_mul_size(n_q_sz, k_sz, result_count) ||
            !can_address_array(result_count, sizeof(float)) ||
            !can_address_array(result_count, sizeof(uint64_t))) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        if (!all_finite(queries, query_count)) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }

        std::vector<score_id> heap;
        std::vector<score_id> drained;
        heap.reserve(k_sz);
        drained.reserve(k_sz);
        for (int q = 0; q < n_q; ++q) {
            search_one(*idx, queries + static_cast<size_t>(q) * static_cast<size_t>(dim), k,
                       out_scores + static_cast<size_t>(q) * static_cast<size_t>(k),
                       out_ids + static_cast<size_t>(q) * static_cast<size_t>(k), heap, drained);
        }
    } catch (const std::bad_alloc &) {
        return GGML_VEC_INDEX_E_OOM;
    } catch (...) {
        return GGML_VEC_INDEX_E_INTERNAL;
    }
    return GGML_VEC_INDEX_OK;
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

int ggml_vec_index_write(const ggml_vec_index_t * idx, const char * path) {
    try {
        if (idx == nullptr || path == nullptr) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        const size_t n      = idx->slot_to_id.size();
        const size_t dim_sz = static_cast<size_t>(idx->dim);
        const int    preflight_status = snapshot_write_preflight(n, dim_sz, idx->data.size());
        if (preflight_status != GGML_VEC_INDEX_OK) {
            return preflight_status;
        }

        const std::filesystem::path dst_path(path);
        const std::filesystem::path tmp_path = make_tmp_path(dst_path);

        std::FILE * raw = open_exclusive(tmp_path);
        if (raw == nullptr) {
            return GGML_VEC_INDEX_E_IO;
        }
        tmp_file_guard                         tmp_guard(tmp_path);
        std::unique_ptr<std::FILE, file_closer> f(raw);
        const auto                             fail_io = [&]() {
            f.reset();
            return GGML_VEC_INDEX_E_IO;
        };

        // Header: 16 bytes. Layout matches the comment block in the header file.
        uint8_t header[kTvimHeaderSize] = {};
        std::memcpy(header, kTvimMagic, 4);
        header[4]             = kTvimVersion;
        header[5]             = static_cast<uint8_t>(idx->bit_width);
        header[6]             = 0;
        header[7]             = 0;
        const uint32_t dim_le = static_cast<uint32_t>(idx->dim);
        const uint32_t n_le   = static_cast<uint32_t>(idx->slot_to_id.size());
        put_u32_le(header + 8, dim_le);
        put_u32_le(header + 12, n_le);

        if (std::fwrite(header, 1, sizeof(header), f.get()) != sizeof(header)) {
            return fail_io();
        }

        if (!write_f32_array_le(f.get(), idx->data)) {
            return fail_io();
        }

        if (!write_u64_array_le(f.get(), idx->slot_to_id)) {
            return fail_io();
        }

        if (std::fflush(f.get()) != 0) {
            return fail_io();
        }
#ifdef _WIN32
        const intptr_t os_handle = _get_osfhandle(_fileno(f.get()));
        if (os_handle == -1 || FlushFileBuffers(reinterpret_cast<HANDLE>(os_handle)) == 0) {
            return fail_io();
        }
#else
        if (!sync_file(f.get())) {
            return fail_io();
        }
        if (!set_snapshot_permissions(f.get(), dst_path)) {
            return fail_io();
        }
        if (!sync_file(f.get())) {
            return fail_io();
        }
#endif
        std::FILE * raw_file = f.release();
        if (std::fclose(raw_file) != 0) {
            return GGML_VEC_INDEX_E_IO;
        }

#ifdef _WIN32
        if (!copy_permissions_if_exists(dst_path, tmp_path)) {
            return fail_io();
        }
#endif

        const replace_result replace_status = replace_file(tmp_path, dst_path);
        if (replace_status == replace_result::failure) {
            return GGML_VEC_INDEX_E_IO;
        }
        tmp_guard.dismiss();
        if (replace_status == replace_result::not_durable) {
            return GGML_VEC_INDEX_E_NOT_DURABLE;
        }
        return GGML_VEC_INDEX_OK;
    } catch (const std::bad_alloc &) {
        return GGML_VEC_INDEX_E_OOM;
    } catch (...) {
        return GGML_VEC_INDEX_E_INTERNAL;
    }
}

ggml_vec_index_t * ggml_vec_index_load(const char * path) {
    try {
        if (path == nullptr) {
            return nullptr;
        }
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) {
            return nullptr;
        }
        f.seekg(0, std::ios::end);
        if (!f) {
            return nullptr;
        }
        const std::streamoff end_off = f.tellg();
        if (end_off < 0 ||
            static_cast<uintmax_t>(end_off) > static_cast<uintmax_t>(std::numeric_limits<size_t>::max())) {
            return nullptr;
        }
        const size_t file_size = static_cast<size_t>(end_off);
        f.seekg(0, std::ios::beg);
        if (!f) {
            return nullptr;
        }

        uint8_t header[kTvimHeaderSize] = {};
        f.read(reinterpret_cast<char *>(header), sizeof(header));
        if (!f || f.gcount() != static_cast<std::streamsize>(sizeof(header))) {
            return nullptr;
        }
        if (std::memcmp(header, kTvimMagic, 4) != 0) {
            return nullptr;
        }
        if (header[4] != kTvimVersion) {
            return nullptr;
        }
        if (header[6] != 0 || header[7] != 0) {
            return nullptr;
        }

        const int      bit_width = static_cast<int>(header[5]);
        const uint32_t dim_le    = get_u32_le(header + 8);
        const uint32_t n_le      = get_u32_le(header + 12);
        if (dim_le == 0 || dim_le > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
            return nullptr;
        }
        if (n_le > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
            return nullptr;
        }
        const int    dim           = static_cast<int>(dim_le);
        const size_t dim_sz        = static_cast<size_t>(dim);
        const size_t n             = static_cast<size_t>(n_le);
        size_t       expected_size = 0;
        if (!supported_snapshot_size(n, dim_sz, expected_size) || file_size != expected_size) {
            return nullptr;
        }

        std::unique_ptr<ggml_vec_index_t, decltype(&ggml_vec_index_free)> idx(ggml_vec_index_create(dim, bit_width),
                                                                              ggml_vec_index_free);
        if (idx == nullptr) {
            return nullptr;
        }

        idx->data.resize(n * dim_sz);
        idx->slot_to_id.resize(n);
        idx->id_to_slot.reserve(n);

        if (!read_f32_array_le(f, idx->data)) {
            return nullptr;
        }
        for (float v : idx->data) {
            if (!std::isfinite(v)) {
                return nullptr;
            }
        }

        if (!read_u64_array_le(f, idx->slot_to_id)) {
            return nullptr;
        }

        for (size_t slot = 0; slot < n; ++slot) {
            const uint64_t id = idx->slot_to_id[slot];
            if (id == kPaddingId) {
                return nullptr;
            }
            const bool inserted = idx->id_to_slot.emplace(id, slot).second;
            if (!inserted) {
                // Duplicate id in persisted file: corrupted.
                return nullptr;
            }
        }

        return idx.release();
    } catch (...) {
        return nullptr;
    }
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

int ggml_vec_index_len(const ggml_vec_index_t * idx) {
    if (idx == nullptr) {
        return 0;
    }
    if (idx->slot_to_id.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return std::numeric_limits<int>::max();
    }
    return static_cast<int>(idx->slot_to_id.size());
}

int ggml_vec_index_dim(const ggml_vec_index_t * idx) {
    return idx ? idx->dim : 0;
}

int ggml_vec_index_bit_width(const ggml_vec_index_t * idx) {
    return idx ? idx->bit_width : 0;
}
