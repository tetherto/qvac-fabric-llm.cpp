// ggml-vector-index-persistence.cpp - f32 snapshot shim for early search split.

#include "ggml-vector-index-impl.h"

MappedFile::~MappedFile() = default;

void test_maybe_throw_bad_alloc() {}
void test_wait_after_delta_validate() {}
void test_wait_after_load_with_delta_snapshot() {}

#ifdef GGML_VEC_INDEX_TEST_HOOKS
static std::atomic<int64_t> g_write_bytes_remaining{ -1 };
static std::atomic<bool> g_parent_fsync_fail{ false };

extern "C" {
GGML_API void ggml_vec_index_test_set_write_fail_after(int64_t bytes) {
    g_write_bytes_remaining.store(bytes, std::memory_order_relaxed);
}

GGML_API void ggml_vec_index_test_set_parent_fsync_fail(int fail) {
    g_parent_fsync_fail.store(fail != 0, std::memory_order_relaxed);
}
}

bool test_consume_write_bytes(size_t n) {
    int64_t remaining = g_write_bytes_remaining.load(std::memory_order_relaxed);
    if (remaining < 0) {
        return true;
    }
    if (n > static_cast<size_t>(remaining)) {
        return false;
    }
    g_write_bytes_remaining.store(remaining - static_cast<int64_t>(n), std::memory_order_relaxed);
    return true;
}
#else
bool test_consume_write_bytes(size_t) {
    return true;
}
#endif

bool is_supported_bit_width(int bit_width) {
    return bit_width == 4 || bit_width == 8 || bit_width == 32;
}

bool is_valid_id(uint64_t id) {
    return id != UINT64_MAX;
}

uint32_t float_to_u32(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float u32_to_float(uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void put_u32_le(uint8_t * dst, uint32_t v) {
    dst[0] = static_cast<uint8_t>(v >> 0);
    dst[1] = static_cast<uint8_t>(v >> 8);
    dst[2] = static_cast<uint8_t>(v >> 16);
    dst[3] = static_cast<uint8_t>(v >> 24);
}

uint32_t get_u32_le(const uint8_t * src) {
    return (static_cast<uint32_t>(src[0]) << 0) |
           (static_cast<uint32_t>(src[1]) << 8) |
           (static_cast<uint32_t>(src[2]) << 16) |
           (static_cast<uint32_t>(src[3]) << 24);
}

void put_u64_le(uint8_t * dst, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        dst[i] = static_cast<uint8_t>(v >> (8 * i));
    }
}

uint64_t get_u64_le(const uint8_t * src) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= static_cast<uint64_t>(src[i]) << (8 * i);
    }
    return v;
}

uint32_t crc32c_update(uint32_t crc, const void * data, size_t len) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < len; ++i) {
        crc ^= bytes[i];
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1) ^ (0x82f63b78u & (0u - (crc & 1u)));
        }
    }
    return crc;
}

uint32_t crc32c_update_u32(uint32_t crc, uint32_t v) {
    uint8_t bytes[4];
    put_u32_le(bytes, v);
    return crc32c_update(crc, bytes, sizeof(bytes));
}

uint32_t crc32c_update_u64(uint32_t crc, uint64_t v) {
    uint8_t bytes[8];
    put_u64_le(bytes, v);
    return crc32c_update(crc, bytes, sizeof(bytes));
}

bool checked_add_u64(uint64_t a, uint64_t b, uint64_t & out) {
    if (b > std::numeric_limits<uint64_t>::max() - a) {
        return false;
    }
    out = a + b;
    return true;
}

bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t & out) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
        return false;
    }
    out = a * b;
    return true;
}

bool expected_v1_snapshot_size(uint64_t n, uint64_t dim, uint64_t & expected) {
    uint64_t values = 0;
    uint64_t vector_bytes = 0;
    uint64_t id_bytes = 0;
    uint64_t payload_bytes = 0;
    return checked_mul_u64(n, dim, values) &&
           checked_mul_u64(values, sizeof(float), vector_bytes) &&
           checked_mul_u64(n, sizeof(uint64_t), id_bytes) &&
           checked_add_u64(vector_bytes, id_bytes, payload_bytes) &&
           checked_add_u64(kTvimV1HeaderSize, payload_bytes, expected);
}

bool supported_v1_snapshot_size(uint64_t n, uint64_t dim, uint64_t & expected) {
    return expected_v1_snapshot_size(n, dim, expected) && expected <= kMaxSnapshotBytes;
}

int snapshot_write_v1_preflight(size_t n, size_t dim) {
    if (n > kMaxIndexLen || n > std::numeric_limits<uint32_t>::max()) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    uint64_t expected = 0;
    if (!supported_v1_snapshot_size(static_cast<uint64_t>(n), static_cast<uint64_t>(dim), expected)) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    return GGML_VEC_INDEX_OK;
}

bool filesystem_path_from_utf8(const char * path, std::filesystem::path & out) {
    if (path == nullptr) {
        return false;
    }
    out = std::filesystem::path(path);
    return true;
}

bool filesystem_paths_equal(const char * lhs, const char * rhs) {
    std::error_code ec;
    return std::filesystem::equivalent(lhs, rhs, ec) && !ec;
}

uint64_t slot_state_hash(const ggml_vec_index & idx, size_t slot) {
    uint64_t h = idx.slot_to_id[slot] ^ (static_cast<uint64_t>(idx.bit_width) << 32);
    h ^= static_cast<uint64_t>(idx.dim) * 0x9e3779b185ebca87ull;
    return h;
}

void add_state_hash(ggml_vec_index & idx, uint64_t hash) {
    idx.state_hash_xor ^= hash;
    idx.state_hash_sum += hash;
    idx.state_hash_sum_rot += (hash << 17) | (hash >> 47);
}

void remove_state_hash(ggml_vec_index & idx, uint64_t hash) {
    idx.state_hash_xor ^= hash;
    idx.state_hash_sum -= hash;
    idx.state_hash_sum_rot -= (hash << 17) | (hash >> 47);
}

void rebuild_state_hash(ggml_vec_index & idx) {
    idx.state_hash_xor = 0;
    idx.state_hash_sum = 0;
    idx.state_hash_sum_rot = 0;
    for (size_t slot = 0; slot < idx.slot_to_id.size(); ++slot) {
        if (slot_is_active(idx, slot)) {
            add_state_hash(idx, slot_state_hash(idx, slot));
        }
    }
}

uint32_t index_state_crc32c(const ggml_vec_index &) { return 0; }
uint32_t index_state_crc32c_after_remove(const ggml_vec_index &, uint64_t) { return 0; }
uint32_t index_state_token(const ggml_vec_index &) { return 0; }
uint32_t index_state_token_after_remove(const ggml_vec_index &, uint64_t) { return 0; }
DeltaStateWide index_state_wide(const ggml_vec_index &) { return {}; }
DeltaStateWide index_state_wide_after_remove(const ggml_vec_index &, uint64_t) { return {}; }
DeltaStateKind delta_state_kind_for_format(DeltaLogFormat) { return DeltaStateKind::legacy_crc; }
DeltaLogFormat delta_log_format_for_append(const char *) { return DeltaLogFormat::v1; }
uint32_t current_delta_state(const ggml_vec_index &, DeltaStateKind) { return 0; }
DeltaStateWide current_delta_state_wide(const ggml_vec_index &) { return {}; }
void invalidate_delta_tail_cache(ggml_vec_index &) {}
bool bind_delta_log_path(ggml_vec_index &, const char *) { return false; }
bool delta_log_matches_index_unlocked(const ggml_vec_index_t *, const char *) { return false; }
bool validate_logged_add_args(const ggml_vec_index_t *, const float *, int, const uint64_t *) { return false; }
int check_logged_add_duplicates(const ggml_vec_index_t *, int, const uint64_t *) { return GGML_VEC_INDEX_E_INVALID_ARG; }
bool build_add_delta_payload_f32(const ggml_vec_index_t *, const float *, int, const uint64_t *, std::vector<uint8_t> &) { return false; }
bool build_add_delta_payload_from_slots(const ggml_vec_index_t *, size_t, int, std::vector<uint8_t> &) { return false; }
std::vector<uint8_t> build_remove_delta_payload(uint64_t) { return {}; }
DeltaAppendResult append_delta_record_locked(ggml_vec_index &, const char *, DeltaLogFormat, uint8_t, uint32_t, uint32_t, uint32_t, const DeltaStateWide &, const DeltaStateWide &, const std::vector<uint8_t> &) { return { GGML_VEC_INDEX_E_INVALID_ARG, false }; }

static bool write_all(std::FILE * f, const void * data, size_t size) {
    return test_consume_write_bytes(size) && std::fwrite(data, 1, size, f) == size;
}

static bool write_u32(std::FILE * f, uint32_t v) {
    uint8_t bytes[4];
    put_u32_le(bytes, v);
    return write_all(f, bytes, sizeof(bytes));
}

static bool write_u64(std::FILE * f, uint64_t v) {
    uint8_t bytes[8];
    put_u64_le(bytes, v);
    return write_all(f, bytes, sizeof(bytes));
}

static bool create_file_write_binary_exclusive(const std::filesystem::path & path, std::FILE *& out) {
#ifdef _WIN32
    const int fd = _wopen(
        path.wstring().c_str(),
        _O_WRONLY | _O_CREAT | _O_EXCL | _O_BINARY,
        _S_IREAD | _S_IWRITE);
    if (fd < 0) {
        return false;
    }
    out = _fdopen(fd, "wb");
    if (out == nullptr) {
        _close(fd);
        _wunlink(path.wstring().c_str());
        return false;
    }
#else
    const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0666);
    if (fd < 0) {
        return false;
    }
    out = fdopen(fd, "wb");
    if (out == nullptr) {
        close(fd);
        unlink(path.c_str());
        return false;
    }
#endif
    return true;
}

static bool flush_file_to_disk(std::FILE * f) {
    if (std::fflush(f) != 0) {
        return false;
    }
#ifdef _WIN32
    const int fd = _fileno(f);
    return fd >= 0 && _commit(fd) == 0;
#else
    const int fd = fileno(f);
    if (fd < 0) {
        return false;
    }
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
#endif
}

static bool close_file(std::FILE *& f) {
    if (f == nullptr) {
        return true;
    }
    const bool ok = std::fclose(f) == 0;
    f = nullptr;
    return ok;
}

static bool sync_parent_dir(const std::filesystem::path & path) {
#ifdef GGML_VEC_INDEX_TEST_HOOKS
    if (g_parent_fsync_fail.load(std::memory_order_relaxed)) {
        return false;
    }
#endif
#ifdef _WIN32
    (void) path;
    return true;
#else
    const std::filesystem::path parent = path.parent_path().empty() ? std::filesystem::path(".") : path.parent_path();
    const int fd = open(parent.c_str(), O_RDONLY);
    if (fd < 0) {
        return false;
    }
    const bool ok = fsync(fd) == 0;
    return close(fd) == 0 && ok;
#endif
}

static bool rename_replace(const std::filesystem::path & src, const std::filesystem::path & dst) {
#ifdef _WIN32
    return MoveFileExW(
        src.wstring().c_str(),
        dst.wstring().c_str(),
        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    return std::rename(src.c_str(), dst.c_str()) == 0;
#endif
}

static std::filesystem::path snapshot_tmp_path(const std::filesystem::path & path, int attempt) {
    const std::filesystem::path parent = path.parent_path().empty() ? std::filesystem::path(".") : path.parent_path();
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::string name =
        ".ggml-vi-" + std::to_string(std::filesystem::hash_value(path)) + "-" +
        std::to_string(nonce) + "-" + std::to_string(attempt) + ".tmp";
    return parent / std::filesystem::path(name);
}

static bool read_u32(std::ifstream & f, uint32_t & v) {
    uint8_t bytes[4];
    f.read(reinterpret_cast<char *>(bytes), sizeof(bytes));
    if (!f) {
        return false;
    }
    v = get_u32_le(bytes);
    return true;
}

static bool read_u64(std::ifstream & f, uint64_t & v) {
    uint8_t bytes[8];
    f.read(reinterpret_cast<char *>(bytes), sizeof(bytes));
    if (!f) {
        return false;
    }
    v = get_u64_le(bytes);
    return true;
}

static bool validate_tvim_v1_layout(
        std::ifstream & f,
        uint32_t dim,
        uint32_t n,
        size_t & component_count) {
    component_count = 0;
    if (dim == 0 || dim > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        n > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        return false;
    }

    const std::streampos current = f.tellg();
    if (current < 0) {
        return false;
    }
    f.seekg(0, std::ios::end);
    const std::streampos end = f.tellg();
    f.seekg(current);
    if (!f || end < 0) {
        return false;
    }
    const uint64_t file_size = static_cast<uint64_t>(end);

    uint64_t components = 0;
    uint64_t expected_size = 0;
    if (!checked_mul_u64(static_cast<uint64_t>(n), static_cast<uint64_t>(dim), components) ||
        !supported_v1_snapshot_size(n, dim, expected_size)) {
        return false;
    }
    if (components > static_cast<uint64_t>(std::vector<float>().max_size()) ||
        static_cast<uint64_t>(n) > static_cast<uint64_t>(std::vector<uint64_t>().max_size())) {
        return false;
    }

    if (expected_size != file_size) {
        return false;
    }
    component_count = static_cast<size_t>(components);
    return true;
}

int ggml_vec_index_write(ggml_vec_index_t * idx, const char * path) {
    if (idx == nullptr || path == nullptr) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    try {
        std::unique_lock<std::shared_mutex> lock(idx->mutex);
        if (idx->bit_width != 32 || idx->read_only_mmap || idx->delta_log_bound) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        const size_t n = active_count(*idx);
        const int preflight_status = snapshot_write_v1_preflight(n, static_cast<size_t>(idx->dim));
        if (preflight_status != GGML_VEC_INDEX_OK) {
            return preflight_status;
        }
        std::filesystem::path dst_path;
        if (!filesystem_path_from_utf8(path, dst_path)) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        std::error_code ec;
        const auto existing_status = std::filesystem::status(dst_path, ec);
        const bool preserve_permissions = !ec && std::filesystem::exists(existing_status);
        const auto existing_permissions = preserve_permissions ? existing_status.permissions() : std::filesystem::perms::unknown;

        std::filesystem::path tmp_path;
        std::FILE * f = nullptr;
        for (int attempt = 0; attempt < 100 && f == nullptr; ++attempt) {
            tmp_path = snapshot_tmp_path(dst_path, attempt);
            if (create_file_write_binary_exclusive(tmp_path, f)) {
                break;
            }
        }
        if (f == nullptr) {
            return GGML_VEC_INDEX_E_IO;
        }
        if (preserve_permissions) {
            std::filesystem::permissions(tmp_path, existing_permissions, std::filesystem::perm_options::replace, ec);
            if (ec) {
                close_file(f);
                std::filesystem::remove(tmp_path, ec);
                return GGML_VEC_INDEX_E_IO;
            }
        }
        const uint32_t n_le = static_cast<uint32_t>(n);
        bool ok = write_all(f, kTvimMagic, 4);
        const uint8_t header_tail[4] = { kTvimVersionV1, 32, 0, 0 };
        ok = ok && write_all(f, header_tail, sizeof(header_tail));
        if (!write_u32(f, static_cast<uint32_t>(idx->dim)) || !write_u32(f, n_le)) {
            close_file(f);
            std::filesystem::remove(tmp_path, ec);
            return GGML_VEC_INDEX_E_IO;
        }
        const size_t dim = static_cast<size_t>(idx->dim);
        const float * data = f32_data_ptr(*idx);
        for (size_t slot = 0; slot < idx->slot_to_id.size(); ++slot) {
            if (!slot_is_active(*idx, slot)) {
                continue;
            }
            for (size_t i = 0; i < dim; ++i) {
                if (!write_u32(f, float_to_u32(data[slot * dim + i]))) {
                    close_file(f);
                    std::filesystem::remove(tmp_path, ec);
                    return GGML_VEC_INDEX_E_IO;
                }
            }
        }
        for (size_t slot = 0; slot < idx->slot_to_id.size(); ++slot) {
            if (slot_is_active(*idx, slot) && !write_u64(f, idx->slot_to_id[slot])) {
                close_file(f);
                std::filesystem::remove(tmp_path, ec);
                return GGML_VEC_INDEX_E_IO;
            }
        }
        ok = ok && flush_file_to_disk(f);
        ok = close_file(f) && ok;
        if (!ok) {
            std::filesystem::remove(tmp_path, ec);
            return GGML_VEC_INDEX_E_IO;
        }
        if (!rename_replace(tmp_path, dst_path)) {
            std::filesystem::remove(tmp_path, ec);
            return GGML_VEC_INDEX_E_IO;
        }
        if (!sync_parent_dir(dst_path)) {
            return GGML_VEC_INDEX_E_NOT_DURABLE;
        }
        return GGML_VEC_INDEX_OK;
    } catch (const std::bad_alloc &) {
        return GGML_VEC_INDEX_E_OOM;
    } catch (...) {
        return GGML_VEC_INDEX_E_INTERNAL;
    }
}

int ggml_vec_index_load_ex(const char * path, ggml_vec_index_t ** out) {
    if (out == nullptr) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    *out = nullptr;
    if (path == nullptr) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    try {
        std::filesystem::path src_path;
        if (!filesystem_path_from_utf8(path, src_path)) {
            return GGML_VEC_INDEX_E_INVALID_ARG;
        }
        std::ifstream f(src_path, std::ios::binary);
        if (!f.is_open()) {
            return GGML_VEC_INDEX_E_IO;
        }
        f.seekg(0, std::ios::end);
        if (!f) {
            return GGML_VEC_INDEX_E_IO;
        }
        const std::streamoff end_off = f.tellg();
        if (end_off < 0) {
            return GGML_VEC_INDEX_E_IO;
        }
        f.seekg(0, std::ios::beg);
        if (!f) {
            return GGML_VEC_INDEX_E_IO;
        }
        uint8_t magic[4];
        f.read(reinterpret_cast<char *>(magic), sizeof(magic));
        if (!f || std::memcmp(magic, kTvimMagic, sizeof(magic)) != 0) {
            return GGML_VEC_INDEX_E_BAD_MAGIC;
        }
        const int version = f.get();
        const int bit_width = f.get();
        const int r0 = f.get();
        const int r1 = f.get();
        if (version != kTvimVersionV1 || bit_width != 32 || r0 != 0 || r1 != 0) {
            return GGML_VEC_INDEX_E_BAD_VERSION;
        }
        uint32_t dim = 0;
        uint32_t n = 0;
        size_t component_count = 0;
        if (!read_u32(f, dim) || !read_u32(f, n) ||
            !validate_tvim_v1_layout(f, dim, n, component_count)) {
            return GGML_VEC_INDEX_E_IO;
        }
        std::vector<float> vectors(component_count);
        for (float & v : vectors) {
            uint32_t bits = 0;
            if (!read_u32(f, bits)) {
                return GGML_VEC_INDEX_E_IO;
            }
            v = u32_to_float(bits);
            if (!std::isfinite(v)) {
                return GGML_VEC_INDEX_E_IO;
            }
        }
        std::vector<uint64_t> ids(static_cast<size_t>(n));
        for (uint64_t & id : ids) {
            if (!read_u64(f, id) || !is_valid_id(id)) {
                return GGML_VEC_INDEX_E_IO;
            }
        }
        if (f.peek() != std::ifstream::traits_type::eof()) {
            return GGML_VEC_INDEX_E_IO;
        }
        std::unique_ptr<ggml_vec_index_t, decltype(&ggml_vec_index_free)> idx(
            ggml_vec_index_create(static_cast<int>(dim), 32), ggml_vec_index_free);
        if (idx == nullptr) {
            return GGML_VEC_INDEX_E_OOM;
        }
        const int rc = ggml_vec_index_add(idx.get(), vectors.data(), static_cast<int>(n), ids.data());
        if (rc == GGML_VEC_INDEX_E_DUPLICATE) {
            return GGML_VEC_INDEX_E_IO;
        }
        if (rc != GGML_VEC_INDEX_OK) {
            return rc;
        }
        idx->delta_log_start_allowed = true;
        *out = idx.release();
        return GGML_VEC_INDEX_OK;
    } catch (const std::bad_alloc &) {
        return GGML_VEC_INDEX_E_OOM;
    } catch (...) {
        return GGML_VEC_INDEX_E_IO;
    }
}

ggml_vec_index_t * ggml_vec_index_load(const char * path) {
    ggml_vec_index_t * out = nullptr;
    return ggml_vec_index_load_ex(path, &out) == GGML_VEC_INDEX_OK ? out : nullptr;
}

int ggml_vec_index_load_mmap_ex(const char *, ggml_vec_index_t ** out) {
    if (out == nullptr) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    *out = nullptr;
    return GGML_VEC_INDEX_E_BAD_VERSION;
}

ggml_vec_index_t * ggml_vec_index_load_mmap(const char *) {
    return nullptr;
}

int ggml_vec_index_load_with_delta_ex(const char *, const char *, ggml_vec_index_t ** out) {
    if (out == nullptr) {
        return GGML_VEC_INDEX_E_INVALID_ARG;
    }
    *out = nullptr;
    return GGML_VEC_INDEX_E_INVALID_ARG;
}

ggml_vec_index_t * ggml_vec_index_load_with_delta(const char *, const char *) {
    return nullptr;
}

int ggml_vec_index_compact_delta(ggml_vec_index_t *, const char *, const char *) {
    return GGML_VEC_INDEX_E_INVALID_ARG;
}
