// GgufSource: a GGUF read through one open descriptor.
// ggml-base only: gguf_*, ggml_get_type_traits()->to_float. No ggml-cpu symbol
// (GGML_BACKEND_DL builds load ggml-cpu as a module).

#include "gguf_source.h"

#include "qvac_coreml/gguf_stream.h"

#include "ggml.h"
#include "gguf.h"

#include "nlohmann/json.hpp"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <map>

#include <fcntl.h>
#include <unistd.h>

namespace QVAC_COREML_NS {

// recipe.cpp names ggml types without ggml; keep its table honest.
static_assert(GGML_TYPE_F32 == 0 && GGML_TYPE_F16 == 1 && GGML_TYPE_Q4_0 == 2 && GGML_TYPE_Q4_1 == 3 &&
                  GGML_TYPE_Q5_0 == 6 && GGML_TYPE_Q5_1 == 7 && GGML_TYPE_Q8_0 == 8 && GGML_TYPE_Q8_1 == 9 &&
                  GGML_TYPE_Q2_K == 10 && GGML_TYPE_Q3_K == 11 && GGML_TYPE_Q4_K == 12 && GGML_TYPE_Q5_K == 13 &&
                  GGML_TYPE_Q6_K == 14 && GGML_TYPE_Q8_K == 15 && GGML_TYPE_IQ4_NL == 20 &&
                  GGML_TYPE_IQ4_XS == 23 && GGML_TYPE_F64 == 28 && GGML_TYPE_BF16 == 30,
              "ggml_type ABI changed: update recipe.cpp kGgmlTypes");
static_assert(+GGUF_TYPE_UINT32 == +recipe::GGUF_U32 && GGUF_TYPE_INT32 == +recipe::GGUF_I32 &&
                  GGUF_TYPE_FLOAT32 == +recipe::GGUF_F32 && GGUF_TYPE_BOOL == +recipe::GGUF_BOOL &&
                  GGUF_TYPE_STRING == +recipe::GGUF_STRING && GGUF_TYPE_ARRAY == +recipe::GGUF_ARRAY &&
                  GGUF_TYPE_FLOAT64 == +recipe::GGUF_F64,
              "gguf_type ABI changed: update recipe.h GgufType");

namespace {

void put_le(std::vector<uint8_t> & out, const void * p, size_t n) {
    const auto * b = static_cast<const uint8_t *>(p);
    out.insert(out.end(), b, b + n);  // hosts are little-endian (interp.cpp asserts it)
}

void put_str(std::vector<uint8_t> & out, const char * s) {
    const uint32_t n = static_cast<uint32_t>(std::strlen(s));
    put_le(out, &n, 4);
    put_le(out, s, n);
}

size_t scalar_size(gguf_type t) {
    switch (t) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL: return 1;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16: return 2;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64: return 8;
        default: return 0;
    }
}

nlohmann::json scalar_json(gguf_type t, const void * p) {
    switch (t) {
        case GGUF_TYPE_UINT8: return *static_cast<const uint8_t *>(p);
        case GGUF_TYPE_INT8: return *static_cast<const int8_t *>(p);
        case GGUF_TYPE_UINT16: return *static_cast<const uint16_t *>(p);
        case GGUF_TYPE_INT16: return *static_cast<const int16_t *>(p);
        case GGUF_TYPE_UINT32: return *static_cast<const uint32_t *>(p);
        case GGUF_TYPE_INT32: return *static_cast<const int32_t *>(p);
        case GGUF_TYPE_FLOAT32: return static_cast<double>(*static_cast<const float *>(p));
        case GGUF_TYPE_UINT64: return *static_cast<const uint64_t *>(p);
        case GGUF_TYPE_INT64: return *static_cast<const int64_t *>(p);
        case GGUF_TYPE_FLOAT64: return *static_cast<const double *>(p);
        case GGUF_TYPE_BOOL: return *static_cast<const int8_t *>(p) != 0;
        default: return nullptr;
    }
}

}  // namespace

struct GgufSource::Impl {
    int fd = -1;
    gguf_context * g = nullptr;
    ggml_context * meta = nullptr;
    std::unique_ptr<qvac_gguf::gguf_stream_reader> reader;
    std::vector<recipe::TensorInfo> tensors;
    std::map<std::string, size_t> index;

    ~Impl() {
        reader.reset();
        if (g) gguf_free(g);
        if (meta) ggml_free(meta);
        if (fd >= 0) ::close(fd);
    }
};

GgufSource::GgufSource(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
GgufSource::~GgufSource() = default;

std::unique_ptr<GgufSource> GgufSource::open(const std::string & path, Error & err) {
    auto im = std::make_unique<Impl>();
    im->fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (im->fd < 0) {
        const int e = errno;
        err = Error::transient("open " + path + ": " + std::strerror(e), "NSPOSIXErrorDomain", e);
        return nullptr;
    }
    // Parse the header through the same open file (a duplicate descriptor
    // shares it), never by path again.
    const int dup_fd = ::dup(im->fd);
    FILE * f = dup_fd >= 0 ? ::fdopen(dup_fd, "rb") : nullptr;
    if (!f) {
        const int e = errno;
        if (dup_fd >= 0) ::close(dup_fd);
        err = Error::transient("fdopen " + path + ": " + std::strerror(e), "NSPOSIXErrorDomain", e);
        return nullptr;
    }
    gguf_init_params p{};
    p.no_alloc = true;
    p.ctx = &im->meta;
    im->g = gguf_init_from_file_ptr(f, p);
    std::fclose(f);
    if (!im->g || !im->meta) {
        err = Error::deterministic("not a readable GGUF: " + path);
        return nullptr;
    }
    const int64_t n = gguf_get_n_tensors(im->g);
    for (int64_t i = 0; i < n; ++i) {
        recipe::TensorInfo ti;
        ti.name = gguf_get_tensor_name(im->g, i);
        ti.type = static_cast<uint32_t>(gguf_get_tensor_type(im->g, i));
        ti.nbytes = gguf_get_tensor_size(im->g, i);
        const ggml_tensor * t = ggml_get_tensor(im->meta, ti.name.c_str());
        if (!t) {
            err = Error::deterministic("GGUF tensor without metadata: " + ti.name);
            return nullptr;
        }
        for (int d = 0; d < 4; ++d) ti.ne[d] = t->ne[d];
        ti.n_dims = ggml_n_dims(t);
        im->index[ti.name] = im->tensors.size();
        im->tensors.push_back(std::move(ti));
    }
    im->reader = std::make_unique<qvac_gguf::gguf_stream_reader>(im->g, im->fd);
    if (!im->reader->ok()) {
        err = Error::transient("cannot read " + path + " through its descriptor", "NSPOSIXErrorDomain", EIO);
        return nullptr;
    }
    err = Error{};
    return std::unique_ptr<GgufSource>(new GgufSource(std::move(im)));
}

const recipe::TensorInfo * GgufSource::tensor(const std::string & name) const {
    auto it = impl_->index.find(name);
    return it == impl_->index.end() ? nullptr : &impl_->tensors[it->second];
}

const std::vector<recipe::TensorInfo> & GgufSource::tensors() const { return impl_->tensors; }

int GgufSource::fd() const { return impl_->fd; }

const void * GgufSource::gguf() const { return impl_->g; }

std::vector<std::string> GgufSource::meta_keys() const {
    std::vector<std::string> keys;
    const int64_t n = gguf_get_n_kv(impl_->g);
    for (int64_t i = 0; i < n; ++i) keys.emplace_back(gguf_get_key(impl_->g, i));
    return keys;
}

bool GgufSource::stat_now(SourceStat & out) const { return stat_fd(impl_->fd, out); }

Error GgufSource::read_raw(const std::string & name, const std::function<bool(const uint8_t *, size_t)> & sink) {
    const recipe::TensorInfo * ti = tensor(name);
    if (!ti) return Error::deterministic("tensor " + name + " is absent");
    bool aborted = false;
    const bool ok = impl_->reader->for_each_chunk(name.c_str(), ti->nbytes, [&](const uint8_t * p, size_t n) {
        if (!sink(p, n)) {
            aborted = true;
            return false;
        }
        return true;
    });
    if (aborted) return Error::transient("read of tensor " + name + " aborted");
    if (!ok) return Error::transient("short read of tensor " + name, "NSPOSIXErrorDomain", EIO);
    return {};
}

Error GgufSource::read_f32(const std::string & name, std::vector<float> & out, const recipe::RawTap & tap) {
    const recipe::TensorInfo * ti = tensor(name);
    if (!ti) return Error::deterministic("tensor " + name + " is absent");
    const auto type = static_cast<ggml_type>(ti->type);
    const ggml_type_traits * tr = ggml_get_type_traits(type);
    if (type != GGML_TYPE_F32 && (!tr || !tr->to_float)) {
        return Error::deterministic(std::string("no ggml to_float for type ") + ggml_type_name(type));
    }
    const int64_t nel = ti->ne[0] * ti->ne[1] * ti->ne[2] * ti->ne[3];
    const size_t blck = type == GGML_TYPE_F32 ? 1 : static_cast<size_t>(tr->blck_size);
    const size_t bsz = type == GGML_TYPE_F32 ? sizeof(float) : tr->type_size;
    if (nel < 0 || static_cast<uint64_t>(nel) % blck != 0 || static_cast<uint64_t>(nel) / blck * bsz != ti->nbytes) {
        return Error::deterministic("tensor " + name + ": size does not match its type");
    }
    out.assign(static_cast<size_t>(nel), 0.0f);
    std::vector<uint8_t> carry;  // a block split across two chunks
    size_t w = 0;                // floats written
    auto convert = [&](const uint8_t * p, size_t blocks) {
        if (blocks == 0) return;
        if (type == GGML_TYPE_F32) {
            std::memcpy(out.data() + w, p, blocks * sizeof(float));
        } else {
            tr->to_float(p, out.data() + w, static_cast<int64_t>(blocks * blck));
        }
        w += blocks * blck;
    };
    Error e = read_raw(name, [&](const uint8_t * p, size_t n) {
        if (tap) tap(p, n);
        if (!carry.empty()) {
            const size_t need = bsz - carry.size();
            const size_t take = std::min(need, n);
            carry.insert(carry.end(), p, p + take);
            p += take;
            n -= take;
            if (carry.size() < bsz) return true;
            convert(carry.data(), 1);
            carry.clear();
        }
        const size_t blocks = n / bsz;
        convert(p, blocks);
        carry.assign(p + blocks * bsz, p + n);
        return true;
    });
    if (!e.ok()) return e;
    if (!carry.empty() || w != out.size()) return Error::transient("tensor " + name + ": truncated payload");
    return {};
}

bool GgufSource::meta_canonical(const std::string & key, uint32_t & gguf_type_out, std::vector<uint8_t> & bytes,
                                uint32_t * array_elem_type) const {
    const int64_t id = gguf_find_key(impl_->g, key.c_str());
    if (id < 0) return false;
    const gguf_type t = gguf_get_kv_type(impl_->g, id);
    gguf_type_out = static_cast<uint32_t>(t);
    bytes.clear();
    if (array_elem_type) *array_elem_type = 0;
    if (t == GGUF_TYPE_STRING) {
        put_str(bytes, gguf_get_val_str(impl_->g, id));
        return true;
    }
    if (t == GGUF_TYPE_ARRAY) {
        const gguf_type et = gguf_get_arr_type(impl_->g, id);
        if (array_elem_type) *array_elem_type = static_cast<uint32_t>(et);
        const uint64_t n = gguf_get_arr_n(impl_->g, id);
        put_le(bytes, &n, 8);
        if (et == GGUF_TYPE_STRING) {
            for (uint64_t i = 0; i < n; ++i) put_str(bytes, gguf_get_arr_str(impl_->g, id, i));
        } else {
            put_le(bytes, gguf_get_arr_data(impl_->g, id), scalar_size(et) * n);
        }
        return true;
    }
    put_le(bytes, gguf_get_val_data(impl_->g, id), scalar_size(t));
    return true;
}

std::string GgufSource::meta_json(const std::string & key) const {
    const int64_t id = gguf_find_key(impl_->g, key.c_str());
    if (id < 0) return {};
    const gguf_type t = gguf_get_kv_type(impl_->g, id);
    nlohmann::json j;
    if (t == GGUF_TYPE_STRING) {
        j = gguf_get_val_str(impl_->g, id);
    } else if (t == GGUF_TYPE_ARRAY) {
        const gguf_type et = gguf_get_arr_type(impl_->g, id);
        const size_t n = gguf_get_arr_n(impl_->g, id);
        j = nlohmann::json::array();
        if (et == GGUF_TYPE_STRING) {
            for (size_t i = 0; i < n; ++i) j.push_back(gguf_get_arr_str(impl_->g, id, i));
        } else {
            const auto * base = static_cast<const uint8_t *>(gguf_get_arr_data(impl_->g, id));
            for (size_t i = 0; i < n; ++i) j.push_back(scalar_json(et, base + i * scalar_size(et)));
        }
    } else {
        j = scalar_json(t, gguf_get_val_data(impl_->g, id));
    }
    return j.dump();
}

}  // namespace QVAC_COREML_NS
