// qvac-coreml-tool: host CLI of the portable Core ML generator (see
// coreml/README.md). Built with the engines' *_BUILD_TESTS (and by the standalone
// coreml build when ggml is found); Linux and macOS.
//
//   dump     <gguf> [-o reference-gguf.json]
//            Shape manifest: every metadata key/value and every tensor's name,
//            type and ne, plus per-tensor value statistics (min, max, mean, std).
//   twin     --manifest <reference-gguf.json> --seed <S> --types <t[,t...]> -o <out.gguf>
//            [--set key=value ...]
//            A production-geometry random GGUF: values per tensor inside the
//            recorded range (around the recorded value when the range is a
//            single value, so every tensor differs between seeds); tensors that are quantised (or f16/bf16) in the
//            reference take the listed types in turn, the rest stay f32.
//            Quantised with ggml's own quantizers; --types takes only types
//            ggml can both write and read back (twin_writes()). --set
//            overrides a string metadata value (extraction sentinels).
//   f32      <in.gguf> -o <out.gguf>
//            An F32 copy dequantised by ggml (to_float): the exporter reference
//            for quantised sources.
//   generate --template <dir> --gguf <file> -o <out.mlpackage>
//            Runs the C++ interpreter and MILBlob writer (no compile).
//   digest   --template <dir> --gguf <file> [--options out=fp16]
//            template_sha256, source_digest, artifact_key and memo_id.
//   verify   --template <dir> (--gguf <file> | --ours <weight.bin>) --ref <weight.bin>
//            Per blob: offset, exact|ulp, differing elements, max ulp distance.
//            Exit 1 when any blob fails.
//
// The tool links the tests' copy of the runtime (namespace qvac_coreml_test).

#include "qvac_coreml/recipe.h"

#include "cache.h"
#include "digest.h"
#include "gguf_source.h"
#include "milblob.h"
#include "sha256.h"
#include "shape.h"

#include "ggml.h"
#include "gguf.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <limits.h>
#include <stdlib.h>

using namespace QVAC_COREML_NS;
using json = nlohmann::json;

namespace {

[[noreturn]] void die(const std::string & m) {
    std::fprintf(stderr, "qvac-coreml-tool: %s\n", m.c_str());
    std::exit(2);
}

struct Args {
    std::vector<std::string> pos;
    std::map<std::string, std::vector<std::string>> opt;

    std::string get(const std::string & k, const std::string & def = {}) const {
        auto it = opt.find(k);
        return it == opt.end() || it->second.empty() ? def : it->second.back();
    }
    std::string need(const std::string & k) const {
        const std::string v = get(k);
        if (v.empty()) die("missing " + k);
        return v;
    }
};

Args parse_args(int argc, char ** argv, int from) {
    Args a;
    for (int i = from; i < argc; ++i) {
        std::string s = argv[i];
        if (s.size() > 1 && s[0] == '-') {
            if (i + 1 >= argc) die("missing value for " + s);
            a.opt[s].push_back(argv[++i]);
        } else {
            a.pos.push_back(s);
        }
    }
    return a;
}

std::unique_ptr<GgufSource> open_gguf(const std::string & path) {
    Error err;
    auto src = GgufSource::open(path, err);
    if (!src) die(err.message);
    return src;
}

// ---- tensor dimension counts as a GGUF file records them -------------------------
//
// ggml's GGUF reader reports, and its writer records, ggml_n_dims(): the
// dimension count without trailing 1s. gguf-py writes the numpy rank instead,
// so a [1, 96, 7] tensor is 3-D in a converted GGUF and 2-D in a file ggml
// wrote. Exporters read the rank through gguf-py, so `dump` records the
// file's count and every GGUF this tool writes keeps it.

class HeaderReader {
public:
    explicit HeaderReader(const std::string & path) : f_(std::fopen(path.c_str(), "rb")), path_(path) {
        if (!f_) die("cannot open " + path);
    }
    explicit HeaderReader(const std::vector<uint8_t> & bytes) : buf_(&bytes) {}
    ~HeaderReader() {
        if (f_) std::fclose(f_);
    }
    HeaderReader(const HeaderReader &) = delete;
    HeaderReader & operator=(const HeaderReader &) = delete;

    void bytes(void * out, size_t n) {
        if (f_) {
            if (n > 0 && std::fread(out, 1, n, f_) != n) die("truncated GGUF header in " + path_);
        } else {
            if (n > buf_->size() - off_) die("truncated GGUF header");
            if (n > 0) std::memcpy(out, buf_->data() + off_, n);
        }
        off_ += n;
    }
    void skip(uint64_t n) {
        std::vector<uint8_t> tmp(static_cast<size_t>(std::min<uint64_t>(n, uint64_t{1} << 20)));
        while (n > 0) {
            const size_t k = static_cast<size_t>(std::min<uint64_t>(n, tmp.size()));
            bytes(tmp.data(), k);
            n -= k;
        }
    }
    template <class T> T get() {
        T v{};
        bytes(&v, sizeof(v));
        return v;
    }
    std::string str() {
        const uint64_t n = get<uint64_t>();
        if (n > (uint64_t{1} << 30)) die("implausible string length in a GGUF header");
        std::string s(static_cast<size_t>(n), '\0');
        bytes(s.data(), s.size());
        return s;
    }
    size_t offset() const { return off_; }

private:
    FILE * f_ = nullptr;
    const std::vector<uint8_t> * buf_ = nullptr;
    std::string path_;
    size_t off_ = 0;
};

size_t gguf_scalar_size(uint32_t type) {
    switch (type) {
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

// Reads the fixed header and the key/value section; returns the tensor count.
uint64_t skip_gguf_kv(HeaderReader & r) {
    char magic[4];
    r.bytes(magic, 4);
    if (std::memcmp(magic, "GGUF", 4) != 0) die("not a GGUF file");
    if (r.get<uint32_t>() < 2) die("GGUF version 1 is not supported");
    const uint64_t n_tensors = r.get<uint64_t>();
    const uint64_t n_kv = r.get<uint64_t>();
    auto skip_values = [&](uint32_t t, uint64_t count) {
        if (t == GGUF_TYPE_STRING) {
            for (uint64_t k = 0; k < count; ++k) r.str();
            return;
        }
        const size_t sz = gguf_scalar_size(t);
        if (sz == 0) die("unknown GGUF value type " + std::to_string(t));
        r.skip(sz * count);
    };
    for (uint64_t i = 0; i < n_kv; ++i) {
        r.str();
        const uint32_t type = r.get<uint32_t>();
        if (type == GGUF_TYPE_ARRAY) {
            const uint32_t elem = r.get<uint32_t>();
            skip_values(elem, r.get<uint64_t>());
        } else {
            skip_values(type, 1);
        }
    }
    return n_tensors;
}

// tensor name -> the dimension count the file records.
std::map<std::string, int> file_tensor_dims(const std::string & path) {
    HeaderReader r(path);
    const uint64_t n = skip_gguf_kv(r);
    std::map<std::string, int> dims;
    for (uint64_t i = 0; i < n; ++i) {
        std::string name = r.str();
        const uint32_t n_dims = r.get<uint32_t>();
        if (n_dims > 4) die("tensor " + name + " has more than 4 dimensions");
        r.skip(8ull * n_dims + 4 + 8);  // ne, type, offset
        dims[std::move(name)] = static_cast<int>(n_dims);
    }
    return dims;
}

// The header ggml serialised (`meta`, tensor infos at ggml_n_dims) with
// tensor i recorded as `dims[i]`-dimensional (extra trailing ne of 1) and
// padded to `alignment`. Tensor offsets are relative to the data section, so
// they stay valid.
std::vector<uint8_t> with_file_dims(const std::vector<uint8_t> & meta, const std::vector<int> & dims,
                                    size_t alignment) {
    HeaderReader r(meta);
    const uint64_t n = skip_gguf_kv(r);
    if (n != dims.size()) die("tensor count mismatch while writing a GGUF header");
    std::vector<uint8_t> out(meta.begin(), meta.begin() + static_cast<std::ptrdiff_t>(r.offset()));
    auto put = [&](const void * p, size_t k) {
        const auto * b = static_cast<const uint8_t *>(p);
        out.insert(out.end(), b, b + k);
    };
    for (uint64_t i = 0; i < n; ++i) {
        const std::string name = r.str();
        const uint32_t have = r.get<uint32_t>();
        if (have > 4) die("tensor " + name + " has more than 4 dimensions");
        int64_t ne[4] = {1, 1, 1, 1};
        for (uint32_t d = 0; d < have; ++d) ne[d] = r.get<int64_t>();
        const uint32_t type = r.get<uint32_t>();
        const uint64_t offset = r.get<uint64_t>();
        const uint32_t want = static_cast<uint32_t>(std::max<int>(dims[i], static_cast<int>(have)));
        if (want > 4) die("tensor " + name + " has more than 4 dimensions");
        const uint64_t len = name.size();
        put(&len, 8);
        put(name.data(), name.size());
        put(&want, 4);
        put(ne, 8ull * want);
        put(&type, 4);
        put(&offset, 8);
    }
    out.resize((out.size() + alignment - 1) / alignment * alignment, 0);
    return out;
}

void write_text(const std::string & path, const std::string & text) {
    if (path.empty() || path == "-") {
        std::fwrite(text.data(), 1, text.size(), stdout);
        std::fputc('\n', stdout);
        return;
    }
    std::ofstream out(path, std::ios::binary);
    out << text << "\n";
    if (!out) die("cannot write " + path);
}

// ---- dump ------------------------------------------------------------------------

json meta_entry(const gguf_context * g, int64_t id) {
    json e;
    e["key"] = gguf_get_key(g, id);
    const gguf_type t = gguf_get_kv_type(g, id);
    e["gguf_type"] = static_cast<int>(t);
    if (t == GGUF_TYPE_ARRAY) e["elem_type"] = static_cast<int>(gguf_get_arr_type(g, id));
    return e;
}

int cmd_dump(const Args & a) {
    if (a.pos.size() != 1) die("dump <gguf> [-o out.json]");
    auto src = open_gguf(a.pos[0]);
    const auto * g = static_cast<const gguf_context *>(src->gguf());
    json j;
    j["schema"] = 1;
    j["gguf_version"] = gguf_get_version(g);
    json meta = json::array();
    for (int64_t i = 0; i < gguf_get_n_kv(g); ++i) {
        json e = meta_entry(g, i);
        e["value"] = json::parse(src->meta_json(e["key"].get<std::string>()));
        meta.push_back(e);
    }
    j["metadata"] = meta;
    const std::map<std::string, int> dims = file_tensor_dims(a.pos[0]);
    json tensors = json::array();
    for (const recipe::TensorInfo & ti : src->tensors()) {
        json t;
        t["name"] = ti.name;
        t["type"] = ggml_type_name(static_cast<ggml_type>(ti.type));
        t["ne"] = std::vector<int64_t>(ti.ne.begin(), ti.ne.end());
        const auto d = dims.find(ti.name);
        t["n_dims"] = d != dims.end() ? std::max(d->second, ti.n_dims) : ti.n_dims;
        std::vector<float> v;
        Error e = src->read_f32(ti.name, v, nullptr);
        if (e.ok() && !v.empty()) {
            double mn = v[0], mx = v[0], sum = 0.0, sq = 0.0;
            for (float x : v) {
                mn = std::min(mn, static_cast<double>(x));
                mx = std::max(mx, static_cast<double>(x));
                sum += x;
            }
            const double mean = sum / static_cast<double>(v.size());
            for (float x : v) sq += (x - mean) * (x - mean);
            t["stats"] = {{"min", mn}, {"max", mx}, {"mean", mean}, {"std", std::sqrt(sq / static_cast<double>(v.size()))}};
        } else {
            t["stats"] = nullptr;  // not a float tensor
        }
        tensors.push_back(t);
    }
    j["tensors"] = tensors;
    write_text(a.get("-o"), j.dump(1));
    return 0;
}

// ---- GGUF writing (meta first, then each tensor's data streamed) -------------------

class StreamWriter {
public:
    StreamWriter(gguf_context * g, std::string path) : g_(g), path_(std::move(path)) {
        ggml_init_params p{};
        p.mem_size = ggml_tensor_overhead() * 32768;
        p.no_alloc = true;
        ctx_ = ggml_init(p);
    }
    ~StreamWriter() {
        if (ctx_) ggml_free(ctx_);
    }

    // `n_dims` is the count the file records (see with_file_dims).
    void add(const std::string & name, ggml_type type, int n_dims, const int64_t * ne) {
        ggml_tensor * t = ggml_new_tensor(ctx_, type, n_dims, ne);
        ggml_set_name(t, name.c_str());
        gguf_add_tensor(g_, t);
        order_.push_back(t);
        dims_.push_back(n_dims);
    }

    // Meta, then `fill(i, bytes)` for every tensor in order.
    template <class Fill>
    void write(Fill fill) {
        const size_t align = gguf_get_alignment(g_);
        std::vector<uint8_t> meta(gguf_get_meta_size(g_));
        gguf_get_meta_data(g_, meta.data());
        meta = with_file_dims(meta, dims_, align);
        FILE * f = std::fopen(path_.c_str(), "wb");
        if (!f) die("cannot write " + path_);
        if (std::fwrite(meta.data(), 1, meta.size(), f) != meta.size()) die("short write to " + path_);
        std::vector<uint8_t> bytes;
        for (size_t i = 0; i < order_.size(); ++i) {
            bytes.clear();
            fill(i, order_[i], bytes);
            if (bytes.size() != ggml_nbytes(order_[i])) die(std::string("size mismatch for ") + order_[i]->name);
            const size_t pad = (align - bytes.size() % align) % align;
            bytes.resize(bytes.size() + pad, 0);
            if (std::fwrite(bytes.data(), 1, bytes.size(), f) != bytes.size()) die("short write to " + path_);
        }
        if (std::fclose(f) != 0) die("cannot close " + path_);
    }

private:
    gguf_context * g_;
    std::string path_;
    ggml_context * ctx_ = nullptr;
    std::vector<ggml_tensor *> order_;
    std::vector<int> dims_;
};

// The encodings `twin` writes: f32 and f16 itself, the others through
// ggml_quantize_chunk(), which aborts on a type it has no encoder for. A
// type qualifies when ggml-base both quantises it (from_float_ref) and reads
// it back (to_float), so a twin is always a GGUF the generator and `f32` can
// read. That leaves out f64 and q8_k (no encoder) and q8_1 (no decoder).
bool twin_writes(ggml_type type) {
    if (type == GGML_TYPE_F32 || type == GGML_TYPE_F16) return true;
    const ggml_type_traits * tr = ggml_get_type_traits(type);
    return tr && tr->from_float_ref && tr->to_float && !ggml_quantize_requires_imatrix(type);
}

// The --types names twin accepts, in ggml type order.
std::string twin_type_names() {
    std::string s;
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char * name = recipe::ggml_type_name_of(static_cast<uint32_t>(t));
        if (std::strcmp(name, "?") == 0 || !twin_writes(static_cast<ggml_type>(t))) continue;
        if (!s.empty()) s += ", ";
        s += name;
    }
    return s;
}

std::vector<uint8_t> encode(ggml_type type, const std::vector<float> & v, int64_t ne0) {
    std::vector<uint8_t> out;
    const int64_t n = static_cast<int64_t>(v.size());
    if (type == GGML_TYPE_F32) {
        out.resize(v.size() * 4);
        std::memcpy(out.data(), v.data(), out.size());
    } else if (type == GGML_TYPE_F16) {
        out.resize(v.size() * 2);
        for (size_t i = 0; i < v.size(); ++i) {
            const uint16_t h = recipe::f32_to_f16(v[i]);
            std::memcpy(out.data() + 2 * i, &h, 2);
        }
    } else {
        const int64_t rows = n / ne0;
        out.resize(ggml_row_size(type, ne0) * static_cast<size_t>(rows));
        ggml_quantize_chunk(type, v.data(), out.data(), 0, rows, ne0, nullptr);
    }
    return out;
}

// ---- twin ------------------------------------------------------------------------

// splitmix64: the same stream on every host.
struct Rng {
    uint64_t s;
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double uniform() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }
    // Irwin-Hall(12) - 6: approximately N(0, 1) with no libm, so twins are
    // byte-identical across hosts.
    double normal() {
        double s = 0.0;
        for (int i = 0; i < 12; ++i) s += uniform();
        return s - 6.0;
    }
};

void set_meta(gguf_context * g, const json & e) {
    const std::string k = e.at("key");
    const int t = e.at("gguf_type");
    const json & v = e.at("value");
    switch (t) {
        case GGUF_TYPE_UINT8: gguf_set_val_u8(g, k.c_str(), v.get<uint8_t>()); break;
        case GGUF_TYPE_INT8: gguf_set_val_i8(g, k.c_str(), v.get<int8_t>()); break;
        case GGUF_TYPE_UINT16: gguf_set_val_u16(g, k.c_str(), v.get<uint16_t>()); break;
        case GGUF_TYPE_INT16: gguf_set_val_i16(g, k.c_str(), v.get<int16_t>()); break;
        case GGUF_TYPE_UINT32: gguf_set_val_u32(g, k.c_str(), v.get<uint32_t>()); break;
        case GGUF_TYPE_INT32: gguf_set_val_i32(g, k.c_str(), v.get<int32_t>()); break;
        case GGUF_TYPE_FLOAT32: gguf_set_val_f32(g, k.c_str(), v.get<float>()); break;
        case GGUF_TYPE_UINT64: gguf_set_val_u64(g, k.c_str(), v.get<uint64_t>()); break;
        case GGUF_TYPE_INT64: gguf_set_val_i64(g, k.c_str(), v.get<int64_t>()); break;
        case GGUF_TYPE_FLOAT64: gguf_set_val_f64(g, k.c_str(), v.get<double>()); break;
        case GGUF_TYPE_BOOL: gguf_set_val_bool(g, k.c_str(), v.get<bool>()); break;
        case GGUF_TYPE_STRING: gguf_set_val_str(g, k.c_str(), v.get<std::string>().c_str()); break;
        case GGUF_TYPE_ARRAY: {
            const int et = e.at("elem_type");
            if (et == GGUF_TYPE_STRING) {
                std::vector<std::string> s = v.get<std::vector<std::string>>();
                std::vector<const char *> p;
                for (const std::string & x : s) p.push_back(x.c_str());
                gguf_set_arr_str(g, k.c_str(), p.data(), p.size());
                break;
            }
            std::vector<uint8_t> raw;
            for (const json & x : v) {
                auto put = [&](auto val) {
                    const size_t at = raw.size();
                    raw.resize(at + sizeof(val));
                    std::memcpy(raw.data() + at, &val, sizeof(val));
                };
                switch (et) {
                    case GGUF_TYPE_UINT8: put(x.get<uint8_t>()); break;
                    case GGUF_TYPE_INT8: put(x.get<int8_t>()); break;
                    case GGUF_TYPE_UINT16: put(x.get<uint16_t>()); break;
                    case GGUF_TYPE_INT16: put(x.get<int16_t>()); break;
                    case GGUF_TYPE_UINT32: put(x.get<uint32_t>()); break;
                    case GGUF_TYPE_INT32: put(x.get<int32_t>()); break;
                    case GGUF_TYPE_FLOAT32: put(x.get<float>()); break;
                    case GGUF_TYPE_UINT64: put(x.get<uint64_t>()); break;
                    case GGUF_TYPE_INT64: put(x.get<int64_t>()); break;
                    case GGUF_TYPE_FLOAT64: put(x.get<double>()); break;
                    case GGUF_TYPE_BOOL: put(static_cast<int8_t>(x.get<bool>())); break;
                    default: die("unsupported array element type for " + k);
                }
            }
            gguf_set_arr_data(g, k.c_str(), static_cast<gguf_type>(et), raw.data(), v.size());
            break;
        }
        default: die("unsupported metadata type for " + k);
    }
}

int cmd_twin(const Args & a) {
    // Every --types entry is checked before anything is read or written.
    std::vector<ggml_type> types;
    {
        std::stringstream ss(a.need("--types"));
        std::string t;
        while (std::getline(ss, t, ',')) {
            auto ty = recipe::ggml_type_from_name(t);
            if (!ty) die("unknown type " + t);
            if (!twin_writes(static_cast<ggml_type>(*ty))) {
                die("--types: cannot write " + t + " (supported: " + twin_type_names() + ")");
            }
            types.push_back(static_cast<ggml_type>(*ty));
        }
    }
    std::ifstream in(a.need("--manifest"));
    if (!in) die("cannot read the manifest");
    const json m = json::parse(in, nullptr, false);
    if (!m.is_object() || m.value("schema", 0) != 1) die("not a schema-1 reference-gguf.json");
    const uint64_t seed = std::stoull(a.need("--seed"));
    std::map<std::string, std::string> sets;
    if (auto it = a.opt.find("--set"); it != a.opt.end()) {
        for (const std::string & kv : it->second) {
            const size_t eq = kv.find('=');
            if (eq == std::string::npos) die("--set key=value");
            sets[kv.substr(0, eq)] = kv.substr(eq + 1);
        }
    }
    gguf_context * g = gguf_init_empty();
    for (const json & e : m.at("metadata")) {
        const std::string k = e.at("key");
        if (k == GGUF_KEY_GENERAL_ALIGNMENT) continue;  // the writer's own alignment
        if (sets.count(k)) {
            gguf_set_val_str(g, k.c_str(), sets[k].c_str());
            sets.erase(k);
            continue;
        }
        set_meta(g, e);
    }
    for (const auto & [k, v] : sets) gguf_set_val_str(g, k.c_str(), v.c_str());

    struct Plan {
        std::string name;
        ggml_type type;
        int n_dims;
        int64_t ne[4];
        double lo, hi, mean, sd;
        bool has_stats;
    };
    std::vector<Plan> plan;
    size_t rr = 0;
    for (const json & t : m.at("tensors")) {
        Plan p{};
        p.name = t.at("name");
        p.n_dims = t.at("n_dims");
        const auto ne = t.at("ne").get<std::vector<int64_t>>();
        for (int d = 0; d < 4; ++d) p.ne[d] = ne.at(d);
        const auto ref = recipe::ggml_type_from_name(t.at("type").get<std::string>());
        p.has_stats = t.contains("stats") && t["stats"].is_object();
        if (!p.has_stats) die("tensor " + p.name + " has no value statistics (not a float tensor)");
        p.lo = t["stats"]["min"];
        p.hi = t["stats"]["max"];
        p.mean = t["stats"]["mean"];
        p.sd = t["stats"]["std"];
        if (!(p.hi > p.lo)) {
            // Every recorded value is the same (a scalar bias, a constant vector). Drawn with
            // no spread it would be equal in every twin, and extraction could not see whether
            // the exporter folds it into the program. Draw it around the recorded value
            // instead, keeping its sign: a different value per seed.
            const double r = std::max(std::fabs(p.mean), 1e-3) * 0.25;
            p.lo = p.mean - r;
            p.hi = p.mean + r;
            p.sd = r / 3.0;
        }
        p.type = GGML_TYPE_F32;
        const bool low_precision = ref && *ref != GGML_TYPE_F32;
        if (low_precision) {
            const ggml_type want = types[rr % types.size()];
            if (want == GGML_TYPE_F32 || p.ne[0] % ggml_blck_size(want) == 0) {
                p.type = want;
                ++rr;
            }
        }
        plan.push_back(p);
    }
    const std::string out = a.need("-o");
    StreamWriter w(g, out);
    for (const Plan & p : plan) w.add(p.name, p.type, p.n_dims, p.ne);
    w.write([&](size_t i, ggml_tensor *, std::vector<uint8_t> & bytes) {
        const Plan & p = plan[i];
        detail::Sha256 h;
        h.str(p.name);
        const auto d = h.finish();
        uint64_t s = seed;
        for (int b = 0; b < 8; ++b) s ^= static_cast<uint64_t>(d[b]) << (8 * b);
        Rng rng{s};
        const std::optional<uint64_t> n = detail::element_count({p.ne[0], p.ne[1], p.ne[2], p.ne[3]});
        if (!n) die("tensor " + p.name + " has more than " + std::to_string(detail::MAX_ELEMENTS) + " elements");
        std::vector<float> v(static_cast<size_t>(*n));
        for (float & x : v) {
            double y = p.mean + p.sd * rng.normal();
            y = std::min(std::max(y, p.lo), p.hi);
            x = static_cast<float>(y);
        }
        bytes = encode(p.type, v, p.ne[0]);
    });
    gguf_free(g);
    return 0;
}

// ---- f32 -------------------------------------------------------------------------

int cmd_f32(const Args & a) {
    if (a.pos.size() != 1) die("f32 <in.gguf> -o <out.gguf>");
    auto src = open_gguf(a.pos[0]);
    gguf_context * g = gguf_init_empty();
    gguf_set_kv(g, static_cast<const gguf_context *>(src->gguf()));
    gguf_remove_key(g, GGUF_KEY_GENERAL_ALIGNMENT);
    StreamWriter w(g, a.need("-o"));
    const std::map<std::string, int> dims = file_tensor_dims(a.pos[0]);
    std::vector<bool> raw;
    for (const recipe::TensorInfo & ti : src->tensors()) {
        const auto type = static_cast<ggml_type>(ti.type);
        const ggml_type_traits * tr = ggml_get_type_traits(type);
        const bool convert = type == GGML_TYPE_F32 || (tr && tr->to_float);
        raw.push_back(!convert);
        const auto d = dims.find(ti.name);
        w.add(ti.name, convert ? GGML_TYPE_F32 : type, d != dims.end() ? std::max(d->second, ti.n_dims) : ti.n_dims,
              ti.ne.data());
    }
    w.write([&](size_t i, ggml_tensor *, std::vector<uint8_t> & bytes) {
        const recipe::TensorInfo & ti = src->tensors()[i];
        if (raw[i]) {
            Error e = src->read_raw(ti.name, [&](const uint8_t * p, size_t n) {
                bytes.insert(bytes.end(), p, p + n);
                return true;
            });
            if (!e.ok()) die(e.message);
            return;
        }
        std::vector<float> v;
        Error e = src->read_f32(ti.name, v, nullptr);
        if (!e.ok()) die(e.message);
        bytes.resize(v.size() * 4);
        std::memcpy(bytes.data(), v.data(), bytes.size());
    });
    gguf_free(g);
    return 0;
}

// ---- generate / digest / verify --------------------------------------------------------

std::shared_ptr<const recipe::Template> load_template(const Args & a) {
    try {
        return recipe::Template::from_dir(a.need("--template"));
    } catch (const std::exception & e) {
        die(e.what());
    }
}

void require_match(const recipe::Template & t, const recipe::TensorSource & src) {
    const recipe::MatchResult m = recipe::match(t, src);
    if (!m.ok) die("no_template: " + m.reason);
}

int cmd_generate(const Args & a) {
    auto t = load_template(a);
    auto src = open_gguf(a.need("--gguf"));
    require_match(*t, *src);
    digest::SourceDigest dg;
    Error e = digest::source_digest(t->recipe(), *src, dg);
    if (!e.ok()) die(e.message);
    recipe::EvalStats stats;
    const std::string pkg = a.need("-o");
    e = cache::write_package(*t, *src, pkg, &stats);
    if (!e.ok()) die(e.message);
    std::ifstream wb(pkg + "/Data/com.apple.CoreML/weights/weight.bin", std::ios::binary);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(wb)), std::istreambuf_iterator<char>());
    json j;
    j["template"] = t->id();
    j["template_sha256"] = t->sha256_hex();
    j["source_digest"] = detail::to_hex(dg.digest);
    j["artifact_key"] = digest::artifact_key(GENERATOR_VERSION, t->sha256(), dg.digest, a.get("--options", digest::DEFAULT_OPTIONS));
    j["weight_bin_sha256"] = detail::to_hex(detail::sha256(bytes.data(), bytes.size()));
    j["weight_bin_size"] = bytes.size();
    j["generation_peak_bytes"] = stats.peak_bytes;
    j["generator_version"] = GENERATOR_VERSION;
    write_text("-", j.dump(1));
    return 0;
}

int cmd_digest(const Args & a) {
    auto t = load_template(a);
    const std::string path = a.need("--gguf");
    auto src = open_gguf(path);
    require_match(*t, *src);
    digest::SourceDigest dg;
    Error e = digest::source_digest(t->recipe(), *src, dg);
    if (!e.ok()) die(e.message);
    SourceStat st;
    src->stat_now(st);
    char rp[PATH_MAX];
    const std::string real = ::realpath(path.c_str(), rp) ? rp : path;
    const std::string options = a.get("--options", digest::DEFAULT_OPTIONS);
    json j;
    j["template"] = t->id();
    j["template_sha256"] = t->sha256_hex();
    j["source_digest"] = detail::to_hex(dg.digest);
    j["source_tensors"] = dg.tensors;
    j["source_bytes"] = dg.bytes;
    j["artifact_key"] = digest::artifact_key(GENERATOR_VERSION, t->sha256(), dg.digest, options);
    j["memo_id"] = digest::memo_id(real, st.size, st.mtime_ns, st.ino, st.dev, t->sha256(), GENERATOR_VERSION, options);
    j["realpath"] = real;
    j["options"] = options;
    write_text("-", j.dump(1));
    return 0;
}

std::vector<uint8_t> read_bytes(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) die("cannot read " + path);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

int cmd_verify(const Args & a) {
    auto t = load_template(a);
    std::vector<uint8_t> ours;
    if (!a.get("--ours").empty()) {
        ours = read_bytes(a.get("--ours"));
    } else {
        auto src = open_gguf(a.need("--gguf"));
        require_match(*t, *src);
        const recipe::Recipe & r = t->recipe();
        milblob::Writer w(r.blob_count, r.weight_bin_size, [&](const uint8_t * p, size_t n) -> Error {
            ours.insert(ours.end(), p, p + n);
            return {};
        });
        Error e = w.begin();
        if (e.ok()) {
            e = recipe::evaluate(r, *src, [&](const recipe::BlobOut & b) {
                return w.blob(b.offset, b.dtype == recipe::BlobType::fp16 ? milblob::FLOAT16 : milblob::FLOAT32, b.data,
                              b.bytes);
            });
        }
        if (e.ok()) e = w.finish();
        if (!e.ok()) die(e.message);
    }
    const std::vector<uint8_t> ref = read_bytes(a.need("--ref"));
    const recipe::WeightBinCheck v =
        recipe::verify_weight_bin(t->recipe(), ours.data(), ours.size(), ref.data(), ref.size());
    for (const std::string & e : v.errors) std::printf("FAIL %s\n", e.c_str());
    std::printf("%-12s %-5s %-6s %10s %8s %s\n", "offset", "class", "dtype", "differing", "max_ulp", "result");
    size_t bi = 0;
    for (const recipe::Item & it : t->recipe().items) {
        if (it.kind != recipe::Item::Kind::blob) continue;
        const recipe::BlobCheck & b = v.blobs[bi++];
        if (!b.layout.empty()) {
            std::printf("%-12llu %-5s %-6s %10s %8s FAIL (layout: %s)\n", static_cast<unsigned long long>(it.offset),
                        it.ulp ? "ulp" : "exact", recipe::to_string(it.dtype), "-", "-", b.layout.c_str());
            continue;
        }
        std::printf("%-12llu %-5s %-6s %10llu %8lld %s\n", static_cast<unsigned long long>(it.offset),
                    it.ulp ? "ulp" : "exact", recipe::to_string(it.dtype),
                    static_cast<unsigned long long>(b.cmp.differing), static_cast<long long>(b.cmp.max_ulp),
                    b.cmp.pass ? "ok" : "FAIL");
    }
    const bool ok = v.pass;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: qvac-coreml-tool dump|twin|f32|generate|digest|verify ...\n");
        return 2;
    }
    const std::string cmd = argv[1];
    const Args a = parse_args(argc, argv, 2);
    if (cmd == "dump") return cmd_dump(a);
    if (cmd == "twin") return cmd_twin(a);
    if (cmd == "f32") return cmd_f32(a);
    if (cmd == "generate") return cmd_generate(a);
    if (cmd == "digest") return cmd_digest(a);
    if (cmd == "verify") return cmd_verify(a);
    die("unknown command " + cmd);
}
