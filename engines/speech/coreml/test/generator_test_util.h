#pragma once

// Shared helpers of the portable generator tests (test-coreml-recipe,
// test-coreml-milblob, test-coreml-cache): a no-skip check harness, per-test
// temp dirs, an in-memory and a file-backed TensorSource (no ggml needed),
// template builders and a fake compile step (the Apple compile is not portable).

#include "qvac_coreml/recipe.h"

#include "cache.h"
#include "gguf_source.h"
#include "milblob.h"
#include "sha256.h"

#include "nlohmann/json.hpp"

#include <array>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <fcntl.h>
#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>

namespace gtest {

using namespace QVAC_COREML_NS;
namespace fs = std::filesystem;
using json = nlohmann::json;

// ---- harness ---------------------------------------------------------------------

inline int g_checks = 0;
inline int g_failures = 0;
inline std::string g_case;

inline void check(bool ok, const char * what, const char * file, int line) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::fprintf(stderr, "FAIL [%s] %s:%d: %s\n", g_case.c_str(), file, line, what);
    }
}
#define CHECK(c) ::gtest::check(static_cast<bool>(c), #c, __FILE__, __LINE__)
#define CHECK_MSG(c, msg)                                                                     \
    do {                                                                                      \
        const bool ok_ = static_cast<bool>(c);                                                \
        ::gtest::check(ok_, #c, __FILE__, __LINE__);                                          \
        if (!ok_) std::fprintf(stderr, "      %s\n", std::string(msg).c_str());               \
    } while (0)

struct Case {
    const char * name;
    void (*fn)();
};

// Runs every case (or --only <name>); prints "ok <name>"; exit 1 on any
// failure. No skip path: every case runs on every host.
inline int run_cases(int argc, char ** argv, const std::vector<Case> & cases, const char * suite) {
    std::string only;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--only") only = argv[i + 1];
    }
    int ran = 0;
    for (const Case & c : cases) {
        if (!only.empty() && only != c.name) continue;
        g_case = c.name;
        const int before = g_failures;
        c.fn();
        ++ran;
        std::printf("%s %s\n", g_failures == before ? "ok" : "FAILED", c.name);
        std::fflush(stdout);
    }
    std::printf("%s: %d case(s), %d check(s), %d failure(s)\n", suite, ran, g_checks, g_failures);
    if (ran == 0) return 1;
    return g_failures == 0 ? 0 : 1;
}

// ---- temp dirs ---------------------------------------------------------------------

struct TempDir {
    std::string path;
    TempDir() {
        const char * t = std::getenv("TMPDIR");
        std::string base = (t && *t ? std::string(t) : std::string("/tmp")) + "/qvac-coreml-gen-XXXXXX";
        std::vector<char> buf(base.begin(), base.end());
        buf.push_back('\0');
        if (!::mkdtemp(buf.data())) {
            std::perror("mkdtemp");
            std::exit(1);
        }
        char rp[PATH_MAX];
        path = ::realpath(buf.data(), rp) ? rp : buf.data();
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
    std::string operator/(const std::string & s) const { return path + "/" + s; }
};

inline void write_bytes(const std::string & path, const std::string & data) {
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << data;
}

inline std::string read_bytes(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// ---- sources -----------------------------------------------------------------------

constexpr uint32_t T_F32 = 0;
constexpr uint32_t T_F16 = 1;

struct TT {
    std::string name;
    uint32_t type = T_F32;
    std::vector<int64_t> shape;  // numpy order
    std::vector<float> values;
};

struct TM {
    std::string key;
    std::string mtype;  // u32 | f32 | str | arr_u32
    json value;
};

inline void le(std::string & out, const void * p, size_t n) { out.append(static_cast<const char *>(p), n); }

inline std::string canonical(const TM & m, uint32_t & type, uint32_t & elem) {
    std::string b;
    elem = 0;
    if (m.mtype == "u32") {
        type = recipe::GGUF_U32;
        const uint32_t v = m.value.get<uint32_t>();
        le(b, &v, 4);
    } else if (m.mtype == "f32") {
        type = recipe::GGUF_F32;
        const float v = m.value.get<float>();
        le(b, &v, 4);
    } else if (m.mtype == "str") {
        type = recipe::GGUF_STRING;
        const std::string s = m.value.get<std::string>();
        const uint32_t n = static_cast<uint32_t>(s.size());
        le(b, &n, 4);
        b += s;
    } else if (m.mtype == "arr_u32") {
        type = recipe::GGUF_ARRAY;
        elem = recipe::GGUF_U32;
        const uint64_t n = m.value.size();
        le(b, &n, 8);
        for (const json & x : m.value) {
            const uint32_t v = x.get<uint32_t>();
            le(b, &v, 4);
        }
    } else {
        std::fprintf(stderr, "test: unsupported mtype %s\n", m.mtype.c_str());
        std::exit(1);
    }
    return b;
}

inline recipe::TensorInfo info_of(const TT & t) {
    recipe::TensorInfo ti;
    ti.name = t.name;
    ti.type = t.type;
    ti.n_dims = static_cast<int>(t.shape.size());
    for (size_t d = 0; d < t.shape.size(); ++d) ti.ne[d] = t.shape[t.shape.size() - 1 - d];
    ti.nbytes = t.values.size() * (t.type == T_F16 ? 2 : 4);
    return ti;
}

inline std::string payload_of(const TT & t) {
    std::string b;
    for (float v : t.values) {
        if (t.type == T_F16) {
            const uint16_t h = recipe::f32_to_f16(v);
            le(b, &h, 2);
        } else {
            le(b, &v, 4);
        }
    }
    return b;
}

// Decodes a payload the way t() does for these types.
inline Error decode(uint32_t type, const std::string & raw, std::vector<float> & out) {
    if (type == T_F32) {
        out.resize(raw.size() / 4);
        std::memcpy(out.data(), raw.data(), raw.size());
        return {};
    }
    if (type == T_F16) {
        out.resize(raw.size() / 2);
        for (size_t i = 0; i < out.size(); ++i) {
            uint16_t h;
            std::memcpy(&h, raw.data() + 2 * i, 2);
            out[i] = recipe::f16_to_f32(h);
        }
        return {};
    }
    return Error::deterministic("test source: unsupported type");
}

// In memory (recipe tests).
class MemSource : public OpenSource {
public:
    MemSource(std::vector<TT> tensors, std::vector<TM> meta) : meta_(std::move(meta)) {
        for (const TT & t : tensors) {
            infos_[t.name] = info_of(t);
            raw_[t.name] = payload_of(t);
        }
    }
    int raw_reads = 0;

    const recipe::TensorInfo * tensor(const std::string & n) const override {
        auto it = infos_.find(n);
        return it == infos_.end() ? nullptr : &it->second;
    }
    Error read_raw(const std::string & n, const std::function<bool(const uint8_t *, size_t)> & sink) override {
        ++raw_reads;
        const std::string & r = raw_.at(n);
        sink(reinterpret_cast<const uint8_t *>(r.data()), r.size());
        return {};
    }
    Error read_f32(const std::string & n, std::vector<float> & out, const recipe::RawTap & tap) override {
        const std::string & r = raw_.at(n);
        if (tap) tap(reinterpret_cast<const uint8_t *>(r.data()), r.size());
        return decode(infos_.at(n).type, r, out);
    }
    bool meta_canonical(const std::string & key, uint32_t & type, std::vector<uint8_t> & bytes,
                        uint32_t * elem) const override {
        for (const TM & m : meta_) {
            if (m.key != key) continue;
            uint32_t e = 0;
            const std::string b = canonical(m, type, e);
            if (elem) *elem = e;
            bytes.assign(b.begin(), b.end());
            return true;
        }
        return false;
    }
    std::string meta_json(const std::string & key) const override {
        for (const TM & m : meta_) {
            if (m.key == key) return m.value.dump();
        }
        return {};
    }
    bool stat_now(SourceStat & out) const override {
        out = SourceStat{};
        return true;
    }

    std::map<std::string, std::string> raw_;

private:
    std::map<std::string, recipe::TensorInfo> infos_;
    std::vector<TM> meta_;
};

// A file in a test-only container format, read through one open fd like
// GgufSource: "QVTS", u32 n_tensors, per tensor {str name, u32 type, i64 ne[4],
// u64 nbytes, u64 offset}, u32 n_meta, per meta {str key, str mtype, str json},
// then the payloads.
inline void write_source_file(const std::string & path, const std::vector<TT> & tensors, const std::vector<TM> & meta) {
    std::string head = "QVTS";
    auto put_str = [&](const std::string & s) {
        const uint32_t n = static_cast<uint32_t>(s.size());
        le(head, &n, 4);
        head += s;
    };
    std::string data;
    const uint32_t nt = static_cast<uint32_t>(tensors.size());
    le(head, &nt, 4);
    std::vector<std::string> payloads;
    for (const TT & t : tensors) payloads.push_back(payload_of(t));
    uint64_t off = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
        const recipe::TensorInfo ti = info_of(tensors[i]);
        put_str(ti.name);
        le(head, &ti.type, 4);
        for (int64_t ne : ti.ne) le(head, &ne, 8);
        le(head, &ti.nbytes, 8);
        le(head, &off, 8);
        off += ti.nbytes;
    }
    const uint32_t nm = static_cast<uint32_t>(meta.size());
    le(head, &nm, 4);
    for (const TM & m : meta) {
        put_str(m.key);
        put_str(m.mtype);
        put_str(m.value.dump());
    }
    const uint64_t hl = head.size() + 8;
    std::string file = head;
    le(file, &hl, 8);
    for (const std::string & p : payloads) file += p;
    // Write in place (keeps the inode, like an editor saving over a file).
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0 || ::write(fd, file.data(), file.size()) != static_cast<ssize_t>(file.size())) {
        std::perror("write_source_file");
        std::exit(1);
    }
    ::close(fd);
}

// Called before every raw read (TOCTOU tests mutate the file here).
inline std::function<void(const std::string & tensor)> g_before_read;
inline std::atomic<int> g_file_raw_reads{0};  // every tensor read (digest or t())

class FileSource : public OpenSource {
public:
    static std::unique_ptr<OpenSource> open(const std::string & path, Error & err) {
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            err = Error::transient("open " + path, "NSPOSIXErrorDomain", errno);
            return nullptr;
        }
        std::unique_ptr<FileSource> s(new FileSource(fd));
        if (!s->parse()) {
            err = Error::deterministic("not a QVTS file: " + path);
            return nullptr;
        }
        return s;
    }
    ~FileSource() override { ::close(fd_); }

    const recipe::TensorInfo * tensor(const std::string & n) const override {
        auto it = infos_.find(n);
        return it == infos_.end() ? nullptr : &it->second.first;
    }
    Error read_raw(const std::string & n, const std::function<bool(const uint8_t *, size_t)> & sink) override {
        std::string r;
        Error e = raw(n, r);
        if (!e.ok()) return e;
        if (!sink(reinterpret_cast<const uint8_t *>(r.data()), r.size())) return Error::transient("aborted");
        return {};
    }
    Error read_f32(const std::string & n, std::vector<float> & out, const recipe::RawTap & tap) override {
        std::string r;
        Error e = raw(n, r);
        if (!e.ok()) return e;
        if (tap) tap(reinterpret_cast<const uint8_t *>(r.data()), r.size());
        return decode(infos_.at(n).first.type, r, out);
    }
    bool meta_canonical(const std::string & key, uint32_t & type, std::vector<uint8_t> & bytes,
                        uint32_t * elem) const override {
        auto it = meta_.find(key);
        if (it == meta_.end()) return false;
        uint32_t e = 0;
        const std::string b = canonical(it->second, type, e);
        if (elem) *elem = e;
        bytes.assign(b.begin(), b.end());
        return true;
    }
    std::string meta_json(const std::string & key) const override {
        auto it = meta_.find(key);
        return it == meta_.end() ? std::string() : it->second.value.dump();
    }
    bool stat_now(SourceStat & out) const override { return stat_fd(fd_, out); }

private:
    explicit FileSource(int fd) : fd_(fd) {}

    Error raw(const std::string & n, std::string & r) {
        ++g_file_raw_reads;
        if (g_before_read) g_before_read(n);
        auto it = infos_.find(n);
        if (it == infos_.end()) return Error::deterministic("no tensor " + n);
        r.resize(it->second.first.nbytes);
        const ssize_t got = ::pread(fd_, r.data(), r.size(), static_cast<off_t>(data_ + it->second.second));
        if (got != static_cast<ssize_t>(r.size())) return Error::transient("short read", "NSPOSIXErrorDomain", EIO);
        return {};
    }

    bool parse() {
        struct stat st{};
        if (::fstat(fd_, &st) != 0) return false;
        std::string f(static_cast<size_t>(st.st_size), '\0');
        if (::pread(fd_, f.data(), f.size(), 0) != static_cast<ssize_t>(f.size())) return false;
        if (f.compare(0, 4, "QVTS") != 0) return false;
        size_t p = 4;
        auto u32 = [&]() {
            uint32_t v;
            std::memcpy(&v, f.data() + p, 4);
            p += 4;
            return v;
        };
        auto u64 = [&]() {
            uint64_t v;
            std::memcpy(&v, f.data() + p, 8);
            p += 8;
            return v;
        };
        auto str = [&]() {
            const uint32_t n = u32();
            std::string s = f.substr(p, n);
            p += n;
            return s;
        };
        const uint32_t nt = u32();
        for (uint32_t i = 0; i < nt; ++i) {
            recipe::TensorInfo ti;
            ti.name = str();
            ti.type = u32();
            for (int64_t & ne : ti.ne) ne = static_cast<int64_t>(u64());
            ti.nbytes = u64();
            const uint64_t off = u64();
            infos_[ti.name] = {ti, off};
        }
        const uint32_t nm = u32();
        for (uint32_t i = 0; i < nm; ++i) {
            TM m;
            m.key = str();
            m.mtype = str();
            m.value = json::parse(str());
            meta_[m.key] = m;
        }
        data_ = u64();
        return true;
    }

    int fd_;
    uint64_t data_ = 0;
    std::map<std::string, std::pair<recipe::TensorInfo, uint64_t>> infos_;
    std::map<std::string, TM> meta_;
};

// ---- templates ---------------------------------------------------------------------

struct RItem {
    bool let = false;
    std::string text;   // let: "name = expr"; blob, patch: the expression
    std::string dtype;  // blob, patch: fp16 | fp32
    uint64_t count = 0;
    bool patch = false;
    uint64_t spec_offset = 0;  // patch
};

inline RItem let_(const std::string & text) { return RItem{true, text, "", 0}; }
inline RItem blob(const std::string & dtype, uint64_t count, const std::string & expr) {
    return RItem{false, expr, dtype, count};
}
inline RItem patch_(uint64_t spec_offset, const std::string & dtype, uint64_t count, const std::string & expr) {
    return RItem{false, expr, dtype, count, true, spec_offset};
}

// A complete recipe with offsets, blob count and weight_bin_size filled in.
inline std::string recipe_text(const std::string & id, const std::string & spec_sha, const std::string & decls,
                               const std::vector<RItem> & items, uint64_t * wbs_out = nullptr) {
    std::string body;
    uint64_t off = milblob::HEADER_BYTES;
    uint64_t end = milblob::HEADER_BYTES;
    uint32_t n = 0;
    for (const RItem & it : items) {
        if (it.let) {
            body += "let " + it.text + "\n";
            continue;
        }
        if (it.patch) {
            body += "patch " + std::to_string(it.spec_offset) + " " + it.dtype + " " + std::to_string(it.count) + " " +
                    it.text + "\n";
            continue;
        }
        body += "blob " + std::to_string(off) + " " + it.dtype + " " + std::to_string(it.count) + " " + it.text + "\n";
        end = off + 64 + it.count * (it.dtype == "fp16" ? 2 : 4);
        off = milblob::align_up(end);
        ++n;
    }
    const uint64_t wbs = milblob::align_up(end);
    if (wbs_out) *wbs_out = wbs;
    return "mlrecipe 1\ntemplate " + id + "\nspec_sha256 " + spec_sha + "\nweight_bin_size " + std::to_string(wbs) +
           "\nblobs " + std::to_string(n) + "\n" + decls + body + "end\n";
}

inline std::string zeros64() { return std::string(64, '0'); }

struct TemplateSpec {
    std::string id = "toy-dac";
    std::string decls;  // source / meta lines
    std::vector<RItem> items;
    json match = json::object();  // {"architecture": ..., "metadata": {...}}
    std::string spec_bytes;       // model.mlmodel; default "fake-spec:<id>"
    json extra = json::object();  // merged into template.json
    std::string literals_bin;     // written as literals.bin when non-empty
};

// Writes <dir> = {model.mlmodel, Manifest.json, recipe.txt, template.json} and
// returns the loaded template.
inline std::shared_ptr<const recipe::Template> make_template(const std::string & dir, const TemplateSpec & s) {
    fs::create_directories(dir);
    const std::string spec = s.spec_bytes.empty() ? "fake-spec:" + s.id : s.spec_bytes;
    const std::string sha = detail::to_hex(detail::sha256(spec.data(), spec.size()));
    uint64_t wbs = 0;
    const std::string r = recipe_text(s.id, sha, s.decls, s.items, &wbs);
    write_bytes(dir + "/model.mlmodel", spec);
    write_bytes(dir + "/recipe.txt", r);
    if (!s.literals_bin.empty()) write_bytes(dir + "/literals.bin", s.literals_bin);
    write_bytes(dir + "/Manifest.json",
                R"({"fileFormatVersion":"1.0.0","itemInfoEntries":{},"rootModelIdentifier":"x"})");
    json t = {{"schema", 1},
              {"id", s.id},
              {"family", "toy"},
              {"stage", "dac"},
              {"functions", json::array({{{"name", "main"}}})},
              {"default_compute_units", "all"},
              {"stateful", false},
              {"shareable", true},
              {"export_kind", "toy-v1"},
              {"weight_bin_size", wbs},
              {"generation_peak_bytes", 0},
              {"match", s.match},
              {"patches", json::array()},
              {"literals", json::array()},
              {"provided", json::array()}};
    for (auto it = s.extra.begin(); it != s.extra.end(); ++it) t[it.key()] = it.value();
    write_bytes(dir + "/template.json", t.dump(1));
    return recipe::Template::from_dir(dir);
}

// ---- fake compile (the Apple compile is not portable) -----------------------------------

inline std::atomic<int> g_compiles{0};
inline std::string g_compile_marker_dir;  // cross-process count: one file per compile
inline std::function<cache::CompileResult()> g_compile_override;  // replaces the fake compile's result
inline std::function<void()> g_compile_gate;                      // runs first (holds a compile open)

inline cache::CompileResult fake_compile(const std::string & pkg, const std::string & out) {
    ++g_compiles;
    if (g_compile_gate) g_compile_gate();
    if (!g_compile_marker_dir.empty()) {
        static std::mt19937_64 rng{std::random_device{}()};
        write_bytes(g_compile_marker_dir + "/" + std::to_string(::getpid()) + "." + std::to_string(rng()), "");
    }
    if (g_compile_override) return g_compile_override();
    std::error_code ec;
    fs::create_directories(out + "/weights", ec);
    fs::copy_file(pkg + "/Data/com.apple.CoreML/model.mlmodel", out + "/model.mil", ec);
    fs::copy_file(pkg + "/Data/com.apple.CoreML/weights/weight.bin", out + "/weights/weight.bin", ec);
    write_bytes(out + "/coremldata.bin", "fake-coremldata");
    write_bytes(out + "/analytics/coremldata.bin", "fake-analytics");
    cache::CompileResult r;
    r.os_build = "TEST1";
    r.coremlc = "0.0";
    if (ec) r.error = Error::transient("fake compile: " + ec.message());
    return r;
}

inline int count_files(const std::string & dir) {
    int n = 0;
    std::error_code ec;
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) ++n;
    return n;
}

}  // namespace gtest
