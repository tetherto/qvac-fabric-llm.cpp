// The generation cache: layout, key locks, memos, artifacts, adoption, the
// prepare pipeline and prune.

#include "cache.h"

#include "failed.h"
#include "hints.h"
#include "milblob.h"
#include "sha256.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <sstream>

#include <fcntl.h>
#include <limits.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <sys/mount.h>
#include <sys/param.h>
#else
#include <sys/statfs.h>
#include <sys/statvfs.h>
#endif

namespace QVAC_COREML_NS {
namespace cache {

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

std::string posix_domain() { return "NSPOSIXErrorDomain"; }

Error io_error(const std::string & what, int e) {
    return Error::transient(what + ": " + std::strerror(e), posix_domain(), e);
}

std::string rand_hex() {
    static thread_local std::mt19937_64 rng{std::random_device{}() ^
                                            (static_cast<uint64_t>(::getpid()) << 32)};
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(rng()));
    return buf;
}

std::string real_path(const std::string & p) {
    char buf[PATH_MAX];
    if (::realpath(p.c_str(), buf) != nullptr) return buf;
    return p;
}

int64_t mtime_s(const std::string & p) {
    struct stat st{};
    if (::stat(p.c_str(), &st) != 0) return -1;
    return static_cast<int64_t>(st.st_mtime);
}

bool exists(const std::string & p) {
    struct stat st{};
    return ::lstat(p.c_str(), &st) == 0;
}

std::vector<std::string> list_dir(const std::string & dir) {
    std::vector<std::string> out;
    std::error_code ec;
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        const std::string n = it->path().filename().string();
        if (!n.empty() && n[0] != '.') out.push_back(n);
    }
    std::sort(out.begin(), out.end());
    return out;
}

void remove_tree(const std::string & p) {
    std::error_code ec;
    fs::remove_all(p, ec);
}

Error write_all(int fd, const uint8_t * p, size_t n, const std::string & what) {
    while (n > 0) {
        const ssize_t w = ::write(fd, p, n);
        if (w < 0) {
            if (errno == EINTR) continue;
            return io_error("write " + what, errno);
        }
        p += w;
        n -= static_cast<size_t>(w);
    }
    return {};
}

bool fsync_path(const std::string & p) {
    const int fd = ::open(p.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;
    const bool ok = ::fsync(fd) == 0;
    ::close(fd);
    return ok;
}

// Writes `body` to `path` through a temp file in the same directory: write,
// fsync, rename.
Error write_atomic(const std::string & path, const std::string & body) {
    const std::string dir = fs::path(path).parent_path().string();
    const std::string tmp = dir + "/.tmp." + std::to_string(::getpid()) + "." + rand_hex();
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (fd < 0) return io_error("create " + tmp, errno);
    Error e = write_all(fd, reinterpret_cast<const uint8_t *>(body.data()), body.size(), tmp);
    if (e.ok() && ::fsync(fd) != 0) e = io_error("fsync " + tmp, errno);
    ::close(fd);
    if (e.ok() && std::rename(tmp.c_str(), path.c_str()) != 0) e = io_error("rename " + tmp, errno);
    if (!e.ok()) ::unlink(tmp.c_str());
    return e;
}

std::optional<std::string> read_text(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return std::nullopt;
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

bool is_hex32(const std::string & s) {
    if (s.size() != 32) return false;
    for (char c : s) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    }
    return true;
}

// <memo_id>.<key32>.json
bool parse_memo_name(const std::string & n, std::string & memo_id, std::string & key32) {
    if (n.size() != 32 + 1 + 32 + 5 || n.compare(n.size() - 5, 5, ".json") != 0 || n[32] != '.') return false;
    memo_id = n.substr(0, 32);
    key32 = n.substr(33, 32);
    return is_hex32(memo_id) && is_hex32(key32);
}

// <template-id>.<key32>
bool parse_artifact_name(const std::string & n, std::string & tid, std::string & key32) {
    if (n.size() < 34 || n[n.size() - 33] != '.') return false;
    tid = n.substr(0, n.size() - 33);
    key32 = n.substr(n.size() - 32);
    return is_hex32(key32);
}

// quarantine/<template-id>.<key32>.<pid>[...]
bool parse_quarantine_name(const std::string & n, std::string & tid) {
    size_t pos = 0;
    while ((pos = n.find('.', pos)) != std::string::npos) {
        if (pos + 33 <= n.size() && is_hex32(n.substr(pos + 1, 32)) &&
            (pos + 33 == n.size() || n[pos + 33] == '.')) {
            tid = n.substr(0, pos);
            return !tid.empty();
        }
        ++pos;
    }
    return false;
}

int64_t ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

// ---- Layout --------------------------------------------------------------------

Layout::Layout(const std::string & cache_dir) {
    root = cache_dir + "/" + ROOT;
    sources = root + "/sources";
    artifacts = root + "/artifacts";
    locks = root + "/locks";
    staging = root + "/staging";
    used = root + "/used";
    hints = root + "/hints";
    failed = root + "/failed";
    quarantine = root + "/quarantine";
}

std::string Layout::artifact_dir(const std::string & template_id, const std::string & key32) const {
    return artifacts + "/" + template_id + "." + key32;
}
std::string Layout::lock_path(const std::string & key32) const { return locks + "/" + key32 + ".lock"; }
std::string Layout::memo_path(const std::string & memo_id, const std::string & key32) const {
    return sources + "/" + memo_id + "." + key32 + ".json";
}
std::string Layout::used_path(const std::string & key32) const { return used + "/" + key32; }

Error Layout::ensure() const {
    for (const std::string * d : {&sources, &artifacts, &locks, &staging, &used, &hints, &failed, &quarantine}) {
        std::error_code ec;
        fs::create_directories(*d, ec);
        if (ec) return Error::transient("create " + *d + ": " + ec.message(), posix_domain(), ec.value());
    }
    return {};
}

// ---- Key locks -----------------------------------------------------------------

KeyLock & KeyLock::operator=(KeyLock && o) noexcept {
    if (this != &o) {
        release();
        fd_ = o.fd_;
        o.fd_ = -1;
    }
    return *this;
}

void KeyLock::release() {
    if (fd_ >= 0) {
        ::close(fd_);  // drops the flock
        fd_ = -1;
    }
}

KeyLock::Result KeyLock::take(const Layout & l, const std::string & key32, Mode mode, KeyLock & out, Error * err) {
    out.release();
    std::error_code ec;
    fs::create_directories(l.locks, ec);
    const std::string p = l.lock_path(key32);
    const int fd = ::open(p.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (fd < 0) {
        if (err) *err = io_error("open " + p, errno);
        return Result::error;
    }
    for (;;) {
        if (::flock(fd, (mode == Mode::shared ? LOCK_SH : LOCK_EX) | LOCK_NB) == 0) break;
        const int e = errno;
        if (e == EINTR) continue;
        ::close(fd);
        if (e == EWOULDBLOCK) return Result::would_block;
        if (err) *err = io_error("flock " + p, e);
        return Result::error;
    }
    out.fd_ = fd;
    return Result::ok;
}

// ---- Memos -----------------------------------------------------------------------

std::string to_json(const Memo & m) {
    json j;
    j["schema"] = 1;
    j["path"] = m.path;
    j["size"] = m.size;
    j["mtime_ns"] = m.mtime_ns;
    j["ino"] = m.ino;
    j["dev"] = m.dev;
    j["template_sha256"] = m.template_sha256;
    j["generator_version"] = m.generator_version;
    j["options"] = m.options;
    j["template_id"] = m.template_id;
    j["source_digest"] = m.source_digest;
    j["artifact_key"] = m.artifact_key;
    return j.dump();
}

std::optional<Memo> memo_from_json(const std::string & text) {
    const json j = json::parse(text, nullptr, false);
    if (!j.is_object()) return std::nullopt;
    // Any type error (another writer's file in a shared cache) is "unreadable".
    try {
        if (j.value("schema", 0) != 1) return std::nullopt;
        Memo m;
        m.path = j.at("path").get<std::string>();
        m.size = j.at("size").get<uint64_t>();
        m.mtime_ns = j.at("mtime_ns").get<int64_t>();
        m.ino = j.at("ino").get<uint64_t>();
        m.dev = j.at("dev").get<uint64_t>();
        m.template_sha256 = j.at("template_sha256").get<std::string>();
        m.generator_version = j.at("generator_version").get<uint32_t>();
        m.options = j.at("options").get<std::string>();
        m.template_id = j.at("template_id").get<std::string>();
        m.source_digest = j.value("source_digest", std::string());
        m.artifact_key = j.at("artifact_key").get<std::string>();
        return m;
    } catch (const json::exception &) {
        return std::nullopt;
    }
}

namespace {

MemoFile load_memo_file(const Layout & l, const std::string & name, const std::string & memo_id,
                        const std::string & key32) {
    MemoFile f;
    f.file = l.sources + "/" + name;
    f.memo_id = memo_id;
    f.key32 = key32;
    f.mtime_s = mtime_s(f.file);
    if (auto t = read_text(f.file)) f.memo = memo_from_json(*t);
    return f;
}

}  // namespace

std::optional<MemoFile> find_memo(const Layout & l, const std::string & memo_id) {
    for (const std::string & n : list_dir(l.sources)) {
        std::string m, k;
        if (parse_memo_name(n, m, k) && m == memo_id) return load_memo_file(l, n, m, k);
    }
    return std::nullopt;
}

std::vector<MemoFile> memos_for_key(const Layout & l, const std::string & key32) {
    std::vector<MemoFile> out;
    for (const std::string & n : list_dir(l.sources)) {
        std::string m, k;
        if (parse_memo_name(n, m, k) && k == key32) out.push_back(load_memo_file(l, n, m, k));
    }
    return out;
}

std::vector<MemoFile> all_memos(const Layout & l) {
    std::vector<MemoFile> out;
    for (const std::string & n : list_dir(l.sources)) {
        std::string m, k;
        if (parse_memo_name(n, m, k)) out.push_back(load_memo_file(l, n, m, k));
    }
    return out;
}

bool memo_present(const Layout & l, const std::string & memo_id, const Memo & m) {
    const std::optional<std::string> t = read_text(l.memo_path(memo_id, m.artifact_key));
    if (!t) return false;
    const std::optional<Memo> on_disk = memo_from_json(*t);
    return on_disk && to_json(*on_disk) == to_json(m);
}

Error write_memo(const Layout & l, const std::string & memo_id, const Memo & m) {
    std::error_code ec;
    fs::create_directories(l.sources, ec);
    return write_atomic(l.memo_path(memo_id, m.artifact_key), to_json(m));
}

// ---- Artifacts -------------------------------------------------------------------

std::string to_json(const ArtifactInfo & a) {
    json j;
    j["schema"] = 1;
    j["complete"] = a.complete;
    j["artifact_key"] = a.artifact_key;
    j["template"] = {{"id", a.template_id}, {"sha256", a.template_sha256}};
    j["generator_version"] = a.generator_version;
    j["options"] = a.options;
    j["source"] = {{"digest", a.source_digest},
                   {"tensors", a.source_tensors},
                   {"bytes", a.source_bytes},
                   {"provenance",
                    {{"path", a.provenance.path},
                     {"size", a.provenance.size},
                     {"mtime_ns", a.provenance.mtime_ns},
                     {"ino", a.provenance.ino},
                     {"dev", a.provenance.dev}}}};
    j["compiled"] = {{"os_build", a.os_build}, {"coremlc", a.coremlc}};
    j["export_kind"] = a.export_kind;
    j["functions"] = a.functions;
    j["default_compute_units"] = a.default_compute_units;
    json inv = json::array();
    for (const InventoryEntry & e : a.inventory) inv.push_back({{"path", e.path}, {"bytes", e.bytes}});
    j["inventory"] = inv;
    j["timings_ms"] = {{"digest", a.digest_ms}, {"write", a.write_ms}, {"compile", a.compile_ms}};
    return j.dump(1);
}

std::optional<ArtifactInfo> read_artifact_json(const std::string & dir) {
    auto t = read_text(dir + "/artifact.json");
    if (!t) return std::nullopt;
    const json j = json::parse(*t, nullptr, false);
    if (!j.is_object()) return std::nullopt;
    try {
        if (j.value("schema", 0) != 1) return std::nullopt;
        ArtifactInfo a;
        a.complete = j.at("complete").get<bool>();
        a.artifact_key = j.at("artifact_key").get<std::string>();
        a.template_id = j.at("template").at("id").get<std::string>();
        a.template_sha256 = j.at("template").at("sha256").get<std::string>();
        a.generator_version = j.at("generator_version").get<uint32_t>();
        a.options = j.value("options", std::string());
        if (j.contains("source")) {
            const json & s = j["source"];
            a.source_digest = s.value("digest", std::string());
            a.source_tensors = s.value("tensors", 0u);
            a.source_bytes = s.value("bytes", uint64_t{0});
        }
        if (j.contains("compiled")) {
            a.os_build = j["compiled"].value("os_build", std::string());
            a.coremlc = j["compiled"].value("coremlc", std::string());
        }
        a.export_kind = j.value("export_kind", std::string());
        a.functions = j.value("functions", std::vector<std::string>{});
        a.default_compute_units = j.value("default_compute_units", std::string());
        for (const json & e : j.at("inventory")) {
            a.inventory.push_back({e.at("path").get<std::string>(), e.at("bytes").get<uint64_t>()});
        }
        return a;
    } catch (const json::exception &) {
        return std::nullopt;
    }
}

Validity validate(const std::string & dir, std::string * why) {
    auto bad = [&](Validity v, const std::string & m) {
        if (why) *why = m;
        return v;
    };
    if (!exists(dir)) return bad(Validity::missing, "no artifact directory");
    if (!exists(dir + "/artifact.json")) return bad(Validity::invalid, "artifact.json is missing");
    const std::optional<ArtifactInfo> a = read_artifact_json(dir);
    if (!a) return bad(Validity::invalid, "artifact.json is unreadable or not schema 1");
    if (!a->complete) return bad(Validity::invalid, "artifact.json is not complete");
    if (a->inventory.empty()) return bad(Validity::invalid, "empty inventory");
    for (const InventoryEntry & e : a->inventory) {
        if (e.path.empty() || e.path[0] == '/' || e.path.find("..") != std::string::npos) {
            return bad(Validity::invalid, "bad inventory path " + e.path);
        }
        struct stat st{};
        if (::lstat((dir + "/" + e.path).c_str(), &st) != 0 || !S_ISREG(st.st_mode)) {
            return bad(Validity::invalid, e.path + " is missing");
        }
        if (static_cast<uint64_t>(st.st_size) != e.bytes) {
            return bad(Validity::invalid, e.path + " is " + std::to_string(st.st_size) + " bytes, not " +
                                              std::to_string(e.bytes));
        }
    }
    return Validity::ok;
}

std::vector<InventoryEntry> inventory_of(const std::string & base, const std::string & root) {
    std::vector<InventoryEntry> out;
    std::error_code ec;
    for (fs::recursive_directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec)) {
        if (!it->is_regular_file(ec)) continue;
        InventoryEntry e;
        e.path = fs::relative(it->path(), base, ec).generic_string();
        e.bytes = it->file_size(ec);
        out.push_back(e);
    }
    std::sort(out.begin(), out.end(), [](const InventoryEntry & a, const InventoryEntry & b) { return a.path < b.path; });
    return out;
}

// ---- Adoption --------------------------------------------------------------------

Adoption adopt(const Layout & l, const std::string & template_id, const std::string & key32,
               const std::string * memo_id, const Memo * memo) {
    Adoption a;
    a.dir = l.artifact_dir(template_id, key32);
    switch (KeyLock::take(l, key32, KeyLock::Mode::shared, a.lease, &a.error)) {
        case KeyLock::Result::would_block: a.kind = Adoption::Kind::would_block; return a;
        case KeyLock::Result::error: a.kind = Adoption::Kind::error; return a;
        case KeyLock::Result::ok: break;
    }
    std::string why;
    const Validity v = validate(a.dir, &why);
    if (v != Validity::ok) {
        a.lease.release();
        a.kind = v == Validity::missing ? Adoption::Kind::missing : Adoption::Kind::invalid;
        a.error = Error::transient("artifact " + a.dir + ": " + why, failed::ERROR_DOMAIN, failed::INVALID_ARTIFACT);
        return a;
    }
    if (memo_id && memo && !memo_present(l, *memo_id, *memo)) {
        // The memo is the persistent reference that keeps the artifact from
        // prune once this lease is gone: no memo, no adoption.
        if (Error e = write_memo(l, *memo_id, *memo); !e.ok()) {
            a.lease.release();
            a.kind = Adoption::Kind::error;
            a.error = std::move(e);
            return a;
        }
    }
    a.kind = Adoption::Kind::adopted;
    return a;
}

namespace {

// Holding EX on the key.
Error quarantine_locked(const Layout & l, const std::string & template_id, const std::string & key32) {
    const std::string src = l.artifact_dir(template_id, key32);
    if (!exists(src)) return {};
    std::error_code ec;
    fs::create_directories(l.quarantine, ec);
    const std::string base = l.quarantine + "/" + template_id + "." + key32 + "." + std::to_string(::getpid());
    std::string dst = base;
    for (int n = 1; exists(dst); ++n) dst = base + "." + std::to_string(n);
    if (std::rename(src.c_str(), dst.c_str()) != 0) return io_error("quarantine " + src, errno);
    return {};
}

}  // namespace

KeyLock::Result quarantine(const Layout & l, const std::string & template_id, const std::string & key32, Error * err) {
    KeyLock ex;
    const KeyLock::Result r = KeyLock::take(l, key32, KeyLock::Mode::exclusive, ex, err);
    if (r != KeyLock::Result::ok) return r;
    Error e = quarantine_locked(l, template_id, key32);
    if (!e.ok()) {
        if (err) *err = e;
        return KeyLock::Result::error;
    }
    return KeyLock::Result::ok;
}

bool artifact_identity(const Layout & l, const std::string & template_id, const std::string & key32, uint64_t & dev,
                       uint64_t & ino) {
    struct stat st{};
    if (::stat(l.artifact_dir(template_id, key32).c_str(), &st) != 0) return false;
    dev = static_cast<uint64_t>(st.st_dev);
    ino = static_cast<uint64_t>(st.st_ino);
    return true;
}

KeyLock::Result quarantine_if(const Layout & l, const std::string & template_id, const std::string & key32,
                              uint64_t dev, uint64_t ino, bool * replaced, Error * err) {
    if (replaced) *replaced = false;
    KeyLock ex;
    const KeyLock::Result r = KeyLock::take(l, key32, KeyLock::Mode::exclusive, ex, err);
    if (r != KeyLock::Result::ok) return r;
    uint64_t now_dev = 0, now_ino = 0;
    if (!artifact_identity(l, template_id, key32, now_dev, now_ino) || now_dev != dev || now_ino != ino) {
        if (replaced) *replaced = true;
        return KeyLock::Result::ok;
    }
    Error e = quarantine_locked(l, template_id, key32);
    if (!e.ok()) {
        if (err) *err = e;
        return KeyLock::Result::error;
    }
    return KeyLock::Result::ok;
}

std::string prune_trash_path(const Layout & l, const std::string & artifact_name) {
    return l.quarantine + "/" + artifact_name + "." + std::to_string(::getpid()) + ".prune." + rand_hex();
}

void touch_used(const Layout & l, const std::string & key32) {
    std::error_code ec;
    fs::create_directories(l.used, ec);
    const std::string p = l.used_path(key32);
    const int fd = ::open(p.c_str(), O_WRONLY | O_CREAT | O_CLOEXEC, 0644);
    if (fd >= 0) ::close(fd);
    ::utimes(p.c_str(), nullptr);
}

// ---- Filesystem pre-flight ---------------------------------------------------------

bool default_fs_probe(const std::string & dir, FsFacts & out) {
#if defined(__APPLE__)
    struct statfs s{};
    if (::statfs(dir.c_str(), &s) != 0) return false;
    out.free_bytes = static_cast<uint64_t>(s.f_bavail) * static_cast<uint64_t>(s.f_bsize);
    out.local = (s.f_flags & MNT_LOCAL) != 0;
    out.fstype = s.f_fstypename;
#else
    struct statfs s{};
    if (::statfs(dir.c_str(), &s) != 0) return false;
    out.free_bytes = static_cast<uint64_t>(s.f_bavail) * static_cast<uint64_t>(s.f_bsize);
    switch (static_cast<unsigned long>(s.f_type)) {
        case 0xEF53UL: out.fstype = "ext4"; break;
        case 0x01021994UL: out.fstype = "tmpfs"; break;
        case 0x58465342UL: out.fstype = "xfs"; break;
        case 0x9123683EUL: out.fstype = "btrfs"; break;
        case 0x794C7630UL: out.fstype = "overlay"; break;
        case 0xF2F52010UL: out.fstype = "f2fs"; break;
        default: out.fstype = "unknown"; break;
    }
    out.local = out.fstype != "unknown";
#endif
    return true;
}

bool fs_supported(const FsFacts & f) {
    if (!f.local) return false;
#if defined(__APPLE__)
    return f.fstype == "apfs" || f.fstype == "hfs";
#else
    return f.fstype != "unknown";
#endif
}

// ---- Writing a package (step 5) ------------------------------------------------------

Error write_package(const recipe::Template & t, recipe::TensorSource & src, const std::string & pkg,
                    recipe::EvalStats * stats, const recipe::ReadObserver & observer) {
    const std::string data = pkg + "/Data/com.apple.CoreML";
    std::error_code ec;
    fs::create_directories(data + "/weights", ec);
    if (ec) return Error::transient("create " + data + ": " + ec.message(), posix_domain(), ec.value());
    // The spec with its patches applied (inline constants computed from the
    // GGUF); a template without patch lines is written as it is.
    const recipe::TemplateFile * spec = t.file("model.mlmodel");
    std::vector<uint8_t> patched;
    recipe::EvalStats patch_stats;
    if (t.recipe().patch_count() > 0) {
        Error pe = recipe::patched_spec(t.recipe(), src, spec->data, spec->size, patched, observer, &patch_stats);
        if (!pe.ok()) return pe;
    }
    for (const auto & [name, dst] : {std::pair<const char *, std::string>{"Manifest.json", pkg + "/Manifest.json"},
                                     std::pair<const char *, std::string>{"model.mlmodel", data + "/model.mlmodel"}}) {
        const recipe::TemplateFile * f = t.file(name);
        const uint8_t * bytes = f->data;
        size_t size = f->size;
        if (f == spec && !patched.empty()) {
            bytes = patched.data();
            size = patched.size();
        }
        const int fd = ::open(dst.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
        if (fd < 0) return io_error("create " + dst, errno);
        Error e = write_all(fd, bytes, size, dst);
        if (e.ok() && ::fsync(fd) != 0) e = io_error("fsync " + dst, errno);
        ::close(fd);
        if (!e.ok()) return e;
    }
    std::vector<uint8_t>().swap(patched);  // written: the blob pass never holds the spec
    const std::string wb = data + "/weights/weight.bin";
    const int fd = ::open(wb.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) return io_error("create " + wb, errno);
    std::vector<uint8_t> buf;
    buf.reserve(1u << 20);
    auto flush = [&]() -> Error {
        Error e = write_all(fd, buf.data(), buf.size(), wb);
        buf.clear();
        return e;
    };
    const recipe::Recipe & r = t.recipe();
    milblob::Writer w(r.blob_count, r.weight_bin_size, [&](const uint8_t * p, size_t n) -> Error {
        if (buf.size() + n > buf.capacity()) {
            Error e = flush();
            if (!e.ok()) return e;
        }
        if (n >= buf.capacity()) return write_all(fd, p, n, wb);
        buf.insert(buf.end(), p, p + n);
        return {};
    });
    Error e = w.begin();
    if (e.ok()) {
        e = recipe::evaluate(
            r, src,
            [&](const recipe::BlobOut & b) {
                return w.blob(b.offset, b.dtype == recipe::BlobType::fp16 ? milblob::FLOAT16 : milblob::FLOAT32,
                              b.data, b.bytes);
            },
            stats, observer);
    }
    if (stats) {
        // The two passes run one after the other: the larger peak is the generator's.
        stats->peak_bytes = std::max(stats->peak_bytes, patch_stats.peak_bytes);
        stats->tensors_read += patch_stats.tensors_read;
    }
    if (e.ok()) e = w.finish();
    if (e.ok()) e = flush();
    if (e.ok() && ::fsync(fd) != 0) e = io_error("fsync " + wb, errno);
    ::close(fd);
    return e;
}

// ---- The prepare pipeline ------------------------------------------------------------

namespace {

struct StagingGuard {
    std::string dir;
    ~StagingGuard() {
        if (!dir.empty()) remove_tree(dir);
    }
};

void fsync_tree(const std::string & root) {
    std::error_code ec;
    for (fs::recursive_directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec)) {
        fsync_path(it->path().string());
    }
    fsync_path(root);
}

struct Job {
    const PrepareInputs & in;
    PrepareContext * ctx;
    const recipe::Template & t;
    Layout l;
    OpenSource & src;
    SourceStat st0;
    std::string memo_id;
    Memo memo;
    digest::SourceDigest dg;
    PrepareOutcome & out;
    int64_t now;

    void phase(PreparePhase p) {
        if (ctx) ctx->phase(p);
    }

    // Under EX: steps 5-7. Returns the failing stage with the error.
    std::pair<Error, failed::Stage> generate(const std::string & key) {
        StagingGuard guard;
        const std::string pid = std::to_string(::getpid());
        const std::string nonce = rand_hex();
        guard.dir = l.staging + "/" + key + "." + pid + "." + nonce;
        // Core ML compiles into <user temp>/<package stem>.mlmodelc, which every
        // process and cache directory shares, while the key lock covers only
        // this cache directory: the stem is unique to this attempt so that
        // concurrent prepares of the same key never meet there. The
        // qvac-coreml- prefix lets routine prune sweep a killed compile's output.
        const std::string pkg = guard.dir + "/qvac-coreml-" + key + "-" + pid + "-" + nonce + ".mlpackage";
        {
            std::error_code ec;
            fs::create_directories(guard.dir, ec);
            if (ec) {
                return {Error::transient("create " + guard.dir + ": " + ec.message(), posix_domain(), ec.value()),
                        failed::Stage::prepare};
            }
            // The owner, so prune can tell this library's staging from another addon's.
            Error oe = write_atomic(guard.dir + "/" + STAGING_OWNER_FILE, t.id());
            if (!oe.ok()) return {oe, failed::Stage::prepare};
        }
        phase(PreparePhase::generating);
        auto t0 = std::chrono::steady_clock::now();
        recipe::EvalStats stats;
        // The digest is recomputed during the write: every tensor read must
        // carry the bytes the key was computed from.
        auto observer = [&](const std::string & name, const std::array<uint8_t, 32> & sha) -> Error {
            auto it = dg.tensor_sha256.find(name);
            if (it == dg.tensor_sha256.end() || it->second != sha) {
                return Error::transient("source_changed: tensor " + name + " changed after the digest", failed::ERROR_DOMAIN,
                                        failed::SOURCE_CHANGED);
            }
            return {};
        };
        Error e = write_package(t, src, pkg, &stats, observer);
        out.generation_peak_bytes = stats.peak_bytes;
        if (e.ok()) {
            SourceStat st1;
            if (!src.stat_now(st1) || !same_identity(st0, st1)) {
                e = Error::transient("source_changed: the GGUF changed during generation", failed::ERROR_DOMAIN,
                                     failed::SOURCE_CHANGED);
            }
        }
        out.write_ms = ms_since(t0);
        if (!e.ok()) return {e, failed::Stage::prepare};

        // Step 6: compile into staging/.../publish/model.mlmodelc.
        phase(PreparePhase::compiling);
        const std::string publish = guard.dir + "/publish";
        std::error_code ec;
        fs::create_directories(publish, ec);
        if (ec) return {Error::transient("create " + publish + ": " + ec.message(), posix_domain(), ec.value()),
                        failed::Stage::prepare};
        t0 = std::chrono::steady_clock::now();
        CompileResult cr = in.compile(pkg, publish + "/model.mlmodelc");
        out.compile_ms = ms_since(t0);
        out.compile_peak_bytes = cr.compile_peak_bytes;
        if (!cr.error.ok()) return {cr.error, cr.stage};

        ArtifactInfo a;
        a.artifact_key = key;
        a.template_id = t.id();
        a.template_sha256 = t.sha256_hex();
        a.generator_version = GENERATOR_VERSION;
        a.options = in.options;
        a.source_digest = detail::to_hex(dg.digest);
        a.source_tensors = dg.tensors;
        a.source_bytes = dg.bytes;
        a.provenance = st0;
        a.provenance.path = memo.path;
        a.os_build = cr.os_build;
        a.coremlc = cr.coremlc;
        a.export_kind = t.info().export_kind;
        a.functions = t.info().functions;
        a.default_compute_units = t.info().default_compute_units;
        a.inventory = inventory_of(publish, publish + "/model.mlmodelc");
        a.digest_ms = out.digest_ms;
        a.write_ms = out.write_ms;
        a.compile_ms = out.compile_ms;
        if (a.inventory.empty()) {
            return {Error::deterministic("compile produced no model.mlmodelc"), failed::Stage::compile};
        }
        e = write_atomic(publish + "/artifact.json", to_json(a));
        if (!e.ok()) return {e, failed::Stage::prepare};

        // Step 7: publish.
        fsync_tree(publish);
        const std::string dst = l.artifact_dir(t.id(), key);
        if (std::rename(publish.c_str(), dst.c_str()) != 0) {
            const int err = errno;
            if (err != EEXIST && err != ENOTEMPTY) return {io_error("publish " + dst, err), failed::Stage::prepare};
            // The target exists: the winner's copy is used.
        }
        fsync_path(l.artifacts);
        e = write_memo(l, memo_id, memo);
        if (!e.ok()) return {e, failed::Stage::prepare};
        return {Error{}, failed::Stage::prepare};
    }
};

PrepareOutcome failed_outcome(PrepareOutcome & out, Error e, std::string reason = {}) {
    out.kind = PrepareOutcome::Kind::failed;
    out.error = std::move(e);
    out.reason = std::move(reason);
    return std::move(out);
}

}  // namespace

PrepareOutcome prepare(const PrepareInputs & in, PrepareContext * ctx) {
    PrepareOutcome out;
    if (!in.tmpl || in.cache_dir.empty() || !in.compile || !in.open) {
        return failed_outcome(out, Error::deterministic("prepare: template, cache_dir, compile and open are required"));
    }
    const recipe::Template & t = *in.tmpl;
    const int64_t now = in.now_s != 0 ? in.now_s : detail::unix_now_s();
    Layout l(in.cache_dir);
    if (Error e = l.ensure(); !e.ok()) return failed_outcome(out, e);

    // Step 1: open the source once; it must be the file the engine loaded.
    if (in.expected.changed) {
        return failed_outcome(out,
                              Error::transient("source_changed: " + in.expected.path +
                                                   " is not certainly the file the engine loaded its weights from",
                                               failed::ERROR_DOMAIN, failed::SOURCE_CHANGED),
                              "source_changed");
    }
    Error err;
    std::unique_ptr<OpenSource> src = in.open(in.expected.path, err);
    if (!src) return failed_outcome(out, err.ok() ? Error::transient("cannot open " + in.expected.path) : err);
    SourceStat st0;
    if (!src->stat_now(st0)) return failed_outcome(out, io_error("fstat " + in.expected.path, errno));
    if (!same_identity(st0, in.expected)) {
        return failed_outcome(out,
                              Error::transient("source_changed: " + in.expected.path +
                                                   " is not the file the engine loaded",
                                               failed::ERROR_DOMAIN, failed::SOURCE_CHANGED),
                              "source_changed");
    }
    const std::string rp = real_path(in.expected.path);
    out.memo_id = digest::memo_id(rp, st0.size, st0.mtime_ns, st0.ino, st0.dev, t.sha256(), GENERATOR_VERSION,
                                  in.options);
    Job job{in, ctx, t, l, *src, st0, out.memo_id, Memo{}, digest::SourceDigest{}, out, now};
    job.memo.path = rp;
    job.memo.size = st0.size;
    job.memo.mtime_ns = st0.mtime_ns;
    job.memo.ino = st0.ino;
    job.memo.dev = st0.dev;
    job.memo.template_sha256 = t.sha256_hex();
    job.memo.generator_version = GENERATOR_VERSION;
    job.memo.options = in.options;
    job.memo.template_id = t.id();
    bool digested = false;

    auto ready = [&](Adoption & a, const std::string & key, bool hit, bool memo_hit) {
        out.kind = PrepareOutcome::Kind::ready;
        out.cache_hit = hit;
        out.memo_hit = memo_hit;
        out.key32 = key;
        out.artifact_dir = a.dir;
        out.model_path = a.dir + "/model.mlmodelc";
        out.lease = std::move(a.lease);
        return std::move(out);
    };
    auto would_block = [&]() {
        out.kind = PrepareOutcome::Kind::would_block;
        return std::move(out);
    };

    for (int round = 0; round < 4; ++round) {
        // Step 2: memo lookup (the hot path: no hashing).
        if (!digested) {
            if (std::optional<MemoFile> mf = find_memo(l, out.memo_id);
                mf && mf->memo && mf->memo->template_id == t.id() && mf->memo->artifact_key == mf->key32) {
                if (in.after_memo_lookup) in.after_memo_lookup();
                // Re-checks the memo under the lease: a prune may have deleted
                // it since the lookup.
                Adoption a = adopt(l, t.id(), mf->key32, &out.memo_id, &*mf->memo);
                if (a.kind == Adoption::Kind::adopted) return ready(a, mf->key32, true, true);
                if (a.kind == Adoption::Kind::would_block) return would_block();
                if (a.kind == Adoption::Kind::error) return failed_outcome(out, a.error);
                // missing or invalid: regenerate (the digest gives the same key)
            }
            const recipe::MatchResult m = recipe::match(t, *src);
            if (!m.ok) return failed_outcome(out, Error::deterministic("template does not match: " + m.reason));
            job.phase(PreparePhase::digesting);
            const auto t0 = std::chrono::steady_clock::now();
            Error e = digest::source_digest(t.recipe(), *src, job.dg);
            out.digest_ms = ms_since(t0);
            if (!e.ok()) return failed_outcome(out, e);
            digested = true;
            out.key32 = digest::artifact_key(GENERATOR_VERSION, t.sha256(), job.dg.digest, in.options);
            job.memo.source_digest = detail::to_hex(job.dg.digest);
            job.memo.artifact_key = out.key32;
        }
        const std::string key = out.key32;

        failed::Verdict fv = failed::check(l.failed, key, in.os_build, now);
        if (fv.blocked) return failed_outcome(out, fv.error, "failed");

        Adoption a = adopt(l, t.id(), key, &out.memo_id, &job.memo);
        if (a.kind == Adoption::Kind::adopted) return ready(a, key, true, false);
        if (a.kind == Adoption::Kind::would_block) return would_block();
        if (a.kind == Adoption::Kind::error) return failed_outcome(out, a.error);

        // Step 3: pre-flight.
        FsFacts f;
        const FsProbe & probe = in.fs_probe ? in.fs_probe : FsProbe(default_fs_probe);
        if (!probe(l.root, f)) return failed_outcome(out, io_error("statfs " + l.root, errno));
        if (!fs_supported(f)) {
            out.kind = PrepareOutcome::Kind::unsupported;
            out.reason = "cache_fs";
            out.error = Error::deterministic("cache volume is " + f.fstype + (f.local ? "" : " (not local)") +
                                                 ": generation needs a local apfs or hfs volume",
                                             failed::ERROR_DOMAIN, failed::CACHE_FS);
            return out;
        }

        // Step 4: generate under LOCK_EX|LOCK_NB, re-checking first.
        KeyLock ex;
        Error le;
        switch (KeyLock::take(l, key, KeyLock::Mode::exclusive, ex, &le)) {
            case KeyLock::Result::would_block: return would_block();
            case KeyLock::Result::error: return failed_outcome(out, le);
            case KeyLock::Result::ok: break;
        }
        const std::string dir = l.artifact_dir(t.id(), key);
        const Validity v = validate(dir);
        if (v == Validity::ok) {
            // Another process published meanwhile: write our memo, adopt next round.
            failed::clear(l.failed, key);
            if (Error me = write_memo(l, out.memo_id, job.memo); !me.ok()) return failed_outcome(out, me);
            ex.release();
            continue;
        }
        // The space check of the pre-flight runs under EX, so its failure
        // record is ordered against another process's success like the
        // generation outcome below (a late record would revive a cleared streak).
        if (f.free_bytes < 2 * t.recipe().weight_bin_size) {
            Error e = Error::transient("no_space: " + std::to_string(f.free_bytes) + " bytes free, need " +
                                           std::to_string(2 * t.recipe().weight_bin_size),
                                       failed::ERROR_DOMAIN, failed::NO_SPACE);
            failed::record(l.failed, key, failed::Stage::prepare, e, t.id(), in.os_build, now);
            return failed_outcome(out, e, "no_space");  // ~KeyLock releases EX after the record
        }
        if (v == Validity::invalid) {
            // Unusable artifact: aside, regenerate at the same key.
            if (Error qe = quarantine_locked(l, t.id(), key); !qe.ok()) return failed_outcome(out, qe);
        }
        auto [ge, stage] = job.generate(key);
        // Both outcomes are published under EX, so a failure record and the
        // clear of a later success are ordered by the key lock: a failure can
        // never land after another process's success and resurrect a streak.
        if (ge.ok()) {
            failed::clear(l.failed, key);  // a success ends the failure streak
        } else {
            failed::record(l.failed, key, stage, ge, t.id(), in.os_build, now);
        }
        ex.release();
        if (in.after_key_unlock) in.after_key_unlock();
        if (!ge.ok()) {
            const bool changed = ge.domain == failed::ERROR_DOMAIN && ge.code == failed::SOURCE_CHANGED;
            return failed_outcome(out, ge, changed ? "source_changed" : std::string(failed::to_string(stage)));
        }
        // Step 8: adopt (the lease) and hand over to the load lane.
        Adoption mine = adopt(l, t.id(), key, &out.memo_id, &job.memo);
        if (mine.kind == Adoption::Kind::adopted) return ready(mine, key, false, false);
        if (mine.kind == Adoption::Kind::would_block) return would_block();
        if (mine.kind == Adoption::Kind::error) return failed_outcome(out, mine.error);
        // Pruned between publish and adoption: go round again.
    }
    return failed_outcome(out, Error::transient("artifact " + out.key32 + " kept disappearing"));
}

// ---- Prune -------------------------------------------------------------------------

bool Catalog::owns(const std::string & id) const {
    for (const auto & t : templates) {
        if (t.first == id) return true;
    }
    return std::find(retired_ids.begin(), retired_ids.end(), id) != retired_ids.end();
}

bool Catalog::current(const std::string & id, const std::string & sha, uint32_t gv) const {
    if (gv != generator_version) return false;
    for (const auto & t : templates) {
        if (t.first == id && t.second == sha) return true;
    }
    return false;
}

PrunePolicy PrunePolicy::routine(int64_t grace_s) {
    PrunePolicy p;
    p.mode = Mode::routine;
    p.grace_s = grace_s;
    return p;
}

PrunePolicy PrunePolicy::scoped(std::vector<std::string> dead_paths) {
    PrunePolicy p;
    p.mode = Mode::scoped;
    p.dead_paths = std::move(dead_paths);
    return p;
}

PrunePolicy PrunePolicy::all() {
    PrunePolicy p;
    p.mode = Mode::all;
    return p;
}

bool memo_live(const Memo & m, const Catalog & c) {
    if (!c.current(m.template_id, m.template_sha256, m.generator_version)) return false;
    SourceStat st;
    if (!stat_path(m.path, st)) return false;
    SourceStat rec;
    rec.size = m.size;
    rec.mtime_ns = m.mtime_ns;
    rec.ino = m.ino;
    rec.dev = m.dev;
    return same_identity(st, rec);
}

namespace {

// Spellings a deleted path may have in a memo (memos hold realpaths, and the
// file itself is gone).
std::vector<std::string> scope_roots(const std::vector<std::string> & dead_paths) {
    std::vector<std::string> out;
    for (const std::string & d : dead_paths) {
        if (d.empty()) continue;
        std::string n = fs::path(d).lexically_normal().string();
        while (n.size() > 1 && n.back() == '/') n.pop_back();
        out.push_back(n);
        const fs::path p(n);
        if (exists(n)) out.push_back(real_path(n));
        if (p.has_parent_path() && exists(p.parent_path().string())) {
            out.push_back((fs::path(real_path(p.parent_path().string())) / p.filename()).string());
        }
    }
    return out;
}

bool under(const std::string & path, const std::vector<std::string> & roots) {
    for (const std::string & r : roots) {
        if (path == r) return true;
        if (path.size() > r.size() && path.compare(0, r.size(), r) == 0 && (r == "/" || path[r.size()] == '/')) {
            return true;
        }
    }
    return false;
}

struct Pruner {
    const Layout & l;
    const PrunePolicy & p;
    const Catalog & c;
    int64_t now;
    std::vector<std::string> roots;
    PruneReport rep;

    bool old(int64_t t) const { return t >= 0 && now - t > p.grace_s; }

    void delete_memo(const MemoFile & m) {
        if (::unlink(m.file.c_str()) == 0) rep.deleted_memos.push_back(fs::path(m.file).filename().string());
    }

    // Holding EX: memos first, then the artifact (renamed aside, then removed),
    // so a crash never leaves a memo pointing at nothing.
    void delete_artifact(const std::string & name, const std::string & key, const std::vector<MemoFile> & memos) {
        for (const MemoFile & m : memos) delete_memo(m);
        const std::string src = l.artifacts + "/" + name;
        std::error_code ec;
        fs::create_directories(l.quarantine, ec);
        const std::string trash = prune_trash_path(l, name);
        if (std::rename(src.c_str(), trash.c_str()) == 0) {
            remove_tree(trash);
        } else {
            remove_tree(src);
        }
        ::unlink(l.used_path(key).c_str());
        rep.deleted_artifacts.push_back(name);
    }

    void artifacts(std::set<std::string> & seen_keys) {
        for (const std::string & name : list_dir(l.artifacts)) {
            std::string tid, key;
            if (!parse_artifact_name(name, tid, key) || !c.owns(tid)) continue;  // unknown: another addon's
            seen_keys.insert(key);
            KeyLock ex;
            if (KeyLock::take(l, key, KeyLock::Mode::exclusive, ex) != KeyLock::Result::ok) {
                rep.busy_artifacts.push_back(name);
                continue;
            }
            const std::string dir = l.artifacts + "/" + name;
            if (!exists(dir)) continue;  // deleted meanwhile
            // Re-scan under EX.
            std::vector<MemoFile> memos;
            for (MemoFile & m : memos_for_key(l, key)) {
                if (m.memo && m.memo->template_id == tid) memos.push_back(std::move(m));
            }
            std::vector<bool> live(memos.size());
            bool any_live = false;
            for (size_t i = 0; i < memos.size(); ++i) {
                live[i] = memo_live(*memos[i].memo, c);
                any_live = any_live || live[i];
            }
            switch (p.mode) {
                case PrunePolicy::Mode::all: delete_artifact(name, key, memos); break;
                case PrunePolicy::Mode::routine: {
                    int64_t age = mtime_s(l.used_path(key));
                    if (age < 0) age = mtime_s(dir + "/artifact.json");
                    if (age < 0) age = mtime_s(dir);
                    if (!any_live && old(age)) {
                        delete_artifact(name, key, memos);
                    } else {
                        for (size_t i = 0; i < memos.size(); ++i) {
                            if (!live[i] && old(memos[i].mtime_s)) delete_memo(memos[i]);
                        }
                    }
                    break;
                }
                case PrunePolicy::Mode::scoped: {
                    bool all_dead_in_scope = !memos.empty();
                    for (size_t i = 0; i < memos.size(); ++i) {
                        if (live[i] || !under(memos[i].memo->path, roots)) all_dead_in_scope = false;
                    }
                    if (all_dead_in_scope) {
                        delete_artifact(name, key, memos);
                    } else {
                        for (size_t i = 0; i < memos.size(); ++i) {
                            if (!live[i] && under(memos[i].memo->path, roots)) delete_memo(memos[i]);
                        }
                    }
                    break;
                }
            }
        }
    }

    // Memos whose artifact is gone.
    void orphan_memos(const std::set<std::string> & seen_keys) {
        for (const MemoFile & m : all_memos(l)) {
            if (seen_keys.count(m.key32) || !m.memo || !c.owns(m.memo->template_id)) continue;
            KeyLock ex;
            if (KeyLock::take(l, m.key32, KeyLock::Mode::exclusive, ex) != KeyLock::Result::ok) continue;
            if (exists(l.artifact_dir(m.memo->template_id, m.key32))) continue;  // published meanwhile
            const bool live = memo_live(*m.memo, c);
            bool del = false;
            switch (p.mode) {
                case PrunePolicy::Mode::all: del = true; break;
                case PrunePolicy::Mode::routine: del = !live && old(m.mtime_s); break;
                case PrunePolicy::Mode::scoped: del = !live && under(m.memo->path, roots); break;
            }
            if (del && exists(m.file)) delete_memo(m);
        }
    }

    void staging() {
        if (p.mode == PrunePolicy::Mode::scoped) return;
        for (const std::string & n : list_dir(l.staging)) {
            const std::string dir = l.staging + "/" + n;
            if (p.mode == PrunePolicy::Mode::routine && !(mtime_s(dir) >= 0 && now - mtime_s(dir) > 3600)) continue;
            // <key32>.<pid>.<rand>; anything else has no key lock to take.
            const std::string key = n.substr(0, 32);
            if (!is_hex32(key) || n.size() <= 32 || n[32] != '.') continue;
            KeyLock ex;
            if (KeyLock::take(l, key, KeyLock::Mode::exclusive, ex) != KeyLock::Result::ok) continue;  // in use
            // Under EX: only staging this library owns; an owner that cannot be
            // established (no or unreadable owner file) is left alone.
            const std::optional<std::string> owner = read_text(dir + "/" + STAGING_OWNER_FILE);
            if (!owner || !c.owns(*owner)) continue;
            remove_tree(dir);
            ++rep.staging;
        }
    }

    void tmp() {
        if (p.mode != PrunePolicy::Mode::routine) return;
        std::string dir = p.tmp_dir;
        if (dir.empty()) {
            const char * e = std::getenv("TMPDIR");
            dir = e && *e ? e : "/tmp";
        }
        for (const std::string & n : list_dir(dir)) {
            if (n.rfind("qvac-coreml-", 0) != 0 || n.size() < 9 || n.compare(n.size() - 9, 9, ".mlmodelc") != 0) continue;
            const std::string pth = dir + "/" + n;
            const int64_t t = mtime_s(pth);
            if (t >= 0 && now - t > 3600) {
                remove_tree(pth);
                ++rep.tmp;
            }
        }
    }

    // A record this library owns that the policy deletes: any in `all`, an
    // expired one in `routine`.
    bool stale_record(const std::string & pth) const {
        const std::optional<failed::Record> r = failed::read(pth);
        return r && c.owns(r->template_id) && (p.mode == PrunePolicy::Mode::all || failed::expired(*r, now));
    }

    // Generation writes and clears a key's failure records under the key's
    // EX, and a loaded model holds the key's lease. So a record is deleted
    // only under EX and after a second read: a replacement written since the
    // scan (a fresh backoff, a promotion to deterministic) is kept. A held key
    // is skipped, never waited for.
    void failed_records() {
        if (p.mode == PrunePolicy::Mode::scoped) return;
        for (const std::string & n : list_dir(l.failed)) {
            // <key32>.json or <key32>.<fn_units_hash>.json; anything else has no key lock.
            const std::string key = n.substr(0, 32);
            if (!is_hex32(key) || n.size() <= 32 || n[32] != '.') continue;
            const std::string pth = l.failed + "/" + n;
            if (!stale_record(pth)) continue;
            if (p.after_failed_scan) p.after_failed_scan();
            KeyLock ex;
            if (KeyLock::take(l, key, KeyLock::Mode::exclusive, ex) != KeyLock::Result::ok) continue;  // in use
            if (stale_record(pth) && ::unlink(pth.c_str()) == 0) ++rep.failed;
        }
    }

    // all: hints for this library's artifacts. A hint names the artifact it
    // was recorded for (its realpath); hints for prebuilt sidecars, another
    // cache, unknown template ids or without that field are left alone, and a
    // hint whose key is leased (a loaded model) is skipped.
    void hints() {
        if (p.mode != PrunePolicy::Mode::all) return;
        const std::string artifacts_real = real_path(l.artifacts);
        for (const std::string & n : list_dir(l.hints)) {
            if (n.size() < 6 || n.compare(n.size() - 5, 5, ".json") != 0 || n[0] == '.') continue;
            const std::string pth = l.hints + "/" + n;
            const std::optional<detail::HintRecord> h = detail::read_hint(pth);
            if (!h || h->artifact.empty()) continue;
            fs::path a(h->artifact);
            if (a.filename() == "model.mlmodelc") a = a.parent_path();
            std::string tid, key;
            if (a.parent_path().string() != artifacts_real || !parse_artifact_name(a.filename().string(), tid, key) ||
                !c.owns(tid)) {
                continue;
            }
            KeyLock ex;
            if (KeyLock::take(l, key, KeyLock::Mode::exclusive, ex) != KeyLock::Result::ok) continue;
            if (::unlink(pth.c_str()) == 0) ++rep.hints;
        }
    }

    void quarantine_dirs() {
        if (p.mode == PrunePolicy::Mode::scoped) return;
        for (const std::string & n : list_dir(l.quarantine)) {
            std::string tid;
            if (!parse_quarantine_name(n, tid) || !c.owns(tid)) continue;
            remove_tree(l.quarantine + "/" + n);
            ++rep.quarantine;
        }
    }
};

}  // namespace

PruneReport prune(const std::string & cache_dir, const PrunePolicy & policy, const Catalog & catalog) {
    Layout l(cache_dir);
    Pruner pr{l, policy, catalog, policy.now_s ? *policy.now_s : detail::unix_now_s(), {}, {}};
    if (!exists(l.root)) return pr.rep;
    if (policy.mode == PrunePolicy::Mode::scoped) pr.roots = scope_roots(policy.dead_paths);
    std::set<std::string> seen;
    pr.artifacts(seen);
    pr.orphan_memos(seen);
    pr.staging();
    pr.tmp();
    pr.failed_records();
    pr.hints();
    pr.quarantine_dirs();
    return pr.rep;
}

}  // namespace cache
}  // namespace QVAC_COREML_NS
