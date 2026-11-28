#include "hints.h"

#include "sha256.h"

#include "nlohmann/json.hpp"

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <system_error>

#include <limits.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

namespace QVAC_COREML_NS {
namespace detail {

namespace fs = std::filesystem;

namespace {

int64_t stat_mtime_ns(const struct stat & st) {
#if defined(__APPLE__)
    return (int64_t) st.st_mtimespec.tv_sec * 1000000000LL + st.st_mtimespec.tv_nsec;
#else
    return (int64_t) st.st_mtim.tv_sec * 1000000000LL + st.st_mtim.tv_nsec;
#endif
}

std::string real_path(const std::string & p) {
    char buf[PATH_MAX];
    if (::realpath(p.c_str(), buf) != nullptr) return buf;
    return p;
}

}  // namespace

std::string hint_id(std::string_view host_id, std::string_view artifact_realpath, int64_t model_mil_mtime_ns,
                    std::string_view function, std::string_view units, std::string_view os_build) {
    Sha256 h;
    static const char tag[] = "qvac-coreml-hint";
    h.update(tag, sizeof(tag));  // includes the terminating NUL ("...\0")
    h.str(host_id);
    h.str(artifact_realpath);
    h.i64le(model_mil_mtime_ns);
    h.str(function);
    h.str(units);
    h.str(os_build);
    return to_hex(h.finish()).substr(0, 32);
}

std::string hint_path_for(const std::string & cache_dir, const std::string & host_id, const std::string & artifact,
                          const std::string & function, const std::string & units, const std::string & os_build) {
    if (cache_dir.empty() || artifact.empty()) return {};
    const std::string rp = real_path(artifact);
    struct stat st{};
    int64_t mil_mtime = 0;
    if (::stat((rp + "/model.mil").c_str(), &st) == 0) mil_mtime = stat_mtime_ns(st);
    const std::string id = hint_id(host_id, rp, mil_mtime, function, units, os_build);
    return (fs::path(cache_dir) / "qvac-coreml-v1" / "hints" / (id + ".json")).string();
}

int64_t unix_now_s() {
    using namespace std::chrono;
    return duration_cast<seconds>(system_clock::now().time_since_epoch()).count();
}

std::string iso8601_utc(int64_t unix_s) {
    const std::time_t t = static_cast<std::time_t>(unix_s);
    std::tm tm{};
    gmtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

std::optional<int64_t> parse_iso8601_utc(std::string_view s) {
    std::tm tm{};
    int y = 0, mo = 0, d = 0, h = 0, mi = 0, sec = 0;
    const std::string str(s);
    if (std::sscanf(str.c_str(), "%d-%d-%dT%d:%d:%d", &y, &mo, &d, &h, &mi, &sec) != 6) return std::nullopt;
    tm.tm_year = y - 1900;
    tm.tm_mon = mo - 1;
    tm.tm_mday = d;
    tm.tm_hour = h;
    tm.tm_min = mi;
    tm.tm_sec = sec;
    return static_cast<int64_t>(timegm(&tm));
}

std::optional<HintRecord> read_hint(const std::string & path) {
    if (path.empty()) return std::nullopt;
    std::ifstream in(path, std::ios::binary);
    if (!in) return std::nullopt;
    std::stringstream ss;
    ss << in.rdbuf();
    const nlohmann::json j = nlohmann::json::parse(ss.str(), nullptr, /*allow_exceptions=*/false);
    if (!j.is_object()) return std::nullopt;
    if (auto s = j.find("schema"); s == j.end() || !s->is_number_integer() || s->get<int64_t>() != 1) {
        return std::nullopt;
    }
    HintRecord r;
    auto num = [&](const char * k) -> std::optional<int64_t> {
        auto it = j.find(k);
        if (it == j.end() || !it->is_number()) return std::nullopt;
        return it->get<int64_t>();
    };
    r.warm_load_ms = num("warm_load_ms");
    r.cold_load_ms = num("cold_load_ms");
    if (auto it = j.find("last_ok"); it != j.end() && it->is_string()) r.last_ok = it->get<std::string>();
    if (auto it = j.find("stale_until"); it != j.end() && it->is_string()) {
        r.stale_until_s = parse_iso8601_utc(it->get<std::string>());
    }
    if (auto it = j.find("artifact"); it != j.end() && it->is_string()) r.artifact = it->get<std::string>();
    struct stat st{};
    if (::stat(path.c_str(), &st) == 0) r.mtime_ns = stat_mtime_ns(st);
    return r;
}

bool write_hint(const std::string & path, const HintRecord & rec) {
    if (path.empty()) return false;
    std::error_code ec;
    const fs::path dir = fs::path(path).parent_path();
    fs::create_directories(dir, ec);
    nlohmann::json j;
    j["schema"] = 1;
    j["warm_load_ms"] = rec.warm_load_ms ? nlohmann::json(*rec.warm_load_ms) : nlohmann::json(nullptr);
    j["cold_load_ms"] = rec.cold_load_ms ? nlohmann::json(*rec.cold_load_ms) : nlohmann::json(nullptr);
    j["last_ok"] = rec.last_ok;
    j["stale_until"] = rec.stale_until_s ? nlohmann::json(iso8601_utc(*rec.stale_until_s)) : nlohmann::json(nullptr);
    if (!rec.artifact.empty()) j["artifact"] = real_path(rec.artifact);
    const std::string body = j.dump();

    std::random_device rd;
    const std::string tmp =
        (dir / (".tmp." + std::to_string(::getpid()) + "." + std::to_string(rd()))).string();
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) return false;
        out << body;
        out.flush();
        if (!out) {
            out.close();
            fs::remove(tmp, ec);
            return false;
        }
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        fs::remove(tmp, ec);
        return false;
    }
    return true;
}

}  // namespace detail
}  // namespace QVAC_COREML_NS
