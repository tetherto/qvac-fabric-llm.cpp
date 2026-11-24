#include "failed.h"

#include "hints.h"
#include "sha256.h"

#include "nlohmann/json.hpp"

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>

#include <unistd.h>

namespace QVAC_COREML_NS {
namespace failed {

namespace fs = std::filesystem;

const char * to_string(Stage s) {
    switch (s) {
        case Stage::prepare: return "prepare";
        case Stage::compile: return "compile";
        case Stage::load: return "load";
    }
    return "?";
}

std::string record_path(const std::string & failed_dir, const std::string & key32, const std::string & function,
                        const std::string & units) {
    if (function.empty()) return failed_dir + "/" + key32 + ".json";
    detail::Sha256 h;
    h.str(function);
    h.str(units);
    return failed_dir + "/" + key32 + "." + detail::to_hex(h.finish()).substr(0, 16) + ".json";
}

std::optional<Record> read(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return std::nullopt;
    std::stringstream ss;
    ss << in.rdbuf();
    const nlohmann::json j = nlohmann::json::parse(ss.str(), nullptr, false);
    if (!j.is_object()) return std::nullopt;
    // A wrong-typed field (another writer's file in a shared cache) makes the
    // record unreadable, which callers treat as no record.
    try {
        if (j.value("schema", 0) != 1) return std::nullopt;
        Record r;
        const std::string cls = j.value("class", std::string());
        if (cls == "deterministic") {
            r.cls = ErrorClass::deterministic;
        } else if (cls == "transient") {
            r.cls = ErrorClass::transient;
        } else {
            return std::nullopt;
        }
        r.attempts = j.value("attempts", 1);
        r.domain = j.value("domain", std::string());
        r.code = j.value("code", int64_t{0});
        r.os_build = j.value("os_build", std::string());
        if (j.contains("function") && j["function"].is_string()) r.function = j["function"];
        if (j.contains("units") && j["units"].is_string()) r.units = j["units"];
        r.last_at = detail::parse_iso8601_utc(j.value("last_at", std::string())).value_or(0);
        r.template_id = j.value("template", std::string());
        r.process = j.value("process", std::string());
        r.message = j.value("message", std::string());
        return r;
    } catch (const nlohmann::json::exception &) {
        return std::nullopt;
    }
}

bool write(const std::string & path, const Record & r) {
    nlohmann::json j;
    j["schema"] = 1;
    j["class"] = to_string(r.cls);
    j["attempts"] = r.attempts;
    j["domain"] = r.domain;
    j["code"] = r.code;
    j["os_build"] = r.os_build;
    j["function"] = r.function.empty() ? nlohmann::json(nullptr) : nlohmann::json(r.function);
    j["units"] = r.units.empty() ? nlohmann::json(nullptr) : nlohmann::json(r.units);
    j["last_at"] = detail::iso8601_utc(r.last_at);
    j["template"] = r.template_id;
    j["process"] = r.process;
    j["message"] = r.message;
    std::error_code ec;
    const fs::path dir = fs::path(path).parent_path();
    fs::create_directories(dir, ec);
    std::random_device rd;
    const std::string tmp = (dir / (".tmp." + std::to_string(::getpid()) + "." + std::to_string(rd()))).string();
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) return false;
        out << j.dump();
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

namespace {

// A process's token. Published once and then never changed or freed, so the
// reference process_token() returns stays valid. A child after fork still
// sees its parent's token (another pid) and publishes its own; `prev` keeps
// the parent's reachable.
struct ProcessToken {
    pid_t pid;
    std::string text;
    const ProcessToken * prev;
};

std::atomic<const ProcessToken *> g_process_token{nullptr};

}  // namespace

const std::string & process_token() {
    const pid_t pid = ::getpid();
    const ProcessToken * cur = g_process_token.load(std::memory_order_acquire);
    while (cur == nullptr || cur->pid != pid) {
        // First use in this process. Threads that race here each build a
        // token; the first to publish it wins, and every thread returns that one.
        std::random_device rd;
        char buf[48];
        std::snprintf(buf, sizeof(buf), "%ld-%08x%08x", static_cast<long>(pid), rd(), rd());
        const ProcessToken * mine = new ProcessToken{pid, buf, cur};
        if (g_process_token.compare_exchange_strong(cur, mine, std::memory_order_acq_rel,
                                                    std::memory_order_acquire)) {
            return mine->text;
        }
        delete mine;  // `cur` now holds the token another thread published
    }
    return cur->text;
}

bool is_resource(const std::string & domain, int64_t code) {
    if (domain == ERROR_DOMAIN) return code == NO_SPACE || code == OUT_OF_MEMORY || code == TIMEOUT || code == INTERRUPTED;
    if (domain == "NSPOSIXErrorDomain") {
        return code == ENOMEM || code == ENOSPC || code == EDQUOT || code == ETIMEDOUT || code == EINTR ||
               code == EAGAIN || code == ECANCELED;
    }
    if (domain == "NSCocoaErrorDomain") {
        // NSFileWriteOutOfSpaceError, NSUserCancelledError, NSXPCConnectionInterrupted /
        // Invalid / ReplyInvalid.
        return code == 640 || code == 3072 || code == 4097 || code == 4099 || code == 4101;
    }
    if (domain == "com.apple.CoreML") return code == 11;  // MLModelErrorPredictionCancelled
    return false;
}

bool expired(const Record & r, int64_t now_s) {
    return r.cls == ErrorClass::transient && now_s - r.last_at > TRANSIENT_EXPIRY_S;
}

int64_t backoff_s(int attempts) {
    int64_t s = 30;
    for (int i = 1; i < attempts && s < 300; ++i) s *= 2;
    return s < 300 ? s : 300;
}

Record record(const std::string & failed_dir, const std::string & key32, Stage stage, const Error & err,
              const std::string & template_id, const std::string & os_build, int64_t now_s,
              const std::string & function, const std::string & units) {
    const std::string path = record_path(failed_dir, key32, function, units);
    const std::optional<Record> prev = read(path);
    Record r;
    r.domain = err.domain.empty() ? ERROR_DOMAIN : err.domain;
    r.code = err.code;
    r.os_build = os_build;
    r.function = function;
    r.units = units;
    r.last_at = now_s;
    r.template_id = template_id;
    r.process = process_token();
    r.message = err.message;
    const bool same = prev && !expired(*prev, now_s) && prev->os_build == os_build && prev->domain == r.domain &&
                      prev->code == r.code;
    if (stage == Stage::prepare) {
        r.cls = err.cls == ErrorClass::deterministic ? ErrorClass::deterministic : ErrorClass::transient;
        r.attempts = same && prev->cls == r.cls ? prev->attempts + 1 : 1;
    } else {
        // Compile and load failures start transient; a repeat in another
        // process on the same os_build promotes, unless it is a resource code.
        r.cls = ErrorClass::transient;
        r.attempts = same ? prev->attempts + 1 : 1;
        if (same && (prev->cls == ErrorClass::deterministic ||
                     (prev->process != r.process && !is_resource(r.domain, r.code)))) {
            r.cls = ErrorClass::deterministic;
        }
    }
    write(path, r);
    return r;
}

void clear(const std::string & failed_dir, const std::string & key32, const std::string & function,
           const std::string & units) {
    std::error_code ec;
    fs::remove(record_path(failed_dir, key32, function, units), ec);
}

Verdict check(const std::string & failed_dir, const std::string & key32, const std::string & os_build, int64_t now_s,
              const std::string & function, const std::string & units) {
    Verdict v;
    const std::optional<Record> r = read(record_path(failed_dir, key32, function, units));
    if (!r || expired(*r, now_s)) return v;
    v.rec = *r;
    const std::string what = "earlier " + std::string(to_string(r->cls)) + " failure (" + r->domain + " " +
                             std::to_string(r->code) + ", " + std::to_string(r->attempts) +
                             " attempt(s)): " + r->message;
    if (r->cls == ErrorClass::deterministic) {
        if (r->os_build != os_build) return v;  // a new OS build may compile it
        v.blocked = true;
        v.error = Error::deterministic(what, r->domain, r->code);
        return v;
    }
    if (r->process == process_token()) {
        v.retry_at = r->last_at + backoff_s(r->attempts);
        if (now_s < v.retry_at) {
            v.blocked = true;
            v.error = Error::transient(what, r->domain, r->code);
        }
    }
    return v;
}

}  // namespace failed
}  // namespace QVAC_COREML_NS
