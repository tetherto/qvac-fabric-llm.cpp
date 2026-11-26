#pragma once

// Failure records:
//   failed/<key32>.json                    preparation and compile failures
//   failed/<key32>.<fn_units_hash>.json    load failures of one (function, units)
//
// Deterministic records are never retried automatically until the generator
// version (part of the key) or os_build changes, and never expire. Transient
// records are retried by the next process, with an in-process backoff from
// 30 s to 5 min, and expire after 7 days. Compile and load failures start
// transient; the same (domain, code) on the same os_build again, from another
// process, promotes the record to deterministic, unless the code is on the
// resource list (memory, disk, timeout, interruption, XPC), which never
// promotes.
//
// A record also carries "template" (the template id, so prune can tell
// which library owns it) and "process" (who wrote it, so promotion can tell
// "another process" from a retry in this one).

#include "qvac_coreml/slot.h"
#include "qvac_coreml/status.h"

#include <cstdint>
#include <optional>
#include <string>

#ifndef QVAC_COREML_NS
#error "QVAC_COREML_NS must be defined"
#endif

namespace QVAC_COREML_NS {
namespace failed {

// The generator's own error domain and codes.
constexpr const char * ERROR_DOMAIN = "qvac.coreml";
enum Code : int64_t {
    SOURCE_CHANGED = 1,    // transient: the GGUF changed under the job (open or write-time re-check)
    NO_SPACE = 2,          // transient: statfs below 2 x weight_bin_size
    CACHE_FS = 3,          // unsupported: not a local apfs/hfs volume
    OUT_OF_MEMORY = 4,     // resource
    TIMEOUT = 5,           // resource
    INTERRUPTED = 6,       // resource
    INVALID_ARTIFACT = 7,  // a published artifact failed its inventory check
};

enum class Stage { prepare, compile, load };

const char * to_string(Stage s);

struct Record {
    ErrorClass cls = ErrorClass::transient;
    int attempts = 1;
    std::string domain;
    int64_t code = 0;
    std::string os_build;
    std::string function;  // load records
    std::string units;     // load records
    int64_t last_at = 0;   // unix seconds (ISO 8601 on disk)
    std::string template_id;
    std::string process;
    std::string message;
};

constexpr int64_t TRANSIENT_EXPIRY_S = 7 * 24 * 3600;

// failed/<key32>.json, or failed/<key32>.<fn_units_hash>.json when `function`
// is set (load failures), fn_units_hash = hex(H(str(function) || str(units)))[0:16].
std::string record_path(const std::string & failed_dir, const std::string & key32, const std::string & function = {},
                        const std::string & units = {});

std::optional<Record> read(const std::string & path);
bool write(const std::string & path, const Record & r);  // atomic (temp + rename)

// This process's token, the same for every thread; safe to call from several
// threads at once, including the first call. A child after fork gets a new one.
const std::string & process_token();

// Codes that never promote (the resource list lives here).
bool is_resource(const std::string & domain, int64_t code);

bool expired(const Record & r, int64_t now_s);
// In-process backoff after `attempts` failures: 30 s doubling, at most 5 min.
int64_t backoff_s(int attempts);

// Records a failure of `stage` and returns the record as written.
Record record(const std::string & failed_dir, const std::string & key32, Stage stage, const Error & err,
              const std::string & template_id, const std::string & os_build, int64_t now_s,
              const std::string & function = {}, const std::string & units = {});

// Removes the record for (key, function, units) after a success: promotion
// counts only consecutive failures.
// prepare() calls it after generation succeeds or finds the artifact published;
// the load lane calls it with (function, units) after a successful load.
void clear(const std::string & failed_dir, const std::string & key32, const std::string & function = {},
           const std::string & units = {});

struct Verdict {
    bool blocked = false;
    Record rec;
    int64_t retry_at = 0;  // transient: when this process may retry
    Error error;           // what a blocked job reports (the record's class)
};

// Whether a job for (key, function, units) may run now on `os_build`.
Verdict check(const std::string & failed_dir, const std::string & key32, const std::string & os_build, int64_t now_s,
              const std::string & function = {}, const std::string & units = {});

}  // namespace failed
}  // namespace QVAC_COREML_NS
