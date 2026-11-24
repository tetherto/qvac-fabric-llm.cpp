#pragma once

// Observed-warm hint records:
//   <cache_dir>/qvac-coreml-v1/hints/<hint_id>.json
//   {"schema": 1, "warm_load_ms": n|null, "cold_load_ms": n|null,
//    "last_ok": "<iso8601>", "stale_until": "<iso8601>"|null,
//    "artifact": "<realpath>"}
// "artifact" is the realpath hashed into hint_id; it lets the cache's prune
// attribute a hint to a cache artifact and its template id.
// A hint records that this host loaded this artifact before; it is not proof
// that the ANE plan is still cached.

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#ifndef QVAC_COREML_NS
#error "QVAC_COREML_NS must be defined"
#endif

namespace QVAC_COREML_NS {
namespace detail {

struct HintRecord {
    std::optional<int64_t> warm_load_ms;
    std::optional<int64_t> cold_load_ms;
    std::string last_ok;                    // ISO 8601 UTC
    std::optional<int64_t> stale_until_s;   // unix seconds; nullopt = not stale
    int64_t mtime_ns = 0;                   // file mtime: the most recent record wins
    std::string artifact;                   // the loaded artifact; written as its realpath
};

// hint_id = hex(H("qvac-coreml-hint\0" || str(host_id) || str(artifact_realpath) ||
//                 i64le(model_mil_mtime_ns) || str(function) || str(units) || str(os_build)))[0:32]
std::string hint_id(std::string_view host_id, std::string_view artifact_realpath, int64_t model_mil_mtime_ns,
                    std::string_view function, std::string_view units, std::string_view os_build);

// Resolves realpath(artifact) and the mtime of <artifact>/model.mil (0 when
// absent), then returns the record path; "" without a cache_dir.
std::string hint_path_for(const std::string & cache_dir, const std::string & host_id, const std::string & artifact,
                          const std::string & function, const std::string & units, const std::string & os_build);

std::optional<HintRecord> read_hint(const std::string & path);
// Atomic: temp file in the same directory, then rename.
bool write_hint(const std::string & path, const HintRecord & rec);

int64_t unix_now_s();
std::string iso8601_utc(int64_t unix_s);
std::optional<int64_t> parse_iso8601_utc(std::string_view s);

inline bool hint_stale(const HintRecord & r, int64_t now_s) {
    return r.stale_until_s.has_value() && now_s < *r.stale_until_s;
}

}  // namespace detail
}  // namespace QVAC_COREML_NS
