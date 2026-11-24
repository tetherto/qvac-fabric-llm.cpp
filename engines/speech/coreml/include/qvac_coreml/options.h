#pragma once

// Core ML runtime options and their one-time resolution.
//
// Internal header (BUILD_INTERFACE only, never installed). Every engine library
// compiles its own copy of the runtime inside its own namespace, chosen by
// qvac_coreml_add_to(... NAMESPACE <ns>), so QVAC_COREML_NS is always defined
// for code that includes this header. Engines expose their own public
// CoremlOptions and convert it field by field to Options below.

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#ifndef QVAC_COREML_NS
#error "QVAC_COREML_NS must name the engine's runtime namespace (qvac_coreml_add_to sets it)"
#endif

namespace QVAC_COREML_NS {

enum class Use { unset, automatic, off, require };
enum class Generate { unset, never, background };
enum class Wait { unset, none, construct, call };
enum class Prewarm { unset, none, minimal, predicted, all };

// Resolved compute-unit vocabulary. Distinct from the result labels
// (coreml-all | coreml-ane | coreml-gpu | coreml-cpu | ggml | mixed).
enum class ComputeUnits { all, cpu_and_ane, cpu_and_gpu, cpu_only };

const char * to_string(Use v);
const char * to_string(Generate v);
const char * to_string(Wait v);
const char * to_string(Prewarm v);
const char * to_string(ComputeUnits v);

// Accepts exactly the enum spellings above, as every engine does today.
std::optional<ComputeUnits> parse_compute_units(std::string_view text);

// all and cpu_and_ane may compile an ANE plan (subject to memory admission).
bool units_include_ane(ComputeUnits units);

// Internal mirror of the public per-engine CoremlOptions. Every field is
// unset by default.
struct Options {
    Use use = Use::unset;
    Generate generate = Generate::unset;
    Wait wait = Wait::unset;
    Prewarm prewarm = Prewarm::unset;
    std::optional<std::string> cache_dir;           // unset -> QVAC_COREML_CACHE_DIR -> ""
    std::optional<std::string> compute_units;       // all | cpu_and_ane | cpu_and_gpu | cpu_only
    std::optional<int64_t>     strict_wait_ms;      // unset -> unbounded
    std::optional<int64_t>     memory_budget_bytes; // unset -> no budget, no memory-degrade
    // Development override: generate from <template_dir>/<template-id> instead
    // of the embedded template. Honoured only by builds with
    // QVAC_COREML_DEV (an engine's *_BUILD_TESTS=ON or QVAC_COREML_DEV=ON);
    // elsewhere it is ignored with a warning.
    std::optional<std::string> template_dir;        // unset -> QVAC_COREML_TEMPLATE_DIR (dev builds) -> ""
};

// Whether this library honours Options::template_dir and
// QVAC_COREML_TEMPLATE_DIR.
bool template_dir_override_compiled();

enum class Platform { macos, ios, other };

struct OsVersion {
    int major = 0;
    int minor = 0;
    int patch = 0;
};

bool os_at_least(const OsVersion & v, int major, int minor);

// Which input a resolved field came from (diagnostics and tests).
enum class Source { explicit_field, engine_env, shared_env, default_value };

const char * to_string(Source v);

struct Resolved {
    Use use = Use::automatic;  // automatic | off | require; never unset
    Generate generate = Generate::background;
    Wait wait = Wait::none;
    Prewarm prewarm = Prewarm::predicted;
    std::string cache_dir;                      // "" = no cache: no generation and no hints
    std::optional<ComputeUnits> compute_units;  // nullopt -> the stage's own default
    std::optional<int64_t> strict_wait_ms;      // nullopt -> unbounded
    std::optional<int64_t> memory_budget_bytes; // nullopt -> no budget, no memory-degrade
    std::string template_dir;                   // "" = the embedded templates (always "" without QVAC_COREML_DEV)

    std::string disabled_reason;        // why use == off: "<P>_COREML_DISABLE", "use=off" or "memory"
    std::vector<std::string> warnings;  // inherited values that were ignored

    Source use_source = Source::default_value;
    Source units_source = Source::default_value;
    Source cache_dir_source = Source::default_value;

    bool enabled() const { return use != Use::off; }
    bool strict() const { return use == Use::require; }
    // Strict behaves like `call` for acquisitions.
    Wait acquire_wait() const { return strict() ? Wait::call : wait; }
    ComputeUnits units_or(ComputeUnits stage_default) const {
        return compute_units ? *compute_units : stage_default;
    }
};

using GetenvFn = const char * (*)(const char * name);

// Resolves the options once. `env_prefix` names the model family the
// existing environment variables use (e.g. "CHATTERBOX", "PARLER", "PARAKEET").
//
//   <P>_COREML_DISABLE (presence, as today) -> use = off, overriding everything.
//   Otherwise per field: the explicit field, then the per-engine variable
//   (<P>_COREML_STRICT presence -> use = require; <P>_COREML_COMPUTE_UNITS),
//   then QVAC_COREML_CACHE_DIR (cache directory only), then the default.
//
// Defaults: use automatic, generate background, wait none on macOS and call on
// iOS, prewarm predicted, cache_dir "", compute units per stage, strict wait
// unbounded. An explicit invalid compute_units, a negative strict_wait_ms or a
// negative memory budget throws std::invalid_argument; an invalid inherited
// compute-units value is ignored and reported in `warnings`, as today. The
// environment is read only here: resolved values reach loaders as arguments.
Resolved resolve_coreml(const Options & opts, std::string_view env_prefix, GetenvFn getenv_fn = nullptr,
                        std::optional<Platform> platform = std::nullopt);

// ---- Fit interface (route and memory-degrade only) ------------------------

// The engine's two verdicts. coreml_* is the ggml estimate (double residency
// kept) plus the stage's Core ML envelope.
struct FitVerdict {
    bool ggml_only_fits = true;
    bool coreml_fits = true;
    uint64_t coreml_bytes = 0;
    bool validated = false;
};

// Memory-degrade: only with a budget. When the Core ML-inclusive
// verdict fails and the ggml-only verdict passes, the instance resolves
// use = off (disabled_reason "memory"), so Core ML never turns a working ggml
// verdict into a refusal. Returns true when it degraded.
bool apply_memory_degrade(Resolved & resolved, const FitVerdict & verdict);

// {fits, coreml: {fits, bytes, validated, disabled_reason}, ggml_only: {fits}}
struct FitReport {
    bool fits = true;
    FitVerdict verdict;
    std::string disabled_reason;
};

FitReport make_fit_report(const Resolved & resolved, const FitVerdict & verdict);
std::string to_json(const FitReport & report);

}  // namespace QVAC_COREML_NS
