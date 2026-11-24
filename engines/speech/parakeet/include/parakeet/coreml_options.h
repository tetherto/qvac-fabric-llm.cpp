#pragma once

// Core ML options for the Parakeet engine.
//
// Every field is unset by default. The engine resolves them once, at
// construction, in this order per field: the explicit field, then the
// per-engine environment variable (PARAKEET_COREML_STRICT -> use = require;
// PARAKEET_COREML_COMPUTE_UNITS), then QVAC_COREML_CACHE_DIR (cache
// directory only), then the default. PARAKEET_COREML_DISABLE forces
// use = off and overrides everything.
//
// Defaults: use automatic, generate background, wait none on macOS and call on
// iOS, prewarm predicted, no cache directory (no generation, no warm hints),
// the stage's own compute units, strict waits unbounded, no memory budget.
//
// Builds without PARAKEET_COREML accept and ignore these options.

#include <cstdint>
#include <optional>
#include <string>

namespace parakeet {

struct CoremlOptions {
    enum class Use      { unset, automatic, off, require };
    enum class Generate { unset, never, background };
    enum class Wait     { unset, none, construct, call };
    enum class Prewarm  { unset, none, minimal, predicted, all };

    // automatic: Core ML sidecars run when they are ready, ggml until then.
    // off: ggml only. require: like wait = call, but a sidecar that cannot
    // serve a call throws (an exception derived from std::runtime_error)
    // instead of falling back to ggml: a sidecar that failed to load or timed
    // out, an input longer than a fixed-capacity encoder (for example with
    // long-form windowing disabled, or a standard TDT input past the
    // sidecar's capacity, which is never split into Core ML windows), a
    // shape the sidecar rejects, or a failed prediction (which also retires
    // the sidecar). Engine::stream_start refuses a push stream whose windows
    // are longer than a fixed-capacity encoder (every offline TDT push
    // stream with a 15 s standard sidecar). A diarization pass throws when
    // every diarizer capacity that covers it failed, and when it is longer
    // than every capacity. Calls the engine keeps on ggml by design
    // (per-stage capture harnesses, model families or shapes with no
    // sidecar) still run on ggml.
    Use use = Use::unset;
    Generate generate = Generate::unset;
    // none: never block (apart from the bounded wait on a warm hint).
    // construct: the constructor waits until the construction group finished.
    // call: each call waits until its sidecar is ready or has failed.
    Wait wait = Wait::unset;
    // Which sidecars are requested at construction: none, the minimal set
    // (encoder; the smallest diarizer capacity), predicted (= minimal; each
    // call also requests what it needs), or all (every diarizer capacity).
    Prewarm prewarm = Prewarm::unset;
    std::optional<std::string> cache_dir;       // unset -> QVAC_COREML_CACHE_DIR -> "" (no generation)
    std::optional<std::string> compute_units;   // all | cpu_and_ane | cpu_and_gpu | cpu_only
    std::optional<int64_t>     strict_wait_ms;  // unset -> unbounded
    std::optional<int64_t>     memory_budget_bytes; // bytes fit was assessed against; unset -> no memory-degrade
};

}  // namespace parakeet
