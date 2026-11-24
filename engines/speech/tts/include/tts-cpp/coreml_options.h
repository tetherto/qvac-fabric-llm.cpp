#pragma once

// Core ML options for tts-cpp engines that load sidecars through the shared
// Core ML runtime (today: the Parler DAC decoder, and the Chatterbox S3Gen
// encoder, CFM estimator and HiFT decoder).
//
// Every field is unset by default. An engine resolves them once, at
// construction (or once per call for direct pipeline callers), in this order
// per field: the explicit field, then the model family's environment variable
// (<FAMILY>_COREML_STRICT -> use = require; <FAMILY>_COREML_COMPUTE_UNITS,
// e.g. PARLER_ or CHATTERBOX_), then QVAC_COREML_CACHE_DIR (cache directory
// only), then the default. <FAMILY>_COREML_DISABLE forces use = off and
// overrides everything.
//
// Defaults: use automatic, generate background, wait none on macOS and call on
// iOS, prewarm predicted, no cache directory (no generation and no warm
// hints), the stage's own compute units, strict waits unbounded, no memory
// budget.
//
// Builds without TTS_CPP_COREML accept these options; an explicit invalid
// compute_units still throws there, and everything else is ignored.

#include <cstdint>
#include <optional>
#include <string>

namespace tts_cpp {

struct CoremlOptions {
    enum class Use      { unset, automatic, off, require };
    enum class Generate { unset, never, background };
    enum class Wait     { unset, none, construct, call };
    enum class Prewarm  { unset, none, minimal, predicted, all };

    // automatic: a stage runs on its Core ML sidecar once the sidecar is
    // loaded, on ggml until then. off: ggml only. require: like wait = call,
    // but a sidecar that cannot serve a call (failed to generate or load,
    // failed a prediction, an input wider than its widest function, no usable
    // function) makes the call fail instead of falling back to ggml. An
    // explicit value wins over an engine's legacy aliases (Chatterbox's
    // coreml_strict / coreml_disabled).
    Use use = Use::unset;
    // background: with a cache directory and no prebuilt sidecar next to the
    // GGUF, build the sidecar from the GGUF in the background (on a supported
    // OS) and run the stage on ggml until it is ready (Parler). never: no
    // generation; prebuilt sidecars still load. Chatterbox loads only the
    // prebuilt sidecars next to its S3Gen GGUF today, so this has no effect
    // there yet.
    Generate generate = Generate::unset;
    // none: never block, apart from a short bounded wait for a load this host
    //       finished quickly before (a warm hint), and only while it fits the
    //       time the stage would take on ggml.
    // construct: the constructor waits until its construction group (its
    //       sidecars) finished: ready or failed.
    // call: each call waits until the sidecar functions it needs are ready or
    //       have failed, unless an already loaded function covers it
    //       (Chatterbox's wider widths).
    Wait wait = Wait::unset;
    // Which sidecars are requested at construction: none (only on first use);
    // minimal (each engine's minimal set: Parler the DAC; Chatterbox HiFT plus
    // the encoder and CFM widths this host loaded most recently); predicted
    // (minimal at construction, then each call requests what it needs:
    // Chatterbox's widths for its sentences); or all (every function). See
    // the engine's documentation.
    Prewarm prewarm = Prewarm::unset;
    // Where generated sidecars (Parler) and warm hints are kept. Unset ->
    // QVAC_COREML_CACHE_DIR -> "" (no generation, no hints, so no bounded
    // wait).
    std::optional<std::string> cache_dir;
    // all | cpu_and_ane | cpu_and_gpu | cpu_only
    std::optional<std::string> compute_units;
    // Longest wait of a strict (use = require) acquisition; unset -> unbounded.
    std::optional<int64_t> strict_wait_ms;
    // Bytes fit was assessed against; unset -> no memory-degrade.
    std::optional<int64_t> memory_budget_bytes;
};

}  // namespace tts_cpp
