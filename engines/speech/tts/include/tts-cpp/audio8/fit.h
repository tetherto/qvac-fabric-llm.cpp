#pragma once

// Audio8 memory-fit preflight: project whether the Audio8 GGUF set (LM +
// codec decoder, optionally the codec encoder for cloning) fits the device
// memory available right now, without reading any weight data.
//
// The projection mirrors one Engine load plus one synthesize() BY
// CONSTRUCTION: it drives the same loaders (metadata-only), the same graph
// builders, and the same allocation policies (direct gallocr vs the
// [backend, CPU] scheduler fallback, and the codec's memory-budgeted
// synthesis block plan) through ggml's size-only APIs. What is priced:
//
//   weights  -- LM + decoder (+ encoder) weight buffers
//   state    -- the LM's slow/fast KV slabs, allocated at full context
//               capacity exactly as load_lm allocates them
//   lm       -- the resident LM graph arenas: the worst of the prompt prefill
//               and the deepest decode step (they share one allocator, which
//               grows to the larger), plus the fast-head step and chained
//               whole-frame graphs
//   codec    -- the full-sequence latent graph at the projected frame count,
//               plus the synthesis block the runtime's own block planner
//               would pick against the device memory available right now
//               (and, when cloning, the encoder's convolution block and
//               full-reference analysis graphs)
//
// See tts-cpp/fit.h for the status contract and result layout.

#include "tts-cpp/export.h"
#include "tts-cpp/fit.h"

#include <cstdint>
#include <string>

namespace tts_cpp {
namespace audio8 {

struct FitOptions {
    // Same roles as EngineOptions: LM and decoder are required; the encoder
    // adds the cloning path to the projection.
    std::string lm_gguf_path;
    std::string codec_decoder_gguf_path;
    std::string codec_encoder_gguf_path;

    // Same semantics as EngineOptions::n_gpu_layers: any positive value
    // requests the validated-GPU backend (with the same runtime fallbacks a
    // real load applies); 0 projects for the CPU backend.
    int n_gpu_layers = 0;

    // Same semantics as EngineOptions::backend: "auto" / "" selects the
    // tier-based GPU/CPU policy honouring n_gpu_layers, "cpu" forces CPU,
    // and a specific name ("hexagon", an exact ggml device name, etc.)
    // projects against that device. A projection made with the wrong
    // backend selector can misreport whether the model will fit at
    // synthesis time, so the fit path mirrors the engine selector.
    std::string backend = "auto";

    // Same semantics as EngineOptions::backends_dir.
    std::string backends_dir;

    // ── Workload ──────────────────────────────────────────────────────────
    // ChatML prompt width in tokens (text + template overhead), excluding any
    // cloning reference frames, which are added from reference_seconds.
    int prompt_tokens = 128;

    // Same semantics as EngineOptions::max_frames (0 = the engine's default
    // of 512 frames, ~24 s). The projection prices generation to the full
    // budget: the LM context depth and the codec sequence both follow it.
    int max_frames = 0;

    // Cloning reference length; used only when codec_encoder_gguf_path is
    // set. The reference contributes prompt frames AND the encoder graphs.
    float reference_seconds = 10.0f;

    // Free-memory headroom that must remain on the device for the projection
    // to count as fitting.
    uint64_t margin_bytes = 256ull * 1024 * 1024;
};

// Project the model set + workload in `opts` against the device memory
// available right now. Reads only GGUF metadata, never weight data; builds
// and measures graphs but never allocates or executes them. Never throws for
// a "does not fit" outcome (that is a valid Failure result) or an unreadable
// model (Error); see FitStatus in tts-cpp/fit.h.
TTS_CPP_API FitResult fit_params(const FitOptions & opts);

}  // namespace audio8
}  // namespace tts_cpp

// CLI front-end over audio8::fit_params (the audio8-fit-params tool); lives
// in the library so hosts can link it directly. Exit code == fit status:
// 0 fits, 1 does not fit, 2 error.
extern "C" TTS_CPP_API int audio8_fit_cli_main(int argc, char ** argv);
