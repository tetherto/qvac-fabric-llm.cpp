#pragma once

// Supertonic memory-fit preflight: project whether the Supertonic GGUF (all
// four stages in one file) fits the device memory available right now,
// without reading any weight data.
//
// The projection mirrors one Engine load plus one synthesize() BY
// CONSTRUCTION: the same loader runs metadata-only (same backend policy and
// capability probes, the same per-tensor storage-type decisions --
// precision / f16-weights materialisation -- the same pre-baked F2/F6
// declarations and the GPU pre-transpose roster), and the same stage graph
// builders are priced through ggml's size-only APIs.  What is priced:
//
//   weights -- buffer_w (every tensor at its RESOLVED storage type, plus the
//              pre-baked audit tensors) and, on GPU, buffer_w_extra (the
//              pre-transposed copies of every 2-D matmul weight -- roughly a
//              second copy of the matmul set)
//   compute -- the resident per-stage graph-cache arenas, which all stay
//              alive across the whole synthesis and across calls, so they
//              SUM.  Each stage is priced through the ONE-GRAPH builder the
//              runtime dispatches off the CPU backend, so the priced graph is
//              the executed graph by construction: the one-graph text encoder
//              (embedding + convnext chain + relpos attn/ffn stack + speech-
//              prompted tail; its persistent L x L rel_band makes it
//              quadratic in the text length), the one-graph duration sentence
//              encoder, the vector estimator's all-steps-in-one [C, T] loop
//              graph (the dominant arena; CFG batches cond | uncond along
//              time, roughly doubling it on supertonic3), and the vocoder
//              graph (dual-path priced: its CPU-fallback portion lands in
//              host extras)
//
// Host extras cover the persistent host caches a load keeps (unicode
// indexer, RoPE theta, layer-norm/tanh_k pre-downloads, the ~5 MiB scalar
// weight cache) and the synthesis working set (text embedding, encoder
// scratch, latents, the waveform); the LOAD TRANSIENT -- gguf_init's full
// host copy of the tensor data plus the conversion staging, which often IS
// the process peak -- is projected separately and the verdict takes the
// larger of load-time and synth-time host pressure.
//
// COVERAGE / REFUSALS (honest partial coverage; a wrong FITS is never
// emitted):
//   * The fused one-graph dispatch paths are modelled: GPU backends and CPU
//     builds without Accelerate/CBLAS pointwise kernels. When the resolved
//     CPU backend uses those pointwise kernels, or a one-graph path is env-disabled
//     (SUPERTONIC_DISABLE_LOOP_GRAPH / SUPERTONIC_DISABLE_ONE_GRAPH /
//     SUPERTONIC_DISABLE_TEXT_ONE_GRAPH /
//     SUPERTONIC_DISABLE_DURATION_ONE_GRAPH), fit_params returns
//     Error / "compute-path-not-supported": the pointwise path runs per-island
//     multi-cache sets this projection does not model yet (follow-up under
//     QVAC-24283).
//   * The projection covers the batch path; streaming runs the same stages
//     per chunk at smaller shapes, so the batch projection at the same text
//     length bounds it.
//   * A latent length is a PREDICTION of the duration model at runtime; the
//     audio_seconds knob sets the projected utterance length explicitly.
//
// See tts-cpp/fit.h for the status contract and result layout.

#include "tts-cpp/export.h"
#include "tts-cpp/fit.h"

#include <cstdint>
#include <string>

namespace tts_cpp {
namespace supertonic {

struct FitOptions {
    // Required; same role as EngineOptions::model_gguf_path.
    std::string model_gguf_path;

    // Same semantics as the EngineOptions fields of the same name. The
    // resolved dispatch must use the fused graph path (see coverage above):
    // a validated GPU, or CPU without Accelerate/CBLAS pointwise kernels.
    int         n_gpu_layers  = 0;
    int         vulkan_device = 0;
    std::string backend;
    // Mirrors the default Engine allocation policy by construction.
    std::string precision = "auto";
    // EngineOptions::f16_weights tri-state (-1 auto / 0 off / 1 on).
    int         f16_weights = -1;
    std::string backends_dir;

    // ── Workload ──────────────────────────────────────────────────────────
    // Text length in Supertonic text tokens (Unicode code points).  The text
    // encoder and duration graphs follow it; the relative-position caches
    // are quadratic in it.
    int text_tokens = 128;

    // Projected utterance length in seconds; sets the latent length
    // (ceil(seconds * sample_rate / (base_chunk_size *
    // ttl_chunk_compress_factor)) frames) the vector estimator and vocoder
    // are priced at.  At runtime this comes out of the duration predictor.
    float audio_seconds = 10.0f;

    // CFM steps; 0 = the GGUF's default_steps.  The loop graph's node budget
    // and arena scale with it.
    int steps = 0;

    // Free-memory headroom that must remain on the device for the projection
    // to count as fitting.
    uint64_t margin_bytes = 256ull * 1024 * 1024;
};

// Project the model + workload in `opts` against the device memory available
// right now.  Reads only GGUF metadata, never weight data; builds and
// measures graphs but never allocates or executes them.  Never throws for a
// "does not fit" outcome (a valid Failure result), an unreadable model
// (Error), or an unmodelled dispatch path (Error /
// "compute-path-not-supported"); see FitStatus in tts-cpp/fit.h.
TTS_CPP_API FitResult fit_params(const FitOptions & opts);

}  // namespace supertonic
}  // namespace tts_cpp

// CLI front-end over supertonic::fit_params (the supertonic-fit-params
// tool); lives in the library so hosts can link it directly.  Exit code ==
// fit status: 0 fits, 1 does not fit, 2 error.
extern "C" TTS_CPP_API int supertonic_fit_cli_main(int argc, char ** argv);
