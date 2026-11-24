// Core ML sidecar for the Supertonic vocoder (Apple-only, TTS_CPP_USE_COREML).
// Export contract (scripts/export-supertonic-coreml.py): one MLMultiArray
// input {1, latent_channels, W} (W latent frames), one output
// {1, 1, W * samples_per_frame}, Float16 or Float32. Every entry point fails
// soft so the caller falls back to the ggml vocoder graph.

#pragma once

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

struct supertonic_coreml_vocoder_context;

struct supertonic_coreml_vocoder_context * supertonic_coreml_vocoder_init(
    const char * path_mlmodelc,
    int64_t      latent_channels,
    int64_t      samples_per_frame);

void supertonic_coreml_vocoder_free(struct supertonic_coreml_vocoder_context * ctx);

// The exported window width in latent frames.
int64_t supertonic_coreml_vocoder_window_frames(
    const struct supertonic_coreml_vocoder_context * ctx);

// latent: the engine's whole channel-major [latent_channels, latent_len] host
// buffer.  Frames [begin, begin + filled) fill the window (zero-padded past
// `filled`); of the window's output, frames [skip_frames, skip_frames +
// keep_frames) are decoded into wav (keep_frames * samples_per_frame
// samples).  The windowed gather and the kept-slice decode live here so the
// caller pays no staging copies.
int supertonic_coreml_vocoder_synthesize(struct supertonic_coreml_vocoder_context * ctx,
                                         const float * latent,
                                         int64_t       latent_len,
                                         int64_t       begin,
                                         int64_t       filled,
                                         int64_t       skip_frames,
                                         int64_t       keep_frames,
                                         float       * wav);

// e.g. "coreml-all"; the requested placement, not a per-op guarantee.
const char * supertonic_coreml_vocoder_backend_label(
    const struct supertonic_coreml_vocoder_context * ctx);

#if defined(__cplusplus)
}
#endif
