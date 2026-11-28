// Core ML sidecar for the Audio8 codec synthesis stack (Apple-only,
// TTS_CPP_USE_COREML). Export contract (scripts/export-audio8-codec-coreml.py):
// one MLMultiArray input {1, latent_dim, W}, one output {1, 1, W * frame_size},
// Float16 or Float32. Every entry point fails soft so the caller falls back to
// the ggml synthesis blocks.

#pragma once

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

struct audio8_coreml_codec_context;

struct audio8_coreml_codec_context * audio8_coreml_codec_init(const char * path_mlmodelc,
                                                              int64_t      latent_dim,
                                                              int64_t      frame_size);

void audio8_coreml_codec_free(struct audio8_coreml_codec_context * ctx);

int64_t audio8_coreml_codec_window_frames(const struct audio8_coreml_codec_context * ctx);

// post: window_frames frames, channels inner-most; pcm: window_frames * frame_size samples.
int audio8_coreml_codec_synthesize(struct audio8_coreml_codec_context * ctx,
                                   const float * post,
                                   float       * pcm);

// e.g. "coreml-all"; the requested placement, not a per-op guarantee.
const char * audio8_coreml_codec_backend_label(const struct audio8_coreml_codec_context * ctx);

#if defined(__cplusplus)
}
#endif
