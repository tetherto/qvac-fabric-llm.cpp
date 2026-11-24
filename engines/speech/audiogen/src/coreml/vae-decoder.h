// Core ML Oobleck VAE decoder sidecar (Apple-only, compiled under
// AUDIOGEN_USE_COREML). Wraps a compiled `.mlmodelc` with one fixed-shape
// MLMultiArray input {1, 64, T_latent} and output {1, 2, T_latent * 1920}
// (Float16 or Float32), produced by scripts/export-vae-coreml.py. Every entry
// point fails soft (nullptr / non-zero) so the caller falls back to the ggml
// decode.

#pragma once

#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

struct acestep_coreml_vae_context;

struct acestep_coreml_vae_context * acestep_coreml_vae_init(const char * path_mlmodelc);

void acestep_coreml_vae_free(struct acestep_coreml_vae_context * ctx);

int64_t acestep_coreml_vae_window_frames(const struct acestep_coreml_vae_context * ctx);

// latent: time-major window of exactly window_frames frames, latent[t*64 + c].
// pcm: interleaved stereo destination of window_frames * 1920 * 2 floats.
int acestep_coreml_vae_decode(struct acestep_coreml_vae_context * ctx,
                              const float * latent,
                              float       * pcm);

const char * acestep_coreml_vae_backend_label(const struct acestep_coreml_vae_context * ctx);

#if defined(__cplusplus)
}
#endif
