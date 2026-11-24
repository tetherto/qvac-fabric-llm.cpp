// Core ML FastConformer encoder sidecar for parakeet-cpp.
//
// An opaque context wraps a compiled `.mlmodelc` encoder; callers provide mel
// features and receive encoder hidden states. Compute policy and the validated
// sidecar determine CPU, GPU or Neural Engine execution. Preprocessing, language
// classification and stateful decoding stay native; offline TDT decode can use
// its separate Core ML sidecar.
//
// This header is referenced only with PARAKEET_USE_COREML. Initialization and
// prediction failures return status so callers can recover with native weights
// when those weights are available.

#pragma once

#include <stdint.h>
#ifndef __cplusplus
#include <stdbool.h>
#endif

#if defined(__cplusplus)
extern "C" {
#endif

struct parakeet_coreml_context;

// Load a compiled Core ML encoder from a `.mlmodelc` directory.
// Returns nullptr on any failure (missing directory, load error, unsupported
// OS, unexpected model interface) so the caller falls back to ggml.
struct parakeet_coreml_context * parakeet_coreml_init(const char * path_mlmodelc);

// Reliable policy for variable-length full-EU programs: current MPS dynamic
// gather/reshape kernels are not safe, so these programs execute on Core ML CPU.
struct parakeet_coreml_context * parakeet_coreml_init_cpu(const char * path_mlmodelc);
struct parakeet_coreml_context * parakeet_coreml_init_ane(const char * path_mlmodelc);
struct parakeet_coreml_context * parakeet_coreml_init_gpu(const char * path_mlmodelc);
// Select a named full-EU function in a shared-weight bundle (macOS 15/iOS 18+).
// The returned context owns its model and prediction buffers independently.
struct parakeet_coreml_context * parakeet_coreml_init_eu_function(
    const char * path_mlmodelc, const char * function_name, bool cpu_only);
bool parakeet_coreml_is_fp32(const struct parakeet_coreml_context * ctx);
bool parakeet_coreml_preserves_encoder_padding(const struct parakeet_coreml_context * ctx);

struct parakeet_coreml_context * parakeet_coreml_init_with_units_env(const char * path_mlmodelc,
                                                                     const char * units_env);

// Loads with resolved compute units (`all`, `cpu_and_ane`, `cpu_and_gpu`,
// `cpu_only`; null selects the export's own default, see
// parakeet_coreml_encoder_prefers_ane). Never reads the environment.
struct parakeet_coreml_context * parakeet_coreml_init_with_units(const char * path_mlmodelc,
                                                                 const char * units);

// The same load, reporting why it failed in `error` (may be null).
struct parakeet_coreml_load_error;
struct parakeet_coreml_context * parakeet_coreml_init_with_units_error(
    const char * path_mlmodelc, const char * units, struct parakeet_coreml_load_error * error);

// True when the bundle is the fixed-window Neural Engine export (its
// metadata declares the pad_mask / key_bias inputs), read without loading
// it: its default compute units are cpu_and_ane, every other export's are all.
bool parakeet_coreml_encoder_prefers_ane(const char * path_mlmodelc);

// The bundle's export kind from its metadata.json, without loading it: 1 the
// fixed-window Neural Engine export, 0 conclusively another export (its input
// schema names inputs, without the Neural Engine ones), -1 unknown (missing,
// unreadable or incomplete metadata; load the model to find out).
int parakeet_coreml_encoder_export_kind(const char * path_mlmodelc);

// Bytes of every fixed-shape input and output metadata.json declares (one
// function's I/O arrays, which the wrapper allocates at their declared
// shape), or 0 when the metadata does not declare complete fixed shapes.
uint64_t parakeet_coreml_declared_io_bytes(const char * path_mlmodelc);

// Release a context created by parakeet_coreml_init. Safe to call with nullptr.
void parakeet_coreml_free(struct parakeet_coreml_context * ctx);

// Full-EU sidecars must identify their route and encoder depth so legacy
// EU-sub encoders and accidentally exchanged sidecars cannot produce a transcript.
bool parakeet_coreml_matches_eu_route(const struct parakeet_coreml_context * ctx,
                                     const char * route, int layers, bool flexible);

// Return the fixed mel-frame capacity of the model input for `n_mels`.
// Returns 0 for flexible, enumerated-multi-shape, or unrecognised inputs.
// The value describes model capacity, not the current utterance length.
int64_t parakeet_coreml_fixed_mel_frames(
        const struct parakeet_coreml_context * ctx,
        int64_t n_mels);

int64_t parakeet_coreml_fixed_output_rows(
        const struct parakeet_coreml_context * ctx,
        int64_t cols);

// Encoder frames of seam context each long-form window should carry on both
// sides when the caller has no preference of its own: the fixed-window Neural
// Engine export was validated with 16 frames (1.28 s at the 80 ms stride),
// capped at a quarter of its window; PARAKEET_COREML_ANE_CONTEXT_FRAMES
// overrides it. 0 for every other export, meaning "use the engine default".
int64_t parakeet_coreml_seam_context_frames(const struct parakeet_coreml_context * ctx);

// Run the encoder on the Apple Neural Engine.
//
//   mel           log-mel features, row-major (n_mel_frames, n_mels),
//                 n_mels contiguous (the layout `compute_log_mel` emits).
//   n_mel_valid   frames of `mel` that hold audio; the frames from there to
//                 n_mel_frames are the front-end's zeroed padding and the
//                 fixed-window export masks them out at every resolution
//                 (<= 0 or >= n_mel_frames: every frame holds audio). Other
//                 Masked offline TDT exports use the same valid prefix; legacy
//                 single-input exports take the whole slab and ignore it.
//   encoder_out   destination for the encoder hidden states, row-major
//                 (n_enc_frames, d_model), d_model contiguous. The caller
//                 sizes it to n_enc_frames * d_model floats.
//
// The wrapper adapts the mel/output tensor orientation to whatever the exported
// model declares (features-major or time-major), so the mobius export only has
// to agree on the dimension sizes, not their order.
//
// Returns 0 on success; non-zero on a shape mismatch or prediction failure, in
// which case the caller falls back to the ggml encoder.
int parakeet_coreml_encode(struct parakeet_coreml_context * ctx,
                           int64_t       n_mel_frames,
                           int64_t       n_mel_valid,
                           int64_t       n_mels,
                           const float * mel,
                           int64_t       n_enc_frames,
                           int64_t       d_model,
                           float       * encoder_out);

// Fixed TDT sidecars exported with mel_valid_mask exclude padded frames from
// subsampling, attention and convolution. Legacy single-input sidecars return false.
bool parakeet_coreml_masks_mel_padding(const struct parakeet_coreml_context * ctx);

// valid_mel_frames is a positive contiguous prefix of the supplied mel array.
// For mask-aware sidecars n_enc_frames must be the subsampled valid length.
// Legacy sidecars keep their historical unmasked behavior.
int parakeet_coreml_encode_valid(struct parakeet_coreml_context * ctx,
                                 int64_t n_mel_frames, int64_t valid_mel_frames,
                                 int64_t n_mels, const float * mel,
                                 int64_t n_enc_frames, int64_t d_model,
                                 float * encoder_out);

// Sortformer v2.1 AOSC forward. `pre_encode` and `encoder_out` are both
// row-major (n_encoder_frames, d_model). The compiled input is a fixed
// capacity; the wrapper supplies the exported `valid_mask` when padding.
int parakeet_coreml_encode_embeddings(
        struct parakeet_coreml_context * ctx,
        int64_t n_encoder_frames,
        int64_t d_model,
        const float * pre_encode,
        float * encoder_out);

// Human-readable requested-compute label for stats/logging, e.g. "coreml-ane".
// Core ML may still fall back to CPU for individual unsupported operations.
const char * parakeet_coreml_backend_label(const struct parakeet_coreml_context * ctx);

#if defined(__cplusplus)
}
#endif
