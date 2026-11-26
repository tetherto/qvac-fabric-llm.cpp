#pragma once
// Pocket Core ML sidecars (TTS_CPP_COREML builds, Apple only).
//
// scripts/export-pocket-coreml.py exports them next to their GGUFs under the
// shared naming rule (coreml_sidecar_path.h):
//
//   flow-lm.gguf -> flow-lm-step.mlmodelc
//   mimi.gguf    -> mimi-decoder.mlmodelc
//
// The step is one generated frame of the FlowLM loop (input_linear, the
// transformer, out_norm and a one-step LSD flow) with the KV cache in Core ML
// state. The voice prefix and text prefill stay on ggml; their KV rows are
// copied into the state before a chunk's first frame, then every frame is one
// prediction. A chunk whose positions exceed the step's context runs FlowLM
// on ggml.
//
// The decoder is one Mimi decode frame with its streaming state (conv
// histories, the attention window) as explicit inputs and outputs kept here,
// so it runs on the Neural Engine. Either stage runs on ggml once a missing
// or retired sidecar leaves it.

#include "pocket/flow_lm.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct moss_coreml_model;

namespace tts_cpp::pocket::detail {

constexpr const char * COREML_DISABLE_ENV = "POCKET_COREML_DISABLE";
constexpr const char * COREML_STRICT_ENV = "POCKET_COREML_STRICT";
constexpr const char * COREML_UNITS_ENV = "POCKET_COREML_COMPUTE_UNITS";
// Core ML cannot place model state on the Neural Engine (a cpu_and_ane load
// fails); CPU + GPU measured fastest with the predictable latency on an M2.
constexpr const char * COREML_STEP_DEFAULT_UNITS = "cpu_and_gpu";
// The stateless decoder runs fastest on the Neural Engine, off the GPU the
// step uses (M2: 1.5 ms per frame vs 5.1 ms on CPU + GPU).
constexpr const char * COREML_DECODER_DEFAULT_UNITS = "cpu_and_ane";
constexpr const char * COREML_STEP_SUFFIX = "-step";
constexpr const char * COREML_DECODER_SUFFIX = "-decoder";
constexpr const char * COREML_SCHEMA = "1";
constexpr const char * GGML_STAGE_BACKEND = "ggml";
constexpr const char * MIXED_STAGE_BACKEND = "mixed";
// Must equal MASKED in scripts/pocket_coreml_models.py.
constexpr float COREML_MASKED = -1e4f;

// Resolved once per Engine. Explicit values never write the environment.
// Empty compute_units keeps each sidecar's default (coreml_units()).
struct CoremlPolicy {
    std::string compute_units;
    bool strict = false;
    bool disabled = false;
};

bool coreml_valid_units(const std::string & units);

// Empty `compute_units` inherits POCKET_COREML_COMPUTE_UNITS (an invalid
// inherited value warns and keeps the defaults); an explicit invalid value
// throws. A value applies to both sidecars. Unset `strict` inherits
// POCKET_COREML_STRICT (presence means true). POCKET_COREML_DISABLE keeps
// FlowLM and Mimi on ggml.
CoremlPolicy resolve_coreml_policy(const std::string & compute_units = {},
                                   std::optional<bool> strict = std::nullopt);

// The policy's compute units, or `sidecar_default` when it sets none.
std::string coreml_units(const CoremlPolicy & policy, const char * sidecar_default);

std::string coreml_step_path(const std::string & flow_lm_gguf);
std::string coreml_decoder_path(const std::string & mimi_gguf);

// True when this build has Core ML, the policy leaves it enabled and the
// sidecar sits next to the GGUF (presence only; nothing is loaded).
bool coreml_step_enabled(const std::string & flow_lm_gguf, const CoremlPolicy & policy);
bool coreml_decoder_enabled(const std::string & mimi_gguf, const CoremlPolicy & policy);

// cos/sin [head_dim / 2] of ggml_rope_ext mode 0 (adjacent pairs) at `position`.
void coreml_rope(int position, int head_dim, float base, std::vector<float> & cos, std::vector<float> & sin);

class CoremlStep {
public:
    // Loads the sidecar and checks, before any prediction, the shape of every
    // input, output and state the step uses against `config` (they are fixed
    // for the model's life, and the shared runner copies by the declared
    // shapes), its schema and that it was exported from the checkpoint
    // `source_hash` names; then runs one warm-up prediction at position 0,
    // which also fails for a sidecar with any further feature. Throws
    // std::runtime_error.
    CoremlStep(const std::string & path, const std::string & compute_units, const FlowConfig & config,
               const std::string & source_hash);
    ~CoremlStep();
    CoremlStep(const CoremlStep &) = delete;
    CoremlStep & operator=(const CoremlStep &) = delete;

    // Positions the state holds (voice + text + generated frames).
    int context() const { return context_; }
    const std::string & label() const { return label_; }

    // Copies cache rows (FlowLM::cache_rows) into the state at their positions.
    void write_rows(const CacheRows & rows);
    // Replaces the whole state with a zeroed one, then writes `rows`. Needed
    // after a failed prediction: its blended write can leave non-finite values
    // in any row, and neither a later blend (0 * NaN) nor the attention mask
    // clears them.
    void reset(const CacheRows & rows);

    // One frame at `position` (the cache holds every earlier one): latent
    // (bos_latent() for a chunk's first frame) and noise [latent_dim] ->
    // hidden [dim] (out_norm) and next_latent [latent_dim] (one flow step).
    // Throws on a failed prediction or a non-finite output.
    void step(int position, const std::vector<float> & latent, const std::vector<float> & noise,
              std::vector<float> & hidden, std::vector<float> & next_latent);

private:
    moss_coreml_model * model_ = nullptr;
    FlowConfig config_;
    int context_ = 0;
    std::string label_ = "coreml";
    std::vector<float> cos_, sin_, mask_, write_mask_, staging_;
    int masked_from_ = 0;  // mask_ is 0 before this position, MASKED from it
    int written_ = -1;     // position write_mask_ marks
};

// Mimi decoder geometry; must match scripts/pocket_coreml_mimi.py.
struct CoremlDecoderGeometry {
    static constexpr int dim = 512, latent = 32, heads = 8, head_dim = 64, layers = 2;
    static constexpr int keep = 249;            // retained attention rows (window - 1)
    static constexpr int upsample = 16;         // transformer rows per latent frame
    static constexpr int frame_samples = 1920;  // PCM samples per latent frame
    static constexpr int frames = 1;            // latent frames per prediction
};

class CoremlDecoder {
public:
    // Loads the sidecar and checks the shape of every input and output it
    // uses, its schema and that it was exported from the checkpoint
    // `source_hash` names, then runs one warm-up prediction (a sidecar with
    // any further feature fails it) and resets. Throws std::runtime_error.
    CoremlDecoder(const std::string & path, const std::string & compute_units, const std::string & source_hash);
    ~CoremlDecoder();
    CoremlDecoder(const CoremlDecoder &) = delete;
    CoremlDecoder & operator=(const CoremlDecoder &) = delete;

    const std::string & label() const { return label_; }
    // Starts an independent stream (Mimi::reset_decoder()).
    void reset();
    // latents [frames, 32], any frame count -> frames * 1920 PCM samples,
    // continuing the stream. Throws on a failed prediction or a non-finite
    // output; the stream is then unusable until reset().
    std::vector<float> decode(const std::vector<float> & latents);

private:
    void predict(const float * latent, float * pcm);

    moss_coreml_model * model_ = nullptr;
    std::string label_ = "coreml";
    int64_t position_ = 0;
    std::vector<float> keys_, values_, cos_, sin_, mask_, row_cos_, row_sin_;
    std::vector<std::vector<float>> histories_;
};

} // namespace tts_cpp::pocket::detail
