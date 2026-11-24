#pragma once

// ACE-Step DiT (Diffusion Transformer) — ggml compute engine.
//
// The generative core: a 24-layer transformer (AdaLN, GQA self-attention with
// per-layer sliding-window / full alternation, cross-attention to the text
// encoder states, SwiGLU MLP) run as a ggml graph, one forward per flow-matching
// step. Ported from acestep.cpp/src/dit*.h. Every op it needs already exists in
// the fabric ggml (rms_norm, mul_mat, rope_ext, flash_attn_ext /
// soft_max_ext, swiglu[_split], timestep_embedding, conv_transpose_1d) — no new
// custom op, unlike the VAE. On CPU, attention uses the F32 soft_max_ext path
// (flash_attn_ext accumulates in F16 and drifts over 24 layers x 8 steps).
//
// Layout: math [S, H] == ggml ne[0]=H, ne[1]=S. Latents are channel-major per
// frame: latent[t*C + c]. The sampler (Euler loop) lives in the engine glue.

#include "ggml-backend.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace tts_cpp::acestep {

inline constexpr float DIT_REPAINT_DISABLED_RATIO = 0.0f;
inline constexpr int DIT_REPAINT_DISABLED_CROSSFADE = 0;
inline constexpr float DIT_FLOW_EDIT_MIN_RATIO = 0.0f;
inline constexpr float DIT_FLOW_EDIT_MAX_RATIO = 1.0f;
inline constexpr int DIT_FLOW_EDIT_DEFAULT_AVERAGES = 1;

// Populated from GGUF metadata (acestep-dit.* / acestep.*).
struct DitConfig {
    int   hidden_size       = 0;
    int   intermediate_size = 0;
    int   n_heads           = 0;
    int   n_kv_heads        = 0;
    int   head_dim          = 0;
    int   n_layers          = 0;
    int   in_channels       = 0;
    int   out_channels      = 0;
    int   patch_size        = 0;
    int   sliding_window    = 0;
    float rope_theta        = 0.0f;
    float rms_norm_eps      = 0.0f;
    int   enc_hidden_size   = 0;     // condition_embedder input dim (from weight shape)
    bool  is_turbo          = true;  // acestep.is_turbo: turbo=8 steps/shift 3; base/sft=50/shift 1
    bool  is_sft            = false; // general.name contains "sft": CFG-capable, no stem tasks
    std::string model_name;          // general.name from the GGUF (checkpoint dir name)
};

// convert.py stamps general.name with the checkpoint directory name
// (acestep-v15-base / acestep-v15-sft / acestep-v15-xl-sft / ...).
inline bool is_sft_model_name(const std::string & model_name) {
    return model_name.find("sft") != std::string::npos;
}

struct DitModel;             // opaque: fused weight tensors + backend weight buffer
struct AcestepStageMeasure;  // fit_measure.h

// Load DiT weights from `path` onto `backend` (borrowed). Returns nullptr on
// failure. Reads config from GGUF metadata; fuses QKV / gate-up when tensor
// types match and pre-permutes proj_in/proj_out convs at load time.
DitModel *        dit_model_load(const std::string & path, ggml_backend_t backend, int n_threads, bool verbose);
void              dit_model_free(DitModel * m);
const DitConfig & dit_model_config(const DitModel * m);
size_t            dit_model_weight_bytes(const DitModel * m);

// Metadata-only load for the memory-fit preflight: identical tensor wiring to
// dit_model_load, but the weight allocation is SIZED into `measure` instead of
// performed and no tensor data is read. Only good for the measure-mode forward
// below; free with dit_model_free.
DitModel * dit_model_load_metadata_only(const std::string & path, ggml_backend_t backend,
                                        int n_threads, bool verbose, AcestepStageMeasure & measure);

// Compute-buffer bytes of the persistent forward-graph cache (0 when no graph
// has been built); lets the fit parity tests compare projection vs the real
// resident allocation.
size_t dit_model_compute_buffer_bytes(const DitModel * m);

// Inputs for one DiT forward (velocity prediction). N = batch (CFG uses N=2).
struct DitForwardInputs {
    const float * input_latents = nullptr;  // [in_channels, T, N], channel-major per frame
    int           T             = 0;        // temporal length (multiple of patch_size)
    int           N             = 1;        // batch size

    const float * enc_hidden = nullptr;     // [H_enc, enc_S, N] text-encoder states (cond-embedded input)
    int           enc_S      = 0;
    int           H_enc      = 0;

    float t   = 0.0f;   // flow-matching timestep
    float t_r = 0.0f;   // reference timestep (turbo: t_r == t)

    // Attention masks (F16), row-major [KV, Q, 1, N]; null = unmasked.
    const void * sa_mask_sw = nullptr;  // [S, S, 1, N] self-attn sliding window
    const void * ca_mask    = nullptr;  // [enc_S, S, 1, N] cross-attn encoder padding

    // false = the inputs are unchanged since the previous call on this model,
    // so their uploads are skipped (their graph slots persist). cond_dirty
    // covers enc_hidden and ca_mask (they alternate between the cond and
    // uncond pass under CFG); constants_dirty covers positions and sa_mask.
    bool cond_dirty      = true;
    bool constants_dirty = true;
};

// Run one forward pass. Writes velocity [out_channels, T, N] (channel-major per
// frame) to `velocity_out`. Returns false on failure.
// When `measure_compute` is non-null the call builds the identical graph for
// `in`'s shapes but only SIZES its compute buffer (the size-only twin of the
// persistent graph-cache allocation) -- nothing is allocated, uploaded, or
// computed, and the graph cache is left untouched. The data pointers in `in`
// may then be null; `sa_mask_sw` / `ca_mask` still select the masked graph
// shape by non-nullness (pass any non-null pointer, as dit_sample always does).
bool dit_model_forward(DitModel * m, const DitForwardInputs & in, std::vector<float> & velocity_out,
                       size_t * measure_compute = nullptr);

// One cross-attention head to capture during the lyric-alignment probe.
struct DitAttentionHead {
    int layer = 0;
    int head  = 0;
};

// Inputs for the final-timestep probe used by lyric alignment: one forward at
// t = 1/num_steps over the retained conditioning and the denoised latent.
struct DitAttentionProbeInputs {
    const float * context    = nullptr;  // [in_channels-out_channels, T] frame-major
    const float * latent     = nullptr;  // [out_channels, T] denoised latent
    const float * enc_hidden = nullptr;  // [H_enc, enc_S]
    int           T          = 0;
    int           enc_S      = 0;
    int           H_enc      = 0;
    int           real_enc_S = 0;
    int           num_steps  = 0;
    long long     seed       = 0;        // philox noise blended into the probe latent
};

// Runs the probe with explicit softmax on the captured cross-attention layers,
// building the graph only up to the deepest captured layer. Appends one
// [enc_S * S] buffer per head in ggml layout (enc_S contiguous per frame row).
// Fully separate from the sampling graph cache: normal inference is untouched.
bool dit_probe_cross_attention(DitModel *                            m,
                               const DitAttentionProbeInputs &       in,
                               const std::vector<DitAttentionHead> & heads,
                               std::vector<std::vector<float>> &     captured_out);

// Flow-matching timestep schedule (ACE-Step default):
//   t_i = shift * t / (1 + (shift-1)*t),  t = 1 - i/num_steps.
// Turbo: shift=3.0, num_steps=8. Base/SFT: shift=1.0, num_steps=50.
void dit_build_schedule(float shift, int num_steps, std::vector<float> & schedule_out);

// Apply the official ACE-Step single-level Haar DCW correction along the
// temporal axis. x_next and denoised use row-major [N, T, C] layout.
void dit_apply_haar_dcw(std::vector<float> &       x_next,
                        const std::vector<float> & denoised,
                        int                        T,
                        int                        C,
                        int                        N,
                        float                      low_scale,
                        float                      high_scale);

// One full flow-matching denoise (Euler). guidance_scale > 1 with a
// null-condition embedding runs classifier-free guidance via APG (Adaptive
// Projected Guidance, official ACE-Step base/sft path): a second unconditional
// forward per step whose encoder states are the null embedding broadcast over
// enc_S, combined as pred_cond + (scale-1) * orthogonal(momentum(diff)).
struct DitSampleParams {
    const float * noise           = nullptr;  // [out_channels, T, N] initial x_T
    const float * context_latents = nullptr;  // [in_channels-out_channels, T, N] conditioning
    const float * enc_hidden      = nullptr;  // [H_enc, enc_S, N] text-encoder states
    int           enc_S           = 0;
    int           H_enc           = 0;
    int           T               = 0;
    int           N               = 1;
    const float * schedule        = nullptr;  // [num_steps] descending timesteps
    int           num_steps       = 0;
    float         guidance_scale  = 1.0f;     // <= 1 disables CFG
    const float * null_cond_emb   = nullptr;  // [H_enc] cond-model null embedding
    const int *   real_enc_S      = nullptr;  // [N] valid encoder lengths; null = all enc_S
    bool          dcw_enabled     = true;     // official ACE-Step Haar "double" mode
    float         dcw_scaler      = 0.05f;    // low band: t_curr * scaler
    float         dcw_high_scaler = 0.02f;    // high band: (1-t_curr) * scaler

    const float * repaint_mask          = nullptr;
    const float * clean_source_latents  = nullptr;
    float         repaint_injection_ratio = DIT_REPAINT_DISABLED_RATIO;
    int           repaint_crossfade_frames = DIT_REPAINT_DISABLED_CROSSFADE;
    bool          repaint_preserve_latent = false;

    // Cover conditioning switch (audio_cover_strength < 1): from step
    // `cover_switch_step` on, the context channels swap to `context_switch`
    // (silence) and the encoder states swap to `enc_hidden_switch` (the
    // text2music-instruction encoding, zero-padded to enc_S rows;
    // `real_enc_S_switch` carries its true per-batch lengths for the
    // cross-attention mask). -1 disables the switch.
    const float * context_switch    = nullptr;  // [in_channels-out_channels, T, N]
    const float * enc_hidden_switch = nullptr;  // [H_enc, enc_S, N]
    const int *   real_enc_S_switch = nullptr;  // [N]
    int           cover_switch_step = -1;

    // Optional per-step progress hook, fired at the start of each Euler step
    // with (step, num_steps). Return false to request cancellation (the sampler
    // then aborts and dit_sample returns false). The diffusion loop is the bulk
    // of generation time, so this is the signal that drives real UI progress.
    std::function<bool(int step, int total)> on_step;
};

// Writes the denoised latent [out_channels, T, N] to `latent_out`. Rebuilds the
// DiT graph per step (bring-up simplicity); correctness first, fusion later.
bool dit_sample(DitModel * m, const DitSampleParams & p, std::vector<float> & latent_out);

// APG combine for one step: velocity holds the conditional prediction on entry
// and the guided result on exit. momentum is the caller-held running average
// ([out_channels * T * N] doubles, zero-initialized before the first step).
void dit_apg_guide(std::vector<float> &       velocity,
                   const std::vector<float> & velocity_uncond,
                   std::vector<double> &      momentum,
                   float                      guidance_scale,
                   int                        T,
                   int                        Oc,
                   int                        N);

struct DitFlowEditCondition {
    const float * context_latents = nullptr;
    const float * enc_hidden      = nullptr;
    int           enc_S           = 0;
    int           H_enc           = 0;
    int           real_enc_S      = 0;
};

struct DitFlowEditParams {
    const float * source_latents = nullptr;
    int           T              = 0;
    const float * schedule       = nullptr;
    int           num_steps      = 0;
    float         n_min          = DIT_FLOW_EDIT_MIN_RATIO;
    float         n_max          = DIT_FLOW_EDIT_MAX_RATIO;
    int           n_avg          = DIT_FLOW_EDIT_DEFAULT_AVERAGES;
    uint64_t      seed           = 0;
    DitFlowEditCondition source;
    DitFlowEditCondition target;
    std::function<bool(int step, int total)> on_step;
};

bool dit_flow_edit(DitModel * m, const DitFlowEditParams & p,
                   std::vector<float> & latent_out);

} // namespace tts_cpp::acestep
