#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace tts_cpp::minimax::detail {

constexpr int kDefaultFrameRate = 25;
constexpr int kDefaultMaxFrames = 9000;
constexpr int kDefaultFlowSteps = 20;
constexpr float kDefaultCfgScale = 1.7f;
constexpr int kWindowFrames = 200;

// How a window's latents are carried into, blended with and cropped against its
// neighbours. Everything follows the region two consecutive windows share,
// window_latents - hop_latents: the previous window's latents over that whole
// region are carried, the next window blends against the first half of it
// while sampling, and the seam sits a quarter of the way in, in the middle of
// the blend. Derived from the GGUF's own window and hop so the four values
// cannot drift from the file.
struct FlowWindowGeometry {
    int64_t carry_span = 0;
    int64_t overlap    = 0;
    int64_t crop_left  = 0;
    int64_t crop_right = 0;
};

// Where one RVQ depth-decoder step sits in the frame's K/V cache: how many
// positions it may attend over, how many it appends, and the first it appends.
struct DepthStepLayout {
    int64_t window = 0;
    int64_t tokens = 0;
    int64_t first  = 0;
};

struct ConditionRate {
    int input_sampling_rate = 24000;
    int input_hop_length = 960;
    int output_sampling_rate = 44100;
    int output_hop_length = 512;
};

struct CropSpan {
    int64_t left = 0;
    int64_t length = 0;
};

struct CarryRange {
    int64_t start = 0;
    int64_t end = 0;
};

struct ArCandidates {
    std::vector<float> conditional;
    std::vector<float> unconditional;
    std::vector<float> guided;
};

// The compact LM head packs the semantic block at row-offset 0, EOS as the last row.
constexpr int64_t kCompactHeadSemanticOffset = 0;

struct CompactHeadCopy {
    size_t src_offset = 0;
    size_t dst_offset = 0;
    size_t nbytes = 0;
};

struct SynthesisContract {
    std::string scheduler;
    bool invert_sigmas = false;
    float shift = 0.0f;
    int64_t train_timesteps = 0;
    std::string rope_type;
    std::string glu_order;
    bool timestep_token_prepended = false;
    bool pre_post_conv_residual = false;
    bool attn_bias = false;
};

struct AudioMetrics {
    float peak = 0.0f;
    double rms = 0.0;
};

struct ModelCompatibility {
    int64_t lm_embedding = 0;
    int64_t lm_codebooks = 0;
    int64_t lm_acoustic_vocab = 0;
    int64_t frame_rate = 0;
    int64_t max_audio_frames = 0;
    int64_t max_prompt_tokens = 0;
    int64_t depth_embedding = 0;
    int64_t depth_codebooks = 0;
    int64_t depth_acoustic_vocab = 0;
    int64_t condition_layers = 0;
    int64_t condition_hidden = 0;
    int64_t condition_out = 0;
    ConditionRate condition_rate{0, 0, 0, 0};
    int64_t dit_condition = 0;
    int64_t dit_channels = 0;
    int64_t window_frames = 0;
    int64_t hop_frames = 0;
    int64_t window_latents = 0;
    int64_t hop_latents = 0;
    int64_t vocoder_latent_channels = 0;
    int64_t vocoder_sampling_rate = 0;
    int64_t vocoder_channels = 0;
    int64_t vocoder_upsample = 0;
    std::vector<std::string> components;
};

struct ModelPair {
    std::string lm;
    std::string synth;
    std::string quant;
};

int64_t frames_from_duration(double duration, int frame_rate, int max_frames);
int64_t validate_frames(int64_t frames, int max_frames);
std::string build_prompt(const std::string & caption, const std::string & lyrics);
std::vector<int32_t> mask_unconditional(const std::vector<int32_t> & conditional, int32_t mask_token);
void fill_noise(uint64_t seed, int64_t window, std::vector<float> & output, int64_t count);
void flow_schedule(int steps, std::vector<float> & sigmas, std::vector<float> & timesteps);
void blend_latent_overlap(float * latents, const float * noise, const float * previous,
                          int64_t channels, int64_t latent_length, int64_t overlap,
                          int64_t previous_stride, float timestep);
void pin_latent_overlap(float * latents, const float * previous, int64_t channels,
                        int64_t latent_length, int64_t overlap, int64_t previous_stride);
int64_t condition_latent_length(const ConditionRate & rate, int64_t frames);
std::vector<int64_t> window_starts(int64_t frames, int64_t window_frames, int64_t hop_frames);
CropSpan crop_span(int64_t latent_length, int64_t window_index, int64_t window_count, int64_t upsample,
                   const FlowWindowGeometry & geometry);
CarryRange carry_range(int64_t latent_length, int64_t carry_span, int64_t overlap);
bool copy_carry_layout(const std::vector<float> & latents, const std::vector<float> & condition,
                       int64_t channels, int64_t condition_dimension, int64_t latent_length,
                       CarryRange range, std::vector<float> & carry_latents,
                       std::vector<float> & carry_condition);
bool copy_planar_window(const std::vector<float> & source, int64_t channels,
                        int64_t source_length, int64_t source_offset,
                        int64_t destination_length, int64_t destination_offset,
                        int64_t copy_length, std::vector<float> & destination);
int64_t stitched_sample_count(const std::vector<int64_t> & latent_lengths, int64_t upsample,
                              const FlowWindowGeometry & geometry);
std::string vocoder_upsample_error(const std::vector<int32_t> & rates, uint32_t total_upsample);
std::vector<std::string> validate_synthesis_contract(const SynthesisContract & contract);
std::string vocoder_output_shape_error(int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
                                       int64_t elements, size_t bytes, int64_t expected_length);
bool copy_range_fits(size_t source_elements, size_t destination_elements,
                     int64_t source_offset, int64_t destination_offset, int64_t copy_length);
bool stereo_copy_ranges_fit(size_t source_elements, size_t destination_elements,
                            int64_t source_length, int64_t source_offset,
                            int64_t destination_length, int64_t destination_offset,
                            int64_t copy_length);
bool finalize_audio(std::vector<float> & audio, AudioMetrics & metrics, size_t & nonfinite_index);
int64_t sample_top_k(const float * logits, int64_t count, int top_k, std::mt19937_64 & random);
bool collect_ar_candidates(const float * logits, int64_t row_stride, int64_t eos,
                           int64_t semantic_offset, int64_t semantic_vocab,
                           ArCandidates & candidates, int64_t & nonfinite_count);
int64_t compact_head_row_count(int64_t semantic_vocab);
bool compact_head_copy_plan(int64_t vocab, int64_t semantic_offset, int64_t semantic_vocab, int64_t eos,
                            size_t row_size, std::array<CompactHeadCopy, 2> & plan);
void guide_ar_candidates(ArCandidates & candidates, float cfg_scale);
void apply_conditional_top_k(ArCandidates & candidates, int top_k);
void build_acoustic_rows(const int32_t * codes, int64_t codebooks, int64_t acoustic_vocab,
                         std::vector<int32_t> & rows);
int64_t resolve_ar_frame_cap(int64_t requested_frames, int64_t checkpoint_frames,
                             bool forced, int64_t forced_length, std::string & error);
int resolve_flow_steps(int requested_steps);
FlowWindowGeometry flow_window_geometry(int64_t window_latents, int64_t hop_latents);
DepthStepLayout depth_step_layout(int codebook);
std::vector<std::string> validate_model_compatibility(const ModelCompatibility & model);
ModelPair resolve_model_pair(const std::string & model_dir, const std::string & explicit_lm,
                             const std::string & explicit_synth);
bool backend_configuration_matches(int active_references, int active_threads,
                                   const std::string & active_backends_dir, int requested_threads,
                                   const std::string & requested_backends_dir);
bool engine_instance_available(int active_instances);
bool cancellation_requested(const std::function<bool()> & callback);

}
