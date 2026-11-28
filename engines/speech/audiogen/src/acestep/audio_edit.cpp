#include "audio_edit.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace tts_cpp::acestep {
namespace {

constexpr float ROUND_HALF = 0.5f;
constexpr int FADE_ENDPOINT_OFFSET = 1;
constexpr int ODD_INTEGER_BIT = 1;
constexpr char REPAINT_RANGE_FINITE_ERROR[] = "repaint range must be finite";
constexpr char REPAINT_SOURCE_EMPTY_ERROR[] = "repaint source is empty";
constexpr char REPAINT_OUTPAINT_ERROR[] =
    "repaint outpainting is not supported; range must stay inside source audio";
constexpr char REPAINT_RANGE_ORDER_ERROR[] = "repaint end must be greater than start";
constexpr char REPAINT_RANGE_SHORT_ERROR[] = "repaint range is shorter than one latent frame";
constexpr char REPAINT_INJECTION_SHAPE_ERROR[] = "repaint source injection shape mismatch";
constexpr char REPAINT_BLEND_SHAPE_ERROR[] = "repaint latent blend shape mismatch";
constexpr char REPAINT_CHANNELS_ERROR[] = "repaint waveform channels must be positive";
constexpr char FLOW_SOURCE_CAPTION_ERROR[] = "flow-edit source_caption is required";
constexpr char FLOW_TARGET_CAPTION_ERROR[] = "flow-edit target_caption is required";
constexpr char FLOW_RANGE_ERROR[] = "flow-edit requires 0 <= n_min <= n_max <= 1";
constexpr char FLOW_AVERAGE_ERROR[] = "flow-edit n_avg must be at least 1";
constexpr char FLOW_GUIDANCE_ERROR[] =
    "flow-edit v1 supports turbo no-CFG only (diffusion_guidance_scale must be 1)";
constexpr char FLOW_DCW_ERROR[] = "flow-edit v1 does not support DCW";
constexpr char FLOW_ADG_ERROR[] = "flow-edit v1 does not support ADG";
constexpr char FLOW_HEUN_ERROR[] = "flow-edit v1 supports Euler only";
constexpr char FLOW_SOURCE_SHAPE_ERROR[] = "flow-edit source/noise shape mismatch";
constexpr char FLOW_TARGET_SHAPE_ERROR[] = "flow-edit target shape mismatch";
constexpr char FLOW_VELOCITY_SHAPE_ERROR[] = "flow-edit velocity shape mismatch";
constexpr char REPAINT_CAPABILITY_ERROR[] = "repaint capability is unavailable";
constexpr char FLOW_CAPABILITY_ERROR[] = "flow-edit capability is unavailable";
constexpr char EDIT_OPERATION_NULL_ERROR[] = "audio edit operation cannot be null";
constexpr char REPAINT_MATERIALIZATION_ERROR[] = "repaint materialization capability is unavailable";

void inject_unmasked_frames(std::vector<float> & current, const float * clean_source,
                            const float * original_noise, const float * mask,
                            size_t frames, float t_next, int channels) {
    for (size_t frame = 0; frame < frames; ++frame) {
        if (mask[frame] != AUDIO_EDIT_MIN_RATIO) continue;
        for (int channel = 0; channel < channels; ++channel) {
            const size_t index = frame * (size_t) channels + channel;
            current[index] =
                t_next * original_noise[index] + (AUDIO_EDIT_MAX_RATIO - t_next) * clean_source[index];
        }
    }
}

void fill_left_fade(std::vector<float> & mask, int fade_start, int edit_start) {
    for (int frame = fade_start; frame < edit_start; ++frame) {
        mask[(size_t) frame] =
            (float) (frame - fade_start + FADE_ENDPOINT_OFFSET) /
            (float) (edit_start - fade_start + FADE_ENDPOINT_OFFSET);
    }
}

void fill_right_fade(std::vector<float> & mask, int edit_end, int fade_end) {
    for (int frame = edit_end; frame < fade_end; ++frame) {
        mask[(size_t) frame] =
            (float) (fade_end - frame) /
            (float) (fade_end - edit_end + FADE_ENDPOINT_OFFSET);
    }
}

void blend_latent_frames(std::vector<float> & generated, const float * clean_source,
                         const std::vector<float> & mask, int channels) {
    for (size_t frame = 0; frame < mask.size(); ++frame) {
        const float generated_weight = mask[frame];
        for (int channel = 0; channel < channels; ++channel) {
            const size_t index = frame * (size_t) channels + channel;
            generated[index] = generated_weight * generated[index] +
                               (AUDIO_EDIT_MAX_RATIO - generated_weight) * clean_source[index];
        }
    }
}

void blend_waveform_samples(std::vector<float> & generated, const std::vector<float> & source,
                            const std::vector<float> & mask, int samples, int channels) {
    for (int sample = 0; sample < samples; ++sample) {
        for (int channel = 0; channel < channels; ++channel) {
            const size_t index = (size_t) sample * channels + channel;
            generated[index] = mask[(size_t) sample] * generated[index] +
                               (AUDIO_EDIT_MAX_RATIO - mask[(size_t) sample]) * source[index];
        }
    }
}

void mix_source_with_noise(const std::vector<float> & source, const std::vector<float> & noise,
                           float timestep, std::vector<float> & output) {
    for (size_t index = 0; index < source.size(); ++index) {
        output[index] = (AUDIO_EDIT_MAX_RATIO - timestep) * source[index] + timestep * noise[index];
    }
}

void offset_edit_with_source_noise(const std::vector<float> & edit,
                                   const std::vector<float> & noisy_source,
                                   const std::vector<float> & source,
                                   std::vector<float> & output) {
    for (size_t index = 0; index < source.size(); ++index) {
        output[index] = edit[index] + noisy_source[index] - source[index];
    }
}

void integrate_velocity_difference(std::vector<float> & edit,
                                   const std::vector<float> & target_velocity,
                                   const std::vector<float> & source_velocity, float dt) {
    for (size_t index = 0; index < edit.size(); ++index) {
        edit[index] += dt * (target_velocity[index] - source_velocity[index]);
    }
}

void execute_operations(const std::vector<std::unique_ptr<AudioEditOperation>> & operations,
                        AudioEditArtifact & artifact, const AudioEditCapabilities & capabilities) {
    for (const auto & operation : operations) {
        if (capabilities.cancelled && capabilities.cancelled()) break;
        if (operation->requires_materialized_source() && !artifact.pcm_is_current) {
            if (!capabilities.prepare_repaint_source) {
                throw std::logic_error(REPAINT_MATERIALIZATION_ERROR);
            }
            capabilities.prepare_repaint_source(artifact);
        }
        if (capabilities.cancelled && capabilities.cancelled()) break;
        operation->execute(artifact, capabilities);
    }
}

void add_plan_operations(AudioEditPipeline & pipeline, const std::vector<AudioEditParams> & plan) {
    for (const AudioEditParams & params : plan) {
        pipeline.add(make_audio_edit_operation(params));
    }
}

}

int audio_edit_round_ties_to_even(float value) {
    const float lower = std::floor(value);
    const float fraction = value - lower;
    if (fraction < ROUND_HALF) return (int) lower;
    if (fraction > ROUND_HALF) return (int) lower + 1;
    const int lower_integer = (int) lower;
    return (lower_integer & ODD_INTEGER_BIT) == 0 ? lower_integer
                                                   : lower_integer + ODD_INTEGER_BIT;
}

RepaintConfig resolve_repaint_config(RepaintMode mode, float strength) {
    strength = std::clamp(strength, AUDIO_EDIT_MIN_RATIO, AUDIO_EDIT_MAX_RATIO);
    if (mode == RepaintMode::Aggressive) {
        return { REPAINT_AGGRESSIVE_INJECTION_RATIO, REPAINT_AGGRESSIVE_BLEND_FRAMES,
                 REPAINT_AGGRESSIVE_FADE_SECONDS, false };
    }
    if (mode == RepaintMode::Conservative) {
        return { REPAINT_CONSERVATIVE_INJECTION_RATIO, REPAINT_CONSERVATIVE_BLEND_FRAMES,
                 REPAINT_CONSERVATIVE_FADE_SECONDS, true };
    }
    const float preserve = AUDIO_EDIT_MAX_RATIO - strength;
    return { preserve,
             audio_edit_round_ties_to_even((float) REPAINT_CONSERVATIVE_BLEND_FRAMES * preserve),
             REPAINT_CONSERVATIVE_FADE_SECONDS * preserve, true };
}

std::string resolve_repaint_range(float start_seconds, float end_seconds, int source_samples, int latent_frames,
                                  RepaintRange & range) {
    if (!std::isfinite(start_seconds) || !std::isfinite(end_seconds)) return REPAINT_RANGE_FINITE_ERROR;
    if (source_samples <= 0 || latent_frames <= 0) return REPAINT_SOURCE_EMPTY_ERROR;
    const float source_duration = (float) source_samples / AUDIO_EDIT_SAMPLE_RATE;
    const bool ends_at_source = end_seconds < AUDIO_EDIT_MIN_RATIO ||
                                std::fabs(end_seconds - source_duration) <= REPAINT_RANGE_EPSILON_SECONDS;
    if (end_seconds < AUDIO_EDIT_MIN_RATIO) end_seconds = source_duration;
    if (start_seconds < AUDIO_EDIT_MIN_RATIO ||
        end_seconds > source_duration + REPAINT_RANGE_EPSILON_SECONDS) {
        return REPAINT_OUTPAINT_ERROR;
    }
    if (end_seconds <= start_seconds) return REPAINT_RANGE_ORDER_ERROR;
    range.start_seconds = start_seconds;
    range.end_seconds = std::min(end_seconds, source_duration);
    range.sample_start = std::clamp((int) (range.start_seconds * AUDIO_EDIT_SAMPLE_RATE), 0, source_samples);
    range.sample_end = ends_at_source
                           ? source_samples
                           : std::clamp((int) (range.end_seconds * AUDIO_EDIT_SAMPLE_RATE),
                                        range.sample_start, source_samples);
    range.latent_start = std::clamp((int) (range.start_seconds * AUDIO_EDIT_LATENT_RATE), 0, latent_frames);
    range.latent_end = ends_at_source
                           ? latent_frames
                           : std::clamp((int) (range.end_seconds * AUDIO_EDIT_LATENT_RATE),
                                        range.latent_start, latent_frames);
    if (range.sample_end <= range.sample_start || range.latent_end <= range.latent_start) {
        return REPAINT_RANGE_SHORT_ERROR;
    }
    return {};
}

std::vector<float> make_repaint_mask(int latent_frames, int start_frame, int end_frame) {
    std::vector<float> mask((size_t) std::max(0, latent_frames), AUDIO_EDIT_MIN_RATIO);
    start_frame = std::clamp(start_frame, 0, latent_frames);
    end_frame = std::clamp(end_frame, start_frame, latent_frames);
    std::fill(mask.begin() + start_frame, mask.begin() + end_frame, AUDIO_EDIT_MAX_RATIO);
    return mask;
}

void repaint_inject_source(std::vector<float> & current, const float * clean_source,
                           const float * original_noise, const float * mask, size_t frames,
                           float t_next, int channels) {
    if (channels <= 0 || !clean_source || !original_noise || !mask ||
        current.size() != frames * (size_t) channels) {
        throw std::invalid_argument(REPAINT_INJECTION_SHAPE_ERROR);
    }
    inject_unmasked_frames(
        current, clean_source, original_noise, mask, frames, t_next, channels);
}

void repaint_blend_latent(std::vector<float> & generated, const float * clean_source,
                          const float * mask, size_t frames, int crossfade_frames,
                          int channels) {
    if (channels <= 0 || !clean_source || !mask ||
        generated.size() != frames * (size_t) channels) {
        throw std::invalid_argument(REPAINT_BLEND_SHAPE_ERROR);
    }
    std::vector<float> soft(mask, mask + frames);
    if (crossfade_frames > 0) {
        const auto first = std::find(soft.begin(), soft.end(), AUDIO_EDIT_MAX_RATIO);
        const auto last = std::find(soft.rbegin(), soft.rend(), AUDIO_EDIT_MAX_RATIO);
        if (first != soft.end() && last != soft.rend()) {
            const int left = (int) std::distance(soft.begin(), first);
            const int right = (int) soft.size() - (int) std::distance(soft.rbegin(), last);
            const int fade_start = std::max(0, left - crossfade_frames);
            const int fade_end = std::min((int) soft.size(), right + crossfade_frames);
            fill_left_fade(soft, fade_start, left);
            fill_right_fade(soft, right, fade_end);
        }
    }
    blend_latent_frames(generated, clean_source, soft, channels);
}

void repaint_splice_waveform(std::vector<float> & generated, const std::vector<float> & source,
                             int start_sample, int end_sample, int crossfade_samples, int channels) {
    if (channels <= 0) throw std::invalid_argument(REPAINT_CHANNELS_ERROR);
    const int samples = (int) (std::min(generated.size(), source.size()) / (size_t) channels);
    if (samples <= 0) return;
    generated.resize((size_t) samples * channels);
    start_sample = std::clamp(start_sample, 0, samples);
    end_sample = std::clamp(end_sample, start_sample, samples);
    if (start_sample == 0 && end_sample == samples) return;
    std::vector<float> mask((size_t) samples, AUDIO_EDIT_MIN_RATIO);
    std::fill(mask.begin() + start_sample, mask.begin() + end_sample, AUDIO_EDIT_MAX_RATIO);
    if (crossfade_samples > 0) {
        const int fade_start = std::max(0, start_sample - crossfade_samples);
        const int fade_end = std::min(samples, end_sample + crossfade_samples);
        fill_left_fade(mask, fade_start, start_sample);
        fill_right_fade(mask, end_sample, fade_end);
    }
    blend_waveform_samples(generated, source, mask, samples, channels);
}

std::string validate_flow_edit_params(const FlowEditParams & params) {
    if (params.source_caption.empty()) return FLOW_SOURCE_CAPTION_ERROR;
    if (params.target_caption.empty()) return FLOW_TARGET_CAPTION_ERROR;
    if (!std::isfinite(params.n_min) || !std::isfinite(params.n_max) ||
        params.n_min < AUDIO_EDIT_MIN_RATIO || params.n_min > params.n_max ||
        params.n_max > AUDIO_EDIT_MAX_RATIO) {
        return FLOW_RANGE_ERROR;
    }
    if (params.n_avg < FLOW_EDIT_DEFAULT_AVERAGES) return FLOW_AVERAGE_ERROR;
    if (!std::isfinite(params.diffusion_guidance_scale) ||
        params.diffusion_guidance_scale != FLOW_EDIT_NO_CFG_SCALE) {
        return FLOW_GUIDANCE_ERROR;
    }
    if (params.dcw_enabled) return FLOW_DCW_ERROR;
    if (params.use_adg) return FLOW_ADG_ERROR;
    if (params.use_heun) return FLOW_HEUN_ERROR;
    return {};
}

void flow_edit_make_source(const std::vector<float> & source, const std::vector<float> & noise, float timestep,
                           std::vector<float> & output) {
    if (source.size() != noise.size()) throw std::invalid_argument(FLOW_SOURCE_SHAPE_ERROR);
    output.resize(source.size());
    mix_source_with_noise(source, noise, timestep, output);
}

void flow_edit_make_target(const std::vector<float> & edit, const std::vector<float> & noisy_source,
                           const std::vector<float> & source, std::vector<float> & output) {
    if (edit.size() != source.size() || noisy_source.size() != source.size()) {
        throw std::invalid_argument(FLOW_TARGET_SHAPE_ERROR);
    }
    output.resize(source.size());
    offset_edit_with_source_noise(edit, noisy_source, source, output);
}

void flow_edit_integrate_delta(std::vector<float> & edit, const std::vector<float> & target_velocity,
                               const std::vector<float> & source_velocity, float dt) {
    if (edit.size() != target_velocity.size() || edit.size() != source_velocity.size()) {
        throw std::invalid_argument(FLOW_VELOCITY_SHAPE_ERROR);
    }
    integrate_velocity_difference(edit, target_velocity, source_velocity, dt);
}

RepaintOperation::RepaintOperation(RepaintParams params) : params_(std::move(params)) {}
const char * RepaintOperation::name() const { return AUDIO_EDIT_REPAINT_STAGE; }
bool RepaintOperation::requires_materialized_source() const { return true; }
void RepaintOperation::execute(AudioEditArtifact & artifact, const AudioEditCapabilities & capabilities) const {
    if (!capabilities.repaint) throw std::logic_error(REPAINT_CAPABILITY_ERROR);
    capabilities.repaint(params_, artifact);
}

FlowEditOperation::FlowEditOperation(FlowEditParams params) : params_(std::move(params)) {}
const char * FlowEditOperation::name() const { return AUDIO_EDIT_FLOW_STAGE; }
bool FlowEditOperation::requires_materialized_source() const { return false; }
void FlowEditOperation::execute(AudioEditArtifact & artifact, const AudioEditCapabilities & capabilities) const {
    if (!capabilities.flow_edit) throw std::logic_error(FLOW_CAPABILITY_ERROR);
    capabilities.flow_edit(params_, artifact);
}

std::unique_ptr<AudioEditOperation> make_audio_edit_operation(const AudioEditParams & params) {
    return std::visit(
        [](const auto & value) -> std::unique_ptr<AudioEditOperation> {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, RepaintParams>) {
                return std::make_unique<RepaintOperation>(value);
            } else {
                return std::make_unique<FlowEditOperation>(value);
            }
        },
        params);
}

void AudioEditPipeline::add(std::unique_ptr<AudioEditOperation> operation) {
    if (!operation) throw std::invalid_argument(EDIT_OPERATION_NULL_ERROR);
    operations_.push_back(std::move(operation));
}

void AudioEditPipeline::execute(AudioEditArtifact & artifact, const AudioEditCapabilities & capabilities) const {
    execute_operations(operations_, artifact, capabilities);
}

AudioEditPipeline make_audio_edit_pipeline(const std::vector<AudioEditParams> & plan) {
    AudioEditPipeline pipeline;
    add_plan_operations(pipeline, plan);
    return pipeline;
}

}
