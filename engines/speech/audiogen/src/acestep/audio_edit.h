#pragma once

#include "audiogen-cpp/acestep/engine.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::acestep {

inline constexpr int AUDIO_EDIT_SAMPLE_RATE = 48000;
inline constexpr int AUDIO_EDIT_CHANNELS = 2;
inline constexpr int AUDIO_EDIT_LATENT_RATE = 25;
inline constexpr int AUDIO_EDIT_LATENT_CHANNELS = 64;
inline constexpr float REPAINT_CONSERVATIVE_INJECTION_RATIO = 1.0f;
inline constexpr int REPAINT_CONSERVATIVE_BLEND_FRAMES = 25;
inline constexpr float REPAINT_CONSERVATIVE_FADE_SECONDS = 0.05f;
inline constexpr float REPAINT_AGGRESSIVE_INJECTION_RATIO = 0.0f;
inline constexpr int REPAINT_AGGRESSIVE_BLEND_FRAMES = 0;
inline constexpr float REPAINT_AGGRESSIVE_FADE_SECONDS = 0.0f;
inline constexpr int REPAINT_BALANCED_DEFAULT_BLEND_FRAMES = 13;
inline constexpr float REPAINT_BALANCED_DEFAULT_FADE_SECONDS = 0.025f;
inline constexpr float REPAINT_RANGE_EPSILON_SECONDS = 1e-5f;

struct RepaintConfig {
    float injection_ratio = REPAINT_DEFAULT_STRENGTH;
    int latent_blend_frames = REPAINT_BALANCED_DEFAULT_BLEND_FRAMES;
    float waveform_fade_sec = REPAINT_BALANCED_DEFAULT_FADE_SECONDS;
    bool preserve_waveform = true;
};

struct RepaintRange {
    int latent_start = 0;
    int latent_end = 0;
    int sample_start = 0;
    int sample_end = 0;
    float start_seconds = 0.0f;
    float end_seconds = 0.0f;
};

int audio_edit_round_ties_to_even(float value);
RepaintConfig resolve_repaint_config(RepaintMode mode, float strength);
std::string resolve_repaint_range(float start_seconds, float end_seconds, int source_samples, int latent_frames,
                                  RepaintRange & range);
std::vector<float> make_repaint_mask(int latent_frames, int start_frame, int end_frame);

void repaint_inject_source(std::vector<float> & current, const float * clean_source,
                           const float * original_noise, const float * mask, size_t frames,
                           float t_next, int channels);
void repaint_blend_latent(std::vector<float> & generated, const float * clean_source,
                          const float * mask, size_t frames, int crossfade_frames,
                          int channels);
void repaint_splice_waveform(std::vector<float> & generated, const std::vector<float> & source, int start_sample,
                             int end_sample, int crossfade_samples, int channels = AUDIO_EDIT_CHANNELS);

std::string validate_flow_edit_params(const FlowEditParams & params);
void flow_edit_make_source(const std::vector<float> & source, const std::vector<float> & noise, float t,
                           std::vector<float> & output);
void flow_edit_make_target(const std::vector<float> & edit, const std::vector<float> & noisy_source,
                           const std::vector<float> & source, std::vector<float> & output);
void flow_edit_integrate_delta(std::vector<float> & edit, const std::vector<float> & target_velocity,
                               const std::vector<float> & source_velocity, float dt);

struct AudioEditArtifact {
    std::vector<float> latent;
    int latent_frames = 0;
    std::vector<float> pcm;
    bool pcm_is_current = true;
    bool pending_waveform_splice = false;
    RepaintRange pending_range;
    int pending_crossfade_samples = 0;
};

struct AudioEditCapabilities {
    std::function<bool()> cancelled;
    std::function<void(AudioEditArtifact &)> prepare_repaint_source;
    std::function<void(const RepaintParams &, AudioEditArtifact &)> repaint;
    std::function<void(const FlowEditParams &, AudioEditArtifact &)> flow_edit;
};

class AudioEditOperation {
  public:
    virtual ~AudioEditOperation() = default;
    virtual const char * name() const = 0;
    virtual bool requires_materialized_source() const = 0;
    virtual void execute(AudioEditArtifact & artifact, const AudioEditCapabilities & capabilities) const = 0;
};

class RepaintOperation final : public AudioEditOperation {
  public:
    explicit RepaintOperation(RepaintParams params);
    const char * name() const override;
    bool requires_materialized_source() const override;
    void execute(AudioEditArtifact & artifact, const AudioEditCapabilities & capabilities) const override;
    const RepaintParams & params() const { return params_; }

  private:
    RepaintParams params_;
};

class FlowEditOperation final : public AudioEditOperation {
  public:
    explicit FlowEditOperation(FlowEditParams params);
    const char * name() const override;
    bool requires_materialized_source() const override;
    void execute(AudioEditArtifact & artifact, const AudioEditCapabilities & capabilities) const override;
    const FlowEditParams & params() const { return params_; }

  private:
    FlowEditParams params_;
};

std::unique_ptr<AudioEditOperation> make_audio_edit_operation(const AudioEditParams & params);

class AudioEditPipeline {
  public:
    void add(std::unique_ptr<AudioEditOperation> operation);
    size_t size() const { return operations_.size(); }
    const AudioEditOperation & at(size_t index) const { return *operations_.at(index); }
    void execute(AudioEditArtifact & artifact, const AudioEditCapabilities & capabilities) const;

  private:
    std::vector<std::unique_ptr<AudioEditOperation>> operations_;
};

AudioEditPipeline make_audio_edit_pipeline(const std::vector<AudioEditParams> & plan);

}
