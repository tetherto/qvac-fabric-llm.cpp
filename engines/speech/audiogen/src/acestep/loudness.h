#pragma once

// Percentile loudness normalization for generated PCM, ported from
// acestep.cpp/src/audio-io.h audio_normalize. The reference sample is the
// (1 - peak_clip/1e6) percentile of the absolute values: scaling it to 1.0
// maximizes perceived loudness while hard-clipping only the tiny tail above
// it (peak_clip 0 = plain peak normalization, 10 = clip the top 0.001%).

#include <algorithm>
#include <cmath>
#include <vector>

namespace tts_cpp::acestep {

inline constexpr int   LOUDNESS_DEFAULT_PEAK_CLIP = 10;
inline constexpr int   LOUDNESS_MAX_PEAK_CLIP     = 999;
inline constexpr float LOUDNESS_SILENCE_FLOOR     = 1e-6f;

inline int clamp_peak_clip(int peak_clip) {
    return std::clamp(peak_clip, 0, LOUDNESS_MAX_PEAK_CLIP);
}

inline std::vector<float> loudness_absolute_values(const std::vector<float> & pcm) {
    std::vector<float> absolute(pcm.size());
    for (size_t i = 0; i < pcm.size(); i++) {
        absolute[i] = std::fabs(pcm[i]);
    }
    return absolute;
}

inline float loudness_percentile_reference(std::vector<float> & absolute, int peak_clip) {
    const double percentile = 1.0 - (double) clamp_peak_clip(peak_clip) / 1000000.0;
    const size_t index      = (size_t) ((double) (absolute.size() - 1) * percentile);
    std::nth_element(absolute.begin(), absolute.begin() + index, absolute.end());
    return absolute[index];
}

inline void apply_loudness_gain(std::vector<float> & pcm, float gain) {
    for (float & sample : pcm) {
        sample = std::clamp(sample * gain, -1.0f, 1.0f);
    }
}

inline void normalize_loudness(std::vector<float> & pcm, int peak_clip = LOUDNESS_DEFAULT_PEAK_CLIP) {
    if (pcm.empty()) return;
    std::vector<float> absolute  = loudness_absolute_values(pcm);
    const float        reference = loudness_percentile_reference(absolute, peak_clip);
    if (reference < LOUDNESS_SILENCE_FLOOR) return;
    apply_loudness_gain(pcm, 1.0f / reference);
}

} // namespace tts_cpp::acestep
