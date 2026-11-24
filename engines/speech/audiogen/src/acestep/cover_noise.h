#pragma once

#include <cmath>
#include <vector>

namespace tts_cpp::acestep {

struct CoverNoiseResult {
    float nearest_time = 1.0f;
    int   remaining_steps = 0;
};

inline int find_nearest_schedule_index(const std::vector<float> & schedule, float target) {
    int   nearest = 0;
    float distance = std::fabs(schedule.front() - target);

    for (int index = 1; index < (int) schedule.size(); ++index) {
        const float candidate = std::fabs(schedule[(size_t) index] - target);
        if (candidate < distance) {
            distance = candidate;
            nearest  = index;
        }
    }
    return nearest;
}

inline void blend_cover_noise_frame(float * noise, const float * source, int channels, float time) {
    for (int channel = 0; channel < channels; ++channel) {
        noise[channel] = time * noise[channel] + (1.0f - time) * source[channel];
    }
}

inline void blend_cover_noise_frames(std::vector<float> & noise, const std::vector<float> & source,
                                     int frames, int source_frames, int channels, float time) {
    for (int frame = 0; frame < frames; ++frame) {
        const int source_frame = frame < source_frames ? frame : source_frames - 1;
        blend_cover_noise_frame(
            noise.data() + (size_t) frame * channels,
            source.data() + (size_t) source_frame * channels,
            channels,
            time);
    }
}

inline CoverNoiseResult apply_cover_noise(std::vector<float> & noise, const std::vector<float> & source,
                                          int frames, int source_frames, int channels, float strength,
                                          std::vector<float> & schedule) {
    const float target = 1.0f - strength;
    const int   start  = find_nearest_schedule_index(schedule, target);
    const float time   = schedule[(size_t) start];

    blend_cover_noise_frames(noise, source, frames, source_frames, channels, time);
    schedule.erase(schedule.begin(), schedule.begin() + start);
    return { time, (int) schedule.size() };
}

} // namespace tts_cpp::acestep
