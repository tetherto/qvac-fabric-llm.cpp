#pragma once

#include "logic.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

struct MM3WindowDimensions {
    int64_t latent_channels = 0;
    int64_t audio_channels = 0;
    int64_t condition_dimension = 0;
    int64_t upsample = 0;
    int64_t window_frames = 0;
    int64_t hop_frames = 0;
    int64_t carry_span = 0;
    int64_t overlap = 0;
    int64_t crop_left = 0;
    int64_t crop_right = 0;
    int64_t flow_steps = 0;
};

struct MM3WindowOperations {
    std::function<bool(const std::string &, int64_t, int64_t, int64_t, int64_t,
                       std::string *)>
        progress;
    std::function<bool(std::string *)> continue_generation;
    std::function<bool(int64_t, int64_t, int64_t, int64_t, std::vector<float> &,
                       int64_t &, std::string *)>
        condition;
    std::function<bool(int64_t, int64_t, std::vector<float> &)> noise;
    std::function<bool(int64_t, int64_t, int64_t, const std::vector<float> &,
                       const std::vector<float> &, int64_t, const std::vector<float> &,
                       std::vector<float> &,
                       const std::function<void(int64_t, int64_t)> &, std::string *)>
        flow;
    std::function<bool(int64_t, int64_t, const std::vector<float> &, int64_t,
                       std::vector<float> &, std::string *)>
        vocoder;
};

struct MM3WindowOrchestration {
    std::vector<int64_t> starts;
    std::vector<int64_t> frame_lengths;
    std::vector<int64_t> latent_lengths;
    std::vector<int64_t> overlaps;
    std::vector<int64_t> carry_starts;
    std::vector<int64_t> carry_ends;
    std::vector<int64_t> forced_noise;
    std::vector<std::vector<float>> latents;
    std::vector<float> audio;
    int64_t vocoder_calls = 0;
    int64_t samples_per_channel = 0;
    tts_cpp::minimax::detail::AudioMetrics metrics;
};

struct MM3WindowCarryState {
    std::vector<float> latents;
    std::vector<float> condition;
    int64_t length = 0;
};

static bool mm3_window_fail(std::string * error, const std::string & message) {
    if (error) {
        *error = message;
    }
    return false;
}

static bool mm3_validate_window_dimensions(const MM3WindowDimensions & dimensions,
                                           std::string * error) {
    if (dimensions.latent_channels <= 0 || dimensions.audio_channels <= 0 ||
        dimensions.condition_dimension <= 0 ||
        dimensions.upsample <= 0 || dimensions.window_frames <= 0 ||
        dimensions.hop_frames <= 0 || dimensions.hop_frames > dimensions.window_frames ||
        dimensions.carry_span < 0 || dimensions.overlap < 0 ||
        dimensions.overlap > dimensions.carry_span || dimensions.crop_left < 0 ||
        dimensions.crop_right < 0 || dimensions.flow_steps <= 0) {
        return mm3_window_fail(error, "window orchestration dimensions are invalid");
    }
    return true;
}

static bool mm3_emit_window_progress(const MM3WindowOperations & operations,
                                     const std::string & stage, int64_t window,
                                     int64_t window_count, int64_t current, int64_t total,
                                     std::string * error) {
    return !operations.progress ||
           operations.progress(stage, window, window_count, current, total, error);
}

static bool mm3_continue_window_generation(
    const MM3WindowOperations & operations, std::string * error) {
    return !operations.continue_generation ||
           operations.continue_generation(error);
}

static int64_t mm3_copy_condition_carry(const MM3WindowDimensions & dimensions,
                                        const MM3WindowCarryState & carry,
                                        int64_t latent_length,
                                        std::vector<float> & condition) {
    const int64_t overlap = std::min(carry.length, latent_length);
    if (overlap > 0) {
        memcpy(condition.data(), carry.condition.data(),
               static_cast<size_t>(overlap * dimensions.condition_dimension) *
                   sizeof(float));
    }
    return overlap;
}

static bool mm3_update_carry(const MM3WindowDimensions & dimensions,
                             const std::vector<float> & latents,
                             const std::vector<float> & condition, int64_t latent_length,
                             MM3WindowCarryState & carry,
                             MM3WindowOrchestration & orchestration,
                             std::string * error) {
    const auto range = tts_cpp::minimax::detail::carry_range(
        latent_length, dimensions.carry_span, dimensions.overlap);
    orchestration.carry_starts.push_back(range.start);
    orchestration.carry_ends.push_back(range.end);
    carry.length = range.end - range.start;
    if (!tts_cpp::minimax::detail::copy_carry_layout(
            latents, condition, dimensions.latent_channels,
            dimensions.condition_dimension,
            latent_length, range, carry.latents, carry.condition)) {
        return mm3_window_fail(error, "window carry layout is invalid");
    }
    return true;
}

static bool mm3_generate_orchestration_window(
    int64_t frames, int64_t window, const MM3WindowDimensions & dimensions,
    const MM3WindowOperations & operations, MM3WindowCarryState & carry,
    MM3WindowOrchestration & orchestration, std::string * error) {
    const int64_t window_count = static_cast<int64_t>(orchestration.starts.size());
    const int64_t start = orchestration.starts[static_cast<size_t>(window)];
    const int64_t frame_length =
        std::min(start + dimensions.window_frames, frames) - start;
    orchestration.frame_lengths.push_back(frame_length);
    if (!mm3_emit_window_progress(operations, "cond", window, window_count, 0, 1,
                                  error)) {
        return false;
    }
    std::vector<float> condition;
    int64_t latent_length = 0;
    if (!operations.condition ||
        !operations.condition(window, window_count, start, frame_length, condition,
                              latent_length, error)) {
        return false;
    }
    if (latent_length <= 0 ||
        condition.size() !=
            static_cast<size_t>(latent_length * dimensions.condition_dimension)) {
        return mm3_window_fail(error, "condition operation returned an invalid layout");
    }
    orchestration.latent_lengths.push_back(latent_length);
    const int64_t overlap =
        mm3_copy_condition_carry(dimensions, carry, latent_length, condition);
    orchestration.overlaps.push_back(overlap);
    std::vector<float> noise;
    const int64_t noise_count = dimensions.latent_channels * latent_length;
    const bool forced_noise =
        operations.noise && operations.noise(window, noise_count, noise);
    orchestration.forced_noise.push_back(forced_noise ? 1 : 0);
    if (noise.size() != static_cast<size_t>(noise_count)) {
        return mm3_window_fail(error, "noise operation returned an invalid layout");
    }
    if (!mm3_emit_window_progress(operations, "flow", window, window_count, 0,
                                  dimensions.flow_steps, error)) {
        return false;
    }
    std::vector<float> latents;
    const auto on_step = [&operations, window, window_count](
                             int64_t current, int64_t total) {
        if (operations.progress) {
            operations.progress("flow", window, window_count, current, total, nullptr);
        }
    };
    if (!operations.flow ||
        !operations.flow(window, window_count, latent_length, noise, carry.latents,
                         carry.length, condition, latents, on_step, error)) {
        return false;
    }
    if (latents.size() != static_cast<size_t>(noise_count)) {
        return mm3_window_fail(error, "flow operation returned an invalid layout");
    }
    if (!mm3_update_carry(dimensions, latents, condition, latent_length, carry,
                          orchestration, error)) {
        return false;
    }
    orchestration.latents.push_back(std::move(latents));
    return true;
}

static bool mm3_generate_orchestration_windows(
    int64_t frames, const MM3WindowDimensions & dimensions,
    const MM3WindowOperations & operations, MM3WindowOrchestration & orchestration,
    std::string * error) {
    MM3WindowCarryState carry;
    const int64_t window_count = static_cast<int64_t>(orchestration.starts.size());
    for (int64_t window = 0; window < window_count; ++window) {
        if (!mm3_generate_orchestration_window(frames, window, dimensions, operations,
                                               carry, orchestration, error)) {
            return false;
        }
    }
    return true;
}

static bool mm3_vocode_orchestration_windows(
    const MM3WindowOperations & operations, MM3WindowOrchestration & orchestration,
    std::vector<std::vector<float>> & waveforms, std::string * error) {
    const int64_t window_count = static_cast<int64_t>(orchestration.latents.size());
    waveforms.resize(static_cast<size_t>(window_count));
    for (int64_t window = 0; window < window_count; ++window) {
        if (!mm3_emit_window_progress(operations, "vocode", window, window_count, 0, 1,
                                      error)) {
            return false;
        }
        if (!operations.vocoder ||
            !operations.vocoder(window, window_count,
                                orchestration.latents[static_cast<size_t>(window)],
                                orchestration.latent_lengths[static_cast<size_t>(window)],
                                waveforms[static_cast<size_t>(window)], error)) {
            return false;
        }
        ++orchestration.vocoder_calls;
    }
    return true;
}

static tts_cpp::minimax::detail::CropSpan mm3_window_crop_span(
    const MM3WindowDimensions & dimensions, int64_t latent_length, int64_t window,
    int64_t window_count) {
    const int64_t left =
        window == 0 ? 0 : dimensions.crop_left * dimensions.upsample;
    const int64_t right =
        window == window_count - 1 ? 0 : dimensions.crop_right * dimensions.upsample;
    return {left, std::max<int64_t>(
                      latent_length * dimensions.upsample - left - right, 0)};
}

static bool mm3_calculate_orchestration_spans(
    const MM3WindowDimensions & dimensions, const MM3WindowOperations & operations,
    const MM3WindowOrchestration & orchestration,
    std::vector<tts_cpp::minimax::detail::CropSpan> & spans, int64_t & total,
    std::string * error) {
    const int64_t window_count =
        static_cast<int64_t>(orchestration.latent_lengths.size());
    spans.reserve(static_cast<size_t>(window_count));
    total = 0;
    for (int64_t window = 0; window < window_count; ++window) {
        if (!mm3_continue_window_generation(operations, error)) {
            return false;
        }
        const auto span = mm3_window_crop_span(
            dimensions, orchestration.latent_lengths[static_cast<size_t>(window)],
            window, window_count);
        if (span.length > std::numeric_limits<int64_t>::max() - total) {
            return mm3_window_fail(error, "stitched waveform length overflows int64");
        }
        spans.push_back(span);
        total += span.length;
    }
    return true;
}

static bool mm3_copy_orchestration_waveforms(
    const MM3WindowDimensions & dimensions, const MM3WindowOperations & operations,
    const std::vector<std::vector<float>> & waveforms,
    const std::vector<tts_cpp::minimax::detail::CropSpan> & spans, int64_t total,
    MM3WindowOrchestration & orchestration, std::string * error) {
    const int64_t window_count = static_cast<int64_t>(waveforms.size());
    int64_t destination_offset = 0;
    for (int64_t window = 0; window < window_count; ++window) {
        if (!mm3_continue_window_generation(operations, error)) {
            return false;
        }
        const int64_t source_length =
            orchestration.latent_lengths[static_cast<size_t>(window)] *
            dimensions.upsample;
        const auto span = spans[static_cast<size_t>(window)];
        if (!tts_cpp::minimax::detail::copy_planar_window(
                waveforms[static_cast<size_t>(window)], dimensions.audio_channels,
                source_length, span.left, total, destination_offset, span.length,
                orchestration.audio)) {
            return mm3_window_fail(error, "planar waveform copy range is invalid");
        }
        destination_offset += span.length;
    }
    return true;
}

static bool mm3_finalize_orchestration_audio(
    const MM3WindowOperations & operations, MM3WindowOrchestration & orchestration,
    std::string * error) {
    if (!mm3_continue_window_generation(operations, error)) {
        return false;
    }
    size_t nonfinite_index = 0;
    if (!tts_cpp::minimax::detail::finalize_audio(
            orchestration.audio, orchestration.metrics, nonfinite_index)) {
        return mm3_window_fail(error, "non-finite audio sample at index " +
                                          std::to_string(nonfinite_index));
    }
    return mm3_continue_window_generation(operations, error);
}

static bool mm3_stitch_orchestration_windows(
    const MM3WindowDimensions & dimensions, const MM3WindowOperations & operations,
    const std::vector<std::vector<float>> & waveforms,
    MM3WindowOrchestration & orchestration, std::string * error) {
    const int64_t window_count = static_cast<int64_t>(waveforms.size());
    if (!mm3_emit_window_progress(operations, "stitch", -1, window_count, 0, 1,
                                  error)) {
        return false;
    }
    std::vector<tts_cpp::minimax::detail::CropSpan> spans;
    int64_t total = 0;
    if (!mm3_calculate_orchestration_spans(
            dimensions, operations, orchestration, spans, total, error)) {
        return false;
    }
    if (total < 0 ||
        static_cast<uint64_t>(total) >
            static_cast<uint64_t>(std::numeric_limits<size_t>::max() /
                                  dimensions.audio_channels)) {
        return mm3_window_fail(error, "stitched waveform size is invalid");
    }
    orchestration.audio.assign(
        static_cast<size_t>(total * dimensions.audio_channels), 0.0f);
    if (!mm3_copy_orchestration_waveforms(
            dimensions, operations, waveforms, spans, total, orchestration, error)) {
        return false;
    }
    orchestration.samples_per_channel = total;
    return mm3_finalize_orchestration_audio(operations, orchestration, error);
}

static bool mm3_orchestrate_windows(int64_t frames,
                                    const MM3WindowDimensions & dimensions,
                                    const MM3WindowOperations & operations,
                                    MM3WindowOrchestration * orchestration,
                                    std::string * error) {
    if (!orchestration || frames <= 0 ||
        !mm3_validate_window_dimensions(dimensions, error)) {
        return false;
    }
    *orchestration = MM3WindowOrchestration{};
    orchestration->starts = tts_cpp::minimax::detail::window_starts(
        frames, dimensions.window_frames, dimensions.hop_frames);
    if (!mm3_generate_orchestration_windows(frames, dimensions, operations,
                                            *orchestration, error)) {
        return false;
    }
    std::vector<std::vector<float>> waveforms;
    if (!mm3_vocode_orchestration_windows(operations, *orchestration, waveforms,
                                          error) ||
        !mm3_stitch_orchestration_windows(dimensions, operations, waveforms,
                                          *orchestration, error)) {
        return false;
    }
    return mm3_emit_window_progress(
        operations, "done", -1, static_cast<int64_t>(orchestration->starts.size()),
        static_cast<int64_t>(orchestration->starts.size()),
        static_cast<int64_t>(orchestration->starts.size()), error);
}
