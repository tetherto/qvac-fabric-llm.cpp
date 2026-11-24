#pragma once

#include "parakeet_ctc.h"
#include "parakeet_tdt.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace parakeet {

struct UnifiedStreamState {
    struct Impl;
    std::unique_ptr<Impl> impl;

    std::vector<float> cache_channel;
    std::vector<float> cache_time;
    std::vector<int32_t> token_ids;

    int cache_length = 0;
    int chunk_frames = -1;
    int right_context_frames = -1;
    int step_index = 0;
    int64_t emitted_encoder_frames = 0;
    bool cancelled = false;
    bool finalized = false;

    UnifiedStreamState();
    ~UnifiedStreamState();
    UnifiedStreamState(UnifiedStreamState &&) noexcept;
    UnifiedStreamState & operator=(UnifiedStreamState &&) noexcept;
    UnifiedStreamState(const UnifiedStreamState &) = delete;
    UnifiedStreamState & operator=(const UnifiedStreamState &) = delete;
};

struct UnifiedStreamStepResult {
    std::vector<float> encoder_committed;
    std::vector<int32_t> new_token_ids;
    std::string text;
    int committed_frames = 0;
    int provisional_frames = 0;
    int decoder_steps = 0;
};

bool unified_operating_point_supported(
    const ParakeetCtcModel & model,
    int chunk_frames,
    int right_context_frames);

int init_unified_stream_state(
    const ParakeetCtcModel & model,
    int chunk_frames,
    int right_context_frames,
    UnifiedStreamState & state);

void reset_unified_stream(UnifiedStreamState & state);
void cancel_unified_stream(UnifiedStreamState & state);
int unified_pending_mel_frames(const UnifiedStreamState & state);

int append_unified_pcm(
    const ParakeetCtcModel & model,
    UnifiedStreamState & state,
    const float * samples,
    int n_samples,
    bool finalize);

int append_unified_mel_frames(
    UnifiedStreamState & state,
    const float * mel,
    int n_frames,
    int n_mels);

int next_unified_processed_signal(
    UnifiedStreamState & state,
    int n_mels,
    bool finalize,
    std::vector<float> & processed_signal,
    int & n_frames);

int run_unified_stream_step(
    ParakeetCtcModel & model,
    TdtRuntimeWeights & runtime,
    const float * processed_signal,
    int n_mel_frames,
    int n_mels,
    bool finalize,
    UnifiedStreamState & state,
    UnifiedStreamStepResult & result);

}
