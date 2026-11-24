#pragma once
#include "pocket/mimi.h"
#include <cmath>
#include <cstddef>

namespace tts_cpp::pocket::detail {
struct FrameBudget {
    int eos_search_frames;
    int tail_frames;
    // Includes the final advance that detects completion (not emitted).
    int max_frames() const { return eos_search_frames + tail_frames; }
};
inline FrameBudget frame_budget(size_t tokens, int tail_frames) {
    // Match the upstream text-duration estimate, reserving the requested
    // post-EOS tail separately so a late EOS can still complete.
    const double seconds = tokens/3.0 + 2;
    return {int(std::ceil(seconds*Mimi::sample_rate()/Mimi::frame_samples())), tail_frames};
}
// EOS must arrive within the text estimate. Once detected, allow the entire
// tail even when EOS is on the estimate's last frame.
class FrameProgress {
    FrameBudget budget_;
    int frame_ = 0, eos_ = -1;
    bool complete_ = false;
public:
    explicit FrameProgress(FrameBudget budget) : budget_(budget) {}
    bool has_next() const {
        return !complete_ && frame_ < (eos_ < 0 ? budget_.eos_search_frames : budget_.max_frames());
    }
    bool finish_frame(bool eos) {
        if (eos && eos_ < 0) eos_ = frame_;
        complete_ = eos_ >= 0 && frame_ >= eos_ + budget_.tail_frames;
        ++frame_;
        return complete_;
    }
};
} // namespace tts_cpp::pocket::detail
