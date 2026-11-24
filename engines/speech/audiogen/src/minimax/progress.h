#pragma once

#include "logic.h"

#include <cstdint>
#include <functional>
#include <string>

inline constexpr const char * MM3_ERR_CANCELLED = "cancelled";

struct MM3GenProgress {
    const char * stage = "";
    int64_t window = -1;
    int64_t n_windows = 0;
    int64_t step = 0;
    int64_t n_steps = 0;
};

using MM3ProgressCb = std::function<void(const MM3GenProgress &)>;

static bool mm3_continue_generation(const std::function<bool()> & should_cancel, std::string * error) {
    if (!tts_cpp::minimax::detail::cancellation_requested(should_cancel)) {
        return true;
    }
    if (error) {
        *error = MM3_ERR_CANCELLED;
    }
    return false;
}

static bool mm3_emit_progress(const MM3ProgressCb & progress, const MM3GenProgress & state,
                              const std::function<bool()> & should_cancel, std::string * error) {
    if (progress) {
        progress(state);
    }
    return mm3_continue_generation(should_cancel, error);
}
