#pragma once

#include "audiogen-cpp/acestep/fit.h"
#include "audiogen-cpp/export.h"
#include "audiogen-cpp/minimax/engine.h"

#include <cstdint>

namespace tts_cpp::minimax {

using FitStatus          = acestep::FitStatus;
using FitStageProjection = acestep::FitStageProjection;
using FitDevicePool      = acestep::FitDevicePool;
using FitResult          = acestep::FitResult;

inline constexpr int64_t  kFitDefaultMaxFrames    = 300;
inline constexpr int64_t  kFitDefaultPromptTokens = 1024;
inline constexpr uint64_t kFitDefaultMarginBytes  = 256ull * 1024 * 1024;

struct FitWorkload {
    int64_t  max_frames    = kFitDefaultMaxFrames;
    int64_t  prompt_tokens = kFitDefaultPromptTokens;
    uint64_t margin_bytes  = kFitDefaultMarginBytes;
};

AUDIOGEN_API FitResult fit_params(const EngineOptions & options, const FitWorkload & workload = {});

}
