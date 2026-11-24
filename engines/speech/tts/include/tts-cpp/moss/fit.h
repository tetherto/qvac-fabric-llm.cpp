#pragma once

#include "tts-cpp/fit.h"
#include "tts-cpp/moss/engine.h"

namespace tts_cpp::moss {

struct FitWorkload {
    int prompt_rows;
    int64_t reference_samples;
    bool streaming;
    uint64_t margin_bytes;

    FitWorkload(int prompt, int64_t reference, bool stream, uint64_t margin)
        : prompt_rows(prompt), reference_samples(reference), streaming(stream), margin_bytes(margin) {}
};

TTS_CPP_API FitResult fit_params(const EngineOptions & options, const FitWorkload & workload);

}
