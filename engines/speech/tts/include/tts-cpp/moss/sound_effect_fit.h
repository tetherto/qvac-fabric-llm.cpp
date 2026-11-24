#pragma once
#include "tts-cpp/fit.h"
#include "tts-cpp/moss/sound_effect.h"

namespace tts_cpp::moss {

TTS_CPP_API FitResult fit_params(const SoundEffectOptions & options,
                                const SoundEffectRequest & request, uint64_t margin_bytes);

}
