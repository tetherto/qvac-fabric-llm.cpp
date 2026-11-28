#pragma once
#include "moss/sfx_model.h"
#include "tts-cpp/moss/sound_effect.h"

namespace tts_cpp::moss::detail {

struct ResolvedRequest {
    std::string conditioning;
    std::string negative_prompt;
    int tenths = 0;
    int steps = 0;
    float guidance = 0.0f;
    float shift = 0.0f;
    uint32_t seed = 0;
};

ResolvedRequest resolve_sfx_request(const SfxConfig & config, const SoundEffectRequest & request);
inline constexpr int SFX_DECODE_WINDOW_FRAMES = 256;

}
