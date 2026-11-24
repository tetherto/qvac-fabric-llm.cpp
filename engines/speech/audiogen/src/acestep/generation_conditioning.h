#pragma once

#include "audiogen-cpp/acestep/engine.h"
#include "generation_plan.h"

#include <functional>
#include <vector>

namespace tts_cpp::acestep {

struct EncodedAudio {
    std::vector<float> latent;
    int                frames = 0;
};

struct GenerationConditioning {
    EncodedAudio source;
    EncodedAudio reference;
};

struct TimbreInput {
    const float * data = nullptr;
    int           frames = 0;
};

using AudioEncoder = std::function<bool(const std::vector<float> &, const char *, EncodedAudio &)>;

inline bool prepare_generation_conditioning(const GenerateParams & params, const GenerationPlan & plan,
                                            const AudioEncoder & encode, GenerationConditioning & conditioning) {
    if (plan.encode_source && !encode(params.source_audio, "source", conditioning.source)) return false;
    if (plan.encode_reference && !encode(params.reference_audio, "reference", conditioning.reference)) return false;
    return true;
}

inline TimbreInput resolve_timbre_input(const GenerationPlan & plan, const EncodedAudio & reference,
                                        const std::vector<float> & source, int source_frames,
                                        const std::vector<float> & silence) {
    if (plan.reuse_source_reference) return { source.data(), source_frames };
    if (!reference.latent.empty()) return { reference.latent.data(), reference.frames };
    if (!silence.empty()) return { silence.data(), 1 };
    return {};
}

} // namespace tts_cpp::acestep
