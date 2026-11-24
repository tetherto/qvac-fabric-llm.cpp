#pragma once

#include "audiogen-cpp/acestep/engine.h"
#include "generate_task.h"

namespace tts_cpp::acestep {

struct GenerationPlan {
    bool encode_source           = false;
    bool encode_reference        = false;
    bool reuse_source_reference  = false;
    bool run_lm                  = true;
    bool run_detokenizer         = true;
    bool blend_cover_noise       = false;
};

inline GenerationPlan make_generation_plan(const GenerateParams & params, const GenerateTask & task) {
    const bool cover_nofsq = task.type == TASK_COVER_NOFSQ;
    const bool lego        = is_lego_task(task.type);
    const bool source_task = cover_nofsq || lego;

    GenerationPlan plan;
    plan.encode_source          = source_task;
    plan.encode_reference       = !params.reference_audio.empty();
    plan.reuse_source_reference = cover_nofsq && params.reference_audio.empty();
    plan.run_lm                 = !source_task;
    plan.run_detokenizer        = !source_task;
    plan.blend_cover_noise      = cover_nofsq && task.cover_noise_strength > 0.0f;
    return plan;
}

} // namespace tts_cpp::acestep
