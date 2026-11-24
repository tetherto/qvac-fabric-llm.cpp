#pragma once

#include "encoder_route.h"
#include "long_form.h"
#include "parakeet/engine.h"
#include "parakeet_ctc.h"

#include <atomic>

namespace parakeet {

struct WindowedEncoderStats {
    double encoder_ms = 0.0;
    int coreml_windows = 0;
    int ggml_windows = 0;
    EncoderBackendTally tally;  // per encoded window, with the program that ran it
};

// The long-form plan for one unit of work. The Core ML fixed-shape plan is
// used only when `route` holds a handle; with a null route the plan is the
// ggml-only plan, so ggml transcripts never change because a sidecar exists.
LongFormPlan resolve_long_form_plan(const ParakeetCtcModel & model,
                                    const EngineOptions & opts,
                                    int n_mel_frames,
                                    const EncoderRoute & route,
                                    bool preserve_eu_context = false);

int run_encoder_windowed(ParakeetCtcModel & model,
                         const float * mel, int n_mel_frames, int n_mels,
                         const LongFormPlan & plan,
                         const EncoderRoute & route,
                         std::atomic<bool> & cancel_flag,
                         EncoderOutputs & out,
                         WindowedEncoderStats & stats, bool prefer_precise_eu = false);

}  // namespace parakeet
