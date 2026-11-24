#pragma once

#include "encoder_route.h"

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace parakeet {

struct ParakeetCtcModel;
struct TdtRuntimeWeights;
struct TdtDecodeResult;

inline bool allows_independent_eu_decoder(int requested_window, int requested_context) {
    return requested_window == 0 && requested_context == 0;
}

inline bool needs_precise_eu_continuity(bool full_eu, int requested_window,
                                        int requested_context, int frames, int fixed_capacity) {
    return full_eu && allows_independent_eu_decoder(requested_window, requested_context) &&
        frames > fixed_capacity && fixed_capacity > 0;
}

inline bool should_batch_eu_decoder(size_t segments, const std::string & backend,
                                     bool mobile, bool forced) {
    if (forced) return true;
    constexpr size_t kDesktopCpuMinimumSegments = 4;
    constexpr size_t kMobileAneMinimumSegments = 8;
    return mobile ? backend == "coreml-ane" && segments >= kMobileAneMinimumSegments
                  : backend == "coreml-cpu" && segments >= kDesktopCpuMinimumSegments;
}

struct IndependentSegmentsOutcome {
    std::vector<int32_t> ids;
    std::string text;
    int encoder_frames = 0;
    bool used_coreml = false;
    bool decoder_used_coreml = false;
    EncoderBackendTally tally;  // per-segment encoder backends (the result label)
    double encoder_ms = 0.0, decode_ms = 0.0, segment_plan_ms = 0.0, assemble_ms = 0.0;
};

#ifdef PARAKEET_USE_COREML
// Shared by Engine and CLI so their full-EU encoder, segment planner and
// batched decoder execute the same path. False retains the normal context plan.
// `route` is the call's encoder route: it caps the segments and encodes them.
bool transcribe_independent_segments(ParakeetCtcModel & model, TdtRuntimeWeights & runtime,
                                     const EncoderRoute & route,
                                     const float * mel, int frames, int bins, bool verbose,
                                     std::atomic<bool> & cancel,
                                     IndependentSegmentsOutcome & result);

// A forced Core ML decoder can process one complete encoder sequence without
// cutting it into segments, preserving explicit encoder windows and context.
bool decode_coreml_tdt_sequence(const ParakeetCtcModel & model, TdtRuntimeWeights & runtime,
                                 const float * encoder, int frames, int dim,
                                 TdtDecodeResult & result);
#endif

}  // namespace parakeet
