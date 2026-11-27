#pragma once

#include "parakeet/fit.h"
#include "parakeet/moss_transcribe.h"

namespace parakeet::moss {

// An active Core ML encoder sidecar returns Error with reason
// "coreml-memory-estimate-unavailable".

PARAKEET_API FitResult fit_params(const TranscribeOptions & options, const TranscribeRequest & request,
                                 double audio_seconds, uint64_t margin_bytes);

}
