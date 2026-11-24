#pragma once

#include "parakeet/fit.h"
#include "parakeet/moss_transcribe.h"

namespace parakeet::moss {

PARAKEET_API FitResult fit_params(const TranscribeOptions & options, const TranscribeRequest & request,
                                 double audio_seconds, uint64_t margin_bytes);

}
