#pragma once

// The identity of the GGUF parler_load_gguf read the weights from (see
// parler_dac_source_record in dac_coreml.h). Core ML builds only: the stat
// tuple is the shared runtime's.

#ifdef TTS_CPP_USE_COREML
#include "qvac_coreml/slot.h"

namespace tts_cpp {
namespace parler {
namespace detail {

struct parler_dac_source {
    QVAC_COREML_NS::SourceStat stat;  // fstat of the descriptor the load read through
    bool ok = false;                  // false: no descriptor or no fstat
};

} // namespace detail
} // namespace parler
} // namespace tts_cpp
#endif
