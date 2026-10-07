#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

GGML_API void ggml_critical_section_start(void);
GGML_API void ggml_critical_section_end(void);

// Reads GGML_TQ_NORM_CORRECTION once, safely from any thread.
GGML_API int ggml_tbq_norm_correction_enabled(void);

#ifdef __cplusplus
}
#endif
