#include "ggml-impl.h"

// Keep logging references separate from ggml.cpp's terminate-handler initializer.
static thread_local ggml_log_level g_log_min_level = GGML_LOG_LEVEL_NONE;

ggml_log_level ggml_log_get_thread_min_level(void) {
    return g_log_min_level;
}

ggml_log_level ggml_log_set_thread_min_level(ggml_log_level level) {
    const auto previous = g_log_min_level;
    g_log_min_level = level;
    return previous;
}
