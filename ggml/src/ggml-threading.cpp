#include "ggml-threading.h"
#include <cstdlib>
#include <mutex>

std::mutex ggml_critical_section_mutex;

void ggml_critical_section_start() {
    ggml_critical_section_mutex.lock();
}

void ggml_critical_section_end(void) {
    ggml_critical_section_mutex.unlock();
}

static int read_tbq_norm_correction_flag() {
    const char * env = getenv("GGML_TQ_NORM_CORRECTION");
    return (env && env[0] == '1') ? 1 : 0;
}

int ggml_tbq_norm_correction_enabled(void) {
    static const int enabled = read_tbq_norm_correction_flag();
    return enabled;
}
