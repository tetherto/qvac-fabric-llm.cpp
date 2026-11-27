// QVAC patch, see PATCHES.md.
// WHISPER_FORWARD_GGML_LOG decides who owns the process-wide ggml logger: with it
// whisper_log_set takes the ggml log over, without it the host's logger stays.
#include "whisper.h"

#include "ggml.h"

#include <cstdio>

static void host_logger(ggml_log_level, const char *, void *) {}
static void whisper_sink(ggml_log_level, const char *, void *) {}

static ggml_log_callback installed_ggml_logger() {
    ggml_log_callback callback = nullptr;
    void * user_data = nullptr;
    ggml_log_get(&callback, &user_data);
    return callback;
}

int main() {
    ggml_log_set(&host_logger, nullptr);
    whisper_log_set(&whisper_sink, nullptr);
    const bool host_kept = installed_ggml_logger() == &host_logger;
#ifdef WHISPER_FORWARD_GGML_LOG
    const bool passed = !host_kept;
#else
    const bool passed = host_kept;
#endif
    std::printf("[whisper-ggml-log-ownership] host logger kept=%d: %s\n", (int) host_kept, passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}
