// PARAKEET_FORWARD_GGML_LOG decides who owns the process-wide ggml logger: with it
// the parakeet log callback takes the ggml log over, without it the host's logger stays.
#include "parakeet_log.h"

#include "ggml.h"

#include <cstdio>

namespace {

void host_logger(ggml_log_level, const char *, void *) {}
void engine_sink(ggml_log_level, const char *, void *) {}

ggml_log_callback installed_ggml_logger() {
    ggml_log_callback callback = nullptr;
    void * user_data = nullptr;
    ggml_log_get(&callback, &user_data);
    return callback;
}

bool expect(bool ok, const char * what) {
    std::printf("[ggml-log-ownership] %s: %s\n", what, ok ? "ok" : "FAIL");
    return ok;
}

} // namespace

int main() {
    ggml_log_set(&host_logger, nullptr);
    parakeet::log_set_callback(&engine_sink, nullptr);
    const bool host_kept = installed_ggml_logger() == &host_logger;
#ifdef PARAKEET_FORWARD_GGML_LOG
    const bool passed = expect(!host_kept, "the parakeet callback takes over the ggml log");
#else
    const bool passed = expect(host_kept, "the host keeps the ggml log");
#endif
    parakeet::log_set_callback(nullptr, nullptr);
    std::printf("[ggml-log-ownership] %s\n", passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}
