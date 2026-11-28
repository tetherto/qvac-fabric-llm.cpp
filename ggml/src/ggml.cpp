#include "ggml-impl.h"

#include <cstdlib>
#include <exception>

static thread_local ggml_log_level g_log_min_level = GGML_LOG_LEVEL_NONE;

ggml_log_level ggml_log_get_thread_min_level(void) {
    return g_log_min_level;
}

ggml_log_level ggml_log_set_thread_min_level(ggml_log_level level) {
    const auto previous = g_log_min_level;
    g_log_min_level = level;
    return previous;
}

static std::terminate_handler previous_terminate_handler;

GGML_NORETURN static void ggml_uncaught_exception() {
    ggml_print_backtrace();
    if (previous_terminate_handler) {
        previous_terminate_handler();
    }
    abort(); // unreachable unless previous_terminate_handler was nullptr
}

static bool ggml_uncaught_exception_init = []{
    const char * GGML_NO_BACKTRACE = getenv("GGML_NO_BACKTRACE");
    if (GGML_NO_BACKTRACE) {
        return false;
    }
    const auto prev{std::get_terminate()};
    GGML_ASSERT(prev != ggml_uncaught_exception);
    previous_terminate_handler = prev;
    std::set_terminate(ggml_uncaught_exception);
    return true;
}();
