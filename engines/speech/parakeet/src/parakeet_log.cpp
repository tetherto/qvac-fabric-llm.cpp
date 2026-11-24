// log_impl and log_set_callback; installs hook and forwards to ggml_log_set.

#include "parakeet_log.h"

#include "parakeet/log.h"

#include <condition_variable>
#include <memory>
#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

namespace parakeet {

namespace {

// One host sink registration and the callbacks running on it. Background
// work (a Core ML sidecar load finishing after its engine was destroyed) can
// log from another thread; replacing or clearing the sink waits for every
// callback still running on the registration it replaces, so the host may
// free that sink's user data as soon as parakeet_log_set returns. ggml gets a
// fixed forwarding callback, so its messages are counted the same way.
//
// A replacement waits only for the registration it replaced, and never for
// the calling thread's own callbacks on it: a callback may log or replace the
// sink itself, and two callbacks on different threads that both replace it
// do not wait on each other.
struct Sink {
    ggml_log_callback callback = nullptr;
    void * user_data = nullptr;
    int in_flight = 0;
};

std::mutex g_mutex;
std::condition_variable g_idle;
std::shared_ptr<Sink> g_sink;
thread_local std::vector<const Sink *> t_running;  // sinks this thread is calling into

void dispatch(ggml_log_level level, const char * text) {
    std::shared_ptr<Sink> sink;
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        sink = g_sink;
        if (sink) ++sink->in_flight;
    }
    if (!sink) {
        std::fputs(text, stderr);
        return;
    }
    t_running.push_back(sink.get());
    sink->callback(level, text, sink->user_data);
    t_running.pop_back();
    std::lock_guard<std::mutex> lock(g_mutex);
    --sink->in_flight;
    g_idle.notify_all();
}

void forward_ggml(ggml_log_level level, const char * text, void *) { dispatch(level, text); }

}

void log_set_callback(ggml_log_callback cb, void * user_data) {
    std::shared_ptr<Sink> replaced;
    {
        std::unique_lock<std::mutex> lock(g_mutex);
        replaced = std::move(g_sink);
        if (cb) {
            g_sink = std::make_shared<Sink>();
            g_sink->callback = cb;
            g_sink->user_data = user_data;
        }
        ggml_log_set(cb ? &forward_ggml : nullptr, nullptr);
        if (replaced) {
            const Sink * old = replaced.get();
            g_idle.wait(lock, [&] {
                const int own = (int) std::count(t_running.begin(), t_running.end(), old);
                return replaced->in_flight == own;
            });
        }
    }
}

void log_impl(ggml_log_level level, const char * fmt, ...) {
    char stack_buf[1024];

    va_list args;
    va_start(args, fmt);
    va_list args_copy;
    va_copy(args_copy, args);
    const int n = std::vsnprintf(stack_buf, sizeof(stack_buf), fmt, args);
    va_end(args);

    const char * text = stack_buf;
    std::vector<char> heap_buf;
    if (n >= (int) sizeof(stack_buf)) {
        heap_buf.resize((size_t) n + 1);
        std::vsnprintf(heap_buf.data(), heap_buf.size(), fmt, args_copy);
        text = heap_buf.data();
    }
    va_end(args_copy);

    dispatch(level, text);
}

}

extern "C" PARAKEET_API void parakeet_log_set(ggml_log_callback cb,
                                                        void *            user_data) {
    parakeet::log_set_callback(cb, user_data);
}
