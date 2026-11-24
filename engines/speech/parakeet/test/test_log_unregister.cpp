// The host log sink is never called after the host replaced or cleared it.
// Background work (a Core ML sidecar load finishing after its engine was
// destroyed) logs from another thread; parakeet_log_set must return only
// after every callback still running on the old sink has returned, so the
// host may free that sink's user data at once.
//
//   drain        a callback in flight when the sink is cleared finishes
//                before parakeet_log_set returns
//   reentrant    a callback that logs through parakeet, or clears the sink
//                itself, does not deadlock
//   ggml         ggml's messages reach the sink through parakeet's forwarder
//                (never the host callback directly), and are drained alike
//   concurrent   two callbacks on different threads that both clear the sink
//                finish (neither waits for the other)
//
// Hermetic; exit 0 = pass, 1 = fail.

#include "parakeet/log.h"
#include "parakeet_log.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <thread>

namespace {

int g_failures = 0;

void expect(bool ok, const char * what) {
    std::printf("[%s] %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++g_failures;
}

struct Sink {
    std::atomic<bool> entered{false};
    std::atomic<bool> freed{false};        // the host released its user data
    std::atomic<bool> used_after_free{false};
};

void slow_callback(ggml_log_level, const char *, void * user) {
    auto * sink = static_cast<Sink *>(user);
    sink->entered = true;
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    if (sink->freed) sink->used_after_free = true;
}

void case_drain() {
    Sink sink;
    parakeet_log_set(&slow_callback, &sink);
    std::thread worker([] { PARAKEET_LOG_INFO("a background load finished\n"); });
    while (!sink.entered) std::this_thread::yield();
    parakeet_log_set(nullptr, nullptr);
    sink.freed = true;  // what a host does right after clearing its sink
    worker.join();
    expect(!sink.used_after_free, "drain: the sink's user data is not used after parakeet_log_set returns");
}

std::atomic<int> g_nested{0};

void reentrant_callback(ggml_log_level, const char * text, void *) {
    if (g_nested++ == 0) {
        PARAKEET_LOG_INFO("nested: %s", text);
        parakeet_log_set(nullptr, nullptr);
    }
}

void case_reentrant() {
    parakeet_log_set(&reentrant_callback, nullptr);
    PARAKEET_LOG_INFO("outer\n");
    expect(g_nested.load() == 2, "reentrant: a callback may log and clear the sink without deadlock");
    parakeet_log_set(nullptr, nullptr);
}

void case_ggml() {
    Sink sink;
    parakeet_log_set(&slow_callback, &sink);
    ggml_log_callback forwarder = nullptr;
    void * forwarder_data = nullptr;
    ggml_log_get(&forwarder, &forwarder_data);
    expect(forwarder != nullptr && forwarder != &slow_callback,
           "ggml: ggml calls parakeet's forwarder, not the host callback");
    std::thread worker([&] {
        if (forwarder) forwarder(GGML_LOG_LEVEL_INFO, "ggml message\n", forwarder_data);
    });
    while (!sink.entered && forwarder) std::this_thread::yield();
    parakeet_log_set(nullptr, nullptr);
    sink.freed = true;
    worker.join();
    expect(!sink.used_after_free, "ggml: a ggml callback in flight is drained before parakeet_log_set returns");
}

std::atomic<int> g_entered{0};

void clearing_callback(ggml_log_level, const char *, void *) {
    ++g_entered;
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (g_entered.load() < 2 && std::chrono::steady_clock::now() < until) std::this_thread::yield();
    parakeet_log_set(nullptr, nullptr);
}

void case_concurrent() {
    parakeet_log_set(&clearing_callback, nullptr);
    auto both = std::async(std::launch::async, [] {
        std::thread a([] { PARAKEET_LOG_INFO("a\n"); });
        std::thread b([] { PARAKEET_LOG_INFO("b\n"); });
        a.join();
        b.join();
    });
    const bool finished = both.wait_for(std::chrono::seconds(10)) == std::future_status::ready;
    expect(finished && g_entered.load() == 2,
           "concurrent: two callbacks that both clear the sink finish without waiting on each other");
    if (!finished) {
        std::printf("test-log-unregister: deadlocked\n");
        std::fflush(stdout);
        std::_Exit(1);
    }
}

}  // namespace

int main() {
    case_drain();
    case_reentrant();
    case_ggml();
    case_concurrent();
    std::printf("test-log-unregister: %d failure(s)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
