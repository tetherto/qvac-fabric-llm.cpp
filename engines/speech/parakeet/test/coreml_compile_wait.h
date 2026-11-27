#pragma once

#include <chrono>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>

namespace parakeet::test {

// Digesting the GGUF and writing its package can exceed a minute before compile.
inline constexpr auto coreml_compile_wait_timeout = std::chrono::minutes(5);
inline constexpr auto coreml_compile_poll_interval = std::chrono::milliseconds(10);

enum class CompileWaitState { entered, failed, timed_out };

struct CompileWaitResult {
    CompileWaitState state;
    std::string status;
};

struct CompileWaitCallbacks {
    std::function<bool()> entered;
    std::function<std::string()> status;
    std::function<std::chrono::steady_clock::time_point()> now = [] {
        return std::chrono::steady_clock::now();
    };
    std::function<void(std::chrono::milliseconds)> sleep = [](std::chrono::milliseconds delay) {
        std::this_thread::sleep_for(delay);
    };
};

inline bool compile_wait_terminal(const std::string & status) {
    return status.find("\"state\":\"failed\"") != std::string::npos ||
           status.find("\"state\":\"ready\"") != std::string::npos;
}

inline CompileWaitResult wait_for_compile_entry(
        const CompileWaitCallbacks & callbacks,
        std::chrono::milliseconds timeout = coreml_compile_wait_timeout,
        std::chrono::milliseconds poll_interval = coreml_compile_poll_interval) {
    if (timeout < std::chrono::milliseconds::zero() || poll_interval <= std::chrono::milliseconds::zero()) {
        throw std::invalid_argument("Invalid compile wait timeout or poll interval");
    }
    const auto deadline = callbacks.now() + timeout;
    for (;;) {
        if (callbacks.entered()) return {CompileWaitState::entered, {}};
        const auto status = callbacks.status();
        if (compile_wait_terminal(status)) return {CompileWaitState::failed, status};
        if (callbacks.now() >= deadline) return {CompileWaitState::timed_out, status};
        callbacks.sleep(poll_interval);
    }
}

} // namespace parakeet::test
