#include "coreml_compile_wait.h"

#include <cstdio>
#include <stdexcept>

namespace {
using namespace std::chrono_literals;
using parakeet::test::CompileWaitCallbacks;
using parakeet::test::CompileWaitState;
using parakeet::test::wait_for_compile_entry;

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

struct FakePreparation {
    std::chrono::steady_clock::time_point clock{};
    std::chrono::milliseconds elapsed{0};
    std::chrono::milliseconds preparation_time{70000};
    std::string state = "{\"state\":\"preparing\"}";
    int polls = 0;

    CompileWaitCallbacks callbacks() {
        return {
            [this] { return elapsed >= preparation_time; },
            [this] { ++polls; return state; },
            [this] { return clock; },
            [this](std::chrono::milliseconds delay) { clock += delay; elapsed += delay; },
        };
    }
};

void test_prepare_can_exceed_one_minute() {
    FakePreparation preparation;
    const auto result = wait_for_compile_entry(preparation.callbacks());
    require(result.state == CompileWaitState::entered, "Valid preparation above sixty seconds timed out");
    require(preparation.elapsed == preparation.preparation_time, "Compile entry was not awaited");
}

void test_failure_keeps_diagnostics() {
    FakePreparation preparation;
    preparation.state = "{\"state\":\"failed\",\"error\":\"generation failed\"}";
    const auto result = wait_for_compile_entry(preparation.callbacks());
    require(result.state == CompileWaitState::failed && result.status == preparation.state,
            "Preparation failure lost its status diagnostic");
    require(preparation.elapsed == 0ms && preparation.polls == 1, "A failed job waited for the timeout");
}

void test_timeout_keeps_last_status() {
    FakePreparation preparation;
    const auto result = wait_for_compile_entry(preparation.callbacks(), 50ms, 10ms);
    require(result.state == CompileWaitState::timed_out && result.status == preparation.state,
            "Compile timeout lost the last preparation status");
    require(preparation.elapsed == 50ms, "Compile wait exceeded its bounded timeout");
}

void test_ready_without_compile_does_not_satisfy_hook() {
    FakePreparation preparation;
    preparation.state = "{\"state\":\"ready\"}";
    const auto result = wait_for_compile_entry(preparation.callbacks());
    require(result.state == CompileWaitState::failed, "A ready cache bypassed the compile-entry assertion");
    require(preparation.elapsed == 0ms, "An unexpected ready result waited for the timeout");
}

void test_immediate_compile_entry() {
    FakePreparation preparation;
    preparation.preparation_time = 0ms;
    const auto result = wait_for_compile_entry(preparation.callbacks());
    require(result.state == CompileWaitState::entered && preparation.polls == 0,
            "An entered compile hook still polled preparation status");
}

template<class Action>
void require_invalid_wait(Action action) {
    try {
        action();
    } catch (const std::invalid_argument &) {
        return;
    }
    throw std::runtime_error("Invalid compile wait bounds were accepted");
}

void test_invalid_bounds() {
    FakePreparation preparation;
    require_invalid_wait([&] { wait_for_compile_entry(preparation.callbacks(), -1ms, 10ms); });
    require_invalid_wait([&] { wait_for_compile_entry(preparation.callbacks(), 10ms, 0ms); });
    require_invalid_wait([&] { wait_for_compile_entry(preparation.callbacks(), 10ms, -1ms); });
}
} // namespace

int main() {
    try {
        test_prepare_can_exceed_one_minute();
        test_failure_keeps_diagnostics();
        test_timeout_keeps_last_status();
        test_ready_without_compile_does_not_satisfy_hook();
        test_immediate_compile_entry();
        test_invalid_bounds();
        std::puts("Core ML compile-entry wait regressions passed");
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
