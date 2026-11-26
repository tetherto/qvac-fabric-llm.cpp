// Hermetic tests for bench-ttr.h (no Core ML). Built and run by
// test_bench_load_time.py; standalone:
//   c++ -std=c++17 -pthread -I scripts/coreml scripts/coreml/test_bench_ttr.cpp -o t && ./t

#include <atomic>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "bench-ttr.h"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::fprintf(stderr, "%s:%d: CHECK(%s)\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                    \
        }                                                                    \
    } while (0)

std::atomic<int> g_live{0};

struct Counted {
    Counted() { ++g_live; }
    ~Counted() { --g_live; }
};

double fake_now() {
    static std::atomic<int> t{0};
    return ++t;
}

// Every model that loads stays alive until the outcome is dropped: the live
// count seen by each load grows by one per earlier load, and after the run it
// equals the number of successful loads.
void manual_retains_every_loaded_model(bool generate, bool gated) {
    const std::vector<bench_ttr::ManualStage> stages = {{"encoder", "enc", {"enc_a", "enc_b"}},
                                                        {"cfm", "cfm", {"cfm_a", "cfm_b", "cfm_c"}}};
    std::vector<int> live_at_load;
    auto prepare = [](const std::string & stage, const std::string &, std::string & out) {
        out = "prepared-" + stage;
        return true;
    };
    auto load = [&](const std::string &, const std::string & path, const std::string &) {
        if (generate) CHECK(path.rfind("prepared-", 0) == 0);
        live_at_load.push_back(g_live.load());
        return std::static_pointer_cast<void>(std::make_shared<Counted>());
    };
    {
        bench_ttr::ManualOutcome o = bench_ttr::run_manual(stages, generate, gated, prepare, load, fake_now, 0.0);
        CHECK(o.results.size() == 5);
        CHECK(o.retained.size() == 5);
        CHECK(g_live.load() == 5);
        CHECK((live_at_load == std::vector<int>{0, 1, 2, 3, 4}));
        CHECK(bench_ttr::manual_ready_ms(o.results) > 0);
    }
    CHECK(g_live.load() == 0);
}

// A failed load is reported not-ready and is not counted as resident.
void manual_failure_is_not_retained() {
    const std::vector<bench_ttr::ManualStage> stages = {{"hift", "hift", {"ok_1", "bad", "ok_2"}}};
    auto prepare = [](const std::string &, const std::string &, std::string &) { return false; };
    auto load = [](const std::string &, const std::string &, const std::string & fn) {
        if (fn == "bad") return std::shared_ptr<void>();
        return std::static_pointer_cast<void>(std::make_shared<Counted>());
    };
    {
        bench_ttr::ManualOutcome o = bench_ttr::run_manual(stages, false, true, prepare, load, fake_now, 0.0);
        CHECK(o.retained.size() == 2);
        CHECK(g_live.load() == 2);
        CHECK(o.results.size() == 3);
        CHECK(!o.results[1].ok && o.results[1].ready_ms < 0);
        CHECK(bench_ttr::manual_ready_ms(o.results) == -1);
    }
    CHECK(g_live.load() == 0);
}

// A failed prepare fails every load of its stage without calling load.
void manual_failed_prepare_fails_its_stage() {
    const std::vector<bench_ttr::ManualStage> stages = {{"cfm", "cfm", {"a", "b"}}};
    int loads = 0;
    auto prepare = [](const std::string &, const std::string &, std::string &) { return false; };
    auto load = [&](const std::string &, const std::string &, const std::string &) {
        ++loads;
        return std::static_pointer_cast<void>(std::make_shared<Counted>());
    };
    bench_ttr::ManualOutcome o = bench_ttr::run_manual(stages, true, false, prepare, load, fake_now, 0.0);
    CHECK(loads == 0);
    CHECK(o.retained.empty());
    CHECK(bench_ttr::manual_ready_ms(o.results) == -1);
}

// Fake slots for poll_slots: the clock advances 10 ms per now() call.
struct FakeGroup {
    std::vector<bench_ttr::Observation> states;
    double clock = 0;
    int finished_calls = 0;
    int complete_on_call = 3;  // finished() turns true on this call
};

bench_ttr::Observation obs(bench_ttr::Kind k) {
    switch (k) {
        case bench_ttr::Kind::ready: return {"ready", k};
        case bench_ttr::Kind::deferred_memory: return {"deferred(memory)", k};
        default: return {"loading", k};
    }
}

bench_ttr::PollOutcome poll(FakeGroup & g, double deadline, std::function<void(FakeGroup &)> on_complete) {
    return bench_ttr::poll_slots(
        g.states.size(), [&](size_t i) { return g.states[i]; },
        [&] {
            if (++g.finished_calls < g.complete_on_call) return false;
            on_complete(g);
            return true;
        },
        [&] { return g.clock += 10; }, [] {}, 0.0, deadline);
}

// The last load becomes ready after the previous sample, and the group reports
// completion in the same instant: the slot must still be seen ready.
void poll_sees_a_load_that_completes_with_the_group() {
    FakeGroup g;
    g.states = {obs(bench_ttr::Kind::ready), obs(bench_ttr::Kind::other)};
    const bench_ttr::PollOutcome o =
        poll(g, 1e9, [](FakeGroup & f) { f.states[1] = obs(bench_ttr::Kind::ready); });
    CHECK(o.finished);
    CHECK(o.ready_at[0] > 0);
    CHECK(o.ready_at[1] > 0);
    CHECK(bench_ttr::all_ready_ms(o.ready_at) == o.ready_at[1]);
    CHECK(!o.transitions.empty() && o.transitions.back().slot == 1 && o.transitions.back().state == "ready");
}

// Deferred(memory) time accrues per sample; a slot that never becomes ready
// fails the run; the deadline ends a group that never finishes.
void poll_deferred_time_and_deadline() {
    FakeGroup g;
    g.complete_on_call = 1 << 30;
    g.states = {obs(bench_ttr::Kind::deferred_memory), obs(bench_ttr::Kind::ready)};
    const bench_ttr::PollOutcome o = poll(g, 100, [](FakeGroup &) {});
    CHECK(!o.finished);
    CHECK(o.ready_at[0] < 0);
    CHECK(o.deferred_ms[0] >= 100);
    CHECK(o.deferred_ms[1] == 0);
    CHECK(bench_ttr::all_ready_ms(o.ready_at) == -1);
}

// The start readings come before the run loads anything, the end readings
// while every model it loaded is still held, and the models are released only
// after the end readings. A fake available-memory reading drops by 100 per
// live model, so a start reading taken after the run would read 600, and an
// end reading taken after the release would see no live models.
void readings_bracket_the_run() {
    std::vector<std::string> order;
    auto available = [] { return 1000 - 100 * g_live.load(); };
    const auto r = bench_ttr::with_readings(
        [&] {
            order.push_back("start");
            return available();
        },
        [&](std::shared_ptr<void> & keep) {
            order.push_back("run");
            auto held = std::make_shared<std::vector<std::shared_ptr<Counted>>>();
            for (int i = 0; i < 4; ++i) held->push_back(std::make_shared<Counted>());
            keep = held;
            return 4;
        },
        [&] {
            order.push_back("end");
            return std::make_pair(g_live.load(), available());
        });
    CHECK((order == std::vector<std::string>{"start", "run", "end"}));
    CHECK(r.start == 1000);
    CHECK(r.run == 4);
    CHECK(r.end.first == 4);
    CHECK(r.end.second == 600);
    CHECK(g_live.load() == 0);
}

}  // namespace

int main() {
    readings_bracket_the_run();
    poll_sees_a_load_that_completes_with_the_group();
    poll_deferred_time_and_deadline();
    for (bool generate : {false, true})
        for (bool gated : {false, true}) manual_retains_every_loaded_model(generate, gated);
    manual_failure_is_not_retained();
    manual_failed_prepare_fails_its_stage();
    if (g_failures) {
        std::fprintf(stderr, "test_bench_ttr: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_bench_ttr: ok\n");
    return 0;
}
