#include "lavasr/bench_timing.h"

#include <cstdio>

int main() {
    using namespace tts_cpp::lavasr::bench;
    int failures = 0;
    auto expect = [&](bool condition, const char * name) {
        if (!condition) { std::fprintf(stderr, "FAIL: %s\n", name); ++failures; }
    };
    int calls = 0;
    auto infer = [&]() { ++calls; return Timing{0.125, 0.125, 0.25}; };

    const auto single = measure(1, 0, 0.0, infer);
    expect(calls == 1, "desktop --runs 1 --warmup 0 performs exactly one inference");
    expect(single.first_call_measured && single.calls.size() == 1 && single.warmups.empty(),
           "the single first call is the measured sample");
    expect(single.measured_seconds == 0.25 && single.calls[0].total == single.first.total,
           "single-call processing duration is counted once");

    calls = 0;
    const auto warmed = measure(5, 1, 0.0, infer);
    expect(calls == 7 && !warmed.first_call_measured && warmed.calls.size() == 5 && warmed.warmups.size() == 1,
           "warmed measurements retain first plus one warmup plus five measured calls");
    expect(warmed.measured_seconds == 1.25, "warmed timing excludes first and warmup");

    calls = 0;
    const auto repeated = measure(3, 0, 0.0, infer);
    expect(calls == 4 && !repeated.first_call_measured && repeated.calls.size() == 3,
           "multi-run mode retains a separate first call");

    calls = 0;
    const auto sustained = measure(1, 0, 0.6, infer);
    expect(calls == 4 && !sustained.first_call_measured && sustained.calls.size() == 3,
           "a minimum duration retains the separate first call");
    expect(sustained.measured_seconds == 0.75, "only measured calls meet the sustained duration");

    calls = 0;
    const auto bounded = measure(3, 1, 0.1, infer);
    expect(calls == 5 && bounded.calls.size() == 3 && bounded.measured_seconds == 0.75,
           "the requested run count is still a lower bound when the duration target is shorter");

    if (failures) return 1;
    std::puts("PASS: 9 LavaSR benchmark timing cases");
    return 0;
}
