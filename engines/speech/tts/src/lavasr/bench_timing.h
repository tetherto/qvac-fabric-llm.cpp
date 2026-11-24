#pragma once

#include <cstdio>
#include <stdexcept>
#include <vector>

namespace tts_cpp::lavasr::bench {

struct Timing { double denoiser = 0.0, enhancer = 0.0, total = 0.0; };

struct Measurements {
    Timing first;
    std::vector<Timing> warmups, calls;
    double measured_seconds = 0.0;
    bool first_call_measured = false;
};

template <typename Call>
Measurements measure(int runs, int warmup, double min_seconds, Call call) {
    Measurements result;
    result.first = call();
    std::fprintf(stderr, "[first] total=%.6f s\n", result.first.total);
    auto record = [&](const Timing & timing) {
        result.calls.push_back(timing);
        result.measured_seconds += timing.total;
        std::fprintf(stderr, "[measure %zu/%d] denoiser=%.6f enhancer=%.6f total=%.6f s\n",
                     result.calls.size(), runs, timing.denoiser, timing.enhancer, timing.total);
    };
    // The desktop time-wrapped runner requests exactly one inference and
    // divides whole-process time by one clip. Do not add a hidden first call.
    result.first_call_measured = runs == 1 && warmup == 0 && min_seconds == 0.0;
    if (result.first_call_measured) {
        record(result.first);
        return result;
    }
    for (int i = 0; i < warmup; ++i) result.warmups.push_back(call());
    for (int i = 0; i < runs || result.measured_seconds < min_seconds; ++i) {
        if (i >= 100000) throw std::runtime_error("measurement call limit exceeded");
        record(call());
    }
    return result;
}

} // namespace tts_cpp::lavasr::bench
