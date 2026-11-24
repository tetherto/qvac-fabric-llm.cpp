#include "lavasr/bench_validation.h"

#include <cstdio>
#include <limits>

int main() {
    using namespace tts_cpp::lavasr::bench;
    int failures = 0;
    auto expect = [&](bool condition, const char * name) {
        if (!condition) { std::fprintf(stderr, "FAIL: %s\n", name); ++failures; }
    };
    auto accepted = [](const Difference & d) { return passes(d, 2e-3, 2e-3, 0.9999); };
    expect(accepted(compare({0.1f, -0.2f}, {0.1f, -0.2f})), "identical PCM passes");
    expect(accepted(compare({0, 0}, {0, 0})), "identical silence passes");
    expect(!accepted(compare({0.1f}, {0.1f, 0})), "a matching prefix cannot hide a length mismatch");
    expect(!accepted(compare({std::numeric_limits<float>::quiet_NaN()}, {0})), "NaN fails");
    expect(!accepted(compare({0}, {std::numeric_limits<float>::infinity()})), "nonfinite reference fails");
    expect(!accepted(compare({0.1f, 0.2f}, {-0.1f, -0.2f})), "inverted waveform fails");
    expect(!accepted(compare({0.01f, 0}, {0, 0})), "nonzero output against silence fails");
    expect(!accepted(compare({0.101f}, {0.1f})), "relative error is checked as well as absolute error");
    if (failures) return 1;
    std::puts("PASS: 8 LavaSR benchmark validation cases");
    return 0;
}
