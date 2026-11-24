// Pins cosyvoice_sinegen2_source's serial-vs-parallel equality. The vocoder
// was trained on this exact excitation (see the causal nearest-neighbour
// phase-upsampling note in cosyvoice_pipeline.cpp), so threading the harmonic
// loop must not change a single byte: every multi-threaded run is compared
// memcmp-equal against the 1-thread reference for the same f0 pattern and
// seed. Synthetic inputs; no fixtures, no backend.

#include "cosyvoice_pipeline.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr int   kSampleRate      = 24000;
constexpr int   kHarmonicNum     = 8;
constexpr float kSineAmp         = 0.1f;
constexpr float kNoiseStd        = 0.003f;
constexpr float kVoicedThreshold = 10.0f;
constexpr int   kUpsampleScale   = 480;
constexpr int   kMelFrames       = 96;

// Per-frame f0 mixing voiced sweeps, unvoiced stretches, and values that
// straddle the voiced threshold, upsampled by repetition exactly as
// cosyvoice_hift_synth builds f0_up from the f0 predictor's output.
std::vector<float> make_f0_wav() {
    std::vector<float> f0(kMelFrames);
    for (int i = 0; i < kMelFrames; ++i) {
        if (i % 16 < 3)      f0[i] = 0.0f;
        else if (i % 16 < 5) f0[i] = 9.0f + (float)(i % 3);
        else                 f0[i] = 120.0f + 80.0f * std::sin(0.13f * (float)i);
    }
    std::vector<float> f0_wav((size_t)kMelFrames * kUpsampleScale);
    for (int i = 0; i < kMelFrames; ++i) {
        for (int j = 0; j < kUpsampleScale; ++j) f0_wav[(size_t)i * kUpsampleScale + j] = f0[i];
    }
    return f0_wav;
}

std::vector<float> make_l_linear_w() {
    std::vector<float> w(kHarmonicNum + 1);
    for (int h = 0; h <= kHarmonicNum; ++h) w[h] = 0.3f - 0.05f * (float)h;
    return w;
}

std::vector<float> run(const std::vector<float> & f0_wav, const std::vector<float> & l_linear_w,
                       uint32_t seed, int n_threads) {
    return cosyvoice_sinegen2_source(f0_wav, kSampleRate, kHarmonicNum,
                                     kSineAmp, kNoiseStd, kVoicedThreshold, kUpsampleScale,
                                     l_linear_w, -0.02f, seed, n_threads);
}

bool identical(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

bool check_seed(const std::vector<float> & f0_wav, const std::vector<float> & l_linear_w,
                uint32_t seed) {
    const std::vector<float> serial = run(f0_wav, l_linear_w, seed, 1);
    const int thread_counts[] = {2, 3, 8, 16, 0};
    for (int n : thread_counts) {
        const std::vector<float> parallel = run(f0_wav, l_linear_w, seed, n);
        if (!identical(serial, parallel)) {
            fprintf(stderr, "FAIL: seed %u, n_threads=%d diverged from serial (%zu samples)\n",
                    seed, n, serial.size());
            return false;
        }
        fprintf(stderr, "seed %u, n_threads=%d: bit-identical (%zu samples)\n",
                seed, n, parallel.size());
    }
    return true;
}

} // namespace

int main() {
    const std::vector<float> f0_wav = make_f0_wav();
    const std::vector<float> l_linear_w = make_l_linear_w();
    const uint32_t seeds[] = {42u, 0u, 123456789u};
    for (uint32_t seed : seeds) {
        if (!check_seed(f0_wav, l_linear_w, seed)) return 1;
    }
    fprintf(stderr, "PASS\n");
    return 0;
}
