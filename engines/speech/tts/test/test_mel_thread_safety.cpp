// Concurrency guard for the shared mel/fbank extraction path.
//
// mel_graph_run (mel_extract_stft.cpp) is reached from every engine's voice
// bake — Chatterbox reference conditioning, the CosyVoice3 cloning front-end
// (fbank_kaldi_80 + mel_extract_24k_80) — and engines are constructed
// concurrently by callers.  Its graph-metadata arena was once a function-local
// static, which corrupted ggml graph metadata under exactly that load.  This
// harness pins the fix: N threads extract features from distinct synthetic
// inputs in a loop, and every result must be bit-identical to the
// single-threaded reference for the same input.
//
// Fixture-free; runs in the always-on unit tier.

#include "voice_features.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace {

// Deterministic pseudo-random floats (no <random> variance across libs).
float prng(uint32_t & state) {
    state = state * 1664525u + 1013904223u;
    return (float)(state >> 8) / (float)(1u << 24) - 0.5f;
}

std::vector<float> synth_wav(int n, uint32_t seed) {
    std::vector<float> w(n);
    uint32_t s = seed;
    for (int i = 0; i < n; ++i) {
        w[i] = 0.3f * std::sin(2.0 * 3.14159265358979 * 180.0 * i / 16000.0) +
               0.05f * prng(s);
    }
    return w;
}

std::vector<float> synth_fb(int n_mels, int n_bins, uint32_t seed) {
    std::vector<float> fb((size_t)n_mels * n_bins);
    uint32_t s = seed;
    for (auto & v : fb) v = 0.5f + 0.5f * std::fabs(prng(s));
    return fb;
}

bool bit_equal(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

} // namespace

int main() {
    const int n_threads = 8;
    const int n_iters = 6;

    // One distinct input pair per thread; kaldi fbank @ 16 kHz (512-point FFT,
    // 257 bins) and the 24 kHz 80-mel bank (1920-point FFT, 961 bins).
    std::vector<std::vector<float>> wavs_16k, wavs_24k;
    std::vector<std::vector<float>> ref_fbank, ref_mel;
    const std::vector<float> fb_kaldi = synth_fb(80, 257, 7);
    const std::vector<float> fb_24k = synth_fb(80, 961, 11);
    for (int t = 0; t < n_threads; ++t) {
        wavs_16k.push_back(synth_wav(16000 + 400 * t, 100 + t));
        wavs_24k.push_back(synth_wav(24000 + 600 * t, 200 + t));
        ref_fbank.push_back(fbank_kaldi_80(wavs_16k.back(), fb_kaldi));
        ref_mel.push_back(mel_extract_24k_80(wavs_24k.back(), fb_24k, 1e-9f));
        if (ref_fbank.back().empty() || ref_mel.back().empty()) {
            fprintf(stderr, "FAIL: single-threaded reference extraction failed\n");
            return 1;
        }
    }

    std::atomic<int> failures{0};
    std::vector<std::thread> pool;
    for (int t = 0; t < n_threads; ++t) {
        pool.emplace_back([&, t]() {
            for (int it = 0; it < n_iters; ++it) {
                const std::vector<float> got_fb = fbank_kaldi_80(wavs_16k[t], fb_kaldi);
                const std::vector<float> got_mel = mel_extract_24k_80(wavs_24k[t], fb_24k, 1e-9f);
                if (!bit_equal(got_fb, ref_fbank[t]) || !bit_equal(got_mel, ref_mel[t])) {
                    failures.fetch_add(1);
                    return;
                }
            }
        });
    }
    for (auto & th : pool) th.join();

    if (failures.load() != 0) {
        fprintf(stderr, "FAIL: %d thread(s) diverged from the single-threaded "
                        "reference\n", failures.load());
        return 1;
    }
    fprintf(stderr, "PASS (%d threads x %d iterations, bit-identical)\n",
            n_threads, n_iters);
    return 0;
}
