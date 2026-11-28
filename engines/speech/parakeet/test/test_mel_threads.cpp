// The mel front-end fans its frame loops out over worker threads; the result must
// not depend on the thread count. Pure unit test, no GGUF.
//
// Usage: test-mel-threads
// Exit 0 on success; non-zero on failure.
#include "mel_preprocess.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr float kPI = 3.14159265358979323846f;

std::vector<float> make_signal(int n_samples) {
    std::vector<float> x(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        const float t = (float) i / 16000.0f;
        x[i] = std::sin(2.0f * kPI * 440.0f * t) + 0.3f * std::sin(2.0f * kPI * 1320.0f * t)
             + 1e-3f * std::sin(2.0f * kPI * 7777.0f * t + 0.123f);
    }
    return x;
}

std::vector<float> make_hann_window(int win_length) {
    std::vector<float> w(win_length);
    for (int i = 0; i < win_length; ++i) {
        w[i] = 0.5f * (1.0f - std::cos(2.0f * kPI * (float) i / (float) (win_length - 1)));
    }
    return w;
}

std::vector<float> make_synthetic_filterbank(int n_mels, int n_bins) {
    std::vector<float> fb((size_t) n_mels * n_bins);
    for (int m = 0; m < n_mels; ++m) {
        const int center = m * n_bins / n_mels;
        for (int k = 0; k < n_bins; ++k) {
            const float d = std::fabs((float) (k - center)) / (float) n_bins;
            fb[(size_t) m * n_bins + k] = 1e-3f + (d < 0.1f ? 1.0f - 10.0f * d : 0.0f);
        }
    }
    return fb;
}

parakeet::MelConfig make_config(int n_threads) {
    parakeet::MelConfig cfg;
    cfg.filterbank = make_synthetic_filterbank(cfg.n_mels, cfg.n_fft / 2 + 1);
    cfg.window     = make_hann_window(cfg.win_length);
    cfg.n_threads  = n_threads;
    return cfg;
}

bool mel_equal_across_threads(int n_samples) {
    const std::vector<float> x = make_signal(n_samples);
    std::vector<float> serial, threaded;
    int n_serial = 0, n_threaded = 0;
    if (parakeet::compute_log_mel(x.data(), n_samples, make_config(1), serial, n_serial) != 0) return false;
    if (parakeet::compute_log_mel(x.data(), n_samples, make_config(7), threaded, n_threaded) != 0) return false;
    if (n_serial != n_threaded || serial.size() != threaded.size()) {
        std::fprintf(stderr, "frame count differs: %d vs %d\n", n_serial, n_threaded);
        return false;
    }
    if (std::memcmp(serial.data(), threaded.data(), serial.size() * sizeof(float)) != 0) {
        std::fprintf(stderr, "mel differs between 1 and 7 threads (%d frames)\n", n_serial);
        return false;
    }
    std::printf("[mel-threads] %d samples -> %d frames byte-equal across thread counts\n", n_samples, n_serial);
    return true;
}

}  // namespace

int main() {
    // 30 s engages every worker; 0.5 s stays on one thread by the frame heuristic.
    if (!mel_equal_across_threads(16000 * 30)) return 1;
    if (!mel_equal_across_threads(8000)) return 1;
    std::printf("[mel-threads] PASS\n");
    return 0;
}
