// The mel filterbank runs each filter over its non-zero band and applies the log
// inside the frame loop; the result must match the dense, two-pass formulation up to
// summation order (the skipped weights are exact zeros, but a compiler that fuses or
// vectorises the multiply-add chain rounds a shorter chain differently). Pure unit
// test, no GGUF.
//
// Usage: test-mel-bands
// Exit 0 on success; non-zero on failure.
#include "mel_preprocess.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
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

// Triangular filters with exact zeros outside their support, like the NeMo bank;
// `dense` fills every weight instead, so no band is narrower than the row.
std::vector<float> make_filterbank(int n_mels, int n_bins, bool dense) {
    std::vector<float> fb((size_t) n_mels * n_bins);
    for (int m = 0; m < n_mels; ++m) {
        const int center = (m + 1) * n_bins / (n_mels + 2);
        const int half   = 1 + m / 8;
        for (int k = 0; k < n_bins; ++k) {
            const int d = std::abs(k - center);
            const float tri = d < half ? 1.0f - (float) d / (float) half : 0.0f;
            fb[(size_t) m * n_bins + k] = dense ? tri + 1e-3f : tri;
        }
    }
    return fb;
}

parakeet::MelConfig make_config(bool dense) {
    parakeet::MelConfig cfg;
    cfg.filterbank = make_filterbank(cfg.n_mels, cfg.n_fft / 2 + 1, dense);
    cfg.window     = make_hann_window(cfg.win_length);
    cfg.n_threads  = 3;
    return cfg;
}

// The reference: every weight summed in double, then the log as a separate pass.
void dense_log_mel_frame(const float * frame_power, int n_bins, int n_mels,
                         const float * fb, float guard, float * out) {
    for (int m = 0; m < n_mels; ++m) {
        const float * row = fb + m * n_bins;
        double acc = 0.0;
        for (int k = 0; k < n_bins; ++k) acc += (double) row[k] * (double) frame_power[k];
        out[m] = (float) std::log(acc + (double) guard);
    }
}

// A float sum of n terms can differ from the double sum by up to n roundings.
bool within_summation_error(float got, float want, int n_terms) {
    const float ulp = std::nextafter(std::fabs(want), std::numeric_limits<float>::infinity()) - std::fabs(want);
    return std::fabs(got - want) <= (float) (n_terms + 1) * ulp;
}

bool frames_match(const std::vector<float> & got, const std::vector<float> & want,
                  const std::vector<parakeet::MelFilterBand> & bands) {
    for (size_t m = 0; m < got.size(); ++m) {
        if (!within_summation_error(got[m], want[m], bands[m].hi - bands[m].lo)) {
            std::fprintf(stderr, "filter %zu: banded %.9g vs dense %.9g\n", m, got[m], want[m]);
            return false;
        }
    }
    return true;
}

bool bands_match_dense(bool dense_bank) {
    const parakeet::MelConfig cfg = make_config(dense_bank);
    const int n_bins = cfg.n_fft / 2 + 1;
    const std::vector<parakeet::MelFilterBand> bands =
        parakeet::make_filter_bands(cfg.filterbank, cfg.n_mels, n_bins);
    if ((int) bands.size() != cfg.n_mels) {
        std::fprintf(stderr, "band count %zu != %d\n", bands.size(), cfg.n_mels);
        return false;
    }
    for (int m = 0; m < cfg.n_mels; ++m) {
        const float * row = cfg.filterbank.data() + (size_t) m * n_bins;
        for (int k = 0; k < n_bins; ++k) {
            const bool inside = k >= bands[m].lo && k < bands[m].hi;
            if (!inside && row[k] != 0.0f) {
                std::fprintf(stderr, "filter %d: non-zero weight at %d outside [%d, %d)\n",
                             m, k, bands[m].lo, bands[m].hi);
                return false;
            }
        }
    }
    std::vector<float> power(n_bins);
    for (int k = 0; k < n_bins; ++k) power[k] = 1e-6f + 0.37f * (float) ((k * 7919) % 101);
    std::vector<float> got(cfg.n_mels), want(cfg.n_mels);
    parakeet::log_mel_frame(power.data(), n_bins, cfg.n_mels, cfg.filterbank.data(), bands.data(),
                            cfg.log_zero_guard_value, got.data());
    dense_log_mel_frame(power.data(), n_bins, cfg.n_mels, cfg.filterbank.data(),
                        cfg.log_zero_guard_value, want.data());
    if (!frames_match(got, want, bands)) {
        std::fprintf(stderr, "banded frame differs from the dense reference (dense bank: %d)\n", dense_bank);
        return false;
    }
    return true;
}

bool batch_matches_incremental() {
    const int n_samples = 16000 * 3;
    const std::vector<float> x = make_signal(n_samples);
    const parakeet::MelConfig cfg = make_config(false);
    std::vector<float> batch;
    int n_batch = 0;
    if (parakeet::compute_log_mel(x.data(), n_samples, cfg, batch, n_batch) != 0) return false;
    if (n_batch <= 0 || batch.size() != (size_t) n_batch * cfg.n_mels) {
        std::fprintf(stderr, "batch mel shape %d x %zu\n", n_batch, batch.size());
        return false;
    }
    std::printf("[mel-bands] batch mel %d frames computed with banded filters\n", n_batch);
    return true;
}

}  // namespace

int main() {
    if (!bands_match_dense(false)) return 1;
    if (!bands_match_dense(true)) return 1;
    if (!batch_matches_incremental()) return 1;
    std::printf("[mel-bands] PASS\n");
    return 0;
}
