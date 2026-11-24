// Parity test — CosyVoice3 CausalHiFT vocoder: mel -> 24 kHz waveform.
//
// Runs cosyvoice_hift_synth() on a reference mel and asserts the waveform
// correlates with the PyTorch reference (hift_wav.npy) above a threshold.
// HiFT's NSF source excitation is partly stochastic (unvoiced noise, different
// RNG than the reference), so an exact match is not expected — the gate is a
// Pearson correlation, which caught the earlier metallic phase bug
// (0.136 -> 0.996 after the fix). Exits non-zero on failure; registered as a
// ctest, auto-disabled when fixtures are absent.
//
// Fixtures (generate with scripts/dump-cosyvoice3-reference.py):
//   <in-dir>/{hift_mel_in,hift_wav}.npy
//
// Usage:
//   test-cosyvoice-hift --hift-gguf HIFT.gguf --in-dir DIR [--min-corr 0.90]
//
// Set COSYVOICE_TEST_GPU=1 to run the check on the selected GPU backend
// (mirrors PARLER_TEST_GPU).  The GPU run gates in four steps: SineGen2
// integrates f0 into sine phases, so sub-Hz cross-backend f0 noise
// decorrelates the raw waveform with no audible effect.  (1) GPU f0 must
// match CPU f0 in length, at cosine >= --min-f0-cosine, and within
// --max-f0-hz-diff per frame (cosine alone is scale-invariant and would
// accept an octave error); (2) the voiced/unvoiced decision each f0 frame
// implies may flip on at most --max-voiced-flips frames (the Hz bound is
// blind to a pair straddling SineGen2's voiced threshold); (3) unpinned
// production legs — each backend synthesizing with its own f0, exactly as
// the engine runs — must agree on their 10 ms RMS energy envelopes at
// >= --min-env-corr (phase-insensitive, so the f0-noise decorrelation does
// not mask a real synthesis regression); (4) the GPU synth runs once more
// with the CPU f0 pinned and faces the same --min-corr waveform gate
// against the PyTorch reference.

#include "npy.h"
#include "backend_selection.h"
#include "cosyvoice_pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static double pearson(const float * a, const float * b, size_t n) {
    double ma = 0, mb = 0;
    for (size_t i = 0; i < n; ++i) { ma += a[i]; mb += b[i]; }
    ma /= (double)n; mb /= (double)n;
    double cov = 0, va = 0, vb = 0;
    for (size_t i = 0; i < n; ++i) {
        const double da = a[i] - ma, db = b[i] - mb;
        cov += da * db; va += da * da; vb += db * db;
    }
    return cov / (std::sqrt(va) * std::sqrt(vb));
}

static double cosine(const std::vector<float> & a, const std::vector<float> & b) {
    double dot = 0, na = 0, nb = 0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) { dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

// SineGen2's voiced/unvoiced switch (cosyvoice_pipeline.cpp voiced_threshold):
// an f0 pair straddling it produces qualitatively different excitation, which
// the Hz bound alone cannot see when the true f0 sits near the threshold.
static const float kVoicedThresholdHz = 10.0f;

static size_t count_voiced_flips(const std::vector<float> & a, const std::vector<float> & b) {
    size_t flips = 0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        if ((a[i] > kVoicedThresholdHz) != (b[i] > kVoicedThresholdHz)) ++flips;
    }
    return flips;
}

// Frame RMS energy at a 10 ms hop. Phase-insensitive: cross-backend f0 noise
// decorrelates the raw waveform (SineGen2 integrates f0 into sine phases), so
// the unpinned production legs are compared on their energy envelopes.
static std::vector<float> energy_envelope(const std::vector<float> & wav) {
    const size_t hop = 240;
    std::vector<float> env;
    env.reserve(wav.size() / hop + 1);
    for (size_t start = 0; start < wav.size(); start += hop) {
        const size_t end = std::min(start + hop, wav.size());
        double acc = 0;
        for (size_t i = start; i < end; ++i) acc += (double)wav[i] * wav[i];
        env.push_back((float)std::sqrt(acc / (double)(end - start)));
    }
    return env;
}

int main(int argc, char ** argv) {
    std::string gguf, in_dir;
    double min_corr = 0.90;
    double min_f0_cosine = 0.9999;
    double max_f0_hz_diff = 5.0;
    double min_env_corr = 0.90;
    size_t max_voiced_flips = 0;
    int seed = 42;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--hift-gguf" && i + 1 < argc) gguf = argv[++i];
        else if (a == "--in-dir" && i + 1 < argc) in_dir = argv[++i];
        else if (a == "--min-corr" && i + 1 < argc) min_corr = std::atof(argv[++i]);
        else if (a == "--min-f0-cosine" && i + 1 < argc) min_f0_cosine = std::atof(argv[++i]);
        else if (a == "--max-f0-hz-diff" && i + 1 < argc) max_f0_hz_diff = std::atof(argv[++i]);
        else if (a == "--min-env-corr" && i + 1 < argc) {
            char * end = nullptr;
            min_env_corr = std::strtod(argv[++i], &end);
            if (end == argv[i] || *end != '\0' || !(min_env_corr >= 0.0 && min_env_corr <= 1.0)) {
                fprintf(stderr, "--min-env-corr takes [0,1], got \"%s\"\n", argv[i]); return 2;
            }
        }
        else if (a == "--max-voiced-flips" && i + 1 < argc) {
            char * end = nullptr;
            const long v = std::strtol(argv[++i], &end, 10);
            if (end == argv[i] || *end != '\0' || v < 0) {
                fprintf(stderr, "--max-voiced-flips takes a nonnegative int, got \"%s\"\n", argv[i]); return 2;
            }
            max_voiced_flips = (size_t)v;
        }
        else if (a == "--seed" && i + 1 < argc) seed = std::atoi(argv[++i]);
        else { fprintf(stderr, "usage: %s --hift-gguf HIFT.gguf --in-dir DIR [--min-corr 0.90] [--min-f0-cosine 0.9999] [--max-f0-hz-diff 5.0]\n"
                               "          [--min-env-corr 0.90] [--max-voiced-flips 0]\n", argv[0]); return 2; }
    }
    if (gguf.empty() || in_dir.empty()) { fprintf(stderr, "missing --hift-gguf / --in-dir\n"); return 2; }

    const int MEL = 80;
    ggml_backend_t backend = nullptr;
    if (std::getenv("COSYVOICE_TEST_GPU")) {
        // Same requirement as the production engine (cosyvoice_engine.cpp), so
        // on a mixed-backend host the harness validates a backend the engine
        // would actually select instead of whichever device sorts first.
        backend = ::tts_cpp::detail::init_gpu_backend(
            99, /*verbose=*/false, "test-cosyvoice", /*vulkan_device=*/0,
            /*allow_arm_mali=*/false, /*out_gpu_present_but_unused=*/nullptr,
            cosyvoice_gpu_requirement());
        if (!backend) { fprintf(stderr, "FAIL: COSYVOICE_TEST_GPU set but no GPU backend\n"); return 1; }
    }
    model_ctx m = cosyvoice_load_gguf(gguf, backend);

    // mel npy shape (80, T) -> channel-major flat[ch*T + t].
    npy_array mel_a = npy_load(in_dir + "/hift_mel_in.npy");
    int T_mel = (int)mel_a.shape[1];
    std::vector<float> mel((size_t)T_mel * MEL);
    std::memcpy(mel.data(), npy_as_f32(mel_a), mel.size() * sizeof(float));

    npy_array ref_a = npy_load(in_dir + "/hift_wav.npy");
    size_t ref_n = 1;
    for (size_t d = 0; d < ref_a.shape.size(); ++d) ref_n *= (size_t)ref_a.shape[d];

    std::vector<float> wav;
    if (backend) {
        model_ctx m_cpu = cosyvoice_load_gguf(gguf);
        std::vector<float> f0_cpu = cosyvoice_hift_f0(m_cpu, mel, T_mel);
        std::vector<float> f0_gpu = cosyvoice_hift_f0(m, mel, T_mel);
        if (f0_gpu.size() != f0_cpu.size()) {
            fprintf(stderr, "FAIL: GPU f0 length %zu != CPU %zu\n",
                    f0_gpu.size(), f0_cpu.size());
            return 1;
        }
        // Cosine catches shape divergence but is scale-invariant (an octave
        // error scores 1.0), so the unpinned GPU f0 additionally faces an
        // absolute per-frame Hz bound against the CPU trajectory.
        double f0_cos = cosine(f0_cpu, f0_gpu);
        double max_hz = 0;
        for (size_t i = 0; i < f0_cpu.size(); ++i) {
            max_hz = std::max(max_hz, (double)std::fabs(f0_gpu[i] - f0_cpu[i]));
        }
        fprintf(stderr,
                "f0 cpu-vs-gpu cosine = %.6f (>= %.4f)  max |diff| = %.3f Hz "
                "(<= %.1f), %d frames\n",
                f0_cos, min_f0_cosine, max_hz, max_f0_hz_diff, T_mel);
        if (!(f0_cos >= min_f0_cosine) || !(max_hz <= max_f0_hz_diff)) {
            fprintf(stderr, "FAIL: GPU f0 predictor diverged from CPU\n");
            return 1;
        }
        // The Hz bound is blind to the voiced/unvoiced switch: a pair
        // straddling SineGen2's threshold changes the excitation
        // qualitatively even inside the bound, so decision parity is gated
        // separately.
        const size_t flips = count_voiced_flips(f0_cpu, f0_gpu);
        fprintf(stderr, "f0 voiced-decision flips = %zu (<= %zu, threshold %.1f Hz)\n",
                flips, max_voiced_flips, (double)kVoicedThresholdHz);
        if (flips > max_voiced_flips) {
            fprintf(stderr, "FAIL: GPU f0 flips voiced decisions against CPU\n");
            return 1;
        }
        // Unpinned production leg: each backend synthesizes with its OWN f0,
        // exactly as the engine runs it. Raw-waveform correlation is not
        // meaningful here (f0 noise decorrelates phases), so the legs are
        // compared on their 10 ms RMS energy envelopes.
        std::vector<float> wav_gpu_unpinned = cosyvoice_hift_synth(m, mel, T_mel, seed);
        std::vector<float> wav_cpu_unpinned = cosyvoice_hift_synth(m_cpu, mel, T_mel, seed);
        if (wav_gpu_unpinned.size() != wav_cpu_unpinned.size()) {
            fprintf(stderr, "FAIL: unpinned GPU waveform length %zu != CPU %zu\n",
                    wav_gpu_unpinned.size(), wav_cpu_unpinned.size());
            return 1;
        }
        const std::vector<float> env_gpu = energy_envelope(wav_gpu_unpinned);
        const std::vector<float> env_cpu = energy_envelope(wav_cpu_unpinned);
        const double env_corr = pearson(env_gpu.data(), env_cpu.data(), env_gpu.size());
        fprintf(stderr, "unpinned energy-envelope corr = %.6f (>= %.4f, %zu frames)\n",
                env_corr, min_env_corr, env_gpu.size());
        if (!(env_corr >= min_env_corr)) {
            fprintf(stderr, "FAIL: unpinned GPU synthesis energy envelope diverged from CPU\n");
            return 1;
        }
        // Pinned diagnostic leg (kept): CPU f0 into the GPU synthesizer makes
        // the raw waveform comparable against the PyTorch reference below.
        wav = cosyvoice_hift_synth(m, mel, T_mel, seed, nullptr, &f0_cpu);
    } else {
        wav = cosyvoice_hift_synth(m, mel, T_mel, seed);
    }

    size_t n = std::min(wav.size(), ref_n);
    if (n == 0) { fprintf(stderr, "FAIL: empty waveform\n"); return 1; }
    double corr = pearson(wav.data(), npy_as_f32(ref_a), n);

    fprintf(stderr, "hift waveform corr = %.6f  (threshold %.4f, got=%zu ref=%zu)\n",
            corr, min_corr, wav.size(), ref_n);
    if (!(corr >= min_corr)) { fprintf(stderr, "FAIL: hift waveform correlation below threshold\n"); return 1; }
    fprintf(stderr, "PASS\n");
    return 0;
}
