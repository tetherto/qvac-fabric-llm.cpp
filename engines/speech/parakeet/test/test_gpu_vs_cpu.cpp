// CPU vs GPU encoder bisect.
//
// Loads the model on CPU and on the GPU backend the build enables (Metal,
// Vulkan, ...), runs both encoders on identical mel, reports per-stage parity
// (max abs diff + relative L2), and requires the greedy CTC decodes of both
// logit sets to agree token for token.
//
// Usage:
//   test-gpu-vs-cpu <gguf> <wav> [language]
//
// Exit 0 on success; non-zero on failure or invalid arguments.

#include "parakeet_ctc.h"
#include "mel_preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

void compute_parity(const std::vector<float> & a, const std::vector<float> & b,
                    double & out_max_abs, double & out_rel) {
    double max_abs = 0.0, ref_sq = 0.0, diff_sq = 0.0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        max_abs = std::max(max_abs, std::abs(d));
        diff_sq += d * d;
        ref_sq  += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    out_max_abs = max_abs;
    out_rel = (ref_sq > 0) ? std::sqrt(diff_sq / ref_sq) : std::sqrt(diff_sq);
}

bool any_nan_or_inf(const std::vector<float> & v) {
    for (float x : v) {
        if (std::isnan(x) || std::isinf(x)) return true;
    }
    return false;
}

// rel_tol: stages accumulate fp16 rounding across 24 blocks, so later
// stages need a looser gate. 5e-2 is well above observed noise (~1.4e-2
// on block_last_out) but catches real divergence (broken GLU was >0.8).
bool summarize(const char * name,
               const std::vector<float> & cpu,
               const std::vector<float> & gpu,
               double rel_tol = 5e-2) {
    if (cpu.size() != gpu.size()) {
        std::fprintf(stderr,
                     "FAIL stage %-25s size mismatch: cpu=%zu gpu=%zu\n",
                     name, cpu.size(), gpu.size());
        return false;
    }
    double max_abs = 0.0, rel = 0.0;
    compute_parity(gpu, cpu, max_abs, rel);
    const bool nan_cpu = any_nan_or_inf(cpu);
    const bool nan_gpu  = any_nan_or_inf(gpu);
    const bool pass = !nan_cpu && !nan_gpu && rel < rel_tol;
    std::fprintf(stderr,
                 "%s stage %-25s n=%zu  max_abs=%9.3e  rel=%9.3e (tol=%7.1e)  %s%s\n",
                 pass ? "PASS" : "FAIL",
                 name, cpu.size(), max_abs, rel, rel_tol,
                 nan_cpu ? "[CPU has NaN/Inf!] " : "",
                 nan_gpu  ? "[GPU has NaN/Inf!] " : "");
    return pass;
}

enum class DecodeParity { Match, Mismatch, ConfigError };

DecodeParity compare_greedy_decode(const parakeet::ParakeetCtcModel & m_cpu,
                                   const parakeet::ParakeetCtcModel & m_gpu,
                                   const parakeet::EncoderOutputs   & out_cpu,
                                   const parakeet::EncoderOutputs   & out_gpu,
                                   const std::string                & language) {
    if (m_cpu.vocab_size <= 0 || m_cpu.vocab_size != m_gpu.vocab_size) {
        std::fprintf(stderr,
                     "FAIL greedy decode: no usable CTC head (cpu vocab=%d gpu vocab=%d)\n",
                     m_cpu.vocab_size, m_gpu.vocab_size);
        return DecodeParity::Mismatch;
    }
    if (out_cpu.logits.size() != out_gpu.logits.size()) {
        std::fprintf(stderr,
                     "FAIL greedy decode: logits size mismatch cpu=%zu gpu=%zu\n",
                     out_cpu.logits.size(), out_gpu.logits.size());
        return DecodeParity::Mismatch;
    }
    if (out_cpu.logits.empty()) {
        std::fprintf(stderr, "FAIL greedy decode: encoder produced no logits\n");
        return DecodeParity::Mismatch;
    }

    parakeet::CtcDecodeOptions dopts;
    try {
        dopts = parakeet::resolve_ctc_decode_options(m_cpu, language);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return DecodeParity::ConfigError;
    }

    const int n_frames = static_cast<int>(out_cpu.logits.size() / m_cpu.vocab_size);
    const std::vector<int32_t> ids_cpu = parakeet::ctc_greedy_decode(
        out_cpu.logits.data(), n_frames, m_cpu.vocab_size, m_cpu.blank_id, &dopts);
    const std::vector<int32_t> ids_gpu = parakeet::ctc_greedy_decode(
        out_gpu.logits.data(), n_frames, m_gpu.vocab_size, m_gpu.blank_id, &dopts);

    if (ids_cpu != ids_gpu) {
        std::fprintf(stderr,
                     "FAIL greedy decode: token ids differ (cpu=%zu gpu=%zu tokens)\n",
                     ids_cpu.size(), ids_gpu.size());
        return DecodeParity::Mismatch;
    }
    std::fprintf(stderr, "PASS greedy decode: %zu tokens identical\n", ids_cpu.size());
    return DecodeParity::Match;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: %s <parakeet-ctc.gguf> <wav> [language]\n"
                     "\n"
                     "Loads the model twice (CPU + GPU), runs the same mel\n"
                     "through both encoders, reports per-stage parity so we\n"
                     "can localise where the GPU diverges, and requires the\n"
                     "greedy CTC decodes of both logit sets to match. The\n"
                     "language id is required for multilingual GGUFs that\n"
                     "carry language masks (e.g. hi).\n",
                     argv[0]);
        return 2;
    }

    const std::string gguf_path = argv[1];
    const std::string wav_path  = argv[2];
    const std::string language  = argc > 3 ? argv[3] : "";

    using namespace parakeet;
    using namespace parakeet;

    // ---- load CPU model ----
    std::fprintf(stderr, "[*] loading CPU model...\n");
    ParakeetCtcModel m_cpu;
    if (int rc = load_from_gguf(gguf_path, m_cpu, 0, /*n_gpu_layers=*/0, false); rc != 0) {
        std::fprintf(stderr, "error: CPU load_from_gguf rc=%d\n", rc);
        return 10;
    }

    // ---- load GPU model ----
    std::fprintf(stderr, "[*] loading GPU model...\n");
    ParakeetCtcModel m_gpu;
    if (int rc = load_from_gguf(gguf_path, m_gpu, 0, /*n_gpu_layers=*/1, false); rc != 0) {
        std::fprintf(stderr, "error: GPU load_from_gguf rc=%d\n", rc);
        return 11;
    }
    if (!model_has_gpu_backend(m_gpu)) {
        std::fprintf(stderr,
                     "error: no GPU backend selected (got '%s'); a CPU-vs-CPU "
                     "run would pass parity without testing anything\n",
                     model_active_backend_name(m_gpu).c_str());
        return 12;
    }

    // ---- read wav and compute mel on CPU (once) ----
    std::fprintf(stderr, "[*] reading wav %s\n", wav_path.c_str());
    std::vector<float> samples;
    int sr = 0;
    if (int rc = load_wav_mono_f32(wav_path, samples, sr); rc != 0) {
        std::fprintf(stderr, "error: load_wav_mono_f32 rc=%d\n", rc);
        return 20;
    }
    if (sr != m_cpu.mel_cfg.sample_rate) {
        std::fprintf(stderr, "error: wav sr=%d != model sr=%d\n", sr, m_cpu.mel_cfg.sample_rate);
        return 21;
    }

    std::vector<float> mel;
    int n_mel_frames = 0;
    compute_log_mel(samples.data(), (int) samples.size(), m_cpu.mel_cfg, mel, n_mel_frames);
    std::fprintf(stderr, "[*] mel: %d frames x %d mels\n", n_mel_frames, m_cpu.mel_cfg.n_mels);

    // ---- run encoder on both backends with the same mel ----
    std::fprintf(stderr, "[*] running CPU encoder...\n");
    EncoderOutputs out_cpu;
    if (int rc = run_encoder(m_cpu, mel.data(), n_mel_frames, m_cpu.mel_cfg.n_mels, out_cpu, parakeet::EncoderRoute::ggml()); rc != 0) {
        std::fprintf(stderr, "error: CPU run_encoder rc=%d\n", rc);
        return 30;
    }

    std::fprintf(stderr, "[*] running GPU encoder...\n");
    EncoderOutputs out_gpu;
    if (int rc = run_encoder(m_gpu, mel.data(), n_mel_frames, m_gpu.mel_cfg.n_mels, out_gpu, parakeet::EncoderRoute::ggml()); rc != 0) {
        std::fprintf(stderr, "error: GPU run_encoder rc=%d\n", rc);
        return 31;
    }

    // ---- per-stage parity ----
    std::fprintf(stderr, "\n=== per-stage parity (GPU vs CPU baseline) ===\n");
    int n_fail = 0;
    if (!summarize("subsampling_out",   out_cpu.subsampling_out,   out_gpu.subsampling_out))   ++n_fail;
    if (!summarize("block0_post_ff1",   out_cpu.block_0_post_ff1,  out_gpu.block_0_post_ff1))  ++n_fail;
    if (!summarize("block0_post_attn",  out_cpu.block_0_post_attn, out_gpu.block_0_post_attn)) ++n_fail;
    if (!summarize("block0_post_conv",  out_cpu.block_0_post_conv, out_gpu.block_0_post_conv)) ++n_fail;
    if (!summarize("block0_post_ff2",   out_cpu.block_0_post_ff2,  out_gpu.block_0_post_ff2))  ++n_fail;
    if (!summarize("block0_out",        out_cpu.block_0_out,       out_gpu.block_0_out))       ++n_fail;
    if (!summarize("block_last_out",    out_cpu.block_last_out,    out_gpu.block_last_out))    ++n_fail;
    if (!summarize("encoder_out",       out_cpu.encoder_out,       out_gpu.encoder_out))       ++n_fail;
    if (!summarize("logits",            out_cpu.logits,            out_gpu.logits))             ++n_fail;

    switch (compare_greedy_decode(m_cpu, m_gpu, out_cpu, out_gpu, language)) {
        case DecodeParity::ConfigError: return 40;
        case DecodeParity::Mismatch:    ++n_fail; break;
        case DecodeParity::Match:       break;
    }

    if (n_fail > 0) {
        std::fprintf(stderr, "\nFAILED: %d stage(s) exceeded tolerance\n", n_fail);
        return 1;
    }
    std::fprintf(stderr, "\nall stages passed\n");
    return 0;
}
