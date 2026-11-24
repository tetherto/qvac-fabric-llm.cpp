// Parity harness for the CosyVoice3 speech_tokenizer_v3 GGUF (the speech
// tokenizer behind zero-shot voice cloning).  Unlike test_s3tokenizer, which
// only reports metrics, this harness asserts its bounds and fails the build's
// ctest run when parity regresses.
//
// Usage:
//   ./build/test-s3tokenizer-v3 --s3tok-gguf S3TOK.gguf --in-dir DIR
//       [--max-mel-abs 1e-3] [--min-token-match 0.97]
//
// DIR holds the fixtures written by scripts/dump-s3tokenizer-v3-reference.py:
//   wav_16k.npy   float32 (1, N) or (N,)   16 kHz mono waveform
//   log_mel.npy   float32 (1, 128, T)      whisper-style log-mel fed upstream
//   tokens.npy    int32/int64 (T_tok,)     reference token stream
//
// Bounds are asserted two-sided per the review standard: the elementwise
// mel bound is scale-sensitive (log-mel units), and token parity requires the
// exact reference length plus a minimum exact-match ratio.  Measured on the
// zero_shot_prompt.wav fixture: mel max_abs ~2.3e-5; token match 87/87 vs the
// ONNX session and 86/87 vs this port (one FSQ near-boundary flip), f16
// identical to f32, q8_0 83/87 — the CMake q8_0 variant passes a looser
// --min-token-match measured with margin.
//
// COSYVOICE_TEST_GPU=1 runs the encoder graph on the same GPU backend the
// CosyVoice3 engine would select (cosyvoice_gpu_requirement), so the ctest
// gpu label validates the exact device class production uses.

#include "s3tokenizer.h"
#include "npy.h"
#include "backend_selection.h"
#include "cosyvoice_pipeline.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <charconv>
#include <string>
#include <vector>

namespace {

bool parse_bounded_arg(const char * s, double lo, double hi, double & out) {
    char * end = nullptr;
    const double v = std::strtod(s, &end);
    if (end == s || *end != '\0' || !(v >= lo) || !(v <= hi)) return false;
    out = v;
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    std::string gguf, in_dir;
    double max_mel_abs = 1e-3;
    double min_token_match = 0.97;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char * flag) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", flag); return nullptr; }
            return argv[++i];
        };
        if (a == "--s3tok-gguf") { const char * v = next("--s3tok-gguf"); if (!v) return 2; gguf = v; }
        else if (a == "--in-dir") { const char * v = next("--in-dir"); if (!v) return 2; in_dir = v; }
        else if (a == "--max-mel-abs") {
            const char * v = next("--max-mel-abs");
            if (!v || !parse_bounded_arg(v, 0.0, 1e6, max_mel_abs)) {
                fprintf(stderr, "--max-mel-abs takes a nonnegative number\n"); return 2;
            }
        }
        else if (a == "--min-token-match") {
            const char * v = next("--min-token-match");
            if (!v || !parse_bounded_arg(v, 0.0, 1.0, min_token_match)) {
                fprintf(stderr, "--min-token-match takes a value in [0, 1]\n"); return 2;
            }
        }
        else {
            fprintf(stderr, "usage: %s --s3tok-gguf S3TOK.gguf --in-dir DIR "
                            "[--max-mel-abs 1e-3] [--min-token-match 0.97]\n", argv[0]);
            return 2;
        }
    }
    if (gguf.empty() || in_dir.empty()) {
        fprintf(stderr, "missing --s3tok-gguf / --in-dir\n");
        return 2;
    }

    ggml_backend_t backend = nullptr;
    if (std::getenv("COSYVOICE_TEST_GPU")) {
        backend = ::tts_cpp::detail::init_gpu_backend(
            99, /*verbose=*/false, "test-s3tokenizer-v3", /*vulkan_device=*/0,
            /*allow_arm_mali=*/false, /*out_gpu_present_but_unused=*/nullptr,
            cosyvoice_gpu_requirement());
        if (!backend) { fprintf(stderr, "FAIL: COSYVOICE_TEST_GPU set but no GPU backend\n"); return 1; }
    }

    fprintf(stderr, "[1/3] loading tokenizer weights from %s\n", gguf.c_str());
    s3tokv2_weights w;
    if (!s3tokv2_load(gguf, w)) return 1;
    fprintf(stderr, "      version=%d n_mels=%d n_state=%d n_head=%d n_layer=%d codebook=%d\n",
            w.version, w.n_mels, w.n_state, w.n_head, w.n_layer, w.codebook_size);
    if (w.version != 3 || w.n_layer != 12) {
        fprintf(stderr, "FAIL: expected speech_tokenizer_v3 (version=3, 12 layers)\n");
        return 1;
    }

    npy_array wav_npy = npy_load(in_dir + "/wav_16k.npy");
    std::vector<float> wav((const float *)wav_npy.data.data(),
                           (const float *)wav_npy.data.data() + wav_npy.n_elements());
    fprintf(stderr, "      wav: %zu samples (%.2f s)\n", wav.size(), (double)wav.size() / 16000.0);

    fprintf(stderr, "[2/3] log-mel parity\n");
    int T_mel = 0;
    std::vector<float> mel_cpp = s3tokv2_log_mel(wav, w, T_mel);
    npy_array mel_ref = npy_load(in_dir + "/log_mel.npy");
    if (mel_cpp.size() != mel_ref.n_elements()) {
        fprintf(stderr, "FAIL: log-mel size %zu != reference %zu\n",
                mel_cpp.size(), mel_ref.n_elements());
        return 1;
    }
    const float * mr = (const float *)mel_ref.data.data();
    double mel_ma = 0.0;
    for (size_t i = 0; i < mel_cpp.size(); ++i) {
        const double d = std::fabs((double)mel_cpp[i] - (double)mr[i]);
        if (d > mel_ma) mel_ma = d;
    }
    fprintf(stderr, "      log_mel: T=%d max_abs=%.4e (bound %.4e)\n", T_mel, mel_ma, max_mel_abs);
    if (mel_ma > max_mel_abs) {
        fprintf(stderr, "FAIL: log-mel max_abs %.4e exceeds bound %.4e\n", mel_ma, max_mel_abs);
        return 1;
    }

    fprintf(stderr, "[3/3] token parity%s\n", backend ? " (GPU)" : " (CPU)");
    std::vector<int32_t> tokens;
    if (!s3tokv2_tokenize(wav, w, -1, tokens, /*n_threads=*/0, backend)) return 1;

    npy_array tok_ref = npy_load(in_dir + "/tokens.npy");
    std::vector<int64_t> ref(tok_ref.n_elements());
    if (tok_ref.dtype == "<i4") {
        const int32_t * p = (const int32_t *)tok_ref.data.data();
        for (size_t i = 0; i < ref.size(); ++i) ref[i] = p[i];
    } else if (tok_ref.dtype == "<i8") {
        const int64_t * p = (const int64_t *)tok_ref.data.data();
        for (size_t i = 0; i < ref.size(); ++i) ref[i] = p[i];
    } else {
        fprintf(stderr, "FAIL: tokens.npy dtype %s (want <i4 or <i8)\n", tok_ref.dtype.c_str());
        return 1;
    }

    if (tokens.size() != ref.size()) {
        fprintf(stderr, "FAIL: token count %zu != reference %zu\n", tokens.size(), ref.size());
        return 1;
    }
    size_t n_match = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if ((int64_t)tokens[i] == ref[i]) {
            ++n_match;
        } else {
            fprintf(stderr, "      mismatch @%zu: cpp=%d ref=%lld\n",
                    i, tokens[i], (long long)ref[i]);
        }
    }
    const double ratio = ref.empty() ? 1.0 : (double)n_match / (double)ref.size();
    fprintf(stderr, "      tokens: %zu/%zu match (%.4f, bound %.4f)\n",
            n_match, ref.size(), ratio, min_token_match);
    if (ratio < min_token_match) {
        fprintf(stderr, "FAIL: token match %.4f below bound %.4f\n", ratio, min_token_match);
        return 1;
    }

    fprintf(stderr, "PASS\n");
    return 0;
}
