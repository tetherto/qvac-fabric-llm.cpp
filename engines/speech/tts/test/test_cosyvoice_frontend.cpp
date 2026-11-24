// Round-trip parity harness for the CosyVoice3 voice-cloning front-end.
//
// Full mode (fixtures required):
//   ./build/test-cosyvoice-frontend --s3tok-gguf S3TOK.gguf
//       --campplus-gguf CAMPPLUS.gguf --wav zero_shot_prompt.wav
//       --in-dir cosyvoice3-ref
//       [--min-stok-match 0.95] [--max-feat-abs 5e-3]
//       [--min-emb-cosine 0.999] [--max-emb-abs 0.15]
//
// The token and embedding bounds are looser than the isolated component
// harnesses': test-s3tokenizer-v3 and test-voice-embedding feed the models
// the fixtures' own torch-resampled 16 kHz stream (86/87 tokens, embedding
// max_abs 1.8e-5 on this clip), while this harness runs the production
// resample_sinc front-end, whose waveform-level differences from torchaudio's
// resampler flip a few extra near-boundary FSQ codes (84/87 measured) and
// shift the raw embedding elementwise (max_abs 5.2e-2 measured at cosine
// 0.9997; the flow only consumes the L2-normalised direction).  prompt_feat
// stays tight (7e-5 measured) because its 24 kHz leg does not resample this
// fixture.  Content-level equivalence downstream is covered by the
// engine-level clone test.
//
// The fixtures are the upstream frontend_zero_shot outputs captured by
// scripts/dump-cosyvoice3-reference.py for the same wav — the tensors
// voice.gguf was baked from.  Running the native front-end on that wav and
// matching them is the end-to-end proof that native cloning reproduces the
// baked-voice pipeline: token stream (exact-match ratio + exact count),
// prompt mel (elementwise bound, scale-sensitive log-mel units), CAM++
// embedding (cosine AND elementwise bound), plus the mel_len1 ==
// 2 * n_tokens alignment invariant and the prompt_stok == prompt_token
// shared-stream property of the fixtures themselves.
//
// Guards mode (no fixtures, always-on unit test):
//   ./build/test-cosyvoice-frontend --guards-only
// Exercises the fail-closed paths: sub-0.5 s and over-30 s references, a
// NaN-bearing float WAV, and a missing wav must all return their specific
// errors, and a valid input with a bogus GGUF path must fail with the
// tokenizer error (proving the audio gates run first).
//
// COSYVOICE_TEST_GPU=1 runs the tokenizer encoder on the engine-selected GPU.

#include "cosyvoice_frontend.h"
#include "backend_selection.h"
#include "cosyvoice_pipeline.h"
#include "npy.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
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

bool all_finite(const std::vector<float> & v) {
    for (const float x : v) {
        if (!std::isfinite(x)) return false;
    }
    return true;
}

// Minimal float32 WAV writer for the synthetic guard inputs.  With
// inject_nan set, one sample in the middle carries a NaN payload — legal in
// IEEE-float WAV and exactly what the front-end must reject.
bool write_wav_f32(const std::string & path, int sr, double seconds,
                   bool inject_nan = false) {
    const uint32_t n = (uint32_t)(sr * seconds);
    const uint32_t data_bytes = n * 4;
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) return false;
    auto u32 = [&](uint32_t v) { fwrite(&v, 4, 1, f); };
    auto u16 = [&](uint16_t v) { fwrite(&v, 2, 1, f); };
    fwrite("RIFF", 1, 4, f); u32(36 + data_bytes); fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); u32(16); u16(3 /*IEEE float*/); u16(1);
    u32((uint32_t)sr); u32((uint32_t)sr * 4); u16(4); u16(32);
    fwrite("data", 1, 4, f); u32(data_bytes);
    for (uint32_t i = 0; i < n; ++i) {
        float v = 0.1f * std::sin(2.0 * 3.14159265358979 * 220.0 * i / sr);
        if (inject_nan && i == n / 2) v = std::numeric_limits<float>::quiet_NaN();
        fwrite(&v, 4, 1, f);
    }
    fclose(f);
    return true;
}

int run_guards() {
    const std::string dir = ".";
    const std::string wav_short = dir + "/cosyvoice-frontend-guard-short.wav";
    const std::string wav_long  = dir + "/cosyvoice-frontend-guard-long.wav";
    const std::string wav_ok    = dir + "/cosyvoice-frontend-guard-ok.wav";
    const std::string wav_nan   = dir + "/cosyvoice-frontend-guard-nan.wav";
    if (!write_wav_f32(wav_short, 16000, 0.3) ||
        !write_wav_f32(wav_long, 16000, 31.0) ||
        !write_wav_f32(wav_ok, 16000, 1.0) ||
        !write_wav_f32(wav_nan, 16000, 1.0, /*inject_nan=*/true)) {
        fprintf(stderr, "FAIL: cannot write synthetic wavs\n");
        return 1;
    }

    cosyvoice_prompt out;
    std::string err;
    int rc = 0;

    if (cosyvoice_frontend_run(wav_short, "missing.gguf", "missing.gguf",
                               nullptr, 0, out, err) ||
        err.find("too short") == std::string::npos) {
        fprintf(stderr, "FAIL: 0.3 s reference accepted (err='%s')\n", err.c_str());
        rc = 1;
    }
    err.clear();
    if (cosyvoice_frontend_run(wav_long, "missing.gguf", "missing.gguf",
                               nullptr, 0, out, err) ||
        err.find("too long") == std::string::npos) {
        fprintf(stderr, "FAIL: 31 s reference accepted (err='%s')\n", err.c_str());
        rc = 1;
    }
    err.clear();
    if (cosyvoice_frontend_run(wav_ok, "missing.gguf", "missing.gguf",
                               nullptr, 0, out, err) ||
        err.find("speech tokenizer") == std::string::npos) {
        fprintf(stderr, "FAIL: bogus tokenizer GGUF not rejected (err='%s')\n", err.c_str());
        rc = 1;
    }
    err.clear();
    if (cosyvoice_frontend_run(wav_nan, "missing.gguf", "missing.gguf",
                               nullptr, 0, out, err) ||
        err.find("non-finite") == std::string::npos) {
        fprintf(stderr, "FAIL: NaN-bearing reference accepted (err='%s')\n", err.c_str());
        rc = 1;
    }
    err.clear();
    if (cosyvoice_frontend_run(dir + "/no-such-file.wav", "missing.gguf",
                               "missing.gguf", nullptr, 0, out, err) ||
        err.find("cannot load reference audio") == std::string::npos) {
        fprintf(stderr, "FAIL: missing wav not rejected (err='%s')\n", err.c_str());
        rc = 1;
    }

    std::remove(wav_short.c_str());
    std::remove(wav_long.c_str());
    std::remove(wav_ok.c_str());
    std::remove(wav_nan.c_str());
    if (rc == 0) fprintf(stderr, "PASS (guards)\n");
    return rc;
}

} // namespace

int main(int argc, char ** argv) {
    std::string s3tok, campplus, wav, in_dir;
    double min_stok_match = 0.95;
    double max_feat_abs = 5e-3;
    double min_emb_cosine = 0.999;
    double max_emb_abs = 0.15;
    bool guards_only = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char * flag) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", flag); return nullptr; }
            return argv[++i];
        };
        auto bounded = [&](const char * flag, double lo, double hi, double & dst) {
            const char * v = next(flag);
            if (!v || !parse_bounded_arg(v, lo, hi, dst)) {
                fprintf(stderr, "%s takes a number in [%g, %g]\n", flag, lo, hi);
                return false;
            }
            return true;
        };
        if      (a == "--guards-only")   { guards_only = true; }
        else if (a == "--s3tok-gguf")    { const char * v = next("--s3tok-gguf");    if (!v) return 2; s3tok = v; }
        else if (a == "--campplus-gguf") { const char * v = next("--campplus-gguf"); if (!v) return 2; campplus = v; }
        else if (a == "--wav")           { const char * v = next("--wav");           if (!v) return 2; wav = v; }
        else if (a == "--in-dir")        { const char * v = next("--in-dir");        if (!v) return 2; in_dir = v; }
        else if (a == "--min-stok-match"){ if (!bounded("--min-stok-match", 0.0, 1.0, min_stok_match)) return 2; }
        else if (a == "--max-feat-abs")  { if (!bounded("--max-feat-abs", 0.0, 1e6, max_feat_abs)) return 2; }
        else if (a == "--min-emb-cosine"){ if (!bounded("--min-emb-cosine", 0.0, 1.0, min_emb_cosine)) return 2; }
        else if (a == "--max-emb-abs")   { if (!bounded("--max-emb-abs", 0.0, 1e6, max_emb_abs)) return 2; }
        else {
            fprintf(stderr, "usage: %s --s3tok-gguf G --campplus-gguf C --wav W --in-dir DIR\n"
                            "          [--min-stok-match 0.97] [--max-feat-abs 5e-3]\n"
                            "          [--min-emb-cosine 0.999] [--max-emb-abs 0.15]\n"
                            "       %s --guards-only\n", argv[0], argv[0]);
            return 2;
        }
    }

    if (guards_only) return run_guards();

    if (s3tok.empty() || campplus.empty() || wav.empty() || in_dir.empty()) {
        fprintf(stderr, "missing --s3tok-gguf / --campplus-gguf / --wav / --in-dir\n");
        return 2;
    }

    ggml_backend_t backend = nullptr;
    if (std::getenv("COSYVOICE_TEST_GPU")) {
        backend = ::tts_cpp::detail::init_gpu_backend(
            99, /*verbose=*/false, "test-cosyvoice-frontend", /*vulkan_device=*/0,
            /*allow_arm_mali=*/false, /*out_gpu_present_but_unused=*/nullptr,
            cosyvoice_gpu_requirement());
        if (!backend) { fprintf(stderr, "FAIL: COSYVOICE_TEST_GPU set but no GPU backend\n"); return 1; }
    }

    fprintf(stderr, "[1/4] front-end on %s%s\n", wav.c_str(), backend ? " (GPU tokenizer)" : "");
    cosyvoice_prompt got;
    std::string err;
    if (!cosyvoice_frontend_run(wav, s3tok, campplus, backend, 0, got, err)) {
        fprintf(stderr, "FAIL: %s\n", err.c_str());
        return 1;
    }
    fprintf(stderr, "      tokens=%zu mel_len1=%d emb=%zu\n",
            got.prompt_stok.size(), got.mel_len1, got.embedding.size());
    // Max-abs loops skip NaN (fabs(NaN) > max is false) and a NaN cosine is
    // not below any bound, so non-finite outputs would pass every threshold
    // below; reject them up front.
    if (!all_finite(got.prompt_feat) || !all_finite(got.embedding)) {
        fprintf(stderr, "FAIL: non-finite values in prompt_feat/embedding\n");
        return 1;
    }
    if (got.mel_len1 != 2 * (int)got.prompt_stok.size()) {
        fprintf(stderr, "FAIL: mel_len1 %d != 2 * n_tokens %zu\n",
                got.mel_len1, got.prompt_stok.size());
        return 1;
    }

    fprintf(stderr, "[2/4] token parity\n");
    npy_array stok_a = npy_load(in_dir + "/prompt_stok.npy");
    npy_array ptok_a = npy_load(in_dir + "/prompt_token.npy");
    if (stok_a.n_elements() != ptok_a.n_elements() ||
        std::memcmp(stok_a.data.data(), ptok_a.data.data(), stok_a.data.size()) != 0) {
        fprintf(stderr, "FAIL: fixture prompt_stok and prompt_token differ; the "
                        "shared-stream assumption is broken\n");
        return 1;
    }
    const int32_t * sr_ = npy_as_i32(stok_a);
    if (got.prompt_stok.size() != stok_a.n_elements()) {
        fprintf(stderr, "FAIL: token count %zu != fixture %zu\n",
                got.prompt_stok.size(), stok_a.n_elements());
        return 1;
    }
    size_t n_match = 0;
    for (size_t i = 0; i < got.prompt_stok.size(); ++i) {
        if (got.prompt_stok[i] == sr_[i]) ++n_match;
        else fprintf(stderr, "      token mismatch @%zu: cpp=%d ref=%d\n",
                     i, got.prompt_stok[i], sr_[i]);
    }
    const double ratio = (double)n_match / (double)got.prompt_stok.size();
    fprintf(stderr, "      tokens: %zu/%zu match (%.4f, bound %.4f)\n",
            n_match, got.prompt_stok.size(), ratio, min_stok_match);
    if (ratio < min_stok_match) { fprintf(stderr, "FAIL: token match below bound\n"); return 1; }

    fprintf(stderr, "[3/4] prompt_feat parity\n");
    npy_array feat_a = npy_load(in_dir + "/prompt_feat.npy");
    if (got.prompt_feat.size() != feat_a.n_elements()) {
        fprintf(stderr, "FAIL: prompt_feat size %zu != fixture %zu\n",
                got.prompt_feat.size(), feat_a.n_elements());
        return 1;
    }
    const float * fr = (const float *)feat_a.data.data();
    double feat_ma = 0.0;
    for (size_t i = 0; i < got.prompt_feat.size(); ++i) {
        const double d = std::fabs((double)got.prompt_feat[i] - (double)fr[i]);
        if (d > feat_ma) feat_ma = d;
    }
    fprintf(stderr, "      prompt_feat: (%d, 80) max_abs=%.4e (bound %.4e)\n",
            got.mel_len1, feat_ma, max_feat_abs);
    if (feat_ma > max_feat_abs) { fprintf(stderr, "FAIL: prompt_feat above bound\n"); return 1; }

    fprintf(stderr, "[4/4] embedding parity\n");
    npy_array emb_a = npy_load(in_dir + "/embedding.npy");
    if (got.embedding.size() != emb_a.n_elements()) {
        fprintf(stderr, "FAIL: embedding size %zu != fixture %zu\n",
                got.embedding.size(), emb_a.n_elements());
        return 1;
    }
    const float * er = (const float *)emb_a.data.data();
    double dot = 0.0, na = 0.0, nb = 0.0, emb_ma = 0.0;
    for (size_t i = 0; i < got.embedding.size(); ++i) {
        const double a = got.embedding[i], b = er[i];
        dot += a * b; na += a * a; nb += b * b;
        const double d = std::fabs(a - b);
        if (d > emb_ma) emb_ma = d;
    }
    const double cosine = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
    fprintf(stderr, "      embedding: cosine=%.6f (bound %.4f) max_abs=%.4e (bound %.4e)\n",
            cosine, min_emb_cosine, emb_ma, max_emb_abs);
    if (!std::isfinite(cosine)) { fprintf(stderr, "FAIL: non-finite embedding cosine\n"); return 1; }
    if (cosine < min_emb_cosine) { fprintf(stderr, "FAIL: embedding cosine below bound\n"); return 1; }
    if (emb_ma > max_emb_abs) { fprintf(stderr, "FAIL: embedding max_abs above bound\n"); return 1; }

    fprintf(stderr, "PASS\n");
    return 0;
}
