// Encoder output parity with and without the cached positional projections.
//
// The cached graph computes attn_pos_w x pe once per graph size; this runs the
// same audio through a fresh model with the cache on and off and requires the
// encoder output to match byte for byte.
//
// Usage:
//   test-pos-proj-cache-parity <parakeet.gguf> <wav> [n_gpu_layers]
//
// Exit 0 on success; non-zero on failure or invalid arguments.

#include "parakeet_ctc.h"
#include "mel_preprocess.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

void set_pos_proj_cache(bool on) {
#ifdef _WIN32
    _putenv_s("PARAKEET_POS_PROJ_CACHE", on ? "" : "0");
#else
    if (on) unsetenv("PARAKEET_POS_PROJ_CACHE"); else setenv("PARAKEET_POS_PROJ_CACHE", "0", 1);
#endif
}

int run_encoder_fresh(const std::string & gguf_path, const std::vector<float> & samples,
                      int n_gpu_layers, parakeet::EncoderOutputs & out) {
    using namespace parakeet;
    ParakeetCtcModel model;
    if (int rc = load_from_gguf(gguf_path, model, /*n_threads=*/0, n_gpu_layers, /*verbose=*/false); rc != 0) {
        std::fprintf(stderr, "  load_from_gguf failed rc=%d\n", rc);
        return 100 + rc;
    }
    std::vector<float> mel;
    int n_frames = 0;
    if (int rc = compute_log_mel(samples.data(), (int) samples.size(), model.mel_cfg, mel, n_frames); rc != 0) {
        std::fprintf(stderr, "  compute_log_mel failed rc=%d\n", rc);
        return 130 + rc;
    }
    if (int rc = run_encoder(model, mel.data(), n_frames, model.mel_cfg.n_mels, out, parakeet::EncoderRoute::ggml()); rc != 0) {
        std::fprintf(stderr, "  run_encoder failed rc=%d\n", rc);
        return 140 + rc;
    }
    return 0;
}

bool same_bytes(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <parakeet.gguf> <wav> [n_gpu_layers]\n", argv[0]);
        return 2;
    }
    const std::string gguf_path = argv[1];
    const std::string wav_path  = argv[2];
    const int n_gpu_layers = argc > 3 ? std::atoi(argv[3]) : 0;

    std::vector<float> samples;
    int sr = 0;
    if (int rc = parakeet::load_wav_mono_f32(wav_path, samples, sr); rc != 0) {
        std::fprintf(stderr, "  load_wav failed rc=%d\n", rc);
        return 120 + rc;
    }

    parakeet::EncoderOutputs cached, uncached;
    set_pos_proj_cache(true);
    if (int rc = run_encoder_fresh(gguf_path, samples, n_gpu_layers, cached); rc != 0) return rc;
    set_pos_proj_cache(false);
    if (int rc = run_encoder_fresh(gguf_path, samples, n_gpu_layers, uncached); rc != 0) return rc;
    set_pos_proj_cache(true);

    if (cached.n_enc_frames != uncached.n_enc_frames || !same_bytes(cached.encoder_out, uncached.encoder_out)) {
        std::fprintf(stderr, "FAIL: encoder output differs with the positional projection cache (%d vs %d frames)\n",
                     cached.n_enc_frames, uncached.n_enc_frames);
        return 1;
    }
    std::printf("PASS: pos-proj cache parity, %d frames x %d, n_gpu_layers=%d\n",
                cached.n_enc_frames, cached.d_model, n_gpu_layers);
    return 0;
}
