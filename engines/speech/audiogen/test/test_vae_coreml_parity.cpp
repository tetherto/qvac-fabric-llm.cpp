// Core ML VAE decoder parity: decodes one deterministic latent through the
// Core ML sidecar and through the ggml decoder (forced via
// ACESTEP_COREML_DISABLE) and compares the audio by cosine similarity, so the
// production sidecar path is exercised end to end against the ggml reference.
//
// Skips (exit 77) when the build has no AUDIOGEN_COREML, when
// AUDIOGEN_TEST_MODELS_DIR is not staged, or when no compiled
// `vae-decoder.mlmodelc` sits next to the VAE GGUF.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "parity_checks.h"

namespace {
constexpr int SKIP_EXIT_CODE = 77;
}

#ifndef AUDIOGEN_USE_COREML

int main() {
    std::fprintf(stderr, "[coreml-parity] built without AUDIOGEN_COREML; skipping\n");
    return SKIP_EXIT_CODE;
}

#else

#include "audiogen-cpp/acestep/vae.h"
#include "vae_coreml_path.h"

#include <sys/stat.h>

namespace {

constexpr int    LATENT_CHANNELS = 64;
constexpr int    T_LATENT        = 1024;
constexpr double MIN_COSINE      = 0.999;

bool path_exists(const std::string & path) {
    struct stat st{};
    return stat(path.c_str(), &st) == 0;
}

std::string find_vae_gguf(const std::string & dir) {
    const char * candidates[] = {"vae-BF16.gguf", "vae-F16.gguf", "vae.gguf"};
    for (const char * name : candidates) {
        const std::string path = dir + "/" + name;
        if (path_exists(path)) return path;
    }
    return {};
}

// Deterministic latent in the amplitude range real DiT latents occupy.
std::vector<float> make_latent() {
    std::vector<float> latent((size_t) T_LATENT * LATENT_CHANNELS);
    uint32_t state = 0x2468ace1u;
    for (float & v : latent) {
        state = state * 1664525u + 1013904223u;
        v = ((float) (state >> 8) / (float) (1u << 24)) * 4.0f - 2.0f;
    }
    return latent;
}

double cosine(const std::vector<float> & a, const std::vector<float> & b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += (double) a[i] * (double) b[i];
        na  += (double) a[i] * (double) a[i];
        nb  += (double) b[i] * (double) b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
}

std::vector<float> decode_once(const std::string & gguf, const std::vector<float> & latent) {
    tts_cpp::acestep::VaeOptions opts;
    opts.with_encoder = false;
    auto vae = tts_cpp::acestep::Vae::load(gguf, opts);
    return vae->decode(latent, T_LATENT);
}

// A callback returning false must cancel and yield an empty result, never
// fall back to the ggml decode.
bool check_coreml_cancellation(const std::string & gguf, const std::vector<float> & latent) {
    tts_cpp::acestep::VaeOptions opts;
    opts.with_encoder = false;
    auto vae = tts_cpp::acestep::Vae::load(gguf, opts);
    int calls = 0;
    const std::vector<float> pcm = vae->decode(latent, T_LATENT, [&calls](int, int) {
        ++calls;
        return false;
    });
    if (!pcm.empty() || calls != 1) {
        std::fprintf(stderr, "[coreml-parity] FAIL: cancellation returned %zu samples after %d callbacks\n",
                     pcm.size(), calls);
        return false;
    }
    return true;
}

}  // namespace

int main() {
    const char * models_dir = std::getenv("AUDIOGEN_TEST_MODELS_DIR");
    if (models_dir == nullptr || *models_dir == '\0') {
        std::fprintf(stderr, "[coreml-parity] AUDIOGEN_TEST_MODELS_DIR not set; skipping\n");
        return SKIP_EXIT_CODE;
    }
    const std::string gguf = find_vae_gguf(models_dir);
    if (gguf.empty()) {
        std::fprintf(stderr, "[coreml-parity] no VAE GGUF under %s; skipping\n", models_dir);
        return SKIP_EXIT_CODE;
    }
    const std::string sidecar = tts_cpp::acestep::coreml_vae_sidecar_path(gguf);
    if (!path_exists(sidecar)) {
        std::fprintf(stderr, "[coreml-parity] no sidecar at %s; skipping\n", sidecar.c_str());
        return SKIP_EXIT_CODE;
    }

    const std::vector<float> latent = make_latent();

    // Strict mode with the sidecar disabled must fail the decode instead of
    // silently producing the ggml audio, or the parity below could compare
    // ggml against itself and pass vacuously.
    setenv("ACESTEP_COREML_DISABLE", "1", 1);
    setenv("ACESTEP_COREML_STRICT", "1", 1);
    if (!decode_once(gguf, latent).empty()) {
        std::fprintf(stderr, "[coreml-parity] FAIL: strict decode without a sidecar returned audio\n");
        return 1;
    }
    unsetenv("ACESTEP_COREML_STRICT");

    const std::vector<float> pcm_ggml = decode_once(gguf, latent);
    unsetenv("ACESTEP_COREML_DISABLE");

    // Strict for the Core ML leg: a sidecar init or predict failure now fails
    // the run instead of measuring the ggml fallback as Core ML.
    setenv("ACESTEP_COREML_STRICT", "1", 1);
    const std::vector<float> pcm_coreml = decode_once(gguf, latent);
    unsetenv("ACESTEP_COREML_STRICT");

    if (pcm_ggml.empty() || pcm_coreml.empty() || pcm_ggml.size() != pcm_coreml.size()) {
        std::fprintf(stderr, "[coreml-parity] FAIL: decode sizes ggml=%zu coreml=%zu\n",
                     pcm_ggml.size(), pcm_coreml.size());
        return 1;
    }

    const double cos = cosine(pcm_ggml, pcm_coreml);
    std::fprintf(stderr, "[coreml-parity] T_latent=%d samples=%zu cosine=%.7f (min %.4f)\n",
                 T_LATENT, pcm_ggml.size(), cos, MIN_COSINE);
    if (!audiogen_test::cosine_passes(cos, MIN_COSINE)) return 1;

    return check_coreml_cancellation(gguf, latent) ? 0 : 1;
}

#endif  // AUDIOGEN_USE_COREML
