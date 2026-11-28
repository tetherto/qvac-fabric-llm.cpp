// Supertonic Core ML vocoder sidecar: parity against the ggml vocoder graph
// through the production dispatch, the load-status and per-call backend
// reports, and the fallbacks (disabled, strict without a sidecar, invalid
// sidecar). Skips (77) without TTS_CPP_COREML or a staged
// SUPERTONIC_COREML_TEST_MODELS_DIR holding a Supertonic GGUF with a compiled
// `<model>-vocoder.mlmodelc` next to it.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
constexpr int SKIP_EXIT_CODE = 77;
}

#ifndef TTS_CPP_USE_COREML

int main() {
    std::fprintf(stderr, "[supertonic-coreml-parity] built without TTS_CPP_COREML; skipping\n");
    return SKIP_EXIT_CODE;
}

#else

#include "supertonic_coreml_path.h"
#include "supertonic_coreml_vocoder.h"
#include "supertonic_internal.h"
#include "tts-cpp/supertonic/engine.h"

#include <filesystem>
#include <fstream>

using namespace tts_cpp::supertonic::detail;
namespace fs = std::filesystem;

namespace {

constexpr double MIN_COSINE = 0.999;
constexpr int    N_THREADS  = 4;

int g_failures = 0;

void fail(const std::string & what) {
    std::fprintf(stderr, "[supertonic-coreml-parity] FAIL: %s\n", what.c_str());
    ++g_failures;
}

void expect(bool ok, const std::string & what) {
    if (!ok) fail(what);
}

struct scoped_env {
    const char * name;
    scoped_env(const char * n, const char * value) : name(n) { setenv(name, value, 1); }
    ~scoped_env() { unsetenv(name); }
};

std::string find_supertonic_gguf(const std::string & dir) {
    const char * candidates[] = {"supertonic2.gguf",      "supertonic3.gguf",
                                 "supertonic.gguf",       "supertonic2-f32.gguf",
                                 "supertonic2-f16.gguf",  "supertonic2-q8_0.gguf",
                                 "supertonic3-f32.gguf",  "supertonic3-q8_0.gguf"};
    for (const char * name : candidates) {
        const std::string path = dir + "/" + name;
        if (fs::exists(path)) return path;
    }
    return {};
}

std::vector<float> make_latent(int channels, int latent_len, uint32_t seed) {
    std::vector<float> latent((size_t) channels * latent_len);
    uint32_t state = seed;
    for (float & v : latent) {
        state = state * 1664525u + 1013904223u;
        // Uniform in [-2, 2): wide enough to exercise the stack, bounded so
        // the fp16 sidecar stays in a well-conditioned range.
        v = ((float) (state >> 8) / 8388608.0f) * 2.0f - 2.0f;
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

struct owned_model {
    supertonic_model model;
    bool loaded = false;
    ~owned_model() { if (loaded) free_supertonic_model(model); }
};

bool load(const std::string & gguf, owned_model & owned) {
    try {
        owned.loaded = load_supertonic_gguf(gguf, owned.model, /*n_gpu_layers=*/0);
        if (owned.loaded) supertonic_set_n_threads(owned.model, N_THREADS);
    } catch (const std::exception & e) {
        fail("load_supertonic_gguf(" + gguf + "): " + e.what());
        return false;
    }
    if (!owned.loaded) fail("load_supertonic_gguf(" + gguf + ") returned false");
    return owned.loaded;
}

void check_parity(const supertonic_model & ggml_model, const supertonic_model & coreml_model,
                  int latent_len, uint32_t seed) {
    const std::string tag = "latent_len=" + std::to_string(latent_len);
    const std::vector<float> latent =
        make_latent(ggml_model.hparams.latent_channels, latent_len, seed);

    std::vector<float> wav_ggml, wav_coreml;
    std::string error, backend;
    if (!supertonic_vocoder_forward_ggml(ggml_model, latent.data(), latent_len, wav_ggml,
                                         &error, &backend)) {
        fail(tag + ": ggml vocoder failed: " + error);
        return;
    }
    expect(backend == "ggml", tag + ": ggml reference reports backend " + backend);

    scoped_env strict("SUPERTONIC_COREML_STRICT", "1");
    if (!supertonic_vocoder_forward_ggml(coreml_model, latent.data(), latent_len, wav_coreml,
                                         &error, &backend)) {
        fail(tag + ": Core ML vocoder failed: " + error);
        return;
    }
    expect(backend.rfind("coreml", 0) == 0, tag + ": Core ML run reports backend " + backend);
    expect(wav_coreml.size() == wav_ggml.size(),
           tag + ": size mismatch (" + std::to_string(wav_coreml.size()) + " vs " +
           std::to_string(wav_ggml.size()) + ")");
    if (wav_coreml.size() != wav_ggml.size()) return;

    const double cos = cosine(wav_ggml, wav_coreml);
    std::fprintf(stderr, "[supertonic-coreml-parity] %s cosine=%.6f\n", tag.c_str(), cos);
    expect(cos >= MIN_COSINE, tag + ": cosine " + std::to_string(cos) + " below " +
           std::to_string(MIN_COSINE));
}

void check_engine_report(const std::string & gguf) {
    using tts_cpp::supertonic::Engine;
    using tts_cpp::supertonic::EngineOptions;
    using tts_cpp::supertonic::SynthesisResult;
    try {
        EngineOptions opts;
        opts.model_gguf_path = gguf;
        opts.n_threads = N_THREADS;
        Engine engine(opts);
        expect(engine.vocoder_on_coreml(), "engine: vocoder_on_coreml() is false");
        SynthesisResult result = engine.synthesize("Core ML parity check.");
        expect(result.vocoder_synthesis_backend.rfind("coreml", 0) == 0,
               "engine: SynthesisResult::vocoder_synthesis_backend is " + result.vocoder_synthesis_backend);
        expect(!result.pcm.empty(), "engine: empty PCM");

        // Streaming with every chunk on the sidecar keeps the sidecar label
        // rather than degrading to "mixed".
        EngineOptions stream_opts = opts;
        stream_opts.stream_chunk_tokens = 40;
        Engine stream_engine(stream_opts);
        int chunks = 0;
        SynthesisResult streamed = stream_engine.synthesize(
            "The quick brown fox jumps over the lazy dog. The curious cat watches "
            "from the warm windowsill. A lone bell rings across the quiet valley.",
            [&](const float *, std::size_t, int, bool) { ++chunks; });
        expect(chunks >= 2, "engine: streaming produced " + std::to_string(chunks) +
                            " chunk(s); the multi-chunk label path never ran");
        expect(streamed.vocoder_synthesis_backend.rfind("coreml", 0) == 0,
               "engine: streamed vocoder_synthesis_backend is " +
               streamed.vocoder_synthesis_backend);
    } catch (const std::exception & e) {
        fail(std::string("engine synthesis: ") + e.what());
    }
}

// A CPU-backed engine (n_gpu_layers = 0) with a sidecar attached must warm it:
// the ctor pre-warm's vocoder pass shows up as a "vocoder,coreml" profile row
// before any operator-visible synthesize().
void check_cpu_warm_up_reaches_sidecar(const std::string & gguf) {
    using tts_cpp::supertonic::Engine;
    using tts_cpp::supertonic::EngineOptions;
    std::error_code ec;
    const fs::path csv = fs::temp_directory_path(ec) / "supertonic-coreml-warmup.csv";
    fs::remove(csv, ec);
    supertonic_profile_csv_set_path(csv.string().c_str());
    try {
        EngineOptions opts;
        opts.model_gguf_path = gguf;
        opts.n_threads = N_THREADS;
        opts.prewarm_text = "Warm up the vocoder sidecar.";
        Engine engine(opts);
        expect(engine.vocoder_on_coreml(), "warm-up: sidecar not attached");
    } catch (const std::exception & e) {
        fail(std::string("warm-up engine: ") + e.what());
    }
    supertonic_profile_csv_set_path(nullptr);
    std::ifstream rows(csv);
    std::string line;
    bool warmed = false;
    while (std::getline(rows, line)) {
        if (line.rfind("vocoder,coreml,", 0) == 0) {
            warmed = true;
            break;
        }
    }
    expect(warmed, "warm-up: the ctor pre-warm left the Core ML vocoder cold "
                   "(no vocoder,coreml profile row)");
    fs::remove(csv, ec);
}

void check_disabled_fallback(const std::string & gguf, int latent_len) {
    scoped_env disable("SUPERTONIC_COREML_DISABLE", "1");
    owned_model owned;
    if (!load(gguf, owned)) return;
    expect(!owned.model.vocoder_on_coreml,
           "SUPERTONIC_COREML_DISABLE: sidecar still attached");
    const std::vector<float> latent =
        make_latent(owned.model.hparams.latent_channels, latent_len, 7u);
    std::vector<float> wav;
    std::string error, backend;
    expect(supertonic_vocoder_forward_ggml(owned.model, latent.data(), latent_len, wav,
                                           &error, &backend),
           "SUPERTONIC_COREML_DISABLE: ggml vocoder failed: " + error);
    expect(backend == "ggml", "SUPERTONIC_COREML_DISABLE: backend is " + backend);

    // Strict mode with no sidecar attached must fail rather than silently
    // measure ggml.
    scoped_env strict("SUPERTONIC_COREML_STRICT", "1");
    std::vector<float> refused;
    expect(!supertonic_vocoder_forward_ggml(owned.model, latent.data(), latent_len, refused,
                                            &error, &backend),
           "strict without a sidecar: forward did not fail");
}

// A sidecar path that exists but is not a Core ML model must leave the load
// on ggml and the forward on the fallback.
void check_invalid_sidecar(const std::string & gguf, int latent_len) {
    std::error_code ec;
    const fs::path dir = fs::temp_directory_path(ec) / "supertonic-coreml-invalid";
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    if (ec) {
        fail("invalid sidecar: cannot stage " + dir.string());
        return;
    }
    const fs::path linked = dir / "supertonic-invalid.gguf";
    fs::create_symlink(fs::absolute(gguf), linked, ec);
    if (ec) {
        fail("invalid sidecar: cannot link the GGUF: " + ec.message());
        return;
    }
    fs::create_directories(coreml_vocoder_sidecar_path(linked.string()), ec);

    owned_model owned;
    if (!load(linked.string(), owned)) return;
    expect(!owned.model.vocoder_on_coreml, "invalid sidecar: load-status reports attached");
    const std::vector<float> latent =
        make_latent(owned.model.hparams.latent_channels, latent_len, 5u);
    std::vector<float> wav;
    std::string error, backend;
    expect(supertonic_vocoder_forward_ggml(owned.model, latent.data(), latent_len, wav,
                                           &error, &backend),
           "invalid sidecar: ggml fallback failed: " + error);
    expect(backend == "ggml", "invalid sidecar: backend is " + backend);
    fs::remove_all(dir, ec);
}

}  // namespace

int main() {
    const char * env_dir = std::getenv("SUPERTONIC_COREML_TEST_MODELS_DIR");
    if (env_dir == nullptr) {
        std::fprintf(stderr,
                     "[supertonic-coreml-parity] SUPERTONIC_COREML_TEST_MODELS_DIR is not "
                     "set; skipping\n");
        return SKIP_EXIT_CODE;
    }
    const std::string dir = env_dir;
    const std::string gguf = find_supertonic_gguf(dir);
    if (gguf.empty()) {
        std::fprintf(stderr,
                     "[supertonic-coreml-parity] no Supertonic GGUF under %s; skipping\n",
                     dir.c_str());
        return SKIP_EXIT_CODE;
    }
    const std::string sidecar = coreml_vocoder_sidecar_path(gguf);
    if (!fs::exists(sidecar)) {
        std::fprintf(stderr, "[supertonic-coreml-parity] no sidecar at %s; skipping\n",
                     sidecar.c_str());
        return SKIP_EXIT_CODE;
    }

    owned_model ggml_owned;
    {
        scoped_env disable("SUPERTONIC_COREML_DISABLE", "1");
        if (!load(gguf, ggml_owned)) return 1;
    }
    expect(!ggml_owned.model.vocoder_on_coreml, "disabled load still attached the sidecar");

    owned_model coreml_owned;
    if (!load(gguf, coreml_owned)) return 1;
    if (!coreml_owned.model.vocoder_on_coreml) {
        fail("sidecar exists at " + sidecar + " but did not attach");
        return 1;
    }

    const int window =
        (int) supertonic_coreml_vocoder_window_frames(coreml_owned.model.coreml_vocoder);
    const int context = supertonic_coreml_vocoder_context_frames(coreml_owned.model);
    std::fprintf(stderr, "[supertonic-coreml-parity] %s window=%d context=%d\n",
                 sidecar.c_str(), window, context);
    expect(window > context, "exported window does not clear the causal context");

    // Shorter than the window (single padded call), exactly the window, and
    // long enough to need several stitched windows.
    check_parity(ggml_owned.model, coreml_owned.model, std::max(1, window / 2), 11u);
    check_parity(ggml_owned.model, coreml_owned.model, window, 13u);
    check_parity(ggml_owned.model, coreml_owned.model, 3 * window - window / 3, 17u);

    check_disabled_fallback(gguf, std::max(1, window / 2));
    check_invalid_sidecar(gguf, std::max(1, window / 2));
    check_engine_report(gguf);
    check_cpu_warm_up_reaches_sidecar(gguf);

    if (g_failures == 0) {
        std::printf("test_supertonic_coreml_parity: OK\n");
        return 0;
    }
    std::fprintf(stderr, "test_supertonic_coreml_parity: %d failure(s)\n", g_failures);
    return 1;
}

#endif  // TTS_CPP_USE_COREML
