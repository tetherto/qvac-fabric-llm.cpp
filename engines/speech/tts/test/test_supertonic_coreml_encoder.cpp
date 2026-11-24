// Real-model encoder parity and dispatch. Stage an F32/F16 GGUF with its
// compiled encoder sidecar and set SUPERTONIC_COREML_ENCODER_TEST_GGUF.
#include <cstdio>
#include <cstdlib>

#ifndef TTS_CPP_USE_COREML
int main() { std::puts("SKIP: built without Core ML"); return 77; }
#else
#include "supertonic_coreml_encoder.h"
#include "supertonic_internal.h"
#include "tts-cpp/supertonic/engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

using namespace tts_cpp::supertonic::detail;

namespace {
void check(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error(message);
}
struct Model {
    supertonic_model value;
    ~Model() { free_supertonic_model(value); }
};
struct Env {
    std::string key, old;
    bool existed;
    Env(const char * name, const char * value) : key(name), existed(std::getenv(name) != nullptr) {
        if (existed) old = std::getenv(name);
        setenv(name, value, 1);
    }
    ~Env() { if (existed) setenv(key.c_str(), old.c_str(), 1); else unsetenv(key.c_str()); }
};
double cosine(const std::vector<float> & a, const std::vector<float> & b, double & max_error) {
    check(a.size() == b.size() && !a.empty(), "embedding shape mismatch");
    double dot = 0, aa = 0, bb = 0;
    max_error = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        check(std::isfinite(a[i]) && std::isfinite(b[i]), "nonfinite embedding");
        dot += double(a[i]) * b[i]; aa += double(a[i]) * a[i]; bb += double(b[i]) * b[i];
        max_error = std::max(max_error, std::abs(double(a[i]) - b[i]));
    }
    return dot / (std::sqrt(aa * bb) + 1e-12);
}
}

int main() {
    const char * path = std::getenv("SUPERTONIC_COREML_ENCODER_TEST_GGUF");
    if (!path || !*path) { std::puts("SKIP: SUPERTONIC_COREML_ENCODER_TEST_GGUF not set"); return 77; }
    try {
        Model owned;
        auto & model = owned.value;
        check(load_supertonic_gguf(path, model, 0), "GGUF load failed");
        check(model.encoder_on_coreml && model.coreml_encoder, "encoder sidecar not loaded");
        supertonic_set_n_threads(model, 4);
        const int capacity = supertonic_coreml_encoder_capacity(model.coreml_encoder);
        check(!model.voices.empty(), "no voices");
        const auto & voice = model.voices.begin()->second;
        std::vector<float> style((size_t) ggml_nelements(voice.ttl));
        ggml_backend_tensor_get(voice.ttl, style.data(), 0, style.size() * sizeof(float));
        const auto * table = require_source_tensor(model, "text_encoder:tts.ttl.text_encoder.text_embedder.char_embedder.weight");
        for (int length : {1, std::min(17, capacity), capacity, capacity + 1}) {
            std::vector<int64_t> ids(length);
            for (int i = 0; i < length; ++i) ids[i] = (i * 7 + 3) % table->ne[1];
            std::vector<float> got, want;
            std::string error, backend;
            check(supertonic_text_encoder_forward_one_graph_ggml(model, ids.data(), length, style.data(), want, &error), error);
            check(supertonic_text_encoder_forward_ggml(model, ids.data(), length, style.data(), got, &error, &backend), error);
            check((backend != "ggml") == (length <= capacity), "wrong dispatch at token length " + std::to_string(length));
            double max_error;
            double sim = cosine(got, want, max_error);
            std::printf("length=%d backend=%s cosine=%.8f max_abs=%.6f\n", length, backend.c_str(), sim, max_error);
            // ANE uses FP16 intermediates; v3's dilated stack amplifies their
            // rounding more than v2. Bound both direction and worst component.
            check(sim >= 0.9999 && max_error <= 0.1, "encoder parity failed");
            if (length <= capacity) {
                // Alternate styles on the same context and repeat the original
                // request: catches stale cached I/O and baked-in trace inputs.
                auto other = style;
                for (float & v : other) v *= 0.5f;
                check(supertonic_text_encoder_forward_ggml(model, ids.data(), length, other.data(), got, &error), error);
                check(supertonic_text_encoder_forward_one_graph_ggml(model, ids.data(), length, other.data(), want, &error), error);
                check(cosine(got, want, max_error) >= 0.9999 && max_error <= 0.1, "alternate style parity failed");
                // Warm stage-only timings, including host I/O on both paths.
                std::vector<double> times[2];
                for (int run = 0; run < 12; ++run) for (int route = 0; route < 2; ++route) {
                    const auto start = std::chrono::steady_clock::now();
                    bool ok = route == 0
                        ? supertonic_text_encoder_forward_one_graph_ggml(model, ids.data(), length, style.data(), got, &error)
                        : supertonic_text_encoder_forward_ggml(model, ids.data(), length, style.data(), got, &error);
                    check(ok, error);
                    if (run >= 2) times[route].push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count());
                }
                for (auto & t : times) std::sort(t.begin(), t.end());
                std::printf("length=%d ggml_cpu_ms=%.3f coreml_ms=%.3f\n", length, times[0][5], times[1][5]);
            } else {
                Env strict("SUPERTONIC_COREML_ENCODER_STRICT", "1");
                check(!supertonic_text_encoder_forward_ggml(model, ids.data(), length, style.data(), got, &error), "strict overflow unexpectedly succeeded");
            }
        }
        std::vector<float> output = {123};
        const int64_t bad = table->ne[1];
        check(!supertonic_coreml_encoder_encode(model.coreml_encoder, &bad, 1, style.data(), output), "invalid token accepted");
        check(output == std::vector<float>{123}, "failure modified output");
        check(!supertonic_coreml_encoder_encode(model.coreml_encoder, &bad, 0, style.data(), output), "empty input accepted");
        check(supertonic_coreml_encoder_init(path, table->ne[1]) == nullptr, "malformed sidecar accepted");
        {
            Env disabled("SUPERTONIC_COREML_ENCODER_DISABLE", "1");
            Model plain;
            check(load_supertonic_gguf(path, plain.value, 0), "disabled GGUF load failed");
            check(!plain.value.encoder_on_coreml && !plain.value.coreml_encoder, "disable ignored");
        }
        // The public engine reports the actual path and the sidecar load state.
        tts_cpp::supertonic::EngineOptions options;
        options.model_gguf_path = path;
        options.n_gpu_layers = 0;
        tts_cpp::supertonic::Engine engine(options);
        check(engine.encoder_on_coreml(), "public load status false");
        auto result = engine.synthesize("Hello.");
        check(!result.pcm.empty() && result.text_encoder_synthesis_backend != "ggml", "public synthesis bypassed encoder");
        options.stream_chunk_tokens = 20;
        tts_cpp::supertonic::Engine streaming(options);
        int chunks = 0;
        result = streaming.synthesize("Hello there. This is another short sentence. Here is the final sentence.",
            [&](const float *, std::size_t, int, bool) { ++chunks; });
        check(chunks >= 2 && result.text_encoder_synthesis_backend.rfind("coreml-", 0) == 0,
              "streaming lost the encoder backend report");
        std::puts("test_supertonic_coreml_encoder: OK");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
#endif
