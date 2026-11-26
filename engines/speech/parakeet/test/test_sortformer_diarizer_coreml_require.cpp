// test-sortformer-diarizer-coreml-require: Engine::diarize on a Sortformer
// v2.1 GGUF with no adjacent diarizer bundle and no cache directory, in a
// build that embeds the diarizer template. The template capacity (410
// frames) is status-only, unsupported(no_cache_dir). Under use = require an
// offline pass it covers throws, naming that stage and reason, instead of
// running on the ordinary encoder; under use = automatic the same call runs
// on ggml.
//
// Usage: test-sortformer-diarizer-coreml-require <v2.1 GGUF> <wav, at most 410 encoder frames>
// Never skips: the checks need the generator built in (macOS 15).

#include "parakeet/engine.h"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace {

bool g_passed = true;

void check(bool condition, const char * label) {
    std::printf("[%s] %s\n", condition ? "ok  " : "FAIL", label);
    g_passed = g_passed && condition;
}

bool contains(const std::string & text, const char * needle) { return text.find(needle) != std::string::npos; }

parakeet::EngineOptions engine_options(const std::string & gguf, parakeet::CoremlOptions::Use use) {
    parakeet::EngineOptions opts;
    opts.model_gguf_path = gguf;
    opts.coreml.use = use;  // no cache_dir: the template capacity is status-only
    return opts;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <audio.wav>\n", argv[0]);
        return 2;
    }
    const std::string gguf = argv[1];
    const std::string wav = argv[2];

    {
        parakeet::Engine engine(engine_options(gguf, parakeet::CoremlOptions::Use::require));
        const std::string status = engine.coreml_status_json();
        std::printf("require: %s\n", status.c_str());
        check(contains(status, "\"stage\":\"diarizer-410\"") && contains(status, "unsupported(no_cache_dir)"),
              "the template capacity is listed, unsupported(no_cache_dir)");
        std::string error;
        bool returned = false;
        try {
            (void) engine.diarize(wav);
            returned = true;
        } catch (const std::runtime_error & e) {
            error = e.what();
        }
        std::printf("require, offline pass: %s\n", returned ? "returned" : error.c_str());
        check(!returned, "use = require: an offline pass the capacity covers does not run on ggml");
        check(contains(error, "diarizer-410") && contains(error, "unsupported(no_cache_dir)"),
              "the exception names the diarizer stage and unsupported(no_cache_dir)");
    }
    {
        parakeet::Engine engine(engine_options(gguf, parakeet::CoremlOptions::Use::automatic));
        bool ran = false;
        try {
            const parakeet::DiarizationResult r = engine.diarize(wav);
            ran = r.n_frames > 0 && !r.encoder_used_coreml;
        } catch (const std::exception & e) {
            std::printf("automatic: %s\n", e.what());
        }
        check(ran, "use = automatic: the same pass runs on ggml");
    }

    std::printf("%s\n", g_passed ? "PASS" : "FAIL");
    return g_passed ? 0 : 1;
}
