#include "parakeet/engine.h"
#include "ggml-backend.h"

#include <cstdio>
#include <stdexcept>

namespace {

constexpr int kReuses = 3;

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

void compare(const parakeet::EngineResult & expected, const parakeet::EngineResult & actual) {
    require(actual.text == expected.text, "XDNA transcript differs from CPU");
    require(actual.token_ids == expected.token_ids, "XDNA tokens differ from CPU");
}

parakeet::EngineResult reference(parakeet::EngineOptions options, const char * wav) {
    options.backend = "cpu";
    parakeet::Engine engine(options);
    const auto result = engine.transcribe(wav);
    require(!result.text.empty() && !result.token_ids.empty(), "CPU speech fixture produced no transcript");
    return result;
}

void check_reuse(const parakeet::EngineOptions & options, const char * wav,
                 const parakeet::EngineResult & expected) {
    parakeet::Engine engine(options);
    require(engine.backend_device() == parakeet::BackendDevice::NPU, "XDNA is not reported as an NPU");
    require(engine.backend_name() == "XDNA0" || engine.backend_name() == "XDNA",
            "explicit XDNA request selected another backend");
    for (int iteration = 0; iteration < kReuses; ++iteration) {
        const auto actual = engine.transcribe(wav);
        compare(expected, actual);
        std::printf("XDNA reuse %d: encoder %.3f ms, total %.3f ms\n",
                    iteration, actual.encoder_ms, actual.total_ms);
    }
    parakeet::StreamingOptions streaming;
    streaming.chunk_ms = 60000;
    compare(expected, engine.transcribe_stream(wav, streaming, [](const parakeet::StreamingSegment &) {}));
}

}

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "Usage: test-xdna-transcription model.gguf speech.wav\n");
        return 1;
    }
    ggml_backend_load_all();
    if (!ggml_backend_dev_by_name("XDNA0")) {
        std::puts("XDNA0 is unavailable");
        return 3;
    }
    try {
        parakeet::EngineOptions options;
        options.model_gguf_path = argv[1];
        options.n_threads = 4;
        options.backend = "xdna";
        const auto expected = reference(options, argv[2]);
        check_reuse(options, argv[2], expected);
        options.backend = "XDNA0";
        parakeet::Engine recreated(options);
        compare(expected, recreated.transcribe(argv[2]));
        std::puts("Native XDNA CPU parity, graph reuse, streaming and recreation passed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
