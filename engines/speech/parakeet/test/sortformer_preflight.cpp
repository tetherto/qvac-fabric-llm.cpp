#include "parakeet/engine.h"
#include <cstdio>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>

namespace {
constexpr int kCalls = 2;
constexpr int kGpuLayers = 999;

void write_probabilities(std::ostream & out, const parakeet::DiarizationResult & result) {
    const char * separator = "";
    for (float probability : result.speaker_probs) {
        out << separator << probability;
        separator = ",";
    }
}

void write_segments(std::ostream & out, const parakeet::DiarizationResult & result) {
    const char * separator = "";
    for (const auto & segment : result.segments) {
        out << separator << "{\"speaker\":" << segment.speaker_id
            << ",\"start\":" << segment.start_s << ",\"end\":" << segment.end_s << "}";
        separator = ",";
    }
}

void write_result(const std::string & path, const parakeet::DiarizationResult & result) {
    std::ofstream out(path);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out << std::setprecision(std::numeric_limits<double>::max_digits10)
        << "{\"coreml\":" << (result.encoder_used_coreml ? "true" : "false")
        << ",\"frames\":" << result.n_frames << ",\"speakers\":" << result.num_spks
        << ",\"stride\":" << result.frame_stride_s << ",\"probabilities\":[";
    write_probabilities(out, result);
    out << "],\"segments\":[";
    write_segments(out, result);
    out << "]}\n";
}

void run_calls(parakeet::Engine & engine, const char * audio, const char * prefix) {
    for (int call = 1; call <= kCalls; ++call) {
        const auto result = engine.diarize(audio);
        write_result(std::string(prefix) + "-" + std::to_string(call) + ".json", result);
    }
}
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s MODEL.gguf AUDIO.wav OUTPUT_PREFIX\n", argv[0]);
        return 2;
    }
    try {
        parakeet::EngineOptions options;
        options.model_gguf_path = argv[1];
        options.backend = "metal";
        options.n_gpu_layers = kGpuLayers;
        parakeet::Engine engine(options);
        run_calls(engine, argv[2], argv[3]);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
