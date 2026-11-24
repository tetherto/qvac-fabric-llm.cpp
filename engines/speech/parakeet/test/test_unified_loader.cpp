#include "parakeet_ctc.h"
#include "parakeet/engine.h"
#include "parakeet/streaming.h"

#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const std::vector<int32_t> kExpectedChunkFrames = {1, 2, 7, 13};
const std::vector<int32_t> kExpectedRightContextFrames = {0, 1, 2, 3, 4, 7, 13};

int load_model(const std::string & path, parakeet::ParakeetCtcModel & model) {
    const int rc = parakeet::load_from_gguf(path, model, 0, 0, false);
    if (rc != 0) {
        std::fprintf(stderr, "load_from_gguf failed: %d\n", rc);
    }
    return rc;
}

int check_family(const parakeet::ParakeetCtcModel & model) {
    if (model.model_type != parakeet::ParakeetModelType::RNNT ||
        std::string(parakeet::model_type_name(model.model_type)) != "rnnt") {
        std::fprintf(stderr, "Unified RNN-T family dispatch failed\n");
        return 2;
    }
    if (!model.encoder_cfg.att_dynamic_chunking || model.encoder_cfg.causal_downsampling) {
        std::fprintf(stderr, "Unified encoder geometry not detected\n");
        return 3;
    }
    return 0;
}

int check_streaming_metadata(const parakeet::ParakeetCtcModel & model) {
    const parakeet::UnifiedStreamingConfig & cfg = model.unified_cfg;
    if (!model.supports_streaming || !cfg.available) {
        std::fprintf(stderr, "Unified streaming metadata not exposed\n");
        return 4;
    }
    if (cfg.left_context_frames != 70 || cfg.cache_time_steps != 4) {
        std::fprintf(stderr, "unexpected Unified cache geometry: left=%d conv=%d\n",
                     cfg.left_context_frames, cfg.cache_time_steps);
        return 5;
    }
    if (cfg.allowed_chunk_frames != kExpectedChunkFrames ||
        cfg.allowed_right_context_frames != kExpectedRightContextFrames) {
        std::fprintf(stderr, "unexpected Unified operating points\n");
        return 6;
    }
    return 0;
}

bool rejects(const parakeet::ParakeetCtcModel & malformed) {
    try {
        parakeet::validate_unified_streaming_model(malformed);
    } catch (const std::runtime_error &) {
        return true;
    }
    return false;
}

int check_rejects_zero_left_context(const parakeet::ParakeetCtcModel & model) {
    parakeet::ParakeetCtcModel malformed = model;
    malformed.unified_cfg.left_context_frames = 0;
    if (!rejects(malformed)) {
        std::fprintf(stderr, "zero left context was accepted\n");
        return 7;
    }
    return 0;
}

int check_rejects_wrong_convolution_cache(const parakeet::ParakeetCtcModel & model) {
    parakeet::ParakeetCtcModel malformed = model;
    malformed.unified_cfg.cache_time_steps = 8;
    if (!rejects(malformed)) {
        std::fprintf(stderr, "wrong convolution cache size was accepted\n");
        return 8;
    }
    return 0;
}

int check_rejects_empty_operating_points(const parakeet::ParakeetCtcModel & model) {
    parakeet::ParakeetCtcModel malformed = model;
    malformed.unified_cfg.allowed_chunk_frames.clear();
    if (!rejects(malformed)) {
        std::fprintf(stderr, "empty chunk list was accepted\n");
        return 9;
    }
    return 0;
}

int check_rejects_negative_right_context(const parakeet::ParakeetCtcModel & model) {
    parakeet::ParakeetCtcModel malformed = model;
    malformed.unified_cfg.allowed_right_context_frames = {0, -1};
    if (!rejects(malformed)) {
        std::fprintf(stderr, "negative right context was accepted\n");
        return 10;
    }
    return 0;
}

int check_rejects_offline_attention_style(const parakeet::ParakeetCtcModel & model) {
    parakeet::ParakeetCtcModel malformed = model;
    malformed.encoder_cfg.att_dynamic_chunking = false;
    if (!rejects(malformed)) {
        std::fprintf(stderr, "offline attention style was accepted for streaming\n");
        return 11;
    }
    return 0;
}

int check_rejects_oversized_left_context(const parakeet::ParakeetCtcModel & model) {
    parakeet::ParakeetCtcModel malformed = model;
    malformed.unified_cfg.left_context_frames = std::numeric_limits<int>::max();
    if (!rejects(malformed)) {
        std::fprintf(stderr, "oversized left context was accepted\n");
        return 13;
    }
    return 0;
}

int check_rejects_oversized_operating_points(const parakeet::ParakeetCtcModel & model) {
    parakeet::ParakeetCtcModel malformed = model;
    malformed.unified_cfg.allowed_chunk_frames = {std::numeric_limits<int32_t>::max()};
    if (!rejects(malformed)) {
        std::fprintf(stderr, "oversized chunk size was accepted\n");
        return 14;
    }
    malformed = model;
    malformed.unified_cfg.allowed_right_context_frames = {std::numeric_limits<int32_t>::max()};
    if (!rejects(malformed)) {
        std::fprintf(stderr, "oversized right context was accepted\n");
        return 15;
    }
    return 0;
}

int check_rejects_too_many_operating_points(const parakeet::ParakeetCtcModel & model) {
    parakeet::ParakeetCtcModel malformed = model;
    malformed.unified_cfg.allowed_chunk_frames.assign(1024, 7);
    if (!rejects(malformed)) {
        std::fprintf(stderr, "unbounded operating point list was accepted\n");
        return 16;
    }
    return 0;
}

parakeet::StreamingOptions streaming_options(int chunk_ms, int right_lookahead_ms) {
    parakeet::StreamingOptions options;
    options.chunk_ms = chunk_ms;
    options.right_lookahead_ms = right_lookahead_ms;
    return options;
}

int feed_silence_and_finalize(parakeet::Engine & engine, const parakeet::StreamingOptions & options) {
    const parakeet::StreamingCallback callback = [](const parakeet::StreamingSegment &) {};
    auto session = engine.stream_start(options, callback);
    const std::vector<float> silence(16000, 0.0f);
    session->feed_pcm_f32(silence.data(), static_cast<int>(silence.size()));
    session->finalize();
    try {
        session->feed_pcm_f32(silence.data(), static_cast<int>(silence.size()));
    } catch (const std::runtime_error &) {
        return 0;
    }
    std::fprintf(stderr, "feed after finalize was accepted\n");
    return 12;
}

int check_engine_stream_creation(const std::string & path) {
    parakeet::EngineOptions options;
    options.model_gguf_path = path;
    options.prewarm = false;
    parakeet::Engine engine(options);
    if (engine.model_type() != "rnnt" || !engine.is_transcription_model()) {
        std::fprintf(stderr, "Engine did not expose the RNN-T family\n");
        return 13;
    }
    try {
        if (int rc = feed_silence_and_finalize(engine, streaming_options(1000, 2000)); rc != 0) return rc;
        if (int rc = feed_silence_and_finalize(engine, streaming_options(40, 0)); rc != 0) return rc;
        if (int rc = feed_silence_and_finalize(engine, streaming_options(560, 560)); rc != 0) return rc;
    } catch (const std::runtime_error & error) {
        std::fprintf(stderr, "Unified stream session failed: %s\n", error.what());
        return 14;
    }
    return 0;
}

int run_checks(const std::string & path) {
    parakeet::ParakeetCtcModel model;
    if (int rc = load_model(path, model); rc != 0) return 1;
    if (int rc = check_family(model); rc != 0) return rc;
    if (int rc = check_streaming_metadata(model); rc != 0) return rc;
    if (int rc = check_rejects_zero_left_context(model); rc != 0) return rc;
    if (int rc = check_rejects_wrong_convolution_cache(model); rc != 0) return rc;
    if (int rc = check_rejects_empty_operating_points(model); rc != 0) return rc;
    if (int rc = check_rejects_negative_right_context(model); rc != 0) return rc;
    if (int rc = check_rejects_offline_attention_style(model); rc != 0) return rc;
    if (int rc = check_rejects_oversized_left_context(model); rc != 0) return rc;
    if (int rc = check_rejects_oversized_operating_points(model); rc != 0) return rc;
    if (int rc = check_rejects_too_many_operating_points(model); rc != 0) return rc;
    return check_engine_stream_creation(path);
}

}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <parakeet-unified.gguf>\n", argv[0]);
        return 2;
    }
    const int rc = run_checks(argv[1]);
    if (rc == 0) {
        std::fprintf(stderr, "Unified loader tests passed\n");
    } else {
        std::fprintf(stderr, "Unified loader tests failed (%d)\n", rc);
    }
    return rc;
}
