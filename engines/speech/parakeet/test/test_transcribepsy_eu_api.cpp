#include "parakeet/engine.h"
#include "parakeet/log.h"
#include "test_utils.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
constexpr int kSampleRate = 16000;
constexpr int kGateSamples = 20 * kSampleRate;
constexpr int kChunkSamples = 2 * kSampleRate;

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

struct RouteLog {
    std::string stream_route;
    std::string gate_route;
    int gates = 0;

    static void capture(ggml_log_level, const char * text, void * user) {
        auto & log = *static_cast<RouteLog *>(user);
        const std::string message(text);
        const auto route = message.find("route=");
        if (route == std::string::npos) return;
        const std::string value = message.substr(route + 6, message.find(',', route) - route - 6);
        if (message.find("EU stream encoder:") != std::string::npos) log.stream_route = value;
        if (message.find("EU gate: p(") != std::string::npos) {
            log.gate_route = value;
            ++log.gates;
        }
    }

    RouteLog() { parakeet_log_set(capture, this); }
    ~RouteLog() { parakeet_log_set(nullptr, nullptr); }
};

std::vector<float> read_samples(const char * path) {
    std::vector<float> pcm;
    int rate = 0;
    require(parakeet_test::load_wav_pcm16le_mono(path, pcm, rate) && rate == kSampleRate,
            "could not read 16 kHz fixture");
    return pcm;
}

parakeet::StreamingOptions stream_options() {
    parakeet::StreamingOptions options;
    options.left_context_ms = 1000;
    options.right_lookahead_ms = 0;
    return options;
}

struct StreamOutput {
    std::vector<int32_t> tokens;
    int segments = 0;
};

std::unique_ptr<parakeet::StreamSession> start_stream(
        parakeet::Engine & engine, RouteLog & log, StreamOutput & output,
        const char * expected_route) {
    return engine.stream_start(stream_options(), [&, expected_route](const auto & segment) {
        require(log.stream_route == expected_route, "stream used another call's EU route");
        require(!segment.encoder_used_coreml, "native API test unexpectedly used Core ML");
        output.tokens.insert(output.tokens.end(), segment.token_ids.begin(), segment.token_ids.end());
        ++output.segments;
    });
}

StreamOutput stream_reference(parakeet::Engine & engine, RouteLog & log,
                              const std::vector<float> & pcm, const char * route) {
    StreamOutput output;
    auto session = start_stream(engine, log, output, route);
    session->feed_pcm_f32(pcm.data(), kGateSamples);
    session->feed_pcm_f32(pcm.data() + kGateSamples, (int) pcm.size() - kGateSamples);
    session->finalize();
    require(!output.tokens.empty(), "reference stream emitted no tokens");
    return output;
}

void check_offline_interleave(parakeet::Engine & engine, RouteLog & log,
                              const std::vector<float> & english, const std::vector<float> & long_french,
                              const StreamOutput & french_reference) {
    StreamOutput output;
    auto session = start_stream(engine, log, output, "other");
    require(session->options().chunk_ms == 2000, "automatic full-EU chunk is not 2000 ms");
    const int gates_before = log.gates;
    session->feed_pcm_f32(long_french.data(), kGateSamples - 1);
    require(output.segments == 0 && log.gates == gates_before,
            "automatic stream emitted or gated before 20 seconds");
    session->feed_pcm_f32(long_french.data() + kGateSamples - 1, 1);
    require(output.segments > 0 && log.gates == gates_before + 1 && log.gate_route == "other",
            "automatic stream did not select French at 20 seconds");
    require(!engine.transcribe_samples(english.data(), (int) english.size(), kSampleRate).text.empty(),
            "interleaved English offline transcription is empty");
    require(log.gate_route == "en", "offline English did not exercise the opposite route");
    session->feed_pcm_f32(long_french.data() + kGateSamples, (int) long_french.size() - kGateSamples);
    session->finalize();
    require(output.tokens == french_reference.tokens, "offline call changed the session transcript");
    const int finalized_segments = output.segments;
    session->finalize();
    require(output.segments == finalized_segments, "repeated finalize emitted another segment");
}

void check_alternating_sessions(parakeet::Engine & engine, RouteLog & log,
                                const std::vector<float> & long_english,
                                const std::vector<float> & long_french,
                                const StreamOutput & english_reference,
                                const StreamOutput & french_reference) {
    StreamOutput french_output, english_output;
    auto french_session = start_stream(engine, log, french_output, "other");
    auto english_session = start_stream(engine, log, english_output, "en");
    french_session->feed_pcm_f32(long_french.data(), kGateSamples);
    english_session->feed_pcm_f32(long_english.data(), kGateSamples);
    french_session->feed_pcm_f32(long_french.data() + kGateSamples, (int) long_french.size() - kGateSamples);
    english_session->feed_pcm_f32(long_english.data() + kGateSamples, (int) long_english.size() - kGateSamples);
    french_session->finalize();
    english_session->finalize();
    require(french_output.tokens == french_reference.tokens && english_output.tokens == english_reference.tokens,
            "alternating sessions changed their transcripts");
}

void check_short_finalize(parakeet::Engine & engine, RouteLog & log,
                          const std::vector<float> & french) {
    StreamOutput short_output;
    auto short_session = start_stream(engine, log, short_output, "other");
    const int short_gates_before = log.gates;
    short_session->feed_pcm_f32(french.data(), (int) french.size());
    require(short_output.segments == 0 && log.gates == short_gates_before,
            "short automatic stream gated before finalize");
    short_session->finalize();
    require(short_output.segments > 0 && log.gates == short_gates_before + 1 && log.gate_route == "other",
            "short automatic stream did not gate on finalize");
    auto empty = engine.stream_start(stream_options(), [](const auto &) {
        throw std::runtime_error("empty stream emitted a segment");
    });
    empty->finalize();
}

void check_explicit_session(const parakeet::EngineOptions & automatic, RouteLog & log,
                            const std::vector<float> & english, const char * language) {
    auto options = automatic;
    options.language = language;
    parakeet::Engine engine(options);
    StreamOutput output;
    auto session = start_stream(engine, log, output, language);
    const int gates_before = log.gates;
    session->feed_pcm_f32(english.data(), kChunkSamples);
    require(output.segments == 1 && log.gates == gates_before,
            "explicit language did not emit its first chunk immediately");
    session->finalize();
    auto explicit_options = stream_options();
    explicit_options.chunk_ms = 1000;
    auto explicit_session = engine.stream_start(explicit_options, [](const auto &) {});
    require(explicit_session->options().chunk_ms == 1000, "explicit 1000 ms chunk was overridden");
    explicit_session->cancel();
    explicit_session->finalize();
}
}  // namespace

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s model.gguf english.wav french.wav\n", argv[0]);
        return 2;
    }
    try {
#ifdef _WIN32
        _putenv_s("PARAKEET_COREML_DISABLE", "1");
#else
        setenv("PARAKEET_COREML_DISABLE", "1", 1);
#endif
        RouteLog log;
        parakeet::EngineOptions options;
        options.model_gguf_path = argv[1];
        options.n_threads = 4;
        options.backend = "cpu";
        options.n_gpu_layers = 0;
        options.language = "auto";
        parakeet::Engine engine(options);
        for (int i = 2; i < argc; ++i) {
            const std::string wav = argv[i];
            const auto direct = engine.transcribe(wav);
            const auto via_mel = engine.transcribe_mel(engine.compute_mel_wav(wav));
            if (direct.text.empty() || direct.text != via_mel.text ||
                direct.token_ids != via_mel.token_ids) {
                std::fprintf(stderr, "TranscribePsy EU API mismatch for %s\n", argv[i]);
                return 1;
            }
        }
        auto malformed = engine.compute_mel_wav(argv[2]);
        malformed.eu_gate_frames = 2001;
        malformed.eu_gate_mel = {0.0f};
        try {
            engine.transcribe_mel(std::move(malformed));
            std::fprintf(stderr, "TranscribePsy EU accepted an undersized gate mel\n");
            return 1;
        } catch (const std::runtime_error & error) {
            if (std::string(error.what()).find("invalid EU gate mel buffer") == std::string::npos)
                throw;
        }
        const auto english = read_samples(argv[2]);
        const auto french = read_samples(argv[3]);
        const auto long_french = parakeet_test::tile_to_seconds(french, kSampleRate, 22);
        const auto long_english = parakeet_test::tile_to_seconds(english, kSampleRate, 22);
        const auto french_reference = stream_reference(engine, log, long_french, "other");
        const auto english_reference = stream_reference(engine, log, long_english, "en");
        check_offline_interleave(engine, log, english, long_french, french_reference);
        check_alternating_sessions(engine, log, long_english, long_french, english_reference, french_reference);
        check_short_finalize(engine, log, french);
        check_explicit_session(options, log, english, "en");
        check_explicit_session(options, log, english, "other");
        std::printf("native EU API routing, interleaved sessions, gate timing and chunk defaults passed\n");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
