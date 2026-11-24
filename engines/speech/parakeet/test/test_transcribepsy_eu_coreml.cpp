#include "parakeet/engine.h"
#include "../src/parakeet_ctc.h"
#include "../src/long_form_encoder.h"
#include "../src/parakeet_coreml_path.h"
#include "test_utils.h"

#include <algorithm>
#include <cstdio>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int kRate = 16000;
constexpr int kThreads = 4;
constexpr int kGateSeconds = 20;

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

void check_encoder_ownership(const char * path) {
    if (!std::getenv("PARAKEET_COREML_ONLY")) return;
    parakeet::ParakeetCtcModel model;
    require(parakeet::load_from_gguf(path, model, kThreads, 0, false, "cpu") == 0,
            "owned full-EU model failed to load");
    require(parakeet::model_encoder_owned_by_coreml(model), "Core ML-only option did not own encoder");
    require(model.blocks.empty() && model.subsampling.conv0_w == nullptr,
            "owned full-EU model allocated native encoder weights");
    std::printf("full-EU Core ML ownership verified: native encoder weights absent\n");
}

std::vector<float> read_samples(const char * path) {
    std::vector<float> pcm;
    int rate = 0;
    require(parakeet_test::load_wav_pcm16le_mono(path, pcm, rate) && rate == kRate,
            "could not read 16 kHz fixture");
    return pcm;
}

std::string push(parakeet::Engine & engine, const std::vector<float> & pcm,
                 int block, bool automatic, bool large_context = false,
                 int chunk_ms = 0, bool expect_coreml = true) {
    parakeet::StreamingOptions options;
    if (chunk_ms > 0) options.chunk_ms = chunk_ms;
    if (large_context) options.left_context_ms = kGateSeconds * 1000;
    std::string text;
    int segments = 0;
    size_t fed = 0;
    double first_ms = -1.0, first_input_s = -1.0;
    const auto started = std::chrono::steady_clock::now();
    auto session = engine.stream_start(options, [&](const parakeet::StreamingSegment & segment) {
        require(segment.encoder_used_coreml == expect_coreml, "push encoder or gate dispatch differs");
        if (segments == 0) {
            first_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
            first_input_s = (double) fed / kRate;
        }
        text += segment.text;
        ++segments;
    });
    require(session->options().chunk_ms == (chunk_ms > 0 ? chunk_ms : 2000),
            "full-EU stream did not resolve automatic or explicit chunk size");
    for (size_t offset = 0; offset < pcm.size();) {
        const int count = (int) std::min((size_t) block, pcm.size() - offset);
        fed += count;
        session->feed_pcm_f32(pcm.data() + offset, count);
        offset += count;
        if (automatic && offset < (size_t) kGateSeconds * kRate)
            require(segments == 0, "auto streaming emitted before its complete gate prefix");
    }
    session->finalize();
    const int finalized_segments = segments;
    session->finalize();
    require(segments == finalized_segments, "finalize emitted a duplicate tail");
    require(segments > 0, "stream emitted no segments");
    const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    std::printf("API_PUSH audio_s=%.6f block=%d context_ms=%d automatic=%d segments=%d first_input_s=%.6f first_callback_ms=%.3f total_ms=%.3f\n",
        (double) pcm.size()/kRate, block, options.left_context_ms, (int) automatic, segments, first_input_s, first_ms, elapsed);
    return text;
}

void check_api_routes(parakeet::Engine & engine, const std::vector<float> & pcm,
                       bool automatic) {
    const auto batch = engine.transcribe_samples(pcm.data(), (int) pcm.size(), kRate);
    require(batch.encoder_used_coreml && !batch.text.empty(), "batch encoder or gate fell back");
    // The result names the program that ran (the model picks one per
    // dispatch), the same one Engine::encoder_backend() reports.
    require(batch.encoder_backend.rfind("coreml", 0) == 0 &&
                batch.encoder_backend == engine.encoder_backend(),
            "batch encoder_backend does not name the Core ML program that ran");
    const char * batch_override = std::getenv("PARAKEET_ANE_BATCH_DECODE");
    const bool forced = batch_override && std::string(batch_override) == "1" &&
        !std::getenv("PARAKEET_ANE_DECODE_HOST");
    require(batch.decoder_used_coreml == forced, "short-clip decoder dispatch was misreported");
    auto mel = engine.compute_mel(pcm.data(), (int) pcm.size(), kRate);
    const auto precomputed = engine.transcribe_mel(std::move(mel));
    require(precomputed.encoder_used_coreml && precomputed.token_ids == batch.token_ids,
            "precomputed mel changed the result or dispatch");
    parakeet::StreamingOptions options;
    const auto callbacks = engine.transcribe_samples_stream(
        pcm.data(), (int) pcm.size(), kRate, options,
        [](const parakeet::StreamingSegment & segment) {
            require(segment.encoder_used_coreml, "callback encoder or gate fell back");
        });
    require(callbacks.encoder_used_coreml && !callbacks.text.empty(),
            "callback encoder or gate fell back");
    require(callbacks.encoder_backend.rfind("coreml", 0) == 0,
            "callback encoder_backend does not name a Core ML program");
    require(engine.encoder_backend() == "coreml-cpu" || engine.encoder_backend() == "coreml-cpu-gpu",
            "stateful callback did not use an FP32 Core ML backend");
    const auto fragmented = push(engine, pcm, 127, automatic);
    require(fragmented == push(engine, pcm, 4096, automatic),
            "feed boundaries changed transcription or reused stale state");
    require(!fragmented.empty(), "push transcription is empty");
    std::printf("API quality batch=[%s] push=[%s] normalized_similarity=%.6f\n",
                batch.text.c_str(), fragmented.c_str(),
                parakeet_test::word_similarity(batch.text, fragmented));
}
}  // namespace

namespace {
int check_language(const parakeet::EngineOptions & options, const std::vector<float> & english,
                    const std::vector<float> & french) {
    parakeet::Engine engine(options);
    if (!engine.encoder_on_coreml()) return 77;
    const bool automatic = options.language == "auto";
    check_api_routes(engine, english, automatic);
    check_api_routes(engine, french, automatic);
    check_api_routes(engine, english, automatic);
    const auto long_pcm = parakeet_test::tile_to_seconds(english, kRate, 30);
    require(!push(engine, long_pcm, 4096, automatic, true).empty(), "large-context stream is empty");
    auto cancelled = engine.stream_start({}, [](const auto &) {});
    cancelled->feed_pcm_f32(english.data(), 127);
    cancelled->cancel();
    cancelled->finalize();
    if (automatic) {
        std::vector<float> prefixed((size_t) kGateSeconds * kRate, 0.0f);
        prefixed.insert(prefixed.end(), french.begin(), french.end());
        require(push(engine, prefixed, 127, true) == push(engine, prefixed, 4096, true),
                "silence prefix or feed boundaries changed the gate route");
        auto empty = engine.stream_start({}, [](const auto &) {
            throw std::runtime_error("empty stream emitted a segment");
        });
        empty->finalize();
    }
    std::printf("full-EU Core ML API/stream/reset checks passed: %s\n", options.language.c_str());
    return 0;
}

class EnvironmentScope {
public:
    EnvironmentScope(const char * name, const char * value) : name_(name) {
        const char * previous = std::getenv(name);
        had_value_ = previous != nullptr;
        previous_ = previous ? previous : "";
        if (value) setenv(name, value, 1);
        else unsetenv(name);
    }
    ~EnvironmentScope() {
        if (had_value_) setenv(name_.c_str(), previous_.c_str(), 1);
        else unsetenv(name_.c_str());
    }
private:
    std::string name_, previous_;
    bool had_value_ = false;
};

void check_initial_placement(const char * path) {
    EnvironmentScope enabled("PARAKEET_COREML_DISABLE", nullptr);
    for (const char * owned : {static_cast<const char *>(nullptr), "1"}) {
        EnvironmentScope ownership("PARAKEET_COREML_ONLY", owned);
        for (const char * units : {"cpu_only", "cpu_and_gpu", static_cast<const char *>(nullptr)}) {
            EnvironmentScope policy("PARAKEET_COREML_COMPUTE_UNITS", units);
            parakeet::ParakeetCtcModel model;
            require(parakeet::load_from_gguf(path, model, kThreads, 0, false, "cpu") == 0,
                    "initial placement model failed to load");
            const std::string expected = units ? "coreml-cpu" : "coreml-ane";
            const auto actual = parakeet::model_encoder_backend_name(model);
            require(actual == expected, "initial fixed metadata model has the wrong compute policy");
            require(parakeet::model_encoder_owned_by_coreml(model) == (owned != nullptr),
                    "initial placement ownership differs");
            std::printf("INITIAL_PLACEMENT units=%s owned=%d backend=%s before_encode=1\n",
                        units ? units : "default", (int) (owned != nullptr), actual.c_str());
        }
    }
}

class SidecarFixture {
public:
    explicit SidecarFixture(const char * source) : source_(source) {
        directory_ = std::filesystem::temp_directory_path() / ("eu-sidecar-selection-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directory(directory_);
        model = (directory_ / source_.filename()).string();
        std::filesystem::create_symlink(std::filesystem::absolute(source_), model);
    }
    ~SidecarFixture() { std::filesystem::remove_all(directory_); }
    void link(const std::string & sidecar) {
        link_as(sidecar, std::filesystem::path(sidecar).filename().string());
    }
    void link_as(const std::string & sidecar, const std::string & filename) {
        std::filesystem::create_directory_symlink(std::filesystem::absolute(sidecar), directory_ / filename);
    }
    std::string model;
private:
    std::filesystem::path source_, directory_;
};

void check_partial_route_fallback(const parakeet::EngineOptions & options,
                                  const std::vector<float> & french) {
    parakeet::Engine engine(options);
    require(engine.encoder_on_coreml(), "English sidecar availability missing from partial fixture");
    parakeet::EngineResult expected;
    {
        EnvironmentScope disabled("PARAKEET_COREML_DISABLE", "1");
        parakeet::Engine native(options);
        expected = native.transcribe_samples(french.data(), (int) french.size(), kRate);
    }
    const auto actual = engine.transcribe_samples(french.data(), (int) french.size(), kRate);
    require(!actual.encoder_used_coreml && !actual.decoder_used_coreml,
            "partial other route unexpectedly executed Core ML");
    require(!expected.token_ids.empty() && actual.token_ids == expected.token_ids,
            "unavailable adapted sidecar changed native continuous decoding");
}

void check_sidecar_selection(const char * path, const std::vector<float> & english,
                             const std::vector<float> & french) {
    EnvironmentScope enabled("PARAKEET_COREML_DISABLE", nullptr);
    EnvironmentScope ownership("PARAKEET_COREML_ONLY", nullptr);
    EnvironmentScope policy("PARAKEET_COREML_COMPUTE_UNITS", nullptr);
    EnvironmentScope batch("PARAKEET_ANE_BATCH_DECODE", nullptr);
    SidecarFixture partial(path);
    partial.link(parakeet::coreml_encoder_sidecar_path(path));
    parakeet::EngineOptions options;
    options.model_gguf_path = partial.model;
    options.n_threads = kThreads;
    options.backend = "cpu";
    options.language = "other";
    const auto long_french = parakeet_test::tile_to_seconds(french, kRate, 22);
    check_partial_route_fallback(options, long_french);
    // A decoder sidecar must not independently enable encoder segmentation.
    partial.link(parakeet::coreml_decoder_sidecar_path(path));
    check_partial_route_fallback(options, long_french);
    partial.link_as(parakeet::coreml_encoder_sidecar_path(path),
        std::filesystem::path(parakeet::coreml_eu_other_sidecar_path(path)).filename().string());
    check_partial_route_fallback(options, long_french);

    SidecarFixture flexible(path);
    flexible.link(parakeet::coreml_eu_flexible_sidecar_path(path, false));
    flexible.link(parakeet::coreml_eu_flexible_sidecar_path(path, true));
    auto gate = parakeet::coreml_eu_gate_sidecar_path(path);
    gate.insert(gate.size() - std::string(".mlmodelc").size(), "-flexible");
    flexible.link(gate);
    options.model_gguf_path = flexible.model;
    options.long_form_window_frames = -1;
    for (const char * language : {"en", "other", "auto"}) {
        options.language = language;
        parakeet::Engine engine(options);
        require(engine.encoder_on_coreml(), "flexible-only encoder capability missing before inference");
        const auto & pcm = options.language == "en" ? english : french;
        const auto actual = engine.transcribe_samples(pcm.data(), (int) pcm.size(), kRate);
        require(actual.encoder_used_coreml && !actual.text.empty(),
                "flexible-only route or gate did not execute Core ML");
    }
    std::printf("sidecar selection: three native-continuity fallbacks and three flexible-only routes passed\n");
}

std::unique_ptr<parakeet::Engine> native_reference(const parakeet::EngineOptions & options) {
    EnvironmentScope disabled("PARAKEET_COREML_DISABLE", "1");
    return std::make_unique<parakeet::Engine>(options);
}

void compare_stateful(parakeet::Engine & native, parakeet::Engine & coreml,
                       const std::vector<float> & pcm, bool automatic) {
    parakeet::StreamingOptions options;
    options.chunk_ms = 2000;
    const auto reference = native.transcribe_samples_stream(pcm.data(), (int) pcm.size(), kRate, options, [](const auto &) {});
    const auto actual = coreml.transcribe_samples_stream(pcm.data(), (int) pcm.size(), kRate, options,
        [](const auto & segment) { require(segment.encoder_used_coreml, "precise callback fell back"); });
    require(actual.encoder_used_coreml && !reference.encoder_used_coreml && actual.token_ids == reference.token_ids,
            "precise callback tokens differ from the same-context native reference");
    const auto expected_push = push(native, pcm, 4096, automatic, false, 2000, false);
    const auto actual_push = push(coreml, pcm, 4096, automatic, false, 2000, true);
    if (expected_push != actual_push)
        std::fprintf(stderr, "PUSH_REFERENCE [%s]\nPUSH_ACTUAL [%s]\n", expected_push.c_str(), actual_push.c_str());
    require(expected_push == actual_push, "precise push text differs from the same-context native reference");
    std::printf("STATEFUL_PARITY audio_s=%.3f encoder_backend=%s callback_tokens=%zu push_equal=1\n",
        (double) pcm.size()/kRate, coreml.encoder_backend().c_str(), actual.token_ids.size());
}

void compare_padding_boundary(parakeet::ParakeetCtcModel & native,
                              parakeet::ParakeetCtcModel & coreml, int frames) {
    constexpr int kMelBins = 128;
    std::vector<float> mel((size_t) frames * kMelBins);
    for (size_t i = 0; i < mel.size(); ++i) mel[i] = 0.1f * std::sin((double) i / 31.0);
    std::fill(mel.end() - kMelBins, mel.end(), 0.0f);
    parakeet::EncoderOutputs expected, actual;
    require(parakeet::run_encoder(native, mel.data(), frames, kMelBins, expected,
                parakeet::EncoderRoute::ggml(), -1, true, true, {}, frames - 1, true) == 0,
            "native padding reference failed");
    require(parakeet::run_encoder(coreml, mel.data(), frames, kMelBins, actual,
                parakeet::acquire_encoder_route(coreml), -1, false, true, {}, frames - 1, true) == 0 &&
                actual.used_coreml,
            "precise padding boundary did not execute Core ML");
    require(actual.n_enc_frames == expected.n_enc_frames,
            "Core ML changed the native frontend encoder frame grid");
    double error = 0.0, magnitude = 0.0;
    for (size_t i = 0; i < expected.encoder_out.size(); ++i) {
        const double delta = actual.encoder_out[i] - expected.encoder_out[i];
        error += delta * delta;
        magnitude += (double) expected.encoder_out[i] * expected.encoder_out[i];
    }
    constexpr double kNativeHalfActivationTolerance = 0.01;
    const double relative = std::sqrt(error / std::max(magnitude, 1e-20));
    require(relative < kNativeHalfActivationTolerance, "precise padded features differ from native");
    std::printf("PADDING_PARITY frames=%d valid=%d encoder_frames=%d adapters=%d relative_l2=%.8f\n",
        frames, frames - 1, actual.n_enc_frames, (int) coreml.eu_use_adapters, relative);
}

void check_padding_boundaries(const char * path) {
    parakeet::ParakeetCtcModel native, coreml;
    {
        EnvironmentScope disabled("PARAKEET_COREML_DISABLE", "1");
        require(parakeet::load_from_gguf(path, native, kThreads, 0, false, "cpu") == 0,
                "native padding model failed to load");
    }
    require(parakeet::load_from_gguf(path, coreml, kThreads, 0, false, "cpu") == 0,
            "Core ML padding model failed to load");
    for (bool adapters : {false, true}) {
        native.eu_use_adapters = coreml.eu_use_adapters = adapters;
        for (int frames : {399, 400, 401, 402, 407, 408, 409})
            compare_padding_boundary(native, coreml, frames);
    }
}

void check_stateful_precision(const char * model, const std::vector<float> & english,
                               const std::vector<float> & french) {
    for (const std::string language : {"en", "other", "auto"}) {
        parakeet::EngineOptions options;
        options.model_gguf_path = model;
        options.backend = "cpu";
        options.n_threads = kThreads;
        options.language = language;
        auto native = native_reference(options);
        parakeet::Engine coreml(options);
        compare_stateful(*native, coreml, english, language == "auto");
        if (language != "en") compare_stateful(*native, coreml, french, language == "auto");
        if (language == "auto") {
            constexpr int kRepeats = 12;
            std::vector<float> long_french;
            for (int i = 0; i < kRepeats; ++i) {
                long_french.insert(long_french.end(), french.begin(), french.end());
                long_french.insert(long_french.end(), kRate, 0.0f);
            }
            compare_stateful(*native, coreml, long_french, true);
        }
    }
}

std::string normalize_english(const std::string & text) {
    std::string result;
    for (unsigned char value : text) {
        if (std::isalnum(value)) result += (char) std::tolower(value);
        else if (std::isspace(value) && !result.empty() && result.back() != ' ') result += ' ';
    }
    if (!result.empty() && result.back() == ' ') result.pop_back();
    return result;
}

void check_context_plan(const char * path) {
    constexpr int kLongMelFrames = 30001;
    parakeet::EngineOptions options;
    parakeet::ParakeetCtcModel model;
    {
        EnvironmentScope disabled("PARAKEET_COREML_DISABLE", "1");
        require(parakeet::load_from_gguf(path, model, kThreads, 0, false, "cpu") == 0,
                "native context reference failed to load");
    }
    const parakeet::EncoderRoute route = parakeet::acquire_encoder_route(model);
    const auto automatic = parakeet::resolve_long_form_plan(model, options, kLongMelFrames, route);
    options.long_form_window_frames = 750;
    options.long_form_context_frames = 94;
    const auto explicit_default = parakeet::resolve_long_form_plan(model, options, kLongMelFrames, route);
    std::printf("NATIVE_PLAN automatic_window_frames=%d automatic_context_frames=%d explicit_window_frames=%d explicit_context_frames=%d\n",
        automatic.window_frames, automatic.context_frames,
        explicit_default.window_frames, explicit_default.context_frames);
}

void check_explicit_context(const char * path, const std::vector<float> & english) {
    check_context_plan(path);
    const auto pcm = parakeet_test::tile_to_seconds(english, kRate, 120);
    std::string expected;
    for (const char * mode : {"auto", "0", "1"}) {
        EnvironmentScope decoder("PARAKEET_ANE_BATCH_DECODE", std::string(mode) == "auto" ? nullptr : mode);
        parakeet::EngineOptions options;
        options.model_gguf_path = path;
        options.backend = "cpu";
        options.n_threads = kThreads;
        options.language = "en";
        options.long_form_window_frames = 750;
        options.long_form_context_frames = 25;
        parakeet::Engine engine(options);
        const auto result = engine.transcribe_samples(pcm.data(), (int) pcm.size(), kRate);
        require(result.encoder_used_coreml && !result.text.empty(), "explicit-context encoder fell back");
        require(result.decoder_used_coreml == (std::string(mode) == "1"),
                "explicit-context decoder did not honor auto/force0/force1");
        const auto normalized = normalize_english(result.text);
        if (expected.empty()) expected = normalized;
        require(normalized == expected, "explicit-context decoding changed normalized English words");
        std::printf("EXPLICIT_CONTEXT mode=%s requested_window=750 requested_context=25 decoder_coreml=%d words_equal=1\n",
            mode, (int) result.decoder_used_coreml);
    }
}

std::vector<float> repeated_french(const std::vector<float> & french) {
    constexpr int kUtterances = 12;
    std::vector<float> pcm;
    for (int index = 0; index < kUtterances; ++index) {
        pcm.insert(pcm.end(), french.begin(), french.end());
        pcm.insert(pcm.end(), kRate, 0.0f);
    }
    return pcm;
}

void measure_warm_stream(parakeet::Engine & engine, const std::vector<float> & pcm,
                         const char * backend, bool coreml, bool pushed) {
    std::string expected;
    constexpr int kWarmRepeats = 3;
    for (int repeat = 0; repeat <= kWarmRepeats; ++repeat) {
        const auto started = std::chrono::steady_clock::now();
        std::string text;
        if (pushed) text = push(engine, pcm, 4096, true, false, 0, coreml);
        else {
            const auto result = engine.transcribe_samples_stream(pcm.data(), (int) pcm.size(), kRate, {},
                [coreml](const auto & segment) {
                    require(segment.encoder_used_coreml == coreml, "warm callback dispatch differs");
                });
            require(result.encoder_used_coreml == coreml, "warm callback gate dispatch differs");
            text = result.text;
        }
        const double elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        if (repeat == 0) expected = text;
        require(text == expected, "reused streaming session changed text");
        std::printf("WARM_STREAM backend=%s mode=%s repeat=%d audio_s=%.6f total_ms=%.3f encoder_backend=%s same_text=1\n",
            backend, pushed ? "push" : "callback", repeat, (double) pcm.size()/kRate,
            elapsed, engine.encoder_backend().c_str());
    }
}

void check_warm_stream(const char * path, const std::vector<float> & french) {
    const auto pcm = repeated_french(french);
    for (const std::string backend : {"metal", "coreml"}) {
        parakeet::EngineOptions options;
        options.model_gguf_path = path;
        options.language = "auto";
        options.n_threads = kThreads;
        options.backend = backend == "metal" ? "metal" : "cpu";
        auto engine = backend == "metal" ? native_reference(options) :
            std::make_unique<parakeet::Engine>(options);
        measure_warm_stream(*engine, pcm, backend.c_str(), backend == "coreml", false);
        measure_warm_stream(*engine, pcm, backend.c_str(), backend == "coreml", true);
    }
}

void check_unwindowed(parakeet::EngineOptions options, const std::vector<float> & english,
                       const std::vector<float> & french) {
    options.long_form_window_frames = -1;
    parakeet::Engine engine(options);
    const bool automatic = options.language == "auto";
    const auto long_english = parakeet_test::tile_to_seconds(english, kRate, 30);
    const auto long_french = parakeet_test::tile_to_seconds(french, kRate, 30);
    check_api_routes(engine, long_english, automatic);
    check_api_routes(engine, long_french, automatic);
    require(!push(engine, long_english, 4096, automatic, true).empty(),
            "unwindowed large-context stream is empty");
    std::printf("full-EU Core ML unwindowed API/stream checks passed: %s\n", options.language.c_str());
}
}  // namespace

int transcribepsy_eu_coreml_test(int argc, char ** argv) {
    if (argc != 4 && !(argc == 5 && (std::string(argv[4]) == "--unwindowed" ||
                                      std::string(argv[4]) == "--context-parity" ||
                                      std::string(argv[4]) == "--explicit-context" ||
                                      std::string(argv[4]) == "--warm-stream" ||
                                      std::string(argv[4]) == "--initial-placement" ||
                                      std::string(argv[4]) == "--sidecar-selection"))) {
        std::fprintf(stderr, "usage: %s full-eu.gguf english.wav french.wav [--unwindowed|--context-parity|--explicit-context|--warm-stream|--initial-placement|--sidecar-selection]\n", argv[0]);
        return 2;
    }
    try {
        if (argc == 5 && std::string(argv[4]) == "--initial-placement") {
            check_initial_placement(argv[1]);
            return 0;
        }
        check_encoder_ownership(argv[1]);
        const auto english = read_samples(argv[2]);
        const auto french = read_samples(argv[3]);
        if (argc == 5 && std::string(argv[4]) == "--sidecar-selection") {
            check_sidecar_selection(argv[1], english, french);
            return 0;
        }
        if (argc == 5 && std::string(argv[4]) == "--explicit-context") {
            check_explicit_context(argv[1], english);
            return 0;
        }
        if (argc == 5 && std::string(argv[4]) == "--warm-stream") {
            check_warm_stream(argv[1], french);
            return 0;
        }
        if (argc == 5 && std::string(argv[4]) == "--context-parity") {
            check_padding_boundaries(argv[1]);
            check_stateful_precision(argv[1], english, french);
            return 0;
        }
        for (const std::string language : {"en", "other", "auto"}) {
            parakeet::EngineOptions options;
            options.model_gguf_path = argv[1];
            options.backend = "cpu";
            options.n_threads = kThreads;
            options.language = language;
            if (int result = check_language(options, english, french)) return result;
            if (argc == 5) check_unwindowed(options, english, french);
        }
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}

#ifndef PARAKEET_EU_COREML_TEST_LIBRARY
int main(int argc, char ** argv) {
    return transcribepsy_eu_coreml_test(argc, argv);
}
#endif
