#include "backend_util.h"
#include "mel_preprocess.h"
#include "parakeet/engine.h"
#include "parakeet_ctc.h"
#include "parakeet_diarization_v3.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kSampleRate = 16000;
constexpr int kStreamSeconds = 8;
constexpr int kStreamChunkMs = 1040;
constexpr int kStreamRightContextMs = 80;
constexpr int kStreamCacheFrames = 16;
constexpr int kStreamFifoFrames = 8;
constexpr int kStreamUpdateFrames = 4;
constexpr double kFrameStrideSeconds = 0.01;
constexpr double kMaximumAbsoluteError = 0.04;
constexpr double kMaximumRelativeL2Error = 0.003;
constexpr float kPositiveSampleScale = 0.1f;
constexpr float kNormalizationFloor = 1.0e-3f;
constexpr double kRegressionSeparation = 3.0;
constexpr int kTailSamples = 1280;
constexpr int kShortHistoryMs = 1500;
constexpr int kStreamLeftContextMs = 160;
constexpr int kExplicitLeftContextMs = 80;
constexpr int kFirstWindowMs = kStreamChunkMs + kStreamRightContextMs;
constexpr float kQuietAudioScale = 0.1f;
constexpr float kMaximumGainScoreDifference = 0.0002f;
constexpr float kPrewarmSeconds = 1.0f;
constexpr int kForcedLongFormFrames = 100;
constexpr double kMaximumLongFormDifference = 1.0e-4;
constexpr int kBeyondLimitRepeats = 15;
constexpr float kSpeakerActivityThreshold = 0.5f;
constexpr double kTailCoverageSeconds = 1.0;
constexpr double kMaximumRepeatActivityShare = 0.05;
constexpr double kMaximumPrewarmDifference = 1.0e-5;

static bool valid_probabilities(const std::vector<float> & probabilities) {
    for (float probability : probabilities) {
        if (!std::isfinite(probability) || probability < 0.0f || probability > 1.0f) {
            return false;
        }
    }
    return true;
}

static bool valid_segments(const parakeet::DiarizationResult & result) {
    if (result.segments.empty()) return false;
    for (const auto & segment : result.segments) {
        if (segment.speaker_id < 0 || segment.speaker_id >= result.num_spks ||
            segment.start_s < 0.0 || segment.end_s <= segment.start_s ||
            segment.end_s > result.audio_samples /
                static_cast<double>(result.sample_rate) + result.frame_stride_s) {
            return false;
        }
    }
    return true;
}

static bool matches_reference(
    const std::vector<float> & probabilities, const char * path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() !=
        static_cast<std::streamoff>(probabilities.size() * sizeof(float))) return false;
    input.seekg(0);
    std::vector<float> reference(probabilities.size());
    input.read(reinterpret_cast<char *>(reference.data()),
        static_cast<std::streamsize>(reference.size() * sizeof(float)));
    if (!input) return false;
    double maximum_error = 0.0;
    double difference_energy = 0.0;
    double reference_energy = 0.0;
    for (size_t index = 0; index < reference.size(); ++index) {
        const double difference = probabilities[index] - reference[index];
        maximum_error = std::max(maximum_error, std::abs(difference));
        difference_energy += difference * difference;
        reference_energy += static_cast<double>(reference[index]) * reference[index];
    }
    const double relative_error = std::sqrt(difference_energy / reference_energy);
    if (maximum_error > kMaximumAbsoluteError ||
        relative_error > kMaximumRelativeL2Error) {
        std::fprintf(stderr, "Nemotron probability mismatch: max=%f relative=%f\n",
            maximum_error, relative_error);
        return false;
    }
    return true;
}

static std::vector<float> make_negative_biased_audio(
    const std::vector<float> & samples) {
    std::vector<float> biased(samples.begin(), samples.begin() + kSampleRate);
    for (float & sample : biased) {
        if (sample > 0.0f) sample *= kPositiveSampleScale;
    }
    biased.front() = -1.0f;
    return biased;
}

static bool infer_probabilities(
    const parakeet::ParakeetCtcModel & model,
    const std::vector<float> & audio, std::vector<float> & probabilities,
    parakeet::NemotronAttention attention = parakeet::NemotronAttention::Automatic) {
    std::vector<float> mel;
    int mel_frames = 0;
    if (parakeet::compute_log_mel(audio.data(), static_cast<int>(audio.size()),
        model.mel_cfg, mel, mel_frames) != 0) return false;
    return parakeet::run_nemotron_diarization(
        model, mel.data(), mel_frames, probabilities, attention) == 0;
}

static void apply_peak_gain(std::vector<float> & audio, float peak) {
    for (float & sample : audio) sample /= peak + kNormalizationFloor;
}

static float peak_magnitude(const std::vector<float> & audio) {
    return std::fabs(*std::max_element(audio.begin(), audio.end(),
        [](float lhs, float rhs) { return std::fabs(lhs) < std::fabs(rhs); }));
}

static bool unfused_attention_matches_reference(
    const parakeet::ParakeetCtcModel & model, const std::vector<float> & samples,
    const char * reference_path) {
    std::vector<float> normalized = samples;
    apply_peak_gain(normalized, peak_magnitude(samples));
    std::vector<float> probabilities;
    return infer_probabilities(model, normalized, probabilities,
               parakeet::NemotronAttention::Unfused) &&
           matches_reference(probabilities, reference_path);
}

static double maximum_probability_error(
    const std::vector<float> & actual, const std::vector<float> & reference) {
    if (actual.size() != reference.size()) return 1.0;
    double maximum_error = 0.0;
    for (size_t index = 0; index < reference.size(); ++index) {
        maximum_error = std::max(maximum_error,
            static_cast<double>(std::fabs(actual[index] - reference[index])));
    }
    return maximum_error;
}

static bool matches_negative_peak_reference(
    parakeet::Engine & engine, const parakeet::ParakeetCtcModel & model,
    const std::vector<float> & samples) {
    std::vector<float> input = make_negative_biased_audio(samples);
    std::vector<float> normalized = input;
    apply_peak_gain(normalized,
        std::fabs(*std::min_element(input.begin(), input.end())));
    std::vector<float> expected;
    if (!infer_probabilities(model, normalized, expected)) return false;
    std::vector<float> signed_normalized = input;
    apply_peak_gain(signed_normalized,
        *std::max_element(input.begin(), input.end()));
    std::vector<float> incorrect;
    if (!infer_probabilities(model, signed_normalized, incorrect)) return false;
    const auto actual = engine.diarize_samples(
        input.data(), static_cast<int>(input.size()), kSampleRate, {});
    const double expected_error = maximum_probability_error(
        actual.speaker_probs, expected);
    const double incorrect_error = maximum_probability_error(
        actual.speaker_probs, incorrect);
    if (expected_error < kMaximumAbsoluteError &&
        expected_error * kRegressionSeparation < incorrect_error) return true;
    std::fprintf(stderr, "Nemotron negative peak errors: expected=%f incorrect=%f\n",
        expected_error, incorrect_error);
    return false;
}

static bool emits_short_final_tail(
    parakeet::Engine & engine, const std::vector<float> & samples) {
    parakeet::SortformerStreamingOptions options;
    options.threshold = 0.0f;
    options.min_segment_ms = 0;
    double emitted_end_s = 0.0;
    auto session = engine.diarize_start(options,
        [&emitted_end_s](const parakeet::StreamingDiarizationSegment & segment) {
            if (segment.speaker_id >= 0) {
                emitted_end_s = std::max(emitted_end_s, segment.end_s);
            }
        });
    session->feed_pcm_f32(samples.data(), kTailSamples);
    session->finalize();
    return emitted_end_s >=
        static_cast<double>(kTailSamples) / kSampleRate - kFrameStrideSeconds;
}

struct FirstChunkOutput {
    double last_segment_end_s = 0.0;
    float vad_score = -1.0f;
};

static FirstChunkOutput run_first_chunk(
    parakeet::Engine & engine, const std::vector<float> & samples) {
    FirstChunkOutput output;
    parakeet::SortformerStreamingOptions options;
    options.threshold = 0.0f;
    options.min_segment_ms = 0;
    options.chunk_left_context_ms = kStreamLeftContextMs;
    options.on_event = [&output](const parakeet::StreamEvent & event) {
        output.vad_score = event.vad_score;
    };
    auto session = engine.diarize_start(options,
        [&output](const parakeet::StreamingDiarizationSegment & segment) {
            if (segment.speaker_id >= 0) {
                output.last_segment_end_s = std::max(
                    output.last_segment_end_s, segment.end_s);
            }
        });
    session->feed_pcm_f32(samples.data(),
        kSampleRate * kFirstWindowMs / 1000);
    return output;
}

static std::vector<float> scale_audio(
    const std::vector<float> & samples, float scale) {
    std::vector<float> scaled = samples;
    for (float & sample : scaled) sample *= scale;
    return scaled;
}

static bool preserves_first_chunk_and_gain(
    parakeet::Engine & engine, const std::vector<float> & samples) {
    const auto original = run_first_chunk(engine, samples);
    const auto quiet = run_first_chunk(
        engine, scale_audio(samples, kQuietAudioScale));
    const double expected_end_s = static_cast<double>(kStreamChunkMs) / 1000.0;
    const bool passed = original.last_segment_end_s >= expected_end_s - kFrameStrideSeconds &&
        quiet.last_segment_end_s >= expected_end_s - kFrameStrideSeconds &&
        original.vad_score >= 0.0f && quiet.vad_score >= 0.0f &&
        std::fabs(original.vad_score - quiet.vad_score) < kMaximumGainScoreDifference;
    if (!passed) {
        std::fprintf(stderr, "first chunk: end=%f quiet end=%f score=%f quiet score=%f\n",
            original.last_segment_end_s, quiet.last_segment_end_s,
            original.vad_score, quiet.vad_score);
    }
    return passed;
}

static bool long_form_window_resolves(const parakeet::ParakeetCtcModel & model) {
    const int limit = model.nemotron_diarization_cfg.position_limit;
    return parakeet::nemotron_diarization_long_form_frames(model, 0) ==
               std::min(parakeet::kNemotronLongFormAutoFrames, limit) &&
           parakeet::nemotron_diarization_long_form_frames(model, -1) == 0 &&
           parakeet::nemotron_diarization_long_form_frames(model, limit + 1) == limit &&
           parakeet::nemotron_diarization_long_form_frames(model, kForcedLongFormFrames) ==
               kForcedLongFormFrames;
}

static bool long_form_matches_single_pass(
    const char * model_path, int gpu_layers, const std::vector<float> & samples,
    const parakeet::DiarizationResult & expected) {
    parakeet::EngineOptions options;
    options.model_gguf_path = model_path;
    options.prewarm = false;
    options.n_gpu_layers = gpu_layers;
    options.long_form_window_frames = kForcedLongFormFrames;
    parakeet::Engine engine(options);
    const auto actual = engine.diarize_samples(
        samples.data(), static_cast<int>(samples.size()), kSampleRate, {});
    const double difference = maximum_probability_error(
        actual.speaker_probs, expected.speaker_probs);
    if (actual.n_frames == expected.n_frames && valid_segments(actual) &&
        difference <= kMaximumLongFormDifference) return true;
    std::fprintf(stderr, "Nemotron long-form frames=%d expected=%d difference=%f\n",
        actual.n_frames, expected.n_frames, difference);
    return false;
}

static std::vector<float> repeat_audio(const std::vector<float> & samples, int repeats) {
    std::vector<float> repeated;
    repeated.reserve(samples.size() * repeats);
    for (int repeat = 0; repeat < repeats; ++repeat) {
        repeated.insert(repeated.end(), samples.begin(), samples.end());
    }
    return repeated;
}

static int active_frames(
    const parakeet::DiarizationResult & result, int speaker, int first, int last) {
    int active = 0;
    for (int frame = first; frame < last; ++frame) {
        active += result.speaker_probs[static_cast<size_t>(frame) * result.num_spks + speaker] >
            kSpeakerActivityThreshold;
    }
    return active;
}

static bool repeats_keep_speaker_activity(
    const parakeet::DiarizationResult & result, int repeats) {
    const int block = result.n_frames / repeats;
    const int tolerance = static_cast<int>(block * kMaximumRepeatActivityShare);
    for (int speaker = 0; speaker < result.num_spks; ++speaker) {
        const int first = active_frames(result, speaker, 0, block);
        const int last = active_frames(result, speaker, (repeats - 1) * block, repeats * block);
        if (std::abs(first - last) > tolerance) {
            std::fprintf(stderr, "speaker %d active frames: first repeat %d, last repeat %d\n",
                speaker, first, last);
            return false;
        }
    }
    return true;
}

static double last_segment_end(const parakeet::DiarizationResult & result) {
    double end_s = 0.0;
    for (const auto & segment : result.segments) end_s = std::max(end_s, segment.end_s);
    return end_s;
}

static bool diarizes_beyond_position_limit(
    parakeet::Engine & engine, const parakeet::ParakeetCtcModel & model,
    const std::vector<float> & samples) {
    const std::vector<float> audio = repeat_audio(samples, kBeyondLimitRepeats);
    const int samples_count = static_cast<int>(audio.size());
    const long long mel_frames = parakeet::nemotron_diarization_mel_frames(model, samples_count);
    if (parakeet::nemotron_diarization_encoder_frames(model, mel_frames) <=
        model.nemotron_diarization_cfg.position_limit) {
        std::fprintf(stderr, "beyond-limit input is not beyond the position limit\n");
        return false;
    }
    const auto result = engine.diarize_samples(audio.data(), samples_count, kSampleRate, {});
    const double duration_s = static_cast<double>(samples_count) / kSampleRate;
    const bool covered = result.n_frames == mel_frames &&
        result.speaker_probs.size() == static_cast<size_t>(result.n_frames) * result.num_spks &&
        valid_probabilities(result.speaker_probs) && valid_segments(result) &&
        last_segment_end(result) >= duration_s - kTailCoverageSeconds;
    if (!covered) {
        std::fprintf(stderr, "beyond-limit frames=%d expected=%lld last segment end=%f of %f\n",
            result.n_frames, mel_frames, last_segment_end(result), duration_s);
        return false;
    }
    return repeats_keep_speaker_activity(result, kBeyondLimitRepeats);
}

static bool prewarmed_engine_matches(
    const char * model_path, int gpu_layers, const std::vector<float> & samples,
    const parakeet::DiarizationResult & expected) {
    parakeet::EngineOptions options;
    options.model_gguf_path = model_path;
    options.prewarm = true;
    options.prewarm_audio_seconds = kPrewarmSeconds;
    options.n_gpu_layers = gpu_layers;
    parakeet::Engine engine(options);
    const auto actual = engine.diarize_samples(
        samples.data(), static_cast<int>(samples.size()), kSampleRate, {});
    const double difference = maximum_probability_error(
        actual.speaker_probs, expected.speaker_probs);
    if (difference <= kMaximumPrewarmDifference) return true;
    std::fprintf(stderr, "Nemotron prewarmed engine difference: %f\n", difference);
    return false;
}

}

int main(int argc, char ** argv) {
    if (argc != 4 && argc != 5) return 2;
    const int gpu_layers = argc == 5 ? std::stoi(argv[4]) : 0;
    parakeet::ParakeetCtcModel model;
    if (parakeet::load_from_gguf(argv[1], model, 1, gpu_layers, false) != 0) {
        std::fprintf(stderr, "Nemotron model load failed\n");
        return 1;
    }
    if (gpu_layers > 0 && !parakeet::model_has_gpu_backend(model)) return 3;
    std::vector<float> samples;
    int sample_rate = 0;
    if (parakeet::load_wav_mono_f32(argv[2], samples, sample_rate) != 0 ||
        sample_rate != model.mel_cfg.sample_rate) {
        std::fprintf(stderr, "Nemotron audio input failed\n");
        return 1;
    }
    std::vector<float> mel;
    int mel_frames = 0;
    if (parakeet::compute_log_mel(samples.data(), static_cast<int>(samples.size()),
        model.mel_cfg, mel, mel_frames) != 0) {
        std::fprintf(stderr, "Nemotron mel preprocessing failed\n");
        return 1;
    }
    std::vector<float> probabilities;
    if (parakeet::run_nemotron_diarization(model, mel.data(), mel_frames,
        probabilities) != 0) {
        std::fprintf(stderr, "Nemotron direct inference failed\n");
        return 1;
    }
    if (probabilities.size() != static_cast<size_t>(mel_frames) *
        model.nemotron_diarization_cfg.speakers) {
        std::fprintf(stderr, "Nemotron direct output shape mismatch\n");
        return 1;
    }
    if (!valid_probabilities(probabilities)) {
        std::fprintf(stderr, "Nemotron direct probabilities are invalid\n");
        return 1;
    }
    parakeet::EngineOptions options;
    options.model_gguf_path = argv[1];
    options.prewarm = false;
    options.n_gpu_layers = gpu_layers;
    parakeet::Engine engine(options);
    if (!engine.is_diarization_model()) {
        std::fprintf(stderr, "Nemotron engine model type mismatch\n");
        return 1;
    }
    const auto result = engine.diarize_samples(
        samples.data(), static_cast<int>(samples.size()), sample_rate, {});
    if (result.num_spks != model.nemotron_diarization_cfg.speakers ||
        result.n_frames != mel_frames ||
        std::fabs(result.frame_stride_s - kFrameStrideSeconds) > 1.0e-8 ||
        result.speaker_probs.size() != static_cast<size_t>(result.n_frames) * result.num_spks ||
        !valid_probabilities(result.speaker_probs) || !valid_segments(result)) {
        std::fprintf(stderr, "Nemotron engine output contract mismatch\n");
        return 1;
    }
    if (!matches_reference(result.speaker_probs, argv[3])) return 1;
    if (!unfused_attention_matches_reference(model, samples, argv[3])) {
        std::fprintf(stderr, "Nemotron unfused attention reference mismatch\n");
        return 1;
    }
    if (gpu_layers == 0 && !parakeet::nemotron_diarization_uses_fused_attention(model)) {
        std::fprintf(stderr, "Nemotron CPU inference did not select fused attention\n");
        return 1;
    }
    if (parakeet::nemotron_diarization_uses_f32_matmuls(model) !=
        parakeet::backend_is_adreno(parakeet::model_active_backend(model))) {
        std::fprintf(stderr, "Nemotron F32 matmul precision does not follow the Adreno GPU\n");
        return 1;
    }
    if (parakeet::prewarm_nemotron_diarization(model, kPrewarmSeconds) != 0) {
        std::fprintf(stderr, "Nemotron prewarm failed\n");
        return 1;
    }
    if (!prewarmed_engine_matches(argv[1], gpu_layers, samples, result)) return 1;
    if (!long_form_window_resolves(model)) {
        std::fprintf(stderr, "Nemotron long-form window resolution mismatch\n");
        return 1;
    }
    if (!long_form_matches_single_pass(argv[1], gpu_layers, samples, result)) return 1;
    if (!diarizes_beyond_position_limit(engine, model, samples)) {
        std::fprintf(stderr, "Nemotron diarization beyond the position limit failed\n");
        return 1;
    }
    if (!matches_negative_peak_reference(engine, model, samples)) {
        std::fprintf(stderr, "Nemotron negative peak reference mismatch\n");
        return 1;
    }
    if (!emits_short_final_tail(engine, samples)) {
        std::fprintf(stderr, "Nemotron short final tail missing\n");
        return 1;
    }
    const auto default_session = engine.diarize_start({}, {});
    if (!default_session->aosc_active() ||
        default_session->options().chunk_ms != kStreamChunkMs ||
        default_session->options().chunk_left_context_ms != 0) return 1;
    parakeet::SortformerStreamingOptions explicit_context;
    explicit_context.chunk_left_context_ms = kExplicitLeftContextMs;
    const auto explicit_session = engine.diarize_start(explicit_context, {});
    if (explicit_session->options().chunk_left_context_ms !=
        kExplicitLeftContextMs) return 1;
    if (!preserves_first_chunk_and_gain(engine, samples)) {
        std::fprintf(stderr, "Nemotron first chunk or streaming gain mismatch\n");
        return 1;
    }
    parakeet::SortformerStreamingOptions short_history;
    short_history.history_ms = kShortHistoryMs;
    const auto short_history_session = engine.diarize_start(short_history, {});
    if (short_history_session->options().chunk_ms != kStreamChunkMs ||
        short_history_session->options().history_ms != kShortHistoryMs) return 1;
    try {
        parakeet::SortformerStreamingOptions unaligned;
        unaligned.chunk_ms = 1000;
        engine.diarize_start(unaligned, {});
        return 1;
    } catch (const std::runtime_error &) {
    }
    parakeet::SortformerStreamingOptions streaming;
    streaming.chunk_ms = kStreamChunkMs;
    streaming.history_ms = 2000;
    streaming.chunk_right_context_ms = kStreamRightContextMs;
    streaming.spkcache_len = kStreamCacheFrames;
    streaming.fifo_len = kStreamFifoFrames;
    streaming.spkcache_update_period = kStreamUpdateFrames;
    bool finalized = false;
    int emitted_segments = 0;
    auto session = engine.diarize_start(streaming,
        [&finalized, &emitted_segments](
            const parakeet::StreamingDiarizationSegment & segment) {
            if (segment.is_final && segment.speaker_id == -1) finalized = true;
            if (segment.speaker_id >= 0 && segment.end_s > segment.start_s) {
                ++emitted_segments;
            }
        });
    if (!session->aosc_active()) return 1;
    session->feed_pcm_f32(samples.data(), kSampleRate * kStreamSeconds);
    session->finalize();
    if (!finalized || emitted_segments == 0) return 1;
    return 0;
}
