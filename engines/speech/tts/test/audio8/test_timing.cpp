// Audio8 per-stage timing.
//
// The Metal work is ranked by where the time actually goes, so the breakdown
// has to be trustworthy before any of it is believed. What is checked here is
// the bookkeeping, not the durations: every stage that ran reports a positive
// time, every stage that did not stays at zero, the stages are disjoint and so
// cannot sum past the total, and the accounting is per call rather than
// accumulated across the life of the engine.

#include "tts-cpp/audio8/engine.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

using tts_cpp::audio8::Engine;
using tts_cpp::audio8::EngineOptions;
using tts_cpp::audio8::StageTimings;
using tts_cpp::audio8::VoicePrompt;

// Enough frames for the decode loop and one codec block, few enough to keep
// the test quick.
constexpr int TEST_FRAMES = 8;
constexpr int RESAMPLED_RATE = 24000;
constexpr const char * TEST_TEXT = "Timing the stages of this engine.";
constexpr const char * REFERENCE_TEXT = "The morning light spills across the quiet garden wall.";

struct paths {
    std::string lm;
    std::string decoder;
    std::string encoder;
    std::string reference_wav;
    int threads = 4;
};

EngineOptions options_for(const paths & where) {
    EngineOptions opts;
    opts.lm_gguf_path = where.lm;
    opts.codec_decoder_gguf_path = where.decoder;
    opts.codec_encoder_gguf_path = where.encoder;
    opts.n_threads = where.threads;
    opts.greedy = true;
    opts.max_frames = TEST_FRAMES;
    return opts;
}

bool positive(const char * tag, const char * stage, double value) {
    if (value > 0.0) return true;
    std::fprintf(stderr, "%s: FAIL %s reported %.6f ms, expected a positive time\n", tag,
                 stage, value);
    return false;
}

bool zero(const char * tag, const char * stage, double value) {
    if (value == 0.0) return true;
    std::fprintf(stderr, "%s: FAIL %s reported %.6f ms, expected zero\n", tag, stage,
                 value);
    return false;
}

double sum_of_stages(const StageTimings & t) {
    return t.voice_encode_ms + t.prompt_ms + t.prefill_ms + t.sample_ms +
           t.fast_decode_ms + t.slow_decode_ms + t.codec_latent_ms + t.codec_synth_ms +
           t.resample_ms;
}

void print_timings(const char * tag, const StageTimings & t) {
    std::fprintf(stderr,
                 "  [%s] voice-encode %.1f prompt %.1f prefill %.1f sample %.1f "
                 "fast %.1f slow %.1f latent %.1f synth %.1f resample %.1f total %.1f\n",
                 tag, t.voice_encode_ms, t.prompt_ms, t.prefill_ms, t.sample_ms,
                 t.fast_decode_ms, t.slow_decode_ms, t.codec_latent_ms, t.codec_synth_ms,
                 t.resample_ms, t.total_ms);
}

// Every stage nests inside the run, and no two overlap, so their sum is bounded
// by the total. The comparison carries a hair of slack for the clock reads
// themselves.
bool check_disjoint(const char * tag, const StageTimings & t) {
    const double sum = sum_of_stages(t);
    if (sum <= t.total_ms * 1.01) return true;
    std::fprintf(stderr, "%s: FAIL stages sum to %.3f ms, past the %.3f ms total\n", tag,
                 sum, t.total_ms);
    return false;
}

bool check_common(const char * tag, const StageTimings & t) {
    bool ok = positive(tag, "prompt", t.prompt_ms);
    ok &= positive(tag, "prefill", t.prefill_ms);
    ok &= positive(tag, "sample", t.sample_ms);
    ok &= positive(tag, "fast-decode", t.fast_decode_ms);
    ok &= positive(tag, "slow-decode", t.slow_decode_ms);
    ok &= positive(tag, "codec-latent", t.codec_latent_ms);
    ok &= positive(tag, "codec-synth", t.codec_synth_ms);
    ok &= positive(tag, "total", t.total_ms);
    ok &= check_disjoint(tag, t);
    return ok;
}

bool run_text_case(const paths & where) {
    Engine engine(options_for(where));
    const StageTimings t = engine.synthesize(TEST_TEXT).timings;
    print_timings("text", t);
    bool ok = check_common("text", t);
    ok &= zero("text", "voice-encode", t.voice_encode_ms);
    ok &= zero("text", "resample", t.resample_ms);
    return ok;
}

bool run_resample_case(const paths & where) {
    EngineOptions opts = options_for(where);
    opts.output_sample_rate = RESAMPLED_RATE;
    Engine engine(opts);
    const StageTimings t = engine.synthesize(TEST_TEXT).timings;
    print_timings("resample", t);
    bool ok = check_common("resample", t);
    ok &= positive("resample", "resample", t.resample_ms);
    return ok;
}

// The engine caches the codes of the most recent voice, so the second call
// must not charge for an encode it never ran. That only reads correctly if the
// breakdown is reset per call rather than accumulated.
bool run_clone_case(const paths & where) {
    Engine engine(options_for(where));
    const VoicePrompt voice =
        tts_cpp::audio8::load_voice_prompt(where.reference_wav, REFERENCE_TEXT);

    const StageTimings first = engine.synthesize(TEST_TEXT, voice).timings;
    print_timings("clone", first);
    bool ok = check_common("clone", first);
    ok &= positive("clone", "voice-encode", first.voice_encode_ms);

    const StageTimings second = engine.synthesize(TEST_TEXT, voice).timings;
    print_timings("clone-cached", second);
    ok &= check_common("clone-cached", second);
    ok &= zero("clone-cached", "voice-encode", second.voice_encode_ms);
    return ok;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s <lm.gguf> <decoder.gguf> <encoder.gguf> "
                             "<reference.wav> [threads]\n", argv[0]);
        return 2;
    }
    paths where;
    where.lm = argv[1];
    where.decoder = argv[2];
    where.encoder = argv[3];
    where.reference_wav = argv[4];
    where.threads = argc > 5 ? std::atoi(argv[5]) : 4;

    try {
        bool ok = run_text_case(where);
        ok &= run_resample_case(where);
        ok &= run_clone_case(where);
        std::fprintf(stderr, "%s\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL %s\n", error.what());
        return 1;
    }
}
