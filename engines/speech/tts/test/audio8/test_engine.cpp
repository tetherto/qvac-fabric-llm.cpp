// Audio8 end-to-end parity through the public engine API.
//
// The stage tests pin the language model and the codec separately against
// their own fixtures. This one drives the same path a caller drives -- text in,
// waveform out -- so it also covers what only shows up when the stages are
// joined: the ChatML prompt the tokenizer builds, the frame budget, the
// codebook-major transposition between the two halves, and, on the cloning
// side, the engine encoding a WAV into the speaker history by itself. The last
// case covers the other side of that joining: a set of GGUFs that does not
// agree on the shape of a frame has to be refused rather than indexed.
//
// Greedy decoding throughout, because that is the only trajectory the fixtures
// can pin; the sampler has its own test.

#include "tts-cpp/audio8/engine.h"

#include "ggml.h"
#include "gguf.h"
#include "gpu_arm.h"
#include "test_env_portable.h"
#include "json.hpp"
#include "npy.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr double WAVEFORM_TOLERANCE = 5e-5;
constexpr double GPU_WAVEFORM_TOLERANCE = 1e-4;
// A GPU rollout is free-running -- text and clone alike drive the LM, so the
// first argmax that lands differently re-rolls every later frame, and neither
// the codes nor the waveform can be compared to the reference trace. A
// backend whose encoder legitimately flips a few percent of prompt codes
// (see the codec harness bar) then continues into different audio, which a
// correlation bar reads as failure on content that is merely different. The
// LM's per-step numerics and token agreement are gated by the teacher-forced
// test-audio8-lm and the encoder by the codec codes rate; what the engine
// test owns on a GPU is that the rollout produces sane audio -- finite
// samples with energy on the reference's order. Correlation is reported for
// information.
constexpr double GPU_MIN_ENERGY_RATIO = 0.25;
constexpr double GPU_MAX_ENERGY_RATIO = 4.0;
constexpr int GPU_LAYERS = 99;

using audio8_test::is_gpu_test;

struct fixture {
    std::string dir;

    npy_array load(const std::string & name) const {
        return npy_load(dir + "/" + name + ".npy");
    }

    nlohmann::json meta() const {
        std::ifstream file(dir + "/meta.json");
        if (!file) throw std::runtime_error("cannot open " + dir + "/meta.json");
        return nlohmann::json::parse(file);
    }
};

struct paths {
    std::string lm;
    std::string decoder;
    std::string encoder;
    std::string reference_wav;
    int threads = 4;
};

const float * as_f32(const npy_array & array) {
    return reinterpret_cast<const float *>(array.data.data());
}

tts_cpp::audio8::EngineOptions options_for(const paths & where, const nlohmann::json & meta) {
    tts_cpp::audio8::EngineOptions opts;
    opts.lm_gguf_path = where.lm;
    opts.codec_decoder_gguf_path = where.decoder;
    opts.codec_encoder_gguf_path = where.encoder;
    opts.n_threads = where.threads;
    opts.greedy = true;
    opts.max_frames = meta.at("max_new_tokens").get<int>();
    opts.n_gpu_layers = is_gpu_test() ? GPU_LAYERS : 0;
    return opts;
}

tts_cpp::audio8::VoicePrompt voice_from(const std::string & wav_path,
                                        const nlohmann::json & meta) {
    return tts_cpp::audio8::load_voice_prompt(
        wav_path, meta.at("reference_text").get<std::string>());
}

double mean(const float * values, size_t count) {
    double sum = 0.0;
    for (size_t index = 0; index < count; ++index) sum += values[index];
    return count == 0 ? 0.0 : sum / count;
}

double correlation(const float * left, const float * right, size_t count) {
    const double left_mean = mean(left, count);
    const double right_mean = mean(right, count);
    double covariance = 0.0;
    double left_variance = 0.0;
    double right_variance = 0.0;
    for (size_t index = 0; index < count; ++index) {
        const double left_delta = left[index] - left_mean;
        const double right_delta = right[index] - right_mean;
        covariance += left_delta * right_delta;
        left_variance += left_delta * left_delta;
        right_variance += right_delta * right_delta;
    }
    return covariance / std::sqrt(left_variance * right_variance);
}

bool check_gpu_waveform_sanity(const char * tag, const std::vector<float> & got,
                               const npy_array & want) {
    double got_energy = 0, want_energy = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i])) {
            std::fprintf(stderr, "%s: FAIL non-finite samples\n", tag);
            return false;
        }
        got_energy += (double) got[i] * got[i];
        want_energy += (double) as_f32(want)[i] * as_f32(want)[i];
    }
    const double ratio = want_energy > 0 ? got_energy / want_energy : got_energy;
    std::fprintf(stderr, "%s: energy ratio %.3f, correlation %.3f against the reference\n",
                 tag, ratio, correlation(got.data(), as_f32(want), got.size()));
    if (ratio < GPU_MIN_ENERGY_RATIO || ratio > GPU_MAX_ENERGY_RATIO) {
        std::fprintf(stderr, "%s: FAIL energy ratio outside [%.2f, %.2f]\n", tag,
                     GPU_MIN_ENERGY_RATIO, GPU_MAX_ENERGY_RATIO);
        return false;
    }
    return true;
}

bool check_waveform(const char * tag, const std::vector<float> & got,
                    const npy_array & want, bool cloning) {
    if (got.size() != want.n_elements()) {
        std::fprintf(stderr, "%s: FAIL %zu samples, reference has %zu\n", tag, got.size(),
                     want.n_elements());
        return false;
    }
    if (is_gpu_test()) return check_gpu_waveform_sanity(tag, got, want);
    const compare_stats stats = compare_f32(got.data(), as_f32(want), got.size());
    print_compare(tag, stats);
    if (!std::isfinite(stats.max_abs_err)) {
        std::fprintf(stderr, "%s: FAIL non-finite samples\n", tag);
        return false;
    }
    const double tolerance = is_gpu_test() ? GPU_WAVEFORM_TOLERANCE : WAVEFORM_TOLERANCE;
    if (stats.max_abs_err > tolerance) {
        std::fprintf(stderr, "%s: FAIL max|delta| %.3e > %.1e\n", tag, stats.max_abs_err,
                     tolerance);
        return false;
    }
    return true;
}

bool check_codes(const char * tag, const std::vector<int> & got, const npy_array & want,
                 bool cloning) {
    if (want.shape.size() != 2) {
        std::fprintf(stderr, "%s codes: FAIL reference has rank %zu, expected 2\n", tag,
                     want.shape.size());
        return false;
    }
    if (got.size() != want.n_elements()) {
        std::fprintf(stderr, "%s codes: FAIL %zu values, reference has %zu\n", tag,
                     got.size(), want.n_elements());
        return false;
    }
    const int books = static_cast<int>(want.shape[0]);
    const int frames = static_cast<int>(want.shape[1]);
    const int32_t * reference = reinterpret_cast<const int32_t *>(want.data.data());
    int mismatches = 0;
    for (int frame = 0; frame < frames; ++frame) {
        for (int book = 0; book < books; ++book) {
            const size_t mine = static_cast<size_t>(frame) * books + book;
            const size_t theirs = static_cast<size_t>(book) * frames + frame;
            if (got[mine] != reference[theirs]) ++mismatches;
        }
    }
    std::printf("  [%s codes] n=%zu  mismatches=%d\n", tag, got.size(), mismatches);
    if (audio8_test::is_gpu_test()) return true;
    if (mismatches == 0) return true;
    std::fprintf(stderr, "%s codes: FAIL %d of %zu differ from the reference\n", tag,
                 mismatches, got.size());
    return false;
}

bool check_frames(const char * tag, int got, const nlohmann::json & meta) {
    const int want = meta.at("generated_frames").get<int>();
    if (got == want) return true;
    std::fprintf(stderr, "%s: FAIL %d frames, reference emitted %d\n", tag, got, want);
    return false;
}

// A GPU arm that quietly fell back to CPU, or that ran on the other arm's GPU,
// would still pass every numeric check below, so the backend the arm was
// registered for is asserted before its numbers are read as that arm's result.
bool check_backend(const char * tag, const tts_cpp::audio8::Engine & engine) {
    if (!is_gpu_test()) return true;
    const std::string name = engine.backend_name();
    // This harness pins the ggml synthesis (AUDIO8_COREML_DISABLE in main), so
    // the sidecar report must say so even when a sidecar sits next to the GGUF.
    if (engine.codec_on_coreml()) {
        std::fprintf(stderr, "engine: FAIL codec_on_coreml() is true under AUDIO8_COREML_DISABLE\n");
        return false;
    }
    if (engine.backend_device() == tts_cpp::BackendDevice::GPU &&
        audio8_test::instance_is_requested(name)) {
        return true;
    }
    return audio8_test::report_wrong_gpu(tag, name);
}

// An encoder from another checkpoint emits fewer rows per frame than the
// prompt reads, which used to be a read past the codes. The engine now
// compares the three files before loading any of them, so the regression is
// a copy of the encoder's metadata with the codebook count moved by one --
// no weights, since it is rejected before they would be read.
constexpr const char * MISMATCHED_ENCODER = "audio8-mismatched-encoder.gguf";
// Both, so the copy stays a coherent nine-codebook encoder rather than one
// that contradicts its own quantizer bank -- the point is the disagreement
// with the other two files, not a malformed file.
const char * const NARROWED_KEYS[] = {"audio8.codec.num_codebooks",
                                      "audio8.codec.residual_codebooks"};

bool narrow_by_one(gguf_context * gguf, const char * key) {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0) {
        std::fprintf(stderr, "mismatch: %s has no %s\n", MISMATCHED_ENCODER, key);
        return false;
    }
    gguf_set_val_u32(gguf, key, gguf_get_val_u32(gguf, id) - 1);
    return true;
}

bool narrow_all(gguf_context * gguf) {
    for (const char * key : NARROWED_KEYS) {
        if (!narrow_by_one(gguf, key)) return false;
    }
    return true;
}

bool write_mismatched_encoder(const std::string & source, const char * out) {
    ggml_context * meta = nullptr;
    gguf_init_params params = {/*no_alloc=*/true, &meta};
    gguf_context * gguf = gguf_init_from_file(source.c_str(), params);
    if (!gguf) {
        std::fprintf(stderr, "mismatch: cannot read %s\n", source.c_str());
        return false;
    }
    const bool written = narrow_all(gguf) && gguf_write_to_file(gguf, out,
                                                                /*only_meta=*/true);
    gguf_free(gguf);
    ggml_free(meta);
    if (!written) std::fprintf(stderr, "mismatch: cannot write %s\n", out);
    return written;
}

bool rejected_for(const char * tag, const std::exception & failure, const char * reason) {
    if (std::strstr(failure.what(), reason)) {
        std::printf("%s: rejected with \"%s\"\n", tag, failure.what());
        return true;
    }
    std::fprintf(stderr, "%s: FAIL rejected for the wrong reason: %s\n", tag,
                 failure.what());
    return false;
}

bool run_mismatch_case(const char * tag, const paths & where) {
    if (!write_mismatched_encoder(where.encoder, MISMATCHED_ENCODER)) return false;
    tts_cpp::audio8::EngineOptions opts;
    opts.lm_gguf_path = where.lm;
    opts.codec_decoder_gguf_path = where.decoder;
    opts.codec_encoder_gguf_path = MISMATCHED_ENCODER;
    opts.n_threads = where.threads;

    bool ok = false;
    try {
        tts_cpp::audio8::Engine engine(opts);
        std::fprintf(stderr, "%s: FAIL accepted an encoder the decoder disagrees with\n",
                     tag);
    } catch (const std::exception & failure) {
        ok = rejected_for(tag, failure, "the codebook count disagrees");
    }
    std::remove(MISMATCHED_ENCODER);
    return ok;
}

bool run_case(const char * tag, const paths & where, const fixture & data, bool cloning) {
    const nlohmann::json meta = data.meta();
    tts_cpp::audio8::Engine engine(options_for(where, meta));
    const std::string text = meta.at("text").get<std::string>();

    const tts_cpp::audio8::SynthesisResult result =
        cloning ? engine.synthesize(text, voice_from(where.reference_wav, meta))
                : engine.synthesize(text);
    std::printf("%s: %d frames, %.2f s at %d Hz\n", tag, result.frames, result.duration_s,
                result.sample_rate);

    bool ok = check_backend(tag, engine);
    ok &= check_frames(tag, result.frames, meta);
    ok &= check_codes(tag, result.codes, data.load("codes"), cloning);
    ok &= check_waveform(tag, result.pcm, data.load("wav"), cloning);
    return ok;
}

}  // namespace

int main(int argc, char ** argv) {
    // This harness holds the ggml stack to the torch reference at ggml
    // tolerances; the Core ML sidecar has its own parity gate
    // (test-audio8-codec-coreml-parity), so a sidecar next to the fixture
    // GGUF must not take over here.
    setenv("AUDIO8_COREML_DISABLE", "1", 1);
    if (argc < 7) {
        std::fprintf(stderr,
                     "usage: %s <lm.gguf> <codec-decoder.gguf> <codec-encoder.gguf> "
                     "<ref-dir> <clone-ref-dir> <reference.wav> [threads]\n",
                     argv[0]);
        return 1;
    }
    paths where;
    where.lm = argv[1];
    where.decoder = argv[2];
    where.encoder = argv[3];
    where.reference_wav = argv[6];
    where.threads = argc > 7 ? std::atoi(argv[7]) : 4;

    try {
        bool ok = run_case("text", where, fixture{argv[4]}, /*cloning=*/false);
        ok &= run_case("clone", where, fixture{argv[5]}, /*cloning=*/true);
        ok &= run_mismatch_case("mismatch", where);
        std::printf("\n%s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception & failure) {
        std::fprintf(stderr, "engine: %s\n", failure.what());
        return 1;
    }
}
