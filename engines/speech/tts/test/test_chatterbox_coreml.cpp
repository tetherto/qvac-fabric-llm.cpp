// Chatterbox S3Gen Core ML sidecars.
//
//   test-chatterbox-coreml                   options and their legacy aliases,
//                                            naming, function buckets, HiFT
//                                            window plan, call shapes and the
//                                            Metal estimate, the CFM rollback
//                                            with stand-in steps, fit
//   test-chatterbox-coreml --stages-fixture  each sidecar against ggml on the
//                                            same tokens, one stage at a time,
//                                            then all three; routing past the
//                                            widest function and strict mode
//   test-chatterbox-coreml --shape-fixture   every feature shape a sidecar
//                                            declares is checked before
//                                            predicting
//   test-chatterbox-coreml --pending-fixture background loading and routing
//                                            on tiny multi-width sidecars:
//                                            pending calls, width choice,
//                                            load order, bounded waits, the
//                                            resident cap, retirement and
//                                            destruction mid-load
//   test-chatterbox-coreml --engine-fixture  Engine sentence overlap against
//                 [--slow-load]              in-order sequential synthesis,
//                                            cancellation, worker errors, the
//                                            CFM rollback; with --slow-load,
//                                            the first synthesis while every
//                                            function is still loading equals
//                                            a use = off run, through the
//                                            Engine and tts-cli
//
// The stages fixture reads CHATTERBOX_COREML_TEST_S3GEN (an S3Gen GGUF with
// the exported `<stem>-{encoder,cfm,hift}.mlmodelc` next to it); the engine
// fixture also reads CHATTERBOX_COREML_TEST_T3 (the matching T3 GGUF) and,
// with --slow-load, CHATTERBOX_COREML_TEST_CLI (a tts-cli binary); the shape
// and pending fixtures read CHATTERBOX_COREML_TEST_SHAPES (the directory
// prepare_chatterbox_coreml_shape_fixture.py writes). Each reports a skip
// (77) without its inputs or without Core ML.

#include "chatterbox_coreml.h"
#include "gguf.h"
#include "coreml_sidecar_path.h"
#include "npy.h"
#include "test_env_portable.h"
#include "text_preprocess.h"
#include "tts-cpp/chatterbox/engine.h"
#include "tts-cpp/chatterbox/fit.h"
#include "tts-cpp/chatterbox/s3gen_pipeline.h"
#include "tts-cpp/tts-cpp.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#endif

namespace cb = tts_cpp::chatterbox::detail;
namespace fs = std::filesystem;
using tts_cpp::chatterbox::Engine;
using tts_cpp::chatterbox::EngineOptions;
using tts_cpp::chatterbox::SynthesisResult;

namespace {

constexpr int SKIP = 77;

void require(bool ok, const std::string & what) {
    if (!ok) throw std::runtime_error(what);
}

struct env_guard {
    std::string name;
    bool present;
    std::string value;
    explicit env_guard(const char * n) : name(n), present(std::getenv(n) != nullptr),
                                          value(present ? std::getenv(n) : "") {}
    ~env_guard() {
        if (present) setenv(name.c_str(), value.c_str(), 1);
        else unsetenv(name.c_str());
    }
};

struct temp_dir {
    fs::path path;
    explicit temp_dir(const std::string & tag)
        : path(fs::path(test_tmpdir()) / ("chatterbox-coreml-" + tag + "-" + test_process_tag())) {
        std::error_code error;
        fs::remove_all(path, error);
        fs::create_directories(path);
    }
    ~temp_dir() {
        std::error_code error;
        fs::remove_all(path, error);
    }
};

using tts_cpp::CoremlOptions;

double max_abs_vec(const std::vector<float> & a, const std::vector<float> & b) {
    double m = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) m = std::max(m, (double) std::fabs(a[i] - b[i]));
    return m;
}

template <class E, class F> bool throws(F && f) {
    try {
        f();
    } catch (const E &) {
        return true;
    }
    return false;
}

void check_options() {
    // The legacy s3gen_synthesize_opts / EngineOptions fields are aliases:
    // each fills its CoremlOptions field only when that is unset.
    CoremlOptions none;
    auto merged = cb::merge_legacy_coreml(none, "cpu_only", 1, false);
    require(merged.compute_units == std::string("cpu_only") && merged.use == CoremlOptions::Use::require,
            "legacy units and strict fill unset fields");
    merged = cb::merge_legacy_coreml(none, "", 0, false);
    require(!merged.compute_units && merged.use == CoremlOptions::Use::automatic,
            "an explicit non-strict alias overrides the strict environment variable");
    require(cb::merge_legacy_coreml(none, "", -1, false).use == CoremlOptions::Use::unset,
            "an unset strict alias leaves use unset");
    CoremlOptions set;
    set.compute_units = "all";
    set.use = CoremlOptions::Use::automatic;
    merged = cb::merge_legacy_coreml(set, "cpu_only", 1, false);
    require(merged.compute_units == std::string("all") && merged.use == CoremlOptions::Use::automatic,
            "explicit CoremlOptions fields win over the aliases");
    require(cb::merge_legacy_coreml(set, "", 1, true).use == CoremlOptions::Use::automatic,
            "an explicit use wins over coreml_disabled");
    CoremlOptions required;
    required.use = CoremlOptions::Use::require;
    require(cb::merge_legacy_coreml(required, "", -1, true).use == CoremlOptions::Use::require,
            "an explicit use = require is never turned off by coreml_disabled");
    require(cb::merge_legacy_coreml(none, "", 1, true).use == CoremlOptions::Use::off,
            "coreml_disabled is use = off when use is unset, and wins over coreml_strict");
    s3gen_synthesize_opts explicit_require;
    explicit_require.coreml.use = CoremlOptions::Use::require;
    explicit_require.coreml_disabled = true;
    require(cb::call_coreml_options(explicit_require).use == CoremlOptions::Use::require,
            "a call's explicit use = require survives its coreml_disabled alias");

    s3gen_synthesize_opts sopts;
    sopts.coreml_compute_units = "cpu_and_gpu";
    sopts.coreml_strict = 1;
    merged = cb::call_coreml_options(sopts);
    require(merged.compute_units == std::string("cpu_and_gpu") && merged.use == CoremlOptions::Use::require,
            "a call's options fold its aliases in");

    CoremlOptions bad;
    bad.compute_units = "gpu_please";
    require(throws<std::invalid_argument>([&] { cb::validate_coreml_options(bad); }),
            "invalid explicit units throw in every build");
    require(throws<std::invalid_argument>([&] { cb::merge_legacy_coreml(none, "gpu_please", -1, false);
                                                cb::validate_coreml_options(
                                                    cb::merge_legacy_coreml(none, "gpu_please", -1, false)); }),
            "an invalid legacy unit throws once folded in");
    CoremlOptions negative;
    negative.strict_wait_ms = -1;
    require(throws<std::invalid_argument>([&] { cb::validate_coreml_options(negative); }),
            "a negative strict wait throws");
    for (const char * units : {"all", "cpu_and_ane", "cpu_and_gpu", "cpu_only"}) {
        require(cb::coreml_valid_units(units), std::string("valid units ") + units);
    }
}

// With no sidecar in use a request group is empty: a success with nothing
// to prepare, in every build.
void check_empty_group() {
    const std::string json = cb::coreml_empty_group_json("prewarm");
    require(json.find("\"success\":true") != std::string::npos &&
                json.find("\"nothing_to_prepare\":true") != std::string::npos &&
                json.find("\"group\":\"prewarm\"") != std::string::npos && json.find("\"slots\":[]") != std::string::npos,
            "an empty group succeeds with nothing to prepare: " + json);
}

// tts-cli (CHATTERBOX_COREML_TEST_CLI; CTest sets it) refuses two writers of
// stdout, the raw PCM of --out - and the status JSON of
// --coreml-status-json -, before it loads anything.
void check_cli_arguments() {
#ifndef _WIN32
    const char * cli = std::getenv("CHATTERBOX_COREML_TEST_CLI");
    if (!cli || !*cli || !fs::exists(cli)) return;
    temp_dir dir("cli-arguments");
    const fs::path log = dir.path / "stderr.log", out = dir.path / "stdout.out";
    const std::string cmd = "'" + std::string(cli) + "' --model none.gguf --s3gen-gguf none.gguf --text hello " +
                            "--out - --coreml-status-json - > '" + out.string() + "' 2> '" + log.string() + "'";
    const int status = std::system(cmd.c_str());
    const int rc = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    std::ifstream err(log);
    const std::string text((std::istreambuf_iterator<char>(err)), std::istreambuf_iterator<char>());
    require(rc != 0 && text.find("--coreml-status-json - and --out -") != std::string::npos &&
                fs::file_size(out) == 0,
            "tts-cli rejects --out - with --coreml-status-json -");
    // The same file under two spellings: the status would replace the WAV.
    const fs::path wav = dir.path / "result.wav";
    const std::string same = "'" + std::string(cli) + "' --model none.gguf --s3gen-gguf none.gguf --text hello " +
                             "--out '" + wav.string() + "' --coreml-status-json '" + (dir.path / "." / "result.wav").string() +
                             "' > '" + out.string() + "' 2> '" + log.string() + "'";
    const int same_status = std::system(same.c_str());
    std::ifstream same_err(log);
    const std::string same_text((std::istreambuf_iterator<char>(same_err)), std::istreambuf_iterator<char>());
    require(WIFEXITED(same_status) && WEXITSTATUS(same_status) != 0 &&
                same_text.find("--coreml-status-json and --out name the same file") != std::string::npos &&
                !fs::exists(wav),
            "tts-cli rejects a status JSON path that names its WAV output");
    // A spelling only the filesystem equates (case, on a case-insensitive
    // volume) of a WAV that already exists: refused, the WAV untouched. (The
    // engine fixture covers the WAV a run writes after validation.)
    const fs::path upper = dir.path / "Result.wav", lower = dir.path / "result.wav";
    { std::ofstream(upper, std::ios::binary) << "RIFF-audio"; }
    std::error_code ec;
    if (fs::equivalent(upper, lower, ec)) {
        const std::string cased = "'" + std::string(cli) + "' --model none.gguf --s3gen-gguf none.gguf --text hello " +
                                  "--out '" + upper.string() + "' --coreml-status-json '" + lower.string() + "' > '" +
                                  out.string() + "' 2> '" + log.string() + "'";
        (void) std::system(cased.c_str());
        std::ifstream cased_err(log), audio(upper, std::ios::binary);
        const std::string cased_text((std::istreambuf_iterator<char>(cased_err)), std::istreambuf_iterator<char>());
        const std::string audio_text((std::istreambuf_iterator<char>(audio)), std::istreambuf_iterator<char>());
        require(audio_text == "RIFF-audio" && (cased_text.find("name the same file") != std::string::npos ||
                                                cased_text.find("is the --out file") != std::string::npos),
                "tts-cli never writes the status over a case-differing spelling of its WAV");
    } else {
        std::puts("[cli arguments] case-sensitive volume: the case-differing check does not apply");
    }
#endif
}

// The test hooks record nothing unless a test turns tracing on (production
// loads never grow them).
void check_hooks_record_only_when_tracing() {
    auto & hooks = cb::coreml_test();
    hooks.clear_loads();
    hooks.clear_events();
    hooks.record_load("cfm/frames_16");
    hooks.trace("t3 0");
    require(hooks.loads().empty() && hooks.events().empty(), "the hooks record nothing by default");
    hooks.tracing = true;
    hooks.record_load("cfm/frames_16");
    hooks.trace("t3 0");
    hooks.tracing = false;
    require(hooks.loads().size() == 1 && hooks.events().size() == 1, "the hooks record while tracing");
    hooks.clear_loads();
    hooks.clear_events();
}

void check_naming() {
    require(cb::coreml_stage_path("models/chatterbox-s3gen-q8_0.gguf", cb::COREML_CFM_SUFFIX) ==
                "models/chatterbox-s3gen-cfm.mlmodelc", "turbo CFM sidecar name");
    require(cb::coreml_stage_path("m/chatterbox-s3gen-mtl-f32.gguf", cb::COREML_HIFT_SUFFIX) ==
                "m/chatterbox-s3gen-mtl-hift.mlmodelc", "multilingual HiFT sidecar name");
    require(cb::coreml_stage_path("chatterbox-s3gen.gguf", cb::COREML_ENCODER_SUFFIX) ==
                "chatterbox-s3gen-encoder.mlmodelc", "untagged encoder sidecar name");
}

void check_buckets() {
    const auto frames = cb::coreml_buckets("frames_640,tokens_256,frames_512,frames_,frames_x1,frames_640,main",
                                           cb::COREML_CFM_PREFIX);
    require(frames == std::vector<int>{512, 640}, "frame buckets");
    require(cb::coreml_buckets("tokens_256,tokens_320", cb::COREML_ENCODER_PREFIX) == std::vector<int>{256, 320},
            "token buckets");
    require(cb::coreml_pick_bucket(frames, 1) == 512 && cb::coreml_pick_bucket(frames, 512) == 512 &&
                cb::coreml_pick_bucket(frames, 513) == 640 && cb::coreml_pick_bucket(frames, 641) == -1,
            "bucket selection");
}

void check_windows() {
    const int window = 128, context = cb::HIFT_CONTEXT_FRAMES;
    require(cb::coreml_hift_windows(10, 2 * context, context).empty(), "a window without room for context");
    for (int frames = 1; frames <= 700; ++frames) {
        const auto plan = cb::coreml_hift_windows(frames, window, context);
        int covered = 0;
        for (const auto & w : plan) {
            require(w.keep_begin == covered, "kept spans tile the utterance");
            require(w.keep_end > w.keep_begin && w.start <= w.keep_begin, "non-empty kept span inside its window");
            require(w.keep_end - w.start <= window, "kept span inside the window");
            require(w.start == 0 || w.keep_begin - w.start >= context, "left context");
            require(w.keep_end == frames || w.start + window - w.keep_end >= context, "right context");
            covered = w.keep_end;
        }
        require(covered == frames, "every frame kept once");
        require(frames > window || plan.size() == 1, "one call when the utterance fits");
    }
}

void check_shapes_and_estimate() {
    // A batch call: prompt + speech + 3 lookahead tokens, two frames each.
    auto batch = cb::s3gen_shape(250, 120, 2, true, true);
    require(batch.tokens == 373 && batch.frames == 746, "batch call shape");
    // A non-final streaming chunk: no lookahead, the last 6 frames trimmed.
    auto chunk = cb::s3gen_shape(250, 25, 2, false, false);
    require(chunk.tokens == 275 && chunk.frames == 544, "streaming chunk shape");

    // The call shape follows the call's own prompt precedence and drops
    // tokens outside the speech vocabulary, as s3gen_synthesize_to_wav does.
    cb::s3gen_gguf_facts facts;
    facts.builtin_prompt_tokens = 250;
    facts.speech_vocab = 6561;
    s3gen_synthesize_opts sopts;
    const std::vector<int32_t> tokens = {1, 2, 6561, 7000, -1, 3};
    require(cb::s3gen_call_shape_for(sopts, tokens, facts).tokens == 250 + 3 + 3, "built-in voice prompt");
    const std::vector<int32_t> prompt(40, 5);
    sopts.prompt_token_view_data = prompt.data();
    sopts.prompt_token_view_size = prompt.size();
    require(cb::s3gen_call_shape_for(sopts, tokens, facts).tokens == 40 + 3 + 3, "a prompt view replaces it");
    s3gen_synthesize_opts embedding_only;
    embedding_only.embedding_override.assign(192, 0.0f);
    require(cb::s3gen_call_shape_for(embedding_only, tokens, facts).tokens == 0 + 3 + 3,
            "conditioning without a prompt has no prompt tokens");

    // The Metal estimate is the measured per-variant line, never negative.
    require(cb::s3gen_estimate_metal_ms_frames(true, 0) == 0.0 && cb::s3gen_estimate_metal_ms_frames(false, 0) == 0.0,
            "the estimate is clamped at zero");
    const double turbo = cb::s3gen_estimate_metal_ms_frames(true, 1000);
    const double mtl = cb::s3gen_estimate_metal_ms_frames(false, 1000);
    require(std::fabs(turbo - (-1347.8 + 2.9867 * 1000)) < 1e-6 && std::fabs(mtl - (-1992.6 + 9.2146 * 1000)) < 1e-6,
            "per-variant Metal cost lines");
    require(cb::s3gen_estimate_metal_ms_frames(true, 1200) > turbo, "the estimate grows with the frames");
}

// The CFM integration with stand-in steps: a Core ML failure at a later step
// restores the initial noise and integrates every step on ggml, so the result
// equals a whole ggml integration; nothing in it draws random numbers.
void check_cfm_rollback() {
    std::mt19937 rng(11);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::vector<float> z0(64);
    for (float & v : z0) v = gauss(rng);
    const size_t steps = 4;
    // The two backends differ slightly, as fp16 Core ML and ggml do.
    auto make_step = [&](int fail_at, std::vector<std::pair<size_t, bool>> & trace) {
        return [fail_at, &trace](size_t s, std::vector<float> & z, bool on_coreml) -> int {
            trace.emplace_back(s, on_coreml);
            if (on_coreml && (int) s == fail_at) throw cb::coreml_stage_failure("stand-in failure");
            const float rate = on_coreml ? 0.251f : 0.25f;
            for (size_t i = 0; i < z.size(); ++i) z[i] += rate * std::sin(z[i] + (float) s);
            return 0;
        };
    };
    std::vector<std::pair<size_t, bool>> ggml_trace;
    std::vector<float> ggml_z = z0;
    require(cb::cfm_integrate(ggml_z, steps, false, make_step(-1, ggml_trace), [](auto &, size_t) {
                throw std::runtime_error("no failure expected");
            }) == 0,
            "ggml integration");

    std::vector<std::pair<size_t, bool>> coreml_trace;
    std::vector<float> coreml_z = z0;
    cb::cfm_integrate(coreml_z, steps, true, make_step(-1, coreml_trace), [](auto &, size_t) {});
    require(max_abs_vec(coreml_z, ggml_z) > 0.0, "the stand-in backends differ");

    for (int fail_at : {1, 2, 3}) {
        std::vector<std::pair<size_t, bool>> trace;
        std::vector<float> z = z0;
        int failures = 0;
        size_t failed_step = 0;
        const int rc = cb::cfm_integrate(z, steps, true, make_step(fail_at, trace),
                                         [&](const cb::coreml_stage_failure &, size_t s) {
                                             ++failures;
                                             failed_step = s;
                                         });
        require(rc == 0 && failures == 1 && failed_step == (size_t) fail_at, "one failure, reported at its step");
        require(z == ggml_z, "a failure at step " + std::to_string(fail_at + 1) +
                                 " gives exactly the whole ggml integration from the initial noise");
        // Core ML for steps 0..fail_at, then every step again on ggml.
        size_t i = 0;
        for (; i <= (size_t) fail_at; ++i) require(trace[i] == std::make_pair(i, true), "Core ML until the failure");
        for (size_t s = 0; s < steps; ++s, ++i) require(trace[i] == std::make_pair(s, false), "then ggml from step 0");
        require(trace.size() == i, "nothing after the ggml pass");
    }

    // A failure handler that stops the call (strict) propagates; a step's
    // non-zero result (a cancel) stops the integration.
    std::vector<std::pair<size_t, bool>> trace;
    std::vector<float> z = z0;
    require(throws<std::runtime_error>([&] {
                cb::cfm_integrate(z, steps, true, make_step(1, trace),
                                  [](const cb::coreml_stage_failure & e, size_t) { throw std::runtime_error(e.what()); });
            }),
            "strict stops the integration");
    require(cb::cfm_integrate(z, steps, false,
                              [](size_t s, std::vector<float> &, bool) { return s == 1 ? 2 : 0; },
                              [](auto &, size_t) {}) == 2,
            "a step's result stops the integration");
}

// Planning reads token_mel_ratio with the loader's limit: a ratio-10 GGUF
// (valid for the loader) plans 10 frames per token, so a call of 11 tokens
// takes the 112-frame width, not the undersized 96 a fallback ratio of 2
// would give (22 frames).
void check_ratio_planning() {
    temp_dir dir("ratio");
    auto write = [&](const std::string & name, uint32_t ratio) {
        gguf_context * g = gguf_init_empty();
        gguf_set_val_u32(g, "s3gen.encoder.token_mel_ratio", ratio);
        gguf_set_val_u32(g, "s3gen.speech_vocab_size", 6561);
        const std::string path = (dir.path / name).string();
        require(gguf_write_to_file(g, path.c_str(), /*only_meta=*/true), "write a metadata-only GGUF");
        gguf_free(g);
        return path;
    };
    const std::string ten = write("ratio10.gguf", 10);
    const cb::s3gen_gguf_facts facts = cb::s3gen_read_facts(ten);
    require(facts.token_mel_ratio == 10, "a ratio-10 GGUF plans with ratio 10");
    s3gen_synthesize_opts opts;
    opts.s3gen_gguf_path = ten;
    const cb::s3gen_call_shape shape = cb::s3gen_call_shape_for(opts, std::vector<int32_t>(8, 1), facts);
    require(shape.tokens == 11 && shape.frames == 110, "8 speech tokens + 3 lookahead at ratio 10 = 110 frames");
    require(cb::coreml_pick_bucket({96, 112, 128}, shape.frames) == 112, "the covering CFM width is 112");
    require(cb::s3gen_read_facts(write("ratio16.gguf", cb::S3GEN_MAX_TOKEN_MEL_RATIO)).token_mel_ratio ==
                cb::S3GEN_MAX_TOKEN_MEL_RATIO,
            "the loader's largest ratio plans as itself");
}

void check_warm_estimates() {
    require(cb::coreml_expected_speech_tokens(0) == 0 && cb::coreml_expected_speech_tokens(100) == 115,
            "speech-token estimate");
    // 60 tokens streamed in 25s, first chunk 10: cumulative windows, or
    // capped by 25 tokens of left context.
    require(cb::coreml_stream_windows(60, 10, 25, 0) == std::vector<int>{10, 35, 60}, "cumulative windows");
    require(cb::coreml_stream_windows(60, 10, 25, 25) == std::vector<int>{10, 35, 50}, "left-context windows");
    require(cb::coreml_stream_windows(0, 10, 25, 0).empty() && cb::coreml_stream_windows(5, 0, 0, 0).empty(),
            "no windows");
}

bool same_fit(const tts_cpp::FitResult & a, const tts_cpp::FitResult & b) {
    return a.status == b.status && a.fits == b.fits && a.reason == b.reason && a.model_variant == b.model_variant &&
           a.device_name == b.device_name && a.report == b.report;
}

// The same projection: variant, device and every projected byte count (the
// free-memory reading, and so the verdict text, may differ between calls).
bool same_projection(const tts_cpp::FitResult & a, const tts_cpp::FitResult & b) {
    return a.model_variant == b.model_variant && a.device_name == b.device_name &&
           a.device.weights_bytes == b.device.weights_bytes && a.device.state_bytes == b.device.state_bytes &&
           a.device.lm_compute_bytes == b.device.lm_compute_bytes &&
           a.device.codec_compute_bytes == b.device.codec_compute_bytes &&
           a.device.total_bytes == b.device.total_bytes && a.host_bytes == b.host_bytes;
}

bool projected(const tts_cpp::FitResult & r) {
    return (r.reason == "fits" || r.reason == "does-not-fit") && r.device.total_bytes > 0;
}

void check_fit_and_detection() {
    env_guard disabled(cb::COREML_DISABLE_ENV);
    unsetenv(cb::COREML_DISABLE_ENV);
    temp_dir dir("fit");
    tts_cpp::chatterbox::FitOptions options;
    options.t3_gguf_path = (dir.path / "chatterbox-t3-turbo-q8_0.gguf").string();
    options.s3gen_gguf_path = (dir.path / "chatterbox-s3gen-q8_0.gguf").string();
    options.backend = "cpu";
    const auto plain = tts_cpp::chatterbox::fit_params(options);
    require(plain.reason != "coreml-memory-estimate-unavailable", "no sidecar, no Core ML refusal");
    require(!cb::coreml_prebuilt_route(options.s3gen_gguf_path, {}), "nothing to detect");

    for (const char * suffix : {cb::COREML_ENCODER_SUFFIX, cb::COREML_CFM_SUFFIX, cb::COREML_HIFT_SUFFIX}) {
        const std::string sidecar = cb::coreml_stage_path(options.s3gen_gguf_path, suffix);
        fs::create_directory(sidecar);
        const auto result = tts_cpp::chatterbox::fit_params(options);
#ifdef TTS_CPP_USE_COREML
        require(cb::coreml_prebuilt_route(options.s3gen_gguf_path, {}), "sidecar detected");
        require(result.status == tts_cpp::FitStatus::Error && !result.fits &&
                    result.reason == "coreml-memory-estimate-unavailable" && !result.report.empty(),
                std::string("fit refuses with an enabled ") + suffix + " sidecar");
#else
        require(result.reason != "coreml-memory-estimate-unavailable", "no Core ML refusal without Core ML");
#endif
        // use = off and the kill switch skip the refusal (the projections
        // themselves are compared on real GGUFs by the engine fixture).
        auto off = options;
        off.coreml.use = CoremlOptions::Use::off;
        require(tts_cpp::chatterbox::fit_params(off).reason != "coreml-memory-estimate-unavailable",
                "fit with use = off does not refuse");
        setenv(cb::COREML_DISABLE_ENV, "1", 1);
        require(tts_cpp::chatterbox::fit_params(options).reason != "coreml-memory-estimate-unavailable",
                "CHATTERBOX_COREML_DISABLE does not refuse");
        unsetenv(cb::COREML_DISABLE_ENV);
        fs::remove(sidecar);
    }
    auto bad = options;
    bad.coreml.compute_units = "gpu_please";
    require(tts_cpp::chatterbox::fit_params(bad).reason == "invalid-arguments", "invalid Core ML options");
}

// ---------------------------------------------------------------- fixture --

struct run_result {
    std::vector<float> wav;
    std::vector<float> mu;
    std::vector<float> mel;
    std::string backend;
};

std::vector<float> load_npy(const std::string & path) {
    npy_array a = npy_load(path);
    const float * data = reinterpret_cast<const float *>(a.data.data());
    return std::vector<float>(data, data + a.n_elements());
}

run_result synthesize(const std::string & gguf, const std::vector<int32_t> & tokens, const fs::path & scratch,
                      bool disabled, int strict) {
    s3gen_synthesize_opts opts;
    opts.s3gen_gguf_path = gguf;
    opts.n_gpu_layers = 99;
    opts.seed = 42;
    opts.coreml_disabled = disabled;
    opts.coreml_strict = strict;
    run_result r;
    opts.pcm_out = &r.wav;
    opts.s3gen_backend_out = &r.backend;
    opts.dump_mel_path = (scratch / "dump.npy").string();
    if (s3gen_synthesize_to_wav(tokens, opts) != 0) throw std::runtime_error("s3gen_synthesize_to_wav failed");
    r.mu = load_npy((scratch / "dump_mu.npy").string());
    r.mel = load_npy((scratch / "dump.npy").string());
    return r;
}

double cosine(const std::vector<float> & a, const std::vector<float> & b) {
    require(a.size() == b.size() && !a.empty(), "compared tensors differ in size");
    double ab = 0, aa = 0, bb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        ab += (double) a[i] * b[i];
        aa += (double) a[i] * a[i];
        bb += (double) b[i] * b[i];
    }
    return ab / (std::sqrt(aa * bb) + 1e-30);
}

double max_abs(const std::vector<float> & a, const std::vector<float> & b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, (double) std::fabs(a[i] - b[i]));
    return m;
}

// Directory holding a link to the GGUF and links to the named sidecars only.
fs::path stage_dir(const fs::path & root, const std::string & gguf, const std::vector<const char *> & suffixes,
                   const std::string & tag) {
    const fs::path dir = root / tag;
    fs::create_directories(dir);
    const fs::path link = dir / fs::path(gguf).filename();
    fs::create_symlink(fs::absolute(gguf), link);
    for (const char * suffix : suffixes) {
        const fs::path sidecar = cb::coreml_stage_path(gguf, suffix);
        fs::create_directory_symlink(fs::absolute(sidecar), cb::coreml_stage_path(link.string(), suffix));
    }
    return dir;
}

std::vector<int32_t> fixture_tokens(int count) {
    std::mt19937 rng(7);
    std::uniform_int_distribution<int32_t> token(0, 6560);
    std::vector<int32_t> tokens((size_t) count);
    for (auto & t : tokens) t = token(rng);
    return tokens;
}

int stages_fixture() {
#ifndef TTS_CPP_USE_COREML
    std::puts("SKIP: built without TTS_CPP_COREML");
    return SKIP;
#else
    const char * env = std::getenv("CHATTERBOX_COREML_TEST_S3GEN");
    if (!env || !*env) {
        std::puts("SKIP: CHATTERBOX_COREML_TEST_S3GEN is not set");
        return SKIP;
    }
    const std::string gguf = env;
    for (const char * suffix : {cb::COREML_ENCODER_SUFFIX, cb::COREML_CFM_SUFFIX, cb::COREML_HIFT_SUFFIX}) {
        if (!fs::exists(cb::coreml_stage_path(gguf, suffix))) {
            std::printf("SKIP: %s is missing\n", cb::coreml_stage_path(gguf, suffix).c_str());
            return SKIP;
        }
    }
    env_guard units(cb::COREML_UNITS_ENV), strict(cb::COREML_STRICT_ENV), disabled(cb::COREML_DISABLE_ENV);
    unsetenv(cb::COREML_UNITS_ENV);
    unsetenv(cb::COREML_STRICT_ENV);
    unsetenv(cb::COREML_DISABLE_ENV);
    temp_dir root("stages");
    const std::vector<int32_t> tokens = fixture_tokens(90);

    const fs::path base = stage_dir(root.path, gguf, {}, "ggml");
    const std::string base_gguf = (base / fs::path(gguf).filename()).string();
    const run_result ref = synthesize(base_gguf, tokens, base, true, -1);
    require(ref.backend == cb::GGML_STAGE_BACKEND, "ggml reference reports ggml");

    auto stage = [&](const char * suffix, const char * name) {
        const fs::path dir = stage_dir(root.path, gguf, {suffix}, name);
        return synthesize((dir / fs::path(gguf).filename()).string(), tokens, dir, false, 1);
    };
    const run_result enc = stage(cb::COREML_ENCODER_SUFFIX, "encoder");
    const double enc_cos = cosine(enc.mu, ref.mu);
    std::printf("[encoder] backend=%s mu cosine=%.6f max_abs=%.3e\n", enc.backend.c_str(), enc_cos,
                max_abs(enc.mu, ref.mu));
    require(enc.backend == cb::MIXED_STAGE_BACKEND && enc_cos > 0.9999, "encoder parity");

    const run_result cfm = stage(cb::COREML_CFM_SUFFIX, "cfm");
    const double cfm_mu = cosine(cfm.mu, ref.mu), cfm_cos = cosine(cfm.mel, ref.mel);
    std::printf("[cfm] mel cosine=%.6f max_abs=%.3e\n", cfm_cos, max_abs(cfm.mel, ref.mel));
    require(cfm_mu > 0.999999 && cfm_cos > 0.999, "CFM parity");

    const run_result hift = stage(cb::COREML_HIFT_SUFFIX, "hift");
    const double hift_cos = cosine(hift.wav, ref.wav);
    std::printf("[hift] waveform cosine=%.6f max_abs=%.3e\n", hift_cos, max_abs(hift.wav, ref.wav));
    require(cosine(hift.mel, ref.mel) > 0.999999 && hift_cos > 0.995, "HiFT parity");

    const fs::path all_dir = stage_dir(root.path, gguf,
                                       {cb::COREML_ENCODER_SUFFIX, cb::COREML_CFM_SUFFIX, cb::COREML_HIFT_SUFFIX}, "all");
    const std::string all_gguf = (all_dir / fs::path(gguf).filename()).string();
    const run_result all = synthesize(all_gguf, tokens, all_dir, false, 1);
    const double all_cos = cosine(all.mel, ref.mel);
    std::printf("[all] backend=%s mel cosine=%.6f waveform cosine=%.6f\n", all.backend.c_str(), all_cos,
                cosine(all.wav, ref.wav));
    require(all.backend == "coreml-ane" && all_cos > 0.999 && all.wav.size() == ref.wav.size(), "all stages");

    // Past the widest encoder/CFM function those stages stay on ggml (HiFT
    // windows any length); strict turns that fallback into an error.
    const std::vector<int32_t> long_tokens = fixture_tokens(1200);
    const run_result past = synthesize(all_gguf, long_tokens, all_dir, false, 0);
    std::printf("[routing] %zu tokens: backend=%s\n", long_tokens.size(), past.backend.c_str());
    require(past.backend == cb::MIXED_STAGE_BACKEND, "fallback past the widest function");
    bool threw = false;
    try {
        synthesize(all_gguf, long_tokens, all_dir, false, 1);
    } catch (const std::runtime_error & e) {
        threw = std::string(e.what()).find("cannot serve") != std::string::npos;
    }
    require(threw, "strict refuses an input past the widest function");
    s3gen_unload();
    return 0;
#endif
}

// ---------------------------------------------------------- shape fixture --

// The features each stage packs, as prepare_chatterbox_coreml_shape_fixture.py
// writes them: good sidecars declare exactly these (encoder tokens_8, CFM
// frames_16, HiFT window 32); `<stage>-<feature>` declares one of them with a
// doubled leading dimension.
const std::vector<std::pair<std::string, std::vector<std::string>>> SHAPE_FEATURES = {
    {"encoder", {"embedding", "valid", "mu"}},
    {"cfm", {"x", "mu", "cond", "spks", "t_emb", "valid", "velocity"}},
    {"hift", {"mel", "source", "valid", "spectrum"}},
};

// Shape-fixture sidecars on the CPU, each acquisition waiting for its load.
CoremlOptions fixture_options() {
    CoremlOptions o;
    o.compute_units = "cpu_only";
    o.wait = CoremlOptions::Wait::call;
    o.prewarm = CoremlOptions::Prewarm::none;
    return o;
}

// One call of `stage` on inputs narrower than its function; true when served.
bool serve(cb::s3gen_sidecars & sidecars, const std::string & stage) {
    const auto plan = sidecars.acquire({6, 12}, std::chrono::steady_clock::now());
    require(plan != nullptr, "a plan");
    std::mt19937 rng(3);
    std::normal_distribution<float> noise(0.0f, 1.0f);
    auto random = [&](size_t n) {
        std::vector<float> v(n);
        for (float & x : v) x = noise(rng);
        return v;
    };
    if (stage == "encoder") {
        constexpr int tokens = 6;
        std::vector<float> mu;
        const bool ok = cb::plan_encode(*plan, random((size_t) tokens * 512), tokens, 2, mu);
        require(!ok || mu.size() == (size_t) tokens * 2 * 80, "encoder output size");
        return ok;
    }
    if (stage == "cfm") {
        constexpr int frames = 12;
        const auto x = random(80 * frames), mu = random(80 * frames), cond = random(80 * frames);
        const auto spks = random(80), t_emb = random(1024);
        std::vector<std::vector<float>> out;
        // Like the pipeline: the CFG pair, of which a batch-1 sidecar uses the first.
        const bool ok = cb::plan_velocity(*plan, 0, 1, frames, {x.data(), x.data()}, {mu.data(), mu.data()},
                                          {cond.data(), cond.data()}, {spks.data(), spks.data()},
                                          {t_emb.data(), t_emb.data()}, out);
        require(!ok || (out.size() == 1 && out[0].size() == (size_t) 80 * frames), "CFM output size");
        return ok;
    }
    constexpr int frames = 40;  // two HiFT windows
    const size_t stft = (size_t) frames * cb::HIFT_STFT_PER_FRAME + 1;
    std::vector<float> spectrum;
    const bool ok = cb::plan_vocode(*plan, random((size_t) 80 * frames), frames, random(cb::HIFT_STFT_BINS * stft),
                                    spectrum);
    require(!ok || spectrum.size() == cb::HIFT_STFT_BINS * stft, "HiFT output size");
    return ok;
}

int shape_fixture() {
#ifndef TTS_CPP_USE_COREML
    std::puts("SKIP: built without TTS_CPP_COREML");
    return SKIP;
#else
    const char * env = std::getenv("CHATTERBOX_COREML_TEST_SHAPES");
    if (!env || !*env) {
        std::puts("SKIP: CHATTERBOX_COREML_TEST_SHAPES is not set");
        return SKIP;
    }
    const fs::path dir = env;
    {
        auto good = cb::s3gen_sidecars::create((dir / "good-s3gen.gguf").string(), fixture_options());
        require(good != nullptr, "the good sidecars are found");
        for (const auto & [stage, features] : SHAPE_FEATURES) {
            require(serve(*good, stage), "the matching " + stage + " sidecar serves");
        }
    }
    // The shared runner copies each feature by its declared shape, so every
    // one must be rejected before predicting, not only the ones the old
    // checks covered.
    for (const auto & [stage, features] : SHAPE_FEATURES) {
        for (const std::string & feature : features) {
            const std::string stem = stage + "-" + feature + "-s3gen";
            require(fs::exists(dir / (stem + "-" + stage + ".mlmodelc")), "fixture has " + stem);
            auto sidecars = cb::s3gen_sidecars::create((dir / (stem + ".gguf")).string(), fixture_options());
            require(sidecars != nullptr, "the " + stem + " sidecar is found");
            std::string error;
            try {
                serve(*sidecars, stage);
            } catch (const std::runtime_error & e) {
                error = e.what();
            }
            std::printf("[%s %s] %s\n", stage.c_str(), feature.c_str(), error.c_str());
            require(error.find("declares " + feature + " ") != std::string::npos,
                    stage + " rejects a sidecar declaring a larger " + feature);
        }
    }
    return 0;
#endif
}

// -------------------------------------------------------- pending fixture --

#ifdef TTS_CPP_USE_COREML
using steady = std::chrono::steady_clock;

double ms_since(steady::time_point t0) {
    return std::chrono::duration<double, std::milli>(steady::now() - t0).count();
}

bool wait_for(const std::function<bool()> & done, int timeout_ms) {
    const auto until = steady::now() + std::chrono::milliseconds(timeout_ms);
    while (!done()) {
        if (steady::now() > until) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return true;
}

// No function of the set is queued, loading or warming.
bool settled(const cb::s3gen_sidecars & set) {
    const std::string json = set.status_json();
    for (const char * busy : {"\"state\":\"queued\"", "\"state\":\"loading\"", "\"state\":\"warming\"",
                              "\"state\":\"deferred(memory)\""}) {
        if (json.find(busy) != std::string::npos) return false;
    }
    return true;
}

void settle(const cb::s3gen_sidecars & set, const std::string & what) {
    if (!wait_for([&] { return settled(set); }, 30000)) {
        std::printf("%s\n", set.status_json().c_str());
        throw std::runtime_error(what + ": the loads finish");
    }
}

// A directory holding links to the named routing-fixture sidecars under a
// fresh path (its own registry entries); returns the GGUF path the sidecars
// sit next to (the GGUF itself need not exist).
std::string routing_case(const fs::path & fixture, const fs::path & root, const std::string & name,
                         const std::vector<const char *> & suffixes) {
    const fs::path dir = root / name;
    fs::create_directories(dir);
    for (const char * suffix : suffixes) {
        fs::create_directory_symlink(fs::absolute(fixture / (std::string("routing-s3gen") + suffix + ".mlmodelc")),
                                     dir / (std::string("routing-s3gen") + suffix + ".mlmodelc"));
    }
    return (dir / "routing-s3gen.gguf").string();
}

// prewarm = predicted: predict() loads a call's widths ahead of time (no
// construction group is issued unless a case starts one).
CoremlOptions routing_options(const std::string & cache_dir = {},
                              CoremlOptions::Prewarm prewarm = CoremlOptions::Prewarm::predicted) {
    CoremlOptions o;
    o.compute_units = "cpu_only";
    o.wait = CoremlOptions::Wait::none;
    o.prewarm = prewarm;
    if (!cache_dir.empty()) o.cache_dir = cache_dir;
    return o;
}

cb::s3gen_call_shape shape(int tokens) {
    return cb::s3gen_call_shape{tokens, 2 * tokens};
}

cb::plan_stage_view view(const std::shared_ptr<S3genCoremlPlan> & plan, cb::s3gen_stage stage) {
    return cb::plan_view(*plan, stage);
}
#endif

int pending_fixture() {
#ifndef TTS_CPP_USE_COREML
    std::puts("SKIP: built without TTS_CPP_COREML");
    return SKIP;
#else
    const char * env = std::getenv("CHATTERBOX_COREML_TEST_SHAPES");
    if (!env || !*env) {
        std::puts("SKIP: CHATTERBOX_COREML_TEST_SHAPES is not set");
        return SKIP;
    }
    const fs::path fixture = env;
    for (const char * suffix : {cb::COREML_ENCODER_SUFFIX, cb::COREML_CFM_SUFFIX, cb::COREML_HIFT_SUFFIX}) {
        if (!fs::exists(fixture / (std::string("routing-s3gen") + suffix + ".mlmodelc"))) {
            std::printf("SKIP: the fixture has no routing-s3gen%s.mlmodelc (rerun "
                        "prepare_chatterbox_coreml_shape_fixture.py)\n", suffix);
            return SKIP;
        }
    }
    env_guard units(cb::COREML_UNITS_ENV), strict(cb::COREML_STRICT_ENV), disabled(cb::COREML_DISABLE_ENV);
    env_guard cache("QVAC_COREML_CACHE_DIR");
    unsetenv(cb::COREML_UNITS_ENV);
    unsetenv(cb::COREML_STRICT_ENV);
    unsetenv(cb::COREML_DISABLE_ENV);
    unsetenv("QVAC_COREML_CACHE_DIR");
    temp_dir root("pending");
    const std::vector<const char *> all = {cb::COREML_ENCODER_SUFFIX, cb::COREML_CFM_SUFFIX, cb::COREML_HIFT_SUFFIX};
    auto & hooks = cb::coreml_test();
    hooks.tracing = true;  // the cases below read the loads started
    using S = cb::s3gen_stage;

    // 1. A call whose functions are still loading runs every stage on ggml,
    //    marked pending, without blocking; a later call gets the models.
    {
        auto set = cb::s3gen_sidecars::create(routing_case(fixture, root.path, "pending", all), routing_options());
        require(set && set->buckets(S::cfm) == std::vector<int>{16, 24, 32, 40} &&
                    set->buckets(S::encoder) == std::vector<int>{8, 12, 16, 20} && set->has(S::hift),
                "the routing sidecars list their widths");
        hooks.load_delay_ms = 1000;
        const auto t0 = steady::now();
        auto plan = set->acquire(shape(8), steady::now() + std::chrono::milliseconds(10));
        const double took = ms_since(t0);
        for (S s : {S::encoder, S::cfm, S::hift}) {
            require(!view(plan, s).model && view(plan, s).pending && view(plan, s).label == cb::GGML_STAGE_BACKEND,
                    std::string("pending ") + cb::to_string(s) + " runs on ggml");
        }
        require(cb::plan_pending(*plan) && took < 200, "a pending call is marked and never blocks");
        hooks.load_delay_ms = 0;
        settle(*set, "pending");
        plan = set->acquire(shape(8), steady::now());
        require(view(plan, S::encoder).bucket == 8 && view(plan, S::cfm).bucket == 16 && view(plan, S::hift).model &&
                    view(plan, S::cfm).label == "coreml-cpu" && !cb::plan_pending(*plan),
                "the next call runs on the loaded functions");
        std::printf("[pending] acquired in %.1f ms, all three on ggml; the next call on Core ML\n", took);
    }

    // 2. A short sentence whose widths are ready, then a longer one whose
    //    widths are still loading: the first runs on Core ML, the second on
    //    ggml (a narrower ready function never serves a wider call).
    {
        auto set = cb::s3gen_sidecars::create(routing_case(fixture, root.path, "short-long", all), routing_options());
        set->predict({shape(8)});
        settle(*set, "short-long");
        hooks.load_delay_ms = 800;
        auto short_plan = set->acquire(shape(8), steady::now());
        auto long_plan = set->acquire(shape(12), steady::now());
        require(view(short_plan, S::cfm).bucket == 16 && view(short_plan, S::encoder).bucket == 8,
                "the short sentence runs on its ready widths");
        require(!view(long_plan, S::cfm).model && view(long_plan, S::cfm).pending && !view(long_plan, S::encoder).model,
                "the longer sentence runs on ggml while its widths load");
        require(view(long_plan, S::hift).model, "HiFT serves every length");
        hooks.load_delay_ms = 0;
        settle(*set, "short-long");
        std::printf("[short-long] ready widths served, the longer sentence on ggml\n");
    }

    // 3. The actual length exceeds the prediction: the predicted width is
    //    loaded, the call needs a wider one; that one is requested at once
    //    and the call runs on ggml meanwhile.
    {
        auto set = cb::s3gen_sidecars::create(routing_case(fixture, root.path, "longer", all), routing_options());
        set->predict({shape(8)});
        settle(*set, "longer");
        hooks.clear_loads();
        hooks.load_delay_ms = 500;
        auto plan = set->acquire(shape(16), steady::now());
        require(!view(plan, S::cfm).model && !view(plan, S::encoder).model, "the longer call runs on ggml");
        hooks.load_delay_ms = 0;
        settle(*set, "longer");
        const auto loads = hooks.loads();
        require(std::find(loads.begin(), loads.end(), "cfm/frames_32") != loads.end() &&
                    std::find(loads.begin(), loads.end(), "encoder/tokens_16") != loads.end(),
                "the actual widths are loaded");
        require(set->state(S::cfm, 32) == "ready", "and ready for the next call");
        std::printf("[longer] actual widths loaded after the call\n");
    }

    // 4. A CFM failure at a later Euler step (stand-in steps; the real
    //    pipeline is the engine fixture's).
    check_cfm_rollback();
    std::printf("[cfm-rollback] a later-step failure equals a whole ggml integration from z0\n");

    // 5. A failure during an overlapped call: the plan defers it (the worker
    //    hands the call back), the function is retired process-wide and
    //    dropped from the plan, and the replay runs on that same plan: the
    //    failed stage on ggml even though a wider covering width is ready,
    //    the other stages on their own models. Later calls use the wider one.
    {
        auto set = cb::s3gen_sidecars::create(routing_case(fixture, root.path, "retire", all), routing_options());
        set->predict({shape(8), shape(12)});
        settle(*set, "retire");
        require(set->state(S::cfm, 24) == "ready", "a wider covering CFM width is ready");
        std::shared_ptr<const S3genCoremlPlan> plan = set->acquire(shape(8), steady::now());
        require(cb::plan_view(*plan, S::cfm).bucket == 16, "the overlapped call holds frames_16");
        cb::defer_coreml_failures(*plan);
        require(cb::coreml_failures_deferred(*plan), "the worker's plan defers failures");
        hooks.fail_stage = (int) S::cfm;
        std::vector<float> x(80 * 16, 0.0f), spks(80, 0.0f), t_emb(1024, 0.0f);
        std::vector<std::vector<float>> out;
        auto velocity = [&] {
            return cb::plan_velocity(*plan, 0, 1, 16, {x.data()}, {x.data()}, {x.data()}, {spks.data()},
                                     {t_emb.data()}, out);
        };
        bool failed = false;
        try {
            velocity();
        } catch (const cb::coreml_stage_failure &) {
            failed = true;
        }
        require(failed && hooks.fail_stage == -1, "the injected prediction failure surfaces as a stage failure");
        cb::plan_retire(*plan, S::cfm, "injected");
        plan = cb::replay_plan(plan);
        require(!cb::coreml_failures_deferred(*plan), "the replay no longer defers failures");
        require(!cb::plan_view(*plan, S::cfm).model && cb::plan_view(*plan, S::cfm).label == cb::GGML_STAGE_BACKEND,
                "the replay runs CFM on ggml");
        require(!velocity(), "the replay's CFM step does not reach Core ML");
        require(cb::plan_view(*plan, S::encoder).bucket == 8 && cb::plan_view(*plan, S::hift).model,
                "the replay keeps the sentence's encoder and HiFT models");
        require(!s3gen_coreml_plan_offloads(*plan), "the replay never overlaps T3");
        require(set->state(S::cfm, 16) == "retired", "the failed function is retired");
        auto later = set->acquire(shape(8), steady::now());
        require(view(later, S::cfm).bucket == 24, "a later call runs CFM on the next ready width");
        std::printf("[retire] frames_16 retired; the replay runs CFM on ggml, a later call on frames_24\n");
    }

    // 5b. The only sidecar present cannot serve (a CFM package without
    //     frames_* functions): use = require refuses the call, otherwise the
    //     stage runs on ggml.
    {
        const fs::path dir = root.path / "lone-broken";
        fs::create_directories(dir);
        fs::create_directory_symlink(fs::absolute(fixture / "routing-s3gen-hift.mlmodelc"),
                                     dir / "routing-s3gen-cfm.mlmodelc");
        const std::string gguf = (dir / "routing-s3gen.gguf").string();
        CoremlOptions require_opts = routing_options();
        require_opts.use = CoremlOptions::Use::require;
        auto strict = cb::s3gen_sidecars::create(gguf, require_opts);
        require(strict != nullptr && !strict->has(S::cfm), "a lone unusable CFM sidecar still makes a set");
        std::string error;
        try {
            strict->acquire(shape(8), steady::now());
        } catch (const std::exception & e) {
            error = e.what();
        }
        require(error.find("cannot serve") != std::string::npos, "use = require refuses a lone unusable sidecar");
        auto lenient = cb::s3gen_sidecars::create(gguf, routing_options());
        // Preparation never reports success for it.
        const std::string prepared = lenient->prewarm(CoremlOptions::Prewarm::all, true);
        lenient->start_construction();
        const std::string constructed = lenient->construction_json();
        for (const std::string & json : {prepared, constructed}) {
            require(json.find("\"success\":false") != std::string::npos &&
                        json.find("\"nothing_to_prepare\":false") != std::string::npos &&
                        json.find("\"stage\":\"cfm\"") != std::string::npos &&
                        json.find("\"state\":\"failed\"") != std::string::npos,
                    "a lone unusable CFM sidecar fails preparation with its diagnostic: " + json);
        }
        require(!lenient->wait(0) && lenient->status_json().find("\"construction\":") != std::string::npos,
                "waiting fails, and the status carries the construction result");
        auto plan = lenient->acquire(shape(8), steady::now());
        require(view(plan, S::cfm).present && !view(plan, S::cfm).model && !cb::plan_pending(*plan),
                "without require the unusable stage runs on ggml");
        std::printf("[lone-broken] require: %s\n", error.c_str());
    }

    // 6. Dropping the sidecars while a load is in flight returns at once.
    {
        auto set = cb::s3gen_sidecars::create(routing_case(fixture, root.path, "destroy", all), routing_options());
        hooks.clear_loads();
        hooks.load_delay_ms = 1500;
        set->predict({shape(8)});
        require(wait_for([&] { return !hooks.loads().empty(); }, 5000), "a load is in flight");
        const auto t0 = steady::now();
        set.reset();
        const double took = ms_since(t0);
        std::printf("[destroy] dropped mid-load in %.2f ms\n", took);
        require(took < 100.0, "destruction mid-load returns in under 100 ms");
        hooks.load_delay_ms = 0;
        std::this_thread::sleep_for(std::chrono::milliseconds(1700));  // let the started load end
    }

    // 7. The exact width is still loading and a wider one is ready: the wider
    //    one serves the call (padding masked).
    {
        auto set = cb::s3gen_sidecars::create(routing_case(fixture, root.path, "wider", all), routing_options());
        set->predict({shape(20)});
        settle(*set, "wider");
        hooks.load_delay_ms = 800;
        auto plan = set->acquire(shape(12), steady::now());
        require(view(plan, S::cfm).bucket == 40 && view(plan, S::encoder).bucket == 20 && !cb::plan_pending(*plan),
                "a ready wider function serves the call");
        hooks.load_delay_ms = 0;
        settle(*set, "wider");
        require(set->state(S::cfm, 24) == "ready", "the exact width loaded meanwhile");
        std::printf("[wider] frames_40 served a 24-frame call\n");
    }

    // 8. The hinted widths are loading at construction and the first call
    //    needs another one: it is the next load, right after the one in
    //    flight.
    {
        const std::string cache_dir = (root.path / "order-cache").string();
        const std::string gguf = routing_case(fixture, root.path, "order", {cb::COREML_CFM_SUFFIX});
        {
            auto set = cb::s3gen_sidecars::create(gguf, routing_options(cache_dir));
            set->predict({shape(8), shape(12), shape(16)});
            settle(*set, "order (hints)");
        }
        hooks.clear_loads();
        hooks.load_delay_ms = 400;
        auto set = cb::s3gen_sidecars::create(gguf, routing_options(cache_dir, CoremlOptions::Prewarm::minimal));
        set->start_construction();
        require(wait_for([&] { return hooks.loads().size() == 1; }, 5000), "a construction load is in flight");
        auto plan = set->acquire(shape(20), steady::now());
        require(!view(plan, S::cfm).model, "the call runs on ggml");
        hooks.load_delay_ms = 0;
        settle(*set, "order");
        const auto loads = hooks.loads();
        std::string order;
        for (const auto & l : loads) order += (order.empty() ? "" : " ") + l;
        std::printf("[order] %s\n", order.c_str());
        const std::set<std::string> hinted = {"cfm/frames_16", "cfm/frames_24", "cfm/frames_32"};
        require(loads.size() == 4 && hinted.count(loads[0]) == 1 && loads[1] == "cfm/frames_40",
                "the call's width loads second, right after the one in flight");
        require(std::set<std::string>(loads.begin() + 2, loads.end()).size() == 2, "then the other hinted widths");
    }

    // 9. A stale hint: the slow load is waited for only until the deadline,
    //    the hint cools down, and the next call does not wait at all.
    {
        const std::string cache_dir = (root.path / "stale-cache").string();
        const std::string gguf = routing_case(fixture, root.path, "stale", {cb::COREML_CFM_SUFFIX});
        // Two processes' worth of loads: the first records a cold load, the
        // second (hinted) an observed warm load time.
        for (int round = 0; round < 2; ++round) {
            auto set = cb::s3gen_sidecars::create(gguf, routing_options(cache_dir));
            set->predict({shape(8)});
            settle(*set, "stale (hint)");
        }
        hooks.load_delay_ms = 2000;
        auto set = cb::s3gen_sidecars::create(gguf, routing_options(cache_dir));
        auto t0 = steady::now();
        auto plan = set->acquire(shape(8), t0 + std::chrono::milliseconds(300));
        const double first = ms_since(t0);
        t0 = steady::now();
        auto again = set->acquire(shape(8), t0 + std::chrono::milliseconds(300));
        const double second = ms_since(t0);
        std::printf("[stale] first call waited %.0f ms (deadline 300), the next %.0f ms\n", first, second);
        require(!view(plan, S::cfm).model && first >= 250 && first < 450,
                "the warm-hinted slow load is waited for until the deadline, not beyond");
        require(!view(again, S::cfm).model && second < 100, "the stale hint gives the next call no wait");
        hooks.load_delay_ms = 0;
        settle(*set, "stale");
    }

    // 10. The resident cap never releases a held or a predicted width.
    {
        const std::string gguf = routing_case(fixture, root.path, "cap", {cb::COREML_CFM_SUFFIX});
        auto set = cb::s3gen_sidecars::create(gguf, routing_options());
        set->predict({shape(8)});
        settle(*set, "cap");
        auto held = set->acquire(shape(8), steady::now());
        require(view(held, S::cfm).bucket == 16, "a call holds frames_16");
        // Five calls predicted at frames_24: each acquisition below drops
        // one, so frames_24 stays predicted for the one still ahead.
        set->predict({shape(12), shape(12), shape(12), shape(12), shape(12)});
        settle(*set, "cap");
        size_t most = 0;
        for (int tokens : {16, 20, 16, 20}) {
            set->acquire(shape(tokens), steady::now());
            require(wait_for([&] {
                        most = std::max(most, set->live(S::cfm));
                        return settled(*set);
                    }, 30000),
                    "cap: the loads finish");
            require(set->state(S::cfm, 2 * tokens) == "ready", "the requested width loads");
            require(set->state(S::cfm, 16) == "ready" && set->state(S::cfm, 24) == "ready",
                    "the held and the predicted widths stay loaded");
        }
        std::printf("[cap] at most %zu CFM functions live; frames_16 (held) and frames_24 (predicted) kept\n", most);
        require(most <= cb::COREML_RESIDENT_CAP && set->live(S::cfm) <= cb::COREML_RESIDENT_CAP,
                "never more live functions than the cap");
        set->end_call();
    }

    // 12. prewarm none and minimal never load a call's predicted widths ahead
    //     of time; the call's own demand still loads what it needs.
    for (const auto scope : {CoremlOptions::Prewarm::none, CoremlOptions::Prewarm::minimal}) {
        const char * name = scope == CoremlOptions::Prewarm::none ? "none" : "minimal";
        auto set = cb::s3gen_sidecars::create(routing_case(fixture, root.path, std::string("scope-") + name, all),
                                              routing_options({}, scope));
        set->start_construction();
        settle(*set, std::string("scope ") + name);
        hooks.clear_loads();
        set->predict({shape(8), shape(16)});
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        settle(*set, std::string("scope ") + name);
        const auto speculative = hooks.loads();
        require(speculative.empty(), std::string("prewarm ") + name + " queues no predicted width");
        set->acquire(shape(8), steady::now());
        settle(*set, std::string("scope ") + name);
        require(set->state(S::cfm, 16) == "ready" && set->state(S::encoder, 8) == "ready" &&
                    set->state(S::cfm, 32).empty() == false && set->state(S::cfm, 32) != "ready",
                std::string("prewarm ") + name + ": the demanded widths load, the merely predicted one does not");
        set->end_call();
        std::printf("[scope %s] no speculative loads; the demanded widths loaded\n", name);
    }

    // 13. A full resident pool: CFM frames_16 and frames_24 are held by
    //     running calls and frames_40 is ready; a call of 32 frames takes
    //     frames_40 at once, and requesting its exact frames_32 neither
    //     evicts frames_40 nor makes the call wait, under wait = call and
    //     under use = require.
    for (const bool strict_mode : {false, true}) {
        const char * name = strict_mode ? "full-pool-require" : "full-pool-call";
        CoremlOptions full_opts = routing_options();
        if (strict_mode) {
            full_opts.use = CoremlOptions::Use::require;
        } else {
            full_opts.wait = CoremlOptions::Wait::call;
        }
        auto set = cb::s3gen_sidecars::create(routing_case(fixture, root.path, name, all), full_opts);
        set->predict({shape(8), shape(12), shape(20)});
        settle(*set, name);
        set->end_call();  // nothing is predicted any more: frames_40 is evictable
        auto held16 = set->acquire(shape(8), steady::now());
        auto held24 = set->acquire(shape(12), steady::now());
        require(view(held16, S::cfm).bucket == 16 && view(held24, S::cfm).bucket == 24 &&
                    set->state(S::cfm, 40) == "ready" && set->live(S::cfm) == cb::COREML_RESIDENT_CAP,
                std::string(name) + ": frames_16 and frames_24 held, frames_40 ready, the pool full");
        hooks.load_delay_ms = 3000;
        const auto t0 = steady::now();
        auto plan = set->acquire(shape(16), steady::now() + std::chrono::milliseconds(10));
        const double took = ms_since(t0);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        const std::string state40 = set->state(S::cfm, 40);
        std::printf("[%s] frames_%d in %.0f ms; frames_40 %s, frames_32 %s\n", name, view(plan, S::cfm).bucket,
                    took, state40.c_str(), set->state(S::cfm, 32).c_str());
        require(view(plan, S::cfm).bucket == 40 && took < 1000,
                std::string(name) + ": the ready frames_40 serves at once");
        require(state40 == "ready", std::string(name) + ": requesting frames_32 does not evict the held frames_40");
        plan.reset();
        held16.reset();
        held24.reset();
        hooks.load_delay_ms = 0;
        settle(*set, name);
    }

    // 11. A ready function covers the call: it serves at once. wait = call
    //     does not wait through the exact function's slow load, and
    //     use = require does not refuse a call whose exact function retired.
    {
        CoremlOptions call_opts = routing_options();
        call_opts.wait = CoremlOptions::Wait::call;
        auto set = cb::s3gen_sidecars::create(routing_case(fixture, root.path, "covering-call", all), call_opts);
        set->predict({shape(16)});
        settle(*set, "covering-call");
        hooks.load_delay_ms = 3000;
        const auto t0 = steady::now();
        auto plan = set->acquire(shape(8), steady::now() + std::chrono::milliseconds(10));
        const double took = ms_since(t0);
        hooks.load_delay_ms = 0;
        std::printf("[covering] wait = call: frames_%d and tokens_%d in %.0f ms\n", view(plan, S::cfm).bucket,
                    view(plan, S::encoder).bucket, took);
        require(view(plan, S::cfm).bucket == 32 && view(plan, S::encoder).bucket == 16 && took < 1000,
                "wait = call takes a ready covering function instead of waiting for the exact one");
        settle(*set, "covering-call");

        CoremlOptions require_opts = routing_options();
        require_opts.use = CoremlOptions::Use::require;
        auto strict = cb::s3gen_sidecars::create(routing_case(fixture, root.path, "covering-require", all),
                                                 require_opts);
        strict->predict({shape(8), shape(12)});
        settle(*strict, "covering-require");
        auto held = strict->acquire(shape(8), steady::now());
        require(view(held, S::cfm).bucket == 16, "use = require holds frames_16");
        cb::plan_retire(*held, S::cfm, "injected");
        std::shared_ptr<S3genCoremlPlan> again;
        std::string error;
        try {
            again = strict->acquire(shape(8), steady::now());
        } catch (const std::exception & e) {
            error = e.what();
        }
        std::printf("[covering] use = require after retirement: %s\n",
                    again ? ("frames_" + std::to_string(view(again, S::cfm).bucket)).c_str() : error.c_str());
        require(again && view(again, S::cfm).bucket == 24,
                "use = require serves a call whose exact function retired from a ready covering one");
    }

    // 14. Four sentences of ascending widths under use = require: one width
    //     per stage more than the resident cap. Each sentence acquires after
    //     its T3 and releases its models once rendered, the next one
    //     acquiring while the previous one still renders (the overlap). Only
    //     the sentences still ahead stay predicted, so the fourth width
    //     replaces a finished sentence's width instead of failing the call.
    {
        CoremlOptions require_opts = routing_options();
        require_opts.use = CoremlOptions::Use::require;
        require_opts.strict_wait_ms = 60000;  // a load that never gets a place fails, never hangs
        auto set = cb::s3gen_sidecars::create(routing_case(fixture, root.path, "ascending-require", all),
                                              require_opts);
        const std::vector<int> sentences = {8, 12, 16, 20};
        std::vector<cb::s3gen_call_shape> calls;
        for (int tokens : sentences) calls.push_back(shape(tokens));
        set->predict(calls);
        std::shared_ptr<S3genCoremlPlan> rendering;
        size_t most = 0;
        std::string served;
        for (int tokens : sentences) {
            std::shared_ptr<S3genCoremlPlan> plan;
            std::string error;
            try {
                plan = set->acquire(shape(tokens), steady::now() + std::chrono::seconds(60));
            } catch (const std::exception & e) {
                error = e.what();
            }
            most = std::max({most, set->live(S::cfm), set->live(S::encoder)});
            if (!plan) std::printf("[ascending require] sentence of %d tokens: %s\n", tokens, error.c_str());
            require(plan != nullptr, "use = require serves every sentence of four ascending widths");
            require(view(plan, S::encoder).bucket == tokens && view(plan, S::cfm).bucket == 2 * tokens &&
                        view(plan, S::hift).model,
                    "each sentence runs on its exact widths");
            served += (served.empty() ? "" : ", ") + std::to_string(view(plan, S::encoder).bucket) + "/" +
                      std::to_string(view(plan, S::cfm).bucket);
            rendering = std::move(plan);  // the previous sentence finished rendering
        }
        rendering.reset();
        set->end_call();
        settle(*set, "ascending-require");
        most = std::max({most, set->live(S::cfm), set->live(S::encoder)});
        std::printf("[ascending require] encoder/CFM widths %s; at most %zu live per stage\n", served.c_str(), most);
        require(most <= cb::COREML_RESIDENT_CAP, "ascending widths: never more live functions than the cap");
    }
    return 0;
#endif
}

// --------------------------------------------------------- engine fixture --

// Three segments at 60 bytes; with the CFM and HiFT sidecars present the
// Engine renders each segment's S3Gen on a worker while T3 decodes the next.
const char * OVERLAP_TEXT = "The quick brown fox jumps over the lazy dog. "
                            "Core ML renders this sentence while the next one decodes. "
                            "Every segment must come back in order.";
constexpr int OVERLAP_SENTENCE_CHARS = 60;

EngineOptions engine_options(const std::string & t3, const std::string & s3gen) {
    EngineOptions opts;
    opts.t3_gguf_path = t3;
    opts.s3gen_gguf_path = s3gen;
    opts.n_gpu_layers = 99;
    opts.seed = 42;
    opts.max_sentence_chars = OVERLAP_SENTENCE_CHARS;
    return opts;
}

bool same_pcm(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && !a.empty() && max_abs(a, b) == 0.0;
}

#ifdef TTS_CPP_USE_COREML
// The Engine's Core ML functions are all loaded (none queued or loading).
bool engine_settled(const Engine & engine) {
    const std::string json = engine.coreml_status();
    for (const char * busy : {"\"state\":\"queued\"", "\"state\":\"loading\"", "\"state\":\"warming\"",
                              "\"state\":\"deferred(memory)\""}) {
        if (json.find(busy) != std::string::npos) return false;
    }
    return true;
}

std::string read_file(const fs::path & path) {
    std::ifstream in(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

// Runs tts-cli on the overlap text; returns its exit code.
int run_cli(const std::string & cli, const std::string & t3, const std::string & s3gen, const fs::path & wav,
            const fs::path & log, const std::string & extra) {
    const std::string cmd = "'" + cli + "' --model '" + t3 + "' --s3gen-gguf '" + s3gen + "' --text '" +
                            OVERLAP_TEXT + "' --out '" + wav.string() + "' --seed 42 --n-gpu-layers 99 " +
                            "--max-sentence-chars " + std::to_string(OVERLAP_SENTENCE_CHARS) + " --verbose " + extra +
                            " 2> '" + log.string() + "'";
    const int status = std::system(cmd.c_str());
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

// Stage labels tts-cli reported ("[encoder] 12.3 ms (coreml-ane)", ...).
std::set<std::string> cli_labels(const std::string & log) {
    std::set<std::string> labels;
    std::istringstream lines(log);
    std::string line;
    while (std::getline(lines, line)) {
        const bool stage = line.find("[encoder]") != std::string::npos || line.find("[cfm_step") != std::string::npos ||
                           line.find("[hift_decode]") != std::string::npos;
        const size_t open = line.rfind('('), close = line.rfind(')');
        if (stage && open != std::string::npos && close != std::string::npos && close > open) {
            labels.insert(line.substr(open + 1, close - open - 1));
        }
    }
    return labels;
}

// While every function is still loading (held, or delayed past the whole
// run for tts-cli: a slow, cold load), the first
// synthesis runs entirely on ggml and equals a use = off run with the same
// seed, sample for sample; once loaded, later calls run on Core ML. Through
// the Engine and through tts-cli.
void slow_load_checks(const std::string & t3, const std::string & s3gen, const std::string & cli,
                      const fs::path & root) {
    auto & hooks = cb::coreml_test();
    EngineOptions off_opts = engine_options(t3, s3gen);
    off_opts.coreml.use = CoremlOptions::Use::off;
    Engine off_engine(off_opts);
    const SynthesisResult reference = off_engine.synthesize(OVERLAP_TEXT);
    require(reference.s3gen_backend == cb::GGML_STAGE_BACKEND, "use = off runs on ggml");
    // Nothing to prepare is a success, not a failure.
    const std::string off_group = off_engine.coreml_prewarm(CoremlOptions::Prewarm::all, true);
    require(off_engine.coreml_wait(0) && off_group.find("\"nothing_to_prepare\":true") != std::string::npos &&
                off_group.find("\"success\":true") != std::string::npos,
            "use = off: waiting and prewarming succeed with nothing to prepare");
    const std::string off_construction = off_engine.coreml_construction();
    require(off_construction.find("\"nothing_to_prepare\":true") != std::string::npos &&
                off_construction.find("\"success\":true") != std::string::npos &&
                off_engine.coreml_status().find("\"construction\":") != std::string::npos,
            "use = off: the construction result is empty, successful and in the status");

    hooks.hold_loads = true;
    EngineOptions slow_opts = engine_options(t3, s3gen);
    slow_opts.coreml.cache_dir = (root / "engine-cache").string();
    Engine engine(slow_opts);
    const SynthesisResult first = engine.synthesize(OVERLAP_TEXT);
    std::printf("[slow-load engine] first call: backend=%s pending=%d, max |first - use=off| = %.3e\n",
                first.s3gen_backend.c_str(), (int) first.coreml_pending,
                first.pcm.size() == reference.pcm.size() ? max_abs(first.pcm, reference.pcm) : -1.0);
    require(first.s3gen_backend == cb::GGML_STAGE_BACKEND && first.coreml_pending,
            "the first call runs on ggml while the functions load");
    require(same_pcm(first.pcm, reference.pcm), "the first call equals a use = off run with the same seed");
    hooks.hold_loads = false;
    require(wait_for([&] { return engine_settled(engine); }, 900000), "the requested functions load");
    const SynthesisResult later = engine.synthesize(OVERLAP_TEXT);
    std::printf("[slow-load engine] later call: backend=%s pending=%d\n", later.s3gen_backend.c_str(),
                (int) later.coreml_pending);
    require(later.s3gen_backend == "coreml-ane" && !later.coreml_pending, "later calls run on Core ML");

    // tts-cli runs its own sentence loop.
    const fs::path off_wav = root / "cli-off.wav", slow_wav = root / "cli-slow.wav", warm_wav = root / "cli-warm.wav";
    const std::string cache = " --coreml-cache-dir '" + (root / "cli-cache").string() + "'";
    require(run_cli(cli, t3, s3gen, off_wav, root / "cli-off.log", "--coreml-use off") == 0, "tts-cli use = off");
    require(run_cli(cli, t3, s3gen, slow_wav, root / "cli-slow.log", "--coreml-slow-load-ms 120000" + cache) == 0,
            "tts-cli with slow loads");
    const std::set<std::string> slow_labels = cli_labels(read_file(root / "cli-slow.log"));
    require(slow_labels == std::set<std::string>{cb::GGML_STAGE_BACKEND}, "tts-cli's first call runs on ggml");
    require(!read_file(off_wav).empty() && read_file(off_wav) == read_file(slow_wav),
            "tts-cli's first call equals its use = off run");
    const fs::path status = root / "cli-warm-status.json";
    require(run_cli(cli, t3, s3gen, warm_wav, root / "cli-warm.log",
                    "--coreml-wait call --coreml-status-json '" + status.string() + "'" + cache) == 0,
            "tts-cli once loaded");
    const std::string status_text = read_file(status);
    require(status_text.find("\"construction\":{") != std::string::npos &&
                status_text.find("\"group\":\"construction\"") != std::string::npos,
            "tts-cli's status JSON carries the construction result");
    const std::set<std::string> warm_labels = cli_labels(read_file(root / "cli-warm.log"));
    std::printf("[slow-load cli] first call ggml and equal to use = off; loaded call labels:");
    for (const auto & l : warm_labels) std::printf(" %s", l.c_str());
    std::printf("\n");
    require(warm_labels == std::set<std::string>{"coreml-ane"}, "tts-cli runs every stage on Core ML once loaded");
}

// Fit on the real GGUFs: with no sidecar a projection succeeds; with the
// sidecars present, use = off and CHATTERBOX_COREML_DISABLE give exactly that
// projection, and the enabled verdict (the prebuilt route refuses) depends on
// presence and options only: `loaded` false before any function loaded, true
// after the Engine loaded them.
tts_cpp::FitResult fit_check(const std::string & t3, const std::string & s3gen, const fs::path & root, bool loaded,
                             const tts_cpp::FitResult * before) {
    tts_cpp::chatterbox::FitOptions options;
    options.t3_gguf_path = t3;
    options.s3gen_gguf_path = (stage_dir(root, s3gen, {}, loaded ? "fit-plain-loaded" : "fit-plain") /
                               fs::path(s3gen).filename()).string();
    options.n_gpu_layers = 99;
    const auto plain = tts_cpp::chatterbox::fit_params(options);
    require(projected(plain), "fit: a projection without sidecars succeeds (" + plain.reason + ")");
    options.s3gen_gguf_path = s3gen;
    auto off = options;
    off.coreml.use = CoremlOptions::Use::off;
    const auto off_fit = tts_cpp::chatterbox::fit_params(off);
    require(projected(off_fit) && same_projection(off_fit, plain), "fit: use = off projects exactly the plain fit");
    setenv(cb::COREML_DISABLE_ENV, "1", 1);
    const auto killed = tts_cpp::chatterbox::fit_params(options);
    unsetenv(cb::COREML_DISABLE_ENV);
    require(projected(killed) && same_projection(killed, plain),
            "fit: CHATTERBOX_COREML_DISABLE projects exactly the plain fit");
    const auto enabled = tts_cpp::chatterbox::fit_params(options);
    require(enabled.reason == "coreml-memory-estimate-unavailable", "fit: the prebuilt route refuses");
    if (before) require(same_fit(enabled, *before), "fit: the enabled verdict is the same after the loads");
    std::printf("[fit %s] plain %s, %.1f MiB on %s; use = off identical; enabled: %s\n",
                loaded ? "after loads" : "before loads", plain.reason.c_str(),
                (double) plain.device.total_bytes / (1024.0 * 1024.0), plain.device_name.c_str(),
                enabled.reason.c_str());
    return enabled;
}

// The status path differs from --out only in case (a case-insensitive volume)
// and neither exists when the arguments are checked: the run writes the WAV,
// and the exit writer must not replace it with the status JSON.
void cli_status_case_check(const std::string & t3, const std::string & s3gen, const std::string & cli,
                           const fs::path & root) {
    const fs::path upper = root / "Cased.wav", lower = root / "cased.wav";
    { std::ofstream(root / "probe-A") << "x"; }
    std::error_code ec;
    const bool insensitive = fs::exists(root / "probe-a", ec);
    fs::remove(root / "probe-A", ec);
    if (!insensitive) {
        std::puts("[cli status case] case-sensitive volume: does not apply");
        return;
    }
    const fs::path log = root / "cli-status-case.log";
    const int rc = run_cli(cli, t3, s3gen, upper, log, "--coreml-use off --coreml-status-json '" + lower.string() + "'");
    const std::string wav = read_file(upper), err = read_file(log);
    std::printf("[cli status case] exit %d, %s starts with %s\n", rc, upper.filename().c_str(),
                wav.substr(0, 4).c_str());
    require(rc == 0 && wav.rfind("RIFF", 0) == 0 && err.find("is the --out file") != std::string::npos,
            "tts-cli never writes the status JSON over its WAV under a case-differing spelling");
}

// tts-cli's sentence loop with a CFM prediction failure on an overlapped
// sentence: the call hands the sentence back instead of starting ggml while
// T3 of the next sentence may hold the GPU; the CLI lets that T3 finish, then
// renders the sentence again on its own plan, CFM on ggml, the encoder and
// HiFT on Core ML. Nothing falls back in place.
void cli_replay_check(const std::string & t3, const std::string & s3gen, const std::string & cli,
                      const fs::path & root) {
    const fs::path wav = root / "cli-replay.wav", log_path = root / "cli-replay.log";
    require(run_cli(cli, t3, s3gen, wav, log_path, "--coreml-wait call --coreml-fail-stage cfm") == 0,
            "tts-cli with a failing CFM prediction");
    const std::string log = read_file(log_path);
    const size_t handed_back = log.find("rendering the call again");
    // T3 of segment 2 reports its tokens once it finished.
    size_t next_t3 = std::string::npos;
    for (size_t at = log.find("[t3 segment 2/3] "); at != std::string::npos;
         at = log.find("[t3 segment 2/3] ", at + 1)) {
        const size_t eol = log.find('\n', at);
        if (log.substr(at, eol - at).find("speech tokens") != std::string::npos) {
            next_t3 = at;
            break;
        }
    }
    const std::string marker = "--- segment 1/3: rendering again after T3 of segment 2 ---";
    const size_t replay = log.find(marker);
    std::printf("[cli replay] handed back at %zu, T3 of segment 2 done at %zu, replay at %zu\n", handed_back, next_t3,
                replay);
    require(handed_back != std::string::npos && log.find("ggml takes over") == std::string::npos,
            "tts-cli: the overlapped call hands the sentence back, nothing falls back in place");
    require(replay != std::string::npos && next_t3 != std::string::npos && handed_back < replay && next_t3 < replay,
            "tts-cli: the sentence is rendered again after the next sentence's T3 finished");
    const size_t replay_end = log.find("\n--- segment 2/3", replay);
    std::set<std::string> cfm, others;
    std::istringstream lines(log.substr(replay, replay_end == std::string::npos ? std::string::npos
                                                                                  : replay_end - replay));
    std::string line;
    while (std::getline(lines, line)) {
        const size_t open = line.rfind('('), close = line.rfind(')');
        if (open == std::string::npos || close == std::string::npos || close < open) continue;
        const std::string label = line.substr(open + 1, close - open - 1);
        if (line.find("[cfm_step") != std::string::npos) cfm.insert(label);
        if (line.find("[encoder]") != std::string::npos || line.find("[hift_decode]") != std::string::npos) {
            others.insert(label);
        }
    }
    require(cfm == std::set<std::string>{cb::GGML_STAGE_BACKEND}, "tts-cli: the replay runs CFM on ggml");
    require(others == std::set<std::string>{"coreml-ane"}, "tts-cli: the replay keeps the encoder and HiFT models");
}

// Two live Engines on the same GGUF and options own separate slots: one
// Engine's prewarm group is its own, and destroying it while its loads are
// queued cancels them (nothing loads for it afterwards), while the other
// Engine's construction group still completes.
void two_engines_check(const std::string & t3, const std::string & s3gen) {
    auto & hooks = cb::coreml_test();
    hooks.tracing = true;
    hooks.hold_loads = true;
    hooks.clear_loads();
    Engine first(engine_options(t3, s3gen));
    std::vector<std::string> before;
    {
        Engine second(engine_options(t3, s3gen));
        second.coreml_prewarm(CoremlOptions::Prewarm::all, false);
        require(wait_for([&] { return !hooks.loads().empty(); }, 5000), "two engines: a load is in flight");
        before = hooks.loads();
    }
    hooks.hold_loads = false;
    require(first.coreml_wait(120000), "two engines: the remaining Engine's construction group completes");
    require(wait_for([&] { return engine_settled(first); }, 120000), "two engines: the remaining loads finish");
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    const auto after = hooks.loads();
    size_t widths = 0;
    for (const auto & l : after) widths += l.rfind("hift/", 0) != 0;
    size_t widths_before = 0;
    for (const auto & l : before) widths_before += l.rfind("hift/", 0) != 0;
    std::printf("[two engines] loads started before the second Engine was dropped: %zu, after: %zu (%zu widths)\n",
                before.size(), after.size(), widths);
    require(widths == widths_before, "two engines: the dropped Engine's queued prewarm loads were cancelled");
    hooks.tracing = false;
}

// tts-cli run in-process (embedded) ends its call: the widths it predicted on
// the shared set are no longer protected once it returns, so a later direct
// S3Gen call in the process is not cap-blocked by them.
void cli_in_process_check(const std::string & t3, const std::string & s3gen, const fs::path & root) {
    const std::string cache = (root / "cli-in-process-cache").string(), wav = (root / "cli-in-process.wav").string();
    std::vector<std::string> args = {"tts-cli", "--model", t3, "--s3gen-gguf", s3gen, "--text", OVERLAP_TEXT,
                                     "--out", wav, "--seed", "42", "--n-gpu-layers", "99", "--max-sentence-chars",
                                     std::to_string(OVERLAP_SENTENCE_CHARS), "--coreml-cache-dir", cache};
    std::vector<char *> argv;
    for (auto & a : args) argv.push_back(a.data());
    argv.push_back(nullptr);
    require(tts_cpp_cli_main((int) args.size(), argv.data()) == 0, "tts-cli in-process");
    CoremlOptions cli_options;
    cli_options.cache_dir = cache;
    auto shared = cb::s3gen_sidecars::get(s3gen, cli_options);
    // The CLI's own set (a fresh one would have nothing requested yet).
    require(shared && shared->state(cb::s3gen_stage::hift, 0) != "idle", "the shared set is the one tts-cli used");
    std::printf("[cli in-process] shared set %s, predictions %s after tts-cli returned\n",
                shared ? "found" : "missing", shared && shared->predicting() ? "still marked" : "cleared");
    require(shared && !shared->predicting(), "tts-cli clears its predictions when it returns");
}

// An Engine constructed while CHATTERBOX_COREML_DISABLE is set stays on ggml
// for its whole life, even once the variable is cleared: its calls carry the
// Engine's own (empty) plan and never acquire Core ML models through the
// shared table, so no load starts.
void disabled_engine_check(const std::string & t3, const std::string & s3gen) {
    auto & hooks = cb::coreml_test();
    setenv(cb::COREML_DISABLE_ENV, "1", 1);
    Engine engine(engine_options(t3, s3gen));
    unsetenv(cb::COREML_DISABLE_ENV);
    hooks.clear_loads();
    hooks.tracing = true;
    const SynthesisResult result = engine.synthesize("Core ML stays off for this engine.");
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    hooks.tracing = false;
    const auto loads = hooks.loads();
    std::printf("[disabled engine] backend=%s, loads started after the variable was cleared: %zu\n",
                result.s3gen_backend.c_str(), loads.size());
    require(loads.empty() && result.s3gen_backend == cb::GGML_STAGE_BACKEND && !result.coreml_pending,
            "an Engine constructed with Core ML disabled never loads a sidecar function");
}

// The real CFM latch: a Core ML failure at the second Euler step restores the
// initial noise and integrates on ggml, so the call equals use = off sample
// for sample (only the CFM sidecar is present, so the encoder and HiFT run
// on ggml in both).
void cfm_rollback_check(const std::string & s3gen, const fs::path & root) {
    const fs::path dir = stage_dir(root, s3gen, {cb::COREML_CFM_SUFFIX}, "cfm-rollback");
    const std::string gguf = (dir / fs::path(s3gen).filename()).string();
    const std::vector<int32_t> tokens = fixture_tokens(90);
    auto run = [&](const CoremlOptions & coreml) {
        s3gen_synthesize_opts opts;
        opts.s3gen_gguf_path = gguf;
        opts.n_gpu_layers = 99;
        opts.seed = 42;
        opts.coreml = coreml;
        run_result r;
        opts.pcm_out = &r.wav;
        opts.s3gen_backend_out = &r.backend;
        require(s3gen_synthesize_to_wav(tokens, opts) == 0, "s3gen_synthesize_to_wav");
        return r;
    };
    CoremlOptions off;
    off.use = CoremlOptions::Use::off;
    const run_result reference = run(off);
    CoremlOptions waited;
    waited.wait = CoremlOptions::Wait::call;  // the CFM function is loaded before the call
    waited.compute_units = "cpu_only";
    const run_result served = run(waited);
    require(served.backend == cb::MIXED_STAGE_BACKEND, "CFM runs on Core ML");
    cb::coreml_test().fail_cfm_step = 1;
    cb::coreml_test().clear_events();
    cb::coreml_test().tracing = true;
    const run_result rolled = run(waited);
    cb::coreml_test().tracing = false;
    require(cb::coreml_test().fail_cfm_step == -1, "the failure was injected");
    // No random number is drawn after the initial noise: not by any Euler
    // step, nor by the rollback.
    std::string draws;
    for (const auto & e : cb::coreml_test().events()) {
        if (e.rfind("cfm draws ", 0) == 0) draws = e;
    }
    const size_t init_at = draws.find("init="), end_at = draws.find(" end=");
    require(init_at != std::string::npos && end_at != std::string::npos, "the CFM noise draws are traced");
    const std::string init = draws.substr(init_at + 5, end_at - init_at - 5), end = draws.substr(end_at + 5);
    std::printf("[cfm-rollback] %s\n", draws.c_str());
    require(std::stoull(init) > 0 && init == end, "no CFM noise draw after initialisation, rollback included");
    std::printf("[cfm-rollback] backend=%s, max |rolled back - use=off| = %.3e\n", rolled.backend.c_str(),
                rolled.wav.size() == reference.wav.size() ? max_abs(rolled.wav, reference.wav) : -1.0);
    require(rolled.backend == cb::GGML_STAGE_BACKEND, "after the failure the whole call ran on ggml");
    require(same_pcm(rolled.wav, reference.wav), "a CFM failure at step 2 equals a whole ggml integration from z0");
}

// A CFM failure on a call whose failures are deferred (an overlapped
// sentence): the call returns S3GEN_RC_COREML_REDO with nothing rendered, and
// its replay on the same plan integrates CFM on ggml from the call's own
// initial noise, equal to use = off sample for sample, although a wider
// covering CFM width is ready (a fresh plan would take that one). Only the
// CFM sidecar is present, so the encoder and HiFT run on ggml in both.
void cfm_replay_check(const std::string & s3gen, const fs::path & root) {
    const fs::path dir = stage_dir(root, s3gen, {cb::COREML_CFM_SUFFIX}, "cfm-replay");
    const std::string gguf = (dir / fs::path(s3gen).filename()).string();
    CoremlOptions waited;
    waited.wait = CoremlOptions::Wait::call;
    waited.compute_units = "cpu_only";
    waited.prewarm = CoremlOptions::Prewarm::predicted;  // predict() below loads the two widths
    auto sidecars = cb::s3gen_sidecars::get(gguf, waited);
    require(sidecars != nullptr, "replay: the CFM sidecar is found");
    const std::vector<int> widths = sidecars->buckets(cb::s3gen_stage::cfm);
    require(widths.size() >= 2, "replay: the CFM sidecar has two widths");
    const cb::s3gen_gguf_facts & facts = sidecars->facts();
    // The call fits a width below the widest, so a wider one covers it too.
    const int speech = std::min(90, widths[widths.size() - 2] / facts.token_mel_ratio - facts.builtin_prompt_tokens -
                                        cb::S3GEN_LOOKAHEAD_TOKENS);
    require(speech >= 8, "replay: a call fits below the widest CFM width");
    const std::vector<int32_t> tokens = fixture_tokens(speech);
    s3gen_synthesize_opts opts;
    opts.s3gen_gguf_path = gguf;
    opts.n_gpu_layers = 99;
    opts.seed = 42;
    opts.coreml = waited;
    const cb::s3gen_call_shape call = cb::s3gen_call_shape_for(opts, tokens, facts);
    const int exact = cb::coreml_pick_bucket(widths, call.frames);
    require(exact > 0 && exact < widths.back(), "replay: the call's width is not the widest");
    const int next = *std::upper_bound(widths.begin(), widths.end(), exact);
    cb::s3gen_call_shape wider = call;
    wider.frames = next;
    sidecars->predict({call, wider});
    require(wait_for([&] {
                return sidecars->state(cb::s3gen_stage::cfm, exact) == "ready" &&
                       sidecars->state(cb::s3gen_stage::cfm, next) == "ready";
            }, 900000),
            "replay: the call's width and a wider one load");
    sidecars->end_call();

    auto run = [&](std::shared_ptr<const S3genCoremlPlan> plan, const CoremlOptions & coreml, int & rc) {
        s3gen_synthesize_opts o = opts;
        o.coreml = coreml;
        o.coreml_plan = std::move(plan);
        run_result r;
        o.pcm_out = &r.wav;
        o.s3gen_backend_out = &r.backend;
        rc = s3gen_synthesize_to_wav(tokens, o);
        return r;
    };
    CoremlOptions off;
    off.use = CoremlOptions::Use::off;
    int rc = -1;
    const run_result reference = run(nullptr, off, rc);
    require(rc == 0, "replay: the use = off reference");

    auto plan = s3gen_acquire_coreml_plan(opts, tokens, std::chrono::steady_clock::now());
    require(plan && cb::plan_view(*plan, cb::s3gen_stage::cfm).bucket == exact,
            "replay: the call holds its exact CFM width");
    cb::defer_coreml_failures(*plan);
    cb::coreml_test().fail_cfm_step = 1;
    const run_result handed_back = run(plan, waited, rc);
    require(cb::coreml_test().fail_cfm_step == -1, "replay: the failure was injected");
    require(rc == cb::S3GEN_RC_COREML_REDO && handed_back.wav.empty(),
            "replay: a deferred failure hands the call back with nothing rendered");
    const run_result replayed = run(cb::replay_plan(plan), waited, rc);
    require(rc == 0, "replay: the replay renders");
    auto fresh = s3gen_acquire_coreml_plan(opts, tokens, std::chrono::steady_clock::now());
    std::printf("[cfm-replay] frames_%d failed; replay backend=%s, max |replay - use=off| = %.3e; a fresh plan "
                "would take frames_%d\n",
                exact, replayed.backend.c_str(),
                replayed.wav.size() == reference.wav.size() ? max_abs(replayed.wav, reference.wav) : -1.0,
                fresh ? cb::plan_view(*fresh, cb::s3gen_stage::cfm).bucket : -1);
    require(fresh && cb::plan_view(*fresh, cb::s3gen_stage::cfm).bucket == next,
            "replay: a wider ready width covers the call");
    require(replayed.backend == cb::GGML_STAGE_BACKEND, "replay: CFM runs on ggml, not on the wider width");
    require(same_pcm(replayed.wav, reference.wav), "replay: equal to a whole ggml integration from z0");
}
#endif

int engine_fixture(bool slow_load) {
#ifndef TTS_CPP_USE_COREML
    (void) slow_load;
    std::puts("SKIP: built without TTS_CPP_COREML");
    return SKIP;
#else
    const char * t3_env = std::getenv("CHATTERBOX_COREML_TEST_T3");
    const char * s3gen_env = std::getenv("CHATTERBOX_COREML_TEST_S3GEN");
    if (!t3_env || !*t3_env || !s3gen_env || !*s3gen_env) {
        std::puts("SKIP: CHATTERBOX_COREML_TEST_T3 / CHATTERBOX_COREML_TEST_S3GEN are not set");
        return SKIP;
    }
    const char * cli_env = std::getenv("CHATTERBOX_COREML_TEST_CLI");
    if (slow_load && (!cli_env || !*cli_env || !fs::exists(cli_env))) {
        std::puts("SKIP: --slow-load needs CHATTERBOX_COREML_TEST_CLI (a tts-cli binary)");
        return SKIP;
    }
    const std::string t3 = t3_env, s3gen = s3gen_env;
    for (const char * suffix : {cb::COREML_ENCODER_SUFFIX, cb::COREML_CFM_SUFFIX, cb::COREML_HIFT_SUFFIX}) {
        if (!fs::exists(cb::coreml_stage_path(s3gen, suffix))) {
            std::printf("SKIP: %s is missing\n", cb::coreml_stage_path(s3gen, suffix).c_str());
            return SKIP;
        }
    }
    env_guard units(cb::COREML_UNITS_ENV), strict(cb::COREML_STRICT_ENV), disabled(cb::COREML_DISABLE_ENV);
    env_guard cache("QVAC_COREML_CACHE_DIR");
    unsetenv(cb::COREML_UNITS_ENV);
    unsetenv(cb::COREML_STRICT_ENV);
    unsetenv(cb::COREML_DISABLE_ENV);
    unsetenv("QVAC_COREML_CACHE_DIR");
    temp_dir root("engine");
    const auto segments = tts_cpp::chatterbox::text_preprocess::split_text_for_tts(OVERLAP_TEXT,
                                                                                   OVERLAP_SENTENCE_CHARS);
    require(segments.size() == 3, "the overlap text splits into three segments");

    // First, before anything is loaded in this process.
    const tts_cpp::FitResult fit_before = fit_check(t3, s3gen, root.path, false, nullptr);
    if (slow_load) slow_load_checks(t3, s3gen, cli_env, root.path);
    two_engines_check(t3, s3gen);
    disabled_engine_check(t3, s3gen);
    cli_in_process_check(t3, s3gen, root.path);
    if (slow_load) cli_replay_check(t3, s3gen, cli_env, root.path);
    if (slow_load) cli_status_case_check(t3, s3gen, cli_env, root.path);

    // Overlapped synthesis equals the segments synthesized one call at a time
    // (one segment never overlaps) and crossfaded in order, sample for sample:
    // T3 and the CFM noise are seeded per segment, so only order can differ.
    // Every function is loaded at construction (prewarm = all, wait =
    // construct), so each sentence overlaps and always runs on its exact
    // widths (a call takes the narrowest ready covering width).
    EngineOptions waited_opts = engine_options(t3, s3gen);
    waited_opts.coreml.wait = CoremlOptions::Wait::construct;
    waited_opts.coreml.prewarm = CoremlOptions::Prewarm::all;
    Engine engine(waited_opts);
    require(engine.coreml_wait(0), "every function is loaded at construction");
    const std::string constructed = engine.coreml_construction();
    std::printf("[construction] success=%d nothing_to_prepare=%d\n",
                (int) (constructed.find("\"success\":true") != std::string::npos),
                (int) (constructed.find("\"nothing_to_prepare\":true") != std::string::npos));
    require(constructed.find("\"success\":true") != std::string::npos &&
                constructed.find("\"nothing_to_prepare\":false") != std::string::npos &&
                constructed.find("\"state\":\"ready\"") != std::string::npos,
            "the construction result lists every function ready");
    fit_check(t3, s3gen, root.path, true, &fit_before);
    const SynthesisResult overlapped = engine.synthesize(OVERLAP_TEXT);
    // Timed again once every function is compiled and loaded, so the cancel
    // points below fall inside a warm request.
    const auto t0 = std::chrono::steady_clock::now();
    require(same_pcm(engine.synthesize(OVERLAP_TEXT).pcm, overlapped.pcm), "overlapped synthesis is repeatable");
    const auto elapsed = std::chrono::steady_clock::now() - t0;
    std::vector<float> sequential;
    std::set<std::string> backends;
    int tokens = 0;
    for (const std::string & segment : segments) {
        const SynthesisResult one = engine.synthesize(segment);
        tts_cpp::chatterbox::text_preprocess::append_pcm_crossfade(sequential, one.pcm, one.sample_rate,
                                                                   engine.options().crossfade_ms);
        backends.insert(one.s3gen_backend);
        tokens += one.t3_tokens;
    }
    std::printf("[overlap] %zu samples in %.2f s, backend=%s, max |overlapped - sequential| = %.3e\n",
                overlapped.pcm.size(), std::chrono::duration<double>(elapsed).count(),
                overlapped.s3gen_backend.c_str(),
                overlapped.pcm.size() == sequential.size() ? max_abs(overlapped.pcm, sequential) : -1.0);
    require(backends == std::set<std::string>{"coreml-ane"} && overlapped.s3gen_backend == "coreml-ane",
            "every segment and the aggregate report the Core ML backend");
    require(overlapped.t3_tokens == tokens, "overlapped T3 token count matches the segments'");
    require(same_pcm(overlapped.pcm, sequential), "overlapped PCM equals the in-order sequential PCM");

    // Engine::cancel() at points across the request: each run either finishes
    // or throws a cancellation (from T3, the S3Gen worker or the segment
    // boundary), never anything else, and the next request is unaffected.
    int cancelled = 0;
    for (double fraction : {0.2, 0.45, 0.7}) {
        std::thread canceller([&] {
            std::this_thread::sleep_for(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed * fraction));
            engine.cancel();
        });
        std::string error;
        try {
            engine.synthesize(OVERLAP_TEXT);
        } catch (const std::runtime_error & e) {
            error = e.what();
        }
        canceller.join();
        std::printf("[cancel at %.0f%%] %s\n", fraction * 100, error.empty() ? "finished" : error.c_str());
        require(error.empty() || error.find("cancelled") != std::string::npos, "a cancel surfaces as a cancellation");
        cancelled += !error.empty();
    }
    require(cancelled > 0, "at least one cancel lands mid-request");
    require(same_pcm(engine.synthesize(OVERLAP_TEXT).pcm, overlapped.pcm), "a cancelled engine synthesizes again");

    // A CFM sidecar that cannot serve (here, a package without frames_*
    // functions): strict mode refuses the call, and without strict the
    // segments run CFM on ggml (so they do not overlap) and aggregate as
    // mixed.
    const fs::path broken = stage_dir(root.path, s3gen, {cb::COREML_ENCODER_SUFFIX, cb::COREML_HIFT_SUFFIX}, "broken");
    const std::string broken_gguf = (broken / fs::path(s3gen).filename()).string();
    fs::create_directory_symlink(fs::absolute(cb::coreml_stage_path(s3gen, cb::COREML_HIFT_SUFFIX)),
                                 cb::coreml_stage_path(broken_gguf, cb::COREML_CFM_SUFFIX));
    EngineOptions strict_opts = engine_options(t3, broken_gguf);
    strict_opts.coreml_strict = true;
    std::string error;
    try {
        Engine(strict_opts).synthesize(OVERLAP_TEXT);
    } catch (const std::runtime_error & e) {
        error = e.what();
    }
    std::printf("[worker error] %s\n", error.c_str());
    require(error.find("cannot serve") != std::string::npos, "the worker's strict-mode error reaches the caller");
    EngineOptions lenient_opts = engine_options(t3, broken_gguf);
    lenient_opts.coreml_strict = false;
    // Its own encoder and HiFT functions load before each sentence (an
    // Engine shares no slots with the one above).
    lenient_opts.coreml.wait = CoremlOptions::Wait::call;
    Engine lenient(lenient_opts);
    const std::string broken_construction = lenient.coreml_construction();
    require(broken_construction.find("\"success\":false") != std::string::npos &&
                broken_construction.find("\"stage\":\"cfm\"") != std::string::npos,
            "a CFM sidecar without usable functions fails the construction result");
    const SynthesisResult mixed = lenient.synthesize(OVERLAP_TEXT);
    std::printf("[fallback] %zu samples, backend=%s\n", mixed.pcm.size(), mixed.s3gen_backend.c_str());
    require(!mixed.pcm.empty() && mixed.s3gen_backend == cb::MIXED_STAGE_BACKEND,
            "a CFM fallback aggregates as mixed");

    cfm_rollback_check(s3gen, root.path);
    cfm_replay_check(s3gen, root.path);

    // A prediction failure on the worker during the overlap (sentence 1's
    // CFM; last: the retirement is process-wide): the worker hands the
    // sentence back instead of running ggml there, and the dispatching thread
    // renders it again only after T3 of sentence 2 finished, CFM on ggml (every
    // width is loaded, so a wider covering one was ready), the encoder and
    // HiFT on Core ML.
    auto & hooks = cb::coreml_test();
    hooks.clear_events();
    hooks.tracing = true;
    hooks.fail_stage = (int) cb::s3gen_stage::cfm;
    const SynthesisResult redone = engine.synthesize(OVERLAP_TEXT);
    hooks.tracing = false;
    require(hooks.fail_stage == -1, "the worker's CFM prediction failed");
    const auto events = hooks.events();
    std::string trace;
    for (const auto & e : events) trace += (trace.empty() ? "" : ", ") + e;
    std::printf("[overlap retire] %s; backend=%s\n", trace.c_str(), redone.s3gen_backend.c_str());
    auto at = [&](const std::string & e) {
        return (size_t) (std::find(events.begin(), events.end(), e) - events.begin());
    };
    require(at("handed back 0") < events.size(), "the worker returned the sentence for a redo");
    require(at("replay 0 cfm=ggml") < events.size(), "the replay runs CFM on ggml");
    require(at("t3 1") < at("replay 0 cfm=ggml") && at("handed back 0") < at("replay 0 cfm=ggml"),
            "the replay starts after the overlapping T3 finished");
    require(redone.s3gen_backend == cb::MIXED_STAGE_BACKEND, "the replayed sentence mixes ggml CFM with Core ML");
    s3gen_unload();
    return 0;
#endif
}

// Streaming through the Engine while its Core ML functions load. Every load
// is held until the first chunk's callback, which releases them and waits
// until they finished, between two callbacks. The first chunk runs on ggml,
// marked pending, and equals a use = off stream's first chunk sample for
// sample; every later chunk, across all three sentences, runs on Core ML on
// the plan it acquires for itself. The request aggregates as mixed with
// coreml_pending set; chunk indices run 0..n-1, only the last one is
// is_last, it runs on Core ML and the chunks concatenate to result.pcm. A
// second stream, everything loaded, runs on Core ML with nothing pending.
int streaming_fixture() {
#ifndef TTS_CPP_USE_COREML
    std::puts("SKIP: built without TTS_CPP_COREML");
    return SKIP;
#else
    const char * t3_env = std::getenv("CHATTERBOX_COREML_TEST_T3");
    const char * s3gen_env = std::getenv("CHATTERBOX_COREML_TEST_S3GEN");
    if (!t3_env || !*t3_env || !s3gen_env || !*s3gen_env) {
        std::puts("SKIP: CHATTERBOX_COREML_TEST_T3 / CHATTERBOX_COREML_TEST_S3GEN are not set");
        return SKIP;
    }
    const std::string t3 = t3_env, s3gen = s3gen_env;
    for (const char * suffix : {cb::COREML_ENCODER_SUFFIX, cb::COREML_CFM_SUFFIX, cb::COREML_HIFT_SUFFIX}) {
        if (!fs::exists(cb::coreml_stage_path(s3gen, suffix))) {
            std::printf("SKIP: %s is missing\n", cb::coreml_stage_path(s3gen, suffix).c_str());
            return SKIP;
        }
    }
    env_guard units(cb::COREML_UNITS_ENV), strict(cb::COREML_STRICT_ENV), disabled(cb::COREML_DISABLE_ENV);
    env_guard cache("QVAC_COREML_CACHE_DIR");
    unsetenv(cb::COREML_UNITS_ENV);
    unsetenv(cb::COREML_STRICT_ENV);
    unsetenv(cb::COREML_DISABLE_ENV);
    unsetenv("QVAC_COREML_CACHE_DIR");
    temp_dir root("streaming");

    auto stream_options = [&](CoremlOptions::Use use) {
        EngineOptions opts = engine_options(t3, s3gen);
        opts.stream_first_chunk_tokens = 10;
        opts.stream_chunk_tokens = 25;
        opts.stream_left_context_tokens = 25;
        opts.coreml.use = use;
        opts.coreml.cache_dir = (root.path / "cache").string();
        return opts;
    };
    struct chunk {
        std::vector<float> pcm;
        int index = 0;
        bool last = false;
    };
    auto stream = [](Engine & engine, std::vector<chunk> & chunks, const std::function<void(int)> & after) {
        return engine.synthesize(OVERLAP_TEXT, [&](const float * pcm, std::size_t n, int index, bool last) {
            chunks.push_back(chunk{std::vector<float>(pcm, pcm + n), index, last});
            if (after) after(index);
        });
    };

    std::vector<chunk> off_chunks;
    {
        Engine off_engine(stream_options(CoremlOptions::Use::off));
        const SynthesisResult off = stream(off_engine, off_chunks, nullptr);
        require(off.s3gen_backend == cb::GGML_STAGE_BACKEND && !off.coreml_pending && off_chunks.size() > 3,
                "use = off streams on ggml, several chunks");
    }

    auto & hooks = cb::coreml_test();
    hooks.hold_loads = true;
    Engine engine(stream_options(CoremlOptions::Use::unset));
    hooks.clear_events();
    hooks.tracing = true;
    std::vector<chunk> chunks;
    bool released_settled = false;
    std::string error;
    SynthesisResult first;
    try {
        first = stream(engine, chunks, [&](int index) {
            if (index != 0) return;
            hooks.hold_loads = false;  // between the first and the second callback
            released_settled = wait_for([&] { return engine_settled(engine); }, 900000);
        });
    } catch (const std::exception & e) {
        error = e.what();
    }
    hooks.tracing = false;
    hooks.hold_loads = false;
    require(error.empty(), "the held stream finishes: " + error);
    require(released_settled, "the functions load once released between two callbacks");

    std::vector<std::string> routes;
    for (const std::string & e : hooks.events()) {
        if (e.rfind("chunk ", 0) == 0) routes.push_back(e);
    }
    std::string trace;
    for (const auto & r : routes) trace += (trace.empty() ? "" : ", ") + r;
    std::printf("[streaming] %zu chunks: %s\n", chunks.size(), trace.c_str());
    std::printf("[streaming] aggregate backend=%s pending=%d; max |first chunk - use=off| = %.3e\n",
                first.s3gen_backend.c_str(), (int) first.coreml_pending,
                !chunks.empty() && !off_chunks.empty() && chunks[0].pcm.size() == off_chunks[0].pcm.size()
                    ? max_abs(chunks[0].pcm, off_chunks[0].pcm)
                    : -1.0);
    require(chunks.size() > 3 && routes.size() == chunks.size(), "every chunk reports its routing");
    require(routes[0] == "chunk 0 s3gen=ggml pending", "the first chunk runs on ggml while every load is held");
    require(same_pcm(chunks[0].pcm, off_chunks[0].pcm), "the first chunk equals a use = off stream's first chunk");
    for (size_t i = 1; i < routes.size(); ++i) {
        require(routes[i] == "chunk " + std::to_string(i) + " s3gen=coreml-ane",
                "every chunk after the release runs on Core ML (" + routes[i] + ")");
    }
    require(first.s3gen_backend == cb::MIXED_STAGE_BACKEND && first.coreml_pending,
            "the request aggregates as mixed with coreml_pending set");
    std::vector<float> joined;
    for (size_t i = 0; i < chunks.size(); ++i) {
        require(chunks[i].index == (int) i, "chunk indices run 0..n-1");
        require(chunks[i].last == (i + 1 == chunks.size()), "only the final chunk is last");
        joined.insert(joined.end(), chunks[i].pcm.begin(), chunks[i].pcm.end());
    }
    require(!chunks.back().pcm.empty(), "the final chunk carries audio");
    require(same_pcm(joined, first.pcm), "the chunks concatenate to the result's PCM");
    std::printf("[streaming] final chunk %d last=1 on Core ML, %zu samples; chunks join to result.pcm\n",
                chunks.back().index, chunks.back().pcm.size());

    std::vector<chunk> warm_chunks;
    hooks.clear_events();
    hooks.tracing = true;
    const SynthesisResult warm = stream(engine, warm_chunks, nullptr);
    hooks.tracing = false;
    std::printf("[streaming] loaded: %zu chunks, backend=%s pending=%d\n", warm_chunks.size(),
                warm.s3gen_backend.c_str(), (int) warm.coreml_pending);
    require(warm.s3gen_backend == "coreml-ane" && !warm.coreml_pending && !warm_chunks.empty() &&
                warm_chunks.back().last,
            "once loaded, a stream runs every chunk on Core ML with nothing pending");
    s3gen_unload();
    return 0;
#endif
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc > 1 && std::string(argv[1]) == "--stages-fixture") return stages_fixture();
        if (argc > 1 && std::string(argv[1]) == "--shape-fixture") return shape_fixture();
        if (argc > 1 && std::string(argv[1]) == "--pending-fixture") return pending_fixture();
        if (argc > 1 && std::string(argv[1]) == "--streaming-fixture") return streaming_fixture();
        if (argc > 1 && std::string(argv[1]) == "--engine-fixture") {
            return engine_fixture(argc > 2 && std::string(argv[2]) == "--slow-load");
        }
        check_options();
        check_cli_arguments();
        check_empty_group();
        check_naming();
        check_hooks_record_only_when_tracing();
        check_buckets();
        check_windows();
        check_warm_estimates();
        check_ratio_planning();
        check_shapes_and_estimate();
        check_cfm_rollback();
        check_fit_and_detection();
        std::puts("test-chatterbox-coreml: ok");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "test-chatterbox-coreml: FAIL: %s\n", e.what());
        return 1;
    }
}
