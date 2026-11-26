// Parakeet Core ML runtime adoption on the device. `--case` picks one group,
// each registered as its own CTest test so a missing fixture shows as a test
// that did not run, never as a pass:
//   transcription  (default) --gguf TDT.gguf (with its adjacent
//                  `-encoder.mlmodelc`) --wav W.wav: every case below except
//                  the two that follow
//   diarizer       --diar-gguf D.gguf (with its `-diarizer-<frames>.mlmodelc`
//                  sidecars) --wav W.wav --gguf TDT.gguf: diarizer-pending,
//                  the diarizer load-error classes, cli-status
//   encoderless    --encoderless-gguf E.gguf: encoderless-use-off,
//                  owner-push-stream
//   eu-labels      --eu-gguf EU.gguf (with its Core ML programs) --wav W.wav
//                  (longer than the 20 s language gate): eu-labels
//   fit-layouts    --eu-gguf EU.gguf --sortformer-gguf S.gguf (v2.1): fit-layouts
//   sortformer     --sortformer-gguf S.gguf (v2.1, with its 410-frame
//                  `-diarizer-410.mlmodelc`) --wav W.wav: strict-too-wide
//   pending-encoder --sub-gguf T.gguf --sub-encoder E.mlmodelc --sub-decoder
//                  D.mlmodelc --wav W.wav (a standard TDT model, its encoder
//                  and decoder sidecars): decoder-ready-encoder-pending
//
//   ggml-unchanged        a call whose sidecar is still loading runs on ggml
//                         (coreml_pending) and its transcript equals use = off
//   prebuilt-slot-loads   the prebuilt encoder loads in the background through
//                         its slot; the next call runs on Core ML
//   capacity-miss         with long-form windowing disabled, an input longer
//                         than the fixed encoder's capacity runs that call on
//                         ggml and leaves the slot ready; the next call that
//                         fits runs on Core ML (a capacity miss never retires)
//   strict-over-capacity  the same over-capacity call with use = require
//                         throws instead of running on ggml, and
//                         the slot stays ready; with default windowing too,
//                         since a standard TDT sidecar is never split into
//                         Core ML windows
//   load-error-class      a corrupt encoder bundle fails its slot as transient,
//                         naming the Core ML error domain; a diarizer bundle
//                         whose capacity is not the one its name says fails
//                         as deterministic, a corrupt one as transient
//   fit-stable            fit_params on the same GGUF and options is identical
//                         with no hint, after the encoder load wrote its warm
//                         hint, and with that hint stale; each charges the
//                         sidecar (nonzero), so a projection that read the
//                         cache would show
//   fit-degrade           memory-degrade at construction follows the host's
//                         fit (EngineOptions::coreml_fit, assessed with a
//                         nondefault margin and audio length): at the host's
//                         Core ML-inclusive total the sidecar stays on; one
//                         byte less it is degraded (encoder slot disabled), although fit
//                         with the defaults (longer audio, larger margin)
//                         would not fit ggml there and so would keep it
//   strict-push-stream    with use = require, stream_start refuses a TDT push
//                         stream whose windows are longer than the sidecar,
//                         before any audio; use = automatic still starts it
//   diarizer-pending      a diarizer capacity that is not ready -> kNoSidecar
//                         -> ggml; ready after prewarm all -> Core ML
//   coreml-only-full      PARAKEET_COREML_ONLY with a full GGUF keeps the ggml
//                         encoder when Core ML is off or the takeover fails:
//                         the engine is not on Core ML and transcribes as
//                         use = off
//   encoderless-use-off   an encoder-less GGUF with use = off fails to load
//                         with kGgufLoadNoEncoderWeights (rc=17), as today
//   eu-labels             TranscribePsy EU reports the program each dispatch
//                         ran on: with the gate programs removed the gate runs
//                         on ggml and the main pass on Core ML, so the call is
//                         "mixed" and not encoder_used_coreml; with only the
//                         flexible programs the label names that program, not
//                         the held route's
//   fit-layouts           fit charges the programs the loader opens: a
//                         TranscribePsy EU deployment with only the shared
//                         multifunction bundle and a Sortformer v2.1 one with
//                         only the bypass bundle both get a Core ML charge, so
//                         a budget between the ggml-only and the inclusive
//                         totals separates the two verdicts (the bundles are
//                         staged: fit reads their size, never loads them)
//   strict-too-wide       use = require: a Sortformer pass longer than every
//                         diarizer capacity throws a capacity error (under
//                         automatic it runs on ggml)
//   owner-push-stream     on that GGUF with its fixed-capacity sidecar owning
//                         the encoder, stream_start refuses windows longer
//                         than the capacity under use = require and under
//                         automatic (no ggml weights), before any audio
//
// Public Engine API only, so it links against a shared qvac-parakeet too; the
// loader-level ownership checks live in test-coreml-encoder-owner. Every case
// runs or fails: there is no skip path (exit 0 = pass, 1 = fail, 2 = bad
// arguments).

#include "coreml_test_options.h"
#include "parakeet/cli.h"
#include "parakeet/engine.h"
#include "parakeet/fit.h"
#include "test_utils.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iterator>
#include <memory>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include <unistd.h>

namespace fs = std::filesystem;
using parakeet::CoremlOptions;

namespace {

int g_failures = 0;

void expect(bool ok, const char * what) {
    std::printf("[%s] %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++g_failures;
}

constexpr const char * kCoremlOnlyEnv = "PARAKEET_COREML_ONLY";

fs::path fresh_dir(const char * tag) {
    const fs::path dir = fs::temp_directory_path() /
                         ("parakeet-coreml-runtime-" + std::string(tag) + "-" + std::to_string(::getpid()));
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);
    return dir;
}

parakeet::EngineOptions engine_options(const std::string & gguf, CoremlOptions coreml) {
    parakeet::EngineOptions o;
    o.model_gguf_path = gguf;
    o.n_gpu_layers = 99;
    o.coreml = std::move(coreml);
    return o;
}

CoremlOptions off() {
    CoremlOptions o;
    o.use = CoremlOptions::Use::off;
    return o;
}

// Nothing requested at construction and a cache with no hints: the first
// call's acquisition requests the load and cannot wait for it (no warm hint,
// no bounded wait), so it runs on ggml.
CoremlOptions lazy(const fs::path & cache) {
    CoremlOptions o;
    o.prewarm = CoremlOptions::Prewarm::none;
    o.wait = CoremlOptions::Wait::none;
    o.cache_dir = cache.string();
    return o;
}

bool has(const std::string & text, const char * needle) { return text.find(needle) != std::string::npos; }

void case_transcription(const std::string & gguf, const std::string & wav,
                        parakeet::EngineResult & reference) {
    parakeet::Engine ggml_engine(engine_options(gguf, off()));
    reference = ggml_engine.transcribe(wav);
    expect(reference.encoder_backend == "ggml" && !reference.coreml_pending,
           "use = off: encoder_backend ggml, not pending");
    expect(!reference.text.empty(), "use = off produced a transcript");

    const fs::path cache = fresh_dir("cache");
    parakeet::Engine engine(engine_options(gguf, lazy(cache)));
    const parakeet::EngineResult first = engine.transcribe(wav);
    std::printf("       pending call: backend=%s pending=%d\n", first.encoder_backend.c_str(),
                first.coreml_pending ? 1 : 0);
    expect(first.encoder_backend == "ggml" && first.coreml_pending && !first.encoder_used_coreml,
           "ggml-unchanged: the first call ran on ggml because the sidecar was loading");
    expect(first.text == reference.text && first.token_ids == reference.token_ids,
           "ggml-unchanged: the pending call's transcript equals use = off (no template is embedded "
           "in this build, so this is the without-template side)");

    const std::string prewarm = engine.coreml_prewarm(CoremlOptions::Prewarm::minimal, /*wait=*/true);
    expect(has(prewarm, "\"success\":true") || has(prewarm, "\"success\": true"),
           "prebuilt-slot-loads: the minimal group reached ready");
    const std::string status = engine.coreml_status_json();
    std::printf("       status: %s\n", status.c_str());
    expect(has(status, "\"route\":\"prebuilt\"") || has(status, "\"route\": \"prebuilt\""),
           "prebuilt-slot-loads: status reports the prebuilt route");
    const parakeet::EngineResult second = engine.transcribe(wav);
    std::printf("       ready call: backend=%s pending=%d\n", second.encoder_backend.c_str(),
                second.coreml_pending ? 1 : 0);
    expect(second.encoder_used_coreml && second.encoder_backend.rfind("coreml-", 0) == 0 &&
               !second.coreml_pending,
           "prebuilt-slot-loads: the next call runs on Core ML");
    std::error_code ec;
    fs::remove_all(cache, ec);
}

// `"key":"value"` in the status JSON, compact or pretty-printed.
bool has_kv(const std::string & json, const char * key, const char * value) {
    const std::string k = std::string("\"") + key + "\":";
    const std::string v = std::string("\"") + value + "\"";
    return json.find(k + v) != std::string::npos || json.find(k + " " + v) != std::string::npos;
}

// The status object of one stage ("" when absent): the innermost object
// around its "stage" key, found by matching braces (keys may come in any
// order, and a slot object holds nested objects).
std::string stage_status(const std::string & status, const std::string & stage) {
    for (const char * sep : {"\"stage\":\"", "\"stage\": \""}) {
        const size_t at = status.find(sep + stage + "\"");
        if (at == std::string::npos) continue;
        size_t begin = std::string::npos;
        for (size_t i = at, depth = 0; i-- > 0;) {
            if (status[i] == '}') ++depth;
            else if (status[i] == '{') {
                if (depth == 0) { begin = i; break; }
                --depth;
            }
        }
        if (begin == std::string::npos) continue;
        for (size_t i = begin, depth = 0; i < status.size(); ++i) {
            if (status[i] == '{') ++depth;
            else if (status[i] == '}' && --depth == 0) return status.substr(begin, i - begin + 1);
        }
    }
    return {};
}

// A directory Core ML cannot load as a model.
void write_corrupt_bundle(const fs::path & bundle) {
    fs::create_directories(bundle / "weights");
    std::ofstream(bundle / "coremldata.bin") << "not a Core ML model";
    std::ofstream(bundle / "model.mil") << "not MIL";
    std::ofstream(bundle / "weights" / "weight.bin") << std::string(64, 'x');
}

void case_encoder_load_error(const std::string & gguf) {
    const fs::path dir = fresh_dir("corrupt-encoder");
    const fs::path staged = dir / "model.gguf";
    fs::create_symlink(fs::absolute(gguf), staged);
    write_corrupt_bundle(dir / "model-encoder.mlmodelc");
    parakeet::Engine engine(engine_options(staged.string(), coreml_ready_options()));
    const std::string encoder = stage_status(engine.coreml_status_json(), "encoder");
    std::printf("       corrupt encoder: %s\n", encoder.c_str());
    expect(has_kv(encoder, "state", "failed") && has_kv(encoder, "error_class", "transient") &&
               encoder.find("com.apple.CoreML") != std::string::npos,
           "load-error-class: a corrupt encoder bundle fails as transient with the Core ML error domain");

    // use = require with the failed encoder: construction (with the ggml
    // prewarm pass) and the readiness accessors never throw; the call does.
    CoremlOptions strict = coreml_ready_options();
    strict.use = CoremlOptions::Use::require;
    parakeet::EngineOptions so = engine_options(staged.string(), strict);
    so.prewarm = true;
    bool probes_ok = false, call_threw = false;
    try {
        parakeet::Engine strict_engine(so);
        probes_ok = !strict_engine.encoder_on_coreml() && !strict_engine.encoder_backend().empty();
        try {
            std::vector<float> silence(16000, 0.0f);
            (void) strict_engine.transcribe_samples(silence.data(), (int) silence.size(), 16000);
        } catch (const std::runtime_error &) {
            call_threw = true;
        }
    } catch (const std::exception & e) {
        std::printf("       strict construction threw: %s\n", e.what());
    }
    expect(probes_ok && call_threw,
           "strict-probe: under require a failed encoder never throws from construction or the accessors, only from a call");
    std::error_code ec;
    fs::remove_all(dir, ec);
}

void case_strict_too_wide(const std::string & gguf, const std::string & wav) {
    std::vector<float> source;
    int rate = 0;
    if (!parakeet_test::load_wav_pcm16le_mono(wav, source, rate) || source.empty()) {
        expect(false, "strict-too-wide: the wav loads");
        return;
    }
    std::vector<float> audio;  // 60 s: 750 encoder frames, above the 410-frame capacity
    while (audio.size() < (size_t) rate * 60) audio.insert(audio.end(), source.begin(), source.end());
    CoremlOptions strict = coreml_ready_options();
    strict.use = CoremlOptions::Use::require;
    std::string error;
    try {
        parakeet::Engine engine(engine_options(gguf, strict));
        (void) engine.diarize_samples(audio.data(), (int) audio.size(), rate);
    } catch (const std::runtime_error & e) {
        error = e.what();
    }
    std::printf("       strict too-wide diarize: %s\n", error.empty() ? "(ran)" : error.c_str());
    expect(has(error, "longer than every Core ML diarizer capacity"),
           "strict-too-wide: under require a pass longer than every capacity throws a capacity error");
    parakeet::Engine automatic(engine_options(gguf, coreml_ready_options()));
    const parakeet::DiarizationResult r = automatic.diarize_samples(audio.data(), (int) audio.size(), rate);
    expect(r.encoder_backend == "ggml" && r.n_frames > 0, "strict-too-wide: under automatic that pass runs on ggml");
}

// The CLI with --diarization-model: the --coreml-* options reach the
// diarization engine too, and the status file describes both engines.
void case_cli_diarization_status(const std::string & tdt_gguf, const std::string & diar_gguf,
                                 const std::string & wav) {
    const fs::path dir = fresh_dir("cli-status");
    const std::string status_path = (dir / "status.json").string();
    std::vector<std::string> args = {"parakeet", "--model", tdt_gguf, "--diarization-model", diar_gguf,
                                     "--wav", wav, "--coreml-use", "off", "--coreml-status-json", status_path};
    std::vector<char *> argv;
    for (std::string & a : args) argv.push_back(a.data());
    const int rc = parakeet_cli_main((int) argv.size(), argv.data());
    std::ifstream in(status_path);
    const std::string status((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const size_t diar_at = status.find("\"diarization\":");
    const std::string diar = diar_at == std::string::npos ? std::string() : status.substr(diar_at);
    std::printf("       cli rc=%d status=%.300s\n", rc, status.c_str());
    expect(rc == 0 && status.find("\"transcription\":") != std::string::npos && !diar.empty() &&
               diar.find("diarizer-") != std::string::npos && has(diar, "\"state\":\"disabled\"") &&
               !has(diar, "\"state\":\"ready\""),
           "cli-status: with --coreml-use off the diarization engine's sidecars are disabled, and the "
           "status names both engines");
    std::error_code ec;
    fs::remove_all(dir, ec);
}

void case_diarizer_load_errors(const std::string & gguf) {
    // The smallest capacity bundle, staged under a larger capacity's name.
    const fs::path source_dir = fs::absolute(gguf).parent_path();
    const std::string stem = fs::path(gguf).stem().string();
    const std::string base = stem.substr(0, stem.find('.'));
    fs::path smallest;
    int smallest_frames = 0;
    for (const auto & entry : fs::directory_iterator(source_dir)) {
        const std::string name = entry.path().filename().string();
        const std::string prefix = base + "-diarizer-";
        if (name.rfind(prefix, 0) != 0 || entry.path().extension() != ".mlmodelc") continue;
        const int frames = std::atoi(name.c_str() + prefix.size());
        if (frames > 0 && (smallest_frames == 0 || frames < smallest_frames)) {
            smallest_frames = frames;
            smallest = entry.path();
        }
    }
    expect(smallest_frames > 0, "load-error-class: a diarizer bundle is next to the GGUF");
    if (smallest_frames == 0) return;
    const fs::path dir = fresh_dir("diarizer-errors");
    const fs::path staged = dir / (base + ".gguf");
    fs::create_symlink(fs::absolute(gguf), staged);
    const int renamed = smallest_frames * 2;
    const int corrupt = smallest_frames * 3;
    fs::create_directory_symlink(fs::canonical(smallest),
                                 dir / (base + "-diarizer-" + std::to_string(renamed) + ".mlmodelc"));
    write_corrupt_bundle(dir / (base + "-diarizer-" + std::to_string(corrupt) + ".mlmodelc"));
    // An engine on the original bundle stays alive: the alias at the same
    // canonical path must still be loaded and validated for its own capacity.
    parakeet::Engine original(engine_options(gguf, coreml_ready_options()));
    parakeet::Engine engine(engine_options(staged.string(), coreml_ready_options()));
    const std::string status = engine.coreml_status_json();
    const std::string wrong = stage_status(status, "diarizer-" + std::to_string(renamed));
    const std::string broken = stage_status(status, "diarizer-" + std::to_string(corrupt));
    std::printf("       renamed diarizer: %s\n       corrupt diarizer: %s\n", wrong.c_str(), broken.c_str());
    expect(has_kv(wrong, "state", "failed") && has_kv(wrong, "error_class", "deterministic"),
           "load-error-class: a diarizer whose capacity is not its name's fails as deterministic");
    expect(has_kv(broken, "state", "failed") && has_kv(broken, "error_class", "transient") &&
               broken.find("com.apple.CoreML") != std::string::npos,
           "load-error-class: a corrupt diarizer bundle fails as transient with the Core ML error domain");
    std::error_code ec;
    fs::remove_all(dir, ec);
}

// A directory holding the EU GGUF and the named sidecars next to it.
fs::path stage_eu(const std::string & gguf, const char * tag, const std::vector<std::string> & sidecars) {
    const fs::path dir = fresh_dir(tag);
    const fs::path source = fs::absolute(gguf).parent_path();
    const std::string base = fs::path(gguf).stem().string().substr(0, fs::path(gguf).stem().string().find('.'));
    fs::create_symlink(fs::canonical(gguf), dir / (base + ".gguf"));
    for (const std::string & suffix : sidecars) {
        const fs::path bundle = source / (base + suffix);
        if (fs::exists(bundle)) fs::create_directory_symlink(fs::canonical(bundle), dir / (base + suffix));
    }
    return dir / (base + ".gguf");
}

void case_eu_labels(const std::string & gguf, const std::string & wav) {
    std::vector<float> source;
    int rate = 0;
    expect(parakeet_test::load_wav_pcm16le_mono(wav, source, rate) && !source.empty(), "eu-labels: the wav loads");
    if (source.empty()) return;
    std::vector<float> audio;  // over the 20 s language gate
    while (audio.size() < (size_t) rate * 25) audio.insert(audio.end(), source.begin(), source.end());

    {
        const fs::path staged = stage_eu(gguf, "eu-no-gate", {"-encoder.mlmodelc", "-encoder-other.mlmodelc",
                                                             "-encoder-flexible.mlmodelc",
                                                             "-encoder-other-flexible.mlmodelc"});
        parakeet::EngineOptions o = engine_options(staged.string(), coreml_ready_options());
        o.language = "auto";
        parakeet::Engine engine(o);
        const parakeet::EngineResult r = engine.transcribe_samples(audio.data(), (int) audio.size(), rate);
        std::printf("       gate on ggml: backend=%s used_coreml=%d\n", r.encoder_backend.c_str(),
                    r.encoder_used_coreml ? 1 : 0);
        expect(r.encoder_backend == "mixed" && !r.encoder_used_coreml,
               "eu-labels: a ggml gate with a Core ML main pass reports mixed, not encoder_used_coreml");
        std::error_code ec;
        fs::remove_all(staged.parent_path(), ec);
    }
    {
        const fs::path staged = stage_eu(gguf, "eu-flexible", {"-encoder-flexible.mlmodelc"});
        parakeet::EngineOptions o = engine_options(staged.string(), coreml_ready_options());
        o.language = "en";
        parakeet::Engine engine(o);
        const parakeet::EngineResult r = engine.transcribe_samples(audio.data(), (int) audio.size(), rate);
        std::printf("       flexible only: backend=%s used_coreml=%d\n", r.encoder_backend.c_str(),
                    r.encoder_used_coreml ? 1 : 0);
        expect(r.encoder_used_coreml && r.encoder_backend.rfind("coreml-", 0) == 0,
               "eu-labels: a call on a program other than the route's is labelled with that program");
        std::error_code ec;
        fs::remove_all(staged.parent_path(), ec);
    }
}

// A bundle fit can size: weights/weight.bin of `bytes` (sparse) and, when
// given, a metadata.json declaring one fixed Float32 input and output of
// `io_floats` elements each. Never loaded.
void write_sized_bundle(const fs::path & bundle, uint64_t bytes, uint64_t io_floats = 0) {
    fs::create_directories(bundle / "weights");
    std::ofstream(bundle / "weights" / "weight.bin").close();
    fs::resize_file(bundle / "weights" / "weight.bin", bytes);
    if (io_floats > 0) {
        const std::string shape = "[1, " + std::to_string(io_floats) + "]";
        std::ofstream(bundle / "metadata.json")
            << R"([{"inputSchema":[{"name":"x","shape":")" << shape << R"(","dataType":"Float32"}],)"
            << R"("outputSchema":[{"name":"y","shape":")" << shape << R"(","dataType":"Float32"}]}])";
    }
}

uint64_t with_margin(uint64_t x) { return x + x / 4; }

parakeet::FitResult fit_at(const fs::path & staged, float seconds) {
    parakeet::FitOptions fo;
    fo.model_gguf_path = staged.string();
    fo.n_gpu_layers = 99;
    fo.audio_seconds = seconds;
    return parakeet::fit_params(fo);
}

// Construction at the inclusive boundary: one byte under it the engine
// degrades (its sidecar slots are disabled), at it it does not.
void check_boundary(const char * name, const fs::path & staged, const parakeet::FitResult & r) {
    auto encoder_state = [&](uint64_t budget) {
        CoremlOptions coreml;
        coreml.memory_budget_bytes = (int64_t) budget;
        coreml.prewarm = CoremlOptions::Prewarm::none;
        parakeet::EngineOptions o = engine_options(staged.string(), coreml);
        o.coreml_fit = parakeet::CoremlFitTotals{r.ggml_required_bytes, r.coreml.bytes};
        parakeet::Engine engine(o);
        return engine.coreml_status_json();
    };
    const uint64_t total = r.ggml_required_bytes + r.coreml.bytes;
    expect(has(encoder_state(total - 1), "\"disabled\"") && !has(encoder_state(total), "\"disabled\""),
           (std::string("fit-layouts: ") + name + " degrades construction exactly below its inclusive total").c_str());
}

void case_fit_layouts(const std::string & eu_gguf, const std::string & sortformer_gguf) {
    constexpr uint64_t kMiB = 1ull << 20;
    {
        // TranscribePsy EU with only the shared bundle: three fixed functions
        // (default units reach the Neural Engine), each charged its weights and
        // its declared I/O, plus one Neural Engine load transient.
        constexpr uint64_t kWeights = 1024 * kMiB;  // above the GGUF's fp16 encoder floor
        constexpr uint64_t kIoFloats = 8 * 1024 * 1024;
        const fs::path dir = fresh_dir("fit-eu-shared");
        const fs::path staged = dir / "transcribepsy-250m-eu.gguf";
        fs::create_symlink(fs::canonical(eu_gguf), staged);
        write_sized_bundle(dir / "transcribepsy-250m-eu-encoder-shared.mlmodelc", kWeights, kIoFloats);
        const uint64_t io = 2 * kIoFloats * 4;
        const uint64_t expected = with_margin(3 * (kWeights + 2 * io)) + with_margin(kWeights / 4);
        const parakeet::FitResult r = fit_at(staged, 60.0f);
        const parakeet::FitResult short_clip = fit_at(staged, 5.0f);
        std::printf("       EU shared: coreml=%llu expected=%llu short=%llu\n", (unsigned long long) r.coreml.bytes,
                    (unsigned long long) expected, (unsigned long long) short_clip.coreml.bytes);
        expect(r.coreml.encoder_route == "prebuilt" && r.coreml.bytes == expected,
               "fit-layouts: EU shared bundle only charges three functions' weights and declared I/O plus one transient");
        expect(short_clip.coreml.bytes == expected, "fit-layouts: a shorter clip does not shrink the fixed programs' charge");
        std::error_code ec;
        fs::remove_all(dir, ec);
    }
    {
        // Sortformer v2.1 with only the bypass bundle and two diarizer
        // capacities below the input: construction loads the smallest, the
        // input itself runs on ggml. Bypass (all units: 1.0x transient) and
        // diarizer (Neural Engine: 0.25x) are resident together; one transient.
        constexpr uint64_t kBypass = 64 * kMiB, kSmall = 32 * kMiB, kLarge = 48 * kMiB;
        constexpr uint64_t kIoFloats = 4 * 1024 * 1024;
        const fs::path dir = fresh_dir("fit-sortformer-bypass");
        const fs::path staged = dir / "sortformer.gguf";
        fs::create_symlink(fs::canonical(sortformer_gguf), staged);
        write_sized_bundle(dir / "sortformer-encoder-bypass-pre-encode.mlmodelc", kBypass, kIoFloats);
        write_sized_bundle(dir / "sortformer-diarizer-100.mlmodelc", kSmall);
        write_sized_bundle(dir / "sortformer-diarizer-200.mlmodelc", kLarge);
        const uint64_t bypass_resident = with_margin(kBypass + 2 * (2 * kIoFloats * 4));
        const uint64_t bypass_transient = with_margin(kBypass);
        const uint64_t diarizer_resident = 2 * kSmall + 160 * kMiB;  // two weight copies + runtime, measured
        const uint64_t diarizer_transient = kSmall / 4;
        const uint64_t expected =
            bypass_resident + diarizer_resident + std::max(bypass_transient, diarizer_transient);
        const parakeet::FitResult r = fit_at(staged, 60.0f);
        std::printf("       Sortformer bypass + diarizers: coreml=%llu expected=%llu legacy=%llu\n",
                    (unsigned long long) r.coreml.bytes, (unsigned long long) expected,
                    (unsigned long long) r.coreml_bytes);
        expect(r.coreml.encoder_route == "prebuilt" && r.coreml.bytes == expected,
               "fit-layouts: Sortformer bypass and construction diarizer resident together with one transient");
        expect(r.coreml_bytes == 0, "fit-layouts: today's diarizer charge stays the workload's (none covers 60 s)");
        check_boundary("Sortformer", staged, r);
        parakeet::FitOptions off;
        off.model_gguf_path = staged.string();
        off.n_gpu_layers = 99;
        off.audio_seconds = 60.0f;
        off.coreml.use = CoremlOptions::Use::off;
        const parakeet::FitResult disabled = parakeet::fit_params(off);
        expect(disabled.coreml.bytes == 0 && disabled.coreml_bytes == 0,
               "fit-layouts: use = off charges no sidecar");
        std::error_code ec;
        fs::remove_all(dir, ec);
    }
}

bool same_fit(const parakeet::FitResult & a, const parakeet::FitResult & b) {
    return a.status == b.status && a.ggml_required_bytes == b.ggml_required_bytes &&
           a.ggml_only_fits == b.ggml_only_fits && a.coreml.fits == b.coreml.fits &&
           a.coreml.bytes == b.coreml.bytes && a.coreml.encoder_route == b.coreml.encoder_route &&
           a.coreml.disabled_reason == b.coreml.disabled_reason && a.coreml_bytes == b.coreml_bytes;
}

void case_fit_stable(const std::string & gguf) {
    const fs::path cache = fresh_dir("fit-stable-cache");
    parakeet::FitOptions fo;
    fo.model_gguf_path = gguf;
    fo.n_gpu_layers = 99;
    fo.audio_seconds = 60.0f;
    fo.coreml.cache_dir = cache.string();
    const parakeet::FitResult before = parakeet::fit_params(fo);
    std::printf("       fit before any load: route=%s coreml=%llu\n", before.coreml.encoder_route.c_str(),
                (unsigned long long) before.coreml.bytes);
    expect(before.status != parakeet::FitStatus::Error && before.coreml.encoder_route == "prebuilt" &&
               before.coreml.bytes > 0,
           "fit-stable: fit_params charges the adjacent encoder sidecar");

    {
        CoremlOptions coreml = coreml_ready_options();
        coreml.cache_dir = cache.string();
        parakeet::Engine engine(engine_options(gguf, coreml));
        expect(engine.encoder_on_coreml(), "fit-stable: the encoder loaded");
    }
    std::vector<fs::path> hints;
    std::error_code ec;
    for (const auto & e : fs::recursive_directory_iterator(cache, ec)) {
        if (e.path().parent_path().filename() == "hints" && e.path().extension() == ".json") hints.push_back(e.path());
    }
    expect(!hints.empty(), "fit-stable: the load wrote its warm hint");
    const parakeet::FitResult warm = parakeet::fit_params(fo);
    expect(same_fit(before, warm) && warm.coreml.bytes > 0, "fit-stable: fit with a warm hint equals fit with none");

    for (const fs::path & hint : hints) {
        std::ifstream in(hint);
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const size_t at = text.find("\"stale_until\"");
        if (at == std::string::npos) continue;
        const size_t colon = text.find(':', at);
        const size_t end = text.find_first_of(",}", colon);
        text.replace(colon + 1, end - colon - 1, "\"2999-01-01T00:00:00Z\"");
        std::ofstream(hint) << text;
    }
    const parakeet::FitResult stale = parakeet::fit_params(fo);
    expect(same_fit(before, stale) && stale.coreml.bytes > 0,
           "fit-stable: fit with a stale hint equals fit with none");
    fs::remove_all(cache, ec);
}

void case_fit_degrade(const std::string & gguf) {
    parakeet::FitOptions host;  // the host's own assessment
    host.model_gguf_path = gguf;
    host.n_gpu_layers = 99;
    host.audio_seconds = 11.0f;
    host.margin_bytes = 0;
    const parakeet::FitResult assessed = parakeet::fit_params(host);
    parakeet::FitOptions defaults;  // what construction projected without the host's totals
    defaults.model_gguf_path = gguf;
    defaults.n_gpu_layers = 99;
    const parakeet::FitResult projected = parakeet::fit_params(defaults);
    const uint64_t budget = assessed.ggml_required_bytes + assessed.coreml.bytes;
    const uint64_t under = budget - 1;
    std::printf("       host fit: ggml=%llu coreml=%llu; default fit: ggml=%llu coreml=%llu; budget=%llu\n",
                (unsigned long long) assessed.ggml_required_bytes, (unsigned long long) assessed.coreml.bytes,
                (unsigned long long) projected.ggml_required_bytes, (unsigned long long) projected.coreml.bytes,
                (unsigned long long) budget);
    expect(assessed.coreml.bytes > 0 && assessed.ggml_required_bytes <= under &&
               !(projected.ggml_required_bytes <= under &&
                 projected.ggml_required_bytes + projected.coreml.bytes > under),
           "fit-degrade: one byte under the host's inclusive total, the host's fit degrades and the "
           "default projection would not");

    auto engine_at = [&](uint64_t b) {
        CoremlOptions coreml = coreml_ready_options();
        coreml.memory_budget_bytes = (int64_t) b;
        parakeet::EngineOptions o = engine_options(gguf, coreml);
        o.coreml_fit = parakeet::CoremlFitTotals{assessed.ggml_required_bytes, assessed.coreml.bytes};
        return std::make_unique<parakeet::Engine>(o);
    };
    expect(engine_at(budget)->encoder_on_coreml(),
           "fit-degrade: the host's Core ML-inclusive fit holds, so construction keeps the sidecar");
    const auto degraded = engine_at(under);
    const std::string degraded_status = degraded->coreml_status_json();
    std::printf("       degraded: on_coreml=%d status=%s\n", degraded->encoder_on_coreml() ? 1 : 0,
                degraded_status.c_str());
    // Memory-degrade resolves use = off: the encoder slot is disabled, with no route.
    const std::string encoder = stage_status(degraded_status, "encoder");
    expect(!degraded->encoder_on_coreml() && has_kv(encoder, "state", "disabled") &&
               has_kv(encoder, "route", "none"),
           "fit-degrade: one byte under the host's inclusive total, construction degrades for memory");
}

void case_diarizer(const std::string & gguf, const std::string & wav) {
    const fs::path cache = fresh_dir("diar-cache");
    parakeet::Engine engine(engine_options(gguf, lazy(cache)));
    const parakeet::DiarizationResult first = engine.diarize(wav);
    std::printf("       pending diarize: backend=%s pending=%d\n", first.encoder_backend.c_str(),
                first.coreml_pending ? 1 : 0);
    expect(first.encoder_backend == "ggml" && first.coreml_pending && !first.encoder_used_coreml,
           "diarizer-pending: a capacity that is not ready gives kNoSidecar -> ggml");

    engine.coreml_prewarm(CoremlOptions::Prewarm::all, /*wait=*/true);
    const parakeet::DiarizationResult second = engine.diarize(wav);
    std::printf("       ready diarize: backend=%s pending=%d\n", second.encoder_backend.c_str(),
                second.coreml_pending ? 1 : 0);
    expect(second.encoder_used_coreml && second.encoder_backend != "ggml",
           "diarizer-pending: after prewarm all the diarizer runs on Core ML");
    std::error_code ec;
    fs::remove_all(cache, ec);
}

void case_coreml_only_full(const std::string & gguf, const std::string & wav,
                           const parakeet::EngineResult & reference) {
    ::setenv(kCoremlOnlyEnv, "1", 1);
    {
        parakeet::Engine engine(engine_options(gguf, off()));
        const parakeet::EngineResult r = engine.transcribe(wav);
        expect(!engine.encoder_on_coreml() && r.encoder_backend == "ggml" && r.text == reference.text,
               "coreml-only-full: use = off keeps the GGUF's ggml encoder");
    }
    {
        // The takeover attempt fails (no sidecar next to this copy): the full
        // GGUF keeps its ggml encoder and transcribes as before.
        const fs::path dir = fresh_dir("no-sidecar");
        const fs::path staged = dir / fs::path(gguf).filename();
        fs::create_symlink(fs::absolute(gguf), staged);
        parakeet::Engine engine(engine_options(staged.string(), coreml_ready_options()));
        const parakeet::EngineResult r = engine.transcribe(wav);
        expect(!engine.encoder_on_coreml() && r.text == reference.text && r.encoder_backend == "ggml",
               "coreml-only-full: a failed takeover keeps the ggml encoder, which transcribes as use = off");
        std::error_code ec;
        fs::remove_all(dir, ec);
    }
    ::unsetenv(kCoremlOnlyEnv);
}

void case_encoderless(const std::string & gguf) {
    std::string error;
    try {
        parakeet::Engine engine(engine_options(gguf, off()));
    } catch (const std::runtime_error & e) {
        error = e.what();
    }
    std::printf("       encoder-less load: %s\n", error.empty() ? "(loaded)" : error.c_str());
    expect(has(error, "(rc=17)"), "encoderless-use-off: kGgufLoadNoEncoderWeights (rc=17), as today");

    // A 30 s chunk plus the default context is longer than the sidecar's
    // fixed window: no full window could run on Core ML or ggml.
    parakeet::StreamingOptions stream;
    stream.chunk_ms = 30000;
    for (const bool require : {true, false}) {
        CoremlOptions options = coreml_ready_options();
        if (require) options.use = CoremlOptions::Use::require;
        parakeet::Engine engine(engine_options(gguf, options));
        std::string refused;
        try {
            (void) engine.stream_start(stream, [](const parakeet::StreamingSegment &) {});
        } catch (const std::runtime_error & e) {
            refused = e.what();
        }
        std::printf("       owner stream_start (%s): %s\n", require ? "require" : "automatic",
                    refused.empty() ? "(started)" : refused.c_str());
        expect(engine.encoder_on_coreml() && has(refused, "push-stream windows") &&
                   has(refused, "fixed capacity"),
               require ? "owner-push-stream: use = require refuses windows over the owner's capacity"
                       : "owner-push-stream: automatic refuses them too (no ggml encoder weights)");
    }
}

}  // namespace

// The TDT fixture's encoder is a fixed 1501-frame export (15 s). 30 s is
// above it, so with windowing disabled the encoder cannot serve the call.
// decoder-ready-encoder-pending: a standard TDT model whose decoder sidecar
// loaded at construction while its encoder sidecar is still loading. The
// call runs the sequential ggml plan (coreml_pending, no independent
// segments, no Core ML decoder lanes) and transcribes like use = off.
void case_pending_encoder(const std::string & gguf, const std::string & encoder, const std::string & decoder,
                          const std::string & wav) {
    const fs::path dir = fresh_dir("pending-encoder");
    const fs::path model = dir / "m.gguf";
    fs::create_symlink(fs::absolute(gguf), model);
    fs::create_directory_symlink(fs::absolute(encoder), dir / "m-encoder.mlmodelc");
    fs::create_directory_symlink(fs::absolute(decoder), dir / "m-decoder.mlmodelc");
    parakeet::Engine ggml_engine(engine_options(model.string(), off()));
    const parakeet::EngineResult reference = ggml_engine.transcribe(wav);
    const parakeet::EngineResult first = parakeet::Engine(engine_options(model.string(), lazy(dir / "cache")))
                                             .transcribe(wav);
    std::printf("       pending call: backend=%s pending=%d segment_plan_ms=%.3f decoder_coreml=%d\n",
                first.encoder_backend.c_str(), first.coreml_pending ? 1 : 0, first.segment_plan_ms,
                first.decoder_used_coreml ? 1 : 0);
    expect(first.encoder_backend == "ggml" && first.coreml_pending && !first.encoder_used_coreml,
           "decoder-ready-encoder-pending: the call runs on ggml, coreml_pending");
    expect(first.segment_plan_ms == 0.0 && !first.decoder_used_coreml,
           "decoder-ready-encoder-pending: the sequential ggml plan, not independent segments");
    expect(first.text == reference.text && first.token_ids == reference.token_ids,
           "decoder-ready-encoder-pending: the transcript equals use = off");
    std::error_code ec;
    fs::remove_all(dir, ec);
}

void case_capacity_miss(const std::string & gguf, const std::string & wav) {
    std::vector<float> source;
    int rate = 0;
    expect(parakeet_test::load_wav_pcm16le_mono(wav, source, rate) && !source.empty(),
           "capacity-miss: the wav loads");
    if (source.empty()) return;
    std::vector<float> long_input;
    const size_t target = (size_t) rate * 30;
    while (long_input.size() < target) {
        const size_t n = std::min(source.size(), target - long_input.size());
        long_input.insert(long_input.end(), source.begin(), source.begin() + (long) n);
    }

    parakeet::EngineOptions o = engine_options(gguf, coreml_ready_options());
    o.long_form_window_frames = -1;
    parakeet::Engine engine(o);
    expect(engine.encoder_on_coreml(), "capacity-miss: the encoder is ready after construction");
    const parakeet::EngineResult over =
        engine.transcribe_samples(long_input.data(), (int) long_input.size(), rate);
    std::printf("       over-capacity call: mel=%d backend=%s\n", over.mel_frames,
                over.encoder_backend.c_str());
    expect(over.encoder_backend == "ggml" && !over.encoder_used_coreml && !over.text.empty(),
           "capacity-miss: the over-capacity single-pass call ran on ggml");
    const std::string status = engine.coreml_status_json();
    expect(!has(status, "\"retired\"") && engine.encoder_on_coreml(),
           "capacity-miss: the encoder slot is still ready (not retired)");
    const parakeet::EngineResult next = engine.transcribe(wav);
    std::printf("       next call: backend=%s\n", next.encoder_backend.c_str());
    expect(next.encoder_used_coreml && next.encoder_backend.rfind("coreml-", 0) == 0,
           "capacity-miss: the next call that fits runs on Core ML");

    // use = require: the call the sidecar cannot serve is an error.
    CoremlOptions strict = coreml_ready_options();
    strict.use = CoremlOptions::Use::require;
    parakeet::EngineOptions so = engine_options(gguf, strict);
    so.long_form_window_frames = -1;
    parakeet::Engine strict_engine(so);
    expect(strict_engine.encoder_on_coreml(), "strict-over-capacity: the encoder is ready");
    bool threw = false;
    try {
        (void) strict_engine.transcribe_samples(long_input.data(), (int) long_input.size(), rate);
    } catch (const std::runtime_error & e) {
        threw = has(e.what(), "Core ML is required");
        std::printf("       strict over-capacity call threw: %s\n", e.what());
    }
    expect(threw, "strict-over-capacity: the call throws instead of running on ggml");
    expect(!has(strict_engine.coreml_status_json(), "\"retired\"") && strict_engine.encoder_on_coreml(),
           "strict-over-capacity: the encoder slot is still ready");

    // Default windowing: a standard TDT sidecar (no validated seam context)
    // is never split into Core ML windows, so the same input is still one
    // dispatch the sidecar cannot serve.
    parakeet::Engine strict_default(engine_options(gguf, strict));
    expect(strict_default.encoder_on_coreml(),
           "strict-over-capacity: default windowing, the standard TDT sidecar is ready");
    threw = false;
    try {
        (void) strict_default.transcribe_samples(long_input.data(), (int) long_input.size(), rate);
    } catch (const std::runtime_error & e) {
        threw = has(e.what(), "Core ML is required");
        std::printf("       strict default-windowing call threw: %s\n", e.what());
    }
    expect(threw, "strict-over-capacity: with default windowing the standard TDT call throws too");
    expect(!has(strict_default.coreml_status_json(), "\"retired\"") && strict_default.encoder_on_coreml(),
           "strict-over-capacity: default windowing leaves the encoder slot ready");
    const parakeet::EngineResult fits = strict_default.transcribe(wav);
    expect(fits.encoder_used_coreml && fits.encoder_backend.rfind("coreml-", 0) == 0,
           "strict-over-capacity: a call within capacity still runs on Core ML");

    // Push streams: the offline TDT stream's 30 s chunk plus context is
    // longer than the 15 s sidecar, so under require every full window would
    // throw; stream_start refuses the stream before any audio.
    threw = false;
    try {
        (void) strict_default.stream_start({}, [](const parakeet::StreamingSegment &) {});
    } catch (const std::runtime_error & e) {
        threw = has(e.what(), "push-stream windows") && has(e.what(), "Core ML is required");
        std::printf("       strict stream_start threw: %s\n", e.what());
    }
    expect(threw, "strict-push-stream: stream_start refuses the stream, naming the window and capacity");
    bool started = false;
    try {
        started = engine.stream_start({}, [](const parakeet::StreamingSegment &) {}) != nullptr;
    } catch (const std::exception & e) {
        std::printf("       automatic stream_start threw: %s\n", e.what());
    }
    expect(started, "strict-push-stream: use = automatic still starts the same stream");
}

int main(int argc, char ** argv) {
    std::string which = "transcription", gguf, wav, diar_gguf, encoderless, eu_gguf, sortformer_gguf;
    std::string sub_gguf, sub_encoder, sub_decoder;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--case") which = argv[i + 1];
        else if (a == "--gguf") gguf = argv[i + 1];
        else if (a == "--wav") wav = argv[i + 1];
        else if (a == "--diar-gguf") diar_gguf = argv[i + 1];
        else if (a == "--encoderless-gguf") encoderless = argv[i + 1];
        else if (a == "--eu-gguf") eu_gguf = argv[i + 1];
        else if (a == "--sortformer-gguf") sortformer_gguf = argv[i + 1];
        else if (a == "--sub-gguf") sub_gguf = argv[i + 1];
        else if (a == "--sub-encoder") sub_encoder = argv[i + 1];
        else if (a == "--sub-decoder") sub_decoder = argv[i + 1];
        else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    const bool ok_args = (which == "transcription" && !gguf.empty() && !wav.empty()) ||
                         (which == "diarizer" && !diar_gguf.empty() && !wav.empty() && !gguf.empty()) ||
                         (which == "encoderless" && !encoderless.empty()) ||
                         (which == "eu-labels" && !eu_gguf.empty() && !wav.empty()) ||
                         (which == "fit-layouts" && !eu_gguf.empty() && !sortformer_gguf.empty()) ||
                         (which == "sortformer" && !sortformer_gguf.empty() && !wav.empty()) ||
                         (which == "pending-encoder" && !sub_gguf.empty() && !sub_encoder.empty() &&
                          !sub_decoder.empty() && !wav.empty());
    if (!ok_args) {
        std::fprintf(stderr, "usage: %s [--case transcription] --gguf TDT.gguf --wav W.wav\n"
                             "       %s --case diarizer --diar-gguf D.gguf --wav W.wav --gguf TDT.gguf\n"
                             "       %s --case encoderless --encoderless-gguf E.gguf\n"
                             "       %s --case eu-labels --eu-gguf EU.gguf --wav W.wav\n"
                             "       %s --case fit-layouts --eu-gguf EU.gguf --sortformer-gguf S.gguf\n"
                             "       %s --case sortformer --sortformer-gguf S.gguf --wav W.wav\n"
                             "       %s --case pending-encoder --sub-gguf T.gguf --sub-encoder E.mlmodelc "
                             "--sub-decoder D.mlmodelc --wav W.wav\n",
                     argv[0], argv[0], argv[0], argv[0], argv[0], argv[0], argv[0]);
        return 2;
    }
    try {
        if (which == "transcription") {
            parakeet::EngineResult reference;
            case_transcription(gguf, wav, reference);
            case_capacity_miss(gguf, wav);
            case_coreml_only_full(gguf, wav, reference);
            case_encoder_load_error(gguf);
            case_fit_stable(gguf);
            case_fit_degrade(gguf);
        } else if (which == "diarizer") {
            case_diarizer(diar_gguf, wav);
            case_diarizer_load_errors(diar_gguf);
            case_cli_diarization_status(gguf, diar_gguf, wav);
        } else if (which == "eu-labels") {
            case_eu_labels(eu_gguf, wav);
        } else if (which == "fit-layouts") {
            case_fit_layouts(eu_gguf, sortformer_gguf);
        } else if (which == "sortformer") {
            case_strict_too_wide(sortformer_gguf, wav);
        } else if (which == "pending-encoder") {
            case_pending_encoder(sub_gguf, sub_encoder, sub_decoder, wav);
        } else {
            case_encoderless(encoderless);
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: exception: %s\n", e.what());
        return 1;
    }
    std::printf("test-parakeet-coreml-runtime (%s): %d failure(s)\n", which.c_str(), g_failures);
    return g_failures == 0 ? 0 : 1;
}
