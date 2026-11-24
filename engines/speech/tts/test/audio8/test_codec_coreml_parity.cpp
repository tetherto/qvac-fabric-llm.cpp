// Core ML codec sidecar: parity against the ggml synthesis through the
// production dispatch, the stream that overlaps it with generation, the public
// load-status and per-call backend reports, and the fallbacks (no sidecar,
// invalid sidecar, sidecar that loads but cannot serve). Skips (77) without TTS_CPP_COREML, a staged
// AUDIO8_COREML_TEST_MODELS_DIR, or a compiled sidecar next to the decoder
// GGUF there. Optional: AUDIO8_COREML_FALLBACK_MODELS_DIR (a sidecar whose
// window cannot carry the causal context) and AUDIO8_COREML_TEST_LM_GGUF (a
// real language model, for a synthesis through the public Engine).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
constexpr int SKIP_EXIT_CODE = 77;
}

#ifndef TTS_CPP_USE_COREML

int main() {
    std::fprintf(stderr, "[coreml-parity] built without TTS_CPP_COREML; skipping\n");
    return SKIP_EXIT_CODE;
}

#else

#include "audio8/coreml/codec-synth.h"
#include "audio8/coreml_path.h"
#include "audio8/coreml_windows.h"
#include "audio8/internal.h"
#include "audio8/window_stream.h"
#include "test_env_portable.h"
#include "tiny_lm.h"
#include "tts-cpp/audio8/engine.h"

#include <filesystem>
#include <fstream>
#include <memory>

using namespace tts_cpp::audio8::detail;
namespace fs = std::filesystem;

namespace {

constexpr double MIN_COSINE   = 0.999;
// Streamed and batch synthesis both run on the sidecar; they differ only in
// the rounding of post passes over different spans.
constexpr double STREAM_MIN_COSINE = 0.99999;
constexpr int    SHORT_FRAMES = 10;
constexpr int    LONG_FRAMES  = 200;
constexpr int    N_THREADS    = 4;

int g_failures = 0;

void fail(const std::string & what) {
    std::fprintf(stderr, "[coreml-parity] FAIL: %s\n", what.c_str());
    ++g_failures;
}

void expect(bool ok, const std::string & what) {
    if (!ok) fail(what);
}

struct scoped_env {
    const char * name;
    scoped_env(const char * n, const char * value) : name(n) { setenv(name, value, 1); }
    ~scoped_env() { unsetenv(name); }
};

std::string find_decoder_gguf(const std::string & dir) {
    const char * candidates[] = {"audio8-codec-decoder-f32.gguf", "audio8-codec-decoder-f16.gguf",
                                 "audio8-codec-decoder-q8_0.gguf", "audio8-codec-decoder.gguf"};
    for (const char * name : candidates) {
        const std::string path = dir + "/" + name;
        if (fs::exists(path)) return path;
    }
    return {};
}

std::vector<int32_t> make_codes(const codec_hparams & hp, int n_frames, uint32_t seed) {
    std::vector<int32_t> codes(static_cast<size_t>(hp.num_codebooks) * n_frames);
    uint32_t state = seed;
    for (int book = 0; book < hp.num_codebooks; ++book) {
        const int limit = book == 0 ? hp.semantic_codebook_size : hp.residual_codebook_size;
        for (int frame = 0; frame < n_frames; ++frame) {
            state = state * 1664525u + 1013904223u;
            codes[static_cast<size_t>(book) * n_frames + frame] =
                static_cast<int32_t>((state >> 8) % static_cast<uint32_t>(limit));
        }
    }
    return codes;
}

double cosine(const std::vector<float> & a, const std::vector<float> & b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += (double) a[i] * (double) b[i];
        na  += (double) a[i] * (double) a[i];
        nb  += (double) b[i] * (double) b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
}

double max_abs_err(const std::vector<float> & a, const std::vector<float> & b) {
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, (double) std::fabs(a[i] - b[i]));
    return worst;
}

struct owned_codec {
    codec_model model;
    bool loaded = false;
    ~owned_codec() { free_codec(model); }
};

bool load(const std::string & gguf, owned_codec & codec) {
    std::string error;
    codec.loaded = load_codec(gguf, /*n_gpu_layers=*/0, /*backend=*/"", codec.model, &error);
    if (!codec.loaded) fail("load_codec(" + gguf + "): " + error);
    return codec.loaded;
}

bool decode(codec_model & model, const std::vector<int32_t> & codes, int n_frames,
            const cancel_hook & cancel, std::vector<float> & pcm, decode_timing & timing,
            std::string & error) {
    return decode_codes(model, codes.data(), n_frames, N_THREADS, cancel, pcm, &error, nullptr,
                        &timing);
}

bool check_parity(const char * tag, codec_model & ggml, codec_model & coreml, int n_frames) {
    const std::vector<int32_t> codes = make_codes(ggml.hp, n_frames, 0x2468ace1u);
    std::vector<float> pcm_ggml, pcm_coreml;
    decode_timing t_ggml, t_coreml;
    std::string error;
    {
        scoped_env disabled("AUDIO8_COREML_DISABLE", "1");
        if (!decode(ggml, codes, n_frames, nullptr, pcm_ggml, t_ggml, error)) {
            fail(std::string(tag) + ": ggml decode: " + error);
            return false;
        }
    }
    {
        scoped_env strict("AUDIO8_COREML_STRICT", "1");
        if (!decode(coreml, codes, n_frames, nullptr, pcm_coreml, t_coreml, error)) {
            fail(std::string(tag) + ": Core ML decode: " + error);
            return false;
        }
    }
    expect(t_ggml.synthesis_backend == "ggml" && t_coreml.synthesis_backend.rfind("coreml", 0) == 0,
           std::string(tag) + ": backends were '" + t_ggml.synthesis_backend + "' and '" +
               t_coreml.synthesis_backend + "'");
    const size_t want = static_cast<size_t>(n_frames) * ggml.hp.frame_size;
    if (pcm_ggml.size() != want || pcm_coreml.size() != want) {
        fail(std::string(tag) + ": sizes ggml=" + std::to_string(pcm_ggml.size()) +
             " coreml=" + std::to_string(pcm_coreml.size()) + " want=" + std::to_string(want));
        return false;
    }
    const double cos = cosine(pcm_ggml, pcm_coreml);
    std::fprintf(stderr,
                 "[coreml-parity] %s: frames=%d window=%d samples=%zu cosine=%.7f max_abs=%.3e "
                 "(min cosine %.4f, %s)\n",
                 tag, n_frames, t_coreml.block_frames, want, cos, max_abs_err(pcm_ggml, pcm_coreml),
                 MIN_COSINE, t_coreml.synthesis_backend.c_str());
    expect(cos >= MIN_COSINE, std::string(tag) + ": cosine below the gate");
    return cos >= MIN_COSINE;
}

void check_strict_refusal(const std::string & gguf) {
    scoped_env disabled("AUDIO8_COREML_DISABLE", "1");
    scoped_env strict("AUDIO8_COREML_STRICT", "1");
    owned_codec codec;
    if (!load(gguf, codec)) return;
    expect(!codec.model.synthesis_on_coreml, "a disabled sidecar was still attached");
    const std::vector<int32_t> codes = make_codes(codec.model.hp, SHORT_FRAMES, 7u);
    std::vector<float> pcm;
    decode_timing timing;
    std::string error;
    expect(!decode(codec.model, codes, SHORT_FRAMES, nullptr, pcm, timing, error),
           "strict decode without a sidecar returned audio");
}

void check_cancellation(codec_model & coreml) {
    const std::vector<int32_t> codes = make_codes(coreml.hp, LONG_FRAMES, 99u);
    int asked = 0;
    const cancel_hook after_one = [&asked] { return asked++ >= 1; };
    std::vector<float> pcm;
    decode_timing timing;
    std::string error;
    scoped_env strict("AUDIO8_COREML_STRICT", "1");
    const bool ran = decode(coreml, codes, LONG_FRAMES, after_one, pcm, timing, error);
    expect(!ran && error == CANCELLED, "cancel " + std::string(ran ? "was ignored" : "stopped with '" + error + "'"));
    const int window = static_cast<int>(audio8_coreml_codec_window_frames(coreml.coreml));
    const size_t one_window = static_cast<size_t>(std::min(window, LONG_FRAMES)) * coreml.hp.frame_size;
    expect(pcm.size() == one_window, "cancel kept " + std::to_string(pcm.size()) +
                                         " samples, expected one window of " + std::to_string(one_window));
    if (g_failures == 0) std::fprintf(stderr, "[coreml-parity] cancel: one window kept (%zu samples)\n", pcm.size());
}

void place_frame(const std::vector<int32_t> & rows, int books, int n_frames, int frame,
                 std::vector<int32_t> & frames) {
    for (int book = 0; book < books; ++book) {
        frames[static_cast<size_t>(frame) * books + book] =
            rows[static_cast<size_t>(book) * n_frames + frame];
    }
}

// Codebook-major codes back to the frame-major order the language model emits.
std::vector<int32_t> frame_major(const std::vector<int32_t> & rows, int books, int n_frames) {
    std::vector<int32_t> frames(rows.size());
    for (int frame = 0; frame < n_frames; ++frame) place_frame(rows, books, n_frames, frame, frames);
    return frames;
}

bool stream_frames(window_stream & stream, int n_frames, std::string & error) {
    for (int frame = 1; frame <= n_frames; ++frame) {
        if (!stream.advance(frame, &error)) return false;
    }
    return true;
}

int leading_windows(codec_model & model, int n_frames) {
    const int window = static_cast<int>(audio8_coreml_codec_window_frames(model.coreml));
    return static_cast<int>(
               plan_coreml_windows(n_frames, window, synthesis_context_frames(model)).size()) -
           1;
}

// The engine's path: frames offered one at a time while they are generated,
// the rest after the last one. It has to stream every window that becomes
// final early and stitch the waveform the batch windows produce.
void check_streamed(codec_model & coreml, int n_frames) {
    const std::vector<int32_t> rows = make_codes(coreml.hp, n_frames, 0x13579bdfu);
    const std::vector<int32_t> frames = frame_major(rows, coreml.hp.num_codebooks, n_frames);
    std::vector<float> batch, streamed;
    decode_timing t_batch, t_stream;
    std::string error;
    expect(decode(coreml, rows, n_frames, nullptr, batch, t_batch, error),
           "batch decode failed: " + error);
    const std::unique_ptr<window_stream> stream =
        open_synthesis_stream(coreml, frames, N_THREADS, nullptr);
    if (!stream) {
        fail("no synthesis stream opened although a sidecar is attached");
        return;
    }
    expect(stream_frames(*stream, n_frames, error), "streamed advance failed: " + error);
    expect(finish_synthesis_stream(coreml, *stream, rows.data(), n_frames, N_THREADS, nullptr,
                                   streamed, &error, &t_stream),
           "streamed finish failed: " + error);
    expect(t_stream.streamed_windows == leading_windows(coreml, n_frames),
           "streamed " + std::to_string(t_stream.streamed_windows) + " windows, expected " +
               std::to_string(leading_windows(coreml, n_frames)));
    expect(t_stream.synthesis_backend.rfind("coreml", 0) == 0,
           "streamed synthesis ran on " + t_stream.synthesis_backend);
    expect(streamed.size() == batch.size(), "streamed and batch lengths differ");
    if (streamed.size() != batch.size()) return;
    const double cos = cosine(streamed, batch);
    std::fprintf(stderr, "[coreml-parity] streamed: frames=%d windows during generation=%d "
                         "cosine=%.7f max_abs=%.3e against batch (min cosine %.5f)\n",
                 n_frames, t_stream.streamed_windows, cos, max_abs_err(streamed, batch),
                 STREAM_MIN_COSINE);
    expect(cos >= STREAM_MIN_COSINE, "streamed: cosine below the gate");
}

void check_streamed_cancel(codec_model & coreml) {
    const std::vector<int32_t> rows = make_codes(coreml.hp, LONG_FRAMES, 11u);
    const std::vector<int32_t> frames = frame_major(rows, coreml.hp.num_codebooks, LONG_FRAMES);
    const std::unique_ptr<window_stream> stream =
        open_synthesis_stream(coreml, frames, N_THREADS, [] { return true; });
    if (!stream) {
        fail("no synthesis stream opened for the cancel check");
        return;
    }
    std::vector<float> pcm;
    std::string error;
    stream_frames(*stream, LONG_FRAMES, error);
    const bool ran = finish_synthesis_stream(coreml, *stream, rows.data(), LONG_FRAMES,
                                             N_THREADS, nullptr, pcm, &error, nullptr);
    expect(!ran && error == CANCELLED,
           "streamed cancel " + std::string(ran ? "was ignored" : "stopped with '" + error + "'"));
}

// A language model the public Engine accepts next to this decoder: the two
// GGUFs must agree on the codebook count and the semantic codebook size.
std::string write_matching_tiny_lm(const codec_hparams & hp) {
    audio8_test::tiny_lm p;
    p.num_codebooks = hp.num_codebooks;
    p.codebook_size = hp.semantic_codebook_size;
    return audio8_test::write_tiny_lm_gguf(
        p, test_tmpdir() + "/test-audio8-coreml-tiny-lm-" + test_process_tag() + ".gguf");
}

tts_cpp::audio8::EngineOptions engine_options(const std::string & lm, const std::string & decoder) {
    tts_cpp::audio8::EngineOptions opts;
    opts.lm_gguf_path            = lm;
    opts.codec_decoder_gguf_path = decoder;
    opts.n_threads               = N_THREADS;
    opts.greedy                  = true;
    opts.max_frames              = SHORT_FRAMES;
    return opts;
}

void check_public_load_status(const std::string & lm, const std::string & decoder, bool want_loaded,
                              const char * tag) {
    try {
        tts_cpp::audio8::Engine engine(engine_options(lm, decoder));
        expect(engine.codec_on_coreml() == want_loaded,
               std::string(tag) + ": Engine::codec_on_coreml() = " +
                   (engine.codec_on_coreml() ? "true" : "false") + ", expected " +
                   (want_loaded ? "true" : "false"));
        expect(engine.backend_name() == "CPU",
               std::string(tag) + ": backend_name() should stay the ggml backend, got " + engine.backend_name());
    } catch (const std::exception & e) {
        fail(std::string(tag) + ": Engine construction threw: " + e.what());
    }
}

void check_public_synthesis_backend(const std::string & lm, const std::string & decoder,
                                    const char * want_prefix, bool want_attached_after,
                                    const char * tag) {
    try {
        tts_cpp::audio8::Engine engine(engine_options(lm, decoder));
        expect(engine.codec_on_coreml(), std::string(tag) + ": sidecar not attached at load");
        const tts_cpp::audio8::SynthesisResult result = engine.synthesize("Hello from the parity test.");
        expect(result.codec_synthesis_backend.rfind(want_prefix, 0) == 0,
               std::string(tag) + ": codec_synthesis_backend = '" + result.codec_synthesis_backend +
                   "', expected prefix '" + want_prefix + "'");
        expect(engine.codec_on_coreml() == want_attached_after,
               std::string(tag) + ": codec_on_coreml() after synthesis = " +
                   (engine.codec_on_coreml() ? "true" : "false"));
        std::fprintf(stderr, "[coreml-parity] %s: %d frames, codec synthesis on %s (sidecar attached after: %s)\n",
                     tag, result.frames, result.codec_synthesis_backend.c_str(),
                     engine.codec_on_coreml() ? "yes" : "no");
    } catch (const std::exception & e) {
        fail(std::string(tag) + ": synthesis threw: " + e.what());
    }
}

// The staged decoder next to a sidecar directory that is not a model.
std::string stage_invalid_sidecar(const std::string & gguf) {
    const fs::path dir = fs::path(test_tmpdir()) / ("test-audio8-coreml-invalid-" + test_process_tag());
    fs::remove_all(dir);
    fs::create_directories(dir);
    const fs::path link = dir / fs::path(gguf).filename();
    fs::create_symlink(fs::absolute(gguf), link);
    const fs::path bogus = fs::path(coreml_codec_sidecar_path(link.string()));
    fs::create_directories(bogus);
    std::ofstream(bogus / "coremldata.bin") << "not a model";
    return link.string();
}

void check_invalid_sidecar(const std::string & gguf, const std::string & lm) {
    const std::string staged = stage_invalid_sidecar(gguf);
    owned_codec codec;
    if (!load(staged, codec)) return;
    expect(codec.model.coreml == nullptr && !codec.model.synthesis_on_coreml,
           "an invalid sidecar was attached");
    const std::vector<int32_t> codes = make_codes(codec.model.hp, SHORT_FRAMES, 3u);
    std::vector<float> pcm;
    decode_timing timing;
    std::string error;
    expect(decode(codec.model, codes, SHORT_FRAMES, nullptr, pcm, timing, error),
           "decode next to an invalid sidecar failed: " + error);
    expect(timing.synthesis_backend == "ggml", "invalid sidecar: synthesis ran on " + timing.synthesis_backend);
    check_public_load_status(lm, staged, /*want_loaded=*/false, "invalid sidecar");
    fs::remove_all(fs::path(staged).parent_path());
}

// A sidecar that loads but cannot serve any utterance (window <= causal
// context): fails under STRICT, otherwise falls back to ggml bit-exactly and is
// retired, so the next call goes straight to ggml with no sidecar attached.
void check_unservable_sidecar(const std::string & dir, const std::string & lm) {
    const std::string gguf = find_decoder_gguf(dir);
    if (gguf.empty() || !fs::exists(coreml_codec_sidecar_path(gguf))) {
        fail("AUDIO8_COREML_FALLBACK_MODELS_DIR has no decoder GGUF + sidecar");
        return;
    }
    owned_codec codec;
    if (!load(gguf, codec)) return;
    expect(codec.model.coreml != nullptr && codec.model.synthesis_on_coreml,
           "the fallback fixture's sidecar did not load");
    const int window = static_cast<int>(audio8_coreml_codec_window_frames(codec.model.coreml));
    const int context = synthesis_context_frames(codec.model);
    expect(window <= context, "the fallback fixture's window (" + std::to_string(window) +
                                  ") carries the context (" + std::to_string(context) + "); it would serve");

    const std::vector<int32_t> codes = make_codes(codec.model.hp, SHORT_FRAMES, 5u);
    const std::vector<int32_t> no_frames;
    expect(!open_synthesis_stream(codec.model, no_frames, N_THREADS, nullptr),
           "a sidecar that cannot carry the context opened a synthesis stream");
    std::string error;
    {
        owned_codec strict_codec;
        if (!load(gguf, strict_codec)) return;
        scoped_env strict("AUDIO8_COREML_STRICT", "1");
        std::vector<float> pcm;
        decode_timing timing;
        expect(!decode(strict_codec.model, codes, SHORT_FRAMES, nullptr, pcm, timing, error),
               "strict decode with an unservable sidecar returned audio");
        expect(error.find("unavailable") != std::string::npos,
               "strict failure did not name the unavailable sidecar: " + error);
    }
    std::vector<float> pcm_fallback, pcm_ggml, pcm_retired;
    decode_timing t_fallback, t_ggml, t_retired;
    expect(decode(codec.model, codes, SHORT_FRAMES, nullptr, pcm_fallback, t_fallback, error),
           "fallback decode failed: " + error);
    expect(t_fallback.synthesis_backend == "ggml",
           "fallback synthesis ran on " + t_fallback.synthesis_backend);
    expect(codec.model.coreml == nullptr && !codec.model.synthesis_on_coreml,
           "the failed sidecar was not retired after the fallback");
    expect(decode(codec.model, codes, SHORT_FRAMES, nullptr, pcm_retired, t_retired, error) &&
               t_retired.synthesis_backend == "ggml",
           "decode after retirement did not go straight to ggml: " + error);
    {
        scoped_env disabled("AUDIO8_COREML_DISABLE", "1");
        expect(decode(codec.model, codes, SHORT_FRAMES, nullptr, pcm_ggml, t_ggml, error),
               "reference decode failed: " + error);
    }
    expect(pcm_fallback == pcm_ggml && pcm_retired == pcm_ggml,
           "fallback outputs are not bit-identical to the ggml decode");
    check_public_load_status(lm, gguf, /*want_loaded=*/true, "unservable sidecar");
    if (const char * real_lm = std::getenv("AUDIO8_COREML_TEST_LM_GGUF"); real_lm && *real_lm) {
        check_public_synthesis_backend(real_lm, gguf, "ggml", /*want_attached_after=*/false,
                                       "unservable sidecar, public synthesis");
    }
    std::fprintf(stderr, "[coreml-parity] unservable sidecar (window %d <= context %d): loaded, fell back to ggml, retired\n",
                 window, context);
}

}  // namespace

int main() {
    const char * models_dir = std::getenv("AUDIO8_COREML_TEST_MODELS_DIR");
    if (models_dir == nullptr || *models_dir == '\0') {
        std::fprintf(stderr, "[coreml-parity] AUDIO8_COREML_TEST_MODELS_DIR not set; skipping\n");
        return SKIP_EXIT_CODE;
    }
    const std::string gguf = find_decoder_gguf(models_dir);
    if (gguf.empty()) {
        std::fprintf(stderr, "[coreml-parity] no codec decoder GGUF under %s; skipping\n", models_dir);
        return SKIP_EXIT_CODE;
    }
    const std::string sidecar = coreml_codec_sidecar_path(gguf);
    if (!fs::exists(sidecar)) {
        std::fprintf(stderr, "[coreml-parity] no sidecar at %s; skipping\n", sidecar.c_str());
        return SKIP_EXIT_CODE;
    }

    check_strict_refusal(gguf);

    owned_codec ggml;
    {
        scoped_env disabled("AUDIO8_COREML_DISABLE", "1");
        if (!load(gguf, ggml)) return 1;
    }
    owned_codec coreml;
    {
        scoped_env strict("AUDIO8_COREML_STRICT", "1");
        if (!load(gguf, coreml)) return 1;
    }
    if (!coreml.model.synthesis_on_coreml || !coreml.model.coreml) {
        fail("sidecar at " + sidecar + " did not initialise");
        return 1;
    }
    const int window = static_cast<int>(audio8_coreml_codec_window_frames(coreml.model.coreml));
    const int context = synthesis_context_frames(coreml.model);
    std::fprintf(stderr, "[coreml-parity] sidecar %s: window %d frames, causal context %d frames\n",
                 sidecar.c_str(), window, context);
    if (context >= window) {
        fail("the window cannot carry the causal context");
        return 1;
    }

    check_parity("short", ggml.model, coreml.model, SHORT_FRAMES);
    check_parity("long", ggml.model, coreml.model, LONG_FRAMES);
    check_cancellation(coreml.model);
    check_streamed(coreml.model, SHORT_FRAMES);
    check_streamed(coreml.model, LONG_FRAMES);
    check_streamed_cancel(coreml.model);

    const std::string tiny_lm = write_matching_tiny_lm(coreml.model.hp);
    check_public_load_status(tiny_lm, gguf, /*want_loaded=*/true, "sidecar present");
    {
        scoped_env disabled("AUDIO8_COREML_DISABLE", "1");
        check_public_load_status(tiny_lm, gguf, /*want_loaded=*/false, "sidecar disabled");
    }
    check_invalid_sidecar(gguf, tiny_lm);
    if (const char * fallback_dir = std::getenv("AUDIO8_COREML_FALLBACK_MODELS_DIR"); fallback_dir && *fallback_dir) {
        check_unservable_sidecar(fallback_dir, tiny_lm);
    } else {
        std::fprintf(stderr, "[coreml-parity] AUDIO8_COREML_FALLBACK_MODELS_DIR not set; unservable-sidecar checks not run\n");
    }
    if (const char * real_lm = std::getenv("AUDIO8_COREML_TEST_LM_GGUF"); real_lm && *real_lm) {
        check_public_synthesis_backend(real_lm, gguf, "coreml", /*want_attached_after=*/true,
                                       "sidecar present, public synthesis");
    }
    fs::remove(tiny_lm);

    std::printf("\n%s\n", g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}

#endif  // TTS_CPP_USE_COREML
