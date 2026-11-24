// Fixture test for the loader's Core ML-owned encoder mode: a GGUF without
// encoder weights (the TranscribePsy Core ML distribution) and the
// PARAKEET_COREML_ONLY switch for a full GGUF.
//
// Usage:
//   test-coreml-encoder-owner --coreml-gguf <encoder-less.gguf>
//                             [--full-gguf <q8.gguf> --wav <input.wav>]
//   test-coreml-encoder-owner --standard-tdt <q8.gguf> --wav <input.wav>
//   test-coreml-encoder-owner --stub-tdt-skip-load
//
// On every platform it asserts that the encoder-less GGUF is rejected with
// the documented code when no sidecar can take the encoder over (Core ML
// disabled through PARAKEET_COREML_DISABLE, which also covers builds without
// Core ML) and that the memory-fit loader rejects it the same way.
//
// On a Core ML build with the encoder sidecar next to the GGUF it further
// asserts that the GGUF loads with Core ML owning the encoder, that the
// encoder runs, that a request for the ggml encoder is refused instead of
// reading unallocated tensors, and that PARAKEET_COREML_ONLY=1 with a full
// GGUF skips the encoder weights (a smaller weights buffer) while the same
// GGUF without the switch loads them. Without the sidecar that part exits 3
// (skipped).
// Standard TDT sidecars without validated seams must retain native weights
// under PARAKEET_COREML_ONLY and reject encoder-less GGUFs even with a sidecar;
// with use = off or no sidecar next to it, a full GGUF under
// PARAKEET_COREML_ONLY keeps its ggml encoder too.
// `--stub-tdt-skip-load` needs no fixture: with a tensor-less TDT GGUF and a
// stub bundle that holds only metadata.json, the loader skips the synchronous
// sidecar load when the metadata shows an export that cannot own a TDT
// encoder, and still attempts it for Neural Engine metadata, for none, and
// for metadata that does not show the export.
//
// Exit 0 on success; non-zero on failure or invalid arguments.

#include "mel_preprocess.h"
#include "parakeet_coreml_path.h"
#include "parakeet_ctc.h"
#include "parakeet_log.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#ifdef _WIN32
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

using namespace parakeet;

namespace {

constexpr int kSkipExitCode = 3;
constexpr const char * kDisableEnv = "PARAKEET_COREML_DISABLE";
constexpr const char * kOnlyEnv    = "PARAKEET_COREML_ONLY";

int g_failures = 0;

void expect(bool cond, const std::string & what) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

void unset_env(const char * name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

int load(const std::string & gguf, ParakeetCtcModel & model) {
    // Ownership is synchronous; a full GGUF's background sidecar is awaited
    // so the checks below see it loaded.
    CoremlOptions coreml;
    coreml.wait = CoremlOptions::Wait::construct;
    return load_from_gguf(gguf, model, /*n_threads=*/0, /*n_gpu_layers=*/0, /*verbose=*/false,
                          "auto", coreml);
}

void check_rejected_without_sidecar(const std::string & coreml_gguf) {
    set_env(kDisableEnv, "1");
    ParakeetCtcModel model;
    expect(load(coreml_gguf, model) == kGgufLoadNoEncoderWeights,
           "an encoder-less GGUF must be rejected with kGgufLoadNoEncoderWeights without a sidecar");

    ParakeetCtcModel measured;
    GgufLoadMeasure measure;
    expect(load_from_gguf_metadata_only(coreml_gguf, measured, 0, 0, false, measure) ==
               kGgufLoadNoEncoderWeights,
           "the memory-fit loader must reject an encoder-less GGUF");
    unset_env(kDisableEnv);

    // use = off: the same rejection, whether or not a sidecar is present.
    CoremlOptions off;
    off.use = CoremlOptions::Use::off;
    ParakeetCtcModel off_model;
    expect(load_from_gguf(coreml_gguf, off_model, 0, 0, false, "auto", off) == kGgufLoadNoEncoderWeights,
           "an encoder-less GGUF with use = off must be rejected with kGgufLoadNoEncoderWeights");
}

#ifdef PARAKEET_USE_COREML

bool sidecar_present(const std::string & gguf) {
    return std::filesystem::is_directory(coreml_encoder_sidecar_path(gguf));
}

std::vector<float> mel_of(const ParakeetCtcModel & model, const std::string & wav, int & n_frames) {
    std::vector<float> samples;
    int sample_rate = 0;
    expect(load_wav_mono_f32(wav, samples, sample_rate) == 0, "could not load " + wav);
    std::vector<float> mel;
    expect(compute_log_mel(samples.data(), (int) samples.size(), model.mel_cfg, mel, n_frames) == 0,
           "mel failed");
    return mel;
}

void check_owned_encoder_runs(const std::string & coreml_gguf, const std::string & wav) {
    ParakeetCtcModel model;
    expect(load(coreml_gguf, model) == 0, "the encoder-less GGUF must load next to its sidecar");
    if (!model.impl) return;
    expect(model_encoder_owned_by_coreml(model), "Core ML must own the encoder");
    expect(model_encoder_on_coreml(model), "the encoder sidecar must be loaded");

    int n_frames = 0;
    const std::vector<float> mel = mel_of(model, wav, n_frames);

    EncoderOutputs out;
    expect(run_encoder(model, mel.data(), n_frames, model.mel_cfg.n_mels, out, parakeet::acquire_encoder_route(model),
                       /*max_layers=*/-1, /*capture_intermediates=*/false,
                       /*allow_coreml_padded=*/true) == 0 && out.n_enc_frames > 0 && out.used_coreml,
           "the owned encoder must run on Core ML");

    // capture_intermediates routes every model to the ggml encoder, whose
    // weights were never allocated: the request must fail, not read them.
    EncoderOutputs refused;
    expect(run_encoder(model, mel.data(), n_frames, model.mel_cfg.n_mels, refused, parakeet::acquire_encoder_route(model),
                       /*max_layers=*/-1, /*capture_intermediates=*/true) != 0,
           "a ggml encoder request must be refused when Core ML owns the encoder");
}

// A directory where the full GGUF carries the Core ML distribution's names, so
// its sidecar is found next to it.
std::filesystem::path stage_full_gguf_with_sidecar(const std::string & full_gguf,
                                                   const std::string & coreml_gguf) {
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / ("parakeet-coreml-owner-test-" + std::to_string(::getpid()));
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::filesystem::path staged = dir / "staged.gguf";
    std::filesystem::create_symlink(std::filesystem::absolute(full_gguf), staged);
    std::filesystem::create_directory_symlink(
        std::filesystem::absolute(coreml_encoder_sidecar_path(coreml_gguf)),
        coreml_encoder_sidecar_path(staged.string()));
    return staged;
}

void check_coreml_only_skips_encoder_weights(const std::string & full_gguf,
                                             const std::string & coreml_gguf) {
    const std::filesystem::path staged = stage_full_gguf_with_sidecar(full_gguf, coreml_gguf);

    ParakeetCtcModel full;
    expect(load(staged.string(), full) == 0, "the full GGUF must load without the switch");
    expect(full.impl && !model_encoder_owned_by_coreml(full),
           "without PARAKEET_COREML_ONLY the ggml encoder weights are loaded");

    set_env(kOnlyEnv, "1");
    ParakeetCtcModel owned;
    expect(load(staged.string(), owned) == 0, "the full GGUF must load under PARAKEET_COREML_ONLY");
    unset_env(kOnlyEnv);
    expect(owned.impl && model_encoder_owned_by_coreml(owned),
           "PARAKEET_COREML_ONLY must hand the encoder to Core ML");
    if (full.impl && owned.impl) {
        expect(model_weights_buffer_bytes(owned) < model_weights_buffer_bytes(full),
               "skipping the encoder weights must shrink the weights buffer");
    }
    std::filesystem::remove_all(staged.parent_path());
}

void check_standard_tdt_rejects_encoderless(const std::string & gguf) {
    const gguf_init_params params = {true, nullptr};
    std::unique_ptr<gguf_context, decltype(&gguf_free)> source(
        gguf_init_from_file(gguf.c_str(), params), gguf_free);
    expect(source != nullptr, "read standard TDT metadata");
    if (!source) return;
    std::unique_ptr<gguf_context, decltype(&gguf_free)> stripped(gguf_init_empty(), gguf_free);
    gguf_set_kv(stripped.get(), source.get());

    const auto dir = std::filesystem::temp_directory_path() / ("parakeet-standard-tdt-owner-test-" + std::to_string(::getpid()));
    std::filesystem::create_directories(dir);
    const auto path = dir / "encoderless.gguf";
    expect(gguf_write_to_file(stripped.get(), path.string().c_str(), false),
           "write encoder-less standard TDT fixture");
    std::filesystem::create_directory_symlink(
        std::filesystem::absolute(coreml_encoder_sidecar_path(gguf)),
        coreml_encoder_sidecar_path(path.string()));
    ParakeetCtcModel rejected;
    expect(load(path.string(), rejected) == kGgufLoadNoEncoderWeights,
           "standard TDT without encoder weights must reject an unvalidated sidecar");
    std::filesystem::remove_all(dir);
}

// PARAKEET_COREML_ONLY with a full GGUF keeps its ggml encoder whenever the
// takeover does not happen: Core ML is off (use = off), or no sidecar sits
// next to the GGUF, so the takeover attempt fails.
void check_coreml_only_full_keeps_encoder(const std::string & gguf) {
    set_env(kOnlyEnv, "1");
    {
        CoremlOptions off;
        off.use = CoremlOptions::Use::off;
        ParakeetCtcModel model;
        expect(load_from_gguf(gguf, model, 0, 0, false, "auto", off) == 0 &&
                   !model_encoder_owned_by_coreml(model),
               "PARAKEET_COREML_ONLY with use = off keeps the GGUF's ggml encoder");
    }
    {
        const auto dir = std::filesystem::temp_directory_path() / ("parakeet-coreml-only-no-sidecar-" + std::to_string(::getpid()));
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);
        const auto staged = dir / std::filesystem::path(gguf).filename();
        std::filesystem::create_symlink(std::filesystem::absolute(gguf), staged);
        ParakeetCtcModel model;
        expect(load(staged.string(), model) == 0 && !model_encoder_owned_by_coreml(model),
               "PARAKEET_COREML_ONLY with a failed takeover keeps the ggml encoder");
        std::filesystem::remove_all(dir);
    }
    unset_env(kOnlyEnv);
}

void check_standard_tdt_keeps_encoder_weights(const std::string & gguf,
                                              const std::string & wav) {
    unset_env(kOnlyEnv);
    ParakeetCtcModel full;
    expect(load(gguf, full) == 0, "standard TDT must load without the ownership switch");
    if (!full.impl) return;
    expect(full.model_type == ParakeetModelType::TDT && model_coreml_seam_context_frames(full) == 0,
           "ownership regression requires standard TDT without validated seams");
    expect(model_encoder_on_coreml(full), "standard TDT sidecar must be loaded");

    set_env(kOnlyEnv, "1");
    ParakeetCtcModel retained;
    expect(load(gguf, retained) == 0, "standard TDT must load with PARAKEET_COREML_ONLY=1");
    unset_env(kOnlyEnv);
    if (!retained.impl) return;
    expect(!model_encoder_owned_by_coreml(retained), "standard TDT must retain its native encoder");
    expect(model_encoder_on_coreml(retained), "retaining weights must keep the sidecar available");
    expect(model_weights_buffer_bytes(retained) == model_weights_buffer_bytes(full),
           "ownership switch must not reduce standard TDT encoder weights");

    int frames = 0;
    std::vector<float> mel = mel_of(retained, wav, frames);
    frames = std::max(frames, model_coreml_fixed_mel_frames(retained) + 1);
    mel.resize(static_cast<size_t>(frames) * retained.mel_cfg.n_mels, 0.0f);
    EncoderOutputs out;
    expect(run_encoder(retained, mel.data(), frames, retained.mel_cfg.n_mels, out,
                       parakeet::acquire_encoder_route(retained), -1, false, true) == 0 &&
               out.n_enc_frames > 0 && !out.used_coreml,
           "oversized TDT input must execute the retained native encoder");
    check_standard_tdt_rejects_encoderless(gguf);
    check_coreml_only_full_keeps_encoder(gguf);
}

void capture_log(ggml_log_level, const char * text, void * user) {
    static_cast<std::string *>(user)->append(text);
}

// Loads `gguf` with the loader's log captured into `log`.
int load_logged(const std::string & gguf, std::string & log) {
    log.clear();
    log_set_callback(&capture_log, &log);
    ParakeetCtcModel model;
    const int rc = load(gguf, model);
    log_set_callback(nullptr, nullptr);
    return rc;
}

void write_text(const std::filesystem::path & path, const std::string & text) {
    std::ofstream(path) << text;
}

// A standard TDT sidecar owns the encoder only when it is the fixed-window
// Neural Engine export (the one with a validated seam context). When the
// bundle's metadata shows another export, the synchronous load before weight
// allocation could only be thrown away, so the loader skips it; with Neural
// Engine metadata, or none, it still loads the bundle to find out (here the
// stub bundle fails to load). The encoder-less GGUF is rejected either way.
void check_stub_tdt_skips_throwaway_load() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / ("parakeet-stub-tdt-skip-load-" + std::to_string(::getpid()));
    fs::remove_all(dir);
    fs::create_directories(dir);
    const fs::path gguf = dir / "stub.gguf";
    std::unique_ptr<gguf_context, decltype(&gguf_free)> stub(gguf_init_empty(), gguf_free);
    gguf_set_val_str(stub.get(), "general.architecture", "parakeet-ctc");
    gguf_set_val_str(stub.get(), "parakeet.model.type", "tdt");
    expect(gguf_write_to_file(stub.get(), gguf.string().c_str(), false),
           "write the tensor-less TDT GGUF");
    const fs::path bundle = coreml_encoder_sidecar_path(gguf.string());
    fs::create_directories(bundle);
    const char * kFailedLoad = "failed to load Core ML encoder";

    std::string log;
    write_text(bundle / "metadata.json", R"([{"inputSchema":[{"name":"mel"}]}])");
    expect(load_logged(gguf.string(), log) == kGgufLoadNoEncoderWeights,
           "single-input TDT metadata: the encoder-less GGUF is rejected");
    expect(log.find("no validated seam policy") != std::string::npos &&
               log.find(kFailedLoad) == std::string::npos,
           "single-input TDT metadata: the synchronous sidecar load is skipped");

    write_text(bundle / "metadata.json",
               R"([{"inputSchema":[{"name":"mel"},{"name":"pad_mask"},{"name":"key_bias"}]}])");
    expect(load_logged(gguf.string(), log) == kGgufLoadNoEncoderWeights &&
               log.find(kFailedLoad) != std::string::npos,
           "Neural Engine metadata: the loader still loads the sidecar to read its seam context");

    fs::remove(bundle / "metadata.json");
    expect(load_logged(gguf.string(), log) == kGgufLoadNoEncoderWeights &&
               log.find(kFailedLoad) != std::string::npos,
           "no metadata: the loader still loads the sidecar to find out");

    // The Neural Engine inputs named with JSON escapes: the decoded names are
    // what counts, so the loader still loads the bundle to read its seams.
    write_text(bundle / "metadata.json",
               R"([{"inputSchema":[{"name":"mel"},{"name":"pad\u005fmask"},{"name":"key\u005fbias"}]}])");
    expect(load_logged(gguf.string(), log) == kGgufLoadNoEncoderWeights &&
               log.find(kFailedLoad) != std::string::npos &&
               log.find("no validated seam policy") == std::string::npos,
           "Neural Engine inputs in escaped metadata: the loader still loads the sidecar");
    // One Neural Engine name alone does not make the export.
    write_text(bundle / "metadata.json",
               R"([{"inputSchema":[{"name":"mel"},{"name":"pad_mask"}],"shortDescription":"key_bias"}])");
    expect(load_logged(gguf.string(), log) == kGgufLoadNoEncoderWeights &&
               log.find("no validated seam policy") != std::string::npos,
           "a schema with only one Neural Engine input is another export: the load is skipped");

    // Metadata that does not show the export (unparsable, or no input names)
    // is inconclusive: only the model's own description can rule ownership
    // out, so the bundle is loaded, as with no metadata.
    for (const char * inconclusive : {"not json", "{}", R"([{"inputSchema":[]}])",
                                      R"([{"shortDescription":"encoder"}])"}) {
        write_text(bundle / "metadata.json", inconclusive);
        expect(load_logged(gguf.string(), log) == kGgufLoadNoEncoderWeights &&
                   log.find(kFailedLoad) != std::string::npos &&
                   log.find("no validated seam policy") == std::string::npos,
               std::string("inconclusive metadata '") + inconclusive +
                   "': the loader still loads the sidecar to find out");
    }
    fs::remove_all(dir);
}

#endif  // PARAKEET_USE_COREML

}  // namespace

int main(int argc, char ** argv) {
    std::string coreml_gguf, full_gguf, standard_tdt, wav;
    if (argc == 2 && std::strcmp(argv[1], "--stub-tdt-skip-load") == 0) {
#ifdef PARAKEET_USE_COREML
        check_stub_tdt_skips_throwaway_load();
        if (g_failures == 0) std::printf("test-coreml-encoder-owner: stub TDT checks OK\n");
        return g_failures == 0 ? 0 : 1;
#else
        return kSkipExitCode;
#endif
    }
    for (int i = 1; i + 1 < argc; i += 2) {
        if (std::strcmp(argv[i], "--coreml-gguf") == 0) coreml_gguf = argv[i + 1];
        else if (std::strcmp(argv[i], "--full-gguf") == 0) full_gguf = argv[i + 1];
        else if (std::strcmp(argv[i], "--wav") == 0) wav = argv[i + 1];
        else if (std::strcmp(argv[i], "--standard-tdt") == 0) standard_tdt = argv[i + 1];
    }
    if (!standard_tdt.empty() && !wav.empty()) {
#ifdef PARAKEET_USE_COREML
        check_standard_tdt_keeps_encoder_weights(standard_tdt, wav);
        return g_failures == 0 ? 0 : 1;
#else
        return kSkipExitCode;
#endif
    }
    if (coreml_gguf.empty()) {
        std::fprintf(stderr, "usage: test-coreml-encoder-owner --coreml-gguf <gguf> "
                             "[--full-gguf <gguf> --wav <wav>]\n");
        return 2;
    }

    check_rejected_without_sidecar(coreml_gguf);

    const bool want_owner_checks = !full_gguf.empty() && !wav.empty();
#ifdef PARAKEET_USE_COREML
    if (want_owner_checks) {
        if (!sidecar_present(coreml_gguf)) {
            std::printf("SKIP: no Core ML encoder sidecar next to %s\n", coreml_gguf.c_str());
            return g_failures == 0 ? kSkipExitCode : 1;
        }
        check_owned_encoder_runs(coreml_gguf, wav);
        check_coreml_only_skips_encoder_weights(full_gguf, coreml_gguf);
    }
#else
    if (want_owner_checks) {
        std::printf("note: Core ML is not compiled in; only the rejection checks ran\n");
    }
#endif

    if (g_failures == 0) {
        std::printf("test-coreml-encoder-owner: OK\n");
        return 0;
    }
    std::fprintf(stderr, "test-coreml-encoder-owner: %d failure(s)\n", g_failures);
    return 1;
}
