// QVAC (see PATCHES.md): whisper_fit_params must project a model from the
// registry's weightless description exactly as it projects the model itself.
// The committed *.fit.gguf fixtures are the descriptions qvac's registry-server
// (packages/registry-server/lib/fit-description) writes for the committed
// for-tests models; for-tests-ggml-bci.bin is a header-only BCI model from the
// same package's test fixtures. A malformed VAD description is derived from the
// committed one at run time.
//
// Usage: test-whisper-fit-description <vad.bin> <vad.fit.gguf>
//            <model.bin> <model.fit.gguf> [<model.bin> <model.fit.gguf> ...] [gpu]
// The "gpu" form exits 77 (ctest SKIP) when no GPU/IGPU device exists.

#include "whisper.h"

#include "ggml-backend.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

constexpr int    EXIT_SKIP     = 77;
constexpr int    EXIT_USAGE    = 2;
constexpr float  AUDIO_SECONDS = 30.0f;
constexpr int    DECODERS      = 5;
constexpr size_t PAIR_ARGS     = 2;
constexpr size_t MIN_PAIRS     = 2;
constexpr const char * GPU_ARG = "gpu";

constexpr size_t TRUNCATED_VAD_ENCODER_LAYERS = 3;
constexpr const char * TRUNCATED_VAD_DESCRIPTION = "test-whisper-fit-description-three-vad-encoders.fit.gguf";
constexpr const char * VAD_ENCODER_KEYS[] = {
    "whisper_vad.encoder_in_channels", "whisper_vad.encoder_out_channels", "whisper_vad.encoder_kernel_size",
};

int g_failures = 0;

void fail(const std::string & what) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
}

void expect(bool cond, const std::string & what) {
    if (!cond) fail(what);
}

void expect_eq(uint64_t artifact, uint64_t description, const std::string & what) {
    if (artifact != description) {
        fail(what + ": artifact " + std::to_string(artifact) + " != description " + std::to_string(description));
    }
}

void log_callback_null(ggml_log_level, const char *, void *) {}

struct fit_pair {
    std::string artifact;
    std::string description;
};

bool has_gpu_device() {
    ggml_backend_load_all();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        const enum ggml_backend_dev_type t = ggml_backend_dev_type(ggml_backend_dev_get(i));
        if (t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            return true;
        }
    }
    return false;
}

whisper_fit_result project(const std::string & model, const std::string & vad, bool use_gpu) {
    whisper_fit_options opts = whisper_fit_default_options();
    opts.model_path     = model.c_str();
    opts.vad_model_path = vad.empty() ? nullptr : vad.c_str();
    opts.use_gpu        = use_gpu;
    opts.n_decoders     = DECODERS;
    opts.audio_seconds  = AUDIO_SECONDS;

    whisper_fit_result fit;
    whisper_fit_params(&opts, &fit);
    return fit;
}

void expect_same_projection(const whisper_fit_result & artifact, const whisper_fit_result & description,
                            const std::string & label) {
    expect(artifact.status != WHISPER_FIT_ERROR, label + ": artifact projection failed (" + artifact.reason + ")");
    expect(description.status != WHISPER_FIT_ERROR, label + ": description projection failed (" + description.reason + ")");
    expect(std::strcmp(artifact.model_type, description.model_type) == 0, label + ": model_type differs");
    expect(std::strcmp(artifact.device_name, description.device_name) == 0, label + ": device_name differs");
    expect(artifact.device_is_cpu == description.device_is_cpu, label + ": device_is_cpu differs");
    expect_eq(artifact.device_total_bytes,         description.device_total_bytes,         label + ": device_total_bytes");
    expect_eq(artifact.device.weights_bytes,       description.device.weights_bytes,       label + ": weights_bytes");
    expect_eq(artifact.device.kv_bytes,            description.device.kv_bytes,            label + ": kv_bytes");
    expect_eq(artifact.device.compute_bytes,       description.device.compute_bytes,       label + ": compute_bytes");
    expect_eq(artifact.device.vad_bytes,           description.device.vad_bytes,           label + ": vad_bytes");
    expect_eq(artifact.device.total_bytes,         description.device.total_bytes,         label + ": total_bytes");
    expect_eq(artifact.device.host_overflow_bytes, description.device.host_overflow_bytes, label + ": host_overflow_bytes");
    expect_eq(artifact.host_bytes,                 description.host_bytes,                 label + ": host_bytes");
}

void check_model(const fit_pair & model, bool use_gpu) {
    expect_same_projection(project(model.artifact, "", use_gpu), project(model.description, "", use_gpu),
                           model.artifact);
}

void check_models(const std::vector<fit_pair> & models, bool use_gpu) {
    for (const fit_pair & model : models) {
        check_model(model, use_gpu);
    }
}

void check_vad(const fit_pair & model, const fit_pair & vad, bool use_gpu) {
    const whisper_fit_result artifact = project(model.artifact, vad.artifact, use_gpu);
    expect(artifact.device.vad_bytes > 0, "the VAD model adds no device bytes");
    expect_same_projection(artifact, project(model.description, vad.description, use_gpu), "model + VAD descriptions");
    expect_same_projection(artifact, project(model.artifact, vad.description, use_gpu), "VAD description only");
}

void check_wrong_kind(const fit_pair & model, const fit_pair & vad) {
    const whisper_fit_result vad_as_model = project(vad.description, "", false);
    expect(vad_as_model.status == WHISPER_FIT_ERROR && std::strcmp(vad_as_model.reason, "model-unreadable") == 0,
           "a VAD description is refused as a model");

    const whisper_fit_result model_as_vad = project(model.artifact, model.description, false);
    expect(model_as_vad.status == WHISPER_FIT_ERROR && std::strcmp(model_as_vad.reason, "vad-model-unreadable") == 0,
           "a model description is refused as a VAD model");
}

void truncate_vad_encoder_arrays(gguf_context * gguf) {
    for (const char * key : VAD_ENCODER_KEYS) {
        const int32_t *            values = (const int32_t *) gguf_get_arr_data(gguf, gguf_find_key(gguf, key));
        const std::vector<int32_t> kept(values, values + TRUNCATED_VAD_ENCODER_LAYERS);
        gguf_set_arr_data(gguf, key, GGUF_TYPE_INT32, kept.data(), kept.size());
    }
}

std::string write_truncated_vad_description(const std::string & description) {
    ggml_context *   meta   = nullptr;
    gguf_init_params params = {};
    params.no_alloc = true;
    params.ctx      = &meta;
    gguf_context * gguf = gguf_init_from_file(description.c_str(), params);
    if (gguf == nullptr) return "";

    truncate_vad_encoder_arrays(gguf);
    const std::string path    = (std::filesystem::temp_directory_path() / TRUNCATED_VAD_DESCRIPTION).string();
    const bool        written = gguf_write_to_file(gguf, path.c_str(), /*only_meta =*/ true);
    gguf_free(gguf);
    ggml_free(meta);
    return written ? path : "";
}

void check_truncated_vad_encoder(const fit_pair & model, const fit_pair & vad) {
    const std::string truncated = write_truncated_vad_description(vad.description);
    expect(!truncated.empty(), "the three-encoder VAD description was written");

    const whisper_fit_result fit = project(model.artifact, truncated, false);
    expect(fit.status == WHISPER_FIT_ERROR && std::strcmp(fit.reason, "vad-model-unreadable") == 0,
           "a VAD description without four encoder layers is refused");
    std::filesystem::remove(truncated);
}

void check_not_loadable(const fit_pair & model) {
    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false;
    whisper_context * ctx = whisper_init_from_file_with_params(model.description.c_str(), cparams);
    expect(ctx == nullptr, "a description loads as a model");
    whisper_free(ctx);
}

std::vector<fit_pair> pairs_from(int argc, char ** argv, bool & use_gpu) {
    std::vector<std::string> paths;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], GPU_ARG) == 0) {
            use_gpu = true;
        } else {
            paths.push_back(argv[i]);
        }
    }

    std::vector<fit_pair> pairs;
    for (size_t i = 0; i + 1 < paths.size(); i += PAIR_ARGS) {
        pairs.push_back({ paths[i], paths[i + 1] });
    }
    return pairs;
}

}  // namespace

int main(int argc, char ** argv) {
    bool use_gpu = false;
    const std::vector<fit_pair> pairs = pairs_from(argc, argv, use_gpu);
    if (pairs.size() < MIN_PAIRS) {
        std::fprintf(stderr, "usage: %s <vad.bin> <vad.fit.gguf> <model.bin> <model.fit.gguf> "
                             "[<model.bin> <model.fit.gguf> ...] [gpu]\n", argv[0]);
        return EXIT_USAGE;
    }

    whisper_log_set(log_callback_null, nullptr);

    if (use_gpu && !has_gpu_device()) {
        std::printf("test-whisper-fit-description: gpu requested but no GPU/IGPU device -- skipping\n");
        return EXIT_SKIP;
    }

    const fit_pair &            vad = pairs.front();
    const std::vector<fit_pair> models(pairs.begin() + 1, pairs.end());

    check_models(models, use_gpu);
    check_vad(models.front(), vad, use_gpu);
    check_wrong_kind(models.front(), vad);
    check_truncated_vad_encoder(models.front(), vad);
    check_not_loadable(models.front());

    if (g_failures == 0) {
        std::printf("test-whisper-fit-description: %zu model and VAD descriptions project like their artifacts\n",
                    models.size());
    }
    return g_failures;
}
