// common_fit_params_measured must report what common_get_device_memory_data
// reports for the parameters the fit leaves in place, on every outcome.

#include "../common/fit.h"

#include <atomic>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

static int failures = 0;

// every probe builds a context, and every context build logs this line
static std::atomic<int> contexts_built{0};

static void count_contexts(ggml_log_level, const char * text, void *) {
    if (std::strstr(text, "constructing llama_context") != nullptr) {
        contexts_built++;
    }
}

static void expect(const char * label, bool ok) {
    if (!ok) {
        fprintf(stderr, "FAIL %s\n", label);
        failures++;
    }
}

// free memory moves between probes, so only the demand and the model figures are compared
static void expect_matches_probe(const char * label, const char * path,
        const llama_model_params & mparams, const llama_context_params & cparams,
        const common_fit_measurement & measurement) {
    std::vector<ggml_backend_dev_t> devs;
    uint32_t n_layer = 0;
    uint32_t n_ctx_train = 0;
    uint32_t n_expert = 0;
    const common_device_memory_data_vec probe = common_get_device_memory_data(
        path, &mparams, &cparams, devs, n_layer, n_ctx_train, n_expert, GGML_LOG_LEVEL_ERROR);

    bool same = measurement.measured && measurement.memory.size() == probe.size() && measurement.devs == devs &&
        measurement.n_layer == n_layer && measurement.n_ctx_train == n_ctx_train && measurement.n_expert == n_expert;
    for (size_t i = 0; same && i < probe.size(); i++) {
        same = measurement.memory[i].total == probe[i].total && measurement.memory[i].model == probe[i].model &&
            measurement.memory[i].context == probe[i].context && measurement.memory[i].compute == probe[i].compute;
    }
    expect(label, same);
}

static common_params_fit_status fit(const char * path, llama_model_params & mparams, llama_context_params & cparams,
        size_t margin, common_fit_measurement & measurement) {
    static std::vector<float> split(llama_max_devices(), 0.0f);
    static std::vector<llama_model_tensor_buft_override> overrides(llama_max_tensor_buft_overrides());
    std::vector<size_t> margins(llama_max_devices(), margin);
    overrides[0] = { nullptr, nullptr };
    mparams.tensor_split          = split.data();
    mparams.tensor_buft_overrides = overrides.data();
    return common_fit_params_measured(path, &mparams, &cparams, split.data(), overrides.data(), margins.data(),
        0, nullptr, false, GGML_LOG_LEVEL_ERROR, &measurement);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }
    const char * path = argv[1];
    llama_backend_init();

    {
        auto mparams = llama_model_default_params();
        auto cparams = llama_context_default_params();
        common_fit_measurement measurement;
        contexts_built = 0;
        llama_log_set(count_contexts, nullptr);
        const auto status = fit(path, mparams, cparams, 0, measurement);
        llama_log_set(nullptr, nullptr);
        expect("a fit with room succeeds", status == COMMON_PARAMS_FIT_STATUS_SUCCESS);
        expect("a fit with room builds the model once", contexts_built == 1);
        expect_matches_probe("a fit with room measures its parameters", path, mparams, cparams, measurement);
    }

    {
        // building the context rescales the training context, which the automatic context then follows
        auto mparams = llama_model_default_params();
        auto cparams = llama_context_default_params();
        cparams.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN;
        cparams.rope_freq_scale   = 0.25f;
        cparams.n_ctx             = 0;
        common_fit_measurement measurement;
        const auto status = fit(path, mparams, cparams, 0, measurement);
        expect("a rescaled fit with room succeeds", status == COMMON_PARAMS_FIT_STATUS_SUCCESS);
        expect("a rescaled fit uses the rescaled context", cparams.n_ctx == measurement.n_ctx_train);
        expect_matches_probe("a rescaled fit measures its parameters", path, mparams, cparams, measurement);
    }

    {
        auto mparams = llama_model_default_params();
        auto cparams = llama_context_default_params();
        cparams.n_ctx = 512;
        common_fit_measurement measurement;
        const auto status = fit(path, mparams, cparams, SIZE_MAX / 4, measurement);
        if (measurement.devs.empty()) {
            expect("a fit without room fails without a device", status == COMMON_PARAMS_FIT_STATUS_FAILURE);
        } else {
            expect("a fit without room reduces the placement", status == COMMON_PARAMS_FIT_STATUS_SUCCESS &&
                mparams.n_gpu_layers != llama_model_default_params().n_gpu_layers);
        }
        expect_matches_probe("a reduced fit measures its parameters", path, mparams, cparams, measurement);
    }

    {
        auto mparams = llama_model_default_params();
        auto cparams = llama_context_default_params();
        mparams.n_gpu_layers = 99;
        cparams.n_ctx = 512;
        common_fit_measurement measurement;
        const auto status = fit(path, mparams, cparams, SIZE_MAX / 4, measurement);
        expect("a pinned placement without room fails", status == COMMON_PARAMS_FIT_STATUS_FAILURE);
        expect_matches_probe("a failed fit measures the restored parameters", path, mparams, cparams, measurement);
    }

    {
        auto mparams = llama_model_default_params();
        auto cparams = llama_context_default_params();
        common_fit_measurement measurement;
        measurement.measured = true;
        const auto status = fit("/nonexistent-fit-test/model.gguf", mparams, cparams, 0, measurement);
        expect("an unreadable model is an error", status == COMMON_PARAMS_FIT_STATUS_ERROR);
        expect("an error measures nothing", !measurement.measured && measurement.memory.empty());
    }

    llama_backend_free();
    return failures == 0 ? 0 : 1;
}
