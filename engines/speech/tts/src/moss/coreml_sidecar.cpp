#include "moss/coreml_sidecar.h"

#include "coreml_sidecar_path.h"

#ifdef TTS_CPP_USE_COREML
#include "moss/coreml/runner.h"
#endif

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iterator>
#include <stdexcept>

namespace tts_cpp::moss::detail {
namespace {

constexpr const char * DIT_SUFFIX = "-dit";
constexpr const char * VAE_SUFFIX = "-vae";
constexpr const char * TOKENIZER_SUFFIX = "-tokenizer";
constexpr const char * DECODER_SUFFIX = "";
constexpr const char * KNOWN_COMPUTE_UNITS[] = {"all", "cpu_and_gpu", "cpu_and_ane", "cpu_only"};

bool env_set(const char * name) {
    return std::getenv(name) != nullptr;
}

bool known_compute_units(const std::string & units) {
    return std::find(std::begin(KNOWN_COMPUTE_UNITS), std::end(KNOWN_COMPUTE_UNITS), units) !=
           std::end(KNOWN_COMPUTE_UNITS);
}

#ifdef TTS_CPP_USE_COREML
constexpr int MAX_RANK = 8;

std::vector<const char *> tensor_names(const std::vector<CoremlTensor> & tensors) {
    std::vector<const char *> names;
    for (const CoremlTensor & tensor : tensors) {
        names.push_back(tensor.name.c_str());
    }
    return names;
}
#endif

} // namespace

CoremlPolicy read_coreml_policy() {
    CoremlPolicy policy;
    policy.disabled = env_set(COREML_DISABLE_ENV);
    policy.strict = env_set(COREML_STRICT_ENV);
    return policy;
}

std::string requested_coreml_units() {
    const char * requested = std::getenv(COREML_UNITS_ENV);
    return requested != nullptr && known_compute_units(requested) ? requested : COREML_DEFAULT_UNITS;
}

std::string sfx_dit_sidecar_path(const std::string & model_path) {
    return ::tts_cpp::detail::coreml_sidecar_path(model_path, DIT_SUFFIX);
}

std::string sfx_vae_sidecar_path(const std::string & model_path) {
    return ::tts_cpp::detail::coreml_sidecar_path(model_path, VAE_SUFFIX);
}

std::string speech_tokenizer_sidecar_path(const std::string & codec_path) {
    return ::tts_cpp::detail::coreml_sidecar_path(codec_path, TOKENIZER_SUFFIX);
}

std::string codec_decoder_sidecar_path(const std::string & decoder_path) {
    return ::tts_cpp::detail::coreml_sidecar_path(decoder_path, DECODER_SUFFIX);
}

void fail_strict(const std::string & owner, const std::string & stage) {
    throw std::runtime_error(owner + ": no working Core ML " + stage + " sidecar and " + COREML_STRICT_ENV +
                             " is set");
}

void StageBackends::begin() {
    used_.clear();
}

void StageBackends::note(const std::string & backend) {
    used_.insert(backend);
}

std::string StageBackends::summary() const {
    if (used_.empty()) {
        return std::string();
    }
    return used_.size() == 1 ? *used_.begin() : MIXED_STAGE_BACKEND;
}

int64_t tensor_elements(const CoremlTensor & tensor) {
    int64_t count = tensor.dims.empty() ? 0 : 1;
    for (int64_t dim : tensor.dims) {
        count *= dim;
    }
    return count;
}

CoremlModel::CoremlModel(moss_coreml_model * model) : model_(model) {}

bool CoremlModel::declares(const std::vector<CoremlTensor> & tensors) const {
    for (const CoremlTensor & tensor : tensors) {
        if (!declares(tensor)) {
            return false;
        }
    }
    return true;
}

#ifdef TTS_CPP_USE_COREML
std::unique_ptr<CoremlModel> CoremlModel::open(const std::string & path, const CoremlPolicy & policy) {
    if (policy.disabled || !std::filesystem::is_directory(path)) {
        return nullptr;
    }
    moss_coreml_model * model = moss_coreml_load(path.c_str(), requested_coreml_units().c_str());
    return model != nullptr ? std::unique_ptr<CoremlModel>(new CoremlModel(model)) : nullptr;
}

CoremlModel::~CoremlModel() {
    moss_coreml_free(model_);
}

bool CoremlModel::declares(const CoremlTensor & tensor) const {
    return dims(tensor.name) == tensor.dims;
}

std::vector<int64_t> CoremlModel::dims(const std::string & name) const {
    int64_t declared[MAX_RANK] = {};
    const int rank = moss_coreml_feature_dims(model_, name.c_str(), declared, MAX_RANK);
    return std::vector<int64_t>(declared, declared + std::max(rank, 0));
}

bool CoremlModel::stateful() const {
    return moss_coreml_has_state(model_) != 0;
}

bool CoremlModel::reset_state() {
    return moss_coreml_reset_state(model_) == 0;
}

bool CoremlModel::predict(const std::vector<CoremlTensor> & inputs, const std::vector<const float *> & input_data,
                          const std::vector<CoremlTensor> & outputs, const std::vector<float *> & output_data) {
    const std::vector<const char *> input_names = tensor_names(inputs);
    const std::vector<const char *> output_names = tensor_names(outputs);
    return moss_coreml_predict(model_, input_names.data(), input_data.data(), (int) inputs.size(),
            output_names.data(), output_data.data(), (int) outputs.size()) == 0;
}

const char * CoremlModel::label() const {
    return moss_coreml_label(model_);
}
#else
std::unique_ptr<CoremlModel> CoremlModel::open(const std::string &, const CoremlPolicy &) {
    return nullptr;
}

CoremlModel::~CoremlModel() = default;

bool CoremlModel::declares(const CoremlTensor &) const {
    return false;
}

std::vector<int64_t> CoremlModel::dims(const std::string &) const {
    return {};
}

bool CoremlModel::stateful() const {
    return false;
}

bool CoremlModel::reset_state() {
    return false;
}

bool CoremlModel::predict(const std::vector<CoremlTensor> &, const std::vector<const float *> &,
                          const std::vector<CoremlTensor> &, const std::vector<float *> &) {
    return false;
}

const char * CoremlModel::label() const {
    return GGML_STAGE_BACKEND;
}
#endif

} // namespace tts_cpp::moss::detail
