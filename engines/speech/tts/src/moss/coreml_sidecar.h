#pragma once

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>

struct moss_coreml_model;

namespace tts_cpp::moss::detail {

constexpr const char * COREML_DISABLE_ENV = "MOSS_COREML_DISABLE";
constexpr const char * COREML_STRICT_ENV = "MOSS_COREML_STRICT";
constexpr const char * COREML_UNITS_ENV = "MOSS_COREML_COMPUTE_UNITS";
constexpr const char * GGML_STAGE_BACKEND = "ggml";
constexpr const char * MIXED_STAGE_BACKEND = "mixed";
constexpr const char * COREML_DEFAULT_UNITS = "cpu_and_gpu";

struct CoremlPolicy {
    bool disabled = false;
    bool strict = false;
};

CoremlPolicy read_coreml_policy();
std::string requested_coreml_units();

std::string sfx_dit_sidecar_path(const std::string & model_path);
std::string sfx_vae_sidecar_path(const std::string & model_path);
std::string speech_tokenizer_sidecar_path(const std::string & codec_path);
std::string codec_decoder_sidecar_path(const std::string & decoder_path);

[[noreturn]] void fail_strict(const std::string & owner, const std::string & stage);

class StageBackends {
public:
    void begin();
    void note(const std::string & backend);
    std::string summary() const;

private:
    std::set<std::string> used_;
};

struct CoremlTensor {
    std::string name;
    std::vector<int64_t> dims;
};

class CoremlModel {
public:
    static std::unique_ptr<CoremlModel> open(const std::string & path, const CoremlPolicy & policy);
    ~CoremlModel();
    CoremlModel(const CoremlModel &) = delete;
    CoremlModel & operator=(const CoremlModel &) = delete;

    bool declares(const CoremlTensor & tensor) const;
    bool declares(const std::vector<CoremlTensor> & tensors) const;
    std::vector<int64_t> dims(const std::string & name) const;
    bool stateful() const;
    bool reset_state();
    bool predict(const std::vector<CoremlTensor> & inputs, const std::vector<const float *> & input_data,
                 const std::vector<CoremlTensor> & outputs, const std::vector<float *> & output_data);
    const char * label() const;

private:
    explicit CoremlModel(moss_coreml_model * model);

    moss_coreml_model * model_;
};

int64_t tensor_elements(const CoremlTensor & tensor);

} // namespace tts_cpp::moss::detail
