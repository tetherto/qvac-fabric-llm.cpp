#pragma once

#include "moss/transcribe_model.h"

#include <memory>
#include <set>
#include <string>
#include <vector>

namespace parakeet::moss::detail {

constexpr const char * COREML_DISABLE_ENV = "MOSS_COREML_DISABLE";
constexpr const char * COREML_STRICT_ENV = "MOSS_COREML_STRICT";
constexpr const char * COREML_UNITS_ENV = "MOSS_COREML_COMPUTE_UNITS";
constexpr const char * GGML_ENCODER_BACKEND = "ggml";
constexpr const char * MIXED_ENCODER_BACKEND = "mixed";

struct CoremlPolicy {
    bool disabled = false;
    bool strict = false;
};

CoremlPolicy read_coreml_policy();

class TranscribeEncoderSidecar {
public:
    virtual ~TranscribeEncoderSidecar() = default;
    virtual bool encode(const std::vector<float> & mel, std::vector<float> & embeddings) = 0;
    virtual const char * label() const = 0;
};

std::string transcribe_encoder_sidecar_path(const std::string & model_path);
std::vector<float> frame_major_mel(const std::vector<float> & mel, int n_mels, int frames);
std::unique_ptr<TranscribeEncoderSidecar> open_transcribe_encoder_sidecar(const std::string & model_path,
                                                                          const TranscribeConfig & config,
                                                                          const CoremlPolicy & policy);

class TranscribeAudioEncoder {
public:
    TranscribeAudioEncoder(TranscribeModel & model, std::unique_ptr<TranscribeEncoderSidecar> sidecar, bool strict);

    void begin_run();
    std::vector<float> encode(const std::vector<float> & mel, int tokens);
    bool on_coreml() const;
    std::string run_backend() const;

private:
    bool try_sidecar(const std::vector<float> & mel, int tokens, std::vector<float> & embeddings);
    std::vector<float> leading_rows(const std::vector<float> & rows, int tokens) const;

    TranscribeModel & model_;
    std::unique_ptr<TranscribeEncoderSidecar> sidecar_;
    bool strict_ = false;
    std::set<std::string> run_backends_;
};

} // namespace parakeet::moss::detail
