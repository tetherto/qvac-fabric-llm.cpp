#pragma once
// models/supertonic2-q8_0.gguf -> models/supertonic2-vocoder.mlmodelc
// (extension and quantisation tag dropped: one sidecar serves every tier).

#include <string>
#include <utility>
#include <vector>

namespace tts_cpp {
namespace supertonic {
namespace detail {

std::string coreml_vocoder_sidecar_path(const std::string & gguf_path);
std::string coreml_encoder_sidecar_path(const std::string & gguf_path);
std::string coreml_vector_sidecar_path(const std::string & gguf_path, int capacity);
std::vector<std::pair<int, std::string>> coreml_vector_sidecar_paths(const std::string & gguf_path);

}  // namespace detail
}  // namespace supertonic
}  // namespace tts_cpp
