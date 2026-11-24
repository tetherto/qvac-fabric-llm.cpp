#pragma once
// models/audio8-codec-decoder-q8_0.gguf -> models/audio8-codec-decoder.mlmodelc
// (extension and quantisation tag dropped: one sidecar serves every tier).

#include <string>

namespace tts_cpp {
namespace audio8 {
namespace detail {

std::string coreml_codec_sidecar_path(const std::string & gguf_path);

}  // namespace detail
}  // namespace audio8
}  // namespace tts_cpp
