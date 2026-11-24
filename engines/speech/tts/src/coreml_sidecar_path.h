#pragma once
// Shared Core ML sidecar naming rule: drop the GGUF extension and a trailing
// quantisation tag, append `<suffix>.mlmodelc` — one sidecar serves every
// quantisation tier of the same model.
//   models/audio8-codec-decoder-q8_0.gguf + ""         -> models/audio8-codec-decoder.mlmodelc
//   models/supertonic2-q8_0.gguf          + "-vocoder" -> models/supertonic2-vocoder.mlmodelc

#include <string>

namespace tts_cpp {
namespace detail {

std::string coreml_sidecar_path(const std::string & gguf_path, const std::string & suffix);

bool coreml_sidecar_exists(const std::string & path);

}  // namespace detail
}  // namespace tts_cpp
