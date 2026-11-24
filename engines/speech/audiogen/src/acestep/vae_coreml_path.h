#pragma once

#include <string>

namespace tts_cpp::acestep {

// "vae-BF16.gguf" -> "vae-decoder.mlmodelc", next to the GGUF.
std::string coreml_vae_sidecar_path(const std::string & gguf_path);

}  // namespace tts_cpp::acestep
