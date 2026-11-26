#pragma once

// Moved to coreml/include/qvac_coreml/gguf_stream.h: the Core ML
// generator reads source tensors with the same bounded reader.
// This forwarding header keeps tts-cpp's call sites unchanged; the relative
// path works for every target that compiles tts sources, with or without
// tts-cpp's include directories.
#include "../../coreml/include/qvac_coreml/gguf_stream.h"

namespace tts_cpp {
namespace detail {
using ::qvac_gguf::gguf_stream_reader;
}  // namespace detail
}  // namespace tts_cpp
