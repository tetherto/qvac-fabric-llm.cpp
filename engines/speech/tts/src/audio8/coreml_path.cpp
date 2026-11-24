#include "audio8/coreml_path.h"

#include "coreml_sidecar_path.h"

namespace tts_cpp {
namespace audio8 {
namespace detail {

std::string coreml_codec_sidecar_path(const std::string & gguf_path) {
    return ::tts_cpp::detail::coreml_sidecar_path(gguf_path, "");
}

}  // namespace detail
}  // namespace audio8
}  // namespace tts_cpp
