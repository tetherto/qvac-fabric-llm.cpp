#include "supertonic_coreml_path.h"

#include "coreml_sidecar_path.h"

namespace tts_cpp {
namespace supertonic {
namespace detail {

std::string coreml_vocoder_sidecar_path(const std::string & gguf_path) {
    return ::tts_cpp::detail::coreml_sidecar_path(gguf_path, "-vocoder");
}

std::string coreml_encoder_sidecar_path(const std::string & gguf_path) {
    return ::tts_cpp::detail::coreml_sidecar_path(gguf_path, "-encoder");
}

std::string coreml_vector_sidecar_path(const std::string & gguf_path, int capacity) {
    return ::tts_cpp::detail::coreml_sidecar_path(gguf_path, "-vector-L" + std::to_string(capacity));
}

std::vector<std::pair<int, std::string>> coreml_vector_sidecar_paths(const std::string & gguf_path) {
    std::vector<std::pair<int, std::string>> paths;
    for (int capacity : {64, 128, 256, 512}) {
        auto path = coreml_vector_sidecar_path(gguf_path, capacity);
        if (::tts_cpp::detail::coreml_sidecar_exists(path)) paths.emplace_back(capacity, std::move(path));
    }
    return paths;
}

}  // namespace detail
}  // namespace supertonic
}  // namespace tts_cpp
