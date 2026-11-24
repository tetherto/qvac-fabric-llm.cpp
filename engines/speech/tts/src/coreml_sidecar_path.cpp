#include "coreml_sidecar_path.h"

#include <sys/stat.h>

#include <array>
#include <cctype>
#include <cstddef>

namespace tts_cpp {
namespace detail {
namespace {

constexpr std::array<const char *, 6> QUANT_TAGS = {"f32", "f16", "bf16", "q8_0", "q5_0", "q4_0"};

std::string lowercase(std::string token) {
    for (char & c : token) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return token;
}

bool is_quant_tag(const std::string & token) {
    const std::string low = lowercase(token);
    for (const char * tag : QUANT_TAGS) {
        if (low == tag) return true;
    }
    return false;
}

std::size_t basename_separator(const std::string & path, std::size_t slash) {
    const std::size_t sep = path.find_last_of(".-");
    if (sep == std::string::npos) return std::string::npos;
    if (slash != std::string::npos && sep <= slash) return std::string::npos;
    return sep;
}

}  // namespace

std::string coreml_sidecar_path(const std::string & gguf_path, const std::string & suffix) {
    std::string path = gguf_path;
    const std::size_t slash = path.find_last_of("/\\");

    const std::size_t dot = path.rfind('.');
    if (dot != std::string::npos && (slash == std::string::npos || dot > slash)) {
        path.erase(dot);
    }

    const std::size_t sep = basename_separator(path, slash);
    if (sep != std::string::npos && is_quant_tag(path.substr(sep + 1))) {
        path.erase(sep);
    }

    return path + suffix + ".mlmodelc";
}

bool coreml_sidecar_exists(const std::string & path) {
    struct stat st {};
    return stat(path.c_str(), &st) == 0;
}

}  // namespace detail
}  // namespace tts_cpp
