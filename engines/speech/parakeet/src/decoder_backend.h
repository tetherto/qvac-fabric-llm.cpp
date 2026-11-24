#pragma once

#include <string>

namespace parakeet {

inline std::string resolve_decoder_backend_name(const std::string & active_backend,
                                                const std::string & cpu_backend,
                                                bool decoder_on_cpu,
                                                const std::string & coreml_backend,
                                                bool include_coreml) {
    if (include_coreml && !coreml_backend.empty()) return coreml_backend;
    if (decoder_on_cpu && !cpu_backend.empty()) return cpu_backend;
    return active_backend;
}

}  // namespace parakeet
