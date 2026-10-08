#pragma once

// Adreno detection and backend policy shared by ggml-backend-reg.cpp and
// tests/test-adreno-version.cpp. Header-only so the test needs no GPU.

#include <algorithm>
#include <cctype>
#include <regex>
#include <string>

// Returns the Adreno generation from a device description ("Adreno (TM) 830" -> 830),
// -1 when the device is not an Adreno GPU, -3 when the generation does not parse.
// The generation is anchored to the "adreno" marker so the API version in an OpenCL
// description such as "qualcomm adreno(tm) (opencl 3.0 adreno(tm) 740)" is skipped.
inline int ggml_adreno_version_from_description(const std::string & gpu_description) {
    std::string lowered = gpu_description;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (lowered.find("dreno") == std::string::npos) {
        return -1;
    }

    static const std::regex adreno_regex(R"(dreno\D*?(\d{3,4}))");
    std::smatch             matches;
    if (std::regex_search(lowered, matches, adreno_regex) && matches.size() > 1) {
        try {
            return std::stoi(matches[1].str());
        } catch (const std::exception &) {
            return -3;
        }
    }
    return -3;
}

// Android backend policy for the smallest Adreno generation found:
//   not Adreno (<= 0) -> no OpenCL, keep Vulkan/CPU
//   Adreno > 700      -> load OpenCL alongside Vulkan
//   Adreno 1..700     -> CPU only: unload Vulkan, no OpenCL
struct ggml_adreno_backend_policy {
    bool load_opencl;
    bool unload_vulkan;
};

inline ggml_adreno_backend_policy ggml_adreno_resolve_backend_policy(int min_adreno_version) {
    if (min_adreno_version <= 0) {
        return ggml_adreno_backend_policy{ /*load_opencl=*/false, /*unload_vulkan=*/false };
    }
    if (min_adreno_version > 700) {
        return ggml_adreno_backend_policy{ /*load_opencl=*/true, /*unload_vulkan=*/false };
    }
    return ggml_adreno_backend_policy{ /*load_opencl=*/false, /*unload_vulkan=*/true };
}
