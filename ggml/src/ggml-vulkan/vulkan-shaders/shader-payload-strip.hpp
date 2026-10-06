#pragma once

#include <string>

inline constexpr const char * const kStripShaderNameTokens[] = {
    "q1_0",
    "iq1_",
    "iq2_",
    "iq3_",
    "iq4_",
    "mxfp4",
    "nvfp4",
    "repeat_back",
    "rms_norm_back",
    "silu_back",
    "geglu_back",
    "soft_max_back",
    "cross_entropy_loss",
    "opt_step",
    "out_prod",
};

inline bool shader_name_contains_token(const std::string & name, const char * token) {
    return name.find(token) != std::string::npos;
}

inline bool shader_name_contains_any_listed_token(const std::string & name) {
    for (const char * token : kStripShaderNameTokens) {
        if (shader_name_contains_token(name, token)) {
            return true;
        }
    }
    return false;
}

inline bool should_strip_shader_payload(const std::string & name) {
    return shader_name_contains_any_listed_token(name);
}

inline bool should_strip_shader_payload(const std::string & name, bool strip_enabled) {
    return strip_enabled && should_strip_shader_payload(name);
}
