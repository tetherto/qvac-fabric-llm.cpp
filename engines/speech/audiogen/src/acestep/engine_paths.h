#pragma once

// models_dir -> per-stage GGUF path classification, shared by Engine::create
// (engine.cpp) and the memory-fit preflight (fit.cpp) so both resolve the same
// files by construction.

#include "audiogen-cpp/acestep/engine.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <string>
#include <system_error>

namespace tts_cpp::acestep {

inline std::string acestep_to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return s;
}

// Classify the four ACE-Step GGUFs in models_dir by anchoring on their known
// filename stems (Qwen3-Embedding / acestep-*-lm / vae / acestep-v15-{turbo,sft}).
// Explicit paths in EngineOptions always win over the scan. The stems are chosen
// so no bare short token (like "lm") can match an unrelated file; the most
// specific stages (embedding, vae) are tested before the shorter lm/dit stems.
inline void resolve_stage_paths(EngineOptions & o) {
    namespace fs = std::filesystem;
    if (o.models_dir.empty()) return;
    std::error_code ec;
    for (auto & e : fs::directory_iterator(o.models_dir, ec)) {
        if (!e.is_regular_file()) continue;
        std::string path = e.path().string();
        std::string name = acestep_to_lower(e.path().filename().string());
        if (name.size() < 5 || name.substr(name.size() - 5) != ".gguf") continue;
        auto has = [&](const char * s) { return name.find(s) != std::string::npos; };
        if (has("embedding") || has("text-enc") || has("textenc")) {
            if (o.text_enc_model_path.empty()) o.text_enc_model_path = path;
        } else if (has("vae")) {
            if (o.vae_model_path.empty()) o.vae_model_path = path;
        } else if (has("-lm") || has("lm-") || has("_lm") || has("ace-lm") || has("5hz-lm")) {
            if (o.lm_model_path.empty()) o.lm_model_path = path;
        } else if (has("turbo") || has("dit") || has("v15") || has("sft")) {
            if (o.dit_model_path.empty()) o.dit_model_path = path;
        }
    }
}

}  // namespace tts_cpp::acestep
