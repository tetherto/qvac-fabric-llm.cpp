#pragma once

#include "moss/transcribe_text.h"
#include <stdexcept>

namespace parakeet::moss::detail {

constexpr int TRANSCRIBE_PREFILL_BATCH_TOKENS = 256;
constexpr size_t TRANSCRIBE_MAX_PROMPT_BYTES = 8192;

inline std::string resolved_transcribe_prompt(const TranscribeConfig & config, const TranscribeRequest & request) {
    const auto hotwords = sanitize_hotwords(request.hotwords);
    if (hotwords.empty()) return request.prompt;
    if (!strip_whitespace(request.prompt).empty()) {
        throw std::runtime_error("moss transcribe: hotwords extend the default prompt; write them into the custom prompt instead");
    }
    return hotword_prompt(config, hotwords);
}

inline int resolved_transcribe_limit(const TranscribeConfig & config, const TranscribeRequest & request) {
    if (request.max_new_tokens < 0) {
        throw std::runtime_error("moss transcribe: max_new_tokens must be positive, or 0 for the model default");
    }
    return request.max_new_tokens > 0 ? request.max_new_tokens : config.default_max_new_tokens;
}

}
