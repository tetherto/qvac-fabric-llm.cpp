#include "moss/sfx_request.h"
#include "moss/sfx_prompt.h"
#include <cmath>
#include <stdexcept>

namespace tts_cpp::moss::detail {
namespace {
constexpr int MAX_STEPS = 1000;
constexpr float MAX_GUIDANCE = 50.0f;
constexpr float MAX_SHIFT = 100.0f;
constexpr float UNGUIDED = 1.0f;
constexpr int TENTHS_PER_SECOND = 10;
constexpr size_t MAX_PROMPT_BYTES = 8192;

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("moss sfx: " + message);
}

bool is_valid_override(float value) {
    return std::isfinite(value) && value >= 0.0f;
}

void validate_overrides(const SoundEffectRequest & request) {
    if (request.steps < 0) {
        fail("steps must be 1.." + std::to_string(MAX_STEPS) + ", or 0 for the model default");
    }
    if (!is_valid_override(request.guidance)) {
        fail("guidance must be in [1, " + std::to_string((int) MAX_GUIDANCE) + "], or 0 for the model default");
    }
    if (!is_valid_override(request.shift)) {
        fail("shift must be in (0, " + std::to_string((int) MAX_SHIFT) + "], or 0 for the model default");
    }
    if (!std::isfinite(request.seconds)) {
        fail("seconds must be a finite number");
    }
    if (request.prompt.size() > MAX_PROMPT_BYTES || request.negative_prompt.size() > MAX_PROMPT_BYTES) {
        fail("prompts must be at most " + std::to_string(MAX_PROMPT_BYTES) + " bytes");
    }
}

void validate_prompt(const std::string & prompt) {
    if (detail::clean_prompt(prompt).empty()) {
        fail("prompt must not be empty");
    }
}

void validate(const SfxConfig & config, const ResolvedRequest & request) {
    const float max_seconds = config.max_seconds;
    if (request.tenths < 1 || (float) request.tenths > max_seconds * TENTHS_PER_SECOND) {
        fail("seconds must be in (0, " + std::to_string(max_seconds) + "]");
    }
    if (request.steps > MAX_STEPS) {
        fail("steps must be 1.." + std::to_string(MAX_STEPS));
    }
    if (request.guidance < UNGUIDED || request.guidance > MAX_GUIDANCE) {
        fail("guidance must be in [1, " + std::to_string((int) MAX_GUIDANCE) + "]");
    }
    if (request.shift > MAX_SHIFT) {
        fail("shift must be in (0, " + std::to_string((int) MAX_SHIFT) + "]");
    }
}

}

ResolvedRequest resolve_sfx_request(const SfxConfig & config, const SoundEffectRequest & request) {
    validate_overrides(request);
    ResolvedRequest resolved;
    resolved.tenths = detail::seconds_in_tenths(request.seconds);
    resolved.conditioning = detail::clean_prompt(detail::duration_prompt(request.prompt, resolved.tenths));
    resolved.negative_prompt = detail::clean_prompt(request.negative_prompt);
    resolved.steps = request.steps > 0 ? request.steps : config.default_steps;
    resolved.guidance = request.guidance > 0.0f ? request.guidance : config.default_guidance;
    resolved.shift = request.shift > 0.0f ? request.shift : config.sigma_shift;
    resolved.seed = request.seed;
    validate_prompt(request.prompt);
    validate(config, resolved);
    return resolved;
}

}
