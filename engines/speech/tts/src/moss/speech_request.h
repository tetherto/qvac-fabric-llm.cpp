#pragma once

#include "tts-cpp/moss/speech.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace tts_cpp::moss::detail {

void validate_speech_request(const SpeechRequest & request);
void check_speech_context(size_t prompt_rows, int max_new_tokens, int n_ctx_train);
int speech_generation_budget(size_t prompt_rows, int requested_tokens, int n_ctx_train);
// Whether the device's free memory, with the LM loaded, holds the decoder and
// still leaves room for a turn. An unknown free figure (0) never fits.
bool speech_stages_fit(uint64_t device_free_bytes, uint64_t decoder_bytes);
// Whether a decode's projected device bytes stay within what the resident
// stages were sized for. A failed projection (UINT64_MAX) never does.
bool speech_decode_within_plan(uint64_t planned_bytes, uint64_t decode_bytes);

// Runs the optional residency `plan`; if it throws, calls `release` and
// reports the stages as staged instead of failing the engine.
template <typename Plan, typename Release>
bool try_hold_stages(Plan && plan, Release && release) {
    try {
        return plan();
    } catch (const std::exception &) {
        release();
        return false;
    }
}

// Runs `attempt`; if it fails while the stages are resident, calls `release`
// and runs it once more. `attempt` must start its work from scratch.
template <typename Attempt, typename Release>
auto retry_after_releasing_stages(bool stages_resident, Attempt && attempt, Release && release)
        -> decltype(attempt()) {
    try {
        return attempt();
    } catch (const std::runtime_error &) {
        if (!stages_resident) {
            throw;
        }
    }
    release();
    return attempt();
}

} // namespace tts_cpp::moss::detail
