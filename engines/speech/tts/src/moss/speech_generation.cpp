#include "moss/speech_generation.h"

#include "moss/generation.h"
#include "moss/speech_prompt.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace tts_cpp::moss::detail {
namespace {

void divide(std::vector<float> & logits, float temperature) {
    for (float & value : logits) {
        value /= temperature;
    }
}

bool is_reply_code(int32_t audio, const SpeechTokens & tokens) {
    return audio != tokens.speech_end;
}

bool is_speech_row(const SpeechRow & row, const SpeechTokens & tokens) {
    return row.text == tokens.text_placeholder;
}

size_t first_speech_row(const std::vector<SpeechRow> & rows, size_t end, const SpeechTokens & tokens) {
    size_t i = 0;
    while (i < end && !is_speech_row(rows[i], tokens)) {
        ++i;
    }
    return i;
}

} // namespace

SpeechGenerationState::SpeechGenerationState(const SpeechTokens & tokens, const SpeechRow & last_prompt_row,
                                             const SpeechLimits & limits)
    : tokens_(tokens), limits_(limits), last_(last_prompt_row) {
    if (last_prompt_row.text == tokens.text_placeholder) {
        channel_ = SpeechChannel::Audio;
    }
    if (last_prompt_row.audio == tokens.audio_pad) {
        channel_ = SpeechChannel::Text;
    }
    stopping_ = limits.max_new_tokens < 1;
}

bool SpeechGenerationState::stopping() const { return stopping_; }
bool SpeechGenerationState::truncated() const { return truncated_; }
SpeechChannel SpeechGenerationState::channel() const { return channel_; }
const std::vector<SpeechRow> & SpeechGenerationState::generated() const { return generated_; }

SpeechChannel SpeechGenerationState::channel_after(const SpeechRow & last) const {
    if (channel_ == SpeechChannel::Text && last.text == tokens_.speech_start) {
        return SpeechChannel::Audio;
    }
    if (channel_ == SpeechChannel::Audio && last.audio == tokens_.speech_end) {
        return SpeechChannel::Text;
    }
    return channel_;
}

SpeechChannel SpeechGenerationState::upcoming_channel() const {
    return channel_after(last_);
}

void SpeechGenerationState::constrain_audio(std::vector<float> & audio) const {
    const float blocked = -std::numeric_limits<float>::infinity();
    const size_t first_invalid = std::min(audio.size(), (size_t) tokens_.speech_end + 1);
    std::fill(audio.begin() + (std::ptrdiff_t) first_invalid, audio.end(), blocked);
    if ((int) generated_.size() + 1 < limits_.min_new_tokens && (size_t) tokens_.speech_end < audio.size()) {
        audio[(size_t) tokens_.speech_end] = blocked;
    }
}

int32_t SpeechGenerationState::sample(std::vector<float> logits, const SpeechSampling & sampling,
                                      std::mt19937 & rng) const {
    if (sampling.do_sample && sampling.temperature > 0.0f) {
        divide(logits, sampling.temperature);
    }
    return sample_row(std::move(logits), sampling.top_p, sampling.top_k, sampling.do_sample, rng);
}

void SpeechGenerationState::constrain_text(std::vector<float> & text) const {
    if ((int) generated_.size() < limits_.min_new_tokens && (size_t) tokens_.im_end < text.size()) {
        text[(size_t) tokens_.im_end] = -std::numeric_limits<float>::infinity();
    }
}

int32_t SpeechGenerationState::sample_text(std::vector<float> logits, const SpeechSampling & sampling,
                                           std::mt19937 & rng) const {
    if (channel_ == SpeechChannel::Audio) {
        return tokens_.text_placeholder;
    }
    if (logits.empty()) {
        throw std::runtime_error("moss speech: the text channel needs text logits");
    }
    constrain_text(logits);
    return sample(std::move(logits), sampling, rng);
}

int32_t SpeechGenerationState::limit_reply(int32_t audio) {
    if (channel_ != SpeechChannel::Audio || !is_reply_code(audio, tokens_)) {
        return audio;
    }
    if (limits_.max_reply_codes > 0 && reply_codes_ >= limits_.max_reply_codes) {
        truncated_ = true;
        return tokens_.speech_end;
    }
    reply_codes_++;
    return audio;
}

void SpeechGenerationState::record(const SpeechRow & row) {
    generated_.push_back(row);
    last_ = row;
    const bool stop_row = is_speech_stop_row(row, tokens_);
    const bool out_of_tokens = (int) generated_.size() >= limits_.max_new_tokens;
    stopping_ = stop_row || out_of_tokens;
    truncated_ = truncated_ || (out_of_tokens && !stop_row);
}

SpeechRow SpeechGenerationState::next(SpeechLogits logits, const SpeechSampling & sampling, std::mt19937 & rng) {
    channel_ = channel_after(last_);
    constrain_audio(logits.audio);
    SpeechRow row{sample_text(std::move(logits.text), sampling, rng), sample(std::move(logits.audio), sampling, rng)};
    row.audio = limit_reply(row.audio);
    record(row);
    return row;
}

bool run_speech_generation(SpeechLM & lm, SpeechGenerationState & state, SpeechLogits logits,
                           const SpeechSampling & sampling, uint32_t seed, const SpeechContinue & keep_going) {
    std::mt19937 rng(seed);
    while (!state.stopping()) {
        const SpeechRow row = state.next(std::move(logits), sampling, rng);
        if (!keep_going((int) state.generated().size())) {
            return false;
        }
        if (!state.stopping()) {
            logits = lm.step(row, {state.upcoming_channel() == SpeechChannel::Text, true});
        }
    }
    return true;
}

std::vector<int32_t> speech_reply_codes(const std::vector<SpeechRow> & generated, const SpeechTokens & tokens) {
    std::vector<int32_t> codes;
    const size_t kept = speech_reply_rows(generated, tokens);
    for (size_t i = first_speech_row(generated, kept, tokens);
         i < kept && is_speech_row(generated[i], tokens) && is_reply_code(generated[i].audio, tokens); ++i) {
        codes.push_back(generated[i].audio);
    }
    return codes;
}

} // namespace tts_cpp::moss::detail
