#pragma once

#include "moss/speech_lm.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::moss::detail {

class SpeechTextTokenizer {
public:
    SpeechTextTokenizer(const std::vector<std::string> & tokens, const std::vector<std::string> & merges,
                        const std::vector<int32_t> & types);
    ~SpeechTextTokenizer();
    SpeechTextTokenizer(const SpeechTextTokenizer &) = delete;
    SpeechTextTokenizer & operator=(const SpeechTextTokenizer &) = delete;

    std::vector<int32_t> encode(const std::string & text) const;
    std::string decode(const std::vector<int32_t> & ids) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

enum class SpeechTurnRole { System, User, Assistant };

struct SpeechTurn {
    SpeechTurnRole role = SpeechTurnRole::User;
    std::string text;
    std::vector<int32_t> audio_codes;
    bool is_audio = false;
};

std::vector<SpeechRow> speech_text_rows(const std::vector<int32_t> & ids, const SpeechTokens & tokens);
std::vector<SpeechRow> speech_audio_rows(const std::vector<int32_t> & codes, const SpeechTokens & tokens);
std::vector<SpeechRow> speech_prompt(const SpeechLmConfig & config, const SpeechTextTokenizer & tokenizer,
                                     const std::vector<SpeechTurn> & turns, bool audio_reply);
bool is_speech_stop_row(const SpeechRow & row, const SpeechTokens & tokens);
size_t speech_reply_rows(const std::vector<SpeechRow> & generated, const SpeechTokens & tokens);
std::string speech_reply_text(const SpeechTextTokenizer & tokenizer, const std::vector<SpeechRow> & generated,
                              const SpeechTokens & tokens);

} // namespace tts_cpp::moss::detail
