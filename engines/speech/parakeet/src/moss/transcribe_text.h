#pragma once

#include "moss/transcribe_model.h"

#include "parakeet/moss_transcribe.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace parakeet::moss::detail {

class TranscribeTokenizer {
public:
    TranscribeTokenizer(const std::vector<std::string> & tokens, const std::vector<std::string> & merges,
                        const std::vector<int32_t> & types);
    explicit TranscribeTokenizer(const TranscribeModel & model);
    ~TranscribeTokenizer();
    TranscribeTokenizer(const TranscribeTokenizer &) = delete;
    TranscribeTokenizer & operator=(const TranscribeTokenizer &) = delete;

    std::vector<int32_t> encode(const std::string & text) const;
    std::string decode(const std::vector<int32_t> & ids) const;
    uint64_t storage_bytes() const;
    size_t max_piece_bytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::vector<int32_t> transcribe_audio_span(const TranscribeConfig & config, const TranscribeTokenizer & tokenizer,
                                           int audio_tokens);
constexpr size_t MAX_HOTWORDS = 64;
constexpr size_t MAX_HOTWORD_BYTES = 64;

double parse_decimal(const std::string & text);
std::string format_decimal(double value);

std::string sanitize_hotword(const std::string & hotword);
std::vector<std::string> sanitize_hotwords(const std::vector<std::string> & hotwords);
std::string hotword_prompt(const TranscribeConfig & config, const std::vector<std::string> & hotwords);

std::vector<int32_t> transcribe_prompt(const TranscribeConfig & config, const TranscribeTokenizer & tokenizer,
                                       int audio_tokens, const std::string & prompt);

class TranscriptParser {
public:
    std::vector<TranscriptSegment> feed(const std::string & chunk);
    std::vector<TranscriptSegment> close();

private:
    enum class State { SeekStart, ReadStart, ExpectSpeakerOpen, ReadSpeaker, ReadText, ReadEnd, AfterEnd };

    void consume(char ch, std::vector<TranscriptSegment> & segments);
    void reset();
    void seek_start(char ch);
    void read_start(char ch);
    void expect_speaker_open(char ch);
    void read_speaker(char ch);
    void read_text(char ch);
    void read_end(char ch);
    void accept_end();
    void reject_end();
    void after_end(char ch, std::vector<TranscriptSegment> & segments);
    void emit(std::vector<TranscriptSegment> & segments);
    void restart_on_bracket(char ch);

    State state_ = State::SeekStart;
    std::string token_;
    std::string text_;
    std::string pending_after_end_;
    std::string end_token_;
    std::string speaker_;
    double start_ = 0;
    double end_ = 0;
    bool has_start_ = false;
    bool has_end_ = false;
};

std::vector<TranscriptSegment> parse_transcript(const std::string & text);
std::string strip_whitespace(const std::string & text);

} // namespace parakeet::moss::detail
