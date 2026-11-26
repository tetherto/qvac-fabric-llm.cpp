#include "moss/transcribe_text.h"
#include "fit_util.h"

#include "moss/transcribe_bpe.h"

#include <algorithm>
#include <locale>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace parakeet::moss::detail {
namespace {

constexpr int32_t TOKEN_TYPE_CONTROL = 3;
constexpr size_t MAX_TIMESTAMP_CHARS = 32;
constexpr size_t MAX_SPEAKER_CHARS = 16;
constexpr int DECIMAL_DIGITS = 6;
constexpr const char * WHITESPACE = " \t\n\v\f\r";

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("moss transcribe: " + message);
}

bool is_space(char ch) {
    return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\v' || ch == '\f' || ch == '\r';
}

bool is_digit(char ch) {
    return ch >= '0' && ch <= '9';
}

bool is_timestamp_char(char ch) {
    return is_digit(ch) || ch == '.';
}

bool is_speaker_char(char ch) {
    return ch == 'S' || is_digit(ch);
}

size_t count_char(const std::string & text, char wanted) {
    size_t count = 0;
    for (char ch : text) {
        count += ch == wanted ? 1 : 0;
    }
    return count;
}

bool all_timestamp_chars(const std::string & text) {
    for (char ch : text) {
        if (!is_timestamp_char(ch)) {
            return false;
        }
    }
    return true;
}

bool all_digits(const std::string & text) {
    for (char ch : text) {
        if (!is_digit(ch)) {
            return false;
        }
    }
    return true;
}

bool parse_timestamp(const std::string & text, double & value) {
    if (text.empty() || !all_timestamp_chars(text) || count_char(text, '.') > 1 || count_char(text, '.') == text.size()) {
        return false;
    }
    value = parse_decimal(text);
    return true;
}

bool is_speaker(const std::string & text) {
    return text.size() >= 2 && text[0] == 'S' && all_digits(text.substr(1));
}

size_t utf8_length(unsigned char lead) {
    if (lead < 0x80) return 1;
    if ((lead >> 5) == 0x6) return 2;
    if ((lead >> 4) == 0xE) return 3;
    if ((lead >> 3) == 0x1E) return 4;
    return 1;
}

using ByteMap = std::unordered_map<std::string, char>;

ByteMap reverse_byte_map(const QwenByteBpe & tokenizer) {
    ByteMap map;
    for (int byte = 0; byte < 256; ++byte) {
        map.emplace(tokenizer.byte_symbols()[(size_t) byte], (char) byte);
    }
    return map;
}

std::string bytes_of(const std::string & token, const ByteMap & map) {
    std::string bytes;
    for (size_t i = 0; i < token.size();) {
        const size_t length = std::min(utf8_length((unsigned char) token[i]), token.size() - i);
        const std::string symbol = token.substr(i, length);
        const auto found = map.find(symbol);
        bytes += found != map.end() ? std::string(1, found->second) : symbol;
        i += length;
    }
    return bytes;
}

void append(std::vector<int32_t> & ids, const std::vector<int32_t> & more) {
    ids.insert(ids.end(), more.begin(), more.end());
}

void append_pads(std::vector<int32_t> & span, int32_t pad, int count) {
    span.insert(span.end(), (size_t) std::max(count, 0), pad);
}

std::vector<int32_t> digit_ids(const TranscribeTokenizer & tokenizer) {
    std::vector<int32_t> ids;
    for (char digit = '0'; digit <= '9'; ++digit) {
        const std::vector<int32_t> encoded = tokenizer.encode(std::string(1, digit));
        if (encoded.size() != 1) {
            fail("digit is not a single token in the vocabulary");
        }
        ids.push_back(encoded[0]);
    }
    return ids;
}

void append_marker(std::vector<int32_t> & span, int seconds, const std::vector<int32_t> & digits) {
    for (char ch : std::to_string(seconds)) {
        span.push_back(digits[(size_t) (ch - '0')]);
    }
}

int append_marked_pads(std::vector<int32_t> & span, int32_t pad, int every, int per_marker, int duration,
                       const std::vector<int32_t> & digits) {
    int consumed = 0;
    for (int seconds = every; seconds <= duration; seconds += every) {
        const int position = (seconds / every) * per_marker;
        append_pads(span, pad, position - consumed);
        consumed = std::max(consumed, position);
        append_marker(span, seconds, digits);
    }
    return consumed;
}

std::vector<int32_t> prompt_body(const TranscribeConfig & config, const TranscribeTokenizer & tokenizer,
                                 const std::string & prompt) {
    if (!strip_whitespace(prompt).empty()) {
        return tokenizer.encode("\n" + prompt);
    }
    std::vector<int32_t> body = tokenizer.encode("\n");
    append(body, config.default_prompt_ids);
    return body;
}

} // namespace

struct TranscribeTokenizer::Impl {
    QwenByteBpe tokenizer;
    std::vector<std::string> pieces;

    Impl(const std::vector<std::string> & tokens, const std::vector<std::string> & merges)
        : tokenizer(tokens, merges) {}

    void build_pieces(const std::vector<std::string> & tokens, const std::vector<int32_t> & types) {
        const ByteMap map = reverse_byte_map(tokenizer);
        pieces.resize(tokens.size());
        for (size_t id = 0; id < tokens.size(); ++id) {
            pieces[id] = piece(tokens[id], id < types.size() ? types[id] : 0, map);
        }
    }

    static std::string piece(const std::string & token, int32_t type, const ByteMap & map) {
        if (type == TOKEN_TYPE_CONTROL) {
            return {};
        }
        return type == 1 || type == 0 ? bytes_of(token, map) : token;
    }
};

TranscribeTokenizer::TranscribeTokenizer(const std::vector<std::string> & tokens,
                                         const std::vector<std::string> & merges,
                                         const std::vector<int32_t> & types) {
    if (tokens.empty() || merges.empty()) {
        fail("GGUF is missing the tokenizer vocabulary");
    }
    impl_ = std::make_unique<Impl>(tokens, merges);
    impl_->build_pieces(tokens, types);
}

TranscribeTokenizer::TranscribeTokenizer(const TranscribeModel & model)
    : TranscribeTokenizer(model.tokenizer_tokens(), model.tokenizer_merges(), model.tokenizer_types()) {}

TranscribeTokenizer::~TranscribeTokenizer() = default;

std::vector<int32_t> TranscribeTokenizer::encode(const std::string & text) const {
    return impl_->tokenizer.encode(text);
}

std::string TranscribeTokenizer::decode(const std::vector<int32_t> & ids) const {
    std::string text;
    for (int32_t id : ids) {
        if (id >= 0 && (size_t) id < impl_->pieces.size()) {
            text += impl_->pieces[(size_t) id];
        }
    }
    return text;
}

uint64_t TranscribeTokenizer::storage_bytes() const {
    uint64_t bytes = fitutil::sat_add(sizeof(Impl), impl_->tokenizer.storage_bytes());
    bytes = fitutil::sat_add(bytes, impl_->pieces.capacity() * sizeof(std::string));
    for (const auto & piece : impl_->pieces) bytes = fitutil::sat_add(bytes, piece.capacity() + 1);
    return bytes;
}

size_t TranscribeTokenizer::max_piece_bytes() const {
    size_t bytes = 0;
    for (const auto & piece : impl_->pieces) bytes = std::max(bytes, piece.size());
    return bytes;
}

std::vector<int32_t> transcribe_audio_span(const TranscribeConfig & config, const TranscribeTokenizer & tokenizer,
                                           int audio_tokens) {
    std::vector<int32_t> span;
    const int32_t pad = config.tokens.audio_pad;
    const int every = config.time_marker_every_seconds;
    const int per_marker = (int) (config.audio_tokens_per_second * (float) every);
    if (!config.time_markers || audio_tokens <= 0 || every <= 0 || per_marker <= 0) {
        append_pads(span, pad, audio_tokens);
        return span;
    }
    const int duration = (int) ((double) audio_tokens / (double) config.audio_tokens_per_second);
    const int consumed = append_marked_pads(span, pad, every, per_marker, duration, digit_ids(tokenizer));
    append_pads(span, pad, audio_tokens - consumed);
    return span;
}

std::vector<int32_t> transcribe_prompt(const TranscribeConfig & config, const TranscribeTokenizer & tokenizer,
                                       int audio_tokens, const std::string & prompt) {
    const TranscribeTokens & tokens = config.tokens;
    std::vector<int32_t> ids{tokens.im_start};
    append(ids, tokenizer.encode("system\n" + config.system_prompt));
    ids.push_back(tokens.im_end);
    append(ids, tokenizer.encode("\n"));
    ids.push_back(tokens.im_start);
    append(ids, tokenizer.encode("user\n"));
    ids.push_back(tokens.audio_start);
    append(ids, transcribe_audio_span(config, tokenizer, audio_tokens));
    ids.push_back(tokens.audio_end);
    append(ids, prompt_body(config, tokenizer, prompt));
    ids.push_back(tokens.im_end);
    append(ids, tokenizer.encode("\n"));
    ids.push_back(tokens.im_start);
    append(ids, tokenizer.encode("assistant\n"));
    return ids;
}

namespace {

bool is_hotword_breaker(unsigned char ch) {
    return ch < 0x20 || ch == 0x7F || ch == '[' || ch == ']' || ch == '<' || ch == '>' || ch == '|';
}

std::string collapse_spaces(const std::string & text) {
    std::string out;
    for (char ch : text) {
        if (ch != ' ' || (!out.empty() && out.back() != ' ')) {
            out.push_back(ch);
        }
    }
    return out;
}

std::string replace_breakers(const std::string & text) {
    std::string out = text;
    for (char & ch : out) {
        ch = is_hotword_breaker((unsigned char) ch) ? ' ' : ch;
    }
    return out;
}

bool contains(const std::vector<std::string> & values, const std::string & value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

void append_unique(std::vector<std::string> & values, const std::string & value) {
    if (!value.empty() && !contains(values, value)) {
        values.push_back(value);
    }
}

std::string join(const std::vector<std::string> & values, const std::string & separator) {
    std::string out;
    for (size_t i = 0; i < values.size(); ++i) {
        out += (i == 0 ? "" : separator) + values[i];
    }
    return out;
}

} // namespace

std::string sanitize_hotword(const std::string & hotword) {
    const std::string clean = strip_whitespace(collapse_spaces(replace_breakers(hotword)));
    if (clean.size() > MAX_HOTWORD_BYTES) {
        fail("each hotword must be at most " + std::to_string(MAX_HOTWORD_BYTES) + " bytes");
    }
    return clean;
}

std::vector<std::string> sanitize_hotwords(const std::vector<std::string> & hotwords) {
    if (hotwords.size() > MAX_HOTWORDS) {
        fail("at most " + std::to_string(MAX_HOTWORDS) + " hotwords are allowed");
    }
    std::vector<std::string> clean;
    for (const std::string & hotword : hotwords) {
        append_unique(clean, sanitize_hotword(hotword));
    }
    return clean;
}

std::string hotword_prompt(const TranscribeConfig & config, const std::vector<std::string> & hotwords) {
    return config.default_prompt + config.hotword_prefix + join(hotwords, config.hotword_separator);
}

std::string strip_whitespace(const std::string & text) {
    const size_t first = text.find_first_not_of(WHITESPACE);
    if (first == std::string::npos) {
        return {};
    }
    return text.substr(first, text.find_last_not_of(WHITESPACE) - first + 1);
}

void TranscriptParser::reset() {
    state_ = State::SeekStart;
    token_.clear();
    text_.clear();
    pending_after_end_.clear();
    end_token_.clear();
    speaker_.clear();
    has_start_ = false;
    has_end_ = false;
}

void TranscriptParser::restart_on_bracket(char ch) {
    reset();
    if (ch == '[') {
        state_ = State::ReadStart;
    }
}

void TranscriptParser::seek_start(char ch) {
    if (ch == '[') {
        token_.clear();
        state_ = State::ReadStart;
    }
}

void TranscriptParser::read_start(char ch) {
    if (ch == ']') {
        double start = 0;
        if (!parse_timestamp(token_, start)) {
            reset();
            return;
        }
        start_ = start;
        has_start_ = true;
        state_ = State::ExpectSpeakerOpen;
        token_.clear();
        return;
    }
    if (is_timestamp_char(ch)) {
        token_.push_back(ch);
        if (token_.size() <= MAX_TIMESTAMP_CHARS) {
            return;
        }
    }
    restart_on_bracket(ch);
}

void TranscriptParser::expect_speaker_open(char ch) {
    if (ch == '[') {
        token_.clear();
        state_ = State::ReadSpeaker;
    } else if (!is_space(ch)) {
        reset();
    }
}

void TranscriptParser::read_speaker(char ch) {
    if (ch == ']') {
        if (!is_speaker(token_)) {
            reset();
            return;
        }
        speaker_ = token_;
        text_.clear();
        state_ = State::ReadText;
        token_.clear();
        return;
    }
    if (is_speaker_char(ch)) {
        token_.push_back(ch);
        if (token_.size() <= MAX_SPEAKER_CHARS) {
            return;
        }
    }
    restart_on_bracket(ch);
}

void TranscriptParser::read_text(char ch) {
    if (ch == '[') {
        token_.clear();
        state_ = State::ReadEnd;
    } else {
        text_.push_back(ch);
    }
}

void TranscriptParser::accept_end() {
    end_token_ = token_;
    pending_after_end_.clear();
    has_end_ = true;
    state_ = State::AfterEnd;
}

void TranscriptParser::reject_end() {
    text_ += "[" + token_ + "]";
    state_ = State::ReadText;
}

void TranscriptParser::read_end(char ch) {
    if (ch == ']') {
        double end = 0;
        if (parse_timestamp(token_, end) && has_start_ && end >= start_) {
            end_ = end;
            accept_end();
        } else {
            reject_end();
        }
        token_.clear();
        return;
    }
    if (is_timestamp_char(ch)) {
        token_.push_back(ch);
        if (token_.size() <= MAX_TIMESTAMP_CHARS) {
            return;
        }
    }
    text_ += "[" + token_ + std::string(1, ch);
    token_.clear();
    state_ = State::ReadText;
}

void TranscriptParser::after_end(char ch, std::vector<TranscriptSegment> & segments) {
    if (ch == '[') {
        emit(segments);
        token_.clear();
        state_ = State::ReadStart;
        return;
    }
    if (is_space(ch)) {
        pending_after_end_.push_back(ch);
        return;
    }
    text_ += "[" + end_token_ + "]" + pending_after_end_ + std::string(1, ch);
    pending_after_end_.clear();
    has_end_ = false;
    end_token_.clear();
    state_ = State::ReadText;
}

void TranscriptParser::emit(std::vector<TranscriptSegment> & segments) {
    if (!has_start_ || !has_end_ || speaker_.empty()) {
        reset();
        return;
    }
    const std::string text = strip_whitespace(text_);
    if (!text.empty()) {
        segments.push_back({start_, end_, speaker_, text});
    }
    reset();
}

void TranscriptParser::consume(char ch, std::vector<TranscriptSegment> & segments) {
    switch (state_) {
        case State::SeekStart:         seek_start(ch); break;
        case State::ReadStart:         read_start(ch); break;
        case State::ExpectSpeakerOpen: expect_speaker_open(ch); break;
        case State::ReadSpeaker:       read_speaker(ch); break;
        case State::ReadText:          read_text(ch); break;
        case State::ReadEnd:           read_end(ch); break;
        case State::AfterEnd:          after_end(ch, segments); break;
    }
}

std::vector<TranscriptSegment> TranscriptParser::feed(const std::string & chunk) {
    std::vector<TranscriptSegment> segments;
    for (char ch : chunk) {
        consume(ch, segments);
    }
    return segments;
}

std::vector<TranscriptSegment> TranscriptParser::close() {
    std::vector<TranscriptSegment> segments;
    if (state_ == State::AfterEnd) {
        emit(segments);
    }
    reset();
    return segments;
}

std::vector<TranscriptSegment> parse_transcript(const std::string & text) {
    TranscriptParser parser;
    std::vector<TranscriptSegment> segments = parser.feed(text);
    const std::vector<TranscriptSegment> tail = parser.close();
    segments.insert(segments.end(), tail.begin(), tail.end());
    return segments;
}

double parse_decimal(const std::string & text) {
    std::istringstream stream(text);
    stream.imbue(std::locale::classic());
    double value = 0.0;
    stream >> value;
    return value;
}

std::string format_decimal(double value) {
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream.precision(DECIMAL_DIGITS);
    stream << value;
    return stream.str();
}

} // namespace parakeet::moss::detail
