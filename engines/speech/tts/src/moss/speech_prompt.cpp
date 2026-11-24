#include "moss/speech_prompt.h"

#include "qwen_tokenizer.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace tts_cpp::moss::detail {
namespace {

constexpr int32_t TOKEN_TYPE_CONTROL = 3;
constexpr int32_t TOKEN_TYPE_NORMAL = 1;
const char * const EMPTY_MARK = "<|empty|>";
const char * const END_EMPTY_MARK = "<|end_empty|>";
const char * const EMPTY_REPLACEMENT = ".";
const char * const END_EMPTY_REPLACEMENT = ":";

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("moss speech: " + message);
}

using ByteMap = std::unordered_map<std::string, char>;

ByteMap reverse_byte_map(const QwenTokenizer & tokenizer) {
    ByteMap map;
    for (int byte = 0; byte < 256; ++byte) {
        map.emplace(tokenizer.byte2u[byte], (char) byte);
    }
    return map;
}

size_t utf8_length(unsigned char lead) {
    if (lead < 0x80) return 1;
    if ((lead >> 5) == 0x6) return 2;
    if ((lead >> 4) == 0xE) return 3;
    if ((lead >> 3) == 0x1E) return 4;
    return 1;
}

std::string bytes_of(const std::string & token, const ByteMap & map) {
    std::string bytes;
    for (size_t i = 0; i < token.size();) {
        const size_t length = std::min(utf8_length((unsigned char) token[i]), token.size() - i);
        const auto found = map.find(token.substr(i, length));
        bytes += found != map.end() ? std::string(1, found->second) : token.substr(i, length);
        i += length;
    }
    return bytes;
}

std::string piece_of(const std::string & token, int32_t type, const ByteMap & map) {
    if (type == TOKEN_TYPE_CONTROL) {
        return {};
    }
    return type == TOKEN_TYPE_NORMAL || type == 0 ? bytes_of(token, map) : token;
}

void load_vocabulary(QwenTokenizer & tokenizer, const std::vector<std::string> & tokens) {
    tokenizer.vocab.reserve(tokens.size());
    for (size_t id = 0; id < tokens.size(); ++id) {
        tokenizer.vocab.emplace(tokens[id], (int) id);
    }
}

void load_merges(QwenTokenizer & tokenizer, const std::vector<std::string> & merges) {
    tokenizer.merge_rank.reserve(merges.size());
    for (size_t rank = 0; rank < merges.size(); ++rank) {
        tokenizer.merge_rank.emplace(merges[rank], (int) rank);
    }
}

void replace_all(std::string & text, const std::string & from, const std::string & to) {
    for (size_t at = text.find(from); at != std::string::npos; at = text.find(from, at + to.size())) {
        text.replace(at, from.size(), to);
    }
}

void append(std::vector<SpeechRow> & rows, const std::vector<SpeechRow> & more) {
    rows.insert(rows.end(), more.begin(), more.end());
}

const char * role_name(SpeechTurnRole role) {
    switch (role) {
        case SpeechTurnRole::System:    return "system";
        case SpeechTurnRole::Assistant: return "assistant";
        case SpeechTurnRole::User:      break;
    }
    return "user";
}

std::vector<SpeechRow> turn_header(const SpeechLmConfig & config, const SpeechTextTokenizer & tokenizer,
                                   SpeechTurnRole role) {
    std::vector<int32_t> ids{config.tokens.im_start};
    const std::vector<int32_t> name = tokenizer.encode(std::string(role_name(role)) + "\n");
    ids.insert(ids.end(), name.begin(), name.end());
    return speech_text_rows(ids, config.tokens);
}

std::vector<SpeechRow> turn_footer(const SpeechLmConfig & config, const SpeechTextTokenizer & tokenizer) {
    std::vector<int32_t> ids{config.tokens.im_end};
    const std::vector<int32_t> newline = tokenizer.encode("\n");
    ids.insert(ids.end(), newline.begin(), newline.end());
    return speech_text_rows(ids, config.tokens);
}

std::vector<SpeechRow> turn_rows(const SpeechLmConfig & config, const SpeechTextTokenizer & tokenizer,
                                 const SpeechTurn & turn) {
    std::vector<SpeechRow> rows = turn_header(config, tokenizer, turn.role);
    append(rows, turn.is_audio ? speech_audio_rows(turn.audio_codes, config.tokens)
                               : speech_text_rows(tokenizer.encode(turn.text), config.tokens));
    append(rows, turn_footer(config, tokenizer));
    return rows;
}

std::vector<SpeechRow> conversation_rows(const SpeechLmConfig & config, const SpeechTextTokenizer & tokenizer,
                                         const std::vector<SpeechTurn> & turns) {
    std::vector<SpeechRow> rows;
    for (const SpeechTurn & turn : turns) {
        append(rows, turn_rows(config, tokenizer, turn));
    }
    return rows;
}

std::vector<SpeechRow> reply_prefix(const SpeechLmConfig & config, const SpeechTextTokenizer & tokenizer,
                                    bool audio_reply) {
    std::vector<SpeechRow> rows = turn_header(config, tokenizer, SpeechTurnRole::Assistant);
    if (audio_reply) {
        rows.push_back({config.tokens.speech_start, config.tokens.audio_pad});
    }
    return rows;
}

std::vector<int32_t> text_channel(const std::vector<SpeechRow> & rows) {
    std::vector<int32_t> ids;
    for (const SpeechRow & row : rows) {
        ids.push_back(row.text);
    }
    return ids;
}

} // namespace

struct SpeechTextTokenizer::Impl {
    mutable QwenTokenizer tokenizer;
    std::vector<std::string> pieces;

    void build_pieces(const std::vector<std::string> & tokens, const std::vector<int32_t> & types) {
        const ByteMap map = reverse_byte_map(tokenizer);
        pieces.resize(tokens.size());
        for (size_t id = 0; id < tokens.size(); ++id) {
            pieces[id] = piece_of(tokens[id], id < types.size() ? types[id] : 0, map);
        }
    }
};

SpeechTextTokenizer::SpeechTextTokenizer(const std::vector<std::string> & tokens,
                                         const std::vector<std::string> & merges,
                                         const std::vector<int32_t> & types) : impl_(new Impl) {
    if (tokens.empty() || merges.empty()) {
        fail("GGUF is missing the tokenizer vocabulary");
    }
    impl_->tokenizer.build_byte_map();
    load_vocabulary(impl_->tokenizer, tokens);
    load_merges(impl_->tokenizer, merges);
    impl_->build_pieces(tokens, types);
}

SpeechTextTokenizer::~SpeechTextTokenizer() = default;

std::vector<int32_t> SpeechTextTokenizer::encode(const std::string & text) const {
    if (text.empty()) {
        return {};
    }
    const std::vector<int> ids = impl_->tokenizer.encode(text);
    return std::vector<int32_t>(ids.begin(), ids.end());
}

std::string SpeechTextTokenizer::decode(const std::vector<int32_t> & ids) const {
    std::string text;
    for (int32_t id : ids) {
        if (id >= 0 && (size_t) id < impl_->pieces.size()) {
            text += impl_->pieces[(size_t) id];
        }
    }
    return text;
}

std::vector<SpeechRow> speech_text_rows(const std::vector<int32_t> & ids, const SpeechTokens & tokens) {
    std::vector<SpeechRow> rows;
    for (int32_t id : ids) {
        rows.push_back({id, tokens.audio_pad});
    }
    return rows;
}

std::vector<SpeechRow> speech_audio_rows(const std::vector<int32_t> & codes, const SpeechTokens & tokens) {
    std::vector<SpeechRow> rows{{tokens.speech_start, tokens.audio_pad}};
    for (int32_t code : codes) {
        rows.push_back({tokens.text_placeholder, code});
    }
    rows.push_back({tokens.text_placeholder, tokens.speech_end});
    return rows;
}

std::vector<SpeechRow> speech_prompt(const SpeechLmConfig & config, const SpeechTextTokenizer & tokenizer,
                                     const std::vector<SpeechTurn> & turns, bool audio_reply) {
    if (turns.empty()) {
        fail("the conversation needs at least one turn");
    }
    std::vector<SpeechRow> rows = conversation_rows(config, tokenizer, turns);
    if (turns.front().role != SpeechTurnRole::System) {
        SpeechTurn system{SpeechTurnRole::System, audio_reply ? config.audio_system_prompt : config.text_system_prompt, {},
                          false};
        append(rows, turn_rows(config, tokenizer, system));
    }
    append(rows, reply_prefix(config, tokenizer, audio_reply));
    return rows;
}

bool is_speech_stop_row(const SpeechRow & row, const SpeechTokens & tokens) {
    return row.text == tokens.pad || row.text == tokens.im_end;
}

size_t speech_reply_rows(const std::vector<SpeechRow> & generated, const SpeechTokens & tokens) {
    const bool ends_on_stop = !generated.empty() && is_speech_stop_row(generated.back(), tokens);
    return generated.size() - (ends_on_stop ? 1 : 0);
}

std::string speech_reply_text(const SpeechTextTokenizer & tokenizer, const std::vector<SpeechRow> & generated,
                              const SpeechTokens & tokens) {
    const std::vector<SpeechRow> kept(generated.begin(),
            generated.begin() + (std::ptrdiff_t) speech_reply_rows(generated, tokens));
    std::string text = tokenizer.decode(text_channel(kept));
    replace_all(text, EMPTY_MARK, EMPTY_REPLACEMENT);
    replace_all(text, END_EMPTY_MARK, END_EMPTY_REPLACEMENT);
    return text;
}

} // namespace tts_cpp::moss::detail
