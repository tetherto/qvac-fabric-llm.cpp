// BPE detokenize(): merges pieces, SentencePiece space markers, strips specials and locale tags.

#include "sentencepiece_bpe.h"

#include <algorithm>
#include <string>

namespace parakeet {

namespace {

constexpr size_t k_min_language_subtag_chars = 2;
constexpr size_t k_max_language_subtag_chars = 3;
constexpr size_t k_min_region_subtag_chars   = 2;
constexpr size_t k_max_region_subtag_chars   = 4;

bool is_locale_tag_id(const BpeVocab & vocab, int32_t id) {
    return std::binary_search(vocab.locale_tag_ids.begin(), vocab.locale_tag_ids.end(), id);
}

bool is_special_or_invalid(const BpeVocab & vocab, int32_t id) {
    if (id < 0 || id >= static_cast<int32_t>(vocab.pieces.size())) return true;
    return id == vocab.blank_id || id == vocab.bos_id || id == vocab.eos_id || id == vocab.pad_id ||
           is_locale_tag_id(vocab, id);
}

bool is_ascii_letter(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

bool is_ascii_alnum(char c) {
    return is_ascii_letter(c) || (c >= '0' && c <= '9');
}

bool is_subtag(const std::string & text, size_t min_chars, size_t max_chars, bool (*accept)(char)) {
    return text.size() >= min_chars && text.size() <= max_chars &&
           std::all_of(text.begin(), text.end(), accept);
}

bool is_locale_body(const std::string & body) {
    const size_t dash = body.find('-');
    if (dash == std::string::npos) return false;
    return is_subtag(body.substr(0, dash), k_min_language_subtag_chars,
                     k_max_language_subtag_chars, is_ascii_letter) &&
           is_subtag(body.substr(dash + 1), k_min_region_subtag_chars,
                     k_max_region_subtag_chars, is_ascii_alnum);
}

void append_piece(const std::string & piece, std::string & out) {
    for (size_t i = 0; i < piece.size(); ) {
        const unsigned char c0 = static_cast<unsigned char>(piece[i]);
        if (c0 == 0xE2 && i + 2 < piece.size() &&
            static_cast<unsigned char>(piece[i+1]) == 0x96 &&
            static_cast<unsigned char>(piece[i+2]) == 0x81) {
            out.push_back(' ');
            i += 3;
        } else {
            out.push_back(piece[i]);
            ++i;
        }
    }
}

}

void append_token_pieces(const BpeVocab & vocab,
                         const std::vector<int32_t> & token_ids,
                         std::string & pieces) {
    for (int32_t id : token_ids) {
        if (is_special_or_invalid(vocab, id)) continue;
        append_piece(vocab.pieces[id], pieces);
    }
}

std::string strip_leading_spaces(const std::string & text) {
    size_t start = 0;
    while (start < text.size() && text[start] == ' ') ++start;
    return text.substr(start);
}

std::string collapse_spaces(const std::string & text) {
    std::string out;
    out.reserve(text.size());
    for (const char c : text) {
        if (c == ' ' && (out.empty() || out.back() == ' ')) continue;
        out.push_back(c);
    }
    if (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

std::string pieces_to_text(const BpeVocab & vocab, const std::string & pieces) {
    return vocab.locale_tag_ids.empty() ? strip_leading_spaces(pieces) : collapse_spaces(pieces);
}

bool piece_is_locale_tag(const std::string & piece) {
    if (piece.size() < 2 || piece.front() != '<' || piece.back() != '>') return false;
    return is_locale_body(piece.substr(1, piece.size() - 2));
}

std::vector<int32_t> find_locale_tag_ids(const BpeVocab & vocab) {
    std::vector<int32_t> ids;
    for (size_t id = 0; id < vocab.pieces.size(); ++id) {
        if (piece_is_locale_tag(vocab.pieces[id])) ids.push_back(static_cast<int32_t>(id));
    }
    return ids;
}

bool tokens_start_word(const BpeVocab & vocab,
                       const std::vector<int32_t> & token_ids) {
    const auto first_text_token = std::find_if(
        token_ids.begin(), token_ids.end(),
        [&](int32_t id) { return !is_special_or_invalid(vocab, id); });
    return first_text_token == token_ids.end() || token_is_word_start(vocab, *first_text_token);
}

std::string detokenize(const BpeVocab & vocab,
                       const std::vector<int32_t> & token_ids) {
    std::string out;
    out.reserve(token_ids.size() * 3);
    append_token_pieces(vocab, token_ids, out);
    return pieces_to_text(vocab, out);
}

bool token_is_word_start(const BpeVocab & vocab, int32_t token_id) {
    if (is_special_or_invalid(vocab, token_id)) return false;
    const std::string & piece = vocab.pieces[token_id];
    if (piece.size() < 3) return false;
    const unsigned char c0 = static_cast<unsigned char>(piece[0]);
    const unsigned char c1 = static_cast<unsigned char>(piece[1]);
    const unsigned char c2 = static_cast<unsigned char>(piece[2]);
    return c0 == 0xE2 && c1 == 0x96 && c2 == 0x81;
}

}
