#include "moss/transcribe_bpe.h"
#include "fit_util.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <stdexcept>

namespace parakeet::moss::detail {
namespace {

using Codepoints = std::vector<uint32_t>;

struct UnicodeRange {
    uint32_t first;
    uint32_t last;
};

#include "moss/transcribe_unicode.inc"

constexpr uint32_t FIRST_PRINTABLE = 33;
constexpr uint32_t LAST_PRINTABLE = 126;
constexpr uint32_t FIRST_LATIN = 161;
constexpr uint32_t LAST_LATIN_BEFORE_SOFT_HYPHEN = 172;
constexpr uint32_t FIRST_LATIN_AFTER_SOFT_HYPHEN = 174;
constexpr uint32_t LAST_BYTE = 255;
constexpr uint32_t REMAPPED_BYTE_BASE = 256;
template <size_t Size>
bool in_ranges(uint32_t cp, const UnicodeRange (&ranges)[Size]) {
    const auto found = std::upper_bound(std::begin(ranges), std::end(ranges), cp,
            [](uint32_t value, const UnicodeRange & range) { return value < range.first; });
    return found != std::begin(ranges) && cp <= std::prev(found)->last;
}

bool is_letter(uint32_t cp) {
    return in_ranges(cp, LETTER_RANGES);
}

bool is_number(uint32_t cp) {
    return in_ranges(cp, NUMBER_RANGES);
}

bool is_space(uint32_t cp) {
    return in_ranges(cp, WHITE_SPACE_RANGES);
}

bool is_newline(uint32_t cp) {
    return cp == '\r' || cp == '\n';
}

bool is_punctuation(uint32_t cp) {
    return !is_space(cp) && !is_letter(cp) && !is_number(cp);
}

uint32_t lower(uint32_t cp) {
    return cp >= 'A' && cp <= 'Z' ? cp + ('a' - 'A') : cp;
}

std::string utf8_of(uint32_t cp) {
    std::string out;
    if (cp < 0x80) {
        out += (char) cp;
    } else if (cp < 0x800) {
        out += (char) (0xC0 | (cp >> 6));
        out += (char) (0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += (char) (0xE0 | (cp >> 12));
        out += (char) (0x80 | ((cp >> 6) & 0x3F));
        out += (char) (0x80 | (cp & 0x3F));
    } else {
        out += (char) (0xF0 | (cp >> 18));
        out += (char) (0x80 | ((cp >> 12) & 0x3F));
        out += (char) (0x80 | ((cp >> 6) & 0x3F));
        out += (char) (0x80 | (cp & 0x3F));
    }
    return out;
}

int utf8_length(unsigned char lead) {
    if (lead < 0x80) return 1;
    if ((lead >> 5) == 0x6) return 2;
    if ((lead >> 4) == 0xE) return 3;
    return 4;
}

uint32_t utf8_lead_bits(unsigned char lead, int length) {
    static constexpr unsigned char MASKS[] = {0x00, 0x7F, 0x1F, 0x0F, 0x07};
    return lead & MASKS[length];
}

uint32_t decode_codepoint(const std::string & text, size_t & i) {
    const unsigned char lead = (unsigned char) text[i];
    const int length = utf8_length(lead);
    uint32_t cp = utf8_lead_bits(lead, length);
    for (int k = 1; k < length && i + (size_t) k < text.size(); ++k) {
        cp = (cp << 6) | ((unsigned char) text[i + (size_t) k] & 0x3F);
    }
    i += (size_t) length;
    return cp;
}

Codepoints decode_utf8(const std::string & text) {
    Codepoints cps;
    for (size_t i = 0; i < text.size();) {
        cps.push_back(decode_codepoint(text, i));
    }
    return cps;
}

std::string encode_range(const Codepoints & cps, size_t first, size_t last) {
    std::string out;
    for (size_t k = first; k < last; ++k) {
        out += utf8_of(cps[k]);
    }
    return out;
}

size_t run_end(const Codepoints & cps, size_t i, bool (*keep)(uint32_t)) {
    while (i < cps.size() && keep(cps[i])) {
        ++i;
    }
    return i;
}

uint32_t lower_at(const Codepoints & cps, size_t i) {
    return i < cps.size() ? lower(cps[i]) : 0;
}

size_t contraction_end(const Codepoints & cps, size_t i) {
    if (cps[i] != '\'' || i + 1 >= cps.size()) {
        return i;
    }
    const uint32_t a = lower_at(cps, i + 1);
    if (a == 's' || a == 't' || a == 'm' || a == 'd') {
        return i + 2;
    }
    const uint32_t b = lower_at(cps, i + 2);
    const bool pair = (a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l');
    return pair && i + 2 < cps.size() ? i + 3 : i;
}

size_t word_end(const Codepoints & cps, size_t i) {
    if (is_letter(cps[i])) {
        return run_end(cps, i, is_letter);
    }
    const bool prefixed = !is_newline(cps[i]) && !is_number(cps[i]) && i + 1 < cps.size() && is_letter(cps[i + 1]);
    return prefixed ? run_end(cps, i + 1, is_letter) : i;
}

size_t punctuation_end(const Codepoints & cps, size_t i) {
    size_t j = i;
    if (cps[j] == ' ' && j + 1 < cps.size() && is_punctuation(cps[j + 1])) {
        ++j;
    }
    if (j >= cps.size() || !is_punctuation(cps[j])) {
        return i;
    }
    return run_end(cps, run_end(cps, j, is_punctuation), is_newline);
}

size_t last_newline_end(const Codepoints & cps, size_t first, size_t last) {
    size_t end = first;
    for (size_t k = first; k < last; ++k) {
        end = is_newline(cps[k]) ? k + 1 : end;
    }
    return end;
}

size_t whitespace_end(const Codepoints & cps, size_t i) {
    const size_t j = run_end(cps, i, is_space);
    const size_t newline = last_newline_end(cps, i, j);
    if (newline > i) {
        return newline;
    }
    return j < cps.size() && j - i > 1 ? j - 1 : j;
}

size_t piece_end(const Codepoints & cps, size_t i) {
    for (size_t end : {contraction_end(cps, i), word_end(cps, i)}) {
        if (end > i) return end;
    }
    if (is_number(cps[i])) {
        return i + 1;
    }
    const size_t punctuation = punctuation_end(cps, i);
    if (punctuation > i) {
        return punctuation;
    }
    return is_space(cps[i]) ? whitespace_end(cps, i) : i + 1;
}

bool is_identity_byte(uint32_t byte) {
    return (byte >= FIRST_PRINTABLE && byte <= LAST_PRINTABLE) ||
           (byte >= FIRST_LATIN && byte <= LAST_LATIN_BEFORE_SOFT_HYPHEN) ||
           (byte >= FIRST_LATIN_AFTER_SOFT_HYPHEN && byte <= LAST_BYTE);
}

std::array<std::string, 256> byte_to_symbol_table() {
    std::array<std::string, 256> table;
    uint32_t remapped = 0;
    for (uint32_t byte = 0; byte <= LAST_BYTE; ++byte) {
        table[byte] = utf8_of(is_identity_byte(byte) ? byte : REMAPPED_BYTE_BASE + remapped++);
    }
    return table;
}

std::unordered_map<std::string, int32_t> index_tokens(const std::vector<std::string> & tokens) {
    std::unordered_map<std::string, int32_t> vocab;
    vocab.reserve(tokens.size());
    for (size_t id = 0; id < tokens.size(); ++id) {
        vocab.emplace(tokens[id], (int32_t) id);
    }
    return vocab;
}

std::unordered_map<std::string, int> rank_merges(const std::vector<std::string> & merges) {
    std::unordered_map<std::string, int> ranks;
    ranks.reserve(merges.size());
    for (size_t rank = 0; rank < merges.size(); ++rank) {
        ranks.emplace(merges[rank], (int) rank);
    }
    return ranks;
}

struct BestMerge {
    size_t index = 0;
    int rank = std::numeric_limits<int>::max();
};

BestMerge best_merge(const std::vector<std::string> & word, const std::unordered_map<std::string, int> & ranks) {
    BestMerge best;
    for (size_t i = 0; i + 1 < word.size(); ++i) {
        const auto found = ranks.find(word[i] + " " + word[i + 1]);
        if (found != ranks.end() && found->second < best.rank) {
            best = {i, found->second};
        }
    }
    return best;
}

std::vector<std::string> byte_symbols_of(const std::string & piece, const std::array<std::string, 256> & table) {
    std::vector<std::string> word;
    for (unsigned char byte : piece) {
        word.push_back(table[byte]);
    }
    return word;
}

} // namespace

std::vector<std::string> qwen_pretokenize(const std::string & text) {
    const Codepoints cps = decode_utf8(text);
    std::vector<std::string> pieces;
    for (size_t i = 0; i < cps.size();) {
        const size_t end = piece_end(cps, i);
        pieces.push_back(encode_range(cps, i, end));
        i = end;
    }
    return pieces;
}

QwenByteBpe::QwenByteBpe(const std::vector<std::string> & tokens, const std::vector<std::string> & merges)
    : byte_symbols_(byte_to_symbol_table()), vocab_(index_tokens(tokens)), merge_rank_(rank_merges(merges)) {}

const std::array<std::string, 256> & QwenByteBpe::byte_symbols() const {
    return byte_symbols_;
}

std::vector<std::string> QwenByteBpe::merge_word(const std::string & piece) const {
    std::vector<std::string> word = byte_symbols_of(piece, byte_symbols_);
    for (BestMerge best = best_merge(word, merge_rank_); best.rank != std::numeric_limits<int>::max();
         best = best_merge(word, merge_rank_)) {
        word[best.index] += word[best.index + 1];
        word.erase(word.begin() + (std::ptrdiff_t) best.index + 1);
    }
    return word;
}

void QwenByteBpe::append_symbol(const std::string & symbol, std::vector<int32_t> & ids) const {
    const auto found = vocab_.find(symbol);
    if (found == vocab_.end()) {
        throw std::runtime_error("moss transcribe: tokenizer vocabulary has no symbol for '" + symbol + "'");
    }
    ids.push_back(found->second);
}

void QwenByteBpe::append_piece(const std::string & piece, std::vector<int32_t> & ids) const {
    for (const std::string & symbol : merge_word(piece)) {
        append_symbol(symbol, ids);
    }
}

std::vector<int32_t> QwenByteBpe::encode(const std::string & text) const {
    std::vector<int32_t> ids;
    for (const std::string & piece : qwen_pretokenize(text)) {
        append_piece(piece, ids);
    }
    return ids;
}

namespace {

template <typename Map>
uint64_t map_storage_bytes(const Map & map) {
    uint64_t bytes = fitutil::sat_mul(map.bucket_count(), sizeof(void *));
    for (const auto & entry : map) {
        bytes = fitutil::sat_add(bytes, sizeof(entry) + sizeof(void *) * 2 + entry.first.capacity() + 1);
    }
    return bytes;
}

}

uint64_t QwenByteBpe::storage_bytes() const {
    return fitutil::sat_add(sizeof(*this), fitutil::sat_add(map_storage_bytes(vocab_), map_storage_bytes(merge_rank_)));
}

} // namespace parakeet::moss::detail
