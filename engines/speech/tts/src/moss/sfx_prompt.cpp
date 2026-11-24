#include "moss/sfx_prompt.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace tts_cpp::moss::detail {
namespace {

constexpr int HTML_UNESCAPE_PASSES = 2;
constexpr int MAX_FIXED_POINT_PASSES = 16;
constexpr int TENTHS_PER_SECOND = 10;
constexpr double MAX_ROUNDED_SECONDS = 1e6;
constexpr int ROUNDING_BUFFER = 32;
constexpr size_t MAX_ENTITY_DIGITS = 8;
constexpr uint32_t REPLACEMENT = 0xFFFD;
constexpr uint32_t MAX_CODE_POINT = 0x10FFFF;
constexpr uint32_t SURROGATE_FIRST = 0xD800;
constexpr uint32_t SURROGATE_LAST = 0xDFFF;
constexpr uint32_t FULLWIDTH_FIRST = 0xFF01;
constexpr uint32_t FULLWIDTH_LAST = 0xFF5E;
constexpr uint32_t FULLWIDTH_OFFSET = 0xFEE0;
constexpr uint32_t IDEOGRAPHIC_SPACE = 0x3000;
constexpr char HTML_TAG_OPEN = '<';

struct NamedEntity {
    const char * name;
    const char * text;
};

constexpr NamedEntity NAMED_ENTITIES[] = {
    {"&amp;", "&"}, {"&lt;", "<"}, {"&gt;", ">"}, {"&quot;", "\""}, {"&apos;", "'"}, {"&nbsp;", " "},
};

struct CodePointMapping {
    uint32_t code;
    const char * text;
};

constexpr CodePointMapping TYPOGRAPHIC_MAPPINGS[] = {
    {0x2018, "'"}, {0x2019, "'"}, {0x201A, "'"}, {0x201B, "'"},
    {0x201C, "\""}, {0x201D, "\""}, {0x201E, "\""}, {0x201F, "\""},
    {0xFB00, "ff"}, {0xFB01, "fi"}, {0xFB02, "fl"}, {0xFB03, "ffi"}, {0xFB04, "ffl"},
    {0xFB05, "st"}, {0xFB06, "st"},
};

constexpr uint32_t UNICODE_SPACES[] = {
    0x1C, 0x1D, 0x1E, 0x1F, 0x85, 0xA0, 0x1680, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000,
};
constexpr uint32_t EN_QUAD = 0x2000;
constexpr uint32_t HAIR_SPACE = 0x200A;
constexpr uint32_t ASCII_CONTROL_SPACE_FIRST = 0x09;
constexpr uint32_t ASCII_CONTROL_SPACE_LAST = 0x0D;
constexpr size_t MAX_SEQUENCE_LENGTH = 4;
constexpr uint32_t SMALLEST_FOR_LENGTH[MAX_SEQUENCE_LENGTH + 1] = {0, 0, 0x80, 0x800, 0x10000};

std::string encode_utf8(uint32_t code) {
    std::string out;
    if (code < 0x80) {
        out += (char) code;
    } else if (code < 0x800) {
        out += (char) (0xC0 | (code >> 6));
        out += (char) (0x80 | (code & 0x3F));
    } else if (code < 0x10000) {
        out += (char) (0xE0 | (code >> 12));
        out += (char) (0x80 | ((code >> 6) & 0x3F));
        out += (char) (0x80 | (code & 0x3F));
    } else {
        out += (char) (0xF0 | (code >> 18));
        out += (char) (0x80 | ((code >> 12) & 0x3F));
        out += (char) (0x80 | ((code >> 6) & 0x3F));
        out += (char) (0x80 | (code & 0x3F));
    }
    return out;
}

size_t sequence_length(unsigned char lead) {
    if (lead < 0x80) return 1;
    if ((lead >> 5) == 0x6) return 2;
    if ((lead >> 4) == 0xE) return 3;
    if ((lead >> 3) == 0x1E) return 4;
    return 0;
}

bool continuation_bytes_valid(const std::string & text, size_t pos, size_t length) {
    if (pos + length > text.size()) {
        return false;
    }
    for (size_t i = 1; i < length; ++i) {
        if (((unsigned char) text[pos + i] >> 6) != 0x2) {
            return false;
        }
    }
    return true;
}

bool is_well_formed(uint32_t code, size_t length) {
    return code >= SMALLEST_FOR_LENGTH[length] && code <= MAX_CODE_POINT &&
           !(code >= SURROGATE_FIRST && code <= SURROGATE_LAST);
}

uint32_t decode_sequence(const std::string & text, size_t pos, size_t length) {
    static constexpr unsigned char LEAD_MASKS[] = {0, 0x7F, 0x1F, 0x0F, 0x07};
    uint32_t code = (unsigned char) text[pos] & LEAD_MASKS[length];
    for (size_t i = 1; i < length; ++i) {
        code = (code << 6) | ((unsigned char) text[pos + i] & 0x3F);
    }
    return code;
}

struct DecodedCharacter {
    uint32_t code;
    size_t consumed;
};

DecodedCharacter decode_at(const std::string & text, size_t pos) {
    const size_t length = sequence_length((unsigned char) text[pos]);
    if (length == 1) {
        return {(unsigned char) text[pos], 1};
    }
    if (length == 0 || !continuation_bytes_valid(text, pos, length)) {
        return {REPLACEMENT, 1};
    }
    const uint32_t code = decode_sequence(text, pos, length);
    return is_well_formed(code, length) ? DecodedCharacter{code, length} : DecodedCharacter{REPLACEMENT, 1};
}

std::vector<uint32_t> decode_utf8(const std::string & text) {
    std::vector<uint32_t> codes;
    codes.reserve(text.size());
    for (size_t pos = 0; pos < text.size();) {
        const DecodedCharacter decoded = decode_at(text, pos);
        codes.push_back(decoded.code);
        pos += decoded.consumed;
    }
    return codes;
}

bool is_listed_space(uint32_t code) {
    return std::find(std::begin(UNICODE_SPACES), std::end(UNICODE_SPACES), code) != std::end(UNICODE_SPACES);
}

bool is_unicode_space(uint32_t code) {
    return (code >= ASCII_CONTROL_SPACE_FIRST && code <= ASCII_CONTROL_SPACE_LAST) || code == ' ' ||
           (code >= EN_QUAD && code <= HAIR_SPACE) || is_listed_space(code);
}

const char * typographic_replacement(uint32_t code) {
    for (const CodePointMapping & mapping : TYPOGRAPHIC_MAPPINGS) {
        if (mapping.code == code) {
            return mapping.text;
        }
    }
    return nullptr;
}

std::string normalize_code_point(uint32_t code) {
    if (code >= FULLWIDTH_FIRST && code <= FULLWIDTH_LAST) {
        return encode_utf8(code - FULLWIDTH_OFFSET);
    }
    if (code == IDEOGRAPHIC_SPACE) {
        return " ";
    }
    const char * replacement = typographic_replacement(code);
    return replacement ? std::string(replacement) : encode_utf8(code);
}

std::string normalize_characters(const std::string & text) {
    std::string out;
    out.reserve(text.size());
    for (uint32_t code : decode_utf8(text)) {
        out += normalize_code_point(code);
    }
    return out;
}

bool match_named_entity(const std::string & text, size_t pos, std::string & out, size_t & consumed) {
    for (const NamedEntity & entity : NAMED_ENTITIES) {
        const std::string name(entity.name);
        if (text.compare(pos, name.size(), name) == 0) {
            out += entity.text;
            consumed = name.size();
            return true;
        }
    }
    return false;
}

bool is_entity_digit(char c, bool hex) {
    return (c >= '0' && c <= '9') || (hex && ((c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')));
}

size_t count_entity_digits(const std::string & text, size_t first, bool hex) {
    size_t count = 0;
    while (first + count < text.size() && count <= MAX_ENTITY_DIGITS && is_entity_digit(text[first + count], hex)) {
        count++;
    }
    return count;
}

uint32_t checked_code_point(unsigned long code) {
    if (code == 0 || code > MAX_CODE_POINT || (code >= SURROGATE_FIRST && code <= SURROGATE_LAST)) {
        return REPLACEMENT;
    }
    return (uint32_t) code;
}

bool match_numeric_entity(const std::string & text, size_t pos, std::string & out, size_t & consumed) {
    if (text.compare(pos, 2, "&#") != 0 || pos + 2 >= text.size()) {
        return false;
    }
    const bool hex = text[pos + 2] == 'x' || text[pos + 2] == 'X';
    const size_t first = pos + (hex ? 3 : 2);
    const size_t digits = count_entity_digits(text, first, hex);
    if (digits == 0 || digits > MAX_ENTITY_DIGITS) {
        return false;
    }
    const unsigned long code = std::strtoul(text.substr(first, digits).c_str(), nullptr, hex ? 16 : 10);
    out += encode_utf8(checked_code_point(code));
    const size_t end = first + digits;
    consumed = end - pos + (end < text.size() && text[end] == ';' ? 1 : 0);
    return true;
}

std::string unescape_html(const std::string & text) {
    std::string out;
    out.reserve(text.size());
    for (size_t pos = 0; pos < text.size();) {
        size_t consumed = 0;
        if (text[pos] == '&' && (match_named_entity(text, pos, out, consumed) ||
                                 match_numeric_entity(text, pos, out, consumed))) {
            pos += consumed;
            continue;
        }
        out += text[pos++];
    }
    return out;
}

std::string unescape_passes(std::string text, int passes) {
    for (int pass = 0; pass < passes; ++pass) {
        text = unescape_html(text);
    }
    return text;
}

std::string unescape_until_stable(std::string text) {
    for (int pass = 0; pass < MAX_FIXED_POINT_PASSES; ++pass) {
        std::string next = unescape_html(text);
        if (next == text) {
            break;
        }
        text = std::move(next);
    }
    return text;
}

std::string fix_text(const std::string & text) {
    const bool looks_like_html = text.find(HTML_TAG_OPEN) != std::string::npos;
    return normalize_characters(looks_like_html ? text : unescape_until_stable(text));
}

std::string collapse_whitespace(const std::string & text) {
    std::string out;
    bool pending_space = false;
    for (uint32_t code : decode_utf8(text)) {
        if (is_unicode_space(code)) {
            pending_space = !out.empty();
            continue;
        }
        if (pending_space) {
            out += ' ';
            pending_space = false;
        }
        out += encode_utf8(code);
    }
    return out;
}

} // namespace

std::string clean_prompt(const std::string & prompt) {
    return collapse_whitespace(unescape_passes(fix_text(prompt), HTML_UNESCAPE_PASSES));
}

int seconds_in_tenths(double seconds) {
    if (!(seconds > 0.0)) {
        return 0;
    }
    if (seconds > MAX_ROUNDED_SECONDS) {
        return std::numeric_limits<int>::max();
    }
    char rounded[ROUNDING_BUFFER];
    std::snprintf(rounded, sizeof(rounded), "%.1f", seconds);
    return (int) std::llround(std::strtod(rounded, nullptr) * TENTHS_PER_SECOND);
}

std::string duration_prompt(const std::string & prompt, int tenths) {
    return prompt + " duration: " + std::to_string(tenths / TENTHS_PER_SECOND) + "." +
           std::to_string(tenths % TENTHS_PER_SECOND) + "s";
}

} // namespace tts_cpp::moss::detail
