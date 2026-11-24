
#pragma once
#include "gguf.h"
#include "unicode_categories.h"

#include <cassert>
#include <climits>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

static void build_byte_encoder(std::string byte2str[256]) {

    int bs[256], cs[256], n = 0, total = 0;

    for (int b = '!'; b <= '~'; b++) {
        bs[total] = b;
        cs[total] = b;
        total++;
    }
    for (int b = 0xA1; b <= 0xAC; b++) {
        bs[total] = b;
        cs[total] = b;
        total++;
    }
    for (int b = 0xAE; b <= 0xFF; b++) {
        bs[total] = b;
        cs[total] = b;
        total++;
    }

    bool used[256] = {};
    for (int i = 0; i < total; i++) {
        used[bs[i]] = true;
    }
    for (int b = 0; b < 256; b++) {
        if (!used[b]) {
            bs[total] = b;
            cs[total] = 256 + n;
            n++;
            total++;
        }
    }
    assert(total == 256);

    for (int i = 0; i < 256; i++) {
        int  cp = cs[i];
        char buf[4];
        int  len;
        if (cp < 0x80) {
            buf[0] = (char) cp;
            len    = 1;
        } else if (cp < 0x800) {
            buf[0] = (char) (0xC0 | (cp >> 6));
            buf[1] = (char) (0x80 | (cp & 0x3F));
            len    = 2;
        } else {
            buf[0] = (char) (0xE0 | (cp >> 12));
            buf[1] = (char) (0x80 | ((cp >> 6) & 0x3F));
            buf[2] = (char) (0x80 | (cp & 0x3F));
            len    = 3;
        }
        byte2str[bs[i]] = std::string(buf, len);
    }
}

static bool utf8_is_continuation(unsigned char value) {
    return (value & 0xC0) == 0x80;
}

static int utf8_codepoint(const char * s, int remaining, int * advance) {
    if (remaining <= 0) {
        *advance = 0;
        return -1;
    }
    const unsigned char c = static_cast<unsigned char>(s[0]);
    if (c < 0x80) {
        *advance = 1;
        return c;
    }
    if (remaining >= 2 && c >= 0xC2 && c <= 0xDF &&
        utf8_is_continuation(static_cast<unsigned char>(s[1]))) {
        *advance = 2;
        return ((c & 0x1F) << 6) | (static_cast<unsigned char>(s[1]) & 0x3F);
    }
    if (remaining >= 3 && c >= 0xE0 && c <= 0xEF &&
        utf8_is_continuation(static_cast<unsigned char>(s[1])) &&
        utf8_is_continuation(static_cast<unsigned char>(s[2])) &&
        !(c == 0xE0 && static_cast<unsigned char>(s[1]) < 0xA0) &&
        !(c == 0xED && static_cast<unsigned char>(s[1]) >= 0xA0)) {
        *advance = 3;
        return ((c & 0x0F) << 12) | ((static_cast<unsigned char>(s[1]) & 0x3F) << 6) |
               (static_cast<unsigned char>(s[2]) & 0x3F);
    }
    if (remaining >= 4 && c >= 0xF0 && c <= 0xF4 &&
        utf8_is_continuation(static_cast<unsigned char>(s[1])) &&
        utf8_is_continuation(static_cast<unsigned char>(s[2])) &&
        utf8_is_continuation(static_cast<unsigned char>(s[3])) &&
        !(c == 0xF0 && static_cast<unsigned char>(s[1]) < 0x90) &&
        !(c == 0xF4 && static_cast<unsigned char>(s[1]) >= 0x90)) {
        *advance = 4;
        return ((c & 0x07) << 18) | ((static_cast<unsigned char>(s[1]) & 0x3F) << 12) |
               ((static_cast<unsigned char>(s[2]) & 0x3F) << 6) |
               (static_cast<unsigned char>(s[3]) & 0x3F);
    }
    *advance = 1;
    return c;
}

static bool is_letter(int cp) {
    return cp >= 0 && mm3_unicode_is_letter((uint32_t) cp);
}

static bool is_digit(int cp) {
    return cp >= 0 && mm3_unicode_is_number((uint32_t) cp);
}

static bool is_whitespace(int cp) {
    return cp >= 0 && mm3_unicode_is_whitespace((uint32_t) cp);
}

static bool is_newline(int cp) {
    return cp == '\n' || cp == '\r';
}

typedef bool (*CodepointPredicate)(int);

static bool is_punctuation(int cp) {
    return !is_newline(cp) && !is_letter(cp) && !is_digit(cp) && !is_whitespace(cp);
}

static int scan_codepoints(const char * s, int len, int start, CodepointPredicate predicate) {
    int position = start;
    while (position < len) {
        int advance;
        int cp = utf8_codepoint(s + position, len - position, &advance);
        if (!predicate(cp)) {
            break;
        }
        position += advance;
    }
    return position;
}

static int scan_letters(const char * s, int len, int start) {
    return scan_codepoints(s, len, start, is_letter);
}

static int scan_newlines(const char * s, int len, int start) {
    return scan_codepoints(s, len, start, is_newline);
}

static int scan_whitespace(const char * s, int len, int start) {
    return scan_codepoints(s, len, start, is_whitespace);
}

static int scan_punctuation(const char * s, int len, int start) {
    return scan_codepoints(s, len, start, is_punctuation);
}

static void append_pre_token(const char * s, int start, int end,
                             std::vector<std::string> & chunks) {
    chunks.push_back(std::string(s + start, end - start));
}

static char ascii_lower(char value) {
    const int case_offset = 'a' - 'A';
    if (value >= 'A' && value <= 'Z') {
        return (char) (value + case_offset);
    }
    return value;
}

static bool ascii_case_insensitive_match(const char * text, const char * expected, int length) {
    for (int i = 0; i < length; i++) {
        if (ascii_lower(text[i]) != expected[i]) {
            return false;
        }
    }
    return true;
}

static bool contraction_suffix_matches(const char * s, int len, int suffix_start,
                                       const char * suffix, int suffix_length) {
    const int remaining = len - suffix_start;
    if (remaining < suffix_length ||
        !ascii_case_insensitive_match(s + suffix_start, suffix, suffix_length)) {
        return false;
    }
    if (remaining == suffix_length) {
        return true;
    }
    int advance;
    int cp = utf8_codepoint(s + suffix_start + suffix_length,
                            remaining - suffix_length, &advance);
    return !is_letter(cp);
}

static bool try_append_contraction_suffix(const char * s, int len, int start,
                                          int apostrophe_length, const char * suffix,
                                          std::vector<std::string> & chunks, int & position) {
    const int suffix_length = (int) std::strlen(suffix);
    const int suffix_start = start + apostrophe_length;
    if (!contraction_suffix_matches(s, len, suffix_start, suffix, suffix_length)) {
        return false;
    }
    position = suffix_start + suffix_length;
    append_pre_token(s, start, position, chunks);
    return true;
}

static bool try_append_contraction(const char * s, int len, int cp, int advance,
                                   std::vector<std::string> & chunks, int & position) {
    const int right_single_quotation_mark = 0x2019;
    if ((cp != '\'' && cp != right_single_quotation_mark) || position + advance >= len) {
        return false;
    }
    const int start = position;
    return try_append_contraction_suffix(s, len, start, advance, "ll", chunks, position) ||
           try_append_contraction_suffix(s, len, start, advance, "re", chunks, position) ||
           try_append_contraction_suffix(s, len, start, advance, "ve", chunks, position) ||
           try_append_contraction_suffix(s, len, start, advance, "s", chunks, position) ||
           try_append_contraction_suffix(s, len, start, advance, "t", chunks, position) ||
           try_append_contraction_suffix(s, len, start, advance, "m", chunks, position) ||
           try_append_contraction_suffix(s, len, start, advance, "d", chunks, position);
}

static bool try_append_letters(const char * s, int len, int cp,
                               std::vector<std::string> & chunks, int & position) {
    if (!is_letter(cp)) {
        return false;
    }
    const int start = position;
    position = scan_letters(s, len, position);
    append_pre_token(s, start, position, chunks);
    return true;
}

static bool try_append_prefixed_letters(const char * s, int len, int cp, int advance,
                                        std::vector<std::string> & chunks, int & position) {
    const int letter_start = position + advance;
    if (!is_punctuation(cp) || letter_start >= len) {
        return false;
    }
    int letter_advance;
    int next_cp = utf8_codepoint(s + letter_start, len - letter_start, &letter_advance);
    if (!is_letter(next_cp)) {
        return false;
    }
    const int start = position;
    position = scan_letters(s, len, letter_start);
    append_pre_token(s, start, position, chunks);
    return true;
}

static bool try_append_number(const char * s, int cp, int advance,
                              std::vector<std::string> & chunks, int & position) {
    if (!is_digit(cp)) {
        return false;
    }
    const int start = position;
    position += advance;
    append_pre_token(s, start, position, chunks);
    return true;
}

static bool try_append_newlines(const char * s, int len, int cp,
                                std::vector<std::string> & chunks, int & position) {
    if (!is_newline(cp)) {
        return false;
    }
    const int start = position;
    position = scan_newlines(s, len, position);
    append_pre_token(s, start, position, chunks);
    return true;
}

struct WhitespaceRun {
    int end;
    int last_start;
    int count;
};

static WhitespaceRun scan_non_newline_whitespace(const char * s, int len, int start) {
    WhitespaceRun run = {start, start, 0};
    while (run.end < len) {
        int advance;
        int cp = utf8_codepoint(s + run.end, len - run.end, &advance);
        if (!is_whitespace(cp) || is_newline(cp)) {
            break;
        }
        run.last_start = run.end;
        run.end += advance;
        run.count++;
    }
    return run;
}

static bool is_followed_by_non_whitespace(const char * s, int len, int position) {
    if (position >= len) {
        return false;
    }
    int advance;
    int cp = utf8_codepoint(s + position, len - position, &advance);
    return !is_whitespace(cp) && !is_newline(cp);
}

static void append_punctuation_with_newlines(const char * s, int len, int start,
                                             int punctuation_start,
                                             std::vector<std::string> & chunks,
                                             int & position) {
    position = scan_punctuation(s, len, punctuation_start);
    position = scan_newlines(s, len, position);
    append_pre_token(s, start, position, chunks);
}

static bool try_append_whitespace(const char * s, int len, int cp, int advance,
                                  std::vector<std::string> & chunks, int & position) {
    if (!is_whitespace(cp)) {
        return false;
    }
    const int start = position;
    const WhitespaceRun run = scan_non_newline_whitespace(s, len, start);
    if (run.count > 1 && is_followed_by_non_whitespace(s, len, run.end)) {
        position = run.last_start;
        append_pre_token(s, start, position, chunks);
        return true;
    }

    position = start + advance;
    if (position < len) {
        int next_advance;
        int next_cp = utf8_codepoint(s + position, len - position, &next_advance);
        if (is_letter(next_cp)) {
            position = scan_letters(s, len, position);
            append_pre_token(s, start, position, chunks);
            return true;
        }
        if (is_digit(next_cp)) {
            append_pre_token(s, start, position, chunks);
            return true;
        }
        if (!is_whitespace(next_cp) && !is_newline(next_cp)) {
            append_punctuation_with_newlines(s, len, start, position, chunks, position);
            return true;
        }
    }

    position = scan_whitespace(s, len, run.end);
    append_pre_token(s, start, position, chunks);
    return true;
}

static void append_punctuation(const char * s, int len,
                               std::vector<std::string> & chunks, int & position) {
    const int start = position;
    append_punctuation_with_newlines(s, len, start, start, chunks, position);
}

static void append_next_pre_token(const char * s, int len,
                                  std::vector<std::string> & chunks, int & position) {
    int advance;
    int cp = utf8_codepoint(s + position, len - position, &advance);
    if (try_append_contraction(s, len, cp, advance, chunks, position) ||
        try_append_letters(s, len, cp, chunks, position) ||
        try_append_prefixed_letters(s, len, cp, advance, chunks, position) ||
        try_append_number(s, cp, advance, chunks, position) ||
        try_append_newlines(s, len, cp, chunks, position) ||
        try_append_whitespace(s, len, cp, advance, chunks, position)) {
        return;
    }
    append_punctuation(s, len, chunks, position);
}

static std::vector<std::string> gpt2_pre_tokenize(const std::string & text) {
    std::vector<std::string> chunks;
    const char * s = text.c_str();
    const int len = (int) text.size();
    int position = 0;
    while (position < len) {
        append_next_pre_token(s, len, chunks, position);
    }
    return chunks;
}

struct BPETokenizer {
    std::unordered_map<std::string, int> vocab;
    std::unordered_map<std::string, int> merges;
    std::string                          byte2str[256];
    int                                  eos_id;
    int                                  n_vocab;
    std::vector<std::string>             id_to_str;
};

static bool load_bpe_from_gguf(BPETokenizer * tok, const char * gguf_path) {
    build_byte_encoder(tok->byte2str);

    struct gguf_init_params gp  = { true, NULL };
    struct gguf_context *   ctx = gguf_init_from_file(gguf_path, gp);
    if (!ctx) {
        fprintf(stderr, "[BPE] Failed to open %s\n", gguf_path);
        return false;
    }

    int64_t tok_key = gguf_find_key(ctx, "tokenizer.ggml.tokens");
    int64_t mrg_key = gguf_find_key(ctx, "tokenizer.ggml.merges");
    if (tok_key < 0 || mrg_key < 0) {
        fprintf(stderr, "[BPE] Tokenizer not found in %s\n", gguf_path);
        gguf_free(ctx);
        return false;
    }

    int n_tokens = (int) gguf_get_arr_n(ctx, tok_key);
    int n_merges = (int) gguf_get_arr_n(ctx, mrg_key);

    for (int i = 0; i < n_tokens; i++) {
        const char * s             = gguf_get_arr_str(ctx, tok_key, (size_t) i);
        tok->vocab[std::string(s)] = i;
    }

    for (int i = 0; i < n_merges; i++) {
        const char * s              = gguf_get_arr_str(ctx, mrg_key, (size_t) i);
        tok->merges[std::string(s)] = i;
    }

    gguf_free(ctx);

    tok->n_vocab = (int) tok->vocab.size();
    tok->eos_id  = 151643;

    tok->id_to_str.resize(tok->n_vocab);
    for (auto & kv : tok->vocab) {
        if (kv.second >= 0 && kv.second < tok->n_vocab) {
            tok->id_to_str[kv.second] = kv.first;
        }
    }

    fprintf(stderr, "[BPE] Loaded from GGUF: %d vocab, %d merges\n", tok->n_vocab, n_merges);
    return true;
}

static std::string byte_level_encode(const BPETokenizer * tok, const std::string & text) {
    std::string out;
    for (unsigned char c : text) {
        out += tok->byte2str[c];
    }
    return out;
}

static std::vector<std::string> bpe_merge(const std::unordered_map<std::string, int> & merge_rank,
                                          const std::vector<std::string> &             symbols) {
    if (symbols.size() <= 1) {
        return symbols;
    }

    std::vector<std::string> work = symbols;

    while (work.size() > 1) {

        int best_rank = INT_MAX;
        int best_pos  = -1;
        for (int i = 0; i < (int) work.size() - 1; i++) {
            std::string key = work[i] + " " + work[i + 1];
            auto        it  = merge_rank.find(key);
            if (it != merge_rank.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos  = i;
            }
        }
        if (best_pos < 0) {
            break;
        }

        std::string merged = work[best_pos] + work[best_pos + 1];
        work[best_pos]     = merged;
        work.erase(work.begin() + best_pos + 1);
    }
    return work;
}

static void encode_chunk(const BPETokenizer * tok, const std::string & chunk, std::vector<int> & ids) {

    std::string encoded = byte_level_encode(tok, chunk);

    std::vector<std::string> symbols;
    const char *             s   = encoded.c_str();
    int                      len = (int) encoded.size();
    int                      i   = 0;
    while (i < len) {
        int adv;
        utf8_codepoint(s + i, len - i, &adv);
        symbols.push_back(std::string(s + i, adv));
        i += adv;
    }

    std::vector<std::string> merged = bpe_merge(tok->merges, symbols);

    for (const auto & piece : merged) {
        auto it = tok->vocab.find(piece);
        if (it != tok->vocab.end()) {
            ids.push_back(it->second);
        } else {

            fprintf(stderr, "[BPE] WARNING: unknown token '%s'\n", piece.c_str());
            for (unsigned char c : piece) {
                auto it2 = tok->vocab.find(std::string(1, c));
                if (it2 != tok->vocab.end()) {
                    ids.push_back(it2->second);
                }
            }
        }
    }
}

static std::vector<int> bpe_encode(const BPETokenizer * tok, const std::string & text, bool add_eos = true) {
    std::vector<int>  ids;
    const std::string special = "<|endoftext|>";

    size_t pos = 0;
    while (pos < text.size()) {
        size_t      found   = text.find(special, pos);
        std::string segment = (found == std::string::npos) ? text.substr(pos) : text.substr(pos, found - pos);

        if (!segment.empty()) {
            auto chunks = gpt2_pre_tokenize(segment);
            for (const auto & chunk : chunks) {
                encode_chunk(tok, chunk, ids);
            }
        }

        if (found == std::string::npos) {
            break;
        }
        ids.push_back(tok->eos_id);
        pos = found + special.size();
    }

    if (add_eos) {
        ids.push_back(tok->eos_id);
    }
    return ids;
}
