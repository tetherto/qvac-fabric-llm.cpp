#pragma once

#include "bpe_tokenizer.h"

namespace {

void build_test_byte2str(std::string byte2str[256]) {
    bool direct[256] = {};
    for (int b = '!'; b <= '~'; ++b) direct[b] = true;
    for (int b = 0xA1; b <= 0xAC; ++b) direct[b] = true;
    for (int b = 0xAE; b <= 0xFF; ++b) direct[b] = true;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        const int cp = direct[b] ? b : 256 + n++;
        std::string out;
        if (cp < 0x80) {
            out += (char) cp;
        } else if (cp < 0x800) {
            out += (char) (0xC0 | (cp >> 6));
            out += (char) (0x80 | (cp & 0x3F));
        } else {
            out += (char) (0xE0 | (cp >> 12));
            out += (char) (0x80 | ((cp >> 6) & 0x3F));
            out += (char) (0x80 | (cp & 0x3F));
        }
        byte2str[b] = out;
    }
}

// Vocab ids: h=0 e=1 l=2 o=3 a=4 b=5 1=6 2=7 he=8 ll=9 hello=10 Ġhello=11.
// The merge chain h+e, l+l, he+ll, hell+o, Ġ+hello reaches "hello" and
// " hello"; "a b" produces a piece deliberately absent from the vocab.
tts_cpp::acestep::BpeTokenizer make_test_bpe_tokenizer() {
    tts_cpp::acestep::BpeTokenizer tok;
    build_test_byte2str(tok.byte2str);
    const std::string g_space = tok.byte2str[(unsigned char) ' '];
    const std::vector<std::string> tokens = {
        "h", "e", "l", "o", "a", "b", "1", "2", "he", "ll", "hello", g_space + "hello",
    };
    for (size_t i = 0; i < tokens.size(); ++i) tok.vocab[tokens[i]] = (int) i;
    tok.n_vocab   = (int) tokens.size();
    tok.id_to_str = tokens;
    const std::vector<std::string> merge_list = {
        "h e", "l l", "he ll", "hell o", g_space + " hello", "a b",
    };
    for (size_t i = 0; i < merge_list.size(); ++i) tok.merges[merge_list[i]] = (int) i;
    return tok;
}

}  // namespace
