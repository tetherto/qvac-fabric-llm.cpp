#pragma once

#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define MM3_MAX_PROMPT_TOKENS 5000
#define MM3_MAX_AUDIO_FRAMES  9000

#define MM3_TPL_IM_START      "<|im_start|>"
#define MM3_TPL_IM_END        "<|im_end|>"
#define MM3_TPL_CAPTION_START "<|caption_start|>"
#define MM3_TPL_CAPTION_END   "<|caption_end|>"
#define MM3_TPL_LYRICS_START  "<|lyrics_start|>"
#define MM3_TPL_LYRICS_END    "<|lyrics_end|>"
#define MM3_TPL_AUDIO_START   "<|audio_start|>"

#define MM3_INSTRUMENTAL_LYRIC "[instrumental]"

static inline bool mm3_is_space(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\v' || c == '\f';
}

static std::string mm3_rstrip(const std::string & s) {
    size_t e = s.size();
    while (e > 0 && (mm3_is_space(s[e - 1]) || s[e - 1] == '\n')) {
        e--;
    }
    return s.substr(0, e);
}

static std::string mm3_strip(const std::string & s) {
    size_t b = 0, e = s.size();
    while (b < e && (mm3_is_space(s[b]) || s[b] == '\n')) {
        b++;
    }
    while (e > b && (mm3_is_space(s[e - 1]) || s[e - 1] == '\n')) {
        e--;
    }
    return s.substr(b, e - b);
}

static bool mm3_str_blank(const std::string & s) {
    for (char c : s) {
        if (!mm3_is_space(c) && c != '\n') {
            return false;
        }
    }
    return true;
}

static std::vector<std::string> mm3_split_lines(const std::string & s) {
    std::vector<std::string> out;
    size_t                   start = 0;
    for (size_t i = 0; i <= s.size(); i++) {
        if (i == s.size() || s[i] == '\n') {
            out.push_back(s.substr(start, i - start));
            start = i + 1;
        }
    }
    return out;
}

static std::vector<std::string> mm3_splitlines(const std::string & s) {
    std::vector<std::string> out;
    size_t                   start = 0;
    size_t                   i     = 0;
    while (i < s.size()) {
        if (s[i] == '\n' || s[i] == '\r') {
            out.push_back(s.substr(start, i - start));
            if (s[i] == '\r' && i + 1 < s.size() && s[i + 1] == '\n') {
                i++;
            }
            start = i + 1;
        }
        i++;
    }
    if (start < s.size()) {
        out.push_back(s.substr(start));
    }
    return out;
}

static std::string mm3_join_lines(const std::vector<std::string> & v) {
    std::string out;
    for (size_t i = 0; i < v.size(); i++) {
        if (i) {
            out += '\n';
        }
        out += v[i];
    }
    return out;
}

static void mm3_replace_all(std::string & s, const std::string & from, const std::string & to) {
    if (from.empty()) {
        return;
    }
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        if (s.compare(i, from.size(), from) == 0) {
            out += to;
            i += from.size();
        } else {
            out += s[i];
            i++;
        }
    }
    s.swap(out);
}

static std::string mm3_rewrite_special_tags(const std::string & in) {
    std::string out;
    out.reserve(in.size());
    size_t i = 0;
    while (i < in.size()) {
        if (in[i] == '<' && i + 1 < in.size() && in[i + 1] == '|') {
            size_t j = i + 2;
            while (j < in.size() && in[j] != '|') {
                j++;
            }
            if (j + 1 < in.size() && in[j] == '|' && in[j + 1] == '>') {
                const std::string inner = mm3_strip(in.substr(i + 2, j - (i + 2)));
                size_t            k     = 0;
                while (k < inner.size() && !mm3_is_space(inner[k]) && inner[k] != '\n') {
                    k++;
                }
                if (k < inner.size()) {
                    size_t r = k;
                    while (r < inner.size() && (mm3_is_space(inner[r]) || inner[r] == '\n')) {
                        r++;
                    }
                    if (r < inner.size()) {
                        out += inner.substr(0, k) + " is " + inner.substr(r);
                    } else {

                        out += inner.substr(0, k);
                    }
                } else {
                    out += inner;
                }
                i = j + 2;
                continue;
            }
        }
        out += in[i];
        i++;
    }
    return out;
}

static void mm3_strip_heading(std::string & line) {
    size_t i = 0;
    while (i < line.size() && i < 3 && mm3_is_space(line[i])) {
        i++;
    }
    size_t h = 0;
    while (i + h < line.size() && h < 6 && line[i + h] == '#') {
        h++;
    }
    if (h == 0) {
        return;
    }
    size_t j = i + h;
    if (j >= line.size() || !mm3_is_space(line[j])) {
        return;
    }
    while (j < line.size() && mm3_is_space(line[j])) {
        j++;
    }
    line.erase(0, j);
}

static void mm3_strip_bullet(std::string & line, const char * marks) {
    size_t i = 0;
    while (i < line.size() && mm3_is_space(line[i])) {
        i++;
    }
    if (i >= line.size() || !strchr(marks, line[i])) {
        return;
    }
    size_t j = i + 1;
    if (j >= line.size() || !mm3_is_space(line[j])) {
        return;
    }
    while (j < line.size() && mm3_is_space(line[j])) {
        j++;
    }
    line.erase(0, j);
}

static std::string mm3_unwrap_bold_once(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        if (s[i] == '*' && i + 1 < s.size() && s[i + 1] == '*') {
            size_t j = i + 2;
            while (j < s.size() && s[j] != '*') {
                j++;
            }
            if (j > i + 2 && j + 1 < s.size() && s[j] == '*' && s[j + 1] == '*') {
                out += s.substr(i + 2, j - (i + 2));
                i = j + 2;
                continue;
            }
        }
        out += s[i];
        i++;
    }
    return out;
}

static std::string mm3_unwrap_italic(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        if (s[i] == '*' && (i == 0 || s[i - 1] != '*')) {
            size_t j = i + 1;
            while (j < s.size() && s[j] != '*' && s[j] != '\n') {
                j++;
            }
            if (j > i + 1 && j < s.size() && s[j] == '*' && (j + 1 >= s.size() || s[j + 1] != '*')) {
                out += s.substr(i + 1, j - (i + 1));
                i = j + 1;
                continue;
            }
        }
        out += s[i];
        i++;
    }
    return out;
}

static bool mm3_is_hrule(const std::string & line) {
    size_t i = 0;
    while (i < line.size() && mm3_is_space(line[i])) {
        i++;
    }
    size_t n = 0;
    while (i + n < line.size() && (line[i + n] == '-' || line[i + n] == '*' || line[i + n] == '_')) {
        n++;
    }
    if (n < 3) {
        return false;
    }
    size_t j = i + n;
    while (j < line.size() && mm3_is_space(line[j])) {
        j++;
    }
    return j == line.size();
}

static std::string mm3_clean_caption(const std::string & caption) {
    std::string text = mm3_rewrite_special_tags(caption);

    std::vector<std::string> lines_out;
    for (std::string line : mm3_splitlines(text)) {
        mm3_strip_heading(line);
        mm3_strip_bullet(line, "*+-");
        mm3_strip_bullet(line, "*");
        while (line.find("**") != std::string::npos) {
            const std::string updated = mm3_unwrap_bold_once(line);
            if (updated == line) {
                break;
            }
            line = updated;
        }
        line = mm3_unwrap_italic(line);
        lines_out.push_back(mm3_rstrip(line));
    }

    for (auto & line : lines_out) {
        if (mm3_is_hrule(line)) {
            line.clear();
        }
    }
    text = mm3_join_lines(lines_out);

    mm3_replace_all(text, "\xE2\x80\xA2 ", "");
    mm3_replace_all(text, "    ", "");

    std::string collapsed;
    collapsed.reserve(text.size());
    size_t i = 0;
    while (i < text.size()) {
        if (text[i] == '\n') {
            size_t j = i;
            while (j < text.size() && text[j] == '\n') {
                j++;
            }
            collapsed += '\n';
            i = j;
        } else {
            collapsed += text[i];
            i++;
        }
    }
    return collapsed;
}

static bool mm3_leading_tags(const std::string & line, std::string * out) {
    size_t i = 0;
    while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) {
        i++;
    }
    const size_t first = i;
    size_t       last  = i;
    int          n     = 0;
    while (i < line.size() && line[i] == '[') {
        size_t j = i + 1;
        while (j < line.size() && line[j] != ']') {
            j++;
        }
        if (j >= line.size() || j == i + 1) {
            break;
        }
        i = j + 1;
        while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) {
            i++;
        }
        last = i;
        n++;
    }
    if (n == 0) {
        return false;
    }
    *out = mm3_strip(line.substr(first, last - first));
    return true;
}

static std::string mm3_lower_tags(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        if (s[i] == '[') {
            size_t j = i + 1;
            while (j < s.size() && s[j] != ']') {
                j++;
            }
            if (j < s.size() && j > i + 1) {
                out += '[';
                for (size_t k = i + 1; k < j; k++) {
                    out += (char) std::tolower((unsigned char) s[k]);
                }
                out += ']';
                i = j + 1;
                continue;
            }
        }
        out += s[i];
        i++;
    }
    return out;
}

static std::string mm3_normalize_lyrics(const std::string & lyrics) {
    std::vector<std::string> out;
    for (const std::string & line : mm3_split_lines(lyrics)) {
        std::string tags;
        out.push_back(mm3_leading_tags(line, &tags) ? tags : line);
    }
    std::string text = mm3_join_lines(out);
    mm3_replace_all(text, "] ", "]\n");
    mm3_replace_all(text, " [", "\n[");
    mm3_replace_all(text, " ^ ", "\n");
    text = mm3_lower_tags(text);
    return "[start]\n" + text;
}

static std::string mm3_assemble_prompt(const std::string & caption, const std::string & lyrics,
                                       bool * out_instrumental = nullptr) {
    const bool instrumental = mm3_str_blank(lyrics);
    if (out_instrumental) {
        *out_instrumental = instrumental;
    }
    return std::string(MM3_TPL_IM_START) + MM3_TPL_CAPTION_START + mm3_clean_caption(caption) + MM3_TPL_CAPTION_END +
           MM3_TPL_LYRICS_START + mm3_normalize_lyrics(instrumental ? std::string(MM3_INSTRUMENTAL_LYRIC) : lyrics) +
           MM3_TPL_LYRICS_END + MM3_TPL_IM_END + MM3_TPL_AUDIO_START;
}
