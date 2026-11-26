#include "pocket/frontend.h"
#include "json.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace tts_cpp::pocket::detail {
namespace {
[[noreturn]] void fail(const std::string & s) { throw std::runtime_error("pocket frontend: " + s); }
const std::string marker = "\xe2\x96\x81";
// Decode strictly so byte indexing never splits invalid UTF-8 sequences.
unsigned codepoint(const std::string & s, size_t & i) {
    const unsigned char lead = s.at(i++);
    if (lead < 0x80) return lead;
    const int n = lead >= 0xc2 && lead <= 0xdf ? 1 : lead >= 0xe0 && lead <= 0xef ? 2 : lead >= 0xf0 && lead <= 0xf4 ? 3 : 0;
    if (!n || i+n > s.size()) fail("invalid UTF-8");
    unsigned cp = lead & ((1u << (6-n))-1);
    for (int k = 0; k < n; ++k) {
        unsigned char b = s[i++]; if ((b & 0xc0) != 0x80) fail("invalid UTF-8");
        cp = (cp << 6) | (b & 0x3f);
    }
    if ((n == 1 && cp < 0x80) || (n == 2 && cp < 0x800) || (n == 3 && cp < 0x10000) ||
        cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) fail("invalid UTF-8");
    return cp;
}
void replace_all(std::string & s, const std::string & from, const std::string & to) {
    for (size_t at = 0; (at = s.find(from, at)) != std::string::npos; at += to.size()) s.replace(at, from.size(), to);
}
}

Frontend::Frontend(const std::string & path) {
    if (std::filesystem::file_size(path) > 4*1024*1024) fail("frontend metadata too large");
    std::ifstream input(path); nlohmann::json j; input >> j;
    if (j.at("architecture") != "pocket-tts-frontend" || j.at("schema_version") != 1) fail("unsupported metadata");
    hash_ = j.at("source_sha256").get<std::string>();
    if (hash_.size() != 64 || hash_.find_first_not_of("0123456789abcdef") != std::string::npos) fail("invalid source hash");
    pad_ = j.at("pad_short").get<bool>(); semicolons_ = j.at("remove_semicolons").get<bool>();
    punctuation_ = j.at("append_punctuation").get<bool>();
    if (!j.at("frames_after_eos").is_null()) tail_ = j.at("frames_after_eos").get<int>();
    if (tail_ < -1 || tail_ > 100) fail("invalid EOS tail");
    for (const auto & cp : j.at("whitespace")) spaces_.insert(cp.get<unsigned>());
    for (const auto & cp : j.at("digits")) digits_.insert(cp.get<unsigned>());
    for (auto it = j.at("uppercase").begin(); it != j.at("uppercase").end(); ++it) upper_[std::stoul(it.key())] = it.value().get<std::string>();
    std::fill(std::begin(bytes_), std::end(bytes_), -1);
    trie_.emplace_back(); float minimum = 0;
    const auto & vocab = j.at("pieces");
    if (vocab.size() < 256 || vocab.size() > 65536) fail("invalid vocabulary size");
    for (const auto & piece : vocab) {
        const int id = pieces_.size(), type = piece.at("type").get<int>();
        const auto text = piece.at("text").get<std::string>(); const auto score = piece.at("score").get<float>();
        if (text.empty() || text.size() > 1024 || !std::isfinite(score)) fail("invalid vocabulary entry");
        pieces_.push_back(text); scores_.push_back(score); types_.push_back(type);
        if (type == 1) {
            minimum = std::min(minimum, score); int node = 0;
            for (unsigned char b : text) {
                auto it = trie_[node].children.find(b);
                if (it == trie_[node].children.end()) {
                    const int next = trie_.size(); trie_[node].children[b] = next;
                    trie_.emplace_back(); node = next;
                } else node = it->second;
            }
            if (trie_[node].id >= 0) fail("duplicate token"); trie_[node].id = id;
        } else if (type == 6) {
            if (text.size() != 6 || text.substr(0,3) != "<0x" || text.back() != '>') fail("invalid byte token");
            size_t parsed; const auto value = std::stoul(text.substr(3,2), &parsed, 16);
            if (parsed != 2 || value > 255 || bytes_[value] != -1) fail("invalid byte token");
            bytes_[value] = id;
        } else if (type != 2 && type != 3) fail("unsupported piece type");
    }
    if (std::find(std::begin(bytes_), std::end(bytes_), -1) != std::end(bytes_)) fail("incomplete byte fallback");
    unknown_score_ = minimum-10;
}

std::vector<int> Frontend::encode(const std::string & text) const {
    if (text.empty()) return {};
    if (text.size() > 65536) fail("text exceeds 65536 UTF-8 bytes");
    std::string s = marker;
    for (size_t i = 0; i < text.size();) {
        const auto first = i; auto cp = codepoint(text, i);
        s += cp == ' ' ? marker : text.substr(first, i-first);
    }
    struct End { float score = -std::numeric_limits<float>::infinity(); int id = -1; size_t from = 0; };
    std::vector<End> best(s.size()+1); best[0].score = 0;
    for (size_t i = 0; i < s.size();) {
        size_t next = i; codepoint(s, next); bool single = false; int node = 0;
        for (size_t end = i; end < s.size();) {
            auto it = trie_[node].children.find(static_cast<unsigned char>(s[end++]));
            if (it == trie_[node].children.end()) break;
            node = it->second; const int id = trie_[node].id;
            if (id >= 0) {
                if (end == next) single = true;
                const float score = best[i].score + scores_[id];
                if (score > best[end].score) best[end] = {score, id, i};
            }
        }
        if (!single && best[i].score+unknown_score_ > best[next].score) best[next] = {best[i].score+unknown_score_, -1, i};
        i = next;
    }
    std::vector<std::vector<int>> reverse;
    for (size_t end = s.size(); end;) {
        const auto & entry = best[end]; if (entry.from >= end) fail("unreachable token lattice");
        std::vector<int> ids;
        if (entry.id >= 0) ids.push_back(entry.id);
        else for (size_t i = entry.from; i < end; ++i) ids.push_back(bytes_[static_cast<unsigned char>(s[i])]);
        reverse.push_back(std::move(ids)); end = entry.from;
    }
    std::vector<int> ids;
    for (auto it = reverse.rbegin(); it != reverse.rend(); ++it) ids.insert(ids.end(), it->begin(), it->end());
    return ids;
}

std::string Frontend::decode(const std::vector<int> & ids) const {
    std::string out; bool first = true;
    for (int id : ids) {
        if (id < 0 || id >= int(pieces_.size())) fail("invalid token id");
        auto piece = pieces_[id];
        if (types_[id] == 1) {
            if (first && piece.compare(0, marker.size(), marker) == 0) piece.erase(0, marker.size());
            replace_all(piece, marker, " "); out += piece; first = false;
        } else if (types_[id] == 6) { out += char(std::stoul(piece.substr(3,2), nullptr, 16)); first = false; }
        else if (types_[id] == 2) { out += " ⁇ "; first = false; }
    }
    return out;
}
std::string Frontend::trim(const std::string & s) const {
    size_t first = s.size(), last = 0;
    for (size_t i = 0; i < s.size();) {
        size_t begin = i; if (!spaces_.count(codepoint(s, i))) { first = std::min(first, begin); last = i; }
    }
    return last ? s.substr(first, last-first) : "";
}
TextPrompt Frontend::prepare(const std::string & raw) const {
    if (raw.size() > 65536) fail("text exceeds 65536 UTF-8 bytes");
    std::string s = trim(raw); if (s.empty()) fail("text is empty");
    replace_all(s, "\n", " "); replace_all(s, "\r", " "); replace_all(s, "  ", " ");
    if (semicolons_) replace_all(s, ";", ",");
    int words = 0; bool word = false;
    for (size_t i = 0; i < s.size();) {
        const bool current = !spaces_.count(codepoint(s, i)); if (current && !word) ++words; word = current;
    }
    size_t next = 0; const auto cp = codepoint(s, next); auto upper = upper_.find(cp);
    if (upper != upper_.end()) s.replace(0, next, upper->second);
    if (punctuation_) {
        const std::set<unsigned> closers = {'"','\'',0x201d,0x2019,')',']',0xbb,' '};
        const std::set<unsigned> weak = {',',';',':','-',0x2013,0x2014};
        const std::set<unsigned> terminal = {'.','!','?',0x2026};
        std::vector<std::pair<size_t,unsigned>> chars;
        for (size_t i = 0; i < s.size();) { const auto at = i; const auto c = codepoint(s, i); chars.push_back({at,c}); }
        size_t end = chars.size(); while (end && closers.count(chars[end-1].second)) --end;
        if (end && !terminal.count(chars[end-1].second)) {
            if (weak.count(chars[end-1].second)) {
                const auto suffix = end < chars.size() ? trim(s.substr(chars[end].first)) : "";
                while (end && (weak.count(chars[end-1].second) || chars[end-1].second == ' ')) --end;
                s = s.substr(0, end < chars.size() ? chars[end].first : s.size()) + "." + suffix;
            } else s += ".";
        }
    }
    if (pad_ && words < 5) s.insert(0, 8, ' ');
    return {s, tail_ >= 0 ? tail_ : words <= 4 ? 5 : 3};
}

std::vector<int> Frontend::boundaries(const std::vector<int> & ids, const std::set<int> & ends, bool decimal) const {
    std::vector<int> result{0}; bool previous = false;
    for (int i = 0; i < int(ids.size()); ++i) {
        if (ends.count(ids[i])) previous = true;
        else if (previous) {
            previous = false;
            if (decimal) {
                const auto a = decode({ids.begin(), ids.begin()+i}), b = decode({ids.begin()+i, ids.end()});
                unsigned previous_cp = 0, last_cp = 0;
                for (size_t at = 0; at < a.size();) { previous_cp = last_cp; last_cp = codepoint(a, at); }
                size_t at = 0;
                if (last_cp == '.' && digits_.count(previous_cp) && !b.empty() && digits_.count(codepoint(b, at))) continue;
            }
            result.push_back(i);
        }
    }
    result.push_back(ids.size()); return result;
}
std::vector<TextPrompt> Frontend::split(const std::string & raw, int max_tokens) const {
    if (max_tokens < 1 || max_tokens > 1024) fail("max tokens must be 1..1024");
    auto ids = encode(trim(prepare(raw).text));
    auto ends = encode(".!...?"); auto fallback = encode(",;:");
    std::vector<std::pair<int,std::string>> segments;
    auto append = [&](const std::vector<int> & tokens, const std::vector<int> & bounds) {
        for (size_t i = 1; i < bounds.size(); ++i) segments.push_back({bounds[i]-bounds[i-1], decode({tokens.begin()+bounds[i-1],tokens.begin()+bounds[i]})});
    };
    const auto bounds = boundaries(ids, {ends.begin()+1, ends.end()}, true);
    for (size_t i = 1; i < bounds.size(); ++i) {
        const int count = bounds[i]-bounds[i-1];
        const auto sentence = decode({ids.begin()+bounds[i-1], ids.begin()+bounds[i]});
        if (count <= max_tokens) segments.push_back({count,sentence});
        else {
            const auto sub = encode(trim(sentence)); const auto sub_bounds = boundaries(sub, {fallback.begin()+1,fallback.end()}, false);
            if (sub_bounds.size() > 2) append(sub, sub_bounds); else segments.push_back({count,sentence});
        }
    }
    std::vector<TextPrompt> chunks; std::string current; int count = 0;
    for (const auto & segment : segments) {
        if (current.empty()) { current = segment.second; count = segment.first; }
        else if (count+segment.first > max_tokens) { chunks.push_back(prepare(trim(current))); current = segment.second; count = segment.first; }
        else { current += " " + segment.second; count += segment.first; }
    }
    if (!current.empty()) chunks.push_back(prepare(trim(current)));
    return chunks;
}
uint64_t Frontend::memory_bytes() const {
    uint64_t bytes = sizeof(*this) + trie_.capacity()*sizeof(Node) +
        pieces_.capacity()*sizeof(std::string) + scores_.capacity()*sizeof(float) +
        types_.capacity()*sizeof(int);
    // Include conservative map/set node and allocator overhead, plus string
    // capacity even when the library stores a small string inline.
    for (const auto & node : trie_) bytes += node.children.size()*(64+sizeof(std::pair<const unsigned char,int>));
    for (const auto & piece : pieces_) bytes += piece.capacity()+1;
    bytes += upper_.size()*(64+sizeof(std::pair<const unsigned,std::string>));
    for (const auto & item : upper_) bytes += item.second.capacity()+1;
    bytes += (spaces_.size()+digits_.size())*(64+sizeof(unsigned));
    return bytes + hash_.capacity()+1;
}

}
