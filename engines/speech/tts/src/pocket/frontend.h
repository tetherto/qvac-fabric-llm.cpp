#pragma once
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace tts_cpp::pocket::detail {
struct TextPrompt { std::string text; int tail_frames; };
class Frontend {
public:
    explicit Frontend(const std::string & path);
    std::vector<int> encode(const std::string & text) const;
    std::string decode(const std::vector<int> & ids) const;
    TextPrompt prepare(const std::string & text) const;
    std::vector<TextPrompt> split(const std::string & text, int max_tokens) const;
    const std::string & source_hash() const { return hash_; }
    int vocab_size() const { return pieces_.size(); }
    uint64_t memory_bytes() const;
private:
    struct Node { std::map<unsigned char, int> children; int id = -1; };
    std::vector<Node> trie_;
    std::vector<std::string> pieces_;
    std::vector<float> scores_;
    std::vector<int> types_;
    std::map<unsigned, std::string> upper_;
    std::set<unsigned> spaces_, digits_;
    int bytes_[256];
    float unknown_score_ = 0;
    bool pad_ = false, semicolons_ = false, punctuation_ = true;
    int tail_ = -1;
    std::string hash_;
    std::string trim(const std::string & text) const;
    std::vector<int> boundaries(const std::vector<int> &, const std::set<int> &, bool decimal) const;
};
}
