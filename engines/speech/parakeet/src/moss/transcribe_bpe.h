#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace parakeet::moss::detail {

class QwenByteBpe {
public:
    QwenByteBpe(const std::vector<std::string> & tokens, const std::vector<std::string> & merges);

    std::vector<int32_t> encode(const std::string & text) const;
    const std::array<std::string, 256> & byte_symbols() const;
    uint64_t storage_bytes() const;

private:
    std::vector<std::string> merge_word(const std::string & piece) const;
    void append_piece(const std::string & piece, std::vector<int32_t> & ids) const;
    void append_symbol(const std::string & symbol, std::vector<int32_t> & ids) const;

    std::array<std::string, 256> byte_symbols_;
    std::unordered_map<std::string, int32_t> vocab_;
    std::unordered_map<std::string, int> merge_rank_;
};

std::vector<std::string> qwen_pretokenize(const std::string & text);

} // namespace parakeet::moss::detail
