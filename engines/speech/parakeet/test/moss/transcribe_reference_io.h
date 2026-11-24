#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace moss_transcribe_reference_io {

inline std::runtime_error invalid_fixture(const std::filesystem::path & path) {
    return std::runtime_error("invalid or truncated reference fixture: " + path.string());
}

inline uint64_t remaining_bytes(std::ifstream & input, const std::filesystem::path & path) {
    const auto position = input.tellg();
    input.seekg(0, std::ios::end);
    const auto end = input.tellg();
    if (position < 0 || end < position) throw invalid_fixture(path);
    input.seekg(position);
    if (!input) throw invalid_fixture(path);
    return static_cast<uint64_t>(end - position);
}

inline void read_exactly(std::ifstream & input, void * output, uint64_t bytes,
                         const std::filesystem::path & path) {
    if (bytes > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        throw invalid_fixture(path);
    }
    if (bytes && !input.read(static_cast<char *>(output), static_cast<std::streamsize>(bytes))) {
        throw invalid_fixture(path);
    }
}

template <typename T>
std::vector<T> read_bin(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read " + path.string());
    const uint64_t bytes = remaining_bytes(input, path);
    if (bytes % sizeof(T) != 0 || bytes / sizeof(T) > std::vector<T>().max_size()) {
        throw invalid_fixture(path);
    }
    std::vector<T> data(static_cast<size_t>(bytes / sizeof(T)));
    read_exactly(input, data.data(), bytes, path);
    return data;
}

inline std::vector<float> prefix(const std::vector<float> & values, size_t count) {
    if (count > values.size()) throw std::runtime_error("audio embedding reference is shorter than the encoder output");
    return {values.begin(), values.begin() + static_cast<std::ptrdiff_t>(count)};
}

struct TokenizerCase {
    std::string text;
    std::vector<int32_t> ids;
};

inline uint32_t read_u32(std::ifstream & input, const std::filesystem::path & path) {
    uint32_t value = 0;
    read_exactly(input, &value, sizeof(value), path);
    return value;
}

inline TokenizerCase read_case(std::ifstream & input, const std::filesystem::path & path) {
    TokenizerCase item;
    const uint32_t text_bytes = read_u32(input, path);
    const uint64_t remaining = remaining_bytes(input, path);
    if (text_bytes > remaining || remaining - text_bytes < sizeof(uint32_t)) throw invalid_fixture(path);
    item.text.assign(text_bytes, '\0');
    read_exactly(input, item.text.data(), text_bytes, path);
    const uint32_t count = read_u32(input, path);
    if (count > remaining_bytes(input, path) / sizeof(int32_t) || count > item.ids.max_size()) {
        throw invalid_fixture(path);
    }
    item.ids.resize(count);
    read_exactly(input, item.ids.data(), static_cast<uint64_t>(count) * sizeof(int32_t), path);
    return item;
}

inline std::vector<TokenizerCase> read_tokenizer_cases(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read " + path.string());
    std::vector<TokenizerCase> cases;
    while (remaining_bytes(input, path) != 0) cases.push_back(read_case(input, path));
    if (cases.empty()) throw std::runtime_error("tokenizer reference contains no cases: " + path.string());
    return cases;
}

} // namespace moss_transcribe_reference_io
