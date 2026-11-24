#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct gguf_context;

namespace parakeet::moss::detail {

class GgufMetadata {
public:
    GgufMetadata(const gguf_context * file, std::string owner);

    uint32_t u32(const std::string & key) const;
    float f32(const std::string & key) const;
    bool boolean(const std::string & key) const;
    std::string str(const std::string & key) const;
    std::vector<std::string> str_array(const std::string & key) const;
    std::vector<int32_t> int_array(const std::string & key, size_t max_count) const;

private:
    int64_t require(const std::string & key) const;
    [[noreturn]] void fail(const std::string & message) const;

    const gguf_context * file_;
    std::string owner_;
};

} // namespace parakeet::moss::detail
