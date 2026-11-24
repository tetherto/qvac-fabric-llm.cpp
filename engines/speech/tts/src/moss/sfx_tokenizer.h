#pragma once

#include "moss/sfx_model.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::moss::detail {

class SfxTokenizer {
public:
    explicit SfxTokenizer(const SfxModel & model);
    ~SfxTokenizer();
    SfxTokenizer(const SfxTokenizer &) = delete;
    SfxTokenizer & operator=(const SfxTokenizer &) = delete;

    std::vector<int32_t> encode(const std::string & text) const;
    uint64_t storage_bytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tts_cpp::moss::detail
