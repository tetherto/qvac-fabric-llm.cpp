#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::pocket::detail {
// Header validation and decoding share one open file. No PCM is allocated by
// construction; metadata-only preflight uses the same limits as model loading.
class ReferenceAudio {
public:
    explicit ReferenceAudio(const std::string & path);
    ~ReferenceAudio();
    uint64_t file_bytes() const;
    uint64_t frames() const;
    int sample_rate() const;
    std::vector<float> read_mono();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
