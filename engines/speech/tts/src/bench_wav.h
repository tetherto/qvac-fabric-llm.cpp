#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

// Shared version of the benchmark PCM16 RIFF writer. Validate before opening
// the destination and check buffered writes, including errors on close.
inline bool bench_write_wav(const std::string & path, const std::vector<float> & pcm,
                            int sample_rate, std::string & error) {
    if (pcm.empty() || sample_rate <= 0 ||
        static_cast<uint64_t>(sample_rate) * 2 > UINT32_MAX ||
        pcm.size() > (UINT32_MAX - 36ULL) / 2) {
        error = "invalid PCM length or sample rate";
        return false;
    }
    for (float sample : pcm) {
        if (!std::isfinite(sample)) {
            error = "non-finite PCM sample";
            return false;
        }
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) { error = "cannot open WAV output: " + path; return false; }
    // RIFF fields and PCM samples are little-endian on every host.
    auto le = [&](uint32_t value, int count) {
        for (int i = 0; i < count; ++i) out.put(static_cast<char>((value >> (8 * i)) & 0xff));
    };
    const uint32_t data_size = static_cast<uint32_t>(pcm.size() * 2);
    out.write("RIFF", 4); le(36 + data_size, 4); out.write("WAVEfmt ", 8);
    le(16, 4); le(1, 2); le(1, 2); le(static_cast<uint32_t>(sample_rate), 4);
    le(static_cast<uint32_t>(sample_rate) * 2, 4); le(2, 2); le(16, 2);
    out.write("data", 4); le(data_size, 4);
    for (float sample : pcm) {
        const float clipped = std::max(-1.0f, std::min(1.0f, sample));
        le(static_cast<uint16_t>(static_cast<int16_t>(std::lrintf(clipped * 32767.0f))), 2);
    }
    out.close();
    if (!out) { error = "failed to write WAV output: " + path; return false; }
    return true;
}
