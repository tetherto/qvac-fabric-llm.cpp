#pragma once

#include <cstdio>
#include <string>
#include <vector>

namespace tts_cpp::acestep {

struct WavReadResult {
    std::vector<float> pcm;
    int                frames = 0;
    int                sample_rate = 0;
    std::string        error;
};

WavReadResult read_pcm16_wav(FILE * file);
WavReadResult load_pcm16_wav(const char * path);

} // namespace tts_cpp::acestep
