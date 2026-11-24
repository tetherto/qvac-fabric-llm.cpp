#pragma once

#include "tts-cpp/moss/engine.h"
#include "tts-cpp/moss/sound_effect.h"
#include "tts-cpp/moss/speech.h"

#include <string>

namespace tts_cpp::moss::cli {

enum class Mode { Speech, SoundEffect, SpeechToSpeech };

struct CliArgs {
    Mode mode = Mode::Speech;
    EngineOptions options;
    SoundEffectOptions sound_options;
    SoundEffectRequest sound_request;
    SpeechOptions s2s_options;
    SpeechRequest s2s_request;
    std::string audio_path;
    std::string voice_path;
    std::string system_prompt;
    std::string text;
    std::string out_path = "moss-out.wav";
    bool stream = false;
};

void print_usage();
bool parse_args(int argc, const char * const * argv, CliArgs & args);
int run(const CliArgs & args);

} // namespace tts_cpp::moss::cli
