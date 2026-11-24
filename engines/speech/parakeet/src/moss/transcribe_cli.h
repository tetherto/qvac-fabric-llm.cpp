#pragma once

#include "parakeet/moss_transcribe.h"

#include <string>

namespace parakeet::moss::cli {

struct TranscribeCliArgs {
    TranscribeOptions options;
    TranscribeRequest request;
    std::string audio_path;
    std::string out_path;
};

void print_usage();
bool parse_args(int argc, const char * const * argv, TranscribeCliArgs & args);
int run(const TranscribeCliArgs & args);
std::string transcript_json(const TranscribeResult & result);

} // namespace parakeet::moss::cli
