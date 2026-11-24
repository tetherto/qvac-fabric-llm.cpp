#pragma once

#include "audiogen-cpp/acestep/engine.h"

#include <string>
#include <vector>

namespace tts_cpp::acestep::music_cli {

bool parse_edit_plan_json(const std::string &json,
                          std::vector<AudioEditParams> &plan,
                          std::string &error);
bool parse_edit_flags(int argc, char **argv, GenerateParams &params,
                      std::string &error);
bool validate_edit_configuration(const GenerateParams &params,
                                 std::string &error);

} // namespace tts_cpp::acestep::music_cli
