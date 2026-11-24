#pragma once

#include <string>
#include <vector>

namespace tts_cpp::acestep::music_cli {

std::string read_text_file(const char *path);
bool json_field(const std::string &json, const char *key, std::string &value);
std::vector<std::string> json_array_objects(const std::string &json,
                                            const char *key);

} // namespace tts_cpp::acestep::music_cli
