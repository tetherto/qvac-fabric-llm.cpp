#pragma once

#include <string>

namespace tts_cpp::moss::detail {

std::string clean_prompt(const std::string & prompt);
int seconds_in_tenths(double seconds);
std::string duration_prompt(const std::string & prompt, int tenths);

} // namespace tts_cpp::moss::detail
