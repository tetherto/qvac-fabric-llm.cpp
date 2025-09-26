#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>

namespace common_load_into_memory {
bool is_split_file(const char * const model_path);

std::vector<uint8_t> load_file_into_buffer(const char * const model_path);
}  // namespace common_load_into_memory
