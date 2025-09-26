#pragma once

#include "ggml-binary-stream-buffer.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <memory>
#include <sstream>
#include <streambuf>
#include <string>
#include <vector>

namespace common_load_into_memory {
bool is_split_file(const char * const model_path);

std::vector<uint8_t> load_file_into_buffer(const char * const model_path);

std::unique_ptr<std::basic_streambuf<char>> load_file_into_streambuf(const char * const model_path);

struct llama_file_entry {
    std::string                                 path;
    std::unique_ptr<std::basic_streambuf<char>> streambuf;
};

std::vector<llama_file_entry> load_files_into_streambuf(const char * const model_path);
llama_file_entry              load_tensor_list_file(const char * const model_path);
}  // namespace common_load_into_memory
