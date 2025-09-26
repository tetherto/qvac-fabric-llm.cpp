#include "load-into-memory.h"

#include <fstream>

bool common_load_into_memory::is_split_file(const char * const model_path) {
    if (!model_path) {
        fprintf(stderr, "No model file provided\n");
        exit(EXIT_FAILURE);
    }

    std::string path(model_path);
    return path.find("-of-") != std::string::npos;
}

std::vector<uint8_t> common_load_into_memory::load_file_into_buffer(const char * const model_path) {
    std::ifstream file_stream(model_path, std::ios::binary | std::ios::ate);
    if (!file_stream) {
        fprintf(stderr, "Failed to open file %s for reading into streambuf\n", model_path);
        exit(EXIT_FAILURE);
    }

    const size_t file_size = file_stream.tellg();
    file_stream.seekg(0, std::ios::beg);

    static_assert(sizeof(std::uint8_t) == sizeof(char), "uint8_t must be same size as char");
    std::vector<std::uint8_t> buffer(file_size);
    if (!file_stream.read((char *) buffer.data(), file_size)) {
        fprintf(stderr, "Failed to read entire file into buffer\n");
        exit(EXIT_FAILURE);
    }

    return buffer;
}
