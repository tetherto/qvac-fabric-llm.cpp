#include "load-into-memory.h"

#include <fstream>
#include <streambuf>

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

std::unique_ptr<std::basic_streambuf<char>> common_load_into_memory::load_file_into_streambuf(
    const char * const model_path) {
    return std::make_unique<ggml_binary_stream_buffer>(load_file_into_buffer(model_path));
}

std::vector<common_load_into_memory::llama_file_entry> common_load_into_memory::load_files_into_streambuf(
    const char * const model_path) {
    std::vector<llama_file_entry> files;

    // Extract pattern from first file path
    std::string path(model_path);

    // Split by '-'
    std::vector<std::string> parts;
    std::stringstream        ss(path);
    std::string              item;
    while (std::getline(ss, item, '-')) {
        parts.push_back(item);
    }

    // Split the last part by '.'
    std::string last_part = parts.back();
    parts.pop_back();
    size_t dot_pos = last_part.find('.');
    if (dot_pos != std::string::npos) {
        parts.push_back(last_part.substr(0, dot_pos));
        parts.push_back(last_part.substr(dot_pos + 1));  // extension
    } else {
        parts.push_back(last_part);
    }

    // Check if we have enough parts
    if (parts.size() < 4) {
        fprintf(stderr, "Model path does not contain expected pattern\n");
        exit(EXIT_FAILURE);
    }

    // Get total files from [-2] position (before the extension)
    int total_files = std::stoi(parts[parts.size() - 2]);

    // Get base path by joining all parts except -start-of-end.gguf
    std::string base_path;
    for (size_t i = 0; i < parts.size() - 4; i++) {
        if (i > 0) {
            base_path += "-";
        }
        base_path += parts[i];
    }

    for (int i = 1; i <= total_files; i++) {
        char numbered_path[1024];
        snprintf(numbered_path, sizeof(numbered_path), "%s-%05d-of-%05d.gguf", base_path.c_str(), i, total_files);

        files.push_back({ numbered_path, load_file_into_streambuf(numbered_path) });
    }

    return files;
}

common_load_into_memory::llama_file_entry common_load_into_memory::load_tensor_list_file(
    const char * const model_path) {
    std::string path(model_path);

    // Split by '-'
    std::vector<std::string> parts;
    std::stringstream        ss(path);
    std::string              item;
    while (std::getline(ss, item, '-')) {
        parts.push_back(item);
    }

    // Split the last part by '.'
    std::string last_part = parts.back();
    parts.pop_back();
    size_t dot_pos = last_part.find('.');
    if (dot_pos != std::string::npos) {
        parts.push_back(last_part.substr(0, dot_pos));
        parts.push_back(last_part.substr(dot_pos + 1));  // extension
    } else {
        parts.push_back(last_part);
    }

    // Check if we have enough parts
    if (parts.size() < 4) {
        fprintf(stderr, "Model path does not contain expected pattern\n");
        exit(EXIT_FAILURE);
    }

    // Get base path by joining all parts except -start-of-end.gguf
    std::string base_path;
    for (size_t i = 0; i < parts.size() - 4; i++) {
        if (i > 0) {
            base_path += "-";
        }
        base_path += parts[i];
    }

    // Construct tensor list file path
    std::string tensor_list_path = base_path + ".tensors.txt";

    printf("Loading tensor list file: %s\n", tensor_list_path.c_str());
    return { tensor_list_path, load_file_into_streambuf(tensor_list_path.c_str()) };
}
