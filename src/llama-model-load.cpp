#include "llama-model-load.h"

#include <memory>
#include <stdexcept>
#include <variant>

#include "llama-model-loader.h"

gguf_file_load::gguf_file_load(struct ggml_context ** ctx, load_input_t load_input) :
    params({
        /*.no_alloc = */ true,
        /*.ctx      = */ ctx,
    }) {
    using namespace load_input_variant;
    if (std::holds_alternative<fname_load_input>(load_input)) {
        const auto & file_input = std::get<fname_load_input>(load_input);
        meta.reset(gguf_init_from_file(file_input.fname.c_str(), params));
        if (!meta) {
            throw std::runtime_error(format("%s: failed to load model from %s", __func__, file_input.fname.c_str()));
        }
        file = std::make_unique<llama_file_disk>(file_input.fname.c_str(), "ro");
    } else {
        const auto & buffer_input = std::get<buffer_load_input>(load_input);
        meta.reset(gguf_init_from_buffer(*buffer_input.streambuf, params));
        if (!meta) {
            throw std::runtime_error(format("%s: failed to load model from buffer", __func__));
        }
        file = std::make_unique<llama_file_buffer_ro>(std::move(buffer_input.streambuf));
    }
}
