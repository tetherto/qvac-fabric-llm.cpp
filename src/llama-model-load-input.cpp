#include "llama-model-load-input.h"

namespace load_input_variant {

const char * identifier(load_input_t & load_input) {
    if (std::holds_alternative<fname_load_input>(load_input)) {
        const auto & file_input = std::get<fname_load_input>(load_input);
        return file_input.fname.c_str();
    }
    static const char * buffer_id_str = "buffer";
    return buffer_id_str;
}

fname_load_input split_name_from_variant(load_input_t & load_input) {
    auto file_input = std::get<fname_load_input>(load_input);
    return file_input;
}

bool variant_supports_split_load(load_input_t & load_input) {
    return std::holds_alternative<fname_load_input>(load_input);
}

}  // namespace load_input_variant
