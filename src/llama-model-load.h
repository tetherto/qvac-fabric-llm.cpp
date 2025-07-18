#pragma once

#include <cstdint>
#include <set>

#include "ggml-cpp.h"
#include "llama-mmap.h"
#include "llama-model-load-input.h"

struct llama_model_loader;

/// @brief Immediately loads and stores relevant data in the struct fields.
struct gguf_file_load {
    struct gguf_init_params     params;
    gguf_context_ptr            meta;
    std::unique_ptr<llama_file> file = nullptr;

    gguf_file_load(struct ggml_context ** ctx, load_input_t load_input);
};
