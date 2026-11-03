#pragma once

#include "ggml-opt.h"
#include "llama.h"

#include <cstdint>
#include <string>

// Build an SFT dataset (tokens, next-token labels, assistant-only loss masks)
// from JSONL chat conversations rendered through the model's chat template.
ggml_opt_dataset_t common_opt_sft_dataset_init(
        struct llama_context * ctx,
        const std::string    & json_content,
        int64_t                stride,
        const std::string    & chat_template_path = "");
