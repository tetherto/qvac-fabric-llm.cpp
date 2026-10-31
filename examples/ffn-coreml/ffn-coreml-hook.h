#pragma once

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ggml_tensor * (*llama_ffn_coreml_builder)(struct ggml_context *, struct ggml_tensor *, struct ggml_tensor *, int, void *);

LLAMA_API void llama_ffn_coreml_set_builder(llama_ffn_coreml_builder builder, void * userdata);

// For a 1024-token microbatch run as two interleaved 512-token halves: stage 0 starts the FFN of one half
// (input, residual) and returns a handle, stage 1 returns the FFN output plus residual for that handle.
// Only used when LLAMA_QWEN35_HALVES is set.
typedef struct ggml_tensor * (*llama_ffn_coreml_split_builder)(struct ggml_context *, struct ggml_tensor *, struct ggml_tensor *, int, int, void *);

LLAMA_API void llama_ffn_coreml_set_split_builder(llama_ffn_coreml_split_builder builder, void * userdata);

#ifdef __cplusplus
}
#endif
