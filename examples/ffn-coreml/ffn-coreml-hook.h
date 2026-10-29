#pragma once

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ggml_tensor * (*llama_ffn_coreml_builder)(struct ggml_context *, struct ggml_tensor *, struct ggml_tensor *, int, void *);

LLAMA_API void llama_ffn_coreml_set_builder(llama_ffn_coreml_builder builder, void * userdata);

#ifdef __cplusplus
}
#endif
