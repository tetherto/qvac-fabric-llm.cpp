#ifndef LLAMA_COREML_FFN_H
#define LLAMA_COREML_FFN_H

#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

struct llama_coreml_ffn_params {
    // Writable application cache directory; templates are embedded in the library.
    const char * cache_dir;
    int32_t metal_channels;
    int32_t n_packages;
    int32_t min_tokens;
};

LLAMA_API struct llama_coreml_ffn_params llama_coreml_ffn_default_params(void);
LLAMA_API bool llama_supports_coreml_ffn(void);

// Optional Qwen3.8-27B FFN acceleration, built with LLAMA_COREML=ON (or LLAMA_COREML_FFN=ON).
// Requires a model loaded from an unchanged single-file GGUF on disk.
// Call after context creation, before inference. NULL params disables it.
// Configuration is context-owned; failure preserves the previous configuration.
// Active LoRA adapters, training and small batches use the normal FFN.
// The caller must serialize configuration and inference on the same context.
// A prediction failure fails that encode/decode only; discard that request's KV state, later calls use the sidecar again.
LLAMA_API bool llama_set_coreml_ffn(
        struct llama_context * ctx, const struct llama_coreml_ffn_params * params,
        char * error, size_t error_size);

#ifdef __cplusplus
}
#endif
#endif
