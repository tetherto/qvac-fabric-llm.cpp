#ifndef LLAMA_EMBEDDING_COREML_H
#define LLAMA_EMBEDDING_COREML_H

#include "llama.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct llama_coreml_embedding;

// Tokens must use the original EmbeddingGemma 2 vocabulary, including BOS and EOS.
struct llama_coreml_embedding_sequence {
    const int32_t * tokens;
    int32_t n_tokens;
};

struct llama_coreml_embedding_stats {
    uint64_t calls;
    double preparation_ms;
    double prediction_ms;
    double output_ms;
};

// Sum of per-call wall times since reset; asynchronous prediction times can overlap.
LLAMA_API int32_t llama_coreml_embedding_get_stats(struct llama_coreml_embedding * ctx, struct llama_coreml_embedding_stats * stats, bool reset);

// Directory contains EmbeddingGemma2Text.mlmodelc (or .mlpackage) and embeddings.bf16.
// Uses CPU + ANE compute units. Core ML controls placement; this is not a ggml backend.
LLAMA_API struct llama_coreml_embedding * llama_coreml_embedding_load(const char * directory, char * error, size_t error_size);
// Requires embedding_raw outputs and returns unnormalized mean-pooled vectors.
LLAMA_API struct llama_coreml_embedding * llama_coreml_embedding_load_raw(const char * directory, char * error, size_t error_size);
LLAMA_API void llama_coreml_embedding_free(struct llama_coreml_embedding * ctx);

// One uses synchronous prediction (default); 2..8 use shared-model asynchronous prediction.
// Encode still waits for all outputs, and input/output buffers stay valid until it returns.
LLAMA_API int32_t llama_coreml_embedding_set_max_inflight(struct llama_coreml_embedding * ctx, int32_t max_inflight);

// Writes n_sequences * 768 floats in input order. Output capacity is in floats.
// Accepts 1..512 tokens per sequence; never silently truncates. Calls on one context are serialized.
// Returns 0 on success, -1 on error. Output may be partially written on failure.
LLAMA_API int32_t llama_coreml_embedding_encode(
        struct llama_coreml_embedding * ctx,
        const struct llama_coreml_embedding_sequence * sequences, int32_t n_sequences,
        bool pack, float * output, size_t output_capacity, char * error, size_t error_size);

#ifdef __cplusplus
}
#endif
#endif
