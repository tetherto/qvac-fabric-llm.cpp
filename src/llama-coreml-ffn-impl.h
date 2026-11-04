#pragma once

#include "llama-coreml-ffn.h"
#include "ggml.h"

struct llama_model;
struct llama_coreml_ffn;

llama_coreml_ffn * llama_coreml_ffn_load(const llama_model & model, const llama_coreml_ffn_params & params);
int llama_coreml_ffn_neural_engine_cores();
void llama_coreml_ffn_free(llama_coreml_ffn * runtime);
bool llama_coreml_ffn_check(llama_coreml_ffn * runtime);
ggml_tensor * llama_coreml_ffn_build(llama_coreml_ffn * runtime, ggml_context * ctx, ggml_cgraph * graph,
        ggml_tensor * input, ggml_tensor * residual, ggml_tensor * gate, ggml_tensor * up, ggml_tensor * down, int layer);
