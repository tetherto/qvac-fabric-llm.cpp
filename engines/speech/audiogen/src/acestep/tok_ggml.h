#pragma once

// ACE-Step FSQ tokenizer — ggml compute engine (the detokenizer's inverse).
//
// Bridges VAE latents to LM audio semantic codes:
//   latents [64, T_25Hz] --audio_acoustic_proj(64->2048)--> embed_tokens
//   --prepend CLS--> 2L Qwen3 encoder (S=6, non-causal) --norm--> CLS column
//   --project_in(2048->6)--> FSQ quantize --> integer code, one per 5 frames.
//
// Weights live in the DiT GGUF under "tokenizer.*"; the trailing partial
// group is padded with the GGUF's own silence_latent frames. The 2-layer
// encoder is the shared Qwen3 backbone (qwen3_block.h). Ported from
// acestep.cpp/src/fsq-tok.h. No new ggml op.

#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <vector>

namespace tts_cpp::acestep {

struct TokModel;  // opaque

// Load the tokenizer weights from the DiT GGUF onto `backend` (borrowed).
TokModel * tok_model_load(const std::string & path, ggml_backend_t backend, bool verbose);
void       tok_model_free(TokModel * m);
size_t     tok_model_weight_bytes(const TokModel * m);

// Tokenize T_25Hz latent frames (frame t of channel c at index t*64 + c)
// into FSQ codes. codes_out is filled with ceil(T_25Hz / 5) codes; the last
// group is padded with silence frames when T_25Hz is not a multiple of 5.
// Returns the code count on success, -1 on failure.
int tok_model_encode(TokModel * m, const float * latents, int T_25Hz, std::vector<int> & codes_out);

// FSQ encode: 6 raw projector outputs -> flat integer index (strides
// {8,8,8,5,5,5}), the inverse of fsq_decode_index. Weight-free / pure;
// exposed for unit testing.
int fsq_encode_index(const float * raw_vals);

} // namespace tts_cpp::acestep
