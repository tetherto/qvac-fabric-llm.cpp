#pragma once

// ACE-Step text encoder (Qwen3-Embedding-0.6B) — ggml compute engine.
//
// The prompt encoder: a standard Qwen3 backbone (RMSNorm, GQA self-attention with
// per-head QK-norm + NEOX RoPE, SwiGLU MLP, causal). Token IDs -> hidden states
// that feed the DiT cross-attention (via the cond-encoder). Ported from
// acestep.cpp/src/qwen3-enc.h. Every op already exists in the fabric ggml —
// no new custom op. On CPU, attention uses the F32 soft_max_ext path.
//
// Layout: hidden [H, S] == ggml ne[0]=H, ne[1]=S.

#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <vector>

namespace tts_cpp::acestep {

// Qwen3-Embedding-0.6B text-encoder config (fixed model; matches upstream).
struct TextEncConfig {
    int   hidden_size       = 1024;
    int   intermediate_size = 3072;
    int   n_heads           = 16;
    int   n_kv_heads        = 8;
    int   head_dim          = 128;
    int   n_layers          = 28;
    float rope_theta        = 1000000.0f;
    float rms_norm_eps      = 1e-6f;
    bool  is_causal         = true;
};

struct TextEncModel;         // opaque: weight tensors + backend weight buffer
struct AcestepStageMeasure;  // fit_measure.h

// Load the Qwen3-Embedding GGUF onto `backend` (borrowed). Returns nullptr on
// failure. Q/K/V and gate/up are loaded as separate tensors (no fusion).
TextEncModel *        textenc_model_load(const std::string & path, ggml_backend_t backend, bool verbose);
void                  textenc_model_free(TextEncModel * m);
const TextEncConfig & textenc_model_config(const TextEncModel * m);
size_t                textenc_model_weight_bytes(const TextEncModel * m);

// Metadata-only load for the memory-fit preflight: identical backend
// resolution and tensor wiring to textenc_model_load, but every allocation is
// SIZED into `measure` (ggml_backend_alloc_ctx_tensors_from_buft_size) instead
// of performed, and no tensor data is read. The returned model can only build
// measure graphs (see the `measure_compute` parameters below); it must never
// run a real forward. Free with textenc_model_free.
TextEncModel * textenc_model_load_metadata_only(const std::string & path, ggml_backend_t backend,
                                                bool verbose, AcestepStageMeasure & measure);

// Encode token IDs [S] into hidden states [H, S] (H contiguous per token),
// written to `hidden_out`. Returns false on failure.
// When `measure_compute` is non-null the call builds the identical graph but
// only SIZES its compute buffer (ggml_gallocr_reserve_n_size) into it -- no
// allocation, no upload, no compute; `token_ids` may then be null.
bool textenc_model_forward(TextEncModel * m, const int32_t * token_ids, int S, std::vector<float> & hidden_out,
                           size_t * measure_compute = nullptr);

// Embedding-table lookup only (no transformer): token IDs [S] -> [H, S] rows of
// embed_tokens. Used for the cond-encoder lyric path. Returns false on failure.
// `measure_compute`: same size-only contract as textenc_model_forward.
bool textenc_model_embed_lookup(TextEncModel * m, const int32_t * token_ids, int S, std::vector<float> & embed_out,
                                size_t * measure_compute = nullptr);

// Compute-buffer bytes the most recent real forward allocated (0 before the
// first). Lets the fit parity tests compare the size-only projection against a
// real allocation byte-for-byte.
size_t textenc_model_compute_buffer_bytes(const TextEncModel * m);

} // namespace tts_cpp::acestep
