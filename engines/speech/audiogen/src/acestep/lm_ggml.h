#pragma once

// ACE-Step LM (ace-lm, Qwen3 0.6B) — ggml compute engine.
//
// Autoregressive Qwen3 causal LM with a persistent KV cache: prefill the prompt
// then decode audio-code / text tokens one step at a time. Same Qwen3 backbone
// as the encoders (shared qwen3_block.h) plus KV-cache read/write via set_rows
// and a tied LM head (logits = embed_tokens^T @ hidden). Ported from
// acestep.cpp/src/qwen3-lm.h. No new ggml op.
//
// This is the model core. It keeps independent KV sets for conditional and
// unconditional decoding, uses F32 flash attention plus batched CFG when the
// selected GPU advertises support, and retains the F32 manual-attention path as
// the CPU/unsupported-backend fallback. The BPE tokenizer, metadata FSM and
// top-k/p sampling live above it in the pipeline.

#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace tts_cpp::acestep {

struct LMConfig {
    int   vocab_size   = 0;
    int   hidden_size  = 0;
    int   n_heads      = 0;
    int   n_kv_heads   = 0;
    int   head_dim     = 0;
    int   n_layers     = 0;
    float rope_theta   = 1000000.0f;
    float rms_norm_eps = 1e-6f;
    int   max_seq_len  = 4096;  // KV cache capacity
};

struct LMModel;  // opaque
struct DitGGUF;
struct Qwen3Layer;
struct AcestepStageMeasure;  // fit_measure.h

// Fused-load internals, exposed for unit tests: copy one GGUF tensor into the
// row-concatenated dst at byte offset `off` (advancing it), and load one
// layer's q|k|v + gate|up blocks. Both return false on a missing tensor so a
// corrupt GGUF fails the load instead of leaving misaligned fused weights.
bool lm_load_row_block(ggml_tensor * dst, size_t & off, const DitGGUF & g, const std::string & name);
bool lm_load_layer_fused(const DitGGUF & g, const std::string & prefix, Qwen3Layer & ly, ggml_tensor * qkv,
                         ggml_tensor * gateup);

// Split tied table, exposed for unit tests. A backend that cannot look up the
// embedding table or project the whole vocabulary in one MUL_MAT gets this
// layout from lm_model_load: the lookup dequantizes GGUF rows on the host and
// the tied head runs as row chunks, cut at the pipeline's logit-range edges
// and at most `max_rows` long. lm_model_load_split_table forces it with
// `head_chunk_rows`-row chunks on any backend; lm_head_chunk_ranges returns the
// (first row, rows) of each chunk.
LMModel * lm_model_load_split_table(const std::string & path, ggml_backend_t backend, int max_seq_len, int n_kv_sets,
                                    int head_chunk_rows);
std::vector<std::pair<int, int>> lm_head_chunk_ranges(int vocab, int max_rows);

// Load ace-lm GGUF onto `backend` (borrowed). Config is derived from tensor
// shapes (H, V, layer count, head counts). `n_kv_sets` independent KV caches are
// allocated (>=2 enables classifier-free guidance: cond=0, uncond=1). Returns
// nullptr on failure.
LMModel *        lm_model_load(const std::string & path, ggml_backend_t backend, int max_seq_len, bool verbose,
                               int n_kv_sets = 1);
void             lm_model_free(LMModel * m);
const LMConfig & lm_model_config(const LMModel * m);
size_t           lm_model_weight_bytes(const LMModel * m);
int              lm_num_kv_sets(const LMModel * m);
bool             lm_model_embeddings_quantized(const LMModel * m);

// Metadata-only load for the memory-fit preflight: identical backend
// resolution, tensor wiring (fusion decision included), and KV-cache layout to
// lm_model_load, but every allocation is SIZED into `measure` instead of
// performed and no tensor data is read. Only good for the measure-mode
// forwards below; free with lm_model_free.
LMModel * lm_model_load_metadata_only(const std::string & path, ggml_backend_t backend, int max_seq_len,
                                      bool verbose, int n_kv_sets, AcestepStageMeasure & measure);

// KV-cache buffer bytes of a real load (0 on a metadata-only model, whose KV
// bytes are reported in its AcestepStageMeasure instead).
size_t lm_model_kv_bytes(const LMModel * m);

// Compute-buffer bytes of the persistent forward-graph cache (0 when no graph
// has been built); lets the fit parity tests compare projection vs the real
// resident allocation.
size_t lm_model_compute_buffer_bytes(const LMModel * m);

// Bytes of the compact tied-head copy lm_build_partial_head would allocate for
// `count` rows (same tensor shape/type, sized not allocated); 0 for a split
// tied table, whose row chunks serve every range without a copy.
size_t lm_measure_partial_head_bytes(const LMModel * m, int count);

// Size-only twins of the two forward paths, for the memory-fit preflight.
// Build the identical graph at the given shape and write its compute-buffer
// size (ggml_gallocr_reserve_n_size) -- nothing is allocated, uploaded, or
// computed, the graph cache and KV positions are left untouched.
//   measure_prefill: one forward over n_tokens at KV position 0.
//   measure_decode: one single-stream decode step whose KV window has reached
//     kv_len (the graph's mask/view size is GGML_PAD(kv_len, 256) capped at
//     max_seq_len -- pass max_seq_len for the worst case).
//   measure_decode_batch: the batched-CFG decode step (N consecutive sets) at
//     kv_len, with the compact head at logit_offset (0 = full head). Fails
//     (returns false) where the real path would: no flash attention or
//     n_kv_sets < N.
bool lm_model_measure_prefill(LMModel * m, int n_tokens, int logit_limit, size_t & compute_bytes);
bool lm_model_measure_decode(LMModel * m, int kv_len, int logit_limit, size_t & compute_bytes);
bool lm_model_measure_decode_batch(LMModel * m, int N, int kv_len, int logit_offset, size_t & compute_bytes);

// Reset the KV cache for one set (start a new sequence).
void lm_reset(LMModel * m, int set = 0);
int  lm_kv_pos(const LMModel * m, int set = 0);

// Run one forward over `n_tokens` (prefill: n_tokens>1, decode: 1) at the
// current KV position of `set`, appending to that cache. Writes the last token's
// logits [vocab_size] to `logits_out`. Returns false on failure.
// When `layer_states_out` is non-null it receives each layer's hidden state
// [hidden_size * n_tokens] concatenated in layer order, for CPU/GPU parity
// debugging. It forces those tensors to stay resident, so leave it null in
// production paths.
// logit_limit > 0 projects only the tied-head prefix rows [0, logit_limit) and
// returns that many logits - valid only when the caller can never select a
// token past the limit (FSM-constrained Phase 1).
// `measure_compute` non-null: size-only mode (see lm_model_measure_prefill),
// token_ids may be null; prefer the lm_model_measure_* wrappers.
bool lm_model_forward(LMModel * m, const int32_t * token_ids, int n_tokens, std::vector<float> & logits_out,
                      int set = 0, std::vector<float> * layer_states_out = nullptr, int logit_limit = 0,
                      size_t * measure_compute = nullptr);

// Decode one token for each KV set in a single batched graph. `sets` must name
// consecutive caches and `logit_offset` optionally projects only the tied-head
// rows [logit_offset, vocab_size), matching ACE-Step's Phase-2 compact head.
// Returns logits as N consecutive vectors of size vocab_size-logit_offset.
bool lm_model_forward_batch(LMModel * m, const int32_t * token_ids, const int * sets, int n,
                            std::vector<float> & logits_out, int logit_offset = 0,
                            size_t * measure_compute = nullptr);
bool lm_model_supports_batched_decode(const LMModel * m);

} // namespace tts_cpp::acestep
