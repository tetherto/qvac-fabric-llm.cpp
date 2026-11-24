#pragma once

// ACE-Step FSQ detokenizer — ggml compute engine.
//
// Bridges the LM audio semantic codes to the DiT context latents:
//   codes [T_5Hz] --FSQ decode--> [6, T_5Hz] --project_out--> [2048, T_5Hz]
//   per 5Hz token: embed + special_tokens broadcast (5 frames) + 2L Qwen3
//   encoder + norm + proj_out --> [64, 5]  => context latents [64, T_25Hz]
//   with T_25Hz = T_5Hz * 5.
//
// Weights live in the DiT GGUF under "tokenizer.quantizer.*" and
// "detokenizer.*". The 2-layer encoder is the shared Qwen3 backbone
// (qwen3_block.h). Ported from acestep.cpp/src/fsq-detok.h. No new ggml op.

#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <vector>

namespace tts_cpp::acestep {

struct DetokModel;           // opaque
struct AcestepStageMeasure;  // fit_measure.h

// Load the detokenizer weights from the DiT GGUF onto `backend` (borrowed).
DetokModel * detok_model_load(const std::string & path, ggml_backend_t backend, bool verbose);
void         detok_model_free(DetokModel * m);
size_t       detok_model_weight_bytes(const DetokModel * m);

// Metadata-only load for the memory-fit preflight: identical tensor wiring to
// detok_model_load, but the weight allocation is SIZED into `measure` instead
// of performed and no tensor data is read. Only good for the measure-mode
// decode below; free with detok_model_free.
DetokModel * detok_model_load_metadata_only(const std::string & path, ggml_backend_t backend,
                                            bool verbose, AcestepStageMeasure & measure);

// Compute-buffer bytes of the most recent real detok_model_decode (0 before
// the first); lets the fit parity tests compare projection vs real allocation.
size_t detok_model_compute_buffer_bytes(const DetokModel * m);

// Decode T_5Hz LM codes into context latents. context_out is filled with
// [64, T_25Hz] (frame t of channel c at index t*64 + c), T_25Hz = T_5Hz * 5.
// Returns T_25Hz on success, -1 on failure. Caller sizes context_out to
// 64 * T_5Hz * 5 floats.
// When `measure_compute` is non-null the call builds the identical (fixed
// S=5) per-token graph but only SIZES its compute buffer into it -- no
// allocation, upload, or compute; `codes` / `context_out` may then be null.
int detok_model_decode(DetokModel * m, const int * codes, int T_5Hz, float * context_out,
                       size_t * measure_compute = nullptr);

// FSQ decode: integer index -> 6 normalized floats in [-1, 1] (strides
// {8,8,8,5,5,5}). Weight-free / pure; exposed for unit testing.
void fsq_decode_index(int index, float * out);

} // namespace tts_cpp::acestep
