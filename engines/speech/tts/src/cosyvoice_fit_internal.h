#pragma once

// Library-internal measurement surface for the CosyVoice3 memory-fit
// preflight (include/tts-cpp/cosyvoice/fit.h).  Implemented in
// cosyvoice_pipeline.cpp next to the loaders and stage graph builders it
// prices, so the priced graph is the executed graph by construction; consumed
// by src/cosyvoice_fit.cpp and the parity test.
//
// The engine's pipeline is STAGED: llm / flow / hift are loaded, run, and
// freed one at a time inside run() (cosyvoice_engine.cpp), so the resident
// peak is the largest single phase, not the sum.  Each measure function below
// prices ONE phase; the fitter takes the max.

#include "cosyvoice_pipeline.h"

#include <cstdint>
#include <string>

// Sizes of the buffers one cosyvoice_load_gguf allocates, filled by
// cosyvoice_load_gguf_metadata_only.
struct cosyvoice_fit_load_measure {
    // ctx_w on the load backend (the whole GGUF on CPU; everything but the
    // host-resident lookup tables on GPU).
    uint64_t device_bytes = 0;
    // ctx_h on the CPU buffer type (GPU loads keep lm/embed_tokens and
    // lm/speech_embedding host-resident; 0 on CPU loads and non-LM GGUFs).
    uint64_t host_bytes = 0;
};

// Metadata-only twin of cosyvoice_load_gguf: same backend borrowing, tensor
// wiring, host/device split, and KV-metadata capture, but the weight buffers
// are SIZED instead of allocated and no tensor data leaves the disk.  The
// mmap-in-place path is intentionally bypassed: the projection prices the
// allocate-and-stream fallback, which bounds the fully-touched mapping from
// above (wrong only in the strict direction).  All tensors come back marked
// externally allocated so graph pricing counts only compute scratch.  Free
// with cosyvoice_free as usual (no buffers exist in measure mode).
model_ctx cosyvoice_load_gguf_metadata_only(const std::string & path,
                                            ggml_backend_t backend,
                                            cosyvoice_fit_load_measure & out);

// One priced quantity: the device portion plus the CPU-fallback portion of
// scheduler-dispatched graphs (charged to host).
struct cosyvoice_fit_price {
    uint64_t device_bytes = 0;
    uint64_t host_bytes   = 0;
};

// LLM phase: sizes the KV slab exactly as qwen_kvcache::init allocates it for
// max_P = L0 + max_steps + 1 positions, and prices the decode arena as the
// max of the prefill graph (Lq = L0) and the deepest decode step
// (Lq = 1 at cache position L0 + max_steps - 1) -- one gallocr serves the
// whole decode, so it grows to the larger.  `m` must be a metadata-only LM
// model.
bool cosyvoice_fit_measure_llm(model_ctx & m, const qwen_hp & hp, int L0, int max_steps,
                               uint64_t & kv_bytes, cosyvoice_fit_price & arena,
                               std::string * error);

// Flow phase: prices the front-end graph at T_tok tokens and the DiT graph at
// N = TM frames (CFG batch B = 2, exactly as run_euler_steps builds it).
// Their allocators never coexist (the front-end's is freed before the DiT's
// is created), so the phase arena is the max.  `m` must be a metadata-only
// flow model.
bool cosyvoice_fit_measure_flow(model_ctx & m, int T_tok, int TM, int SPK,
                                cosyvoice_fit_price & arena, std::string * error);

// HiFT phase: prices the f0-predictor graph (T_mel), the STFT graph
// (T_src = 480 * T_mel excitation samples), and the decode graph
// (T_mel, T_stft) -- run sequentially with their own allocators, so the phase
// arena is the max.  `m` must be a metadata-only HiFT model.
bool cosyvoice_fit_measure_hift(model_ctx & m, int T_mel,
                                cosyvoice_fit_price & arena, std::string * error);

// ---- real-allocation parity probes (test support) ---------------------------
// Byte-parity anchors for the measures above: each drives the SAME graph
// builders on a REAL model through the real dispatch (gallocr reserve + alloc,
// and for the LLM a real prefill + decode-step compute against a real KV
// cache), reports the allocated sizes, and frees everything.  The runtime
// stage functions free their allocators internally, so these probes are the
// way a test observes what one phase actually allocates.

bool cosyvoice_fit_llm_parity_probe(model_ctx & m, const qwen_hp & hp, int L0, int n_steps,
                                    uint64_t & kv_bytes, uint64_t & arena_bytes,
                                    std::string * error);
bool cosyvoice_fit_flow_parity_probe(model_ctx & m, int T_tok, int TM, int SPK,
                                     uint64_t & frontend_bytes, uint64_t & dit_bytes,
                                     std::string * error);
bool cosyvoice_fit_hift_parity_probe(model_ctx & m, int T_mel,
                                     uint64_t & f0_bytes, uint64_t & stft_bytes,
                                     uint64_t & decode_bytes, std::string * error);
