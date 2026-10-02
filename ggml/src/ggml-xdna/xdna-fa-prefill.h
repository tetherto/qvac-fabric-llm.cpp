#pragma once

// GGML_OP_FLASH_ATTN_EXT offload for the six full-attention prefill layers of
// Qwen3.5, via kernels/fa.py + kernels/attn-fa.cc. Causal attention with D=256, 8 query
// heads against 2 KV heads, f16 cache; a dispatch covers 128 queries and 512
// keys of every head, and the host chains the rounds and the key chunks.
// Auto-active when the artifact is present; anything outside that geometry
// stays on the CPU.

#include "ggml.h"

struct xdna_device;

// True when the xclbin + insts artifacts are present.

// True when `node` is a GGML_OP_FLASH_ATTN_EXT the kernel covers: the geometry
// above, scale 1/sqrt(256), no ALiBi, no logit softcap, no sinks, and a mask
// that is plain causal (checked, not assumed).
bool xdna_fa_prefill_supported(const struct ggml_tensor * node);

// Run the op. Synchronous. Returns false on failure (caller keeps the CPU
// fallback).
bool xdna_fa_prefill_run(struct xdna_device * dev, struct ggml_tensor * node);
