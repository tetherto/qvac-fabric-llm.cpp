#pragma once

// GDN prefill offload (kernels/gdn_prefill.py): replaces the CPU
// GGML_OP_GATED_DELTA_NET execution of a prefill recurrent layer (M > 1
// tokens) by chained device runs of the CS=64-token bf16 recurrence kernel.
// The DMA schedule is a host-built TXN stream (xdna_gdn_prefill_seq_build),
// so no .insts.bin is read at runtime. Anything outside the baked geometry
// (S=128 H=16 CS=64, single sequence, K=1, scalar gate) falls back to the CPU.

#include "ggml.h"

struct xdna_device;

// True when the xclbin artifact is present.

// True when `node` is a GGML_OP_GATED_DELTA_NET prefill op the baked kernel
// covers (see header comment). False keeps the node on the CPU.
bool xdna_gdn_prefill_supported(const struct ggml_tensor * node);

// Run the op on the device: pack q/k/v/g/beta of every CS=64-token chunk, seed
// the bf16 strip state from the llama cache state, chain the chunks on the
// device, and write the ggml result (attn scores + final state) back into
// `node`. Synchronous (submit + wait per chunk). Returns false on failure.
bool xdna_gdn_prefill_run(struct xdna_device * dev, struct ggml_tensor * node);
