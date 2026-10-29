#pragma once

// Prefill attention on the host's SIMD units, for the FLASH_ATTN_EXT nodes the glue would hand to ggml-cpu.
// GGML_XDNA_HOST_FA=0 leaves them to ggml-cpu; 2 runs the portable kernels.

struct ggml_tensor;

// True when `node` is an attention this kernel computes: f32 Q, f16 or f32 K
// and V, a head size that is a multiple of 16, no sinks, ALiBi or softcap.
bool xdna_host_attn_supported(const struct ggml_tensor * node);

// Run it on the glue's threads and write the f32 output.
bool xdna_host_attn_run(struct ggml_tensor * node);

// A new graph: its mask may be the last one's memory rewritten.
void xdna_host_attn_graph_begin(void);
