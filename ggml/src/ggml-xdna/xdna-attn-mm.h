#pragma once

// Prefill attention on the mmul (kernels/attn_mm.py) for Qwen3.5's six
// full-attention layers: GGML_OP_FLASH_ATTN_EXT with 8 query heads, 2 KV heads,
// D = 256, scale 1/16 and a plain causal mask, the f16 cache in llama's own
// layout. GGML_XDNA_FA_MM=0 leaves it to the host.

#include <stddef.h>

struct ggml_tensor;
struct xdna_kernel_pool;

// True when `node` is an attention the design covers (the mask is checked,
// not assumed) and the artifact is present.
bool xdna_attn_mm_supported(const struct ggml_tensor * node);

// Run it to completion and write the f32 output.
bool xdna_attn_mm_run(struct xdna_kernel_pool * pool, struct ggml_tensor * node);

// The output left where the array wrote it, for a reader that takes its
// rows from there (xdna_pgemm_run_gate): planned per graph with
// xdna_attn_mm_keep, the run of such a node writes nothing into its node.
// Token t, head h's 256 values are two halves of 128 (xdna_attn_mm_row, the
// second O_OBJ = 4096 floats after the first) until the next run;
// xdna_attn_mm_materialize
// writes it into the node (for a reader that falls back to the host).
struct xdna_attn_rows {
    const float * base     = nullptr;
    size_t        o_col    = 0;  // floats a column
    long long     n_tokens = 0;
};

// the two 128-value halves of token t, head h (the second 4096 floats on)
static inline const float * xdna_attn_mm_row(const struct xdna_attn_rows * r, long long t, int h) {
    const int p = (int) (t / 64), pos = (int) (t % 64), pidx = pos / 8, g = h / 4;
    const int c = 4 * g + pidx / 2, i0 = 2 * (pidx % 2), row = (h % 4) * 8 + pos % 8;
    return r->base + (size_t) c * r->o_col + ((size_t) p * 4 + i0) * 4096 + (size_t) row * 128;
}

// A new graph: its mask may be the last one's memory rewritten.
void xdna_attn_mm_graph_begin(void);
void xdna_attn_mm_keep_clear(void);
void xdna_attn_mm_keep(const struct ggml_tensor * node);
bool xdna_attn_mm_rows(const struct ggml_tensor * node, struct xdna_attn_rows * rows);
bool xdna_attn_mm_materialize(const struct ggml_tensor * node);
