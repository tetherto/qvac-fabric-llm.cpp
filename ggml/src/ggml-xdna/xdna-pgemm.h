#pragma once

#include <stddef.h>

// The prefill GEMM (kernels/pgemm.py): a
// quantized MUL_MAT at prefill batch size on the whole array, every core
// expanding the decode GEMV's own packed weight tiles into bf16 and running
// the bfp16 mmul. One artifact serves every shape: the cores read their loop
// counts from a header object at the head of the activation stream, and the
// DMA schedule is built here per shape.

struct ggml_tensor;
struct xdna_kernel_pool;

// True when `op` is a MUL_MAT this route runs: a weight the decode GEMV packs
// (or Q8_0, into its 8-bit form),
// at least 64 activation rows, and the artifact present (GGML_XDNA_PGEMM=0
// turns the route off).
bool xdna_pgemm_supported(const struct ggml_tensor * op);

// A graph is starting: the activation layouts kept for the projections that
// share an input belong to the previous one.
void xdna_pgemm_graph_begin(void);

// Drop the packed weights. They are keyed by a tensor's data pointer, which
// does not survive the model that owned it.
void xdna_pgemm_release(void);

// Run `node` to completion and write its f32 output.
bool xdna_pgemm_run(struct xdna_kernel_pool * pool, struct ggml_tensor * node);

// Run `node` into a buffer of the backend's own instead of its data: rows of
// N floats from byte `off` of `bo`, which has room for xdna_pgemm_into_rows(M)
// rows. The node's own data is not written.
struct xdna_buffer;
// The rows a call of M rows writes in place: an even count of 128-row blocks.
size_t xdna_pgemm_into_rows(int M);
bool xdna_pgemm_run_into(struct xdna_kernel_pool * pool,
                         struct ggml_tensor *      node,
                         struct xdna_buffer *      bo,
                         size_t                    off);

// A SWIGLU of two such MUL_MATs of one activation (the FFN's gate and up,
// weights of one type) the array runs whole: one call, the cores applying
// silu(gate) * up before C leaves them (GGML_XDNA_PGEMM_GLU=0: off). The
// caller skips the two MUL_MATs, whose outputs nothing else may read.
bool xdna_pgemm_glu_supported(const struct ggml_tensor * glu);
bool xdna_pgemm_run_glu(struct xdna_kernel_pool * pool, struct ggml_tensor * glu);
// The same SwiGLU written by the cores in bf16 straight into the A layout of
// `down`, the MUL_MAT that reads it (GGML_XDNA_PGEMM_GLU_A=0: off): the GLU's
// own output is not written, and down's call finds its A laid out. The caller
// runs down next and lets nothing else read the GLU's node.
bool xdna_pgemm_glu_fused_supported(const struct ggml_tensor * glu, const struct ggml_tensor * down);
bool xdna_pgemm_run_glu_fused(struct xdna_kernel_pool * pool, struct ggml_tensor * glu);

// An RMS_NORM's output times its weight (`mul`, the MUL of the norm and a
// vector) laid out straight into the A layout of the MUL_MATs that read it
// (GGML_XDNA_PGEMM_NORM_A=0: off), one pass over the norm's input: neither
// the norm's nor the MUL's node is written. The caller skips the norm, lets
// nothing but such MUL_MATs read `mul`, and runs them in this graph.
bool xdna_pgemm_norm_supported(const struct ggml_tensor * mul);
// With `add` (the norm's input is that ADD, xdna_pgemm_add_supported) the
// sum is computed in the same pass and written into add's node, which the
// caller then skips.
bool xdna_pgemm_add_supported(const struct ggml_tensor * add, const struct ggml_tensor * mul);
bool xdna_pgemm_run_norm(struct xdna_kernel_pool * pool, struct ggml_tensor * mul, struct ggml_tensor * add);

// A recurrent layer's output, g = (RMS_NORM(x) * w) * SILU(z) as
// [D, heads, tokens], read through the RESHAPE `v` to [D * heads, tokens] by
// prefill GEMMs: laid out straight into their A in one pass
// (GGML_XDNA_PGEMM_GATED_A=0: off). None of the chain's nodes is written; the
// caller skips the norm, both MULs' first and the SILU, and lets nothing but
// such MUL_MATs read `v`.
bool xdna_pgemm_gated_supported(const struct ggml_tensor * g, const struct ggml_tensor * v);
// With `xr` the norm's input rows are read where the GDN run left them
// (xdna_gdn_mm_rows: D = 128, 16 heads) instead of from its node.
struct xdna_gdn_rows;
bool xdna_pgemm_run_gated(struct xdna_kernel_pool *    pool,
                          const struct ggml_tensor *   g,
                          const struct ggml_tensor *   v,
                          const struct xdna_gdn_rows * xr);

// An attention output times its gate, g = a * SIGMOID(c) (c a CONT of the
// gate's view, or the gate itself), read by prefill GEMMs: laid out straight
// into their A in one pass, the gate read through its view
// (GGML_XDNA_PGEMM_GATE_A=0: off). Neither g, the SIGMOID nor the CONT is
// written; the caller skips the last two.
bool xdna_pgemm_gate_supported(const struct ggml_tensor * g);
// With `ar` the attention output is read where its run left it
// (xdna_attn_mm_rows: 8 heads of 256) instead of from its node.
struct xdna_attn_rows;
bool xdna_pgemm_run_gate(struct xdna_kernel_pool *     pool,
                         const struct ggml_tensor *    g,
                         const struct xdna_attn_rows * ar);

// Two such MUL_MATs of one activation that fit one chunk together (the
// recurrent layers' alpha and beta, 16 columns each): one call, the outputs
// into a_out and b_out, rows of each one's N floats - for the caller to copy
// into the nodes once the graph reaches them. The nodes are not written.
bool xdna_pgemm_pair_supported(const struct ggml_tensor * a, const struct ggml_tensor * b);
bool xdna_pgemm_run_pair(struct xdna_kernel_pool *  pool,
                         const struct ggml_tensor * a,
                         const struct ggml_tensor * b,
                         float *                    a_out,
                         float *                    b_out);
