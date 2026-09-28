#pragma once

#include <stddef.h>

// The prefill gated delta rule on the mmul (kernels/gdn_mm.py):
// GGML_OP_GATED_DELTA_NET of Qwen3.5's recurrent layers - 16 heads, S = 128, a
// scalar gate, one sequence, one state snapshot. The state stays on the array
// for the whole ubatch.
// GGML_XDNA_GDN_MM=0 leaves it to the host.

struct ggml_tensor;
struct xdna_kernel_pool;

bool xdna_gdn_mm_supported(const struct ggml_tensor * node);

// Run it to completion and write the output and the new state.
bool xdna_gdn_mm_run(struct xdna_kernel_pool * pool, struct ggml_tensor * node);

// The attention output left where the array wrote it, for a reader that
// takes its rows from there (xdna_pgemm_run_gated): planned per graph with
// xdna_gdn_mm_keep, the run of such a node writes only the new state into
// its node. Element (t, h, d) of the output is at
//   base + (h / 2) * o_col + ((t / 16) * 4 + 2 * (h % 2) + d / 64) * 1024 + (t % 16) * 64 + d % 64
// until the next run; xdna_gdn_mm_materialize writes it into the node (for a
// reader that falls back to the host).
struct xdna_gdn_rows {
    const float * base  = nullptr;
    size_t        o_col = 0;  // floats a column
    int           n_tok = 0;
};

void xdna_gdn_mm_keep_clear(void);
void xdna_gdn_mm_keep(const struct ggml_tensor * node);
bool xdna_gdn_mm_rows(const struct ggml_tensor * node, struct xdna_gdn_rows * rows);
bool xdna_gdn_mm_materialize(const struct ggml_tensor * node);

// The recurrent layer's conv input as llama builds it - concat(conv state,
// qkv^T), SSM_CONV, SILU, views of Q, K and V, L2_NORM of Q and K and a CPY
// of the new state - done by xdna_gdn_mm_prepare in one pass straight into
// the next xdna_gdn_mm_run's input layout, while the projection is still
// live (at the concat). The run of `node` then lays out only the headers,
// the state and the exponents.
struct xdna_gdn_conv_in {
    const struct ggml_tensor * x         = nullptr;              // qkv^T, [T, channels], channels contiguous
    const struct ggml_tensor * state     = nullptr;              // [KW - 1, channels]
    const struct ggml_tensor * w         = nullptr;              // [KW, channels]
    struct ggml_tensor *       state_out = nullptr;              // the CPY: the new state, contiguous
    float                      eps_q = 0.0f, eps_k = 0.0f;
    long long                  q_off = 0, k_off = 0, v_off = 0;  // channels
};

bool xdna_gdn_mm_prepare(struct xdna_kernel_pool *       pool,
                         const struct ggml_tensor *      node,
                         const struct xdna_gdn_conv_in & in);

// The same on the array (kernels/gdn_conv.py; GGML_XDNA_GDN_CONV=0: off):
// the in-projection is run into xdna_gdn_conv_input's buffer instead of its
// node (xdna_pgemm_run_into at `*off`), and xdna_gdn_mm_prepare_npu then
// convolves it there into the next run's input; the host writes only the
// state's rows before the tokens and the new state.
struct xdna_buffer;
bool                 xdna_gdn_conv_supported(void);
struct xdna_buffer * xdna_gdn_conv_input(struct xdna_kernel_pool * pool, int n_tok, int channels, size_t * off);
bool                 xdna_gdn_mm_prepare_npu(struct xdna_kernel_pool *       pool,
                                             const struct ggml_tensor *      node,
                                             const struct xdna_gdn_conv_in & in);
