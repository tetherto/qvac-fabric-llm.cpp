#pragma once

// R1 of a sliding attention layer as the row command of the merged prefill design (kernels/pfx.py): the Q and K
// heads' RMS norm times q_norm / k_norm, then neox RoPE, and V's RMS norm, on the two row-op columns (kernels/rowop.cc).
// Column 3's core i takes Q head i, column 7's cores the two K heads and then the two V heads; each takes a block of
// 8 positions of its head a step on the W path under its own packet id, and the block's cos and sin on pgemm's A
// broadcast, which the attention columns' cores let pass. The bf16 result leaves through pgemm's C join, two blocks a
// core's C object, straight into the pfx attention's own W, K and V^T objects (`dst`, xdna-pfx-layout.h), the cores
// placing each word where the join and the shim's descriptors put it in that layout (xdna_pfx_row_stream_word,
// xdna_pfx_row_dst_byte).

#include "xdna-pfx-layout.h"

#include <cstddef>
#include <cstdint>
#include <vector>

struct xdna_buffer;
struct xdna_kernel_pool;

// ggml's ROPE parameters (ggml_rope_ext)
struct xdna_pfx_row_rope {
    int   n_dims      = 0;  // rotated dims a head, 0: none
    int   n_ctx_orig  = 0;
    float freq_base   = 10000.0f;
    float freq_scale  = 1.0f;
    float ext_factor  = 0.0f;
    float attn_factor = 1.0f;
    float beta_fast   = 32.0f;
    float beta_slow   = 1.0f;
};

// One sliding layer: Q [n][4 x 256], K and V [n][2 x 256] f32, row-major at byte offsets of `src` as the array sees
// it, each readable to xdna_pfx_row_job_blocks x 8 rows (zero past n_tokens). The command runs whole 32-key tiles of
// dst (pp 16 and n_tokens keys) and writes Q scaled by q_scale.
struct xdna_pfx_row_job {
    xdna_buffer *             src      = nullptr;
    size_t                    q_off    = 0;
    size_t                    k_off    = 0;
    size_t                    v_off    = 0;
    int64_t                   n_tokens = 0;
    const int32_t *           pos      = nullptr;  // n_tokens positions
    xdna_pfx_row_rope         rope;
    float                     eps     = 0.0f;
    const float *             q_norm  = nullptr;  // [256]
    const float *             k_norm  = nullptr;  // [256]
    float                     q_scale = 1.0f;  // the Q cores' scale: the attention's log2(e) x q_scale
    const xdna_pfx_attn_dst * dst     = nullptr;
};

// kernels/rowop.cc's output layouts (setup word P_LAYOUT): the C objects, or the attention's objects
enum xdna_pfx_row_layout : int32_t {
    XDNA_PFX_ROW_LAYOUT_C  = 0,
    XDNA_PFX_ROW_LAYOUT_Q  = 1,
    XDNA_PFX_ROW_LAYOUT_K  = 2,
    XDNA_PFX_ROW_LAYOUT_VT = 3,
};

// Blocks of 8 positions a core takes for the job, two to each C object
int64_t xdna_pfx_row_job_blocks(const xdna_pfx_row_job & job);
// The word of a core's part of the C join's stream, and its bf16 half, that holds dim d of row j of block bp (0, 1)
// of the C object, as the cores write `layout` (Q, K or V^T)
int     xdna_pfx_row_stream_word(int layout, int bp, int j, int d, int * half);
// The byte of dst.bo that word s of core i's part of C object o reaches on row-op column ci
size_t  xdna_pfx_row_dst_byte(const xdna_pfx_attn_dst & dst, int ci, int64_t o, int i, int s);

// The command's inputs as the host lays them out: argument 1 (the header object and a cos / sin object a block) and
// argument 3 (a setup object a core of each row-op column); and its instruction stream
size_t                xdna_pfx_row_a_bytes(const xdna_pfx_row_job & job);
size_t                xdna_pfx_row_setup_bytes(void);
void                  xdna_pfx_row_pack_a(const xdna_pfx_row_job & job, uint16_t * a);
void                  xdna_pfx_row_pack_setup(const xdna_pfx_row_job & job, uint8_t * setup);
std::vector<uint32_t> xdna_pfx_row_stream(const xdna_pfx_row_job & job);

// Argument 1 for the layers of one graph that share positions, token count and RoPE parameters: packed by the first
// run that takes it, kept for the next, given back to the pool with the handle
struct xdna_pfx_row_tables {
    xdna_kernel_pool * pool     = nullptr;
    xdna_buffer *      a        = nullptr;
    const int32_t *    pos      = nullptr;  // what a holds: null, nothing
    int64_t            n_tokens = 0;
    int64_t            n_blocks = 0;
    xdna_pfx_row_rope  rope;

    xdna_pfx_row_tables() = default;
    xdna_pfx_row_tables(const xdna_pfx_row_tables &)             = delete;
    xdna_pfx_row_tables & operator=(const xdna_pfx_row_tables &) = delete;
    ~xdna_pfx_row_tables();
};

// Run it to completion: dst's objects written on the device
bool xdna_pfx_row_run(xdna_kernel_pool * pool, const xdna_pfx_row_job & job);
// The same with argument 1 from tables, packed there first unless they hold job's
bool xdna_pfx_row_run_tables(xdna_kernel_pool * pool, const xdna_pfx_row_job & job, xdna_pfx_row_tables & tables);
