#pragma once

// A full-attention layer of Qwen3.5's decode as one dispatch of the fused
// layer design, from the normed input to l_out:
//
//   qkv       q|gate, k, v in one pool GEMV, drained to a scratch buffer
//   prepare   the prologue tile (act-att.cc) norms q and k, rotates them with
//             the host's cos/sin, tiles q for the pool and writes this
//             position's K and V rows into llama's cache
//   attention the pool over the cache (attn-dec.cc), sixteen partial states
//             drained to the scratch buffer
//   combine   the prologue tile folds the partials, divides, gates and
//             quantizes attn_output's activation
//   tail      attn_output drained into the FFN's raw tiles, then the FFN
//             pair (the stream xdna_rec_tail runs)
//
// The layer boundary is the prologue's too (its row mode): the input's norm
// from the residual rows and the post-attention residual and norm, so the
// host writes nothing but the rope table.

#include <xrt/xrt_kernel.h>

#include <cstddef>

struct ggml_tensor;
struct xdna_att_layer;
struct xdna_buffer;
struct xdna_kernel_pool;

struct xdna_att_layer_w {
    const ggml_tensor * wq        = nullptr;  // [1024 x 4096]: per head q then gate
    const ggml_tensor * wk        = nullptr;  // [1024 x 512]
    const ggml_tensor * wv        = nullptr;  // [1024 x 512]
    const ggml_tensor * gq        = nullptr;  // attn_q_norm, f32 [256]
    const ggml_tensor * gk        = nullptr;  // attn_k_norm, f32 [256]
    const ggml_tensor * attn_norm = nullptr;  // f32 [1024], the input's norm
    const ggml_tensor * wo        = nullptr;  // [2048 x 1024]
    const ggml_tensor * post      = nullptr;  // post_attention_norm, f32 [1024]
    const ggml_tensor * gate      = nullptr;
    const ggml_tensor * up        = nullptr;
    const ggml_tensor * down      = nullptr;
};

xdna_att_layer * xdna_att_layer_create(struct xdna_kernel_pool * pool, const xdna_att_layer_w & w);
void             xdna_att_layer_free(xdna_att_layer * l);

struct xdna_att_layer_in {
    // The residual rows (xdna-gemv.h XDNA_RES_*): the layer's input is F + A,
    // its outputs h_attn and the FFN's land in A and F, h and the mixer
    // output in H and S.
    xdna_buffer * res      = nullptr;
    // Position 0 of the layer's K and V cache (f16, 1024 B a position).
    xdna_buffer * kbo      = nullptr;
    size_t        koff     = 0;
    xdna_buffer * vbo      = nullptr;
    size_t        voff     = 0;
    int           n_valid  = 0;        // positions, this one (row n_valid - 1) included
    int           n_rot    = 0;        // rotated dims, a multiple of 32, <= 256
    const float * cosv     = nullptr;  // n_rot / 2 each, for this position
    const float * sinv     = nullptr;
    float         scale    = 0.0f;     // the attention's 1/sqrt(d)
    float         eps      = 0.0f;     // the q/k norms' epsilon
    float         eps_attn = 0.0f;     // the input norm's
    float         eps_post = 0.0f;     // the post-attention norm's
};

// Positions one dispatch can take.
int xdna_att_layer_max_positions(void);

// Whether the chunks of `n_valid` positions lie inside the cache buffers: the
// last chunk is read whole, so near the end of a cache it runs past it.
bool xdna_att_layer_fits(const xdna_buffer * kbo, size_t koff, const xdna_buffer * vbo, size_t voff, int n_valid);

// Run the layer: A = h + attn_output(attention(rms_norm(h))), F = FFN(A),
// with h = F + A on entry, in `in.res`. False when the dispatch cannot be
// built or run.
bool xdna_att_layer_run(xdna_att_layer * l, const xdna_att_layer_in & in);

// The same in two halves: start it, and after the returned run has been
// waited for, finish it (the host must not keep an older copy of the cache
// row it wrote). Null when it cannot be started.
xrt::run *         xdna_att_layer_start(xdna_att_layer * l, const xdna_att_layer_in & in);
[[nodiscard]] bool xdna_att_layer_finish(const xdna_att_layer_in & in);
