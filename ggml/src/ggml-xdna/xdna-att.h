#pragma once

// Decode attention on the fused layer's GEMV pool (kernels/attn-dec.cc): one
// query token of Qwen3.5's full-attention layers - eight query heads over two
// kv heads of 256 - against llama's f16 KV cache, read where it lies in the
// backend's host BOs. The sixteen pool cores each take every sixteenth chunk
// of five positions; their partial softmax states are combined here.

#include "xdna-gemv.h"
#include "xdna-seq.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

struct xdna_att;
struct xdna_buffer;
struct xdna_device;
struct xdna_kernel_pool;

// The decode attention's geometry of the cache (kernels/attn-dec.cc), shared
// by this dispatch and the full layer's.
namespace xdna_att_geom {

static constexpr int D        = 256;
static constexpr int H        = 8;
static constexpr int KVH      = 2;
static constexpr int P        = 5;  // positions a chunk
static constexpr int CORES    = 16;
static constexpr int PIECES   = 33;
static constexpr int PIECE    = 64;              // floats
static constexpr int ST       = PIECES * PIECE;  // a core's partial state, floats
static constexpr int POS_B    = KVH * D * 2;     // one position of K (or V), f16
static constexpr int ROUND    = CORES * P;
static constexpr int TASK     = 64;              // chunks one BD chain covers
static constexpr int MAX_TASK = 16;
static constexpr int ACTT     = XDNA_GEMV_ACT_TILE;

// The dispatch's argument slots the firmware translates itself; the KV cache
// can sit anywhere in a large BO.
enum { ARG_K = 0, ARG_ACT = 1, ARG_IDS = 2, ARG_V = 4 };

inline int max_positions() {
    return MAX_TASK * TASK * ROUND;
}

inline int chunk_count(int n_valid) {
    return (n_valid + ROUND - 1) / ROUND;
}

inline int task_count(int n_chunks) {
    return (n_chunks + TASK - 1) / TASK;
}

// The chunks of `n_valid` positions have to lie inside the cache buffers: the
// last chunk is read whole, so near the end of a cache it runs past it.
inline bool fits(size_t kbytes, size_t koff, size_t vbytes, size_t voff, int n_valid) {
    const size_t span = (size_t) chunk_count(n_valid) * ROUND * POS_B;
    return koff + span <= kbytes && voff + span <= vbytes && koff + span <= UINT32_MAX && voff + span <= UINT32_MAX;
}

inline void lin(xdna_bd & bd, size_t bytes, uint32_t off) {
    bd.buf_len   = (uint32_t) (bytes / 4);
    bd.buf_off   = off;
    bd.d0_size   = 0;
    bd.d0_stride = 1;
    bd.d1_size   = 0;
    bd.d1_stride = 1;
    bd.d2_stride = 1;
    bd.ax_cache  = 2;
}

// The pool's read of the cache, one BD chain a column over `n_chunks` chunks.
inline void emit_chunks(xdna_seq &           seq,
                        const xdna_gemv_ep * w,
                        size_t               tb,
                        int                  n_task,
                        int                  n_chunks,
                        const uint32_t *     chain,
                        size_t               koff,
                        size_t               voff,
                        int                  argv) {
    const size_t pad = tb - (size_t) 2 * P * POS_B;
    for (int t = 0; t < n_task; t++) {
        const int cnt = std::min(TASK, n_chunks - t * TASK);
        for (int c = 0; c < XDNA_GEMV_COLS; c++) {
            const xdna_gemv_ep & e = w[c];
            if (t > 0) {
                xdna_seq_wait_token(&seq, e.col, 0, xdna_dma_dir::MM2S, e.ch);
            }
            // [K lead | V lead | pad | K partner | V partner | pad]: the core's
            // j-th chunk is the cache's chunk 16 j + its index
            for (int r = 0; r < 2; r++) {
                const size_t   pos = (size_t) t * TASK * ROUND + (size_t) (2 * c + r) * P;
                const uint32_t b0  = chain[c] + 3 * r;
                xdna_bd        bk, bv, bp;
                lin(bk, (size_t) P * POS_B, (uint32_t) (koff + pos * POS_B));
                lin(bv, (size_t) P * POS_B, (uint32_t) (voff + pos * POS_B));
                lin(bp, pad, 0);
                bk.iter_size = bv.iter_size = TASK;
                bk.iter_stride = bv.iter_stride = (uint32_t) ((size_t) ROUND * POS_B / 4);
                bk.next_bd                      = b0 + 1;
                bk.use_next                     = true;
                bv.next_bd                      = b0 + 2;
                bv.use_next                     = true;
                if (r == 0) {
                    bp.next_bd  = b0 + 3;
                    bp.use_next = true;
                }
                xdna_seq_blockwrite(&seq, e.col, 0, b0, &bk);
                xdna_seq_ddr_patch(&seq, e.col, 0, b0, ARG_K, bk.buf_off);
                xdna_seq_blockwrite(&seq, e.col, 0, b0 + 1, &bv);
                xdna_seq_ddr_patch(&seq, e.col, 0, b0 + 1, argv, bv.buf_off);
                xdna_seq_blockwrite(&seq, e.col, 0, b0 + 2, &bp);
                xdna_seq_ddr_patch(&seq, e.col, 0, b0 + 2, ARG_IDS, 0);
            }
            xdna_seq_issue_token(&seq, e.col, 0, xdna_dma_dir::MM2S, e.ch, 0xF);
            xdna_seq_push_queue(&seq, e.col, 0, chain[c], xdna_dma_dir::MM2S, e.ch, true, (uint32_t) (cnt - 1));
        }
    }
    for (int c = 0; c < XDNA_GEMV_COLS; c++) {
        xdna_seq_wait_token(&seq, w[c].col, 0, xdna_dma_dir::MM2S, w[c].ch);
    }
}

}  // namespace xdna_att_geom

xdna_att * xdna_att_create(struct xdna_kernel_pool * pool, struct xdna_device * dev);
void       xdna_att_free(xdna_att * a);

// The cache positions one dispatch can take.
int xdna_att_max_positions(void);

// out[h * 256 + d] = softmax(scale * q_h . K) V for the first n_valid cache
// positions. `k` / `v` are the BO and byte offset of position 0 of the layer's
// K and V (position stride 1024 B, kv-head stride 512 B, f16). The caller has
// synced the rows the host wrote. False when the dispatch cannot be built or
// run - the caller falls back.
bool xdna_att_run(xdna_att *           a,
                  struct xdna_buffer * kbo,
                  size_t               koff,
                  struct xdna_buffer * vbo,
                  size_t               voff,
                  const float *        q,
                  float                scale,
                  int                  n_valid,
                  float *              out);
