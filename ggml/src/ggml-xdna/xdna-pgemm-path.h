#pragma once

// The data paths of the prefill GEMM's artifact (kernels/pgemm.py, and the
// merged prefill design kernels/pfx.py built on them), shared by the runners
// that build its streams: the GEMM (xdna-pgemm.cpp) and the attention mode of
// the merged design (xdna-pfx-attn.cpp).

#include "xdna-seq.h"

#include <cstddef>
#include <cstdint>

namespace xdna_pgemm_path {

constexpr int          A_COL0      = 6;              // the two A streams' shim columns
// The weights' path, fixed by kernels/pgemm.py
constexpr uint32_t     W_SHIM_CH   = 1;        // shim MM2S channel
constexpr uint32_t     W_MT_S2MM   = 5;        // MemTile S2MM channel
constexpr uint32_t     W_MT_MM2S   = 4;        // MemTile MM2S channel of GEMV tile r: 4 + r
constexpr uint32_t     W_LOCK0     = 48;       // slot s: (empty, full) locks 48 + 2 s, + 1
constexpr uint32_t     WS_ADDR     = 0x24000;  // the slot buffer, bytes into the MemTile
constexpr size_t       WS_BYTES    = (size_t) 33 * 10752;
constexpr uint32_t     MT_CTRL_PKT = 26;       // the MemTiles' controller packet id
// The merged design's packet-switched W path: pair r's tiles under id W_PKT_PAIR0 + r
constexpr uint32_t     W_PKT_PAIR0 = 0;
constexpr uint32_t     MT_BD_PKT   = 1u << 31;  // MemTile BD word 0: the send starts with a packet header
constexpr int          MT_BD_PKT_ID_SHIFT = 23;
// MemTile descriptors the static program leaves free (it uses 0-5, 24-29):
// S2MM 5's ring, MM2S 4's, MM2S 5's
constexpr uint32_t     MT_BD_S2MM  = 40;
constexpr uint32_t     MT_BD_MM2S0 = 16;
constexpr uint32_t     MT_BD_MM2S1 = 44;
// MemTile tasks in flight a channel: its queue holds about five and drops the
// rest without a word
constexpr int          MT_Q        = 4;
// shim descriptors: 0 the A header, 1 the A body (columns 6, 7), 2-3 W, 4.. C
constexpr uint32_t     SH_BD_W     = 2;
constexpr uint32_t     SH_BD_C     = 4;

constexpr const char * PGEMM_STEM = "pgemm_c8";
constexpr const char * PFX_STEM   = "pfx_c8";

// GGML_XDNA_PFX=1 with the merged prefill design built (kernels/pfx.py): its GEMM expands only the 8-bit tile form
// and writes the gated product only as the next projection's A (xdna-pgemm.cpp)
bool pfx_loaded();

// The prefill GEMM's artifact
inline const char * pgemm_stem() {
    return pfx_loaded() ? PFX_STEM : PGEMM_STEM;
}

// The W path routes by packet id in the merged design
inline bool w_path_packet() {
    return pfx_loaded();
}

// A linear transfer of `words`, encoded as IRON writes one: unit strides
// (a zero stride encodes as 0xFFFFF) and the AXI cache bits.
inline xdna_bd linear_bd(uint32_t words) {
    xdna_bd bd;
    bd.buf_len   = words;
    bd.d0_stride = bd.d1_stride = bd.d2_stride = 1;
    bd.ax_cache                                = 2;
    return bd;
}

// MemTile DMA (xdna_seq's descriptors are the shim's): a MemTile's own locks
// and memory as its DMA addresses them, locks 64 on and words 0x20000 on.
inline void mt_set_lock(xdna_seq * seq, uint32_t col, uint32_t id, uint32_t value) {
    xdna_seq_write(seq, col, 1, 0xC0000 + id * 0x10, value);
}

// A linear descriptor of `words` at byte `addr` of the MemTile, taking lock
// `acq` at `acq_val` or more and releasing `rel` by `rel_val`; `pkt` is the
// packet header field of word 0 (0: none)
inline void mt_bd(xdna_seq * seq,
                  uint32_t   col,
                  uint32_t   id,
                  uint32_t   words,
                  uint32_t   addr,
                  uint32_t   acq,
                  uint32_t   acq_val,
                  uint32_t   rel,
                  uint32_t   rel_val,
                  uint32_t   pkt = 0) {
    seq->n_instr++;
    seq->ops.push_back(xdna_txn::OP_BLOCKWRITE);
    seq->ops.push_back(0);
    seq->ops.push_back(xdna_txn::tile_reg(col, 1, 0xA0000 + id * 0x20));
    seq->ops.push_back(xdna_txn::SZ_BLOCKWRITE * 4);
    seq->ops.push_back(words | pkt);
    seq->ops.push_back(0x20000 + addr / 4);
    for (int k = 0; k < 5; k++) {
        seq->ops.push_back(0);
    }
    seq->ops.push_back(1u << 31 | (rel_val & 0x7F) << 24 | (64 + rel) << 16 | 1u << 15 | ((0u - acq_val) & 0x7F) << 8 |
                       (64 + acq));
}

// Start MemTile task `id` on a channel, `repeat` more times, with a token
inline void mt_push(xdna_seq * seq, uint32_t col, uint32_t id, xdna_dma_dir dir, uint32_t ch, uint32_t repeat) {
    const uint32_t ctrl = (dir == xdna_dma_dir::MM2S ? 0xA0630 : 0xA0600) + ch * 8;
    xdna_seq_maskwrite(seq, col, 1, ctrl, MT_CTRL_PKT << 8, 0x1F00);
    xdna_seq_write(seq, col, 1, ctrl + 4, id | repeat << 16 | 1u << 31);
}

}  // namespace xdna_pgemm_path
