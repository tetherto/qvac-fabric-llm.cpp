#pragma once

// The data paths of the prefill GEMM's artifact (kernels/pgemm.py, and the
// merged prefill design kernels/pfx.py built on them), shared by the runners
// that build its streams: the GEMM (xdna-pgemm.cpp) and the attention and row
// commands of the merged design (xdna-pfx-attn.cpp, xdna-pfx-row.cpp).

#include "xdna-seq.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

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

// The merged design's header (kernels/pfx.py): the words a core copies out of its A stream's first object
constexpr int          HDR_WORDS     = 16;
constexpr int          HW_ATTN_ON    = 8;   // attention calls
constexpr int          HW_ROW_ON     = 9;   // row command: a row-op core's C objects
constexpr int          HW_A_DRAIN    = 10;  // attention command: the A objects a row-op core lets pass
constexpr int          HW_ATTN_DRAIN = 11;  // row command: the A objects an attention core lets pass
constexpr int          HW_ROW_SETUP  = 12;  // row command: a row-op core's setup objects

// A MemTile access pattern as kernels/pgemm.py writes dims_to_stream and dims_from_stream: (size, stride) a dim,
// outermost first, in elements
struct mt_dim {
    int size;
    int stride;
};

constexpr int MT_DIMS = 4;

// The memory element of the pattern's k-th stream element
inline int mt_walk(const mt_dim (&dims)[MT_DIMS], int k) {
    int at = 0;
    for (int d = MT_DIMS - 1; d >= 0; d--) {
        at += (k % dims[d].size) * dims[d].stride;
        k /= dims[d].size;
    }
    return at;
}

// An A object (64 x 64 bf16) reaches the cores tiled by the A MemTile: the core's element j is the host object's
// element a_tiled_index(j)
constexpr int    A_OBJ_ELEMS            = 64 * 64;
constexpr mt_dim A_TILING[MT_DIMS]      = { { 8, 512 }, { 8, 8 }, { 8, 64 }, { 8, 1 } };
// The C join writes word e of a core's C object (64 x 32 f32) to word mt_walk(C_JOIN, e) of the core's part
constexpr mt_dim C_JOIN[MT_DIMS]        = { { 8, 256 }, { 4, 8 }, { 8, 32 }, { 8, 1 } };

inline int a_tiled_index(int j) {
    return mt_walk(A_TILING, j);
}

// The host object `obj` the cores read as `core`
inline void scatter_a_object(const uint16_t * core, uint16_t * obj) {
    for (int j = 0; j < A_OBJ_ELEMS; j++) {
        obj[a_tiled_index(j)] = core[j];
    }
}

// `n_words` (<= HDR_WORDS) header words into A object `obj`, as the cores read them
inline void write_tiled_words(const int32_t * words, int n_words, uint16_t * obj) {
    uint16_t hb[2 * HDR_WORDS];
    std::memcpy(hb, words, (size_t) n_words * sizeof(int32_t));
    for (int j = 0; j < 2 * n_words; j++) {
        obj[a_tiled_index(j)] = hb[j];
    }
}

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

// Before a MemTile channel's next task: the oldest of the MT_Q in flight done
inline void wait_mt_queue(xdna_seq * seq, uint32_t col, xdna_dma_dir dir, uint32_t ch, int pushed) {
    if (pushed >= MT_Q) {
        xdna_seq_wait_token(seq, col, 1, dir, ch);
    }
}

// The tokens of a channel's last tasks not yet waited: the final `ring` of `pushed`
inline void wait_last(xdna_seq * seq, uint32_t col, uint32_t row, xdna_dma_dir dir, uint32_t ch, int pushed, int ring) {
    for (int k = std::max(0, pushed - ring); k < pushed; k++) {
        xdna_seq_wait_token(seq, col, row, dir, ch);
    }
}

}  // namespace xdna_pgemm_path
