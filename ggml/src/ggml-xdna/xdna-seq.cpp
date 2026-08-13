#include "xdna-seq.h"

// TXN instruction encodings (NPU2 / XDNA2). Each op is a fixed-size word
// group; the last word of each group holds op_size * 4 (the total op size in
// bytes), which the firmware uses to advance the instruction pointer.

static constexpr uint32_t BD_D0_SIZE_SHIFT   = 20;
static constexpr uint32_t BD_D1_SIZE_SHIFT   = 20;
static constexpr uint32_t BD_D1_BURST        = 0xC0000000u;
static constexpr uint32_t BD_D2_CACHE_SHIFT  = 24;
static constexpr uint32_t BD_ITER_SIZE_SHIFT = 20;

void xdna_seq_write(xdna_seq * seq, uint32_t col, uint32_t row, uint32_t reg, uint32_t value) {
    seq->n_instr++;
    seq->ops.push_back(xdna_txn::OP_WRITE);
    seq->ops.push_back(0);
    seq->ops.push_back(xdna_txn::tile_reg(col, row, reg));
    seq->ops.push_back(0);
    seq->ops.push_back(value);
    seq->ops.push_back(xdna_txn::SZ_WRITE * 4);
}

void xdna_seq_blockwrite(xdna_seq * seq, uint32_t col, uint32_t row, uint32_t bd_id, const xdna_bd * bd) {
    seq->n_instr++;
    seq->ops.push_back(xdna_txn::OP_BLOCKWRITE);
    seq->ops.push_back(0);
    seq->ops.push_back(xdna_txn::tile_reg(col, row, xdna_txn::SHIM_BD_BASE + bd_id * xdna_txn::SHIM_BD_STRIDE));
    seq->ops.push_back(xdna_txn::SZ_BLOCKWRITE * 4);
    seq->ops.push_back(bd->buf_len);
    seq->ops.push_back(bd->buf_off);
    seq->ops.push_back((bd->packet_enable ? 1u : 0u) << 30 |
                       (bd->out_of_order_id & 0x3F) << 24 |
                       (bd->packet_id & 0x1F) << 19 |
                       (bd->packet_type & 0x7) << 16);
    seq->ops.push_back((bd->d0_size & 0x3FF) << BD_D0_SIZE_SHIFT | ((bd->d0_stride - 1) & 0xFFFFF));
    seq->ops.push_back(BD_D1_BURST | (bd->d1_size & 0x3FF) << BD_D1_SIZE_SHIFT | ((bd->d1_stride - 1) & 0xFFFFF));
    seq->ops.push_back((bd->ax_cache & 0xFF) << BD_D2_CACHE_SHIFT | ((bd->d2_stride - 1) & 0xFFFFF));
    seq->ops.push_back(((bd->iter_size - 1) & 0x3FF) << BD_ITER_SIZE_SHIFT | ((bd->iter_stride - 1) & 0xFFFFF));
    seq->ops.push_back(xdna_txn::bd_ctrl(bd->next_bd, bd->valid, 0, 0, false, 0, 0));
}

void xdna_seq_ddr_patch(xdna_seq * seq, uint32_t col, uint32_t row, uint32_t bd_id,
                        uint32_t arg_idx, uint32_t arg_offset) {
    seq->n_instr++;
    seq->ops.push_back(xdna_txn::OP_DDR_PATCH);
    seq->ops.push_back(xdna_txn::SZ_DDR_PATCH * 4);
    seq->ops.push_back(0);
    seq->ops.push_back(0);
    seq->ops.push_back(0);
    seq->ops.push_back(0);
    seq->ops.push_back(xdna_txn::tile_reg(col, row, xdna_txn::SHIM_BD_ADDR + bd_id * xdna_txn::SHIM_BD_STRIDE));
    seq->ops.push_back(0);
    seq->ops.push_back(arg_idx);
    seq->ops.push_back(0);
    seq->ops.push_back(arg_offset);
    seq->ops.push_back(0);
}

void xdna_seq_maskwrite(xdna_seq * seq, uint32_t col, uint32_t row, uint32_t reg,
                        uint32_t value, uint32_t mask) {
    seq->n_instr++;
    seq->ops.push_back(xdna_txn::OP_MASKWRITE);
    seq->ops.push_back(0);
    seq->ops.push_back(xdna_txn::tile_reg(col, row, reg));
    seq->ops.push_back(0);
    seq->ops.push_back(value);
    seq->ops.push_back(mask);
    seq->ops.push_back(xdna_txn::SZ_MASKWRITE * 4);
}

void xdna_seq_push_queue(xdna_seq * seq, uint32_t col, uint32_t row, uint32_t bd_id, xdna_dma_dir dir,
                         uint32_t channel, bool issue_token, uint32_t repeat) {
    const uint32_t reg = xdna_txn::SHIM_PUSHQ_BASE +
                         (channel > 0 ? xdna_txn::SHIM_CTRL_STRIDE : 0) +
                         (dir == xdna_dma_dir::MM2S ? xdna_txn::SHIM_DIR_STRIDE : 0);
    uint32_t value = (bd_id & xdna_txn::PUSH_BD_ID_MASK) | (repeat << xdna_txn::PUSH_REPEAT_SHIFT);
    if (issue_token) {
        value |= xdna_txn::PUSH_ISSUE_TOKEN;
    }
    xdna_seq_write(seq, col, row, reg, value);
}

void xdna_seq_issue_token(xdna_seq * seq, uint32_t col, uint32_t row, xdna_dma_dir dir,
                          uint32_t channel, uint32_t ctrl_pkt_id) {
    const uint32_t reg = xdna_txn::SHIM_TOKEN_BASE +
                         channel * xdna_txn::SHIM_CTRL_STRIDE +
                         (dir == xdna_dma_dir::MM2S ? xdna_txn::SHIM_DIR_STRIDE : 0);
    xdna_seq_maskwrite(seq, col, row, reg, ctrl_pkt_id << 8, 0x1F00);
}

void xdna_seq_wait_token(xdna_seq * seq, uint32_t col, uint32_t row, xdna_dma_dir dir, uint32_t channel) {
    seq->n_instr++;
    seq->ops.push_back(xdna_txn::OP_TCT);
    seq->ops.push_back(xdna_txn::SZ_TCT * 4);
    seq->ops.push_back((row & 0xFF) << 8 | (col & 0xFF) << 16 | (dir == xdna_dma_dir::MM2S ? 1u : 0u));
    seq->ops.push_back((channel & 0xFF) << 24 | xdna_txn::TCT_W3_CONST);
}

std::vector<uint32_t> xdna_seq_build(const xdna_seq * seq) {
    std::vector<uint32_t> out;
    out.reserve(4 + seq->ops.size());
    out.push_back(xdna_txn::DEV_MAJOR |
                  xdna_txn::DEV_MINOR << xdna_txn::HDR_W0_MINOR_SHIFT |
                  xdna_txn::DEV_GEN   << xdna_txn::HDR_W0_GEN_SHIFT |
                  seq->n_rows         << xdna_txn::HDR_W0_ROWS_SHIFT);
    out.push_back(seq->n_cols | seq->mem_tile_rows << xdna_txn::HDR_W1_MEMTILE_SHIFT);
    out.push_back(seq->n_instr);
    out.push_back((uint32_t)((4 + seq->ops.size()) * sizeof(uint32_t)));
    out.insert(out.end(), seq->ops.begin(), seq->ops.end());
    return out;
}

// --- GEMM sequence -----------------------------------------------------------

bool xdna_gemm_seq_supported(const xdna_gemm_tiles * tiles, int M, int K, int N) {
    if (M <= 0 || M > tiles->M) {
        return false;
    }
    if (K <= 0 || K % tiles->tile_k != 0) {
        return false;
    }
    const int mem_tile_n = tiles->tile_n * tiles->n_cols;
    if (N < mem_tile_n || N % mem_tile_n != 0) {
        return false;
    }
    return true;
}

// Element stride -> 4-byte address-gen words.
static uint32_t to_words(uint32_t elem_stride, uint32_t elem_bytes) {
    return elem_stride * elem_bytes / 4;
}

static void emit_bd(xdna_seq * seq, int col, int bd_id, const xdna_bd & bd,
                    uint32_t arg, uint32_t off) {
    xdna_seq_blockwrite(seq, (uint32_t) col, 0, (uint32_t) bd_id, &bd);
    xdna_seq_ddr_patch(seq, (uint32_t) col, 0, (uint32_t) bd_id, arg, off);
}

// Emits the same instruction stream IRON's seq_fn produces for the GEMM
// kernel. The BD field semantics follow mlir-aie's AIEDmaToNpu.cpp:
//   - d0_size / d1_size in 4-byte address-gen words (elements*elem_bytes/4)
//   - d2 has stride only; its size is inferred as len/(d0_size*d1_size)
//   - buffer_length is in 4-byte words
//   - the outer tap dim folds into the queue push repeat_count (stride 0)
//     or the BD iteration field (stride > 0)
bool xdna_gemm_seq_build(xdna_seq * seq, const xdna_gemm_tiles * t, int M, int K, int N,
                         uint32_t b_offset) {
    const int n_cols = t->n_cols;
    const int n_rows = t->n_compute_rows;
    const int tile_n = t->tile_n;
    const int tile_k = t->tile_k;

    const int mem_tile_n   = tile_n * n_cols;    // 128 or 256
    const int K_div_k      = K / tile_k;         // per-core K loop count
    const int n_col_tiles  = N / mem_tile_n;     // column tiles per core
    // A shims are capped at the number of compute rows (a column can only feed
    // as many row tiles as there are core rows).
    const int n_shim_mem_A = n_cols < n_rows ? n_cols : n_rows;
    const int M_per_shim   = M / n_shim_mem_A;   // 8 rows of A per A shim

    // RTP writes + barriers (rows 2..5, cols 0..7)
    for (int row = 2; row < 2 + n_rows; row++) {
        for (int col = 0; col < n_cols; col++) {
            xdna_seq_write(seq, (uint32_t) col, (uint32_t) row, (uint32_t) (t->rtp_base + 0x000), (uint32_t) K_div_k);
            xdna_seq_write(seq, (uint32_t) col, (uint32_t) row, (uint32_t) (t->rtp_base + 0x004), (uint32_t) n_col_tiles);
            xdna_seq_write(seq, (uint32_t) col, (uint32_t) row, 0x1f000, 1);
        }
    }

    // DMA: per shim column, C output then A/B inputs. BD ids are allocated per
    // shim (in write order), matching IRON: even shims 0,2,4,6 carry the A of
    // logical col/2 plus their own C and B, odd shims only C and B.
    uint32_t shim_bd[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    for (int col = 0; col < n_cols; col++) {
        // C output [M x N] f32. Column c covers tile_n output columns laid
        // out every mem_tile_n elements (interleaved by the tiler). The d2
        // sweep visits the n_col_tiles column blocks; no queue repeat.
        {
            xdna_bd bd;
            bd.buf_len   = (uint32_t)(M * tile_n * n_col_tiles); // words
            bd.buf_off   = (uint32_t) col * tile_n * 4;          // bytes into C
            bd.d0_size   = (uint32_t) tile_n;                    // words (f32)
            bd.d0_stride = 1;
            bd.d1_size   = (uint32_t) M;                         // elements
            bd.d1_stride = to_words((uint32_t) N, 4);
            bd.d2_stride = n_col_tiles > 1 ? to_words((uint32_t) mem_tile_n, 4) : 1;
            bd.ax_cache  = 2;
            const uint32_t bd_id = shim_bd[col]++;
            emit_bd(seq, col, bd_id, bd, 2, (uint32_t) col * tile_n * 4);
            xdna_seq_issue_token(seq, (uint32_t) col, 0, xdna_dma_dir::S2MM, 0, 0xF);
            xdna_seq_push_queue(seq, (uint32_t) col, 0, bd_id, xdna_dma_dir::S2MM, 0, true, 0);
        }

        // A input [M x K] bf16 on A shim cols 0,2,4,6; shim 2*c reads rows
        // [c*8, c*8+8). One BD covers 8 rows x K, repeated n_col_tiles times
        // (stride 0) to re-feed the cores per column-tile pass.
        if (col < n_shim_mem_A) {
            const int a_col = col * 2;
            xdna_bd bd;
            bd.buf_len   = (uint32_t)(M_per_shim * tile_k * K_div_k / 2); // words
            bd.buf_off   = (uint32_t) col * M_per_shim * K * 2;           // bytes into A
            bd.d0_size   = (uint32_t)(tile_k / 2);                        // words
            bd.d0_stride = 1;
            bd.d1_size   = (uint32_t) M_per_shim;                         // elements
            bd.d1_stride = to_words((uint32_t) K, 2);
            bd.d2_stride = to_words((uint32_t) tile_k, 2);
            bd.ax_cache  = 2;
            const uint32_t bd_id = shim_bd[a_col]++;
            emit_bd(seq, a_col, bd_id, bd, 0, (uint32_t) col * M_per_shim * K * 2);
            xdna_seq_push_queue(seq, (uint32_t) a_col, 0, bd_id, xdna_dma_dir::MM2S, 0, false,
                                n_col_tiles > 1 ? (uint32_t) (n_col_tiles - 1) : 0);
        }

        // B input [K x N] bf16. Column c reads tile_n columns interleaved
        // every mem_tile_n elements. d0/d1 = tile_n x tile_k inner tile, d2
        // sweeps K_div_k k-tiles (stride tile_k*N), iteration sweeps the
        // n_col_tiles column blocks. Channel: A occupies MM2S ch0 on the A
        // shims (cols 0,2,4,6), so B uses ch1 there and ch0 elsewhere.
        {
            const int b_ch = (col % 2 == 0) ? 1 : 0;
            xdna_bd bd;
            bd.buf_len   = (uint32_t)(tile_n * tile_k * K_div_k / 2); // words
            bd.buf_off   = (uint32_t) col * tile_n * 2 + b_offset;    // bytes into B
            bd.d0_size   = (uint32_t)(tile_n / 2);                    // words
            bd.d0_stride = 1;
            bd.d1_size   = (uint32_t) tile_k;                         // elements
            bd.d1_stride = to_words((uint32_t) N, 2);
            bd.d2_stride = to_words((uint32_t)(tile_k * N), 2);
            bd.ax_cache  = 2;
            if (n_col_tiles > 1) {
                bd.iter_size   = (uint32_t) n_col_tiles;
                bd.iter_stride = to_words((uint32_t) mem_tile_n, 2);
            }
            const uint32_t bd_id = shim_bd[col]++;
            emit_bd(seq, col, bd_id, bd, 1, (uint32_t) col * tile_n * 2 + b_offset);
            xdna_seq_push_queue(seq, (uint32_t) col, 0, bd_id, xdna_dma_dir::MM2S, (uint32_t) b_ch, false,
                                n_col_tiles > 1 ? (uint32_t) (n_col_tiles - 1) : 0);
        }
    }

    // Wait for all 8 C output tokens
    for (int col = 0; col < n_cols; col++) {
        xdna_seq_wait_token(seq, (uint32_t) col, 0, xdna_dma_dir::S2MM, 0);
    }

    return true;
}
