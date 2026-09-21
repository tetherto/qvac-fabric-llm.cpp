#include "xdna-seq.h"

#include "ggml-impl.h"

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
    if (row == 0 && col < XDNA_SEQ_MAX_COLS && bd_id + 1 > seq->bd_used[col]) {
        seq->bd_used[col] = bd_id + 1;
    }
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

// The firmware translates a buffer address itself only for the first five
// arguments. From argument five on, the patch has to carry the AIE DDR
// aperture base folded into its offset - mlir-aie says so where it explains
// why its own hrx path does not need to ("the XRT/instruction-buffer path
// instead folds the aperture offset into arg_plus for args >= 5 to match its
// firmware"), and IRON's compiled streams show exactly that: plain offsets on
// arguments 0-4, 0x80000000 | offset on 5 and above.
//
// Getting this wrong does not look like an addressing bug. The transfer is
// issued against an untranslated address, the data lands nowhere the host can
// see, and the completion token never arrives - so it reads as a stream that
// hangs for naming a high argument, or as a drain that only works at offset
// zero (0x80000000 alone is the aperture base with a zero offset, which is
// the one case the old encoding got right by accident).
static constexpr uint32_t XDNA_DDR_APERTURE   = 0x80000000u;
static constexpr uint32_t XDNA_FW_XLAT_ARGS   = 5;

void xdna_seq_ddr_patch(xdna_seq * seq, uint32_t col, uint32_t row, uint32_t bd_id,
                        uint32_t arg_idx, uint32_t arg_offset) {
    if (arg_idx >= XDNA_FW_XLAT_ARGS) {
        arg_offset |= XDNA_DDR_APERTURE;
    }
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
                         channel * xdna_txn::SHIM_CTRL_STRIDE +
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
    if (!tiles || tiles->n_cols <= 0 || (uint32_t) tiles->n_cols > XDNA_SEQ_MAX_COLS) {
        return false;
    }
    if (tiles->tile_m <= 0 || tiles->tile_k <= 0 || tiles->tile_n <= 0 ||
        tiles->n_compute_rows <= 0 || tiles->elem_bytes <= 0 || tiles->rtp_base <= 0) {
        return false;
    }
    // The stream is built for whole sweeps of the baked block; a partial block
    // would emit no DMA at all (see xdna_gemm_seq_build).
    const int sweep_m = tiles->tile_m * tiles->n_compute_rows;
    if (M <= 0 || M > tiles->M || M % sweep_m != 0) {
        return false;
    }
    if (K <= 0 || K % tiles->tile_k != 0) {
        return false;
    }
    const int mem_tile_n = tiles->tile_n * tiles->n_cols;
    if (N < mem_tile_n || N % mem_tile_n != 0) {
        return false;
    }
    // Sizes and strides go to the address generator in 4-byte words, so a
    // dimension that is not a whole word would be truncated into a wrong
    // transfer rather than refused.
    if ((tiles->tile_k * tiles->elem_bytes) % 4 || (tiles->tile_n * tiles->elem_bytes) % 4) {
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
    if (!seq || !xdna_gemm_seq_supported(t, M, K, N)) {
        GGML_LOG_ERROR("%s: gemm: unsupported geometry M%d K%d N%d "
                       "(block M%d tile %dx%dx%d cols %d)\n", "xdna-seq", M, K, N,
                       t ? t->M : 0, t ? t->tile_m : 0, t ? t->tile_k : 0,
                       t ? t->tile_n : 0, t ? t->n_cols : 0);
        return false;
    }
    const int n_cols = t->n_cols;
    const int n_rows = t->n_compute_rows;
    const int tile_n = t->tile_n;
    const int tile_k = t->tile_k;
    const int eb     = t->elem_bytes;   // A/B element size (2 bf16, 1 int8)

    const int mem_tile_n   = tile_n * n_cols;    // 512 (prefill) column tile
    const int K_div_k      = K / tile_k;         // per-core K loop count
    const int n_col_tiles  = N / mem_tile_n;     // column tiles per core
    const int n_shim_mem_A = n_cols < n_rows ? n_cols : n_rows;
    // One row sweep covers tile_m rows per compute row: C rows advance by
    // mem_tile_m_C per sweep, A by mem_tile_m_A per (shim, sweep). Kernels
    // baked with more than one sweep run the row dimension as repeated
    // 64-row groups, each fed by its own C/A/B BDs.
    const int tile_m       = t->tile_m;
    const int mem_tile_m_A = tile_m;                       // A rows per shim per sweep
    const int mem_tile_m_C = tile_m * n_rows;              // C rows per sweep
    const int n_sweeps     = M / mem_tile_m_C;
    GGML_ASSERT(n_sweeps > 0);   // xdna_gemm_seq_supported holds M to a whole sweep

    // RTP writes + barriers (rows 2..5, cols 0..7): per-core loop counts, the
    // core's n_tiles_per_core = row sweeps x column tiles.
    for (int row = 2; row < 2 + n_rows; row++) {
        for (int col = 0; col < n_cols; col++) {
            xdna_seq_write(seq, (uint32_t) col, (uint32_t) row, (uint32_t) (t->rtp_base + 0x000), (uint32_t) K_div_k);
            xdna_seq_write(seq, (uint32_t) col, (uint32_t) row, (uint32_t) (t->rtp_base + 0x004), (uint32_t) (n_sweeps * n_col_tiles));
            xdna_seq_write(seq, (uint32_t) col, (uint32_t) row, 0x1f060, 1);
        }
    }

    // DMA: per shim column, C output then A/B inputs. BD ids are allocated per
    // shim (in write order), matching IRON: even shims 0,2,4,6 carry the A of
    // logical col/2 plus their own C and B, odd shims only C and B. Multi-sweep
    // blocks reuse the per-sweep BD ids after the previous sweep's C tokens are
    // waited (the C completion implies A/B were consumed).
    for (int sweep = 0; sweep < n_sweeps; sweep++) {
        const size_t c_row_off = (size_t) sweep * mem_tile_m_C;    // rows into C
        const size_t c_byte    = c_row_off * N * 4;                // C is f32/int32
        uint32_t shim_bd[XDNA_SEQ_MAX_COLS] = {};
        for (int col = 0; col < n_cols; col++) {
            // C output [M x N] f32/int32. Column c covers tile_n output columns
            // laid out every mem_tile_n elements (interleaved by the tiler).
            // The d2 sweep visits the n_col_tiles column blocks.
            {
                xdna_bd bd;
                bd.buf_len   = (uint32_t)(mem_tile_m_C * tile_n * n_col_tiles); // words
                bd.buf_off   = (uint32_t)(col * tile_n * 4 + c_byte);           // bytes into C
                bd.d0_size   = (uint32_t) tile_n;                               // words
                bd.d0_stride = 1;
                bd.d1_size   = (uint32_t) mem_tile_m_C;                         // elements (rows)
                bd.d1_stride = to_words((uint32_t) N, 4);
                bd.d2_stride = n_col_tiles > 1 ? to_words((uint32_t) mem_tile_n, 4) : 1;
                bd.ax_cache  = 2;
                const uint32_t bd_id = shim_bd[col]++;
                emit_bd(seq, col, bd_id, bd, 2, (uint32_t) (col * tile_n * 4 + c_byte));
                xdna_seq_issue_token(seq, (uint32_t) col, 0, xdna_dma_dir::S2MM, 0, 0xF);
                xdna_seq_push_queue(seq, (uint32_t) col, 0, bd_id, xdna_dma_dir::S2MM, 0, true, 0);
            }

            // A input [M x K] (bf16 or int8) on A shim cols 0,2,4,6; shim
            // col feeds compute row col. Each sweep streams the compute row's
            // tile_m rows (rows col*tile_m of the sweep), repeated per column
            // tile so the cores are re-fed for each column tile.
            if (col < n_shim_mem_A) {
                const int a_col = col * 2;
                const size_t a_row = c_row_off + (size_t) col * mem_tile_m_A;  // rows into A
                const size_t a_byte = a_row * K * eb;
                xdna_bd bd;
                bd.buf_len   = (uint32_t)(mem_tile_m_A * K_div_k * tile_k * eb / 4); // words
                bd.buf_off   = (uint32_t) a_byte;
                bd.d0_size   = (uint32_t)(tile_k * eb / 4);                          // words
                bd.d0_stride = 1;
                bd.d1_size   = (uint32_t) mem_tile_m_A;                              // elements (rows)
                bd.d1_stride = to_words((uint32_t) K, eb);
                bd.d2_stride = to_words((uint32_t) tile_k, eb);
                bd.ax_cache  = 2;
                const uint32_t bd_id = shim_bd[a_col]++;
                emit_bd(seq, a_col, bd_id, bd, 0, (uint32_t) a_byte);
                xdna_seq_push_queue(seq, (uint32_t) a_col, 0, bd_id, xdna_dma_dir::MM2S, 0, false,
                                    n_col_tiles > 1 ? (uint32_t) (n_col_tiles - 1) : 0);
            }

            // B input [K x N] (bf16 or int8). Column c reads tile_n columns
            // interleaved every mem_tile_n elements. Re-fed per sweep.
            {
                const int b_ch = (col % 2 == 0) ? 1 : 0;
                xdna_bd bd;
                bd.buf_len   = (uint32_t)(tile_n * tile_k * K_div_k * eb / 4); // words
                bd.buf_off   = (uint32_t) col * tile_n * eb + b_offset;        // bytes into B
                bd.d0_size   = (uint32_t)(tile_n * eb / 4);                    // words
                bd.d0_stride = 1;
                bd.d1_size   = (uint32_t) tile_k;                              // elements
                bd.d1_stride = to_words((uint32_t) N, eb);
                bd.d2_stride = to_words((uint32_t)(tile_k * N), eb);
                bd.ax_cache  = 2;
                if (n_col_tiles > 1) {
                    bd.iter_size   = (uint32_t) n_col_tiles;
                    bd.iter_stride = to_words((uint32_t) mem_tile_n, eb);
                }
                const uint32_t bd_id = shim_bd[col]++;
                emit_bd(seq, col, bd_id, bd, 1, (uint32_t) col * tile_n * eb + b_offset);
                xdna_seq_push_queue(seq, (uint32_t) col, 0, bd_id, xdna_dma_dir::MM2S, (uint32_t) b_ch, false,
                                    n_col_tiles > 1 ? (uint32_t) (n_col_tiles - 1) : 0);
            }
        }

        // Wait for all 8 C output tokens so the next sweep can reuse the BDs.
        for (int col = 0; col < n_cols; col++) {
            xdna_seq_wait_token(seq, (uint32_t) col, 0, xdna_dma_dir::S2MM, 0);
        }
    }

    return true;
}

// --- GDN prefill sequence ----------------------------------------------------

// One 1D bulk BD on `bd`'s shim BD slot for column `col`: buf_len words from
// host buffer `arg` at byte `byte_off` (the descriptor and the DDR patch use
// the same offset, matching the GEMM emit and the compiled stream).
static void emit_gdn_1d(xdna_seq * seq, uint32_t col, uint32_t bd,
                        uint32_t words, uint32_t byte_off, uint32_t arg) {
    xdna_bd bd_;
    bd_.buf_len   = words;
    bd_.buf_off   = byte_off;
    bd_.d0_size   = 0;              // single-dimension burst transfer
    bd_.d0_stride = 1;
    bd_.d1_size   = 0;
    bd_.d1_stride = 1;
    bd_.d2_stride = 1;
    bd_.ax_cache  = 2;
    bd_.iter_size   = 1;
    bd_.iter_stride = 1;
    xdna_seq_blockwrite(seq, col, 0, bd, &bd_);
    xdna_seq_ddr_patch(seq, col, 0, bd, arg, byte_off);
}

bool xdna_gdn_prefill_seq_build(xdna_seq * seq, const xdna_gdn_prefill_geom * g) {
    if (!seq || !g || g->n_cols <= 0 || g->n_cols > 8) {
        return false;
    }
    const int n_cols = g->n_cols;

    // State seed: one column's packed region per shim (BD0, MM2S ch0, arg 1).
    for (int c = 0; c < n_cols; c++) {
        emit_gdn_1d(seq, (uint32_t) c, 0, (uint32_t) g->state_col_words,
                    (uint32_t) c * g->state_col_bytes, 1);
        xdna_seq_push_queue(seq, (uint32_t) c, 0, 0, xdna_dma_dir::MM2S, 0, false, 0);
    }

    // Per column: tok fill (BD1, MM2S ch1, arg 0) then the packed drain
    // (BD2, S2MM ch0, arg 2). The last column's drain stamps the completion
    // token the trailing wait_token() syncs on.
    for (int c = 0; c < n_cols; c++) {
        emit_gdn_1d(seq, (uint32_t) c, 1, (uint32_t) g->tok_col_words,
                    (uint32_t) c * g->tok_col_bytes, 0);
        xdna_seq_push_queue(seq, (uint32_t) c, 0, 1, xdna_dma_dir::MM2S, 1, false, 0);

        emit_gdn_1d(seq, (uint32_t) c, 2, (uint32_t) g->state_col_words,
                    (uint32_t) c * g->state_col_bytes, 2);
        if (c < n_cols - 1) {
            xdna_seq_push_queue(seq, (uint32_t) c, 0, 2, xdna_dma_dir::S2MM, 0, false, 0);
        } else {
            xdna_seq_issue_token(seq, (uint32_t) c, 0, xdna_dma_dir::S2MM, 0, 0xF);
            xdna_seq_push_queue(seq, (uint32_t) c, 0, 2, xdna_dma_dir::S2MM, 0, true, 0);
        }
    }
    xdna_seq_wait_token(seq, (uint32_t) (n_cols - 1), 0, xdna_dma_dir::S2MM, 0);
    return true;
}
