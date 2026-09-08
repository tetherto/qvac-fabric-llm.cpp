#pragma once

// Instruction stream (TXN) builders for the AMD XDNA NPU, kept out of the
// backend scaffold. A TXN stream is a flat little-endian uint32 array: a
// 4-word header followed by fixed-size instructions. The stream is bound to a
// loaded xclbin as its insts BO. Only GEMM sequences exist so far; other ops
// extend this module with their own builders.

#include <cstdint>
#include <vector>

// TXN format constants (hardware ABI, NPU2/XDNA2, gen=4, 6 rows x 8 cols).
namespace xdna_txn {

static constexpr uint32_t DEV_MAJOR = 0;
static constexpr uint32_t DEV_MINOR = 1;
static constexpr uint32_t DEV_GEN   = 4;

enum {
    HDR_W0_ROWS_SHIFT     = 24,
    HDR_W0_GEN_SHIFT      = 16,
    HDR_W0_MINOR_SHIFT    = 8,
    HDR_W1_MEMTILE_SHIFT  = 8,
};

enum opcode : uint32_t {
    OP_WRITE       = 0x00,   // 6 words - register write / DMA push-queue
    OP_BLOCKWRITE  = 0x01,   // 12 words - shim DMA buffer descriptor
    OP_MASKWRITE   = 0x03,   // 7 words - masked register write / issue token
    OP_TCT         = 0x80,   // 4 words - wait for task-complete token
    OP_DDR_PATCH   = 0x81,   // 12 words - patch a BD address with a buffer's device address
};

static constexpr uint32_t SZ_WRITE      = 6;
static constexpr uint32_t SZ_BLOCKWRITE = 12;
static constexpr uint32_t SZ_MASKWRITE  = 7;
static constexpr uint32_t SZ_TCT        = 4;
static constexpr uint32_t SZ_DDR_PATCH  = 12;

static constexpr uint32_t SHIM_BD_BASE       = 0x1D000;  // BD length register
static constexpr uint32_t SHIM_BD_STRIDE     = 0x20;     // per-BD register stride
static constexpr uint32_t SHIM_BD_ADDR       = 0x1D004;  // BD address register (patched)
static constexpr uint32_t SHIM_PUSHQ_BASE    = 0x1D204;  // push-queue for S2MM ch0
static constexpr uint32_t SHIM_TOKEN_BASE    = 0x1D200;  // token-issue register for S2MM ch0
static constexpr uint32_t SHIM_CTRL_STRIDE   = 0x08;     // per-channel stride
static constexpr uint32_t SHIM_DIR_STRIDE    = 0x10;     // per-direction stride (MM2S)

static constexpr uint32_t BD_NEXT_SHIFT      = 27;
static constexpr uint32_t BD_VALID_SHIFT     = 25;
static constexpr uint32_t BD_LOCK_REL_VAL_SH = 18;
static constexpr uint32_t BD_LOCK_REL_ID_SH  = 13;
static constexpr uint32_t BD_LOCK_ACQ_EN_SH  = 12;
static constexpr uint32_t BD_LOCK_ACQ_VAL_SH = 5;
static constexpr uint32_t BD_LOCK_ACQ_ID_SH  = 0;

static constexpr uint32_t PUSH_BD_ID_MASK    = 0x0F;
static constexpr uint32_t PUSH_REPEAT_SHIFT  = 16;
static constexpr uint32_t PUSH_ISSUE_TOKEN   = 0x80000000u;

static constexpr uint32_t TCT_W3_CONST       = 0x00010100;

// Encode the 20-bit register offset + tile position into the address word.
inline uint32_t tile_reg(uint32_t col, uint32_t row, uint32_t reg) {
    return ((col & 0x7F) << 25) | ((row & 0x1F) << 20) | (reg & 0xFFFFF);
}

// Encode the BLOCKWRITE "next" control word.
inline uint32_t bd_ctrl(uint32_t next_bd, bool valid, uint32_t lock_rel_val,
                        uint32_t lock_rel_id, bool lock_acq_en, uint32_t lock_acq_val,
                        uint32_t lock_acq_id) {
    return ((next_bd & 0x1F) << BD_NEXT_SHIFT) |
           (valid << BD_VALID_SHIFT) |
           ((lock_rel_val & 0x3F) << BD_LOCK_REL_VAL_SH) |
           ((lock_rel_id & 0x1F) << BD_LOCK_REL_ID_SH) |
           ((lock_acq_en ? 1u : 0u) << BD_LOCK_ACQ_EN_SH) |
           ((lock_acq_val & 0x3F) << BD_LOCK_ACQ_VAL_SH) |
           (lock_acq_id & 0x1F);
}

} // namespace xdna_txn

// DMA direction for shim-queue operations.
enum class xdna_dma_dir : uint32_t {
    S2MM = 0,   // host -> device
    MM2S = 1,   // device -> host
};

// Shim DMA buffer descriptor (the payload of a BLOCKWRITE). Strides and sizes
// are given as real element/byte counts; the emitter encodes (value - 1).
struct xdna_bd {
    uint32_t buf_len   = 0;   // total transfer length in bytes
    uint32_t buf_off   = 0;   // byte offset into the host buffer
    uint32_t d0_size   = 0;   // dim0 size (elements)
    uint32_t d0_stride = 0;   // dim0 stride (bytes)
    uint32_t d1_size   = 0;   // dim1 size (elements)
    uint32_t d1_stride = 0;   // dim1 stride (bytes)
    uint32_t d2_stride = 0;   // dim2 stride (bytes); d2 size inferred
    uint32_t ax_cache  = 0;   // AXI cache bits (usually 2)
    uint32_t iter_size   = 1; // iteration count (outermost repeat)
    uint32_t iter_stride = 1; // iteration stride (bytes)
    uint32_t next_bd   = 0;   // next BD id in the chain (0 = none)
    bool     valid     = true;

    // packet header (only when used with packet routing)
    bool     packet_enable   = false;
    uint32_t packet_type     = 0;
    uint32_t packet_id       = 0;
    uint32_t out_of_order_id = 0;
};

// Builder state. Data only; the API lives below as C-style functions.
struct xdna_seq {
    uint32_t n_cols = 8;        // AIE columns
    uint32_t n_rows = 6;        // total rows (mem tiles + cores)
    uint32_t mem_tile_rows = 1;
    std::vector<uint32_t> ops;
    uint32_t n_instr = 0;
};

// 32-bit register write (also used for RTP values and DMA push-queue).
void xdna_seq_write(xdna_seq * seq, uint32_t col, uint32_t row, uint32_t reg, uint32_t value);

// Shim DMA buffer descriptor write.
void xdna_seq_blockwrite(xdna_seq * seq, uint32_t col, uint32_t row, uint32_t bd_id, const xdna_bd * bd);

// Bind a previously-written BD's address register to host buffer arg_idx.
void xdna_seq_ddr_patch(xdna_seq * seq, uint32_t col, uint32_t row, uint32_t bd_id,
                        uint32_t arg_idx, uint32_t arg_offset);

// Masked register write (also used to issue a task-complete token).
void xdna_seq_maskwrite(xdna_seq * seq, uint32_t col, uint32_t row, uint32_t reg,
                        uint32_t value, uint32_t mask);

// Push a BD onto a shim DMA queue. `issue_token` stamps the completion so a
// later wait_token() can sync on this transfer.
void xdna_seq_push_queue(xdna_seq * seq, uint32_t col, uint32_t row, uint32_t bd_id, xdna_dma_dir dir,
                         uint32_t channel, bool issue_token, uint32_t repeat);

// Issue a task-complete token on a DMA channel (MASKWRITE form).
void xdna_seq_issue_token(xdna_seq * seq, uint32_t col, uint32_t row, xdna_dma_dir dir,
                          uint32_t channel, uint32_t ctrl_pkt_id);

// Wait for a task-complete token on a DMA channel.
void xdna_seq_wait_token(xdna_seq * seq, uint32_t col, uint32_t row, xdna_dma_dir dir, uint32_t channel);

// Assemble the full stream (4-word header + instructions).
std::vector<uint32_t> xdna_seq_build(const xdna_seq * seq);

// GEMM tile/geometry constants baked into the IRON design (see kernels/gemm.py
// and ggml/src/ggml-xdna/CMakeLists.txt, which defines GGML_XDNA_* for the
// backend target). The geometry is fixed by the compiled xclbin; only the
// shape is dynamic. The values must match the CMake definitions.
#ifndef GGML_XDNA_GEMM_M
#define GGML_XDNA_GEMM_M 32
#endif
#ifndef GGML_XDNA_TILE_M
#define GGML_XDNA_TILE_M 8
#endif
#ifndef GGML_XDNA_GEMM_M_BIG
#define GGML_XDNA_GEMM_M_BIG 64
#endif
#ifndef GGML_XDNA_TILE_M_BIG
#define GGML_XDNA_TILE_M_BIG 16
#endif

// Per-tile RTP buffer base in the core tile (input_with_addresses.mlir);
// it shifts with the L1 buffer sizes: 0xc800 for tile_m=8 (decode, r=4),
// 0xd000 for tile_m=16 (prefill, bfp16 r=8).
constexpr int XDNA_RTP_BASE_TILE_M8  = 0xc800;
constexpr int XDNA_RTP_BASE_TILE_M16 = 0xd000;
#ifndef GGML_XDNA_TILE_K
#define GGML_XDNA_TILE_K 64
#endif
#ifndef GGML_XDNA_TILE_N
#define GGML_XDNA_TILE_N 64
#endif
#ifndef GGML_XDNA_N_COLS
#define GGML_XDNA_N_COLS 8
#endif
#ifndef GGML_XDNA_N_COMPUTE_ROWS
#define GGML_XDNA_N_COMPUTE_ROWS 4
#endif

struct xdna_gemm_tiles {
    int M        = GGML_XDNA_GEMM_M;      // baked M block
    int tile_m   = GGML_XDNA_TILE_M;      // per-core M tile
    int tile_k   = GGML_XDNA_TILE_K;      // per-core K tile
    int tile_n   = GGML_XDNA_TILE_N;      // per-core N tile
    int n_cols   = GGML_XDNA_N_COLS;      // AIE columns
    int n_compute_rows = GGML_XDNA_N_COMPUTE_ROWS;  // compute tile rows
    // Per-tile RTP buffer base in the core tile (XDNA_RTP_BASE_TILE_M*).
    int rtp_base = XDNA_RTP_BASE_TILE_M8;
};

// True when the dims fit the baked geometry: M <= M block, K multiple of
// tile_k, N at least one column tile and a multiple of tile_n*n_cols.
bool xdna_gemm_seq_supported(const xdna_gemm_tiles * tiles, int M, int K, int N);

// Build the full TXN stream for C = A @ B with the geometry described by
// `tiles`. Buffer layout (row-major): arg 0 = A [M x K] bf16, arg 1 = B
// [K x N] bf16, arg 2 = C [M x N] f32. M must equal the baked block.
// `b_offset` (bytes) shifts the B base address, so a K-block can point into a
// persistent full-weight buffer.
bool xdna_gemm_seq_build(xdna_seq * seq, const xdna_gemm_tiles * tiles, int M, int K, int N,
                         uint32_t b_offset = 0);
