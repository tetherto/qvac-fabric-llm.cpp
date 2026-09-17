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
// A shim buffer descriptor. The address-generation fields describe a strided
// walk: d0 is the innermost run, d1 repeats it, d2 repeats that, and `iter`
// repeats the whole thing. Every size is a count of 32-bit words and every
// stride is in words too; the writer stores each stride as stride-1, so pass
// the stride itself. A size of 0 means "do not wrap", which is what a plain
// linear transfer wants - a size of 1 wraps the descriptor after a single
// word.
//
// The encoding is IRON's own, read back from an artifact it built for a known
// two-dimensional access pattern rather than guessed: for a TensorAccessPattern
// of sizes [4, 128] and strides [1024, 1] over f32 it emits
//
//   buf_len 512   d0 size 128 stride 1   d1 size 4 stride 1024
//
// so a two-dimensional drain of `reps` runs of `run` words, `stride` words
// apart, is buf_len = run*reps, d0_size = run, d0_stride = 1, d1_size = reps,
// d1_stride = stride. Anything unsure here is worth checking the same way:
// build the pattern in IRON, disassemble the .insts.bin it ships beside the
// xclbin (the .insts.bin it ships beside the xclbin and read the words.
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
    // Highest descriptor id + 1 written on each column's shim, so a stream
    // appended to this one can start where it left off. Reprogramming a
    // descriptor whose transfer has not drained loses it silently.
    uint32_t bd_used[8] = {};
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
#define GGML_XDNA_TILE_M_BIG 32
#endif
#ifndef GGML_XDNA_TILE_M_BIG_I8
#define GGML_XDNA_TILE_M_BIG_I8 64
#endif

// Per-tile RTP buffer base in the core tile, read back from the compiled
// input_with_addresses.mlir (the rtp buffer's address; the insts header carries
// it as 0x200000 + this). 0x8800 for tile_m=8 (decode, native r=4) and 0x2d00
// for both prefill tiles: with mlir-aie 1.4.3 the allocator drops the prefill
// rtp right after the first B_L2L1 object, so tile_m 32 (bf16) and 64 (int8)
// share the offset even though their A/B/C buffers differ. The base moves with
// the toolchain's placement, not with the geometry - an earlier 1.4.2 build put
// them at 0xa000 and 0xe000 - so a toolchain change means re-reading this value
// from a freshly compiled .prj, not guessing it.
constexpr int XDNA_RTP_BASE_TILE_M8  = 0x8800;
constexpr int XDNA_RTP_BASE_TILE_M16 = 0x9000;
constexpr int XDNA_RTP_BASE_TILE_M32 = 0x2d00;
constexpr int XDNA_RTP_BASE_TILE_M64 = 0x2d00;
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
    // A/B element size in bytes: 2 = bf16, 1 = int8 (int8 x int8 -> int32).
    // C is always 4 bytes (f32 or raw int32); the stream BD lengths/offsets
    // scale with eb.
    int elem_bytes = 2;
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

// Fixed geometry of one CS=64-token GDN prefill chunk (S=128, H=16, see
// kernels/gdn_prefill.py). Byte/word counts per shim column of the three
// host BOs: arg 0 = tok [H][CS][3*S+2] bf16, arg 1 = packed state|attn
// [H][NS][ROWS*S + CS*ROWS] bf16, arg 2 = same packed buffer (in place).
struct xdna_gdn_prefill_geom {
    int n_cols = 8;             // shim columns (H/2)
    int state_col_bytes = 98304;  // packed column, bytes (= 4 workers x PACKED_N x 2)
    int state_col_words = 24576;  // packed column in 4-byte words
    int tok_col_bytes  = 98816;   // tok column, bytes (= 2 heads x CS x (3*S+2) x 2)
    int tok_col_words  = 24704;
};

// Build the TXN stream for one CS=64-token chunk of the GDN prefill kernel:
// state seed (8 columns, arg 1), then per column the tok fill (arg 0) and the
// packed drain (arg 2); the last column's drain stamps the completion token the
// host waits. The stream is bound once and reused for every chained chunk - the
// state seed re-copies the in-place packed BO, so the updated device state
// flows into the next chunk.
bool xdna_gdn_prefill_seq_build(xdna_seq * seq, const xdna_gdn_prefill_geom * g);

// Geometry of the merged conv+norm+gdn decode kernel (fused_layer.xclbin,
// kernels/fused_layer.py). The worker layout is the rec_full design: conv on
// cols 0..conv_cols-1, norm on the next norm_cols, bf16-vector gdn on the last
// gdn_cols. The defaults match one 1024-wide Qwen3.5 gated-delta-net layer
// (kernels/fused_layer.py resolve()); a default-constructed geometry is valid.
struct xdna_attn_gdn_geom {
    int feed_n    = 1024;   // floats per conv feed block
    // Floats of a slot the conv object carries; see CONV_SLOT in
    // kernels/attn_gdn_gated.py. Less than feed_n leaves the conv weights out
    // of the stream and the stage's output wrong - a diagnostic only.
    int feed_slot = 1024;
    int sv        = 128;    // conv group / head slice width
    int head_norm = 387;    // floats per head in the X BO (q|k|v|eg|b|scale)
    int pkv_n     = 387;    // floats per pkv chunk object
    int pkvb_n    = 3096;   // floats per head pkvb (n_obj * pkv_n)
    int n_block   = 48;     // conv feed blocks (groups)
    // Feed groups one conv object carries. gdn's 16 KB objects move at the
    // shim's full rate where conv's 4 KB ones manage a fifth of it, and the
    // stage has no arithmetic in it, so this is what its time is made of.
    int conv_gpo  = 4;
    // Where the conv stage's two drains meet the shim, per conv column. A
    // shim tile has one port for all of its channels, so the feed streaming in
    // and the x and history writes going out must not share a column - it is
    // the one thing that told this stage apart from the gdn block, which does
    // reach the tile's rate. Read back from the artifact; the design pins them
    // (kernels/attn_gdn_gated.py).
    // Defaults are the conv columns themselves; CONV_SPREAD=1 in the design
    // moves them to { {5,1}, {4,0} } for x and { {5,0}, {7,1} } for the
    // history, which measures the same.
    int conv_x_col[2] = { 0, 1 };
    int conv_x_ch [2] = { 1, 1 };
    int conv_h_col[2] = { 0, 1 };
    int conv_h_ch [2] = { 0, 0 };
    int conv_cols = 2;      // conv columns (cols 0-1)
    int norm_cols = 2;      // norm columns (cols 2-3)
    int gdn_cols  = 4;      // gdn columns (cols 4-7)
    int n_vh      = 16;     // value heads
    int n_obj     = 8;      // pkv chunks per head
    // Shim streams the gdn state is split across, in each direction. The
    // block's cost is that stream: 512 KB in and 512 KB out a layer, which one
    // channel moves in 119 us with the arithmetic hidden underneath.
    int gdn_state_streams = 1;
    // The norm stage hands its chunks to the gdn cores through a MemTile
    // instead of writing them to DDR for the gdn fill to read back (PKV_ONCHIP
    // in kernels/attn_gdn_gated.py). With it the stage has no drain and the
    // block no pkv fill, and the two stop being separate phases.
    int pkv_onchip = 1;
    // The gdn block's attn reaches the gated tile over the array rather than
    // through DDR (GATED_ATTN_ONCHIP in attn_gdn_gated.py). The per-round
    // attn drain goes away and the gated fill moves ahead of the gdn rounds,
    // because the tile now consumes a head while the block is still running.
    int attn_onchip = 1;
    // Bytes of the ssm_out activation the gated stage produces. Non-zero means
    // the design gives it a fifo of its own (GATED_ACT_SPLIT in
    // attn_gdn_gated.py), so the stream drains it with a plainly patched
    // descriptor on its own column - which is what a projection appended to
    // this stream needs in order to read it back. The gated output's own drain
    // cannot be that: it wants the bit31 patch form, and that form wants
    // offset zero.
    //
    // Splitting the *drain* instead - two descriptors over the one object -
    // does not work at all; one object cannot be drained by two descriptors,
    // and the stream times out with or without anything appended.
    int gated_act_bytes = 0;
    int gated_act_col   = 7;   // its shim column (S2MM channel 0)
    int gated_act_arg   = 5;   // the argument it drains into
    int gated_act_off   = 0;   // and the byte offset in it
    int rows      = 2048;   // bf16 state rows per chunk object
    int chunk     = 16;     // state rows per chunk
    int n_cols    = 8;      // shim columns
    // A-half fused epilogue (attn_gdn_gated.xclbin, kernels/attn_gdn_gated.py):
    // when azg_n > 0 the arg4 BO is the azg head buffer (per head
    // [attn|z|gamma|hh], 3*sv+1 floats), gdn attn drains use the azg head
    // stride and a gated phase is appended (azg fill on the first norm column's
    // MM2S ch1, out drain on its S2MM ch1). 0 keeps the legacy attn layout.
    int azg_n     = 0;      // floats per azg head (0 = legacy attn BO)
    int out_words = 0;      // gated out object words (10244 B / 4)
    // Byte offset the gated output is drained to inside its argument. Zero
    // needs the bit31 patch form the compiled IRON stream uses; a non-zero one
    // is patched plainly, the way every other drain in this stream is - and
    // those are the ones another stage reads back reliably in the same
    // dispatch (the conv drains write x and the norm fills read it).
    int out_base = 0;
};

// Hand-built per-token TXN stream for the merged conv+norm+gdn recurrent-core
// xclbin (fused_layer.xclbin, or attn_gdn_gated.xclbin when azg_n is set).
// One host-built stream per token runs the phases in time (conv -> norm -> gdn
// [- > gated]) with BD/channel reuse in ONE xrt run.  Run BO args (same as the
// rec_full/attn_cn/gdn layouts): arg0 feed (n_block x feed_n f32), arg1 x
// (n_vh x head_norm f32 tails + q/k/v slices), arg2 pkvb (n_vh x pkvb_n f32
// norm output), arg3 state (n_vh x n_obj x rows bf16, in place), arg4 attn
// (n_vh x sv f32 readback) or azg (n_vh x azg_n f32 when fused) and, in the
// fused layout only, arg5 out (gated scratch + aq + d_a). `schedule`: 0 = IRON
// column-major per phase; 1 = slot-major; 2 = phased (groups of 2); 4 =
// BD-bank pipelining. The stream is bound once at load and replayed every token
// (BO contents change, offsets do not).
bool xdna_attn_gdn_build(xdna_seq * seq, const xdna_attn_gdn_geom * g,
                             int schedule, int phase = -1);
