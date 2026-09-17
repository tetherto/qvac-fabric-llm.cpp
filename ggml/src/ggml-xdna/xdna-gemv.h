#pragma once

// Decode GEMV on the NPU: one activation row against packed quantized weights
// (kernels/gemv_q4.py + gemv_q4.cc). The group parameters are applied to the
// accumulator rather than materialising bf16 weights, so the weight stream
// stays at 0.75 B/value (q4g32) or 1.25 B/value (q8g16) and the arithmetic is
// exact apart from one f32 rescale per 32 values of K.
//
// This is the decode path only: ggml calls it for MUL_MAT with a single
// activation row, where padding M up to the GEMM's baked 32-row block wastes
// 32x the work.
//
// The core program depends only on (format, n_core, k_tile), so one artifact
// per format serves every shape and the stream is built here
// (xdna_gemv_seq_build) rather than taken from IRON. The per-dispatch counts -
// K tiles and output chunks - ride in the first object of the activation
// stream: a runtime-parameter register is written while the core is already
// running, and the core read the previous dispatch's count.
//
// That matters for more than tidiness. A hardware context is per xclbin, and
// the NPU reconfigures the array when consecutive ops come from different
// ones: measured against the per-shape artifacts, an op whose context was
// already resident cost 0.148 ms and one that switched cost 2.5 ms.

#include "xdna-quant.h"
#include "xdna-runtime.h"
#include "xdna-seq.h"

#include <cstdint>
#include <string>
#include <vector>

// The baked core geometry, shared by every shape. Must agree with the
// XDNA_GEMV_* variables in this directory's CMakeLists and with gemv_q4.py.
enum {
    XDNA_GEMV_N_CORE = 32,     // output columns a core owns
    XDNA_GEMV_K_TILE_Q4 = 512, // K per streamed tile, 4-bit codes
    XDNA_GEMV_K_TILE_Q8 = 256, // K per streamed tile, 8-bit codes
    XDNA_GEMV_ACT_TILE  = 2112,// codes, per-group sums and scales, flags, width
    XDNA_GEMV_COLS   = 8,      // AIE columns the design instantiates
    XDNA_GEMV_ROWS   = 4,      // compute rows per column
    // The same array split the other way: half the rows, twice the columns per
    // core, so a pass still covers 1024 outputs and every column still streams.
    // It exists because the fused layer's GDN core takes one row of every
    // column, leaving two - and because the cost of arriving at a dispatch
    // whose context was not the last to run may scale with how much of the
    // array the design configures, which this measures.
    XDNA_GEMV_ROWS_HALF   = 2,
    XDNA_GEMV_N_CORE_HALF = 64,
    // The split that shares the array with the fused layer's GDN core
    // (kernels/fused_layer.py): one core per column, so the shim feeds it
    // directly - no weight split, no output join, no MemTile channels, which
    // is what let the two designs into one xclbin. A weight tile is four times
    // what it is at 32 columns per core, so the K tile halves to keep L1
    // inside its budget.
    XDNA_GEMV_ROWS_FUSED      = 1,
    XDNA_GEMV_N_CORE_FUSED    = 128,
    XDNA_GEMV_K_TILE_Q4_FUSED = 256,
    XDNA_GEMV_K_TILE_Q8_FUSED = 128,
};

// Which array split to use when the caller has no preference.
// GGML_XDNA_GEMV_ROWS=2 or =4 forces one everywhere; without it the fused
// layer takes the half-height design and the per-op path the full one. Both
// are measured: half the array costs half as much to arrive at (1417 us
// against 3031), which is most of a fused layer, but the per-op path never
// switches and loses 5% to the narrower streaming.
enum xdna_gemv_split {
    XDNA_GEMV_SPLIT_DEFAULT, XDNA_GEMV_SPLIT_FULL,
    XDNA_GEMV_SPLIT_HALF, XDNA_GEMV_SPLIT_FUSED,
};
bool xdna_gemv_use_half(xdna_gemv_split want);

// One decode GEMV: the baked geometry plus the shape this dispatch runs.
struct xdna_gemv_geom {
    xdna_wfmt fmt = XDNA_WFMT_NONE;
    int K      = 0;    // rows of the weight (activation length)
    int N      = 0;    // output length rounded up to a whole pass of the array
    int n_real = 0;    // columns the weight actually has
    int cols = XDNA_GEMV_COLS;
    // The FFN's activation closes on the cores: the dispatch runs the gate and
    // up projections together and returns silu(gate)*up, half as many values.
    bool epilogue = false;

    // Carried per geometry rather than read from the enums, so both array
    // splits can be built and one chosen at run time.
    int  n_core_v = XDNA_GEMV_N_CORE;
    int  rows_v   = XDNA_GEMV_ROWS;
    // Set when this dispatch runs on the merged layer artifact: it changes the
    // K tile, the artifact it loads and where its transfers meet the shim.
    bool fused_v  = false;

    int  rows()   const { return rows_v; }
    int  n_core() const { return n_core_v; }
    // A 4-bit tile spans twice the K of an 8-bit one, which is what makes
    // both the same number of bytes.
    int  k_tile() const {
        if (fused_v) {
            return fmt == XDNA_WFMT_Q4G32 ? XDNA_GEMV_K_TILE_Q4_FUSED
                                          : XDNA_GEMV_K_TILE_Q8_FUSED;
        }
        return fmt == XDNA_WFMT_Q4G32 ? XDNA_GEMV_K_TILE_Q4 : XDNA_GEMV_K_TILE_Q8;
    }
    int  cores()  const { return cols * rows_v; }
    // Output columns one pass over the array produces.
    int  chunk()  const { return cores() * n_core_v; }
    int  n_out()  const { return N / chunk(); }
    // Values a core writes per chunk: the epilogue folds two into one.
    int  out_per_core() const { return epilogue ? n_core_v / 2 : n_core_v; }
    int  n_tiles() const { return K / k_tile(); }
    int  group()  const { return fmt == XDNA_WFMT_Q4G32 ? XDNA_Q4G32_GROUP : XDNA_Q8G16_GROUP; }
    // Packed bytes of one (k_tile x n_core) weight tile.
    size_t tile_bytes() const;
    // Super-block records a tile carries per lane group; see tile_bytes().
    int    n_sup() const;
    // Columns one multiply covers on the core. AIE2P's int8 datapath is 64
    // lanes wide; a core narrower than that has to stay at 32. Mirrors
    // vec_for() in kernels/gemv_q4.py.
    int    lane() const { return n_core() >= 64 ? 64 : 32; }
    // Sized for the wider code, so one object type serves both.
    size_t act_tile_bytes() const { return XDNA_GEMV_ACT_TILE; }
    // The count header, then the K tiles once per output chunk. The repeats
    // are held in DDR rather than replayed by re-pointing a descriptor per
    // chunk, so the whole activation stream is one descriptor: a shim tile has
    // sixteen descriptor ids for all of its channels, and a stream that fills
    // them loses transfers when one is reprogrammed before it has drained.
    size_t act_bytes() const {
        return (size_t) (1 + n_out() * n_tiles()) * act_tile_bytes();
    }
    // Per column: every output chunk, every K tile, every row.
    size_t col_weight_bytes() const {
        return (size_t) n_out() * n_tiles() * rows_v * tile_bytes();
    }
    size_t weight_bytes() const { return (size_t) cols * col_weight_bytes(); }
    size_t out_bytes() const { return (size_t) N * sizeof(float); }

    bool valid() const;
    // Artifact stem: one per format, e.g. "gemv_q4g32_n32_t512_c8".
    std::string stem() const;
    // Cache key of the instruction stream, which does depend on the shape.
    std::string seq_key() const;

    // Every column is always fed. A column left idle would have its cores
    // enter the chunk loop once anyway and block on a fifo that is never
    // written, and they stay blocked into the next dispatch, so N is rounded
    // up with zero weights instead of narrowing the array.
};

// Output chunks one dispatch can carry, limited by the shim descriptor ids.
int xdna_gemv_max_chunks(void);

// Build the instruction stream for `geom`: one linear weight descriptor per
// column, the count header, then an activation and an output descriptor per
// output chunk. Returns false on an invalid geometry.
// Where an appended stream's arguments, descriptors and activation sit. The
// defaults are a stream of its own; the recurrent layer appends the ssm_out
// projection to the core's, so its arguments come after the core's six, its
// descriptors after the ones the core's columns have taken, and its
// activation is the window of the core's output buffer the gated stage wrote.
// Where an appended stream's arguments, descriptors and buffers sit.
//
// Only the low argument indices can be patched: a stream that names arguments
// six through eight times out, the same stream naming zero through two runs,
// and the recurrent core's own stream patches zero through five. So a stream
// appended to another has to fit inside six arguments between them, which is
// why each of the three is placed individually rather than as a base.
struct xdna_gemv_seq_opts {
    int      arg_base = 0;
    int      bd_base  = 0;   // -1 = after the stream's own descriptors
    // Negative means arg_base + 0 / 1 / 2, a buffer set of its own.
    int      w_arg    = -1;
    int      act_arg  = -1;
    int      o_arg    = -1;
    uint32_t w_off    = 0;   // the weights' offset in their argument
    uint32_t act_off  = 0;   // the activation's offset in its argument
    uint32_t out_off  = 0;   // the output's offset in its argument
    // Bytes between one output stream's destination and the next. Zero means
    // they are consecutive, which is a plain output buffer. Set it to the
    // activation tile size and the projection drains straight into the tiles
    // the next dispatch reads - one stream to a tile, because a stream carries
    // exactly a tile's worth of columns - so its result never goes near the
    // host and the two dispatches can share a runlist.
    uint32_t out_stream_stride = 0;
    // The prologue's second input: which argument the host's half of the tiles
    // lives in and where. Negative means the same argument and offset as the
    // activation, which is what every stream wants except the one whose tiles
    // the array writes.
    int      h_arg = -1;
    uint32_t h_off = 0;
    // Bisecting an appended stream: 1 = weights only, 2 = and the activation,
    // 3 = and the output descriptors but no wait, 0 = the whole thing.
    int      stages   = 0;
};

bool xdna_gemv_seq_build(struct xdna_seq * seq, const xdna_gemv_geom & geom,
                         const xdna_gemv_seq_opts * opt = nullptr);

// Where the pair's three buffers are named and based, so the same two phases
// can be appended to somebody else's stream instead of standing alone. The
// defaults are the standalone dispatch: arguments 0, 1, 2 from offset zero,
// descriptors from zero.
struct xdna_gemv_pair_opts {
    int      w_arg  = 0;
    int      a_arg  = 1;
    int      o_arg  = 2;
    uint32_t w_base = 0;
    uint32_t a_base = 0;
    uint32_t o_base = 0;
    // Descriptors are reused, not added to: by the time this section runs,
    // every transfer before it in the stream has been waited for.
    int      bd_base = 0;
    // As above, for the pair's first phase - the one whose tiles come from
    // the dispatch before it.
    int      h_arg = -1;
    uint32_t h_base = 0;
};

// Build the two-phase stream; see xdna_gemv_pair. `w2_off` and `a2_off` are
// where the second dispatch's weights and activation sit in the shared
// buffers, relative to the bases in `opt`.
bool xdna_gemv_seq_build_pair(struct xdna_seq * seq, const xdna_gemv_geom & g1,
                              const xdna_gemv_geom & g2,
                              size_t w2_off, size_t a2_off,
                              const xdna_gemv_pair_opts * opt = nullptr);

// The geometry that serves a [K x N] weight of `type`, or an invalid one when
// none does. N is the total across the weights of one dispatch.
xdna_gemv_geom xdna_gemv_variant(enum ggml_type type, int64_t K, int64_t N,
                                 bool epilogue = false,
                                 xdna_gemv_split split = XDNA_GEMV_SPLIT_DEFAULT);

// Pack ggml-quantized [K x N_i] weights (N_i rows of K) into the tile order
// the design streams: [aie-col][chunk][tile][row], each tile lane-group major.
// Several weights that share an activation are concatenated along N and run as
// one dispatch, which is how a layer's independent projections - gate with up,
// q with k and v - cost one kernel launch instead of several. `dst` must hold
// geom.weight_bytes(). Returns false when a type does not repack.
// `colmap`, when given, maps each destination column to an index in the
// concatenated weights: with the epilogue a core must own the gate half and
// the up half of the same values, which is not their concatenated order.
bool xdna_gemv_pack_weights(const xdna_gemv_geom & geom,
                            const struct ggml_tensor * const * ws, int n_w,
                            const int32_t * colmap,
                            std::vector<uint8_t> & dst);

// The column map for an epilogue dispatch of `geom`, in concatenated order
// with the gate weight first.
void xdna_gemv_colmap(const xdna_gemv_geom & geom, std::vector<int32_t> & map);

// A loaded GEMV runner: the weight BO is filled once per tensor and stays on
// the device; only the activation row crosses per call.
struct xdna_gemv {
    xdna_device *  dev  = nullptr;
    xdna_kernel *  kern = nullptr;
    xdna_gemv_geom geom;

    xdna_buffer * w = nullptr;    // packed weights (persistent)
    xdna_buffer * a = nullptr;    // int8 activation codes + group sums
    xdna_buffer * o = nullptr;    // f32 partials, before the activation scale

    std::vector<uint8_t> host_a;  // staging for the activation tiles
    // The buffer set never changes, so the run is built once and restarted.
    xrt::run run;
};

// Load the artifact for `geom`, allocate the buffer set and upload `packed`
// (which must be geom.weight_bytes()). Returns nullptr on failure.
xdna_gemv * xdna_gemv_create(struct xdna_kernel_pool * pool, const xdna_gemv_geom & geom,
                             const std::vector<uint8_t> & packed);

void xdna_gemv_free(xdna_gemv * g);

// Two dispatches in one instruction stream. The first is an epilogue pair
// whose result is written as activation tiles, in the layout the cores emit,
// straight into the region the second reads; the second projects it. Nothing
// between them belongs to the host, which is the only reason they can share a
// stream - and sharing it saves a dispatch, measured at about 140 us of the
// FFN's 516.
//
// One buffer holds both weight sets and one both activation regions, because
// the kernel takes three arguments and a descriptor names one of them: the
// first dispatch's output descriptors name the activation argument rather
// than the output one.
struct xdna_gemv_pair {
    xdna_device *  dev  = nullptr;
    xdna_kernel *  kern = nullptr;
    xdna_gemv_geom g1;            // the epilogue pair, writing activation tiles
    xdna_gemv_geom g2;            // the projection that consumes them

    xdna_buffer * w = nullptr;    // [g1 weights][g2 weights]
    xdna_buffer * a = nullptr;    // [g1 activation][g2 activation]
    xdna_buffer * o = nullptr;    // g2's output
    // The host's half of g1's tiles, when the prologue has an input for it:
    // the residual, gamma and the words describing a tile. No dispatch writes
    // it, and nothing writes the tiles but the dispatch that drains into them.
    xdna_buffer * ah = nullptr;

    size_t w2_off = 0;            // byte offset of g2's weights
    size_t a2_off = 0;            // byte offset of g2's activation
    size_t o_tail_off = 0;        // where the output sits when it has no argument

    std::vector<uint8_t> host_a;  // staging for g1's activation only
    std::vector<uint32_t> insts;  // the stream, for the rebind probe
    xrt::run run;
};

// Load the artifact, allocate the buffer set, upload both weight sets and
// prime g2's activation headers (the cores fill its codes every token).
xdna_gemv_pair * xdna_gemv_pair_create(struct xdna_kernel_pool * pool,
                                       const xdna_gemv_geom & g1,
                                       const std::vector<uint8_t> & packed1,
                                       const xdna_gemv_geom & g2,
                                       const std::vector<uint8_t> & packed2);

void xdna_gemv_pair_free(xdna_gemv_pair * p);

// Quantize `act` for the first dispatch, run both, and write g2's output.
// The flag word the pair wants in its activation's last tile: close the
// chunk, quantize the epilogue's output, and group it the way the second
// projection reads. The post design writes that activation, so it reads the
// flag out of its input.
int32_t xdna_gemv_pair_last_flags(const xdna_gemv_pair * p);

// The pair with its activation already in tile layout, produced on device.
bool xdna_gemv_pair_run_packed(xdna_gemv_pair * p, const void * act_tiles,
                               float * out);

bool xdna_gemv_pair_run(xdna_gemv_pair * p, const float * act, float * out);

// The same, with the transition left to the array: the activation tiles carry
// this dispatch's input, the residual and the norm's gamma as numbers, and the
// design's prologue tile does the residual add, the reduction, gamma and the
// quantization (GGML_XDNA_ACT_RAW=1, and the artifact built with ACT_RAW=1).
// Nothing of the layer's transition is left on the host, which is what lets a
// layer's two dispatches go into one runlist.
bool xdna_gemv_pair_run_raw(xdna_gemv_pair * p, const float * acc,
                            const float * res, const float * gam, float * out);

// Write the host's half of a raw activation - the residual and gamma - before
// the dispatch that drains the rest of it into the same tiles, and dispatch
// the pair against an activation already complete. Between them the host does
// nothing, which is what a runlist needs.
bool xdna_gemv_pair_prep_raw(xdna_gemv_pair * p, const float * res,
                             const float * gam);
bool xdna_gemv_pair_dispatch(xdna_gemv_pair * p, float * out);
void xdna_gemv_pair_act_sync_from(xdna_gemv_pair * p);
void xdna_gemv_pair_act_sync_to(xdna_gemv_pair * p);

// Read what a fused run left in the tail of the activation buffer, for the
// layer whose FFN rides the core's own stream.
bool xdna_gemv_pair_out_from_tail(xdna_gemv_pair * p, float * out);

// True when this pair's first phase reads raw tiles, so the dispatch before it
// can drain into them; and the buffer those tiles live in.
bool xdna_gemv_pair_raw_act(const xdna_gemv_pair * p);
struct xdna_buffer * xdna_gemv_pair_act_buf(xdna_gemv_pair * p);
struct xdna_buffer * xdna_gemv_pair_w_buf(xdna_gemv_pair * p);
struct xdna_buffer * xdna_gemv_pair_h_buf(xdna_gemv_pair * p);
// True when this build expects the prologue's second input to be fed.
bool xdna_gemv_act_pro_h(void);
size_t xdna_gemv_pair_o_tail(const xdna_gemv_pair * p);
// Identifies the pair's two geometries, for naming a stream that contains it.
std::string xdna_gemv_pair_key(const xdna_gemv_pair * p);
int xdna_gemv_pair_k_tile(const xdna_gemv_pair * p);
int xdna_gemv_pair_n_tiles(const xdna_gemv_pair * p);

// Decode the activation the first dispatch wrote into the second's buffer, so
// the handover can be checked against what the host would have packed.
bool xdna_gemv_pair_read_mid(xdna_gemv_pair * p, std::vector<float> & mid);

// Run one decode token: quantize `act` (K floats) to int8 with a scale per
// group, run the kernel and write `out` (N floats). Returns false on failure.
//
// The scale is per group rather than per row for two reasons. It is local to
// the values a core holds, which is what lets an activation be produced on the
// device by a chained operator instead of coming back through the host; and it
// does not have to cover the row's outliers, which measured 3.5x tighter
// (1.9e-4 against 6.4e-4 relative).
//
// int8 and not int16: the code width is the multiply's operand width, and the
// q4g32 dispatches are limited by the multiply rather than the weight stream.
// See gemv_q4.cc.
bool xdna_gemv_run(xdna_gemv * g, const float * act, float * out);

// The same, with the activation already in the tile layout - written by
// whatever produced it, so nothing is quantized or packed here. Only for a
// dispatch of one output chunk, which is the only case where the tiles are not
// repeated per chunk.
// Read back a projection's output without running it, for a stream that has
// already been dispatched as part of another.
bool xdna_gemv_read_out(xdna_gemv * g, float * out);

bool xdna_gemv_run_packed(xdna_gemv * g, const void * act_tiles, float * out);
