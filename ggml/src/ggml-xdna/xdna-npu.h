#pragma once

// Thin wrapper around the XRT/XDNA runtime. Keeps XRT types out of the ggml
// backend layer so ggml-xdna.cpp stays readable when XRT is not available.

#include <cstddef>
#include <cstdint>

struct ggml_xdna_npu;

enum ggml_xdna_op {
    GGML_XDNA_OP_GEMM,
#if defined(GGML_XDNA_EXPERIMENTAL)
    GGML_XDNA_OP_ADD,
    GGML_XDNA_OP_MUL,
    GGML_XDNA_OP_RMS_NORM,
    GGML_XDNA_OP_ADD_RMS_MUL,
    GGML_XDNA_OP_GDN,
#endif
};

// Geometry of one GEMM artifact. The per-worker tile and the worker grid are
// baked into the xclbin; kt and mb enter the instruction stream through the
// operand sizes. nb is always 1 in the artifact: the host loops over output
// column blocks of grid_c*tile_n.
//
//   C[mb*grid_r*tile_m, grid_c*tile_n] = A[mb*grid_r*tile_m, kt*tile_k]
//                                      @ B[kt*tile_k, grid_c*tile_n]
//
// B is staged once in MemTile and replayed mb times, so a long prefill does
// not re-read weights for every row block.
struct ggml_xdna_gemm_shape {
    int tile_m, tile_k, tile_n;
    int grid_r, grid_c;
    int kt, nb, mb;
};

struct ggml_xdna_npu_stats {
    uint64_t n_calls;    // ggml ops dispatched
    uint64_t n_runs;     // kernel submissions (one per tile)
    uint64_t ns_total;   // wall time inside the submit path, incl. host copies
    uint64_t ns_submit;  // wall time in run.start() + run.wait() only
    // Result readback and its cache maintenance. Kept apart from ns_submit
    // because it is host work that happens after the NPU is already idle.
    uint64_t ns_sync;
};

// Opens NPU device 0 and reports its name. Does not load any kernel.
// Returns false when no XDNA device is usable (or XRT was not compiled in).
bool ggml_xdna_npu_probe(char * name, size_t name_size);

void ggml_xdna_npu_free(ggml_xdna_npu * npu);

void ggml_xdna_npu_get_stats(const ggml_xdna_npu * npu, struct ggml_xdna_npu_stats * stats);

const char * ggml_xdna_op_name(enum ggml_xdna_op op);

// Loads a GEMM artifact. The shape must match what the artifact was built with;
// it decides the BO sizes and is not discoverable from the xclbin.
ggml_xdna_npu * ggml_xdna_npu_gemm_init(const char * xclbin_path,
                                        const char * insts_path,
                                        const struct ggml_xdna_gemm_shape * shape);

int ggml_xdna_npu_gemm_m(const ggml_xdna_npu * npu);
int ggml_xdna_npu_gemm_k(const ggml_xdna_npu * npu);
int ggml_xdna_npu_gemm_n(const ggml_xdna_npu * npu);

// Byte size of the swizzled operand/result streams one submit moves.
size_t ggml_xdna_npu_gemm_a_bytes(const ggml_xdna_npu * npu);
size_t ggml_xdna_npu_gemm_b_bytes(const ggml_xdna_npu * npu);
size_t ggml_xdna_npu_gemm_c_bytes(const ggml_xdna_npu * npu);

// Device-visible memory for repacked weights. Weights are written once at model
// load and read by every submit, so they live here rather than being copied into
// an operand BO per call - at 7 MiB a tensor that copy would cost more than the
// matmul.
struct ggml_xdna_buf;

ggml_xdna_buf * ggml_xdna_buf_alloc(ggml_xdna_npu * npu, size_t nbytes);
void            ggml_xdna_buf_free(ggml_xdna_buf * buf);
void *          ggml_xdna_buf_host(ggml_xdna_buf * buf);
void            ggml_xdna_buf_sync_to_device(ggml_xdna_buf * buf, size_t offset, size_t nbytes);

// GEMM operands live in device-visible memory: the caller packs A into the
// slice it is about to submit and reads C back from the matching slice, so a
// submit moves nothing between host buffers.
//
// The buffers are shared per device, so a caller holds acquire/release around
// the whole pack -> submit -> read sequence.
int            ggml_xdna_npu_gemm_kb_max (const ggml_xdna_npu * npu);
uint16_t *     ggml_xdna_npu_gemm_a_slice(ggml_xdna_npu * npu, int k_block);
const float *  ggml_xdna_npu_gemm_c_slice(ggml_xdna_npu * npu, int k_block);

void ggml_xdna_npu_gemm_acquire(ggml_xdna_npu * npu);
void ggml_xdna_npu_gemm_release(ggml_xdna_npu * npu);

// Wait for the outstanding runlist from submit_list_async, then
// sync C views back to the host. No-op when nothing is pending.
bool ggml_xdna_npu_gemm_runlist_wait(ggml_xdna_npu * npu);

// Like submit_list but returns after execute() without wait/C-sync so the host
// can pack the next A block while the NPU runs. Must call runlist_wait next.
bool ggml_xdna_npu_gemm_submit_list_async(ggml_xdna_npu * npu,
                                          ggml_xdna_buf * buf,
                                          const size_t *  w_offs,
                                          int             kb_n);

// Double-buffered A/C: ping-pong bank in {0,1}. Pack into bank, submit from
// bank, wait/scatter the other. Default bank 0 preserves single-buffer callers.
int  ggml_xdna_npu_gemm_n_banks(const ggml_xdna_npu * npu);
void ggml_xdna_npu_gemm_set_bank(ggml_xdna_npu * npu, int bank);
uint16_t *    ggml_xdna_npu_gemm_a_slice_bank(ggml_xdna_npu * npu, int bank, int k_block);
const float * ggml_xdna_npu_gemm_c_slice_bank(ggml_xdna_npu * npu, int bank, int k_block);

#if defined(GGML_XDNA_EXPERIMENTAL)
// Elementwise / fused / GDN kernels and act-x-act GEMM. Production builds
// compile none of this.

ggml_xdna_npu * ggml_xdna_npu_init(enum ggml_xdna_op op,
                                   const char *      xclbin_path,
                                   const char *      insts_path,
                                   size_t            tile_elems);

bool ggml_xdna_npu_binary_f32(ggml_xdna_npu * npu, const float * a, const float * b, float * dst, size_t n);

bool ggml_xdna_npu_rms_norm_f32(ggml_xdna_npu * npu, const float * x, float * dst, int64_t n_rows, float eps);

bool ggml_xdna_npu_add_rms_mul_f32(ggml_xdna_npu * npu,
                                   const float *  a,
                                   const float *  b,
                                   const float *  weight,
                                   float *        add_dst,
                                   float *        mul_dst,
                                   int64_t        n_rows,
                                   int64_t        weight_n_rows,
                                   float          eps);

bool ggml_xdna_npu_gemm_submit_ab_list(ggml_xdna_npu * npu, int kb_n);
uint16_t * ggml_xdna_npu_gemm_b_host(ggml_xdna_npu * npu);
uint16_t * ggml_xdna_npu_gemm_b_slice(ggml_xdna_npu * npu, int k_block);
bool ggml_xdna_npu_gemm_submit_ab(ggml_xdna_npu * npu, int k_block);

struct ggml_xdna_gdn_shape {
    int S;
    int ROWS;
    int CS;
    int workers; // 1 or S/ROWS
};

ggml_xdna_npu * ggml_xdna_npu_gdn_init(const char * xclbin_path,
                                       const char * insts_path,
                                       const struct ggml_xdna_gdn_shape * shape);

bool ggml_xdna_npu_gdn_submit(ggml_xdna_npu * npu,
                              const float *   tok,
                              const float *   state_strip,
                              float *         attn_strip,
                              float *         new_state_strip);

bool ggml_xdna_npu_gdn_submit_full(ggml_xdna_npu * npu,
                                   const float *   tok,
                                   const float *   state,
                                   float *         attn,
                                   float *         new_state);

bool ggml_xdna_npu_gdn_submit_full_chunks(ggml_xdna_npu * npu,
                                          const float *   tok_chunks,
                                          int             n_chunks,
                                          const float *   state,
                                          float *         attn_chunks,
                                          float *         new_state);
#endif
