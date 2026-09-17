#pragma once

// NPU weight formats for the XDNA backend.
//
// Decode is memory bound: the whole weight set crosses DDR once per token, so
// what the kernel streams decides the speed. The backend used to hand the GEMM
// bf16 weights (2 B/value, dequantized on the host) and the fused decode
// kernels a symmetric int4 grid with one scale per column (0.5 B/value). The
// first is 3-4x more DDR traffic than necessary, the second cannot represent
// what Q4_K stores - Q4_K is affine per 32 values, and collapsing that to one
// symmetric scale per column costs ~2.8e-1 relative RMS on ssm_out.
//
// These formats keep the DDR side quantized and stay exact with respect to the
// ggml block they come from, so the only error left is the bf16 rounding of
// the group scale, which the GEMM already has anyway:
//
//   q4g32  32 values: 16 B packed nibbles + an int8 scale and min  (0.594 B/val)
//          w[i] = q[i] * d + m,  q unsigned [0,15]   - exact for Q4_K
//   q8g16  16 values: 16 B int8 codes + an int8 scale and min      (1.19 B/val)
//          w[i] = q[i] * d + m,  q signed            - exact for Q4_K,
//          Q5_K and Q6_K alike
//
// Both decode with the same affine expression, so one AIE kernel shape serves
// both.
//
// q8g16 covers every quantized type the decode path sees, which is what lets
// the GEMV run from a single artifact. A group of 16 rather than 32 is what
// makes that exact: Q6_K carries one scale per 16 values, so a group of 32
// would have to merge two of them, while Q4_K's per-32 affine group simply
// repeats across its two halves. The cost over the 4-bit form is 0.75 B per
// value on the Q4_K tensors, which buys away a hardware-context switch per
// weight format - and those measured 2.5 ms against 0.1 ms for the work
// itself.
//
// A group's scale and min are an int8 each, scaled by a bf16 pair shared by
// the 256 values of a ggml super-block. That is not an approximation of the
// source: every type this packs from is already built that way - Q4_K and
// Q5_K carry a f16 d and dmin with 6-bit per-group scales and mins, Q6_K a
// f16 d with an int8 scale per 16 - so the super-block record reproduces the
// dequantized weight exactly, the same as the per-group bf16 pair it replaces.
//
// A pair, not a single bf16, for the shared parameter. A single bf16 is not
// enough: the parameter is shared by every value under it, so rounding it is a
// systematic error that does not average out over K, and w = q*d + m is a
// difference of two similar terms for the small weights, which amplifies it -
// bf16 parameters measure 6.7e-3 relative RMS against the ggml dequant, four
// times the 1.7e-3 floor of rounding exact weights to bf16. An f32 parameter
// fixes the accuracy but costs twice the expansion time, because AIE2P
// multiplies f32 by emulation while bf16 is native. The pair gives ~16
// mantissa bits from two native multiply-accumulates, and is now paid once per
// 256 values instead of once per group: 0.75 -> 0.594 B/val on q4g32 and
// 1.5 -> 1.19 on q8g16, which is the weight stream of every decode dispatch.
//
// One record holds one super-block, laid out as the group codes, then the int8
// scales, then the int8 mins, then dS_hi, dS_lo, mS_hi, mS_lo:
//
//   w = q * (dS * d8) + (mS * m8)
//

#include "ggml.h"

#include <cstddef>
#include <cstdint>

// Group sizes and on-wire group strides.
enum {
    XDNA_Q4G32_GROUP = 32,
    XDNA_Q8G16_GROUP = 16,
    // A record is one ggml super-block, whatever the group width.
    XDNA_SB_VALUES       = 256,
    XDNA_SB_PARAM        = 8,        // dS_hi, dS_lo, mS_hi, mS_lo
    XDNA_Q4G32_CODE      = XDNA_Q4G32_GROUP / 2,
    XDNA_Q8G16_CODE      = XDNA_Q8G16_GROUP,
    XDNA_Q4G32_SB_GROUPS = XDNA_SB_VALUES / XDNA_Q4G32_GROUP,   // 8
    XDNA_Q8G16_SB_GROUPS = XDNA_SB_VALUES / XDNA_Q8G16_GROUP,   // 16
    XDNA_Q4G32_SB_BYTES  = XDNA_Q4G32_SB_GROUPS * XDNA_Q4G32_CODE +
                           2 * XDNA_Q4G32_SB_GROUPS + XDNA_SB_PARAM,   // 152
    XDNA_Q8G16_SB_BYTES  = XDNA_Q8G16_SB_GROUPS * XDNA_Q8G16_CODE +
                           2 * XDNA_Q8G16_SB_GROUPS + XDNA_SB_PARAM,   // 296
};

// Which NPU format a ggml type repacks into.
enum xdna_wfmt {
    XDNA_WFMT_NONE = 0,
    XDNA_WFMT_Q4G32,
    XDNA_WFMT_Q8G16,
};

// The format `type` repacks into, or XDNA_WFMT_NONE when it has none.
xdna_wfmt xdna_wfmt_for(enum ggml_type type);

// Bytes one row of `k` values takes in `fmt` (0 when k is not a multiple of
// the group size).
size_t xdna_wfmt_row_bytes(xdna_wfmt fmt, int64_t k);

// Repack one ggml-quantized row of `k` values into `fmt`. `src` points at the
// row's ggml blocks, `dst` at xdna_wfmt_row_bytes() of output. Returns false
// when the type/format pair or `k` is unsupported.
bool xdna_wfmt_repack_row(enum ggml_type type, const void * src, int64_t k, void * dst);

// Repack `src` into an explicitly chosen format. Only Q4_K has a choice: it
// repacks into q4g32 by default and into q8g16 when asked, which is what lets
// the GEMV keep every type on one format.
bool xdna_wfmt_repack_row_as(enum ggml_type type, xdna_wfmt fmt, const void * src,
                             int64_t k, void * dst);

// Host reference decode of a repacked row into `k` floats. Mirrors exactly
// what the AIE dequant computes, so a host check of repack+decode against the
// ggml dequant validates the format end to end.
bool xdna_wfmt_decode_row(xdna_wfmt fmt, const void * src, int64_t k, float * dst);

// Check the repack against the ggml dequant of the same rows and fold the
// worst relative RMS into the verification table (GGML_XDNA_VERIFY). A correct
// q4g32 repack of Q4_K carries ~16 mantissa bits of the group parameters, so
// anything much above 1e-5 is a format bug.
void xdna_wfmt_selfcheck(enum ggml_type type, const void * data, int64_t k, int64_t n_rows,
                         const char * label);

// The format the decode GEMV uses for `type`. Both widths pack into the same
// tile size, so a single artifact streams either and the GEMV never switches
// hardware context.
xdna_wfmt xdna_wfmt_gemv_for(enum ggml_type type);
