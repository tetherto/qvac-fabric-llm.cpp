#pragma once

// Prefill attention as a mode of the merged prefill design (kernels/pfx.py),
// which GGML_XDNA_PFX=1 loads as pfx_c8 in pgemm_c8's place: the attention
// of attn-mm.cc on the GEMM's own hardware context, so a layer's projections
// and its attention run without a context switch between them. Each pass is
// one KV head for every pair of the six attention columns: its K and V^T
// tiles on pgemm's A broadcast, each core's Q on the W path under its own
// packet id, O out through pgemm's C join. The shapes and masks are
// attn_mm's (xdna-attn-mm.cpp, which hands the node over once its mask is
// read).

#include "xdna-pfx-layout.h"

#include <cstdint>

struct ggml_tensor;
struct xdna_kernel_pool;

// True when the prefill GEMM runs on the merged design.
bool xdna_pfx_attn_active(void);

// One attention node: f32 Q [D, n_tokens, n_head], f16 K and V [D, n_kv, 2],
// out f32 [D, n_head, n_tokens]; lo and hi each row's first and last visible
// key (n_tokens of each), q_scale the scale the host puts into Q.
struct xdna_pfx_attn_job {
    const ggml_tensor * q        = nullptr;
    const ggml_tensor * k        = nullptr;
    const ggml_tensor * v        = nullptr;
    ggml_tensor *       out      = nullptr;
    const int32_t *     lo       = nullptr;
    const int32_t *     hi       = nullptr;
    int64_t             n_tokens = 0;
    int64_t             n_kv     = 0;
    int                 hg       = 0;  // query heads a KV head
    float               q_scale  = 0.0f;
};

// Run it to completion and write out: xdna_pfx_attn_prepare, _pack, _run_io, _release.
bool xdna_pfx_attn_run(xdna_kernel_pool * pool, const xdna_pfx_attn_job & job);

// The objects the host owns, into `parent` laid out as xdna_pfx_attn_geometry: both A streams' headers, every W
// object's trailer, and zero Q in the W objects whose positions start at or past the last tile's end.
void xdna_pfx_attn_host_objects(const xdna_pfx_attn_job & job, uint8_t * parent);

// One node's objects from prepare to run, so that another command (the row command, xdna-pfx-row.h) can write Q, K
// and V^T into them on the array.
struct xdna_pfx_attn_io;

// Plan the passes, take the buffers and write the host's objects; job's q, k and v are not read. Null on failure.
xdna_pfx_attn_io *        xdna_pfx_attn_prepare(xdna_kernel_pool * pool, const xdna_pfx_attn_job & job);
const xdna_pfx_attn_dst & xdna_pfx_attn_dst_of(const xdna_pfx_attn_io * io);
// The host path's Q, K and V^T objects from job's q, k and v.
bool                      xdna_pfx_attn_pack(xdna_pfx_attn_io * io);
// Run on io's objects and write job.out.
bool                      xdna_pfx_attn_run_io(xdna_kernel_pool * pool, xdna_pfx_attn_io * io);
void                      xdna_pfx_attn_release(xdna_kernel_pool * pool, xdna_pfx_attn_io * io);
