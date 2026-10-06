#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>
#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "hvx-utils.h"
#include "hex-dma.h"
#include "hex-profile.h"
#include "htp-vtcm.h"

struct htp_im2col_context {
    struct htp_ops_context * octx;
    uint32_t                 npatches_per_thread;  // patches = N*OH*OW (pure-DDR kernel)

    uint32_t pe_rows_per_thread;                   // N*OH rows per worker
    uint32_t pe_src_row_bytes;                     // one output row's source: IC*KH*IW*4, rounded 256
    uint32_t pe_dst_row_bytes;                     // one output row's dst: OW*patch_stride*2, rounded 256

    // Patch-embed DMA path VTCM ping-pong.
    uint8_t * pe_vtcm_src;                         // base of the 2x src buffers region
    uint8_t * pe_vtcm_dst;                         // base of the 2x dst buffers region
    uint32_t  pe_src_size_per_thread;              // 2 * pe_src_row_bytes
    uint32_t  pe_dst_size_per_thread;              // 2 * pe_dst_row_bytes
};

// Per-op VTCM layout for the patch-embed DMA path
struct htp_im2col_vtcm_layout {
    size_t off_src;
    size_t off_dst;
    size_t src_bytes_per_thread;
    size_t dst_bytes_per_thread;
    size_t total_bytes;
};

static inline void htp_im2col_vtcm_layout_build(struct htp_im2col_vtcm_layout * L,
                                                size_t                          src_row_bytes,
                                                size_t                          dst_row_bytes,
                                                uint32_t                        n_threads) {
    L->src_bytes_per_thread = 2 * src_row_bytes;
    L->dst_bytes_per_thread = 2 * dst_row_bytes;

    L->off_src     = 0;
    L->off_dst     = L->off_src + L->src_bytes_per_thread * n_threads;
    L->total_bytes = L->off_dst + L->dst_bytes_per_thread * n_threads;
}

#define IM2COL_PATCHEMBED_BODY(FNAME, DST_CTYPE, COPY_FN, SPLAT_FN, DST_ELEM, TAG)                        \
    static void FNAME(unsigned int nth, unsigned int ith, void * data) {                                  \
        struct htp_im2col_context * ictx        = (struct htp_im2col_context *) data;                     \
        struct htp_ops_context *    octx        = ictx->octx;                                             \
        struct htp_thread_trace * restrict tr   = &octx->ctx->trace[ith];                                 \
        const struct htp_tensor * restrict src1 = octx->src[1];                                           \
        const struct htp_tensor * restrict dst  = octx->dst;                                              \
        const int32_t  s0                       = octx->op_params[0];                                     \
        const int32_t  s1                       = octx->op_params[1];                                     \
        const int32_t  p0                       = octx->op_params[2];                                     \
        const int32_t  p1                       = octx->op_params[3];                                     \
        const int32_t  d0                       = octx->op_params[4];                                     \
        const int32_t  d1                       = octx->op_params[5];                                     \
        const bool is_2D                       = octx->op_params[6] == 1;                                \
        const uint32_t N                        = src1->ne[is_2D ? 3 : 2];                                \
        const uint32_t IC                       = src1->ne[is_2D ? 2 : 1];                                \
        const uint32_t IH                       = is_2D ? src1->ne[1] : 1;                                \
        const uint32_t IW                       = src1->ne[0];                                            \
        const uint32_t KH                       = is_2D ? octx->src[0]->ne[1] : 1;                        \
        const uint32_t KW                       = octx->src[0]->ne[0];                                    \
        const uint32_t OH                       = is_2D ? dst->ne[2] : 1;                                 \
        const uint32_t OW                       = dst->ne[1];                                             \
        const uint32_t patch_stride             = IC * KH * KW;                                           \
        const float * restrict src_data         = (const float *) src1->data;                             \
        DST_CTYPE * restrict dst_data           = (DST_CTYPE *) dst->data;                                \
        const uint32_t npatches                 = N * OH * OW;                                            \
        const uint32_t patch_start              = ictx->npatches_per_thread * ith;                        \
        const uint32_t patch_end                = MIN(patch_start + ictx->npatches_per_thread, npatches); \
        if (patch_start >= patch_end) {                                                                   \
            return;                                                                                       \
        }                                                                                                 \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, patch_start);                                   \
        for (uint32_t p = patch_start; p < patch_end; p++) {                                              \
            const uint32_t iow             = p % OW;                                                      \
            const uint32_t ioh             = (p / OW) % OH;                                               \
            const uint32_t in              = p / (OW * OH);                                               \
            DST_CTYPE * restrict dst_patch = dst_data + (uint64_t) p * patch_stride;                      \
            for (uint32_t iic = 0; iic < IC; iic++) {                                                     \
                const float * restrict src_plane = (const float *) ((const uint8_t *) src_data +        \
                    (uint64_t) in * src1->nb[is_2D ? 3 : 2] + (uint64_t) iic * src1->nb[is_2D ? 2 : 1]); \
                for (uint32_t ikh = 0; ikh < KH; ikh++) {                                                 \
                    const int64_t iih            = (int64_t) ioh * s1 + (int64_t) ikh * d1 - p1;          \
                    DST_CTYPE * restrict out_run = dst_patch + iic * (KH * KW) + ikh * KW;                \
                    if (iih < 0 || iih >= (int32_t) IH) {                                                 \
                        SPLAT_FN(out_run, 0.0f, KW);                                                      \
                        continue;                                                                         \
                    }                                                                                     \
                    const int64_t iiw0             = (int64_t) iow * s0 - p0;                             \
                    if (d0 == 1) {                                                                        \
                        /* contiguous source run: [lo,hi) is in-bounds, tails are zero pad */             \
                        const int64_t lo = iiw0 < 0 ? -iiw0 : 0;                                          \
                        int64_t       hi = (int64_t) IW - iiw0;                                           \
                        if (hi > (int32_t) KW) {                                                          \
                            hi = (int32_t) KW;                                                            \
                        }                                                                                 \
                        if (hi <= lo) {                                                                   \
                            SPLAT_FN(out_run, 0.0f, KW);                                                  \
                        } else {                                                                          \
                            if (lo > 0) {                                                                 \
                                SPLAT_FN(out_run, 0.0f, (uint32_t) lo);                                   \
                            }                                                                             \
                            COPY_FN((uint8_t *) (out_run + lo),                                          \
                                    (const uint8_t *) (src_plane + iih * IW + iiw0 + lo),                \
                                    (uint32_t) (hi - lo));                                                \
                            if (hi < (int32_t) KW) {                                                      \
                                SPLAT_FN(out_run + hi, 0.0f, (KW - (uint32_t) hi));                       \
                            }                                                                             \
                        }                                                                                 \
                        continue;                                                                         \
                    }                                                                                     \
                    for (uint32_t ikw = 0; ikw < KW; ikw++) {                                             \
                        const int64_t iiw = (int64_t) iow * s0 + (int64_t) ikw * d0 - p0;                 \
                        out_run[ikw]      = (iiw < 0 || iiw >= (int32_t) IW) ?                            \
                                                (DST_CTYPE) 0.0f :                                        \
                                                (DST_CTYPE) src_plane[(uint64_t) iih * IW + iiw];         \
                    }                                                                                     \
                }                                                                                         \
            }                                                                                             \
        }                                                                                                 \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, patch_start);                                    \
    }

IM2COL_PATCHEMBED_BODY(im2col_patchembed_thread, __fp16, hvx_copy_f16_f32_uu, hvx_splat_f16_u, sizeof(__fp16), "f32-f16")
IM2COL_PATCHEMBED_BODY(im2col_patchembed_f32_thread, float, hvx_copy_f32_uu, hvx_splat_f32_u, sizeof(float), "f32-f32")

#define IM2COL_PATCHEMBED_DMA_BODY(FNAME, DST_CTYPE, COPY_FN, SPLAT_FN, DST_ELEM, TAG)                               \
    static void FNAME(unsigned int nth, unsigned int ith, void * data) {                                             \
        struct htp_im2col_context * ictx        = (struct htp_im2col_context *) data;                                \
        struct htp_ops_context *    octx        = ictx->octx;                                                        \
        struct htp_thread_trace * restrict tr   = &octx->ctx->trace[ith];                                            \
        const struct htp_tensor * restrict src1 = octx->src[1];                                                      \
        const struct htp_tensor * restrict dst  = octx->dst;                                                         \
        const uint32_t N = src1->ne[3], IC = src1->ne[2], IH = src1->ne[1], IW = src1->ne[0];                        \
        const uint32_t KH = octx->src[0]->ne[1], KW = octx->src[0]->ne[0];                                           \
        const uint32_t OH = dst->ne[2], OW = dst->ne[1];                                                             \
        const uint32_t patch_stride     = IC * KH * KW;                                                              \
        const float * restrict src_data = (const float *) src1->data;                                                \
        DST_CTYPE * restrict dst_data   = (DST_CTYPE *) dst->data;                                                   \
        dma_queue *    dmaq             = octx->ctx->dma[ith];                                                       \
        uint8_t *      src_base         = ictx->pe_vtcm_src + ith * ictx->pe_src_size_per_thread;                    \
        uint8_t *      dst_base         = ictx->pe_vtcm_dst + ith * ictx->pe_dst_size_per_thread;                    \
        float *        srcb             = (float *) src_base;                                                        \
        DST_CTYPE *    dstb             = (DST_CTYPE *) dst_base;                                                    \
        const uint32_t nrows            = N * OH;                                                                    \
        const uint32_t per_thread       = ictx->pe_rows_per_thread;                                                  \
        const uint32_t row_start        = per_thread * ith;                                                          \
        const uint32_t row_end          = MIN(row_start + per_thread, nrows);                                        \
        if (row_start >= row_end)                                                                                    \
            return;                                                                                                  \
        for (uint32_t r = row_start; r < row_end; r++) {                                                             \
            const uint32_t in  = r / OH;                                                                             \
            const uint32_t ioh = r % OH;                                                                             \
            for (uint32_t ikh = 0; ikh < KH; ikh++) {                                                                \
                int32_t iih = (int32_t) ioh * (int32_t) KH + (int32_t) ikh;                                          \
                int     ok  = (iih >= 0 && iih < (int32_t) IH);                                                      \
                for (uint32_t iic = 0; iic < IC; iic++) {                                                            \
                    float *       vdst = srcb + ((uint64_t) (iic * KH + ikh)) * IW;                                  \
                    const float * _vsrc =                                                                            \
                        ok ? (src_data + ((uint64_t) (in * IC + iic) * IH + iih) * IW) : (const float *) vdst;       \
                    dma_queue_push_ddr_to_vtcm(                                                                      \
                        dmaq, dma_make_ptr((uint8_t *) vdst, ok ? (const uint8_t *) _vsrc : (const uint8_t *) vdst), \
                        IW * sizeof(float), IW * sizeof(float), ok ? 1 : 0);                                         \
                }                                                                                                    \
            }                                                                                                        \
            for (uint32_t i = 0; i < IC * KH; i++)                                                                   \
                dma_queue_pop(dmaq);                                                                                 \
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, r);                                                    \
            for (uint32_t iow = 0; iow < OW; iow++) {                                                                \
                DST_CTYPE * dst_patch = dstb + (uint64_t) iow * patch_stride;                                        \
                for (uint32_t ikh = 0; ikh < KH; ikh++) {                                                            \
                    int32_t iih = (int32_t) ioh * (int32_t) KH + (int32_t) ikh;                                      \
                    for (uint32_t iic = 0; iic < IC; iic++) {                                                        \
                        DST_CTYPE * out_run = dst_patch + iic * (KH * KW) + ikh * KW;                                \
                        if (iih < 0 || iih >= (int32_t) IH) {                                                        \
                            SPLAT_FN(out_run, 0.0f, KW);                                                             \
                            continue;                                                                                \
                        }                                                                                            \
                        const float * src_run = srcb + ((uint64_t) (iic * KH + ikh)) * IW + (uint64_t) iow * KW;     \
                        COPY_FN((uint8_t *) out_run, (const uint8_t *) src_run, KW);                                 \
                    }                                                                                                \
                }                                                                                                    \
            }                                                                                                        \
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, r);                                                     \
            DST_CTYPE * ddr_row = dst_data + ((uint64_t) (in * OH + ioh) * OW) * patch_stride;                       \
            dma_queue_push_vtcm_to_ddr(dmaq, dma_make_ptr((uint8_t *) ddr_row, (uint8_t *) dstb),                    \
                                       OW * patch_stride * (DST_ELEM), OW * patch_stride * (DST_ELEM), 1);           \
            dma_queue_flush(dmaq);                                                                                   \
        }                                                                                                            \
    }

IM2COL_PATCHEMBED_DMA_BODY(im2col_patchembed_dma_thread,     __fp16, hvx_copy_f16_f32_uu, hvx_splat_f16_u, sizeof(__fp16), "pe-dma-f16")
IM2COL_PATCHEMBED_DMA_BODY(im2col_patchembed_dma_f32_thread, float,  hvx_copy_f32_uu,     hvx_splat_f32_u, sizeof(float),  "pe-dma-f32")

static bool im2col_use_patchembed_dma(const struct htp_ops_context * octx) {
    const int32_t s0 = octx->op_params[0], s1 = octx->op_params[1];
    const int32_t p0 = octx->op_params[2], p1 = octx->op_params[3];
    const int32_t d0 = octx->op_params[4], d1 = octx->op_params[5];
    const int     is_2D = octx->op_params[6] == 1;
    if (!is_2D) {
        return false;
    }
    // This path flattens the image. Strided channel/batch views use the DDR path.
    const struct htp_tensor * x = octx->src[1];
    if (x->nb[0] != sizeof(float) || x->nb[1] != x->ne[0] * sizeof(float) ||
        x->nb[2] != x->ne[1] * x->nb[1] || x->nb[3] != x->ne[2] * x->nb[2]) {
        return false;
    }
    if (octx->dst->type != HTP_TYPE_F16 && octx->dst->type != HTP_TYPE_F32) {
        return false;
    }
    const uint32_t KH = octx->src[0]->ne[1], KW = octx->src[0]->ne[0];
    if (s0 != (int32_t) KW || s1 != (int32_t) KH) {
        return false;  // non-overlapping
    }
    if (p0 != 0 || p1 != 0) {
        return false;  // no padding
    }
    if (d0 != 1 || d1 != 1) {
        return false;  // no dilation
    }
    return true;
}

// Sizes the per-thread 2x(src,dst) VTCM ping-pong for the patch-embed DMA path.
// Returns false if it doesn't fit the VTCM budget (caller falls back).
static bool im2col_patchembed_dma_fits(struct htp_ops_context *    octx,
                                       struct htp_im2col_context * ictx,
                                       uint32_t                    n_threads) {
    const uint32_t IC = octx->src[1]->ne[2], IW = octx->src[1]->ne[0];
    const uint32_t KH = octx->src[0]->ne[1], KW = octx->src[0]->ne[0];
    const uint32_t OW           = octx->dst->ne[1];
    const uint32_t patch_stride = IC * KH * KW;

    ictx->pe_src_row_bytes  = hex_round_up(IC * KH * IW * sizeof(float), 256);
    const uint32_t dst_elem = (octx->dst->type == HTP_TYPE_F16) ? sizeof(__fp16) : sizeof(float);
    ictx->pe_dst_row_bytes  = hex_round_up(OW * patch_stride * dst_elem, 256);

    // 2 src + 2 dst buffers per thread (ping-pong), src region first then dst.
    struct htp_im2col_vtcm_layout L;
    htp_im2col_vtcm_layout_build(&L, ictx->pe_src_row_bytes, ictx->pe_dst_row_bytes, n_threads);
    if (L.total_bytes > octx->ctx->vtcm_size) {
        return false;
    }

    uint8_t * const base        = octx->ctx->vtcm_base;
    ictx->pe_vtcm_src           = VTCM_LAYOUT_PTR(uint8_t, base, L.off_src);
    ictx->pe_vtcm_dst           = VTCM_LAYOUT_PTR(uint8_t, base, L.off_dst);
    ictx->pe_src_size_per_thread = (uint32_t) L.src_bytes_per_thread;
    ictx->pe_dst_size_per_thread = (uint32_t) L.dst_bytes_per_thread;
    return true;
}

// 1D im2col with channel-major input [IW, IC] and F16 columns [IC * KW, OW]
// (convolution layers of audio models). One output column gathers a tap from
// every channel row, so reading DDR directly touches IC rows that sit a whole
// row pitch apart. Tiles of output positions instead stage each channel's
// input span in VTCM (dma_queue_copy_rows), widen it to F16 with HVX, assemble
// each column with HVX gathers from VTCM, and DMA the finished rows out.
#define IM2COL_1D_MAX_TILE 128
#define IM2COL_1D_F32_ALIGN 32
#define IM2COL_1D_F16_ALIGN 64
#define IM2COL_1D_GATHER_MAX_BYTES 65535

// One gather vector covers 64 consecutive column elements. Its 16-bit byte
// offsets are relative to the first channel row it touches, which keeps every
// offset far below the 64 KiB limit of the halfword gather.
struct htp_im2col_1d_gather_plan {
    HVX_Vector * offsets;
    uint32_t *   row_base;
    uint32_t *   region;
};

struct htp_im2col_1d_tiled {
    struct htp_ops_context *         octx;
    struct htp_im2col_1d_gather_plan plan;
    uint32_t                         n_gather;
    uint32_t                         tile_w;
    uint32_t                         n_tiles;
    uint32_t                         tiles_per_thread;
    uint32_t                         stride_f32;
    uint32_t                         stride_f16;
    uint32_t                         plan_bytes;
    uint32_t                         bytes_per_thread;
    uint32_t                         off_f16;
    uint32_t                         off_tmp;
    uint32_t                         off_out;
};

static inline uint32_t im2col_1d_span(uint32_t tile_w, int32_t s0, int32_t d0, uint32_t KW) {
    return (tile_w - 1) * (uint32_t) s0 + (KW - 1) * (uint32_t) d0 + 1;
}

static void im2col_1d_tiled_layout(struct htp_im2col_1d_tiled * t, uint32_t tile_w) {
    const struct htp_ops_context * octx = t->octx;
    const uint32_t IC   = octx->src[1]->ne[1];
    const uint32_t KW   = octx->src[0]->ne[0];
    const uint32_t PS   = IC * KW;
    const uint32_t span = im2col_1d_span(tile_w, octx->op_params[0], octx->op_params[4], KW);

    t->n_gather         = (PS + VLEN_FP16 - 1) / VLEN_FP16;
    t->plan_bytes       = hex_round_up(t->n_gather * (VLEN + 2 * sizeof(uint32_t)), VLEN);
    t->tile_w           = tile_w;
    t->stride_f32       = hex_round_up(span, IM2COL_1D_F32_ALIGN);
    t->stride_f16       = hex_round_up(span, IM2COL_1D_F16_ALIGN);
    t->off_f16          = hex_round_up(IC * t->stride_f32 * sizeof(float), VLEN);
    t->off_tmp          = t->off_f16 + hex_round_up(IC * t->stride_f16 * sizeof(__fp16), VLEN);
    t->off_out          = t->off_tmp + t->n_gather * VLEN;
    t->bytes_per_thread = t->off_out + hex_round_up(tile_w * PS * sizeof(__fp16), VLEN);
}

static bool im2col_1d_gather_fits(const struct htp_im2col_1d_tiled * t) {
    const uint32_t KW       = t->octx->src[0]->ne[0];
    const uint32_t channels = VLEN_FP16 / KW + 2;
    return (uint64_t) channels * t->stride_f16 * sizeof(__fp16) <= IM2COL_1D_GATHER_MAX_BYTES;
}

// Largest power-of-two tile whose per-thread staging fits the VTCM budget.
static bool im2col_1d_tiled_fit(struct htp_im2col_1d_tiled * t, uint32_t n_threads) {
    for (uint32_t tile_w = IM2COL_1D_MAX_TILE; tile_w >= 1; tile_w /= 2) {
        im2col_1d_tiled_layout(t, tile_w);
        if (t->plan_bytes + (size_t) t->bytes_per_thread * n_threads <= t->octx->ctx->vtcm_size &&
            im2col_1d_gather_fits(t)) {
            return true;
        }
    }
    return false;
}

static bool im2col_use_1d_tiled(const struct htp_ops_context * octx) {
    const struct htp_tensor * x   = octx->src[1];
    const struct htp_tensor * dst = octx->dst;
    const uint32_t            PS  = x->ne[1] * octx->src[0]->ne[0];
    return octx->op_params[6] == 0 && x->ne[2] == 1 && x->ne[3] == 1 && x->nb[0] == sizeof(float) &&
           dst->type == HTP_TYPE_F16 && dst->ne[0] == PS && dst->nb[1] == PS * sizeof(__fp16) &&
           octx->op_params[0] > 0 && octx->op_params[4] > 0 && octx->op_params[2] >= 0;
}

static void im2col_1d_plan_vector(const struct htp_im2col_1d_tiled * t, uint32_t v) {
    const uint32_t KW = t->octx->src[0]->ne[0];
    const uint32_t PS = t->octx->dst->ne[0];
    const uint32_t d0 = t->octx->op_params[4];

    int16_t        lanes[VLEN_FP16] __attribute__((aligned(VLEN)));
    const uint32_t first   = v * VLEN_FP16;
    const uint32_t ic_base = first / KW;
    uint32_t       max_off = 0;
    for (uint32_t lane = 0; lane < VLEN_FP16; lane++) {
        const uint32_t j   = first + lane < PS ? first + lane : first;
        const uint32_t off = ((j / KW - ic_base) * t->stride_f16 + (j % KW) * d0) * sizeof(__fp16);
        lanes[lane]        = (int16_t) off;
        max_off            = MAX(max_off, off);
    }
    t->plan.offsets[v]  = *(const HVX_Vector *) lanes;
    t->plan.row_base[v] = ic_base * t->stride_f16;
    t->plan.region[v]   = max_off + sizeof(__fp16) - 1;
}

static void im2col_1d_plan(struct htp_im2col_1d_tiled * t) {
    uint8_t * base     = t->octx->ctx->vtcm_base;
    t->plan.offsets    = (HVX_Vector *) base;
    t->plan.row_base   = (uint32_t *) (base + t->n_gather * VLEN);
    t->plan.region     = t->plan.row_base + t->n_gather;
    for (uint32_t v = 0; v < t->n_gather; v++) {
        im2col_1d_plan_vector(t, v);
    }
}

// Copies the in-bounds part of every channel's span; positions left of the
// signal or past its end stay zero, which is the convolution's zero padding.
static void im2col_1d_stage(const struct htp_im2col_1d_tiled * t, dma_queue * q, float * in32, int64_t iw0,
                            uint32_t span) {
    const struct htp_tensor * x  = t->octx->src[1];
    const uint32_t            IC = x->ne[1];
    const int64_t             IW = x->ne[0];

    const int64_t lo = iw0 < 0 ? -iw0 : 0;
    const int64_t hi = MIN((int64_t) span, IW - iw0);
    if (lo > 0 || hi < (int64_t) span) {
        hvx_splat_f32_a(in32, 0.0f, IC * t->stride_f32);
    }
    if (hi <= lo) {
        return;
    }
    const uint8_t * src = (const uint8_t *) x->data + (iw0 + lo) * sizeof(float);
    dma_queue_copy_rows(q, dma_make_ptr(in32 + lo, src), t->stride_f32 * sizeof(float), x->nb[1],
                        (size_t) (hi - lo) * sizeof(float), IC);
}

static void im2col_1d_widen(const struct htp_im2col_1d_tiled * t, __fp16 * in16, const float * in32, uint32_t span) {
    const uint32_t IC = t->octx->src[1]->ne[1];
    for (uint32_t ic = 0; ic < IC; ic++) {
        hvx_copy_f16_f32_aa((uint8_t *) (in16 + ic * t->stride_f16), (const uint8_t *) (in32 + ic * t->stride_f32), span);
    }
}

static void im2col_1d_gather_column(const struct htp_im2col_1d_tiled * t, HVX_Vector * tmp, uint8_t * out,
                                    const __fp16 * in16) {
    const uint32_t PS = t->octx->dst->ne[0];
    for (uint32_t v = 0; v < t->n_gather; v++) {
        Q6_vgather_ARMVh(&tmp[v], (size_t) (in16 + t->plan.row_base[v]), t->plan.region[v], t->plan.offsets[v]);
    }
    for (uint32_t v = 0; v < t->n_gather; v++) {
        const uint32_t n = MIN(VLEN_FP16, PS - v * VLEN_FP16);
        hvx_vec_store_u(out + v * VLEN, n * sizeof(__fp16), tmp[v]);
    }
}

static void im2col_1d_gather_tile(const struct htp_im2col_1d_tiled * t, HVX_Vector * tmp, __fp16 * out,
                                  const __fp16 * in16, uint32_t tw) {
    const uint32_t PS = t->octx->dst->ne[0];
    const uint32_t s0 = t->octx->op_params[0];
    for (uint32_t c = 0; c < tw; c++) {
        im2col_1d_gather_column(t, tmp, (uint8_t *) (out + c * PS), in16 + c * s0);
    }
}

static void im2col_1d_tile(const struct htp_im2col_1d_tiled * t, unsigned int ith, uint32_t tile) {
    const struct htp_ops_context * octx = t->octx;
    const struct htp_tensor *      dst  = octx->dst;
    const uint32_t                 PS   = dst->ne[0];
    const uint32_t                 OW   = dst->ne[1];
    const int32_t                  s0   = octx->op_params[0];
    const int32_t                  p0   = octx->op_params[2];

    uint8_t *    base = octx->ctx->vtcm_base + t->plan_bytes + (size_t) ith * t->bytes_per_thread;
    float *      in32 = (float *) base;
    __fp16 *     in16 = (__fp16 *) (base + t->off_f16);
    HVX_Vector * tmp  = (HVX_Vector *) (base + t->off_tmp);
    __fp16 *     out  = (__fp16 *) (base + t->off_out);

    const uint32_t ow0  = tile * t->tile_w;
    const uint32_t tw   = MIN(t->tile_w, OW - ow0);
    const uint32_t span = im2col_1d_span(tw, s0, octx->op_params[4], octx->src[0]->ne[0]);
    dma_queue *    q    = octx->ctx->dma[ith];

    im2col_1d_stage(t, q, in32, (int64_t) ow0 * s0 - p0, span);
    im2col_1d_widen(t, in16, in32, span);
    im2col_1d_gather_tile(t, tmp, out, in16, tw);

    uint8_t * dst_rows = (uint8_t *) dst->data + (size_t) ow0 * dst->nb[1];
    dma_queue_copy_rows(q, dma_make_ptr(dst_rows, out), PS * sizeof(__fp16), PS * sizeof(__fp16), PS * sizeof(__fp16), tw);
}

static void im2col_1d_tiled_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_im2col_1d_tiled * t = (const struct htp_im2col_1d_tiled *) data;
    struct htp_thread_trace * restrict tr = &t->octx->ctx->trace[ith];

    const uint32_t first = t->tiles_per_thread * ith;
    const uint32_t last  = MIN(first + t->tiles_per_thread, t->n_tiles);
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, first);
    for (uint32_t tile = first; tile < last; tile++) {
        im2col_1d_tile(t, ith, tile);
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, first);
}

static bool im2col_run_1d_tiled(struct htp_ops_context * octx) {
    if (!im2col_use_1d_tiled(octx)) {
        return false;
    }
    struct htp_im2col_1d_tiled t = { .octx = octx };
    const uint32_t             OW = octx->dst->ne[1];
    if (!im2col_1d_tiled_fit(&t, octx->n_threads)) {
        return false;
    }
    im2col_1d_plan(&t);
    t.n_tiles                 = (OW + t.tile_w - 1) / t.tile_w;
    const uint32_t n_threads  = MIN(octx->n_threads, t.n_tiles);
    t.tiles_per_thread        = (t.n_tiles + n_threads - 1) / n_threads;
    work_queue_run(octx->ctx->work_queue, im2col_1d_tiled_thread, &t, n_threads);
    return true;
}

// A 1D im2col with a single tap, unit stride and no padding is a transpose of
// the [channels, length] input into [length, channels] columns.
static bool im2col_is_pointwise_f32(const struct htp_ops_context * octx) {
    const struct htp_tensor * x   = octx->src[1];
    const struct htp_tensor * dst = octx->dst;
    return octx->op_params[6] == 0 && octx->src[0]->ne[0] == 1 && octx->op_params[0] == 1 && octx->op_params[2] == 0 &&
           dst->type == HTP_TYPE_F32 && x->nb[0] == sizeof(float) && x->ne[3] == 1 && dst->ne[0] == x->ne[1] &&
           dst->ne[1] == x->ne[0] && dst->nb[0] == sizeof(float) && dst->nb[1] == dst->ne[0] * sizeof(float);
}

static bool im2col_run_pointwise(struct htp_ops_context * octx) {
    if (!im2col_is_pointwise_f32(octx)) {
        return false;
    }
    const struct htp_tensor *      x   = octx->src[1];
    const struct htp_tensor *      dst = octx->dst;
    const struct htp_transpose_f32 job = {
        .octx           = octx,
        .src            = (const uint8_t *) x->data,
        .dst            = (uint8_t *) dst->data,
        .rows           = x->ne[1],
        .cols           = x->ne[0],
        .src_row_stride = x->nb[1],
        .dst_row_stride = dst->nb[1],
        .batch2         = x->ne[2],
        .batch3         = 1,
        .src_stride2    = x->nb[2],
        .src_stride3    = 0,
        .dst_stride2    = dst->nb[2],
        .dst_stride3    = 0,
    };
    return htp_transpose_f32(&job);
}

int op_im2col(struct htp_ops_context * octx) {
    const struct htp_tensor * src1 = octx->src[1];
    const struct htp_tensor * dst  = octx->dst;

    if (src1->type != HTP_TYPE_F32 || (dst->type != HTP_TYPE_F16 && dst->type != HTP_TYPE_F32)) {
        FARF(ERROR, "im2col: only (F32 image -> F16/F32 columns) supported");
        return HTP_STATUS_NO_SUPPORT;
    }

    const bool is_2D        = octx->op_params[6] == 1;
    const uint32_t N         = src1->ne[is_2D ? 3 : 2];
    const uint32_t OH        = is_2D ? dst->ne[2] : 1;
    const uint32_t OW        = dst->ne[1];
    const uint32_t npatches  = N * OH * OW;
    const uint32_t n_threads = MIN(octx->n_threads, npatches);

    if ((octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) || n_threads == 0) {
        return HTP_STATUS_OK;
    }

    struct htp_im2col_context ictx = { 0 };
    ictx.octx                      = octx;
    ictx.npatches_per_thread       = (npatches + n_threads - 1) / n_threads;

    if (im2col_run_pointwise(octx) || im2col_run_1d_tiled(octx)) {
        return HTP_STATUS_OK;
    }

    // Clean non-overlapping patch-embed -> DMA kernel (if it fits VTCM);
    // everything else (padding/dilation/stride edges) -> pure-DDR kernel.
    if (im2col_use_patchembed_dma(octx)) {
        const uint32_t nrows = N * OH;
        const uint32_t pth   = MIN(octx->n_threads, nrows);
        if (pth > 0 && im2col_patchembed_dma_fits(octx, &ictx, pth)) {
            ictx.pe_rows_per_thread = (nrows + pth - 1) / pth;
            if (dst->type == HTP_TYPE_F16) {
                work_queue_run(octx->ctx->work_queue, im2col_patchembed_dma_thread, &ictx, pth);
            } else {
                work_queue_run(octx->ctx->work_queue, im2col_patchembed_dma_f32_thread, &ictx, pth);
            }
            return HTP_STATUS_OK;
        }
        // else: doesn't fit -> fall through to the pure-DDR kernel below.
    }

    if (dst->type == HTP_TYPE_F16) {
        work_queue_run(octx->ctx->work_queue, im2col_patchembed_thread, &ictx, n_threads);
    } else {
        work_queue_run(octx->ctx->work_queue, im2col_patchembed_f32_thread, &ictx, n_threads);
    }
    return HTP_STATUS_OK;
}
