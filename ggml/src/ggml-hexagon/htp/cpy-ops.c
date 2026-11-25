#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>
#include <qurt_memory.h>

#include <math.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "cpy-ops.h"
#include "dma-copy.h"
#include "htp-ctx.h"
#include "htp-fence.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "hvx-utils.h"
#include "gather-rows.h"

struct htp_copy_context {
    struct htp_ops_context *              octx;
    const struct htp_copy_kernel_params * kparams;

    uint32_t row_start;
    uint32_t nrows;
    uint32_t src0_nrows_per_thread;

    uint32_t elem_start;
    uint32_t nelem;
    uint32_t elem_per_thread;

    struct htp_copy_convert_vtcm_layout layout;
    uint8_t * vtcm_src0;
    uint8_t * vtcm_dst;
};

#define cpy_preamble                              \
    const struct htp_tensor *src0 = octx->src[0]; \
    const struct htp_tensor *dst  = octx->dst;    \
                                                  \
    const uint32_t ne00 = src0->ne[0];            \
    const uint32_t ne01 = src0->ne[1];            \
    const uint32_t ne02 = src0->ne[2];            \
    const uint32_t ne03 = src0->ne[3];            \
                                                  \
    const uint32_t nb00 = src0->nb[0];            \
    const uint32_t nb01 = src0->nb[1];            \
    const uint32_t nb02 = src0->nb[2];            \
    const uint32_t nb03 = src0->nb[3];            \
                                                  \
    const uint32_t  ne0 = dst->ne[0];             \
    const uint32_t  ne1 = dst->ne[1];             \
    const uint32_t  ne2 = dst->ne[2];             \
    const uint32_t  ne3 = dst->ne[3];             \
                                                  \
    const uint32_t  nb0 = dst->nb[0];             \
    const uint32_t  nb1 = dst->nb[1];             \
    const uint32_t  nb2 = dst->nb[2];             \
    const uint32_t  nb3 = dst->nb[3];

#define DEFINE_CPY_RESHAPE(NAME, ELEM_TYPE, ELEM_SIZE)                                                \
static void cpy_thread_##NAME##_reshape(unsigned int nth, unsigned int ith, void * data) {            \
    struct htp_copy_context * ct = (struct htp_copy_context *) data;                                  \
    struct htp_ops_context * octx = ct->octx;                                                         \
    cpy_preamble;                                                                                     \
    const uint32_t th_nelem = ct->elem_per_thread;                                                    \
    const uint32_t th_start = ct->elem_start + ith * th_nelem;                                        \
    const uint32_t th_end   = MIN(th_start + th_nelem, ct->elem_start + ct->nelem);                   \
    if (th_start >= th_end) return;                                                                   \
                                                                                                      \
    dma_queue * dma_q = octx->ctx->dma[ith];                                                          \
                                                                                                      \
    const uint32_t ne01_ne00      = ne01 * ne00;                                                      \
    const uint32_t ne02_ne01_ne00 = ne02 * ne01_ne00;                                                 \
    const uint32_t ne1_ne0        = ne1 * ne0;                                                        \
    const uint32_t ne2_ne1_ne0    = ne2 * ne1_ne0;                                                    \
                                                                                                      \
    const struct htp_copy_reshape_params * rsh = &ct->kparams->u.reshape;                             \
    uint32_t e = th_start;                                                                            \
    uint32_t i13 = fastdiv(e, &rsh->div_ne2_ne1_ne0);                                                 \
    uint32_t rem = e - i13 * ne2_ne1_ne0;                                                             \
    uint32_t i12 = fastdiv(rem, &rsh->div_ne1_ne0);                                                   \
    uint32_t rem2 = rem - i12 * ne1_ne0;                                                              \
    uint32_t i11 = fastdiv(rem2, &rsh->div_ne0);                                                      \
    uint32_t i10 = rem2 - i11 * ne0;                                                                  \
                                                                                                      \
    uint32_t i03 = fastdiv(e, &rsh->div_ne02_ne01_ne00);                                              \
    uint32_t rem_s = e - i03 * ne02_ne01_ne00;                                                        \
    uint32_t i02 = fastdiv(rem_s, &rsh->div_ne01_ne00);                                               \
    uint32_t rem2_s = rem_s - i02 * ne01_ne00;                                                        \
    uint32_t i01 = fastdiv(rem2_s, &rsh->div_ne00);                                                   \
    uint32_t i00 = rem2_s - i01 * ne00;                                                               \
                                                                                                      \
    dma_addr_t dst_addr  = dst->data  + i10*nb0  + i11*nb1  + i12*nb2  + i13*nb3;                     \
    dma_addr_t src0_addr = src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03;                    \
                                                                                                      \
    const bool rows_contig = (nb00 == ELEM_SIZE) && (nb0 == ELEM_SIZE);                               \
                                                                                                      \
    while (e < th_end) {                                                                              \
        const uint32_t run = MIN(MIN(ne00 - i00, ne0 - i10), th_end - e);                             \
        if (rows_contig) {                                                                            \
            dma_cpy_sametype_reshape_contig(dma_q, dst_addr, src0_addr, run * ELEM_SIZE);             \
        } else {                                                                                      \
            dma_cpy_push_2d_chunked(dma_q, dst_addr, src0_addr, nb0, nb00, ELEM_SIZE, run);           \
        }                                                                                             \
        e += run;                                                                                     \
                                                                                                      \
        dst_addr += run * nb0;                                                                        \
        i10      += run;                                                                              \
        if (i10 == ne0) {                                                                             \
            i10 = 0;                                                                                  \
            if (++i11 == ne1) {                                                                       \
                i11 = 0;                                                                              \
                if (++i12 == ne2) {                                                                   \
                    i12 = 0;                                                                          \
                    i13++;                                                                            \
                }                                                                                     \
            }                                                                                         \
            dst_addr = dst->data + i11*nb1 + i12*nb2 + i13*nb3;                                       \
        }                                                                                             \
                                                                                                      \
        src0_addr += run * nb00;                                                                      \
        i00       += run;                                                                             \
        if (i00 == ne00) {                                                                            \
            i00 = 0;                                                                                  \
            if (++i01 == ne01) {                                                                      \
                i01 = 0;                                                                              \
                if (++i02 == ne02) {                                                                  \
                    i02 = 0;                                                                          \
                    i03++;                                                                            \
                }                                                                                     \
            }                                                                                         \
            src0_addr = src0->data + i01*nb01 + i02*nb02 + i03*nb03;                                  \
        }                                                                                             \
    }                                                                                                 \
    dma_queue_flush(dma_q);                                                                           \
}

DEFINE_CPY_RESHAPE(f32,  float, 4)
DEFINE_CPY_RESHAPE(f16, __fp16, 2)
DEFINE_CPY_RESHAPE(i32, int32_t, 4)

struct cpy_row_cursor {
    uint32_t r;
    uint32_t i01;
    uint32_t i02;
    uint32_t i03;
};

static inline void cpy_cursor_advance(struct cpy_row_cursor * c, uint32_t n, uint32_t ne01, uint32_t ne02) {
    c->r   += n;
    c->i01 += n;
    if (c->i01 == ne01) {
        c->i01 = 0;
        if (++c->i02 == ne02) {
            c->i02 = 0;
            c->i03++;
        }
    }
}

#define DEFINE_CPY_CONVERT_SAMESHAPE(NAME, CONV_FUNC)                                        \
static void cpy_thread_##NAME##_sameshape(unsigned int nth, unsigned int ith, void * data) { \
    struct htp_copy_context * ct = (struct htp_copy_context *) data;                         \
    struct htp_ops_context * octx = ct->octx;                                                \
    cpy_preamble;                                                                            \
                                                                                             \
    const uint32_t dr  = ct->src0_nrows_per_thread;                                          \
    const uint32_t ir0 = ct->row_start + dr * ith;                                           \
    const uint32_t ir1 = MIN(ir0 + dr, ct->row_start + ct->nrows);                           \
    if (ir0 >= ir1) return;                                                                  \
                                                                                             \
    dma_queue * dma_q = octx->ctx->dma[ith];                                                 \
    struct htp_thread_trace * tr = &octx->ctx->trace[ith];                                   \
                                                                                             \
    const struct htp_copy_convert_params * cvt = &ct->kparams->u.convert;                    \
    const struct htp_copy_convert_vtcm_layout * L = &ct->layout;                             \
    const uint32_t src0_buf_size   = L->src0_buf_size;                                       \
    const uint32_t dst_buf_size    = L->dst_buf_size;                                        \
    const uint32_t src0_row_stride = L->src0_row_stride;                                     \
    const uint32_t dst_row_stride  = L->dst_row_stride;                                      \
    const uint32_t blk_rows        = L->blk_rows;                                            \
    uint8_t * vtcm_src0_base = ct->vtcm_src0 + ith * L->spad0_size_per_thread;               \
    uint8_t * vtcm_dst_base  = ct->vtcm_dst  + ith * L->spad1_size_per_thread;               \
    const uint32_t src0_row_size = ne00 * ct->kparams->src0_type_size;                       \
    const uint32_t dst_row_size  = ne00 * ct->kparams->dst_type_size;                        \
                                                                                             \
    const uint32_t ne02_ne01 = ne02 * ne01;                                                  \
    uint32_t i03 = fastdiv(ir0, &cvt->div_ne02_ne01);                                        \
    uint32_t rem = ir0 - i03 * ne02_ne01;                                                    \
    uint32_t i02 = fastdiv(rem, &cvt->div_ne01);                                             \
    uint32_t i01 = rem - i02 * ne01;                                                         \
                                                                                             \
    struct cpy_row_cursor f = { ir0, i01, i02, i03 };                                        \
    struct cpy_row_cursor c = f;                                                             \
                                                                                             \
    for (uint32_t b = 0; b < 2 && f.r < ir1; ++b) {                                          \
        uint8_t * src_spad = vtcm_src0_base + b * src0_buf_size;                             \
        uint8_t * dst_spad = vtcm_dst_base  + b * dst_buf_size;                              \
        const uint32_t nr = MIN(blk_rows, MIN(ne01 - f.i01, ir1 - f.r));                     \
        dma_addr_t f_src0_addr = src0->data + f.i01*nb01 + f.i02*nb02 + f.i03*nb03;          \
        dma_queue_push(dma_q, dma_make_data(dst->data, dst_spad),                            \
                       dst_row_size, dst_buf_size, dst_row_size, 0);                         \
        dma_queue_push(dma_q, dma_make_data(src_spad, f_src0_addr),                          \
                       src0_row_stride, nb01, src0_row_size, nr);                            \
        cpy_cursor_advance(&f, nr, ne01, ne02);                                              \
    }                                                                                        \
                                                                                             \
    for (uint32_t blk = 0; c.r < ir1; ++blk) {                                               \
        const uint32_t nr = MIN(blk_rows, MIN(ne01 - c.i01, ir1 - c.r));                     \
        uint8_t * dst_spad = (uint8_t *) (uintptr_t) dma_queue_pop(dma_q).src;               \
        uint8_t * src_spad = (uint8_t *) (uintptr_t) dma_queue_pop(dma_q).dst;               \
                                                                                             \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) blk);                   \
        for (uint32_t r = 0; r < nr; ++r) {                                                  \
            CONV_FUNC(dst_spad + r * dst_row_stride, src_spad + r * src0_row_stride, ne00);  \
        }                                                                                    \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) blk);                    \
                                                                                             \
        dma_addr_t c_dst_addr = dst->data + c.i01*nb1 + c.i02*nb2 + c.i03*nb3;               \
        dma_queue_push(dma_q, dma_make_data(c_dst_addr, dst_spad),                           \
                       nb1, dst_row_stride, dst_row_size, nr);                               \
        cpy_cursor_advance(&c, nr, ne01, ne02);                                              \
                                                                                             \
        if (f.r < ir1) {                                                                     \
            const uint32_t fnr = MIN(blk_rows, MIN(ne01 - f.i01, ir1 - f.r));                \
            dma_addr_t f_src0_addr = src0->data + f.i01*nb01 + f.i02*nb02 + f.i03*nb03;      \
            dma_queue_push(dma_q, dma_make_data(src_spad, f_src0_addr),                      \
                           src0_row_stride, nb01, src0_row_size, fnr);                       \
            cpy_cursor_advance(&f, fnr, ne01, ne02);                                         \
        }                                                                                    \
    }                                                                                        \
    dma_queue_flush(dma_q);                                                                  \
}

DEFINE_CPY_CONVERT_SAMESHAPE(f16_f32, hvx_copy_f16_f32_aa)
DEFINE_CPY_CONVERT_SAMESHAPE(f32_f16, hvx_copy_f32_f16_aa)
DEFINE_CPY_CONVERT_SAMESHAPE(i32_f32, hvx_copy_i32_f32_aa)
DEFINE_CPY_CONVERT_SAMESHAPE(f32_i32, hvx_copy_f32_i32_aa)

// Element-wise conversion over an arbitrary stride quadruple on both sides, for
// same-extent sources whose rows are not packed (permuted or transposed views):
// the row-based conversion kernels above need packed inner rows. Scalar on
// purpose; the speech encoders emit a handful of these per layer.
#define DEFINE_CPY_CONVERT_STRIDED(NAME, SRC_TYPE, DST_TYPE)                                          \
static void cpy_thread_##NAME##_strided(unsigned int nth, unsigned int ith, void * data) {          \
    struct htp_copy_context * ct = (struct htp_copy_context *) data;                                \
    struct htp_ops_context * octx = ct->octx;                                                       \
    cpy_preamble;                                                                                   \
                                                                                                    \
    const uint32_t dr  = ct->src0_nrows_per_thread;                                                 \
    const uint32_t ir0 = ct->row_start + dr * ith;                                                  \
    const uint32_t ir1 = MIN(ir0 + dr, ct->row_start + ct->nrows);                                  \
    if (ir0 >= ir1) return;                                                                         \
                                                                                                    \
    const struct htp_copy_strided_params * sp = &ct->kparams->u.strided;                            \
    const uint8_t * src_base = (const uint8_t *) (uintptr_t) src0->data;                            \
    uint8_t *       dst_base = (uint8_t *) (uintptr_t) dst->data;                                   \
                                                                                                    \
    for (uint32_t r = ir0; r < ir1; r++) {                                                          \
        const uint32_t i03 = fastdiv(r, &sp->div_ne02_ne01);                                        \
        const uint32_t rem = r - i03 * ne02 * ne01;                                                 \
        const uint32_t i02 = fastdiv(rem, &sp->div_ne01);                                           \
        const uint32_t i01 = rem - i02 * ne01;                                                      \
        const uint8_t * src_row = src_base + i01 * nb01 + i02 * nb02 + i03 * nb03;                  \
        uint8_t *       dst_row = dst_base + i01 * nb1  + i02 * nb2  + i03 * nb3;                   \
        for (uint32_t i00 = 0; i00 < ne00; i00++) {                                                 \
            SRC_TYPE x;                                                                             \
            memcpy(&x, src_row + i00 * nb00, sizeof(x));                                            \
            const DST_TYPE y = (DST_TYPE) x;                                                        \
            memcpy(dst_row + i00 * nb0, &y, sizeof(y));                                             \
        }                                                                                           \
    }                                                                                               \
}

DEFINE_CPY_CONVERT_STRIDED(f32_f16, float, __fp16)
DEFINE_CPY_CONVERT_STRIDED(f16_f32, __fp16, float)

struct cpy_transpose_job {
    const uint8_t * src;
    uint8_t *       dst;
    uint32_t        n_rows;
    uint32_t        n_cols;
    uint32_t        n_mats;
    uint32_t        n_strips;
    uint32_t        strips_per_thread;
    bool            strips_along_rows;
};

struct cpy_transpose_rect {
    uint32_t row0;
    uint32_t row1;
    uint32_t col0;
    uint32_t col1;
};

static inline void cpy_transpose_shuffle_stage(HVX_Vector * out, const HVX_Vector * in) {
    const uint32_t half = HTP_COPY_TRANSPOSE_TILE / 2;
    for (uint32_t i = 0; i < half; i++) {
        const HVX_VectorPair pair = Q6_W_vshuff_VVR(in[i + half], in[i], -(int) sizeof(float));
        out[2 * i]     = Q6_V_lo_W(pair);
        out[2 * i + 1] = Q6_V_hi_W(pair);
    }
}

static inline void cpy_transpose_tile_load(HVX_Vector * rows, const uint8_t * src, uint32_t src_stride) {
    for (uint32_t i = 0; i < HTP_COPY_TRANSPOSE_TILE; i++) {
        rows[i] = hvx_vmemu(src + i * src_stride);
    }
}

static inline void cpy_transpose_tile_store(uint8_t * dst, uint32_t dst_stride, const HVX_Vector * rows) {
    for (uint32_t i = 0; i < HTP_COPY_TRANSPOSE_TILE; i++) {
        hvx_vmemu(dst + i * dst_stride) = rows[i];
    }
}

static inline void cpy_transpose_tile(uint8_t * dst, uint32_t dst_stride, const uint8_t * src, uint32_t src_stride) {
    HVX_Vector a[HTP_COPY_TRANSPOSE_TILE];
    HVX_Vector b[HTP_COPY_TRANSPOSE_TILE];
    cpy_transpose_tile_load(a, src, src_stride);
    cpy_transpose_shuffle_stage(b, a);
    cpy_transpose_shuffle_stage(a, b);
    cpy_transpose_shuffle_stage(b, a);
    cpy_transpose_shuffle_stage(a, b);
    cpy_transpose_shuffle_stage(b, a);
    cpy_transpose_tile_store(dst, dst_stride, b);
}

static void cpy_transpose_rect_scalar(const struct cpy_transpose_job * job, const uint8_t * src, uint8_t * dst,
                                      const struct cpy_transpose_rect * rect) {
    const float * s = (const float *) src;
    float *       d = (float *) dst;
    for (uint32_t row = rect->row0; row < rect->row1; row++) {
        for (uint32_t col = rect->col0; col < rect->col1; col++) {
            d[col * job->n_rows + row] = s[row * job->n_cols + col];
        }
    }
}

static inline bool cpy_transpose_rect_is_tile(const struct cpy_transpose_rect * rect) {
    return rect->row1 - rect->row0 == HTP_COPY_TRANSPOSE_TILE && rect->col1 - rect->col0 == HTP_COPY_TRANSPOSE_TILE;
}

static inline void cpy_transpose_rect(const struct cpy_transpose_job * job, const uint8_t * src, uint8_t * dst,
                                      const struct cpy_transpose_rect * rect) {
    if (!cpy_transpose_rect_is_tile(rect)) {
        cpy_transpose_rect_scalar(job, src, dst, rect);
        return;
    }
    const uint32_t src_stride = job->n_cols * sizeof(float);
    const uint32_t dst_stride = job->n_rows * sizeof(float);
    cpy_transpose_tile(dst + (rect->col0 * job->n_rows + rect->row0) * sizeof(float), dst_stride,
                       src + (rect->row0 * job->n_cols + rect->col0) * sizeof(float), src_stride);
}

static void cpy_transpose_strip(const struct cpy_transpose_job * job, const uint8_t * src, uint8_t * dst, uint32_t strip) {
    const uint32_t tile  = HTP_COPY_TRANSPOSE_TILE;
    const uint32_t along = job->strips_along_rows ? job->n_rows : job->n_cols;
    const uint32_t cross = job->strips_along_rows ? job->n_cols : job->n_rows;
    const uint32_t lo    = strip * tile;
    const uint32_t hi    = MIN(lo + tile, along);
    for (uint32_t c0 = 0; c0 < cross; c0 += tile) {
        const uint32_t c1 = MIN(c0 + tile, cross);
        const struct cpy_transpose_rect rect = job->strips_along_rows
            ? (struct cpy_transpose_rect) { lo, hi, c0, c1 }
            : (struct cpy_transpose_rect) { c0, c1, lo, hi };
        cpy_transpose_rect(job, src, dst, &rect);
    }
}

static void cpy_thread_transpose(unsigned int nth, unsigned int ith, void * data) {
    const struct cpy_transpose_job * job = (const struct cpy_transpose_job *) data;
    const uint32_t total     = job->n_mats * job->n_strips;
    const uint32_t unit0     = ith * job->strips_per_thread;
    const uint32_t unit1     = MIN(unit0 + job->strips_per_thread, total);
    const uint32_t mat_bytes = job->n_rows * job->n_cols * sizeof(float);
    for (uint32_t unit = unit0; unit < unit1; unit++) {
        const uint32_t mat = unit / job->n_strips;
        cpy_transpose_strip(job, job->src + mat * mat_bytes, job->dst + mat * mat_bytes, unit - mat * job->n_strips);
    }
}

static int cpy_transpose(struct htp_ops_context * octx, const struct htp_copy_kernel_params * kparams) {
    if (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const struct htp_copy_transpose_params * tp = &kparams->transpose;
    const uint32_t n_threads = octx->n_threads;

    struct cpy_transpose_job job;
    job.src               = (const uint8_t *) (uintptr_t) octx->src[0]->data;
    job.dst               = (uint8_t *) (uintptr_t) octx->dst->data;
    job.n_rows            = tp->n_rows;
    job.n_cols            = tp->n_cols;
    job.n_mats            = tp->n_mats;
    job.strips_along_rows = tp->n_rows >= tp->n_cols;
    job.n_strips          = ((job.strips_along_rows ? tp->n_rows : tp->n_cols) + HTP_COPY_TRANSPOSE_TILE - 1) / HTP_COPY_TRANSPOSE_TILE;
    job.strips_per_thread = fastdiv(job.n_mats * job.n_strips + n_threads - 1, &octx->n_threads_div);

    work_queue_run(octx->ctx->work_queue, cpy_thread_transpose, &job, n_threads);
    return HTP_STATUS_OK;
}

static int cpy_scalar(struct htp_ops_context * octx, const struct htp_copy_kernel_params * kparams) {
    if (octx->ctx->mdev.count > 1 && octx->ctx->mdev.idx > 0) {
        return HTP_STATUS_OK;
    }

    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    if (src0->type == dst->type) {
        dma_cpy_sametype_reshape_contig(octx->ctx->dma[0], dst->data, src0->data, kparams->src0_type_size);
        dma_queue_flush(octx->ctx->dma[0]);
        return HTP_STATUS_OK;
    }

    dma_queue * dma_q = octx->ctx->dma[0];
    uint8_t * s_ptr = VTCM_LAYOUT_PTR(uint8_t, octx->ctx->vtcm_base, 0);
    uint8_t * d_ptr = VTCM_LAYOUT_PTR(uint8_t, octx->ctx->vtcm_base, VLEN);
    dma_addr_t s_vtcm = (dma_addr_t)(uintptr_t) s_ptr;
    dma_addr_t d_vtcm = (dma_addr_t)(uintptr_t) d_ptr;
    const uint32_t s_size = kparams->src0_type_size;
    const uint32_t d_size = kparams->dst_type_size;

    dma_queue_push(dma_q, dma_make_data(s_vtcm, src0->data), s_size, s_size, s_size, 1);
    dma_queue_pop(dma_q);

    const HVX_Vector v_src = hvx_vmem(s_ptr);
    HVX_Vector v_dst;

    if (src0->type == HTP_TYPE_F32 && dst->type == HTP_TYPE_I32) {
        v_dst = Q6_Vw_equals_Vsf(v_src);
    } else if (src0->type == HTP_TYPE_I32 && dst->type == HTP_TYPE_F32) {
        v_dst = Q6_Vsf_equals_Vw(v_src);
    } else if (src0->type == HTP_TYPE_F32 && dst->type == HTP_TYPE_F16) {
        v_dst = hvx_vec_f32_to_f16(v_src, v_src);
    } else if (src0->type == HTP_TYPE_F16 && dst->type == HTP_TYPE_F32) {
        v_dst = Q6_V_lo_W(hvx_vec_f16_to_f32(v_src));
    } else {
        return HTP_STATUS_NO_SUPPORT;
    }

    hvx_vmem(d_ptr) = v_dst;

    dma_queue_push(dma_q, dma_make_data(dst->data, d_vtcm), d_size, d_size, d_size, 1);
    dma_queue_flush(dma_q);
    return HTP_STATUS_OK;
}

static int cpy_1d_contig(struct htp_ops_context * octx, const struct htp_copy_kernel_params * kparams) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    uint32_t elem_start = 0;
    uint32_t nelem      = kparams->total_elems;

    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            kparams->total_elems, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        elem_start = range.start;
        nelem      = range.count;
    }

    if (nelem > 0) {
        dma_queue * q = octx->ctx->dma[0];
        const uint32_t type_size = kparams->src0_type_size;
        dma_addr_t dst_addr  = dst->data  + elem_start * type_size;
        dma_addr_t src0_addr = src0->data + elem_start * type_size;
        dma_cpy_sametype_reshape_contig(q, dst_addr, src0_addr, nelem * type_size);
        dma_queue_flush(q);
    }

    return HTP_STATUS_OK;
}

static int cpy_sameshape_sametype(struct htp_ops_context * octx, const struct htp_copy_kernel_params * kparams) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    uint32_t row_start = 0;
    uint32_t nrows     = kparams->total_rows;

    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            kparams->total_rows, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_start = range.start;
        nrows     = range.count;
    }

    if (nrows > 0) {
        dma_queue * q = octx->ctx->dma[0];
        dma_cpy_sametype_sameshape_range(q, dst, src0, kparams->src0_type_size, row_start, nrows);
        dma_queue_flush(q);
    }

    return HTP_STATUS_OK;
}

static int cpy_sameshape_convert(struct htp_ops_context * octx, const struct htp_copy_kernel_params * kparams) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    if (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = kparams->total_rows;

    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            kparams->total_rows, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_start = range.start;
        nrows     = range.count;
    }

    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    if (kparams->vtcm_size > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    const uint32_t n_threads = octx->n_threads;
    const struct htp_copy_convert_params * cvt = &kparams->u.convert;

    struct htp_copy_context ct;
    htp_copy_convert_vtcm_layout_build(&ct.layout, src0->ne[0], kparams->src0_type_size, kparams->dst_type_size, cvt->blk_rows, n_threads);

    ct.octx                  = octx;
    ct.kparams               = kparams;
    ct.row_start             = row_start;
    ct.nrows                 = nrows;
    ct.src0_nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);

    uint8_t * vtcm_base = (uint8_t *) octx->ctx->vtcm_base;
    ct.vtcm_src0 = VTCM_LAYOUT_PTR(uint8_t, vtcm_base, ct.layout.off_src0);
    ct.vtcm_dst  = VTCM_LAYOUT_PTR(uint8_t, vtcm_base, ct.layout.off_dst);

    work_queue_func_t copy_fun = NULL;
    if (dst->type == HTP_TYPE_F16 && src0->type == HTP_TYPE_F32) {
        copy_fun = cpy_thread_f16_f32_sameshape;
    } else if (dst->type == HTP_TYPE_F32 && src0->type == HTP_TYPE_F16) {
        copy_fun = cpy_thread_f32_f16_sameshape;
    } else if (dst->type == HTP_TYPE_I32 && src0->type == HTP_TYPE_F32) {
        copy_fun = cpy_thread_i32_f32_sameshape;
    } else if (dst->type == HTP_TYPE_F32 && src0->type == HTP_TYPE_I32) {
        copy_fun = cpy_thread_f32_i32_sameshape;
    } else {
        return HTP_STATUS_NO_SUPPORT;
    }

    work_queue_run(octx->ctx->work_queue, copy_fun, &ct, n_threads);
    return HTP_STATUS_OK;
}

static int cpy_strided_convert(struct htp_ops_context * octx, const struct htp_copy_kernel_params * kparams) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    if (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    uint32_t row_start = 0;
    uint32_t nrows     = kparams->total_rows;

    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            kparams->total_rows, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        row_start = range.start;
        nrows     = range.count;
    }

    if (nrows == 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t n_threads = octx->n_threads;

    struct htp_copy_context ct;
    ct.octx                  = octx;
    ct.kparams               = kparams;
    ct.row_start             = row_start;
    ct.nrows                 = nrows;
    ct.src0_nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);

    work_queue_func_t copy_fun = NULL;
    if (src0->type == HTP_TYPE_F32 && dst->type == HTP_TYPE_F16) {
        copy_fun = cpy_thread_f32_f16_strided;
    } else if (src0->type == HTP_TYPE_F16 && dst->type == HTP_TYPE_F32) {
        copy_fun = cpy_thread_f16_f32_strided;
    } else {
        return HTP_STATUS_NO_SUPPORT;
    }

    work_queue_run(octx->ctx->work_queue, copy_fun, &ct, n_threads);
    return HTP_STATUS_OK;
}

static int cpy_reshape(struct htp_ops_context * octx, const struct htp_copy_kernel_params * kparams) {
    const struct htp_tensor * src0 = octx->src[0];
    const struct htp_tensor * dst  = octx->dst;

    if (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    uint32_t elem_start = 0;
    uint32_t nelem      = kparams->total_elems;

    if (octx->ctx->mdev.count > 1) {
        const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            kparams->total_elems, 1, octx->ctx->mdev.idx, octx->ctx->mdev.count, &octx->ctx->mdev.count_div);
        elem_start = range.start;
        nelem      = range.count;
    }

    if (nelem == 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t n_threads = octx->n_threads;

    struct htp_copy_context ct;
    ct.octx            = octx;
    ct.kparams         = kparams;
    ct.elem_start      = elem_start;
    ct.nelem           = nelem;
    ct.elem_per_thread = fastdiv(nelem + n_threads - 1, &octx->n_threads_div);

    work_queue_func_t copy_fun = NULL;
    switch (src0->type) {
        case HTP_TYPE_F32: copy_fun = cpy_thread_f32_reshape; break;
        case HTP_TYPE_F16: copy_fun = cpy_thread_f16_reshape; break;
        case HTP_TYPE_I32: copy_fun = cpy_thread_i32_reshape; break;
        default: return HTP_STATUS_NO_SUPPORT;
    }

    work_queue_run(octx->ctx->work_queue, copy_fun, &ct, n_threads);
    return HTP_STATUS_OK;
}

static bool cpy_needs_dma_addressing(const struct htp_ops_context * octx) {
    return htp_tensor_is_extended(octx->src[0]) || htp_tensor_is_extended(octx->dst);
}

int op_cpy(struct htp_ops_context * octx) {
    const struct htp_copy_kernel_params * kparams = (const struct htp_copy_kernel_params *) octx->kernel_params;
    int status = HTP_STATUS_OK;

    switch (kparams->kernel_type) {
        case HTP_COPY_KERNEL_SCALAR:
            status = cpy_scalar(octx, kparams);
            break;
        case HTP_COPY_KERNEL_1D_CONTIG:
            status = cpy_1d_contig(octx, kparams);
            break;
        case HTP_COPY_KERNEL_SAMESHAPE_SAMETYPE:
            status = cpy_sameshape_sametype(octx, kparams);
            break;
        case HTP_COPY_KERNEL_SAMESHAPE_CONVERT:
            status = cpy_sameshape_convert(octx, kparams);
            break;
        case HTP_COPY_KERNEL_RESHAPE:
            status = cpy_reshape(octx, kparams);
            break;
        case HTP_COPY_KERNEL_STRIDED_CONVERT:
            status = cpy_strided_convert(octx, kparams);
            break;
        case HTP_COPY_KERNEL_GATHER_ROWS:
            status = htp_gather_rows(octx, octx->src[0], NULL, &kparams->u.gather, kparams->n_threads);
            break;
        case HTP_COPY_KERNEL_TRANSPOSE:
            status = cpy_needs_dma_addressing(octx) ? cpy_reshape(octx, kparams) : cpy_transpose(octx, kparams);
            break;
        default:
            status = HTP_STATUS_NO_SUPPORT;
            break;
    }

    htp_ops_context_set_status(octx, status);

    if (octx->op == HTP_OP_CPY_FENCE) {
        if (octx->ctx->mdev.count > 1) {
            htp_mdev_group_barrier(octx);
        }

        if (octx->ctx->mdev.idx == 0) {
            const struct htp_tensor * sync = octx->src[1];
            const uint32_t seq = (uint32_t) octx->op_params[0];
            atomic_uint * sync_fence = (atomic_uint *) (uintptr_t) sync->data;
            htp_fence_write(sync_fence, seq, octx->status);

            FARF(HIGH, "ggml-hex: sync-release : fence %p seq 0x%x status %d\n", sync_fence, seq, octx->status);
        }
    }

    return octx->status;
}
