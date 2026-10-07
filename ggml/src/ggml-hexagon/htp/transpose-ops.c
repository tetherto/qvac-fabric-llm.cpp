#include <HAP_farf.h>

#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "hex-dma.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "hvx-utils.h"

// Source rows and columns per tile. Each tile is staged in VTCM with
// dma_queue_copy_rows, transposed with word gathers into a second VTCM tile,
// and written back the same way.
#define TRANSPOSE_TILE 128
#define TRANSPOSE_LANES (VLEN / sizeof(float))

struct htp_transpose_run {
    const struct htp_transpose_f32 * job;
    uint8_t *                        vtcm;
    uint32_t                         bytes_per_thread;
    uint32_t                         row_tiles;
    uint32_t                         col_tiles;
    uint32_t                         n_tiles;
    uint32_t                         tiles_per_thread;
};

static inline uint32_t transpose_tile_bytes(void) {
    return TRANSPOSE_TILE * TRANSPOSE_TILE * sizeof(float);
}

static inline uint32_t transpose_gather_bytes(void) {
    return (TRANSPOSE_TILE / TRANSPOSE_LANES) * VLEN;
}

static HVX_Vector transpose_row_offsets(void) {
    int32_t lanes[TRANSPOSE_LANES] __attribute__((aligned(VLEN)));
    for (uint32_t l = 0; l < TRANSPOSE_LANES; l++) {
        lanes[l] = (int32_t) (l * TRANSPOSE_TILE * sizeof(float));
    }
    return *(const HVX_Vector *) lanes;
}

static void transpose_stage(dma_queue * q, float * in, const uint8_t * src, uint32_t src_row_stride, uint32_t tr,
                            uint32_t tc) {
    dma_queue_copy_rows(q, dma_make_ptr(in, src), TRANSPOSE_TILE * sizeof(float), src_row_stride, tc * sizeof(float), tr);
}

static void transpose_gather_column(HVX_Vector * tmp, float * out_row, const float * in_col, uint32_t tr,
                                    HVX_Vector offsets) {
    const uint32_t region = (TRANSPOSE_LANES - 1) * TRANSPOSE_TILE * sizeof(float) + sizeof(float) - 1;
    const uint32_t n_vec  = (tr + TRANSPOSE_LANES - 1) / TRANSPOSE_LANES;
    for (uint32_t v = 0; v < n_vec; v++) {
        Q6_vgather_ARMVw(&tmp[v], (size_t) (in_col + v * TRANSPOSE_LANES * TRANSPOSE_TILE), region, offsets);
    }
    for (uint32_t v = 0; v < n_vec; v++) {
        ((HVX_Vector *) out_row)[v] = tmp[v];
    }
}

static void transpose_gather_tile(HVX_Vector * tmp, float * out, const float * in, uint32_t tr, uint32_t tc,
                                  HVX_Vector offsets) {
    for (uint32_t c = 0; c < tc; c++) {
        transpose_gather_column(tmp, out + c * TRANSPOSE_TILE, in + c, tr, offsets);
    }
}

static void transpose_write(dma_queue * q, uint8_t * dst, uint32_t dst_row_stride, const float * out, uint32_t tr,
                            uint32_t tc) {
    dma_queue_copy_rows(q, dma_make_ptr(dst, out), dst_row_stride, TRANSPOSE_TILE * sizeof(float), tr * sizeof(float), tc);
}

static void transpose_tile(const struct htp_transpose_run * run, unsigned int ith, uint32_t tile, HVX_Vector offsets,
                           dma_queue * q) {
    const struct htp_transpose_f32 * job = run->job;
    const uint32_t per_batch = run->row_tiles * run->col_tiles;
    const uint32_t b         = tile / per_batch;
    const uint32_t rt        = (tile % per_batch) / run->col_tiles;
    const uint32_t ct        = tile % run->col_tiles;
    const uint32_t r0        = rt * TRANSPOSE_TILE;
    const uint32_t c0        = ct * TRANSPOSE_TILE;
    const uint32_t tr        = MIN(TRANSPOSE_TILE, job->rows - r0);
    const uint32_t tc        = MIN(TRANSPOSE_TILE, job->cols - c0);
    const uint32_t b2        = b % job->batch2;
    const uint32_t b3        = b / job->batch2;

    float *      in  = (float *) (run->vtcm + (size_t) ith * run->bytes_per_thread);
    float *      out = (float *) ((uint8_t *) in + transpose_tile_bytes());
    HVX_Vector * tmp = (HVX_Vector *) ((uint8_t *) out + transpose_tile_bytes());

    const uint8_t * src = job->src + b2 * job->src_stride2 + b3 * job->src_stride3 + r0 * job->src_row_stride +
                          c0 * sizeof(float);
    uint8_t * dst = job->dst + b2 * job->dst_stride2 + b3 * job->dst_stride3 + c0 * job->dst_row_stride +
                    r0 * sizeof(float);

    transpose_stage(q, in, src, job->src_row_stride, tr, tc);
    transpose_gather_tile(tmp, out, in, tr, tc, offsets);
    transpose_write(q, dst, job->dst_row_stride, out, tr, tc);
}

static void transpose_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_transpose_run * run = (const struct htp_transpose_run *) data;
    const HVX_Vector offsets = transpose_row_offsets();
    dma_queue *      q       = run->job->octx->ctx->dma[ith];
    const uint32_t   first   = run->tiles_per_thread * ith;
    const uint32_t   last    = MIN(first + run->tiles_per_thread, run->n_tiles);
    for (uint32_t tile = first; tile < last; tile++) {
        transpose_tile(run, ith, tile, offsets, q);
    }
}

bool htp_transpose_f32(const struct htp_transpose_f32 * job) {
    struct htp_ops_context * octx = job->octx;
    struct htp_transpose_run run  = {
        .job              = job,
        .vtcm             = octx->ctx->vtcm_base,
        .bytes_per_thread = 2 * transpose_tile_bytes() + transpose_gather_bytes(),
        .row_tiles        = (job->rows + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE,
        .col_tiles        = (job->cols + TRANSPOSE_TILE - 1) / TRANSPOSE_TILE,
    };
    run.n_tiles = run.row_tiles * run.col_tiles * job->batch2 * job->batch3;
    if (run.n_tiles == 0) {
        return true;
    }
    const uint32_t n_threads = MIN(octx->n_threads, run.n_tiles);
    if ((size_t) run.bytes_per_thread * n_threads > octx->ctx->vtcm_size) {
        return false;
    }
    run.tiles_per_thread = (run.n_tiles + n_threads - 1) / n_threads;
    work_queue_run(octx->ctx->work_queue, transpose_thread, &run, n_threads);
    return true;
}
