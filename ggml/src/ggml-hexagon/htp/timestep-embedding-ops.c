#include <HAP_farf.h>

#include <math.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-dma-reader.h"

struct htp_timestep_embedding_context {
    struct htp_ops_context * octx;
    uint32_t                 half;
    uint32_t                 rows_per_thread;
    float                    log_max_period;
};

static void timestep_embedding_row(const struct htp_timestep_embedding_context * tctx, uint32_t row, float timestep) {
    const struct htp_tensor * dst = tctx->octx->dst;

    float *     embed    = (float *) ((uint8_t *) dst->data + row * dst->nb[1]);

    for (uint32_t j = 0; j < tctx->half; j++) {
        const float freq = expf(-tctx->log_max_period * (float) j / (float) tctx->half);
        const float arg  = timestep * freq;
        embed[j]              = cosf(arg);
        embed[j + tctx->half] = sinf(arg);
    }
    if (dst->ne[0] % 2 != 0) {
        embed[2 * tctx->half] = 0.0f;
    }
}

static void timestep_embedding_thread(unsigned int nth, unsigned int ith, void * data) {
    const struct htp_timestep_embedding_context * tctx = (const struct htp_timestep_embedding_context *) data;

    const uint32_t rows  = tctx->octx->src[0]->ne[0];
    const uint32_t start = tctx->rows_per_thread * ith;
    const uint32_t end   = MIN(start + tctx->rows_per_thread, rows);

    struct htp_dma_reader reader = htp_dma_reader_init(tctx->octx->src[0], tctx->octx->ctx->dma[ith],
        tctx->octx->ctx->vtcm_base + ith * HTP_DMA_READER_BYTES);
    for (uint32_t row = start; row < end; row++) {
        const float timestep = *(const float *) htp_dma_reader_view(&reader, row * sizeof(float), sizeof(float));
        timestep_embedding_row(tctx, row, timestep);
    }
}

int op_timestep_embedding(struct htp_ops_context * octx) {
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * dst = octx->dst;

    if (src->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32 || src->nb[0] != sizeof(float)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const int32_t dim        = octx->op_params[0];
    const int32_t max_period = octx->op_params[1];
    if (dim <= 0 || max_period <= 0 || (uint32_t) dim != dst->ne[0] || dst->ne[1] != src->ne[0]) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    const uint32_t rows    = src->ne[0];
    const uint32_t threads = MIN(octx->n_threads, rows);
    if (threads == 0) {
        return HTP_STATUS_OK;
    }

    if ((size_t) threads * HTP_DMA_READER_BYTES > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }
    struct htp_timestep_embedding_context tctx = {
        .octx            = octx,
        .half            = (uint32_t) dim / 2,
        .rows_per_thread = (rows + threads - 1) / threads,
        .log_max_period  = logf((float) max_period),
    };
    work_queue_run(octx->ctx->work_queue, timestep_embedding_thread, &tctx, threads);
    return HTP_STATUS_OK;
}
