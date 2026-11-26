#include <math.h>
#include <stdint.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "timestep-ops.h"

static void timestep_embed_one(float * restrict embedding, float timestep, const struct htp_timestep_kernel_params * kparams) {
    const uint32_t half = kparams->dim / 2;

    for (uint32_t j = 0; j < half; j++) {
        const float freq = expf(-kparams->log_max_period * (float) j / (float) half);
        const float arg  = timestep * freq;
        embedding[j]        = cosf(arg);
        embedding[j + half] = sinf(arg);
    }
    if (kparams->dim % 2 != 0) {
        embedding[2 * half] = 0.0f;
    }
}

static void timestep_embed_all(const struct htp_tensor * src, const struct htp_tensor * dst, const struct htp_timestep_kernel_params * kparams) {
    const float * timesteps = (const float *) src->data;

    for (uint32_t i = 0; i < kparams->n_timesteps; i++) {
        float * embedding = (float *) ((uint8_t *) dst->data + (size_t) i * dst->nb[1]);
        timestep_embed_one(embedding, timesteps[i], kparams);
    }
}

int op_timestep_embedding(struct htp_ops_context * octx) {
    const struct htp_tensor * src = octx->src[0];
    const struct htp_tensor * dst = octx->dst;

    if (htp_tensor_is_extended(src) || htp_tensor_is_extended(dst)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    timestep_embed_all(src, dst, (const struct htp_timestep_kernel_params *) octx->kernel_params);
    return HTP_STATUS_OK;
}
