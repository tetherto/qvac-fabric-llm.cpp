#pragma once

#include <stdbool.h>
#include "htp-ops.h"
#include "conv-taps-ops.h"

static inline bool htp_mm_f16_f32_disjoint(uint64_t a, uint32_t a_bytes, uint64_t b, uint32_t b_bytes) {
    return a && b && a_bytes && b_bytes && a <= UINT32_MAX && b <= UINT32_MAX &&
           a_bytes <= UINT32_MAX - a && b_bytes <= UINT32_MAX - b &&
           (a + a_bytes <= b || b + b_bytes <= a);
}

static inline int htp_conv_taps_validate(const struct htp_tensor * w, const struct htp_tensor * x,
        const struct htp_tensor * d, const struct htp_conv_taps_params * p,
        uint32_t n_threads, uint32_t vtcm_size) {
    if (!w || !x || !d || (w->data >> 32) || (x->data >> 32) ||
        (d->data >> 32) || w->type != HTP_TYPE_F16 || x->type != HTP_TYPE_F32 || d->type != HTP_TYPE_F32 ||
        w->ne[0] < 64 || w->ne[0] % 32 || w->ne[1] % 4 || w->ne[2] < 2 || w->ne[2] > HTP_CONV_TAPS_MAX_TAPS ||
        w->ne[3] != 1 || x->ne[2] != 1 || x->ne[3] != 1 || d->ne[2] != 1 || d->ne[3] != 1 ||
        d->ne[0] != w->ne[1] || x->ne[0] != w->ne[0] || !d->ne[0] || !d->ne[1] ||
        w->nb[0] != 2 || w->nb[1] != (uint64_t) w->ne[0] * 2 || w->nb[2] != (uint64_t) w->nb[1] * w->ne[1] ||
        x->nb[0] != 4 || x->nb[1] != (uint64_t) x->ne[0] * 4 || d->nb[0] != 4 || d->nb[1] != (uint64_t) d->ne[0] * 4 ||
        (p->tap_step != -1 && (p->tap_step < 1 || p->tap_step > HTP_CONV_TAPS_MAX_STEP)) ||
        p->first_row != (p->tap_step < 0 ? w->ne[2] - 1 : 0) || p->tile_rows != HTP_CONV_TAPS_TILE_ROWS) return HTP_STATUS_NO_SUPPORT;
    const uint32_t distance = p->tap_step < 0 ? -p->tap_step : p->tap_step;
    const uint64_t required_rows = (uint64_t) d->ne[1] + (uint64_t) (w->ne[2] - 1) * distance;
    if (required_rows > x->ne[1] ||
        (uint64_t) w->nb[2] * w->ne[2] > w->size ||
        (uint64_t) x->nb[1] * required_rows > x->size ||
        (uint64_t) d->nb[1] * d->ne[1] > d->size ||
        !htp_mm_f16_f32_disjoint(d->data, d->size, w->data, w->size) ||
        !htp_mm_f16_f32_disjoint(d->data, d->size, x->data, x->size)) return HTP_STATUS_NO_SUPPORT;
    if (p->worker_bytes < (uint64_t) p->tile_rows * d->ne[0] * sizeof(float) ||
        p->worker_bytes % HTP_CONV_TAPS_ALIGNMENT || !n_threads ||
        (uint64_t) p->worker_bytes * n_threads > p->vtcm_size ||
        p->vtcm_size > vtcm_size) return HTP_STATUS_VTCM_TOO_SMALL;
    return HTP_STATUS_OK;
}
