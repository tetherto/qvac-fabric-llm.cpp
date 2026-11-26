#pragma once

#include "ggml.h"

template<int DKQ, int DV, int ncols2, bool use_logit_softcap, bool use_rows = false>
static bool ggml_cuda_fattn_tile_use_f32_pv(bool is_gfx1151, const ggml_tensor * dst) {
    constexpr int head_size = 128;
    constexpr int small_query_limit = 32;
    return DKQ == head_size && DV == head_size && ncols2 == 1 && !use_logit_softcap && !use_rows &&
           is_gfx1151 && dst->src[0]->ne[1] > small_query_limit &&
           ggml_flash_attn_ext_get_prec(dst) == GGML_PREC_F32 &&
           dst->src[3] == nullptr && dst->src[4] == nullptr &&
           dst->src[0]->ne[2] == dst->src[1]->ne[2];
}
