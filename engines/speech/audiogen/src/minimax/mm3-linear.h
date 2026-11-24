#pragma once

#include "ggml.h"

#include <cstdint>

// Applies a weight matrix to every column of `x` in one matrix product. The
// CFG pair rides in ne[2] of the MiniMax activations; handing that layout to
// ggml_mul_mat as a broadcast batch makes the GPU matrix-vector kernels stream
// the weights once per batch entry, so the columns are folded into ne[1]
// first and the original shape is restored afterwards. `x` must be contiguous.
// GGML_PREC_F32 keeps the activations and the accumulation in f32, which the
// matrix-vector kernels do at no cost for float weights (see
// mm3_weight_precision); large products leave it at the default so they keep
// the half-precision tensor-core GEMMs.
// A GGML_PREC_F32 request is only passed on for float weights. For those the
// small-batch kernels keep f32 activations at no cost; a quantized weight would
// instead leave its integer kernels (CUDA dequantizes the whole matrix for an f32
// GEMM, 4.5x slower on a q8_0 LM decode step), so it keeps the default precision.
static ggml_prec mm3_weight_precision(const ggml_tensor * weight, ggml_prec requested) {
    const bool float_weight =
        weight->type == GGML_TYPE_F32 || weight->type == GGML_TYPE_F16 || weight->type == GGML_TYPE_BF16;
    return float_weight ? requested : GGML_PREC_DEFAULT;
}

static ggml_tensor * mm3_linear_columns(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * x,
                                        ggml_prec precision) {
    const int64_t columns = ggml_nelements(x) / x->ne[0];
    ggml_tensor * y       = ggml_mul_mat(ctx, weight, ggml_reshape_2d(ctx, x, x->ne[0], columns));
    ggml_mul_mat_set_prec(y, mm3_weight_precision(weight, precision));
    return y;
}

static ggml_tensor * mm3_restore_columns(ggml_context * ctx, ggml_tensor * y, const ggml_tensor * x) {
    return ggml_reshape_4d(ctx, y, y->ne[0], x->ne[1], x->ne[2], x->ne[3]);
}

static ggml_tensor * mm3_linear(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * x,
                                ggml_prec precision = GGML_PREC_DEFAULT) {
    return mm3_restore_columns(ctx, mm3_linear_columns(ctx, weight, x, precision), x);
}

// mm3_linear plus a per-output bias. The bias is added before the shape is restored,
// so the add consumes the product directly and a backend can fold it into the
// product (CUDA adds it while converting its half-precision GEMM output).
static ggml_tensor * mm3_linear_bias(ggml_context * ctx, ggml_tensor * weight, ggml_tensor * bias, ggml_tensor * x) {
    ggml_tensor * y = ggml_add(ctx, mm3_linear_columns(ctx, weight, x, GGML_PREC_DEFAULT), bias);
    return mm3_restore_columns(ctx, y, x);
}

// Views rows [offset, offset + D * heads) of every column of a contiguous
// projection output as [D, heads, columns, branches]. Consumers read the view
// directly, so the projection is never copied out.
static ggml_tensor * mm3_head_view(ggml_context * ctx, ggml_tensor * x, int64_t D, int64_t heads, int64_t offset) {
    return ggml_view_4d(ctx, x, D, heads, x->ne[1], x->ne[2], (size_t) D * x->nb[0], x->nb[1], x->nb[2],
                        (size_t) offset * x->nb[0]);
}

// q/k/v as [D, heads, tokens, branches].
struct MM3HeadProjections {
    ggml_tensor * q = nullptr;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
};

struct MM3AttentionInputs {
    ggml_tensor * qkv = nullptr;
    ggml_tensor * q   = nullptr;
    ggml_tensor * k   = nullptr;
    ggml_tensor * v   = nullptr;
};

// Projects `x` onto the attention heads: one product and three views when the
// q/k/v weights are stacked, three products otherwise.
static MM3HeadProjections mm3_project_heads(ggml_context * ctx, const MM3AttentionInputs & w, ggml_tensor * x,
                                            int64_t D, int64_t Nh, int64_t Nkv, ggml_prec precision) {
    MM3HeadProjections heads;
    if (w.qkv) {
        ggml_tensor * qkv = mm3_linear(ctx, w.qkv, x, precision);
        heads.q           = mm3_head_view(ctx, qkv, D, Nh, 0);
        heads.k           = mm3_head_view(ctx, qkv, D, Nkv, D * Nh);
        heads.v           = mm3_head_view(ctx, qkv, D, Nkv, D * (Nh + Nkv));
        return heads;
    }
    const int64_t T = x->ne[1];
    const int64_t B = x->ne[2];
    heads.q = ggml_reshape_4d(ctx, mm3_linear(ctx, w.q, x, precision), D, Nh, T, B);
    heads.k = ggml_reshape_4d(ctx, mm3_linear(ctx, w.k, x, precision), D, Nkv, T, B);
    heads.v = ggml_reshape_4d(ctx, mm3_linear(ctx, w.v, x, precision), D, Nkv, T, B);
    return heads;
}

// silu(gate(x)) * up(x), through one product and the fused SwiGLU when the
// gate and up weights are stacked (gate rows first).
static ggml_tensor * mm3_gated_ffn_input(ggml_context * ctx, ggml_tensor * gate_up, ggml_tensor * gate,
                                         ggml_tensor * up, ggml_tensor * x, ggml_prec precision) {
    if (gate_up) {
        return ggml_swiglu(ctx, mm3_linear(ctx, gate_up, x, precision));
    }
    return ggml_mul(ctx, ggml_silu(ctx, mm3_linear(ctx, gate, x, precision)), mm3_linear(ctx, up, x, precision));
}
