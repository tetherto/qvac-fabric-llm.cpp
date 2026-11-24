#pragma once

#include "../ggml-impl.h"
#include "../ggml-backend-impl.h"

#include <cstdint>

constexpr int GGML_CUDA_H3_HIDDEN_SIZE = 5376;
constexpr int GGML_CUDA_H3_MAX_VALIDATED_ROWS = 15429;

static inline bool ggml_cuda_can_fuse_scale_residual_gfx1151(
        bool is_gfx1151, int warp_size, ggml_backend_buffer_type_t buffer_type,
        const ggml_cgraph * cgraph, int i) {
    if (!is_gfx1151 || warp_size != 32 ||
        i < 0 || i >= cgraph->n_nodes - 1) {
        return false;
    }
    const ggml_tensor * product = cgraph->nodes[i];
    const ggml_tensor * dst = cgraph->nodes[i + 1];
    if (!product || !dst || product->op != GGML_OP_MUL || dst->op != GGML_OP_ADD) {
        return false;
    }
    const ggml_tensor * value = product->src[0];
    const ggml_tensor * scale = product->src[1];
    if (!value || !scale || dst->src[0] != value || dst->src[1] != product) {
        return false;
    }
    for (int slot = 2; slot < GGML_MAX_SRC; ++slot) {
        if (product->src[slot] || dst->src[slot]) {
            return false;
        }
    }
    if (value->ne[0] != GGML_CUDA_H3_HIDDEN_SIZE ||
        value->ne[1] < 1 || value->ne[1] > GGML_CUDA_H3_MAX_VALIDATED_ROWS ||
        value->ne[2] != 1 || value->ne[3] != 1 ||
        scale->ne[0] != GGML_CUDA_H3_HIDDEN_SIZE ||
        scale->ne[1] != 1 || scale->ne[2] != 1 || scale->ne[3] != 1 ||
        !ggml_are_same_shape(value, product) || !ggml_are_same_shape(value, dst)) {
        return false;
    }
    for (const ggml_tensor * tensor : { value, scale, product, dst }) {
        if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor) || !tensor->data ||
            !tensor->buffer || tensor->buffer->buft != buffer_type ||
            (uintptr_t) tensor->data % 16 != 0) {
            return false;
        }
        size_t stride = sizeof(float);
        for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
            if (tensor->nb[dim] != stride) {
                return false;
            }
            stride *= tensor->ne[dim];
        }
    }
    if (!ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_MUL, GGML_OP_ADD }, { i + 1 })) {
        return false;
    }

    struct range { uintptr_t begin, end; };
    auto span = [](const ggml_tensor * tensor, range & result) {
        result.begin = (uintptr_t) tensor->data;
        const size_t bytes = ggml_nbytes(tensor);
        if (bytes == 0 || bytes > UINTPTR_MAX - result.begin) {
            return false;
        }
        result.end = result.begin + bytes;
        return true;
    };
    auto overlaps = [](const range & a, const range & b) {
        return a.begin < b.end && b.begin < a.end;
    };
    auto same = [](const range & a, const range & b) {
        return a.begin == b.begin && a.end == b.end;
    };
    range v, s, p, d;
    if (!span(value, v) || !span(scale, s) || !span(product, p) || !span(dst, d)) {
        return false;
    }
    // Unlike the generic range helper, include leaf inputs and allow exact
    // elementwise dst/value aliasing. Broadcast scale must never be overwritten.
    return !overlaps(d, s) && (!overlaps(d, v) || same(d, v)) &&
           !overlaps(p, v) && !overlaps(p, s) && (!overlaps(p, d) || same(p, d));
}
