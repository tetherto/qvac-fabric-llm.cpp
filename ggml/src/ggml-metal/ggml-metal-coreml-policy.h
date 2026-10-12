#pragma once

#include "ggml.h"
#include <string.h>

// Keep this policy independent of Apple headers so it can be tested on CPU hosts.
static inline bool ggml_metal_coreml_shape_supported(const struct ggml_tensor * op) {
    if (!op || op->op != GGML_OP_MUL_MAT || !op->src[0] || !op->src[1]) {
        return false;
    }
    const struct ggml_tensor * w = op->src[0];
    const struct ggml_tensor * x = op->src[1];
    int32_t prec, hint;
    memcpy(&prec, op->op_params, sizeof(prec));
    memcpy(&hint, (const char *) op->op_params + sizeof(prec), sizeof(hint));
    if (prec != GGML_PREC_DEFAULT || hint != 0 || w->type != GGML_TYPE_Q8_0 ||
        (x->type != GGML_TYPE_F16 && x->type != GGML_TYPE_F32) || op->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t k = w->ne[0], m = w->ne[1], n = x->ne[1];
    if (k < 256 || m < 256 || n < 64 || k > 8192 || m > 8192 || n > 2048 ||
        k % 32 || m % 32 || n % 32 || x->ne[0] != k || op->ne[0] != m || op->ne[1] != n) {
        return false;
    }
    if (k*m*n < 16*1024*1024 || 2*(k*m + k*n + m*n) > 128*1024*1024) {
        return false;
    }
    const struct ggml_tensor * tensors[] = { w, x, op };
    for (int i = 0; i < 3; ++i) {
        if (tensors[i]->ne[2] != 1 || tensors[i]->ne[3] != 1 || !ggml_is_contiguous(tensors[i])) {
            return false;
        }
    }
    return true;
}

static inline bool ggml_metal_coreml_node_supported(const struct ggml_tensor * op) {
    return op && (op->flags & GGML_TENSOR_FLAG_COMPUTE) != 0 && ggml_metal_coreml_shape_supported(op);
}
