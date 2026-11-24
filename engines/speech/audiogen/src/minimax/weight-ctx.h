#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <cstddef>
#include <cstdio>
#include <memory>
#include <vector>

struct WeightCtx {
    struct ggml_context * ctx;
    ggml_backend_buffer_t buffer;

    struct PendingCopy {
        struct ggml_tensor * tensor;
        const void *         src;
        size_t               nbytes;
        size_t               offset;
    };

    std::vector<PendingCopy> pending;

    std::vector<std::unique_ptr<float[]>> staging;
};

static void wctx_init(WeightCtx * wctx, int n_tensors) {
    size_t                  ctx_size = (size_t) n_tensors * ggml_tensor_overhead() + 1024;
    struct ggml_init_params params   = {
        ctx_size,
        NULL,
        true,
    };
    wctx->ctx    = ggml_init(params);
    wctx->buffer = NULL;
    wctx->pending.clear();
    wctx->pending.reserve(n_tensors);
}

static void wctx_queue_copy(WeightCtx * wctx, ggml_tensor * tensor, const void * src, size_t nbytes, size_t offset) {
    if (src) {
        wctx->pending.push_back({ tensor, src, nbytes, offset });
    }
}

static bool wctx_alloc(WeightCtx * wctx, ggml_backend_t backend) {
    wctx->buffer = ggml_backend_alloc_ctx_tensors(wctx->ctx, backend);
    if (!wctx->buffer) {
        fprintf(stderr, "[WeightCtx] FATAL: failed to allocate backend buffer\n");
        return false;
    }

    ggml_backend_buffer_set_usage(wctx->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    size_t total = 0;
    for (auto & pc : wctx->pending) {
        ggml_backend_tensor_set(pc.tensor, pc.src, pc.offset, pc.nbytes);
        total += pc.nbytes;
    }
    fprintf(stderr, "[WeightCtx] Loaded %zu tensors, %.1f MB into backend\n", wctx->pending.size(),
            (float) total / (1024 * 1024));
    wctx->pending.clear();
    wctx->staging.clear();
    return true;
}

static void wctx_free(WeightCtx * wctx) {
    if (wctx->buffer) {
        ggml_backend_buffer_free(wctx->buffer);
    }
    if (wctx->ctx) {
        ggml_free(wctx->ctx);
    }
    wctx->buffer = NULL;
    wctx->ctx    = NULL;
}
