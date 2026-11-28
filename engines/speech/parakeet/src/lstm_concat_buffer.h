#pragma once

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>

namespace parakeet {
namespace detail {

// The direct decoder graphs need the same stacked-GEMV support decision when
// pricing a metadata-only runtime as when preparing its real weight buffer.
// A zero-sized backend buffer records placement without allocating payload.
template <typename Layers>
bool lstm_concat_gemvs_supported(ggml_backend_t backend, const Layers & layers) {
    ggml_init_params ip = { ggml_tensor_overhead() * 2 * layers.size(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;
    bool supported = true;
    for (const auto & cat : layers) {
        ggml_tensor * xh = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cat.w->ne[0]);
        supported = supported && ggml_backend_supports_op(backend, ggml_mul_mat(ctx, cat.w, xh));
    }
    ggml_free(ctx);
    return supported;
}

template <typename Layers>
bool prepare_lstm_concat_buffer(ggml_context * ctx, const Layers & layers,
                                ggml_backend_t backend, ggml_backend_buffer_type_t buft,
                                ggml_backend_buffer_t & buffer, size_t * measure_bytes) {
    const size_t bytes = measure_bytes
        ? ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, buft) : 0;
    buffer = measure_bytes ? ggml_backend_buft_alloc_buffer(buft, 0)
                           : ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (!buffer) return false;

    if (measure_bytes) {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
            // The probe needs the planned weight-buffer kind (HTP's explicit
            // repack buffer), while its tensor values remain unallocated.
            if (!t->data && !t->view_src) t->buffer = buffer;
        }
    }

    if (!lstm_concat_gemvs_supported(backend, layers)) {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
            if (t->buffer == buffer) {
                t->buffer = nullptr;
                t->data = nullptr;
            }
        }
        ggml_backend_buffer_free(buffer);
        buffer = nullptr;
        return false;
    }

    if (measure_bytes) {
        *measure_bytes += bytes;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
            if (!t->data && !t->view_src) {
                t->data = reinterpret_cast<void *>(static_cast<uintptr_t>(1));
            }
        }
    }
    return true;
}

}  // namespace detail
}  // namespace parakeet
