#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace tts_cpp {
namespace audio8 {
namespace detail {
namespace repack_weights {

inline bool is_q8_projection(const ggml_tensor * tensor) {
    if (tensor->type != GGML_TYPE_Q8_0 || ggml_n_dims(tensor) != 2 ||
        tensor->ne[0] % 32 != 0 || tensor->ne[1] > 32768) return false;
    const std::string name = ggml_get_name(tensor);
    const size_t prefix = name.compare(0, 7, "lm/blk/") == 0 ? 7 :
                          (name.compare(0, 9, "fast/blk/") == 0 ? 9 : 0);
    if (!prefix) return false;
    const size_t slash = name.find('/', prefix);
    if (slash == std::string::npos || slash == prefix ||
        name.find_first_not_of("0123456789", prefix) != slash) return false;
    const std::string role = name.substr(slash + 1);
    return role == "wqkv" || role == "wo" || role == "w13" || role == "w2" ||
           role == "wq" || role == "wk" || role == "wv" || role == "w1" || role == "w3";
}

// Allocate eligible projections first. The normal context allocator then skips
// their resident tensors and allocates every other weight in its usual layout.
// The metadata-only path applies the same partition and physical buffer sizing.
inline bool allocate_projection_weights(ggml_context * ctx, ggml_backend_buffer_type_t type,
                                 std::vector<ggml_backend_buffer_t> & buffers,
                                 size_t * measure_bytes, std::string * error) {
    if (!type) return true;
    const size_t alignment = ggml_backend_buft_get_alignment(type);
    const size_t max_size = ggml_backend_buft_get_max_size(type);
    std::vector<ggml_tensor *> tensors;
    size_t chunk_bytes = 0;
    auto allocate_chunk = [&]() {
        if (tensors.empty()) return true;
        if (measure_bytes) {
            const size_t physical_bytes = ggml_backend_buft_get_buffer_alloc_size(type, chunk_bytes);
            if (physical_bytes == std::numeric_limits<size_t>::max() ||
                *measure_bytes > std::numeric_limits<size_t>::max() - physical_bytes) return false;
            *measure_bytes += physical_bytes;
            for (auto * tensor : tensors) {
                tensor->data = reinterpret_cast<void *>(static_cast<uintptr_t>(1));
            }
        } else {
            auto buffer = ggml_backend_buft_alloc_buffer(type, chunk_bytes);
            if (!buffer) return false;
            buffers.push_back(buffer);
            ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            auto allocator = ggml_tallocr_new(buffer);
            for (auto * tensor : tensors) {
                if (ggml_tallocr_alloc(&allocator, tensor) != GGML_STATUS_SUCCESS) return false;
            }
        }
        tensors.clear();
        chunk_bytes = 0;
        return true;
    };
    bool success = alignment > 0 && max_size > 0;
    for (auto * tensor = ggml_get_first_tensor(ctx); success && tensor;
         tensor = ggml_get_next_tensor(ctx, tensor)) {
        if (!is_q8_projection(tensor)) continue;
        const size_t bytes = ggml_backend_buft_get_alloc_size(type, tensor);
        if (bytes > std::numeric_limits<size_t>::max() - (alignment - 1)) {
            success = false;
            break;
        }
        const size_t padded = ((bytes + alignment - 1) / alignment) * alignment;
        if (padded > max_size) {
            success = false;
            break;
        }
        if (padded > max_size - chunk_bytes && !allocate_chunk()) {
            success = false;
            break;
        }
        tensors.push_back(tensor);
        chunk_bytes += padded;
    }
    if (success) success = allocate_chunk();
    if (!success && error) *error = "audio8: failed to allocate repacked projection weights";
    return success;
}

}  // namespace repack_weights
}  // namespace detail
}  // namespace audio8
}  // namespace tts_cpp
