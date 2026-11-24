#pragma once

#include <cstdint>

struct ggml_tensor;

namespace parakeet::moss::detail {

void require_transcribe_weight(const ggml_tensor * tensor, int64_t ne0, int64_t ne1);
void require_transcribe_f32(const ggml_tensor * tensor, int64_t ne0, int64_t ne1 = 1);
void require_transcribe_kernel(const ggml_tensor * tensor, int64_t width, int64_t in_channels, int64_t out_channels);

} // namespace parakeet::moss::detail
