#pragma once

#include "ggml.h"

namespace parakeet {

inline ggml_tensor * flatten_channels_first_convolution(ggml_context * ctx, ggml_tensor * out) {
    // Restore contiguous CWHN order before collapsing the singleton height and batch axes.
    ggml_tensor * channels = ggml_permute(ctx, out, 1, 2, 0, 3);
    return ggml_reshape_2d(ctx, channels, out->ne[2], out->ne[0]);
}

} // namespace parakeet
