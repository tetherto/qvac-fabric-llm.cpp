#pragma once

#include "ggml.h"

namespace parakeet {

// ggml_conv_2d_dw_direct answers a channel-contiguous input as logical (T, 1, d_model) over
// channel-innermost memory; this is the same memory read as a contiguous (d_model, T) tensor.
inline ggml_tensor * channels_first_conv_output(ggml_context * ctx, ggml_tensor * out) {
    return ggml_permute(ctx, out, 1, 2, 0, 3);
}

}
