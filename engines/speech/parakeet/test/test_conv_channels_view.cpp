// The conformer conv module hands the depthwise kernel a channel-contiguous view
// of its [d_model, T] activation. ggml tells that view apart from a plain
// contiguous tensor by nb[1] > nb[0], which needs at least two frames; the
// graph builder falls back to the portable lowering below that. The kernel
// answers in the same channel-innermost memory, which the builder reads back
// as a contiguous [d_model, T] tensor.
#include "conv_channels_layout.h"

#include "ggml.h"

#include <cstdio>

namespace {

constexpr int     k_d_model    = 8;
constexpr int64_t k_min_frames = 2;
constexpr int     k_kernel     = 3;
constexpr int     k_pad        = 1;
constexpr int     k_frames     = 4;

bool view_is_channels_first(ggml_context * ctx, int T) {
    ggml_tensor * y  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k_d_model, T);
    ggml_tensor * y4 = ggml_reshape_4d(ctx, y, k_d_model, T, 1, 1);
    return ggml_is_contiguous_channels(ggml_permute(ctx, y4, 2, 0, 1, 3));
}

ggml_tensor * channels_first_kernel(ggml_context * ctx) {
    ggml_tensor * w = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, k_d_model, k_kernel, 1, 1);
    return ggml_permute(ctx, w, 3, 0, 1, 2);
}

ggml_tensor * channels_first_input(ggml_context * ctx) {
    ggml_tensor * y4 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, k_d_model, k_frames, 1, 1);
    return ggml_permute(ctx, y4, 2, 0, 1, 3);
}

bool conv_output_is_contiguous_activation(ggml_context * ctx) {
    ggml_tensor * conv = ggml_conv_2d_dw_direct(ctx, channels_first_kernel(ctx), channels_first_input(ctx),
                                                1, 1, k_pad, 0, 1, 1);
    ggml_tensor * out = parakeet::channels_first_conv_output(ctx, conv);
    return ggml_is_contiguous(out) && ggml_n_dims(out) == 2 &&
           out->ne[0] == k_d_model && out->ne[1] == k_frames && out->view_src == conv;
}

} // namespace

int main() {
    ggml_init_params ip = { ggml_tensor_overhead() * 32, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return 1;
    int failures = 0;
    for (int T = 1; T <= 4; ++T) {
        const bool expect = T >= k_min_frames;
        const bool got    = view_is_channels_first(ctx, T);
        std::printf("[conv-channels-view] T=%d channel-contiguous=%d expected=%d: %s\n",
                    T, (int) got, (int) expect, got == expect ? "ok" : "FAIL");
        failures += got != expect;
    }
    const bool output_ok = conv_output_is_contiguous_activation(ctx);
    std::printf("[conv-channels-view] conv output reads back as [d_model, T]: %s\n", output_ok ? "ok" : "FAIL");
    failures += !output_ok;
    ggml_free(ctx);
    std::printf("[conv-channels-view] %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
