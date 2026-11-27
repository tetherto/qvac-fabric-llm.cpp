// The conformer conv module hands the depthwise kernel a channel-contiguous view
// of its [d_model, T] activation. ggml tells that view apart from a plain
// contiguous tensor by nb[1] > nb[0], which needs at least two frames; the
// graph builder falls back to the portable lowering below that.
#include "ggml.h"
#include "depthwise_conv_layout.h"

#include <cstdio>
#include <vector>

namespace {

constexpr int     k_d_model    = 8;
constexpr int64_t k_min_frames = 2;
constexpr int k_context_tensors = 128;
constexpr int k_kernel_width = 3;
constexpr int k_kernel_padding = k_kernel_width / 2;

bool view_is_channels_first(ggml_context * ctx, int T) {
    ggml_tensor * y  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k_d_model, T);
    ggml_tensor * y4 = ggml_reshape_4d(ctx, y, k_d_model, T, 1, 1);
    return ggml_is_contiguous_channels(ggml_permute(ctx, y4, 2, 0, 1, 3));
}

ggml_tensor * create_channels_first_convolution(ggml_context * ctx, int frames) {
    auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k_d_model, frames);
    auto * input4 = ggml_reshape_4d(ctx, input, k_d_model, frames, 1, 1);
    auto * channels = ggml_permute(ctx, input4, 2, 0, 1, 3);
    auto * weights = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, k_d_model, k_kernel_width, 1, 1);
    auto * kernel = ggml_permute(ctx, weights, 3, 0, 1, 2);
    return ggml_conv_2d_dw_direct(ctx, kernel, channels, 1, 1, k_kernel_padding, 0, 1, 1);
}

bool output_preserves_channels_and_frames(ggml_context * ctx, int frames) {
    auto * out = create_channels_first_convolution(ctx, frames);
    if (!ggml_is_contiguous_channels(out) || out->nb[0] == sizeof(float)) return false;

    // Populate the GPU output layout without executing a GPU kernel.
    std::vector<float> values(k_d_model * frames);
    for (size_t i = 0; i < values.size(); ++i) values[i] = static_cast<float>(i);
    out->data = values.data();
    auto * flat = parakeet::flatten_channels_first_convolution(ctx, out);
    if (!ggml_is_contiguous(flat) || flat->ne[0] != k_d_model || flat->ne[1] != frames ||
        flat->data != out->data || flat->view_src != out) return false;

    for (int frame = 0; frame < frames; ++frame) {
        for (int channel = 0; channel < k_d_model; ++channel) {
            const auto * source = reinterpret_cast<const float *>(
                static_cast<const char *>(out->data) + frame * out->nb[0] + channel * out->nb[2]);
            if (ggml_get_data_f32(flat)[frame * k_d_model + channel] != *source) return false;
        }
    }
    return true;
}

} // namespace

int main() {
    ggml_init_params ip = { ggml_tensor_overhead() * k_context_tensors, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return 1;
    int failures = 0;
    for (int T = 1; T <= 4; ++T) {
        const bool expect = T >= k_min_frames;
        const bool got    = view_is_channels_first(ctx, T);
        std::printf("[conv-channels-view] T=%d channel-contiguous=%d expected=%d: %s\n",
                    T, (int) got, (int) expect, got == expect ? "ok" : "FAIL");
        failures += got != expect;
        if (expect) {
            const bool preserves = output_preserves_channels_and_frames(ctx, T);
            std::printf("[conv-channels-output] T=%d: %s\n", T, preserves ? "ok" : "FAIL");
            failures += !preserves;
        }
    }
    ggml_free(ctx);
    std::printf("[conv-channels-view] %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
