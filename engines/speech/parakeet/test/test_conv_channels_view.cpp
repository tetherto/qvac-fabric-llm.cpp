// The conformer conv module hands the depthwise kernel a channel-contiguous view
// of its [d_model, T] activation. ggml tells that view apart from a plain
// contiguous tensor by nb[1] > nb[0], which needs at least two frames; the
// graph builder falls back to the portable lowering below that.
#include "ggml.h"

#include <cstdio>

namespace {

constexpr int     k_d_model    = 8;
constexpr int64_t k_min_frames = 2;

bool view_is_channels_first(ggml_context * ctx, int T) {
    ggml_tensor * y  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k_d_model, T);
    ggml_tensor * y4 = ggml_reshape_4d(ctx, y, k_d_model, T, 1, 1);
    return ggml_is_contiguous_channels(ggml_permute(ctx, y4, 2, 0, 1, 3));
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
    ggml_free(ctx);
    std::printf("[conv-channels-view] %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
