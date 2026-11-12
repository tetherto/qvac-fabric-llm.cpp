#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-hexagon.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

static bool check(ggml_backend_t backend, ggml_backend_hexagon_conv_taps_f32_t create,
                  int k, int rows, int taps, int step, float scale = 1.0f) {
    const int channels = 4;
    const int pad = (taps - 1) * std::abs(step);
    const int input_rows = rows + pad;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({1024 * 1024, nullptr, true}), ggml_free);
    if (!ctx) return false;
    auto * w = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, k, channels, taps);
    auto * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, input_rows);
    auto * tiled = create(ctx.get(), backend, w, x, step, rows);
    if (!tiled) { std::fprintf(stderr, "missing DDR tile k=%d rows=%d\n", k, rows); return false; }
    ggml_set_output(tiled);
    auto * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, tiled);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
    if (!buffer) return false;
    ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<ggml_fp16_t> weights(k * channels * taps);
    std::vector<float> input(size_t(k) * input_rows);
    for (int t = 0; t < taps; ++t) for (int c = 0; c < channels; ++c) for (int j = 0; j < k; ++j) {
        weights[(t * channels + c) * k + j] = ggml_fp32_to_fp16(float((j * 5 + c * 7 + t * 11) % 31 - 15) / 1024);
    }
    for (int r = 0; r < input_rows; ++r) for (int j = 0; j < k; ++j) {
        input[size_t(r) * k + j] = r < pad ? 0 : scale * float(((r - pad) * 13 + j * 7) % 127 - 63) / 128;
    }
    ggml_backend_tensor_set(w, weights.data(), 0, weights.size() * sizeof(weights[0]));
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    if (!ggml_backend_supports_op(backend, tiled) ||
        ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return false;
    std::vector<float> actual(size_t(rows) * channels);
    ggml_backend_tensor_get(tiled, actual.data(), 0, actual.size() * sizeof(float));
    const int boundaries[] = {0, 1, 63, 64, 65, 127, 128, 319, 320, 383, 384, rows - 2, rows - 1};
    for (int r : boundaries) for (int c = 0; c < channels; ++c) {
        float sum = 0;
        for (int t = 0; t < taps; ++t) {
            const int pos = step < 0 ? r + taps - 1 - t : r + t * step;
            double product = 0;
            for (int j = 0; j < k; ++j) {
                product += double(ggml_fp16_to_fp32(weights[(t * channels + c) * k + j])) * input[size_t(pos) * k + j];
            }
            sum += float(product);
        }
        if (!std::isfinite(actual[size_t(r) * channels + c]) || actual[size_t(r) * channels + c] != sum) {
            std::fprintf(stderr, "CPU oracle mismatch k=%d row=%d channel=%d\n", k, r, c); return false;
        }
    }
    auto * small = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, 128);
    if (create(ctx.get(), backend, w, small, 1, 65) || create(ctx.get(), backend, w, x, 0, rows)) {
        std::fprintf(stderr, "invalid or existing-VTCM shape accepted\n"); return false;
    }
    std::printf("PASS k=%d channels=%d rows=%d taps=%d step=%d CPU_boundaries=true\n",
                k, channels, rows, taps, step);
    return true;
}

int main() {
    ggml_backend_load_all();
    auto * backend = ggml_backend_init_by_name("HTP0", nullptr);
    if (!backend) return 77;
    const auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    const auto create = reinterpret_cast<ggml_backend_hexagon_conv_taps_f32_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_hexagon_conv_taps_f32"));
    if (!create) { ggml_backend_free(backend); return 1; }
    const bool ok = check(backend, create, 96, 32769, 7, 1) &&
                    check(backend, create, 192, 21847, 3, 9) &&
                    check(backend, create, 384, 10923, 2, -1) &&
                    check(backend, create, 96, 32769, 7, 1, 262144.0f);
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
