#include "ggml.h"
#include "ggml-backend.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

using coordinates = std::array<int64_t, 4>;
static constexpr int64_t width = 7, height = 5, depth = 4, channels = 2, batches = 2;

struct fixture {
    ggml_context * ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    ggml_backend_buffer_t buffer = nullptr;

    fixture() {
        if (!ctx) {
            throw std::runtime_error("context allocation failed");
        }
    }
    ~fixture() {
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
};

static coordinates unravel(int64_t index, const int64_t * shape) {
    coordinates p = {};
    for (size_t axis = 0; axis < p.size(); ++axis) {
        p[axis] = index % shape[axis];
        index /= shape[axis];
    }
    return p;
}

static float value_at(const coordinates & p) {
    if (p[0] < 0 || p[0] >= width || p[1] < 0 || p[1] >= height || p[2] < 0 || p[2] >= depth) {
        return 0.0f;
    }
    return float(p[0] + 11 * p[1] + 101 * p[2] + 1009 * p[3] - 29) / 32.0f;
}

static void fill_input(ggml_tensor * storage, const ggml_tensor * input) {
    std::vector<float> data(ggml_nelements(storage), -1234.0f);
    for (int64_t i = 0; i < ggml_nelements(input); ++i) {
        const auto p = unravel(i, input->ne);
        const size_t offset = input->view_offs + p[0] * input->nb[0] + p[1] * input->nb[1] +
                              p[2] * input->nb[2] + p[3] * input->nb[3];
        data.at(offset / sizeof(float)) = value_at(p);
    }
    ggml_backend_tensor_set(storage, data.data(), 0, ggml_nbytes(storage));
}

static float expected_patch(const coordinates & p) {
    constexpr int64_t kw = 3, kh = 2, kd = 2, od = 5;
    const int64_t column = p[0];
    return value_at({p[1] * 2 - 1 + column % kw,
                     p[2] + 2 * (column / kw % kh),
                     p[3] % od - 1 + column / (kw * kh) % kd,
                     p[3] / od * channels + column / (kw * kh * kd)});
}

static bool compare_output(ggml_tensor * output) {
    std::vector<float> data(ggml_nelements(output));
    if (output->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> halves(data.size());
        ggml_backend_tensor_get(output, halves.data(), 0, ggml_nbytes(output));
        ggml_fp16_to_fp32_row(halves.data(), data.data(), int64_t(data.size()));
    } else {
        ggml_backend_tensor_get(output, data.data(), 0, ggml_nbytes(output));
    }
    for (size_t i = 0; i < data.size(); ++i) {
        float expected = expected_patch(unravel(int64_t(i), output->ne));
        if (output->type == GGML_TYPE_F16) {
            expected = ggml_fp16_to_fp32(ggml_fp32_to_fp16(expected));
        }
        if (!std::isfinite(data[i]) || data[i] != expected) {
            std::fprintf(stderr, "value %zu: got %g, expected %g\n", i, data[i], expected);
            return false;
        }
    }
    return true;
}

static bool test_layout(ggml_backend_t backend, ggml_type type, int layout) {
    fixture test;
    ggml_tensor * storage = ggml_new_tensor_4d(test.ctx, GGML_TYPE_F32,
            layout == 2 ? height : width + (layout == 1 ? 2 : 0),
            layout == 2 ? width : height, depth, channels * batches);
    ggml_tensor * input = storage;
    if (layout == 1) {
        input = ggml_view_4d(test.ctx, storage, width, height, depth, channels * batches,
                            storage->nb[1], storage->nb[2], storage->nb[3], sizeof(float));
    } else if (layout == 2) {
        input = ggml_permute(test.ctx, storage, 1, 0, 2, 3);
    }
    ggml_tensor * kernel = ggml_new_tensor_4d(test.ctx, type, 3, 2, 2, channels * 3);
    ggml_tensor * output = ggml_im2col_3d(test.ctx, kernel, input, channels, 2, 1, 1, 1, 0, 1, 1, 2, 1, type);
    if (output->ne[0] != 24 || output->ne[1] != 4 || output->ne[2] != 3 || output->ne[3] != 10 ||
        !ggml_is_contiguous(output) || !ggml_backend_supports_op(backend, output)) {
        std::fprintf(stderr, "Unsupported layout or unexpected output shape\n");
        return false;
    }
    ggml_cgraph * graph = ggml_new_graph(test.ctx);
    ggml_build_forward_expand(graph, output);
    test.buffer = ggml_backend_alloc_ctx_tensors(test.ctx, backend);
    if (!test.buffer) {
        return false;
    }
    fill_input(storage, input);
    ggml_backend_tensor_memset(kernel, 0, 0, ggml_nbytes(kernel));
    return ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS && compare_output(output);
}

static bool test_guards(ggml_backend_t backend) {
    fixture test;
    ggml_tensor * kernel = ggml_new_tensor_4d(test.ctx, GGML_TYPE_F16, 3, 2, 2, channels);
    ggml_tensor * input = ggml_new_tensor_4d(test.ctx, GGML_TYPE_F16, width, height, depth, channels);
    ggml_tensor * output = ggml_im2col_3d(test.ctx, kernel, input, channels, 1, 1, 1, 0, 0, 0, 1, 1, 1, GGML_TYPE_F32);
    if (ggml_backend_supports_op(backend, output)) {
        return false;
    }
    input = ggml_new_tensor_4d(test.ctx, GGML_TYPE_F32, width, height, depth, channels);
    output = ggml_im2col_3d(test.ctx, kernel, input, channels, 1, 1, 1, 0, 0, 0, 1, 1, 1, GGML_TYPE_BF16);
    return !ggml_backend_supports_op(backend, output);
}

static bool run_layouts(ggml_backend_t backend, bool cuda) {
    bool ok = true;
    for (ggml_type type : {GGML_TYPE_F16, GGML_TYPE_F32}) {
        for (int layout = 0; layout < (cuda ? 3 : 2); ++layout) {
            const bool passed = test_layout(backend, type, layout);
            std::printf("%s layout=%d: %s\n", ggml_type_name(type), layout, passed ? "PASS" : "FAIL");
            ok &= passed;
        }
    }
    return ok;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: test-im2col-3d CPU|CUDA\n");
        return 1;
    }
    ggml_backend_load_all();
    auto reg = ggml_backend_reg_by_name(argv[1]);
    if (!reg || ggml_backend_reg_dev_count(reg) == 0) {
        std::fprintf(stderr, "Required backend %s is unavailable\n", argv[1]);
        return std::strcmp(argv[1], "CUDA") == 0 ? 77 : 1;
    }
    auto backend = ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, 0), nullptr);
    if (!backend) {
        return 1;
    }
    std::printf("backend=%s\n", ggml_backend_name(backend));
    bool ok = run_layouts(backend, std::strcmp(argv[1], "CUDA") == 0);
    if (std::strcmp(argv[1], "CUDA") == 0) {
        const bool guards = test_guards(backend);
        std::printf("unsupported types: %s\n", guards ? "PASS" : "FAIL");
        ok &= guards;
    }
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
