#include "ggml.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-cuda/scale-residual.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
constexpr size_t context_bytes = 2 * 1024 * 1024;
constexpr int skip = 77;

struct fixture {
    ggml_context * ctx = ggml_init({context_bytes, nullptr, true});
    ggml_backend_buffer_type buffer_type = {};
    ggml_backend_buffer buffer = {};
    ggml_tensor * value;
    ggml_tensor * scale;
    ggml_tensor * product;
    ggml_tensor * dst;
    ggml_cgraph * graph;

    fixture(int64_t rows = 1, int64_t width = GGML_CUDA_H3_HIDDEN_SIZE) {
        value = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, rows);
        scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width);
        product = ggml_mul(ctx, value, scale);
        dst = ggml_add(ctx, value, product);
        graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, dst);
        buffer.buft = &buffer_type;
        value->data = reinterpret_cast<void *>(uintptr_t(0x00010000));
        product->data = reinterpret_cast<void *>(uintptr_t(0x20000000));
        dst->data = reinterpret_cast<void *>(uintptr_t(0x40000000));
        scale->data = reinterpret_cast<void *>(uintptr_t(0x60000000));
        for (ggml_tensor * tensor : {value, scale, product, dst}) {
            tensor->buffer = &buffer;
        }
    }

    ~fixture() { ggml_free(ctx); }

    bool selected(bool device = true, int warp_size = 32, int index = 0) {
        return ggml_cuda_can_fuse_scale_residual_gfx1151(device, warp_size, &buffer_type, graph, index);
    }
};

template<typename Change>
bool reject(const char * name, Change change) {
    fixture f;
    change(f);
    if (f.selected()) {
        std::fprintf(stderr, "Unexpected fusion: %s\n", name);
        return false;
    }
    return true;
}

bool check_guards() {
    fixture normal, maximum(GGML_CUDA_H3_MAX_VALIDATED_ROWS), excessive(GGML_CUDA_H3_MAX_VALIDATED_ROWS + 1);
    fixture wrong_width(1, GGML_CUDA_H3_HIDDEN_SIZE - 1);
    if (!normal.selected() || !maximum.selected() || excessive.selected() || wrong_width.selected() ||
        normal.selected(false) || normal.selected(true, 64) || normal.selected(true, 32, -1) ||
        normal.selected(true, 32, 1) || normal.selected(true, 32, 2)) {
        return false;
    }
    normal.dst->data = normal.value->data;
    if (!normal.selected()) { return false; }
    normal.dst->data = normal.product->data;
    if (!normal.selected()) { return false; }
    bool ok = true;
    ok = reject("extra consumer", [](fixture & f) {
        ggml_build_forward_expand(f.graph, ggml_sqr(f.ctx, f.product));
    }) && ok;
    ok = reject("observable product", [](fixture & f) { ggml_set_output(f.product); }) && ok;
    ok = reject("wrong operation", [](fixture & f) { f.product->op = GGML_OP_ADD; }) && ok;
    ok = reject("swapped add inputs", [](fixture & f) {
        f.dst->src[0] = f.product; f.dst->src[1] = f.value;
    }) && ok;
    ok = reject("extra input", [](fixture & f) { f.product->src[2] = f.scale; }) && ok;
    ok = reject("extra output input", [](fixture & f) { f.dst->src[2] = f.scale; }) && ok;
    ok = reject("batched value", [](fixture & f) { f.value->ne[2] = 2; }) && ok;
    ok = reject("broadcast shape", [](fixture & f) { f.scale->ne[1] = 2; }) && ok;
    ok = reject("zero rows", [](fixture & f) { f.value->ne[1] = 0; }) && ok;
    ok = reject("output shape", [](fixture & f) { f.dst->ne[0]--; }) && ok;
    ok = reject("non-F32", [](fixture & f) { f.scale->type = GGML_TYPE_F16; }) && ok;
    ok = reject("strided input", [](fixture & f) { f.value->nb[0] *= 2; }) && ok;
    ok = reject("singleton stride", [](fixture & f) { f.scale->nb[3] *= 2; }) && ok;
    ok = reject("unaligned", [](fixture & f) {
        f.value->data = reinterpret_cast<void *>(uintptr_t(f.value->data) + sizeof(float));
    }) && ok;
    ok = reject("unallocated", [](fixture & f) { f.dst->data = nullptr; }) && ok;
    ok = reject("missing buffer", [](fixture & f) { f.product->buffer = nullptr; }) && ok;
    ok = reject("different buffer type", [](fixture & f) { f.buffer.buft = nullptr; }) && ok;
    ok = reject("scale overwritten", [](fixture & f) { f.dst->data = f.scale->data; }) && ok;
    ok = reject("partial output alias", [](fixture & f) {
        f.dst->data = reinterpret_cast<void *>(uintptr_t(f.value->data) + 16);
    }) && ok;
    ok = reject("product aliases input", [](fixture & f) { f.product->data = f.value->data; }) && ok;
    ok = reject("product aliases scale", [](fixture & f) { f.product->data = f.scale->data; }) && ok;
    ok = reject("partial product alias", [](fixture & f) {
        f.product->data = reinterpret_cast<void *>(uintptr_t(f.dst->data) + 16);
    }) && ok;
    ok = reject("overflowing span", [](fixture & f) {
        f.dst->data = reinterpret_cast<void *>(UINTPTR_MAX - uintptr_t(15));
    }) && ok;
    return ok;
}

float reference(float value, float scale) {
    volatile float product = value * scale;
    volatile float sum = value + product;
    return sum;
}

std::vector<float> make_scale(int64_t width) {
    const float pattern[] = {0.0f, 1.0f, -1.0f, -0x1.fffffep-1f, 0.25f, -2.0f, 4097.0f};
    std::vector<float> scale(width);
    for (size_t i = 0; i < scale.size(); ++i) {
        scale[i] = pattern[i % (sizeof(pattern) / sizeof(pattern[0]))];
    }
    return scale;
}

std::vector<float> make_values(int64_t count) {
    const float pattern[] = {0.0f, -2.5f, 128.0f, 0x1.000002p0f, -128.0f, 3.25f, 4097.0f};
    std::vector<float> values(count);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = pattern[i % (sizeof(pattern) / sizeof(pattern[0]))];
    }
    return values;
}

bool compare(ggml_tensor * output, const std::vector<float> & values, const std::vector<float> & scale) {
    std::vector<float> actual(values.size());
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    for (size_t i = 0; i < actual.size(); ++i) {
        const float expected = reference(values[i], scale[i % scale.size()]);
        if (actual[i] != expected) {
            std::fprintf(stderr, "output[%zu]: expected %a, got %a\n", i, double(expected), double(actual[i]));
            return false;
        }
    }
    return true;
}

bool run_graph(ggml_backend_t backend, ggml_context * ctx, int64_t rows, int64_t width, bool inplace, bool extra_use) {
    auto * value = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, rows);
    auto * scale = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, width);
    auto * product = ggml_mul(ctx, value, scale);
    auto * dst = inplace ? ggml_add_inplace(ctx, value, product) : ggml_add(ctx, value, product);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, dst);
    ggml_tensor * extra = extra_use ? ggml_dup(ctx, product) : nullptr;
    if (extra) { ggml_build_forward_expand(graph, extra); }
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { return false; }
    const bool eligible = ggml_cuda_can_fuse_scale_residual_gfx1151(
        true, 32, ggml_backend_buffer_get_type(buffer), graph, 0);
    if (eligible != (width == GGML_CUDA_H3_HIDDEN_SIZE && !extra_use)) {
        std::fprintf(stderr, "Unexpected graph eligibility\n");
        ggml_backend_buffer_free(buffer);
        return false;
    }
    const auto values = make_values(ggml_nelements(value));
    const auto scales = make_scale(width);
    ggml_backend_tensor_set(value, values.data(), 0, values.size() * sizeof(float));
    ggml_backend_tensor_set(scale, scales.data(), 0, scales.size() * sizeof(float));
    bool ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS && compare(dst, values, scales);
    if (extra && ok) {
        std::vector<float> actual(values.size());
        ggml_backend_tensor_get(extra, actual.data(), 0, actual.size() * sizeof(float));
        for (size_t i = 0; i < actual.size(); ++i) {
            if (actual[i] != float(double(values[i]) * double(scales[i % scales.size()]))) { ok = false; break; }
        }
    }
    ggml_backend_buffer_free(buffer);
    return ok;
}

bool run_case(ggml_backend_t backend, int64_t rows, int64_t width, bool inplace, bool extra_use) {
    auto * ctx = ggml_init({context_bytes, nullptr, true});
    if (!ctx) { return false; }
    const bool ok = run_graph(backend, ctx, rows, width, inplace, extra_use);
    ggml_free(ctx);
    std::printf("rows=%lld width=%lld inplace=%d extra_use=%d: %s\n",
                (long long) rows, (long long) width, inplace, extra_use, ok ? "PASS" : "FAIL");
    return ok;
}

bool run_cases(ggml_backend_t backend) {
    bool ok = true;
    for (int64_t rows : {1, 3, 65}) {
        ok = run_case(backend, rows, GGML_CUDA_H3_HIDDEN_SIZE, false, false) && ok;
        ok = run_case(backend, rows, GGML_CUDA_H3_HIDDEN_SIZE, true, false) && ok;
    }
    ok = run_case(backend, 3, GGML_CUDA_H3_HIDDEN_SIZE, false, true) && ok;
    ok = run_case(backend, 3, GGML_CUDA_H3_HIDDEN_SIZE - 1, false, false) && ok;
    return ok;
}
}

int main(int argc, char ** argv) {
    if (!check_guards()) { std::fprintf(stderr, "Fusion guard checks failed\n"); return 1; }
    const float value = 0x1.000002p0f, scale = -0x1.fffffep-1f;
    if (reference(value, scale) == std::fma(value, scale, value)) { return 1; }
    if (argc == 2 && std::strcmp(argv[1], "guards") == 0) { return 0; }
    ggml_backend_load_all();
    auto * device = ggml_backend_dev_by_name(argc == 2 ? argv[1] : "CPU");
    if (!device) { return skip; }
    auto * backend = ggml_backend_dev_init(device, nullptr);
    if (!backend) { return 1; }
    const bool ok = run_cases(backend);
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
