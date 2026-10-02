#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <vector>

static bool check(ggml_tensor * tensor, const std::vector<unsigned char> & expected, const char * name) {
    std::vector<unsigned char> actual(expected.size());
    ggml_backend_tensor_get(tensor, actual.data(), 0, actual.size());
    if (actual != expected) {
        fprintf(stderr, "FAIL: %s\n", name);
        return false;
    }
    printf("PASS: %s\n", name);
    return true;
}

int main(int argc, char ** argv) {
    if (argc > 1) {
        ggml_backend_load_all_from_path(argv[1]);
    } else {
        ggml_backend_load_all();
    }
    ggml_backend_t backend = ggml_backend_init_by_name("Vulkan0", nullptr);
    if (!backend) {
        fprintf(stderr, "SKIP: no Vulkan device\n");
        return 77;
    }

    ggml_init_params params = {16 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false), nullptr, true};
    ggml_context * ctx = ggml_init(params);
    constexpr size_t bytes = 96 * 1024 * 1024;
    constexpr size_t chunk = 256 * 1024;
    ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, bytes);
    ggml_tensor * view = ggml_view_1d(ctx, tensor, bytes - 256, 256);
    ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1024);
    ggml_tensor * output = ggml_scale(ctx, input, 2.0f);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        fprintf(stderr, "FAIL: tensor allocation\n");
        ggml_free(ctx);
        ggml_backend_free(backend);
        return 1;
    }

    bool ok = true;
    std::vector<unsigned char> expected(bytes), source(chunk);
    std::vector<float> values(1024), result(1024);
    for (unsigned epoch = 0; epoch < 3 && ok; epoch++) {
        for (size_t i = 0; i < values.size(); i++) {
            values[i] = float(i + epoch);
        }
        ggml_backend_tensor_set_async(backend, input, values.data(), 0, values.size() * sizeof(float));
        ok = ggml_backend_graph_compute_async(backend, graph) == GGML_STATUS_SUCCESS;
        // Leave the graph pending while uploads exhaust and reuse the 64 MiB arena.
        for (size_t off = 0; off < bytes; off += chunk) {
            for (size_t j = 0; j < chunk; j++) {
                source[j] = (j * 13 + (off / chunk) * 71 + epoch * 97) % 251;
            }
            memcpy(expected.data() + off, source.data(), chunk);
            ggml_backend_tensor_set_async(backend, tensor, source.data(), off, chunk);
        }
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
        for (size_t i = 0; i < result.size(); i++) {
            if (result[i] != values[i] * 2.0f) {
                fprintf(stderr, "FAIL: upload followed by graph execution\n");
                ok = false;
                break;
            }
        }
        ok = check(tensor, expected, "arena wrap, source reuse and graph execution") && ok;
    }

    constexpr size_t width = 132, rows = 1300, dst_stride = 256, src_stride = 148, offset = 12;
    source.resize(rows * src_stride);
    for (size_t i = 0; i < source.size(); i++) {
        source[i] = (i * 37) % 251;
    }
    for (size_t i = 0; i < rows; i++) {
        memcpy(expected.data() + 256 + offset + i * dst_stride, source.data() + i * src_stride, width);
    }
    ggml_backend_tensor_set_2d_async(backend, view, source.data(), offset, width, rows, dst_stride, src_stride);
    ggml_backend_synchronize(backend);
    ok = check(tensor, expected, "strided view upload and untouched padding") && ok;

    // An oversized upload must retain the synchronous fallback after a pending small upload.
    ggml_backend_tensor_set_async(backend, tensor, source.data(), 0, width);
    for (size_t i = 0; i < expected.size(); i++) {
        expected[i] = (i * 19) % 251;
    }
    ggml_backend_tensor_set_async(backend, tensor, expected.data(), 0, bytes);
    ggml_backend_synchronize(backend);
    ok = check(tensor, expected, "oversized fallback after staged upload") && ok;

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
