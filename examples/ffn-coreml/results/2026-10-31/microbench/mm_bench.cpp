// F16 x F32 matmul throughput on Metal: mm_bench <m> <n> <k>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-metal.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    const int m = atoi(argv[1]), n = atoi(argv[2]), k = atoi(argv[3]);
    ggml_backend_t backend = ggml_backend_metal_init();
    ggml_init_params ip = { 8 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, k, m);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    ggml_tensor * c = ggml_mul_mat(ctx, a, b);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(ga, gf);
    std::vector<ggml_fp16_t> af(ggml_nelements(a), ggml_fp32_to_fp16(0.01f));
    ggml_backend_tensor_set(a, af.data(), 0, ggml_nbytes(a));
    std::vector<float> bf(ggml_nelements(b), 0.01f);
    ggml_backend_tensor_set(b, bf.data(), 0, ggml_nbytes(b));
    ggml_backend_graph_compute(backend, gf);
    std::vector<double> t;
    for (int r = 0; r < 7; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(backend, gf);
        t.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(t.begin(), t.end());
    printf("m=%d n=%d k=%d median_ms=%.3f tflops=%.2f\n", m, n, k, t[3] * 1e3, 2.0 * m * n * k / t[3] / 1e12);
}
