// Standalone probe: times the host CPU cost of the vocabulary-projection
// (lm_head) MUL_MAT in isolation, using the real tensor shape/type from
// unsloth-Q4_K_M.gguf's tied embedding: token_embd.weight, Q6_K [1024, 248320].
// This is the op the XDNA backend routes to host because N (248320) exceeds
// what the GEMV/GEMM routes can carry (14336 / 16384). Random data is used
// (matmul cost depends on shape/type, not values -- same approach
// test-backend-ops perf mode uses).
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>

int main(int argc, char ** argv) {
    int n_threads = argc > 1 ? atoi(argv[1]) : 4;
    int n_reps    = argc > 2 ? atoi(argv[2]) : 30;
    const int n_embd  = 1024;
    const int n_vocab = 248320;

    struct ggml_init_params params = { (size_t) 4 * 1024 * 1024, nullptr, /*no_alloc*/ true };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, n_embd, n_vocab);
    struct ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    struct ggml_tensor * y = ggml_mul_mat(ctx, w, x);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, n_threads);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { fprintf(stderr, "alloc failed\n"); return 1; }

    // fill with pseudo-random bytes / values
    srand(1234);
    {
        size_t wbytes = ggml_nbytes(w);
        std::vector<unsigned char> tmp(wbytes);
        for (size_t i = 0; i < wbytes; i++) tmp[i] = (unsigned char) rand();
        ggml_backend_tensor_set(w, tmp.data(), 0, wbytes);
    }
    {
        std::vector<float> tmp(n_embd);
        for (int i = 0; i < n_embd; i++) tmp[i] = ((rand() % 2000) - 1000) / 1000.0f;
        ggml_backend_tensor_set(x, tmp.data(), 0, n_embd * sizeof(float));
    }

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    // warm-up (first call pays thread-pool spin-up)
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_graph_compute(backend, gf);

    std::vector<double> samples(n_reps);
    for (int i = 0; i < n_reps; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ggml_backend_graph_compute(backend, gf);
        auto t1 = std::chrono::high_resolution_clock::now();
        samples[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    double sum = 0, mn = 1e18, mx = 0;
    for (double v : samples) { sum += v; if (v < mn) mn = v; if (v > mx) mx = v; }
    double avg = sum / n_reps;

    printf("threads=%d n_embd=%d n_vocab=%d reps=%d avg_ms=%.4f min_ms=%.4f max_ms=%.4f nbytes_w=%zu\n",
           n_threads, n_embd, n_vocab, n_reps, avg, mn, mx, ggml_nbytes(w));

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return 0;
}
