// Prefill flash attention microbench for the Qwen3.8-27B full-attention layer shape:
// 512 queries, 24 query heads, 4 KV heads, head size 256, F16 KV, causal mask over the last 512 positions.
// usage: fa_bench <kv> [reps] [nq]
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-metal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main(int argc, char ** argv) {
    const int kv = argc > 1 ? atoi(argv[1]) : 4096;
    const int reps = argc > 2 ? atoi(argv[2]) : 10;
    const int nq = argc > 3 ? atoi(argv[3]) : 512;
    const int D = 256, H = 24, HKV = 4;

    ggml_backend_t backend = ggml_backend_metal_init();
    ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, nq, H, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, kv, HKV, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, kv, HKV, 1);
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv, nq, 1, 1);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f / 16.0f, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(ga, gf);

    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> qf(ggml_nelements(q));
    for (auto & x : qf) { x = nd(rng); }
    ggml_backend_tensor_set(q, qf.data(), 0, ggml_nbytes(q));
    std::vector<ggml_fp16_t> kf(ggml_nelements(k));
    for (auto & x : kf) { x = ggml_fp32_to_fp16(nd(rng)); }
    ggml_backend_tensor_set(k, kf.data(), 0, ggml_nbytes(k));
    for (auto & x : kf) { x = ggml_fp32_to_fp16(nd(rng)); }
    ggml_backend_tensor_set(v, kf.data(), 0, ggml_nbytes(v));
    std::vector<ggml_fp16_t> mf(ggml_nelements(m));
    for (int i = 0; i < nq; ++i) {
        for (int j = 0; j < kv; ++j) {
            mf[(size_t) i * kv + j] = ggml_fp32_to_fp16(j <= kv - nq + i ? 0.0f : -INFINITY);
        }
    }
    ggml_backend_tensor_set(m, mf.data(), 0, ggml_nbytes(m));

    ggml_backend_graph_compute(backend, gf);
    std::vector<double> t;
    for (int r = 0; r < reps; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(backend, gf);
        t.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(t.begin(), t.end());
    const double med = t[t.size() / 2];
    const double flops = 4.0 * D * nq * H * (double) kv;
    std::vector<float> o(ggml_nelements(out));
    ggml_backend_tensor_get(out, o.data(), 0, ggml_nbytes(out));
    double sum = 0;
    for (float x : o) { sum += x; }
    printf("kv=%d nq=%d median_ms=%.3f tflops=%.2f checksum=%.6f\n", kv, nq, med * 1e3, flops / med / 1e12, sum);
    if (argc > 4) {
        // reference file: compare when it exists, otherwise write it
        FILE * fr = fopen(argv[4], "rb");
        if (fr) {
            std::vector<float> ref(o.size());
            const size_t n = fread(ref.data(), sizeof(float), ref.size(), fr);
            fclose(fr);
            double maxd = 0, maxr = 0;
            for (size_t i = 0; i < n; ++i) {
                maxd = std::max(maxd, (double) std::fabs(o[i] - ref[i]));
                maxr = std::max(maxr, (double) std::fabs(ref[i]));
            }
            printf("compare n=%zu max_abs_diff=%.6g max_abs_ref=%.6g\n", n, maxd, maxr);
        } else {
            FILE * fw = fopen(argv[4], "wb");
            fwrite(o.data(), sizeof(float), o.size(), fw);
            fclose(fw);
        }
    }
    return 0;
}
