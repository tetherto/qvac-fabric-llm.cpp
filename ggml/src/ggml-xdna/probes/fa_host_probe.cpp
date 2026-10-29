// Host FLASH_ATTN_EXT in the shapes EmbeddingGemma 2 and Qwen3.5 build: ggml-cpu's op against the backend's own
// host kernel (xdna-host-attn.cpp), timing and accuracy, no NPU and no model needed.
//   fa_host_probe [--d D] [--heads H] [--kv KV] [--n N] [--ctx C] [--mask full|band|causal] [--window W]
//                 [--seqs S] [--scale s] [--sigma x] [--kvtype f16|f32] [--layout cast|cache]
//                 [--threads T] [--reps R] [--no-cpu] [--no-host] [--other-threads T2]
// --other-threads: run the host kernel again with a team of T2 and report whether the output is bit for bit the same.
// Build (from the build directory, next to the libraries; -fopenmp for the kernel's team):
//   g++ -O3 -march=native -fopenmp -std=c++17 -DGGML_USE_OPENMP -I<src>/ggml/include -I<src>/ggml/src \
//       -I<src>/ggml/src/ggml-xdna <src>/ggml/src/ggml-xdna/probes/fa_host_probe.cpp -o bin/fa_host_probe \
//       -Lbin -lggml -lggml-base -lggml-cpu -Wl,-rpath,'$ORIGIN'
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "../xdna-host-attn.cpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct stats {
    double min = 0, med = 0, max = 0;
};

static stats summarize(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return { v.front(), v[v.size() / 2], v.back() };
}

static double nmse(const std::vector<float> & ref, const std::vector<float> & got) {
    double num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        const double d = (double) ref[i] - (double) got[i];
        num += d * d;
        den += (double) ref[i] * (double) ref[i];
    }
    return den > 0 ? num / den : num;
}

int main(int argc, char ** argv) {
    int         D = 256, H = 4, KVH = 2, N = 912, ctx = 0, S = 1, W = 512, reps = 20, threads = 16;
    float       scale = 1.0f, sigma = 0.25f;
    std::string mask = "band", kvtype = "f16", layout = "cast";
    bool        no_cpu = false, no_host = false;
    int         other_threads = 0;
    for (int i = 1; i < argc; i++) {
        auto is = [&](const char * k) { return std::strcmp(argv[i], k) == 0 && i + 1 < argc; };
        if (is("--d")) D = atoi(argv[++i]);
        else if (is("--heads")) H = atoi(argv[++i]);
        else if (is("--kv")) KVH = atoi(argv[++i]);
        else if (is("--n")) N = atoi(argv[++i]);
        else if (is("--ctx")) ctx = atoi(argv[++i]);
        else if (is("--seqs")) S = atoi(argv[++i]);
        else if (is("--window")) W = atoi(argv[++i]);
        else if (is("--reps")) reps = atoi(argv[++i]);
        else if (is("--threads")) threads = atoi(argv[++i]);
        else if (is("--scale")) scale = (float) atof(argv[++i]);
        else if (is("--sigma")) sigma = (float) atof(argv[++i]);
        else if (is("--mask")) mask = argv[++i];
        else if (is("--kvtype")) kvtype = argv[++i];
        else if (is("--other-threads")) other_threads = atoi(argv[++i]);
        else if (is("--layout")) layout = argv[++i];
        else if (std::strcmp(argv[i], "--no-cpu") == 0) no_cpu = true;
        else if (std::strcmp(argv[i], "--no-host") == 0) no_host = true;
        else { std::fprintf(stderr, "bad arg %s\n", argv[i]); return 2; }
    }
    const ggml_type kt = kvtype == "f32" ? GGML_TYPE_F32 : GGML_TYPE_F16;

    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, threads);
    g_xdna_chunk_threads.store(threads);

    ggml_init_params ip  = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    ggml_context *   gc  = ggml_init(ip);
    const int64_t    n_kv = ctx + N;
    ggml_tensor *    qr  = ggml_new_tensor_3d(gc, GGML_TYPE_F32, D, H, N);
    ggml_tensor *    q   = ggml_permute(gc, qr, 0, 2, 1, 3);
    ggml_tensor *    kr, *vr, *k, *v;
    if (layout == "cache") {
        kr = ggml_new_tensor_3d(gc, kt, D, KVH, n_kv);
        vr = ggml_new_tensor_3d(gc, kt, D, KVH, n_kv);
        k  = ggml_permute(gc, kr, 0, 2, 1, 3);
        v  = ggml_permute(gc, vr, 0, 2, 1, 3);
    } else {
        kr = k = ggml_new_tensor_3d(gc, kt, D, n_kv, KVH);
        vr = v = ggml_new_tensor_3d(gc, kt, D, n_kv, KVH);
    }
    ggml_tensor * m = ggml_new_tensor_2d(gc, GGML_TYPE_F16, n_kv, N);
    ggml_tensor * out = ggml_flash_attn_ext(gc, q, k, v, m, scale, 0.0f, 0.0f);
    ggml_prec_set_acc(out, GGML_PREC_F32);
    ggml_cgraph * gf = ggml_new_graph(gc);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(gc, cpu);

    std::mt19937                    rng(77);
    std::normal_distribution<float> nd(0.0f, sigma);
    for (ggml_tensor * t : { qr, kr, vr }) {
        std::vector<float> f((size_t) ggml_nelements(t));
        for (float & x : f) {
            x = nd(rng);
        }
        if (t->type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(t, f.data(), 0, ggml_nbytes(t));
        } else {
            std::vector<ggml_fp16_t> h(f.size());
            ggml_fp32_to_fp16_row(f.data(), h.data(), (int64_t) f.size());
            ggml_backend_tensor_set(t, h.data(), 0, ggml_nbytes(t));
        }
    }
    double visible_pairs = 0;
    {
        const int                len = N / S;
        std::vector<ggml_fp16_t> mk((size_t) (n_kv * N), ggml_fp32_to_fp16(-INFINITY));
        for (int t = 0; t < N; t++) {
            const int st = std::min(t / len, S - 1);
            for (int64_t j = 0; j < n_kv; j++) {
                const bool is_ctx = j < ctx;
                const int  jt     = (int) j - ctx;
                const int  sj     = is_ctx ? 0 : std::min(jt / len, S - 1);
                if (!is_ctx && jt >= N) {
                    continue;
                }
                if (sj != st) {
                    continue;
                }
                const int pt = t - st * len + (st == 0 ? ctx : 0);
                const int pj = is_ctx ? (int) j : jt - sj * len + (sj == 0 ? ctx : 0);
                bool      vis = mask == "full" ? true : mask == "causal" ? pj <= pt : std::abs(pj - pt) <= W;
                if (vis) {
                    mk[(size_t) t * n_kv + j] = ggml_fp32_to_fp16(0.0f);
                    visible_pairs += 1;
                }
            }
        }
        ggml_backend_tensor_set(m, mk.data(), 0, ggml_nbytes(m));
    }

    const double triples = visible_pairs * H;
    std::printf("D %d heads %d/%d n %d ctx %d seqs %d mask %s window %d scale %g kv %s %s threads %d: visible %.3fM pairs, %.3fM triples, %.2f GFLOP\n",
                D, H, KVH, N, ctx, S, mask.c_str(), W, scale, kvtype.c_str(), layout.c_str(), threads, visible_pairs / 1e6,
                triples / 1e6, triples * 4.0 * D / 1e9);

    std::vector<float> ref;
    if (!no_cpu) {
        ggml_backend_graph_compute(cpu, gf);  // warm up: team, caches
        std::vector<double> ts;
        for (int r = 0; r < reps; r++) {
            const double t0 = now_us();
            ggml_backend_graph_compute(cpu, gf);
            ts.push_back(now_us() - t0);
        }
        const stats s = summarize(ts);
        std::printf("ggml-cpu : min %8.1f us  med %8.1f us  max %8.1f us   %.3f ns/triple (min)  %.0f GFLOP/s\n", s.min, s.med,
                    s.max, s.min * 1e3 / triples, triples * 4.0 * D / (s.min * 1e3));
        ref.resize((size_t) ggml_nelements(out));
        ggml_backend_tensor_get(out, ref.data(), 0, ggml_nbytes(out));
    }
    if (!no_host) {
        if (!xdna_host_attn_supported(out)) {
            std::printf("host kernel: shape not supported\n");
            return 1;
        }
        std::vector<float> got;
        for (int r = 0; r < 2; r++) {
            if (!xdna_host_attn_run(out)) {
                std::printf("host kernel: run failed\n");
                return 1;
            }
        }
        std::vector<double> ts;
        for (int r = 0; r < reps; r++) {
            const double t0 = now_us();
            xdna_host_attn_run(out);
            ts.push_back(now_us() - t0);
        }
        const stats s = summarize(ts);
        std::printf("host attn: min %8.1f us  med %8.1f us  max %8.1f us   %.3f ns/triple (min)  %.0f GFLOP/s\n", s.min, s.med,
                    s.max, s.min * 1e3 / triples, triples * 4.0 * D / (s.min * 1e3));
        got.resize((size_t) ggml_nelements(out));
        ggml_backend_tensor_get(out, got.data(), 0, ggml_nbytes(out));
        if (!ref.empty()) {
            std::printf("nmse vs ggml-cpu: %.3g\n", nmse(ref, got));
        }
        if (other_threads > 0) {
            g_xdna_chunk_threads.store(other_threads);
            xdna_host_attn_run(out);
            std::vector<float> again((size_t) ggml_nelements(out));
            ggml_backend_tensor_get(out, again.data(), 0, ggml_nbytes(out));
            std::printf("team of %d against %d: %s\n", other_threads, threads,
                        std::memcmp(got.data(), again.data(), got.size() * sizeof(float)) == 0 ? "bit for bit the same" : "DIFFERENT");
        }
    }
    ggml_backend_buffer_free(buf);
    ggml_free(gc);
    ggml_backend_free(cpu);
    return 0;
}
