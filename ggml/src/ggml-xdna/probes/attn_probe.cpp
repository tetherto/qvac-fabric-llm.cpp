// FLASH_ATTN_EXT on the XDNA backend against the CPU backend, on random data in the
// shapes llama builds, with timings. No model needed.
//   attn_probe [--heads H] [--kv KV] [--d D] [--n N] [--seqs S] [--mask causal|none|band]
//              [--window W] [--scale s] [--sigma x] [--layout cache|cast] [--reps R] [--threads T]
//              [--ctx C] [--pad P] [--proj L] [--no-fa] [--skip-cpu] [--dump file]
// --proj L: L layers of attention then a Q8_0 projection (the attention's output width), to time what the
// two kernels cost when they alternate; --no-fa leaves the attention out of that chain.
// --ctx: C keys of earlier context before the tokens (causal); --pad: the keys round up to a multiple of P, hidden.
// Build (from the build directory, next to the libraries):
//   g++ -O2 -std=c++17 -I<src>/ggml/include -I<src>/ggml/src probes/attn_probe.cpp -o bin/attn_probe \
//       -Lbin -lggml -lggml-base -lggml-cpu -Wl,-rpath,'$ORIGIN'
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char ** argv) {
    int         H = 8, KV = 2, D = 256, N = 512, S = 1, W = 512, reps = 5, threads = 16, nctx = 0, pad = 1;
    float       scale = 0.0625f, sigma = 1.0f;
    std::string mask = "causal", layout = "cache";
    bool        skip_cpu = false, no_fa = false;
    int         proj = 0;
    std::string dump;
    for (int i = 1; i < argc; i++) {
        auto is = [&](const char * k) { return std::strcmp(argv[i], k) == 0 && i + 1 < argc; };
        if (is("--heads")) H = atoi(argv[++i]);
        else if (is("--kv")) KV = atoi(argv[++i]);
        else if (is("--d")) D = atoi(argv[++i]);
        else if (is("--n")) N = atoi(argv[++i]);
        else if (is("--seqs")) S = atoi(argv[++i]);
        else if (is("--window")) W = atoi(argv[++i]);
        else if (is("--reps")) reps = atoi(argv[++i]);
        else if (is("--threads")) threads = atoi(argv[++i]);
        else if (is("--scale")) scale = (float) atof(argv[++i]);
        else if (is("--sigma")) sigma = (float) atof(argv[++i]);
        else if (is("--mask")) mask = argv[++i];
        else if (is("--layout")) layout = argv[++i];
        else if (is("--dump")) dump = argv[++i];
        else if (is("--ctx")) nctx = atoi(argv[++i]);
        else if (is("--pad")) pad = atoi(argv[++i]);
        else if (is("--proj")) proj = atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--no-fa") == 0) no_fa = true;
        else if (std::strcmp(argv[i], "--skip-cpu") == 0) skip_cpu = true;
        else { std::fprintf(stderr, "bad arg %s\n", argv[i]); return 2; }
    }

    ggml_backend_load_all();
    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        if (std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(ggml_backend_dev_get(i))), "XDNA") == 0) {
            dev = ggml_backend_dev_get(i);
        }
    }
    if (!dev) { std::printf("no XDNA device\n"); return 0; }
    ggml_backend_t npu = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_cpu_set_n_threads(cpu, threads);

    // the graph
    ggml_init_params ip = { ggml_tensor_overhead() * 256 + ggml_graph_overhead(), nullptr, true };
    ggml_context *   ctx = ggml_init(ip);
    const int64_t    n_kv = (int64_t) (nctx + N + pad - 1) / pad * pad;
    ggml_tensor *    qr = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, H, N);
    ggml_tensor *    q  = ggml_permute(ctx, qr, 0, 2, 1, 3);
    ggml_tensor *    kr, *vr, *k, *v;
    if (layout == "cache") {
        kr = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, KV, n_kv);
        vr = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, KV, n_kv);
        k  = ggml_permute(ctx, kr, 0, 2, 1, 3);
        v  = ggml_permute(ctx, vr, 0, 2, 1, 3);
    } else {
        kr = k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, n_kv, KV);
        vr = v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, n_kv, KV);
    }
    ggml_tensor * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, N);
    ggml_tensor * o = ggml_flash_attn_ext(ctx, q, k, v, m, scale, 0.0f, 0.0f);
    ggml_prec_set_acc(o, GGML_PREC_F32);
    std::vector<ggml_tensor *> weights;
    ggml_tensor *              chain = o;
    for (int l = 0; l < proj; l++) {
        ggml_tensor * x  = l == 0 ? ggml_reshape_2d(ctx, qr, (int64_t) D * H, N) : chain;
        ggml_tensor * qq = ggml_permute(ctx, ggml_reshape_3d(ctx, x, D, H, N), 0, 2, 1, 3);
        ggml_tensor * fa = no_fa ? qq : ggml_flash_attn_ext(ctx, qq, k, v, m, scale, 0.0f, 0.0f);
        if (!no_fa) ggml_prec_set_acc(fa, GGML_PREC_F32);
        ggml_tensor * y = no_fa ? x : ggml_reshape_2d(ctx, fa, (int64_t) D * H, N);
        weights.push_back(ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, (int64_t) D * H, (int64_t) D * H));
        chain = ggml_mul_mat(ctx, weights.back(), y);
    }
    // a projection of N tokens beside the attention: the backend sizes and profiles a graph by its largest one
    ggml_tensor * dw = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    ggml_tensor * dx = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, N);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, chain);
    ggml_build_forward_expand(gf, ggml_mul_mat(ctx, dw, dx));
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);

    // data
    std::mt19937                    rng(7);
    std::normal_distribution<float> nd(0.0f, sigma);
    auto fill_f32 = [&](ggml_tensor * t) {
        std::vector<float> f(ggml_nelements(t));
        for (float & x : f) x = nd(rng);
        ggml_backend_tensor_set(t, f.data(), 0, ggml_nbytes(t));
    };
    auto fill_f16 = [&](ggml_tensor * t) {
        std::vector<ggml_fp16_t> f(ggml_nelements(t));
        for (auto & x : f) x = ggml_fp32_to_fp16(nd(rng));
        ggml_backend_tensor_set(t, f.data(), 0, ggml_nbytes(t));
    };
    fill_f32(dw);
    fill_f32(dx);
    fill_f32(qr);
    fill_f16(kr);
    fill_f16(vr);
    for (ggml_tensor * w : weights) {
        std::vector<float> f((size_t) ggml_nelements(w));
        std::normal_distribution<float> wd(0.0f, 0.03f);
        for (float & x : f) x = wd(rng);
        std::vector<uint8_t> qv(ggml_nbytes(w));
        ggml_quantize_chunk(w->type, f.data(), qv.data(), 0, w->ne[1], w->ne[0], nullptr);
        ggml_backend_tensor_set(w, qv.data(), 0, qv.size());
    }
    std::vector<int> seq(N), pos(N);
    for (int s = 0, t = 0; s < S; s++) {
        const int len = s + 1 < S ? N / S : N - t;
        for (int i = 0; i < len; i++, t++) { seq[t] = s; pos[t] = i; }
    }
    std::vector<ggml_fp16_t> mk((size_t) n_kv * N);
    const ggml_fp16_t        zero = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
    // key j < nctx is context (sequence 0, position j); j - nctx a token
    auto key_seq = [&](int64_t j) { return j < nctx ? 0 : j - nctx < N ? seq[j - nctx] : -1; };
    auto key_pos = [&](int64_t j) { return j < nctx ? (int) j : j - nctx < N ? pos[j - nctx] + (seq[j - nctx] == 0 ? nctx : 0) : 0; };
    for (int t = 0; t < N; t++) {
        const int pt = pos[t] + (seq[t] == 0 ? nctx : 0);
        for (int64_t j = 0; j < n_kv; j++) {
            bool vis = key_seq(j) == seq[t];
            if (mask == "causal") vis = vis && key_pos(j) <= pt;
            if (mask == "band") vis = vis && std::abs(key_pos(j) - pt) <= W;
            mk[(size_t) t * n_kv + j] = vis ? zero : ninf;
        }
    }
    ggml_backend_tensor_set(m, mk.data(), 0, mk.size() * sizeof(ggml_fp16_t));

    auto compute = [&](ggml_backend_t be, int n, std::vector<double> & ms) {
        for (int i = 0; i < n; i++) {
            const double t0 = now_ms();
            if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
                std::printf("compute failed\n");
                std::exit(1);
            }
            ms.push_back(now_ms() - t0);
        }
    };
    auto median = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };

    std::vector<float>  ref(ggml_nelements(chain)), got(ggml_nelements(chain));
    std::vector<double> ms_cpu, ms_npu;
    if (!skip_cpu) {
        compute(cpu, 1, ms_cpu);
        ggml_backend_tensor_get(chain, ref.data(), 0, ggml_nbytes(chain));
        ms_cpu.clear();
        compute(cpu, reps, ms_cpu);
    }
    // the output is cleared so a node the NPU left unwritten shows
    std::vector<float> zeros(ggml_nelements(chain), 0.0f);
    ggml_backend_tensor_set(chain, zeros.data(), 0, ggml_nbytes(chain));
    compute(npu, 1, ms_npu);
    ggml_backend_tensor_get(chain, got.data(), 0, ggml_nbytes(chain));
    if (!dump.empty()) {
        if (FILE * f = std::fopen(dump.c_str(), "wb")) {
            std::fwrite(got.data(), sizeof(float), got.size(), f);
            std::fclose(f);
        }
    }
    ms_npu.clear();
    compute(npu, reps, ms_npu);

    double err = 0, sq = 0, mx = 0;
    size_t bad = 0;
    for (size_t i = 0; !skip_cpu && i < ref.size(); i++) {
        if (!std::isfinite(got[i])) { bad++; continue; }
        const double d = (double) ref[i] - got[i];
        err += d * d;
        sq += (double) ref[i] * ref[i];
        mx = std::max(mx, std::fabs(d));
    }
    // pairs: visible (query, key) count a head
    double pairs = 0;
    for (int t = 0; t < N; t++) for (int64_t j = 0; j < n_kv; j++) pairs += ggml_fp16_to_fp32(mk[(size_t) t * n_kv + j]) == 0.0f;
    std::printf("H %d KV %d D %d N %d seqs %d mask %s%s scale %g sigma %g layout %s | visible pairs/head %.0f\n", H, KV, D, N,
                S, mask.c_str(), mask == "band" ? ("/" + std::to_string(W)).c_str() : "", scale, sigma, layout.c_str(),
                pairs);
    if (!skip_cpu) {
        std::printf("  nmse %.3e  max abs err %.3e  non-finite %zu\n", sq > 0 ? err / sq : 0.0, mx, bad);
    }
    std::printf("  npu %.3f ms (median of %d)", median(ms_npu), reps);
    if (!skip_cpu) std::printf("   cpu(%d thr) %.3f ms", threads, median(ms_cpu));
    std::printf("\n  npu ns per visible (query,key) pair per head: %.4f\n", median(ms_npu) * 1e6 / (pairs * H));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(npu);
    ggml_backend_free(cpu);
    return 0;
}
