// QVAC-21914: numerical parity test for the ggml-opencl flash-attention
// q-row chunking (ggml_cl_flash_attn). With GGML_OPENCL_FA_MAX_NQ=64 the
// chunked dispatch path engages cheaply; every case is computed on both the
// CPU backend and the OpenCL backend and compared within FA tolerance.
// Covers: unchunked (n_q <= max), exact multi-chunk, partial last chunk,
// n_q == 1 (dedicated kernel), masked (exercises the per-chunk mask-offset
// rewrite) and unmasked (the bidirectional vision-tower shape that crashes
// Adreno 830 at 16k patches without the fix).
//
// SKIPS (exit 0) when no OpenCL device is present - it runs where the
// backend exists (Adreno devices, desktop OpenCL, PoCL).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef _WIN32
static void set_env(const char * name, const char * value) { _putenv_s(name, value); }
#else
static void set_env(const char * name, const char * value) { setenv(name, value, 1); }
#endif

struct fa_case {
    int  n_q;
    bool mask;
};

// Deterministic LCG so both backends see identical inputs without seeding races.
static uint32_t g_rng;
static float frand() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return ((g_rng >> 8) & 0xffffff) / (float) 0x1000000 * 2.0f - 1.0f;  // [-1, 1)
}

static const int D    = 64;  // head size (dk == dv == 64 - supported on OpenCL FA)
static const int NH   = 4;   // heads (no GQA)
static const int N_KV = 256;

// Build + run one FA graph on `backend`; returns the output tensor data (f32).
static std::vector<float> run_case(ggml_backend_t backend, const fa_case & c, uint32_t seed) {
    ggml_init_params ip = {
        /*.mem_size   =*/ 64 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, c.n_q, NH, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, N_KV, NH, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, D, N_KV, NH, 1);
    ggml_tensor * m = c.mask ? ggml_new_tensor_4d(ctx, GGML_TYPE_F16, N_KV, c.n_q, 1, 1) : nullptr;

    ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f / std::sqrt((float) D), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buf != nullptr);

    // Identical data on every backend: reset the PRNG per tensor fill.
    g_rng = seed;
    {
        std::vector<float> qd((size_t) D * c.n_q * NH);
        for (auto & x : qd) x = frand();
        ggml_backend_tensor_set(q, qd.data(), 0, qd.size() * sizeof(float));

        std::vector<ggml_fp16_t> h((size_t) D * N_KV * NH);
        for (auto & x : h) x = ggml_fp32_to_fp16(frand());
        ggml_backend_tensor_set(k, h.data(), 0, h.size() * sizeof(ggml_fp16_t));
        for (auto & x : h) x = ggml_fp32_to_fp16(frand());
        ggml_backend_tensor_set(v, h.data(), 0, h.size() * sizeof(ggml_fp16_t));

        if (m) {
            // Causal-like deterministic pattern; row i sees k rows
            // j <= (i+1)*N_KV/n_q, so every q row keeps >= 1 visible key
            // (no all-masked softmax). Chunk boundaries land mid-pattern,
            // so a wrong per-chunk mask offset shows up immediately.
            std::vector<ggml_fp16_t> md((size_t) N_KV * c.n_q);
            for (int i = 0; i < c.n_q; i++) {
                const int visible = (int) (((int64_t) (i + 1) * N_KV) / c.n_q);
                for (int j = 0; j < N_KV; j++) {
                    md[(size_t) i * N_KV + j] = ggml_fp32_to_fp16(j <= visible ? 0.0f : -INFINITY);
                }
            }
            ggml_backend_tensor_set(m, md.data(), 0, md.size() * sizeof(ggml_fp16_t));
        }
    }

    const ggml_status st = ggml_backend_graph_compute(backend, gf);
    GGML_ASSERT(st == GGML_STATUS_SUCCESS);

    std::vector<float> res(ggml_nelements(out));
    ggml_backend_tensor_get(out, res.data(), 0, res.size() * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    double se = 0.0, ref = 1e-12;
    for (size_t i = 0; i < a.size(); i++) {
        const double d = (double) a[i] - (double) b[i];
        se  += d * d;
        ref += (double) a[i] * (double) a[i];
    }
    return se / ref;
}

int main() {
    // Must be set before the OpenCL context initializes: engage chunking at
    // tiny sizes and a small work budget so the intra-graph flush runs too.
    set_env("GGML_OPENCL_FA_MAX_NQ", "64");
    set_env("GGML_OPENCL_FLUSH_WORK_MB", "1");

    ggml_backend_load_all();

    ggml_backend_dev_t ocl_dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        if (std::strcmp(ggml_backend_reg_name(reg), "OpenCL") == 0) {
            ocl_dev = dev;
            break;
        }
    }
    if (!ocl_dev) {
        std::printf("no OpenCL device found - test skipped\n");
        return 0;
    }

    ggml_backend_t cpu = ggml_backend_dev_init(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr);
    ggml_backend_t ocl = ggml_backend_dev_init(ocl_dev, nullptr);
    GGML_ASSERT(cpu && ocl);

    // With MAX_NQ=64: 1 = dedicated n_q==1 kernel; 63/64 = unchunked
    // boundary; 65 = 64 + 1-row partial chunk; 128 = two exact chunks;
    // 135 = two full + 7-row partial.
    const fa_case cases[] = {
        {  1, false}, {  1, true},
        { 63, true},  { 64, true},
        { 65, false}, { 65, true},
        {128, true},
        {135, false}, {135, true},
    };

    int failures = 0;
    uint32_t seed = 0xC0FFEE;
    for (const auto & c : cases) {
        seed += 101;
        const std::vector<float> ref = run_case(cpu, c, seed);
        const std::vector<float> got = run_case(ocl, c, seed);
        const double err = nmse(ref, got);
        const bool ok = err < 5e-4;  // FA tolerance, matches test-backend-ops
        std::printf("%s n_q=%3d mask=%d nmse=%.3e\n", ok ? "ok  " : "FAIL", c.n_q, c.mask ? 1 : 0, err);
        if (!ok) failures++;
    }

    ggml_backend_free(ocl);
    ggml_backend_free(cpu);

    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("all ok\n");
    return 0;
}
