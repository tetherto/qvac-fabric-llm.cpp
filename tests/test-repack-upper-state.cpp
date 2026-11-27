// Check that the x86 repack mat-mul kernels leave the upper halves of the YMM registers clean.
// Dirty upper state makes later SSE code on the same thread (e.g. libm exp) much slower on Intel CPUs.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)

#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

// XGETBV with ECX=1 returns XINUSE: bit 2 is 0 only when the upper YMM halves are in their initial (zero) state
static bool xinuse_supported() {
#if defined(_MSC_VER)
    int r[4];
    __cpuid(r, 0);
    if (r[0] < 0xD) {
        return false;
    }
    __cpuid(r, 1);
    if (!(r[2] & (1 << 27))) {
        return false;
    }
    __cpuidex(r, 0xD, 1);
    return (r[0] >> 2) & 1;
#else
    unsigned a, b, c, d;
    if (!__get_cpuid(1, &a, &b, &c, &d) || !(c & (1u << 27))) {
        return false;
    }
    if (!__get_cpuid_count(0xD, 1, &a, &b, &c, &d)) {
        return false;
    }
    return (a >> 2) & 1;
#endif
}

#if defined(__GNUC__)
__attribute__((target("xsave")))
#endif
static bool ymm_upper_in_use() {
    return (_xgetbv(1) >> 2) & 1;
}

#if defined(__GNUC__)
__attribute__((target("avx")))
#endif
static void ymm_upper_clear() {
    _mm256_zeroupper();
}

static ggml_backend_buffer_type_t find_repack_buft(ggml_backend_dev_t dev) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto get_extra_bufts = (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
    if (!get_extra_bufts) {
        return nullptr;
    }
    for (ggml_backend_buffer_type_t * buft = get_extra_bufts(dev); buft && *buft; ++buft) {
        if (strcmp(ggml_backend_buft_name(*buft), "CPU_REPACK") == 0) {
            return *buft;
        }
    }
    return nullptr;
}

// returns true if the upper YMM halves are still in use after a mul_mat of type x n_cols
static bool run_case(ggml_backend_t backend, ggml_backend_buffer_type_t repack_buft, ggml_type type, int n_cols) {
    const int k = 256;
    const int m = 64;

    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 8 + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx_w = ggml_init(params);
    ggml_context * ctx   = ggml_init(params);

    ggml_tensor * w   = ggml_new_tensor_2d(ctx_w, type, k, m);
    ggml_tensor * x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n_cols);
    ggml_tensor * out = ggml_mul_mat(ctx, w, x);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf_w = ggml_backend_alloc_ctx_tensors_from_buft(ctx_w, repack_buft);
    ggml_backend_buffer_t buf   = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buf_w && buf);

    std::vector<float> data((size_t) k * m);
    for (auto & v : data) {
        v = (float) rand() / (float) RAND_MAX - 0.5f;
    }
    std::vector<uint8_t> q(ggml_nbytes(w));
    ggml_quantize_chunk(type, data.data(), q.data(), 0, m, k, nullptr);
    ggml_backend_tensor_set(w, q.data(), 0, q.size());
    ggml_backend_tensor_set(x, data.data(), 0, sizeof(float) * k * n_cols);

    ymm_upper_clear();
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
    const bool in_use = ymm_upper_in_use();

    ggml_backend_buffer_free(buf);
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx);
    ggml_free(ctx_w);

    return in_use;
}

int main() {
    if (!xinuse_supported()) {
        printf("skip: XGETBV with ECX=1 is not supported\n");
        return 77;
    }

    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_buffer_type_t repack_buft = dev ? find_repack_buft(dev) : nullptr;
    if (!repack_buft) {
        printf("skip: no CPU_REPACK buffer type\n");
        return 77;
    }

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    GGML_ASSERT(backend);
    // one thread, so the kernel runs on this thread
    ggml_backend_cpu_set_n_threads(backend, 1);

    const ggml_type types[] = { GGML_TYPE_Q4_0, GGML_TYPE_IQ4_NL, GGML_TYPE_MXFP4 };
    const int n_cols[] = { 1, 4 }; // gemv and gemm

    int n_failed = 0;
    for (ggml_type type : types) {
        for (int n : n_cols) {
            const bool in_use = run_case(backend, repack_buft, type, n);
            printf("%-7s n=%d: upper YMM state %s\n", ggml_type_name(type), n, in_use ? "in use - FAIL" : "clean - OK");
            n_failed += in_use;
        }
    }

    ggml_backend_free(backend);

    return n_failed == 0 ? 0 : 1;
}

#else

int main() {
    printf("skip: not x86-64\n");
    return 77;
}

#endif
