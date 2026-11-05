// The decode FFN as one GEMV pair on the device - gate and up with the silu
// epilogue, then the down projection on the tiles the first phase drained -
// against a CPU reference. The down projection runs in one pass (the 0.8B's
// 1024-wide FFN) and in two (the 2B's 2048-wide one, whose second phase
// replays its tiles and drains through one strided descriptor a stream). The
// 4B's and 9B's first phase runs past the descriptor file's chunk cap (18, 24).
//
// The reference quantizes the input and the silu product to int8 in groups of
// 32, as the host and the epilogue do, so it differs from the device only by
// rounding and, for Q6_K, by the 8-bit form's merged group scales (nmse ~1e-5):
// a broken pass reads as garbage, not as a small error.
// Skips the device cases (exit 0) when no NPU or no fused artifact is found.

#include "../xdna-gemv.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

static constexpr int    ACT_GROUP = 32;  // values one int8 scale covers
static constexpr float  CODE_MAX  = 127.0f;
static constexpr float  W_SCALE   = 0.02f;
static constexpr double NMSE_MAX  = 1e-4;  // against the int8 reference
static constexpr int    BOUND_K   = 2048;  // any K the fused split takes

struct pair_case {
    const char * what;
    int          K, F, N;
    ggml_type    down;
};

static const pair_case CASES[] = {
    { "0.8B FFN, down in one pass",                          1024, 3584,  1024, GGML_TYPE_Q6_K },
    { "2B FFN, Q4_K down in two passes",                     2048, 6144,  2048, GGML_TYPE_Q4_K },
    { "2B FFN, Q6_K down in two passes",                     2048, 6144,  2048, GGML_TYPE_Q6_K },
    { "4B FFN, 18-chunk gate|up, Q4_K down in three passes", 2560, 9216,  2560, GGML_TYPE_Q4_K },
    { "9B FFN, 24-chunk gate|up, Q6_K down in four passes",  4096, 12288, 4096, GGML_TYPE_Q6_K },
};

// A [K x N] ggml weight: N rows of K, quantized from normal values.
struct weight {
    ggml_tensor          t{};
    std::vector<uint8_t> q;
    std::vector<float>   f;  // the quantized values, dequantized
};

static void make_weight(weight & w, ggml_type type, int K, int N, std::mt19937 & rng) {
    std::normal_distribution<float> nd(0.0f, W_SCALE);
    std::vector<float>              src((size_t) K * N);
    for (float & v : src) {
        v = nd(rng);
    }
    w.q.resize(ggml_row_size(type, K) * N);
    ggml_quantize_chunk(type, src.data(), w.q.data(), 0, N, K, nullptr);
    w.f.resize(src.size());
    ggml_get_type_traits(type)->to_float(w.q.data(), w.f.data(), (int64_t) K * N);
    w.t.type  = type;
    w.t.ne[0] = K;
    w.t.ne[1] = N;
    w.t.ne[2] = w.t.ne[3] = 1;
    w.t.nb[0]             = ggml_type_size(type);
    w.t.nb[1]             = ggml_row_size(type, K);
    w.t.nb[2] = w.t.nb[3] = w.t.nb[1] * N;
    w.t.data              = w.q.data();
}

// x as the device holds it: int8 codes times their group's scale.
static std::vector<float> quantize_groups(const std::vector<float> & x) {
    std::vector<float> q(x.size());
    for (size_t g = 0; g < x.size(); g += ACT_GROUP) {
        float amax = 0.0f;
        for (int i = 0; i < ACT_GROUP; i++) {
            amax = std::max(amax, std::fabs(x[g + i]));
        }
        const float d = amax > 0.0f ? amax / CODE_MAX : 1.0f;
        for (int i = 0; i < ACT_GROUP; i++) {
            q[g + i] = std::nearbyint(x[g + i] / d) * d;
        }
    }
    return q;
}

// y = W x for a [K x N] weight of N rows.
static std::vector<float> matvec(const std::vector<float> & w, const std::vector<float> & x, int K, int N) {
    std::vector<float> y((size_t) N);
    for (int n = 0; n < N; n++) {
        double s = 0.0;
        for (int k = 0; k < K; k++) {
            s += (double) w[(size_t) n * K + k] * x[(size_t) k];
        }
        y[(size_t) n] = (float) s;
    }
    return y;
}

static std::vector<float> silu_mul(const std::vector<float> & g, const std::vector<float> & u) {
    std::vector<float> m(g.size());
    for (size_t i = 0; i < g.size(); i++) {
        m[i] = g[i] / (1.0f + std::exp(-g[i])) * u[i];
    }
    return m;
}

// normalized mean squared error over [b, e)
static double nmse(const std::vector<float> & ref, const std::vector<float> & got, size_t b, size_t e) {
    double err = 0.0, sq = 0.0;
    for (size_t i = b; i < e; i++) {
        if (!std::isfinite(got[i])) {
            return INFINITY;
        }
        err += ((double) ref[i] - got[i]) * ((double) ref[i] - got[i]);
        sq += (double) ref[i] * ref[i];
    }
    return sq > 0.0 ? err / sq : INFINITY;
}

// Every pass's `chunk` columns against the reference; the count of passes that miss.
static int count_bad_passes(const std::vector<float> & ref, const std::vector<float> & got, int N, int chunk) {
    int bad = 0;
    for (int b = 0; b < N; b += chunk) {
        const double e  = nmse(ref, got, (size_t) b, (size_t) std::min(N, b + chunk));
        const bool   ok = e < NMSE_MAX;
        std::printf("  pass %d: nmse %.3e%s\n", b / chunk, e, ok ? "" : "  <- over the bound");
        bad += ok ? 0 : 1;
    }
    return bad;
}

static bool run_case(xdna_kernel_pool * pool, const pair_case & c, std::mt19937 & rng) {
    weight wg, wu, wd;
    make_weight(wg, GGML_TYPE_Q4_K, c.K, c.F, rng);
    make_weight(wu, GGML_TYPE_Q4_K, c.K, c.F, rng);
    make_weight(wd, c.down, c.F, c.N, rng);

    // Uncapped, as the layers stage it (xdna-rec-gemv.cpp's pack_stage).
    const xdna_gemv_geom g1 = xdna_gemv_variant(GGML_TYPE_Q4_K, c.K, 2 * c.F, true, XDNA_GEMV_SPLIT_FUSED, true);
    const xdna_gemv_geom g2 = xdna_gemv_variant(c.down, c.F, c.N, false, XDNA_GEMV_SPLIT_FUSED, true);
    std::vector<int32_t> colmap;
    xdna_gemv_colmap(g1, colmap);
    const ggml_tensor *  gu[2] = { &wg.t, &wu.t };
    const ggml_tensor *  dn[1] = { &wd.t };
    std::vector<uint8_t> p1, p2;
    if (!g1.valid() || !g2.valid() || !xdna_gemv_pack_weights(g1, gu, 2, colmap.data(), p1) ||
        !xdna_gemv_pack_weights(g2, dn, 1, nullptr, p2)) {
        std::printf("FAIL: %s: the weights do not pack\n", c.what);
        return false;
    }
    std::unique_ptr<xdna_gemv_pair, void (*)(xdna_gemv_pair *)> pair(xdna_gemv_pair_create(pool, g1, p1, g2, p2),
                                                                     xdna_gemv_pair_free);
    if (!pair) {
        std::printf("FAIL: %s: the pair does not build\n", c.what);
        return false;
    }

    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float>              x((size_t) c.K);
    for (float & v : x) {
        v = nd(rng);
    }
    std::vector<float> got((size_t) c.N);
    if (!xdna_gemv_pair_run(pair.get(), x.data(), got.data())) {
        std::printf("FAIL: %s: the dispatch failed\n", c.what);
        return false;
    }

    const std::vector<float> xq  = quantize_groups(x);
    const std::vector<float> mid = quantize_groups(silu_mul(matvec(wg.f, xq, c.K, c.F), matvec(wu.f, xq, c.K, c.F)));
    const std::vector<float> ref = matvec(wd.f, mid, c.F, c.N);
    std::printf("%s (K %d, F %d, N %d, %s down, %d pass(es)):\n", c.what, c.K, c.F, c.N, ggml_type_name(c.down),
                g2.n_out());
    const bool ok = count_bad_passes(ref, got, c.N, g2.chunk()) == 0;
    std::printf("%s: %s\n", ok ? "ok  " : "FAIL", c.what);
    return ok;
}

// A fused Q4_K geometry of `n` output columns, uncapped as the layers stage it.
static xdna_gemv_geom fused_geom(int64_t n) {
    return xdna_gemv_variant(GGML_TYPE_Q4_K, BOUND_K, n, false, XDNA_GEMV_SPLIT_FUSED, true);
}

// Valid at xdna_gemv_max_uncapped_chunks chunks and not one chunk past it.
static bool check_uncapped_bound(void) {
    const xdna_gemv_geom one   = fused_geom(1);
    const int64_t        chunk = one.chunk();
    const int64_t        max   = xdna_gemv_max_uncapped_chunks();
    const bool           at    = fused_geom(max * chunk).valid();
    const bool           past  = fused_geom((max + 1) * chunk).valid();
    const bool           ok    = one.valid() && at && !past;
    std::printf("%s: uncapped bound %lld chunks (valid at it %d, past it %d)\n", ok ? "ok  " : "FAIL", (long long) max,
                at, past);
    return ok;
}

static int count_failures(xdna_kernel_pool * pool) {
    std::mt19937 rng(42);
    int          failures = 0;
    for (const pair_case & c : CASES) {
        failures += run_case(pool, c, rng) ? 0 : 1;
    }
    return failures;
}

int main(void) {
    if (!check_uncapped_bound()) {
        std::printf("test-xdna-pair: the uncapped bound is wrong\n");
        return 1;
    }
    const std::string stem = xdna_gemv_stem(XDNA_GEMV_SPLIT_FUSED);
    if (xdna_artifact_find(stem.c_str(), false).xclbin.empty()) {
        std::printf("test-xdna-pair: no %s artifact, skipped\n", stem.c_str());
        return 0;
    }
    xdna_device * dev = xdna_device_open();
    if (!dev) {
        std::printf("test-xdna-pair: no NPU, skipped\n");
        return 0;
    }
    xdna_kernel_pool * pool = new xdna_kernel_pool;
    pool->device            = dev;
    xdna_kernel_pool_scan(pool);

    const int failures = count_failures(pool);

    for (auto & kv : pool->kernels) {
        xdna_kernel_free(kv.second);
    }
    delete pool;
    delete dev;
    if (failures) {
        std::printf("test-xdna-pair: %d failures\n", failures);
        return 1;
    }
    std::printf("test-xdna-pair: all passed\n");
    return 0;
}
