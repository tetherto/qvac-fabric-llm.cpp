// A full-attention decode layer as one dispatch (xdna-att-layer.h) against a
// CPU reference: the input's norm from the residual rows, q|gate, k and v,
// the q/k norms and rope, this position's K and V into the cache, the
// attention over the cache, the gate, attn_output, the residual and the FFN.
// One head group (the 0.8B's and 2B's eight query heads over two kv heads)
// and two (the 4B's and 9B's sixteen over four), at their residual widths.
//
// The reference is float, so it differs from the device by the int8
// activations, the bf16 q and attention and the f16 cache (nmse ~1e-4 on
// attn_output); a wrong head group or row reads as nmse ~1.
// The cache-reach check runs on the host first: a cache that ends inside the
// second head group's last chunk has to be refused.
// Skips (exit 0) when no NPU or no fused artifact is found.

#include "../xdna-att-layer.h"
#include "../xdna-att.h"
#include "../xdna-gemv.h"
#include "../xdna-runtime.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static constexpr int    HD       = 256;  // head dim
static constexpr int    GQA      = 4;    // query heads a kv head serves
static constexpr int    N_ROT    = 64;
static constexpr int    N_CTX    = 256;  // cache positions
static constexpr int    N_VALID  = 200;  // this position is the last of them
static constexpr float  EPS      = 1e-6f;
static constexpr float  SCALE    = 0.0625f;
static constexpr float  W_SCALE  = 0.02f;
static constexpr float  ROPE_B   = 1e7f;
static constexpr double NMSE_MAX = 1e-2;
// the 4B/9B's position stride: four kv heads, two head groups
static constexpr size_t REACH_POS_B = (size_t) 4 * HD * sizeof(ggml_fp16_t);

struct layer_case {
    const char * what;
    int          K, heads, F;
};

static const layer_case CASES[] = {
    { "0.8B layer, 8/2 heads", 1024, 8,  3584  },
    { "2B layer, 8/2 heads",   2048, 8,  6144  },
    { "4B layer, 16/4 heads",  2560, 16, 9216  },
    { "9B layer, 16/4 heads",  4096, 16, 12288 },
};

// A [K x N] Q4_K weight: N rows of K, and its dequantized values.
struct weight {
    ggml_tensor          t{};
    std::vector<uint8_t> q;
    std::vector<float>   f;
};

static void make_weight(weight & w, int K, int N, std::mt19937 & rng) {
    std::normal_distribution<float> nd(0.0f, W_SCALE);
    std::vector<float>              src((size_t) K * N);
    for (float & v : src) {
        v = nd(rng);
    }
    w.q.resize(ggml_row_size(GGML_TYPE_Q4_K, K) * N);
    ggml_quantize_chunk(GGML_TYPE_Q4_K, src.data(), w.q.data(), 0, N, K, nullptr);
    w.f.resize(src.size());
    ggml_get_type_traits(GGML_TYPE_Q4_K)->to_float(w.q.data(), w.f.data(), (int64_t) K * N);
    w.t.type  = GGML_TYPE_Q4_K;
    w.t.ne[0] = K;
    w.t.ne[1] = N;
    w.t.ne[2] = w.t.ne[3] = 1;
    w.t.nb[0]             = ggml_type_size(GGML_TYPE_Q4_K);
    w.t.nb[1]             = ggml_row_size(GGML_TYPE_Q4_K, K);
    w.t.nb[2] = w.t.nb[3] = w.t.nb[1] * N;
    w.t.data              = w.q.data();
}

// An f32 vector as a tensor, values around 1 (a norm's gamma).
struct vec {
    ggml_tensor        t{};
    std::vector<float> f;
};

static void make_gamma(vec & g, int n, std::mt19937 & rng) {
    std::normal_distribution<float> nd(1.0f, 0.2f);
    g.f.resize((size_t) n);
    for (float & v : g.f) {
        v = nd(rng);
    }
    g.t.type  = GGML_TYPE_F32;
    g.t.ne[0] = n;
    g.t.ne[1] = g.t.ne[2] = g.t.ne[3] = 1;
    g.t.nb[0]                         = sizeof(float);
    g.t.nb[1] = g.t.nb[2] = g.t.nb[3] = (size_t) n * sizeof(float);
    g.t.data                          = g.f.data();
}

static std::vector<float> matvec(const std::vector<float> & w, const float * x, int K, int N) {
    std::vector<float> y((size_t) N);
    for (int n = 0; n < N; n++) {
        double s = 0.0;
        for (int k = 0; k < K; k++) {
            s += (double) w[(size_t) n * K + k] * x[k];
        }
        y[(size_t) n] = (float) s;
    }
    return y;
}

static std::vector<float> rms_norm(const float * x, int n, const float * g) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) {
        ss += (double) x[i] * x[i];
    }
    const double       r = 1.0 / std::sqrt(ss / n + EPS);
    std::vector<float> y((size_t) n);
    for (int i = 0; i < n; i++) {
        y[(size_t) i] = (float) (x[i] * r * g[i]);
    }
    return y;
}

// neox rotation of the first N_ROT dims, pairs (i, i + N_ROT / 2)
static void rope(std::vector<float> & y, const std::vector<float> & c, const std::vector<float> & s) {
    for (int i = 0; i < N_ROT / 2; i++) {
        const float a = y[(size_t) i], b = y[(size_t) (i + N_ROT / 2)];
        y[(size_t) i]               = a * c[(size_t) i] - b * s[(size_t) i];
        y[(size_t) (i + N_ROT / 2)] = a * s[(size_t) i] + b * c[(size_t) i];
    }
}

static float f16(float v) {
    return ggml_fp16_to_fp32(ggml_fp32_to_fp16(v));
}

static double nmse(const float * ref, const float * got, size_t n) {
    double err = 0.0, sq = 0.0;
    for (size_t i = 0; i < n; i++) {
        if (!std::isfinite(got[i])) {
            return INFINITY;
        }
        err += ((double) ref[i] - got[i]) * ((double) ref[i] - got[i]);
        sq += (double) ref[i] * ref[i];
    }
    return sq > 0.0 ? err / sq : INFINITY;
}

struct layer_weights {
    weight wq, wk, wv, wo, gate, up, down;
    vec    gq, gk, attn_norm, post;
};

static void make_layer(layer_weights & w, const layer_case & c, std::mt19937 & rng) {
    const int kvh = c.heads / GQA;
    make_weight(w.wq, c.K, 2 * c.heads * HD, rng);
    make_weight(w.wk, c.K, kvh * HD, rng);
    make_weight(w.wv, c.K, kvh * HD, rng);
    make_weight(w.wo, c.heads * HD, c.K, rng);
    make_weight(w.gate, c.K, c.F, rng);
    make_weight(w.up, c.K, c.F, rng);
    make_weight(w.down, c.F, c.K, rng);
    make_gamma(w.gq, HD, rng);
    make_gamma(w.gk, HD, rng);
    make_gamma(w.attn_norm, c.K, rng);
    make_gamma(w.post, c.K, rng);
}

// What the layer leaves: the residual rows S (attn_output), A (h_attn), F
// (the FFN's output) and this position's K and V rows.
struct layer_out {
    std::vector<float> s, a, f, k, v;
};

// The softmax attention of one query head over the cache's first N_VALID
// positions of its kv head; `kc` / `vc` are the caches, every kv head a row.
static std::vector<float> attend(const std::vector<float> & q,
                                 const std::vector<float> & kc,
                                 const std::vector<float> & vc,
                                 int                        kvh,
                                 int                        g) {
    std::vector<double> sc((size_t) N_VALID);
    double              mx = -INFINITY;
    for (int j = 0; j < N_VALID; j++) {
        double d = 0.0;
        for (int i = 0; i < HD; i++) {
            d += (double) q[(size_t) i] * kc[((size_t) j * kvh + g) * HD + i];
        }
        sc[(size_t) j] = d;
        mx             = std::max(mx, d);
    }
    double              l = 0.0;
    std::vector<float>  o((size_t) HD, 0.0f);
    std::vector<double> od((size_t) HD, 0.0);
    for (int j = 0; j < N_VALID; j++) {
        const double p = std::exp(sc[(size_t) j] - mx);
        l += p;
        for (int i = 0; i < HD; i++) {
            od[(size_t) i] += p * vc[((size_t) j * kvh + g) * HD + i];
        }
    }
    for (int i = 0; i < HD; i++) {
        o[(size_t) i] = (float) (od[(size_t) i] / l);
    }
    return o;
}

static layer_out reference(const layer_weights &      w,
                           const layer_case &         c,
                           const std::vector<float> & h,
                           std::vector<float>         kc,
                           std::vector<float>         vc,
                           const std::vector<float> & cs,
                           const std::vector<float> & sn) {
    const int                kvh = c.heads / GQA;
    const std::vector<float> x   = rms_norm(h.data(), c.K, w.attn_norm.f.data());
    const std::vector<float> qg  = matvec(w.wq.f, x.data(), c.K, 2 * c.heads * HD);
    const std::vector<float> kk  = matvec(w.wk.f, x.data(), c.K, kvh * HD);
    const std::vector<float> vv  = matvec(w.wv.f, x.data(), c.K, kvh * HD);
    layer_out                out;
    const size_t             row = (size_t) (N_VALID - 1) * kvh * HD;
    for (int g = 0; g < kvh; g++) {
        std::vector<float> kr = rms_norm(kk.data() + (size_t) g * HD, HD, w.gk.f.data());
        rope(kr, cs, sn);
        for (int i = 0; i < HD; i++) {
            kc[row + (size_t) g * HD + i] = f16(kr[(size_t) i]);
            vc[row + (size_t) g * HD + i] = f16(vv[(size_t) g * HD + i]);
        }
    }
    out.k.assign(kc.begin() + (long) row, kc.begin() + (long) (row + (size_t) kvh * HD));
    out.v.assign(vc.begin() + (long) row, vc.begin() + (long) (row + (size_t) kvh * HD));
    std::vector<float> og((size_t) c.heads * HD);
    for (int hq = 0; hq < c.heads; hq++) {
        std::vector<float> q = rms_norm(qg.data() + (size_t) hq * 2 * HD, HD, w.gq.f.data());
        for (float & v : q) {
            v *= SCALE;
        }
        rope(q, cs, sn);
        const std::vector<float> o    = attend(q, kc, vc, kvh, hq / GQA);
        const float *            gate = qg.data() + (size_t) hq * 2 * HD + HD;
        for (int i = 0; i < HD; i++) {
            og[(size_t) hq * HD + i] = o[(size_t) i] / (1.0f + std::exp(-gate[i]));
        }
    }
    out.s = matvec(w.wo.f, og.data(), c.heads * HD, c.K);
    out.a.resize((size_t) c.K);
    for (int i = 0; i < c.K; i++) {
        out.a[(size_t) i] = h[(size_t) i] + out.s[(size_t) i];
    }
    const std::vector<float> x2 = rms_norm(out.a.data(), c.K, w.post.f.data());
    const std::vector<float> gt = matvec(w.gate.f, x2.data(), c.K, c.F);
    const std::vector<float> up = matvec(w.up.f, x2.data(), c.K, c.F);
    std::vector<float>       m((size_t) c.F);
    for (int i = 0; i < c.F; i++) {
        m[(size_t) i] = gt[(size_t) i] / (1.0f + std::exp(-gt[(size_t) i])) * up[(size_t) i];
    }
    out.f = matvec(w.down.f, m.data(), c.F, c.K);
    return out;
}

static std::vector<float> normal(size_t n, float sd, std::mt19937 & rng) {
    std::normal_distribution<float> nd(0.0f, sd);
    std::vector<float>              v(n);
    for (float & x : v) {
        x = nd(rng);
    }
    return v;
}

// The cache as f16 in a BO and as the floats it holds.
static bool fill_cache(xdna_buffer * bo, std::vector<float> & vals) {
    ggml_fp16_t * p = (ggml_fp16_t *) bo->data;
    for (size_t i = 0; i < vals.size(); i++) {
        p[i]    = ggml_fp32_to_fp16(vals[i]);
        vals[i] = ggml_fp16_to_fp32(p[i]);
    }
    return xdna_buffer_sync_to_device(bo);
}

static void read_f16(const xdna_buffer * bo, size_t off, size_t n, std::vector<float> & out) {
    const ggml_fp16_t * p = (const ggml_fp16_t *) ((const uint8_t *) bo->data + off);
    out.resize(n);
    for (size_t i = 0; i < n; i++) {
        out[i] = ggml_fp16_to_fp32(p[i]);
    }
}

static bool report(const char * what, const std::vector<float> & ref, const std::vector<float> & got) {
    const double e  = nmse(ref.data(), got.data(), ref.size());
    const bool   ok = e < NMSE_MAX;
    std::printf("  %-4s nmse %.3e%s\n", what, e, ok ? "" : "  <- over the bound");
    return ok;
}

static bool run_case(xdna_kernel_pool * pool, const layer_case & c, std::mt19937 & rng) {
    layer_weights w;
    make_layer(w, c, rng);
    xdna_att_layer_w lw;
    lw.wq              = &w.wq.t;
    lw.wk              = &w.wk.t;
    lw.wv              = &w.wv.t;
    lw.gq              = &w.gq.t;
    lw.gk              = &w.gk.t;
    lw.attn_norm       = &w.attn_norm.t;
    lw.wo              = &w.wo.t;
    lw.post            = &w.post.t;
    lw.gate            = &w.gate.t;
    lw.up              = &w.up.t;
    lw.down            = &w.down.t;
    xdna_att_layer * l = xdna_att_layer_create(pool, lw);
    if (!l) {
        std::printf("FAIL: %s: the layer does not build\n", c.what);
        return false;
    }

    const int     kvh   = c.heads / GQA;
    const size_t  pos_b = (size_t) kvh * HD * sizeof(ggml_fp16_t);
    xdna_device * dev   = pool->device;
    xdna_buffer * res   = xdna_buffer_alloc(dev, XDNA_RES_BYTES);
    xdna_buffer * kbo   = xdna_buffer_alloc(dev, (size_t) N_CTX * pos_b);
    xdna_buffer * vbo   = xdna_buffer_alloc(dev, (size_t) N_CTX * pos_b);
    bool          ok    = res && kbo && vbo && res->data && kbo->data && vbo->data;

    // h = F + A on entry; the cache's rows before this position
    std::vector<float> rf = normal((size_t) c.K, 1.0f, rng), ra = normal((size_t) c.K, 1.0f, rng);
    std::vector<float> kc = normal((size_t) N_CTX * kvh * HD, 1.0f, rng);
    std::vector<float> vc = normal((size_t) N_CTX * kvh * HD, 1.0f, rng);
    std::vector<float> h((size_t) c.K);
    for (int i = 0; i < c.K; i++) {
        h[(size_t) i] = rf[(size_t) i] + ra[(size_t) i];
    }
    std::vector<float> cs((size_t) N_ROT / 2), sn((size_t) N_ROT / 2);
    for (int i = 0; i < N_ROT / 2; i++) {
        const double th = (N_VALID - 1) * std::pow((double) ROPE_B, -2.0 * i / N_ROT);
        cs[(size_t) i]  = (float) std::cos(th);
        sn[(size_t) i]  = (float) std::sin(th);
    }
    if (ok) {
        std::memset(res->data, 0, XDNA_RES_BYTES);
        std::memcpy((uint8_t *) res->data + XDNA_RES_F, rf.data(), (size_t) c.K * sizeof(float));
        std::memcpy((uint8_t *) res->data + XDNA_RES_A, ra.data(), (size_t) c.K * sizeof(float));
        ok = xdna_buffer_sync_to_device(res) && fill_cache(kbo, kc) && fill_cache(vbo, vc);
    }
    xdna_att_layer_in in;
    in.res      = res;
    in.kbo      = kbo;
    in.vbo      = vbo;
    in.pos_b    = pos_b;
    in.n_valid  = N_VALID;
    in.n_rot    = N_ROT;
    in.cosv     = cs.data();
    in.sinv     = sn.data();
    in.scale    = SCALE;
    in.eps      = EPS;
    in.eps_attn = EPS;
    in.eps_post = EPS;
    ok          = ok && xdna_att_layer_run(l, in) && xdna_buffer_sync_from_device(res);
    if (!ok) {
        std::printf("FAIL: %s: the dispatch failed\n", c.what);
    } else {
        const layer_out ref = reference(w, c, h, kc, vc, cs, sn);
        layer_out       got;
        const float *   r = (const float *) res->data;
        got.s.assign(r + XDNA_RES_S / 4, r + XDNA_RES_S / 4 + c.K);
        got.a.assign(r + XDNA_RES_A / 4, r + XDNA_RES_A / 4 + c.K);
        got.f.assign(r + XDNA_RES_F / 4, r + XDNA_RES_F / 4 + c.K);
        read_f16(kbo, (size_t) (N_VALID - 1) * pos_b, (size_t) kvh * HD, got.k);
        read_f16(vbo, (size_t) (N_VALID - 1) * pos_b, (size_t) kvh * HD, got.v);
        std::printf("%s (K %d, F %d, %d heads):\n", c.what, c.K, c.F, c.heads);
        bool good = report("K", ref.k, got.k);
        good      = report("V", ref.v, got.v) && good;
        good      = report("S", ref.s, got.s) && good;
        good      = report("A", ref.a, got.a) && good;
        good      = report("F", ref.f, got.f) && good;
        ok        = good;
        std::printf("%s: %s\n", ok ? "ok  " : "FAIL", c.what);
    }
    xdna_buffer_free(res);
    xdna_buffer_free(kbo);
    xdna_buffer_free(vbo);
    xdna_att_layer_free(l);
    return ok;
}

// Whether the layer takes K and V buffers of `bytes` each for `groups` head
// groups at the 4B/9B's position stride.
static bool reach_fits(size_t bytes, int groups) {
    xdna_buffer kb, vb;
    kb.bytes = bytes;
    vb.bytes = bytes;
    return xdna_att_layer_fits(&kb, 0, &vb, 0, N_VALID, REACH_POS_B, groups);
}

// Head group g reads its last chunk up to `last` + (g + 1) * POS_B: a cache
// that ends below group 1's reach fits one group, not two.
static bool check_group_reach(void) {
    const size_t last   = ((size_t) xdna_att_geom::chunk_count(N_VALID) * xdna_att_geom::ROUND - 1) * REACH_POS_B;
    const size_t reach1 = last + xdna_att_geom::POS_B;
    const size_t reach2 = last + 2 * (size_t) xdna_att_geom::POS_B;
    const bool   ok =
        reach_fits(reach1, 1) && !reach_fits(reach1, 2) && !reach_fits(reach2 - 1, 2) && reach_fits(reach2, 2);
    std::printf("%s: a cache that ends between the head groups' reach is refused\n", ok ? "ok  " : "FAIL");
    return ok;
}

static int count_failures(xdna_kernel_pool * pool) {
    std::mt19937 rng(7);
    int          failures = 0;
    for (const layer_case & c : CASES) {
        failures += run_case(pool, c, rng) ? 0 : 1;
    }
    return failures;
}

int main(void) {
    if (!check_group_reach()) {
        std::printf("test-xdna-att-layer: 1 failure\n");
        return 1;
    }
    const std::string stem = xdna_gemv_stem(XDNA_GEMV_SPLIT_FUSED);
    if (xdna_artifact_find(stem.c_str(), false).xclbin.empty()) {
        std::printf("test-xdna-att-layer: no %s artifact, skipped\n", stem.c_str());
        return 0;
    }
    xdna_device * dev = xdna_device_open();
    if (!dev) {
        std::printf("test-xdna-att-layer: no NPU, skipped\n");
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
        std::printf("test-xdna-att-layer: %d failures\n", failures);
        return 1;
    }
    std::printf("test-xdna-att-layer: all passed\n");
    return 0;
}
