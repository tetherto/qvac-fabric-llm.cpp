// Parity tests for the LavaSR denoiser ggml graph (denoiser_ggml.cpp) vs the scalar
// reference (denoiser_core.cpp), on CPU and an available GPU, without fixtures.

#include "lavasr/denoiser_ggml.h"
#include "backend_selection.h"
#include "lavasr/backend_requirement.h"

#include "lavasr/denoiser.h"      // internal: denoise_with_core / denoise_with_batch_core
#include "lavasr/denoiser_gguf.h" // internal: load_denoiser_gguf

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

using namespace tts_cpp::lavasr;

static int g_failures = 0;
#define CHECK(cond, msg)                                                         \
    do {                                                                         \
        if (!(cond)) {                                                           \
            ++g_failures;                                                        \
            std::fprintf(stderr, "FAIL: %s  (%s:%d)\n", msg, __FILE__, __LINE__);\
        }                                                                        \
    } while (0)

// Case name + fused mode, so a failure identifies which graph variant drifted.
static const char * fmsg(const char * name, bool fused) {
    static char b[160];
    std::snprintf(b, sizeof b, "%s [fused=%d]", name, (int) fused);
    return b;
}

// Scalar PyTorch GRU (gate order r,z,n), zero init state — a verbatim copy of
// denoiser_core.cpp gru_seq, the golden reference.  Wih:[3H,I], Whh:[3H,H].
static std::vector<float> ref_gru_seq(const std::vector<float> & x, int L, int I, int H,
                                      const std::vector<float> & Wih, const std::vector<float> & Whh,
                                      const std::vector<float> & Bih, const std::vector<float> & Bhh) {
    std::vector<float> h(H, 0.0f), ys((size_t) L * H, 0.0f), gi(3 * H), gh(3 * H);
    for (int t = 0; t < L; t++) {
        const float * xt = x.data() + (size_t) t * I;
        for (int o = 0; o < 3 * H; o++) {
            float acc = Bih[o];
            for (int i = 0; i < I; i++) acc += Wih[(size_t) o * I + i] * xt[i];
            gi[o] = acc;
        }
        for (int o = 0; o < 3 * H; o++) {
            float acc = Bhh[o];
            for (int i = 0; i < H; i++) acc += Whh[(size_t) o * H + i] * h[i];
            gh[o] = acc;
        }
        for (int k = 0; k < H; k++) {
            const float r = 1.0f / (1.0f + std::exp(-(gi[k] + gh[k])));
            const float z = 1.0f / (1.0f + std::exp(-(gi[H + k] + gh[H + k])));
            const float n = std::tanh(gi[2 * H + k] + r * gh[2 * H + k]);
            h[k]                  = (1.0f - z) * n + z * h[k];
            ys[(size_t) t * H + k] = h[k];
        }
    }
    return ys;
}

// Backward reference matching gru_bi's convention: run on reversed input, then
// index b[L-1-t] -> hidden at original position t.
static std::vector<float> ref_gru_rev(const std::vector<float> & x, int L, int I, int H,
                                      const std::vector<float> & Wih, const std::vector<float> & Whh,
                                      const std::vector<float> & Bih, const std::vector<float> & Bhh) {
    std::vector<float> xr((size_t) L * I);
    for (int t = 0; t < L; t++)
        for (int i = 0; i < I; i++) xr[(size_t) t * I + i] = x[(size_t) (L - 1 - t) * I + i];
    std::vector<float> b = ref_gru_seq(xr, L, I, H, Wih, Whh, Bih, Bhh);
    std::vector<float> y((size_t) L * H);
    for (int t = 0; t < L; t++)
        for (int k = 0; k < H; k++) y[(size_t) t * H + k] = b[(size_t) (L - 1 - t) * H + k];
    return y;
}

// Run detail::gru_batched on the CPU backend and return y flat [L][B][H]
// (index (l*B+b)*H+k), matching the ggml [H,B,L] memory layout.
static std::vector<float> run_batched(const std::vector<float> & x, int I, int B, int L, int H,
                                      const std::vector<float> & Wih, const std::vector<float> & Whh,
                                      const std::vector<float> & Bih, const std::vector<float> & Bhh,
                                      bool reverse, bool bidir,
                                      const std::vector<float> & WihR, const std::vector<float> & WhhR,
                                      const std::vector<float> & BihR, const std::vector<float> & BhhR,
                                      bool fused, const detail::GruSupport & gru_support = {},
                                      int * gru_nodes = nullptr) {
    ggml_init_params p{ (size_t) 128 * 1024 * 1024, nullptr, false };
    ggml_context *   ctx = ggml_init(p);
    ggml_tensor *    tx  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, I, B, L);
    ggml_tensor *    tWih = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, I, 3 * H);
    ggml_tensor *    tWhh = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, 3 * H);
    ggml_tensor *    tBih = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3 * H);
    ggml_tensor *    tBhh = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3 * H);
    std::memcpy(tx->data, x.data(), ggml_nbytes(tx));
    std::memcpy(tWih->data, Wih.data(), ggml_nbytes(tWih));
    std::memcpy(tWhh->data, Whh.data(), ggml_nbytes(tWhh));
    std::memcpy(tBih->data, Bih.data(), ggml_nbytes(tBih));
    std::memcpy(tBhh->data, Bhh.data(), ggml_nbytes(tBhh));

    ggml_tensor * y = detail::gru_batched(ctx, tx, tWih, tWhh, tBih, tBhh, reverse, fused, gru_support);
    if (bidir) {
        ggml_tensor * tWihR = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, I, 3 * H);
        ggml_tensor * tWhhR = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, 3 * H);
        ggml_tensor * tBihR = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3 * H);
        ggml_tensor * tBhhR = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3 * H);
        std::memcpy(tWihR->data, WihR.data(), ggml_nbytes(tWihR));
        std::memcpy(tWhhR->data, WhhR.data(), ggml_nbytes(tWhhR));
        std::memcpy(tBihR->data, BihR.data(), ggml_nbytes(tBihR));
        std::memcpy(tBhhR->data, BhhR.data(), ggml_nbytes(tBhhR));
        ggml_tensor * yr = detail::gru_batched(ctx, tx, tWihR, tWhhR, tBihR, tBhhR, true, fused, gru_support);
        y                = ggml_concat(ctx, y, yr, 0); // [2H, B, L]
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    if (gru_nodes) {
        *gru_nodes = 0;
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i)
            *gru_nodes += ggml_graph_node(gf, i)->op == GGML_OP_GRU;
    }
    CHECK(ggml_graph_compute_with_ctx(ctx, gf, 1) == GGML_STATUS_SUCCESS,
          "GRU graph computes successfully");
    std::vector<float> out((size_t) ggml_nelements(y));
    std::memcpy(out.data(), y->data, ggml_nbytes(y));
    ggml_free(ctx);
    return out;
}

static void rand_fill(std::vector<float> & v, std::mt19937 & rng) {
    std::uniform_real_distribution<float> d(-0.8f, 0.8f);
    for (float & x : v) x = d(rng);
}

static void test_gru_batched() {
    const int I = 3, H = 4, L = 5, B = 2;
    std::mt19937 rng(1234);
    std::vector<float> Wih(3 * H * I), Whh(3 * H * H), Bih(3 * H), Bhh(3 * H);
    std::vector<float> WihR(3 * H * I), WhhR(3 * H * H), BihR(3 * H), BhhR(3 * H);
    std::vector<float> x((size_t) L * B * I); // [L][B][I]
    rand_fill(Wih, rng); rand_fill(Whh, rng); rand_fill(Bih, rng); rand_fill(Bhh, rng);
    rand_fill(WihR, rng); rand_fill(WhhR, rng); rand_fill(BihR, rng); rand_fill(BhhR, rng);
    rand_fill(x, rng);

    auto x_of = [&](int b) {
        std::vector<float> xb((size_t) L * I);
        for (int l = 0; l < L; l++)
            for (int i = 0; i < I; i++) xb[(size_t) l * I + i] = x[((size_t) l * B + b) * I + i];
        return xb;
    };

    // Forward: y[(l*B+b)*H+k] must equal per-sequence ref_gru_seq.
    for (bool fused : { true, false }) {
        std::vector<float> y = run_batched(x, I, B, L, H, Wih, Whh, Bih, Bhh, false, false,
                                           {}, {}, {}, {}, fused);
        float m = 0.0f;
        for (int b = 0; b < B; b++) {
            std::vector<float> ref = ref_gru_seq(x_of(b), L, I, H, Wih, Whh, Bih, Bhh);
            for (int l = 0; l < L; l++)
                for (int k = 0; k < H; k++)
                    m = std::max(m, std::fabs(y[((size_t) l * B + b) * H + k] - ref[(size_t) l * H + k]));
        }
        CHECK(m < 1e-5f, fmsg("gru_batched forward matches scalar gru_seq", fused));
    }
    // Reverse: matches gru_bi's backward convention (hidden at original position).
    for (bool fused : { true, false }) {
        std::vector<float> y = run_batched(x, I, B, L, H, Wih, Whh, Bih, Bhh, true, false,
                                           {}, {}, {}, {}, fused);
        float m = 0.0f;
        for (int b = 0; b < B; b++) {
            std::vector<float> ref = ref_gru_rev(x_of(b), L, I, H, Wih, Whh, Bih, Bhh);
            for (int l = 0; l < L; l++)
                for (int k = 0; k < H; k++)
                    m = std::max(m, std::fabs(y[((size_t) l * B + b) * H + k] - ref[(size_t) l * H + k]));
        }
        CHECK(m < 1e-5f, fmsg("gru_batched reverse matches scalar backward", fused));
    }
    // BiGRU: concat(forward, reverse) along ne0 == scalar [fwd | bwd].
    for (bool fused : { true, false }) {
        std::vector<float> y = run_batched(x, I, B, L, H, Wih, Whh, Bih, Bhh, false, true,
                                           WihR, WhhR, BihR, BhhR, fused);
        float m = 0.0f;
        for (int b = 0; b < B; b++) {
            std::vector<float> f = ref_gru_seq(x_of(b), L, I, H, Wih, Whh, Bih, Bhh);
            std::vector<float> r = ref_gru_rev(x_of(b), L, I, H, WihR, WhhR, BihR, BhhR);
            for (int l = 0; l < L; l++) {
                for (int k = 0; k < H; k++) {
                    m = std::max(m, std::fabs(y[((size_t) l * B + b) * 2 * H + k] - f[(size_t) l * H + k]));
                    m = std::max(m, std::fabs(y[((size_t) l * B + b) * 2 * H + H + k] - r[(size_t) l * H + k]));
                }
            }
        }
        CHECK(m < 1e-5f, fmsg("gru_batched BiGRU concat matches scalar gru_bi", fused));
    }
}

// HTP may support a real GRU shape independently of its other fused ops. Check
// the queried descriptor, selected graph and scalar parity for both decisions.
static void test_gru_support_selection() {
    struct Shape { int I, H, B, L; };
    const Shape shapes[] = { {3, 1, 2, 4}, {2, 2, 1, 5}, {5, 3, 3, 7}, {4, 17, 2, 5},
                             {3, 65, 3, 3}, {3, 129, 2, 3} };
    std::mt19937 rng(20261028);
    int accepted = 0, rejected = 0;
    for (const Shape & s : shapes) {
        std::vector<float> x((size_t) s.I * s.B * s.L), Wih((size_t) 3 * s.H * s.I),
                           Whh((size_t) 3 * s.H * s.H), Bih(3 * s.H), Bhh(3 * s.H);
        rand_fill(x, rng); rand_fill(Wih, rng); rand_fill(Whh, rng);
        rand_fill(Bih, rng); rand_fill(Bhh, rng);
        for (bool reverse : {false, true}) {
            std::vector<float> reference((size_t) s.H * s.B * s.L);
            for (int b = 0; b < s.B; ++b) {
                std::vector<float> sequence((size_t) s.I * s.L);
                for (int t = 0; t < s.L; ++t)
                    for (int i = 0; i < s.I; ++i)
                        sequence[(size_t) t * s.I + i] = x[((size_t) t * s.B + b) * s.I + i];
                const auto expected = reverse ? ref_gru_rev(sequence, s.L, s.I, s.H, Wih, Whh, Bih, Bhh)
                                              : ref_gru_seq(sequence, s.L, s.I, s.H, Wih, Whh, Bih, Bhh);
                for (int t = 0; t < s.L; ++t)
                    for (int h = 0; h < s.H; ++h)
                        reference[((size_t) t * s.B + b) * s.H + h] = expected[(size_t) t * s.H + h];
            }
            auto parity = [&](const std::vector<float> & output) {
                CHECK(output.size() == reference.size(), "GRU support selection preserves output shape");
                float error = 0.0f;
                for (size_t i = 0; i < std::min(output.size(), reference.size()); ++i) {
                    CHECK(std::isfinite(output[i]), "GRU support selection preserves finite output");
                    error = std::max(error, std::fabs(output[i] - reference[i]));
                }
                CHECK(error < 1e-5f, "GRU support selection retains existing scalar tolerance");
            };
            for (bool allow_supported : {false, true}) {
                int queries = 0, nodes = -1;
                detail::GruSupport support = [&](const ggml_tensor * candidate) {
                    ++queries;
                    CHECK(candidate->op == GGML_OP_GRU, "capability query receives the actual GRU op");
                    CHECK(candidate->type == GGML_TYPE_F32 && candidate->ne[0] == s.H &&
                          candidate->ne[1] == s.B && candidate->ne[2] == s.L && candidate->ne[3] == 1,
                          "capability query receives the actual output shape and dtype");
                    const ggml_tensor * whh = candidate->src[0], *gi = candidate->src[1], *bhh = candidate->src[2];
                    CHECK(whh->type == GGML_TYPE_F32 && whh->ne[0] == s.H && whh->ne[1] == 3 * s.H &&
                          gi->type == GGML_TYPE_F32 && gi->ne[0] == 3 * s.H && gi->ne[1] == s.B && gi->ne[2] == s.L &&
                          bhh->type == GGML_TYPE_F32 && bhh->ne[0] == 3 * s.H,
                          "capability query uses real recurrent weights, input gates and bias");
                    CHECK(ggml_is_contiguous(whh) && ggml_is_contiguous(gi) && ggml_is_contiguous(bhh),
                          "capability query sees actual contiguous GRU inputs");
                    int32_t direction = -1;
                    std::memcpy(&direction, candidate->op_params, sizeof(direction));
                    CHECK(direction == (reverse ? 1 : 0), "capability query preserves the real reverse direction");
                    return allow_supported && whh->ne[0] <= 128;
                };
                parity(run_batched(x, s.I, s.B, s.L, s.H, Wih, Whh, Bih, Bhh, reverse, false,
                                   {}, {}, {}, {}, false, support, &nodes));
                const bool use_gru = allow_supported && s.H <= 128;
                CHECK(queries == 1, "each real GRU candidate is queried once");
                CHECK(nodes == (use_gru ? 1 : 0), "rejected GRU candidates never enter the fallback graph");
                use_gru ? ++accepted : ++rejected;
            }
            int legacy_queries = 0, legacy_nodes = -1, fallback_nodes = -1;
            detail::GruSupport reject = [&](const ggml_tensor *) { ++legacy_queries; return false; };
            parity(run_batched(x, s.I, s.B, s.L, s.H, Wih, Whh, Bih, Bhh, reverse, false,
                               {}, {}, {}, {}, true, reject, &legacy_nodes));
            CHECK(legacy_queries == 0 && legacy_nodes == 1, "existing fused routes do not consult the HTP-only query");
            parity(run_batched(x, s.I, s.B, s.L, s.H, Wih, Whh, Bih, Bhh, reverse, false,
                               {}, {}, {}, {}, false, {}, &fallback_nodes));
            CHECK(fallback_nodes == 0, "an absent capability query preserves the portable unrolled graph");
        }
    }
    std::printf("GRU support selection: %d accepted / %d rejected actual shapes, scalar parity preserved\n",
                accepted, rejected);
}

// Scalar conv2d — verbatim math of denoiser_core.cpp Runner::conv2d on flat
// [C][T][F] buffers (which are byte-identical to ggml [F,T,C]).  W:[Cout,Cin/g,kt,kf].
static std::vector<float> ref_conv2d(const std::vector<float> & x, int Cin, int T, int F,
                                     const std::vector<float> & W, int Cout, int Cing, int kt, int kf,
                                     const std::vector<float> * bias, int stride_f, int pad_f, int groups) {
    const int pad_t_top = kt - 1;
    const int Fout      = (F + 2 * pad_f - kf) / stride_f + 1;
    std::vector<float> out((size_t) Cout * T * Fout, 0.0f);
    auto X = [&](int c, int t, int f) { return x[((size_t) c * T + t) * F + f]; };
    auto O = [&](int c, int t, int f) -> float & { return out[((size_t) c * T + t) * Fout + f]; };
    const int og = Cout / groups;
    for (int oc = 0; oc < Cout; oc++) {
        const int   grp = oc / og, ic0 = grp * Cing;
        const float b   = bias ? (*bias)[oc] : 0.0f;
        for (int t = 0; t < T; t++)
            for (int f = 0; f < Fout; f++) O(oc, t, f) = b;
        for (int ici = 0; ici < Cing; ici++) {
            const int     ic = ic0 + ici;
            const float * wk = W.data() + (((size_t) oc * Cing) + ici) * kt * kf;
            for (int it = 0; it < kt; it++)
                for (int jf = 0; jf < kf; jf++) {
                    const float wv = wk[it * kf + jf];
                    for (int t = 0; t < T; t++) {
                        const int ti = t + it - pad_t_top;
                        if (ti < 0 || ti >= T) continue;
                        for (int f = 0; f < Fout; f++) {
                            const int fi = f * stride_f + jf - pad_f;
                            if (fi < 0 || fi >= F) continue;
                            O(oc, t, f) += wv * X(ic, ti, fi);
                        }
                    }
                }
        }
    }
    return out;
}

static std::vector<float> run_conv2d(const std::vector<float> & x, int Cin, int T, int F,
                                     const std::vector<float> & W, int Cout, int Cing, int kt, int kf,
                                     const std::vector<float> & bias, int stride_f, int pad_f, int groups, bool fused,
                                     bool portable = false, const detail::ConvSupport & conv_support = {},
                                     int * direct_nodes = nullptr, int * column_nodes = nullptr) {
    ggml_init_params p{ (size_t) 128 * 1024 * 1024, nullptr, false };
    ggml_context *   ctx = ggml_init(p);
    ggml_tensor *    tx  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, F, T, Cin);
    ggml_tensor *    tW  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kf, kt, Cing, Cout);
    ggml_tensor *    tb  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, Cout);
    std::memcpy(tx->data, x.data(), ggml_nbytes(tx));
    std::memcpy(tW->data, W.data(), ggml_nbytes(tW));
    std::memcpy(tb->data, bias.data(), ggml_nbytes(tb));
    ggml_tensor * y  = detail::conv2d(ctx, tx, tW, tb, stride_f, pad_f, groups, fused, portable, conv_support);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    if (direct_nodes) *direct_nodes = 0;
    if (column_nodes) *column_nodes = 0;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        const ggml_op op = ggml_graph_node(gf, i)->op;
        if (direct_nodes && op == GGML_OP_CONV_2D) ++*direct_nodes;
        if (column_nodes && op == GGML_OP_IM2COL) ++*column_nodes;
    }
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    std::vector<float> out((size_t) ggml_nelements(y));
    std::memcpy(out.data(), y->data, ggml_nbytes(y));
    ggml_free(ctx);
    return out;
}

static void test_conv2d() {
    std::mt19937 rng(777);
    struct Cfg { int Cin, Cout, kt, kf, stride, pad, groups; const char * name; };
    const Cfg cfgs[] = {
        { 3, 12, 3, 3, 2, 1, 1, "conv2d groups=1 stride=2" },       // encoder block 0
        { 12, 24, 2, 3, 2, 1, 2, "conv2d groups=2 stride=2" },      // grouped
        { 8, 8, 1, 5, 1, 2, 8, "conv2d depthwise stride=1" },       // depthwise (groups==Cout)
    };
    const int T = 6, F = 9;
    for (const Cfg & c : cfgs) {
        const int Cing = c.Cin / c.groups;
        std::vector<float> x((size_t) c.Cin * T * F), W((size_t) c.Cout * Cing * c.kt * c.kf), b(c.Cout);
        rand_fill(x, rng); rand_fill(W, rng); rand_fill(b, rng);
        std::vector<float> ref = ref_conv2d(x, c.Cin, T, F, W, c.Cout, Cing, c.kt, c.kf, &b, c.stride, c.pad, c.groups);
        for (bool fused : { true, false }) {
            std::vector<float> got = run_conv2d(x, c.Cin, T, F, W, c.Cout, Cing, c.kt, c.kf, b, c.stride, c.pad, c.groups, fused);
            float m = 0.0f;
            CHECK(got.size() == ref.size(), fmsg(c.name, fused));
            for (size_t i = 0; i < ref.size() && i < got.size(); i++) m = std::max(m, std::fabs(got[i] - ref[i]));
            CHECK(m < 1e-4f, fmsg(c.name, fused));
        }
    }
}

// Capability checks must use the real per-group tensors and must not insert a
// rejected direct op into the portable graph. All routes retain the scalar gate.
static void test_conv_support_selection() {
    struct Shape { int Cin, Cout, kt, kf, stride, pad, groups; };
    const Shape shapes[] = {
        {3, 5, 1, 1, 1, 0, 1}, {6, 10, 1, 1, 1, 0, 2},
        {2, 4, 3, 3, 2, 1, 1}, {6, 8, 3, 3, 1, 1, 2},
        {6, 8, 3, 5, 1, 2, 2}, // unsupported geometry preserves columns
    };
    const int F = 9, T = 6;
    std::mt19937 rng(20261029);
    int selected = 0, rejected = 0;
    for (const Shape & s : shapes) {
        const int cing = s.Cin / s.groups, cog = s.Cout / s.groups;
        std::vector<float> x((size_t) s.Cin * T * F), w((size_t) s.Cout * cing * s.kt * s.kf), bias(s.Cout);
        rand_fill(x, rng); rand_fill(w, rng); rand_fill(bias, rng);
        const auto reference = ref_conv2d(x, s.Cin, T, F, w, s.Cout, cing, s.kt, s.kf,
                                           &bias, s.stride, s.pad, s.groups);
        auto parity = [&](const std::vector<float> & got) {
            CHECK(got.size() == reference.size(), "convolution selection preserves causal output shape");
            float error = 0.0f;
            for (size_t i = 0; i < std::min(got.size(), reference.size()); ++i) {
                CHECK(std::isfinite(got[i]), "convolution selection output is finite");
                error = std::max(error, std::fabs(got[i] - reference[i]));
            }
            CHECK(error < 1e-4f, "convolution selection retains existing scalar tolerance");
        };
        for (bool allow : {false, true}) {
            int queries = 0, direct = -1, columns = -1;
            detail::ConvSupport support = [&](const ggml_tensor * candidate) {
                ++queries;
                CHECK(candidate->op == GGML_OP_CONV_2D, "convolution query receives actual direct op");
                const ggml_tensor * kernel = candidate->src[0], *input = candidate->src[1];
                CHECK(kernel->type == GGML_TYPE_F32 && input->type == GGML_TYPE_F32 &&
                      candidate->type == GGML_TYPE_F32, "convolution query preserves F32 arithmetic");
                CHECK(kernel->ne[0] == s.kf && kernel->ne[1] == s.kt && kernel->ne[2] == cing &&
                      kernel->ne[3] == cog && input->ne[0] == F && input->ne[1] == T &&
                      input->ne[2] == cing, "convolution query sees real per-group tensors");
                CHECK(ggml_is_contiguous(kernel) && ggml_is_contiguous(input),
                      "grouped convolution query sees contiguous per-group inputs");
                CHECK(candidate->ne[0] == (F + 2 * s.pad - s.kf) / s.stride + 1 &&
                      candidate->ne[1] == T + s.kt - 1 && candidate->ne[2] == cog && candidate->ne[3] == 1,
                      "convolution query sees full output before causal crop");
                int32_t params[6];
                std::memcpy(params, candidate->op_params, sizeof(params));
                CHECK(params[0] == s.stride && params[1] == 1 && params[2] == s.pad &&
                      params[3] == s.kt - 1 && params[4] == 1 && params[5] == 1,
                      "convolution query preserves stride, padding and dilation");
                return allow && s.kt == s.kf && (s.kf == 1 || s.kf == 3);
            };
            parity(run_conv2d(x, s.Cin, T, F, w, s.Cout, cing, s.kt, s.kf, bias,
                              s.stride, s.pad, s.groups, false, true, support, &direct, &columns));
            const bool use_direct = allow && s.kt == s.kf && (s.kf == 1 || s.kf == 3);
            CHECK(queries == s.groups, "each actual per-group convolution queried once");
            CHECK(direct == (use_direct ? s.groups : 0) && columns == (use_direct ? 0 : s.groups),
                  "only accepted direct convolutions enter the graph");
            use_direct ? ++selected : ++rejected;
        }
        int queries = 0, direct = -1, columns = -1;
        detail::ConvSupport reject = [&](const ggml_tensor *) { ++queries; return false; };
        parity(run_conv2d(x, s.Cin, T, F, w, s.Cout, cing, s.kt, s.kf, bias,
                          s.stride, s.pad, s.groups, false, false, reject, &direct, &columns));
        CHECK(queries == 0 && direct == s.groups && columns == 0,
              "existing nonportable convolution route ignores HTP support query");
        parity(run_conv2d(x, s.Cin, T, F, w, s.Cout, cing, s.kt, s.kf, bias,
                          s.stride, s.pad, s.groups, false, true, {}, &direct, &columns));
        CHECK(direct == 0 && columns == s.groups, "absent convolution capability preserves old-runtime graph");
        if (s.groups == 2 && s.kf <= 3) {
            queries = 0;
            detail::ConvSupport mixed = [&](const ggml_tensor *) { return ++queries == 1; };
            parity(run_conv2d(x, s.Cin, T, F, w, s.Cout, cing, s.kt, s.kf, bias,
                              s.stride, s.pad, s.groups, false, true, mixed, &direct, &columns));
            CHECK(queries == 2 && direct == 1 && columns == 1,
                  "per-group capability decisions can mix direct and portable convolutions");
        }
    }
    std::printf("Convolution support selection: %d accepted / %d rejected shapes; scalar parity preserved\n",
                selected, rejected);
}

// Scalar transposed conv — verbatim math of denoiser_core.cpp Runner::conv_transpose2d.
// W (PyTorch transpose kernel): [Cin, Cout/g, kt, kf].
static std::vector<float> ref_conv_transpose2d(const std::vector<float> & x, int Cin, int T, int F,
                                               const std::vector<float> & W, int Coutg, int kt, int kf,
                                               const std::vector<float> * bias, int stride_f, int pad_f, int groups) {
    const int Cout = Coutg * groups;
    const int pt_top = kt - 1, pad_t = kt - 1;
    const int Tp = T + pt_top, Tfull = (Tp - 1) + kt, Ffull = (F - 1) * stride_f + kf;
    std::vector<float> full((size_t) Cout * Tfull * Ffull, 0.0f);
    auto X  = [&](int c, int t, int f) { return x[((size_t) c * T + t) * F + f]; };
    auto FL = [&](int c, int t, int f) -> float & { return full[((size_t) c * Tfull + t) * Ffull + f]; };
    const int ing = Cin / groups;
    for (int ic = 0; ic < Cin; ic++) {
        const int grp = ic / ing, oc0 = grp * Coutg;
        for (int ocg = 0; ocg < Coutg; ocg++) {
            const int     oc = oc0 + ocg;
            const float * wk = W.data() + (((size_t) ic * Coutg) + ocg) * kt * kf;
            for (int t = 0; t < Tp; t++) {
                const int xt = t - pt_top;
                for (int f = 0; f < F; f++) {
                    const float v = (xt < 0) ? 0.0f : X(ic, xt, f);
                    for (int it = 0; it < kt; it++)
                        for (int jf = 0; jf < kf; jf++) FL(oc, t + it, f * stride_f + jf) += v * wk[it * kf + jf];
                }
            }
        }
    }
    if (bias)
        for (int oc = 0; oc < Cout; oc++) {
            const float b = (*bias)[oc];
            for (int t = 0; t < Tfull; t++)
                for (int f = 0; f < Ffull; f++) FL(oc, t, f) += b;
        }
    const int          Tout = Tfull - 2 * pad_t, Fout = Ffull - 2 * pad_f;
    std::vector<float> out((size_t) Cout * Tout * Fout, 0.0f);
    for (int oc = 0; oc < Cout; oc++)
        for (int t = 0; t < Tout; t++)
            for (int f = 0; f < Fout; f++) out[((size_t) oc * Tout + t) * Fout + f] = FL(oc, t + pad_t, f + pad_f);
    return out;
}

// Host reindex: PyTorch transpose kernel Wt[Cin,Cout/g,kt,kf] -> regular-conv Wc
// [Cout,Cin/g,kt,kf] with IC<->OC swap and kt/kf flip (the create()-time transform).
static std::vector<float> reindex_ct(const std::vector<float> & Wt, int Cin, int Coutg, int kt, int kf, int groups) {
    const int          Cout = Coutg * groups, ing = Cin / groups;
    std::vector<float> Wc((size_t) Cout * ing * kt * kf);
    for (int oc = 0; oc < Cout; oc++) {
        const int grp = oc / Coutg, ocg = oc - grp * Coutg;
        for (int ici = 0; ici < ing; ici++)
            for (int it = 0; it < kt; it++)
                for (int jf = 0; jf < kf; jf++) {
                    const int ic = grp * ing + ici;
                    Wc[(((size_t) oc * ing + ici) * kt + it) * kf + jf] =
                        Wt[(((size_t) ic * Coutg + ocg) * kt + (kt - 1 - it)) * kf + (kf - 1 - jf)];
                }
    }
    return Wc;
}

static std::vector<float> run_conv_transpose2d(const std::vector<float> & x, int Cin, int T, int F,
                                               const std::vector<float> & Wc, int Cout, int ing, int kt, int kf,
                                               const std::vector<float> & bias, int stride_f, int pad_f, int groups, bool fused,
                                               bool portable = false, const detail::ConvSupport & conv_support = {},
                                               int * direct_nodes = nullptr) {
    ggml_init_params p{ (size_t) 128 * 1024 * 1024, nullptr, false };
    ggml_context *   ctx = ggml_init(p);
    ggml_tensor *    tx  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, F, T, Cin);
    ggml_tensor *    tW  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kf, kt, ing, Cout);
    ggml_tensor *    tb  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, Cout);
    std::memcpy(tx->data, x.data(), ggml_nbytes(tx));
    std::memcpy(tW->data, Wc.data(), ggml_nbytes(tW));
    std::memcpy(tb->data, bias.data(), ggml_nbytes(tb));
    ggml_tensor * y  = detail::conv_transpose2d(ctx, tx, tW, tb, stride_f, pad_f, groups, fused, portable, conv_support);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    if (direct_nodes) {
        *direct_nodes = 0;
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i)
            if (ggml_graph_node(gf, i)->op == GGML_OP_CONV_2D) ++*direct_nodes;
    }
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    std::vector<float> out((size_t) ggml_nelements(y));
    std::memcpy(out.data(), y->data, ggml_nbytes(y));
    ggml_free(ctx);
    return out;
}

static void test_conv_transpose2d() {
    std::mt19937 rng(555);
    struct Cfg { int Cin, Coutg, kt, kf, stride, pad, groups; const char * name; };
    const Cfg cfgs[] = {
        { 8, 1, 2, 3, 2, 1, 8, "convT depthwise stride=2" },  // decoder i=1 dconv
        { 8, 1, 1, 5, 1, 2, 8, "convT depthwise stride=1" },  // decoder i=4 dconv
        { 8, 4, 3, 3, 2, 1, 1, "convT groups=1 stride=2" },   // decoder i=0 final ops.1
    };
    const int T = 6, F = 7;
    for (const Cfg & c : cfgs) {
        const int Cout = c.Coutg * c.groups, ing = c.Cin / c.groups;
        std::vector<float> x((size_t) c.Cin * T * F), Wt((size_t) c.Cin * c.Coutg * c.kt * c.kf), b(Cout);
        rand_fill(x, rng); rand_fill(Wt, rng); rand_fill(b, rng);
        std::vector<float> ref = ref_conv_transpose2d(x, c.Cin, T, F, Wt, c.Coutg, c.kt, c.kf, &b, c.stride, c.pad, c.groups);
        std::vector<float> Wc  = reindex_ct(Wt, c.Cin, c.Coutg, c.kt, c.kf, c.groups);
        for (bool fused : { true, false }) {
            std::vector<float> got = run_conv_transpose2d(x, c.Cin, T, F, Wc, Cout, ing, c.kt, c.kf, b, c.stride, c.pad, c.groups, fused);
            float m = 0.0f;
            CHECK(got.size() == ref.size(), fmsg(c.name, fused));
            for (size_t i = 0; i < ref.size() && i < got.size(); i++) m = std::max(m, std::fabs(got[i] - ref[i]));
            CHECK(m < 1e-4f, fmsg(c.name, fused));
        }
        for (bool allow : {false, true}) {
            int queries = 0, direct_nodes = -1;
            detail::ConvSupport support = [&](const ggml_tensor * candidate) {
                ++queries;
                CHECK(candidate->op == GGML_OP_CONV_2D && candidate->src[1]->ne[0] == (F - 1) * c.stride + 1,
                      "transposed convolution queries the zero-inserted regular convolution");
                return allow;
            };
            const auto got = run_conv_transpose2d(x, c.Cin, T, F, Wc, Cout, ing, c.kt, c.kf, b,
                                                   c.stride, c.pad, c.groups, false, true, support, &direct_nodes);
            CHECK(got.size() == ref.size(), "portable transposed convolution preserves output shape");
            float error = 0.0f;
            for (size_t i = 0; i < std::min(got.size(), ref.size()); ++i) {
                CHECK(std::isfinite(got[i]), "portable transposed convolution output is finite");
                error = std::max(error, std::fabs(got[i] - ref[i]));
            }
            CHECK(error < 1e-4f, "portable transposed convolution retains existing tolerance");
            const bool depthwise = ing == 1 && c.groups == Cout;
            CHECK(queries == (depthwise ? 0 : c.groups) && direct_nodes == (depthwise || !allow ? 0 : c.groups),
                  "transposed convolution preserves depthwise route and direct-op fallback");
        }
    }
}

static void test_affine_prelu() {
    std::mt19937 rng(4242);
    const int F = 7, T = 5, C = 6;
    std::vector<float> x((size_t) C * T * F), aw((size_t) C * F), ab((size_t) C * F), sl(C);
    rand_fill(x, rng); rand_fill(aw, rng); rand_fill(ab, rng); rand_fill(sl, rng);
    std::vector<float> ref(x.size());
    for (int c = 0; c < C; c++)
        for (int t = 0; t < T; t++)
            for (int f = 0; f < F; f++) {
                const size_t i = ((size_t) c * T + t) * F + f;
                const float  v = x[i];
                ref[i] = (v * aw[(size_t) c * F + f] + ab[(size_t) c * F + f]) + (v > 0.0f ? v : sl[c] * v);
            }
    for (bool fused : { true, false }) {
        ggml_init_params p{ (size_t) 32 * 1024 * 1024, nullptr, false };
        ggml_context *   ctx = ggml_init(p);
        ggml_tensor *    tx  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, F, T, C);
        ggml_tensor *    ta  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, F, C);
        ggml_tensor *    tb  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, F, C);
        ggml_tensor *    ts  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
        std::memcpy(tx->data, x.data(), ggml_nbytes(tx));
        std::memcpy(ta->data, aw.data(), ggml_nbytes(ta));
        std::memcpy(tb->data, ab.data(), ggml_nbytes(tb));
        std::memcpy(ts->data, sl.data(), ggml_nbytes(ts));
        ggml_tensor * y  = detail::affine_prelu(ctx, tx, ta, tb, ts, fused);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_graph_compute_with_ctx(ctx, gf, 1);
        float m = 0.0f;
        CHECK((size_t) ggml_nelements(y) == ref.size(), fmsg("affine_prelu output size", fused));
        const float * out = (const float *) y->data;
        for (size_t i = 0; i < ref.size(); i++) m = std::max(m, std::fabs(out[i] - ref[i]));
        CHECK(m < 1e-5f, fmsg("affine_prelu matches scalar reference", fused));
        ggml_free(ctx);
    }
}

// A real capability query selects only the requested affine candidate. Older
// runtimes reject it and retain the existing seven-op F32 graph.
static void test_affine_support_selection() {
    for (int mode : {0, 1, 2, 3}) { // absent, rejected, accepted, non-contiguous
        ggml_init_params params{(size_t)32 * 1024 * 1024, nullptr, false};
        ggml_context * ctx = ggml_init(params);
        const int F = 33, T = 9, C = 3, B = 2;
        ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, F, T, C, B);
        if (mode == 3) {
            // ReLU's existing fallback requires contiguous rows. Row padding
            // violates full contiguity without violating that precondition.
            ggml_tensor * storage = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (F + 1) * T * C * B);
            x = ggml_view_4d(ctx, storage, F, T, C, B, (F + 1) * sizeof(float),
                             (F + 1) * T * sizeof(float), (F + 1) * T * C * sizeof(float), 0);
            CHECK(ggml_is_contiguous_rows(x) && !ggml_is_contiguous(x), "affine padded-row fixture satisfies fallback preconditions");
        }
        ggml_tensor * aw = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, F, C);
        ggml_tensor * ab = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, F, C);
        ggml_tensor * slope = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
        int queries = 0;
        detail::AffineSupport support;
        if (mode != 0) {
            support = [&](const ggml_tensor * candidate) {
                ++queries;
                CHECK(candidate->op == GGML_OP_AFFINE_PRELU, "affine query receives actual operation");
                CHECK(candidate->src[0] == x && candidate->src[1] == aw && candidate->src[2] == ab &&
                      candidate->src[3] == slope, "affine query preserves actual inputs");
                CHECK(ggml_are_same_shape(candidate, x), "affine query preserves output dimensions");
                return mode == 2;
            };
        }
        ggml_tensor * y = detail::affine_prelu(ctx, x, aw, ab, slope, false, support);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, y);
        int affine = 0, elementwise = 0;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            const ggml_op op = ggml_graph_node(graph, i)->op;
            affine += op == GGML_OP_AFFINE_PRELU;
            elementwise += op == GGML_OP_MUL || op == GGML_OP_ADD || op == GGML_OP_SUB || op == GGML_OP_UNARY;
        }
        CHECK(queries == (mode == 1 || mode == 2 ? 1 : 0), "affine queries absent/rejected/accepted/non-contiguous");
        CHECK(affine == (mode == 2 ? 1 : 0) && elementwise == (mode == 2 ? 0 : 7),
              "affine accepted candidate or unchanged seven-op fallback exclusively enters graph");
        if (mode != 3) {
            for (int i = 0; i < F * T * C * B; ++i) ((float *)x->data)[i] = (i % 17 - 8) * 0.125f;
            for (int i = 0; i < F * C; ++i) {
                ((float *)aw->data)[i] = (i % 7 - 3) * 0.25f;
                ((float *)ab->data)[i] = (i % 9 - 4) * 0.125f;
            }
            for (int c = 0; c < C; ++c) ((float *)slope->data)[c] = (c - 1) * 0.5f;
            ggml_graph_compute_with_ctx(ctx, graph, 1);
            float max_error = 0.0f;
            for (int b = 0; b < B; ++b) for (int c = 0; c < C; ++c)
                for (int t = 0; t < T; ++t) for (int f = 0; f < F; ++f) {
                    const int i = ((b * C + c) * T + t) * F + f;
                    const float v = ((float *)x->data)[i], r = v > 0.0f ? v : 0.0f;
                    const float prelu = r + ((float *)slope->data)[c] * (v - r);
                    const float a = v * ((float *)aw->data)[c * F + f] + ((float *)ab->data)[c * F + f];
                    max_error = std::max(max_error, std::fabs(((float *)y->data)[i] - (a + prelu)));
                }
            CHECK(max_error < 1e-5f, "affine selected/fallback graph matches scalar across batch and frequency tail");
        }
        ggml_free(ctx);
    }
}

static void test_shuffle2() {
    std::mt19937 rng(5151);
    const int F = 5, T = 4, C = 6, half = C / 2;
    std::vector<float> x((size_t) C * T * F);
    rand_fill(x, rng);
    std::vector<float> ref(x.size());
    for (int c = 0; c < half; c++)
        for (int t = 0; t < T; t++)
            for (int f = 0; f < F; f++) {
                ref[((size_t) (2 * c) * T + t) * F + f]     = x[((size_t) c * T + t) * F + f];
                ref[((size_t) (2 * c + 1) * T + t) * F + f] = x[((size_t) (half + c) * T + t) * F + f];
            }
    for (bool fused : { true, false }) {
        ggml_init_params p{ (size_t) 32 * 1024 * 1024, nullptr, false };
        ggml_context *   ctx = ggml_init(p);
        ggml_tensor *    tx  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, F, T, C);
        std::memcpy(tx->data, x.data(), ggml_nbytes(tx));
        ggml_tensor * y  = detail::shuffle2(ctx, tx, fused);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_graph_compute_with_ctx(ctx, gf, 1);
        float m = -1.0f;
        CHECK((size_t) ggml_nelements(y) == ref.size(), fmsg("shuffle2 output size", fused));
        const float * out = (const float *) y->data;
        m = 0.0f;
        for (size_t i = 0; i < ref.size(); i++) m = std::max(m, std::fabs(out[i] - ref[i]));
        CHECK(m == 0.0f, fmsg("shuffle2 is an exact channel interleave", fused));
        ggml_free(ctx);
    }
}

static void test_shuffle2_portable() {
    struct Shape { int F, T, C, Bc; };
    const Shape shapes[] = {{3, 5, 2, 2}, {5, 3, 6, 3}, {5, 7, 12, 3},
                            {3, 9, 24, 2}, {7, 5, 32, 3},
                            {65, 63, 24, 7}, {65, 63, 48, 7}};
    for (const Shape & s : shapes) {
        for (bool strided : {false, true}) {
            ggml_init_params params{size_t(64) * 1024 * 1024, nullptr, false};
            ggml_context * ctx = ggml_init(params);
            ggml_tensor * x;
            if (strided) {
                ggml_tensor * storage = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                    s.F + 2, s.T + 1, s.C + 2, s.Bc + 1);
                x = ggml_view_4d(ctx, storage, s.F, s.T, s.C, s.Bc,
                                  storage->nb[1], storage->nb[2], storage->nb[3], sizeof(float));
            } else {
                x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s.F, s.T, s.C, s.Bc);
            }
            const size_t count = size_t(s.F) * s.T * s.C * s.Bc;
            std::vector<float> tagged(count), expected(count);
            for (size_t i = 0; i < count; ++i) {
                // Distinct exactly representable tags across all four axes,
                // including negative zero which an arithmetic copy could lose.
                tagged[i] = i == 0 ? -0.0f : float(i) * (i % 2 ? -1.0f : 1.0f);
            }
            for (int b = 0; b < s.Bc; ++b)
                for (int c = 0; c < s.C; ++c)
                    for (int t = 0; t < s.T; ++t)
                        for (int f = 0; f < s.F; ++f) {
                            const size_t i = ((size_t(b) * s.C + c) * s.T + t) * s.F + f;
                            std::memcpy(static_cast<char *>(x->data) + size_t(f) * x->nb[0] +
                                size_t(t) * x->nb[1] + size_t(c) * x->nb[2] + size_t(b) * x->nb[3],
                                &tagged[i], sizeof(float));
                            const int source_c = (c % 2) * (s.C / 2) + c / 2;
                            expected[i] = tagged[((size_t(b) * s.C + source_c) * s.T + t) * s.F + f];
                        }
            ggml_tensor * y = detail::shuffle2(ctx, x, false, true);
            ggml_cgraph * graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, y);
            int copies = 0, plane_copies = 0;
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                ggml_tensor * node = ggml_graph_node(graph, i);
                if (node->op == GGML_OP_CONT) {
                    ++copies;
                    if (node->src[0]->op == GGML_OP_PERMUTE && node->ne[0] == int64_t(s.F) * s.T)
                        ++plane_copies;
                }
            }
            CHECK(copies == (strided ? 2 : 1), "portable shuffle only normalizes input and copies channel planes");
            CHECK(plane_copies == 1, "portable shuffle preserves F*T as its contiguous copy width");
            CHECK(ggml_graph_compute_with_ctx(ctx, graph, 1) == GGML_STATUS_SUCCESS,
                  "portable shuffle graph computes successfully");
            CHECK(ggml_nelements(y) == int64_t(count) && y->ne[0] == s.F &&
                  y->ne[1] == s.T && y->ne[2] == s.C && y->ne[3] == s.Bc,
                  "portable shuffle retains all four dimensions");
            CHECK(std::memcmp(y->data, expected.data(), count * sizeof(float)) == 0,
                  "portable shuffle is bit-exact for batches, spatial positions and signed zero");
            ggml_free(ctx);
        }
    }
}

static void test_mean_support_selection() {
    for (int width : {1, 4, 12, 24, 32, 33, 65, 129, 513}) {
        for (bool padded : {false, true}) for (int mode : {0, 1, 2, 3}) {
            // Absent/rejected/accepted portable queries and unchanged nonportable mean.
            ggml_init_params params{size_t(8) * 1024 * 1024, nullptr, false};
            ggml_context * ctx = ggml_init(params);
            const int rows = 3, batches = 2;
            ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, rows, batches);
            if (padded) {
                ggml_tensor * storage = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,
                                                          (width + 2) * rows * batches + 1);
                std::fill_n(static_cast<float *>(storage->data), ggml_nelements(storage), 37.25f);
                x = ggml_view_3d(ctx, storage, width, rows, batches, (width + 2) * sizeof(float),
                                  (width + 2) * rows * sizeof(float), sizeof(float));
            }
            float expected[rows * batches] = {};
            for (int b = 0; b < batches; ++b) for (int r = 0; r < rows; ++r) {
                float sum = 0.0f;
                for (int j = 0; j < width; ++j) {
                    const float value = (j % 5 == 0) ? -0.0f : float((j + 13 * r + 7 * b) % 29 - 14) / 16.0f;
                    std::memcpy(static_cast<char *>(x->data) + j * x->nb[0] + r * x->nb[1] + b * x->nb[2],
                                &value, sizeof(value));
                    sum += value;
                }
                expected[b * rows + r] = sum / float(width);
            }
            int queries = 0;
            detail::MeanSupport support;
            if (mode != 0) support = [&](const ggml_tensor * candidate) {
                ++queries;
                CHECK(candidate->op == GGML_OP_MEAN && candidate->src[0] == x,
                      "mean query receives the actual input and operation");
                CHECK(candidate->type == GGML_TYPE_F32 && candidate->ne[0] == 1 &&
                      candidate->ne[1] == rows && candidate->ne[2] == batches,
                      "mean query preserves row and batch dimensions");
                return mode >= 2;
            };
            ggml_tensor * y = detail::mean_rows(ctx, x, mode != 3, support);
            ggml_cgraph * graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, y);
            int means = 0, sums = 0, scales = 0;
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                const ggml_op op = ggml_graph_node(graph, i)->op;
                means += op == GGML_OP_MEAN;
                sums += op == GGML_OP_SUM_ROWS;
                scales += op == GGML_OP_SCALE;
            }
            const bool selected = mode >= 2;
            CHECK(queries == (mode == 1 || mode == 2 ? 1 : 0), "mean capability query selection");
            CHECK(means == int(selected) && sums == int(!selected) && scales == int(!selected),
                  "mean selected exclusively or complete legacy sum/scale fallback");
            CHECK(ggml_graph_compute_with_ctx(ctx, graph, 1) == GGML_STATUS_SUCCESS, "mean selection graph computes");
            const float * values = static_cast<const float *>(y->data);
            for (int i = 0; i < rows * batches; ++i)
                CHECK(std::fabs(values[i] - expected[i]) < 1e-6f,
                      "mean selected/fallback preserves padded rows, tails and batch order");
            ggml_free(ctx);
        }
    }
}

static void test_channel_energy_mean() {
    struct Shape { int F, T, C, B; };
    const Shape shapes[] = {{1, 1, 1, 1}, {129, 63, 1, 7}, {33, 5, 1, 2}, {7, 5, 3, 2}, {5, 9, 12, 3},
                            {33, 63, 24, 7}, {33, 63, 32, 7}, {65, 63, 12, 7}};
    for (const Shape & s : shapes) {
        for (bool strided : {false, true}) {
            ggml_init_params params{size_t(128) * 1024 * 1024, nullptr, false};
            ggml_context * ctx = ggml_init(params);
            ggml_tensor * x;
            if (strided) {
                ggml_tensor * storage = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s.F + 2, s.T + 1, s.C, s.B);
                std::fill_n(static_cast<float *>(storage->data), ggml_nelements(storage), 37.25f);
                x = ggml_view_4d(ctx, storage, s.F, s.T, s.C, s.B,
                                  storage->nb[1], storage->nb[2], storage->nb[3], sizeof(float));
            } else {
                x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s.F, s.T, s.C, s.B);
            }
            std::vector<float> expected(size_t(s.F) * s.T * s.B, 0.0f);
            for (int b = 0; b < s.B; ++b)
                for (int t = 0; t < s.T; ++t)
                    for (int f = 0; f < s.F; ++f) {
                        float sum = 0.0f;
                        for (int c = 0; c < s.C; ++c) {
                            const int tag = (13 * f + 7 * t + 3 * c + 19 * b) % 113 - 56;
                            const float value = (f == 0 && t == 0) ? -0.0f : float(tag) / 64.0f;
                            std::memcpy(static_cast<char *>(x->data) + size_t(f) * x->nb[0] +
                                        size_t(t) * x->nb[1] + size_t(c) * x->nb[2] + size_t(b) * x->nb[3],
                                        &value, sizeof(value));
                            sum += value * value;
                        }
                        expected[size_t(f) + size_t(s.F) * (t + s.T * b)] = sum / float(s.C);
                    }
            ggml_tensor * optimized = detail::channel_energy_mean(ctx, x, true);
            ggml_tensor * standard = detail::channel_energy_mean(ctx, x, false);
            if (s.C == 1) {
                ggml_cgraph * fast = ggml_new_graph(ctx);
                ggml_build_forward_expand(fast, optimized);
                int squares = 0, copies = 0;
                for (int i = 0; i < ggml_graph_n_nodes(fast); ++i) {
                    const ggml_op op = ggml_graph_node(fast, i)->op;
                    squares += op == GGML_OP_SQR;
                    copies += op == GGML_OP_CONT;
                    CHECK(op != GGML_OP_SUM_ROWS && op != GGML_OP_SCALE && op != GGML_OP_MEAN,
                          "singleton channel energy omits reduction and reciprocal scaling");
                }
                CHECK(squares == 1 && copies == (ggml_is_contiguous(x) ? 0 : 1),
                      "singleton channel energy squares once with only a necessary strided-input copy");
            }
            int mean_queries = 0;
            detail::MeanSupport mean_support = [&](const ggml_tensor * candidate) {
                ++mean_queries;
                CHECK(candidate->op == GGML_OP_MEAN && candidate->src[0]->ne[0] == s.C,
                      "channel energy queries the actual channel mean");
                return true;
            };
            ggml_tensor * selected = detail::channel_energy_mean(ctx, x, true, mean_support);
            CHECK(mean_queries == (s.C == 1 ? 0 : 1),
                  "channel energy queries mean support only for a non-singleton reduction");
            // Preserve the former portable graph as an exact regression oracle.
            ggml_tensor * previous = ggml_cont(ctx, ggml_permute(ctx, x, 2, 1, 0, 3));
            previous = ggml_scale(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, previous)), 1.0f / float(s.C));
            previous = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, previous, 3, 1, 0, 2)), s.F, s.T * s.B);
            ggml_cgraph * graph = ggml_new_graph(ctx);
            for (ggml_tensor * output : {optimized, standard, previous, selected}) ggml_build_forward_expand(graph, output);
            CHECK(ggml_graph_compute_with_ctx(ctx, graph, 2) == GGML_STATUS_SUCCESS,
                  "channel energy graphs compute successfully");
            CHECK(optimized->ne[0] == s.F && optimized->ne[1] == s.T * s.B &&
                  optimized->ne[2] == 1 && optimized->ne[3] == 1, "channel energy retains frequency/time/batch order");
            CHECK(std::memcmp(optimized->data, previous->data, ggml_nbytes(optimized)) == 0,
                  "flattened channel energy is bit-exact with the former portable graph");
            for (ggml_tensor * output : {optimized, standard, selected}) {
                const float * actual = static_cast<const float *>(output->data);
                float error = 0.0f;
                for (size_t i = 0; i < expected.size(); ++i) error = std::max(error, std::fabs(actual[i] - expected[i]));
                CHECK(error < 1e-6f, "channel energy agrees with an independent scalar channel reduction");
            }
            ggml_free(ctx);
        }
    }
}

// ---- DPGRNN parity: scalar reference (denoiser_core.cpp dpgrnn) vs detail::dpgrnn ----
#include <map>
#include <string>

struct WBag {
    std::map<std::string, std::vector<float>> data;
    std::map<std::string, std::vector<int>>   shape; // C-order
    const std::vector<float> & d(const std::string & n) const { return data.at(n); }
    bool                       has(const std::string & n) const { return data.count(n) != 0; }
    int                        dim(const std::string & n, int i) const { return shape.at(n)[i]; }
};

static std::vector<float> ref_linear2d(const std::vector<float> & x, int N, int c_in,
                                       const std::vector<float> & W, int c_out, const std::vector<float> * b) {
    std::vector<float> y((size_t) N * c_out, 0.0f);
    for (int n = 0; n < N; n++)
        for (int o = 0; o < c_out; o++) {
            float acc = b ? (*b)[o] : 0.0f;
            for (int i = 0; i < c_in; i++) acc += x[(size_t) n * c_in + i] * W[(size_t) o * c_in + i];
            y[(size_t) n * c_out + o] = acc;
        }
    return y;
}

static void ref_layernorm_fc(std::vector<float> & x, int T, int Fw, int C,
                             const std::vector<float> & g, const std::vector<float> & b, float eps) {
    const int n = Fw * C;
    for (int t = 0; t < T; t++) {
        float * row  = x.data() + (size_t) t * n;
        double  mean = 0.0;
        for (int i = 0; i < n; i++) mean += row[i];
        mean /= n;
        double var = 0.0;
        for (int i = 0; i < n; i++) { double dd = row[i] - mean; var += dd * dd; }
        var /= n;
        const float inv = 1.0f / std::sqrt((float) var + eps);
        for (int i = 0; i < n; i++) row[i] = ((float) (row[i] - mean) * inv) * g[i] + b[i];
    }
}

static std::vector<float> ref_grnn(const std::vector<float> & x, int L, int I,
                                   const std::string & p, const WBag & W, bool bidir) {
    const int          half = I / 2;
    std::vector<float> x1((size_t) L * half), x2((size_t) L * half);
    for (int t = 0; t < L; t++)
        for (int i = 0; i < half; i++) {
            x1[(size_t) t * half + i] = x[(size_t) t * I + i];
            x2[(size_t) t * half + i] = x[(size_t) t * I + half + i];
        }
    auto run = [&](const std::vector<float> & xi, const std::string & pp) {
        const int H = W.dim(pp + ".weight_hh_l0", 1);
        if (!bidir)
            return ref_gru_seq(xi, L, half, H, W.d(pp + ".weight_ih_l0"), W.d(pp + ".weight_hh_l0"),
                               W.d(pp + ".bias_ih_l0"), W.d(pp + ".bias_hh_l0"));
        std::vector<float> f = ref_gru_seq(xi, L, half, H, W.d(pp + ".weight_ih_l0"), W.d(pp + ".weight_hh_l0"),
                                           W.d(pp + ".bias_ih_l0"), W.d(pp + ".bias_hh_l0"));
        std::vector<float> r = ref_gru_rev(xi, L, half, H, W.d(pp + ".weight_ih_l0_reverse"),
                                           W.d(pp + ".weight_hh_l0_reverse"), W.d(pp + ".bias_ih_l0_reverse"),
                                           W.d(pp + ".bias_hh_l0_reverse"));
        std::vector<float> y((size_t) L * 2 * H);
        for (int t = 0; t < L; t++)
            for (int k = 0; k < H; k++) {
                y[(size_t) t * 2 * H + k]     = f[(size_t) t * H + k];
                y[(size_t) t * 2 * H + H + k] = r[(size_t) t * H + k];
            }
        return y;
    };
    std::vector<float> y1 = run(x1, p + ".rnn1"), y2 = run(x2, p + ".rnn2");
    const int          h1 = (int) (y1.size() / L), h2 = (int) (y2.size() / L);
    std::vector<float> y((size_t) L * (h1 + h2));
    for (int t = 0; t < L; t++) {
        for (int i = 0; i < h1; i++) y[(size_t) t * (h1 + h2) + i]      = y1[(size_t) t * h1 + i];
        for (int i = 0; i < h2; i++) y[(size_t) t * (h1 + h2) + h1 + i] = y2[(size_t) t * h2 + i];
    }
    return y;
}

static std::vector<float> ref_dpgrnn(const std::vector<float> & x, int C, int T, int F,
                                     const std::string & pre, const WBag & W, float eps) {
    auto fcb = [&](const std::string & n) { return W.has(n) ? &W.d(n) : nullptr; };
    std::vector<float> xt((size_t) T * F * C);
    for (int t = 0; t < T; t++)
        for (int f = 0; f < F; f++)
            for (int c = 0; c < C; c++) xt[((size_t) t * F + f) * C + c] = x[((size_t) c * T + t) * F + f];
    std::vector<float> intra((size_t) T * F * C);
    for (int t = 0; t < T; t++) {
        std::vector<float> seq(xt.begin() + (size_t) t * F * C, xt.begin() + (size_t) (t + 1) * F * C);
        std::vector<float> y  = ref_grnn(seq, F, C, pre + ".intra_rnn", W, true);
        const int          hy = (int) (y.size() / F);
        y = ref_linear2d(y, F, hy, W.d(pre + ".intra_fc.weight"), C, fcb(pre + ".intra_fc.bias"));
        std::copy(y.begin(), y.end(), intra.begin() + (size_t) t * F * C);
    }
    ref_layernorm_fc(intra, T, F, C, W.d(pre + ".intra_ln.weight"), W.d(pre + ".intra_ln.bias"), eps);
    for (size_t i = 0; i < intra.size(); i++) intra[i] += xt[i];
    std::vector<float> inter((size_t) T * F * C);
    for (int f = 0; f < F; f++) {
        std::vector<float> seq((size_t) T * C);
        for (int t = 0; t < T; t++)
            for (int c = 0; c < C; c++) seq[(size_t) t * C + c] = intra[((size_t) t * F + f) * C + c];
        std::vector<float> y  = ref_grnn(seq, T, C, pre + ".inter_rnn", W, false);
        const int          hy = (int) (y.size() / T);
        y = ref_linear2d(y, T, hy, W.d(pre + ".inter_fc.weight"), C, fcb(pre + ".inter_fc.bias"));
        for (int t = 0; t < T; t++)
            for (int c = 0; c < C; c++) inter[((size_t) t * F + f) * C + c] = y[(size_t) t * C + c];
    }
    ref_layernorm_fc(inter, T, F, C, W.d(pre + ".inter_ln.weight"), W.d(pre + ".inter_ln.bias"), eps);
    for (size_t i = 0; i < inter.size(); i++) inter[i] += intra[i];
    std::vector<float> out((size_t) C * T * F);
    for (int t = 0; t < T; t++)
        for (int f = 0; f < F; f++)
            for (int c = 0; c < C; c++) out[((size_t) c * T + t) * F + f] = inter[((size_t) t * F + f) * C + c];
    return out;
}

static std::vector<float> run_dpgrnn(const std::vector<float> & x, int C, int T, int F,
                                     const WBag & bag, float eps, bool fused, bool portable = false) {
    ggml_init_params p{ (size_t) 256 * 1024 * 1024, nullptr, false };
    ggml_context *   ctx = ggml_init(p);
    ggml_tensor *    tx  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, F, T, C);
    std::memcpy(tx->data, x.data(), ggml_nbytes(tx));
    std::map<std::string, ggml_tensor *> tmap;
    for (const auto & kv : bag.data) {
        const std::vector<int> & shp = bag.shape.at(kv.first);
        int64_t                  ne[4] = { 1, 1, 1, 1 };
        const int                nd    = (int) shp.size();
        for (int i = 0; i < nd; i++) ne[i] = shp[nd - 1 - i];
        ggml_tensor * t = ggml_new_tensor(ctx, GGML_TYPE_F32, nd, ne);
        std::memcpy(t->data, kv.second.data(), ggml_nbytes(t));
        tmap[kv.first] = t;
    }
    detail::WResolver Wr = [&](const std::string & n) -> ggml_tensor * {
        auto it = tmap.find(n);
        return it == tmap.end() ? nullptr : it->second;
    };
    ggml_tensor * y  = detail::dpgrnn(ctx, tx, "dp", Wr, eps, fused, {}, portable);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);
    ggml_build_forward_expand(gf, y);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    std::vector<float> out((size_t) ggml_nelements(y));
    std::memcpy(out.data(), y->data, ggml_nbytes(y));
    ggml_free(ctx);
    return out;
}

static void test_dpgrnn_output_layout() {
    struct Shape { int C, F, T, B; };
    const Shape shapes[] = {{1, 1, 1, 1}, {1, 7, 5, 3}, {3, 5, 7, 2}, {16, 33, 63, 7},
                            {17, 7, 19, 3}, {32, 16, 8, 2}, {33, 1, 129, 2}, {7, 129, 1, 3}};
    for (const Shape & s : shapes) {
        ggml_init_params params{size_t(16) * 1024 * 1024, nullptr, false};
        ggml_context * ctx = ggml_init(params);
        ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, s.C, s.F, s.T, s.B);
        std::vector<uint32_t> input(ggml_nelements(x)), expected(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = 0x3f000000u + uint32_t(i);
            if (i % 17 == 0) input[i] = 0x80000000u; // Preserve negative zero as raw bits.
            else if (i % 19 == 0) input[i] = 0;
            else if (i % 3 == 0) input[i] |= 0x80000000u;
        }
        std::memcpy(x->data, input.data(), ggml_nbytes(x));
        for (int b = 0; b < s.B; ++b) for (int t = 0; t < s.T; ++t)
            for (int f = 0; f < s.F; ++f) for (int c = 0; c < s.C; ++c) {
                const size_t source = c + size_t(s.C) * (f + s.F * (t + s.T * b));
                const size_t target = f + size_t(s.F) * (t + s.T * (c + s.C * b));
                expected[target] = input[source];
            }
        ggml_tensor * optimized = detail::dpgrnn_output_layout(ctx, x, true);
        ggml_tensor * legacy = detail::dpgrnn_output_layout(ctx, x, false);
        CHECK(optimized->ne[0] == s.F && optimized->ne[1] == s.T &&
              optimized->ne[2] == s.C && optimized->ne[3] == s.B,
              "DPGRNN output layout retains frequency/time/channel/batch dimensions");
        CHECK(optimized->nb[0] == sizeof(float) && optimized->nb[1] == size_t(s.F) * sizeof(float) &&
              optimized->nb[2] == size_t(s.F) * s.T * sizeof(float) &&
              optimized->nb[3] == size_t(s.F) * s.T * s.C * sizeof(float),
              "DPGRNN output layout has canonical contiguous strides");
        ggml_tensor * copy = optimized->src[0];
        CHECK(optimized->op == GGML_OP_RESHAPE && copy->op == GGML_OP_CONT &&
              copy->src[0]->ne[0] == int64_t(s.F) * s.T && copy->src[0]->ne[1] == s.C &&
              copy->src[0]->ne[2] == s.B && copy->src[0]->nb[1] == sizeof(float),
              "portable DPGRNN copy exposes a batched 2-D transpose");
        CHECK(legacy->op == GGML_OP_CONT && legacy->src[0]->op == GGML_OP_PERMUTE &&
              legacy->src[0]->src[0] == x,
              "nonportable DPGRNN copy keeps the original direct permutation");
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, optimized);
        ggml_build_forward_expand(graph, legacy);
        CHECK(ggml_graph_compute_with_ctx(ctx, graph, 2) == GGML_STATUS_SUCCESS,
              "DPGRNN output layout graphs compute successfully");
        CHECK(std::memcmp(optimized->data, legacy->data, ggml_nbytes(optimized)) == 0,
              "portable DPGRNN output layout is byte-exact with the original graph");
        CHECK(std::memcmp(optimized->data, expected.data(), ggml_nbytes(optimized)) == 0,
              "DPGRNN output layout matches independent batch/channel/frequency/time tags");
        ggml_free(ctx);
    }
    std::printf("DPGRNN output layout: 8 byte-exact copy cases passed\n");
}

static void test_dpgrnn() {
    const int    C = 4, F = 5, T = 3, Hg = 3, Hi = 4;
    const float  eps = 1e-8f;
    std::mt19937 rng(2468);
    WBag         bag;
    auto add = [&](const std::string & n, std::vector<int> shp) {
        size_t sz = 1;
        for (int s : shp) sz *= (size_t) s;
        std::vector<float> v(sz);
        rand_fill(v, rng);
        bag.data[n]  = v;
        bag.shape[n] = std::move(shp);
    };
    auto add_gru = [&](const std::string & p, int I, int H, bool rev) {
        add(p + ".weight_ih_l0", { 3 * H, I }); add(p + ".weight_hh_l0", { 3 * H, H });
        add(p + ".bias_ih_l0", { 3 * H }); add(p + ".bias_hh_l0", { 3 * H });
        if (rev) {
            add(p + ".weight_ih_l0_reverse", { 3 * H, I }); add(p + ".weight_hh_l0_reverse", { 3 * H, H });
            add(p + ".bias_ih_l0_reverse", { 3 * H }); add(p + ".bias_hh_l0_reverse", { 3 * H });
        }
    };
    add_gru("dp.intra_rnn.rnn1", C / 2, Hg, true);
    add_gru("dp.intra_rnn.rnn2", C / 2, Hg, true);
    add("dp.intra_fc.weight", { C, 4 * Hg }); add("dp.intra_fc.bias", { C });
    add("dp.intra_ln.weight", { F * C }); add("dp.intra_ln.bias", { F * C });
    add_gru("dp.inter_rnn.rnn1", C / 2, Hi, false);
    add_gru("dp.inter_rnn.rnn2", C / 2, Hi, false);
    add("dp.inter_fc.weight", { C, 2 * Hi }); add("dp.inter_fc.bias", { C });
    add("dp.inter_ln.weight", { F * C }); add("dp.inter_ln.bias", { F * C });

    std::vector<float> x((size_t) C * T * F);
    rand_fill(x, rng);
    std::vector<float> ref = ref_dpgrnn(x, C, T, F, "dp", bag, eps);
    for (bool fused : { true, false }) {
        std::vector<float> got = run_dpgrnn(x, C, T, F, bag, eps, fused);
        std::vector<float> portable = run_dpgrnn(x, C, T, F, bag, eps, fused, true);
        CHECK(got.size() == portable.size() &&
              std::memcmp(got.data(), portable.data(), got.size() * sizeof(float)) == 0,
              fmsg("portable DPGRNN output layout preserves numerical graph result bytes", fused));
        float m = 0.0f;
        CHECK(got.size() == ref.size(), fmsg("dpgrnn output size", fused));
        for (size_t i = 0; i < ref.size() && i < got.size(); i++) m = std::max(m, std::fabs(got[i] - ref[i]));
        CHECK(m < 2e-4f, fmsg("dpgrnn matches scalar denoiser_core dpgrnn", fused));
    }
}

// Generic CPU-backend runner for a resolver-driven graph builder.
static std::vector<float> run_graph(const std::vector<float> & x, int F, int T, int C, const WBag & bag,
                                    const std::function<ggml_tensor *(ggml_context *, ggml_tensor *, const detail::WResolver &)> & build) {
    ggml_init_params p{ (size_t) 256 * 1024 * 1024, nullptr, false };
    ggml_context *   ctx = ggml_init(p);
    ggml_tensor *    tx  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, F, T, C);
    std::memcpy(tx->data, x.data(), ggml_nbytes(tx));
    std::map<std::string, ggml_tensor *> tmap;
    for (const auto & kv : bag.data) {
        const std::vector<int> & shp = bag.shape.at(kv.first);
        int64_t                  ne[4] = { 1, 1, 1, 1 };
        const int                nd    = (int) shp.size();
        for (int i = 0; i < nd; i++) ne[i] = shp[nd - 1 - i];
        ggml_tensor * t = ggml_new_tensor(ctx, GGML_TYPE_F32, nd, ne);
        std::memcpy(t->data, kv.second.data(), ggml_nbytes(t));
        tmap[kv.first] = t;
    }
    detail::WResolver Wr = [&](const std::string & n) -> ggml_tensor * {
        auto it = tmap.find(n);
        return it == tmap.end() ? nullptr : it->second;
    };
    ggml_tensor * y  = build(ctx, tx, Wr);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);
    ggml_build_forward_expand(gf, y);
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    std::vector<float> out((size_t) ggml_nelements(y));
    std::memcpy(out.data(), y->data, ggml_nbytes(y));
    ggml_free(ctx);
    return out;
}

static std::vector<float> ref_ctfa(const std::vector<float> & x, int C, int T, int F,
                                   const std::string & p, const WBag & W, int r) {
    auto X   = [&](int c, int t, int f) { return x[((size_t) c * T + t) * F + f]; };
    auto sig = [](float v) { return 1.0f / (1.0f + std::exp(-v)); };
    auto fcb = [&](const std::string & n) { return W.has(n) ? &W.d(n) : nullptr; };
    // TA
    std::vector<float> zt((size_t) T * C, 0.0f);
    for (int t = 0; t < T; t++)
        for (int c = 0; c < C; c++) {
            double s = 0.0;
            for (int f = 0; f < F; f++) { float v = X(c, t, f); s += (double) v * v; }
            zt[(size_t) t * C + c] = (float) (s / F);
        }
    const int          Hta = W.dim(p + ".ta_gru.weight_hh_l0", 1);
    std::vector<float> at  = ref_gru_seq(zt, T, C, Hta, W.d(p + ".ta_gru.weight_ih_l0"), W.d(p + ".ta_gru.weight_hh_l0"),
                                         W.d(p + ".ta_gru.bias_ih_l0"), W.d(p + ".ta_gru.bias_hh_l0"));
    at = ref_linear2d(at, T, Hta, W.d(p + ".ta_fc.weight"), C, fcb(p + ".ta_fc.bias"));
    for (float & v : at) v = sig(v);
    // FA
    const int          pad_len = (r - F % r) % r, Fp = F + pad_len, Hh = Fp / r;
    const int          Hfa     = W.dim(p + ".fa.gru.weight_hh_l0", 1);
    std::vector<float> af((size_t) T * F, 0.0f);
    for (int t = 0; t < T; t++) {
        std::vector<float> row((size_t) Hh * r, 0.0f);
        for (int h = 0; h < Hh; h++)
            for (int rr = 0; rr < r; rr++) {
                const int f = h * r + rr;
                double    s = 0.0;
                if (f < F) { for (int c = 0; c < C; c++) { float v = X(c, t, f); s += (double) v * v; } s /= C; }
                row[(size_t) h * r + rr] = (float) s;
            }
        std::vector<float> fwd = ref_gru_seq(row, Hh, r, Hfa, W.d(p + ".fa.gru.weight_ih_l0"), W.d(p + ".fa.gru.weight_hh_l0"),
                                             W.d(p + ".fa.gru.bias_ih_l0"), W.d(p + ".fa.gru.bias_hh_l0"));
        std::vector<float> rev = ref_gru_rev(row, Hh, r, Hfa, W.d(p + ".fa.gru.weight_ih_l0_reverse"), W.d(p + ".fa.gru.weight_hh_l0_reverse"),
                                             W.d(p + ".fa.gru.bias_ih_l0_reverse"), W.d(p + ".fa.gru.bias_hh_l0_reverse"));
        std::vector<float> y((size_t) Hh * 2 * Hfa);
        for (int h = 0; h < Hh; h++)
            for (int k = 0; k < Hfa; k++) {
                y[(size_t) h * 2 * Hfa + k]       = fwd[(size_t) h * Hfa + k];
                y[(size_t) h * 2 * Hfa + Hfa + k] = rev[(size_t) h * Hfa + k];
            }
        y = ref_linear2d(y, Hh, 2 * Hfa, W.d(p + ".fa.fc.weight"), r, fcb(p + ".fa.fc.bias"));
        for (int h = 0; h < Hh; h++)
            for (int rr = 0; rr < r; rr++) {
                const int f = h * r + rr;
                if (f < F) af[(size_t) t * F + f] = sig(y[(size_t) h * r + rr]);
            }
    }
    std::vector<float> o((size_t) C * T * F);
    for (int c = 0; c < C; c++)
        for (int t = 0; t < T; t++)
            for (int f = 0; f < F; f++)
                o[((size_t) c * T + t) * F + f] = at[(size_t) t * C + c] * X(c, t, f) * af[(size_t) t * F + f];
    return o;
}

static void test_ctfa() {
    const int    C = 4, T = 3, F = 7, r = 4;
    std::mt19937 rng(13579);
    WBag         bag;
    auto add = [&](const std::string & n, std::vector<int> shp) {
        size_t sz = 1;
        for (int s : shp) sz *= (size_t) s;
        std::vector<float> v(sz);
        rand_fill(v, rng);
        bag.data[n]  = v;
        bag.shape[n] = std::move(shp);
    };
    const int Hta = 2 * C, Hfa = r;
    add("ct.ta_gru.weight_ih_l0", { 3 * Hta, C }); add("ct.ta_gru.weight_hh_l0", { 3 * Hta, Hta });
    add("ct.ta_gru.bias_ih_l0", { 3 * Hta }); add("ct.ta_gru.bias_hh_l0", { 3 * Hta });
    add("ct.ta_fc.weight", { C, Hta }); add("ct.ta_fc.bias", { C });
    for (const char * suf : { "", "_reverse" }) {
        add(std::string("ct.fa.gru.weight_ih_l0") + suf, { 3 * Hfa, r });
        add(std::string("ct.fa.gru.weight_hh_l0") + suf, { 3 * Hfa, Hfa });
        add(std::string("ct.fa.gru.bias_ih_l0") + suf, { 3 * Hfa });
        add(std::string("ct.fa.gru.bias_hh_l0") + suf, { 3 * Hfa });
    }
    add("ct.fa.fc.weight", { r, 2 * Hfa }); add("ct.fa.fc.bias", { r });

    std::vector<float> x((size_t) C * T * F);
    rand_fill(x, rng);
    std::vector<float> ref = ref_ctfa(x, C, T, F, "ct", bag, r);
    for (bool fused : { true, false }) {
        std::vector<float> got = run_graph(x, F, T, C, bag,
                                           [&](ggml_context * ctx, ggml_tensor * tx, const detail::WResolver & Wr) {
                                               return detail::ctfa(ctx, tx, "ct", Wr, r, fused);
                                           });
        float m = 0.0f;
        CHECK(got.size() == ref.size(), fmsg("ctfa output size", fused));
        for (size_t i = 0; i < ref.size() && i < got.size(); i++) m = std::max(m, std::fabs(got[i] - ref[i]));
        CHECK(m < 2e-4f, fmsg("ctfa matches scalar denoiser_core ctfa", fused));
    }
    int mean_queries = 0;
    detail::MeanSupport support = [&](const ggml_tensor * candidate) {
        ++mean_queries;
        CHECK(candidate->op == GGML_OP_MEAN && candidate->type == GGML_TYPE_F32,
              "portable ctfa queries both actual F32 mean candidates");
        return true;
    };
    std::vector<float> selected = run_graph(x, F, T, C, bag,
        [&](ggml_context * ctx, ggml_tensor * tx, const detail::WResolver & Wr) {
            return detail::ctfa(ctx, tx, "ct", Wr, r, false, nullptr, true, {}, support);
        });
    CHECK(mean_queries == 2, "portable ctfa propagates mean support to TA and channel energy");
    CHECK(selected.size() == ref.size(), "portable mean ctfa output size");
    float error = 0.0f;
    for (size_t i = 0; i < ref.size() && i < selected.size(); ++i)
        error = std::max(error, std::fabs(selected[i] - ref[i]));
    CHECK(error < 2e-4f, "portable mean ctfa retains scalar reference accuracy");
}

// End-to-end parity: DenoiserGgml::chunk_forward (ggml-CPU) vs scalar
// denoiser_net_forward on one zero-state chunk from the real GGUF.
static void test_e2e(const char * gguf_path) {
    tts_cpp::lavasr::DenoiserWeights w;
    std::string                      err;
    if (!tts_cpp::lavasr::load_denoiser_gguf(gguf_path, w, &err)) {
        std::fprintf(stderr, "SKIP e2e: cannot load %s (%s)\n", gguf_path, err.c_str());
        return;
    }
    const int    L = w.chunk_frames, F = w.spec_bins;
    std::mt19937 rng(20260708);
    auto         df = [&](std::vector<float> & v) {
        std::uniform_real_distribution<float> d(-2.0f, 2.0f);
        for (float & x : v) x = d(rng);
    };
    std::vector<float> re((size_t) L * F), im((size_t) L * F);
    df(re); df(im);
    std::vector<float> sr, si;
    tts_cpp::lavasr::denoiser_net_forward(w, re, im, L, sr, si);
    auto               g = tts_cpp::lavasr::DenoiserGgml::create(w, -1);
    std::vector<float> gr, gi;
    g->chunk_forward(re, im, L, gr, gi);
    auto rel = [](const std::vector<float> & a, const std::vector<float> & b, float & mx) {
        double num = 0, den = 0;
        mx = 0.0f;
        for (size_t i = 0; i < a.size(); i++) {
            float d = std::fabs(a[i] - b[i]);
            mx      = std::max(mx, d);
            num += (double) d * d;
            den += (double) a[i] * a[i];
        }
        return (float) std::sqrt(num / std::max(den, 1e-20));
    };
    float mr, mi;
    float rr = rel(sr, gr, mr), ri = rel(si, gi, mi);
    std::printf("e2e: real max_abs=%.3e nrmse=%.3e | imag max_abs=%.3e nrmse=%.3e\n", mr, rr, mi, ri);
    CHECK(gr.size() == sr.size() && gi.size() == si.size(), "e2e output size matches scalar");
    CHECK(rr < 2e-3f && ri < 2e-3f, "denoiser ggml chunk_forward matches scalar denoiser_net_forward");
}

// Chunk-batch consistency: batch_forward on N stacked chunks must equal
// chunk_forward on each chunk individually (proves the ne3 batching is correct).
static void test_batch(const char * gguf_path) {
    tts_cpp::lavasr::DenoiserWeights w;
    std::string                      err;
    if (!tts_cpp::lavasr::load_denoiser_gguf(gguf_path, w, &err)) return;
    const int    L = w.chunk_frames, F = w.spec_bins, N = 3;
    std::mt19937 rng(31337);
    auto         df = [&](std::vector<float> & v) {
        std::uniform_real_distribution<float> d(-2.0f, 2.0f);
        for (float & x : v) x = d(rng);
    };
    std::vector<float> re((size_t) N * L * F), im((size_t) N * L * F);
    df(re); df(im);
    auto g = tts_cpp::lavasr::DenoiserGgml::create(w, -1);
    std::vector<float> br, bi;
    g->batch_forward(re, im, L, N, br, bi);
    float m = 0.0f;
    for (int c = 0; c < N; c++) {
        std::vector<float> cre(re.begin() + (size_t) c * L * F, re.begin() + (size_t) (c + 1) * L * F);
        std::vector<float> cim(im.begin() + (size_t) c * L * F, im.begin() + (size_t) (c + 1) * L * F);
        std::vector<float> sr, si;
        g->chunk_forward(cre, cim, L, sr, si);
        for (size_t i = 0; i < sr.size(); i++) {
            m = std::max(m, std::fabs(br[(size_t) c * L * F + i] - sr[i]));
            m = std::max(m, std::fabs(bi[(size_t) c * L * F + i] - si[i]));
        }
    }
    std::printf("batch-consistency (N=%d): max_abs=%.3e\n", N, m);
    CHECK(m < 1e-4f, "batch_forward matches per-chunk chunk_forward");
}

// Uniform(-scale, scale) fill; '*.running_var' instead gets uniform(0.25, 1) —
// the BN fold computes g/sqrt(rv+eps), so rv must stay positive (scale ignored).
static void add_dn(DenoiserWeights & w, const std::string & n, std::vector<int> shape,
                   std::mt19937 & rng, float scale) {
    size_t sz = 1;
    for (int s : shape) sz *= (size_t) s;
    DnTensor t;
    t.data.resize(sz);
    const std::string suf = ".running_var";
    const bool var = n.size() > suf.size() && n.compare(n.size() - suf.size(), suf.size(), suf) == 0;
    std::uniform_real_distribution<float> d(var ? 0.25f : -scale, var ? 1.0f : scale);
    for (float & v : t.data) v = d(rng);
    t.shape = std::move(shape);
    w.t[n]  = std::move(t);
}

// Small but complete synthetic UL-UNAS: spec_bins=17, erb 5+8 (Fe=13 -> F1=7 -> F2=4),
// chunk 8/3.  0.5/sqrt(fan_in) weights keep the deep chain conditioned (mask unsaturated).
static DenoiserWeights make_synthetic_denoiser_weights(std::mt19937 & rng) {
    DenoiserWeights w;
    w.n_fft = 32; w.hop = 16; w.win = 32; w.spec_bins = 17; w.work_sample_rate = 16000;
    w.erb_low = 5; w.erb_high = 8; w.freq_comp_ratio = 4;
    w.chunk_frames = 8; w.chunk_hop = 3;
    w.bn_eps = 1e-5f; w.ln_eps = 1e-8f;

    const int r = w.freq_comp_ratio, F1 = 7, F2 = 4, Cd = 16;
    auto ws  = [](int fan_in) { return 0.5f / std::sqrt((float) fan_in); };
    auto add = [&](const std::string & n, std::vector<int> shp, float sc) { add_dn(w, n, std::move(shp), rng, sc); };
    auto bnw = [&](const std::string & p, int C) {
        add(p + ".weight", { C }, 0.1f); add(p + ".bias", { C }, 0.1f);
        add(p + ".running_mean", { C }, 0.1f); add(p + ".running_var", { C }, 0.0f);
    };
    auto apw = [&](const std::string & p, int C, int F) {
        add(p + ".affine_weight", { C, F }, 0.1f); add(p + ".affine_bias", { C, F }, 0.1f);
        add(p + ".slope_weight", { C }, 0.1f);
    };
    auto gruw = [&](const std::string & p, int I, int H, bool rev) {
        for (const char * suf : { "", "_reverse" }) {
            add(p + ".weight_ih_l0" + suf, { 3 * H, I }, ws(I));
            add(p + ".weight_hh_l0" + suf, { 3 * H, H }, ws(H));
            add(p + ".bias_ih_l0" + suf, { 3 * H }, 0.1f);
            add(p + ".bias_hh_l0" + suf, { 3 * H }, 0.1f);
            if (!rev) break;
        }
    };
    auto ctfaw = [&](const std::string & p, int C) {
        gruw(p + ".ta_gru", C, 2 * C, false);
        add(p + ".ta_fc.weight", { C, 2 * C }, ws(2 * C)); add(p + ".ta_fc.bias", { C }, 0.1f);
        gruw(p + ".fa.gru", r, r, true);
        add(p + ".fa.fc.weight", { r, 2 * r }, ws(2 * r)); add(p + ".fa.fc.bias", { r }, 0.1f);
    };
    auto fcw = [&](const std::string & p, int out, int in) {
        add(p + ".weight", { out, in }, ws(in)); add(p + ".bias", { out }, 0.1f);
    };
    auto convw = [&](const std::string & p, int cout, int cing, int kt, int kf) {
        add(p + ".weight", { cout, cing, kt, kf }, ws(cing * kt * kf)); add(p + ".bias", { cout }, 0.1f);
    };
    // Transposed kernels in raw PyTorch layout [Cin, Cout/g, kt, kf]; create() derives the .ct.
    auto convtw = [&](const std::string & p, int cin, int coutg, int kt, int kf, int g) {
        add(p + ".weight", { cin, coutg, kt, kf }, ws(cin / g * kt * kf));
        add(p + ".bias", { coutg * g }, 0.1f);
    };

    add("erb.erb_fc.weight", { w.erb_high, w.spec_bins - w.erb_low }, ws(w.spec_bins - w.erb_low));
    add("erb.ierb_fc.weight", { w.spec_bins - w.erb_low, w.erb_high }, ws(w.erb_high));
    { // encoder: XConv 1->12 s2, XMB 12->24 s2, XDWS 24->24, XMB 24->32, XDWS 32->16
        const std::string e0 = "encoder.en_convs.0";
        convw(e0 + ".ops.1", 12, 1, 3, 3);
        bnw(e0 + ".ops.2", 12); apw(e0 + ".ops.3", 12, F1); ctfaw(e0 + ".ops.4", 12);
        const std::string e1 = "encoder.en_convs.1";
        convw(e1 + ".pconv1.0", 24, 6, 1, 1); bnw(e1 + ".pconv1.1", 24); apw(e1 + ".pconv1.2", 24, F1);
        convw(e1 + ".dconv.1", 24, 1, 2, 3); bnw(e1 + ".dconv.2", 24); apw(e1 + ".dconv.3", 24, F2);
        convw(e1 + ".pconv2.0", 24, 12, 1, 1); bnw(e1 + ".pconv2.1", 24); ctfaw(e1 + ".pconv2.2", 24);
        const std::string e2 = "encoder.en_convs.2";
        convw(e2 + ".pconv.0", 24, 12, 1, 1); bnw(e2 + ".pconv.1", 24); apw(e2 + ".pconv.2", 24, F2);
        convw(e2 + ".dconv.1", 24, 1, 2, 3); bnw(e2 + ".dconv.2", 24); apw(e2 + ".dconv.3", 24, F2);
        ctfaw(e2 + ".dconv.4", 24);
        const std::string e3 = "encoder.en_convs.3";
        convw(e3 + ".pconv1.0", 32, 12, 1, 1); bnw(e3 + ".pconv1.1", 32); apw(e3 + ".pconv1.2", 32, F2);
        convw(e3 + ".dconv.1", 32, 1, 1, 5); bnw(e3 + ".dconv.2", 32); apw(e3 + ".dconv.3", 32, F2);
        convw(e3 + ".pconv2.0", 32, 16, 1, 1); bnw(e3 + ".pconv2.1", 32); ctfaw(e3 + ".pconv2.2", 32);
        const std::string e4 = "encoder.en_convs.4";
        convw(e4 + ".pconv.0", 16, 16, 1, 1); bnw(e4 + ".pconv.1", 16); apw(e4 + ".pconv.2", 16, F2);
        convw(e4 + ".dconv.1", 16, 1, 1, 5); bnw(e4 + ".dconv.2", 16); apw(e4 + ".dconv.3", 16, F2);
        ctfaw(e4 + ".dconv.4", 16);
    }
    for (int i = 0; i < 2; i++) { // bottleneck: 2x DPGRNN (C=16, F=F2)
        const std::string p = "dpgrnn." + std::to_string(i);
        gruw(p + ".intra_rnn.rnn1", Cd / 2, 4, true); gruw(p + ".intra_rnn.rnn2", Cd / 2, 4, true);
        fcw(p + ".intra_fc", Cd, Cd);
        add(p + ".intra_ln.weight", { F2 * Cd }, 0.1f); add(p + ".intra_ln.bias", { F2 * Cd }, 0.1f);
        gruw(p + ".inter_rnn.rnn1", Cd / 2, 8, false); gruw(p + ".inter_rnn.rnn2", Cd / 2, 8, false);
        fcw(p + ".inter_fc", Cd, Cd);
        add(p + ".inter_ln.weight", { F2 * Cd }, 0.1f); add(p + ".inter_ln.bias", { F2 * Cd }, 0.1f);
    }
    { // decoder mirror: XDWS 16->32, XMB 32->24, XDWS 24->24, XMB 24->12 s2, final XConv 12->1 s2
        const std::string d0 = "decoder.de_convs.0";
        convw(d0 + ".pconv.0", 32, 8, 1, 1); bnw(d0 + ".pconv.1", 32); apw(d0 + ".pconv.2", 32, F2);
        convtw(d0 + ".dconv.1", 32, 1, 1, 5, 32); bnw(d0 + ".dconv.2", 32); apw(d0 + ".dconv.3", 32, F2);
        ctfaw(d0 + ".dconv.4", 32);
        const std::string d1 = "decoder.de_convs.1";
        convw(d1 + ".pconv1.0", 24, 16, 1, 1); bnw(d1 + ".pconv1.1", 24); apw(d1 + ".pconv1.2", 24, F2);
        convtw(d1 + ".dconv.1", 24, 1, 1, 5, 24); bnw(d1 + ".dconv.2", 24); apw(d1 + ".dconv.3", 24, F2);
        convw(d1 + ".pconv2.0", 24, 12, 1, 1); bnw(d1 + ".pconv2.1", 24); ctfaw(d1 + ".pconv2.2", 24);
        const std::string d2 = "decoder.de_convs.2";
        convw(d2 + ".pconv.0", 24, 12, 1, 1); bnw(d2 + ".pconv.1", 24); apw(d2 + ".pconv.2", 24, F2);
        convtw(d2 + ".dconv.1", 24, 1, 2, 3, 24); bnw(d2 + ".dconv.2", 24); apw(d2 + ".dconv.3", 24, F2);
        ctfaw(d2 + ".dconv.4", 24);
        const std::string d3 = "decoder.de_convs.3";
        convw(d3 + ".pconv1.0", 12, 12, 1, 1); bnw(d3 + ".pconv1.1", 12); apw(d3 + ".pconv1.2", 12, F2);
        convtw(d3 + ".dconv.1", 12, 1, 2, 3, 12); bnw(d3 + ".dconv.2", 12); apw(d3 + ".dconv.3", 12, F1);
        convw(d3 + ".pconv2.0", 12, 6, 1, 1); bnw(d3 + ".pconv2.1", 12); ctfaw(d3 + ".pconv2.2", 12);
        const std::string d4 = "decoder.de_convs.4"; // is_last: no .ops.3 affine-PReLU
        convtw(d4 + ".ops.1", 12, 1, 3, 3, 1); bnw(d4 + ".ops.2", 1); ctfaw(d4 + ".ops.4", 1);
    }
    return w;
}

// Always-on e2e parity (no fixture): ggml chunk_forward vs the scalar
// denoiser_net_forward on the synthetic model, same metrics as test_e2e.
static void test_e2e_synthetic(int n_gpu_layers = -1, bool portable = false) {
    std::mt19937    rng(20260710);
    DenoiserWeights w = make_synthetic_denoiser_weights(rng);
    const int       L = w.chunk_frames, F = w.spec_bins;
    auto            df = [&](std::vector<float> & v) {
        std::uniform_real_distribution<float> d(-2.0f, 2.0f);
        for (float & x : v) x = d(rng);
    };
    std::vector<float> re((size_t) L * F), im((size_t) L * F);
    df(re); df(im);
    std::vector<float> sr, si;
    tts_cpp::lavasr::denoiser_net_forward(w, re, im, L, sr, si);
    auto               g = tts_cpp::lavasr::DenoiserGgml::create(w, n_gpu_layers, false, portable);
    if (n_gpu_layers > 0) {
        CHECK(g->is_gpu(), "available GPU must not fall back to CPU for the denoiser");
    }
    std::vector<float> gr, gi;
    g->chunk_forward(re, im, L, gr, gi);
    auto rel = [](const std::vector<float> & a, const std::vector<float> & b, float & mx) {
        double num = 0, den = 0;
        mx = 0.0f;
        for (size_t i = 0; i < a.size(); i++) {
            float d = std::fabs(a[i] - b[i]);
            mx      = std::max(mx, d);
            num += (double) d * d;
            den += (double) a[i] * a[i];
        }
        return (float) std::sqrt(num / std::max(den, 1e-20));
    };
    float mr, mi;
    float rr = rel(sr, gr, mr), ri = rel(si, gi, mi);
    float  omax = 0.0f;
    double oss  = 0.0;
    for (float v : sr) { omax = std::max(omax, std::fabs(v)); oss += (double) v * v; }
    const float orms = (float) std::sqrt(oss / std::max<double>((double) sr.size(), 1.0));
    std::printf("e2e-synthetic: real max_abs=%.3e nrmse=%.3e | imag max_abs=%.3e nrmse=%.3e | out rms=%.3e max=%.3e\n",
                mr, rr, mi, ri, orms, omax);
    CHECK(gr.size() == sr.size() && gi.size() == si.size(), "synthetic e2e output size matches scalar");
    CHECK(rr < 2e-3f && ri < 2e-3f, "synthetic denoiser ggml chunk_forward matches scalar");
    CHECK(omax > 1e-3f, "synthetic outputs are not all ~0 (mask not degenerate)");
    // Reusing the allocator must preserve results and reject malformed input
    // before reading it. This also exercises the portable HTP graph on CPU.
    std::vector<float> repeat_r, repeat_i;
    g->chunk_forward(re, im, L, repeat_r, repeat_i);
    CHECK(gr == repeat_r && gi == repeat_i, "repeated forward is deterministic");
    bool rejected = false;
    try { g->batch_forward(re, im, 0, 1, repeat_r, repeat_i); }
    catch (const std::invalid_argument &) { rejected = true; }
    CHECK(rejected, "nonpositive chunk length is rejected");
    rejected = false;
    try { g->batch_forward({}, im, L, 1, repeat_r, repeat_i); }
    catch (const std::invalid_argument &) { rejected = true; }
    CHECK(rejected, "truncated real plane is rejected");
}

static void test_gpu_e2e_synthetic() {
    constexpr int gpu_layers = 1;
    constexpr int vulkan_device = 0;
    constexpr bool allow_arm_mali = true;
    ggml_backend_t backend = tts_cpp::detail::init_gpu_backend(
        gpu_layers, false, "lavasr-test", vulkan_device, allow_arm_mali, nullptr,
        tts_cpp::lavasr::gpu_requirement());
    if (!backend) {
        std::printf("SKIP: LavaSR GPU parity requires an available GPU backend\n");
        return;
    }
    ggml_backend_free(backend);
    test_e2e_synthetic(gpu_layers);
}

// Always-on chunk-batch consistency on the synthetic model (style of test_batch).
static void test_batch_synthetic(bool portable = false) {
    std::mt19937    rng(20260710);
    DenoiserWeights w = make_synthetic_denoiser_weights(rng);
    const int       L = w.chunk_frames, F = w.spec_bins, N = 3;
    auto            df = [&](std::vector<float> & v) {
        std::uniform_real_distribution<float> d(-2.0f, 2.0f);
        for (float & x : v) x = d(rng);
    };
    std::vector<float> re((size_t) N * L * F), im((size_t) N * L * F);
    df(re); df(im);
    auto g = tts_cpp::lavasr::DenoiserGgml::create(w, -1, false, portable);
    std::vector<float> br, bi;
    g->batch_forward(re, im, L, N, br, bi);
    float m = 0.0f;
    for (int c = 0; c < N; c++) {
        std::vector<float> cre(re.begin() + (size_t) c * L * F, re.begin() + (size_t) (c + 1) * L * F);
        std::vector<float> cim(im.begin() + (size_t) c * L * F, im.begin() + (size_t) (c + 1) * L * F);
        std::vector<float> sr, si;
        g->chunk_forward(cre, cim, L, sr, si);
        for (size_t i = 0; i < sr.size(); i++) {
            m = std::max(m, std::fabs(br[(size_t) c * L * F + i] - sr[i]));
            m = std::max(m, std::fabs(bi[(size_t) c * L * F + i] - si[i]));
        }
    }
    std::printf("batch-consistency synthetic (N=%d, portable=%d): max_abs=%.3e\n", N, portable, m);
    CHECK(m < 1e-4f, "synthetic batch_forward matches per-chunk chunk_forward");
}

// Pipeline parity: denoise_with_batch_core must be BIT-EXACT vs denoise_with_core
// when both wrap the same element-wise core (metadata-only weights, no tensors).
static void test_pipeline_batch_parity() {
    DenoiserWeights w; // defaults = shipped model: 512/256 STFT, F=257, chunk 63/21, 16 kHz

    auto elemwise = [](const std::vector<float> & re, const std::vector<float> & im,
                       std::vector<float> & orr, std::vector<float> & oii) {
        orr.resize(re.size());
        oii.resize(im.size());
        for (size_t i = 0; i < re.size(); i++) {
            orr[i] = 0.5f * re[i] - 0.25f * im[i];
            oii[i] = 0.5f * im[i] + 0.25f * re[i];
        }
    };
    DenoiseChunkCore chunk_core = [&](const std::vector<float> & re, const std::vector<float> & im,
                                      int /*L*/, std::vector<float> & orr, std::vector<float> & oii) {
        elemwise(re, im, orr, oii);
    };
    DenoiseBatchCore batch_core = [&](const std::vector<float> & re, const std::vector<float> & im,
                                      int /*L*/, int /*n_chunks*/,
                                      std::vector<float> & orr, std::vector<float> & oii) {
        elemwise(re, im, orr, oii);
    };

    struct Run { int n, sr; const char * name; };
    const Run runs[] = {
        { 32500, 16000, "multi-chunk forced-tail" }, // T=127: starts 0,21,42,63 + forced 64
        { 4000, 16000, "single-chunk T<=L" },        // T=16 <= 63
        { 48000, 24000, "resampled 24kHz" },         // real resampling; T=126 multi-chunk
    };
    for (const Run & r : runs) {
        std::vector<float> pcm((size_t) r.n);
        for (int i = 0; i < r.n; i++) {
            const double ph = 2.0 * 3.14159265358979323846 * i / r.sr;
            pcm[i] = (float) (0.5 * std::sin(ph * 440.0) + 0.3 * std::sin(ph * 1337.0) +
                              0.2 * std::sin(ph * 3200.0));
        }
        const std::vector<float> a = denoise_with_core(w, pcm, r.sr, chunk_core);
        const std::vector<float> b = denoise_with_batch_core(w, pcm, r.sr, batch_core);
        CHECK(a.size() == pcm.size(), "pipeline parity: output length preserved");
        CHECK(a.size() == b.size(), "pipeline parity: sizes equal");
        float m = 0.0f;
        for (size_t i = 0; i < a.size() && i < b.size(); i++) m = std::max(m, std::fabs(a[i] - b[i]));
        const bool bitexact = a.size() == b.size() &&
                              std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
        std::printf("pipeline-parity %s: n=%zu bitexact=%d max_abs=%.3e\n",
                    r.name, a.size(), bitexact ? 1 : 0, m);
        CHECK(bitexact, "denoise_with_batch_core bit-exact vs denoise_with_core");
    }
}

int main(int argc, char ** argv) {
    test_gru_batched();
    test_gru_support_selection();
    test_conv2d();
    test_conv_support_selection();
    test_conv_transpose2d();
    test_affine_prelu();
    test_affine_support_selection();
    test_shuffle2();
    test_shuffle2_portable();
    test_mean_support_selection();
    test_channel_energy_mean();
    test_dpgrnn_output_layout();
    test_dpgrnn();
    test_ctfa();
    test_e2e_synthetic();
    test_e2e_synthetic(-1, true);
    test_gpu_e2e_synthetic();
    test_batch_synthetic();
    test_batch_synthetic(true);
    test_pipeline_batch_parity();
    if (argc > 1) { test_e2e(argv[1]); test_batch(argv[1]); }
    if (g_failures == 0) {
        std::printf("OK: all LavaSR denoiser ggml tests passed\n");
        return 0;
    }
    std::fprintf(stderr, "%d LavaSR denoiser ggml test(s) failed\n", g_failures);
    return 1;
}
