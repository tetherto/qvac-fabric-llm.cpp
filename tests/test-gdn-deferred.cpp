// Checks ggml_gated_delta_net_deferred against ggml_gated_delta_net over simulated speculative
// decoding cycles: each cycle runs a verify block, then rolls back a random number of tokens.
// The snapshot path keeps the true state by picking a snapshot slot; the deferred path keeps a
// base state, the recorded rows and a pending count. Both must produce the same outputs.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct gdn_cfg {
    int S;
    int H_k;
    int H;
    int n_seqs;
    int K;
    int G; // 1 = scalar gate per head, S = per-channel gate (KDA)
};

static void fill_unit_rows(std::vector<float> & x, int S, std::mt19937 & rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (size_t i = 0; i < x.size(); i += S) {
        float sum = 0.0f;
        for (int j = 0; j < S; ++j) {
            x[i + j] = nd(rng);
            sum += x[i + j] * x[i + j];
        }
        const float inv = 1.0f / std::sqrt(sum);
        for (int j = 0; j < S; ++j) {
            x[i + j] *= inv;
        }
    }
}

static bool run_cfg(ggml_backend_t backend, const gdn_cfg & c, int n_cycles, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);

    const int S = c.S, H = c.H, H_k = c.H_k, n_seqs = c.n_seqs, K = c.K, G = c.G;
    const int64_t n_rec   = ggml_gated_delta_net_rec_size(S, H_k, S, H, G, K);
    const int64_t n_state = (int64_t) S * S * H * n_seqs;

    // the true state for the snapshot path; base, record and pending count for the deferred path.
    // each seq owns two record rows (2*s and 2*s + 1) and alternates between them
    std::vector<float>   state_true(n_state);
    std::vector<float>   base(n_state);
    std::vector<float>   rec(n_rec * 2 * n_seqs, 0.0f);
    std::vector<int32_t> pending(n_seqs, 0);
    std::vector<int32_t> half(n_seqs, 0);

    for (auto & x : state_true) {
        x = 0.1f * ud(rng);
    }
    base = state_true;

    double max_diff = 0.0;
    bool   bitwise  = true;

    for (int cycle = 0; cycle < n_cycles; ++cycle) {
        // a long prompt first, then verify blocks of 1..K+1 tokens with an occasional long one
        int T;
        if (cycle == 0) {
            T = 2 * K + 3;
        } else if (cycle % 7 == 3) {
            T = K + 1 + (int) (rng() % (2 * K));
        } else {
            T = 1 + (int) (rng() % (K + 1));
        }

        ggml_init_params ip = { 32 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);

        ggml_tensor * q    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_k, T, n_seqs);
        ggml_tensor * k    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_k, T, n_seqs);
        ggml_tensor * v    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H,   T, n_seqs);
        ggml_tensor * g    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, G, H,   T, n_seqs);
        ggml_tensor * beta = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H,   T, n_seqs);
        ggml_tensor * st_a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, H, n_seqs);
        ggml_tensor * st_b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, S, H, n_seqs);
        ggml_tensor * rc   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_rec, 2 * n_seqs);
        ggml_tensor * idx  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 3, n_seqs);

        ggml_tensor * out_a = ggml_gated_delta_net(ctx, q, k, v, g, beta, st_a, K);
        ggml_tensor * out_b = ggml_gated_delta_net_deferred(ctx, q, k, v, g, beta, st_b, rc, idx, nullptr, nullptr, K);

        if (!ggml_backend_supports_op(backend, out_a) || !ggml_backend_supports_op(backend, out_b)) {
            ggml_free(ctx);
            printf("  S=%3d H_k=%d H=%2d seqs=%d K=%d G=%3d: not supported, skipped\n", S, H_k, H, n_seqs, K, G);
            return true;
        }

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out_a);
        ggml_build_forward_expand(gf, out_b);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) {
            fprintf(stderr, "failed to allocate tensors\n");
            ggml_free(ctx);
            return false;
        }

        std::vector<float> hq(ggml_nelements(q)), hk(ggml_nelements(k)), hv(ggml_nelements(v));
        std::vector<float> hg(ggml_nelements(g)), hb(ggml_nelements(beta));
        fill_unit_rows(hq, S, rng);
        fill_unit_rows(hk, S, rng);
        for (auto & x : hv) {
            x = ud(rng);
        }
        for (auto & x : hg) {
            x = -std::log1p(std::exp(2.0f * ud(rng))) * 0.5f; // log decay < 0
        }
        for (auto & x : hb) {
            x = 1.0f / (1.0f + std::exp(-2.0f * ud(rng)));
        }

        ggml_backend_tensor_set(q,    hq.data(), 0, ggml_nbytes(q));
        ggml_backend_tensor_set(k,    hk.data(), 0, ggml_nbytes(k));
        ggml_backend_tensor_set(v,    hv.data(), 0, ggml_nbytes(v));
        ggml_backend_tensor_set(g,    hg.data(), 0, ggml_nbytes(g));
        ggml_backend_tensor_set(beta, hb.data(), 0, ggml_nbytes(beta));
        ggml_backend_tensor_set(st_a, state_true.data(), 0, ggml_nbytes(st_a));
        ggml_backend_tensor_set(st_b, base.data(),       0, ggml_nbytes(st_b));
        ggml_backend_tensor_set(rc,   rec.data(),        0, ggml_nbytes(rc));
        std::vector<int32_t> hidx(3 * n_seqs);
        for (int s = 0; s < n_seqs; ++s) {
            hidx[3 * s + 0] = pending[s];
            hidx[3 * s + 1] = 2 * s + half[s];
            hidx[3 * s + 2] = 2 * s + 1 - half[s];
        }
        ggml_backend_tensor_set(idx,  hidx.data(),       0, ggml_nbytes(idx));

        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "graph compute failed\n");
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            return false;
        }

        std::vector<float> ra(ggml_nelements(out_a)), rb(ggml_nelements(out_b));
        ggml_backend_tensor_get(out_a, ra.data(), 0, ggml_nbytes(out_a));
        ggml_backend_tensor_get(out_b, rb.data(), 0, ggml_nbytes(out_b));

        const int64_t n_attn = (int64_t) S * H * T * n_seqs;
        for (int64_t i = 0; i < n_attn; ++i) {
            const double d = std::fabs((double) ra[i] - (double) rb[i]);
            max_diff = std::max(max_diff, d);
            if (ra[i] != rb[i]) {
                bitwise = false;
            }
        }

        // roll back a random number of tokens per sequence, at most what a snapshot can restore
        const int R = std::min(T, K);
        for (int s = 0; s < n_seqs; ++s) {
            const int r = (int) (rng() % std::min(T, K));
            const size_t n_seq_state = (size_t) S * S * H;
            memcpy(state_true.data() + s * n_seq_state,
                   ra.data() + n_attn + (size_t) r * n_state + s * n_seq_state, n_seq_state * sizeof(float));
            pending[s] = R - r;
        }
        memcpy(base.data(), rb.data() + n_attn, n_state * sizeof(float));
        ggml_backend_tensor_get(rc, rec.data(), 0, ggml_nbytes(rc));
        for (int s = 0; s < n_seqs; ++s) {
            half[s] = 1 - half[s];
        }

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);

        if (max_diff > 1e-4) {
            fprintf(stderr, "  cycle %d (T = %d): max diff %g\n", cycle, T, max_diff);
            return false;
        }
    }

    printf("  S=%3d H_k=%d H=%2d seqs=%d K=%d G=%3d: max diff %g%s\n", S, H_k, H, n_seqs, K, G, max_diff,
           bitwise ? " (bitwise)" : "");
    return true;
}

// checks ggml_gdn_conv_deferred over draft/verify/rollback cycles against a host reference that
// keeps the true conv state (the last 3 conv inputs of the kept tokens)
static bool run_conv(ggml_backend_t backend, int C, int head_dim, int n_norm, int n_seqs, int K, int n_cycles, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);

    const int   d_conv = 4;
    const int   n_win  = d_conv - 1;
    const float eps    = 1e-6f;
    const int64_t n_hist = (int64_t) (K + n_win) * C;

    std::vector<float> w(d_conv * C);
    for (auto & x : w) {
        x = ud(rng);
    }

    std::vector<float>   state(n_win * C * n_seqs);  // true window, [n_win][C] per seq
    std::vector<float>   hist(n_hist * 2 * n_seqs);  // two history rows per seq
    std::vector<int32_t> pending(n_seqs, 0);
    std::vector<int32_t> half(n_seqs, 0);

    // start from a non-zero state: put it in the read row with pending 0
    for (int s = 0; s < n_seqs; ++s) {
        for (int j = 0; j < n_win; ++j) {
            for (int c = 0; c < C; ++c) {
                const float v = ud(rng);
                state[(s * n_win + j) * C + c] = v;
                hist[(2 * s) * n_hist + j * C + c] = v;
            }
        }
    }

    double max_diff = 0.0;

    for (int cycle = 0; cycle < n_cycles; ++cycle) {
        const int T = cycle == 0 ? 2 * K + 3 : (cycle % 7 == 3 ? K + 1 + (int) (rng() % (2 * K)) : 1 + (int) (rng() % (K + 1)));

        ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);

        ggml_tensor * x  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, T, n_seqs);
        ggml_tensor * wt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_conv, C);
        ggml_tensor * h  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_hist, 2 * n_seqs);
        ggml_tensor * id = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 3, n_seqs);

        ggml_tensor * y = ggml_gdn_conv_deferred(ctx, x, wt, h, id, K, n_norm, head_dim, eps);

        if (!ggml_backend_supports_op(backend, y)) {
            ggml_free(ctx);
            printf("  conv C=%d seqs=%d K=%d: not supported, skipped\n", C, n_seqs, K);
            return true;
        }

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

        std::vector<float> hx(ggml_nelements(x));
        for (auto & v : hx) {
            v = ud(rng);
        }
        std::vector<int32_t> hidx(3 * n_seqs);
        for (int s = 0; s < n_seqs; ++s) {
            hidx[3 * s + 0] = pending[s];
            hidx[3 * s + 1] = 2 * s + half[s];
            hidx[3 * s + 2] = 2 * s + 1 - half[s];
        }

        ggml_backend_tensor_set(x,  hx.data(),   0, ggml_nbytes(x));
        ggml_backend_tensor_set(wt, w.data(),    0, ggml_nbytes(wt));
        ggml_backend_tensor_set(h,  hist.data(), 0, ggml_nbytes(h));
        ggml_backend_tensor_set(id, hidx.data(), 0, ggml_nbytes(id));

        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            return false;
        }

        std::vector<float> hy(ggml_nelements(y));
        ggml_backend_tensor_get(y, hy.data(), 0, ggml_nbytes(y));
        ggml_backend_tensor_get(h, hist.data(), 0, ggml_nbytes(h));

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);

        const int R = std::min(T, K);
        for (int s = 0; s < n_seqs; ++s) {
            // reference: conv input = [true window | x], then silu and per-head l2 norm
            auto ci = [&](int i, int c) {
                return i < n_win ? state[(s * n_win + i) * C + c] : hx[((size_t) s * T + (i - n_win)) * C + c];
            };
            std::vector<float> ref(C);
            for (int t = 0; t < T; ++t) {
                for (int c = 0; c < C; ++c) {
                    float sum = 0.0f;
                    for (int j = 0; j < d_conv; ++j) {
                        sum += ci(t + j, c) * w[c * d_conv + j];
                    }
                    ref[c] = sum / (1.0f + expf(-sum));
                }
                for (int hh = 0; hh < n_norm; ++hh) {
                    double sq = 0.0;
                    for (int c = 0; c < head_dim; ++c) {
                        sq += ref[hh * head_dim + c] * ref[hh * head_dim + c];
                    }
                    for (int c = 0; c < head_dim; ++c) {
                        ref[hh * head_dim + c] /= sqrtf((float) sq + eps);
                    }
                }
                for (int c = 0; c < C; ++c) {
                    max_diff = std::max(max_diff, (double) std::fabs(ref[c] - hy[((size_t) s * T + t) * C + c]));
                }
            }

            // roll back r of the T tokens; the window becomes the last 3 inputs of the kept prefix
            const int r = (int) (rng() % (R + 1));
            const int m = T - r;
            std::vector<float> win(n_win * C);
            for (int j = 0; j < n_win; ++j) {
                for (int c = 0; c < C; ++c) {
                    win[j * C + c] = ci(m + j, c);
                }
            }
            std::copy(win.begin(), win.end(), state.begin() + (size_t) s * n_win * C);
            pending[s] = R - r;
            half[s] = 1 - half[s];
        }

        if (max_diff > 1e-4) {
            fprintf(stderr, "  conv cycle %d (T = %d): max diff %g\n", cycle, T, max_diff);
            return false;
        }
    }

    printf("  conv C=%d head_dim=%d n_norm=%d seqs=%d K=%d: max diff %g\n", C, head_dim, n_norm, n_seqs, K, max_diff);
    return true;
}

int main() {
    ggml_backend_load_all();

    const std::vector<gdn_cfg> cfgs = {
        {  16, 2,  6, 1, 4,   1 },
        {  16, 2,  6, 3, 4,   1 },
        {  16, 2,  4, 2, 1,   1 },
        {  32, 1,  3, 2, 8,  32 },
        {  32, 2,  6, 2, 4,   1 },
        { 128, 2,  6, 2, 8,   1 },
        { 128, 4, 12, 1, 16,  1 },
        { 128, 16, 48, 1, 9,  1 }, // qwen3.5 27B linear-attention layer, draft block 8
    };

    bool ok = true;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            continue;
        }
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        printf("%s:\n", ggml_backend_dev_name(dev));
        for (size_t j = 0; j < cfgs.size(); ++j) {
            if (!run_cfg(backend, cfgs[j], 40, 1234 + (uint32_t) j)) {
                printf("  config %zu FAILED\n", j);
                ok = false;
            }
        }
        if (!run_conv(backend, 256, 32, 4, 2, 4, 40, 99)) {
            printf("  conv config 0 FAILED\n");
            ok = false;
        }
        if (!run_conv(backend, 10240, 128, 32, 1, 8, 40, 100)) {
            printf("  conv config 1 FAILED\n");
            ok = false;
        }
        ggml_backend_free(backend);
    }

    printf("%s\n", ok ? "OK" : "FAILED");
    return ok ? 0 : 1;
}
