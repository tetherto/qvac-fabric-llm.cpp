#include "xdna-gdn-mm.h"

#include "ggml-impl.h"
#include "xdna-runtime.h"
#include "xdna-seq.h"
#include "xdna-util.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// Host side of kernels/gdn_mm.py. Two arguments:
//   0  X  per column, per object, the four cores' input objects (IN_N bf16):
//         the header, the state (hi then lo, 4 objects), then a chunk an
//         object - K, Q, the core's V half in 8 x 8 tiles, then the chunk's
//         exponents (kernels/gdn-mm.cc)
//   1  O  per column, per object, the four cores' output objects (O_N f32):
//         a chunk's output half an object, then the state (hi then lo, 8)
// Core (c, 2 + i) holds head 2c + i / 2, value half i % 2. Both streams are
// channel 0 of their column's shim (read off the compiled sample stream).
//
// The scalars are the host's: the chunk's cumulative gates, log2 sigma and
// when the state is renormalized depend on the gates alone, so each head's
// are worked out here in order and the core does no scalar f32 arithmetic.

namespace {

constexpr const char * STEM       = "gdn_mm_c8";
constexpr int          COLS       = 8;
constexpr int          H          = 16;
constexpr int          S          = 128;
constexpr int          C          = 16;  // tokens a chunk
constexpr int          DK         = 128;
constexpr int          DV         = 64;
constexpr int          FX_N       = 4 * C + 2 * C * C + 32;  // f32 words of a chunk's factors (the core works them out)
constexpr int          IN_N       = 2 * C * DK + C * DV + FX_N * 2;  // bf16 an input object
constexpr int          O_N        = C * DV;                          // f32 an output object
constexpr int          N_STATE_IN = 4, N_STATE_OUT = 8;
constexpr float        LS_MIN         = -32.0f;
constexpr int64_t      GDN_MIN_TOKENS = 64;  // a sequence's tokens in the ubatch (see xdna_gdn_mm_supported)

// element (r, c) of a (rows x cols) matrix in 8 x 8 tiles
inline size_t t8(int r, int c, int cols) {
    return ((size_t) (r / 8) * (cols / 8) + c / 8) * 64 + (size_t) (r % 8) * 8 + c % 8;
}

struct runner {
    std::mutex                       mtx;
    xdna_kernel *                    kern      = nullptr;
    // X laid out ahead by xdna_gdn_mm_prepare for `prep` - its chunk
    // objects' K, Q and V - in a buffer of the runner's own
    const ggml_tensor *              prep      = nullptr;
    xdna_buffer *                    prep_bo   = nullptr;
    size_t                           prep_cap  = 0;
    // the conv input on the array (kernels/gdn_conv.py): its kernel, the
    // projection's rows (KW - 1 state rows first), the headers
    xdna_kernel *                    cv_kern   = nullptr;
    xdna_buffer *                    cv_in     = nullptr;
    size_t                           cv_in_cap = 0;
    xdna_buffer *                    cv_hdr    = nullptr;
    // the nodes whose attention output stays in their output buffer
    // (xdna_gdn_mm_keep), and the last such run's
    std::vector<const ggml_tensor *> keep;
    const ggml_tensor *              kept      = nullptr;
    xdna_buffer *                    kept_bo   = nullptr;
    xdna_kernel_pool *               kept_pool = nullptr;
    size_t                           kept_col  = 0;
    int                              kept_tok  = 0;
};

runner g_gm;

bool artifact_present() {
    static const bool present = !xdna_artifact_find(STEM, false).xclbin.empty();
    return present;
}

xdna_bd linear_bd(uint32_t words) {
    xdna_bd bd;
    bd.buf_len   = words;
    bd.d0_stride = bd.d1_stride = bd.d2_stride = 1;
    bd.ax_cache                                = 2;
    return bd;
}

std::vector<uint32_t> build_seq(int nx, int no) {
    xdna_seq       seq;
    const uint32_t x_col = (uint32_t) nx * 4 * IN_N * 2;
    const uint32_t o_col = (uint32_t) no * 4 * O_N * 4;
    for (int c = 0; c < COLS; c++) {
        xdna_bd bd = linear_bd(x_col / 4);
        xdna_seq_blockwrite(&seq, c, 0, 0, &bd);
        xdna_seq_ddr_patch(&seq, c, 0, 0, 0, (uint32_t) c * x_col);
        xdna_seq_push_queue(&seq, c, 0, 0, xdna_dma_dir::MM2S, 0, false, 0);
    }
    for (int c = 0; c < COLS; c++) {
        xdna_bd bd = linear_bd(o_col / 4);
        xdna_seq_blockwrite(&seq, c, 0, 1, &bd);
        xdna_seq_ddr_patch(&seq, c, 0, 1, 1, (uint32_t) c * o_col);
        xdna_seq_issue_token(&seq, c, 0, xdna_dma_dir::S2MM, 0, 0xF);
        xdna_seq_push_queue(&seq, c, 0, 1, xdna_dma_dir::S2MM, 0, true, 0);
    }
    for (int c = 0; c < COLS; c++) {
        xdna_seq_wait_token(&seq, c, 0, xdna_dma_dir::S2MM, 0);
    }
    return xdna_seq_build(&seq);
}

}  // namespace

bool xdna_gdn_mm_supported(const ggml_tensor * node) {
    if (!node || node->op != GGML_OP_GATED_DELTA_NET || xdna_env_int("GGML_XDNA_GDN_MM", 1) == 0 ||
        !artifact_present()) {
        return false;
    }
    const ggml_tensor * q = node->src[0];
    const ggml_tensor * k = node->src[1];
    const ggml_tensor * v = node->src[2];
    const ggml_tensor * g = node->src[3];
    const ggml_tensor * b = node->src[4];
    const ggml_tensor * s = node->src[5];
    if (!q || !k || !v || !g || !b || !s) {
        return false;
    }
    // A single token is the decode's, whose layers run fused. Below a pass
    // of 64 tokens the host's recurrence is what runs: 2-3x quicker there
    // (the array's run is two design changes a layer, conv and GDN, ~2.4 ms
    // each), and it gives a token the same bits however the server cut its
    // sequence into ubatches or batched it with others. The chunked form
    // here depends on where the ubatch boundaries fall, so a request served
    // alongside others (its prompt in other pieces) would not reproduce a
    // lone one. Energy breaks even with the host near 128 tokens.
    if (v->ne[2] < GDN_MIN_TOKENS) {
        return false;
    }
    for (const ggml_tensor * t : { q, k, v, g, b, s }) {
        if (t->type != GGML_TYPE_F32) {
            return false;
        }
    }
    if (q->ne[0] != S || q->ne[1] != H || k->ne[0] != S || k->ne[1] != H || v->ne[0] != S || v->ne[1] != H ||
        g->ne[0] != 1 || b->ne[0] != 1) {
        return false;  // the per-head scalar gate only (no KDA vectors)
    }
    if (s->ne[0] != S || s->ne[1] != S || s->ne[2] != H || v->ne[3] != 1 || s->ne[3] != 1 || q->ne[3] != 1 ||
        k->ne[3] != 1 || ggml_get_op_params_i32(node, 0) != 1) {
        return false;  // one sequence, one snapshot
    }
    if (!ggml_is_contiguous(g) || !ggml_is_contiguous(b) || !ggml_is_contiguous(s) || q->nb[0] != sizeof(float) ||
        k->nb[0] != sizeof(float) || v->nb[0] != sizeof(float)) {
        return false;
    }
    return true;
}

bool xdna_gdn_mm_run(xdna_kernel_pool * pool, ggml_tensor * node) {
    std::lock_guard<std::mutex> lock(g_gm.mtx);
    const ggml_tensor *         q     = node->src[0];
    const ggml_tensor *         k     = node->src[1];
    const ggml_tensor *         v     = node->src[2];
    const ggml_tensor *         g     = node->src[3];
    const ggml_tensor *         b     = node->src[4];
    const ggml_tensor *         s     = node->src[5];
    const int                   n_tok = (int) v->ne[2];
    const int                   nch   = (n_tok + C - 1) / C;
    const int                   nx    = 1 + N_STATE_IN + nch;
    const int                   no    = nch + N_STATE_OUT;

    if (!g_gm.kern) {
        g_gm.kern = xdna_kernel_find(pool->device, STEM);
        if (!g_gm.kern) {
            return false;
        }
    }
    const std::vector<uint32_t> insts = build_seq(nx, no);
    if (!xdna_kernel_bind_insts(pool->device, g_gm.kern, insts.data(), insts.size())) {
        return false;                             // xdna-runtime names the stream and the reason
    }
    const size_t x_col = (size_t) nx * 4 * IN_N;  // bf16 a column
    const size_t o_col = (size_t) no * 4 * O_N;   // f32 a column
    if (g_gm.kept_bo) {
        xdna_kernel_pool_release_buffer(g_gm.kept_pool, g_gm.kept_bo);
        g_gm.kept_bo = nullptr;
        g_gm.kept    = nullptr;
    }
    const bool keep     = std::find(g_gm.keep.begin(), g_gm.keep.end(), node) != g_gm.keep.end();
    const bool prepared = g_gm.prep == node && g_gm.prep_cap >= COLS * x_col * 2;
    g_gm.prep           = nullptr;
    xdna_buffer * bo_x  = prepared ? g_gm.prep_bo : xdna_kernel_pool_acquire_buffer(pool, COLS * x_col * 2);
    xdna_buffer * bo_o  = xdna_kernel_pool_acquire_buffer(pool, COLS * o_col * 4);
    if (!bo_x || !bo_o || !bo_x->data || !bo_o->data) {
        return false;
    }

    // each head's gates in log2 units, their chunk sums, log2 sigma before
    // each chunk and the renormalizations
    auto gate = [&](int t, int h) -> float {
        return t < n_tok ? *(const float *) ((const char *) g->data + t * g->nb[2] + h * g->nb[1]) : 0.0f;
    };
    auto betaf = [&](int t, int h) -> float {
        return t < n_tok ? *(const float *) ((const char *) b->data + t * b->nb[2] + h * b->nb[1]) : 0.0f;
    };
    std::vector<float>   ls_at((size_t) H * (nch + 1));
    std::vector<int32_t> shift((size_t) H * nch);
    for (int h = 0; h < H; h++) {
        float ls = 0.0f;
        for (int c = 0; c < nch; c++) {
            ls_at[(size_t) h * (nch + 1) + c] = ls;
            float sum                         = 0.0f;
            for (int t = 0; t < C; t++) {
                sum += gate(c * C + t, h) * (float) M_LOG2E;
            }
            float     ls_new            = ls + sum;
            const int e                 = ls_new < LS_MIN ? (int) (-ls_new) : 0;
            shift[(size_t) h * nch + c] = e;
            ls                          = ls_new + (float) e;
        }
        ls_at[(size_t) h * (nch + 1) + nch] = ls;
    }

    // A job is one head's object: the head's two cores (value halves) share
    // K, Q and the exponents, so those are worked out once. Each object is
    // built in local memory and copied into the BO whole.
    uint16_t * xh = (uint16_t *) bo_x->data;
#pragma omp parallel for num_threads(xdna_host_threads())
    for (int job = 0; job < COLS * 2 * nx; job++) {
        const int            col = job / (2 * nx), p = (job / nx) % 2, ob = job % nx;
        const int            hd = 2 * col + p;
        alignas(64) uint16_t buf[2][IN_N];
        if (!prepared || ob == 0) {
            std::memset(buf, 0, sizeof(buf));
        } else {
            // only the exponents' part is sent from here
            constexpr size_t kqv = 2 * C * DK + C * DV;
            std::memset(buf[0] + kqv, 0, (IN_N - kqv) * 2);
        }
        auto obj = [&](int half) {
            return xh + (size_t) col * x_col + ((size_t) ob * 4 + (size_t) 2 * p + half) * IN_N;
        };
        if (ob == 0) {
            // the chunks; sigma after the last, the new state's factor (a
            // head's two cores the same)
            const int32_t nch32 = nch;
            const float   sigma = std::exp2(ls_at[(size_t) hd * (nch + 1) + nch]);
            std::memcpy(buf[0], &nch32, sizeof(nch32));
            std::memcpy((char *) buf[0] + sizeof(nch32), &sigma, sizeof(sigma));
            std::memcpy(obj(0), buf[0], sizeof(buf[0]));
            std::memcpy(obj(1), buf[0], sizeof(buf[0]));
            continue;
        }
        if (ob <= N_STATE_IN) {
            // the state as ggml keeps it (s is M[j][i] = S[i][j]): each core
            // its 64 value columns, 16 of them (rows of 128 f32) an object
            const float * sm = (const float *) ((const char *) s->data + hd * s->nb[2]);
            for (int half = 0; half < 2; half++) {
                std::memcpy(obj(half), sm + (size_t) (half * DV + (ob - 1) * 16) * S, (size_t) 16 * S * sizeof(float));
            }
            continue;
        }
        const int  c  = ob - 1 - N_STATE_IN;
        uint16_t * ko = buf[0];
        uint16_t * qo = buf[0] + (size_t) C * DK;
        float *    fx = (float *) (buf[0] + (size_t) 2 * C * DK + (size_t) C * DV);
        for (int tt = 0; tt < C && !prepared; tt++) {
            const int t = c * C + tt;
            if (t >= n_tok) {
                continue;
            }
            const float * kr = (const float *) ((const char *) k->data + t * k->nb[2] + hd * k->nb[1]);
            const float * qr = (const float *) ((const char *) q->data + t * q->nb[2] + hd * q->nb[1]);
            const float * vr = (const float *) ((const char *) v->data + t * v->nb[2] + hd * v->nb[1]);
            for (int d8 = 0; d8 < DK; d8 += 8) {
                uint16_t * kd = ko + t8(tt, d8, DK);
                uint16_t * qd = qo + t8(tt, d8, DK);
                for (int j = 0; j < 8; j++) {
                    kd[j] = xdna_bf16(kr[d8 + j]);
                    qd[j] = xdna_bf16(qr[d8 + j]);
                }
            }
            for (int half = 0; half < 2; half++) {
                uint16_t * vo = buf[half] + (size_t) 2 * C * DK;
                for (int d8 = 0; d8 < DV; d8 += 8) {
                    uint16_t * vd = vo + t8(tt, d8, DV);
                    for (int j = 0; j < 8; j++) {
                        vd[j] = xdna_bf16(vr[half * DV + d8 + j]);
                    }
                }
            }
        }
        // what the core works its factors out of (gdn_factors): the gates,
        // beta, log2 sigma before and after the chunk's renormalization and
        // its shift
        for (int tt = 0; tt < C; tt++) {
            fx[tt]     = gate(c * C + tt, hd);
            fx[C + tt] = betaf(c * C + tt, hd);
        }
        fx[(size_t) 2 * C]          = ls_at[(size_t) hd * (nch + 1) + c];
        fx[2 * C + 1]               = ls_at[(size_t) hd * (nch + 1) + c + 1];
        ((int32_t *) fx)[2 * C + 2] = shift[(size_t) hd * nch + c];
        if (prepared) {
            // K, Q and V are in place: only the exponents
            constexpr size_t kqv = 2 * C * DK + C * DV;
            std::memcpy(obj(0) + kqv, buf[0] + kqv, (IN_N - kqv) * 2);
            std::memcpy(obj(1) + kqv, buf[0] + kqv, (IN_N - kqv) * 2);
            continue;
        }
        // the other half: the same K, Q and exponents
        std::memcpy(buf[1], buf[0], (size_t) 2 * C * DK * 2);
        std::memcpy(buf[1] + (size_t) 2 * C * DK + (size_t) C * DV, buf[0] + (size_t) 2 * C * DK + (size_t) C * DV,
                    (size_t) FX_N * 4);
        std::memcpy(obj(0), buf[0], sizeof(buf[0]));
        std::memcpy(obj(1), buf[1], sizeof(buf[1]));
    }
    // the pool's BO may be larger
    if (!xdna_buffer_sync_to_device_range(bo_x, COLS * x_col * 2, 0)) {
        return false;
    }

    xdna_buffer * args[2] = { bo_x, bo_o };
    xrt::run      run     = xdna_kernel_run_start(g_gm.kern, args, 2);
    const bool    run_ok  = xdna_run_wait(run);
    const bool    ok      = run_ok && xdna_buffer_sync_from_device_range(bo_o, COLS * o_col * 4, 0);
    if (ok) {
        const float * oh    = (const float *) bo_o->data;
        float *       attn  = (float *) node->data;
        float *       s_out = attn + (size_t) S * H * n_tok;
#pragma omp parallel for num_threads(xdna_host_threads())
        for (int job = 0; job < COLS * 4; job++) {
            const int col = job / 4, i = job % 4;
            const int hd = 2 * col + i / 2, half = i % 2;
            for (int c = 0; c < nch && !keep; c++) {
                const float * ob = oh + (size_t) col * o_col + ((size_t) c * 4 + i) * O_N;
                for (int tt = 0; tt < C; tt++) {
                    const int t = c * C + tt;
                    if (t >= n_tok) {
                        break;
                    }
                    // the core leaves its value half of each token as a row
                    std::memcpy(attn + ((size_t) t * H + hd) * S + (size_t) half * DV, ob + (size_t) tt * DV,
                                DV * sizeof(float));
                }
            }
            // the state, as ggml keeps it: 8 value columns an object
            const float * st = oh + (size_t) col * o_col + (size_t) nch * 4 * O_N;
            float *       sm = s_out + (size_t) hd * S * S;
            for (int p = 0; p < N_STATE_OUT; p++) {
                std::memcpy(sm + (size_t) (half * DV + p * 8) * S, st + ((size_t) p * 4 + i) * O_N,
                            (size_t) O_N * sizeof(float));
            }
        }
    }
    if (!prepared) {
        xdna_kernel_pool_release_buffer(pool, bo_x);
    }
    if (ok && keep) {
        g_gm.kept      = node;
        g_gm.kept_bo   = bo_o;
        g_gm.kept_pool = pool;
        g_gm.kept_col  = o_col;
        g_gm.kept_tok  = n_tok;
    } else {
        xdna_kernel_pool_release_buffer(pool, bo_o);
    }
    return ok;
}

void xdna_gdn_mm_keep_clear(void) {
    std::lock_guard<std::mutex> lock(g_gm.mtx);
    g_gm.keep.clear();
}

void xdna_gdn_mm_keep(const ggml_tensor * node) {
    std::lock_guard<std::mutex> lock(g_gm.mtx);
    g_gm.keep.push_back(node);
}

bool xdna_gdn_mm_rows(const ggml_tensor * node, xdna_gdn_rows * rows) {
    std::lock_guard<std::mutex> lock(g_gm.mtx);
    if (!node || g_gm.kept != node || !g_gm.kept_bo) {
        return false;
    }
    rows->base  = (const float *) g_gm.kept_bo->data;
    rows->o_col = g_gm.kept_col;
    rows->n_tok = g_gm.kept_tok;
    return true;
}

bool xdna_gdn_mm_materialize(const ggml_tensor * node) {
    std::lock_guard<std::mutex> lock(g_gm.mtx);
    if (!node || g_gm.kept != node || !g_gm.kept_bo) {
        return false;
    }
    const float * oh    = (const float *) g_gm.kept_bo->data;
    float *       attn  = (float *) node->data;
    const int     n_tok = g_gm.kept_tok, nch = (n_tok + C - 1) / C;
    const size_t  o_col = g_gm.kept_col;
#pragma omp parallel for num_threads(xdna_host_threads())
    for (int job = 0; job < COLS * 4; job++) {
        const int col = job / 4, i = job % 4;
        const int hd = 2 * col + i / 2, half = i % 2;
        for (int c = 0; c < nch; c++) {
            const float * ob = oh + (size_t) col * o_col + ((size_t) c * 4 + i) * O_N;
            for (int tt = 0; tt < C && c * C + tt < n_tok; tt++) {
                std::memcpy(attn + ((size_t) (c * C + tt) * H + hd) * S + (size_t) half * DV, ob + (size_t) tt * DV,
                            DV * sizeof(float));
            }
        }
    }
    return true;
}

// The conv input pass is host arithmetic over every channel of every token;
// this backend only runs where an XDNA NPU is, and those ship with Zen 4 and
// Zen 5 cores, so it is built for AVX2 (not FMA: the taps keep ggml's
// separate products and sums).
#pragma GCC push_options
#pragma GCC target("avx2")

namespace {

// f32 1 / (1 + exp(-x)): exp(-x) = 2^n 2^f with xdna_exp2_poly for 2^f, in
// arithmetic the compiler vectorizes
inline float sigmoid_f(float x) {
    const float t  = -x * 1.44269504f;
    // n by the 1.5 * 2^23 add; clamped as an integer (a float clamp is a
    // branch the vectorizer will not take), which past |t| = 126 leaves the
    // sigmoid as saturated as the true one
    const float n  = (t + 12582912.0f) - 12582912.0f;
    const float f  = t - n;
    int32_t     ni = (int32_t) n;
    ni             = ni < -126 ? -126 : ni;
    ni             = ni > 126 ? 126 : ni;
    const float p  = xdna_exp2_poly(f);
    int32_t     bits;
    std::memcpy(&bits, &p, 4);
    bits += (int32_t) ((uint32_t) ni << 23);
    float e;
    std::memcpy(&e, &bits, 4);
    return 1.0f / (1.0f + e);
}

}  // namespace

bool xdna_gdn_mm_prepare(xdna_kernel_pool * pool, const ggml_tensor * node, const xdna_gdn_conv_in & in) {
    std::lock_guard<std::mutex> lock(g_gm.mtx);
    g_gm.prep              = nullptr;
    const ggml_tensor * x  = in.x;
    const ggml_tensor * st = in.state;
    const ggml_tensor * w  = in.w;
    const int64_t       KW = w->ne[0], CH = w->ne[1];
    const int           n_tok = (int) x->ne[0];
    const int           nch   = (n_tok + C - 1) / C;
    const int           nx    = 1 + N_STATE_IN + nch;
    const size_t        x_col = (size_t) nx * 4 * IN_N;
    const size_t        need  = COLS * x_col * 2;
    if (KW > 8 || x->nb[1] != sizeof(float) || in.v_off + (int64_t) H * S > CH) {
        return false;
    }
    if (g_gm.prep_cap < need) {
        if (g_gm.prep_bo) {
            xdna_buffer_free(g_gm.prep_bo);
        }
        g_gm.prep_bo  = xdna_buffer_alloc(pool->device, need);
        g_gm.prep_cap = g_gm.prep_bo ? need : 0;
        if (!g_gm.prep_bo) {
            return false;
        }
    }
    auto xin = [&](int64_t j, int64_t ch) -> float {  // the conv input: the state, then the tokens
        return j < 0 ? *(const float *) ((const char *) st->data + (j + KW - 1) * st->nb[0] + ch * st->nb[1]) :
                       *(const float *) ((const char *) x->data + j * x->nb[0] + ch * sizeof(float));
    };
    uint16_t *         xh = (uint16_t *) g_gm.prep_bo->data;
    // the taps [tap][channel], so a tap's weights for a run of channels are
    // one vector
    std::vector<float> wt((size_t) KW * CH);
    for (int64_t ch = 0; ch < CH; ch++) {
        for (int64_t i = 0; i < KW; i++) {
            wt[(size_t) i * CH + ch] = *(const float *) ((const char *) w->data + i * w->nb[0] + ch * w->nb[1]);
        }
    }
    // a job is one head's chunk: its 16 tokens' conv, SiLU and (Q, K) norms,
    // into the head's two cores' objects in bf16 8 x 8 tiles
#pragma omp parallel for collapse(2) num_threads(xdna_host_threads())
    for (int hd = 0; hd < H; hd++) {
        for (int c = 0; c < nch; c++) {
            alignas(64) uint16_t kq[2 * C * DK];
            alignas(64) uint16_t vv[2][C * DV];
            std::memset(kq, 0, sizeof(kq));
            std::memset(vv, 0, sizeof(vv));
            for (int tt = 0; tt < C; tt++) {
                const int64_t t = (int64_t) c * C + tt;
                if (t >= n_tok) {
                    break;
                }
                alignas(64) float y[3][S];
                const int64_t     off[3] = { in.q_off + (int64_t) hd * S, in.k_off + (int64_t) hd * S,
                                             in.v_off + (int64_t) hd * S };
                for (int part = 0; part < 3; part++) {
                    float * yp = y[part];
                    if (t >= KW - 1) {
                        // the taps summed in order, as ggml's conv
                        for (int d = 0; d < S; d++) {
                            yp[d] = 0.0f;
                        }
                        for (int64_t i = 0; i < KW; i++) {
                            const float * r =
                                (const float *) ((const char *) x->data + (t - KW + 1 + i) * x->nb[0]) + off[part];
                            const float * wi = wt.data() + (size_t) i * CH + off[part];
#pragma omp simd
                            for (int d = 0; d < S; d++) {
                                yp[d] += r[d] * wi[d];
                            }
                        }
                    } else {
                        for (int d = 0; d < S; d++) {
                            const int64_t ch  = off[part] + d;
                            float         sum = 0.0f;
                            for (int64_t i = 0; i < KW; i++) {
                                sum += xin(t - KW + 1 + i, ch) * wt[(size_t) i * CH + ch];
                            }
                            yp[d] = sum;
                        }
                    }
#pragma omp simd
                    for (int d = 0; d < S; d++) {
                        yp[d] = yp[d] * sigmoid_f(yp[d]);
                    }
                }
                for (int part = 0; part < 2; part++) {
                    const float * yp = y[part];
                    double        ss = 0.0;
                    for (int d = 0; d < S; d++) {
                        ss += (double) (yp[d] * yp[d]);
                    }
                    const float          scale = 1.0f / fmaxf(sqrtf((float) ss), part == 0 ? in.eps_q : in.eps_k);
                    alignas(64) uint16_t hb[S];
#pragma omp simd
                    for (int d = 0; d < S; d++) {
                        hb[d] = xdna_bf16(yp[d] * scale);
                    }
                    // Q is the object's second part, K its first
                    uint16_t * o = kq + (part == 0 ? C * DK : 0);
                    for (int d8 = 0; d8 < S; d8 += 8) {
                        std::memcpy(o + t8(tt, d8, DK), hb + d8, 16);
                    }
                }
                {
                    alignas(64) uint16_t hb[S];
                    const float *        yp = y[2];
#pragma omp simd
                    for (int d = 0; d < S; d++) {
                        hb[d] = xdna_bf16(yp[d]);
                    }
                    for (int half = 0; half < 2; half++) {
                        for (int d8 = 0; d8 < DV; d8 += 8) {
                            std::memcpy(vv[half] + t8(tt, d8, DV), hb + (size_t) half * DV + d8, 16);
                        }
                    }
                }
            }
            const int col = hd / 2, ob = 1 + N_STATE_IN + c;
            for (int half = 0; half < 2; half++) {
                const int  i = 2 * (hd % 2) + half;
                uint16_t * o = xh + (size_t) col * x_col + ((size_t) ob * 4 + i) * IN_N;
                std::memcpy(o, kq, sizeof(kq));
                std::memcpy(o + (size_t) 2 * C * DK, vv[half], sizeof(vv[half]));
            }
        }
    }
    // the new conv state: each channel's last KW - 1 inputs, in the state
    // copy's element order (it may be the old state's memory)
    std::vector<float> ns((size_t) (KW - 1) * CH);
    for (int64_t ch = 0; ch < CH; ch++) {
        for (int64_t k = 0; k < KW - 1; k++) {
            ns[(size_t) ch * (KW - 1) + k] = xin(n_tok - (KW - 1) + k, ch);
        }
    }
    std::memcpy(in.state_out->data, ns.data(), ns.size() * sizeof(float));
    g_gm.prep = node;
    return true;
}

#pragma GCC pop_options

// ---------------------------------------------------------------------------
// The conv input on the array (kernels/gdn_conv.py, kernels/gdn-conv.cc).
// Arguments: 0 the projection's rows, [KW - 1 + T (padded)][channels] f32
// (row r is token r - KW + 1: the state's rows first), 1 the headers
// [16 heads][2 cores][CV_IN] f32, 2 this runner's X (the GDN input), 3 a
// sink. A column's two heads' rows come in on its shim's two MM2S channels,
// its four cores' output leaves on S2MM channel 0.
// ---------------------------------------------------------------------------

namespace {

constexpr const char * CV_STEM = "gdn_conv_c8";
constexpr int          CV_KW   = 4;
constexpr int          CV_ROWS = C + CV_KW - 1;
constexpr int          CV_NCH  = 3 * S;                // a head's q, k, v channels
constexpr int          CV_IN   = CV_ROWS * CV_NCH;     // f32 an input object
constexpr int          CV_OBJ  = 2 * C * DK + C * DV;  // bf16: a GDN object's K, Q, V
constexpr int          CV_PUSH = 64;                   // chunks a descriptor iterates

std::vector<uint32_t> cv_seq(int nch, int nche, int channels, long long q_off, size_t x_col) {
    xdna_seq       seq;
    const uint32_t row_w = (uint32_t) channels;  // words a row
    for (int c = 0; c < COLS; c++) {
        for (int sh = 0; sh < 2; sh++) {
            const int      head = 2 * c + sh;
            const uint32_t id0  = (uint32_t) (5 * sh);
            xdna_bd        hb   = linear_bd(2 * CV_IN);
            xdna_seq_blockwrite(&seq, c, 0, id0, &hb);
            xdna_seq_ddr_patch(&seq, c, 0, id0, 1, (uint32_t) head * 2 * CV_IN * 4);
            xdna_seq_push_queue(&seq, c, 0, id0, xdna_dma_dir::MM2S, (uint32_t) sh, false, 0);
            for (int j = 0; j * CV_PUSH < nche; j++) {
                const int      k0 = j * CV_PUSH, n = std::min(CV_PUSH, nche - k0);
                const uint32_t id = id0 + 1 + (uint32_t) j;
                xdna_bd        bd;
                bd.buf_len     = CV_IN;
                bd.d0_size     = S;
                bd.d0_stride   = 1;
                bd.d1_size     = 3;
                bd.d1_stride   = (uint32_t) (H * S);  // q, k, v are H * S channels apart
                bd.d2_stride   = row_w;
                bd.iter_size   = (uint32_t) n;
                bd.iter_stride = (uint32_t) C * row_w;
                bd.ax_cache    = 2;
                xdna_seq_blockwrite(&seq, c, 0, id, &bd);
                xdna_seq_ddr_patch(&seq, c, 0, id, 0,
                                   (uint32_t) (((size_t) k0 * C * row_w + q_off + (size_t) head * S) * 4));
                xdna_seq_push_queue(&seq, c, 0, id, xdna_dma_dir::MM2S, (uint32_t) sh, false, (uint32_t) n - 1);
            }
        }
        // out: per chunk the GDN objects i = 0 .. 3 of the column's chunk,
        // K, Q, V of each (IN_N apart); the padding chunk to the sink
        const uint32_t obj_w = CV_OBJ / 2, in_w = IN_N / 2;
        for (int j = 0; j * CV_PUSH < nch; j++) {
            const int      k0 = j * CV_PUSH, n = std::min(CV_PUSH, nch - k0);
            const uint32_t id = 10 + (uint32_t) j;
            xdna_bd        bd;
            bd.buf_len     = 4 * obj_w;
            bd.d0_size     = obj_w / 4;
            bd.d0_stride   = 1;
            bd.d1_size     = 4;
            bd.d1_stride   = obj_w / 4;
            bd.d2_stride   = in_w;
            bd.iter_size   = (uint32_t) n;
            bd.iter_stride = 4 * in_w;
            bd.ax_cache    = 2;
            xdna_seq_blockwrite(&seq, c, 0, id, &bd);
            xdna_seq_ddr_patch(&seq, c, 0, id, 2,
                               (uint32_t) (((size_t) c * x_col + (size_t) (1 + N_STATE_IN + k0) * 4 * IN_N) * 2));
            const bool last = k0 + n >= nch && nche == nch;
            if (last) {
                xdna_seq_issue_token(&seq, c, 0, xdna_dma_dir::S2MM, 0, 0xF);
            }
            xdna_seq_push_queue(&seq, c, 0, id, xdna_dma_dir::S2MM, 0, last, (uint32_t) n - 1);
        }
        if (nche > nch) {
            xdna_bd bd = linear_bd(4 * obj_w);
            xdna_seq_blockwrite(&seq, c, 0, 15, &bd);
            xdna_seq_ddr_patch(&seq, c, 0, 15, 3, 0);
            xdna_seq_issue_token(&seq, c, 0, xdna_dma_dir::S2MM, 0, 0xF);
            xdna_seq_push_queue(&seq, c, 0, 15, xdna_dma_dir::S2MM, 0, true, 0);
        }
    }
    for (int c = 0; c < COLS; c++) {
        xdna_seq_wait_token(&seq, c, 0, xdna_dma_dir::S2MM, 0);
    }
    return xdna_seq_build(&seq);
}

bool cv_ensure(xdna_kernel_pool * pool, xdna_buffer ** bo, size_t * cap, size_t need) {
    if (*cap >= need) {
        return true;
    }
    if (*bo) {
        xdna_buffer_free(*bo);
    }
    *bo  = xdna_buffer_alloc(pool->device, need);
    *cap = *bo ? need : 0;
    return *bo != nullptr;
}

}  // namespace

bool xdna_gdn_conv_supported(void) {
    static const bool present = !xdna_artifact_find(CV_STEM, false).xclbin.empty();
    return present && xdna_env_int("GGML_XDNA_GDN_CONV", 1) != 0;
}

xdna_buffer * xdna_gdn_conv_input(xdna_kernel_pool * pool, int n_tok, int channels, size_t * off) {
    std::lock_guard<std::mutex> lock(g_gm.mtx);
    // rows: the state's KW - 1, the tokens rounded up to the prefill GEMM's
    // 128 and to an even count of chunks
    const size_t                rows = CV_KW - 1 + (size_t) ((n_tok + 127) / 128 * 128) + (size_t) 2 * C;
    if (!cv_ensure(pool, &g_gm.cv_in, &g_gm.cv_in_cap, rows * channels * sizeof(float))) {
        return nullptr;
    }
    *off = (size_t) (CV_KW - 1) * channels * sizeof(float);
    return g_gm.cv_in;
}

bool xdna_gdn_mm_prepare_npu(xdna_kernel_pool * pool, const ggml_tensor * node, const xdna_gdn_conv_in & in) {
    std::lock_guard<std::mutex> lock(g_gm.mtx);
    g_gm.prep                    = nullptr;
    const ggml_tensor * st       = in.state;
    const ggml_tensor * w        = in.w;
    const int           n_tok    = (int) in.x->ne[0];
    const int           channels = (int) w->ne[1];
    const int           nch      = (n_tok + C - 1) / C;
    const int           nche     = (nch + 1) / 2 * 2;
    const int           nx       = 1 + N_STATE_IN + nch;
    const size_t        x_col    = (size_t) nx * 4 * IN_N;
    if (w->ne[0] != CV_KW || !g_gm.cv_in || nche > 4 * CV_PUSH || in.eps_q != in.eps_k ||
        in.k_off != in.q_off + (int64_t) H * S || in.v_off != in.q_off + (int64_t) 2 * H * S) {
        GGML_LOG_ERROR("%s: the conv preparation does not match the geometry (%d tokens, %d channels)\n", "xdna-gdn-mm",
                       n_tok, channels);
        return false;
    }
    if (!cv_ensure(pool, &g_gm.prep_bo, &g_gm.prep_cap, COLS * x_col * 2)) {
        return false;
    }
    if (!g_gm.cv_hdr) {
        g_gm.cv_hdr = xdna_buffer_alloc(pool->device, (size_t) H * 2 * CV_IN * sizeof(float));
        if (!g_gm.cv_hdr) {
            return false;
        }
    }
    if (!g_gm.prep_bo->data || !g_gm.cv_hdr->data || !g_gm.cv_in->data) {
        return false;
    }
    if (!g_gm.cv_kern) {
        g_gm.cv_kern = xdna_kernel_find(pool->device, CV_STEM);
        if (!g_gm.cv_kern) {
            return false;
        }
    }
    // the state's rows before the tokens, and zero rows past them
    float * xin = (float *) g_gm.cv_in->data;
    for (int r = 0; r < CV_KW - 1; r++) {
        for (int ch = 0; ch < channels; ch++) {
            xin[(size_t) r * channels + ch] =
                *(const float *) ((const char *) st->data + r * st->nb[0] + ch * st->nb[1]);
        }
    }
    if (!xdna_buffer_sync_to_device_range(g_gm.cv_in, (size_t) (CV_KW - 1) * channels * sizeof(float), 0)) {
        return false;
    }
    // the headers: chunks, first chunk, tokens, eps^2 bits, then the taps
    float *     hd   = (float *) g_gm.cv_hdr->data;
    const float eps2 = in.eps_q * in.eps_q;
    for (int h = 0; h < H; h++) {
        for (int p = 0; p < 2; p++) {
            float *   o  = hd + ((size_t) h * 2 + p) * CV_IN;
            int32_t * wd = (int32_t *) o;
            wd[0]        = nche / 2;
            wd[1]        = p;
            wd[2]        = n_tok;
            std::memcpy(&wd[3], &eps2, 4);
            for (int i = 0; i < CV_KW; i++) {
                for (int part = 0; part < 3; part++) {
                    for (int d = 0; d < S; d++) {
                        const int64_t ch = in.q_off + (int64_t) part * H * S + (int64_t) h * S + d;
                        o[64 + i * CV_NCH + part * S + d] =
                            *(const float *) ((const char *) w->data + i * w->nb[0] + ch * w->nb[1]);
                    }
                }
            }
        }
    }
    if (!xdna_buffer_sync_to_device_range(g_gm.cv_hdr, (size_t) H * 2 * CV_IN * sizeof(float), 0)) {
        return false;
    }
    // no dirty line of X may land over what the array writes
    if (!xdna_buffer_sync_to_device_range(g_gm.prep_bo, COLS * x_col * 2, 0)) {
        return false;
    }
    const std::vector<uint32_t> insts = cv_seq(nch, nche, channels, in.q_off, x_col);
    if (!xdna_kernel_bind_insts(pool->device, g_gm.cv_kern, insts.data(), insts.size())) {
        return false;  // xdna-runtime names the stream and the reason
    }
    xdna_buffer * sink = xdna_kernel_pool_acquire_buffer(pool, (size_t) 4 * CV_OBJ * 2);
    if (!sink) {
        return false;
    }
    xdna_buffer * args[4] = { g_gm.cv_in, g_gm.cv_hdr, g_gm.prep_bo, sink };
    xrt::run      run     = xdna_kernel_run_start(g_gm.cv_kern, args, 4);
    const bool    ok      = xdna_run_wait(run);
    xdna_kernel_pool_release_buffer(pool, sink);
    if (!ok) {
        return false;
    }
    // the new conv state: each channel's last KW - 1 inputs, rows n_tok ..
    // n_tok + KW - 2 of the input (its row r is token r - KW + 1)
    if (!xdna_buffer_sync_from_device_range(g_gm.cv_in, (size_t) (CV_KW - 1) * channels * sizeof(float),
                                            (size_t) n_tok * channels * sizeof(float))) {
        return false;
    }
    std::vector<float> ns((size_t) (CV_KW - 1) * channels);
    for (int ch = 0; ch < channels; ch++) {
        for (int k = 0; k < CV_KW - 1; k++) {
            ns[(size_t) ch * (CV_KW - 1) + k] = xin[(size_t) (n_tok + k) * channels + ch];
        }
    }
    std::memcpy(in.state_out->data, ns.data(), ns.size() * sizeof(float));
    g_gm.prep = node;
    return true;
}
