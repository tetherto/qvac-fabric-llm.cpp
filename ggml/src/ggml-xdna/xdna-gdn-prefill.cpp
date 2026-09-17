#include "xdna-gdn-prefill.h"
#include "ggml-impl.h"
#include "xdna-runtime.h"
#include "xdna-seq.h"
#include "xdna-types.h"
#include "xdna-profile.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

// Runner for the bf16 GDN prefill kernel (kernels/gdn-prefill.py). Host ABI of
// the design: three host BOs
//   tok    : [H][CS][3*S+2] bf16   q | k | v | eg | beta
//   packed : [H][NS][ROWS*S + CS*ROWS] bf16. A strip is a slice of the state's
//            value axis (rows ns*ROWS .. +ROWS); its state prefix is held
//            transposed, [S][ROWS], key index major. attn suffix [CS][ROWS].
// One device run advances one CS=64-token chunk of all 16 heads. The state
// lives on the device across chunks: the packed buffer is passed as both the
// SIN and the SOUT argument (the design's state strip is read into L1 before
// the chunk's tokens and rewritten after, so an in-place buffer is safe). The
// DMA schedule is the host-built TXN stream of xdna_gdn_prefill_seq_build, so
// the xclbin's .insts.bin is never read at runtime. A partial tail chunk
// (M % 64) is finished on the host in f32 from the bf16 chunk-boundary state.

namespace gdn_pf {

constexpr int  S      = 128;   // state/head dim (DH in the kernel)
constexpr int  H      = 16;    // value heads (== query/key heads for Qwen3.5)
constexpr int  CS     = 64;    // tokens per kernel chunk
constexpr int  ROWS   = 64;    // state rows per strip
constexpr int  NS     = S / ROWS;   // strips per head
constexpr int  TOK_N  = 3 * S + 2;  // bf16 per (head, token) row
constexpr int  PACKED_N = ROWS * S + CS * ROWS;  // bf16 per (head, strip)

constexpr int  STATEPF_ELEMS = ROWS * S;   // packed state prefix per strip
constexpr int  ATTN_ELEMS    = CS * ROWS;  // attn suffix per strip

constexpr const char * STEM = "gdn_prefill_bf16_S128_H16_CS64_c8";

static size_t tok_bytes()    { return (size_t) H * CS * TOK_N  * sizeof(uint16_t); }
static size_t packed_bytes() { return (size_t) H * NS * PACKED_N * sizeof(uint16_t); }

} // namespace gdn_pf

// Shared device state: one kernel + the tok buffer + one in-place packed
// buffer, created on first use and kept for the process lifetime.
struct gdn_pf_runner {
    std::mutex     mtx;
    xdna_device *  dev  = nullptr;
    xdna_kernel *  kern = nullptr;
    xdna_buffer *  tok  = nullptr;
    xdna_buffer *  pkd  = nullptr;
    std::vector<uint8_t> scratch;   // host readback of the packed buffer
};

static gdn_pf_runner g_runner;

bool xdna_gdn_prefill_enabled(void) {
    // Opt-in (GGML_XDNA_GDN=1). The device recurrence leaves a bf16 state, so
    // decode continues from a state ~4e-2 off the f32 reference after 512
    // tokens (GGML_XDNA_GDN_VERIFY=1 prints it), and it is slower than the host
    // for this model: on the array it costs ~25% of prefill throughput (pp1024
    // 1501 -> 1097 t/s), because the kernel advances 64 tokens per dispatch and
    // every chunk pays its own bf16 packing, submit/wait and read-back. Off by
    // default for speed; set it for full-array coverage.
    const char * v = getenv("GGML_XDNA_GDN");
    if (v == nullptr || v[0] == '\0' || strcmp(v, "0") == 0) {
        return false;
    }
    for (const auto & dir : xdna_kernel_search_dirs()) {
        std::error_code ec;
        if (std::filesystem::exists(dir / (std::string(gdn_pf::STEM) + ".xclbin"), ec)) {
            return true;
        }
    }
    return false;
}

bool xdna_gdn_prefill_supported(const struct ggml_tensor * node) {
    using namespace gdn_pf;
    if (!xdna_gdn_prefill_enabled() || !node || node->op != GGML_OP_GATED_DELTA_NET) {
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
    if (v->ne[2] <= 1) {
        // Single-token decode: keep off by default. GGML_XDNA_GDN_AR=1 also
        // accepts the 1-token shape (the llama fused-op autoregressive probe)
        // so the fused GDN paths can be enabled end to end.
        static const bool allow_ar = []() {
            const char * e = getenv("GGML_XDNA_GDN_AR");
            return e != nullptr && atoi(e) >= 1;
        }();
        if (!allow_ar) {
            return false;
        }
    }
    // Accept any batch: full CS=64-token chunks run on the device and any tail
    // is finished exactly on the host. Accepting small batches also lets the
    // llama fused-op probe (16 tokens) enable the fused chunked GDN path.
    // Baked geometry and op shape.
    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F32 ||
        v->type != GGML_TYPE_F32 || g->type != GGML_TYPE_F32 ||
        b->type != GGML_TYPE_F32 || s->type != GGML_TYPE_F32) {
        return false;
    }
    if (q->ne[0] != S || q->ne[1] != H || k->ne[0] != S || k->ne[1] != H ||
        v->ne[0] != S || v->ne[1] != H) {
        return false;
    }
    if (g->ne[0] != 1 || b->ne[0] != 1) {
        return false;   // per-head scalar gate only (no KDA vectors)
    }
    if (s->ne[0] != S || s->ne[1] != S || s->ne[2] != H) {
        return false;
    }
    if (v->ne[3] != 1 || s->ne[3] != 1 || ggml_get_op_params_i32(node, 0) != 1) {
        return false;   // single sequence, K=1 snapshot
    }
    if (!ggml_is_contiguous(q) || !ggml_is_contiguous(k) ||
        !ggml_is_contiguous(g) || !ggml_is_contiguous(b) || !ggml_is_contiguous(s)) {
        return false;   // v may be a row-contiguous view (conv output)
    }
    return true;
}

// fp32 -> bf16 (round-to-nearest-even). Same conversion the fused decode path
// uses to seed its bf16 state.
static uint16_t f32_to_bf16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint16_t lsb = (uint16_t) ((x >> 16) & 1u);
    const uint32_t rem = x & 0xFFFFu;
    uint32_t r = (x >> 16) & 0xFFFFu;
    if (rem > 0x8000u || (rem == 0x8000u && lsb)) {
        r += 1;
    }
    return (uint16_t) r;
}

static float bf16_to_f32(uint16_t v) {
    const uint32_t x = (uint32_t) v << 16;
    float f;
    std::memcpy(&f, &x, sizeof(f));
    return f;
}

// Generic element read of a tensor by its byte strides (matches the ggml_cpu
// gated-delta-net reference which reads through the tensor strides).
static float tel(const ggml_tensor * t, int64_t i, int64_t j, int64_t k, int64_t l) {
    const char * p = (const char *) t->data;
    p += i * t->nb[0] + j * t->nb[1] + k * t->nb[2] + l * t->nb[3];
    float f;
    std::memcpy(&f, p, sizeof(f));
    return f;
}

// Seed the packed state buffer from the llama cache state (f32, ggml layout:
// per head, cache[j*S + i] is the state cell whose value index is j and whose
// key index is i - see cpu_tail, which reads k over i and v/out over j).
//
// A kernel strip is a slice of the *value* axis: strip ns owns state rows
// j in [ns*ROWS, (ns+1)*ROWS), which is why the design passes j0 = ns*ROWS and
// why scatter_attn lands the strip's attn at output offset ns*ROWS. Inside the
// strip the kernel holds the slice transposed, [S][ROWS] with the key index i
// major and the row-within-strip l on the vector lanes (state + i*ROWS):
//   packed[(h*NS + ns)*PACKED_N + i*ROWS + l] = cache[(ns*ROWS + l)*S + i].
static void seed_state(uint16_t * pkd, const ggml_tensor * sstate) {
    using namespace gdn_pf;
    std::memset(pkd, 0, packed_bytes());
    const float * src = (const float *) sstate->data;   // contiguous f32
    for (int h = 0; h < H; h++) {
        const float * hst = src + (size_t) h * S * S;
        for (int ns = 0; ns < NS; ns++) {
            uint16_t * dst = pkd + ((size_t) h * NS + ns) * PACKED_N;
            for (int l = 0; l < ROWS; l++) {
                const float * row = hst + (size_t) (ns * ROWS + l) * S;
                for (int i = 0; i < S; i++) {
                    dst[(size_t) i * ROWS + l] = f32_to_bf16(row[i]);
                }
            }
        }
    }
}

// Unpack the final bf16 packed state (after all full chunks) back into the
// ggml result's f32 state region - the exact inverse of seed_state.
static void unpack_state(uint16_t * pkd, float * dst_state) {
    using namespace gdn_pf;
    for (int h = 0; h < H; h++) {
        float * hst = dst_state + (size_t) h * S * S;
        for (int ns = 0; ns < NS; ns++) {
            const uint16_t * src = pkd + ((size_t) h * NS + ns) * PACKED_N;
            for (int l = 0; l < ROWS; l++) {
                float * row = hst + (size_t) (ns * ROWS + l) * S;
                for (int i = 0; i < S; i++) {
                    row[i] = bf16_to_f32(src[(size_t) i * ROWS + l]);
                }
            }
        }
    }
}

// Convert one contiguous run of f32 to bf16, round to nearest even. Written
// as plain integer arithmetic on a contiguous run so the compiler can
// vectorise it: the packer below moves 228M values over a prompt, and going
// through a per-element helper that recomputes a strided address and memcpys
// four bytes cost 744 ms of the 6.8 s prompt - as long as the array spent
// computing them (GGML_XDNA_RUNNER_PROF).
static void bf16_run(uint16_t * dst, const float * src, int n) {
    for (int i = 0; i < n; i++) {
        uint32_t x;
        std::memcpy(&x, &src[i], sizeof(x));
        dst[i] = (uint16_t) ((x + 0x7fffu + ((x >> 16) & 1u)) >> 16);
    }
}

// One row of a [S][H][M] tensor - S values for head h at token gt - into dst.
// The fast path is the one the graph actually produces, S contiguous.
static void pack_row(uint16_t * dst, const ggml_tensor * t, int h, int64_t gt) {
    using namespace gdn_pf;
    const char * base = (const char *) t->data + (size_t) h * t->nb[1] +
                        (size_t) gt * t->nb[2];
    if (t->nb[0] == sizeof(float)) {
        bf16_run(dst, (const float *) base, S);
        return;
    }
    for (int i = 0; i < S; i++) {
        float f;
        std::memcpy(&f, base + (size_t) i * t->nb[0], sizeof(f));
        dst[i] = f32_to_bf16(f);
    }
}

// Pack one CS=64-token chunk (token base `t0`) into the bf16 tok buffer.
static void pack_chunk(uint16_t * tok, const ggml_tensor * q, const ggml_tensor * k,
                       const ggml_tensor * v, const ggml_tensor * g,
                       const ggml_tensor * beta, int t0) {
    using namespace gdn_pf;
    for (int h = 0; h < H; h++) {
        for (int t = 0; t < CS; t++) {
            const int64_t gt = (int64_t) t0 + t;
            uint16_t * row = tok + ((size_t) h * CS + t) * TOK_N;
            pack_row(row,         q, h, gt);
            pack_row(row + S,     k, h, gt);
            pack_row(row + 2 * S, v, h, gt);
            const float eg = std::exp(tel(g, 0, h, gt, 0));
            row[3 * S]     = f32_to_bf16(eg);
            row[3 * S + 1] = f32_to_bf16(tel(beta, 0, h, gt, 0));
        }
    }
}

// Scatter the attn suffix of one chunk from the packed readback into the ggml
// result (token t base `t0`, contiguous [t*H + h]*S + s floats per seq). Strip
// ns carries the value rows ns*ROWS .. +ROWS, so it lands at that offset.
static void scatter_attn(const uint16_t * pkd, float * dst, int t0) {
    using namespace gdn_pf;
    for (int h = 0; h < H; h++) {
        for (int ns = 0; ns < NS; ns++) {
            const uint16_t * src = pkd + ((size_t) h * NS + ns) * PACKED_N
                                   + STATEPF_ELEMS;
            for (int t = 0; t < CS; t++) {
                float * out = dst + ((int64_t) (t0 + t) * H + h) * S
                              + ns * ROWS;
                for (int l = 0; l < ROWS; l++) {
                    out[l] = bf16_to_f32(src[(size_t) t * ROWS + l]);
                }
            }
        }
    }
}

// CPU f32 recurrence for the (partial) tail chunk, replicating the ggml_cpu
// K=1 scalar-gate reference exactly. `state` is the ggml-layout f32 state
// (per head block of S*S floats, [j*S + i] = state row i, column j). Token
// indices run [t0, t0 + n_tail). Reads q/k/v/g/beta through their strides and
// writes attn + final state into dst.
static void cpu_tail(const ggml_tensor * q, const ggml_tensor * k,
                     const ggml_tensor * v, const ggml_tensor * g,
                     const ggml_tensor * beta, float * state, float * dst,
                     int t0, int n_tail) {
    using namespace gdn_pf;
    const float scale = 1.0f / std::sqrt((float) S);
    std::vector<float> delta(S);
    for (int t = 0; t < n_tail; t++) {
        const int64_t gt = (int64_t) t0 + t;
        for (int h = 0; h < H; h++) {
            float * s_out = state + (size_t) h * S * S;
            const float eg   = std::exp(tel(g, 0, h, gt, 0));
            const float beta_v = tel(beta, 0, h, gt, 0);
            for (int64_t f = 0; f < S * S; f++) {
                s_out[f] *= eg;
            }
            for (int j = 0; j < S; j++) {
                float sum = 0.0f;
                const float * row = s_out + (size_t) j * S;
                for (int i = 0; i < S; i++) {
                    sum += row[i] * tel(k, i, h, gt, 0);
                }
                delta[j] = (tel(v, j, h, gt, 0) - sum) * beta_v;
            }
            for (int j = 0; j < S; j++) {
                float * row = s_out + (size_t) j * S;
                for (int i = 0; i < S; i++) {
                    row[i] += delta[j] * tel(k, i, h, gt, 0);
                }
            }
            float * out = dst + ((int64_t) gt * H + h) * S;
            for (int j = 0; j < S; j++) {
                float sum = 0.0f;
                const float * row = s_out + (size_t) j * S;
                for (int i = 0; i < S; i++) {
                    sum += row[i] * tel(q, i, h, gt, 0);
                }
                out[j] = sum * scale;
            }
        }
    }
}

// Load the kernel xclbin once, bind the host-built TXN stream (no .insts.bin
// read at runtime) and allocate the two persistent BOs.
static bool gdn_pf_load(xdna_device * dev) {
    using namespace gdn_pf;
    std::lock_guard<std::mutex> lock(g_runner.mtx);
    if (g_runner.kern) {
        return true;
    }
    std::string xclbin;
    for (const auto & dir : xdna_kernel_search_dirs()) {
        std::error_code ec;
        const std::string fx = (dir / (std::string(STEM) + ".xclbin")).string();
        if (std::filesystem::exists(fx, ec)) {
            xclbin = fx;
            break;
        }
    }
    if (xclbin.empty()) {
        fprintf(stderr, "xdna-gdn-prefill: %s.xclbin not found; "
                        "leaving GDN prefill on CPU\n", STEM);
        return false;
    }
    g_runner.dev = dev;
    g_runner.kern = xdna_kernel_load_hw(dev, xclbin.c_str());
    if (!g_runner.kern) {
        return false;
    }
    xdna_seq seq;
    const xdna_gdn_prefill_geom geom;
    if (!xdna_gdn_prefill_seq_build(&seq, &geom)) {
        fprintf(stderr, "xdna-gdn-prefill: TXN stream build failed\n");
        xdna_kernel_free(g_runner.kern);
        g_runner.kern = nullptr;
        return false;
    }
    const std::vector<uint32_t> words = xdna_seq_build(&seq);
    if (const char * path = getenv("GGML_XDNA_GDN_PF_DUMP")) {
        std::ofstream f(path, std::ios::binary);
        f.write((const char *) words.data(),
                (std::streamsize) (words.size() * 4));
    }
    if (words.empty() ||
        !xdna_kernel_bind_insts(dev, g_runner.kern, words.data(), words.size())) {
        xdna_kernel_free(g_runner.kern);
        g_runner.kern = nullptr;
        return false;
    }
    g_runner.tok = xdna_buffer_alloc(dev, tok_bytes());
    g_runner.pkd = xdna_buffer_alloc(dev, packed_bytes());
    if (!g_runner.tok || !g_runner.pkd) {
        xdna_kernel_free(g_runner.kern);
        g_runner.kern = nullptr;
        return false;
    }
    g_runner.scratch.resize(packed_bytes());
    return true;
}

bool xdna_gdn_prefill_run(struct xdna_device * dev, struct ggml_tensor * node) {
    using namespace gdn_pf;
    if (!xdna_gdn_prefill_supported(node)) {
        return false;
    }
    if (!gdn_pf_load(dev)) {
        return false;
    }

    const ggml_tensor * q = node->src[0];
    const ggml_tensor * k = node->src[1];
    const ggml_tensor * v = node->src[2];
    const ggml_tensor * g = node->src[3];
    const ggml_tensor * b = node->src[4];
    const ggml_tensor * s = node->src[5];

    const int M = (int) v->ne[2];
    const int n_full = M / CS;
    const int n_tail = M - n_full * CS;

    float * dst = (float *) node->data;
    GGML_ASSERT(dst != nullptr);
    float * dst_attn  = dst;
    float * dst_state = dst + (size_t) S * H * M;

    // Seed the device state buffer from the llama cache state.
    {
        xdna_rp _p("gdn", "seed");
        seed_state((uint16_t *) g_runner.pkd->bo.map(), s);
        xdna_buffer_sync_to_device(g_runner.pkd);
    }

    uint16_t * tok_map = (uint16_t *) g_runner.tok->bo.map();

    for (int c = 0; c < n_full; c++) {
        {
            xdna_rp _p("gdn", "pack");
            pack_chunk(tok_map, q, k, v, g, b, c * CS);
            xdna_buffer_sync_to_device(g_runner.tok);
        }

        {
            // In-place packed buffer: SIN and SOUT are the same BO.
            xdna_rp _p("gdn", "dispatch");
            xdna_buffer * args[3] = { g_runner.tok, g_runner.pkd, g_runner.pkd };
            xrt::run run = xdna_kernel_run_start(g_runner.kern, args, 3);
            if (!xdna_run_wait(run)) {
                fprintf(stderr, "xdna-gdn-prefill: chunk %d run failed\n", c);
                return false;
            }
        }

        // Refresh the host map (the state and attn suffixes interleave per
        // (head, strip) block, so a partial range sync would miss blocks) and
        // scatter this chunk's attn into the ggml result.
        xdna_rp _p("gdn", "scatter");
        g_runner.pkd->bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        const uint16_t * pkd_map = (const uint16_t *) g_runner.pkd->bo.map();
        scatter_attn(pkd_map, dst_attn, c * CS);
    }

    if (n_tail == 0) {
        xdna_rp _p("gdn", "unpack state");
        g_runner.pkd->bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        unpack_state((uint16_t *) g_runner.pkd->bo.map(), dst_state);
    } else {
        xdna_rp _p("gdn", "host tail");
        // Finish the tail on the host: unpack the chunk-boundary bf16 state to
        // f32 and run the exact CPU recurrence over the remaining tokens.
        g_runner.pkd->bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        std::vector<float> state((size_t) H * S * S);
        unpack_state((uint16_t *) g_runner.pkd->bo.map(), state.data());
        cpu_tail(q, k, v, g, b, state.data(), dst, n_full * CS, n_tail);
        std::memcpy(dst_state, state.data(), state.size() * sizeof(float));
    }

    // GGML_XDNA_GDN_VERIFY=1 recomputes the whole op with the exact recurrence
    // this file already trusts for the tail, from the same seed, and says how
    // far the array's two outputs are from it. A prefill whose state is wrong
    // does not look wrong here - it looks like a decode that answers nothing -
    // so the check has to be on the op, not on the text.
    if (getenv("GGML_XDNA_GDN_VERIFY") && n_full > 0) {
        // Only the calls that actually ran a chunk on the array: the probe's
        // two-token shapes all take the host tail and match by construction.
        static int shown = 0;
        if (shown++ < 4) {
            std::vector<float> rs((size_t) H * S * S);
            std::vector<float> rdst((size_t) S * H * M + (size_t) H * S * S);
            const float * seed = (const float *) s->data;
            std::memcpy(rs.data(), seed, rs.size() * sizeof(float));
            cpu_tail(q, k, v, g, b, rs.data(), rdst.data(), 0, M);
            const auto rel = [](const float * a, const float * r, size_t n) {
                double num = 0, den = 0;
                for (size_t i = 0; i < n; i++) {
                    const double d = (double) a[i] - r[i];
                    num += d * d;
                    den += (double) r[i] * r[i];
                }
                return den > 0 ? std::sqrt(num / den) : -1.0;
            };
            // The attn is close and the state is not, so the recurrence on
            // the array is right and what it writes out is not. Price the
            // state against the reference at several chunk counts: if the
            // drain is a chunk behind, one of them fits.
            const double r_full = rel(dst_state, rs.data(), rs.size());
            double r_lag = -1.0, r_lag2 = -1.0;
            if (n_full >= 2) {
                std::vector<float> ls((size_t) H * S * S);
                std::vector<float> ld(rdst.size());
                std::memcpy(ls.data(), seed, ls.size() * sizeof(float));
                cpu_tail(q, k, v, g, b, ls.data(), ld.data(), 0,
                         (n_full - 1) * CS);
                r_lag = rel(dst_state, ls.data(), ls.size());
                std::memcpy(ls.data(), seed, ls.size() * sizeof(float));
                cpu_tail(q, k, v, g, b, ls.data(), ld.data(), 0, CS);
                r_lag2 = rel(dst_state, ls.data(), ls.size());
            }
            // Where does each of the array's values live in the reference?
            // A permutation shows up as a regular pattern in the answer; noise
            // shows up as no match at all.
            if (getenv("GGML_XDNA_GDN_PERM")) {
                fprintf(stderr, "xdna-gdn-perm: head 0, dst index -> nearest "
                                "reference index (S=%d):\n", S);
                for (int i = 0; i < 8; i++) {
                    const float want = dst_state[i];
                    int    best = -1;
                    double bd   = 1e30;
                    for (int j = 0; j < S * S; j++) {
                        const double d = std::fabs((double) rs[j] - want);
                        if (d < bd) { bd = d; best = j; }
                    }
                    fprintf(stderr, "   %3d (%g) -> %5d (%g) row %d col %d, "
                            "err %.2e\n", i, want, best, rs[best],
                            best % S, best / S, bd);
                }
            }
            // Split the attn: the first n_full*CS tokens are the array's
            // own, the rest is the host tail - which is exact given the state
            // it starts from, so its error is the state's error.
            const size_t per_tok = (size_t) S * H;
            const double a_dev = rel(dst_attn, rdst.data(),
                                     per_tok * (size_t) (n_full * CS));
            const double a_one = rel(dst_attn, rdst.data(), per_tok * CS);
            // The seed is all zeros on a fresh prompt, so every strip layout
            // is indistinguishable going in and only the readback shows which
            // one the kernel actually leaves. Price the candidates.
            if (getenv("GGML_XDNA_GDN_LAYOUT")) {
                // Against the reference after exactly the chunks the array
                // ran, not after all M - the buffer holds the chunk boundary.
                std::vector<float> cs_ref((size_t) H * S * S);
                {
                    std::vector<float> tmp(rdst.size());
                    std::memcpy(cs_ref.data(), seed, cs_ref.size() * sizeof(float));
                    cpu_tail(q, k, v, g, b, cs_ref.data(), tmp.data(), 0,
                             n_full * CS);
                }
                const uint16_t * pk = (const uint16_t *) g_runner.pkd->bo.map();
                // The four readings of the strip: which state axis the
                // strip slices (value, as the kernel's j0 and scatter_attn
                // say, or key) x which index is major inside it.
                const char * names[] = { "strip=value, i-major (current)",
                                         "strip=key,   j-major (old)",
                                         "strip=value, l-major",
                                         "strip=key,   l-major" };
                for (int cand = 0; cand < 4; cand++) {
                    std::vector<float> t((size_t) H * S * S, 0.0f);
                    for (int h = 0; h < H; h++) {
                        float * hst = t.data() + (size_t) h * S * S;
                        for (int ns = 0; ns < NS; ns++) {
                            const uint16_t * src =
                                pk + ((size_t) h * NS + ns) * PACKED_N;
                            for (int a = 0; a < S; a++) {
                                for (int l = 0; l < ROWS; l++) {
                                    const size_t idx =
                                        (cand & 2) ? (size_t) l * S + a
                                                   : (size_t) a * ROWS + l;
                                    const size_t cell =
                                        (cand & 1)
                                            ? (size_t) a * S + ns * ROWS + l
                                            : (size_t) (ns * ROWS + l) * S + a;
                                    hst[cell] = bf16_to_f32(src[idx]);
                                }
                            }
                        }
                    }
                    fprintf(stderr, "xdna-gdn-layout: %-30s rel %.3e\n",
                            names[cand], rel(t.data(), cs_ref.data(), t.size()));
                }
            }
            fprintf(stderr, "xdna-gdn-verify: M=%d full=%d tail=%d | attn rel "
                    "%.3e (array's own %.3e, first chunk %.3e) | state rel %.3e"
                    " (one chunk short %.3e, one chunk only %.3e)\n",
                    M, n_full, n_tail,
                    rel(dst_attn, rdst.data(), (size_t) S * H * M),
                    a_dev, a_one, r_full, r_lag, r_lag2);
        }
    }

    return true;
}
