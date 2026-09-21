#include "xdna-conv-prefill.h"
#include "ggml-impl.h"
#include "xdna-util.h"
#include "xdna-runtime.h"
#include "xdna-types.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// Runner for the token-major causal conv1d kernel (kernels/conv.py). Host ABI:
// arg 0 xw = per (col, mb, row) tile_in bf16: [kx x kc] of x (the conv input
// token window) then [KW x kc] of w (weights); arg 1 out = per (col, mb, row)
// tile_out bf16 = [kt x kc]. One run streams MB tiles (channel/token group
// combinations). The DMA schedule is the compiled .insts.bin (fixed geometry).
// The graph feeds sx = conv_input (the CONCAT of conv history + projection,
// [3+M, 6144] token-major f32) and w = ssm_conv1d.weight ([4, 6144]); dst is
// [6144, M] with channels contiguous per token.

namespace conv_pf {

constexpr int  C      = 16;    // channels per core
constexpr int  T      = 256;   // tokens per tile
constexpr int  KW     = 4;     // conv kernel width
constexpr int  COLS   = 8;
constexpr int  ROWS   = 4;     // cores per column
constexpr int  MB     = 8;     // tiles per run
constexpr int  KC     = ROWS * C;          // 64 channels per column
constexpr int  KT     = T / ROWS;          // 64 tokens per core
constexpr int  KX     = KT + KW - 1;       // 67-token x window per core
constexpr int  C_TOT  = COLS * ROWS * C;   // 512 channels per tile
constexpr int  TILE_IN  = KC * (KX + KW);  // 64 x 71
constexpr int  TILE_OUT = KT * KC;         // 64 x 64

constexpr const char * STEM = "conv_prefill_bf16_c16_t256_kw4_mb8";

// bf16 BOs: the kernel accumulates in f32 and only the operands and the stored
// result are narrowed, which halves the pack write, the dispatch read and write
// and the scatter read.
constexpr size_t ELT = sizeof(ggml_bf16_t);

static size_t xw_bytes()  { return (size_t) COLS * MB * ROWS * TILE_IN  * ELT; }
static size_t out_bytes() { return (size_t) COLS * MB * ROWS * TILE_OUT * ELT; }

// ggml's fp32_to_bf16_row is scalar unless the translation unit is compiled
// with AVX512-BF16, which this backend is not (-O3 only), and pack/scatter move
// 0.45 GB a pass each. The integer form is round-to-nearest-even on the top 16
// bits, and written this way the compiler vectorises it.
static inline uint16_t f32_to_bf16_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    return (uint16_t) ((x + 0x7fffu + ((x >> 16) & 1u)) >> 16);
}

static inline float bf16_bits_to_f32(uint16_t b) {
    const uint32_t x = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &x, sizeof(f));
    return f;
}

} // namespace conv_pf

struct conv_runner {
    std::mutex     mtx;
    xdna_device *  dev  = nullptr;
    xdna_kernel *  kern = nullptr;
    xdna_buffer *  xw   = nullptr;
    xdna_buffer *  out  = nullptr;
};

static conv_runner g_conv;

// CONCAT nodes the graph pass cleared for direct reads (see the header).
struct conv_direct { const float * hist; int64_t rows; };
static std::unordered_map<const ggml_tensor *, conv_direct> g_direct;

void xdna_conv_prefill_direct_reset(void) { g_direct.clear(); }

void xdna_conv_prefill_direct_add(const struct ggml_tensor * concat,
                                  const float * hist, int64_t rows) {
    if (concat && hist && rows > 0) {
        g_direct[concat] = { hist, rows };
    }
}

static bool conv_pf_enabled(void) {
    // On by default (GGML_XDNA_CONV=0 falls back to the host conv). The per-op
    // conv still round-trips its input and output through DDR, so it costs
    // ~10% of prefill throughput against the host, but the op belongs on the
    // array, and the bf16 BOs plus the vectorised pack/scatter keep that gap
    // small.
    if (xdna_env_int("GGML_XDNA_CONV", 1) == 0) {
        return false;
    }
    for (const auto & dir : xdna_kernel_search_dirs()) {
        std::error_code ec;
        if (std::filesystem::exists(dir / (std::string(conv_pf::STEM) + ".xclbin"), ec) &&
            std::filesystem::exists(dir / (std::string(conv_pf::STEM) + ".insts.bin"), ec)) {
            return true;
        }
    }
    return false;
}

bool xdna_conv_prefill_supported(const struct ggml_tensor * node) {
    using namespace conv_pf;
    if (!conv_pf_enabled() || !node || node->op != GGML_OP_SSM_CONV) {
        return false;
    }
    // Decode (single-token) SSM_CONV stays on the CPU: a one-token conv tile
    // round-trips for nothing, and supports_op is also queried on CPU-only
    // runs. Fire only for prefill-sized batches (>= 64 tokens).
    if (node->ne[1] < 64) {
        return false;
    }
    const ggml_tensor * sx = node->src[0];
    const ggml_tensor * w  = node->src[1];
    if (!sx || !w) {
        return false;
    }
    if (sx->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32) {
        return false;
    }
    if (w->ne[0] != KW) {
        return false;
    }
    if (w->ne[2] != 1 || w->ne[3] != 1 || sx->ne[2] != 1) {
        return false;   // single sequence
    }
    if (sx->ne[1] != w->ne[1] || w->ne[1] <= 0) {
        return false;
    }
    if (!ggml_is_contiguous(node) || !ggml_is_contiguous(sx) || !ggml_is_contiguous(w)) {
        return false;
    }
    return true;
}

// Load the kernel once: load the xclbin, bind the compiled .insts.bin, allocate
// the xw/out BOs. The kernel geometry is fixed, so the stream is fixed.
static bool conv_load(xdna_device * dev) {
    using namespace conv_pf;
    std::lock_guard<std::mutex> lock(g_conv.mtx);
    if (g_conv.kern) {
        return true;
    }
    std::string xclbin, insts;
    for (const auto & dir : xdna_kernel_search_dirs()) {
        std::error_code ec;
        const std::string fx = (dir / (std::string(STEM) + ".xclbin")).string();
        const std::string fi = (dir / (std::string(STEM) + ".insts.bin")).string();
        if (std::filesystem::exists(fx, ec) && std::filesystem::exists(fi, ec)) {
            xclbin = fx;
            insts  = fi;
            break;
        }
    }
    if (xclbin.empty()) {
        return false;
    }
    std::vector<uint8_t> words;
    {
        std::ifstream f(insts, std::ios::binary);
        if (!f) {
            return false;
        }
        f.seekg(0, std::ios::end);
        const std::streamoff n = f.tellg();
        f.seekg(0, std::ios::beg);
        words.resize((size_t) n);
        if (n > 0 && !f.read((char *) words.data(), n)) {
            return false;
        }
    }
    if (words.empty() || words.size() % sizeof(uint32_t) != 0) {
        return false;
    }
    g_conv.dev = dev;
    g_conv.kern = xdna_kernel_load_hw(dev, xclbin.c_str());
    if (!g_conv.kern ||
        !xdna_kernel_bind_insts(dev, g_conv.kern,
                                (const uint32_t *) words.data(),
                                words.size() / sizeof(uint32_t))) {
        xdna_kernel_free(g_conv.kern);
        g_conv.kern = nullptr;
        return false;
    }
    g_conv.xw  = xdna_buffer_alloc(dev, xw_bytes());
    g_conv.out = xdna_buffer_alloc(dev, out_bytes());
    if (!g_conv.xw || !g_conv.out) {
        xdna_kernel_free(g_conv.kern);
        g_conv.kern = nullptr;
        return false;
    }
    return true;
}

bool xdna_conv_prefill_run(struct xdna_device * dev, struct ggml_tensor * node) {
    using namespace conv_pf;
    if (!xdna_conv_prefill_supported(node)) {
        return false;
    }
    if (!conv_load(dev)) {
        return false;
    }

    const ggml_tensor * sx = node->src[0];
    const ggml_tensor * w  = node->src[1];

    // Cleared by the graph pass: read the concat's history and projection
    // directly, which is both cheaper and the only correct source when the
    // pass left the concat unmaterialised apart from its tail.
    const ggml_tensor * cb = nullptr;
    const float *       hist = nullptr;
    int64_t             hist_rows = 0;
    if (sx->op == GGML_OP_CONCAT) {
        const auto it = g_direct.find(sx);
        if (it != g_direct.end()) {
            cb        = sx->src[1];
            hist      = it->second.hist;
            hist_rows = it->second.rows;
        }
    }

    const int64_t ncs = sx->ne[0];          // conv input rows (3 + n_t)
    const int64_t nr  = w->ne[1];           // channels (6144)
    const int64_t n_t = node->ne[1];        // output tokens
    const int64_t n_cg = (nr + C_TOT - 1) / C_TOT;
    const int64_t n_tg = (n_t + T - 1) / T;

    // Tile list: every (token group, channel group) is one independent tile.
    struct tile { int64_t t0; int64_t c0; };
    std::vector<tile> tiles;
    tiles.reserve((size_t) (n_tg * n_cg));
    for (int64_t tg = 0; tg < n_tg; tg++) {
        for (int64_t cg = 0; cg < n_cg; cg++) {
            tiles.push_back({ tg * T, cg * C_TOT });
        }
    }

    const int64_t rows_per_col = ROWS;
    const int64_t n_cores = COLS * ROWS;
    const size_t n_submits = (tiles.size() + MB - 1) / MB;

    for (size_t sub = 0; sub < n_submits; sub++) {
        const size_t base = sub * MB;
        {
        ggml_bf16_t * xw_h = (ggml_bf16_t *) g_conv.xw->bo.map();

        for (int mb = 0; mb < MB; mb++) {
            const size_t ti_idx = base + (size_t) mb;
            if (ti_idx >= tiles.size()) {
                break;
            }
            const tile & tl = tiles[ti_idx];
            for (int64_t k = 0; k < n_cores; k++) {
                const int64_t col = k / rows_per_col;
                const int64_t row = k % rows_per_col;
                const size_t slot_k = (size_t) col * MB * rows_per_col +
                                      (size_t) mb * rows_per_col + (size_t) row;
                ggml_bf16_t * pk = xw_h + slot_k * TILE_IN;
                const int64_t ch0 = tl.c0 + col * KC;
                const int64_t c_lim = std::min<int64_t>(KC, nr - ch0);
                if (c_lim <= 0) {
                    continue;
                }
                // x window: rows [row*KT, row*KT + KX) of the column's channels.
                // conv_input is [ncs, nr] with ne0 (tokens) contiguous, so a
                // channel row is strided by nb[1]; gather channels contiguous.
                for (int64_t r = 0; r < KX; r++) {
                    const int64_t ti = tl.t0 + row * KT + r;
                    ggml_bf16_t * xdst = pk + r * KC;
                    if (ti < 0 || ti >= ncs) {
                        // Zero just this padded row. The array reads the whole
                        // BO, so the row still has to be defined - but zeroing
                        // the entire buffer per submit was 0.5 GB of pure
                        // overhead over a 1024-token prefill.
                        std::memset(xdst, 0, (size_t) KC * ELT);
                        continue;
                    }
                    // Reading the concat's sources rather than the concat is
                    // not just about skipping its materialisation: the concat
                    // is token-major, so a channel run in it is strided by
                    // ncs floats and this gather goes one element per cache
                    // line. `qkv_mixed` behind the transpose is
                    // channel-contiguous, so the same run is one memcpy.
                    if (hist && ti < hist_rows) {
                        const float * hs = hist + ti * nr + ch0;
                        uint16_t *    hd = (uint16_t *) xdst;
                        for (int64_t c = 0; c < c_lim; c++) {
                            hd[c] = f32_to_bf16_bits(hs[c]);
                        }
                        continue;
                    }
                    const ggml_tensor * s = hist ? cb : sx;
                    const int64_t si = hist ? ti - hist_rows : ti;
                    const char * srow = (const char *) s->data + si * s->nb[0];
                    if (s->nb[1] == sizeof(float)) {
                        const float * ss = (const float *) (srow + ch0 * s->nb[1]);
                        uint16_t *    sd = (uint16_t *) xdst;
                        for (int64_t c = 0; c < c_lim; c++) {
                            sd[c] = f32_to_bf16_bits(ss[c]);
                        }
                    } else {
                        for (int64_t c = 0; c < c_lim; c++) {
                            xdst[c].bits = f32_to_bf16_bits(
                                ((const float *) (srow + (ch0 + c) * s->nb[1]))[0]);
                        }
                    }
                }
                // The channel tail past nr is a pad the array also reads.
                if (c_lim < KC) {
                    for (int64_t r = 0; r < KX; r++) {
                        std::memset(pk + r * KC + c_lim, 0, (size_t) (KC - c_lim) * ELT);
                    }
                }
                // weights [KW][KC]: ggml layout w[tap][ch] at tap + ch*KW
                ggml_bf16_t * wdst = pk + KX * KC;
                for (int64_t c = 0; c < c_lim; c++) {
                    const float * wp = (const float *) w->data + (int64_t) (ch0 + c) * KW;
                    for (int64_t i = 0; i < KW; i++) {
                        wdst[i * KC + c].bits = f32_to_bf16_bits(wp[i]);
                    }
                }
                if (c_lim < KC) {
                    for (int64_t i = 0; i < KW; i++) {
                        std::memset(wdst + i * KC + c_lim, 0, (size_t) (KC - c_lim) * ELT);
                    }
                }
            }
        }

        // xw_h was packed directly into the BO map; make it NPU-visible.
        xdna_buffer_sync_to_device(g_conv.xw);
        }
        {
            xdna_buffer * args[2] = { g_conv.xw, g_conv.out };
            xrt::run run = xdna_kernel_run_start(g_conv.kern, args, 2);
            if (!xdna_run_wait(run)) {
                return false;
            }
            xdna_buffer_sync_from_device(g_conv.out);
        }

        // Scatter the real tiles of this batch into dst (token-major rows).
        const ggml_bf16_t * o = (const ggml_bf16_t *) g_conv.out->bo.map();
        for (int mb = 0; mb < MB; mb++) {
            const size_t ti_idx = base + (size_t) mb;
            if (ti_idx >= tiles.size()) {
                break;
            }
            const tile & tl = tiles[ti_idx];
            for (int64_t k = 0; k < n_cores; k++) {
                const int64_t col = k / rows_per_col;
                const int64_t row = k % rows_per_col;
                const size_t slot_k = (size_t) col * MB * rows_per_col +
                                      (size_t) mb * rows_per_col + (size_t) row;
                const ggml_bf16_t * ok = o + slot_k * TILE_OUT;
                const int64_t ch0 = tl.c0 + col * KC;
                const int64_t c_lim = std::min<int64_t>(KC, nr - ch0);
                if (c_lim <= 0) {
                    continue;
                }
                for (int64_t t = 0; t < KT; t++) {
                    const int64_t ti = tl.t0 + row * KT + t;
                    if (ti < 0 || ti >= n_t) {
                        continue;
                    }
                    float * drow = (float *) ((char *) node->data + ti * node->nb[1]);
                    const uint16_t * sb = (const uint16_t *) (ok + t * KC);
                    for (int64_t c = 0; c < c_lim; c++) {
                        drow[ch0 + c] = bf16_bits_to_f32(sb[c]);
                    }
                }
            }
        }
    }

    return true;
}
