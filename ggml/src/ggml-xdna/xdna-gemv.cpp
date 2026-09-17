#include "xdna_design_tag.h"
#include "xdna-gemv.h"

#include "ggml-impl.h"
#include "xdna-verify.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <fstream>
#include <map>
#include <mutex>

// --- geometry ---------------------------------------------------------------

namespace {

// Columns per lane group is the kernel's vector width and follows the core
// width; see xdna_gemv_geom::lane().

// Per-group block inside a packed tile: the codes of one group for the lane
// group's columns, then that group's scale and min for each of them. The bf16 pair the two
// scale is per super-block and lives at the end of the tile. Mirrors
// gemv_q4.cc.
int block_code_bytes(xdna_wfmt fmt, int lane) {
    return fmt == XDNA_WFMT_Q4G32 ? XDNA_Q4G32_GROUP * lane / 2 : XDNA_Q8G16_GROUP * lane;
}

int block_param_bytes(xdna_wfmt fmt, int lane) {
    GGML_UNUSED(fmt);
    // The per-group scale and min are integers, but they are carried as bf16:
    // the kernel multiplies them by the super-block pair, and an AIE2P f32
    // multiply is emulated where a bf16 one is native - int8 in the tile
    // measured 25% slower for the 6% of bytes it saved. An integer under 256
    // is exact in bf16, so nothing is lost.
    return 4 * lane;
}

// f32 -> bf16, round to nearest even.
uint16_t bf16_of(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    return (uint16_t) ((b + 0x7FFF + ((b >> 16) & 1)) >> 16);
}

int block_bytes(xdna_wfmt fmt, int lane) {
    return block_code_bytes(fmt, lane) + block_param_bytes(fmt, lane);
}

// Bytes of one lane group's super-block record: d_hi, d_lo, m_hi, m_lo for
// every column of the group.
int sup_bytes(int lane) { return 4 * lane * 2; }

} // namespace

// Super-block records a tile carries, per lane group. A q4g32 tile spans twice
// the values of the q8g16 tile it shares an artifact with, so it needs twice
// as many; both are given the larger count so the two tile sizes stay equal,
// which is what lets one design stream either.
int xdna_gemv_geom::n_sup() const {
    const int kt_q4 = fmt == XDNA_WFMT_Q4G32 ? k_tile() : 2 * k_tile();
    return std::max(1, kt_q4 / XDNA_SB_VALUES);
}

size_t xdna_gemv_geom::tile_bytes() const {
    const int lg = n_core() / lane();
    const int ng = k_tile() / group();
    return (size_t) lg * (ng * block_bytes(fmt, lane()) + n_sup() * sup_bytes(lane()));
}

bool xdna_gemv_geom::valid() const {
    if (fmt == XDNA_WFMT_NONE || K <= 0 || N <= 0 || cols <= 0 || cols > XDNA_GEMV_COLS) {
        return false;
    }
    if (K % k_tile() || k_tile() % group()) {
        return false;
    }
    // One descriptor per chunk for the activations and one for the output,
    // plus the weight descriptor, all inside the shim's 16 ids.
    return N % chunk() == 0 && n_out() >= 1 && n_out() <= xdna_gemv_max_chunks();
}

bool xdna_gemv_use_half(xdna_gemv_split want) {
    static const int forced = []() {
        const char * v = getenv("GGML_XDNA_GEMV_ROWS");
        return v ? atoi(v) : 0;
    }();
    if (forced == XDNA_GEMV_ROWS_HALF) {
        return true;
    }
    if (forced == XDNA_GEMV_ROWS) {
        return false;
    }
    return want == XDNA_GEMV_SPLIT_HALF;
}

std::string xdna_gemv_geom::stem() const {
    // Tagged with the design hash (kernels/design_tag.py) so a stale artifact
    // cannot be driven by a newer stream.
    if (fused_v) {
        return "fused_layer_" XDNA_DESIGN_TAG;
    }
    char buf[96];
    snprintf(buf, sizeof(buf), "gemv_n%d_r%d_c%d_%s", n_core_v, rows_v,
             XDNA_GEMV_COLS, XDNA_DESIGN_TAG);
    return buf;
}

std::string xdna_gemv_geom::seq_key() const {
    char buf[96];
    snprintf(buf, sizeof(buf), "%s_%s_K%d_N%d_a%d%s", stem().c_str(),
             fmt == XDNA_WFMT_Q4G32 ? "q4" : "q8", K, N, cols,
             epilogue ? "_e" : "");
    return buf;
}

xdna_gemv_geom xdna_gemv_variant(enum ggml_type type, int64_t K, int64_t N,
                                 bool epilogue, xdna_gemv_split split) {
    xdna_gemv_geom g;
    g.epilogue = epilogue;
    if (split == XDNA_GEMV_SPLIT_FUSED) {
        g.rows_v   = XDNA_GEMV_ROWS_FUSED;
        g.n_core_v = XDNA_GEMV_N_CORE_FUSED;
        g.fused_v  = true;
    } else if (xdna_gemv_use_half(split)) {
        g.rows_v   = XDNA_GEMV_ROWS_HALF;
        g.n_core_v = XDNA_GEMV_N_CORE_HALF;
    }
    // One format for every quantized type: a second artifact would mean a
    // hardware-context switch whenever consecutive ops differ, which measured
    // 2.5 ms against 0.1 ms for the work itself.
    g.fmt = xdna_wfmt_gemv_for(type);
    if (g.fmt == XDNA_WFMT_NONE || K <= 0 || N <= 0) {
        return xdna_gemv_geom{};
    }
    g.K    = (int) K;
    g.cols = XDNA_GEMV_COLS;
    const int pass = XDNA_GEMV_COLS * g.rows_v * g.n_core_v;
    g.N = (int) ((N + pass - 1) / pass) * pass;
    // N counts the weight columns; the epilogue returns half as many values.
    g.n_real = epilogue ? (int) N / 2 : (int) N;
    if (epilogue && (N % 2 || g.N != (int) N)) {
        return xdna_gemv_geom{};   // the two halves must tile the array exactly
    }
    return g.valid() ? g : xdna_gemv_geom{};
}

// --- instruction stream -----------------------------------------------------

namespace {

// Shim registers, mirroring what IRON emits for this design. The activation
// broadcast is pinned to column 0 and takes that column's MM2S channel 0, so
// column 0's weights go to channel 1 while every other column's take channel
// 0; outputs are on S2MM channel 0 everywhere. The asymmetry is easy to get
// backwards and the symptom is not obvious - the cores never see the header
// that carries the tile counts, so they do nothing at all and return whatever
// the output buffer held, quickly. Confirmed by disassembling the artifact's
// own stream and reading which descriptor each push names; check it the same
// way if the design's fifos change.
constexpr uint32_t BD_WEIGHTS = 0;
constexpr uint32_t BD_ACTS    = 1;

// Shim buffer descriptor ids are four bits. One dispatch needs the weight
// descriptor, one for the whole activation stream, and one output descriptor
// per chunk.
constexpr int MAX_BD     = 16;
constexpr int MAX_CHUNKS = MAX_BD - 2;

// A contiguous transfer. The length field carries the whole thing and the
// address-gen sizes are left at zero - a size of one would wrap the descriptor
// after a single word, which is what a stream built with sizes of one does.
void linear_bd(xdna_bd & bd, size_t bytes, uint32_t off) {
    bd.buf_len   = (uint32_t) (bytes / 4);
    bd.buf_off   = off;
    bd.d0_size   = 0;
    bd.d0_stride = 1;
    bd.d1_size   = 0;
    bd.d1_stride = 1;
    bd.d2_stride = 1;
    bd.ax_cache  = 2;
}

// `d1` runs of `d0` bytes, `stride` bytes apart. The encoding is the one IRON
// emits for a two-dimensional TensorAccessPattern: sizes in 32-bit words,
// strides in words.
void strided_bd(xdna_bd & bd, size_t d0_bytes, uint32_t d1, size_t stride_bytes,
                uint32_t off) {
    bd.buf_len   = (uint32_t) (d0_bytes / 4) * d1;
    bd.buf_off   = off;
    bd.d0_size   = (uint32_t) (d0_bytes / 4);
    bd.d0_stride = 1;
    bd.d1_size   = d1;
    bd.d1_stride = (uint32_t) (stride_bytes / 4);
    bd.d2_stride = 1;
    bd.ax_cache  = 2;
}

} // namespace

// Every output chunk replays the same K tiles, and the host copies the body
// once per chunk. GGML_XDNA_ACT_REPEAT=1 has the stream re-read the single
// copy with a zero-stride descriptor instead - the same bytes off DDR without
// the memcpy, and a device-side producer of the activation would then have
// only one body to write. It measures 29.8 t/s against 33.6: a zero stride
// re-fetches rather than reusing what it just read, and that costs far more
// than the 27 KB memcpy it saves. Off by default; kept because a device
// producer may still want the layout.
static bool act_repeat_bd(void) {
    static const int v = [] {
        const char * e = getenv("GGML_XDNA_ACT_REPEAT");
        return e ? atoi(e) : 0;
    }();
    return v != 0;
}

int xdna_gemv_max_chunks(void) {
    return MAX_CHUNKS;
}

namespace {

// Where one of the design's shim transfers meets the array.
struct shim_ep {
    uint8_t col = 0;
    uint8_t ch  = 0;
};

// The standalone artifacts get one stream per column, with the activation
// broadcast taking column 0's first channel and that column's weights moving
// to the second. The merged layer's are wherever they fit beside the GDN core,
// which is not one per column - column 0 has no output channel left at all -
// and pinning them does not place. Read back from the built artifact with
// kernels/shim_map.py; if either design's fifos change, read it again.
// GGML_XDNA_ACT_PRO_H=1: the artifact was built with ACT_PRO_H=1, so its
// prologue takes a second input and every stream has to feed it. The two have
// to agree - a stream that does not push it hangs the prologue on its fifo.
static bool act_pro_h(void) {
    static const bool on = []() {
        const char * e = getenv("GGML_XDNA_ACT_PRO_H");
        return e != nullptr && atoi(e) != 0;
    }();
    return on;
}

struct gemv_shim_map {
    shim_ep w[XDNA_GEMV_COLS];
    shim_ep a;
    // The prologue's second input, where it takes the host's half of a tile
    // from a buffer no dispatch writes (kernels/gemv_q4.py, ACT_PRO_H). Read
    // out of the artifact the same way as the rest: it is the one MM2S push
    // the design gains.
    shim_ep h;
    bool    has_h = false;
    shim_ep o[XDNA_GEMV_COLS];
    // Columns whose cores share one output stream, joined in a MemTile. The
    // merged design does this to give shim channels back to the core's stages;
    // the standalone artifacts keep one stream per column.
    int     o_group = 1;
    int n_o() const { return XDNA_GEMV_COLS / o_group; }
};

gemv_shim_map shim_map_for(const xdna_gemv_geom & geom) {
    gemv_shim_map m;
    if (geom.fused_v) {
        // shim_map.py fused_layer.insts.bin --tail 17
        //
        // Re-read this whenever either half of the merged design changes its
        // fifos, and never guess it. Consolidating the core's conv stage moved
        // all sixteen of these, and a stale table does not fail cleanly: the
        // model's output degrades and a stage the table has nothing to do with
        // appears ten times slower.
        // One tile each, which is the point of eight streams: a shim tile has
        // one AXI port for all of its channels, and the six tiles the placer
        // used before held the FFN to 19 GB/s.
        static const shim_ep W[XDNA_GEMV_COLS] = {
            {0, 1}, {1, 1}, {2, 1}, {3, 0}, {4, 1}, {5, 1}, {6, 0}, {7, 1},
        };
        // Four streams, not eight: two columns to a stream (fused_layer.py's
        // OUT_GROUP).
        static const shim_ep O[] = { {2, 1}, {3, 0}, {4, 0}, {6, 1} };
        m.o_group = 2;
        for (int c = 0; c < XDNA_GEMV_COLS; c++) {
            m.w[c] = W[c];
        }
        for (int g = 0; g < XDNA_GEMV_COLS / 2; g++) {
            m.o[g] = O[g];
        }
        m.a = {4, 0};
        // shim_map.py fused_layer.insts.bin, built with ACT_PRO_H=1: the one
        // push the design gains over the same design without it.
        m.h = {6, 1};
        m.has_h = act_pro_h();
    } else {
        for (int c = 0; c < XDNA_GEMV_COLS; c++) {
            m.w[c] = {(uint8_t) c, (uint8_t) (c == 0 ? 1 : 0)};
            m.o[c] = {(uint8_t) c, 0};
        }
        m.a = {0, 0};
    }
    return m;
}

} // namespace

// Both dispatches in one stream. The first is the epilogue pair; its output
// descriptors name the activation argument, not the output one, so its result
// lands where the second reads it and the host is not involved between them.
//
// Which tile and which block of it a core's output belongs to follows from
// where its columns sit: core c of chunk oc owns the mid values starting at
// oc*chunk/2 + c*n_core/2, and a tile of the second dispatch holds
// k_tile/(n_core/2) of those blocks.
bool xdna_gemv_seq_build_pair(xdna_seq * seq, const xdna_gemv_geom & g1,
                              const xdna_gemv_geom & g2,
                              size_t w2_off, size_t a2_off,
                              const xdna_gemv_pair_opts * opt) {
    const xdna_gemv_pair_opts po = opt ? *opt : xdna_gemv_pair_opts{};
    const int a_w = po.w_arg, a_a = po.a_arg, a_o = po.o_arg;
    if (!seq || !g1.valid() || !g2.valid() || !g1.epilogue || g2.epilogue) {
        return false;
    }
    if (g1.rows() != 1 || g1.n_core() != g2.n_core() || g1.cols != g2.cols) {
        return false;   // one core to a column, and the same core width
    }
    const gemv_shim_map map = shim_map_for(g1);   // one artifact, one map
    const int og  = map.o_group;
    const int n_o = map.n_o();
    uint32_t next_bd[XDNA_GEMV_COLS];
    for (int c = 0; c < XDNA_GEMV_COLS; c++) {
        next_bd[c] = (uint32_t) po.bd_base;
    }
    const auto take_bd = [&](int col) { return next_bd[col]++; };

    // Everything but the first dispatch's outputs takes a fixed number of
    // descriptors, so those are claimed first and the first dispatch gets what
    // is left. It is the one with many chunks, and the one that can trade
    // descriptors for waits.
    // Reserve exactly what the fills will take, not one apiece: an activation
    // fill is two descriptors whenever the chunks are replayed from one body,
    // and with the prologue's second input every fill has a twin on its
    // endpoint. Reserving less lets next_fixed run into what take_bd hands
    // out for the outputs, and a descriptor written over while its transfer
    // is in flight is lost silently.
    const auto act_bds = [](const xdna_gemv_geom & g) {
        return (act_repeat_bd() && g.n_out() > 1) ? 2 : 1;
    };
    const int n_ab = act_bds(g1) + act_bds(g2);
    for (int pass = 0; pass < 2; pass++) {
        for (int c = 0; c < g1.cols; c++) {
            take_bd(map.w[c].col);
        }
    }
    for (int k = 0; k < n_ab; k++) {
        take_bd(map.a.col);
        if (map.has_h) {
            take_bd(map.h.col);
        }
    }
    std::vector<std::vector<uint32_t>> o2_bd((size_t) n_o);
    for (int c = 0; c < n_o; c++) {
        o2_bd[(size_t) c].push_back(take_bd(map.o[c].col));
    }

    int streams[XDNA_GEMV_COLS] = {};
    for (int c = 0; c < n_o; c++) {
        streams[map.o[c].col]++;
    }
    std::vector<std::vector<uint32_t>> o1_bd((size_t) n_o);
    for (int c = 0; c < n_o; c++) {
        const int col = map.o[c].col;
        if ((int) next_bd[col] >= MAX_BD) {
            fprintf(stderr, "xdna-gemv-pair: column %d is out of descriptors "
                            "(%u of %d) with %d output streams to place\n",
                    col, next_bd[col], MAX_BD, n_o);
            return false;
        }
        o1_bd[(size_t) c].push_back(take_bd(col));
    }

    uint32_t fixed[XDNA_GEMV_COLS];
    for (int c = 0; c < XDNA_GEMV_COLS; c++) {
        fixed[c] = (uint32_t) po.bd_base;
    }
    const auto next_fixed = [&](int col) { return fixed[col]++; };

    const auto fill_weights = [&](const xdna_gemv_geom & g, size_t base) {
        const size_t span = g.col_weight_bytes();
        for (int c = 0; c < g.cols; c++) {
            const uint32_t off   = (uint32_t) (base + (size_t) c * span);
            const uint32_t bd_id = next_fixed(map.w[c].col);
            xdna_bd bd;
            linear_bd(bd, span, off);
            xdna_seq_blockwrite(seq, map.w[c].col, 0, bd_id, &bd);
            xdna_seq_ddr_patch(seq, map.w[c].col, 0, bd_id, a_w, off);
            xdna_seq_push_queue(seq, map.w[c].col, 0, bd_id, xdna_dma_dir::MM2S,
                                map.w[c].ch, false, 0);
        }
    };
    const auto fill_acts_on = [&](const xdna_gemv_geom & g, size_t base,
                                  const shim_ep & ep, int arg) {
        const size_t atb_ = g.act_tile_bytes();
        if (act_repeat_bd() && g.n_out() > 1) {
            const uint32_t h_id = next_fixed(ep.col);
            xdna_bd hbd;
            linear_bd(hbd, atb_, (uint32_t) base);
            xdna_seq_blockwrite(seq, ep.col, 0, h_id, &hbd);
            xdna_seq_ddr_patch(seq, ep.col, 0, h_id, arg, (uint32_t) base);
            xdna_seq_push_queue(seq, ep.col, 0, h_id, xdna_dma_dir::MM2S,
                                ep.ch, false, 0);
            const uint32_t b_id = next_fixed(ep.col);
            const uint32_t boff = (uint32_t) (base + atb_);
            xdna_bd bbd;
            strided_bd(bbd, (size_t) g.n_tiles() * atb_, (uint32_t) g.n_out(),
                       0, boff);
            xdna_seq_blockwrite(seq, ep.col, 0, b_id, &bbd);
            xdna_seq_ddr_patch(seq, ep.col, 0, b_id, arg, boff);
            xdna_seq_push_queue(seq, ep.col, 0, b_id, xdna_dma_dir::MM2S,
                                ep.ch, false, 0);
            return;
        }
        const uint32_t bd_id = next_fixed(ep.col);
        const uint32_t off   = (uint32_t) base;
        xdna_bd bd;
        linear_bd(bd, (size_t) (1 + g.n_out() * g.n_tiles()) * atb_, off);
        xdna_seq_blockwrite(seq, ep.col, 0, bd_id, &bd);
        xdna_seq_ddr_patch(seq, ep.col, 0, bd_id, arg, off);
        xdna_seq_push_queue(seq, ep.col, 0, bd_id, xdna_dma_dir::MM2S,
                            ep.ch, false, 0);
    };

    // Once with the tiles, once with the host's half of them, when the
    // prologue has an input of its own for it.
    const auto fill_acts = [&](const xdna_gemv_geom & g, size_t base) {
        fill_acts_on(g, base, map.a, a_a);
        if (map.has_h) {
            fill_acts_on(g, base, map.h, a_a);
        }
    };

    // ---- the epilogue pair, draining into the second dispatch's activation
    fill_weights(g1, po.w_base);
    // The first phase's tiles are the previous dispatch's drain, so its host
    // half comes from somewhere the host owns alone.
    fill_acts_on(g1, po.a_base, map.a, a_a);
    if (map.has_h) {
        fill_acts_on(g1, po.h_arg >= 0 ? po.h_base : po.a_base, map.h,
                     po.h_arg >= 0 ? po.h_arg : a_a);
    }

    const int    mid_chunk = g1.chunk() / 2;
    const int    mid_core  = g1.n_core() / 2;
    // One descriptor per output stream, so it carries og cores' blocks. They
    // are consecutive blocks of the second dispatch's activation, and a
    // stream's first core always starts a tile (og * mid_core divides the
    // second dispatch's k_tile), so the run stays linear.
    //
    // And one descriptor for the whole stream, not one a chunk: a chunk
    // advances the destination by exactly mid_chunk/k_tile activation tiles,
    // so the chunks are a fixed stride apart. A descriptor per chunk hangs the
    // array on a joined stream - see xdna_gemv_seq_build.
    const size_t obj_b     = (size_t) og * g1.n_core() * sizeof(float);
    const size_t oc_stride = (size_t) (mid_chunk / g2.k_tile()) * XDNA_GEMV_ACT_TILE;
    for (int c = 0; c < n_o; c++) {
        const int mid0 = c * og * mid_core;
        const int t    = mid0 / g2.k_tile();
        const int blk  = (mid0 % g2.k_tile()) / mid_core;
        const uint32_t off = (uint32_t) (po.a_base + a2_off +
                                         (size_t) (1 + t) * XDNA_GEMV_ACT_TILE +
                                         (size_t) blk * g1.n_core() * sizeof(float));
        const uint32_t bd_id = o1_bd[(size_t) c][0];
        xdna_bd bd;
        strided_bd(bd, obj_b, (uint32_t) g1.n_out(), oc_stride, off);
        xdna_seq_blockwrite(seq, map.o[c].col, 0, bd_id, &bd);
        xdna_seq_ddr_patch(seq, map.o[c].col, 0, bd_id, a_a, off);
        xdna_seq_issue_token(seq, map.o[c].col, 0, xdna_dma_dir::S2MM,
                             map.o[c].ch, 0xF);
        xdna_seq_push_queue(seq, map.o[c].col, 0, bd_id, xdna_dma_dir::S2MM,
                            map.o[c].ch, true, 0);
    }
    // The second dispatch's weights do not depend on the first, so they are
    // pushed before waiting for it: the shim starts streaming them into the
    // cores' fifos while the first is still draining. Its activation is the
    // handover and has to wait.
    fill_weights(g2, po.w_base + w2_off);

    // Every tile has to be in memory before the second dispatch reads it, so
    // this wait is the whole handover between the two.
    for (int c = 0; c < n_o; c++) {
        xdna_seq_wait_token(seq, map.o[c].col, 0, xdna_dma_dir::S2MM,
                            map.o[c].ch);
    }

    // ---- the projection that consumes it
    fill_acts(g2, po.a_base + a2_off);
    for (int oc = 0; oc < g2.n_out(); oc++) {
        for (int c = 0; c < n_o; c++) {
            const uint32_t off = (uint32_t) (po.o_base +
                                             ((size_t) oc * g2.chunk() +
                                              (size_t) c * og * g2.rows() * g2.n_core()) *
                                             sizeof(float));
            const uint32_t bd_id = o2_bd[(size_t) c][0];
            xdna_bd bd;
            linear_bd(bd, (size_t) og * g2.rows() * g2.n_core() * sizeof(float), off);
            xdna_seq_blockwrite(seq, map.o[c].col, 0, bd_id, &bd);
            xdna_seq_ddr_patch(seq, map.o[c].col, 0, bd_id, a_o, off);
            xdna_seq_issue_token(seq, map.o[c].col, 0, xdna_dma_dir::S2MM,
                                 map.o[c].ch, 0xF);
            xdna_seq_push_queue(seq, map.o[c].col, 0, bd_id, xdna_dma_dir::S2MM,
                                map.o[c].ch, true, 0);
        }
    }
    for (int oc = 0; oc < g2.n_out(); oc++) {
        for (int c = 0; c < n_o; c++) {
            xdna_seq_wait_token(seq, map.o[c].col, 0, xdna_dma_dir::S2MM,
                                map.o[c].ch);
        }
    }
    return true;
}

bool xdna_gemv_seq_build(xdna_seq * seq, const xdna_gemv_geom & geom,
                         const xdna_gemv_seq_opts * opt) {
    if (!seq || !geom.valid()) {
        return false;
    }
    // Appended to a stream that already carries the recurrent core's
    // transfers: its arguments come after the core's, its descriptors after
    // the ones the core's columns have taken, and its activation is a window
    // of the core's own output buffer rather than a buffer of its own.
    const xdna_gemv_seq_opts o_ = opt ? *opt : xdna_gemv_seq_opts{};
    const int      a_w   = o_.w_arg   >= 0 ? o_.w_arg   : o_.arg_base + 0;
    const int      a_a   = o_.act_arg >= 0 ? o_.act_arg : o_.arg_base + 1;
    const int      a_o   = o_.o_arg   >= 0 ? o_.o_arg   : o_.arg_base + 2;
    const uint32_t aoff0 = o_.act_off;
    const uint32_t ooff0 = o_.out_off;
    const uint32_t woff0 = o_.w_off;
    const int    nt     = geom.n_tiles();
    const int    n_out  = geom.n_out();
    const int    rows   = geom.rows();
    const int    n_core = geom.n_core();
    const size_t span   = geom.col_weight_bytes();
    const size_t atb    = geom.act_tile_bytes();
    const gemv_shim_map map = shim_map_for(geom);
    const int    og     = map.o_group;
    const int    n_o    = map.n_o();
    const size_t out_b  = (size_t) og * rows * n_core * sizeof(float);

    // A column's descriptor file is handed out as transfers are emitted, so a
    // column carrying several of the merged layout's streams keeps them apart.
    uint32_t next_bd[XDNA_GEMV_COLS] = {};
    for (int col = 0; col < XDNA_GEMV_COLS; col++) {
        // A negative base means "after whatever this stream has already
        // written on that column", which is the only safe answer when the
        // core's stages hand out their descriptors round by round.
        next_bd[col] = o_.bd_base >= 0 ? (uint32_t) o_.bd_base
                                       : seq->bd_used[col];
    }
    const auto take_bd = [&](int col) { return next_bd[col]++; };

    // Weights: one linear run per column, ordered [col][chunk][tile][row].
    // GGML_XDNA_GEMV_WSPLIT cuts each run into that many descriptors instead
    // of one. It moves exactly the same bytes to exactly the same place, so
    // what it prices is a descriptor - the fixed cost of a dispatch does not
    // come from its bytes or its arithmetic, and this says whether it comes
    // from the length of the instruction stream.
    int wsplit = 1;
    if (const char * ws = getenv("GGML_XDNA_GEMV_WSPLIT")) {
        wsplit = atoi(ws);
        if (wsplit < 1 || span % (size_t) wsplit != 0) {
            wsplit = 1;
        }
    }
    for (int c = 0; c < geom.cols; c++) {
        for (int p = 0; p < wsplit; p++) {
            const size_t   part  = span / (size_t) wsplit;
            const uint32_t off   = (uint32_t) (woff0 + (size_t) c * span +
                                               (size_t) p * part);
            const uint32_t bd_id = take_bd(map.w[c].col);
            xdna_bd bd;
            linear_bd(bd, part, off);
            xdna_seq_blockwrite(seq, map.w[c].col, 0, bd_id, &bd);
            xdna_seq_ddr_patch(seq, map.w[c].col, 0, bd_id, a_w, off);
            xdna_seq_push_queue(seq, map.w[c].col, 0, bd_id, xdna_dma_dir::MM2S,
                                map.w[c].ch, false, 0);
        }
    }

    if (o_.stages == 1) {
        return true;
    }
    // Activations: one descriptor for the count header and every chunk's copy
    // of the K tiles, which xdna_gemv_run lays out consecutively, broadcast to
    // every core. A descriptor per chunk fits the sixteen ids only up to seven
    // chunks and loses transfers before that once the cores are slow enough; a
    // descriptor per column spends half the array's MM2S channels on a few
    // kilobytes.
    // The same fill twice when the prologue has a second input: once with the
    // tiles, once with the host's half of them. Same shape, same object count -
    // the prologue takes one of each per tile - only the endpoint, argument
    // and offset differ.
    const int      a_h_arg = o_.h_arg >= 0 ? o_.h_arg : a_a;
    const uint32_t h_off0  = o_.h_arg >= 0 ? o_.h_off : (uint32_t) aoff0;
    const auto fill_act_on = [&](const shim_ep & ep, int arg, uint32_t base) {
        if (act_repeat_bd() && n_out > 1) {
            const uint32_t h_id = take_bd(ep.col);
            xdna_bd hbd;
            linear_bd(hbd, atb, base);
            xdna_seq_blockwrite(seq, ep.col, 0, h_id, &hbd);
            xdna_seq_ddr_patch(seq, ep.col, 0, h_id, arg, base);
            xdna_seq_push_queue(seq, ep.col, 0, h_id, xdna_dma_dir::MM2S,
                                ep.ch, false, 0);
            const uint32_t b_id = take_bd(ep.col);
            xdna_bd bbd;
            strided_bd(bbd, (size_t) nt * atb, (uint32_t) n_out, 0,
                       (uint32_t) (base + atb));
            xdna_seq_blockwrite(seq, ep.col, 0, b_id, &bbd);
            xdna_seq_ddr_patch(seq, ep.col, 0, b_id, arg, (uint32_t) (base + atb));
            xdna_seq_push_queue(seq, ep.col, 0, b_id, xdna_dma_dir::MM2S,
                                ep.ch, false, 0);
            return;
        }
        const uint32_t bd_id = take_bd(ep.col);
        xdna_bd bd;
        linear_bd(bd, (size_t) (1 + n_out * nt) * atb, base);
        xdna_seq_blockwrite(seq, ep.col, 0, bd_id, &bd);
        xdna_seq_ddr_patch(seq, ep.col, 0, bd_id, arg, base);
        xdna_seq_push_queue(seq, ep.col, 0, bd_id, xdna_dma_dir::MM2S,
                            ep.ch, false, 0);
    };
    fill_act_on(map.a, a_a, (uint32_t) aoff0);
    if (map.has_h) {
        fill_act_on(map.h, a_h_arg, h_off0);
    }

    if (o_.stages == 2) {
        return true;
    }
    // Output: a column owns rows*n_core consecutive floats of every chunk.
    // Each stream gets as many descriptors as its column can still spare, so
    // that as many chunks as possible are in flight at once; when a stream has
    // fewer than one per chunk it waits for the oldest before reusing it.
    // Reprogramming a descriptor whose transfer has not drained loses it - the
    // whole chunk comes back as uniform NaN - which is what the wait is for.
    int streams[XDNA_GEMV_COLS] = {};
    for (int c = 0; c < n_o; c++) {
        streams[map.o[c].col]++;
    }
    int depth[XDNA_GEMV_COLS] = {};
    for (int col = 0; col < XDNA_GEMV_COLS; col++) {
        if (streams[col] == 0) {
            continue;
        }
        const int spare = (MAX_BD - (int) next_bd[col]) / streams[col];
        depth[col] = std::max(1, std::min(spare, n_out));
    }

    // One strided descriptor for a whole stream when the design joins several
    // columns into it. A descriptor per chunk works when a core drains
    // straight to the shim, but on a joined stream it hangs the array - the
    // artifact's own sequence uses one two-dimensional transfer per stream and
    // so does this.
    if (og > 1) {
        for (int c = 0; c < n_o; c++) {
            const uint32_t bd_id = take_bd(map.o[c].col);
            const uint32_t off   = (uint32_t) (ooff0 +
                (size_t) c * (o_.out_stream_stride ? o_.out_stream_stride
                                                   : (uint32_t) out_b));
            xdna_bd bd;
            strided_bd(bd, out_b, (uint32_t) n_out,
                       (size_t) geom.chunk() * sizeof(float), off);
            xdna_seq_blockwrite(seq, map.o[c].col, 0, bd_id, &bd);
            xdna_seq_ddr_patch(seq, map.o[c].col, 0, bd_id, a_o, off);
            xdna_seq_issue_token(seq, map.o[c].col, 0, xdna_dma_dir::S2MM,
                                 map.o[c].ch, 0xF);
            xdna_seq_push_queue(seq, map.o[c].col, 0, bd_id, xdna_dma_dir::S2MM,
                                map.o[c].ch, true, 0);
        }
        if (o_.stages == 3) {
            return true;
        }
        for (int c = 0; c < n_o; c++) {
            xdna_seq_wait_token(seq, map.o[c].col, 0, xdna_dma_dir::S2MM,
                                map.o[c].ch);
        }
        return true;
    }

    std::vector<std::vector<uint32_t>> o_bd((size_t) n_o);
    for (int c = 0; c < n_o; c++) {
        const int d = depth[map.o[c].col];
        if (d <= 0) {
            return false;   // no descriptor left for this stream
        }
        for (int k = 0; k < d; k++) {
            o_bd[(size_t) c].push_back(take_bd(map.o[c].col));
        }
    }

    for (int oc = 0; oc < n_out; oc++) {
        for (int c = 0; c < n_o; c++) {
            const int d = (int) o_bd[(size_t) c].size();
            if (oc >= d) {
                xdna_seq_wait_token(seq, map.o[c].col, 0, xdna_dma_dir::S2MM,
                                    map.o[c].ch);
            }
            const uint32_t bd_id = o_bd[(size_t) c][(size_t) (oc % d)];
            const uint32_t off   = o_.out_stream_stride
                ? (uint32_t) (ooff0 + (size_t) c * o_.out_stream_stride)
                : (uint32_t) (ooff0 +
                              ((size_t) oc * geom.chunk() +
                               (size_t) c * og * rows * n_core) * sizeof(float));
            xdna_bd bd;
            linear_bd(bd, out_b, off);
            xdna_seq_blockwrite(seq, map.o[c].col, 0, bd_id, &bd);
            xdna_seq_ddr_patch(seq, map.o[c].col, 0, bd_id, a_o, off);
            // Every transfer is stamped and waited for, rather than only the
            // last of a column: dispatches follow one another on the same
            // context, and reprogramming a descriptor whose transfer is still
            // in flight corrupts the run that follows.
            xdna_seq_issue_token(seq, map.o[c].col, 0, xdna_dma_dir::S2MM,
                                 map.o[c].ch, 0xF);
            xdna_seq_push_queue(seq, map.o[c].col, 0, bd_id, xdna_dma_dir::S2MM,
                                map.o[c].ch, true, 0);
        }
    }
    for (int c = 0; c < n_o; c++) {
        const int d = std::min((int) o_bd[(size_t) c].size(), n_out);
        for (int k = 0; k < d; k++) {
            xdna_seq_wait_token(seq, map.o[c].col, 0, xdna_dma_dir::S2MM,
                                map.o[c].ch);
        }
    }
    return true;
}

// --- weight packing ---------------------------------------------------------

void xdna_gemv_colmap(const xdna_gemv_geom & geom, std::vector<int32_t> & map) {
    // The fold happens inside each 32-lane group of a core's accumulator, not
    // once per core: a core owning more than 32 columns holds several groups
    // and pairs gate with up within each one. Walking groups is the same thing
    // when n_core is 32 and the right thing when it is not.
    const int LANE = geom.lane();
    const int half = LANE / 2;
    map.assign((size_t) geom.N, 0);
    for (int u = 0; u < geom.N / LANE; u++) {
        for (int i = 0; i < LANE; i++) {
            const int m = u * half + (i % half);
            map[(size_t) u * LANE + i] = i < half ? m : geom.N / 2 + m;
        }
    }
}

bool xdna_gemv_pack_weights(const xdna_gemv_geom & geom,
                            const struct ggml_tensor * const * ws, int n_w,
                            const int32_t * colmap,
                            std::vector<uint8_t> & dst) {
    if (!geom.valid() || !ws || n_w <= 0) {
        return false;
    }
    const int    K      = geom.K;
    const int    grp    = geom.group();
    const int    n_core = geom.n_core();
    const int    rows   = geom.rows();
    const int    NT     = geom.n_tiles();
    const int    gpt    = geom.k_tile() / grp;      // groups per tile
    const size_t tb     = geom.tile_bytes();
    const size_t row_b  = xdna_wfmt_row_bytes(geom.fmt, K);
    const int    LANE   = geom.lane();
    const int    SUP_BYTES = sup_bytes(LANE);
    const int    cb     = block_code_bytes(geom.fmt, LANE);
    const int    bb     = block_bytes(geom.fmt, LANE);
    const int    code_b = geom.fmt == XDNA_WFMT_Q4G32 ? grp / 2 : grp;
    // A source record is one super-block: its groups' codes, then their int8
    // scales, then their int8 mins, then the bf16 pair the two of them scale.
    const int    sbg    = XDNA_SB_VALUES / grp;
    const int    sb_b   = geom.fmt == XDNA_WFMT_Q4G32 ? XDNA_Q4G32_SB_BYTES
                                                      : XDNA_Q8G16_SB_BYTES;
    const int    lg_all = n_core / LANE;
    const int    nsup   = geom.n_sup();

    dst.assign(geom.weight_bytes(), 0);
    std::vector<uint8_t> row(row_b);

    // A ggml weight is N rows of K, so one column of the GEMV is one ggml row:
    // repack it into the canonical group form, then scatter its codes and
    // parameters into the lane the design expects. Columns past the weights
    // stay zero, which the format decodes to a zero weight, so the padded
    // outputs are zero and simply ignored.
    // Destination column -> (weight, row). Without a map the weights simply
    // concatenate; with one the map says where each column comes from.
    std::vector<const struct ggml_tensor *> owner;
    std::vector<int64_t> ownrow;
    owner.reserve((size_t) geom.n_real);
    ownrow.reserve((size_t) geom.n_real);
    for (int wi = 0; wi < n_w; wi++) {
        const struct ggml_tensor * w = ws[wi];
        // q8g16 represents Q4_K exactly as well as Q5_K and Q6_K, so a group
        // whose members do not share a natural format can still be packed -
        // every one of them into q8g16. The narrow format is the one that
        // cannot take a wider type.
        if (!w || w->ne[0] != K ||
            (geom.fmt != XDNA_WFMT_Q8G16 &&
             xdna_wfmt_gemv_for(w->type) != geom.fmt)) {
            return false;
        }
        for (int64_t r0 = 0; r0 < w->ne[1]; r0++) {
            owner.push_back(w);
            ownrow.push_back(r0);
        }
    }

    for (int n = 0; n < geom.N; n++) {
        const int src_col = colmap ? colmap[n] : n;
        if (src_col >= (int) owner.size()) {
            continue;   // padding column: zero weights decode to a zero output
        }
        {
            const struct ggml_tensor * w = owner[(size_t) src_col];
            const int64_t r0 = ownrow[(size_t) src_col];
            const size_t src_rb = ggml_row_size(w->type, K);
            if (!xdna_wfmt_repack_row_as(w->type, geom.fmt,
                                         (const uint8_t *) w->data + (size_t) r0 * src_rb,
                                         K, row.data())) {
                return false;
            }
            const int oc   = n / geom.chunk();
            const int rem  = n % geom.chunk();
            const int core = rem / n_core;
            const int c    = core / rows;
            const int r    = core % rows;
            const int nc   = rem % n_core;
            const int j    = nc / LANE;
            const int lane = nc % LANE;

            for (int t = 0; t < NT; t++) {
                uint8_t * tile = dst.data() +
                    (((size_t) (c * geom.n_out() + oc) * NT + t) * rows + r) * tb;
                for (int gg = 0; gg < gpt; gg++) {
                    const int gidx = t * gpt + gg;
                    const uint8_t * rec = row.data() + (size_t) (gidx / sbg) * sb_b;
                    const int gi  = gidx % sbg;
                    const uint8_t * src = rec + (size_t) gi * code_b;
                    uint8_t * blk = tile + (size_t) (j * gpt + gg) * bb;

                    if (geom.fmt == XDNA_WFMT_Q4G32) {
                        for (int k = 0; k < grp; k++) {
                            const uint8_t v = (k & 1) ? (uint8_t) (src[k >> 1] >> 4)
                                                      : (uint8_t) (src[k >> 1] & 0x0F);
                            uint8_t * p = blk + (size_t) k * (LANE / 2) + lane / 2;
                            *p = (uint8_t) ((*p & (lane & 1 ? 0x0F : 0xF0)) |
                                            (lane & 1 ? (uint8_t) (v << 4) : v));
                        }
                    } else {
                        for (int k = 0; k < grp; k++) {
                            blk[(size_t) k * LANE + lane] = src[k];
                        }
                    }
                    const int8_t * d8 = (const int8_t *) (rec + (size_t) sbg * code_b);
                    uint16_t * pb = (uint16_t *) (blk + cb);
                    pb[lane]        = bf16_of((float) d8[gi]);
                    pb[LANE + lane] = bf16_of((float) d8[sbg + gi]);

                    // The pair is the same for every group of the super-block,
                    // so this writes it once per group and the last one wins.
                    const uint16_t * par =
                        (const uint16_t *) (rec + (size_t) sbg * code_b + 2 * sbg);
                    uint16_t * out = (uint16_t *)
                        (tile + (size_t) lg_all * gpt * bb +
                         (size_t) (j * nsup + gg / (gpt / nsup)) * SUP_BYTES);
                    for (int i = 0; i < 4; i++) {
                        out[(size_t) i * LANE + lane] = par[i];
                    }
                }
            }
        }
    }
    // GGML_XDNA_VERIFY: read the packed tiles back the way the kernel does and
    // compare against the reference decode of the same row. The repack check in
    // xdna-quant covers the record; this covers the scatter into lanes, which
    // is where a format change actually goes wrong.
    if (xdna_verify_level() > 0) {
        std::vector<float> ref((size_t) K), got((size_t) K);
        double num = 0, den = 0;
        for (int n = 0; n < std::min(geom.N, 64); n++) {
            const int src_col = colmap ? colmap[n] : n;
            if (src_col >= (int) owner.size()) {
                continue;
            }
            const struct ggml_tensor * w = owner[(size_t) src_col];
            const size_t src_rb = ggml_row_size(w->type, K);
            if (!xdna_wfmt_repack_row_as(w->type, geom.fmt,
                                         (const uint8_t *) w->data +
                                             (size_t) ownrow[(size_t) src_col] * src_rb,
                                         K, row.data()) ||
                !xdna_wfmt_decode_row(geom.fmt, row.data(), K, ref.data())) {
                continue;
            }
            const int oc = n / geom.chunk(), rem = n % geom.chunk();
            const int core = rem / n_core, c = core / rows, r = core % rows;
            const int nc = rem % n_core, j = nc / LANE, lane = nc % LANE;
            for (int t = 0; t < NT; t++) {
                const uint8_t * tile = dst.data() +
                    (((size_t) (c * geom.n_out() + oc) * NT + t) * rows + r) * tb;
                const uint16_t * sup = (const uint16_t *)
                    (tile + (size_t) lg_all * gpt * bb);
                for (int gg = 0; gg < gpt; gg++) {
                    const uint8_t * blk = tile + (size_t) (j * gpt + gg) * bb;
                    const uint16_t * sp = sup +
                        (size_t) (j * nsup + gg / (gpt / nsup)) * (SUP_BYTES / 2);
                    const auto bf = [](uint16_t hi, uint16_t lo) {
                        const auto w = [](uint16_t h) {
                            uint32_t b = (uint32_t) h << 16;
                            float f;
                            std::memcpy(&f, &b, 4);
                            return f;
                        };
                        return w(hi) + w(lo);
                    };
                    const float dS = bf(sp[lane], sp[LANE + lane]);
                    const float mS = bf(sp[2 * LANE + lane], sp[3 * LANE + lane]);
                    const uint16_t * pb = (const uint16_t *) (blk + cb);
                    const float d = dS * bf(pb[lane], 0);
                    const float m = mS * bf(pb[LANE + lane], 0);
                    for (int k = 0; k < grp; k++) {
                        float q;
                        if (geom.fmt == XDNA_WFMT_Q4G32) {
                            const uint8_t byte = blk[(size_t) k * (LANE / 2) + lane / 2];
                            q = (float) (lane & 1 ? (byte >> 4) : (byte & 0x0F));
                        } else {
                            q = (float) (int8_t) blk[(size_t) k * LANE + lane];
                        }
                        got[(size_t) (t * gpt + gg) * grp + k] = q * d + m;
                    }
                }
            }
            for (int i = 0; i < K; i++) {
                const double e = (double) got[(size_t) i] - ref[(size_t) i];
                num += e * e;
                den += (double) ref[(size_t) i] * ref[(size_t) i];
            }
        }
        fprintf(stderr, "xdna-verify: gemv tile scatter %s K=%d N=%d rel %.3e\n",
                geom.fmt == XDNA_WFMT_Q4G32 ? "q4g32" : "q8g16", K, geom.N,
                den > 0 ? std::sqrt(num / den) : 0.0);
    }
    return true;
}

// --- runner -----------------------------------------------------------------

xdna_gemv * xdna_gemv_create(xdna_kernel_pool * pool, const xdna_gemv_geom & geom,
                             const std::vector<uint8_t> & packed) {
    // Each of these used to fail into one "no artifact" message at the call
    // site, which sent me looking for a missing file when the geometry was the
    // problem.
    if (!pool) {
        return nullptr;
    }
    if (!geom.valid()) {
        GGML_LOG_ERROR("%s: gemv: invalid geometry K=%d N=%d cols=%d rows=%d "
                       "n_core=%d\n", "xdna-gemv", geom.K, geom.N, geom.cols,
                       geom.rows(), geom.n_core());
        return nullptr;
    }
    if (packed.size() != geom.weight_bytes()) {
        GGML_LOG_ERROR("%s: gemv: packed %zu bytes, geometry wants %zu\n",
                       "xdna-gemv", packed.size(), geom.weight_bytes());
        return nullptr;
    }
    xdna_device * dev = pool->device;

    // The stream is per shape, the xclbin is per format. Going through the
    // pool keeps every shape on the same hardware context, which is what makes
    // consecutive ops cheap.
    xdna_seq seq;
    xdna_gemv_seq_opts so_;
    if (act_pro_h() && geom.fused_v) {
        // The design has a buffer for the prologue's second input, so the
        // output is one along. This dispatch's tiles are the host's own, so
        // its second input is the same buffer as its first.
        so_.h_arg = 1;
        so_.o_arg = 3;
    }
    if (!xdna_gemv_seq_build(&seq, geom, &so_)) {
        return nullptr;
    }
    const std::vector<uint32_t> insts = xdna_seq_build(&seq);


    // GGML_XDNA_GEMV_DUMP=<dir> writes each built stream next to IRON's own,
    // so the two can be compared for a shape both can express.
    if (const char * dir = getenv("GGML_XDNA_GEMV_DUMP")) {
        const std::string path = std::string(dir) + "/" + geom.seq_key() + ".insts.bin";
        std::ofstream f(path, std::ios::binary);
        f.write((const char *) insts.data(), (std::streamsize) (insts.size() * 4));
    }

    xdna_gemv * g = new xdna_gemv;
    g->dev  = dev;
    g->geom = geom;
    g->kern = xdna_kernel_pool_get_built(pool, geom.seq_key(), geom.stem().c_str(),
                                         insts.data(), insts.size());
    if (!g->kern) {
        GGML_LOG_ERROR("%s: gemv: cannot load artifact %s\n", "xdna-gemv",
                       geom.stem().c_str());
        xdna_gemv_free(g);
        return nullptr;
    }

    g->w = xdna_buffer_alloc(dev, geom.weight_bytes());
    g->a = xdna_buffer_alloc(dev, geom.act_bytes());
    g->o = xdna_buffer_alloc(dev, geom.out_bytes());
    if (!g->w || !g->a || !g->o) {
        xdna_gemv_free(g);
        return nullptr;
    }
    std::memcpy(g->w->bo.map(), packed.data(), packed.size());
    xdna_buffer_sync_to_device(g->w);
    g->host_a.resize(geom.act_bytes());

    // The buffer set never changes for this weight, so the run is built once.
    xdna_buffer * args[4] = { g->w, g->a, g->a, g->o };
    if (act_pro_h() && g->geom.fused_v) {
        g->run = xdna_kernel_run_make(g->kern, args, 4);
    } else {
        args[2] = g->o;
        g->run = xdna_kernel_run_make(g->kern, args, 3);
    }
    return g;
}

void xdna_gemv_free(xdna_gemv * g) {
    if (!g) {
        return;
    }
    // The kernel belongs to the pool and is shared across shapes.
    xdna_buffer_free(g->w);
    xdna_buffer_free(g->a);
    xdna_buffer_free(g->o);
    delete g;
}

// Quantize `act` into the activation buffer `dst` expects: a header tile, then
// one tile per K tile, then the whole body repeated once per output chunk.
// `flags` is set on every tile (bit 1: the activation is in the layout the
// cores emit), `last_flags` only on the last one - what the core does when the
// chunk closes (bit 0 epilogue, bit 2 write quantized tiles).
//
// A null `act` writes the header and the flag words but no codes: that is how
// a buffer the cores will fill is prepared, since those words sit past the
// last block and so survive every dispatch.
// GGML_XDNA_GEMV_STAGE_PROF=1 splits every dispatch into the host's share -
// quantizing the activation and flushing it - and the device's, so a stage
// that is short of the array's bandwidth can be told from one that is waiting
// on the host.
namespace {

struct gemv_stage_prof {
    struct row { uint64_t n = 0; double host_us = 0, dev_us = 0, sub_us = 0; };
    bool                             on = false;
    std::map<std::string, row>       rows;

    ~gemv_stage_prof() {
        for (const auto & kv : rows) {
            const double n = (double) kv.second.n;
            fprintf(stderr, "xdna-gemv-stage: %-28s runs=%-6llu host=%.0fus "
                    "submit=%.0fus wait=%.0fus\n", kv.first.c_str(),
                    (unsigned long long) kv.second.n,
                    kv.second.host_us / n, kv.second.sub_us / n,
                    kv.second.dev_us / n);
        }
    }
};

gemv_stage_prof & stage_prof() {
    static gemv_stage_prof p = []() {
        gemv_stage_prof q;
        const char * v = getenv("GGML_XDNA_GEMV_STAGE_PROF");
        q.on = v != nullptr && atoi(v) != 0;
        return q;
    }();
    return p;
}

double stage_lap(std::chrono::steady_clock::time_point & t0) {
    const auto now = std::chrono::steady_clock::now();
    const double us = std::chrono::duration<double, std::micro>(now - t0).count();
    t0 = now;
    return us;
}

} // namespace

// GGML_XDNA_ACT_RAW=1 hands the cores bf16 activations in place of codes and
// lets them quantize the tile themselves (gemv_q4.cc quant_tile, the same
// per-group scale this function applies). Nothing about the result changes;
// what changes is that packing the activation stops being something only the
// host can do, which is what keeps the FFN out of the layer's dispatch.
static bool act_raw(void) {
    static const bool on = []() {
        const char * e = getenv("GGML_XDNA_ACT_RAW");
        return e == nullptr || atoi(e) != 0;
    }();
    return on;
}

// Byte offset of the raw activations inside a tile; mirrors ACT_RAW_OFF in
// gemv_q4.cc. Past any tile's codes and group parameters.
static constexpr int XDNA_ACT_RAW_OFF = 0;

static void gemv_pack_act(const xdna_gemv_geom & geom, const float * act,
                          uint8_t * dst, int32_t flags, int32_t last_flags,
                          const float * res = nullptr,
                          const float * gam = nullptr) {
    // A null `act` with a residual and a gamma means the dispatch before this
    // one drained straight into the tiles: everything but acc is the host's.
    const bool acc_on_device = act == nullptr && res && gam;
    const int grp = geom.group();
    const int32_t fmt_word = geom.fmt == XDNA_WFMT_Q4G32 ? 0 : 1;
    int32_t * hdr = (int32_t *) dst;
    hdr[0] = geom.n_tiles();
    hdr[1] = geom.n_out();
    hdr[XDNA_GEMV_ACT_TILE / 4 - 1] = fmt_word;
    hdr[XDNA_GEMV_ACT_TILE / 4 - 2] = 0;

    const size_t atb = geom.act_tile_bytes();
    const int    gpt = geom.k_tile() / grp;     // groups per tile
    for (int t = 0; t < geom.n_tiles(); t++) {
        uint8_t * tile = dst + (size_t) (t + 1) * atb;
        int8_t *  code = (int8_t *) tile;
        float *   gsum = (float *) (tile + geom.k_tile());
        float *   gd   = gsum + gpt;
        const float * src = act ? act + (size_t) t * geom.k_tile() : nullptr;
        (void) src;

        // The raw form: the numbers the prologue tile needs, in place of the
        // codes it will produce from them - this dispatch's input, the
        // residual and the norm's gamma, one bf16 run each. The host does no
        // norm and no quantization at all on this path.
        // Only on the merged artifact: it is the one built with the prologue
        // tile. Raw tiles reaching cores that expect codes is silent garbage.
        const bool raw = (act || acc_on_device) && act_raw() && geom.fused_v &&
                         res && gam;
        if (raw) {
            const int kt = geom.k_tile();
            float *    ac = (float *) (tile + XDNA_ACT_RAW_OFF);
            uint16_t * rw = (uint16_t *) (tile + XDNA_ACT_RAW_OFF + 4 * kt);
            for (int i = 0; i < kt; i++) {
                if (!acc_on_device) {
                    ac[i] = src[i];
                }
                rw[i]      = bf16_of(res[(size_t) t * kt + i]);
                rw[kt + i] = bf16_of(gam[(size_t) t * kt + i]);
            }
            // The row length the norm divides by, in every tile: the prologue
            // reads it there rather than counting objects.
            ((int32_t *) tile)[XDNA_GEMV_ACT_TILE / 4 - 4] = geom.K;
            // The first tile of a chunk, where the reduction starts.
            ((int32_t *) tile)[XDNA_GEMV_ACT_TILE / 4 - 5] = t == 0 ? 1 : 0;
        }
        for (int gg = 0; act && !raw && gg < gpt; gg++) {
            // One scale per group: no reduction over the whole row, so the
            // same quantization is expressible on the device.
            float amax = 0.0f;
            for (int i = 0; i < grp; i++) {
                amax = std::max(amax, std::fabs(src[gg * grp + i]));
            }
            const float d = amax > 0.0f ? amax / 127.0f : 1.0f;
            const float inv = 1.0f / d;
            int sum = 0;
            for (int i = 0; i < grp; i++) {
                const int idx = gg * grp + i;
                int q = (int) lrintf(src[idx] * inv);
                q = std::max(-127, std::min(127, q));
                code[idx] = (int8_t) q;
                sum += q;
            }
            gsum[gg] = (float) sum;
            gd[gg]   = d;
        }
        // Every tile carries the code width; the last one also tells the core
        // that the chunk is complete and the epilogue may close.
        ((int32_t *) tile)[XDNA_GEMV_ACT_TILE / 4 - 1] = fmt_word;
        ((int32_t *) tile)[XDNA_GEMV_ACT_TILE / 4 - 2] =
            flags | (t == geom.n_tiles() - 1 ? last_flags : 0) |
            (raw ? (1 << 4) : 0);
    }
    // Every output chunk replays the same K tiles. The stream re-reads the one
    // body with a zero-stride descriptor (see xdna_gemv_seq_build); the copies
    // are only for the path that does not.
    if (!act_repeat_bd() || geom.n_out() <= 1) {
        const size_t body = (size_t) geom.n_tiles() * atb;
        for (int k = 1; (act || acc_on_device) && k < geom.n_out(); k++) {
            std::memcpy(dst + atb + (size_t) k * body,
                        dst + atb, body);
        }
    }
}

bool xdna_gemv_run_packed(xdna_gemv * g, const void * act_tiles, float * out) {
    if (!g || !act_tiles || !out || g->geom.n_out() != 1) {
        return false;
    }
    const size_t n = (size_t) (1 + g->geom.n_tiles()) * g->geom.act_tile_bytes();
    std::memcpy(g->a->bo.map(), act_tiles, n);
    xdna_buffer_sync_to_device_range(g->a, n, 0);
    if (!xdna_run_restart(g->run, "gemv packed") || !xdna_run_wait(g->run)) {
        return false;
    }
    xdna_buffer_sync_from_device(g->o);
    std::memcpy(out, g->o->bo.map(), (size_t) g->geom.n_real * sizeof(float));
    return true;
}

bool xdna_gemv_read_out(xdna_gemv * g, float * out) {
    if (!g || !out) {
        return false;
    }
    xdna_buffer_sync_from_device(g->o);
    std::memcpy(out, g->o->bo.map(), (size_t) g->geom.n_real * sizeof(float));
    return true;
}

bool xdna_gemv_run(xdna_gemv * g, const float * act, float * out) {
    if (!g || !act || !out) {
        return false;
    }
    const xdna_gemv_geom & geom = g->geom;
    auto t0 = std::chrono::steady_clock::now();
    gemv_pack_act(geom, act, g->host_a.data(), 0, geom.epilogue ? 1 : 0);
    std::memcpy(g->a->bo.map(), g->host_a.data(), g->host_a.size());
    xdna_buffer_sync_to_device(g->a);
    const double host_us = stage_lap(t0);

    // GGML_XDNA_GEMV_NOP=1 skips the dispatch and returns zeros: the result is
    // meaningless, but the time left is everything around the NPU work, which
    // is how the split between dispatch and orchestration was measured.
    static const bool nop = []() {
        const char * v = getenv("GGML_XDNA_GEMV_NOP");
        return v != nullptr && atoi(v) != 0;
    }();
    if (nop) {
        std::memset(out, 0, (size_t) geom.n_real * sizeof(float));
        return true;
    }

    if (!xdna_run_restart(g->run, "gemv")) {
        return false;
    }
    const double sub_us = stage_lap(t0);
    if (!xdna_run_wait(g->run)) {
        return false;
    }
    xdna_buffer_sync_from_device(g->o);
    if (stage_prof().on) {
        auto & r = stage_prof().rows[geom.seq_key()];
        r.n++;
        r.host_us += host_us;
        r.sub_us  += sub_us;
        r.dev_us  += stage_lap(t0);
    }

    // The group scales are applied on the device, so nothing is left here.
    const float * raw = (const float *) g->o->bo.map();
    if (!geom.epilogue) {
        std::memcpy(out, raw, (size_t) geom.n_real * sizeof(float));
    } else {
        // Every 32-lane group carries 16 valid floats and a zeroed tail.
        const int LANE = geom.lane();
        const int half = LANE / 2;
        // The upper half of every slot is written as zero by the epilogue
        // store. If it comes back non-zero the cores never took that branch -
        // they left 32 raw accumulators - and the gather below is reading the
        // wrong values rather than the epilogue being wrong.
        static const bool dbg = getenv("GGML_XDNA_GEMV_DEBUG") != nullptr;
        if (dbg) {
            static int shown = 0;
            if (shown++ < 2) {
                int nz = 0, tot = 0;
                for (int u = 0; u < geom.N / LANE; u++) {
                    const float * slot = raw + (size_t) u * LANE;
                    for (int i = half; i < LANE; i++, tot++) {
                        nz += slot[i] != 0.0f;
                    }
                }
                fprintf(stderr, "xdna-gemv-debug: epilogue K=%d N=%d: %d/%d "
                        "upper-half values non-zero, out[0..3] = %g %g %g %g\n",
                        geom.K, geom.N, nz, tot,
                        raw[0], raw[1], raw[2], raw[3]);
            }
        }
        for (int u = 0; u < geom.N / LANE; u++) {
            std::memcpy(out + (size_t) u * half, raw + (size_t) u * LANE,
                        (size_t) half * sizeof(float));
        }
    }
    return true;
}

// --- paired dispatch -------------------------------------------------------
//
// The FFN's two GEMVs in one instruction stream. The first writes quantized
// activation tiles from the cores straight into the second's activation
// buffer, so between them there is no sync, no host round trip and no second
// submit - the whole pair costs one dispatch.
//
// It works because a descriptor's address patch names an argument index: the
// first dispatch's output descriptors simply name argument 1 (activation)
// instead of argument 2 (output).

xdna_gemv_pair * xdna_gemv_pair_create(xdna_kernel_pool * pool,
                                       const xdna_gemv_geom & g1,
                                       const std::vector<uint8_t> & packed1,
                                       const xdna_gemv_geom & g2,
                                       const std::vector<uint8_t> & packed2) {
    if (!pool || !g1.valid() || !g2.valid()) {
        return nullptr;
    }
    // Everything the fused stream assumes, checked here rather than becoming a
    // wrong address later: one artifact, one core per column, the first
    // closing with an epilogue, and its result exactly filling the second's K.
    if (g1.stem() != g2.stem()) {
        GGML_LOG_ERROR("%s: gemv pair: %s and %s are different artifacts\n",
                       "xdna-gemv", g1.stem().c_str(), g2.stem().c_str());
        return nullptr;
    }
    if (!g1.epilogue || g2.epilogue || g1.rows() != 1 ||
        g1.n_core() != g2.n_core() || g1.cols != g2.cols) {
        return nullptr;
    }
    if ((g1.chunk() / 2) % g2.k_tile() != 0) {
        GGML_LOG_ERROR("%s: gemv pair: a chunk is not a whole number of "
                       "activation tiles (%d vs %d)\n", "xdna-gemv",
                       g1.chunk() / 2, g2.k_tile());
        return nullptr;
    }
    {
        // One descriptor carries a whole output stream, so the blocks the
        // cores it joins produce have to sit inside one activation tile of the
        // second dispatch.
        const int run = (g1.n_core() / 2) * shim_map_for(g1).o_group;
        if (run == 0 || g2.k_tile() % run != 0) {
            GGML_LOG_ERROR("%s: gemv pair: an output stream straddles two "
                           "activation tiles (%d values against a %d tile)\n",
                           "xdna-gemv", run, g2.k_tile());
            return nullptr;
        }
    }
    if (g1.n_real != g2.K || g2.n_out() != 1) {
        // n_out > 1 would need the tiles replicated per chunk, and only the
        // host can do that - the cores each send their block once.
        GGML_LOG_ERROR("%s: gemv pair: mid %d vs K %d, chunks %d\n",
                       "xdna-gemv", g1.n_real, g2.K, g2.n_out());
        return nullptr;
    }
    if (packed1.size() != g1.weight_bytes() || packed2.size() != g2.weight_bytes()) {
        return nullptr;
    }

    const size_t w2_off = g1.weight_bytes();
    const size_t a2_off = g1.act_bytes();

    xdna_seq seq;
    xdna_gemv_pair_opts po;
    if (act_pro_h()) {
        po.h_arg = 2;
        po.o_arg = 3;
    }
    if (!xdna_gemv_seq_build_pair(&seq, g1, g2, w2_off, a2_off, &po)) {
        return nullptr;
    }
    const std::vector<uint32_t> insts = xdna_seq_build(&seq);

    xdna_gemv_pair * p = new xdna_gemv_pair;
    p->dev    = pool->device;
    p->g1     = g1;
    p->g2     = g2;
    p->w2_off = w2_off;
    p->a2_off = a2_off;

    const std::string key = g1.seq_key() + "+" + g2.seq_key();
    p->kern = xdna_kernel_pool_get_built(pool, key, g1.stem().c_str(),
                                         insts.data(), insts.size());
    if (!p->kern) {
        GGML_LOG_ERROR("%s: gemv pair: cannot load artifact %s\n", "xdna-gemv",
                       g1.stem().c_str());
        xdna_gemv_pair_free(p);
        return nullptr;
    }

    p->w = xdna_buffer_alloc(p->dev, w2_off + g2.weight_bytes());
    // Room for the output at the tail of the activation buffer. When the pair
    // is appended to somebody else's stream there is no argument left for an
    // output of its own, and a tail of the buffer it already names costs
    // nothing (see xdna_rec_core_fuse_ffn).
    p->o_tail_off = a2_off + g2.act_bytes();
    p->a = xdna_buffer_alloc(p->dev, p->o_tail_off + g2.out_bytes());
    p->o = xdna_buffer_alloc(p->dev, g2.out_bytes());
    if (act_pro_h()) {
        p->ah = xdna_buffer_alloc(p->dev, g1.act_bytes());
        if (!p->ah) {
            xdna_gemv_pair_free(p);
            return nullptr;
        }
    }
    if (!p->w || !p->a || !p->o) {
        xdna_gemv_pair_free(p);
        return nullptr;
    }
    uint8_t * wmap = (uint8_t *) p->w->bo.map();
    std::memcpy(wmap, packed1.data(), packed1.size());
    std::memcpy(wmap + w2_off, packed2.data(), packed2.size());
    xdna_buffer_sync_to_device(p->w);

    // The second dispatch's activation is written by the cores except for the
    // header and the per-tile flag words, which sit past the last block and so
    // survive every dispatch. They are laid down once here; after this the
    // region is never flushed again, or the stale host copy would land on top
    // of what the cores wrote.
    std::vector<uint8_t> a2(g2.act_bytes(), 0);
    gemv_pack_act(g2, nullptr, a2.data(), 2, 0);
    std::memcpy((uint8_t *) p->a->bo.map() + a2_off, a2.data(), a2.size());
    xdna_buffer_sync_to_device(p->a);

    p->host_a.assign(g1.act_bytes(), 0);
    p->insts = insts;
    xdna_buffer * args[4] = { p->w, p->a, p->ah ? p->ah : p->a, p->o };
    p->run = xdna_kernel_run_make(p->kern, args, p->ah ? 4 : 3);
    if (!p->ah) {
        args[2] = p->o;
        p->run = xdna_kernel_run_make(p->kern, args, 3);
    }
    return p;
}

void xdna_gemv_pair_free(xdna_gemv_pair * p) {
    if (!p) {
        return;
    }
    xdna_buffer_free(p->w);
    xdna_buffer_free(p->a);
    xdna_buffer_free(p->o);
    delete p;
}

int32_t xdna_gemv_pair_last_flags(const xdna_gemv_pair * p) {
    return p ? (1 | 4 | (p->g2.fmt == XDNA_WFMT_Q8G16 ? 8 : 0)) : 0;
}

bool xdna_gemv_pair_run_packed(xdna_gemv_pair * p, const void * act_tiles,
                               float * out) {
    if (!p || !act_tiles || !out) {
        return false;
    }
    const size_t n = (size_t) (1 + p->g1.n_tiles()) * p->g1.act_tile_bytes();
    std::memcpy(p->a->bo.map(), act_tiles, n);
    // Every output chunk replays the same tiles; the device produced one body.
    const size_t body = (size_t) p->g1.n_tiles() * p->g1.act_tile_bytes();
    for (int k = 1; k < p->g1.n_out(); k++) {
        std::memcpy((uint8_t *) p->a->bo.map() + p->g1.act_tile_bytes() +
                        (size_t) k * body,
                    (const uint8_t *) act_tiles + p->g1.act_tile_bytes(), body);
    }
    xdna_buffer_sync_to_device_range(
        p->a, (size_t) (1 + p->g1.n_out() * p->g1.n_tiles()) *
                  p->g1.act_tile_bytes(), 0);
    // After a dispatch on another context the fused one's next run has hung;
    // probe whether re-binding the instruction stream heals it.
    if (getenv("GGML_XDNA_PAIR_REBIND") && !p->insts.empty()) {
        if (!xdna_kernel_bind_insts(p->dev, p->kern, p->insts.data(),
                                    p->insts.size())) {
            return false;
        }
        xdna_buffer * args[3] = { p->w, p->a, p->o };
        p->run = xdna_kernel_run_make(p->kern, args, 3);
    }
    if (getenv("GGML_XDNA_PAIR_TRACE")) {
        fprintf(stderr, "xdna-pair: packed restart\n");
    }
    if (!xdna_run_restart(p->run, "gemv pair") || !xdna_run_wait(p->run)) {
        return false;
    }
    if (getenv("GGML_XDNA_PAIR_TRACE")) {
        fprintf(stderr, "xdna-pair: packed done\n");
    }
    xdna_buffer_sync_from_device(p->o);
    std::memcpy(out, p->o->bo.map(), (size_t) p->g2.n_real * sizeof(float));
    return true;
}

// The host's half of a raw activation, written before the dispatch that fills
// the rest of it. Everything here is known before the layer starts - the
// residual is the layer's input and gamma is the model's - so doing it here
// rather than between the two dispatches is what leaves nothing between them.
// The host's half of a tile, in the layout the prologue's second input wants:
// the residual, then gamma, then the words that describe the tile.
static void gemv_pack_act_host_half(const xdna_gemv_geom & geom, uint8_t * dst,
                                    int32_t last_flags, const float * res,
                                    const float * gam) {
    const int    kt  = geom.k_tile();
    const size_t atb = geom.act_tile_bytes();
    int32_t * hdr = (int32_t *) dst;
    hdr[0] = geom.n_tiles();
    hdr[1] = geom.n_out();
    hdr[XDNA_GEMV_ACT_TILE / 4 - 1] = geom.fmt == XDNA_WFMT_Q4G32 ? 0 : 1;
    hdr[XDNA_GEMV_ACT_TILE / 4 - 2] = 0;
    for (int t = 0; t < geom.n_tiles(); t++) {
        uint8_t *  tile = dst + (size_t) (t + 1) * atb;
        uint16_t * rw   = (uint16_t *) tile;
        for (int i = 0; i < kt; i++) {
            rw[i]      = bf16_of(res[(size_t) t * kt + i]);
            rw[kt + i] = bf16_of(gam[(size_t) t * kt + i]);
        }
        int32_t * w = (int32_t *) tile;
        w[XDNA_GEMV_ACT_TILE / 4 - 1] = geom.fmt == XDNA_WFMT_Q4G32 ? 0 : 1;
        w[XDNA_GEMV_ACT_TILE / 4 - 2] =
            (t == geom.n_tiles() - 1 ? last_flags : 0) | (1 << 4);
        w[XDNA_GEMV_ACT_TILE / 4 - 4] = geom.K;
        w[XDNA_GEMV_ACT_TILE / 4 - 5] = t == 0 ? 1 : 0;
    }
}

bool xdna_gemv_pair_prep_raw(xdna_gemv_pair * p, const float * res,
                             const float * gam) {
    if (!p || !res || !gam) {
        return false;
    }
    const int32_t last = 1 | 4 | (p->g2.fmt == XDNA_WFMT_Q8G16 ? 8 : 0);
    if (p->ah) {
        // Its own buffer, which no dispatch writes - so the order of this
        // against the drain into the tiles does not matter, and it can happen
        // before the one dispatch that does both halves of the layer.
        gemv_pack_act_host_half(p->g1, (uint8_t *) p->ah->bo.map(), last,
                                res, gam);
        xdna_buffer_sync_to_device(p->ah);
        return true;
    }
    gemv_pack_act(p->g1, nullptr, (uint8_t *) p->a->bo.map(), 0, last, res, gam);
    // The whole buffer. The host has just read it back, so this writes the
    // array's own acc where it already is - and only this works: a partial
    // flush does not deliver the drain to the next dispatch at all (see
    // HANDOFF.md for the rest of that table).
    xdna_buffer_sync_to_device(p->a);
    return true;
}

// The dispatch alone, for an activation prepared earlier and completed by the
// dispatch before this one.
void xdna_gemv_pair_act_sync_from(xdna_gemv_pair * p) {
    if (p) {
        xdna_buffer_sync_from_device(p->a);
    }
}

// The output a fused run left in the tail of the activation buffer.
bool xdna_gemv_pair_out_from_tail(xdna_gemv_pair * p, float * out) {
    if (!p || !out) {
        return false;
    }
    xdna_buffer_sync_from_device(p->a);
    std::memcpy(out, (const uint8_t *) p->a->bo.map() + p->o_tail_off,
                (size_t) p->g2.n_real * sizeof(float));
    return true;
}

void xdna_gemv_pair_act_sync_to(xdna_gemv_pair * p) {
    if (p) {
        xdna_buffer_sync_to_device(p->a);
    }
}

bool xdna_gemv_pair_dispatch(xdna_gemv_pair * p, float * out) {
    if (!p || !out) {
        return false;
    }
    // GGML_XDNA_PAIR_IN_DUMP: everything the dispatch reads out of its
    // activation, summed, and what it produced - the same point in the
    // sequence whichever order the host prepared it in.
    const bool dump = getenv("GGML_XDNA_PAIR_IN_DUMP") != nullptr;
    double s_acc = 0, s_res = 0, s_gam = 0;
    int32_t hdr0 = 0, hdr1 = 0, fl = 0;
    if (dump) {
        // Read what the device sees, not what the host last wrote. The host's
        // half comes from its own buffer when it has one, at its own offsets -
        // the residual at zero and gamma behind it, where the second input
        // wants them, not where a whole tile carries them.
        xdna_buffer_sync_from_device(p->a);
        const uint8_t * b = (const uint8_t *) p->a->bo.map();
        const uint8_t * h = b;
        int hres_off = 0;
        if (p->ah) {
            xdna_buffer_sync_from_device(p->ah);
            h = (const uint8_t *) p->ah->bo.map();
        }
        const int kt = p->g1.k_tile();
        if (!p->ah) {
            hres_off = 4 * kt;   // a whole tile: acc first, then the host half
        }
        hdr0 = ((const int32_t *) h)[0];
        hdr1 = ((const int32_t *) h)[1];
        for (int t = 0; t < p->g1.n_tiles(); t++) {
            const uint8_t * tl = b + (size_t) (t + 1) * XDNA_GEMV_ACT_TILE;
            const uint8_t * hl = h + (size_t) (t + 1) * XDNA_GEMV_ACT_TILE;
            const float *    a = (const float *) tl;
            const uint16_t * r = (const uint16_t *) (hl + hres_off);
            for (int i = 0; i < kt; i++) {
                s_acc += a[i];
                s_res += r[i];
                s_gam += r[kt + i];
            }
            fl = ((const int32_t *) hl)[XDNA_GEMV_ACT_TILE / 4 - 2];
        }
    }
    if (!xdna_run_restart(p->run, "gemv pair") || !xdna_run_wait(p->run)) {
        return false;
    }
    xdna_buffer_sync_from_device(p->o);
    std::memcpy(out, p->o->bo.map(), (size_t) p->g2.n_real * sizeof(float));
    if (dump) {
        static int n = 0;
        if (n++ < 4) {
            fprintf(stderr, "pair-in: hdr %d %d flags %d | acc %.6f res %.1f "
                    "gam %.1f -> out %g %g %g\n",
                    hdr0, hdr1, fl, s_acc, s_res, s_gam, out[0], out[1], out[2]);
        }
    }
    return true;
}

static bool xdna_gemv_pair_run_impl(xdna_gemv_pair * p, const float * act,
                                    const float * res, const float * gam,
                                    float * out);

bool xdna_gemv_pair_raw_act(const xdna_gemv_pair * p) {
    return p && act_raw() && p->g1.fused_v;
}

xdna_buffer * xdna_gemv_pair_act_buf(xdna_gemv_pair * p) {
    return p ? p->a : nullptr;
}

xdna_buffer * xdna_gemv_pair_w_buf(xdna_gemv_pair * p) {
    return p ? p->w : nullptr;
}

xdna_buffer * xdna_gemv_pair_h_buf(xdna_gemv_pair * p) {
    return p ? p->ah : nullptr;
}

bool xdna_gemv_act_pro_h(void) {
    return act_pro_h();
}

std::string xdna_gemv_pair_key(const xdna_gemv_pair * p) {
    return p ? p->g1.seq_key() + "+" + p->g2.seq_key() : std::string();
}

size_t xdna_gemv_pair_o_tail(const xdna_gemv_pair * p) {
    return p ? p->o_tail_off : 0;
}

int xdna_gemv_pair_k_tile(const xdna_gemv_pair * p) {
    return p ? p->g1.k_tile() : 0;
}

int xdna_gemv_pair_n_tiles(const xdna_gemv_pair * p) {
    return p ? p->g1.n_tiles() : 0;
}

bool xdna_gemv_pair_run_raw(xdna_gemv_pair * p, const float * acc,
                            const float * res, const float * gam, float * out) {
    return xdna_gemv_pair_run_impl(p, acc, res, gam, out);
}

bool xdna_gemv_pair_run(xdna_gemv_pair * p, const float * act, float * out) {
    return xdna_gemv_pair_run_impl(p, act, nullptr, nullptr, out);
}

static bool xdna_gemv_pair_run_impl(xdna_gemv_pair * p, const float * act,
                                    const float * res, const float * gam,
                                    float * out) {
    if (!p || !out || (!act && !(res && gam))) {
        return false;
    }
    // Close the chunk with the epilogue (bit 0), store the result as a
    // quantized tile rather than floats (bit 2), and group that tile the way
    // the second dispatch's format reads it (bit 3).
    const int32_t last = 1 | 4 | (p->g2.fmt == XDNA_WFMT_Q8G16 ? 8 : 0);
    auto t0 = std::chrono::steady_clock::now();
    if (!act && res && gam) {
        // The dispatch before this one drained its result into the front of
        // every tile, so the staging copy cannot be pushed over the buffer -
        // it holds a stale acc. Write the host's part in place and flush only
        // that: from the residual to the end of the tile, never the acc.
        const int    kt  = p->g1.k_tile();
        const size_t atb = p->g1.act_tile_bytes();
        gemv_pack_act(p->g1, nullptr, (uint8_t *) p->a->bo.map(), 0, last,
                      res, gam);
        // The whole buffer, not the ranges the host touched. The host has
        // just read it back (acc included), so flushing everything writes the
        // array's own values where they already are - and a partial flush
        // does not get acc to the next dispatch, though the host can read it.
        (void) kt;
        xdna_buffer_sync_to_device(p->a);
    } else {
        gemv_pack_act(p->g1, act, p->host_a.data(), 0, last, res, gam);
        std::memcpy(p->a->bo.map(), p->host_a.data(), p->host_a.size());
        xdna_buffer_sync_to_device_range(p->a, p->host_a.size(), 0);
    }
    const double host_us = stage_lap(t0);

    static int flag_dump = 0;
    if (getenv("GGML_XDNA_GEMV_PAIR_DUMP") && flag_dump++ < 2) {
        const int32_t * m32 = (const int32_t *) p->a->bo.map();
        const size_t    st  = XDNA_GEMV_ACT_TILE / 4;
        fprintf(stderr, "xdna-gemv-pair: g1 K=%d N=%d nt=%d nout=%d fmt=%d "
                "g2 K=%d N=%d nt=%d nout=%d fmt=%d last=%d | tile flags",
                p->g1.K, p->g1.N, p->g1.n_tiles(), p->g1.n_out(), (int) p->g1.fmt,
                p->g2.K, p->g2.N, p->g2.n_tiles(), p->g2.n_out(), (int) p->g2.fmt,
                last);
        for (int t = 0; t < p->g1.n_tiles() * p->g1.n_out(); t++) {
            fprintf(stderr, " %d", m32[(1 + t) * st + st - 2]);
        }
        fprintf(stderr, "\n");
    }

    if (!xdna_run_restart(p->run, "gemv pair") || !xdna_run_wait(p->run)) {
        return false;
    }
    xdna_buffer_sync_from_device(p->o);
    std::memcpy(out, p->o->bo.map(), (size_t) p->g2.n_real * sizeof(float));
    if (stage_prof().on) {
        auto & r = stage_prof().rows["pair " + p->g1.seq_key()];
        r.n++;
        r.host_us += host_us;
        r.dev_us  += stage_lap(t0);
    }

    // A one-off timing probe (GGML_XDNA_RUNLIST_PROBE=N) on the pair, which is
    // a stateless projection: repeating it changes nothing.
    static bool probed = false;
    if (!probed && getenv("GGML_XDNA_RUNLIST_PROBE")) {
        probed = true;
        xdna_buffer * pa[3] = { p->w, p->a, p->o };
        xdna_runlist_probe(p->dev, p->kern, pa, 3);
    }

    return true;
}

bool xdna_gemv_pair_read_mid(xdna_gemv_pair * p, std::vector<float> & mid) {
    if (!p) {
        return false;
    }
    xdna_buffer_sync_from_device(p->a);
    const uint8_t * a2  = (const uint8_t *) p->a->bo.map() + p->a2_off;
    const int mid_core  = p->g1.n_core() / 2;
    const int grp       = p->g2.fmt == XDNA_WFMT_Q8G16 ? XDNA_Q8G16_GROUP
                                                       : XDNA_Q4G32_GROUP;
    const int gpb       = mid_core / grp;
    mid.assign((size_t) p->g1.n_real, 0.0f);
    // What a block holds tells the two failures apart: codes and parameters
    // mean the epilogue quantized, f32 that looks like the result itself means
    // it took the plain store instead.
    static int dumped = 0;
    if (getenv("GGML_XDNA_GEMV_PAIR_DUMP") && dumped++ < 2) {
        const uint8_t * b = a2 + XDNA_GEMV_ACT_TILE;
        const int8_t *  c16 = (const int8_t *) b;
        const float *   f32 = (const float *) b;
        fprintf(stderr, "xdna-gemv-pair: block0 i16 %d %d %d %d | f32 %g %g %g %g "
                "| par@128 %g %g %g %g %g %g %g %g | flags %d\n",
                c16[0], c16[1], c16[2], c16[3], f32[0], f32[1], f32[2], f32[3],
                f32[32], f32[33], f32[34], f32[35], f32[36], f32[37], f32[38], f32[39],
                ((const int32_t *) (a2 + XDNA_GEMV_ACT_TILE))[XDNA_GEMV_ACT_TILE / 4 - 2]);
    }
    for (int m = 0; m < p->g1.n_real; m++) {
        const int t   = m / p->g2.k_tile();
        const int blk = (m % p->g2.k_tile()) / mid_core;
        const int i   = m % mid_core;
        const uint8_t * b = a2 + (size_t) (1 + t) * XDNA_GEMV_ACT_TILE +
                            (size_t) blk * p->g1.n_core() * sizeof(float);
        const int8_t  code  = ((const int8_t *) b)[i];
        const float * par   = (const float *) (b + (size_t) mid_core);
        mid[(size_t) m] = (float) code * par[gpb + i / grp];
    }
    return true;
}
