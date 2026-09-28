#include "xdna-att.h"

#include "ggml-impl.h"
#include "ggml.h"
#include "xdna-gemv.h"
#include "xdna-runtime.h"
#include "xdna-seq.h"
#include "xdna-util.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

namespace {

using namespace xdna_att_geom;

constexpr int OUT_OBJ = 256;  // floats an output stream object carries
constexpr int ARG_OUT = 3;

}  // namespace

struct xdna_att {
    xdna_device *      dev  = nullptr;
    xdna_kernel_pool * pool = nullptr;
    xdna_gemv_geom     g;
    size_t             tb = 0;  // bytes a core takes an object
    xdna_gemv_ep       w[XDNA_GEMV_COLS], a, o[XDNA_GEMV_COLS];
    int                n_o = 0;
    xdna_buffer *      act = nullptr;  // header + two q tiles
    xdna_buffer *      ids = nullptr;  // a core's index object, per core
    xdna_buffer *      out = nullptr;  // n_o streams x PIECES objects

    struct variant {
        xdna_kernel * kern = nullptr;
        xrt::run      run;
        xdna_buffer * kbo = nullptr;
        xdna_buffer * vbo = nullptr;
    };

    std::map<int, variant> var;  // by task count
    std::vector<float>     st;
};

int xdna_att_max_positions(void) {
    return xdna_att_geom::max_positions();
}

xdna_att * xdna_att_create(xdna_kernel_pool * pool, xdna_device * dev) {
    if (!pool || !dev) {
        return nullptr;
    }
    xdna_att * a = new xdna_att;
    a->dev       = dev;
    a->pool      = pool;
    a->g         = xdna_gemv_variant(GGML_TYPE_Q4_K, 256, 1024, false, XDNA_GEMV_SPLIT_FUSED);
    a->tb        = a->g.tile_bytes();
    xdna_gemv_endpoints(a->g, a->w, &a->a, a->o, &a->n_o);
    if (!a->g.valid() || a->tb < (size_t) 2 * P * POS_B + 4 || a->n_o * 2 != XDNA_GEMV_COLS) {
        GGML_LOG_ERROR("%s: attention: the pool's geometry is not the one attn-dec.cc expects\n", "xdna-att");
        delete a;
        return nullptr;
    }
    a->act = xdna_buffer_alloc(dev, (size_t) 3 * ACTT);
    a->ids = xdna_buffer_alloc(dev, (size_t) CORES * a->tb);
    a->out = xdna_buffer_alloc(dev, (size_t) a->n_o * PIECES * OUT_OBJ * sizeof(float));
    if (!a->act || !a->ids || !a->out) {
        xdna_att_free(a);
        return nullptr;
    }
    if (!a->act->data || !a->ids->data || !a->out->data) {
        xdna_att_free(a);
        return nullptr;
    }
    // Column c's index object: [lead: 2c | partner: 2c + 1].
    uint8_t * ids = (uint8_t *) a->ids->data;
    for (int id = 0; id < CORES; id++) {
        ((int32_t *) (ids + (size_t) id * a->tb))[0] = id;
    }
    if (!xdna_buffer_sync_to_device(a->ids)) {
        xdna_att_free(a);
        return nullptr;
    }
    a->st.resize((size_t) CORES * ST);
    return a;
}

void xdna_att_free(xdna_att * a) {
    if (!a) {
        return;
    }
    xdna_buffer_free(a->act);
    xdna_buffer_free(a->ids);
    xdna_buffer_free(a->out);
    delete a;
}

namespace {

// The whole dispatch for `n_chunks` chunks a core in `n_task` BD chains.
std::vector<uint32_t> build(const xdna_att * a, int n_task, int n_chunks, size_t koff, size_t voff, bool kv_same) {
    xdna_seq   seq;
    uint32_t   next[XDNA_SEQ_MAX_COLS] = {};
    const auto take                    = [&](int col) {
        return next[col]++;
    };
    const int argv = kv_same ? ARG_K : ARG_V;

    // outputs first: they only wait for data
    for (int s = 0; s < a->n_o; s++) {
        const xdna_gemv_ep & e   = a->o[s];
        const uint32_t       id  = take(e.col);
        const uint32_t       off = (uint32_t) ((size_t) s * PIECES * OUT_OBJ * sizeof(float));
        xdna_bd              bd;
        lin(bd, (size_t) PIECES * OUT_OBJ * sizeof(float), off);
        xdna_seq_blockwrite(&seq, e.col, 0, id, &bd);
        xdna_seq_ddr_patch(&seq, e.col, 0, id, ARG_OUT, off);
        xdna_seq_issue_token(&seq, e.col, 0, xdna_dma_dir::S2MM, e.ch, 0xF);
        xdna_seq_push_queue(&seq, e.col, 0, id, xdna_dma_dir::S2MM, e.ch, true, 0);
    }
    // the activation: header and the two q tiles
    {
        const uint32_t id = take(a->a.col);
        xdna_bd        bd;
        lin(bd, (size_t) 3 * ACTT, 0);
        xdna_seq_blockwrite(&seq, a->a.col, 0, id, &bd);
        xdna_seq_ddr_patch(&seq, a->a.col, 0, id, ARG_ACT, 0);
        xdna_seq_push_queue(&seq, a->a.col, 0, id, xdna_dma_dir::MM2S, a->a.ch, false, 0);
    }
    // per column: the index object, then the chunks
    uint32_t chain[XDNA_GEMV_COLS];
    for (int c = 0; c < XDNA_GEMV_COLS; c++) {
        const xdna_gemv_ep & e   = a->w[c];
        const uint32_t       id  = take(e.col);
        const uint32_t       off = (uint32_t) ((size_t) 2 * c * a->tb);
        xdna_bd              bd;
        lin(bd, 2 * a->tb, off);
        xdna_seq_blockwrite(&seq, e.col, 0, id, &bd);
        xdna_seq_ddr_patch(&seq, e.col, 0, id, ARG_IDS, off);
        xdna_seq_push_queue(&seq, e.col, 0, id, xdna_dma_dir::MM2S, e.ch, false, 0);
        chain[c] = next[e.col];
        next[e.col] += 6;
    }
    emit_chunks(seq, a->w, a->tb, n_task, n_chunks, chain, koff, voff, argv);
    for (int s = 0; s < a->n_o; s++) {
        xdna_seq_wait_token(&seq, a->o[s].col, 0, xdna_dma_dir::S2MM, a->o[s].ch);
    }
    return xdna_seq_build(&seq);
}

}  // namespace

bool xdna_att_run(xdna_att *    a,
                  xdna_buffer * kbo,
                  size_t        koff,
                  xdna_buffer * vbo,
                  size_t        voff,
                  const float * q,
                  float         scale,
                  int           n_valid,
                  float *       out) {
    if (!a || !kbo || !vbo || !q || !out || !kbo->data || !vbo->data || n_valid <= 0 ||
        n_valid > xdna_att_max_positions()) {
        GGML_LOG_ERROR("%s: attention: bad call: a buffer is null or unmapped (k=%p v=%p q=%p out=%p n_valid=%d)\n",
                       "xdna-att", (const void *) kbo, (const void *) vbo, (const void *) q, (const void *) out,
                       n_valid);
        return false;
    }
    const int n_chunks = xdna_att_geom::chunk_count(n_valid);
    const int n_task   = xdna_att_geom::task_count(n_chunks);
    // every chunk the cores are sent has to be inside the buffers
    if (!xdna_att_geom::fits(kbo->bytes, koff, vbo->bytes, voff, n_valid)) {
        GGML_LOG_ERROR("%s: attention: %d positions do not fit the K/V cache buffers\n", "xdna-att", n_valid);
        return false;
    }
    const bool                  same  = kbo == vbo;
    const std::vector<uint32_t> words = build(a, n_task, n_chunks, koff, voff, same);
    xdna_att::variant &         v     = a->var[n_task];
    if (!v.kern) {
        char name[32];
        snprintf(name, sizeof(name), "att_dec_t%d", n_task);
        v.kern = xdna_kernel_pool_get_built(a->pool, name, a->g.stem().c_str(), words.data(), words.size());
        if (!v.kern) {
            return false;  // the pool names the kernel it could not load
        }
    } else if (!xdna_kernel_rewrite_insts(v.kern, words.data(), words.size())) {
        return false;  // xdna-runtime names the stream and the reason
    }
    if (v.kbo != kbo || v.vbo != vbo) {
        xdna_buffer * args[5] = { kbo, a->act, a->ids, a->out, vbo };
        v.run                 = xdna_kernel_run_make(v.kern, args, 5);
        v.kbo                 = kbo;
        v.vbo                 = vbo;
    }

    // header, then heads 0-3 and 4-7 as bf16 with the scale folded in, tiled
    // for the kernel's QK multiply: [32 blocks of 8 dims][8 dims][4 heads]
    uint8_t * act = (uint8_t *) a->act->data;
    std::memset(act, 0, (size_t) 3 * ACTT);
    int32_t * hdr = (int32_t *) act;
    hdr[2]        = 1 + n_chunks;
    hdr[3]        = 1;
    for (int t = 0; t < 2; t++) {
        uint16_t * qb = (uint16_t *) (act + (size_t) (1 + t) * ACTT);
        for (int hh = 0; hh < 4; hh++) {
            for (int d = 0; d < D; d++) {
                qb[(d / 8) * 32 + (d % 8) * 4 + hh] = xdna_bf16(q[(size_t) (4 * t + hh) * D + d] * scale);
            }
        }
        int32_t * w      = (int32_t *) qb;
        w[4 * D / 2]     = n_valid;
        w[4 * D / 2 + 1] = t;
        // GGML_XDNA_ATTN_SKIP: move the data, skip the arithmetic - what the
        // DMA alone costs (diagnostic; the output is garbage)
        w[4 * D / 2 + 2] = getenv("GGML_XDNA_ATTN_SKIP") ? 1 : 0;
    }
    if (!xdna_buffer_sync_to_device(a->act)) {
        return false;
    }

    if (!xdna_run_restart(v.run) || !xdna_run_wait(v.run)) {
        return false;
    }
    if (!xdna_buffer_sync_from_device(a->out)) {
        return false;
    }
    const float * ob = (const float *) a->out->data;
    // core 2c + r: stream c / 2, the object's quarter (c % 2) * 2 + r
    for (int id = 0; id < CORES; id++) {
        const int c = id / 2, r = id % 2;
        const int s  = c / 2;
        float *   st = a->st.data() + (size_t) id * ST;
        for (int i = 0; i < PIECES; i++) {
            std::memcpy(st + (size_t) i * PIECE,
                        ob + ((size_t) s * PIECES + i) * OUT_OBJ + (size_t) ((c % 2) * 2 + r) * PIECE,
                        PIECE * sizeof(float));
        }
    }
    // log-sum-exp over the sixteen partials. A core's state: m then l, sixteen
    // lanes a kv group with head h % 4 in lane h % 4, then o per group as
    // [32 blocks of 8 dims][4 heads][8 dims]
    for (int h = 0; h < H; h++) {
        const int mi = (h / 4) * 16 + h % 4, li = 32 + mi;
        float     mx = -1e30f;
        for (int id = 0; id < CORES; id++) {
            mx = std::max(mx, a->st[(size_t) id * ST + mi]);
        }
        double              l = 0.0;
        std::vector<double> o((size_t) D, 0.0);
        for (int id = 0; id < CORES; id++) {
            const float * st = a->st.data() + (size_t) id * ST;
            if (st[li] <= 0.0f) {
                continue;
            }
            const double w = std::exp((double) st[mi] - (double) mx);
            l += w * (double) st[li];
            const float * oh = st + 64 + (size_t) (h / 4) * 4 * D + (size_t) (h % 4) * 8;
            for (int d = 0; d < D; d++) {
                o[(size_t) d] += w * (double) oh[(d / 8) * 32 + d % 8];
            }
        }
        for (int d = 0; d < D; d++) {
            out[(size_t) h * D + d] = l > 0.0 ? (float) (o[(size_t) d] / l) : 0.0f;
        }
    }
    return true;
}
