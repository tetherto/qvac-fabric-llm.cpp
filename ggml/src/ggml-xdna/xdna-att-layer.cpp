#include "xdna-att-layer.h"

#include "xdna-gemv.h"
#include "xdna-rec-gemv.h"
#include "xdna-runtime.h"
#include "xdna-seq.h"

#include "ggml-impl.h"
#include "ggml.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <vector>

namespace {

// Mirrors attn-dec.cc, act-att.cc and the pool of kernels/gemv_q4.py.
constexpr int D        = 256;
constexpr int H        = 8;
constexpr int KVH      = 2;
constexpr int P        = 5;                  // positions a chunk
constexpr int CORES    = 16;
constexpr int PIECES   = 33;
constexpr int PIECE    = 64;                 // floats
constexpr int ST       = PIECES * PIECE;     // a core's partial state, floats
constexpr int POS_B    = KVH * D * 2;        // one position of K (or V), f16
constexpr int ROUND    = CORES * P;
constexpr int TASK     = 64;                 // chunks one BD chain covers
constexpr int MAX_TASK = 16;
constexpr int ACTT     = XDNA_GEMV_ACT_TILE;
constexpr int AW       = ACTT / 4;
constexpr int SIDE_W   = 512;                // a side object, words
constexpr int N_PART   = CORES * ST / SIDE_W;   // 66
constexpr int N_QKV    = 2 * H * D + 2 * KVH * D;  // q|gate, k, v: 5120
// The prologue tile's own streams (gemv_q4.py PRO_SIDE_COL / PRO_EMIT_COL).
constexpr int SIDE_COL = 6, SIDE_CH = 1;
constexpr int EMIT_COL = 5, EMIT_CH = 0;
// Their descriptor ids, above what the pool's phases take on those columns.
constexpr uint32_t SIDE_BD = 11;
// The combine's activation, queued during the attention on the broadcast.
constexpr uint32_t ACT_O_BD = 13;
constexpr uint32_t EMIT_BD = 14;

// act-att.cc's words
constexpr int W_NS = AW - 6, W_NE = AW - 7, W_MODE = 518, W_NPART = 519;
constexpr int F_ATT = 1 << 5;

// The dispatch's arguments. The firmware translates the first five itself;
// the rest carry the aperture (xdna_seq_ddr_patch).
constexpr int ARG_K = 0, ARG_ACT = 1, ARG_IDS = 2, ARG_R = 3, ARG_V = 4,
              ARG_W = 5, ARG_FFN = 6, ARG_RES = 7;

// Scratch: the projection's output, then the partial states.
constexpr size_t R_QKV  = 0;
constexpr size_t R_PART = (size_t) N_QKV * sizeof(float);
constexpr size_t R_SIZE = R_PART + (size_t) CORES * ST * sizeof(float);

size_t align4k(size_t n) { return (n + 4095) / 4096 * 4096; }

void lin(xdna_bd & bd, size_t bytes, uint32_t off) {
    bd.buf_len   = (uint32_t) (bytes / 4);
    bd.buf_off   = off;
    bd.d0_size   = 0;
    bd.d0_stride = 1;
    bd.d1_size   = 0;
    bd.d1_stride = 1;
    bd.d2_stride = 1;
    bd.ax_cache  = 2;
}

// `n` runs of `run` words, `stride` words apart. A dimension's size is ten
// bits: a longer run silently becomes a wrong pattern.
void runs(xdna_bd & bd, uint32_t run, uint32_t n, uint32_t stride, uint32_t off) {
    GGML_ASSERT(run < 1024 && n < 1024);
    bd.buf_len   = run * n;
    bd.buf_off   = off;
    bd.d0_size   = run;
    bd.d0_stride = 1;
    bd.d1_size   = n;
    bd.d1_stride = stride;
    bd.d2_stride = 1;
    bd.ax_cache  = 2;
}

int32_t fbits(float f) { int32_t w; memcpy(&w, &f, 4); return w; }

} // namespace

struct xdna_att_layer {
    xdna_kernel_pool * pool = nullptr;
    xdna_device *      dev  = nullptr;
    xdna_rec_gemv *    rg   = nullptr;   // attn_output and the FFN pair
    xdna_gemv_geom     gqkv;
    xdna_gemv_geom     gatt;              // the pool's geometry for the attention phase
    size_t             tb = 0;
    xdna_gemv_ep       w[XDNA_GEMV_COLS], a, o[XDNA_GEMV_COLS];
    int                n_o = 0;

    size_t w_o = 0, w_ffn = 0;           // weight offsets
    size_t a_qkv = 0, a_att = 0, a_const = 0, a_o = 0, a_gattn = 0, a_gpost = 0, a_size = 0;
    float  eps_attn = -1.0f, eps_post = -1.0f;   // what the row tiles carry

    xdna_buffer * wbuf = nullptr;
    xdna_buffer * act  = nullptr;
    xdna_buffer * ids  = nullptr;
    xdna_buffer * r    = nullptr;

    std::vector<float> gq, gk;
    struct variant {
        xdna_kernel * kern = nullptr;
        xrt::run      run;
        xdna_buffer * kbo = nullptr;
        xdna_buffer * vbo = nullptr;
        xdna_buffer * res = nullptr;
        std::vector<xdna_buffer *> args;   // the run's, for a joined token stream
    };
    std::map<int, variant> var;
    std::vector<float> ffn_out;
};

int xdna_att_layer_max_positions(void) {
    return MAX_TASK * TASK * ROUND;
}

bool xdna_att_layer_fits(const xdna_buffer * kbo, size_t koff, const xdna_buffer * vbo,
                         size_t voff, int n_valid) {
    if (!kbo || !vbo || n_valid <= 0 || n_valid > xdna_att_layer_max_positions()) {
        return false;
    }
    const int n_chunks = (n_valid + ROUND - 1) / ROUND;
    const size_t span  = (size_t) n_chunks * ROUND * POS_B;
    return koff + span <= kbo->bytes && voff + span <= vbo->bytes &&
           koff + span <= UINT32_MAX && voff + span <= UINT32_MAX;
}

void xdna_att_layer_free(xdna_att_layer * l) {
    if (!l) {
        return;
    }
    xdna_buffer_free(l->wbuf);
    xdna_buffer_free(l->act);
    xdna_buffer_free(l->ids);
    xdna_buffer_free(l->r);
    xdna_rec_gemv_free(l->rg);
    delete l;
}

xdna_att_layer * xdna_att_layer_create(xdna_kernel_pool * pool, const xdna_att_layer_w & wt) {
    if (!pool || !wt.wq || !wt.wk || !wt.wv || !wt.gq || !wt.gk || !wt.wo || !wt.post ||
        !wt.attn_norm || wt.attn_norm->type != GGML_TYPE_F32 ||
        ggml_nelements(wt.attn_norm) != XDNA_RES_D || wt.wq->ne[0] != XDNA_RES_D ||
        !wt.gate || !wt.up || !wt.down) {
        return nullptr;
    }
    const int64_t K = wt.wq->ne[0];
    if (wt.wq->ne[1] != 2 * H * D || wt.wk->ne[1] != KVH * D || wt.wv->ne[1] != KVH * D ||
        wt.wk->ne[0] != K || wt.wv->ne[0] != K || wt.wo->ne[0] != H * D ||
        wt.gq->type != GGML_TYPE_F32 || wt.gk->type != GGML_TYPE_F32 ||
        wt.post->type != GGML_TYPE_F32 || ggml_nelements(wt.gq) != D ||
        ggml_nelements(wt.gk) != D || ggml_nelements(wt.post) != wt.wo->ne[1]) {
        return nullptr;
    }
    std::unique_ptr<xdna_att_layer> l(new xdna_att_layer);
    l->pool = pool;
    l->rg = xdna_rec_gemv_create(pool, wt.wo, wt.gate, wt.up, wt.down);
    if (!l->rg || !l->rg->ffn || !xdna_gemv_pair_raw_act(l->rg->ffn)) {
        return nullptr;
    }
    xdna_gemv * so = l->rg->so;
    xdna_gemv_pair * ffn = l->rg->ffn;
    l->dev = so->dev;

    // q|gate, k and v in one phase: the widest of the three formats holds all
    // of them (q8g16 takes any), which is one pipeline fill instead of two.
    const ggml_tensor * ws[3] = { wt.wq, wt.wk, wt.wv };
    l->gqkv = xdna_gemv_variant(xdna_gemv_type_of(wt.wq), K, N_QKV, false, XDNA_GEMV_SPLIT_FUSED);
    for (int i = 1; i < 3; i++) {
        const xdna_gemv_geom g = xdna_gemv_variant(xdna_gemv_type_of(ws[i]), K, N_QKV, false,
                                                   XDNA_GEMV_SPLIT_FUSED);
        if (!g.valid()) {
            return nullptr;
        }
        if (g.fmt == XDNA_WFMT_Q8G16) {
            l->gqkv = g;
        }
    }
    std::vector<uint8_t> packed;
    if (!l->gqkv.valid() || l->gqkv.N != N_QKV ||
        !xdna_gemv_pack_weights(l->gqkv, ws, 3, nullptr, packed)) {
        return nullptr;
    }
    // The pool's own view for the attention phase (attn-dec.cc takes its
    // chunks as weight objects of this size).
    l->gatt = xdna_gemv_variant(GGML_TYPE_Q4_K, 256, 1024, false, XDNA_GEMV_SPLIT_FUSED);
    l->tb   = l->gatt.tile_bytes();
    xdna_gemv_endpoints(l->gatt, l->w, &l->a, l->o, &l->n_o);
    if (!l->gatt.valid() || l->tb < (size_t) 2 * P * POS_B + 4 || l->n_o * 2 != XDNA_GEMV_COLS ||
        so->geom.n_tiles() * so->geom.k_tile() != H * D || so->geom.n_out() != 1 ||
        so->geom.fmt != XDNA_WFMT_Q4G32 || so->geom.k_tile() != 256) {
        return nullptr;
    }

    // weights: [qkv | attn_output | FFN]
    l->w_o   = align4k(l->gqkv.weight_bytes());
    l->w_ffn = align4k(l->w_o + so->geom.weight_bytes());
    const size_t ffn_wb = ffn->w2_off + ffn->g2.weight_bytes();
    l->wbuf = xdna_buffer_alloc(l->dev, l->w_ffn + ffn_wb);
    // activation: [qkv input | attention header + 2 q tiles | consts | combine]
    l->a_qkv   = 0;
    l->a_att   = align4k(l->gqkv.act_bytes());
    l->a_const = l->a_att + 3 * ACTT;
    l->a_o     = align4k(l->a_const + SIDE_W * 4);
    l->a_gattn = align4k(l->a_o + so->geom.act_bytes());
    l->a_gpost = l->a_gattn + (size_t) XDNA_RES_D * sizeof(float);
    l->a_size  = l->a_gpost + (size_t) XDNA_RES_D * sizeof(float);
    l->act = xdna_buffer_alloc(l->dev, l->a_size);
    l->ids = xdna_buffer_alloc(l->dev, (size_t) CORES * l->tb);
    l->r   = xdna_buffer_alloc(l->dev, R_SIZE);
    if (!l->wbuf || !l->act || !l->ids || !l->r) {
        return nullptr;
    }
    uint8_t * w = (uint8_t *) l->wbuf->bo.map();
    std::memset(w, 0, l->w_ffn + ffn_wb);
    std::memcpy(w, packed.data(), packed.size());
    std::memcpy(w + l->w_o, so->w->bo.map(), so->geom.weight_bytes());
    std::memcpy(w + l->w_ffn, ffn->w->bo.map(), ffn_wb);
    xdna_buffer_sync_to_device(l->wbuf);

    uint8_t * ids = (uint8_t *) l->ids->bo.map();
    std::memset(ids, 0, (size_t) CORES * l->tb);
    for (int id = 0; id < CORES; id++) {
        ((int32_t *) (ids + (size_t) id * l->tb))[0] = id;
    }
    xdna_buffer_sync_to_device(l->ids);

    l->gq.assign((const float *) wt.gq->data, (const float *) wt.gq->data + D);
    l->gk.assign((const float *) wt.gk->data, (const float *) wt.gk->data + D);

    // The combine's activation is fixed: the projection's count header, which
    // takes the gate while the pool runs the attention, then eight tiles the
    // prologue fills - the first takes the partial states, every one
    // quantizes its head.
    uint8_t * a = (uint8_t *) l->act->bo.map();
    std::memset(a, 0, l->a_size);
    std::memcpy(a + l->a_gattn, wt.attn_norm->data, (size_t) XDNA_RES_D * sizeof(float));
    std::memcpy(a + l->a_gpost, wt.post->data, (size_t) XDNA_RES_D * sizeof(float));
    int32_t * hdr = (int32_t *) (a + l->a_o);
    hdr[0] = so->geom.n_tiles();
    hdr[1] = so->geom.n_out();
    hdr[W_MODE] = 3;
    hdr[W_NS] = H / 2;
    hdr[AW - 1] = 0;
    hdr[AW - 2] = F_ATT;
    for (int k = 0; k < so->geom.n_tiles(); k++) {
        int32_t * t = (int32_t *) (a + l->a_o + (size_t) (1 + k) * ACTT);
        t[517]     = k;
        t[W_MODE]  = 2;
        t[W_NPART] = N_PART;
        t[W_NS]    = k == 0 ? N_PART : 0;
        t[W_NE]    = 0;
        t[AW - 1]  = 0;       // q4g32 codes
        t[AW - 2]  = F_ATT;
    }
    l->ffn_out.resize((size_t) ffn->g2.n_real);
    return l.release();
}

namespace {

std::vector<uint32_t> build(const xdna_att_layer * l, int n_task, int n_chunks, int row,
                            size_t koff, size_t voff, bool kv_same) {
    xdna_seq seq;
    xdna_gemv * so = l->rg->so;
    xdna_gemv_pair * ffn = l->rg->ffn;
    const int argv = kv_same ? ARG_K : ARG_V;
    bool ok = true;

    // 1. q|gate, k, v into the scratch. Its input is the prologue's row:
    //    h = F + A into H, rms_norm(h) * gamma into the tiles.
    const auto row_io = [&](uint32_t acc, uint32_t hres, int g_arg, uint32_t g_off,
                            uint32_t h_out) {
        xdna_seq_row_io(&seq, ARG_RES, acc, hres, g_arg, g_off, h_out);
    };
    const auto row_wait = [&]() { xdna_seq_row_wait(&seq); };
    row_io(XDNA_RES_F, XDNA_RES_A, ARG_ACT, (uint32_t) l->a_gattn, XDNA_RES_H);
    {
        xdna_gemv_seq_opts o;
        o.bd_base = 0;
        o.w_arg   = ARG_W;
        o.w_off   = 0;
        o.act_arg = ARG_ACT;
        o.act_off = (uint32_t) l->a_qkv;
        o.o_arg   = ARG_R;
        o.out_off = (uint32_t) R_QKV;
        ok = ok && xdna_gemv_seq_build(&seq, l->gqkv, &o);
    }
    row_wait();

    // 2. prepare and attention
    uint32_t next[XDNA_SEQ_MAX_COLS] = {};
    const auto take = [&](int col) { return next[col]++; };
    // the partial states, a core's 2112 floats back to back: stream s carries
    // cores 4s..4s+3 as the quarters of each of its 33 objects
    for (int s = 0; s < l->n_o; s++) {
        const xdna_gemv_ep & e = l->o[s];
        const uint32_t id  = take(e.col);
        const uint32_t off = (uint32_t) (R_PART + (size_t) 4 * s * ST * sizeof(float));
        xdna_bd bd;
        bd.buf_len   = PIECES * 4 * PIECE;
        bd.buf_off   = off;
        bd.d0_size   = PIECE;
        bd.d0_stride = 1;
        bd.d1_size   = 4;
        bd.d1_stride = ST;
        bd.d2_stride = PIECE;
        bd.ax_cache  = 2;
        xdna_seq_blockwrite(&seq, e.col, 0, id, &bd);
        xdna_seq_ddr_patch(&seq, e.col, 0, id, ARG_R, off);
        xdna_seq_issue_token(&seq, e.col, 0, xdna_dma_dir::S2MM, e.ch, 0xF);
        xdna_seq_push_queue(&seq, e.col, 0, id, xdna_dma_dir::S2MM, e.ch, true, 0);
    }
    // this position's K and V rows, from the prologue into the cache
    {
        xdna_bd bk, bv;
        lin(bk, POS_B, (uint32_t) (koff + (size_t) row * POS_B));
        lin(bv, POS_B, (uint32_t) (voff + (size_t) row * POS_B));
        bk.next_bd = EMIT_BD + 1;
        bk.use_next = true;
        xdna_seq_blockwrite(&seq, EMIT_COL, 0, EMIT_BD, &bk);
        xdna_seq_ddr_patch(&seq, EMIT_COL, 0, EMIT_BD, ARG_K, bk.buf_off);
        xdna_seq_blockwrite(&seq, EMIT_COL, 0, EMIT_BD + 1, &bv);
        xdna_seq_ddr_patch(&seq, EMIT_COL, 0, EMIT_BD + 1, argv, bv.buf_off);
        xdna_seq_issue_token(&seq, EMIT_COL, 0, xdna_dma_dir::S2MM, EMIT_CH, 0xF);
        xdna_seq_push_queue(&seq, EMIT_COL, 0, EMIT_BD, xdna_dma_dir::S2MM, EMIT_CH, true, 0);
    }
    // the header and the two prepare tiles
    {
        const uint32_t id = take(l->a.col);
        xdna_bd bd;
        lin(bd, 3 * ACTT, (uint32_t) l->a_att);
        xdna_seq_blockwrite(&seq, l->a.col, 0, id, &bd);
        xdna_seq_ddr_patch(&seq, l->a.col, 0, id, ARG_ACT, bd.buf_off);
        xdna_seq_push_queue(&seq, l->a.col, 0, id, xdna_dma_dir::MM2S, l->a.ch, false, 0);
        // and the combine's, so its header takes the gate during the attention
        xdna_bd bo;
        lin(bo, so->geom.act_bytes(), (uint32_t) l->a_o);
        xdna_seq_blockwrite(&seq, l->a.col, 0, ACT_O_BD, &bo);
        xdna_seq_ddr_patch(&seq, l->a.col, 0, ACT_O_BD, ARG_ACT, bo.buf_off);
        xdna_seq_push_queue(&seq, l->a.col, 0, ACT_O_BD, xdna_dma_dir::MM2S, l->a.ch, false, 0);
    }
    // the side objects: consts, q of heads 0-3, k|v, q of 4-7 for the prepare
    // tiles, then the gate for the combine's header
    {
        xdna_bd bc, bq0, bkv, bq1, bg;
        lin(bc, SIDE_W * 4, (uint32_t) l->a_const);
        runs(bq0, D, 4, 2 * D, (uint32_t) R_QKV);
        lin(bkv, (size_t) 2 * KVH * D * sizeof(float),
            (uint32_t) (R_QKV + (size_t) 2 * H * D * sizeof(float)));
        runs(bq1, D, 4, 2 * D, (uint32_t) (R_QKV + (size_t) 4 * 2 * D * sizeof(float)));
        runs(bg, D, H, 2 * D, (uint32_t) (R_QKV + (size_t) D * sizeof(float)));
        bc.next_bd = SIDE_BD + 1;  bc.use_next = true;
        bq0.next_bd = SIDE_BD + 2; bq0.use_next = true;
        bkv.next_bd = SIDE_BD + 3; bkv.use_next = true;
        bq1.next_bd = SIDE_BD + 4; bq1.use_next = true;
        xdna_seq_blockwrite(&seq, SIDE_COL, 0, SIDE_BD, &bc);
        xdna_seq_ddr_patch(&seq, SIDE_COL, 0, SIDE_BD, ARG_ACT, bc.buf_off);
        xdna_seq_blockwrite(&seq, SIDE_COL, 0, SIDE_BD + 1, &bq0);
        xdna_seq_ddr_patch(&seq, SIDE_COL, 0, SIDE_BD + 1, ARG_R, bq0.buf_off);
        xdna_seq_blockwrite(&seq, SIDE_COL, 0, SIDE_BD + 2, &bkv);
        xdna_seq_ddr_patch(&seq, SIDE_COL, 0, SIDE_BD + 2, ARG_R, bkv.buf_off);
        xdna_seq_blockwrite(&seq, SIDE_COL, 0, SIDE_BD + 3, &bq1);
        xdna_seq_ddr_patch(&seq, SIDE_COL, 0, SIDE_BD + 3, ARG_R, bq1.buf_off);
        xdna_seq_blockwrite(&seq, SIDE_COL, 0, SIDE_BD + 4, &bg);
        xdna_seq_ddr_patch(&seq, SIDE_COL, 0, SIDE_BD + 4, ARG_R, bg.buf_off);
        xdna_seq_issue_token(&seq, SIDE_COL, 0, xdna_dma_dir::MM2S, SIDE_CH, 0xF);
        xdna_seq_push_queue(&seq, SIDE_COL, 0, SIDE_BD, xdna_dma_dir::MM2S, SIDE_CH, true, 0);
    }
    // the chunks read the cache: this position's row has to be in it first
    xdna_seq_wait_token(&seq, EMIT_COL, 0, xdna_dma_dir::S2MM, EMIT_CH);
    uint32_t chain[XDNA_GEMV_COLS];
    for (int c = 0; c < XDNA_GEMV_COLS; c++) {
        const xdna_gemv_ep & e = l->w[c];
        const uint32_t id  = take(e.col);
        const uint32_t off = (uint32_t) ((size_t) 2 * c * l->tb);
        xdna_bd bd;
        lin(bd, 2 * l->tb, off);
        xdna_seq_blockwrite(&seq, e.col, 0, id, &bd);
        xdna_seq_ddr_patch(&seq, e.col, 0, id, ARG_IDS, off);
        xdna_seq_push_queue(&seq, e.col, 0, id, xdna_dma_dir::MM2S, e.ch, false, 0);
        chain[c] = next[e.col];
        next[e.col] += 6;
        ok = ok && next[e.col] <= (e.col == SIDE_COL ? SIDE_BD : e.col == EMIT_COL ? EMIT_BD :
                                     e.col == l->a.col ? ACT_O_BD : 16u);
    }
    const size_t pad = l->tb - (size_t) 2 * P * POS_B;
    for (int t = 0; t < n_task; t++) {
        const int cnt = std::min(TASK, n_chunks - t * TASK);
        for (int c = 0; c < XDNA_GEMV_COLS; c++) {
            const xdna_gemv_ep & e = l->w[c];
            if (t > 0) {
                xdna_seq_wait_token(&seq, e.col, 0, xdna_dma_dir::MM2S, e.ch);
            }
            for (int r = 0; r < 2; r++) {
                const size_t pos = (size_t) t * TASK * ROUND + (size_t) (2 * c + r) * P;
                const uint32_t b0 = chain[c] + 3 * r;
                xdna_bd bk, bv, bp;
                lin(bk, (size_t) P * POS_B, (uint32_t) (koff + pos * POS_B));
                lin(bv, (size_t) P * POS_B, (uint32_t) (voff + pos * POS_B));
                lin(bp, pad, 0);
                bk.iter_size = bv.iter_size = TASK;
                bk.iter_stride = bv.iter_stride = (uint32_t) ((size_t) ROUND * POS_B / 4);
                bk.next_bd = b0 + 1; bk.use_next = true;
                bv.next_bd = b0 + 2; bv.use_next = true;
                if (r == 0) {
                    bp.next_bd = b0 + 3;
                    bp.use_next = true;
                }
                xdna_seq_blockwrite(&seq, e.col, 0, b0, &bk);
                xdna_seq_ddr_patch(&seq, e.col, 0, b0, ARG_K, bk.buf_off);
                xdna_seq_blockwrite(&seq, e.col, 0, b0 + 1, &bv);
                xdna_seq_ddr_patch(&seq, e.col, 0, b0 + 1, argv, bv.buf_off);
                xdna_seq_blockwrite(&seq, e.col, 0, b0 + 2, &bp);
                xdna_seq_ddr_patch(&seq, e.col, 0, b0 + 2, ARG_IDS, 0);
            }
            xdna_seq_issue_token(&seq, e.col, 0, xdna_dma_dir::MM2S, e.ch, 0xF);
            xdna_seq_push_queue(&seq, e.col, 0, chain[c], xdna_dma_dir::MM2S, e.ch, true,
                                (uint32_t) (cnt - 1));
        }
    }
    for (int c = 0; c < XDNA_GEMV_COLS; c++) {
        xdna_seq_wait_token(&seq, l->w[c].col, 0, xdna_dma_dir::MM2S, l->w[c].ch);
    }
    xdna_seq_wait_token(&seq, SIDE_COL, 0, xdna_dma_dir::MM2S, SIDE_CH);
    for (int s = 0; s < l->n_o; s++) {
        xdna_seq_wait_token(&seq, l->o[s].col, 0, xdna_dma_dir::S2MM, l->o[s].ch);
    }

    // 3. combine: the partials to the prologue - every core's m and l, then
    //    every core's o - then attn_output on the tiles it builds, drained
    //    into the FFN's raw tiles
    {
        xdna_bd bm, bo;
        runs(bm, PIECE, CORES, ST, (uint32_t) R_PART);
        // o is 2048 words a core and a dimension's size is ten bits: four
        // runs of 512 a core, the cores one dimension further out
        bo.buf_len   = CORES * (ST - PIECE);
        bo.buf_off   = (uint32_t) (R_PART + (size_t) PIECE * sizeof(float));
        bo.d0_size   = SIDE_W;
        bo.d0_stride = 1;
        bo.d1_size   = (ST - PIECE) / SIDE_W;
        bo.d1_stride = SIDE_W;
        bo.d2_stride = ST;
        bo.ax_cache  = 2;
        bm.next_bd = SIDE_BD + 1;
        bm.use_next = true;
        xdna_seq_blockwrite(&seq, SIDE_COL, 0, SIDE_BD, &bm);
        xdna_seq_ddr_patch(&seq, SIDE_COL, 0, SIDE_BD, ARG_R, bm.buf_off);
        xdna_seq_blockwrite(&seq, SIDE_COL, 0, SIDE_BD + 1, &bo);
        xdna_seq_ddr_patch(&seq, SIDE_COL, 0, SIDE_BD + 1, ARG_R, bo.buf_off);
        xdna_seq_issue_token(&seq, SIDE_COL, 0, xdna_dma_dir::MM2S, SIDE_CH, 0xF);
        xdna_seq_push_queue(&seq, SIDE_COL, 0, SIDE_BD, xdna_dma_dir::MM2S, SIDE_CH, true, 0);
    }
    {
        xdna_gemv_seq_opts o;
        o.bd_base = 0;
        o.w_arg   = ARG_W;
        o.w_off   = (uint32_t) l->w_o;
        o.act_arg = ARG_ACT;
        o.act_off = (uint32_t) l->a_o;
        o.o_arg   = ARG_RES;
        o.out_off = XDNA_RES_S;
        o.skip_act = true;
        ok = ok && xdna_gemv_seq_build(&seq, so->geom, &o);
    }
    xdna_seq_wait_token(&seq, SIDE_COL, 0, xdna_dma_dir::MM2S, SIDE_CH);

    // 4. the FFN, its input the prologue's row: h_attn = h + S into A,
    //    rms_norm(h_attn) * gamma into the tiles; its output into F
    row_io(XDNA_RES_S, XDNA_RES_H, ARG_ACT, (uint32_t) l->a_gpost, XDNA_RES_A);
    {
        xdna_gemv_pair_opts po;
        po.w_arg   = ARG_W;
        po.w_base  = (uint32_t) l->w_ffn;
        po.a_arg   = ARG_FFN;
        po.o_arg   = ARG_RES;
        po.o_base  = XDNA_RES_F;
        po.bd_base = 0;
        po.act_replay = false;   // the row tiles are per chunk in DDR
        ok = ok && xdna_gemv_seq_build_pair(&seq, ffn->g1, ffn->g2, ffn->w2_off,
                                            ffn->a2_off, &po);
    }
    row_wait();
    if (!ok) {
        return {};
    }
    return xdna_seq_build(&seq);
}

} // namespace

bool xdna_att_layer_run(xdna_att_layer * l, const xdna_att_layer_in & in) {
    xrt::run * r = xdna_att_layer_start(l, in);
    if (!r || !xdna_run_wait(*r)) {
        GGML_LOG_ERROR("%s: the dispatch did not complete (%d positions)\n", "xdna-att-layer",
                       in.n_valid);
        return false;
    }
    xdna_att_layer_finish(in);
    return true;
}

void xdna_att_layer_finish(const xdna_att_layer_in & in) {
    const int row = in.n_valid - 1;
    xdna_buffer_sync_from_device_range(in.kbo, POS_B, in.koff + (size_t) row * POS_B);
    xdna_buffer_sync_from_device_range(in.vbo, POS_B, in.voff + (size_t) row * POS_B);
}

xrt::run * xdna_att_layer_start(xdna_att_layer * l, const xdna_att_layer_in & in) {
    if (!l || !in.res || !in.kbo || !in.vbo || !in.cosv || !in.sinv ||
        in.n_valid <= 0 || in.n_valid > xdna_att_layer_max_positions() ||
        in.n_rot <= 0 || in.n_rot > D || in.n_rot % 32 != 0) {
        return nullptr;
    }
    if (!xdna_att_layer_fits(in.kbo, in.koff, in.vbo, in.voff, in.n_valid)) {
        return nullptr;
    }
    const int n_chunks = (in.n_valid + ROUND - 1) / ROUND;
    const int n_task   = (n_chunks + TASK - 1) / TASK;
    const int row = in.n_valid - 1;
    const bool same = in.kbo == in.vbo;
    const std::vector<uint32_t> words = build(l, n_task, n_chunks, row, in.koff, in.voff, same);
    if (words.empty()) {
        GGML_LOG_ERROR("%s: the stream does not build\n", "xdna-att-layer");
        return nullptr;
    }
    xdna_att_layer::variant & v = l->var[n_task];
    if (!v.kern) {
        char name[48];
        snprintf(name, sizeof(name), "att_layer_%p_t%d", (void *) l, n_task);
        v.kern = xdna_kernel_pool_get_built(l->pool, name, l->rg->so->geom.stem().c_str(),
                                            words.data(), words.size());
        if (!v.kern) {
            return nullptr;
        }
    } else if (!xdna_kernel_rewrite_insts(v.kern, words.data(), words.size())) {
        return nullptr;
    }
    if (v.kbo != in.kbo || v.vbo != in.vbo || v.res != in.res) {
        xdna_buffer * args[8] = { in.kbo, l->act, l->ids, l->r, in.vbo, l->wbuf,
                                  xdna_gemv_pair_act_buf(l->rg->ffn), in.res };
        v.run = xdna_kernel_run_make(v.kern, args, 8);
        v.args.assign(args, args + 8);
        v.kbo = in.kbo;
        v.vbo = in.vbo;
        v.res = in.res;
    }
    // The two row inputs' tiles carry their norms' epsilon: written once.
    if (in.eps_attn != l->eps_attn || in.eps_post != l->eps_post) {
        xdna_gemv_pair * ffn = l->rg->ffn;
        if (!xdna_gemv_row_act(l->gqkv, (uint8_t *) l->act->bo.map() + l->a_qkv,
                               in.eps_attn, true, 0) ||
            !xdna_gemv_row_act(ffn->g1, (uint8_t *) ffn->a->bo.map(), in.eps_post, true,
                               xdna_gemv_pair_last_flags(ffn))) {
            return nullptr;
        }
        xdna_buffer_sync_to_device(ffn->a);
        l->eps_attn = in.eps_attn;
        l->eps_post = in.eps_post;
    }

    // the host's inputs
    uint8_t * a = (uint8_t *) l->act->bo.map();
    {
        std::memset(a + l->a_att, 0, 3 * ACTT);
        int32_t * hdr = (int32_t *) (a + l->a_att);
        hdr[2] = 1 + n_chunks;
        hdr[3] = 1;
        for (int t = 0; t < 2; t++) {
            int32_t * w = (int32_t *) (a + l->a_att + (size_t) (1 + t) * ACTT);
            std::memcpy(w, l->gq.data(), D * sizeof(float));
            w[512] = in.n_valid;
            w[513] = t;
            w[514] = 0;
            w[515] = in.n_rot;
            w[516] = fbits(in.scale);
            w[517] = fbits(in.eps);
            w[W_MODE] = 1;
            w[W_NS] = t == 0 ? 5 : 2;
            w[W_NE] = t == 0 ? 2 : 0;
            w[AW - 2] = F_ATT;
        }
        float * c = (float *) (a + l->a_const);
        std::memset(c, 0, SIDE_W * 4);
        std::memcpy(c, l->gk.data(), D * sizeof(float));
        std::memcpy(c + D, in.cosv, (size_t) (in.n_rot / 2) * sizeof(float));
        std::memcpy(c + D + 128, in.sinv, (size_t) (in.n_rot / 2) * sizeof(float));
    }
    xdna_buffer_sync_to_device(l->act);
    // Any line of the row the host holds would land on top of the array's
    // write: push them out first.
    xdna_buffer_sync_to_device_range(in.kbo, POS_B, in.koff + (size_t) row * POS_B);
    xdna_buffer_sync_to_device_range(in.vbo, POS_B, in.voff + (size_t) row * POS_B);

    if (!xdna_run_submit(v.kern, v.run, v.args.data(), v.args.size())) {
        return nullptr;
    }
    return &v.run;
}

