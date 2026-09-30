#include "xdna-rec-gemv.h"

#include "ggml-impl.h"
#include "ggml.h"
#include "xdna-util.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>

namespace {

struct stage {
    xdna_gemv_geom       geom;
    std::vector<uint8_t> packed;
};

// Derive the geometry and pack the weights into it. `ws` are concatenated
// along N in the order given.
bool pack_stage(const struct ggml_tensor * const * ws,
                int                                n_w,
                int64_t                            K,
                int64_t                            N,
                bool                               epilogue,
                xdna_gemv_split                    split,
                stage &                            out) {
    if (!ws || n_w <= 0 || !ws[0]) {
        return false;
    }
    out.geom = xdna_gemv_variant(xdna_gemv_type_of(ws[0]), K, N, epilogue, split);
    if (!out.geom.valid()) {
        return false;
    }
    std::vector<int32_t> colmap;
    if (epilogue) {
        // A core has to own the gate half and the up half of the same values,
        // which is not the order the two weights concatenate in.
        xdna_gemv_colmap(out.geom, colmap);
    }
    return xdna_gemv_pack_weights(out.geom, ws, n_w, colmap.empty() ? nullptr : colmap.data(), out.packed);
}

}  // namespace

bool xdna_rec_inproj_pack(const struct ggml_tensor * w_qkv, const struct ggml_tensor * w_z, xdna_rec_inproj & out) {
    if (!w_qkv || !w_z || w_qkv->ne[0] != w_z->ne[0]) {
        return false;
    }
    const int64_t        K   = w_qkv->ne[0];
    const int            n_q = (int) w_qkv->ne[1];
    const int            n_z = (int) w_z->ne[1];
    // The group decides the one format both take: q8g16 holds either, so a
    // group whose members differ packs as the wider one.
    xdna_gemv_geom       g   = xdna_gemv_variant(xdna_gemv_type_of(w_qkv), K, n_q + n_z, false, XDNA_GEMV_SPLIT_FUSED);
    const xdna_gemv_geom gz  = xdna_gemv_variant(xdna_gemv_type_of(w_z), K, n_q + n_z, false, XDNA_GEMV_SPLIT_FUSED);
    if (!g.valid() || !gz.valid()) {
        return false;
    }
    if (gz.fmt == XDNA_WFMT_Q8G16) {
        g = gz;
    }
    const int n_o = xdna_gemv_out_streams(g);
    const int sw  = xdna_gemv_stream_floats(g);
    const int n_c = g.n_out();
    // Exactly: every stream but the last full of qkv, the last full of z.
    if (n_o < 2 || g.N != n_q + n_z || g.chunk() != n_o * sw || n_q != (n_o - 1) * sw * n_c || n_z != sw * n_c) {
        return false;
    }
    std::vector<int32_t> colmap((size_t) g.N);
    for (int j = 0; j < n_c; j++) {
        for (int c = 0; c < n_o; c++) {
            for (int r = 0; r < sw; r++) {
                const int n        = j * g.chunk() + c * sw + r;
                colmap[(size_t) n] = c < n_o - 1 ? (j * (n_o - 1) + c) * sw + r : n_q + j * sw + r;
            }
        }
    }
    const struct ggml_tensor * ws[2] = { w_qkv, w_z };
    out.geom                         = g;
    out.n_qkv                        = n_q;
    out.n_z                          = n_z;
    return xdna_gemv_pack_weights(g, ws, 2, colmap.data(), out.packed);
}

xdna_rec_gemv * xdna_rec_gemv_create(xdna_kernel_pool *         pool,
                                     const struct ggml_tensor * w_so,
                                     const struct ggml_tensor * w_gate,
                                     const struct ggml_tensor * w_up,
                                     const struct ggml_tensor * w_down) {
    if (!pool || !w_so || !w_gate || !w_up || !w_down) {
        return nullptr;
    }
    if (w_gate->ne[0] != w_up->ne[0] || w_gate->ne[1] != w_up->ne[1] || w_gate->type != w_up->type) {
        return nullptr;  // the epilogue pairs the two halves by position
    }

    // The projections run on the merged layer artifact, the same one the core
    // is on, so a layer never changes hardware context.
    constexpr xdna_gemv_split split = XDNA_GEMV_SPLIT_FUSED;

    std::unique_ptr<xdna_rec_gemv> m(new xdna_rec_gemv);

    const struct ggml_tensor * gu[2] = { w_gate, w_up };
    const struct ggml_tensor * dn[1] = { w_down };
    stage                      s_so, s_act, s_down;
    if (!pack_stage(&w_so, 1, w_so->ne[0], w_so->ne[1], false, split, s_so)) {
        return nullptr;
    }
    if (!pack_stage(gu, 2, w_gate->ne[0], w_gate->ne[1] + w_up->ne[1], true, split, s_act) ||
        !pack_stage(dn, 1, w_down->ne[0], w_down->ne[1], false, split, s_down)) {
        return nullptr;
    }
    m->so = xdna_gemv_create(pool, s_so.geom, s_so.packed);

    // Both FFN stages in one stream when the geometry allows it. The fused
    // split puts one core to a column, which is what lets a core's output be
    // one descriptor into the next dispatch's activation.
    m->ffn = xdna_gemv_pair_create(pool, s_act.geom, s_act.packed, s_down.geom, s_down.packed);
    if (!m->ffn) {
        m->act  = xdna_gemv_create(pool, s_act.geom, s_act.packed);
        m->down = xdna_gemv_create(pool, s_down.geom, s_down.packed);
    }

    if (!m->so || (!m->ffn && (!m->act || !m->down))) {
        xdna_rec_gemv_free(m.release());
        return nullptr;
    }

    m->n_out = s_down.geom.n_real;
    m->a.resize((size_t) w_so->ne[0]);
    m->mid.resize((size_t) s_act.geom.n_real);
    m->acc.resize((size_t) std::max(m->so->geom.n_real, m->n_out));
    return m.release();
}

void xdna_rec_gemv_free(xdna_rec_gemv * m) {
    if (!m) {
        return;
    }
    xdna_gemv_free(m->so);
    xdna_gemv_pair_free(m->ffn);
    xdna_gemv_free(m->act);
    xdna_gemv_free(m->down);
    delete m;
}

xdna_gemv * xdna_rec_gemv_so(xdna_rec_gemv * m) {
    return m ? m->so : nullptr;
}

bool xdna_rec_gemv_so_collect(xdna_rec_gemv * m,
                              const void *    out,
                              size_t          off,
                              const void *    act,
                              const float *   hres,
                              float *         h_attn) {
    if (!m || !m->so || !hres || !h_attn) {
        return false;
    }
    if (out) {
        // The fused stream drains into the core's output buffer, past what the
        // core writes: it has no argument of its own to drain to. The gated
        // stage writes the activation in the layout its build was told
        // (GATED_FMT); the projection's geometry derives from the model's
        // ssm_out type. If the two disagree the result is silent garbage.
        if (act) {
            const int32_t * h    = (const int32_t *) act;
            const int       afmt = h[XDNA_GEMV_ACT_TILE / 4 - 1];
            const int       wfmt = m->so->geom.fmt == XDNA_WFMT_Q4G32 ? 0 : 1;
            if (afmt != wfmt) {
                GGML_LOG_ERROR(
                    "%s: the artifact's activation format (%d) does not match "
                    "this model's ssm_out (%d); rebuild the kernels with the "
                    "matching GATED_FMT\n",
                    "xdna-rec-gemv", afmt, wfmt);
                return false;
            }
        }
        std::memcpy(m->acc.data(), (const uint8_t *) out + off, (size_t) m->so->geom.n_real * sizeof(float));
    } else if (!xdna_gemv_read_out(m->so, m->acc.data())) {
        return false;
    }
    const int n = m->so->geom.n_real;
    for (int i = 0; i < n; i++) {
        h_attn[i] = hres[i] + m->acc[i];
    }
    return true;
}

bool xdna_rec_gemv_so_run(xdna_rec_gemv * m,
                          const void *    act_tiles,
                          const int8_t *  aq,
                          float           d_a,
                          const float *   hres,
                          float *         h_attn) {
    if (!m || !hres || !h_attn) {
        return false;
    }
    // The gated stage writes this activation itself, in the layout the GEMV
    // reads, so the common path neither dequantizes nor repacks. The fallback
    // is for the 8-bit weight form, whose groups are 16 wide where the gated
    // kernel writes 32.
    if (act_tiles && m->so->geom.fmt == XDNA_WFMT_Q4G32) {
        if (!xdna_gemv_run_packed(m->so, act_tiles, m->acc.data())) {
            return false;
        }
    } else {
        if (!aq) {
            return false;
        }
        const size_t K = m->a.size();
        for (size_t i = 0; i < K; i++) {
            m->a[i] = (float) aq[i] * d_a;
        }
        if (!xdna_gemv_run(m->so, m->a.data(), m->acc.data())) {
            return false;
        }
    }
    const int n = m->so->geom.n_real;
    for (int i = 0; i < n; i++) {
        h_attn[i] = hres[i] + m->acc[i];
    }
    return true;
}

// The projection's result, gathered out of the activation tiles the array
// drained it into: one stream to a tile, K_TILE floats at the front of each.
// Only the layer's last residual add needs it, and that is after both
// dispatches - not between them.
bool xdna_rec_gemv_acc_from_tiles(xdna_rec_gemv * m, float * acc) {
    if (!m || !m->ffn || !acc) {
        return false;
    }
    xdna_buffer * a = xdna_gemv_pair_act_buf(m->ffn);
    if (!a) {
        return false;
    }
    const int kt = xdna_gemv_pair_k_tile(m->ffn);
    const int nt = xdna_gemv_pair_n_tiles(m->ffn);
    m->settled.resize(a->bytes);
    if (!xdna_buffer_read_settled(a, m->settled.data(), a->bytes)) {
        return false;
    }
    for (int t = 0; t < nt; t++) {
        if (!xdna_buffer_download(a, acc + (size_t) t * kt, (size_t) kt * sizeof(float),
                                  (size_t) (t + 1) * XDNA_GEMV_ACT_TILE)) {
            return false;
        }
    }
    return true;
}

bool xdna_rec_gemv_fused_results(xdna_rec_gemv * m, float * acc, float * ffn_out) {
    if (!m || !m->ffn || !acc || !ffn_out) {
        return false;
    }
    xdna_gemv_pair * p    = m->ffn;
    const size_t     tail = p->o_tail_off + (size_t) p->g2.n_real * sizeof(float);
    if (!xdna_rec_gemv_acc_from_tiles(m, acc) || m->settled.size() < tail) {
        return false;
    }
    std::memcpy(ffn_out, m->settled.data() + p->o_tail_off, (size_t) p->g2.n_real * sizeof(float));
    return true;
}

bool xdna_rec_gemv_ffn_run_raw(xdna_rec_gemv * m,
                               const float *   acc,
                               const float *   hres,
                               const float *   gamma,
                               const float *   h_attn,
                               float *         h_out) {
    // A null acc is legal: the projection drained it into the tiles itself.
    if (!m || !m->ffn || !hres || !gamma || !h_attn || !h_out) {
        return false;
    }
    if (!xdna_gemv_pair_run_raw(m->ffn, acc, hres, gamma, m->acc.data())) {
        return false;
    }
    for (int i = 0; i < m->n_out; i++) {
        h_out[i] = h_attn[i] + m->acc[i];
    }
    return true;
}

bool xdna_rec_gemv_ffn_run(xdna_rec_gemv * m, const float * hff, const float * h_attn, float * h_out) {
    if (!m || !hff || !h_attn || !h_out) {
        return false;
    }
    // One dispatch for gate and up together, returning silu(gate)*up: the two
    // projections share an activation, so they share a launch, and the
    // nonlinearity closes on the cores. With the pair the down projection is
    // in that same stream and its activation never leaves the device.
    if (m->ffn) {
        if (!xdna_gemv_pair_run(m->ffn, hff, m->acc.data())) {
            return false;
        }
    } else {
        if (!xdna_gemv_run(m->act, hff, m->mid.data())) {
            return false;
        }
        if (!xdna_gemv_run(m->down, m->mid.data(), m->acc.data())) {
            return false;
        }
    }
    const int n = m->n_out;
    for (int i = 0; i < n; i++) {
        h_out[i] = h_attn[i] + m->acc[i];
    }
    return true;
}

// --- attention tail --------------------------------------------------------

xdna_rec_tail * xdna_rec_tail_create(xdna_kernel_pool *         pool,
                                     const struct ggml_tensor * w_o,
                                     const struct ggml_tensor * w_gate,
                                     const struct ggml_tensor * w_up,
                                     const struct ggml_tensor * w_down,
                                     xdna_buffer *              res,
                                     const float *              gamma,
                                     float                      eps) {
    if (!res || !gamma) {
        return nullptr;
    }
    std::unique_ptr<xdna_rec_tail> t(new xdna_rec_tail);
    t->gv = xdna_rec_gemv_create(pool, w_o, w_gate, w_up, w_down);
    if (!t->gv || !t->gv->ffn || !xdna_gemv_pair_raw_act(t->gv->ffn)) {
        xdna_rec_tail_free(t.release());
        return nullptr;
    }
    xdna_gemv *      so        = t->gv->so;
    xdna_gemv_pair * ffn       = t->gv->ffn;
    const size_t     ffn_w_off = xdna_align_up(so->geom.weight_bytes());
    const size_t     ffn_wb    = ffn->w2_off + ffn->g2.weight_bytes();

    // [attn_output -> S | the prologue's row: h_attn = H + S into A, the norm
    // into the FFN's tiles | gate+up -> down into F]: arguments 0 (weights),
    // 1 (the projection's activation, then gamma), 2 (the FFN's activation
    // buffer) and 3 (the residual rows).
    const size_t       g_off = xdna_align_up(so->geom.act_bytes());
    xdna_seq           seq;
    xdna_gemv_seq_opts o;
    o.bd_base = 0;
    o.w_arg   = 0;
    o.act_arg = 1;
    o.o_arg   = 3;
    o.out_off = XDNA_RES_S;
    xdna_gemv_pair_opts po;
    po.w_arg      = 0;
    po.w_base     = (uint32_t) ffn_w_off;
    po.a_arg      = 2;
    po.o_arg      = 3;
    po.o_base     = XDNA_RES_F;
    po.bd_base    = 0;
    po.act_replay = false;
    bool ok       = xdna_gemv_seq_build(&seq, so->geom, &o);
    xdna_seq_row_io(&seq, 3, XDNA_RES_S, XDNA_RES_H, 1, (uint32_t) g_off, XDNA_RES_A);
    ok = ok && xdna_gemv_seq_build_pair(&seq, ffn->g1, ffn->g2, ffn->w2_off, ffn->a2_off, &po);
    xdna_seq_row_wait(&seq);
    if (!ok) {
        GGML_LOG_ERROR("%s: attention tail: stream build failed\n", "xdna-rec-gemv");
        xdna_rec_tail_free(t.release());
        return nullptr;
    }
    const std::vector<uint32_t> words = xdna_seq_build(&seq);
    // Pooled by content: layers whose down projection packs differently build
    // different streams.
    uint64_t                    h     = 1469598103934665603ull;
    for (uint32_t v : words) {
        h = (h ^ v) * 1099511628211ull;
    }
    char kname[48];
    snprintf(kname, sizeof(kname), "rec_tail_%016llx", (unsigned long long) h);
    t->kern = xdna_kernel_pool_get_built(pool, kname, so->geom.stem().c_str(), words.data(), words.size());
    t->w    = xdna_buffer_alloc(so->dev, ffn_w_off + ffn_wb);
    t->a    = xdna_buffer_alloc(so->dev, g_off + (size_t) XDNA_RES_D * sizeof(float));
    if (!t->kern || !t->w || !t->a) {
        xdna_rec_tail_free(t.release());
        return nullptr;
    }
    uint8_t * w = (uint8_t *) t->w->data;
    std::memset(w, 0, ffn_w_off + ffn_wb);
    std::memcpy(w, so->w->data, so->geom.weight_bytes());
    std::memcpy(w + ffn_w_off, ffn->w->data, ffn_wb);
    if (!xdna_buffer_sync_to_device(t->w)) {
        xdna_rec_tail_free(t.release());
        return nullptr;
    }
    t->host_a.resize(so->geom.act_bytes());
    std::memset(t->a->data, 0, g_off + (size_t) XDNA_RES_D * sizeof(float));
    std::memcpy((uint8_t *) t->a->data + g_off, gamma, (size_t) XDNA_RES_D * sizeof(float));
    if (!xdna_buffer_sync_to_device(t->a)) {
        xdna_rec_tail_free(t.release());
        return nullptr;
    }
    if (!xdna_gemv_row_act(ffn->g1, (uint8_t *) ffn->a->data, eps, true, xdna_gemv_pair_last_flags(ffn))) {
        xdna_rec_tail_free(t.release());
        return nullptr;
    }
    if (!xdna_buffer_sync_to_device(ffn->a)) {
        xdna_rec_tail_free(t.release());
        return nullptr;
    }
    t->res                = res;
    xdna_buffer * args[4] = { t->w, t->a, xdna_gemv_pair_act_buf(ffn), res };
    t->run                = xdna_kernel_run_make(t->kern, args, 4);
    return t.release();
}

void xdna_rec_tail_free(xdna_rec_tail * t) {
    if (!t) {
        return;
    }
    xdna_buffer_free(t->w);
    xdna_buffer_free(t->a);
    xdna_rec_gemv_free(t->gv);
    delete t;
}

bool xdna_rec_tail_run(xdna_rec_tail * t, const float * act, const float * hres, float * h_attn, float * h_out) {
    if (!t || !act || !hres || !h_attn || !h_out) {
        return false;
    }
    xdna_rec_gemv * m = t->gv;
    xdna_gemv_pack_act_into(m->so->geom, act, t->host_a.data());
    std::memcpy(t->a->data, t->host_a.data(), t->host_a.size());
    if (!xdna_buffer_sync_to_device_range(t->a, t->host_a.size(), 0)) {
        return false;
    }
    // the residual as the row the FFN's boundary reads
    std::memcpy((uint8_t *) t->res->data + XDNA_RES_H, hres, (size_t) XDNA_RES_D * sizeof(float));
    if (!xdna_buffer_sync_to_device_range(t->res, (size_t) XDNA_RES_D * sizeof(float), XDNA_RES_H)) {
        return false;
    }
    if (!xdna_run_restart(t->run) || !xdna_run_wait(t->run)) {
        return false;
    }
    std::vector<float> r2((size_t) 2 * XDNA_RES_D);
    if (!xdna_buffer_read_settled(t->res, r2.data(), r2.size() * sizeof(float), XDNA_RES_A)) {
        return false;
    }
    for (int i = 0; i < m->n_out; i++) {
        h_attn[i] = r2[(size_t) i];
        h_out[i]  = r2[(size_t) i] + r2[(size_t) XDNA_RES_D + (size_t) i];
    }
    return true;
}
