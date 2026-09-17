#include "xdna-rec-gemv.h"

#include "ggml-impl.h"
#include "ggml.h"

#include <algorithm>
#include <cstring>
#include <memory>

namespace {

struct stage {
    xdna_gemv_geom       geom;
    std::vector<uint8_t> packed;
};

// Derive the geometry and pack the weights into it. `ws` are concatenated
// along N in the order given.
bool pack_stage(const struct ggml_tensor * const * ws, int n_w,
                int64_t K, int64_t N, bool epilogue,
                xdna_gemv_split split, stage & out) {
    out.geom = xdna_gemv_variant(ws[0]->type, K, N, epilogue, split);
    if (!out.geom.valid()) {
        return false;
    }
    std::vector<int32_t> colmap;
    if (epilogue) {
        // A core has to own the gate half and the up half of the same values,
        // which is not the order the two weights concatenate in.
        xdna_gemv_colmap(out.geom, colmap);
    }
    return xdna_gemv_pack_weights(out.geom, ws, n_w,
                                  colmap.empty() ? nullptr : colmap.data(),
                                  out.packed);
}

} // namespace

xdna_rec_gemv * xdna_rec_gemv_create(xdna_kernel_pool * pool,
                                     const struct ggml_tensor * w_so,
                                     const struct ggml_tensor * w_gate,
                                     const struct ggml_tensor * w_up,
                                     const struct ggml_tensor * w_down,
                                     bool fused) {
    if (!pool || !w_so || !w_gate || !w_up || !w_down) {
        return nullptr;
    }
    if (w_gate->ne[0] != w_up->ne[0] || w_gate->ne[1] != w_up->ne[1] ||
        w_gate->type != w_up->type) {
        return nullptr;   // the epilogue pairs the two halves by position
    }

    // On the merged artifact the projections share the core's hardware
    // context; otherwise the half-height split, which is half the array and
    // therefore half as expensive to arrive at.
    const xdna_gemv_split split = fused ? XDNA_GEMV_SPLIT_FUSED
                                        : XDNA_GEMV_SPLIT_HALF;

    std::unique_ptr<xdna_rec_gemv> m(new xdna_rec_gemv);

    const struct ggml_tensor * gu[2] = { w_gate, w_up };
    const struct ggml_tensor * dn[1] = { w_down };
    stage s_so, s_act, s_down;
    if (!pack_stage(&w_so, 1, w_so->ne[0], w_so->ne[1], false, split, s_so)) {
        delete m.release();
        return nullptr;
    }
    if (!pack_stage(gu, 2, w_gate->ne[0], w_gate->ne[1] + w_up->ne[1], true,
                    split, s_act) ||
        !pack_stage(dn, 1, w_down->ne[0], w_down->ne[1], false, split, s_down)) {
        delete m.release();
        return nullptr;
    }
    m->so = xdna_gemv_create(pool, s_so.geom, s_so.packed);

    // Both FFN stages in one stream when the geometry allows it. Only the
    // merged artifact's split puts one core to a column, which is what lets a
    // core's output be one descriptor into the next dispatch's activation.
    if (fused) {
        m->ffn = xdna_gemv_pair_create(pool, s_act.geom, s_act.packed,
                                       s_down.geom, s_down.packed);
    }
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

bool xdna_rec_gemv_so_collect(xdna_rec_gemv * m, const void * out, size_t off,
                              const void * act, const float * hres,
                              float * h_attn) {
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
            const int32_t * h = (const int32_t *) act;
            const int afmt = h[XDNA_GEMV_ACT_TILE / 4 - 1];
            const int wfmt = m->so->geom.fmt == XDNA_WFMT_Q4G32 ? 0 : 1;
            if (afmt != wfmt) {
                GGML_LOG_ERROR(
                    "%s: the artifact's activation format (%d) does not match "
                    "this model's ssm_out (%d); rebuild the kernels with the "
                    "matching GATED_FMT\n", "xdna-rec-gemv", afmt, wfmt);
                return false;
            }
        }
        std::memcpy(m->acc.data(), (const uint8_t *) out + off,
                    (size_t) m->so->geom.n_real * sizeof(float));
    } else if (!xdna_gemv_read_out(m->so, m->acc.data())) {
        return false;
    }
    const int n = m->so->geom.n_real;
    for (int i = 0; i < n; i++) {
        h_attn[i] = hres[i] + m->acc[i];
    }
    return true;
}

bool xdna_rec_gemv_so_run(xdna_rec_gemv * m, const void * act_tiles,
                          const int8_t * aq, float d_a,
                          const float * hres, float * h_attn) {
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
    xdna_buffer_sync_from_device(a);
    const uint8_t * base = (const uint8_t *) a->bo.map();
    const int kt = xdna_gemv_pair_k_tile(m->ffn);
    const int nt = xdna_gemv_pair_n_tiles(m->ffn);
    for (int t = 0; t < nt; t++) {
        std::memcpy(acc + (size_t) t * kt,
                    base + (size_t) (t + 1) * XDNA_GEMV_ACT_TILE,
                    (size_t) kt * sizeof(float));
    }
    return true;
}

bool xdna_rec_gemv_ffn_run_raw(xdna_rec_gemv * m, const float * acc,
                               const float * hres, const float * gamma,
                               const float * h_attn, float * h_out) {
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

bool xdna_rec_gemv_ffn_run(xdna_rec_gemv * m, const float * hff,
                           const float * h_attn, float * h_out) {
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
