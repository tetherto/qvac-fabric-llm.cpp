#include <cstring>
#include "xdna-rec-gemv.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

// One dispatch: pack, then load the runner.
xdna_gemv * make_stage(xdna_kernel_pool * pool,
                       const struct ggml_tensor * const * ws, int n_w,
                       int64_t K, int64_t N, bool epilogue,
                       xdna_gemv_split split) {
    stage s;
    if (!pack_stage(ws, n_w, K, N, epilogue, split, s)) {
        return nullptr;
    }
    return xdna_gemv_create(pool, s.geom, s.packed);
}

// GGML_XDNA_GEMV_PAIR=0 keeps the FFN as two dispatches, which is what the
// pair is measured against on the same build.
bool xdna_rec_gemv_pair_enabled() {
    static const bool on = []() {
        const char * v = getenv("GGML_XDNA_GEMV_PAIR");
        return v == nullptr || atoi(v) != 0;
    }();
    return on;
}

bool xdna_rec_gemv_pair_verify() {
    static const bool on = []() {
        const char * v = getenv("GGML_XDNA_GEMV_PAIR_VERIFY");
        return v != nullptr && atoi(v) != 0;
    }();
    return on;
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

    const struct ggml_tensor * so_w[1] = { w_so };
    m->so = make_stage(pool, so_w, 1, w_so->ne[0], w_so->ne[1], false, split);

    const struct ggml_tensor * gu[2] = { w_gate, w_up };
    const struct ggml_tensor * dn[1] = { w_down };
    stage s_act, s_down;
    if (!pack_stage(gu, 2, w_gate->ne[0], w_gate->ne[1] + w_up->ne[1], true,
                    split, s_act) ||
        !pack_stage(dn, 1, w_down->ne[0], w_down->ne[1], false, split, s_down)) {
        xdna_rec_gemv_free(m.release());
        return nullptr;
    }

    // Both in one stream when the geometry allows it. Only the merged
    // artifact's split puts one core to a column, which is what lets a core's
    // output be one descriptor into the next dispatch's activation.
    if (fused && xdna_rec_gemv_pair_enabled()) {
        m->ffn = xdna_gemv_pair_create(pool, s_act.geom, s_act.packed,
                                       s_down.geom, s_down.packed);
    }
    // GGML_XDNA_GEMV_PAIR_VERIFY=1 keeps the two-dispatch path alongside the
    // pair and runs both, which is the only way to tell a bad handover from a
    // bad second dispatch.
    if (!m->ffn || xdna_rec_gemv_pair_verify()) {
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

bool xdna_rec_gemv_so_collect_raw(xdna_rec_gemv * m, const void * out,
                                  size_t off, float * dst) {
    if (!m || !m->so || !out || !dst) {
        return false;
    }
    std::memcpy(dst, (const uint8_t *) out + off,
                (size_t) m->so->geom.n_real * sizeof(float));
    return true;
}

bool xdna_rec_gemv_so_collect(xdna_rec_gemv * m, const void * out, size_t off,
                              const void * act, const float * hres,
                              float * h_attn) {
    if (!m || !m->so || !hres || !h_attn) {
        return false;
    }
    if (out) {
        // The fused stream drains into the core's output buffer, past what the
        // core writes: it has no argument of its own to drain to.
        // The gated stage writes the activation in the layout its build was
        // told (GATED_FMT); the projection's geometry derives from the model's
        // ssm_out type. If the two disagree the result is silent garbage, so
        // say it loudly.
        if (act) {
            const int32_t * h = (const int32_t *) act;
            const int afmt = h[XDNA_GEMV_ACT_TILE / 4 - 1];
            const int wfmt = m->so->geom.fmt == XDNA_WFMT_Q4G32 ? 0 : 1;
            if (afmt != wfmt) {
                fprintf(stderr, "xdna-rec-gemv: the artifact's activation "
                        "format (%d) does not match this model's ssm_out (%d); "
                        "rebuild the kernels with the matching GATED_FMT\n",
                        afmt, wfmt);
                return false;
            }
        }
        if (getenv("GGML_XDNA_SO_ACT_DUMP2")) {
            static int shown = 0;
            if (shown++ < 2) {
                const int32_t * h = (const int32_t *) act;
                fprintf(stderr, "so-act2: hdr nt=%d nout=%d fmt=%d flags=%d "
                        "codes[0..7] =",
                        h[0], h[1], h[2112 / 4 - 1], h[2112 / 4 - 2]);
                const int8_t * c = (const int8_t *) ((const uint8_t *) act + 2112);
                for (int i = 0; i < 8; i++) {
                    fprintf(stderr, " %d", (int) c[i]);
                }
                fprintf(stderr, "\n");
            }
        }
        std::memcpy(m->acc.data(), (const uint8_t *) out + off,
                    (size_t) m->so->geom.n_real * sizeof(float));
        // GGML_XDNA_SO_FUSED_VERIFY=1 runs the same projection the ordinary
        // way on the same activation and compares, which says whether the
        // fused half of the stream computed or only completed.
        if (act && getenv("GGML_XDNA_SO_FUSED_VERIFY")) {
            static int shown = 0;
            if (shown++ < 3) {
                std::vector<float> ref(m->acc.size(), 0.0f);
                if (xdna_gemv_run_packed(m->so, act, ref.data())) {
                    double num = 0, den = 0;
                    for (int i = 0; i < m->so->geom.n_real; i++) {
                        const double e = (double) m->acc[i] - ref[i];
                        num += e * e;
                        den += (double) ref[i] * ref[i];
                    }
                    fprintf(stderr, "so-fused: rel %.3e  fused %g %g %g | "
                            "ref %g %g %g\n",
                            den > 0 ? std::sqrt(num / den) : -1.0,
                            m->acc[0], m->acc[1], m->acc[2],
                            ref[0], ref[1], ref[2]);
                }
            }
        }
    } else if (!xdna_gemv_read_out(m->so, m->acc.data())) {
        return false;
    }
    const int n = m->so->geom.n_real;
    if (getenv("GGML_XDNA_ACT_DUMP")) {
        static int k = 0;
        if (k++ < 3) {
            fprintf(stderr, "xdna-acc(host): %g %g %g | t1 %g %g %g | "
                    "t2 %g %g %g | t3 %g %g %g\n",
                    m->acc[0], m->acc[1], m->acc[2],
                    m->acc[256], m->acc[257], m->acc[258],
                    m->acc[512], m->acc[513], m->acc[514],
                    m->acc[768], m->acc[769], m->acc[770]);
        }
    }
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
    // GGML_XDNA_SO_ACT_DUMP=1 decodes the tiles the gated stage wrote and
    // compares them against the codes it also wrote in the row-scale form.
    if (act_tiles && aq && getenv("GGML_XDNA_SO_ACT_DUMP")) {
        static int shown = 0;
        if (shown++ < 2) {
            const uint8_t * a8 = (const uint8_t *) act_tiles;
            const int32_t * h = (const int32_t *) a8;
            fprintf(stderr, "so-act: hdr nt=%d nout=%d fmt=%d flags=%d\n",
                    h[0], h[1], h[2112 / 4 - 1], h[2112 / 4 - 2]);
            double num = 0, den = 0;
            for (int i = 0; i < (int) m->a.size(); i++) {
                const int t = i / 256, g = (i % 256) / 32;
                const uint8_t * tile = a8 + (size_t) (1 + t) * 2112;
                const float d = ((const float *) (tile + 256))[8 + g];
                const float got = (float) ((const int8_t *) tile)[i % 256] * d;
                const float ref = (float) aq[i] * d_a;
                num += (double) (got - ref) * (got - ref);
                den += (double) ref * ref;
            }
            int i0 = 0;
            while (i0 < (int) m->a.size() && aq[i0] == 0) {
                i0++;
            }
            const int t0 = i0 / 256, g0 = (i0 % 256) / 32;
            const uint8_t * tl = a8 + (size_t) (1 + t0) * 2112;
            fprintf(stderr, "so-act: rel %.3e den %.3e  first nonzero i=%d "
                    "dev code %d scale %g -> %g | ref code %d d_a %g -> %g\n",
                    den > 0 ? std::sqrt(num / den) : -1.0, den, i0,
                    (int) ((const int8_t *) tl)[i0 % 256],
                    ((const float *) (tl + 256))[8 + g0],
                    (float) ((const int8_t *) tl)[i0 % 256] *
                        ((const float *) (tl + 256))[8 + g0],
                    (int) aq[i0], d_a, aq[i0] * d_a);
        }
    }
    static const bool packed_on = []() {
        const char * v = getenv("GGML_XDNA_SO_PACKED");
        return v == nullptr || atoi(v) != 0;
    }();
    if (packed_on && act_tiles && m->so->geom.fmt == XDNA_WFMT_Q4G32) {
        if (!xdna_gemv_run_packed(m->so, act_tiles, m->acc.data())) {
            return false;
        }
        if (aq && getenv("GGML_XDNA_SO_ACT_DUMP")) {
            static int shown2 = 0;
            if (shown2++ < 3) {
                std::vector<float> ref(m->acc.size(), 0.0f);
                const size_t K = m->a.size();
                for (size_t i = 0; i < K; i++) {
                    m->a[i] = (float) aq[i] * d_a;
                }
                xdna_gemv_run(m->so, m->a.data(), ref.data());
                double num = 0, den = 0;
                for (int i = 0; i < m->so->geom.n_real; i++) {
                    const double e = (double) m->acc[i] - ref[i];
                    num += e * e;
                    den += (double) ref[i] * ref[i];
                }
                fprintf(stderr, "so-out: rel %.3e  packed %g %g %g | host %g %g %g\n",
                        den > 0 ? std::sqrt(num / den) : -1.0,
                        m->acc[0], m->acc[1], m->acc[2], ref[0], ref[1], ref[2]);
            }
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
    if (getenv("GGML_XDNA_ACT_DUMP")) {
        static int n = 0;
        if (n++ < 3) {
            fprintf(stderr, "xdna-act: kt=%d nt=%d tiles", kt, nt);
            for (int t = 0; t < nt; t++) {
                const float * f = (const float *)
                    (base + (size_t) (t + 1) * XDNA_GEMV_ACT_TILE);
                fprintf(stderr, " | t%d %g %g %g", t, f[0], f[1], f[2]);
            }
            fprintf(stderr, "\n");
        }
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
    if (m->ffn && xdna_rec_gemv_pair_verify()) {
        std::vector<float> ref(m->acc.size(), 0.0f);
        if (!xdna_gemv_run(m->act, hff, m->mid.data()) ||
            !xdna_gemv_run(m->down, m->mid.data(), ref.data()) ||
            !xdna_gemv_pair_run(m->ffn, hff, m->acc.data())) {
            return false;
        }
        std::vector<float> mid_dev;
        xdna_gemv_pair_read_mid(m->ffn, mid_dev);
        static int shown = 0;
        if (shown++ < 4) {
            const auto rel = [](const std::vector<float> & a,
                                const std::vector<float> & b, int n) {
                double num = 0, den = 0;
                for (int i = 0; i < n; i++) {
                    const double d = (double) a[i] - b[i];
                    num += d * d;
                    den += (double) b[i] * b[i];
                }
                return den > 0 ? std::sqrt(num / den) : 0.0;
            };
            fprintf(stderr, "xdna-gemv-pair: mid rel %.3e (dev %g %g %g | host %g %g %g), "
                    "out rel %.3e (pair %g %g %g | ref %g %g %g)\n",
                    rel(mid_dev, m->mid, (int) m->mid.size()),
                    mid_dev[0], mid_dev[1], mid_dev[2],
                    m->mid[0], m->mid[1], m->mid[2],
                    rel(m->acc, ref, m->n_out),
                    m->acc[0], m->acc[1], m->acc[2],
                    ref[0], ref[1], ref[2]);
        }
    } else if (m->ffn) {
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
