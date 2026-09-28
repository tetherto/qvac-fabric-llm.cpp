#include "xdna-head.h"

#include "ggml-impl.h"
#include "ggml.h"
#include "xdna-gemv.h"
#include "xdna-prof.h"
#include "xdna-util.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace {

// One phase of the head: a run of the weight's rows in one format.
struct part {
    xdna_gemv_geom geom;
    xdna_gemv_geom body;        // one chunk of it: what its activation packs as
    size_t         w_off = 0, a_off = 0, o_off = 0;
    size_t         ra_off = 0;  // its row-mode tiles, every chunk's (xdna_gemv_row_act)
    int            row0   = 0;  // the first of its rows
};

// Rows [r0, r1) of `w` as Q4_K, re-quantized from their dequantized values
// over the host's threads.
std::vector<uint8_t> requant_q4k(const ggml_tensor * w, int64_t r0, int64_t r1) {
    const int64_t            K    = w->ne[0];
    const size_t             row4 = ggml_row_size(GGML_TYPE_Q4_K, K);
    const ggml_type_traits * tt   = ggml_get_type_traits(w->type);
    std::vector<uint8_t>     q4((size_t) (r1 - r0) * row4);
    const int                nt = (int) std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    th.reserve((size_t) nt);
    for (int t = 0; t < nt; t++) {
        th.emplace_back([&, t]() {
            const int64_t      a = r0 + (r1 - r0) * t / nt, b = r0 + (r1 - r0) * (t + 1) / nt;
            std::vector<float> f((size_t) 256 * K);
            for (int64_t r = a; r < b; r += 256) {
                const int64_t n = std::min<int64_t>(256, b - r);
                for (int64_t i = 0; i < n; i++) {
                    tt->to_float((const char *) w->data + (r + i) * w->nb[1], f.data() + i * K, K);
                }
                ggml_quantize_chunk(GGML_TYPE_Q4_K, f.data(), q4.data() + (size_t) (r - r0) * row4, 0, n, K, nullptr);
            }
        });
    }
    for (auto & x : th) {
        x.join();
    }
    return q4;
}

// A view of n rows of a weight, as `type` with data at `data`.
ggml_tensor rows_of(const ggml_tensor * w, ggml_type type, const void * data, int64_t n) {
    ggml_tensor t = *w;
    t.type        = type;
    t.ne[1]       = n;
    t.nb[0]       = ggml_type_size(type);
    t.nb[1]       = ggml_row_size(type, w->ne[0]);
    t.nb[2] = t.nb[3] = t.nb[1] * (size_t) n;
    t.data            = const_cast<void *>(data);
    return t;
}

}  // namespace

struct xdna_head {
    part                       p[2];
    int                        n_parts = 0;
    int                        n_real  = 0;
    xdna_kernel *              kern    = nullptr;
    xdna_buffer *              w = nullptr, *a = nullptr, *o = nullptr;
    xrt::run                   run;
    // the input from the rows: its own stream, its tiles and gamma in ra
    xdna_kernel *              kern_r = nullptr;
    xdna_buffer *              ra     = nullptr;
    xrt::run                   run_r;
    std::vector<xdna_buffer *> args_r;  // run_r's, for a joined token stream
};

xdna_head * xdna_head_create(xdna_kernel_pool *  pool,
                             const ggml_tensor * w,
                             xdna_buffer *       res,
                             const float *       gamma,
                             float               eps) {
    if (!pool || !w || !w->data || w->ne[2] != 1 || w->ne[3] != 1) {
        return nullptr;
    }
    const int64_t            K = w->ne[0], N = w->ne[1];
    const ggml_type_traits * tt = ggml_get_type_traits(w->type);
    if (K % 256 || (w->type != GGML_TYPE_Q4_K && !tt->to_float)) {
        return nullptr;
    }
    const auto           t0        = std::chrono::steady_clock::now();
    // The first rows exact: a byte-pair vocabulary's low ids are its most
    // frequent tokens, the ones the distribution's mass sits on. Measured
    // (KLD against the Q6_K head on the CPU, 0.0049): every row 4-bit 0.0165;
    // the first 16K exact 0.0072, 32K 0.0061, 64K 0.0052.
    // GGML_XDNA_HEAD_EXACT_ROWS, rounded down to a pass of the pool; 0 for all
    // 4-bit. strtoll stops at the first non-digit like atoll and saturates on
    // an out-of-range value.
    static const int64_t exact_env = [] {
        const char * e = getenv("GGML_XDNA_HEAD_EXACT_ROWS");
        return e ? (int64_t) strtoll(e, nullptr, 10) : (int64_t) 65536;
    }();
    const xdna_gemv_geom g8 =
        xdna_gemv_variant(w->type, K, std::max<int64_t>(exact_env, 1), false, XDNA_GEMV_SPLIT_FUSED, true);
    int64_t R = w->type == GGML_TYPE_Q4_K || !g8.valid() ? 0 : std::min(exact_env, N);
    R         = g8.valid() ? R / g8.chunk() * g8.chunk() : 0;

    xdna_head * h = new xdna_head;
    h->n_real     = (int) N;
    std::vector<std::vector<uint8_t>> packed;
    size_t                            w_bytes = 0, a_bytes = 0, o_bytes = 0;
    const auto                        add = [&](const xdna_gemv_geom & g, const ggml_tensor & t, int row0) {
        part & p      = h->p[h->n_parts++];
        p.geom        = g;
        p.body        = g;
        p.body.N      = g.chunk();
        p.body.n_real = g.chunk();
        p.row0        = row0;
        p.w_off       = w_bytes;
        p.a_off       = a_bytes;
        p.o_off       = o_bytes;
        packed.emplace_back();
        const ggml_tensor * ws[1] = { &t };
        if (!g.valid() || !xdna_gemv_pack_weights(g, ws, 1, nullptr, packed.back())) {
            return false;
        }
        w_bytes = xdna_align_up(w_bytes + g.weight_bytes());
        a_bytes = xdna_align_up(a_bytes + p.body.act_bytes());
        o_bytes += (size_t) g.N * sizeof(float);
        return true;
    };
    bool ok = true;
    if (R > 0) {
        // the source rows as they are: q8g16 holds Q6_K to two bits past its own
        const ggml_tensor t8 = rows_of(w, w->type, w->data, R);
        ok                   = add(xdna_gemv_variant(w->type, K, R, false, XDNA_GEMV_SPLIT_FUSED, true), t8, 0);
    }
    if (ok && R < N) {
        std::vector<uint8_t> q4;
        const void *         d4 = (const char *) w->data + (size_t) R * w->nb[1];
        if (w->type != GGML_TYPE_Q4_K) {
            q4 = requant_q4k(w, R, N);
            d4 = q4.data();
        }
        const ggml_tensor t4 = rows_of(w, GGML_TYPE_Q4_K, d4, N - R);
        ok = add(xdna_gemv_variant(GGML_TYPE_Q4_K, K, N - R, false, XDNA_GEMV_SPLIT_FUSED, true), t4, (int) R);
    }
    if (!ok) {
        GGML_LOG_ERROR("%s: the head's weight cannot be packed (K=%lld N=%lld)\n", "xdna-head", (long long) K,
                       (long long) N);
        xdna_head_free(h);
        return nullptr;
    }
    // Both parts in one stream, each its activation's K tiles once, queued a
    // chunk at a time.
    xdna_seq seq;
    for (int i = 0; i < h->n_parts; i++) {
        const part &       p = h->p[i];
        xdna_gemv_seq_opts o;
        o.bd_base    = 0;
        o.w_off      = (uint32_t) p.w_off;
        o.act_off    = (uint32_t) p.a_off;
        o.out_off    = (uint32_t) p.o_off;
        o.act_replay = true;
        if (!xdna_gemv_seq_build(&seq, p.geom, &o)) {
            GGML_LOG_ERROR("%s: the head's stream does not build for %s\n", "xdna-head", p.geom.stem().c_str());
            xdna_head_free(h);
            return nullptr;
        }
    }
    const std::vector<uint32_t> insts = xdna_seq_build(&seq);
    char                        name[128];
    snprintf(name, sizeof(name), "%s_head_%d_%lld", h->p[0].geom.stem().c_str(), h->n_parts, (long long) R);
    h->kern           = xdna_kernel_pool_get_built(pool, name, h->p[0].geom.stem().c_str(), insts.data(), insts.size());
    xdna_device * dev = pool->device;
    h->w              = xdna_buffer_alloc(dev, w_bytes);
    h->a              = xdna_buffer_alloc(dev, a_bytes);
    h->o              = xdna_buffer_alloc(dev, o_bytes);
    if (!h->kern || !h->w || !h->a || !h->o) {
        xdna_head_free(h);
        return nullptr;
    }
    if (!h->w->data || !h->a->data || !h->o->data) {
        xdna_head_free(h);
        return nullptr;
    }
    uint8_t * wm = (uint8_t *) h->w->data;
    for (int i = 0; i < h->n_parts; i++) {
        std::memcpy(wm + h->p[i].w_off, packed[(size_t) i].data(), packed[(size_t) i].size());
    }
    if (!xdna_buffer_sync_to_device(h->w)) {
        xdna_head_free(h);
        return nullptr;
    }
    xdna_buffer * args[3] = { h->w, h->a, h->o };
    h->run                = xdna_kernel_run_make(h->kern, args, 3);

    // The same from the rows: each part's first tile takes F, A and gamma and
    // the prologue quantizes rms_norm(F + A) * gamma; its tiles are every
    // chunk's, in DDR (a replayed first tile would take the rows again).
    if (res && gamma && K == XDNA_RES_D) {
        size_t rb = 0;
        for (int i = 0; i < h->n_parts; i++) {
            h->p[i].ra_off = rb;
            rb             = xdna_align_up(rb + h->p[i].geom.act_bytes());
        }
        const size_t g_off = rb;
        rb += (size_t) XDNA_RES_D * sizeof(float);
        h->ra = xdna_buffer_alloc(dev, rb);
        xdna_seq sr;
        bool     ok_r = h->ra != nullptr;
        for (int i = 0; ok_r && i < h->n_parts; i++) {
            const part & p = h->p[i];
            ok_r           = xdna_gemv_row_act(p.geom, (uint8_t *) h->ra->data + p.ra_off, eps, true, 0);
            xdna_seq_row_io(&sr, 3, XDNA_RES_F, XDNA_RES_A, 1, (uint32_t) g_off, XDNA_RES_H);
            xdna_gemv_seq_opts o;
            o.bd_base = 0;
            o.w_off   = (uint32_t) p.w_off;
            o.act_off = (uint32_t) p.ra_off;
            o.out_off = (uint32_t) p.o_off;
            ok_r      = ok_r && xdna_gemv_seq_build(&sr, p.geom, &o);
            xdna_seq_row_wait(&sr);
        }
        if (ok_r) {
            std::memcpy((uint8_t *) h->ra->data + g_off, gamma, (size_t) XDNA_RES_D * sizeof(float));
            if (!xdna_buffer_sync_to_device(h->ra)) {
                xdna_head_free(h);
                return nullptr;
            }
            const std::vector<uint32_t> ir = xdna_seq_build(&sr);
            snprintf(name, sizeof(name), "%s_head_rows_%d_%lld", h->p[0].geom.stem().c_str(), h->n_parts,
                     (long long) R);
            h->kern_r = xdna_kernel_pool_get_built(pool, name, h->p[0].geom.stem().c_str(), ir.data(), ir.size());
        }
        if (h->kern_r) {
            xdna_buffer * ar[4] = { h->w, h->ra, h->o, res };
            h->run_r            = xdna_kernel_run_make(h->kern_r, ar, 4);
            h->args_r.assign(ar, ar + 4);
        } else {
        }
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    GGML_LOG_INFO(
        "%s: head %s [%lld x %lld]: %lld rows exact (q8g16), %lld rows q4g32, %.0f MB, "
        "packed in %.1f s\n",
        "xdna-head", ggml_type_name(w->type), (long long) K, (long long) N, (long long) R, (long long) (N - R),
        (double) w_bytes / 1e6, s);
    return h;
}

void xdna_head_free(xdna_head * h) {
    if (!h) {
        return;
    }
    xdna_buffer_free(h->w);
    xdna_buffer_free(h->a);
    xdna_buffer_free(h->o);
    xdna_buffer_free(h->ra);
    delete h;
}

bool xdna_head_run(xdna_head * h, const float * x, float * logits) {
    if (!h || !x || !logits || !h->a->data) {
        GGML_LOG_ERROR("%s: bad call or the head's activation has no host mapping (x=%p logits=%p)\n", "xdna-head",
                       (const void *) x, (const void *) logits);
        return false;
    }
    {
        xdna_prof::section_timer st("head: act pack+upload");
        uint8_t *                a = (uint8_t *) h->a->data;
        for (int i = 0; i < h->n_parts; i++) {
            const part & p = h->p[i];
            xdna_gemv_pack_act_into(p.body, x, a + p.a_off);
            ((int32_t *) (a + p.a_off))[1] = p.geom.n_out();  // every chunk
        }
        if (!xdna_buffer_sync_to_device(h->a)) {
            return false;
        }
    }
    {
        xdna_prof::section_timer st("head: dispatch (restart+wait)");
        if (!xdna_run_restart(h->run) || !xdna_run_wait(h->run)) {
            return false;
        }
    }
    return xdna_head_read(h, logits);
}

bool xdna_head_rows(const xdna_head * h) {
    return h && h->kern_r;
}

xrt::run * xdna_head_start_rows(xdna_head * h) {
    if (!xdna_head_rows(h)) {
        return nullptr;
    }
    if (!xdna_run_submit(h->kern_r, h->run_r, h->args_r.data(), h->args_r.size())) {
        return nullptr;
    }
    return &h->run_r;
}

bool xdna_head_read(xdna_head * h, float * logits) {
    xdna_prof::section_timer st("head: logits readback");
    for (int i = 0; i < h->n_parts; i++) {
        const part & p = h->p[i];
        const int    n = std::min(p.geom.n_real, h->n_real - p.row0);
        if (n > 0 && !xdna_buffer_read_settled(h->o, logits + p.row0, (size_t) n * sizeof(float), p.o_off)) {
            return false;
        }
    }
    return true;
}
