// Host-only check of the weight packer: quantize random rows with ggml,
// repack them (xdna_wfmt_repack_row_as) and pack them into GEMV tiles
// (xdna_gemv_pack_weights), then decode both the way gemv-q4.cc does and
// compare against ggml's own dequantization. No NPU involved.
//   pack_probe            - every type, both steps
#include "ggml.h"
#include "../xdna-gemv.h"
#include "../xdna-quant.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static float bf16f(uint16_t b) {
    uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// One record (super-block) of `fmt` -> 256 floats.
static void decode_record(xdna_wfmt fmt, const uint8_t * r, float * y) {
    const bool q4  = fmt == XDNA_WFMT_Q4G32;
    const int  grp = q4 ? XDNA_Q4G32_GROUP : XDNA_Q8G16_GROUP;
    const int  ng  = XDNA_SB_VALUES / grp;
    const int  cb  = q4 ? grp / 2 : grp;
    const int8_t * d8 = (const int8_t *) (r + ng * cb);
    const int8_t * m8 = d8 + ng;
    const uint16_t * p = (const uint16_t *) (r + ng * cb + 2 * ng);
    const float dS = bf16f(p[0]) + bf16f(p[1]);
    const float mS = bf16f(p[2]) + bf16f(p[3]);
    for (int g = 0; g < ng; g++) {
        for (int i = 0; i < grp; i++) {
            int q;
            if (q4) {
                const uint8_t b = r[g * cb + i / 2];
                q = (i & 1) ? (b >> 4) : (b & 15);
            } else {
                q = fmt == XDNA_WFMT_Q8G16 ? (int) (int8_t) r[g * cb + i] : 0;
            }
            y[g * grp + i] = q * (dS * d8[g]) + mS * m8[g];
        }
    }
}

static double rel(const std::vector<float> & a, const std::vector<float> & b) {
    double e = 0, n = 0;
    for (size_t i = 0; i < a.size(); i++) {
        e += (a[i] - b[i]) * (double) (a[i] - b[i]);
        n += b[i] * (double) b[i];
    }
    return std::sqrt(e / (n + 1e-30));
}

int main() {
    const int K = 1024, N = 2048;
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.0f, 0.02f);
    std::vector<float> f((size_t) K * N);
    for (auto & v : f) {
        v = nd(rng);
    }
    const ggml_type types[] = { GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K };
    for (ggml_type t : types) {
        const size_t rb = ggml_row_size(t, K);
        std::vector<uint8_t> q((size_t) N * rb);
        ggml_quantize_chunk(t, f.data(), q.data(), 0, N, K, nullptr);
        std::vector<float> ref((size_t) N * K);
        for (int n = 0; n < N; n++) {
            ggml_get_type_traits(t)->to_float(q.data() + n * rb, ref.data() + (size_t) n * K, K);
        }
        for (xdna_wfmt fmt : { XDNA_WFMT_Q4G32, XDNA_WFMT_Q8G16 }) {
            if (fmt == XDNA_WFMT_Q4G32 && t != GGML_TYPE_Q4_K) {
                continue;
            }
            const size_t xb = xdna_wfmt_row_bytes(fmt, K);
            std::vector<uint8_t> row(xb);
            std::vector<float> dec((size_t) N * K);
            bool ok = true;
            for (int n = 0; n < N && ok; n++) {
                ok = xdna_wfmt_repack_row_as(t, fmt, q.data() + n * rb, K, row.data());
                for (int s = 0; s < K / 256; s++) {
                    decode_record(fmt, row.data() + (size_t) s * (xb / (K / 256)),
                                  dec.data() + (size_t) n * K + s * 256);
                }
            }
            printf("%s -> %s: repack %s, rel err %.3e\n", ggml_type_name(t),
                   fmt == XDNA_WFMT_Q4G32 ? "q4g32" : "q8", ok ? "ok" : "FAILED",
                   rel(dec, ref));
        }

        // The tile packer, on the fused geometry, decoded as the kernel reads.
        ggml_init_params ip = { 64 * 1024, nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * w = ggml_new_tensor_2d(ctx, t, K, N);
        w->data = q.data();
        const xdna_gemv_geom g = xdna_gemv_variant(t, K, N, false, XDNA_GEMV_SPLIT_FUSED);
        std::vector<uint8_t> packed;
        const ggml_tensor * ws[1] = { w };
        if (!g.valid() || !xdna_gemv_pack_weights(g, ws, 1, nullptr, packed)) {
            printf("  pack FAILED\n");
            continue;
        }
        const bool q4   = g.fmt == XDNA_WFMT_Q4G32;
        const int  grp  = g.group(), kt = g.k_tile(), NT = g.n_tiles();
        const int  LANE = g.lane(), nc = g.n_core(), rows = g.rows();
        const int  gpt  = kt / grp, lg = nc / LANE, nsup = g.n_sup();
        const int  cb   = q4 ? grp * LANE / 2 : grp * LANE;
        const int  bb   = cb + 4 * LANE;
        const size_t tb = g.tile_bytes();
        std::vector<float> dec((size_t) N * K, 0.0f);
        for (int n = 0; n < N; n++) {
            const int oc = n / g.chunk(), rem = n % g.chunk();
            const int core = rem / nc, c = core / rows, r = core % rows;
            const int ncol = rem % nc, j = ncol / LANE, lane = ncol % LANE;
            for (int ti = 0; ti < NT; ti++) {
                const uint8_t * tile = packed.data() +
                    (((size_t) (c * g.n_out() + oc) * NT + ti) * rows + r) * tb;
                for (int gg = 0; gg < gpt; gg++) {
                    const uint8_t * blk = tile + (size_t) (j * gpt + gg) * bb;
                    const uint16_t * pb = (const uint16_t *) (blk + cb);
                    const uint16_t * sp = (const uint16_t *) (tile + (size_t) lg * gpt * bb +
                        (size_t) (j * nsup + gg / (gpt / nsup)) * 8 * LANE);
                    const float dS = bf16f(sp[lane]) + bf16f(sp[LANE + lane]);
                    const float mS = bf16f(sp[2 * LANE + lane]) + bf16f(sp[3 * LANE + lane]);
                    const float d = dS * bf16f(pb[lane]), m = mS * bf16f(pb[LANE + lane]);
                    for (int k = 0; k < grp; k++) {
                        int qv;
                        if (q4) {
                            const uint8_t b = blk[(size_t) k * (LANE / 2) + lane / 2];
                            qv = (lane & 1) ? (b >> 4) : (b & 15);
                        } else {
                            qv = (int) (int8_t) blk[(size_t) k * LANE + lane];
                        }
                        dec[(size_t) n * K + ti * kt + gg * grp + k] = qv * d + m;
                    }
                }
            }
        }
        printf("  tiles (%s, k_tile %d, tile %zu B): rel err %.3e\n", q4 ? "q4g32" : "q8",
               kt, tb, rel(dec, ref));
        ggml_free(ctx);
    }
    return 0;
}
