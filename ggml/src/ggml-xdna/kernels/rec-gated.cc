// The gated-attention epilogue of the fused recurrent core: per-head
// normalisation, silu and gamma (ggml_xdna_gated_head), then the row quantisation and
// the activation tiles the projection reads (ggml_xdna_gated_fin). Read by rec_gated.py
// and attn_gdn_gated.py and compiled by IRON as "ggml_xdna_gated_head" / "ggml_xdna_gated_fin".
//
// The per-build knobs (GATED_FMT, ACT_SPLIT, ACT_TILE, ACT_OFF, K_GATE) are
// compile flags; see attn_gdn_gated.py.
//
// State is carried in the OUT object so ggml_xdna_gated_head and ggml_xdna_gated_fin can share it
// without cross-TU statics (IRON compiles each ExternalFunction from the same
// source into its own TU). OUT layout:
//   [0 .. AQ_OFF)      gated f32 scratch (GATED_N floats), written per head
//   [AQ_OFF .. DA_OFF) aq int8 codes, one scale for the row
//   [DA_OFF .. DA_OFF + 4) d_a f32
//   [ACT_OFF ..]       the same values as ssm_out's GEMV activation: a header
//                      tile then one per k_tile, int8 codes with a scale and a
//                      code sum per group of GATED_GRP. Writing it here is what
//                      lets the projection read it without a repack on the host.
#include "xdna-math.h"

#include <math.h>
#include <stdint.h>

#include <aie_api/aie.hpp>
using namespace aie;

namespace {

constexpr int GATED_D   = 128;               // head dim: attn, z and gamma
constexpr int GATED_N   = K_GATE;            // values in a row
constexpr int GATED_HH  = 2 * GATED_D;       // the head's slot in [attn][z][hh]
constexpr int AQ_OFF    = GATED_N * 4;       // int8 codes after the f32 scratch
constexpr int DA_OFF    = AQ_OFF + GATED_N;  // the row's scale
constexpr int GATED_GRP = 32;                // values a group scale covers

#if defined(GATED_FMT) && GATED_FMT == 1
// The 8-bit weight form: tiles of 128 codes, the layout the projection's q8
// kernel reads. Q5_K and Q6_K are the 8-bit form, so the design is built per
// model (GATED_FMT=1) and the tag covers the flag.
constexpr int GATED_KT   = 128;
constexpr int GATED_FMTW = 1;
#else
constexpr int GATED_KT   = 256;
constexpr int GATED_FMTW = 0;
#endif
constexpr int GATED_NG = GATED_KT / GATED_GRP;  // groups a tile

// The last two words of an activation tile: the flags and the code width.
constexpr int ACT_FLAGS_W = ACT_TILE / 4 - 2;
constexpr int ACT_WIDTH_W = ACT_TILE / 4 - 1;

}  // namespace

// az per head = [attn GATED_D][z GATED_D][hh] (hh at the tail keeps attn/z
// 64B-aligned; each call finds its slot by hh).
extern "C" void ggml_xdna_gated_head(uint8_t * out, const float * az, const float * gamma) {
    const int     hh   = (int) az[GATED_HH];
    float *       gbuf = (float *) out + hh * GATED_D;
    const float * a    = az;
    const float * z    = az + GATED_D;
    float         ms   = 0.0f;
    for (int i = 0; i < GATED_D; i++) {
        ms += a[i] * a[i];
    }
    xdna::gated_head(gbuf, a, z, gamma, xdna::rms_scale(ms, GATED_D, xdna::RMS_EPS));
}

#if ACT_SPLIT
extern "C" void ggml_xdna_gated_fin(uint8_t * out, uint8_t * actbuf) {
#else
extern "C" void ggml_xdna_gated_fin(uint8_t * out) {
#endif
    const float *          gbuf    = (const float *) out;
    // Both passes are vector work. They used to be scalar loops over the
    // row with a divide per element, which measured as the single most
    // expensive thing in the fused core - more than the per-head epilogue and
    // four times the data movement of the whole stage.
    aie::vector<float, 16> vmax    = aie::zeros<float, 16>();
    const auto             absmask = aie::broadcast<int32, 16>(0x7FFFFFFF);
    for (int k = 0; k < GATED_N; k += 16) {
        auto v  = aie::load_v<16>(gbuf + k);
        // aie::abs does not hold for f32 on this target; mask the sign bit.
        auto av = aie::bit_and(v.cast_to<int32>(), absmask).cast_to<float>();
        vmax    = aie::max(vmax, av);
    }
    alignas(64) float lanes[16];
    aie::store_v(lanes, vmax);
    float amax = 0.0f;
    for (int i = 0; i < 16; i++) {
        if (lanes[i] > amax) {
            amax = lanes[i];
        }
    }

    const float da   = amax > 0.0f ? amax / xdna::CODE_MAX : 1.0f;
    // One divide for the whole row instead of GATED_N of them.
    const float inv  = 1.0f / da;
    const auto  vinv = aie::broadcast<float, 16>(inv);
    const auto  vlo  = aie::broadcast<float, 16>(xdna::CODE_MIN);
    const auto  vhi  = aie::broadcast<float, 16>(xdna::CODE_MAX);
    int8 *      aq   = (int8 *) (out + AQ_OFF);
    for (int k = 0; k < GATED_N; k += 16) {
        const auto v = aie::mul(aie::load_v<16>(gbuf + k), vinv).to_vector<float>();
        aie::store_v(aq + k, aie::pack(aie::pack(xdna::quant_i32(v, vlo, vhi))));
    }
    float * dap = (float *) (out + DA_OFF);
    dap[0]      = da;

    // The same codes again in the GEMV's activation tile layout, but with a
    // scale and a code sum per group of GATED_GRP rather than one for the row:
    // a group scale is local to the values a core holds, which is what the
    // projection's kernel reads, in either weight form's tiles.
    //
    // A drain of its own when the activation is split off: the projection that
    // reads it back in the same stream needs a plainly patched descriptor, and
    // the gated output's own drain cannot have one.
#if ACT_SPLIT
    uint8_t * act = actbuf;
#else
    uint8_t * act = out + ACT_OFF;
#endif
    int32_t * hdr    = (int32_t *) act;
    hdr[0]           = K_GATE / GATED_KT;
    hdr[1]           = 1;
    // words 2 and 3 select the pool's attention mode (attn-dec.cc): this
    // object is reused, so they are cleared, never left to its last use
    hdr[2]           = 0;
    hdr[3]           = 0;
    hdr[ACT_FLAGS_W] = 0;
    hdr[ACT_WIDTH_W] = GATED_FMTW;
    for (int t = 0; t < K_GATE / GATED_KT; t++) {
        uint8_t * tile = act + (1 + t) * ACT_TILE;
        int8 *    code = (int8 *) tile;
        float *   gsum = (float *) (tile + GATED_KT);
        float *   gd   = gsum + GATED_NG;
        for (int g = 0; g < GATED_NG; g++) {
            const float * v = gbuf + t * GATED_KT + g * GATED_GRP;
            float         gd_g;
            const int     sum = xdna::quant_group(aie::load_v<16>(v), aie::load_v<16>(v + 16), vlo,
                                                  (int8_t *) (code + g * GATED_GRP), gd_g);
            gsum[g]           = (float) sum;
            gd[g]             = gd_g;
        }
        ((int32_t *) tile)[ACT_FLAGS_W] = 0;
        ((int32_t *) tile)[ACT_WIDTH_W] = GATED_FMTW;
    }
}
