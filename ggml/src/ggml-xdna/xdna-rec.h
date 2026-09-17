#pragma once

// Fused recurrent-layer runner for the XDNA decode path of the Qwen3.5
// gated-delta-net. A single-token decode runs each recurrent layer on one
// device design, keeping the recurrent state on the NPU between tokens:
//
//   xdna_rec_core - fused_layer.xclbin (kernels/fused_layer.py): the merged
//     conv+norm+gdn+gated core plus the decode GEMV side by side, driven by a
//     host-built per-token TXN stream (xdna_attn_gdn_build, fused layout).
//     Conv history and the bf16 ssm state stay on the device; per token only
//     the F_Q qkv windows, the x tails and the z lanes of the azg BO are
//     re-uploaded, and the gated epilogue's int8 aq codes + d_a come back.
//   xdna_rec_gemv - the same design's GEMV cores (xdna-rec-gemv.h): the
//     ssm_out projection, appended to the core's own stream, and the whole
//     FFN (gate/up/silu/down) as a second dispatch on it.
//
// The runner is byte-transparent: it only allocates BOs of the sizes in
// xdna_rec_geom and copies the caller's host buffers in/out. It does not know
// the llama graph - the glue in ggml-xdna.cpp finds the per-layer tensors and
// calls the packers here (the C++ mirror of the python persist harness in
// kernels/attn_cn.py / gdn_v.py / rec_gated.py).

#include "xdna-runtime.h"
#include "xdna-seq.h"

#include <cstddef>
#include <string>
#include <cstdint>
#include <vector>

// Buffer / geometry spec of one persistent fused decode run.
struct xdna_rec_geom {
    int64_t feed_bytes = 0;        // full feed buffer (hist + qkv + conv W)
    int64_t feed_blocks = 0;       // conv blocks in the feed buffer
    int64_t feed_block_floats = 0; // floats per conv block (FEEDN = 1024)
    int64_t feed_qkv_off = 0;      // F_Q offset (floats) of the qkv window
    int64_t sv = 0;                // per-block qkv channel width (S_V = 128)
    int64_t x_bytes = 0;           // per-token x tail (gate/beta/scale)
    int64_t pkvb_bytes = 0;        // conv+norm pkv object output buffer
    int64_t state_bytes = 0;       // fp32 ssm state size (gstate holds half)
    int64_t azg_bytes = 0;         // azg BO: n_vh x [attn|z|gamma|hh] f32
    int64_t out_bytes = 0;         // gated out (scratch + aq + d_a) bytes
    int64_t d_out = 0;             // h_attn / h_out width in floats

    bool valid() const;
};

// Fixed geometry constants of one 1024-wide Qwen3.5 gated-delta-net layer.
namespace xdna_rec_pack {
// conv / gdn stage (attn_gdn_gated.xclbin)
constexpr int CH    = 6144;
constexpr int SV    = 128;
constexpr int NVH   = 16;
constexpr int NCONV = 2;    // conv columns
constexpr int CN    = 24;   // 128-channel groups per conv column
constexpr int FQ    = 3 * SV;
constexpr int FW    = 4 * SV;
constexpr int FEEDN = 8 * SV;
constexpr int HEADNORM = 3 * SV + 3;   // x-head + eg + beta + scale
// attn|z|gamma|hh needs 3*SV+1; the stride is rounded up to a multiple of the
// 16-lane vector so a head inside a multi-head object starts on a vector
// boundary (kernels/attn_gdn_gated.py).
constexpr int AZGN    = ((3 * SV + 1 + 15) / 16) * 16;   // 400
constexpr int D_OUT   = 1024;
constexpr int KGATE   = NVH * SV;      // gdn attn / aq row length
// The gated out BO: ssm_out's GEMV activation first (a header tile then one
// per k_tile), then the gated f32 scratch, then aq and d_a. Mirrors OUTN in
// kernels/attn_gdn_gated.py.
constexpr int ACT_TILE_G = 2112;
constexpr int ACT_OFF = 10496;                            // after aq and d_a
constexpr int ACTN    = (1 + KGATE / 256) * ACT_TILE_G;
constexpr int OUTN    = ACT_OFF + ACTN;

int64_t state_floats();   // NVH * N_OBJ * ROWS_N (fp32 ssm state floats)
int64_t feed_blocks();    // NCONV * CN
} // namespace xdna_rec_pack

xdna_rec_geom xdna_rec_pack_geom(void);

// Static + per-token sources of one fused layer (all ggml-layout host data).
//  - w_conv     : f32 [4 taps x 6144 ch], element (tap, ch) at tap + 4*ch
//  - w_ssm_norm : f32 [128] (ssm_norm gamma, seeded into the core azg lanes)
//  - w_post_norm: f32 [1024] (post-attention RMS gamma, host FFN norm)
struct xdna_rec_layer_host {
    int il = 0;
    const float *   w_conv      = nullptr;
    const float *   w_ssm_norm  = nullptr;
    const float *   w_post_norm = nullptr;
};

// Build the one-time host buffer of a layer run (see xdna_rec_core_begin).
//   conv_hist : 3 rows x 6144 f32 (oldest first), or nullptr (zeros)
//   qkv       : 6144 f32 current-token projections (t=0 window)
void xdna_rec_pack_begin(const xdna_rec_layer_host & h,
                         const float * conv_hist, const float * qkv,
                         std::vector<uint8_t> & feed0);

// Per-token packing helpers. gate is the llama exp-input (softplus*ssm_a)
// [16], beta is the sigmoided projection [16]; x tails place
// [eg, beta, 1/sqrt(SV)] per value head.
void xdna_rec_pack_x(const float * gate, const float * beta, std::vector<uint8_t> & x);

// host RMS norm: out = in / sqrt(mean(in^2) + eps) * w
void xdna_rec_rms_norm(const float * in, const float * w, size_t n,
                       float eps, float * out);

// --- attn_gdn_gated core (conv+norm+gdn+gated) ------------------------------

// A loaded conv+norm+gdn+gated single-run runner for fused_layer.xclbin
// (kernels/fused_layer.py). The BO set is the persistent decode state: feed
// (conv history + qkv + conv weights, only the per-token F_Q windows re-synced
// after t=0), x tails, pkvb (norm output, device-written), gstate (bf16 ssm
// state, in place on the device), azg (per head [attn|z|gamma|hh], attn lanes
// device-written by the gdn stage, z host-uploaded per token, gamma/hh seeded
// once) and out (aq codes + d_a, device-written). The kernel program is bound
// once with the host-built TXN stream (xdna_attn_gdn_build, fused layout).
struct xdna_rec_core {
    xdna_device * dev = nullptr;
    bool dev_owned = false;          // true when the runner opened its own device
    xdna_kernel * kern = nullptr;    // fused_layer.xclbin
    // The core's stream, kept so the ssm_out projection can be appended to a
    // copy of it once the GEMV runner exists (xdna_rec_core_fuse_so). The two
    // are already one array configuration and one hardware context; fusing
    // them makes them one dispatch as well, which is what the array's fixed
    // per-phase cost is paid for.
    xdna_seq      base_seq;
    std::string   xclbin;
    // The projection drains into the FFN's activation tiles rather than into
    // an output buffer of its own, so the layer's two dispatches have nothing
    // between them.
    bool          so_to_act = false;
    bool          ffn_fused = false;
    xdna_kernel * so_kern = nullptr;

    struct xdna_gemv * so_gemv = nullptr;
    xrt::run      so_run;
    bool          so_fused = false;

    bool          pkv_onchip = false;
    int           act_arg = 5;
    xdna_buffer * gbuf = nullptr;   // fused: [activation | weights | out]
    size_t        gbuf_out_off = 0;
    xdna_rec_geom g;                 // geometry (copy of the create arg)

    xdna_buffer * feed = nullptr;
    xdna_buffer * x = nullptr;
    xdna_buffer * pkvb = nullptr;
    xdna_buffer * gstate = nullptr;  // bf16 ssm state (state_bytes/2)
    xdna_buffer * azg = nullptr;     // azg head buffer (device attn + host z)
    xdna_buffer * out = nullptr;     // gated scratch + aq + d_a

    int token = 0;                   // next token index (drives qkv handling)
};

// Open a device (when `dev` is null one is opened and owned by the runner),
// load fused_layer.xclbin and bind the host-built phased TXN stream in place
// of an IRON .insts.bin. Returns nullptr on failure.
xdna_rec_core * xdna_rec_core_create(xdna_device * dev, const xdna_rec_geom & g,
                                     const char * xclbin);

void xdna_rec_core_free(xdna_rec_core * core);

// Start a fresh persistent decode: upload the full feed (conv history + t=0
// qkv + conv weights), seed the azg gamma/hh lanes from the layer ssm_norm
// weight (`gamma`, f32[128]) and zero the bf16 gstate.
bool xdna_rec_core_begin(xdna_rec_core * core, const void * feed0,
                         const float * gamma);

// Seed the bf16 gdn state: fp32 -> bf16 value-wise conversion of `state`
// (state_bytes/4 floats), or zero when NULL. Call once before the first token.
bool xdna_rec_core_seed(xdna_rec_core * core, const void * state);

// Run the conv+norm+gdn+gated stage for token `t`: t == 0 reuses the qkv
// already inside feed0; t > 0 uploads only the F_Q qkv window of every conv
// block from `qkv` (feed_blocks*sv floats). x is re-uploaded every token and z
// (KGATE floats) is uploaded into the azg z lanes. The single kernel run
// advances the bf16 state on the device, applies the gated epilogue on-chip
// and returns the int8 aq codes + the int8 scale d_a. Advances the token.
// Returns false on failure.
// The ssm_out activation the gated stage wrote, in the GEMV's tile layout.
// Valid until the next core run.
const void * xdna_rec_core_act(const xdna_rec_core * core);


// The core's output buffer, and the offset a fused projection drains into.
const void * xdna_rec_core_out(const xdna_rec_core * core);
size_t xdna_rec_core_out_off(const xdna_rec_core * core);

// Append the ssm_out projection to the core's stream and bind a run over both
// buffer sets, so a layer's first two dispatches become one. `so` is the
// projection's GEMV and `ffn` its FFN pair, whose activation tiles the
// projection drains into. Returns false when the two streams cannot share a
// column's descriptors, in which case the caller keeps the two dispatches.
bool xdna_rec_core_fuse_so(xdna_rec_core * core, struct xdna_kernel_pool * pool,
                           struct xdna_gemv * so, struct xdna_gemv_pair * ffn);

// True when the fused run is bound and xdna_rec_core_run will also project.
bool xdna_rec_core_so_fused(const xdna_rec_core * core);

// True when that projection drains into the FFN's activation tiles, so its
// result never passes through the host.
bool xdna_rec_core_so_to_act(const xdna_rec_core * core);

// True when the layer's FFN is in that same dispatch, so the caller only has
// to read its output back.
bool xdna_rec_core_ffn_fused(const xdna_rec_core * core);

bool xdna_rec_core_run(xdna_rec_core * core, int t, const void * qkv,
                       const void * x, const float * z, int8_t * aq,
                       float * d_a);

