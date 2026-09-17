#pragma once

// Fused recurrent-layer runner for the XDNA decode path of the Qwen3.5
// gated-delta-net. A single-token decode runs each recurrent layer on three
// device kernels, keeping the recurrent state on the NPU between tokens:
//
//   xdna_rec_core - attn_gdn_gated.xclbin (kernels/attn_gdn_gated.py): the
//     merged conv+norm+gdn+gated A-half in one run per token, driven by a
//     host-built per-token TXN stream (xdna_attn_gdn_txn_build, fused layout).
//     Conv history and the bf16 ssm state stay on the device; per token only
//     the F_Q qkv windows, the x tails and the z lanes of the azg BO are
//     re-uploaded, and the int8 aq codes + d_a come back (attn never leaves).
//   xdna_rec_so  - attn_so_mmul.xclbin (kernels/attn_so_mmul.py): the ssm_out
//     projection on the native int8 x int4 MMUL. The core returns the int8
//     gated aq codes (written into the feed A slabs here) and the host adds the
//     residual (h_attn).
//   xdna_rec_ffn - ffn_mmul_ab.xclbin (kernels/ffn_layer_mmul.py): the whole
//     FFN (gate/up/silu/down) on the int8 x int4 MMUL, adding the FFN output
//     to h_attn (h_out).
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

// A loaded conv+norm+gdn+gated single-run runner for attn_gdn_gated.xclbin
// (kernels/attn_gdn_gated.py). The BO set is the persistent decode state: feed
// (conv history + qkv + conv weights, only the per-token F_Q windows re-synced
// after t=0), x tails, pkvb (norm output, device-written), gstate (bf16 ssm
// state, in place on the device), azg (per head [attn|z|gamma|hh], attn lanes
// device-written by the gdn stage, z host-uploaded per token, gamma/hh seeded
// once) and out (aq codes + d_a, device-written). The kernel program is bound
// once with the host-built TXN stream (xdna_attn_gdn_txn_build, fused layout).
struct xdna_rec_core {
    xdna_device * dev = nullptr;
    bool dev_owned = false;          // true when the runner opened its own device
    xdna_kernel * kern = nullptr;    // attn_gdn_gated.xclbin
    // GGML_XDNA_ATTN_PHASE_PROF=1: the same token split into one dispatch per
    // stage, so each can be timed. The cores loop forever on their fifos, so
    // the work and its order are unchanged - only where the token is cut.
    // They share one hardware context, the xclbin being the same.
    xdna_kernel * phase_kern[4] = {};
    // The FFN transition as its own dispatch on the SAME xclbin
    // (GGML_XDNA_POST_FUSED): fill+drain of the in-design post tile, so no
    // second design and no second context.
    xdna_kernel * post_kern = nullptr;
    // The core's stream, kept so the ssm_out projection can be appended to a
    // copy of it once the GEMV runner exists (xdna_rec_core_fuse_so). The two
    // are already one array configuration and one hardware context; fusing
    // them makes them one dispatch as well, which is what the array's fixed
    // per-phase cost is paid for.
    xdna_seq      base_seq;
    std::string   xclbin;
    // The same design as a full ELF (GGML_XDNA_CORE_ELF, fused_layer_full.elf):
    // its sequence is the ELF's control code and its buffers are plain kernel
    // arguments, so the layer is not limited to the six a patched stream can
    // name. Null unless the switch is on.
    struct xdna_rec_core_elf * elf = nullptr;
    bool          elf_seeded = false;
    // The projection drains into the FFN's activation tiles rather than into
    // an output buffer of its own, so the layer's two dispatches have nothing
    // between them.
    bool          so_to_act = false;
    // The FFN rides the same stream, so the layer is one dispatch.
    bool          ffn_fused = false;
    xdna_kernel * so_kern = nullptr;
    std::vector<uint32_t> ref_words;

    struct xdna_gemv * so_gemv = nullptr;
    xrt::run      so_run;
    bool          so_fused = false;
    struct xdna_rec_post * post = nullptr;  // the standalone FFN transition
    uint32_t      post_in_off = 0;    // residual and gamma go here
    uint32_t      post_out_off = 0;   // hattn and the FFN activation

    bool          pkv_onchip = false;
    int           out_base = 0;
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
// load attn_gdn_gated.xclbin and bind the host-built phased TXN stream
// (schedule 4, fused layout) in place of an IRON .insts.bin. Returns nullptr
// on failure.
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

// Byte offset of that activation inside the core's output buffer.
size_t xdna_rec_core_act_off(void);

// The standalone FFN transition runner (xdna-rec-post.h).
struct xdna_rec_post;
struct xdna_rec_post * xdna_rec_core_post_get(xdna_rec_core * core,
                                              struct xdna_kernel_pool * pool);

// The FFN transition tile's host-side windows, when it rides this stream.
void * xdna_rec_core_post_in(const xdna_rec_core * core);
const void * xdna_rec_core_post_out(const xdna_rec_core * core);
void xdna_rec_core_post_sync(xdna_rec_core * core);
bool xdna_rec_core_post_run(xdna_rec_core * core);


// Where the host leaves the layer's residual and gamma for the post tile, and
// where that tile leaves hattn and the FFN's activation. Null when the post
// transition is not active.
// The core's output buffer, and the offset a fused projection drains into.
const void * xdna_rec_core_out(const xdna_rec_core * core);
size_t xdna_rec_core_out_off(const xdna_rec_core * core);

// Append the ssm_out projection to the core's stream and bind a run over both
// buffer sets, so a layer's first two dispatches become one. `so` is the
// projection's GEMV, `act_off` the byte offset in the core's output buffer
// where the gated stage leaves the activation it reads. Returns false when the
// two streams cannot share a column's descriptors, in which case the caller
// keeps the two dispatches.
bool xdna_rec_core_fuse_so(xdna_rec_core * core, struct xdna_kernel_pool * pool,
                           struct xdna_gemv * so, uint32_t act_off,
                           struct xdna_gemv_pair * ffn);

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

// --- attn_so_mmul (ssm_out) ------------------------------------------------

// Sizes / layout mirrors attn_so_mmul.py resolve(): 8 columns x 2 groups of 64
// out columns, K = 2048 in 32 slabs of 64 rows. Each feed object = padded int8
// A slab (256 B) + one 64x64 int4 W tile (2048 B).
namespace xdna_so_pack {
constexpr int K      = 2048;   // gated row dim (ssm_out K)
constexpr int N      = 1024;   // ssm_out col dim
constexpr int NCOL   = 8;
constexpr int GRP    = 64;
constexpr int CPC    = N / NCOL;      // 128
constexpr int NG     = CPC / GRP;     // 2 groups per column
constexpr int SB     = K / 64;        // 32 slabs per group
constexpr int A_SZ   = 256;
constexpr int B_SZ   = 2048;
constexpr int OBJ    = A_SZ + B_SZ;   // 2304 B per object
constexpr int NT     = NG * SB;       // 64 objects per column
constexpr int CB_Q5  = (K / 256) * 176;  // 1408 B per Q5_K ssm_out column
constexpr int CB_Q4  = (K / 256) * 144;  // 1152 B per Q4_K ssm_out column
constexpr int CB_Q6  = (K / 256) * 210;  // 1680 B per Q6_K ssm_out column

inline int64_t feed_bytes() { return (int64_t) NCOL * NT * OBJ; }
inline int64_t acc_bytes()  { return (int64_t) NCOL * NG * 256 * 4; }
} // namespace xdna_so_pack

// The ssm_out-mmul device state of one fused layer session.
struct xdna_rec_so {
    xdna_device * dev = nullptr;
    xdna_kernel * kern = nullptr;   // attn_so_mmul xclbin + insts
    xdna_buffer * feed = nullptr;   // per-token feed (int4 W resident + int8 A)
    xdna_buffer * acc = nullptr;    // raw int32 group accumulators [4096]
    std::vector<uint8_t> feed_tmpl; // feed with the int4 W grid (constant)
    std::vector<float> dw;          // per-output-column W scale [N]
    // Exact fp32 ssm_out weights [N][K], only for the verified layer
    // (GGML_XDNA_VERIFY); empty otherwise.
    std::vector<float> w_ref;
    int il = 0;
};

// Load the attn_so_mmul xclbin/insts, allocate the BO set, dequant the
// ssm_out [K x N] into the int4 grid + per-column scales and upload the
// constant W-grid feed template. Returns nullptr on failure.
// `so_type` is the ggml_type of the ssm_out weight (Q4_K, Q5_K or Q6_K); the
// column stride and the dequant routine follow from it.
xdna_rec_so * xdna_rec_so_create(xdna_device * dev, int il,
                                 const char * xclbin, const char * insts,
                                 const uint8_t * w_so, int so_type);

void xdna_rec_so_free(xdna_rec_so * m);

// Run one decode ssm_out: the int8 aq codes + d_a scale arrive from the fused
// core (attn_gdn_gated); rewrite the A slab of every feed object from aq, run
// the mmul and write h_attn[n] = hres[n] + acc*d_a*d_w[n].
bool xdna_rec_so_run(xdna_rec_so * m, const int8_t * aq, float d_a,
                     const float * hres, float * h_attn);

// --- ffn_mmul_ab (whole FFN) ------------------------------------------------

// Sizes / layout mirrors ffn_layer_mmul.py resolve(4, 4) + build_a_feed /
// build_b_feed. Group = 64 output cols; each feed object is one (col, group,
// k-slab) tile element.
namespace xdna_rec_ffn_pack {
constexpr int K     = 1024;    // hidden (gate/up row) dim = hff width
constexpr int KDOWN = 3584;    // down row dim = mid length
constexpr int N_MID = 3584;
constexpr int N_OUT = 1024;
// stage A element layout (bytes)
constexpr int A_HDR  = 512;
constexpr int A_AOF  = 512;
constexpr int A_BGO  = 768;
constexpr int A_BUO  = 2816;
constexpr int A_OBJ  = 4864;
// stage B element layout
constexpr int B_TAG  = 2048;
constexpr int B_OBJ  = 2056;
constexpr int NA     = 4;      // stage-A columns
constexpr int NB     = 4;      // stage-B columns
constexpr int GRPA   = 64;
constexpr int GRPB   = 64;
constexpr int SA     = 16;     // stage-A slabs (K / 64)
constexpr int SB     = 56;     // stage-B slabs (KDOWN / 64)
constexpr int NGA    = N_MID / (NA * GRPA);   // 14 stage-A groups per column
constexpr int NGB    = N_OUT / (NB * GRPB);   // 4 stage-B groups per column
constexpr int NTA    = SA * NGA;              // 224 A objects per column
constexpr int NTB    = SB * NGB;              // 224 B objects per column
constexpr int CPCA   = N_MID / NA;            // 896
constexpr int CPCB   = N_OUT / NB;            // 256

// per-token stage-A feed, shared scratch mid/acc, stage-B feed, raw int32 out
inline int64_t af_bytes()   { return (int64_t) NA * NTA * A_OBJ; }
inline int64_t mid_bytes()  { return (int64_t) KDOWN * 4; }
inline int64_t acc_bytes()  { return (int64_t) NA * 512 * 4; }
inline int64_t bf_bytes()   { return (int64_t) NB * NTB * B_OBJ; }
inline int64_t out_bytes()  { return (int64_t) NB * NGB * 256 * 4; }
inline int64_t aux_bytes()  { return (int64_t) NB * 8 * 4; }
} // namespace xdna_rec_ffn_pack

// The mmul-FFN device state of one fused layer session.
struct xdna_rec_ffn {
    xdna_device * dev = nullptr;
    xdna_kernel * kern = nullptr;   // ffn_mmul_ab xclbin + insts
    xdna_buffer * af = nullptr;     // per-token stage-A feed (int8 A + header)
    xdna_buffer * mid = nullptr;    // mid f32 / in-place int8 codes scratch
    xdna_buffer * acc = nullptr;    // stage-A int32 acc drain
    xdna_buffer * bf = nullptr;     // stage-B down int4 feed (constant)
    xdna_buffer * out = nullptr;    // raw int32 down acc [1024]
    xdna_buffer * aux = nullptr;    // per-stage-B-col d_aB

    std::vector<uint8_t> af_tmpl;   // af with the gate/up int4 grids (const)
    std::vector<float> dw_g;        // per-mid-column gate scale [N_MID]
    std::vector<float> dw_u;        // per-mid-column up scale [N_MID]
    std::vector<float> dw_d;        // per-output-column down scale [N_OUT]
    // Exact fp32 weights of the verified layer (GGML_XDNA_VERIFY): gate/up
    // [N_MID][K] and down [N_OUT][KDOWN]. Empty otherwise.
    std::vector<float> wg_ref;
    std::vector<float> wu_ref;
    std::vector<float> wd_ref;
    int il = 0;
};

// Load the ffn_mmul_ab xclbin/insts, allocate the BO set, dequant the Q4_K
// gate/up [K x N_MID] and Q4_K or Q6_K down [KDOWN x N_OUT] weights (down_q6)
// into the int4 grids + per-column scales and build the constant stage-A
// gate/up template and the stage-B feed. Returns nullptr on failure.
xdna_rec_ffn * xdna_rec_ffn_create(xdna_device * dev, int il,
                                   const char * xclbin, const char * insts,
                                   const uint8_t * w_gate, const uint8_t * w_up,
                                   const uint8_t * w_down, bool down_q6);

void xdna_rec_ffn_free(xdna_rec_ffn * m);

// Run one decode (M=1) FFN token: quantize hff -> int8, fill the af header +
// A codes, run the fused kernel and write h_out = h_attn + (d_aB*d_w rescaled
// raw int32). All lengths are fixed by the geometry above.
bool xdna_rec_ffn_run(xdna_rec_ffn * m, const float * hff,
                      const float * h_attn, float * h_out);
