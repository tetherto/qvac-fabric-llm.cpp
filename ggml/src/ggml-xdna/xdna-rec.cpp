#include "xdna-rec.h"

#include "ggml-impl.h"
#include "ggml-quants.h"
#include "ggml.h"
#include "xdna-gemv.h"
#include "xdna-prof.h"
#include "xdna-seq.h"
#include "xdna-util.h"

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/xrt_hw_context.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>

// Fused recurrent-layer runner for the Qwen3.5 gated-delta-net decode (see
// xdna-rec.h). The BO lifecycles and kernel arg layouts are 1:1 ports of the
// persistent python design in kernels/fused_layer.py, so feeding the same
// layer weights produces the same device BO contents the python harness
// validated.

static bool upload(xdna_buffer * b, const void * data) {
    std::memcpy(b->data, data, b->bytes);
    return xdna_buffer_sync_to_device(b);
}

bool xdna_rec_geom::valid() const {
    return feed_bytes > 0 && feed_blocks > 0 && feed_block_floats > 0 && sv > 0 && x_bytes > 0 && pkvb_bytes > 0 &&
           state_bytes > 0 && azg_bytes > 0 && out_bytes > 0 && d_out > 0;
}

// --- host packing -----------------------------------------------------------

namespace xdna_rec_pack {

int64_t state_floats() {
    return (int64_t) NVH * (SV / 16) * ((int64_t) 16 * SV);  // 16 heads x 8 chunks x 2048
}

int64_t feed_blocks() {
    return (int64_t) NCONV * CN;
}

}  // namespace xdna_rec_pack

xdna_rec_geom xdna_rec_pack_geom(void) {
    using namespace xdna_rec_pack;
    xdna_rec_geom g;
    g.feed_bytes        = feed_blocks() * FEEDN * (int64_t) sizeof(float);
    g.feed_blocks       = feed_blocks();
    g.feed_block_floats = FEEDN;
    g.feed_qkv_off      = FQ;
    g.sv                = SV;
    g.x_bytes           = (int64_t) NVH * HEADNORM * sizeof(float);
    g.pkvb_bytes        = (int64_t) NVH * 8 * HEADNORM * sizeof(float);
    g.state_bytes       = state_floats() * (int64_t) sizeof(float);
    g.azg_bytes         = (int64_t) NVH * AZGN * sizeof(float);
    g.out_bytes         = OUTN;
    g.d_out             = D_OUT;
    return g;
}

void xdna_rec_pack_begin(const xdna_rec_layer_host & h,
                         const float *               conv_hist,
                         const float *               qkv,
                         std::vector<uint8_t> &      feed0) {
    using namespace xdna_rec_pack;
    if (!h.w_conv) {
        return;
    }
    feed0.assign((size_t) feed_blocks() * FEEDN * sizeof(float), 0);
    float * feed = (float *) feed0.data();

    const int64_t blocks = feed_blocks();
    // history window [0, 3*SV) and qkv window [FQ, FQ+SV) of every block are
    // contiguous copies of the llama conv state and current qkv rows. Both the
    // llama conv state and the kernel history window keep the 3 taps of each
    // channel adjacent (element (tap t, ch) at 3*ch + t), so one block's SV
    // channels are one contiguous copy from the channel-major llama state.
    for (int64_t b = 0; b < blocks; b++) {
        float * blk = feed + b * FEEDN;
        if (conv_hist) {
            std::memcpy(blk, conv_hist + (size_t) b * SV * 3, (size_t) SV * 3 * sizeof(float));
        }
        if (qkv) {
            std::memcpy(blk + FQ, qkv + (size_t) b * SV, SV * sizeof(float));
        }
        // conv weights: blk[F_W + 4c + t] = w_conv[t + 4 * (b*SV + c)]
        for (int c = 0; c < SV; c++) {
            const float * w = h.w_conv + 4 * ((size_t) b * SV + c);
            std::memcpy(blk + FW + (size_t) 4 * c, w, 4 * sizeof(float));
        }
    }
}

void xdna_rec_pack_x(const float * gate, const float * beta, std::vector<uint8_t> & x) {
    using namespace xdna_rec_pack;
    const int64_t n = (int64_t) NVH * HEADNORM;
    x.assign((size_t) n * sizeof(float), 0);
    float *     X     = (float *) x.data();
    const float scale = 1.0f / std::sqrt((float) SV);
    for (int hh = 0; hh < NVH; hh++) {
        float * tail = X + (size_t) hh * HEADNORM + (size_t) 3 * SV;
        tail[0]      = std::exp(gate[hh]);
        tail[1]      = beta[hh];
        tail[2]      = scale;
    }
}

void xdna_rec_rms_norm(const float * in, const float * w, size_t n, float eps, float * out) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += (double) in[i] * (double) in[i];
    }
    const float rms = (float) std::sqrt(sum / (double) n + (double) eps);
    for (size_t i = 0; i < n; i++) {
        out[i] = in[i] / rms * w[i];
    }
}

// --- attn_gdn_gated core (conv+norm+gdn+gated) ------------------------------

// The FFN transition on the array, inside this design so it shares the one
// context (a second xclbin for it breaks the next fused dispatch): the
// residual add, the norm, the gamma and the quantized activation the FFN
// reads.
//
xdna_rec_core * xdna_rec_core_create(xdna_device * dev, const xdna_rec_geom & g, const char * xclbin) {
    // The merged conv+norm+gdn+gated core runs on fused_layer.xclbin
    // (kernels/fused_layer.py, barrier-free persistent workers) driven by a
    // host-built per-token TXN stream instead of an IRON-compiled stream. The
    // worker geometry and the {feed, x, pkvb, gstate, azg, out} BO set are
    // fixed by xdna_rec_pack_geom; only the bound instruction words differ
    // from a compiled .insts.bin.
    if (!g.valid() || !xclbin) {
        GGML_LOG_ERROR("%s: invalid core create args\n", "xdna-rec");
        return nullptr;
    }
    xdna_seq           seq;
    xdna_attn_gdn_geom tgeom;
    tgeom.azg_n             = xdna_rec_pack::AZGN;      // fused azg head layout
    tgeom.out_words         = (int) (g.out_bytes / 4);  // gated out object (words)
    // The phased schedule puts four gdn rounds in flight before a wait. It is
    // the fastest of the schedules measured end to end on repeated tg64 runs
    // (0 = 18.9, 1 = 19.9, 2 = 20.3, 4 = 21.3).
    //
    // Must match the CONV_GPO the artifact was built with (see
    // kernels/attn_gdn_gated.py): it decides the object the stage's fifos
    // carry, and therefore the shape of every conv descriptor.
    tgeom.conv_gpo          = 4;
    tgeom.pkv_onchip        = 1;
    tgeom.attn_onchip       = 1;
    tgeom.gdn_state_streams = 1;
    tgeom.feed_slot         = tgeom.feed_n;
    // The gated activation drains into the argument the on-chip pkv path freed,
    // at offset zero: argument 5 already carries the gated output's own drain,
    // and that one is patched with the bit31 form.
    tgeom.gated_act_bytes   = (int) (g.out_bytes - xdna_rec_pack::ACT_OFF);
    tgeom.gated_act_col     = 7;
    tgeom.gated_act_arg     = 7;
    tgeom.gated_act_off     = 0;

    if (!xdna_attn_gdn_build(&seq, &tgeom, 4, -1)) {
        GGML_LOG_ERROR("%s: fused layer stream build failed\n", "xdna-rec");
        return nullptr;
    }
    std::vector<uint32_t> words = xdna_seq_build(&seq);
    if (words.empty()) {
        GGML_LOG_ERROR("%s: fused layer stream empty\n", "xdna-rec");
        return nullptr;
    }

    bool owned = false;
    if (!dev) {
        dev = xdna_device_open();
        if (!dev) {
            return nullptr;
        }
        owned = true;
    }
    xdna_rec_core * core = new xdna_rec_core;
    core->dev            = dev;
    core->dev_owned      = owned;
    core->g              = g;

    core->base_seq   = seq;
    core->xclbin     = xclbin ? xclbin : "";
    core->pkv_onchip = tgeom.pkv_onchip != 0;
    core->act_arg    = tgeom.gated_act_arg;
    core->kern       = xdna_kernel_load_hw(dev, xclbin);
    if (!core->kern || !xdna_kernel_bind_insts(dev, core->kern, words.data(), words.size())) {
        xdna_rec_core_free(core);
        return nullptr;
    }

    core->feed = xdna_buffer_alloc(dev, (size_t) g.feed_bytes);
    // Past x, room for the rest of the prologue's tail object when the gates
    // are the array's (xdna_seq_row_io_gates: 5 more head-sets of it).
    core->x    = xdna_buffer_alloc(dev, (size_t) g.x_bytes * 7);
    // Sized by the larger of the two gated layouts, like `out` below: the
    // activation the norm writes has to fit whichever one the artifact is.
    core->pkvb = xdna_buffer_alloc(
        dev, std::max((size_t) g.pkvb_bytes, (size_t) (xdna_rec_pack::OUTN_MAX - xdna_rec_pack::ACT_OFF)));
    core->gstate = xdna_buffer_alloc(dev, (size_t) g.state_bytes / 2);
    core->azg    = xdna_buffer_alloc(dev, (size_t) g.azg_bytes);
    // The gated output, sized by the larger layout, plus room past it for a
    // projection fused into this stream to drain into (it cannot have an
    // argument of its own). The stream's own lengths come from g.out_bytes and
    // follow the layout the artifact was built with.
    core->out    = xdna_buffer_alloc(dev, (size_t) xdna_rec_pack::OUTN_MAX + 8192);
    if (!core->feed || !core->x || !core->pkvb || !core->gstate || !core->azg || !core->out) {
        xdna_rec_core_free(core);
        return nullptr;
    }
    if (!core->feed->data || !core->x->data || !core->pkvb->data || !core->gstate->data || !core->azg->data ||
        !core->out->data) {
        xdna_rec_core_free(core);
        return nullptr;
    }
    std::memset(core->gstate->data, 0, (size_t) g.state_bytes / 2);
    if (!xdna_buffer_sync_to_device(core->gstate)) {
        xdna_rec_core_free(core);
        return nullptr;
    }
    return core;
}

void xdna_rec_core_free(xdna_rec_core * core) {
    if (!core) {
        return;
    }
    xdna_buffer_free(core->feed);
    xdna_buffer_free(core->x);
    xdna_buffer_free(core->pkvb);
    xdna_buffer_free(core->gstate);
    xdna_buffer_free(core->azg);
    xdna_buffer_free(core->out);
    xdna_buffer_free(core->gbuf);
    xdna_buffer_free(core->w6);
    xdna_buffer_free(core->a7);
    if (core->kern) {
        xdna_kernel_free(core->kern);
    }
    if (core->dev_owned && core->dev) {
        delete core->dev;
    }
    delete core;
}

bool xdna_rec_core_begin(xdna_rec_core * core, const void * feed0, const float * gamma) {
    if (!core || !feed0 || !gamma || !core->feed->data || !core->azg->data) {
        GGML_LOG_ERROR("%s: core begin: bad call or a core BO has no host mapping\n", "xdna-rec");
        return false;
    }
    if (!upload(core->feed, feed0)) {
        return false;
    }
    // Seed the azg gamma + hh lanes (host-written once per layer); the attn
    // lanes are overwritten by the gdn stage every token and the z lanes by
    // every run.
    const int64_t azg_head = (int64_t) xdna_rec_pack::AZGN;
    float *       azg      = (float *) core->azg->data;
    std::memset(azg, 0, (size_t) core->g.azg_bytes);
    for (int h = 0; h < xdna_rec_pack::NVH; h++) {
        float * head = azg + h * azg_head;
        std::memcpy(head + (size_t) 2 * xdna_rec_pack::SV, gamma, (size_t) xdna_rec_pack::SV * sizeof(float));
        head[(size_t) 3 * xdna_rec_pack::SV] = (float) h;  // hh tail
    }
    if (!xdna_buffer_sync_to_device(core->azg)) {
        return false;
    }
    core->token = 0;
    return true;
}

bool xdna_rec_core_read_state(xdna_rec_core * core, float * conv_hist, float * ssm) {
    using namespace xdna_rec_pack;
    if (!core || !core->feed || !core->gstate || !conv_hist || !ssm) {
        return false;
    }
    if (!core->feed->data || !core->gstate->data) {
        return false;
    }
    if (!xdna_buffer_sync_from_device(core->feed)) {
        return false;
    }
    const float * feed = (const float *) core->feed->data;
    for (int64_t b = 0; b < core->g.feed_blocks; b++) {
        std::memcpy(conv_hist + (size_t) b * SV * 3, feed + (size_t) b * core->g.feed_block_floats,
                    (size_t) SV * 3 * sizeof(float));
    }
    if (!xdna_buffer_sync_from_device(core->gstate)) {
        return false;
    }
    const uint16_t * st = (const uint16_t *) core->gstate->data;
    const size_t     n  = (size_t) core->g.state_bytes / sizeof(float);
    for (size_t i = 0; i < n; i++) {
        const uint32_t w = (uint32_t) st[i] << 16;
        std::memcpy(ssm + i, &w, sizeof(float));
    }
    return true;
}

bool xdna_rec_core_seed(xdna_rec_core * core, const void * state) {
    if (!core || !core->gstate) {
        return false;
    }
    if (!core->gstate->data) {
        return false;
    }
    const size_t n = (size_t) core->g.state_bytes / sizeof(float);  // state floats
    if (state == nullptr) {
        std::memset(core->gstate->data, 0, (size_t) core->g.state_bytes / 2);
        return xdna_buffer_sync_to_device(core->gstate);
    }
    const float * src = (const float *) state;
    uint16_t *    dst = (uint16_t *) core->gstate->data;
    for (size_t i = 0; i < n; i++) {
        dst[i] = xdna_bf16(src[i]);
    }
    return xdna_buffer_sync_to_device(core->gstate);
}

const void * xdna_rec_core_out(const xdna_rec_core * core) {
    // Where the fused projection drains: an argument of its own.
    return core && core->so_gemv ? core->so_gemv->o->data : nullptr;
}

size_t xdna_rec_core_out_off(const xdna_rec_core * core) {
    return core ? core->gbuf_out_off : 0;
}

const void * xdna_rec_core_act(const xdna_rec_core * core) {
    // The gated stage drains the activation into the projection's own
    // argument, so that is where it is read from.
    if (core && core->a7) {
        return core->a7->data;
    }
    return core && core->so_gemv ? core->so_gemv->a->data : nullptr;
}

bool xdna_rec_core_set_inproj(xdna_rec_core * core, const xdna_gemv_geom & geom, const std::vector<uint8_t> & packed) {
    if (!core || core->so_fused || !geom.valid() || !geom.fused_v || packed.size() != geom.weight_bytes()) {
        return false;
    }
    core->in_geom   = geom;
    core->in_packed = packed;
    core->inproj    = true;
    return true;
}

bool xdna_rec_core_set_rows(xdna_rec_core * core,
                            xdna_buffer *   res,
                            const float *   gamma_attn,
                            float           eps_attn,
                            const float *   gamma_post,
                            float           eps_post) {
    if (!core || !core->inproj || core->so_fused || !res || !gamma_attn || !gamma_post ||
        core->in_geom.K != XDNA_RES_D) {
        return false;
    }
    core->rows = true;
    core->res  = res;
    core->g_attn.assign(gamma_attn, gamma_attn + XDNA_RES_D);
    core->g_post.assign(gamma_post, gamma_post + XDNA_RES_D);
    core->eps_attn = eps_attn;
    core->eps_post = eps_post;
    return true;
}

bool xdna_rec_core_set_gates(xdna_rec_core * core,
                             const float *   w_alpha,
                             const float *   w_beta,
                             const float *   dt,
                             const float *   a) {
    using namespace xdna_rec_pack;
    if (!core || !core->rows || core->so_fused || !w_alpha || !w_beta || !dt || !a) {
        return false;
    }
    // [constants: dt[16], a[16], the scale, padded to 512 words | 32 rows]
    const size_t n = (size_t) 512 * 4 + (size_t) 2 * NVH * XDNA_RES_D * 2;
    core->ab.assign(n, 0);
    float * c = (float *) core->ab.data();
    for (int h = 0; h < NVH; h++) {
        c[h]       = dt[h];
        c[NVH + h] = a[h];
    }
    c[(size_t) 2 * NVH] = 1.0f / std::sqrt((float) SV);
    uint16_t * w        = (uint16_t *) (core->ab.data() + (size_t) 512 * 4);
    for (size_t i = 0; i < (size_t) NVH * XDNA_RES_D; i++) {
        w[i]                             = xdna_bf16(w_alpha[i]);
        w[(size_t) NVH * XDNA_RES_D + i] = xdna_bf16(w_beta[i]);
    }
    core->gates = true;
    return true;
}

bool xdna_rec_core_inproj_act(xdna_rec_core * core, const float * act) {
    if (!core || !core->inproj || !core->a7 || !act || !core->a7->data) {
        GGML_LOG_ERROR("%s: in-projection activation: bad call or the activation BO has no host mapping\n", "xdna-rec");
        return false;
    }
    const size_t n = core->in_geom.act_bytes();
    core->in_host_a.resize(n);
    xdna_gemv_pack_act_into(core->in_geom, act, core->in_host_a.data());
    std::memcpy((uint8_t *) core->a7->data + core->in_a_off, core->in_host_a.data(), n);
    // Only this window: the ssm_out half of the buffer is the array's.
    return xdna_buffer_sync_to_device_range(core->a7, n, core->in_a_off);
}

namespace {

// One cleanup path for the buffers xdna_rec_core_fuse_so allocates before it
// binds a run: a build that fails partway leaves none of them behind, and a
// core the build is attempted on again does not leak the attempt before it.
void fuse_so_release(xdna_rec_core * core) {
    xdna_buffer_free(core->gbuf);
    xdna_buffer_free(core->w6);
    xdna_buffer_free(core->a7);
    core->gbuf = nullptr;
    core->w6   = nullptr;
    core->a7   = nullptr;
}

// The in-projection as the head of a fused stream. Its descriptors start from
// zero on every column - nothing has run before it - and it waits for its own
// drains, so the core's stream after it can hand the same ids out again.
bool build_inproj_head(const xdna_rec_core * core, xdna_seq * seq) {
    using namespace xdna_rec_pack;
    const xdna_gemv_geom & g   = core->in_geom;
    const int              n_o = xdna_gemv_out_streams(g);
    const int              sw  = xdna_gemv_stream_floats(g);
    const int              sv  = (int) core->g.sv;
    if (n_o < 2 || n_o > XDNA_GEMV_COLS || sv <= 0 || sw % sv != 0 || sv != SV) {
        return false;
    }
    const uint32_t     nb   = (uint32_t) (sw / sv);  // blocks a stream carries per chunk
    const uint32_t     fblk = (uint32_t) core->g.feed_block_floats * 4;
    const uint32_t     fq   = (uint32_t) core->g.feed_qkv_off * 4;
    xdna_gemv_out_dest dst[XDNA_GEMV_COLS];
    for (int c = 0; c < n_o - 1; c++) {
        // Stream c of chunk j holds qkv blocks (j*(n_o-1) + c)*nb ...: each
        // one lands in its block's F_Q window.
        dst[c].arg          = 0;
        dst[c].off          = (uint32_t) c * nb * fblk + fq;
        dst[c].blk_floats   = (uint32_t) sv;
        dst[c].n_blk        = nb;
        dst[c].blk_stride   = fblk;
        dst[c].chunk_stride = (uint32_t) (n_o - 1) * nb * fblk;
    }
    // The last stream holds z, nb heads per chunk, into each head's z lane.
    xdna_gemv_out_dest & zd = dst[n_o - 1];
    zd.arg                  = 4;
    zd.off                  = (uint32_t) SV * 4;
    zd.blk_floats           = (uint32_t) sv;
    zd.n_blk                = nb;
    zd.blk_stride           = (uint32_t) AZGN * 4;
    zd.chunk_stride         = nb * (uint32_t) AZGN * 4;

    xdna_gemv_seq_opts opt;
    opt.bd_base  = 0;
    opt.w_arg    = 6;
    opt.w_off    = (uint32_t) core->in_w_off;
    opt.act_arg  = 7;
    opt.act_off  = (uint32_t) core->in_a_off;
    opt.o_arg    = 8;
    opt.out_dest = dst;
    return xdna_gemv_seq_build(seq, g, &opt);
}

}  // namespace

// The core and the ssm_out projection as one dispatch. Two dispatches measure
// 282 + 109 us a layer where the merged one measures 339, which is the array's
// fixed per-phase cost paid once instead of twice: 33.6 -> 34.4 t/s.
//
// Three things had to be fixed to get that far, and each is a rule worth
// keeping:
//   - only the low argument indices can be patched. The same stream naming
//     arguments six through eight times out where naming zero through two
//     runs, and the core's own stream patches zero through five - so the two
//     halves share one set of six.
//   - an argument carries one form of patch. The gated output's drain uses
//     the bit31 form, and any plainly patched descriptor on that argument
//     hangs the stream. So nothing else touches argument 5, the gated stage
//     sends its activation elsewhere (see gated_act_bytes), and everything
//     the projection needs - that activation, its weights, its output - lives
//     in the one argument the on-chip pkv path freed.
//   - the fused kernel has to be shared by every layer. One of its own per
//     layer is twenty-eight hardware contexts created lazily in the middle of
//     NPU work, and the same stream would then run or hang from one process
//     to the next. With the pooled kernel the behaviour is repeatable, which
//     is what let the rest of this be found at all.
//   - the activation window has to hold a valid tile count before the first
//     token. A zero count means the cores never take the weights the stream
//     has already pushed, and the dispatch waits on transfers that cannot
//     drain.
//
// The handover through DDR works and needs nothing special: the drain's
// completion token is enough, the same way the conv drains write x for the
// norm fills every token. What looked like a stale read was the second patch
// form on argument 5.
bool xdna_rec_core_fuse_so(xdna_rec_core * core, xdna_kernel_pool * pool, xdna_gemv * so, xdna_gemv_pair * ffn) {
    if (!core || !so || core->xclbin.empty()) {
        return false;
    }
    if (!core->pkv_onchip) {
        return false;  // argument 2 still carries the pkv round trip
    }
    // The core's columns hand out descriptors from zero and the widest of them
    // takes eight, so the projection starts at eight and has the other half of
    // the file. Its arguments follow the core's six, and its activation is not
    // a buffer of its own at all - it is the window of the core's output that
    // the gated stage has just written, so the fused stream reads it where it
    // lies instead of copying it out and back.
    xdna_seq seq = core->base_seq;
    if (core->inproj) {
        // [in-projection | core | ssm_out]: the head drains qkv and z where
        // the core's fills read them, then the core's own stream as built.
        core->in_w_off = xdna_align_up(so->geom.weight_bytes(), 4096);
        core->in_a_off = xdna_align_up(so->geom.act_bytes(), 4096);
        xdna_seq head  = core->base_seq;
        head.ops.clear();
        head.n_instr = 0;
        for (uint32_t & b : head.bd_used) {
            b = 0;
        }
        if (core->rows) {
            // its input: h = F + A into H, rms_norm(h) * gamma into its tiles
            core->g_attn_off = xdna_align_up(core->in_a_off + core->in_geom.act_bytes(), 4096);
            core->g_post_off = core->g_attn_off + (size_t) XDNA_RES_D * sizeof(float);
            if (core->gates) {
                using namespace xdna_rec_pack;
                // after the in-projection's weights, which follow ssm_out's
                core->ab_off = xdna_align_up(core->in_w_off + core->in_geom.weight_bytes(), 4096);
                xdna_seq_row_io_gates(&head, 9, XDNA_RES_F, XDNA_RES_A, 7, (uint32_t) core->g_attn_off, XDNA_RES_H, 6,
                                      (uint32_t) core->ab_off, (uint32_t) (core->ab.size() / 4), 1,
                                      (uint32_t) (3 * SV * 4), (uint32_t) HEADNORM, (uint32_t) NVH);
            } else {
                xdna_seq_row_io(&head, 9, XDNA_RES_F, XDNA_RES_A, 7, (uint32_t) core->g_attn_off, XDNA_RES_H);
            }
        }
        if (!build_inproj_head(core, &head)) {
            GGML_LOG_ERROR("%s: in-projection head: stream build failed\n", "xdna-rec");
            return false;
        }
        if (core->rows) {
            xdna_seq_row_wait(&head);
        }
        // ssm_out's weights before the core's stages, not after: the pool is
        // idle while they run and the weight channels are the pool's own, so
        // the fill runs as far ahead as the MemTile's fifo lets it. Its
        // descriptors are the ones it always had, after the core's: the head
        // has been waited for, and the FFN's row reserves the top of the file
        // (xdna_seq_row_io) while ssm_out is still in flight.
        uint32_t w_used[XDNA_SEQ_MAX_COLS] = {};
        {
            xdna_seq w = head;
            w.ops.clear();
            w.n_instr = 0;
            for (int c = 0; c < XDNA_SEQ_MAX_COLS; c++) {
                w.bd_used[c] = core->base_seq.bd_used[c];
            }
            xdna_gemv_seq_opts wo;
            wo.bd_base = -1;
            wo.w_arg   = 6;
            wo.w_off   = 0;
            wo.stages  = 1;
            if (!xdna_gemv_seq_build(&w, so->geom, &wo)) {
                GGML_LOG_ERROR("%s: the ssm_out weight stream does not build\n", "xdna-rec");
                return false;
            }
            head.ops.insert(head.ops.end(), w.ops.begin(), w.ops.end());
            head.n_instr += w.n_instr;
            std::copy(std::begin(w.bd_used), std::end(w.bd_used), std::begin(w_used));
        }
        head.ops.insert(head.ops.end(), core->base_seq.ops.begin(), core->base_seq.ops.end());
        head.n_instr += core->base_seq.n_instr;
        for (int c = 0; c < XDNA_SEQ_MAX_COLS; c++) {
            head.bd_used[c] = std::max(core->base_seq.bd_used[c], w_used[c]);
        }
        seq = head;
    }
    // Two rules decide this mapping. Only the low argument indices can be
    // patched - the same stream naming arguments six through eight times out
    // where naming zero through two runs - so the two halves share the core's
    // six. And an argument may carry only one form of patch: the gated
    // output's drain uses the bit31 form, so nothing else may touch argument
    // 5, which is why the gated stage sends its activation elsewhere.
    xdna_gemv_seq_opts opt;
    opt.bd_base = -1;
    opt.w_arg   = 6;
    opt.w_off   = 0;
    opt.act_arg = 7;
    opt.act_off = 0;
    opt.o_arg   = 8;
    opt.out_off = 0;
    // With the transition on the array, argument 8 is not an output buffer at
    // all - it is the FFN's activation, and the projection drains one stream
    // into each of its tiles. A stream carries og*rows*n_core = 256 columns
    // and a tile holds K_TILE = 256, so they line up one to one, and the first
    // tile is the header. Nothing of the projection's result reaches the host
    // between the two dispatches after this.
    if (core->rows && core->inproj && ffn) {
        // into S: the FFN's row reads it
        opt.o_arg       = 9;
        opt.out_off     = XDNA_RES_S;
        core->so_to_act = true;
    } else if (ffn && xdna_gemv_pair_raw_act(ffn)) {
        opt.out_off           = XDNA_GEMV_ACT_TILE;
        opt.out_stream_stride = XDNA_GEMV_ACT_TILE;
        core->so_to_act       = true;
    }
    opt.skip_w = core->inproj;  // queued before the core's stages
    if (!xdna_gemv_seq_build(&seq, so->geom, &opt)) {
        GGML_LOG_ERROR("%s: the fused projection stream does not build\n", "xdna-rec");
        return false;
    }
    // The FFN after it, in the same stream: the projection has just drained
    // into its raw tiles, and their host half - the residual and gamma - is
    // written before the dispatch starts (xdna_gemv_pair_prep_raw), so the
    // stream has everything the prologue reads. Its weights follow the
    // in-projection's in argument 6, and its output lands in the tail of its
    // own activation buffer (argument 8), since it has no argument left.
    // GGML_XDNA_FUSE_FFN=0 keeps it a dispatch of its own.
    core->ffn_fused      = false;
    size_t     ffn_w_off = 0;
    const bool fuse_ffn  = core->inproj && core->so_to_act && ffn &&
                          !(getenv("GGML_XDNA_FUSE_FFN") && getenv("GGML_XDNA_FUSE_FFN")[0] == '0');
    if (fuse_ffn) {
        ffn_w_off = xdna_align_up(core->in_w_off + core->in_geom.weight_bytes(), 4096);
        if (core->gates) {
            ffn_w_off = xdna_align_up(core->ab_off + core->ab.size(), 4096);
        }
        xdna_gemv_pair_opts po;
        po.w_arg      = 6;
        po.w_base     = (uint32_t) ffn_w_off;
        po.a_arg      = 8;
        po.a_base     = 0;
        po.o_arg      = 8;
        po.o_base     = (uint32_t) ffn->o_tail_off;
        po.bd_base    = 0;
        po.act_replay = true;
        if (core->rows) {
            // its input: h_attn = H + S into A, rms_norm(h_attn) * gamma
            // into its tiles, per chunk in DDR; its output into F
            xdna_seq_row_io(&seq, 9, XDNA_RES_S, XDNA_RES_H, 7, (uint32_t) core->g_post_off, XDNA_RES_A);
            po.o_arg      = 9;
            po.o_base     = XDNA_RES_F;
            po.act_replay = false;
        }
        if (!xdna_gemv_seq_build_pair(&seq, ffn->g1, ffn->g2, ffn->w2_off, ffn->a2_off, &po)) {
            GGML_LOG_ERROR("%s: FFN pair: stream build failed\n", "xdna-rec");
            return false;
        }
        if (core->rows) {
            xdna_seq_row_wait(&seq);
        }
        core->ffn_fused = true;
    }
    const std::vector<uint32_t> words = xdna_seq_build(&seq);
    if (words.empty()) {
        GGML_LOG_ERROR("%s: the fused stream is empty\n", "xdna-rec");
        return false;
    }
    const std::string stem = std::filesystem::path(core->xclbin).stem().string();
    // Through the pool, under a name every layer shares: the stream is the
    // same for all of them - only the buffers a run binds differ - and a
    // kernel of one's own per layer is twenty-eight hardware contexts created
    // lazily in the middle of NPU work. That is what made the fused dispatch
    // time out at random: the same stream would run or hang from one process
    // to the next.
    // Shared by name, so the name has to say which stream it is: layers whose
    // weights pack into different formats build different streams (the FFN's
    // down projection is Q4_K in some layers and Q6_K in others), and a
    // layer bound to another's stream runs its own buffers through the wrong
    // shapes - garbage, or a hang.
    uint64_t          h    = 1469598103934665603ull;
    for (uint32_t v : words) {
        h = (h ^ v) * 1099511628211ull;
    }
    char kname[48];
    snprintf(kname, sizeof(kname), "rec_core_so_%016llx", (unsigned long long) h);
    core->so_kern = pool ? xdna_kernel_pool_get_built(pool, kname, stem.c_str(), words.data(), words.size()) : nullptr;
    if (!core->so_kern) {
        return false;  // the pool names the kernel it could not load
    }
    // One buffer for the projection: [activation | weights | output].
    if (!core->gbuf) {
        core->gbuf =
            xdna_buffer_alloc(core->dev, so->geom.act_bytes() + so->geom.weight_bytes() + so->geom.out_bytes());
        if (!core->gbuf || !core->gbuf->data) {
            return false;
        }
        std::memcpy((uint8_t *) core->gbuf->data + so->geom.act_bytes(), so->w->data, so->geom.weight_bytes());
    }
    core->gbuf_out_off  = (size_t) opt.out_off;
    xdna_buffer * tiles = core->so_to_act ? xdna_gemv_pair_act_buf(ffn) : so->o;
    xdna_buffer * wb    = so->w;
    xdna_buffer * abuf  = so->a;
    if (core->inproj) {
        // One buffer per argument, both projections in it.
        const size_t ffn_wb = core->ffn_fused ? ffn->w2_off + ffn->g2.weight_bytes() : 0;
        const size_t wn     = core->ffn_fused ? ffn_w_off + ffn_wb : core->in_w_off + core->in_geom.weight_bytes();
        const size_t an     = core->rows ? core->g_post_off + (size_t) XDNA_RES_D * sizeof(float) :
                                           core->in_a_off + core->in_geom.act_bytes();
        core->w6            = xdna_buffer_alloc(core->dev, wn);
        core->a7            = xdna_buffer_alloc(core->dev, an);
        if (!core->w6 || !core->a7 || !core->w6->data || !core->a7->data) {
            fuse_so_release(core);
            return false;
        }
        uint8_t * w = (uint8_t *) core->w6->data;
        std::memset(w, 0, wn);
        std::memcpy(w, so->w->data, so->geom.weight_bytes());
        std::memcpy(w + core->in_w_off, core->in_packed.data(), core->in_packed.size());
        if (core->gates) {
            std::memcpy(w + core->ab_off, core->ab.data(), core->ab.size());
        }
        if (core->ffn_fused) {
            std::memcpy(w + ffn_w_off, ffn->w->data, ffn_wb);
        }
        if (!xdna_buffer_sync_to_device(core->w6)) {
            fuse_so_release(core);
            return false;
        }
        std::memset(core->a7->data, 0, an);
        std::vector<uint8_t>().swap(core->in_packed);
        wb   = core->w6;
        abuf = core->a7;
    }
    xdna_buffer * a10[10] = { core->feed, core->x, core->pkvb, core->gstate, core->azg,
                              core->out,  wb,      abuf,       tiles,        core->rows ? core->res : tiles };
    core->so_run          = xdna_kernel_run_make(core->so_kern, a10, core->rows ? 10 : 9);
    core->so_args.assign(a10, a10 + (core->rows ? 10 : 9));
    // The projection reads its activation from the window of this buffer that
    // the gated stage writes in the same dispatch. On the first token there is
    // nothing there yet, and a zero tile count means the cores never take the
    // weights the stream has already pushed - so the dispatch waits forever on
    // transfers that cannot drain. Seed a valid empty activation once.
    {
        xdna_buffer * ab = abuf;
        uint8_t *     m  = (uint8_t *) ab->data + opt.act_off;
        std::memset(m, 0, so->geom.act_bytes());
        int32_t * hdr = (int32_t *) m;
        hdr[0]        = so->geom.n_tiles();
        hdr[1]        = so->geom.n_out();
        hdr[2]        = 0;  // attention mode off (attn-dec.cc)
        hdr[3]        = 0;
        for (int t = 0; t <= so->geom.n_tiles(); t++) {
            int32_t * tl                   = (int32_t *) (m + (size_t) t * XDNA_GEMV_ACT_TILE);
            tl[XDNA_GEMV_ACT_TILE / 4 - 1] = so->geom.fmt == XDNA_WFMT_Q4G32 ? 0 : 1;
            tl[XDNA_GEMV_ACT_TILE / 4 - 2] = (t == so->geom.n_tiles() && so->geom.epilogue) ? 1 : 0;
        }
        if (!xdna_buffer_sync_to_device(ab)) {
            fuse_so_release(core);
            return false;
        }
        if (core->rows) {
            // The two rows' tiles and gammas, fixed: the prologue takes the
            // rest from the residual rows every token.
            uint8_t * a7m = (uint8_t *) core->a7->data;
            if (!xdna_gemv_row_act(core->in_geom, a7m + core->in_a_off, core->eps_attn, true, 0,
                                   core->gates ? 2 * xdna_rec_pack::NVH : 0) ||
                !xdna_gemv_row_act(ffn->g1, (uint8_t *) ffn->a->data, core->eps_post, true,
                                   xdna_gemv_pair_last_flags(ffn))) {
                GGML_LOG_ERROR("%s: the fused row activation tiles do not build\n", "xdna-rec");
                fuse_so_release(core);
                return false;
            }
            std::memcpy(a7m + core->g_attn_off, core->g_attn.data(), (size_t) XDNA_RES_D * 4);
            std::memcpy(a7m + core->g_post_off, core->g_post.data(), (size_t) XDNA_RES_D * 4);
            if (!xdna_buffer_sync_to_device(core->a7) || !xdna_buffer_sync_to_device(ffn->a)) {
                fuse_so_release(core);
                return false;
            }
        } else if (core->inproj) {
            // A valid row for the head too, for the same reason.
            std::vector<float> zero((size_t) core->in_geom.K, 0.0f);
            if (!xdna_rec_core_inproj_act(core, zero.data())) {
                fuse_so_release(core);
                return false;
            }
        }
    }
    core->so_gemv  = so;
    core->so_fused = true;
    return true;
}

bool xdna_rec_core_so_fused(const xdna_rec_core * core) {
    return core && core->so_fused;
}

bool xdna_rec_core_so_to_act(const xdna_rec_core * core) {
    return core && core->so_to_act;
}

bool xdna_rec_core_ffn_fused(const xdna_rec_core * core) {
    return core && core->ffn_fused;
}

xrt::run * xdna_rec_core_launch(xdna_rec_core * core, int t) {
    if (!core || !core->so_fused || !core->rows || !core->gates || !core->inproj || t != core->token || t == 0) {
        return nullptr;
    }
    if (!xdna_run_submit(core->so_kern, core->so_run, core->so_args.data(), core->so_args.size())) {
        return nullptr;
    }
    core->token = t + 1;
    return &core->so_run;
}

bool xdna_rec_core_run(xdna_rec_core * core,
                       int             t,
                       const void *    qkv,
                       const void *    x,
                       const float *   z,
                       int8_t *        aq,
                       float *         d_a) {
    using namespace xdna_rec_pack;
    if (!core || !core->kern || t != core->token || !x || (!z && !core->inproj) || !aq || !d_a) {
        GGML_LOG_ERROR("%s: core run: bad call (t=%d token=%d)\n", "xdna-rec", t, core ? core->token : -1);
        return false;
    }
    if (!core->feed->data || !core->x->data || !core->azg->data || !core->out->data) {
        return false;
    }
    if (t > 0 && !core->inproj) {
        if (!qkv) {
            GGML_LOG_ERROR("%s: core run: t>0 needs the qkv window\n", "xdna-rec");
            return false;
        }
        // Upload only the F_Q qkv window of each conv block: the F_H history
        // (device conv state) is never written by the host after t=0.
        xdna_prof::section_timer st("rec: feed qkv windows (memcpy+sync)");
        char *                   map       = (char *) core->feed->data;
        const char *             src       = (const char *) qkv;
        const size_t             win_bytes = (size_t) core->g.sv * sizeof(float);
        for (int64_t b = 0; b < core->g.feed_blocks; b++) {
            const size_t off = (size_t) (b * core->g.feed_block_floats + core->g.feed_qkv_off) * 4;
            std::memcpy(map + off, src + b * win_bytes, win_bytes);
            if (!xdna_buffer_sync_to_device_range(core->feed, win_bytes, off)) {
                return false;
            }
        }
    }
    if (!core->gates) {
        xdna_prof::section_timer st("rec: x upload");
        std::memcpy(core->x->data, x, (size_t) core->g.x_bytes);
        if (!xdna_buffer_sync_to_device_range(core->x, (size_t) core->g.x_bytes, 0)) {
            return false;
        }
    }
    xdna_prof::section_timer st_z("rec: azg z lanes (sync back+copy+sync)");
    if (!core->inproj) {
        // Upload the per-token z into the azg z lanes (head h at h*AZGN + SV).
        // The attn lanes are the array's own output from the previous token and the
        // gamma/hh lanes were seeded once, so pull the buffer back before writing
        // the z lanes: the flush below covers the whole buffer and would otherwise
        // put a stale host image over the attn lanes. The gdn stage does rewrite
        // them before the gated phase reads them, but nothing should depend on that
        // ordering for the buffer to be correct.
        if (!xdna_buffer_sync_from_device(core->azg)) {
            return false;
        }
        float * azg = (float *) core->azg->data;
        for (int h = 0; h < NVH; h++) {
            std::memcpy(azg + (int64_t) h * AZGN + SV, z + (int64_t) h * SV, (size_t) SV * sizeof(float));
        }
        if (!xdna_buffer_sync_to_device(core->azg)) {
            return false;
        }
    }
    st_z.stop();

    // The fused run (conv+norm+gdn+gated, and the ssm_out projection when it
    // is appended): the kernel reads the feed, x tails and the azg z lanes,
    // updates the persistent bf16 state in place, drains the per-head attn
    // into the azg attn lanes and runs the gated epilogue, draining aq + d_a
    // into out. One dispatch for the core and the projection: the stream waits
    // on the gated drain before the projection's activation fill reads it.
    xdna_buffer *            args[6] = { core->feed, core->x, core->pkvb, core->gstate, core->azg, core->out };
    // out layout: [gated f32 scratch KGATE*4][aq int8 KGATE][d_a f32]. Mark the
    // codes and their scale before the run so the read after it can tell the
    // array's write from the previous token's contents.
    xdna_buffer_mark(core->out, (size_t) KGATE + sizeof(float), (size_t) KGATE * 4);
    xdna_prof::section_timer st_run("rec: core dispatch (restart+wait)");
    if (core->so_fused) {
        if (!xdna_run_restart(core->so_run) || !xdna_run_wait(core->so_run)) {
            return false;
        }
    } else {
        xrt::run run = xdna_kernel_run_start(core->kern, args, 6);
        if (!xdna_run_wait(run)) {
            return false;
        }
    }

    st_run.stop();
    xdna_prof::section_timer st_out("rec: out readback");
    // out layout: [gated f32 scratch KGATE*4][aq int8 KGATE][d_a f32]. Read
    // against the mark set before the dispatch: a run reports completion
    // before its last writes are readable, and the codes are exactly such a
    // read.
    if (core->so_fused && !xdna_buffer_sync_from_device(core->x)) {
        return false;
    }
    if (!xdna_buffer_download(core->out, aq, (size_t) KGATE, (size_t) KGATE * 4) ||
        !xdna_buffer_download(core->out, d_a, sizeof(float), (size_t) KGATE * 4 + KGATE)) {
        return false;
    }
    core->token = t + 1;
    return true;
}
