#include "xdna-rec.h"
#include "xdna-gemv.h"
#include "xdna-seq.h"
#include "xdna-util.h"

#include "ggml-impl.h"
#include "ggml.h"
#include "ggml-quants.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/xrt_hw_context.h>

// Fused recurrent-layer runner for the Qwen3.5 gated-delta-net decode (see
// xdna-rec.h). The BO lifecycles and kernel arg layouts are 1:1 ports of the
// persistent python design in kernels/fused_layer.py, so feeding the same
// layer weights produces the same device BO contents the python harness
// validated.

static void upload(xdna_buffer * b, const void * data) {
    std::memcpy(b->bo.map(), data, b->bytes);
    xdna_buffer_sync_to_device(b);
}

bool xdna_rec_geom::valid() const {
    return feed_bytes > 0 && feed_blocks > 0 && feed_block_floats > 0 &&
           sv > 0 && x_bytes > 0 && pkvb_bytes > 0 && state_bytes > 0 &&
           azg_bytes > 0 && out_bytes > 0 && d_out > 0;
}

// --- host packing -----------------------------------------------------------

namespace xdna_rec_pack {

int64_t state_floats() {
    return NVH * (SV / 16) * (16 * SV);   // 16 heads x 8 chunks x 2048
}

int64_t feed_blocks() {
    return NCONV * CN;
}

} // namespace xdna_rec_pack

xdna_rec_geom xdna_rec_pack_geom(void) {
    using namespace xdna_rec_pack;
    xdna_rec_geom g;
    g.feed_bytes        = feed_blocks() * FEEDN * (int64_t) sizeof(float);
    g.feed_blocks       = feed_blocks();
    g.feed_block_floats = FEEDN;
    g.feed_qkv_off      = FQ;
    g.sv                = SV;
    g.x_bytes           = NVH * HEADNORM * (int64_t) sizeof(float);
    g.pkvb_bytes        = NVH * 8 * HEADNORM * (int64_t) sizeof(float);
    g.state_bytes       = state_floats() * (int64_t) sizeof(float);
    g.azg_bytes         = NVH * AZGN * (int64_t) sizeof(float);
    g.out_bytes         = OUTN;
    g.d_out             = D_OUT;
    return g;
}

void xdna_rec_pack_begin(const xdna_rec_layer_host & h,
                         const float * conv_hist, const float * qkv,
                         std::vector<uint8_t> & feed0) {
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
            std::memcpy(blk, conv_hist + (size_t) b * SV * 3, SV * 3 * sizeof(float));
        }
        if (qkv) {
            std::memcpy(blk + FQ, qkv + (size_t) b * SV, SV * sizeof(float));
        }
        // conv weights: blk[F_W + 4c + t] = w_conv[t + 4 * (b*SV + c)]
        for (int c = 0; c < SV; c++) {
            const float * w = h.w_conv + 4 * ((size_t) b * SV + c);
            std::memcpy(blk + FW + 4 * c, w, 4 * sizeof(float));
        }
    }
}

void xdna_rec_pack_x(const float * gate, const float * beta, std::vector<uint8_t> & x) {
    using namespace xdna_rec_pack;
    const int64_t n = NVH * HEADNORM;
    x.assign((size_t) n * sizeof(float), 0);
    float * X = (float *) x.data();
    const float scale = 1.0f / std::sqrt((float) SV);
    for (int hh = 0; hh < NVH; hh++) {
        float * tail = X + (size_t) hh * HEADNORM + 3 * SV;
        tail[0] = std::exp(gate[hh]);
        tail[1] = beta[hh];
        tail[2] = scale;
    }
}

void xdna_rec_rms_norm(const float * in, const float * w, size_t n,
                       float eps, float * out) {
    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        sum += (double) in[i] * in[i];
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
xdna_rec_core * xdna_rec_core_create(xdna_device * dev, const xdna_rec_geom & g,
                                     const char * xclbin) {
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
    xdna_seq seq;
    xdna_attn_gdn_geom tgeom;
    tgeom.azg_n     = xdna_rec_pack::AZGN;      // fused azg head layout
    tgeom.out_words = (int) (g.out_bytes / 4);  // gated out object (words)
    // The phased schedule puts four gdn rounds in flight before a wait. It is
    // the fastest of the schedules measured end to end on repeated tg64 runs
    // (0 = 18.9, 1 = 19.9, 2 = 20.3, 4 = 21.3).
    //
    // Must match the CONV_GPO the artifact was built with (see
    // kernels/attn_gdn_gated.py): it decides the object the stage's fifos
    // carry, and therefore the shape of every conv descriptor.
    tgeom.conv_gpo         = 4;
    tgeom.pkv_onchip       = 1;
    tgeom.attn_onchip      = 1;
    tgeom.gdn_state_streams = 1;
    tgeom.feed_slot        = tgeom.feed_n;
    // The gated activation drains into the argument the on-chip pkv path freed,
    // at offset zero: argument 5 already carries the gated output's own drain,
    // and that one is patched with the bit31 form.
    tgeom.gated_act_bytes = (int) (g.out_bytes - xdna_rec_pack::ACT_OFF);
    tgeom.gated_act_col   = 7;
    tgeom.gated_act_arg   = 7;
    tgeom.gated_act_off   = 0;

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
    core->dev = dev;
    core->dev_owned = owned;
    core->g = g;

    core->base_seq  = seq;
    core->xclbin    = xclbin ? xclbin : "";
    core->pkv_onchip = tgeom.pkv_onchip != 0;
    core->act_arg    = tgeom.gated_act_arg;
    core->kern = xdna_kernel_load_hw(dev, xclbin);
    if (!core->kern ||
        !xdna_kernel_bind_insts(dev, core->kern, words.data(), words.size())) {
        xdna_rec_core_free(core);
        return nullptr;
    }

    core->feed = xdna_buffer_alloc(dev, (size_t) g.feed_bytes);
    core->x    = xdna_buffer_alloc(dev, (size_t) g.x_bytes);
    core->pkvb = xdna_buffer_alloc(dev,
                                   std::max((size_t) g.pkvb_bytes,
                                            (size_t) (g.out_bytes -
                                                      xdna_rec_pack::ACT_OFF)));
    core->gstate = xdna_buffer_alloc(dev, (size_t) g.state_bytes / 2);
    core->azg = xdna_buffer_alloc(dev, (size_t) g.azg_bytes);
    // Room past the gated output for a projection fused into this stream
    // to drain into: it cannot have an argument of its own.
    core->out = xdna_buffer_alloc(dev, (size_t) g.out_bytes + 8192);
    if (!core->feed || !core->x || !core->pkvb || !core->gstate || !core->azg ||
        !core->out) {
        GGML_LOG_ERROR("%s: core BO allocation failed\n", "xdna-rec");
        xdna_rec_core_free(core);
        return nullptr;
    }
    std::memset(core->gstate->bo.map(), 0, (size_t) g.state_bytes / 2);
    xdna_buffer_sync_to_device(core->gstate);
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
    if (core->kern) {
        xdna_kernel_free(core->kern);
    }
    if (core->dev_owned && core->dev) {
        delete core->dev;
    }
    delete core;
}

bool xdna_rec_core_begin(xdna_rec_core * core, const void * feed0,
                         const float * gamma) {
    if (!core || !feed0 || !gamma) {
        return false;
    }
    upload(core->feed, feed0);
    // Seed the azg gamma + hh lanes (host-written once per layer); the attn
    // lanes are overwritten by the gdn stage every token and the z lanes by
    // every run.
    const int64_t azg_head = (int64_t) xdna_rec_pack::AZGN;
    float * azg = (float *) core->azg->bo.map();
    std::memset(azg, 0, (size_t) core->g.azg_bytes);
    for (int h = 0; h < xdna_rec_pack::NVH; h++) {
        float * head = azg + h * azg_head;
        std::memcpy(head + 2 * xdna_rec_pack::SV, gamma,
                    (size_t) xdna_rec_pack::SV * sizeof(float));
        head[3 * xdna_rec_pack::SV] = (float) h;   // hh tail
    }
    xdna_buffer_sync_to_device(core->azg);
    core->token = 0;
    return true;
}

bool xdna_rec_core_seed(xdna_rec_core * core, const void * state) {
    if (!core || !core->gstate) {
        return false;
    }
    const size_t n = (size_t) core->g.state_bytes / sizeof(float);   // state floats
    if (state == nullptr) {
        std::memset(core->gstate->bo.map(), 0, (size_t) core->g.state_bytes / 2);
        xdna_buffer_sync_to_device(core->gstate);
        return true;
    }
    const float * src = (const float *) state;
    uint16_t * dst = (uint16_t *) core->gstate->bo.map();
    for (size_t i = 0; i < n; i++) {
        dst[i] = xdna_bf16(src[i]);
    }
    xdna_buffer_sync_to_device(core->gstate);
    return true;
}

const void * xdna_rec_core_out(const xdna_rec_core * core) {
    // Where the fused projection drains: an argument of its own.
    return core && core->so_gemv ? core->so_gemv->o->bo.map() : nullptr;
}

size_t xdna_rec_core_out_off(const xdna_rec_core * core) {
    return core ? core->gbuf_out_off : 0;
}

const void * xdna_rec_core_act(const xdna_rec_core * core) {
    // The gated stage drains the activation into the projection's own
    // argument, so that is where it is read from.
    return core && core->so_gemv ? core->so_gemv->a->bo.map() : nullptr;
}

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
bool xdna_rec_core_fuse_so(xdna_rec_core * core, xdna_kernel_pool * pool,
                           xdna_gemv * so, xdna_gemv_pair * ffn) {
    if (!core || !so || core->xclbin.empty()) {
        return false;
    }
    if (!core->pkv_onchip) {
        return false;   // argument 2 still carries the pkv round trip
    }
    // The core's columns hand out descriptors from zero and the widest of them
    // takes eight, so the projection starts at eight and has the other half of
    // the file. Its arguments follow the core's six, and its activation is not
    // a buffer of its own at all - it is the window of the core's output that
    // the gated stage has just written, so the fused stream reads it where it
    // lies instead of copying it out and back.
    xdna_seq seq = core->base_seq;
    // Two rules decide this mapping. Only the low argument indices can be
    // patched - the same stream naming arguments six through eight times out
    // where naming zero through two runs - so the two halves share the core's
    // six. And an argument may carry only one form of patch: the gated
    // output's drain uses the bit31 form, so nothing else may touch argument
    // 5, which is why the gated stage sends its activation elsewhere.
    xdna_gemv_seq_opts opt;
    opt.bd_base  = -1;
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
    if (ffn && xdna_gemv_pair_raw_act(ffn)) {
        opt.out_off = XDNA_GEMV_ACT_TILE;
        opt.out_stream_stride = XDNA_GEMV_ACT_TILE;
        core->so_to_act = true;
    }
    if (!xdna_gemv_seq_build(&seq, so->geom, &opt)) {
        return false;
    }
    // The FFN is a dispatch of its own after this one: appending it to the
    // same stream needs its activation's host half written before the
    // dispatch that drains acc into the same buffer, and that order does not
    // hold on this platform.
    core->ffn_fused = false;
    const std::vector<uint32_t> words = xdna_seq_build(&seq);
    if (words.empty()) {
        return false;
    }
    const std::string stem =
        std::filesystem::path(core->xclbin).stem().string();
    // Through the pool, under a name every layer shares: the stream is the
    // same for all of them - only the buffers a run binds differ - and a
    // kernel of one's own per layer is twenty-eight hardware contexts created
    // lazily in the middle of NPU work. That is what made the fused dispatch
    // time out at random: the same stream would run or hang from one process
    // to the next.
    core->so_kern = pool ? xdna_kernel_pool_get_built(pool, "rec_core_so",
                                                      stem.c_str(),
                                                      words.data(), words.size())
                         : nullptr;
    if (!core->so_kern) {
        return false;
    }
    // One buffer for the projection: [activation | weights | output].
    if (!core->gbuf) {
        core->gbuf = xdna_buffer_alloc(core->dev,
                                       (size_t) so->geom.act_bytes() +
                                           so->geom.weight_bytes() +
                                           (size_t) so->geom.out_bytes());
        if (!core->gbuf) {
            return false;
        }
        std::memcpy((uint8_t *) core->gbuf->bo.map() + so->geom.act_bytes(),
                    so->w->bo.map(), so->geom.weight_bytes());
    }
    core->gbuf_out_off = (size_t) opt.out_off;
    xdna_buffer * tiles = core->so_to_act ? xdna_gemv_pair_act_buf(ffn) : so->o;
    xdna_buffer * a10[10] = { core->feed, core->x, core->pkvb, core->gstate,
                              core->azg, core->out, so->w, so->a, tiles,
                              tiles };
    core->so_run = xdna_kernel_run_make(core->so_kern, a10, 9);
    // The projection reads its activation from the window of this buffer that
    // the gated stage writes in the same dispatch. On the first token there is
    // nothing there yet, and a zero tile count means the cores never take the
    // weights the stream has already pushed - so the dispatch waits forever on
    // transfers that cannot drain. Seed a valid empty activation once.
    {
        xdna_buffer * ab = so->a;
        uint8_t * m = (uint8_t *) ab->bo.map() + opt.act_off;
        std::memset(m, 0, (size_t) so->geom.act_bytes());
        int32_t * h = (int32_t *) m;
        h[0] = so->geom.n_tiles();
        h[1] = so->geom.n_out();
        for (int t = 0; t <= so->geom.n_tiles(); t++) {
            int32_t * tl = (int32_t *) (m + (size_t) t * XDNA_GEMV_ACT_TILE);
            tl[XDNA_GEMV_ACT_TILE / 4 - 1] =
                so->geom.fmt == XDNA_WFMT_Q4G32 ? 0 : 1;
            tl[XDNA_GEMV_ACT_TILE / 4 - 2] =
                (t == so->geom.n_tiles() && so->geom.epilogue) ? 1 : 0;
        }
        xdna_buffer_sync_to_device(ab);
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

bool xdna_rec_core_run(xdna_rec_core * core, int t, const void * qkv,
                       const void * x, const float * z, int8_t * aq,
                       float * d_a) {
    using namespace xdna_rec_pack;
    if (!core || !core->kern || t != core->token || !x || !z || !aq || !d_a) {
        GGML_LOG_ERROR("%s: core run: bad call (t=%d token=%d)\n", "xdna-rec", t,
                       core ? core->token : -1);
        return false;
    }
    if (t > 0) {
        if (!qkv) {
            GGML_LOG_ERROR("%s: core run: t>0 needs the qkv window\n", "xdna-rec");
            return false;
        }
        // Upload only the F_Q qkv window of each conv block: the F_H history
        // (device conv state) is never written by the host after t=0.
        char * map = (char *) core->feed->bo.map();
        const char * src = (const char *) qkv;
        const size_t win_bytes = (size_t) core->g.sv * sizeof(float);
        for (int64_t b = 0; b < core->g.feed_blocks; b++) {
            const size_t off = (size_t) (b * core->g.feed_block_floats + core->g.feed_qkv_off) * 4;
            std::memcpy(map + off, src + b * win_bytes, win_bytes);
            core->feed->bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, win_bytes, off);
        }
    }
    upload(core->x, x);
    // Upload the per-token z into the azg z lanes (head h at h*AZGN + SV).
    // The host map keeps the seeded gamma/hh and zero attn lanes; the attn
    // lanes are overwritten by the gdn stage before the gated phase reads them,
    // so a full-buffer sync is safe.
    float * azg = (float *) core->azg->bo.map();
    for (int h = 0; h < NVH; h++) {
        std::memcpy(azg + (int64_t) h * AZGN + SV, z + (int64_t) h * SV,
                    (size_t) SV * sizeof(float));
    }
    xdna_buffer_sync_to_device(core->azg);

    // The fused run (conv+norm+gdn+gated, and the ssm_out projection when it
    // is appended): the kernel reads the feed, x tails and the azg z lanes,
    // updates the persistent bf16 state in place, drains the per-head attn
    // into the azg attn lanes and runs the gated epilogue, draining aq + d_a
    // into out. One dispatch for the core and the projection: the stream waits
    // on the gated drain before the projection's activation fill reads it.
    xdna_buffer * args[6] = { core->feed, core->x, core->pkvb, core->gstate,
                              core->azg, core->out };
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

    // out layout: [gated f32 scratch KGATE*4][aq int8 KGATE][d_a f32]
    xdna_buffer_sync_from_device(core->out);
    if (core->so_fused) {
        xdna_buffer_sync_from_device(core->x);
    }
    const uint8_t * omap = (const uint8_t *) core->out->bo.map();
    std::memcpy(aq, omap + KGATE * 4, (size_t) KGATE);
    std::memcpy(d_a, omap + KGATE * 4 + KGATE, sizeof(float));
    core->token = t + 1;
    return true;
}

