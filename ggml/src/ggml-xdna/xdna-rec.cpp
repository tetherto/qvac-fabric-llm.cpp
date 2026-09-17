#include "xdna-rec.h"
#include "xdna-seq.h"
#include "xdna-gemv.h"
#include "xdna-rec-post.h"
#include "xdna-verify.h"

#include "ggml.h"
#include "ggml-quants.h"

#include <algorithm>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>

#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/xrt_hw_context.h>
#include <fstream>
#include <string>

// Fused recurrent-layer runner for the Qwen3.5 gated-delta-net decode (see
// xdna-rec.h). The BO lifecycles and kernel arg layouts are 1:1 ports of the
// persistent python designs in kernels/attn_gdn_gated.py, attn_so_mmul.py and
// ffn_layer_mmul.py, so feeding the same layer weights produces the same
// device BO contents the python persist harness validated.

static bool read_file(const char * path, std::vector<uint8_t> & buf) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "xdna-rec: cannot open %s\n", path);
        return false;
    }
    f.seekg(0, std::ios::end);
    const std::streamoff n = f.tellg();
    f.seekg(0, std::ios::beg);
    buf.resize((size_t) n);
    if (n > 0 && !f.read((char *) buf.data(), n)) {
        fprintf(stderr, "xdna-rec: short read on %s\n", path);
        return false;
    }
    return true;
}

static void upload(xdna_buffer * b, const void * data) {
    std::memcpy(b->bo.map(), data, b->bytes);
    xdna_buffer_sync_to_device(b);
}


static void read_buffer(xdna_buffer * b, void * host, size_t bytes) {
    xdna_buffer_sync_from_device(b);
    std::memcpy(host, b->bo.map(), bytes);
}

// Per-stage timing of the fused runs (GGML_XDNA_REC_TIME=1). `st` carries the
// stage name -> ms deltas of one call.
static bool rec_time_on(void) {
    return getenv("GGML_XDNA_REC_TIME") != nullptr;
}

struct rec_stage {
    std::chrono::steady_clock::time_point t0;
    double ms = 0.0;
};

static void rec_stage_begin(rec_stage & s) {
    s.t0 = std::chrono::steady_clock::now();
}

static void rec_stage_end(rec_stage & s) {
    s.ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - s.t0).count();
}

// fp32 -> bf16 (round-to-nearest-even), same as the xdna-rec seed path.
static uint16_t f32_to_bf16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint16_t lsb = (uint16_t) ((x >> 16) & 1u);
    const uint32_t rem = x & 0xFFFFu;
    uint32_t r = (x >> 16) & 0xFFFFu;
    if (rem > 0x8000u || (rem == 0x8000u && lsb)) {
        r += 1;
    }
    return (uint16_t) r;
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

// Per-stage timing of the fused core, accumulated over the process and
// printed once at exit. Splitting the token into four dispatches costs a
// little in itself, so the shares matter and the total does not.
struct xdna_attn_phase_prof_t {
    double   us[4] = {};
    uint64_t runs  = 0;

    ~xdna_attn_phase_prof_t() {
        if (runs == 0) {
            return;
        }
        static const char * names[4] = { "conv", "norm", "gdn", "gated" };
        const double n = (double) runs;
        double tot = 0.0;
        for (int i = 0; i < 4; i++) {
            tot += us[i] / n;
        }
        fprintf(stderr, "xdna-attn-phase: layer-runs=%llu",
                (unsigned long long) runs);
        for (int i = 0; i < 4; i++) {
            fprintf(stderr, "  %s=%.0fus (%.0f%%)", names[i], us[i] / n,
                    100.0 * (us[i] / n) / tot);
        }
        fprintf(stderr, "  total=%.0fus\n", tot);
    }
};

static xdna_attn_phase_prof_t & xdna_attn_phase_prof_get(void) {
    static xdna_attn_phase_prof_t p;
    return p;
}

// The appended projection has arguments of its own - the weights on 6, the
// activation on 7, the output on 8. GGML_XDNA_SO_ARGS=0 packs all three into
// the one argument the on-chip pkv path freed instead, which is what this had
// to do while a stream naming arguments this high timed out - the missing DDR
// aperture on every patch from argument five up (xdna-seq.cpp). Separate
// arguments also drop the copy of the weights into that shared buffer:
// tg64 30.26 -> 30.69.
// GGML_XDNA_FFN_FUSED=1 puts the FFN in the core's own stream, one dispatch a
// layer. Off: the stream runs, but its activation's host half has to be
// written before the dispatch that drains acc into the same buffer, and that
// order does not hold on this platform (see HANDOFF.md).
static int ffn_on(void) {
    static const int v = [] {
        const char * e = getenv("GGML_XDNA_FFN_FUSED");
        return e ? atoi(e) : 0;
    }();
    return v;
}

static int so_sep_args(void) {
    static const int v = [] {
        const char * e = getenv("GGML_XDNA_SO_ARGS");
        return e ? atoi(e) : 1;
    }();
    return v;
}

static bool xdna_attn_phase_prof(void) {
    static const bool on = getenv("GGML_XDNA_ATTN_PHASE_PROF") != nullptr;
    return on;
}

// The FFN transition on the array, inside this design so it shares the one
// context (a second xclbin for it breaks the next fused dispatch): the
// residual add, the norm, the gamma and the quantized activation the FFN
// reads.
// The in-design post tile's endpoints. They are not pinned in the artifact
// the backend runs: the merged fused_layer leaves the placer to choose (the
// pinning in attn_gdn_gated.py applies to that design standalone), and the
// placer puts both on column 3 - the fill on MM2S ch1, the drain on S2MM ch1.
// Read them out of the artifact rather than assuming: decoding its compiled
// stream with and without POST_TILE leaves exactly those two transfers as the
// difference. The columns this used to carry, 1 and 2, are the standalone
// design's, and driving them on the merged one is why the transfers never
// completed.
static int post_col(const char * env, int dflt) {
    const char * e = getenv(env);
    return e ? atoi(e) : dflt;
}
#define POST_FILL_COL post_col("GGML_XDNA_POST_FILL_COL", 3)
#define POST_DRN_COL  post_col("GGML_XDNA_POST_DRN_COL", 3)
static const int POST_D   = 1024;
// The post output object depends on the activation form the model needs
// (GATED_FMT=1: 8-bit, tiles of 128 - keep in step with attn_gdn_gated.py).
#ifndef GGML_XDNA_GATED_FMT
#define GGML_XDNA_GATED_FMT 1
#endif
static const int POST_OUT_BYTES = POST_D * 4 +
    (1 + POST_D / (GGML_XDNA_GATED_FMT == 1 ? 128 : 256)) * 2112;
// The post windows live past the state window of the state argument (arg 3,
// plainly patched like every other patch on it): the bit31 form the gated
// output argument carries is only reliable at offset zero, and the post
// needs the gated output itself out of the way anyway.
static const uint32_t POST_OFF = 65536;

static bool xdna_rec_post_seq_build(xdna_seq * seq);

static bool post_on(void) {
    // GGML_XDNA_POST_FUSED: the FFN transition runs as its own dispatch on
    // the SAME xclbin (in-design post tile, no second design or context) -
    // xdna_rec_core_post_run drives it.
    static const int v = [] {
        const char * e = getenv("GGML_XDNA_POST_FUSED");
        return e != nullptr && atoi(e) != 0;
    }();
    return v != 0;
}

// The whole recurrent layer as a full ELF (kernels/fused_layer.py --elf-path).
// The xclbin path binds a host-built instruction stream and can only patch six
// arguments, which is why the projection has to live inside one of them at
// offsets; the ELF names all nine of the design's buffers directly.
//
// GGML_XDNA_CORE_ELF=1 binds the core's own pkvb for argument 2 and the
// projection's activation for argument 7. =2 binds the projection's activation
// for both, which is what the handover needs if the gated stage drains its
// activation into the argument the on-chip pkv path freed - the same place the
// patched stream puts it.
struct xdna_rec_core_elf {
    std::shared_ptr<xrt::elf>         elf;
    std::shared_ptr<xrt::hw_context>  ctx;
    std::shared_ptr<xrt::ext::kernel> kern;
};

static int core_elf_on(void) {
    static const int v = [] {
        const char * e = getenv("GGML_XDNA_CORE_ELF");
        return e ? atoi(e) : 0;
    }();
    return v;
}

static std::string core_elf_path(void) {
    for (const auto & dir : xdna_kernel_search_dirs()) {
        std::error_code ec;
        const std::string f = (dir / "fused_layer_full.elf").string();
        if (std::filesystem::exists(f, ec)) {
            return f;
        }
    }
    return "";
}

xdna_rec_core * xdna_rec_core_create(xdna_device * dev, const xdna_rec_geom & g,
                                     const char * xclbin) {
    // The merged conv+norm+gdn core runs on attn_gdn_txn.xclbin
    // (kernels/attn_gdn_txn.py, barrier-free persistent workers) driven by a
    // host-built per-token TXN stream (schedule 2 = phased) instead of an
    // IRON-compiled stream. The worker geometry and the {feed, x, pkvb,
    // gstate, attn} BO set are fixed by xdna_rec_pack_geom; only the bound
    // instruction words differ from a compiled .insts.bin.
    if (!g.valid() || !xclbin) {
        fprintf(stderr, "xdna-rec: invalid core create args\n");
        return nullptr;
    }
    xdna_seq seq;
    xdna_attn_gdn_geom tgeom;
    tgeom.azg_n     = xdna_rec_pack::AZGN;      // fused azg head layout
    tgeom.out_words = (int) (g.out_bytes / 4);  // gated out object (words)
    // GGML_XDNA_ATTN_GDN_SCHED: 0 = IRON column-major, 1 = slot-major,
    // 2 = phased (groups of 2), 4 = the same with four gdn rounds in flight
    // before a wait. Default 4. End to end, repeated runs of tg64:
    //   0 = 18.9   1 = 19.9   2 = 20.3   4 = 21.3
    // The per-layer figures the profiler reports are too noisy to rank these -
    // one run of it put 2 ahead of 4 by 25%, which repeating tg64 does not
    // bear out at all.
    //
    // 5 and 6 exist to answer what the core's ~2000 us per layer is made of,
    // and both say it is not the stream. 5 gives the conv slots a descriptor
    // bank, so several are in flight before any wait; 6 collapses a column's
    // 24 feed transfers into one descriptor. Neither moves the number:
    //   4 = 21.33 / 21.26    5 = 21.25 / 21.31    6 = 21.20 / 21.22
    // So it is neither token-wait latency nor the stream's instruction count,
    // and the DDR traffic of a layer-run is about a megabyte, some 18 us at
    // the rate the GEMV path measures. What is left is the kernels themselves,
    // on a design whose stages run one after another and never have more than
    // four of its nine cores busy at once.
    // Must match the CONV_GPO the artifact was built with (see
    // kernels/attn_gdn_gated.py): it decides the object the stage's fifos
    // carry, and therefore the shape of every conv descriptor.
    {
        const char * v = getenv("GGML_XDNA_CONV_GPO");
        tgeom.conv_gpo = v ? atoi(v) : 4;
        const char * po = getenv("GGML_XDNA_PKV_ONCHIP");
        tgeom.pkv_onchip = po ? atoi(po) : 1;
        const char * ao = getenv("GGML_XDNA_ATTN_ONCHIP");
        tgeom.attn_onchip = ao ? atoi(ao) : 1;
        // GGML_XDNA_OUT_BASE drains the gated output at this byte offset of
        // its argument instead of zero, so that it can be patched plainly like
        // every other drain in this stream rather than with the bit31 form.
        // It does not work: at any non-zero offset the output never lands and
        // the decode is wrong from the first token. This drain wants the bit31
        // form, and that form wants offset zero. Leave at zero.
        const char * ob = getenv("GGML_XDNA_OUT_BASE");
        tgeom.out_base = ob ? atoi(ob) : 0;
        // Must match GATED_ACT_SPLIT in the artifact.
        const char * as = getenv("GGML_XDNA_ACT_SPLIT");
        if (as == nullptr || atoi(as) != 0) {
            tgeom.gated_act_bytes =
                (int) (g.out_bytes - xdna_rec_pack::ACT_OFF);
            tgeom.gated_act_col = 7;
            // Into the argument the on-chip pkv path freed, at offset zero:
            // argument 5 already carries the gated output's own drain, and
            // that one is patched with the bit31 form.
            tgeom.gated_act_arg = so_sep_args() ? 7 : 2;
            tgeom.gated_act_off = 0;
            if (const char * aa = getenv("GGML_XDNA_ACT_ARG")) {
                tgeom.gated_act_arg = atoi(aa);
                tgeom.gated_act_off = tgeom.gated_act_arg == 5
                                          ? (int) xdna_rec_pack::ACT_OFF : 0;
            }
        }
        const char * ss = getenv("GGML_XDNA_GDN_STREAMS");
        tgeom.gdn_state_streams = ss ? atoi(ss) : 1;
        const char * fs = getenv("GGML_XDNA_CONV_SLOT");
        tgeom.feed_slot = fs && atoi(fs) ? atoi(fs) : tgeom.feed_n;
    }
    const char * sched_env = getenv("GGML_XDNA_ATTN_GDN_SCHED");
    const int sched = sched_env ? atoi(sched_env) : 4;
    if (!xdna_attn_gdn_txn_build(&seq, &tgeom, sched, -1)) {
        fprintf(stderr, "xdna-rec: attn_gdn_gated stream build failed\n");
        return nullptr;
    }
    std::vector<uint32_t> words = xdna_seq_build(&seq);
    if (words.empty()) {
        fprintf(stderr, "xdna-rec: attn_gdn_gated stream empty\n");
        return nullptr;
    }
    // GGML_XDNA_CORE_IRON_INSTS=1 runs the artifact's own compiled stream in
    // place of the one built above. The design's seq_fn is maintained by hand
    // as a mirror of this builder and nothing ever ran it: the standalone
    // harness needs reference modules that are not in the tree, and a probe
    // that binds zero-filled buffers times out on the host-built stream too,
    // so it cannot tell a wrong sequence from an unprepared one. This is the
    // only way to drive seq_fn with the buffers a real token has, and whether
    // it completes decides whether the full-ELF build of seq_fn is worth
    // anything - full-ELF compiles that same sequence.
    // The value may be a path, because the artifact's own insts.bin is built
    // for the default projection geometry and the buffers a real layer binds
    // are not that shape - a mismatch stalls for a reason that has nothing to
    // do with the sequence being right or wrong.
    if (const char * e = getenv("GGML_XDNA_CORE_IRON_INSTS")) {
        if (atoi(e) != 0 || e[0] == '/') {
            const std::filesystem::path ip =
                e[0] == '/' ? std::filesystem::path(e)
                            : std::filesystem::path(xclbin).replace_extension(".insts.bin");
            std::ifstream f(ip, std::ios::binary | std::ios::ate);
            if (!f) {
                fprintf(stderr, "xdna-rec: no IRON insts at %s\n", ip.c_str());
                return nullptr;
            }
            const std::streamsize n = f.tellg();
            f.seekg(0);
            words.assign((size_t) n / 4, 0);
            f.read((char *) words.data(), n);
            fprintf(stderr, "xdna-rec: using the IRON stream (%zu words) from %s\n",
                    words.size(), ip.c_str());
        }
    }
    // GGML_XDNA_ATTN_DUMP=<path> writes the built stream so it can be
    // disassembled next to the one IRON compiles for the same design.
    if (const char * path = getenv("GGML_XDNA_ATTN_DUMP")) {
        std::ofstream f(path, std::ios::binary);
        f.write((const char *) words.data(), (std::streamsize) (words.size() * 4));
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
    core->out_base   = tgeom.out_base;
    core->act_arg    = tgeom.gated_act_bytes ? tgeom.gated_act_arg : 5;
    core->kern = xdna_kernel_load_hw(dev, xclbin);
    if (!core->kern ||
        !xdna_kernel_bind_insts(dev, core->kern, words.data(), words.size())) {
        xdna_rec_core_free(core);
        return nullptr;
    }

    if (core_elf_on()) {
        const std::string ep = core_elf_path();
        if (ep.empty()) {
            fprintf(stderr, "xdna-rec: fused_layer_full.elf not found\n");
            xdna_rec_core_free(core);
            return nullptr;
        }
        core->elf = new xdna_rec_core_elf;
        try {
            core->elf->elf = std::make_shared<xrt::elf>(ep);
            core->elf->ctx = std::make_shared<xrt::hw_context>(
                dev->device, *core->elf->elf, xrt::hw_context::cfg_param_type{});
            core->elf->kern = std::make_shared<xrt::ext::kernel>(
                *core->elf->ctx, "main:sequence");
            fprintf(stderr, "xdna-rec: the layer runs as a full ELF (%s)\n",
                    ep.c_str());
        } catch (const std::exception & e) {
            fprintf(stderr, "xdna-rec: full-ELF load failed: %s\n", e.what());
            xdna_rec_core_free(core);
            return nullptr;
        }
    }

    if (post_on()) {
        // The post stream on the same xclbin: its own instruction words,
        // the shared hardware context the core's kernel already holds.
        xdna_seq pseq;
        if (!xdna_rec_post_seq_build(&pseq)) {
            fprintf(stderr, "xdna-rec: post stream build failed\n");
            xdna_rec_core_free(core);
            return nullptr;
        }
        const std::vector<uint32_t> pw = xdna_seq_build(&pseq);
        if (const char * path = getenv("GGML_XDNA_POST_DUMP")) {
            std::ofstream f(path, std::ios::binary);
            f.write((const char *) pw.data(),
                    (std::streamsize) (pw.size() * 4));
        }
        core->post_kern = xdna_kernel_load_hw(dev, xclbin);
        if (!core->post_kern || pw.empty() ||
            !xdna_kernel_bind_insts(dev, core->post_kern, pw.data(), pw.size())) {
            fprintf(stderr, "xdna-rec: post stream bind failed\n");
            xdna_rec_core_free(core);
            return nullptr;
        }
    }

    if (xdna_attn_phase_prof()) {
        static const char * names[4] = { "conv", "norm", "gdn", "gated" };
        for (int ph = 0; ph < 4; ph++) {
            xdna_seq pseq;
            if (!xdna_attn_gdn_txn_build(&pseq, &tgeom, sched, ph)) {
                fprintf(stderr, "xdna-rec: phase %s stream build failed\n", names[ph]);
                xdna_rec_core_free(core);
                return nullptr;
            }
            const std::vector<uint32_t> pw = xdna_seq_build(&pseq);
            core->phase_kern[ph] = xdna_kernel_load_hw(dev, xclbin);
            if (!core->phase_kern[ph] || pw.empty() ||
                !xdna_kernel_bind_insts(dev, core->phase_kern[ph], pw.data(), pw.size())) {
                fprintf(stderr, "xdna-rec: phase %s bind failed\n", names[ph]);
                xdna_rec_core_free(core);
                return nullptr;
            }
        }
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
    core->out = xdna_buffer_alloc(dev,
                                  (size_t) g.out_bytes + 8192 +
                                      (size_t) core->out_base +
                                      3 * (size_t) POST_D * 4 + 16 +
                                      (size_t) POST_OUT_BYTES);
    if (!core->feed || !core->x || !core->pkvb || !core->gstate || !core->azg ||
        !core->out) {
        fprintf(stderr, "xdna-rec: core BO allocation failed\n");
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
        dst[i] = f32_to_bf16(src[i]);
    }
    xdna_buffer_sync_to_device(core->gstate);
    return true;
}

size_t xdna_rec_core_act_off(void) {
    return (size_t) xdna_rec_pack::ACT_OFF;
}

struct xdna_rec_post * xdna_rec_core_post_get(xdna_rec_core * core,
                                              xdna_kernel_pool * pool) {
    if (!core) {
        return nullptr;
    }
    if (!core->post) {
        core->post = xdna_rec_post_create(pool);
    }
    return core->post;
}


void * xdna_rec_core_post_in(const xdna_rec_core * core) {
    return core && core->out && core->post_in_off
        ? (uint8_t *) core->out->bo.map() + core->post_in_off
        : nullptr;
}

const void * xdna_rec_core_post_out(const xdna_rec_core * core) {
    return core && core->out && core->post_out_off
        ? (const uint8_t *) core->out->bo.map() + core->post_out_off
        : nullptr;
}

void xdna_rec_core_post_sync(xdna_rec_core * core) {
    if (core && core->out && core->post_in_off) {
        // Only the residual, the gamma and the flag word.
        xdna_buffer_sync_to_device_range(core->out,
                                         2 * (size_t) POST_D * 4 + 16,
                                         core->post_in_off);
    }
}

// The post stream: the fill and the drain of the in-design post tile, as
// their own instruction stream on the SAME xclbin the core runs - so the
// transition needs no second design (and no second context, which is what
// used to kill the fused dispatch). Both transfers touch the gated-output
// argument at offset zero with the bit31 patch form, exactly like the
// design's own sequence.
static bool xdna_rec_post_seq_build(xdna_seq * seq) {
    int pstages = 0;
    if (const char * ps = getenv("GGML_XDNA_POST_STAGES")) {
        pstages = atoi(ps);
    }
    const uint32_t bdf = 0;
    xdna_bd fbd;
    fbd.buf_len   = 3 * (uint32_t) POST_D + 4;   // words: one f32 per word
    fbd.buf_off   = POST_OFF;   // mirror the patch offset (the core streams do)
    fbd.d0_stride = 1;
    fbd.d1_stride = 1;
    fbd.d2_stride = 1;
    fbd.ax_cache  = 2;
    xdna_seq_blockwrite(seq, POST_FILL_COL, 0, bdf, &fbd);
    xdna_seq_ddr_patch(seq, POST_FILL_COL, 0, bdf, 3, POST_OFF);
    xdna_seq_push_queue(seq, POST_FILL_COL, 0, bdf, xdna_dma_dir::MM2S,
                        1, false, 0);
    if (pstages != 1) {
        const uint32_t bdd = 0;
        xdna_bd dbd;
        dbd.buf_len   = (uint32_t) POST_OUT_BYTES / 4;   // words: ui8 bytes / 4
        dbd.buf_off   = POST_OFF;   // mirror the patch offset
        dbd.d0_stride = 1;
        dbd.d1_stride = 1;
        dbd.d2_stride = 1;
        dbd.ax_cache  = 2;
        xdna_seq_blockwrite(seq, POST_DRN_COL, 0, bdd, &dbd);
        xdna_seq_ddr_patch(seq, POST_DRN_COL, 0, bdd, 3, POST_OFF);
        // The design's own sequence sets the channel controller's packet id
        // (the tct-route marker) before the push; mirror it exactly.
        xdna_seq_maskwrite(seq, POST_DRN_COL, 0, 0x1D208, 0xF00, 0x1F00);
        xdna_seq_push_queue(seq, POST_DRN_COL, 0, bdd, xdna_dma_dir::S2MM,
                            1, true, 0);
        xdna_seq_wait_token(seq, POST_DRN_COL, 0, xdna_dma_dir::S2MM, 1);
    }
    return true;
}

// The post dispatch on the core's own xclbin. The host has assembled the
// post input at the front of the out buffer (offset zero) before the call;
// on return the front holds [hattn][header][activation tiles].
bool xdna_rec_core_post_run(xdna_rec_core * core) {
    if (!core || !core->post_kern) {
        fprintf(stderr, "xdna-rec: post run: no post kernel\n");
        return false;
    }
    // GGML_XDNA_POST_DELAY_MS: settle delay before the post dispatch - the
    // race bisection switch (does the preceding dispatch's tail work have to
    // finish first?).
    if (const char * dms = getenv("GGML_XDNA_POST_DELAY_MS")) {
        std::this_thread::sleep_for(std::chrono::milliseconds(atoi(dms)));
    }
    xdna_buffer_sync_to_device_range(core->gstate,
                                     (3 * (size_t) POST_D + 4) * 4, POST_OFF);
    xdna_buffer * args[6] = { core->feed, core->x, core->pkvb, core->gstate,
                              core->azg, core->out };
    if (getenv("GGML_XDNA_POST_TRACE")) {
        fprintf(stderr, "xdna-post: args");
        for (int i = 0; i < 6; i++) {
            fprintf(stderr, " [%d]=%zu@%p", i, args[i]->bytes,
                    (void *) args[i]->bo.address());
        }
        fprintf(stderr, "\n");
    }
    xrt::run run = xdna_kernel_run_start(core->post_kern, args, 6);
    if (!xdna_run_wait(run)) {
        fprintf(stderr, "xdna-rec: post run failed\n");
        return false;
    }
    xdna_buffer_sync_from_device(core->gstate);
    return true;
}

const void * xdna_rec_core_out(const xdna_rec_core * core) {
    // Where a fused projection drains: its own argument when it has one,
    // otherwise behind its weights in the argument the on-chip pkv path freed.
    if (!core) {
        return nullptr;
    }
    if (so_sep_args()) {
        return core->so_gemv ? core->so_gemv->o->bo.map() : nullptr;
    }
    return core->gbuf ? core->gbuf->bo.map() : nullptr;
}

size_t xdna_rec_core_out_off(const xdna_rec_core * core) {
    return core ? core->gbuf_out_off : 0;
}

const void * xdna_rec_core_act(const xdna_rec_core * core) {
    if (core && so_sep_args() && core->so_gemv) {
        // The gated stage drains the activation into the projection's own
        // argument, so that is where it is read from as well.
        return core->so_gemv->a->bo.map();
    }
    return core && core->out
        ? (core->act_arg == 2
               ? (const uint8_t *) (core->so_fused ? core->gbuf : core->pkvb)
                     ->bo.map()
               : (const uint8_t *) core->out->bo.map() + core->out_base +
                     xdna_rec_pack::ACT_OFF)
        : nullptr;
}

// The core and the ssm_out projection as one dispatch. Two dispatches measure
// 282 + 109 us a layer where the merged one measures 339, which is the array's
// fixed per-phase cost paid once instead of twice: 33.6 -> 34.4 t/s.
//
// GGML_XDNA_SO_FUSED=0 keeps them apart; 6 runs the same stream as its own
// dispatch, and GGML_XDNA_SO_BD_BASE / _FUSED_DUMP are the rest of what it
// took to get here.
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
static int so_fuse_on(void) {
    static const int v = [] {
        const char * e = getenv("GGML_XDNA_SO_FUSED");
        return e ? atoi(e) : 1;
    }();
    return v;
}

bool xdna_rec_core_fuse_so(xdna_rec_core * core, xdna_kernel_pool * pool,
                           xdna_gemv * so, uint32_t act_off,
                           xdna_gemv_pair * ffn) {
    if (!core || !so || !so_fuse_on() || core->xclbin.empty()) {
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
    // SO_FUSED=4 builds the same stream with the same argument and descriptor
    // mapping but on its own, dispatched after the core's: if that runs, what
    // the projection cannot survive is being appended, not where its
    // arguments sit.
    // SO_FUSED=6 builds the same six-argument mapping on its own stream,
    // dispatched after the core's, to tell the mapping from the append.
    xdna_seq seq;
    if (so_fuse_on() != 6) {
        seq = core->base_seq;
    }
    if (getenv("GGML_XDNA_SO_FUSED_LOG")) {
        fprintf(stderr, "xdna-rec: core descriptors per column:");
        for (int c = 0; c < 8; c++) {
            fprintf(stderr, " %u", seq.bd_used[c]);
        }
        fprintf(stderr, "\n");
    }
    // Two rules decide this mapping. Only the low argument indices can be
    // patched - the same stream naming arguments six through eight times out
    // where naming zero through two runs - so the two halves share the core's
    // six. And an argument may carry only one form of patch: the gated
    // output's drain uses the bit31 form, so nothing else may touch argument
    // 5, which is why the gated stage now sends its activation elsewhere.
    //
    // That leaves exactly one argument, the one the on-chip pkv path freed,
    // and everything the projection needs goes in it: the activation the
    // gated stage drains to its front, then the weights, then its own output.
    // All three plainly patched.
    xdna_gemv_seq_opts opt;
    opt.bd_base  = -1;
    if (const char * bb = getenv("GGML_XDNA_SO_BD_BASE")) {
        opt.bd_base = atoi(bb);
    }
    if (so_sep_args()) {
        opt.w_arg   = 6;
        opt.w_off   = 0;
        opt.act_arg = 7;
        opt.act_off = 0;
        // With the prologue's second input the design has one more buffer, so
        // the projection's output moves out to nine and eight carries the
        // host's half of the tiles.
        opt.o_arg   = xdna_gemv_act_pro_h() ? 9 : 8;
        opt.out_off = 0;
        // With the transition on the array, argument 8 is not an output buffer
        // at all - it is the FFN's activation, and the projection drains one
        // stream into each of its tiles. A stream carries og*rows*n_core = 256
        // columns and a tile holds K_TILE = 256, so they line up one to one,
        // and the first tile is the header. Nothing of the projection's result
        // reaches the host between the two dispatches after this.
        // GGML_XDNA_SO_TO_ACT=1. Off by default: the drain itself is correct
        // - the tiles hold exactly what the host path collects, checked value
        // for value - but something downstream of it is not, and the decode
        // text is wrong. See HANDOFF.md.
        static const bool to_act = []() {
            const char * e = getenv("GGML_XDNA_SO_TO_ACT");
            return e == nullptr || atoi(e) != 0;
        }();
        if (to_act && ffn && xdna_gemv_pair_raw_act(ffn)) {
            opt.out_off = XDNA_GEMV_ACT_TILE;
            opt.out_stream_stride = XDNA_GEMV_ACT_TILE;
            core->so_to_act = true;
        }
    } else {
        opt.w_arg    = 2;
        opt.w_off    = (uint32_t) so->geom.act_bytes();
        opt.act_arg  = 2;
        opt.act_off  = 0;
        opt.o_arg    = 2;
        opt.out_off  = (uint32_t) (so->geom.act_bytes() + so->geom.weight_bytes());
    }
    opt.stages   = so_fuse_on() >= 10 ? so_fuse_on() - 10 : 0;
    if (!xdna_gemv_seq_build(&seq, so->geom, &opt)) {
        return false;
    }
    // The transition to the FFN runs as its own dispatch on this same
    // design (xdna_rec_core_post_run): the gated output argument is read
    // back first, then the host assembles the post input in its window at
    // offset zero and the post stream fills/drains it there. Offsets zero
    // because the bit31 patch form - the one this argument carries - only
    // works at offset zero.
    // The FFN in the same stream, so the layer is one dispatch. The handover
    // it needs - the projection's result read back by the next stage - is a
    // drain and a completion token inside one stream, which is the pattern the
    // conv drains already use; between two dispatches the same handover needs
    // the host to make it visible and that is where it came apart.
    //
    // It fits without a new buffer. The on-chip pkv path leaves argument 2
    // free, so the FFN's weights go there as they are, and its output goes in
    // the tail of the activation buffer on argument 8 - which the projection
    // is already draining into.
    core->ffn_fused = false;
    if (ffn_on() && ffn && core->so_to_act) {
        int hi = 0;
        for (int c = 0; c < 8; c++) {
            hi = std::max(hi, (int) seq.bd_used[c]);
        }
        if (getenv("GGML_XDNA_SO_FUSED_LOG")) {
            fprintf(stderr, "xdna-rec: descriptors after core+projection:");
            for (int c = 0; c < 8; c++) {
                fprintf(stderr, " %u", seq.bd_used[c]);
            }
            fprintf(stderr, "  (high water %d of %d)\n", hi, 16);
        }
        xdna_gemv_pair_opts fo;
        fo.w_arg  = 2;
        fo.a_arg  = xdna_gemv_act_pro_h() ? 9 : 8;
        fo.o_arg  = fo.a_arg;
        fo.o_base = (uint32_t) xdna_gemv_pair_o_tail(ffn);
        if (xdna_gemv_act_pro_h()) {
            fo.h_arg  = 8;
            fo.h_base = 0;
        }
        // After everything the stream has already used, not from zero:
        // reprogramming a descriptor whose transfer has not drained loses it
        // silently, and the fills before this section carry no token to wait
        // on. GGML_XDNA_FFN_BD_BASE overrides it.
        fo.bd_base = hi;
        if (const char * bb = getenv("GGML_XDNA_FFN_BD_BASE")) {
            fo.bd_base = atoi(bb);
        }
        if (xdna_gemv_seq_build_pair(&seq, ffn->g1, ffn->g2,
                                     ffn->w2_off, ffn->a2_off, &fo)) {
            core->ffn_fused = true;
        } else {
            fprintf(stderr, "xdna-rec: the FFN does not append; keeping it a "
                            "dispatch of its own\n");
        }
    }
    const std::vector<uint32_t> words = xdna_seq_build(&seq);
    if (words.empty()) {
        return false;
    }
    // Through the pool, under a name every layer shares: the stream is the
    // same for all of them - only the buffers a run binds differ - and a
    // kernel of one's own per layer is twenty-eight hardware contexts created
    // lazily in the middle of NPU work. That is what made the fused dispatch
    // time out at random: the same stream would run or hang from one process
    // to the next.
    if (const char * path = getenv("GGML_XDNA_SO_FUSED_DUMP")) {
        static int n = 0;
        const std::string f = std::string(path) + "/so_fused_" +
                              std::to_string(n++) + ".insts.bin";
        std::ofstream o(f, std::ios::binary);
        o.write((const char *) words.data(),
                (std::streamsize) (words.size() * 4));
    }
    const std::string stem =
        std::filesystem::path(core->xclbin).stem().string();
    // One kernel shared by every layer whose stream is the same - a kernel of
    // its own per layer is twenty-eight hardware contexts created lazily in
    // the middle of NPU work. But the streams are not all the same once the
    // FFN is in them: this model's ffn_down is Q6_K in half its layers and
    // Q4_K in the other half, which is a different second geometry and a
    // different stream. The name has to say so, or the first layer's stream is
    // handed to a layer it does not fit and the dispatch stalls part way
    // through the token.
    std::string kname = "rec_core_so";
    if (core->ffn_fused && ffn) {
        kname += "+" + xdna_gemv_pair_key(ffn);
    }
    core->so_kern = pool ? xdna_kernel_pool_get_built(pool, kname,
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
                                           so->geom.out_bytes() +
                                           2 * (size_t) POST_D * 4 + 16 +
                                           (size_t) POST_OUT_BYTES);
        if (!core->gbuf) {
            return false;
        }
        std::memcpy((uint8_t *) core->gbuf->bo.map() + so->geom.act_bytes(),
                    so->w->bo.map(), so->geom.weight_bytes());
    }
    core->gbuf_out_off = (size_t) opt.out_off;
    if (so_sep_args()) {
        xdna_buffer * tiles = core->so_to_act ? xdna_gemv_pair_act_buf(ffn)
                                              : so->o;
        xdna_buffer * a10[10] = { core->feed, core->x, core->pkvb, core->gstate,
                                  core->azg, core->out, so->w, so->a, tiles,
                                  tiles };
        if (xdna_gemv_act_pro_h()) {
            // Eight is the prologue's second input, nine the tiles.
            a10[8] = xdna_gemv_pair_h_buf(ffn) ? xdna_gemv_pair_h_buf(ffn)
                                               : so->a;
            a10[9] = tiles;
        }
        if (core->ffn_fused) {
            // Argument 2 carries the FFN's weights now, not the pkv round trip
            // the on-chip path retired.
            a10[2] = xdna_gemv_pair_w_buf(ffn);
        }
        core->so_run = xdna_kernel_run_make(core->so_kern, a10,
                                            xdna_gemv_act_pro_h() ? 10 : 9);
    } else {
        xdna_buffer * args[6] = { core->feed, core->x, core->gbuf, core->gstate,
                                  core->azg, core->out };
        core->so_run = xdna_kernel_run_make(core->so_kern, args, 6);
    }
    if (!core->ref_words.empty()) {
        // The reference sequence needs its full argument set; the extras are
        // dummy buffers - the probe's own dispatch completed this way.
        xdna_buffer * rargs[20] = { core->feed, core->x, core->pkvb,
                                    core->gstate, core->azg, core->out };
        for (int i = 6; i < 20; i++) {
            rargs[i] = core->gbuf;
        }
        core->so_run = xdna_kernel_run_make(core->so_kern, rargs, 20);
    }
    // The projection reads its activation from the window of this buffer that
    // the gated stage writes in the same dispatch. On the first token there is
    // nothing there yet, and a zero tile count means the cores never take the
    // weights the stream has already pushed - so the dispatch waits forever on
    // transfers that cannot drain. Seed a valid empty activation once.
    {
        xdna_buffer * ab = so_sep_args() ? so->a : core->gbuf;
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
    const bool tmr = rec_time_on();
    const auto wall0 = std::chrono::steady_clock::now();
    rec_stage st_upload = {};
    rec_stage st_kern = {};
    rec_stage st_read = {};
    if (!core || !core->kern || t != core->token || !x || !z || !aq || !d_a) {
        fprintf(stderr, "xdna-rec: core run: bad call (t=%d token=%d)\n", t,
                core ? core->token : -1);
        return false;
    }
    rec_stage_begin(st_upload);
    if (t > 0) {
        if (!qkv) {
            fprintf(stderr, "xdna-rec: core run: t>0 needs the qkv window\n");
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
    rec_stage_end(st_upload);

    // attn_gdn_gated run (conv+norm+gdn+gated in one launch): the kernel reads
    // the feed, x tails, azg (z lanes), the persistent bf16 state, writes pkvb,
    // updates the state in place, drains the per-head attn into the azg attn
    // lanes and runs the gated epilogue, draining aq + d_a into out.
    // (opcode, insts, size, feed, x, pkvb, gstate, azg, out)
    xdna_buffer * args[6] = { core->feed, core->x, core->pkvb, core->gstate,
                              core->azg, core->out };
    rec_stage_begin(st_kern);
    if (core->elf) {
        // Nine arguments, the design's own: the six of the core and the
        // projection's three. A fresh run per dispatch - the ELF path binds
        // buffers and starts, there is no stream to restart.
        xdna_gemv * so = core->so_gemv;
        if (!so) {
            fprintf(stderr, "xdna-rec: full-ELF needs the projection's buffers\n");
            return false;
        }
        // The activation the projection reads is device-written, so before
        // the first token it holds nothing - and a zero tile count means the
        // cores never take the weights the sequence has already pushed, so
        // the dispatch waits on transfers that cannot drain. Seed a valid
        // empty one, exactly as the patched stream does into its gbuf.
        if (!core->elf_seeded) {
            uint8_t * m = (uint8_t *) so->a->bo.map();
            std::memset(m, 0, (size_t) so->geom.act_bytes());
            int32_t * h = (int32_t *) m;
            h[0] = so->geom.n_tiles();
            h[1] = so->geom.n_out();
            for (int i = 0; i <= so->geom.n_tiles(); i++) {
                int32_t * tl = (int32_t *) (m + (size_t) i * XDNA_GEMV_ACT_TILE);
                tl[XDNA_GEMV_ACT_TILE / 4 - 1] =
                    so->geom.fmt == XDNA_WFMT_Q4G32 ? 0 : 1;
                tl[XDNA_GEMV_ACT_TILE / 4 - 2] =
                    (i == so->geom.n_tiles() && so->geom.epilogue) ? 1 : 0;
            }
            xdna_buffer_sync_to_device(so->a);
            core->elf_seeded = true;
        }
        // =3 runs the same nine buffers through the xclbin instead, so that
        // with GGML_XDNA_CORE_IRON_INSTS=1 the design's own compiled stream
        // drives them. That separates the two things a stall can be: the
        // sequence full_elf compiles, or the full-ELF execution of it.
        if (core_elf_on() == 3) {
            xdna_buffer * a9[9] = { core->feed, core->x, core->pkvb,
                                    core->gstate, core->azg, core->out,
                                    so->w, so->a, so->o };
            xrt::run run = xdna_kernel_run_start(core->kern, a9, 9);
            if (!xdna_run_wait(run)) {
                fprintf(stderr, "xdna-rec: nine-argument stream did not complete\n");
                return false;
            }
            goto elf_done;
        }
        try {
            xrt::run run(*core->elf->kern);
            xdna_buffer * pkv = core_elf_on() >= 2 ? so->a : core->pkvb;
            run.set_arg(0, core->feed->bo);
            run.set_arg(1, core->x->bo);
            run.set_arg(2, pkv->bo);
            run.set_arg(3, core->gstate->bo);
            run.set_arg(4, core->azg->bo);
            run.set_arg(5, core->out->bo);
            run.set_arg(6, so->w->bo);
            run.set_arg(7, so->a->bo);
            run.set_arg(8, so->o->bo);
            run.start();
            if (run.wait(std::chrono::milliseconds(5000)) !=
                ERT_CMD_STATE_COMPLETED) {
                fprintf(stderr, "xdna-rec: full-ELF dispatch did not complete\n");
                return false;
            }
        } catch (const std::exception & e) {
            fprintf(stderr, "xdna-rec: full-ELF dispatch failed: %s\n", e.what());
            return false;
        }
    elf_done:;
    } else if (core->phase_kern[0]) {
        xdna_attn_phase_prof_t & pp = xdna_attn_phase_prof_get();
        for (int ph = 0; ph < 4; ph++) {
            const auto t0 = std::chrono::steady_clock::now();
            xrt::run prun = xdna_kernel_run_start(core->phase_kern[ph], args, 6);
            if (!xdna_run_wait(prun)) {
                return false;
            }
            pp.us[ph] += std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - t0).count();
        }
        pp.runs++;
    } else if (core->so_fused && so_fuse_on() == 6) {
        xrt::run run = xdna_kernel_run_start(core->kern, args, 6);
        if (!xdna_run_wait(run)) {
            return false;
        }
        if (!xdna_run_restart(core->so_run, "core+so") || !xdna_run_wait(core->so_run)) {
            return false;
        }
    } else if (core->so_fused) {
        // One dispatch for the core and the projection: the stream waits on
        // the gated drain before the projection's activation fill reads it.
        if (!xdna_run_restart(core->so_run, "core+so") || !xdna_run_wait(core->so_run)) {
            return false;
        }
    } else {
        xrt::run run = xdna_kernel_run_start(core->kern, args, 6);
        if (!xdna_run_wait(run)) {
            return false;
        }
    }
    rec_stage_end(st_kern);
    rec_stage_begin(st_read);
    // out layout: [gated f32 scratch KGATE*4][aq int8 KGATE][d_a f32]
    xdna_buffer_sync_from_device(core->out);
    if (core->act_arg == 2) {
        // The ssm_out activation drains to its own argument now, so the host's
        // copy of it comes from there.
        xdna_buffer_sync_from_device(core->so_fused ? core->gbuf : core->pkvb);
    }
    if (core->so_fused) {
        xdna_buffer_sync_from_device(core->x);
    }
    // GGML_XDNA_ATTN_GATED_VERIFY=1 checks the gated epilogue head by head
    // against the same arithmetic on the host. Everything it needs is in the
    // azg buffer the stage reads - attn from the gdn stage, z from this token,
    // gamma seeded once - so the comparison needs no extra plumbing. Judging
    // this stage by whether the model's text still looks right is far too
    // coarse: a wrong head reads as slightly worse prose.
    {
        static const bool gv = getenv("GGML_XDNA_ATTN_GATED_VERIFY") != nullptr;
        static int shown = 0;
        if (gv && shown < 2) {
            shown++;
            xdna_buffer_sync_from_device(core->azg);
            const float * azg_m = (const float *) core->azg->bo.map();
            const float * got   = (const float *) ((const uint8_t *)
                core->out->bo.map() + core->out_base);
            std::string map;
            double worst = 0.0;
            for (int h = 0; h < NVH; h++) {
                const float * head  = azg_m + (size_t) h * AZGN;
                const float * a     = head;
                const float * z     = head + SV;
                const float * gamma = head + 2 * SV;
                double ms = 0.0;
                for (int i = 0; i < SV; i++) {
                    ms += (double) a[i] * a[i];
                }
                const double rsc = 1.0 / std::sqrt(ms / SV + 1e-6);
                double se = 0.0, sr = 0.0;
                for (int i = 0; i < SV; i++) {
                    const double silu = z[i] / (1.0 + std::exp(-(double) z[i]));
                    const double ref  = a[i] * rsc * gamma[i] * silu;
                    const double d    = (double) got[(size_t) h * SV + i] - ref;
                    se += d * d;
                    sr += ref * ref;
                }
                double sg = 0.0;
                for (int i = 0; i < SV; i++) {
                    const double v = got[(size_t) h * SV + i];
                    sg += v * v;
                }
                const double rel = sr > 0.0 ? std::sqrt(se / sr) : 0.0;
                worst = std::max(worst, rel);
                // The device's sigmoid goes through a bf16 tanh, so a head
                // that agrees still sits around 1e-2 against exact arithmetic.
                // What a broken head looks like is an empty one, so the map
                // reports magnitude rather than closeness: '.' agrees, '0' was
                // never written, 'X' is there but wrong.
                const double mag = sr > 0.0 ? std::sqrt(sg / sr) : 1.0;
                map += mag < 1e-3 ? '0' : (rel < 0.2 ? '.' : 'X');
            }
            fprintf(stderr, "xdna-gated-verify: heads [%s] worst rel %.3e\n",
                    map.c_str(), worst);
        }
    }
    const uint8_t * omap =
        (const uint8_t *) core->out->bo.map() + core->out_base;
    std::memcpy(aq, omap + KGATE * 4, (size_t) KGATE);
    std::memcpy(d_a, omap + KGATE * 4 + KGATE, sizeof(float));
    rec_stage_end(st_read);
    core->token = t + 1;
    if (tmr) {
        fprintf(stderr, "RCT core t=%d upload=%.3f kern=%.3f read=%.3f tot=%.3f\n",
                t, st_upload.ms, st_kern.ms, st_read.ms,
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - wall0).count());
    }
    return true;
}

// --- attn_so_mmul (ssm_out) -------------------------------------------------

namespace {

// One 64(k) x 64(n) int4 element into buf[off..off+2048], the w4a8 nibble
// order of gemm_w4a8.py: (kz, nt, kr, nc) -> byte (kz*4+nt)*128 + kr*8 + nc/2,
// low nibble first. codes is column-major ([n*K + k]).
void pack_tile4(const int8_t * codes, int K, int k0, int n0, uint8_t * buf) {
    for (int kz = 0; kz < 4; kz++) {
        for (int nt = 0; nt < 4; nt++) {
            const size_t base = (size_t) (kz * 4 + nt) * 128;
            for (int kr = 0; kr < 16; kr++) {
                for (int nc = 0; nc < 16; nc++) {
                    const int k = k0 + kz * 16 + kr;
                    const int n = n0 + nt * 16 + nc;
                    const uint8_t v = (uint8_t) (codes[(size_t) n * K + k] & 0xF);
                    const size_t o = base + (size_t) kr * 8 + nc / 2;
                    if (nc % 2 == 0) {
                        buf[o] = (uint8_t) (buf[o] | v);
                    } else {
                        buf[o] = (uint8_t) (buf[o] | (v << 4));
                    }
                }
            }
        }
    }
}

// Re-quantize one dequantized [K] column into the int4 grid (column-major
// codes) and return its per-column scale.
float requant_col(const float * row, int K, int8_t * codes /* n*K+k */) {
    float amax = 0.0f;
    for (int k = 0; k < K; k++) {
        amax = std::max(amax, std::fabs(row[k]));
    }
    const float d = amax > 0.0f ? amax / 7.0f : 1.0f;
    for (int k = 0; k < K; k++) {
        int q = (int) lrintf(row[k] / d);
        codes[(size_t) k] = (int8_t) std::max(-8, std::min(7, q));
    }
    return d;
}

} // namespace

xdna_rec_so * xdna_rec_so_create(xdna_device * dev, int il,
                                 const char * xclbin, const char * insts,
                                 const uint8_t * w_so, int so_type) {
    using namespace xdna_so_pack;
    if (!dev || !w_so) {
        fprintf(stderr, "xdna-rec: bad so create args\n");
        return nullptr;
    }
    xdna_rec_so * m = new xdna_rec_so;
    m->dev = dev;
    m->il = il;

    std::vector<uint8_t> insts_buf;
    if (!read_file(insts, insts_buf)) {
        fprintf(stderr, "xdna-rec: cannot read so insts %s\n", insts);
        delete m;
        return nullptr;
    }
    m->kern = xdna_kernel_load_hw(dev, xclbin);
    if (!m->kern ||
        !xdna_kernel_bind_insts(dev, m->kern, (const uint32_t *) insts_buf.data(),
                                insts_buf.size() / sizeof(uint32_t))) {
        fprintf(stderr, "xdna-rec: so kernel load/bind failed for %s\n", xclbin);
        xdna_rec_so_free(m);
        return nullptr;
    }

    m->feed = xdna_buffer_alloc(dev, (size_t) feed_bytes());
    m->acc  = xdna_buffer_alloc(dev, (size_t) acc_bytes());
    if (!m->feed || !m->acc) {
        fprintf(stderr, "xdna-rec: so BO alloc failed\n");
        xdna_rec_so_free(m);
        return nullptr;
    }

    // Column stride and dequant follow the ssm_out type: Q4_K columns at
    // K=2048 are 8 x 144 B, Q5_K 8 x 176 B, Q6_K 8 x 210 B.
    int cb = CB_Q5;
    switch (so_type) {
        case GGML_TYPE_Q4_K: cb = CB_Q4; break;
        case GGML_TYPE_Q5_K: cb = CB_Q5; break;
        case GGML_TYPE_Q6_K: cb = CB_Q6; break;
        default:
            fprintf(stderr, "xdna-rec: unsupported ssm_out type %d\n", so_type);
            xdna_rec_so_free(m);
            return nullptr;
    }
    std::vector<float> row((size_t) K);
    std::vector<int8_t> codes((size_t) N * K);
    std::vector<int8_t> col((size_t) K);
    m->dw.assign(N, 1.0f);
    // Verification keeps an exact fp32 copy of this layer's weights so the run
    // can be compared against a host reference (see xdna-verify.h).
    const bool verify_this = xdna_verify_level() > 0 && il == xdna_verify_layer();
    if (verify_this) {
        m->w_ref.assign((size_t) N * K, 0.0f);
    }
    for (int n = 0; n < N; n++) {
        const uint8_t * c = w_so + (size_t) n * cb;
        switch (so_type) {
            case GGML_TYPE_Q4_K: dequantize_row_q4_K((const block_q4_K *) c, row.data(), K); break;
            case GGML_TYPE_Q5_K: dequantize_row_q5_K((const block_q5_K *) c, row.data(), K); break;
            default:             dequantize_row_q6_K((const block_q6_K *) c, row.data(), K); break;
        }
        m->dw[n] = requant_col(row.data(), K, col.data());
        if (verify_this) {
            std::memcpy(m->w_ref.data() + (size_t) n * K, row.data(), (size_t) K * sizeof(float));
        }
        std::memcpy(codes.data() + (size_t) n * K, col.data(), (size_t) K);
    }

    // Constant feed template: int4 W tiles in every object (the per-token A
    // slab region stays zero and is filled by run()).
    m->feed_tmpl.assign((size_t) feed_bytes(), 0);
    for (int c = 0; c < NCOL; c++) {
        for (int grp = 0; grp < NG; grp++) {
            const int n0 = c * CPC + grp * GRP;
            for (int t = 0; t < SB; t++) {
                uint8_t * o = m->feed_tmpl.data() + (size_t) (c * NT + grp * SB + t) * OBJ;
                pack_tile4(codes.data(), K, t * GRP, n0, o + A_SZ);
            }
        }
    }
    std::memcpy(m->feed->bo.map(), m->feed_tmpl.data(), m->feed_tmpl.size());
    xdna_buffer_sync_to_device(m->feed);
    return m;
}

void xdna_rec_so_free(xdna_rec_so * m) {
    if (!m) {
        return;
    }
    xdna_buffer_free(m->feed);
    xdna_buffer_free(m->acc);
    if (m->kern) {
        xdna_kernel_free(m->kern);
    }
    delete m;
}

bool xdna_rec_so_run(xdna_rec_so * m, const int8_t * aq, float d_a,
                     const float * hres, float * h_attn) {
    using namespace xdna_so_pack;
    const bool tmr = rec_time_on();
    const auto wall0 = std::chrono::steady_clock::now();
    rec_stage st_pack = {};
    rec_stage st_sync = {};
    rec_stage st_kern = {};
    rec_stage st_read = {};
    if (!m || !aq || !hres || !h_attn) {
        fprintf(stderr, "xdna-rec: bad so run args\n");
        return false;
    }

    // The aq codes + d_a arrive from the fused core (attn_gdn_gated): rewrite
    // the A slab of every feed object (256 B per object) in place. The host no
    // longer computes the gated activation or the int8 quant.
    rec_stage_begin(st_pack);
    uint8_t * feed = (uint8_t *) m->feed->bo.map();
    for (int c = 0; c < NCOL; c++) {
        for (int grp = 0; grp < NG; grp++) {
            for (int t = 0; t < SB; t++) {
                uint8_t * o = feed + (size_t) (c * NT + grp * SB + t) * OBJ;
                std::memset(o, 0, A_SZ);
                const int8_t * a0 = aq + t * GRP;
                for (int k = 0; k < GRP; k++) {
                    o[(k >> 4) * 64 + (k & 15)] = (uint8_t) a0[k];
                }
            }
        }
    }
    rec_stage_end(st_pack);   // pack = A rewrite only
    rec_stage_begin(st_sync);
    xdna_buffer_sync_to_device(m->feed);
    rec_stage_end(st_sync);

    // attn_so_mmul run: (opcode, insts, n, feed, acc)
    xdna_buffer * args[2] = { m->feed, m->acc };
    rec_stage_begin(st_kern);
    xrt::run run = xdna_kernel_run_start(m->kern, args, 2);
    if (!xdna_run_wait(run)) {
        fprintf(stderr, "xdna-rec: so kernel run failed (layer %d)\n", m->il);
        return false;
    }
    rec_stage_end(st_kern);
    rec_stage_begin(st_read);
    std::vector<int32_t> raw((size_t) NCOL * NG * 256);
    read_buffer(m->acc, raw.data(), raw.size() * sizeof(int32_t));

    // h_attn[n] = hres[n] + acc * d_a * dw[n] (fp64 rescale like the python
    // gather_out).
    for (int n = 0; n < N; n++) {
        const int c = n / CPC;
        const int rem = n % CPC;
        const int grp = rem / GRP;
        const int cc = rem % GRP;
        const double acc = (double) raw[(size_t) (c * NG + grp) * 256 + (cc >> 4) * 64 + (cc & 15)];
        h_attn[n] = hres[n] + (float) (acc * (double) d_a * (double) m->dw[n]);
    }
    rec_stage_end(st_read);

    // Host reference from the same int8 activation codes and the exact
    // weights: isolates the error the int4 weight grid adds.
    if (!m->w_ref.empty()) {
        std::vector<float> ref((size_t) N);
        for (int n = 0; n < N; n++) {
            const float * w = m->w_ref.data() + (size_t) n * K;
            double acc = 0.0;
            for (int k = 0; k < K; k++) {
                acc += (double) aq[k] * (double) w[k];
            }
            ref[n] = hres[n] + (float) (acc * (double) d_a);
        }
        xdna_verify_add("rec.ssm_out (int4 W)", xdna_verify_rel(h_attn, ref.data(), (size_t) N));
    }

    if (tmr) {
        fprintf(stderr, "RCT so il=%d pack=%.3f sync=%.3f kern=%.3f read=%.3f tot=%.3f\n",
                m->il, st_pack.ms, st_sync.ms, st_kern.ms, st_read.ms,
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - wall0).count());
    }
    return true;
}

// --- ffn_mmul_ab (whole FFN) -------------------------------------------------

namespace {

// ggml fp32 dequant of one [K] weight column at byte offset col*cb.
void dequant_col(const uint8_t * w, bool q6, int K, int cb, int col, float * row) {
    const uint8_t * c = w + (size_t) col * cb;
    if (q6) {
        dequantize_row_q6_K((const block_q6_K *) c, row, K);
    } else {
        dequantize_row_q4_K((const block_q4_K *) c, row, K);
    }
}

// Build the int4 codes + per-column scales of one [K x N] tensor. codes is
// column-major int8 (N*K bytes), dw is N floats.
void build_weight_grid(const uint8_t * w, bool q6, int K, int N,
                       int cb, std::vector<int8_t> & codes,
                       std::vector<float> & dw) {
    codes.assign((size_t) N * K, 0);
    dw.assign(N, 1.0f);
    std::vector<float> row((size_t) K);
    std::vector<int8_t> col((size_t) K);
    for (int n = 0; n < N; n++) {
        dequant_col(w, q6, K, cb, n, row.data());
        dw[n] = requant_col(row.data(), K, col.data());
        std::memcpy(codes.data() + (size_t) n * K, col.data(), (size_t) K);
    }
}

// Round-half-even int8 quant of one hff row (mirrors np.round in python).
void quant_a_row(const float * h, int n, int8_t * codes, float * d_a) {
    float amax = 0.0f;
    for (int i = 0; i < n; i++) {
        amax = std::max(amax, std::fabs(h[i]));
    }
    const float d = amax > 0.0f ? amax / 127.0f : 1.0f;
    for (int i = 0; i < n; i++) {
        const double v = (double) h[i] / d;
        double fl = std::floor(v);
        const double fr = v - fl;
        int q = (int) fl;
        if (fr > 0.5 || (fr == 0.5 && (q & 1))) {
            q++;
        }
        codes[i] = (int8_t) std::max(-128, std::min(127, q));
    }
    *d_a = d;
}

} // namespace

xdna_rec_ffn * xdna_rec_ffn_create(xdna_device * dev, int il,
                                   const char * xclbin, const char * insts,
                                   const uint8_t * w_gate, const uint8_t * w_up,
                                   const uint8_t * w_down, bool down_q6) {
    using namespace xdna_rec_ffn_pack;
    if (!dev || !w_gate || !w_up || !w_down) {
        fprintf(stderr, "xdna-rec: bad ffn create args\n");
        return nullptr;
    }
    xdna_rec_ffn * m = new xdna_rec_ffn;
    m->dev = dev;
    m->il = il;

    std::vector<uint8_t> insts_buf;
    if (!read_file(insts, insts_buf)) {
        fprintf(stderr, "xdna-rec: cannot read ffn insts %s\n", insts);
        delete m;
        return nullptr;
    }
    m->kern = xdna_kernel_load_hw(dev, xclbin);
    if (!m->kern ||
        !xdna_kernel_bind_insts(dev, m->kern, (const uint32_t *) insts_buf.data(),
                                insts_buf.size() / sizeof(uint32_t))) {
        fprintf(stderr, "xdna-rec: ffn kernel load/bind failed for %s\n", xclbin);
        xdna_rec_ffn_free(m);
        return nullptr;
    }

    m->af  = xdna_buffer_alloc(dev, (size_t) af_bytes());
    m->mid = xdna_buffer_alloc(dev, (size_t) mid_bytes());
    m->acc = xdna_buffer_alloc(dev, (size_t) acc_bytes());
    m->bf  = xdna_buffer_alloc(dev, (size_t) bf_bytes());
    m->out = xdna_buffer_alloc(dev, (size_t) out_bytes());
    m->aux = xdna_buffer_alloc(dev, (size_t) aux_bytes());
    if (!m->af || !m->mid || !m->acc || !m->bf || !m->out || !m->aux) {
        fprintf(stderr, "xdna-rec: ffn BO alloc failed\n");
        xdna_rec_ffn_free(m);
        return nullptr;
    }

    // Q4_K columns at K=1024 are 4 x 144 B; down at KDOWN is Q6_K 14 x 210 B
    // or Q4_K 14 x 144 B.
    constexpr int CB4 = (K / 256) * 144;
    const int CBD = (KDOWN / 256) * (down_q6 ? 210 : 144);
    std::vector<int8_t> cg, cu, cd;
    build_weight_grid(w_gate, false, K, N_MID, CB4, cg, m->dw_g);
    build_weight_grid(w_up,   false, K, N_MID, CB4, cu, m->dw_u);
    build_weight_grid(w_down, down_q6, KDOWN, N_OUT, CBD, cd, m->dw_d);

    // Verification keeps an exact fp32 copy of this layer's FFN weights so the
    // run can be compared against a host reference (see xdna-verify.h).
    if (xdna_verify_level() > 0 && il == xdna_verify_layer()) {
        m->wg_ref.assign((size_t) N_MID * K, 0.0f);
        m->wu_ref.assign((size_t) N_MID * K, 0.0f);
        m->wd_ref.assign((size_t) N_OUT * KDOWN, 0.0f);
        for (int n = 0; n < N_MID; n++) {
            dequant_col(w_gate, false, K, CB4, n, m->wg_ref.data() + (size_t) n * K);
            dequant_col(w_up,   false, K, CB4, n, m->wu_ref.data() + (size_t) n * K);
        }
        for (int n = 0; n < N_OUT; n++) {
            dequant_col(w_down, down_q6, KDOWN, CBD, n, m->wd_ref.data() + (size_t) n * KDOWN);
        }
    }

    // Constant stage-A feed template: gate/up int4 grids in every object (the
    // per-token header + A regions stay zero and are filled by run()). The
    // template is uploaded once; run() only rewrites the 768 B of header + A
    // per object in the mapped BO.
    m->af_tmpl.assign((size_t) af_bytes(), 0);
    for (int c = 0; c < NA; c++) {
        for (int grp = 0; grp < NGA; grp++) {
            const int m0 = c * CPCA + grp * GRPA;
            for (int t = 0; t < SA; t++) {
                uint8_t * o = m->af_tmpl.data() + (size_t) (c * NTA + grp * SA + t) * A_OBJ;
                pack_tile4(cg.data(), K, t * GRPA, m0, o + A_BGO);
                pack_tile4(cu.data(), K, t * GRPA, m0, o + A_BUO);
            }
        }
    }
    std::memcpy(m->af->bo.map(), m->af_tmpl.data(), m->af_tmpl.size());
    xdna_buffer_sync_to_device(m->af);

    // Constant stage-B feed: down int4 grid + u16 slab tag per object.
    std::vector<uint8_t> bf((size_t) bf_bytes(), 0);
    for (int bi = 0; bi < NB; bi++) {
        for (int grp = 0; grp < NGB; grp++) {
            const int n0 = bi * CPCB + grp * GRPB;
            for (int t = 0; t < SB; t++) {
                uint8_t * o = bf.data() + (size_t) (bi * NTB + grp * SB + t) * B_OBJ;
                pack_tile4(cd.data(), KDOWN, t * 64, n0, o);
                o[B_TAG]     = (uint8_t) (t & 0xFF);
                o[B_TAG + 1] = (uint8_t) ((t >> 8) & 0xFF);
            }
        }
    }
    std::memcpy(m->bf->bo.map(), bf.data(), bf.size());
    xdna_buffer_sync_to_device(m->bf);
    return m;
}

void xdna_rec_ffn_free(xdna_rec_ffn * m) {
    if (!m) {
        return;
    }
    xdna_buffer_free(m->af);
    xdna_buffer_free(m->mid);
    xdna_buffer_free(m->acc);
    xdna_buffer_free(m->bf);
    xdna_buffer_free(m->out);
    xdna_buffer_free(m->aux);
    if (m->kern) {
        xdna_kernel_free(m->kern);
    }
    delete m;
}

bool xdna_rec_ffn_run(xdna_rec_ffn * m, const float * hff,
                      const float * h_attn, float * h_out) {
    using namespace xdna_rec_ffn_pack;
    const bool tmr = rec_time_on();
    const auto wall0 = std::chrono::steady_clock::now();
    rec_stage st_pack = {};
    rec_stage st_sync = {};
    rec_stage st_kern = {};
    rec_stage st_read = {};
    if (!m || !hff || !h_attn || !h_out) {
        fprintf(stderr, "xdna-rec: bad ffn run args\n");
        return false;
    }
    rec_stage_begin(st_pack);
    std::vector<int8_t> hq((size_t) K);
    float d_a = 1.0f;
    quant_a_row(hff, K, hq.data(), &d_a);

    // Per-token af = resident gate/up grid + header (d_a*d_w) + int8 A. The
    // grid was uploaded at create, so only the 768 B (header + A) of each
    // object are rewritten here.
    uint8_t * af = (uint8_t *) m->af->bo.map();
    std::vector<float> hdr((size_t) 2 * GRPA);
    for (int c = 0; c < NA; c++) {
        for (int grp = 0; grp < NGA; grp++) {
            const int m0 = c * CPCA + grp * GRPA;
            for (int j = 0; j < GRPA; j++) {
                hdr[j] = d_a * m->dw_g[m0 + j];
                hdr[GRPA + j] = d_a * m->dw_u[m0 + j];
            }
            for (int t = 0; t < SA; t++) {
                uint8_t * o = af + (size_t) (c * NTA + grp * SA + t) * A_OBJ;
                std::memcpy(o, hdr.data(), A_HDR);
                uint8_t * at = o + A_AOF;
                std::memset(at, 0, A_BGO - A_AOF);
                const int8_t * a0 = hq.data() + t * GRPA;
                for (int k = 0; k < GRPA; k++) {
                    at[(k >> 4) * 64 + (k & 15)] = (uint8_t) a0[k];
                }
            }
        }
    }
    rec_stage_end(st_pack);   // pack = quant + header + A rewrite
    rec_stage_begin(st_sync);
    xdna_buffer_sync_to_device(m->af);
    rec_stage_end(st_sync);

    // fused ffn_mmul_ab run: (opcode, insts, n, af, mid, acc, bf, out, aux)
    xdna_buffer * args[6] = { m->af, m->mid, m->acc, m->bf, m->out, m->aux };
    rec_stage_begin(st_kern);
    xrt::run run = xdna_kernel_run_start(m->kern, args, 6);
    if (!xdna_run_wait(run)) {
        fprintf(stderr, "xdna-rec: ffn kernel run failed (layer %d)\n", m->il);
        return false;
    }
    rec_stage_end(st_kern);

    rec_stage_begin(st_read);
    std::vector<int32_t> raw((size_t) NB * NGB * 256);
    read_buffer(m->out, raw.data(), raw.size() * sizeof(int32_t));
    float auxv[NB * 8];
    read_buffer(m->aux, auxv, sizeof(auxv));
    const double d_aB = (double) auxv[0];

    // out[n] = acc * d_aB * dw_d[n]; h_out = h_attn + out (fp64 rescale like
    // the python gather_out).
    for (int n = 0; n < N_OUT; n++) {
        const int bi = n / CPCB;
        const int rem = n % CPCB;
        const int grp = rem / GRPB;
        const int c = rem % GRPB;
        const double acc = (double) raw[(size_t) (bi * NGB + grp) * 256 + (c >> 4) * 64 + (c & 15)];
        h_out[n] = h_attn[n] + (float) (acc * d_aB * (double) m->dw_d[n]);
    }
    rec_stage_end(st_read);

    // Host reference of the whole FFN in fp32 from the same hff input and the
    // exact weights: folds in both the int4 weight grid and the int8
    // activation quantization the kernel does internally.
    if (!m->wg_ref.empty()) {
        std::vector<float> mid((size_t) N_MID);
        for (int j = 0; j < N_MID; j++) {
            const float * wg = m->wg_ref.data() + (size_t) j * K;
            const float * wu = m->wu_ref.data() + (size_t) j * K;
            double g = 0.0;
            double u = 0.0;
            for (int k = 0; k < K; k++) {
                g += (double) hff[k] * (double) wg[k];
                u += (double) hff[k] * (double) wu[k];
            }
            mid[j] = (float) ((g / (1.0 + std::exp(-g))) * u);
        }
        std::vector<float> ref((size_t) N_OUT);
        for (int n = 0; n < N_OUT; n++) {
            const float * wd = m->wd_ref.data() + (size_t) n * KDOWN;
            double acc = 0.0;
            for (int j = 0; j < KDOWN; j++) {
                acc += (double) mid[j] * (double) wd[j];
            }
            ref[n] = h_attn[n] + (float) acc;
        }
        xdna_verify_add("rec.ffn (int4 W + int8 A)",
                        xdna_verify_rel(h_out, ref.data(), (size_t) N_OUT));
    }

    if (tmr) {
        fprintf(stderr, "RCT ffn il=%d pack=%.3f sync=%.3f kern=%.3f read=%.3f tot=%.3f\n",
                m->il, st_pack.ms, st_sync.ms, st_kern.ms, st_read.ms,
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - wall0).count());
    }
    return true;
}
