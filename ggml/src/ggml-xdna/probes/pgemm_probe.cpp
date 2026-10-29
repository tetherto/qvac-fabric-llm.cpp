// Command latency of the prefill GEMM (kernels/pgemm.py), on random data: what one command costs in total,
// how that splits into the host's start and the wait, and how it grows with
//   - the instruction stream's size (padding ops of one kind at the head, middle, before the final waits or
//     the tail of the stream; an empty stream is the floor),
//   - the weight bytes (K, N; --q4k packs Q4_K weights, half the bytes of Q8_0 at the same K and N) and the M
//     blocks (--nmb), with the output hashed to prove a rewrite exact,
//   - the idle time before the command (--gap-us, --gap-spin),
//   - a second command behind the first, one by one or in one runlist (--then-k, --then-n).
// The probe includes xdna-pgemm.cpp, so it times the very streams the backend builds; its own xdna_seq_build
// appends the padding.
//   pgemm_probe [--k K] [--n N] [--m M] [--nmb B] [--glu 0|1] [--gelu 0|1] [--reps R] [--rounds R]
//               [--q4k 0|1] [--gap-us US] [--gap-spin 0|1] [--pad-op write|block|mask|patch|mtwrite|mtblock|mtmask]
//               [--pad-at head|mid|wait|tail] [--sweep-pad a,b,c] [--variants pad[/shim][/mt],...] [--rewrite shim,mt]
//               [--empty W,W] [--then-k K --then-n N [--then-glu 0|1] [--then-nmb B]]
// Build (from the build directory, next to the libraries):
//   c++ -O2 -std=gnu++17 -fopenmp -DGGML_USE_OPENMP <the backend's GGML_XDNA_* definitions> -I<src>/ggml/src
//       -I<src>/ggml/include -isystem /opt/xilinx/xrt/include probes/pgemm_probe.cpp -o bin/pgemm_probe -Lbin
//       -lggml-xdna -lggml-base -L/opt/xilinx/xrt/lib -lxrt_coreutil -Wl,-rpath,'$ORIGIN'
#include "../xdna-pgemm.cpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <map>
#include <set>

namespace {

enum class pad_op { write, block, mask, patch, mt_write, mt_block, mt_mask };
enum class pad_at { head, mid, wait, tail };

struct padding {
    int    words = 0;
    pad_op op    = pad_op::write;
    pad_at at    = pad_at::head;
};

padding g_pad;

// Stream rewrites that keep what the array does (the probe compares the output bit for bit)
struct rewrite {
    bool dedup_shim_token = false;  // an S2MM/MM2S token-id write once a (column, register), not once a push
    bool dedup_mt_ctrl    = false;  // the same for the MemTile channels' controller id
};

rewrite g_rw;

// idle time between two commands: a pass's commands follow each other (0) or the host sleeps in between
int  g_gap_us   = 200;
bool g_gap_spin = false;  // the host core spins through the gap instead of sleeping

// the output rows the probe hashes: its M real rows of N columns
uint64_t g_hash_rows = 0, g_hash_cols = 0;

// An op that changes nothing: it hits shim descriptor 15, which no stream of this design uses.
constexpr uint32_t SPARE_BD = 15;
// a MemTile lock, descriptor and channel the stream never touches
constexpr uint32_t SPARE_MT_LOCK = 60, SPARE_MT_BD = 33, SPARE_MT_CH = 3;

void append_pad_ops(std::vector<uint32_t> & out, int words, pad_op op) {
    xdna_seq tmp;
    int      col  = 0;
    int      done = 0;
    while (done < words) {
        const uint32_t before = (uint32_t) tmp.ops.size();
        switch (op) {
            case pad_op::write:
                xdna_seq_write(&tmp, col, 0, xdna_txn::SHIM_BD_BASE + SPARE_BD * xdna_txn::SHIM_BD_STRIDE, 0);
                break;
            case pad_op::block:
                {
                    xdna_bd bd;
                    bd.valid = false;
                    xdna_seq_blockwrite(&tmp, col, 0, SPARE_BD, &bd);
                }
                break;
            case pad_op::mask:
                xdna_seq_maskwrite(&tmp, col, 0, xdna_txn::SHIM_BD_BASE + SPARE_BD * xdna_txn::SHIM_BD_STRIDE, 0, 0);
                break;
            case pad_op::patch:
                xdna_seq_ddr_patch(&tmp, col, 0, SPARE_BD, 0, 0);
                break;
            case pad_op::mt_write:
                mt_set_lock(&tmp, col, SPARE_MT_LOCK, 0);
                break;
            case pad_op::mt_block:
                mt_bd(&tmp, col, SPARE_MT_BD, 16, WS_ADDR, SPARE_MT_LOCK, 0, SPARE_MT_LOCK, 0);
                break;
            case pad_op::mt_mask:
                xdna_seq_maskwrite(&tmp, col, 1, 0xA0600 + SPARE_MT_CH * 8, MT_CTRL_PKT << 8, 0x1F00);
                break;
        }
        done += (int) (tmp.ops.size() - before);
        col = (col + 1) % COLS;
    }
    out.insert(out.end(), tmp.ops.begin(), tmp.ops.end());
}

}  // namespace

uint32_t op_words(uint32_t first) {
    const uint32_t opc = first & 0xFF;
    return opc == xdna_txn::OP_WRITE      ? xdna_txn::SZ_WRITE :
           opc == xdna_txn::OP_BLOCKWRITE ? xdna_txn::SZ_BLOCKWRITE :
           opc == xdna_txn::OP_MASKWRITE  ? xdna_txn::SZ_MASKWRITE :
           opc == xdna_txn::OP_TCT        ? xdna_txn::SZ_TCT :
                                            xdna_txn::SZ_DDR_PATCH;
}

// Where in the ops (a word index on an op boundary) the padding goes: the middle, or before the run of
// completion waits that ends the stream.
size_t pad_offset(const std::vector<uint32_t> & ops, pad_at at) {
    std::vector<size_t> starts;
    for (size_t i = 0; i < ops.size(); i += op_words(ops[i])) {
        starts.push_back(i);
    }
    if (at == pad_at::mid) {
        return starts[starts.size() / 2];
    }
    if (at == pad_at::wait) {
        size_t k = starts.size();
        while (k > 0 && (ops[starts[k - 1]] & 0xFF) == xdna_txn::OP_TCT) {
            k--;
        }
        return k < starts.size() ? starts[k] : ops.size();
    }
    return 0;
}


bool is_ctrl_maskwrite(const uint32_t * op) {
    if ((op[0] & 0xFF) != xdna_txn::OP_MASKWRITE) {
        return false;
    }
    const uint32_t row = (op[2] >> 20) & 0x1F, reg = op[2] & 0xFFFFF;
    const bool     shim = row == 0 && reg >= xdna_txn::SHIM_TOKEN_BASE && reg < xdna_txn::SHIM_TOKEN_BASE + 0x20;
    const bool     mt   = row == 1 && reg >= 0xA0600 && reg < 0xA0660;
    return (shim && g_rw.dedup_shim_token) || (mt && g_rw.dedup_mt_ctrl);
}

// ops without the repeated writes of a controller / token id: each (tile, register) keeps its first
std::vector<uint32_t> dedup_ctrl_writes(const std::vector<uint32_t> & ops) {
    std::vector<uint32_t> out;
    std::set<std::pair<uint32_t, uint32_t>> seen;
    for (size_t i = 0; i < ops.size(); i += op_words(ops[i])) {
        const uint32_t n = op_words(ops[i]);
        if (is_ctrl_maskwrite(&ops[i]) && !seen.insert({ ops[i + 2], ops[i + 4] }).second) {
            continue;
        }
        out.insert(out.end(), ops.begin() + (long) i, ops.begin() + (long) (i + n));
    }
    return out;
}

// The backend's stream builder plus the padding (g_pad): the probe's definition wins over the library's.
std::vector<uint32_t> xdna_seq_build(const xdna_seq * seq_in) {
    xdna_seq seq_rw = *seq_in;
    if (g_rw.dedup_shim_token || g_rw.dedup_mt_ctrl) {
        seq_rw.ops = dedup_ctrl_writes(seq_in->ops);
    }
    const xdna_seq * seq = &seq_rw;
    std::vector<uint32_t> body;
    if (g_pad.words > 0 && g_pad.at == pad_at::head) {
        append_pad_ops(body, g_pad.words, g_pad.op);
    }
    const size_t at = g_pad.words > 0 ? pad_offset(seq->ops, g_pad.at) : seq->ops.size();
    body.insert(body.end(), seq->ops.begin(), seq->ops.begin() + (long) at);
    if (g_pad.words > 0 && (g_pad.at == pad_at::mid || g_pad.at == pad_at::wait)) {
        append_pad_ops(body, g_pad.words, g_pad.op);
    }
    body.insert(body.end(), seq->ops.begin() + (long) at, seq->ops.end());
    if (g_pad.words > 0 && g_pad.at == pad_at::tail) {
        append_pad_ops(body, g_pad.words, g_pad.op);
    }
    // the padding ops are counted by their sizes: every op's last word is its size in bytes, except
    // that a 12-word block write and a 12-word patch both end in a zero word, so count by walking
    uint32_t n_instr = 0;
    for (size_t i = 0; i < body.size();) {
        i += op_words(body[i]);
        n_instr++;
    }
    std::vector<uint32_t> out;
    out.push_back(xdna_txn::DEV_MAJOR | xdna_txn::DEV_MINOR << xdna_txn::HDR_W0_MINOR_SHIFT |
                  xdna_txn::DEV_GEN << xdna_txn::HDR_W0_GEN_SHIFT | seq->n_rows << xdna_txn::HDR_W0_ROWS_SHIFT);
    out.push_back(seq->n_cols | seq->mem_tile_rows << xdna_txn::HDR_W1_MEMTILE_SHIFT);
    out.push_back(n_instr);
    out.push_back((uint32_t) ((4 + body.size()) * sizeof(uint32_t)));
    out.insert(out.end(), body.begin(), body.end());
    return out;
}

namespace {

double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct stats {
    double min = 0, p10 = 0, med = 0, mean = 0;
};

stats stats_of(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    stats s;
    s.min  = v.front();
    s.p10  = v[v.size() / 10];
    s.med  = v[v.size() / 2];
    double sum = 0;
    for (double x : v) {
        sum += x;
    }
    s.mean = sum / (double) v.size();
    return s;
}

struct timing {
    std::vector<double> total, start, wait;
    uint64_t            hash = 0;
};

void report(const char * label, int words, const timing & t) {
    const stats a = stats_of(t.total), b = stats_of(t.start), c = stats_of(t.wait);
    std::printf("%-34s words %6d  total med %7.1f p10 %7.1f min %7.1f | start med %6.1f | wait med %7.1f min %7.1f | out %016llx\n",
                label, words, a.med, a.p10, a.min, b.med, c.med, c.min, (unsigned long long) t.hash);
    std::fflush(stdout);
}

struct q8_0_block {
    ggml_fp16_t d;
    int8_t      qs[32];
};

// Q8_0 blocks of small random values
void fill_q8_0(ggml_tensor * w) {
    std::mt19937                         rng(1234);
    std::uniform_int_distribution<int>   q(-20, 20);
    q8_0_block *                         b = (q8_0_block *) w->data;
    const size_t                         n = ggml_nelements(w) / 32;
    for (size_t i = 0; i < n; i++) {
        b[i].d = ggml_fp32_to_fp16(0.02f);
        for (int j = 0; j < 32; j++) {
            b[i].qs[j] = (int8_t) q(rng);
        }
    }
}

struct q4_k_block {
    ggml_fp16_t d, dmin;
    uint8_t     scales[12];
    uint8_t     qs[128];
};

// Q4_K blocks of random codes and scales
void fill_q4_k(ggml_tensor * w) {
    std::mt19937                       rng(1234);
    std::uniform_int_distribution<int> byte(0, 255);
    q4_k_block *                       b = (q4_k_block *) w->data;
    const size_t                       n = ggml_nelements(w) / 256;
    for (size_t i = 0; i < n; i++) {
        b[i].d    = ggml_fp32_to_fp16(0.01f);
        b[i].dmin = ggml_fp32_to_fp16(0.005f);
        for (uint8_t & x : b[i].scales) {
            x = (uint8_t) byte(rng);
        }
        for (uint8_t & x : b[i].qs) {
            x = (uint8_t) byte(rng);
        }
    }
}

struct call_cfg {
    int  K = 512, N = 1024, M = 15, nmb = 0;
    bool glu = false, gelu = false, q4k = false;
};

// a random weight of the config's type
ggml_tensor * random_weight(ggml_context * ctx, const call_cfg & cfg) {
    ggml_tensor * w = ggml_new_tensor_2d(ctx, cfg.q4k ? GGML_TYPE_Q4_K : GGML_TYPE_Q8_0, cfg.K, cfg.N);
    w->data         = std::malloc(ggml_nbytes(w));
    cfg.q4k ? fill_q4_k(w) : fill_q8_0(w);
    return w;
}

// everything one command needs, kept for the repetitions
struct call_state {
    xdna_kernel_pool pool;
    weights *        ws = nullptr;
    a_geom           ag{ 1, KS };
    c_dest           cd;
    xdna_buffer *    al = nullptr, *dst = nullptr, *sink = nullptr;
    int              nmb = 0, M = 0;
};

a_geom make_geom(int M, int K, int nmb) {
    a_geom ag(M, K);
    if (nmb > 0) {
        ag.n_blk  = nmb;
        ag.slab   = std::min(nmb, MAX_BLOCKS);
        ag.n_sl   = (nmb + ag.slab - 1) / ag.slab;
        ag.h_total = (size_t) ag.n_sl * 2 * HDR_BYTES;
    }
    return ag;
}

bool setup(call_state & st, xdna_device * dev, ggml_context * ctx, const call_cfg & cfg) {
    st.pool.device = dev;
    ggml_tensor * w  = random_weight(ctx, cfg);
    ggml_tensor * w2 = cfg.glu ? random_weight(ctx, cfg) : nullptr;
    st.ws = cfg.glu ? weights_for(&st.pool, { w, w2 }, true) : weights_for(&st.pool, { w });
    if (!st.ws) {
        std::fprintf(stderr, "cannot pack the weight\n");
        return false;
    }
    const xdna_gemv_geom & g    = st.ws->geom;
    st.M                        = cfg.M;
    st.ag                       = make_geom(cfg.M, g.K, cfg.nmb);
    st.nmb                      = st.ag.blocks(0);
    const int  nch              = g.n_out();
    const int  objs             = nch * (cfg.glu ? 1 : 2);
    const size_t dst_bytes      = (size_t) st.ag.n_blk * MBLK * cfg.N * 4;
    st.al   = xdna_buffer_alloc(dev, st.ag.bytes());
    st.dst  = xdna_buffer_alloc(dev, dst_bytes);
    st.sink = xdna_buffer_alloc(dev, (size_t) 2 * MBLK * NC * 4);
    if (!st.al || !st.dst || !st.sink) {
        std::fprintf(stderr, "cannot allocate the buffers\n");
        return false;
    }
    std::memset(st.al->data, 0, st.ag.bytes());
    std::memset(st.dst->data, 0, dst_bytes);
    // the headers, as pgemm_call writes them
    const int mode = (cfg.glu ? PG_MODE_GLU : PG_MODE_PLAIN) | (cfg.glu && cfg.gelu ? PG_MODE_GELU : 0);
    for (int sl = 0; sl < st.ag.n_sl; sl++) {
        for (int h = 0; h < 2; h++) {
            int32_t  words[8] = { st.ag.blocks(sl) * nch, g.n_tiles(), g.fmt != XDNA_WFMT_Q4G32 ? 1 : 0, g.k_tile() / KS,
                                  g.k_tile() / KS / 2,   cfg.glu ? 1 : 2, mode, g.k_tile() / KS / 2 - 1 };
            uint16_t hb[16];
            std::memcpy(hb, words, sizeof(hb));
            uint16_t * hdr = (uint16_t *) st.al->data + (size_t) (sl * 2 + h) * HDR_BYTES / 2;
            for (int j = 0; j < 16; j++) {
                hdr[(j / 8) * KS + j % 8] = hb[j];
            }
        }
    }
    if (!xdna_buffer_sync_to_device(st.al) || !xdna_buffer_sync_to_device(st.dst)) {
        return false;
    }
    // random activations, laid out as the backend lays them out
    std::vector<float>                    act((size_t) cfg.M * g.K);
    std::mt19937                          rng(99);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (float & v : act) {
        v = u(rng);
    }
    const int nstep = g.K / KS;
    auto      row   = [&](int m, uint16_t * o, int sw) {
        const float * r = act.data() + (size_t) m * g.K;
        for (int s = 0; s < nstep; s++) {
            bf16_row(r + (size_t) (s ^ sw) * KS, o + (size_t) s * MB * KS, KS);
        }
    };
    for (int sl = 0; sl < st.ag.n_sl; sl++) {
        if (!lay_slab(st.ag, st.al, sl, cfg.M, g.K, row)) {
            return false;
        }
    }
    st.cd.ldc  = (uint32_t) cfg.N;
    st.cd.objs = objs;
    for (int c = 0; c < COLS; c++) {
        int v = 0;
        for (int o = 0; o < objs; o++) {
            v += o * CHUNK / 2 + c * NC < cfg.N;
        }
        st.cd.valid[c] = v;
    }
    g_hash_rows = (size_t) cfg.M;
    g_hash_cols = (size_t) cfg.N;
    return true;
}

void drop_streams() {
    for (auto & kv : g_pg.seqs) {
        xdna_kernel_free(kv.second.kern);
    }
    g_pg.seqs.clear();
}

xdna_kernel * stream_of(call_state & st) {
    return seq_kernel(&st.pool, st.ws, st.nmb, st.cd, 0, (uint32_t) st.ag.base(0));
}

// FNV-1a of the output rows the probe's M real rows fill
uint64_t hash_dst(xdna_buffer * dst) {
    if (!xdna_buffer_sync_from_device(dst)) {
        return 0;
    }
    uint64_t h = 1469598103934665603ull;
    for (size_t r = 0; r < g_hash_rows; r++) {
        const uint8_t * p = (const uint8_t *) dst->data + r * g_hash_cols * 4;
        for (size_t i = 0; i < g_hash_cols * 4; i++) {
            h = (h ^ p[i]) * 1099511628211ull;
        }
    }
    return h;
}

timing run_reps(xdna_kernel * kern, xdna_buffer ** args, int reps) {
    timing t;
    for (int i = 0; i < reps + 10; i++) {
        const double t0   = now_us();
        xrt::run     run  = xdna_kernel_run_start(kern, args, 4);
        const double t1   = now_us();
        const bool   done = xdna_run_wait(run);
        const double t2   = now_us();
        if (!done) {
            std::fprintf(stderr, "wait failed\n");
            std::exit(3);
        }
        if (i >= 10) {
            t.total.push_back(t2 - t0);
            t.start.push_back(t1 - t0);
            t.wait.push_back(t2 - t1);
        }
        // a short pause so the device idles as it does between a pass's commands
        if (g_gap_us > 0) {
            if (g_gap_spin) {
                const double until = now_us() + g_gap_us;
                while (now_us() < until) {
                }
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(g_gap_us));
            }
        }
    }
    t.hash = hash_dst(args[2]);
    return t;
}

// a stream of nothing but `words` words of padding: the cost of a command that does no work
xdna_kernel * empty_stream(call_state & st, int words) {
    g_pad = { words, g_pad.op, pad_at::head };
    xdna_kernel * kern = xdna_kernel_clone(g_pg.kern);
    xdna_seq      none;
    const std::vector<uint32_t> insts = xdna_seq_build(&none);
    if (!xdna_kernel_bind_insts(st.pool.device, kern, insts.data(), insts.size())) {
        return nullptr;
    }
    return kern;
}

// ops of a built stream by kind and tile row: what the command processor executes
void print_op_histogram(const std::vector<uint32_t> & insts) {
    std::map<std::string, int> n;
    for (size_t i = 4; i < insts.size(); i += op_words(insts[i])) {
        const uint32_t opc = insts[i] & 0xFF;
        const char *   nm  = opc == xdna_txn::OP_WRITE ? "write" : opc == xdna_txn::OP_BLOCKWRITE ? "block" :
                             opc == xdna_txn::OP_MASKWRITE ? "mask" : opc == xdna_txn::OP_TCT ? "tct" : "patch";
        const uint32_t row = opc == xdna_txn::OP_TCT ? (insts[i + 2] >> 8) & 0xFF : (insts[i + 2] >> 20) & 0x1F;
        n[std::string(nm) + (row == 0 ? " shim" : row == 1 ? " memtile" : " core")]++;
    }
    std::printf("  ops:");
    for (const auto & kv : n) {
        std::printf(" %s %d,", kv.first.c_str(), kv.second);
    }
    std::printf("\n");
}

// Two commands one after the other, started and waited one by one, or both in one runlist: what a second
// command costs when the host does not come back between them.
void run_chain(call_state & a, call_state & b, int reps) {
    xdna_buffer * args_a[4] = { a.ws->bo, a.al, a.dst, a.sink };
    xdna_buffer * args_b[4] = { b.ws->bo, b.al, b.dst, b.sink };
    g_pad                   = {};
    xdna_kernel * ka        = stream_of(a);
    g_pg.seqs.clear();
    xdna_kernel * kb = stream_of(b);
    g_pg.seqs.clear();
    std::vector<double> seq_t, list_t;
    for (int i = 0; i < reps + 10; i++) {
        const double t0 = now_us();
        {
            xrt::run ra = xdna_kernel_run_start(ka, args_a, 4);
            xdna_run_wait(ra);
            xrt::run rb = xdna_kernel_run_start(kb, args_b, 4);
            xdna_run_wait(rb);
        }
        const double t1 = now_us();
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        const double t2 = now_us();
        {
            xrt::run ra = xdna_kernel_run_make(ka, args_a, 4);
            xrt::run rb = xdna_kernel_run_make(kb, args_b, 4);
            xdna_batch_begin(2);
            xdna_run_submit(ka, ra, args_a, 4);
            xdna_run_submit(kb, rb, args_b, 4);
            xdna_batch_wait();
        }
        const double t3 = now_us();
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        if (i >= 10) {
            seq_t.push_back(t1 - t0);
            list_t.push_back(t3 - t2);
        }
    }
    const stats s = stats_of(seq_t), l = stats_of(list_t);
    std::printf("chain of 2: one by one med %.1f p10 %.1f | one runlist med %.1f p10 %.1f us\n", s.med, s.p10, l.med, l.p10);
}

std::vector<int> parse_list(const char * s) {
    std::vector<int> v;
    for (const char * p = s; *p;) {
        v.push_back(std::atoi(p));
        while (*p && *p != ',') {
            p++;
        }
        if (*p == ',') {
            p++;
        }
    }
    return v;
}

void set_rewrite(const char * s) {
    g_rw.dedup_shim_token = std::strstr(s, "shim") != nullptr;
    g_rw.dedup_mt_ctrl    = std::strstr(s, "mt") != nullptr;
}

struct variant {
    int     pad = 0;
    rewrite rw;
};

std::vector<variant> pad_variants(const std::vector<int> & pads) {
    std::vector<variant> v;
    for (int p : pads.empty() ? std::vector<int>{ 0 } : pads) {
        v.push_back({ p, g_rw });
    }
    return v;
}

// "words/rw,words/rw": rw is none, shim, mt or shimmt
std::vector<variant> parse_variants(const std::string & spec) {
    std::vector<variant> v;
    size_t               pos = 0;
    while (pos <= spec.size()) {
        const size_t      end = std::min(spec.find(',', pos), spec.size());
        const std::string item = spec.substr(pos, end - pos);
        const size_t      slash = item.find('/');
        variant           x;
        x.pad = std::atoi(item.c_str());
        if (slash != std::string::npos) {
            x.rw.dedup_shim_token = item.find("shim", slash) != std::string::npos;
            x.rw.dedup_mt_ctrl    = item.find("mt", slash) != std::string::npos;
        }
        v.push_back(x);
        pos = end + 1;
    }
    return v;
}

pad_at parse_at(const char * s) {
    return !std::strcmp(s, "tail") ? pad_at::tail : !std::strcmp(s, "mid") ? pad_at::mid :
           !std::strcmp(s, "wait") ? pad_at::wait : pad_at::head;
}

pad_op parse_op(const char * s) {
    return !std::strcmp(s, "block") ? pad_op::block : !std::strcmp(s, "mask") ? pad_op::mask :
           !std::strcmp(s, "patch") ? pad_op::patch : !std::strcmp(s, "mtwrite") ? pad_op::mt_write :
           !std::strcmp(s, "mtblock") ? pad_op::mt_block : !std::strcmp(s, "mtmask") ? pad_op::mt_mask : pad_op::write;
}

}  // namespace

int main(int argc, char ** argv) {
    call_cfg         cfg, then_cfg;
    then_cfg.K = 0;
    int              reps = 100, rounds = 5;
    std::vector<int> sweep, empty;
    std::vector<variant> variants;
    padding          pad;
    for (int i = 1; i < argc; i++) {
        auto is = [&](const char * k) { return !std::strcmp(argv[i], k) && i + 1 < argc; };
        if (is("--k")) cfg.K = std::atoi(argv[++i]);
        else if (is("--n")) cfg.N = std::atoi(argv[++i]);
        else if (is("--m")) cfg.M = std::atoi(argv[++i]);
        else if (is("--nmb")) cfg.nmb = std::atoi(argv[++i]);
        else if (is("--glu")) cfg.glu = std::atoi(argv[++i]) != 0;
        else if (is("--gelu")) cfg.gelu = std::atoi(argv[++i]) != 0;
        else if (is("--q4k")) cfg.q4k = std::atoi(argv[++i]) != 0;
        else if (is("--reps")) reps = std::atoi(argv[++i]);
        else if (is("--rounds")) rounds = std::atoi(argv[++i]);
        else if (is("--gap-us")) g_gap_us = std::atoi(argv[++i]);
        else if (is("--gap-spin")) g_gap_spin = std::atoi(argv[++i]) != 0;
        else if (is("--then-k")) then_cfg.K = std::atoi(argv[++i]);
        else if (is("--then-n")) then_cfg.N = std::atoi(argv[++i]);
        else if (is("--then-glu")) then_cfg.glu = std::atoi(argv[++i]) != 0;
        else if (is("--then-nmb")) then_cfg.nmb = std::atoi(argv[++i]);
        else if (is("--rewrite")) set_rewrite(argv[++i]);
        else if (is("--variants")) variants = parse_variants(argv[++i]);
        else if (is("--pad-op")) pad.op = parse_op(argv[++i]);
        else if (is("--pad-at")) pad.at = parse_at(argv[++i]);
        else if (is("--sweep-pad")) sweep = parse_list(argv[++i]);
        else if (is("--empty")) empty = parse_list(argv[++i]);
        else { std::fprintf(stderr, "bad arg %s\n", argv[i]); return 2; }
    }
    xdna_device * dev = xdna_device_open();
    if (!dev) {
        std::fprintf(stderr, "no device\n");
        return 1;
    }
    ggml_init_params ip = { 1 << 20, nullptr, true };
    ggml_context *   ctx = ggml_init(ip);
    call_state       st;
    st.pool.device = dev;
    g_pg.kern      = xdna_kernel_find(dev, STEM);
    if (!g_pg.kern) {
        std::fprintf(stderr, "no pgemm artifact\n");
        return 1;
    }
    for (int words : empty) {
        g_pad = { words, pad.op, pad_at::head };
        xdna_kernel * k = empty_stream(st, words);
        if (!k) {
            return 1;
        }
        if (!st.al) {
            st.al = xdna_buffer_alloc(dev, 1 << 20);
            st.dst = xdna_buffer_alloc(dev, 1 << 20);
            st.sink = xdna_buffer_alloc(dev, 1 << 20);
            st.ws = nullptr;
        }
        xdna_buffer * w = xdna_buffer_alloc(dev, 1 << 20);
        xdna_buffer * args[4] = { w, st.al, st.dst, st.sink };
        char          label[64];
        std::snprintf(label, sizeof(label), "empty pad %d", words);
        report(label, words, run_reps(k, args, reps));
        xdna_kernel_free(k);
    }
    if (empty.empty() && then_cfg.K > 0) {
        call_state st2;
        then_cfg.M = cfg.M;
        if (!setup(st, dev, ctx, cfg) || !setup(st2, dev, ctx, then_cfg)) {
            return 1;
        }
        run_chain(st, st2, reps);
        return 0;
    }
    if (empty.empty()) {
        g_pad = {};
        if (!setup(st, dev, ctx, cfg)) {
            return 1;
        }
        xdna_buffer * args[4] = { st.ws->bo, st.al, st.dst, st.sink };
        std::vector<variant> vars = variants.empty() ? pad_variants(sweep) : variants;
        std::vector<xdna_kernel *> kerns;
        for (const variant & v : vars) {
            g_pad = { v.pad, pad.op, pad.at };
            g_rw  = v.rw;
            xdna_kernel * kern = stream_of(st);
            if (!kern) {
                return 1;
            }
            kerns.push_back(kern);
            g_pg.seqs.clear();  // the cache is keyed without the padding; the kernels stay with the probe
        }
        std::vector<timing> res(kerns.size());
        const int           per_round = std::max(1, reps / rounds);
        for (int r = 0; r < rounds; r++) {
            for (size_t i = 0; i < kerns.size(); i++) {
                const timing t = run_reps(kerns[i], args, per_round);
                res[i].total.insert(res[i].total.end(), t.total.begin(), t.total.end());
                res[i].start.insert(res[i].start.end(), t.start.begin(), t.start.end());
                res[i].wait.insert(res[i].wait.end(), t.wait.begin(), t.wait.end());
                res[i].hash = t.hash;
            }
        }
        for (size_t i = 0; i < kerns.size(); i++) {
            char label[96];
            std::snprintf(label, sizeof(label), "K%d N%d M%d nmb%d glu%d pad %d%s%s", cfg.K, cfg.N, cfg.M, st.nmb, (int) cfg.glu, vars[i].pad,
                          vars[i].rw.dedup_shim_token ? " -shim" : "", vars[i].rw.dedup_mt_ctrl ? " -mt" : "");
            report(label, (int) kerns[i]->insts_host.size(), res[i]);
            print_op_histogram(kerns[i]->insts_host);
        }
    }
    return 0;
}
