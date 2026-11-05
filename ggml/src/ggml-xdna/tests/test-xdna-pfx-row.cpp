// The merged prefill design's row command (xdna-pfx-row.cpp) as its cores see it, through models of the A MemTile's
// tiling and the shim and MemTile DMAs: the header and cos / sin objects, the setup objects, every core's x of every
// block, that every element lands at the byte the pfx attention reads it from (its layout re-derived here), and that
// every token the stream issues is waited, before its descriptor is written again and before the stream ends. The
// host's own objects (xdna_pfx_attn_host_objects) are checked too.
// Host-only: no device is opened.

#include "../xdna-pfx-attn.h"
#include "../xdna-pfx-row.h"
#include "../xdna-pgemm-path.h"
#include "../xdna-seq.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <set>
#include <tuple>
#include <vector>

// The attention runner asks it only on the device, which this test never opens
bool xdna_pgemm_path::pfx_loaded() {
    return false;
}

namespace {

constexpr int      D            = 256;
constexpr int      Q_HEADS      = 4;
constexpr int      KV_HEADS     = 2;
constexpr int      CORES        = 4;
constexpr int      BLOCK_ROWS   = 8;
constexpr int      N_RCOLS      = 2;
constexpr uint32_t ROW_COLS[]   = { 3, 7 };
constexpr uint32_t A_COLS[]     = { 6, 7 };
constexpr int      TXN_HEADER   = 4;
constexpr uint32_t TB_WORDS     = 10752 / 4;
constexpr int      A_OBJ_WORDS  = 64 * 64 / 2;
constexpr int      C_F32        = 64 * 32;
constexpr int      C_BLOCKS     = 2;  // blocks a C object
constexpr uint32_t W_SHIM_CH    = 1;
constexpr uint32_t MT_S2MM_CH   = 5;
constexpr uint32_t MT_MM2S_CH0  = 4;
constexpr uint32_t MT_BD_REG    = 0xA0000;
constexpr uint32_t MT_BD_STRIDE = 0x20;
constexpr uint32_t MT_S2MM_Q    = 0xA0604;  // start-queue registers: + 8 a channel
constexpr uint32_t MT_MM2S_Q    = 0xA0634;
constexpr uint32_t MT_CH_STRIDE = 8;
constexpr uint32_t MT_CHANNELS  = 6;
constexpr uint32_t MT_ADDR0     = 0x20000;  // a MemTile descriptor's address word for its own word 0
constexpr uint32_t W_PKT_CORE0  = 2;
constexpr int      ARG_W = 0, ARG_A = 1, ARG_C = 2, ARG_SETUP = 3;
constexpr uint32_t SHIM_ROW = 0, MT_ROW = 1;
constexpr int      RO_R1_BLOCK  = 5;
constexpr int      N_DIMS       = 256;
constexpr int      POS_PERIOD   = 37;        // positions restart, as packed texts' do
constexpr float    FREQ_BASE    = 10000.0f;
constexpr float    EPS          = 1e-6f;
constexpr double   TABLE_TOL    = 2e-3;      // the float theta recurrence against a double reference, p < 37
constexpr double   HI_REL       = 1.0 / 256;    // bf16: half an ulp is at most 2^-8 of the value
constexpr double   SPLIT_REL    = 1.0 / 65536;  // hi + lo: 16 bits
constexpr uint64_t UNSET        = ~0ull;
constexpr uint64_t WORD_MASK    = (1ull << 40) - 1;
// The pfx attention's objects (xdna-pfx-attn.cpp), re-derived: W objects a core a pass, A streams of 8 KB objects
constexpr int      HG            = Q_HEADS / KV_HEADS;
constexpr int64_t  DST_TB        = 10752;
constexpr int64_t  DST_OBJ       = 8192;
constexpr int64_t  DST_PP        = 16;
constexpr int64_t  DST_PASS      = 12 * DST_PP;
constexpr int64_t  DST_NK        = 32;
constexpr int64_t  DST_COLS      = 6;
constexpr int64_t  DST_ALIGN     = 4096;
constexpr int64_t  DST_TRAILER   = 8192;  // a W object's trailer byte
constexpr int      DST_HDR_WORDS = 16;
constexpr int      HDR_ATTN_ON = 8, HDR_A_DRAIN = 10;
constexpr uint8_t  FILL         = 0xAB;
constexpr uint32_t DRAIN_IDS[]  = { 4, 5, 10, 11, 12, 13, 14, 15 };
constexpr float    LOG2E        = 1.44269504f;

// A MemTile pattern: (size, stride) a dim, outermost first; the A tiling as kernels/pgemm.py writes it
constexpr int A_DIMS[4][2] = { { 8, 512 }, { 8, 8 }, { 8, 64 }, { 8, 1 } };

int failures = 0;

void check(bool ok, const char * what) {
    std::printf("%s: %s\n", ok ? "ok  " : "FAIL", what);
    failures += ok ? 0 : 1;
}

// The addresses a pattern visits, in stream order
void visit(const int (*dims)[2], int n, int base, std::vector<int> & out) {
    for (int i = 0; i < dims[0][0]; i++) {
        const int at = base + i * dims[0][1];
        if (n == 1) {
            out.push_back(at);
        } else {
            visit(dims + 1, n - 1, at, out);
        }
    }
}

std::vector<int> order_of(const int (*dims)[2]) {
    std::vector<int> out;
    visit(dims, 4, 0, out);
    return out;
}

// A host object's elements in the order a core receives them
std::vector<uint16_t> core_view(const uint16_t * obj) {
    static const std::vector<int> order = order_of(A_DIMS);
    std::vector<uint16_t>         v(order.size());
    for (size_t k = 0; k < order.size(); k++) {
        v[k] = obj[order[k]];
    }
    return v;
}

float f32_of_bf16(uint16_t h) {
    const uint32_t b = (uint32_t) h << 16;
    float          f;
    std::memcpy(&f, &b, 4);
    return f;
}

int32_t bits_of(float f) {
    int32_t w;
    std::memcpy(&w, &f, 4);
    return w;
}

struct test_job {
    std::vector<int32_t> pos;
    std::vector<float>   q_norm, k_norm;
    xdna_pfx_attn_dst    dst;
    xdna_pfx_row_job     job;
};

void fill_norms(test_job & t) {
    for (int d = 0; d < D; d++) {
        t.q_norm[d] = 1.0f + 0.3f * std::sin(0.37f * (float) d);
        t.k_norm[d] = 0.9f - 0.2f * std::cos(0.11f * (float) d);
    }
}

void fill_positions(test_job & t) {
    for (size_t i = 0; i < t.pos.size(); i++) {
        t.pos[i] = (int32_t) (i % POS_PERIOD);
    }
}

// Q, K and V at distinct, unaligned-to-each-other offsets of argument 0; the attention's objects as dst
void make_job(int64_t n, test_job & t) {
    t.pos.resize((size_t) n);
    t.q_norm.resize(D);
    t.k_norm.resize(D);
    fill_positions(t);
    fill_norms(t);
    t.dst                = xdna_pfx_attn_geometry(n, n, HG);
    t.job.dst            = &t.dst;
    t.job.q_scale        = LOG2E;
    t.job.n_tokens       = n;
    const size_t rows    = (size_t) xdna_pfx_row_job_blocks(t.job) * BLOCK_ROWS;
    t.job.q_off          = 4096;
    t.job.k_off          = t.job.q_off + rows * Q_HEADS * D * 4 + 8192;
    t.job.v_off          = t.job.k_off + rows * KV_HEADS * D * 4 + 4096;
    t.job.n_tokens       = n;
    t.job.pos            = t.pos.data();
    t.job.rope.n_dims    = N_DIMS;
    t.job.rope.freq_base = FREQ_BASE;
    t.job.eps            = EPS;
    t.job.q_norm         = t.q_norm.data();
    t.job.k_norm         = t.k_norm.data();
}

// --- the A objects ---

void check_header(const uint16_t * a, int64_t n_blocks) {
    const std::vector<uint16_t> v = core_view(a);
    int32_t                     w[xdna_pgemm_path::HDR_WORDS];
    std::memcpy(w, v.data(), sizeof(w));
    int32_t expect[xdna_pgemm_path::HDR_WORDS] = {};
    expect[xdna_pgemm_path::HW_ROW_ON]         = (int32_t) (n_blocks / 2);
    expect[xdna_pgemm_path::HW_ATTN_DRAIN]     = (int32_t) n_blocks;
    expect[xdna_pgemm_path::HW_ROW_SETUP]      = 1;
    check(std::memcmp(w, expect, sizeof(w)) == 0, "header: the cores read words 9, 11, 12 = N / 2, N, 1, the rest 0");
}

bool row_ok(const float * cs, const float * sn, int32_t p) {
    bool ok = true;
    for (int i = 0; i < N_DIMS / 2; i++) {
        const double theta = p * std::pow((double) FREQ_BASE, -2.0 * i / N_DIMS);
        ok = ok && std::fabs(cs[i] - std::cos(theta)) <= TABLE_TOL && std::fabs(sn[i] - std::sin(theta)) <= TABLE_TOL;
    }
    return ok;
}

bool zero_row(const float * cs, const float * sn) {
    bool ok = true;
    for (int i = 0; i < N_DIMS / 2; i++) {
        ok = ok && cs[i] == 0.0f && sn[i] == 0.0f;
    }
    return ok;
}

bool table_ok(const test_job & t, const uint16_t * obj, int64_t b) {
    const std::vector<uint16_t> v = core_view(obj);
    std::vector<float>          tab(v.size() / 2);
    std::memcpy(tab.data(), v.data(), tab.size() * sizeof(float));
    const float * f  = tab.data();
    const int     h  = N_DIMS / 2;
    bool          ok = true;
    for (int j = 0; j < BLOCK_ROWS; j++) {
        const int64_t tk = b * BLOCK_ROWS + j;
        const float * cs = f + j * h, *sn = f + (BLOCK_ROWS + j) * h;
        ok = ok && (tk < t.job.n_tokens ? row_ok(cs, sn, t.pos[(size_t) tk]) : zero_row(cs, sn));
    }
    return ok;
}

bool tables_ok(const test_job & t, const uint16_t * a, int64_t n_blocks) {
    bool ok = true;
    for (int64_t b = 0; b < n_blocks; b++) {
        ok = ok && table_ok(t, a + (size_t) (1 + b) * 2 * A_OBJ_WORDS, b);
    }
    return ok;
}

void check_a(const test_job & t) {
    const int64_t         n_blocks = xdna_pfx_row_job_blocks(t.job);
    std::vector<uint16_t> a(xdna_pfx_row_a_bytes(t.job) / 2);
    xdna_pfx_row_pack_a(t.job, a.data());
    check(a.size() == (size_t) (1 + n_blocks) * 2 * A_OBJ_WORDS, "A: a header object and one object a block");
    check_header(a.data(), n_blocks);
    check(tables_ok(t, a.data(), n_blocks),
          "tables: each row's cos [128] then sin [128] as the cores read them, zero past n_tokens");
}

// --- the setup objects ---

bool split_ok(const float * gamma, const uint16_t * hi, const uint16_t * lo) {
    bool ok = true;
    for (int d = 0; d < D; d++) {
        const double g = gamma ? gamma[d] : 1.0;
        const double h = f32_of_bf16(hi[d]), l = f32_of_bf16(lo[d]);
        ok = ok && std::fabs(g - h) <= std::fabs(g) * HI_REL && std::fabs(g - h - l) <= std::fabs(g) * SPLIT_REL;
        ok = ok && (gamma || (hi[d] == 0x3F80 && lo[d] == 0));
    }
    return ok;
}

int layout_of(int ci, int i) {
    return ci == 0 ? XDNA_PFX_ROW_LAYOUT_Q : i < KV_HEADS ? XDNA_PFX_ROW_LAYOUT_K : XDNA_PFX_ROW_LAYOUT_VT;
}

bool setup_obj_ok(const test_job & t, const uint8_t * obj, int ci, int i) {
    const bool    v     = ci == 1 && i >= KV_HEADS;
    const float * gamma = v ? nullptr : ci == 0 ? t.q_norm.data() : t.k_norm.data();
    const float   scale = ci == 0 ? t.job.q_scale : 1.0f;
    int32_t       prm[16];
    std::memcpy(prm, obj, sizeof(prm));
    const bool head = prm[0] == RO_R1_BLOCK && prm[1] == D && prm[2] == (v ? 0 : N_DIMS) &&
                      prm[3] == bits_of(scale) && prm[4] == bits_of(EPS) && prm[5] == layout_of(ci, i);
    uint16_t gs[2 * D];
    std::memcpy(gs, obj + sizeof(prm), sizeof(gs));
    return head && split_ok(gamma, gs, gs + D);
}

void check_setup(const test_job & t) {
    std::vector<uint8_t> s(xdna_pfx_row_setup_bytes());
    xdna_pfx_row_pack_setup(t.job, s.data());
    bool ok = s.size() == (size_t) N_RCOLS * CORES * TB_WORDS * 4;
    for (int k = 0; ok && k < N_RCOLS * CORES; k++) {
        ok = setup_obj_ok(t, s.data() + (size_t) k * TB_WORDS * 4, k / CORES, k % CORES);
    }
    check(ok,
          "setup: mode, d, n_rot (V 0), Q's scale (K and V 1), eps, layouts Q / K / V^T; the norm weight's bf16 "
          "hi / lo (V 1, 0)");
}

// --- the stream through a model of the DMAs ---

uint64_t src_of(int arg, uint64_t word) {
    return (uint64_t) arg << 40 | word;
}

struct shim_bd {
    uint32_t p[8] = {};
    int      arg  = -1;
    uint32_t off  = 0;
};

struct mt_task {
    uint32_t words = 0, addr = 0, pkt_id = 0;
    bool     pkt   = false;
};

using chan_key = std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>;  // col, row, dir, channel

struct model {
    std::map<std::pair<uint32_t, uint32_t>, shim_bd>                 sbd;   // (col, id)
    std::map<std::pair<uint32_t, uint32_t>, mt_task>                 mbd;   // (col, id)
    std::map<uint32_t, std::vector<uint64_t>>                        w_in;  // a row-op column's W from the shim
    std::map<uint32_t, size_t>                                       w_read;
    std::map<uint32_t, std::vector<uint64_t>>                        a_in;  // an A shim's MM2S 0
    std::map<uint32_t, std::vector<uint64_t>>                        mem;   // a MemTile's words
    std::map<std::pair<uint32_t, int>, std::vector<std::vector<uint64_t>>> objs;  // (col, core): its W buffers
    std::map<uint32_t, std::vector<uint64_t>>                        c_out;  // where its C stream's words land
    std::map<uint32_t, std::vector<std::set<uint32_t>>>              c_ids;  // each C task's descriptors
    std::map<chan_key, std::deque<std::set<uint32_t>>>               tokens;
    bool                                                             ok = true;
};

// Word k of a shim descriptor's walk, in words from its base
uint64_t bd_walk(const shim_bd & b, uint64_t k) {
    const uint64_t d0 = (b.p[3] >> 20) & 0x3FF, s0 = (b.p[3] & 0xFFFFF) + 1;
    const uint64_t d1 = (b.p[4] >> 20) & 0x3FF, s1 = (b.p[4] & 0xFFFFF) + 1, s2 = (b.p[5] & 0xFFFFF) + 1;
    if (d0 == 0) {
        return k * s0;
    }
    const uint64_t rest = k / d0;
    return (k % d0) * s0 + (d1 ? (rest % d1) * s1 + (rest / d1) * s2 : rest * s1);
}

void append_bd(const shim_bd & b, uint32_t rep, std::vector<uint64_t> & out) {
    const uint32_t iter_size = ((b.p[6] >> 20) & 0x3F) + 1, iter_stride = (b.p[6] & 0xFFFFF) + 1;
    const uint64_t base      = b.off / 4 + (uint64_t) (rep % iter_size) * iter_stride;
    for (uint32_t k = 0; k < b.p[0]; k++) {
        out.push_back(src_of(b.arg, base + bd_walk(b, k)));
    }
}

void expand_pass(model & m, uint32_t col, uint32_t id, uint32_t rep, std::vector<uint64_t> & out,
                 std::set<uint32_t> & ids) {
    for (int hop = 0; hop < 16; hop++) {
        const shim_bd & b = m.sbd[{ col, id }];
        ids.insert(id);
        append_bd(b, rep, out);
        if (!((b.p[7] >> 26) & 1)) {
            return;
        }
        id = (b.p[7] >> 27) & 0x1F;
    }
    m.ok = false;  // a chain that does not end
}

void expand_task(model & m, uint32_t col, uint32_t id, uint32_t repeat, std::vector<uint64_t> & out,
                 std::set<uint32_t> & ids) {
    for (uint32_t rep = 0; rep <= repeat; rep++) {
        expand_pass(m, col, id, rep, out, ids);
    }
}

// A descriptor written while a task on it may still run
void check_free(model & m, uint32_t col, uint32_t row, uint32_t id) {
    for (const auto & kv : m.tokens) {
        if (std::get<0>(kv.first) != col || std::get<1>(kv.first) != row) {
            continue;
        }
        for (const std::set<uint32_t> & ids : kv.second) {
            m.ok = m.ok && ids.count(id) == 0;
        }
    }
}

void on_blockwrite(model & m, uint32_t col, uint32_t row, uint32_t reg, const uint32_t * p) {
    if (row == SHIM_ROW) {
        const uint32_t id = (reg - xdna_txn::SHIM_BD_BASE) / xdna_txn::SHIM_BD_STRIDE;
        check_free(m, col, row, id);
        shim_bd b;
        std::memcpy(b.p, p, sizeof(b.p));
        m.sbd[{ col, id }] = b;
        return;
    }
    const uint32_t id = (reg - MT_BD_REG) / MT_BD_STRIDE;
    check_free(m, col, row, id);
    m.mbd[{ col, id }] = { p[0] & 0x1FFFF, (p[1] - MT_ADDR0) * 4, (p[0] >> 23) & 0x1F, (p[0] >> 31) != 0 };
}

void on_patch(model & m, const uint32_t * w) {
    const uint32_t reg = w[6] & 0xFFFFF, col = (w[6] >> 25) & 0x7F;
    shim_bd &      b   = m.sbd[{ col, (reg - xdna_txn::SHIM_BD_ADDR) / xdna_txn::SHIM_BD_STRIDE }];
    b.arg              = (int) w[8];
    b.off              = w[10];
}

void on_shim_push(model & m, uint32_t col, uint32_t dir, uint32_t ch, uint32_t value) {
    const uint32_t     id = value & 0xF, repeat = (value >> 16) & 0xFF;
    std::set<uint32_t> ids;
    const bool         row_col = col == ROW_COLS[0] || col == ROW_COLS[1];
    if (dir == 1 && ch == W_SHIM_CH) {
        expand_task(m, col, id, repeat, m.w_in[col], ids);
    } else if (dir == 1 && ch == 0) {
        expand_task(m, col, id, repeat, m.a_in[col], ids);
    } else if (dir == 0 && ch == 0 && row_col) {
        expand_task(m, col, id, repeat, m.c_out[col], ids);
        m.c_ids[col].push_back(ids);
    }
    if (value >> 31) {
        m.tokens[chan_key{ col, SHIM_ROW, dir, ch }].push_back(ids);
    }
}

void mt_s2mm(model & m, uint32_t col, const mt_task & t) {
    std::vector<uint64_t> & in  = m.w_in[col];
    std::vector<uint64_t> & mem = m.mem[col];
    size_t &                at  = m.w_read[col];
    if (mem.empty()) {
        mem.assign((xdna_pgemm_path::WS_ADDR + xdna_pgemm_path::WS_BYTES) / 4, UNSET);
    }
    m.ok = m.ok && at + t.words <= in.size() && t.addr / 4 + t.words <= mem.size();
    for (uint32_t k = 0; m.ok && k < t.words; k++) {
        mem[t.addr / 4 + k] = in[at++];
    }
}

void mt_mm2s(model & m, uint32_t col, uint32_t ch, const mt_task & t) {
    const int core = (int) t.pkt_id - (int) W_PKT_CORE0;
    m.ok           = m.ok && t.pkt && core >= 0 && core < CORES && (uint32_t) core / 2 == ch - MT_MM2S_CH0;
    const std::vector<uint64_t> & mem = m.mem[col];
    m.ok                              = m.ok && t.addr / 4 + t.words <= mem.size();
    if (m.ok) {
        m.objs[{ col, core }].emplace_back(mem.begin() + t.addr / 4, mem.begin() + t.addr / 4 + t.words);
    }
}

void on_mt_push(model & m, uint32_t col, uint32_t dir, uint32_t ch, uint32_t value) {
    const uint32_t id = value & 0xFF;
    const mt_task  t  = m.mbd[{ col, id }];
    if (dir == 0) {
        m.ok = m.ok && ch == MT_S2MM_CH;
        mt_s2mm(m, col, t);
    } else {
        mt_mm2s(m, col, ch, t);
    }
    m.tokens[chan_key{ col, MT_ROW, dir, ch }].push_back({ id });
}

void on_write(model & m, uint32_t col, uint32_t row, uint32_t reg, uint32_t value) {
    const uint32_t shim_q = reg - xdna_txn::SHIM_PUSHQ_BASE;
    if (row == SHIM_ROW && reg >= xdna_txn::SHIM_PUSHQ_BASE && shim_q < 2 * xdna_txn::SHIM_DIR_STRIDE) {
        on_shim_push(m, col, shim_q / xdna_txn::SHIM_DIR_STRIDE,
                     (shim_q % xdna_txn::SHIM_DIR_STRIDE) / xdna_txn::SHIM_CTRL_STRIDE, value);
    } else if (row == MT_ROW && reg >= MT_S2MM_Q && reg < MT_S2MM_Q + MT_CHANNELS * MT_CH_STRIDE) {
        on_mt_push(m, col, 0, (reg - MT_S2MM_Q) / MT_CH_STRIDE, value);
    } else if (row == MT_ROW && reg >= MT_MM2S_Q && reg < MT_MM2S_Q + MT_CHANNELS * MT_CH_STRIDE) {
        on_mt_push(m, col, 1, (reg - MT_MM2S_Q) / MT_CH_STRIDE, value);
    }
}

void on_tct(model & m, const uint32_t * w) {
    const chan_key key{ (w[2] >> 16) & 0xFF, (w[2] >> 8) & 0xFF, w[2] & 1, w[3] >> 24 };
    std::deque<std::set<uint32_t>> & q = m.tokens[key];
    m.ok                               = m.ok && !q.empty();
    if (!q.empty()) {
        q.pop_front();
    }
}

uint32_t op_words(uint32_t opcode) {
    switch (opcode) {
        case xdna_txn::OP_WRITE:      return xdna_txn::SZ_WRITE;
        case xdna_txn::OP_BLOCKWRITE: return xdna_txn::SZ_BLOCKWRITE;
        case xdna_txn::OP_MASKWRITE:  return xdna_txn::SZ_MASKWRITE;
        case xdna_txn::OP_TCT:        return xdna_txn::SZ_TCT;
        default:                      return xdna_txn::SZ_DDR_PATCH;
    }
}

void on_op(model & m, const uint32_t * w) {
    const uint32_t col = (w[2] >> 25) & 0x7F, row = (w[2] >> 20) & 0x1F, reg = w[2] & 0xFFFFF;
    switch (w[0]) {
        case xdna_txn::OP_BLOCKWRITE: on_blockwrite(m, col, row, reg, w + 4); break;
        case xdna_txn::OP_DDR_PATCH:  on_patch(m, w); break;
        case xdna_txn::OP_WRITE:      on_write(m, col, row, reg, w[4]); break;
        case xdna_txn::OP_TCT:        on_tct(m, w); break;
        default:                      break;
    }
}

void replay(const std::vector<uint32_t> & s, model & m) {
    for (size_t at = TXN_HEADER; m.ok && at < s.size(); at += op_words(s[at])) {
        on_op(m, s.data() + at);
    }
}

bool tokens_all_waited(const model & m) {
    bool ok = true;
    for (const auto & kv : m.tokens) {
        ok = ok && kv.second.empty();
    }
    return ok;
}

// The word of argument 0 core i of row-op column ci reads as row j, dim d of block b
uint64_t x_source(const test_job & t, int ci, int i, int64_t b, int j, int d) {
    const int64_t row = b * BLOCK_ROWS + j;
    if (ci == 0) {
        return src_of(ARG_W, t.job.q_off / 4 + (uint64_t) row * Q_HEADS * D + (uint64_t) i * D + d);
    }
    const size_t off = i < KV_HEADS ? t.job.k_off : t.job.v_off;
    return src_of(ARG_W, off / 4 + (uint64_t) row * KV_HEADS * D + (uint64_t) (i % KV_HEADS) * D + d);
}

bool x_obj_ok(const test_job & t, const std::vector<uint64_t> & o, int ci, int i, int64_t b) {
    bool ok = o.size() == TB_WORDS;
    for (int k = 0; ok && k < BLOCK_ROWS * D; k++) {
        ok = o[k] == x_source(t, ci, i, b, k / D, k % D);
    }
    return ok;
}

bool setup_src_ok(const std::vector<uint64_t> & o, int ci, int i) {
    bool ok = o.size() == TB_WORDS;
    for (uint32_t k = 0; ok && k < TB_WORDS; k++) {
        ok = o[k] == src_of(ARG_SETUP, (uint64_t) (ci * CORES + i) * TB_WORDS + k);
    }
    return ok;
}

bool core_objs_ok(const test_job & t, const model & m, int ci, int i) {
    const int64_t n_blocks = xdna_pfx_row_job_blocks(t.job);
    const auto    it       = m.objs.find({ ROW_COLS[ci], i });
    if (it == m.objs.end() || (int64_t) it->second.size() != 1 + n_blocks || !setup_src_ok(it->second[0], ci, i)) {
        return false;
    }
    bool ok = true;
    for (int64_t b = 0; ok && b < n_blocks; b++) {
        ok = x_obj_ok(t, it->second[(size_t) (1 + b)], ci, i, b);
    }
    return ok;
}

bool w_ok(const test_job & t, const model & m) {
    bool ok = true;
    for (int k = 0; k < N_RCOLS * CORES; k++) {
        ok = ok && core_objs_ok(t, m, k / CORES, k % CORES);
    }
    return ok;
}

bool a_stream_ok(const model & m, uint32_t col, int64_t n_blocks) {
    const auto it = m.a_in.find(col);
    bool       ok = it != m.a_in.end() && (int64_t) it->second.size() == (1 + n_blocks) * A_OBJ_WORDS;
    for (size_t k = 0; ok && k < it->second.size(); k++) {
        ok = it->second[k] == src_of(ARG_A, k);
    }
    return ok;
}

// --- the stream into the attention's objects ---

struct dst_geom {
    int64_t n_tiles = 0, npg = 0, w_bytes = 0, a_off = 0, stream = 0;
};

dst_geom geom_of(int64_t n) {
    dst_geom g;
    g.n_tiles = (n + DST_NK - 1) / DST_NK;
    g.npg     = (n + DST_PASS - 1) / DST_PASS;
    g.w_bytes = DST_COLS * 2 * g.npg * 4 * DST_TB;
    g.a_off   = (g.w_bytes + DST_ALIGN - 1) / DST_ALIGN * DST_ALIGN;
    g.stream  = (2 + 2 * KV_HEADS * g.n_tiles) * DST_OBJ;
    return g;
}

// Q head `head`'s dim d at position t: its pair's W object, row 16 hh + t % 16 of (d/8, row/8, 8, 8)
int64_t q_byte(const dst_geom & g, int head, int64_t t, int d) {
    const int64_t k = t / DST_PASS, pidx = (t % DST_PASS) / DST_PP, row = DST_PP * (head % HG) + t % DST_PP;
    const int64_t pass = (head / HG) * g.npg + k, core = 2 * (pidx % 2) + d / 128, dl = d % 128;
    const int64_t obj  = (((pidx / 2) * 2 * g.npg + pass) * 4 + core) * DST_TB;
    return obj + 2 * (((dl / 8) * 4 + row / 8) * 64 + (dl % 8) * 8 + row % 8);
}

// KV head g's key's dim d: its tile's K object (key/8, d/8, 8, 8 after the A tiling) or V^T object
int64_t kv_byte(const dst_geom & g, bool vt, int kvh, int64_t key, int d) {
    const int64_t t = key / DST_NK, kl = key % DST_NK, dl = d % 128;
    const int64_t obj = 2 + (kvh * g.n_tiles + t) * 2 + (vt ? 1 : 0);
    const int64_t el  = vt ? (8 * (dl / 16) + dl % 8) * 64 + 32 * ((dl / 8) % 2) + kl :
                             (16 * (kl / 8) + 8 * (dl / 64) + kl % 8) * 64 + dl % 64;
    return g.a_off + (d / 128) * g.stream + obj * DST_OBJ + 2 * el;
}

int64_t expect_byte(const dst_geom & g, int ci, int i, int64_t t, int d) {
    return ci == 0 ? q_byte(g, i, t, d) : kv_byte(g, i >= KV_HEADS, i % KV_HEADS, t, d);
}

struct dst_check {
    const test_job *  t = nullptr;
    const model *     m = nullptr;
    dst_geom          g;
    std::vector<char> seen;  // bf16 slots of the parent written
    bool              placed = true, once = true, mirror = true, slots = true;
};

// One element of core i's part of C object o on column ci: where the replayed drains put it
void check_elem(dst_check & k, int ci, int64_t o, int i, int e, std::vector<char> & slot) {
    const int bp = e / (BLOCK_ROWS * D), j = (e / D) % BLOCK_ROWS, d = e % D;
    int       half = 0;
    const int s    = xdna_pfx_row_stream_word(layout_of(ci, i), bp, j, d, &half);
    k.slots        = k.slots && s >= 0 && s < C_F32 && !slot[(size_t) (2 * s + half)];
    slot[(size_t) (2 * s + half) % slot.size()] = 1;
    const std::vector<uint64_t> & out = k.m->c_out.at(ROW_COLS[ci]);
    const size_t                  at  = ((size_t) o * CORES + (size_t) i) * C_F32 + (size_t) s;
    if (at >= out.size() || (out[at] >> 40) != (uint64_t) ARG_C) {
        k.placed = false;
        return;
    }
    const int64_t byte = 4 * (int64_t) (out[at] & WORD_MASK) + 2 * half;
    k.placed           = k.placed && byte == expect_byte(k.g, ci, i, 2 * BLOCK_ROWS * o + BLOCK_ROWS * bp + j, d);
    k.mirror = k.mirror && (int64_t) xdna_pfx_row_dst_byte(*k.t->job.dst, ci, o, i, s) == byte - 2 * half;
    const size_t slot_at = (size_t) byte / 2;
    k.once               = k.once && slot_at < k.seen.size() && !k.seen[slot_at];
    if (slot_at < k.seen.size()) {
        k.seen[slot_at] = 1;
    }
}

void check_part(dst_check & k, int ci, int64_t o, int i) {
    std::vector<char> slot(2 * C_F32, 0);
    for (int e = 0; e < C_BLOCKS * BLOCK_ROWS * D; e++) {
        check_elem(k, ci, o, i, e, slot);
    }
}

// Every element of every core part of every C object on both columns
void check_parts(dst_check & k) {
    const int64_t objs = 2 * k.g.n_tiles;
    for (int64_t x = 0; x < N_RCOLS * objs * CORES; x++) {
        check_part(k, (int) (x / (objs * CORES)), (x / CORES) % objs, (int) (x % CORES));
    }
}

bool drain_ids_ok(const model & m) {
    bool ok = true;
    for (const auto & kv : m.c_ids) {
        for (const std::set<uint32_t> & ids : kv.second) {
            ok = ok && ids.size() == (size_t) CORES;
            for (uint32_t id : ids) {
                ok = ok && std::find(std::begin(DRAIN_IDS), std::end(DRAIN_IDS), id) != std::end(DRAIN_IDS);
            }
        }
    }
    return ok;
}

bool c_stream_sizes_ok(const model & m, int64_t n_tiles) {
    bool ok = true;
    for (uint32_t col : ROW_COLS) {
        const auto it = m.c_out.find(col);
        ok = ok && it != m.c_out.end() && it->second.size() == (size_t) (2 * n_tiles * CORES * C_F32) &&
             m.c_ids.at(col).size() == (size_t) n_tiles;
    }
    return ok;
}

// The written slots: each W object's Q where the command's rows reach, and every K and V^T object, nothing else
bool coverage_ok(const dst_check & k) {
    size_t written = 0;
    for (char c : k.seen) {
        written += c != 0;
    }
    const size_t q_objs = (size_t) (2 * k.g.n_tiles) * HG * 2;  // 16-position groups x KV heads x D halves
    return written == (q_objs * DST_TRAILER + (size_t) (2 * KV_HEADS * k.g.n_tiles) * 2 * DST_OBJ) / 2;
}

void check_dst_stream(const test_job & t) {
    const int64_t n = t.job.n_tokens;
    model         m;
    replay(xdna_pfx_row_stream(t.job), m);
    dst_check k;
    k.t = &t;
    k.m = &m;
    k.g = geom_of(n);
    k.seen.assign((size_t) (k.g.a_off + 2 * k.g.stream) / 2, 0);
    char what[200];
    std::snprintf(what, sizeof(what), "dst stream n %lld (%lld tiles): replays, no descriptor rewritten under a task",
                  (long long) n, (long long) k.g.n_tiles);
    check(m.ok, what);
    check(t.dst.w_bytes == (size_t) k.g.w_bytes && t.dst.a_off == (size_t) k.g.a_off &&
              t.dst.stream_bytes == (size_t) k.g.stream && t.dst.n_tiles == k.g.n_tiles,
          "  xdna_pfx_attn_geometry equals the re-derived layout");
    check(tokens_all_waited(m), "  every token issued is waited");
    check(w_ok(t, m), "  each core: its setup object, then each block's 8 rows of its head, whole key tiles");
    check(a_stream_ok(m, A_COLS[0], xdna_pfx_row_job_blocks(t.job)) &&
              a_stream_ok(m, A_COLS[1], xdna_pfx_row_job_blocks(t.job)),
          "  both A streams: the header and the tables, once");
    check(drain_ids_ok(m) && c_stream_sizes_ok(m, k.g.n_tiles),
          "  C: one 4-chain task a tile and column, on descriptors 4, 5, 10-15, two C objects each");
    if (!m.ok || !c_stream_sizes_ok(m, k.g.n_tiles)) {
        return;
    }
    check_parts(k);
    check(k.slots, "  each layout puts a core part's 4096 elements in 4096 distinct stream slots");
    check(k.placed, "  every element lands at the byte the attention reads it from (Q W objects, K, V^T)");
    check(k.once && coverage_ok(k), "  each target written once; trailers, headers and the rest never");
    check(k.mirror, "  xdna_pfx_row_dst_byte agrees with the replayed descriptors");
}

// --- the host's objects with dst (xdna_pfx_attn_host_objects) ---

struct attn_case {
    std::vector<int32_t> lo, hi;
    xdna_pfx_attn_job    job;
};

// T texts of equal length, each row seeing its own text's keys
void make_attn(int64_t n, int texts, attn_case & a) {
    const int64_t len = (n + texts - 1) / texts;
    a.lo.resize((size_t) n);
    a.hi.resize((size_t) n);
    for (int64_t t = 0; t < n; t++) {
        a.lo[(size_t) t] = (int32_t) (t / len * len);
        a.hi[(size_t) t] = (int32_t) (std::min(n, (t / len + 1) * len) - 1);
    }
    a.job.lo       = a.lo.data();
    a.job.hi       = a.hi.data();
    a.job.n_tokens = n;
    a.job.n_kv     = n;
    a.job.hg       = HG;
}

struct pass_tiles {
    int64_t lo = 0, hi = 0;
};

// Pass k of a KV head: the tiles its rows see
pass_tiles tiles_of_pass(const attn_case & a, int64_t k) {
    const int64_t t1 = std::min(a.job.n_tokens, (k + 1) * DST_PASS);
    int32_t       lo = a.lo[(size_t) (k * DST_PASS)], hi = a.hi[(size_t) (k * DST_PASS)];
    for (int64_t t = k * DST_PASS; t < t1; t++) {
        lo = std::min(lo, a.lo[(size_t) t]);
        hi = std::max(hi, a.hi[(size_t) t]);
    }
    return { lo / DST_NK, hi / DST_NK + 1 };
}

std::vector<int32_t> core_words(const uint8_t * obj) {
    const std::vector<uint16_t> v = core_view((const uint16_t *) obj);
    std::vector<int32_t>        w(DST_HDR_WORDS);
    std::memcpy(w.data(), v.data(), w.size() * sizeof(int32_t));
    return w;
}

bool headers_ok(const attn_case & a, const dst_geom & g, const uint8_t * parent) {
    int32_t drain = 1;
    for (int64_t k = 0; k < g.npg; k++) {
        const pass_tiles pt = tiles_of_pass(a, k);
        drain += (int32_t) (2 * 2 * (pt.hi - pt.lo));  // both KV heads' passes
    }
    std::vector<int32_t> pg(DST_HDR_WORDS, 0), fa(DST_HDR_WORDS, 0);
    pg[HDR_ATTN_ON] = 1;
    pg[HDR_A_DRAIN] = drain;
    fa[0]           = (int32_t) (2 * g.npg);
    bool ok         = true;
    for (int h = 0; h < 2; h++) {
        const uint8_t * s = parent + g.a_off + h * g.stream;
        ok                = ok && core_words(s) == pg && core_words(s + DST_OBJ) == fa;
    }
    return ok;
}

// W object (ci, pass, core): its trailer, and zero Q only where its pair starts at or past the last tile's end
bool w_obj_host_ok(const attn_case & a, const dst_geom & g, const uint8_t * parent, int64_t x) {
    const int64_t ci = x / (2 * g.npg * 4), p = (x / 4) % (2 * g.npg), core = x % 4;
    const int64_t k = p % g.npg, pidx = 2 * ci + core / 2, t0 = k * DST_PASS + pidx * DST_PP;
    const pass_tiles pt = tiles_of_pass(a, k);
    const uint8_t *  o  = parent + x * DST_TB;
    int32_t          tr[2 + 2 * 32];
    std::memcpy(tr, o + DST_TRAILER, sizeof(tr));
    bool ok = tr[0] == pt.lo * DST_NK && tr[1] == pt.hi - pt.lo;
    for (int r = 0; r < 32; r++) {
        const int64_t t = t0 + r % DST_PP;
        ok = ok && tr[2 + r] == (t < a.job.n_tokens ? a.lo[(size_t) t] : tr[0]) &&
             tr[2 + 32 + r] == (t < a.job.n_tokens ? a.hi[(size_t) t] : tr[0]);
    }
    const uint8_t want = t0 >= g.n_tiles * DST_NK ? 0 : FILL;
    return ok && std::all_of(o, o + DST_TRAILER, [want](uint8_t b) { return b == want; });
}

void check_host_objects(int64_t n, int texts) {
    attn_case a;
    make_attn(n, texts, a);
    const dst_geom       g = geom_of(n);
    std::vector<uint8_t> parent((size_t) (g.a_off + 2 * g.stream), FILL);
    xdna_pfx_attn_host_objects(a.job, parent.data());
    bool w_ok_all = true;
    for (int64_t x = 0; x < DST_COLS * 2 * g.npg * 4; x++) {
        w_ok_all = w_ok_all && w_obj_host_ok(a, g, parent.data(), x);
    }
    bool kv_untouched = true;
    for (int h = 0; h < 2; h++) {
        const uint8_t * s = parent.data() + g.a_off + h * g.stream + 2 * DST_OBJ;
        kv_untouched      = kv_untouched && std::all_of(s, s + g.stream - 2 * DST_OBJ, [](uint8_t b) { return b == FILL; });
    }
    char what[160];
    std::snprintf(what, sizeof(what), "host objects n %lld, %d texts: both A streams' headers", (long long) n, texts);
    check(headers_ok(a, g, parent.data()), what);
    check(w_ok_all, "  every W object's trailer; zero Q only past the last tile; the rest of Q untouched");
    check(kv_untouched, "  K and V^T objects untouched");
}

void check_dst(int64_t n) {
    test_job t;
    make_job(n, t);
    if (n == 912) {
        check_a(t);
        check_setup(t);
    }
    check_dst_stream(t);
}

}  // namespace

int main(void) {
    constexpr int64_t SMALL = 40, EG2 = 912, THREE_TASKS = 1100, TEXTS = 8;
    constexpr int64_t DST_NS[] = { 1, 15, 16, 17, 31, 32, 33, SMALL, 191, 192, 193, EG2, THREE_TASKS, 1188 };
    test_job          small;
    make_job(SMALL, small);
    check_a(small);
    check_setup(small);
    for (int64_t n : DST_NS) {
        check_dst(n);
    }
    check_host_objects(EG2, 1);
    check_host_objects(1188, TEXTS);
    check_host_objects(SMALL, 3);
    if (failures) {
        std::printf("test-xdna-pfx-row: %d failures\n", failures);
        return 1;
    }
    std::printf("test-xdna-pfx-row: all passed\n");
    return 0;
}
