#include "xdna-host-attn.h"

#include "ggml.h"
#include "xdna-attn-mask.h"
#include "xdna-host-attn-ops.h"
#include "xdna-util.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

// An online-softmax attention over blocks of KB keys. One work item is the
// rows of a few positions over all the query heads of one KV head, so the K
// and V blocks it streams are read once for every head of the group. For a
// block the item computes the scores of its rows in 4 x 4 register tiles (a
// dot over the head size per row and key), then the softmax of each row, then
// P V in 4 x 64 tiles. A tile that sees no key of the block skips it, so a
// band, a causal mask or a packed sequence cost the keys they show, not the
// whole matrix. The blocks sit on the grid of absolute keys, so a row's result
// does not depend on how the rows are split into items, or on the thread count.
// Accumulation is f32, as ggml-cpu's.

namespace {

using namespace xdna_attn_mask;
using namespace xdna_host_attn;

constexpr int     R_MAX            = 32;  // rows an item
constexpr int     TARGET_ROWS      = 32;  // rows an item, before the thread count shrinks it
constexpr int     ITEMS_PER_THREAD = 4;   // items a thread: the dynamic schedule's room to balance
constexpr int     TILES_MAX        = R_MAX / TILE_ROWS;
constexpr int64_t D_MAX            = 1024;
constexpr int64_t MIN_QUERIES      = 8;   // below it ggml-cpu's op, which splits the keys over the threads
// A mask smaller than this is scanned on the calling thread: waking the OpenMP team costs more than the scan.
constexpr int64_t SCAN_PARALLEL_MIN_ENTRIES = 1 << 17;

// What the node asks for.
struct shape {
    int64_t dk = 0, dv = 0, n_q = 0, n_kv = 0, n_head = 0, n_kvh = 0;
    int     hg     = 0;  // query heads a KV head
    float   scale  = 1.0f;
    bool    kv_f16 = true;
    size_t  q_nb1 = 0, q_nb2 = 0, k_nb1 = 0, k_nb2 = 0, v_nb1 = 0, v_nb2 = 0;
};

bool kv_type_ok(const ggml_tensor * k, const ggml_tensor * v) {
    return k->type == v->type && (k->type == GGML_TYPE_F16 || k->type == GGML_TYPE_F32) &&
           k->nb[0] == ggml_type_size(k->type) && v->nb[0] == ggml_type_size(v->type);
}

bool mask_ok(const ggml_tensor * m, int64_t n_kv, int64_t n_q) {
    return !m || (m->type == GGML_TYPE_F16 && m->nb[0] == sizeof(uint16_t) && m->ne[0] >= n_kv && m->ne[1] >= n_q &&
                  m->ne[2] == 1 && m->ne[3] == 1);
}

// No ALiBi, no softcap, no sinks, one sequence, f32 accumulation.
bool options_ok(const ggml_tensor * n, float * scale) {
    float params[3];
    std::memcpy(params, n->op_params, sizeof(params));
    *scale          = params[0];
    const int prec  = (int) ggml_flash_attn_ext_get_prec(n);
    return !n->src[4] && params[1] == 0.0f && params[2] == 0.0f && (prec == GGML_PREC_DEFAULT || prec == GGML_PREC_F32);
}

bool layout_ok(const ggml_tensor * n, const ggml_tensor * q, const ggml_tensor * k, const ggml_tensor * v) {
    return q->type == GGML_TYPE_F32 && q->nb[0] == sizeof(float) && kv_type_ok(k, v) && q->ne[3] == 1 && k->ne[3] == 1 &&
           v->ne[3] == 1 && k->ne[1] == v->ne[1] && k->ne[2] == v->ne[2] && k->ne[2] >= 1 && q->ne[2] % k->ne[2] == 0 &&
           q->ne[0] == k->ne[0] && n->type == GGML_TYPE_F32 && ggml_is_contiguous(n);
}

bool sizes_ok(const ggml_tensor * n, const shape & s) {
    return s.dk % LANES == 0 && s.dv % LANES == 0 && s.dk <= D_MAX && s.dv <= D_MAX && s.hg <= R_MAX && s.n_kv >= 1 &&
           s.n_q >= MIN_QUERIES && n->ne[0] == s.dv && n->ne[1] == s.n_head && n->ne[2] == s.n_q && n->ne[3] == 1;
}

bool shape_of(const ggml_tensor * n, shape * s) {
    const ggml_tensor * q = n->src[0];
    const ggml_tensor * k = n->src[1];
    const ggml_tensor * v = n->src[2];
    if (n->op != GGML_OP_FLASH_ATTN_EXT || !q || !k || !v || !options_ok(n, &s->scale) || !layout_ok(n, q, k, v)) {
        return false;
    }
    s->dk     = k->ne[0];
    s->dv     = v->ne[0];
    s->n_q    = q->ne[1];
    s->n_kv   = k->ne[1];
    s->n_head = q->ne[2];
    s->n_kvh  = k->ne[2];
    s->hg     = (int) (s->n_head / s->n_kvh);
    s->kv_f16 = k->type == GGML_TYPE_F16;
    s->q_nb1  = q->nb[1], s->q_nb2 = q->nb[2];
    s->k_nb1  = k->nb[1], s->k_nb2 = k->nb[2];
    s->v_nb1  = v->nb[1], s->v_nb2 = v->nb[2];
    return sizes_ok(n, *s) && mask_ok(n->src[3], s->n_kv, s->n_q);
}

// What a mask says about each row: the keys it shows, first to last. `pure`
// when every row shows all of them with 0, so the kernel reads no mask value.
struct row_spans {
    std::vector<int32_t> lo, hi;  // lo > hi: the row sees nothing
    bool                 pure = true;
};

void scan_rows(const ggml_tensor * m, const shape & s, row_spans * out) {
    out->lo.assign((size_t) s.n_q, 0);
    out->hi.assign((size_t) s.n_q, (int32_t) (s.n_kv - 1));
    if (!m) {
        return;
    }
    int64_t impure = 0;
#pragma omp parallel for num_threads(xdna_chunk_threads()) if (s.n_q * s.n_kv >= SCAN_PARALLEL_MIN_ENTRIES) reduction(+ : impure)
    for (int64_t t = 0; t < s.n_q; t++) {
        const uint16_t * row   = (const uint16_t *) ((const char *) m->data + t * m->nb[1]);
        const int64_t    first = first_not_hidden(row, s.n_kv);
        const int64_t    last  = last_not_hidden(row, s.n_kv);
        out->lo[t]             = (int32_t) first;
        out->hi[t]             = (int32_t) last;
        if (first <= last && !all_visible(row, first, last)) {
            impure++;
        }
    }
    out->pure = impure == 0;
}

// Every attention layer of a graph reads one of two masks; the spans of each
// are scanned once a graph (xdna_host_attn_graph_begin starts a new one: a
// new ubatch rewrites the mask in place).
struct span_memo {
    static constexpr int SLOTS = 4;
    struct slot {
        uint64_t                         at   = 0;
        const void *                     data = nullptr;
        int64_t                          n_kv = 0, n_q = 0, ne0 = 0;
        size_t                           nb1  = 0;
        std::shared_ptr<const row_spans> rows;
    };
    std::mutex mtx;
    uint64_t   epoch = 1;
    int        next  = 0;
    slot       slots[SLOTS];
} g_spans;

bool slot_holds(const span_memo::slot & sl, const ggml_tensor * m, const shape & s) {
    return sl.at == g_spans.epoch && sl.data == m->data && sl.n_kv == s.n_kv && sl.n_q == s.n_q && sl.ne0 == m->ne[0] &&
           sl.nb1 == m->nb[1];
}

std::shared_ptr<const row_spans> find_spans(const ggml_tensor * m, const shape & s) {
    for (const auto & sl : g_spans.slots) {
        if (slot_holds(sl, m, s)) {
            return sl.rows;
        }
    }
    return nullptr;
}

void keep_spans(const ggml_tensor * m, const shape & s, std::shared_ptr<const row_spans> rows) {
    auto & sl    = g_spans.slots[g_spans.next];
    g_spans.next = (g_spans.next + 1) % span_memo::SLOTS;
    sl           = { g_spans.epoch, m->data, s.n_kv, s.n_q, m->ne[0], m->nb[1], std::move(rows) };
}

std::shared_ptr<const row_spans> spans_of(const ggml_tensor * m, const shape & s) {
    std::lock_guard<std::mutex> lock(g_spans.mtx);
    if (m) {
        if (auto kept = find_spans(m, s)) {
            return kept;
        }
    }
    auto rows = std::make_shared<row_spans>();
    scan_rows(m, s, rows.get());
    if (m) {
        keep_spans(m, s, rows);
    }
    return rows;
}

// One item: the rows of np positions over the heads of KV head `kvh`.
struct item {
    int     kvh = 0, p0 = 0, np = 0;
    int64_t cost = 0;  // keys its rows see
};

struct plan {
    shape             s;
    const row_spans * spans    = nullptr;
    const char *      mask     = nullptr;  // null: no mask values to read
    size_t            mask_nb1 = 0;
    const char *      q        = nullptr;
    const char *      k        = nullptr;
    const char *      v        = nullptr;
    float *           dst      = nullptr;
    std::vector<item> items;
};

int64_t item_count(const shape & s, int np) {
    return s.n_kvh * ((s.n_q + np - 1) / np);
}

// Positions an item takes: rows near TARGET_ROWS, fewer until every thread
// has several items to take.
int positions_of_item(const shape & s, int threads) {
    int np = std::max(1, TARGET_ROWS / s.hg);
    while (np > 1 && item_count(s, np) < (int64_t) ITEMS_PER_THREAD * threads) {
        np = (np + 1) / 2;
    }
    return np;
}

int64_t keys_seen(const row_spans & sp, int p0, int np) {
    int64_t n = 0;
    for (int p = p0; p < p0 + np; p++) {
        n += std::max<int64_t>(0, (int64_t) sp.hi[p] - sp.lo[p] + 1);
    }
    return n;
}

void add_items_of_kv_head(std::vector<item> * items, const shape & s, const row_spans & sp, int kvh, int np) {
    for (int p0 = 0; p0 < s.n_q; p0 += np) {
        const int n = (int) std::min<int64_t>(np, s.n_q - p0);
        items->push_back({ kvh, p0, n, keys_seen(sp, p0, n) });
    }
}

// Largest first: the dynamic schedule then ends with the small ones.
std::vector<item> items_of(const shape & s, const row_spans & sp, int np) {
    std::vector<item> items;
    items.reserve((size_t) item_count(s, np));
    for (int kvh = 0; kvh < s.n_kvh; kvh++) {
        add_items_of_kv_head(&items, s, sp, kvh, np);
    }
    std::stable_sort(items.begin(), items.end(), [](const item & a, const item & b) { return a.cost > b.cost; });
    return items;
}

// Per-thread working memory: an item's O rows and its score block.
struct scratch {
    std::vector<float> mem;
    float *            o = nullptr;
    float *            s = nullptr;

    void reserve(int64_t dv) {
        constexpr size_t ALIGN = 64;
        const size_t     need  = (size_t) R_MAX * (size_t) (dv + KB) + ALIGN / sizeof(float);
        if (mem.size() < need) {
            mem.assign(need, 0.0f);
        }
        const uintptr_t at = ((uintptr_t) mem.data() + ALIGN - 1) & ~(uintptr_t) (ALIGN - 1);
        o                  = (float *) at;
        s                  = o + (size_t) R_MAX * dv;
    }
};

thread_local scratch t_scratch;

// The rows of an item and the state of their softmax.
struct item_rows {
    const float *    q[R_MAX];
    float *          dst[R_MAX];
    const uint16_t * mask[R_MAX];
    int32_t          lo[R_MAX], hi[R_MAX];
    float            M[R_MAX], L[R_MAX], rs[R_MAX];
    int32_t          tile_lo[TILES_MAX], tile_hi[TILES_MAX];  // the keys a tile of rows sees
    const char *     k = nullptr, * v = nullptr;              // this KV head's keys and values
    int              n = 0, tiles = 0;
};

void set_row(const plan & pl, const item & it, int pi, int hh, item_rows * r) {
    const shape & s    = pl.s;
    const int     pos  = it.p0 + pi;
    const int     row  = pi * s.hg + hh;
    const int     head = it.kvh * s.hg + hh;
    r->q[row]          = (const float *) (pl.q + (size_t) pos * s.q_nb1 + (size_t) head * s.q_nb2);
    r->dst[row]        = pl.dst + ((size_t) pos * s.n_head + head) * s.dv;
    r->mask[row]       = pl.mask ? (const uint16_t *) (pl.mask + (size_t) pos * pl.mask_nb1) : nullptr;
    r->lo[row]         = pl.spans->lo[pos];
    r->hi[row]         = pl.spans->hi[pos];
    r->M[row]          = NEG_INF;
    r->L[row]          = 0.0f;
    r->rs[row]         = 1.0f;
}

void set_position_rows(const plan & pl, const item & it, int pi, item_rows * r) {
    for (int hh = 0; hh < pl.s.hg; hh++) {
        set_row(pl, it, pi, hh, r);
    }
}

// The keys of the rows [first, last) of a tile (lo > hi: none).
void set_tile_span(item_rows * r, int tile, int first, int last) {
    int32_t lo = INT32_MAX, hi = -1;
    for (int row = first; row < last; row++) {
        if (r->lo[row] <= r->hi[row]) {
            lo = std::min(lo, r->lo[row]);
            hi = std::max(hi, r->hi[row]);
        }
    }
    r->tile_lo[tile] = lo;
    r->tile_hi[tile] = hi;
}

void set_tile_spans(item_rows * r) {
    r->tiles = (r->n + TILE_ROWS - 1) / TILE_ROWS;
    for (int t = 0; t < r->tiles; t++) {
        set_tile_span(r, t, t * TILE_ROWS, std::min(r->n, (t + 1) * TILE_ROWS));
    }
}

void set_item_rows(const plan & pl, const item & it, item_rows * r) {
    r->n = it.np * pl.s.hg;
    r->k = pl.k + (size_t) it.kvh * pl.s.k_nb2;
    r->v = pl.v + (size_t) it.kvh * pl.s.v_nb2;
    for (int pi = 0; pi < it.np; pi++) {
        set_position_rows(pl, it, pi, r);
    }
    set_tile_spans(r);
}

// The keys any tile of the item sees: [first, last], empty when last < first.
void key_range_of_item(const item_rows & r, int64_t * first, int64_t * last) {
    *first = INT64_MAX;
    *last  = -1;
    for (int t = 0; t < r.tiles; t++) {
        if (r.tile_lo[t] <= r.tile_hi[t]) {
            *first = std::min<int64_t>(*first, r.tile_lo[t]);
            *last  = std::max<int64_t>(*last, r.tile_hi[t]);
        }
    }
}

bool tile_in_block(const item_rows & r, int t, int64_t kb0, int nk) {
    return r.tile_lo[t] < kb0 + nk && r.tile_hi[t] >= kb0;
}

// The keys of a tile within a block, as offsets from the block's first key
struct tile_keys {
    int first = 0, end = 0;
};

tile_keys keys_of_tile(const item_rows & r, int t, int64_t kb0, int nk) {
    return { (int) (std::max<int64_t>(r.tile_lo[t], kb0) - kb0), (int) (std::min<int64_t>(r.tile_hi[t] + 1, kb0 + nk) - kb0) };
}

template <class Ops, typename T>
void scores_of_tile(const plan & pl, const item_rows & r, int t, int64_t kb0, int nk, float * S) {
    constexpr int GROUP  = TILE_KEYS;
    const tile_keys keys = keys_of_tile(r, t, kb0, nk);
    const int       rt   = std::min(TILE_ROWS, r.n - t * TILE_ROWS);
    const int       from = keys.first / GROUP * GROUP;  // a tile's key groups start on the group grid
    const char *    k    = r.k + (size_t) (kb0 + from) * pl.s.k_nb1;
    for (int j = from; j < keys.end; j += GROUP) {
        Ops::template qk<T>(rt, std::min(GROUP, nk - j), r.q + t * TILE_ROWS, k, pl.s.k_nb1, (int) pl.s.dk,
                            S + (size_t) t * TILE_ROWS * KB + j, KB);
        k += (size_t) GROUP * pl.s.k_nb1;
    }
}

template <class Ops, typename T>
void scores_of_block(const plan & pl, const item_rows & r, int64_t kb0, int nk, float * S) {
    for (int t = 0; t < r.tiles; t++) {
        if (tile_in_block(r, t, kb0, nk)) {
            scores_of_tile<Ops, T>(pl, r, t, kb0, nk, S);
        }
    }
}

template <class Ops>
void weights_of_block(const plan & pl, item_rows * r, int64_t kb0, int nk, float * S) {
    for (int row = 0; row < r->n; row++) {
        r->rs[row] = 1.0f;
        if (!tile_in_block(*r, row / TILE_ROWS, kb0, nk)) {
            continue;
        }
        const uint16_t * mask = pl.spans->pure || !r->mask[row] ? nullptr : r->mask[row] + kb0;
        r->rs[row] = Ops::softmax_row(S + (size_t) row * KB, (int) (r->lo[row] - kb0), (int) (r->hi[row] - kb0), nk,
                                      pl.s.scale, mask, &r->M[row], &r->L[row]);
    }
}

template <class Ops, typename T>
void accumulate_tile(const plan & pl, const item_rows & r, int t, int64_t kb0, int nk, const float * S, float * O) {
    const shape &   s      = pl.s;
    const tile_keys keys   = keys_of_tile(r, t, kb0, nk);
    const int       rt     = std::min(TILE_ROWS, r.n - t * TILE_ROWS);
    const int       chunks = (int) (s.dv / LANES);
    for (int c0 = 0; c0 < chunks; c0 += TILE_CHUNKS) {
        Ops::template pv<T>(rt, std::min(TILE_CHUNKS, chunks - c0), S + (size_t) t * TILE_ROWS * KB + keys.first, KB,
                            r.v + (size_t) (kb0 + keys.first) * s.v_nb1 + (size_t) c0 * LANES * sizeof(T), s.v_nb1,
                            keys.end - keys.first, O + (size_t) t * TILE_ROWS * s.dv + c0 * LANES, (int) s.dv,
                            r.rs + t * TILE_ROWS);
    }
}

template <class Ops, typename T>
void accumulate_block(const plan & pl, const item_rows & r, int64_t kb0, int nk, const float * S, float * O) {
    for (int t = 0; t < r.tiles; t++) {
        if (tile_in_block(r, t, kb0, nk)) {
            accumulate_tile<Ops, T>(pl, r, t, kb0, nk, S, O);
        }
    }
}

template <class Ops>
void write_rows(const plan & pl, const item_rows & r, const float * O) {
    for (int row = 0; row < r.n; row++) {
        Ops::finalize_row(O + (size_t) row * pl.s.dv, r.L[row], r.dst[row], (int) pl.s.dv);
    }
}

// The blocks of the grid of absolute keys that hold keys [first, last]
template <class Ops, typename T>
void run_blocks(const plan & pl, item_rows * r, int64_t first, int64_t last, const scratch & sc) {
    for (int64_t kb0 = first & ~(int64_t) (KB - 1); kb0 <= last; kb0 += KB) {
        const int nk = (int) std::min<int64_t>(KB, last + 1 - kb0);
        scores_of_block<Ops, T>(pl, *r, kb0, nk, sc.s);
        weights_of_block<Ops>(pl, r, kb0, nk, sc.s);
        accumulate_block<Ops, T>(pl, *r, kb0, nk, sc.s, sc.o);
    }
}

template <class Ops, typename T>
void run_item(const plan & pl, const item & it) {
    scratch & sc = t_scratch;
    sc.reserve(pl.s.dv);
    item_rows r;
    set_item_rows(pl, it, &r);
    int64_t first, last;
    key_range_of_item(r, &first, &last);
    std::fill(sc.o, sc.o + (size_t) r.n * pl.s.dv, 0.0f);
    run_blocks<Ops, T>(pl, &r, first, last, sc);
    write_rows<Ops>(pl, r, sc.o);
}

template <class Ops, typename T>
void run_items(const plan & pl) {
    const int64_t n = (int64_t) pl.items.size();
#pragma omp parallel for schedule(dynamic, 1) num_threads(xdna_chunk_threads())
    for (int64_t i = 0; i < n; i++) {
        run_item<Ops, T>(pl, pl.items[(size_t) i]);
    }
}

template <class Ops>
void run_plan(const plan & pl) {
    if (pl.s.kv_f16) {
        run_items<Ops, uint16_t>(pl);
    } else {
        run_items<Ops, float>(pl);
    }
}

bool avx512_available() {
#ifdef XDNA_HOST_ATTN_AVX512
    static const bool ok = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
                           __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq") &&
                           __builtin_cpu_supports("f16c") && __builtin_cpu_supports("fma");
    return ok;
#else
    return false;
#endif
}

// GGML_XDNA_HOST_FA: 0 ggml-cpu's attention, 1 these kernels where the CPU
// has the vector units for them, 2 the portable kernels.
enum class kernels { off, fast, portable };

kernels kernels_of_env() {
    const int mode = xdna_env_int("GGML_XDNA_HOST_FA", 1);
    if (mode == 0) {
        return kernels::off;
    }
    if (mode == 2) {
        return kernels::portable;
    }
    return avx512_available() ? kernels::fast : kernels::off;
}

void set_plan(const ggml_tensor * node, const row_spans * spans, plan * pl) {
    const ggml_tensor * m = node->src[3];
    pl->spans             = spans;
    pl->mask              = m && !spans->pure ? (const char *) m->data : nullptr;
    pl->mask_nb1          = m ? m->nb[1] : 0;
    pl->q                 = (const char *) node->src[0]->data;
    pl->k                 = (const char *) node->src[1]->data;
    pl->v                 = (const char *) node->src[2]->data;
    pl->dst               = (float *) node->data;
    pl->items             = items_of(pl->s, *spans, positions_of_item(pl->s, xdna_chunk_threads()));
}

}  // namespace

bool xdna_host_attn_supported(const ggml_tensor * node) {
    shape s;
    return node && kernels_of_env() != kernels::off && shape_of(node, &s);
}

bool xdna_host_attn_run(ggml_tensor * node) {
    const kernels which = kernels_of_env();
    plan          pl;
    if (!node || which == kernels::off || !shape_of(node, &pl.s)) {
        return false;
    }
    const auto spans = spans_of(node->src[3], pl.s);
    set_plan(node, spans.get(), &pl);
#ifdef XDNA_HOST_ATTN_AVX512
    if (which == kernels::fast) {
        run_plan<avx512_ops>(pl);
        return true;
    }
#endif
    run_plan<portable_ops>(pl);
    return true;
}

void xdna_host_attn_graph_begin(void) {
    std::lock_guard<std::mutex> lock(g_spans.mtx);
    g_spans.epoch++;
}
