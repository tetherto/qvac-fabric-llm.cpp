#pragma once

// Minimal decode profiler, off by default (GGML_XDNA_PROF=1). Additive only:
// every call site this header touches runs exactly the same work in exactly
// the same order either way - the timers sit around existing calls, they do
// not add, remove, or reorder anything. When the env var is unset, every
// entry point here is a single already-cached bool check.
//
// What it times, and where the call sites are:
//   - each fused_layer dispatch (ggml-xdna.cpp, around xdna_rec_run()),
//     bucketed per GDN layer index.
//   - each decode GEMV dispatch (xdna-ops.cpp, gemv_compute_group()'s call to
//     xdna_gemv_run()), bucketed by the weight tensor name(s) it carries -
//     several projections can share one dispatch (xdna_ops_plan_gemv groups
//     them), so the label lists all of them.
//   - each host-glue CPU compute (ggml-xdna.cpp, xdna_glue_flush() and
//     xdna_glue_host_run()), bucketed by an exact op-type signature of the
//     batch ("ADD:1,MUL_MAT:1,...", sorted by name). A batch's CPU compute is
//     one call, so this is the finest true resolution available without
//     changing how the glue batches - a singleton batch gives an exact
//     per-op time, a mixed batch gives the batch's total under its exact
//     composition. glue_op_count is a separate, unconditionally exact node
//     count per op type (independent of batching).
//   - the whole ggml_backend_xdna_graph_compute() call, split into: the first
//     single-token decode call the process makes (decode_warmup - this is
//     where every weight's first-use packing and every layer's fused-core
//     creation happens, so it is not representative of a steady-state
//     token), every later single-token call (decode_total, what the
//     per-token report averages over), and any non-single-token call
//     (other_total: prefill / larger chunks).
//
// The three "current call" buckets (steady decode / first decode / other)
// are chosen once per graph_compute call by call_timer's constructor and
// read by every timer nested inside that same call.

#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace xdna_prof {

inline bool enabled(void) {
    static const bool v = [] {
        const char * e = getenv("GGML_XDNA_PROF");
        return e && e[0] && e[0] != '0';
    }();
    return v;
}

// GGML_XDNA_PROF=2 also lists, once, the nodes each single-token glue flush
// runs on the host: which ops the array still leaves to the CPU, by name.
inline bool list_glue(void) {
    static const bool v = [] {
        const char * e = getenv("GGML_XDNA_PROF");
        return e && e[0] == '2';
    }();
    return v;
}

template <typename T>
inline void glue_names(T * const * nodes, size_t n, int64_t n_tokens) {
    static int left = 400;
    if (!list_glue() || n_tokens != 1 || left <= 0) {
        return;
    }
    left--;
    fprintf(stderr, "[xdna-glue]");
    for (size_t j = 0; j < n; j++) {
        fprintf(stderr, " %s:%s", ggml_op_name(nodes[j]->op), nodes[j]->name);
    }
    fprintf(stderr, "\n");
}

using clock_t = std::chrono::steady_clock;

inline uint64_t now_ns(void) {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
               clock_t::now().time_since_epoch())
        .count();
}

struct accum {
    uint64_t ns = 0;
    uint64_t n  = 0;
    void add(uint64_t dt) {
        ns += dt;
        n++;
    }
};

// One set of the per-component buckets. Kept separate for the three call
// classes (steady decode / warm-up decode / other) so a report over `steady`
// never mixes in the one-time setup cost or a prefill chunk's glue.
struct buckets {
    std::map<int, accum>         fused_layer;   // key: GDN layer index (il)
    std::map<std::string, accum> gemv;          // key: weight name(s)
    std::map<std::string, accum> glue;          // key: op-type signature
    std::map<std::string, uint64_t> glue_op_count; // key: op name, exact node count
    // Time spent inside one of the buckets above, broken down. Reported on its
    // own and never added into the sum: every section is already counted by
    // the timer it nests in.
    std::map<std::string, accum> section;
};

struct state {
    std::mutex mu;

    accum decode_total;   // steady-state single-token graph_compute calls
    accum decode_warmup;  // the first single-token call only
    accum other_total;    // non single-token graph_compute calls (prefill)

    uint64_t decode_calls_seen = 0;

    // set by call_timer's constructor, read by every timer nested in the
    // same graph_compute call (single-threaded dispatch loop, see README
    // "One backend context per process holds the dispatch state").
    bool cur_is_decode = false;

    // fused_layer's own warm-up/steady split, at fire granularity rather
    // than call granularity (see fused_layer_timer). A decode token's
    // XDNA-claimed nodes are not always one graph_compute call - the
    // scheduler can split them - so the first CALL is not the same thing as
    // the first TOKEN: xdna_rec_run() does a layer's one-time session
    // creation and weight packing (tens of ms) on that layer's first fire,
    // and if that first fire lands in a call this header would otherwise
    // have called "steady" (because it is not the very first call), the
    // steady average for that layer is contaminated by a one-time cost. GDN
    // layer indices fire in ascending order within a token, so "the next
    // fire's index is not greater than the last one seen" is an exact
    // token-boundary signal, independent of how many calls the token spans.
    bool fl_first_token_done = false;
    int  fl_last_il          = -1;

    buckets steady;
    buckets warmup;
    buckets other;
};

inline state & S(void) {
    static state s;
    return s;
}

// Warmup here means "before the first token's fused_layer fires finished
// wrapping around" (see state::fl_first_token_done / fused_layer_timer) -
// read live, at the calling event's own completion time, not decided once
// at call start, since a decode token's XDNA-claimed nodes are not always
// one graph_compute call.
inline buckets & cur_bucket(void) {
    state & s = S();
    if (!s.cur_is_decode) {
        return s.other;
    }
    return !s.fl_first_token_done ? s.warmup : s.steady;
}

void print_summary(void);

inline void ensure_atexit_registered(void) {
    // Force state's construction *before* registering the exit hook. Magic
    // statics destruct in the reverse of their construction order,
    // interleaved with atexit calls in the reverse of their registration
    // order - so if the hook were registered first, `state` would be
    // destroyed (freeing every label string) before print_summary ever
    // runs, and print_summary would read already-freed std::string objects.
    // (This was an actual bug here: op-name strings, which are static string
    // literals ggml_op_name() returns, always printed fine; every
    // std::string this header stored itself came out as garbage bytes at
    // exit - a textbook static-destruction-order symptom.)
    S();
    static const bool once = [] {
        std::atexit(print_summary);
        return true;
    }();
    (void) once;
}

// Wraps one whole ggml_backend_xdna_graph_compute() call. `is_decode` is
// g_glue_n_tokens == 1, already computed by the caller before this is
// constructed - every timer nested inside the call reads what this decided.
struct call_timer {
    uint64_t t0     = 0;
    bool     on     = false;
    bool     is_decode = false;

    explicit call_timer(bool decode) : on(enabled()), is_decode(decode) {
        if (!on) {
            return;
        }
        ensure_atexit_registered();
        state & s = S();
        {
            std::lock_guard<std::mutex> lk(s.mu);
            s.cur_is_decode = decode;
            s.decode_calls_seen += decode ? 1 : 0;
        }
        t0 = now_ns();
    }
    ~call_timer() {
        if (!on) {
            return;
        }
        const uint64_t dt = now_ns() - t0;
        state & s = S();
        std::lock_guard<std::mutex> lk(s.mu);
        if (!is_decode) {
            s.other_total.add(dt);
        } else if (!s.fl_first_token_done) {
            // Decided here, at call end, not call start: if this call
            // contains the wrap fire (see fused_layer_timer), it already
            // belongs with the first token by the time we get here.
            s.decode_warmup.add(dt);
        } else {
            s.decode_total.add(dt);
        }
    }
};

struct fused_layer_timer {
    uint64_t t0 = 0;
    int      il;
    bool     on;
    bool     to_warmup = false;  // fire-granularity, not call-granularity - see state::fl_*
    explicit fused_layer_timer(int il_) : il(il_), on(enabled()) {
        if (!on) {
            return;
        }
        {
            state & s = S();
            std::lock_guard<std::mutex> lk(s.mu);
            if (!s.fl_first_token_done) {
                if (il <= s.fl_last_il) {
                    // Layer index did not increase: a new token's fires
                    // started, so the first token (whatever calls it spanned)
                    // is over as of this fire.
                    s.fl_first_token_done = true;
                    to_warmup             = false;
                } else {
                    s.fl_last_il = il;
                    to_warmup    = true;
                }
            }
        }
        t0 = now_ns();
    }
    ~fused_layer_timer() {
        if (!on) {
            return;
        }
        const uint64_t dt = now_ns() - t0;
        state & s = S();
        std::lock_guard<std::mutex> lk(s.mu);
        buckets & b = to_warmup ? s.warmup : s.steady;
        b.fused_layer[il].add(dt);
    }
};

struct gemv_timer {
    uint64_t    t0 = 0;
    std::string key;
    bool        on;
    explicit gemv_timer(std::string k) : key(std::move(k)), on(enabled()) {
        if (on) {
            t0 = now_ns();
        }
    }
    ~gemv_timer() {
        if (!on) {
            return;
        }
        const uint64_t dt = now_ns() - t0;
        std::lock_guard<std::mutex> lk(S().mu);
        cur_bucket().gemv[key].add(dt);
    }
};

// A named slice of work nested inside one of the other timers, e.g. the host's
// half of a fused-layer fire. Only ever reported next to that timer.
struct section_timer {
    uint64_t    t0 = 0;
    const char * key;
    bool        on;
    explicit section_timer(const char * k) : key(k), on(enabled()) {
        if (on) {
            t0 = now_ns();
        }
    }
    // Ends the section early; the destructor then does nothing.
    void stop() {
        if (!on) {
            return;
        }
        on = false;
        const uint64_t dt = now_ns() - t0;
        std::lock_guard<std::mutex> lk(S().mu);
        cur_bucket().section[key].add(dt);
    }
    ~section_timer() { stop(); }
};

// `sig` is the batch's op-type signature (see header comment); `op_names`
// lists every node's op name (with duplicates) for the exact per-op count.
struct glue_timer {
    uint64_t                t0 = 0;
    std::string              sig;
    std::vector<std::string> op_names;
    bool                      on;
    glue_timer(std::string s, std::vector<std::string> ops)
        : sig(std::move(s)), op_names(std::move(ops)), on(enabled()) {
        if (on) {
            t0 = now_ns();
        }
    }
    ~glue_timer() {
        if (!on) {
            return;
        }
        const uint64_t dt = now_ns() - t0;
        std::lock_guard<std::mutex> lk(S().mu);
        buckets & b = cur_bucket();
        b.glue[sig].add(dt);
        for (auto & o : op_names) {
            b.glue_op_count[o]++;
        }
    }
};

// Builds a canonical "OP:count,OP2:count2" signature, sorted by op name. Used
// for both a single-node glue run (trivially exact) and a batch flush (exact
// down to the batch, which is the finest the CPU compute call exposes).
inline std::string glue_signature(const std::vector<std::string> & op_names) {
    std::map<std::string, int> counts;
    for (const auto & o : op_names) {
        counts[o]++;
    }
    std::string sig;
    for (const auto & kv : counts) {
        if (!sig.empty()) {
            sig += ",";
        }
        sig += kv.first + ":" + std::to_string(kv.second);
    }
    return sig;
}

inline void print_bucket(const char * label, const buckets & b, double denom_tokens) {
    if (b.fused_layer.empty() && b.gemv.empty() && b.glue.empty()) {
        return;
    }
    fprintf(stderr, "  -- %s --\n", label);

    uint64_t fl_ns = 0, fl_n = 0;
    fprintf(stderr, "  fused_layer fires, per GDN layer index:\n");
    for (const auto & kv : b.fused_layer) {
        const accum & a = kv.second;
        fl_ns += a.ns;
        fl_n  += a.n;
        fprintf(stderr,
                "    layer %2d: %6" PRIu64 " fires, %8.1f us/fire, %9.1f us/token\n",
                kv.first, a.n, a.n ? (double) a.ns / a.n / 1e3 : 0.0,
                denom_tokens > 0 ? (double) a.ns / 1e3 / denom_tokens : 0.0);
    }
    fprintf(stderr, "    TOTAL: %" PRIu64 " fires, %.1f us/fire avg, %.1f us/token\n",
            fl_n, fl_n ? (double) fl_ns / fl_n / 1e3 : 0.0,
            denom_tokens > 0 ? (double) fl_ns / 1e3 / denom_tokens : 0.0);

    uint64_t gv_ns = 0, gv_n = 0;
    fprintf(stderr, "  decode GEMV dispatches, per label:\n");
    for (const auto & kv : b.gemv) {
        const accum & a = kv.second;
        gv_ns += a.ns;
        gv_n  += a.n;
        fprintf(stderr, "    %-60s %6" PRIu64 " fires, %8.1f us/fire, %9.1f us/token\n",
                kv.first.c_str(), a.n, a.n ? (double) a.ns / a.n / 1e3 : 0.0,
                denom_tokens > 0 ? (double) a.ns / 1e3 / denom_tokens : 0.0);
    }
    fprintf(stderr, "    TOTAL: %" PRIu64 " fires, %.1f us/fire avg, %.1f us/token\n",
            gv_n, gv_n ? (double) gv_ns / gv_n / 1e3 : 0.0,
            denom_tokens > 0 ? (double) gv_ns / 1e3 / denom_tokens : 0.0);

    uint64_t gl_ns = 0, gl_n = 0;
    fprintf(stderr, "  host-glue CPU flushes, per op-type signature:\n");
    for (const auto & kv : b.glue) {
        const accum & a = kv.second;
        gl_ns += a.ns;
        gl_n  += a.n;
        fprintf(stderr, "    %-60s %6" PRIu64 " flushes, %8.1f us/flush, %9.1f us/token\n",
                kv.first.c_str(), a.n, a.n ? (double) a.ns / a.n / 1e3 : 0.0,
                denom_tokens > 0 ? (double) a.ns / 1e3 / denom_tokens : 0.0);
    }
    fprintf(stderr, "    TOTAL: %" PRIu64 " flushes, %.1f us/flush avg, %.1f us/token\n",
            gl_n, gl_n ? (double) gl_ns / gl_n / 1e3 : 0.0,
            denom_tokens > 0 ? (double) gl_ns / 1e3 / denom_tokens : 0.0);
    fprintf(stderr, "  host-glue exact node counts by op type (independent of batching):\n");
    for (const auto & kv : b.glue_op_count) {
        fprintf(stderr, "    %-20s %" PRIu64 " nodes%s\n", kv.first.c_str(), kv.second,
                denom_tokens > 0 ? "" : "");
    }

    if (!b.section.empty()) {
        fprintf(stderr, "  nested sections (already inside the buckets above, NOT in the sum):\n");
        for (const auto & kv : b.section) {
            const accum & a = kv.second;
            fprintf(stderr, "    %-60s %6" PRIu64 " times, %8.1f us/each, %9.1f us/token\n",
                    kv.first.c_str(), a.n, a.n ? (double) a.ns / a.n / 1e3 : 0.0,
                    denom_tokens > 0 ? (double) a.ns / 1e3 / denom_tokens : 0.0);
        }
    }

    fprintf(stderr,
            "  captured sum (fused_layer + gemv + glue): %.1f us/token\n",
            denom_tokens > 0 ? (double) (fl_ns + gv_ns + gl_ns) / 1e3 / denom_tokens : 0.0);
}

// The mode of the per-GDN-layer fire counts in a bucket: every complete
// decode token fires every GDN layer exactly once, so once a bucket only
// holds complete tokens (see fused_layer_timer / fl_first_token_done), every
// layer's count should agree. Used as the per-token denominator instead of
// a call count, since one token's XDNA-claimed nodes are not always one
// graph_compute call (see state::fl_first_token_done for why that matters).
inline uint64_t mode_fire_count(const std::map<int, accum> & fl) {
    std::map<uint64_t, int> hist;
    for (const auto & kv : fl) {
        hist[kv.second.n]++;
    }
    uint64_t best     = 0;
    int      best_cnt = -1;
    for (const auto & kv : hist) {
        if (kv.second > best_cnt) {
            best_cnt = kv.second;
            best     = kv.first;
        }
    }
    return best;
}

inline void print_summary(void) {
    state & s = S();
    std::lock_guard<std::mutex> lk(s.mu);
    fprintf(stderr, "\n[xdna-prof] GGML_XDNA_PROF summary (stderr, at process exit)\n");
    fprintf(stderr,
            "[xdna-prof] steady-state decode tokens: the whole first decode "
            "token is excluded (decode_warmup below), however many "
            "graph_compute calls the scheduler split it into - it is where "
            "every weight's first-use packing and every fused_layer "
            "session's creation happens, so it is not representative. The "
            "boundary is exact: GDN layer indices fire in ascending order "
            "within a token, so the first fire whose index does not exceed "
            "the previous one marks the first token's end.\n");

    fprintf(stderr, "\n[xdna-prof] decode_warmup (the first decode token, all its calls):\n");
    if (s.decode_warmup.n) {
        fprintf(stderr, "  whole call(s): %.3f ms total over %" PRIu64 " graph_compute call(s)\n",
                (double) s.decode_warmup.ns / 1e6, s.decode_warmup.n);
    } else {
        fprintf(stderr, "  (none seen)\n");
    }
    print_bucket("decode_warmup breakdown", s.warmup, s.decode_warmup.n ? 1.0 : 0.0);

    const uint64_t n_tok_i = mode_fire_count(s.steady.fused_layer);
    const double   n_tok   = (double) n_tok_i;
    fprintf(stderr,
            "\n[xdna-prof] decode steady state: %" PRIu64 " graph_compute calls, "
            "inferred as %" PRIu64 " real decode tokens (the mode of each GDN "
            "layer's fire count - a token can span more than one call, so this "
            "is not the same number; see the per-layer fire counts below, "
            "they should all agree)\n",
            s.decode_total.n, n_tok_i);
    if (n_tok_i) {
        fprintf(stderr, "  whole ggml_backend_xdna_graph_compute(): %.1f us/token total "
                        "(%.1f us/call, %" PRIu64 " calls)\n",
                (double) s.decode_total.ns / 1e3 / n_tok,
                s.decode_total.n ? (double) s.decode_total.ns / 1e3 / (double) s.decode_total.n : 0.0,
                s.decode_total.n);
    }
    print_bucket("decode steady-state breakdown", s.steady, n_tok);
    if (n_tok_i) {
        // Find the captured sum again to report the remainder explicitly:
        // whole-call time this header did not attribute to fused_layer, gemv,
        // or glue - loop bookkeeping, xdna_ops_finalize() waits between them,
        // consumed-set/rec_caps bookkeeping, and anything else the dispatch
        // loop does between the instrumented calls.
        uint64_t fl = 0, gv = 0, gl = 0;
        for (auto & kv : s.steady.fused_layer) fl += kv.second.ns;
        for (auto & kv : s.steady.gemv)        gv += kv.second.ns;
        for (auto & kv : s.steady.glue)        gl += kv.second.ns;
        const double captured_us  = (double) (fl + gv + gl) / 1e3 / n_tok;
        const double whole_us     = (double) s.decode_total.ns / 1e3 / n_tok;
        fprintf(stderr,
                "  remainder (whole call - fused_layer - gemv - glue): %.1f us/token "
                "(%.1f%% of the whole graph_compute call)\n",
                whole_us - captured_us, whole_us > 0 ? 100.0 * (whole_us - captured_us) / whole_us : 0.0);
    }

    fprintf(stderr, "\n[xdna-prof] other (non single-token, e.g. prefill) graph_compute calls: %"
                    PRIu64 ", %.3f ms total\n",
            s.other_total.n, (double) s.other_total.ns / 1e6);
    print_bucket("other breakdown", s.other, s.other_total.n ? (double) s.other_total.n : 0.0);
    fprintf(stderr, "[xdna-prof] end of summary\n\n");
}

}  // namespace xdna_prof
