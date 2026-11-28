#include "server-spec-n-auto.h"

#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <vector>

// step seconds of a 27B Q4_0 verify on an M3 Ultra by row count (llama-bench, one forward of n tokens)
static double t_m3(int32_t rows) {
    if (rows <= 1)  { return 0.025; }
    if (rows <= 8)  { return 0.039; }
    if (rows <= 16) { return 0.063; }
    if (rows <= 32) { return 0.111; }
    if (rows <= 48) { return 0.159; }
    return 0.204;
}

static std::vector<int32_t> seqs_of(int32_t n_gen) {
    std::vector<int32_t> seqs;
    for (int32_t i = 0; i < n_gen; ++i) {
        seqs.push_back(i);
    }
    return seqs;
}

// the first sample of a row count is dropped
static void time_rows(server_spec_n_auto & sa, int32_t rows, double t) {
    sa.observe_step(rows, t);
    if (sa.n_timed[rows] == 1) {
        sa.observe_step(rows, t);
    }
}

// runs n_steps steps with the given cost model and per-position acceptance p, returns how often each k was picked
// over the last half. like the server, a draft shorter than n_min comes back empty, and such a step (fewer rows than
// planned) is not timed
static std::map<int32_t, int32_t> simulate(server_spec_n_auto & sa, int32_t n_gen, int32_t n_steps, double p, int32_t n_min = 0) {
    // raw mt19937 output is the same everywhere, unlike the standard distributions
    std::mt19937 rng(42);
    auto u = [&rng]() { return (double) rng() / 4294967296.0; };
    std::map<int32_t, int32_t> hist;
    const auto seqs = seqs_of(n_gen);
    for (int32_t step = 0; step < n_steps; ++step) {
        const int32_t k_pick = sa.pick(seqs);
        if (step >= n_steps / 2) {
            hist[k_pick]++;
        }
        const int32_t k = k_pick < n_min ? 0 : k_pick;
        if (k == k_pick) {
            sa.observe_step(n_gen * (k + 1), t_m3(n_gen * (k + 1)));
        }
        for (const int32_t seq : seqs) {
            int32_t n_acc = 0;
            while (n_acc < k && u() < p) {
                ++n_acc;
            }
            if (k > 0) {
                sa.observe_verify(seq, k, n_acc);
            }
        }
    }
    return hist;
}

static int32_t mode_of(const std::map<int32_t, int32_t> & hist) {
    int32_t k_mode = -1;
    int32_t n_mode = -1;
    for (const auto & [k, n] : hist) {
        if (n > n_mode) {
            n_mode = n;
            k_mode = k;
        }
    }
    return k_mode;
}

// nothing measured: k = 0 is timed first, and its first sample does not count
static bool check_first_pick() {
    server_spec_n_auto sa;
    sa.init(1, 7, 0);
    const auto seqs = seqs_of(1);
    if (sa.pick(seqs) != 0) {
        return false;
    }
    sa.observe_step(1, 0.025);
    if (sa.t_meas(1) != 0.0 || sa.pick(seqs) != 0) {
        return false;
    }
    sa.observe_step(1, 0.025);
    return sa.t_meas(1) == 0.025;
}

// the estimate is the minimum of the last t_window samples, also when fewer are kept
static bool check_window_min() {
    server_spec_n_auto sa;
    sa.init(1, 3, 0);
    sa.observe_step(4, 0.010); // dropped
    sa.observe_step(4, 0.050);
    sa.observe_step(4, 0.040);
    if (sa.t_meas(4) != 0.040) {
        return false;
    }
    sa.observe_step(4, 0.060);
    sa.observe_step(4, 0.070);
    sa.observe_step(4, 0.080);
    if (sa.t_meas(4) != 0.040) {
        return false;
    }
    sa.observe_step(4, 0.090); // the 0.040 sample leaves the window
    return sa.t_meas(4) == 0.060;
}

// an unmeasured row count takes the time of the nearest measured smaller one
static bool check_estimate() {
    server_spec_n_auto sa;
    sa.init(2, 7, 0);
    time_rows(sa, 2, 0.030);
    time_rows(sa, 8, 0.040);
    return sa.t_est(5) == 0.030 && sa.t_est(12) == 0.040 && sa.t_est(1) == 0.0;
}

// probes stay in range when the best k is 0 or n_max
static bool check_probe_bounds() {
    for (const int32_t n_gen : { 1, 8 }) {
        server_spec_n_auto sa;
        sa.init(8, 7, 0);
        const auto hist = simulate(sa, n_gen, 400, n_gen == 1 ? 0.9 : 0.05);
        for (const auto & [k, n] : hist) {
            if (k < 0 || k > 7) {
                return false;
            }
        }
        if (mode_of(hist) != (n_gen == 1 ? 7 : 0)) {
            return false;
        }
    }
    return true;
}

// with --spec-draft-n-min, drafts shorter than it come back empty, so they are never picked
static bool check_n_min() {
    server_spec_n_auto sa;
    sa.init(1, 7, 3);
    const auto hist = simulate(sa, 1, 300, 0.7, 3);
    return hist.count(1) == 0 && hist.count(2) == 0 && mode_of(hist) == 7;
}

// the last k stays unless another one is more than (1 - r_keep) better
static bool check_hysteresis() {
    server_spec_n_auto sa;
    sa.init(1, 2, 0);
    const auto seqs = seqs_of(1);
    // no verifies yet: every position accepts with the prior 0.5, len(0..2) = 1, 1.5, 1.75
    time_rows(sa, 1, 0.025);  // 40 tokens/s
    time_rows(sa, 2, 0.030);  // 50
    time_rows(sa, 3, 0.0345); // 50.7
    sa.k_last[0] = 1;
    if (sa.pick(seqs) != 1) {
        return false;
    }
    sa.observe_step(3, 0.033); // 53.0
    return sa.pick(seqs) == 2;
}

// a new task on a slot drops the slot's acceptance, the all-slot rate stays as the prior
static bool check_reset_seq() {
    server_spec_n_auto sa;
    sa.init(2, 3, 0);
    for (int32_t i = 0; i < 20; ++i) {
        sa.observe_verify(0, 3, 0);
        sa.observe_verify(1, 3, 3);
    }
    const double p_all = (sa.n_hit[0] + 1.0) / (sa.n_reached[0] + 2.0);
    if (sa.p_accept(0, 0) >= p_all) {
        return false;
    }
    sa.reset_seq(0);
    return std::fabs(sa.p_accept(0, 0) - p_all) < 1e-12 && sa.p_accept(1, 0) > p_all;
}

// on the M3 cost model with p = 0.7 per position: 7 drafts for one slot, 1 for four, 0 or 1 for eight (within 5%)
static bool check_converges() {
    const struct {
        int32_t n_gen;
        int32_t k_lo;
        int32_t k_hi;
    } cases[] = { { 1, 7, 7 }, { 4, 1, 1 }, { 8, 0, 1 } };
    for (const auto & c : cases) {
        server_spec_n_auto sa;
        sa.init(8, 7, 0);
        const int32_t k = mode_of(simulate(sa, c.n_gen, 400, 0.7));
        if (k < c.k_lo || k > c.k_hi) {
            fprintf(stderr, "n_gen = %d: picked k = %d\n", c.n_gen, k);
            return false;
        }
    }
    return true;
}

int main() {
    const struct {
        bool (*fn)();
        const char * name;
    } checks[] = {
        { check_first_pick,   "nothing measured: k = 0 first, first sample dropped" },
        { check_window_min,   "estimate is the minimum of the last samples" },
        { check_estimate,     "unmeasured rows take the nearest smaller measured time" },
        { check_probe_bounds, "probes stay in range at k = 0 and k = n_max" },
        { check_n_min,        "drafts shorter than n_min are never picked" },
        { check_hysteresis,   "the last k stays within r_keep of the best" },
        { check_reset_seq,    "a new task drops the slot's acceptance" },
        { check_converges,    "converges on the M3 cost model" },
    };

    int failures = 0;
    for (const auto & check : checks) {
        if (!check.fn()) {
            fprintf(stderr, "FAIL: %s\n", check.name);
            ++failures;
        }
    }

    return failures == 0 ? 0 : 1;
}
