#pragma once

#include "server-common.h"

#include <algorithm>
#include <cstdint>
#include <vector>

// --spec-draft-n-auto
//
// one draft length k per step for all generating slots: the k with the most expected tokens per second,
// E[len(k)] / T(n_gen * (k + 1) rows), from measured step times and per-position acceptance
// once the verify batch is compute bound, more slots means each slot should draft less
struct server_spec_n_auto {
    int32_t n_seq      = 0;
    int32_t n_max      = 0;
    int32_t k_min      = 1; // shortest non-empty draft (--spec-draft-n-min), shorter ones come back empty
    int32_t n_rows_max = 0;

    // step times per batch row count: last t_window samples, the first one dropped (graph setup)
    // the estimate is their minimum, since other server work only makes a step slower
    static constexpr int32_t t_window = 4;
    std::vector<double>  t_samples; // [n_rows_max + 1][t_window]
    std::vector<int32_t> n_timed;   // [n_rows_max + 1]

    // per n_gen: steps taken (for the probe) and the last k picked
    std::vector<int64_t> n_steps;
    std::vector<int32_t> k_last;

    // per draft position, all slots and per slot [n_seq][n_max]: decayed counts of verifies that reached / accepted it
    std::vector<double> n_reached;
    std::vector<double> n_hit;
    std::vector<double> n_reached_seq;
    std::vector<double> n_hit_seq;

    static constexpr double  p_decay     = 0.99; // count decay per verify, all slots
    static constexpr double  p_decay_seq = 0.95; // count decay per verify, own slot
    static constexpr double  p_prior     = 4.0;  // weight of the all-slot rate in a slot's rate, in verifies
    static constexpr double  r_keep      = 0.98; // keep the last k while it gives this fraction of the best rate
    static constexpr int64_t probe_every = 32;   // re-time a neighbor of the best k every this many steps

    void init(int32_t n_seq_, int32_t n_max_, int32_t n_min_) {
        n_seq      = std::max(1, n_seq_);
        n_max      = std::max(0, n_max_);
        k_min      = std::max(1, n_min_);
        n_rows_max = n_seq * (n_max + 1);
        t_samples    .assign((size_t) (n_rows_max + 1) * t_window, 0.0);
        n_timed      .assign((size_t) (n_rows_max + 1), 0);
        n_steps      .assign(n_seq, 0);
        k_last       .assign(n_seq, -1);
        n_reached    .assign(n_max, 0.0);
        n_hit        .assign(n_max, 0.0);
        n_reached_seq.assign((size_t) n_seq * n_max, 0.0);
        n_hit_seq    .assign((size_t) n_seq * n_max, 0.0);
    }

    bool ready() const {
        return !n_timed.empty();
    }

    bool allowed(int32_t k) const {
        return k == 0 || (k >= k_min && k <= n_max);
    }

    // 0 if not measured
    double t_meas(int32_t rows) const {
        if (rows < 1 || rows > n_rows_max) {
            return 0.0;
        }
        const int32_t n = std::min(n_timed[rows] - 1, t_window);
        double t = 0.0;
        for (int32_t s = 0; s < n; ++s) {
            const double ts = t_samples[(size_t) rows * t_window + s];
            t = t > 0.0 ? std::min(t, ts) : ts;
        }
        return t;
    }

    // unmeasured: the time of the nearest measured step with fewer rows (optimistic, so that it gets tried once)
    double t_est(int32_t rows) const {
        double t = t_meas(rows);
        for (int32_t r = std::min(rows, n_rows_max + 1) - 1; t <= 0.0 && r >= 1; --r) {
            t = t_meas(r);
        }
        return t;
    }

    // the slot's own rate, with the all-slot rate as prior
    double p_accept(int32_t seq, int32_t i) const {
        const double p_all = (n_hit[i] + 1.0) / (n_reached[i] + 2.0);
        const size_t j     = (size_t) seq * n_max + i;
        return (n_hit_seq[j] + p_prior * p_all) / (n_reached_seq[j] + p_prior);
    }

    // expected tokens per step: per slot 1 + sum_i prod_{j <= i} p_j
    double len(const std::vector<int32_t> & seqs, int32_t k) const {
        double e = 0.0;
        for (const int32_t seq : seqs) {
            double q = 1.0;
            e += 1.0;
            for (int32_t i = 0; i < k; ++i) {
                q *= p_accept(seq, i);
                e += q;
            }
        }
        return e;
    }

    // seqs: the generating slots
    int32_t pick(const std::vector<int32_t> & seqs) {
        const int32_t n_gen = std::min((int32_t) seqs.size(), n_seq);
        if (n_gen < 1) {
            return 0;
        }

        for (int32_t k = 0; k <= n_max; ++k) {
            if (allowed(k) && t_est(n_gen * (k + 1)) <= 0.0) {
                return k;
            }
        }

        int32_t k_best = 0;
        double  r_best = -1.0;
        for (int32_t k = 0; k <= n_max; ++k) {
            if (!allowed(k)) {
                continue;
            }
            const double r = len(seqs, k) / t_est(n_gen * (k + 1));
            if (r > r_best) {
                r_best = r;
                k_best = k;
            }
        }

        // steps in the same row tiles cost about the same, do not flip between them on noise
        int32_t & k_prev = k_last[n_gen - 1];
        if (k_prev >= 0 && k_prev != k_best && len(seqs, k_prev) / t_est(n_gen * (k_prev + 1)) >= r_keep * r_best) {
            k_best = k_prev;
        }
        if (k_best != k_prev) {
            SRV_INF("draft length for %d generating slots: %d (%.2f tokens per step in %.1f ms)\n",
                    n_gen, k_best, len(seqs, k_best), 1e3 * t_est(n_gen * (k_best + 1)));
            k_prev = k_best;
        }

        // the other times go stale as the context grows: time the next allowed k above or below now and then
        const int64_t i = n_steps[n_gen - 1]++;
        if (i % probe_every == probe_every - 1) {
            const int32_t dir = (i / probe_every) % 2 == 0 ? 1 : -1;
            for (int32_t k = k_best + dir; k >= 0 && k <= n_max; k += dir) {
                if (allowed(k)) {
                    return k;
                }
            }
        }

        return k_best;
    }

    // rows: the rows of a step that ran exactly the picked k on every generating slot
    void observe_step(int32_t rows, double t) {
        if (rows < 1 || rows > n_rows_max) {
            return;
        }
        if (n_timed[rows]++ == 0) {
            return;
        }
        t_samples[(size_t) rows * t_window + (n_timed[rows] - 2) % t_window] = t;
    }

    // a new task on the slot: its acceptance starts over
    void reset_seq(int32_t seq) {
        if (seq < 0 || seq >= n_seq) {
            return;
        }
        std::fill(n_reached_seq.begin() + (size_t) seq * n_max, n_reached_seq.begin() + (size_t) (seq + 1) * n_max, 0.0);
        std::fill(n_hit_seq    .begin() + (size_t) seq * n_max, n_hit_seq    .begin() + (size_t) (seq + 1) * n_max, 0.0);
    }

    void observe_verify(int32_t seq, int32_t n_draft, int32_t n_accepted) {
        if (seq < 0 || seq >= n_seq) {
            return;
        }
        double * reached = n_reached_seq.data() + (size_t) seq * n_max;
        double * hit     = n_hit_seq    .data() + (size_t) seq * n_max;
        for (int32_t i = 0; i < n_draft && i < n_max && i <= n_accepted; ++i) {
            const double a = i < n_accepted ? 1.0 : 0.0;
            n_reached[i] = p_decay     * n_reached[i] + 1.0;
            n_hit[i]     = p_decay     * n_hit[i]     + a;
            reached[i]   = p_decay_seq * reached[i]   + 1.0;
            hit[i]       = p_decay_seq * hit[i]       + a;
        }
    }
};
