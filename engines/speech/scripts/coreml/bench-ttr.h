// bench-ttr.h: the time-to-ready bookkeeping of bench-load-driver.mm, kept
// free of Core ML and of the runtime so test_bench_ttr.cpp can drive it with
// fakes on any host.
//
//   poll_slots    samples slot states until the group finishes, recording
//                 per-slot ready times and deferred(memory) time.
//   run_manual    the hand-built baseline: one prepare thread and one load
//                 thread, optionally sharing one mutex (the gate by hand).
//                 Every model that loads is retained until the caller drops
//                 the outcome, so both baselines measure the same resident
//                 workload as the Manager, whose slots hold their models.
//   with_readings the order of a run's readings: start readings before the
//                 run loads anything, end readings while every model it
//                 loaded is still held, then the models are released.

#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace bench_ttr {

enum class Kind { other, deferred_memory, ready };

struct Observation {
    std::string name;  // the state's display name, recorded on every change
    Kind kind = Kind::other;
};

struct Transition {
    double t_ms;  // since issue
    size_t slot;
    std::string state;
};

struct PollOutcome {
    std::vector<double> ready_at;     // ms since issue; -1 = never seen ready
    std::vector<double> deferred_ms;  // time spent in deferred(memory)
    std::vector<Transition> transitions;
    bool finished = false;  // the group reported completion (not the deadline)
};

// sample(i) -> Observation, finished() -> bool, now() -> ms, sleep() -> void.
//
// Completion is read *before* the states are sampled, and the loop exits only
// after that sample: a load that becomes ready between a sample and the
// completion check is therefore always seen ready.
template <class Sample, class Finished, class Now, class Sleep>
PollOutcome poll_slots(size_t n, Sample sample, Finished finished, Now now, Sleep sleep, double t_issue,
                       double deadline) {
    PollOutcome o;
    o.ready_at.assign(n, -1.0);
    o.deferred_ms.assign(n, 0.0);
    std::vector<std::string> last(n);
    double prev = now();
    while (true) {
        const bool done = finished();
        const double t = now();
        for (size_t i = 0; i < n; ++i) {
            const Observation s = sample(i);
            if (s.kind == Kind::deferred_memory) o.deferred_ms[i] += t - prev;
            if (s.name != last[i]) {
                o.transitions.push_back({t - t_issue, i, s.name});
                last[i] = s.name;
            }
            if (s.kind == Kind::ready && o.ready_at[i] < 0) o.ready_at[i] = t - t_issue;
        }
        prev = t;
        if (done) {
            o.finished = true;
            break;
        }
        if (t > deadline) break;
        sleep();
    }
    return o;
}

// The latest ready time, or -1 when any slot never became ready.
inline double all_ready_ms(const std::vector<double> & ready_at) {
    double m = 0;
    for (double r : ready_at) {
        if (r < 0) return -1;
        m = std::max(m, r);
    }
    return m;
}

struct ManualStage {
    std::string stage;
    std::string path;
    std::vector<std::string> functions;
};

struct ManualResult {
    std::string slot;  // stage/function
    double ready_ms;   // -1 on failure
    bool ok;
};

struct ManualOutcome {
    std::vector<ManualResult> results;
    std::vector<std::shared_ptr<void>> retained;  // every model that loaded and warmed
};

// prepare(stage, path, out_path) -> bool, run only when `generate`;
// load(stage, path, function) -> std::shared_ptr<void>, the loaded and warmed
// model or null on failure; now() -> ms.
template <class Prepare, class Load, class Now>
ManualOutcome run_manual(const std::vector<ManualStage> & stages, bool generate, bool gated, Prepare prepare,
                         Load load, Now now, double t_issue) {
    struct Item {
        std::string stage, path, fn;
    };
    std::mutex gate, q_mu;
    std::condition_variable q_cv;
    std::deque<Item> loads;
    bool prepares_done = false;
    ManualOutcome out;

    std::thread prep([&] {
        for (const ManualStage & s : stages) {
            std::string path = s.path;
            if (generate) {
                std::unique_lock<std::mutex> g(gate, std::defer_lock);
                if (gated) g.lock();
                std::string prepared;
                if (prepare(s.stage, s.path, prepared)) path = prepared;
                else path.clear();
            }
            std::lock_guard<std::mutex> lk(q_mu);
            for (const std::string & fn : s.functions) loads.push_back({s.stage, path, fn});
            q_cv.notify_all();
        }
        std::lock_guard<std::mutex> lk(q_mu);
        prepares_done = true;
        q_cv.notify_all();
    });

    std::thread loader([&] {
        while (true) {
            Item it;
            {
                std::unique_lock<std::mutex> lk(q_mu);
                q_cv.wait(lk, [&] { return !loads.empty() || prepares_done; });
                if (loads.empty()) return;
                it = loads.front();
                loads.pop_front();
            }
            std::shared_ptr<void> model;
            if (!it.path.empty()) {
                std::unique_lock<std::mutex> g(gate, std::defer_lock);
                if (gated) g.lock();
                model = load(it.stage, it.path, it.fn);
            }
            const bool ok = model != nullptr;
            out.results.push_back({it.stage + "/" + it.fn, ok ? now() - t_issue : -1.0, ok});
            if (ok) out.retained.push_back(std::move(model));
        }
    });
    prep.join();
    loader.join();
    return out;
}

inline double manual_ready_ms(const std::vector<ManualResult> & results) {
    double m = 0;
    for (const ManualResult & r : results) {
        if (r.ready_ms < 0) return -1;
        m = std::max(m, r.ready_ms);
    }
    return m;
}

template <class S, class R, class E>
struct Readings {
    S start;  // taken before the run
    R run;    // what the run returned
    E end;    // taken while the run's models are still held
};

// start() -> S; run(std::shared_ptr<void> & keep) -> R, which hands every model
// it loaded to `keep`; end() -> E. `keep` is released after end().
template <class Start, class Run, class End>
auto with_readings(Start start, Run run, End end) {
    auto s = start();
    std::shared_ptr<void> keep;
    auto r = run(keep);
    auto e = end();
    keep.reset();
    return Readings<decltype(s), decltype(r), decltype(e)>{std::move(s), std::move(r), std::move(e)};
}

}  // namespace bench_ttr
