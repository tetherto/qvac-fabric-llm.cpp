// test-coreml-exit-load: a process that ends while the runtime's load worker
// is inside a real Core ML model load must still exit cleanly.
//
// Each run copies the model to a fresh path (so Core ML compiles it again
// rather than reusing a plan cached for an earlier path) and spawns a child
// under its own code-signing identity: a copy of this program, ad-hoc signed
// with a unique identifier, so its e5rt cache (~/Library/Caches/<identifier>,
// ~20 MB per compile of a vocoder) is its own and is deleted after the run,
// whatever else runs on the host. The child makes a prebuilt slot on the
// per-image manager, requests it, waits until the slot is loading, sleeps for
// that run's delay and returns from main. The delays sweep 0..--max-delay-ms
// in equal steps (default 60 runs over 0-360 ms), so some exits land in each
// early phase of the load. The fatal window is the in-process validation of
// the program for the Neural Engine: for an exported Supertonic vocoder on an
// M2, exits 160-330 ms after the load started crashed without the exit wait.
//
// A run fails if the child dies of a signal (the failure this guards against
// is a SIGSEGV inside Apple's compiler while exit() runs static destructors),
// exits non-zero (also when its load never got in flight within 60 s), takes
// longer than 20 s to exit (a hang: the exit wait's bound is 15 s), or ends
// in any state but ready, loading or warming. Where the exit wait applies
// (engine on the main thread, or armed, or drained), a run that exited with
// the load in flight must also show that Core ML returned a model before
// any destructor registered during the load had run. Before the sweep, one control
// child must load the model to ready; and at least half of the runs must exit
// with the load in flight, since a run whose load already finished checks
// nothing.
//
// Before it starts the runtime, the child creates a static object whose
// destructor blocks for --finalizer-ms (default 500), as ggml's Metal backend
// does when it frees its devices at exit in every engine process. It is
// destroyed after the frameworks' statics (it was constructed first), so a
// load left running at exit keeps going inside a half torn-down process
// instead of being cut short by the end of a 2 ms exit. The child also runs
// with MallocScribble=1 (unless --no-scribble), so memory a destructor freed
// is overwritten at once: Apple's compiler keeps lookup tables in
// function-local statics, built on first use during the compile and therefore
// destroyed first at exit, and a read through a freed table then faults every
// time instead of only when another destructor happened to reuse the memory.
//
// Where the engine lives (for measurements; the default is what the test
// asserts):
//   --engine-thread main     the main thread makes and requests the slot, so
//                            it arms the exit guard (default);
//   --engine-thread worker   another thread does (an addon's load), so
//                            nothing arms the guard unless --arm is given;
//   --arm                    the main thread calls arm_exit_guard() first;
//   --drain-ms D             the main thread calls shutdown_for_exit(D)
//                            before it returns;
//   --exit-wait-ms W         the exit wait's bound (0: no wait, the behaviour
//                            before the wait existed; used as a control).
//   --report-only            print the results, exit 0 whatever they are.
//
//   test-coreml-exit-load --model <x.mlmodelc> [--runs N] [--max-delay-ms M]
//                         [--units all|cpu_and_ane|cpu_and_gpu|cpu_only]
//                         [--finalizer-ms F] [--no-scribble] [options above]

#include "qvac_coreml/manager.h"

#include "test_hooks.h"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char ** environ;

using namespace QVAC_COREML_NS;

namespace {

constexpr int kHangMs = 20000;

struct ExitTestOptions {
    std::string model;
    std::string units = "all";
    int runs = 60;
    int max_delay = 360;
    int finalizer_ms = 500;
    bool scribble = true;
    bool worker = false;
    bool arm = false;
    int drain_ms = -1;
    int exit_wait_ms = -1;
    bool report_only = false;
};

int64_t mono_ns() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);  // system-wide: comparable across processes
    return int64_t(ts.tv_sec) * 1000000000 + ts.tv_nsec;
}

MLComputeUnits ml_units(ComputeUnits u) {
    switch (u) {
        case ComputeUnits::cpu_only: return MLComputeUnitsCPUOnly;
        case ComputeUnits::cpu_and_gpu: return MLComputeUnitsCPUAndGPU;
        case ComputeUnits::cpu_and_ane: return MLComputeUnitsCPUAndNeuralEngine;
        case ComputeUnits::all: return MLComputeUnitsAll;
    }
    return MLComputeUnitsAll;
}

std::atomic<bool> g_torn_down{false};

struct TearDownMarker {
    ~TearDownMarker() { g_torn_down = true; }
};

// Written straight to stdout (fd 1): it may run while exit() is in progress.
void report_line(const std::string & s) {
    const std::string line = s + "\n";
    (void) !::write(1, line.data(), line.size());
}

LoadResult load_mlmodel(const LoadRequest & r) {
    // Built during the load, as the frameworks build their lazy statics, so
    // its destructor is among the first that exit() runs.
    static TearDownMarker lazy;
    (void) lazy;
    LoadResult out;
    struct Report {
        bool native_ok = false;
        ~Report() {
            report_line(std::string("load=") + (g_torn_down ? "torn-down" : "intact") +
                        " native_load=" + (native_ok ? "ok" : "failed"));
        }
    } report;
    @autoreleasepool {
        MLModelConfiguration * cfg = [MLModelConfiguration new];
        cfg.computeUnits = ml_units(r.units);
        NSError * err = nil;
        NSURL * url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:r.artifact_path.c_str()]];
        MLModel * m = [MLModel modelWithContentsOfURL:url configuration:cfg error:&err];
        if (m == nil) {
            out.error = Error::deterministic(err ? err.localizedDescription.UTF8String : "load failed");
            return out;
        }
        out.model = std::shared_ptr<void>((__bridge_retained void *) m, [](void * p) { CFRelease(p); });
        report.native_ok = true;
    }
    return out;
}

struct SlowFinalizer {
    int ms;
    ~SlowFinalizer() { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }
};

// Child: request the load, wait for it to be in flight, then end the process
// `delay_ms` later by returning from main (delay < 0: the control run, which
// waits for the load to finish). Prints the state it exited in.
int child(const ExitTestOptions & o, int delay_ms) {
    static SlowFinalizer slow{o.finalizer_ms};
    if (o.exit_wait_ms >= 0) detail::test_hooks().exit_wait_ms = o.exit_wait_ms;
    if (o.arm && !arm_exit_guard()) return 66;
    SlotSpec s;
    s.engine = "exit-load";
    s.stage = "model";
    s.units = *parse_compute_units(o.units);
    s.route = Route::prebuilt;
    s.artifact_path = o.model;
    s.load_fn = &load_mlmodel;
    // Leaked: the engine's slot stays alive through exit, as an addon's does.
    auto * slot = new std::shared_ptr<Slot>();
    auto make = [&] {
        *slot = Manager::instance()->make_slot(s, SlotPolicy{});
        (*slot)->request(Priority::minimal);
    };
    if (o.worker) {
        std::thread(make).join();
    } else {
        make();
    }
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    auto settled = [&](SlotState st) {
        return st == SlotState::ready || st == SlotState::failed ||
               (delay_ms >= 0 && (st == SlotState::loading || st == SlotState::warming));
    };
    SlotState st = (*slot)->state();
    while (!settled(st) && std::chrono::steady_clock::now() < end) {
        std::this_thread::sleep_for(std::chrono::microseconds(200));
        st = (*slot)->state();
    }
    if (!settled(st)) {
        std::printf("state=%s never-in-flight\n", to_string(st));
        return 70;
    }
    if (delay_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    st = (*slot)->state();
    std::printf("state=%s exit_ns=%lld\n", to_string(st), (long long) mono_ns());
    std::fflush(stdout);
    if (o.drain_ms >= 0) shutdown_for_exit(std::chrono::milliseconds(o.drain_ms));
    return 0;
}

struct RunResult {
    bool exited = false;
    int status = 0;
    std::string state;
    std::string load;  // "intact", "torn-down" or "" (the load never returned)
    std::string native_load;  // "ok", "failed" or "" (the load never returned)
    double exit_ms = -1;  // from the child's report (just before main returns) to its reaping
};

bool copy_item(const std::string & src, const std::string & dst) {
    NSError * err = nil;
    return [[NSFileManager defaultManager] copyItemAtPath:[NSString stringWithUTF8String:src.c_str()]
                                                   toPath:[NSString stringWithUTF8String:dst.c_str()]
                                                    error:&err];
}

void remove_tree(const std::string & p) {
    [[NSFileManager defaultManager] removeItemAtPath:[NSString stringWithUTF8String:p.c_str()] error:nil];
}

int spawn_wait(const std::vector<std::string> & args, const std::string & out_path, int timeout_s, int & status,
               int64_t & reaped_ns) {
    std::vector<char *> argv;
    for (const std::string & a : args) argv.push_back(const_cast<char *>(a.c_str()));
    argv.push_back(nullptr);
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, 1, out_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);
    pid_t pid = 0;
    const int rc = posix_spawn(&pid, argv[0], &fa, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&fa);
    if (rc != 0) return -1;
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
    while (std::chrono::steady_clock::now() < end) {
        if (waitpid(pid, &status, WNOHANG) == pid) {
            reaped_ns = mono_ns();
            return 1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    kill(pid, SIGKILL);
    waitpid(pid, &status, 0);
    return 0;
}

// Copies this program under a new name and signs it ad hoc with that name as
// its identifier: a host the e5rt cache has never seen.
std::string fresh_identity(const std::string & self, const std::string & dir, std::string & ident) {
    char id[64];
    std::snprintf(id, sizeof id, "qvac-exit-load-%d-%08x", getpid(), arc4random());
    ident = id;
    const std::string bin = dir + "/" + ident;
    if (!copy_item(self, bin)) return "";
    int status = 0;
    int64_t t = 0;
    if (spawn_wait({"/usr/bin/codesign", "-f", "-s", "-", "-i", ident, bin}, dir + "/codesign.out", 60, status, t) != 1 ||
        !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        return "";
    }
    return bin;
}

std::vector<std::string> child_args(const ExitTestOptions & o, const std::string & bin, const std::string & model,
                                    int delay_ms) {
    std::vector<std::string> a = {bin,        "--child-run",  model, "--delay", std::to_string(delay_ms),
                                  "--units",  o.units,        "--finalizer-ms", std::to_string(o.finalizer_ms)};
    if (o.worker) a.insert(a.end(), {"--engine-thread", "worker"});
    if (o.arm) a.push_back("--arm");
    if (o.drain_ms >= 0) a.insert(a.end(), {"--drain-ms", std::to_string(o.drain_ms)});
    if (o.exit_wait_ms >= 0) a.insert(a.end(), {"--exit-wait-ms", std::to_string(o.exit_wait_ms)});
    return a;
}

RunResult run_once(const ExitTestOptions & o, const std::string & self, int delay_ms) {
    RunResult rr;
    const char * tmp = std::getenv("TMPDIR");
    std::string tmpl = std::string(tmp && *tmp ? tmp : "/tmp") + "/coreml-exit-load-XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    if (mkdtemp(buf.data()) == nullptr) return rr;
    const std::string dir = buf.data();
    std::string name = o.model;
    while (!name.empty() && name.back() == '/') name.pop_back();
    name = name.substr(name.find_last_of('/') + 1);
    const std::string copy = dir + "/" + name;
    std::string ident;
    const std::string bin = copy_item(o.model, copy) ? fresh_identity(self, dir, ident) : "";
    if (bin.empty()) {
        std::fprintf(stderr, "cannot stage the model copy or the signed program copy in %s\n", dir.c_str());
        remove_tree(dir);
        return rr;
    }
    if (o.scribble) setenv("MallocScribble", "1", 1);
    int64_t reaped_ns = 0;
    const int w = spawn_wait(child_args(o, bin, copy, delay_ms), dir + "/child.out", 120, rr.status, reaped_ns);
    unsetenv("MallocScribble");
    rr.exited = w == 1;
    if (FILE * f = std::fopen((dir + "/child.out").c_str(), "r")) {
        char line[512] = {0};
        while (std::fgets(line, sizeof line, f)) {
            char st[64] = {0};
            char native[64] = {0};
            long long ns = 0;
            if (std::sscanf(line, "state=%63s exit_ns=%lld", st, &ns) == 2) {
                rr.state = st;
                if (reaped_ns > 0) rr.exit_ms = double(reaped_ns - ns) / 1e6;
            } else if (std::sscanf(line, "load=%63s native_load=%63s", st, native) == 2) {
                rr.load = st;
                rr.native_load = native;
            }
        }
        std::fclose(f);
    }
    remove_tree(dir);
    const char * home = std::getenv("HOME");
    if (home && !ident.empty()) remove_tree(std::string(home) + "/Library/Caches/" + ident);
    return rr;
}

std::string describe(const RunResult & r) {
    char how[64];
    if (!r.exited) {
        std::snprintf(how, sizeof how, "timed out");
    } else if (WIFSIGNALED(r.status)) {
        std::snprintf(how, sizeof how, "signal %d", WTERMSIG(r.status));
    } else {
        std::snprintf(how, sizeof how, "exit %d", WEXITSTATUS(r.status));
    }
    return how;
}

bool parse(int argc, char ** argv, ExitTestOptions & o, bool & is_child, int & delay) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : nullptr; };
        const char * v = nullptr;
        if (a == "--child-run" && (v = next())) {
            is_child = true;
            o.model = v;
        } else if (a == "--delay" && (v = next())) {
            delay = std::atoi(v);
        } else if (a == "--model" && (v = next())) {
            o.model = v;
        } else if (a == "--runs" && (v = next())) {
            o.runs = std::atoi(v);
        } else if (a == "--max-delay-ms" && (v = next())) {
            o.max_delay = std::atoi(v);
        } else if (a == "--units" && (v = next())) {
            o.units = v;
        } else if (a == "--finalizer-ms" && (v = next())) {
            o.finalizer_ms = std::atoi(v);
        } else if (a == "--no-scribble") {
            o.scribble = false;
        } else if (a == "--engine-thread" && (v = next())) {
            if (std::strcmp(v, "worker") == 0) {
                o.worker = true;
            } else if (std::strcmp(v, "main") != 0) {
                return false;
            }
        } else if (a == "--arm") {
            o.arm = true;
        } else if (a == "--drain-ms" && (v = next())) {
            o.drain_ms = std::atoi(v);
        } else if (a == "--exit-wait-ms" && (v = next())) {
            o.exit_wait_ms = std::atoi(v);
        } else if (a == "--report-only") {
            o.report_only = true;
        } else {
            return false;
        }
    }
    return !o.model.empty() && o.runs >= 1 && o.max_delay >= 0 && o.finalizer_ms >= 0 &&
           parse_compute_units(o.units).has_value();
}

}  // namespace

int main(int argc, char ** argv) {
    ExitTestOptions o;
    bool is_child = false;
    int delay = 0;
    if (!parse(argc, argv, o, is_child, delay)) {
        std::fprintf(stderr,
                     "usage: %s --model <x.mlmodelc> [--runs N] [--max-delay-ms M] [--units U] [--finalizer-ms F]\n"
                     "       [--no-scribble] [--engine-thread main|worker] [--arm] [--drain-ms D]\n"
                     "       [--exit-wait-ms W] [--report-only]\n",
                     argv[0]);
        return 2;
    }
    if (is_child) return child(o, delay);

    std::printf("engine on the %s thread%s%s, MallocScribble %s, finalizer %d ms, exit wait %s\n",
                o.worker ? "worker" : "main", o.arm ? ", main armed" : "",
                o.drain_ms >= 0 ? (", drain " + std::to_string(o.drain_ms) + " ms").c_str() : "",
                o.scribble ? "on" : "off", o.finalizer_ms,
                o.exit_wait_ms >= 0 ? (std::to_string(o.exit_wait_ms) + " ms").c_str() : "default");
    // Control: the model must load to ready at all on this host.
    const RunResult ctl = run_once(o, argv[0], -1);
    const bool control_ok = ctl.exited && WIFEXITED(ctl.status) && WEXITSTATUS(ctl.status) == 0 &&
                            ctl.state == "ready" && ctl.native_load == "ok";
    std::printf("%s control load: state %s, %s\n", control_ok ? "ok  " : "FAIL",
                ctl.state.empty() ? "?" : ctl.state.c_str(), describe(ctl).c_str());
    // Where the exit wait applies, require native success and intact teardown order.
    const bool expect_wait = (!o.worker || o.arm || o.drain_ms > 0) && o.exit_wait_ms != 0;
    int failures = control_ok ? 0 : 1, signals = 0, hangs = 0, in_flight = 0, torn = 0;
    int native_failures = 0;
    double max_exit_ms = 0;
    for (int i = 0; i < o.runs; ++i) {
        const int d = o.runs == 1 ? 0 : o.max_delay * i / (o.runs - 1);
        const RunResult r = run_once(o, argv[0], d);
        const bool signalled = r.exited && WIFSIGNALED(r.status);
        const bool hung = !r.exited || r.exit_ms > kHangMs;
        const bool loading = r.state == "loading" || r.state == "warming";
        const bool not_waited = expect_wait && loading && r.load != "intact";
        const bool native_failed = ((expect_wait && loading) || r.state == "ready") && r.native_load != "ok";
        const bool clean = r.exited && WIFEXITED(r.status) && WEXITSTATUS(r.status) == 0 && !hung &&
                           (loading || r.state == "ready") && !not_waited && !native_failed;
        native_failures += native_failed ? 1 : 0;
        if (loading) ++in_flight;
        torn += not_waited ? 1 : 0;
        if (r.exit_ms > max_exit_ms) max_exit_ms = r.exit_ms;
        signals += signalled ? 1 : 0;
        hangs += hung ? 1 : 0;
        std::printf("%s run %d delay %d ms: state %s, %s, exit took %.1f ms, load %s, native load %s%s\n",
                    clean ? "ok  " : "FAIL", i,
                    d, r.state.empty() ? "?" : r.state.c_str(), describe(r).c_str(), r.exit_ms,
                    r.load.empty() ? "not returned" : r.load.c_str(),
                    r.native_load.empty() ? "not returned" : r.native_load.c_str(), hung ? " (hang)" : "");
        std::fflush(stdout);
        if (!clean) ++failures;
    }
    const bool vacuous = in_flight * 2 < o.runs;
    if (vacuous) {
        std::printf("FAIL only %d of %d runs exited with the load in flight: the model loads too fast for this test\n",
                    in_flight, o.runs);
    }
    const bool pass = failures == 0 && !vacuous;
    std::printf("%s: %d runs, %d exited with the load in flight, %d failures (%d signals, %d hangs over %d ms, "
                "%d in-flight loads not waited for, %d missing or failed native loads), slowest exit %.1f ms\n",
                o.report_only ? "REPORT" : (pass ? "PASS" : "FAIL"), o.runs, in_flight, failures, signals, hangs,
                kHangMs, torn, native_failures, max_exit_ms);
    return o.report_only || pass ? 0 : 1;
}
