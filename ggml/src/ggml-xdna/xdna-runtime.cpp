#include "xdna-runtime.h"

#include "ggml-impl.h"
#include "xdna-profile.h"

#include <xrt/experimental/xrt_kernel.h>
#include <xrt/experimental/xrt_xclbin.h>
#include <xrt/xrt_kernel.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// NPU kernel ABI: arg 0 is the opcode, where 3 = RUN the instruction stream.
static constexpr int XAIE_NPU_OPCODE_RUN = 3;

// Loaded xclbins: uuid -> shared hw_context. All variants of one xclbin share
// a context; kernels hold a shared_ptr so contexts outlive every kernel that
// references them.
static std::mutex                                            g_ctx_mutex;
static std::map<xrt::uuid, std::shared_ptr<xrt::hw_context>> g_ctxs;

// --- device -----------------------------------------------------------

xdna_device * xdna_device_open(void) {
    xdna_device * dev = new xdna_device;
    try {
        dev->device      = xrt::device(0);
        dev->name        = dev->device.get_info<xrt::info::device::name>();
        dev->description = dev->device.get_info<xrt::info::device::bdf>();
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: failed to open NPU device: %s\n", "xdna-runtime", e.what());
        delete dev;
        return nullptr;
    }
    return dev;
}

// --- kernel -------------------------------------------------------------

std::vector<fs::path> xdna_kernel_search_dirs(void) {
    std::vector<fs::path> dirs;
#ifdef GGML_BACKEND_DIR
    dirs.emplace_back(GGML_BACKEND_DIR);
#endif
#ifdef __linux__
    std::error_code ec;
    fs::path exe = fs::read_symlink("/proc/self/exe", ec);
    if (!ec) {
        dirs.push_back(exe.parent_path());
    }
#endif
    dirs.emplace_back(fs::current_path());
    return dirs;
}

// Load an xclbin into a kernel handle without an instruction stream. The
// stream is bound later with xdna_kernel_bind_insts. Returns nullptr on
// failure.
xdna_kernel * xdna_kernel_load_hw(xdna_device * dev, const char * xclbin_path) {
    if (!dev) {
        GGML_LOG_ERROR("%s: kernel load: no device\n", "xdna-runtime");
        return nullptr;
    }

    try {
        const std::string xclbin_path_str(xclbin_path);
        const xrt::xclbin xclbin{xclbin_path_str};
        dev->device.register_xclbin(xclbin);
        const xrt::uuid xclbin_uuid = xclbin.get_uuid();

        // Reuse the hw_context for this xclbin across all kernel variants.
        std::shared_ptr<xrt::hw_context> context;
        {
            std::lock_guard<std::mutex> lock(g_ctx_mutex);
            auto it = g_ctxs.find(xclbin_uuid);
            if (it != g_ctxs.end()) {
                context = it->second;
            } else {
                context = std::make_shared<xrt::hw_context>(dev->device, xclbin_uuid);
                g_ctxs.emplace(xclbin_uuid, context);
            }
        }

        const auto kernels = xclbin.get_kernels();
        if (kernels.empty()) {
            GGML_LOG_ERROR("%s: no kernel found in %s\n", "xdna-runtime", xclbin_path);
            return nullptr;
        }

        xdna_kernel * kern = new xdna_kernel;
        kern->context      = context;
        kern->kernel       = xrt::kernel(*context, kernels[0].get_name());
        kern->xclbin_name  = xclbin_path_str;
        return kern;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: failed to load kernel %s: %s\n", "xdna-runtime", xclbin_path, e.what());
        return nullptr;
    }
}

// Bind an in-memory instruction stream to a loaded kernel. The insts BO is
// created on group 1 (the instruction buffer). `dev` must be the same handle
// used for the data buffers (XRT treats copies of xrt::device as distinct;
// mixing them can wedge submissions). Returns false on failure.
bool xdna_kernel_bind_insts(xdna_device * dev, xdna_kernel * kern, const uint32_t * insts, size_t n_words) {
    if (!dev || !kern || !insts || n_words == 0) {
        GGML_LOG_ERROR("%s: bind insts: null device/kernel/insts or empty stream\n", "xdna-runtime");
        return false;
    }
    try {
        kern->insts_bytes = (int64_t)(n_words * sizeof(uint32_t));
        kern->insts_bo    = xrt::bo(dev->device, (size_t) kern->insts_bytes,
                                    xrt::bo::flags::cacheable, kern->kernel.group_id(1));
        std::memcpy(kern->insts_bo.map(), insts, (size_t) kern->insts_bytes);
        kern->insts_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        return true;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: failed to bind instruction stream: %s\n", "xdna-runtime", e.what());
        return false;
    }
}

void xdna_kernel_free(xdna_kernel * kern) {
    delete kern;
}

// --- buffer -------------------------------------------------------------

xdna_buffer * xdna_buffer_alloc(xdna_device * dev, size_t bytes) {
    if (!dev) {
        GGML_LOG_ERROR("%s: buffer alloc: no device\n", "xdna-runtime");
        return nullptr;
    }
    xdna_buffer * buf = new xdna_buffer;
    try {
        buf->bo    = xrt::bo(dev->device, bytes, xrt::bo::flags::host_only, 0);
        buf->bytes = bytes;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: failed to allocate %zu-byte BO: %s\n", "xdna-runtime", bytes, e.what());
        delete buf;
        return nullptr;
    }
    return buf;
}

xdna_buffer * xdna_buffer_sub(xdna_buffer * parent, size_t offset, size_t bytes) {
    if (!parent || offset + bytes > parent->bytes) {
        GGML_LOG_ERROR("%s: buffer view %zu+%zu outside a %zu-byte BO\n",
                       "xdna-runtime", offset, bytes, parent ? parent->bytes : 0);
        return nullptr;
    }
    xdna_buffer * buf = new xdna_buffer;
    try {
        buf->bo    = xrt::bo(parent->bo, bytes, offset);
        buf->bytes = bytes;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: failed to view %zu+%zu of a BO: %s\n",
                       "xdna-runtime", offset, bytes, e.what());
        delete buf;
        return nullptr;
    }
    return buf;
}

void xdna_buffer_free(xdna_buffer * buf) {
    delete buf;
}

void xdna_buffer_sync_to_device(xdna_buffer * buf) {
    if (buf) {
        buf->bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
}

void xdna_buffer_sync_to_device_range(xdna_buffer * buf, size_t bytes, size_t offset) {
    if (buf && bytes) {
        buf->bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, bytes, offset);
    }
}

void xdna_buffer_sync_from_device(xdna_buffer * buf) {
    if (buf) {
        buf->bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    }
}

// --- execution ------------------------------------------------------------

// Configure a run for `kern` with `n_args` host buffers (ABI: 0=opcode,
// 1=instruction BO, 2=ninstr, 3..=host buffers).
static xrt::run make_run(xdna_kernel * kern, xdna_buffer ** args, size_t n_args) {
    xrt::run run(kern->kernel);
    run.set_arg(0, XAIE_NPU_OPCODE_RUN);
    run.set_arg(1, kern->insts_bo);
    run.set_arg(2, kern->insts_bytes);
    for (size_t i = 0; i < n_args; i++) {
        run.set_arg((int) (3 + i), args[i]->bo);
    }
    return run;
}

xrt::run xdna_kernel_run_make(xdna_kernel * kern, xdna_buffer ** args, size_t n_args) {
    if (!kern || !args || n_args == 0) {
        GGML_LOG_ERROR("%s: run make: null kernel/args or no buffers\n", "xdna-runtime");
        return xrt::run{};
    }
    try {
        return make_run(kern, args, n_args);
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: kernel run make exception: %s\n", "xdna-runtime", e.what());
        return xrt::run{};
    }
}

static void xdna_design_prof_restart(const char * tag);  // with the profiler below

bool xdna_run_restart(xrt::run & run, const char * tag) {
    try {
        // Counted separately: a prepared run costs the same submission as a
        // fresh one, and the decode path is almost all of these.
        xdna_design_prof_restart(tag);
        run.start();
        return true;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: kernel run restart exception: %s\n", "xdna-runtime", e.what());
        return false;
    }
}

// See xdna-profile.h. Kept here rather than in a header so every runner files
// into one table.
struct xdna_runner_prof {
    bool on = false;
    std::map<std::string, std::pair<int64_t, long>> t;   // "runner phase" -> (us, calls)
    ~xdna_runner_prof() {
        if (!on || t.empty()) {
            return;
        }
        int64_t total = 0;
        for (const auto & kv : t) {
            total += kv.second.first;
        }
        fprintf(stderr, "xdna-runner-prof: %.0f ms inside the NPU runners\n",
                (double) total / 1000.0);
        for (const auto & kv : t) {
            fprintf(stderr, "xdna-runner-prof:   %-28s %8.1f ms  x%-6ld %7.1f us each\n",
                    kv.first.c_str(), (double) kv.second.first / 1000.0,
                    kv.second.second,
                    (double) kv.second.first / (double) kv.second.second);
        }
    }
};

static xdna_runner_prof & xdna_runner_prof_get(void) {
    static xdna_runner_prof p = []() {
        xdna_runner_prof q;
        q.on = getenv("GGML_XDNA_RUNNER_PROF") != nullptr;
        return q;
    }();
    return p;
}

bool xdna_runner_prof_on(void) { return xdna_runner_prof_get().on; }

void xdna_runner_prof_add(const char * runner, const char * phase, int64_t us) {
    auto & e = xdna_runner_prof_get().t[std::string(runner) + " " + phase];
    e.first  += us;
    e.second += 1;
}

// GGML_XDNA_DESIGN_PROF=1: how many dispatches each xclbin took and how often
// the array had to switch between them. A switch reconfigures every column the
// design occupies (~0.37 ms for one, ~2.8 ms for eight), so in a prompt that
// interleaves three designs per recurrent layer the switches can cost more
// than the work.
struct xdna_design_prof {
    bool on = false;
    std::map<std::string, long> runs;
    std::map<std::string, long> switches_in;
    std::string last;
    long total = 0, switches = 0, restarts = 0;
    std::map<std::string, long> restart_by;
    // Only dispatches whose wait is the next thing that happens are timed, so
    // batched submissions (the GEMM path) are left out of the averages.
    std::map<std::string, double> us_first, us_rest;
    std::map<std::string, long>   n_first, n_rest;
    int    pending = 0;
    bool   timed   = false;
    bool   first   = false;
    std::string    t_name;
    std::chrono::steady_clock::time_point t0;
    ~xdna_design_prof() {
        if (!on || total == 0) {
            return;
        }
        fprintf(stderr, "xdna-design-prof: %ld dispatches (%ld first starts, "
                "%ld restarts of a prepared run), %ld switches\n",
                total + restarts, total, restarts, switches);
        for (const auto & kv : restart_by) {
            fprintf(stderr, "xdna-design-prof:   restart %-20s x%ld\n",
                    kv.first.c_str(), kv.second);
        }
        for (const auto & kv : runs) {
            const std::string & k = kv.first;
            fprintf(stderr, "xdna-design-prof:   %-44s runs=%-6ld switched-in=%-4ld"
                    " first %.0fus x%ld, rest %.0fus x%ld\n",
                    k.c_str(), kv.second, switches_in[k],
                    n_first[k] ? us_first[k] / n_first[k] : 0.0, n_first[k],
                    n_rest[k]  ? us_rest[k]  / n_rest[k]  : 0.0, n_rest[k]);
        }
    }
};

static xdna_design_prof & xdna_design_prof_get(void);

static void xdna_design_prof_restart(const char * tag) {
    xdna_design_prof & dp = xdna_design_prof_get();
    if (dp.on) {
        dp.restarts++;
        dp.restart_by[tag ? tag : "?"]++;
    }
}

static xdna_design_prof & xdna_design_prof_get(void) {
    static xdna_design_prof p = []() {
        xdna_design_prof q;
        const char * v = getenv("GGML_XDNA_DESIGN_PROF");
        q.on = v != nullptr && atoi(v) != 0;
        return q;
    }();
    return p;
}

xrt::run xdna_kernel_run_start(xdna_kernel * kern, xdna_buffer ** args, size_t n_args) {
    if (!kern || !args || n_args == 0) {
        GGML_LOG_ERROR("%s: run start: null kernel/args or no buffers\n", "xdna-runtime");
        return xrt::run{};
    }
    try {
        // GGML_XDNA_RUN_TRACE: one line per dispatch, in launch order -
        // which dispatches precede the first post dispatch is what the
        // run trace answers.
        if (getenv("GGML_XDNA_RUN_TRACE")) {
            static int seq_no = 0;
            fprintf(stderr, "xdna-run: #%d start xclbin=%s n_args=%zu\n",
                    seq_no++, kern->xclbin_name.c_str(), n_args);
        }
        xdna_design_prof & dp = xdna_design_prof_get();
        if (dp.on) {
            dp.total++;
            dp.runs[kern->xclbin_name]++;
            const bool sw = dp.last != kern->xclbin_name;
            if (sw) {
                dp.switches++;
                dp.switches_in[kern->xclbin_name]++;
                dp.last = kern->xclbin_name;
            }
            dp.timed   = dp.pending == 0;
            dp.first   = sw;
            dp.t_name  = kern->xclbin_name;
            dp.t0      = std::chrono::steady_clock::now();
            dp.pending++;
        }
        xrt::run run = make_run(kern, args, n_args);
        run.start();
        return run;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: kernel run start exception: %s\n", "xdna-runtime", e.what());
        return xrt::run{};
    }
}

// GGML_XDNA_SPIN=0 turns off the spin and blocks in XRT instead.
static bool xdna_spin_wait(void) {
    static const bool on = []() {
        const char * v = getenv("GGML_XDNA_SPIN");
        return v == nullptr || atoi(v) != 0;
    }();
    return on;
}

// GGML_XDNA_DISPATCH_COUNT=1 reports how many times the array was launched.
// It is the number that decides how much of the decode is the per-phase floor
// rather than the work.
struct xdna_dispatch_counter {
    uint64_t n = 0;
    ~xdna_dispatch_counter() {
        if (n && getenv("GGML_XDNA_DISPATCH_COUNT")) {
            fprintf(stderr, "xdna-runtime: %llu dispatches\n",
                    (unsigned long long) n);
        }
    }
};

static xdna_dispatch_counter & xdna_dispatches(void) {
    static xdna_dispatch_counter c;
    return c;
}

// GGML_XDNA_CTX_PROBE=<xclbin> times the same dispatch run on one context
// against the same dispatch alternating between two. The second artifact is a
// uuid-distinct copy of the first (xclbinutil --force rewrites the uuid and
// nothing else), so the hardware is identical and the only difference is that
// consecutive dispatches come from different contexts. The backend is built
// around a claim that this costs 1.4-2.6 ms; FastFlowLM ships eight xclbins
// and holds 38.8 t/s, which the claim does not allow.
static void xdna_ctx_probe(xdna_device * dev, xdna_kernel * kern,
                           xdna_buffer ** args, size_t n_args) {
    const char * path = getenv("GGML_XDNA_CTX_PROBE");
    if (!path || !kern) {
        return;
    }
    using clock = std::chrono::steady_clock;
    // The copy is loaded on the same device and bound with this kernel's own
    // stream, so the two dispatch identically.
    xdna_kernel * k2 = xdna_kernel_load_hw(dev, path);
    if (!k2) {
        fprintf(stderr, "xdna-ctx-probe: could not load %s\n", path);
        return;
    }
    const uint32_t * w = (const uint32_t *) kern->insts_bo.map();
    if (!xdna_kernel_bind_insts(dev, k2, w, (size_t) kern->insts_bytes / 4)) {
        fprintf(stderr, "xdna-ctx-probe: could not bind the stream to the copy\n");
        return;
    }
    const int reps = 32;
    try {
        xrt::run r1 = make_run(kern, args, n_args);
        xrt::run r2 = make_run(k2, args, n_args);
        for (int i = 0; i < 4; i++) { r1.start(); r1.wait(); r2.start(); r2.wait(); }
        const auto t0 = clock::now();
        for (int i = 0; i < reps; i++) { r1.start(); r1.wait(); }
        const double one_us =
            std::chrono::duration<double, std::micro>(clock::now() - t0).count() / reps;
        const auto t1 = clock::now();
        for (int i = 0; i < reps / 2; i++) {
            r1.start(); r1.wait();
            r2.start(); r2.wait();
        }
        const double alt_us =
            std::chrono::duration<double, std::micro>(clock::now() - t1).count() / reps;
        fprintf(stderr,
                "xdna-ctx-probe: one context %.1f us/dispatch, alternating two "
                "%.1f us (+%.1f us a switch)\n",
                one_us, alt_us, alt_us - one_us);
    } catch (const std::exception & ex) {
        fprintf(stderr, "xdna-ctx-probe: %s\n", ex.what());
    }
}

void xdna_runlist_probe(xdna_device * dev, xdna_kernel * kern,
                        xdna_buffer ** args, size_t n_args) {
    xdna_ctx_probe(dev, kern, args, n_args);
    const char * e = getenv("GGML_XDNA_RUNLIST_PROBE");
    const int reps = e ? atoi(e) : 0;
    if (reps <= 0 || !kern || !args) {
        return;
    }
    using clock = std::chrono::steady_clock;
    try {
        // Two independent sets: a run that has already been executed cannot
        // be handed to a runlist.
        std::vector<xrt::run> runs, listed;
        runs.reserve((size_t) reps);
        listed.reserve((size_t) reps);
        for (int i = 0; i < reps; i++) {
            runs.push_back(make_run(kern, args, n_args));
            listed.push_back(make_run(kern, args, n_args));
        }
        // one at a time, each waited before the next is submitted
        const auto t0 = clock::now();
        for (int i = 0; i < reps; i++) {
            runs[(size_t) i].start();
            runs[(size_t) i].wait();
        }
        const double seq_us =
            std::chrono::duration<double, std::micro>(clock::now() - t0).count();
        fprintf(stderr, "xdna-runlist-probe: one at a time %.1f us each\n",
                seq_us / reps);
        // the same work as one list, executed in order on the device
        xrt::runlist rl(*kern->context);
        for (int i = 0; i < reps; i++) {
            rl.add(listed[(size_t) i]);
        }
        const auto t1 = clock::now();
        try {
            rl.execute();
        } catch (const std::exception & ex) {
            fprintf(stderr, "xdna-runlist-probe: execute: %s\n", ex.what());
            return;
        }
        try {
            rl.wait();
        } catch (const std::exception & ex) {
            fprintf(stderr, "xdna-runlist-probe: wait: %s (state of run 0: %d)\n",
                    ex.what(), (int) listed[0].state());
            return;
        }
        const double rl_us =
            std::chrono::duration<double, std::micro>(clock::now() - t1).count();
        fprintf(stderr,
                "xdna-runlist-probe: %s x%d: one at a time %.1f us each, "
                "as a runlist %.1f us each (%.2fx)\n",
                kern->xclbin_name.c_str(), reps, seq_us / reps, rl_us / reps,
                rl_us > 0 ? seq_us / rl_us : 0.0);
        // A layer's dispatches are different instruction streams, not the same
        // one repeated, so the question that decides whether a runlist can
        // carry one is whether it takes runs from more than one kernel of the
        // same context. Two kernel handles on the same xclbin share the
        // context; give them a stream each and put both in one list.
        xdna_kernel * k2 = xdna_kernel_load_hw(dev, kern->xclbin_name.c_str());
        if (k2 && xdna_kernel_bind_insts(dev, k2,
                                         (const uint32_t *) kern->insts_bo.map(),
                                         (size_t) kern->insts_bytes / 4)) {
            std::vector<xrt::run> mixed;
            for (int i = 0; i < reps; i++) {
                mixed.push_back(make_run(i % 2 ? k2 : kern, args, n_args));
            }
            xrt::runlist rl2(*kern->context);
            for (auto & r : mixed) {
                rl2.add(r);
            }
            const auto t2 = clock::now();
            try {
                rl2.execute();
                rl2.wait();
                fprintf(stderr,
                        "xdna-runlist-probe: two kernels of one context in one "
                        "list: %.1f us each\n",
                        std::chrono::duration<double, std::micro>(
                            clock::now() - t2).count() / reps);
            } catch (const std::exception & ex) {
                fprintf(stderr, "xdna-runlist-probe: mixed list: %s\n", ex.what());
            }
        }
    } catch (const std::exception & ex) {
        fprintf(stderr, "xdna-runlist-probe: %s\n", ex.what());
    }
}

bool xdna_run_wait(xrt::run & run) {
    xdna_dispatches().n++;
    xdna_design_prof & dp = xdna_design_prof_get();
    struct dp_time {
        xdna_design_prof & p;
        ~dp_time() {
            if (!p.on || p.pending == 0) {
                return;
            }
            p.pending--;
            if (!p.timed || p.pending != 0) {
                return;   // batched: the wait does not bound this dispatch
            }
            const double us = std::chrono::duration<double, std::micro>(
                                  std::chrono::steady_clock::now() - p.t0).count();
            (p.first ? p.us_first : p.us_rest)[p.t_name] += us;
            (p.first ? p.n_first  : p.n_rest)[p.t_name]++;
        }
    } dp_guard{ dp };
    try {
        // Poll before blocking. A dispatch shorter than XRT's completion path
        // measures as that path and not as its own work: the decode's short
        // ones sit on a floor of about 125 us where their weights need 60.
        // The decode is sequential, so the thread that would block here has
        // nothing else to do with the time.
        if (xdna_spin_wait()) {
            for (int i = 0; i < (1 << 22); i++) {
                const ert_cmd_state s = run.state();
                if (s == ERT_CMD_STATE_COMPLETED) {
                    return true;
                }
                if (s == ERT_CMD_STATE_ERROR || s == ERT_CMD_STATE_ABORT ||
                    s == ERT_CMD_STATE_TIMEOUT) {
                    GGML_LOG_ERROR("%s: kernel spin state %d\n", "xdna-runtime", (int) s);
                    return false;
                }
            }
        }
        const ert_cmd_state st = run.wait();
        if (st != ERT_CMD_STATE_COMPLETED) {
            GGML_LOG_ERROR("%s: kernel wait state %d\n", "xdna-runtime", (int) st);
            return false;
        }
        return true;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: kernel wait exception: %s\n", "xdna-runtime", e.what());
        return false;
    }
}


