// Facades over the scheduler (runtime.h): Manager, Slot, Group, ResidentPool,
// SharedModel and Handle; route planning; the per-image instance and
// the exit protocol (exit flag, exit guard, bounded wait for running jobs).

#include "qvac_coreml/manager.h"

#include "device.h"
#include "hints.h"
#include "runtime.h"
#include "test_hooks.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include <pthread.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif
#if defined(__linux__)
#include <sys/syscall.h>
#endif

namespace QVAC_COREML_NS {

namespace detail {

namespace {
// Trivially destructible on purpose: read by worker threads during exit().
std::atomic<bool> g_exiting{false};
std::atomic<bool> g_atexit_registered{false};
std::atomic<int> g_jobs{0};  // jobs a worker is running (enter_job/leave_job)
std::atomic<bool> g_guard_armed{false};
std::atomic<bool> g_explicit_arm{false};  // arm_exit_guard() (the public hook) armed it
std::atomic<bool> g_automatic_guard{true};
// The atexit fallback ran: process finalization has begun, and the frameworks'
// lazily registered destructors may already have run. Nothing waits after it.
std::atomic<bool> g_finalizing{false};
std::atomic<bool> g_refusal_logged{false};
// The one exit deadline of this image (steady clock, ns); 0 until the first
// wait with a positive bound sets it. Every wait ends at the earlier of its
// own deadline and this one.
std::atomic<int64_t> g_deadline_ns{0};
thread_local bool t_in_job = false;  // this thread is a worker running a job

// Cold loads measured on an M2 (one job; the admission gate runs one at a
// time per image): Supertonic vocoder ~1 s, Pocket 6-8 s, ACE-Step VAE 8.7 s,
// Parler DAC 10.6 s. A process that ends while one runs waits for the rest of
// it rather than crash, up to this bound. iOS gives a terminating app a few
// seconds in all, so the bound there is short.
#if defined(__APPLE__) && TARGET_OS_IPHONE && !TARGET_OS_MACCATALYST
constexpr std::chrono::milliseconds kExitWait{2000};
#else
constexpr std::chrono::milliseconds kExitWait{15000};
#endif
// Larger bounds (a JS host passing Number.MAX_SAFE_INTEGER) are clamped, so
// the nanosecond arithmetic cannot overflow.
constexpr std::chrono::milliseconds kMaxExitWait{3600000};

// Registered when the first manager is created, so it runs after every
// destructor registered later, including those the frameworks register during
// a compile. Waiting here was measured to do harm (an engine made off the main
// thread: crashes on the destroyed tables, and jobs spinning on them until the
// bound), so it only stops new work.
void on_process_exit() {
    g_finalizing.store(true);
    g_exiting.store(true);
}

// A child of fork() has no worker thread: nothing of the parent's runs there.
void on_fork_child() { g_jobs.store(0); }

int64_t steady_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

struct ExitGuard {
    ~ExitGuard() {
        // set_automatic_exit_guard(false) is honoured here too, whenever it
        // was called; a guard the host armed itself still waits.
        if (!g_automatic_guard.load() && !g_explicit_arm.load()) return;
        wait_for_jobs_at_exit(exit_wait());
    }
};
}  // namespace

bool on_main_thread() {
#if defined(__APPLE__)
    return pthread_main_np() == 1;
#elif defined(__linux__)
    return static_cast<pid_t>(syscall(SYS_gettid)) == getpid();
#else
    return false;
#endif
}

bool exiting() { return g_exiting.load(std::memory_order_relaxed); }

void register_atexit_once() {
    if (!g_atexit_registered.exchange(true)) std::atexit(&on_process_exit);
}

void register_fork_handler_once() {
    // pthread_atfork() cannot be undone: only a pinned image registers it.
    // pthread_once: a concurrent first caller returns only after it is
    // registered, so no fork can copy a counted job without the handler.
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, [] { (void) pthread_atfork(nullptr, nullptr, &on_fork_child); });
}

bool finalizing() { return g_finalizing.load(); }

void note_explicit_arm() { g_explicit_arm.store(true); }

bool automatic_exit_guard() { return g_automatic_guard.load(std::memory_order_relaxed); }

void set_automatic_exit_guard_flag(bool on) { g_automatic_guard.store(on, std::memory_order_relaxed); }

bool arm_exit_guard() {
    if (!on_main_thread() || g_finalizing.load()) return false;
    if (g_guard_armed.load(std::memory_order_relaxed)) return true;
    // The main thread's thread_local destructors are the first thing exit()
    // runs, before any static destructor or atexit handler (C++
    // [basic.start.term]; Darwin's exit() calls _tlv_exit() first, glibc's
    // runs __call_tls_dtors first). On Darwin they also run when the main
    // thread ends with pthread_exit() and the process goes on
    // (dispatch_main() does that): exit has then begun for this runtime, and
    // later requests fail with kExitingReason (set_automatic_exit_guard(false)
    // avoids this). glibc does not run them there.
    thread_local ExitGuard guard;
    (void) guard;
    g_guard_armed.store(true, std::memory_order_relaxed);
    return true;
}

bool enter_job() {
    g_jobs.fetch_add(1);
    if (g_exiting.load()) {
        g_jobs.fetch_sub(1);
        return false;
    }
    t_in_job = true;
    return true;
}

void leave_job() {
    t_in_job = false;
    g_jobs.fetch_sub(1);
}

std::chrono::milliseconds exit_wait() {
#ifdef QVAC_COREML_TEST_HOOKS
    const int64_t ms = test_hooks().exit_wait_ms.load();
    if (ms >= 0) return std::chrono::milliseconds(ms);
#endif
    return kExitWait;
}

bool wait_for_jobs_at_exit(std::chrono::milliseconds bound) {
    g_exiting.store(true);
    // A job that calls this does not wait for itself, and is still running:
    // the answer is then false.
    const int own = t_in_job ? 1 : 0;
    auto done = [own] { return own == 0 && g_jobs.load() == 0; };
    // A zero bound only closes admission. Once finalization began nothing
    // waits: the frameworks' destructors may already have run.
    if (bound.count() <= 0 || g_finalizing.load()) return done();
    const int64_t mine = steady_ns() + std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::min(bound, kMaxExitWait)).count();
    int64_t unset = 0;
    g_deadline_ns.compare_exchange_strong(unset, mine);
    const int64_t deadline = std::min(mine, g_deadline_ns.load());
    // Polls: no mutex or condition variable that a static destructor could
    // already have destroyed.
    while (g_jobs.load() > own) {
        if (steady_ns() >= deadline) return false;
        ::usleep(1000);
    }
    return done();
}

void note_refused_request() {
    if (!g_refusal_logged.exchange(true)) std::fprintf(stderr, "qvac-coreml: %s\n", kExitingReason);
}

}  // namespace detail

// ---- small types ----------------------------------------------------------

const char * to_string(Priority p) {
    switch (p) {
        case Priority::prewarm: return "prewarm";
        case Priority::minimal: return "minimal";
        case Priority::current: return "current";
        case Priority::now: return "now";
    }
    return "?";
}

const char * to_string(GroupKind k) { return k == GroupKind::construction ? "construction" : "prewarm"; }

Error Error::transient(std::string msg, std::string domain, int64_t code) {
    return Error{ErrorClass::transient, std::move(domain), code, std::move(msg)};
}

Error Error::deterministic(std::string msg, std::string domain, int64_t code) {
    return Error{ErrorClass::deterministic, std::move(domain), code, std::move(msg)};
}

SlotPolicy SlotPolicy::from(const Resolved & r) {
    SlotPolicy p;
    p.wait = r.acquire_wait() == Wait::call ? Wait::call : Wait::none;
    p.strict = r.strict();
    p.strict_wait_ms = r.strict_wait_ms;
    p.cache_dir = r.cache_dir;
    return p;
}

// ---- SharedModel / Handle -------------------------------------------------

SharedModel::SharedModel(std::shared_ptr<void> impl, LoadRequest req, std::string key, int lease_fd,
                         std::weak_ptr<detail::Runtime> rt)
    : impl_(std::move(impl)), req_(std::move(req)), key_(std::move(key)), lease_fd_(lease_fd), rt_(std::move(rt)) {}

SharedModel::~SharedModel() {
    impl_.reset();  // the engine's MLModel wrapper goes first
    if (lease_fd_ >= 0) ::close(lease_fd_);
    if (std::shared_ptr<detail::Runtime> rt = rt_.lock()) rt->on_model_released(*this);
}

void SharedModel::retire(const std::string & reason) {
    if (std::shared_ptr<detail::Runtime> rt = rt_.lock()) {
        rt->retire_model(*this, reason);
    } else {
        retired_ = true;
    }
}

Handle::Handle(std::shared_ptr<SharedModel> m) : m_(std::move(m)) {}

Handle::Handle(std::shared_ptr<SharedModel> m, std::shared_ptr<detail::PoolRef> ref)
    : m_(std::move(m)), ref_(std::move(ref)) {
    if (m_ && ref_) ref_->holds.fetch_add(1);
}

Handle & Handle::operator=(Handle && o) noexcept {
    if (this != &o) {
        reset();
        m_ = std::move(o.m_);
        ref_ = std::move(o.ref_);
    }
    return *this;
}

void Handle::reset() {
    std::shared_ptr<SharedModel> m = std::move(m_);
    std::shared_ptr<detail::PoolRef> ref = std::move(ref_);
    m_.reset();
    ref_.reset();
    if (!m || !ref) return;
    std::shared_ptr<detail::Runtime> rt = m->rt_.lock();
    // The reference goes before the hold and before the scheduler is woken, so
    // a worker that sees the release never finds this handle still alive.
    m.reset();
    if (ref->holds.fetch_sub(1) == 1 && rt) rt->notify();  // the pool may evict it now
}

// ---- Slot -----------------------------------------------------------------

Slot::~Slot() {
    if (core_) core_->rt->release_slot(core_);
}

void Slot::request(Priority p) { core_->rt->request(core_, p, false); }

Acquired Slot::acquire(Demand d, std::optional<Clock::time_point> deadline, const std::atomic<bool> * cancel) {
    return core_->rt->acquire(core_, d, deadline, cancel);
}

void Slot::evict() { core_->rt->evict(core_); }

void Slot::retire(const std::string & reason) { core_->rt->retire_slot(core_, reason); }

SlotState Slot::state() const { return core_->rt->state(core_); }

SlotStatus Slot::status() const { return core_->rt->status(core_); }

std::optional<HintView> Slot::hint() const { return status().hint; }

std::string Slot::hint_path() const { return core_->rt->hint_path(core_); }

int64_t Slot::hint_mtime_ns() const {
    std::optional<detail::HintRecord> h = detail::read_hint(hint_path());
    return h ? h->mtime_ns : 0;
}

const SlotSpec & Slot::spec() const { return core_->spec; }

// ---- ResidentPool ---------------------------------------------------------

size_t ResidentPool::cap() const { return core_->cap; }

void ResidentPool::set_predicted(const std::vector<std::shared_ptr<Slot>> & slots) {
    std::vector<std::shared_ptr<detail::SlotCore>> cores;
    for (const auto & s : slots) {
        if (s) cores.push_back(s->core_);
    }
    rt_->pool_set_predicted(*core_, cores);
}

void ResidentPool::clear_predicted() { rt_->pool_set_predicted(*core_, {}); }

size_t ResidentPool::live() const { return rt_->pool_live(*core_); }

// ---- Group ----------------------------------------------------------------

PrewarmResult Group::collect(bool & finished, detail::Graveyard & gy) const {
    PrewarmResult r;
    r.group = to_string(kind_);
    r.skipped_route_none = skipped_;
    r.nothing_to_prepare = entries_.empty();
    finished = true;
    bool all_ready = true;
    const Clock::time_point now = Clock::now();
    for (Entry & e : entries_) {
        std::shared_ptr<detail::SlotCore> c = e.core.lock();
        gy.keep(c);
        if (c && !e.reached_ready && c->ready_events > e.baseline) {
            // It reached ready since the group was issued: record that result,
            // even if the engine evicts the slot afterwards.
            e.reached_ready = true;
            e.last = c->ready_snapshot;
        }
        if (!e.reached_ready) {
            all_ready = false;
            if (c) {
                rt_->refuse_queued_after_exit_locked(c, gy);
                e.last = rt_->status_locked(*c, now);
                const SlotState st = e.last.state;
                const bool done = is_terminal(st) || st == SlotState::deferred_memory || st == SlotState::idle ||
                                  rt_->cap_blocked_now_locked(*c, gy);
                if (!done) finished = false;
            }
        }
        r.slots.push_back(e.last);
    }
    r.success = finished && all_ready;
    return r;
}

bool Group::finished() const {
    detail::Graveyard gy;
    std::lock_guard<std::mutex> lk(rt_->mu);
    bool fin = false;
    collect(fin, gy);
    return fin;
}

PrewarmResult Group::result() const {
    detail::Graveyard gy;
    std::lock_guard<std::mutex> lk(rt_->mu);
    bool fin = false;
    return collect(fin, gy);
}

PrewarmResult Group::wait(std::optional<Clock::duration> timeout, const std::atomic<bool> * cancel) {
    const Clock::time_point deadline = timeout ? Clock::now() + *timeout : Clock::time_point::max();
    for (;;) {
        detail::Graveyard gy;  // released each round, after the lock
        std::unique_lock<std::mutex> lk(rt_->mu);
        bool fin = false;
        PrewarmResult r = collect(fin, gy);
        if (fin || detail::exiting()) return r;
        if (cancel && cancel->load()) return r;
        const Clock::time_point now = Clock::now();
        if (now >= deadline) return r;
        rt_->state_cv.wait_until(lk, std::min(deadline, now + rt_->cfg.wait_poll));
    }
}

// ---- Manager --------------------------------------------------------------

Manager::~Manager() {
    if (rt_) rt_->stop();
}

std::shared_ptr<Manager> Manager::instance() {
    // Leaked on purpose: the scheduler outlives static destruction. Its
    // threads are never joined; at exit the runtime waits, bounded, for the
    // jobs they are running (runtime.h, "Process exit").
    static std::shared_ptr<Manager> * inst = [] {
        detail::register_atexit_once();
        return new std::shared_ptr<Manager>(new Manager(std::make_shared<detail::Runtime>(Config{})));
    }();
    return *inst;
}

std::shared_ptr<Manager> Manager::create(Config cfg) {
    detail::register_atexit_once();
    return std::shared_ptr<Manager>(new Manager(std::make_shared<detail::Runtime>(std::move(cfg))));
}

std::shared_ptr<Slot> Manager::make_slot(SlotSpec spec, SlotPolicy policy) {
    auto c = std::make_shared<detail::SlotCore>();
    c->spec = std::move(spec);
    c->policy = std::move(policy);
    c->rt = rt_;
    if (c->spec.route == Route::none) {
        c->state = c->spec.none_state;
        c->reason = c->spec.none_reason;
        c->status_route = StatusRoute::none;
    } else {
        c->state = SlotState::idle;
        if (c->spec.route == Route::prebuilt) {
            c->status_route = StatusRoute::prebuilt;
            c->artifact_path = c->spec.artifact_path;
            // Prebuilt sidecars get hints keyed by their realpath.
            c->hint_path = detail::hint_path_for(c->policy.cache_dir, rt_->host_id, c->artifact_path,
                                                 c->spec.function, to_string(c->spec.units), rt_->os_build);
        }
    }
    // A slot that can load, made on the main thread, arms the exit guard (and
    // so pins this image) even if its loads are requested from another
    // thread: the main thread is the one that usually calls exit(). Off the
    // main thread nothing is pinned until work is requested.
    if (c->spec.route != Route::none && detail::automatic_exit_guard() && detail::on_main_thread()) {
        rt_->pin_for_work();
    }
    rt_->add_slot(c);
    if (c->spec.pool && c->spec.pool->rt_ == rt_) {
        c->pool = c->spec.pool->core_;
        rt_->pool_add(*c->spec.pool->core_, c);
    }
    return std::shared_ptr<Slot>(new Slot(c));
}

std::shared_ptr<ResidentPool> Manager::make_pool(size_t cap) {
    auto core = std::make_shared<detail::PoolCore>();
    core->cap = cap;
    return std::shared_ptr<ResidentPool>(new ResidentPool(core, rt_));
}

std::shared_ptr<Group> Manager::issue_group(GroupKind kind, const std::vector<GroupRequest> & requests, bool waited,
                                            std::vector<SlotStatus> skipped_route_none) {
    auto g = std::shared_ptr<Group>(new Group());
    g->kind_ = kind;
    g->rt_ = rt_;
    g->skipped_ = std::move(skipped_route_none);
    rt_->pin_for_work();
    detail::Graveyard gy;
    std::lock_guard<std::mutex> lk(rt_->mu);
    const Clock::time_point now = Clock::now();
    for (const GroupRequest & req : requests) {
        if (!req.slot) continue;
        const std::shared_ptr<detail::SlotCore> & c = req.slot->core_;
        Group::Entry e;
        e.core = c;
        e.baseline = c->ready_events;
        if (c->state == SlotState::ready && c->model && !c->model->retired()) {
            e.reached_ready = true;
            e.last = rt_->status_locked(*c, now);
        }
        rt_->request_locked(c, req.priority, waited, gy);
        g->entries_.push_back(std::move(e));
    }
    return g;
}

void Manager::notify() { rt_->notify(); }

EngineStatus Manager::status(const std::string & engine, const std::string & cache_dir,
                             const std::vector<std::shared_ptr<Slot>> & slots, GenerationStatus generation) const {
    EngineStatus s;
    s.engine = engine;
    s.cache_dir = cache_dir;
    s.generation = std::move(generation);
    for (const auto & slot : slots) {
        if (slot) s.stages.push_back(slot->status());
    }
    return s;
}

const std::string & Manager::host_id() const { return rt_->host_id; }
const std::string & Manager::os_build() const { return rt_->os_build; }
OsVersion Manager::os_version() const { return rt_->os; }
Platform Manager::platform() const { return rt_->platform; }
Stats Manager::stats() const { return rt_->stats_snapshot(); }
std::vector<std::string> Manager::events() const { return rt_->events_snapshot(); }

// ---- plan() ---------------------------------------------------------------

Plan plan(const PlanInput & in, const Resolved & r) {
    auto none = [](SlotState st, std::string reason) {
        Plan p;
        p.route = Route::none;
        p.none_state = st;
        p.reason = std::move(reason);
        return p;
    };
    if (!in.coreml_built) return none(SlotState::unsupported, "build");
    if (r.use == Use::off) return none(SlotState::disabled, r.disabled_reason.empty() ? "use=off" : r.disabled_reason);
    if (in.platform == Platform::other) return none(SlotState::unsupported, "platform");
    // An adjacent prebuilt wins over the cache and generation, keeps
    // today's OS support and fit, and is never suppressed by validation.
    if (in.prebuilt_present) {
        Plan p;
        p.route = Route::prebuilt;
        return p;
    }
    if (!in.generation_built || !in.template_available) return none(SlotState::absent, "no sidecar");
    if (!in.template_matched) return none(SlotState::no_template, "template does not match the GGUF");
    if (r.generate == Generate::never) return none(SlotState::disabled, "generate=never");
    if (r.cache_dir.empty()) return none(SlotState::unsupported, "no_cache_dir");
    if (in.platform != Platform::macos) return none(SlotState::unsupported, "os");  // first deliverable: macOS
    if (!os_at_least(in.os, 15, 0) || (in.multifunction_ane && !os_at_least(in.os, 15, 7))) {
        return none(SlotState::unsupported, "os");
    }
    if (in.device_excluded) return none(SlotState::unsupported, "device");
    if (std::find(in.validated_units.begin(), in.validated_units.end(), in.units) == in.validated_units.end()) {
        return none(SlotState::unsupported, kUnvalidatedUnits);
    }
    Plan p;
    p.route = Route::generate;
    return p;
}

// ---- process exit hooks ---------------------------------------------------------

bool arm_exit_guard() {
    // Off the main thread, or once finalization began, nothing is pinned or
    // registered.
    if (!detail::on_main_thread() || detail::finalizing()) return false;
    detail::register_atexit_once();
    // The guard's destructor is code in this image: pin it first.
    const detail::PinState ps = detail::pin_image();
    if (ps != detail::PinState::pinned && ps != detail::PinState::main_image) return false;
    detail::register_fork_handler_once();
    detail::note_explicit_arm();
    return detail::arm_exit_guard();
}

bool shutdown_for_exit(std::chrono::milliseconds bound) {
    // Even an unused image must retain its closed admission state if the
    // host unloads and reopens it. During finalization only report drainage.
    if (!detail::finalizing()) detail::pin_image();
    return detail::wait_for_jobs_at_exit(bound);
}

void set_automatic_exit_guard(bool on) { detail::set_automatic_exit_guard_flag(on); }

}  // namespace QVAC_COREML_NS
