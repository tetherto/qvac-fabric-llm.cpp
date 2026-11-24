// Scheduler: prepare queue + load lane behind one admission gate,
// memory admission for cold ANE loads, would_block re-queue, resident cap,
// observed-warm hints and the bounded wait.

#include "runtime.h"

#include "device.h"
#include "hints.h"
#include "test_hooks.h"

#include <algorithm>
#include <functional>

#if defined(__APPLE__)
#include <pthread.h>
#include <sys/qos.h>
#endif
#include <unistd.h>

namespace QVAC_COREML_NS {
namespace detail {

namespace {

int64_t ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
}

bool same_core(const std::weak_ptr<SlotCore> & w, const std::shared_ptr<SlotCore> & c) {
    return !w.owner_before(c) && !c.owner_before(w);
}

int64_t * timing_slot(Timings & t, SlotState phase) {
    switch (phase) {
        case SlotState::digesting: return &t.digest;
        case SlotState::generating: return &t.write;
        case SlotState::compiling: return &t.compile;
        case SlotState::loading: return &t.load;
        case SlotState::warming: return &t.warm;
        default: return nullptr;
    }
}

constexpr int64_t kStaleSeconds = 24 * 60 * 60;
// Consecutive regenerations of a slot whose generated artifact vanished before
// its load; then the slot fails.
constexpr int kMaxRegenerations = 2;

#ifdef QVAC_COREML_TEST_HOOKS
#define QVAC_COREML_TEST_HOOK(name)                              \
    do {                                                         \
        if (void (*hook_)() = test_hooks().name.load()) hook_(); \
    } while (0)
#else
#define QVAC_COREML_TEST_HOOK(name) \
    do {                            \
    } while (0)
#endif

// A pool's reference to an instance one of its members dropped, kept in a
// Graveyard: destroyed outside mu, it releases the reference first and then
// removes its entry from the pool's pending releases (drop_model_locked). The
// pool counts the instance while any entry for it remains, so the order in
// which a graveyard releases several references to it does not matter.
struct PendingRelease {
    std::shared_ptr<SharedModel> model;
    const SharedModel * id = nullptr;
    std::weak_ptr<PoolCore> pool;
    std::weak_ptr<Runtime> rt;
    PendingRelease(std::shared_ptr<SharedModel> m, std::weak_ptr<PoolCore> p, std::weak_ptr<Runtime> r)
        : model(std::move(m)), id(model.get()), pool(std::move(p)), rt(std::move(r)) {}
    PendingRelease(const PendingRelease &) = delete;
    PendingRelease & operator=(const PendingRelease &) = delete;
    ~PendingRelease() {
        model.reset();  // may run the engine's deleter (takes mu itself)
        QVAC_COREML_TEST_HOOK(pending_released);
        if (std::shared_ptr<Runtime> r = rt.lock()) r->on_pool_released(pool, id);
    }
};

}  // namespace

#ifdef QVAC_COREML_TEST_HOOKS
TestHooks & test_hooks() {
    static TestHooks h;
    return h;
}
#endif

namespace {
void key_field(std::string & k, const std::string & f) {
    k += std::to_string(f.size());
    k += ':';
    k += f;
}

std::string retire_key_of(const SharedModel & m) {
    const LoadRequest & r = m.request();
    return retire_key(r.artifact_path, r.function, r.units);
}

bool has_prefix(const std::string & s, const std::string & p) { return s.compare(0, p.size(), p) == 0; }
}  // namespace

std::string retire_key(const std::string & artifact, const std::string & function, ComputeUnits units) {
    std::string k;
    key_field(k, artifact);
    key_field(k, function);
    key_field(k, to_string(units));
    return k;
}

std::string model_key(const std::string & artifact, const std::string & function, ComputeUnits units,
                      const std::vector<int64_t> & geometry) {
    // The geometry is part of the sharing identity: an owner whose loader
    // validates another geometry never adopts this model or joins its load,
    // so every geometry is validated by its own LoadFn. Retirement ignores it.
    std::string g;
    for (int64_t v : geometry) key_field(g, std::to_string(v));
    std::string k = retire_key(artifact, function, units);
    key_field(k, g);
    return k;
}

Runtime::Runtime(Config c) : cfg(std::move(c)) {
    host_id = cfg.host_id.empty() ? device_host_id() : cfg.host_id;
    os_build = cfg.os_build.empty() ? device_os_build() : cfg.os_build;
    os = cfg.os ? *cfg.os : device_os_version();
    platform = cfg.platform ? *cfg.platform : device_platform();
}

Runtime::~Runtime() {
    // stop() joined the threads, or this is the last reference released by a
    // worker thread that already left its loop.
    if (prepare_thread_.joinable()) prepare_thread_.detach();
    if (load_thread_.joinable()) load_thread_.detach();
}

void Runtime::stop() {
    {
        std::lock_guard<std::mutex> lk(mu);
        stopping_ = true;
        bump_locked();
    }
    for (std::thread * t : {&prepare_thread_, &load_thread_}) {
        if (!t->joinable()) continue;
        if (t->get_id() == std::this_thread::get_id()) {
            t->detach();
        } else {
            t->join();
        }
    }
}

void Runtime::bump_locked() {
    ++gen_;
    work_cv.notify_all();
    state_cv.notify_all();
}

void Runtime::notify() {
    std::lock_guard<std::mutex> lk(mu);
    bump_locked();
}

void Runtime::event_locked(std::string e) {
    if (events_.size() >= 4096) events_.erase(events_.begin(), events_.begin() + 1024);
    events_.push_back(std::move(e));
}

void Runtime::ensure_threads_locked() {
    if (threads_started_ || stopping_) return;
    threads_started_ = true;
    std::shared_ptr<Runtime> self = shared_from_this();
    prepare_thread_ = std::thread([self] { self->worker_main(JobKind::prepare); });
    load_thread_ = std::thread([self] { self->worker_main(JobKind::load); });
}

Stats Runtime::stats_snapshot() {
    std::lock_guard<std::mutex> lk(mu);
    return stats_;
}

std::vector<std::string> Runtime::events_snapshot() {
    std::lock_guard<std::mutex> lk(mu);
    return events_;
}

// ---- subscriptions --------------------------------------------------------

std::shared_ptr<SlotCore> Runtime::live_core(const Sub & s, const Job & j, Graveyard & gy) const {
    std::shared_ptr<SlotCore> c = s.core.lock();
    if (!c) return nullptr;
    gy.keep(c);
    if (c->epoch != s.epoch || c->job.get() != &j) return nullptr;
    return c;
}

Runtime::JobView Runtime::view_locked(const Job & j, Graveyard & gy) const {
    JobView v;
    bool all_deferrable = true;
    for (const Sub & s : j.subs) {
        if (!live_core(s, j, gy)) continue;
        v.live = true;
        if (static_cast<int>(s.prio) > static_cast<int>(v.prio)) v.prio = s.prio;
        if (s.prio != Priority::prewarm || s.non_deferrable) all_deferrable = false;
    }
    v.deferrable = v.live && all_deferrable;
    return v;
}

void Runtime::set_subs_state_locked(Job & j, SlotState st, Graveyard & gy) {
    for (const Sub & s : j.subs) {
        if (std::shared_ptr<SlotCore> c = live_core(s, j, gy)) c->state = st;
    }
}

void Runtime::cancel_locked(const std::shared_ptr<Job> & j) {
    auto it = std::find(queue_.begin(), queue_.end(), j);
    if (it != queue_.end()) queue_.erase(it);
    auto f = inflight_.find(j->dedup_key);
    if (f != inflight_.end() && f->second == j) inflight_.erase(f);
    j->done = true;
    ++stats_.cancelled;
    event_locked("cancel:" + std::string(j->kind == JobKind::load ? "load:" + j->load.stage + "/" + j->load.function
                                                                     : "prepare:" + j->prepare.key));
}

void Runtime::unsubscribe_locked(const std::shared_ptr<SlotCore> & c, Graveyard & gy) {
    std::shared_ptr<Job> j = std::move(c->job);
    c->job.reset();
    if (!j) return;
    j->subs.erase(std::remove_if(j->subs.begin(), j->subs.end(), [&](const Sub & s) { return same_core(s.core, c); }),
                  j->subs.end());
    if (!j->started && !j->done && !view_locked(*j, gy).live) cancel_locked(j);
}

void Runtime::drop_model_locked(SlotCore & c, Graveyard & gy) {
    std::shared_ptr<SharedModel> m = std::move(c.model);
    c.model.reset();
    if (!m) return;
    std::shared_ptr<PoolCore> p = c.pool.lock();
    if (!p) {
        gy.keep(m);  // no resident cap to account for
        return;
    }
    // Also when another member or a Handle still counts the instance: that
    // reference may be released first, on another thread, leaving this one
    // the last.
    p->releasing.push_back(m.get());
    gy.items.push_back(std::make_shared<PendingRelease>(std::move(m), p, weak_from_this()));
}

void Runtime::on_pool_released(const std::weak_ptr<PoolCore> & wp, const SharedModel * m) {
    std::shared_ptr<PoolCore> p = wp.lock();
    {
        std::lock_guard<std::mutex> lk(mu);
        if (p) {
            auto it = std::find(p->releasing.begin(), p->releasing.end(), m);
            if (it != p->releasing.end()) p->releasing.erase(it);
        }
        bump_locked();  // admission may take the place now
    }
}

void Runtime::evict_locked(const std::shared_ptr<SlotCore> & c, Graveyard & gy) {
    drop_model_locked(*c, gy);
    ++c->epoch;
    unsubscribe_locked(c, gy);
    if (!is_terminal(c->state)) c->state = SlotState::idle;
    c->queued_behind = false;
    ++stats_.evictions;
    event_locked("evict:" + c->spec.stage + "/" + c->spec.function);
    bump_locked();
}

void Runtime::subscribe_load_locked(const std::shared_ptr<SlotCore> & c, Priority p, bool nd, Graveyard & gy) {
    const std::string key = model_key(c->artifact_path, c->spec.function, c->spec.units, c->spec.geometry);
    const std::string rkey = retire_key(c->artifact_path, c->spec.function, c->spec.units);
    if (auto r = retired_.find(rkey); r != retired_.end()) {
        c->state = SlotState::retired;
        c->reason = r->second;
        return;
    }
    // Another owner's live instance is adopted at once only when this slot's
    // pool already counts it or has room. Otherwise a queued job takes
    // it through admission (evict or stay cap-blocked) and run_load()
    // publishes the existing instance without loading again. The only strong
    // reference taken here becomes the slot's own.
    const SharedModel * existing = c->spec.shareable ? registered_locked(key, rkey) : nullptr;
    if (existing && pool_takes_locked(*c, existing, gy)) {
        if (std::shared_ptr<SharedModel> m = registry_[key].model.lock()) {
            if (c->model != m) drop_model_locked(*c, gy);
            c->model = std::move(m);
            c->state = SlotState::ready;
            c->last_used = Clock::now();
            ++c->ready_events;
            c->ready_snapshot = status_locked(*c, Clock::now());
            return;
        }
    }
    // Stateful templates never share a model; the same slot may still rejoin
    // its own running job after an eviction.
    const std::string dedup = c->spec.shareable ? key : key + "#" + std::to_string(c->id);
    if (auto it = inflight_.find(dedup); it != inflight_.end() && !it->second->done) {
        std::shared_ptr<Job> j = it->second;
        j->subs.push_back(Sub{c, c->epoch, p, nd});
        c->job = j;
        c->state = j->started ? j->phase : (j->deferred_memory ? SlotState::deferred_memory : SlotState::queued);
        // A late subscriber to a running load: its pool was not admitted at
        // start. It gets a place if the pool has room; otherwise publish()
        // hands it a queued job of its own.
        if (j->started) reserve_locked(*j, *c, gy);
        return;
    }
    auto j = std::make_shared<Job>();
    j->kind = JobKind::load;
    j->seq = ++seq_;
    j->dedup_key = dedup;
    j->model_key = key;
    j->retire_key = rkey;
    j->load.engine = c->spec.engine;
    j->load.stage = c->spec.stage;
    j->load.artifact_path = c->artifact_path;
    j->load.function = c->spec.function;
    j->load.units = c->spec.units;
    j->load.tmpl = c->spec.tmpl;
    j->load.geometry = c->spec.geometry;
    j->load_fn = c->spec.load_fn;
    j->warm_fn = c->spec.warm_fn;
    j->shareable = c->spec.shareable;
    j->ane = units_include_ane(c->spec.units);
    j->ane_reserve = c->spec.ane_compile_reserve_bytes;
    j->hint_path = c->hint_path;
    j->expected_warm_ms = c->spec.expected_warm_load_ms;
    j->enqueued_at = Clock::now();
    j->subs.push_back(Sub{c, c->epoch, p, nd});
    queue_.push_back(j);
    inflight_[dedup] = j;
    c->job = j;
    c->state = SlotState::queued;
    c->queued_behind = false;
}

void Runtime::subscribe_prepare_locked(const std::shared_ptr<SlotCore> & c, Priority p, bool nd) {
    const std::string dedup = c->spec.prepare.key.empty()
                                  ? "prepare#" + std::to_string(c->id)
                                  : "prepare\n" + c->spec.prepare.key;
    if (auto it = inflight_.find(dedup); it != inflight_.end() && !it->second->done) {
        std::shared_ptr<Job> j = it->second;
        j->subs.push_back(Sub{c, c->epoch, p, nd});
        c->job = j;
        c->state = j->started ? j->phase : SlotState::queued;
        return;
    }
    auto j = std::make_shared<Job>();
    j->kind = JobKind::prepare;
    j->seq = ++seq_;
    j->dedup_key = dedup;
    j->prepare = c->spec.prepare;
    j->prepare_fn = c->spec.prepare_fn;
    j->load.engine = c->spec.engine;
    j->load.stage = c->spec.stage;
    j->load.function = c->spec.function;
    j->load.units = c->spec.units;
    j->enqueued_at = Clock::now();
    j->subs.push_back(Sub{c, c->epoch, p, nd});
    queue_.push_back(j);
    inflight_[dedup] = j;
    c->job = j;
    c->state = SlotState::queued;
}

bool Runtime::refuse_after_exit_locked(const std::shared_ptr<SlotCore> & c, Graveyard & gy) {
    // After exit began no job starts, so a slot with no model fails with the
    // reason instead of staying idle or queued for good: a fresh request, and
    // a slot whose job never started (unsubscribed; a job left with no
    // subscriber is cancelled). A ready slot keeps its model; a slot whose
    // job is running keeps it, and that job still publishes.
    if (c->spec.route == Route::none || c->model || is_terminal(c->state)) return false;
    if (c->job && c->job->started) return false;
    unsubscribe_locked(c, gy);
    c->state = SlotState::failed;
    c->error = Error::deterministic(kExitingReason);
    c->reason = c->error.message;
    c->queued_behind = false;
    note_refused_request();
    bump_locked();
    return true;
}

void Runtime::refuse_queued_after_exit_locked(const std::shared_ptr<SlotCore> & c, Graveyard & gy) {
    if (c->job && exiting()) refuse_after_exit_locked(c, gy);
}

void Runtime::request_locked(const std::shared_ptr<SlotCore> & c, Priority p, bool nd, Graveyard & gy) {
    if (stopping_) return;
    if (exiting()) {
        refuse_after_exit_locked(c, gy);
        return;
    }
    if (c->model && c->model->retired()) {
        drop_model_locked(*c, gy);
        c->state = SlotState::retired;
        if (auto r = retired_.find(retire_key(c->artifact_path, c->spec.function, c->spec.units)); r != retired_.end()) {
            c->reason = r->second;
        }
        return;
    }
    if (c->state == SlotState::ready) {
        c->last_used = Clock::now();
        return;
    }
    if (is_terminal(c->state)) return;
    if (c->job) {
        for (Sub & s : c->job->subs) {
            if (same_core(s.core, c) && s.epoch == c->epoch) {
                if (static_cast<int>(p) > static_cast<int>(s.prio)) s.prio = p;
                s.non_deferrable = s.non_deferrable || nd;
            }
        }
        bump_locked();
        return;
    }
    if (c->spec.route == Route::none) return;
    if (cfg.pin_image && pin_state() != PinState::pinned && pin_state() != PinState::main_image) {
        // No asynchronous work from an image that a dlclose could unmap.
        // pin_for_work() ran before mu was taken; this slot falls back to ggml.
        c->state = SlotState::failed;
        c->error = Error::deterministic("the library image could not be pinned against unload");
        c->reason = c->error.message;
        bump_locked();
        return;
    }
    if (!c->artifact_path.empty()) {
        subscribe_load_locked(c, p, nd, gy);
    } else if (c->spec.route == Route::generate && c->spec.prepare_fn != nullptr) {
        subscribe_prepare_locked(c, p, nd);
    } else {
        c->state = SlotState::failed;
        c->error = Error::deterministic("no artifact path and no prepare function");
        c->reason = c->error.message;
        bump_locked();
        return;
    }
    ensure_threads_locked();
    bump_locked();
}

void Runtime::pin_for_work() {
    // Synchronous, on the caller's thread, before any request can start a
    // worker or publish a job: the caller still holds its reference to this
    // image, so a dlclose right after it returns cannot unmap running code.
    if (cfg.pin_image) pin_image();
    // The guard's destructor runs at exit, so only an image that stays mapped
    // may arm it.
    const PinState ps = pin_state();
    if (ps != PinState::pinned && ps != PinState::main_image) return;
    register_fork_handler_once();
    if (automatic_exit_guard()) arm_exit_guard();
}

void Runtime::request(const std::shared_ptr<SlotCore> & c, Priority p, bool nd) {
    pin_for_work();
    Graveyard gy;
    std::lock_guard<std::mutex> lk(mu);
    request_locked(c, p, nd, gy);
}

// ---- admission ------------------------------------------------------------

bool Runtime::low_power_locked() const { return cfg.low_power ? cfg.low_power() : device_low_power(); }

bool Runtime::hinted(const std::string & path) const {
    if (path.empty()) return false;
    std::optional<HintRecord> h = read_hint(path);
    return h.has_value() && !hint_stale(*h, unix_now_s());
}

// Expected warm time of a job's load: -1 when it has no non-stale hint, or
// when neither an observed warm time nor a template estimate exists.
int64_t Runtime::job_w(const Job & j) const {
    if (j.kind != JobKind::load || j.hint_path.empty()) return -1;
    std::optional<HintRecord> h = read_hint(j.hint_path);
    if (!h || hint_stale(*h, unix_now_s())) return -1;
    if (h->warm_load_ms) return *h->warm_load_ms;
    if (j.expected_warm_ms) return *j.expected_warm_ms;
    return -1;
}

std::shared_ptr<PoolRef> Runtime::pool_ref_locked(const SlotCore & c) {
    std::shared_ptr<PoolCore> p = c.pool.lock();
    if (!p || !c.model) return nullptr;
    auto & v = p->refs;
    // A ref without holds counts nothing; the handle being built adds its own.
    v.erase(std::remove_if(v.begin(), v.end(),
                           [&](const std::shared_ptr<PoolRef> & r) {
                               return r->holds.load() == 0 && r->model != c.model.get();
                           }),
            v.end());
    for (const std::shared_ptr<PoolRef> & r : v) {
        if (r->model == c.model.get()) return r;
    }
    auto r = std::make_shared<PoolRef>();
    r->model = c.model.get();
    v.push_back(r);
    return r;
}

// The identity of the live, non-retired registry instance for `key`, or null.
// No strong reference: identity only, for comparisons under mu.
const SharedModel * Runtime::registered_locked(const std::string & key, const std::string & retire) const {
    if (retired_.count(retire) != 0) return nullptr;
    auto it = registry_.find(key);
    if (it == registry_.end() || it->second.model.expired()) return nullptr;
    return it->second.id;
}

// The live instance a load job would publish without loading (another owner's,
// in the registry), or null. A started job publishes exactly the instance it
// retained at admission (Job::adopt), and loads when it retained none; a queued
// job is only compared against the registry, by identity.
const SharedModel * Runtime::existing_locked(const Job & j) const {
    if (j.kind != JobKind::load || !j.shareable) return nullptr;
    if (j.started) return j.adopt.get();
    return registered_locked(j.model_key, j.retire_key);
}

size_t Runtime::pool_live_locked(const PoolCore & p, Graveyard & gy) const {
    // Distinct instances this pool references (see PoolCore), without strong
    // references to them, plus the load in flight for this pool unless that
    // load brings an instance the pool already counts.
    std::vector<const SharedModel *> seen;
    auto add = [&](const SharedModel * m) {
        if (m && std::find(seen.begin(), seen.end(), m) == seen.end()) seen.push_back(m);
    };
    for (const auto & w : p.members) {
        std::shared_ptr<SlotCore> c = w.lock();
        if (!c) continue;
        gy.keep(c);
        add(c->model.get());
    }
    for (const std::shared_ptr<PoolRef> & r : p.refs) {
        if (r->holds.load() > 0) add(r->model);
    }
    for (const SharedModel * m : p.releasing) add(m);
    const size_t live = seen.size();
    if (!running_serves_locked(p)) return live;
    const SharedModel * existing = existing_locked(*running_);
    const bool counted = existing && std::find(seen.begin(), seen.end(), existing) != seen.end();
    return counted ? live : live + 1;
}

bool Runtime::pool_counts_locked(const PoolCore & p, const SharedModel * m, Graveyard & gy) const {
    for (const auto & w : p.members) {
        std::shared_ptr<SlotCore> c = w.lock();
        if (!c) continue;
        gy.keep(c);
        if (c->model.get() == m) return true;
    }
    for (const std::shared_ptr<PoolRef> & r : p.refs) {
        if (r->model == m && r->holds.load() > 0) return true;
    }
    return false;
}

// Whether slot `c` may reference the existing instance `m` without admission:
// no pool, the pool already counts `m`, or the pool has room for it.
bool Runtime::pool_takes_locked(const SlotCore & c, const SharedModel * m, Graveyard & gy) const {
    std::shared_ptr<PoolCore> p = c.pool.lock();
    if (!p) return true;
    gy.keep(p);
    return pool_counts_locked(*p, m, gy) || pool_live_locked(*p, gy) < p->cap;
}

// The running job `j` takes a place in the pool of subscriber `c` when the
// pool has room (the in-flight load then counts there, see pool_live_locked)
// or already counts the instance `j` brings (no second place: it counts once);
// a full pool reserves nothing and its subscriber goes through admission
// again when the instance is published.
void Runtime::reserve_locked(Job & j, const SlotCore & c, Graveyard & gy) {
    std::shared_ptr<PoolCore> p = c.pool.lock();
    if (!p) return;
    gy.keep(p);
    for (const auto & wp : j.pools) {
        if (wp.lock() == p) return;
    }
    const SharedModel * existing = existing_locked(j);
    if ((existing && pool_counts_locked(*p, existing, gy)) || pool_live_locked(*p, gy) < p->cap) j.pools.push_back(p);
}

bool Runtime::running_serves_locked(const PoolCore & p) const {
    if (!running_ || running_->kind != JobKind::load || running_->done) return false;
    for (const auto & wp : running_->pools) {
        if (wp.lock().get() == &p) return true;
    }
    return false;
}

void Runtime::job_pools_locked(const Job & j, std::vector<std::shared_ptr<SlotCore>> & own,
                               std::vector<std::shared_ptr<PoolCore>> & pools, Graveyard & gy) const {
    for (const Sub & s : j.subs) {
        std::shared_ptr<SlotCore> c = live_core(s, j, gy);
        if (!c) continue;
        own.push_back(c);
        if (std::shared_ptr<PoolCore> p = c->pool.lock()) {
            if (std::find(pools.begin(), pools.end(), p) == pools.end()) pools.push_back(p);
        }
    }
}

// The least recently used instance this pool can drop: no unit retains it
// through the pool, and every member referencing it is neither predicted for
// the current call nor part of the job asking for room. Other owners'
// references do not matter (they count in their own pools). Returns every
// member referencing it, so evicting them all frees its place; empty if none.
std::vector<std::shared_ptr<SlotCore>> Runtime::pool_victim_locked(const PoolCore & p,
                                                                   const std::vector<std::shared_ptr<SlotCore>> & own,
                                                                   Graveyard & gy) const {
    struct Cand {
        const SharedModel * model = nullptr;
        bool evictable = true;
        Clock::time_point last_used{};
        std::vector<std::shared_ptr<SlotCore>> members;
    };
    std::vector<Cand> cands;
    for (const auto & w : p.members) {
        std::shared_ptr<SlotCore> m = w.lock();
        if (!m) continue;
        gy.keep(m);
        if (!m->model) continue;
        auto it = std::find_if(cands.begin(), cands.end(), [&](const Cand & k) { return k.model == m->model.get(); });
        if (it == cands.end()) {
            Cand k;
            k.model = m->model.get();
            it = cands.insert(cands.end(), std::move(k));
        }
        bool predicted = false;
        for (const auto & pw : p.predicted) predicted = predicted || same_core(pw, m);
        if (predicted || std::find(own.begin(), own.end(), m) != own.end()) it->evictable = false;
        it->last_used = std::max(it->last_used, m->last_used);
        it->members.push_back(m);
    }
    const Cand * victim = nullptr;
    for (const Cand & k : cands) {
        if (!k.evictable) continue;
        bool held = false;
        for (const std::shared_ptr<PoolRef> & r : p.refs) held = held || (r->model == k.model && r->holds.load() > 0);
        if (held) continue;
        if (!victim || k.last_used < victim->last_used) victim = &k;
    }
    return victim ? victim->members : std::vector<std::shared_ptr<SlotCore>>{};
}

// Whether the pool of subscriber `c` denies it the instance job `j` brings:
// full, not already counting that instance, nothing in it evictable for
// this job, and no load in flight for it (that load may publish an evictable
// instance: undecided until it does). A pool the running job reserved a place
// in serves its subscribers.
bool Runtime::sub_denied_locked(const Job & j, const SlotCore & c, const std::vector<std::shared_ptr<SlotCore>> & own,
                                Graveyard & gy) const {
    std::shared_ptr<PoolCore> p = c.pool.lock();
    if (!p) return false;
    gy.keep(p);
    if (running_serves_locked(*p)) return false;
    if (pool_live_locked(*p, gy) < p->cap) return false;
    if (!p->releasing.empty()) return false;  // undecided until the pending release completes
    if (const SharedModel * existing = existing_locked(j); existing && pool_counts_locked(*p, existing, gy)) {
        return false;
    }
    return pool_victim_locked(*p, own, gy).empty();
}

// The job is denied only when every live subscriber is: a capped owner never
// holds back another owner's load of the same instance.
bool Runtime::cap_denied_locked(const Job & j, Graveyard & gy) const {
    std::vector<std::shared_ptr<SlotCore>> own;
    std::vector<std::shared_ptr<PoolCore>> pools;
    job_pools_locked(j, own, pools, gy);
    if (own.empty()) return false;
    for (const std::shared_ptr<SlotCore> & c : own) {
        if (!sub_denied_locked(j, *c, own, gy)) return false;
    }
    return true;
}

bool Runtime::cap_blocked_now_locked(const SlotCore & c, Graveyard & gy) const {
    const std::shared_ptr<Job> & j = c.job;
    if (!j || j->kind != JobKind::load || j->done || j->deferred_memory) return false;
    std::vector<std::shared_ptr<SlotCore>> own;
    std::vector<std::shared_ptr<PoolCore>> pools;
    job_pools_locked(*j, own, pools, gy);
    return sub_denied_locked(*j, c, own, gy);
}

void Runtime::set_cap_blocked_locked(Job & j, bool blocked) {
    if (j.cap_blocked == blocked) return;
    j.cap_blocked = blocked;
    if (blocked) event_locked("blocked:cap:" + j.load.stage + "/" + j.load.function);
    state_cv.notify_all();
}

Runtime::PoolVerdict Runtime::pool_admits_locked(Job & j, bool retain, std::shared_ptr<SharedModel> & adopt,
                                                 Graveyard & gy) {
    std::vector<std::shared_ptr<SlotCore>> own;
    std::vector<std::shared_ptr<PoolCore>> pools;
    job_pools_locked(j, own, pools, gy);
    // At most two passes: the second runs only when the instance the first one
    // admitted the job to adopt expired before it could be retained.
    for (;;) {
        std::vector<std::shared_ptr<PoolCore>> denied;
        std::vector<std::vector<std::shared_ptr<SlotCore>>> victims;
        // A pool that already counts the instance this job would publish takes
        // it without a new place: nothing is evicted for it. The scan compares
        // identities only and takes no strong reference, so a scan that does
        // not admit the job (blocked, waiting for a release, or handed to the
        // other worker) can never hold the last reference to a model and run
        // its deleter outside every pool's count.
        const SharedModel * existing = existing_locked(j);
        bool releasing = false;
        for (const std::shared_ptr<PoolCore> & p : pools) {
            if (pool_live_locked(*p, gy) < p->cap) continue;
            if (existing && pool_counts_locked(*p, existing, gy)) continue;
            if (!p->releasing.empty()) {
                // A release is on its way out: wait for it rather than evicting
                // a second victim meanwhile.
                releasing = true;
                continue;
            }
            std::vector<std::shared_ptr<SlotCore>> v = pool_victim_locked(*p, own, gy);
            if (v.empty()) {
                denied.push_back(p);
            } else {
                victims.push_back(std::move(v));
            }
        }
        // A denied pool holds back only its own subscribers: they get no place
        // when the job starts and stay pending after it publishes (publish()).
        bool served = own.empty();
        for (const std::shared_ptr<SlotCore> & c : own) {
            std::shared_ptr<PoolCore> p = c->pool.lock();
            served = served || !p || std::find(denied.begin(), denied.end(), p) == denied.end();
        }
        if (!served) return PoolVerdict::blocked;
        if (victims.empty() && !releasing) {
            // Only the worker that starts the job takes the instance, under this
            // same lock, and the caller moves it into Job::adopt before mu is
            // released: selecting the instance and owning it are one step. The
            // prepare worker only hands the job off; the load worker admits it
            // again. If the instance expired since its identity was read, the
            // job is evaluated again as a fresh load.
            if (existing && retain) {
                auto it = registry_.find(j.model_key);
                std::shared_ptr<SharedModel> m = it != registry_.end() ? it->second.model.lock() : nullptr;
                if (!m) continue;
                adopt = std::move(m);
            }
            return PoolVerdict::admit;
        }
        // Each victim counts as a pending release until the reference is
        // released outside mu, which wakes the workers to pick this job again.
        for (const auto & v : victims) {
            for (const std::shared_ptr<SlotCore> & c : v) evict_locked(c, gy);
        }
        return PoolVerdict::wait_release;
    }
}

// Admission screening, separate from job selection so it also runs while
// another job holds the gate: drops jobs without live subscribers, marks
// unhinted cold ANE loads deferred(memory) or lifts that, and refreshes the
// resident-cap denial seen by acquisitions. `wake` gets the memory recheck.
void Runtime::screen_locked(Clock::time_point now, Clock::time_point & wake, Graveyard & gy) {
    std::optional<uint64_t> mem;
    for (size_t i = 0; i < queue_.size();) {
        std::shared_ptr<Job> j = queue_[i];
        if (!view_locked(*j, gy).live) {
            cancel_locked(j);  // erases queue_[i]
            continue;
        }
        ++i;
        if (j->kind != JobKind::load) continue;
        bool short_of_memory = false;
        if (j->ane && j->ane_reserve > 0 && !hinted(j->hint_path)) {
            if (!mem) mem = cfg.available_memory ? cfg.available_memory() : device_available_memory();
            short_of_memory = *mem < j->ane_reserve;
        }
        if (short_of_memory) {
            if (!j->deferred_memory) {
                j->deferred_memory = true;
                j->deferred_since = now;
                set_subs_state_locked(*j, SlotState::deferred_memory, gy);
                event_locked("defer:memory:" + j->load.stage + "/" + j->load.function);
                state_cv.notify_all();
            }
            wake = std::min(wake, now + cfg.memory_recheck);
            continue;
        }
        if (j->deferred_memory) {
            j->deferred_memory = false;
            set_subs_state_locked(*j, SlotState::queued, gy);
            state_cv.notify_all();
        }
        set_cap_blocked_locked(*j, cap_denied_locked(*j, gy));
    }
}

std::shared_ptr<Job> Runtime::pick_locked(Clock::time_point now, Clock::time_point & wake, bool retain,
                                          std::shared_ptr<SharedModel> & adopt, Graveyard & gy) {
    screen_locked(now, wake, gy);
    const bool lp = low_power_locked();
    struct Cand {
        std::shared_ptr<Job> job;
        Priority prio;
    };
    std::vector<Cand> cands;
    for (size_t i = 0; i < queue_.size();) {
        std::shared_ptr<Job> j = queue_[i++];
        if (j->deferred_memory) continue;  // screened above, with the recheck in `wake`
        const JobView v = view_locked(*j, gy);
        if (j->retry_at > now) {
            wake = std::min(wake, j->retry_at);
            continue;
        }
        if (lp && v.deferrable) {
            wake = std::min(wake, now + cfg.memory_recheck);
            continue;
        }
        cands.push_back({j, v.prio});
    }
    std::stable_sort(cands.begin(), cands.end(), [](const Cand & a, const Cand & b) {
        if (a.prio != b.prio) return static_cast<int>(a.prio) > static_cast<int>(b.prio);
        return a.job->seq < b.job->seq;
    });
    for (const Cand & c : cands) {
        if (c.job->kind != JobKind::load) return c.job;
        switch (pool_admits_locked(*c.job, retain, adopt, gy)) {
            case PoolVerdict::admit:
                set_cap_blocked_locked(*c.job, false);
                return c.job;
            case PoolVerdict::wait_release:
                set_cap_blocked_locked(*c.job, false);
                return nullptr;  // re-picked when the evicted model is released
            case PoolVerdict::blocked:
                // Retried when a release bumps the generation (Handle::reset,
                // the model's destructor, set_predicted) or at the recheck.
                set_cap_blocked_locked(*c.job, true);
                wake = std::min(wake, now + cfg.memory_recheck);
                continue;  // a lower-priority job of another stage may run meanwhile
        }
    }
    return nullptr;
}

void Runtime::start_locked(const std::shared_ptr<Job> & j, Clock::time_point now, Graveyard & gy) {
    running_ = j;
    handoff_ = -1;
    queue_.erase(std::remove(queue_.begin(), queue_.end(), j), queue_.end());
    j->started = true;
    j->started_at = now;
    j->phase_at = now;
    j->timings.queued = ms_between(j->enqueued_at, now);
    j->phase = j->kind == JobKind::load ? SlotState::loading : SlotState::digesting;
    j->pools.clear();
    for (const Sub & s : j->subs) {
        std::shared_ptr<SlotCore> c = live_core(s, *j, gy);
        if (!c) continue;
        c->state = j->phase;
        c->timings.queued = j->timings.queued;
        reserve_locked(*j, *c, gy);
    }
    if (j->kind == JobKind::load) {
        j->hinted_at_start = hinted(j->hint_path);
        ++stats_.loads_started;
        event_locked("start:load:" + j->load.stage + "/" + j->load.function);
    } else {
        ++stats_.prepares_started;
        event_locked("start:prepare:" + j->prepare.key);
    }
    state_cv.notify_all();
}

void Runtime::worker_main(JobKind kind) {
#if defined(__APPLE__)
    if (kind == JobKind::prepare) pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
#endif
    std::unique_lock<std::mutex> lk(mu);
    while (!stopping_) {
        // A handoff is a one-shot wake token: consumed by this wake whether or
        // not a job starts, so a handed-off job that became ineligible (low
        // power, cancelled) never keeps the wait predicate true.
        if (handoff_ == static_cast<int>(kind)) handoff_ = -1;
        Graveyard gy;
        const Clock::time_point now = Clock::now();
        Clock::time_point wake = now + std::chrono::hours(1);
        const uint64_t seen = gen_;
        std::shared_ptr<Job> job;
        // Set only when this (load) worker admitted a job to adopt a registry
        // instance; moved into the job below without releasing mu.
        std::shared_ptr<SharedModel> adopt;
        bool handed_off = false;
        // Counted before anything is picked or screened, and released only
        // after the job and after the models this pass dropped (pool victims,
        // cancelled jobs' instances) are released outside mu: an exit that
        // begins meanwhile either waits for that work or makes enter_job()
        // refuse it, and then nothing is picked or screened.
        bool counted = enter_job();
        if (counted && !running_) {
            job = pick_locked(now, wake, kind == JobKind::load, adopt, gy);
            if (job && job->kind == kind) QVAC_COREML_TEST_HOOK(job_selected);
            // Admission is decided again where the job would commit to
            // starting: an exit (or shutdown_for_exit) that began during the
            // selection, while this worker was counted, leaves the job
            // queued for good. A retained adoption is released outside mu.
            if (job && exiting()) {
                gy.keep(adopt);
                adopt.reset();
                job.reset();
            }
            if (job && job->kind != kind) {
                handoff_ = static_cast<int>(job->kind);
                work_cv.notify_all();
                job.reset();  // the other worker admits the job again
                handed_off = true;
            }
        } else if (counted) {
            // The other worker holds the gate: requests and the periodic
            // memory recheck are still screened, without selecting a job.
            screen_locked(now, wake, gy);
        }
        if (job) {
            job->adopt = std::move(adopt);
            start_locked(job, now, gy);
            lk.unlock();
            gy.items.clear();
            QVAC_COREML_TEST_HOOK(job_started);
            if (kind == JobKind::load) {
                run_load(job);
            } else {
                run_prepare(job);
            }
            lk.lock();
            if (running_ == job) running_.reset();
            bump_locked();
            job.reset();
            leave_job();
            continue;
        }
        if (handed_off || !gy.items.empty()) {
            lk.unlock();
            if (handed_off) QVAC_COREML_TEST_HOOK(handed_off);
            gy.items.clear();
            lk.lock();
            if (counted) leave_job();
            if (gen_ != seen) continue;
        } else if (counted) {
            leave_job();
        }
        work_cv.wait_until(lk, wake, [&] {
            return stopping_ || gen_ != seen || handoff_ == static_cast<int>(kind);
        });
    }
}

// ---- job execution (unlocked) --------------------------------------------

void Runtime::set_phase(const std::shared_ptr<Job> & j, SlotState st) {
    Graveyard gy;
    std::lock_guard<std::mutex> lk(mu);
    const Clock::time_point now = Clock::now();
    if (int64_t * t = timing_slot(j->timings, j->phase)) *t = ms_between(j->phase_at, now);
    j->phase = st;
    j->phase_at = now;
    set_subs_state_locked(*j, st, gy);
    state_cv.notify_all();
}

void Runtime::fail(const std::shared_ptr<Job> & j, const Error & e, SlotState st, const std::string & reason) {
    Graveyard gy;
    std::lock_guard<std::mutex> lk(mu);
    fail_locked(j, e, st, gy, reason);
}

void Runtime::fail_locked(const std::shared_ptr<Job> & j, const Error & e, SlotState st, Graveyard & gy,
                          const std::string & reason) {
    j->done = true;
    gy.keep(j->adopt);  // released outside mu
    j->adopt.reset();
    queue_.erase(std::remove(queue_.begin(), queue_.end(), j), queue_.end());  // a job not started yet
    if (auto f = inflight_.find(j->dedup_key); f != inflight_.end() && f->second == j) inflight_.erase(f);
    for (const Sub & s : j->subs) {
        std::shared_ptr<SlotCore> c = live_core(s, *j, gy);
        if (!c) continue;
        c->job.reset();
        c->state = st;
        c->error = e;
        c->reason = reason.empty() ? e.message : reason;
    }
    if (j->kind == JobKind::load) ++stats_.load_failures;
    event_locked("fail:" + j->load.stage + "/" + j->load.function + ":" + e.message);
    bump_locked();
}

void Runtime::regenerate(const std::shared_ptr<Job> & j, const Error & e, int limit) {
    Graveyard gy;
    std::lock_guard<std::mutex> lk(mu);
    j->done = true;
    gy.keep(j->adopt);  // released outside mu
    j->adopt.reset();
    if (auto f = inflight_.find(j->dedup_key); f != inflight_.end() && f->second == j) inflight_.erase(f);
    ++stats_.load_failures;
    event_locked("gone:" + j->load.stage + "/" + j->load.function + ":" + e.message);
    for (const Sub & s : j->subs) {
        std::shared_ptr<SlotCore> c = live_core(s, *j, gy);
        if (!c) continue;
        c->job.reset();
        if (c->spec.route != Route::generate || c->spec.prepare_fn == nullptr ||
            c->regenerations >= limit || stopping_ || exiting()) {
            c->state = SlotState::failed;
            c->error = e;
            c->reason = e.message;
            continue;
        }
        // Back to step 2 of the pipeline: the prepare job regenerates (or
        // adopts what another process published meanwhile).
        ++c->regenerations;
        c->artifact_path.clear();
        c->artifact_key.clear();
        c->hint_path.clear();
        subscribe_prepare_locked(c, s.prio, s.non_deferrable);
    }
    ensure_threads_locked();
    bump_locked();
}

void Runtime::requeue(const std::shared_ptr<Job> & j) {
    Graveyard gy;
    std::lock_guard<std::mutex> lk(mu);
    if (auto r = retired_.find(j->retire_key); j->kind == JobKind::load && r != retired_.end()) {
        // Retired while it ran: a retired key never has a queued load.
        fail_locked(j, Error::deterministic(r->second), SlotState::retired, gy);
        return;
    }
    const Clock::time_point now = Clock::now();
    const int64_t initial = cfg.backoff_initial.count();
    const int64_t maxb = cfg.backoff_max.count();
    j->backoff_ms = j->backoff_ms == 0 ? initial : std::min(j->backoff_ms * 2, maxb);
    j->retry_at = now + std::chrono::milliseconds(j->backoff_ms);
    j->started = false;
    j->completing = false;
    j->phase = SlotState::queued;
    queue_.push_back(j);
    set_subs_state_locked(*j, SlotState::queued, gy);
    ++stats_.requeues;
    event_locked("requeue:" + std::string(j->kind == JobKind::load ? "load:" + j->load.stage + "/" + j->load.function
                                                                      : "prepare:" + j->prepare.key));
    bump_locked();
}

std::optional<HintView> Runtime::record_hint(const Job & j, int64_t ms, bool warm) {
    if (j.hint_path.empty()) return std::nullopt;
    std::lock_guard<std::mutex> hl(hint_mu_);
    HintRecord rec = read_hint(j.hint_path).value_or(HintRecord{});
    if (warm) {
        rec.warm_load_ms = ms;
    } else {
        rec.cold_load_ms = ms;
    }
    rec.last_ok = iso8601_utc(unix_now_s());
    rec.artifact = j.load.artifact_path;  // lets the cache prune attribute the hint
    // stale_until written meanwhile is preserved (read just above).
    write_hint(j.hint_path, rec);
    {
        std::lock_guard<std::mutex> lk(mu);
        ++stats_.hint_writes;
    }
    return HintView{rec.warm_load_ms, rec.cold_load_ms, hint_stale(rec, unix_now_s())};
}

void Runtime::write_stale(const std::string & path, const std::string & artifact) {
    if (path.empty()) return;
    std::lock_guard<std::mutex> hl(hint_mu_);
    HintRecord rec = read_hint(path).value_or(HintRecord{});
    rec.stale_until_s = unix_now_s() + kStaleSeconds;
    rec.artifact = artifact;
    write_hint(path, rec);
}

void Runtime::run_load(const std::shared_ptr<Job> & j) {
    {
        std::shared_ptr<SharedModel> existing;
        bool retired = false;
        std::string reason;
        {
            std::lock_guard<std::mutex> lk(mu);
            if (auto r = retired_.find(j->retire_key); r != retired_.end()) {
                retired = true;
                reason = r->second;
            } else {
                // Retained since admission: the instance this job was admitted
                // to adopt is still alive whoever released it meanwhile. A job
                // admitted without one loads; it never looks the registry up
                // again, since a lookup that found nothing would turn an
                // admitted adoption into a load no pool made room for.
                existing = j->adopt;
            }
        }
        if (retired) {
            fail(j, Error::deterministic(reason), SlotState::retired);
            return;
        }
        if (existing) {
            // Another owner's model is still alive (e.g. a unit retains it after
            // an eviction): no second instance.
            QVAC_COREML_TEST_HOOK(before_adopt);
            publish(j, existing, false, std::nullopt);
            return;
        }
    }

    const Clock::time_point t0 = Clock::now();
    LoadResult r = j->load_fn ? j->load_fn(j->load) : LoadResult{};
    if (!j->load_fn) r.error = Error::deterministic("no LoadFn");
    if (r.would_block) {
        requeue(j);
        return;
    }
    if (r.artifact_gone && !r.model) {
        if (r.lease_fd >= 0) ::close(r.lease_fd);
        // A rejected artifact is regenerated once (its content is the same,
        // only the compile is redone); a vanished one up to kMaxRegenerations.
        regenerate(j, r.error.ok() ? Error::transient("the generated artifact is gone") : r.error,
                   r.rejected ? 1 : kMaxRegenerations);
        return;
    }
    if (!r.error.ok() || !r.model) {
        if (r.lease_fd >= 0) ::close(r.lease_fd);
        fail(j, r.error.ok() ? Error::deterministic("LoadFn returned no model") : r.error);
        return;
    }
    const Clock::time_point t1 = Clock::now();
    if (j->warm_fn) {
        set_phase(j, SlotState::warming);
        Error e = j->warm_fn(r.model.get(), j->load);
        if (!e.ok()) {
            if (r.lease_fd >= 0) ::close(r.lease_fd);
            r.model.reset();
            fail(j, e);
            return;
        }
    }
    const Clock::time_point t2 = Clock::now();
    bool expired = false;
    {
        std::lock_guard<std::mutex> lk(mu);
        j->completing = true;
        expired = j->wait_expired.load();
        j->timings.load = ms_between(t0, t1);
        j->timings.warm = j->warm_fn ? ms_between(t1, t2) : 0;
    }
    // A load counts as warm if a non-stale hint existed when it started; one
    // whose bounded wait expired does not train warm_load_ms.
    const std::optional<HintView> hint = record_hint(*j, ms_between(t0, t2), j->hinted_at_start.load() && !expired);
    std::shared_ptr<SharedModel> sm(
        new SharedModel(std::move(r.model), j->load, j->model_key, r.lease_fd, weak_from_this()));
    publish(j, sm, true, hint);
}

namespace {
class PrepareCtx final : public PrepareContext {
public:
    PrepareCtx(std::function<void(PreparePhase)> on_phase, std::function<bool()> on_abandoned)
        : on_phase_(std::move(on_phase)), on_abandoned_(std::move(on_abandoned)) {}
    void phase(PreparePhase p) override { on_phase_(p); }
    bool abandoned() const override { return on_abandoned_(); }

private:
    std::function<void(PreparePhase)> on_phase_;
    std::function<bool()> on_abandoned_;
};
}  // namespace

void Runtime::run_prepare(const std::shared_ptr<Job> & j) {
    PrepareCtx ctx(
        [this, j](PreparePhase p) {
            set_phase(j, p == PreparePhase::digesting    ? SlotState::digesting
                         : p == PreparePhase::generating ? SlotState::generating
                                                         : SlotState::compiling);
        },
        [this, j]() {
            if (exiting()) return true;
            Graveyard gy;
            std::lock_guard<std::mutex> lk(mu);
            return !view_locked(*j, gy).live;
        });
    PrepareResult r;
    if (j->prepare_fn) {
        r = j->prepare_fn(j->prepare, ctx);
    } else {
        r.error = Error::deterministic("no PrepareFn");
    }
    if (r.would_block) {
        requeue(j);
        return;
    }
    if (!r.unsupported.empty()) {
        // Not a failure: generation cannot run here (status unsupported(<reason>)).
        Error e = r.error;
        e.cls = ErrorClass::none;
        if (e.message.empty()) e.message = "generation is unsupported: " + r.unsupported;
        fail(j, e, SlotState::unsupported, r.unsupported);
        return;
    }
    if (!r.error.ok() || r.artifact_path.empty()) {
        fail(j, r.error.ok() ? Error::deterministic("PrepareFn returned no artifact") : r.error);
        return;
    }
    Graveyard gy;
    std::lock_guard<std::mutex> lk(mu);
    const Clock::time_point now = Clock::now();
    if (int64_t * t = timing_slot(j->timings, j->phase)) *t = ms_between(j->phase_at, now);
    j->done = true;
    if (auto f = inflight_.find(j->dedup_key); f != inflight_.end() && f->second == j) inflight_.erase(f);
    ++stats_.prepares_completed;
    event_locked("prepared:" + j->prepare.key);
    // Every live subscriber adopts the artifact and hands it to the load lane.
    for (const Sub & s : j->subs) {
        std::shared_ptr<SlotCore> c = live_core(s, *j, gy);
        if (!c) continue;
        c->job.reset();
        c->artifact_path = r.artifact_path;
        c->artifact_key = r.artifact_key;
        c->status_route = r.cache_hit ? StatusRoute::cached : StatusRoute::generated;
        c->timings.digest = j->timings.digest;
        c->timings.write = j->timings.write;
        c->timings.compile = j->timings.compile;
        c->hint_path = hint_path_for(c->policy.cache_dir, host_id, c->artifact_path, c->spec.function,
                                     to_string(c->spec.units), os_build);
        subscribe_load_locked(c, s.prio, s.non_deferrable, gy);
    }
    bump_locked();
}

void Runtime::publish(const std::shared_ptr<Job> & j, const std::shared_ptr<SharedModel> & m, bool loaded,
                      const std::optional<HintView> & hint) {
    Graveyard gy;
    std::lock_guard<std::mutex> lk(mu);
    const Clock::time_point now = Clock::now();
    j->done = true;
    gy.keep(j->adopt);  // released outside mu; subscribers hold `m` from here
    j->adopt.reset();
    if (auto f = inflight_.find(j->dedup_key); f != inflight_.end() && f->second == j) inflight_.erase(f);
    if (auto r = retired_.find(j->retire_key); r != retired_.end()) {
        // Retired while this load ran: deliver nothing. The instance is
        // released when the caller drops it, and reports the retirement
        // meanwhile, like every instance of the key.
        m->retired_ = true;
        for (const Sub & s : j->subs) {
            std::shared_ptr<SlotCore> c = live_core(s, *j, gy);
            if (!c) continue;
            c->job.reset();
            c->state = SlotState::retired;
            c->reason = r->second;
        }
        bump_locked();
        return;
    }
    if (j->shareable) registry_[j->model_key] = RegEntry{m, m.get()};
    if (loaded) instances_[j->retire_key].push_back(m.get());  // an adopted instance is listed already
    const std::vector<Sub> subs = j->subs;  // a subscriber may move to a new job below
    for (const Sub & s : subs) {
        std::shared_ptr<SlotCore> c = live_core(s, *j, gy);
        if (!c) continue;  // evicted or destroyed meanwhile: never repopulated
        bool admitted = false;
        if (std::shared_ptr<PoolCore> p = c->pool.lock()) {
            for (const auto & wp : j->pools) admitted = admitted || wp.lock() == p;
        }
        if (!admitted && !pool_takes_locked(*c, m.get(), gy)) {
            // Its pool was neither admitted at start nor reserved when it
            // joined, and it is full: the instance goes through admission
            // like any other load. The subscriber stays pending.
            c->job.reset();
            subscribe_load_locked(c, s.prio, s.non_deferrable, gy);
            continue;
        }
        if (c->model != m) drop_model_locked(*c, gy);  // never drop a model under mu
        c->model = m;
        c->job.reset();
        c->state = SlotState::ready;
        c->regenerations = 0;
        c->reason.clear();
        c->error = Error{};
        c->queued_behind = false;
        c->last_used = now;
        c->timings.queued = j->timings.queued;
        c->timings.load = j->timings.load;
        c->timings.warm = j->timings.warm;
        ++c->ready_events;
        c->ready_snapshot = status_locked(*c, now);
        c->ready_snapshot.hint = hint;
    }
    if (loaded) {
        ++stats_.loads_completed;
        event_locked("ready:" + j->load.stage + "/" + j->load.function);
    }
    bump_locked();
}

// ---- acquisition ----------------------------------------------------------

Runtime::WaitPlan Runtime::plan_bounded_wait_locked(SlotCore & c, std::optional<Clock::time_point> deadline,
                                                    Clock::time_point now, Graveyard & gy) {
    WaitPlan none;
    if (!deadline || c.hint_path.empty()) return none;
    std::optional<HintRecord> h = read_hint(c.hint_path);
    if (!h || hint_stale(*h, unix_now_s())) return none;  // only a non-stale hint is waited on
    int64_t w = -1;
    if (h->warm_load_ms) {
        w = *h->warm_load_ms;
    } else if (c.spec.expected_warm_load_ms) {
        w = *c.spec.expected_warm_load_ms;
    }
    if (w < 0) return none;
    std::shared_ptr<Job> own = c.job;
    if (!own || own->kind != JobKind::load) return none;  // preparation ahead: unbounded
    if (cap_blocked_now_locked(c, gy)) return none;    // nothing can be released for it
    int64_t remaining_ms = 0;
    const int64_t budget = w + w / 2;  // w x 1.5
    if (c.state == SlotState::loading || c.state == SlotState::warming) {
        if (running_ != own) return none;
        remaining_ms = std::max<int64_t>(0, budget - ms_between(own->started_at, now));
    } else if (c.state == SlotState::queued) {
        remaining_ms = budget;
        if (running_ && running_ != own) {
            if (running_->kind != JobKind::load) return none;  // an admitted prepare job ahead
            const int64_t wr = job_w(*running_);
            if (wr < 0) return none;  // an unhinted (possibly cold) load ahead
            remaining_ms += wr;
        }
        const bool lp = low_power_locked();
        for (const std::shared_ptr<Job> & q : queue_) {
            if (q == own || q->retry_at > now || q->deferred_memory) continue;
            const JobView v = view_locked(*q, gy);
            if (!v.live || (lp && v.deferrable)) continue;
            if (v.prio != Priority::now || q->seq > own->seq) continue;
            if (q->kind != JobKind::load) return none;
            const int64_t wq = job_w(*q);
            if (wq < 0) return none;
            remaining_ms += wq;
        }
    } else {
        return none;
    }
    if (now + std::chrono::milliseconds(remaining_ms) > *deadline) return none;
    return WaitPlan{true, w};
}

Acquired Runtime::acquire(const std::shared_ptr<SlotCore> & c, Demand d, std::optional<Clock::time_point> deadline,
                          const std::atomic<bool> * cancel) {
    Acquired out;
    std::string stale_path, stale_artifact;
    if (d == Demand::now && !finalizing()) pin_for_work();
    {
        Graveyard gy;
        std::unique_lock<std::mutex> lk(mu);
        if (d == Demand::now) request_locked(c, Priority::now, false, gy);
        const bool strict = c->policy.strict;
        const Wait wait = strict ? Wait::call : c->policy.wait;
        std::optional<Clock::time_point> strict_deadline;
        if (strict && c->policy.strict_wait_ms) {
            strict_deadline = Clock::now() + std::chrono::milliseconds(*c->policy.strict_wait_ms);
        }
        bool planned = false;
        WaitPlan wp;
        // What a poll collected is released outside mu before the next one,
        // so a long wait never accumulates or pins other slots' state.
        auto release_graveyard = [&] {
            if (gy.items.empty()) return;
            lk.unlock();
            gy.items.clear();
            lk.lock();
        };
        for (;;) {
            const Clock::time_point now = Clock::now();
            if (c->model && c->model->retired()) {
                drop_model_locked(*c, gy);
                c->state = SlotState::retired;
            }
            const SlotState st = c->state;
            out.state = st;
            if (st == SlotState::ready && c->model) {
                c->last_used = now;
                c->queued_behind = false;
                out.model = Handle(c->model, pool_ref_locked(*c));
                out.pending = false;
                return out;
            }
            if (is_terminal(st)) {
                out.pending = false;
                if (strict) {
                    throw CoremlError("coreml: " + c->spec.stage + "/" + c->spec.function + " is " +
                                          state_label(st, c->reason) + (c->reason.empty() ? "" : ": " + c->reason),
                                      st, c->error.cls);
                }
                return out;
            }
            out.pending = true;
            if (stopping_) return out;
            if (exiting()) {
                // Fail a slot whose job cannot start. A running job can
                // still publish, but a late host callback must not wait
                // after framework teardown has begun.
                if ((d == Demand::now || c->job) && refuse_after_exit_locked(c, gy)) continue;
                if (!c->job || !c->job->started) return out;
            }
            if (d == Demand::if_ready || finalizing()) return out;
            if (cancel && cancel->load()) return out;
            if (wait == Wait::call) {
                if (st == SlotState::deferred_memory) {
                    if (strict) {
                        throw CoremlError("coreml: " + c->spec.stage + "/" + c->spec.function +
                                              " is deferred(memory)",
                                          st, ErrorClass::transient);
                    }
                    return out;  // waits never block on memory
                }
                if (st == SlotState::idle) return out;  // evicted or cancelled meanwhile
                if (cap_blocked_now_locked(*c, gy)) {
                    // Nothing can be released, so the replacement is not
                    // loaded and the unit runs with what is ready; strict
                    // cannot serve the unit.
                    if (strict) {
                        throw CoremlError("coreml: " + c->spec.stage + "/" + c->spec.function +
                                              " cannot be served: resident cap reached and every live instance is "
                                              "held or predicted",
                                          st, ErrorClass::transient);
                    }
                    return out;
                }
                Clock::time_point until = now + cfg.wait_poll;
                if (strict_deadline) {
                    if (now >= *strict_deadline) {
                        throw CoremlError("coreml: " + c->spec.stage + "/" + c->spec.function +
                                              " not ready within strict_wait_ms (" + state_label(st, c->reason) + ")",
                                          st, ErrorClass::transient);
                    }
                    until = std::min(until, *strict_deadline);
                }
                state_cv.wait_until(lk, until);
                release_graveyard();
                continue;
            }
            // none / construct: the bounded wait, decided once.
            if (!planned) {
                planned = true;
                wp = plan_bounded_wait_locked(*c, deadline, now, gy);
                if (!wp.wait) return out;
            }
            if (st != SlotState::queued && st != SlotState::loading && st != SlotState::warming) return out;
            if (cap_blocked_now_locked(*c, gy)) return out;
            if (now >= *deadline) {
                const int64_t budget = wp.w_ms + wp.w_ms / 2;
                if ((st == SlotState::loading || st == SlotState::warming) && c->job && running_ == c->job &&
                    !c->job->completing && ms_between(c->job->started_at, now) > budget) {
                    // Its own load ran past w x 1.5: cool the hint down for 24 h
                    // so a cold load is never waited on twice.
                    c->job->wait_expired = true;
                    stale_path = c->hint_path;
                    stale_artifact = c->artifact_path;
                } else if (st == SlotState::queued) {
                    c->queued_behind = true;  // hint untouched
                }
                break;
            }
            state_cv.wait_until(lk, std::min(*deadline, now + cfg.wait_poll));
            release_graveyard();
        }
    }
    write_stale(stale_path, stale_artifact);
    return out;
}

// ---- eviction, retirement, status ----------------------------------------

void Runtime::evict(const std::shared_ptr<SlotCore> & c) {
    Graveyard gy;
    std::lock_guard<std::mutex> lk(mu);
    evict_locked(c, gy);
}

void Runtime::release_slot(const std::shared_ptr<SlotCore> & c) {
    Graveyard gy;
    std::lock_guard<std::mutex> lk(mu);
    drop_model_locked(*c, gy);
    ++c->epoch;
    unsubscribe_locked(c, gy);
    if (!is_terminal(c->state)) c->state = SlotState::idle;
    bump_locked();
}

void Runtime::retire_model(SharedModel & m, const std::string & reason) {
    Graveyard gy;
    std::lock_guard<std::mutex> lk(mu);
    retire_model_locked(m, reason, gy);
}

// Retirement is by key, not by instance: a stateful template has one instance
// per owner, and retiring through one owner's failure path reaches the
// others'. Every instance of the key reports retired(), including one that only
// a unit's Handle still holds (the unit finishes with it). Every slot holding
// one drops it through drop_model_locked, so a pooled slot's reference leaves
// as a pending release of its pool and the place is kept until the native
// deleter has run. `m` and the dropped instances stay alive until `gy` is
// released. A load of the key not started yet ends now, before its admission
// could evict a resident for it; one already running completes without
// delivering anything (publish()).
void Runtime::retire_model_locked(SharedModel & m, const std::string & reason, Graveyard & gy) {
    // By (artifact, function, units): every geometry's instances, queued
    // loads and later acquisitions, not only the failing model's sharers.
    const std::string key = retire_key_of(m);
    m.retired_ = true;
    retired_[key] = reason;
    for (auto it = registry_.begin(); it != registry_.end();) {
        it = has_prefix(it->first, key) ? registry_.erase(it) : std::next(it);
    }
    if (auto it = instances_.find(key); it != instances_.end()) {
        for (SharedModel * i : it->second) i->retired_ = true;
    }
    for (const auto & w : slots_) {
        std::shared_ptr<SlotCore> c = w.lock();
        if (!c) continue;
        gy.keep(c);
        if (!c->model || retire_key_of(*c->model) != key) continue;
        drop_model_locked(*c, gy);
        c->state = SlotState::retired;
        c->reason = reason;
    }
    std::vector<std::shared_ptr<Job>> unstarted;
    for (const std::shared_ptr<Job> & q : queue_) {
        if (q->kind == JobKind::load && q->retire_key == key) unstarted.push_back(q);
    }
    for (const std::shared_ptr<Job> & q : unstarted) {
        fail_locked(q, Error::deterministic(reason), SlotState::retired, gy);
    }
    event_locked("retire:" + m.req_.stage + "/" + m.req_.function + ":" + reason);
    bump_locked();
}

// One locked section and no strong reference of its own: the slot's reference
// leaves only through drop_model_locked, so its pool keeps the place until
// that reference is released, never before.
void Runtime::retire_slot(const std::shared_ptr<SlotCore> & c, const std::string & reason) {
    Graveyard gy;
    std::lock_guard<std::mutex> lk(mu);
    if (c->model) {
        retire_model_locked(*c->model, reason, gy);
        return;
    }
    unsubscribe_locked(c, gy);
    c->state = SlotState::retired;
    c->reason = reason;
    bump_locked();
}

void Runtime::on_model_released(const SharedModel & released) {
    const SharedModel * m = &released;
    std::lock_guard<std::mutex> lk(mu);
    if (auto it = registry_.find(released.key_); it != registry_.end() && it->second.model.expired()) registry_.erase(it);
    if (auto it = instances_.find(retire_key_of(released)); it != instances_.end()) {
        std::vector<SharedModel *> & v = it->second;
        v.erase(std::remove(v.begin(), v.end(), m), v.end());
        if (v.empty()) instances_.erase(it);
    }
    bump_locked();
}

SlotState Runtime::state(const std::shared_ptr<SlotCore> & c) {
    Graveyard gy;
    std::lock_guard<std::mutex> lk(mu);
    refuse_queued_after_exit_locked(c, gy);
    if (c->model && c->model->retired()) return SlotState::retired;
    return c->state;
}

std::string Runtime::hint_path(const std::shared_ptr<SlotCore> & c) {
    std::lock_guard<std::mutex> lk(mu);
    return c->hint_path;
}

SlotStatus Runtime::status_locked(SlotCore & c, Clock::time_point now) const {
    SlotStatus s;
    s.stage = c.spec.stage;
    s.function = c.spec.function;
    s.state = (c.model && c.model->retired()) ? SlotState::retired : c.state;
    s.reason = c.reason;
    s.route = c.status_route;
    s.compute_units = to_string(c.spec.units);
    s.label = s.state == SlotState::ready ? label_for(c.spec.units) : "ggml";
    s.artifact_key = c.artifact_key;
    s.path = c.artifact_path;
    s.timings_ms = c.timings;
    if (c.job && (c.state == SlotState::loading || c.state == SlotState::warming || c.state == SlotState::digesting ||
                  c.state == SlotState::generating || c.state == SlotState::compiling)) {
        s.timings_ms.queued = c.job->timings.queued;
    }
    s.error = c.error.message;
    s.error_class = c.error.cls;
    if (c.queued_behind) s.wait = "queued_behind";
    if (s.state == SlotState::deferred_memory && c.job) s.deferred_ms = ms_between(c.job->deferred_since, now);
    return s;
}

SlotStatus Runtime::status(const std::shared_ptr<SlotCore> & c) {
    SlotStatus s;
    std::string path;
    {
        Graveyard gy;
        std::lock_guard<std::mutex> lk(mu);
        refuse_queued_after_exit_locked(c, gy);
        s = status_locked(*c, Clock::now());
        path = c->hint_path;
    }
    if (std::optional<HintRecord> h = read_hint(path)) {
        s.hint = HintView{h->warm_load_ms, h->cold_load_ms, hint_stale(*h, unix_now_s())};
    }
    return s;
}

void Runtime::add_slot(const std::shared_ptr<SlotCore> & c) {
    Graveyard gy;
    std::lock_guard<std::mutex> lk(mu);
    slots_.erase(std::remove_if(slots_.begin(), slots_.end(), [](const std::weak_ptr<SlotCore> & w) { return w.expired(); }),
                 slots_.end());
    c->id = ++slot_ids_;
    slots_.push_back(c);
}

void Runtime::pool_add(PoolCore & p, const std::shared_ptr<SlotCore> & c) {
    std::lock_guard<std::mutex> lk(mu);
    p.members.push_back(c);
}

void Runtime::pool_set_predicted(PoolCore & p, const std::vector<std::shared_ptr<SlotCore>> & cores) {
    std::lock_guard<std::mutex> lk(mu);
    p.predicted.assign(cores.begin(), cores.end());
    bump_locked();
}

size_t Runtime::pool_live(const PoolCore & p) {
    Graveyard gy;
    std::lock_guard<std::mutex> lk(mu);
    return pool_live_locked(p, gy);
}

}  // namespace detail
}  // namespace QVAC_COREML_NS
