#pragma once

// Slots, shared models and handles.
//
// Ownership contract:
//  - An engine owns its slots (std::shared_ptr<Slot>); a Slot holds a
//    std::shared_ptr<SlotCore>. Jobs and groups hold only weak references, so
//    dropping a Slot never waits: unstarted jobs left without a subscriber are
//    cancelled, and a started job runs to completion and publishes only to live
//    subscribers of the current epoch.
//  - Job inputs (LoadRequest, PrepareRequest) are immutable values. Callbacks
//    are plain function pointers, so they cannot capture an engine, an addon
//    environment or a ggml tensor.
//  - Loaded models live in a process-wide (per library image) registry keyed by
//    (artifact path, function, units) as weak references; a SharedModel dies
//    with its last user. Every prediction takes the model's mutex. Stateful
//    templates (shareable = false) get one model per owner.

#include "qvac_coreml/options.h"
#include "qvac_coreml/status.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace QVAC_COREML_NS {

namespace detail {
class Runtime;
struct SlotCore;
struct PoolCore;
struct PoolRef;
}  // namespace detail

using Clock = std::chrono::steady_clock;

// Scheduling priorities: now > current > minimal > prewarm.
enum class Priority : int { prewarm = 0, minimal = 1, current = 2, now = 3 };

const char * to_string(Priority p);

enum class Demand {
    now,       // a unit is starting: raise the slot to `now` and apply the wait mode
    if_ready,  // return the model only if it is ready; request nothing, never wait
};

struct Error {
    ErrorClass cls = ErrorClass::none;
    std::string domain;
    int64_t code = 0;
    std::string message;

    bool ok() const { return cls == ErrorClass::none; }
    static Error transient(std::string msg, std::string domain = {}, int64_t code = 0);
    static Error deterministic(std::string msg, std::string domain = {}, int64_t code = 0);
};

struct LoadRequest {
    std::string engine;
    std::string stage;
    std::string artifact_path;      // .mlmodelc (prebuilt or published)
    std::string function = "main";  // multifunction packages name the function
    ComputeUnits units = ComputeUnits::all;  // resolved; loaders never read the environment
    const void * tmpl = nullptr;     // static embedded template; null for prebuilt
    std::vector<int64_t> geometry;   // what the loader validates, e.g. Parler {latent_dim, hop, rf_frames}
};

struct LoadResult {
    std::shared_ptr<void> model;  // the engine's MLModel wrapper; null on failure
    Error error;
    bool would_block = false;  // a key lock was busy: re-queue with backoff
    int lease_fd = -1;         // shared lease from adoption; closed with the model
    // A generated artifact was gone or incomplete when the load took its lease
    // (e.g. pruned after its preparation published it): the slot returns to
    // preparation, which regenerates it, instead of failing.
    bool artifact_gone = false;
    // Set by the LoadFn: the framework refused this generated artifact (it
    // did not load, or loaded with the wrong interface). The error keeps the
    // class the engine reports; end_load quarantines the artifact unless the
    // error is a resource one, and the slot regenerates it once.
    bool rejected = false;
    // The publication begin_load adopted (its directory's dev and inode),
    // taken under the shared lease: end_load quarantines only that one.
    uint64_t artifact_dev = 0;
    uint64_t artifact_ino = 0;
};

using LoadFn = LoadResult (*)(const LoadRequest & req);
using WarmFn = Error (*)(void * model, const LoadRequest & req);

struct SourceStat {
    std::string path;
    uint64_t size = 0;
    int64_t mtime_ns = 0;
    uint64_t ino = 0;
    uint64_t dev = 0;
    // The file at `path` is not certainly the one the engine loaded its
    // weights from (it was replaced during the load, or the engine generates
    // for another path): prepare fails with source_changed and generates
    // nothing. See generation::loaded_source.
    bool changed = false;
};

struct PrepareRequest {
    std::string key;  // deduplication key (artifact identity); "" -> never deduplicated
    std::string template_id;
    SourceStat source;
    std::string cache_dir;
    std::string options;  // e.g. "out=fp16"
    const void * tmpl = nullptr;
};

enum class PreparePhase { digesting, generating, compiling };

class PrepareContext {
public:
    virtual ~PrepareContext() = default;
    virtual void phase(PreparePhase p) = 0;
    // True when no live subscriber remains or the process is exiting. A started
    // job is never pre-empted; this only lets a generator skip optional work.
    virtual bool abandoned() const = 0;
};

struct PrepareResult {
    std::string artifact_path;  // the published .mlmodelc on success
    std::string artifact_key;
    bool cache_hit = false;     // status route: cached vs generated
    Error error;
    bool would_block = false;   // a key lock was busy: re-queue with backoff
    // Non-empty: generation cannot run here, for this status reason (e.g.
    // "cache_fs"). The slot reports unsupported(<reason>), not failed.
    std::string unsupported;
};

using PrepareFn = PrepareResult (*)(const PrepareRequest & req, PrepareContext & ctx);

// One loaded MLModel instance (per (artifact, function, units, geometry) when
// shareable; a prediction failure retires every instance of the (artifact,
// function, units), whatever its geometry).
class SharedModel {
public:
    SharedModel(const SharedModel &) = delete;
    SharedModel & operator=(const SharedModel &) = delete;
    ~SharedModel();

    void * get() const { return impl_.get(); }
    template <class T> T * as() const { return static_cast<T *>(impl_.get()); }

    // Every prediction holds this lock: one in-flight prediction per instance.
    std::unique_lock<std::mutex> lock_for_prediction() { return std::unique_lock<std::mutex>(predict_mu_); }

    // Prediction failure: retire this (artifact, function, units) process-wide.
    // Later acquisitions return null; a handle already acquired finishes or
    // fails its current unit.
    void retire(const std::string & reason);
    bool retired() const { return retired_.load(); }

    const LoadRequest & request() const { return req_; }
    const char * label() const { return label_for(req_.units); }

private:
    friend class detail::Runtime;
    friend class Handle;
    SharedModel(std::shared_ptr<void> impl, LoadRequest req, std::string key, int lease_fd,
                std::weak_ptr<detail::Runtime> rt);

    std::shared_ptr<void> impl_;
    std::mutex predict_mu_;
    LoadRequest req_;
    std::string key_;
    int lease_fd_ = -1;
    std::atomic<bool> retired_{false};
    std::weak_ptr<detail::Runtime> rt_;
};

// What a unit retains for its whole duration. While a Handle lives, the model
// counts as held: a resident cap never evicts it.
class Handle {
public:
    Handle() = default;
    explicit Handle(std::shared_ptr<SharedModel> m);
    Handle(const Handle &) = delete;
    Handle & operator=(const Handle &) = delete;
    Handle(Handle && o) noexcept : m_(std::move(o.m_)), ref_(std::move(o.ref_)) {}
    Handle & operator=(Handle && o) noexcept;
    ~Handle() { reset(); }

    void reset();
    explicit operator bool() const { return m_ != nullptr; }
    SharedModel * operator->() const { return m_.get(); }
    SharedModel & operator*() const { return *m_; }
    SharedModel * get() const { return m_.get(); }

private:
    friend class detail::Runtime;
    // Acquired through a member of a resident pool: counted in that pool while
    // this handle lives.
    Handle(std::shared_ptr<SharedModel> m, std::shared_ptr<detail::PoolRef> ref);

    std::shared_ptr<SharedModel> m_;
    std::shared_ptr<detail::PoolRef> ref_;
};

struct Acquired {
    Handle model;              // null -> run this unit on ggml with today's plan
    bool pending = false;      // ggml only because the slot was not ready yet (coreml_pending)
    SlotState state = SlotState::idle;  // the slot's state when acquisition returned

    explicit operator bool() const { return static_cast<bool>(model); }
    // coreml-* when a model was acquired, otherwise "ggml".
    std::string label() const { return model ? model->label() : "ggml"; }
};

// Thrown by strict acquisitions: a terminal non-ready state, a timeout,
// deferred(memory), or a stage that cannot serve the unit (resident cap).
class CoremlError : public std::runtime_error {
public:
    CoremlError(const std::string & what, SlotState state, ErrorClass cls)
        : std::runtime_error(what), state(state), cls(cls) {}
    SlotState state;
    ErrorClass cls;
};

class ResidentPool;

struct SlotSpec {
    std::string engine;  // status "engine"
    std::string stage;   // e.g. "cfm"
    std::string function = "main";
    ComputeUnits units = ComputeUnits::all;  // resolved
    Route route = Route::none;               // from plan()
    SlotState none_state = SlotState::unsupported;  // used when route == none
    std::string none_reason;
    std::string artifact_path;  // route prebuilt: today's adjacent .mlmodelc
    PrepareRequest prepare;     // route generate
    PrepareFn prepare_fn = nullptr;
    LoadFn load_fn = nullptr;
    WarmFn warm_fn = nullptr;
    const void * tmpl = nullptr;
    // What the LoadFn validates. Part of the model's identity: owners of one
    // artifact share a ready model or a running load only with equal geometry.
    std::vector<int64_t> geometry;
    bool shareable = true;                   // false: stateful template, one model per owner
    uint64_t ane_compile_reserve_bytes = 0;  // memory admission for cold ANE loads; 0 = never deferred
    std::optional<int64_t> expected_warm_load_ms;  // template.json, when no warm load was observed
    std::shared_ptr<ResidentPool> pool;      // engines that cap resident functions (Chatterbox)
};

struct SlotPolicy {
    Wait wait = Wait::none;  // acquisitions: none (bounded wait) | call
    bool strict = false;
    std::optional<int64_t> strict_wait_ms;  // nullopt -> unbounded
    std::string cache_dir;                  // hints live under it; "" -> no hints

    static SlotPolicy from(const Resolved & r);
};

class Slot {
public:
    Slot(const Slot &) = delete;
    Slot & operator=(const Slot &) = delete;
    // Unsubscribes and releases the slot's model. Never waits.
    ~Slot();

    // Request a load (or prepare + load) at `p`; raises an existing request.
    void request(Priority p);

    // Acquire once at the start of a unit of work and hold the result for the
    // whole unit. With Demand::now the slot is raised to `now`; then:
    //  - wait none/construct: the bounded wait against `deadline` (the
    //    unit's single aggregate deadline), otherwise no blocking;
    //  - wait call: block until ready or terminal; deferred(memory) and a
    //    resident cap with nothing evictable (the caller may hold every
    //    instance) fall back at once, the load staying queued;
    //  - strict: like call, but throw CoremlError on a terminal non-ready
    //    state, deferred(memory), a resident-cap denial or strict_wait_ms.
    // `cancel` interrupts waits; it is polled every 50 ms.
    Acquired acquire(Demand d, std::optional<Clock::time_point> deadline = std::nullopt,
                     const std::atomic<bool> * cancel = nullptr);

    // Release the model, bump the epoch and cancel unstarted jobs. A started
    // job runs on, writes its hint, and never repopulates this slot; a later
    // request subscribes to it instead of starting a second load.
    void evict();

    // Retire after a prediction failure (process-wide for the model's key).
    void retire(const std::string & reason);

    SlotState state() const;
    SlotStatus status() const;
    std::optional<HintView> hint() const;
    // mtime of the hint record in nanoseconds (0 without one): "most recent".
    int64_t hint_mtime_ns() const;
    std::string hint_path() const;
    const SlotSpec & spec() const;

private:
    friend class Manager;
    friend class ResidentPool;
    explicit Slot(std::shared_ptr<detail::SlotCore> core) : core_(std::move(core)) {}
    std::shared_ptr<detail::SlotCore> core_;
};

// Cap on live model instances for one engine stage (the resident cap): ready
// models, handles still retained by a unit, loads in flight and evicted
// instances whose release has not completed yet. A replacement
// load is admitted only after an eviction made room; a held function or one
// predicted for the current call is never evicted.
class ResidentPool {
public:
    size_t cap() const;
    // The current call's predictions (replaces the previous set).
    void set_predicted(const std::vector<std::shared_ptr<Slot>> & slots);
    void clear_predicted();
    size_t live() const;

private:
    friend class Manager;
    friend class detail::Runtime;
    ResidentPool(std::shared_ptr<detail::PoolCore> core, std::shared_ptr<detail::Runtime> rt)
        : core_(std::move(core)), rt_(std::move(rt)) {}
    std::shared_ptr<detail::PoolCore> core_;
    std::shared_ptr<detail::Runtime> rt_;
};

}  // namespace QVAC_COREML_NS
