#pragma once

// The per-library-image Core ML manager: scheduler, model registry, request
// groups and route planning.
//
// Scheduler (one per manager):
//  - a prepare queue (1 thread, QOS_CLASS_UTILITY) and a load lane (1 thread),
//    both priority-ordered (now > current > minimal > prewarm, then FIFO);
//  - an admission gate: one Core ML job (prepare, or load + warm) at a time;
//  - a job whose key lock is busy (would_block) releases the gate and re-queues
//    with backoff (1 s doubling to 30 s) while other jobs run;
//  - memory admission: a load whose units include the ANE and that has no
//    non-stale hint waits, reported as deferred(memory), until the available
//    memory reading reaches the slot's ane_compile_reserve_bytes; deferred jobs
//    hold neither the gate nor the lane;
//  - low power: `prewarm` work is deferred unless a waited group needs it.
//
// "Process-wide" means per engine library image: each engine library compiles
// its own copy of this runtime in its own namespace. Manager::instance() is
// leaked on purpose. The image pins itself (dladdr + dlopen RTLD_NOLOAD | RTLD_NODELETE)
// so code and callbacks outlive any dlclose: at its first request for work,
// when the main thread makes a slot that can load, or when arm_exit_guard()
// is called on the main thread. Process exit is described with the exit hooks at the end of this
// header and in the README ("Process exit").

#include "qvac_coreml/options.h"
#include "qvac_coreml/slot.h"
#include "qvac_coreml/status.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace QVAC_COREML_NS {

namespace detail {
struct Graveyard;
}  // namespace detail

struct Config {
    // Device facts; null/empty reads the device. Tests inject fakes.
    uint64_t (*available_memory)() = nullptr;
    bool (*low_power)() = nullptr;
    std::string host_id;
    std::string os_build;
    std::optional<Platform> platform;  // tests: plan() as on that platform
    std::optional<OsVersion> os;

    std::chrono::milliseconds memory_recheck{5000};  // deferred(memory) re-check period
    std::chrono::milliseconds backoff_initial{1000}; // would_block re-queue backoff
    std::chrono::milliseconds backoff_max{30000};
    std::chrono::milliseconds wait_poll{50};         // cancellation polling of waits
    bool pin_image = true;
};

struct Stats {
    uint64_t loads_started = 0;
    uint64_t loads_completed = 0;
    uint64_t load_failures = 0;
    uint64_t prepares_started = 0;
    uint64_t prepares_completed = 0;
    uint64_t requeues = 0;
    uint64_t cancelled = 0;
    uint64_t evictions = 0;
    uint64_t hint_writes = 0;
};

// ---- Route planning ---------------------------------

struct PlanInput {
    bool coreml_built = false;        // <E>_COREML compiled in
    bool generation_built = false;    // the build embeds templates (<E>_COREML_GENERATE)
    bool prebuilt_present = false;    // adjacent <stem><suffix>.mlmodelc by today's path rules
    bool template_available = false;  // the library embeds a template for this stage
    bool template_matched = false;    // template.json.match accepted the GGUF
    bool multifunction_ane = false;   // template needs macOS 15.7 (F11)
    bool device_excluded = false;     // template prefer_ggml_on matched this device
    std::vector<ComputeUnits> validated_units;  // template.json memory.validated_for units
    ComputeUnits units = ComputeUnits::all;     // resolved units for this stage
    Platform platform = Platform::other;
    OsVersion os;
};

// plan()'s reason for a generated route whose resolved units the template's
// memory envelope was not validated for (state unsupported(unvalidated_units)).
inline constexpr char kUnvalidatedUnits[] = "unvalidated_units";

struct Plan {
    Route route = Route::none;
    SlotState none_state = SlotState::unsupported;  // when route == none
    std::string reason;
};

// Decides from policy, OS, build, device class, template match and adjacent
// prebuilt presence; never reads the cache, so fit is identical before and
// after the first load. A generated route whose resolved units are not in
// validated_units is suppressed as unsupported(unvalidated_units); prebuilt
// routes are never suppressed by validation.
Plan plan(const PlanInput & in, const Resolved & resolved);

// ---- Request groups -----------------------------------------------

enum class GroupKind { construction, prewarm };

const char * to_string(GroupKind k);

struct GroupRequest {
    std::shared_ptr<Slot> slot;
    Priority priority = Priority::minimal;
};

class Group {
public:
    // Every slot is terminal, deferred(memory), idle (cancelled) or reached
    // `ready` since the group was issued.
    bool finished() const;
    // Blocks until finished, `timeout` or `cancel` (polled every 50 ms).
    PrewarmResult wait(std::optional<Clock::duration> timeout = std::nullopt,
                       const std::atomic<bool> * cancel = nullptr);
    PrewarmResult result() const;

private:
    friend class Manager;
    struct Entry {
        std::weak_ptr<detail::SlotCore> core;
        SlotStatus last;       // last observed status (ready snapshot once it reached ready)
        uint64_t baseline = 0; // ready events at issue
        bool reached_ready = false;
    };
    // Under the runtime mutex; strong references go to the caller's graveyard.
    PrewarmResult collect(bool & finished, detail::Graveyard & gy) const;

    GroupKind kind_ = GroupKind::construction;
    std::shared_ptr<detail::Runtime> rt_;
    mutable std::vector<Entry> entries_;
    std::vector<SlotStatus> skipped_;
};

class Manager {
public:
    Manager(const Manager &) = delete;
    Manager & operator=(const Manager &) = delete;
    // Stops this manager's threads. The per-image instance() is never destroyed.
    ~Manager();

    // The per-library-image manager (leaked on purpose).
    static std::shared_ptr<Manager> instance();
    // An independent manager (tests; "the next process").
    static std::shared_ptr<Manager> create(Config cfg = {});

    std::shared_ptr<Slot> make_slot(SlotSpec spec, SlotPolicy policy);
    std::shared_ptr<ResidentPool> make_pool(size_t cap);

    // Issues a group: requests every slot at its priority. `waited` marks the
    // group as explicitly waited on (wait:'construct' or coreml_prewarm(...,
    // wait=true)), which makes its prewarm work non-deferrable in low power.
    // `skipped_route_none` lists slots whose plan() route is none.
    std::shared_ptr<Group> issue_group(GroupKind kind, const std::vector<GroupRequest> & requests, bool waited,
                                       std::vector<SlotStatus> skipped_route_none = {});

    // Re-run admission now (a memory reading or the low-power state changed).
    void notify();

    EngineStatus status(const std::string & engine, const std::string & cache_dir,
                        const std::vector<std::shared_ptr<Slot>> & slots, GenerationStatus generation) const;

    const std::string & host_id() const;
    const std::string & os_build() const;
    OsVersion os_version() const;
    Platform platform() const;

    // Test introspection.
    Stats stats() const;
    std::vector<std::string> events() const;

private:
    explicit Manager(std::shared_ptr<detail::Runtime> rt) : rt_(std::move(rt)) {}
    std::shared_ptr<detail::Runtime> rt_;
};

// ---- process exit -------------------------------------------------------------
//
// A Core ML compile or load cannot be cancelled, and exit() destroys framework
// state that a running one still reads (some of it built lazily during the
// first compile, so its destructors are among the first ones exit() runs).
// Each engine library (each has its own copy of this runtime) therefore has
// one bounded exit wait for the job its worker is running and the models it
// is releasing. The automatic default bound is 15 s on macOS and 2 s on iOS.
// Admission closes when the wait starts: a job not yet started never starts,
// and a slot without a model or a running job fails with kExitingReason (a
// strict acquisition throws, also one already waiting). The wait runs:
//  - from the exit guard: a thread_local object of the main thread. When the
//    main thread exits the process, its destructor runs before every static
//    destructor and atexit handler (returning from main first destroys main's
//    automatic objects, and thread_local objects of the main thread built
//    after the guard are destroyed before it). It is armed automatically when
//    the main thread makes a slot that can load or requests work, or by
//    arm_exit_guard();
//  - from shutdown_for_exit(), which a host calls before process
//    finalization begins.
// When exit starts with neither (an engine made and loaded off the main
// thread with nothing armed, or exit() called on another thread), an atexit
// handler only closes admission: a job still running can then crash in the
// frameworks, as it could before. The same holds past the deadline. Nothing
// waits after that handler ran: hosts must not drain from atexit handlers or
// static destructors, which run after the frameworks' lazily registered
// destructors (and some may run before this runtime's handler).

// The reason of a slot that had no model and no running job when process
// exit began (after shutdown_for_exit(), or when an armed main thread ended).
inline constexpr const char * kExitingReason = "the process is exiting: no new Core ML work starts";

// Arms this library's exit guard. Call it on the main thread (for a JS host,
// the JS thread when that is the process's main thread), e.g. where the addon
// instance is created; it pins this library image first. Returns true when
// the guard is armed, now or earlier. Returns false, pinning and registering
// nothing, on any other thread and once process finalization has begun; also
// when the image cannot be pinned. A guard armed this way waits even after
// set_automatic_exit_guard(false).
bool arm_exit_guard();

// Terminal: closes admission in this library for the rest of the process,
// for every user of it (the image stays pinned, so reloading an addon does
// not reopen it), then waits for the job a worker is running. The wait ends
// at the earlier of now + `bound` (clamped to one hour) and the library's one
// exit deadline, which the first call with a positive bound sets (the exit
// guard's wait shares it). A zero or negative bound only closes admission and
// reports, and fixes no deadline. Returns true when no job is running; false
// on timeout, and always when called from inside a job (which does not wait
// for itself). Call it from any thread right before the process exits, while
// its main queue can still make progress; not on an addon unload in a process
// that goes on, and not from an atexit handler or a static destructor (after
// finalization began it only reports).
bool shutdown_for_exit(std::chrono::milliseconds bound);

// Per engine library. With false, an automatically armed guard does not wait,
// whenever the call is made (a guard armed by arm_exit_guard() still does).
// For hosts whose main thread may end with pthread_exit() while the process
// goes on (dispatch_main() does that): on Apple platforms an armed guard then
// runs and closes admission for the rest of the process (glibc does not run
// the main thread's thread_local destructors there). Such hosts call
// shutdown_for_exit() right before they exit.
void set_automatic_exit_guard(bool on);

}  // namespace QVAC_COREML_NS
