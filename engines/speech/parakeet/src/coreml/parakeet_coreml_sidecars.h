#pragma once

// Parakeet's adoption of the shared Core ML runtime (coreml).
//
// Every background-loaded sidecar of one model gets a slot: the encoder, the
// Sortformer bypass encoder, and one per diarizer capacity. Slots load on the
// runtime's load lane, in priority order, with memory admission for cold ANE
// compiles; until a slot is ready its stage runs on ggml. A stage with no
// adjacent sidecar whose embedded template matches the GGUF is generated
// into the cache directory first (route `generate`, prepare queue). The TDT
// decoder bundle and an encoder sidecar that owns the encoder (an encoder-less
// GGUF, or PARAKEET_COREML_ONLY when the sidecar may own it) stay synchronous
// and are not slots. A standard TDT sidecar never owns the encoder, so with
// PARAKEET_COREML_ONLY and a full GGUF it is an ordinary slot.
//
// Portable C++ against the runtime's namespace (QVAC_COREML_NS): no Core ML or
// Objective-C here, so the routing glue builds and is tested on every
// platform with fake loaders.

#include "coreml_fit.h"
#include "coreml/parakeet_coreml_load_error.h"
#include "encoder_route.h"
#include "parakeet/coreml_options.h"
#include "qvac_coreml/generation.h"
#include "qvac_coreml/manager.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace parakeet {

namespace pcm = QVAC_COREML_NS;

// Public options -> the runtime's internal mirror (field by field).
pcm::Options to_runtime_options(const CoremlOptions & opts);

// Resolves once with Parakeet's environment prefix (PARAKEET_COREML_*).
pcm::Resolved resolve_parakeet_coreml(const CoremlOptions & opts);

pcm::Prewarm to_runtime_prewarm(CoremlOptions::Prewarm scope);

// The resolved compute units as the argument the synchronous loaders take
// (the TDT decoder bundle and the encoder owner): their status name, or
// null when nothing was set, so the loader keeps its own stage default.
// Loaders never read the environment themselves.
inline const char * loader_units(const pcm::Resolved & resolved) {
    return resolved.compute_units ? pcm::to_string(*resolved.compute_units) : nullptr;
}

// Operations on a loaded encoder model (the SharedModel payload). Plain
// function pointers, so tests substitute fakes for the Core ML calls.
struct EncoderModelOps {
    parakeet_coreml_context * (*context)(void * model) = nullptr;
    EncoderRouteGeometry (*describe)(parakeet_coreml_context * ctx, int n_mels) = nullptr;
    const char * (*label)(parakeet_coreml_context * ctx) = nullptr;
    EncoderRoute::EncodeFn encode = nullptr;
};

// The unit's route from one acquisition: a handle gives a Core ML route whose
// failure retires the model process-wide; no handle gives the ggml route,
// marked pending when the slot was not ready yet. `strict` (use = require):
// a dispatch the handle cannot serve throws pcm::CoremlError instead of
// running on ggml.
EncoderRoute route_from_acquired(pcm::Acquired && acquired, const EncoderModelOps & ops, int n_mels,
                                 bool strict = false);

// A sidecar load failure as the runtime's error. A framework failure (the
// model did not load: a corrupt or unreadable bundle, a compile error, memory
// pressure) is transient and keeps the framework's domain and code, so a
// repeat can be recognised; only a verified incompatibility (the model loaded
// but its interface or capacity is not the one expected) is deterministic.
pcm::Error sidecar_load_error(const std::string & what, const parakeet_coreml_load_error & error);

// The slot's model if it is ready now, for readiness probes (accessors,
// capacity lookups, the ggml prewarm pass). It never requests, waits or
// throws: a strict slot that failed is reported by the acquisition a unit of
// inference makes, not by a probe.
pcm::Acquired acquire_if_ready(pcm::Slot & slot);

// Bytes ANECompilerService needs for a cold compile of `mlmodelc` (by
// default 1.6 x weights/weight.bin), or 0 when `units` cannot use the ANE.
uint64_t ane_compile_reserve_bytes(const std::string & mlmodelc, pcm::ComputeUnits units);

// The loader that opens a sidecar, first in its geometry: an encoder, bypass
// encoder and diarizer bundle at one path are never one shared model.
enum class SidecarLoader : int64_t { encoder = 1, bypass_encoder = 2, diarizer = 3 };

inline std::vector<int64_t> sidecar_geometry(SidecarLoader loader, std::vector<int64_t> rest = {}) {
    rest.insert(rest.begin(), static_cast<int64_t>(loader));
    return rest;
}

// One sidecar to register (route from plan(): prebuilt when present).
struct SidecarRequest {
    std::string stage;          // status "stage": encoder, bypass-encoder, diarizer-<frames>
    std::string path;           // adjacent .mlmodelc (made absolute and canonical by add())
    bool present = false;       // the directory exists (today's path rules)
    pcm::ComputeUnits stage_default = pcm::ComputeUnits::all;
    pcm::LoadFn load_fn = nullptr;
    // Validated by the loader and part of the runtime's sharing key: the
    // loader kind first (sidecar_geometry), then e.g. the diarizer capacity,
    // so two owners share a model only when the same loader would accept it.
    std::vector<int64_t> geometry;
    bool minimal = false;       // part of the minimal (construction) set
    // Route generate: the stage's template as probed against the GGUF (null:
    // prebuilt only) and the GGUF's stat tuple, taken when the model loaded.
    const pcm::generation::Candidate * generated = nullptr;
    pcm::SourceStat source;
};

// The sidecar slots of one loaded model.
class CoremlSidecars {
public:
    // `manager` null uses the library image's manager. `memory_degraded`:
    // the Core ML-inclusive fit failed where the ggml-only one passed, so
    // the instance resolves use = off with the reason "memory".
    explicit CoremlSidecars(const CoremlOptions & opts,
                            std::shared_ptr<pcm::Manager> manager = nullptr,
                            bool memory_degraded = false);
    CoremlSidecars(const CoremlSidecars &) = delete;
    CoremlSidecars & operator=(const CoremlSidecars &) = delete;
    // Drops the slots; never waits.
    ~CoremlSidecars();

    const pcm::Resolved & resolved() const { return resolved_; }
    pcm::Manager & manager() const { return *manager_; }

    // The route plan() gives `request`: `prebuilt` when the bundle is present
    // and Core ML is enabled; `generate` when there is none and its matched
    // template can be generated (generator built, cache directory set,
    // generate != never, macOS 15, the resolved units validated); otherwise
    // `none` (disabled, absent, no_template, unsupported).
    pcm::Plan plan(const SidecarRequest & request) const;

    // Registers a slot with the route plan() gives it; a `none` route is
    // reported in status and listed as skipped in groups.
    std::shared_ptr<pcm::Slot> add(const SidecarRequest & request);
    // The same with `plan` = plan(request), when the caller has planned it.
    std::shared_ptr<pcm::Slot> add(const SidecarRequest & request, const pcm::Plan & plan);

    // Whether this instance can generate sidecars, and why not (status
    // "generation"). The reason is "unvalidated_units" once every generated
    // route registered so far was suppressed for the resolved units.
    pcm::GenerationStatus generation_status() const;

    // Issues the construction group for the resolved prewarm scope; with
    // wait = construct, blocks until it finished. Call once, after every add().
    void start();

    // Another prewarm group (Engine::coreml_prewarm).
    pcm::PrewarmResult prewarm(CoremlOptions::Prewarm scope, bool wait,
                               const std::atomic<bool> * cancel = nullptr);

    // Waits for the construction group (timeout < 0: no limit). True when
    // every slot in it reached ready.
    bool wait_construction(int64_t timeout_ms, const std::atomic<bool> * cancel = nullptr);
    std::string construction_json() const;
    std::string status_json() const;

    std::vector<std::shared_ptr<pcm::Slot>> slots() const;

private:
    struct Entry {
        std::shared_ptr<pcm::Slot> slot;
        bool minimal = false;
    };
    std::vector<pcm::GroupRequest> requests(pcm::Prewarm scope, pcm::GroupKind kind,
                                            std::vector<pcm::SlotStatus> & skipped) const;

    pcm::Resolved resolved_;
    std::shared_ptr<pcm::Manager> manager_;
    std::vector<Entry> entries_;
    // Registered template candidates with a generated route: planned
    // `generate`, or suppressed for the resolved units.
    int generated_routes_ = 0;
    int unvalidated_routes_ = 0;
    std::shared_ptr<pcm::Group> construction_;
};

// ---- Diarizer capacities -------------------------------------------------

// One diarizer capacity of a model: an adjacent prebuilt bundle, or a
// template the runtime can generate the bundle from (a prebuilt bundle wins).
struct DiarizerCapacity {
    int frames = 0;
    bool prebuilt = false;
    // The template probed against the GGUF (template capacities only). Read
    // only while the capacity is planned or registered.
    const pcm::generation::Candidate * generated = nullptr;
};

// The diarizer capacities of one model, ascending: every adjacent prebuilt
// bundle (`prebuilt`), and every template capacity with no adjacent bundle
// (`template_frames[i]`, probed as `probed[i]`). Discovery registers each of
// them, and fit charges the template capacities that plan() routes to
// generate (diarizer_fit_charges); both start from this list and plan the
// same diarizer_request(), so they always choose the same capacities.
std::vector<DiarizerCapacity> diarizer_capacities(const std::vector<int> & prebuilt,
                                                  const std::vector<int> & template_frames,
                                                  const std::vector<pcm::generation::Candidate> & probed);

// The templates `template_ids` probed against `gguf_path`. With Core ML
// disabled nothing is read: every candidate is unavailable, and plan() reports
// the capacity disabled before it looks at the template.
std::vector<pcm::generation::Candidate> probe_diarizer_templates(const CoremlSidecars & sidecars,
                                                                 const std::string & gguf_path,
                                                                 const std::vector<std::string> & template_ids);

// The sidecar request of one diarizer capacity: stage diarizer-<frames>, the
// adjacent bundle path, the stage default cpu_and_ane, the capacity as the
// loader's geometry. `source` is the GGUF's stat tuple (route generate).
SidecarRequest diarizer_request(const std::string & gguf_path, const DiarizerCapacity & c, pcm::LoadFn load_fn,
                                const pcm::SourceStat & source);

// Fit: one charge per capacity of `capacities`. A template capacity that plan()
// routes to generate carries its template's unmargined resident and
// transient parts for the resolved units; every other capacity is listed
// uncharged, because a prebuilt one still shapes which capacities a range
// loads (coreml_generated_diarizer_envelope).
std::vector<CoremlDiarizerCharge> diarizer_fit_charges(const CoremlSidecars & sidecars, const std::string & gguf_path,
                                                       const std::vector<DiarizerCapacity> & capacities);

// Whether a diarization model's encoder and bypass-encoder sidecars are in
// the minimal (construction) set: only when no diarizer capacity can serve a
// pass (`serving_diarizer_capacities`, DiarizerSlots::capacities()). A
// template capacity that is only status-only (no cache directory, generate =
// never, an unsupported OS or unvalidated units, a GGUF the template does not
// match) serves nothing, so the adjacent prebuilt encoders keep loading at
// construction as they do with no diarizer at all.
inline bool encoder_sidecars_minimal(const std::vector<int> & serving_diarizer_capacities) {
    return serving_diarizer_capacities.empty();
}

// DiarizerSlots::run's status when the pass runs on ggml.
constexpr int kNoDiarizerSidecar = -1;

// The outcome of one diarizer pass.
struct DiarizerPass {
    // 0: ran on Core ML; kNoDiarizerSidecar: ggml serves the pass; other
    // values: the prediction failed and its capacity was retired.
    int status = kNoDiarizerSidecar;
    bool not_ready = false;  // ggml only because the covering capacity was still loading
};

// The runtime slots of one model's diarizer capacities, and which capacity
// serves a pass. Every capacity gets a slot; a template capacity whose route
// is not generate gets a status-only slot (no_template, unsupported(...),
// disabled) that status and the construction group list but that never
// serves a pass. A pass uses the smallest covering capacity that can serve
// it; a load or prediction failure retires that capacity for the process.
// Under use = require a pass that covering capacities cannot serve throws
// pcm::CoremlError: every covering capacity failed or is status-only, a
// timeout, or a failed prediction, and so does a pass longer than every
// capacity (under use = automatic that pass runs on ggml).
class DiarizerSlots {
public:
    // Runs one prediction on a ready model of `capacity` frames; 0 on success.
    using Predict = std::function<int(pcm::Handle & model, int capacity)>;

    // Registers one slot per capacity. The smallest capacity that can serve
    // a pass and every status-only capacity are in the minimal (construction)
    // set; the others are requested by the pass that needs them, or by
    // prewarm = all. `source` is the identity of the GGUF whose weights the
    // model loaded (generation::source_stat_fd of the descriptor it read
    // through): generated capacities prepare only from that file. Without it
    // the path is stat'ed now.
    DiarizerSlots(const std::string & gguf_path, std::vector<DiarizerCapacity> capacities, CoremlSidecars & sidecars,
                  pcm::LoadFn load_fn, std::optional<pcm::SourceStat> source = std::nullopt);

    // A capacity that has not failed covers `frames` (independent of
    // readiness). Under use = require every capacity covers, so run() refuses
    // the pass rather than the caller falling back to ggml.
    bool covers(int frames) const;
    // Requests the covering capacity now; true when it is ready.
    bool prepare(int frames) const;
    // Runs `predict` on the smallest covering capacity that is ready (see the
    // class comment for ggml and use = require). `cancel` (the owning
    // engine's flag, may be null) stops a wait for a capacity.
    DiarizerPass run(int frames, const std::atomic<bool> * cancel, const Predict & predict) const;

    bool strict() const { return strict_; }
    // The capacities that can serve a pass: prebuilt, or route generate.
    std::vector<int> capacities() const;
    // The status-only slots of template capacities that are not generated.
    std::vector<std::shared_ptr<pcm::Slot>> status_only() const;

private:
    struct Capacity {
        int frames = 0;
        std::shared_ptr<pcm::Slot> slot;
        bool serves = false;  // prebuilt or route generate
    };
    // use = require: a pass longer than every capacity is refused (throws),
    // like any stage too wide for its sidecar, before a caller falls back to
    // ggml.
    void refuse_too_wide(int frames) const;
    bool strict_ = false;
    std::vector<Capacity> capacities_;  // ascending
};

}  // namespace parakeet
