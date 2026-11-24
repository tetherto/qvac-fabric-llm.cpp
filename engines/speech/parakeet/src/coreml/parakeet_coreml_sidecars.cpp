#include "coreml/parakeet_coreml_sidecars.h"

#include "parakeet_coreml_path.h"

#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <optional>
#include <filesystem>
#include <system_error>
#include <utility>

namespace parakeet {

namespace {

constexpr const char * kEngine = "parakeet";
constexpr const char * kEnvPrefix = "PARAKEET";
constexpr const char * kWeightsFile = "/weights/weight.bin";
// Default ane_compile_reserve_bytes: ANECompilerService peaked at about
// 1.6x the weights compiling a 199 MB model.
constexpr double kAneCompileReserveFactor = 1.6;

uint64_t file_bytes(const std::string & path) {
    struct stat st {};
    return ::stat(path.c_str(), &st) == 0 ? static_cast<uint64_t>(st.st_size) : 0;
}

// The bundle path a slot keys its shared model by and loads from: absolute
// and canonical, resolved once at registration. A relative path would name
// a different bundle after the process changes directory, and two models
// opened by the same relative name from different directories would share
// one model.
std::string canonical_bundle_path(const std::string & path) {
    if (path.empty()) return path;
    std::error_code ec;
    std::filesystem::path p = std::filesystem::weakly_canonical(path, ec);
    if (ec) p = std::filesystem::absolute(path, ec).lexically_normal();
    return ec ? path : p.string();
}

template <class E, class F>
F map_enum(E v) {
    return static_cast<F>(static_cast<int>(v));
}

void retire_handle(void * hold, const char * reason) {
    auto * handle = static_cast<pcm::Handle *>(hold);
    if (handle != nullptr && *handle) (*handle)->retire(reason != nullptr ? reason : "prediction failed");
}

std::unique_lock<std::mutex> lock_handle(void * hold) {
    auto * handle = static_cast<pcm::Handle *>(hold);
    if (handle == nullptr || !*handle) return {};
    return (*handle)->lock_for_prediction();
}

[[noreturn]] void refuse_strict(void *, const std::string & reason, bool prediction_failed) {
    if (prediction_failed) {
        throw pcm::CoremlError(reason, pcm::SlotState::retired, pcm::ErrorClass::transient);
    }
    throw pcm::CoremlError(reason, pcm::SlotState::ready, pcm::ErrorClass::deterministic);
}

}  // namespace

pcm::Options to_runtime_options(const CoremlOptions & o) {
    // The public enums mirror the runtime's member for member.
    static_assert(static_cast<int>(CoremlOptions::Use::require) == static_cast<int>(pcm::Use::require), "Use");
    static_assert(static_cast<int>(CoremlOptions::Generate::background) ==
                      static_cast<int>(pcm::Generate::background), "Generate");
    static_assert(static_cast<int>(CoremlOptions::Wait::call) == static_cast<int>(pcm::Wait::call), "Wait");
    static_assert(static_cast<int>(CoremlOptions::Prewarm::all) == static_cast<int>(pcm::Prewarm::all), "Prewarm");
    pcm::Options r;
    r.use = map_enum<CoremlOptions::Use, pcm::Use>(o.use);
    r.generate = map_enum<CoremlOptions::Generate, pcm::Generate>(o.generate);
    r.wait = map_enum<CoremlOptions::Wait, pcm::Wait>(o.wait);
    r.prewarm = map_enum<CoremlOptions::Prewarm, pcm::Prewarm>(o.prewarm);
    r.cache_dir = o.cache_dir;
    r.compute_units = o.compute_units;
    r.strict_wait_ms = o.strict_wait_ms;
    r.memory_budget_bytes = o.memory_budget_bytes;
    return r;
}

pcm::Resolved resolve_parakeet_coreml(const CoremlOptions & opts) {
    return pcm::resolve_coreml(to_runtime_options(opts), kEnvPrefix);
}

pcm::Prewarm to_runtime_prewarm(CoremlOptions::Prewarm scope) {
    return map_enum<CoremlOptions::Prewarm, pcm::Prewarm>(scope);
}

EncoderRoute route_from_acquired(pcm::Acquired && acquired, const EncoderModelOps & ops, int n_mels,
                                 bool strict) {
    if (!acquired.model || ops.context == nullptr) return EncoderRoute::ggml(acquired.pending);
    parakeet_coreml_context * ctx = ops.context(acquired.model->get());
    if (ctx == nullptr) return EncoderRoute::ggml(acquired.pending);
    const EncoderRouteGeometry geometry = ops.describe != nullptr ? ops.describe(ctx, n_mels) : EncoderRouteGeometry{};
    const char * label = ops.label != nullptr ? ops.label(ctx) : nullptr;
    auto hold = std::make_shared<pcm::Handle>(std::move(acquired.model));
    EncoderRoute::Ops route_ops;
    route_ops.retire = &retire_handle;
    route_ops.lock = &lock_handle;
    route_ops.encode = ops.encode;
    route_ops.refuse = &refuse_strict;
    return EncoderRoute::coreml(ctx, geometry, label != nullptr ? label : "coreml", std::move(hold),
                                route_ops, /*owner=*/false, strict);
}

pcm::Error sidecar_load_error(const std::string & what, const parakeet_coreml_load_error & error) {
    std::string message = "failed to load the " + what;
    if (error.domain[0] != '\0') message += " (" + std::string(error.domain) + " " + std::to_string(error.code) + ")";
    if (error.message[0] != '\0') message += ": " + std::string(error.message);
    if (error.incompatible != 0) return pcm::Error::deterministic(message, "parakeet", 0);
    return pcm::Error::transient(message, error.domain, error.code);
}

pcm::Acquired acquire_if_ready(pcm::Slot & slot) {
    if (slot.state() != pcm::SlotState::ready) return {};
    try {
        return slot.acquire(pcm::Demand::if_ready);
    } catch (const pcm::CoremlError &) {
        return {};  // retired between the state read and the acquisition
    }
}

uint64_t ane_compile_reserve_bytes(const std::string & mlmodelc, pcm::ComputeUnits units) {
    if (!pcm::units_include_ane(units)) return 0;
    return static_cast<uint64_t>(kAneCompileReserveFactor *
                                 static_cast<double>(file_bytes(mlmodelc + kWeightsFile)));
}

CoremlSidecars::CoremlSidecars(const CoremlOptions & opts, std::shared_ptr<pcm::Manager> manager,
                               bool memory_degraded)
    : resolved_(resolve_parakeet_coreml(opts)),
      manager_(manager ? std::move(manager) : pcm::Manager::instance()) {
    if (memory_degraded) {
        pcm::FitVerdict verdict;
        verdict.ggml_only_fits = true;
        verdict.coreml_fits = false;
        pcm::apply_memory_degrade(resolved_, verdict);
    }
}

CoremlSidecars::~CoremlSidecars() = default;

namespace {

pcm::PlanInput plan_input(const SidecarRequest & request, const pcm::Resolved & resolved, const pcm::Manager & m) {
    pcm::PlanInput in;
    in.coreml_built = true;
    in.generation_built = pcm::generation::built();
    in.prebuilt_present = request.present;
    if (request.generated != nullptr) {
        in.template_available = request.generated->available;
        in.template_matched = request.generated->matched;
        in.validated_units = request.generated->memory.validated_units;
    }
    in.units = resolved.units_or(request.stage_default);
    in.platform = m.platform();
    in.os = m.os_version();
    return in;
}

}  // namespace

pcm::Plan CoremlSidecars::plan(const SidecarRequest & request) const {
    return pcm::plan(plan_input(request, resolved_, *manager_), resolved_);
}

std::shared_ptr<pcm::Slot> CoremlSidecars::add(const SidecarRequest & request) { return add(request, plan(request)); }

std::shared_ptr<pcm::Slot> CoremlSidecars::add(const SidecarRequest & request, const pcm::Plan & plan) {
    const pcm::PlanInput in = plan_input(request, resolved_, *manager_);

    pcm::SlotSpec spec;
    spec.engine = kEngine;
    spec.stage = request.stage;
    spec.units = in.units;
    spec.route = plan.route;
    spec.none_state = plan.none_state;
    spec.none_reason = plan.reason;
    spec.load_fn = request.load_fn;
    spec.geometry = request.geometry;
    if (plan.route == pcm::Route::generate) {
        // The artifact path comes from the prepare job (the published bundle).
        spec.prepare = pcm::generation::prepare_request(*request.generated, request.source, resolved_);
        spec.prepare_fn = &pcm::generation::prepare;
        spec.tmpl = request.generated->tmpl;
        spec.ane_compile_reserve_bytes = pcm::generation::ane_compile_reserve_bytes(request.generated->memory, in.units);
    } else {
        spec.artifact_path = canonical_bundle_path(request.path);
        spec.ane_compile_reserve_bytes =
            plan.route == pcm::Route::prebuilt ? ane_compile_reserve_bytes(spec.artifact_path, in.units) : 0;
    }
    if (request.generated != nullptr && !request.present) {
        const bool unvalidated = plan.route == pcm::Route::none && plan.reason == pcm::kUnvalidatedUnits;
        if (plan.route == pcm::Route::generate || unvalidated) ++generated_routes_;
        if (unvalidated) ++unvalidated_routes_;
    }
    auto slot = manager_->make_slot(std::move(spec), pcm::SlotPolicy::from(resolved_));
    entries_.push_back(Entry{slot, request.minimal});
    return slot;
}

pcm::GenerationStatus CoremlSidecars::generation_status() const {
    pcm::GenerationStatus g;
    if (!pcm::generation::built()) {
        g.reason = "build";
    } else if (!resolved_.enabled()) {
        g.reason = resolved_.disabled_reason.empty() ? "use=off" : resolved_.disabled_reason;
    } else if (resolved_.generate == pcm::Generate::never) {
        g.reason = "generate=never";
    } else if (resolved_.cache_dir.empty()) {
        g.reason = "no_cache_dir";
    } else if (manager_->platform() != pcm::Platform::macos || !pcm::os_at_least(manager_->os_version(), 15, 0)) {
        g.reason = "os";
    } else if (generated_routes_ > 0 && unvalidated_routes_ == generated_routes_) {
        g.reason = pcm::kUnvalidatedUnits;
    } else {
        g.supported = true;
    }
    return g;
}

std::vector<pcm::GroupRequest> CoremlSidecars::requests(pcm::Prewarm scope, pcm::GroupKind kind,
                                                        std::vector<pcm::SlotStatus> & skipped) const {
    std::vector<pcm::GroupRequest> out;
    if (scope == pcm::Prewarm::none || scope == pcm::Prewarm::unset) return out;
    for (const Entry & e : entries_) {
        if (scope != pcm::Prewarm::all && !e.minimal) continue;
        if (e.slot->spec().route == pcm::Route::none) {
            skipped.push_back(e.slot->status());
            continue;
        }
        // The construction-time minimal set is `minimal`; `all` extras and
        // later prewarm groups are `prewarm`, which loads after every minimal
        // request.
        const bool minimal = kind == pcm::GroupKind::construction && e.minimal;
        out.push_back(pcm::GroupRequest{e.slot, minimal ? pcm::Priority::minimal : pcm::Priority::prewarm});
    }
    return out;
}

void CoremlSidecars::start() {
    std::vector<pcm::SlotStatus> skipped;
    auto reqs = requests(resolved_.prewarm, pcm::GroupKind::construction, skipped);
    const bool waited = resolved_.wait == pcm::Wait::construct;
    construction_ = manager_->issue_group(pcm::GroupKind::construction, reqs, waited, std::move(skipped));
    if (waited) construction_->wait();
}

pcm::PrewarmResult CoremlSidecars::prewarm(CoremlOptions::Prewarm scope, bool wait,
                                           const std::atomic<bool> * cancel) {
    std::vector<pcm::SlotStatus> skipped;
    const auto reqs = requests(to_runtime_prewarm(scope), pcm::GroupKind::prewarm, skipped);
    auto group = manager_->issue_group(pcm::GroupKind::prewarm, reqs, wait, std::move(skipped));
    return wait ? group->wait(std::nullopt, cancel) : group->result();
}

bool CoremlSidecars::wait_construction(int64_t timeout_ms, const std::atomic<bool> * cancel) {
    if (!construction_) return true;  // no group issued: nothing to wait for
    std::optional<pcm::Clock::duration> timeout;
    if (timeout_ms >= 0) timeout = std::chrono::milliseconds(timeout_ms);
    return construction_->wait(timeout, cancel).success;
}

std::string CoremlSidecars::construction_json() const {
    if (construction_) return pcm::to_json(construction_->result());
    pcm::PrewarmResult empty;
    empty.group = "construction";
    empty.success = true;
    empty.nothing_to_prepare = true;
    return pcm::to_json(empty);
}

std::string CoremlSidecars::status_json() const {
    return pcm::to_json(manager_->status(kEngine, resolved_.cache_dir, slots(), generation_status()));
}

std::vector<std::shared_ptr<pcm::Slot>> CoremlSidecars::slots() const {
    std::vector<std::shared_ptr<pcm::Slot>> out;
    out.reserve(entries_.size());
    for (const Entry & e : entries_) out.push_back(e.slot);
    return out;
}

// ---- Diarizer capacities -------------------------------------------------

std::vector<DiarizerCapacity> diarizer_capacities(const std::vector<int> & prebuilt,
                                                  const std::vector<int> & template_frames,
                                                  const std::vector<pcm::generation::Candidate> & probed) {
    std::vector<DiarizerCapacity> out;
    for (int frames : prebuilt) {
        if (std::none_of(out.begin(), out.end(), [&](const DiarizerCapacity & c) { return c.frames == frames; })) {
            out.push_back(DiarizerCapacity{frames, true, nullptr});
        }
    }
    for (size_t i = 0; i < template_frames.size() && i < probed.size(); ++i) {
        const int frames = template_frames[i];
        // An adjacent prebuilt bundle wins; a capacity is listed once.
        if (std::any_of(out.begin(), out.end(), [&](const DiarizerCapacity & c) { return c.frames == frames; })) {
            continue;
        }
        out.push_back(DiarizerCapacity{frames, false, &probed[i]});
    }
    std::sort(out.begin(), out.end(),
              [](const DiarizerCapacity & a, const DiarizerCapacity & b) { return a.frames < b.frames; });
    return out;
}

std::vector<pcm::generation::Candidate> probe_diarizer_templates(const CoremlSidecars & sidecars,
                                                                 const std::string & gguf_path,
                                                                 const std::vector<std::string> & template_ids) {
    if (template_ids.empty()) return {};
    if (!sidecars.resolved().enabled()) {
        std::vector<pcm::generation::Candidate> unprobed(template_ids.size());
        for (size_t i = 0; i < template_ids.size(); ++i) unprobed[i].template_id = template_ids[i];
        return unprobed;
    }
    return pcm::generation::probe(template_ids, gguf_path, sidecars.resolved());
}

SidecarRequest diarizer_request(const std::string & gguf_path, const DiarizerCapacity & c, pcm::LoadFn load_fn,
                                const pcm::SourceStat & source) {
    SidecarRequest request;
    request.stage = "diarizer-" + std::to_string(c.frames);
    request.path = coreml_diarizer_sidecar_path(gguf_path, c.frames);
    request.present = c.prebuilt;
    request.stage_default = pcm::ComputeUnits::cpu_and_ane;
    request.load_fn = load_fn;
    request.geometry = sidecar_geometry(SidecarLoader::diarizer, {c.frames});
    request.generated = c.prebuilt ? nullptr : c.generated;
    request.source = source;
    return request;
}

std::vector<CoremlDiarizerCharge> diarizer_fit_charges(const CoremlSidecars & sidecars, const std::string & gguf_path,
                                                       const std::vector<DiarizerCapacity> & capacities) {
    std::vector<CoremlDiarizerCharge> charges;
    charges.reserve(capacities.size());
    for (const DiarizerCapacity & c : capacities) {
        CoremlDiarizerCharge charge;
        charge.frames = c.frames;
        charge.prebuilt = c.prebuilt;
        const SidecarRequest request = diarizer_request(gguf_path, c, nullptr, pcm::SourceStat{});
        if (request.generated != nullptr && sidecars.plan(request).route == pcm::Route::generate) {
            const pcm::generation::TemplateMemory & memory = request.generated->memory;
            const pcm::ComputeUnits units = sidecars.resolved().units_or(request.stage_default);
            charge.generated = true;
            charge.resident_bytes = pcm::generation::resident_bytes(memory);
            charge.transient_bytes = pcm::generation::transient_bytes(memory, units);
            charge.margin = memory.margin;
        }
        charges.push_back(charge);
    }
    return charges;
}

DiarizerSlots::DiarizerSlots(const std::string & gguf_path, std::vector<DiarizerCapacity> capacities,
                             CoremlSidecars & sidecars, pcm::LoadFn load_fn, std::optional<pcm::SourceStat> loaded)
    : strict_(sidecars.resolved().strict()) {
    std::sort(capacities.begin(), capacities.end(),
              [](const DiarizerCapacity & a, const DiarizerCapacity & b) { return a.frames < b.frames; });
    pcm::SourceStat source;
    if (loaded) {
        source = *loaded;
    } else if (std::any_of(capacities.begin(), capacities.end(), [](const DiarizerCapacity & c) {
                   return !c.prebuilt && c.generated != nullptr && c.generated->matched;
               })) {
        pcm::generation::source_stat(gguf_path, source);
    }
    bool minimal_served = false;
    for (const DiarizerCapacity & c : capacities) {
        SidecarRequest request = diarizer_request(gguf_path, c, load_fn, source);
        const pcm::Plan plan = sidecars.plan(request);
        const bool serves = c.prebuilt || plan.route == pcm::Route::generate;
        // A status-only slot never loads; in the minimal set the construction
        // group lists it as skipped.
        request.minimal = !serves || !minimal_served;
        minimal_served = minimal_served || serves;
        capacities_.push_back(Capacity{c.frames, sidecars.add(request, plan), serves});
    }
}

void DiarizerSlots::refuse_too_wide(int frames) const {
    if (!strict_ || capacities_.empty() || capacities_.back().frames >= frames) return;
    throw pcm::CoremlError("coreml: a diarization pass of " + std::to_string(frames) +
                               " frames is longer than every Core ML diarizer capacity (largest " +
                               std::to_string(capacities_.back().frames) + ") and Core ML is required",
                           pcm::SlotState::ready, pcm::ErrorClass::deterministic);
}

bool DiarizerSlots::covers(int frames) const {
    if (frames <= 0) return false;
    refuse_too_wide(frames);
    for (const Capacity & c : capacities_) {
        if (c.frames >= frames && (strict_ || !pcm::is_terminal(c.slot->state()))) return true;
    }
    return false;
}

bool DiarizerSlots::prepare(int frames) const {
    if (frames <= 0) return false;
    for (const Capacity & c : capacities_) {
        if (c.frames < frames || pcm::is_terminal(c.slot->state())) continue;
        c.slot->request(pcm::Priority::current);
        return c.slot->state() == pcm::SlotState::ready;
    }
    return false;
}

DiarizerPass DiarizerSlots::run(int frames, const std::atomic<bool> * cancel, const Predict & predict) const {
    DiarizerPass pass;
    if (frames <= 0) return pass;
    refuse_too_wide(frames);
    // Strict: why the covering capacities tried so far could not serve the pass.
    std::optional<pcm::CoremlError> refused;
    for (const Capacity & c : capacities_) {
        if (c.frames < frames) continue;
        // Strict acquisitions of a failed or status-only capacity throw;
        // otherwise the capacity is skipped.
        if (!strict_ && pcm::is_terminal(c.slot->state())) continue;
        // One acquisition per pass (a chunk or block is the unit).
        pcm::Acquired acquired;
        try {
            acquired = c.slot->acquire(pcm::Demand::now, std::nullopt, cancel);
        } catch (const pcm::CoremlError & error) {
            // A failed capacity gives way to the next covering one; a timeout
            // or a memory deferral refuses the pass at once.
            if (!pcm::is_terminal(error.state)) throw;
            if (!refused) refused = error;
            continue;
        }
        if (!acquired.model) {
            // Failed while this pass waited (wait = call): try the next capacity.
            if (pcm::is_terminal(acquired.state)) continue;
            pass.not_ready = true;
            return pass;
        }
        pass.status = predict(acquired.model, c.frames);
        if (pass.status != 0) {
            const std::string reason = "prediction failed (rc=" + std::to_string(pass.status) + ")";
            c.slot->retire(reason);
            if (strict_) {
                throw pcm::CoremlError("coreml: " + c.slot->spec().stage + " " + reason + " and Core ML is required",
                                       pcm::SlotState::retired, pcm::ErrorClass::transient);
            }
        }
        return pass;
    }
    if (refused) throw *refused;
    return pass;
}

std::vector<int> DiarizerSlots::capacities() const {
    std::vector<int> out;
    for (const Capacity & c : capacities_) {
        if (c.serves) out.push_back(c.frames);
    }
    return out;
}

std::vector<std::shared_ptr<pcm::Slot>> DiarizerSlots::status_only() const {
    std::vector<std::shared_ptr<pcm::Slot>> out;
    for (const Capacity & c : capacities_) {
        if (!c.serves) out.push_back(c.slot);
    }
    return out;
}

}  // namespace parakeet
