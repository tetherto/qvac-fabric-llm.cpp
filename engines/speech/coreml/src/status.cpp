#include "qvac_coreml/status.h"

#include "nlohmann/json.hpp"

namespace QVAC_COREML_NS {

const char * to_string(SlotState s) {
    switch (s) {
        case SlotState::idle: return "idle";
        case SlotState::unsupported: return "unsupported";
        case SlotState::disabled: return "disabled";
        case SlotState::absent: return "absent";
        case SlotState::no_template: return "no_template";
        case SlotState::queued: return "queued";
        case SlotState::deferred_memory: return "deferred(memory)";
        case SlotState::digesting: return "digesting";
        case SlotState::generating: return "generating";
        case SlotState::compiling: return "compiling";
        case SlotState::loading: return "loading";
        case SlotState::warming: return "warming";
        case SlotState::ready: return "ready";
        case SlotState::failed: return "failed";
        case SlotState::retired: return "retired";
    }
    return "?";
}

std::string state_label(SlotState s, std::string_view reason) {
    if (s == SlotState::unsupported && !reason.empty()) return "unsupported(" + std::string(reason) + ")";
    return to_string(s);
}

bool is_terminal(SlotState s) {
    switch (s) {
        case SlotState::unsupported:
        case SlotState::disabled:
        case SlotState::absent:
        case SlotState::no_template:
        case SlotState::failed:
        case SlotState::retired: return true;
        default: return false;
    }
}

const char * to_string(ErrorClass c) {
    switch (c) {
        case ErrorClass::none: return "none";
        case ErrorClass::deterministic: return "deterministic";
        case ErrorClass::transient: return "transient";
    }
    return "?";
}

const char * to_string(Route r) {
    switch (r) {
        case Route::none: return "none";
        case Route::prebuilt: return "prebuilt";
        case Route::generate: return "generate";
    }
    return "?";
}

const char * to_string(StatusRoute r) {
    switch (r) {
        case StatusRoute::none: return "none";
        case StatusRoute::prebuilt: return "prebuilt";
        case StatusRoute::cached: return "cached";
        case StatusRoute::generated: return "generated";
    }
    return "?";
}

const char * label_for(ComputeUnits u) {
    switch (u) {
        case ComputeUnits::all: return "coreml-all";
        case ComputeUnits::cpu_and_ane: return "coreml-ane";
        case ComputeUnits::cpu_and_gpu: return "coreml-gpu";
        case ComputeUnits::cpu_only: return "coreml-cpu";
    }
    return "ggml";
}

std::string merge_labels(std::string_view acc, std::string_view next) {
    if (acc.empty()) return std::string(next);
    if (next.empty() || acc == next) return std::string(acc);
    return "mixed";
}

namespace {

nlohmann::json str_or_null(const std::string & s) { return s.empty() ? nlohmann::json(nullptr) : nlohmann::json(s); }

nlohmann::json slot_json(const SlotStatus & s) {
    nlohmann::json j;
    j["stage"] = s.stage;
    j["function"] = s.function;
    j["state"] = state_label(s.state, s.reason);
    j["route"] = to_string(s.route);
    j["compute_units"] = s.compute_units;
    j["label"] = s.label;
    j["artifact_key"] = str_or_null(s.artifact_key);
    j["path"] = str_or_null(s.path);
    if (s.hint) {
        j["hint"] = {
            {"warm_load_ms", s.hint->warm_load_ms ? nlohmann::json(*s.hint->warm_load_ms) : nlohmann::json(nullptr)},
            {"stale", s.hint->stale},
        };
    } else {
        j["hint"] = nullptr;
    }
    j["timings_ms"] = {
        {"queued", s.timings_ms.queued}, {"digest", s.timings_ms.digest}, {"write", s.timings_ms.write},
        {"compile", s.timings_ms.compile}, {"load", s.timings_ms.load},   {"warm", s.timings_ms.warm},
    };
    j["bytes_on_disk"] = s.bytes_on_disk;
    const bool has_error = s.state == SlotState::failed || s.state == SlotState::retired;
    j["error"] = has_error ? str_or_null(s.error.empty() ? s.reason : s.error) : nlohmann::json(nullptr);
    j["error_class"] = s.error_class == ErrorClass::none ? nlohmann::json(nullptr) : nlohmann::json(to_string(s.error_class));
    if (!s.wait.empty()) j["wait"] = s.wait;
    if (s.state == SlotState::deferred_memory) j["deferred_ms"] = s.deferred_ms < 0 ? 0 : s.deferred_ms;
    return j;
}

}  // namespace

std::string to_json(const SlotStatus & s) { return slot_json(s).dump(); }

std::string to_json(const EngineStatus & s) {
    nlohmann::json j;
    j["schema"] = 1;
    j["engine"] = s.engine;
    j["cache_dir"] = s.cache_dir;
    j["generation"] = {{"supported", s.generation.supported}, {"reason", str_or_null(s.generation.reason)}};
    nlohmann::json stages = nlohmann::json::array();
    for (const auto & st : s.stages) stages.push_back(slot_json(st));
    j["stages"] = std::move(stages);
    return j.dump();
}

std::string to_json(const PrewarmResult & r) {
    nlohmann::json j;
    j["group"] = r.group;
    j["success"] = r.success;
    j["nothing_to_prepare"] = r.nothing_to_prepare;
    nlohmann::json slots = nlohmann::json::array();
    for (const auto & s : r.slots) slots.push_back(slot_json(s));
    nlohmann::json skipped = nlohmann::json::array();
    for (const auto & s : r.skipped_route_none) skipped.push_back(slot_json(s));
    j["slots"] = std::move(slots);
    j["skipped_route_none"] = std::move(skipped);
    return j.dump();
}

}  // namespace QVAC_COREML_NS
