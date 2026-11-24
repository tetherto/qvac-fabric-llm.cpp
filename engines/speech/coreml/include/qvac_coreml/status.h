#pragma once

// Slot states, routes, labels and the status JSON.

#include "qvac_coreml/options.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace QVAC_COREML_NS {

// Slot states. `idle` is a slot whose route is not `none` and for which
// nothing has been requested (or whose request an eviction cancelled).
enum class SlotState {
    idle,
    // route = none
    unsupported,
    disabled,
    absent,
    no_template,
    // pipeline
    queued,
    deferred_memory,
    digesting,
    generating,
    compiling,
    loading,
    warming,
    ready,
    // terminal
    failed,
    retired,
};

const char * to_string(SlotState s);

// `unsupported(<reason>)`, `deferred(memory)`, otherwise to_string().
std::string state_label(SlotState s, std::string_view reason);

// Terminal states return immediately from every wait.
bool is_terminal(SlotState s);

enum class ErrorClass { none, deterministic, transient };

const char * to_string(ErrorClass c);

// plan() vocabulary. Whether an artifact was cached or generated
// appears only in status (StatusRoute).
enum class Route { none, prebuilt, generate };
enum class StatusRoute { none, prebuilt, cached, generated };

const char * to_string(Route r);
const char * to_string(StatusRoute r);

// Today's result labels: coreml-all | coreml-ane | coreml-gpu | coreml-cpu.
const char * label_for(ComputeUnits units);

// Per-call aggregation: equal labels stay, anything else is "mixed"; an empty
// accumulator takes the next label.
std::string merge_labels(std::string_view acc, std::string_view next);

struct HintView {
    std::optional<int64_t> warm_load_ms;
    std::optional<int64_t> cold_load_ms;
    bool stale = false;
};

struct Timings {
    int64_t queued = 0;
    int64_t digest = 0;
    int64_t write = 0;
    int64_t compile = 0;
    int64_t load = 0;
    int64_t warm = 0;
};

struct SlotStatus {
    std::string stage;
    std::string function;
    SlotState state = SlotState::idle;
    std::string reason;  // unsupported / disabled / absent reason, failure or retirement reason
    StatusRoute route = StatusRoute::none;
    std::string compute_units;  // resolved enum vocabulary
    std::string label;          // coreml-* when ready, else ggml
    std::string artifact_key;
    std::string path;
    std::optional<HintView> hint;
    Timings timings_ms;
    uint64_t bytes_on_disk = 0;
    std::string error;
    ErrorClass error_class = ErrorClass::none;
    std::string wait;          // "queued_behind" after a bounded wait expired while queued
    int64_t deferred_ms = -1;  // >= 0 while deferred(memory)
};

struct GenerationStatus {
    bool supported = false;
    std::string reason;  // "" -> null
};

struct EngineStatus {
    std::string engine;
    std::string cache_dir;
    GenerationStatus generation;
    std::vector<SlotStatus> stages;
};

struct PrewarmResult {
    std::string group;  // "construction" | "prewarm"
    bool success = false;
    bool nothing_to_prepare = false;
    std::vector<SlotStatus> slots;
    std::vector<SlotStatus> skipped_route_none;
};

std::string to_json(const SlotStatus & s);
std::string to_json(const EngineStatus & s);
std::string to_json(const PrewarmResult & r);

}  // namespace QVAC_COREML_NS
