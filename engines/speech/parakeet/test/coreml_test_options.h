#pragma once

// Device tests that compare a Core ML sidecar against ggml wait for every
// sidecar at construction. Sidecars load in the background, so
// without this the first calls would run on ggml while the load is in flight.

#include "parakeet/coreml_options.h"

#include <string>

inline parakeet::CoremlOptions coreml_ready_options() {
    parakeet::CoremlOptions options;
    options.wait = parakeet::CoremlOptions::Wait::construct;
    options.prewarm = parakeet::CoremlOptions::Prewarm::all;
    return options;
}

// True when the engine planned a Core ML sidecar (an adjacent prebuilt
// `.mlmodelc`, or a generation route), from Engine::coreml_status_json(). A
// gate that finds Core ML inactive while a sidecar is planned must fail, not
// skip: a skip there would let the lane go green without exercising Core ML.
inline bool coreml_sidecar_planned(const std::string & status_json) {
    return status_json.find("\"route\":\"prebuilt\"") != std::string::npos ||
           status_json.find("\"route\":\"generate\"") != std::string::npos;
}
