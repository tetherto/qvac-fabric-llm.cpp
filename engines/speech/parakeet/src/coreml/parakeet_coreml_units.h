// Compute-unit selection shared by the Core ML encoder, diarizer and decoder
// wrappers. (The MOSS-Transcribe encoder, not yet on the runtime, reads its
// units from MOSS_COREML_COMPUTE_UNITS through
// parakeet_coreml_init_with_units_env until it is adopted.)
//
// Apple-only, Objective-C++ header: included by the .mm wrappers under
// PARAKEET_USE_COREML only.

#pragma once

#import <CoreML/CoreML.h>

#include <cstring>
#include <string>

// Maps resolved compute units (`all`, `cpu_and_ane`,
// `cpu_and_gpu`, `cpu_only`) to MLComputeUnits and writes the matching stats
// label. The engine resolves the units once (CoremlOptions::compute_units,
// then PARAKEET_COREML_COMPUTE_UNITS, then the stage default) and passes them
// here; loaders never read the environment themselves.
//
// `units` null or unrecognised selects `prefer_neural_engine ? cpu_and_ane :
// all`: the graphs exported for the Neural Engine (the fixed-window encoder,
// the diarizers and the batched TDT decoder) are fully ANE-shaped, and
// MLComputeUnitsAll is only a preference that may resolve to the GPU. The
// generic encoder exports keep MLComputeUnitsAll, since their few
// GPU-preferred operations would otherwise fall back to the CPU.
inline MLComputeUnits parakeet_coreml_units_for(const char * units,
                                                bool prefer_neural_engine,
                                                std::string * label) {
    if (units != nullptr && std::strcmp(units, "all") == 0) {
        *label = "coreml-all";
        return MLComputeUnitsAll;
    }
    if (units != nullptr && std::strcmp(units, "cpu_and_gpu") == 0) {
        *label = "coreml-cpu-gpu";
        return MLComputeUnitsCPUAndGPU;
    }
    if (units != nullptr && std::strcmp(units, "cpu_only") == 0) {
        *label = "coreml-cpu";
        return MLComputeUnitsCPUOnly;
    }
    const bool want_ane =
        prefer_neural_engine || (units != nullptr && std::strcmp(units, "cpu_and_ane") == 0);
    if (want_ane) {
        if (@available(macOS 13.0, iOS 16.0, tvOS 16.0, watchOS 9.0, *)) {
            *label = "coreml-ane";
            return MLComputeUnitsCPUAndNeuralEngine;
        }
    }
    *label = "coreml-all";
    return MLComputeUnitsAll;
}
