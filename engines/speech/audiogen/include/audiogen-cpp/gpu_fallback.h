#pragma once

// Why a GPU-requested run ended up on the CPU.
//
// The backend registry knows this at the point it gives up, but until now it
// returned a bare null and the reason survived only as a verbose-only stderr
// line. A caller that sees CPU work after asking for a GPU cannot otherwise
// tell a missing backend module apart from a device that failed to initialise,
// and those have different fixes.

namespace tts_cpp {

enum class GpuFallbackReason {
    // A GPU backend was acquired; the run is not on the CPU.
    none,
    // The caller never asked for a GPU (n_gpu_layers <= 0).
    not_requested,
    // The registry enumerated no GPU or integrated-GPU device. Typically the
    // backend module never loaded: wrong backends_dir, or a .so the process
    // cannot reach.
    no_devices,
    // Devices were enumerated but none could initialise a usable backend.
    init_failed
};

// Stable identifier for logs and for a caller that forwards the reason on.
inline const char * gpu_fallback_reason_name(GpuFallbackReason reason) {
    switch (reason) {
        case GpuFallbackReason::none:          return "none";
        case GpuFallbackReason::not_requested: return "not-requested";
        case GpuFallbackReason::no_devices:    return "no-devices";
        case GpuFallbackReason::init_failed:   return "init-failed";
    }
    return "unknown";
}

} // namespace tts_cpp
