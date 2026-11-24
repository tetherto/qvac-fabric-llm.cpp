#pragma once

#include "../backend_selection.h"

namespace tts_cpp::lavasr {

// LavaSR runs its graphs directly on the selected backend, with no per-op CPU
// fallback. Hexagon uses F32 graph decompositions for the denoiser's fused
// operations; WebGPU is not validated for either neural graph.
inline ::tts_cpp::detail::GpuBackendRequirement gpu_requirement() {
    using ::tts_cpp::detail::GpuBackendRequirement;
    return GpuBackendRequirement::Vulkan | GpuBackendRequirement::Metal |
           GpuBackendRequirement::OpenCL | GpuBackendRequirement::CUDA |
           GpuBackendRequirement::Hexagon;
}

} // namespace tts_cpp::lavasr
