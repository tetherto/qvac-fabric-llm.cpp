#pragma once

#include <cstddef>
#include <functional>
#include <limits>
#include <string>
#include <vector>

using MM3DitReadback = std::function<bool(float *, size_t, std::string *)>;

static bool mm3_read_dit_output(float * output, size_t count,
                                const MM3DitReadback & readback, std::string * error) {
    if (!output || !readback || count > std::numeric_limits<size_t>::max() / sizeof(float)) {
        if (error) {
            *error = "DiT output readback inputs are invalid";
        }
        return false;
    }
    if (!readback(output, count * sizeof(float), error)) {
        return false;
    }
    return true;
}

static bool mm3_integrate_flow_step(std::vector<float> & latents,
                                    const std::vector<float> & conditional,
                                    const std::vector<float> & unconditional,
                                    const std::vector<float> & sigmas, size_t step,
                                    float cfg_scale, std::string * error) {
    if (conditional.size() != latents.size() || unconditional.size() != latents.size() ||
        step >= sigmas.size() || step + 1 >= sigmas.size()) {
        if (error) {
            *error = "flow update inputs are invalid";
        }
        return false;
    }
    const float sigma_delta = sigmas[step + 1] - sigmas[step];
    for (size_t index = 0; index < latents.size(); ++index) {
        const float unconditioned = unconditional[index];
        const float velocity =
            unconditioned + cfg_scale * (conditional[index] - unconditioned);
        latents[index] += sigma_delta * velocity;
    }
    return true;
}
