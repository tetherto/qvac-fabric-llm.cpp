#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tts_cpp::moss::detail {

std::vector<float> flow_sigmas(int steps, float shift);
void euler_step(std::vector<float> & latents, const std::vector<float> & velocity, float sigma, float next_sigma);
std::vector<float> guided_velocity(const std::vector<float> & positive, const std::vector<float> & negative,
                                   float guidance);
std::vector<float> gaussian_noise(size_t count, uint32_t seed);

} // namespace tts_cpp::moss::detail
