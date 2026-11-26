#include "moss/sfx_sampler.h"

#include <cmath>
#include <random>
#include <stdexcept>

namespace tts_cpp::moss::detail {
namespace {

constexpr double TWO_PI = 6.283185307179586;
constexpr double UNIT_INTERVAL = 4294967296.0;

float shifted_sigma(float sigma, float shift) {
    return shift * sigma / (1.0f + (shift - 1.0f) * sigma);
}

double uniform_open(std::mt19937 & rng) {
    return ((double) rng() + 0.5) / UNIT_INTERVAL;
}

std::vector<float> shifted_schedule(int steps, float shift) {
    std::vector<float> sigmas((size_t) steps);
    for (int i = 0; i < steps; ++i) {
        sigmas[(size_t) i] = shifted_sigma(1.0f - (float) i / (float) steps, shift);
    }
    return sigmas;
}

void advance_latents(std::vector<float> & latents, const std::vector<float> & velocity, float delta) {
    for (size_t i = 0; i < latents.size(); ++i) {
        latents[i] += velocity[i] * delta;
    }
}

std::vector<float> blend_guidance(const std::vector<float> & positive, const std::vector<float> & negative,
                                  float guidance) {
    std::vector<float> guided(positive.size());
    for (size_t i = 0; i < guided.size(); ++i) {
        guided[i] = negative[i] + guidance * (positive[i] - negative[i]);
    }
    return guided;
}

void fill_box_muller(std::vector<float> & noise, std::mt19937 & rng) {
    for (size_t i = 0; i < noise.size(); i += 2) {
        const double radius = std::sqrt(-2.0 * std::log(uniform_open(rng)));
        const double angle = TWO_PI * uniform_open(rng);
        noise[i] = (float) (radius * std::cos(angle));
        if (i + 1 < noise.size()) {
            noise[i + 1] = (float) (radius * std::sin(angle));
        }
    }
}

} // namespace

std::vector<float> flow_sigmas(int steps, float shift) {
    if (steps < 1 || !(shift > 0.0f)) {
        throw std::runtime_error("moss sfx: steps must be positive and shift must be > 0");
    }
    return shifted_schedule(steps, shift);
}

void euler_step(std::vector<float> & latents, const std::vector<float> & velocity, float sigma, float next_sigma) {
    if (latents.size() != velocity.size()) {
        throw std::runtime_error("moss sfx: velocity and latents disagree in size");
    }
    advance_latents(latents, velocity, next_sigma - sigma);
}

std::vector<float> guided_velocity(const std::vector<float> & positive, const std::vector<float> & negative,
                                   float guidance) {
    if (positive.size() != negative.size()) {
        throw std::runtime_error("moss sfx: guidance branches disagree in size");
    }
    return blend_guidance(positive, negative, guidance);
}

std::vector<float> gaussian_noise(size_t count, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<float> noise(count);
    fill_box_muller(noise, rng);
    return noise;
}

} // namespace tts_cpp::moss::detail
