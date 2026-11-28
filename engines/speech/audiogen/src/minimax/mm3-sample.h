#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <random>
#include <vector>

static int64_t mm3_sample_top_k(const float * logits, int64_t n, int top_k, std::mt19937_64 & rng,
                                std::vector<float> * scratch = nullptr) {
    if (n <= 0) {
        return 0;
    }
    std::vector<float>   local;
    std::vector<float> & v = scratch ? *scratch : local;
    v.resize((size_t) n);

    int64_t arg_best = 0;
    float   val_best = -INFINITY;
    for (int64_t i = 0; i < n; i++) {
        float x = logits[i];
        if (std::isnan(x)) {
            x = -1e9f;
        } else if (std::isinf(x)) {
            x = x > 0.0f ? 1e9f : -1e9f;
        }
        v[(size_t) i] = x;
        if (x > val_best) {
            val_best = x;
            arg_best = i;
        }
    }

    int64_t k = top_k > 0 ? (int64_t) top_k : n;
    if (k > n) {
        k = n;
    }

    float threshold = -INFINITY;
    if (k < n) {
        std::vector<float> sel(v);
        std::nth_element(sel.begin(), sel.begin() + (size_t) (k - 1), sel.end(), std::greater<float>());
        threshold = sel[(size_t) (k - 1)];
    }

    float max_v = -INFINITY;
    for (int64_t i = 0; i < n; i++) {
        if (v[(size_t) i] >= threshold && v[(size_t) i] > max_v) {
            max_v = v[(size_t) i];
        }
    }
    if (!std::isfinite(max_v)) {
        return arg_best;
    }

    double sum = 0.0;
    for (int64_t i = 0; i < n; i++) {
        if (v[(size_t) i] >= threshold) {
            const double p = std::exp((double) v[(size_t) i] - (double) max_v);
            sum += p;
            v[(size_t) i] = (float) p;
        } else {
            v[(size_t) i] = 0.0f;
        }
    }
    if (!(sum > 0.0)) {
        return arg_best;
    }

    const double u   = std::uniform_real_distribution<double>(0.0, 1.0)(rng) * sum;
    double       acc = 0.0;
    for (int64_t i = 0; i < n; i++) {
        acc += (double) v[(size_t) i];
        if (acc > u) {
            return i;
        }
    }
    return arg_best;
}
