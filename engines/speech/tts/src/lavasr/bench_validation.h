#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace tts_cpp::lavasr::bench {

struct Difference {
    bool length_ok = false;
    bool finite = true;
    double max_abs = 0.0;
    double nrmse = 0.0;
    double cosine = 1.0;
};

inline Difference compare(const std::vector<float> & actual, const std::vector<float> & reference) {
    Difference d;
    d.length_ok = actual.size() == reference.size();
    if (!d.length_ok) return d;
    double squared_error = 0.0, aa = 0.0, bb = 0.0, ab = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(reference[i])) {
            d.finite = false;
            return d;
        }
        const double a = actual[i], b = reference[i], error = a - b;
        d.max_abs = std::max(d.max_abs, std::fabs(error));
        squared_error += error * error;
        aa += a * a;
        bb += b * b;
        ab += a * b;
    }
    d.nrmse = std::sqrt(squared_error / std::max(bb, 1e-20));
    d.cosine = aa == 0.0 && bb == 0.0 ? 1.0 : ab / std::max(std::sqrt(aa * bb), 1e-20);
    return d;
}

inline bool passes(const Difference & d, double max_abs, double nrmse, double min_cosine) {
    return d.length_ok && d.finite && std::isfinite(d.max_abs) && std::isfinite(d.nrmse) &&
           std::isfinite(d.cosine) && d.max_abs < max_abs && d.nrmse < nrmse && d.cosine >= min_cosine;
}

} // namespace tts_cpp::lavasr::bench
