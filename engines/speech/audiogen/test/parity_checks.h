#pragma once

#include <cmath>

namespace audiogen_test {

inline bool samples_match(float expected, float actual, float tolerance) {
    return std::isfinite(expected) && std::isfinite(actual) &&
           std::fabs(expected - actual) <= tolerance;
}

inline bool cosine_passes(double cosine, double minimum) {
    return std::isfinite(cosine) && cosine >= minimum;
}

inline bool relative_error_passes(double error, double maximum) {
    return std::isfinite(error) && error >= 0.0 && error <= maximum;
}

}  // namespace audiogen_test
