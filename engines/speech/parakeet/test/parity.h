#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace parakeet_test {

inline bool matches_matrix(const std::vector<float> & values,
                           const std::vector<int64_t> & shape,
                           int64_t rows, int64_t columns) {
    if (rows <= 0 || columns <= 0 || shape != std::vector<int64_t>{rows, columns}) return false;
    const auto width = static_cast<uint64_t>(columns);
    return static_cast<uint64_t>(rows) <= std::numeric_limits<size_t>::max() / width &&
           values.size() == static_cast<size_t>(rows) * static_cast<size_t>(columns);
}

inline void compute_parity(const std::vector<float> & actual, const std::vector<float> & expected,
                           double & max_abs, double & relative) {
    max_abs = relative = std::numeric_limits<double>::infinity();
    if (actual.empty() || actual.size() != expected.size()) return;
    double maximum = 0.0;
    double reference_squared = 0.0;
    double difference_squared = 0.0;
    for (size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index]) || !std::isfinite(expected[index])) return;
        const double difference = static_cast<double>(actual[index]) - expected[index];
        maximum = std::max(maximum, std::abs(difference));
        difference_squared += difference * difference;
        reference_squared += static_cast<double>(expected[index]) * expected[index];
    }
    max_abs = maximum;
    relative = reference_squared > 0.0 ? std::sqrt(difference_squared / reference_squared)
                                       : std::sqrt(difference_squared);
}

}  // namespace parakeet_test
