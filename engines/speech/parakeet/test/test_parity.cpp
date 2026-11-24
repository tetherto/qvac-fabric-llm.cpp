#include "parity.h"

#include <cstdio>

namespace {

bool rejects(const std::vector<float> & actual, const std::vector<float> & expected) {
    double maximum = 0.0;
    double relative = 0.0;
    parakeet_test::compute_parity(actual, expected, maximum, relative);
    return std::isinf(maximum) && std::isinf(relative);
}

bool check_invalid_outputs() {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();
    return rejects({}, {}) && rejects({}, {1.0f}) && rejects({1.0f}, {}) &&
           rejects({1.0f}, {1.0f, 99.0f}) && rejects({1.0f, 99.0f}, {1.0f}) &&
           rejects({nan}, {1.0f}) && rejects({1.0f}, {nan}) &&
           rejects({infinity}, {infinity}) && rejects({1.0f}, {-infinity});
}

bool check_matrix_shapes() {
    using parakeet_test::matches_matrix;
    const std::vector<float> values(6, 1.0f);
    return matches_matrix(values, {2, 3}, 2, 3) &&
           !matches_matrix(values, {3, 2}, 2, 3) &&
           !matches_matrix(values, {6}, 2, 3) &&
           !matches_matrix(values, {2, 3, 1}, 2, 3) &&
           !matches_matrix({}, {0, 3}, 0, 3) &&
           !matches_matrix(values, {2, -3}, 2, -3) &&
           !matches_matrix({1.0f}, {2, 3}, 2, 3);
}

bool check_numerical_error() {
    double maximum = 0.0;
    double relative = 0.0;
    parakeet_test::compute_parity({3.0f, 4.0f}, {3.0f, 4.0f}, maximum, relative);
    if (maximum != 0.0 || relative != 0.0) return false;
    parakeet_test::compute_parity({6.0f, 8.0f}, {3.0f, 4.0f}, maximum, relative);
    if (maximum != 4.0 || relative != 1.0) return false;
    parakeet_test::compute_parity({3.0f, 4.0f}, {0.0f, 0.0f}, maximum, relative);
    return maximum == 4.0 && relative == 5.0;
}

}  // namespace

int main() {
    if (check_invalid_outputs() && check_matrix_shapes() && check_numerical_error()) return 0;
    std::fprintf(stderr, "Parity validation regression\n");
    return 1;
}
