#include "parity_checks.h"

#include <cstdio>
#include <initializer_list>
#include <limits>

int main() {
    constexpr float tolerance = 0.01f;
    constexpr double minimum_cosine = 0.999;
    constexpr double maximum_error = 0.1;
    int failures = 0;
    const auto check = [&failures](bool condition) {
        if (!condition) ++failures;
    };
    check(audiogen_test::samples_match(1.0f, 1.0f, tolerance));
    check(!audiogen_test::samples_match(1.0f, 2.0f, tolerance));
    check(audiogen_test::cosine_passes(1.0, minimum_cosine));
    check(!audiogen_test::cosine_passes(0.5, minimum_cosine));
    check(audiogen_test::relative_error_passes(0.0, maximum_error));
    check(!audiogen_test::relative_error_passes(1.0, maximum_error));
    for (float invalid : {std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::infinity(),
                          -std::numeric_limits<float>::infinity()}) {
        check(!audiogen_test::samples_match(1.0f, invalid, tolerance));
        check(!audiogen_test::samples_match(invalid, 1.0f, tolerance));
        check(!audiogen_test::cosine_passes(invalid, minimum_cosine));
        check(!audiogen_test::relative_error_passes(invalid, maximum_error));
    }
    if (failures) std::fprintf(stderr, "%d parity assertion failures\n", failures);
    return failures ? 1 : 0;
}
