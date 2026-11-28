#pragma once

#include "npy.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace tts_test {

static_assert(sizeof(float) == 4 && sizeof(int32_t) == 4,
              "Parity fixtures require 32-bit floats and integers");

inline bool check_layout(const char * name, size_t count,
                         const std::vector<int64_t> & shape,
                         const npy_array & ref, const char * dtype) {
    if (count == 0 || ref.shape != shape || ref.n_elements() != count ||
        ref.dtype != dtype || ref.data.size() != count * sizeof(float)) {
        std::fprintf(stderr, "FAIL [%s]: expected a nonempty %s reference with matching shape and size\n",
                     name, dtype);
        return false;
    }
    return true;
}

inline bool check_f32(const char * name, const std::vector<float> & got,
                      const std::vector<int64_t> & shape,
                      const npy_array & ref, double tolerance) {
    if (!check_layout(name, got.size(), shape, ref, "<f4")) return false;
    // Copy into typed storage; the NPY reader stores raw bytes.
    std::vector<float> expected(got.size());
    std::memcpy(expected.data(), ref.data.data(), ref.data.size());
    const auto stats = compare_f32(got.data(), expected.data(), got.size());
    print_compare(name, stats);
    if (compare_within(stats, tolerance)) return true;
    std::fprintf(stderr, "FAIL [%s]: relative error must be below %.3e, with finite values\n",
                 name, tolerance);
    return false;
}

inline bool check_i32(const char * name, const std::vector<int32_t> & got,
                      const std::vector<int64_t> & shape, const npy_array & ref) {
    if (!check_layout(name, got.size(), shape, ref, "<i4")) return false;
    std::vector<int32_t> expected(got.size());
    std::memcpy(expected.data(), ref.data.data(), ref.data.size());
    size_t matches = 0;
    for (size_t i = 0; i < got.size(); ++i) matches += got[i] == expected[i];
    std::fprintf(stderr, "[%s]: %zu / %zu tokens match\n", name, matches, got.size());
    return got == expected;
}

} // namespace tts_test
