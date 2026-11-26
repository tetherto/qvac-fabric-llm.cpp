#pragma once

// Element counts of recipe shapes, with overflow checks.
//
// A recipe shape is valid only when its element count, times the widest
// element the interpreter holds (f64, 8 bytes), fits in int64_t and in
// size_t. The parser rejects every shape above that bound, so the byte sizes,
// strides and loop counts the interpreter derives from a parsed shape never
// wrap, including on 32-bit hosts.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#ifndef QVAC_COREML_NS
#error "QVAC_COREML_NS must be defined"
#endif

namespace QVAC_COREML_NS {
namespace detail {

inline constexpr uint64_t MAX_ELEMENTS =
    (static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) < std::numeric_limits<size_t>::max()
         ? static_cast<uint64_t>(std::numeric_limits<int64_t>::max())
         : static_cast<uint64_t>(std::numeric_limits<size_t>::max())) /
    sizeof(double);

// a * b, or nullopt when the product exceeds MAX_ELEMENTS.
inline std::optional<uint64_t> mul_elements(uint64_t a, uint64_t b) {
    if (b != 0 && a > MAX_ELEMENTS / b) return std::nullopt;
    return a * b;
}

// a + b, or nullopt when the sum exceeds MAX_ELEMENTS.
inline std::optional<uint64_t> add_elements(uint64_t a, uint64_t b) {
    if (a > MAX_ELEMENTS || b > MAX_ELEMENTS - a) return std::nullopt;
    return a + b;
}

// Product of the dims of `shape` (1 for a scalar), or nullopt when a dim is
// negative or the product exceeds MAX_ELEMENTS.
inline std::optional<uint64_t> element_count(const std::vector<int64_t> & shape) {
    uint64_t n = 1;
    for (int64_t d : shape) {
        if (d < 0) return std::nullopt;
        const std::optional<uint64_t> m = mul_elements(n, static_cast<uint64_t>(d));
        if (!m) return std::nullopt;
        n = *m;
    }
    return n;
}

}  // namespace detail
}  // namespace QVAC_COREML_NS
