#pragma once

// Reading an attention mask on its f16 bits, shared by the array route
// (xdna-attn-mm.cpp) and the host kernel (xdna-host-attn.cpp).

#include "ggml.h"

#include <cstdint>
#include <cstring>

namespace xdna_attn_mask {

// Only a graph input holds one set of values at one address for the whole graph: a computed mask's memory
// is reused or rewritten by later nodes, so what was read off it by address may not be kept.
inline bool is_graph_input(const ggml_tensor * m) {
    return m && (m->flags & GGML_TENSOR_FLAG_INPUT) != 0;
}

// On the f16 bits: a visible key is 0.0 (0x0000 or 0x8000), a hidden one below
// -1e30, which in f16 is only -inf (0xFC00). Converting each value to f32 cost
// pp4096 18% (1604 -> 1309 t/s): the whole mask, every attention layer of
// every ubatch, twice a node.
constexpr uint16_t MASK_HIDDEN = 0xFC00;

constexpr uint64_t MASK_HIDDEN_X4 = 0xFC00FC00FC00FC00ull;  // four hidden keys

inline bool four_hidden(const uint16_t * p) {
    uint64_t w;
    std::memcpy(&w, p, sizeof(w));
    return w == MASK_HIDDEN_X4;
}

inline int64_t first_not_hidden(const uint16_t * row, int64_t n) {
    int64_t j = 0;
    while (j + 4 <= n && four_hidden(row + j)) {
        j += 4;
    }
    while (j < n && row[j] == MASK_HIDDEN) {
        j++;
    }
    return j;
}

inline int64_t last_not_hidden(const uint16_t * row, int64_t n) {
    int64_t j = n - 1;
    while (j >= 3 && four_hidden(row + j - 3)) {
        j -= 4;
    }
    while (j >= 0 && row[j] == MASK_HIDDEN) {
        j--;
    }
    return j;
}

// True when the keys lo .. hi of the row are all visible.
inline bool all_visible(const uint16_t * row, int64_t lo, int64_t hi) {
    uint16_t bad = 0;
    for (int64_t j = lo; j <= hi; j++) {
        bad |= row[j] & 0x7FFF;
    }
    return bad == 0;
}

}  // namespace xdna_attn_mask
