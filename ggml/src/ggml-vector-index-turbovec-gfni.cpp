// ggml-vector-index-turbovec-gfni.cpp - the GFNI form of the x86 permute-dot
// kernel (upstream turbovec 1.1.2 search.rs
// `search_multi_query_permute_dot_gfni`). Built with AVX-512 F, BW, VNNI
// and GFNI enabled and called only after a runtime check for them; it sums
// the same integers as the VNNI object's kernel.

#include "ggml-vector-index-x86.h"
#include "ggml-vector-index-turbovec-pd-x86.h"

#include <immintrin.h>

namespace ggml_vec_index_detail {

namespace {

// `c >> 4` per byte in one instruction: a logical shift is linear over
// GF(2), so `vgf2p8affineqb` computes it as a bit-matrix product, and
// unlike `vpsrlw` it needs no follow-up AND (upstream H38).
struct HiNibbleGfni {
    __m512i operator()(__m512i c, __m512i) const {
        return _mm512_gf2p8affine_epi64_epi8(c, _mm512_set1_epi64(static_cast<long long>(0x1020408000000000ull)), 0);
    }
};

} // namespace

void turbovec_permute_dot_sums_gfni(
        const int8_t * levels16,
        const int8_t * const * weights,
        const int32_t * zeros,
        int nq,
        const uint8_t * blocks,
        size_t n_blocks,
        size_t n_byte_groups,
        size_t bytes_after,
        int32_t * sums) {
    permute_dot_sums_dispatch(
        levels16, weights, zeros, nq, blocks, n_blocks, n_byte_groups, bytes_after, sums, HiNibbleGfni{});
}

} // namespace ggml_vec_index_detail
