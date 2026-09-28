//
// The lead core of a two-row decode GEMV pair (gemv_q4.py's PAIR) copies its
// partner's block of outputs after its own, so the column leaves the array as
// one stream. The partner's block arrives through the memory the two tiles
// share. Separate translation unit, like gemv-zero.cc: the design compiles one
// object per core function.

#include <aie_api/aie.hpp>

#ifndef N_CORE
#    define N_CORE 64
#endif

extern "C" void ggml_xdna_gemv_merge(const float * p, float * o) {
    constexpr int VEC = 16;
    for (int j = 0; j < N_CORE / VEC; j++) {
        aie::store_v(o + N_CORE + j * VEC, aie::load_v<VEC>(p + j * VEC));
    }
}
