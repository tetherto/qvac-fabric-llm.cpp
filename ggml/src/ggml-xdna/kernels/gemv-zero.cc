// SPDX-License-Identifier: MIT
//
// Zero one core's f32 accumulator before the K loop of the decode GEMV
// (gemv_q4.cc). Separate translation unit: the design compiles one object per
// core function, so two functions sharing a source would link twice.

#include <aie_api/aie.hpp>

#ifndef N_CORE
#define N_CORE 64
#endif

extern "C" void ggml_xdna_gemv_zero(float *out)
{
    constexpr int VEC = 32;
    const aie::vector<float, VEC> z = aie::zeros<float, VEC>();
    for (int j = 0; j < N_CORE / VEC; j++) {
        aie::store_v(out + j * VEC, z);
    }
}
