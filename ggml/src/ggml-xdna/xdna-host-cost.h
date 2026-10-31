#pragma once

#include "ggml.h"

#include <stdint.h>

// What ggml-cpu takes for a MUL_MAT of M rows of K against an N x K weight, from probes/host_probe.cpp: graphs of many
// different cold weights run in order, as a pass reads them, on a Ryzen AI MAX+ 395 with GOMP_SPINCOUNT=20000 (the wait policy
// of the run line; a passive wait roughly doubles every figure). One fit a weight type and team size:
//   ms = a + bytes * per_byte + 2 M K N * per_flop + M K * per_mk + M N * per_mn
// whose terms are the dispatch and barrier, the weight read from memory, the arithmetic, the activation's conversion to the
// type the weight's dot product takes, and the result's write. The mean error over the measured shapes (K 256 to 2048, N 16 to
// 12288, 15 to 2560 rows for Q8_0 and BF16, 15 to 128 for the K-quants) is 4 to 12%, the largest 16 to 39%. A team of 8 or more
// threads takes the 16-thread fit, a smaller one the 4-thread fit. F16 and the other types have no fit: a projection of one keeps
// its array route (the model's host time is negative for it).
struct xdna_host_fit {
    double a_ms, per_byte, per_flop, per_mk, per_mn;
};

static constexpr int XDNA_HOST_BIG_TEAM = 8;

static inline const xdna_host_fit * xdna_host_fit_for(enum ggml_type type, int threads) {
    static const xdna_host_fit q8_big  = { 1.04e-3, 1.078e-8, 3.519e-10, 1.697e-7, 1.413e-7 };
    static const xdna_host_fit q8_few  = { 1.20e-3, 3.621e-8, 1.339e-9, 0.0, 2.385e-7 };
    static const xdna_host_fit bf_big  = { 2.70e-3, 7.848e-9, 6.789e-11, 1.808e-7, 2.196e-7 };
    static const xdna_host_fit bf_few  = { 2.48e-3, 1.763e-8, 4.630e-10, 0.0, 3.091e-7 };
    static const xdna_host_fit q4_big  = { 9.23e-3, 1.484e-8, 1.556e-10, 4.407e-7, 0.0 };
    static const xdna_host_fit q4_few  = { 5.18e-3, 3.426e-8, 4.809e-10, 2.704e-7, 0.0 };
    static const xdna_host_fit q5_big  = { 7.58e-3, 1.644e-8, 1.418e-10, 4.608e-7, 0.0 };
    static const xdna_host_fit q5_few  = { 7.11e-3, 4.094e-8, 4.162e-10, 2.825e-7, 0.0 };
    static const xdna_host_fit q6_big  = { 3.41e-3, 2.008e-8, 1.955e-10, 3.803e-7, 0.0 };
    static const xdna_host_fit q6_few  = { 3.08e-3, 4.368e-8, 4.658e-10, 3.172e-7, 0.0 };
    const bool                 big     = threads >= XDNA_HOST_BIG_TEAM;
    switch (type) {
        case GGML_TYPE_Q8_0:
            return big ? &q8_big : &q8_few;
        case GGML_TYPE_BF16:
            return big ? &bf_big : &bf_few;
        case GGML_TYPE_Q4_K:
            return big ? &q4_big : &q4_few;
        case GGML_TYPE_Q5_K:
            return big ? &q5_big : &q5_few;
        case GGML_TYPE_Q6_K:
            return big ? &q6_big : &q6_few;
        default:
            return nullptr;
    }
}

// The host's time for the product in ms, or a negative number when no fit covers the weight type.
static inline double xdna_host_mul_mat_ms(enum ggml_type type, int threads, int64_t M, int64_t K, int64_t N) {
    const xdna_host_fit * f = xdna_host_fit_for(type, threads);
    if (f == nullptr) {
        return -1.0;
    }
    const double bytes = (double) ggml_row_size(type, K) * (double) N;
    return f->a_ms + bytes * f->per_byte + 2.0 * (double) M * (double) K * (double) N * f->per_flop +
           (double) M * (double) K * f->per_mk + (double) M * (double) N * f->per_mn;
}
