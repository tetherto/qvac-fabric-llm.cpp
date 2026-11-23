#pragma once

#include "common.cuh"

// Row L2 clip, as Inkling 2 builds it from 8 or 9 ops:
//   s = x [* scale];  f = clamp(thr / (sqrt(sum_rows(s*s)) * b_scale + b_bias), lo, hi);  out = s * f
struct ggml_cuda_l2_clip_match {
    const ggml_tensor * x     = nullptr; // [ncols, nrows...], contiguous
    const ggml_tensor * scale = nullptr; // optional per-channel scale, broadcast over the rows of x
    ggml_tensor *       dst   = nullptr;
    float thr     = 0.0f;
    float b_scale = 1.0f;
    float b_bias  = 0.0f;
    float lo      = 0.0f;
    float hi      = 1.0f;
    int   n_nodes = 0;   // nodes of the cgraph that the kernel replaces
};

// matches the clip chain that starts at node i (the optional scale MUL or the SQR)
bool ggml_cuda_match_l2_clip(const ggml_cgraph * cgraph, int i, ggml_cuda_l2_clip_match & m);

void ggml_cuda_op_l2_clip(ggml_backend_cuda_context & ctx, const ggml_cuda_l2_clip_match & m);
