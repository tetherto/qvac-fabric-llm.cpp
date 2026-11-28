#pragma once

#include "common.cuh"

// Row L2 clip of Inkling 2: [x*w ->] l2_norm(., eps) -> scale(t) [-> *f, one f per token]
struct ggml_cuda_l2_clip_match {
    const ggml_tensor * x = nullptr; // [ncols, nrows...], contiguous
    const ggml_tensor * w = nullptr; // optional per-column weight, broadcast over the rows of x
    const ggml_tensor * f = nullptr; // optional factor of shape [1, 1, ne2, ne3], broadcast over each token's rows
    ggml_tensor *       dst = nullptr;
    float eps     = 0.0f;
    float t       = 1.0f;
    float t_bias  = 0.0f;
    int   n_nodes = 0;   // nodes of the cgraph that the kernel replaces
};

// matches the clip chain that starts at node i (the optional weight MUL or the L2_NORM)
bool ggml_cuda_match_l2_clip(const ggml_cgraph * cgraph, int i, ggml_cuda_l2_clip_match & m);

void ggml_cuda_op_l2_clip(ggml_backend_cuda_context & ctx, const ggml_cuda_l2_clip_match & m);
