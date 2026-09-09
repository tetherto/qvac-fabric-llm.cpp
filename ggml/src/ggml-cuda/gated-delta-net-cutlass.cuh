#pragma once

#include "vendors/cuda.h"

#include <cstddef>
#include <cstdint>

struct ggml_cuda_gdn_cute_args {
    const float * q;
    const float * k;
    const float * v;
    const float * g;
    const float * beta;
    const float * state;
    float *       dst;
    float *       state_out;
    void *        workspace;
    size_t        workspace_size;

    int64_t H;
    int64_t H_k;
    int64_t n_tokens;
    int64_t n_seqs;
    int64_t rq3;
    int64_t sq1;
    int64_t sq2;
    int64_t sq3;
    int64_t sv1;
    int64_t sv2;
    int64_t sv3;
    int64_t sb1;
    int64_t sb2;
    int64_t sb3;
    float   scale;
    bool    eligible;
};

bool ggml_cuda_gdn_cute_available(int device, const ggml_cuda_gdn_cute_args & args);
size_t ggml_cuda_gdn_cute_get_alloc_size(int device, const ggml_cuda_gdn_cute_args & args, size_t logical_size);
bool ggml_cuda_gdn_cute_launch(int device, const ggml_cuda_gdn_cute_args & args, cudaStream_t stream);
