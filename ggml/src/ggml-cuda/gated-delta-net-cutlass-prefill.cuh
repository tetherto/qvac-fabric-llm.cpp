#pragma once

#include "gated-delta-net-cutlass.cuh"

#include <cstddef>

struct ggml_cuda_gdn_chunked_workspace {
    size_t n_chunks;
    size_t w_offset;
    size_t u_offset;
    size_t bytes;
};

bool ggml_cuda_gdn_chunked_init(int device);
bool ggml_cuda_gdn_chunked_workspace_layout(const ggml_cuda_gdn_cute_args &   args,
                                            ggml_cuda_gdn_chunked_workspace & workspace);
bool ggml_cuda_gdn_chunked_launch(const ggml_cuda_gdn_cute_args &         args,
                                  const ggml_cuda_gdn_chunked_workspace & workspace,
                                  cudaStream_t                            stream);
