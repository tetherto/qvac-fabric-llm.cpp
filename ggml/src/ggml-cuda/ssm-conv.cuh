#include "common.cuh"

#define GGML_CUDA_SSM_CONV_MAX_L2 2

// per-head L2 norms of channel slices of the (silu) conv output (the q/k heads of a gated delta net), computed in the
// conv kernel's epilogue; each slice is [head_dim = 128 channels, n_heads] at channel ch0, its normalized rows go to
// dst with the given strides (in floats) and an optional folded scale
struct ggml_cuda_ssm_conv_l2 {
    int     n = 0;
    int     ch0[GGML_CUDA_SSM_CONV_MAX_L2];
    int     ch1[GGML_CUDA_SSM_CONV_MAX_L2];
    float   eps[GGML_CUDA_SSM_CONV_MAX_L2];
    float   scale[GGML_CUDA_SSM_CONV_MAX_L2];
    float * dst[GGML_CUDA_SSM_CONV_MAX_L2];
    int64_t nb1[GGML_CUDA_SSM_CONV_MAX_L2]; // head stride
    int64_t nb2[GGML_CUDA_SSM_CONV_MAX_L2]; // token stride
    int64_t nb3[GGML_CUDA_SSM_CONV_MAX_L2]; // sequence stride
};

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr,
                           const ggml_cuda_ssm_conv_l2 * l2 = nullptr);
