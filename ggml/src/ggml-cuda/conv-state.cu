#include "conv-state.cuh"

#define CONV_STATE_BLOCK 256

// one thread per channel (one sequence): the state and token values are read into registers before anything is
// written, so the state row may be updated in place
template <int d_conv>
static __global__ void conv_state_pack(const ggml_cuda_conv_state_args a) {
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= a.n_ch) {
        return;
    }
    constexpr int n_state = d_conv - 1;
    const int     n_cols  = n_state + a.n_t;

    float v[n_state + GGML_CUDA_CONV_STATE_MAX_T];
    const float * st = a.states + (int64_t) a.idx[0]*a.states_stride + (int64_t) c*n_state;
#pragma unroll
    for (int j = 0; j < n_state; ++j) {
        v[j] = st[j];
    }
    for (int t = 0; t < a.n_t; ++t) {
        v[n_state + t] = a.x[(int64_t) t*a.x_stride_t + c];
    }

    float * out = a.conv_in + (int64_t) c*n_cols;
    for (int j = 0; j < n_cols; ++j) {
        out[j] = v[j];
    }
    float * next = a.cache + (int64_t) c*n_state;
#pragma unroll
    for (int j = 0; j < n_state; ++j) {
        next[j] = v[a.n_t + j];
    }
}

bool ggml_cuda_conv_state_supported(const ggml_cuda_conv_state_args & args) {
    const bool d_conv_ok = args.d_conv == 3 || args.d_conv == 4 || args.d_conv == 5 || args.d_conv == 9;
    return d_conv_ok && args.n_seqs == 1 && args.n_t >= 1 && args.n_t <= GGML_CUDA_CONV_STATE_MAX_T;
}

void ggml_cuda_op_conv_state(ggml_backend_cuda_context & ctx, const ggml_cuda_conv_state_args & args) {
    GGML_ASSERT(ggml_cuda_conv_state_supported(args));
    const dim3 grid((args.n_ch + CONV_STATE_BLOCK - 1)/CONV_STATE_BLOCK, 1, 1);
    switch (args.d_conv) {
        case 3: conv_state_pack<3><<<grid, CONV_STATE_BLOCK, 0, ctx.stream()>>>(args); break;
        case 4: conv_state_pack<4><<<grid, CONV_STATE_BLOCK, 0, ctx.stream()>>>(args); break;
        case 5: conv_state_pack<5><<<grid, CONV_STATE_BLOCK, 0, ctx.stream()>>>(args); break;
        case 9: conv_state_pack<9><<<grid, CONV_STATE_BLOCK, 0, ctx.stream()>>>(args); break;
        default: GGML_ABORT("conv_state: unsupported d_conv %d", args.d_conv);
    }
}
