#include "common.cuh"

#define GGML_CUDA_SSM_CONV_MAX_L2 2

// per-head L2 norms of channel slices of the (silu) conv output (the q/k heads of a gated delta net), computed in the
// conv kernel's epilogue; each slice is [head_dim = 128 channels, n_heads] at channel ch0, its normalized rows go to
// dst with the given strides (in floats) and an optional folded scale. pack[r] (optional, token batches only) also
// receives the normalized rows as bf16 in the FlashInfer layout [n_t][pack_heads][128], head h of the slice repeated
// at every value head hv with hv % n_heads == h (the head mapping of gated_delta_net_flashinfer_prepare8_cuda);
// dst[r] may then be nullptr when the pack is the only copy read
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
    __nv_bfloat16 * pack[GGML_CUDA_SSM_CONV_MAX_L2] = {nullptr, nullptr};
    int     pack_heads = 0;
};

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr,
                           const ggml_cuda_ssm_conv_l2 * l2 = nullptr);

// the conv of one sequence straight from the token-major projection: the first d_conv-1 inputs of a channel come
// from its state row, the rest from x; writes silu(conv + bias) as [n_ch, n_t], the per-head L2 norms of the q/k
// slices, and the next state row (the last d_conv-1 tokens of x). Prefill batches only (n_t above the conv_state_pack
// limit); consecutive threads own consecutive channels so every x read and y write is coalesced
struct ggml_cuda_ssm_conv_tokens_args {
    const float   * states;        // [state_size, mem_size]; row idx[0] holds [n_ch][d_conv-1]
    const int32_t * idx;           // [1]
    const float   * x;             // [n_ch, n_t], token stride x_stride_t floats
    const float   * w;             // [d_conv, n_ch], channel stride w_stride floats
    const float   * bias;          // [n_ch] or nullptr
    float         * y;             // [n_ch, n_t] contiguous (the silu output); nullptr when only the packs are read
    float         * cache;         // [n_ch][d_conv-1] the next state row (may be the row states[idx[0]])
    __nv_bfloat16 * v_pack = nullptr; // optional: channels [v_ch0, v_ch0 + v_nch) of y as bf16 [n_t][v_nch]
    int64_t states_stride;
    int64_t x_stride_t;
    int64_t w_stride;
    int     n_ch;
    int     d_conv;                // 3, 4, 5 or 9
    int     n_t;                   // > GGML_CUDA_CONV_STATE_MAX_T
    int     v_ch0 = 0;             // multiples of 128
    int     v_nch = 0;
};

bool ggml_cuda_ssm_conv_tokens_supported(const ggml_cuda_ssm_conv_tokens_args & a);

void ggml_cuda_op_ssm_conv_tokens(ggml_backend_cuda_context & ctx, const ggml_cuda_ssm_conv_tokens_args & a, const ggml_cuda_ssm_conv_l2 & l2);
