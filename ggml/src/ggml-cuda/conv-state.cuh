#include "common.cuh"

// the conv input of a recurrent layer and its next state in one launch (get_rows of the state row, concat with the
// new tokens, copy of the last d_conv-1 columns back into the state cache), one sequence, a few tokens:
//   conv_input[c][j] = j < d_conv-1 ? states[idx[0]][c][j] : x[c][j - (d_conv-1)]
//   cache[c][j]      = conv_input[c][n_t + j]
#define GGML_CUDA_CONV_STATE_MAX_T 8

struct ggml_cuda_conv_state_args {
    const float   * states;   // [state_size, mem_size] rows of (d_conv-1) x n_ch
    const int32_t * idx;      // [1] the row of the sequence
    const float   * x;        // [n_ch, n_t]
    float         * conv_in;  // [d_conv-1+n_t, n_ch] contiguous
    float         * cache;    // [state_size] the next state row
    int64_t states_stride;    // row stride in floats
    int64_t x_stride_t;       // token stride in floats
    int64_t x_stride_s;       // unused with one sequence
    int64_t cache_stride;     // unused with one sequence
    int     n_ch;
    int     d_conv;           // 3, 4, 5 or 9
    int     n_t;              // 1 .. GGML_CUDA_CONV_STATE_MAX_T
    int     n_seqs;           // 1
};

bool ggml_cuda_conv_state_supported(const ggml_cuda_conv_state_args & args);

void ggml_cuda_op_conv_state(ggml_backend_cuda_context & ctx, const ggml_cuda_conv_state_args & args);
