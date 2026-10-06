#include "common.cuh"

#define CUDA_LSTM_CELL_BLOCK_SIZE 256

void ggml_cuda_op_lstm_cell(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
