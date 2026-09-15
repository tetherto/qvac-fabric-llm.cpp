#include "common.cuh"

// the two gate projections of a gated delta net at small batch (one warp per output), with their epilogues:
//   g = softplus(x . w_alpha + dt_bias) * a      b = sigmoid(x . w_beta)
// x [k, n_tokens] F32 rows, w_alpha / w_beta [k, n] F16 or BF16 rows, dt_bias / a [n] F32, g / b [n, n_tokens] F32
bool ggml_cuda_gdn_gates_supported(const ggml_tensor * x, const ggml_tensor * w_alpha, const ggml_tensor * w_beta,
                                   const ggml_tensor * dt_bias, const ggml_tensor * a, const ggml_tensor * g, const ggml_tensor * b);

void ggml_cuda_op_gdn_gates(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * w_alpha, const ggml_tensor * w_beta,
                            const ggml_tensor * dt_bias, const ggml_tensor * a, ggml_tensor * g, ggml_tensor * b);
