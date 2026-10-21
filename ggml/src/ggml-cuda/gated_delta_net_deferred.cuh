#include "common.cuh"

// GGML_OP_GATED_DELTA_NET_DEFERRED; base_out != nullptr writes the new base state there (the recurrent cache,
// see ggml_cuda_try_gdn_deferred_cache_fusion) instead of the dst tail. per-seq stride is always D = S_v*S_v*H
void ggml_cuda_op_gated_delta_net_deferred(ggml_backend_cuda_context & ctx, ggml_tensor * dst, float * base_out);

// GGML_OP_GDN_CONV_DEFERRED
void ggml_cuda_op_gdn_conv_deferred(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
