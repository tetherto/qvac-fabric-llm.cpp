#pragma once

#include <xrt/xrt_kernel.h>

// The decode's vocabulary projection (lm_head: Qwen3.5 ties it to
// token_embd, Q6_K [1024 x 248320]) on the fused layer design's pool (the
// layers' hardware context: another xclbin costs ~2.5 ms each way). The array
// pulls weights at ~52 GB/s whatever the channel count, so the head's time is
// its bytes: the weight is re-quantized once, at load, to Q4_K and packed as
// q4g32 (0.594 B a value), the density FastFlowLM's head has - past its first
// rows, which stay exact (xdna_head_create).

struct ggml_tensor;
struct xdna_head;
struct xdna_kernel_pool;

struct xdna_buffer;

// Re-quantize (unless it is Q4_K already), pack and load. Null when the
// weight's shape or type does not serve. With `res` and the output norm's
// gamma and epsilon the head can also take its input from the residual rows
// (xdna-gemv.h XDNA_RES_*): rms_norm(F + A) * gamma on the prologue.
xdna_head * xdna_head_create(struct xdna_kernel_pool * pool, const struct ggml_tensor * w,
                             struct xdna_buffer * res = nullptr, const float * gamma = nullptr,
                             float eps = 0.0f);
void        xdna_head_free(xdna_head * h);

// logits = W x for one row x of the weight's K.
bool xdna_head_run(xdna_head * h, const float * x, float * logits);

// Whether the head takes its input from the rows; start it so, without
// waiting, and read its logits once the returned run has been waited for.
bool       xdna_head_rows(const xdna_head * h);
xrt::run * xdna_head_start_rows(xdna_head * h);
void       xdna_head_read(xdna_head * h, float * logits);
