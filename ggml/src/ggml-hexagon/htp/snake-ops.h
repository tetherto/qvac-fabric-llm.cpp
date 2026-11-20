#ifndef HTP_SNAKE_OPS_H
#define HTP_SNAKE_OPS_H

#include <stdint.h>

// The SNAKE kernel derives its split from octx->n_threads; the host solver only
// gates support, so the params blob carries nothing the kernel reads.
struct htp_snake_kernel_params {
    uint32_t n_threads;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_snake_kernel_params) <= 128, "htp_snake_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_snake_kernel_params) <= 128, "htp_snake_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_SNAKE_OPS_H
