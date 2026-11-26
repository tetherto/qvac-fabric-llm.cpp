#ifndef HTP_TIMESTEP_OPS_H
#define HTP_TIMESTEP_OPS_H

#include <stdint.h>

struct htp_timestep_kernel_params {
    uint32_t dim;
    uint32_t n_timesteps;
    float    log_max_period;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_timestep_kernel_params) <= 128, "htp_timestep_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_timestep_kernel_params) <= 128, "htp_timestep_kernel_params is too large for kernel_params blob");
#endif

#endif /* HTP_TIMESTEP_OPS_H */
