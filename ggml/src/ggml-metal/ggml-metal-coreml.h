#pragma once

#include "ggml-metal-device.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ggml_metal_coreml * ggml_metal_coreml_t;

struct ggml_metal_buffer_id ggml_metal_coreml_buffer_id(const struct ggml_tensor * t);

ggml_metal_coreml_t ggml_metal_coreml_init(ggml_metal_device_t dev);
void ggml_metal_coreml_free(ggml_metal_coreml_t ctx);
bool ggml_metal_coreml_candidate(const struct ggml_tensor * op);
bool ggml_metal_coreml_enabled(ggml_metal_coreml_t ctx);
// Call after preceding Metal work completes. On failure recompute op with Metal.
bool ggml_metal_coreml_compute(ggml_metal_coreml_t ctx, struct ggml_tensor * op);
void ggml_metal_coreml_log(ggml_metal_coreml_t ctx, const struct ggml_tensor * op, const char * reason);

#ifdef __cplusplus
}
#endif
