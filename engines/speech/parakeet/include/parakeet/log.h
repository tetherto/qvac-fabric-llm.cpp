#pragma once

// Install a ggml_log_callback for libqvac-parakeet; nullptr restores stderr.
//
// ggml's messages are forwarded to the same sink (parakeet installs its own
// forwarding callback with ggml_log_set). Uses ggml.h types.
//
// Blocking: parakeet_log_set returns only after every callback still running
// on the previous sink has returned, on any thread (background Core ML loads
// can log after their engine is gone), so the previous user data may be freed
// as soon as it returns. It does not wait for the calling thread's own
// callbacks: a callback may log, or replace or clear the sink itself.

#include "export.h"

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

PARAKEET_API void parakeet_log_set(ggml_log_callback cb,
                                             void *            user_data);

#ifdef __cplusplus
}
#endif
