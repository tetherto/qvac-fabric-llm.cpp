#pragma once

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// qvac fork: downstream ggml_op insertions shift the serialized op ids
// relative to upstream, so the wire format is incompatible with stock
// llama.cpp peers even though the message framing is unchanged. Keep the
// major version offset (+100) from upstream's so mismatched peers are
// rejected at the HELLO handshake instead of misdecoding graphs. The
// HELLO fields are uint8_t on the wire, so the value must stay <= 255.
#define RPC_PROTO_MAJOR_VERSION    107
#define RPC_PROTO_MINOR_VERSION    0
#define RPC_PROTO_PATCH_VERSION    0

#ifdef  __cplusplus
static_assert(GGML_OP_COUNT == 113, "GGML_OP_COUNT has changed - update the RPC protocol version");
#endif

#define GGML_RPC_MAX_SERVERS       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device);
GGML_BACKEND_API bool ggml_backend_is_rpc(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device);

GGML_BACKEND_API void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total);

GGML_BACKEND_API void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
                                                    size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_reg(void);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint);

// Connects to `endpoint` and caches the connection without registering it as a backend.
// Different endpoints can connect concurrently; same-endpoint callers share one attempt.
GGML_BACKEND_API bool ggml_backend_rpc_prefetch_connection(const char * endpoint);

#ifdef  __cplusplus
}
#endif
