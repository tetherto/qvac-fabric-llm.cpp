#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_hexagon_init(void);

GGML_BACKEND_API bool ggml_backend_is_hexagon(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_hexagon_reg(void);

// Optional registry procedure "ggml_backend_hexagon_get_runtime_info_v1".
// Unlike environment variables, these values describe the initialized sessions.
#define GGML_HEXAGON_RUNTIME_INFO_VERSION 1
struct ggml_backend_hexagon_runtime_info_v1 {
    uint32_t version;
    uint32_t op_stage;       // bit 0: submit to DSP, bit 1: execute on DSP
    uint32_t session_count;  // successfully initialized registry sessions
    uint32_t reserved;
    uint64_t host_bytes_bound;
    uint64_t shared_bytes;
    uint64_t max_buffer_bytes; // effective payload cap, excludes tensor guard
    uint64_t max_batch_vmem_bytes;
    uint64_t vtcm_bytes;
    uint64_t tensor_guard_bytes;
    uint64_t buffer_alignment;
    uint64_t allocation_page_bytes;
};

// host_bytes_bound covers retained host metadata and prospective metadata for
// graphs with at most max_graph_nodes nodes, across all registry sessions.
// shared_bytes covers known, page-rounded session queue allocations. Neither
// includes model/graph tensor buffers, allocator bookkeeping or opaque vendor
// driver/DSP heaps; callers must retain an appropriate memory margin. VTCM and
// max_batch_vmem_bytes are separate limits, not additional Android RAM pools.
// Snapshots are synchronized with graph execution. The caller must serialize
// later work if it needs the snapshot to remain valid. False leaves *out intact
// (wrong backend, wrong info_size, overflow or max_graph_nodes > INT32_MAX).
// See docs/hexagon-runtime-info.md for the host allocation policy and limits.
typedef bool (*ggml_backend_hexagon_get_runtime_info_v1_t)(
    ggml_backend_t backend, uint64_t max_graph_nodes,
    struct ggml_backend_hexagon_runtime_info_v1 * out, size_t info_size);
GGML_BACKEND_API bool ggml_backend_hexagon_get_runtime_info_v1(
    ggml_backend_t backend, uint64_t max_graph_nodes,
    struct ggml_backend_hexagon_runtime_info_v1 * out, size_t info_size);

// Optional registry procedure "ggml_backend_hexagon_conv_taps_f32". Weights
// are contiguous F16 [in, out, taps], input is padded F32 [in, time]. Positive
// tap_step advances through a causal input; -1 reads delayed transpose groups.
// Returns NULL when the shape, scratch budget or working-set size is unsuitable.
typedef struct ggml_tensor * (*ggml_backend_hexagon_conv_taps_f32_t)(
    struct ggml_context *, ggml_backend_t, struct ggml_tensor *, struct ggml_tensor *, int, int);
GGML_BACKEND_API struct ggml_tensor * ggml_backend_hexagon_conv_taps_f32(
    struct ggml_context * ctx, ggml_backend_t backend, struct ggml_tensor * weights,
    struct ggml_tensor * input, int tap_step, int output_rows);

#ifdef  __cplusplus
}
#endif
