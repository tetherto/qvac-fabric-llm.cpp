#pragma once

#include "common.cuh"

// Generic elementwise fusion: groups of pointwise, row-norm, concat and ssm_conv nodes are
// compiled at runtime with NVRTC into one kernel each. Build with GGML_CUDA_NVRTC_FUSION,
// disable at runtime with GGML_CUDA_NVRTC_FUSION=0.

struct ggml_cuda_nvrtc_plan;

#include <vector>

// an FP8 activation group with a static input scale: a generated kernel that stores src1 can also write
// the e4m3 copy that the group's matmuls read
struct ggml_cuda_nvrtc_fp8_sink {
    const ggml_tensor *              src1        = nullptr;
    uint8_t *                        values      = nullptr; // row-major like src1
    float *                          scale       = nullptr; // receives input_scale
    float                            input_scale = 0.0f;
    int                              min_anchor  = 0;       // first node at which the workspace is free
    std::vector<const ggml_tensor *> matmuls;               // nodes that read the e4m3 copy
    int                              group       = -1;

    // set by the plan
    bool fulfilled    = false;
    bool src1_written = true;
};

#ifdef GGML_CUDA_NVRTC_FUSION

bool ggml_cuda_nvrtc_fusion_enabled();

// graph_optimize: keep the inputs of each fused group alive until the group runs
void ggml_cuda_nvrtc_fusion_add_alloc_deps(
        const ggml_cgraph * cgraph, void (*add_alloc_dep)(void *, ggml_tensor *, ggml_tensor *), void * user_data);

// build the plan for one evaluation of cgraph (before CUDA graph capture), nullptr if nothing fuses
ggml_cuda_nvrtc_plan * ggml_cuda_nvrtc_fusion_plan(
        ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, std::vector<ggml_cuda_nvrtc_fp8_sink> * sinks);
void ggml_cuda_nvrtc_fusion_free(ggml_cuda_nvrtc_plan * plan);

// true if node i is handled by the plan: its group was launched here or it is computed by a group
bool ggml_cuda_nvrtc_fusion_run(ggml_cuda_nvrtc_plan * plan, ggml_backend_cuda_context & ctx, int i);

// GGML_CUDA_NVRTC_VERIFY=1: compare the groups that end at or before last_done against the normal path
void ggml_cuda_nvrtc_fusion_after(ggml_cuda_nvrtc_plan * plan, ggml_backend_cuda_context & ctx, int last_done);

// index of the first node after i that the plan handles (cgraph->n_nodes if none)
int ggml_cuda_nvrtc_fusion_next_claimed(const ggml_cuda_nvrtc_plan * plan, int i);

#endif // GGML_CUDA_NVRTC_FUSION
