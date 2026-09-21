#pragma once

// SSM_CONV (conv1d) offload via kernels/conv.py (conv_prefill_* artifact),
// ported from the qvac-fabric max2 branch. Causal depthwise KW=4 conv over the
// 6144 gated-delta-net channels: 512 channels (8 cols x 4 rows x 16) and 256
// tokens per tile, token-major layout so the host neither gathers nor scatters
// across channels. Auto-active when the artifact is present; anything outside
// the baked geometry stays on the CPU.

#include "ggml.h"

struct xdna_device;

// True when the xclbin + insts artifacts are present.

// True when `node` is a GGML_OP_SSM_CONV prefill op the kernel covers.
bool xdna_conv_prefill_supported(const struct ggml_tensor * node);

// Mark a GGML_OP_CONCAT node as safe to read through: `run` then packs the x
// window from the concat's two sources instead of from its (unmaterialised)
// result. The caller must have checked that the concat feeds this conv with
// nothing in between - see xdna_concat_tail_plan in ggml-xdna.cpp. The set is
// per graph; reset it at the top of every graph_compute.
// `hist` is a caller-owned snapshot of the concat's first source, [rows][nr]
// with the channels contiguous, taken before the graph's CPY overwrites the
// conv state with this step's tail. It must outlive the graph.
void xdna_conv_prefill_direct_reset(void);
void xdna_conv_prefill_direct_add(const struct ggml_tensor * concat,
                                  const float * hist, int64_t rows);

// Run the op: tile the conv over channel groups and token tiles, pack x/w into
// the token-major xw buffer, run MB-tile batches, scatter out into the ggml
// dst. Synchronous. Returns false on failure (caller keeps CPU fallback).
bool xdna_conv_prefill_run(struct xdna_device * dev, struct ggml_tensor * node);
