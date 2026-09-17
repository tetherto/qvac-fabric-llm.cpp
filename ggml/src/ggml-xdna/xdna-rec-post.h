#pragma once

#include "xdna-runtime.h"

#include <cstdint>

// The FFN transition as its own xclbin (kernels/post_norm.py): one core, one
// input object, one output object. The transition runs on the array between
// the fused dispatch and the FFN pair - three dispatches a layer for now, but
// everything the layer needs is on the NPU.
struct xdna_rec_post {
    xdna_device * dev   = nullptr;
    struct xdna_rec_post_elf * elf_handle = nullptr;  // full-ELF state
    xdna_buffer * in    = nullptr;   // [so_out 1024][residual 1024][gamma 1024][flags]
    xdna_buffer * out   = nullptr;   // [hattn 1024][header][8 activation tiles]
    bool          ok    = false;
};

xdna_rec_post * xdna_rec_post_create(struct xdna_kernel_pool * pool);
bool xdna_rec_post_run(xdna_rec_post * p);
void xdna_rec_post_free(xdna_rec_post * p);
