#pragma once

// Thin runtime over XRT for the AMD XDNA NPU: device, kernel (xclbin + insts)
// and host-visible buffers. Struct definitions live in xdna-types.h.

#include "xdna-types.h"

// --- device -----------------------------------------------------------

// Open the first NPU. Returns nullptr when no device is available.
xdna_device * xdna_device_open(void);

void          xdna_device_close(xdna_device * dev);

// --- kernel -------------------------------------------------------------

// Load an xclbin + instruction stream into a kernel. xclbins are loaded once
// per uuid; subsequent loads reuse the shared hw_context. Returns nullptr on
// failure.
xdna_kernel * xdna_kernel_load(xdna_device * dev, const char * xclbin_path, const char * insts_path);

// Like xdna_kernel_load, but resolve `xclbin_name`/`insts_name` against the
// kernel search dirs (GGML_XDNA_KERNELS_DIR, the backend dir, the executable
// dir, cwd).
xdna_kernel * xdna_kernel_load_search(xdna_device * dev, const char * xclbin_name, const char * insts_name);

void          xdna_kernel_free(xdna_kernel * kern);

// --- buffer -------------------------------------------------------------

// Allocate a host-visible device buffer object of `bytes` bytes.
xdna_buffer * xdna_buffer_alloc(xdna_device * dev, size_t bytes);

void          xdna_buffer_free(xdna_buffer * buf);

// Copy host -> device (memcpy + sync).
void          xdna_buffer_write(xdna_buffer * buf, const void * host, size_t bytes);

// Copy device -> host (sync + memcpy).
void          xdna_buffer_read(xdna_buffer * buf, void * host, size_t bytes);

// Make host-side writes to the mapped memory visible to the NPU.
void          xdna_buffer_sync_to_device(xdna_buffer * buf);

// --- execution ------------------------------------------------------------

// Run a kernel with `n_args` host buffers (args 3.. onward; 0=opcode,
// 1=instruction BO, 2=ninstr). Sizes are baked into the instruction stream.
bool xdna_kernel_run(xdna_kernel * kern, xdna_buffer ** args, size_t n_args);
