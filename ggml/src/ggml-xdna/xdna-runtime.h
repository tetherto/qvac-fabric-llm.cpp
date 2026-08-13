#pragma once

// Thin runtime over XRT for the AMD XDNA NPU: device, kernel (xclbin + insts)
// and host-visible buffers. Struct definitions live in xdna-types.h.

#include "xdna-types.h"

#include <filesystem>
#include <vector>

// --- device -----------------------------------------------------------

// Open the first NPU. Returns nullptr when no device is available.
xdna_device * xdna_device_open(void);

void          xdna_device_close(xdna_device * dev);

// --- kernel -------------------------------------------------------------

// Kernel search dirs (GGML_XDNA_KERNELS_DIR, the backend dir, the executable
// dir, cwd).
std::vector<std::filesystem::path> xdna_kernel_search_dirs(void);

// Load an xclbin + instruction stream into a kernel. xclbins are loaded once
// per uuid; subsequent loads reuse the shared hw_context. Returns nullptr on
// failure.
xdna_kernel * xdna_kernel_load(xdna_device * dev, const char * xclbin_path, const char * insts_path);

// Load an xclbin into a kernel handle without an instruction stream; bind one
// later with xdna_kernel_bind_insts. Returns nullptr on failure.
xdna_kernel * xdna_kernel_load_hw(xdna_device * dev, const char * xclbin_path);

// Bind an in-memory instruction stream (TXN words) to a loaded kernel. `dev`
// must be the same handle used for the data buffers.
bool xdna_kernel_bind_insts(xdna_device * dev, xdna_kernel * kern, const uint32_t * insts, size_t n_words);

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

// Submit a kernel without waiting (so multiple kernels can run back-to-back).
// Returns a run handle that must be waited with xdna_run_wait() before the
// buffers are reused; an empty handle means the submission failed.
xrt::run xdna_kernel_run_start(xdna_kernel * kern, xdna_buffer ** args, size_t n_args);

// Wait for a started run. Returns true on success.
bool xdna_run_wait(xrt::run & run);

// --- kernel pool ----------------------------------------------------------

// Populate `pool->names` from the kernel search dirs. Idempotent per pool.
void xdna_kernel_pool_scan(xdna_kernel_pool * pool);

// Load (or fetch from cache) the kernel for `name`. A failed lookup is
// cached and not retried.
xdna_kernel * xdna_kernel_pool_get(xdna_kernel_pool * pool, const std::string & name);

// Load (or fetch from cache) a kernel for `name` with an in-memory built
// instruction stream: on a miss, load the hw from `xclbin_name` and bind the
// stream. A failed lookup is cached and not retried.
xdna_kernel * xdna_kernel_pool_get_built(xdna_kernel_pool * pool, const std::string & name,
                                         const char * xclbin_name, const uint32_t * insts, size_t n_words);

// Acquire a device buffer of at least `bytes` bytes, reusing an idle one from
// the pool when possible.
xdna_buffer * xdna_kernel_pool_acquire_buffer(xdna_kernel_pool * pool, size_t bytes);

// Return a buffer to the pool, freeing the LRU entry past the limit.
void xdna_kernel_pool_release_buffer(xdna_kernel_pool * pool, xdna_buffer * buf);

void xdna_kernel_pool_clear(xdna_kernel_pool * pool);
