#pragma once

// Thin runtime over XRT for the AMD XDNA NPU: device, kernel (xclbin + insts)
// and host-visible buffers. Struct definitions live in xdna-types.h.

#include "xdna-types.h"

#include <filesystem>
#include <vector>

// --- device -----------------------------------------------------------

// Open the first NPU. Returns nullptr when no device is available.
xdna_device * xdna_device_open(void);

// --- kernel -------------------------------------------------------------

// Kernel search dirs (the backend dir, the executable dir, cwd).
std::vector<std::filesystem::path> xdna_kernel_search_dirs(void);

// Load an xclbin into a kernel handle without an instruction stream; bind one
// later with xdna_kernel_bind_insts. Returns nullptr on failure.
xdna_kernel * xdna_kernel_load_hw(xdna_device * dev, const char * xclbin_path);

// Bind an in-memory instruction stream (TXN words) to a loaded kernel. `dev`
// must be the same handle used for the data buffers.
bool xdna_kernel_bind_insts(xdna_device * dev, xdna_kernel * kern, const uint32_t * insts, size_t n_words);

void          xdna_kernel_free(xdna_kernel * kern);

// --- buffer -------------------------------------------------------------

// Allocate a host-visible device buffer object of `bytes` bytes.
xdna_buffer * xdna_buffer_alloc(xdna_device * dev, size_t bytes);

// A window onto an existing buffer, for handing a kernel one slice of a larger
// one: the sequence's descriptors always read from offset zero, so a design
// that walks a buffer in chunks needs a view per chunk rather than an offset
// argument. Shares the parent's memory; free it before the parent.
xdna_buffer * xdna_buffer_sub(xdna_buffer * parent, size_t offset, size_t bytes);

void          xdna_buffer_free(xdna_buffer * buf);

// Make device-side writes to a mapped BO visible to the host (cache
// invalidation). Read the data through the BO's own map afterwards.
void          xdna_buffer_sync_from_device(xdna_buffer * buf);

// Make host-side writes to the mapped memory visible to the NPU.
void          xdna_buffer_sync_to_device(xdna_buffer * buf);
// Flush only part of a buffer. Needed when the device writes into the rest of
// it and a full flush would push the stale host copy over what it wrote.
void          xdna_buffer_sync_to_device_range(xdna_buffer * buf, size_t bytes,
                                               size_t offset);

// --- execution ------------------------------------------------------------

// Configure a run without submitting it. When the buffer set never changes,
// building the run once and restarting it avoids rebuilding the command
// buffer on every dispatch, which dominates a small kernel's cost.
xrt::run xdna_kernel_run_make(xdna_kernel * kern, xdna_buffer ** args, size_t n_args);

// Resubmit a run built by xdna_kernel_run_make.
// `tag` names the call site for GGML_XDNA_DESIGN_PROF, which is how the
// decode's submissions are attributed - almost all of them are restarts.
bool xdna_run_restart(xrt::run & run, const char * tag = nullptr);

// Submit a kernel without waiting (so multiple kernels can run back-to-back).
// Returns a run handle that must be waited with xdna_run_wait() before the
// buffers are reused; an empty handle means the submission failed.
xrt::run xdna_kernel_run_start(xdna_kernel * kern, xdna_buffer ** args, size_t n_args);

// GGML_XDNA_RUNLIST_PROBE=N times N dispatches of the same run submitted one
// at a time against the same N submitted as one xrt::runlist, which the device
// executes back to back without returning to the host between them. The
// difference is what a dispatch costs to submit rather than to run, and that
// decides whether the fixed per-dispatch cost is worth merging streams to
// avoid or can simply be batched away.
void xdna_runlist_probe(xdna_device * dev, xdna_kernel * kern,
                        xdna_buffer ** args, size_t n_args);

// Wait for a started run. Returns true on success.
bool xdna_run_wait(xrt::run & run);

// --- kernel pool ----------------------------------------------------------

// Populate `pool->names` from the kernel search dirs. Idempotent per pool.
void xdna_kernel_pool_scan(xdna_kernel_pool * pool);

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
