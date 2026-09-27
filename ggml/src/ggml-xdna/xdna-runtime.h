#pragma once

// Thin runtime over XRT for the AMD XDNA NPU: device, kernel (xclbin + insts),
// host-visible buffers, kernel pool. Struct definitions live in xdna-types.h.

#include "xdna-types.h"

#include <filesystem>
#include <string>
#include <vector>

// --- device ----------------------------------------------------------------

// Open the first NPU. Returns nullptr when no device is available.
xdna_device * xdna_device_open(void);

// --- kernel ----------------------------------------------------------------

// Kernel search dirs (the backend dir, the executable dir, cwd).
std::vector<std::filesystem::path> xdna_kernel_search_dirs(void);

// Load an xclbin into a kernel handle without an instruction stream; bind one
// later with xdna_kernel_bind_insts. Returns nullptr on failure.
xdna_kernel * xdna_kernel_load_hw(xdna_device * dev, const char * xclbin_path);

// Bind an in-memory instruction stream (TXN words) to a loaded kernel. `dev`
// must be the same handle used for the data buffers.
bool xdna_kernel_bind_insts(xdna_device * dev, xdna_kernel * kern, const uint32_t * insts, size_t n_words);

void xdna_kernel_free(xdna_kernel * kern);

// --- buffer ----------------------------------------------------------------

// Allocate a host-visible device buffer object of `bytes` bytes.
xdna_buffer * xdna_buffer_alloc(xdna_device * dev, size_t bytes);

// A window onto an existing buffer, for handing a kernel one slice of a larger
// one: the sequence's descriptors always read from offset zero, so a design
// that walks a buffer in chunks needs a view per chunk rather than an offset
// argument. Shares the parent's memory; free it before the parent.
xdna_buffer * xdna_buffer_sub(xdna_buffer * parent, size_t offset, size_t bytes);

void xdna_buffer_free(xdna_buffer * buf);

// Make device-side writes to a mapped BO visible to the host. On this platform
// XRT never issues the driver's cache-maintenance ioctl for a host_only BO and
// the mapping is ordinary cacheable memory; what it does here is nothing.
void xdna_buffer_sync_from_device(xdna_buffer * buf);

// The pattern xdna_buffer_mark leaves in a range the device is about to
// overwrite: a quiet NaN with a payload of its own, which no kernel in this
// backend produces.
static constexpr uint32_t XDNA_POISON_F32 = 0x7FC0DEADu;

// Mark the range [offset, offset+bytes) as not yet written by the device, and
// write the pattern through the CPU caches: left dirty, our own lines could
// land on top of the device's output.
//
// A dispatch reports completion before its last writes are readable, so a read
// taken right after it can return the previous contents of the buffer. There
// is no barrier for that from the host - the completion is honest and the
// cache ioctl does nothing - so the read has to tell a landed write from a
// pending one, which is what the pattern is for.
void xdna_buffer_mark(xdna_buffer * buf, size_t bytes, size_t offset = 0);

// Wait until the device has written every word of [offset, offset+bytes) and
// return the buffer's map at that offset, so the caller reads it in place; or
// nullptr when the pattern survived every pass.
//
// The same evidence as xdna_buffer_download without the copy, for a reader
// that consumes the range once and can walk the device's memory directly - a
// GEMM's C block folded straight into its destination, where a staging copy
// would double the bytes moved.
uint8_t * xdna_buffer_wait_written(xdna_buffer * buf, size_t bytes,
                                   size_t offset = 0);

// Copy a marked range out, returning only once every word of it has been
// written by the device, i.e. once the pattern is gone. `dst` must hold
// `bytes`. Returns false when the pattern survived every pass; the copy is
// made either way, so the caller keeps working on the last value it saw.
bool xdna_buffer_download(xdna_buffer * buf, void * dst, size_t bytes,
                          size_t offset = 0);

// Make host-side writes to the mapped memory visible to the NPU.
void xdna_buffer_sync_to_device(xdna_buffer * buf);

// Flush only part of a buffer. Needed when the device writes into the rest of
// it and a full flush would push the stale host copy over what it wrote.
void xdna_buffer_sync_to_device_range(xdna_buffer * buf, size_t bytes, size_t offset);

// --- execution -------------------------------------------------------------

// Configure a run without submitting it. When the buffer set never changes,
// building the run once and restarting it avoids rebuilding the command
// buffer on every dispatch, which dominates a small kernel's cost.
xrt::run xdna_kernel_run_make(xdna_kernel * kern, xdna_buffer ** args, size_t n_args);

// Resubmit a run built by xdna_kernel_run_make.
bool xdna_run_restart(xrt::run & run);

// Submit a kernel without waiting (so multiple kernels can run back to back).
// Returns a run handle that must be waited with xdna_run_wait() before the
// buffers are reused; an empty handle means the submission failed.
xrt::run xdna_kernel_run_start(xdna_kernel * kern, xdna_buffer ** args, size_t n_args);

// Wait for a started run. Returns true on success.
bool xdna_run_wait(xrt::run & run);

// --- kernel pool -----------------------------------------------------------

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
