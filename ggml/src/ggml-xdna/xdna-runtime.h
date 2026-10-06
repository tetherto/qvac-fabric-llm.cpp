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

// The artifacts of one design, as full paths.
struct xdna_artifact {
    std::string xclbin;
    std::string insts;
};

// First search dir that holds `stem`'s artifacts; xclbin is empty when none
// does. `need_insts` also requires the instruction stream in that same dir.
xdna_artifact xdna_artifact_find(const char * stem, bool need_insts);

// xdna_artifact_find(stem, false), loaded into a kernel handle; nullptr when
// the xclbin is absent or the load fails.
xdna_kernel * xdna_kernel_find(xdna_device * dev, const char * stem);

// Validate a TXN stream against its own header, in 32-bit words: word 2 is the
// instruction count and word 3 the total size in bytes. `n_words` is the stream
// length in words, which is also what the run's instruction-count argument
// takes (xrt argument 2), not a byte count.
bool xdna_insts_stream_ok(const char * name, const uint32_t * insts, size_t n_words);

// Read a `.insts.bin` into `words` and validate it as a TXN stream: a size
// that is not whole words, or a header that does not describe the file, is an
// ERROR naming the file, its declared count and byte size, and the actual
// size, instead of a garbage dispatch. Prefer this over reading the artifact
// directly; every stream is checked again by xdna_kernel_bind_insts.
bool xdna_insts_read_file(const char * path, std::vector<uint32_t> & words);

// Bind an in-memory instruction stream (TXN words) to a loaded kernel. `dev`
// must be the same handle used for the data buffers.
bool xdna_kernel_bind_insts(xdna_device * dev, xdna_kernel * kern, const uint32_t * insts, size_t n_words);

// Overwrite a bound stream with one of the same length, for a dispatch whose
// counts and offsets change from run to run while its shape does not. Runs
// read the instruction buffer each time they start.
bool xdna_kernel_rewrite_insts(xdna_kernel * kern, const uint32_t * insts, size_t n_words);

void xdna_kernel_free(xdna_kernel * kern);

// --- buffer ----------------------------------------------------------------

// Allocate a host-visible device buffer object of `bytes` bytes.
xdna_buffer * xdna_buffer_alloc(xdna_device * dev, size_t bytes);

// While one is open, xdna_buffer_alloc carves buffers out of the decode's
// arena (GGML_XDNA_ARENA_MB chunks, 256; 0: never) instead of giving each its
// own BO, so a token's streams can be joined into one command. Open it while
// creating the objects the decode's queued runs name.
struct xdna_arena_scope {
    xdna_arena_scope();
    ~xdna_arena_scope();
};

// Free the arena's chunks once nothing carved out of them is left; false, and
// nothing freed, while a view or a scope still is. What the backend built from
// a model goes with its last context, so the next model's decode starts a
// fresh arena rather than growing the old one.
bool xdna_arena_release(void);

// A window onto an existing buffer, for handing a kernel one slice of a larger
// one: the sequence's descriptors always read from offset zero, so a design
// that walks a buffer in chunks needs a view per chunk rather than an offset
// argument. Shares the parent's memory; free it before the parent.
xdna_buffer * xdna_buffer_sub(xdna_buffer * parent, size_t offset, size_t bytes);

void xdna_buffer_free(xdna_buffer * buf);

// Make device-side writes to a mapped BO visible to the host (cache
// invalidation). Read the data through the BO's own map afterwards. False when
// the buffer is missing or XRT rejected the sync; zero bytes is a no-op.
[[nodiscard]] bool xdna_buffer_sync_from_device(xdna_buffer * buf);
// Invalidate part of a buffer, for a range the array wrote while the host
// may hold older cache lines of it.
[[nodiscard]] bool xdna_buffer_sync_from_device_range(xdna_buffer * buf, size_t bytes, size_t offset);

// The pattern xdna_buffer_mark leaves in a range the device is about to
// overwrite: a quiet NaN with a payload of its own, which no kernel in this
// backend produces.
static constexpr uint32_t XDNA_POISON_F32 = 0x7FC0DEADu;

// Mark the range [offset, offset+bytes) as not yet written by the device, and
// write the pattern through the CPU caches: left dirty, our own lines could
// land on top of the device's output. Mark before the dispatch that writes it.
void xdna_buffer_mark(xdna_buffer * buf, size_t bytes, size_t offset = 0);

// Wait until the device has written every word of [offset, offset+bytes) and
// return the buffer's map at that offset, so the caller reads it in place; or
// nullptr when the pattern survived every pass.
//
// The same evidence as xdna_buffer_download without the copy, for a reader that
// consumes the range once and can walk the device's memory directly - a GEMM's
// C block folded straight into its destination, where a staging copy would
// double the bytes moved.
uint8_t * xdna_buffer_wait_written(xdna_buffer * buf, size_t bytes, size_t offset = 0);

// Copy a marked range out, returning only once every word of it has been
// written by the device, i.e. once the pattern is gone. `dst` must hold
// `bytes`. Returns false when the pattern survived every pass; the copy is made
// either way, so the caller keeps working on the last value it saw.
bool xdna_buffer_download(xdna_buffer * buf, void * dst, size_t bytes, size_t offset = 0);
// Read `bytes` from a buffer the array writes into. A dispatch can report
// completion before its last writes are visible to the host, and a read taken
// in that window hands back - and caches - the previous contents. Read,
// invalidate the cache, read again, and keep reading while the value changes.
// False when a sync failed; `dst` then holds the last read taken.
[[nodiscard]] bool xdna_buffer_read_settled(xdna_buffer * buf, void * dst, size_t bytes, size_t offset = 0);

// Make host-side writes to the mapped memory visible to the NPU.
[[nodiscard]] bool xdna_buffer_sync_to_device(xdna_buffer * buf);

// Flush only part of a buffer. Needed when the device writes into the rest of
// it and a full flush would push the stale host copy over what it wrote.
[[nodiscard]] bool xdna_buffer_sync_to_device_range(xdna_buffer * buf, size_t bytes, size_t offset);

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

// Batched submission. Between xdna_batch_begin(k) and xdna_batch_wait(), a
// run given to xdna_run_submit is not started on its own: it joins an
// xrt::runlist on its kernel's hardware context, and a list goes to the
// device once it holds k runs (and at the wait). The NPU's cost is per
// command - the driver, the firmware and a completion interrupt each - and at
// ~1300 commands a second the SoC's power stays ~8 W above what the same
// work in a few commands costs. Any run started otherwise (xdna_run_restart)
// submits what the batch holds first, so the order on the device is kept.
// Runs that went through a batch are waited by xdna_batch_wait, not one by
// one. Outside a batch xdna_run_submit is xdna_run_restart.
// k < 0 is token mode: the runs' streams are joined into one command, sent
// at the wait or before any run started otherwise (the token's first run
// starts on its own, GGML_XDNA_BATCH_LEAD). A run joins only with its buffers
// given, and the join renames each buffer to the root BO it is a view of -
// the decode's arena (xdna_arena_scope) is what keeps those few.
void xdna_batch_begin(int k);
bool xdna_run_submit(struct xdna_kernel *         kern,
                     xrt::run &                   run,
                     struct xdna_buffer * const * args   = nullptr,
                     size_t                       n_args = 0);
bool xdna_batch_wait(void);
bool xdna_batch_active(void);

// --- kernel pool -----------------------------------------------------------

// Populate `pool->names` from the kernel search dirs. Idempotent per pool.
void xdna_kernel_pool_scan(xdna_kernel_pool * pool);

// Load (or fetch from cache) a kernel for `name` with an in-memory built
// instruction stream: on a miss, load the hw from `xclbin_name` and bind the
// stream. Cached by name and stream content together, so two different
// streams under one name never share a kernel. A failed lookup is cached and
// not retried.
xdna_kernel * xdna_kernel_pool_get_built(xdna_kernel_pool *  pool,
                                         const std::string & name,
                                         const char *        xclbin_name,
                                         const uint32_t *    insts,
                                         size_t              n_words);

// Acquire a device buffer of at least `bytes` bytes, reusing an idle one from
// the pool when possible.
xdna_buffer * xdna_kernel_pool_acquire_buffer(xdna_kernel_pool * pool, size_t bytes);

// Return a buffer to the pool, freeing the LRU entry past the limit.
void xdna_kernel_pool_release_buffer(xdna_kernel_pool * pool, xdna_buffer * buf);
