#include "xdna-runtime.h"

#include "ggml-impl.h"
#include "xdna-seq.h"
#include "xdna-util.h"

#include <xrt/experimental/xrt_kernel.h>
#include <xrt/experimental/xrt_xclbin.h>
#include <xrt/xrt_kernel.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#include <map>
#include <memory>
#include <thread>
#include <mutex>
#include <system_error>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

// A TXN stream is a 4-word header plus fixed-size instructions: word 2 is the
// instruction count and word 3 the total size in bytes (xdna-seq.h). A stream
// whose header does not describe it would dispatch garbage, so it is rejected
// before anything reaches the device. `name` identifies the stream.
bool xdna_insts_stream_ok(const char * name, const uint32_t * insts, size_t n_words) {
    const auto fail = [&](const char * reason, uint32_t n_instr, uint32_t bytes) {
        GGML_LOG_ERROR("%s: instruction stream %s: %s (%u instructions and %u bytes declared, %zu bytes actual)\n",
                       "xdna-runtime", name, reason, n_instr, bytes, n_words * sizeof(uint32_t));
        return false;
    };
    if (n_words < 4) {
        return fail("too small for a TXN header", 0, 0);
    }
    if ((uint64_t) insts[3] != (uint64_t) n_words * sizeof(uint32_t)) {
        return fail("header declares a different size", insts[2], insts[3]);
    }
    if ((uint64_t) insts[2] * 4 > (uint64_t) (n_words - 4)) {
        return fail("header declares more instructions than fit", insts[2], insts[3]);
    }
    return true;
}

// NPU kernel ABI: arg 0 is the opcode, where 3 = RUN the instruction stream.
static constexpr int XAIE_NPU_OPCODE_RUN = 3;

// Loaded xclbins: uuid -> shared hw_context. All variants of one xclbin share
// a context; kernels hold a shared_ptr so contexts outlive every kernel that
// references them.
static std::mutex                                            g_ctx_mutex;
static std::map<xrt::uuid, std::shared_ptr<xrt::hw_context>> g_ctxs;

// --- device ----------------------------------------------------------------

xdna_device * xdna_device_open(void) {
    xdna_device * dev = new xdna_device;
    try {
        dev->device      = xrt::device(0);
        // one NPU per system: named like the other backends' devices (CUDA0,
        // Vulkan0), so --device takes it unquoted; XRT's name goes in the
        // description
        dev->name        = "XDNA0";
        dev->description = dev->device.get_info<xrt::info::device::name>() + " (" +
                           dev->device.get_info<xrt::info::device::bdf>() + ")";
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: failed to open NPU device: %s\n", "xdna-runtime", e.what());
        delete dev;
        return nullptr;
    }
    return dev;
}

// --- kernel ----------------------------------------------------------------

std::vector<fs::path> xdna_kernel_search_dirs(void) {
    std::vector<fs::path> dirs;
#ifdef GGML_BACKEND_DIR
    dirs.emplace_back(GGML_BACKEND_DIR);
#endif
#ifdef __linux__
    std::error_code ec;
    fs::path        exe = fs::read_symlink("/proc/self/exe", ec);
    if (!ec) {
        dirs.push_back(exe.parent_path());
    }
#endif
    dirs.emplace_back(fs::current_path());
    return dirs;
}

xdna_kernel * xdna_kernel_load_hw(xdna_device * dev, const char * xclbin_path) {
    if (!xclbin_path || !*xclbin_path) {
        GGML_LOG_ERROR("%s: kernel load: no xclbin path\n", "xdna-runtime");
        return nullptr;
    }
    if (!dev) {
        GGML_LOG_ERROR("%s: kernel load %s: no device\n", "xdna-runtime", xclbin_path);
        return nullptr;
    }

    try {
        const std::string xclbin_path_str(xclbin_path);
        const xrt::xclbin xclbin{ xclbin_path_str };
        dev->device.register_xclbin(xclbin);
        const xrt::uuid xclbin_uuid = xclbin.get_uuid();

        // Reuse the hw_context for this xclbin across all kernel variants.
        std::shared_ptr<xrt::hw_context> context;
        {
            std::lock_guard<std::mutex> lock(g_ctx_mutex);
            auto                        it = g_ctxs.find(xclbin_uuid);
            if (it != g_ctxs.end()) {
                context = it->second;
            } else {
                context = std::make_shared<xrt::hw_context>(dev->device, xclbin_uuid);
                g_ctxs.emplace(xclbin_uuid, context);
            }
        }

        const auto kernels = xclbin.get_kernels();
        if (kernels.empty()) {
            GGML_LOG_ERROR("%s: kernel load %s: the xclbin defines no kernel\n", "xdna-runtime", xclbin_path);
            return nullptr;
        }

        xdna_kernel * kern = new xdna_kernel;
        kern->device       = dev->device;
        kern->context      = context;
        kern->kernel       = xrt::kernel(*context, kernels[0].get_name());
        kern->xclbin_name  = xclbin_path_str;
        return kern;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: failed to load kernel %s: %s\n", "xdna-runtime", xclbin_path, e.what());
        return nullptr;
    }
}

xdna_artifact xdna_artifact_find(const char * stem, bool need_insts) {
    const std::string base = stem ? stem : "";
    xdna_artifact     art;
    for (const fs::path & dir : xdna_kernel_search_dirs()) {
        std::error_code   ec;
        const std::string xclbin = (dir / (base + ".xclbin")).string();
        if (!fs::exists(xclbin, ec)) {
            continue;
        }
        if (need_insts) {
            const std::string insts = (dir / (base + ".insts.bin")).string();
            if (!fs::exists(insts, ec)) {
                continue;
            }
            art.insts = insts;
        }
        art.xclbin = xclbin;
        break;
    }
    return art;
}

xdna_kernel * xdna_kernel_find(xdna_device * dev, const char * stem) {
    const xdna_artifact art = xdna_artifact_find(stem, false);
    if (art.xclbin.empty()) {
        return nullptr;
    }
    return xdna_kernel_load_hw(dev, art.xclbin.c_str());
}

bool xdna_insts_read_file(const char * path, std::vector<uint32_t> & out) {
    out.clear();
    if (!path || !*path) {
        GGML_LOG_ERROR("%s: instruction stream read: no path\n", "xdna-runtime");
        return false;
    }
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        GGML_LOG_ERROR("%s: cannot open instruction stream %s\n", "xdna-runtime", path);
        return false;
    }
    f.seekg(0, std::ios::end);
    const std::streamoff n = f.tellg();
    if (n <= 0) {
        GGML_LOG_ERROR("%s: instruction stream %s: %lld bytes, empty\n", "xdna-runtime", path, (long long) n);
        return false;
    }
    if (n % (std::streamoff) sizeof(uint32_t) != 0) {
        GGML_LOG_ERROR("%s: instruction stream %s: %lld bytes is not a multiple of %zu\n", "xdna-runtime", path,
                       (long long) n, sizeof(uint32_t));
        return false;
    }
    f.seekg(0, std::ios::beg);
    out.resize((size_t) n / sizeof(uint32_t));
    if (!f.read((char *) out.data(), n)) {
        GGML_LOG_ERROR("%s: instruction stream %s: read of %lld bytes failed\n", "xdna-runtime", path, (long long) n);
        out.clear();
        return false;
    }
    if (!xdna_insts_stream_ok(path, out.data(), out.size())) {
        out.clear();
        return false;
    }
    return true;
}

bool xdna_kernel_bind_insts(xdna_device * dev, xdna_kernel * kern, const uint32_t * insts, size_t n_words) {
    if (!dev || !kern || !insts || n_words == 0) {
        GGML_LOG_ERROR("%s: bind insts: null device/kernel/insts or empty stream\n", "xdna-runtime");
        return false;
    }
    if (!xdna_insts_stream_ok(kern->xclbin_name.c_str(), insts, n_words)) {
        return false;
    }
    try {
        kern->insts_bytes = (int64_t) (n_words * sizeof(uint32_t));
        kern->insts_bo =
            xrt::bo(dev->device, (size_t) kern->insts_bytes, xrt::bo::flags::cacheable, kern->kernel.group_id(1));
        void * map = kern->insts_bo.map();
        if (!map) {
            GGML_LOG_ERROR("%s: instruction stream for %s: %zu words, the BO has no host mapping\n", "xdna-runtime",
                           kern->xclbin_name.c_str(), n_words);
            kern->insts_bytes = 0;
            kern->insts_host.clear();
            return false;
        }
        std::memcpy(map, insts, (size_t) kern->insts_bytes);
        kern->insts_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        kern->insts_host.assign(insts, insts + n_words);
        return true;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: failed to bind instruction stream for %s: %s\n", "xdna-runtime", kern->xclbin_name.c_str(),
                       e.what());
        kern->insts_bytes = 0;
        kern->insts_host.clear();
        return false;
    }
}

bool xdna_kernel_rewrite_insts(xdna_kernel * kern, const uint32_t * insts, size_t n_words) {
    if (!kern || !insts) {
        GGML_LOG_ERROR("%s: rewrite insts: null kernel/insts\n", "xdna-runtime");
        return false;
    }
    if ((int64_t) (n_words * sizeof(uint32_t)) != kern->insts_bytes) {
        GGML_LOG_ERROR("%s: rewrite insts for %s: %zu words does not match the bound %lld bytes\n", "xdna-runtime",
                       kern->xclbin_name.c_str(), n_words, (long long) kern->insts_bytes);
        return false;
    }
    if (!xdna_insts_stream_ok(kern->xclbin_name.c_str(), insts, n_words)) {
        return false;
    }
    try {
        void * map = kern->insts_bo.map();
        if (!map) {
            GGML_LOG_ERROR("%s: instruction stream for %s: %zu words, the BO has no host mapping\n", "xdna-runtime",
                           kern->xclbin_name.c_str(), n_words);
            return false;
        }
        std::memcpy(map, insts, (size_t) kern->insts_bytes);
        kern->insts_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        kern->insts_host.assign(insts, insts + n_words);
        return true;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: failed to rewrite instruction stream for %s: %s\n", "xdna-runtime",
                       kern->xclbin_name.c_str(), e.what());
        return false;
    }
}

void xdna_kernel_free(xdna_kernel * kern) {
    delete kern;
}

// --- buffer ----------------------------------------------------------------

// The decode's arena: the buffers its designs name, carved out of a few large
// BOs so that a whole token's streams can be joined into one command - a run
// has only a handful of buffer arguments, and every layer's buffers then sit
// in one of them. Grown in chunks as the decode's objects are created; freed
// once nothing carved out of them is left (xdna_arena_release).
namespace {
struct xdna_arena_state {
    std::vector<xdna_buffer *>              chunks;
    std::unordered_set<const xdna_buffer *> views;  // handed out, not yet freed
    size_t                                  used  = 0;  // in the last chunk
    int                                     depth = 0;  // open xdna_arena_scope's
};

xdna_arena_state g_arena;
std::mutex       g_arena_mutex;

size_t arena_chunk_bytes() {
    static const size_t v = (size_t) xdna_env_int("GGML_XDNA_ARENA_MB", 256) << 20;
    return v;
}
}  // namespace

xdna_arena_scope::xdna_arena_scope() {
    std::lock_guard<std::mutex> lock(g_arena_mutex);
    g_arena.depth++;
}

xdna_arena_scope::~xdna_arena_scope() {
    std::lock_guard<std::mutex> lock(g_arena_mutex);
    g_arena.depth--;
}

static xdna_buffer * xdna_buffer_alloc_own(xdna_device * dev, size_t bytes);

static std::atomic<int> g_alloc_failures{ 0 };

int xdna_buffer_alloc_failures(void) {
    return g_alloc_failures.load();
}

static xdna_buffer * xdna_arena_alloc(xdna_device * dev, size_t bytes) {
    std::lock_guard<std::mutex> lock(g_arena_mutex);
    if (g_arena.depth <= 0 || arena_chunk_bytes() == 0) {
        return nullptr;
    }
    const size_t need = xdna_align_up(bytes);
    if (g_arena.chunks.empty() || g_arena.used + need > g_arena.chunks.back()->bytes) {
        xdna_buffer * c = xdna_buffer_alloc_own(dev, std::max(arena_chunk_bytes(), need));
        if (!c) {
            return nullptr;
        }
        g_arena.chunks.push_back(c);
        g_arena.used = 0;
    }
    xdna_buffer * chunk = g_arena.chunks.back();
    xdna_buffer * buf   = xdna_buffer_sub(chunk, g_arena.used, bytes);
    if (buf) {
        g_arena.used += need;
        g_arena.views.insert(buf);
    }
    return buf;  // zeroed: the chunk was, and nothing else had this range
}

bool xdna_arena_release(void) {
    std::lock_guard<std::mutex> lock(g_arena_mutex);
    if (g_arena.depth > 0 || !g_arena.views.empty()) {
        return false;
    }
    for (xdna_buffer * c : g_arena.chunks) {
        delete c;
    }
    g_arena.chunks.clear();
    g_arena.used = 0;
    return true;
}

xdna_buffer * xdna_buffer_alloc(xdna_device * dev, size_t bytes) {
    if (!dev) {
        GGML_LOG_ERROR("%s: buffer alloc: no device\n", "xdna-runtime");
        return nullptr;
    }
    if (xdna_buffer * a = xdna_arena_alloc(dev, bytes)) {
        return a;
    }
    return xdna_buffer_alloc_own(dev, bytes);
}

static xdna_buffer * xdna_buffer_alloc_own(xdna_device * dev, size_t bytes) {
    xdna_buffer * buf = new xdna_buffer;
    try {
        buf->bo     = xrt::bo(dev->device, bytes, xrt::bo::flags::host_only, 0);
        buf->bytes  = bytes;
        // The host mapping is the only way to reach the BO's memory, and it can
        // fail: take it once here, where the failure can still free the buffer,
        // and cache it in `data`.
        void * data = buf->bo.map();
        if (!data) {
            GGML_LOG_ERROR("%s: %zu-byte BO allocation returned no host mapping\n", "xdna-runtime", bytes);
            delete buf;
            return nullptr;
        }
        buf->data = data;
        // Zero on allocation. Several buffers are only partly written by the
        // host and partly by the array, and a whole-buffer flush then pushes
        // whatever the host never wrote back to the device. Without this the
        // bytes a kernel reads out of those holes are the allocator's leftovers
        // and differ from process to process.
        std::memset(buf->data, 0, bytes);
        buf->bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    } catch (const std::exception & e) {
        g_alloc_failures++;
        GGML_LOG_ERROR("%s: failed to allocate %zu-byte BO: %s\n", "xdna-runtime", bytes, e.what());
        delete buf;
        return nullptr;
    }
    return buf;
}

xdna_buffer * xdna_buffer_sub(xdna_buffer * parent, size_t offset, size_t bytes) {
    if (!parent || !parent->data || bytes > parent->bytes || offset > parent->bytes - bytes) {
        GGML_LOG_ERROR("%s: buffer view %zu+%zu outside a %zu-byte BO or its parent has no mapping\n", "xdna-runtime",
                       offset, bytes, parent ? parent->bytes : 0);
        return nullptr;
    }
    xdna_buffer * buf = new xdna_buffer;
    try {
        buf->bo       = xrt::bo(parent->bo, bytes, offset);
        buf->bytes    = bytes;
        buf->root     = parent->root ? parent->root : parent;
        buf->root_off = parent->root_off + offset;
        // The view's host pointer comes from XRT, not from arithmetic on the
        // parent's: where a sub-buffer maps is XRT's contract to keep, and a
        // wrong host pointer here costs more than the call it would save.
        void * mapped = buf->bo.map();
        if (!mapped) {
            GGML_LOG_ERROR("%s: view %zu+%zu has no host mapping\n", "xdna-runtime", offset, bytes);
            delete buf;
            return nullptr;
        }
        buf->data = mapped;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: failed to view %zu+%zu of a BO: %s\n", "xdna-runtime", offset, bytes, e.what());
        delete buf;
        return nullptr;
    }
    return buf;
}

void xdna_buffer_free(xdna_buffer * buf) {
    if (buf && buf->root) {
        std::lock_guard<std::mutex> lock(g_arena_mutex);
        g_arena.views.erase(buf);
    }
    delete buf;
}

// The failure is propagated, not swallowed: a failed write leaves the kernel
// reading stale input and a failed read hands back stale output, and neither is
// visible downstream. There is no benign failure here - `bo.sync` throws only
// on a real driver error, and a run on the host-only BOs this backend uses
// never trips it, so propagating cannot turn a working op into a failed one.
static bool xdna_bo_sync(xrt::bo & bo, xclBOSyncDirection dir, size_t bytes, size_t offset) {
    const bool to = dir == XCL_BO_SYNC_BO_TO_DEVICE;
    try {
        bo.sync(dir, bytes, offset);
        return true;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: sync %zu bytes %s device at offset %zu failed: %s\n", "xdna-runtime", bytes,
                       to ? "to" : "from", offset, e.what());
        return false;
    }
}

// A missing buffer is a failure, not a no-op: every caller here needs the
// buffer to exist, so reporting success for a null one would hide the bug that
// let it be null. A zero-byte sync has nothing to move and succeeds.
bool xdna_buffer_sync_to_device(xdna_buffer * buf) {
    return buf && (buf->bytes == 0 || xdna_bo_sync(buf->bo, XCL_BO_SYNC_BO_TO_DEVICE, buf->bytes, 0));
}

bool xdna_buffer_sync_to_device_range(xdna_buffer * buf, size_t bytes, size_t offset) {
    return buf && (bytes == 0 || xdna_bo_sync(buf->bo, XCL_BO_SYNC_BO_TO_DEVICE, bytes, offset));
}

bool xdna_buffer_sync_from_device(xdna_buffer * buf) {
    return buf && (buf->bytes == 0 || xdna_bo_sync(buf->bo, XCL_BO_SYNC_BO_FROM_DEVICE, buf->bytes, 0));
}

bool xdna_buffer_sync_from_device_range(xdna_buffer * buf, size_t bytes, size_t offset) {
    return buf && (bytes == 0 || xdna_bo_sync(buf->bo, XCL_BO_SYNC_BO_FROM_DEVICE, bytes, offset));
}

// --- reading what the array wrote ------------------------------------------
//
// A dispatch reports completion before the last of its DMA writes is readable,
// and a read taken in that window hands back - and caches - the previous
// contents of the buffer. There is no host-side barrier for it: the completion
// is honest, and the cache ioctl above does nothing on this platform for a
// host_only BO. So the read has to tell a landed write from a pending one, and
// that is what the pattern is for: mark the range before the dispatch, read
// until the mark is gone.

// Only whole words can carry the pattern.
static size_t xdna_markable(size_t bytes) {
    return bytes & ~(size_t) (sizeof(uint32_t) - 1);
}

static void xdna_poison_range(void * p, size_t bytes) {
    uint32_t *   words = (uint32_t *) p;
    const size_t n     = bytes / sizeof(uint32_t);
    for (size_t i = 0; i < n; i++) {
        words[i] = XDNA_POISON_F32;
    }
}

static size_t xdna_poison_left(const void * p, size_t bytes) {
    const uint32_t * words = (const uint32_t *) p;
    const size_t     n     = bytes / sizeof(uint32_t);
    size_t           left  = 0;
    for (size_t i = 0; i < n; i++) {
        left += (words[i] == XDNA_POISON_F32);
    }
    return left;
}

static void xdna_cache_writeback(void * p, size_t bytes) {
#if defined(__x86_64__) || defined(__i386__)
    const size_t line = 64;
    uintptr_t    a    = (uintptr_t) p & ~(uintptr_t) (line - 1);
    const uintptr_t end = ((uintptr_t) p + bytes + line - 1) & ~(uintptr_t) (line - 1);
    for (; a < end; a += line) {
        _mm_clflush((const void *) a);
    }
    _mm_mfence();
#else
    (void) p;
    (void) bytes;
#endif
}

void xdna_buffer_mark(xdna_buffer * buf, size_t bytes, size_t offset) {
    if (!buf || !buf->data || bytes == 0 || offset >= buf->bytes) {
        return;
    }
    if (bytes > buf->bytes - offset) {
        bytes = buf->bytes - offset;
    }
    bytes = xdna_markable(bytes);
    if (bytes == 0) {
        return;
    }
    uint8_t * base = (uint8_t *) buf->data + offset;
    xdna_poison_range(base, bytes);
    // Written back, not left dirty: our own cache lines could otherwise land
    // on top of what the device writes.
    xdna_cache_writeback(base, bytes);
}

uint8_t * xdna_buffer_wait_written(xdna_buffer * buf, size_t bytes, size_t offset) {
    if (!buf || !buf->data || bytes == 0 || offset >= buf->bytes) {
        return nullptr;
    }
    if (bytes > buf->bytes - offset) {
        bytes = buf->bytes - offset;
    }
    bytes = xdna_markable(bytes);
    if (bytes == 0) {
        return nullptr;
    }
    uint8_t * base = (uint8_t *) buf->data + offset;
    static const int passes      = xdna_env_int("GGML_XDNA_SETTLE_PASSES", 2000);
    static const int pause_loops = xdna_env_int("GGML_XDNA_SETTLE_PAUSE", 400);
    for (int pass = 0;; pass++) {
        if (xdna_poison_left(base, bytes) == 0) {
            return base;
        }
        if (pass >= passes) {
            GGML_LOG_WARN("%s: %zu bytes at %zu still hold the mark after %d "
                          "passes; the device has not written them\n",
                          "xdna-runtime", bytes, offset, passes);
            return nullptr;
        }
        for (int i = 0; i < pause_loops; i++) {
#if defined(__x86_64__) || defined(__i386__)
            _mm_pause();
#endif
        }
        xdna_cache_writeback((void *) base, bytes);
    }
}

bool xdna_buffer_download(xdna_buffer * buf, void * dst, size_t bytes, size_t offset) {
    if (!buf || !buf->data || !dst || bytes == 0 || offset >= buf->bytes) {
        return false;
    }
    if (bytes > buf->bytes - offset) {
        bytes = buf->bytes - offset;
    }
    bytes = xdna_markable(bytes);
    if (bytes == 0) {
        return false;
    }
    uint8_t * base = (uint8_t *) buf->data + offset;
    // Bounded by the drain, not by a guess: the pattern is the evidence, and it
    // disappears exactly when the device's write lands. A pass is a pause, not
    // a timer sleep - sleep_for(2us) costs a tick and measures about 80us.
    static const int passes      = xdna_env_int("GGML_XDNA_SETTLE_PASSES", 2000);
    static const int pause_loops = xdna_env_int("GGML_XDNA_SETTLE_PAUSE", 400);
    for (int pass = 0;; pass++) {
        std::memcpy(dst, base, bytes);
        if (xdna_poison_left(dst, bytes) == 0) {
            return true;
        }
        if (pass >= passes) {
            GGML_LOG_WARN("%s: %zu bytes at %zu still hold the mark after %d "
                          "passes; the device has not written them\n",
                          "xdna-runtime", bytes, offset, passes);
            return false;
        }
        for (int i = 0; i < pause_loops; i++) {
#if defined(__x86_64__) || defined(__i386__)
            _mm_pause();
#endif
        }
        // A platform whose device writes are not coherent with the CPU caches
        // needs the read range dropped before looking again.
        xdna_cache_writeback((void *) base, bytes);
    }
}

bool xdna_buffer_read_settled(xdna_buffer * buf, void * dst, size_t bytes, size_t offset) {
    if (!buf || !buf->data || !dst || bytes == 0) {
        return false;
    }
    if (offset >= buf->bytes) {
        return false;
    }
    if (bytes > buf->bytes - offset) {
        bytes = buf->bytes - offset;
    }
    uint8_t *  base    = (uint8_t *) buf->data + offset;
    // Spin, not sleep: a sleep_for(20us) costs a timer tick and measured about
    // 80us, and this is paid once per read.
    const auto spin_us = [](int us) {
        const std::chrono::steady_clock::time_point until =
            std::chrono::steady_clock::now() + std::chrono::microseconds(us);
        while (std::chrono::steady_clock::now() < until) {
        }
    };
    static const int settle_us = xdna_env_int("GGML_XDNA_SETTLE_US", 20);
    if (!xdna_buffer_sync_from_device(buf)) {
        return false;
    }
    std::memcpy(dst, base, bytes);
    for (int attempt = 0; attempt < 4; attempt++) {
        if (settle_us > 0) {
            spin_us(settle_us);
        }
        if (!xdna_buffer_sync_from_device(buf)) {
            return false;
        }
        if (std::memcmp(dst, base, bytes) == 0) {
            return true;
        }
        std::memcpy(dst, base, bytes);
    }
    return true;
}

// --- execution -------------------------------------------------------------

// Configure a run for `kern` with `n_args` host buffers (ABI: 0=opcode,
// 1=instruction BO, 2=ninstr in 32-bit words, 3..=host buffers).
static xrt::run make_run(xdna_kernel * kern, xdna_buffer ** args, size_t n_args) {
    xrt::run run(kern->kernel);
    run.set_arg(0, XAIE_NPU_OPCODE_RUN);
    run.set_arg(1, kern->insts_bo);
    run.set_arg(2, (uint32_t) (kern->insts_bytes / (int64_t) sizeof(uint32_t)));
    for (size_t i = 0; i < n_args; i++) {
        run.set_arg((int) (3 + i), args[i]->bo);
    }
    return run;
}

xrt::run xdna_kernel_run_make(xdna_kernel * kern, xdna_buffer ** args, size_t n_args) {
    if (!kern || !args || n_args == 0) {
        GGML_LOG_ERROR("%s: run make: null kernel/args or no buffers\n", "xdna-runtime");
        return xrt::run{};
    }
    try {
        return make_run(kern, args, n_args);
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: kernel run make exception: %s\n", "xdna-runtime", e.what());
        return xrt::run{};
    }
}

namespace {
struct xdna_batch_state {
    int                                        k = 0;  // runs a list; 0 = not batching
    std::vector<std::unique_ptr<xrt::runlist>> sent;
    std::unique_ptr<xrt::runlist>              cur;
    const xrt::hw_context *                    cur_ctx = nullptr;
    int                                        n_cur   = 0;
    int                                        lead    = 0;  // runs still to start on their own
    std::vector<xrt::run *>                    direct;       // the ones that were
    // token mode: the runs' streams joined into one command at the send
    bool                                       token = false;
    int                                        join  = 0;  // runs a joined command, 0: all of them

    struct entry {
        xdna_kernel *              kern;
        xrt::run *                 run;
        std::vector<xdna_buffer *> args;
    };

    std::vector<entry>       tok;
    std::vector<xrt::run>    tok_runs;  // sent, to wait for
    // the joined commands' kernels and instruction BOs, one a command of the
    // token (a BO in flight cannot be rewritten), reused token to token
    std::vector<xdna_kernel> joined;
    std::vector<size_t>      joined_cap;
    std::vector<uint32_t>    words;
};

xdna_batch_state g_batch;

// Words a TXN op takes (xdna-seq.h); 0 for one this does not know.
uint32_t op_words(uint32_t op) {
    switch (op) {
        case xdna_txn::OP_WRITE:
            return xdna_txn::SZ_WRITE;
        case xdna_txn::OP_BLOCKWRITE:
            return xdna_txn::SZ_BLOCKWRITE;
        case xdna_txn::OP_MASKWRITE:
            return xdna_txn::SZ_MASKWRITE;
        case xdna_txn::OP_TCT:
            return xdna_txn::SZ_TCT;
        case xdna_txn::OP_DDR_PATCH:
            return xdna_txn::SZ_DDR_PATCH;
        default:
            return 0;
    }
}

bool token_start_each() {
    auto & b = g_batch;
    for (auto & e : b.tok) {
        try {
            e.run->start();
        } catch (const std::exception & ex) {
            GGML_LOG_ERROR("%s: run start exception: %s\n", "xdna-runtime", ex.what());
            return false;
        }
        b.direct.push_back(e.run);
    }
    b.tok.clear();
    return true;
}

// One command for everything collected: the streams back to back, each
// DDR patch renamed to the root BO its buffer lives in (one argument slot a
// root) at the buffer's offset in it. Falls back to a command each when the
// roots do not fit the firmware-translated slots or a stream does not parse.
bool token_send() {
    auto & b = g_batch;
    if (b.tok.empty()) {
        return true;
    }
    // Two root pools. Slots 5..9 carry the DDR aperture in bit 31, are not
    // translated by the firmware, and therefore hold a root only when every
    // offset into it is under 2 GB. Slots 0..4 are firmware-translated and take
    // a plain 32-bit offset, which is what the KV cache needs: it is one 3 GB
    // BO at a 262144 context and its fifth attention layer starts past 2 GB.
    std::vector<xdna_buffer *> roots_lo, roots_hi;
    constexpr int              slot0   = (int) xdna_txn::XDNA_FW_XLAT_ARGS;
    const auto                 root_of = [](xdna_buffer * a) -> xdna_buffer * {
        return a->root ? a->root : a;
    };
    const auto slot_lo = [&](xdna_buffer * a) -> int {
        xdna_buffer * r = root_of(a);
        for (size_t i = 0; i < roots_lo.size(); i++) {
            if (roots_lo[i] == r) {
                return (int) i;
            }
        }
        if (roots_lo.size() >= (size_t) slot0) {
            return -1;
        }
        roots_lo.push_back(r);
        return (int) roots_lo.size() - 1;
    };
    const auto slot_hi = [&](xdna_buffer * a) -> int {
        xdna_buffer * r = root_of(a);
        for (size_t i = 0; i < roots_hi.size(); i++) {
            if (roots_hi[i] == r) {
                return slot0 + (int) i;
            }
        }
        if (slot0 + (int) roots_hi.size() >= 10) {
            return -1;
        }
        roots_hi.push_back(r);
        return slot0 + (int) roots_hi.size() - 1;
    };
    // Rewrite one DDR patch in place: choose the pool the offset needs, write
    // the slot and the offset back into `dst` at `at`.
    const auto place = [&](uint32_t arg, uint32_t off_raw, std::vector<uint32_t> & dst, size_t at,
                           const auto & e) -> bool {
        if (arg >= e.args.size() || !e.args[arg]) {
            return false;
        }
        const bool     src_low = arg < (uint32_t) xdna_txn::XDNA_FW_XLAT_ARGS;
        const uint32_t off     = src_low ? off_raw : (off_raw & ~xdna_txn::XDNA_DDR_APERTURE);
        xdna_buffer *  a       = e.args[arg];
        const uint64_t noff    = (uint64_t) a->root_off + off;
        if (noff >= xdna_txn::XDNA_DDR_APERTURE && src_low) {
            const int s = slot_lo(a);
            if (s < 0 || noff > 0xFFFFFFFFull) {
                return false;
            }
            dst[at + 8]  = (uint32_t) s;
            dst[at + 10] = (uint32_t) noff;
            return true;
        }
        const int s = slot_hi(a);
        if (s < 0 || noff >= xdna_txn::XDNA_DDR_APERTURE) {
            return false;
        }
        dst[at + 8]  = (uint32_t) s;
        dst[at + 10] = (uint32_t) noff | xdna_txn::XDNA_DDR_APERTURE;
        return true;
    };
    std::vector<uint32_t> & w = b.words;
    w.assign(4, 0);
    uint32_t n_instr    = 0;
    size_t   prev_start = 4, cur_start = 4;
    for (auto & e : b.tok) {
        prev_start                       = cur_start;
        cur_start                        = w.size();
        const std::vector<uint32_t> & in = e.kern->insts_host;
        if (in.size() < 4 || e.kern->context != b.tok[0].kern->context) {
            return token_start_each();
        }
        if (w[0] == 0) {
            w[0] = in[0];
            w[1] = in[1];
        }
        n_instr += in[2];
        // The run's leading weight fills (a BD, its patch and its MM2S push on
        // one column, descriptor 0) go before the previous run's trailing
        // waits, so the next layer's weights stream in while the last one's
        // outputs drain - when that run's tail since its last wait before them
        // does not write the descriptor. +1.2% tok/s; the rest of a layer's
        // head reads what the previous layer is still writing.
        constexpr bool    hoist = true;
        std::vector<char> moved;
        if (hoist && w.size() > 4) {
            moved.assign(in.size(), 0);
            // the previous run's trailing waits in w, and its tail before them
            size_t tw = w.size();
            while (tw >= 8 && w[tw - 4] == 0x80) {
                tw -= 4;
            }
            size_t j = 4;
            for (; j < in.size() && in[j] != 0x80;) {
                const uint32_t nw = op_words(in[j]);
                if (nw == 0) {
                    break;
                }
                if (in[j] == 0x01 && j + 12 + 12 + 6 <= in.size() && in[j + 12] == 0x81 && in[j + 24] == 0x00) {
                    const uint32_t r = in[j + 2], col = (r >> 25) & 0x7F;
                    const uint32_t bd = ((r & 0xFFFFF) - 0x1D000) / 0x20;
                    const uint32_t pr = in[j + 26] & 0xFFFFF, pcol = (in[j + 26] >> 25) & 0x7F;
                    const bool     mm2s = pr >= 0x1D214 && pr < 0x1D224;
                    if (bd == 0 && pcol == col && mm2s && (in[j + 28] & 0xF) == 0) {
                        // the previous run's tail since its last wait before the
                        // trailing block must not write (col, bd 0)
                        bool   clash = false;
                        size_t tail0 = prev_start;
                        for (size_t q = prev_start; q < tw;) {
                            const uint32_t qn = op_words(w[q]);
                            if (qn == 0) {
                                clash = true;
                                break;
                            }
                            if (w[q] == 0x80) {
                                tail0 = q;
                            }
                            q += qn;
                        }
                        for (size_t q = tail0; !clash && q < tw;) {
                            const uint32_t qn = op_words(w[q]);
                            if (w[q] == 0x01 && ((w[q + 2] >> 25) & 0x7F) == col &&
                                (((w[q + 2] & 0xFFFFF) - 0x1D000) / 0x20) == 0) {
                                clash = true;
                            }
                            q += qn;
                        }
                        if (!clash) {
                            for (size_t q = j; q < j + 30; q++) {
                                moved[q] = 1;
                            }
                            j += 30;
                            continue;
                        }
                    }
                }
                j += nw;
            }
        }
        if (hoist && !moved.empty()) {
            // splice the moved groups in before the trailing waits
            size_t tw = w.size();
            while (tw >= 8 && w[tw - 4] == 0x80) {
                tw -= 4;
            }
            std::vector<uint32_t> ins;
            for (size_t q = 4; q < in.size();) {
                const uint32_t qn = op_words(in[q]);
                if (qn == 0) {
                    break;
                }
                if (moved[q]) {
                    ins.insert(ins.end(), in.begin() + (std::ptrdiff_t) q, in.begin() + (std::ptrdiff_t)(q + qn));
                }
                q += qn;
            }
            // relocate the patches in `ins` as the main loop would
            for (size_t q = 0; q < ins.size();) {
                const uint32_t qn = op_words(ins[q]);
                if (ins[q] == 0x81 && !place(ins[q + 8], ins[q + 10], ins, q, e)) {
                    return token_start_each();
                }
                q += qn;
            }
            w.insert(w.begin() + (long) tw, ins.begin(), ins.end());
        }
        for (size_t i = 4; i < in.size();) {
            if (!moved.empty() && moved[i]) {
                i += op_words(in[i]);
                continue;
            }
            const uint32_t nw = op_words(in[i]);
            if (nw == 0 || i + nw > in.size()) {
                return token_start_each();
            }
            const size_t at = w.size();
            w.insert(w.end(), in.begin() + (std::ptrdiff_t) i, in.begin() + (std::ptrdiff_t)(i + nw));
            if (in[i] == 0x81 && !place(in[i + 8], in[i + 10], w, at, e)) {
                return token_start_each();
            }
            i += nw;
        }
    }
    w[2]               = n_instr;
    w[3]               = (uint32_t) (w.size() * sizeof(uint32_t));
    const size_t bytes = w.size() * sizeof(uint32_t);
    try {
        const size_t ji = b.tok_runs.size();
        if (b.joined.size() <= ji) {
            b.joined.resize(ji + 1);
            b.joined_cap.resize(ji + 1, 0);
        }
        xdna_kernel & j  = b.joined[ji];
        xdna_kernel * k0 = b.tok[0].kern;
        if (j.context != k0->context || b.joined_cap[ji] < bytes) {
            j.device         = k0->device;
            j.context        = k0->context;
            j.kernel         = k0->kernel;
            b.joined_cap[ji] = bytes * 2;
            j.insts_bo       = xrt::bo(j.device, b.joined_cap[ji], xrt::bo::flags::cacheable, j.kernel.group_id(1));
        }
        void * jmap = j.insts_bo.map();
        if (!jmap) {
            GGML_LOG_ERROR("%s: joined %zu-byte instruction stream has no host mapping\n", "xdna-runtime", bytes);
            return false;
        }
        std::memcpy(jmap, w.data(), bytes);
        j.insts_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, bytes, 0);
        xrt::run run(j.kernel);
        run.set_arg(0, XAIE_NPU_OPCODE_RUN);
        run.set_arg(1, j.insts_bo);
        run.set_arg(2, (uint32_t) w.size());
        for (int s = 0; s < 10; s++) {
            // the argument list is positional, so every slot gets a BO even
            // when this token's streams do not name it
            xdna_buffer * bo = nullptr;
            if (s < slot0) {
                if (!roots_lo.empty()) {
                    bo = roots_lo[std::min((size_t) s, roots_lo.size() - 1)];
                } else if (!roots_hi.empty()) {
                    bo = roots_hi[0];
                }
            } else {
                const size_t r = (size_t) (s - slot0);
                if (r < roots_hi.size()) {
                    bo = roots_hi[r];
                } else if (!roots_hi.empty()) {
                    bo = roots_hi[0];
                } else if (!roots_lo.empty()) {
                    bo = roots_lo[0];
                }
            }
            if (bo) {
                run.set_arg(3 + s, bo->bo);
            }
        }
        run.start();
        b.tok_runs.push_back(std::move(run));
    } catch (const std::exception & ex) {
        GGML_LOG_ERROR("%s: joined token stream exception: %s\n", "xdna-runtime", ex.what());
        return false;
    }
    b.tok.clear();
    return true;
}

bool batch_send() {
    auto & b = g_batch;
    if (b.token) {
        return token_send();
    }
    if (!b.cur || b.n_cur == 0) {
        return true;
    }
    try {
        b.cur->execute();
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: runlist execute exception: %s\n", "xdna-runtime", e.what());
        return false;
    }
    b.sent.push_back(std::move(b.cur));
    b.cur_ctx = nullptr;
    b.n_cur   = 0;
    return true;
}
}  // namespace

void xdna_batch_begin(int k) {
    g_batch.token = k < 0;
    g_batch.join  = k < -1 ? -k : 0;
    g_batch.k     = k != 0 ? 1 << 30 : 0;
    if (k > 0) {
        g_batch.k = k;
    }
    // The token's first run starts on its own: the array has work while the
    // host prepares the first list, instead of waiting for all of it.
    static const int lead = [] {
        const char * e = getenv("GGML_XDNA_BATCH_LEAD");
        return e ? (int) std::clamp(strtol(e, nullptr, 10), (long) INT_MIN, (long) INT_MAX) : 1;
    }();
    g_batch.lead = lead;
}

bool xdna_batch_active(void) {
    return g_batch.k > 0;
}

bool xdna_run_submit(xdna_kernel * kern, xrt::run & run, xdna_buffer * const * args, size_t n_args) {
    auto & b = g_batch;
    if (b.k <= 0 || !kern || !kern->context) {
        return xdna_run_restart(run);
    }
    if (b.token && args && !kern->insts_host.empty()) {
        if (b.lead > 0 && b.tok.empty() && b.tok_runs.empty()) {
            b.lead--;
            if (!xdna_run_restart(run)) {
                return false;
            }
            b.direct.push_back(&run);
            return true;
        }
        b.tok.push_back({ kern, &run, std::vector<xdna_buffer *>(args, args + n_args) });
        if (b.join > 0 && (int) b.tok.size() >= b.join) {
            return token_send();
        }
        return true;
    }
    if (b.token) {
        // a run the token cannot join: what it holds goes first
        if (!token_send()) {
            return false;
        }
        if (!xdna_run_restart(run)) {
            return false;
        }
        b.direct.push_back(&run);
        return true;
    }
    if (b.lead > 0 && !b.cur && b.sent.empty()) {
        b.lead--;
        if (!xdna_run_restart(run)) {
            return false;
        }
        b.direct.push_back(&run);
        return true;
    }
    try {
        if (b.cur && b.cur_ctx != kern->context.get() && !batch_send()) {
            return false;
        }
        if (!b.cur) {
            b.cur     = std::make_unique<xrt::runlist>(*kern->context);
            b.cur_ctx = kern->context.get();
        }
        b.cur->add(run);
        if (++b.n_cur >= b.k) {
            return batch_send();
        }
        return true;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: runlist add exception: %s\n", "xdna-runtime", e.what());
        return false;
    }
}

bool xdna_batch_wait(void) {
    auto & b  = g_batch;
    bool   ok = batch_send();
    for (xrt::run * r : b.direct) {
        ok = xdna_run_wait(*r) && ok;
    }
    b.direct.clear();
    for (xrt::run & r : b.tok_runs) {
        ok = xdna_run_wait(r) && ok;
    }
    b.tok_runs.clear();
    for (auto & l : b.sent) {
        try {
            l->wait();
            if (l->state() != ERT_CMD_STATE_COMPLETED) {
                GGML_LOG_ERROR("%s: runlist state %d\n", "xdna-runtime", (int) l->state());
                ok = false;
            }
        } catch (const std::exception & e) {
            GGML_LOG_ERROR("%s: runlist wait exception: %s\n", "xdna-runtime", e.what());
            ok = false;
        }
    }
    b.sent.clear();
    b.k     = 0;
    b.token = false;
    return ok;
}

bool xdna_run_restart(xrt::run & run) {
    // anything a batch holds goes first, so the device sees the host's order
    if ((g_batch.cur || !g_batch.tok.empty()) && !batch_send()) {
        return false;
    }
    try {
        run.start();
        return true;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: kernel run restart exception: %s\n", "xdna-runtime", e.what());
        return false;
    }
}

xrt::run xdna_kernel_run_start(xdna_kernel * kern, xdna_buffer ** args, size_t n_args) {
    if (!kern || !args || n_args == 0) {
        GGML_LOG_ERROR("%s: run start: null kernel/args or no buffers\n", "xdna-runtime");
        return xrt::run{};
    }
    if ((g_batch.cur || !g_batch.tok.empty()) && !batch_send()) {
        return xrt::run{};
    }
    try {
        xrt::run run = make_run(kern, args, n_args);
        run.start();
        return run;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: kernel run start exception: %s\n", "xdna-runtime", e.what());
        return xrt::run{};
    }
}

bool xdna_run_wait(xrt::run & run) {
    // A signal interrupts the wait's own ioctl, and the shim answers that with
    // an exception - "unexpected command state" - rather than a state. The
    // command is still the array's, so the wait is retried while it is in
    // flight: without this a Ctrl+C reads as a dead command, the decode fails
    // and the tool reports an error instead of stopping.
    for (int attempt = 0;; attempt++) {
        try {
            // Block, do not poll. Polling kept a core at full power for 70% of a
            // decode token (it waits once a token for its whole queue, ~20 ms) -
            // the host energy FLM does not spend - and bought nothing: blocking
            // measured the same tok/s in the server, at 0.29 of the CPU time
            // polling 200 us before blocking took. GGML_XDNA_SPIN_US polls that
            // long first; -1 polls without limit, as it used to by default.
            static const int spin_us = xdna_env_int("GGML_XDNA_SPIN_US", 0);
            if (spin_us != 0) {
                const auto until =
                    std::chrono::steady_clock::now() + std::chrono::microseconds(spin_us < 0 ? 0 : spin_us);
                for (;;) {
                    const ert_cmd_state s = run.state();
                    if (s == ERT_CMD_STATE_COMPLETED) {
                        return true;
                    }
                    if (s == ERT_CMD_STATE_ERROR || s == ERT_CMD_STATE_ABORT || s == ERT_CMD_STATE_TIMEOUT) {
                        GGML_LOG_ERROR("%s: kernel spin state %d\n", "xdna-runtime", (int) s);
                        return false;
                    }
                    if (spin_us > 0 && std::chrono::steady_clock::now() >= until) {
                        break;
                    }
                }
            }
            const ert_cmd_state st = run.wait();
            if (st != ERT_CMD_STATE_COMPLETED) {
                GGML_LOG_ERROR("%s: kernel wait state %d\n", "xdna-runtime", (int) st);
                return false;
            }
            return true;
        } catch (const std::exception & e) {
            ert_cmd_state state = ERT_CMD_STATE_NEW;
            bool         known = false;
            try {
                state = run.state();
                known = true;
            } catch (...) {
            }
            if (known && state == ERT_CMD_STATE_COMPLETED) {
                return true;
            }
            // A state that is dead is the old path; one still in flight is the
            // interrupted wait, and that is retried.
            if (known && (state == ERT_CMD_STATE_ERROR || state == ERT_CMD_STATE_ABORT ||
                          state == ERT_CMD_STATE_TIMEOUT)) {
                GGML_LOG_ERROR("%s: kernel wait exception: %s (state %d)\n", "xdna-runtime", e.what(), (int) state);
                return false;
            }
            if (attempt >= 8) {
                GGML_LOG_ERROR("%s: kernel wait exception: %s (state %d, %d retries)\n", "xdna-runtime", e.what(),
                               known ? (int) state : -1, attempt);
                return false;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    }
}

// --- kernel pool -----------------------------------------------------------

void xdna_kernel_pool_scan(xdna_kernel_pool * pool) {
    pool->names.clear();
    for (const fs::path & dir : xdna_kernel_search_dirs()) {
        std::error_code ec;
        for (const auto & entry : fs::directory_iterator(dir, ec)) {
            if (ec) {
                break;
            }
            if (entry.path().extension() != ".xclbin") {
                continue;
            }
            pool->names.push_back(entry.path().stem().string());
        }
    }
}

xdna_kernel * xdna_kernel_pool_get_built(xdna_kernel_pool *  pool,
                                         const std::string & name,
                                         const char *        xclbin_name,
                                         const uint32_t *    insts,
                                         size_t              n_words) {
    std::lock_guard<std::mutex> lock(pool->kernel_mutex);

    // Keyed by the stream as well as the name: a name does not say everything
    // a stream was built with (the head's leaves out the row length, so a
    // second model's head ran the first one's stream and the array hung), and
    // a caller handed another stream bound under its name runs its buffers
    // through that one. Identical streams still share a kernel.
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n_words; i++) {
        h = (h ^ insts[i]) * 1099511628211ull;
    }
    char hs[24];
    snprintf(hs, sizeof(hs), "#%016llx", (unsigned long long) h);
    const std::string key = name + hs;

    auto it = pool->kernels.find(key);
    if (it != pool->kernels.end()) {
        return it->second;
    }

    // nullptr is sticky: a failed lookup is not retried.
    pool->kernels[key] = nullptr;

    // `xclbin_name` is the artifact stem; resolve it to a real path.
    xdna_kernel * kern = xdna_kernel_find(pool->device, xclbin_name);
    if (!kern) {
        GGML_LOG_WARN("%s: kernel %s (hw %s) not found\n", "xdna-runtime", name.c_str(), xclbin_name);
        return nullptr;
    }
    if (!xdna_kernel_bind_insts(pool->device, kern, insts, n_words)) {
        xdna_kernel_free(kern);
        return nullptr;
    }
    pool->kernels[key] = kern;
    return kern;
}

xdna_buffer * xdna_kernel_pool_acquire_buffer(xdna_kernel_pool * pool, size_t bytes) {
    {
        std::lock_guard<std::mutex> lock(pool->pool_mutex);
        size_t                      best      = pool->pool.size();
        size_t                      best_size = 0;
        for (size_t i = 0; i < pool->pool.size(); i++) {
            if (!pool->pool[i].buf) {
                continue;
            }
            const size_t sz = pool->pool[i].buf->bytes;
            if (sz >= bytes && (best == pool->pool.size() || sz < best_size)) {
                best      = i;
                best_size = sz;
            }
        }
        if (best != pool->pool.size()) {
            xdna_buffer * buf = pool->pool[best].buf;
            pool->pool.erase(pool->pool.begin() + (std::ptrdiff_t) best);
            return buf;
        }
    }
    return xdna_buffer_alloc(pool->device, bytes);
}

void xdna_kernel_pool_release_buffer(xdna_kernel_pool * pool, xdna_buffer * buf) {
    if (!pool || !buf) {
        return;
    }
    std::lock_guard<std::mutex> lock(pool->pool_mutex);
    pool->pool.push_back({ buf, ++pool->pool_tick });

    if (pool->pool.size() > xdna_kernel_pool::MAX_POOL_SIZE) {
        size_t lru = 0;
        for (size_t i = 1; i < pool->pool.size(); i++) {
            if (pool->pool[i].seq < pool->pool[lru].seq) {
                lru = i;
            }
        }
        xdna_buffer_free(pool->pool[lru].buf);
        pool->pool.erase(pool->pool.begin() + (std::ptrdiff_t) lru);
    }
}
