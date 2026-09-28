#include "xdna-runtime.h"

#include "ggml-impl.h"
#include "xdna-util.h"

#include <xrt/experimental/xrt_kernel.h>
#include <xrt/experimental/xrt_xclbin.h>
#include <xrt/xrt_kernel.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <system_error>
#include <unordered_set>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace fs = std::filesystem;

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
        dev->name        = dev->device.get_info<xrt::info::device::name>();
        dev->description = dev->device.get_info<xrt::info::device::bdf>();
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
    fs::path exe = fs::read_symlink("/proc/self/exe", ec);
    if (!ec) {
        dirs.push_back(exe.parent_path());
    }
#endif
    dirs.emplace_back(fs::current_path());
    return dirs;
}

xdna_kernel * xdna_kernel_load_hw(xdna_device * dev, const char * xclbin_path) {
    if (!dev) {
        GGML_LOG_ERROR("%s: kernel load: no device\n", "xdna-runtime");
        return nullptr;
    }

    try {
        const std::string xclbin_path_str(xclbin_path);
        const xrt::xclbin xclbin{xclbin_path_str};
        dev->device.register_xclbin(xclbin);
        const xrt::uuid xclbin_uuid = xclbin.get_uuid();

        // Reuse the hw_context for this xclbin across all kernel variants.
        std::shared_ptr<xrt::hw_context> context;
        {
            std::lock_guard<std::mutex> lock(g_ctx_mutex);
            auto it = g_ctxs.find(xclbin_uuid);
            if (it != g_ctxs.end()) {
                context = it->second;
            } else {
                context = std::make_shared<xrt::hw_context>(dev->device, xclbin_uuid);
                g_ctxs.emplace(xclbin_uuid, context);
            }
        }

        const auto kernels = xclbin.get_kernels();
        if (kernels.empty()) {
            GGML_LOG_ERROR("%s: no kernel found in %s\n", "xdna-runtime", xclbin_path);
            return nullptr;
        }

        xdna_kernel * kern = new xdna_kernel;
        kern->device      = dev->device;
        kern->context     = context;
        kern->kernel      = xrt::kernel(*context, kernels[0].get_name());
        kern->xclbin_name = xclbin_path_str;
        return kern;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: failed to load kernel %s: %s\n", "xdna-runtime", xclbin_path, e.what());
        return nullptr;
    }
}

bool xdna_kernel_bind_insts(xdna_device * dev, xdna_kernel * kern, const uint32_t * insts, size_t n_words) {
    if (!dev || !kern || !insts || n_words == 0) {
        GGML_LOG_ERROR("%s: bind insts: null device/kernel/insts or empty stream\n", "xdna-runtime");
        return false;
    }
    try {
        kern->insts_bytes = (int64_t)(n_words * sizeof(uint32_t));
        kern->insts_bo    = xrt::bo(dev->device, (size_t) kern->insts_bytes,
                                    xrt::bo::flags::cacheable, kern->kernel.group_id(1));
        std::memcpy(kern->insts_bo.map(), insts, (size_t) kern->insts_bytes);
        kern->insts_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        kern->insts_host.assign(insts, insts + n_words);
        return true;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: failed to bind instruction stream: %s\n", "xdna-runtime", e.what());
        return false;
    }
}

bool xdna_kernel_rewrite_insts(xdna_kernel * kern, const uint32_t * insts, size_t n_words) {
    if (!kern || !insts || (int64_t) (n_words * sizeof(uint32_t)) != kern->insts_bytes) {
        return false;
    }
    std::memcpy(kern->insts_bo.map(), insts, (size_t) kern->insts_bytes);
    kern->insts_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    kern->insts_host.assign(insts, insts + n_words);
    return true;
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
    std::vector<xdna_buffer *> chunks;
    std::unordered_set<const xdna_buffer *> views;   // handed out, not yet freed
    size_t used = 0;          // in the last chunk
    int    depth = 0;         // open xdna_arena_scope's
};
xdna_arena_state g_arena;
std::mutex       g_arena_mutex;

size_t arena_chunk_bytes() {
    static const size_t v = (size_t) xdna_env_int("GGML_XDNA_ARENA_MB", 256) << 20;
    return v;
}
} // namespace

xdna_arena_scope::xdna_arena_scope() {
    std::lock_guard<std::mutex> lock(g_arena_mutex);
    g_arena.depth++;
}

xdna_arena_scope::~xdna_arena_scope() {
    std::lock_guard<std::mutex> lock(g_arena_mutex);
    g_arena.depth--;
}

static xdna_buffer * xdna_buffer_alloc_own(xdna_device * dev, size_t bytes);

static xdna_buffer * xdna_arena_alloc(xdna_device * dev, size_t bytes) {
    std::lock_guard<std::mutex> lock(g_arena_mutex);
    if (g_arena.depth <= 0 || arena_chunk_bytes() == 0) {
        return nullptr;
    }
    const size_t need = (bytes + 4095) / 4096 * 4096;
    if (g_arena.chunks.empty() || g_arena.used + need > g_arena.chunks.back()->bytes) {
        xdna_buffer * c = xdna_buffer_alloc_own(dev, std::max(arena_chunk_bytes(), need));
        if (!c) {
            return nullptr;
        }
        g_arena.chunks.push_back(c);
        g_arena.used = 0;
    }
    xdna_buffer * chunk = g_arena.chunks.back();
    xdna_buffer * buf = xdna_buffer_sub(chunk, g_arena.used, bytes);
    if (buf) {
        g_arena.used += need;
        g_arena.views.insert(buf);
    }
    return buf;   // zeroed: the chunk was, and nothing else had this range
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
        buf->bo    = xrt::bo(dev->device, bytes, xrt::bo::flags::host_only, 0);
        buf->bytes = bytes;
        // Zero on allocation. Several buffers are only partly written by the
        // host and partly by the array, and a whole-buffer flush then pushes
        // whatever the host never wrote back to the device. Without this the
        // bytes a kernel reads out of those holes are the allocator's leftovers
        // and differ from process to process.
        std::memset(buf->bo.map(), 0, bytes);
        buf->bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: failed to allocate %zu-byte BO: %s\n", "xdna-runtime", bytes, e.what());
        delete buf;
        return nullptr;
    }
    return buf;
}

xdna_buffer * xdna_buffer_sub(xdna_buffer * parent, size_t offset, size_t bytes) {
    if (!parent || offset + bytes > parent->bytes) {
        GGML_LOG_ERROR("%s: buffer view %zu+%zu outside a %zu-byte BO\n",
                       "xdna-runtime", offset, bytes, parent ? parent->bytes : 0);
        return nullptr;
    }
    xdna_buffer * buf = new xdna_buffer;
    try {
        buf->bo    = xrt::bo(parent->bo, bytes, offset);
        buf->bytes = bytes;
        buf->root     = parent->root ? parent->root : parent;
        buf->root_off = parent->root_off + offset;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: failed to view %zu+%zu of a BO: %s\n",
                       "xdna-runtime", offset, bytes, e.what());
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

void xdna_buffer_sync_to_device(xdna_buffer * buf) {
    if (buf) {
        buf->bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
}

void xdna_buffer_sync_to_device_range(xdna_buffer * buf, size_t bytes, size_t offset) {
    if (buf && bytes) {
        buf->bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, bytes, offset);
    }
}

void xdna_buffer_sync_from_device(xdna_buffer * buf) {
    if (buf) {
        buf->bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    }
}

// Write a range back through the caches. The pattern a mark leaves is written
// by the host, so its lines are dirty; a device write to the same address can
// be overtaken by them when they are evicted. Nothing else here needs this:
// the host's own activation writes are read only by the array, which waits for
// them, and a read leaves its lines clean.
static void xdna_cache_writeback(void * p, size_t bytes) {
#if defined(__x86_64__) || defined(__i386__)
    const size_t line = 64;
    uintptr_t a = (uintptr_t) p & ~(uintptr_t) (line - 1);
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

static void xdna_poison_range(void * p, size_t bytes) {
    uint32_t * words = (uint32_t *) p;
    const size_t n = bytes / sizeof(uint32_t);
    for (size_t i = 0; i < n; i++) {
        words[i] = XDNA_POISON_F32;
    }
}

static bool xdna_poison_any(const void * p, size_t bytes) {
    const uint32_t * words = (const uint32_t *) p;
    const size_t n = bytes / sizeof(uint32_t);
#if defined(__x86_64__) || defined(__i386__)
    const __m128i pat = _mm_set1_epi32((int) XDNA_POISON_F32);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m128i v = _mm_loadu_si128((const __m128i *) (words + i));
        if (_mm_movemask_epi8(_mm_cmpeq_epi32(v, pat)) != 0) {
            return true;
        }
    }
    for (; i < n; i++) {
        if (words[i] == XDNA_POISON_F32) {
            return true;
        }
    }
    return false;
#else
    for (size_t i = 0; i < n; i++) {
        if (words[i] == XDNA_POISON_F32) {
            return true;
        }
    }
    return false;
#endif
}

// The pattern is four bytes wide, so a range is marked and checked a word at a
// time; a byte count that is not a multiple of four keeps its tail unmarked
// and unread, which is what every caller reads anyway.
static size_t xdna_markable(size_t bytes) {
    return bytes & ~(size_t) (sizeof(uint32_t) - 1);
}

// The pass budget and its pause. Bounded by the drain, not by a guess: the
// pattern is the evidence, and it disappears exactly when the device's write
// lands. A pass is a pause, not a timer sleep - sleep_for(2us) costs a tick
// and measures about 80us.
static int xdna_settle_passes(void) {
    static const int passes = xdna_env_int("GGML_XDNA_SETTLE_PASSES", 2000);
    return passes;
}

static void xdna_settle_pause(void) {
    static const int pause_loops = xdna_env_int("GGML_XDNA_SETTLE_PAUSE", 400);
    for (int i = 0; i < pause_loops; i++) {
#if defined(__x86_64__) || defined(__i386__)
        _mm_pause();
#endif
    }
}

static void xdna_settle_give_up(size_t bytes, size_t offset) {
    GGML_LOG_WARN("%s: %zu bytes at %zu still hold the mark after %d passes; "
                  "the device has not written them\n",
                  "xdna-runtime", bytes, offset, xdna_settle_passes());
}

void xdna_buffer_mark(xdna_buffer * buf, size_t bytes, size_t offset) {
    if (!buf || bytes == 0 || offset >= buf->bytes) {
        return;
    }
    if (bytes > buf->bytes - offset) {
        bytes = buf->bytes - offset;
    }
    bytes = xdna_markable(bytes);
    if (bytes == 0) {
        return;
    }
    uint8_t * base = (uint8_t *) buf->bo.map() + offset;
    xdna_poison_range(base, bytes);
    xdna_cache_writeback(base, bytes);
}

bool xdna_buffer_download(xdna_buffer * buf, void * dst, size_t bytes,
                          size_t offset) {
    if (!buf || !dst || bytes == 0 || offset >= buf->bytes) {
        return false;
    }
    if (bytes > buf->bytes - offset) {
        bytes = buf->bytes - offset;
    }
    bytes = xdna_markable(bytes);
    if (bytes == 0) {
        return false;
    }
    uint8_t * base = (uint8_t *) buf->bo.map() + offset;
    for (int pass = 0;; pass++) {
        std::memcpy(dst, base, bytes);
        if (!xdna_poison_any(dst, bytes)) {
            return true;
        }
        if (pass >= xdna_settle_passes()) {
            xdna_settle_give_up(bytes, offset);
            return false;
        }
        xdna_settle_pause();
        xdna_cache_writeback(base, bytes);
    }
}

uint8_t * xdna_buffer_wait_written(xdna_buffer * buf, size_t bytes,
                                   size_t offset) {
    if (!buf || bytes == 0 || offset >= buf->bytes) {
        return nullptr;
    }
    if (bytes > buf->bytes - offset) {
        bytes = buf->bytes - offset;
    }
    bytes = xdna_markable(bytes);
    if (bytes == 0) {
        return nullptr;
    }
    uint8_t * base = (uint8_t *) buf->bo.map() + offset;
    for (int pass = 0;; pass++) {
        if (!xdna_poison_any(base, bytes)) {
            return base;
        }
        if (pass >= xdna_settle_passes()) {
            xdna_settle_give_up(bytes, offset);
            return nullptr;
        }
        xdna_settle_pause();
        xdna_cache_writeback(base, bytes);
    }
}

void xdna_buffer_sync_from_device_range(xdna_buffer * buf, size_t bytes, size_t offset) {
    if (buf && bytes) {
        buf->bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE, bytes, offset);
    }
}

void xdna_buffer_read_settled(xdna_buffer * buf, void * dst, size_t bytes,
                              size_t offset) {
    if (!buf || !dst || bytes == 0) {
        return;
    }
    if (offset >= buf->bytes) {
        return;
    }
    if (bytes > buf->bytes - offset) {
        bytes = buf->bytes - offset;
    }
    uint8_t * base = (uint8_t *) buf->bo.map() + offset;
    // Spin, not sleep: a sleep_for(20us) costs a timer tick and measured about
    // 80us, and this is paid once per read.
    const auto spin_us = [](int us) {
        const std::chrono::steady_clock::time_point until =
            std::chrono::steady_clock::now() + std::chrono::microseconds(us);
        while (std::chrono::steady_clock::now() < until) {
        }
    };
    static const int settle_us = xdna_env_int("GGML_XDNA_SETTLE_US", 20);
    xdna_buffer_sync_from_device(buf);
    std::memcpy(dst, base, bytes);
    for (int attempt = 0; attempt < 4; attempt++) {
        if (settle_us > 0) {
            spin_us(settle_us);
        }
        xdna_buffer_sync_from_device(buf);
        if (std::memcmp(dst, base, bytes) == 0) {
            return;
        }
        std::memcpy(dst, base, bytes);
    }
}

// --- execution -------------------------------------------------------------

// Configure a run for `kern` with `n_args` host buffers (ABI: 0=opcode,
// 1=instruction BO, 2=ninstr, 3..=host buffers).
static xrt::run make_run(xdna_kernel * kern, xdna_buffer ** args, size_t n_args) {
    xrt::run run(kern->kernel);
    run.set_arg(0, XAIE_NPU_OPCODE_RUN);
    run.set_arg(1, kern->insts_bo);
    // Instruction words, not bytes: the stream is a TXN blob that carries its
    // own end, so a count four times too large ran anyway, but it is not what
    // the argument means.
    run.set_arg(2, (int64_t) (kern->insts_bytes / (int64_t) sizeof(uint32_t)));
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
    int k = 0;                        // runs a list; 0 = not batching
    std::vector<std::unique_ptr<xrt::runlist>> sent;
    std::unique_ptr<xrt::runlist> cur;
    const xrt::hw_context * cur_ctx = nullptr;
    int n_cur = 0;
    int lead = 0;                     // runs still to start on their own
    std::vector<xrt::run *> direct;   // the ones that were
    // token mode: the runs' streams joined into one command at the send
    bool token = false;
    int  join  = 0;                   // runs a joined command, 0: all of them
    struct entry {
        xdna_kernel * kern;
        xrt::run *    run;
        std::vector<xdna_buffer *> args;
    };
    std::vector<entry> tok;
    std::vector<xrt::run> tok_runs;   // sent, to wait for
    // the joined commands' kernels and instruction BOs, one a command of the
    // token (a BO in flight cannot be rewritten), reused token to token
    std::vector<xdna_kernel> joined;
    std::vector<size_t> joined_cap;
    std::vector<uint32_t> words;
};
xdna_batch_state g_batch;

constexpr uint32_t APERTURE = 0x80000000u;   // xdna-seq.cpp: args >= 5
constexpr uint32_t XLAT_ARGS = 5;

// Words a TXN op takes (xdna-seq.h); 0 for one this does not know.
uint32_t op_words(uint32_t op) {
    switch (op) {
        case 0x00: return 6;    // write
        case 0x01: return 12;   // blockwrite
        case 0x03: return 7;    // maskwrite
        case 0x80: return 4;    // wait for a task-complete token
        case 0x81: return 12;   // DDR patch
        default:   return 0;
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
    std::vector<xdna_buffer *> roots;
    // Roots take slots 5 to 9, the ones whose offset carries the DDR aperture
    // and which the firmware does not translate (xdna-seq.cpp) - the slots the
    // per-layer streams already name for their weights.
    constexpr int slot0 = (int) XLAT_ARGS;
    const auto slot_of = [&](xdna_buffer * a) -> int {
        xdna_buffer * r = a->root ? a->root : a;
        for (size_t i = 0; i < roots.size(); i++) {
            if (roots[i] == r) {
                return slot0 + (int) i;
            }
        }
        roots.push_back(r);
        return slot0 + (int) roots.size() - 1;
    };
    std::vector<uint32_t> & w = b.words;
    w.assign(4, 0);
    uint32_t n_instr = 0;
    size_t prev_start = 4, cur_start = 4;
    for (auto & e : b.tok) {
        prev_start = cur_start;
        cur_start  = w.size();
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
        constexpr bool hoist = true;
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
                if (in[j] == 0x01 && j + 12 + 12 + 6 <= in.size() && in[j + 12] == 0x81 &&
                    in[j + 24] == 0x00) {
                    const uint32_t r = in[j + 2], col = (r >> 25) & 0x7F;
                    const uint32_t bd = ((r & 0xFFFFF) - 0x1D000) / 0x20;
                    const uint32_t pr = in[j + 26] & 0xFFFFF, pcol = (in[j + 26] >> 25) & 0x7F;
                    const bool mm2s = pr >= 0x1D214 && pr < 0x1D224;
                    if (bd == 0 && pcol == col && mm2s && (in[j + 28] & 0xF) == 0) {
                        // the previous run's tail since its last wait before the
                        // trailing block must not write (col, bd 0)
                        bool clash = false;
                        size_t tail0 = prev_start;
                        for (size_t q = prev_start; q < tw;) {
                            const uint32_t qn = op_words(w[q]);
                            if (qn == 0) { clash = true; break; }
                            if (w[q] == 0x80) tail0 = q;
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
                            for (size_t q = j; q < j + 30; q++) moved[q] = 1;
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
                if (qn == 0) break;
                if (moved[q]) {
                    ins.insert(ins.end(), in.begin() + q, in.begin() + q + qn);
                }
                q += qn;
            }
            // relocate the patches in `ins` as the main loop would
            for (size_t q = 0; q < ins.size();) {
                const uint32_t qn = op_words(ins[q]);
                if (ins[q] == 0x81) {
                    const uint32_t arg = ins[q + 8];
                    uint32_t off = ins[q + 10];
                    if (arg >= XLAT_ARGS) off &= ~APERTURE;
                    if (arg >= e.args.size() || !e.args[arg]) return token_start_each();
                    xdna_buffer * a = e.args[arg];
                    const int s2 = slot_of(a);
                    const uint64_t noff = (uint64_t) a->root_off + off;
                    if (s2 >= 10 || noff >= APERTURE) return token_start_each();
                    ins[q + 8]  = (uint32_t) s2;
                    ins[q + 10] = (uint32_t) noff | (s2 >= (int) XLAT_ARGS ? APERTURE : 0u);
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
            w.insert(w.end(), in.begin() + i, in.begin() + i + nw);
            if (in[i] == 0x81) {
                const uint32_t arg = in[i + 8];
                uint32_t off = in[i + 10];
                if (arg >= XLAT_ARGS) {
                    off &= ~APERTURE;
                }
                if (arg >= e.args.size() || !e.args[arg]) {
                    return token_start_each();
                }
                xdna_buffer * a = e.args[arg];
                const int s = slot_of(a);
                const uint64_t noff = (uint64_t) a->root_off + off;
                if (s >= 10 || (s < slot0) || noff >= APERTURE ||
                    (slot0 < (int) XLAT_ARGS && s >= (int) XLAT_ARGS)) {
                    return token_start_each();
                }
                w[at + 8]  = (uint32_t) s;
                w[at + 10] = (uint32_t) noff | (s >= (int) XLAT_ARGS ? APERTURE : 0u);
            }
            i += nw;
        }
    }
    w[2] = n_instr;
    w[3] = (uint32_t) (w.size() * sizeof(uint32_t));
    const size_t bytes = w.size() * sizeof(uint32_t);
    try {
        const size_t ji = b.tok_runs.size();
        if (b.joined.size() <= ji) {
            b.joined.resize(ji + 1);
            b.joined_cap.resize(ji + 1, 0);
        }
        xdna_kernel & j = b.joined[ji];
        xdna_kernel * k0 = b.tok[0].kern;
        if (j.context != k0->context || b.joined_cap[ji] < bytes) {
            j.device  = k0->device;
            j.context = k0->context;
            j.kernel  = k0->kernel;
            b.joined_cap[ji] = bytes * 2;
            j.insts_bo = xrt::bo(j.device, b.joined_cap[ji], xrt::bo::flags::cacheable,
                                 j.kernel.group_id(1));
        }
        std::memcpy(j.insts_bo.map(), w.data(), bytes);
        j.insts_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, bytes, 0);
        xrt::run run(j.kernel);
        run.set_arg(0, XAIE_NPU_OPCODE_RUN);
        run.set_arg(1, j.insts_bo);
        run.set_arg(2, (uint32_t) w.size());
        for (int s = 0; s < 10; s++) {
            // every slot below the first root's gets one too: the argument list
            // is positional
            const int r = s < slot0 ? 0 : s - slot0;
            if (r < (int) roots.size()) {
                run.set_arg(3 + s, roots[(size_t) r]->bo);
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
    b.n_cur = 0;
    return true;
}
} // namespace

void xdna_batch_begin(int k) {
    g_batch.token = k < 0;
    g_batch.join  = k < -1 ? -k : 0;
    g_batch.k = k != 0 ? 1 << 30 : 0;
    if (k > 0) {
        g_batch.k = k;
    }
    // The token's first run starts on its own: the array has work while the
    // host prepares the first list, instead of waiting for all of it.
    static const int lead = [] {
        const char * e = getenv("GGML_XDNA_BATCH_LEAD");
        return e ? atoi(e) : 1;
    }();
    g_batch.lead = lead;
}

bool xdna_batch_active(void) {
    return g_batch.k > 0;
}

bool xdna_run_submit(xdna_kernel * kern, xrt::run & run, xdna_buffer * const * args,
                     size_t n_args) {
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
    auto & b = g_batch;
    bool ok = batch_send();
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
    b.k = 0;
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
    try {
        // Block, do not poll. Polling kept a core at full power for 70% of a
        // decode token (it waits once a token for its whole queue, ~20 ms) -
        // the host energy FLM does not spend - and bought nothing: blocking
        // measured the same tok/s in the server, at 0.29 of the CPU time
        // polling 200 us before blocking took. GGML_XDNA_SPIN_US polls that
        // long first; -1 polls without limit, as it used to by default.
        static const int spin_us = xdna_env_int("GGML_XDNA_SPIN_US", 0);
        if (spin_us != 0) {
            const auto until = std::chrono::steady_clock::now() +
                               std::chrono::microseconds(spin_us < 0 ? 0 : spin_us);
            for (;;) {
                const ert_cmd_state s = run.state();
                if (s == ERT_CMD_STATE_COMPLETED) {
                    return true;
                }
                if (s == ERT_CMD_STATE_ERROR || s == ERT_CMD_STATE_ABORT ||
                    s == ERT_CMD_STATE_TIMEOUT) {
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
        GGML_LOG_ERROR("%s: kernel wait exception: %s\n", "xdna-runtime", e.what());
        return false;
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

xdna_kernel * xdna_kernel_pool_get_built(xdna_kernel_pool * pool, const std::string & name,
                                         const char * xclbin_name, const uint32_t * insts, size_t n_words) {
    std::lock_guard<std::mutex> lock(pool->kernel_mutex);

    auto it = pool->kernels.find(name);
    if (it != pool->kernels.end()) {
        return it->second;
    }

    // nullptr is sticky: a failed lookup is not retried.
    pool->kernels[name] = nullptr;

    // `xclbin_name` is the artifact stem; resolve it to a real path.
    xdna_kernel * kern = nullptr;
    for (const fs::path & dir : xdna_kernel_search_dirs()) {
        const fs::path xclbin = dir / (std::string(xclbin_name) + ".xclbin");
        if (fs::exists(xclbin)) {
            kern = xdna_kernel_load_hw(pool->device, xclbin.c_str());
            break;
        }
    }
    if (!kern) {
        GGML_LOG_WARN("%s: kernel %s (hw %s) not found\n", "xdna-runtime",
                      name.c_str(), xclbin_name);
        return nullptr;
    }
    if (!xdna_kernel_bind_insts(pool->device, kern, insts, n_words)) {
        xdna_kernel_free(kern);
        return nullptr;
    }
    pool->kernels[name] = kern;
    return kern;
}

xdna_buffer * xdna_kernel_pool_acquire_buffer(xdna_kernel_pool * pool, size_t bytes) {
    {
        std::lock_guard<std::mutex> lock(pool->pool_mutex);
        size_t best = pool->pool.size();
        size_t best_size = 0;
        for (size_t i = 0; i < pool->pool.size(); i++) {
            if (!pool->pool[i].buf) {
                continue;
            }
            const size_t sz = pool->pool[i].buf->bytes;
            if (sz >= bytes && (best == pool->pool.size() || sz < best_size)) {
                best = i;
                best_size = sz;
            }
        }
        if (best != pool->pool.size()) {
            xdna_buffer * buf = pool->pool[best].buf;
            pool->pool.erase(pool->pool.begin() + best);
            return buf;
        }
    }
    return xdna_buffer_alloc(pool->device, bytes);
}

void xdna_kernel_pool_release_buffer(xdna_kernel_pool * pool, xdna_buffer * buf) {
    if (!buf) {
        return;
    }
    std::lock_guard<std::mutex> lock(pool->pool_mutex);
    pool->pool.push_back({buf, ++pool->pool_tick});

    if (pool->pool.size() > xdna_kernel_pool::MAX_POOL_SIZE) {
        size_t lru = 0;
        for (size_t i = 1; i < pool->pool.size(); i++) {
            if (pool->pool[i].seq < pool->pool[lru].seq) {
                lru = i;
            }
        }
        xdna_buffer_free(pool->pool[lru].buf);
        pool->pool.erase(pool->pool.begin() + lru);
    }
}
