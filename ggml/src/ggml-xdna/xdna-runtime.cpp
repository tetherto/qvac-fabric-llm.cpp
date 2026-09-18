#include "xdna-runtime.h"

#include "ggml-impl.h"
#include "xdna-util.h"

#include <xrt/experimental/xrt_kernel.h>
#include <xrt/experimental/xrt_xclbin.h>
#include <xrt/xrt_kernel.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <system_error>

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
        return true;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: failed to bind instruction stream: %s\n", "xdna-runtime", e.what());
        return false;
    }
}

void xdna_kernel_free(xdna_kernel * kern) {
    delete kern;
}

// --- buffer ----------------------------------------------------------------

xdna_buffer * xdna_buffer_alloc(xdna_device * dev, size_t bytes) {
    if (!dev) {
        GGML_LOG_ERROR("%s: buffer alloc: no device\n", "xdna-runtime");
        return nullptr;
    }
    xdna_buffer * buf = new xdna_buffer;
    try {
        buf->bo    = xrt::bo(dev->device, bytes, xrt::bo::flags::host_only, 0);
        buf->bytes = bytes;
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
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: failed to view %zu+%zu of a BO: %s\n",
                       "xdna-runtime", offset, bytes, e.what());
        delete buf;
        return nullptr;
    }
    return buf;
}

void xdna_buffer_free(xdna_buffer * buf) {
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

// --- execution -------------------------------------------------------------

// Configure a run for `kern` with `n_args` host buffers (ABI: 0=opcode,
// 1=instruction BO, 2=ninstr, 3..=host buffers).
static xrt::run make_run(xdna_kernel * kern, xdna_buffer ** args, size_t n_args) {
    xrt::run run(kern->kernel);
    run.set_arg(0, XAIE_NPU_OPCODE_RUN);
    run.set_arg(1, kern->insts_bo);
    run.set_arg(2, kern->insts_bytes);
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

bool xdna_run_restart(xrt::run & run) {
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
        // Poll before blocking. A dispatch shorter than XRT's completion path
        // measures as that path and not as its own work: the decode's short
        // ones sit on a floor of about 125 us where their weights need 60.
        // The decode is sequential, so the thread that would block here has
        // nothing else to do with the time.
        static const bool spin = xdna_env_on("GGML_XDNA_SPIN");
        if (spin) {
            for (int i = 0; i < (1 << 22); i++) {
                const ert_cmd_state s = run.state();
                if (s == ERT_CMD_STATE_COMPLETED) {
                    return true;
                }
                if (s == ERT_CMD_STATE_ERROR || s == ERT_CMD_STATE_ABORT ||
                    s == ERT_CMD_STATE_TIMEOUT) {
                    GGML_LOG_ERROR("%s: kernel spin state %d\n", "xdna-runtime", (int) s);
                    return false;
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
