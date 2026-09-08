#include "xdna-runtime.h"

#include "ggml-impl.h"

#include <system_error>

namespace fs = std::filesystem;

void xdna_kernel_pool_scan(xdna_kernel_pool * pool) {
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
        GGML_LOG_WARN("%s: kernel %s (hw %s) not found\n", "xdna-kernel-pool",
                      name.c_str(), xclbin_name);
        pool->kernels[name] = nullptr;
        return nullptr;
    }
    if (!xdna_kernel_bind_insts(pool->device, kern, insts, n_words)) {
        xdna_kernel_free(kern);
        pool->kernels[name] = nullptr;
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
