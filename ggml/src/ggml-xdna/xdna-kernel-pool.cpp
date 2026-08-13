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
            // Only usable kernels: the instruction stream must sit next to it.
            if (!fs::exists(dir / (entry.path().stem().string() + ".insts.bin"))) {
                continue;
            }
            pool->names.push_back(entry.path().stem().string());
        }
    }
}

xdna_kernel * xdna_kernel_pool_get(xdna_kernel_pool * pool, const std::string & name) {
    std::lock_guard<std::mutex> lock(pool->kernel_mutex);

    auto it = pool->kernels.find(name);
    if (it != pool->kernels.end()) {
        return it->second;
    }

    // nullptr is sticky: a failed lookup is not retried.
    pool->kernels[name] = nullptr;

    xdna_kernel * kern = xdna_kernel_load_search(pool->device,
            (name + ".xclbin").c_str(), (name + ".insts.bin").c_str());
    if (!kern) {
        GGML_LOG_WARN("%s: kernel %s not found (set GGML_XDNA_KERNELS_DIR)\n",
                      "xdna-kernel-pool", name.c_str());
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

void xdna_kernel_pool_clear(xdna_kernel_pool * pool) {
    for (auto & kv : pool->kernels) {
        if (kv.second) {
            xdna_kernel_free(kv.second);
        }
    }
    pool->kernels.clear();
    for (const auto & e : pool->pool) {
        xdna_buffer_free(e.buf);
    }
    pool->pool.clear();
}
