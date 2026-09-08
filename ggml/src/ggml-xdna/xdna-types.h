#pragma once

// Transparent XDNA types. The backend always links XRT, so the structs hold
// XRT handles directly instead of hiding them behind opaque pointers.

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// The NPU device. One per process.
struct xdna_device {
    xrt::device device;
    std::string name;
    std::string description;
};

// A loaded kernel: one xclbin plus one instruction stream (insts BO) bound to
// a shared hw_context. All variants of the same xclbin share the context via
// the shared_ptr; the runtime keeps contexts alive for the process lifetime.
struct xdna_kernel {
    std::shared_ptr<xrt::hw_context> context;
    xrt::kernel                      kernel;
    xrt::bo                          insts_bo;
    int64_t                          insts_bytes = 0;
};

// A host-visible device buffer object (BO).
struct xdna_buffer {
    xrt::bo  bo;
    size_t   bytes = 0;
};

// Pool of kernels: auto-scans the search dirs for kernel artifacts (xclbin
// files), exposes their names, and lazily loads kernels on demand. Backend-specific selection (which kernel fits an op) is
// done by the caller against the names. The runtime caches one hw_context per
// xclbin uuid, so variants of one xclbin share it with no reload penalty.
// Also pools host-visible buffers. Data only; the pool API lives in
// xdna-runtime.h as C-style functions.
struct xdna_kernel_pool {
    struct pool_entry {
        xdna_buffer * buf;
        uint64_t      seq;   // idle stamp, lower = older
    };

    static constexpr size_t MAX_POOL_SIZE = 16;

    xdna_device * device = nullptr;

    std::vector<std::string>                       names;    // artifact stems
    std::unordered_map<std::string, xdna_kernel *> kernels;  // name -> kernel
    std::mutex                                     kernel_mutex;

    std::vector<pool_entry> pool;
    uint64_t                pool_tick = 0;
    std::mutex              pool_mutex;
};
