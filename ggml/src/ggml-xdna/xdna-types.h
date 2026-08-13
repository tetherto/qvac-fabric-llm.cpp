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
#include <string>

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
