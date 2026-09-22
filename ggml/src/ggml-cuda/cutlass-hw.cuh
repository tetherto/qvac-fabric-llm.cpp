// shared by the CUTLASS wrappers (mmf8-cutlass.cu, fattn-cutlass.cu); compiled for sm_90a only
#pragma once

#include "cutlass/kernel_hardware_info.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>

// SM count per device, queried once (the current device is set by the caller); duplicate first queries are harmless
static inline cutlass::KernelHardwareInfo ggml_cutlass_hw_info_for_current_device() {
    constexpr int max_devices = 16;
    static std::atomic<int> sm_counts[max_devices];
    cutlass::KernelHardwareInfo info;
    const cudaError_t err = cudaGetDevice(&info.device_id);
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: cudaGetDevice failed: %s\n", __func__, cudaGetErrorString(err));
        abort();
    }
    if (info.device_id < 0 || info.device_id >= max_devices) {
        info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(info.device_id);
        return info;
    }
    int n = sm_counts[info.device_id].load(std::memory_order_relaxed);
    if (n == 0) {
        n = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(info.device_id);
        sm_counts[info.device_id].store(n, std::memory_order_relaxed);
    }
    info.sm_count = n;
    return info;
}
