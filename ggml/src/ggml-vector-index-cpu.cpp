// ggml-vector-index-cpu.cpp - CPU feature gates that pick between TurboVec
// kernels the way upstream turbovec does.

#include "ggml-vector-index-impl.h"

#if defined(__APPLE__) && (defined(__aarch64__) || defined(_M_ARM64))
#include <sys/sysctl.h>
#elif (defined(__linux__) || defined(__ANDROID__)) && defined(__aarch64__)
#include <sys/auxv.h>
#endif

namespace {

std::atomic<int> g_vector_major_override{ -1 };
std::atomic<bool> g_dot_product_disabled{ false };

bool detect_vector_major() {
#if defined(__aarch64__) || defined(_M_ARM64)
    // ARMv8.2 dot product (SDOT), upstream `use_vector_major` on aarch64.
#if defined(__APPLE__)
    int value = 0;
    size_t size = sizeof(value);
    return sysctlbyname("hw.optional.arm.FEAT_DotProd", &value, &size, nullptr, 0) == 0 && value != 0;
#elif defined(__linux__) || defined(__ANDROID__)
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1 << 20)
#endif
    return (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#elif defined(_WIN32)
    return IsProcessorFeaturePresent(43 /* PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE */) != 0;
#else
    return false;
#endif
#elif defined(__x86_64__) || defined(_M_X64)
    // Upstream also runs these kernels on x86 with AVX-512 F, BW, VBMI and
    // VNNI. This port has no AVX-512 kernel yet, and its portable fallback
    // is far slower than the LUT scan, so x86 keeps the LUT scan.
    return false;
#else
    return false;
#endif
}

}  // namespace

bool turbovec_cpu_dot_product() {
    static const bool detected = detect_vector_major();
    return detected && !g_dot_product_disabled.load(std::memory_order_relaxed);
}

bool turbovec_cpu_vector_major() {
    const int forced = g_vector_major_override.load(std::memory_order_relaxed);
    if (forced >= 0) {
        return forced != 0;
    }
    return turbovec_cpu_dot_product();
}

#ifdef GGML_VEC_INDEX_TEST_HOOKS
void turbovec_force_vector_major_for_test(int mode) {
    g_vector_major_override.store(mode, std::memory_order_relaxed);
}

void turbovec_disable_dot_product_for_test(int disabled) {
    g_dot_product_disabled.store(disabled != 0, std::memory_order_relaxed);
}
#endif
