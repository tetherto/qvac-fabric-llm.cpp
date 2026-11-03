// ggml-vector-index-cpu.cpp - CPU feature gates that pick between TurboVec
// kernels the way upstream turbovec does.

#include "ggml-vector-index-impl.h"

#if defined(__APPLE__) && (defined(__aarch64__) || defined(_M_ARM64))
#include <sys/sysctl.h>
#elif (defined(__linux__) || defined(__ANDROID__)) && defined(__aarch64__)
#include <sys/auxv.h>
#endif

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) && defined(_MSC_VER)
#include <intrin.h>
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

VecIndexCpuFeatures detect_cpu_features() {
    VecIndexCpuFeatures f;
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#if defined(_MSC_VER)
    int regs[4] = {};
    __cpuid(regs, 0);
    const int max_leaf = regs[0];
    if (max_leaf < 1) {
        return f;
    }
    __cpuidex(regs, 1, 0);
    const unsigned ecx = static_cast<unsigned>(regs[2]);
    f.ssse3 = (ecx & (1u << 9)) != 0;
    f.sse42 = (ecx & (1u << 20)) != 0;
    const bool osxsave = (ecx & (1u << 27)) != 0;
    const bool avx = (ecx & (1u << 28)) != 0;
    const bool fma = (ecx & (1u << 12)) != 0;
    const unsigned long long xcr0 = osxsave ? _xgetbv(0) : 0;
    // The OS must save the YMM (and for AVX-512 the opmask and ZMM) state.
    const bool ymm_state = (xcr0 & 0x6) == 0x6;
    const bool zmm_state = (xcr0 & 0xe6) == 0xe6;
    if (max_leaf >= 7) {
        __cpuidex(regs, 7, 0);
        const unsigned ebx = static_cast<unsigned>(regs[1]);
        f.avx2 = avx && fma && ymm_state && (ebx & (1u << 5)) != 0;
        f.avx512f = f.avx2 && zmm_state && (ebx & (1u << 16)) != 0;
    }
#elif defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    f.ssse3 = __builtin_cpu_supports("ssse3");
    f.sse42 = __builtin_cpu_supports("sse4.2");
    // The AVX2 object is compiled with FMA enabled, as the search kernels
    // there require, so it runs only where both are present.
    f.avx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    f.avx512f = f.avx2 && __builtin_cpu_supports("avx512f");
#endif
#elif defined(__aarch64__) || defined(_M_ARM64)
#if defined(__APPLE__)
    int value = 0;
    size_t size = sizeof(value);
    f.arm_crc32 = sysctlbyname("hw.optional.armv8_crc32", &value, &size, nullptr, 0) == 0 && value != 0;
#elif defined(__linux__) || defined(__ANDROID__)
#ifndef HWCAP_CRC32
#define HWCAP_CRC32 (1 << 7)
#endif
    f.arm_crc32 = (getauxval(AT_HWCAP) & HWCAP_CRC32) != 0;
#elif defined(_WIN32)
    f.arm_crc32 = IsProcessorFeaturePresent(31 /* PF_ARM_V8_CRC32_INSTRUCTIONS_AVAILABLE */) != 0;
#elif defined(__ARM_FEATURE_CRC32)
    f.arm_crc32 = true;
#endif
#endif
    return f;
}

}  // namespace

const VecIndexCpuFeatures & vec_index_cpu_features() {
    static const VecIndexCpuFeatures features = detect_cpu_features();
    return features;
}

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
