// ggml-vector-index-cpu.cpp - CPU feature gates that pick between TurboVec
// kernels the way upstream turbovec does.

#include "ggml-vector-index-impl.h"

#include <cstdlib>
#include <cstring>

#if defined(__APPLE__) && (defined(__aarch64__) || defined(_M_ARM64))
#include <sys/sysctl.h>
#elif (defined(__linux__) || defined(__ANDROID__)) && defined(__aarch64__)
#include <sys/auxv.h>
#endif

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) && defined(_MSC_VER)
#include <intrin.h>
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <cpuid.h>
#endif

namespace {

std::atomic<int> g_vector_major_override{ -1 };
std::atomic<bool> g_dot_product_disabled{ false };
std::atomic<int> g_vm8_override{ -1 };
std::atomic<bool> g_i8mm_disabled{ false };
std::atomic<int> g_vm8_2bit_override{ -1 };
std::atomic<int> g_x86_rules_override{ -1 };

// Upstream reads these once per process; set to anything but "0", either
// one keeps the classic layout (TURBOVEC_NO_VNNI is the older spelling
// from when the layout was x86-only).
bool env_flag_set(const char * name) {
    const char * value = std::getenv(name);
    return value != nullptr && std::strcmp(value, "0") != 0;
}

// The hardware half of upstream's `use_vector_major`: the instructions its
// dot-product kernels need, without the environment opt-out.
// It follows the CPU, not the build: where the kernels are not compiled in
// (MSVC cl on ARM64 has no SDOT kernel; an x86 build without the VNNI
// objects), the vector-major layout is scored by the scalar models, which
// keep upstream's results but run far slower than the LUT scan.
bool detect_dot_product() {
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
    // AVX-512 F, BW, VBMI and VNNI: `vpermb` for the 2- and 3-bit scan and
    // `vpdpbusd` for every vector-major kernel.
    const VecIndexCpuFeatures & f = vec_index_cpu_features();
    return f.avx512f && f.avx512bw && f.avx512vbmi && f.avx512vnni;
#else
    return false;
#endif
}

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
// AVX-512 extensions from CPUID leaf 7 (EBX, ECX), only on top of AVX-512 F,
// whose check covers the OS saving the opmask and ZMM state. The search
// kernels use GFNI in its 512-bit form, so it is gated the same way.
void set_avx512_extensions(VecIndexCpuFeatures & f, unsigned ebx, unsigned ecx) {
    if (!f.avx512f) {
        return;
    }
    f.avx512bw = (ebx & (1u << 30)) != 0;
    f.avx512vbmi = (ecx & (1u << 1)) != 0;
    f.gfni = (ecx & (1u << 8)) != 0;
    f.avx512vnni = (ecx & (1u << 11)) != 0;
    f.avx512vpopcntdq = (ecx & (1u << 14)) != 0;
}
#endif

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
        set_avx512_extensions(f, ebx, static_cast<unsigned>(regs[2]));
    }
#elif defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    f.ssse3 = __builtin_cpu_supports("ssse3");
    f.sse42 = __builtin_cpu_supports("sse4.2");
    // The AVX2 object is compiled with FMA enabled, as the search kernels
    // there require, so it runs only where both are present.
    f.avx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    f.avx512f = f.avx2 && __builtin_cpu_supports("avx512f");
    // The extensions straight from CPUID leaf 7: older GCC and Clang do not
    // know every name (gfni, avx512vpopcntdq) to __builtin_cpu_supports.
    // avx512f above already checked that the OS saves the AVX-512 state.
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) != 0) {
        set_avx512_extensions(f, ebx, ecx);
    }
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

bool detect_i8mm() {
#if defined(__aarch64__) || defined(_M_ARM64)
    // ARMv8.6 int8 matrix multiply (SMMLA), upstream `have_i8mm`; optional
    // before v8.6, so detected like dot product.
#if defined(__APPLE__)
    int value = 0;
    size_t size = sizeof(value);
    return sysctlbyname("hw.optional.arm.FEAT_I8MM", &value, &size, nullptr, 0) == 0 && value != 0;
#elif defined(__linux__) || defined(__ANDROID__)
#ifndef HWCAP2_I8MM
#define HWCAP2_I8MM (1 << 13)
#endif
    return (getauxval(AT_HWCAP2) & HWCAP2_I8MM) != 0;
#elif defined(_WIN32)
    // Older SDKs lack the name; 66 is its value, as in ggml-feats.h.
    return IsProcessorFeaturePresent(66 /* PF_ARM_V82_I8MM_INSTRUCTIONS_AVAILABLE */) != 0;
#else
    return false;
#endif
#else
    return false;
#endif
}

}  // namespace

const VecIndexCpuFeatures & vec_index_cpu_features() {
    static const VecIndexCpuFeatures features = detect_cpu_features();
    return features;
}

bool turbovec_cpu_dot_product() {
    static const bool detected = detect_dot_product();
    return detected && !g_dot_product_disabled.load(std::memory_order_relaxed);
}

bool turbovec_cpu_vector_major() {
    const int forced = g_vector_major_override.load(std::memory_order_relaxed);
    if (forced >= 0) {
        return forced != 0;
    }
    // Upstream pack.rs `use_vector_major`, decided once per process.
    static const bool opted_out = env_flag_set("TURBOVEC_NO_VECTOR_MAJOR") || env_flag_set("TURBOVEC_NO_VNNI");
    return !opted_out && turbovec_cpu_dot_product();
}

bool turbovec_cpu_gfni() {
    // Upstream `have_gfni`: TURBOVEC_NO_GFNI (set at all) keeps the
    // baseline permute-dot kernel.
    static const bool enabled = std::getenv("TURBOVEC_NO_GFNI") == nullptr && vec_index_cpu_features().gfni;
    return enabled;
}

bool turbovec_x86_rules() {
    const int forced = g_x86_rules_override.load(std::memory_order_relaxed);
    if (forced >= 0) {
        return forced != 0;
    }
#if defined(__x86_64__) || defined(_M_X64)
    return true;
#else
    return false;
#endif
}

bool turbovec_cpu_i8mm() {
    // Upstream `have_i8mm`: TURBOVEC_NO_I8MM (set at all) keeps the SDOT
    // kernels and the vector-major layout.
    static const bool detected = std::getenv("TURBOVEC_NO_I8MM") == nullptr && detect_i8mm();
    return detected && !g_i8mm_disabled.load(std::memory_order_relaxed);
}

bool turbovec_cpu_vm8() {
    // Upstream pack.rs `use_vm8`: aarch64 with i8mm, on top of the
    // vector-major layout. No other kernel reads the layout.
    if (!turbovec_cpu_vector_major()) {
        return false;
    }
    const int forced = g_vm8_override.load(std::memory_order_relaxed);
    if (forced >= 0) {
        return forced != 0;
    }
#if defined(__aarch64__) || defined(_M_ARM64)
    return !turbovec_x86_rules() && turbovec_cpu_i8mm();
#else
    return false;
#endif
}

bool turbovec_vm8_2bit_enabled() {
    const int forced = g_vm8_2bit_override.load(std::memory_order_relaxed);
    if (forced >= 0) {
        return forced != 0;
    }
    // Upstream pack.rs `use_vm8_2bit`: read once per process, on only for
    // exactly "1".
    static const bool enabled = [] {
        const char * value = std::getenv("TURBOVEC_2BIT_VM8");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

#ifdef GGML_VEC_INDEX_TEST_HOOKS
void turbovec_force_vm8_for_test(int mode) {
    g_vm8_override.store(mode, std::memory_order_relaxed);
}

void turbovec_disable_i8mm_for_test(int disabled) {
    g_i8mm_disabled.store(disabled != 0, std::memory_order_relaxed);
}

void turbovec_force_vm8_2bit_for_test(int mode) {
    g_vm8_2bit_override.store(mode, std::memory_order_relaxed);
}

void turbovec_force_vector_major_for_test(int mode) {
    g_vector_major_override.store(mode, std::memory_order_relaxed);
}

void turbovec_force_x86_rules_for_test(int mode) {
    g_x86_rules_override.store(mode, std::memory_order_relaxed);
}

void turbovec_disable_dot_product_for_test(int disabled) {
    g_dot_product_disabled.store(disabled != 0, std::memory_order_relaxed);
}

// Whether this build compiled the kernels a SIMD parity check exercises, so
// a test can tell a kernel left out of the build from one this CPU lacks.
// Ids: 0 NEON, 1 AVX2, 2 AVX-512 F, 3 SSSE3, 4 ARMv8 CRC32, 5 SSE4.2,
// 6 AVX-512 VNNI, 7 AVX-512 GFNI, 8 AVX-512 BW, 9 AVX-512 VPOPCNTDQ.
int vec_index_simd_kernel_built_for_test(int id) {
    switch (id) {
        case 0:
            return GGML_VEC_INDEX_USE_NEON ? 1 : 0;
        case 1:
#if defined(GGML_VEC_INDEX_HAVE_AVX2_KERNEL)
            return 1;
#else
            return 0;
#endif
        case 2:
#if defined(GGML_VEC_INDEX_HAVE_AVX512_KERNEL)
            return 1;
#else
            return 0;
#endif
        case 3:
        case 5:
#if defined(GGML_VEC_INDEX_HAVE_SSE42_KERNEL)
            return 1;
#else
            return 0;
#endif
        case 4:
            // persistence.cpp builds the ARMv8 CRC32 path under these.
#if (defined(__aarch64__) || defined(_M_ARM64)) && !defined(__AARCH64EB__) && \
    (defined(_MSC_VER) || defined(__ARM_FEATURE_CRC32) || defined(__clang__) || (defined(__GNUC__) && __GNUC__ >= 10))
            return 1;
#else
            return 0;
#endif
        case 6:
#if defined(GGML_VEC_INDEX_HAVE_VNNI_KERNEL)
            return 1;
#else
            return 0;
#endif
        case 7:
#if defined(GGML_VEC_INDEX_HAVE_VNNI_KERNEL) && defined(GGML_VEC_INDEX_HAVE_GFNI_KERNEL)
            return 1;
#else
            return 0;
#endif
        case 8:
#if defined(GGML_VEC_INDEX_HAVE_AVX512BW_KERNEL)
            return 1;
#else
            return 0;
#endif
        case 9:
#if defined(GGML_VEC_INDEX_HAVE_VPOPCNT_KERNEL)
            return 1;
#else
            return 0;
#endif
        default:
            return 0;
    }
}
#endif
