#include "../ggml/src/ggml-opencl/ggml-opencl-workgroup.h"

#include <cassert>

static constexpr size_t ADRENO_WAVE_SIZE = 64;
static constexpr int WIDE_SIMDGROUPS = 16;
static constexpr size_t REQUIRED_WORKGROUP_SIZE = ADRENO_WAVE_SIZE * WIDE_SIMDGROUPS;
static constexpr size_t LIMIT_BELOW_REQUIRED = REQUIRED_WORKGROUP_SIZE - 1;
static constexpr size_t LIMIT_EQUAL_REQUIRED = REQUIRED_WORKGROUP_SIZE;
static constexpr size_t LIMIT_ABOVE_REQUIRED = REQUIRED_WORKGROUP_SIZE + 1;
static constexpr int WIDE_MAX_M = 2048;
static constexpr int Q4_M = 512;
static constexpr int Q4_K_BLOCKS = 28;
static constexpr int Q8_M = 896;
static constexpr int Q8_K_BLOCKS = 28;

static void verify_q4_limit(size_t kernel_limit, bool expected) {
    assert(ggml_opencl_should_use_wide_gemv(
        ADRENO_WAVE_SIZE,
        WIDE_SIMDGROUPS,
        kernel_limit,
        Q4_M,
        WIDE_MAX_M,
        Q4_K_BLOCKS) == expected);
}

static void verify_q8_limit(size_t kernel_limit, bool expected) {
    assert(ggml_opencl_should_use_wide_gemv(
        ADRENO_WAVE_SIZE,
        WIDE_SIMDGROUPS,
        kernel_limit,
        Q8_M,
        WIDE_MAX_M,
        Q8_K_BLOCKS) == expected);
}

static void test_q4_kernel_limits() {
    verify_q4_limit(LIMIT_BELOW_REQUIRED, false);
    verify_q4_limit(LIMIT_EQUAL_REQUIRED, true);
    verify_q4_limit(LIMIT_ABOVE_REQUIRED, true);
}

static void test_q8_kernel_limits() {
    verify_q8_limit(LIMIT_BELOW_REQUIRED, false);
    verify_q8_limit(LIMIT_EQUAL_REQUIRED, true);
    verify_q8_limit(LIMIT_ABOVE_REQUIRED, true);
}

int main() {
    test_q4_kernel_limits();
    test_q8_kernel_limits();
    return 0;
}
