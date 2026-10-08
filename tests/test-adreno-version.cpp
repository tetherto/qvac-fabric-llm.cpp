#include "ggml-adreno.h"

#include <cstdio>
#include <string>

static int g_failures = 0;

static void expect_version(const std::string & description, int expected) {
    const int got = ggml_adreno_version_from_description(description);
    if (got != expected) {
        std::printf("FAIL: \"%s\" -> %d (expected %d)\n", description.c_str(), got, expected);
        g_failures++;
    }
}

static void expect_policy(int min_adreno_version, bool load_opencl, bool unload_vulkan) {
    const ggml_adreno_backend_policy got = ggml_adreno_resolve_backend_policy(min_adreno_version);
    if (got.load_opencl != load_opencl || got.unload_vulkan != unload_vulkan) {
        std::printf("FAIL: policy(%d) -> {load_opencl=%d, unload_vulkan=%d} (expected {%d, %d})\n", min_adreno_version,
                    got.load_opencl, got.unload_vulkan, load_opencl, unload_vulkan);
        g_failures++;
    }
}

static void check_vulkan_descriptions() {
    expect_version("Adreno (TM) 830", 830);
    expect_version("Adreno (TM) 750", 750);
    expect_version("Adreno (TM) 660", 660);
    expect_version("Adreno 730", 730);
    expect_version("Adreno(TM)619", 619);
    expect_version("ADRENO 830", 830);
    expect_version("adreno 612", 612);
}

static void check_opencl_descriptions() {
    expect_version("QUALCOMM Adreno(TM) (OpenCL 3.0 Adreno(TM) 740)", 740);
    expect_version("QUALCOMM Adreno(TM) (OpenCL 3.0 Adreno(TM) 830)", 830);
    expect_version("QUALCOMM Adreno(TM)", -3);
}

static void check_non_adreno_descriptions() {
    expect_version("Mali-G715", -1);
    expect_version("Mali-G78 MP14", -1);
    expect_version("NVIDIA GeForce RTX 5090", -1);
    expect_version("AMD Radeon (RADV RAPHAEL_MENDOCINO)", -1);
    expect_version("Apple M2", -1);
    expect_version("llvmpipe (LLVM 20.1.2, 256 bits)", -1);
    expect_version("Intel(R) Arc(TM) A770 Graphics", -1);
    expect_version("", -1);
    expect_version("Adreno (TM)", -3);
    expect_version("Adreno", -3);
}

static void check_policy_tiers() {
    expect_policy(-2, false, false);
    expect_policy(-1, false, false);
    expect_policy(830, true, false);
    expect_policy(750, true, false);
    expect_policy(701, true, false);
    expect_policy(700, false, true);
    expect_policy(660, false, true);
    expect_policy(601, false, true);
    expect_policy(600, false, true);
    expect_policy(500, false, true);
    expect_policy(1, false, true);
}

int main() {
    check_vulkan_descriptions();
    check_opencl_descriptions();
    check_non_adreno_descriptions();
    check_policy_tiers();

    if (g_failures == 0) {
        std::printf("All Adreno version and policy cases passed.\n");
        return 0;
    }
    std::printf("%d Adreno case(s) failed.\n", g_failures);
    return 1;
}
