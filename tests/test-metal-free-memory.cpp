#include "../ggml/src/ggml-metal/ggml-metal-memory.h"

#include <cstdio>
#include <cstdlib>

static constexpr size_t MIB = 1024ull * 1024;
static constexpr size_t WORKING_SET = 10922 * MIB;
static constexpr size_t ALLOCATED_BELOW = 4096 * MIB;
static constexpr size_t ALLOCATED_ABOVE = 11008 * MIB;

static bool expect_free(const char * name, size_t allocated, size_t expected) {
    const size_t actual = ggml_metal_free_memory(WORKING_SET, allocated);
    const bool ok = actual == expected;
    printf("  %-48s %s\n", name, ok ? "OK" : "FAIL");
    if (!ok) {
        fprintf(stderr, "  %s: expected %zu, got %zu\n", name, expected, actual);
    }
    return ok;
}

int main() {
    bool ok = expect_free("nothing allocated leaves the working set", 0, WORKING_SET);
    ok = expect_free("allocation below the working set", ALLOCATED_BELOW, WORKING_SET - ALLOCATED_BELOW) && ok;
    ok = expect_free("allocation equal to the working set", WORKING_SET, 0) && ok;
    ok = expect_free("allocation past the working set clamps to zero", ALLOCATED_ABOVE, 0) && ok;
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
