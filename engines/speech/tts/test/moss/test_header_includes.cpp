#include "moss/sfx_sampler.h"

#include <cstdio>

using namespace tts_cpp::moss::detail;

bool gguf_metadata_header_reads_int_array();

namespace {

constexpr size_t NOISE_SAMPLES = 5;
constexpr uint32_t NOISE_SEED = 1;

bool sfx_sampler_header_fills_noise() {
    return gaussian_noise(NOISE_SAMPLES, NOISE_SEED).size() == NOISE_SAMPLES;
}

int report(bool ok, const char * check) {
    if (ok) return 0;
    std::fprintf(stderr, "FAIL: %s\n", check);
    return 1;
}

}

int main() {
    int failures = 0;
    failures += report(sfx_sampler_header_fills_noise(), "sfx_sampler.h compiles first and fills noise");
    failures += report(gguf_metadata_header_reads_int_array(), "gguf_metadata.h compiles first and reads arrays");
    if (failures > 0) return 1;
    std::printf("test-moss-header-includes: all checks passed\n");
    return 0;
}
