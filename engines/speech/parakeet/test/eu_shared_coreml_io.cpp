#include "coreml/parakeet-encoder.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
constexpr int kChannels = 8;
constexpr int kStride = 8;
constexpr int kCapacity = 2001;
using Context = std::unique_ptr<parakeet_coreml_context, decltype(&parakeet_coreml_free)>;

void require(bool value) {
    if (!value) throw std::runtime_error("shared encoder contract failed");
}

void check_route(const char * path, const char * route, int offset, bool cpu, bool reject) {
    Context ctx(parakeet_coreml_init_eu_function(path, route, cpu), parakeet_coreml_free);
    require(ctx != nullptr);
    const int layers = std::strcmp(route, "gate") == 0 ? 1 : 2;
    if (reject) {
        require(!parakeet_coreml_matches_eu_route(ctx.get(), route, layers, false));
        return;
    }
    require(parakeet_coreml_matches_eu_route(ctx.get(), route, layers, false));
    require(!parakeet_coreml_matches_eu_route(ctx.get(), route, layers + 1, false));
    require(!parakeet_coreml_matches_eu_route(ctx.get(), route, layers, true));
    require(!parakeet_coreml_matches_eu_route(ctx.get(), "unknown", layers, false));
    require(parakeet_coreml_fixed_mel_frames(ctx.get(), kChannels) == kCapacity);
    for (int frames : {1, 17, kCapacity, 9, 17}) {
        const int rows = (frames + kStride - 1) / kStride;
        std::vector<float> mel(frames * kChannels, 0.5f);
        std::vector<float> out(rows * kChannels);
        require(parakeet_coreml_encode_valid(ctx.get(), frames, frames, kChannels,
                                             mel.data(), rows, kChannels, out.data()) == 0);
        for (float value : out) require(std::abs(value - (offset + 0.5f)) < 0.001f);
    }
    std::vector<float> oversized((kCapacity + 1) * kChannels);
    require(parakeet_coreml_encode_valid(ctx.get(), kCapacity + 1, kCapacity + 1, kChannels,
                                         oversized.data(), 1, kChannels, oversized.data()) != 0);
}
}

int main(int argc, char ** argv) {
    try {
        require(argc == 3);
        require(parakeet_coreml_init_eu_function(argv[1], "unknown", true) == nullptr);
        check_route(argv[1], "en", 1, std::strcmp(argv[2], "ane") != 0, std::strcmp(argv[2], "reject") == 0);
        check_route(argv[1], "other", 2, std::strcmp(argv[2], "ane") != 0, std::strcmp(argv[2], "reject") == 0);
        check_route(argv[1], "gate", 3, std::strcmp(argv[2], "ane") != 0, std::strcmp(argv[2], "reject") == 0);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
