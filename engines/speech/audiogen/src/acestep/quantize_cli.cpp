// acestep-quantize: GGUF requantizer for the ACE-Step stage files (and the
// MiniMax LM GGUF, which shares the layout), ported from tools/quantize.cpp in
// acestep.cpp (MIT). Thin driver over quantize_gguf.h, where the planning and
// streaming-writer logic lives and is unit-tested.

#include <cstdio>
#include <string>

#include "quantize_gguf.h"

namespace {

using tts_cpp::acestep::QuantVariant;

constexpr const char * TAG = "acestep-quantize";

void print_usage(const char * prog) {
    fprintf(stderr, "Usage: %s <input.gguf> <output.gguf> <type>\n", prog);
    fprintf(stderr, "Types:");
    size_t               count    = 0;
    const QuantVariant * variants = tts_cpp::acestep::quant_variants(count);
    for (size_t i = 0; i < count; ++i) {
        fprintf(stderr, " %s", variants[i].name);
    }
    fprintf(stderr, "\n");
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 4) {
        print_usage(argv[0]);
        return 1;
    }

    const char *         inp_path = argv[1];
    const char *         out_path = argv[2];
    const QuantVariant * variant  = tts_cpp::acestep::find_quant_variant(argv[3]);

    if (!variant) {
        fprintf(stderr, "[%s] unknown type: %s\n", TAG, argv[3]);
        print_usage(argv[0]);
        return 1;
    }

    fprintf(stderr, "[%s] %s -> %s (%s)\n", TAG, inp_path, out_path, variant->name);

    tts_cpp::acestep::QuantizeStats stats;
    std::string                     error;
    if (!tts_cpp::acestep::quantize_gguf_file(inp_path, out_path, *variant, stats, error)) {
        fprintf(stderr, "[%s] %s\n", TAG, error.c_str());
        return 1;
    }

    fprintf(stderr, "[%s] quantized %d/%d tensors, promoted %d to F32\n", TAG, stats.n_quantized,
            stats.n_tensors, stats.n_promoted);
    fprintf(stderr, "[%s] %.1f GB -> %.1f GB (%.1fx)\n", TAG, (double) stats.bytes_in / 1e9,
            (double) stats.bytes_out / 1e9,
            stats.bytes_out > 0 ? (double) stats.bytes_in / (double) stats.bytes_out : 0.0);
    fprintf(stderr, "[%s] wrote %s\n", TAG, out_path);
    return 0;
}
