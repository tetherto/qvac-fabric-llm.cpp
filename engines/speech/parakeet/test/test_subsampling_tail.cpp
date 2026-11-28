// Subsampling of a mel whose tail frames are zero: the padded run must keep the padded
// frame count, match the truncated run bit for bit on the valid frames, and hold the
// projection bias on every padded frame.
//
// Usage:
//   test-subsampling-tail --model <parakeet gguf> [--n-gpu-layers N]
//
// Exit 0 on success; non-zero on failure or invalid arguments.

#include "parakeet_ctc.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string model;
    int n_gpu_layers = 0;
};

bool parse_args(int argc, char ** argv, Args & args) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) {
            args.model = argv[++i];
        } else if (a == "--n-gpu-layers" && i + 1 < argc) {
            args.n_gpu_layers = std::atoi(argv[++i]);
        } else {
            return false;
        }
    }
    return !args.model.empty();
}

std::vector<float> random_mel(int n_frames, int n_mels, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    std::vector<float> mel((size_t) n_frames * n_mels);
    for (float & v : mel) v = dist(rng);
    return mel;
}

void zero_tail_frames(std::vector<float> & mel, int n_mels, int n_valid) {
    std::fill(mel.begin() + (size_t) n_valid * n_mels, mel.end(), 0.0f);
}

bool read_bias(const ggml_tensor * t, std::vector<float> & out) {
    out.resize(ggml_nelements(t));
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, out.size() * sizeof(float));
        return true;
    }
    if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> half(out.size());
        ggml_backend_tensor_get(t, half.data(), 0, half.size() * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < out.size(); ++i) out[i] = ggml_fp16_to_fp32(half[i]);
        return true;
    }
    return false;
}

bool frames_hold_bias(const std::vector<float> & feats, int t_begin, int t_end,
                      const std::vector<float> & bias) {
    const size_t d = bias.size();
    for (int t = t_begin; t < t_end; ++t) {
        const float * row = feats.data() + (size_t) t * d;
        for (size_t i = 0; i < d; ++i) {
            if (row[i] != bias[i]) return false;
        }
    }
    return true;
}

int check_tail(parakeet::ParakeetCtcModel & model, int n_frames, int n_valid, unsigned seed) {
    const int n_mels = model.mel_cfg.n_mels;
    std::vector<float> mel = random_mel(n_frames, n_mels, seed);
    zero_tail_frames(mel, n_mels, n_valid);

    std::vector<float> padded;
    std::vector<float> prefix;
    int n_padded = 0;
    int n_prefix = 0;
    if (int rc = parakeet::run_subsampling(model, mel.data(), n_frames, n_mels, padded, n_padded); rc != 0) {
        std::fprintf(stderr, "  run_subsampling(padded) rc=%d\n", rc);
        return 1;
    }
    if (int rc = parakeet::run_subsampling(model, mel.data(), n_valid, n_mels, prefix, n_prefix); rc != 0) {
        std::fprintf(stderr, "  run_subsampling(prefix) rc=%d\n", rc);
        return 1;
    }

    const size_t d = (size_t) model.encoder_cfg.d_model;
    if (n_padded < n_prefix || padded.size() != (size_t) n_padded * d || prefix.size() != (size_t) n_prefix * d) {
        std::fprintf(stderr, "  frame counts: padded %d, prefix %d\n", n_padded, n_prefix);
        return 1;
    }
    if (std::memcmp(padded.data(), prefix.data(), prefix.size() * sizeof(float)) != 0) {
        std::fprintf(stderr, "  valid frames differ between the padded and the truncated mel\n");
        return 1;
    }
    std::vector<float> bias;
    if (!read_bias(model.subsampling.out_b, bias) || bias.size() != d) {
        std::fprintf(stderr, "  projection bias unreadable\n");
        return 1;
    }
    if (!frames_hold_bias(padded, n_prefix, n_padded, bias)) {
        std::fprintf(stderr, "  a padded frame does not hold the projection bias\n");
        return 1;
    }
    std::printf("  mel %d frames, %d valid -> %d subsampled (%d valid): prefix identical, tail = bias\n",
                n_frames, n_valid, n_padded, n_prefix);
    return 0;
}

} // namespace

int main(int argc, char ** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) {
        std::fprintf(stderr, "usage: %s --model <parakeet gguf> [--n-gpu-layers N]\n", argv[0]);
        return 2;
    }
    parakeet::ParakeetCtcModel model;
    if (int rc = parakeet::load_from_gguf(args.model, model, /*n_threads=*/0, args.n_gpu_layers, /*verbose=*/false); rc != 0) {
        std::fprintf(stderr, "[test-subsampling-tail] load_from_gguf rc=%d\n", rc);
        return 1;
    }
    struct Case { int n_frames; int n_valid; };
    const Case cases[] = { {3001, 3000}, {1101, 1093}, {512, 512} };
    for (const Case & c : cases) {
        if (check_tail(model, c.n_frames, c.n_valid, 7u) != 0) {
            std::fprintf(stderr, "[test-subsampling-tail] FAIL at %d frames, %d valid\n", c.n_frames, c.n_valid);
            return 1;
        }
    }
    std::printf("[test-subsampling-tail] PASS\n");
    return 0;
}
