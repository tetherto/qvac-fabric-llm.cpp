// The unrolled decoder graph must reproduce the sequential per-step loop token
// for token: decode the same utterance with the sequential loop, with one
// greedy step per graph and with eight, and compare the token streams.
//   test-tdt-unroll-parity <tdt.gguf> <wav> [--n-gpu-layers N] [--backends-dir DIR]
//                          [--require-unrolled]
#include "parakeet_ctc.h"
#include "parakeet_tdt.h"
#include "mel_preprocess.h"
#include "backend_util.h"

#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

struct DecodeVariant {
    const char * name;
    int  unroll_steps;
    bool unrolled_used = false;
    int  graph_nodes   = 0;
    bool fused_backend = false; // CUDA or Metal: the unrolled path must exist there
    std::vector<int32_t> tokens;
};

int decode_with(const parakeet::ParakeetCtcModel & model,
                const parakeet::EncoderOutputs & enc,
                DecodeVariant & variant) {
    using namespace parakeet;
    TdtRuntimeWeights rt;
    if (int rc = tdt_prepare_runtime(model, rt); rc != 0) return 150 + rc;
    tdt_set_unroll_steps(rt, variant.unroll_steps);

    TdtDecodeOptions dopts;
    TdtDecodeResult  dres;
    if (int rc = tdt_greedy_decode(model, rt, enc.encoder_out.data(),
                                   enc.n_enc_frames, enc.d_model, dopts, dres); rc != 0) {
        return 160 + rc;
    }
    // Ground truth, not inference: the graph pointer says whether the unrolled
    // path actually ran for this decode.
    variant.unrolled_used = rt.g_unroll != nullptr;
    variant.graph_nodes   = rt.g_unroll ? ggml_graph_n_nodes(rt.g_unroll) : 0;
    variant.fused_backend = backend_is_cuda(rt.backend) || backend_is_metal(rt.backend);
    variant.tokens = std::move(dres.token_ids);
    return 0;
}

int compare_to_reference(const DecodeVariant & reference, const DecodeVariant & variant) {
    if (reference.tokens == variant.tokens) return 0;
    std::fprintf(stderr, "[tdt-unroll-parity] FAIL: %zu tokens (%s) vs %zu tokens (%s) differ\n",
                 reference.tokens.size(), reference.name, variant.tokens.size(), variant.name);
    return 1;
}

} // namespace

int main(int argc, char ** argv) {
    using namespace parakeet;
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <tdt.gguf> <wav> [--n-gpu-layers N] [--require-unrolled]\n", argv[0]);
        return 2;
    }
    int  n_gpu_layers     = 0;
    bool require_unrolled = false;
    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--require-unrolled") == 0) require_unrolled = true;
        if (i + 1 >= argc) continue;
        if (std::strcmp(argv[i], "--n-gpu-layers") == 0) n_gpu_layers = std::atoi(argv[i + 1]);
        if (std::strcmp(argv[i], "--backends-dir") == 0) set_backends_directory(argv[i + 1]);
    }

    ParakeetCtcModel model;
    if (int rc = load_from_gguf(argv[1], model, /*n_threads=*/0, n_gpu_layers, /*verbose=*/false); rc != 0) {
        std::fprintf(stderr, "[tdt-unroll-parity] load_from_gguf rc=%d\n", rc);
        return 1;
    }
    std::vector<float> samples;
    int sr = 0;
    if (int rc = load_wav_mono_f32(argv[2], samples, sr); rc != 0) {
        std::fprintf(stderr, "[tdt-unroll-parity] load_wav rc=%d\n", rc);
        return 1;
    }
    std::vector<float> mel;
    int n_frames = 0;
    if (int rc = compute_log_mel(samples.data(), (int) samples.size(), model.mel_cfg, mel, n_frames); rc != 0) {
        std::fprintf(stderr, "[tdt-unroll-parity] compute_log_mel rc=%d\n", rc);
        return 1;
    }
    EncoderOutputs enc;
    if (int rc = run_encoder(model, mel.data(), n_frames, model.mel_cfg.n_mels, enc, parakeet::EncoderRoute::ggml()); rc != 0) {
        std::fprintf(stderr, "[tdt-unroll-parity] run_encoder rc=%d\n", rc);
        return 1;
    }

    DecodeVariant sequential { "sequential loop", 0 };
    DecodeVariant unroll1    { "K=1",             1 };
    DecodeVariant unroll8    { "K=8",             8 };
    for (DecodeVariant * v : { &sequential, &unroll1, &unroll8 }) {
        if (int rc = decode_with(model, enc, *v); rc != 0) return rc;
    }

    if (sequential.unrolled_used) {
        std::fprintf(stderr, "[tdt-unroll-parity] FAIL: the sequential variant built an unrolled graph\n");
        return 1;
    }
    if (!unroll1.unrolled_used || !unroll8.unrolled_used) {
        if (require_unrolled && unroll8.fused_backend) {
            std::fprintf(stderr, "[tdt-unroll-parity] FAIL: no unrolled decode graph on this backend"
                                 " (n_gpu_layers=%d)\n", n_gpu_layers);
            return 1;
        }
        if (require_unrolled) {
            std::fprintf(stderr, "[tdt-unroll-parity] SKIP: this backend has no fused decode ops"
                                 " (n_gpu_layers=%d)\n", n_gpu_layers);
            return 3;
        }
        std::printf("[tdt-unroll-parity] PASS: backend has no graph decode path,"
                    " all three variants ran the sequential loop (%zu tokens)\n",
                    sequential.tokens.size());
        return 0;
    }

    if (int rc = compare_to_reference(sequential, unroll1); rc != 0) return rc;
    if (int rc = compare_to_reference(sequential, unroll8); rc != 0) return rc;

    std::printf("[tdt-unroll-parity] graph nodes: K=1 %d, K=8 %d, per extra step %d\n",
                unroll1.graph_nodes, unroll8.graph_nodes,
                (unroll8.graph_nodes - unroll1.graph_nodes)/7);

    std::printf("[tdt-unroll-parity] PASS: %zu tokens identical for the sequential loop, K=1 and K=8\n",
                sequential.tokens.size());
    return 0;
}
