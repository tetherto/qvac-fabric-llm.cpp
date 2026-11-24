// The fused LSTM cell op and the concatenated-input LSTM GEMV must both
// reproduce the decomposed per-gate decoder graph token for token: decode the
// same utterance with each runtime on the requested backend and compare tokens.
//   test-tdt-lstm-parity <tdt.gguf> <wav> [--n-gpu-layers N] [--backends-dir DIR]
#include "parakeet_ctc.h"
#include "parakeet_tdt.h"
#include "mel_preprocess.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct DecodeVariant {
    const char * name;
    bool allow_fused;
    bool allow_concat;
    bool fused_used  = false;
    bool concat_used = false;
    std::vector<int32_t> tokens;
};

int decode_with(const parakeet::ParakeetCtcModel & model,
                const parakeet::EncoderOutputs & enc,
                DecodeVariant & variant) {
    using namespace parakeet;
    TdtRuntimeWeights rt;
    if (int rc = tdt_prepare_runtime(model, rt, variant.allow_fused, variant.allow_concat); rc != 0) {
        return 150 + rc;
    }
    variant.fused_used  = rt.fused_lstm_cell;
    variant.concat_used = rt.concat_lstm_input;
    TdtDecodeOptions dopts;
    TdtDecodeResult  dres;
    if (int rc = tdt_greedy_decode(model, rt, enc.encoder_out.data(),
                                   enc.n_enc_frames, enc.d_model, dopts, dres); rc != 0) return 160 + rc;
    variant.tokens = std::move(dres.token_ids);
    return 0;
}

int compare_to_reference(const DecodeVariant & reference, const DecodeVariant & variant) {
    if (reference.tokens == variant.tokens) return 0;
    std::fprintf(stderr, "[tdt-lstm-parity] FAIL: %zu tokens (%s) vs %zu tokens (%s) differ\n",
                 reference.tokens.size(), reference.name, variant.tokens.size(), variant.name);
    return 1;
}

} // namespace

int main(int argc, char ** argv) {
    using namespace parakeet;
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <tdt.gguf> <wav> [--n-gpu-layers N]\n", argv[0]);
        return 2;
    }
    int n_gpu_layers = 0;
    for (int i = 3; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--n-gpu-layers") == 0) n_gpu_layers = std::atoi(argv[i + 1]);
        if (std::strcmp(argv[i], "--backends-dir") == 0) set_backends_directory(argv[i + 1]);
    }

    ParakeetCtcModel model;
    if (int rc = load_from_gguf(argv[1], model, /*n_threads=*/0, n_gpu_layers, /*verbose=*/false); rc != 0) {
        std::fprintf(stderr, "[tdt-lstm-parity] load_from_gguf rc=%d\n", rc);
        return 1;
    }
    std::vector<float> samples;
    int sr = 0;
    if (int rc = load_wav_mono_f32(argv[2], samples, sr); rc != 0) {
        std::fprintf(stderr, "[tdt-lstm-parity] load_wav rc=%d\n", rc);
        return 1;
    }
    std::vector<float> mel;
    int n_frames = 0;
    if (int rc = compute_log_mel(samples.data(), (int) samples.size(), model.mel_cfg, mel, n_frames); rc != 0) {
        std::fprintf(stderr, "[tdt-lstm-parity] compute_log_mel rc=%d\n", rc);
        return 1;
    }
    EncoderOutputs enc;
    if (int rc = run_encoder(model, mel.data(), n_frames, model.mel_cfg.n_mels, enc, parakeet::EncoderRoute::ggml()); rc != 0) {
        std::fprintf(stderr, "[tdt-lstm-parity] run_encoder rc=%d\n", rc);
        return 1;
    }

    DecodeVariant plain   { "decomposed",     false, false };
    DecodeVariant fused   { "fused cell",     true,  false };
    DecodeVariant concat  { "concat input",   true,  true  };
    for (DecodeVariant * v : { &plain, &fused, &concat }) {
        if (int rc = decode_with(model, enc, *v); rc != 0) return rc;
    }
    if (!fused.fused_used) {
        std::fprintf(stderr, "[tdt-lstm-parity] SKIP: backend has no fused LSTM cell (n_gpu_layers=%d)\n", n_gpu_layers);
        return 3;
    }
    if (int rc = compare_to_reference(plain, fused); rc != 0) return rc;
    if (!concat.concat_used) {
        std::fprintf(stderr, "[tdt-lstm-parity] SKIP: backend cannot stack the LSTM input weights\n");
        return 3;
    }
    if (int rc = compare_to_reference(plain, concat); rc != 0) return rc;

    std::printf("[tdt-lstm-parity] PASS: %zu tokens identical for the decomposed, fused and concatenated LSTM graphs\n",
                plain.tokens.size());
    return 0;
}
