#include "cosyvoice_frontend.h"

#include "campplus.h"
#include "gguf_stream.h"
#include "s3tokenizer.h"
#include "voice_features.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace {

bool all_samples_finite(const std::vector<float> & wav) {
    for (const float s : wav) {
        if (!std::isfinite(s)) return false;
    }
    return true;
}

// Per-utterance column-mean subtraction over the (T, 80) fbank, matching the
// upstream extract_feature() normalisation CAM++ expects.
void mean_center_fbank(std::vector<float> & fbank, int T) {
    std::vector<float> col_mean(80, 0.0f);
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < 80; ++c) col_mean[c] += fbank[(size_t)t * 80 + c];
    for (int c = 0; c < 80; ++c) col_mean[c] /= (float)T;
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < 80; ++c) fbank[(size_t)t * 80 + c] -= col_mean[c];
}

// CAM++ runs on the scalar CPU reference path regardless of the engine
// backend: the ggml-graph variant produces an antipodal embedding on real
// voice input, and the bake runs once per voice, so correctness wins.
bool campplus_embed_forced_scalar(const std::vector<float> & fbank, int T,
                                  const campplus_weights & w,
                                  std::vector<float> & out_emb) {
    return campplus_embed(fbank, T, w, /*backend=*/nullptr, out_emb);
}

// Pull one host-resident F32 tensor out of a GGUF via the streamed reader
// (no full-file staging; the caller only needs the small filterbanks).
bool read_f32_tensor(const std::string & gguf_path, const char * name,
                     std::vector<float> & out, std::string & err) {
    ggml_context * tmp = nullptr;
    gguf_init_params gp = { /*.no_alloc=*/ true, /*.ctx=*/ &tmp };
    gguf_context * g = gguf_init_from_file(gguf_path.c_str(), gp);
    if (!g) { err = "cannot open " + gguf_path; return false; }
    ggml_tensor * t = tmp ? ggml_get_tensor(tmp, name) : nullptr;
    if (!t) {
        err = std::string(name) + " missing from " + gguf_path +
              " (re-run the converter)";
        gguf_free(g); if (tmp) ggml_free(tmp);
        return false;
    }
    if (t->type != GGML_TYPE_F32) {
        // The buffer below is sized in floats; a wider type would make the
        // byte read overflow it, so reject anything the converters would
        // never write.
        err = std::string(name) + " in " + gguf_path + " has type " +
              ggml_type_name(t->type) + " (want f32; re-run the converter)";
        gguf_free(g); if (tmp) ggml_free(tmp);
        return false;
    }
    out.resize((size_t)ggml_nelements(t));
    ::tts_cpp::detail::gguf_stream_reader reader(g, gguf_path);
    const bool ok = reader.ok() && reader.to_host(name, out.data(), out.size() * sizeof(float));
    gguf_free(g); if (tmp) ggml_free(tmp);
    if (!ok) err = std::string("failed to read ") + name + " from " + gguf_path;
    return ok;
}

} // namespace

bool cosyvoice_frontend_run(const std::string & reference_wav,
                            const std::string & s3tok_gguf,
                            const std::string & campplus_gguf,
                            ggml_backend_t backend,
                            int n_threads,
                            cosyvoice_prompt & out,
                            std::string & err) {
    std::vector<float> wav;
    int sr = 0;
    if (!wav_load(reference_wav, wav, sr) || wav.empty() || sr <= 0) {
        err = "cannot load reference audio " + reference_wav;
        return false;
    }
    if (!all_samples_finite(wav)) {
        // Float WAVs can legally carry NaN/inf payloads; they would propagate
        // through every downstream tensor and bake unusable conditioning.
        err = "reference audio contains non-finite samples: " + reference_wav;
        return false;
    }
    const double dur = (double)wav.size() / (double)sr;
    if (dur < 0.5) {
        err = "reference audio is too short (" + std::to_string(dur) +
              " s); need at least 0.5 s of speech";
        return false;
    }
    if (dur > 30.0) {
        err = "reference audio is too long (" + std::to_string(dur) +
              " s); the speech tokenizer supports at most 30 s";
        return false;
    }
    if (dur < 3.0) {
        fprintf(stderr, "cosyvoice: reference audio is only %.1f s; 5-15 s of "
                        "clean speech clones more reliably\n", dur);
    }

    // ---- speech tokens (16 kHz stream) ----
    std::vector<float> wav_16k = (sr == 16000) ? wav : resample_sinc(wav, sr, 16000);

    s3tokv2_weights tok_w;
    if (!s3tokv2_load(s3tok_gguf, tok_w)) {
        err = "cannot load speech tokenizer from " + s3tok_gguf;
        return false;
    }
    if (tok_w.version != 3) {
        // A v2 file (e.g. a chatterbox-s3gen GGUF) loads mechanically but
        // produces tokens from a different token space than the CosyVoice3
        // LM was trained on — silently accepting it would clone the wrong
        // content, so reject up front.
        err = s3tok_gguf + " is not a speech_tokenizer_v3 GGUF (tokenizer_version=" +
              std::to_string(tok_w.version) + ")";
        return false;
    }
    std::vector<int32_t> tokens;
    if (!s3tokv2_tokenize(wav_16k, tok_w, /*max_tokens=*/-1, tokens, n_threads, backend)) {
        err = "speech tokenizer failed on " + reference_wav;
        return false;
    }
    if (tokens.empty()) {
        err = "speech tokenizer produced no tokens for " + reference_wav;
        return false;
    }

    // ---- speaker embedding (same 16 kHz stream) ----
    campplus_weights cam_w;
    if (!campplus_load(campplus_gguf, cam_w)) {
        err = "cannot load CAM++ from " + campplus_gguf;
        return false;
    }
    std::vector<float> kaldi_fb;
    if (!read_f32_tensor(campplus_gguf, "campplus/mel_fb_kaldi_80", kaldi_fb, err)) {
        return false;
    }
    std::vector<float> fbank = fbank_kaldi_80(wav_16k, kaldi_fb);
    if (fbank.empty()) {
        err = "fbank extraction failed on " + reference_wav;
        return false;
    }
    const int T_fb = (int)(fbank.size() / 80);
    mean_center_fbank(fbank, T_fb);
    if (!campplus_embed_forced_scalar(fbank, T_fb, cam_w, out.embedding)) {
        err = "CAM++ embedding failed on " + reference_wav;
        return false;
    }

    // ---- prompt mel (native rate → 24 kHz, NOT via the 16 kHz stream) ----
    std::vector<float> mel_fb_24k;
    if (!read_f32_tensor(s3tok_gguf, "cosyvoice3/mel_fb_24k_80", mel_fb_24k, err)) {
        return false;
    }
    std::vector<float> wav_24k = (sr == 24000) ? wav : resample_sinc(wav, sr, 24000);
    // The matcha mel this must match computes sqrt(|spec|^2 + 1e-9).
    std::vector<float> mel = mel_extract_24k_80(wav_24k, mel_fb_24k, /*mag_eps=*/1e-9f);
    if (mel.empty()) {
        err = "prompt mel extraction failed on " + reference_wav;
        return false;
    }
    const int T_mel = (int)(mel.size() / 80);

    // ---- 2:1 token/mel alignment (the flow's token_mel_ratio) ----
    const int token_len = std::min<int>(T_mel / 2, (int)tokens.size());
    if (token_len <= 0) {
        err = "reference audio yields no aligned prompt frames";
        return false;
    }
    tokens.resize((size_t)token_len);
    mel.resize((size_t)token_len * 2 * 80);

    out.prompt_stok = std::move(tokens);
    out.prompt_feat = std::move(mel);
    out.mel_len1    = token_len * 2;
    return true;
}
