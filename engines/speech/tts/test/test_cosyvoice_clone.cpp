// Engine-level zero-shot / cross-lingual cloning harness.
//
// Usage:
//   ./build/test-cosyvoice-clone --model-dir DIR --wav zero_shot_prompt.wav
//       --tokens-npy cosyvoice3-ref/gen_tokens.npy --s3tok-gguf S3TOK.gguf
//       [--prompt-text TRANSCRIPT] [--min-mel-cosine 0.98] [--min-env-corr 0.90]
//
// Three legs, all with the LM trajectory pinned via force_speech_tokens
// (sampling is chaotic; pinning makes the flow + vocoder stages comparable):
//
//  A. Cloned-vs-baked equivalence.  voice.gguf was baked from the same
//     reference wav this test clones from, so an engine cloning that wav with
//     its transcript must produce equivalent audio to the baked-voice engine:
//     same length, mel-spectral cosine above the bound (measured 0.990;
//     upstream-vs-baked full-pipeline parity only reaches 0.85 on the same
//     material, so 0.98 is a tight gate), and a CAM++ cosine between the two
//     outputs that is at least the baked output's own SPLIT-HALF self-cosine
//     minus a margin.  The self-calibrating form matters: raw (non-PLDA,
//     non-normalised) CAM++ cosines are not absolute similarity scores on
//     synthetic out-of-domain audio — the same clip's two halves measured
//     only 0.36 — while cloned-vs-baked measured 0.39, i.e. as similar as
//     the clip is to itself.  Stricter gates were measured and rejected:
//     raw-waveform cosine ~0.001 (HiFT integrates f0 into phase, so the
//     small front-end delta — a few FSQ token flips + 5e-2 embedding shift
//     from resampler differences — accumulates phase drift) and 10 ms
//     energy-envelope correlation ~0.33 (the DiT attends over the whole
//     token sequence, so flipped prompt tokens micro-shift generated timing).
//
//  B. Cross-lingual branch isolation.  The same cloned engine without a
//     transcript switches to cross-lingual mode, which by construction only
//     changes the LM prompt.  With the trajectory pinned, its output must be
//     bit-identical to leg A's cloned output — proving the branch never
//     touches the flow/vocoder conditioning.
//
//  C. Fail-closed.  reference_audio with an unloadable tokenizer GGUF, and a
//     0.3 s reference, must both throw at construction — never silently fall
//     back to the baked voice.
//
//  D. Cross-lingual LM path, unforced.  A real greedy decode of a very short
//     text in cross-lingual mode must respect the upstream minimum-length
//     basis: upstream carries the <|endofprompt|> template inside tts_text
//     there (prompt_text is deleted), so the template's ~7 BPE tokens count
//     toward min_len = 2 x basis.  Without that, EOS can legally fire after
//     2 x n_text tokens — 2 tokens for a one-token text — truncating short
//     utterances.  The decode loop hard-masks EOS below min_len, so the
//     token-count floor asserts the basis directly.
//
//  E. Prefill composition, unforced.  Legs A-C pin the LM trajectory, so a
//     regression that silently drops the transcript or the reference speech
//     tokens from the LM prefill would not fail them.  Two checks: the
//     zero-shot and cross-lingual greedy trajectories for the same text must
//     differ (the reference prompt block demonstrably conditions the LM),
//     and the prefill composition reported in StageTimings must track the
//     mode — prompt speech tokens present in zero-shot and absent in
//     cross-lingual, text-id count growing with the transcript.  Trajectory
//     INEQUALITY across different transcripts is deliberately not asserted:
//     greedy argmax after a strong same-reference speech prompt is robust to
//     distant-context logit shifts, and empirically two transcripts can
//     yield identical argmax sequences while both being present in the
//     prefill.
//
// --prompt-text defaults to the zero_shot_prompt.wav transcript baked into
// voice.gguf (the canonical fixture pair).

#include "tts-cpp/cosyvoice/engine.h"
#include "campplus.h"
#include "npy.h"
#include "voice_features.h"

#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

bool parse_bounded_arg(const char * s, double lo, double hi, double & out) {
    char * end = nullptr;
    const double v = std::strtod(s, &end);
    if (end == s || *end != '\0' || !(v >= lo) || !(v <= hi)) return false;
    out = v;
    return true;
}

bool write_wav_f32(const std::string & path, int sr, double seconds) {
    const uint32_t n = (uint32_t)(sr * seconds);
    const uint32_t data_bytes = n * 4;
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) return false;
    auto u32 = [&](uint32_t v) { fwrite(&v, 4, 1, f); };
    auto u16 = [&](uint16_t v) { fwrite(&v, 2, 1, f); };
    fwrite("RIFF", 1, 4, f); u32(36 + data_bytes); fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); u32(16); u16(3); u16(1);
    u32((uint32_t)sr); u32((uint32_t)sr * 4); u16(4); u16(32);
    fwrite("data", 1, 4, f); u32(data_bytes);
    for (uint32_t i = 0; i < n; ++i) {
        const float v = 0.1f * std::sin(2.0 * 3.14159265358979 * 220.0 * i / sr);
        fwrite(&v, 4, 1, f);
    }
    fclose(f);
    return true;
}

bool all_finite(const std::vector<float> & v) {
    for (const float x : v) {
        if (!std::isfinite(x)) return false;
    }
    return true;
}

double vec_cosine(const std::vector<float> & a, const std::vector<float> & b) {
    const size_t n = std::min(a.size(), b.size());
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; ++i) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
}

bool load_gguf_f32(const std::string & gguf_path, const char * name,
                   std::vector<float> & out) {
    ggml_context * tmp = nullptr;
    gguf_init_params gp = { /*.no_alloc=*/ false, /*.ctx=*/ &tmp };
    gguf_context * g = gguf_init_from_file(gguf_path.c_str(), gp);
    if (!g) return false;
    ggml_tensor * t = tmp ? ggml_get_tensor(tmp, name) : nullptr;
    bool ok = t != nullptr;
    if (ok) {
        out.resize((size_t)ggml_nelements(t));
        std::memcpy(out.data(), ggml_get_data(t), ggml_nbytes(t));
    }
    gguf_free(g); if (tmp) ggml_free(tmp);
    return ok;
}

// Speaker embedding of a 24 kHz synthesis output: resample to 16 kHz, Kaldi
// fbank, per-utterance mean subtraction, CAM++ (scalar path — same chain as
// the cloning front-end applies to reference audio).
bool output_speaker_embedding(const std::vector<float> & pcm_24k,
                              const campplus_weights & cam_w,
                              const std::vector<float> & kaldi_fb,
                              std::vector<float> & out_emb) {
    const std::vector<float> wav_16k = resample_sinc(pcm_24k, 24000, 16000);
    std::vector<float> fbank = fbank_kaldi_80(wav_16k, kaldi_fb);
    if (fbank.empty()) return false;
    const int T = (int)(fbank.size() / 80);
    std::vector<float> mean(80, 0.0f);
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < 80; ++c) mean[c] += fbank[(size_t)t * 80 + c];
    for (int c = 0; c < 80; ++c) mean[c] /= (float)T;
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < 80; ++c) fbank[(size_t)t * 80 + c] -= mean[c];
    return campplus_embed(fbank, T, cam_w, /*backend=*/nullptr, out_emb);
}

} // namespace

int main(int argc, char ** argv) {
    using namespace tts_cpp::cosyvoice;

    std::string model_dir, wav, tokens_npy, s3tok_gguf, campplus_gguf;
    // Transcript of CosyVoice/asset/zero_shot_prompt.wav, as baked into
    // voice.gguf's voice.prompt_text metadata.
    std::string prompt_text = "\xe5\xb8\x8c\xe6\x9c\x9b\xe4\xbd\xa0\xe4\xbb\xa5\xe5\x90\x8e\xe8\x83\xbd"
                              "\xe5\xa4\x9f\xe5\x81\x9a\xe7\x9a\x84\xe6\xaf\x94\xe6\x88\x91\xe8\xbf\x98"
                              "\xe5\xa5\xbd\xe5\x91\xa6\xe3\x80\x82";
    double min_mel_cosine = 0.98;
    double spk_margin = 0.10;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char * flag) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", flag); return nullptr; }
            return argv[++i];
        };
        if      (a == "--model-dir")  { const char * v = next("--model-dir");  if (!v) return 2; model_dir = v; }
        else if (a == "--wav")        { const char * v = next("--wav");        if (!v) return 2; wav = v; }
        else if (a == "--tokens-npy") { const char * v = next("--tokens-npy"); if (!v) return 2; tokens_npy = v; }
        else if (a == "--prompt-text"){ const char * v = next("--prompt-text");if (!v) return 2; prompt_text = v; }
        else if (a == "--s3tok-gguf") { const char * v = next("--s3tok-gguf"); if (!v) return 2; s3tok_gguf = v; }
        else if (a == "--campplus-gguf") { const char * v = next("--campplus-gguf"); if (!v) return 2; campplus_gguf = v; }
        else if (a == "--min-mel-cosine") {
            const char * v = next("--min-mel-cosine");
            if (!v || !parse_bounded_arg(v, 0.0, 1.0, min_mel_cosine)) {
                fprintf(stderr, "--min-mel-cosine takes a value in [0, 1]\n"); return 2;
            }
        }
        else if (a == "--spk-margin") {
            const char * v = next("--spk-margin");
            if (!v || !parse_bounded_arg(v, 0.0, 1.0, spk_margin)) {
                fprintf(stderr, "--spk-margin takes a value in [0, 1]\n"); return 2;
            }
        }
        else {
            fprintf(stderr, "usage: %s --model-dir DIR --wav REF.wav --tokens-npy TOKENS.npy\n"
                            "          --s3tok-gguf S3TOK.gguf --campplus-gguf CAMPPLUS.gguf\n"
                            "          [--prompt-text TRANSCRIPT]\n"
                            "          [--min-mel-cosine 0.98] [--spk-margin 0.10]\n", argv[0]);
            return 2;
        }
    }
    if (model_dir.empty() || wav.empty() || tokens_npy.empty() ||
        s3tok_gguf.empty() || campplus_gguf.empty()) {
        fprintf(stderr, "missing --model-dir / --wav / --tokens-npy / --s3tok-gguf / --campplus-gguf\n");
        return 2;
    }
    std::vector<float> mel_fb, kaldi_fb;
    if (!load_gguf_f32(s3tok_gguf, "cosyvoice3/mel_fb_24k_80", mel_fb)) {
        fprintf(stderr, "cannot read cosyvoice3/mel_fb_24k_80 from %s\n", s3tok_gguf.c_str());
        return 2;
    }
    if (!load_gguf_f32(campplus_gguf, "campplus/mel_fb_kaldi_80", kaldi_fb)) {
        fprintf(stderr, "cannot read campplus/mel_fb_kaldi_80 from %s\n", campplus_gguf.c_str());
        return 2;
    }
    campplus_weights cam_w;
    if (!campplus_load(campplus_gguf, cam_w)) {
        fprintf(stderr, "cannot load CAM++ from %s\n", campplus_gguf.c_str());
        return 2;
    }

    npy_array tok_a = npy_load(tokens_npy);
    std::vector<int> forced(tok_a.n_elements());
    { const int32_t * p = npy_as_i32(tok_a); for (size_t i = 0; i < forced.size(); ++i) forced[i] = p[i]; }
    fprintf(stderr, "pinned LM trajectory: %zu tokens\n", forced.size());

    const bool gpu = std::getenv("COSYVOICE_TEST_GPU") != nullptr;
    auto base_opts = [&]() {
        EngineOptions o;
        o.model_dir = model_dir;
        o.n_gpu_layers = gpu ? 99 : 0;
        o.force_speech_tokens = forced;
        o.seed = 1986;
        return o;
    };
    const std::string text = "The quick brown fox jumps over the lazy dog.";

    fprintf(stderr, "[A] baked-voice engine\n");
    std::vector<float> pcm_baked;
    {
        Engine eng(base_opts());
        pcm_baked = eng.synthesize(text).pcm;
    }
    fprintf(stderr, "[A] cloned engine (zero-shot: transcript given)\n");
    std::vector<float> pcm_cloned;
    {
        EngineOptions o = base_opts();
        o.reference_audio = wav;
        o.prompt_text = prompt_text;
        Engine eng(o);
        pcm_cloned = eng.synthesize(text).pcm;
    }
    if (pcm_baked.empty() || pcm_cloned.empty()) {
        fprintf(stderr, "FAIL: empty synthesis output\n");
        return 1;
    }
    // NaN comparisons are false, so a NaN-filled output would sail through
    // every threshold below; reject it up front.
    if (!all_finite(pcm_baked) || !all_finite(pcm_cloned)) {
        fprintf(stderr, "FAIL: non-finite samples in synthesis output\n");
        return 1;
    }
    if (pcm_baked.size() != pcm_cloned.size()) {
        fprintf(stderr, "FAIL: cloned length %zu != baked %zu (prompt-token count "
                        "must match for the same reference clip)\n",
                pcm_cloned.size(), pcm_baked.size());
        return 1;
    }
    const std::vector<float> mel_baked  = mel_extract_24k_80(pcm_baked, mel_fb);
    const std::vector<float> mel_cloned = mel_extract_24k_80(pcm_cloned, mel_fb);
    const double mel_cos = vec_cosine(mel_baked, mel_cloned);
    std::vector<float> spk_baked, spk_cloned;
    if (!output_speaker_embedding(pcm_baked, cam_w, kaldi_fb, spk_baked) ||
        !output_speaker_embedding(pcm_cloned, cam_w, kaldi_fb, spk_cloned)) {
        fprintf(stderr, "FAIL: CAM++ on synthesis output failed\n");
        return 1;
    }
    const double spk_cos = vec_cosine(spk_baked, spk_cloned);

    // In-run calibration baseline: how similar the baked output is to ITSELF
    // (first half vs second half) under the same embedding chain.
    double self_cos = 1.0;
    {
        std::vector<float> half1(pcm_baked.begin(), pcm_baked.begin() + pcm_baked.size() / 2);
        std::vector<float> half2(pcm_baked.begin() + pcm_baked.size() / 2, pcm_baked.end());
        std::vector<float> e1, e2;
        if (!output_speaker_embedding(half1, cam_w, kaldi_fb, e1) ||
            !output_speaker_embedding(half2, cam_w, kaldi_fb, e2)) {
            fprintf(stderr, "FAIL: CAM++ split-half baseline failed\n");
            return 1;
        }
        self_cos = vec_cosine(e1, e2);
    }
    if (const char * pfx = std::getenv("COSYVOICE_CLONE_DUMP_PREFIX")) {
        auto dump = [&](const std::string & path, const std::vector<float> & pcm) {
            FILE * f = fopen(path.c_str(), "wb");
            if (f) { fwrite(pcm.data(), 4, pcm.size(), f); fclose(f); }
        };
        dump(std::string(pfx) + "-baked.f32", pcm_baked);
        dump(std::string(pfx) + "-cloned.f32", pcm_cloned);
    }
    fprintf(stderr, "[A] cloned-vs-baked: mel cosine = %.6f (bound %.4f), "
                    "output speaker cosine = %.6f (baseline: baked split-half %.6f - margin %.2f), "
                    "%zu samples\n",
            mel_cos, min_mel_cosine, spk_cos, self_cos, spk_margin, pcm_baked.size());
    if (!std::isfinite(mel_cos) || !std::isfinite(spk_cos) || !std::isfinite(self_cos)) {
        fprintf(stderr, "FAIL: non-finite comparison metric\n");
        return 1;
    }
    if (mel_cos < min_mel_cosine || spk_cos < self_cos - spk_margin) {
        fprintf(stderr, "FAIL: cloned output diverges from the baked voice it was "
                        "enrolled from\n");
        return 1;
    }

    fprintf(stderr, "[B] cloned engine (cross-lingual: no transcript)\n");
    std::vector<float> pcm_xl;
    {
        EngineOptions o = base_opts();
        o.reference_audio = wav;
        Engine eng(o);
        pcm_xl = eng.synthesize(text).pcm;
    }
    if (pcm_xl.size() != pcm_cloned.size()) {
        fprintf(stderr, "FAIL: cross-lingual length %zu != zero-shot %zu\n",
                pcm_xl.size(), pcm_cloned.size());
        return 1;
    }
    if (!all_finite(pcm_xl)) {
        fprintf(stderr, "FAIL: non-finite samples in cross-lingual output\n");
        return 1;
    }
    double xl_ma = 0.0;
    for (size_t i = 0; i < pcm_xl.size(); ++i) {
        const double d = std::fabs((double)pcm_xl[i] - (double)pcm_cloned[i]);
        if (d > xl_ma) xl_ma = d;
    }
    fprintf(stderr, "[B] cross-lingual vs zero-shot max_abs = %.3e (pinned trajectory; "
                    "the mode may only change the LM prompt)\n", xl_ma);
    if (xl_ma != 0.0) {
        fprintf(stderr, "FAIL: cross-lingual mode changed the flow/vocoder output\n");
        return 1;
    }

    fprintf(stderr, "[C] fail-closed\n");
    {
        EngineOptions o = base_opts();
        o.reference_audio = wav;
        o.s3tok_gguf_path = model_dir + "/no-such-s3tok.gguf";
        bool threw = false;
        try { Engine eng(o); } catch (const std::exception & e) {
            threw = true;
            fprintf(stderr, "      unloadable s3tok gguf -> threw: %s\n", e.what());
        }
        if (!threw) {
            fprintf(stderr, "FAIL: unloadable tokenizer GGUF did not throw\n");
            return 1;
        }
    }
    {
        const std::string wav_short = "./cosyvoice-clone-guard-short.wav";
        if (!write_wav_f32(wav_short, 16000, 0.3)) {
            fprintf(stderr, "FAIL: cannot write synthetic wav\n");
            return 1;
        }
        EngineOptions o = base_opts();
        o.reference_audio = wav_short;
        bool threw = false;
        try { Engine eng(o); } catch (const std::exception & e) {
            threw = true;
            fprintf(stderr, "      0.3 s reference -> threw: %s\n", e.what());
        }
        std::remove(wav_short.c_str());
        if (!threw) {
            fprintf(stderr, "FAIL: 0.3 s reference did not throw\n");
            return 1;
        }
    }

    fprintf(stderr, "[D] cross-lingual LM path (unforced greedy decode, short text)\n");
    std::vector<int> xl_tokens;
    StageTimings xl_timings;
    {
        EngineOptions o = base_opts();
        o.force_speech_tokens.clear();
        o.greedy = true;
        o.reference_audio = wav;
        Engine eng(o);
        const SynthesisResult res = eng.synthesize("Hi.");
        // Conservative floor: the bare template is ~7 BPE tokens plus the
        // text's own, so min_len is at least ~16; 14 leaves margin for
        // tokenizer drift while still failing the broken basis (which allows
        // EOS from ~2 tokens on this text).
        fprintf(stderr, "      %zu speech tokens, %zu samples\n",
                res.speech_tokens.size(), res.pcm.size());
        if (res.speech_tokens.size() < 14 || res.pcm.empty()) {
            fprintf(stderr, "FAIL: cross-lingual min-length basis does not cover "
                            "the template tokens\n");
            return 1;
        }
        xl_tokens = res.speech_tokens;
        xl_timings = res.timings;
    }

    fprintf(stderr, "[E] prefill composition (unforced greedy zero-shot)\n");
    {
        auto greedy_zero_shot = [&](const std::string & transcript) {
            EngineOptions o = base_opts();
            o.force_speech_tokens.clear();
            o.greedy = true;
            o.reference_audio = wav;
            o.prompt_text = transcript;
            Engine eng(o);
            return eng.synthesize("Hi.");
        };
        const SynthesisResult zs = greedy_zero_shot(prompt_text);
        const SynthesisResult zs_long = greedy_zero_shot(
            prompt_text + " And several additional words to lengthen the transcript.");
        fprintf(stderr, "      prefill: zs text_ids=%d prompt_stok=%d, "
                        "zs_long text_ids=%d, xl text_ids=%d prompt_stok=%d\n",
                zs.timings.n_text_ids, zs.timings.n_prompt_speech_tokens,
                zs_long.timings.n_text_ids,
                xl_timings.n_text_ids, xl_timings.n_prompt_speech_tokens);
        if (zs.speech_tokens == xl_tokens) {
            fprintf(stderr, "FAIL: zero-shot and cross-lingual decode identically - "
                            "the reference prompt block is not conditioning the LM\n");
            return 1;
        }
        if (zs.timings.n_prompt_speech_tokens <= 0 ||
            xl_timings.n_prompt_speech_tokens != 0) {
            fprintf(stderr, "FAIL: prompt speech tokens must be present in zero-shot "
                            "and absent in cross-lingual\n");
            return 1;
        }
        if (zs.timings.n_text_ids <= xl_timings.n_text_ids ||
            zs_long.timings.n_text_ids <= zs.timings.n_text_ids) {
            fprintf(stderr, "FAIL: text-id count does not grow with the transcript - "
                            "the transcript is not entering the LM prefill\n");
            return 1;
        }
    }

    fprintf(stderr, "PASS\n");
    return 0;
}
