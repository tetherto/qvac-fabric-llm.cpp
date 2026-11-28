#pragma once

// CosyVoice3 zero-shot voice-cloning front-end: turns a reference recording
// into the four prompt tensors the engine otherwise reads from voice.gguf
// (prompt_stok / prompt_token share one token stream, prompt_feat, embedding).
//
// The chain mirrors the upstream CosyVoice frontend exactly — and therefore
// deliberately does NOT reuse Chatterbox's reference-audio preprocessing
// (-27 LUFS loudness normalisation, 10 s conditioning trims): those are
// Chatterbox training constants, and the CosyVoice frontend applies neither.
//
//   tokens    : load_wav → mono → resample→16 kHz → speech_tokenizer_v3
//               (whisper-style 128-mel log-mel → 12-block FSMN encoder → FSQ)
//   embedding : the same 16 kHz stream → Kaldi fbank(80) → per-utterance
//               mean subtraction → CAM++ → raw (unnormalised) 192-d vector
//   prompt_feat: load_wav → mono → resample native→24 kHz (NOT via the 16 kHz
//               stream) → 80-mel log-mel (n_fft 1920, hop 480, fmin 0,
//               fmax sr/2)
//   alignment : token_len = min(mel_rows / 2, n_tokens); both streams are
//               truncated so mel_len1 == 2 * token_len (the flow's
//               token_mel_ratio invariant).
//
// Duration limits (enforced here, surfaced by the engine as load errors):
// hard minimum 0.5 s (below that the token/mel streams are degenerate), hard
// maximum 30 s (the upstream tokenizer asserts <= 30 s; long prompts also eat
// the flow's 15000-frame noise budget), advisory warning under 3 s.

#include <cstdint>
#include <string>
#include <vector>

typedef struct ggml_backend * ggml_backend_t;

struct cosyvoice_prompt {
    // 25 Hz speech tokens of the reference clip; fed to the LM prompt
    // (zero-shot mode) and always to the flow as its prompt_token prefix.
    std::vector<int32_t> prompt_stok;
    // Row-major (mel_len1, 80) log-mel of the reference clip.
    std::vector<float>   prompt_feat;
    int                  mel_len1 = 0;   // == 2 * prompt_stok.size()
    // Raw CAM++ speaker embedding (192); the flow front-end L2-normalises.
    std::vector<float>   embedding;
};

// Runs the full front-end.  `backend` is where the tokenizer encoder graph
// runs (nullptr → internal CPU backend); CAM++ always runs on the scalar CPU
// path.  On failure returns false with a one-line reason in `err`.
bool cosyvoice_frontend_run(const std::string & reference_wav,
                            const std::string & s3tok_gguf,
                            const std::string & campplus_gguf,
                            ggml_backend_t backend,
                            int n_threads,
                            cosyvoice_prompt & out,
                            std::string & err);
