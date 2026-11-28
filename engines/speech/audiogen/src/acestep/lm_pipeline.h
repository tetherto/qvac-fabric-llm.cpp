#pragma once

// ACE-Step LM pipeline: prompt/CoT building, top-k/p sampling and
// the Phase-2 audio-code decode loop layered on top of the LM core (lm_ggml) and
// the BPE tokenizer (bpe_tokenizer). Ported from acestep.cpp (prompt.h,
// sampling.h, pipeline-lm.cpp).
//
// This first cut targets the turbo text2music path with a fully-specified prompt
// (caption + lyrics + all metadata), so Phase 1 (CoT/lyric generation + metadata
// FSM) can be skipped and only Phase 2 (audio semantic codes) runs. CFG is not
// used here (single KV set); it can be added once the core grows multi-set KV.

#include "bpe_tokenizer.h"
#include "lm_ggml.h"

#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace tts_cpp::acestep {

// User-facing music prompt (mirrors acestep.cpp AcePrompt).
struct AcePrompt {
    std::string caption;
    std::string lyrics;
    float       duration      = 0.0f;  // seconds
    int         bpm           = 0;
    std::string keyscale;
    std::string timesignature;
    std::string vocal_language;
};

struct LmSampleParams {
    float    temperature    = 0.85f;
    float    top_p          = 0.9f;
    int      top_k          = 0;      // 0 = disabled (top_p only), matches upstream
    float    cfg_scale      = 2.0f;   // >1 enables CFG (needs a 2nd KV set)
    uint32_t seed           = 0;
    int      max_new_tokens = 0;      // 0 => derive from duration (dur*5 + 100)
    bool     verbose        = false;  // gate progress/metadata/code-dump logs

    // Optional progress hook, fired periodically with (tokens_generated,
    // max_tokens) by both the Phase-1 metadata/lyrics loop and the Phase-2
    // code loop (the longest stage on CPU). Return false to request
    // cancellation: the active loop aborts and the phase returns false.
    std::function<bool(int cur, int total)> on_step;
};

// Temperature -> top_k -> top_p -> softmax -> multinomial. Mutates `logits`.
// Ported verbatim from acestep.cpp/src/sampling.h.
// index_base: `logits` is the [index_base, V_full) suffix of a vector whose
// dropped prefix is fully masked. Picks return slice_index + index_base; the
// degenerate r==0 / all-masked edges return absolute 0, where the historical
// full-vector walk lands.
int sample_top_k_p(float * logits, int V, float temperature, float top_p, int top_k, std::mt19937 & rng,
                   int index_base = 0);

// FSM forced-token fast path: equivalent to masking all but `token` to -1e9 and
// running sample_top_k_p, including its RNG consumption and r==0 edge case.
int lm_consume_forced(int token, float temperature, std::mt19937 & rng);

// YAML CoT block for the prompt's set fields (Python yaml.dump parity:
// sorted keys, captions wrapped past column 80). Shared with quality scoring.
std::string lm_cot_yaml(const AcePrompt & prompt);

// Build the Qwen3 chat prompt with an injected CoT metadata block (Phase 2).
// The assistant turn stays open so the LM emits audio codes then <|im_end|>.
std::vector<int> build_lm_prompt_with_cot(const BpeTokenizer & bpe, const AcePrompt & prompt);

// User message for the INSPIRE prompt: the caption, plus the instrumental hint
// when the caller explicitly requested "[Instrumental]" lyrics.
std::string lm_inspire_user_message(const std::string & caption, const std::string & lyrics);

// Phase-1 expansion mode.
enum class LmPhase1Mode {
    // Gap-fill: only empty fields are generated; a bare caption (no lyrics)
    // runs the INSPIRE expansion.
    Auto,
    // Simple Mode: always run the INSPIRE expansion — the caption is rewritten
    // and lyrics are written even when metadata is complete. "[Instrumental]"
    // lyrics forward the instrumental hint.
    Inspire,
    // Query Rewriting: FORMAT the request — the caption is rewritten into a
    // detailed musical description and the lyrics are regenerated preserving
    // their content ("# Caption / # Lyric" user message).
    Format,
};

// User message for the FORMAT prompt (Query Rewriting).
std::string lm_format_user_message(const std::string & caption, const std::string & lyrics);

// Phase 1: auto-generate missing metadata (bpm/keyscale/duration/timesignature/
// language) — and lyrics for a bare caption — via FSM-constrained decoding.
// Mutates `prompt` in place (gap-fill: only empty fields are overwritten;
// Inspire/Format also rewrite the caption and lyrics). use_fsm=false lets the
// LM free-run (lower reliability). Returns false on failure or cancellation
// via params.on_step.
bool lm_generate_phase1(LMModel * m, const BpeTokenizer & bpe, AcePrompt & prompt, const LmSampleParams & params,
                        bool use_fsm = true, bool use_cot_caption = true,
                        LmPhase1Mode mode = LmPhase1Mode::Auto);

// Phase 2: generate FSQ audio semantic codes for a fully-specified prompt.
// Single sequence, no CFG. Returns the raw codes (already offset-subtracted,
// i.e. token - AUDIO_CODE_BASE). Returns false on failure.
bool lm_generate_codes(LMModel *              m,
                       const BpeTokenizer &   bpe,
                       const AcePrompt &      prompt,
                       const LmSampleParams & params,
                       std::vector<int> &     codes_out);

// Understand ("listener") prompt: the system understand instruction with the
// FSQ codes as the user turn's raw audio tokens (AUDIO_CODE_BASE + code).
// The no-input variant is the unconditional baseline quality scoring uses.
std::vector<int> lm_understand_prompt(const BpeTokenizer & bpe, const int * codes, int n_codes);
std::vector<int> lm_understand_unconditional_prompt(const BpeTokenizer & bpe);

// Decode-token budget for lm_understand: the caller's request (or the 2048
// default) capped to the KV room left after the prompt, never negative.
// Pure; exposed for unit testing.
int lm_understand_token_budget(int max_seq_len, size_t prompt_tokens, int requested);

// Reverse pipeline decode: describe FSQ audio codes as metadata + caption.
// FSM-constrained CoT for the metadata block; decoding stops at </think> (the
// caption lives inside the CoT block and the reference's post-think lyric
// text is unused here). A non-empty language_hint is forced into the FSM and
// echoed to the output. Honors params.on_step for progress/cancellation.
// Returns false on failure or cancellation.
bool lm_understand(LMModel *              m,
                   const BpeTokenizer &   bpe,
                   const std::vector<int> & codes,
                   const LmSampleParams & params,
                   const std::string &    language_hint,
                   AcePrompt &            out);

} // namespace tts_cpp::acestep
