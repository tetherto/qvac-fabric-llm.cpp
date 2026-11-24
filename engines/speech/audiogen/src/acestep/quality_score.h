#pragma once

// ACE-Step LM quality scoring: teacher-forced evaluation of how well the
// generated audio codes match the request, ported from the acestep.cpp
// prototype (pipeline-score) which mirrors acestep/core/scoring/lm_score.py.
//
// The scoring LM re-reads the audio codes through the "understand" prompt and
// teacher-forces each condition as a target continuation:
//   - metadata fields (bpm, duration, keyscale, language, timesignature) score
//     as rank-weighted top-k recall of their YAML line;
//   - caption and lyrics score as normalized PMI: the sigmoid-squashed gap
//     between the target's mean log-prob conditioned on the codes and the
//     same target's log-prob under a no-input prompt.
// The global score is the weight-normalized sum of caption, lyrics, and the
// mean metadata component.

#include "bpe_tokenizer.h"
#include "lm_ggml.h"
#include "lm_pipeline.h"

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace tts_cpp::acestep {

struct QualityScoreParams {
    int    top_k           = 10;
    double score_scale     = 0.1;
    double caption_weight  = 0.50;
    double lyrics_weight   = 0.30;
    double metadata_weight = 0.20;

    // Optional progress hook fired once per teacher-forced target token with
    // (tokens_scored, total_target_tokens). Return false to cancel.
    std::function<bool(int cur, int total)> on_step;
};

enum class QualityMetric {
    TopKRecall,
    PmiNormalized,
};

struct QualityCondition {
    double                score  = 0.0;
    QualityMetric         metric = QualityMetric::TopKRecall;
    std::map<int, double> recall_per_k;
    double                conditional_log_prob   = 0.0;
    double                unconditional_log_prob = 0.0;
    double                pmi                    = 0.0;
};

struct QualityScoreResult {
    double                                  global_score = 0.0;
    std::map<std::string, QualityCondition> conditions;
    std::string                             report;
};

// Sigmoid-normalized PMI, numerically stable on both tails.
double quality_normalized_pmi(double conditional, double unconditional, double scale);

// YAML value formatting for teacher-forced metadata targets (Python
// yaml.dump parity: plain scalars stay bare, everything else single-quotes).
bool        quality_yaml_plain_safe(const std::string & value);
std::string quality_yaml_string(const std::string & value);

// Target text for one metadata condition: "<think>\n<key>: <value>\n</think>\n".
// The caption target embeds the CoT YAML block (wrapping included) instead of
// a plain scalar, matching lm_score.py's metadata.setdefault("caption", ...).
std::string quality_metadata_target(const std::string & key, const std::string & value);
std::string quality_metadata_target(const std::string & key, long long value);
std::string quality_caption_target(const std::string & caption);
std::string quality_lyrics_target(const std::string & lyrics);

// Tokenize a target while mapping <think>/</think> to their added-token ids
// (the tokenizer's text pass would otherwise split them into plain BPE runs).
std::vector<int> quality_encode_target(const BpeTokenizer & bpe, const std::string & text);

// Weight-normalized global score over the evaluated conditions. Returns false
// with `error` set when no component carries positive weight. `report` gets
// the human-readable per-component and per-condition breakdown.
bool quality_weighted_global(const std::map<std::string, QualityCondition> & conditions,
                             const QualityScoreParams & params, double & global_score,
                             std::string & report, std::string & error);

// Score `codes` (FSQ indices, i.e. LM token - AUDIO_CODE_BASE) against the
// fully-resolved prompt. Runs teacher-forced LM forwards on KV set 0 and
// resets it between conditions; the model's KV contents are not preserved.
// Returns false with `error` set on failure or cancellation via on_step.
bool compute_quality_score(LMModel * m, const BpeTokenizer & bpe, const AcePrompt & prompt,
                           const std::vector<int> & codes, const QualityScoreParams & params,
                           QualityScoreResult & result, std::string & error);

} // namespace tts_cpp::acestep
