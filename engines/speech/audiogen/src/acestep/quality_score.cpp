#include "quality_score.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

namespace tts_cpp::acestep {

namespace {

struct TeacherForcedStats {
    double                mean_log_prob  = -std::numeric_limits<double>::infinity();
    double                weighted_top_k = 0.0;
    std::map<int, double> recall_per_k;
};

struct ScoreTarget {
    std::string      key;
    std::vector<int> tokens;
    bool             pmi = false;
};

struct ScoreProgress {
    const std::function<bool(int cur, int total)> & on_step;
    int                                             done  = 0;
    int                                             total = 0;

    bool tick() {
        done++;
        return !on_step || on_step(done, total);
    }
};

void append_text(const BpeTokenizer & bpe, const std::string & text, std::vector<int> & ids) {
    const std::vector<int> part = bpe_encode(bpe, text, false);
    ids.insert(ids.end(), part.begin(), part.end());
}

bool validate_target_fits(const LMModel * m, const std::vector<int> & prompt,
                          const std::vector<int> & target, std::string & error) {
    if (prompt.empty()) {
        error = "scoring prompt tokenized to an empty sequence";
        return false;
    }
    if (target.empty()) {
        error = "scoring target tokenized to an empty sequence";
        return false;
    }
    const size_t needed = prompt.size() + target.size();
    if (needed > (size_t) lm_model_config(m).max_seq_len) {
        error = "prompt + target needs " + std::to_string(needed) + " tokens, exceeding LM max_seq " +
                std::to_string(lm_model_config(m).max_seq_len);
        return false;
    }
    return true;
}

double target_log_prob(const std::vector<float> & logits, int wanted, int * rank_out) {
    const float max_logit = *std::max_element(logits.begin(), logits.end());
    double      exp_sum   = 0.0;
    int         rank      = 1;
    for (float logit : logits) {
        exp_sum += std::exp((double) logit - max_logit);
        if (logit > logits[(size_t) wanted]) rank++;
    }
    *rank_out = rank;
    if (!std::isfinite(exp_sum) || exp_sum <= 0.0) return std::numeric_limits<double>::quiet_NaN();
    return (double) logits[(size_t) wanted] - max_logit - std::log(exp_sum);
}

void fill_recall_per_k(const std::vector<int> & ranks, int top_k, int vocab,
                       std::map<int, double> & recall_per_k) {
    for (int k = 1; k <= top_k; k++) {
        int hits = 0;
        for (int rank : ranks) {
            if (rank <= std::min(k, vocab)) hits++;
        }
        recall_per_k[k] = (double) hits / (double) ranks.size();
    }
}

bool score_target_tokens(LMModel * m, const std::vector<int> & target, int top_k,
                         ScoreProgress & progress, std::vector<float> & logits,
                         TeacherForcedStats & stats, std::string & error) {
    const int        vocab           = lm_model_config(m).vocab_size;
    const int        effective_top_k = std::min(top_k, vocab);
    double           log_prob_sum    = 0.0;
    double           rank_sum        = 0.0;
    std::vector<int> ranks;
    ranks.reserve(target.size());

    for (size_t pos = 0; pos < target.size(); pos++) {
        const int wanted = target[pos];
        if (wanted < 0 || wanted >= vocab) {
            error = "target token " + std::to_string(wanted) + " is outside LM vocabulary";
            return false;
        }
        int          rank     = 0;
        const double log_prob = target_log_prob(logits, wanted, &rank);
        if (!std::isfinite(log_prob)) {
            error = "LM produced non-finite logits during teacher forcing";
            return false;
        }
        log_prob_sum += log_prob;
        ranks.push_back(rank);
        if (rank <= effective_top_k) {
            rank_sum += 1.0 - (double) (rank - 1) / (double) top_k;
        }
        if (!progress.tick()) {
            error = "quality scoring cancelled";
            return false;
        }
        if (pos + 1 < target.size()) {
            const int32_t token = wanted;
            if (!lm_model_forward(m, &token, 1, logits)) {
                error = "LM forward failed during teacher forcing";
                return false;
            }
        }
    }

    stats.mean_log_prob  = log_prob_sum / (double) target.size();
    stats.weighted_top_k = rank_sum / (double) target.size();
    fill_recall_per_k(ranks, top_k, vocab, stats.recall_per_k);
    return true;
}

bool teacher_force(LMModel * m, const std::vector<int> & prompt, const std::vector<int> & target,
                   int top_k, ScoreProgress & progress, TeacherForcedStats & stats,
                   std::string & error) {
    if (!validate_target_fits(m, prompt, target, error)) return false;

    std::vector<int32_t> prompt32(prompt.begin(), prompt.end());
    std::vector<float>   logits;
    lm_reset(m, 0);
    if (!lm_model_forward(m, prompt32.data(), (int) prompt32.size(), logits)) {
        error = "LM forward failed on the scoring prompt";
        return false;
    }
    return score_target_tokens(m, target, top_k, progress, logits, stats, error);
}

std::string think_wrap(const std::string & body) {
    return "<think>\n" + body + "\n</think>\n";
}

void append_metadata_targets(const BpeTokenizer & bpe, const AcePrompt & prompt,
                             std::vector<ScoreTarget> & targets) {
    if (prompt.bpm > 0) {
        targets.push_back({ "bpm", quality_encode_target(bpe, quality_metadata_target("bpm", (long long) prompt.bpm)), false });
    }
    if (prompt.duration > 0.0f) {
        targets.push_back({ "duration",
                            quality_encode_target(bpe, quality_metadata_target("duration", (long long) prompt.duration)),
                            false });
    }
    if (!prompt.keyscale.empty()) {
        targets.push_back({ "keyscale", quality_encode_target(bpe, quality_metadata_target("keyscale", prompt.keyscale)), false });
    }
    if (!prompt.vocal_language.empty()) {
        targets.push_back({ "language",
                            quality_encode_target(bpe, quality_metadata_target("language", prompt.vocal_language)),
                            false });
    }
    if (!prompt.timesignature.empty()) {
        targets.push_back({ "timesignature",
                            quality_encode_target(bpe, quality_metadata_target("timesignature", prompt.timesignature)),
                            false });
    }
}

std::vector<ScoreTarget> build_targets(const BpeTokenizer & bpe, const AcePrompt & prompt) {
    std::vector<ScoreTarget> targets;
    append_metadata_targets(bpe, prompt, targets);
    targets.push_back({ "caption", quality_encode_target(bpe, quality_caption_target(prompt.caption)), true });
    if (!prompt.lyrics.empty()) {
        targets.push_back({ "lyrics", quality_encode_target(bpe, quality_lyrics_target(prompt.lyrics)), true });
    }
    return targets;
}

int total_scored_tokens(const std::vector<ScoreTarget> & targets) {
    int total = 0;
    for (const ScoreTarget & target : targets) {
        total += (int) target.tokens.size() * (target.pmi ? 2 : 1);
    }
    return total;
}

bool score_recall_condition(LMModel * m, const std::vector<int> & conditional_prompt,
                            const ScoreTarget & target, const QualityScoreParams & params,
                            ScoreProgress & progress, QualityCondition & condition,
                            std::string & error) {
    TeacherForcedStats stats;
    if (!teacher_force(m, conditional_prompt, target.tokens, params.top_k, progress, stats, error)) {
        error = target.key + " recall failed: " + error;
        return false;
    }
    condition.score        = stats.weighted_top_k;
    condition.metric       = QualityMetric::TopKRecall;
    condition.recall_per_k = std::move(stats.recall_per_k);
    return true;
}

bool score_pmi_condition(LMModel * m, const std::vector<int> & conditional_prompt,
                         const std::vector<int> & unconditional_prompt, const ScoreTarget & target,
                         const QualityScoreParams & params, ScoreProgress & progress,
                         QualityCondition & condition, std::string & error) {
    TeacherForcedStats conditional;
    TeacherForcedStats unconditional;
    if (!teacher_force(m, conditional_prompt, target.tokens, params.top_k, progress, conditional, error)) {
        error = target.key + " conditional PMI failed: " + error;
        return false;
    }
    if (!teacher_force(m, unconditional_prompt, target.tokens, params.top_k, progress, unconditional, error)) {
        error = target.key + " unconditional PMI failed: " + error;
        return false;
    }
    condition.metric                 = QualityMetric::PmiNormalized;
    condition.conditional_log_prob   = conditional.mean_log_prob;
    condition.unconditional_log_prob = unconditional.mean_log_prob;
    condition.pmi                    = conditional.mean_log_prob - unconditional.mean_log_prob;
    condition.score = quality_normalized_pmi(conditional.mean_log_prob, unconditional.mean_log_prob,
                                             params.score_scale);
    return true;
}

bool score_all_targets(LMModel * m, const std::vector<int> & conditional_prompt,
                       const std::vector<int> & unconditional_prompt,
                       const std::vector<ScoreTarget> & targets, const QualityScoreParams & params,
                       ScoreProgress & progress, QualityScoreResult & result, std::string & error) {
    for (const ScoreTarget & target : targets) {
        QualityCondition condition;
        if (target.pmi) {
            if (!score_pmi_condition(m, conditional_prompt, unconditional_prompt, target, params,
                                     progress, condition, error)) {
                return false;
            }
        } else if (!score_recall_condition(m, conditional_prompt, target, params, progress,
                                           condition, error)) {
            return false;
        }
        result.conditions[target.key] = std::move(condition);
    }
    return true;
}

struct ScoreComponent {
    const char * name;
    double       score;
    double       weight;
};

double metadata_component_score(const std::map<std::string, QualityCondition> & conditions,
                                int & metadata_count) {
    double sum     = 0.0;
    metadata_count = 0;
    for (const auto & item : conditions) {
        if (item.first != "caption" && item.first != "lyrics") {
            sum += item.second.score;
            metadata_count++;
        }
    }
    return metadata_count > 0 ? sum / (double) metadata_count : 0.0;
}

std::vector<ScoreComponent> collect_components(const std::map<std::string, QualityCondition> & conditions,
                                               const QualityScoreParams & params) {
    std::vector<ScoreComponent> components;
    auto                        caption = conditions.find("caption");
    if (caption != conditions.end()) {
        components.push_back({ "caption", caption->second.score, params.caption_weight });
    }
    auto lyrics = conditions.find("lyrics");
    if (lyrics != conditions.end()) {
        components.push_back({ "lyrics", lyrics->second.score, params.lyrics_weight });
    }
    int          metadata_count = 0;
    const double metadata_mean  = metadata_component_score(conditions, metadata_count);
    if (metadata_count > 0) {
        components.push_back({ "metadata", metadata_mean, params.metadata_weight });
    }
    std::sort(components.begin(), components.end(),
              [](const ScoreComponent & a, const ScoreComponent & b) { return a.weight > b.weight; });
    return components;
}

void append_condition_report(const std::map<std::string, QualityCondition> & conditions,
                             std::ostringstream & report) {
    report << "\nPer-condition scores (0-1):";
    for (const auto & item : conditions) {
        report << "\n  " << item.first << ": " << item.second.score << " ("
               << (item.second.metric == QualityMetric::TopKRecall ? "Top-k Recall" : "PMI (Norm)")
               << ")";
    }
}

} // namespace

double quality_normalized_pmi(double conditional, double unconditional, double scale) {
    const double x = (conditional - unconditional) / scale;
    if (x >= 0.0) {
        return 1.0 / (1.0 + std::exp(-x));
    }
    const double e = std::exp(x);
    return e / (1.0 + e);
}

bool quality_yaml_plain_safe(const std::string & value) {
    if (value.empty() || value.front() == ' ' || value.back() == ' ') return false;
    static const char unsafe_leads[] = "-?:!&*#{}[],|>@`";
    for (char lead : unsafe_leads) {
        if (lead != '\0' && value.front() == lead) return false;
    }
    if (value.find('\n') != std::string::npos || value.find('\r') != std::string::npos ||
        value.find(": ") != std::string::npos || value.find(" #") != std::string::npos) {
        return false;
    }
    std::string lower;
    lower.reserve(value.size());
    for (char c : value) lower.push_back((char) std::tolower((unsigned char) c));
    static const char * reserved[] = {
        "null", "~", "true", "false", "yes", "no", "on", "off", ".nan", ".inf", "-.inf",
    };
    for (const char * word : reserved) {
        if (lower == word) return false;
    }
    char * end = nullptr;
    (void) strtod(value.c_str(), &end);
    return !(end && *end == '\0');
}

std::string quality_yaml_string(const std::string & value) {
    if (quality_yaml_plain_safe(value)) return value;
    std::string quoted = "'";
    for (char c : value) {
        if (c == '\'') {
            quoted += "''";
        } else if (c == '\n') {
            quoted += "\n  ";
        } else if (c != '\r') {
            quoted += c;
        }
    }
    quoted += "'";
    return quoted;
}

std::string quality_metadata_target(const std::string & key, const std::string & value) {
    return think_wrap(key + ": " + quality_yaml_string(value));
}

std::string quality_metadata_target(const std::string & key, long long value) {
    return think_wrap(key + ": " + std::to_string(value));
}

std::string quality_caption_target(const std::string & caption) {
    if (caption.empty()) {
        return think_wrap("caption: " + quality_yaml_string(caption));
    }
    AcePrompt caption_only;
    caption_only.caption = caption;
    std::string yaml     = lm_cot_yaml(caption_only);
    if (!yaml.empty() && yaml.back() == '\n') yaml.pop_back();
    return think_wrap(yaml);
}

std::string quality_lyrics_target(const std::string & lyrics) {
    return "<think>\n</think>\n# Lyric\n" + lyrics + "\n";
}

std::vector<int> quality_encode_target(const BpeTokenizer & bpe, const std::string & text) {
    std::vector<int> ids;
    size_t           pos = 0;
    while (pos < text.size()) {
        const size_t think     = text.find("<think>", pos);
        const size_t think_end = text.find("</think>", pos);
        const size_t next      = std::min(think == std::string::npos ? text.size() : think,
                                          think_end == std::string::npos ? text.size() : think_end);
        if (next > pos) append_text(bpe, text.substr(pos, next - pos), ids);
        if (next == text.size()) break;
        if (next == think) {
            ids.push_back(TOKEN_THINK);
            pos = next + strlen("<think>");
        } else {
            ids.push_back(TOKEN_THINK_END);
            pos = next + strlen("</think>");
        }
    }
    return ids;
}

bool quality_weighted_global(const std::map<std::string, QualityCondition> & conditions,
                             const QualityScoreParams & params, double & global_score,
                             std::string & report, std::string & error) {
    const std::vector<ScoreComponent> components = collect_components(conditions, params);
    double                            total_weight = 0.0;
    for (const ScoreComponent & component : components) total_weight += component.weight;
    if (total_weight <= 0.0) {
        error = "active score components have zero total weight";
        return false;
    }

    global_score = 0.0;
    std::ostringstream out;
    out << std::fixed << std::setprecision(4);
    for (const ScoreComponent & component : components) {
        const double normalized_weight = component.weight / total_weight;
        const double contribution      = component.score * normalized_weight;
        global_score += contribution;
        out << "  " << component.name << " | Score: " << component.score
            << " | Weight: " << std::setprecision(2) << normalized_weight
            << " -> Contrib: +" << std::setprecision(4) << contribution << "\n";
    }
    append_condition_report(conditions, out);
    report = out.str();
    return true;
}

bool compute_quality_score(LMModel * m, const BpeTokenizer & bpe, const AcePrompt & prompt,
                           const std::vector<int> & codes, const QualityScoreParams & params,
                           QualityScoreResult & result, std::string & error) {
    result = {};
    if (!m) {
        error = "quality scoring requires a loaded LM";
        return false;
    }
    if (codes.empty()) {
        error = "quality scoring requires audio codes";
        return false;
    }
    if (params.top_k <= 0 || params.score_scale <= 0.0) {
        error = "top_k and score_scale must be positive";
        return false;
    }
    if (params.caption_weight < 0.0 || params.lyrics_weight < 0.0 || params.metadata_weight < 0.0) {
        error = "score component weights cannot be negative";
        return false;
    }
    const int max_code = std::min(AUDIO_CODE_COUNT - 1, lm_model_config(m).vocab_size - AUDIO_CODE_BASE - 1);
    for (int code : codes) {
        if (code < 0 || code > max_code) {
            error = "audio code " + std::to_string(code) + " is outside [0, " + std::to_string(max_code) + "]";
            return false;
        }
    }

    const std::vector<int>         conditional_prompt   = lm_understand_prompt(bpe, codes.data(), (int) codes.size());
    const std::vector<int>         unconditional_prompt = lm_understand_unconditional_prompt(bpe);
    const std::vector<ScoreTarget> targets              = build_targets(bpe, prompt);
    if (targets.empty()) {
        error = "no conditions to evaluate";
        return false;
    }

    ScoreProgress progress = { params.on_step, 0, total_scored_tokens(targets) };
    if (!score_all_targets(m, conditional_prompt, unconditional_prompt, targets, params, progress,
                           result, error)) {
        return false;
    }
    return quality_weighted_global(result.conditions, params, result.global_score, result.report, error);
}

} // namespace tts_cpp::acestep
