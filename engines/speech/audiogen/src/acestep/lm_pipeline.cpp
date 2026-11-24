#include "lm_pipeline.h"

#include "metadata_fsm.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Ported from acestep.cpp: sampling.h (sample_top_k_p), prompt.h (CoT/prompt
// building) and pipeline-lm.cpp (run_phase2_batch + generate_phase1_batch,
// single-sequence subset).

namespace tts_cpp::acestep {

// LM system instructions (task-types.h).
static const char * LM_INSTRUCTION = "Generate audio semantic tokens based on the given conditions:";
static const char * LM_INSPIRE_INSTRUCTION =
    "Expand the user's input into a more detailed and specific musical description:";
static const char * LM_FORMAT_INSTRUCTION =
    "Format the user's input into a more detailed and specific musical description:";
static const char * LM_UNDERSTAND_INSTRUCTION =
    "Understand the given musical conditions and describe the audio semantics accordingly:";
static const char * LM_INSTRUMENTAL_LYRICS       = "[Instrumental]";
static const char * LM_INSPIRE_INSTRUMENTAL_HINT = "\n\ninstrumental: true";

std::string lm_inspire_user_message(const std::string & caption, const std::string & lyrics) {
    if (lyrics == LM_INSTRUMENTAL_LYRICS) return caption + LM_INSPIRE_INSTRUMENTAL_HINT;
    return caption;
}

std::string lm_format_user_message(const std::string & caption, const std::string & lyrics) {
    return "# Caption\n" + caption + "\n\n# Lyric\n" + lyrics;
}

struct TokenProb {
    int   id;
    float prob;
};

static double lm_ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// Phase 1 stops at <|im_end|> long before its token cap, so its length is
// unknowable up front and its ticks report the engine's unknown-total marker.
static constexpr int LM_PROGRESS_UNKNOWN_TOTAL = -1;

// Understand decode cap when the caller sets none; matches Phase 1's default.
static constexpr int LM_UNDERSTAND_DEFAULT_MAX_NEW = 2048;

// Throttled progress for the decode loops: every 8 tokens (each step is a full
// LM forward, so this is cheap relative to the work) plus the final step.
// Returns false when the hook requests cancellation.
static bool lm_report_progress(const LmSampleParams & params, int step, int total, int shown_total) {
    if (!params.on_step) return true;
    if ((step & 7) != 0 && step != total - 1) return true;
    return params.on_step(step + 1, shown_total);
}

// Phase timing prints: on with verbose or ACESTEP_LM_TIMING=1.
static bool lm_timing_enabled(bool verbose) {
    return verbose || std::getenv("ACESTEP_LM_TIMING") != nullptr;
}

// Forced-token fast path: with a single live logit the masked sampler's softmax
// sum is exactly 1.0 (exp(0)=1, exp(-inf/-1e9)=0), so it draws from dist(0,1)
// and returns the live token - except the r==0 edge, where its acc-walk stops
// at index 0. Consume one draw to keep the RNG stream identical.
int lm_consume_forced(int token, float temperature, std::mt19937 & rng) {
    if (temperature <= 0.0f) return token;  // argmax path draws nothing
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    const float r = dist(rng);
    if (r == 0.0f && token != 0) return 0;
    return token;
}

int sample_top_k_p(float * logits, int V, float temperature, float top_p, int top_k, std::mt19937 & rng,
                   int index_base) {
    if (temperature <= 0.0f) {
        return (int) (std::max_element(logits, logits + V) - logits) + index_base;
    }

    static thread_local std::vector<float>     tmp_buf;
    static thread_local std::vector<TokenProb> sorted_buf;
    static thread_local std::vector<int>       cand_id;
    static thread_local std::vector<float>     cand_e;
    static thread_local std::vector<uint8_t>   cand_dead;

    float inv_temp  = 1.0f / temperature;
    float max_logit = -INFINITY;
    for (int i = 0; i < V; i++) {
        logits[i] *= inv_temp;
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    if (top_k > 0 && top_k < V) {
        tmp_buf.resize(V);
        memcpy(tmp_buf.data(), logits, V * sizeof(float));
        std::nth_element(tmp_buf.begin(), tmp_buf.begin() + (top_k - 1), tmp_buf.end(), std::greater<float>());
        float threshold = tmp_buf[top_k - 1];
        for (int i = 0; i < V; i++) {
            if (logits[i] < threshold) logits[i] = -INFINITY;
        }
    }

    if (top_p > 0.0f && top_p < 1.0f) {
        // One ascending sweep: exp + sum + candidate collect. Bit-identical to
        // the historical multi-pass form: expf(-inf) == 0.0f exactly, adding
        // 0.0f preserves the accumulator, and top_k masking never drops the max.
        float cutoff  = max_logit - 16.0f;
        float sum_exp = 0.0f;
        cand_id.clear();
        cand_e.clear();
        for (int i = 0; i < V; i++) {
            float x = logits[i];
            float e = (x == -INFINITY) ? 0.0f : expf(x - max_logit);
            sum_exp += e;
            if (x >= cutoff) {
                cand_id.push_back(i);
                cand_e.push_back(e);
            }
        }

        int K = (int) cand_id.size();
        if (K == 0) return 0;  // all -inf: historical acc-walk lands on index 0

        float inv_sum = 1.0f / sum_exp;
        sorted_buf.clear();
        for (int j = 0; j < K; j++) sorted_buf.push_back({ j, cand_e[j] * inv_sum });
        std::sort(sorted_buf.begin(), sorted_buf.end(),
                  [](const TokenProb & a, const TokenProb & b) { return a.prob > b.prob; });

        cand_dead.assign((size_t) K, 0);
        float cum = 0.0f;
        for (int k = 0; k < K; k++) {
            if (k > 0 && cum >= top_p) cand_dead[sorted_buf[k].id] = 1;
            cum += sorted_buf[k].prob;
        }

        // The top-prob candidate survives and carries max_logit, so the
        // historical re-softmax reduces to the surviving e values in id order.
        float sum = 0.0f;
        for (int j = 0; j < K; j++)
            if (!cand_dead[j]) sum += cand_e[j];

        std::uniform_real_distribution<float> dist(0.0f, sum);
        float r = dist(rng);
        if (r == 0.0f) return 0;  // historical acc-walk edge: absolute index 0 wins at acc 0
        float acc = 0.0f;
        for (int j = 0; j < K; j++) {
            if (cand_dead[j]) continue;
            acc += cand_e[j];
            if (acc >= r) return cand_id[j] + index_base;
        }
        return 0;
    }

    float max_val = -INFINITY;
    for (int i = 0; i < V; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < V; i++) {
        logits[i] = expf(logits[i] - max_val);
        sum += logits[i];
    }

    std::uniform_real_distribution<float> dist(0.0f, sum);
    float                                 r = dist(rng);
    if (r == 0.0f) return 0;  // same acc-walk edge: absolute index 0 wins at acc 0
    float acc = 0.0f;
    for (int i = 0; i < V; i++) {
        acc += logits[i];
        if (acc >= r) return i + index_base;
    }
    return 0;
}

// YAML CoT block (acestep.cpp prompt.h build_cot_yaml). Matches Python
// yaml.dump(allow_unicode=True, sort_keys=True): break + 2-space indent past col 80.
std::string lm_cot_yaml(const AcePrompt & prompt) {
    auto yaml_wrap = [](const std::string & key, const std::string & val) -> std::string {
        std::string result = key + ":";
        int         col    = (int) (key.size() + 1);
        size_t      i      = 0;
        while (i < val.size()) {
            size_t end = val.find(' ', i);
            if (end == std::string::npos) end = val.size();
            std::string word = val.substr(i, end - i);
            if (col > 80) {
                result += "\n  ";
                col = 2;
            } else {
                result += " ";
                col += 1;
            }
            result += word;
            col += (int) word.size();
            i = (end < val.size()) ? end + 1 : val.size();
        }
        result += "\n";
        return result;
    };

    std::string yaml;
    if (prompt.bpm > 0) yaml += "bpm: " + std::to_string(prompt.bpm) + "\n";
    if (!prompt.caption.empty()) yaml += yaml_wrap("caption", prompt.caption);
    if (prompt.duration > 0) yaml += "duration: " + std::to_string((int) prompt.duration) + "\n";
    if (!prompt.keyscale.empty()) yaml += "keyscale: " + prompt.keyscale + "\n";
    if (!prompt.vocal_language.empty()) yaml += "language: " + prompt.vocal_language + "\n";
    if (!prompt.timesignature.empty()) yaml += "timesignature: " + prompt.timesignature + "\n";
    return yaml;
}

std::vector<int> build_lm_prompt_with_cot(const BpeTokenizer & bpe, const AcePrompt & prompt) {
    std::string      cot = lm_cot_yaml(prompt);
    std::vector<int> ids;
    auto             append = [&](const std::string & text) {
        auto t = bpe_encode(bpe, text, false);
        ids.insert(ids.end(), t.begin(), t.end());
    };
    ids.push_back(TOKEN_IM_START);
    append(std::string("system\n# Instruction\n") + LM_INSTRUCTION + "\n\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    append("user\n# Caption\n" + prompt.caption + "\n\n# Lyric\n" + prompt.lyrics + "\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    append("assistant\n");
    ids.push_back(TOKEN_THINK);
    append("\n" + cot);
    ids.push_back(TOKEN_THINK_END);
    append("\n\n");
    return ids;
}

// Unconditional prompt for CFG (acestep.cpp build_lm_prompt_uncond_with_cot):
// bare user turn (no Caption/Lyric wrapper), empty CoT. Matches the training
// CFG dropout. negative_prompt is optional.
static std::vector<int> build_lm_prompt_uncond_with_cot(const BpeTokenizer & bpe, const std::string & negative_prompt) {
    std::vector<int> ids;
    auto             append = [&](const std::string & text) {
        auto t = bpe_encode(bpe, text, false);
        ids.insert(ids.end(), t.begin(), t.end());
    };
    ids.push_back(TOKEN_IM_START);
    append(std::string("system\n# Instruction\n") + LM_INSTRUCTION + "\n\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    append("user\n" + negative_prompt);
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    append("assistant\n");
    ids.push_back(TOKEN_THINK);
    append("\n\n");
    ids.push_back(TOKEN_THINK_END);
    append("\n\n");
    return ids;
}

// ---------------------------------------------------------------------------
// Phase 1: metadata / CoT auto-generation (acestep.cpp prompt.h + pipeline-lm)
// ---------------------------------------------------------------------------

// Parse the LM Phase-1 output text into metadata + lyrics (prompt.h
// parse_cot_and_lyrics). Returns true if any of bpm/duration was found.
static bool parse_cot_and_lyrics(const std::string & text, AcePrompt & out) {
    size_t ts = text.find("<think>");
    size_t te = text.find("</think>");

    std::string cot;
    std::string lyrics_after;
    if (ts != std::string::npos && te != std::string::npos) {
        cot          = text.substr(ts + 7, te - ts - 7);
        lyrics_after = text.substr(te + 8);
    } else if (te != std::string::npos) {
        cot          = text.substr(0, te);
        lyrics_after = text.substr(te + 8);
    } else {
        cot = text;
    }

    auto get_field = [&](const std::string & key) -> std::string {
        std::string needle = key + ":";
        size_t      p      = cot.find(needle);
        if (p == std::string::npos) return "";
        p += needle.size();
        while (p < cot.size() && (cot[p] == ' ' || cot[p] == '\'')) p++;
        size_t end = cot.find('\n', p);
        if (end == std::string::npos) end = cot.size();
        std::string val = cot.substr(p, end - p);
        while (!val.empty() && (val.back() == ' ' || val.back() == '\'' || val.back() == '\r')) val.pop_back();
        return val;
    };

    std::string bpm_s = get_field("bpm");
    if (!bpm_s.empty()) out.bpm = atoi(bpm_s.c_str());
    std::string dur_s = get_field("duration");
    if (!dur_s.empty()) out.duration = (float) atof(dur_s.c_str());
    std::string ks = get_field("keyscale");
    if (!ks.empty()) out.keyscale = ks;
    std::string ts_s = get_field("timesignature");
    if (!ts_s.empty()) out.timesignature = ts_s;
    std::string lang = get_field("language");
    if (!lang.empty()) out.vocal_language = lang;

    std::string cap = get_field("caption");
    if (!cap.empty()) {
        size_t cp = cot.find("caption:");
        if (cp != std::string::npos) {
            cp += 8;
            size_t end = cot.find("\nduration:", cp);
            if (end == std::string::npos) end = cot.find("\nkeyscale:", cp);
            if (end == std::string::npos) end = cot.size();
            std::string full_cap = cot.substr(cp, end - cp);
            std::string cleaned;
            bool        in_space = true;
            for (char ch : full_cap) {
                if (ch == '\n' || ch == '\r') ch = ' ';
                if (ch == ' ') {
                    if (!in_space) cleaned += ' ';
                    in_space = true;
                } else {
                    cleaned += ch;
                    in_space = false;
                }
            }
            while (!cleaned.empty() && cleaned.back() == ' ') cleaned.pop_back();
            while (!cleaned.empty() && cleaned.front() == ' ') cleaned.erase(cleaned.begin());
            if (!cleaned.empty()) out.caption = cleaned;
        }
    }

    if (!lyrics_after.empty()) {
        size_t s = lyrics_after.find_first_not_of(" \t\n\r");
        if (s != std::string::npos) lyrics_after = lyrics_after.substr(s);
        size_t lp = lyrics_after.find("# Lyric\n");
        if (lp != std::string::npos && lp < 64) lyrics_after = lyrics_after.substr(lp + 8);
        while (!lyrics_after.empty() &&
               (lyrics_after.back() == ' ' || lyrics_after.back() == '\n' || lyrics_after.back() == '\r'))
            lyrics_after.pop_back();
        if (!lyrics_after.empty()) out.lyrics = lyrics_after;
    }

    return (out.bpm > 0 || out.duration > 0);
}

// Phase-1 gap-fill prompt with lyrics/metadata (prompt.h build_lm_prompt): the
// assistant turn is left OPEN (no <think>) — the FSM drives the CoT YAML.
static std::vector<int> build_lm_prompt_phase1(const BpeTokenizer & bpe, const AcePrompt & prompt) {
    std::vector<int> ids;
    auto             append = [&](const std::string & text) {
        auto t = bpe_encode(bpe, text, false);
        ids.insert(ids.end(), t.begin(), t.end());
    };
    ids.push_back(TOKEN_IM_START);
    append(std::string("system\n# Instruction\n") + LM_INSTRUCTION + "\n\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    append("user\n# Caption\n" + prompt.caption + "\n\n# Lyric\n" + prompt.lyrics + "\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    append("assistant\n");
    return ids;
}

// INSPIRE prompt for bare captions (prompt.h build_custom_prompt).
static std::vector<int> build_custom_prompt(const BpeTokenizer & bpe, const std::string & sys,
                                            const std::string & user) {
    std::vector<int> ids;
    auto             append = [&](const std::string & text) {
        auto t = bpe_encode(bpe, text, false);
        ids.insert(ids.end(), t.begin(), t.end());
    };
    ids.push_back(TOKEN_IM_START);
    append("system\n" + sys + "\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    append("user\n" + user + "\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    append("assistant\n");
    return ids;
}

bool lm_generate_phase1(LMModel * m, const BpeTokenizer & bpe, AcePrompt & prompt, const LmSampleParams & params,
                        bool use_fsm, bool use_cot_caption, LmPhase1Mode mode) {
    const LMConfig & cfg = lm_model_config(m);
    const int        V   = cfg.vocab_size;

    const AcePrompt base        = prompt;
    const bool      format      = mode == LmPhase1Mode::Format;
    const bool      need_lyrics = base.lyrics.empty();
    const bool      inspire     = !format && (need_lyrics || mode == LmPhase1Mode::Inspire);
    const bool      gen_lyrics  = format || inspire || need_lyrics;
    const bool      lyrics_mode = gen_lyrics;
    const bool      stop_at_reasoning = !gen_lyrics;

    // With the FSM active and the reasoning stop set, no sampled token can ever
    // reach the audio-code band, so the forward projects only the prefix head.
    const int V_lim = (use_fsm && stop_at_reasoning) ? AUDIO_CODE_BASE : 0;
    const int V_eff = V_lim > 0 ? V_lim : V;

    std::vector<int> prompt_tokens;
    if (format) {
        prompt_tokens = build_custom_prompt(bpe, std::string("# Instruction\n") + LM_FORMAT_INSTRUCTION,
                                            lm_format_user_message(base.caption, base.lyrics));
    } else if (inspire) {
        prompt_tokens = build_custom_prompt(bpe, std::string("# Instruction\n") + LM_INSPIRE_INSTRUCTION,
                                            lm_inspire_user_message(base.caption, base.lyrics));
    } else {
        prompt_tokens = build_lm_prompt_phase1(bpe, base);
    }

    // FSM setup: constrain the CoT YAML. Force any user-provided metadata.
    MetadataFSM fsm;
    if (use_fsm) {
        fsm.init(bpe, V_eff, params.verbose);
        fsm.skip_caption = !use_cot_caption && !inspire && !format;
        if (base.bpm > 0)               fsm.force_field(bpe, MetadataFSM::BPM_VALUE, std::to_string(base.bpm));
        if (base.duration > 0)          fsm.force_field(bpe, MetadataFSM::DURATION_VALUE,
                                                        std::to_string((int) base.duration));
        if (!base.keyscale.empty())     fsm.force_field(bpe, MetadataFSM::KEYSCALE_VALUE, base.keyscale);
        if (!base.vocal_language.empty()) fsm.force_field(bpe, MetadataFSM::LANGUAGE_VALUE, base.vocal_language);
        if (!base.timesignature.empty()) fsm.force_field(bpe, MetadataFSM::TIMESIG_VALUE, base.timesignature);
    }

    std::mt19937       rng(params.seed);
    std::vector<int>   gen_tokens;
    std::vector<float> lg;

    const bool timing       = lm_timing_enabled(params.verbose);
    double     prefill_ms   = 0.0;
    double     decode_ms    = 0.0;
    double     sampler_ms   = 0.0;
    int        decode_steps = 0;

    auto t_prefill = std::chrono::steady_clock::now();
    lm_reset(m, 0);
    if (!lm_model_forward(m, prompt_tokens.data(), (int) prompt_tokens.size(), lg, 0, nullptr, V_lim)) {
        fprintf(stderr, "[lm-phase1] prefill failed\n");
        return false;
    }
    prefill_ms = lm_ms_since(t_prefill);
    if (params.verbose)
        fprintf(stderr, "[lm-phase1] prefill %zu tokens, fsm=%d inspire=%d gen_lyrics=%d\n", prompt_tokens.size(),
                (int) use_fsm, (int) inspire, (int) gen_lyrics);

    bool codes_phase = false;
    int  tok;
    {
        auto      t_sample = std::chrono::steady_clock::now();
        const int forced   = (use_fsm && fsm.enabled) ? fsm.forced_token() : -1;
        if (forced >= 0) {
            tok = lm_consume_forced(forced, params.temperature, rng);
        } else {
            if (use_fsm && fsm.enabled) fsm.apply_mask(lg.data());
            tok = sample_top_k_p(lg.data(), V_eff, params.temperature, params.top_p, params.top_k, rng);
        }
        sampler_ms += lm_ms_since(t_sample);
        if (tok != TOKEN_IM_END) {
            if (use_fsm && fsm.enabled) fsm.update(tok);
            if (tok == TOKEN_THINK_END) codes_phase = true;
            gen_tokens.push_back(tok);
        }
    }

    const int max_new = params.max_new_tokens > 0 ? params.max_new_tokens : 2048;
    for (int step = 0; step < max_new && tok != TOKEN_IM_END; step++) {
        if (codes_phase && stop_at_reasoning) break;
        if (!lm_report_progress(params, step, max_new, LM_PROGRESS_UNKNOWN_TOTAL)) {
            fprintf(stderr, "[lm-phase1] cancelled at step %d\n", step);
            return false;
        }
        int32_t t        = (int32_t) tok;
        auto    t_decode = std::chrono::steady_clock::now();
        if (!lm_model_forward(m, &t, 1, lg, 0, nullptr, V_lim)) {
            fprintf(stderr, "[lm-phase1] decode step %d failed\n", step);
            return false;
        }
        decode_ms += lm_ms_since(t_decode);
        decode_steps++;
        auto      t_sample = std::chrono::steady_clock::now();
        float *   lc       = lg.data();
        const int forced   = (use_fsm && fsm.enabled && !codes_phase) ? fsm.forced_token() : -1;
        if (forced >= 0) {
            tok = lm_consume_forced(forced, params.temperature, rng);
        } else if (codes_phase && !lyrics_mode) {
            for (int v = TOKEN_IM_END + 1; v < AUDIO_CODE_BASE; v++) lc[v] = -1e9f;
            int V_eff = V - TOKEN_IM_END;
            tok       = sample_top_k_p(lc + TOKEN_IM_END, V_eff, params.temperature, params.top_p, params.top_k, rng) +
                  TOKEN_IM_END;
        } else {
            if (use_fsm && fsm.enabled && !codes_phase) fsm.apply_mask(lc);
            tok = sample_top_k_p(lc, V_eff, params.temperature, params.top_p, params.top_k, rng);
        }
        sampler_ms += lm_ms_since(t_sample);

        if (tok == TOKEN_IM_END) break;
        if (use_fsm && fsm.enabled && !codes_phase) fsm.update(tok);
        if (tok == TOKEN_THINK_END && !codes_phase) {
            codes_phase = true;
            gen_tokens.push_back(tok);
            if (stop_at_reasoning) break;
            continue;
        }
        gen_tokens.push_back(tok);
    }

    if (timing)
        fprintf(stderr, "[lm-timing] phase1 prefill=%.1f decode=%.1f sampler=%.1f steps=%d\n", prefill_ms, decode_ms,
                sampler_ms, decode_steps);

    std::string text = bpe_decode(bpe, gen_tokens);
    if (params.verbose)
        fprintf(stderr, "[lm-phase1] generated %zu tokens:\n%s\n", gen_tokens.size(), text.c_str());

    AcePrompt parsed = {};
    parse_cot_and_lyrics(text, parsed);

    // Gap-fill: only overwrite fields the user left empty.
    if (parsed.bpm > 0 && base.bpm <= 0)                              prompt.bpm = parsed.bpm;
    if (parsed.duration > 0 && base.duration <= 0)                    prompt.duration = parsed.duration;
    if (!parsed.keyscale.empty() && base.keyscale.empty())            prompt.keyscale = parsed.keyscale;
    if (!parsed.timesignature.empty() && base.timesignature.empty())  prompt.timesignature = parsed.timesignature;
    if (!parsed.vocal_language.empty() && (base.vocal_language.empty() || base.vocal_language == "unknown"))
        prompt.vocal_language = parsed.vocal_language;
    if (!parsed.caption.empty() && use_cot_caption)                   prompt.caption = parsed.caption;
    if (gen_lyrics && !parsed.lyrics.empty())                         prompt.lyrics = parsed.lyrics;
    if (prompt.duration <= 0)  prompt.duration = 120.0f;
    if (prompt.duration > 600) prompt.duration = 600.0f;

    if (params.verbose)
        fprintf(stderr, "[lm-phase1] filled: bpm=%d dur=%.0f key='%s' tsig='%s' lang='%s'\n", prompt.bpm,
                prompt.duration, prompt.keyscale.c_str(), prompt.timesignature.c_str(), prompt.vocal_language.c_str());
    return true;
}

bool lm_generate_codes(LMModel *              m,
                       const BpeTokenizer &   bpe,
                       const AcePrompt &      prompt,
                       const LmSampleParams & params,
                       std::vector<int> &     codes_out) {
    codes_out.clear();

    const LMConfig & cfg = lm_model_config(m);
    const int        V   = cfg.vocab_size;

    const bool use_cfg = params.cfg_scale > 1.0f && lm_num_kv_sets(m) >= 2;

    std::vector<int> tokens = build_lm_prompt_with_cot(bpe, prompt);
    std::vector<int> uncond;
    if (use_cfg) uncond = build_lm_prompt_uncond_with_cot(bpe, /*negative_prompt=*/"");

    if (const char * path = std::getenv("ACESTEP_LM_DUMP_TOKENS")) {
        if (FILE * f = fopen(path, "w")) {
            for (size_t i = 0; i < tokens.size(); i++) fprintf(f, "%s%d", i ? "," : "", tokens[i]);
            fprintf(f, "\n");
            fclose(f);
            fprintf(stderr, "[lm-dbg] wrote prompt tokens -> %s (%zu tokens)\n", path, tokens.size());
        }
    }

    int max_tokens = params.max_new_tokens > 0 ? params.max_new_tokens : (int) (prompt.duration * 5) + 100;

    std::mt19937 rng(params.seed);

    std::vector<float> lc, lu;  // cond / uncond logits

    const bool timing          = lm_timing_enabled(params.verbose);
    double     prefill_cond_ms   = 0.0;
    double     prefill_uncond_ms = 0.0;
    double     decode_ms         = 0.0;
    double     sampler_ms        = 0.0;
    int        decode_steps      = 0;

    const int out_v = V - TOKEN_IM_END;                // compact logit space: EOS at index 0
    const int code0 = AUDIO_CODE_BASE - TOKEN_IM_END;  // first audio code, compact index

    // CFG combine (cond + w*(cond-uncond)) then restrict to EOS + audio codes,
    // then sample - all in the compact [TOKEN_IM_END, V) logit space. Entries
    // below TOKEN_IM_END were only ever masked to -1e9, so index_base sampling
    // is exact - including the r==0 draw, which lands on absolute token 0.
    auto combine_mask_sample = [&](float * c, const float * u) -> int {
        if (use_cfg && u) {
            const float w = params.cfg_scale;
            c[0] = u[0] + w * (c[0] - u[0]);
            for (int v = code0; v < out_v; v++) c[v] = u[v] + w * (c[v] - u[v]);
        }
        for (int v = 1; v < code0; v++) c[v] = -1e9f;
        return sample_top_k_p(c, out_v, params.temperature, params.top_p, params.top_k, rng, TOKEN_IM_END);
    };

    const char *       dump_layers_path = std::getenv("ACESTEP_LM_DUMP_LAYERS");
    std::vector<float> layer_states;

    auto t_prefill = std::chrono::steady_clock::now();
    lm_reset(m, 0);
    if (!lm_model_forward(m, tokens.data(), (int) tokens.size(), lc, 0,
                          dump_layers_path ? &layer_states : nullptr)) {
        fprintf(stderr, "[lm-pipeline] cond prefill failed\n");
        return false;
    }
    prefill_cond_ms = lm_ms_since(t_prefill);
    if (const char * path = std::getenv("ACESTEP_LM_DUMP_LOGITS")) {
        if (FILE * f = fopen(path, "wb")) {
            fwrite(lc.data(), sizeof(float), lc.size(), f);
            fclose(f);
            fprintf(stderr, "[lm-dbg] wrote prefill logits -> %s (%zu floats)\n", path, lc.size());
        }
    }
    if (dump_layers_path && !layer_states.empty()) {
        const int32_t n_layers  = lm_model_config(m).n_layers;
        const int32_t per_layer = (int32_t) (layer_states.size() / (size_t) n_layers);
        const int32_t hdr[3]    = { 2, n_layers, per_layer };
        if (FILE * f = fopen(dump_layers_path, "wb")) {
            fwrite(hdr, sizeof(hdr), 1, f);
            fwrite(layer_states.data(), sizeof(float), layer_states.size(), f);
            fclose(f);
            fprintf(stderr, "[lm-dbg] wrote layer states -> %s (%zu floats)\n", dump_layers_path,
                    layer_states.size());
        }
    }
    if (use_cfg) {
        t_prefill = std::chrono::steady_clock::now();
        lm_reset(m, 1);
        if (!lm_model_forward(m, uncond.data(), (int) uncond.size(), lu, 1)) {
            fprintf(stderr, "[lm-pipeline] uncond prefill failed\n");
            return false;
        }
        prefill_uncond_ms = lm_ms_since(t_prefill);
    }
    const bool batch_cfg = use_cfg && lm_model_supports_batched_decode(m);
    if (params.verbose)
        fprintf(stderr,
                "[lm-pipeline] prefill cond=%zu uncond=%zu, max_new=%d, temp=%.2f top_p=%.2f top_k=%d cfg=%.2f%s%s\n",
                tokens.size(), uncond.size(), max_tokens, params.temperature, params.top_p, params.top_k,
                use_cfg ? params.cfg_scale : 1.0f, use_cfg ? "" : " (CFG off)",
                batch_cfg ? " (batched decode)" : "");

    auto t_sample = std::chrono::steady_clock::now();
    int  tok = combine_mask_sample(lc.data() + TOKEN_IM_END, use_cfg ? lu.data() + TOKEN_IM_END : nullptr);
    sampler_ms += lm_ms_since(t_sample);
    if (tok != TOKEN_IM_END && tok >= AUDIO_CODE_BASE && tok < AUDIO_CODE_BASE + AUDIO_CODE_COUNT) {
        codes_out.push_back(tok - AUDIO_CODE_BASE);
    }

    std::vector<float> batched;  // compact [TOKEN_IM_END, V) logits x {cond, uncond}
    for (int step = 0; step < max_tokens && tok != TOKEN_IM_END; step++) {
        int32_t       t        = (int32_t) tok;
        auto          t_decode = std::chrono::steady_clock::now();
        float *       samp_c   = nullptr;
        const float * samp_u   = nullptr;
        if (batch_cfg) {
            const int32_t ids[2] = { t, t };
            const int     sets[2] = { 0, 1 };
            if (!lm_model_forward_batch(m, ids, sets, 2, batched, TOKEN_IM_END)) {
                fprintf(stderr, "[lm-pipeline] batched CFG decode step %d failed\n", step);
                return false;
            }
            samp_c = batched.data();
            samp_u = batched.data() + out_v;
        } else {
            if (!lm_model_forward(m, &t, 1, lc, 0)) {
                fprintf(stderr, "[lm-pipeline] cond decode step %d failed\n", step);
                return false;
            }
            if (use_cfg && !lm_model_forward(m, &t, 1, lu, 1)) {
                fprintf(stderr, "[lm-pipeline] uncond decode step %d failed\n", step);
                return false;
            }
            samp_c = lc.data() + TOKEN_IM_END;
            samp_u = use_cfg ? lu.data() + TOKEN_IM_END : nullptr;
        }
        decode_ms += lm_ms_since(t_decode);
        decode_steps++;
        t_sample = std::chrono::steady_clock::now();
        tok      = combine_mask_sample(samp_c, samp_u);
        sampler_ms += lm_ms_since(t_sample);
        if (tok == TOKEN_IM_END) break;
        if (tok >= AUDIO_CODE_BASE && tok < AUDIO_CODE_BASE + AUDIO_CODE_COUNT) {
            codes_out.push_back(tok - AUDIO_CODE_BASE);
        }
        if (params.verbose && (step + 1) % 100 == 0) {
            fprintf(stderr, "[lm-pipeline]   step %d, %zu codes\n", step + 1, codes_out.size());
        }
        if (!lm_report_progress(params, step, max_tokens, max_tokens)) {
            fprintf(stderr, "[lm-pipeline] cancelled at step %d, %zu codes\n", step, codes_out.size());
            return false;
        }
    }

    if (timing)
        fprintf(stderr, "[lm-timing] phase2 prefill-cond=%.1f prefill-uncond=%.1f decode=%.1f sampler=%.1f steps=%d\n",
                prefill_cond_ms, prefill_uncond_ms, decode_ms, sampler_ms, decode_steps);

    if (params.verbose) {
        fprintf(stderr, "[lm-pipeline] generated %zu audio codes\n", codes_out.size());
        std::string csv;
        for (size_t i = 0; i < codes_out.size(); i++) {
            if (i) csv += ',';
            csv += std::to_string(codes_out[i]);
        }
        fprintf(stderr, "[lm-pipeline] codes: %s\n", csv.c_str());
    }
    return true;
}

std::vector<int> lm_understand_prompt(const BpeTokenizer & bpe, const int * codes, int n_codes) {
    std::vector<int> ids;
    auto             append = [&](const std::string & text) {
        auto t = bpe_encode(bpe, text, false);
        ids.insert(ids.end(), t.begin(), t.end());
    };
    ids.push_back(TOKEN_IM_START);
    append(std::string("system\n# Instruction\n") + LM_UNDERSTAND_INSTRUCTION + "\n\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    append("user\n");
    for (int i = 0; i < n_codes; i++) ids.push_back(AUDIO_CODE_BASE + codes[i]);
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    append("assistant\n");
    return ids;
}

std::vector<int> lm_understand_unconditional_prompt(const BpeTokenizer & bpe) {
    // The trailing newline matters: the reference's system text ends "\n\n"
    // before <|im_end|>, and the conditional understand prompt does too. A
    // single "\n" here would silently shift the quality-score baseline.
    return build_custom_prompt(bpe, std::string("# Instruction\n") + LM_UNDERSTAND_INSTRUCTION + "\n",
                               "NO USER INPUT");
}

static int lm_understand_sample(MetadataFSM & fsm, float * logits, int V,
                                const LmSampleParams & params, std::mt19937 & rng) {
    const int forced = fsm.enabled ? fsm.forced_token() : -1;
    if (forced >= 0) return lm_consume_forced(forced, params.temperature, rng);
    if (fsm.enabled) fsm.apply_mask(logits);
    return sample_top_k_p(logits, V, params.temperature, params.top_p, params.top_k, rng);
}

int lm_understand_token_budget(int max_seq_len, size_t prompt_tokens, int requested) {
    const int room = max_seq_len - (int) prompt_tokens;
    if (room <= 0) return 0;
    const int wanted = requested > 0 ? requested : LM_UNDERSTAND_DEFAULT_MAX_NEW;
    return wanted < room ? wanted : room;
}

bool lm_understand(LMModel * m, const BpeTokenizer & bpe, const std::vector<int> & codes,
                   const LmSampleParams & params, const std::string & language_hint,
                   AcePrompt & out) {
    out = {};
    if (!m || codes.empty()) return false;

    const int V = lm_model_config(m).vocab_size;

    MetadataFSM fsm;
    fsm.init(bpe, V, params.verbose);
    if (!language_hint.empty() && language_hint != "unknown") {
        fsm.force_field(bpe, MetadataFSM::LANGUAGE_VALUE, language_hint);
    }

    const std::vector<int>     prompt = lm_understand_prompt(bpe, codes.data(), (int) codes.size());
    const std::vector<int32_t> prompt32(prompt.begin(), prompt.end());
    if (prompt.size() + 1 > (size_t) lm_model_config(m).max_seq_len) {
        fprintf(stderr, "[lm-understand] prompt needs %zu tokens, exceeding LM max_seq %d\n",
                prompt.size(), lm_model_config(m).max_seq_len);
        return false;
    }

    std::vector<float> logits;
    lm_reset(m, 0);
    if (!lm_model_forward(m, prompt32.data(), (int) prompt32.size(), logits)) {
        fprintf(stderr, "[lm-understand] prefill failed\n");
        return false;
    }
    if (params.verbose) {
        fprintf(stderr, "[lm-understand] prefill %zu tokens (%zu codes + framing)\n", prompt.size(),
                codes.size());
    }

    std::mt19937     rng(params.seed);
    std::vector<int> gen_tokens;
    // The description lives entirely inside the CoT block (caption included),
    // so decoding stops at </think>: the lyric text the reference generates
    // afterwards is unused here and would only burn KV budget. The budget is
    // capped to the KV room left after the prompt — a long clip's code prompt
    // can leave far less than the default.
    const int max_new = lm_understand_token_budget(lm_model_config(m).max_seq_len, prompt.size(),
                                                   params.max_new_tokens);

    for (int step = 0; step < max_new; step++) {
        if (!lm_report_progress(params, step, max_new, LM_PROGRESS_UNKNOWN_TOTAL)) {
            fprintf(stderr, "[lm-understand] cancelled at step %d\n", step);
            return false;
        }
        const int tok = lm_understand_sample(fsm, logits.data(), V, params, rng);
        if (tok == TOKEN_IM_END) break;
        if (fsm.enabled) fsm.update(tok);
        gen_tokens.push_back(tok);
        if (tok == TOKEN_THINK_END) break;

        const int32_t t = (int32_t) tok;
        if (!lm_model_forward(m, &t, 1, logits)) {
            fprintf(stderr, "[lm-understand] decode step %d failed\n", step);
            return false;
        }
    }

    const std::string text = bpe_decode(bpe, gen_tokens);
    if (params.verbose) {
        fprintf(stderr, "[lm-understand] generated %zu tokens:\n%s\n", gen_tokens.size(), text.c_str());
    }
    parse_cot_and_lyrics(text, out);
    if (!language_hint.empty() && language_hint != "unknown") {
        out.vocal_language = language_hint;
    }
    return true;
}

} // namespace tts_cpp::acestep
