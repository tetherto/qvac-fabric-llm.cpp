#pragma once

#include "audiogen-cpp/acestep/engine.h"

#include <algorithm>
#include <cmath>
#include <string>

namespace tts_cpp::acestep {

inline constexpr const char * TASK_TEXT2MUSIC  = "text2music";
inline constexpr const char * TASK_COVER       = "cover";
inline constexpr const char * TASK_COVER_NOFSQ = "cover-nofsq";
inline constexpr const char * TASK_LEGO        = "lego";

inline constexpr const char * LEGO_TRACK_NAMES[] = {
    "vocals",     "backing_vocals", "drums", "bass", "guitar", "keyboard",
    "percussion", "strings",        "synth", "fx",   "brass",  "woodwinds",
};

inline bool is_cover_task(const std::string & task) {
    return task == TASK_COVER || task == TASK_COVER_NOFSQ;
}

inline bool is_lego_task(const std::string & task) {
    return task == TASK_LEGO;
}

inline bool is_source_task(const std::string & task) {
    return is_cover_task(task) || is_lego_task(task);
}

inline bool is_valid_lego_track(const std::string & track) {
    for (const char * name : LEGO_TRACK_NAMES) {
        if (track == name) return true;
    }
    return false;
}

struct GenerateTask {
    std::string type;
    std::string track;
    float       audio_cover_strength = 1.0f;
    float       cover_noise_strength = 0.0f;
};

inline float clamp_strength(float value) {
    return std::clamp(value, 0.0f, 1.0f);
}

inline constexpr const char * DEFAULT_VOCAL_LANGUAGE = "en";
inline constexpr const char * EDIT_VOCAL_LANGUAGE    = "unknown";
inline constexpr const char * INSTRUMENTAL_LYRICS    = "[Instrumental]";

// Simple mode and query rewriting keep an unset language empty so the LM
// expansion pass picks it; otherwise the engine defaults apply before the
// prompt is built, with the neutral language for the edit path and lego (a
// single language token skews 50-step CFG sampling toward vocals).
inline std::string resolve_prompt_language(const GenerateParams & params) {
    if (!params.vocal_language.empty()) return params.vocal_language;
    if (params.simple_mode || params.rewrite_query) return {};
    const bool language_neutral = !params.edit_plan.empty() || is_lego_task(params.task_type);
    return language_neutral ? EDIT_VOCAL_LANGUAGE : DEFAULT_VOCAL_LANGUAGE;
}

// Simple mode keeps unset lyrics empty so the LM inspire pass writes them
// (the explicit "[Instrumental]" request still flows through as the hint).
inline std::string resolve_prompt_lyrics(const GenerateParams & params) {
    if (!params.lyrics.empty()) return params.lyrics;
    if (params.simple_mode) return {};
    return INSTRUMENTAL_LYRICS;
}

inline std::string validate_simple_mode(const GenerateParams & params, const std::string & task_type) {
    if (!params.simple_mode) return {};
    if (params.caption.empty()) {
        return "acestep engine: simple_mode requires a caption query";
    }
    if (task_type != TASK_TEXT2MUSIC) {
        return "acestep engine: simple_mode supports only task 'text2music', got '" + task_type + "'";
    }
    if (!params.audio_codes.empty()) {
        return "acestep engine: simple_mode cannot take pre-supplied audio_codes";
    }
    if (!params.lyrics.empty() && params.lyrics != AUDIO_EDIT_DEFAULT_LYRICS) {
        return "acestep engine: simple_mode lyrics must be empty (LM writes them) or \"[Instrumental]\"";
    }
    if (!params.edit_plan.empty()) {
        return "acestep engine: simple_mode cannot be combined with edit_plan";
    }
    if (!params.lm_phase1) {
        return "acestep engine: simple_mode requires lm_phase1";
    }
    return {};
}

inline constexpr float TURBO_GUIDANCE_SCALE    = 1.0f;
inline constexpr float STANDARD_GUIDANCE_SCALE = 7.0f;

inline std::string lego_model_error(bool is_turbo, bool is_sft) {
    if (is_turbo) {
        return "acestep engine: task 'lego' requires a base DiT (turbo does not support stem tasks)";
    }
    if (is_sft) {
        return "acestep engine: task 'lego' requires a base DiT (sft does not support stem tasks)";
    }
    return {};
}

// Turbo is guidance-distilled: CFG is untrained there, so explicit overrides
// clamp to 1.0. Base/sft default to the official 7.0 when unset.
inline float resolve_guidance_scale(float requested, bool is_turbo) {
    if (is_turbo) return TURBO_GUIDANCE_SCALE;
    return requested > 0.0f ? requested : STANDARD_GUIDANCE_SCALE;
}

// Haar DCW is a turbo-preset correction; the official base/sft preset disables it.
inline bool resolve_dcw_enabled(bool requested, bool is_turbo) {
    return requested && is_turbo;
}

inline std::string validate_lego_track(const std::string & track) {
    if (track.empty()) {
        return "acestep engine: task 'lego' requires a track name";
    }
    if (!is_valid_lego_track(track)) {
        return "acestep engine: unknown lego track '" + track +
               "' (expected one of vocals|backing_vocals|drums|bass|guitar|keyboard|"
               "percussion|strings|synth|fx|brass|woodwinds)";
    }
    return {};
}

inline std::string validate_lrc_request(const GenerateParams & params) {
    if (!params.generate_lrc) return {};
    if (!params.edit_plan.empty()) {
        return "acestep engine: generate_lrc is unavailable on the audio edit path";
    }
    if (params.simple_mode) return {};
    if (params.lyrics.empty() || params.lyrics == INSTRUMENTAL_LYRICS) {
        return "acestep engine: generate_lrc requires lyrics to align";
    }
    return {};
}

inline std::string validate_rewrite_query(const GenerateParams & params, const std::string & task_type) {
    if (!params.rewrite_query) return {};
    if (params.simple_mode) {
        return "acestep engine: rewrite_query cannot be combined with simple_mode";
    }
    if (params.caption.empty()) {
        return "acestep engine: rewrite_query requires a caption";
    }
    if (params.lyrics.empty()) {
        return "acestep engine: rewrite_query requires lyrics to format (use simple_mode for a bare query)";
    }
    if (params.lyrics == INSTRUMENTAL_LYRICS) {
        return "acestep engine: rewrite_query requires lyric text to preserve (use simple_mode with "
               "\"[Instrumental]\" lyrics for an instrumental request)";
    }
    if (task_type != TASK_TEXT2MUSIC) {
        return "acestep engine: rewrite_query supports only task 'text2music', got '" + task_type + "'";
    }
    if (!params.audio_codes.empty()) {
        return "acestep engine: rewrite_query cannot take pre-supplied audio_codes";
    }
    if (!params.edit_plan.empty()) {
        return "acestep engine: rewrite_query cannot be combined with edit_plan";
    }
    if (!params.lm_phase1) {
        return "acestep engine: rewrite_query requires lm_phase1";
    }
    return {};
}

inline std::string validate_quality_score_request(const GenerateParams & params, const std::string & task_type) {
    if (!params.compute_quality_score) return {};
    if (!params.edit_plan.empty()) {
        return "acestep engine: compute_quality_score is unavailable on the audio edit path";
    }
    if (is_source_task(task_type)) {
        return "acestep engine: compute_quality_score requires the LM code path, unavailable on task '" +
               task_type + "'";
    }
    return {};
}

inline std::string resolve_generate_task(const GenerateParams & params, GenerateTask & task) {
    task.type = params.task_type.empty() ? TASK_TEXT2MUSIC : params.task_type;

    if (task.type != TASK_TEXT2MUSIC && task.type != TASK_COVER &&
        task.type != TASK_COVER_NOFSQ && task.type != TASK_LEGO) {
        return "acestep engine: unsupported task_type '" + task.type +
               "' (expected text2music|cover|cover-nofsq|lego)";
    }

    const std::string rewrite_error = validate_rewrite_query(params, task.type);
    if (!rewrite_error.empty()) return rewrite_error;

    const std::string simple_mode_error = validate_simple_mode(params, task.type);
    if (!simple_mode_error.empty()) return simple_mode_error;

    const std::string lrc_error = validate_lrc_request(params);
    if (!lrc_error.empty()) return lrc_error;

    const std::string quality_score_error = validate_quality_score_request(params, task.type);
    if (!quality_score_error.empty()) return quality_score_error;

    if (!std::isfinite(params.audio_cover_strength)) {
        return "acestep engine: audio_cover_strength must be finite";
    }
    if (!std::isfinite(params.cover_noise_strength)) {
        return "acestep engine: cover_noise_strength must be finite";
    }

    task.track                = params.track;
    task.audio_cover_strength = clamp_strength(params.audio_cover_strength);
    task.cover_noise_strength = clamp_strength(params.cover_noise_strength);

    if (!params.reference_audio.empty() && (params.reference_audio.size() & 1u) != 0) {
        return "acestep engine: reference_audio must be interleaved stereo";
    }

    if (!is_source_task(task.type)) return {};

    if (params.source_audio.empty()) {
        return "acestep engine: task '" + task.type + "' requires source_audio";
    }
    if ((params.source_audio.size() & 1u) != 0) {
        return "acestep engine: source_audio must be interleaved stereo";
    }
    if (is_lego_task(task.type)) {
        return validate_lego_track(task.track);
    }
    if (task.type == TASK_COVER) {
        return "acestep engine: task 'cover' is not implemented yet (needs FSQ tokenizer); use cover-nofsq";
    }
    return {};
}

inline bool needs_cover_conditioning_switch(const GenerateTask & task) {
    return is_cover_task(task.type) && task.audio_cover_strength < 1.0f;
}

// Step index where the DiT drops the source conditioning; -1 disables the switch.
inline int resolve_cover_switch_step(const GenerateTask & task, int num_steps) {
    if (!needs_cover_conditioning_switch(task)) return -1;
    return (int) ((float) num_steps * task.audio_cover_strength);
}

} // namespace tts_cpp::acestep
