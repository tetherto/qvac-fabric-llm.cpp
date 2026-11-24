// QVAC test: pins the default-parameter contract of the public API. An
// upstream sync that silently changes a decode default alters transcripts
// without any code in this repo changing — exactly the class of drift the
// divergence guard cannot see. Model-free.

#include "whisper.h"

#include <cfloat>
#include <cstdio>
#include <cstring>

static int g_failures = 0;

static void expect(bool ok, const char * what) {
    if (ok) {
        return;
    }
    fprintf(stderr, "FAIL: %s\n", what);
    g_failures++;
}

static void check_shared_full_defaults(const whisper_full_params & p) {
    expect(p.n_threads >= 1 && p.n_threads <= 4, "n_threads in [1,4]");
    expect(p.n_max_text_ctx == 16384, "n_max_text_ctx == 16384");
    expect(p.offset_ms == 0, "offset_ms == 0");
    expect(p.duration_ms == 0, "duration_ms == 0");
    expect(!p.translate, "translate off");
    expect(p.no_context, "no_context on");
    expect(!p.no_timestamps, "timestamps on");
    expect(!p.single_segment, "single_segment off");
    expect(!p.token_timestamps, "token_timestamps off");
    expect(p.thold_pt == 0.01f, "thold_pt == 0.01");
    expect(p.thold_ptsum == 0.01f, "thold_ptsum == 0.01");
    expect(p.max_len == 0, "max_len == 0");
    expect(!p.split_on_word, "split_on_word off");
    expect(p.max_tokens == 0, "max_tokens == 0");
    expect(p.audio_ctx == 0, "audio_ctx == 0");
    expect(!p.tdrz_enable, "tdrz off");
    expect(p.suppress_regex == nullptr, "suppress_regex null");
    expect(p.initial_prompt == nullptr, "initial_prompt null");
    expect(!p.carry_initial_prompt, "carry_initial_prompt off");
    expect(strcmp(p.language, "en") == 0, "language == en");
    expect(!p.detect_language, "detect_language off");
    expect(p.suppress_blank, "suppress_blank on");
    expect(!p.suppress_nst, "suppress_nst off");
    expect(p.temperature == 0.0f, "temperature == 0");
    expect(p.max_initial_ts == 1.0f, "max_initial_ts == 1");
    expect(p.length_penalty == -1.0f, "length_penalty == -1");
    expect(p.temperature_inc == 0.2f, "temperature_inc == 0.2");
    expect(p.entropy_thold == 2.4f, "entropy_thold == 2.4");
    expect(p.logprob_thold == -1.0f, "logprob_thold == -1");
    expect(p.no_speech_thold == 0.6f, "no_speech_thold == 0.6");
    expect(p.grammar_rules == nullptr, "grammar_rules null");
    expect(p.grammar_penalty == 100.0f, "grammar_penalty == 100");
    expect(!p.vad, "vad off by default");
    expect(p.vad_model_path == nullptr, "vad_model_path null");
}

static void check_greedy_defaults() {
    const whisper_full_params p =
        whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    expect(p.strategy == WHISPER_SAMPLING_GREEDY, "greedy strategy kept");
    check_shared_full_defaults(p);
    expect(p.greedy.best_of == 5, "greedy best_of == 5");
    expect(p.beam_search.beam_size == -1, "greedy leaves beam_size unset");
}

static void check_beam_defaults() {
    const whisper_full_params p =
        whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH);
    expect(p.strategy == WHISPER_SAMPLING_BEAM_SEARCH, "beam strategy kept");
    check_shared_full_defaults(p);
    expect(p.beam_search.beam_size == 5, "beam beam_size == 5");
    expect(p.beam_search.patience == -1.0f, "beam patience == -1");
    expect(p.greedy.best_of == -1, "beam leaves best_of unset");
}

static void check_context_defaults() {
    const whisper_context_params p = whisper_context_default_params();
    expect(p.use_gpu, "use_gpu on");
    expect(p.flash_attn, "flash_attn on");
    expect(p.gpu_device == 0, "gpu_device == 0");
    expect(!p.dtw_token_timestamps, "dtw_token_timestamps off");
    expect(p.dtw_aheads_preset == WHISPER_AHEADS_NONE, "dtw preset none");
    expect(p.dtw_n_top == -1, "dtw_n_top == -1");
    expect(p.dtw_mem_size == 1024 * 1024 * 128, "dtw_mem_size == 128 MiB");
}

static void check_vad_defaults() {
    const whisper_vad_params p = whisper_vad_default_params();
    expect(p.threshold == 0.5f, "vad threshold == 0.5");
    expect(p.min_speech_duration_ms == 250, "vad min_speech == 250 ms");
    expect(p.min_silence_duration_ms == 100, "vad min_silence == 100 ms");
    expect(p.max_speech_duration_s == FLT_MAX, "vad max_speech unbounded");
    expect(p.speech_pad_ms == 30, "vad speech_pad == 30 ms");
    expect(p.samples_overlap == 0.1f, "vad samples_overlap == 0.1");
}

static void check_by_ref_matches_by_value() {
    whisper_full_params * ref =
        whisper_full_default_params_by_ref(WHISPER_SAMPLING_GREEDY);
    expect(ref != nullptr, "by_ref returns a params block");
    if (ref != nullptr) {
        const whisper_full_params val =
            whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        expect(ref->strategy == val.strategy &&
               ref->greedy.best_of == val.greedy.best_of &&
               ref->temperature_inc == val.temperature_inc &&
               strcmp(ref->language, val.language) == 0,
               "by_ref matches by-value defaults");
        whisper_free_params(ref);
    }
}

int main() {
    check_greedy_defaults();
    check_beam_defaults();
    check_context_defaults();
    check_vad_defaults();
    check_by_ref_matches_by_value();

    if (g_failures > 0) {
        fprintf(stderr, "test-whisper-params: %d failure(s)\n", g_failures);
        return 1;
    }
    printf("test-whisper-params: all checks passed\n");
    return 0;
}
