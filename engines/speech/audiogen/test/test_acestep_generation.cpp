// Model-free ACE-Step generation regression tests.

#include "acestep_test_support.h"
#include "audiogen-cpp/acestep/engine.h"
#include "generate_task.h"
#include "dit_ggml.h"
#include "generation_plan.h"
#include "generation_conditioning.h"
#include "audio_edit.h"
#include "cover_noise.h"
#include "cancellation_scope.h"
#include "lm_pipeline.h"

namespace {

void test_generate_task_kinds() {
    using tts_cpp::acestep::TASK_COVER;
    using tts_cpp::acestep::TASK_COVER_NOFSQ;
    using tts_cpp::acestep::TASK_TEXT2MUSIC;
    using tts_cpp::acestep::is_cover_task;

    CHECK(is_cover_task(TASK_COVER));
    CHECK(is_cover_task(TASK_COVER_NOFSQ));
    CHECK(!is_cover_task(TASK_TEXT2MUSIC));
    CHECK(!is_cover_task(""));
}

void test_generate_task_defaults() {
    using tts_cpp::acestep::GenerateParams;
    using tts_cpp::acestep::TASK_TEXT2MUSIC;
    using tts_cpp::acestep::GenerateTask;
    using tts_cpp::acestep::resolve_generate_task;

    GenerateParams params;
    params.source_audio.assign(8, 0.25f);
    params.reference_audio.assign(8, 0.5f);
    const float * source_data = params.source_audio.data();
    const float * reference_data = params.reference_audio.data();
    GenerateTask task;
    CHECK(resolve_generate_task(params, task).empty());
    CHECK(task.type == TASK_TEXT2MUSIC);
    CHECK(approx(task.audio_cover_strength, 1.0f));
    CHECK(approx(task.cover_noise_strength, 0.0f));
    CHECK(params.source_audio.data() == source_data);
    CHECK(params.reference_audio.data() == reference_data);
}

void test_generate_task_audio_layout() {
    using tts_cpp::acestep::GenerateParams;
    using tts_cpp::acestep::TASK_COVER_NOFSQ;
    using tts_cpp::acestep::GenerateTask;
    using tts_cpp::acestep::resolve_generate_task;

    GenerateParams params;
    GenerateTask task;

    params.task_type = TASK_COVER_NOFSQ;
    CHECK(resolve_generate_task(params, task).find("requires source_audio") != std::string::npos);

    params.source_audio.assign(3, 0.0f);
    CHECK(resolve_generate_task(params, task).find("source_audio must be interleaved stereo") != std::string::npos);

    params.source_audio.assign(4, 0.0f);
    params.reference_audio.assign(5, 0.0f);
    CHECK(resolve_generate_task(params, task).find("reference_audio must be interleaved stereo") != std::string::npos);

    params.reference_audio.assign(6, 0.0f);
    CHECK(resolve_generate_task(params, task).empty());
}

void test_generate_task_errors() {
    using tts_cpp::acestep::GenerateParams;
    using tts_cpp::acestep::TASK_COVER;
    using tts_cpp::acestep::TASK_COVER_NOFSQ;
    using tts_cpp::acestep::GenerateTask;
    using tts_cpp::acestep::resolve_generate_task;

    GenerateParams params;
    GenerateTask task;

    params.task_type = "repaint";
    CHECK(resolve_generate_task(params, task).find("unsupported task_type") != std::string::npos);

    params.task_type = TASK_COVER;
    params.source_audio.assign(4, 0.0f);
    CHECK(resolve_generate_task(params, task).find("not implemented") != std::string::npos);

    params.task_type = TASK_COVER_NOFSQ;
    params.audio_cover_strength = 0.5f;
    CHECK(resolve_generate_task(params, task).empty());
    CHECK(approx(task.audio_cover_strength, 0.5f));
}

void test_simple_mode_policy() {
    using tts_cpp::acestep::GenerateParams;
    using tts_cpp::acestep::TASK_COVER_NOFSQ;
    using tts_cpp::acestep::GenerateTask;
    using tts_cpp::acestep::resolve_generate_task;

    GenerateParams params;
    GenerateTask task;
    params.simple_mode = true;
    params.caption = "a romantic modern salsa with male lead vocals for a wedding";
    params.lyrics.clear();
    CHECK(resolve_generate_task(params, task).empty());

    params.lyrics = "[Instrumental]";
    CHECK(resolve_generate_task(params, task).empty());

    params.lyrics = "[verse]\nuser lyrics";
    CHECK(resolve_generate_task(params, task).find("lyrics must be empty") != std::string::npos);

    params.lyrics.clear();
    params.caption.clear();
    CHECK(resolve_generate_task(params, task).find("requires a caption") != std::string::npos);

    params.caption = "a short query";
    params.task_type = TASK_COVER_NOFSQ;
    params.source_audio.assign(4, 0.0f);
    CHECK(resolve_generate_task(params, task).find("only task 'text2music'") != std::string::npos);

    params.task_type.clear();
    params.source_audio.clear();
    params.audio_codes.assign(4, 1);
    CHECK(resolve_generate_task(params, task).find("pre-supplied audio_codes") != std::string::npos);

    params.audio_codes.clear();
    params.edit_plan.push_back(tts_cpp::acestep::RepaintParams{});
    CHECK(resolve_generate_task(params, task).find("combined with edit_plan") != std::string::npos);

    params.edit_plan.clear();
    params.lm_phase1 = false;
    CHECK(resolve_generate_task(params, task).find("requires lm_phase1") != std::string::npos);

    params.lm_phase1 = true;
    params.simple_mode = false;
    params.lyrics = "[verse]\nuser lyrics";
    CHECK(resolve_generate_task(params, task).empty());
}

void test_simple_mode_prompt_resolvers() {
    using tts_cpp::acestep::GenerateParams;
    using tts_cpp::acestep::RepaintParams;
    using tts_cpp::acestep::resolve_prompt_language;
    using tts_cpp::acestep::resolve_prompt_lyrics;

    GenerateParams params;
    params.lyrics.clear();
    CHECK(resolve_prompt_lyrics(params) == "[Instrumental]");
    CHECK(resolve_prompt_language(params) == "en");

    params.simple_mode = true;
    CHECK(resolve_prompt_lyrics(params).empty());
    CHECK(resolve_prompt_language(params).empty());

    params.simple_mode   = false;
    params.rewrite_query = true;
    CHECK(resolve_prompt_language(params).empty());
    params.rewrite_query = false;

    params.simple_mode = true;
    params.lyrics = "[Instrumental]";
    params.vocal_language = "es";
    CHECK(resolve_prompt_lyrics(params) == "[Instrumental]");
    CHECK(resolve_prompt_language(params) == "es");

    params.simple_mode = false;
    params.lyrics.clear();
    params.vocal_language.clear();
    params.edit_plan.push_back(RepaintParams{});
    CHECK(resolve_prompt_language(params) == "unknown");

    params.edit_plan.clear();
    params.task_type = tts_cpp::acestep::TASK_LEGO;
    CHECK(resolve_prompt_language(params) == "unknown");
}

void test_lrc_request_policy() {
    using tts_cpp::acestep::GenerateParams;
    using tts_cpp::acestep::GenerateTask;
    using tts_cpp::acestep::RepaintParams;
    using tts_cpp::acestep::resolve_generate_task;

    GenerateParams params;
    GenerateTask task;
    params.generate_lrc = true;
    params.lyrics = "[verse]\na line to align";
    CHECK(resolve_generate_task(params, task).empty());

    params.lyrics = "[Instrumental]";
    CHECK(resolve_generate_task(params, task).find("requires lyrics") != std::string::npos);

    params.lyrics.clear();
    CHECK(resolve_generate_task(params, task).find("requires lyrics") != std::string::npos);

    params.simple_mode = true;
    params.caption = "a short query";
    CHECK(resolve_generate_task(params, task).empty());

    params.simple_mode = false;
    params.lyrics = "[verse]\na line to align";
    params.edit_plan.push_back(RepaintParams{});
    CHECK(resolve_generate_task(params, task).find("audio edit path") != std::string::npos);
}

void test_format_user_message() {
    using tts_cpp::acestep::lm_format_user_message;

    CHECK(lm_format_user_message("a salsa song", "[verse]\nhello") ==
          "# Caption\na salsa song\n\n# Lyric\n[verse]\nhello");
}

void test_rewrite_query_policy() {
    using tts_cpp::acestep::GenerateParams;
    using tts_cpp::acestep::GenerateTask;
    using tts_cpp::acestep::RepaintParams;
    using tts_cpp::acestep::resolve_generate_task;

    GenerateParams params;
    GenerateTask   task;
    params.rewrite_query = true;
    params.caption       = "a salsa song";
    params.lyrics        = "[verse]\nhello";
    CHECK(resolve_generate_task(params, task).empty());

    params.simple_mode = true;
    CHECK(resolve_generate_task(params, task).find("cannot be combined with simple_mode") !=
          std::string::npos);
    params.simple_mode = false;

    params.caption.clear();
    CHECK(resolve_generate_task(params, task).find("requires a caption") != std::string::npos);
    params.caption = "a salsa song";

    params.lyrics.clear();
    CHECK(resolve_generate_task(params, task).find("requires lyrics to format") != std::string::npos);

    params.lyrics = "[Instrumental]";
    CHECK(resolve_generate_task(params, task).find("requires lyric text to preserve") !=
          std::string::npos);
    params.lyrics = "[verse]\nhello";

    params.task_type = tts_cpp::acestep::TASK_COVER_NOFSQ;
    params.source_audio.assign(96, 0.0f);
    CHECK(resolve_generate_task(params, task).find("only task 'text2music'") != std::string::npos);
    params.task_type = tts_cpp::acestep::TASK_TEXT2MUSIC;
    params.source_audio.clear();

    params.audio_codes.assign(4, 1);
    CHECK(resolve_generate_task(params, task).find("pre-supplied audio_codes") != std::string::npos);
    params.audio_codes.clear();

    params.edit_plan.push_back(RepaintParams{});
    CHECK(resolve_generate_task(params, task).find("edit_plan") != std::string::npos);
    params.edit_plan.clear();

    params.lm_phase1 = false;
    CHECK(resolve_generate_task(params, task).find("requires lm_phase1") != std::string::npos);
}

void test_inspire_user_message() {
    using tts_cpp::acestep::lm_inspire_user_message;

    CHECK(lm_inspire_user_message("a short query", "") == "a short query");
    CHECK(lm_inspire_user_message("a short query", "[Instrumental]") ==
          "a short query\n\ninstrumental: true");
    CHECK(lm_inspire_user_message("a short query", "[verse]\nwords") == "a short query");
}

void test_cover_conditioning_switch() {
    using tts_cpp::acestep::GenerateTask;
    using tts_cpp::acestep::TASK_COVER_NOFSQ;
    using tts_cpp::acestep::TASK_TEXT2MUSIC;
    using tts_cpp::acestep::needs_cover_conditioning_switch;
    using tts_cpp::acestep::resolve_cover_switch_step;

    GenerateTask task;
    task.type = TASK_COVER_NOFSQ;
    task.audio_cover_strength = 1.0f;
    CHECK(!needs_cover_conditioning_switch(task));
    CHECK(resolve_cover_switch_step(task, 50) == -1);

    task.audio_cover_strength = 0.5f;
    CHECK(needs_cover_conditioning_switch(task));
    CHECK(resolve_cover_switch_step(task, 50) == 25);
    CHECK(resolve_cover_switch_step(task, 8) == 4);

    task.audio_cover_strength = 0.75f;
    CHECK(resolve_cover_switch_step(task, 8) == 6);

    task.audio_cover_strength = 0.0f;
    CHECK(resolve_cover_switch_step(task, 8) == 0);

    task.type = TASK_TEXT2MUSIC;
    task.audio_cover_strength = 0.5f;
    CHECK(!needs_cover_conditioning_switch(task));
    CHECK(resolve_cover_switch_step(task, 50) == -1);
}

void test_generate_task_strengths() {
    using tts_cpp::acestep::GenerateParams;
    using tts_cpp::acestep::GenerateTask;
    using tts_cpp::acestep::resolve_generate_task;

    GenerateParams params;
    GenerateTask task;

    params.audio_cover_strength = 2.0f;
    params.cover_noise_strength = -0.5f;
    CHECK(resolve_generate_task(params, task).empty());
    CHECK(approx(task.audio_cover_strength, 1.0f));
    CHECK(approx(task.cover_noise_strength, 0.0f));

    params.audio_cover_strength = std::numeric_limits<float>::quiet_NaN();
    CHECK(resolve_generate_task(params, task).find("must be finite") != std::string::npos);

    params.audio_cover_strength = std::numeric_limits<float>::infinity();
    CHECK(resolve_generate_task(params, task).find("must be finite") != std::string::npos);

    params.audio_cover_strength = 1.0f;
    params.cover_noise_strength = std::numeric_limits<float>::quiet_NaN();
    CHECK(resolve_generate_task(params, task).find("must be finite") != std::string::npos);

    params.cover_noise_strength = std::numeric_limits<float>::infinity();
    CHECK(resolve_generate_task(params, task).find("must be finite") != std::string::npos);
}

void test_generation_plans() {
    using tts_cpp::acestep::GenerateParams;
    using tts_cpp::acestep::GenerateTask;
    using tts_cpp::acestep::GenerationPlan;
    using tts_cpp::acestep::TASK_COVER_NOFSQ;
    using tts_cpp::acestep::make_generation_plan;
    using tts_cpp::acestep::resolve_generate_task;

    GenerateParams params;
    GenerateTask task;
    CHECK(resolve_generate_task(params, task).empty());

    GenerationPlan plan = make_generation_plan(params, task);
    CHECK(!plan.encode_source);
    CHECK(!plan.encode_reference);
    CHECK(plan.run_lm);
    CHECK(plan.run_detokenizer);

    params.reference_audio.assign(4, 0.0f);
    CHECK(resolve_generate_task(params, task).empty());
    plan = make_generation_plan(params, task);
    CHECK(plan.encode_reference);
    CHECK(plan.run_lm);

    params.task_type = TASK_COVER_NOFSQ;
    params.source_audio.assign(4, 0.0f);
    params.reference_audio.clear();
    CHECK(resolve_generate_task(params, task).empty());
    plan = make_generation_plan(params, task);
    CHECK(plan.encode_source);
    CHECK(plan.reuse_source_reference);
    CHECK(!plan.run_lm);
    CHECK(!plan.run_detokenizer);
    CHECK(!plan.blend_cover_noise);

    params.reference_audio.assign(4, 0.0f);
    params.cover_noise_strength = 0.5f;
    CHECK(resolve_generate_task(params, task).empty());
    plan = make_generation_plan(params, task);
    CHECK(plan.encode_reference);
    CHECK(!plan.reuse_source_reference);
    CHECK(plan.blend_cover_noise);
}

void test_lego_task_kinds() {
    using tts_cpp::acestep::TASK_COVER_NOFSQ;
    using tts_cpp::acestep::TASK_LEGO;
    using tts_cpp::acestep::TASK_TEXT2MUSIC;
    using tts_cpp::acestep::LEGO_TRACK_NAMES;
    using tts_cpp::acestep::is_lego_task;
    using tts_cpp::acestep::is_source_task;
    using tts_cpp::acestep::is_valid_lego_track;

    CHECK(is_lego_task(TASK_LEGO));
    CHECK(!is_lego_task(TASK_TEXT2MUSIC));
    CHECK(!is_lego_task(TASK_COVER_NOFSQ));
    CHECK(is_source_task(TASK_LEGO));
    CHECK(is_source_task(TASK_COVER_NOFSQ));
    CHECK(!is_source_task(TASK_TEXT2MUSIC));

    for (const char * name : LEGO_TRACK_NAMES) {
        CHECK(is_valid_lego_track(name));
    }
    CHECK(!is_valid_lego_track(""));
    CHECK(!is_valid_lego_track("piano"));
    CHECK(!is_valid_lego_track("GUITAR"));
}

void test_lego_task_validation() {
    using tts_cpp::acestep::GenerateParams;
    using tts_cpp::acestep::GenerateTask;
    using tts_cpp::acestep::TASK_LEGO;
    using tts_cpp::acestep::resolve_generate_task;

    GenerateParams params;
    GenerateTask task;

    params.task_type = TASK_LEGO;
    CHECK(resolve_generate_task(params, task).find("requires source_audio") != std::string::npos);

    params.source_audio.assign(3, 0.0f);
    CHECK(resolve_generate_task(params, task).find("source_audio must be interleaved stereo") != std::string::npos);

    params.source_audio.assign(4, 0.0f);
    CHECK(resolve_generate_task(params, task).find("requires a track name") != std::string::npos);

    params.track = "accordion";
    CHECK(resolve_generate_task(params, task).find("unknown lego track") != std::string::npos);

    params.track = "guitar";
    CHECK(resolve_generate_task(params, task).empty());
    CHECK(task.type == TASK_LEGO);
    CHECK(task.track == "guitar");
}

void test_lego_generation_plan() {
    using tts_cpp::acestep::GenerateParams;
    using tts_cpp::acestep::GenerateTask;
    using tts_cpp::acestep::GenerationPlan;
    using tts_cpp::acestep::TASK_LEGO;
    using tts_cpp::acestep::make_generation_plan;
    using tts_cpp::acestep::resolve_generate_task;

    GenerateParams params;
    GenerateTask task;
    params.task_type = TASK_LEGO;
    params.track = "drums";
    params.source_audio.assign(4, 0.0f);
    CHECK(resolve_generate_task(params, task).empty());

    GenerationPlan plan = make_generation_plan(params, task);
    CHECK(plan.encode_source);
    CHECK(!plan.encode_reference);
    CHECK(!plan.reuse_source_reference);
    CHECK(!plan.run_lm);
    CHECK(!plan.run_detokenizer);
    CHECK(!plan.blend_cover_noise);

    params.reference_audio.assign(4, 0.0f);
    CHECK(resolve_generate_task(params, task).empty());
    plan = make_generation_plan(params, task);
    CHECK(plan.encode_reference);
    CHECK(!plan.reuse_source_reference);
}

void test_lego_model_policy() {
    using tts_cpp::acestep::is_sft_model_name;
    using tts_cpp::acestep::lego_model_error;

    CHECK(is_sft_model_name("acestep-v15-sft"));
    CHECK(is_sft_model_name("acestep-v15-xl-sft"));
    CHECK(!is_sft_model_name("acestep-v15-base"));
    CHECK(!is_sft_model_name("acestep-v15-xl-base"));
    CHECK(!is_sft_model_name("acestep-v15-turbo"));
    CHECK(!is_sft_model_name(""));

    CHECK(lego_model_error(false, false).empty());
    CHECK(lego_model_error(true, false).find("requires a base DiT") != std::string::npos);
    CHECK(lego_model_error(true, false).find("turbo") != std::string::npos);
    CHECK(lego_model_error(false, true).find("requires a base DiT") != std::string::npos);
    CHECK(lego_model_error(false, true).find("sft") != std::string::npos);
}

void test_generation_conditioning() {
    using tts_cpp::acestep::AudioEncoder;
    using tts_cpp::acestep::EncodedAudio;
    using tts_cpp::acestep::GenerateParams;
    using tts_cpp::acestep::GenerateTask;
    using tts_cpp::acestep::GenerationConditioning;
    using tts_cpp::acestep::GenerationPlan;
    using tts_cpp::acestep::TASK_COVER_NOFSQ;
    using tts_cpp::acestep::TimbreInput;
    using tts_cpp::acestep::make_generation_plan;
    using tts_cpp::acestep::prepare_generation_conditioning;
    using tts_cpp::acestep::resolve_generate_task;
    using tts_cpp::acestep::resolve_timbre_input;

    GenerateParams params;
    params.task_type = TASK_COVER_NOFSQ;
    params.source_audio = { 1.0f, 2.0f, 3.0f, 4.0f };
    params.reference_audio = { 5.0f, 6.0f };

    GenerateTask task;
    CHECK(resolve_generate_task(params, task).empty());
    const GenerationPlan plan = make_generation_plan(params, task);

    int source_calls = 0;
    int reference_calls = 0;
    const AudioEncoder encode = [&](const std::vector<float> & pcm, const char * stage, EncodedAudio & output) {
        if (std::string(stage) == "source") ++source_calls;
        if (std::string(stage) == "reference") ++reference_calls;
        output.latent = { pcm.front(), pcm.back() };
        output.frames = (int) (pcm.size() / 2);
        return true;
    };

    GenerationConditioning conditioning;
    CHECK(prepare_generation_conditioning(params, plan, encode, conditioning));
    CHECK(source_calls == 1);
    CHECK(reference_calls == 1);
    CHECK(conditioning.source.latent.front() == params.source_audio.front());
    CHECK(conditioning.reference.latent.front() == params.reference_audio.front());

    const std::vector<float> silence = { -1.0f, -1.0f };
    const TimbreInput explicit_reference =
        resolve_timbre_input(plan, conditioning.reference, conditioning.source.latent,
                             conditioning.source.frames, silence);
    CHECK(explicit_reference.data == conditioning.reference.latent.data());
    CHECK(explicit_reference.frames == conditioning.reference.frames);

    params.reference_audio.clear();
    CHECK(resolve_generate_task(params, task).empty());
    const GenerationPlan reuse_plan = make_generation_plan(params, task);
    GenerationConditioning reuse_conditioning;
    CHECK(prepare_generation_conditioning(params, reuse_plan, encode, reuse_conditioning));
    CHECK(source_calls == 2);
    CHECK(reference_calls == 1);
    const TimbreInput reused_source =
        resolve_timbre_input(reuse_plan, reuse_conditioning.reference, reuse_conditioning.source.latent,
                             reuse_conditioning.source.frames, silence);
    CHECK(reused_source.data == reuse_conditioning.source.latent.data());
    CHECK(reused_source.frames == reuse_conditioning.source.frames);
}

void test_cover_noise_blending() {
    using tts_cpp::acestep::CoverNoiseResult;
    using tts_cpp::acestep::apply_cover_noise;

    std::vector<float> noise = { 1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f };
    const std::vector<float> source = { 5.0f, 7.0f, 11.0f, 13.0f };
    std::vector<float> schedule = { 1.0f, 0.75f, 0.5f, 0.25f };

    const CoverNoiseResult result = apply_cover_noise(noise, source, 3, 2, 2, 0.5f, schedule);
    CHECK(approx(result.nearest_time, 0.5f));
    CHECK(result.remaining_steps == 2);
    CHECK(schedule.size() == 2);
    CHECK(approx(noise[0], 3.0f));
    CHECK(approx(noise[1], 4.0f));
    CHECK(approx(noise[2], 6.5f));
    CHECK(approx(noise[3], 7.5f));
    CHECK(approx(noise[4], 7.0f));
    CHECK(approx(noise[5], 8.0f));
}

constexpr float TEST_CONSERVATIVE_STRENGTH = 0.9f;
constexpr float TEST_BALANCED_STRENGTH = 0.4f;
constexpr float TEST_BALANCED_HALF_STRENGTH = 0.5f;
constexpr int TEST_BALANCED_BLEND_FRAMES = 15;
constexpr int TEST_BALANCED_HALF_BLEND_FRAMES = 12;
constexpr int TEST_REPAINT_SECONDS = 4;
constexpr int TEST_REPAINT_LATENT_FRAMES = 100;
constexpr int TEST_REPAINT_TRAILING_SAMPLES = 321;
constexpr char TEST_OUTPAINT_ERROR[] = "outpainting";
constexpr char TEST_RANGE_ORDER_ERROR[] = "greater";
constexpr char TEST_SOURCE_CAPTION_ERROR[] = "source_caption";
constexpr char TEST_FLOW_RANGE_ERROR[] = "n_min";
constexpr char TEST_FLOW_AVERAGE_ERROR[] = "n_avg";
constexpr char TEST_FLOW_DCW_ERROR[] = "DCW";
constexpr char TEST_SOURCE_CAPTION[] = "source";
constexpr char TEST_TARGET_CAPTION[] = "target";
constexpr char TEST_MATERIALIZE_EVENT[] = "materialize";
constexpr char TEST_FLOW_EVENT[] = "flow";

void test_repaint_config() {
    using namespace tts_cpp::acestep;

    const RepaintConfig conservative =
        resolve_repaint_config(RepaintMode::Conservative, TEST_CONSERVATIVE_STRENGTH);
    CHECK(approx(conservative.injection_ratio, REPAINT_CONSERVATIVE_INJECTION_RATIO));
    CHECK(conservative.latent_blend_frames == REPAINT_CONSERVATIVE_BLEND_FRAMES);
    CHECK(approx(conservative.waveform_fade_sec, REPAINT_CONSERVATIVE_FADE_SECONDS));
    CHECK(conservative.preserve_waveform);

    const RepaintConfig balanced =
        resolve_repaint_config(RepaintMode::Balanced, TEST_BALANCED_STRENGTH);
    CHECK(approx(balanced.injection_ratio, AUDIO_EDIT_MAX_RATIO - TEST_BALANCED_STRENGTH));
    CHECK(balanced.latent_blend_frames == TEST_BALANCED_BLEND_FRAMES);
    CHECK(approx(balanced.waveform_fade_sec,
                 REPAINT_CONSERVATIVE_FADE_SECONDS *
                     (AUDIO_EDIT_MAX_RATIO - TEST_BALANCED_STRENGTH)));
    CHECK(resolve_repaint_config(RepaintMode::Balanced, TEST_BALANCED_HALF_STRENGTH)
              .latent_blend_frames == TEST_BALANCED_HALF_BLEND_FRAMES);
    CHECK(audio_edit_round_ties_to_even(3.5f) == 4);
    CHECK(audio_edit_round_ties_to_even(4.5f) == 4);

    const RepaintConfig aggressive =
        resolve_repaint_config(RepaintMode::Aggressive, AUDIO_EDIT_MIN_RATIO);
    CHECK(approx(aggressive.injection_ratio, REPAINT_AGGRESSIVE_INJECTION_RATIO));
    CHECK(aggressive.latent_blend_frames == REPAINT_AGGRESSIVE_BLEND_FRAMES);
    CHECK(!aggressive.preserve_waveform);
}

void test_repaint_range() {
    using namespace tts_cpp::acestep;
    const int source_samples = AUDIO_EDIT_SAMPLE_RATE * TEST_REPAINT_SECONDS;
    RepaintRange range;
    CHECK(resolve_repaint_range(AUDIO_EDIT_MIN_RATIO, REPAINT_SOURCE_END_SECONDS,
                                source_samples, TEST_REPAINT_LATENT_FRAMES, range).empty());
    CHECK(range.sample_start == 0);
    CHECK(range.sample_end == source_samples);
    CHECK(range.latent_start == 0);
    CHECK(range.latent_end == TEST_REPAINT_LATENT_FRAMES);
    const int trailing_source_samples = source_samples + TEST_REPAINT_TRAILING_SAMPLES;
    const int trailing_latent_frames = TEST_REPAINT_LATENT_FRAMES + 1;
    CHECK(resolve_repaint_range(AUDIO_EDIT_MIN_RATIO, REPAINT_SOURCE_END_SECONDS,
                                trailing_source_samples, trailing_latent_frames, range).empty());
    CHECK(range.sample_end == trailing_source_samples);
    CHECK(range.latent_end == trailing_latent_frames);
    CHECK(resolve_repaint_range(-0.1f, 1.0f, source_samples, TEST_REPAINT_LATENT_FRAMES, range)
              .find(TEST_OUTPAINT_ERROR) != std::string::npos);
    CHECK(resolve_repaint_range(1.0f, 4.1f, source_samples, TEST_REPAINT_LATENT_FRAMES, range)
              .find(TEST_OUTPAINT_ERROR) != std::string::npos);
    CHECK(resolve_repaint_range(2.0f, 2.0f, source_samples, TEST_REPAINT_LATENT_FRAMES, range)
              .find(TEST_RANGE_ORDER_ERROR) != std::string::npos);
}

void check_all_samples_equal(const std::vector<float> & samples, float expected) {
    for (float sample : samples) CHECK(approx(sample, expected));
}

void test_repaint_mask_injection_blend_and_splice() {
    using namespace tts_cpp::acestep;

    const std::vector<float> mask = make_repaint_mask(5, 1, 4);
    CHECK(mask == std::vector<float>({ 0.0f, 1.0f, 1.0f, 1.0f, 0.0f }));

    std::vector<float> current(10, 10.0f);
    const std::vector<float> clean = { 2, 4, 2, 4, 2, 4, 2, 4, 2, 4 };
    const std::vector<float> noise(10, 8.0f);
    repaint_inject_source(
        current, clean.data(), noise.data(), mask.data(), mask.size(), 0.25f, 2);
    CHECK(approx(current[0], 3.5f));
    CHECK(approx(current[1], 5.0f));
    CHECK(approx(current[2], 10.0f));
    CHECK(approx(current[8], 3.5f));

    std::vector<float> generated(10, 10.0f);
    repaint_blend_latent(
        generated, clean.data(), mask.data(), mask.size(), 0, 2);
    CHECK(approx(generated[0], 2.0f));
    CHECK(approx(generated[1], 4.0f));
    CHECK(approx(generated[2], 10.0f));
    CHECK(approx(generated[8], 2.0f));

    generated.assign(10, 10.0f);
    repaint_blend_latent(
        generated, clean.data(), mask.data(), mask.size(), 1, 2);
    CHECK(generated[0] > clean[0] && generated[0] < 10.0f);
    CHECK(generated[8] > clean[8] && generated[8] < 10.0f);

    std::vector<float> wav(12, 1.0f);
    const std::vector<float> source(12, -1.0f);
    repaint_splice_waveform(wav, source, 2, 4, 0, 2);
    CHECK(approx(wav[0], -1.0f));
    CHECK(approx(wav[2], -1.0f));
    CHECK(approx(wav[4], 1.0f));
    CHECK(approx(wav[8], -1.0f));

    wav.assign(12, 1.0f);
    repaint_splice_waveform(wav, source, 0, 6, 2, 2);
    check_all_samples_equal(wav, 1.0f);
}

void test_flow_edit_validation_and_math() {
    using namespace tts_cpp::acestep;

    FlowEditParams params;
    CHECK(validate_flow_edit_params(params).find(TEST_SOURCE_CAPTION_ERROR) != std::string::npos);
    params.source_caption = TEST_SOURCE_CAPTION;
    params.target_caption = TEST_TARGET_CAPTION;
    CHECK(validate_flow_edit_params(params).empty());
    params.n_min = 0.8f;
    params.n_max = 0.2f;
    CHECK(validate_flow_edit_params(params).find(TEST_FLOW_RANGE_ERROR) != std::string::npos);
    params.n_min = AUDIO_EDIT_MIN_RATIO;
    params.n_max = AUDIO_EDIT_MAX_RATIO;
    params.n_avg = 0;
    CHECK(validate_flow_edit_params(params).find(TEST_FLOW_AVERAGE_ERROR) != std::string::npos);
    params.n_avg = FLOW_EDIT_DEFAULT_AVERAGES;
    params.dcw_enabled = true;
    CHECK(validate_flow_edit_params(params).find(TEST_FLOW_DCW_ERROR) != std::string::npos);

    const std::vector<float> source = { 2.0f, 4.0f };
    const std::vector<float> noise = { 10.0f, 12.0f };
    std::vector<float> noisy_source;
    flow_edit_make_source(source, noise, 0.25f, noisy_source);
    CHECK(approx(noisy_source[0], 4.0f));
    CHECK(approx(noisy_source[1], 6.0f));

    std::vector<float> edit = { 3.0f, 5.0f };
    std::vector<float> noisy_target;
    flow_edit_make_target(edit, noisy_source, source, noisy_target);
    CHECK(approx(noisy_target[0], 5.0f));
    CHECK(approx(noisy_target[1], 7.0f));
    flow_edit_integrate_delta(edit, { 5.0f, 7.0f }, { 1.0f, 2.0f }, -0.1f);
    CHECK(approx(edit[0], 2.6f));
    CHECK(approx(edit[1], 4.5f));
}

void test_audio_edit_factory_and_order() {
    using namespace tts_cpp::acestep;

    RepaintParams repaint;
    FlowEditParams flow;
    flow.source_caption = TEST_SOURCE_CAPTION;
    flow.target_caption = TEST_TARGET_CAPTION;

    std::unique_ptr<AudioEditOperation> repaint_operation =
        make_audio_edit_operation(AudioEditParams(repaint));
    std::unique_ptr<AudioEditOperation> flow_operation =
        make_audio_edit_operation(AudioEditParams(flow));
    CHECK(dynamic_cast<RepaintOperation *>(repaint_operation.get()) != nullptr);
    CHECK(dynamic_cast<FlowEditOperation *>(flow_operation.get()) != nullptr);

    const std::vector<AudioEditParams> plan = { flow, repaint, flow, repaint };
    AudioEditPipeline pipeline = make_audio_edit_pipeline(plan);
    CHECK(pipeline.size() == 4);
    CHECK(std::string(pipeline.at(0).name()) == AUDIO_EDIT_FLOW_STAGE);
    CHECK(std::string(pipeline.at(1).name()) == AUDIO_EDIT_REPAINT_STAGE);

    AudioEditArtifact artifact;
    std::vector<std::string> order;
    AudioEditCapabilities capabilities;
    capabilities.prepare_repaint_source = [&](AudioEditArtifact & value) {
        order.push_back(TEST_MATERIALIZE_EVENT);
        value.pcm_is_current = true;
    };
    capabilities.flow_edit = [&](const FlowEditParams &, AudioEditArtifact & value) {
        order.push_back(TEST_FLOW_EVENT);
        value.pcm_is_current = false;
    };
    capabilities.repaint = [&](const RepaintParams &, AudioEditArtifact & value) {
        order.push_back(AUDIO_EDIT_REPAINT_STAGE);
        value.pcm_is_current = false;
    };
    pipeline.execute(artifact, capabilities);
    const std::vector<std::string> expected = {
        TEST_FLOW_EVENT, TEST_MATERIALIZE_EVENT, AUDIO_EDIT_REPAINT_STAGE,
        TEST_FLOW_EVENT, TEST_MATERIALIZE_EVENT, AUDIO_EDIT_REPAINT_STAGE,
    };
    CHECK(order == expected);
}

void test_quality_score_request_policy() {
    using tts_cpp::acestep::GenerateParams;
    using tts_cpp::acestep::GenerateTask;
    using tts_cpp::acestep::RepaintParams;
    using tts_cpp::acestep::resolve_generate_task;

    GenerateParams params;
    GenerateTask   task;
    params.compute_quality_score = true;
    params.caption               = "a caption";
    CHECK(resolve_generate_task(params, task).empty());

    params.task_type = tts_cpp::acestep::TASK_COVER_NOFSQ;
    params.source_audio.assign(96, 0.0f);
    CHECK(resolve_generate_task(params, task).find("LM code path") != std::string::npos);

    params.task_type = tts_cpp::acestep::TASK_LEGO;
    params.track     = "drums";
    CHECK(resolve_generate_task(params, task).find("LM code path") != std::string::npos);

    params.task_type = tts_cpp::acestep::TASK_TEXT2MUSIC;
    params.edit_plan.push_back(RepaintParams{});
    CHECK(resolve_generate_task(params, task).find("audio edit path") != std::string::npos);
}

void test_cancellation_scope_consumes_on_exit() {
    using tts_cpp::acestep::CancellationScope;

    std::atomic<bool> cancelled{ true };
    {
        CancellationScope scope(cancelled);
        CHECK(cancelled.load());
    }
    CHECK(!cancelled.load());

    cancelled.store(false);
    {
        CancellationScope scope(cancelled);
        CHECK(!cancelled.load());
        cancelled.store(true);
    }
    CHECK(!cancelled.load());
}


}  // namespace

int main() {
    test_generate_task_kinds();
    test_generate_task_defaults();
    test_generate_task_audio_layout();
    test_generate_task_errors();
    test_simple_mode_policy();
    test_simple_mode_prompt_resolvers();
    test_lrc_request_policy();
    test_format_user_message();
    test_rewrite_query_policy();
    test_inspire_user_message();
    test_cover_conditioning_switch();
    test_generate_task_strengths();
    test_generation_plans();
    test_lego_task_kinds();
    test_lego_task_validation();
    test_lego_generation_plan();
    test_lego_model_policy();
    test_generation_conditioning();
    test_cover_noise_blending();
    test_repaint_config();
    test_repaint_range();
    test_repaint_mask_injection_blend_and_splice();
    test_flow_edit_validation_and_math();
    test_audio_edit_factory_and_order();
    test_quality_score_request_policy();
    test_cancellation_scope_consumes_on_exit();
    std::fprintf(stderr, "[test-acestep-generation] %d/%d checks passed\n", g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
