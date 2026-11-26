// Model-free ACE-Step audio regression tests.

#include "acestep_test_support.h"
#include "acestep_test_tokenizer.h"
#include "vae_coreml_path.h"
#include "vae_coreml_windows.h"
#include "vae_encode_windows.h"
#include "vae_ggml.h"
#include "loudness.h"
#include "lyrics_alignment.h"
#include "quality_score.h"
#include "wav_reader.h"

namespace {

void test_vae_progress() {
    using tts_cpp::acestep::vae_progress_pct;

    // Endpoints and a midpoint.
    CHECK(vae_progress_pct(0, 200) == 0);
    CHECK(vae_progress_pct(100, 200) == 50);
    CHECK(vae_progress_pct(200, 200) == 100);

    // Overshoot: the scheduler fires past `total` -> clamped to 100, never more.
    CHECK(vae_progress_pct(201, 200) == 100);
    CHECK(vae_progress_pct(10000, 200) == 100);

    // Degenerate inputs are safe (no div-by-zero, no negative pct).
    CHECK(vae_progress_pct(5, 0) == 0);
    CHECK(vae_progress_pct(5, -1) == 0);
    CHECK(vae_progress_pct(-3, 200) == 0);

    // Monotone non-decreasing and bounded across a full sweep that overshoots,
    // mirroring the eval callback incrementing `done` once per fired node.
    const int total = 137;  // odd, to exercise integer division
    int       prev  = -1;
    int       last_emitted = -1;
    for (int done = 1; done <= total + 25; ++done) {  // +25 => simulate extra copy/split nodes
        int pct = vae_progress_pct(done, total);
        CHECK(pct >= 0 && pct <= 100);
        CHECK(pct >= prev);  // never goes backwards
        prev = pct;
        // Throttle: only distinct percentages would be surfaced to the user cb.
        if (pct != last_emitted) last_emitted = pct;
    }
    CHECK(prev == 100);          // ends exactly at 100
    CHECK(last_emitted == 100);  // the final surfaced value is 100
}

void test_coreml_sidecar_path() {
    using tts_cpp::acestep::coreml_vae_sidecar_path;

    CHECK(coreml_vae_sidecar_path("models/vae-BF16.gguf") == "models/vae-decoder.mlmodelc");
    CHECK(coreml_vae_sidecar_path("vae.f16.gguf") == "vae-decoder.mlmodelc");
    CHECK(coreml_vae_sidecar_path("models/vae.gguf") == "models/vae-decoder.mlmodelc");
    CHECK(coreml_vae_sidecar_path("C:\\m\\vae-BF16.gguf") == "C:\\m\\vae-decoder.mlmodelc");
    CHECK(coreml_vae_sidecar_path("my-vae.gguf") == "my-vae-decoder.mlmodelc");
}

// The sidecar's input shape is fixed, so every planned window must be exactly
// window_frames long (the last one end-aligned), with the kept cores disjoint
// and covering the latent exactly.
void test_coreml_plan_windows() {
    using tts_cpp::acestep::vae_coreml_plan_windows;
    using tts_cpp::acestep::VaeCoremlWindow;

    using tts_cpp::acestep::VAE_COREML_OVERLAP;
    CHECK(VAE_COREML_OVERLAP == 8);   // measured bound, see vae_coreml_windows.h

    CHECK(vae_coreml_plan_windows(300, 352, 48).empty());   // shorter than one window
    CHECK(vae_coreml_plan_windows(352, 96, 48).empty());    // no core left after overlap

    const std::vector<VaeCoremlWindow> tiny = vae_coreml_plan_windows(740, 64, VAE_COREML_OVERLAP);
    CHECK(!tiny.empty());
    CHECK(tiny.front().core_a == 0 && tiny.back().core_b == 740);

    const std::vector<VaeCoremlWindow> single = vae_coreml_plan_windows(352, 352, 48);
    CHECK(single.size() == 1);
    CHECK(single[0].win_a == 0 && single[0].core_a == 0 && single[0].core_b == 352);

    for (int T : {353, 740, 1024, 4096}) {
        const std::vector<VaeCoremlWindow> plan = vae_coreml_plan_windows(T, 352, 48);
        CHECK(!plan.empty());
        CHECK(plan.front().core_a == 0);
        CHECK(plan.back().core_b == T);
        for (size_t i = 0; i < plan.size(); ++i) {
            const VaeCoremlWindow & w = plan[i];
            CHECK(w.win_a >= 0 && w.win_a + 352 <= T);              // window inside the latent
            CHECK(w.core_a >= w.win_a && w.core_b <= w.win_a + 352);  // core inside the window
            CHECK(w.core_a < w.core_b);
            if (i > 0) CHECK(w.core_a == plan[i - 1].core_b);       // disjoint, gapless cores
            if (w.core_a > 0) CHECK(w.core_a - w.win_a >= 48);      // left context except at 0
            if (w.core_b < T) CHECK(w.win_a + 352 - w.core_b >= 48);  // right context except at T
        }
    }
}

// The decoder's im2col node grows linearly with the window and cannot be split
// across allocations, so a backend that caps allocation size caps the window.
// Adreno 740 reports 1024 MB and the 256+2*48 window needs 1155 MiB, which is
// what aborted a 30 s GPU decode; these are those measured numbers.
void test_vae_window_core() {
    using tts_cpp::acestep::vae_shrink_window_core;

    const int    core = 256, ov = 48, core_min = 64;
    const size_t adreno_cap = (size_t) 1024 * 1024 * 1024;
    const size_t peak_352   = (size_t) 1155 * 1024 * 1024;  // measured at 256 + 2*48 frames

    // A backend that does not cap allocations (CPU reports SIZE_MAX) keeps 256,
    // so CPU / Metal / iOS behaviour is untouched.
    CHECK(vae_shrink_window_core(core, ov, peak_352, SIZE_MAX, core_min) == core);
    CHECK(vae_shrink_window_core(core, ov, (size_t) 900 * 1024 * 1024, adreno_cap, core_min) == core);

    // Adreno: must shrink, and the result must actually fit once rescaled.
    const int fitted = vae_shrink_window_core(core, ov, peak_352, adreno_cap, core_min);
    CHECK(fitted < core);
    CHECK(fitted >= core_min);
    const double bytes_per_frame = (double) peak_352 / (double) (core + 2 * ov);
    CHECK((size_t) (bytes_per_frame * (fitted + 2 * ov)) <= adreno_cap);

    // Never below the floor, however small the cap.
    CHECK(vae_shrink_window_core(core, ov, peak_352, 1024 * 1024, core_min) == core_min);
    CHECK(vae_shrink_window_core(core_min, ov, peak_352, 1024 * 1024, core_min) == core_min);

    // Converges: re-applying with the rescaled peak is a fixed point.
    const size_t peak_fitted = (size_t) (bytes_per_frame * (fitted + 2 * ov));
    CHECK(vae_shrink_window_core(fitted, ov, peak_fitted, adreno_cap, core_min) == fitted);

    // Always makes progress while it does not fit, so the caller's loop ends.
    int c = core;
    for (int i = 0; i < 16 && c > core_min; ++i) {
        const int next = vae_shrink_window_core(c, ov, peak_352, adreno_cap, core_min);
        if (next == c) break;
        CHECK(next < c);
        c = next;
    }
}

void test_normalize_loudness() {
    using tts_cpp::acestep::normalize_loudness;

    std::vector<float> pcm = { 0.1f, -0.2f, 0.4f, 0.05f };
    normalize_loudness(pcm, 0);
    CHECK(approx(pcm[0], 0.25f));
    CHECK(approx(pcm[1], -0.5f));
    CHECK(approx(pcm[2], 1.0f));
    CHECK(approx(pcm[3], 0.125f));

    std::vector<float> clipped = { 0.1f, -0.2f, 0.4f, 0.05f };
    normalize_loudness(clipped, 10);
    CHECK(approx(clipped[0], 0.5f));
    CHECK(approx(clipped[1], -1.0f));
    CHECK(approx(clipped[2], 1.0f));
    CHECK(approx(clipped[3], 0.25f));

    std::vector<float> silence(16, 0.0f);
    normalize_loudness(silence);
    CHECK(silence == std::vector<float>(16, 0.0f));

    std::vector<float> empty;
    normalize_loudness(empty);
    CHECK(empty.empty());
}

bool near_value(double actual, double expected, double tolerance) {
    return std::fabs(actual - expected) <= tolerance;
}

void test_lyrics_matrix_conversion() {
    using tts_cpp::acestep::lyrics::Matrix;
    using tts_cpp::acestep::lyrics::matrix_from_column_major;

    const std::vector<float> ggml_values = {
        1, 5, 9,
        2, 6, 10,
        3, 7, 11,
        4, 8, 12,
    };
    const Matrix matrix = matrix_from_column_major(ggml_values.data(), 3, 4);
    CHECK(matrix.values == std::vector<float>({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }));
}

void test_lyrics_dtw() {
    using tts_cpp::acestep::lyrics::Matrix;
    using tts_cpp::acestep::lyrics::dtw;
    using tts_cpp::acestep::lyrics::DtwPath;

    const Matrix costs(3, 3, {
        0.0f, 2.0f, 2.0f,
        2.0f, 0.0f, 2.0f,
        2.0f, 2.0f, 0.0f,
    });
    DtwPath path = dtw(costs);
    CHECK(path.token_indices == std::vector<int32_t>({ 0, 1, 2 }));
    CHECK(path.frame_indices == std::vector<int32_t>({ 0, 1, 2 }));

    const Matrix rectangular(2, 5);
    path = dtw(rectangular);
    CHECK(path.token_indices.front() == 0 && path.token_indices.back() == 1);
    CHECK(path.frame_indices.front() == 0 && path.frame_indices.back() == 4);

    // Reference-parity tie-break (_dtw.py): with diagonal == vertical <
    // horizontal at the last cell, the strict comparison chain falls through
    // to horizontal, so the path detours through (1, 0) instead of stepping
    // diagonally.
    const Matrix tie(2, 2, {
        0.0f, 0.0f,
        1.0f, 0.0f,
    });
    path = dtw(tie);
    CHECK(path.token_indices == std::vector<int32_t>({ 0, 1, 1 }));
    CHECK(path.frame_indices == std::vector<int32_t>({ 0, 0, 1 }));
}

void test_lyrics_preprocessing() {
    using namespace tts_cpp::acestep::lyrics;

    const Matrix spike(1, 5, { 0.0f, 0.0f, 10.0f, 0.0f, 0.0f });
    const Matrix filtered = median_filter(spike, 3);
    CHECK(filtered.values == std::vector<float>({ 0, 0, 0, 0, 0 }));

    const Matrix head_a(2, 3, { 1, 2, 3, 4, 5, 6 });
    const Matrix head_b(2, 3, { 3, 4, 5, 6, 7, 8 });
    const auto scoring = preprocess_scoring({ head_a, head_b }, 1);
    CHECK(near_value(scoring.average_matrix(0, 0), 2.0, 1e-6));
    CHECK(near_value(scoring.energy_matrix(0, 0), 0.0, 1e-6));
    CHECK(near_value(scoring.energy_matrix(1, 2), 1.0, 1e-6));
    CHECK(near_value(scoring.calc_matrix(0, 1), 0.04, 1e-6));

    const auto alignment = preprocess_alignment({ head_a, head_b }, 2.0f, 1);
    CHECK(alignment.calc_matrix.rows == 2 && alignment.calc_matrix.cols == 3);
    CHECK(alignment.energy_matrix.rows == 2 && alignment.energy_matrix.cols == 3);
}

void test_lyrics_metrics_and_score() {
    using namespace tts_cpp::acestep::lyrics;

    CHECK(token_type_mask({ "hello", "[verse", "]", "world" }) ==
          std::vector<int32_t>({ 1, 0, 0, 1 }));

    const Matrix energy(3, 4, {
        0.8f, 0.2f, 0.0f, 0.0f,
        0.0f, 0.5f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.2f, 0.8f,
    });
    DtwPath path;
    path.token_indices = { 0, 1, 2 };
    path.frame_indices = { 0, 1, 3 };
    const AlignmentMetrics metrics =
        compute_alignment_metrics(energy, path, { 1, 0, 1 }, 0.01, 0.0, 0.5);
    CHECK(near_value(metrics.coverage, 1.0, 1e-12));
    CHECK(near_value(metrics.monotonicity, 1.0, 1e-12));
    CHECK(near_value(metrics.path_confidence, 0.74, 1e-7));
    CHECK(near_value(calculate_lyrics_score(metrics), 0.74, 1e-12));
}

void test_lyrics_timestamps_and_lrc() {
    using namespace tts_cpp::acestep::lyrics;

    const Matrix attention(2, 4, {
        5.0f, 5.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 5.0f, 5.0f,
    });
    const std::vector<TokenTimestamp> aligned =
        token_timestamps(attention, { 1, 2 }, { "Hello", "\n" }, 4.0);
    CHECK(near_value(aligned[0].start, 0.0, 1e-12));
    CHECK(near_value(aligned[0].end, 1.0, 1e-12));
    CHECK(near_value(aligned[1].start, 1.0, 1e-12));
    CHECK(near_value(aligned[1].end, 3.0, 1e-12));

    const std::vector<TokenTimestamp> tokens = {
        { 1, "Hello", 1.23456, 2.0, 0.2 },
        { 2, "\n", 2.0, 2.5, 0.4 },
        { 3, "World", 61.0, 62.0, 0.8 },
    };
    const auto decoder = [](const std::vector<int> & ids) {
        std::string text;
        for (int id : ids) {
            if (id == 1) text += "Hello";
            if (id == 2) text += "\n";
            if (id == 3) text += "World";
        }
        return text;
    };
    const std::vector<SentenceTimestamp> sentences = sentence_timestamps(tokens, decoder);
    CHECK(sentences.size() == 2);
    CHECK(sentences[0].text == "Hello" && sentences[1].text == "World");
    CHECK(near_value(sentences[0].start, 1.235, 1e-12));
    CHECK(near_value(sentences[0].confidence, 0.0, 1e-12));
    CHECK(near_value(sentences[1].confidence, 1.0, 1e-12));
    CHECK(format_lrc(sentences) == "[00:01.24]Hello\n[01:01.00]World");
    CHECK(format_lrc(sentences, true) ==
          "[00:01.24][00:02.50]Hello\n[01:01.00][01:02.00]World");
}

void fill_fake_pcm(std::vector<float> & pcm, int frames) {
    for (int frame = 0; frame < frames; ++frame) {
        pcm[(size_t) frame * 2] = (float) frame;
        pcm[(size_t) frame * 2 + 1] = (float) -frame;
    }
}

void fill_fake_latent_frame(std::vector<float> & latent, int frame, float value) {
    using tts_cpp::acestep::VAE_LATENT_CHANNELS;
    for (int channel = 0; channel < VAE_LATENT_CHANNELS; ++channel) {
        latent[(size_t) frame * VAE_LATENT_CHANNELS + channel] = value + (float) channel;
    }
}

int encode_fake_vae(const float * pcm, int frames, std::vector<float> & latent) {
    using tts_cpp::acestep::VAE_ENCODER_UPSAMPLE;
    using tts_cpp::acestep::VAE_LATENT_CHANNELS;

    const int latent_frames = frames / VAE_ENCODER_UPSAMPLE;
    latent.resize((size_t) latent_frames * VAE_LATENT_CHANNELS);
    for (int frame = 0; frame < latent_frames; ++frame) {
        const float value = pcm[(size_t) frame * VAE_ENCODER_UPSAMPLE * 2];
        fill_fake_latent_frame(latent, frame, value);
    }
    return latent_frames;
}

void test_vae_encode_window_boundaries() {
    using tts_cpp::acestep::VAE_AUDIO_CHUNK_FRAMES;
    using tts_cpp::acestep::VAE_AUDIO_OVERLAP_FRAMES;
    using tts_cpp::acestep::make_vae_encode_window;
    using tts_cpp::acestep::vae_encode_window_count;

    CHECK(vae_encode_window_count(VAE_AUDIO_CHUNK_FRAMES) == 1);
    CHECK(vae_encode_window_count(VAE_AUDIO_CHUNK_FRAMES + 1) == 2);

    const auto first = make_vae_encode_window(0, VAE_AUDIO_CHUNK_FRAMES + 1);
    const auto second = make_vae_encode_window(1, VAE_AUDIO_CHUNK_FRAMES + 1);
    CHECK(first.window_start == 0);
    CHECK(second.window_start == second.core_start - VAE_AUDIO_OVERLAP_FRAMES);
    CHECK(first.core_end == second.core_start);
}

void test_vae_encode_window_parity() {
    using tts_cpp::acestep::VAE_AUDIO_CHUNK_FRAMES;
    using tts_cpp::acestep::VAE_AUDIO_STRIDE_FRAMES;
    using tts_cpp::acestep::encode_vae_pcm_bounded;

    const int frames = VAE_AUDIO_CHUNK_FRAMES + VAE_AUDIO_STRIDE_FRAMES;
    std::vector<float> pcm((size_t) frames * 2);
    fill_fake_pcm(pcm, frames);

    std::vector<float> expected;
    const int expected_frames = encode_fake_vae(pcm.data(), frames, expected);
    int actual_frames = 0;
    const std::vector<float> actual =
        encode_vae_pcm_bounded(pcm, frames, encode_fake_vae, &actual_frames);

    CHECK(actual_frames == expected_frames);
    CHECK(actual == expected);
}

template <typename T>
void write_test_value(FILE * file, T value) {
    fwrite(&value, sizeof(T), 1, file);
}

void write_test_wav(FILE * file, uint16_t channels, const std::vector<int16_t> & samples) {
    constexpr uint16_t PCM_FORMAT = 1;
    constexpr uint16_t BITS = 16;
    constexpr uint32_t RATE = 48000;
    constexpr uint32_t FORMAT_SIZE = 16;

    const uint16_t block_align = channels * (BITS / 8);
    const uint32_t byte_rate = RATE * block_align;
    const uint32_t data_size = (uint32_t) (samples.size() * sizeof(int16_t));
    const uint32_t riff_size = 36 + data_size;

    fwrite("RIFF", 1, 4, file);
    write_test_value(file, riff_size);
    fwrite("WAVE", 1, 4, file);
    fwrite("fmt ", 1, 4, file);
    write_test_value(file, FORMAT_SIZE);
    write_test_value(file, PCM_FORMAT);
    write_test_value(file, channels);
    write_test_value(file, RATE);
    write_test_value(file, byte_rate);
    write_test_value(file, block_align);
    write_test_value(file, BITS);
    fwrite("data", 1, 4, file);
    write_test_value(file, data_size);
    fwrite(samples.data(), sizeof(int16_t), samples.size(), file);
}

FILE * open_test_file() {
    FILE * file = tmpfile();
    CHECK(file != nullptr);
    return file;
}

void test_wav_reader_mono_and_stereo() {
    using tts_cpp::acestep::WavReadResult;
    using tts_cpp::acestep::read_pcm16_wav;

    FILE * mono = open_test_file();
    if (!mono) return;
    write_test_wav(mono, 1, { 16384, -16384 });
    const WavReadResult mono_result = read_pcm16_wav(mono);
    fclose(mono);
    CHECK(mono_result.error.empty());
    CHECK(mono_result.frames == 2);
    CHECK(approx(mono_result.pcm[0], 0.5f));
    CHECK(approx(mono_result.pcm[1], 0.5f));

    FILE * stereo = open_test_file();
    if (!stereo) return;
    write_test_wav(stereo, 2, { 16384, -16384 });
    const WavReadResult stereo_result = read_pcm16_wav(stereo);
    fclose(stereo);
    CHECK(stereo_result.error.empty());
    CHECK(stereo_result.frames == 1);
    CHECK(approx(stereo_result.pcm[0], 0.5f));
    CHECK(approx(stereo_result.pcm[1], -0.5f));
}

void test_wav_reader_rejects_multichannel() {
    using tts_cpp::acestep::WavReadResult;
    using tts_cpp::acestep::read_pcm16_wav;

    FILE * file = open_test_file();
    if (!file) return;
    write_test_wav(file, 3, { 1, 2, 3 });
    const WavReadResult result = read_pcm16_wav(file);
    fclose(file);
    CHECK(!result.error.empty());
    CHECK(result.pcm.empty());
}

bool quality_near(double a, double b) {
    return std::fabs(a - b) < 1e-9;
}

void test_quality_normalized_pmi() {
    using tts_cpp::acestep::quality_normalized_pmi;

    CHECK(quality_near(quality_normalized_pmi(-1.0, -1.0, 0.1), 0.5));
    CHECK(quality_near(quality_normalized_pmi(-0.9, -1.0, 0.1), 1.0 / (1.0 + std::exp(-1.0))));
    CHECK(quality_near(quality_normalized_pmi(-1.1, -1.0, 0.1), 1.0 - 1.0 / (1.0 + std::exp(-1.0))));
    CHECK(quality_normalized_pmi(0.0, -100.0, 0.1) > 0.999999);
    CHECK(quality_normalized_pmi(-100.0, 0.0, 0.1) < 0.000001);
}

void test_quality_yaml_formatting() {
    using tts_cpp::acestep::quality_yaml_plain_safe;
    using tts_cpp::acestep::quality_yaml_string;

    CHECK(quality_yaml_plain_safe("C major"));
    CHECK(quality_yaml_plain_safe("4/4"));
    CHECK(quality_yaml_plain_safe("d'or"));
    CHECK(!quality_yaml_plain_safe(""));
    CHECK(!quality_yaml_plain_safe("null"));
    CHECK(!quality_yaml_plain_safe("Yes"));
    CHECK(!quality_yaml_plain_safe("120"));
    CHECK(!quality_yaml_plain_safe("-3.5"));
    CHECK(!quality_yaml_plain_safe("key: value"));
    CHECK(!quality_yaml_plain_safe(" padded"));
    CHECK(!quality_yaml_plain_safe("[verse]"));

    CHECK(quality_yaml_string("C major") == "C major");
    CHECK(quality_yaml_string("null") == "'null'");
    CHECK(quality_yaml_string("") == "''");
    CHECK(quality_yaml_string("d''") == "d''");
    CHECK(quality_yaml_string("it's: quoted") == "'it''s: quoted'");
    CHECK(quality_yaml_string("two\nlines") == "'two\n  lines'");
}

void test_quality_targets() {
    using tts_cpp::acestep::quality_caption_target;
    using tts_cpp::acestep::quality_lyrics_target;
    using tts_cpp::acestep::quality_metadata_target;

    CHECK(quality_metadata_target("bpm", 120LL) == "<think>\nbpm: 120\n</think>\n");
    CHECK(quality_metadata_target("keyscale", std::string("C major")) ==
          "<think>\nkeyscale: C major\n</think>\n");
    CHECK(quality_metadata_target("language", std::string("null")) ==
          "<think>\nlanguage: 'null'\n</think>\n");
    CHECK(quality_caption_target("test caption") == "<think>\ncaption: test caption\n</think>\n");
    CHECK(quality_caption_target("") == "<think>\ncaption: ''\n</think>\n");
    CHECK(quality_lyrics_target("[verse]\nhello") == "<think>\n</think>\n# Lyric\n[verse]\nhello\n");

    const std::string long_caption(100, 'x');
    const std::string wrapped = quality_caption_target(long_caption);
    CHECK(wrapped.rfind("<think>\ncaption: ", 0) == 0);
    CHECK(wrapped.find('\n', strlen("<think>\n")) != std::string::npos);
}

void test_quality_encode_target() {
    using tts_cpp::acestep::quality_encode_target;
    using tts_cpp::acestep::TOKEN_THINK;
    using tts_cpp::acestep::TOKEN_THINK_END;

    const tts_cpp::acestep::BpeTokenizer tok = make_test_bpe_tokenizer();
    const std::vector<int> ids = quality_encode_target(tok, "hello<think>hello</think>");
    CHECK(ids == std::vector<int>({ 10, TOKEN_THINK, 10, TOKEN_THINK_END }));
    CHECK(quality_encode_target(tok, "<think></think>") ==
          std::vector<int>({ TOKEN_THINK, TOKEN_THINK_END }));
    CHECK(quality_encode_target(tok, "hello") == std::vector<int>({ 10 }));
}

void test_quality_weighted_global() {
    using tts_cpp::acestep::QualityCondition;
    using tts_cpp::acestep::QualityMetric;
    using tts_cpp::acestep::QualityScoreParams;
    using tts_cpp::acestep::quality_weighted_global;

    std::map<std::string, QualityCondition> conditions;
    conditions["caption"].score  = 0.8;
    conditions["caption"].metric = QualityMetric::PmiNormalized;
    conditions["lyrics"].score   = 0.6;
    conditions["lyrics"].metric  = QualityMetric::PmiNormalized;
    conditions["bpm"].score      = 1.0;
    conditions["keyscale"].score = 0.5;

    QualityScoreParams params;
    double             global = 0.0;
    std::string        report;
    std::string        error;
    CHECK(quality_weighted_global(conditions, params, global, report, error));
    CHECK(quality_near(global, 0.8 * 0.5 + 0.6 * 0.3 + 0.75 * 0.2));
    CHECK(report.find("caption") != std::string::npos);
    CHECK(report.find("Per-condition scores") != std::string::npos);

    conditions.erase("lyrics");
    CHECK(quality_weighted_global(conditions, params, global, report, error));
    CHECK(quality_near(global, (0.8 * 0.5 + 0.75 * 0.2) / 0.7));

    QualityScoreParams zero_weights;
    zero_weights.caption_weight  = 0.0;
    zero_weights.lyrics_weight   = 0.0;
    zero_weights.metadata_weight = 0.0;
    CHECK(!quality_weighted_global(conditions, zero_weights, global, report, error));
    CHECK(!error.empty());
}


}  // namespace

int main() {
    test_vae_progress();
    test_coreml_sidecar_path();
    test_coreml_plan_windows();
    test_vae_window_core();
    test_normalize_loudness();
    test_lyrics_matrix_conversion();
    test_lyrics_dtw();
    test_lyrics_preprocessing();
    test_lyrics_metrics_and_score();
    test_lyrics_timestamps_and_lrc();
    test_vae_encode_window_boundaries();
    test_vae_encode_window_parity();
    test_wav_reader_mono_and_stereo();
    test_wav_reader_rejects_multichannel();
    test_quality_normalized_pmi();
    test_quality_yaml_formatting();
    test_quality_targets();
    test_quality_encode_target();
    test_quality_weighted_global();
    std::fprintf(stderr, "[test-acestep-audio] %d/%d checks passed\n", g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
