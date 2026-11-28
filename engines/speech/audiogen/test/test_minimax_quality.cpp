// Model-backed quality regression for the MiniMax-Music3 synthesis stack.
//
// Guards the two failure modes that shipped "valid WAV but noise" audio:
//   1. a flipped DiT velocity (the removed mm3.dit.output_negated readback
//      negation) — the Euler trajectory then never reaches the data manifold;
//   2. a stale DiT condition upload — the graph allocator recycles input
//      blocks between computes, so conditioning must be re-uploaded on every
//      forward or steps 1..N-1 run on garbage.
//
// Both defects leave the final flow latents far from the data manifold: the
// learned latent distribution has a per-element std of ~2.2, while a broken
// trajectory stalls near the ~1.0 std of the Gaussian noise it started from
// (measured 1.39 for both historical bugs). A healthy run must also keep the
// DiT deterministic across repeated computes with interleaved CFG branches, and
// the vocoder's overlapped tiles must decode without seams.
//
// Requires AUDIOGEN_TEST_MINIMAX_MODELS_DIR; exits 77 (ctest skip) otherwise.
// With --q4 the model directory comes from AUDIOGEN_TEST_MINIMAX_Q4_MODELS_DIR
// instead, so the same checks gate a quantized pair in a separate process
// (the static LM/DiT graph caches are not rebuilt for a second in-process load).

#include "minimax/backend.h"
#include "minimax/logic.h"
#include "minimax/mm3-pipeline.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char * expression) {
    if (condition) return;
    ++failures;
    std::fprintf(stderr, "FAIL %s\n", expression);
}

#define CHECK(condition) check((condition), #condition)

double latent_std(const std::vector<float> & latents) {
    if (latents.empty()) return 0.0;
    double sum = 0.0, sum_sq = 0.0;
    for (float value : latents) {
        sum += value;
        sum_sq += (double) value * value;
    }
    const double mean = sum / (double) latents.size();
    return std::sqrt(sum_sq / (double) latents.size() - mean * mean);
}

// Latent frames between the two equal-length tiles of the context check.
constexpr int64_t kVocoderTileShift = 64;

// How closely a GPU's tiled decode must track its single-shot decode. GPU
// backends choose GEMM kernels by product shape, so a 256-frame tile and the
// 689-frame window round differently; an RTX 5090 measures 0.99999994 on CUDA
// and 0.999996 on Vulkan, while a misplaced or cropped tile falls far below.
constexpr double kMinTiledCorrelation = 0.9999;

// Final-latent std bounds. The lower one separates a healthy trajectory from
// the stalled ones described above. The upper one catches a blow-up; the Q4
// pair lands higher than f16 (3.08-3.23 across ggml builds on Metal, against
// 2.1-2.2), so it gets its own.
constexpr double kMinFinalLatentStd = 1.7;
constexpr double kMaxFinalLatentStdF16 = 3.2;
constexpr double kMaxFinalLatentStdQ4 = 3.6;

double max_final_latent_std(bool q4_run) {
    return q4_run ? kMaxFinalLatentStdQ4 : kMaxFinalLatentStdF16;
}

bool decode_vocoder_window(const MM3Model & model, const std::vector<float> & latents, int64_t L,
                           std::vector<float> & audio, std::string * error) {
    const int64_t channels = (int64_t) model.synth_cfg.voc.channels;
    audio.assign((size_t) (channels * L * (int64_t) model.synth_cfg.voc.total_upsample), 0.0f);
    return mm3_vocoder_decode_tiled(model, latents, L, audio, L, MM3_VOC_OVERLAP, error);
}

// Frames [start, start + width) of every row of planar [rows][L] latents.
std::vector<float> latent_window(const std::vector<float> & latents, int64_t L, int64_t rows, int64_t start,
                                 int64_t width) {
    std::vector<float> window((size_t) (rows * width));
    for (int64_t row = 0; row < rows; row++) {
        std::memcpy(window.data() + row * width, latents.data() + row * L + start, (size_t) width * sizeof(float));
    }
    return window;
}

double waveform_correlation(const std::vector<float> & a, const std::vector<float> & b) {
    double ab = 0.0, aa = 0.0, bb = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        ab += (double) a[i] * b[i];
        aa += (double) a[i] * a[i];
        bb += (double) b[i] * b[i];
    }
    return ab / std::sqrt(aa * bb);
}

// The tiled vocoder decode must reproduce a single-shot decode of the whole
// window: bit for bit on the CPU, whose kernels compute every element the same
// way at any length, and to rounding on a GPU, whose GEMM kernels depend on the
// product shape. vocoder_tile_interiors_ignore_context checks the seams exactly.
bool vocoder_tiling_matches_single_shot(const MM3Model & model, std::string * error) {
    const int64_t L  = (int64_t) model.synth_cfg.dit.window_latents;
    const int64_t FC = (int64_t) model.synth_cfg.voc.fold_channels;
    if (L <= MM3_VOC_CHUNK) {
        *error = "window_latents does not exercise the tiled path";
        return false;
    }
    std::vector<float> latents;
    tts_cpp::minimax::detail::fill_noise(11, 0, latents, L * 2 * FC);

    std::vector<float> single, tiled;
    if (!mm3_vocoder_prepare(model, &g_mm3_voc, error) || !decode_vocoder_window(model, latents, L, single, error)) {
        return false;
    }
    tiled.assign(single.size(), 0.0f);
    if (!mm3_vocoder_decode_tiled(model, latents, L, tiled, MM3_VOC_CHUNK, MM3_VOC_OVERLAP, error)) {
        return false;
    }
    if (!g_backend_cache.has_gpu) {
        if (std::memcmp(single.data(), tiled.data(), single.size() * sizeof(float)) != 0) {
            *error = "tiled vocoder output differs from single-shot output on the CPU";
            return false;
        }
        return true;
    }
    const double correlation = waveform_correlation(single, tiled);
    std::fprintf(stderr, "[quality] tiled vs single-shot vocoder correlation %.9f\n", correlation);
    if (!(correlation >= kMinTiledCorrelation)) {
        *error = "tiled vocoder output correlates with single-shot output at " + std::to_string(correlation);
        return false;
    }
    return true;
}

// A tile's output must not depend on anything more than MM3_VOC_OVERLAP frames
// beyond it, or the tiled decode would carry seams. Two tiles of the production
// length, shifted against each other, run the same kernels on every backend, so
// their shared interior must match bit for bit wherever the overlap covers the
// conv stack's receptive field.
bool vocoder_tile_interiors_ignore_context(const MM3Model & model, std::string * error) {
    const int64_t L        = (int64_t) model.synth_cfg.dit.window_latents;
    const int64_t channels = (int64_t) model.synth_cfg.voc.channels;
    const int64_t rows     = channels * (int64_t) model.synth_cfg.voc.fold_channels;
    const int64_t up       = (int64_t) model.synth_cfg.voc.total_upsample;
    const int64_t width    = MM3_VOC_CHUNK;
    const int64_t interior = width - kVocoderTileShift - 2 * MM3_VOC_OVERLAP;
    if (L < width + kVocoderTileShift || interior <= 0) {
        *error = "window_latents is too short for two shifted vocoder tiles";
        return false;
    }
    std::vector<float> latents;
    tts_cpp::minimax::detail::fill_noise(13, 0, latents, L * rows);

    std::vector<float> first, shifted;
    if (!mm3_vocoder_prepare(model, &g_mm3_voc, error) ||
        !decode_vocoder_window(model, latent_window(latents, L, rows, 0, width), width, first, error) ||
        !decode_vocoder_window(model, latent_window(latents, L, rows, kVocoderTileShift, width), width, shifted,
                               error)) {
        return false;
    }
    const size_t channel = (size_t) (width * up);
    const size_t samples = (size_t) (interior * up) * sizeof(float);
    for (size_t ch = 0; ch < (size_t) channels; ch++) {
        const float * a = first.data() + ch * channel + (size_t) ((kVocoderTileShift + MM3_VOC_OVERLAP) * up);
        const float * b = shifted.data() + ch * channel + (size_t) (MM3_VOC_OVERLAP * up);
        if (std::memcmp(a, b, samples) != 0) {
            *error = "vocoder tile interiors depend on context beyond MM3_VOC_OVERLAP frames";
            return false;
        }
    }
    return true;
}

bool dit_is_deterministic_across_computes(const MM3Model & model, std::string * error) {
    const int64_t L = 64;
    const int64_t N = (int64_t) model.synth_cfg.dit.in_channels * L;
    const int64_t CN = (int64_t) model.synth_cfg.dit.condition_dim * L;
    std::vector<float> latents, condition;
    tts_cpp::minimax::detail::fill_noise(7, 0, latents, N);
    tts_cpp::minimax::detail::fill_noise(7, 1, condition, CN);
    std::vector<float> first((size_t) N), first_unconditional((size_t) N);
    std::vector<float> swapped((size_t) N), swapped_unconditional((size_t) N);
    std::vector<float> repeat((size_t) N), repeat_unconditional((size_t) N);
    const std::vector<MM3DitBranch> swapped_branches = {
        { MM3_DIT_UNCONDITIONED_GATE, swapped_unconditional.data() },
        { MM3_DIT_CONDITIONED_GATE, swapped.data() },
    };
    if (!mm3_dit_prepare(model, &g_mm3_dit, error) ||
        !mm3_dit_run_cfg(model, &g_mm3_dit, latents.data(), condition.data(), 0.5f, L, first.data(),
                         first_unconditional.data(), error) ||
        !mm3_dit_run(model, &g_mm3_dit, latents.data(), condition.data(), swapped_branches, 0.5f, L, error) ||
        !mm3_dit_run_cfg(model, &g_mm3_dit, latents.data(), condition.data(), 0.5f, L, repeat.data(),
                         repeat_unconditional.data(), error)) {
        return false;
    }
    if (first != repeat || first_unconditional != repeat_unconditional ||
        first != swapped || first_unconditional != swapped_unconditional) {
        if (error) {
            *error = "DiT output changed between identical computes";
        }
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    const bool   q4_run  = argc > 1 && std::strcmp(argv[1], "--q4") == 0;
    const char * env_var = q4_run ? "AUDIOGEN_TEST_MINIMAX_Q4_MODELS_DIR" : "AUDIOGEN_TEST_MINIMAX_MODELS_DIR";
    const char * models_dir = std::getenv(env_var);
    if (!models_dir || !*models_dir) {
        std::fprintf(stderr, "SKIP set %s to run\n", env_var);
        return 77;
    }

    backend_configure_cpu(4, std::getenv("AUDIOGEN_TEST_BACKENDS_DIR") ? std::getenv("AUDIOGEN_TEST_BACKENDS_DIR") : "");
    backend_configure_device("");  // cpu unless MM3_DEVICE overrides

    MM3Model model;
    const tts_cpp::minimax::detail::ModelPair pair =
        tts_cpp::minimax::detail::resolve_model_pair(models_dir, "", "");
    if (q4_run && pair.quant != "q4_k_m" && pair.quant != "q4_k_s") {
        std::fprintf(stderr, "FAIL --q4 selected %s; provide a directory containing only the Q4 pair\n",
                     pair.quant.c_str());
        return 1;
    }
    model.models_dir = models_dir;
    mm3_probe_file(pair.lm, &model.lm_file, &model.lm_cfg, nullptr, &model.meta_errors);
    mm3_probe_file(pair.synth, &model.synth_file, nullptr, &model.synth_cfg, &model.meta_errors);
    CHECK(model.meta_errors.empty());
    std::string error;
    if (!mm3_load(&model, &error)) {
        std::fprintf(stderr, "FAIL mm3_load: %s\n", error.c_str());
        return 1;
    }

    if (!dit_is_deterministic_across_computes(model, &error)) {
        std::fprintf(stderr, "FAIL dit determinism: %s\n", error.c_str());
        ++failures;
    }

    if (!vocoder_tiling_matches_single_shot(model, &error)) {
        std::fprintf(stderr, "FAIL vocoder tiling: %s\n", error.c_str());
        ++failures;
    }

    if (!vocoder_tile_interiors_ignore_context(model, &error)) {
        std::fprintf(stderr, "FAIL vocoder tile context: %s\n", error.c_str());
        ++failures;
    }

    MM3GenRequest request;
    request.prompt = tts_cpp::minimax::detail::build_prompt("A short warm piano note.", "");
    request.max_frames = 48;
    request.seed = 7;
    request.steps = 30;
    request.cfg_flow = model.synth_cfg.flow.cfg_scale > 0 ? model.synth_cfg.flow.cfg_scale : 1.7f;
    request.keep_window_latents = true;

    MM3Tokenizer tokenizer;
    MM3GenResult result;
    if (!mm3_generate(model, request, &tokenizer, nullptr, &result, &error)) {
        std::fprintf(stderr, "FAIL mm3_generate: %s\n", error.c_str());
        mm3_unload(&model);
        return 1;
    }

    CHECK(result.frames > 0);
    CHECK(result.n_windows == 1);
    CHECK(result.n_samples > 0);
    CHECK(!result.ids_cond_used.empty());
    CHECK(!result.has_nan);
    CHECK(result.peak > 0.005f);
    CHECK(result.rms > 0.001);
    CHECK(result.window_latents.size() == 1);

    // On-manifold check: a correct flow trajectory ends at the learned latent
    // distribution (std ~2.2 for 300-frame songs, ~1.9 for this short prompt);
    // the historical velocity-sign and stale-condition bugs both stalled near
    // the noise distribution (std ~1.4).
    const double std_dev = latent_std(result.window_latents[0]);
    std::fprintf(stderr, "[quality] final latent std %.4f (accepted %.1f-%.1f, broken ~1.4)\n", std_dev,
                 kMinFinalLatentStd, max_final_latent_std(q4_run));
    CHECK(std_dev > kMinFinalLatentStd);
    CHECK(std_dev < max_final_latent_std(q4_run));

    mm3_unload(&model);
    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr, "OK\n");
    return 0;
}
