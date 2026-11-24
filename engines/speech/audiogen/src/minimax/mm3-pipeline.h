#pragma once

#include "mm3-ar-loop.h"
#include "mm3-cond-graph.h"
#include "mm3-dit-graph.h"
#include "mm3-model.h"
#include "mm3-tokenizer.h"
#include "mm3-vocoder-graph.h"
#include "mm3-window-orchestrator.h"
#include "logic.h"
#include "progress.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

static bool mm3_flow_sample_chunk(const MM3Model & m, const float * noise, const float * cond, int64_t L, int steps,
                                  float cfg_scale, int64_t overlap, const float * prev_latent, int64_t prev_stride,
                                  std::vector<float> & out_latents, MM3FlowStats * stats,
                                  const std::function<void(int, int)> & on_step,
                                  const std::function<bool()> & should_cancel, std::string * err) {
    if (L <= 0 || L > MM3_DIT_MAX_FRAMES) {
        if (err) {
            *err = "frames must be in 1.." + std::to_string(MM3_DIT_MAX_FRAMES);
        }
        return false;
    }
    if (steps <= 0 || steps > 1000) {
        if (err) {
            *err = "steps must be in 1..1000";
        }
        return false;
    }
    if (overlap < 0 || overlap > L) {
        if (err) {
            *err = "overlap must be in 0..L";
        }
        return false;
    }
    if (overlap > 0 && (!prev_latent || prev_stride < overlap)) {
        if (err) {
            *err = "a positive overlap needs a previous-window carry at least that long";
        }
        return false;
    }
    if (!mm3_dit_prepare(m, &g_mm3_dit, err)) {
        return false;
    }

    const int64_t C = (int64_t) m.synth_cfg.dit.in_channels;
    const int64_t N = C * L;
    out_latents.assign((size_t) N, 0.0f);
    memcpy(out_latents.data(), noise, (size_t) N * sizeof(float));

    std::vector<float> sigmas, timesteps;
    mm3_flow_sigmas(steps, &sigmas, &timesteps);

    std::vector<float> pred_c((size_t) N);
    std::vector<float> pred_u((size_t) N);

    const auto t_all  = std::chrono::steady_clock::now();
    double     fwd_ms = 0.0, first_ms = 0.0, last_ms = 0.0;

    for (int i = 0; i < steps; i++) {
        if (tts_cpp::minimax::detail::cancellation_requested(should_cancel)) {
            if (err) {
                *err = MM3_ERR_CANCELLED;
            }
            return false;
        }
        const float t = timesteps[(size_t) i];

        if (overlap > 0) {
            tts_cpp::minimax::detail::blend_latent_overlap(
                out_latents.data(), noise, prev_latent, C, L, overlap, prev_stride, t);
        }

        const auto t0 = std::chrono::steady_clock::now();

        // The condition tensor must be re-uploaded on every forward: the graph
        // allocator recycles input blocks for intermediates once their last
        // consumer has run, so data left in mm3_dit_cond does not survive a
        // graph compute.
        if (!mm3_dit_run_cfg(m, &g_mm3_dit, out_latents.data(), cond, t, L, pred_c.data(), pred_u.data(), err)) {
            return false;
        }

        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        fwd_ms += ms;
        if (i == 0) {
            first_ms = ms;
        }
        last_ms = ms;

        if (!mm3_integrate_flow_step(out_latents, pred_c, pred_u, sigmas,
                                     static_cast<size_t>(i), cfg_scale, err)) {
            return false;
        }

        if (on_step) {
            on_step(i + 1, steps);
        }

        if (should_cancel && should_cancel()) {
            if (err) {
                *err = MM3_ERR_CANCELLED;
            }
            return false;
        }
    }

    if (overlap > 0) {
        tts_cpp::minimax::detail::pin_latent_overlap(
            out_latents.data(), prev_latent, C, L, overlap, prev_stride);
    }

    const double total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_all).count();
    if (stats) {
        stats->steps         = steps;
        stats->forwards      = steps * 2;
        stats->total_ms      = total_ms;
        stats->forward_ms    = fwd_ms;
        stats->first_ms      = first_ms;
        stats->last_ms       = last_ms;
        stats->compute_bytes = g_mm3_dit.compute_bytes;
    }
    fprintf(stderr,
            "[MM3-Flow] Window: L=%lld, ov=%lld, %d steps, cfg %.2f -> %.0f ms (%.0f ms/step, %.0f ms/forward)\n",
            (long long) L, (long long) overlap, steps, (double) cfg_scale, total_ms, total_ms / (double) steps,
            fwd_ms / (double) (steps * 2));
    return true;
}

struct MM3GenRequest {

    std::string          prompt;
    std::vector<int32_t> ids_cond;
    std::vector<int32_t> ids_uncond;

    int64_t  max_frames = 300;
    uint64_t seed       = 42;
    int      steps      = 30;
    float    cfg_flow   = 1.7f;

    std::vector<int32_t> forced_semantic;
    std::vector<int32_t> forced_acoustic;

    std::vector<std::vector<float>> forced_noise;

    int64_t dump_iters = 0;

    bool keep_window_latents = false;

    std::function<bool()> should_cancel;
};

struct MM3GenResult {
    std::vector<float> audio;
    int64_t            n_samples = 0;
    int                sample_rate = 0;

    int64_t              frames    = 0;
    int64_t              n_windows = 0;
    std::vector<int64_t> chunk_starts;
    std::vector<int64_t> chunk_frames;
    std::vector<int64_t> window_L;
    std::vector<int64_t> window_overlap;
    std::vector<int64_t> forced_noise_used;

    // The conditional prompt ids the run actually used (tokenized from the
    // prompt when ids_cond was empty), so a full run records the replay input.
    std::vector<int32_t> ids_cond_used;

    std::vector<std::vector<float>> window_latents;

    MM3ArResult ar;

    double  ar_ms    = 0.0;
    double  cond_ms  = 0.0;
    double  flow_ms  = 0.0;
    double  voc_ms   = 0.0;
    double  total_ms = 0.0;
    int64_t flow_forwards = 0;

    size_t dit_compute_bytes = 0;

    bool  has_nan = false;
    float peak    = 0.0f;
    double rms    = 0.0;
};

struct MM3PipelineDimensions {
    int64_t acoustic_codebooks = 0;
    int64_t hidden = 0;
    int64_t layers = 0;
    int64_t channels = 0;
    int64_t upsample = 0;
    int64_t window_frames = 0;
    int64_t hop_frames = 0;
};

static bool mm3_fail(std::string * error, const std::string & message) {
    if (error) {
        *error = message;
    }
    return false;
}

static MM3PipelineDimensions mm3_pipeline_dimensions(const MM3Model & model) {
    MM3PipelineDimensions dimensions;
    dimensions.acoustic_codebooks = static_cast<int64_t>(model.lm_cfg.num_codebooks) - 1;
    dimensions.hidden = static_cast<int64_t>(model.lm_cfg.embedding_length);
    dimensions.layers = static_cast<int64_t>(model.lm_cfg.num_codebooks);
    dimensions.channels = static_cast<int64_t>(model.synth_cfg.dit.in_channels);
    dimensions.upsample = static_cast<int64_t>(model.synth_cfg.voc.total_upsample);
    dimensions.window_frames = static_cast<int64_t>(model.synth_cfg.dit.window_frames);
    dimensions.hop_frames = static_cast<int64_t>(model.synth_cfg.dit.hop_frames);
    return dimensions;
}

static bool mm3_prepare_token_ids(const MM3Model & model, const MM3GenRequest & request,
                                  MM3Tokenizer * tokenizer, std::vector<int32_t> & conditional,
                                  std::vector<int32_t> & unconditional, std::string * error) {
    conditional = request.ids_cond;
    unconditional = request.ids_uncond;
    if (conditional.empty()) {
        if (request.prompt.empty()) {
            return mm3_fail(error, "need either a prompt or ids_cond");
        }
        if (!tokenizer || !mm3_tokenizer_load(model, tokenizer, error)) {
            return false;
        }
        mm3_tokenizer_encode(*tokenizer, request.prompt, &conditional);
    }
    if (conditional.size() < 3) {
        return mm3_fail(error, "the prompt must tokenise to at least 3 tokens");
    }
    if (conditional.size() > model.lm_cfg.max_prompt_tokens) {
        return mm3_fail(error, "the prompt exceeds the model token limit");
    }
    if (unconditional.empty()) {
        mm3_tokenizer_uncond(model.lm_cfg, conditional, &unconditional);
    }
    if (unconditional.size() != conditional.size()) {
        return mm3_fail(error, "ids_uncond must be the same length as ids_cond");
    }
    return true;
}

static bool mm3_prepare_ar_options(const MM3GenRequest & request, int64_t acoustic_codebooks,
                                   const MM3ProgressCb & progress, MM3ArOptions & options,
                                   std::string * error) {
    options.max_frames = request.max_frames;
    options.seed = request.seed;
    options.collect_hiddens = true;
    options.dump_iters = request.dump_iters;
    options.should_cancel = request.should_cancel;
    if (!request.forced_semantic.empty()) {
        const int64_t expected =
            static_cast<int64_t>(request.forced_semantic.size()) * acoustic_codebooks;
        if (static_cast<int64_t>(request.forced_acoustic.size()) != expected) {
            return mm3_fail(error, "forced_acoustic must contain " +
                                       std::to_string(acoustic_codebooks) +
                                       " entries per semantic token");
        }
        options.forced_semantic = request.forced_semantic.data();
        options.forced_acoustic = request.forced_acoustic.data();
        options.forced_len = static_cast<int64_t>(request.forced_semantic.size());
    }
    if (progress) {
        options.on_frame = [progress](int64_t frame, int64_t total) {
            progress(MM3GenProgress{"ar", -1, 0, frame, total});
        };
    }
    return true;
}

static bool mm3_run_ar_stage(const MM3Model & model, const MM3GenRequest & request,
                             const MM3PipelineDimensions & dimensions,
                             const std::vector<int32_t> & conditional,
                             const std::vector<int32_t> & unconditional,
                             const MM3ProgressCb & progress, MM3GenResult * result,
                             std::string * error) {
    MM3ArOptions options;
    if (!mm3_prepare_ar_options(request, dimensions.acoustic_codebooks, progress, options, error)) {
        return false;
    }
    const auto started = std::chrono::steady_clock::now();
    if (!mm3_ar_plan(model, conditional.data(), unconditional.data(),
                     static_cast<int64_t>(conditional.size()), options, &result->ar, error)) {
        return false;
    }
    result->ar_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    result->frames = result->ar.n_frames;
    if (result->frames <= 0) {
        return mm3_fail(error, "the AR stage emitted zero frames");
    }
    const int64_t expected = result->frames * dimensions.layers * dimensions.hidden;
    if (static_cast<int64_t>(result->ar.frame_hiddens.size()) != expected) {
        return mm3_fail(error, "the AR stage returned a frame-hidden block of the wrong size");
    }
    // The synth stages never touch the LM again; releasing its KV cache and
    // sched compute buffers here frees a few hundred MiB before the DiT and
    // vocoder allocate, which is what lets the whole pipeline fit a 10 GiB
    // GPU. mm3_lm_prepare rebuilds everything on the next generation.
    mm3_lm_free(&g_mm3_lm);
    return true;
}

static bool mm3_prepare_window_noise(const MM3GenRequest & request, int64_t window,
                                     int64_t count, std::vector<float> & noise) {
    if (static_cast<int64_t>(request.forced_noise.size()) > window &&
        static_cast<int64_t>(request.forced_noise[static_cast<size_t>(window)].size()) == count) {
        noise = request.forced_noise[static_cast<size_t>(window)];
        return true;
    }
    tts_cpp::minimax::detail::fill_noise(request.seed, window, noise, count);
    return false;
}

static bool mm3_finish_generation(
    const std::chrono::steady_clock::time_point & started, MM3GenResult * result) {
    result->total_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    fprintf(stderr,
            "[MM3-Pipe] %lld frames -> %lld window(s) -> %lld samples/ch (%.2fs @ %d Hz) | "
            "AR %.0f ms, cond %.0f ms, flow %.0f ms, voc %.0f ms, total %.0f ms\n",
            static_cast<long long>(result->frames),
            static_cast<long long>(result->n_windows),
            static_cast<long long>(result->n_samples),
            static_cast<double>(result->n_samples) /
                static_cast<double>(result->sample_rate > 0 ? result->sample_rate : 1),
            result->sample_rate, result->ar_ms, result->cond_ms, result->flow_ms,
            result->voc_ms, result->total_ms);
    return true;
}

static MM3WindowDimensions mm3_window_dimensions(
    const MM3Model & model, const MM3GenRequest & request,
    const MM3PipelineDimensions & dimensions) {
    MM3WindowDimensions window;
    window.latent_channels = dimensions.channels;
    window.audio_channels = static_cast<int64_t>(model.synth_cfg.voc.channels);
    window.condition_dimension =
        static_cast<int64_t>(model.synth_cfg.cond.out_dim);
    window.upsample = dimensions.upsample;
    window.window_frames = dimensions.window_frames;
    window.hop_frames = dimensions.hop_frames;
    const auto geometry = tts_cpp::minimax::detail::flow_window_geometry(
        static_cast<int64_t>(model.synth_cfg.dit.window_latents),
        static_cast<int64_t>(model.synth_cfg.dit.hop_latents));
    window.carry_span = geometry.carry_span;
    window.overlap = geometry.overlap;
    window.crop_left = geometry.crop_left;
    window.crop_right = geometry.crop_right;
    window.flow_steps = request.steps;
    return window;
}

static MM3WindowOperations mm3_window_operations(
    const MM3Model & model, const MM3GenRequest & request,
    const MM3ProgressCb & progress, MM3GenResult * result) {
    MM3WindowOperations operations;
    operations.progress =
        [&request, &progress](const std::string & stage, int64_t window,
                             int64_t window_count, int64_t current, int64_t total,
                             std::string * error) {
            const MM3GenProgress state{
                stage.c_str(), window, window_count, current, total};
            return mm3_emit_progress(progress, state, request.should_cancel, error);
        };
    operations.continue_generation =
        [&request](std::string * error) {
            return mm3_continue_generation(request.should_cancel, error);
        };
    operations.condition =
        [&model, result](int64_t, int64_t, int64_t start, int64_t frames,
                         std::vector<float> & condition, int64_t & latent_length,
                         std::string * error) {
            const auto started = std::chrono::steady_clock::now();
            const int64_t layers = static_cast<int64_t>(model.lm_cfg.num_codebooks);
            const int64_t hidden =
                static_cast<int64_t>(model.lm_cfg.embedding_length);
            const size_t offset = static_cast<size_t>(start * layers * hidden);
            if (!mm3_cond_encode(model, result->ar.frame_hiddens.data() + offset,
                                 frames, condition, &latent_length, error)) {
                return false;
            }
            result->cond_ms += std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - started)
                                   .count();
            return true;
        };
    operations.noise =
        [&request](int64_t window, int64_t count, std::vector<float> & noise) {
            return mm3_prepare_window_noise(request, window, count, noise);
        };
    operations.flow =
        [&model, &request, result](
            int64_t, int64_t, int64_t latent_length,
            const std::vector<float> & noise,
            const std::vector<float> & carry_latents, int64_t carry_length,
            const std::vector<float> & condition, std::vector<float> & latents,
            const std::function<void(int64_t, int64_t)> & on_step,
            std::string * error) {
            MM3FlowStats stats;
            const auto step = [&on_step](int current, int total) {
                on_step(current, total);
            };
            const float * previous =
                carry_length > 0 ? carry_latents.data() : nullptr;
            if (!mm3_flow_sample_chunk(
                    model, noise.data(), condition.data(), latent_length, request.steps,
                    request.cfg_flow, std::min(carry_length, latent_length), previous,
                    carry_length, latents, &stats, step, request.should_cancel, error)) {
                return false;
            }
            result->flow_ms += stats.total_ms;
            result->flow_forwards += stats.forwards;
            result->dit_compute_bytes = stats.compute_bytes;
            return true;
        };
    operations.vocoder =
        [&model, result](int64_t, int64_t, const std::vector<float> & latents,
                         int64_t latent_length, std::vector<float> & waveform,
                         std::string * error) {
            const auto started = std::chrono::steady_clock::now();
            if (!mm3_vocoder_decode(model, latents, latent_length, waveform, error)) {
                return false;
            }
            result->voc_ms += std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - started)
                                  .count();
            return true;
        };
    return operations;
}

static void mm3_apply_window_orchestration(
    const MM3GenRequest & request, MM3WindowOrchestration & orchestration,
    MM3GenResult * result) {
    result->chunk_starts = orchestration.starts;
    result->chunk_frames = orchestration.frame_lengths;
    result->window_L = orchestration.latent_lengths;
    result->window_overlap = orchestration.overlaps;
    result->forced_noise_used = orchestration.forced_noise;
    result->n_windows = static_cast<int64_t>(orchestration.starts.size());
    result->n_samples = orchestration.samples_per_channel;
    result->audio = std::move(orchestration.audio);
    result->peak = orchestration.metrics.peak;
    result->rms = orchestration.metrics.rms;
    if (request.keep_window_latents) {
        result->window_latents = std::move(orchestration.latents);
    }
}

static bool mm3_generate(const MM3Model & model, const MM3GenRequest & request,
                         MM3Tokenizer * tokenizer, const MM3ProgressCb & progress,
                         MM3GenResult * result, std::string * error) {
    const auto started = std::chrono::steady_clock::now();
    const MM3PipelineDimensions dimensions = mm3_pipeline_dimensions(model);
    *result = MM3GenResult{};
    result->sample_rate = static_cast<int>(model.synth_cfg.voc.sampling_rate);
    std::vector<int32_t> conditional;
    std::vector<int32_t> unconditional;
    if (!mm3_prepare_token_ids(model, request, tokenizer, conditional, unconditional, error)) {
        return false;
    }
    result->ids_cond_used = conditional;
    if (!mm3_run_ar_stage(model, request, dimensions, conditional, unconditional,
                          progress, result, error)) {
        return false;
    }
    MM3WindowOrchestration orchestration;
    const auto window_dimensions = mm3_window_dimensions(model, request, dimensions);
    const auto operations =
        mm3_window_operations(model, request, progress, result);
    if (!mm3_orchestrate_windows(result->frames, window_dimensions, operations,
                                 &orchestration, error)) {
        return false;
    }
    mm3_apply_window_orchestration(request, orchestration, result);
    return mm3_finish_generation(started, result);
}
