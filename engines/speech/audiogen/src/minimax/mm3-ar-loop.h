#pragma once

#include "logic.h"
#include "mm3-depth-graph.h"
#include "mm3-lm-graph.h"
#include "mm3-model.h"
#include "mm3-sample.h"
#include "mm3-tokenizer.h"
#include "progress.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

struct MM3ArDump {
    std::vector<float> last_hidden;
    std::vector<float> sem_logits;
    std::vector<float> guided;
    std::vector<float> feedback;
    std::vector<float> depth_hidden;
};

struct MM3ArResult {
    int64_t n_iterations = 0;
    int64_t n_frames = 0;
    int64_t hidden_dim = 0;
    int64_t n_codebooks = 0;
    int64_t sem_vocab = 0;
    bool eos_hit = false;

    std::vector<int32_t> semantic_all;
    std::vector<int32_t> acoustic_all;
    std::vector<float> frame_hiddens;
    std::vector<float> prefill_hidden;
    std::vector<MM3ArDump> dumps;

    int64_t nonfinite_logits = 0;
    double prefill_ms = 0.0;
    double lm_ms = 0.0;
    double depth_ms = 0.0;
    double host_ms = 0.0;
    double total_ms = 0.0;
    int64_t lm_steps = 0;
};

struct MM3ArOptions {
    int64_t max_frames = 300;
    uint64_t seed = 42;

    const int32_t * forced_semantic = nullptr;
    const int32_t * forced_acoustic = nullptr;
    int64_t forced_len = 0;

    int64_t dump_iters = 0;
    bool collect_hiddens = true;

    std::function<void(int64_t, int64_t)> on_frame;
    std::function<bool()> should_cancel;
};

struct MM3ArConfig {
    int64_t hidden = 0;
    int64_t vocab = 0;
    int64_t semantic_vocab = 0;
    int64_t compact_vocab = 0;
    int64_t acoustic_codebooks = 0;
    int64_t acoustic_vocab = 0;
    int64_t semantic_offset = 0;
    int64_t eos = 0;
    int64_t max_frames = 0;
    float cfg_scale = 0.0f;
    int top_k = 0;
    bool forced = false;
};

struct MM3ArWorkspace {
    explicit MM3ArWorkspace(uint64_t seed) : random(seed) {
    }

    std::vector<float> hidden;
    std::vector<float> logits;
    std::vector<float> feedback;
    tts_cpp::minimax::detail::ArCandidates candidates;
    std::vector<float> sampling_scratch;
    std::vector<int32_t> acoustic_rows;
    MM3DepthFrame frame;
    std::mt19937_64 random;
};

enum class MM3ArStep {
    proceed,
    stop,
    failure,
};

static MM3LmGraph g_mm3_lm;

static bool mm3_ar_fail(std::string * error, const std::string & message) {
    if (error) {
        *error = message;
    }
    return false;
}

static bool mm3_ar_cancelled(const MM3ArOptions & options, std::string * error) {
    if (!tts_cpp::minimax::detail::cancellation_requested(options.should_cancel)) {
        return false;
    }
    if (error) {
        *error = MM3_ERR_CANCELLED;
    }
    return true;
}

static bool mm3_ar_validate_prompt(const MM3LmConfig & config, int64_t prompt_length,
                                   std::string * error) {
    if (prompt_length <= 0) {
        return mm3_ar_fail(error, "the prompt tokenised to zero tokens");
    }
    if (config.max_prompt_tokens > 0 &&
        prompt_length > static_cast<int64_t>(config.max_prompt_tokens)) {
        return mm3_ar_fail(
            error, "the prompt is " + std::to_string(prompt_length) +
                       " tokens; the checkpoint's limit is " +
                       std::to_string(config.max_prompt_tokens));
    }
    return true;
}

static bool mm3_ar_validate_vocabulary(const MM3ArConfig & resolved,
                                       std::string * error) {
    if (resolved.eos < 0 || resolved.eos >= resolved.vocab ||
        resolved.semantic_offset < 0 ||
        resolved.semantic_offset >
            resolved.vocab - resolved.semantic_vocab) {
        return mm3_ar_fail(
            error,
            "the LM vocabulary metadata does not cover the semantic range and EOS");
    }
    return true;
}

static bool mm3_ar_resolve_frame_cap(const MM3LmConfig & config,
                                     const MM3ArOptions & options,
                                     MM3ArConfig & resolved, std::string * error) {
    if (resolved.forced && !options.forced_acoustic) {
        return mm3_ar_fail(
            error,
            "forced replay needs both forced_semantic and forced_acoustic, and a positive length");
    }
    std::string resolution_error;
    resolved.max_frames = tts_cpp::minimax::detail::resolve_ar_frame_cap(
        options.max_frames, static_cast<int64_t>(config.max_audio_frames),
        resolved.forced, options.forced_len, resolution_error);
    if (!resolution_error.empty()) {
        return mm3_ar_fail(error, resolution_error);
    }
    return true;
}

static bool mm3_ar_resolve_config(const MM3Model & model,
                                  const MM3ArOptions & options,
                                  int64_t prompt_length, MM3ArConfig & resolved,
                                  std::string * error) {
    const MM3LmConfig & config = model.lm_cfg;
    resolved.hidden = static_cast<int64_t>(config.embedding_length);
    resolved.vocab = static_cast<int64_t>(config.vocab_size);
    resolved.semantic_vocab =
        static_cast<int64_t>(config.semantic_vocab_size);
    resolved.compact_vocab =
        tts_cpp::minimax::detail::compact_head_row_count(resolved.semantic_vocab);
    resolved.acoustic_codebooks =
        static_cast<int64_t>(config.num_codebooks) - 1;
    resolved.acoustic_vocab =
        static_cast<int64_t>(config.acoustic_vocab_size);
    resolved.semantic_offset =
        static_cast<int64_t>(config.semantic_vocab_offset);
    resolved.eos = static_cast<int64_t>(config.eos_audio);
    resolved.cfg_scale =
        config.ar_cfg_scale > 0.0f ? config.ar_cfg_scale : 1.5f;
    resolved.top_k = config.ar_top_k > 0 ? static_cast<int>(config.ar_top_k) : 50;
    resolved.forced = options.forced_semantic != nullptr;
    if (!mm3_ar_validate_prompt(config, prompt_length, error) ||
        !mm3_ar_validate_vocabulary(resolved, error) ||
        !mm3_ar_resolve_frame_cap(config, options, resolved, error)) {
        return false;
    }
    if (!mm3_lm_positions_fit(config.context_length, prompt_length,
                              resolved.max_frames)) {
        return mm3_ar_fail(
            error, "the prompt and generated frames exceed qwen3.context_length");
    }
    return true;
}

static void mm3_ar_initialize_result(const MM3ArConfig & config,
                                     const MM3ArOptions & options,
                                     MM3ArResult * result) {
    *result = MM3ArResult{};
    result->hidden_dim = config.hidden;
    result->n_codebooks = config.acoustic_codebooks + 1;
    result->sem_vocab = config.semantic_vocab;
    result->semantic_all.reserve(
        static_cast<size_t>(config.max_frames + 1));
    result->acoustic_all.reserve(static_cast<size_t>(
        (config.max_frames + 1) * config.acoustic_codebooks));
    if (options.collect_hiddens) {
        result->frame_hiddens.reserve(static_cast<size_t>(
            config.max_frames * (config.acoustic_codebooks + 1) *
            config.hidden));
    }
}

static void mm3_ar_initialize_workspace(const MM3ArConfig & config,
                                        MM3ArWorkspace & workspace) {
    workspace.hidden.resize(
        static_cast<size_t>(config.hidden * MM3_LM_CFG_ROWS));
    workspace.logits.resize(
        static_cast<size_t>(config.compact_vocab * MM3_LM_CFG_ROWS));
    workspace.feedback.resize(static_cast<size_t>(config.hidden));
    workspace.acoustic_rows.resize(
        static_cast<size_t>(config.acoustic_codebooks));
}

static bool mm3_ar_prefill(const MM3Model & model, const int32_t * conditional,
                           const int32_t * unconditional, int64_t prompt_length,
                           const MM3ArOptions & options, MM3ArWorkspace & workspace,
                           MM3ArResult * result, std::string * error) {
    if (mm3_ar_cancelled(options, error)) {
        return false;
    }
    const auto started = std::chrono::steady_clock::now();
    if (!mm3_lm_prefill(model, &g_mm3_lm, conditional, unconditional,
                        prompt_length, workspace.hidden.data(),
                        workspace.logits.data(), error)) {
        return false;
    }
    result->prefill_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - started)
                             .count();
    result->prefill_hidden = workspace.hidden;
    return true;
}

static bool mm3_ar_collect_candidates(const MM3ArConfig & config,
                                      MM3ArWorkspace & workspace,
                                      MM3ArResult * result,
                                      std::string * error) {
    // Compact head layout: semantic block at row-offset 0, EOS is the last row.
    if (!tts_cpp::minimax::detail::collect_ar_candidates(
            workspace.logits.data(), config.compact_vocab, config.semantic_vocab,
            tts_cpp::minimax::detail::kCompactHeadSemanticOffset, config.semantic_vocab,
            workspace.candidates, result->nonfinite_logits)) {
        return mm3_ar_fail(error, "the LM candidate layout is invalid");
    }
    return true;
}

static void mm3_ar_apply_cfg_and_top_k(const MM3ArConfig & config,
                                       MM3ArWorkspace & workspace) {
    tts_cpp::minimax::detail::guide_ar_candidates(workspace.candidates,
                                                   config.cfg_scale);
    tts_cpp::minimax::detail::apply_conditional_top_k(workspace.candidates,
                                                       config.top_k);
}

static MM3ArStep mm3_ar_choose_semantic(const MM3ArConfig & config,
                                        const MM3ArOptions & options,
                                        int64_t iteration,
                                        MM3ArWorkspace & workspace,
                                        MM3ArResult * result, int32_t & semantic,
                                        std::string * error) {
    if (config.forced) {
        semantic = options.forced_semantic[iteration];
        if (semantic < 0 ||
            static_cast<int64_t>(semantic) >= config.semantic_vocab) {
            mm3_ar_fail(error,
                        "forced semantic code " + std::to_string(semantic) +
                            " at iteration " + std::to_string(iteration) +
                            " is outside [0, " +
                            std::to_string(config.semantic_vocab) + ")");
            return MM3ArStep::failure;
        }
        return MM3ArStep::proceed;
    }
    const int64_t selected = mm3_sample_top_k(
        workspace.candidates.guided.data(),
        static_cast<int64_t>(workspace.candidates.guided.size()), config.top_k,
        workspace.random, &workspace.sampling_scratch);
    if (selected == 0) {
        result->eos_hit = true;
        return MM3ArStep::stop;
    }
    semantic = static_cast<int32_t>(selected - 1);
    return MM3ArStep::proceed;
}

static bool mm3_ar_decode_depth(const MM3Model & model,
                                const MM3ArConfig & config,
                                const MM3ArOptions & options, int64_t iteration,
                                int32_t semantic, MM3ArWorkspace & workspace,
                                MM3ArResult * result, std::string * error) {
    const int32_t * forced_acoustic =
        config.forced
            ? options.forced_acoustic +
                  iteration * config.acoustic_codebooks
            : nullptr;
    if (!mm3_depth_decode_frame(
            model, workspace.hidden.data(),
            workspace.hidden.data() + config.hidden, semantic, forced_acoustic,
            &workspace.frame, error,
            config.forced ? nullptr : &workspace.random, config.top_k)) {
        return false;
    }
    if (mm3_ar_cancelled(options, error)) {
        return false;
    }
    result->depth_ms += workspace.frame.ms;
    if (workspace.frame.n_codes != config.acoustic_codebooks) {
        return mm3_ar_fail(
            error, "the depth decoder returned " +
                       std::to_string(workspace.frame.n_codes) +
                       " codes, expected " +
                       std::to_string(config.acoustic_codebooks));
    }
    return true;
}

static void mm3_ar_append_acoustic_codes(const MM3ArConfig & config,
                                         const MM3DepthFrame & frame,
                                         MM3ArResult * result) {
    for (int64_t index = 0; index < config.acoustic_codebooks; ++index) {
        result->acoustic_all.push_back(frame.codes[index]);
    }
}

static void mm3_ar_record_iteration(const MM3ArConfig & config, int32_t semantic,
                                    MM3ArWorkspace & workspace,
                                    MM3ArResult * result) {
    result->semantic_all.push_back(semantic);
    mm3_ar_append_acoustic_codes(config, workspace.frame, result);
    ++result->n_iterations;
}

static bool mm3_ar_dumping(const MM3ArOptions & options,
                           const MM3ArResult & result) {
    return static_cast<int64_t>(result.dumps.size()) < options.dump_iters;
}

static void mm3_ar_create_dump(const MM3ArConfig & config,
                               MM3ArWorkspace & workspace,
                               MM3ArResult * result) {
    MM3ArDump dump;
    dump.last_hidden = workspace.hidden;
    dump.sem_logits.resize(static_cast<size_t>(2 * config.semantic_vocab));
    memcpy(dump.sem_logits.data(),
           workspace.logits.data() + tts_cpp::minimax::detail::kCompactHeadSemanticOffset,
           static_cast<size_t>(config.semantic_vocab) * sizeof(float));
    memcpy(dump.sem_logits.data() + config.semantic_vocab,
           workspace.logits.data() + config.compact_vocab + tts_cpp::minimax::detail::kCompactHeadSemanticOffset,
           static_cast<size_t>(config.semantic_vocab) * sizeof(float));
    dump.guided.assign(workspace.candidates.guided.begin() + 1,
                       workspace.candidates.guided.end());
    dump.feedback.assign(static_cast<size_t>(2 * config.hidden), 0.0f);
    dump.depth_hidden = workspace.frame.hiddens;
    result->dumps.push_back(std::move(dump));
}

static void mm3_ar_collect_frame_hiddens(const MM3ArConfig & config,
                                         const MM3ArWorkspace & workspace,
                                         MM3ArResult * result) {
    result->frame_hiddens.insert(result->frame_hiddens.end(),
                                 workspace.hidden.begin(),
                                 workspace.hidden.begin() + config.hidden);
    result->frame_hiddens.insert(result->frame_hiddens.end(),
                                 workspace.frame.hiddens.begin(),
                                 workspace.frame.hiddens.end());
}

static MM3ArStep mm3_ar_emit_frame(const MM3ArConfig & config,
                                   const MM3ArOptions & options,
                                   int64_t iteration,
                                   const MM3ArWorkspace & workspace,
                                   MM3ArResult * result,
                                   std::string * error) {
    if (iteration == 0) {
        return MM3ArStep::proceed;
    }
    if (options.collect_hiddens) {
        mm3_ar_collect_frame_hiddens(config, workspace, result);
    }
    ++result->n_frames;
    if (options.on_frame) {
        options.on_frame(result->n_frames, config.max_frames);
    }
    if (mm3_ar_cancelled(options, error)) {
        return MM3ArStep::failure;
    }
    return result->n_frames >= config.max_frames ? MM3ArStep::stop
                                                  : MM3ArStep::proceed;
}

static bool mm3_ar_decode_feedback(const MM3Model & model,
                                   const MM3ArConfig & config,
                                   const MM3ArOptions & options,
                                   int32_t semantic, bool dumping,
                                   MM3ArWorkspace & workspace,
                                   MM3ArResult * result,
                                   std::string * error) {
    tts_cpp::minimax::detail::build_acoustic_rows(
        workspace.frame.codes, config.acoustic_codebooks,
        config.acoustic_vocab, workspace.acoustic_rows);
    if (mm3_ar_cancelled(options, error)) {
        return false;
    }
    const auto started = std::chrono::steady_clock::now();
    if (!mm3_lm_decode(model, &g_mm3_lm,
                       semantic + static_cast<int32_t>(config.semantic_offset),
                       workspace.acoustic_rows.data(), workspace.hidden.data(),
                       workspace.logits.data(), workspace.feedback.data(), error)) {
        return false;
    }
    result->lm_ms += std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - started)
                         .count();
    ++result->lm_steps;
    if (dumping) {
        MM3ArDump & dump = result->dumps.back();
        memcpy(dump.feedback.data(), workspace.feedback.data(),
               static_cast<size_t>(config.hidden) * sizeof(float));
        memcpy(dump.feedback.data() + config.hidden, workspace.feedback.data(),
               static_cast<size_t>(config.hidden) * sizeof(float));
    }
    return true;
}

static MM3ArStep mm3_ar_run_iteration(const MM3Model & model,
                                      const MM3ArConfig & config,
                                      const MM3ArOptions & options,
                                      int64_t iteration,
                                      MM3ArWorkspace & workspace,
                                      MM3ArResult * result,
                                      std::string * error) {
    const auto host_started = std::chrono::steady_clock::now();
    if (!mm3_ar_collect_candidates(config, workspace, result, error)) {
        return MM3ArStep::failure;
    }
    mm3_ar_apply_cfg_and_top_k(config, workspace);
    int32_t semantic = 0;
    const MM3ArStep choice = mm3_ar_choose_semantic(
        config, options, iteration, workspace, result, semantic, error);
    if (choice == MM3ArStep::failure) {
        return choice;
    }
    result->host_ms += std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - host_started)
                           .count();
    if (choice == MM3ArStep::stop) {
        return choice;
    }
    if (!mm3_ar_decode_depth(model, config, options, iteration, semantic,
                             workspace, result, error)) {
        return MM3ArStep::failure;
    }
    mm3_ar_record_iteration(config, semantic, workspace, result);
    const bool dumping = mm3_ar_dumping(options, *result);
    if (dumping) {
        mm3_ar_create_dump(config, workspace, result);
    }
    const MM3ArStep emission =
        mm3_ar_emit_frame(config, options, iteration, workspace, result, error);
    if (emission != MM3ArStep::proceed) {
        return emission;
    }
    return mm3_ar_decode_feedback(model, config, options, semantic, dumping,
                                  workspace, result, error)
               ? MM3ArStep::proceed
               : MM3ArStep::failure;
}

static bool mm3_ar_run_iterations(const MM3Model & model,
                                  const MM3ArConfig & config,
                                  const MM3ArOptions & options,
                                  MM3ArWorkspace & workspace,
                                  MM3ArResult * result,
                                  std::string * error) {
    for (int64_t iteration = 0; iteration <= config.max_frames; ++iteration) {
        if (mm3_ar_cancelled(options, error)) {
            return false;
        }
        if (config.forced && iteration >= options.forced_len) {
            break;
        }
        const MM3ArStep step = mm3_ar_run_iteration(
            model, config, options, iteration, workspace, result, error);
        if (step == MM3ArStep::failure) {
            return false;
        }
        if (step == MM3ArStep::stop) {
            break;
        }
    }
    return true;
}

static void mm3_ar_log_result(const MM3ArResult & result) {
    fprintf(stderr,
            "[MM3-AR] %lld frames (%lld iterations%s) in %.0f ms — prefill %.0f, LM %.0f (%lld steps, %.1f ms/step), "
            "depth %.0f (%.1f ms/frame), host %.0f\n",
            static_cast<long long>(result.n_frames),
            static_cast<long long>(result.n_iterations),
            result.eos_hit ? ", EOS" : "", result.total_ms, result.prefill_ms,
            result.lm_ms, static_cast<long long>(result.lm_steps),
            result.lm_steps ? result.lm_ms / static_cast<double>(result.lm_steps)
                            : 0.0,
            result.depth_ms,
            result.n_iterations
                ? result.depth_ms / static_cast<double>(result.n_iterations)
                : 0.0,
            result.host_ms);
    if (result.nonfinite_logits) {
        fprintf(stderr,
                "[MM3-AR] WARNING: %lld non-finite candidate logits were clamped to -inf\n",
                static_cast<long long>(result.nonfinite_logits));
    }
}

static bool mm3_ar_finalize(const std::chrono::steady_clock::time_point & started,
                            MM3ArResult * result, std::string * error) {
    result->total_ms = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - started)
                           .count();
    if (result->n_frames == 0) {
        return mm3_ar_fail(
            error, result->eos_hit
                       ? "the LM emitted EOS on the first iteration; zero audio frames were generated"
                       : "zero audio frames were generated");
    }
    mm3_ar_log_result(*result);
    return true;
}

static bool mm3_ar_plan(const MM3Model & model, const int32_t * conditional,
                        const int32_t * unconditional, int64_t prompt_length,
                        const MM3ArOptions & options, MM3ArResult * result,
                        std::string * error) {
    MM3ArConfig config;
    if (!mm3_ar_resolve_config(model, options, prompt_length, config, error)) {
        return false;
    }
    if (!mm3_lm_prepare(model, &g_mm3_lm, prompt_length + config.max_frames,
                        error)) {
        return false;
    }
    mm3_ar_initialize_result(config, options, result);
    MM3ArWorkspace workspace(options.seed);
    mm3_ar_initialize_workspace(config, workspace);
    const auto started = std::chrono::steady_clock::now();
    if (!mm3_ar_prefill(model, conditional, unconditional, prompt_length, options,
                        workspace, result, error) ||
        !mm3_ar_run_iterations(model, config, options, workspace, result, error)) {
        return false;
    }
    return mm3_ar_finalize(started, result, error);
}
