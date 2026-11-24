#pragma once

#include "audiogen-cpp/minimax/fit.h"
#include "minimax/backend.h"
#include "minimax/logic.h"
#include "minimax/mm3-pipeline.h"
#include "minimax/model-files.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace minimax_fit_parity {

constexpr int64_t kDitBranches = 2;

struct Shapes {
    int64_t              prompt = 0;
    int64_t              frames = 0;
    std::vector<int64_t> condition_frames;
    std::vector<int64_t> dit_latents;
    std::vector<int64_t> vocoder_tiles;
};

struct RealBytes {
    uint64_t weights         = 0;
    uint64_t lm_state        = 0;
    uint64_t lm_compute      = 0;
    uint64_t depth_state     = 0;
    uint64_t depth_compute   = 0;
    uint64_t depth_host      = 0;
    uint64_t cond_compute    = 0;
    uint64_t cond_host       = 0;
    uint64_t dit_compute     = 0;
    uint64_t vocoder_compute = 0;
};

inline uint64_t host_staging_bytes(ggml_backend_sched_t sched, ggml_backend_t backend, ggml_backend_t cpu_backend) {
    return backend == cpu_backend ? 0 : ggml_backend_sched_get_buffer_size(sched, cpu_backend);
}

inline void add_tiles(int64_t latents, std::vector<int64_t> & tiles) {
    if (mm3_voc_single_shot(latents, MM3_VOC_CHUNK)) {
        tiles.push_back(latents);
        return;
    }
    const int64_t core = mm3_voc_tile_core(MM3_VOC_CHUNK, MM3_VOC_OVERLAP);
    for (int64_t start = 0; start < latents; start += core) {
        const MM3VocTile tile = mm3_voc_tile(start, latents, core, MM3_VOC_OVERLAP);
        tiles.push_back(tile.end - tile.start);
    }
}

inline Shapes runtime_shapes(const MM3Model & model, int64_t prompt, int64_t frames) {
    const MM3DitConfig & dit = model.synth_cfg.dit;
    const int64_t        window = static_cast<int64_t>(dit.window_frames);
    Shapes               shapes;
    shapes.prompt = prompt;
    shapes.frames = frames;
    for (const int64_t start : tts_cpp::minimax::detail::window_starts(frames, window, dit.hop_frames)) {
        const int64_t count   = std::min(start + window, frames) - start;
        const int64_t latents = mm3_cond_latent_length(model.synth_cfg.cond, count);
        shapes.condition_frames.push_back(count);
        shapes.dit_latents.push_back(latents);
        add_tiles(latents, shapes.vocoder_tiles);
    }
    return shapes;
}

inline uint64_t buffer_bytes(ggml_backend_buffer_t buffer) {
    return buffer ? ggml_backend_buffer_get_size(buffer) : 0;
}

inline bool build_decode_slots(const MM3Model & model, MM3LmGraph & lm, const Shapes & shapes, std::string & error) {
    int64_t current = 0;
    for (int64_t position = shapes.prompt + 1; position <= shapes.prompt + shapes.frames; position++) {
        int64_t pad = 0;
        mm3_lm_bucket_for_context(position, lm.n_ctx, &pad);
        if (pad == current) {
            continue;
        }
        current = pad;
        mm3_lm_free_slot(&lm.decode);
        if (!mm3_lm_build_slot(model, &lm, &lm.decode, 1, pad, true, &error)) {
            return false;
        }
    }
    return true;
}

inline bool measure_lm(const MM3Model & model, const Shapes & shapes, RealBytes & real, std::string & error) {
    MM3LmGraph lm;
    if (!mm3_lm_prepare(model, &lm, shapes.prompt + shapes.frames, &error)) {
        return false;
    }
    int64_t prefill_pad = 0;
    mm3_lm_bucket_for_context(shapes.prompt, lm.n_ctx, &prefill_pad);
    lm.prefill.sched = backend_sched_new(lm.bp, MM3_LM_MAX_NODES * 2);
    lm.decode.sched  = backend_sched_new(lm.bp, MM3_LM_MAX_NODES * 2);
    const bool built = mm3_lm_build_slot(model, &lm, &lm.prefill, shapes.prompt, prefill_pad, false, &error) &&
                       build_decode_slots(model, lm, shapes, error);
    real.lm_state   = lm.kv_bytes;
    real.lm_compute = ggml_backend_sched_get_buffer_size(lm.prefill.sched, lm.backend) +
                      ggml_backend_sched_get_buffer_size(lm.decode.sched, lm.backend);
    mm3_lm_free(&lm);
    return built;
}

inline uint64_t depth_steps_bytes(const MM3DepthGraph & depth) {
    uint64_t total = 0;
    for (int step = 0; step < depth.n_steps; step++) {
        total += ggml_backend_sched_get_buffer_size(depth.step[step].sched, depth.backend);
    }
    return total;
}

inline uint64_t depth_steps_host_bytes(const MM3DepthGraph & depth) {
    uint64_t total = 0;
    for (int step = 0; step < depth.n_steps; step++) {
        total += host_staging_bytes(depth.step[step].sched, depth.backend, depth.cpu_backend);
    }
    return total + mm3_depth_step_context_bytes() * static_cast<uint64_t>(depth.n_steps);
}

inline bool measure_depth(const MM3Model & model, RealBytes & real, std::string & error) {
    MM3DepthGraph depth;
    const bool    prepared = mm3_depth_prepare(model, &depth, &error);
    if (prepared) {
        real.weights += buffer_bytes(depth.prep.buffer);
        real.depth_state   = buffer_bytes(depth.kv_buf);
        real.depth_compute = depth_steps_bytes(depth);
        real.depth_host    = depth_steps_host_bytes(depth);
    }
    mm3_depth_free(&depth);
    return prepared;
}

inline bool measure_cond(const MM3Model & model, const Shapes & shapes, RealBytes & real, std::string & error) {
    MM3CondGraph cond;
    bool         ok = mm3_cond_prepare(model, &cond, &error);
    for (size_t i = 0; ok && i < shapes.condition_frames.size(); i++) {
        ok = mm3_cond_ensure_graph(model, &cond, shapes.condition_frames[i], &error);
    }
    if (ok) {
        real.weights += buffer_bytes(cond.prep.buffer);
        real.cond_compute = ggml_backend_sched_get_buffer_size(cond.sched, cond.backend);
        real.cond_host    = host_staging_bytes(cond.sched, cond.backend, cond.cpu_backend) +
                         mm3_cond_graph_context_bytes();
    }
    mm3_cond_free(&cond);
    return ok;
}

inline bool measure_dit(const MM3Model & model, const Shapes & shapes, RealBytes & real, std::string & error) {
    MM3DitGraph dit;
    bool        ok = mm3_dit_prepare(model, &dit, &error);
    for (size_t i = 0; ok && i < shapes.dit_latents.size(); i++) {
        ok = mm3_dit_ensure_graph(model, &dit, shapes.dit_latents[i], kDitBranches, &error);
    }
    if (ok) {
        real.dit_compute = ggml_backend_sched_get_buffer_size(dit.sched, dit.backend);
    }
    mm3_dit_free(&dit);
    return ok;
}

inline bool measure_vocoder(const MM3Model & model, const Shapes & shapes, RealBytes & real, std::string & error) {
    MM3VocGraph vocoder;
    bool        ok = mm3_vocoder_prepare(model, &vocoder, &error);
    for (size_t i = 0; ok && i < shapes.vocoder_tiles.size(); i++) {
        ok = mm3_voc_ensure_graph(model, &vocoder, shapes.vocoder_tiles[i], &error);
    }
    if (ok) {
        real.weights += buffer_bytes(vocoder.prep.buffer);
        real.vocoder_compute = ggml_backend_sched_get_buffer_size(vocoder.sched, vocoder.backend);
    }
    mm3_vocoder_free(&vocoder);
    return ok;
}

inline bool load_model(const tts_cpp::minimax::EngineOptions & options, MM3Model & model, std::string & error) {
    const tts_cpp::minimax::EngineOptions resolved = tts_cpp::minimax::detail::resolve_model_paths(options);
    backend_configure_cpu(options.n_threads, options.backends_dir);
    backend_configure_device(options.device);
    tts_cpp::minimax::detail::probe_model_files(model, resolved);
    return mm3_load(&model, &error);
}

inline bool measure_real(const tts_cpp::minimax::EngineOptions & options, int64_t prompt, int64_t frames,
                         RealBytes & real, std::string & error) {
    MM3Model model;
    if (!load_model(options, model, error)) {
        return false;
    }
    const Shapes shapes = runtime_shapes(model, prompt, frames);
    real.weights        = model.vram_lm + model.vram_synth;
    const bool ok       = measure_lm(model, shapes, real, error) && measure_depth(model, real, error) &&
                    measure_cond(model, shapes, real, error) && measure_dit(model, shapes, real, error) &&
                    measure_vocoder(model, shapes, real, error);
    mm3_unload(&model);
    return ok;
}

inline const tts_cpp::minimax::FitStageProjection * find_stage(const tts_cpp::minimax::FitResult & fit,
                                                                 const std::string & name) {
    for (const auto & stage : fit.stages) {
        if (stage.name == name) {
            return &stage;
        }
    }
    return nullptr;
}

inline uint64_t projected_weights(const tts_cpp::minimax::FitResult & fit) {
    uint64_t total = 0;
    for (const auto & stage : fit.stages) {
        total += stage.weights_bytes;
    }
    return total;
}

}
