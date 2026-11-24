#pragma once

// Backend resolution shared by Engine::create (engine.cpp) and the memory-fit
// preflight (fit.cpp): one primary backend (GPU when requested and available,
// CPU otherwise), a CPU backend for the stages the placement policy pins off
// the GPU, and the per-stage assignments from stage_placement.h. Extracted
// verbatim from Engine::create so the preflight resolves the same devices a
// real load would by construction -- change it here and both agree.

#include "audiogen-cpp/gpu_fallback.h"

#include "acestep/backend_registry.h"
#include "acestep/stage_placement.h"

#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

namespace tts_cpp::acestep {

struct AcestepBackends {
    ggml_backend_t backend     = nullptr;  // primary (GPU or CPU); owned
    ggml_backend_t backend_cpu = nullptr;  // owned when != backend
    // Per-stage assignments (aliases of the two above, never owned):
    // textenc + cond, the autoregressive LM, and the FSQ detokenizer. The DiT
    // uses `backend` directly and the VAE creates its own backend (Vae::load).
    ggml_backend_t enc   = nullptr;
    ggml_backend_t lm    = nullptr;
    ggml_backend_t detok = nullptr;
    ggml_backend_t lm_extra = nullptr;

    bool              on_gpu              = false;
    GpuFallbackReason gpu_fallback_reason = GpuFallbackReason::not_requested;
    int               nth                 = 4;
};

struct BackendRequest {
    int         n_gpu_layers = 0;
    std::string backend      = BACKEND_REQUEST_AUTO;
    std::string lm_backend   = BACKEND_REQUEST_AUTO;
    int         n_threads    = 0;
    bool        verbose      = false;
};

inline int resolve_thread_count(int n_threads) {
    const int nth = n_threads > 0 ? n_threads : (int) std::thread::hardware_concurrency();
    return nth < 1 ? 4 : nth;
}

inline ggml_backend_t acquire_primary_backend(const BackendRequest & req, GpuFallbackReason & reason) {
    if (!backend_request_is_auto(req.backend)) {
        ggml_backend_t backend = backend_requested_init(req.backend, &reason);
        if (!backend) {
            fprintf(stderr, "[acestep-engine] requested backend '%s' unavailable (%s); no fallback\n",
                    req.backend.c_str(), gpu_fallback_reason_name(reason));
        }
        return backend;
    }
    if (req.n_gpu_layers <= 0) return nullptr;
    ggml_backend_t backend = backend_gpu_init(&reason);
    if (!backend && req.verbose) {
        fprintf(stderr, "[acestep-engine] GPU requested but no GPU backend available (%s); using CPU\n",
                gpu_fallback_reason_name(reason));
    }
    return backend;
}

// Free the owned backends (safe on a default-constructed struct).
inline void free_acestep_backends(AcestepBackends & b) {
    if (b.lm_extra) ggml_backend_free(b.lm_extra);
    if (b.backend_cpu && b.backend_cpu != b.backend) ggml_backend_free(b.backend_cpu);
    if (b.backend) ggml_backend_free(b.backend);
    b = AcestepBackends{};
}

inline bool place_requested_lm(const BackendRequest & req, AcestepBackends & out) {
    ggml_backend_dev_t dev = backend_requested_device(req.lm_backend);
    if (!dev) {
        fprintf(stderr, "[acestep-engine] requested LM backend '%s' unavailable; no fallback\n", req.lm_backend.c_str());
        return false;
    }
    if (dev == ggml_backend_get_device(out.backend)) {
        out.lm = out.backend;
    } else if (dev == ggml_backend_get_device(out.backend_cpu)) {
        out.lm = out.backend_cpu;
    } else {
        out.lm_extra = ggml_backend_dev_init(dev, nullptr);
        out.lm       = out.lm_extra;
    }
    return out.lm != nullptr;
}

inline bool attach_cpu_backend(AcestepBackends & out) {
    if (!out.on_gpu) {
        backend_set_n_threads(out.backend, out.nth);
        out.backend_cpu = out.backend;
        return true;
    }
    out.backend_cpu = backend_cpu_init();
    if (!out.backend_cpu) return false;
    backend_set_n_threads(out.backend_cpu, out.nth);
    return true;
}

inline void assign_stage_backends(AcestepBackends & out) {
    out.enc   = out.backend;
    out.lm    = out.backend;
    out.detok = out.backend;
    if (!out.on_gpu) return;
    const StagePlacement place = resolve_stage_placement(backend_reg_name(out.backend),
                                                         backend_dev_description(out.backend),
                                                         placement_overrides_from_env());
    if (!place.enc_on_gpu)   out.enc   = out.backend_cpu;
    if (!place.lm_on_gpu)    out.lm    = out.backend_cpu;
    if (!place.detok_on_gpu) out.detok = out.backend_cpu;
}

inline void report_stage_backends(const AcestepBackends & out) {
    fprintf(stderr, "[acestep-engine] backends: enc=%s lm=%s detok=%s dit=%s\n", ggml_backend_name(out.enc),
            ggml_backend_name(out.lm), ggml_backend_name(out.detok), ggml_backend_name(out.backend));
}

// Resolve the backends exactly as Engine::create does.
// load_backends(backends_dir) must have run first. Returns false when no
// backend can be initialised at all, or when an explicit request fails.
inline bool resolve_acestep_backends(const BackendRequest & req, AcestepBackends & out) {
    out     = AcestepBackends{};
    out.nth = resolve_thread_count(req.n_threads);

    out.backend = acquire_primary_backend(req, out.gpu_fallback_reason);
    if (!out.backend && !backend_request_is_auto(req.backend)) return false;
    if (!out.backend) out.backend = backend_cpu_init();
    if (!out.backend) return false;
    out.on_gpu = !backend_is_cpu_device(out.backend);
    if (!out.on_gpu && !backend_request_is_auto(req.backend)) out.gpu_fallback_reason = GpuFallbackReason::not_requested;
    if (out.on_gpu && req.verbose) {
        fprintf(stderr, "[acestep-engine] DiT on requested accelerator: %s\n", ggml_backend_name(out.backend));
    }
    if (!attach_cpu_backend(out)) {
        free_acestep_backends(out);
        return false;
    }
    assign_stage_backends(out);
    if (!backend_request_is_auto(req.lm_backend) && !place_requested_lm(req, out)) {
        free_acestep_backends(out);
        return false;
    }
    if (req.verbose && (out.on_gpu || out.lm_extra)) report_stage_backends(out);
    return true;
}

// ── VAE backend ──────────────────────────────────────────────────────────────
// The VAE acquires its own backend (Vae::load) rather than borrowing the
// primary one; the engine feeds it EngineOptions::n_gpu_layers with the
// ACESTEP_VAE_GPU diagnostic override. Both pieces live here so Vae::load, the
// engine, and the memory-fit projection resolve the same device by
// construction -- a future change lands in all three at once.

// ACESTEP_VAE_GPU forces the VAE backend independently of the other stages so
// a decode can be compared CPU-vs-GPU on an identical latent (=1 -> GPU,
// =0 -> CPU); leaves the LM/DiT backend untouched.
inline int vae_gpu_layers_from_env(int n_gpu_layers) {
    if (const char * e = std::getenv("ACESTEP_VAE_GPU")) {
        return (e[0] == '1') ? 99 : 0;
    }
    return n_gpu_layers;
}

inline std::string vae_backend_request_from_env(const std::string & backend) {
    const char * e = std::getenv("ACESTEP_VAE_GPU");
    if (!e) return backend_request_is_xdna(backend) ? BACKEND_REQUEST_CPU : backend;
    if (backend_request_is_auto(backend) || e[0] == '1') return backend;
    return BACKEND_REQUEST_CPU;
}

inline BackendRequest vae_backend_request(const BackendRequest & engine_req) {
    BackendRequest req = engine_req;
    req.n_gpu_layers   = vae_gpu_layers_from_env(engine_req.n_gpu_layers);
    req.backend        = vae_backend_request_from_env(engine_req.backend);
    return req;
}

// The requested device, or for "auto" the GPU when n_gpu_layers asks for one
// (the two custom VAE ops have Metal/Vulkan/validated-OpenCL kernels in the
// fabric ggml) with CPU fallback. CPU backends get the thread count.
// Returns null when an explicit request fails or no CPU backend can be
// initialised; the returned backend is owned by the caller.
inline ggml_backend_t resolve_vae_backend(const BackendRequest & req) {
    ggml_backend_t backend = nullptr;
    if (!backend_request_is_auto(req.backend)) {
        backend = backend_requested_init(req.backend);
        if (!backend) {
            fprintf(stderr, "[acestep-vae] requested backend '%s' unavailable; no fallback\n", req.backend.c_str());
            return nullptr;
        }
    } else if (req.n_gpu_layers > 0) {
        backend = backend_gpu_init();
        if (!backend && req.verbose) {
            fprintf(stderr, "[acestep-vae] GPU requested but no GPU backend available; using CPU\n");
        }
    }
    if (!backend) backend = backend_cpu_init();
    if (backend && backend_is_cpu_device(backend)) backend_set_n_threads(backend, resolve_thread_count(req.n_threads));
    return backend;
}

}  // namespace tts_cpp::acestep
