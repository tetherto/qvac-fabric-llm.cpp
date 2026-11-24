// FastConformer encoder ggml graph, GGUF load, CTC head, and encoder execution.

#include "parakeet_ctc.h"
#include "decoder_backend.h"
#include "coreml/parakeet_coreml_shape.h"
#include "long_form.h"
#include "mel_preprocess.h"
#include "parakeet_log.h"
#include "backend_util.h"
#include "encoder_graph_allocator.h"
#include "packed_affine.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#ifdef PARAKEET_USE_COREML
#include "coreml/coreml_diarizers.h"
#include "coreml/parakeet_coreml_sidecars.h"
#include "coreml/parakeet-encoder.h"
#include "coreml/parakeet-decoder.h"
#include "parakeet_coreml_path.h"
#include <sys/stat.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

#if defined(__ANDROID__) || defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

namespace parakeet {

// run_encoder result when the ggml encoder is asked for while Core ML owns the
// encoder and its weights were never loaded.
constexpr int kEncoderWeightsNotLoaded = 9;

struct EncoderGraph {
    ggml_context * graph_ctx = nullptr;
    ggml_cgraph  * cgraph    = nullptr;
    int            T_mel     = 0;
    int            T_mel_valid = 0;        // non-zero mel frames the subsampling was built for
    int            T_enc     = 0;          // post-subsampling frame count
    int            n_run_layers = 0;
    bool           all_valid = false;
    bool           bypass_pre_encode = false; // true: skip subsampling, pre_encode_in is direct input
    bool           eu_use_adapters = false;

    std::vector<float> pe_host;
    std::vector<float> att_mask_host;   // (T_enc, T_enc) row-major; 0 for visible, -inf for masked

    ggml_tensor * mel_in   = nullptr;
    ggml_tensor * pre_encode_in = nullptr;  // set only when bypass_pre_encode is true; shape (d_model, T_enc)
    ggml_tensor * pe_in    = nullptr;
    ggml_tensor * att_mask = nullptr;   // null when the encoder uses unrestricted attention

    ggml_tensor * sub_out_node         = nullptr;
    ggml_tensor * post_ff1_0_node      = nullptr;
    ggml_tensor * post_attn_0_node     = nullptr;
    ggml_tensor * post_conv_0_node     = nullptr;
    ggml_tensor * post_ff2_0_node      = nullptr;
    ggml_tensor * block_0_out_node     = nullptr;
    ggml_tensor * block_last_out_node  = nullptr;
    ggml_tensor * encoder_out_node     = nullptr;
    ggml_tensor * logits_node          = nullptr;

    // Diagnostic attention captures for block 0 only.
    ggml_tensor * block_0_attn_qkv_node      = nullptr;
    ggml_tensor * block_0_attn_k_raw_node    = nullptr;
    ggml_tensor * block_0_attn_v_raw_node    = nullptr;
    ggml_tensor * block_0_attn_q_perm_node   = nullptr;
    ggml_tensor * block_0_attn_k_perm_node   = nullptr;
    ggml_tensor * block_0_attn_q_u_node      = nullptr;
    ggml_tensor * block_0_attn_ac_node       = nullptr;
    ggml_tensor * block_0_attn_bd_node       = nullptr;
    ggml_tensor * block_0_attn_scores_node   = nullptr;
    ggml_tensor * block_0_attn_softmax_node  = nullptr;
    ggml_tensor * block_0_attn_out_node      = nullptr;

    std::vector<std::pair<ggml_tensor *, std::array<ggml_tensor *, GGML_MAX_SRC>>> src_backup;

    EncoderGraphAllocator allocator;

    // Per-layer positional projections, computed once per graph size (see build_pos_proj_cache).
    ggml_context         * pos_ctx = nullptr;
    ggml_backend_buffer_t  pos_buf = nullptr;
    std::vector<ggml_tensor *> pos_proj;

    void release_pos_proj() {
        if (pos_buf) { ggml_backend_buffer_free(pos_buf); pos_buf = nullptr; }
        if (pos_ctx) { ggml_free(pos_ctx); pos_ctx = nullptr; }
        pos_proj.clear();
    }

    void free_() {
        allocator.release();
        release_pos_proj();
        if (graph_ctx) { ggml_free(graph_ctx);     graph_ctx = nullptr; }
        cgraph = nullptr;
        mel_in = pe_in = nullptr;
        pre_encode_in = nullptr;
        sub_out_node = post_ff1_0_node = post_attn_0_node = nullptr;
        post_conv_0_node = post_ff2_0_node = block_0_out_node = nullptr;
        block_last_out_node = encoder_out_node = logits_node = nullptr;
        block_0_attn_qkv_node = block_0_attn_ac_node = nullptr;
        block_0_attn_bd_node = block_0_attn_scores_node = nullptr;
        block_0_attn_softmax_node = block_0_attn_out_node = nullptr;
        block_0_attn_k_raw_node = block_0_attn_v_raw_node = nullptr;
        block_0_attn_q_perm_node = block_0_attn_k_perm_node = nullptr;
        block_0_attn_q_u_node = nullptr;
        T_mel = 0;
        T_mel_valid = 0;
        T_enc = 0;
        all_valid = false;
        bypass_pre_encode = false;
        eu_use_adapters = false;
        pe_host.clear();
        att_mask_host.clear();
        src_backup.clear();
    }

    // Own the allocator + graph context via RAII so a build_encoder_graph_cached
    // failure (which destroys a half-built cache entry with cache.pop_back())
    // cannot leak them. free_() is idempotent, so the explicit free_() calls in
    // ~Impl / cache eviction stay safe (double free_() is a no-op).
    ~EncoderGraph() { free_(); }
};

struct NemotronPromptGraph {
    ggml_context * context = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * prompt = nullptr;
    int32_t prompt_id = -1;
    ggml_tensor * output = nullptr;
    int n_frames = 0;

    void clear() {
        if (allocator) {
            ggml_gallocr_free(allocator);
            allocator = nullptr;
        }
        if (context) {
            ggml_free(context);
            context = nullptr;
        }
        graph = nullptr;
        input = nullptr;
        prompt = nullptr;
        prompt_id = -1;
        output = nullptr;
        n_frames = 0;
    }

    ~NemotronPromptGraph() { clear(); }
};

// The Core ML sidecar intentionally stops at encoder hidden states so it can
// be shared by every decoder family. CTC models therefore keep their small
// vocabulary projection on the active ggml backend. Cache this graph by frame
// count just like the native encoder graphs to avoid rebuilding/allocating it
// on every benchmark iteration and streaming chunk.
struct CtcHeadGraph {
    ggml_context * context = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    int n_frames = 0;

    void clear() {
        if (allocator) {
            ggml_gallocr_free(allocator);
            allocator = nullptr;
        }
        if (context) {
            ggml_free(context);
            context = nullptr;
        }
        graph = nullptr;
        input = nullptr;
        output = nullptr;
        n_frames = 0;
    }

    ~CtcHeadGraph() { clear(); }
};

#ifdef PARAKEET_EXPERIMENTAL_FLASH_ATTN
constexpr bool k_flash_attn_compiled = true;
#else
constexpr bool k_flash_attn_compiled = false;
#endif

// `PARAKEET_F16_ACTIVATIONS` (build flag, default ON) lets the encoder cast
// specific tensors to F16 so the Hexagon HMX v79 path (F16-only) can accept
// them. The encoder residual stream stays F32 because Hexagon's quantized MM
// kernels only carry a `quantize_f32_q8_0_*` activation quantizer
// (matmul-ops.c:843) — feeding F16 activations to Q8_0 FF/Conv/attn
// projections fails with DSP NO_SUPPORT. The flag scope is narrow: cast
// (and 32-align pad) `k_perm` / `v_for_mm` / softmax output right before the
// batched attention MULMATs in `rel_pos_mha_unfused_graph`, matching whisper's
// convention of casting only at the attention matmul boundary. When the flag
// is OFF the helpers below are compile-time no-ops.
#ifdef PARAKEET_F16_ACTIVATIONS
constexpr bool k_f16_activations_compiled = true;
#else
constexpr bool k_f16_activations_compiled = false;
#endif

// Cast an F32 tensor to F16. No-op when the flag is OFF or when `t` is already
// F16. Used inside `pad_and_cast_for_hmx` at the batched attention MULMAT
// boundary so `hmx_mm_f16_f32_batched` (src0=F16, src1=F32) engages.
static inline ggml_tensor * cast_to_activation(ggml_context * ctx, ggml_tensor * t) {
#ifdef PARAKEET_F16_ACTIVATIONS
    if (t && t->type != GGML_TYPE_F16) return ggml_cast(ctx, t, GGML_TYPE_F16);
#else
    (void) ctx;
#endif
    return t;
}

// Round N up to the next multiple of 32. The Hexagon HMX F16-batched kernel
// (hmx_mm_f16_f32_batched) refuses non-32-aligned N or K at runtime
// (matmul-ops.c:3081); zero-padding the src0 tensor to a 32-aligned tail
// clears that gate.
static inline int hmx_align_up_32(int n) { return (n + 31) & ~31; }

// F32 tail-zero-pad along `dim` (0 or 1) up to a 32-aligned length, then cast
// to activation dtype. Order matters: PAD must run in F32 because Hexagon's
// PAD kernel only accepts F32 (`ggml_hexagon_supported_pad`). No-op when
// PARAKEET_F16_ACTIVATIONS is OFF or when `t` is already 32-aligned along
// `dim`, so the F32 fast path is preserved bit-exactly.
static inline ggml_tensor * pad_and_cast_for_hmx(ggml_context * ctx,
                                                 ggml_tensor * t, int dim) {
    if (!k_f16_activations_compiled) return t;
    const int n     = (int) t->ne[dim];
    const int n_pad = hmx_align_up_32(n);
    if (n_pad != n) {
        int rp[4] = {0, 0, 0, 0};
        rp[dim] = n_pad - n;
        t = ggml_pad(ctx, t, rp[0], rp[1], rp[2], rp[3]);
    }
    return cast_to_activation(ctx, t);
}

// Tail-zero-pad an F32 tensor along dim 0 to a 32-aligned length; no cast.
// Used on the softmax output before the batched `attn_v` matmul so its K axis
// matches the padded `v_for_mm` src0. Padding zeros produce zero contribution
// at padded positions, keeping the F32 result byte-identical to the unpadded
// computation.
static inline ggml_tensor * pad_ne0_for_hmx(ggml_context * ctx, ggml_tensor * t) {
    if (!k_f16_activations_compiled) return t;
    const int n     = (int) t->ne[0];
    const int n_pad = hmx_align_up_32(n);
    if (n_pad != n) {
        t = ggml_pad(ctx, t, n_pad - n, 0, 0, 0);
    }
    return t;
}

struct ParakeetCtcModel::Impl {
    PackedAffineWeights affine_weights;
    gguf_context         * gguf           = nullptr;
    ggml_context         * ctx            = nullptr;
    ggml_backend_t         backend_cpu    = nullptr;
    ggml_backend_t         backend_blas   = nullptr;
    ggml_backend_t         backend_gpu    = nullptr;
    ggml_backend_t         backend_active = nullptr;
    // True when a GPU was detected but skipped as known-bad (Mali), so
    // backend_active fell back to CPU.
    bool                   gpu_unsupported = false;
    // True when the active GPU backend is Mali-Vulkan: the Sortformer head is
    // routed to CPU there (its transformer block 0 miscomputes to NaN on the
    // Valhall driver) while the encoder + CTC/TDT/EOU heads stay on the Mali GPU.
    bool                   sortformer_force_cpu = false;
    // Depthwise-conv lowering the active backend can run directly (GGML_OP_CONV_2D_DW):
    // planar (W,H,C) input for the subsampler, channel-contiguous input for the
    // conformer conv module. False falls back to the portable im2col lowering.
    bool                   dw_direct_planar   = false;
    bool                   dw_direct_channels = false;
    // Fused or unfused attention for the active backend, and whether the fused
    // kernel takes the relative-position bias as one mask per head (otherwise
    // heads fold into the batch dimension).
    AttnPath               attn;
    bool                   glu_fused          = false;
    // CPU-resident copies of the Sortformer head weights (Mali-Vulkan only), so
    // the CPU head graph reads them from the CPU backend instead of
    // dereferencing the GPU-resident originals.
    ggml_context         * sortformer_cpu_ctx    = nullptr;
    ggml_backend_buffer_t  sortformer_cpu_buffer = nullptr;
    // CPU-resident copies of the TDT / RNN-T decoder weights (OpenCL only).
    ggml_context         * tdt_cpu_ctx           = nullptr;
    ggml_backend_buffer_t  tdt_cpu_buffer        = nullptr;
    ggml_backend_buffer_t  weights_buffer = nullptr;
    // CPU "extra" buffer-type (CPU_REPACK) buffers holding the repack-eligible
    // encoder GEMM weights on CPU-only runs. See alloc_cpu_repack_weights.
    std::vector<ggml_backend_buffer_t> weights_extra_buffers;
    // Compute scheduler over [active backend, CPU] (CPU last). Routes ops the
    // active backend cannot run to CPU per-op; a single-split pass-through when
    // every op is supported. Must be freed before the backends it references.
    ggml_backend_sched_t   sched          = nullptr;
    std::vector<std::unique_ptr<EncoderGraph>> encoder_graphs;
    static constexpr size_t k_encoder_graph_cache_max = 3;
    size_t encoder_graph_cache_limit = k_encoder_graph_cache_max;
    static constexpr size_t k_pos_proj_cache_max_bytes = (size_t) 256 << 20;
    std::unique_ptr<NemotronPromptGraph> nemotron_prompt_graph;
    std::unique_ptr<CtcHeadGraph> ctc_head_graph;

#ifdef PARAKEET_USE_COREML
    // Background Core ML sidecars: one runtime slot each for the
    // encoder (`<model>-encoder.mlmodelc`), the Sortformer v2.1 AOSC bypass
    // encoder (`-encoder-bypass-pre-encode`) and every diarizer capacity. They
    // load on the runtime's lane; each call acquires its route at its start
    // and runs on ggml with today's ggml plan until its slot is ready.
    std::unique_ptr<CoremlSidecars> coreml;
    std::shared_ptr<pcm::Slot> encoder_slot;
    std::shared_ptr<pcm::Slot> bypass_slot;
    // Synchronous encoder sidecar that owns the encoder: loaded before
    // weight allocation when the GGUF has no encoder weights or
    // PARAKEET_COREML_ONLY is set, and kept only when it may own the encoder
    // (coreml_can_own_encoder). Never a slot; a failure is an error. A
    // standard TDT sidecar never owns it, so with PARAKEET_COREML_ONLY and a
    // full GGUF it is not kept here: the encoder slot loads it in the
    // background like any other sidecar.
    parakeet_coreml_context * ctx_coreml = nullptr;
    // TranscribePsy EU (model variant transcribepsy-250m-eu) is not on the
    // runtime: its programs load synchronously and are selected per dispatch
    // (selected_coreml_encoder). `ctx_coreml` holds its English fixed program
    // (owner or not, see coreml_owns_encoder); the others are below, the
    // flexible and precise ones loaded on first use.
    parakeet_coreml_context * ctx_coreml_eu_other = nullptr;
    parakeet_coreml_context * ctx_coreml_eu_gate = nullptr;
    parakeet_coreml_context * ctx_coreml_eu_flexible = nullptr;
    parakeet_coreml_context * ctx_coreml_eu_other_flexible = nullptr;
    std::string eu_flexible_path, eu_other_flexible_path, eu_gate_flexible_path, eu_gguf_path;
    parakeet_coreml_context * ctx_coreml_eu_precise_gpu = nullptr;
    std::string eu_precise_gpu_path;
    std::vector<std::string> eu_invalid_gpu_paths;
    parakeet_coreml_context * ctx_coreml_eu_gate_flexible = nullptr;
    // Backend of the last EU encoder dispatch (Engine::encoder_backend()).
    std::string last_coreml_encoder_backend;
    // Optional Apple Neural Engine TDT decoder sidecar, loaded from a
    // `<model>-decoder.mlmodelc` on the same presence-driven terms as the
    // encoder and independently of it. Non-null only enables the batched lane
    // decode of the engine's offline transcribe path (tdt_decode_lanes);
    // streaming paths keep the sequential decoder.
    parakeet_coreml_decoder * ctx_coreml_dec = nullptr;
    std::unique_ptr<CoremlDiarizers> coreml_diarizers;
    // The identity (fstat) of the one descriptor the GGUF's metadata and
    // tensor data were read through: a generated diarizer is prepared only
    // from the file whose weights this model loaded.
    pcm::SourceStat source;
    bool source_ok = false;
    // The owning engine's cancel flag: every unit's acquisition (an
    // encoder window, a diarizer pass, an AOSC step) stops waiting on it.
    const std::atomic<bool> * unit_cancel = nullptr;
#endif
    // Set when the encoder sidecar loaded before weight allocation and the
    // `encoder.*` tensors were therefore never given memory: the GGUF has no
    // encoder weights, or PARAKEET_COREML_ONLY skipped them. The ggml encoder
    // cannot run in this state and every path that would reach it fails.
    bool coreml_owns_encoder = false;

    ~Impl() {
#ifdef PARAKEET_USE_COREML
        // Slots never wait: a load in flight finishes on the runtime's
        // lane and is released there.
        coreml_diarizers.reset();
        encoder_slot.reset();
        bypass_slot.reset();
        coreml.reset();
        if (ctx_coreml) parakeet_coreml_free(ctx_coreml);
        if (ctx_coreml_eu_other) parakeet_coreml_free(ctx_coreml_eu_other);
        if (ctx_coreml_eu_gate) parakeet_coreml_free(ctx_coreml_eu_gate);
        if (ctx_coreml_eu_precise_gpu) parakeet_coreml_free(ctx_coreml_eu_precise_gpu);
        if (ctx_coreml_eu_gate_flexible) parakeet_coreml_free(ctx_coreml_eu_gate_flexible);
        if (ctx_coreml_eu_flexible) parakeet_coreml_free(ctx_coreml_eu_flexible);
        if (ctx_coreml_eu_other_flexible) parakeet_coreml_free(ctx_coreml_eu_other_flexible);
        if (ctx_coreml_dec) parakeet_coreml_decoder_free(ctx_coreml_dec);
#endif
        for (auto & g : encoder_graphs) {
            if (g) g->free_();
        }
        encoder_graphs.clear();
        nemotron_prompt_graph.reset();
        ctc_head_graph.reset();
        if (sched)          ggml_backend_sched_free(sched);
        if (weights_buffer) ggml_backend_buffer_free(weights_buffer);
        for (ggml_backend_buffer_t b : weights_extra_buffers) {
            ggml_backend_buffer_free(b);
        }
        weights_extra_buffers.clear();
        if (sortformer_cpu_buffer) ggml_backend_buffer_free(sortformer_cpu_buffer);
        if (sortformer_cpu_ctx)    ggml_free(sortformer_cpu_ctx);
        if (tdt_cpu_buffer)        ggml_backend_buffer_free(tdt_cpu_buffer);
        if (tdt_cpu_ctx)           ggml_free(tdt_cpu_ctx);
        if (ctx)            ggml_free(ctx);
        if (gguf)           gguf_free(gguf);
        if (backend_blas)   ggml_backend_free(backend_blas);
        if (backend_gpu)    ggml_backend_free(backend_gpu);
        if (backend_cpu)    ggml_backend_free(backend_cpu);
    }
};

ggml_backend_t ParakeetCtcModel::backend_active() const {
    return impl ? impl->backend_active : nullptr;
}

ggml_backend_t ParakeetCtcModel::backend_cpu() const {
    return impl ? impl->backend_cpu : nullptr;
}

ggml_context * ParakeetCtcModel::weights_ctx() const {
    return impl ? impl->ctx : nullptr;
}


namespace {

// Backends-dir / OpenCL-cache-dir override + warning state. The
// setters are intended to be called by the first Engine
// construction; both are consumed once and then frozen for the rest
// of the process lifetime (the ggml-backend registry and
// $GGML_OPENCL_CACHE_DIR are both process-singleton state -- see
// comment on `ensure_backends_loaded` and the analogous note in
// `set_opencl_cache_dir`).
//
// `g_backends_loaded` is the canonical "registry already populated"
// flag, set inside `ensure_backends_loaded()` *before* the load-all
// call returns AND under the mutex so concurrent `set_*` calls
// either land their write (and have it picked up by the in-flight
// load) or atomically observe the flag and warn. We track it
// separately from `g_recorded_backends_dir` because the first
// Engine may have legitimately constructed with an empty
// `backends_dir` (default ggml search path), in which case
// `g_recorded_backends_dir` stays empty and is no longer a reliable
// "have we loaded?" sentinel -- a subsequent setter would otherwise
// silently write to `g_backends_dir`, never get re-scanned, and
// surface zero diagnostic to the caller.
std::mutex     g_backends_dir_mutex;
std::string    g_backends_dir;
std::string    g_recorded_backends_dir;
std::string    g_recorded_opencl_cache_dir;
std::atomic<bool> g_backends_loaded{false};
std::atomic<bool> g_backends_dir_warned{false};
std::atomic<bool> g_opencl_cache_dir_warned{false};

void prepare_hexagon_library_path(const std::string & dir);

// Trigger one-time discovery + load of every available ggml backend.
// Idempotent: repeated calls inside the same process are no-ops once
// the registry is populated. Routed through a static guard so we don't
// pay the directory-walk cost on every model load.
//
// Why this instead of the per-backend ggml_backend_<x>_init() entry
// points the cascade used to call directly: with GGML_BACKEND_DL=ON
// (the dynamic-loader mode embedded host applications typically
// ship with) the CUDA / Metal / Vulkan / OpenCL / BLAS / ggml-cpu
// backends live in separate shared libraries that are dlopened at
// runtime; their concrete init symbols are not linkable from
// libqvac-parakeet, and the only supported entry point is the registry.
// With GGML_BACKEND_DL=OFF the backends are statically linked into
// libggml, registered at constructor time, and
// ggml_backend_load_all() is a cheap no-op. Both modes therefore
// reach the same registry walk below, matching the convention used
// by llama.cpp and other ggml-based libraries.
//
// The optional backends dir comes from `set_backends_directory()`
// (typically wired from `EngineOptions::backends_dir`). When set and
// non-empty, the loader walks that single directory instead of the
// compile-time defaults so embedded host apps can ship the
// `lib<prefix>ggml-{vulkan,opencl,cpu-*}.so` files in their own
// per-module folder rather than relying on `LD_LIBRARY_PATH` /
// `dlopen()` heuristics.
void ensure_backends_loaded() {
    static const bool loaded = []() {
        std::string dir;
        {
            std::lock_guard<std::mutex> lock(g_backends_dir_mutex);
            dir = g_backends_dir;
            g_recorded_backends_dir = g_backends_dir;
            // Discovery initializes all backend registries, including HTP
            // even when this first model requests CPU/OpenCL. Configure the
            // DSP loader once here so a later explicit HTP0 request works.
            prepare_hexagon_library_path(dir);
            // Flip the loaded sentinel under the mutex (and *before*
            // we release it for the load-all call below) so any
            // concurrent setter that's about to acquire the mutex
            // sees the registry as already-claimed and falls into
            // its warn-once branch. Without this, a setter racing
            // a first Engine construction would land its value
            // *after* we already captured `dir` into the local --
            // the registry would scan against the wrong directory
            // (or the default), and the second Engine would have
            // no idea its override was lost.
            g_backends_loaded.store(true, std::memory_order_release);
        }
        if (!dir.empty()) {
            ggml_backend_load_all_from_path(dir.c_str());
        } else {
            ggml_backend_load_all();
        }
        return true;
    }();
    (void) loaded;
}

// Parse the Adreno generation number from a device name /
// description string. Returns:
//   - a 3-or-4-digit generation number ("Adreno (TM) 750" -> 750,
//     "Adreno 830" -> 830, "Adreno 660" -> 660)
//   - a synthetic 800 for the "Adreno X<n>" naming used by
//     Snapdragon X Elite parts (X1-85 / X1-45 etc.). These are
//     7xx/8xx-tier silicon with kernels that ggml-opencl supports
//     and outperform Vulkan on. Mapped to 800 here so they take
//     the OpenCL branch in the tier policy.
//   - -1 when no Adreno marker is present (Mali, desktop GPUs, ...)
//
// Used to drive the OpenCL vs Vulkan tier policy below: Adreno
// 7xx/8xx/X<n> ship OpenCL kernels that outperform Vulkan on those
// parts, while Adreno 6xx ggml-opencl is known broken (incorrect
// results). Uses the same lowercase + regex approach as llm-llamacpp's
// BackendSelection.cpp::parseAdrenoVersion; this variant additionally
// ignores the "OpenCL 3.0" API-version noise in the combined OpenCL
// device description and maps the Snapdragon-X "X<n>" naming to 800.
// (Converging parakeet/tts-cpp/llm-llamacpp/ggml onto one shared parser
// is tracked separately.)
int parse_adreno_version(const char * s) {
    if (!s) return -1;
    std::string lowered(s);
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    // After an "adreno" marker (skipping "(tm)", spaces, punctuation), the model
    // is a 3-4 digit generation ("740"/"830") or the Snapdragon-X "x<n>" token
    // ("x1-85" -> 800-tier). Scan every marker and keep the highest; requiring
    // 3-4 digits skips the "opencl 3.0" noise in the combined OpenCL description.
    static const std::regex re(R"(dreno\D*?(\d{3,4}|x\d))", std::regex::optimize);
    int best = -1;
    for (std::sregex_iterator it(lowered.begin(), lowered.end(), re), end; it != end; ++it) {
        const std::string tok = (*it)[1].str();
        const int v = (tok[0] == 'x') ? 800 : std::stoi(tok);
        if (v > best) best = v;
    }
    return best;
}

bool is_adreno_6xx(const char * s) {
    const int v = parse_adreno_version(s);
    return v >= 600 && v < 700;
}

bool is_adreno_700plus(const char * s) {
    const int v = parse_adreno_version(s);
    return v >= 700;
}

// True when a device name/description identifies an ARM Mali GPU ("Mali-G715",
// "ARM Mali-G78", ...). Used to route the Sortformer head to CPU on Mali-Vulkan.
bool desc_is_mali(const char * s) {
    if (!s) return false;
    std::string lowered(s);
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return lowered.find("mali") != std::string::npos;
}

const char * dev_reg_name(ggml_backend_dev_t dev) {
    if (!dev) return "";
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    return reg ? ggml_backend_reg_name(reg) : "";
}


// Ask the backend whether it runs GGML_OP_CONV_2D_DW on a representative F32 kernel/input
// pair, for either the planar (W,H,C,N) input layout or the channel-contiguous one the
// conformer conv module produces. Uses a throwaway no-alloc context; nothing is computed.
bool backend_runs_conv_2d_dw(ggml_backend_t backend, bool channels_contiguous) {
    if (!backend) return false;
    ggml_init_params ip = { ggml_tensor_overhead() * 8, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;
    const int K = 3, C = 8, W = 16, Hh = 4;
    ggml_tensor * kernel = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, K, K, 1, C);
    ggml_tensor * input  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, Hh, C, 1);
    if (channels_contiguous) {
        kernel = ggml_permute(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, C, K, K, 1), 3, 0, 1, 2);
        input  = ggml_permute(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, C, W, Hh, 1), 2, 0, 1, 3);
    }
    ggml_tensor * out = ggml_conv_2d_dw_direct(ctx, kernel, input, 1, 1, 1, 1, 1, 1);
    const bool ok = ggml_backend_supports_op(backend, out);
    ggml_free(ctx);
    return ok;
}

// Pick a GPU backend using the same tier policy as llm-llamacpp's
// BackendSelection and tts/src/backend_selection.cpp: ggml-opencl
// is only used when an Adreno 700+ device is present (where its kernels
// are validated and faster than Vulkan); CUDA is preferred over Vulkan on
// the same NVIDIA card (vendor-native, measurably faster); and among the
// remaining non-OpenCL GPUs, discrete adapters are preferred over integrated
// (iGPU) ones, matching the discrete-first behavior already used by llm and
// diffusion. Adreno 6xx OpenCL is known broken (incorrect outputs) and is
// force-skipped unless the caller opts in via `PARAKEET_ALLOW_ADRENO_6XX=1`.
//
// Routed exclusively through the ggml-backend registry
// (`ggml_backend_load_all` + `ggml_backend_dev_*`). No direct calls
// to `ggml_backend_vulkan_init` / `ggml_backend_opencl_init` /
// `ggml_backend_metal_init` are made anywhere in parakeet — under
// the GGML_BACKEND_DL=ON build mode embedded host applications ship
// with, those entry points live in separate shared libraries that
// are dlopen()'d at runtime and are not linkable from libqvac-parakeet.
// The registry walk reaches the same backends in both modes.
// Ask the backend whether ggml_flash_attn_ext accepts a mask with one slice per head.
// Whether the backend runs the sigmoid-gated GLU on a [2C, T] tensor (the conformer conv module's
// a * sigmoid(b) over the two channel halves) without materialising the halves.
bool backend_runs_siglu(ggml_backend_t backend, int C) {
    if (!backend) return false;
    ggml_init_params ip = { ggml_tensor_overhead() * 4, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;
    const int T = 8;
    ggml_tensor * out = ggml_siglu(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2 * C, T));
    const bool ok = ggml_backend_supports_op(backend, out);
    ggml_free(ctx);
    return ok;
}

bool backend_runs_flash_attn_per_head_mask(ggml_backend_t backend, int HD, int H) {
    if (!backend) return false;
    ggml_init_params ip = { ggml_tensor_overhead() * 8, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;
    const int T = 64;
    ggml_tensor * q    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, HD, T, H);
    ggml_tensor * k    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, HD, T, H);
    ggml_tensor * v    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, HD, T, H);
    ggml_tensor * mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, T, T, H);
    ggml_tensor * out  = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f, 0.0f, 0.0f);
    const bool ok = ggml_backend_supports_op(backend, out);
    ggml_free(ctx);
    return ok;
}

// The fused attention node exactly as rel_pos_mha_flash_graph emits it: head-major
// views for k and v, the f16 mask per head or, with heads folded into the batch, as
// [T, T, 1, H].
bool backend_runs_flash_attn(ggml_backend_t backend, int HD, int H, bool per_head_mask) {
    if (!backend) return false;
    ggml_init_params ip = { ggml_tensor_overhead() * 16, nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) return false;
    const int T = 64;
    ggml_tensor * q    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, HD, T, H);
    ggml_tensor * kp   = ggml_permute(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, HD, H, T), 0, 2, 1, 3);
    ggml_tensor * vp   = ggml_permute(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, HD, H, T), 0, 2, 1, 3);
    ggml_tensor * mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, T, T, H);
    ggml_tensor * out;
    if (per_head_mask) {
        out = ggml_flash_attn_ext(ctx, q, kp, vp, mask, 1.0f, 0.0f, 0.0f);
    } else {
        ggml_tensor * q4 = ggml_view_4d(ctx, q,  HD, T, 1, H, q->nb[1],  q->nb[2],  q->nb[2],  0);
        ggml_tensor * k4 = ggml_view_4d(ctx, kp, HD, T, 1, H, kp->nb[1], kp->nb[2], kp->nb[2], 0);
        ggml_tensor * v4 = ggml_view_4d(ctx, vp, HD, T, 1, H, vp->nb[1], vp->nb[2], vp->nb[2], 0);
        ggml_tensor * m4 = ggml_reshape_4d(ctx, mask, T, T, 1, H);
        out = ggml_flash_attn_ext(ctx, q4, k4, v4, m4, 1.0f, 0.0f, 0.0f);
    }
    const bool ok = ggml_backend_supports_op(backend, out);
    ggml_free(ctx);
    return ok;
}

AttnPath select_attn_path(ggml_backend_t backend, int HD, int H) {
    AttnPath attn;
    attn.matmul_head_dim = attention_matmul_head_dim(backend, HD);
    if (!flash_attn_allowed(k_flash_attn_compiled, backend)) return attn;
    attn.per_head_mask = backend_runs_flash_attn_per_head_mask(backend, HD, H);
    attn.flash_attn    = backend_runs_flash_attn(backend, HD, H, attn.per_head_mask);
    return attn;
}

ggml_backend_t init_gpu_backend(int n_gpu_layers, bool verbose,
                                bool & out_skipped_unsupported_gpu,
                                bool & out_is_mali_vulkan) {
    out_skipped_unsupported_gpu = false;
    out_is_mali_vulkan = false;
    if (n_gpu_layers <= 0) return nullptr;

    ensure_backends_loaded();

    // Collect GPU/IGPU devices into per-tier buckets so we can apply the
    // tier policy after the walk. We keep the device handles + their
    // human-readable names for both the policy decision and the final
    // log line.
    struct Cand {
        ggml_backend_dev_t dev;
        const char *       name;
        const char *       desc;
        const char *       reg_name;
    };
    std::vector<Cand> opencl_adreno_700plus;
    std::vector<Cand> cuda_gpu_discrete;      // CUDA on a dGPU (typical NVIDIA), preferred over Vulkan on the same card
    std::vector<Cand> cuda_gpu_integrated;    // CUDA on an iGPU (Tegra/Jetson) — still CUDA-first, but ranks below cuda_gpu_discrete
    std::vector<Cand> other_gpu_discrete;     // Discrete non-OpenCL GPUs (Vulkan/Metal on dGPU, ...)
    std::vector<Cand> other_gpu_integrated;   // Integrated non-OpenCL GPUs (iGPU / UMA)
    std::vector<Cand> opencl_other;           // Non-Adreno OpenCL (e.g. desktop)
    int max_adreno_version = -1;

    const size_t n_dev = ggml_backend_dev_count();
    for (size_t i = 0; i < n_dev; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        if (type != GGML_BACKEND_DEVICE_TYPE_GPU &&
            type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }
        const char * name     = ggml_backend_dev_name(dev);
        const char * desc     = ggml_backend_dev_description(dev);
        const char * reg_name = dev_reg_name(dev);
        if (reg_name && std::strcmp(reg_name, "XDNA") == 0) continue;
        const bool   is_opencl = std::strcmp(reg_name, "OpenCL") == 0;
        const bool   is_cuda   = std::strcmp(reg_name, "CUDA")   == 0;
        const bool   is_integrated = (type == GGML_BACKEND_DEVICE_TYPE_IGPU);

        const int adreno_v = std::max(parse_adreno_version(name),
                                      parse_adreno_version(desc));
        if (adreno_v > max_adreno_version) max_adreno_version = adreno_v;

        if (is_opencl) {
            if (adreno_v >= 700) {
                opencl_adreno_700plus.push_back({dev, name, desc, reg_name});
            } else if (adreno_v >= 600 && adreno_v < 700) {
                const char * reported = name ? name : (desc ? desc : "unknown");
                const char * override_env = getenv("PARAKEET_ALLOW_ADRENO_6XX");
                if (!override_env || override_env[0] != '1') {
                    if (verbose) PARAKEET_LOG_WARN(
                        "parakeet: OpenCL device '%s' is Adreno 6xx; "
                        "skipping (7xx/8xx/X1E supported, set "
                        "PARAKEET_ALLOW_ADRENO_6XX=1 to override)\n",
                        reported);
                    // GPU detected but routed to CPU as known-bad (model_gpu_unsupported).
                    out_skipped_unsupported_gpu = true;
                    continue;
                }
                if (verbose) PARAKEET_LOG_INFO(
                    "parakeet: PARAKEET_ALLOW_ADRENO_6XX=1 set; "
                    "keeping OpenCL backend on '%s' anyway\n", reported);
                opencl_other.push_back({dev, name, desc, reg_name});
            } else {
                opencl_other.push_back({dev, name, desc, reg_name});
            }
        } else if (is_cuda) {
            // CUDA — preferred over Vulkan on NVIDIA. Split by device type so
            // a discrete NVIDIA CUDA adapter is tried before a Tegra/Jetson
            // integrated one, matching the GpuTier ranking in backend_util.h
            // (CudaDiscrete < CudaIntegrated). In practice a single host has
            // only one CUDA device, but keeping the split honest lets the
            // pure gpu_tier_for unit test guard the ordering.
            if (is_integrated) cuda_gpu_integrated.push_back({dev, name, desc, reg_name});
            else               cuda_gpu_discrete.push_back({dev, name, desc, reg_name});
        } else {
            // Non-OpenCL GPUs (Vulkan, Metal, Mali iGPU, Intel, ...). Mali
            // (Valhall) Vulkan runs the encoder + CTC/TDT/EOU heads correctly;
            // only its Sortformer head is routed to CPU (see sortformer_force_cpu),
            // so unlike the Adreno-6xx skip above it is not guarded out here.
            //
            // Bucketed by ggml device type so the tier policy can prefer
            // discrete GPUs over integrated ones, matching llm-llamacpp /
            // diffusion selection behavior.
            if (is_integrated) other_gpu_integrated.push_back({dev, name, desc, reg_name});
            else               other_gpu_discrete.push_back({dev, name, desc, reg_name});
        }
    }

    // Tier policy:
    //   1. Adreno 700+: prefer OpenCL (validated, faster than Vulkan
    //      on Snapdragon 8 Gen 2/3/4 etc.).
    //   2. CUDA on a discrete GPU: vendor-native on NVIDIA, and measurably
    //      faster than the same card's Vulkan adapter, so it outranks
    //      Vulkan when a build carries both.
    //   3. CUDA on an integrated GPU: Tegra/Jetson (CUDA reported as IGPU).
    //      Still CUDA-first over Vulkan, but ranked below discrete CUDA.
    //   4. Discrete non-OpenCL GPU: Vulkan on a dGPU, Metal on Apple,
    //      Vulkan on Adreno (Android), etc. Preferred over iGPUs to
    //      match the discrete-first behavior of llm/diffusion.
    //   5. Integrated non-OpenCL GPU: iGPU / UMA Vulkan as the fallback
    //      (Mali iGPU, Intel iGPU, ...).
    //   6. Last resort: any other OpenCL device (e.g. desktop OpenCL
    //      or non-Adreno mobile when no Vulkan is registered).
    auto try_init = [&](const std::vector<Cand> & bucket) -> ggml_backend_t {
        for (const Cand & c : bucket) {
            ggml_backend_t b = ggml_backend_dev_init(c.dev, nullptr);
            if (!b) continue;
            if (verbose) PARAKEET_LOG_INFO(
                "parakeet: using %s backend (%s)\n",
                c.reg_name && *c.reg_name ? c.reg_name : "GPU",
                c.name ? c.name : (c.desc ? c.desc : "unknown"));
            out_is_mali_vulkan = c.reg_name && std::strcmp(c.reg_name, "Vulkan") == 0
                              && (desc_is_mali(c.desc) || desc_is_mali(c.name));
            return b;
        }
        return nullptr;
    };

    if (!opencl_adreno_700plus.empty()) {
        if (ggml_backend_t b = try_init(opencl_adreno_700plus)) return b;
    }
    if (ggml_backend_t b = try_init(cuda_gpu_discrete)) return b;
    if (ggml_backend_t b = try_init(cuda_gpu_integrated)) return b;
    if (ggml_backend_t b = try_init(other_gpu_discrete)) return b;
    if (ggml_backend_t b = try_init(other_gpu_integrated)) return b;
    if (ggml_backend_t b = try_init(opencl_other)) return b;

    if (verbose) {
        if (max_adreno_version >= 600 && max_adreno_version < 700) {
            PARAKEET_LOG_INFO(
                "parakeet: only Adreno 6xx OpenCL detected (broken); "
                "falling back to CPU\n");
        } else {
            PARAKEET_LOG_INFO("parakeet: no GPU backend available, falling back to CPU\n");
        }
    }
    return nullptr;
}

ggml_backend_t init_explicit_backend(const std::string & requested, ggml_backend_t cpu,
                                     bool & out_is_mali_vulkan) {
    ensure_backends_loaded();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev || !backend_selection_matches(requested, dev_reg_name(dev),
                    ggml_backend_dev_name(dev), ggml_backend_dev_type(dev))) continue;
        // Reuse the initialized CPU instance so it is owned/freed only once.
        if (dev == ggml_backend_get_device(cpu)) return cpu;
        const char * name = ggml_backend_dev_name(dev);
        const char * desc = ggml_backend_dev_description(dev);
        const bool opencl = std::strcmp(dev_reg_name(dev), "OpenCL") == 0;
        const int adreno = std::max(parse_adreno_version(name), parse_adreno_version(desc));
        const char * override_env = getenv("PARAKEET_ALLOW_ADRENO_6XX");
        if (opencl && adreno >= 600 && adreno < 700 && (!override_env || override_env[0] != '1')) {
            continue;
        }
        if (ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr)) {
            out_is_mali_vulkan = std::strcmp(dev_reg_name(dev), "Vulkan") == 0 &&
                                (desc_is_mali(desc) || desc_is_mali(name));
            PARAKEET_LOG_INFO("parakeet: explicitly selected %s backend (%s)\n",
                              dev_reg_name(dev), name ? name : "unknown");
            return backend;
        }
    }
    PARAKEET_LOG_ERROR("parakeet: requested backend '%s' unavailable or failed initialization; no fallback\n",
                       requested.c_str());
    return nullptr;
}

// FastRPC resolves DSP-side libraries separately from ggml's host .so loader.
// Configure it before CPU initialization, which discovers *all* backends.
// Snapdragon devices supported by ggml-hexagon use DSP_LIBRARY_PATH; unlike
// legacy ADSP_LIBRARY_PATH, this prepends to the vendor's default search paths.
void prepare_hexagon_library_path(const std::string & dir) {
#ifdef __ANDROID__
    if (dir.empty()) return;
    if (dir.find(';') != std::string::npos) {
        PARAKEET_LOG_WARN("parakeet: cannot add backends_dir to FastRPC search path: directory contains ';'\n");
        return;
    }
    const char * prior = std::getenv("DSP_LIBRARY_PATH");
    if (!prior) prior = std::getenv("ADSP_LIBRARY_PATH");
    const std::string path = prepend_dsp_library_directory(dir, prior ? prior : "");
    if (setenv("DSP_LIBRARY_PATH", path.c_str(), 1) != 0) {
        PARAKEET_LOG_WARN("parakeet: failed to configure DSP_LIBRARY_PATH\n");
    }
#else
    (void) dir;
#endif
}

ggml_backend_t init_cpu_backend() {
    ensure_backends_loaded();
    return ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
}

ggml_backend_t init_blas_backend() {
    ensure_backends_loaded();
    const size_t n_dev = ggml_backend_dev_count();
    for (size_t i = 0; i < n_dev; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_ACCEL) continue;
        const char * reg_name = dev_reg_name(dev);
        if (std::strcmp(reg_name, "BLAS") != 0) continue;
        return ggml_backend_dev_init(dev, nullptr);
    }
    return nullptr;
}

int find_key(const gguf_context * g, const std::string & k) {
    return (int) gguf_find_key(g, k.c_str());
}

uint32_t get_u32(const gguf_context * g, const std::string & k, uint32_t fallback) {
    const int id = find_key(g, k);
    if (id < 0) return fallback;
    return gguf_get_val_u32(g, id);
}

float get_f32(const gguf_context * g, const std::string & k, float fallback) {
    const int id = find_key(g, k);
    if (id < 0) return fallback;
    return gguf_get_val_f32(g, id);
}

bool get_bool(const gguf_context * g, const std::string & k, bool fallback) {
    const int id = find_key(g, k);
    if (id < 0) return fallback;
    return gguf_get_val_bool(g, id);
}

int require_key(const gguf_context * g, const std::string & key) {
    const int id = find_key(g, key);
    if (id < 0) {
        throw std::runtime_error(
            "gguf: missing required metadata '" + key + "'");
    }
    return id;
}

uint32_t require_u32(
    const gguf_context * g,
    const std::string & key) {
    const int id = require_key(g, key);
    if (gguf_get_kv_type(g, id) != GGUF_TYPE_UINT32) {
        throw std::runtime_error(
            "gguf: metadata '" + key + "' must be uint32");
    }
    return gguf_get_val_u32(g, id);
}

int32_t require_i32(
    const gguf_context * g,
    const std::string & key) {
    const int id = require_key(g, key);
    if (gguf_get_kv_type(g, id) != GGUF_TYPE_INT32) {
        throw std::runtime_error(
            "gguf: metadata '" + key + "' must be int32");
    }
    return gguf_get_val_i32(g, id);
}

bool require_bool(
    const gguf_context * g,
    const std::string & key) {
    const int id = require_key(g, key);
    if (gguf_get_kv_type(g, id) != GGUF_TYPE_BOOL) {
        throw std::runtime_error(
            "gguf: metadata '" + key + "' must be bool");
    }
    return gguf_get_val_bool(g, id);
}

std::string require_string(
    const gguf_context * g,
    const std::string & key) {
    const int id = require_key(g, key);
    if (gguf_get_kv_type(g, id) != GGUF_TYPE_STRING) {
        throw std::runtime_error(
            "gguf: metadata '" + key + "' must be string");
    }
    return gguf_get_val_str(g, id);
}

std::vector<int32_t> require_i32_array(
    const gguf_context * g,
    const std::string & key) {
    const int id = require_key(g, key);
    if (gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(g, id) != GGUF_TYPE_INT32) {
        throw std::runtime_error(
            "gguf: metadata '" + key + "' must be int32[]");
    }

    const size_t count = gguf_get_arr_n(g, id);
    const auto * values = static_cast<const int32_t *>(
        gguf_get_arr_data(g, id));
    return std::vector<int32_t>(values, values + count);
}

std::vector<std::string> require_string_array(
    const gguf_context * g,
    const std::string & key) {
    const int id = require_key(g, key);
    if (gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(g, id) != GGUF_TYPE_STRING) {
        throw std::runtime_error(
            "gguf: metadata '" + key + "' must be string[]");
    }

    std::vector<std::string> values;
    const size_t count = gguf_get_arr_n(g, id);
    values.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        const char * value = gguf_get_arr_str(g, id, index);
        values.emplace_back(value ? value : "");
    }
    return values;
}

void load_nemotron_metadata(
    const gguf_context * g,
    ParakeetCtcModel & model) {
    (void) require_u32(g, "parakeet.encoder.d_model");
    (void) require_u32(g, "parakeet.encoder.n_layers");
    (void) require_u32(g, "parakeet.encoder.n_heads");
    (void) require_u32(g, "parakeet.encoder.conv_kernel");
    (void) require_u32(g, "parakeet.encoder.subsampling_factor");
    (void) require_bool(g, "parakeet.encoder.use_bias");
    (void) require_string(g, "parakeet.encoder.conv_norm_type");
    (void) require_bool(g, "parakeet.encoder.causal_downsampling");
    (void) require_string(g, "parakeet.encoder.conv_context_size");
    (void) require_string(g, "parakeet.encoder.att_context_style");
    (void) require_i32(g, "parakeet.encoder.att_context_size_left");
    (void) require_i32(g, "parakeet.encoder.att_context_size_right");
    (void) require_u32(g, "parakeet.preproc.sample_rate");
    (void) require_u32(g, "parakeet.preproc.n_mels");

    NemotronConfig & cfg = model.nemotron_cfg;

    cfg.pred_hidden =
        require_u32(g, "parakeet.nemotron.pred_hidden");
    cfg.pred_rnn_layers =
        require_u32(g, "parakeet.nemotron.pred_rnn_layers");
    cfg.joint_hidden =
        require_u32(g, "parakeet.nemotron.joint_hidden");
    cfg.max_symbols_per_step =
        require_u32(g, "parakeet.nemotron.max_symbols_per_step");

    model.vocab_size =
        require_u32(g, "parakeet.nemotron.vocab_size");
    model.blank_id =
        require_u32(g, "parakeet.nemotron.blank_id");

    cfg.num_prompts =
        require_u32(g, "parakeet.nemotron.num_prompts");
    cfg.prompt_width =
        require_u32(g, "parakeet.nemotron.prompt_width");
    cfg.prompt_input_width =
        require_u32(g, "parakeet.nemotron.prompt_input_width");
    cfg.left_context_frames =
        require_u32(g, "parakeet.nemotron.left_context_frames");
    cfg.cache_time_steps =
        require_u32(g, "parakeet.nemotron.cache_time_steps");

    cfg.allowed_right_context_frames = require_i32_array(
        g, "parakeet.nemotron.allowed_right_context_frames");
    cfg.allowed_chunk_ms = require_i32_array(
        g, "parakeet.nemotron.allowed_chunk_ms");
    cfg.default_locale = require_string(
        g, "parakeet.nemotron.default_locale");

    const std::vector<std::string> aliases = require_string_array(
        g, "parakeet.nemotron.locale_aliases");
    const std::vector<int32_t> prompt_ids = require_i32_array(
        g, "parakeet.nemotron.locale_prompt_ids");

    if (aliases.size() != prompt_ids.size()) {
        throw std::runtime_error(
            "gguf: Nemotron locale aliases and prompt IDs differ in length");
    }

    cfg.locale_prompts.reserve(aliases.size());
    for (size_t index = 0; index < aliases.size(); ++index) {
        cfg.locale_prompts.push_back(
            {aliases[index], prompt_ids[index]});
    }

    if (!require_bool(g, "parakeet.encoder.streaming.enabled")) {
        throw std::runtime_error(
            "gguf: Nemotron requires cache-aware streaming metadata");
    }
}

constexpr int kUnifiedDefaultLeftContextFrames = 70;
const std::vector<int32_t> kUnifiedDefaultChunkFrames = {1, 2, 7, 13};
const std::vector<int32_t> kUnifiedDefaultRightContextFrames = {0, 1, 2, 3, 4, 7, 13};

int symmetric_convolution_cache_frames(const EncoderConfig & encoder) {
    return (encoder.conv_kernel - 1) / 2;
}

bool has_unified_streaming_keys(const gguf_context * g) {
    return find_key(g, "parakeet.unified.left_context_frames") >= 0;
}

void load_unified_streaming_keys(const gguf_context * g, UnifiedStreamingConfig & cfg) {
    cfg.left_context_frames = require_u32(g, "parakeet.unified.left_context_frames");
    cfg.cache_time_steps = require_u32(g, "parakeet.unified.cache_time_steps");
    cfg.allowed_chunk_frames = require_i32_array(g, "parakeet.unified.allowed_chunk_frames");
    cfg.allowed_right_context_frames =
        require_i32_array(g, "parakeet.unified.allowed_right_context_frames");
    cfg.available = true;
}

void apply_unified_streaming_defaults(const EncoderConfig & encoder, UnifiedStreamingConfig & cfg) {
    cfg.left_context_frames = kUnifiedDefaultLeftContextFrames;
    cfg.cache_time_steps = symmetric_convolution_cache_frames(encoder);
    cfg.allowed_chunk_frames = kUnifiedDefaultChunkFrames;
    cfg.allowed_right_context_frames = kUnifiedDefaultRightContextFrames;
    cfg.available = true;
    PARAKEET_LOG_INFO("gguf: no parakeet.unified.* metadata; using the published "
                      "parakeet-unified-en-0.6b streaming contexts\n");
}

bool is_unified_streaming_candidate(const ParakeetCtcModel & model) {
    return model.model_type == ParakeetModelType::RNNT &&
           model.encoder_cfg.att_dynamic_chunking;
}

void load_unified_streaming_metadata(const gguf_context * g, ParakeetCtcModel & model) {
    if (!is_unified_streaming_candidate(model)) {
        return;
    }
    if (has_unified_streaming_keys(g)) {
        load_unified_streaming_keys(g, model.unified_cfg);
    } else {
        apply_unified_streaming_defaults(model.encoder_cfg, model.unified_cfg);
    }
    model.supports_streaming = true;
    validate_unified_streaming_model(model);
}

ggml_tensor * require_tensor(ggml_context * ctx, const std::string & name) {
    ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
    if (!t) throw std::runtime_error("gguf: missing required tensor '" + name + "'");
    return t;
}

ggml_tensor * maybe_tensor(ggml_context * ctx, const std::string & name) {
    return ggml_get_tensor(ctx, name.c_str());
}

void load_transducer_weights(ggml_context * ctx,
                             const std::string & prefix,
                             int pred_rnn_layers,
                             TdtWeights & weights) {
    weights.predict_embed = require_tensor(ctx, prefix + ".predict.embed.weight");
    for (int layer = 0; layer < pred_rnn_layers; ++layer) {
        const std::string path =
            prefix + ".predict.lstm." + std::to_string(layer) + ".";
        TdtLstmLayer lstm_layer;
        lstm_layer.w_ih = require_tensor(ctx, path + "w_ih");
        lstm_layer.w_hh = require_tensor(ctx, path + "w_hh");
        lstm_layer.b_ih = require_tensor(ctx, path + "b_ih");
        lstm_layer.b_hh = require_tensor(ctx, path + "b_hh");
        weights.lstm.push_back(lstm_layer);
    }
    weights.joint_enc_w = require_tensor(ctx, prefix + ".joint.enc.weight");
    weights.joint_enc_b = require_tensor(ctx, prefix + ".joint.enc.bias");
    weights.joint_pred_w = require_tensor(ctx, prefix + ".joint.pred.weight");
    weights.joint_pred_b = require_tensor(ctx, prefix + ".joint.pred.bias");
    weights.joint_out_w = require_tensor(ctx, prefix + ".joint.out.weight");
    weights.joint_out_b = require_tensor(ctx, prefix + ".joint.out.bias");
}

std::string get_str(const gguf_context * g, const std::string & k, const std::string & fallback) {
    const int id = find_key(g, k);
    if (id < 0) return fallback;
    return gguf_get_val_str(g, id);
}

std::vector<float> read_filterbank_to_vector(ggml_tensor * t) {
    const size_t n_elts = ggml_nelements(t);
    std::vector<float> out(n_elts);
    if (t->type != GGML_TYPE_F32) {
        throw std::runtime_error("preproc tensor type must be f32");
    }
    // Tensor storage may live on a non-CPU backend (Vulkan/CUDA/Metal), in
    // which case `t->data` is not a host-accessible pointer. Always go via
    // the backend buffer API; it copies device->host where needed and is a
    // no-op for CPU buffers.
    ggml_backend_tensor_get(t, out.data(), 0, n_elts * sizeof(float));
    return out;
}

void require_nemotron_shape(
    const ggml_tensor * tensor,
    const std::string & name,
    std::initializer_list<int64_t> expected) {
    if (!tensor) {
        throw std::runtime_error(
            "gguf: missing required Nemotron tensor '" + name + "'");
    }
    if (ggml_n_dims(tensor) != static_cast<int>(expected.size())) {
        throw std::runtime_error(
            "gguf: unexpected rank for Nemotron tensor '" + name + "'");
    }

    size_t dimension = 0;
    for (const int64_t expected_size : expected) {
        if (tensor->ne[dimension] != expected_size) {
            throw std::runtime_error(
                "gguf: unexpected shape for Nemotron tensor '" + name + "'");
        }
        ++dimension;
    }
}

constexpr int UNIFIED_MAX_LEFT_CONTEXT_FRAMES = 4096;
constexpr int UNIFIED_MAX_CHUNK_FRAMES = 4096;
constexpr int UNIFIED_MAX_RIGHT_CONTEXT_FRAMES = 4096;
constexpr int UNIFIED_MAX_CACHE_TIME_STEPS = 4096;
constexpr size_t UNIFIED_MAX_OPERATING_POINTS = 64;

bool in_range(int32_t value, int32_t low, int32_t high) {
    return value >= low && value <= high;
}

bool all_in_range(const std::vector<int32_t> & values, int32_t low, int32_t high) {
    return std::all_of(values.begin(), values.end(),
                       [low, high](int32_t v) { return in_range(v, low, high); });
}

bool all_positive(const std::vector<int32_t> & values) {
    return all_in_range(values, 1, UNIFIED_MAX_CHUNK_FRAMES);
}

bool all_non_negative(const std::vector<int32_t> & values) {
    return all_in_range(values, 0, UNIFIED_MAX_RIGHT_CONTEXT_FRAMES);
}

}

void validate_unified_streaming_model(const ParakeetCtcModel & model) {
    const EncoderConfig & encoder = model.encoder_cfg;
    const UnifiedStreamingConfig & cfg = model.unified_cfg;
    if (model.model_type != ParakeetModelType::RNNT || !cfg.available) {
        throw std::runtime_error(
            "validate_unified_streaming_model called without unified streaming metadata");
    }
    if (!encoder.att_dynamic_chunking ||
        encoder.causal_downsampling ||
        encoder.conv_causal ||
        encoder.conv_kernel % 2 == 0) {
        throw std::runtime_error(
            "gguf: incompatible Unified FastConformer geometry");
    }
    if (!in_range(cfg.left_context_frames, 1, UNIFIED_MAX_LEFT_CONTEXT_FRAMES) ||
        !in_range(cfg.cache_time_steps, 0, UNIFIED_MAX_CACHE_TIME_STEPS) ||
        cfg.cache_time_steps != symmetric_convolution_cache_frames(encoder)) {
        throw std::runtime_error(
            "gguf: incompatible Unified streaming cache geometry");
    }
    if (cfg.allowed_chunk_frames.empty() ||
        cfg.allowed_right_context_frames.empty() ||
        cfg.allowed_chunk_frames.size() > UNIFIED_MAX_OPERATING_POINTS ||
        cfg.allowed_right_context_frames.size() > UNIFIED_MAX_OPERATING_POINTS ||
        !all_positive(cfg.allowed_chunk_frames) ||
        !all_non_negative(cfg.allowed_right_context_frames)) {
        throw std::runtime_error(
            "gguf: invalid Unified streaming operating points");
    }
    if (!model.supports_streaming) {
        throw std::runtime_error(
            "gguf: Unified streaming metadata present but streaming is disabled");
    }
}

void validate_nemotron_model(const ParakeetCtcModel & model) {
    static const std::vector<int32_t> expected_right_contexts = {
        0, 1, 3, 6, 13,
    };
    static const std::vector<int32_t> expected_chunk_ms = {
        80, 160, 320, 560, 1120,
    };
    constexpr const char * expected_variant =
        "nemotron-3.5-asr-streaming-0.6b-v1";

    if (model.model_type != ParakeetModelType::NEMOTRON) {
        throw std::runtime_error(
            "validate_nemotron_model called for non-Nemotron model");
    }
    if (model.model_variant != expected_variant) {
        throw std::runtime_error(
            "gguf: unsupported Nemotron variant '" +
            model.model_variant + "'");
    }

    const EncoderConfig & encoder = model.encoder_cfg;
    if (encoder.n_layers != 24 ||
        encoder.d_model != 1024 ||
        encoder.n_heads != 8 ||
        encoder.conv_kernel != 9 ||
        encoder.subsampling_factor != 8 ||
        encoder.conv_norm_type != ConvNormType::LayerNorm ||
        encoder.use_bias ||
        !encoder.causal_downsampling ||
        !encoder.conv_causal ||
        !encoder.att_chunked_limited ||
        encoder.att_context_left != 56 ||
        encoder.att_context_right != 3) {
        throw std::runtime_error(
            "gguf: incompatible Nemotron FastConformer geometry");
    }
    if (!model.supports_streaming ||
        model.mel_cfg.sample_rate != 16000 ||
        model.mel_cfg.n_mels != 128) {
        throw std::runtime_error(
            "gguf: incompatible Nemotron streaming/preprocessor metadata");
    }

    const NemotronConfig & config = model.nemotron_cfg;
    if (model.vocab_size != 13087 ||
        model.blank_id != 13087 ||
        config.pred_hidden != 640 ||
        config.pred_rnn_layers != 2 ||
        config.joint_hidden != 640 ||
        config.max_symbols_per_step != 10 ||
        config.num_prompts != 128 ||
        config.prompt_width != 128 ||
        config.prompt_input_width !=
            encoder.d_model + config.prompt_width ||
        config.left_context_frames != 56 ||
        config.cache_time_steps != 8 ||
        config.default_locale != "auto") {
        throw std::runtime_error(
            "gguf: incompatible Nemotron prompt/RNNT metadata");
    }
    if (config.allowed_right_context_frames != expected_right_contexts ||
        config.allowed_chunk_ms != expected_chunk_ms) {
        throw std::runtime_error(
            "gguf: unsupported Nemotron streaming operating points");
    }
    if (config.locale_prompts.empty()) {
        throw std::runtime_error(
            "gguf: Nemotron locale table is empty");
    }

    bool has_default_locale = false;
    for (const NemotronLocalePrompt & locale : config.locale_prompts) {
        if (locale.alias.empty() ||
            locale.prompt_id < 0 ||
            locale.prompt_id >= config.num_prompts) {
            throw std::runtime_error(
                "gguf: invalid Nemotron locale prompt entry");
        }
        if (locale.alias == config.default_locale) {
            has_default_locale = true;
        }
    }
    if (!has_default_locale) {
        throw std::runtime_error(
            "gguf: Nemotron default locale is absent from the locale table");
    }

    const NemotronWeights & weights = model.nemotron;
    require_nemotron_shape(
        weights.prompt.proj_0_w,
        "nemotron.prompt.proj.0.weight",
        {1152, 2048});
    require_nemotron_shape(
        weights.prompt.proj_0_b,
        "nemotron.prompt.proj.0.bias",
        {2048});
    require_nemotron_shape(
        weights.prompt.proj_2_w,
        "nemotron.prompt.proj.2.weight",
        {2048, 1024});
    require_nemotron_shape(
        weights.prompt.proj_2_b,
        "nemotron.prompt.proj.2.bias",
        {1024});

    const RnntWeights & rnnt = weights.rnnt;
    require_nemotron_shape(
        rnnt.predict_embed,
        "nemotron.predict.embed.weight",
        {640, 13088});
    if (rnnt.lstm.size() != 2) {
        throw std::runtime_error(
            "gguf: Nemotron predictor must contain two LSTM layers");
    }
    for (size_t layer = 0; layer < rnnt.lstm.size(); ++layer) {
        const TdtLstmLayer & lstm = rnnt.lstm[layer];
        const std::string prefix =
            "nemotron.predict.lstm." + std::to_string(layer);
        require_nemotron_shape(lstm.w_ih, prefix + ".w_ih", {640, 2560});
        require_nemotron_shape(lstm.w_hh, prefix + ".w_hh", {640, 2560});
        require_nemotron_shape(lstm.b_ih, prefix + ".b_ih", {2560});
        require_nemotron_shape(lstm.b_hh, prefix + ".b_hh", {2560});
    }

    require_nemotron_shape(
        rnnt.joint_enc_w,
        "nemotron.joint.enc.weight",
        {1024, 640});
    require_nemotron_shape(
        rnnt.joint_enc_b,
        "nemotron.joint.enc.bias",
        {640});
    require_nemotron_shape(
        rnnt.joint_pred_w,
        "nemotron.joint.pred.weight",
        {640, 640});
    require_nemotron_shape(
        rnnt.joint_pred_b,
        "nemotron.joint.pred.bias",
        {640});
    require_nemotron_shape(
        rnnt.joint_out_w,
        "nemotron.joint.out.weight",
        {640, 13088});
    require_nemotron_shape(
        rnnt.joint_out_b,
        "nemotron.joint.out.bias",
        {13088});
}

int32_t resolve_nemotron_prompt_id(
    const ParakeetCtcModel & model,
    const std::string & language) {
    if (model.model_type != ParakeetModelType::NEMOTRON) {
        throw std::runtime_error(
            "resolve_nemotron_prompt_id called for non-Nemotron model");
    }

    const NemotronConfig & config = model.nemotron_cfg;
    const std::string requested =
        language.empty() ? config.default_locale : language;
    for (const NemotronLocalePrompt & locale : config.locale_prompts) {
        if (locale.alias == requested) {
            return locale.prompt_id;
        }
    }

    std::string supported;
    for (const NemotronLocalePrompt & locale : config.locale_prompts) {
        if (!supported.empty()) {
            supported += ", ";
        }
        supported += locale.alias;
    }
    throw std::runtime_error(
        "unsupported Nemotron locale '" + requested +
        "'; supported locales: " + supported);
}

void set_backends_directory(const std::string & dir) {
    std::lock_guard<std::mutex> lock(g_backends_dir_mutex);
    if (g_backends_loaded.load(std::memory_order_acquire)) {
        // Registry already populated for this process. We can't
        // re-scan a different directory mid-flight (ggml's registry
        // is a process-wide singleton), so log the conflict at most
        // once and otherwise stay silent on subsequent identical
        // sets (the common case when a host instantiates several
        // Engines back-to-back from the same backends folder, or
        // when the second value happens to match the recorded one).
        if (dir != g_recorded_backends_dir &&
            !g_backends_dir_warned.exchange(true)) {
            if (g_recorded_backends_dir.empty()) {
                // First Engine constructed without an explicit
                // backends_dir, so ggml's compile-time default
                // search path was used. The current caller wanted
                // a specific dir but missed the window.
                PARAKEET_LOG_WARN(
                    "parakeet: set_backends_directory('%s') ignored -- the "
                    "ggml-backend registry was already populated against "
                    "ggml's default search path (no explicit backends_dir on "
                    "the first Engine). Call set_backends_directory() (or "
                    "construct an Engine with backends_dir set) before the "
                    "first Engine to influence which directory is scanned.\n",
                    dir.c_str());
            } else {
                PARAKEET_LOG_WARN(
                    "parakeet: set_backends_directory('%s') ignored -- backends "
                    "already loaded from '%s' earlier in this process.\n",
                    dir.c_str(), g_recorded_backends_dir.c_str());
            }
        }
        return;
    }
    g_backends_dir = dir;
}

void set_opencl_cache_dir(const std::string & dir) {
#if defined(__ANDROID__)
    // Same "first Engine wins" contract as set_backends_directory:
    // ggml-opencl reads $GGML_OPENCL_CACHE_DIR once per process at
    // backend init (before the first kernel build), so a setenv
    // after init is effectively a no-op on the cache binding. Gate
    // on the shared g_backends_loaded flag because the OpenCL
    // backend is registered at the same `ggml_backend_load_all*`
    // call that flips the flag -- conservative because it might
    // still take effect when the host hasn't yet instantiated a
    // GPU device, but matches what the engine-ctor documentation
    // promises and avoids the same silent-failure mode as
    // set_backends_directory's previous gate.
    std::lock_guard<std::mutex> lock(g_backends_dir_mutex);
    if (g_backends_loaded.load(std::memory_order_acquire)) {
        if (!dir.empty() && dir != g_recorded_opencl_cache_dir &&
            !g_opencl_cache_dir_warned.exchange(true)) {
            if (g_recorded_opencl_cache_dir.empty()) {
                PARAKEET_LOG_WARN(
                    "parakeet: set_opencl_cache_dir('%s') ignored -- backends "
                    "were already loaded with no explicit OpenCL cache dir "
                    "earlier in this process ($GGML_OPENCL_CACHE_DIR either "
                    "unset or set by another consumer). Call "
                    "set_opencl_cache_dir() before the first Engine to take "
                    "effect.\n",
                    dir.c_str());
            } else {
                PARAKEET_LOG_WARN(
                    "parakeet: set_opencl_cache_dir('%s') ignored -- "
                    "$GGML_OPENCL_CACHE_DIR already pinned to '%s' earlier in "
                    "this process.\n",
                    dir.c_str(), g_recorded_opencl_cache_dir.c_str());
            }
        }
        return;
    }
    if (dir.empty()) return;
    // ggml-opencl's program-binary-cache patch reads this once per
    // process at backend init (before the first kernel build). Set
    // it before constructing the first Engine; later calls don't
    // re-bind the cache but cost nothing.
    ::setenv("GGML_OPENCL_CACHE_DIR", dir.c_str(), /*overwrite=*/1);
    g_recorded_opencl_cache_dir = dir;
#else
    (void) dir;
#endif
}

// Tripwire: build_tdt_cpu_weights() hand-enumerates every field; fail the build if one is added/removed.
static_assert(sizeof(TdtLstmLayer) == 4 * sizeof(ggml_tensor *),
              "TdtLstmLayer layout changed; update build_tdt_cpu_weights()");
static_assert(sizeof(TdtWeights) == 7 * sizeof(ggml_tensor *) + sizeof(std::vector<TdtLstmLayer>),
              "TdtWeights layout changed; update build_tdt_cpu_weights()");

// Duplicate the TDT / RNN-T decoder weights into a fresh context + buffer on
// the CPU backend, for an encoder on OpenCL: ggml-opencl drops the in-place
// state writes the decoder graphs rely on, so they run on the CPU backend and
// must read weights that live there. Measure mode sizes the buffer only.
static bool build_tdt_cpu_weights(ParakeetCtcModel::Impl * impl,
                                  const TdtWeights & src,
                                  TdtWeights & dst,
                                  size_t * measure_bytes = nullptr) {
    dst = TdtWeights{};
    dst.lstm.resize(src.lstm.size());

    struct Field { ggml_tensor * src; ggml_tensor ** dst; };
    std::vector<Field> fields;
    auto add = [&](ggml_tensor * s, ggml_tensor ** d) {
        if (s) fields.push_back({ s, d });
    };
    add(src.predict_embed, &dst.predict_embed);
    for (size_t i = 0; i < src.lstm.size(); ++i) {
        add(src.lstm[i].w_ih, &dst.lstm[i].w_ih);  add(src.lstm[i].w_hh, &dst.lstm[i].w_hh);
        add(src.lstm[i].b_ih, &dst.lstm[i].b_ih);  add(src.lstm[i].b_hh, &dst.lstm[i].b_hh);
    }
    add(src.joint_enc_w,  &dst.joint_enc_w);   add(src.joint_enc_b,  &dst.joint_enc_b);
    add(src.joint_pred_w, &dst.joint_pred_w);  add(src.joint_pred_b, &dst.joint_pred_b);
    add(src.joint_out_w,  &dst.joint_out_w);   add(src.joint_out_b,  &dst.joint_out_b);

    ggml_init_params p = {
        /*mem_size=*/   ggml_tensor_overhead() * (fields.size() + 8),
        /*mem_buffer=*/ nullptr,
        /*no_alloc=*/   true,
    };
    ggml_context * cpu_ctx = ggml_init(p);
    if (!cpu_ctx) return false;

    for (Field & f : fields) {
        ggml_tensor * t = ggml_new_tensor(cpu_ctx, f.src->type, GGML_MAX_DIMS, f.src->ne);
        ggml_set_name(t, ggml_get_name(f.src));
        *f.dst = t;
    }

    if (measure_bytes) {
        *measure_bytes = ggml_backend_alloc_ctx_tensors_from_buft_size(
            cpu_ctx, ggml_backend_get_default_buffer_type(impl->backend_cpu));
        ggml_free(cpu_ctx);
        dst = TdtWeights{};
        return true;
    }

    impl->tdt_cpu_ctx    = cpu_ctx;
    impl->tdt_cpu_buffer = ggml_backend_alloc_ctx_tensors(cpu_ctx, impl->backend_cpu);
    if (!impl->tdt_cpu_buffer) return false;

    std::vector<uint8_t> buf;
    for (Field & f : fields) {
        const size_t nb = ggml_nbytes(f.src);
        buf.resize(nb);
        ggml_backend_tensor_get(f.src, buf.data(), 0, nb);
        ggml_backend_tensor_set(*f.dst, buf.data(), 0, nb);
    }
    return true;
}

static const TdtWeights * transducer_decoder_weights(const ParakeetCtcModel & m) {
    if (m.model_type == ParakeetModelType::TDT)  return &m.tdt;
    if (m.model_type == ParakeetModelType::RNNT) return &m.rnnt;
    return nullptr;
}

bool model_build_tdt_cpu_weights(ParakeetCtcModel & m) {
    const TdtWeights * src = transducer_decoder_weights(m);
    if (!m.impl || !src) return false;
    return build_tdt_cpu_weights(m.impl.get(), *src, m.tdt_cpu);
}

bool model_tdt_decoder_on_cpu(const ParakeetCtcModel & m) {
    return m.tdt_cpu.predict_embed != nullptr;
}

// An encoder on OpenCL leaves the TDT / RNN-T decoder graphs to the CPU
// backend (see build_tdt_cpu_weights); Nemotron keeps its host decoder.
static bool decoder_moves_to_cpu(const ParakeetCtcModel & m, const ParakeetCtcModel::Impl * impl) {
    return impl->backend_gpu && backend_is_opencl(impl->backend_gpu) &&
           transducer_decoder_weights(m) != nullptr;
}

// Tripwire: build_sortformer_cpu_weights() hand-enumerates every field; fail the build if one is added/removed.
static_assert(sizeof(SortformerTransformerBlock) == 16 * sizeof(ggml_tensor *),
              "SortformerTransformerBlock layout changed; update build_sortformer_cpu_weights()");
static_assert(sizeof(SortformerWeights) ==
                  6 * sizeof(ggml_tensor *) + sizeof(std::vector<SortformerTransformerBlock>),
              "SortformerWeights layout changed; update build_sortformer_cpu_weights()");

// Duplicate the Sortformer head weights into a fresh context + buffer on the
// CPU backend. Used on Mali-Vulkan, where the head runs on CPU while its encoder
// stays on the GPU: the head graph must read weights that live on the same
// backend it executes on. Returns false on allocation failure.
//
// Measure mode (`measure_bytes != nullptr`, fit projection): the same shadow
// tensors are created into a throwaway context, their CPU buffer is *sized*
// into `*measure_bytes` instead of allocated, no weight bytes are copied, and
// `impl` / `dst` are left untouched.
static bool build_sortformer_cpu_weights(ParakeetCtcModel::Impl * impl,
                                         const SortformerWeights & src,
                                         SortformerWeights & dst,
                                         size_t * measure_bytes = nullptr) {
    dst = SortformerWeights{};
    dst.transformer.resize(src.transformer.size());

    struct Field { ggml_tensor * src; ggml_tensor ** dst; };
    std::vector<Field> fields;
    auto add = [&](ggml_tensor * s, ggml_tensor ** d) {
        if (s) fields.push_back({ s, d });
    };
    add(src.encoder_proj_w, &dst.encoder_proj_w);
    add(src.encoder_proj_b, &dst.encoder_proj_b);
    for (size_t i = 0; i < src.transformer.size(); ++i) {
        const SortformerTransformerBlock & sb = src.transformer[i];
        SortformerTransformerBlock       & db = dst.transformer[i];
        add(sb.attn_q_w,  &db.attn_q_w);   add(sb.attn_q_b,  &db.attn_q_b);
        add(sb.attn_k_w,  &db.attn_k_w);   add(sb.attn_k_b,  &db.attn_k_b);
        add(sb.attn_v_w,  &db.attn_v_w);   add(sb.attn_v_b,  &db.attn_v_b);
        add(sb.attn_o_w,  &db.attn_o_w);   add(sb.attn_o_b,  &db.attn_o_b);
        add(sb.ln1_w,     &db.ln1_w);      add(sb.ln1_b,     &db.ln1_b);
        add(sb.ffn_in_w,  &db.ffn_in_w);   add(sb.ffn_in_b,  &db.ffn_in_b);
        add(sb.ffn_out_w, &db.ffn_out_w);  add(sb.ffn_out_b, &db.ffn_out_b);
        add(sb.ln2_w,     &db.ln2_w);      add(sb.ln2_b,     &db.ln2_b);
    }
    add(src.head_h2h_w, &dst.head_h2h_w);  add(src.head_h2h_b, &dst.head_h2h_b);
    add(src.head_h2s_w, &dst.head_h2s_w);  add(src.head_h2s_b, &dst.head_h2s_b);

    const size_t n = fields.size();
    ggml_init_params p = {
        /*mem_size=*/   ggml_tensor_overhead() * (n + 8),
        /*mem_buffer=*/ nullptr,
        /*no_alloc=*/   true,
    };
    ggml_context * cpu_ctx = ggml_init(p);
    if (!cpu_ctx) return false;

    for (Field & f : fields) {
        ggml_tensor * t = ggml_new_tensor(cpu_ctx, f.src->type,
                                          GGML_MAX_DIMS, f.src->ne);
        ggml_set_name(t, ggml_get_name(f.src));
        *f.dst = t;
    }

    if (measure_bytes) {
        // dst now holds handles into cpu_ctx, which is freed below: the caller
        // must pass a throwaway dst in measure mode and discard it.
        *measure_bytes = ggml_backend_alloc_ctx_tensors_from_buft_size(
            cpu_ctx, ggml_backend_get_default_buffer_type(impl->backend_cpu));
        ggml_free(cpu_ctx);
        return true;
    }

    impl->sortformer_cpu_ctx = cpu_ctx;
    impl->sortformer_cpu_buffer =
        ggml_backend_alloc_ctx_tensors(impl->sortformer_cpu_ctx, impl->backend_cpu);
    if (!impl->sortformer_cpu_buffer) return false;

    std::vector<uint8_t> buf;
    for (Field & f : fields) {
        const size_t nb = ggml_nbytes(f.src);
        buf.resize(nb);
        ggml_backend_tensor_get(f.src, buf.data(), 0, nb);
        ggml_backend_tensor_set(*f.dst, buf.data(), 0, nb);
    }
    return true;
}

// ---------------------------------------------------------------------------
// CPU "extra" buffer-type (repack) weight placement.
//
// ggml's CPU backend exposes optional extra buffer types (CPU_REPACK) that
// rewrite quantized weights at upload time into an interleaved layout
// (e.g. q4_0 -> q4_0x8) so mul_mat runs on the block-parallel
// dotprod/i8mm/AVX2 GEMM kernels instead of the generic row-wise vec_dot
// path; without it q4_0 is measurably *slower* than q8_0 on CPU.
//
// A repacked tensor is only valid as the direct src0 of ggml_mul_mat: the
// buffer has no get_tensor readback, and reshape/view nodes drop the repack
// traits (the view computes on the repacked bytes as if they were plain
// q4_0). Placement is therefore allowlisted to the encoder GEMM weights.
// Everything else must stay in the default buffer:
//   - TDT/EOU predictor+joint weights: read back via ggml_backend_tensor_get
//     (dequantize_to_f32) on the CPU decode path,
//   - tdt/eou predict embeddings: consumed by ggml_get_rows,
//   - ctc.decoder.weight: 1025 rows, repack needs a multiple of 4/8,
//   - subsampling / depthwise conv kernels: consumed by conv ops,
//   - Sortformer head weights: cloned via tensor_get on Mali-Vulkan.

static bool repack_allowlisted(const char * name) {
    // encoder.blk.*.conv.pw{1,2}.weight are eligible because
    // conformer_conv_graph uses the 2D weight directly (no reshape) when the
    // GGUF already stores it squeezed to (d_model, N).
    static const char * const k_suffixes[] = {
        ".ff1.linear1.weight", ".ff1.linear2.weight",
        ".ff2.linear1.weight", ".ff2.linear2.weight",
        ".attn.q.weight", ".attn.k.weight", ".attn.v.weight",
        ".attn.qkv.weight", ".attn.out.weight", ".attn.pos.weight",
        ".conv.pw1.weight", ".conv.pw2.weight",
    };
    if (std::strcmp(name, "encoder.subsampling.out.weight") == 0) {
        return true;
    }
    if (std::strncmp(name, "encoder.blk.", 12) != 0) {
        return false;
    }
    const size_t len = std::strlen(name);
    for (const char * suf : k_suffixes) {
        const size_t slen = std::strlen(suf);
        if (len > slen && std::strcmp(name + len - slen, suf) == 0) {
            return true;
        }
    }
    return false;
}

// True when `dev` can run mul_mat(w, f32) with `w` placed in `buft`.
// Mirrors whisper.cpp's weight_buft_supported: whether an extra buffer type
// applies depends on runtime CPU features (dotprod/i8mm/AVX2) and the
// weight's type/shape, and it only answers through
// ggml_backend_dev_supports_op with the weight's buffer assigned, so probe
// with a dummy op instead of duplicating the layout rules here.
static bool cpu_extra_buft_supports_mul_mat(ggml_backend_dev_t dev,
                                            ggml_backend_buffer_type_t buft,
                                            ggml_tensor * w) {
    ggml_init_params ip = {
        /*mem_size=*/   3 * ggml_tensor_overhead(),
        /*mem_buffer=*/ nullptr,
        /*no_alloc=*/   true,
    };
    ggml_context * c = ggml_init(ip);
    if (!c) {
        return false;
    }
    ggml_tensor * b  = ggml_new_tensor_2d(c, GGML_TYPE_F32, w->ne[0], 512);
    ggml_tensor * op = affine_mul_mat(c, w, b);
    bool ok = false;
    w->buffer = ggml_backend_buft_alloc_buffer(buft, 0);
    if (w->buffer) {
        ok = ggml_backend_dev_supports_op(dev, op);
        ggml_backend_buffer_free(w->buffer);
        w->buffer = nullptr;
    }
    ggml_free(c);
    return ok;
}

// Allocate the repack-eligible weights of `ctx` into CPU extra buffer types.
// Must run before ggml_backend_alloc_ctx_tensors, which then sweeps up every
// tensor this pass left unallocated. Returns the buffers it created (empty
// when the platform has no repack kernels for these weights). The subsequent
// whole-tensor ggml_backend_tensor_set in the GGUF data loop performs the
// actual repack on upload.
//
// Measure mode (`measure_bytes != nullptr`, fit projection): the selection
// loop runs identically, but instead of allocating the buffers the padded
// group sizes are summed into `*measure_bytes` and the selected tensors are
// marked externally-allocated (dummy non-NULL `data`) -- the same "claimed"
// signal a real ggml_tallocr placement leaves -- so the later
// ggml_backend_alloc_ctx_tensors_from_buft_size sweep skips them exactly like
// the real allocation pass would. Returns no buffers.
static std::vector<ggml_backend_buffer_t> alloc_cpu_repack_weights(
        ggml_context * ctx, ggml_backend_t backend_cpu, bool verbose,
        size_t * measure_bytes = nullptr) {
    std::vector<ggml_backend_buffer_t> buffers;

    ggml_backend_dev_t cpu_dev = ggml_backend_get_device(backend_cpu);
    ggml_backend_reg_t cpu_reg = cpu_dev ? ggml_backend_dev_backend_reg(cpu_dev) : nullptr;
    if (!cpu_reg) {
        return buffers;
    }
    // Registry proc lookup, not a direct ggml-cpu symbol: parakeet must stay
    // linkable under GGML_BACKEND_DL where ggml-cpu is a runtime-loaded module.
    auto get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)
        ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts");
    ggml_backend_buffer_type_t * extra_bufts =
        get_extra_bufts_fn ? get_extra_bufts_fn(cpu_dev) : nullptr;

    for (; extra_bufts && *extra_bufts; ++extra_bufts) {
        ggml_backend_buffer_type_t buft = *extra_bufts;
        const size_t align = ggml_backend_buft_get_alignment(buft);

        std::vector<ggml_tensor *> group;
        size_t group_bytes = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
            if (t->data || t->view_src)                     continue; // claimed by an earlier buft
            if (!ggml_is_quantized(t->type))                continue;
            if (!repack_allowlisted(ggml_get_name(t)))      continue;
            if (!cpu_extra_buft_supports_mul_mat(cpu_dev, buft, t)) continue;
            group.push_back(t);
            group_bytes += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), align);
        }
        if (group.empty()) {
            continue;
        }

        if (measure_bytes) {
            *measure_bytes += group_bytes;
            for (ggml_tensor * t : group) {
                t->data = reinterpret_cast<void *>(static_cast<uintptr_t>(1));
            }
            continue;
        }

        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, group_bytes);
        if (!buf) {
            PARAKEET_LOG_WARN("gguf: failed to allocate %zu MB %s buffer; "
                              "falling back to the default CPU buffer\n",
                              group_bytes / (1024 * 1024), ggml_backend_buft_name(buft));
            continue;
        }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        ggml_tallocr talloc = ggml_tallocr_new(buf);
        size_t placed = 0;
        for (ggml_tensor * t : group) {
            // Cannot fail: group_bytes reserves each tensor's alloc size padded
            // to the buffer alignment, matching ggml_tallocr's placement math.
            if (ggml_tallocr_alloc(&talloc, t) != GGML_STATUS_SUCCESS) {
                PARAKEET_LOG_ERROR("gguf: tensor '%s' did not fit the %s buffer\n",
                                   ggml_get_name(t), ggml_backend_buft_name(buft));
                break;
            }
            placed++;
        }
        buffers.push_back(buf);
        if (verbose) {
            PARAKEET_LOG_INFO("parakeet: %zu encoder weights -> %s (%.1f MB)\n",
                              placed, ggml_backend_buft_name(buft),
                              group_bytes / (1024.0 * 1024.0));
        }
    }
    return buffers;
}

static constexpr const char * kParakeetArchitecture = "parakeet-ctc";

static bool gguf_has_encoder_weights(const gguf_context * g) {
    return get_str(g, "general.architecture", "") != kParakeetArchitecture ||
           gguf_find_tensor(g, "encoder.subsampling.conv0.weight") >= 0;
}

#ifdef PARAKEET_USE_COREML
static bool path_is_directory(const std::string & path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

static constexpr const char * kCoremlOnlyEnv    = "PARAKEET_COREML_ONLY";
static constexpr const char * kCoremlDecoderUnitsEnv = "PARAKEET_COREML_DECODER_COMPUTE_UNITS";


// Loads the TDT decoder sidecar when one sits next to the GGUF. Only TDT
// models have one: the sidecar is the prediction network plus joint. The lane
// decoder cross-checks the geometry it reports against the GGUF before use.
static void maybe_init_coreml_decoder(const std::string & gguf_path,
                                      ParakeetCtcModel  & model,
                                      bool                verbose) {
    if (!model.impl->coreml->resolved().enabled()) return;
    if (model.model_type != ParakeetModelType::TDT) return;

    const std::string path = coreml_decoder_sidecar_path(gguf_path);
    if (!path_is_directory(path)) {
        if (verbose) PARAKEET_LOG_INFO("parakeet: no Core ML decoder at '%s'; using ggml decoder\n", path.c_str());
        return;
    }
    // Synchronous, with the resolved units like every other loader, after
    // the decoder-only PARAKEET_COREML_DECODER_COMPUTE_UNITS unless the
    // units were set explicitly. With none set, the TranscribePsy EU decoder
    // keeps its measured CPU default on macOS and every other decoder its
    // cpu_and_ane default.
    const pcm::Resolved & resolved = model.impl->coreml->resolved();
    const char * units = loader_units(resolved);
    if (resolved.units_source != pcm::Source::explicit_field) {
        if (const char * decoder_units = std::getenv(kCoremlDecoderUnitsEnv)) units = decoder_units;
    }
    model.impl->ctx_coreml_dec = parakeet_coreml_decoder_init_with_units(
        path.c_str(), units, model.model_variant == "transcribepsy-250m-eu");
    if (model.impl->ctx_coreml_dec == nullptr) {
        PARAKEET_LOG_WARN("parakeet: failed to load Core ML decoder at '%s'; falling back to ggml decoder\n", path.c_str());
        return;
    }
    int64_t lanes = 0, h_pred = 0, h_dim = 0, d_enc = 0, width = 0, v_out = 0;
    parakeet_coreml_decoder_geometry(model.impl->ctx_coreml_dec,
                                     &lanes, &h_pred, &h_dim, &d_enc, &width, &v_out);
    PARAKEET_LOG_INFO("parakeet: Core ML decoder loaded from '%s' "
                      "(lanes=%lld width=%lld h_pred=%lld h_dim=%lld d_enc=%lld v_out=%lld)\n",
                      path.c_str(), (long long) lanes, (long long) width, (long long) h_pred,
                      (long long) h_dim, (long long) d_enc, (long long) v_out);
}

// Whether the encoder sidecar has to be loaded before weight allocation: the
// GGUF carries no encoder weights, or PARAKEET_COREML_ONLY asks to skip them.
static bool coreml_must_own_encoder(bool gguf_has_encoder) {
    return !gguf_has_encoder || std::getenv(kCoremlOnlyEnv) != nullptr;
}

static bool coreml_can_own_encoder(const gguf_context * g,
                                   parakeet_coreml_context * ctx) {
    if (!ctx) return false;
    const bool needs_native_tdt = get_str(g, "parakeet.model.type", "ctc") == "tdt" &&
        get_str(g, "parakeet.model_variant", "") != "transcribepsy-250m-eu" &&
        parakeet_coreml_seam_context_frames(ctx) <= 0;
    if (needs_native_tdt) {
        PARAKEET_LOG_WARN("parakeet: TDT sidecar has no validated seam policy; native encoder "
                          "weights are required for oversized inputs and push streaming\n");
    }
    return !needs_native_tdt;
}

// Reads the offline-encoder test straight off GGUF metadata, for the early
// sidecar load that runs before the authoritative parse into EncoderConfig.
static bool gguf_encoder_is_offline(const gguf_context * g) {
    const int id_l = find_key(g, "parakeet.encoder.att_context_size_left");
    const int id_r = find_key(g, "parakeet.encoder.att_context_size_right");
    return (id_l < 0 || gguf_get_val_i32(g, id_l) < 0)
        && (id_r < 0 || gguf_get_val_i32(g, id_r) < 0)
        && get_str(g, "parakeet.encoder.att_context_style", "regular") != "chunked_limited"
        && get_str(g, "parakeet.encoder.conv_context_size", "default") != "causal"
        && !get_bool(g, "parakeet.encoder.causal_downsampling", false);
}

constexpr int kTranscribePsyEuMelBins = 128;
enum class EuCoremlKind { FixedAne, FlexibleCpu, PreciseGpu };
// `units`: the resolved compute units (CoremlOptions::compute_units, then
// PARAKEET_COREML_COMPUTE_UNITS; null when unset). The EU loaders pick their
// placement from them instead of reading the environment.
static parakeet_coreml_context * load_eu_coreml_encoder(
        const std::string & path, const char * route, int layers, const char * units,
        EuCoremlKind kind = EuCoremlKind::FixedAne, int expected_mel_frames = 0,
        bool shared = false);

static parakeet_coreml_context * load_eu_shared_encoders(
        const std::string & gguf_path, int layers, int gate_layers, int mels,
        int gate_mel_frames, const char * units, ParakeetCtcModel::Impl * impl,
        bool load_auxiliaries = true) {
    const auto path = coreml_eu_shared_sidecar_path(gguf_path);
    if (!path_is_directory(path)) return nullptr;
    auto * english = load_eu_coreml_encoder(path, "en", layers, units, EuCoremlKind::FixedAne, 0, true);
    // Precise policies only need the English capacity/seam metadata. Ownership
    // and fixed ANE inference require all routes to validate together.
    if (!load_auxiliaries) return english;
    auto * other = load_eu_coreml_encoder(path, "other", layers, units, EuCoremlKind::FixedAne, 0, true);
    auto * gate = load_eu_coreml_encoder(path, "gate", gate_layers, units, EuCoremlKind::FixedAne, 0, true);
    const auto capacity = parakeet_coreml_fixed_mel_frames(english, mels);
    if (!english || !other || !gate || capacity < gate_mel_frames ||
        capacity != parakeet_coreml_fixed_mel_frames(other, mels) ||
        capacity != parakeet_coreml_fixed_mel_frames(gate, mels)) {
        parakeet_coreml_free(english);
        parakeet_coreml_free(other);
        parakeet_coreml_free(gate);
        PARAKEET_LOG_WARN("parakeet: incompatible full-EU shared encoder; trying separate sidecars\n");
        return nullptr;
    }
    impl->ctx_coreml_eu_other = other;
    impl->ctx_coreml_eu_gate = gate;
    return english;
}

static parakeet_coreml_context * init_eu_coreml_owner(
        const gguf_context * g, const std::string & gguf_path, const char * units,
        ParakeetCtcModel::Impl * impl) {
    const int mels = (int) require_u32(g, "parakeet.preproc.n_mels");
    const int rate = (int) require_u32(g, "parakeet.preproc.sample_rate");
    const int hop = (int) require_u32(g, "parakeet.preproc.hop_length");
    const int layers = (int) get_u32(g, "parakeet.encoder.n_layers", 0);
    if (hop > 0) {
        auto * shared = load_eu_shared_encoders(gguf_path, layers,
            (int) get_u32(g, "parakeet.eu.gate_layer", 0) + 1, mels,
            1 + kTranscribePsyEuGateSeconds * rate / hop, units, impl);
        if (shared) return shared;
    }
    auto * english = load_eu_coreml_encoder(coreml_encoder_sidecar_path(gguf_path), "en", layers, units);
    auto * other = load_eu_coreml_encoder(coreml_eu_other_sidecar_path(gguf_path), "other", layers, units);
    auto * gate = load_eu_coreml_encoder(coreml_eu_gate_sidecar_path(gguf_path), "gate",
                                       (int) get_u32(g, "parakeet.eu.gate_layer", 0) + 1, units);
    const bool compatible = english && other && gate &&
        parakeet_coreml_fixed_mel_frames(english, mels) ==
            parakeet_coreml_fixed_mel_frames(other, mels) &&
        hop > 0 && parakeet_coreml_fixed_mel_frames(gate, mels) >=
            1 + kTranscribePsyEuGateSeconds * rate / hop;
    if (!compatible) {
        parakeet_coreml_free(english);
        parakeet_coreml_free(other);
        parakeet_coreml_free(gate);
        PARAKEET_LOG_WARN("parakeet: full-EU Core ML-only loading requires compatible en/other/gate sidecars\n");
        return nullptr;
    }
    impl->ctx_coreml_eu_other = other;
    impl->ctx_coreml_eu_gate = gate;
    return english;
}

// Loads the encoder sidecar before weight allocation so the `encoder.*`
// tensors can be skipped. Only the offline fixed-capacity families qualify
// (the same ones coreml_encoder_eligibility pads); returns null, after logging
// why, when no sidecar loads. coreml_can_own_encoder then decides whether the
// loaded sidecar may own the encoder.
static parakeet_coreml_context * init_coreml_only_encoder(const gguf_context * g,
                                                          const std::string & gguf_path,
                                                          const pcm::Resolved & coreml,
                                                          bool verbose,
                                                          ParakeetCtcModel::Impl * impl) {
    if (!coreml.enabled()) {
        PARAKEET_LOG_WARN("parakeet: Core ML is disabled (%s); the GGUF's ggml encoder weights are kept\n",
                          coreml.disabled_reason.c_str());
        return nullptr;
    }
    if (get_str(g, "parakeet.model_variant", "") == "transcribepsy-250m-eu")
        return init_eu_coreml_owner(g, gguf_path, loader_units(coreml), impl);
    const std::string type = get_str(g, "parakeet.model.type", "ctc");
    if ((type != "tdt" && type != "ctc" && type != "rnnt") || !gguf_encoder_is_offline(g)) {
        PARAKEET_LOG_WARN("parakeet: a Core ML-only encoder needs an offline CTC, RNN-T or TDT "
                          "model; '%s' keeps its ggml encoder\n", type.c_str());
        return nullptr;
    }
    const std::string path = coreml_encoder_sidecar_path(gguf_path);
    if (!path_is_directory(path)) {
        PARAKEET_LOG_WARN("parakeet: no Core ML encoder at '%s'\n", path.c_str());
        return nullptr;
    }
    // A standard TDT sidecar owns the encoder only with a validated seam
    // context, which only the fixed-window Neural Engine export provides
    // (coreml_can_own_encoder). When the bundle's metadata shows another
    // export, a load here would only be thrown away, so it is skipped: a
    // full GGUF gets the background encoder slot and an encoder-less one is
    // rejected, as after the load. A bundle whose metadata is missing,
    // unreadable or does not list its inputs is loaded to find out.
    if (type == "tdt" && parakeet_coreml_encoder_export_kind(path.c_str()) == 0) {
        PARAKEET_LOG_WARN("parakeet: TDT sidecar has no validated seam policy; native encoder "
                          "weights are required for oversized inputs and push streaming\n");
        return nullptr;
    }
    // Synchronous on purpose: ownership fixes how the weights are
    // allocated, so it is decided before allocation and never in the background.
    parakeet_coreml_context * ctx = parakeet_coreml_init_with_units(path.c_str(), loader_units(coreml));
    if (ctx == nullptr) {
        PARAKEET_LOG_WARN("parakeet: failed to load Core ML encoder at '%s'\n", path.c_str());
        return nullptr;
    }
    if (verbose) PARAKEET_LOG_INFO("parakeet: Core ML encoder loaded from '%s'\n", path.c_str());
    return ctx;
}

std::atomic<GgufLoadObserver> g_gguf_load_observer{nullptr};

void set_gguf_load_observer(GgufLoadObserver fn) { g_gguf_load_observer.store(fn); }

static void maybe_init_coreml_diarizers(const std::string & gguf_path,
                                        ParakeetCtcModel  & model,
                                        bool                verbose) {
    if (std::getenv(kCoremlDiarizerDisableEnv) != nullptr) return;
    // Nemotron 3 Diarization and Sortformer v2.1 capacities without an
    // adjacent bundle are generated from the library's templates (when this
    // instance can).
    const std::vector<CoremlDiarizerTemplate> & templates =
        diarizer_templates_for(model.model_type, model.model_variant);
    // Without the descriptor's identity nothing is generated (source_changed).
    pcm::SourceStat loaded = model.impl->source;
    loaded.changed = loaded.changed || !model.impl->source_ok;
    model.impl->coreml_diarizers = discover_coreml_diarizers(gguf_path, *model.impl->coreml, templates, loaded);
    const std::vector<int> capacities =
        model.impl->coreml_diarizers ? model.impl->coreml_diarizers->capacities() : std::vector<int>();
    if (!capacities.empty()) {
        for (int capacity : capacities) {
            const std::string path = coreml_diarizer_sidecar_path(gguf_path, capacity);
            if (path_is_directory(path)) {
                PARAKEET_LOG_INFO("parakeet: Core ML diarizer available for %d frames at '%s'\n", capacity,
                                  path.c_str());
            } else {
                PARAKEET_LOG_INFO("parakeet: Core ML diarizer for %d frames is generated from the GGUF into '%s'\n",
                                  capacity, model.impl->coreml->resolved().cache_dir.c_str());
            }
        }
    } else if (verbose) {
        PARAKEET_LOG_INFO("parakeet: no Core ML diarizer beside '%s'; using ggml\n",
            gguf_path.c_str());
    }
}

static parakeet_coreml_context * load_eu_coreml_encoder(
        const std::string & path, const char * route, int layers, const char * units,
        EuCoremlKind kind, int expected_mel_frames, bool shared) {
    if (!path_is_directory(path)) return nullptr;
    const auto started = std::chrono::steady_clock::now();
    const bool flexible = kind == EuCoremlKind::FlexibleCpu;
    const bool fixed_cpu = kind == EuCoremlKind::FixedAne &&
        !parakeet::coreml_eu_loads_fixed_auxiliaries(units);
    auto * ctx = shared ? parakeet_coreml_init_eu_function(path.c_str(), route, fixed_cpu) :
        (flexible || fixed_cpu) ? parakeet_coreml_init_cpu(path.c_str()) :
        kind == EuCoremlKind::PreciseGpu ? parakeet_coreml_init_gpu(path.c_str()) :
                                         parakeet_coreml_init_ane(path.c_str());
    if (!ctx) return nullptr;
    if (!parakeet_coreml_matches_eu_route(ctx, route, layers, flexible) ||
        (kind == EuCoremlKind::PreciseGpu && (!parakeet_coreml_is_fp32(ctx) ||
            parakeet_coreml_fixed_mel_frames(ctx, kTranscribePsyEuMelBins) != expected_mel_frames))) {
        PARAKEET_LOG_WARN("parakeet: incompatible full-EU Core ML %s sidecar '%s'; using ggml\n",
                          route, path.c_str());
        parakeet_coreml_free(ctx);
        return nullptr;
    }
    const double load_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    PARAKEET_LOG_INFO("parakeet: full-EU Core ML %s encoder loaded from '%s' (%.1f ms)\n",
                      route, path.c_str(), load_ms);
    return ctx;
}

// TranscribePsy EU's fixed programs load synchronously here, not as runtime
// slots: the shared multifunction program when one sits next to the GGUF;
// otherwise the English one, then (unless the units exclude the Neural
// Engine) the other-language and gate programs.
static void init_eu_coreml_encoders(const std::string & path, ParakeetCtcModel & model) {
    auto & impl = *model.impl;
    const char * units = loader_units(impl.coreml->resolved());
    if (model.mel_cfg.hop_length > 0) {
        impl.ctx_coreml = load_eu_shared_encoders(path, model.encoder_cfg.n_layers,
            model.eu_gate.layer + 1, model.mel_cfg.n_mels,
            1 + kTranscribePsyEuGateSeconds * model.mel_cfg.sample_rate / model.mel_cfg.hop_length, units, &impl,
            parakeet::coreml_eu_loads_fixed_auxiliaries(units));
        if (impl.ctx_coreml) return;
    }
    impl.ctx_coreml = load_eu_coreml_encoder(
        coreml_encoder_sidecar_path(path), "en", model.encoder_cfg.n_layers, units);
    // Keep the English handle's shape/seam metadata. Policies excluding ANE
    // use precise programs and never select these two fixed auxiliaries.
    if (!parakeet::coreml_eu_loads_fixed_auxiliaries(units)) return;
    impl.ctx_coreml_eu_other = load_eu_coreml_encoder(
        coreml_eu_other_sidecar_path(path), "other", model.encoder_cfg.n_layers, units);
    impl.ctx_coreml_eu_gate = load_eu_coreml_encoder(
        coreml_eu_gate_sidecar_path(path), "gate", model.eu_gate.layer + 1, units);
    if (impl.ctx_coreml && impl.ctx_coreml_eu_other &&
        parakeet_coreml_fixed_mel_frames(impl.ctx_coreml, model.mel_cfg.n_mels) !=
        parakeet_coreml_fixed_mel_frames(impl.ctx_coreml_eu_other, model.mel_cfg.n_mels)) {
        PARAKEET_LOG_WARN("parakeet: full-EU Core ML route capacities differ; using ggml for other\n");
        parakeet_coreml_free(impl.ctx_coreml_eu_other);
        impl.ctx_coreml_eu_other = nullptr;
    }
}

// Runs on the runtime's load lane with the resolved compute units.
static pcm::LoadResult load_encoder_sidecar(const pcm::LoadRequest & request) {
    pcm::LoadResult result;
    parakeet_coreml_load_error error{};
    parakeet_coreml_context * ctx = parakeet_coreml_init_with_units_error(
        request.artifact_path.c_str(), pcm::to_string(request.units), &error);
    if (ctx == nullptr) {
        result.error = sidecar_load_error("Core ML " + request.stage, error);
        PARAKEET_LOG_WARN("parakeet: failed to load Core ML %s at '%s' (%s); falling back to ggml\n",
                          request.stage.c_str(), request.artifact_path.c_str(),
                          result.error.message.c_str());
        return result;
    }
    PARAKEET_LOG_INFO("parakeet: Core ML %s loaded from '%s'\n",
                      request.stage.c_str(), request.artifact_path.c_str());
    result.model = std::shared_ptr<void>(ctx, [](void * p) {
        parakeet_coreml_free(static_cast<parakeet_coreml_context *>(p));
    });
    return result;
}

static SidecarRequest encoder_sidecar_request(const char * stage, const std::string & path,
                                              bool minimal) {
    SidecarRequest request;
    request.stage = stage;
    request.path = path;
    request.present = path_is_directory(path);
    // The fixed-window Neural Engine export is pinned to CPU and Neural
    // Engine; every other export keeps all processors (today's defaults).
    request.stage_default = request.present && parakeet_coreml_encoder_prefers_ane(path.c_str())
        ? pcm::ComputeUnits::cpu_and_ane
        : pcm::ComputeUnits::all;
    request.load_fn = &load_encoder_sidecar;
    request.geometry = sidecar_geometry(std::string(stage) == "bypass-encoder" ? SidecarLoader::bypass_encoder
                                                                               : SidecarLoader::encoder);
    request.minimal = minimal;
    return request;
}

// Registers the encoder sidecar slots. Presence-driven and additive, as
// before, but the load is a background job: the model is usable
// on ggml at once, and a missing, disabled or failed sidecar leaves ggml.
static void maybe_init_coreml_encoder(const std::string & gguf_path,
                                      ParakeetCtcModel  & model,
                                      bool                verbose) {
    // TranscribePsy EU gets no slot: its fixed programs load synchronously
    // here (or, when they own the encoder, before weight allocation) and its
    // flexible and precise programs on first use, as before the runtime.
    if (model.model_variant == "transcribepsy-250m-eu") {
        if (!model.impl->coreml->resolved().enabled()) {
            if (verbose) {
                PARAKEET_LOG_INFO("parakeet: Core ML disabled (%s); using ggml\n",
                                  model.impl->coreml->resolved().disabled_reason.c_str());
            }
            return;
        }
        model.impl->eu_gguf_path = gguf_path;
        model.impl->eu_gate_flexible_path = coreml_eu_gate_sidecar_path(gguf_path);
        model.impl->eu_gate_flexible_path.insert(model.impl->eu_gate_flexible_path.size() - std::string(".mlmodelc").size(), "-flexible");
        model.impl->eu_flexible_path = coreml_eu_flexible_sidecar_path(gguf_path, false);
        model.impl->eu_other_flexible_path = coreml_eu_flexible_sidecar_path(gguf_path, true);
        if (model.impl->ctx_coreml == nullptr) init_eu_coreml_encoders(gguf_path, model);
        return;
    }
    if (model.impl->ctx_coreml != nullptr) return;  // owns the encoder (loaded before weight allocation)
    const bool supported_sortformer =
        model.model_type == ParakeetModelType::SORTFORMER &&
        model.model_variant == "sortformer-streaming-v2.1-aosc";
    if (model.model_type != ParakeetModelType::CTC &&
        model.model_type != ParakeetModelType::RNNT &&
        model.model_type != ParakeetModelType::TDT &&
        model.model_type != ParakeetModelType::EOU &&
        model.model_type != ParakeetModelType::NEMOTRON &&
        !supported_sortformer) {
        if (verbose) {
            PARAKEET_LOG_INFO(
                "parakeet: Core ML encoder supports CTC/IndicConformer, Unified RNN-T, "
                "TDT, EOU, Nemotron, and Sortformer v2.1; "
                "using ggml for model type %s variant '%s'\n",
                model_type_name(model.model_type), model.model_variant.c_str());
        }
        return;
    }
    if (!model.impl->coreml->resolved().enabled() && verbose) {
        PARAKEET_LOG_INFO("parakeet: Core ML disabled (%s); using ggml\n",
                          model.impl->coreml->resolved().disabled_reason.c_str());
    }

    // A diarizer capacity that can serve a pass is the Sortformer minimal
    // set; otherwise (no diarizer, or only status-only template capacities)
    // the encoders are.
    if (supported_sortformer) maybe_init_coreml_diarizers(gguf_path, model, verbose);
    const bool encoder_minimal = encoder_sidecars_minimal(
        model.impl->coreml_diarizers ? model.impl->coreml_diarizers->capacities() : std::vector<int>());

    const std::string path = coreml_encoder_sidecar_path(gguf_path);
    model.impl->encoder_slot = model.impl->coreml->add(
        encoder_sidecar_request("encoder", path, encoder_minimal));
    if (verbose) {
        PARAKEET_LOG_INFO("parakeet: Core ML encoder at '%s': %s\n", path.c_str(),
                          pcm::state_label(model.impl->encoder_slot->state(), "").c_str());
    }

    if (supported_sortformer) {
        const std::string bypass_path = coreml_bypass_encoder_sidecar_path(gguf_path);
        model.impl->bypass_slot = model.impl->coreml->add(
            encoder_sidecar_request("bypass-encoder", bypass_path, encoder_minimal));
        if (verbose) {
            PARAKEET_LOG_INFO("parakeet: Core ML bypass encoder at '%s': %s\n", bypass_path.c_str(),
                              pcm::state_label(model.impl->bypass_slot->state(), "").c_str());
        }
    }
}
#endif  // PARAKEET_USE_COREML

static constexpr int kDiarizationEncoderWidth = 512;
static constexpr int kDiarizationEncoderLayers = 31;
static constexpr int kDiarizationAttentionHeads = 8;
static constexpr int kDiarizationFeedForwardWidth = 2048;
static constexpr int kDiarizationOutputWidth = 192;
static constexpr int kDiarizationSpeakers = 8;
static constexpr int kDiarizationStackFactor = 8;
static constexpr int kDiarizationOutputStride = 1;

void validate_nemotron_silence_embedding_type(const ggml_tensor * tensor) {
    if (!tensor || tensor->type != GGML_TYPE_F32) {
        throw std::runtime_error("Nemotron 3 Diarization tensor type mismatch: learnable_sil_emb");
    }
}

static ggml_tensor * require_diarization_tensor(
    ggml_context * context, const std::string & name,
    int64_t width, int64_t height = 1) {
    ggml_tensor * tensor = ggml_get_tensor(context, name.c_str());
    if (!tensor || tensor->ne[0] != width || tensor->ne[1] != height) {
        throw std::runtime_error("Nemotron 3 Diarization tensor shape mismatch: " + name);
    }
    return tensor;
}

static void load_diarization_layers(
    ggml_context * context, const NemotronDiarizationConfig & cfg,
    NemotronDiarizationWeights & weights) {
    weights.layers.resize(cfg.encoder_layers);
    for (int index = 0; index < cfg.encoder_layers; ++index) {
        auto & layer = weights.layers[index];
        const std::string prefix = "encoder.layers." + std::to_string(index) + ".";
        layer.norm1_w = require_diarization_tensor(context, prefix + "norm1.weight", cfg.encoder_width);
        layer.norm1_b = require_diarization_tensor(context, prefix + "norm1.bias", cfg.encoder_width);
        layer.qkv_w = require_diarization_tensor(context, prefix + "attn.w_qkv.weight", cfg.encoder_width, 3 * cfg.encoder_width);
        layer.attention_w = require_diarization_tensor(context, prefix + "attn.out_proj.weight", cfg.encoder_width, cfg.encoder_width);
        layer.attention_b = require_diarization_tensor(context, prefix + "attn.out_proj.bias", cfg.encoder_width);
        layer.norm2_w = require_diarization_tensor(context, prefix + "norm2.weight", cfg.encoder_width);
        layer.norm2_b = require_diarization_tensor(context, prefix + "norm2.bias", cfg.encoder_width);
        layer.feed_forward_in_w = require_diarization_tensor(context, prefix + "ffn.net.0.weight", cfg.encoder_width, cfg.feed_forward_width);
        layer.feed_forward_in_b = require_diarization_tensor(context, prefix + "ffn.net.0.bias", cfg.feed_forward_width);
        layer.feed_forward_out_w = require_diarization_tensor(context, prefix + "ffn.net.3.weight", cfg.feed_forward_width, cfg.encoder_width);
        layer.feed_forward_out_b = require_diarization_tensor(context, prefix + "ffn.net.3.bias", cfg.encoder_width);
    }
}

static void load_diarization_weights(ggml_context * context, ParakeetCtcModel & model) {
    const auto & cfg = model.nemotron_diarization_cfg;
    auto & weights = model.nemotron_diarization;
    const int stacked_width = model.mel_cfg.n_mels * cfg.subsampling_factor;
    weights.feature_projection = require_diarization_tensor(context, "encoder.pre_encode.proj.weight", stacked_width, cfg.encoder_width);
    weights.embedding_norm_w = require_diarization_tensor(context, "encoder.embed_norm.weight", cfg.encoder_width);
    weights.embedding_norm_b = require_diarization_tensor(context, "encoder.embed_norm.bias", cfg.encoder_width);
    load_diarization_layers(context, cfg, weights);
    weights.final_norm_w = require_diarization_tensor(context, "encoder.final_norm.weight", cfg.encoder_width);
    weights.final_norm_b = require_diarization_tensor(context, "encoder.final_norm.bias", cfg.encoder_width);
    weights.encoder_projection_w = require_diarization_tensor(context, "encoder_proj.weight", cfg.encoder_width, cfg.output_width);
    weights.encoder_projection_b = require_diarization_tensor(context, "encoder_proj.bias", cfg.output_width);
    weights.upsample_w = ggml_get_tensor(context, "subpixel_upsample.weight");
    if (!weights.upsample_w || weights.upsample_w->ne[0] != 3 ||
        weights.upsample_w->ne[1] != cfg.output_width ||
        weights.upsample_w->ne[2] !=
            static_cast<int64_t>(cfg.output_width) * cfg.subsampling_factor) {
        throw std::runtime_error("Nemotron 3 Diarization tensor shape mismatch: subpixel_upsample.weight");
    }
    weights.upsample_b = require_diarization_tensor(context, "subpixel_upsample.bias",
        static_cast<int64_t>(cfg.output_width) * cfg.subsampling_factor);
    weights.hidden_w = require_diarization_tensor(context, "head.first_hidden_to_hidden.weight", cfg.output_width, cfg.output_width);
    weights.hidden_b = require_diarization_tensor(context, "head.first_hidden_to_hidden.bias", cfg.output_width);
    weights.speakers_w = require_diarization_tensor(context, "head.single_hidden_to_spks.weight", cfg.output_width, cfg.speakers);
    weights.speakers_b = require_diarization_tensor(context, "head.single_hidden_to_spks.bias", cfg.speakers);
    weights.silence_embedding = require_diarization_tensor(context, "learnable_sil_emb", cfg.encoder_width);
    validate_nemotron_silence_embedding_type(weights.silence_embedding);
}

static void load_diarization_metadata(gguf_context * gguf, ParakeetCtcModel & model) {
    if (get_str(gguf, "sortformer.version", "") != "v3" ||
        get_str(gguf, "sortformer.encoder.type", "") != "transformer_rope" ||
        get_str(gguf, "sortformer.encoder.subsampling_type", "") != "feature_stacking" ||
        !get_bool(gguf, "sortformer.high_resolution", false) ||
        !get_bool(gguf, "sortformer.learnable_silence", false) ||
        !get_bool(gguf, "sortformer.encoder.pre_block_norm", false) ||
        get_bool(gguf, "sortformer.encoder.qkv_bias", true) ||
        get_bool(gguf, "sortformer.encoder.qk_norm", false) ||
        get_u32(gguf, "sortformer.transformer.n_layers", 1) != 0) {
        throw std::runtime_error("unsupported Nemotron 3 Diarization model contract");
    }
    auto & cfg = model.nemotron_diarization_cfg;
    cfg.encoder_width = get_u32(gguf, "sortformer.encoder.d_model", 0);
    cfg.encoder_layers = get_u32(gguf, "sortformer.encoder.n_layers", 0);
    cfg.attention_heads = get_u32(gguf, "sortformer.encoder.n_heads", 0);
    cfg.feed_forward_width = get_u32(gguf, "sortformer.encoder.d_ff", 0);
    cfg.output_width = get_u32(gguf, "sortformer.transformer.hidden_size", 0);
    cfg.speakers = get_u32(gguf, "sortformer.num_speakers", 0);
    cfg.subsampling_factor = get_u32(gguf, "sortformer.encoder.subsampling_factor", 0);
    cfg.output_stride = get_u32(gguf, "sortformer.output_subsampling_factor", 0);
    cfg.position_limit = get_u32(gguf, "sortformer.encoder.pos_emb_max_len", 0);
    cfg.rope_base = get_f32(gguf, "sortformer.encoder.rope_base", 0.0f);
    cfg.rotary_fraction = get_f32(gguf, "sortformer.encoder.rotary_fraction", 0.0f);
    if (cfg.encoder_width != kDiarizationEncoderWidth ||
        cfg.encoder_layers != kDiarizationEncoderLayers ||
        cfg.attention_heads != kDiarizationAttentionHeads ||
        cfg.feed_forward_width != kDiarizationFeedForwardWidth ||
        cfg.output_width != kDiarizationOutputWidth ||
        cfg.speakers != kDiarizationSpeakers ||
        cfg.subsampling_factor != kDiarizationStackFactor ||
        get_u32(gguf, "sortformer.upsample_factor", 0) != kDiarizationStackFactor ||
        cfg.output_stride != kDiarizationOutputStride ||
        cfg.position_limit < 1 || cfg.rope_base <= 0.0f ||
        cfg.rotary_fraction != 1.0f) {
        throw std::runtime_error("unsupported Nemotron 3 Diarization dimensions");
    }
    model.model_type = ParakeetModelType::NEMOTRON_DIARIZATION;
    model.model_variant = "nemotron-3-diarization";
    model.mel_cfg.sample_rate = get_u32(gguf, "sortformer.preprocessor.sample_rate", 0);
    model.mel_cfg.n_fft = get_u32(gguf, "sortformer.preprocessor.n_fft", 0);
    model.mel_cfg.n_mels = get_u32(gguf, "sortformer.preprocessor.features", 0);
    model.mel_cfg.win_length = static_cast<int>(
        get_f32(gguf, "sortformer.preprocessor.window_size", 0.0f) * model.mel_cfg.sample_rate + 0.5f);
    model.mel_cfg.hop_length = static_cast<int>(
        get_f32(gguf, "sortformer.preprocessor.window_stride", 0.0f) * model.mel_cfg.sample_rate + 0.5f);
    model.mel_cfg.preemph = get_f32(gguf, "sortformer.preprocessor.preemph", 0.0f);
    model.mel_cfg.log_zero_guard_value = get_f32(gguf, "sortformer.preprocessor.log_zero_guard", kDefaultLogZeroGuard);
    model.mel_cfg.normalize = MelNormalize::None;
    model.mel_cfg.right_zero_pad = true;
    if (model.mel_cfg.sample_rate != 16000 || model.mel_cfg.n_fft != 512 ||
        model.mel_cfg.n_mels != 128 || model.mel_cfg.win_length != 400 ||
        model.mel_cfg.hop_length != 160 ||
        get_str(gguf, "sortformer.preprocessor.normalize", "") != "NA") {
        throw std::runtime_error("unsupported Nemotron 3 Diarization frontend");
    }
}

static void make_diarization_window(MelConfig & config) {
    config.window.resize(config.win_length);
    const float denominator = static_cast<float>(config.win_length - 1);
    const float full_circle = 2.0f * 3.14159265358979323846f;
    for (int sample = 0; sample < config.win_length; ++sample) {
        config.window[sample] = 0.5f - 0.5f * std::cos(full_circle * sample / denominator);
    }
}

static void exclude_weights_from_graph_measure(ggml_context * ctx) {
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t;
         t = ggml_get_next_tensor(ctx, t)) {
        if (!t->data && !t->view_src) {
            t->data = reinterpret_cast<void *>(static_cast<uintptr_t>(1));
        }
    }
}

// True for the weights the encoder owns. The GGUF keeps them under the
// `encoder.` prefix; everything outside it is the decoder head and the mel
// front-end.
static bool tensor_is_encoder(const char * name) {
    return std::strncmp(name, "encoder.", 8) == 0;
}

// ggml_backend_alloc_ctx_tensors_from_buft, minus the encoder: allocates every
// unallocated tensor in the context except the `encoder.*` weights, for use
// when Core ML owns the encoder. Sizing and placement mirror the stock helper,
// so a tensor that fits there fits here.
static ggml_backend_buffer_t alloc_ctx_tensors_without_encoder(ggml_context * ctx,
                                                               ggml_backend_buffer_type_t buft) {
    const size_t align = ggml_backend_buft_get_alignment(buft);

    std::vector<ggml_tensor *> keep;
    size_t bytes = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (t->data || t->view_src)              continue;
        if (tensor_is_encoder(ggml_get_name(t))) continue;
        keep.push_back(t);
        bytes += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), align);
    }
    if (keep.empty()) return nullptr;

    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, bytes);
    if (!buf) return nullptr;

    ggml_tallocr talloc = ggml_tallocr_new(buf);
    for (ggml_tensor * t : keep) {
        if (ggml_tallocr_alloc(&talloc, t) != GGML_STATUS_SUCCESS) {
            PARAKEET_LOG_ERROR("gguf: tensor '%s' did not fit the weights buffer\n",
                               ggml_get_name(t));
            ggml_backend_buffer_free(buf);
            return nullptr;
        }
    }
    return buf;
}

static void load_conformer_block_weights(ggml_context * ctx, const ParakeetCtcModel & model,
                                         bool gate_qkv_stack, int i, BlockWeights & b) {
    const std::string p = "encoder.blk." + std::to_string(i) + ".";

    b.norm_ff1_w  = require_tensor(ctx, p + "norm_ff1.weight");
    b.norm_ff1_b  = require_tensor(ctx, p + "norm_ff1.bias");
    b.ff1_l1_w    = require_tensor(ctx, p + "ff1.linear1.weight");
    b.ff1_l1_b    = maybe_tensor(ctx, p + "ff1.linear1.bias");
    b.ff1_l2_w    = require_tensor(ctx, p + "ff1.linear2.weight");
    b.ff1_l2_b    = maybe_tensor(ctx, p + "ff1.linear2.bias");

    b.norm_attn_w = require_tensor(ctx, p + "norm_attn.weight");
    b.norm_attn_b = require_tensor(ctx, p + "norm_attn.bias");
    b.attn_q_w    = require_tensor(ctx, p + "attn.q.weight");
    b.attn_q_b    = maybe_tensor(ctx, p + "attn.q.bias");
    b.attn_k_w    = require_tensor(ctx, p + "attn.k.weight");
    b.attn_k_b    = maybe_tensor(ctx, p + "attn.k.bias");
    b.attn_v_w    = require_tensor(ctx, p + "attn.v.weight");
    b.attn_v_b    = maybe_tensor(ctx, p + "attn.v.bias");
    b.attn_qkv_w  = gate_qkv_stack ? maybe_tensor(ctx, p + "attn.qkv.weight") : nullptr;
    b.attn_qkv_b  = gate_qkv_stack ? maybe_tensor(ctx, p + "attn.qkv.bias")   : nullptr;
    b.attn_out_w  = require_tensor(ctx, p + "attn.out.weight");
    b.attn_out_b  = maybe_tensor(ctx, p + "attn.out.bias");
    b.attn_pos_w  = require_tensor(ctx, p + "attn.pos.weight");
    b.pos_bias_u  = require_tensor(ctx, p + "attn.pos_bias_u");
    b.pos_bias_v  = require_tensor(ctx, p + "attn.pos_bias_v");

    b.norm_conv_w = require_tensor(ctx, p + "norm_conv.weight");
    b.norm_conv_b = require_tensor(ctx, p + "norm_conv.bias");
    b.conv_pw1_w  = require_tensor(ctx, p + "conv.pw1.weight");
    b.conv_pw1_b  = maybe_tensor(ctx, p + "conv.pw1.bias");
    b.conv_dw_w   = require_tensor(ctx, p + "conv.dw.weight");
    b.conv_dw_b   = maybe_tensor(ctx, p + "conv.dw.bias");
    if (model.encoder_cfg.conv_norm_type == ConvNormType::LayerNorm) {
        b.conv_norm_w = require_tensor(ctx, p + "conv.norm.weight");
        b.conv_norm_b = require_tensor(ctx, p + "conv.norm.bias");
    } else {
        b.conv_bn_scale = require_tensor(ctx, p + "conv.bn.scale");
        b.conv_bn_shift = require_tensor(ctx, p + "conv.bn.shift");
    }
    b.conv_pw2_w  = require_tensor(ctx, p + "conv.pw2.weight");
    b.conv_pw2_b  = maybe_tensor(ctx, p + "conv.pw2.bias");

    b.norm_ff2_w  = require_tensor(ctx, p + "norm_ff2.weight");
    b.norm_ff2_b  = require_tensor(ctx, p + "norm_ff2.bias");
    b.ff2_l1_w    = require_tensor(ctx, p + "ff2.linear1.weight");
    b.ff2_l1_b    = maybe_tensor(ctx, p + "ff2.linear1.bias");
    b.ff2_l2_w    = require_tensor(ctx, p + "ff2.linear2.weight");
    b.ff2_l2_b    = maybe_tensor(ctx, p + "ff2.linear2.bias");

    b.norm_out_w  = require_tensor(ctx, p + "norm_out.weight");
    b.norm_out_b  = require_tensor(ctx, p + "norm_out.bias");
}

// Binds the ggml encoder's weights: the subsampling stack and the conformer
// blocks. Every tensor is required, so a GGUF without encoder weights fails
// here unless Core ML owns the encoder and the caller skipped this.
static void load_encoder_weights(ParakeetCtcModel::Impl * impl, ParakeetCtcModel & out_model) {
    ggml_context * ctx = impl->ctx;
    out_model.subsampling.conv0_w    = require_tensor(ctx, "encoder.subsampling.conv0.weight");
    out_model.subsampling.conv0_b    = maybe_tensor(ctx, "encoder.subsampling.conv0.bias");
    out_model.subsampling.conv1_dw_w = require_tensor(ctx, "encoder.subsampling.conv1_dw.weight");
    out_model.subsampling.conv1_dw_b = maybe_tensor(ctx, "encoder.subsampling.conv1_dw.bias");
    out_model.subsampling.conv1_pw_w = require_tensor(ctx, "encoder.subsampling.conv1_pw.weight");
    out_model.subsampling.conv1_pw_b = maybe_tensor(ctx, "encoder.subsampling.conv1_pw.bias");
    out_model.subsampling.conv2_dw_w = require_tensor(ctx, "encoder.subsampling.conv2_dw.weight");
    out_model.subsampling.conv2_dw_b = maybe_tensor(ctx, "encoder.subsampling.conv2_dw.bias");
    out_model.subsampling.conv2_pw_w = require_tensor(ctx, "encoder.subsampling.conv2_pw.weight");
    out_model.subsampling.conv2_pw_b = maybe_tensor(ctx, "encoder.subsampling.conv2_pw.bias");
    out_model.subsampling.out_w      = require_tensor(ctx, "encoder.subsampling.out.weight");
    out_model.subsampling.out_b      = maybe_tensor(ctx, "encoder.subsampling.out.bias");

    // Use converter-pre-stacked encoder.blk.*.attn.qkv on GPU when the wide
    // M=3*n_embd matmul helps; keep unstacked Q/K/V on CPU for cache locality.
    // (Heuristic: stacked wins where separate matmuls under-fill the device;
    // Metal stays unstacked here.)
    const bool gate_qkv_stack =
        impl->backend_active &&
        !backend_is_cpu(impl->backend_active) &&
        !backend_is_metal(impl->backend_active);

    out_model.blocks.resize(out_model.encoder_cfg.n_layers);
    for (int i = 0; i < out_model.encoder_cfg.n_layers; ++i) {
        load_conformer_block_weights(ctx, out_model, gate_qkv_stack, i, out_model.blocks[i]);
    }
}

static void load_transcribepsy_eu_weights(const gguf_context * g,
                                         ParakeetCtcModel::Impl * impl,
                                         ParakeetCtcModel & out_model,
                                         bool measure) {
    const int id = require_key(g, "parakeet.eu.adapter_layers");
    if (gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(g, id) != GGUF_TYPE_INT32)
        throw std::runtime_error("gguf: invalid TranscribePsy EU adapter layer type");
    const size_t count = gguf_get_arr_n(g, id);
    if (count == 0 || count > (size_t) out_model.encoder_cfg.n_layers)
        throw std::runtime_error("gguf: invalid TranscribePsy EU adapter count");
    const int n = (int) count;
    const int32_t * layers = (const int32_t *) gguf_get_arr_data(g, id);
    const int dim = out_model.encoder_cfg.d_model;
    const int rank = (int) get_u32(g, "parakeet.eu.adapter_rank", 0);
    auto shape = [](ggml_tensor * t, int64_t n0, int64_t n1 = 1) {
        return t && t->ne[0] == n0 && t->ne[1] == n1;
    };
    if (n <= 0 || dim <= 0 || rank <= 0 || !layers)
        throw std::runtime_error("gguf: invalid TranscribePsy EU adapter metadata");
    for (int i = 0; i < n; ++i) {
        TranscribePsyEuAdapter a;
        a.layer = layers[i];
        if (a.layer < 0 || a.layer >= out_model.encoder_cfg.n_layers ||
            (i > 0 && a.layer <= layers[i - 1]))
            throw std::runtime_error("gguf: invalid TranscribePsy EU adapter layer");
        const std::string p = "eu.adapter." + std::to_string(i) + ".";
        a.down_w = require_tensor(impl->ctx, p + "down.weight");
        a.down_b = require_tensor(impl->ctx, p + "down.bias");
        a.up_w = require_tensor(impl->ctx, p + "up.weight");
        a.up_b = require_tensor(impl->ctx, p + "up.bias");
        if (!shape(a.down_w, dim, rank) || !shape(a.down_b, rank) ||
            !shape(a.up_w, rank, dim) || !shape(a.up_b, dim))
            throw std::runtime_error("gguf: invalid TranscribePsy EU adapter tensor shape");
        out_model.eu_adapters.push_back(a);
    }
    auto & gate = out_model.eu_gate;
    gate.layer = (int) get_u32(g, "parakeet.eu.gate_layer", out_model.encoder_cfg.n_layers);
    gate.attn_dim = (int) get_u32(g, "parakeet.eu.gate_attn_dim", 0);
    gate.hidden = (int) get_u32(g, "parakeet.eu.gate_hidden", 0);
    gate.threshold = get_f32(g, "parakeet.eu.gate_threshold", -1.0f);
    if (gate.layer < 0 || gate.layer >= out_model.encoder_cfg.n_layers ||
        gate.attn_dim <= 0 || gate.hidden <= 0 ||
        !(gate.threshold > 0.0f && gate.threshold < 1.0f))
        throw std::runtime_error("gguf: invalid TranscribePsy EU gate metadata");
    auto read = [&](const std::string & name, int64_t n0, int64_t n1 = 1) {
        ggml_tensor * t = require_tensor(impl->ctx, "eu.gate." + name);
        if (!shape(t, n0, n1) || t->type != GGML_TYPE_F32)
            throw std::runtime_error("gguf: invalid TranscribePsy EU gate tensor " + name);
        return measure ? std::vector<float>{} : read_filterbank_to_vector(t);
    };
    gate.proj0_w = read("pool.proj.0.weight", dim, gate.attn_dim);
    gate.proj0_b = read("pool.proj.0.bias", gate.attn_dim);
    gate.proj2_w = read("pool.proj.2.weight", gate.attn_dim, dim);
    gate.proj2_b = read("pool.proj.2.bias", dim);
    gate.norm_w = read("norm.weight", 2 * dim);
    gate.norm_b = read("norm.bias", 2 * dim);
    gate.mlp0_w = read("mlp.0.weight", 2 * dim, gate.hidden);
    gate.mlp0_b = read("mlp.0.bias", gate.hidden);
    gate.mlp3_w = read("mlp.3.weight", gate.hidden, 1);
    gate.mlp3_b = read("mlp.3.bias", 1);
}

// Shared body of load_from_gguf and load_from_gguf_metadata_only. When
// `measure` is non-null the load is metadata-only: every allocation the real
// path makes is sized into `measure` instead of performed and no tensor data
// is read; see the declaration comments in parakeet_ctc.h.
static int load_from_gguf_impl(const std::string & gguf_path,
                               ParakeetCtcModel  & out_model,
                               int                 n_threads,
                               int                 n_gpu_layers,
                               bool                verbose,
                               GgufLoadMeasure   * measure,
                               const std::string & backend = "auto",
                               const CoremlOptions & coreml = {},
                               bool memory_degraded = false) {
    auto impl = std::make_shared<ParakeetCtcModel::Impl>();
#ifdef PARAKEET_USE_COREML
    // Resolved once; a metadata-only load never touches Core ML.
    if (!measure) impl->coreml = std::make_unique<CoremlSidecars>(coreml, nullptr, memory_degraded);
#else
    (void) coreml;
    (void) memory_degraded;
#endif

    impl->backend_cpu = init_cpu_backend();
    if (!impl->backend_cpu) {
        PARAKEET_LOG_ERROR("gguf: failed to initialize CPU backend (no CPU device registered?)\n");
        return 10;
    }
    int resolved_threads = n_threads;
    if (resolved_threads <= 0) {
        const unsigned hc = std::thread::hardware_concurrency();
        resolved_threads = hc > 0 ? (int) hc : 4;
    }
    backend_set_n_threads(impl->backend_cpu, resolved_threads);

    impl->backend_blas = init_blas_backend();
    if (impl->backend_blas) {
        backend_set_n_threads(impl->backend_blas, resolved_threads);
    }

    bool skipped_unsupported_gpu = false;
    bool gpu_is_mali_vulkan = false;
    if (backend_selection_is_auto(backend)) {
        impl->backend_gpu = init_gpu_backend(n_gpu_layers, verbose, skipped_unsupported_gpu,
                                             gpu_is_mali_vulkan);
    } else {
        ggml_backend_t selected = init_explicit_backend(backend, impl->backend_cpu, gpu_is_mali_vulkan);
        if (!selected) return 11;
        if (selected != impl->backend_cpu) impl->backend_gpu = selected;
    }
    impl->backend_active = impl->backend_gpu ? impl->backend_gpu : impl->backend_cpu;
    impl->gpu_unsupported = skipped_unsupported_gpu && impl->backend_gpu == nullptr;
    // On Mali-Vulkan the Sortformer head miscomputes (transformer block 0 -> NaN);
    // route only that head to CPU while the encoder + CTC/TDT/EOU heads stay on GPU.
    impl->sortformer_force_cpu =
        gpu_is_mali_vulkan && impl->backend_active == impl->backend_gpu;
    if (impl->backend_gpu && impl->backend_active == impl->backend_gpu && !gpu_is_mali_vulkan) {
        impl->dw_direct_planar   = backend_runs_conv_2d_dw(impl->backend_gpu, false);
        impl->dw_direct_channels = backend_runs_conv_2d_dw(impl->backend_gpu, true);
        // Force the subsampler im2col+mul+sum_rows lowering on OpenCL for the
        // same reason Mali-Vulkan takes it: the ggml_conv_2d_dw_direct path
        // miscomputes on ggml-opencl. The conformer conv2d_dw path is
        // unaffected and keeps its probe result.
        if (backend_is_opencl(impl->backend_gpu)) {
            impl->dw_direct_planar = false;
        }
        if (verbose) {
            PARAKEET_LOG_INFO("parakeet: depthwise conv lowering: subsampler %s, conformer %s\n",
                              impl->dw_direct_planar   ? "direct" : "im2col",
                              impl->dw_direct_channels ? "direct" : "im2col");
        }
    } else if (impl->backend_active == impl->backend_cpu) {
        // ggml-cpu runs CONV_2D_DW directly; the im2col+mul+sum_rows lowering exists for Mali/OpenCL.
        impl->dw_direct_planar = backend_runs_conv_2d_dw(impl->backend_cpu, false);
        if (verbose) {
            PARAKEET_LOG_INFO("parakeet: depthwise conv lowering: subsampler %s (cpu)\n",
                              impl->dw_direct_planar ? "direct" : "im2col");
        }
    }
    if (impl->sortformer_force_cpu && verbose) {
        PARAKEET_LOG_INFO(
            "parakeet: Sortformer diarization head -> CPU on Mali-Vulkan "
            "(encoder + CTC/TDT/EOU stay on the GPU)\n");
    }

    // Compute scheduler over the active backend + CPU (CPU MUST be last; ggml
    // asserts this). When the active backend is the GPU, ops it cannot run fall
    // back to CPU per-op; when CPU-only, the scheduler is a single-backend
    // pass-through. op_offload=false: all Parakeet weights live on the active
    // backend, so the CPU-weight->GPU offload heuristic never applies here.
    // graph_size mirrors the encoder cgraph capacity (build_encoder_graph_cached);
    // actual node counts are far smaller (verify via GGML_SCHED_DEBUG).
    {
        ggml_backend_t sched_backends[2];
        int n_sched = 0;
        if (impl->backend_gpu && impl->backend_active == impl->backend_gpu) {
            sched_backends[n_sched++] = impl->backend_gpu;
        }
        sched_backends[n_sched++] = impl->backend_cpu;   // CPU last (mandatory)
        impl->sched = ggml_backend_sched_new(
            sched_backends, /*bufts=*/nullptr, n_sched,
            /*graph_size=*/GGML_DEFAULT_GRAPH_SIZE * 16,
            /*parallel=*/false, /*op_offload=*/false);
        if (!impl->sched) {
            PARAKEET_LOG_ERROR("gguf: ggml_backend_sched_new failed\n");
            return 16;
        }
    }

    gguf_init_params params = { /*no_alloc=*/ true, &impl->ctx };
#ifdef PARAKEET_USE_COREML
    // Open the GGUF once: its identity comes from this descriptor, and the
    // metadata and every tensor are read through it, so a generated sidecar
    // can only be prepared from the file whose weights the model holds,
    // whatever the path points at before or after.
    std::unique_ptr<FILE, int (*)(FILE *)> gguf_file(nullptr, &std::fclose);
    if (!measure) {
        if (GgufLoadObserver observer = g_gguf_load_observer.load()) observer("open");
        gguf_file.reset(std::fopen(gguf_path.c_str(), "rb"));
        if (!gguf_file) {
            PARAKEET_LOG_ERROR("gguf: failed to open %s\n", gguf_path.c_str());
            return 1;
        }
        impl->source_ok = pcm::generation::source_stat_fd(fileno(gguf_file.get()), gguf_path, impl->source);
        if (GgufLoadObserver observer = g_gguf_load_observer.load()) observer("opened");
    }
    impl->gguf = gguf_file ? gguf_init_from_file_ptr(gguf_file.get(), params)
                           : gguf_init_from_file(gguf_path.c_str(), params);
#else
    impl->gguf = gguf_init_from_file(gguf_path.c_str(), params);
#endif
    if (!impl->gguf) {
        PARAKEET_LOG_ERROR("gguf: failed to open %s\n", gguf_path.c_str());
        return 1;
    }

    gguf_context * g = impl->gguf;
    impl->affine_weights = load_packed_affine(g, impl->ctx);

    // A Parakeet GGUF without encoder weights (the Core ML distributions) runs
    // only with its encoder sidecar, which then has to load before allocation
    // so the `encoder.*` tensors get no memory. PARAKEET_COREML_ONLY requests the
    // same for a full GGUF when its sidecar preserves the required routing.
    // The memory-fit projection never loads Core ML and cannot project an
    // encoder-less GGUF.
    const bool gguf_has_encoder = gguf_has_encoder_weights(g);
#ifdef PARAKEET_USE_COREML
    if (!measure && coreml_must_own_encoder(gguf_has_encoder)) {
        impl->ctx_coreml = init_coreml_only_encoder(g, gguf_path, impl->coreml->resolved(), verbose,
                                                    impl.get());
        impl->coreml_owns_encoder = coreml_can_own_encoder(g, impl->ctx_coreml);
        if (impl->ctx_coreml != nullptr && !impl->coreml_owns_encoder) {
            // A sidecar that may not own the encoder stays an ordinary one:
            // the encoder slot loads it in the background like any full
            // GGUF's, so it follows the resolved wait and use policy. (A
            // bundle whose metadata already rules ownership out is not
            // loaded here at all.)
            parakeet_coreml_free(impl->ctx_coreml);
            impl->ctx_coreml = nullptr;
        }
    }
#endif
    if (!gguf_has_encoder && !impl->coreml_owns_encoder) {
        PARAKEET_LOG_ERROR("gguf: %s holds no encoder weights; it requires a compatible "
                           "encoder-owning %s sidecar on an Apple build with PARAKEET_COREML=ON\n",
                           gguf_path.c_str(), "-encoder.mlmodelc");
        return kGgufLoadNoEncoderWeights;
    }
    const bool owns_encoder = impl->coreml_owns_encoder;

    // CPU-only runs: place repack-eligible encoder GEMM weights in the CPU
    // extra buffer type first; the default allocation below picks up every
    // tensor this pass leaves unallocated. Skipped when the model lives on a
    // GPU backend, when Core ML owns the encoder, or for packed-affine carriers.
    if (impl->backend_active == impl->backend_cpu && !owns_encoder && impl->affine_weights.empty()) {
        if (measure) {
            alloc_cpu_repack_weights(impl->ctx, impl->backend_cpu, verbose,
                                     &measure->repack_bytes);
        } else {
            impl->weights_extra_buffers =
                alloc_cpu_repack_weights(impl->ctx, impl->backend_cpu, verbose);
        }
    }

    // On the default Hexagon buffer, CPU kernels would read tiled bytes as
    // GGUF blocks and corrupt the Sortformer head's very first projection.
    ggml_backend_buffer_type_t weights_buft = weight_buffer_type(impl->backend_active);

    if (measure) {
        measure->weights_bytes = ggml_backend_alloc_ctx_tensors_from_buft_size(
            impl->ctx, weights_buft);
    } else {
        impl->weights_buffer = owns_encoder
            ? alloc_ctx_tensors_without_encoder(impl->ctx, weights_buft)
            : ggml_backend_alloc_ctx_tensors_from_buft(impl->ctx, weights_buft);
        if (!impl->weights_buffer) {
            PARAKEET_LOG_ERROR("gguf: ggml_backend_alloc_ctx_tensors failed\n");
            return 12;
        }
        ggml_backend_buffer_set_usage(impl->weights_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

#ifdef PARAKEET_USE_COREML
        // The tensor data comes through the descriptor the metadata did.
        FILE * data_file = gguf_file.get();
#else
        const void * data_file = nullptr;
#endif
        std::ifstream f;
        if (data_file == nullptr) {
            f.open(gguf_path, std::ios::binary);
            if (!f) {
                PARAKEET_LOG_ERROR("gguf: cannot reopen %s for tensor data\n", gguf_path.c_str());
                return 13;
            }
        }
        const size_t data_offset = gguf_get_data_offset(g);
        const int64_t n_tensors = gguf_get_n_tensors(g);
        std::vector<char> buf;
        for (int64_t i = 0; i < n_tensors; ++i) {
            const char  * name = gguf_get_tensor_name(g, i);
            ggml_tensor * t    = ggml_get_tensor(impl->ctx, name);
            if (!t) continue;
            // No buffer means the allocation pass skipped it: Core ML owns the
            // encoder, so these bytes are never read.
            if (!t->buffer) continue;
            const size_t off   = gguf_get_tensor_offset(g, i);
            const size_t nbytes = ggml_nbytes(t);
            buf.resize(nbytes);
            bool read_ok = false;
#ifdef PARAKEET_USE_COREML
            if (data_file != nullptr) {
                read_ok = fseeko(data_file, (off_t) (data_offset + off), SEEK_SET) == 0 &&
                          std::fread(buf.data(), 1, nbytes, data_file) == nbytes;
            } else
#endif
            {
                f.seekg((std::streamoff)(data_offset + off), std::ios::beg);
                read_ok = static_cast<bool>(f.read(buf.data(), nbytes));
            }
            if (!read_ok) {
                PARAKEET_LOG_ERROR("gguf: short read on tensor '%s' (%zu bytes)\n", name, nbytes);
                return 14;
            }
            ggml_backend_tensor_set(t, buf.data(), 0, nbytes);
        }
#ifdef PARAKEET_USE_COREML
        // The descriptor again after the last tensor read: a file written in
        // place during the upload (same inode, a new size or mtime) may have
        // given some tensors the new bytes, so the identity taken at the open
        // is not the loaded weights' and must never join a job for it.
        if (gguf_file && impl->source_ok) {
            pcm::SourceStat after;
            const bool same = pcm::generation::source_stat_fd(fileno(gguf_file.get()), gguf_path, after) &&
                              after.size == impl->source.size && after.mtime_ns == impl->source.mtime_ns &&
                              after.ino == impl->source.ino && after.dev == impl->source.dev;
            if (!same) impl->source.changed = true;
        }
        if (GgufLoadObserver observer = g_gguf_load_observer.load()) observer("read");
#endif
    }

    {
        const int id = find_key(g, "general.architecture");
        if (id < 0) {
            PARAKEET_LOG_ERROR("gguf: missing general.architecture\n");
            return 2;
        }
        const char * arch = gguf_get_val_str(g, id);
        if (std::strcmp(arch, "sortformer") == 0) {
            load_diarization_metadata(g, out_model);
            load_diarization_weights(impl->ctx, out_model);
            out_model.mel_filterbank = require_diarization_tensor(
                impl->ctx, "preprocessor.fb", out_model.mel_cfg.n_fft / 2 + 1,
                out_model.mel_cfg.n_mels);
            if (measure) {
                exclude_weights_from_graph_measure(impl->ctx);
            } else {
                out_model.mel_cfg.filterbank = read_filterbank_to_vector(out_model.mel_filterbank);
                make_diarization_window(out_model.mel_cfg);
            }
            if (impl->backend_blas) {
                ggml_backend_free(impl->backend_blas);
                impl->backend_blas = nullptr;
            }
            out_model.impl = impl;
#ifdef PARAKEET_USE_COREML
            if (!measure) {
                maybe_init_coreml_diarizers(gguf_path, out_model, verbose);
                impl->coreml->start();
            }
#endif
            return 0;
        }
        const bool affine_arch = std::strcmp(arch, "parakeet-affine-v1") == 0;
        if (affine_arch != !impl->affine_weights.empty()) {
            PARAKEET_LOG_ERROR("gguf: packed-affine architecture/schema mismatch\n");
            return 2;
        }
        if (std::strcmp(arch, kParakeetArchitecture) != 0 && !affine_arch) {
            PARAKEET_LOG_ERROR("gguf: expected arch=%s, got '%s'\n", kParakeetArchitecture, arch);
            return 2;
        }
    }

    out_model.encoder_cfg.d_model        = get_u32(g, "parakeet.encoder.d_model", 1024);
    out_model.encoder_cfg.n_layers       = get_u32(g, "parakeet.encoder.n_layers", 24);
    out_model.encoder_cfg.n_heads        = get_u32(g, "parakeet.encoder.n_heads", 8);
    out_model.encoder_cfg.head_dim       = get_u32(g, "parakeet.encoder.head_dim",
                                                   out_model.encoder_cfg.d_model / out_model.encoder_cfg.n_heads);
    out_model.encoder_cfg.ff_dim         = get_u32(g, "parakeet.encoder.ff_dim", 4096);
    out_model.encoder_cfg.conv_kernel    = get_u32(g, "parakeet.encoder.conv_kernel", 9);
    out_model.encoder_cfg.subsampling_factor    = get_u32(g, "parakeet.encoder.subsampling_factor", 8);
    out_model.encoder_cfg.subsampling_channels  = get_u32(g, "parakeet.encoder.subsampling_conv_channels", 256);
    out_model.encoder_cfg.subsampling_freq_bins = get_u32(g, "parakeet.encoder.subsampling_freq_bins", 10);
    out_model.encoder_cfg.pos_emb_max_len = get_u32(g, "parakeet.encoder.pos_emb_max_len", 5000);
    out_model.encoder_cfg.xscaling        = get_bool(g, "parakeet.encoder.xscaling", true);
    out_model.encoder_cfg.untie_biases    = get_bool(g, "parakeet.encoder.untie_biases", true);
    out_model.encoder_cfg.use_bias        = get_bool(g, "parakeet.encoder.use_bias", true);

    {
        const std::string conv_norm = get_str(g, "parakeet.encoder.conv_norm_type", "batch_norm");
        out_model.encoder_cfg.conv_norm_type = (conv_norm == "layer_norm")
                                                 ? ConvNormType::LayerNorm
                                                 : ConvNormType::BatchNorm;
    }
    out_model.encoder_cfg.causal_downsampling =
        get_bool(g, "parakeet.encoder.causal_downsampling", false);
    {
        const std::string conv_ctx = get_str(g, "parakeet.encoder.conv_context_size", "default");
        out_model.encoder_cfg.conv_causal = (conv_ctx == "causal");
        const std::string conv_style =
            get_str(g, "parakeet.encoder.conv_context_style", "regular");
        out_model.encoder_cfg.conv_dynamic_chunking = (conv_style == "dcc");
    }
    {
        const int id_l = find_key(g, "parakeet.encoder.att_context_size_left");
        const int id_r = find_key(g, "parakeet.encoder.att_context_size_right");
        if (id_l >= 0) out_model.encoder_cfg.att_context_left  = gguf_get_val_i32(g, id_l);
        if (id_r >= 0) out_model.encoder_cfg.att_context_right = gguf_get_val_i32(g, id_r);
    }
    {
        const std::string style = get_str(g, "parakeet.encoder.att_context_style", "regular");
        out_model.encoder_cfg.att_chunked_limited = (style == "chunked_limited");
        out_model.encoder_cfg.att_dynamic_chunking =
            (style == "chunked_limited_with_rc");
    }
    // The graph-shape probes take the model's own dimensions, so they wait for
    // its encoder configuration.
    impl->attn = select_attn_path(impl->backend_active, out_model.encoder_cfg.head_dim,
                                  out_model.encoder_cfg.n_heads);
    impl->glu_fused = !gpu_is_mali_vulkan && backend_runs_siglu(impl->backend_active, out_model.encoder_cfg.d_model);

    out_model.supports_streaming = get_bool(g, "parakeet.encoder.streaming.enabled", false);

    const std::string mtype_str = get_str(g, "parakeet.model.type", "ctc");
    if (mtype_str == "ctc") {
        out_model.model_type = ParakeetModelType::CTC;
    } else if (mtype_str == "rnnt") {
        out_model.model_type = ParakeetModelType::RNNT;
    } else if (mtype_str == "tdt") {
        out_model.model_type = ParakeetModelType::TDT;
    } else if (mtype_str == "eou") {
        out_model.model_type = ParakeetModelType::EOU;
    } else if (mtype_str == "nemotron") {
        out_model.model_type = ParakeetModelType::NEMOTRON;
    } else if (mtype_str == "sortformer") {
        out_model.model_type = ParakeetModelType::SORTFORMER;
    } else {
        throw std::runtime_error(
            "gguf: unsupported parakeet.model.type '" + mtype_str + "'");
    }

    // Optional variant tag (empty for legacy GGUFs that predate the key).
    out_model.model_variant = get_str(g, "parakeet.model_variant", "");
    if (out_model.model_variant == "transcribepsy-250m-eu" &&
        out_model.model_type != ParakeetModelType::TDT) {
        throw std::runtime_error("gguf: TranscribePsy EU requires a TDT head");
    }
    long_form_hint_for_model(get_u32(g, "parakeet.long_form.window_frames", 0),
                             get_u32(g, "parakeet.long_form.context_frames", 0),
                             get_str(g, "general.description", ""),
                             out_model.encoder_cfg.long_form_window_hint,
                             out_model.encoder_cfg.long_form_context_hint,
                             out_model.model_variant);
    if (out_model.model_type == ParakeetModelType::NEMOTRON) {
        load_nemotron_metadata(g, out_model);
    }
    load_unified_streaming_metadata(g, out_model);

    if (out_model.model_type == ParakeetModelType::RNNT) {
        out_model.encoder_cfg.rnnt_pred_hidden =
            get_u32(g, "parakeet.rnnt.pred_hidden", 640);
        out_model.encoder_cfg.rnnt_pred_rnn_layers =
            get_u32(g, "parakeet.rnnt.pred_rnn_layers", 2);
        out_model.encoder_cfg.rnnt_joint_hidden =
            get_u32(g, "parakeet.rnnt.joint_hidden", 640);
        out_model.encoder_cfg.rnnt_max_symbols_per_step =
            get_u32(g, "parakeet.rnnt.max_symbols_per_step", 10);
        out_model.vocab_size = get_u32(g, "parakeet.rnnt.vocab_size", 1024);
        out_model.blank_id =
            get_u32(g, "parakeet.rnnt.blank_id", out_model.vocab_size);
    }

    if (out_model.model_type == ParakeetModelType::TDT) {
        out_model.encoder_cfg.tdt_pred_hidden     = get_u32(g, "parakeet.tdt.pred_hidden",     640);
        out_model.encoder_cfg.tdt_pred_rnn_layers = get_u32(g, "parakeet.tdt.pred_rnn_layers", 2);
        out_model.encoder_cfg.tdt_joint_hidden    = get_u32(g, "parakeet.tdt.joint_hidden",    640);
        out_model.encoder_cfg.tdt_num_durations   = get_u32(g, "parakeet.tdt.num_durations",   5);
        out_model.vocab_size = get_u32(g, "parakeet.tdt.vocab_size", 8192);
        out_model.blank_id   = get_u32(g, "parakeet.tdt.blank_id",   out_model.vocab_size);

        const int did = find_key(g, "parakeet.tdt.durations");
        if (did >= 0) {
            const size_t n = gguf_get_arr_n(g, did);
            const int32_t * data = static_cast<const int32_t *>(gguf_get_arr_data(g, did));
            out_model.tdt_durations.assign(data, data + n);
        } else {
            out_model.tdt_durations = {0, 1, 2, 3, 4};
        }
    }

    if (out_model.model_type == ParakeetModelType::EOU) {
        out_model.encoder_cfg.eou_pred_hidden           = get_u32(g, "parakeet.eou.pred_hidden",           640);
        out_model.encoder_cfg.eou_pred_rnn_layers       = get_u32(g, "parakeet.eou.pred_rnn_layers",       1);
        out_model.encoder_cfg.eou_joint_hidden          = get_u32(g, "parakeet.eou.joint_hidden",          640);
        out_model.encoder_cfg.eou_chunk_mel_frames      = get_u32(g, "parakeet.eou.encoder_chunk_mel_frames", 25);
        out_model.encoder_cfg.eou_cache_lookback_frames = get_u32(g, "parakeet.eou.cache_lookback_frames", 70);
        out_model.encoder_cfg.eou_cache_time_steps      = get_u32(g, "parakeet.eou.cache_time_steps",      8);
        out_model.encoder_cfg.eou_max_symbols_per_step  = get_u32(g, "parakeet.eou.max_symbols_per_step",  5);

        out_model.vocab_size = get_u32(g, "parakeet.eou.vocab_size", 1026);
        out_model.blank_id   = get_u32(g, "parakeet.eou.blank_id",   out_model.vocab_size);

        const int id_eou = find_key(g, "parakeet.eou.eou_id");
        const int id_eob = find_key(g, "parakeet.eou.eob_id");
        out_model.eou_id = id_eou >= 0 ? gguf_get_val_i32(g, id_eou) : -1;
        out_model.eob_id = id_eob >= 0 ? gguf_get_val_i32(g, id_eob) : -1;
    }

    if (out_model.model_type == ParakeetModelType::SORTFORMER) {
        out_model.encoder_cfg.sortformer_num_spks      = get_u32 (g, "parakeet.sortformer.num_spks",      4);
        out_model.encoder_cfg.sortformer_fc_d_model    = get_u32 (g, "parakeet.sortformer.fc_d_model",    512);
        out_model.encoder_cfg.sortformer_tf_d_model    = get_u32 (g, "parakeet.sortformer.tf_d_model",    192);
        out_model.encoder_cfg.sortformer_tf_n_layers   = get_u32 (g, "parakeet.sortformer.tf_n_layers",   18);
        out_model.encoder_cfg.sortformer_tf_n_heads    = get_u32 (g, "parakeet.sortformer.tf_n_heads",    8);
        out_model.encoder_cfg.sortformer_tf_inner_size = get_u32 (g, "parakeet.sortformer.tf_inner_size", 768);
        out_model.encoder_cfg.sortformer_tf_pre_ln     = get_bool(g, "parakeet.sortformer.tf_pre_ln",     false);
    }

    out_model.mel_cfg.sample_rate = get_u32(g, "parakeet.preproc.sample_rate", 16000);
    out_model.mel_cfg.n_fft       = get_u32(g, "parakeet.preproc.n_fft",       512);
    out_model.mel_cfg.win_length  = get_u32(g, "parakeet.preproc.win_length",  400);
    out_model.mel_cfg.hop_length  = get_u32(g, "parakeet.preproc.hop_length",  160);
    out_model.mel_cfg.n_mels      = get_u32(g, "parakeet.preproc.n_mels",      80);
    out_model.mel_cfg.preemph     = get_f32(g, "parakeet.preproc.preemph",     0.97f);
    out_model.mel_cfg.log_zero_guard_value =
        get_f32(g, "parakeet.preproc.log_zero_guard_value", kDefaultLogZeroGuard);
    {
        const std::string norm = get_str(g, "parakeet.preproc.normalize", "per_feature");
        out_model.mel_cfg.normalize = (norm == "NA" || norm == "none" || norm == "None")
                                        ? MelNormalize::None
                                        : MelNormalize::PerFeature;
    }

    if (out_model.model_type == ParakeetModelType::CTC) {
        out_model.vocab_size = get_u32(g, "parakeet.ctc.vocab_size", 1025);
        out_model.blank_id   = get_u32(g, "parakeet.ctc.blank_id",   1024);

        const int id_langs = find_key(g, "parakeet.ctc.lang_ids");
        const int id_start = find_key(g, "parakeet.ctc.lang_token_start");
        const int id_end   = find_key(g, "parakeet.ctc.lang_token_end");
        if (id_langs >= 0 && id_start >= 0 && id_end >= 0 &&
            gguf_get_arr_type(g, id_langs) == GGUF_TYPE_STRING &&
            gguf_get_arr_n(g, id_langs) == gguf_get_arr_n(g, id_start) &&
            gguf_get_arr_n(g, id_langs) == gguf_get_arr_n(g, id_end)) {
            const size_t n = gguf_get_arr_n(g, id_langs);
            const int32_t * starts = static_cast<const int32_t *>(gguf_get_arr_data(g, id_start));
            const int32_t * ends   = static_cast<const int32_t *>(gguf_get_arr_data(g, id_end));
            out_model.ctc_lang_ranges.resize(n);
            for (size_t i = 0; i < n; ++i) {
                const char * s = gguf_get_arr_str(g, id_langs, i);
                out_model.ctc_lang_ranges[i].id = s ? s : "";
                out_model.ctc_lang_ranges[i].token_start = starts[i];
                out_model.ctc_lang_ranges[i].token_end   = ends[i];
            }
        }
    }
    out_model.vocab.blank_id = out_model.blank_id;

    {
        const int id = find_key(g, "tokenizer.ggml.tokens");
        if (id >= 0 && gguf_get_arr_type(g, id) == GGUF_TYPE_STRING) {
            const size_t n = gguf_get_arr_n(g, id);
            out_model.vocab.pieces.resize(n);
            for (size_t i = 0; i < n; ++i) {
                const char * s = gguf_get_arr_str(g, id, i);
                if (s) out_model.vocab.pieces[i] = s;
            }
        }
        const int id_sc = find_key(g, "tokenizer.ggml.scores");
        if (id_sc >= 0 && gguf_get_arr_n(g, id_sc) == out_model.vocab.pieces.size()) {
            const float * p = (const float *) gguf_get_arr_data(g, id_sc);
            out_model.vocab.scores.assign(p, p + out_model.vocab.pieces.size());
        }
        const int id_tp = find_key(g, "tokenizer.ggml.token_type");
        if (id_tp >= 0 && gguf_get_arr_n(g, id_tp) == out_model.vocab.pieces.size()) {
            const int8_t * p = (const int8_t *) gguf_get_arr_data(g, id_tp);
            out_model.vocab.piece_types.assign(p, p + out_model.vocab.pieces.size());
        }
        out_model.vocab.unk_id = (int32_t) get_u32(g, "tokenizer.ggml.unk_token_id", (uint32_t) -1);
        out_model.vocab.bos_id = (int32_t) get_u32(g, "tokenizer.ggml.bos_token_id", (uint32_t) -1);
        out_model.vocab.eos_id = (int32_t) get_u32(g, "tokenizer.ggml.eos_token_id", (uint32_t) -1);
        out_model.vocab.pad_id = (int32_t) get_u32(g, "tokenizer.ggml.pad_token_id", (uint32_t) -1);
    }

    out_model.mel_filterbank = require_tensor(impl->ctx, "preproc.mel_filterbank");
    out_model.window         = require_tensor(impl->ctx, "preproc.window");

    if (!measure) {  // reads tensor data; a metadata-only model never runs the mel front-end
        out_model.mel_cfg.filterbank = read_filterbank_to_vector(out_model.mel_filterbank);
        out_model.mel_cfg.window     = read_filterbank_to_vector(out_model.window);
    }

    // The ggml encoder's weights. When Core ML owns the encoder they were never
    // allocated and may be absent from the file altogether, and nothing reads
    // them: build_encoder_graph_cached refuses the ggml encoder in that state.
    if (!owns_encoder) {
        load_encoder_weights(impl.get(), out_model);
    }

    if (out_model.model_variant == "transcribepsy-250m-eu")
        load_transcribepsy_eu_weights(g, impl.get(), out_model, measure != nullptr);

    if (out_model.model_type == ParakeetModelType::CTC) {
        out_model.ctc.w = require_tensor(impl->ctx, "ctc.decoder.weight");
        out_model.ctc.b = require_tensor(impl->ctx, "ctc.decoder.bias");
    } else if (out_model.model_type == ParakeetModelType::EOU) {
        out_model.eou.predict_embed = require_tensor(impl->ctx, "eou.predict.embed.weight");
        for (int l = 0; l < out_model.encoder_cfg.eou_pred_rnn_layers; ++l) {
            const std::string pl = "eou.predict.lstm." + std::to_string(l) + ".";
            TdtLstmLayer lyr;
            lyr.w_ih = require_tensor(impl->ctx, pl + "w_ih");
            lyr.w_hh = require_tensor(impl->ctx, pl + "w_hh");
            lyr.b_ih = require_tensor(impl->ctx, pl + "b_ih");
            lyr.b_hh = require_tensor(impl->ctx, pl + "b_hh");
            out_model.eou.lstm.push_back(lyr);
        }
        out_model.eou.joint_enc_w  = require_tensor(impl->ctx, "eou.joint.enc.weight");
        out_model.eou.joint_enc_b  = require_tensor(impl->ctx, "eou.joint.enc.bias");
        out_model.eou.joint_pred_w = require_tensor(impl->ctx, "eou.joint.pred.weight");
        out_model.eou.joint_pred_b = require_tensor(impl->ctx, "eou.joint.pred.bias");
        out_model.eou.joint_out_w  = require_tensor(impl->ctx, "eou.joint.out.weight");
        out_model.eou.joint_out_b  = require_tensor(impl->ctx, "eou.joint.out.bias");
    } else if (out_model.model_type == ParakeetModelType::SORTFORMER) {
        out_model.sortformer.encoder_proj_w = require_tensor(impl->ctx, "sortformer.encoder_proj.weight");
        out_model.sortformer.encoder_proj_b = require_tensor(impl->ctx, "sortformer.encoder_proj.bias");
        out_model.sortformer.transformer.resize(out_model.encoder_cfg.sortformer_tf_n_layers);
        for (int i = 0; i < out_model.encoder_cfg.sortformer_tf_n_layers; ++i) {
            const std::string p = "sortformer.transformer.blk." + std::to_string(i) + ".";
            SortformerTransformerBlock & b = out_model.sortformer.transformer[i];
            b.attn_q_w  = require_tensor(impl->ctx, p + "attn.q.weight");
            b.attn_q_b  = require_tensor(impl->ctx, p + "attn.q.bias");
            b.attn_k_w  = require_tensor(impl->ctx, p + "attn.k.weight");
            b.attn_k_b  = require_tensor(impl->ctx, p + "attn.k.bias");
            b.attn_v_w  = require_tensor(impl->ctx, p + "attn.v.weight");
            b.attn_v_b  = require_tensor(impl->ctx, p + "attn.v.bias");
            b.attn_o_w  = require_tensor(impl->ctx, p + "attn.out.weight");
            b.attn_o_b  = require_tensor(impl->ctx, p + "attn.out.bias");
            b.ln1_w     = require_tensor(impl->ctx, p + "ln1.weight");
            b.ln1_b     = require_tensor(impl->ctx, p + "ln1.bias");
            b.ffn_in_w  = require_tensor(impl->ctx, p + "ffn.in.weight");
            b.ffn_in_b  = require_tensor(impl->ctx, p + "ffn.in.bias");
            b.ffn_out_w = require_tensor(impl->ctx, p + "ffn.out.weight");
            b.ffn_out_b = require_tensor(impl->ctx, p + "ffn.out.bias");
            b.ln2_w     = require_tensor(impl->ctx, p + "ln2.weight");
            b.ln2_b     = require_tensor(impl->ctx, p + "ln2.bias");
        }
        out_model.sortformer.head_h2h_w = require_tensor(impl->ctx, "sortformer.head.first_hidden_to_hidden.weight");
        out_model.sortformer.head_h2h_b = require_tensor(impl->ctx, "sortformer.head.first_hidden_to_hidden.bias");
        out_model.sortformer.head_h2s_w = require_tensor(impl->ctx, "sortformer.head.single_hidden_to_spks.weight");
        out_model.sortformer.head_h2s_b = require_tensor(impl->ctx, "sortformer.head.single_hidden_to_spks.bias");
    } else if (out_model.model_type == ParakeetModelType::NEMOTRON) {
        load_transducer_weights(
            impl->ctx, "nemotron",
            out_model.nemotron_cfg.pred_rnn_layers,
            out_model.nemotron.rnnt);
        out_model.nemotron.prompt.proj_0_w =
            require_tensor(impl->ctx, "nemotron.prompt.proj.0.weight");
        out_model.nemotron.prompt.proj_0_b =
            require_tensor(impl->ctx, "nemotron.prompt.proj.0.bias");
        out_model.nemotron.prompt.proj_2_w =
            require_tensor(impl->ctx, "nemotron.prompt.proj.2.weight");
        out_model.nemotron.prompt.proj_2_b =
            require_tensor(impl->ctx, "nemotron.prompt.proj.2.bias");
    } else if (out_model.model_type == ParakeetModelType::RNNT) {
        load_transducer_weights(
            impl->ctx, "rnnt",
            out_model.encoder_cfg.rnnt_pred_rnn_layers, out_model.rnnt);
    } else {
        load_transducer_weights(
            impl->ctx, "tdt",
            out_model.encoder_cfg.tdt_pred_rnn_layers, out_model.tdt);
    }

    if (out_model.model_type == ParakeetModelType::NEMOTRON) {
        validate_nemotron_model(out_model);
        out_model.vocab.locale_tag_ids = find_locale_tag_ids(out_model.vocab);
    }

    if (impl->backend_blas) {
        ggml_backend_free(impl->backend_blas);
        impl->backend_blas = nullptr;
    }

    out_model.impl = impl;

#ifdef PARAKEET_USE_COREML
    if (!measure) {
        maybe_init_coreml_encoder(gguf_path, out_model, verbose);
        maybe_init_coreml_decoder(gguf_path, out_model, verbose);
        // The construction group: the minimal set, requested once the ggml
        // upload completed; with wait = construct this blocks.
        impl->coreml->start();
    }
#endif

    if (out_model.model_type == ParakeetModelType::SORTFORMER &&
        impl->sortformer_force_cpu) {
        if (measure) {
            SortformerWeights scratch;  // measure mode leaves handles into a freed ctx; discard
            if (!build_sortformer_cpu_weights(impl.get(), out_model.sortformer,
                                              scratch,
                                              &measure->sortformer_cpu_bytes)) {
                PARAKEET_LOG_ERROR(
                    "parakeet: failed to size CPU-resident Sortformer head weights\n");
                return 15;
            }
        } else if (!build_sortformer_cpu_weights(impl.get(), out_model.sortformer,
                                                 out_model.sortformer_cpu)) {
            PARAKEET_LOG_ERROR(
                "parakeet: failed to build CPU-resident Sortformer head weights\n");
            return 15;
        }
        if (!measure && verbose) PARAKEET_LOG_INFO(
            "parakeet: built CPU-resident Sortformer head weights "
            "(diarization head runs on CPU; encoder stays on the Mali GPU)\n");
    }

    if (decoder_moves_to_cpu(out_model, impl.get())) {
        const TdtWeights & src = *transducer_decoder_weights(out_model);
        if (measure) {
            TdtWeights scratch;
            if (!build_tdt_cpu_weights(impl.get(), src, scratch, &measure->tdt_cpu_bytes)) {
                PARAKEET_LOG_ERROR("parakeet: failed to size CPU-resident decoder weights\n");
                return 15;
            }
        } else if (!build_tdt_cpu_weights(impl.get(), src, out_model.tdt_cpu)) {
            PARAKEET_LOG_ERROR("parakeet: failed to build CPU-resident decoder weights\n");
            return 15;
        }
        if (!measure && verbose) PARAKEET_LOG_INFO(
            "parakeet: built CPU-resident decoder weights "
            "(TDT decode graphs run on the CPU; encoder stays on OpenCL)\n");
    }

    if (measure) {
        exclude_weights_from_graph_measure(impl->ctx);
    }

    if (verbose) {
        print_model_summary(out_model);
        const char * be = impl->backend_gpu
                            ? ggml_backend_name(impl->backend_gpu)
                            : "CPU";
        PARAKEET_LOG_INFO("  backend: %s  (threads=%d)\n", be, resolved_threads);
    }
    return 0;
}

int load_from_gguf(const std::string & gguf_path,
                   ParakeetCtcModel  & out_model,
                   int                 n_threads,
                   int                 n_gpu_layers,
                   bool                verbose) {
    return load_from_gguf_impl(gguf_path, out_model, n_threads, n_gpu_layers,
                               verbose, /*measure=*/nullptr);
}

int load_from_gguf(const std::string & gguf_path, ParakeetCtcModel & out_model,
                   int n_threads, int n_gpu_layers, bool verbose, const std::string & backend) {
    return load_from_gguf_impl(gguf_path, out_model, n_threads, n_gpu_layers,
                               verbose, /*measure=*/nullptr, backend);
}

int load_from_gguf(const std::string & gguf_path, ParakeetCtcModel & out_model,
                   int n_threads, int n_gpu_layers, bool verbose, const std::string & backend,
                   const CoremlOptions & coreml, bool memory_degraded) {
    return load_from_gguf_impl(gguf_path, out_model, n_threads, n_gpu_layers,
                               verbose, /*measure=*/nullptr, backend, coreml, memory_degraded);
}

int load_from_gguf_metadata_only(const std::string & gguf_path,
                                 ParakeetCtcModel  & out_model,
                                 int                 n_threads,
                                 int                 n_gpu_layers,
                                 bool                verbose,
                                 GgufLoadMeasure   & out_measure,
                                 const std::string & backend) {
    out_measure = GgufLoadMeasure{};
    return load_from_gguf_impl(gguf_path, out_model, n_threads, n_gpu_layers,
                               verbose, &out_measure, backend);
}

size_t model_weights_buffer_bytes(const ParakeetCtcModel & m) {
    if (!m.impl) return 0;
    size_t bytes = 0;
    if (m.impl->weights_buffer) {
        bytes += ggml_backend_buffer_get_size(m.impl->weights_buffer);
    }
    for (ggml_backend_buffer_t b : m.impl->weights_extra_buffers) {
        if (b) bytes += ggml_backend_buffer_get_size(b);
    }
    return bytes;
}

size_t model_encoder_compute_buffer_bytes(const ParakeetCtcModel & m) {
    if (!m.impl || m.impl->encoder_graphs.empty()) return 0;
    const EncoderGraph & g = *m.impl->encoder_graphs.back();
    const size_t compute = g.allocator.bytes();
    return compute + (g.pos_buf ? ggml_backend_buffer_get_size(g.pos_buf) : 0);
}

bool model_has_gpu_backend(const ParakeetCtcModel & m) {
    return m.impl && m.impl->backend_gpu != nullptr;
}

bool model_gpu_unsupported(const ParakeetCtcModel & m) {
    return m.impl && m.impl->gpu_unsupported;
}

std::string model_active_backend_name(const ParakeetCtcModel & m) {
    if (!m.impl) return "CPU";
    ggml_backend_t b = m.impl->backend_active;
    if (!b) return "CPU";
    const char * name = ggml_backend_name(b);
    return name ? std::string(name) : std::string("CPU");
}

std::string model_active_device_name(const ParakeetCtcModel & m) {
    if (!m.impl) return "CPU";
    ggml_backend_t b = m.impl->backend_active;
    if (!b) return "CPU";
    ggml_backend_dev_t dev = ggml_backend_get_device(b);
    const char * name = dev ? ggml_backend_dev_name(dev) : nullptr;
    return name ? std::string(name) : std::string("CPU");
}

#ifdef PARAKEET_USE_COREML
CoremlDiarizers * model_coreml_diarizers(const ParakeetCtcModel & m) {
    return m.impl ? m.impl->coreml_diarizers.get() : nullptr;
}

namespace {

parakeet_coreml_context * encoder_context_of(void * model) {
    return static_cast<parakeet_coreml_context *>(model);
}

EncoderRouteGeometry describe_encoder(parakeet_coreml_context * ctx, int n_mels) {
    EncoderRouteGeometry g;
    const int64_t frames = parakeet_coreml_fixed_mel_frames(ctx, n_mels);
    if (frames > 0 && frames <= std::numeric_limits<int>::max()) g.fixed_mel_frames = (int) frames;
    g.seam_context_frames = (int) parakeet_coreml_seam_context_frames(ctx);
    g.masks_mel_padding = parakeet_coreml_masks_mel_padding(ctx);
    return g;
}

const char * encoder_label(parakeet_coreml_context * ctx) {
    return parakeet_coreml_backend_label(ctx);
}

int encode_on_context(void *, parakeet_coreml_context * ctx, int n_mel_frames, int mel_valid,
                      int n_mels, const float * mel, int n_enc_frames, int d_model,
                      float * encoder_out) {
    return parakeet_coreml_encode_valid(ctx, n_mel_frames, mel_valid, n_mels, mel,
                                        n_enc_frames, d_model, encoder_out);
}

const EncoderModelOps & encoder_ops() {
    static const EncoderModelOps ops{&encoder_context_of, &describe_encoder, &encoder_label,
                                     &encode_on_context};
    return ops;
}

// The synchronous context as a route. The owner never retires, and a
// failure is an error (there are no ggml encoder weights to fall back to).
// Under use = require it is strict like a slot's route: a dispatch it cannot
// serve throws.
// TranscribePsy EU's English program gives the route its geometry (the
// long-form plan and the segment cap); its dispatches pick their program in
// run_encoder.
EncoderRoute synchronous_route(const ParakeetCtcModel & m, bool strict) {
    EncoderRoute::Ops ops;
    ops.encode = &encode_on_context;
    return EncoderRoute::coreml(m.impl->ctx_coreml, describe_encoder(m.impl->ctx_coreml, m.mel_cfg.n_mels),
                                parakeet_coreml_backend_label(m.impl->ctx_coreml),
                                std::static_pointer_cast<void>(m.impl), ops,
                                /*owner=*/m.impl->coreml_owns_encoder,
                                strict);
}

bool resolved_strict(const ParakeetCtcModel & m) {
    return m.impl && m.impl->coreml && m.impl->coreml->resolved().strict();
}

// The encoder route if it is ready now, without requesting anything. A probe
// (encoder_on_coreml, encoder_backend, the ggml prewarm pass): never throws
// and is never strict, whatever use is.
EncoderRoute ready_encoder_route(const ParakeetCtcModel & m) {
    if (!m.impl) return {};
    if (m.impl->ctx_coreml != nullptr) return synchronous_route(m, /*strict=*/false);
    if (!m.impl->encoder_slot) return {};
    return route_from_acquired(acquire_if_ready(*m.impl->encoder_slot), encoder_ops(), m.mel_cfg.n_mels,
                               /*strict=*/false);
}

}  // namespace
#endif

EncoderRoute acquire_encoder_route(ParakeetCtcModel & m, const std::atomic<bool> * cancel) {
#ifdef PARAKEET_USE_COREML
    if (!m.impl) return {};
    if (cancel == nullptr) cancel = m.impl->unit_cancel;
    if (m.impl->ctx_coreml != nullptr) return synchronous_route(m, resolved_strict(m));
    if (!m.impl->encoder_slot) return {};
    // One acquisition per unit: raised to `now`, then the resolved wait mode
    // (none: no blocking; call: until ready or terminal; strict throws).
    return route_from_acquired(m.impl->encoder_slot->acquire(pcm::Demand::now, std::nullopt, cancel),
                               encoder_ops(), m.mel_cfg.n_mels, m.impl->coreml->resolved().strict());
#else
    (void) m;
    (void) cancel;
    return {};
#endif
}

void model_set_cancel_flag(ParakeetCtcModel & m, const std::atomic<bool> * cancel) {
#ifdef PARAKEET_USE_COREML
    if (!m.impl) return;
    m.impl->unit_cancel = cancel;
    if (m.impl->coreml_diarizers) m.impl->coreml_diarizers->set_cancel(cancel);
#else
    (void) m;
    (void) cancel;
#endif
}

EncoderRoute encoder_route_if_ready(ParakeetCtcModel & m) {
#ifdef PARAKEET_USE_COREML
    return ready_encoder_route(m);
#else
    (void) m;
    return {};
#endif
}

std::string model_coreml_status_json(const ParakeetCtcModel & m) {
#ifdef PARAKEET_USE_COREML
    if (m.impl && m.impl->coreml) return m.impl->coreml->status_json();
#else
    (void) m;
#endif
    return "{}";
}

// A group with no sidecar to prepare (no Core ML in the build, no sidecar,
// or none planned): finished and successful at once.
static std::string empty_group_json(const char * group) {
    return std::string("{\"group\":\"") + group +
           "\",\"nothing_to_prepare\":true,\"skipped_route_none\":[],\"slots\":[],\"success\":true}";
}

std::string model_coreml_construction_json(const ParakeetCtcModel & m) {
#ifdef PARAKEET_USE_COREML
    if (m.impl && m.impl->coreml) return m.impl->coreml->construction_json();
#else
    (void) m;
#endif
    return empty_group_json("construction");
}

std::string model_coreml_prewarm(ParakeetCtcModel & m, CoremlOptions::Prewarm scope, bool wait,
                                 const std::atomic<bool> * cancel) {
#ifdef PARAKEET_USE_COREML
    if (m.impl && m.impl->coreml) return pcm::to_json(m.impl->coreml->prewarm(scope, wait, cancel));
#else
    (void) m;
    (void) scope;
    (void) wait;
    (void) cancel;
#endif
    return empty_group_json("prewarm");
}

bool model_coreml_wait(ParakeetCtcModel & m, int64_t timeout_ms, const std::atomic<bool> * cancel) {
#ifdef PARAKEET_USE_COREML
    if (m.impl && m.impl->coreml) return m.impl->coreml->wait_construction(timeout_ms, cancel);
#else
    (void) m;
    (void) timeout_ms;
    (void) cancel;
#endif
    return true;  // nothing to wait for
}

void model_coreml_reset_call_stats(ParakeetCtcModel & m) {
#ifdef PARAKEET_USE_COREML
    if (m.impl && m.impl->coreml_diarizers) m.impl->coreml_diarizers->reset_call_stats();
#else
    (void) m;
#endif
}

CoremlCallStats model_coreml_call_stats(ParakeetCtcModel & m) {
    CoremlCallStats out;
#ifdef PARAKEET_USE_COREML
    if (m.impl && m.impl->coreml_diarizers) {
        const CoremlDiarizerCallStats stats = m.impl->coreml_diarizers->call_stats();
        out.coreml_passes = stats.coreml;
        out.pending_passes = stats.not_ready;
        out.label = stats.label;
        out.capacity = stats.capacity;
    }
#else
    (void) m;
#endif
    return out;
}

bool model_encoder_on_coreml(const ParakeetCtcModel & m) {
#ifdef PARAKEET_USE_COREML
    if (!m.impl) return false;
    if (ready_encoder_route(m).holds()) return true;
    // TranscribePsy EU without its English fixed program: a flexible or
    // precise program, validated on demand.
    if (m.model_variant == "transcribepsy-250m-eu" && model_selected_encoder_on_coreml(m, 1)) {
        return true;
    }
    if (!m.impl->coreml_diarizers) return false;
    for (const auto & slot : m.impl->coreml->slots()) {
        if (slot != m.impl->encoder_slot && slot != m.impl->bypass_slot &&
            slot->state() == pcm::SlotState::ready) {
            return true;
        }
    }
    return false;
#else
    (void) m;
    return false;
#endif
}

bool model_encoder_owned_by_coreml(const ParakeetCtcModel & m) {
    return m.impl && m.impl->coreml_owns_encoder;
}

int model_coreml_seam_context_frames(const ParakeetCtcModel & m) {
#ifdef PARAKEET_USE_COREML
    return ready_encoder_route(m).geometry().seam_context_frames;
#else
    (void) m;
    return 0;
#endif
}

::parakeet_coreml_decoder * model_coreml_decoder(const ParakeetCtcModel & m) {
#ifdef PARAKEET_USE_COREML
    return m.impl ? m.impl->ctx_coreml_dec : nullptr;
#else
    (void) m;
    return nullptr;
#endif
}

std::string model_decoder_backend_name(const ParakeetCtcModel & m, bool include_coreml) {
    std::string coreml_backend;
#ifdef PARAKEET_USE_COREML
    if (m.impl && m.impl->ctx_coreml_dec) {
        coreml_backend = parakeet_coreml_decoder_backend_label(m.impl->ctx_coreml_dec);
    }
#endif
    const std::string cpu_backend = m.impl && m.impl->backend_cpu
        ? ggml_backend_name(m.impl->backend_cpu) : "";
    return resolve_decoder_backend_name(model_active_backend_name(m), cpu_backend,
        model_tdt_decoder_on_cpu(m), coreml_backend, include_coreml);
}

bool model_bypass_encoder_on_coreml(const ParakeetCtcModel & m) {
#ifdef PARAKEET_USE_COREML
    return m.impl && m.impl->bypass_slot && m.impl->bypass_slot->state() == pcm::SlotState::ready;
#else
    (void) m;
    return false;
#endif
}

int model_coreml_fixed_mel_frames(const ParakeetCtcModel & m) {
#ifdef PARAKEET_USE_COREML
    return ready_encoder_route(m).geometry().fixed_mel_frames;
#else
    (void) m;
    return 0;
#endif
}

int model_coreml_bypass_fixed_frames(const ParakeetCtcModel & m) {
#ifdef PARAKEET_USE_COREML
    if (m.impl && m.impl->bypass_slot) {
        pcm::Acquired a = acquire_if_ready(*m.impl->bypass_slot);
        if (a.model) {
            return (int) parakeet_coreml_fixed_mel_frames(
                a.model->as<parakeet_coreml_context>(), m.encoder_cfg.d_model);
        }
    }
#else
    (void) m;
#endif
    return 0;
}

std::string model_encoder_backend_name(const ParakeetCtcModel & m) {
#ifdef PARAKEET_USE_COREML
    // TranscribePsy EU reports where its last dispatch ran (its programs
    // differ per route, length and placement).
    if (m.impl && m.model_variant == "transcribepsy-250m-eu" &&
        !m.impl->last_coreml_encoder_backend.empty()) {
        return m.impl->last_coreml_encoder_backend;
    }
    if (const EncoderRoute route = ready_encoder_route(m); route.holds()) {
        return route.coreml_label();
    }
    if (m.impl && m.impl->coreml_diarizers && model_encoder_on_coreml(m)) {
        return m.impl->coreml_diarizers->label();
    }
#endif
    return model_active_backend_name(m);
}

ggml_backend_t init_engine_backend(bool use_gpu) {
    bool skipped_unsupported_gpu = false;
    bool is_mali_vulkan = false;
    ggml_backend_t backend = use_gpu ? init_gpu_backend(1, false, skipped_unsupported_gpu, is_mali_vulkan) : nullptr;
    return backend ? backend : init_cpu_backend();
}

ggml_backend_t init_engine_cpu_backend() {
    return init_cpu_backend();
}

ggml_backend_t init_engine_backend(bool use_gpu, const std::string & requested) {
    if (backend_selection_is_auto(requested)) return init_engine_backend(use_gpu);
    ggml_backend_t cpu = init_cpu_backend();
    if (!cpu) return nullptr;
    bool is_mali_vulkan = false;
    ggml_backend_t selected = init_explicit_backend(requested, cpu, is_mali_vulkan);
    if (selected != cpu) ggml_backend_free(cpu);
    return selected;
}

ggml_backend_t model_active_backend(ParakeetCtcModel & m) {
    if (!m.impl) return nullptr;
    return m.impl->backend_active;
}

ggml_backend_t model_sortformer_backend(const ParakeetCtcModel & m) {
    if (!m.impl) return nullptr;
    return m.impl->sortformer_force_cpu ? m.impl->backend_cpu
                                        : m.impl->backend_active;
}

bool model_sortformer_on_cpu(const ParakeetCtcModel & m) {
    return m.impl && m.impl->sortformer_force_cpu;
}

ggml_backend_sched_t model_sched(const ParakeetCtcModel & m) {
    return m.impl ? m.impl->sched : nullptr;
}

const char * model_type_name(ParakeetModelType model_type) {
    switch (model_type) {
        case ParakeetModelType::RNNT:       return "rnnt";
        case ParakeetModelType::TDT:        return "tdt";
        case ParakeetModelType::EOU:        return "eou";
        case ParakeetModelType::NEMOTRON:   return "nemotron";
        case ParakeetModelType::SORTFORMER: return "sortformer";
        case ParakeetModelType::NEMOTRON_DIARIZATION: return "nemotron-diarization";
        case ParakeetModelType::CTC:
        default:                            return "ctc";
    }
}

namespace {

std::string join_frames(const std::vector<int32_t> & values) {
    std::string out = "{";
    for (size_t i = 0; i < values.size(); ++i) {
        out += (i ? "," : "") + std::to_string(values[i]);
    }
    return out + "}";
}

}

void print_model_summary(const ParakeetCtcModel & m) {
    const char * mt = model_type_name(m.model_type);
    PARAKEET_LOG_INFO("parakeet-%s loaded:\n", mt);
    if (m.model_type == ParakeetModelType::NEMOTRON_DIARIZATION) {
        const auto & cfg = m.nemotron_diarization_cfg;
        PARAKEET_LOG_INFO(
            "  encoder: RoPE %d layers, width=%d, heads=%d, feature stack=%dx\n"
            "  diarization: speakers=%d, output_stride=%d mel frame\n",
            cfg.encoder_layers, cfg.encoder_width, cfg.attention_heads,
            cfg.subsampling_factor, cfg.speakers, cfg.output_stride);
        PARAKEET_LOG_INFO("  preproc: sr=%d n_fft=%d win=%d hop=%d n_mels=%d\n",
            m.mel_cfg.sample_rate, m.mel_cfg.n_fft, m.mel_cfg.win_length,
            m.mel_cfg.hop_length, m.mel_cfg.n_mels);
        return;
    }
    const char * conv_norm = m.encoder_cfg.conv_norm_type == ConvNormType::LayerNorm ? "ln" : "bn";
    PARAKEET_LOG_INFO("  encoder: d_model=%d n_layers=%d n_heads=%d head_dim=%d ff_dim=%d conv_k=%d sub=%dx xscaling=%d untie=%d use_bias=%d conv_norm=%s\n",
                      m.encoder_cfg.d_model, m.encoder_cfg.n_layers, m.encoder_cfg.n_heads,
                      m.encoder_cfg.head_dim, m.encoder_cfg.ff_dim, m.encoder_cfg.conv_kernel,
                      m.encoder_cfg.subsampling_factor,
                      (int) m.encoder_cfg.xscaling, (int) m.encoder_cfg.untie_biases,
                      (int) m.encoder_cfg.use_bias, conv_norm);
    if (m.encoder_cfg.att_chunked_limited ||
        m.encoder_cfg.att_dynamic_chunking ||
        m.encoder_cfg.conv_dynamic_chunking ||
        m.encoder_cfg.causal_downsampling ||
        m.encoder_cfg.conv_causal) {
        PARAKEET_LOG_INFO("  streaming: att_ctx=[%d,%d] style=%s causal_ds=%d conv_ctx=%s\n",
                          m.encoder_cfg.att_context_left, m.encoder_cfg.att_context_right,
                          m.encoder_cfg.att_dynamic_chunking
                              ? "chunked_limited_with_rc"
                              : (m.encoder_cfg.att_chunked_limited ? "chunked_limited" : "regular"),
                          (int) m.encoder_cfg.causal_downsampling,
                          m.encoder_cfg.conv_dynamic_chunking
                              ? "dcc"
                              : (m.encoder_cfg.conv_causal ? "causal" : "default"));
    }
    if (m.unified_cfg.available) {
        PARAKEET_LOG_INFO("  unified streaming: left=%d frames conv_cache=%d chunks=%s right=%s\n",
                          m.unified_cfg.left_context_frames,
                          m.unified_cfg.cache_time_steps,
                          join_frames(m.unified_cfg.allowed_chunk_frames).c_str(),
                          join_frames(m.unified_cfg.allowed_right_context_frames).c_str());
    }
    PARAKEET_LOG_INFO("  preproc: sr=%d n_fft=%d win=%d hop=%d n_mels=%d preemph=%.2f log_guard=%.2e\n",
                      m.mel_cfg.sample_rate, m.mel_cfg.n_fft, m.mel_cfg.win_length,
                      m.mel_cfg.hop_length, m.mel_cfg.n_mels, m.mel_cfg.preemph,
                      (double) m.mel_cfg.log_zero_guard_value);
    if (m.model_type == ParakeetModelType::CTC) {
        PARAKEET_LOG_INFO("  ctc:     vocab=%d blank=%d langs=%zu\n",
                          m.vocab_size, m.blank_id, m.ctc_lang_ranges.size());
    } else if (m.model_type == ParakeetModelType::RNNT) {
        PARAKEET_LOG_INFO(
            "  rnnt:    vocab=%d blank=%d pred_hidden=%d pred_layers=%d "
            "joint_hidden=%d max_syms=%d\n",
            m.vocab_size, m.blank_id,
            m.encoder_cfg.rnnt_pred_hidden,
            m.encoder_cfg.rnnt_pred_rnn_layers,
            m.encoder_cfg.rnnt_joint_hidden,
            m.encoder_cfg.rnnt_max_symbols_per_step);
    } else if (m.model_type == ParakeetModelType::EOU) {
        PARAKEET_LOG_INFO("  eou:     vocab=%d blank=%d eou_id=%d eob_id=%d "
                          "pred_hidden=%d pred_layers=%d joint_hidden=%d "
                          "chunk_mel=%d cache_lookback=%d cache_time=%d max_syms=%d\n",
                          m.vocab_size, m.blank_id, m.eou_id, m.eob_id,
                          m.encoder_cfg.eou_pred_hidden, m.encoder_cfg.eou_pred_rnn_layers,
                          m.encoder_cfg.eou_joint_hidden,
                          m.encoder_cfg.eou_chunk_mel_frames,
                          m.encoder_cfg.eou_cache_lookback_frames,
                          m.encoder_cfg.eou_cache_time_steps,
                          m.encoder_cfg.eou_max_symbols_per_step);
    } else if (m.model_type == ParakeetModelType::NEMOTRON) {
        PARAKEET_LOG_INFO(
            "  nemotron: vocab=%d blank=%d prompts=%d pred_hidden=%d "
            "pred_layers=%d joint_hidden=%d locales=%zu default_locale=%s\n",
            m.vocab_size,
            m.blank_id,
            m.nemotron_cfg.num_prompts,
            m.nemotron_cfg.pred_hidden,
            m.nemotron_cfg.pred_rnn_layers,
            m.nemotron_cfg.joint_hidden,
            m.nemotron_cfg.locale_prompts.size(),
            m.nemotron_cfg.default_locale.c_str());
    } else if (m.model_type == ParakeetModelType::SORTFORMER) {
        PARAKEET_LOG_INFO("  sortformer: num_spks=%d  fc_d_model=%d  tf=%dlx%dh d_model=%d inner=%d pre_ln=%d\n",
                          m.encoder_cfg.sortformer_num_spks,
                          m.encoder_cfg.sortformer_fc_d_model,
                          m.encoder_cfg.sortformer_tf_n_layers,
                          m.encoder_cfg.sortformer_tf_n_heads,
                          m.encoder_cfg.sortformer_tf_d_model,
                          m.encoder_cfg.sortformer_tf_inner_size,
                          (int) m.encoder_cfg.sortformer_tf_pre_ln);
    } else {
        PARAKEET_LOG_INFO("  tdt:     vocab=%d blank=%d pred_hidden=%d pred_layers=%d joint_hidden=%d durations=[",
                          m.vocab_size, m.blank_id,
                          m.encoder_cfg.tdt_pred_hidden, m.encoder_cfg.tdt_pred_rnn_layers,
                          m.encoder_cfg.tdt_joint_hidden);
        for (size_t i = 0; i < m.tdt_durations.size(); ++i) {
            PARAKEET_LOG_INFO("%s%d", i ? "," : "", m.tdt_durations[i]);
        }
        PARAKEET_LOG_INFO("]\n");
    }
    PARAKEET_LOG_INFO("  tensors: filterbank=%ldx%ld window=%ld blocks=%zu\n",
                      (long) m.mel_filterbank->ne[0], (long) m.mel_filterbank->ne[1],
                      (long) m.window->ne[0], m.blocks.size());
}

namespace {

ggml_tensor * zero_pad_dim0(ggml_context * ctx, ggml_tensor * x, int p_front, int p_back);
ggml_tensor * zero_pad_dim1(ggml_context * ctx, ggml_tensor * x, int p_front, int p_back);

ggml_tensor * conv_bias_bcast(ggml_context * ctx, ggml_tensor * bias, int64_t C) {
    return ggml_reshape_4d(ctx, bias, 1, 1, C, 1);
}

// The mel front-end zero-fills every frame past the valid count, so the subsampling runs on
// the valid prefix alone and its output is zero-padded back to the padded frame count.
ggml_tensor * valid_mel_prefix(ggml_context * ctx, ggml_tensor * mel_in, int n_valid_frames) {
    if (n_valid_frames >= mel_in->ne[1]) return mel_in;
    return ggml_view_4d(ctx, mel_in, mel_in->ne[0], n_valid_frames, mel_in->ne[2], mel_in->ne[3],
                        mel_in->nb[1], mel_in->nb[2], mel_in->nb[3], 0);
}

// ggml_conv_2d's trailing permute+cont only reorders the batch dim, so for N == 1 the matmul
// result already has the [OW, OH, OC, 1] layout and the copy is skipped.
ggml_tensor * conv_2d_im2col(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b, int s, int p) {
    if (b->ne[3] != 1) return ggml_conv_2d(ctx, a, b, s, s, p, p, 1, 1);
    ggml_tensor * im2col = ggml_im2col(ctx, a, b, s, s, p, p, 1, 1, true, a->type);
    ggml_tensor * cols   = ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[1] * im2col->ne[2]);
    ggml_tensor * kernel = ggml_reshape_2d(ctx, a, a->ne[0] * a->ne[1] * a->ne[2], a->ne[3]);
    ggml_tensor * y = affine_mul_mat(ctx, cols, kernel);
    return ggml_reshape_4d(ctx, y, im2col->ne[1], im2col->ne[2], a->ne[3], 1);
}

// A 1x1 kernel's im2col is just the channel-innermost copy of the input, so take
// that copy directly: same F16 rounding, same GEMM operands, one pass instead of im2col.
ggml_tensor * pointwise_conv_2d(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    if (a->ne[0] != 1 || a->ne[1] != 1 || b->ne[3] != 1) return conv_2d_im2col(ctx, a, b, 1, 0);
    ggml_tensor * cols = ggml_cast(ctx, ggml_permute(ctx, b, 1, 2, 0, 3), a->type);
    cols = ggml_reshape_2d(ctx, cols, b->ne[2], b->ne[0] * b->ne[1]);
    ggml_tensor * kernel = ggml_reshape_2d(ctx, a, a->ne[2], a->ne[3]);
    ggml_tensor * y = affine_mul_mat(ctx, cols, kernel);
    return ggml_reshape_4d(ctx, y, b->ne[0], b->ne[1], a->ne[3], 1);
}

ggml_tensor * subsampling_graph(ggml_context    * gctx,
                                ggml_tensor     * mel_in,
                                const SubsamplingWeights & S,
                                int               subsampling_channels,
                                int               /*d_model*/,
                                int               n_valid_frames,
                                int               n_out_frames,
                                bool              causal_downsampling,
                                bool              dw_direct) {
    ggml_tensor * x = valid_mel_prefix(gctx, mel_in, n_valid_frames);

    // NeMo's CausalConv2D for `causal_downsampling=true` applies an
    // asymmetric (L=stride, R=stride-1) zero-pad on **both** the freq
    // (ne[0]) and time (ne[1]) axes per stride-2 conv (kernel=3), then
    // calls the conv with `padding=0`. Total pad = stride+stride-1 = 3,
    // so each layer's spatial output is `(F+3-3)/2 + 1 = F/2 + 1` --
    // freq goes 128 -> 65 -> 33 -> 17 (instead of the 128 -> 64 -> 32
    // -> 16 the symmetric `pad=1` baseline produces). The trained
    // `encoder.subsampling.out.weight` has 17 freq-bin slots, so this
    // asymmetric padding is mandatory for the matmul to line up.
    auto causal_pad = [&](ggml_tensor * xin) {
        if (!causal_downsampling) return xin;
        xin = zero_pad_dim0(gctx, xin, /*L=*/2, /*R=*/1);
        xin = zero_pad_dim1(gctx, xin, /*L=*/2, /*R=*/1);
        return xin;
    };
    const int conv_pad = causal_downsampling ? 0 : 1;

    // Mali-Vulkan fix: the subsampler depthwise conv, lowered by hand like the
    // portable ggml_conv_2d_dw. Upstream finishes the lowering with a broadcast
    // ggml_mul_mat (kernel batched per-channel, broadcast over the input batch)
    // that miscomputes to inf on Mali-G715/Valhall Vulkan. Express the identical
    // per-channel contraction WITHOUT a mul_mat instead: an elementwise ggml_mul
    // (kernel broadcast over the spatial + batch dims) then ggml_sum_rows over the
    // KH*KW dim. The im2col stays F32 (ggml-vulkan SUM_ROWS needs an F32,
    // contiguous-rows src0). Adreno/OpenCL and Apple/Metal compute the depthwise
    // correctly already and stay bit-identical.
    auto conv_2d_dw_f32 = [&](ggml_tensor * a, ggml_tensor * b,
                              int s0, int s1, int p0, int p1, int d0, int d1) -> ggml_tensor * {
        ggml_tensor * kf32 = a->type == GGML_TYPE_F32 ? a : ggml_cast(gctx, a, GGML_TYPE_F32);
        if (dw_direct) {
            ggml_tensor * k4 = ggml_reshape_4d(gctx, kf32, kf32->ne[0], kf32->ne[1], 1, kf32->ne[2] * kf32->ne[3]);
            return ggml_conv_2d_dw_direct(gctx, k4, b, s0, s1, p0, p1, d0, d1);
        }
        ggml_tensor * new_a = ggml_reshape_4d(gctx, kf32, kf32->ne[0], kf32->ne[1], 1, kf32->ne[2] * kf32->ne[3]);
        ggml_tensor * im2col = ggml_im2col(gctx, new_a,
                                  ggml_reshape_4d(gctx, b, b->ne[0], b->ne[1], 1, b->ne[2] * b->ne[3]),
                                  s0, s1, p0, p1, d0, d1, true, GGML_TYPE_F32);
        ggml_tensor * new_b = ggml_reshape_4d(gctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1], b->ne[2], b->ne[3]);
        new_a = ggml_reshape_4d(gctx, new_a, new_a->ne[0] * new_a->ne[1], new_a->ne[2], new_a->ne[3], 1);
        ggml_tensor * prod = ggml_mul(gctx, new_b, new_a);   // [KH*KW, OH*OW, C, N]: kernel bcast over spatial + batch
        ggml_tensor * summed = ggml_sum_rows(gctx, prod);    // [1, OH*OW, C, N]: contract over the KH*KW dim
        return ggml_reshape_4d(gctx, summed, im2col->ne[1], im2col->ne[2], b->ne[2], b->ne[3]);
    };

    x = causal_pad(x);
    x = conv_2d_im2col(gctx, S.conv0_w, x, 2, conv_pad);
    x = ggml_add(gctx, x, conv_bias_bcast(gctx, S.conv0_b, subsampling_channels));
    x = ggml_relu(gctx, x);

    x = causal_pad(x);
    x = conv_2d_dw_f32(S.conv1_dw_w, x, 2, 2, conv_pad, conv_pad, 1, 1);
    x = ggml_add(gctx, x, conv_bias_bcast(gctx, S.conv1_dw_b, subsampling_channels));
    x = pointwise_conv_2d(gctx, S.conv1_pw_w, x);
    x = ggml_add(gctx, x, conv_bias_bcast(gctx, S.conv1_pw_b, subsampling_channels));
    x = ggml_relu(gctx, x);

    x = causal_pad(x);
    x = conv_2d_dw_f32(S.conv2_dw_w, x, 2, 2, conv_pad, conv_pad, 1, 1);
    x = ggml_add(gctx, x, conv_bias_bcast(gctx, S.conv2_dw_b, subsampling_channels));
    x = pointwise_conv_2d(gctx, S.conv2_pw_w, x);
    x = ggml_add(gctx, x, conv_bias_bcast(gctx, S.conv2_pw_b, subsampling_channels));
    x = ggml_relu(gctx, x);

    const int64_t W = x->ne[0];
    const int64_t H = x->ne[1];
    const int64_t C = x->ne[2];

    x = ggml_permute(gctx, x, 0, 2, 1, 3);
    x = ggml_cont(gctx, x);
    x = ggml_reshape_2d(gctx, x, W * C, H);
    x = zero_pad_dim1(gctx, x, 0, n_out_frames - (int) H);

    x = affine_mul_mat(gctx, S.out_w, x);
    x = ggml_add(gctx, x, S.out_b);
    return x;
}

int _conv_out_len(int L, int k, int s, int p) {
    return (L + 2 * p - k) / s + 1;
}

std::vector<float> compute_rel_pos_encoding(int T, int D) {
    const int L = 2 * T - 1;
    std::vector<float> pe((size_t) L * D, 0.0f);
    const float log10000 = std::log(10000.0f);
    std::vector<float> div_term(D / 2);
    for (int i = 0; i < D / 2; ++i) {
        div_term[i] = std::exp(-((float)(2 * i) * log10000 / (float) D));
    }
    std::vector<std::vector<float>> pos_pe(T, std::vector<float>(D, 0.0f));
    std::vector<std::vector<float>> neg_pe(T, std::vector<float>(D, 0.0f));
    for (int i = 0; i < T; ++i) {
        for (int k = 0; k < D / 2; ++k) {
            pos_pe[i][2*k]     = std::sin( (float) i * div_term[k]);
            pos_pe[i][2*k + 1] = std::cos( (float) i * div_term[k]);
            neg_pe[i][2*k]     = std::sin(-(float) i * div_term[k]);
            neg_pe[i][2*k + 1] = std::cos(-(float) i * div_term[k]);
        }
    }
    for (int t = 0; t < T; ++t) {
        int src = T - 1 - t;
        for (int d = 0; d < D; ++d) pe[(size_t) t * D + d] = pos_pe[src][d];
    }
    for (int t = 1; t < T; ++t) {
        for (int d = 0; d < D; ++d) pe[(size_t) (T - 1 + t) * D + d] = neg_pe[t][d];
    }
    return pe;
}

ggml_tensor * zero_pad_dim0(ggml_context * ctx, ggml_tensor * x, int p_front, int p_back) {
    if (p_front <= 0 && p_back <= 0) return x;
    ggml_tensor * y = x;
    int front_remaining = p_front;
    while (front_remaining > 0) {
        const int chunk = std::min<int64_t>(front_remaining, y->ne[0]);
        ggml_tensor * head = ggml_view_4d(ctx, y, chunk, y->ne[1], y->ne[2], y->ne[3],
                                          y->nb[1], y->nb[2], y->nb[3], 0);
        ggml_tensor * z = ggml_scale(ctx, ggml_cont(ctx, head), 0.0f);
        y = ggml_concat(ctx, z, y, 0);
        front_remaining -= chunk;
    }
    int back_remaining = p_back;
    while (back_remaining > 0) {
        const int chunk = std::min<int64_t>(back_remaining, y->ne[0]);
        ggml_tensor * tail = ggml_view_4d(ctx, y, chunk, y->ne[1], y->ne[2], y->ne[3],
                                          y->nb[1], y->nb[2], y->nb[3], 0);
        ggml_tensor * z = ggml_scale(ctx, ggml_cont(ctx, tail), 0.0f);
        y = ggml_concat(ctx, y, z, 0);
        back_remaining -= chunk;
    }
    return y;
}

// Asymmetric zero-pad along the second axis (ne[1]). Mirrors
// ``zero_pad_dim0`` but for the H dim so it can pre-pad the
// time axis of the (W=freq, H=time) mel layout used in the
// dw_striding subsampler.
ggml_tensor * zero_pad_dim1(ggml_context * ctx, ggml_tensor * x, int p_front, int p_back) {
    if (p_front <= 0 && p_back <= 0) return x;
    ggml_tensor * y = x;
    int front_remaining = p_front;
    while (front_remaining > 0) {
        const int chunk = std::min<int64_t>(front_remaining, y->ne[1]);
        ggml_tensor * head = ggml_view_4d(ctx, y, y->ne[0], chunk, y->ne[2], y->ne[3],
                                          y->nb[1], y->nb[2], y->nb[3], 0);
        ggml_tensor * z = ggml_scale(ctx, ggml_cont(ctx, head), 0.0f);
        y = ggml_concat(ctx, z, y, 1);
        front_remaining -= chunk;
    }
    int back_remaining = p_back;
    while (back_remaining > 0) {
        const int chunk = std::min<int64_t>(back_remaining, y->ne[1]);
        ggml_tensor * tail = ggml_view_4d(ctx, y, y->ne[0], chunk, y->ne[2], y->ne[3],
                                          y->nb[1], y->nb[2], y->nb[3], 0);
        ggml_tensor * z = ggml_scale(ctx, ggml_cont(ctx, tail), 0.0f);
        y = ggml_concat(ctx, y, z, 1);
        back_remaining -= chunk;
    }
    return y;
}

ggml_tensor * conv1d_via_matmul(ggml_context * ctx,
                                ggml_tensor * kernel, ggml_tensor * input,
                                int stride, int padding, int dilation) {
    ggml_tensor * kf32 = kernel->type == GGML_TYPE_F32
                       ? kernel
                       : ggml_cast(ctx, kernel, GGML_TYPE_F32);
    ggml_tensor * im2col = ggml_im2col(ctx, kf32, input,
                                       stride, 0, padding, 0, dilation, 0,
                                       false, GGML_TYPE_F32);
    ggml_tensor * r = affine_mul_mat(ctx,
        ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]),
        ggml_reshape_2d(ctx, kf32, kf32->ne[0] * kf32->ne[1], kf32->ne[2]));
    return ggml_reshape_3d(ctx, r, im2col->ne[1], kf32->ne[2], im2col->ne[2]);
}

ggml_tensor * layer_norm_affine(ggml_context * ctx, ggml_tensor * x,
                                ggml_tensor * w, ggml_tensor * b, float eps) {
    x = ggml_norm(ctx, x, eps);
    x = ggml_mul(ctx, x, w);
    x = ggml_add(ctx, x, b);
    return x;
}

ggml_tensor * maybe_add_bias(ggml_context * ctx, ggml_tensor * x, ggml_tensor * bias) {
    return bias ? ggml_add(ctx, x, bias) : x;
}

// The second linear carries the headroom: an early block's feed-forward can
// exceed the F16 range there (TranscribePsy's block 0 peaks near 76k on
// speech), while the encoder's other matmuls stay far below it.
ggml_tensor * conformer_ff_graph(ggml_context * ctx, ggml_tensor * x,
                                 ggml_tensor * norm_w, ggml_tensor * norm_b,
                                 ggml_tensor * l1_w,  ggml_tensor * l1_b,
                                 ggml_tensor * l2_w,  ggml_tensor * l2_b,
                                 float eps, float l2_headroom) {
    x = layer_norm_affine(ctx, x, norm_w, norm_b, eps);
    x = maybe_add_bias(ctx, affine_mul_mat(ctx, l1_w, x), l1_b);
    x = ggml_silu(ctx, x);
    x = maybe_add_bias(ctx, affine_mul_mat_with_headroom(ctx, l2_w, x, l2_headroom), l2_b);
    return x;
}

// Transformer-XL relative shift as a view: bd is [P (position), T (query), H] and the
// score bias for (key j, query i) is bd[T-1-i+j, i]. Reading each query row with a stride of
// (P - 1) elements from offset T-1 lands exactly on that element, so no padding or copies.
// P is normally 2T-1, but when the upstream `p_perm` gets tail-zero-padded to a 32-aligned
// row width (so the batched attention `bd = mul_mat(p_perm, q_v)` can land on the HMX F16
// batched kernel) P grows to `hmx_align_up_32(2T-1)`. The rel-shift trick still holds because
// only the (P - 1)-element stride depends on the actual row width, and the trailing padded
// columns (T-1 + i0 ... P-1) are never referenced by any query row (max col reached is
// (T-1) + (T-1) = 2T-2 < P). Parametrising on `bd->ne[0]` keeps this backward-compatible:
// when p_perm is unpadded, `bd->ne[0] - 1 = 2T - 2` exactly.
ggml_tensor * rel_shift_view(ggml_context * ctx, ggml_tensor * bd, int T) {
    const size_t f = sizeof(float);
    return ggml_view_3d(ctx, bd, T, T, bd->ne[2], (size_t) (bd->ne[0] - 1) * f, bd->nb[2], (size_t) (T - 1) * f);
}

struct RelPosAttnInputs {
    ggml_tensor * q;
    ggml_tensor * k;
    ggml_tensor * v;
    ggml_tensor * p;
    ggml_tensor * u_bias;
    ggml_tensor * v_bias;
    float         scale;
    ggml_tensor * p_perm = nullptr;  // cached (HD, 2T-1, H) projection for the unfused path
};

ggml_tensor * rel_pos_mha_flash_graph(ggml_context * ctx, const RelPosAttnInputs & in,
                                      ggml_tensor * att_mask, const BlockWeights & W,
                                      int H, int HD, int T, bool per_head_mask);

// Diagnostic capture handles. Populated in the unfused path when
// non-null; ignored on flash-attn (single FLASH_ATTN_EXT node, nothing to
// snapshot). Only block 0 in the encoder passes a non-null pointer.
struct AttnCapture {
    ggml_tensor * qkv     = nullptr;   // post-projection stacked qkv, or Q alone in the split-w path
    ggml_tensor * k_raw   = nullptr;   // post-projection K, before permute-cont
    ggml_tensor * v_raw   = nullptr;   // post-projection V, before permute-cont
    ggml_tensor * q_perm  = nullptr;   // post cont(permute(q, 0, 2, 1, 3))
    ggml_tensor * k_perm  = nullptr;   // post cont(permute(k, 0, 2, 1, 3))
    ggml_tensor * q_u     = nullptr;   // q_perm + u_bias
    ggml_tensor * ac      = nullptr;
    ggml_tensor * bd      = nullptr;
    ggml_tensor * scores  = nullptr;
    ggml_tensor * softmax = nullptr;
    ggml_tensor * out     = nullptr;
};

ggml_tensor * rel_pos_mha_unfused_graph(ggml_context * ctx, const RelPosAttnInputs & in,
                                        ggml_tensor * att_mask, const BlockWeights & W,
                                        int H, int HD, int T, int matmul_head_dim,
                                        AttnCapture * cap = nullptr);

ggml_tensor * rel_pos_mha_graph(ggml_context * ctx, ggml_tensor * xn,
                                ggml_tensor * pos_emb,
                                ggml_tensor * pos_proj,
                                ggml_tensor * att_mask,
                                const BlockWeights & W,
                                int H, int HD, int T,
                                const AttnPath & attn,
                                AttnCapture * cap = nullptr) {
    ggml_tensor * q;
    ggml_tensor * k;
    ggml_tensor * v;
    ggml_tensor * qkv_full = nullptr;
    if (W.attn_qkv_w) {
        // Pre-stacked encoder.blk.*.attn.qkv.weight from the converter:
        // one Q8_0 mat-mul produces (3 * n_embd, T) and Q / K / V are
        // strided ggml_view_3d slices straight into the (HD, H, T)
        // shape the reshape branch below produces.  Parent T-stride is
        // 3 * n_embd * f instead of n_embd * f, but the next ops
        // (permute → cont) walk by per-element nb01 / nb02 strides so
        // the wider stride is transparent.
        //
        // Whether `W.attn_qkv_w` was loaded at all is decided in the
        // model loader (gated to non-Apple-Metal backends — see the
        // `backend_is_metal` branch in `load_model_gguf`).  M3
        // Ultra Metal already saturates the un-stacked path's tile
        // grid (M=1024 / T=252 → 16 row × 8 col tiles ≈ 128 chunks vs
        // 60 cores: one wave fills the GPU); the stacked M=3072 path
        // adds 3× row tiles where one wave was sufficient, which is
        // measured neutral-to-slightly-bad on Apple.  CUDA / Vulkan
        // hardware with higher per-dispatch overhead and proportionally
        // smaller tiles relative to SM count is the predicted-positive
        // case; that's what the loader gate is for.
        ggml_tensor * qkv = affine_mul_mat(ctx, W.attn_qkv_w, xn);
        if (W.attn_qkv_b) qkv = ggml_add(ctx, qkv, W.attn_qkv_b);
        qkv_full = qkv;
        const int n_embd = HD * H;
        const size_t f = sizeof(float);
        const size_t row_stride = (size_t) 3 * n_embd * f;
        q = ggml_view_3d(ctx, qkv, HD, H, T, HD * f, row_stride, 0 * (size_t) n_embd * f);
        k = ggml_view_3d(ctx, qkv, HD, H, T, HD * f, row_stride, 1 * (size_t) n_embd * f);
        v = ggml_view_3d(ctx, qkv, HD, H, T, HD * f, row_stride, 2 * (size_t) n_embd * f);
    } else {
        q = maybe_add_bias(ctx, affine_mul_mat(ctx, W.attn_q_w, xn), W.attn_q_b);
        k = maybe_add_bias(ctx, affine_mul_mat(ctx, W.attn_k_w, xn), W.attn_k_b);
        v = maybe_add_bias(ctx, affine_mul_mat(ctx, W.attn_v_w, xn), W.attn_v_b);
        if (cap) { cap->qkv = q; cap->k_raw = k; cap->v_raw = v; }
        q = ggml_reshape_3d(ctx, q, HD, H, T);
        k = ggml_reshape_3d(ctx, k, HD, H, T);
        v = ggml_reshape_3d(ctx, v, HD, H, T);
    }
    if (cap && !cap->qkv) cap->qkv = qkv_full;
    // The cache holds the projection in the layout its consumer reads: the flash path
    // permutes a (HD, H, 2T-1) tensor as before, the unfused path takes it pre-permuted.
    ggml_tensor * p = attn.flash_attn ? pos_proj : nullptr;
    if (!pos_proj) {
        p = affine_mul_mat(ctx, W.attn_pos_w, pos_emb);
        p = ggml_reshape_3d(ctx, p, HD, H, pos_emb->ne[1]);
    }

    ggml_tensor * u_bias = ggml_reshape_3d(ctx, W.pos_bias_u, HD, 1, H);
    ggml_tensor * v_bias = ggml_reshape_3d(ctx, W.pos_bias_v, HD, 1, H);

    const float scale = 1.0f / std::sqrt((float) HD);

    const RelPosAttnInputs in = { q, k, v, p, u_bias, v_bias, scale, attn.flash_attn ? nullptr : pos_proj };
    if (attn.flash_attn) {
        // Flash-attn collapses the whole attention into one op; the qkv
        // capture is still useful, the others cannot be intercepted.
        return rel_pos_mha_flash_graph(ctx, in, att_mask, W, H, HD, T, attn.per_head_mask);
    }
    return rel_pos_mha_unfused_graph(ctx, in, att_mask, W, H, HD, T, attn.matmul_head_dim, cap);
}

ggml_tensor * rel_pos_mha_flash_graph(ggml_context * ctx, const RelPosAttnInputs & in,
                                      ggml_tensor * att_mask, const BlockWeights & W,
                                      int H, int HD, int T, bool per_head_mask) {
    ggml_tensor * q = in.q;
    ggml_tensor * k = in.k;
    ggml_tensor * v = in.v;
    ggml_tensor * p = in.p;
    ggml_tensor * u_bias = in.u_bias;
    ggml_tensor * v_bias = in.v_bias;
    const float   scale  = in.scale;
    // Head-major views feed the bias adds, the position mat-mul and the fused
    // attention directly; nothing is materialised. The relative-position term is
    // built pre-scaled so the attention mask is softmax(scale * qk + mask) with
    //   mask = rel_shift(scale * (q + v_bias) . p) [+ att_mask]
    // and the Transformer-XL shift is a strided view (see rel_shift_view).
    ggml_tensor * qp = ggml_permute(ctx, q, 0, 2, 1, 3);
    ggml_tensor * kp = ggml_permute(ctx, k, 0, 2, 1, 3);
    ggml_tensor * vp = ggml_permute(ctx, v, 0, 2, 1, 3);
    ggml_tensor * pp = ggml_permute(ctx, p, 0, 2, 1, 3);
    ggml_tensor * q_u = ggml_add(ctx, qp, u_bias);
    ggml_tensor * q_v = ggml_scale(ctx, ggml_add(ctx, qp, v_bias), scale);

    ggml_tensor * bd       = affine_mul_mat(ctx, pp, q_v);
    ggml_tensor * bd_shift = rel_shift_view(ctx, bd, T);
    // Mode 2 / 3 streaming windows pass a (T_k, T_q, 1, 1) f32 additive mask that
    // must be folded in, otherwise attention leaks outside the window.
    ggml_tensor * fa_mask  = att_mask ? ggml_add(ctx, bd_shift, att_mask) : bd_shift;
    ggml_tensor * bd_mask  = ggml_cast(ctx, fa_mask, GGML_TYPE_F16);
    ggml_tensor * flat;
    if (per_head_mask) {
        ggml_tensor * attn_out = ggml_flash_attn_ext(ctx, q_u, kp, vp, bd_mask, scale, 0.0f, 0.0f);
        flat = ggml_reshape_2d(ctx, attn_out, HD * H, T);
    } else {
        // Backends whose fused attention only broadcasts the mask over heads (CUDA)
        // see H independent single-head sequences instead: heads move to dim 3.
        ggml_tensor * q4 = ggml_view_4d(ctx, q_u, HD, T, 1, H, q_u->nb[1], q_u->nb[2], q_u->nb[2], 0);
        ggml_tensor * k4 = ggml_view_4d(ctx, kp,  HD, T, 1, H, kp->nb[1],  kp->nb[2],  kp->nb[2],  0);
        ggml_tensor * v4 = ggml_view_4d(ctx, vp,  HD, T, 1, H, vp->nb[1],  vp->nb[2],  vp->nb[2],  0);
        ggml_tensor * m4 = ggml_reshape_4d(ctx, bd_mask, T, T, 1, H);
        ggml_tensor * attn_out = ggml_flash_attn_ext(ctx, q4, k4, v4, m4, scale, 0.0f, 0.0f);
        ggml_tensor * heads = ggml_reshape_3d(ctx, attn_out, HD, T, H);
        flat = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, heads, 0, 2, 1, 3)), HD * H, T);
    }
    return maybe_add_bias(ctx, affine_mul_mat(ctx, W.attn_out_w, flat), W.attn_out_b);
}

// Query rows [i0, i0+n) only meet relative positions [T-i0-n, 2T-2-i0], so per block the
// bd GEMM covers T-1+n positions instead of 2T-1 (same dots, same softmax rows, bit-identical).
constexpr int k_attn_block_rows  = 256;
constexpr int k_attn_block_min_T = 1024;

ggml_tensor * query_rows_view(ggml_context * ctx, ggml_tensor * x, int i0, int n) {
    return ggml_view_3d(ctx, x, x->ne[0], n, x->ne[2], x->nb[1], x->nb[2], (size_t) i0 * x->nb[1]);
}

ggml_tensor * rel_shift_block_view(ggml_context * ctx, ggml_tensor * bd_b, int T, int n) {
    const size_t f  = sizeof(float);
    const size_t Tp = (size_t) bd_b->ne[0];
    return ggml_view_3d(ctx, bd_b, T, n, bd_b->ne[2], (Tp - 1) * f, bd_b->nb[2], (size_t) (n - 1) * f);
}

ggml_tensor * attn_probs_block(ggml_context * ctx, ggml_tensor * k_perm, ggml_tensor * p_perm,
                               ggml_tensor * q_u, ggml_tensor * q_v, ggml_tensor * att_mask,
                               float scale, int T, int H, int HD, int i0, int n) {
    const int Tp = T - 1 + n;
    ggml_tensor * p_b  = ggml_view_3d(ctx, p_perm, HD, Tp, H, p_perm->nb[1], p_perm->nb[2],
                                      (size_t) (T - i0 - n) * p_perm->nb[1]);
    // ggml-cpu takes the tinyBLAS path only for a contiguous src1, so copy the query rows.
    ggml_tensor * bd_b = affine_mul_mat(ctx, p_b, ggml_cont(ctx, query_rows_view(ctx, q_v, i0, n)));
    ggml_tensor * ac_b = affine_mul_mat(ctx, k_perm, ggml_cont(ctx, query_rows_view(ctx, q_u, i0, n)));
    ggml_tensor * scores = ggml_add(ctx, ac_b, rel_shift_block_view(ctx, bd_b, T, n));
    ggml_tensor * mask_b = att_mask
        ? ggml_view_4d(ctx, att_mask, T, n, 1, 1, att_mask->nb[1], att_mask->nb[2], att_mask->nb[3],
                       (size_t) i0 * att_mask->nb[1])
        : nullptr;
    return ggml_soft_max_ext(ctx, scores, mask_b, scale, 0.0f);
}

// Attention context (HD*H, T) assembled from query blocks.
ggml_tensor * attn_context_blocked(ggml_context * ctx, ggml_tensor * k_perm, ggml_tensor * p_perm,
                                   ggml_tensor * v_for_mm, ggml_tensor * q_u, ggml_tensor * q_v,
                                   ggml_tensor * att_mask, float scale, int T, int H, int HD) {
    ggml_tensor * flat = nullptr;
    for (int i0 = 0; i0 < T; i0 += k_attn_block_rows) {
        const int n = std::min(k_attn_block_rows, T - i0);
        ggml_tensor * attn_b   = attn_probs_block(ctx, k_perm, p_perm, q_u, q_v, att_mask, scale, T, H, HD, i0, n);
        ggml_tensor * attn_v_b = affine_mul_mat(ctx, v_for_mm, attn_b);
        ggml_tensor * flat_b   = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, attn_v_b, 0, 2, 1, 3)), HD * H, n);
        flat = flat ? ggml_concat(ctx, flat, flat_b, 1) : flat_b;
    }
    return flat;
}

// Zero-pads dim 0 of `t` from `HD` to `head_dim`; a no-op when they match.
ggml_tensor * pad_head_dim(ggml_context * ctx, ggml_tensor * t, int HD, int head_dim) {
    return head_dim > HD ? ggml_pad(ctx, t, head_dim - HD, 0, 0, 0) : t;
}

ggml_tensor * rel_pos_mha_unfused_graph(ggml_context * ctx, const RelPosAttnInputs & in,
                                        ggml_tensor * att_mask, const BlockWeights & W,
                                        int H, int HD, int T, int matmul_head_dim,
                                        AttnCapture * cap) {
    ggml_tensor * q = in.q;
    ggml_tensor * k = in.k;
    ggml_tensor * v = in.v;
    ggml_tensor * p = in.p;
    ggml_tensor * u_bias = in.u_bias;
    ggml_tensor * v_bias = in.v_bias;
    const float   scale  = in.scale;
    ggml_tensor * q_perm = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    ggml_tensor * k_perm = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    ggml_tensor * v_perm = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
    ggml_tensor * p_perm = in.p_perm ? in.p_perm : ggml_cont(ctx, ggml_permute(ctx, p, 0, 2, 1, 3));
    ggml_tensor * q_u = ggml_add(ctx, q_perm, u_bias);
    ggml_tensor * q_v = ggml_add(ctx, q_perm, v_bias);
    if (cap) { cap->q_perm = q_perm; cap->k_perm = k_perm; cap->q_u = q_u; }

    // Batched attention matmul src0 (`k_perm`, `p_perm`, `v_for_mm`) is
    // derived from activations so `ggml_mul_mat` currently sees an F32 src0.
    // Casting src0 to F16 lets the Hexagon dispatcher pick the F16 batched
    // HMX path (`hmx_mm_f16_f32_batched`: src0=F16, src1=F32) instead of
    // falling through to HVX-batched-F32. Identity no-op when
    // PARAKEET_F16_ACTIVATIONS is OFF, preserving the F32 fast path.
    //
    // The HMX F16-batched kernel additionally requires k%32==0 and n%32==0
    // (matmul-ops.c:3081). Parakeet's encoder T (e.g. 376 for 60 s audio) is
    // not 32-aligned, so we also tail-zero-pad the relevant axis of every
    // src0:
    //   - `k_perm`   ne1=T → hmx_align_up_32(T)      (drives `ac`)
    //   - `p_perm`   ne1=2T-1 → hmx_align_up_32(2T-1) (drives `bd`)
    //   - `v_for_mm` ne0=T → hmx_align_up_32(T)      (drives `attn_v`)
    // Downstream, `rel_shift_view` is parametrised on `bd->ne[0]` so its
    // stride math walks the padded row width transparently; `ac` is sliced
    // back to (T, T, H); the softmax output is padded on ne0 to line up
    // with `v_for_mm`'s padded K axis. All three batched attention MULMATs
    // then dispatch to `hmx-tiled` (F16 batched) per block × 24 blocks.
    //
    // The T ≥ k_attn_block_min_T (1024) blocked branch below intentionally
    // bypasses the pad+cast: its per-block `attn_probs_block` builds strided
    // views over `k_perm` / `p_perm` and mixes `ac_b` (whose ne0 is src0.ne1)
    // with `rel_shift_block_view(bd_b)` (whose ne0 stays T), so a padded
    // src0.ne1 would break the downstream ADD's shape check
    // (`ggml_can_repeat` failure in `attn_probs_block`). Long-form audio hits
    // this branch and stays on HVX-batched-F32 as before; Parakeet-CTC
    // production audio is well under T=1024 so the HMX-F16 dispatch is
    // unaffected in practice.
    const bool use_blocked_attn = T >= k_attn_block_min_T;
    // A head size the HMX kernel cannot take (TranscribePsy's 80) is zero-padded
    // on every attention matmul operand as well (see attention_matmul_head_dim).
    const int HDm = (!use_blocked_attn && matmul_head_dim > HD) ? matmul_head_dim : HD;
    ggml_tensor * k_perm_mm = use_blocked_attn ? k_perm : pad_and_cast_for_hmx(ctx, pad_head_dim(ctx, k_perm, HD, HDm), /*dim=*/1);
    ggml_tensor * p_perm_mm = use_blocked_attn ? p_perm : pad_and_cast_for_hmx(ctx, pad_head_dim(ctx, p_perm, HD, HDm), /*dim=*/1);

    if (use_blocked_attn) {
        ggml_tensor * v_for_mm = ggml_cont(ctx, ggml_permute(ctx, v_perm, 1, 0, 2, 3));
        ggml_tensor * flat = attn_context_blocked(ctx, k_perm_mm, p_perm_mm, v_for_mm, q_u, q_v, att_mask, scale, T, H, HD);
        ggml_tensor * out = maybe_add_bias(ctx, affine_mul_mat(ctx, W.attn_out_w, flat), W.attn_out_b);
        if (cap) cap->out = out;
        return out;
    }

    ggml_tensor * bd = affine_mul_mat(ctx, p_perm_mm, pad_head_dim(ctx, q_v, HD, HDm));
    ggml_tensor * ac     = affine_mul_mat(ctx, k_perm_mm, pad_head_dim(ctx, q_u, HD, HDm));

    // When `k_perm` was tail-padded above the `ac` output is (T_pad, T, H);
    // slice the padded tail off (its rows are dots against zero keys) and
    // materialise a contiguous copy so the subsequent add + softmax see
    // exactly the pre-refactor (T, T, H) shape. No-op when unpadded, so the
    // F32 fast path is preserved.
    if (ac->ne[0] > (int64_t) T) {
        ac = ggml_cont(ctx, ggml_view_3d(ctx, ac, T, T, H, ac->nb[1], ac->nb[2], 0));
    }

    ggml_tensor * scores = ggml_add(ctx, ac, rel_shift_view(ctx, bd, T));

    ggml_tensor * attn;
    if (att_mask) {
        // ggml_soft_max_ext computes softmax(scale * x + mask) along the
        // last dim of `scores`. The kernel requires an f16 mask whose
        // shape broadcasts over the head axis: (T_k, T_q, 1, 1).
        attn = ggml_soft_max_ext(ctx, scores, att_mask, scale, 0.0f);
    } else {
        attn = ggml_soft_max_ext(ctx, scores, nullptr, scale, 0.0f);
    }

    // For the batched `attn_v = mul_mat(v_for_mm, attn)`, K==T is the matmul
    // reduction axis. Pad `v_for_mm` on its K axis (ne0=T→T_pad) and pad
    // `attn` correspondingly with zeros so the reduction over padded keys
    // contributes nothing. `attn_v`'s F32 output shape (HD, T, H) is
    // unchanged either way.
    ggml_tensor * v_for_mm = ggml_cont(ctx, ggml_permute(ctx, pad_head_dim(ctx, v_perm, HD, HDm), 1, 0, 2, 3));
    v_for_mm = pad_and_cast_for_hmx(ctx, v_for_mm, /*dim=*/0);
    ggml_tensor * attn_for_mm = pad_ne0_for_hmx(ctx, attn);
    ggml_tensor * attn_v   = affine_mul_mat(ctx, v_for_mm, attn_for_mm);
    if (HDm > HD) attn_v = ggml_view_3d(ctx, attn_v, HD, T, H, attn_v->nb[1], attn_v->nb[2], 0);
    ggml_tensor * merged   = ggml_cont(ctx, ggml_permute(ctx, attn_v, 0, 2, 1, 3));
    ggml_tensor * flat     = ggml_reshape_2d(ctx, merged, HD * H, T);

    ggml_tensor * out = maybe_add_bias(ctx, affine_mul_mat(ctx, W.attn_out_w, flat), W.attn_out_b);

    if (cap) {
        cap->ac      = ac;
        cap->bd      = bd;
        cap->scores  = scores;
        cap->softmax = attn;
        cap->out     = out;
    }
    return out;
}

// How the conformer depthwise conv is lowered on the backend that runs the graph.
enum class DwConvLowering {
    Im2col,         // ggml_conv_1d_dw: im2col + batched mat-mul, portable
    DirectPlanar,   // GGML_OP_CONV_2D_DW on a (T, d_model) transposed copy
    DirectChannels, // GGML_OP_CONV_2D_DW straight on the (d_model, T) activations, no transposes
};

// Channel-contiguous view of a (d_model, T) activation: logical (W=T, H=1, C=d_model, N=1)
// with the channel stride equal to the element size, which ggml_conv_2d_dw_direct treats
// as CWHN and answers in the same layout.
ggml_tensor * channels_first_view(ggml_context * ctx, ggml_tensor * y, int d_model) {
    ggml_tensor * y4 = ggml_reshape_4d(ctx, y, d_model, y->ne[1], 1, 1);
    return ggml_permute(ctx, y4, 2, 0, 1, 3);
}

// Depthwise kernel [K, C] re-laid as CWHN: a [K, 1, 1, C] logical view over a [C, K] copy.
ggml_tensor * channels_first_kernel(ggml_context * ctx, ggml_tensor * w_f32, int conv_kernel, int d_model) {
    ggml_tensor * w2 = ggml_reshape_2d(ctx, w_f32, conv_kernel, d_model);
    ggml_tensor * wt = ggml_cont(ctx, ggml_transpose(ctx, w2));
    return ggml_permute(ctx, ggml_reshape_4d(ctx, wt, d_model, conv_kernel, 1, 1), 3, 0, 1, 2);
}

ggml_tensor * depthwise_conv_channels_first(ggml_context * ctx, ggml_tensor * y,
                                            const BlockWeights & W,
                                            int d_model, int conv_kernel, int pad) {
    const int T = (int) y->ne[1];
    ggml_tensor * w_f32 = W.conv_dw_w->type == GGML_TYPE_F32
                        ? W.conv_dw_w
                        : ggml_cast(ctx, W.conv_dw_w, GGML_TYPE_F32);
    ggml_tensor * kernel = channels_first_kernel(ctx, w_f32, conv_kernel, d_model);
    ggml_tensor * out    = ggml_conv_2d_dw_direct(ctx, kernel, channels_first_view(ctx, y, d_model),
                                                  1, 1, pad, 0, 1, 1);
    return ggml_view_2d(ctx, out, d_model, T, (size_t) d_model * sizeof(float), 0);
}

// The channel-contiguous view of a [d_model, T] activation is only distinguishable
// from a plain contiguous tensor with two or more frames (nb[1] > nb[0]).
constexpr int64_t k_channels_first_min_frames = 2;

// CPU keeps its transposed direct kernel; a GPU backend that answered the CONV_2D_DW
// probe runs the depthwise conv in place, anything else uses the portable im2col lowering.
DwConvLowering select_dw_lowering(const ParakeetCtcModel & model, ggml_backend_t backend) {
    if (backend_is_cpu(backend))        return DwConvLowering::DirectPlanar;
    if (model.impl->dw_direct_channels) return DwConvLowering::DirectChannels;
    return DwConvLowering::Im2col;
}

ggml_tensor * mask_convolution_padding(ggml_context * ctx, ggml_tensor * x,
                                       int valid_frames) {
    if (valid_frames >= x->ne[1]) return x;
    const int padding = static_cast<int>(x->ne[1]) - valid_frames;
    auto * prefix = ggml_cont(ctx, valid_mel_prefix(ctx, x, valid_frames));
    return zero_pad_dim1(ctx, prefix, 0, padding);
}

ggml_tensor * conformer_conv_graph(ggml_context * ctx, ggml_tensor * xn,
                                   const BlockWeights & W,
                                   int d_model, int /*T*/, int conv_kernel,
                                   DwConvLowering dw_lowering,
                                   bool glu_fused,
                                   ConvNormType conv_norm_type,
                                   bool conv_causal,
                                   float layer_norm_eps, int valid_frames) {
    // Use the weight directly when the GGUF already stores it squeezed to
    // (d_model, 2*d_model): a reshape is a view node, and a view over a
    // CPU-repack weight drops the repack traits (the mul_mat would then read
    // the interleaved bytes as plain q4_0). Legacy 3D GGUFs still reshape --
    // those tensors are never placed in a repack buffer.
    ggml_tensor * pw1_w_2d = affine_weight_for_view(ctx, W.conv_pw1_w, xn);
    if (!(ggml_n_dims(pw1_w_2d) == 2 &&
          pw1_w_2d->ne[0] == d_model && pw1_w_2d->ne[1] == 2 * d_model)) {
        pw1_w_2d = ggml_reshape_2d(ctx, pw1_w_2d, d_model, 2 * d_model);
    }
    ggml_tensor * y = affine_mul_mat(ctx, pw1_w_2d, xn);
    y = maybe_add_bias(ctx, y, W.conv_pw1_b);

    // Conformer GLU: split (2*d_model, T, B) along the channel axis into two
    // halves, then y = half1 * sigmoid(half2). The gated op reads both halves
    // in place; where the backend lacks it, the halves are materialised first
    // because ggml-vulkan miscomputes the sigmoid / mul kernels on strided
    // inputs (validated on RTX 5060; see test-vk-vs-cpu bisect:
    // block0_conv_post_glu rel jumps from 1e-3 to ~0.8 without the contig).
    if (glu_fused) {
        y = ggml_siglu(ctx, y);
    } else {
        ggml_tensor * half1 = ggml_cont(ctx,
            ggml_view_3d(ctx, y, d_model, y->ne[1], y->ne[2],
                         y->nb[1], y->nb[2], 0));
        ggml_tensor * half2 = ggml_cont(ctx,
            ggml_view_3d(ctx, y, d_model, y->ne[1], y->ne[2],
                         y->nb[1], y->nb[2],
                         (size_t) d_model * y->nb[0]));
        y = ggml_mul(ctx, half1, ggml_sigmoid(ctx, half2));
    }

    y = mask_convolution_padding(ctx, y, valid_frames);

    const int pad_left  = conv_causal ? (conv_kernel - 1) : ((conv_kernel - 1) / 2);
    const int pad_right = conv_causal ? 0                 : ((conv_kernel - 1) / 2);

    const bool channels_first = dw_lowering == DwConvLowering::DirectChannels && !conv_causal &&
                                y->ne[1] >= k_channels_first_min_frames;
    if (channels_first) {
        y = depthwise_conv_channels_first(ctx, y, W, d_model, conv_kernel, pad_left);
        y = maybe_add_bias(ctx, y, W.conv_dw_b);
        if (conv_norm_type == ConvNormType::LayerNorm) {
            y = layer_norm_affine(ctx, y, W.conv_norm_w, W.conv_norm_b, layer_norm_eps);
        } else {
            y = ggml_mul(ctx, y, W.conv_bn_scale);
            y = ggml_add(ctx, y, W.conv_bn_shift);
        }
        y = ggml_silu(ctx, y);
        return maybe_add_bias(ctx, affine_mul_mat(ctx, W.conv_pw2_w, y), W.conv_pw2_b);
    }

    ggml_tensor * yt = ggml_cont(ctx, ggml_permute(ctx, y, 1, 0, 2, 3));

    if (dw_lowering == DwConvLowering::DirectPlanar) {
        const int T_local = (int) yt->ne[0];
        ggml_tensor * yt_4d = ggml_reshape_4d(ctx, yt, T_local, 1, d_model, 1);
        if (conv_causal && pad_left > 0) {
            yt_4d = zero_pad_dim0(ctx, yt_4d, pad_left, pad_right);
        }
        ggml_tensor * dw_kernel_f32 = W.conv_dw_w->type == GGML_TYPE_F32
                                    ? W.conv_dw_w
                                    : ggml_cast(ctx, W.conv_dw_w, GGML_TYPE_F32);
        ggml_tensor * dw_kernel_4d = ggml_reshape_4d(ctx, dw_kernel_f32, conv_kernel, 1, 1, d_model);
        const int dw_pad = conv_causal ? 0 : pad_left;
        ggml_tensor * dw_out = ggml_conv_2d_dw_direct(ctx, dw_kernel_4d, yt_4d, 1, 1, dw_pad, 0, 1, 1);
        yt = ggml_reshape_3d(ctx, dw_out, dw_out->ne[0], d_model, 1);
    } else {
        if (conv_causal && pad_left > 0) {
            yt = zero_pad_dim0(ctx, yt, pad_left, pad_right);
            yt = ggml_conv_1d_dw(ctx, W.conv_dw_w, yt, 1, 0, 1);
        } else {
            yt = ggml_conv_1d_dw(ctx, W.conv_dw_w, yt, 1, pad_left, 1);
        }
    }
    if (W.conv_dw_b) {
        yt = ggml_add(ctx, yt, ggml_reshape_2d(ctx, W.conv_dw_b, 1, d_model));
    }

    if (conv_norm_type == ConvNormType::LayerNorm) {
        // NeMo applies LayerNorm over the channel axis. After the depthwise
        // conv we are in (T, d_model) layout; transpose to (d_model, T) so
        // ggml_norm reduces across the channel dim, run LN, then continue
        // in (d_model, T) (saves one permute vs the BN path).
        y = ggml_cont(ctx, ggml_permute(ctx, yt, 1, 0, 2, 3));
        y = layer_norm_affine(ctx, y, W.conv_norm_w, W.conv_norm_b, layer_norm_eps);
        y = ggml_silu(ctx, y);
    } else {
        yt = ggml_mul(ctx, yt, ggml_reshape_2d(ctx, W.conv_bn_scale, 1, d_model));
        yt = ggml_add(ctx, yt, ggml_reshape_2d(ctx, W.conv_bn_shift, 1, d_model));
        yt = ggml_silu(ctx, yt);
        y  = ggml_cont(ctx, ggml_permute(ctx, yt, 1, 0, 2, 3));
    }

    // Same repack-traits consideration as pw1 above.
    ggml_tensor * pw2_w_2d = affine_weight_for_view(ctx, W.conv_pw2_w, y);
    if (!(ggml_n_dims(pw2_w_2d) == 2 &&
          pw2_w_2d->ne[0] == d_model && pw2_w_2d->ne[1] == d_model)) {
        pw2_w_2d = ggml_reshape_2d(ctx, pw2_w_2d, d_model, d_model);
    }
    y = affine_mul_mat(ctx, pw2_w_2d, y);
    y = maybe_add_bias(ctx, y, W.conv_pw2_b);

    return y;
}

ggml_tensor * conformer_block_graph(ggml_context * ctx, ggml_tensor * x,
                                    ggml_tensor * pos_emb,
                                    ggml_tensor * pos_proj,
                                    ggml_tensor * att_mask,
                                    const BlockWeights & W,
                                    int d_model, int H, int HD, int T,
                                    int conv_kernel, float eps,
                                    DwConvLowering dw_lowering,
                                    const AttnPath & attn,
                                    bool glu_fused,
                                    ConvNormType conv_norm_type,
                                    bool conv_causal,
                                    float ff_headroom, int valid_frames) {
    ggml_tensor * residual = x;
    ggml_tensor * y = conformer_ff_graph(ctx, x,
                                         W.norm_ff1_w, W.norm_ff1_b,
                                         W.ff1_l1_w,   W.ff1_l1_b,
                                         W.ff1_l2_w,   W.ff1_l2_b, eps, ff_headroom);
    y = ggml_scale(ctx, y, 0.5f);
    x = ggml_add(ctx, residual, y);

    residual = x;
    ggml_tensor * xn = layer_norm_affine(ctx, x, W.norm_attn_w, W.norm_attn_b, eps);
    y = rel_pos_mha_graph(ctx, xn, pos_emb, pos_proj, att_mask, W, H, HD, T, attn);
    x = ggml_add(ctx, residual, y);

    residual = x;
    xn = layer_norm_affine(ctx, x, W.norm_conv_w, W.norm_conv_b, eps);
    y = conformer_conv_graph(ctx, xn, W, d_model, T, conv_kernel, dw_lowering, glu_fused,
                             conv_norm_type, conv_causal, eps, valid_frames);
    x = ggml_add(ctx, residual, y);

    residual = x;
    y = conformer_ff_graph(ctx, x,
                           W.norm_ff2_w, W.norm_ff2_b,
                           W.ff2_l1_w,   W.ff2_l1_b,
                           W.ff2_l2_w,   W.ff2_l2_b, eps, ff_headroom);
    y = ggml_scale(ctx, y, 0.5f);
    x = ggml_add(ctx, residual, y);

    x = layer_norm_affine(ctx, x, W.norm_out_w, W.norm_out_b, eps);
    return x;
}

}

// Standalone subsampling graph (run_subsampling / measure_subsampling_compute):
// tensor metadata + cgraph for one L0-mel-frame pre-encode forward. Both
// callers build through this so the fit projection prices exactly the graph
// the per-chunk streaming path executes. The caller owns `ctx` (RAII below).
namespace {
struct SubsamplingGraphBuild {
    ggml_context * ctx     = nullptr;
    ggml_cgraph  * gf      = nullptr;
    ggml_tensor  * mel_in  = nullptr;
    ggml_tensor  * out     = nullptr;
    ~SubsamplingGraphBuild() { if (ctx) ggml_free(ctx); }
};
}  // namespace

static int build_subsampling_graph_ctx(ParakeetCtcModel & model,
                                       int L0, int n_mels,
                                       int mel_valid, int L3,
                                       SubsamplingGraphBuild & g) {
    const int C_sub   = model.encoder_cfg.subsampling_channels;
    const int d_model = model.encoder_cfg.d_model;

    const size_t overhead = ggml_tensor_overhead() * (GGML_DEFAULT_GRAPH_SIZE + 64)
                          + ggml_graph_overhead();
    ggml_init_params gp = { overhead, nullptr, /*no_alloc=*/ true };
    g.ctx = ggml_init(gp);
    if (!g.ctx) return -2;

    g.mel_in  = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, n_mels, L0, 1, 1);
    ggml_set_name(g.mel_in, "mel_in");
    ggml_set_input(g.mel_in);

    g.out = subsampling_graph(g.ctx, g.mel_in, model.subsampling, C_sub, d_model,
                              mel_valid, L3,
                              model.encoder_cfg.causal_downsampling,
                              model.impl->dw_direct_planar);
    ggml_set_name(g.out, "sub_out");
    ggml_set_output(g.out);

    g.gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, g.out);
    return 0;
}

int run_subsampling(ParakeetCtcModel   & model,
                    const float        * mel,
                    int                  n_mel_frames,
                    int                  n_mels,
                    std::vector<float> & out_feats,
                    int                & out_n_frames) {
    if (!model.impl || !model.impl->backend_active || !model.impl->sched) return -1;

    int mel_valid = 0;
    for (int t = 0; t < n_mel_frames; ++t) {
        bool nonzero = false;
        for (int m = 0; m < n_mels; ++m) {
            if (mel[(size_t) t * n_mels + m] != 0.0f) { nonzero = true; break; }
        }
        if (nonzero) mel_valid = t + 1;
    }
    if (mel_valid == 0) mel_valid = n_mel_frames;

    const bool causal_ds = model.encoder_cfg.causal_downsampling;
    auto sub_out_len = [&](int Lin) {
        return causal_ds ? (Lin / 2 + 1) : _conv_out_len(Lin, 3, 2, 1);
    };

    const int L0 = n_mel_frames;
    const int L3 = sub_out_len(sub_out_len(sub_out_len(L0)));

    // Build through the shared builder so the fit projection
    // (measure_subsampling_compute) prices exactly this graph.
    SubsamplingGraphBuild g;
    if (int rc = build_subsampling_graph_ctx(model, L0, n_mels, mel_valid, L3, g);
        rc != 0) {
        return rc;
    }

    // Reset at the HEAD (the previous run already downloaded its outputs to host);
    // the shared sched owns allocation. Never reset at the tail.
    ggml_backend_sched_reset(model.impl->sched);
    if (!ggml_backend_sched_alloc_graph(model.impl->sched, g.gf)) {
        return -3;
    }

    ggml_backend_tensor_set(g.mel_in, mel, 0, (size_t) n_mels * L0 * sizeof(float));

    if (ggml_backend_sched_graph_compute(model.impl->sched, g.gf) != GGML_STATUS_SUCCESS) {
        return -4;
    }

    const int H_out = (int) g.out->ne[1];
    const int W_out = (int) g.out->ne[0];
    out_feats.resize((size_t) W_out * H_out);
    ggml_backend_tensor_get(g.out, out_feats.data(), 0, out_feats.size() * sizeof(float));
    out_n_frames = H_out;

    return 0;
}

int measure_subsampling_compute(ParakeetCtcModel & model,
                                int                n_mel_frames,
                                int                n_mels,
                                size_t           & out_active_bytes,
                                size_t           & out_host_input_bytes) {
    out_active_bytes     = 0;
    out_host_input_bytes = 0;
    if (!model.impl || !model.impl->backend_active) return -1;
    if (n_mel_frames <= 0 || n_mels <= 0) return -1;

    const bool causal_ds = model.encoder_cfg.causal_downsampling;
    auto sub_out_len = [&](int Lin) {
        return causal_ds ? (Lin / 2 + 1) : _conv_out_len(Lin, 3, 2, 1);
    };
    const int L0 = n_mel_frames;
    const int L3 = sub_out_len(sub_out_len(sub_out_len(L0)));

    // Worst case for the projection: every mel frame valid.
    SubsamplingGraphBuild g;
    if (int rc = build_subsampling_graph_ctx(model, L0, n_mels, /*mel_valid=*/L0, L3, g);
        rc != 0) {
        return rc;
    }

    // Price with a plain gallocr on the active backend's default buffer type.
    // The real per-chunk path allocates through the shared scheduler, but the
    // subsampling ops are fully supported on every active backend -- the
    // offline encoder graph embeds the same ops and runs single-split (1
    // split / 0 copies) everywhere -- so the scheduler's reservation is one
    // buffer on the active backend with the same layout the gallocr
    // computes; test-parakeet-fit-params asserts that byte equality against the
    // scheduler buffer a real streaming session reserves. Pricing through
    // ggml_backend_sched_reserve_size itself is not possible on a
    // metadata-only model: its externally-marked weights carry no backend
    // buffer, the scheduler's leaf resolution leaves them unassigned
    // (GGML_ASSERT(buffer_id >= 0) inside the reserve), and explicit
    // ggml_backend_sched_set_tensor_backend pins do not survive the reset at
    // the head of reserve_size.
    ggml_gallocr_t pricer = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(model.impl->backend_active));
    if (!pricer) {
        return -3;
    }
    ggml_gallocr_reserve_n_size(pricer, g.gf, nullptr, nullptr, &out_active_bytes);
    ggml_gallocr_free(pricer);

    // On a GPU backend the scheduler assigns graph INPUTS to the CPU (last)
    // backend and copies them into the device split, so the originals occupy
    // a second, CPU-side scheduler buffer the single-buft measure above does
    // not see. Size that host portion explicitly (per-tensor, padded to the
    // CPU buffer alignment like the scheduler's gallocr does); zero on
    // CPU-only runs where the originals ARE the measured allocations.
    if (!backend_is_cpu(model.impl->backend_active) && model.impl->backend_cpu) {
        ggml_backend_buffer_type_t cpu_buft =
            ggml_backend_get_default_buffer_type(model.impl->backend_cpu);
        const size_t align = ggml_backend_buft_get_alignment(cpu_buft);
        out_host_input_bytes +=
            GGML_PAD(ggml_backend_buft_get_alloc_size(cpu_buft, g.mel_in), align);
    }
    return 0;
}

// Capture every compute node's source pointers in their pristine, just-built
// state, keyed by node pointer (see EncoderGraph::src_backup). Called once after
// the graph is constructed, before it is ever handed to the scheduler.
static void snapshot_encoder_graph_srcs(EncoderGraph & g) {
    g.src_backup.clear();
    if (!g.cgraph) return;
    const int n_nodes = ggml_graph_n_nodes(g.cgraph);
    g.src_backup.reserve((size_t) n_nodes);
    for (int i = 0; i < n_nodes; ++i) {
        ggml_tensor * node = ggml_graph_node(g.cgraph, i);
        std::array<ggml_tensor *, GGML_MAX_SRC> srcs;
        for (int j = 0; j < GGML_MAX_SRC; ++j) srcs[j] = node->src[j];
        g.src_backup.emplace_back(node, srcs);
    }
}

// Restore the source pointers captured by snapshot_encoder_graph_srcs. The
// scheduler rewrites node->src[j] in place when a per-op CPU fallback inserts a
// cross-backend copy; that copy lives in the scheduler's per-run context, which
// is freed before the next allocation. Restoring at the head of each run (before
// allocation) returns the cached graph to its pristine topology so the run starts
// clean. Keyed by node pointer, so it is unaffected by backends that reorder
// cgraph->nodes[]. The restore only writes the saved pointers and never reads the
// stale ones, so it is safe regardless of whether the prior copies were freed.
static void restore_encoder_graph_srcs(EncoderGraph & g) {
    for (auto & entry : g.src_backup) {
        ggml_tensor * node = entry.first;
        for (int j = 0; j < GGML_MAX_SRC; ++j) node->src[j] = entry.second[j];
    }
}

bool pos_proj_cache_disabled() {
    const char * v = std::getenv("PARAKEET_POS_PROJ_CACHE");
    return v && v[0] == '0';
}

size_t pos_proj_cache_bytes(const ParakeetCtcModel & model, int n_layers, int T) {
    const EncoderConfig & enc = model.encoder_cfg;
    const size_t bytes = (size_t) n_layers * enc.head_dim * (2 * (size_t) T - 1) * enc.n_heads * sizeof(float);
    bool affine_position = false;
    for (const auto & block : model.blocks) {
        affine_position |= model.impl->affine_weights.count(block.attn_pos_w) != 0;
    }
    if (n_layers <= 0 || affine_position || pos_proj_cache_disabled() ||
        bytes > ParakeetCtcModel::Impl::k_pos_proj_cache_max_bytes) {
        return 0;
    }
    return bytes;
}

// One layer's projection (attn_pos_w x pe) copied into `dst`, permuted to (HD, 2T-1, H) when asked.
ggml_tensor * pos_proj_node(ggml_context * ctx, const BlockWeights & W, ggml_tensor * pe,
                            ggml_tensor * dst, int H, int HD, bool permuted) {
    ggml_tensor * p = affine_mul_mat(ctx, W.attn_pos_w, pe);
    p = ggml_reshape_3d(ctx, p, HD, H, pe->ne[1]);
    return ggml_cpy(ctx, permuted ? ggml_permute(ctx, p, 0, 2, 1, 3) : p, dst);
}

bool plan_pos_proj_tensors(EncoderGraph & g, int n_layers, int H, int HD, int L, bool permuted) {
    ggml_init_params params = { ggml_tensor_overhead() * (size_t) n_layers, nullptr, true };
    g.pos_ctx = ggml_init(params);
    if (!g.pos_ctx) return false;
    for (int i = 0; i < n_layers; ++i) {
        ggml_tensor * t = permuted ? ggml_new_tensor_3d(g.pos_ctx, GGML_TYPE_F32, HD, L, H)
                                   : ggml_new_tensor_3d(g.pos_ctx, GGML_TYPE_F32, HD, H, L);
        ggml_format_name(t, "pos_proj_%d", i);
        g.pos_proj.push_back(t);
    }
    return true;
}

bool alloc_pos_proj_tensors(EncoderGraph & g, ggml_backend_t backend, int n_layers, int H, int HD, int L,
                            bool permuted) {
    if (!plan_pos_proj_tensors(g, n_layers, H, HD, L, permuted)) return false;
    g.pos_buf = ggml_backend_alloc_ctx_tensors(g.pos_ctx, backend);
    return g.pos_buf != nullptr;
}

bool compute_pos_proj_cache(const ParakeetCtcModel & model, EncoderGraph & g, ggml_backend_t backend,
                            int n_layers, int H, int HD, int L, bool permuted) {
    const size_t n_nodes = (size_t) n_layers * 4 + 2;
    ggml_init_params params = { ggml_tensor_overhead() * (n_nodes + 1) + ggml_graph_overhead_custom(n_nodes, false),
                                nullptr, true };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;
    ggml_tensor * pe = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, model.encoder_cfg.d_model, L);
    ggml_set_input(pe);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, n_nodes, false);
    for (int i = 0; i < n_layers; ++i) {
        ggml_build_forward_expand(gf, pos_proj_node(ctx, model.blocks[i], pe, g.pos_proj[i], H, HD, permuted));
    }
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    bool ok = ga && ggml_gallocr_alloc_graph(ga, gf);
    if (ok) {
        ggml_backend_tensor_set(pe, g.pe_host.data(), 0, g.pe_host.size() * sizeof(float));
        ok = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    }
    if (ga) ggml_gallocr_free(ga);
    ggml_free(ctx);
    return ok;
}

// The positional projections depend only on T, so a cached graph computes them once and
// the attention blocks read them from g.pos_buf; in-graph projection stays when over the cap.
// `measure_only` plans the tensors without a buffer so a sized graph prices them like a built one.
void build_pos_proj_cache(const ParakeetCtcModel & model, EncoderGraph & g, ggml_backend_t backend,
                          int n_layers, int T, bool measure_only) {
    if (pos_proj_cache_bytes(model, n_layers, T) == 0) return;
    const EncoderConfig & enc = model.encoder_cfg;
    const int L = 2 * T - 1;
    const bool permuted = !model.impl->attn.flash_attn;
    if (measure_only) {
        if (!plan_pos_proj_tensors(g, n_layers, enc.n_heads, enc.head_dim, L, permuted)) g.release_pos_proj();
        return;
    }
    if (!alloc_pos_proj_tensors(g, backend, n_layers, enc.n_heads, enc.head_dim, L, permuted) ||
        !compute_pos_proj_cache(model, g, backend, n_layers, enc.n_heads, enc.head_dim, L, permuted)) {
        g.release_pos_proj();
    }
}

constexpr float kMaskedAttentionScore = -1.0e30f;

void mask_attention_padding(std::vector<float> & mask, int frames, int valid_frames) {
    if (valid_frames >= frames) return;
    if (mask.empty()) mask.assign((size_t) frames * frames, 0.0f);
    for (int query = 0; query < frames; ++query) {
        auto row = mask.begin() + (size_t) query * frames;
        std::fill(row + valid_frames, row + frames, kMaskedAttentionScore);
    }
}

// `measure_compute` (fit projection): when non-null the graph is built
// normally but its compute buffer is only *sized* into `*measure_compute`
// instead of reserved; no compute buffer is allocated and
// the returned graph must be discarded, never run. Requires the model's
// weight tensors to be marked externally-allocated (metadata-only load), or
// the measurement would count the weights as graph-owned leafs.
static int build_encoder_graph_cached(const ParakeetCtcModel & model,
                                      EncoderGraph & g,
                                      int n_mel_frames, int mel_valid, int n_mels,
                                      int n_run_layers_override,
                                      bool all_valid,
                                      ggml_backend_t backend,
                                      bool bypass_pre_encode = false,
                                      int  T_enc_override   = 0,
                                      size_t * measure_compute = nullptr) {
    // Every ggml encoder path builds its graph here, so this is where a model
    // whose encoder weights were never loaded is refused instead of reading
    // through unallocated tensors.
    if (model.impl->coreml_owns_encoder) {
        PARAKEET_LOG_ERROR("parakeet: the ggml encoder was requested but Core ML owns the "
                           "encoder and its weights were not loaded (bypass=%d, layers=%d)\n",
                           (int) bypass_pre_encode, n_run_layers_override);
        return kEncoderWeightsNotLoaded;
    }

    const EncoderConfig & enc = model.encoder_cfg;
    const int C_sub = enc.subsampling_channels;
    const int d_model = enc.d_model;
    const int H  = enc.n_heads;
    const int HD = enc.head_dim;
    const int N_LAYERS = enc.n_layers;
    const int conv_kernel = enc.conv_kernel;
    const float eps = enc.layer_norm_eps;

    const DwConvLowering dw_lowering = select_dw_lowering(model, backend);

    auto sub_out_len = [&](int Lin) {
        return enc.causal_downsampling ? (Lin / 2 + 1) : _conv_out_len(Lin, 3, 2, 1);
    };

    int L0 = 0, L1 = 0, L2 = 0, L3 = 0, T = 0;
    if (bypass_pre_encode) {
        T = T_enc_override;
    } else {
        L0 = n_mel_frames;
        L1 = sub_out_len(L0);
        L2 = sub_out_len(L1);
        L3 = sub_out_len(L2);
        T  = L3;
    }

    // Standard offline TDT follows NeMo's padding masks in attention and convolution.
    const bool mask_tdt_padding = !bypass_pre_encode &&
        model.model_type == ParakeetModelType::TDT &&
        model.model_variant != kTranscribePsyEuVariant;
    const int valid_frames = mask_tdt_padding
        ? sub_out_len(sub_out_len(sub_out_len(mel_valid))) : T;

    g.pe_host = compute_rel_pos_encoding(T, d_model);

    // Build the chunked-limited attention mask host-side once per graph.
    // For an `att_context_size = [left, right]` with `att_context_style =
    // chunked_limited`, frames are grouped into chunks of `chunk_size =
    // right + 1`. A query frame at position i (in chunk c = i / chunk_size)
    // attends to keys in [c*chunk_size - left, (c+1)*chunk_size - 1].
    // Mask is `0.0f` for visible positions and `-INFINITY` for masked.
    //
    // The mask is shape (T, T) row-major in NumPy / (T, T, 1, 1) in ggml
    // (ne[0]=T_k, ne[1]=T_q). Stored as f32; ggml_soft_max_ext accepts f32
    // mask tensors and broadcasts over the head axis.
    const bool use_chunked_mask = enc.att_chunked_limited &&
                                  enc.att_context_left  >= 0 &&
                                  enc.att_context_right >= 0;
    if (use_chunked_mask) {
        const int left  = enc.att_context_left;
        const int right = enc.att_context_right;
        const int chunk = right + 1;
        // A finite score masks keys without relying on infinity under Apple fast-math.
        g.att_mask_host.assign((size_t) T * T, kMaskedAttentionScore);
        for (int i = 0; i < T; ++i) {
            const int c          = i / chunk;
            const int win_start  = c * chunk - left;
            const int win_end    = (c + 1) * chunk - 1;
            const int j0 = std::max(0, win_start);
            const int j1 = std::min(T - 1, win_end);
            float * row = g.att_mask_host.data() + (size_t) i * T;
            for (int j = j0; j <= j1; ++j) row[j] = 0.0f;
        }
    } else {
        g.att_mask_host.clear();
    }
    mask_attention_padding(g.att_mask_host, T, valid_frames);

    const size_t graph_slots = GGML_DEFAULT_GRAPH_SIZE * 16;
    const size_t overhead = ggml_tensor_overhead() * graph_slots
                          + ggml_graph_overhead_custom(graph_slots, false);
    ggml_init_params gp = { overhead, nullptr, /*no_alloc=*/ true };
    g.graph_ctx = ggml_init(gp);
    if (!g.graph_ctx) return -2;
    ggml_context * gctx = g.graph_ctx;
    g.cgraph = ggml_new_graph_custom(gctx, graph_slots, false);
    const bool fused_affine = affine_fused_requested() && !model.impl->affine_weights.empty();
    if (fused_affine && model_active_backend_name(model).rfind("MTL", 0) != 0)
        throw std::runtime_error("fused affine requires the explicit Metal backend");
    PackedAffineGraphScope affine_scope(gctx, g.cgraph, model.impl->affine_weights, fused_affine);

    if (bypass_pre_encode) {
        // Pre-subsampled input fed directly: (d_model, T). Subsampling block is
        // skipped, no mel/masks. Used by the v2.1 streaming AOSC path where the
        // speaker cache + FIFO + chunk are concatenated in pre-encode space and
        // re-contextualised by the conformer layers in a single forward.
        g.mel_in  = nullptr;
        g.pre_encode_in = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, d_model, T);
        ggml_set_name(g.pre_encode_in, "pre_encode_in");
        ggml_set_input(g.pre_encode_in);
    } else {
        g.mel_in  = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, n_mels, L0, 1, 1);
        ggml_set_name(g.mel_in,  "mel_in");
        ggml_set_input(g.mel_in);
        g.pre_encode_in = nullptr;
    }
    g.pe_in   = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, d_model, 2 * T - 1);
    if (!g.att_mask_host.empty()) {
        g.att_mask = ggml_new_tensor_4d(gctx, GGML_TYPE_F32, T, T, 1, 1);
        ggml_set_name(g.att_mask, "att_mask");
        ggml_set_input(g.att_mask);
    } else {
        g.att_mask = nullptr;
    }
    ggml_set_name(g.pe_in,   "pe_in");
    ggml_set_input(g.pe_in);

    ggml_tensor * x;
    if (bypass_pre_encode) {
        x = g.pre_encode_in;
        g.sub_out_node = nullptr;
    } else {
        x = subsampling_graph(gctx, g.mel_in, model.subsampling, C_sub, d_model,
                              mel_valid, L3, enc.causal_downsampling, model.impl->dw_direct_planar);
        g.sub_out_node = x;
        ggml_set_name(g.sub_out_node, "subsampling_out");
        ggml_set_output(g.sub_out_node);
    }

    // NB: with PARAKEET_F16_ACTIVATIONS we intentionally keep the encoder
    // residual stream in F32. Hexagon's quantized MM kernels only carry a
    // `quantize_f32_q8_0_*` activation quantizer (matmul-ops.c:843) — feeding
    // them F16 activations fails with DSP NO_SUPPORT on every FF/Conv/attn
    // projection matmul. Following whisper.cpp's convention the flag only
    // pads+casts to F16 at the batched attention matmul boundary in
    // `rel_pos_mha_unfused_graph`, which is the site the HMX F16-batched
    // kernel (`hmx_mm_f16_f32_batched`) can accept (src0=F16, src1=F32).

    if (enc.xscaling) {
        x = ggml_scale(gctx, x, std::sqrt((float) d_model));
    }

    int n_run_layers = n_run_layers_override;
    if (n_run_layers < 0) {
        n_run_layers = std::getenv("PARAKEET_MAX_LAYERS")
                     ? std::atoi(std::getenv("PARAKEET_MAX_LAYERS"))
                     : N_LAYERS;
    }
    if (n_run_layers > N_LAYERS) n_run_layers = N_LAYERS;
    if (n_run_layers < 0)        n_run_layers = 0;
    g.n_run_layers = n_run_layers;
    g.eu_use_adapters = model.eu_use_adapters;

    build_pos_proj_cache(model, g, backend, n_run_layers, T, /*measure_only=*/measure_compute != nullptr);
    const float ff_headroom = matmul_output_headroom(backend);
    auto layer_pos_proj = [&](int i) -> ggml_tensor * {
        return i < (int) g.pos_proj.size() ? g.pos_proj[i] : nullptr;
    };

    for (int i = 0; i < n_run_layers; ++i) {
        if (i == 0) {
            const BlockWeights & W = model.blocks[0];
            ggml_tensor * residual = x;
            ggml_tensor * y = conformer_ff_graph(gctx, x,
                                                 W.norm_ff1_w, W.norm_ff1_b,
                                                 W.ff1_l1_w,   W.ff1_l1_b,
                                                 W.ff1_l2_w,   W.ff1_l2_b, eps, ff_headroom);
            y = ggml_scale(gctx, y, 0.5f);
            x = ggml_add(gctx, residual, y);
            g.post_ff1_0_node = x;
            ggml_set_name(g.post_ff1_0_node, "block_0_post_ff1");
            ggml_set_output(g.post_ff1_0_node);

            residual = x;
            ggml_tensor * xn = layer_norm_affine(gctx, x, W.norm_attn_w, W.norm_attn_b, eps);
            AttnCapture blk0_cap;
            y = rel_pos_mha_graph(gctx, xn, g.pe_in, layer_pos_proj(0), g.att_mask, W, H, HD, T, model.impl->attn, &blk0_cap);
            // Diagnostic: expose block 0 attention intermediates
            // so a parity test can localize where HTP0 diverges from CPU.
            // Guarded by set_name/set_output so the scheduler keeps the
            // tensors materialised on the primary backend for host copy.
            if (blk0_cap.qkv) {
                g.block_0_attn_qkv_node = blk0_cap.qkv;
                ggml_set_name(g.block_0_attn_qkv_node, "block_0_attn_qkv");
                ggml_set_output(g.block_0_attn_qkv_node);
            }
            if (blk0_cap.k_raw) {
                g.block_0_attn_k_raw_node = blk0_cap.k_raw;
                ggml_set_name(g.block_0_attn_k_raw_node, "block_0_attn_k_raw");
                ggml_set_output(g.block_0_attn_k_raw_node);
            }
            if (blk0_cap.v_raw) {
                g.block_0_attn_v_raw_node = blk0_cap.v_raw;
                ggml_set_name(g.block_0_attn_v_raw_node, "block_0_attn_v_raw");
                ggml_set_output(g.block_0_attn_v_raw_node);
            }
            if (blk0_cap.q_perm) {
                g.block_0_attn_q_perm_node = blk0_cap.q_perm;
                ggml_set_name(g.block_0_attn_q_perm_node, "block_0_attn_q_perm");
                ggml_set_output(g.block_0_attn_q_perm_node);
            }
            if (blk0_cap.k_perm) {
                g.block_0_attn_k_perm_node = blk0_cap.k_perm;
                ggml_set_name(g.block_0_attn_k_perm_node, "block_0_attn_k_perm");
                ggml_set_output(g.block_0_attn_k_perm_node);
            }
            if (blk0_cap.q_u) {
                g.block_0_attn_q_u_node = blk0_cap.q_u;
                ggml_set_name(g.block_0_attn_q_u_node, "block_0_attn_q_u");
                ggml_set_output(g.block_0_attn_q_u_node);
            }
            if (blk0_cap.ac) {
                g.block_0_attn_ac_node = blk0_cap.ac;
                ggml_set_name(g.block_0_attn_ac_node, "block_0_attn_ac");
                ggml_set_output(g.block_0_attn_ac_node);
            }
            if (blk0_cap.bd) {
                g.block_0_attn_bd_node = blk0_cap.bd;
                ggml_set_name(g.block_0_attn_bd_node, "block_0_attn_bd");
                ggml_set_output(g.block_0_attn_bd_node);
            }
            if (blk0_cap.scores) {
                g.block_0_attn_scores_node = blk0_cap.scores;
                ggml_set_name(g.block_0_attn_scores_node, "block_0_attn_scores");
                ggml_set_output(g.block_0_attn_scores_node);
            }
            if (blk0_cap.softmax) {
                g.block_0_attn_softmax_node = blk0_cap.softmax;
                ggml_set_name(g.block_0_attn_softmax_node, "block_0_attn_softmax");
                ggml_set_output(g.block_0_attn_softmax_node);
            }
            if (blk0_cap.out) {
                g.block_0_attn_out_node = blk0_cap.out;
                ggml_set_name(g.block_0_attn_out_node, "block_0_attn_out");
                ggml_set_output(g.block_0_attn_out_node);
            }
            x = ggml_add(gctx, residual, y);
            g.post_attn_0_node = x;
            ggml_set_name(g.post_attn_0_node, "block_0_post_attn");
            ggml_set_output(g.post_attn_0_node);

            residual = x;
            xn = layer_norm_affine(gctx, x, W.norm_conv_w, W.norm_conv_b, eps);
            y = conformer_conv_graph(gctx, xn, W, d_model, T, conv_kernel, dw_lowering,
                                     model.impl->glu_fused, enc.conv_norm_type, enc.conv_causal, eps,
                                     valid_frames);
            x = ggml_add(gctx, residual, y);
            g.post_conv_0_node = x;
            ggml_set_name(g.post_conv_0_node, "block_0_post_conv");
            ggml_set_output(g.post_conv_0_node);

            residual = x;
            y = conformer_ff_graph(gctx, x,
                                   W.norm_ff2_w, W.norm_ff2_b,
                                   W.ff2_l1_w,   W.ff2_l1_b,
                                   W.ff2_l2_w,   W.ff2_l2_b, eps, ff_headroom);
            y = ggml_scale(gctx, y, 0.5f);
            x = ggml_add(gctx, residual, y);
            g.post_ff2_0_node = x;
            ggml_set_name(g.post_ff2_0_node, "block_0_post_ff2");
            ggml_set_output(g.post_ff2_0_node);

            x = layer_norm_affine(gctx, x, W.norm_out_w, W.norm_out_b, eps);

            g.block_0_out_node = x;
            ggml_set_name(g.block_0_out_node, "block_0_out");
            ggml_set_output(g.block_0_out_node);
        } else {
            x = conformer_block_graph(gctx, x, g.pe_in, layer_pos_proj(i), g.att_mask, model.blocks[i],
                                      d_model, H, HD, T, conv_kernel, eps,
                                      dw_lowering, model.impl->attn, model.impl->glu_fused,
                                      enc.conv_norm_type, enc.conv_causal, ff_headroom, valid_frames);
        }
        if (model.eu_use_adapters) {
            for (const auto & a : model.eu_adapters) {
                if (a.layer != i) continue;
                ggml_tensor * y = maybe_add_bias(gctx, affine_mul_mat(gctx, a.down_w, x), a.down_b);
                y = ggml_gelu_erf(gctx, y);
                y = maybe_add_bias(gctx, affine_mul_mat(gctx, a.up_w, y), a.up_b);
                x = ggml_add(gctx, x, y);
                break;
            }
        }
        if (i == n_run_layers - 1) {
            g.block_last_out_node = x;
            ggml_set_name(g.block_last_out_node, "block_last_out");
            ggml_set_output(g.block_last_out_node);
        }
    }

    g.encoder_out_node = x;
    ggml_set_name(g.encoder_out_node, "encoder_out");
    ggml_set_output(g.encoder_out_node);

    if (model.model_type == ParakeetModelType::CTC && model.ctc.w && model.ctc.b) {
        g.logits_node = ggml_add(gctx, affine_mul_mat(gctx, model.ctc.w, x), model.ctc.b);
        ggml_set_name(g.logits_node, "logits");
        ggml_set_output(g.logits_node);
    } else {
        g.logits_node = nullptr;
    }

    if (g.sub_out_node)        ggml_build_forward_expand(g.cgraph, g.sub_out_node);
    if (g.post_ff1_0_node)     ggml_build_forward_expand(g.cgraph, g.post_ff1_0_node);
    if (g.block_0_attn_qkv_node)     ggml_build_forward_expand(g.cgraph, g.block_0_attn_qkv_node);
    if (g.block_0_attn_k_raw_node)   ggml_build_forward_expand(g.cgraph, g.block_0_attn_k_raw_node);
    if (g.block_0_attn_v_raw_node)   ggml_build_forward_expand(g.cgraph, g.block_0_attn_v_raw_node);
    if (g.block_0_attn_q_perm_node)  ggml_build_forward_expand(g.cgraph, g.block_0_attn_q_perm_node);
    if (g.block_0_attn_k_perm_node)  ggml_build_forward_expand(g.cgraph, g.block_0_attn_k_perm_node);
    if (g.block_0_attn_q_u_node)     ggml_build_forward_expand(g.cgraph, g.block_0_attn_q_u_node);
    if (g.block_0_attn_ac_node)      ggml_build_forward_expand(g.cgraph, g.block_0_attn_ac_node);
    if (g.block_0_attn_bd_node)      ggml_build_forward_expand(g.cgraph, g.block_0_attn_bd_node);
    if (g.block_0_attn_scores_node)  ggml_build_forward_expand(g.cgraph, g.block_0_attn_scores_node);
    if (g.block_0_attn_softmax_node) ggml_build_forward_expand(g.cgraph, g.block_0_attn_softmax_node);
    if (g.block_0_attn_out_node)     ggml_build_forward_expand(g.cgraph, g.block_0_attn_out_node);
    if (g.post_attn_0_node)    ggml_build_forward_expand(g.cgraph, g.post_attn_0_node);
    if (g.post_conv_0_node)    ggml_build_forward_expand(g.cgraph, g.post_conv_0_node);
    if (g.post_ff2_0_node)     ggml_build_forward_expand(g.cgraph, g.post_ff2_0_node);
    if (g.block_0_out_node)    ggml_build_forward_expand(g.cgraph, g.block_0_out_node);
    if (g.block_last_out_node) ggml_build_forward_expand(g.cgraph, g.block_last_out_node);
    ggml_build_forward_expand(g.cgraph, g.encoder_out_node);
    if (g.logits_node) ggml_build_forward_expand(g.cgraph, g.logits_node);

    if (!model.impl->affine_weights.empty()) {
        for (int i = 0; i < ggml_graph_n_nodes(g.cgraph); ++i) {
            auto * node = ggml_graph_node(g.cgraph, i);
            if (!ggml_backend_supports_op(backend, node)) {
                PARAKEET_LOG_ERROR("packed-affine: active backend cannot execute %s; refusing fallback\n",
                                   ggml_op_name(node->op));
                return -4;
            }
        }
    }
    snapshot_encoder_graph_srcs(g);

    if (!g.allocator.initialize(backend, model.impl->backend_cpu, graph_slots, backend_is_xdna(backend)) ||
        !g.allocator.reserve(g.cgraph, measure_compute)) {
        return -3;
    }
    if (measure_compute && g.allocator.scheduler() && g.pos_ctx) {
        const size_t positional = ggml_backend_alloc_ctx_tensors_from_buft_size(
            g.pos_ctx, ggml_backend_get_default_buffer_type(backend));
        if (positional > std::numeric_limits<size_t>::max() - *measure_compute) return -3;
        *measure_compute += positional;
    }

    g.T_mel = bypass_pre_encode ? 0 : n_mel_frames;
    g.T_mel_valid = bypass_pre_encode ? 0 : mel_valid;
    g.T_enc = T;
    g.all_valid = all_valid;
    g.bypass_pre_encode = bypass_pre_encode;
    return 0;
}

// Core ML encoder dispatch (portable: the sidecar call itself goes through the
// unit's EncoderRoute, so builds without Core ML simply never hold a route).
//
// Post-subsampling encoder frame count (T_enc), derived purely from the mel frame
// count via the three stride-2 subsampling convs -- the same recurrence run_encoder
// uses to size the ggml graph, so the Core ML output slab matches frame-for-frame.
static int coreml_encoder_out_frames(const EncoderConfig & enc, int n_mel_frames) {
    auto next = [&](int len) {
        return enc.causal_downsampling ? (len / 2 + 1) : _conv_out_len(len, 3, 2, 1);
    };
    return next(next(next(n_mel_frames)));
}

// CTC/IndicConformer, Unified RNN-T, TDT, and Sortformer v2.1 batch sidecars use the
// full-context contract.
static bool encoder_is_offline(const EncoderConfig & enc) {
    return enc.att_context_left  < 0
        && enc.att_context_right < 0
        && !enc.att_chunked_limited
        && !enc.conv_causal
        && !enc.causal_downsampling;
}

#ifdef PARAKEET_USE_COREML
// ── TranscribePsy EU encoder programs ──────────────────────────────────
// Not on the runtime: run_encoder picks one of these per dispatch and runs it
// directly; a failure falls back to ggml (or is an error for the owner).

// The resolved compute units the EU placement policy reads (null: unset).
static const char * eu_policy_units(const ParakeetCtcModel & model) {
    return model.impl && model.impl->coreml ? loader_units(model.impl->coreml->resolved()) : nullptr;
}

static void clear_eu_precise_main_contexts(ParakeetCtcModel::Impl & impl) {
    parakeet_coreml_free(impl.ctx_coreml_eu_flexible);
    parakeet_coreml_free(impl.ctx_coreml_eu_other_flexible);
    parakeet_coreml_free(impl.ctx_coreml_eu_precise_gpu);
    impl.ctx_coreml_eu_flexible = nullptr;
    impl.ctx_coreml_eu_other_flexible = nullptr;
    impl.ctx_coreml_eu_precise_gpu = nullptr;
    impl.eu_precise_gpu_path.clear();
}

static parakeet_coreml_context * selected_eu_precise_gpu(const ParakeetCtcModel & model, int frames, int valid_frames) {
    auto & impl = *model.impl;
    const char * units = eu_policy_units(model);
#if defined(__APPLE__) && TARGET_OS_IPHONE
    // iOS 18 MPSGraph aborts when static FP32 programs follow the CPU gate or
    // reload across routes. The masked FP32 CPU program preserves every mode.
    constexpr bool kMobilePreciseEncoder = true;
#else
    constexpr bool kMobilePreciseEncoder = false;
#endif
    if (frames <= 0 || impl.eu_gguf_path.empty() ||
        !parakeet::coreml_eu_allows_gpu_precision(units, kMobilePreciseEncoder)) return nullptr;
    constexpr int kPreciseGpuMelCapacities[] = {1500, 6008};
    for (int capacity : kPreciseGpuMelCapacities) {
        if (valid_frames > capacity ||
            coreml_encoder_out_frames(model.encoder_cfg, frames) >
                coreml_encoder_out_frames(model.encoder_cfg, capacity)) continue;
        const auto path = coreml_eu_precise_sidecar_path(impl.eu_gguf_path, model.eu_use_adapters, capacity);
        if (std::find(impl.eu_invalid_gpu_paths.begin(), impl.eu_invalid_gpu_paths.end(), path) !=
            impl.eu_invalid_gpu_paths.end() || !path_is_directory(path)) continue;
        if (impl.ctx_coreml_eu_precise_gpu && impl.eu_precise_gpu_path == path)
            return impl.ctx_coreml_eu_precise_gpu;
        clear_eu_precise_main_contexts(impl);
        impl.ctx_coreml_eu_precise_gpu = load_eu_coreml_encoder(path,
            model.eu_use_adapters ? "other" : "en", model.encoder_cfg.n_layers, units,
            EuCoremlKind::PreciseGpu, capacity);
        if (impl.ctx_coreml_eu_precise_gpu) {
            impl.eu_precise_gpu_path = path;
            return impl.ctx_coreml_eu_precise_gpu;
        }
        impl.eu_invalid_gpu_paths.push_back(path);
    }
    return nullptr;
}

// The TranscribePsy EU program for one dispatch: the gate tap, the fixed
// program of the current language route when the input fits it, else a
// precise (FP32 GPU, then flexible CPU) program, loaded on first use.
static parakeet_coreml_context * selected_coreml_encoder(
        const ParakeetCtcModel & model, int max_layers, int n_mel_frames, int mel_valid, bool prefer_precise_eu) {
    if (!model.impl || model.model_variant != "transcribepsy-250m-eu") return nullptr;
    const char * units = eu_policy_units(model);
    const bool cpu_only = parakeet::coreml_eu_requires_cpu_precision(units);
    if (max_layers == model.eu_gate.layer + 1 && !model.eu_use_adapters) {
        if (!cpu_only && !prefer_precise_eu && model.impl->ctx_coreml_eu_gate)
            return model.impl->ctx_coreml_eu_gate;
        auto & gate = model.impl->ctx_coreml_eu_gate_flexible;
        if (!gate && !model.impl->eu_gate_flexible_path.empty())
            gate = load_eu_coreml_encoder(model.impl->eu_gate_flexible_path, "gate",
                                          model.eu_gate.layer + 1, units, EuCoremlKind::FlexibleCpu);
        return gate;
    }
    if (max_layers >= 0 && max_layers < model.encoder_cfg.n_layers) return nullptr;
    auto * fixed = model.eu_use_adapters ? model.impl->ctx_coreml_eu_other : model.impl->ctx_coreml;
    const int64_t capacity = parakeet_coreml_fixed_mel_frames(fixed, model.mel_cfg.n_mels);
    if (!cpu_only && !prefer_precise_eu && fixed && capacity >= n_mel_frames) return fixed;
    if (auto * gpu = selected_eu_precise_gpu(model, n_mel_frames, mel_valid)) return gpu;
    auto & flexible = model.eu_use_adapters
        ? model.impl->ctx_coreml_eu_other_flexible : model.impl->ctx_coreml_eu_flexible;
    const auto & path = model.eu_use_adapters
        ? model.impl->eu_other_flexible_path : model.impl->eu_flexible_path;
    if (!flexible && !path.empty()) {
        parakeet_coreml_free(model.impl->ctx_coreml_eu_precise_gpu);
        model.impl->ctx_coreml_eu_precise_gpu = nullptr;
        model.impl->eu_precise_gpu_path.clear();
        // Keep at most one large precise program resident per Engine.
        auto & previous = model.eu_use_adapters
            ? model.impl->ctx_coreml_eu_flexible : model.impl->ctx_coreml_eu_other_flexible;
        parakeet_coreml_free(previous);
        previous = nullptr;
        flexible = load_eu_coreml_encoder(path, model.eu_use_adapters ? "other" : "en",
                                          model.encoder_cfg.n_layers, units, EuCoremlKind::FlexibleCpu);
    }
    return flexible;
}

// Whether the EU program `ctx` can take this dispatch. A program that
// preserves encoder padding takes any input whose encoder frames fit it.
static bool eu_coreml_encoder_takes(const ParakeetCtcModel & model, parakeet_coreml_context * ctx,
                                    int n_mel_frames, bool capture_intermediates) {
    if (!ctx || capture_intermediates) return false;  // per-stage parity harnesses stay on ggml
    if (model.model_type != ParakeetModelType::TDT) return false;
    const int64_t capacity = parakeet_coreml_fixed_mel_frames(ctx, model.mel_cfg.n_mels);
    const bool padding_fits = parakeet_coreml_preserves_encoder_padding(ctx) &&
        coreml_encoder_out_frames(model.encoder_cfg, n_mel_frames) <=
            coreml_encoder_out_frames(model.encoder_cfg, capacity);
    return encoder_is_offline(model.encoder_cfg) &&
        (capacity <= 0 || n_mel_frames <= capacity || padding_fits);
}
#endif  // PARAKEET_USE_COREML

static bool coreml_encoder_takes_shape(const ParakeetCtcModel & model,
                                       const EncoderRoute & route,
                                       int n_mel_frames,
                                       bool all_valid,
                                       bool capture_intermediates,
                                       bool allow_coreml_padded) {
    // TranscribePsy EU picks its program per dispatch (eu_coreml_encoder_takes).
    if (model.model_variant == "transcribepsy-250m-eu") return false;
    if (!model.impl || !route.usable()) return false;
    if (model.model_type != ParakeetModelType::CTC &&
        model.model_type != ParakeetModelType::RNNT &&
        model.model_type != ParakeetModelType::TDT &&
        model.model_type != ParakeetModelType::EOU &&
        model.model_type != ParakeetModelType::NEMOTRON &&
        model.model_type != ParakeetModelType::SORTFORMER) return false;
    if (!all_valid && !allow_coreml_padded) return false;
    if (capture_intermediates) return false;  // per-stage parity harnesses stay on ggml
    // CTC, RNN-T and TDT: the capacity is checked by the caller
    // (coreml_encoder_eligibility), so a longer input is "cannot serve".
    if (model.model_type == ParakeetModelType::CTC) {
        return encoder_is_offline(model.encoder_cfg) &&
               model.encoder_cfg.conv_norm_type == ConvNormType::BatchNorm;
    }
    if (model.model_type == ParakeetModelType::RNNT) {
        return encoder_is_offline(model.encoder_cfg) &&
               model.encoder_cfg.conv_norm_type == ConvNormType::BatchNorm;
    }
    if (model.model_type == ParakeetModelType::TDT) {
        return encoder_is_offline(model.encoder_cfg);
    }
    if (model.model_type == ParakeetModelType::SORTFORMER) {
        const int fixed_frames = route.geometry().fixed_mel_frames;
        return model.model_variant == "sortformer-streaming-v2.1-aosc" &&
               fixed_frames > 0 &&
               n_mel_frames == fixed_frames &&
               encoder_is_offline(model.encoder_cfg);
    }

    // EOU and Nemotron graphs bake causal/chunked attention geometry and shape
    // into the compiled program. Never pad a shorter invocation: added future
    // encoder frames can change token emission (and the EOU boundary).
    const EncoderConfig & enc = model.encoder_cfg;
    const int fixed_frames = route.geometry().fixed_mel_frames;
    return fixed_frames > 0
        && n_mel_frames == fixed_frames
        && enc.causal_downsampling
        && enc.conv_causal
        && enc.conv_norm_type == ConvNormType::LayerNorm
        && enc.att_chunked_limited
        && enc.att_context_left >= 0
        && enc.att_context_right >= 0;
}

// Where one encoder dispatch may run. A fixed-capacity handle (an offline
// fixed export or the ANE window) cannot serve a longer CTC, RNN-T or TDT
// input: that dispatch is "cannot serve", which runs on ggml and leaves the
// route usable (a capacity miss is not a prediction failure), or throws on a
// strict route. EOU, Nemotron and the AOSC bypass need the exact compiled
// shape, so any other length is simply a ggml dispatch.
static EncoderEligibility coreml_encoder_eligibility(const ParakeetCtcModel & model,
                                                     const EncoderRoute & route,
                                                     int n_mel_frames,
                                                     bool all_valid,
                                                     bool capture_intermediates,
                                                     bool allow_coreml_padded) {
    if (!coreml_encoder_takes_shape(model, route, n_mel_frames, all_valid,
                                    capture_intermediates, allow_coreml_padded)) {
        return EncoderEligibility::ggml;
    }
    const bool capacity_bound = model.model_type == ParakeetModelType::CTC ||
                                model.model_type == ParakeetModelType::RNNT ||
                                model.model_type == ParakeetModelType::TDT;
    if (capacity_bound && !route_fits_capacity(route, n_mel_frames)) {
        return EncoderEligibility::cannot_serve;
    }
    return EncoderEligibility::coreml;
}

void model_coreml_check_stream_window(ParakeetCtcModel & m, int window_mel_frames,
                                      const std::atomic<bool> * cancel) {
#ifdef PARAKEET_USE_COREML
    // Strict streams, and streams on an encoder Core ML owns (no ggml
    // weights to run a window the sidecar cannot take), are checked.
    if (!m.impl || !m.impl->coreml) return;
    const bool owned = m.impl->ctx_coreml != nullptr && m.impl->coreml_owns_encoder;
    if (!owned && !m.impl->coreml->resolved().strict()) return;
    const EncoderRoute route = acquire_encoder_route(m, cancel);
    // The same eligibility the window's run_encoder computes (every frame
    // valid, no capture, no padding).
    const EncoderEligibility eligibility = coreml_encoder_eligibility(
        m, route, window_mel_frames, /*all_valid=*/true, /*capture_intermediates=*/false,
        /*allow_coreml_padded=*/false);
    const std::string reason = strict_stream_window_refusal(route, eligibility, window_mel_frames);
    if (!reason.empty()) route.refuse("Engine::stream_start: " + reason, false);
#else
    (void) m;
    (void) window_mel_frames;
    (void) cancel;
#endif
}

static int run_coreml_ctc_head(ParakeetCtcModel & model,
                               const std::vector<float> & encoder_out,
                               int n_frames,
                               std::vector<float> & logits) {
    if (!model.impl || !model.impl->backend_active || !model.ctc.w || !model.ctc.b ||
        n_frames <= 0 || encoder_out.size() !=
            static_cast<size_t>(n_frames) * model.encoder_cfg.d_model) {
        return -1;
    }

    auto & cached = model.impl->ctc_head_graph;
    if (!cached || cached->n_frames != n_frames) {
        cached = std::make_unique<CtcHeadGraph>();
        CtcHeadGraph & graph = *cached;
        constexpr size_t graph_size = 16;
        const size_t graph_memory =
            ggml_tensor_overhead() * graph_size +
            ggml_graph_overhead_custom(graph_size, false) + 4096;
        ggml_init_params params = {};
        params.mem_size = graph_memory;
        params.no_alloc = true;
        graph.context = ggml_init(params);
        if (!graph.context) return -2;

        graph.input = ggml_new_tensor_2d(
            graph.context, GGML_TYPE_F32, model.encoder_cfg.d_model, n_frames);
        ggml_set_input(graph.input);
        graph.output = ggml_add(
            graph.context,
            affine_mul_mat(graph.context, model.ctc.w, graph.input),
            model.ctc.b);
        ggml_set_output(graph.output);
        graph.graph = ggml_new_graph_custom(graph.context, graph_size, false);
        ggml_build_forward_expand(graph.graph, graph.output);
        graph.allocator = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(model.impl->backend_active));
        if (!graph.allocator ||
            !ggml_gallocr_alloc_graph(graph.allocator, graph.graph)) {
            return -3;
        }
        graph.n_frames = n_frames;
    }

    CtcHeadGraph & graph = *cached;
    ggml_backend_tensor_set(graph.input, encoder_out.data(), 0,
                            encoder_out.size() * sizeof(float));
    if (ggml_backend_graph_compute(model.impl->backend_active, graph.graph) !=
        GGML_STATUS_SUCCESS) {
        return -4;
    }
    logits.resize(static_cast<size_t>(n_frames) * model.vocab_size);
    ggml_backend_tensor_get(graph.output, logits.data(), 0,
                            logits.size() * sizeof(float));
    return 0;
}

// The shared tail of a Core ML encoder dispatch: rejects non-finite
// features, runs the CTC projection on ggml when required and clears the
// per-stage captures. Returns 0 when the dispatch is complete; non-zero lets
// the caller fall back to the complete native encoder graph.
static int finish_coreml_encoder_output(ParakeetCtcModel & model, int T, EncoderOutputs & out) {
#ifdef PARAKEET_USE_COREML
    if (!parakeet::coreml_features_are_finite(out.encoder_out.data(), out.encoder_out.size())) {
        PARAKEET_LOG_WARN("parakeet: Core ML encoder returned non-finite features\n");
        constexpr int kNonFiniteCoremlFeatures = -10;
        return kNonFiniteCoremlFeatures;
    }
#endif

    if (model.model_type == ParakeetModelType::CTC) {
        const int head_rc = run_coreml_ctc_head(
            model, out.encoder_out, T, out.logits);
        if (head_rc != 0) return 100 - head_rc;
    } else {
        out.logits.clear();
    }

    out.used_coreml = true;
    out.subsampling_out.clear();
    out.block_0_post_ff1.clear();
    out.block_0_post_attn.clear();
    out.block_0_post_conv.clear();
    out.block_0_post_ff2.clear();
    out.block_0_out.clear();
    out.block_last_out.clear();
    return 0;
}

// Fills out.encoder_out from the Core ML sidecar, then runs the CTC projection
// on ggml when required. Returns 0 when handled on Core ML; non-zero lets the
// caller fall back to the complete native encoder graph.
static int run_encoder_coreml(ParakeetCtcModel & model,
                              const EncoderRoute & route,
                              const float      * mel,
                              int                n_mel_frames,
                              int                mel_valid,
                              int                n_mels,
                              EncoderOutputs   & out) {
    const EncoderConfig & enc     = model.encoder_cfg;
    const int             d_model = enc.d_model;
    const bool masked = route.geometry().masks_mel_padding;
    const int T = coreml_encoder_out_frames(enc, masked ? mel_valid : n_mel_frames);

    out.n_enc_frames = T;
    out.d_model      = d_model;
    out.vocab_size   = model.vocab_size;
    out.encoder_out.resize((size_t) T * d_model);

    // Under the shared model's prediction lock (one in flight per instance).
    const int rc = route.encode(n_mel_frames, mel_valid, n_mels, mel,
                                T, d_model, out.encoder_out.data());
    if (rc != 0) return rc;
    return finish_coreml_encoder_output(model, T, out);
}

#ifdef PARAKEET_USE_COREML
// One TranscribePsy EU dispatch on the program selected_coreml_encoder chose.
// Every EU program masks the mel padding; one that preserves the encoder
// padding returns the frames of the whole (padded) input.
static int run_eu_encoder_coreml(ParakeetCtcModel & model,
                                 parakeet_coreml_context * ctx,
                                 const float * mel,
                                 int n_mel_frames,
                                 int mel_valid,
                                 int n_mels,
                                 EncoderOutputs & out) {
    const EncoderConfig & enc     = model.encoder_cfg;
    const int             d_model = enc.d_model;
    const bool preserve_padding = parakeet_coreml_preserves_encoder_padding(ctx);
    const int T = coreml_encoder_out_frames(enc, preserve_padding ? n_mel_frames : mel_valid);
    const int fixed_capacity = (int) parakeet_coreml_fixed_mel_frames(ctx, n_mels);
    const int input_frames = preserve_padding
        ? (fixed_capacity > 0 ? std::min(n_mel_frames, fixed_capacity) : n_mel_frames)
        : mel_valid;

    out.n_enc_frames = T;
    out.d_model      = d_model;
    out.vocab_size   = model.vocab_size;
    out.encoder_out.resize((size_t) T * d_model);

    const int rc = parakeet_coreml_encode_valid(ctx,
                                           input_frames, mel_valid, n_mels, mel,
                                           T, d_model, out.encoder_out.data());
    if (rc != 0) return rc;
    const int finish_rc = finish_coreml_encoder_output(model, T, out);
    if (finish_rc == 0) model.impl->last_coreml_encoder_backend = parakeet_coreml_backend_label(ctx);
    return finish_rc;
}
#endif  // PARAKEET_USE_COREML

bool model_selected_encoder_on_coreml(const ParakeetCtcModel & model, int n_mel_frames) {
#ifdef PARAKEET_USE_COREML
    if (n_mel_frames <= 0) return false;
    auto * ctx = selected_coreml_encoder(model, -1, n_mel_frames, n_mel_frames, false);
    return eu_coreml_encoder_takes(model, ctx, n_mel_frames, /*capture_intermediates=*/false);
#else
    (void) model;
    (void) n_mel_frames;
    return false;
#endif
}

// Build (and, unless measuring, allocate) the two-layer locale-prompt
// projection graph at `n_frames` encoder frames. Shared by
// run_nemotron_prompt_projection and measure_nemotron_prompt_compute so the
// fit projection prices exactly the graph the runtime builds.
//
// `measure_bytes` (fit projection): when non-null the graph is built normally
// but its compute buffer is only *sized* into `*measure_bytes*`
// (ggml_gallocr_reserve_n_size) instead of allocated; `out.allocator` stays
// null and `out` must be discarded after reading the size.
static int build_nemotron_prompt_graph(
    const ParakeetCtcModel & model,
    int n_frames,
    NemotronPromptGraph & out,
    size_t * measure_bytes = nullptr) {
    out.clear();
    constexpr size_t graph_size = 64;
    const size_t graph_memory =
        ggml_tensor_overhead() * graph_size +
        ggml_graph_overhead_custom(graph_size, false) +
        16 * 1024;
    ggml_init_params params = {};
    params.mem_size = graph_memory;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    out.context = ggml_init(params);
    if (!out.context) {
        return -3;
    }

    out.input = ggml_new_tensor_2d(
        out.context, GGML_TYPE_F32, model.encoder_cfg.d_model, n_frames);
    out.prompt = ggml_new_tensor_2d(
        out.context, GGML_TYPE_F32, model.nemotron_cfg.prompt_width, n_frames);
    ggml_set_input(out.input);
    ggml_set_input(out.prompt);
    ggml_set_output(out.prompt);
    ggml_tensor * conditioned = ggml_concat(out.context, out.input, out.prompt, 0);
    ggml_tensor * hidden = affine_mul_mat(
        out.context, model.nemotron.prompt.proj_0_w, conditioned);
    hidden = ggml_add(
        out.context, hidden, model.nemotron.prompt.proj_0_b);
    hidden = ggml_relu(out.context, hidden);
    out.output = affine_mul_mat(
        out.context, model.nemotron.prompt.proj_2_w, hidden);
    out.output = ggml_add(
        out.context, out.output, model.nemotron.prompt.proj_2_b);
    ggml_set_output(out.output);

    out.graph =
        ggml_new_graph_custom(out.context, graph_size, false);
    ggml_build_forward_expand(out.graph, out.output);

    ggml_backend_t backend = model.impl->backend_active;
    if (measure_bytes) {
        ggml_gallocr_t pricer = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(backend));
        if (!pricer) {
            return -4;
        }
        *measure_bytes = 0;
        ggml_gallocr_reserve_n_size(pricer, out.graph, nullptr, nullptr,
                                    measure_bytes);
        ggml_gallocr_free(pricer);
        return 0;
    }
    out.allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    if (!out.allocator ||
        !ggml_gallocr_alloc_graph(out.allocator, out.graph)) {
        return -4;
    }
    out.n_frames = n_frames;
    return 0;
}

int measure_nemotron_prompt_compute(ParakeetCtcModel & model,
                                    int                n_frames,
                                    size_t           & out_bytes) {
    out_bytes = 0;
    if (model.model_type != ParakeetModelType::NEMOTRON ||
        !model.impl || !model.impl->backend_active || n_frames <= 0) {
        return -1;
    }
    NemotronPromptGraph tmp;  // RAII: destructor frees the graph ctx
    return build_nemotron_prompt_graph(model, n_frames, tmp, &out_bytes);
}

size_t model_nemotron_prompt_buffer_bytes(const ParakeetCtcModel & m) {
    if (!m.impl || !m.impl->nemotron_prompt_graph) return 0;
    const NemotronPromptGraph & g = *m.impl->nemotron_prompt_graph;
    return g.allocator ? ggml_gallocr_get_buffer_size(g.allocator, 0) : 0;
}

static int prepare_nemotron_prompt(ParakeetCtcModel & model, int n_frames, int32_t prompt_id) {
    if (n_frames <= 0 || prompt_id < 0 || prompt_id >= model.nemotron_cfg.num_prompts ||
        model.nemotron_cfg.prompt_width != model.nemotron_cfg.num_prompts ||
        model.nemotron_cfg.prompt_input_width != model.encoder_cfg.d_model + model.nemotron_cfg.prompt_width) {
        return -2;
    }
    auto & cached = model.impl->nemotron_prompt_graph;
    if (!cached || cached->n_frames != n_frames) {
        cached = std::make_unique<NemotronPromptGraph>();
        if (int rc = build_nemotron_prompt_graph(model, n_frames, *cached); rc != 0) {
            cached.reset();
            return rc;
        }
    }
    NemotronPromptGraph & graph = *cached;
    if (!ggml_gallocr_alloc_graph(graph.allocator, graph.graph)) return -4;
    if (graph.prompt_id != prompt_id) {
        const int width = model.nemotron_cfg.prompt_width;
        std::vector<float> prompt(static_cast<size_t>(width) * n_frames, 0.0f);
        for (int frame = 0; frame < n_frames; ++frame) {
            prompt[static_cast<size_t>(frame) * width + prompt_id] = 1.0f;
        }
        ggml_backend_tensor_set(graph.prompt, prompt.data(), 0, prompt.size() * sizeof(float));
        graph.prompt_id = prompt_id;
    }
    return 0;
}

int run_nemotron_prompt_projection_device(ParakeetCtcModel & model,
    ggml_tensor * encoder_out, int32_t prompt_id, ggml_tensor * & projected) {
    projected = nullptr;
    if (model.model_type != ParakeetModelType::NEMOTRON || !model.impl || !model.impl->backend_active || !encoder_out ||
        encoder_out->type != GGML_TYPE_F32 || !ggml_is_contiguous(encoder_out) ||
        encoder_out->ne[0] != model.encoder_cfg.d_model || encoder_out->ne[1] <= 0 ||
        encoder_out->ne[1] > std::numeric_limits<int>::max() ||
        encoder_out->ne[2] != 1 || encoder_out->ne[3] != 1 || !encoder_out->buffer ||
        prompt_id < 0 || prompt_id >= model.nemotron_cfg.num_prompts) return -1;
    if (int rc = prepare_nemotron_prompt(model, (int) encoder_out->ne[1], prompt_id); rc != 0) return rc;
    NemotronPromptGraph & graph = *model.impl->nemotron_prompt_graph;
    ggml_backend_tensor_copy_async(model.impl->backend_active, model.impl->backend_active,
        encoder_out, graph.input);
    if (ggml_backend_graph_compute(model.impl->backend_active, graph.graph) != GGML_STATUS_SUCCESS) return -5;
    projected = graph.output;
    return 0;
}

int run_nemotron_prompt_projection(
    ParakeetCtcModel & model,
    const float * encoder_out,
    int n_frames,
    int d_model,
    int32_t prompt_id,
    std::vector<float> & projected) {
    projected.clear();
    if (model.model_type != ParakeetModelType::NEMOTRON ||
        !model.impl ||
        !model.impl->backend_active ||
        !encoder_out ||
        n_frames <= 0 ||
        d_model != model.encoder_cfg.d_model ||
        prompt_id < 0 ||
        prompt_id >= model.nemotron_cfg.num_prompts) {
        return -1;
    }

    const int prompt_width = model.nemotron_cfg.prompt_width;
    const int input_width = model.nemotron_cfg.prompt_input_width;
    if (prompt_width != model.nemotron_cfg.num_prompts ||
        input_width != d_model + prompt_width) {
        return -2;
    }

    if (int rc = prepare_nemotron_prompt(model, n_frames, prompt_id); rc != 0) return rc;
    NemotronPromptGraph & graph = *model.impl->nemotron_prompt_graph;
    ggml_backend_tensor_set(
        graph.input,
        encoder_out,
        0,
        static_cast<size_t>(d_model) * n_frames * sizeof(float));
    if (ggml_backend_graph_compute(
            model.impl->backend_active, graph.graph) != GGML_STATUS_SUCCESS) {
        return -5;
    }

    projected.resize(static_cast<size_t>(d_model) * n_frames);
    ggml_backend_tensor_get(
        graph.output,
        projected.data(),
        0,
        projected.size() * sizeof(float));
    return 0;
}

// Frees the least recently used encoder graphs until at most `keep` remain.
static void evict_encoder_graphs_to(std::vector<std::unique_ptr<EncoderGraph>> & cache,
                                    size_t keep) {
    while (cache.size() > keep) {
        cache.front()->free_();
        cache.erase(cache.begin());
    }
}

size_t set_encoder_graph_cache_limit(ParakeetCtcModel & model, size_t limit) {
    if (!model.impl) return 0;
    const size_t prev = model.impl->encoder_graph_cache_limit;
    model.impl->encoder_graph_cache_limit = std::max<size_t>(1, limit);
    evict_encoder_graphs_to(model.impl->encoder_graphs, model.impl->encoder_graph_cache_limit);
    return prev;
}

size_t encoder_graph_cache_size(const ParakeetCtcModel & model) {
    return model.impl ? model.impl->encoder_graphs.size() : 0;
}

static int run_encoder_ggml(ParakeetCtcModel & model, const float * mel, int n_mel_frames,
                            int n_mels, EncoderOutputs & out, int max_layers,
                            bool capture_intermediates, const EncoderOutputConsumer & consumer,
                            int mel_valid, bool all_valid);

int run_encoder(ParakeetCtcModel   & model,
                const float        * mel,
                int                  n_mel_frames,
                int                  n_mels,
                EncoderOutputs     & out,
                const EncoderRoute & route,
                int                  max_layers,
                bool                 capture_intermediates,
                bool                 allow_coreml_padded,
                const EncoderOutputConsumer & consumer,
                int valid_mel_frames_override, bool prefer_precise_eu) {
    // Set pessimistically on every invocation. Only a successful Core ML
    // prediction may change this to true.
    out.used_coreml = false;
    out.encoder_consumed = false;
    if (!model.impl || !model.impl->backend_active) return -1;

    // Scan from the end for the last non-zero mel frame: the mel
    // pre-processor zeros out trailing pad frames (per-feature CMVN
    // sets them to 0), and `compute_log_mel` always produces at least
    // one valid frame. Reverse-scan with early-exit replaces the
    // worst-case O(n_mel_frames * n_mels) full sweep with the typical
    // case "the last frame is valid -> 1 inner iteration only" path
    // -- the fast path for non-streaming `Engine::transcribe()` calls
    // and the tail of any chunk in Mode 2 / Mode 3 streaming.
    int mel_valid = 0;
    for (int t = n_mel_frames - 1; t >= 0; --t) {
        const float * row = mel + (size_t) t * n_mels;
        for (int m = 0; m < n_mels; ++m) {
            if (row[m] != 0.0f) { mel_valid = t + 1; break; }
        }
        if (mel_valid != 0) break;
    }
    if (mel_valid == 0) mel_valid = n_mel_frames;
    if (valid_mel_frames_override > 0 && valid_mel_frames_override <= n_mel_frames)
        mel_valid = valid_mel_frames_override;
    const bool all_valid = (mel_valid == n_mel_frames);

#ifdef PARAKEET_USE_COREML
    // TranscribePsy EU runs its own programs, picked per dispatch (its route
    // only shaped the plan). A failure falls back to ggml, or is an error
    // when the programs own the encoder.
    if (model.model_variant == "transcribepsy-250m-eu") {
        model.impl->last_coreml_encoder_backend = ggml_backend_name(model.impl->backend_active);
        auto * coreml = selected_coreml_encoder(model, max_layers, n_mel_frames, mel_valid, prefer_precise_eu);
        if (eu_coreml_encoder_takes(model, coreml, n_mel_frames, capture_intermediates)) {
            const int rc = run_eu_encoder_coreml(model, coreml, mel, n_mel_frames, mel_valid, n_mels, out);
            if (rc == 0) return 0;
            if (model.impl->coreml_owns_encoder) {
                PARAKEET_LOG_ERROR("parakeet: Core ML encoder failed (rc=%d) and the GGUF holds no "
                                   "encoder weights to fall back to\n", rc);
                return rc;
            }
            PARAKEET_LOG_WARN("parakeet: Core ML encoder failed (rc=%d); falling back to ggml encoder\n", rc);
        }
        return run_encoder_ggml(model, mel, n_mel_frames, n_mels, out, max_layers,
                                capture_intermediates, consumer, mel_valid, all_valid);
    }
#else
    (void) prefer_precise_eu;
#endif

    // Core ML sidecar on the unit's route: a validated CTC/
    // IndicConformer, Unified RNN-T, TDT, EOU, Nemotron, or Sortformer
    // FastConformer encoder hands encoder_out back to the ggml decoder or
    // diarization head. A failure retires the sidecar; this dispatch and the
    // rest of the unit run on ggml with the plan already chosen. A pass over
    // only the first layers (a tap) never goes to the sidecar.
    const bool partial_layers = max_layers >= 0 && max_layers < model.encoder_cfg.n_layers;
    const EncoderEligibility eligibility = partial_layers
        ? EncoderEligibility::ggml
        : coreml_encoder_eligibility(model, route, n_mel_frames, all_valid,
                                     capture_intermediates, allow_coreml_padded);
    bool used_coreml = false;
    int coreml_rc = 0;
    const int rc = dispatch_on_route(
        route, eligibility,
        [&] { return run_encoder_coreml(model, route, mel, n_mel_frames, mel_valid, n_mels, out); },
        [&] {
            if (encode_prediction_failed(coreml_rc)) {
                PARAKEET_LOG_WARN("parakeet: Core ML encoder failed (rc=%d); retired it, "
                                  "falling back to the ggml encoder\n", coreml_rc);
            } else if (coreml_rc != 0) {
                PARAKEET_LOG_WARN("parakeet: Core ML encoder cannot serve this call (rc=%d); "
                                  "running it on the ggml encoder\n", coreml_rc);
            }
            return run_encoder_ggml(model, mel, n_mel_frames, n_mels, out, max_layers,
                                    capture_intermediates, consumer, mel_valid, all_valid);
        },
        used_coreml, &coreml_rc);
    if (rc != 0 && route.owner() && coreml_rc != 0 && !used_coreml) {
        PARAKEET_LOG_ERROR("parakeet: Core ML encoder failed (rc=%d) and the GGUF holds no "
                           "encoder weights to fall back to\n", rc);
    }
    return rc;
}

static int run_encoder_ggml(ParakeetCtcModel & model, const float * mel, int n_mel_frames,
                            int n_mels, EncoderOutputs & out, int max_layers,
                            bool capture_intermediates, const EncoderOutputConsumer & consumer,
                            int mel_valid, bool all_valid) {
    ggml_backend_t backend = model.impl->backend_active;
    const EncoderConfig & enc = model.encoder_cfg;
    const int d_model = enc.d_model;

    auto & cache = model.impl->encoder_graphs;
    const int layers_key = std::clamp(
        max_layers >= 0 ? max_layers :
            (std::getenv("PARAKEET_MAX_LAYERS") ? std::atoi(std::getenv("PARAKEET_MAX_LAYERS"))
                                                : enc.n_layers),
        0, enc.n_layers);

    EncoderGraph * g_ptr = nullptr;
    for (size_t i = 0; i < cache.size(); ++i) {
        EncoderGraph & e = *cache[i];
        const bool layers_match = e.n_run_layers == layers_key;
        if (!e.bypass_pre_encode && e.T_mel == n_mel_frames && e.T_mel_valid == mel_valid &&
            e.eu_use_adapters == model.eu_use_adapters && layers_match) {
            if (i + 1 != cache.size()) {
                auto moved = std::move(cache[i]);
                cache.erase(cache.begin() + i);
                cache.push_back(std::move(moved));
            }
            g_ptr = cache.back().get();
            break;
        }
    }

    if (!g_ptr) {
        evict_encoder_graphs_to(cache, model.impl->encoder_graph_cache_limit - 1);
        cache.push_back(std::make_unique<EncoderGraph>());
        EncoderGraph & e = *cache.back();
        if (int rc = build_encoder_graph_cached(model, e, n_mel_frames, mel_valid, n_mels, max_layers, all_valid,
                                                backend, /*bypass_pre_encode=*/false, /*T_enc_override=*/0); rc != 0) {
            cache.pop_back();
            return rc;
        }
        g_ptr = &e;
    }
    EncoderGraph & g = *g_ptr;

    auto sub_out_len = [&](int Lin) {
        return enc.causal_downsampling ? (Lin / 2 + 1) : _conv_out_len(Lin, 3, 2, 1);
    };

    const int L0 = n_mel_frames;
    const int L1 = sub_out_len(L0);
    const int L2 = sub_out_len(L1);
    const int L3 = sub_out_len(L2);

    const int V0 = mel_valid;
    const int V1 = sub_out_len(V0);
    const int V2 = sub_out_len(V1);
    const int V3 = sub_out_len(V2);

    const int T = L3;
    const int vocab_size = model.vocab_size;

    if (!g.allocator.scheduler()) restore_encoder_graph_srcs(g);
    if (!g.allocator.allocate(g.cgraph)) {
        return -3;
    }

    auto safe_set = [](ggml_tensor * t, const void * src, size_t bytes) {
        if (t && t->buffer) ggml_backend_tensor_set(t, src, 0, bytes);
    };
    safe_set(g.mel_in,  mel,                 (size_t) n_mels * L0 * sizeof(float));
    safe_set(g.pe_in,   g.pe_host.data(),    g.pe_host.size()     * sizeof(float));
    if (g.att_mask) {
        safe_set(g.att_mask, g.att_mask_host.data(),
                 g.att_mask_host.size() * sizeof(float));
    }

    if (g.allocator.compute(g.cgraph) != GGML_STATUS_SUCCESS) {
        return -4;
    }

    out.n_enc_frames =
        model.model_type == ParakeetModelType::NEMOTRON ? V3 : T;
    out.d_model      = d_model;
    out.vocab_size   = vocab_size;

    auto copy_tensor = [&](ggml_tensor * t, std::vector<float> & dst) {
        if (!t) { dst.clear(); return; }
        dst.resize((size_t) ggml_nelements(t));
        ggml_backend_tensor_get(t, dst.data(), 0, dst.size() * sizeof(float));
    };
    // The production transcribe/diarize/stream path only consumes
    // `encoder_out` (TDT/EOU/Sortformer decoders) and `logits` (CTC).
    // Skip the per-stage host copies in that case -- saves roughly
    // 7 * d_model * T_enc * 4 bytes per inference call (~4-5 MB on
    // 0.6B at T_enc=137), which is on the GPU<->host critical path
    // for OpenCL / CUDA / Vulkan / Metal streaming workloads where
    // each chunk drives a fresh `run_encoder()` round-trip.
    if (capture_intermediates) {
        copy_tensor(g.sub_out_node,         out.subsampling_out);
        copy_tensor(g.post_ff1_0_node,      out.block_0_post_ff1);
        copy_tensor(g.post_attn_0_node,     out.block_0_post_attn);
        copy_tensor(g.post_conv_0_node,     out.block_0_post_conv);
        copy_tensor(g.post_ff2_0_node,      out.block_0_post_ff2);
        copy_tensor(g.block_0_out_node,     out.block_0_out);
        copy_tensor(g.block_last_out_node,  out.block_last_out);
        copy_tensor(g.block_0_attn_qkv_node,     out.block_0_attn_qkv);
        copy_tensor(g.block_0_attn_k_raw_node,   out.block_0_attn_k_raw);
        copy_tensor(g.block_0_attn_v_raw_node,   out.block_0_attn_v_raw);
        copy_tensor(g.block_0_attn_q_perm_node,  out.block_0_attn_q_perm);
        copy_tensor(g.block_0_attn_k_perm_node,  out.block_0_attn_k_perm);
        copy_tensor(g.block_0_attn_q_u_node,     out.block_0_attn_q_u);
        copy_tensor(g.block_0_attn_ac_node,      out.block_0_attn_ac);
        copy_tensor(g.block_0_attn_bd_node,      out.block_0_attn_bd);
        copy_tensor(g.block_0_attn_scores_node,  out.block_0_attn_scores);
        copy_tensor(g.block_0_attn_softmax_node, out.block_0_attn_softmax);
        copy_tensor(g.block_0_attn_out_node,     out.block_0_attn_out);
    } else {
        out.subsampling_out.clear();
        out.block_0_post_ff1.clear();
        out.block_0_post_attn.clear();
        out.block_0_post_conv.clear();
        out.block_0_post_ff2.clear();
        out.block_0_out.clear();
        out.block_last_out.clear();
        out.block_0_attn_qkv.clear();
        out.block_0_attn_k_raw.clear();
        out.block_0_attn_v_raw.clear();
        out.block_0_attn_q_perm.clear();
        out.block_0_attn_k_perm.clear();
        out.block_0_attn_q_u.clear();
        out.block_0_attn_ac.clear();
        out.block_0_attn_bd.clear();
        out.block_0_attn_scores.clear();
        out.block_0_attn_softmax.clear();
        out.block_0_attn_out.clear();
    }
    if (consumer && !capture_intermediates && g.encoder_out_node &&
        g.encoder_out_node->ne[1] == out.n_enc_frames) {
        const int consumed = consumer(g.encoder_out_node, backend);
        if (consumed < 0) return consumed;
        out.encoder_consumed = consumed == kEncoderOutputConsumed;
    }
    if (out.encoder_consumed) {
        out.encoder_out.clear();
    } else {
        copy_tensor(g.encoder_out_node, out.encoder_out);
    }
    copy_tensor(g.logits_node,          out.logits);

    return 0;
}

static void eu_gate_dense(const float * w, const float * b, const float * x,
                          int in, int out, float * y) {
    for (int o = 0; o < out; ++o) {
        float v = b[o];
        const float * row = w + (size_t) o * in;
        for (int i = 0; i < in; ++i) v += row[i] * x[i];
        y[o] = v;
    }
}

static std::vector<float> eu_gate_pool(const TranscribePsyEuGate & gate,
                                       const std::vector<float> & features,
                                       int frames, int dim) {
    const int attn = gate.attn_dim;
    std::vector<float> attention((size_t) frames * dim);
    std::vector<float> hidden(attn);
    for (int t = 0; t < frames; ++t) {
        const float * x = features.data() + (size_t) t * dim;
        eu_gate_dense(gate.proj0_w.data(), gate.proj0_b.data(), x, dim, attn, hidden.data());
        for (float & v : hidden) v = std::tanh(v);
        eu_gate_dense(gate.proj2_w.data(), gate.proj2_b.data(), hidden.data(), attn, dim,
                      attention.data() + (size_t) t * dim);
    }
    std::vector<float> pooled(2 * dim);
    for (int d = 0; d < dim; ++d) {
        float max_logit = std::numeric_limits<float>::lowest();
        for (int t = 0; t < frames; ++t)
            max_logit = std::max(max_logit, attention[(size_t) t * dim + d]);
        double sum = 0.0, mean = 0.0, second = 0.0;
        for (int t = 0; t < frames; ++t) {
            const double w = std::exp(attention[(size_t) t * dim + d] - max_logit);
            const double x = features[(size_t) t * dim + d];
            sum += w;
            mean += w * x;
            second += w * x * x;
        }
        mean /= sum;
        pooled[d] = (float) mean;
        pooled[dim + d] = (float) std::sqrt(std::max(1e-8, second / sum - mean * mean));
    }
    return pooled;
}

static float eu_gate_head(const TranscribePsyEuGate & gate, std::vector<float> pooled) {
    double avg = 0.0, variance = 0.0;
    for (float v : pooled) avg += v;
    avg /= pooled.size();
    for (float v : pooled) variance += (v - avg) * (v - avg);
    variance /= pooled.size();
    const float inv_std = (float) (1.0 / std::sqrt(variance + 1e-5));
    for (size_t i = 0; i < pooled.size(); ++i)
        pooled[i] = (pooled[i] - (float) avg) * inv_std * gate.norm_w[i] + gate.norm_b[i];
    std::vector<float> mlp(gate.hidden);
    eu_gate_dense(gate.mlp0_w.data(), gate.mlp0_b.data(), pooled.data(), (int) pooled.size(),
                  gate.hidden, mlp.data());
    for (float & v : mlp)
        v = 0.5f * v * (1.0f + std::erf(v * 0.7071067811865475f));
    float logit = gate.mlp3_b[0];
    for (int i = 0; i < gate.hidden; ++i) logit += gate.mlp3_w[i] * mlp[i];
    return 1.0f / (1.0f + std::exp(-logit));
}

struct EuGateMel {
    std::vector<float> owned;
    const float * external = nullptr;
    int frames = 0;

    const float * data() const { return owned.empty() ? external : owned.data(); }
};

static EuGateMel prepare_eu_gate_mel(const ParakeetCtcModel & model,
                                     const float * mel, int n_mel_frames, int n_mels,
                                     const float * cropped_mel, int cropped_mel_frames) {
    const int max_frames = 1 + kTranscribePsyEuGateSeconds *
        model.mel_cfg.sample_rate / model.mel_cfg.hop_length;
    EuGateMel input;
    input.external = mel;
    input.frames = std::min(n_mel_frames, max_frames);
    if (cropped_mel && cropped_mel_frames > 0) {
        input.external = cropped_mel;
        input.frames = cropped_mel_frames;
    } else if (n_mel_frames > max_frames) {
        input.owned.assign(mel, mel + (size_t) input.frames * n_mels);
        if (model.mel_cfg.normalize == MelNormalize::PerFeature)
            apply_per_feature_cmvn(input.owned, input.frames - 1, n_mels);
        std::fill(input.owned.end() - n_mels, input.owned.end(), 0.0f);
    }
    return input;
}

static int eu_gate_valid_mel_frames(const ParakeetCtcModel & model,
                                    const EuGateMel & input, int audio_samples) {
    if (audio_samples <= 0) return std::max(1, input.frames - 1);
    const int64_t gate_samples = (int64_t) kTranscribePsyEuGateSeconds *
        model.mel_cfg.sample_rate;
    const int64_t samples = std::min((int64_t) audio_samples, gate_samples);
    const int64_t hop = model.mel_cfg.hop_length;
    const int valid_frames = (int) std::max<int64_t>(1, (samples + hop - 1) / hop);
    return std::min(valid_frames, input.frames);
}

static int eu_gate_valid_encoder_frames(const EncoderConfig & config, int mel_frames,
                                         int total_encoder_frames) {
    int frames = mel_frames;
    for (int factor = 1; factor < config.subsampling_factor; factor *= 2)
        frames = config.causal_downsampling ? frames / 2 + 1 : (frames + 1) / 2;
    return std::min(frames, total_encoder_frames);
}

bool resolve_transcribepsy_eu_route(ParakeetCtcModel & model,
                                    const std::string & language,
                                    const float * mel, int n_mel_frames, int n_mels,
                                    float * p_non_english,
                                    const float * cropped_mel,
                                    int cropped_mel_frames,
                                    int audio_samples,
                                    bool * encoder_used_coreml, bool prefer_precise_eu,
                                    EncoderBackendTally * gate_tally) {
    if (encoder_used_coreml) *encoder_used_coreml = true;
    if (model.model_variant != "transcribepsy-250m-eu") return false;
    model.eu_use_adapters = false;
    if (p_non_english) *p_non_english = -1.0f;
    if (language == "en") return false;
    if (language == "other") {
        model.eu_use_adapters = true;
        return true;
    }
    if (!language.empty() && language != "auto")
        throw std::invalid_argument("TranscribePsy EU language must be en, other, or auto");
    const auto & gate = model.eu_gate;
    const int dim = model.encoder_cfg.d_model;
    if (!mel || n_mel_frames <= 0 || n_mels != model.mel_cfg.n_mels ||
        gate.proj0_w.empty())
        throw std::runtime_error("TranscribePsy EU gate is unavailable");

    const EuGateMel input = prepare_eu_gate_mel(
        model, mel, n_mel_frames, n_mels, cropped_mel, cropped_mel_frames);
    EncoderOutputs tap;
    // The gate tap is a partial-layer pass: not a runtime route; run_encoder
    // picks the EU gate program for it.
    const int rc = run_encoder(model, input.data(), input.frames, n_mels, tap,
                               EncoderRoute::ggml(), gate.layer + 1,
                               /*capture_intermediates=*/false,
                               /*allow_coreml_padded=*/true, {},
                               eu_gate_valid_mel_frames(model, input, audio_samples), prefer_precise_eu);
    if (rc != 0 || tap.n_enc_frames <= 0 ||
        tap.encoder_out.size() != (size_t) tap.n_enc_frames * dim)
        throw std::runtime_error("TranscribePsy EU gate encoder failed (rc=" +
                                 std::to_string(rc) + ")");
    const int frames = eu_gate_valid_encoder_frames(
        model.encoder_cfg, eu_gate_valid_mel_frames(model, input, audio_samples),
        tap.n_enc_frames);
    const float probability = eu_gate_head(gate, eu_gate_pool(gate, tap.encoder_out, frames, dim));
    if (encoder_used_coreml) *encoder_used_coreml = tap.used_coreml;
    // The gate is one dispatch of the call; its program is the EU dispatch
    // label run_encoder just recorded.
    if (gate_tally) gate_tally->add(tap.used_coreml, tap.used_coreml ? model_encoder_backend_name(model) : std::string());
    PARAKEET_LOG_INFO("TranscribePsy EU gate encoder: %s\n", tap.used_coreml ? "coreml" : "ggml");
    if (p_non_english) *p_non_english = probability;
    model.eu_use_adapters = probability >= gate.threshold;
    PARAKEET_LOG_INFO("TranscribePsy EU gate: p(non-English)=%.4f, threshold=%.4f, route=%s, frames=%d/%d\n",
                      probability, gate.threshold,
                      model.eu_use_adapters ? "other" : "en", frames, tap.n_enc_frames);
    return model.eu_use_adapters;
}

int measure_encoder_compute(ParakeetCtcModel & model,
                            int                n_mel_frames,
                            int                n_mels,
                            size_t           & out_bytes) {
    if (!model.impl || !model.impl->backend_active) return -1;
    out_bytes = 0;
    EncoderGraph tmp;
    return build_encoder_graph_cached(model, tmp, n_mel_frames, /*mel_valid=*/n_mel_frames, n_mels,
                                      /*n_run_layers_override=*/-1,
                                      /*all_valid=*/true,
                                      model.impl->backend_active,
                                      /*bypass_pre_encode=*/false,
                                      /*T_enc_override=*/0,
                                      &out_bytes);
}

int run_encoder_bypass_pre_encode(ParakeetCtcModel & model,
                                  const float      * pre_encode_in,
                                  int                n_pre_encode_frames,
                                  int                d_model_in,
                                  EncoderOutputs   & out,
                                  int                max_layers) {
    out.used_coreml = false;
    out.encoder_consumed = false;
    if (!model.impl || !model.impl->backend_active) return -1;
    if (!pre_encode_in || n_pre_encode_frames <= 0) return -1;

    ggml_backend_t backend = model.impl->backend_active;
    const EncoderConfig & enc = model.encoder_cfg;
    const int d_model = enc.d_model;
    if (d_model_in != d_model) {
        std::fprintf(stderr,
            "run_encoder_bypass_pre_encode: d_model mismatch (%d vs %d)\n",
            d_model_in, d_model);
        return -1;
    }

#ifdef PARAKEET_USE_COREML
    // The AOSC sidecar has a fixed capacity. Shorter cache/FIFO/chunk slabs are
    // zero-padded and accompanied by an attention-validity mask inside the
    // Core ML wrapper; oversized/custom geometries keep the ggml path. One
    // AOSC step is the unit: it acquires the slot once, runs on ggml while the
    // slot is loading, and a failure retires the sidecar.
    if (model.impl->bypass_slot != nullptr &&
        model.model_type == ParakeetModelType::SORTFORMER &&
        model.model_variant == "sortformer-streaming-v2.1-aosc" &&
        max_layers < 0) {
        pcm::Acquired acquired =
            model.impl->bypass_slot->acquire(pcm::Demand::now, std::nullopt, model.impl->unit_cancel);
        auto * ctx = acquired.model ? acquired.model->as<parakeet_coreml_context>() : nullptr;
        const int fixed_frames = ctx ? (int) parakeet_coreml_fixed_mel_frames(ctx, d_model) : 0;
        const bool strict = model.impl->coreml->resolved().strict();
        if (strict && ctx != nullptr && !(fixed_frames > 0 && n_pre_encode_frames <= fixed_frames)) {
            throw pcm::CoremlError("the Core ML bypass encoder cannot serve this step (" +
                                       std::to_string(n_pre_encode_frames) +
                                       " frames) and Core ML is required",
                                   pcm::SlotState::ready, pcm::ErrorClass::deterministic);
        }
        if (fixed_frames > 0 && n_pre_encode_frames <= fixed_frames) {
            out.n_enc_frames = n_pre_encode_frames;
            out.d_model = d_model;
            out.vocab_size = model.vocab_size;
            out.encoder_out.resize((size_t) n_pre_encode_frames * d_model);
            int rc = 0;
            {
                auto prediction = acquired.model->lock_for_prediction();
                rc = parakeet_coreml_encode_embeddings(ctx, n_pre_encode_frames, d_model,
                                                       pre_encode_in, out.encoder_out.data());
            }
            if (rc == 0) {
                out.used_coreml = true;
                out.subsampling_out.clear();
                out.block_0_post_ff1.clear();
                out.block_0_post_attn.clear();
                out.block_0_post_conv.clear();
                out.block_0_post_ff2.clear();
                out.block_0_out.clear();
                out.block_last_out.clear();
                out.logits.clear();
                return 0;
            }
            if (encode_prediction_failed(rc)) {
                PARAKEET_LOG_WARN(
                    "parakeet: Core ML bypass encoder failed (rc=%d); "
                    "retiring the sidecar and falling back to ggml encoder\n", rc);
                model.impl->bypass_slot->retire("prediction failed (rc=" + std::to_string(rc) + ")");
            } else if (!strict) {
                PARAKEET_LOG_WARN(
                    "parakeet: Core ML bypass encoder cannot serve this step (rc=%d); "
                    "running it on the ggml encoder\n", rc);
            }
            if (strict) {
                const bool failed = encode_prediction_failed(rc);
                throw pcm::CoremlError("the Core ML bypass encoder " +
                                           std::string(failed ? "prediction failed" : "cannot serve this step") +
                                           " (rc=" + std::to_string(rc) + ") and Core ML is required",
                                       failed ? pcm::SlotState::retired : pcm::SlotState::ready,
                                       failed ? pcm::ErrorClass::transient : pcm::ErrorClass::deterministic);
            }
        }
    }
#endif

    auto & cache = model.impl->encoder_graphs;
    const int layers_key = (max_layers >= 0) ? max_layers : -1;

    EncoderGraph * g_ptr = nullptr;
    for (size_t i = 0; i < cache.size(); ++i) {
        EncoderGraph & e = *cache[i];
        const bool layers_match = (layers_key < 0) || (e.n_run_layers == layers_key);
        if (e.bypass_pre_encode && e.T_enc == n_pre_encode_frames &&
            e.eu_use_adapters == model.eu_use_adapters && layers_match) {
            if (i + 1 != cache.size()) {
                auto moved = std::move(cache[i]);
                cache.erase(cache.begin() + i);
                cache.push_back(std::move(moved));
            }
            g_ptr = cache.back().get();
            break;
        }
    }

    if (!g_ptr) {
        evict_encoder_graphs_to(cache, model.impl->encoder_graph_cache_limit - 1);
        cache.push_back(std::make_unique<EncoderGraph>());
        EncoderGraph & e = *cache.back();
        if (int rc = build_encoder_graph_cached(model, e, /*n_mel_frames=*/0, /*mel_valid=*/0, /*n_mels=*/0,
                                                max_layers, /*all_valid=*/true, backend,
                                                /*bypass_pre_encode=*/true,
                                                /*T_enc_override=*/n_pre_encode_frames); rc != 0) {
            cache.pop_back();
            return rc;
        }
        g_ptr = &e;
    }
    EncoderGraph & g = *g_ptr;

    if (!g.allocator.scheduler()) restore_encoder_graph_srcs(g);
    if (!g.allocator.allocate(g.cgraph)) {
        return -3;
    }

    auto safe_set = [](ggml_tensor * t, const void * src, size_t bytes) {
        if (t && t->buffer) ggml_backend_tensor_set(t, src, 0, bytes);
    };
    safe_set(g.pre_encode_in, pre_encode_in,
             (size_t) d_model * (size_t) n_pre_encode_frames * sizeof(float));
    safe_set(g.pe_in, g.pe_host.data(), g.pe_host.size() * sizeof(float));
    if (g.att_mask) {
        safe_set(g.att_mask, g.att_mask_host.data(),
                 g.att_mask_host.size() * sizeof(float));
    }

    if (g.allocator.compute(g.cgraph) != GGML_STATUS_SUCCESS) {
        return -4;
    }

    out.n_enc_frames = n_pre_encode_frames;
    out.d_model      = d_model;
    out.vocab_size   = model.vocab_size;

    auto copy_tensor = [&](ggml_tensor * t, std::vector<float> & dst) {
        if (!t) { dst.clear(); return; }
        dst.resize((size_t) ggml_nelements(t));
        ggml_backend_tensor_get(t, dst.data(), 0, dst.size() * sizeof(float));
    };
    out.subsampling_out.clear();
    out.block_0_post_ff1.clear();
    out.block_0_post_attn.clear();
    out.block_0_post_conv.clear();
    out.block_0_post_ff2.clear();
    out.block_0_out.clear();
    out.block_last_out.clear();
    copy_tensor(g.encoder_out_node, out.encoder_out);
    copy_tensor(g.logits_node,      out.logits);
    return 0;
}

namespace {

struct SubstageGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph  * cgraph = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_tensor  * x_in  = nullptr;
    ggml_tensor  * pe_in = nullptr;
    ggml_tensor  * out   = nullptr;
    void free_() {
        if (alloc) ggml_gallocr_free(alloc);
        if (ctx)   ggml_free(ctx);
        *this = SubstageGraph{};
    }
};

enum class Substage { FF1, ATTN, CONV, FF2, NORM_OUT, FULL_BLOCK };

static int build_substage_graph(const ParakeetCtcModel & model,
                                SubstageGraph & g,
                                int T, Substage stage,
                                ggml_backend_t backend) {
    const float ff_headroom = matmul_output_headroom(backend);
    const EncoderConfig & enc = model.encoder_cfg;
    const int d_model = enc.d_model;
    const int H  = enc.n_heads;
    const int HD = enc.head_dim;
    const int conv_kernel = enc.conv_kernel;
    const float eps = enc.layer_norm_eps;

    const size_t graph_slots = 4096;
    const size_t overhead = ggml_tensor_overhead() * graph_slots
                          + ggml_graph_overhead_custom(graph_slots, false);
    ggml_init_params gp = { overhead, nullptr, /*no_alloc=*/ true };
    g.ctx = ggml_init(gp);
    if (!g.ctx) return -1;

    g.x_in  = ggml_new_tensor_4d(g.ctx, GGML_TYPE_F32, d_model, T, 1, 1);
    g.pe_in = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, d_model, 2 * T - 1);
    ggml_set_name(g.x_in, "x_in");
    ggml_set_name(g.pe_in, "pe_in");

    const BlockWeights & W = model.blocks[0];
    ggml_tensor * x = g.x_in;

    if (stage == Substage::FF1 || stage == Substage::FULL_BLOCK) {
        ggml_tensor * r = x;
        ggml_tensor * y = conformer_ff_graph(g.ctx, x,
                                             W.norm_ff1_w, W.norm_ff1_b,
                                             W.ff1_l1_w,   W.ff1_l1_b,
                                             W.ff1_l2_w,   W.ff1_l2_b, eps, ff_headroom);
        y = ggml_scale(g.ctx, y, 0.5f);
        x = ggml_add(g.ctx, r, y);
    }
    if (stage == Substage::ATTN || stage == Substage::FULL_BLOCK) {
        ggml_tensor * r = x;
        ggml_tensor * xn = layer_norm_affine(g.ctx, x, W.norm_attn_w, W.norm_attn_b, eps);
        ggml_tensor * y = rel_pos_mha_graph(g.ctx, xn, g.pe_in, /*pos_proj=*/nullptr, /*att_mask=*/nullptr,
                                            W, H, HD, T, model.impl->attn);
        x = ggml_add(g.ctx, r, y);
    }
    if (stage == Substage::CONV || stage == Substage::FULL_BLOCK) {
        ggml_tensor * r = x;
        ggml_tensor * xn = layer_norm_affine(g.ctx, x, W.norm_conv_w, W.norm_conv_b, eps);
        ggml_tensor * y = conformer_conv_graph(g.ctx, xn, W, d_model, T, conv_kernel,
                                               select_dw_lowering(model, backend), model.impl->glu_fused,
                                               enc.conv_norm_type, enc.conv_causal, eps, T);
        x = ggml_add(g.ctx, r, y);
    }
    if (stage == Substage::FF2 || stage == Substage::FULL_BLOCK) {
        ggml_tensor * r = x;
        ggml_tensor * y = conformer_ff_graph(g.ctx, x,
                                             W.norm_ff2_w, W.norm_ff2_b,
                                             W.ff2_l1_w,   W.ff2_l1_b,
                                             W.ff2_l2_w,   W.ff2_l2_b, eps, ff_headroom);
        y = ggml_scale(g.ctx, y, 0.5f);
        x = ggml_add(g.ctx, r, y);
    }
    if (stage == Substage::NORM_OUT || stage == Substage::FULL_BLOCK) {
        x = layer_norm_affine(g.ctx, x, W.norm_out_w, W.norm_out_b, eps);
    }

    g.out = x;
    ggml_set_output(g.out);

    g.cgraph = ggml_new_graph_custom(g.ctx, graph_slots, false);
    ggml_build_forward_expand(g.cgraph, g.out);

    g.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!g.alloc || !ggml_gallocr_reserve(g.alloc, g.cgraph)) {
        g.free_();
        return -2;
    }
    return 0;
}

}

int profile_block_substages(ParakeetCtcModel & model,
                            int T_enc,
                            int warmup_runs,
                            int timed_runs,
                            BlockSubstageTimes & out) {
    if (!model.impl || !model.impl->backend_active) return -1;
    // The isolated substage builder has no affine graph scope. Refuse packed
    // models rather than timing raw integer carriers as ordinary weights.
    if (!model.impl->affine_weights.empty()) {
        PARAKEET_LOG_ERROR("[profile] packed-affine models do not support sub-stage profiling; "
                           "use full encoder/inference measurements instead\n");
        return -2;
    }
    ggml_backend_t backend = model.impl->backend_active;
    const int d_model = model.encoder_cfg.d_model;

    std::vector<float> x_host((size_t) d_model * T_enc);
    for (size_t i = 0; i < x_host.size(); ++i) {
        x_host[i] = 1e-3f * (float)((int)(i * 2654435761u) % 1000 - 500);
    }
    std::vector<float> pe_host = compute_rel_pos_encoding(T_enc, d_model);

    auto measure = [&](Substage stage) -> double {
        SubstageGraph g;
        if (build_substage_graph(model, g, T_enc, stage, backend) != 0) return -1.0;

        std::vector<double> timings;
        timings.reserve(timed_runs);
        for (int r = 0; r < warmup_runs + timed_runs; ++r) {
            if (!ggml_gallocr_alloc_graph(g.alloc, g.cgraph)) { g.free_(); return -1.0; }
            ggml_backend_tensor_set(g.x_in,  x_host.data(),  0, x_host.size()  * sizeof(float));
            if (g.pe_in->buffer) {
                ggml_backend_tensor_set(g.pe_in, pe_host.data(), 0, pe_host.size() * sizeof(float));
            }

            const auto t0 = std::chrono::steady_clock::now();
            ggml_backend_graph_compute(backend, g.cgraph);
            const double dt = std::chrono::duration_cast<std::chrono::microseconds>(
                                  std::chrono::steady_clock::now() - t0).count() / 1000.0;
            if (r >= warmup_runs) timings.push_back(dt);
        }
        g.free_();

        std::sort(timings.begin(), timings.end());
        return timings.empty() ? -1.0
             : (timings.size() % 2 == 1 ? timings[timings.size()/2]
                                        : 0.5 * (timings[timings.size()/2 - 1] + timings[timings.size()/2]));
    };

    out.ff1_ms      = measure(Substage::FF1);
    out.attn_ms     = measure(Substage::ATTN);
    out.conv_ms     = measure(Substage::CONV);
    out.ff2_ms      = measure(Substage::FF2);
    out.norm_out_ms = measure(Substage::NORM_OUT);
    out.block_full_ms = measure(Substage::FULL_BLOCK);
    return 0;
}

std::vector<int32_t> ctc_greedy_decode(const float * logits,
                                       int           n_frames,
                                       int           vocab_size,
                                       int32_t       blank_id,
                                       const CtcDecodeOptions * opts) {
    std::vector<int32_t> decoded;
    int32_t prev = -1;
    ctc_greedy_decode_window(logits, 0, n_frames, vocab_size, blank_id,
                             prev, decoded, nullptr, opts);
    return decoded;
}

bool find_ctc_language_range(const ParakeetCtcModel & model,
                             const std::string      & language,
                             int32_t                & out_start,
                             int32_t                & out_end) {
    if (language.empty()) return false;
    for (const auto & range : model.ctc_lang_ranges) {
        if (range.id == language) {
            out_start = range.token_start;
            out_end   = range.token_end;
            return true;
        }
    }
    return false;
}

CtcDecodeOptions resolve_ctc_decode_options(const ParakeetCtcModel & model,
                                            const std::string      & language) {
    CtcDecodeOptions dopts;
    if (language.empty()) {
        if (!model.ctc_lang_ranges.empty()) {
            throw std::runtime_error(
                "parakeet: this CTC GGUF requires EngineOptions::language "
                "(multilingual language masks present)");
        }
        return dopts;
    }
    // Documented contract: --language / EngineOptions::language is ignored on
    // monolingual CTC GGUFs that do not advertise lang_* ranges.
    if (model.ctc_lang_ranges.empty()) {
        return dopts;
    }
    int32_t start = 0;
    int32_t end   = -1;
    if (!find_ctc_language_range(model, language, start, end)) {
        std::string available;
        for (size_t i = 0; i < model.ctc_lang_ranges.size(); ++i) {
            if (i) available += ",";
            available += model.ctc_lang_ranges[i].id;
        }
        throw std::runtime_error(
            "parakeet: unknown CTC language '" + language + "'"
            " (available: " + available + ")");
    }
    dopts.token_start = start;
    dopts.token_end   = end;
    return dopts;
}

void ctc_greedy_decode_window(const float * logits,
                              int           start_frame,
                              int           end_frame,
                              int           vocab_size,
                              int32_t       blank_id,
                              int32_t     & inout_prev_token,
                              std::vector<int32_t> & out_tokens,
                              std::vector<int>     * out_first_frame,
                              const CtcDecodeOptions * opts) {
    if (start_frame < 0) start_frame = 0;
    if (end_frame < start_frame) end_frame = start_frame;

    const bool mask = opts != nullptr && opts->token_end > opts->token_start;
    int32_t range_start = 0;
    int32_t range_end   = vocab_size;
    if (mask) {
        range_start = opts->token_start;
        range_end   = opts->token_end;
        if (range_start < 0) range_start = 0;
        if (range_end > vocab_size) range_end = vocab_size;
        if (range_start >= range_end) {
            range_start = 0;
            range_end   = vocab_size;
        }
    }

    int32_t prev = inout_prev_token;
    for (int t = start_frame; t < end_frame; ++t) {
        const float * __restrict row = logits + static_cast<size_t>(t) * vocab_size;
        if (mask) {
            int32_t best = range_start;
            float best_score = row[range_start];
            for (int32_t i = range_start + 1; i < range_end; ++i) {
                const float v = row[i];
                if (v > best_score) { best_score = v; best = i; }
            }
            if (blank_id >= 0 && blank_id < vocab_size &&
                (blank_id < range_start || blank_id >= range_end)) {
                const float v = row[blank_id];
                if (v > best_score) { best_score = v; best = blank_id; }
            }
            if (best != blank_id && best != prev) {
                out_tokens.push_back(best);
                if (out_first_frame) out_first_frame->push_back(t);
            }
            prev = best;
            continue;
        }

        int32_t best       = 0;
        float   best_score = row[0];
        // The argmax-with-index reduction has a loop-carried dep on
        // `best_score` / `best` so it doesn't auto-vectorise as cleanly
        // as a plain reduction. `__restrict` + the explicit read into a
        // register at least lets the compiler use a fused max-with-mask
        // pattern on AVX2 / AVX-512. Same shape as the gemv treatment in
        // parakeet_tdt.cpp::gemv_f32.
        #pragma GCC ivdep
        for (int i = 1; i < vocab_size; ++i) {
            const float v = row[i];
            if (v > best_score) { best_score = v; best = i; }
        }
        if (best != blank_id && best != prev) {
            out_tokens.push_back(best);
            if (out_first_frame) out_first_frame->push_back(t);
        }
        prev = best;
    }
    inout_prev_token = prev;
}


bool flash_attn_compiled() { return k_flash_attn_compiled; }

// Hexagon's HMX batched matmuls take only 32-aligned reduction and row counts, so
// the unfused attention zero-pads a head size that is not one; the padded columns
// contribute nothing to any score or output.
int attention_matmul_head_dim(ggml_backend_t backend, int HD) {
    if (!k_f16_activations_compiled || !backend_is_hexagon(backend)) return 0;
    const int padded = hmx_align_up_32(HD);
    return padded > HD ? padded : 0;
}

int model_attention_matmul_head_dim(const ParakeetCtcModel & model) {
    return model.impl ? model.impl->attn.matmul_head_dim : 0;
}

bool f16_activations_compiled() { return k_f16_activations_compiled; }

// Builds one encoder block on the model's active backend and reports whether the
// graph contains `op`: lets a test see which attention path the loader chose.
bool encoder_graph_uses_op(const ParakeetCtcModel & model, enum ggml_op op) {
    constexpr int probe_mel_frames = 128;
    EncoderGraph g;
    if (build_encoder_graph_cached(model, g, probe_mel_frames, probe_mel_frames, model.mel_cfg.n_mels,
                                   /*n_run_layers_override=*/1, /*all_valid=*/true,
                                   model.impl->backend_active) != 0) {
        return false;
    }
    for (int i = 0; i < ggml_graph_n_nodes(g.cgraph); ++i) {
        if (ggml_graph_node(g.cgraph, i)->op == op) return true;
    }
    return false;
}

}
