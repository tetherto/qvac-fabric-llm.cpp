#pragma once

#include "ggml.h"
#include "mtmd.h"

#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
#include <algorithm>
#include <cmath>
#include <map>
#endif

// !!! Internal header, to be used by mtmd only !!!

#define MTMD_INTERNAL_HEADER

struct clip_ctx;

struct clip_image_size {
    int width;
    int height;
    bool operator==(const clip_image_size & other) const {
        return width == other.width && height == other.height;
    }
    bool operator!=(const clip_image_size & other) const {
        return !(*this == other);
    }
    int area() const {
        // avoid overflow when computing area
        GGML_ASSERT(width  >= 0 && width  <= 46000);
        GGML_ASSERT(height >= 0 && height <= 46000);
        return width * height;
    }
};

struct clip_image_f32;
struct clip_image_f32_batch;

enum clip_modality {
    CLIP_MODALITY_VISION,
    CLIP_MODALITY_AUDIO,
    CLIP_MODALITY_GEN_AUDIO,
};

enum clip_flash_attn_type {
    CLIP_FLASH_ATTN_TYPE_AUTO     = -1,
    CLIP_FLASH_ATTN_TYPE_DISABLED = 0,
    CLIP_FLASH_ATTN_TYPE_ENABLED  = 1,
};

// WARNING: value 0 is BATCHED, which is NOT the default mode (sequential is).
// A zero-initialized clip_context_params/mtmd_context_params (`{}`, memset, calloc)
// therefore selects BATCHED — the one-forward-pass path whose ne[3] batching is not
// yet verified on every backend (see the NOTE in models/qwen3vl.cpp). Do not rely on
// zero-init for the default: initialize via mtmd_context_params_default() (sets
// sequential) or set image_tile_mode explicitly. These values are part of the shipped
// API (string "0"/"1"/"2" in consumers), so they are intentionally not renumbered.
enum clip_image_tile_mode {
    CLIP_IMAGE_TILE_MODE_BATCHED    = 0, // NOT the default; zero-init lands here — see warning above
    CLIP_IMAGE_TILE_MODE_SEQUENTIAL = 1, // the default (via mtmd_context_params_default)
    CLIP_IMAGE_TILE_MODE_DISABLED   = 2,
};

struct clip_context_params {
    bool use_gpu;
    ggml_backend_dev_t device;
    enum clip_flash_attn_type flash_attn_type;
    int image_min_tokens;
    int image_max_tokens;
    bool warmup;
    bool has_bf16_weights;
    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;
    bool no_alloc;
    mtmd_progress_callback progress_callback;
    void * progress_callback_user_data;
    const char * backend_device; // optional, if null will use env var or default GPU backend
    int image_tile_mode;   // 0=batched, 1=sequential (default), 2=disabled. NOTE: 0 (batched) is the zero value but NOT the default — init via mtmd_context_params_default() or set explicitly.
    int image_max_tiles;   // override preproc_max_tiles; -1 or 0 = use GGUF/model default (only a positive value overrides)
    // override preproc_no_upscale (idefics3-style preprocessing); -1 = use GGUF/model
    // default, 0 = off, 1 = on. WARNING: a zero-initialized struct lands on 0 (off), not
    // -1 (model default), the one case that silently forces base-style preprocessing
    // on a Flash-style model. Do not rely on zero-init: use mtmd_context_params_default()
    // or set this field explicitly.
    int image_no_upscale;
};

struct clip_init_result {
    struct clip_ctx * ctx_v; // vision context
    struct clip_ctx * ctx_a; // audio context
    struct clip_ctx * ctx_gen_a; // audio generation context
};

struct clip_init_result clip_init(const char * fname, struct clip_context_params ctx_params);

void clip_free(struct clip_ctx * ctx);

// TODO: should be enum, not string
const char * clip_patch_merge_type(const struct clip_ctx * ctx);

int clip_n_output_tokens(const clip_ctx * ctx, const clip_image_f32 * img);

// for M-RoPE, this will be the number of token positions in X and Y directions
// for other models, X will be the total number of tokens and Y will be 1
int clip_n_output_tokens_x(const clip_ctx * ctx, const clip_image_f32 * img);
int clip_n_output_tokens_y(const clip_ctx * ctx, const clip_image_f32 * img);

// this should be equal to the embedding dimension of the text model
int clip_n_mmproj_embd(const struct clip_ctx * ctx);

// TODO: remove clip_image_encode() and always use batched version
bool clip_image_encode      (struct clip_ctx * ctx, int n_threads, const clip_image_f32 * img, std::vector<float> & out_vec);
bool clip_image_batch_encode(struct clip_ctx * ctx, int n_threads, const struct clip_image_f32_batch * imgs, std::vector<float> & out_batch_embd);

enum clip_gen_process_type {
    CLIP_GEN_PROCESS_GEN_UNKNOWN,
    CLIP_GEN_PROCESS_GEN_CODE, // h_state to codes
    CLIP_GEN_PROCESS_GEN_WAV,  // codes to raw PCM audio
};
struct clip_encode_params {
    int n_threads = 1;
    const clip_image_f32_batch * imgs = nullptr;
    std::vector<float> * out_embd = nullptr;

    // for audio gen, imgs has exactly one entry: hidden state from backbone (GEN_CODE) or unused (GEN_WAV)
    clip_gen_process_type gen_process = CLIP_GEN_PROCESS_GEN_UNKNOWN;

    // GEN_CODE: out_embd receives the embd to feed back to the backbone
    int32_t code0 = 0; // semantic code sampled by the backbone
    int32_t top_k = 50;
    float   top_p = 1.0f;
    std::vector<int32_t> * out_codes = nullptr; // this frame's 16 sampled codes
    std::vector<float> * out_feats = nullptr; // continuous counterpart of out_codes
    uint32_t seed = UINT32_MAX;               // UINT32_MAX for random
    float   temp = 0.0f;                      // sampling temperature, noise scale for flow-matching decoders
    bool * out_is_eos = nullptr;

    // GEN_WAV
    const std::vector<int32_t> * codes = nullptr;     // this frame's 16 RVQ codes
    const std::vector<float> *   feats = nullptr;     // continuous counterpart of codes
    std::vector<float> * out_audio = nullptr;         // decoded PCM samples, F32
    const std::vector<uint8_t> * state_in  = nullptr; // state from previous call, null or wrong size means cold start
    std::vector<uint8_t> *       state_out = nullptr; // state for the next call
};
bool clip_encode(struct clip_ctx * ctx, struct clip_encode_params * params);

bool clip_is_llava(const struct clip_ctx * ctx);
// note for contributor: this clip_is_(model) pattern is deprecated
//                       do NOT add new functions like this

bool clip_has_vision_encoder(const struct clip_ctx * ctx);
bool clip_has_audio_encoder(const struct clip_ctx * ctx);
bool clip_has_whisper_encoder(const struct clip_ctx * ctx);

bool clip_support_batch(const struct clip_ctx * ctx);

int clip_model_n_temporal_merge(const struct clip_ctx * ctx); // TODO @ngxson : remove, refactor this

#ifdef __cplusplus
// qvac: per-device memory usage of an initialised clip_ctx — weight buffers
// summed with the scheduler's compute reservations. Used by mtmd_get_memory_usage
// (and ultimately by common/fit.cpp's heuristic). Restored from upstream b9341.
std::map<ggml_backend_dev_t, size_t> clip_get_mem_usage(const struct clip_ctx * ctx);

// QVAC-21914: pure arithmetic of the flash-attention AUTO budget decision.
// Returns the effective explicit-attention cutoff in n_patches given the
// configured cutoff and the device memory probe:
//
//  - total memory provides the STABLE fast-path clamp (session-independent —
//    the explicit scratch, ~3*n^2*n_head*4 bytes, must fit in a quarter of
//    total): normal-size images keep the fast explicit path regardless of
//    momentary memory pressure.
//  - free memory (a volatile, load-dependent number) may only LOWER the cutoff
//    further, and only by the hard-fit requirement: when reported, the explicit
//    scratch must also fit in what is actually free right now. It can never
//    extend the explicit path beyond the total-memory clamp.
//  - neither reported: fail SAFE toward memory-frugal FA with a conservative
//    constant cap instead of the raw default.
//
// Defined inline here (not out-of-line in clip.cpp) so unit tests can exercise
// it without linking an unexported symbol across the Windows mtmd.dll boundary
// — the internal clip_* API carries no MTMD_API export decoration.
inline int clip_fa_effective_min_kv(int auto_min_kv, size_t total_mem, size_t free_mem, int n_head) {
    // Conservative cutoff cap when the device reports no memory information at
    // all: 2048 patches keeps the explicit scratch around ~0.8 GB at n_head=16
    // instead of trusting the raw 4096 default (~3.2 GB) blind.
    constexpr int NO_MEMINFO_CAP = 2048;
    if (auto_min_kv <= 0) {
        return auto_min_kv;
    }
    const double heads = (double) (n_head > 0 ? n_head : 1);
    int eff_min_kv = auto_min_kv;
    if (total_mem > 0) {
        // Stable clamp: explicit scratch (3 * n^2 * n_head * 4) <= total/4
        // =>  n <= sqrt((total/2) / (24*n_head))
        const double n_max_total = std::sqrt(((double) total_mem / 2.0) / (24.0 * heads));
        eff_min_kv = std::min(eff_min_kv, (int) n_max_total);
    }
    if (free_mem > 0) {
        // Hard fit: 3 * n^2 * n_head * 4 <= free  =>  n <= sqrt(free / (12*n_head))
        const double n_max_free = std::sqrt((double) free_mem / (12.0 * heads));
        eff_min_kv = std::min(eff_min_kv, (int) n_max_free);
    }
    if (total_mem == 0 && free_mem == 0) {
        eff_min_kv = std::min(eff_min_kv, NO_MEMINFO_CAP);
    }
    return eff_min_kv;
}
#endif

struct clip_cap {
    bool has_vision;
    bool has_audio;
};
struct clip_cap clip_get_cap(const char * fname);
