#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "mtmd-impl.h"
#include "mtmd-model.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <random>
#include <vector>

// !!! Internal header, to be used by mtmd only !!!

// runs the graphs of one mtmd_model on the shared backends and scheduler

enum mtmd_flash_attn_type {
    MTMD_FLASH_ATTN_TYPE_AUTO     = -1,
    MTMD_FLASH_ATTN_TYPE_DISABLED = 0,
    MTMD_FLASH_ATTN_TYPE_ENABLED  = 1,
};

struct mtmd_subctx_params {
    enum mtmd_flash_attn_type flash_attn_type;
    bool warmup;
};

// the device the weights and the graphs are placed on, nullptr means CPU
// backend_name (or the MTMD_BACKEND_DEVICE env var) takes precedence over device, falling back if not found
ggml_backend_dev_t mtmd_select_device(bool use_gpu, ggml_backend_dev_t device, const char * backend_name = nullptr);

// shared by the subcontexts, not thread-safe
struct mtmd_backends {
    ggml_backend_t backend     = nullptr; // backend_cpu if no accelerator
    ggml_backend_t backend_cpu = nullptr;

    // for ggml_backend_sched_new(), backend_cpu last
    std::vector<ggml_backend_t>             ptrs;
    std::vector<ggml_backend_buffer_type_t> bufts;

    std::shared_ptr<ggml_threadpool> threadpool; // can be nullptr

    // its buffers fit the largest graph reserved by the subcontexts
    ggml_backend_sched_ptr sched;

    // device nullptr means CPU
    mtmd_backends(ggml_backend_dev_t device, int n_threads, int max_nodes,
                  ggml_backend_sched_eval_callback cb_eval, void * cb_eval_user_data);
    ~mtmd_backends();

    std::map<ggml_backend_dev_t, size_t> get_mem_compute() const;

    mtmd_backends(const mtmd_backends &) = delete;
    mtmd_backends & operator=(const mtmd_backends &) = delete;
};

// compute the log-mel of n_frames frames, the waveform must already be padded and window must hold one whole frame
// out has the layout [n_frames, n_mel]; must not run at the same time as another pass
struct mtmd_audio_mel_config;
bool mtmd_audio_log_mel(const mtmd_audio_mel_config & config, const float * samples, const float * window, int64_t n_frames, float * out);

enum mtmd_subctx_gen_process {
    MTMD_SUBCTX_GEN_PROCESS_GEN_UNKNOWN,
    MTMD_SUBCTX_GEN_PROCESS_GEN_CODE, // h_state to codes
    MTMD_SUBCTX_GEN_PROCESS_GEN_WAV,  // codes to raw PCM audio
    MTMD_SUBCTX_GEN_PROCESS_TTS,        // full utterance of codes to raw PCM audio
    MTMD_SUBCTX_GEN_PROCESS_TTS_VOCODE, // internal: mel + source stft to istft input
    MTMD_SUBCTX_GEN_PROCESS_TOKENIZE,   // raw PCM audio to semantic speech tokens
};

// persistent state of a causal audio encoder, for streaming input (one per stream)
// it holds the sliding window of past keys/values, so each encode() call only takes the new frames
struct mtmd_subctx_enc_state {
    int32_t n_ctx  = 0; // ring slots: attention window + frames per call
    int32_t n_past = 0; // absolute position of the next frame

    std::vector<int32_t> slot_pos;  // absolute position held by each slot, -1 if empty
    std::vector<float>   stem_tail; // last conv stem output frame, empty on a cold start

    ggml_context_ptr        ctx_data;
    ggml_backend_buffer_ptr buf;
    std::vector<ggml_tensor *> k; // per layer [n_embd_kv, n_ctx]
    std::vector<ggml_tensor *> v;

    // lfm3a: same order as list_lfm3a_state_slots(), F32 [ne0, ne1], updated in place
    std::vector<ggml_tensor *> slots;

    // nemotron transcribe: the window's normalized log-mel, cache and FIFO rows in; its speaker rows and predictions out
    std::vector<float> mel_norm;
    std::vector<float> spk_prev;
    std::vector<float> spk_chunk, spk_preds;
};

// per-sequence state of audio generation: RNG and the slots kept between GEN_WAV calls
struct mtmd_subctx_gen_state {
    std::mt19937 rng;

    ggml_context_ptr        ctx_data;
    ggml_backend_buffer_ptr buf;
    std::vector<ggml_tensor *> slots; // same order as list_gen_state_slots(), F32 [ne0, ne1]

    // clears the slots and keeps the RNG stream
    void reset() {
        if (buf) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    }
};

struct mtmd_subctx_encode_params {
    int n_threads = 1;
    const mtmd_image_f32_batch * imgs = nullptr;
    std::vector<float> * out_embd = nullptr;

    // for audio gen, imgs has exactly one entry: hidden state from backbone (GEN_CODE) or unused (GEN_WAV)
    mtmd_subctx_gen_process gen_process = MTMD_SUBCTX_GEN_PROCESS_GEN_UNKNOWN;

    // GEN_CODE: out_embd receives the embd to feed back to the backbone
    int32_t code0 = 0; // semantic code sampled by the backbone
    int32_t top_k = 50;
    float   top_p = 1.0f;
    std::vector<int32_t> * out_codes = nullptr; // this frame's 16 sampled codes
    std::vector<float> * out_feats = nullptr; // continuous counterpart of out_codes
    float   temp = 0.0f;                      // sampling temperature, noise scale for flow-matching decoders
    bool * out_is_eos = nullptr;

    // TOKENIZE: speech embedding rows of the produced codes
    std::vector<float> * out_code_embd = nullptr;

    // GEN_WAV
    const std::vector<int32_t> * codes = nullptr;     // this frame's 16 RVQ codes
    const std::vector<float> *   feats = nullptr;     // continuous counterpart of codes
    std::vector<float> * out_audio = nullptr;         // decoded PCM samples, F32
    const mtmd_gen_ref * ref = nullptr; // TTS reference conditioning of the voice
    // TTS_VOCODE internal stage inputs
    const std::vector<float> * mel_in   = nullptr;
    const std::vector<float> * sstft_in = nullptr;
    int vocode_n_mel  = 0;
    int vocode_n_stft = 0;
    std::vector<float> * out_spec = nullptr;

    // audio gen: state of one sequence, encode() updates it in place
    struct mtmd_subctx_gen_state * gen_state = nullptr;

    // streaming audio input, updated in place by encode()
    struct mtmd_subctx_enc_state * enc_state = nullptr;
};

struct mtmd_subcontext {
    const mtmd_model & model;
    mtmd_backends & backends;

    std::vector<uint8_t> buf_compute_meta;

    int max_nodes = 8192;
    mtmd_flash_attn_type flash_attn_type = MTMD_FLASH_ATTN_TYPE_AUTO;
    // Whether the active backend can run GGML_OP_FLASH_ATTN_EXT at all. Probed
    // once at warmup; in AUTO mode the per-image decision falls back to the
    // explicit attention path when this is false.
    bool flash_attn_supported = true;
    bool is_allocated = false;

    // Inputs to the AUTO flash-attn budget decision, resolved once (getenv +
    // Mali detection + device-memory query) and cached, since none of them
    // change between images. Populated lazily on the first AUTO resolve.
    bool   fa_budget_cached = false;
    int    fa_auto_min_kv   = 0;   // explicit-attention cutoff (n_patches); <=0 disables it
    size_t fa_mem_total     = 0;   // device total memory, stable clamp input (0 = unknown)
    size_t fa_mem_free      = 0;   // device free memory at first resolve, hard-fit check only (0 = unknown)
    // Set at warmup when the backend reports it lacks efficient (coopmat) FA
    // (ggml_backend_supports_efficient_fa == false, e.g. Mali via ggml-vulkan).
    // Turns on the AUTO cutoff default so the budget heuristic prefers the
    // explicit path for short sequences but keeps scalar FA for huge ones.
    bool   fa_backend_inefficient = false;

    bool debug_output_embeddings = false;

    bool support_batch = false;

    // model and backends must outlive the subcontext
    mtmd_subcontext(const mtmd_model & model, mtmd_backends & backends, const mtmd_subctx_params & params);

    bool encode(mtmd_subctx_encode_params * params);

    // reserve the compute buffers for the given batch
    void warmup(const mtmd_image_f32_batch & batch);

    // TODO: remove encode_image() and always use batched version
    bool encode_image(int n_threads, const mtmd_image_f32 * img, std::vector<float> & out_vec);
    bool encode_batch(int n_threads, const mtmd_image_f32_batch * imgs, std::vector<float> & out_batch_embd);

    // n_frames_max is the maximum number of encoder frames per encode() call, free the result with delete
    mtmd_subctx_enc_state * enc_state_init(int32_t n_frames_max);

    // seed UINT32_MAX means random, free the result with delete
    mtmd_subctx_gen_state * gen_state_init(uint32_t seed);

private:
    struct support_info_op {
        ggml_tensor * op;

        // true if the op runs on the accelerated backend
        bool is_accel = true;
    };

    struct support_info_graph {
        // whether the backend supports flash attention
        bool fattn = true;
        ggml_tensor * fattn_op = nullptr; // for debugging

        std::vector<support_info_op> ops;
    };

    mtmd_image_f32_batch get_dummy_batch() const;

    // only initialize backend buffers, but do not allocate them yet
    support_info_graph reserve_compute_meta(const mtmd_image_f32_batch & batch);
};

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
inline int mtmd_fa_effective_min_kv(int auto_min_kv, size_t total_mem, size_t free_mem, int n_head) {
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
