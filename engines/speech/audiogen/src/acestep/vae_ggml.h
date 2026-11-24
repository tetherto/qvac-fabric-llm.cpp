#pragma once

// ACE-Step Oobleck VAE — ggml compute engine.
//
// Encoder (audio -> 64-ch latent) and decoder (latent -> 48 kHz stereo audio)
// as ggml graphs. The decoder's transposed convolutions use ggml_col2im_1d and
// every stage uses ggml_snake -- the two custom ops landed in the fabric
// ggml. Everything else is stock ggml. F32 activations, bf16 weights fused to F16.
//
// Downsample / upsample factor is 1920 (= 2*4*4*6*10). Memory scales with the
// audio length, so callers should decode/encode bounded windows on CPU.

#include "ggml-backend.h"

#include <functional>
#include <string>
#include <vector>

namespace tts_cpp::acestep {

struct VaeModel;             // opaque: weight tensors + backend weight buffer
struct AcestepStageMeasure;  // fit_measure.h

// Load VAE weights from `path` onto `backend` (borrowed). Loads the decoder
// always; the encoder only when `with_encoder` (needed for the reconstruction
// roundtrip, not for pure DiT-latent decode). Returns nullptr on failure.
VaeModel * vae_model_load(const std::string & path, ggml_backend_t backend, bool with_encoder, bool verbose);
void       vae_model_free(VaeModel * m);
bool       vae_model_has_encoder(const VaeModel * m);
size_t     vae_model_weight_bytes(const VaeModel * m);

// Metadata-only load for the memory-fit preflight: identical tensor wiring to
// vae_model_load, but the weight allocation is SIZED into `measure` instead of
// performed and no tensor data is read. Only good for the measure calls below;
// free with vae_model_free.
VaeModel * vae_model_load_metadata_only(const std::string & path, ggml_backend_t backend, bool with_encoder,
                                        bool verbose, AcestepStageMeasure & measure);

// The chunked-decode window core this model/backend would use (the same
// backend-adaptive probe vae_model_decode runs, ACESTEP_VAE_WIN_CORE
// included). Exposed for the memory-fit preflight and its parity tests.
int vae_model_decode_window_frames(VaeModel * m);

// Size-only twins of one decode window / one encode window, for the memory-fit
// preflight: build the identical graph the real call would use for a
// T_latent-frame decode (its worst resident window when chunked) or a
// `frames`-frame encode window, and price it with a plain gallocr on m's
// default buffer type -- `backend_bytes` on m's backend, `host_input_bytes`
// the host-side graph input the real sched stages on its CPU (last) backend (0
// when m's backend IS the CPU). Exact on CPU; on GPU a bounded overcount of at
// most one device-side input slot (see vae_measure_graph). Nothing is
// allocated, uploaded, or computed. Returns false on graph/allocator
// construction failure.
bool vae_model_measure_decode(VaeModel * m, int T_latent, size_t & backend_bytes, size_t & host_input_bytes);
bool vae_model_measure_encode(VaeModel * m, int frames, size_t & backend_bytes, size_t & host_input_bytes);

// Scheduler compute bytes of the most recent real decode window / encode (sum
// over the sched's backends; 0 before the first). For the fit parity tests.
size_t vae_model_compute_buffer_bytes(const VaeModel * m);

// Decode a 64-ch latent (time-major, latent[t*64 + c]) into interleaved stereo
// 48 kHz PCM. Returns T_audio frames (= T_latent * 1920) or -1 on failure.
//
// `on_node` (optional) fires once per computed graph node with (done, total)
// node counts, for VAE-stage progress; return false to cancel (returns -1).
int vae_model_decode(VaeModel * m, const float * latent, int T_latent, std::vector<float> & pcm_out,
                     const std::function<bool(int done, int total)> & on_node = {});

// Encode interleaved stereo PCM (frames*2 samples, 48 kHz) into the 64-ch mean
// latent (time-major, out[t*64 + c]). Returns T_latent or -1 on failure or
// cancellation through `on_node`.
int vae_model_encode(VaeModel * m, const float * pcm, int frames,
                     std::vector<float> & latent_out,
                     const std::function<bool(int done, int total)> & on_node = {});

// Decode-progress percentage from the per-node eval callback. `total` is
// ggml_graph_n_nodes(gf), but a GPU+CPU scheduler can insert extra copy/split
// nodes, so `done` may exceed `total`; both are clamped so the reported value is
// monotone and bounded to [0, 100] (no progress-bar overshoot). Pure + header-
// only so the throttle/clamp logic is unit-tested without a GGUF fixture.
inline int vae_progress_pct(int done, int total) {
    if (total <= 0) return 0;
    if (done < 0) done = 0;
    if (done > total) done = total;
    return (int) ((long long) done * 100 / total);
}

// Shrink a chunked-decode window so the graph's largest single node fits one
// backend allocation. ggml_gallocr already spreads an arena over several buffers
// when a backend caps allocation size, but it cannot split a tensor, so the
// biggest node has to fit on its own. Node size is linear in the window, so the
// measured `peak_bytes` at `core + 2*overlap` frames gives the scale directly.
// Returns `core` unchanged when it already fits. Pure + header-only so the
// sizing is unit-tested without a GPU.
inline int vae_shrink_window_core(int core, int overlap, size_t peak_bytes, size_t max_alloc, int core_min) {
    if (peak_bytes <= max_alloc || core <= core_min) return core;
    const int win = core + 2 * overlap;
    const int fit = (int) ((double) win * (double) max_alloc / (double) peak_bytes) - 2 * overlap;
    if (fit < core_min) return core_min;
    return fit < core ? fit : core - 1;
}

} // namespace tts_cpp::acestep
