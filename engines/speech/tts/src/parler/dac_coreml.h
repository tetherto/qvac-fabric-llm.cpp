#pragma once

// Core ML sidecar for Parler's DAC decoder (scripts/parler/export-dac-coreml.py).
//
// The Engine loads the DAC sidecar through the shared Core ML runtime
// (parler_dac_coreml below): a prebuilt `<gguf stem minus quant tag>-dac.mlmodelc`
// next to the GGUF loads in the background; without one, and with a cache
// directory, the sidecar is generated from the GGUF's own DAC tensors
// (template parler-dac-w128) on macOS 15 or later. Until it is ready every
// decode call runs on ggml; each parler_dac_decode call takes its backend once,
// at entry, so a call never mixes backends.
//
// The sidecar is `<gguf stem minus quant tag>-dac.mlmodelc` next to the GGUF.
// It decodes one fixed window of summed RVQ latents, latent [1, latent_dim, W]
// plus a 0/1 frame mask valid [1, 1, W], to pcm [1, 1, W * hop]; frames past
// the end of the utterance are masked out, which reproduces the zero padding
// of a shorter decode. The quantizer runs on the host from the GGUF's
// codebooks, and parler_dac_decode walks the utterance in windows with
// parler_dac_rf_frames of real context on each side, exactly like the ggml
// decode, so a sidecar decode matches a whole-sequence decode within fp16
// tolerance.
//
// Environment:
//   PARLER_COREML_DISABLE        set: never open the sidecar
//   PARLER_COREML_STRICT         set: a missing or failing sidecar is an error
//                                instead of a fallback to ggml
//   PARLER_COREML_COMPUTE_UNITS  all | cpu_and_ane | cpu_and_gpu | cpu_only

#include "coreml_policy.h"
#include "tts-cpp/coreml_options.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace tts_cpp {
namespace parler {
namespace detail {

struct parler_model;

constexpr const char * PARLER_DAC_SIDECAR_SUFFIX = "-dac";
constexpr const char * PARLER_DAC_GGML_BACKEND   = "ggml";
constexpr const char * PARLER_DAC_MIXED_BACKEND  = "mixed";

std::string parler_dac_sidecar_path(const std::string & gguf_path);
bool parler_coreml_disabled();
bool parler_coreml_strict();
// Whether an Engine on `gguf_path` would try the sidecar: Core ML compiled
// in, not disabled, and `<stem>-dac.mlmodelc` present.
bool parler_dac_sidecar_enabled(const std::string & gguf_path);

// The identity of the GGUF parler_load_gguf read the weights from (Core ML
// builds; nothing elsewhere). parler_load_gguf opens the file once and reads
// its metadata, mapping and streamed weights through that one descriptor
// (`fd`, -1 when it could not open one); this records fstat of it.
// parler_dac_coreml::create generates only from that file, so a GGUF swapped
// at the path at any point of the load, or since, is never generated from.
struct parler_dac_source;
void parler_dac_source_record(parler_model & model, int fd, const std::string & path);

// One fixed-window DAC decode. Implementations: the Core ML model, and test
// doubles that exercise the fallback without Core ML.
class parler_dac_sidecar {
public:
    virtual ~parler_dac_sidecar() = default;
    // Frames per call, context included.
    virtual int window() const = 0;
    // "coreml-all", "coreml-ane", ... -- reported as the DAC backend.
    virtual const char * label() const = 0;
    // latent: [latent_dim, window()] channel-major; valid: [window()] 0/1;
    // pcm: [window() * hop]. False on any failure.
    virtual bool decode(const float * latent, const float * valid, float * pcm) = 0;
};

// Host copy of the residual vector quantizer: per codebook the code vectors
// [codebook_size, cb_dim], the 1x1 output projection [latent_dim, cb_dim] and
// its bias [latent_dim].
struct parler_dac_rvq {
    int n_q = 0, codebook_size = 0, cb_dim = 0, latent_dim = 0;
    std::vector<std::vector<float>> codebook, proj, bias;

    // Sums every codebook's projection for frames [f0, f1) of the
    // [n_q, n_frames] code matrix into out[c * stride + (t - f0)].
    void latent(const int32_t * codes, int n_frames, int f0, int f1,
                float * out, int stride) const;
};

// Reads the quantizer tensors from a loaded model (any backend). False when
// they are not F32 or are missing.
bool parler_dac_rvq_load(const parler_model & model, parler_dac_rvq & rvq, std::string * err);

// Opens the sidecar next to `gguf_path` and checks its feature shapes against
// the model. nullptr when Core ML is not compiled in, the sidecar is absent or
// disabled, or it does not match (with the reason in *why, if given).
std::shared_ptr<parler_dac_sidecar> parler_open_dac_sidecar(const std::string & gguf_path,
                                                            const parler_model & model,
                                                            std::string * why = nullptr);
std::shared_ptr<parler_dac_sidecar> parler_open_dac_sidecar(const std::string & gguf_path,
                                                            const parler_model & model,
                                                            const parler_coreml_policy & policy,
                                                            std::string * why = nullptr);

// Attaches `sidecar` (may be null) to the model, loading the host quantizer it
// needs. False, leaving the model on ggml, when the quantizer cannot be read.
bool parler_attach_dac_sidecar(parler_model & model, std::shared_ptr<parler_dac_sidecar> sidecar,
                               std::string * err = nullptr);
bool parler_attach_dac_sidecar(parler_model & model, std::shared_ptr<parler_dac_sidecar> sidecar,
                               const parler_coreml_policy & policy, std::string * err = nullptr);

// Opens and attaches the sidecar next to `gguf_path`. Returns whether one is
// attached; under strict policy a missing or unusable sidecar throws unless
// globally disabled. Legacy overloads resolve the environment at attachment.
bool parler_attach_dac_sidecar(parler_model & model, const std::string & gguf_path);
bool parler_attach_dac_sidecar(parler_model & model, const std::string & gguf_path,
                               const parler_coreml_policy & policy);

// The decode of frames [out_begin, out_end) on `sidecar` (see parler_dac_decode).
// False on any sidecar failure; pcm_out / latent_out are then unspecified.
bool parler_dac_decode_sidecar(const parler_model & model, parler_dac_sidecar & sidecar, const int32_t * codes,
                               int n_frames, std::vector<float> & pcm_out, std::vector<float> * latent_out,
                               int out_begin, int out_end);

// ---- the DAC on the shared Core ML runtime ----------------------------------------

// What one parler_dac_decode call runs on, acquired once at its entry and held
// for the whole call.
struct parler_dac_call {
    std::shared_ptr<parler_dac_sidecar> sidecar;  // null: this call runs on ggml
    bool pending = false;  // on ggml only because the sidecar was not ready yet
    // A prediction failed: retire the sidecar (for every engine on it). The
    // call is then redone on ggml, or fails under strict.
    std::function<void(const std::string & reason)> retire;
};

// The seam parler_dac_decode acquires from when the model has no statically
// attached sidecar (model.dac_pending). The Engine installs parler_dac_coreml;
// tests install fakes.
class parler_dac_pending {
public:
    virtual ~parler_dac_pending() = default;
    // Once per decode call of `out_frames` output frames. A wait for the
    // sidecar returns early once `cancel` (optional) is set. Throws
    // std::runtime_error when use = require and the sidecar cannot serve the
    // call (failed, retired, timed out).
    virtual parler_dac_call acquire(int out_frames, const std::atomic<bool> * cancel) = 0;
    virtual bool strict() const = 0;
};

// Bytes of the host copy of the quantizer (parler_dac_rvq) for `model`;
// computed from tensor shapes, so it works on a metadata-only load.
uint64_t parler_dac_rvq_bytes(const parler_model & model);

// The DAC stage's route, decided from options, OS, build, template match and
// the presence of a prebuilt sidecar, never from the cache, so it is the same
// before and after the first generation. For the generated route it carries
// the Core ML memory envelope fit charges.
struct parler_dac_route {
    std::string route = "none";  // prebuilt | generate | none
    std::string state;           // when none: unsupported | disabled | absent | no_template
    std::string reason;
    std::string compute_units;   // resolved
    bool strict = false;
    bool validated = false;      // generate: the envelope is validated for compute_units
    uint64_t envelope_bytes = 0; // generate: (resident + one transient) x margin
    // generate: the envelope's parts before the margin, from the template.
    uint64_t resident_bytes = 0;      // weights, 2 x I/O, state, workspace and the adapter copies
    uint64_t workspace_bytes = 0;     // the prediction workspace (included in resident_bytes)
    uint64_t prepare_peak_bytes = 0;  // generation and compile transient
    uint64_t load_peak_bytes = 0;     // load and warm-up transient for compute_units
    // generate: the free memory a cold Neural Engine load waits for (runtime
    // admission; fit does not charge it). 0 when compute_units has no ANE.
    uint64_t ane_compile_reserve_bytes = 0;
};

// `adapter_bytes`: engine-side copies the stage keeps (parler_dac_rvq_bytes).
// Throws std::invalid_argument for invalid explicit options.
parler_dac_route parler_dac_plan_route(const std::string & gguf_path, const CoremlOptions & options,
                                       uint64_t adapter_bytes);

// The DAC stage of one Engine on the shared Core ML runtime: the resolved
// options, the route, the slot, the construction group and status.
class parler_dac_coreml final : public parler_dac_pending {
public:
    // Resolves `options` (with the PARLER_* environment) and plans the route.
    // Unless the route is none it builds model.dac_rvq on this thread, creates
    // the slot and issues the construction group (prewarm none: nothing until
    // the first call), waiting for it with wait = construct. `memory_degrade`
    // resolves use = off (fit's memory-degrade). Returns null in builds
    // without TTS_CPP_COREML. Throws std::invalid_argument for invalid
    // options and std::runtime_error when use = require cannot be met.
    static std::shared_ptr<parler_dac_coreml> create(parler_model & model, const std::string & gguf_path,
                                                     const CoremlOptions & options, bool memory_degrade = false);
    ~parler_dac_coreml() override;

    parler_dac_call acquire(int out_frames, const std::atomic<bool> * cancel) override;
    bool strict() const override;

    const parler_dac_route & route() const;
    // Whether a slot exists (route != none).
    bool active() const;
    // The status JSON (schema 1: engine, cache_dir, generation, stages).
    std::string status_json() const;
    // The construction group's result JSON ("{}" before or without one).
    std::string construction_json() const;
    // Blocks until the DAC is ready or terminal, `timeout_ms` passes (< 0: no
    // limit) or `cancel` (optional) is set. True when it is ready. Returns
    // false at once while nothing has requested the sidecar (idle: prewarm
    // none before the first call) or it is deferred(memory).
    // `cancels` (when set) ends the wait once it differs from `since`.
    bool wait(int64_t timeout_ms, const std::atomic<uint64_t> * cancels = nullptr, uint64_t since = 0) const;

    struct impl;
private:
    explicit parler_dac_coreml(std::unique_ptr<impl> p);
    std::unique_ptr<impl> p_;
};

// Label merge across the calls of one synthesis: equal labels stay, anything
// else is "mixed" (an empty accumulator takes the next label).
std::string parler_merge_dac_backend(const std::string & acc, const std::string & next);

} // namespace detail
} // namespace parler
} // namespace tts_cpp
