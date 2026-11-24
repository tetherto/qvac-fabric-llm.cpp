#pragma once

// Persistent CosyVoice3 engine (Fun-CosyVoice3-0.5B / 1.5B).
//
// Native C++/ggml implementation of Fun-CosyVoice3 on CPU or the validated
// per-platform GPU paths: Metal (macOS / iOS), Vulkan (desktop Linux /
// Windows), and OpenCL (Android / Adreno). The Hexagon NPU runs only on
// explicit request (EngineOptions::backend). It uses the same persistent
// Engine shape as the other synthesis families and returns 24 kHz speech.
//
// Pipeline:
//   text --(Qwen2 BPE tokenizer)--> Qwen2.5 LM --> speech tokens [0, 6561)
//        --> DiT conditional-flow-matching (Euler ODE) --> mel
//        --> CausalHiFT vocoder --> 24 kHz mono PCM
//   Zero-shot voice cloning adds: reference audio --> speech_tokenizer_v3
//   (prompt tokens) + CAM++ (192-d speaker embedding) + prompt mel, all
//   computed natively at construction when reference_audio is set; otherwise
//   the baked voice.gguf tensors are used.  With a reference transcript
//   (prompt_text) the LM is prompted zero-shot; without one the engine runs
//   the cross-lingual variant (timbre only, no LM prompt tokens).  Instruct
//   mode (VoiceControls) selects an emotion / pace, or a raw dialect / volume /
//   style instruction, mirroring CosyVoice3 inference_instruct2.
//
// CosyVoice3 ships as a small set of GGUFs (llm / flow / hift / s3tok /
// campplus / voices).  Point `model_dir` at the folder that holds them, or
// set the per-component paths explicitly.
//
// Usage:
//
//     using tts_cpp::cosyvoice::Engine;
//     using tts_cpp::cosyvoice::EngineOptions;
//
//     EngineOptions opts;
//     opts.model_dir     = "models/cosyvoice3-0.5b";
//     opts.n_gpu_layers  = 0;                 // 0 = CPU; >0 = Metal / OpenCL / Vulkan
//
//     Engine engine(opts);
//     auto result = engine.synthesize("Hello world.");
//     write_wav(result.pcm, result.sample_rate);   // 24 kHz mono float32
//
// Threading model (mirrors the sibling engines):
//   - synthesize() on the same Engine instance is NOT safe to call
//     concurrently.
//   - synthesize() on different Engine instances from different threads is
//     safe.
//   - cancel() is safe from any thread.
//
// Implemented in src/cosyvoice_engine.cpp.

#include "tts-cpp/backend.h"
#include "tts-cpp/export.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::cosyvoice {

// Conditioning controls for one synthesis (CosyVoice3 inference_instruct2).
//
// CosyVoice3 is trained on exactly ONE instruction per synthesis.  A channel is
// ENGAGED when its value resolves to a non-empty instruction; only "moderate"
// disengages one.  Engaging two or more throws naming each one -- silently
// dropping a control would ship conditioning nobody asked for.
struct VoiceControls {
    // One of controls::supported_emotions(controls::EngineId::CosyVoice):
    // anger | happy | neutral | sad.  Case-insensitive.  The other canonical
    // emotions -- and the upstream spelling "angry" -- throw: Fun-CosyVoice3
    // has no trained instruction for them and we do not paraphrase one.
    // Every value here, "neutral" included, engages the instruct path.
    std::string emotion;

    // One of controls::supported_paces(controls::EngineId::CosyVoice):
    // slow | moderate | fast.  Case-insensitive.  "moderate" emits NO
    // instruction and so engages nothing; CosyVoice3 has no middle step.
    std::string pace;

    // Escape hatch for the controls with no canonical vocabulary yet (dialect /
    // volume / style), e.g. "请用广东话表达。".  Used verbatim: pass the bare
    // sentence, the engine adds the wrapper and <|endofprompt|>.
    std::string instruct_text;

    // Inline on purpose: the library builds with hidden visibility, so an
    // out-of-line method would not resolve for shared-lib consumers.
    bool empty() const {
        return emotion.empty() && pace.empty() && instruct_text.empty();
    }
};

struct EngineOptions {
    // Directory holding the standard CosyVoice3 GGUFs
    // (cosyvoice3-{llm,flow,hift,s3tok,campplus,voices}-*.gguf).  Either set
    // this, or set the per-component paths below explicitly (explicit paths
    // win over the ones discovered under model_dir).
    std::string model_dir;

    std::string llm_gguf_path;       // Qwen2.5 LM (text -> speech tokens)
    std::string flow_gguf_path;      // DiT conditional-flow-matching (tokens -> mel)
    std::string hift_gguf_path;      // CausalHiFT vocoder (mel -> 24 kHz PCM)
    std::string s3tok_gguf_path;     // supervised S3 speech tokenizer (zero-shot prompt)
    std::string campplus_gguf_path;  // CAM++ speaker encoder (zero-shot)

    // Qwen2 byte-level BPE text frontend (vocab.json + merges.txt) and the
    // baked default-voice GGUF (voice/prompt_stok, prompt_token, prompt_feat,
    // embedding — see scripts/bake-cosyvoice3-voice.py).  Resolved under
    // model_dir when left empty (vocab.json / merges.txt / voice.gguf).
    std::string vocab_path;
    std::string merges_path;
    std::string voice_gguf_path;

    // ---- Zero-shot / cross-lingual voice cloning (optional) ----
    // reference_audio: path to a mono recording of the target speaker
    // (0.5-30 s hard limits; 5-15 s of clean speech recommended).  At
    // construction the native front-end tokenizes it (speech_tokenizer_v3),
    // extracts the CAM++ speaker embedding and the prompt mel, and replaces
    // the baked voice.gguf tensors.  Requires the s3tok + campplus GGUFs
    // (resolved from model_dir as cosyvoice3-s3tok*.gguf /
    // cosyvoice3-campplus*.gguf when the explicit paths above are empty);
    // construction THROWS if they are missing or the audio is unusable —
    // there is deliberately no silent fallback to the baked voice.
    //
    // prompt_text selects the cloning mode, mirroring the upstream frontends:
    //  - verbatim transcript of the reference => zero-shot (the LM is
    //    prompted with transcript + reference speech tokens; best fidelity
    //    when synthesizing the same language as the reference);
    //  - empty => cross-lingual (timbre-only conditioning through the flow;
    //    the LM gets no reference prompt, best when the target language
    //    differs from the reference).
    // Without reference_audio, prompt_text still overrides the baked voice's
    // transcript metadata (voice.gguf voice.prompt_text) for the LM prompt.
    std::string reference_audio;
    std::string prompt_text;

    // Conditioning applied by the synthesize() overloads that take no
    // VoiceControls.  Validated in the constructor, so a bad default fails at
    // load rather than at first synthesis.  When it resolves to an instruction
    // the LM is conditioned on it and drops its prompt speech tokens; the
    // baked/selected voice still supplies the timbre.  All-empty (default) =
    // zero-shot (prompt_text / baked-voice transcript path).
    VoiceControls default_controls;

    // Reserved: named-voice selection from a multi-voice voices.gguf is not yet
    // wired.  Use voice_gguf_path to point at a single baked voice.gguf.
    std::string voice;

    // Language hint.  Reserved: the text-normalization frontend (wetext) is not
    // yet integrated, so this is currently accepted but not acted on.
    std::string language = "en";

    int seed         = 42;
    int n_threads    = 0;   // 0 = leave the backend's default thread pool.
    int n_gpu_layers = 0;   // 0 = CPU; >0 selects the GPU path (Metal on Apple,
                            // OpenCL/Adreno on Android, Vulkan on desktop;
                            // others fall back to CPU).

    // Explicit compute device. Empty or "auto" (default) keeps the
    // n_gpu_layers policy walk. "cpu", "opencl", "hexagon" (the Snapdragon
    // HTP0 NPU) or an exact ggml device name select that device and override
    // n_gpu_layers; there is no fallback, so the constructor throws when it
    // is missing. The automatic walk never picks Hexagon.
    std::string backend;

    // Vulkan adapter index, consulted only when the selection walk lands on
    // Vulkan.  0 (default) = first adapter in registry order; N > 0 = the Nth
    // adapter (0-indexed), throwing on out-of-range so a CLI typo fails loud
    // instead of silently falling back to CPU; -1 = auto-pick argmax(free
    // VRAM) with a UMA bias that skips integrated adapters whenever a
    // discrete one is visible (see backend_selection.h).
    int vulkan_device = 0;

    // Argmax speech-token decode instead of RAS nucleus sampling.  Sampling is
    // chaotic in the logits -- a 1-ulp difference re-rolls the whole token
    // trajectory -- so greedy is what makes a CPU-vs-GPU comparison meaningful.
    // Off by default: sampling is the better-sounding production path.
    bool greedy      = false;

    // Treat the baked-voice prompt frames as attention conditioning only
    // inside the flow DiT (queries and feed-forward skip them).  Faster flow,
    // but the prompt hidden states stop updating per layer, so output
    // deviates slightly from the PyTorch reference.  Off by default.
    bool flow_cut_prompt = false;

    // Pin the LM speech-token trajectory instead of decoding one.  Sampling is
    // chaotic in the logits, so two backends will not agree on tokens even at
    // the same seed; pinning them is what makes the flow + vocoder stages
    // comparable across backends, and what makes two benchmark legs do provably
    // equal work.  Empty (the default) = decode normally.
    std::vector<int> force_speech_tokens;

    // Reserved: output resampling is not yet implemented.  CosyVoice3 output is
    // always the native 24 kHz (reported on SynthesisResult::sample_rate),
    // regardless of this value.
    int output_sample_rate = 0;

    // Reserved: the DiT flow currently runs a fixed 10 Euler steps (the
    // parity-validated schedule); this override is not yet honored.
    int cfm_steps = 0;

    // ---------------- Streaming synthesis ----------------------------
    //
    // When `stream_chunk_tokens > 0` AND a non-empty callback is passed to
    // synthesize(), the callback is driven progressively with 24 kHz PCM in
    // chunks.  The returned SynthesisResult.pcm still holds the full concatenated
    // audio (the callback is an *addition*, not a replacement), so
    // `result.pcm == concat(callback chunks)` holds.  Batch path when
    // stream_chunk_tokens == 0 OR the callback is empty.
    //
    // NOTE (iteration 1): the audio is computed in full first, then the callback
    // is driven over it in chunks.  The native token-by-token token2wav loop that
    // would make early chunks *arrive* early (true low first-audio latency), and
    // stream_left_context_tokens (per-chunk cost bounding), are reserved and not
    // yet realized.
    //
    //   stream_chunk_tokens        Speech tokens per emitted chunk (0 = batch).
    //   stream_first_chunk_tokens  Size of the first emitted chunk (0 = same as
    //                              stream_chunk_tokens).
    //   stream_left_context_tokens Reserved (see note); not yet used.
    int stream_chunk_tokens        = 0;
    int stream_first_chunk_tokens  = 0;
    int stream_left_context_tokens = 0;

    // Directory to scan for dynamically-loaded ggml backend plugins.  Consulted
    // only when n_gpu_layers > 0.  Process-global and first-Engine-wins.
    std::string backends_dir;

    // Writable directory for the OpenCL program-binary cache (Android).  Empty
    // leaves ggml's default, which on Android resolves under HOME and is inert.
    std::string opencl_cache_dir;
};

// Per-chunk PCM callback for streaming synthesis.  Receives `samples`
// consecutive float32 mono samples at SynthesisResult::sample_rate (24 kHz by
// default).  The buffer is owned by the engine and must not be retained past
// the callback; copy out if you need the data.
//   `chunk_index`  0-based; increments by exactly 1 per chunk.
//   `is_last`      true only on the final chunk.
// Throwing from this callback aborts synthesis (the exception propagates out
// of synthesize()).
using StreamCallback = std::function<void(
    const float * pcm, std::size_t samples, int chunk_index, bool is_last)>;

// Native sample rate (Hz) the CosyVoice3 CausalHiFT vocoder emits. Kept in
// lock-step with the internal vocoder constant by a static_assert in
// cosyvoice_engine.cpp, so this public value can't silently drift from it.
constexpr int kNativeSampleRate = 24000;

// Per-stage wall-clock accounting for one synthesize() call, in milliseconds.
// Measured after a backend sync, so an async GPU submission cannot read as
// speed.  The counters exist so two runs can be PROVEN to have done the same
// work before their times are compared -- a run that decoded a different number
// of tokens is void, not faster.
struct StageTimings {
    double lm_prefill_ms    = 0;
    double lm_decode_ms     = 0;
    double flow_frontend_ms = 0;
    double dit_euler_ms     = 0;
    double hift_f0_ms       = 0;
    double hift_source_ms   = 0;
    double hift_stft_ms     = 0;
    double hift_decode_ms   = 0;
    double total_ms         = 0;

    int n_decode_steps  = 0;
    int n_speech_tokens = 0;
    int tm              = 0;   // flow frames in (prompt + generated)
    int mel_len         = 0;   // mel frames out (after the prompt trim)

    // Composition of the LM prefill this call constructed (also filled when a
    // pinned trajectory skips the LM): text ids = template + transcript +
    // synthesis text; prompt speech tokens = the reference/baked tokens the
    // LM continues from (0 in instruct and cross-lingual modes).  Lets
    // callers and tests verify the prompt actually carries what the mode
    // promises without re-deriving tokenization.
    int n_text_ids            = 0;
    int n_prompt_speech_tokens = 0;
};

struct SynthesisResult {
    std::vector<float> pcm;
    int   sample_rate = kNativeSampleRate;
    float duration_s  = 0.0f;

    StageTimings     timings;        // per-stage wall clock for this call
    std::vector<int> speech_tokens;  // the LM trajectory this call actually used
};

// Persistent engine.  Loads the model once at construction; subsequent
// synthesize() calls reuse the resident model.
class TTS_CPP_API Engine {
public:
    // Loads the model and prepares the resident engine.  Throws
    // std::runtime_error on a hard failure (e.g. a required GGUF missing under
    // model_dir).  Loading happens once here; synthesize() reuses it.
    explicit Engine(const EngineOptions & opts);
    ~Engine();

    Engine(const Engine &)             = delete;
    Engine & operator=(const Engine &) = delete;
    Engine(Engine &&) noexcept;
    Engine & operator=(Engine &&) noexcept;

    // Synthesize `text` into PCM (24 kHz mono float32 by default) using
    // options().default_controls.  Throws std::runtime_error on failure.  Empty
    // `text` is rejected.  Not safe to call concurrently on the same Engine.
    SynthesisResult synthesize(const std::string & text);

    // Streaming variant — see EngineOptions streaming block.  Falls through
    // to the batch path when streaming is disabled.
    SynthesisResult synthesize(const std::string & text,
                               const StreamCallback & on_chunk);

    // Per-call conditioning.  `controls` REPLACES options().default_controls
    // for this call -- it is not merged, so passing VoiceControls{} always
    // gets back to plain zero-shot.  Throws std::invalid_argument on an
    // unsupported value or on two engaged channels.
    //
    // Pass an explicitly-typed argument: synthesize(text, {}) is ambiguous
    // between VoiceControls and StreamCallback.
    SynthesisResult synthesize(const std::string & text, const VoiceControls & controls);

    SynthesisResult synthesize(const std::string & text, const VoiceControls & controls,
                               const StreamCallback & on_chunk);

    // Best-effort cancel of an in-flight synthesize() on another thread.
    void cancel();

    const EngineOptions & options() const;

    // Registered name of the resolved backend ("CPU", "Metal", ...).
    std::string backend_name() const;

    // Resolved compute device (CPU, or GPU for Metal / Vulkan / OpenCL).
    BackendDevice backend_device() const;

    // True when a GPU device was present but unusable (fell back to CPU).
    bool gpu_unsupported() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

// Convenience one-shot wrapper.  Equivalent to:
//   Engine e(opts); return e.synthesize(text);
TTS_CPP_API SynthesisResult synthesize(const EngineOptions & opts, const std::string & text);

TTS_CPP_API SynthesisResult synthesize(const EngineOptions & opts, const std::string & text,
                                       const VoiceControls & controls);

} // namespace tts_cpp::cosyvoice
