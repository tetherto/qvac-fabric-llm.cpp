#pragma once

// Streaming transcription: Mode 2 (full audio, chunked callbacks) and Mode 3 (StreamSession push PCM).
//
// Declares StreamSession, StreamingOptions/Segment, and cross-engine StreamEvent
// callbacks for VAD state changes and EOU end-of-turn signals.

#include "export.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace parakeet {

// Optional StreamEvent callback: VadStateChanged and EndOfTurn alongside segment text.
//
// EOU models emit EndOfTurn when `<EOU>` fires; this is a turn-boundary signal,
// not a VAD state transition. Sortformer emits VadStateChanged from speaker_probs
// versus threshold. CTC/RNN-T/TDT can emit VadStateChanged from optional RMS EnergyVad.

enum class VadState : int {
    Unknown  = 0,
    Speaking = 1,
    Silent   = 2,
};

enum class StreamEventType : int {
    VadStateChanged = 1,
    EndOfTurn       = 2,
};

struct StreamEvent {
    StreamEventType type  = StreamEventType::VadStateChanged;

    // Wall-clock seconds since the streaming session started feeding samples.
    // For chunk-aligned events this is the chunk's emit-end time; for
    // sample-level events (energy-VAD transitions) it is the transition
    // sample boundary in seconds.
    double timestamp_s = 0.0;

    // Index of the chunk that produced the event, when known. -1 when the
    // event was synthesised between chunks (e.g. an energy-VAD silence
    // transition during long quiet inputs).
    int    chunk_index = -1;

    // VadStateChanged fields
    VadState vad_state  = VadState::Unknown;
    int      speaker_id = -1;     // argmax speaker on entering Speaking; -1 otherwise
    float    vad_score  = 0.0f;   // 0..1; provenance-specific (max speaker prob, RMS, ...)

    // EndOfTurn fields
    float    eot_confidence    = 0.0f;  // 0..1; for EOU = 1.0 when `<EOU>` fired
};

using StreamEventCallback = std::function<void(const StreamEvent &)>;

struct StreamingOptions {
    int sample_rate  = 16000;
    // Zero selects the model default: 2000 ms for full TranscribePsy EU,
    // 30000 ms for other TDT push streams, 1000 ms for other transcription
    // families and encode-once streams. Standard TDT push streams reject
    // explicit values below 30000 ms because short windows can drop phrases.
    // Nemotron accepts only 80, 160, 320, 560, or 1120 ms operating points.
    static constexpr int automatic_chunk_ms = 0;
    int chunk_ms = automatic_chunk_ms;

    // Used by the sliding-window implementation. Nemotron maintains its
    // trained 56-frame attention cache and ignores these two window knobs.
    int left_context_ms    = 10000;
    int right_lookahead_ms = 2000;

    bool emit_partials = false;

    // Optional; nullptr disables StreamEvent delivery (segment-only streaming).
    StreamEventCallback on_event = nullptr;

    // Optional energy VAD. When true, CTC / RNN-T / TDT sessions compute a
    // simple RMS-thresholded VAD over the input PCM and fire
    // `StreamEventType::VadStateChanged` events on transitions. Sortformer
    // provides its own speaker-probability VAD events. EOU emits EndOfTurn,
    // not VadStateChanged. Default off; opt in for CTC/RNN-T/TDT/Nemotron
    // consumers that want VadState events.
    bool  enable_energy_vad = false;

    // Energy-VAD knobs (dB-scale; applies only when enable_energy_vad).
    // Defaults are tuned for clean 16 kHz mono speech: speech enters above
    // -35 dBFS RMS over a 30 ms window, falls back to silent after 200 ms
    // of below-threshold audio.
    float energy_vad_threshold_db = -35.0f;
    int   energy_vad_window_ms    = 30;
    int   energy_vad_hangover_ms  = 200;
};

struct StreamingSegment {
    std::string text;
    std::vector<int32_t> token_ids;

    double start_s = 0.0;
    double end_s   = 0.0;

    int  chunk_index = 0;
    bool is_final    = true;

    // True when this segment's first token is a SentencePiece word-start
    // (the piece begins with the `▁` U+2581 marker), false when it is a
    // wordpiece continuation of the previous segment's last token.
    //
    // Streaming consumers building a running transcript should insert a
    // separator (e.g. " ") between successive segments only when the
    // *new* segment has `starts_word == true`. Concatenating verbatim
    // when `starts_word == false` joins the splits like
    // ["pun", "ctuation"] back into "punctuation"; inserting a space
    // there would yield "pun ctuation" instead.
    //
    // Always true on the very first segment of a session and on any
    // segment whose token list is empty (defensive default).
    bool   starts_word = true;

    // EOU-only: true when this segment contains a decoded `<EOU>` boundary.
    // The same boundary may be delivered as StreamEventType::EndOfTurn through
    // `on_event`. CTC/RNN-T/TDT leave this flag false and do not synthesize end-of-turn.
    bool   is_eou_boundary = false;
    float  eot_confidence  = 0.0f;

    double encoder_ms = 0.0;
    double decode_ms  = 0.0;
    // Actual dispatch for this segment. Sequential streaming decoders stay
    // native; the encoder can still run through its Core ML sidecar.
    bool encoder_used_coreml = false;
};

using StreamingCallback = std::function<void(const StreamingSegment &)>;

class PARAKEET_API StreamSession {
public:
    struct Impl;
    explicit StreamSession(std::unique_ptr<Impl> impl);
    ~StreamSession();

    StreamSession(const StreamSession &)            = delete;
    StreamSession & operator=(const StreamSession &) = delete;
    StreamSession(StreamSession &&) noexcept;
    StreamSession & operator=(StreamSession &&) noexcept;

    void feed_pcm_f32(const float * samples, int n_samples);
    void feed_pcm_i16(const int16_t * samples, int n_samples);
    void finalize();
    void cancel();

    const StreamingOptions & options() const;

private:
    std::unique_ptr<Impl> pimpl_;
};

}
