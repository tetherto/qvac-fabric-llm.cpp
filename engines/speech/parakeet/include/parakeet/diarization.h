#pragma once

// Sortformer diarization: offline results and sliding-history streaming sessions.
//
// Offline: segments plus per-frame speaker_probs. Streaming: push PCM; each step runs Sortformer
// over the last `history_ms`, emits overlapping segments, advances the chunk cursor.
// Optional StreamEvent VadStateChanged from speaker_probs (see <parakeet/streaming.h>).

#include "export.h"
#include "streaming.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace parakeet {

struct DiarizationOptions {
    float threshold      = 0.5f;
    int   min_segment_ms = 0;
};

struct DiarizationSegment {
    int    speaker_id = 0;
    double start_s    = 0.0;
    double end_s      = 0.0;
};

struct DiarizationResult {
    std::vector<DiarizationSegment> segments;
    std::vector<float> speaker_probs;
    int    n_frames        = 0;
    int    num_spks        = 0;
    double frame_stride_s  = 0.08;

    // True only when every network pass of this call ran its encoder on
    // Core ML: a diarizer sidecar, or a Sortformer v2.1 encoder sidecar ahead
    // of the ggml speaker head. False when any pass ran on ggml, and when the
    // call ran no pass. Unlike Engine::encoder_on_coreml(), this reports
    // actual per-call routing rather than sidecar availability.
    bool   encoder_used_coreml = false;

    // Per-call backend of the network passes: "ggml", a Core ML label, or
    // "mixed" (see EngineResult::encoder_backend).
    std::string encoder_backend = "ggml";
    // True when a pass ran on ggml only because its Core ML sidecar (encoder
    // or diarizer capacity) was still loading.
    bool   coreml_pending = false;

    int    audio_samples   = 0;
    int    sample_rate     = 16000;
    double preprocess_ms   = 0.0;
    double encoder_ms      = 0.0;
    double decode_ms       = 0.0;
    double total_ms        = 0.0;
};

// One speaker span emitted by SortformerStreamSession. Every non-cancelled
// finalize() emits exactly one LAST callback with `is_final=true`: a synthetic
// terminator whose `speaker_id = -1` and `start_s == end_s`. Real speaker
// segments, including segments drained from a trailing partial chunk, always
// have `is_final=false`. Consumers should treat `speaker_id < 0` as "session
// done, no new segment" and skip any text/append logic for it.
struct StreamingDiarizationSegment {
    int    speaker_id  = 0;
    double start_s     = 0.0;
    double end_s       = 0.0;
    int    chunk_index = 0;
    bool   is_final    = false;
};

struct SortformerStreamingOptions {
    int   sample_rate     = 16000;

    int   chunk_ms        = 2000;
    int   history_ms      = 30000;

    float threshold       = 0.5f;
    int   min_segment_ms  = 200;

    bool  emit_partials   = true;

    // Optional StreamEvent delivery (VadStateChanged from speaker_probs); nullptr disables.
    StreamEventCallback on_event = nullptr;

    bool  spkcache_enable        = true;
    int   spkcache_len           = 188;    // total cache rows (encoder frames)
    int   fifo_len               = 188;    // FIFO warmup buffer (encoder frames)
    static constexpr int unspecified_left_context_ms = -1;
    int   chunk_left_context_ms  = unspecified_left_context_ms;
    int   chunk_right_context_ms = 560;    // ~7 encoder frames at v2.1 (560ms)
    int   spkcache_update_period = 144;    // pop_out_len on FIFO overflow
};

using SortformerSegmentCallback =
    std::function<void(const StreamingDiarizationSegment &)>;

class PARAKEET_API SortformerStreamSession {
public:
    struct Impl;
    explicit SortformerStreamSession(std::unique_ptr<Impl> impl);
    ~SortformerStreamSession();

    SortformerStreamSession(const SortformerStreamSession &)            = delete;
    SortformerStreamSession & operator=(const SortformerStreamSession &) = delete;
    SortformerStreamSession(SortformerStreamSession &&) noexcept;
    SortformerStreamSession & operator=(SortformerStreamSession &&) noexcept;

    void feed_pcm_f32(const float * samples, int n_samples);
    void feed_pcm_i16(const int16_t * samples, int n_samples);
    void finalize();
    void cancel();

    const SortformerStreamingOptions & options() const;

    bool aosc_active() const;

private:
    std::unique_ptr<Impl> pimpl_;
};

}
