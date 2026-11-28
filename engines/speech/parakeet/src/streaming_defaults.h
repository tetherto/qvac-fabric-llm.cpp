#pragma once

#include <stdexcept>

namespace parakeet {

constexpr int kTdtMinimumPushChunkMs = 30000;

inline int resolve_tdt_push_chunk_ms(int requested_ms) {
    if (requested_ms == 0) return kTdtMinimumPushChunkMs;
    if (requested_ms < kTdtMinimumPushChunkMs) {
        throw std::runtime_error(
            "Engine::stream_start: offline TDT models require chunk_ms >= 30000; "
            "shorter push-stream windows can silently drop phrases. "
            "Use chunk_ms=30000 or a cache-aware streaming model.");
    }
    return requested_ms;
}

inline int resolve_transcription_chunk_ms(int requested_ms, bool full_eu) {
    constexpr int kAutomaticChunkMs = 0;
    constexpr int kDefaultChunkMs = 1000;
    constexpr int kFullEuChunkMs = 2000;
    if (requested_ms != kAutomaticChunkMs) return requested_ms;
    return full_eu ? kFullEuChunkMs : kDefaultChunkMs;
}

}  // namespace parakeet
