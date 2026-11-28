#pragma once

// A cancellation is consumed by the generation that observes it, never erased
// at startup: cancel() arriving between a caller's own precheck and generate()
// entry must cancel that run instead of being discarded. Clearing the flag on
// scope exit keeps the next run clean on every return path.
//
// Mirrors the MiniMax engine's CancellationScope; kept in a header so the
// contract is unit-testable without model weights.

#include <atomic>

namespace tts_cpp::acestep {

class CancellationScope {
public:
    explicit CancellationScope(std::atomic<bool> & cancelled) : cancelled_(cancelled) {}

    CancellationScope(const CancellationScope &) = delete;
    CancellationScope & operator=(const CancellationScope &) = delete;

    ~CancellationScope() {
        cancelled_.store(false);
    }

private:
    std::atomic<bool> & cancelled_;
};

}
