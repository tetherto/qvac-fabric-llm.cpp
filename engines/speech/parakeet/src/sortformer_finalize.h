#pragma once

#include <utility>

namespace parakeet::detail {

// Model-independent Sortformer finalization state machine. Keeping the tail
// drain and final emission in one function prevents either path from returning
// before the synthetic terminator is emitted.
template <typename FlushReadyFn,
          typename HasTailFn,
          typename DrainTailFn,
          typename EmitFinalFn>
void finalize_sortformer_stream(bool & finalized,
                                const bool & cancelled,
                                FlushReadyFn && flush_ready,
                                HasTailFn && has_tail,
                                DrainTailFn && drain_tail,
                                EmitFinalFn && emit_final) {
    if (finalized) return;
    finalized = true;
    if (cancelled) return;

    std::forward<FlushReadyFn>(flush_ready)();
    if (cancelled) return;

    if (std::forward<HasTailFn>(has_tail)()) {
        std::forward<DrainTailFn>(drain_tail)();
    }
    if (cancelled) return;

    std::forward<EmitFinalFn>(emit_final)();
}

} // namespace parakeet::detail
