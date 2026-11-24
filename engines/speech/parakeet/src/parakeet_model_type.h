#pragma once

// The Parakeet model families a GGUF can hold. Kept in its own header so the
// pure planning helpers (long_form.h) can map a family to its Core ML shape
// contract without depending on the model runtime.

namespace parakeet {

enum class ParakeetModelType {
    CTC,
    RNNT,
    TDT,
    EOU,
    NEMOTRON,
    SORTFORMER,
    NEMOTRON_DIARIZATION,
};

}  // namespace parakeet
