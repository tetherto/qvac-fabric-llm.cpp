#pragma once

#include "parakeet_ctc.h"
#include "parakeet_tdt.h"

#include <chrono>

namespace parakeet {

// Decode within run_encoder's tensor lifetime; no graph-buffer handle escapes.
inline EncoderOutputConsumer make_device_encoder_consumer(
    ParakeetCtcModel & model, TdtRuntimeWeights & runtime,
    int32_t prompt_id, const TdtDecodeOptions & options,
    TdtDecodeResult & result, double & decode_ms) {
    return [&model, &runtime, prompt_id, options, &result, &decode_ms](
        ggml_tensor * encoded, ggml_backend_t backend) -> int {
        if (!tdt_accepts_device_input(runtime, encoded, backend)) return kEncoderOutputHost;
        const auto started = std::chrono::steady_clock::now();
        ggml_tensor * input = encoded;
        if (model.model_type == ParakeetModelType::NEMOTRON) {
            if (int rc = run_nemotron_prompt_projection_device(model, encoded, prompt_id, input); rc != 0) {
                return rc;
            }
        }
        if (int rc = tdt_greedy_decode_device(model, runtime, input, options, result); rc != 0) return -rc;
        decode_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        return kEncoderOutputConsumed;
    };
}

} // namespace parakeet
