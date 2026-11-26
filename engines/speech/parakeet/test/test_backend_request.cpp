// Public API routing regression: explicit unavailable devices must fail before
// opening the model, never continue into CPU/automatic model loading. No model
// fixture or accelerator is required; CPU is required by every Parakeet load.
#include "parakeet/engine.h"

#include <cstdio>
#include <stdexcept>
#include <string>

int main() {
    parakeet::EngineOptions opts;
    opts.model_gguf_path = "/nonexistent-parakeet-backend-request-test/model.gguf";
    opts.n_threads = 1;
    int failures = 0;
    auto expect_failure = [&](const char * backend, int layers, int expected_rc) {
        opts.backend = backend;
        opts.n_gpu_layers = layers;
        try {
            parakeet::Engine engine(opts);
            ++failures;
        } catch (const std::runtime_error & e) {
            const std::string error = e.what();
            if (error.find("rc=" + std::to_string(expected_rc) + ")") == std::string::npos ||
                error.find(opts.backend) == std::string::npos) {
                std::fprintf(stderr, "FAIL: backend %s expected rc=%d: %s\n", backend, expected_rc, e.what());
                ++failures;
            }
        }
    };
    // Load CPU first, then request unavailable devices in the populated
    // registry: explicit routing is per Engine, not a process-global choice.
    // CPU must get to the missing model even when GPU offload was requested.
    expect_failure("cpu", 999, 1);
    expect_failure("nonexistent-parakeet-device", 0, 11);
    expect_failure("auto", 0, 1);
    expect_failure("nonexistent-parakeet-device", 999, 11);
    return failures == 0 ? 0 : 1;
}
