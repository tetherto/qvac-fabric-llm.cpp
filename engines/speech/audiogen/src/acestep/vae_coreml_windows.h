#pragma once

#include <vector>

namespace tts_cpp::acestep {

struct VaeCoremlWindow {
    int win_a;
    int core_a;
    int core_b;
};

// The decoder's receptive field spans at most 12 latent frames: stitched
// windows reproduce the full decode bit-exactly down to a 12-frame overlap on
// real and adversarial latents alike, and only break at 4. At 8 frames the
// boundary error (max_abs 1.7e-4 real, 2.2e-3 at 3x-amplitude noise) stays at
// or below the sidecar's own fp16 conversion error (max_abs 4e-3), so 8 buys
// a 20% larger core stride without moving the overall error budget.
inline constexpr int VAE_COREML_OVERLAP = 8;

inline std::vector<VaeCoremlWindow> vae_coreml_plan_windows(int T_latent, int window_frames, int overlap) {
    std::vector<VaeCoremlWindow> plan;
    if (T_latent < window_frames || window_frames <= 2 * overlap) {
        return plan;
    }
    const int core_stride = window_frames - 2 * overlap;
    int prev_core_b = 0;
    for (int i = 0;; ++i) {
        int win_a = i * core_stride;
        if (win_a > T_latent - window_frames) {
            win_a = T_latent - window_frames;
        }
        const int win_b  = win_a + window_frames;
        const int core_a = prev_core_b;
        const int core_b = (win_b == T_latent) ? T_latent : win_b - overlap;
        plan.push_back({win_a, core_a, core_b});
        prev_core_b = core_b;
        if (win_b == T_latent) {
            return plan;
        }
    }
}

}  // namespace tts_cpp::acestep
