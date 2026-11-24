#pragma once

#include <functional>
#include <vector>

namespace tts_cpp::acestep {

inline constexpr int VAE_LATENT_CHANNELS       = 64;
inline constexpr int VAE_PCM_CHANNELS          = 2;
inline constexpr int VAE_ENCODER_UPSAMPLE      = 1920;
inline constexpr int VAE_LATENT_CHUNK_FRAMES   = 1024;
inline constexpr int VAE_LATENT_OVERLAP_FRAMES = 64;
inline constexpr int VAE_AUDIO_CHUNK_FRAMES    = VAE_LATENT_CHUNK_FRAMES * VAE_ENCODER_UPSAMPLE;
inline constexpr int VAE_AUDIO_OVERLAP_FRAMES  = VAE_LATENT_OVERLAP_FRAMES * VAE_ENCODER_UPSAMPLE;
inline constexpr int VAE_AUDIO_STRIDE_FRAMES   = VAE_AUDIO_CHUNK_FRAMES - 2 * VAE_AUDIO_OVERLAP_FRAMES;

struct VaeEncodeWindow {
    int core_start = 0;
    int core_end   = 0;
    int window_start = 0;
    int window_end   = 0;
};

using VaeWindowEncoder = std::function<int(const float *, int, std::vector<float> &)>;

int vae_encode_window_count(int frames);
VaeEncodeWindow make_vae_encode_window(int index, int frames);
std::vector<float> encode_vae_pcm_bounded(const std::vector<float> & pcm, int frames,
                                          const VaeWindowEncoder & encode, int * latent_frames);

} // namespace tts_cpp::acestep
