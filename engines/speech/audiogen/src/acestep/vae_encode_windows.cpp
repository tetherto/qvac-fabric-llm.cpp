#include "vae_encode_windows.h"

#include <algorithm>
#include <cmath>

namespace tts_cpp::acestep {

int vae_encode_window_count(int frames) {
    if (frames <= VAE_AUDIO_CHUNK_FRAMES) return 1;
    return (frames + VAE_AUDIO_STRIDE_FRAMES - 1) / VAE_AUDIO_STRIDE_FRAMES;
}

VaeEncodeWindow make_vae_encode_window(int index, int frames) {
    VaeEncodeWindow window;
    window.core_start   = index * VAE_AUDIO_STRIDE_FRAMES;
    window.core_end     = std::min(window.core_start + VAE_AUDIO_STRIDE_FRAMES, frames);
    window.window_start = std::max(0, window.core_start - VAE_AUDIO_OVERLAP_FRAMES);
    window.window_end   = std::min(frames, window.core_end + VAE_AUDIO_OVERLAP_FRAMES);
    return window;
}

static bool append_vae_window_core(std::vector<float> & output, const std::vector<float> & tile,
                                   int tile_frames, const VaeEncodeWindow & window, float downsample) {
    const int trim_start = (int) std::round((float) (window.core_start - window.window_start) * downsample);
    const int trim_end   = (int) std::round((float) (window.window_end - window.core_end) * downsample);
    const int end        = trim_end > 0 ? tile_frames - trim_end : tile_frames;
    if (trim_start < 0 || end <= trim_start || end > tile_frames) return false;

    output.insert(
        output.end(),
        tile.begin() + (size_t) trim_start * VAE_LATENT_CHANNELS,
        tile.begin() + (size_t) end * VAE_LATENT_CHANNELS);
    return true;
}

static std::vector<float> encode_vae_windows(const std::vector<float> & pcm, int frames,
                                             const VaeWindowEncoder & encode, int * latent_frames) {
    const int count = vae_encode_window_count(frames);
    std::vector<float> output;
    output.reserve(((size_t) frames / VAE_ENCODER_UPSAMPLE + (size_t) count) * VAE_LATENT_CHANNELS);
    float downsample = 0.0f;

    for (int index = 0; index < count; ++index) {
        const VaeEncodeWindow window = make_vae_encode_window(index, frames);
        std::vector<float> tile;
        const int window_frames = window.window_end - window.window_start;
        const float * audio = pcm.data() + (size_t) window.window_start * 2;
        const int tile_frames = encode(audio, window_frames, tile);
        if (tile_frames <= 0) return {};

        if (index == 0) downsample = (float) tile_frames / (float) window_frames;
        if (!append_vae_window_core(output, tile, tile_frames, window, downsample)) return {};
    }

    if (latent_frames) *latent_frames = (int) (output.size() / VAE_LATENT_CHANNELS);
    return output;
}

static std::vector<float> encode_vae_single_pass(const std::vector<float> & pcm, int frames,
                                                 const VaeWindowEncoder & encode, int * latent_frames) {
    std::vector<float> output;
    const int encoded_frames = encode(pcm.data(), frames, output);
    if (encoded_frames < 0) return {};
    if (latent_frames) *latent_frames = encoded_frames;
    return output;
}

std::vector<float> encode_vae_pcm_bounded(const std::vector<float> & pcm, int frames,
                                          const VaeWindowEncoder & encode, int * latent_frames) {
    if (latent_frames) *latent_frames = 0;
    if (frames <= 0 || (int) pcm.size() < frames * 2) return {};
    if (frames <= VAE_AUDIO_CHUNK_FRAMES) {
        return encode_vae_single_pass(pcm, frames, encode, latent_frames);
    }
    return encode_vae_windows(pcm, frames, encode, latent_frames);
}

} // namespace tts_cpp::acestep
