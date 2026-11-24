#include "moss/sfx_networks.h"

#include "moss/sfx_coreml.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace tts_cpp::moss::detail {
namespace {

constexpr int VAE_GRAPH_NODES = 8192;
constexpr int RESIDUAL_UNITS = 3;
constexpr int RESIDUAL_DILATIONS[RESIDUAL_UNITS] = {1, 3, 9};
constexpr int RESIDUAL_KERNEL = 7;
constexpr int EDGE_KERNEL = 7;
constexpr int CHANNEL_REDUCTION = 2;
constexpr int UPSAMPLE_KERNEL_PER_STRIDE = 2;

struct VaeGraph {
    SfxModel & model;
    const SfxVaeConfig & config;
    SfxGraph & graph;

    ggml_context * ctx() const { return graph.ctx(); }

    ggml_tensor * snake(ggml_tensor * x, const std::string & name) const {
        return ggml_snake(ctx(), x, model.tensor(name + ".alpha"), model.tensor(name + ".inv"));
    }

    ggml_tensor * conv(ggml_tensor * x, const std::string & name, int padding, int dilation) const {
        ggml_tensor * y = ggml_conv_1d(ctx(), model.tensor(name + ".weight"), x, 1, padding, dilation);
        y = ggml_reshape_2d(ctx(), y, y->ne[0], y->ne[1]);
        return ggml_add(ctx(), y, model.tensor(name + ".bias"));
    }

    ggml_tensor * upsample(ggml_tensor * x, const std::string & name, int stride) const {
        const int padding = (stride + 1) / 2;
        const int output_padding = stride % 2;
        const int out_channels = (int) model.tensor(name + ".bias")->ne[1];
        ggml_tensor * columns = ggml_mul_mat(ctx(), model.tensor(name + ".weight"),
                ggml_cont(ctx(), ggml_transpose(ctx(), x)));
        ggml_tensor * y = ggml_col2im_1d(ctx(), columns, stride, out_channels, padding - output_padding);
        const int64_t samples = x->ne[0] * stride;
        y = ggml_view_2d(ctx(), y, samples, y->ne[1], y->nb[1], (size_t) output_padding * y->nb[0]);
        return ggml_add(ctx(), ggml_cont(ctx(), y), model.tensor(name + ".bias"));
    }

    ggml_tensor * residual_unit(ggml_tensor * x, const std::string & name, int dilation) const {
        const int padding = (RESIDUAL_KERNEL - 1) * dilation / 2;
        ggml_tensor * y = conv(snake(x, name + ".snake1"), name + ".conv1", padding, dilation);
        y = conv(snake(y, name + ".snake2"), name + ".conv2", 0, 1);
        return ggml_add(ctx(), x, y);
    }

    ggml_tensor * residual_units(ggml_tensor * x, const std::string & name) const {
        for (int unit = 0; unit < RESIDUAL_UNITS; ++unit) {
            x = residual_unit(x, name + ".res." + std::to_string(unit), RESIDUAL_DILATIONS[unit]);
        }
        return x;
    }

    ggml_tensor * decoder_block(ggml_tensor * x, int block) const {
        const std::string name = "vae.blk." + std::to_string(block);
        x = upsample(snake(x, name + ".snake"), name + ".up", config.rates[(size_t) block]);
        return residual_units(x, name);
    }

    ggml_tensor * decoder_blocks(ggml_tensor * x) const {
        for (int block = 0; block < (int) config.rates.size(); ++block) {
            x = decoder_block(x, block);
        }
        return x;
    }

    ggml_tensor * post_quant(ggml_tensor * latents) const {
        ggml_tensor * x = ggml_mul_mat(ctx(), model.tensor("vae.post_quant.weight"),
                ggml_cont(ctx(), ggml_transpose(ctx(), latents)));
        x = ggml_cont(ctx(), ggml_transpose(ctx(), x));
        return ggml_add(ctx(), x, model.tensor("vae.post_quant.bias"));
    }

    ggml_tensor * forward(ggml_tensor * latents) const {
        ggml_tensor * x = conv(post_quant(latents), "vae.conv_in", EDGE_KERNEL / 2, 1);
        x = decoder_blocks(x);
        x = conv(snake(x, "vae.snake_out"), "vae.conv_out", EDGE_KERNEL / 2, 1);
        return ggml_tanh(ctx(), x);
    }
};

void require_kernel(const SfxModel & model, const std::string & name, int kernel, int in, int out) {
    const ggml_tensor * weight = model.tensor(name + ".weight");
    require_shape(weight, kernel, in, out);
    require_vector(model.tensor(name + ".bias"), 1, out);
    if (weight->type != GGML_TYPE_F16) {
        throw std::runtime_error("moss sfx: " + name + ".weight must be f16");
    }
}

void require_snake(const SfxModel & model, const std::string & name, int channels) {
    require_vector(model.tensor(name + ".alpha"), 1, channels);
    require_vector(model.tensor(name + ".inv"), 1, channels);
}

void validate_residual_units(const SfxModel & model, const std::string & name, int channels) {
    for (int unit = 0; unit < RESIDUAL_UNITS; ++unit) {
        const std::string prefix = name + ".res." + std::to_string(unit);
        require_snake(model, prefix + ".snake1", channels);
        require_kernel(model, prefix + ".conv1", RESIDUAL_KERNEL, channels, channels);
        require_snake(model, prefix + ".snake2", channels);
        require_kernel(model, prefix + ".conv2", 1, channels, channels);
    }
}

void validate_decoder_block(const SfxModel & model, int block, int in, int stride) {
    const std::string name = "vae.blk." + std::to_string(block);
    const int out = in / CHANNEL_REDUCTION;
    require_snake(model, name + ".snake", in);
    require_matrix(model.tensor(name + ".up.weight"), in, (int64_t) UPSAMPLE_KERNEL_PER_STRIDE * stride * out);
    require_vector(model.tensor(name + ".up.bias"), 1, out);
    validate_residual_units(model, name, out);
}

int validate_decoder_blocks(const SfxModel & model, const SfxVaeConfig & config) {
    int channels = config.decoder_dim;
    for (int block = 0; block < (int) config.rates.size(); ++block) {
        validate_decoder_block(model, block, channels, config.rates[(size_t) block]);
        channels /= CHANNEL_REDUCTION;
    }
    return channels;
}

std::vector<float> window_latents(const std::vector<float> & latents, int frames, int channels,
                                  const SfxVaeWindow & window) {
    std::vector<float> slice((size_t) window.count * channels);
    for (int channel = 0; channel < channels; ++channel) {
        const auto source = latents.begin() + (std::ptrdiff_t) channel * frames + window.first;
        std::copy(source, source + window.count, slice.begin() + (std::ptrdiff_t) channel * window.count);
    }
    return slice;
}

ggml_tensor * build_decode_graph(SfxModel & model, SfxGraph & graph, int frames) {
    const SfxConfig & config = model.config();
    VaeGraph builder{model, config.vae, graph};
    ggml_tensor * input = graph.input_f32(frames, config.vae.latent_dim);
    ggml_tensor * audio = builder.forward(input);
    ggml_set_output(audio);
    ggml_build_forward_expand(graph.graph(), audio);
    ggml_set_name(input, "sfx_decode_input");
    return audio;
}

std::vector<float> decode_window(SfxModel & model, const std::vector<float> & slice, int frames) {
    SfxGraph graph(VAE_GRAPH_NODES);
    auto * audio = build_decode_graph(model, graph, frames);
    auto * input = ggml_get_tensor(graph.ctx(), "sfx_decode_input");
    model.allocate(graph);
    ggml_backend_tensor_set(input, slice.data(), 0, slice.size() * sizeof(float));
    model.compute(graph);
    std::vector<float> pcm((size_t) ggml_nelements(audio));
    ggml_backend_tensor_get(audio, pcm.data(), 0, ggml_nbytes(audio));
    return pcm;
}

std::vector<SfxVaeWindow> plan_windows(int frames, int keep_frames, int window_frames) {
    std::vector<SfxVaeWindow> windows;
    for (int first = 0; first < keep_frames; first += window_frames) {
        SfxVaeWindow window;
        window.keep_count = std::min(window_frames, keep_frames - first);
        window.first = std::max(0, first - SFX_VAE_CONTEXT_FRAMES);
        const int last = std::min(frames, first + window.keep_count + SFX_VAE_CONTEXT_FRAMES);
        window.count = last - window.first;
        window.keep_first = first - window.first;
        windows.push_back(window);
    }
    return windows;
}

SfxVaeWindow fixed_window(int frames, int keep_frames, int window, int context, int first) {
    SfxVaeWindow planned;
    planned.keep_count = std::min(window - 2 * context, keep_frames - first);
    planned.first = std::clamp(first - context, 0, frames - window);
    planned.count = window;
    planned.keep_first = first - planned.first;
    return planned;
}

void append_window(std::vector<float> & pcm, const std::vector<float> & decoded, const SfxVaeWindow & window,
                   int hop) {
    const auto first = decoded.begin() + (std::ptrdiff_t) window.keep_first * hop;
    pcm.insert(pcm.end(), first, first + (std::ptrdiff_t) window.keep_count * hop);
}

void validate_request(const SfxModel & model, const std::vector<float> & latents, int frames,
                      int keep_frames, int window_frames) {
    if (frames < 1 || keep_frames < 1 || keep_frames > frames || window_frames < 1 ||
        latents.size() != (size_t) frames * model.config().vae.latent_dim) {
        throw std::runtime_error("moss sfx: invalid VAE decode request");
    }
}

bool decode_windows(SfxModel & model, const std::vector<float> & latents, int frames,
                    const std::vector<SfxVaeWindow> & windows, const SfxDecodeProgress & progress,
                    std::vector<float> & pcm) {
    const SfxVaeConfig & config = model.config().vae;
    for (size_t i = 0; i < windows.size(); ++i) {
        const std::vector<float> slice = window_latents(latents, frames, config.latent_dim, windows[i]);
        append_window(pcm, decode_window(model, slice, windows[i].count), windows[i], config.hop_length());
        if (progress && !progress((int) i + 1, (int) windows.size())) {
            return false;
        }
    }
    return true;
}

SidecarDecode decode_sidecar_windows(SfxVaeSidecar & sidecar, const SfxVaeConfig & config,
                                     const std::vector<float> & latents, int frames,
                                     const std::vector<SfxVaeWindow> & windows,
                                     const SfxDecodeProgress & progress, std::vector<float> & pcm) {
    std::vector<float> decoded;
    for (size_t i = 0; i < windows.size(); ++i) {
        if (!sidecar.decode(window_latents(latents, frames, config.latent_dim, windows[i]), decoded)) {
            return SidecarDecode::Failed;
        }
        append_window(pcm, decoded, windows[i], config.hop_length());
        if (progress && !progress((int) i + 1, (int) windows.size())) {
            return SidecarDecode::Cancelled;
        }
    }
    return SidecarDecode::Done;
}

} // namespace

std::vector<SfxVaeWindow> plan_fixed_vae_windows(int frames, int keep_frames, int window, int context) {
    std::vector<SfxVaeWindow> windows;
    const int core = window - 2 * context;
    if (core < 1 || window > frames || keep_frames < 1 || keep_frames > frames) {
        return windows;
    }
    for (int first = 0; first < keep_frames; first += core) {
        windows.push_back(fixed_window(frames, keep_frames, window, context, first));
    }
    return windows;
}

SidecarDecode decode_latents_on_sidecar(SfxVaeSidecar & sidecar, const SfxVaeConfig & config,
                                        const std::vector<float> & latents, int frames, int keep_frames,
                                        const SfxDecodeProgress & progress, std::vector<float> & pcm) {
    const std::vector<SfxVaeWindow> windows =
            plan_fixed_vae_windows(frames, keep_frames, sidecar.window(), SFX_VAE_CONTEXT_FRAMES);
    if (windows.empty() || latents.size() != (size_t) frames * config.latent_dim) {
        return SidecarDecode::Failed;
    }
    pcm.clear();
    pcm.reserve((size_t) keep_frames * config.hop_length());
    return decode_sidecar_windows(sidecar, config, latents, frames, windows, progress, pcm);
}

void validate_vae(const SfxModel & model) {
    const SfxVaeConfig & config = model.config().vae;
    require_matrix(model.tensor("vae.post_quant.weight"), config.latent_dim, config.latent_dim);
    require_vector(model.tensor("vae.post_quant.bias"), 1, config.latent_dim);
    require_kernel(model, "vae.conv_in", EDGE_KERNEL, config.latent_dim, config.decoder_dim);
    const int channels = validate_decoder_blocks(model, config);
    require_snake(model, "vae.snake_out", channels);
    require_kernel(model, "vae.conv_out", EDGE_KERNEL, channels, 1);
}

std::vector<float> decode_latents(SfxModel & model, const std::vector<float> & latents, int frames,
                                  int keep_frames, int window_frames, const SfxDecodeProgress & progress) {
    validate_request(model, latents, frames, keep_frames, window_frames);
    std::vector<float> pcm;
    pcm.reserve((size_t) keep_frames * model.config().vae.hop_length());
    const std::vector<SfxVaeWindow> windows = plan_windows(frames, keep_frames, window_frames);
    if (!decode_windows(model, latents, frames, windows, progress, pcm)) {
        return {};
    }
    return pcm;
}

SfxMemory measure_decode(SfxModel & model, int frames, int keep_frames, int window_frames) {
    SfxMemory peak;
    const auto windows = plan_windows(frames, keep_frames, window_frames);
    std::vector<int> measured;
    for (const auto & window : windows) {
        if (std::find(measured.begin(), measured.end(), window.count) != measured.end()) continue;
        measured.push_back(window.count);
        SfxGraph graph(VAE_GRAPH_NODES);
        build_decode_graph(model, graph, window.count);
        const auto price = model.measure(graph);
        peak.device_bytes = std::max(peak.device_bytes, price.device_bytes);
        peak.host_compute_bytes = std::max(peak.host_compute_bytes, price.host_compute_bytes);
        const uint64_t slice = (uint64_t) window.count * model.config().vae.latent_dim * sizeof(float);
        const uint64_t audio = (uint64_t) window.count * model.config().vae.hop_length() * sizeof(float);
        peak.host_bytes = std::max(peak.host_bytes, price.host_bytes + slice + audio);
    }
    peak.host_bytes += windows.capacity() * sizeof(SfxVaeWindow);
    return peak;
}

} // namespace tts_cpp::moss::detail
