#include "parakeet_ctc.h"
#include "parakeet_tdt.h"
#include "cached_encoder.h"
#include "sentencepiece_bpe.h"
#include "backend_util.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <utility>

namespace parakeet {
namespace {

// GGUF-validated Nemotron cache geometry. validate_nemotron_model rejects
// checkpoints that disagree; these defaults match that contract when a
// caller has not yet populated the config.
constexpr int kDefaultChannelCacheFrames = 56;
constexpr int kDefaultConvolutionCacheFrames = 8;
constexpr int kDefaultSubsamplingFactor = 8;
constexpr int kPreEncodeCacheFrames = 9;
constexpr int kSubsamplingOverlapFrames = 2;

int channel_cache_frames(const ParakeetCtcModel & model) {
    return model.nemotron_cfg.left_context_frames > 0
        ? model.nemotron_cfg.left_context_frames
        : kDefaultChannelCacheFrames;
}

int convolution_cache_frames(const ParakeetCtcModel & model) {
    return model.nemotron_cfg.cache_time_steps > 0
        ? model.nemotron_cfg.cache_time_steps
        : kDefaultConvolutionCacheFrames;
}

int encoder_subsampling_factor(const ParakeetCtcModel & model) {
    return model.encoder_cfg.subsampling_factor > 0
        ? model.encoder_cfg.subsampling_factor
        : kDefaultSubsamplingFactor;
}
}

namespace cached_encoder {

// ggml-opencl mis-handles a handful of non-contiguous view feeds (byte-offset
// views into a concat, depthwise conv on a view). Force a materialised copy on
// OpenCL so downstream ops see a contiguous src. No-op on other backends.
ggml_tensor * ensure_contig_on_opencl(
    ggml_context * ctx,
    const ParakeetCtcModel & model,
    ggml_tensor * tensor) {
    return backend_is_opencl(model.backend_active())
        ? ggml_cont(ctx, tensor)
        : tensor;
}

}

namespace {

using namespace cached_encoder;

struct NemotronStepGraph {
    ggml_context * context = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_gallocr_t allocator = nullptr;

    ggml_tensor * encoder_input = nullptr;
    ggml_tensor * attention_mask = nullptr;
    ggml_tensor * position_input = nullptr;
    ggml_tensor * encoder_output = nullptr;
    std::vector<ggml_tensor *> channel_cache_inputs;
    std::vector<ggml_tensor *> channel_cache_outputs;
    std::vector<ggml_tensor *> time_cache_inputs;
    std::vector<ggml_tensor *> time_cache_outputs;

    std::vector<float> positions;
    int encoder_frames = 0;
    int channel_frames = 0;
    int convolution_frames = 0;

    void clear() {
        if (allocator) {
            ggml_gallocr_free(allocator);
            allocator = nullptr;
        }
        if (context) {
            ggml_free(context);
            context = nullptr;
        }
        graph = nullptr;
        encoder_input = nullptr;
        attention_mask = nullptr;
        position_input = nullptr;
        encoder_output = nullptr;
        channel_cache_inputs.clear();
        channel_cache_outputs.clear();
        time_cache_inputs.clear();
        time_cache_outputs.clear();
        positions.clear();
        encoder_frames = 0;
        channel_frames = 0;
        convolution_frames = 0;
    }

    ~NemotronStepGraph() {
        clear();
    }
};

}

namespace cached_encoder {

ggml_tensor * add_bias(
    ggml_context * context,
    ggml_tensor * value,
    ggml_tensor * bias) {
    return bias ? ggml_add(context, value, bias) : value;
}

ggml_tensor * layer_norm(
    ggml_context * context,
    ggml_tensor * value,
    ggml_tensor * weight,
    ggml_tensor * bias,
    float epsilon) {
    value = ggml_norm(context, value, epsilon);
    value = ggml_mul(context, value, weight);
    return ggml_add(context, value, bias);
}

ggml_tensor * feed_forward(
    ggml_context * context,
    ggml_tensor * value,
    ggml_tensor * norm_weight,
    ggml_tensor * norm_bias,
    ggml_tensor * first_weight,
    ggml_tensor * first_bias,
    ggml_tensor * second_weight,
    ggml_tensor * second_bias,
    float epsilon) {
    value = layer_norm(
        context, value, norm_weight, norm_bias, epsilon);
    value = add_bias(
        context,
        ggml_mul_mat(context, first_weight, value),
        first_bias);
    value = ggml_silu(context, value);
    return add_bias(
        context,
        ggml_mul_mat(context, second_weight, value),
        second_bias);
}

std::vector<float> relative_positions(int total_frames, int width) {
    const int position_count = 2 * total_frames - 1;
    std::vector<float> result(
        static_cast<size_t>(position_count) * width, 0.0f);
    const float log_base = std::log(10000.0f);
    for (int position = 0; position < position_count; ++position) {
        const int relative = total_frames - 1 - position;
        float * row =
            result.data() + static_cast<size_t>(position) * width;
        for (int index = 0; index < width / 2; ++index) {
            const float scale = std::exp(
                -static_cast<float>(2 * index) * log_base /
                static_cast<float>(width));
            row[2 * index] = std::sin(relative * scale);
            row[2 * index + 1] = std::cos(relative * scale);
        }
    }
    return result;
}

ggml_tensor * relative_shift(
    ggml_context * context,
    ggml_tensor * value,
    int query_frames,
    int key_frames,
    int heads) {
    ggml_tensor * prefix = ggml_view_3d(
        context,
        value,
        1,
        query_frames,
        heads,
        value->nb[1],
        value->nb[2],
        0);
    prefix = ggml_scale(context, ggml_cont(context, prefix), 0.0f);
    ggml_tensor * padded =
        ggml_concat(context, prefix, value, 0);
    ggml_tensor * viewed = ggml_reshape_3d(
        context, padded, query_frames, 2 * key_frames, heads);
    viewed = ggml_view_3d(
        context,
        viewed,
        query_frames,
        2 * key_frames - 1,
        heads,
        viewed->nb[1],
        viewed->nb[2],
        viewed->nb[1]);
    ggml_tensor * shifted = ggml_reshape_3d(
        context,
        ggml_cont(context, viewed),
        2 * key_frames - 1,
        query_frames,
        heads);
    shifted = ggml_view_3d(
        context,
        shifted,
        key_frames,
        query_frames,
        heads,
        shifted->nb[1],
        shifted->nb[2],
        0);
    return ggml_cont(context, shifted);
}

ggml_tensor * cached_attention(
    ggml_context * context,
    ggml_tensor * query_input,
    ggml_tensor * key_value_input,
    ggml_tensor * position_input,
    ggml_tensor * attention_mask,
    const BlockWeights & weights,
    int heads,
    int head_width,
    int query_frames,
    int key_frames) {
    ggml_tensor * query = add_bias(
        context,
        ggml_mul_mat(context, weights.attn_q_w, query_input),
        weights.attn_q_b);
    ggml_tensor * key = add_bias(
        context,
        ggml_mul_mat(context, weights.attn_k_w, key_value_input),
        weights.attn_k_b);
    ggml_tensor * value = add_bias(
        context,
        ggml_mul_mat(context, weights.attn_v_w, key_value_input),
        weights.attn_v_b);

    query = ggml_reshape_3d(
        context, query, head_width, heads, query_frames);
    key = ggml_reshape_3d(
        context, key, head_width, heads, key_frames);
    value = ggml_reshape_3d(
        context, value, head_width, heads, key_frames);

    ggml_tensor * position = ggml_mul_mat(
        context, weights.attn_pos_w, position_input);
    position = ggml_reshape_3d(
        context,
        position,
        head_width,
        heads,
        2 * key_frames - 1);

    ggml_tensor * query_permuted = ggml_cont(
        context, ggml_permute(context, query, 0, 2, 1, 3));
    ggml_tensor * key_permuted = ggml_cont(
        context, ggml_permute(context, key, 0, 2, 1, 3));
    ggml_tensor * value_permuted = ggml_cont(
        context, ggml_permute(context, value, 0, 2, 1, 3));
    ggml_tensor * position_permuted = ggml_cont(
        context, ggml_permute(context, position, 0, 2, 1, 3));

    ggml_tensor * content_bias = ggml_reshape_3d(
        context, weights.pos_bias_u, head_width, 1, heads);
    ggml_tensor * position_bias = ggml_reshape_3d(
        context, weights.pos_bias_v, head_width, 1, heads);
    ggml_tensor * biased_query_content = ggml_add(
        context, query_permuted, content_bias);
    ggml_tensor * biased_query_position = ggml_add(
        context, query_permuted, position_bias);

    ggml_tensor * position_scores = ggml_mul_mat(
        context, position_permuted, biased_query_position);
    position_scores = relative_shift(
        context,
        position_scores,
        query_frames,
        key_frames,
        heads);

    ggml_tensor * content_scores = ggml_mul_mat(
        context, key_permuted, biased_query_content);
    ggml_tensor * scores = ggml_add(
        context, content_scores, position_scores);
    const float scale = 1.0f / std::sqrt(
        static_cast<float>(head_width));
    ggml_tensor * probabilities = ggml_soft_max_ext(
        context, scores, attention_mask, scale, 0.0f);

    ggml_tensor * value_for_matmul = ggml_cont(
        context,
        ggml_permute(context, value_permuted, 1, 0, 2, 3));
    ggml_tensor * attended = ggml_mul_mat(
        context, value_for_matmul, probabilities);
    ggml_tensor * merged = ggml_cont(
        context, ggml_permute(context, attended, 0, 2, 1, 3));
    merged = ggml_reshape_2d(
        context, merged, head_width * heads, query_frames);
    return add_bias(
        context,
        ggml_mul_mat(context, weights.attn_out_w, merged),
        weights.attn_out_b);
}

ggml_tensor * update_channel_cache(
    ggml_context * context,
    ggml_tensor * cache,
    ggml_tensor * current,
    int width,
    int current_frames,
    int channel_frames) {
    if (current_frames >= channel_frames) {
        return ggml_view_2d(
            context,
            current,
            width,
            channel_frames,
            current->nb[1],
            static_cast<size_t>(
                current_frames - channel_frames) *
                current->nb[1]);
    }
    ggml_tensor * retained = ggml_view_2d(
        context,
        cache,
        width,
        channel_frames - current_frames,
        cache->nb[1],
        static_cast<size_t>(current_frames) * cache->nb[1]);
    return ggml_concat(context, retained, current, 1);
}

}

namespace {

using namespace cached_encoder;

ggml_tensor * cached_convolution(
    ggml_context * context,
    const ParakeetCtcModel & model,
    ggml_tensor * value,
    ggml_tensor * time_cache,
    const BlockWeights & weights,
    int width,
    int current_frames,
    int convolution_frames,
    float epsilon,
    ggml_tensor ** next_time_cache) {
    ggml_tensor * pointwise_weight = weights.conv_pw1_w;
    if (ggml_n_dims(pointwise_weight) != 2) {
        pointwise_weight = ggml_reshape_2d(
            context, pointwise_weight, width, 2 * width);
    }
    ggml_tensor * projected = add_bias(
        context,
        ggml_mul_mat(context, pointwise_weight, value),
        weights.conv_pw1_b);
    ggml_tensor * first_half = ggml_cont(
        context,
        ggml_view_2d(
            context,
            projected,
            width,
            current_frames,
            projected->nb[1],
            0));
    ggml_tensor * second_half = ggml_cont(
        context,
        ggml_view_2d(
            context,
            projected,
            width,
            current_frames,
            projected->nb[1],
            static_cast<size_t>(width) * projected->nb[0]));
    ggml_tensor * gated = ggml_mul(
        context, first_half, ggml_sigmoid(context, second_half));
    ggml_tensor * time_major = ggml_cont(
        context, ggml_permute(context, gated, 1, 0, 2, 3));
    ggml_tensor * convolution_input = ggml_concat(
        context, time_cache, time_major, 0);

    *next_time_cache = ggml_cont(
        context,
        ggml_view_2d(
            context,
            convolution_input,
            convolution_frames,
            width,
            convolution_input->nb[1],
            static_cast<size_t>(current_frames) *
                convolution_input->nb[0]));

    ggml_tensor * convolved = ggml_conv_1d_dw(
        context,
        weights.conv_dw_w,
        ensure_contig_on_opencl(context, model, convolution_input),
        1,
        0,
        1);
    if (weights.conv_dw_b) {
        convolved = ggml_add(
            context,
            convolved,
            ggml_reshape_2d(
                context, weights.conv_dw_b, 1, width));
    }

    ggml_tensor * normalized = ggml_cont(
        context,
        ggml_permute(context, convolved, 1, 0, 2, 3));
    normalized = layer_norm(
        context,
        normalized,
        weights.conv_norm_w,
        weights.conv_norm_b,
        epsilon);
    normalized = ggml_silu(context, normalized);

    ggml_tensor * second_pointwise_weight = weights.conv_pw2_w;
    if (ggml_n_dims(second_pointwise_weight) != 2) {
        second_pointwise_weight = ggml_reshape_2d(
            context, second_pointwise_weight, width, width);
    }
    return add_bias(
        context,
        ggml_mul_mat(
            context, second_pointwise_weight, normalized),
        weights.conv_pw2_b);
}

ggml_tensor * build_cached_block(
    ggml_context * context,
    const ParakeetCtcModel & model,
    ggml_tensor * value,
    ggml_tensor * channel_cache,
    ggml_tensor * time_cache,
    ggml_tensor * position_input,
    ggml_tensor * attention_mask,
    const BlockWeights & weights,
    const EncoderConfig & config,
    int current_frames,
    int channel_frames,
    int convolution_frames,
    ggml_tensor ** next_channel_cache,
    ggml_tensor ** next_time_cache) {
    ggml_tensor * residual = value;
    ggml_tensor * transformed = feed_forward(
        context,
        value,
        weights.norm_ff1_w,
        weights.norm_ff1_b,
        weights.ff1_l1_w,
        weights.ff1_l1_b,
        weights.ff1_l2_w,
        weights.ff1_l2_b,
        config.layer_norm_eps);
    value = ggml_add(
        context,
        residual,
        ggml_scale(context, transformed, 0.5f));

    residual = value;
    ggml_tensor * normalized_attention = layer_norm(
        context,
        value,
        weights.norm_attn_w,
        weights.norm_attn_b,
        config.layer_norm_eps);
    ggml_tensor * key_value = ggml_concat(
        context, channel_cache, normalized_attention, 1);
    *next_channel_cache = update_channel_cache(
        context,
        channel_cache,
        normalized_attention,
        config.d_model,
        current_frames,
        channel_frames);
    transformed = cached_attention(
        context,
        normalized_attention,
        key_value,
        position_input,
        attention_mask,
        weights,
        config.n_heads,
        config.head_dim,
        current_frames,
        channel_frames + current_frames);
    value = ggml_add(context, residual, transformed);

    residual = value;
    ggml_tensor * normalized_convolution = layer_norm(
        context,
        value,
        weights.norm_conv_w,
        weights.norm_conv_b,
        config.layer_norm_eps);
    transformed = cached_convolution(
        context,
        model,
        normalized_convolution,
        time_cache,
        weights,
        config.d_model,
        current_frames,
        convolution_frames,
        config.layer_norm_eps,
        next_time_cache);
    value = ggml_add(context, residual, transformed);

    residual = value;
    transformed = feed_forward(
        context,
        value,
        weights.norm_ff2_w,
        weights.norm_ff2_b,
        weights.ff2_l1_w,
        weights.ff2_l1_b,
        weights.ff2_l2_w,
        weights.ff2_l2_b,
        config.layer_norm_eps);
    value = ggml_add(
        context,
        residual,
        ggml_scale(context, transformed, 0.5f));
    return layer_norm(
        context,
        value,
        weights.norm_out_w,
        weights.norm_out_b,
        config.layer_norm_eps);
}

// `measure_bytes` (fit projection): when non-null the graph is built normally
// but its device buffer is only *sized* into `*measure_bytes`
// (ggml_gallocr_reserve_n_size, the size-only twin of the reserve the real
// path performs) and `output` is cleared before returning -- nothing is
// allocated and no host mirrors are filled.
int build_step_graph(
    const ParakeetCtcModel & model,
    int encoder_frames,
    NemotronStepGraph & output,
    size_t * measure_bytes = nullptr) {
    output.clear();
    const EncoderConfig & config = model.encoder_cfg;
    const int channel_frames = channel_cache_frames(model);
    const int convolution_frames = convolution_cache_frames(model);
    const int key_frames = channel_frames + encoder_frames;
    const size_t graph_size = GGML_DEFAULT_GRAPH_SIZE * 16;
    const size_t memory =
        ggml_tensor_overhead() * graph_size +
        ggml_graph_overhead_custom(graph_size, false) +
        128 * 1024;
    ggml_init_params params = {};
    params.mem_size = memory;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    output.context = ggml_init(params);
    if (!output.context) {
        return -1;
    }

    ggml_context * context = output.context;
    output.encoder_input = ggml_new_tensor_2d(
        context,
        GGML_TYPE_F32,
        config.d_model,
        encoder_frames);
    output.attention_mask = ggml_new_tensor_2d(
        context,
        GGML_TYPE_F32,
        key_frames,
        encoder_frames);
    output.position_input = ggml_new_tensor_2d(
        context,
        GGML_TYPE_F32,
        config.d_model,
        2 * key_frames - 1);
    ggml_set_input(output.encoder_input);
    ggml_set_input(output.attention_mask);
    ggml_set_input(output.position_input);

    ggml_tensor * value = output.encoder_input;
    output.channel_cache_inputs.reserve(config.n_layers);
    output.channel_cache_outputs.reserve(config.n_layers);
    output.time_cache_inputs.reserve(config.n_layers);
    output.time_cache_outputs.reserve(config.n_layers);
    for (int layer = 0; layer < config.n_layers; ++layer) {
        ggml_tensor * channel_cache = ggml_new_tensor_2d(
            context,
            GGML_TYPE_F32,
            config.d_model,
            channel_frames);
        ggml_tensor * time_cache = ggml_new_tensor_2d(
            context,
            GGML_TYPE_F32,
            convolution_frames,
            config.d_model);
        ggml_set_input(channel_cache);
        ggml_set_input(time_cache);
        output.channel_cache_inputs.push_back(channel_cache);
        output.time_cache_inputs.push_back(time_cache);

        ggml_tensor * next_channel_cache = nullptr;
        ggml_tensor * next_time_cache = nullptr;
        value = build_cached_block(
            context,
            model,
            value,
            channel_cache,
            time_cache,
            output.position_input,
            output.attention_mask,
            model.blocks[layer],
            config,
            encoder_frames,
            channel_frames,
            convolution_frames,
            &next_channel_cache,
            &next_time_cache);
        ggml_set_output(next_channel_cache);
        ggml_set_output(next_time_cache);
        output.channel_cache_outputs.push_back(next_channel_cache);
        output.time_cache_outputs.push_back(next_time_cache);
    }

    output.encoder_output = value;
    ggml_set_output(output.encoder_output);
    output.graph = ggml_new_graph_custom(
        context, graph_size, false);
    for (ggml_tensor * cache : output.channel_cache_outputs) {
        ggml_build_forward_expand(output.graph, cache);
    }
    for (ggml_tensor * cache : output.time_cache_outputs) {
        ggml_build_forward_expand(output.graph, cache);
    }
    ggml_build_forward_expand(output.graph, output.encoder_output);

    if (measure_bytes) {
        ggml_gallocr_t pricer = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(model.backend_active()));
        if (!pricer) {
            output.clear();
            return -2;
        }
        *measure_bytes = 0;
        ggml_gallocr_reserve_n_size(
            pricer, output.graph, nullptr, nullptr, measure_bytes);
        ggml_gallocr_free(pricer);
        output.clear();
        return 0;
    }

    output.allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(model.backend_active()));
    if (!output.allocator ||
        !ggml_gallocr_reserve(output.allocator, output.graph)) {
        output.clear();
        return -2;
    }
    output.encoder_frames = encoder_frames;
    output.channel_frames = channel_frames;
    output.convolution_frames = convolution_frames;
    output.positions = relative_positions(key_frames, config.d_model);
    return 0;
}

void fill_attention_mask(
    int cache_length,
    int current_frames,
    int channel_frames,
    std::vector<float> & mask) {
    const int key_frames =
        channel_frames + current_frames;
    const int first_valid_cache =
        channel_frames - cache_length;
    mask.assign(
        static_cast<size_t>(key_frames) * current_frames,
        -1.0e30f);
    for (int query = 0; query < current_frames; ++query) {
        float * row =
            mask.data() + static_cast<size_t>(query) * key_frames;
        for (int key = first_valid_cache;
             key < key_frames;
             ++key) {
            row[key] = 0.0f;
        }
    }
}

}

struct NemotronStreamState::Impl {
    RnntDecodeState decoder;
    std::unique_ptr<NemotronStepGraph> graph;
    IncrementalMelState incremental_mel;
    std::vector<float> mel_history;
    std::vector<float> pending_mel;
    std::string piece_text;
    int mel_width = 0;
    int mel_chunks_emitted = 0;
    int subsampling_factor = 8;
};

NemotronStreamState::NemotronStreamState()
    : impl(std::make_unique<Impl>()) {}

NemotronStreamState::~NemotronStreamState() = default;
NemotronStreamState::NemotronStreamState(
    NemotronStreamState &&) noexcept = default;
NemotronStreamState & NemotronStreamState::operator=(
    NemotronStreamState &&) noexcept = default;

void reset_nemotron_stream(NemotronStreamState & state) {
    state.cache_channel.clear();
    state.cache_time.clear();
    state.token_ids.clear();
    state.cache_length = 0;
    state.prompt_id = -1;
    state.right_context_frames = -1;
    state.step_index = 0;
    state.emitted_encoder_frames = 0;
    state.max_graph_encoder_frames = 0;
    state.cancelled = false;
    state.finalized = false;
    state.impl = std::make_unique<NemotronStreamState::Impl>();
}

void cancel_nemotron_stream(NemotronStreamState & state) {
    state.cancelled = true;
}

int nemotron_pending_mel_frames(const NemotronStreamState & state) {
    if (!state.impl || state.impl->mel_width <= 0) {
        return 0;
    }
    return static_cast<int>(
        state.impl->pending_mel.size() /
        static_cast<size_t>(state.impl->mel_width));
}

int init_nemotron_stream_state(
    const ParakeetCtcModel & model,
    const std::string & language,
    int right_context_frames,
    NemotronStreamState & state) {
    if (model.model_type != ParakeetModelType::NEMOTRON) {
        return -1;
    }
    if (std::find(
            model.nemotron_cfg.allowed_right_context_frames.begin(),
            model.nemotron_cfg.allowed_right_context_frames.end(),
            right_context_frames) ==
        model.nemotron_cfg.allowed_right_context_frames.end()) {
        return -2;
    }

    reset_nemotron_stream(state);
    state.prompt_id = resolve_nemotron_prompt_id(model, language);
    state.right_context_frames = right_context_frames;
    state.impl->subsampling_factor = encoder_subsampling_factor(model);
    const int channel_frames = channel_cache_frames(model);
    const int convolution_frames = convolution_cache_frames(model);
    state.cache_channel.assign(
        static_cast<size_t>(model.encoder_cfg.n_layers) *
            channel_frames *
            model.encoder_cfg.d_model,
        0.0f);
    state.cache_time.assign(
        static_cast<size_t>(model.encoder_cfg.n_layers) *
            model.encoder_cfg.d_model *
            convolution_frames,
        0.0f);
    return 0;
}

int append_nemotron_mel_frames(
    NemotronStreamState & state,
    const float * mel,
    int n_frames,
    int n_mels) {
    if (!state.impl ||
        state.prompt_id < 0 ||
        state.cancelled ||
        state.finalized) {
        return -1;
    }
    if (!mel || n_frames <= 0 || n_mels <= 0) {
        return -2;
    }
    if (state.impl->mel_width != 0 &&
        state.impl->mel_width != n_mels) {
        return -3;
    }
    state.impl->mel_width = n_mels;
    state.impl->pending_mel.insert(
        state.impl->pending_mel.end(),
        mel,
        mel + static_cast<size_t>(n_frames) * n_mels);
    return 0;
}

int append_nemotron_pcm(
    const ParakeetCtcModel & model,
    NemotronStreamState & state,
    const float * samples,
    int n_samples,
    bool finalize) {
    if (!state.impl ||
        state.prompt_id < 0 ||
        state.cancelled ||
        state.finalized) {
        return -1;
    }
    std::vector<float> mel;
    int mel_frames = 0;
    if (int rc = append_log_mel(
            samples,
            n_samples,
            finalize,
            model.mel_cfg,
            state.impl->incremental_mel,
            mel,
            mel_frames); rc != 0) {
        return rc;
    }
    if (mel_frames == 0) {
        return 0;
    }
    return append_nemotron_mel_frames(
        state,
        mel.data(),
        mel_frames,
        model.mel_cfg.n_mels);
}

int next_nemotron_processed_signal(
    NemotronStreamState & state,
    int n_mels,
    bool finalize,
    std::vector<float> & processed_signal,
    int & n_frames) {
    processed_signal.clear();
    n_frames = 0;
    if (state.impl &&
        state.impl->mel_width == 0 &&
        !state.cancelled &&
        !state.finalized) {
        return 0;
    }
    if (!state.impl ||
        state.prompt_id < 0 ||
        state.cancelled ||
        state.finalized ||
        n_mels <= 0 ||
        state.impl->mel_width != n_mels) {
        return -1;
    }

    const bool first = state.impl->mel_chunks_emitted == 0;
    const int factor = state.impl->subsampling_factor;
    const int steady_frames =
        factor * (state.right_context_frames + 1);
    const int required_frames =
        first ? 1 + factor * state.right_context_frames : steady_frames;
    // Non-first chunks need `factor` mel frames for one encoder frame
    // after the two-frame subsampling overlap. A 1..(factor-1) remainder
    // cannot produce an encoder frame, so finalize drops it (NeMo does
    // the same last-chunk stride).
    const int minimum_final_frames = first ? 1 : factor;
    const int pending_frames = static_cast<int>(
        state.impl->pending_mel.size() /
        static_cast<size_t>(n_mels));
    if ((!finalize && pending_frames < required_frames) ||
        (finalize && pending_frames < minimum_final_frames)) {
        return 0;
    }

    const int consumed_frames =
        std::min(required_frames, pending_frames);
    const size_t consumed_values =
        static_cast<size_t>(consumed_frames) * n_mels;
    if (!first) {
        const int history_frames = static_cast<int>(
            state.impl->mel_history.size() /
            static_cast<size_t>(n_mels));
        const int missing_history =
            std::max(0, kPreEncodeCacheFrames - history_frames);
        processed_signal.assign(
            static_cast<size_t>(missing_history) * n_mels, 0.0f);
        processed_signal.insert(
            processed_signal.end(),
            state.impl->mel_history.begin(),
            state.impl->mel_history.end());
    }
    processed_signal.insert(
        processed_signal.end(),
        state.impl->pending_mel.begin(),
        state.impl->pending_mel.begin() +
            static_cast<std::ptrdiff_t>(consumed_values));
    n_frames = static_cast<int>(
        processed_signal.size() /
        static_cast<size_t>(n_mels));

    state.impl->mel_history.insert(
        state.impl->mel_history.end(),
        state.impl->pending_mel.begin(),
        state.impl->pending_mel.begin() +
            static_cast<std::ptrdiff_t>(consumed_values));
    const size_t maximum_history_values =
        static_cast<size_t>(kPreEncodeCacheFrames) * n_mels;
    if (state.impl->mel_history.size() > maximum_history_values) {
        state.impl->mel_history.erase(
            state.impl->mel_history.begin(),
            state.impl->mel_history.end() -
                static_cast<std::ptrdiff_t>(maximum_history_values));
    }
    state.impl->pending_mel.erase(
        state.impl->pending_mel.begin(),
        state.impl->pending_mel.begin() +
            static_cast<std::ptrdiff_t>(consumed_values));
    ++state.impl->mel_chunks_emitted;
    return 1;
}

namespace {

void upload_layer_caches(
    const ParakeetCtcModel & model,
    const NemotronStepGraph & graph,
    const NemotronStreamState & state) {
    const size_t channel_layer_size =
        static_cast<size_t>(graph.channel_frames) *
        model.encoder_cfg.d_model;
    const size_t time_layer_size =
        static_cast<size_t>(graph.convolution_frames) *
        model.encoder_cfg.d_model;
    for (int layer = 0; layer < model.encoder_cfg.n_layers; ++layer) {
        ggml_backend_tensor_set(
            graph.channel_cache_inputs[layer],
            state.cache_channel.data() +
                static_cast<size_t>(layer) * channel_layer_size,
            0,
            channel_layer_size * sizeof(float));
        ggml_backend_tensor_set(
            graph.time_cache_inputs[layer],
            state.cache_time.data() +
                static_cast<size_t>(layer) * time_layer_size,
            0,
            time_layer_size * sizeof(float));
    }
}

void download_layer_caches(
    const ParakeetCtcModel & model,
    const NemotronStepGraph & graph,
    NemotronStreamState & state) {
    const size_t channel_layer_size =
        static_cast<size_t>(graph.channel_frames) *
        model.encoder_cfg.d_model;
    const size_t time_layer_size =
        static_cast<size_t>(graph.convolution_frames) *
        model.encoder_cfg.d_model;
    for (int layer = 0; layer < model.encoder_cfg.n_layers; ++layer) {
        ggml_backend_tensor_get(
            graph.channel_cache_outputs[layer],
            state.cache_channel.data() +
                static_cast<size_t>(layer) * channel_layer_size,
            0,
            channel_layer_size * sizeof(float));
        ggml_backend_tensor_get(
            graph.time_cache_outputs[layer],
            state.cache_time.data() +
                static_cast<size_t>(layer) * time_layer_size,
            0,
            time_layer_size * sizeof(float));
    }
}

int prepare_step_inputs(
    NemotronStepGraph & graph,
    const std::vector<float> & encoder_input,
    int cache_length,
    int encoder_frames) {
    ggml_backend_tensor_set(
        graph.encoder_input,
        encoder_input.data(),
        0,
        encoder_input.size() * sizeof(float));
    std::vector<float> mask;
    fill_attention_mask(
        cache_length,
        encoder_frames,
        graph.channel_frames,
        mask);
    ggml_backend_tensor_set(
        graph.attention_mask,
        mask.data(),
        0,
        mask.size() * sizeof(float));
    ggml_backend_tensor_set(
        graph.position_input,
        graph.positions.data(),
        0,
        graph.positions.size() * sizeof(float));
    return 0;
}

}

int run_nemotron_stream_step(
    ParakeetCtcModel & model,
    TdtRuntimeWeights & runtime,
    const float * processed_signal,
    int n_mel_frames,
    int n_mels,
    bool finalize,
    NemotronStreamState & state,
    NemotronStreamStepResult & result) {
    result = NemotronStreamStepResult{};
    if (state.cancelled) {
        return -1;
    }
    if (state.finalized) {
        return finalize ? 0 : -2;
    }
    if (!state.impl ||
        state.prompt_id < 0 ||
        state.right_context_frames < 0) {
        return -3;
    }
    if (!processed_signal || n_mel_frames <= 0) {
        if (finalize) {
            state.finalized = true;
            return 0;
        }
        return -4;
    }
    if (n_mels != model.mel_cfg.n_mels) {
        return -5;
    }

    const auto encoder_started = std::chrono::steady_clock::now();

    std::vector<float> subsampled;
    int subsampled_frames = 0;
    if (int rc = run_subsampling(
            model,
            processed_signal,
            n_mel_frames,
            n_mels,
            subsampled,
            subsampled_frames); rc != 0) {
        return rc;
    }

    const int dropped_frames =
        state.step_index == 0 ? 0 : kSubsamplingOverlapFrames;
    if (subsampled_frames <= dropped_frames) {
        return -6;
    }
    const int encoder_frames =
        subsampled_frames - dropped_frames;
    const int expected_frames =
        state.right_context_frames + 1;
    if (!finalize && encoder_frames != expected_frames) {
        return -7;
    }

    const int width = model.encoder_cfg.d_model;
    std::vector<float> encoder_input(
        subsampled.begin() +
            static_cast<size_t>(dropped_frames) * width,
        subsampled.end());

    if (!state.impl->graph ||
        state.impl->graph->encoder_frames != encoder_frames) {
        state.impl->graph = std::make_unique<NemotronStepGraph>();
        if (int rc = build_step_graph(
                model,
                encoder_frames,
                *state.impl->graph); rc != 0) {
            return rc;
        }
    }
    NemotronStepGraph & graph = *state.impl->graph;
    if (!ggml_gallocr_alloc_graph(graph.allocator, graph.graph)) {
        return -8;
    }

    if (int rc = prepare_step_inputs(
            graph,
            encoder_input,
            state.cache_length,
            encoder_frames); rc != 0) {
        return rc;
    }
    upload_layer_caches(model, graph, state);

    if (ggml_backend_graph_compute(
            model.backend_active(),
            graph.graph) != GGML_STATUS_SUCCESS) {
        return -9;
    }

    result.encoder_frames = encoder_frames;
    result.encoder_raw.resize(
        static_cast<size_t>(encoder_frames) * width);
    ggml_backend_tensor_get(
        graph.encoder_output,
        result.encoder_raw.data(),
        0,
        result.encoder_raw.size() * sizeof(float));
    download_layer_caches(model, graph, state);

    if (int rc = run_nemotron_prompt_projection(
            model,
            result.encoder_raw.data(),
            encoder_frames,
            width,
            state.prompt_id,
            result.encoder_conditioned); rc != 0) {
        return rc;
    }

    result.encoder_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - encoder_started).count();

    const auto decoder_started = std::chrono::steady_clock::now();

    RnntDecodeOptions options;
    options.max_symbols_per_step =
        model.nemotron_cfg.max_symbols_per_step;
    if (int rc = rnnt_decode_window(
            model,
            runtime,
            result.encoder_conditioned.data(),
            encoder_frames,
            width,
            options,
            state.impl->decoder,
            result.new_token_ids,
            result.decoder_steps); rc != 0) {
        return rc;
    }
    result.decoder_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - decoder_started).count();

    state.token_ids.insert(
        state.token_ids.end(),
        result.new_token_ids.begin(),
        result.new_token_ids.end());
    append_token_pieces(
        model.vocab, result.new_token_ids, state.impl->piece_text);
    result.text = pieces_to_text(model.vocab, state.impl->piece_text);

    state.cache_length = std::min(
        graph.channel_frames,
        state.cache_length + encoder_frames);
    ++state.step_index;
    state.emitted_encoder_frames += encoder_frames;
    state.max_graph_encoder_frames = std::max(
        state.max_graph_encoder_frames, encoder_frames);
    if (finalize) {
        state.finalized = true;
    }
    return 0;
}

int run_nemotron_cache_aware_offline(
        ParakeetCtcModel & model,
        TdtRuntimeWeights & runtime,
        const float * mel,
        int n_mel_frames,
        int n_mels,
        const std::string & language,
        int right_context_frames,
        std::atomic<bool> & cancel_flag,
        NemotronOfflineResult & result) {
    result = NemotronOfflineResult();

    NemotronStreamState state;

    if (int rc = init_nemotron_stream_state(
                model, language, right_context_frames, state); rc != 0) {
        return rc;
    }

    if (int rc = append_nemotron_mel_frames(
            state, mel, n_mel_frames, n_mels); rc != 0) {
        return rc;
    }

    while (!cancel_flag.load()) {
        std::vector<float> processed_signal;
        int processed_frames = 0;
        const int ready = next_nemotron_processed_signal(
                state, n_mels, true, processed_signal, processed_frames);

        if (ready < 0) {
            return ready;
        }

        if (ready == 0) {
            break;
        }

        const bool final_chunk = nemotron_pending_mel_frames(state) == 0;
        NemotronStreamStepResult step;
        if (int rc = run_nemotron_stream_step(
                    model,
                    runtime,
                    processed_signal.data(),
                    processed_frames,
                    n_mels,
                    final_chunk,
                    state,
                    step); rc != 0) {
            return rc;
        }

        result.encoder_ms += step.encoder_ms;
        result.decoder_ms += step.decoder_ms;
        result.text = std::move(step.text);
        if (final_chunk) {
            break;
        }
    }

    if (cancel_flag.load()) {
        cancel_nemotron_stream(state);
    } else if (!state.finalized) {
        NemotronStreamStepResult final_step;
        if (int rc = run_nemotron_stream_step(
                    model,
                    runtime,
                    nullptr,
                    0,
                    n_mels,
                    true,
                    state,
                    final_step); rc != 0) {
            return rc;
        }
    }

    result.token_ids = state.token_ids;
    result.encoder_frames = static_cast<int>(state.emitted_encoder_frames);
    return 0;
}

int nemotron_measure_stream(ParakeetCtcModel         & model,
                            int                        right_context_frames,
                            NemotronStreamFitMeasure & out) {
    out = NemotronStreamFitMeasure{};
    if (model.model_type != ParakeetModelType::NEMOTRON ||
        !model.backend_active()) {
        return -1;
    }
    const auto & allowed = model.nemotron_cfg.allowed_right_context_frames;
    if (std::find(allowed.begin(), allowed.end(), right_context_frames) ==
        allowed.end()) {
        return -2;
    }

    const EncoderConfig & config = model.encoder_cfg;
    const int factor = encoder_subsampling_factor(model);
    const int channel_frames = channel_cache_frames(model);
    const int convolution_frames = convolution_cache_frames(model);
    // Steady-state step geometry (run_nemotron_stream_step): every non-final
    // chunk yields right_context + 1 encoder frames, and the session rebuilds
    // its single step graph only when that count changes -- the final partial
    // chunk is never larger, so the steady-state graph is the session's peak.
    const int encoder_frames = right_context_frames + 1;
    const int key_frames = channel_frames + encoder_frames;

    {
        NemotronStepGraph scratch;  // measure mode clears it before returning
        size_t graph_bytes = 0;
        if (int rc = build_step_graph(
                model, encoder_frames, scratch, &graph_bytes); rc != 0) {
            return rc;
        }
        out.device_step_graph_bytes = graph_bytes;
    }

    // Per-chunk pre-encode: next_nemotron_processed_signal hands
    // run_subsampling kPreEncodeCacheFrames of mel history plus the
    // factor * (right_context + 1) freshly consumed frames. The history-free
    // first chunk (1 + factor * right_context mel frames) is smaller, so the
    // steady chunk is the per-chunk peak.
    const int n_mels = model.mel_cfg.n_mels;
    const int steady_mel = factor * encoder_frames;
    const int chunk_mel = kPreEncodeCacheFrames + steady_mel;
    size_t sub_active = 0;
    size_t sub_host_inputs = 0;
    if (int rc = measure_subsampling_compute(
            model, chunk_mel, n_mels, sub_active, sub_host_inputs);
        rc != 0) {
        return rc;
    }
    out.device_subsampling_bytes = sub_active;
    out.host_subsampling_input_bytes = sub_host_inputs;

    // Host-resident session state, in f32 elements. Geometry is
    // GGUF-validated (validate_nemotron_model), so plain size_t arithmetic
    // cannot overflow here: every factor is bounded by the fixed contract
    // (24 layers, d_model 1024, right context <= 13).
    size_t host_f32 = 0;
    const size_t d_model = (size_t) config.d_model;
    const size_t n_layers = (size_t) config.n_layers;
    host_f32 += n_layers * channel_frames * d_model;      // state.cache_channel
    host_f32 += n_layers * d_model * convolution_frames;  // state.cache_time
    host_f32 += (size_t) (2 * key_frames - 1) * d_model;  // graph.positions mirror
    host_f32 += (size_t) key_frames * encoder_frames;     // attention-mask staging
    host_f32 += (size_t) chunk_mel * n_mels;              // processed_signal
    host_f32 += (size_t) kPreEncodeCacheFrames * n_mels;  // impl->mel_history
    host_f32 += (size_t) steady_mel * n_mels;             // impl->pending_mel (one steady chunk)
    host_f32 += ((size_t) encoder_frames + kSubsamplingOverlapFrames)
                * d_model;                                // run_subsampling output
    host_f32 += (size_t) encoder_frames * d_model;        // step encoder_input
    host_f32 += 2 * (size_t) encoder_frames * d_model;    // result raw + conditioned
    host_f32 += (size_t) encoder_frames *
                ((size_t) model.nemotron_cfg.prompt_input_width + d_model);
                                                          // per-step prompt staging
    host_f32 += (2 * (size_t) model.nemotron_cfg.pred_rnn_layers + 1) *
                (size_t) model.nemotron_cfg.pred_hidden;  // RnntDecodeState h/c/pred
    // GPU runs: the scheduler keeps the subsampling graph-input originals in
    // a CPU-side buffer and copies them into the device split (host RAM;
    // zero on CPU-only runs).
    out.host_bytes = host_f32 * sizeof(float) + sub_host_inputs;
    return 0;
}

size_t nemotron_stream_graph_buffer_bytes(const NemotronStreamState & state) {
    if (!state.impl || !state.impl->graph || !state.impl->graph->allocator) {
        return 0;
    }
    return ggml_gallocr_get_buffer_size(state.impl->graph->allocator, 0);
}

}
