#include "minimax/logic.h"

#include "minimax/mm3-sample.h"
#include "minimax/request-utils.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <limits>
#include <map>
#include <stdexcept>

namespace tts_cpp::minimax::detail {
namespace {

namespace fs = std::filesystem;

uint64_t splitmix64(uint64_t & state) {
    uint64_t value = (state += UINT64_C(0x9E3779B97F4A7C15));
    value = (value ^ (value >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94D049BB133111EB);
    return value ^ (value >> 31);
}

void fill_noise_pairs(uint64_t & state, std::vector<float> & output) {
    constexpr double kUnitScale = 1.0 / 9007199254740992.0;
    constexpr double kMinimumUniform = 1e-300;
    constexpr double kTau = 6.283185307179586476925286766559;
    for (size_t index = 0; index < output.size(); index += 2) {
        double first = 0.0;
        do {
            first = static_cast<double>(splitmix64(state) >> 11) * kUnitScale;
        } while (first <= kMinimumUniform);
        const double second = static_cast<double>(splitmix64(state) >> 11) * kUnitScale;
        const double radius = std::sqrt(-2.0 * std::log(first));
        const double angle = kTau * second;
        output[index] = static_cast<float>(radius * std::cos(angle));
        if (index + 1 < output.size()) {
            output[index + 1] = static_cast<float>(radius * std::sin(angle));
        }
    }
}

int64_t sum_crop_lengths(const std::vector<int64_t> & latent_lengths, int64_t upsample,
                         const FlowWindowGeometry & geometry) {
    int64_t total = 0;
    for (size_t index = 0; index < latent_lengths.size(); ++index) {
        total += crop_span(latent_lengths[index], static_cast<int64_t>(index),
                           static_cast<int64_t>(latent_lengths.size()), upsample, geometry)
                     .length;
    }
    return total;
}

void add_error(bool condition, const std::string & message, std::vector<std::string> & errors) {
    if (!condition) {
        errors.push_back(message);
    }
}

bool find_nonfinite_sample(const std::vector<float> & audio, size_t & index) {
    for (size_t current = 0; current < audio.size(); ++current) {
        if (!std::isfinite(audio[current])) {
            index = current;
            return true;
        }
    }
    return false;
}

void clip_audio_and_measure(std::vector<float> & audio, AudioMetrics & metrics) {
    double sum_squared = 0.0;
    for (float & value : audio) {
        value = std::clamp(value, -1.0f, 1.0f);
        metrics.peak = std::max(metrics.peak, std::fabs(value));
        sum_squared += static_cast<double>(value) * static_cast<double>(value);
    }
    metrics.rms = audio.empty() ? 0.0 : std::sqrt(sum_squared / static_cast<double>(audio.size()));
}

void validate_components(const ModelCompatibility & model, std::vector<std::string> & errors) {
    std::vector<std::string> components = model.components;
    std::sort(components.begin(), components.end());
    add_error(components == std::vector<std::string>({"cond", "depth", "dit", "vocoder"}),
              "synth components must be exactly cond, depth, dit, and vocoder", errors);
}

void validate_dimensions(const ModelCompatibility & model, std::vector<std::string> & errors) {
    add_error(model.lm_embedding > 0 && model.lm_embedding == model.depth_embedding,
              "LM embedding dimension must match depth embedding dimension", errors);
    add_error(model.lm_embedding > 0 && model.lm_embedding == model.condition_hidden,
              "LM embedding dimension must match condition hidden dimension", errors);
    add_error(model.lm_codebooks > 0 && model.lm_codebooks == model.depth_codebooks,
              "LM codebook count must match depth codebook count", errors);
    add_error(model.lm_codebooks > 0 && model.lm_codebooks == model.condition_layers,
              "LM codebook count must match condition layer count", errors);
    add_error(model.lm_acoustic_vocab > 0 && model.lm_acoustic_vocab == model.depth_acoustic_vocab,
              "LM acoustic vocabulary must match depth acoustic vocabulary", errors);
    add_error(model.condition_out > 0 && model.condition_out == model.dit_condition,
              "condition output dimension must match DiT condition dimension", errors);
    add_error(model.dit_channels > 0 && model.dit_channels == model.vocoder_latent_channels,
              "DiT latent channels must match vocoder latent channels", errors);
}

bool has_positive_condition_rate(const ModelCompatibility & model) {
    return model.condition_rate.input_sampling_rate > 0 && model.condition_rate.input_hop_length > 0 &&
           model.condition_rate.output_sampling_rate > 0 && model.condition_rate.output_hop_length > 0;
}

void validate_rates(const ModelCompatibility & model, std::vector<std::string> & errors) {
    add_error(model.frame_rate == kDefaultFrameRate, "LM frame rate must be 25", errors);
    add_error(model.max_audio_frames == kDefaultMaxFrames, "LM maximum audio frames must be 9000", errors);
    add_error(model.max_prompt_tokens > 0, "LM maximum prompt tokens must be positive", errors);
    add_error(has_positive_condition_rate(model), "condition rates and hop lengths must be positive", errors);
    if (has_positive_condition_rate(model)) {
        add_error(model.condition_rate.input_sampling_rate ==
                      model.frame_rate * model.condition_rate.input_hop_length,
                  "condition input rate and hop must match LM frame rate", errors);
        add_error(model.condition_rate.output_sampling_rate == model.vocoder_sampling_rate,
                  "condition output rate must match vocoder sampling rate", errors);
        add_error(model.condition_rate.output_hop_length == model.vocoder_upsample,
                  "condition output hop must match vocoder upsample", errors);
    }
    add_error(model.vocoder_sampling_rate > 0, "vocoder sampling rate must be positive", errors);
    add_error(model.vocoder_channels == 2, "vocoder channels must be 2", errors);
}

void validate_windows(const ModelCompatibility & model, std::vector<std::string> & errors) {
    add_error(model.window_frames == kWindowFrames, "DiT window frames must be 200", errors);
    add_error(model.hop_frames > 0 && model.hop_frames <= model.window_frames,
              "DiT hop frames must be positive and at most the window", errors);
    if (!has_positive_condition_rate(model) || model.window_frames <= 0 || model.hop_frames <= 0) {
        return;
    }
    add_error(model.window_latents == condition_latent_length(model.condition_rate, model.window_frames),
              "DiT window latents do not match the condition rate", errors);
    add_error(model.hop_latents == condition_latent_length(model.condition_rate, model.hop_frames),
              "DiT hop latents do not match the condition rate", errors);
    add_error(model.window_latents - model.hop_latents >= 4,
              "DiT window and hop must leave a shared region of at least 4 latents", errors);
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return value;
}

struct CanonicalName {
    std::string role;
    std::string quant;
};

CanonicalName parse_canonical_name(const fs::path & path) {
    const std::string name = lowercase(path.filename().string());
    constexpr const char * kLmPrefix = "mm3-lm-";
    constexpr const char * kSynthPrefix = "mm3-synth-";
    constexpr const char * kSuffix = ".gguf";
    const std::string prefix = name.rfind(kLmPrefix, 0) == 0
                                   ? kLmPrefix
                                   : (name.rfind(kSynthPrefix, 0) == 0 ? kSynthPrefix : "");
    if (prefix.empty() || name.size() <= prefix.size() + std::char_traits<char>::length(kSuffix) ||
        name.compare(name.size() - std::char_traits<char>::length(kSuffix),
                     std::char_traits<char>::length(kSuffix), kSuffix) != 0) {
        return {};
    }
    return {prefix == kLmPrefix ? "lm" : "synth",
            name.substr(prefix.size(), name.size() - prefix.size() - std::char_traits<char>::length(kSuffix))};
}

using CandidateMap = std::map<std::string, std::vector<std::string>>;

void collect_directory_candidates(const fs::path & directory, CandidateMap & lm, CandidateMap & synth) {
    std::error_code error;
    if (!fs::is_directory(directory, error)) {
        return;
    }
    for (const fs::directory_entry & entry : fs::directory_iterator(directory, error)) {
        if (!entry.is_regular_file(error)) {
            continue;
        }
        const CanonicalName name = parse_canonical_name(entry.path());
        if (name.role == "lm") {
            lm[name.quant].push_back(entry.path().string());
        } else if (name.role == "synth") {
            synth[name.quant].push_back(entry.path().string());
        }
    }
}

void reject_duplicate_candidates(const CandidateMap & candidates, const std::string & role) {
    for (const auto & candidate : candidates) {
        if (candidate.second.size() > 1) {
            throw std::runtime_error("minimax engine: ambiguous " + role + " candidates for quant " +
                                     candidate.first);
        }
    }
}

std::string candidate_for_quant(const CandidateMap & candidates, const std::string & quant) {
    const auto match = candidates.find(quant);
    return match == candidates.end() ? "" : match->second.front();
}

ModelPair select_prioritized_pair(const CandidateMap & lm, const CandidateMap & synth) {
    // Every variant acestep-quantize can emit, best fidelity first after the
    // established q8_0 > f16 > bf16 head. f32 sits last: a dequantized-to-F32
    // pair (scripts/dequant_gguf.py) is a diagnostic reference, only picked
    // when nothing production-shaped exists.
    static const std::vector<std::string> priorities = {"q8_0",   "f16",    "bf16",   "q6_k",
                                                        "q5_k_m", "q5_k_s", "q4_k_m", "q4_k_s",
                                                        "q3_k_l", "q3_k_m", "q3_k_s", "q2_k",
                                                        "f32"};
    for (const std::string & quant : priorities) {
        const std::string lm_path = candidate_for_quant(lm, quant);
        const std::string synth_path = candidate_for_quant(synth, quant);
        if (!lm_path.empty() && !synth_path.empty()) {
            return {lm_path, synth_path, quant};
        }
    }
    throw std::runtime_error("minimax engine: no matched canonical LM and synth GGUF pair found");
}

std::string normalized_directory(const std::string & directory) {
    if (directory.empty()) {
        return "";
    }
    fs::path normalized = fs::path(directory).lexically_normal();
    if (!normalized.has_filename() && normalized != normalized.root_path()) {
        normalized = normalized.parent_path();
    }
    return normalized.string();
}

}

int64_t frames_from_duration(double duration, int frame_rate, int max_frames) {
    if (!std::isfinite(duration) || duration <= 0.0) {
        throw std::invalid_argument("duration must be finite and greater than zero");
    }
    if (frame_rate <= 0) {
        throw std::invalid_argument("frame rate must be greater than zero");
    }
    return validate_frames(static_cast<int64_t>(std::llround(duration * frame_rate)), max_frames);
}

int64_t validate_frames(int64_t frames, int max_frames) {
    if (max_frames <= 0) {
        throw std::invalid_argument("maximum frame count must be greater than zero");
    }
    if (frames <= 0) {
        throw std::invalid_argument("frame count must be greater than zero");
    }
    return frames > max_frames ? max_frames : frames;
}

std::string build_prompt(const std::string & caption, const std::string & lyrics) {
    if (mm3_str_blank(caption)) {
        throw std::invalid_argument("caption must not be empty");
    }
    return mm3_assemble_prompt(caption, lyrics);
}

std::vector<int32_t> mask_unconditional(const std::vector<int32_t> & conditional, int32_t mask_token) {
    std::vector<int32_t> result = conditional;
    if (result.size() < 4) {
        return result;
    }
    for (size_t index = 1; index + 2 < result.size(); ++index) {
        result[index] = mask_token;
    }
    return result;
}

void fill_noise(uint64_t seed, int64_t window, std::vector<float> & output, int64_t count) {
    if (window < 0 || count < 0) {
        throw std::invalid_argument("noise window and count must not be negative");
    }
    uint64_t state =
        seed ^ (UINT64_C(0xA24BAED4963EE407) * static_cast<uint64_t>(window + 1));
    splitmix64(state);
    output.resize(static_cast<size_t>(count));
    fill_noise_pairs(state, output);
}

void flow_schedule(int steps, std::vector<float> & sigmas, std::vector<float> & timesteps) {
    if (steps <= 0) {
        throw std::invalid_argument("flow steps must be greater than zero");
    }
    sigmas.assign(static_cast<size_t>(steps) + 1, 0.0f);
    timesteps.assign(static_cast<size_t>(steps), 0.0f);
    const double start = 1.0;
    const double stop = 1.0 / static_cast<double>(steps);
    const double delta = steps > 1 ? (stop - start) / static_cast<double>(steps - 1) : 0.0;
    for (int index = 0; index < steps; ++index) {
        const double linear = index == steps - 1 ? stop : static_cast<double>(index) * delta + start;
        const float value = 1.0f - static_cast<float>(linear);
        sigmas[static_cast<size_t>(index)] = value;
        timesteps[static_cast<size_t>(index)] = value;
    }
    sigmas[static_cast<size_t>(steps)] = 1.0f;
}

void blend_latent_overlap(float * latents, const float * noise, const float * previous,
                          int64_t channels, int64_t latent_length, int64_t overlap,
                          int64_t previous_stride, float timestep) {
    const float noise_scale = 1.0f - (1.0f - 1e-6f) * timestep;
    for (int64_t channel = 0; channel < channels; ++channel) {
        const float * channel_noise = noise + channel * latent_length;
        const float * channel_previous = previous + channel * previous_stride;
        float * channel_latents = latents + channel * latent_length;
        for (int64_t index = 0; index < overlap; ++index) {
            channel_latents[index] =
                noise_scale * channel_noise[index] + timestep * channel_previous[index];
        }
    }
}

void pin_latent_overlap(float * latents, const float * previous, int64_t channels,
                        int64_t latent_length, int64_t overlap, int64_t previous_stride) {
    for (int64_t channel = 0; channel < channels; ++channel) {
        memcpy(latents + channel * latent_length, previous + channel * previous_stride,
               static_cast<size_t>(overlap) * sizeof(float));
    }
}

int64_t condition_latent_length(const ConditionRate & rate, int64_t frames) {
    if (frames <= 0 || rate.input_sampling_rate <= 0 || rate.input_hop_length <= 0 ||
        rate.output_sampling_rate <= 0 || rate.output_hop_length <= 0) {
        throw std::invalid_argument("condition rate and frame values must be greater than zero");
    }
    const double value = static_cast<double>(frames) * rate.output_sampling_rate /
                         rate.input_sampling_rate * rate.input_hop_length / rate.output_hop_length;
    const int64_t length = static_cast<int64_t>(value);
    return length < 1 ? 1 : length;
}

std::vector<int64_t> window_starts(int64_t frames, int64_t window_frames, int64_t hop_frames) {
    if (frames <= 0 || window_frames <= 0 || hop_frames <= 0 || hop_frames > window_frames) {
        throw std::invalid_argument("window values are invalid");
    }
    std::vector<int64_t> starts;
    if (frames <= window_frames) {
        starts.push_back(0);
        return starts;
    }
    // Keep adding windows until one reaches the end. Stopping at
    // `frames - hop_frames` only covers the tail while a window spans exactly
    // two hops, and silently drops it for any wider hop.
    for (int64_t start = 0;; start += hop_frames) {
        starts.push_back(start);
        if (start + window_frames >= frames) {
            break;
        }
    }
    return starts;
}

CropSpan crop_span(int64_t latent_length, int64_t window_index, int64_t window_count, int64_t upsample,
                   const FlowWindowGeometry & geometry) {
    if (latent_length < 0 || window_index < 0 || window_count <= 0 || window_index >= window_count ||
        upsample <= 0 || geometry.crop_left < 0 || geometry.crop_right < 0) {
        throw std::invalid_argument("crop values are invalid");
    }
    const int64_t left = window_index == 0 ? 0 : geometry.crop_left * upsample;
    const int64_t right = window_index == window_count - 1 ? 0 : geometry.crop_right * upsample;
    const int64_t available = latent_length * upsample - left - right;
    return {left, available > 0 ? available : 0};
}

CarryRange carry_range(int64_t latent_length, int64_t carry_span, int64_t overlap) {
    if (latent_length < 0 || carry_span < 0 || overlap < 0 || overlap > carry_span) {
        throw std::invalid_argument("carry values are invalid");
    }
    const int64_t start = std::max<int64_t>(latent_length - carry_span, 0);
    const int64_t end = std::max<int64_t>(latent_length - overlap, start);
    return {start, end};
}

bool copy_carry_layout(const std::vector<float> & latents, const std::vector<float> & condition,
                       int64_t channels, int64_t condition_dimension, int64_t latent_length,
                       CarryRange range, std::vector<float> & carry_latents,
                       std::vector<float> & carry_condition) {
    if (channels <= 0 || condition_dimension <= 0 || latent_length < 0 ||
        range.start < 0 || range.end < range.start || range.end > latent_length) {
        return false;
    }
    const int64_t length = range.end - range.start;
    if (!copy_range_fits(latents.size(), latents.size(), 0, 0, channels * latent_length) ||
        !copy_range_fits(condition.size(), condition.size(), 0, 0,
                         condition_dimension * latent_length)) {
        return false;
    }
    carry_latents.assign(static_cast<size_t>(channels * length), 0.0f);
    for (int64_t channel = 0; channel < channels; ++channel) {
        memcpy(carry_latents.data() + channel * length,
               latents.data() + channel * latent_length + range.start,
               static_cast<size_t>(length) * sizeof(float));
    }
    carry_condition.assign(static_cast<size_t>(condition_dimension * length), 0.0f);
    memcpy(carry_condition.data(), condition.data() + range.start * condition_dimension,
           static_cast<size_t>(condition_dimension * length) * sizeof(float));
    return true;
}

bool copy_planar_window(const std::vector<float> & source, int64_t channels,
                        int64_t source_length, int64_t source_offset,
                        int64_t destination_length, int64_t destination_offset,
                        int64_t copy_length, std::vector<float> & destination) {
    if (channels <= 0 || source_length < 0 || destination_length < 0 ||
        source_offset < 0 || destination_offset < 0 || copy_length < 0 ||
        source_offset > source_length || copy_length > source_length - source_offset ||
        destination_offset > destination_length ||
        copy_length > destination_length - destination_offset ||
        static_cast<uint64_t>(channels) * static_cast<uint64_t>(source_length) != source.size() ||
        static_cast<uint64_t>(channels) * static_cast<uint64_t>(destination_length) !=
            destination.size()) {
        return false;
    }
    for (int64_t channel = 0; channel < channels; ++channel) {
        memcpy(destination.data() + channel * destination_length + destination_offset,
               source.data() + channel * source_length + source_offset,
               static_cast<size_t>(copy_length) * sizeof(float));
    }
    return true;
}

int64_t stitched_sample_count(const std::vector<int64_t> & latent_lengths, int64_t upsample,
                              const FlowWindowGeometry & geometry) {
    if (latent_lengths.empty()) {
        return 0;
    }
    return sum_crop_lengths(latent_lengths, upsample, geometry);
}

std::string vocoder_upsample_error(const std::vector<int32_t> & rates, uint32_t total_upsample) {
    if (rates.empty()) {
        return "vocoder upsample rates must not be empty";
    }
    uint32_t product = 1;
    for (const int32_t rate : rates) {
        if (rate <= 0) {
            return "vocoder upsample rates must be positive";
        }
        const uint32_t positive_rate = static_cast<uint32_t>(rate);
        if (product > std::numeric_limits<uint32_t>::max() / positive_rate) {
            return "vocoder upsample rate product overflows uint32";
        }
        product *= positive_rate;
    }
    if (product != total_upsample) {
        return "vocoder upsample rate product must equal total upsample";
    }
    return "";
}

std::vector<std::string> validate_synthesis_contract(const SynthesisContract & contract) {
    std::vector<std::string> errors;
    add_error(contract.scheduler == "FlowMatchEulerDiscrete",
              "flow scheduler must be FlowMatchEulerDiscrete", errors);
    add_error(contract.invert_sigmas, "flow invert_sigmas must be true", errors);
    add_error(std::isfinite(contract.shift) && contract.shift == 1.0f,
              "flow shift must be finite and exactly 1", errors);
    add_error(contract.train_timesteps == 1, "flow train timesteps must be 1", errors);
    add_error(contract.rope_type == "neox", "DiT rope type must be neox", errors);
    add_error(contract.glu_order == "value_gate", "DiT GLU order must be value_gate", errors);
    add_error(contract.timestep_token_prepended, "DiT timestep token must be prepended", errors);
    add_error(contract.pre_post_conv_residual, "DiT pre/post convolution residual must be enabled", errors);
    add_error(!contract.attn_bias, "DiT attention bias must be disabled", errors);
    return errors;
}

std::string vocoder_output_shape_error(int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
                                       int64_t elements, size_t bytes, int64_t expected_length) {
    if (expected_length <= 0) {
        return "vocoder expected output length must be positive";
    }
    if (ne0 != expected_length || ne1 != 1 || ne2 != 1 || ne3 != 1) {
        return "vocoder output dimensions do not match the expected mono waveform";
    }
    if (elements != expected_length) {
        return "vocoder output element count does not match the expected waveform length";
    }
    if (static_cast<uint64_t>(expected_length) >
        static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(float))) {
        return "vocoder expected output byte count overflows size_t";
    }
    if (bytes != static_cast<size_t>(expected_length) * sizeof(float)) {
        return "vocoder output byte count does not match the expected F32 waveform";
    }
    return "";
}

bool copy_range_fits(size_t source_elements, size_t destination_elements,
                     int64_t source_offset, int64_t destination_offset, int64_t copy_length) {
    if (source_offset < 0 || destination_offset < 0 || copy_length < 0) {
        return false;
    }
    const uint64_t source_start = static_cast<uint64_t>(source_offset);
    const uint64_t destination_start = static_cast<uint64_t>(destination_offset);
    const uint64_t count = static_cast<uint64_t>(copy_length);
    return source_start <= source_elements && count <= source_elements - source_start &&
           destination_start <= destination_elements && count <= destination_elements - destination_start;
}

bool stereo_copy_ranges_fit(size_t source_elements, size_t destination_elements,
                            int64_t source_length, int64_t source_offset,
                            int64_t destination_length, int64_t destination_offset,
                            int64_t copy_length) {
    if (source_length < 0 || source_offset < 0 || destination_length < 0 ||
        destination_offset < 0 || copy_length < 0) {
        return false;
    }
    if (source_offset > source_length || copy_length > source_length - source_offset) {
        return false;
    }
    if (destination_offset > destination_length ||
        copy_length > destination_length - destination_offset) {
        return false;
    }
    const uint64_t source_stereo = static_cast<uint64_t>(source_length) * 2;
    const uint64_t destination_stereo = static_cast<uint64_t>(destination_length) * 2;
    if (source_stereo != source_elements || destination_stereo != destination_elements) {
        return false;
    }
    for (int64_t channel = 0; channel < 2; ++channel) {
        if (!copy_range_fits(source_elements, destination_elements,
                             channel * source_length + source_offset,
                             channel * destination_length + destination_offset, copy_length)) {
            return false;
        }
    }
    return true;
}

bool finalize_audio(std::vector<float> & audio, AudioMetrics & metrics, size_t & nonfinite_index) {
    metrics = {};
    nonfinite_index = 0;
    if (find_nonfinite_sample(audio, nonfinite_index)) {
        return false;
    }
    clip_audio_and_measure(audio, metrics);
    return true;
}

int64_t sample_top_k(const float * logits, int64_t count, int top_k, std::mt19937_64 & random) {
    return mm3_sample_top_k(logits, count, top_k, random);
}

bool collect_ar_candidates(const float * logits, int64_t row_stride, int64_t eos,
                           int64_t semantic_offset, int64_t semantic_vocab,
                           ArCandidates & candidates, int64_t & nonfinite_count) {
    if (!logits || row_stride <= 0 || eos < 0 || eos >= row_stride ||
        semantic_offset < 0 || semantic_vocab <= 0 ||
        semantic_offset > row_stride - semantic_vocab) {
        return false;
    }
    const int64_t count = semantic_vocab + 1;
    candidates.conditional.resize(static_cast<size_t>(count));
    candidates.unconditional.resize(static_cast<size_t>(count));
    candidates.guided.resize(static_cast<size_t>(count));
    const float * conditional = logits;
    const float * unconditional = logits + row_stride;
    const auto normalize = [&nonfinite_count](float value) {
        if (std::isnan(value) || (std::isinf(value) && value > 0.0f)) {
            ++nonfinite_count;
            return -INFINITY;
        }
        return value;
    };
    candidates.conditional[0] = normalize(conditional[eos]);
    candidates.unconditional[0] = normalize(unconditional[eos]);
    for (int64_t index = 0; index < semantic_vocab; ++index) {
        candidates.conditional[static_cast<size_t>(index + 1)] =
            normalize(conditional[semantic_offset + index]);
        candidates.unconditional[static_cast<size_t>(index + 1)] =
            normalize(unconditional[semantic_offset + index]);
    }
    return true;
}

int64_t compact_head_row_count(int64_t semantic_vocab) {
    return semantic_vocab + 1;
}

bool compact_head_copy_plan(int64_t vocab, int64_t semantic_offset, int64_t semantic_vocab, int64_t eos,
                            size_t row_size, std::array<CompactHeadCopy, 2> & plan) {
    if (row_size == 0 || vocab <= 0 || semantic_offset < 0 || semantic_vocab <= 0 ||
        semantic_offset > vocab - semantic_vocab || eos < 0 || eos >= vocab) {
        return false;
    }
    plan[0] = { static_cast<size_t>(semantic_offset) * row_size, 0, static_cast<size_t>(semantic_vocab) * row_size };
    plan[1] = { static_cast<size_t>(eos) * row_size, static_cast<size_t>(semantic_vocab) * row_size, row_size };
    return true;
}

void guide_ar_candidates(ArCandidates & candidates, float cfg_scale) {
    const size_t count = candidates.conditional.size();
    candidates.guided.resize(count);
    for (size_t index = 0; index < count; ++index) {
        const float unconditional = candidates.unconditional[index];
        candidates.guided[index] =
            unconditional + (candidates.conditional[index] - unconditional) * cfg_scale;
    }
}

void apply_conditional_top_k(ArCandidates & candidates, int top_k) {
    const int64_t count = static_cast<int64_t>(candidates.conditional.size());
    int64_t selected = std::min<int64_t>(std::max(top_k, 1), count);
    float threshold = -INFINITY;
    if (selected < count) {
        std::vector<float> scratch = candidates.conditional;
        std::nth_element(scratch.begin(), scratch.begin() + selected - 1, scratch.end(),
                         std::greater<float>());
        threshold = scratch[static_cast<size_t>(selected - 1)];
    }
    for (int64_t index = 0; index < count; ++index) {
        if (candidates.conditional[static_cast<size_t>(index)] < threshold) {
            candidates.guided[static_cast<size_t>(index)] = -INFINITY;
        }
    }
}

void build_acoustic_rows(const int32_t * codes, int64_t codebooks, int64_t acoustic_vocab,
                         std::vector<int32_t> & rows) {
    rows.resize(static_cast<size_t>(codebooks));
    for (int64_t index = 0; index < codebooks; ++index) {
        rows[static_cast<size_t>(index)] =
            codes[index] + static_cast<int32_t>(index * acoustic_vocab);
    }
}

int64_t resolve_ar_frame_cap(int64_t requested_frames, int64_t checkpoint_frames,
                             bool forced, int64_t forced_length, std::string & error) {
    error.clear();
    if (requested_frames <= 0) {
        error = "max_frames must be positive";
        return 0;
    }
    int64_t frames = checkpoint_frames > 0
                         ? std::min(requested_frames, checkpoint_frames)
                         : requested_frames;
    if (!forced) {
        return frames;
    }
    if (forced_length <= 0) {
        error = "forced replay needs both forced_semantic and forced_acoustic, and a positive length";
        return 0;
    }
    frames = std::min(frames, forced_length - 1);
    if (frames <= 0) {
        error = "forced replay needs at least 2 iterations (one un-emitted, one emitted)";
        return 0;
    }
    return frames;
}

// mm3.flow.steps is written by our own converter from a fixed constant, so it
// carries no per-checkpoint information and only records what the engine
// recommended when the file was built. The engine's current recommendation is
// used instead, so a measured change reaches files converted earlier; the
// metadata value is validated but not consulted.
int resolve_flow_steps(int requested_steps) {
    return requested_steps > 0 ? requested_steps : kDefaultFlowSteps;
}

// The shared region is rounded down to a multiple of four so the quarter split
// is exact. Between that and latent_length's truncation per window, each seam
// can land up to three latents from the exact position in either direction;
// the stitcher concatenates the kept spans as they are and the output length
// moves by that much. At the shipped hop the slip is one latent.
FlowWindowGeometry flow_window_geometry(int64_t window_latents, int64_t hop_latents) {
    if (hop_latents <= 0 || window_latents <= hop_latents) {
        throw std::invalid_argument("window latents must exceed a positive hop");
    }
    const int64_t shared = (window_latents - hop_latents) / 4 * 4;
    if (shared <= 0) {
        throw std::invalid_argument("window and hop leave no shared region");
    }
    FlowWindowGeometry geometry;
    geometry.carry_span = shared;
    geometry.overlap    = shared / 2;
    geometry.crop_left  = shared / 4;
    geometry.crop_right = shared - geometry.crop_left;
    return geometry;
}

// The depth decoder's sequence grows by exactly one code per step, so step cb
// attends over positions 0..cb. The first step is the exception: it seeds both
// the projected LM hidden and the semantic code, so it writes two positions.
DepthStepLayout depth_step_layout(int codebook) {
    if (codebook < 1) {
        throw std::invalid_argument("depth codebook index must be positive");
    }
    DepthStepLayout layout;
    layout.window = (int64_t) codebook + 1;
    layout.tokens = codebook == 1 ? 2 : 1;
    layout.first  = layout.window - layout.tokens;
    return layout;
}

std::vector<std::string> validate_model_compatibility(const ModelCompatibility & model) {
    std::vector<std::string> errors;
    validate_components(model, errors);
    validate_dimensions(model, errors);
    validate_rates(model, errors);
    validate_windows(model, errors);
    return errors;
}

ModelPair resolve_model_pair(const std::string & model_dir, const std::string & explicit_lm,
                             const std::string & explicit_synth) {
    const CanonicalName lm_name = parse_canonical_name(explicit_lm);
    const CanonicalName synth_name = parse_canonical_name(explicit_synth);
    if (!explicit_lm.empty() && !explicit_synth.empty()) {
        if (!lm_name.quant.empty() && !synth_name.quant.empty() && lm_name.quant != synth_name.quant) {
            throw std::runtime_error("minimax engine: explicit canonical GGUF files use different quants");
        }
        return {explicit_lm, explicit_synth, lm_name.quant};
    }
    if (model_dir.empty()) {
        throw std::runtime_error("minimax engine: model directory is required when a GGUF path is missing");
    }
    CandidateMap lm;
    CandidateMap synth;
    collect_directory_candidates(fs::path(model_dir) / "mm3", lm, synth);
    collect_directory_candidates(model_dir, lm, synth);
    reject_duplicate_candidates(lm, "LM");
    reject_duplicate_candidates(synth, "synth");
    if (!explicit_lm.empty() || !explicit_synth.empty()) {
        const CanonicalName explicit_name = explicit_lm.empty() ? synth_name : lm_name;
        if (explicit_name.quant.empty()) {
            throw std::runtime_error(
                "minimax engine: an explicit noncanonical GGUF path requires both model paths");
        }
        const std::string counterpart =
            candidate_for_quant(explicit_lm.empty() ? lm : synth, explicit_name.quant);
        if (counterpart.empty()) {
            throw std::runtime_error("minimax engine: matching canonical GGUF counterpart is missing");
        }
        return explicit_lm.empty() ? ModelPair{counterpart, explicit_synth, explicit_name.quant}
                                   : ModelPair{explicit_lm, counterpart, explicit_name.quant};
    }
    return select_prioritized_pair(lm, synth);
}

bool backend_configuration_matches(int active_references, int active_threads,
                                   const std::string & active_backends_dir, int requested_threads,
                                   const std::string & requested_backends_dir) {
    return active_references <= 0 ||
           (active_threads == requested_threads &&
            normalized_directory(active_backends_dir) == normalized_directory(requested_backends_dir));
}

bool engine_instance_available(int active_instances) {
    return active_instances == 0;
}

bool cancellation_requested(const std::function<bool()> & callback) {
    return callback && callback();
}

}
