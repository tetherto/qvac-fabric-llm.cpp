#include "pocket/coreml.h"

#include "coreml_sidecar_path.h"

#ifdef TTS_CPP_USE_COREML
#include "moss/coreml/runner.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <stdexcept>

namespace tts_cpp::pocket::detail {
namespace {

[[noreturn]] void fail(const std::string & message) {
    throw std::runtime_error("Pocket TTS Core ML step: " + message);
}

#ifdef TTS_CPP_USE_COREML
constexpr int MAX_RANK = 8;
constexpr int METADATA_BYTES = 256;

std::string dims_text(const std::vector<int64_t> & dims) {
    std::ostringstream text;
    text << '[';
    for (size_t i = 0; i < dims.size(); ++i) text << (i ? ", " : "") << dims[i];
    text << ']';
    return text.str();
}

std::vector<int64_t> feature_dims(moss_coreml_model * model, const char * name, bool state) {
    int64_t dims[MAX_RANK] = {};
    const int rank = state ? moss_coreml_state_dims(model, name, dims, MAX_RANK)
                           : moss_coreml_feature_dims(model, name, dims, MAX_RANK);
    return std::vector<int64_t>(dims, dims + std::max(rank, 0));
}

std::string metadata(moss_coreml_model * model, const char * key) {
    char value[METADATA_BYTES] = {};
    return moss_coreml_metadata(model, key, value, METADATA_BYTES) < 0 ? std::string() : std::string(value);
}

void require_finite(const std::vector<float> & values, const char * name) {
    if (!std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); }))
        fail(std::string("non-finite ") + name);
}

struct expected_feature {
    std::string name;
    bool state;
    std::vector<int64_t> dims;
};

void require_features(moss_coreml_model * model, const std::string & path, const std::vector<expected_feature> & expected) {
    for (const auto & feature : expected) {
        const auto declared = feature_dims(model, feature.name.c_str(), feature.state);
        if (declared != feature.dims)
            fail(path + " declares " + feature.name + " " + dims_text(declared) + " where the runtime needs " +
                 dims_text(feature.dims));
    }
}

void require_stamp(moss_coreml_model * model, const std::string & path, const std::string & source_hash) {
    if (metadata(model, "pocket.coreml.schema") != COREML_SCHEMA)
        fail(path + " has an unsupported schema; re-export it with scripts/export-pocket-coreml.py");
    if (metadata(model, "pocket.source_sha256") != source_hash)
        fail(path + " was exported from another checkpoint than its GGUF");
}

using Geometry = CoremlDecoderGeometry;

// (name, channels, length) of each Mimi causal-conv history, as
// histories() in scripts/pocket_coreml_mimi.py orders them.
struct history_spec {
    const char * name;
    const char * output;
    int channels, length;
};
constexpr std::array<history_spec, 9> HISTORIES = {{
    {"h_upsample", "new_h_upsample", 512, 1}, {"h_conv0", "new_h_conv0", 512, 6},
    {"h_convtr0", "new_h_convtr0", 512, 1}, {"h_residual0", "new_h_residual0", 256, 2},
    {"h_convtr1", "new_h_convtr1", 256, 1}, {"h_residual1", "new_h_residual1", 128, 2},
    {"h_convtr2", "new_h_convtr2", 128, 1}, {"h_residual2", "new_h_residual2", 64, 2},
    {"h_final", "new_h_final", 64, 2},
}};
constexpr int DECODER_ROWS = Geometry::frames * Geometry::upsample;
constexpr size_t KV_SIZE = size_t(Geometry::layers) * Geometry::heads * Geometry::keep * Geometry::head_dim;
#endif

} // namespace

bool coreml_valid_units(const std::string & units) {
    return units == "all" || units == "cpu_and_ane" || units == "cpu_and_gpu" || units == "cpu_only";
}

std::string coreml_units(const CoremlPolicy & policy, const char * sidecar_default) {
    return policy.compute_units.empty() ? sidecar_default : policy.compute_units;
}

CoremlPolicy resolve_coreml_policy(const std::string & compute_units, std::optional<bool> strict) {
    CoremlPolicy policy;
    policy.disabled = std::getenv(COREML_DISABLE_ENV) != nullptr;
    policy.strict = strict.has_value() ? *strict : std::getenv(COREML_STRICT_ENV) != nullptr;
    if (!compute_units.empty()) {
        if (!coreml_valid_units(compute_units))
            throw std::runtime_error("Pocket TTS: invalid coreml_compute_units='" + compute_units +
                                     "' (expected all, cpu_and_ane, cpu_and_gpu or cpu_only)");
        policy.compute_units = compute_units;
    } else if (const char * env = std::getenv(COREML_UNITS_ENV); env && *env) {
        if (coreml_valid_units(env)) policy.compute_units = env;
        else std::fprintf(stderr, "Pocket TTS: ignoring %s=%s (expected all, cpu_and_ane, cpu_and_gpu or cpu_only)\n",
                          COREML_UNITS_ENV, env);
    }
    return policy;
}

std::string coreml_step_path(const std::string & flow_lm_gguf) {
    return ::tts_cpp::detail::coreml_sidecar_path(flow_lm_gguf, COREML_STEP_SUFFIX);
}

std::string coreml_decoder_path(const std::string & mimi_gguf) {
    return ::tts_cpp::detail::coreml_sidecar_path(mimi_gguf, COREML_DECODER_SUFFIX);
}

namespace {
bool sidecar_enabled(const std::string & gguf, const std::string & sidecar, const CoremlPolicy & policy) {
#ifdef TTS_CPP_USE_COREML
    return !policy.disabled && !gguf.empty() && ::tts_cpp::detail::coreml_sidecar_exists(sidecar);
#else
    (void) gguf;
    (void) sidecar;
    (void) policy;
    return false;
#endif
}
} // namespace

bool coreml_step_enabled(const std::string & flow_lm_gguf, const CoremlPolicy & policy) {
    return sidecar_enabled(flow_lm_gguf, coreml_step_path(flow_lm_gguf), policy);
}

bool coreml_decoder_enabled(const std::string & mimi_gguf, const CoremlPolicy & policy) {
    return sidecar_enabled(mimi_gguf, coreml_decoder_path(mimi_gguf), policy);
}

void coreml_rope(int position, int head_dim, float base, std::vector<float> & cos, std::vector<float> & sin) {
    cos.resize(size_t(head_dim / 2));
    sin.resize(cos.size());
    for (int i = 0; i < head_dim / 2; ++i) {
        const double theta = double(position) * std::pow(double(base), -2.0 * i / head_dim);
        cos[size_t(i)] = float(std::cos(theta));
        sin[size_t(i)] = float(std::sin(theta));
    }
}

#ifdef TTS_CPP_USE_COREML
CoremlStep::CoremlStep(const std::string & path, const std::string & compute_units, const FlowConfig & config,
                       const std::string & source_hash) : config_(config) {
    model_ = moss_coreml_load(path.c_str(), compute_units.c_str());
    if (model_ == nullptr) {
        fail("cannot load " + path + " on " + compute_units +
             (compute_units == "cpu_and_ane" ? " (its KV cache is Core ML state, which the Neural Engine cannot hold; "
                                               "use cpu_and_gpu, all or cpu_only)" : ""));
    }
    try {
        if (!moss_coreml_has_state(model_)) fail(path + " declares no KV state; re-export it");
        require_stamp(model_, path, source_hash);
        const int64_t heads = config.heads, head_dim = config.dim / config.heads;
        const std::vector<int64_t> mask = feature_dims(model_, "mask", false);
        context_ = mask.size() == 4 ? int(mask[3]) : 0;
        if (context_ < 2) fail(path + " declares mask " + dims_text(mask) + "; expected [1, 1, 1, context]");
        const int64_t c = context_, latent = config.latent_dim;
        require_features(model_, path, {
            {"latent", false, {1, latent}}, {"noise", false, {1, latent}},
            {"cos", false, {1, head_dim / 2}}, {"sin", false, {1, head_dim / 2}},
            {"mask", false, {1, 1, 1, c}}, {"write_mask", false, {1, 1, c, 1}},
            {"hidden", false, {1, config.dim}}, {"next_latent", false, {1, latent}},
            {"keys", true, {config.layers, heads, c, head_dim}}, {"values", true, {config.layers, heads, c, head_dim}},
        });
    } catch (...) {
        moss_coreml_free(model_);
        model_ = nullptr;
        throw;
    }
    label_ = moss_coreml_label(model_);
    mask_.assign(size_t(context_), COREML_MASKED);
    write_mask_.assign(size_t(context_), 0.0f);
    // One prediction at position 0 (the caller then writes the voice rows over
    // it): a sidecar declaring inputs or outputs beyond the checked ones fails
    // here rather than mid-synthesis, and the first frame skips Core ML's
    // first-prediction setup.
    try {
        std::vector<float> zeros(size_t(config.latent_dim)), hidden, next;
        step(0, zeros, zeros, hidden, next);
    } catch (...) {
        moss_coreml_free(model_);
        model_ = nullptr;
        throw;
    }
}

CoremlStep::~CoremlStep() {
    if (model_ != nullptr) moss_coreml_free(model_);
}

void CoremlStep::write_rows(const CacheRows & rows) {
    const int heads = config_.heads, head_dim = config_.dim / config_.heads, n = rows.count;
    if (rows.start < 0 || n < 1 || rows.start + n > context_ || int(rows.keys.size()) != config_.layers ||
        rows.values.size() != rows.keys.size())
        fail("cache rows outside the sidecar's context");
    // ggml rows are [position][head][head_dim]; the state is [layer][head][position][head_dim].
    staging_.resize(size_t(n) * config_.dim);
    for (int layer = 0; layer < config_.layers; ++layer) {
        for (int kv = 0; kv < 2; ++kv) {
            const auto & source = kv ? rows.values[size_t(layer)] : rows.keys[size_t(layer)];
            if (source.size() != staging_.size()) fail("cache rows of the wrong width");
            for (int p = 0; p < n; ++p)
                for (int h = 0; h < heads; ++h)
                    std::copy_n(source.data() + (size_t(p) * heads + h) * head_dim, head_dim,
                                staging_.data() + (size_t(h) * n + p) * head_dim);
            const int64_t start[4] = {layer, 0, rows.start, 0};
            const int64_t count[4] = {1, heads, n, head_dim};
            if (moss_coreml_write_state(model_, kv ? "values" : "keys", staging_.data(), start, count, 4) != 0)
                fail("cannot write the KV state");
        }
    }
}

void CoremlStep::reset(const CacheRows & rows) {
    if (moss_coreml_reset_state(model_) != 0) fail("cannot replace the KV state");
    write_rows(rows);
}

void CoremlStep::step(int position, const std::vector<float> & latent, const std::vector<float> & noise,
                      std::vector<float> & hidden, std::vector<float> & next_latent) {
    if (position < 0 || position >= context_) fail("position outside the sidecar's context");
    if (latent.size() != size_t(config_.latent_dim) || noise.size() != latent.size())
        fail("invalid latent or noise dimensions");
    coreml_rope(position, config_.dim / config_.heads, config_.rope_base, cos_, sin_);
    // Attend to positions 0..position and write this frame's row at position.
    if (masked_from_ != position + 1) {
        std::fill(mask_.begin(), mask_.end(), COREML_MASKED);
        std::fill_n(mask_.begin(), position + 1, 0.0f);
        masked_from_ = position + 1;
    }
    if (written_ != position) {
        if (written_ >= 0) write_mask_[size_t(written_)] = 0.0f;
        write_mask_[size_t(position)] = 1.0f;
        written_ = position;
    }
    hidden.resize(size_t(config_.dim));
    next_latent.resize(size_t(config_.latent_dim));
    const char * in_names[] = {"latent", "noise", "cos", "sin", "mask", "write_mask"};
    const float * in_data[] = {latent.data(), noise.data(), cos_.data(), sin_.data(), mask_.data(), write_mask_.data()};
    const char * out_names[] = {"hidden", "next_latent"};
    float * out_data[] = {hidden.data(), next_latent.data()};
    if (moss_coreml_predict(model_, in_names, in_data, 6, out_names, out_data, 2) != 0) fail("prediction failed");
    require_finite(hidden, "hidden output");
    require_finite(next_latent, "latent output");
}
CoremlDecoder::CoremlDecoder(const std::string & path, const std::string & compute_units,
                             const std::string & source_hash) {
    model_ = moss_coreml_load(path.c_str(), compute_units.c_str());
    if (model_ == nullptr) fail("cannot load " + path + " on " + compute_units);
    try {
        if (moss_coreml_has_state(model_)) fail(path + " declares Core ML state; re-export it");
        require_stamp(model_, path, source_hash);
        const std::vector<int64_t> kv = {Geometry::layers, Geometry::heads, Geometry::keep, Geometry::head_dim};
        std::vector<expected_feature> expected = {
            {"latents", false, {Geometry::frames, Geometry::latent}},
            {"cos", false, {DECODER_ROWS, Geometry::head_dim / 2}}, {"sin", false, {DECODER_ROWS, Geometry::head_dim / 2}},
            {"mask", false, {DECODER_ROWS, Geometry::keep + DECODER_ROWS}},
            {"keys", false, kv}, {"values", false, kv}, {"new_keys", false, kv}, {"new_values", false, kv},
            {"pcm", false, {int64_t(Geometry::frames) * Geometry::frame_samples}},
        };
        for (const auto & h : HISTORIES) {
            expected.push_back({h.name, false, {h.channels, h.length}});
            expected.push_back({h.output, false, {h.channels, h.length}});
        }
        require_features(model_, path, expected);
        label_ = moss_coreml_label(model_);
        reset();
        decode(std::vector<float>(size_t(Geometry::frames) * Geometry::latent));
        reset();
    } catch (...) {
        moss_coreml_free(model_);
        model_ = nullptr;
        throw;
    }
}

CoremlDecoder::~CoremlDecoder() {
    if (model_ != nullptr) moss_coreml_free(model_);
}

void CoremlDecoder::reset() {
    position_ = 0;
    keys_.assign(KV_SIZE, 0.0f);
    values_.assign(KV_SIZE, 0.0f);
    histories_.resize(HISTORIES.size());
    for (size_t i = 0; i < HISTORIES.size(); ++i)
        histories_[i].assign(size_t(HISTORIES[i].channels) * HISTORIES[i].length, 0.0f);
}

// One prediction: RoPE tables for this call's transformer rows, and the mask
// letting row i see the WINDOW positions up to its own among the KEEP
// retained rows (unfilled before position KEEP) and this call's rows. The
// new state comes back into the same buffers: the runner copies every input
// before predicting and writes outputs after.
void CoremlDecoder::predict(const float * latent, float * pcm) {
    constexpr int keep = Geometry::keep, half = Geometry::head_dim / 2, columns = keep + DECODER_ROWS;
    cos_.resize(size_t(DECODER_ROWS) * half);
    sin_.resize(cos_.size());
    for (int row = 0; row < DECODER_ROWS; ++row) {
        coreml_rope(int(position_) + row, Geometry::head_dim, 10000.0f, row_cos_, row_sin_);
        std::copy(row_cos_.begin(), row_cos_.end(), cos_.begin() + size_t(row) * half);
        std::copy(row_sin_.begin(), row_sin_.end(), sin_.begin() + size_t(row) * half);
    }
    mask_.assign(size_t(DECODER_ROWS) * columns, COREML_MASKED);
    const int first = keep - int(std::min<int64_t>(position_, keep));
    for (int row = 0; row < DECODER_ROWS; ++row)
        std::fill(mask_.begin() + size_t(row) * columns + std::max(row, first),
                  mask_.begin() + size_t(row) * columns + keep + row + 1, 0.0f);
    std::vector<const char *> in_names = {"latents", "cos", "sin", "mask", "keys", "values"};
    std::vector<const float *> in_data = {latent, cos_.data(), sin_.data(), mask_.data(), keys_.data(), values_.data()};
    std::vector<const char *> out_names = {"pcm", "new_keys", "new_values"};
    std::vector<float *> out_data = {pcm, keys_.data(), values_.data()};
    for (size_t i = 0; i < HISTORIES.size(); ++i) {
        in_names.push_back(HISTORIES[i].name);
        in_data.push_back(histories_[i].data());
        out_names.push_back(HISTORIES[i].output);
        out_data.push_back(histories_[i].data());
    }
    if (moss_coreml_predict(model_, in_names.data(), in_data.data(), int(in_names.size()), out_names.data(),
                            out_data.data(), int(out_names.size())) != 0)
        fail("Mimi decoder prediction failed");
    position_ += DECODER_ROWS;
}

std::vector<float> CoremlDecoder::decode(const std::vector<float> & latents) {
    constexpr size_t frame = size_t(Geometry::frames) * Geometry::latent;
    if (latents.empty() || latents.size() % frame) fail("invalid Mimi decoder latents");
    std::vector<float> pcm(latents.size() / frame * Geometry::frames * Geometry::frame_samples);
    for (size_t at = 0, out = 0; at < latents.size(); at += frame, out += size_t(Geometry::frames) * Geometry::frame_samples)
        predict(latents.data() + at, pcm.data() + out);
    require_finite(pcm, "Mimi decoder output");
    return pcm;
}
#else
CoremlStep::CoremlStep(const std::string &, const std::string &, const FlowConfig & config, const std::string &)
    : config_(config) {
    fail("this build has no Core ML (TTS_CPP_COREML=OFF)");
}
CoremlStep::~CoremlStep() = default;
void CoremlStep::write_rows(const CacheRows &) { fail("this build has no Core ML"); }
void CoremlStep::reset(const CacheRows &) { fail("this build has no Core ML"); }
void CoremlStep::step(int, const std::vector<float> &, const std::vector<float> &, std::vector<float> &,
                      std::vector<float> &) {
    fail("this build has no Core ML");
}
CoremlDecoder::CoremlDecoder(const std::string &, const std::string &, const std::string &) {
    fail("this build has no Core ML (TTS_CPP_COREML=OFF)");
}
CoremlDecoder::~CoremlDecoder() = default;
void CoremlDecoder::reset() {}
void CoremlDecoder::predict(const float *, float *) { fail("this build has no Core ML"); }
std::vector<float> CoremlDecoder::decode(const std::vector<float> &) { fail("this build has no Core ML"); }
#endif

} // namespace tts_cpp::pocket::detail
