#include "moss/codec_coreml.h"

#include <algorithm>
#include <cmath>

namespace tts_cpp::moss::detail {
namespace {

constexpr float MASKED_SCORE = -1e4f;
constexpr int SINGLE_BATCH = 1;
constexpr int COMMIT_ELEMENTS = 1;
constexpr float COMMITTED = 1.0f;
constexpr float PREVIEW = 0.0f;
constexpr int LATENT_RANK = 3;

double inverse_frequency(int index, int head_dim, float max_period) {
    return std::pow((double) max_period, -2.0 * index / head_dim);
}

void fill_rope_row(int64_t position, int head_dim, float max_period, float * cos_row, float * sin_row) {
    const int half = head_dim / 2;
    for (int i = 0; i < half; ++i) {
        const double angle = (double) position * inverse_frequency(i, head_dim, max_period);
        cos_row[i] = cos_row[i + half] = (float) std::cos(angle);
        sin_row[i] = sin_row[i + half] = (float) std::sin(angle);
    }
}

bool key_visible(int64_t query, int64_t key, int context) {
    return key >= 0 && key <= query && query - key < context;
}

void fill_mask_row(int64_t query, int64_t first, int count, int capacity, int context, float * row) {
    for (int column = 0; column < capacity + count; ++column) {
        const int64_t key = column < capacity ? first - capacity + column : first + (column - capacity);
        row[column] = key_visible(query, key, context) ? 0.0f : MASKED_SCORE;
    }
}

void append_stage_tables(const CodecStage & stage, int chunk_frames, int64_t committed,
                         std::vector<std::vector<float>> & tables) {
    const int64_t first = committed * stage.rate;
    const int count = chunk_frames * stage.rate;
    std::vector<float> cos;
    std::vector<float> sin;
    codec_rope_tables(first, count, stage.head_dim, stage.max_period, cos, sin);
    tables.push_back(std::move(cos));
    tables.push_back(std::move(sin));
    tables.push_back(codec_band_mask(first, count, stage.context - 1, stage.context));
}

std::vector<CoremlTensor> codec_inputs(const CodecGeometry & geometry, int chunk) {
    std::vector<CoremlTensor> inputs = {{"latents", {SINGLE_BATCH, chunk, geometry.code_dim}},
                                        {"commit", {COMMIT_ELEMENTS}}};
    for (size_t s = 0; s < geometry.stages.size(); ++s) {
        const CodecStage & stage = geometry.stages[s];
        const int64_t rows = (int64_t) chunk * stage.rate;
        const std::string index = std::to_string(s);
        inputs.push_back({"cos_" + index, {rows, stage.head_dim}});
        inputs.push_back({"sin_" + index, {rows, stage.head_dim}});
        inputs.push_back({"mask_" + index, {rows, stage.context - 1 + rows}});
    }
    return inputs;
}

int declared_chunk(const CoremlModel & model, const CodecGeometry & geometry) {
    const std::vector<int64_t> dims = model.dims("latents");
    if (dims.size() != LATENT_RANK || dims[0] != SINGLE_BATCH || dims[1] < 1 || dims[2] != geometry.code_dim) {
        return 0;
    }
    return (int) dims[1];
}

class CoremlCodecSidecar final : public CodecSidecarModel {
public:
    CoremlCodecSidecar(std::unique_ptr<CoremlModel> model, std::vector<CoremlTensor> inputs, CoremlTensor output,
                       int chunk)
        : model_(std::move(model)), inputs_(std::move(inputs)), output_(std::move(output)), chunk_(chunk) {}

    int chunk_frames() const override {
        return chunk_;
    }

    bool reset() override {
        return model_->reset_state();
    }

    bool decode_chunk(const std::vector<float> & latents, float commit, const std::vector<std::vector<float>> & tables,
                      std::vector<float> & pcm) override {
        std::vector<const float *> data = {latents.data(), &commit};
        for (const std::vector<float> & table : tables) {
            data.push_back(table.data());
        }
        if (!sizes_match(latents, tables)) {
            return false;
        }
        pcm.assign((size_t) tensor_elements(output_), 0.0f);
        return model_->predict(inputs_, data, {output_}, {pcm.data()});
    }

    const char * label() const override {
        return model_->label();
    }

private:
    bool sizes_match(const std::vector<float> & latents, const std::vector<std::vector<float>> & tables) const {
        if (tables.size() + 2 != inputs_.size() || (int64_t) latents.size() != tensor_elements(inputs_[0])) {
            return false;
        }
        for (size_t i = 0; i < tables.size(); ++i) {
            if ((int64_t) tables[i].size() != tensor_elements(inputs_[i + 2])) {
                return false;
            }
        }
        return true;
    }

    std::unique_ptr<CoremlModel> model_;
    std::vector<CoremlTensor> inputs_;
    CoremlTensor output_;
    int chunk_;
};

} // namespace

void codec_rope_tables(int64_t first, int count, int head_dim, float max_period, std::vector<float> & cos,
                       std::vector<float> & sin) {
    cos.assign((size_t) count * head_dim, 0.0f);
    sin.assign((size_t) count * head_dim, 0.0f);
    for (int row = 0; row < count; ++row) {
        fill_rope_row(first + row, head_dim, max_period, cos.data() + (size_t) row * head_dim,
                sin.data() + (size_t) row * head_dim);
    }
}

std::vector<float> codec_band_mask(int64_t first, int count, int capacity, int context) {
    const size_t width = (size_t) capacity + count;
    std::vector<float> mask((size_t) count * width, MASKED_SCORE);
    for (int row = 0; row < count; ++row) {
        fill_mask_row(first + row, first, count, capacity, context, mask.data() + (size_t) row * width);
    }
    return mask;
}

std::vector<std::vector<float>> codec_chunk_tables(const CodecGeometry & geometry, int chunk_frames,
                                                   int64_t committed_frames) {
    std::vector<std::vector<float>> tables;
    for (const CodecStage & stage : geometry.stages) {
        append_stage_tables(stage, chunk_frames, committed_frames, tables);
    }
    return tables;
}

std::unique_ptr<CodecSidecarModel> open_codec_sidecar(const std::string & decoder_path,
                                                      const CodecGeometry & geometry, const CoremlPolicy & policy) {
    std::unique_ptr<CoremlModel> model = CoremlModel::open(codec_decoder_sidecar_path(decoder_path), policy);
    if (!model || !model->stateful()) {
        return nullptr;
    }
    const int chunk = declared_chunk(*model, geometry);
    std::vector<CoremlTensor> inputs = codec_inputs(geometry, chunk);
    CoremlTensor output{"pcm", {SINGLE_BATCH, (int64_t) chunk * geometry.hop}};
    if (chunk < 1 || !model->declares(inputs) || !model->declares(output)) {
        return nullptr;
    }
    return std::make_unique<CoremlCodecSidecar>(std::move(model), std::move(inputs), std::move(output), chunk);
}

CodecSidecarStream::CodecSidecarStream(CodecSidecarModel & model, CodecGeometry geometry)
    : model_(model), geometry_(std::move(geometry)) {}

bool CodecSidecarStream::begin() {
    pending_.clear();
    committed_ = 0;
    emitted_ = 0;
    return model_.reset();
}

bool CodecSidecarStream::decode(const std::vector<float> & latents, std::vector<float> & pcm) {
    pcm.clear();
    pending_.insert(pending_.end(), latents.begin(), latents.end());
    return commit_full_chunks(pcm) && preview_remainder(pcm);
}

int64_t CodecSidecarStream::pending_frames() const {
    return (int64_t) (pending_.size() / (size_t) geometry_.code_dim);
}

bool CodecSidecarStream::commit_full_chunks(std::vector<float> & pcm) {
    const int chunk = model_.chunk_frames();
    const size_t chunk_values = (size_t) chunk * geometry_.code_dim;
    while (pending_frames() >= chunk) {
        const std::vector<float> latents(pending_.begin(), pending_.begin() + (std::ptrdiff_t) chunk_values);
        std::vector<float> decoded;
        if (!run_chunk(latents, COMMITTED, decoded)) {
            return false;
        }
        append_new_frames(decoded, committed_, chunk, pcm);
        committed_ += chunk;
        pending_.erase(pending_.begin(), pending_.begin() + (std::ptrdiff_t) chunk_values);
    }
    return true;
}

bool CodecSidecarStream::preview_remainder(std::vector<float> & pcm) {
    const int64_t remaining = pending_frames();
    if (remaining == 0 || emitted_ >= committed_ + remaining) {
        return true;
    }
    std::vector<float> latents(pending_);
    latents.resize((size_t) model_.chunk_frames() * geometry_.code_dim, 0.0f);
    std::vector<float> decoded;
    if (!run_chunk(latents, PREVIEW, decoded)) {
        return false;
    }
    append_new_frames(decoded, committed_, remaining, pcm);
    return true;
}

bool CodecSidecarStream::run_chunk(const std::vector<float> & latents, float commit, std::vector<float> & decoded) {
    return model_.decode_chunk(latents, commit, codec_chunk_tables(geometry_, model_.chunk_frames(), committed_),
            decoded);
}

void CodecSidecarStream::append_new_frames(const std::vector<float> & decoded, int64_t first, int64_t count,
                                           std::vector<float> & pcm) {
    const int64_t from = std::max(emitted_, first);
    const int64_t to = first + count;
    if (from >= to) {
        return;
    }
    const auto begin = decoded.begin() + (std::ptrdiff_t) ((from - first) * geometry_.hop);
    pcm.insert(pcm.end(), begin, begin + (std::ptrdiff_t) ((to - from) * geometry_.hop));
    emitted_ = to;
}

} // namespace tts_cpp::moss::detail
