#pragma once

#include "moss/coreml_sidecar.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tts_cpp::moss::detail {

struct CodecStage {
    int rate = 1;
    int context = 0;
    int head_dim = 0;
    float max_period = 0.0f;
};

struct CodecGeometry {
    std::vector<CodecStage> stages;
    int code_dim = 0;
    int hop = 0;
};

void codec_rope_tables(int64_t first, int count, int head_dim, float max_period, std::vector<float> & cos,
                       std::vector<float> & sin);
std::vector<float> codec_band_mask(int64_t first, int count, int capacity, int context);
std::vector<std::vector<float>> codec_chunk_tables(const CodecGeometry & geometry, int chunk_frames,
                                                   int64_t committed_frames);

class CodecSidecarModel {
public:
    virtual ~CodecSidecarModel() = default;
    virtual int chunk_frames() const = 0;
    virtual bool reset() = 0;
    virtual bool decode_chunk(const std::vector<float> & latents, float commit,
                              const std::vector<std::vector<float>> & tables, std::vector<float> & pcm) = 0;
    virtual const char * label() const = 0;
};

std::unique_ptr<CodecSidecarModel> open_codec_sidecar(const std::string & decoder_path,
                                                      const CodecGeometry & geometry, const CoremlPolicy & policy);

class CodecSidecarStream {
public:
    CodecSidecarStream(CodecSidecarModel & model, CodecGeometry geometry);

    bool begin();
    bool decode(const std::vector<float> & latents, std::vector<float> & pcm);

private:
    bool commit_full_chunks(std::vector<float> & pcm);
    bool preview_remainder(std::vector<float> & pcm);
    bool run_chunk(const std::vector<float> & latents, float commit, std::vector<float> & decoded);
    void append_new_frames(const std::vector<float> & decoded, int64_t first, int64_t count, std::vector<float> & pcm);
    int64_t pending_frames() const;

    CodecSidecarModel & model_;
    CodecGeometry geometry_;
    std::vector<float> pending_;
    int64_t committed_ = 0;
    int64_t emitted_ = 0;
};

} // namespace tts_cpp::moss::detail
