#include "parakeet_ctc.h"
#include "parakeet_sortformer.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

int encoder_parity_main(int, char **);
int tdt_encoder_parity_main(int, char **);
int sortformer_parity_main(int, char **);

namespace {

constexpr int kFrames = 2;
constexpr int kWidth = 2;
constexpr int kAutomaticValidMelFrames = 0;
constexpr float kValue = 1.0f;
constexpr float kLogProbability = -0.69314718056f;
constexpr int kSampleRate = 16000;
const std::vector<const char *> kStages = {
    "mel", "subsampling_out", "block_0_post_ff1", "block_0_post_attn", "block_0_post_conv",
    "block_0_post_ff2", "block_0_out", "block_last_out", "encoder_out", "logits", "speaker_probs",
};

void write_reference(const std::filesystem::path & path, const std::string & shape = "(2, 2)",
                     size_t count = kFrames * kWidth, float value = kValue) {
    const std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape + ", }\n";
    const uint16_t length = static_cast<uint16_t>(header.size());
    std::ofstream file(path, std::ios::binary);
    file.write("\x93NUMPY\x01\x00", 8);
    file.write(reinterpret_cast<const char *>(&length), sizeof(length));
    file.write(header.data(), header.size());
    const std::vector<float> values(count, value);
    file.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float));
}

void write_references(const std::filesystem::path & directory) {
    for (const auto * stage : kStages) {
        const bool logits = std::string(stage) == "logits";
        write_reference(directory / (std::string(stage) + ".npy"), "(2, 2)",
                        kFrames * kWidth, logits ? kLogProbability : kValue);
    }
}

int invoke(const std::string & family, const std::filesystem::path & directory) {
    std::vector<std::string> arguments = {"parity", family};
    if (family != "ctc") arguments.push_back("audio.wav");
    arguments.push_back(directory.string());
    std::vector<char *> pointers;
    for (auto & argument : arguments) pointers.push_back(argument.data());
    if (family == "ctc") return encoder_parity_main(static_cast<int>(pointers.size()), pointers.data());
    if (family == "tdt") return tdt_encoder_parity_main(static_cast<int>(pointers.size()), pointers.data());
    return sortformer_parity_main(static_cast<int>(pointers.size()), pointers.data());
}

bool rejects_reference(const std::string & family, const std::filesystem::path & directory,
                       const std::string & stage) {
    const auto path = directory / (stage + ".npy");
    write_references(directory);
    std::filesystem::remove(path);
    if (invoke(family, directory) == 0) return false;
    write_reference(path, "(2, 2)", 1);
    if (invoke(family, directory) == 0) return false;
    write_reference(path, "(1, 4)");
    return invoke(family, directory) != 0;
}

bool check_family(const std::string & family, const std::filesystem::path & directory) {
    write_references(directory);
    if (invoke(family, directory) != 0) return false;
    for (const auto * stage : {"mel", "encoder_out"}) {
        if (!rejects_reference(family, directory, stage)) return false;
    }
    if (family == "ctc") {
        for (const auto * stage : kStages) {
            if (std::string(stage) != "speaker_probs" && !rejects_reference(family, directory, stage)) return false;
        }
    }
    if (family == "sortformer" && !rejects_reference(family, directory, "speaker_probs")) return false;
    return true;
}

}  // namespace

// Synthetic model outputs let the real parity entry points run without model weights or a backend.
namespace parakeet {

int load_from_gguf(const std::string & family, ParakeetCtcModel & model, int, int, bool) {
    model.model_type = family == "ctc" ? ParakeetModelType::CTC :
                       family == "tdt" ? ParakeetModelType::TDT : ParakeetModelType::SORTFORMER;
    model.mel_cfg.n_mels = kWidth;
    return 0;
}

int load_wav_mono_f32(const std::string &, std::vector<float> & samples, int & sample_rate) {
    samples = {kValue};
    sample_rate = kSampleRate;
    return 0;
}

int compute_log_mel(const float *, int, const MelConfig &, std::vector<float> & mel, int & frames) {
    // An extra preprocessing frame must not change the encoder parity input.
    frames = kFrames + 1;
    mel.assign(frames * kWidth, kValue);
    return 0;
}

int run_encoder(ParakeetCtcModel &, const float * mel, int frames, int width,
                EncoderOutputs & output, const EncoderRoute &, int, bool, bool,
                const EncoderOutputConsumer &, int valid_mel_frames_override,
                bool prefer_precise_eu) {
    if (valid_mel_frames_override != kAutomaticValidMelFrames || prefer_precise_eu) return 1;
    if (frames != kFrames || width != kWidth || mel[0] != kValue) return 1;
    output.n_enc_frames = kFrames;
    output.d_model = output.vocab_size = kWidth;
    const std::vector<float> values(kFrames * kWidth, kValue);
    output.subsampling_out = output.block_0_post_ff1 = output.block_0_post_attn = values;
    output.block_0_post_conv = output.block_0_post_ff2 = output.block_0_out = values;
    output.block_last_out = output.encoder_out = output.logits = values;
    return 0;
}

int sortformer_diarize_ggml(const ParakeetCtcModel &, const float *, int, int,
                            const SortformerDiarizationOptions &, SortformerDiarizationResult & output) {
    output.n_frames = kFrames;
    output.num_spks = kWidth;
    output.speaker_probs.assign(kFrames * kWidth, kValue);
    return 0;
}

}  // namespace parakeet

int main(int argc, char ** argv) {
    if (argc != 2) return 1;
    const std::filesystem::path directory(argv[1]);
    std::filesystem::create_directories(directory);
    bool passed = true;
    for (const auto * family : {"ctc", "tdt", "sortformer"}) {
        if (!check_family(family, directory)) {
            std::fprintf(stderr, "Parity harness regression: %s\n", family);
            passed = false;
        }
    }
    for (const auto * stage : kStages) {
        std::filesystem::remove(directory / (std::string(stage) + ".npy"));
    }
    std::filesystem::remove(directory);
    return passed ? 0 : 1;
}
