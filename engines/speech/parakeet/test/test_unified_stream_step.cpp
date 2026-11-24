#include "parakeet_ctc.h"
#include "parakeet_tdt.h"
#include "parakeet_unified.h"
#include "mel_preprocess.h"
#include "sentencepiece_bpe.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr int kChunkFrames = 7;
constexpr int kRightContextFrames = 7;
constexpr double kMinCosineFirstStep = 0.999;
constexpr double kMinCosineCachedStep = 0.99;
constexpr float kMaxAbsDiff = 0.15f;

double min_cosine_for_step(int step) {
    return step == 0 ? kMinCosineFirstStep : kMinCosineCachedStep;
}

int load_npy(const std::string & path, std::vector<float> & data, std::vector<int64_t> & shape) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return 1;
    char magic[6];
    file.read(magic, sizeof(magic));
    if (std::memcmp(magic, "\x93NUMPY", sizeof(magic)) != 0) return 2;
    uint8_t major = 0;
    uint8_t minor = 0;
    file.read(reinterpret_cast<char *>(&major), 1);
    file.read(reinterpret_cast<char *>(&minor), 1);
    uint32_t header_length = 0;
    if (major == 1) {
        uint16_t length = 0;
        file.read(reinterpret_cast<char *>(&length), sizeof(length));
        header_length = length;
    } else {
        file.read(reinterpret_cast<char *>(&header_length), sizeof(header_length));
    }
    std::string header(header_length, '\0');
    file.read(header.data(), header_length);
    if (header.find("'fortran_order': True") != std::string::npos) return 5;
    const size_t shape_key = header.find("'shape':");
    const size_t left = header.find('(', shape_key);
    const size_t right = header.find(')', left);
    if (shape_key == std::string::npos || left == std::string::npos || right == std::string::npos) return 3;
    shape.clear();
    size_t position = left + 1;
    while (position < right) {
        while (position < right && (header[position] == ' ' || header[position] == ',')) ++position;
        size_t end = position;
        while (end < right && header[end] >= '0' && header[end] <= '9') ++end;
        if (end == position) break;
        shape.push_back(std::stoll(header.substr(position, end - position)));
        position = end;
    }
    size_t count = 1;
    for (const int64_t dimension : shape) count *= static_cast<size_t>(dimension);
    data.resize(count);
    file.read(reinterpret_cast<char *>(data.data()), count * sizeof(float));
    return file ? 0 : 4;
}

std::string load_text(const std::string & path) {
    std::ifstream file(path);
    if (!file) return {};
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) text.pop_back();
    return text;
}

std::vector<float> transpose_to_frames_major(const std::vector<float> & channel_major, int width, int frames) {
    std::vector<float> out(static_cast<size_t>(frames) * width);
    for (int frame = 0; frame < frames; ++frame) {
        for (int channel = 0; channel < width; ++channel) {
            out[static_cast<size_t>(frame) * width + channel] =
                channel_major[static_cast<size_t>(channel) * frames + frame];
        }
    }
    return out;
}

double cosine(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || a.empty()) return 0.0;
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
    }
    return dot / std::sqrt(na * nb);
}

float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    float worst = 0.0f;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        worst = std::max(worst, std::fabs(a[i] - b[i]));
    }
    return worst;
}

std::string step_dir(const std::string & reference_dir, int step) {
    char name[32];
    std::snprintf(name, sizeof(name), "/buffered/step-%03d", step);
    return reference_dir + name;
}

int count_reference_steps(const std::string & reference_dir) {
    int steps = 0;
    while (std::ifstream(step_dir(reference_dir, steps) + "/encoder_chunk.npy")) ++steps;
    return steps;
}

struct Session {
    parakeet::ParakeetCtcModel model;
    parakeet::TdtRuntimeWeights runtime;
    parakeet::UnifiedStreamState state;
};

constexpr int kSkipReturnCode = 3;

int open_session(const std::string & gguf, int n_gpu_layers, Session & session) {
    if (int rc = parakeet::load_from_gguf(gguf, session.model, 0, n_gpu_layers, false); rc != 0) {
        std::fprintf(stderr, "load_from_gguf failed: %d\n", rc);
        return 1;
    }
    if (n_gpu_layers > 0 && !parakeet::model_has_gpu_backend(session.model)) {
        std::fprintf(stderr, "no GPU backend available; skipping\n");
        return kSkipReturnCode;
    }
    if (parakeet::tdt_prepare_runtime(session.model, session.runtime) != 0) {
        std::fprintf(stderr, "tdt_prepare_runtime failed\n");
        return 2;
    }
    if (int rc = parakeet::init_unified_stream_state(session.model, kChunkFrames, kRightContextFrames, session.state);
        rc != 0) {
        std::fprintf(stderr, "init_unified_stream_state failed: %d\n", rc);
        return 3;
    }
    return 0;
}

int feed_wav(Session & session, const std::string & wav) {
    std::vector<float> samples;
    int sample_rate = 0;
    if (parakeet::load_wav_mono_f32(wav, samples, sample_rate) != 0 || sample_rate != session.model.mel_cfg.sample_rate) {
        std::fprintf(stderr, "cannot load %s\n", wav.c_str());
        return 4;
    }
    return parakeet::append_unified_pcm(session.model, session.state, samples.data(),
                                        static_cast<int>(samples.size()), true);
}

int run_next_step(Session & session, bool finalize, parakeet::UnifiedStreamStepResult & result) {
    if (session.state.finalized) return 0;
    std::vector<float> processed;
    int frames = 0;
    const int n_mels = session.model.mel_cfg.n_mels;
    const int ready = parakeet::next_unified_processed_signal(session.state, n_mels, finalize, processed, frames);
    if (ready <= 0) return ready < 0 ? ready : 0;
    const bool last = finalize && parakeet::unified_pending_mel_frames(session.state) == 0;
    if (int rc = parakeet::run_unified_stream_step(session.model, session.runtime, processed.data(), frames,
                                                   n_mels, last, session.state, result); rc != 0) {
        std::fprintf(stderr, "run_unified_stream_step failed: %d\n", rc);
        return rc;
    }
    return 1;
}

int compare_step(const Session & session, const parakeet::UnifiedStreamStepResult & result,
                 const std::string & reference_dir, int step) {
    std::vector<float> expected;
    std::vector<int64_t> shape;
    if (load_npy(step_dir(reference_dir, step) + "/encoder_chunk.npy", expected, shape) != 0 || shape.size() != 2) {
        std::fprintf(stderr, "cannot load reference step %d\n", step);
        return 10;
    }
    const int width = static_cast<int>(shape[0]);
    const int frames = static_cast<int>(shape[1]);
    if (width != session.model.encoder_cfg.d_model || frames != result.committed_frames) {
        std::fprintf(stderr, "step %d shape mismatch: reference %dx%d, ours %d committed frames\n",
                     step, width, frames, result.committed_frames);
        return 11;
    }
    const std::vector<float> reference = transpose_to_frames_major(expected, width, frames);
    const double similarity = cosine(result.encoder_committed, reference);
    const float worst = max_abs_diff(result.encoder_committed, reference);
    std::fprintf(stderr, "Unified step %d vs NeMo buffered: cosine=%.6f max|diff|=%.4f provisional=%d cache_length=%d\n",
                 step, similarity, worst, result.provisional_frames, session.state.cache_length);
    if (similarity < min_cosine_for_step(step) || worst > kMaxAbsDiff ||
        result.provisional_frames != kRightContextFrames) {
        return 12;
    }
    return 0;
}

int check_reference_steps(Session & session, const std::string & reference_dir, int reference_steps) {
    for (int step = 0; step < reference_steps; ++step) {
        parakeet::UnifiedStreamStepResult result;
        if (run_next_step(session, false, result) != 1) {
            std::fprintf(stderr, "no step %d produced\n", step);
            return 20;
        }
        if (int rc = compare_step(session, result, reference_dir, step); rc != 0) return rc;
    }
    return 0;
}

int drain(Session & session) {
    parakeet::UnifiedStreamStepResult result;
    int ready = 1;
    while (ready == 1) ready = run_next_step(session, false, result);
    if (ready < 0) return ready;
    ready = 1;
    while (ready == 1) ready = run_next_step(session, true, result);
    return ready < 0 ? ready : 0;
}

int check_final_transcript(const Session & session, const std::string & reference_dir) {
    const std::string expected = load_text(reference_dir + "/buffered/transcript.txt");
    const std::string actual = parakeet::detokenize(session.model.vocab, session.state.token_ids);
    std::fprintf(stderr, "final transcript: %s\n", actual.c_str());
    if (expected.empty() || actual != expected) {
        std::fprintf(stderr, "transcript mismatch\n  ref: %s\n  got: %s\n", expected.c_str(), actual.c_str());
        return 30;
    }
    return 0;
}

int run(const std::string & gguf, const std::string & reference_dir, const std::string & wav, int n_gpu_layers) {
    const int reference_steps = count_reference_steps(reference_dir);
    if (reference_steps == 0) {
        std::fprintf(stderr, "no buffered reference steps under %s\n", reference_dir.c_str());
        return 40;
    }
    Session session;
    if (int rc = open_session(gguf, n_gpu_layers, session); rc != 0) return rc;
    if (int rc = feed_wav(session, wav); rc != 0) return rc;
    if (int rc = check_reference_steps(session, reference_dir, reference_steps); rc != 0) return rc;
    if (int rc = drain(session); rc != 0) return rc;
    return check_final_transcript(session, reference_dir);
}

}

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <parakeet-unified.gguf> <unified-ref-dir> <input.wav> [--n-gpu-layers N]\n", argv[0]);
        return 2;
    }
    int n_gpu_layers = 0;
    for (int i = 4; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--n-gpu-layers") == 0) n_gpu_layers = std::atoi(argv[i + 1]);
    }
    const int rc = run(argv[1], argv[2], argv[3], n_gpu_layers);
    if (rc == 0) {
        std::fprintf(stderr, "Unified stream step tests passed\n");
    } else {
        std::fprintf(stderr, "Unified stream step tests failed (%d)\n", rc);
    }
    return rc;
}
