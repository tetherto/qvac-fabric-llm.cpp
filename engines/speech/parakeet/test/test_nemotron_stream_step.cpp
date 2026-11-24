#include "parakeet_ctc.h"
#include "parakeet_tdt.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

int load_npy(
    const std::string & path,
    std::vector<float> & data,
    std::vector<int64_t> & shape) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return 1;
    }

    char magic[6];
    file.read(magic, sizeof(magic));
    if (std::memcmp(magic, "\x93NUMPY", sizeof(magic)) != 0) {
        return 2;
    }
    uint8_t major = 0;
    uint8_t minor = 0;
    file.read(reinterpret_cast<char *>(&major), 1);
    file.read(reinterpret_cast<char *>(&minor), 1);
    (void) minor;

    uint32_t header_length = 0;
    if (major == 1) {
        uint16_t length = 0;
        file.read(reinterpret_cast<char *>(&length), sizeof(length));
        header_length = length;
    } else {
        file.read(
            reinterpret_cast<char *>(&header_length),
            sizeof(header_length));
    }
    std::string header(header_length, '\0');
    file.read(header.data(), header_length);

    const size_t shape_key = header.find("'shape':");
    const size_t left = header.find('(', shape_key);
    const size_t right = header.find(')', left);
    if (shape_key == std::string::npos ||
        left == std::string::npos ||
        right == std::string::npos) {
        return 3;
    }

    shape.clear();
    const std::string dimensions =
        header.substr(left + 1, right - left - 1);
    size_t position = 0;
    while (position < dimensions.size()) {
        while (position < dimensions.size() &&
               (dimensions[position] == ' ' ||
                dimensions[position] == ',')) {
            ++position;
        }
        size_t end = position;
        while (end < dimensions.size() &&
               dimensions[end] >= '0' &&
               dimensions[end] <= '9') {
            ++end;
        }
        if (end == position) {
            break;
        }
        shape.push_back(
            std::stoll(dimensions.substr(position, end - position)));
        position = end;
    }

    size_t count = 1;
    for (const int64_t dimension : shape) {
        count *= static_cast<size_t>(dimension);
    }
    std::vector<float> stored(count);
    file.read(
        reinterpret_cast<char *>(stored.data()),
        count * sizeof(float));
    if (!file) {
        return 4;
    }
    const bool fortran_order =
        header.find("'fortran_order': True") != std::string::npos;
    if (!fortran_order || shape.size() < 2) {
        data = std::move(stored);
        return 0;
    }

    data.resize(count);
    for (size_t c_index = 0; c_index < count; ++c_index) {
        size_t remainder = c_index;
        std::vector<size_t> coordinates(shape.size(), 0);
        for (size_t reverse = shape.size(); reverse > 0; --reverse) {
            const size_t dimension_index = reverse - 1;
            const size_t dimension =
                static_cast<size_t>(shape[dimension_index]);
            coordinates[dimension_index] = remainder % dimension;
            remainder /= dimension;
        }
        size_t f_index = 0;
        size_t f_stride = 1;
        for (size_t dimension_index = 0;
             dimension_index < shape.size();
             ++dimension_index) {
            f_index += coordinates[dimension_index] * f_stride;
            f_stride *= static_cast<size_t>(shape[dimension_index]);
        }
        data[c_index] = stored[f_index];
    }
    return 0;
}

double cosine(
    const std::vector<float> & actual,
    const std::vector<float> & expected) {
    if (actual.size() != expected.size() || actual.empty()) {
        return 0.0;
    }
    double dot = 0.0;
    double actual_norm = 0.0;
    double expected_norm = 0.0;
    for (size_t index = 0; index < actual.size(); ++index) {
        dot += static_cast<double>(actual[index]) * expected[index];
        actual_norm +=
            static_cast<double>(actual[index]) * actual[index];
        expected_norm +=
            static_cast<double>(expected[index]) * expected[index];
    }
    if (actual_norm == 0.0 && expected_norm == 0.0) {
        return 1.0;
    }
    return dot / std::sqrt(actual_norm * expected_norm);
}

std::string load_text(const std::string & path) {
    std::ifstream file(path);
    if (!file) {
        return {};
    }
    std::string text(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());
    while (!text.empty() &&
           (text.back() == '\n' || text.back() == '\r')) {
        text.pop_back();
    }
    return text;
}

int load_processed_signal(
    const std::string & path,
    std::vector<float> & signal,
    int & frames,
    int & features) {
    std::vector<float> source;
    std::vector<int64_t> shape;
    if (int rc = load_npy(path, source, shape); rc != 0) {
        return rc;
    }
    if (shape.size() != 3 || shape[0] != 1) {
        return 5;
    }
    features = static_cast<int>(shape[1]);
    frames = static_cast<int>(shape[2]);
    signal.resize(static_cast<size_t>(frames) * features);
    for (int frame = 0; frame < frames; ++frame) {
        for (int feature = 0; feature < features; ++feature) {
            signal[static_cast<size_t>(frame) * features + feature] =
                source[static_cast<size_t>(feature) * frames + frame];
        }
    }
    return 0;
}

int check_step(
    parakeet::ParakeetCtcModel & model,
    parakeet::TdtRuntimeWeights & runtime,
    parakeet::NemotronStreamState & state,
    const std::string & step_dir,
    int expected_cache_length) {
    std::vector<float> processed_signal;
    int mel_frames = 0;
    int features = 0;
    if (int rc = load_processed_signal(
            step_dir + "/processed_signal.npy",
            processed_signal,
            mel_frames,
            features); rc != 0) {
        return rc;
    }

    parakeet::NemotronStreamStepResult result;
    if (int rc = parakeet::run_nemotron_stream_step(
            model,
            runtime,
            processed_signal.data(),
            mel_frames,
            features,
            false,
            state,
            result); rc != 0) {
        std::fprintf(stderr, "Nemotron stream step failed: %d\n", rc);
        return rc;
    }

    std::vector<int64_t> shape;
    std::vector<float> expected_encoder;
    if (load_npy(
            step_dir + "/encoder_raw.npy",
            expected_encoder,
            shape) != 0) {
        return 10;
    }
    const double encoder_cosine =
        cosine(result.encoder_raw, expected_encoder);

    std::vector<float> expected_channel_cache;
    if (load_npy(
            step_dir + "/cache_channel_after.npy",
            expected_channel_cache,
            shape) != 0) {
        return 11;
    }
    const double channel_cosine =
        cosine(state.cache_channel, expected_channel_cache);

    std::vector<float> expected_time_cache;
    if (load_npy(
            step_dir + "/cache_time_after.npy",
            expected_time_cache,
            shape) != 0) {
        return 12;
    }
    const double time_cosine =
        cosine(state.cache_time, expected_time_cache);

    std::fprintf(
        stderr,
        "Nemotron step %d cosine: encoder=%.8f channel=%.8f time=%.8f\n",
        state.step_index - 1,
        encoder_cosine,
        channel_cosine,
        time_cosine);
    if (time_cosine < 0.999) {
        std::fprintf(stderr, "time cache sample actual/reference:");
        for (size_t index = 0;
             index < 16 && index < state.cache_time.size();
             ++index) {
            std::fprintf(
                stderr,
                " %.5f/%.5f",
                state.cache_time[index],
                expected_time_cache[index]);
        }
        std::fprintf(stderr, "\n");
    }
    if (encoder_cosine < 0.999 ||
        channel_cosine < 0.999 ||
        time_cosine < 0.999 ||
        state.cache_length != expected_cache_length) {
        return 13;
    }
    return 0;
}

int check_lifecycle(
    parakeet::ParakeetCtcModel & model,
    parakeet::TdtRuntimeWeights & runtime,
    const std::string & first_step_dir,
    int right_context) {
    parakeet::NemotronStreamState state;
    if (parakeet::init_nemotron_stream_state(
            model, "en-US", right_context, state) != 0) {
        return 20;
    }

    std::vector<float> signal;
    int frames = 0;
    int features = 0;
    if (load_processed_signal(
            first_step_dir + "/processed_signal.npy",
            signal,
            frames,
            features) != 0) {
        return 21;
    }

    parakeet::cancel_nemotron_stream(state);
    parakeet::NemotronStreamStepResult result;
    if (parakeet::run_nemotron_stream_step(
            model,
            runtime,
            signal.data(),
            frames,
            features,
            false,
            state,
            result) != -1) {
        return 22;
    }

    if (parakeet::init_nemotron_stream_state(
            model, "en-US", right_context, state) != 0) {
        return 23;
    }
    if (parakeet::run_nemotron_stream_step(
            model,
            runtime,
            signal.data(),
            frames,
            features,
            true,
            state,
            result) != 0 ||
        !state.finalized) {
        return 24;
    }
    if (parakeet::run_nemotron_stream_step(
            model,
            runtime,
            nullptr,
            0,
            features,
            true,
            state,
            result) != 0) {
        return 25;
    }

    parakeet::reset_nemotron_stream(state);
    if (!state.cache_channel.empty() ||
        !state.cache_time.empty() ||
        state.cache_length != 0 ||
        state.finalized ||
        state.cancelled) {
        return 26;
    }
    return 0;
}

int check_finalize_drops_short_remainder(
    parakeet::ParakeetCtcModel & model,
    int right_context) {
    parakeet::NemotronStreamState state;
    if (parakeet::init_nemotron_stream_state(
            model, "en-US", right_context, state) != 0) {
        return 27;
    }

    const int n_mels = model.mel_cfg.n_mels;
    const int factor = model.encoder_cfg.subsampling_factor;
    const int first_frames = 1 + factor * right_context;
    std::vector<float> first(
        static_cast<size_t>(first_frames) * n_mels, 0.0f);
    if (parakeet::append_nemotron_mel_frames(
            state, first.data(), first_frames, n_mels) != 0) {
        return 28;
    }

    std::vector<float> processed;
    int frames = 0;
    if (parakeet::next_nemotron_processed_signal(
            state, n_mels, false, processed, frames) != 1) {
        return 29;
    }

    const int leftover = factor - 1;
    std::vector<float> remainder(
        static_cast<size_t>(leftover) * n_mels, 0.0f);
    if (parakeet::append_nemotron_mel_frames(
            state, remainder.data(), leftover, n_mels) != 0) {
        return 30;
    }
    if (parakeet::next_nemotron_processed_signal(
            state, n_mels, true, processed, frames) != 0) {
        std::fprintf(
            stderr,
            "finalize consumed a remainder shorter than one encoder frame\n");
        return 31;
    }
    return 0;
}

int check_bounded_work(
    parakeet::ParakeetCtcModel & model,
    parakeet::TdtRuntimeWeights & runtime,
    const std::string & reference_dir,
    int right_context) {
    parakeet::NemotronStreamState state;
    if (parakeet::init_nemotron_stream_state(
            model, "en-US", right_context, state) != 0) {
        return 40;
    }

    std::vector<float> first_signal;
    int first_frames = 0;
    int features = 0;
    if (load_processed_signal(
            reference_dir + "/step-000/processed_signal.npy",
            first_signal,
            first_frames,
            features) != 0) {
        return 41;
    }

    parakeet::NemotronStreamStepResult result;
    if (int rc = parakeet::run_nemotron_stream_step(
            model,
            runtime,
            first_signal.data(),
            first_frames,
            features,
            false,
            state,
            result); rc != 0) {
        return rc;
    }

    std::vector<float> steady_signal;
    int steady_frames = 0;
    if (load_processed_signal(
            reference_dir + "/step-001/processed_signal.npy",
            steady_signal,
            steady_frames,
            features) != 0) {
        return 42;
    }
    const int encoder_frames_per_step = right_context + 1;
    const int total_steps = std::max(
        16,
        (56 + encoder_frames_per_step - 1) /
            encoder_frames_per_step);
    for (int step = 1; step < total_steps; ++step) {
        if (int rc = parakeet::run_nemotron_stream_step(
                model,
                runtime,
                steady_signal.data(),
                steady_frames,
                features,
                false,
                state,
                result); rc != 0) {
            return rc;
        }
    }
    if (state.cache_length != 56 ||
        state.max_graph_encoder_frames != right_context + 1 ||
        state.cache_channel.size() !=
            static_cast<size_t>(24 * 56 * 1024) ||
        state.cache_time.size() !=
            static_cast<size_t>(24 * 1024 * 8)) {
        return 43;
    }
    return 0;
}

int check_mel_chunking(
    const parakeet::ParakeetCtcModel & model,
    const std::string & reference_dir,
    int right_context) {
    std::vector<float> source;
    std::vector<int64_t> shape;
    if (load_npy(
            reference_dir + "/offline/mel.npy",
            source,
            shape) != 0 ||
        shape.size() != 2) {
        return 50;
    }
    const int frames = static_cast<int>(shape[0]);
    const int features = static_cast<int>(shape[1]);

    parakeet::NemotronStreamState state;
    if (parakeet::init_nemotron_stream_state(
            model, "en-US", right_context, state) != 0 ||
        parakeet::append_nemotron_mel_frames(
            state, source.data(), frames, features) != 0) {
        return 51;
    }

    for (int step = 0; step < 2; ++step) {
        std::vector<float> actual;
        int actual_frames = 0;
        if (parakeet::next_nemotron_processed_signal(
                state,
                features,
                false,
                actual,
                actual_frames) != 1) {
            return 52;
        }

        std::vector<float> expected;
        int expected_frames = 0;
        int expected_features = 0;
        char step_name[32];
        std::snprintf(
            step_name, sizeof(step_name), "/step-%03d", step);
        if (load_processed_signal(
                reference_dir + step_name +
                    "/processed_signal.npy",
                expected,
                expected_frames,
                expected_features) != 0) {
            return 53;
        }
        const double chunk_cosine = cosine(actual, expected);
        if (actual_frames != expected_frames ||
            expected_features != features ||
            chunk_cosine < 0.999999) {
            std::fprintf(
                stderr,
                "mel chunk %d mismatch: frames=%d/%d cosine=%.8f\n",
                step,
                actual_frames,
                expected_frames,
                chunk_cosine);
            return 53;
        }
    }
    return 0;
}

int check_complete_stream(
    parakeet::ParakeetCtcModel & model,
    parakeet::TdtRuntimeWeights & runtime,
    const std::string & reference_dir,
    int right_context) {
    std::vector<float> mel;
    std::vector<int64_t> shape;
    if (load_npy(
            reference_dir + "/offline/mel.npy",
            mel,
            shape) != 0 ||
        shape.size() != 2) {
        return 60;
    }
    const int mel_frames = static_cast<int>(shape[0]);
    const int features = static_cast<int>(shape[1]);

    parakeet::NemotronStreamState state;
    if (parakeet::init_nemotron_stream_state(
            model, "en-US", right_context, state) != 0 ||
        parakeet::append_nemotron_mel_frames(
            state,
            mel.data(),
            mel_frames,
            features) != 0) {
        return 61;
    }

    parakeet::NemotronStreamStepResult result;
    while (true) {
        std::vector<float> chunk;
        int chunk_frames = 0;
        const int ready = parakeet::next_nemotron_processed_signal(
            state,
            features,
            false,
            chunk,
            chunk_frames);
        if (ready < 0) {
            return ready;
        }
        if (ready == 0) {
            break;
        }
        if (int rc = parakeet::run_nemotron_stream_step(
                model,
                runtime,
                chunk.data(),
                chunk_frames,
                features,
                false,
                state,
                result); rc != 0) {
            return rc;
        }
    }

    std::vector<float> final_chunk;
    int final_chunk_frames = 0;
    const int final_ready = parakeet::next_nemotron_processed_signal(
        state,
        features,
        true,
        final_chunk,
        final_chunk_frames);
    if (final_ready < 0) {
        return final_ready;
    }
    if (final_ready == 1) {
        if (int rc = parakeet::run_nemotron_stream_step(
                model,
                runtime,
                final_chunk.data(),
                final_chunk_frames,
                features,
                true,
                state,
                result); rc != 0) {
            return rc;
        }
    } else if (int rc = parakeet::run_nemotron_stream_step(
                   model,
                   runtime,
                   nullptr,
                   0,
                   features,
                   true,
                   state,
                   result); rc != 0) {
        return rc;
    }

    const std::string expected =
        load_text(reference_dir + "/transcript.txt");
    const std::string actual =
        parakeet::detokenize(model.vocab, state.token_ids);
    if (actual != expected) {
        std::fprintf(
            stderr,
            "stream transcript mismatch\nactual:   %s\nexpected: %s\n",
            actual.c_str(),
            expected.c_str());
        return 62;
    }
    return 0;
}

int drain_pcm_chunks(
    parakeet::ParakeetCtcModel & model,
    parakeet::TdtRuntimeWeights & runtime,
    parakeet::NemotronStreamState & state,
    bool finalize,
    parakeet::NemotronStreamStepResult & result) {
    while (true) {
        std::vector<float> chunk;
        int chunk_frames = 0;
        const int ready = parakeet::next_nemotron_processed_signal(
            state,
            model.mel_cfg.n_mels,
            finalize,
            chunk,
            chunk_frames);
        if (ready <= 0) {
            return ready;
        }
        if (int rc = parakeet::run_nemotron_stream_step(
                model,
                runtime,
                chunk.data(),
                chunk_frames,
                model.mel_cfg.n_mels,
                finalize,
                state,
                result); rc != 0) {
            return rc;
        }
        if (finalize) {
            return 0;
        }
    }
}

int check_incremental_pcm(
    parakeet::ParakeetCtcModel & model,
    parakeet::TdtRuntimeWeights & runtime,
    const std::string & reference_dir,
    const std::string & wav_path,
    int right_context) {
    std::vector<float> samples;
    int sample_rate = 0;
    if (parakeet::load_wav_mono_f32(
            wav_path, samples, sample_rate) != 0 ||
        sample_rate != model.mel_cfg.sample_rate) {
        return 70;
    }

    parakeet::NemotronStreamState state;
    if (parakeet::init_nemotron_stream_state(
            model, "en-US", right_context, state) != 0) {
        return 71;
    }

    const int burst_sizes[] = {
        37, 997, 160, 4093, 511, 73, 2048,
    };
    size_t offset = 0;
    size_t burst_index = 0;
    parakeet::NemotronStreamStepResult result;
    while (offset < samples.size()) {
        const size_t count = std::min(
            samples.size() - offset,
            static_cast<size_t>(
                burst_sizes[
                    burst_index %
                    (sizeof(burst_sizes) / sizeof(burst_sizes[0]))]));
        if (int rc = parakeet::append_nemotron_pcm(
                model,
                state,
                samples.data() + offset,
                static_cast<int>(count),
                false); rc != 0) {
            return rc;
        }
        if (int rc = drain_pcm_chunks(
                model,
                runtime,
                state,
                false,
                result); rc < 0) {
            return rc;
        }
        offset += count;
        ++burst_index;
    }

    if (int rc = parakeet::append_nemotron_pcm(
            model, state, nullptr, 0, true); rc != 0) {
        return rc;
    }
    if (int rc = drain_pcm_chunks(
            model,
            runtime,
            state,
            false,
            result); rc < 0) {
        return rc;
    }
    if (int rc = drain_pcm_chunks(
            model,
            runtime,
            state,
            true,
            result); rc < 0) {
        return rc;
    }
    if (!state.finalized) {
        if (int rc = parakeet::run_nemotron_stream_step(
                model,
                runtime,
                nullptr,
                0,
                model.mel_cfg.n_mels,
                true,
                state,
                result); rc != 0) {
            return rc;
        }
    }

    const std::string actual =
        parakeet::detokenize(model.vocab, state.token_ids);
    const std::string expected =
        load_text(reference_dir + "/transcript.txt");
    if (actual != expected) {
        std::fprintf(
            stderr,
            "incremental PCM transcript mismatch\n"
            "actual:   %s\nexpected: %s\n",
            actual.c_str(),
            expected.c_str());
        return 72;
    }
    return 0;
}

}

int main(int argc, char ** argv) {
    if (argc != 5 && argc != 7) {
        std::fprintf(
            stderr,
            "usage: %s <nemotron.gguf> <reference-dir> "
            "<right-context> <wav> "
            "[--n-gpu-layers N]\n",
            argv[0]);
        return 2;
    }

    int n_gpu_layers = 0;
    if (argc == 7) {
        if (std::strcmp(argv[5], "--n-gpu-layers") != 0) {
            std::fprintf(stderr, "expected --n-gpu-layers, got %s\n", argv[5]);
            return 2;
        }
        n_gpu_layers = std::atoi(argv[6]);
        if (n_gpu_layers < 0) {
            std::fprintf(stderr, "--n-gpu-layers must be non-negative\n");
            return 2;
        }
    }

    const int right_context = std::atoi(argv[3]);

    parakeet::ParakeetCtcModel model;
    if (int rc = parakeet::load_from_gguf(
            argv[1], model, 0, n_gpu_layers, false); rc != 0) {
        return rc;
    }
    if (n_gpu_layers > 0 && !parakeet::model_has_gpu_backend(model)) {
        std::fprintf(
            stderr,
            "GPU execution requested, but active backend is %s\n",
            parakeet::model_active_backend_name(model).c_str());
        return 1;
    }

    parakeet::TdtRuntimeWeights runtime;
    if (int rc = parakeet::tdt_prepare_runtime(
            model, runtime); rc != 0) {
        return rc;
    }

    parakeet::NemotronStreamState state;
    if (int rc = parakeet::init_nemotron_stream_state(
            model, "en-US", right_context, state); rc != 0) {
        return rc;
    }
    const std::string reference_dir = argv[2];
    if (int rc = check_step(
            model,
            runtime,
            state,
            reference_dir + "/step-000",
            right_context + 1); rc != 0) {
        return rc;
    }
    if (int rc = check_step(
            model,
            runtime,
            state,
            reference_dir + "/step-001",
            2 * (right_context + 1)); rc != 0) {
        return rc;
    }
    if (state.max_graph_encoder_frames != right_context + 1) {
        std::fprintf(stderr, "Nemotron graph input grew unexpectedly\n");
        return 30;
    }
    if (int rc = check_lifecycle(
            model,
            runtime,
            reference_dir + "/step-000",
            right_context); rc != 0) {
        return rc;
    }
    if (int rc = check_finalize_drops_short_remainder(
            model, right_context); rc != 0) {
        return rc;
    }
    if (int rc = check_bounded_work(
            model,
            runtime,
            reference_dir,
            right_context); rc != 0) {
        return rc;
    }
    if (int rc = check_mel_chunking(
            model, reference_dir, right_context); rc != 0) {
        return rc;
    }
    if (int rc = check_complete_stream(
            model,
            runtime,
            reference_dir,
            right_context); rc != 0) {
        return rc;
    }
    if (int rc = check_incremental_pcm(
            model,
            runtime,
            reference_dir,
            argv[4],
            right_context); rc != 0) {
        return rc;
    }

    std::fprintf(stderr, "Nemotron cache-aware step tests passed\n");
    return 0;
}
