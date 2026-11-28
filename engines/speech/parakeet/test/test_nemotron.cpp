#include "mel_preprocess.h"
#include "parakeet/engine.h"
#include "parakeet_ctc.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <regex>
#include <string>
#include <vector>

namespace {

template <typename T>
int load_npy(
    const std::string & path,
    std::vector<T> & data,
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
    const size_t shape_start = header.find("'shape':");
    const size_t left = header.find('(', shape_start);
    const size_t right = header.find(')', left);
    if (shape_start == std::string::npos ||
        left == std::string::npos ||
        right == std::string::npos) {
        return 3;
    }

    shape.clear();
    const std::string shape_text =
        header.substr(left + 1, right - left - 1);
    size_t position = 0;
    while (position < shape_text.size()) {
        while (position < shape_text.size() &&
               (shape_text[position] == ' ' ||
                shape_text[position] == ',')) {
            ++position;
        }
        size_t end = position;
        while (end < shape_text.size() &&
               shape_text[end] >= '0' &&
               shape_text[end] <= '9') {
            ++end;
        }
        if (end == position) {
            break;
        }
        shape.push_back(std::stoll(shape_text.substr(position, end - position)));
        position = end;
    }

    size_t element_count = 1;
    for (const int64_t dimension : shape) {
        element_count *= static_cast<size_t>(dimension);
    }
    data.resize(element_count);
    file.read(
        reinterpret_cast<char *>(data.data()),
        element_count * sizeof(T));
    return file ? 0 : 4;
}

std::string load_text(const std::string & path) {
    std::ifstream file(path);
    std::string text(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());
    while (!text.empty() &&
           (text.back() == '\n' || text.back() == '\r')) {
        text.pop_back();
    }
    return text;
}

double cosine(
    const std::vector<float> & actual,
    const std::vector<float> & expected) {
    if (actual.size() < expected.size() || expected.empty()) {
        return 0.0;
    }
    double dot = 0.0;
    double actual_norm = 0.0;
    double expected_norm = 0.0;
    for (size_t index = 0; index < expected.size(); ++index) {
        dot += static_cast<double>(actual[index]) * expected[index];
        actual_norm +=
            static_cast<double>(actual[index]) * actual[index];
        expected_norm +=
            static_cast<double>(expected[index]) * expected[index];
    }
    return dot / std::sqrt(actual_norm * expected_norm);
}

int check_stage_parity(
    const std::string & model_path,
    const std::string & wav_path,
    const std::string & reference_dir) {
    parakeet::ParakeetCtcModel model;
    if (int rc = parakeet::load_from_gguf(
            model_path, model, 0, 0, false); rc != 0) {
        return rc;
    }

    std::vector<float> samples;
    int sample_rate = 0;
    if (int rc = parakeet::load_wav_mono_f32(
            wav_path, samples, sample_rate); rc != 0) {
        return rc;
    }

    std::vector<float> mel;
    int mel_frames = 0;
    if (int rc = parakeet::compute_log_mel(
            samples.data(),
            static_cast<int>(samples.size()),
            model.mel_cfg,
            mel,
            mel_frames); rc != 0) {
        return rc;
    }

    std::vector<int64_t> shape;
    std::vector<float> expected_mel;
    if (load_npy(
            reference_dir + "/mel.npy",
            expected_mel,
            shape) != 0 ||
        shape != std::vector<int64_t>{mel_frames, model.mel_cfg.n_mels}) {
        std::fprintf(stderr, "invalid Nemotron mel reference\n");
        return 20;
    }
    const double mel_cosine = cosine(mel, expected_mel);
    if (mel_cosine < 0.99999) {
        std::fprintf(
            stderr,
            "Nemotron mel cosine %.8f is below threshold\n",
            mel_cosine);
        return 21;
    }

    parakeet::EncoderOutputs encoded;
    if (int rc = parakeet::run_encoder(
            model,
            mel.data(),
            mel_frames,
            model.mel_cfg.n_mels,
            encoded, parakeet::EncoderRoute::ggml(),
            -1,
            false); rc != 0) {
        return rc;
    }

    std::vector<float> expected_encoder;
    if (load_npy(
            reference_dir + "/encoder_raw.npy",
            expected_encoder,
            shape) != 0 ||
        shape != std::vector<int64_t>{
            encoded.n_enc_frames,
            encoded.d_model}) {
        std::fprintf(stderr, "invalid Nemotron encoder reference\n");
        return 22;
    }
    const double encoder_cosine =
        cosine(encoded.encoder_out, expected_encoder);
    if (encoder_cosine < 0.99999) {
        std::fprintf(
            stderr,
            "Nemotron encoder cosine %.8f is below threshold\n",
            encoder_cosine);
        return 23;
    }

    std::vector<float> projected;
    const int32_t prompt_id =
        parakeet::resolve_nemotron_prompt_id(model, "en-US");
    if (int rc = parakeet::run_nemotron_prompt_projection(
            model,
            encoded.encoder_out.data(),
            encoded.n_enc_frames,
            encoded.d_model,
            prompt_id,
            projected); rc != 0) {
        return rc;
    }

    std::vector<float> expected_prompt;
    if (load_npy(
            reference_dir + "/prompt_output.npy",
            expected_prompt,
            shape) != 0 ||
        shape != std::vector<int64_t>{
            encoded.n_enc_frames,
            encoded.d_model}) {
        std::fprintf(stderr, "invalid Nemotron prompt reference\n");
        return 24;
    }
    const double prompt_cosine = cosine(projected, expected_prompt);
    if (prompt_cosine < 0.9999) {
        std::fprintf(
            stderr,
            "Nemotron prompt cosine %.8f is below threshold\n",
            prompt_cosine);
        return 25;
    }

    std::fprintf(
        stderr,
        "Nemotron stage cosine: mel=%.8f encoder=%.8f prompt=%.8f\n",
        mel_cosine,
        encoder_cosine,
        prompt_cosine);
    return 0;
}

std::string remove_locale_tags(const std::string & text) {
    static const std::regex locale_tag(R"(<[A-Za-z]{2,3}-[A-Za-z0-9]{2,4}>)");
    static const std::regex space_run(R"( {2,})");
    const std::string without_tags = std::regex_replace(
        std::regex_replace(text, locale_tag, ""), space_run, " ");
    const size_t first = without_tags.find_first_not_of(' ');
    const size_t last = without_tags.find_last_not_of(' ');
    return first == std::string::npos
        ? std::string()
        : without_tags.substr(first, last - first + 1);
}

std::string load_tag_free_transcript(const std::string & reference_dir) {
    return remove_locale_tags(load_text(reference_dir + "/transcript.txt"));
}

int check_transcript(
    const std::string & model_path,
    const std::string & wav_path,
    const std::string & language,
    const std::string & reference_dir) {
    parakeet::EngineOptions options;
    options.model_gguf_path = model_path;
    options.language = language;
    parakeet::Engine engine(options);
    const parakeet::EngineResult result = engine.transcribe(wav_path);

    std::vector<int32_t> expected_tokens;
    std::vector<int64_t> shape;
    if (load_npy(
            reference_dir + "/token_ids.npy",
            expected_tokens,
            shape) != 0) {
        std::fprintf(stderr, "invalid Nemotron token reference\n");
        return 30;
    }
    if (result.token_ids != expected_tokens) {
        std::fprintf(
            stderr,
            "Nemotron token mismatch for locale %s\n",
            language.c_str());
        return 31;
    }

    const std::string expected_text =
        load_tag_free_transcript(reference_dir);
    if (result.text != expected_text) {
        std::fprintf(
            stderr,
            "Nemotron transcript mismatch for locale %s\n",
            language.c_str());
        return 32;
    }
    return 0;
}

int check_public_streaming(
    const std::string & model_path,
    const std::string & wav_path,
    const std::string & reference_dir) {
    parakeet::EngineOptions options;
    options.model_gguf_path = model_path;
    options.language = "en-US";
    parakeet::Engine engine(options);

    parakeet::StreamingOptions streaming_options;
    streaming_options.sample_rate = 16000;
    streaming_options.chunk_ms = 320;

    const std::string expected_text =
        load_tag_free_transcript(reference_dir);

    int callback_index = 0;
    bool callback_order_ok = true;
    std::string callback_text;
    const auto collect_segment =
        [&](const parakeet::StreamingSegment & segment) {
            if (segment.chunk_index != callback_index ||
                segment.end_s < segment.start_s) {
                callback_order_ok = false;
            }
            callback_text += segment.text;
            ++callback_index;
        };

    const parakeet::EngineResult callback_result =
        engine.transcribe_stream(
            wav_path,
            streaming_options,
            collect_segment);
    if (!callback_order_ok ||
        callback_index == 0 ||
        callback_text != expected_text ||
        callback_result.text != expected_text) {
        std::fprintf(
            stderr,
            "Nemotron callback streaming mismatch\n"
            "chunks=%d order_ok=%d\n"
            "actual:   %s\n"
            "result:   %s\n"
            "expected: %s\n",
            callback_index,
            callback_order_ok ? 1 : 0,
            callback_text.c_str(),
            callback_result.text.c_str(),
            expected_text.c_str());
        return 40;
    }

    std::vector<float> samples;
    int sample_rate = 0;
    if (int rc = parakeet::load_wav_mono_f32(
            wav_path, samples, sample_rate); rc != 0) {
        return rc;
    }

    callback_index = 0;
    callback_order_ok = true;
    callback_text.clear();
    auto session = engine.stream_start(
        streaming_options,
        collect_segment);

    const int burst_sizes[] = {
        37, 997, 160, 4093, 511, 73, 2048,
    };
    size_t offset = 0;
    size_t burst_index = 0;
    while (offset < samples.size()) {
        const size_t count = std::min(
            samples.size() - offset,
            static_cast<size_t>(burst_sizes[
                burst_index %
                (sizeof(burst_sizes) / sizeof(burst_sizes[0]))]));
        session->feed_pcm_f32(
            samples.data() + offset,
            static_cast<int>(count));
        offset += count;
        ++burst_index;
    }
    session->finalize();

    if (!callback_order_ok ||
        callback_index == 0 ||
        callback_text != expected_text) {
        std::fprintf(
            stderr,
            "Nemotron live StreamSession mismatch\n"
            "chunks=%d order_ok=%d\n"
            "actual:   %s\n"
            "expected: %s\n",
            callback_index,
            callback_order_ok ? 1 : 0,
            callback_text.c_str(),
            expected_text.c_str());
        return 41;
    }

    std::fprintf(
        stderr,
        "Nemotron public streaming passed: %d chunks\n",
        callback_index);
    return 0;
}

}

int main(int argc, char ** argv) {
    if (argc != 7) {
        std::fprintf(
            stderr,
            "usage: %s <model> <jfk.wav> <en-ref> <auto-ref> "
            "<hi.wav> <hi-ref>\n",
            argv[0]);
        return 2;
    }

    if (int rc = check_stage_parity(argv[1], argv[2], argv[3]); rc != 0) {
        return rc;
    }
    if (int rc = check_transcript(
            argv[1], argv[2], "en-US", argv[3]); rc != 0) {
        return rc;
    }
    if (int rc = check_public_streaming(
            argv[1], argv[2], argv[3]); rc != 0) {
        return rc;
    }
    if (int rc = check_transcript(
            argv[1], argv[2], "", argv[4]); rc != 0) {
        return rc;
    }
    if (int rc = check_transcript(
            argv[1], argv[5], "hi-IN", argv[6]); rc != 0) {
        return rc;
    }

    std::fprintf(stderr, "Nemotron tests passed\n");
    return 0;
}
