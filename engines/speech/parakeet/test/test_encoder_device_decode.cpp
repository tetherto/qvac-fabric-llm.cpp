#include "encoder_device_decode.h"
#include "backend_util.h"
#include "parakeet/engine.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

using namespace parakeet;

namespace {
constexpr int kThreads = 8;
constexpr int kRepeatedRuns = 3;
constexpr int kInjectedError = -73;
constexpr int kBypassFrames = 16;
constexpr int kValidMelFrames = 1001;
constexpr int kPaddingMelFrames = 1;
constexpr int kValidEncoderFrames = 126;
constexpr int kPaddedEncoderFrames = 127;

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

TdtDecodeOptions decode_options(const ParakeetCtcModel & model) {
    TdtDecodeOptions options;
    if (model.model_type == ParakeetModelType::RNNT) {
        options.max_symbols_per_step = model.encoder_cfg.rnnt_max_symbols_per_step;
    } else if (model.model_type == ParakeetModelType::NEMOTRON) {
        options.max_symbols_per_step = model.nemotron_cfg.max_symbols_per_step;
    }
    return options;
}

int32_t prompt_id(const ParakeetCtcModel & model) {
    return model.model_type == ParakeetModelType::NEMOTRON
        ? resolve_nemotron_prompt_id(model, "en-US") : -1;
}

TdtDecodeResult decode_host(ParakeetCtcModel & model, TdtRuntimeWeights & runtime,
                            const EncoderOutputs & encoded) {
    std::vector<float> conditioned;
    const float * input = encoded.encoder_out.data();
    if (model.model_type == ParakeetModelType::NEMOTRON) {
        require(run_nemotron_prompt_projection(model, input, encoded.n_enc_frames,
            encoded.d_model, prompt_id(model), conditioned) == 0, "Host prompt failed");
        input = conditioned.data();
    }
    TdtDecodeResult result;
    require(tdt_greedy_decode(model, runtime, input, encoded.n_enc_frames,
        encoded.d_model, decode_options(model), result) == 0, "Host decode failed");
    return result;
}

void check_callbacks(ParakeetCtcModel & model, const std::vector<float> & mel, int frames,
                     const EncoderOutputs & reference) {
    EncoderOutputs output;
    int calls = 0;
    const EncoderOutputConsumer decline = [&](ggml_tensor *, ggml_backend_t) {
        ++calls;
        return kEncoderOutputHost;
    };
    require(run_encoder(model, mel.data(), frames, model.mel_cfg.n_mels, output, parakeet::EncoderRoute::ggml(),
        -1, false, false, decline) == 0, "Declined consumer failed");
    require(calls == 1 && !output.encoder_consumed && output.encoder_out == reference.encoder_out,
        "Declined consumer did not preserve host output");
    calls = 0;
    require(run_encoder(model, mel.data(), frames, model.mel_cfg.n_mels, output, parakeet::EncoderRoute::ggml(),
        -1, true, false, decline) == 0, "Diagnostic encode failed");
    require(calls == 0 && !output.encoder_consumed && !output.encoder_out.empty(),
        "Diagnostic capture invoked consumer");
    const EncoderOutputConsumer fail = [](ggml_tensor *, ggml_backend_t) { return kInjectedError; };
    require(run_encoder(model, mel.data(), frames, model.mel_cfg.n_mels, output, parakeet::EncoderRoute::ggml(),
        -1, false, false, fail) == kInjectedError, "Consumer error was lost");
}

void check_handoff(ParakeetCtcModel & model, TdtRuntimeWeights & runtime,
                    const std::vector<float> & mel, int frames,
                    const std::vector<int32_t> & expected) {
    EncoderOutputs output;
    const bool gpu = !backend_is_cpu(model.backend_active());
    for (int run = 0; run < kRepeatedRuns; ++run) {
        TdtDecodeResult decoded;
        double elapsed = 0;
        const auto decode = make_device_encoder_consumer(model, runtime, prompt_id(model),
            decode_options(model), decoded, elapsed);
        const EncoderOutputConsumer consumer = [&](ggml_tensor * input, ggml_backend_t backend) {
            ggml_tensor invalid = *input;
            invalid.type = GGML_TYPE_F16;
            require(!tdt_accepts_device_input(runtime, &invalid, backend), "Accepted F16 input");
            invalid = *input;
            invalid.ne[1] = runtime.enc_proj_T_max + 1;
            require(!tdt_accepts_device_input(runtime, &invalid, backend), "Accepted oversized input");
            require(!tdt_accepts_device_input(runtime, input, nullptr), "Accepted missing backend");
            if (model.model_type == ParakeetModelType::NEMOTRON) {
                ggml_tensor * projected = nullptr;
                const int width = model.nemotron_cfg.prompt_input_width;
                model.nemotron_cfg.prompt_input_width = width + 1;
                const int rc = run_nemotron_prompt_projection_device(model, input, prompt_id(model), projected);
                model.nemotron_cfg.prompt_input_width = width;
                require(rc < 0 && !projected, "Accepted inconsistent prompt dimensions");
                require(run_nemotron_prompt_projection_device(model, input,
                    model.nemotron_cfg.num_prompts, projected) < 0, "Accepted invalid prompt ID");
            }
            return decode(input, backend);
        };
        require(run_encoder(model, mel.data(), frames, model.mel_cfg.n_mels, output, parakeet::EncoderRoute::ggml(),
            -1, false, false, consumer) == 0, "Device encode/decode failed");
        require(output.encoder_consumed == gpu, "Unexpected handoff selection");
        if (gpu) {
            require(output.encoder_out.empty() && elapsed > 0, "Handoff retained host output");
            require(decoded.token_ids == expected, "Device tokens differ from host");
        } else {
            require(decode_host(model, runtime, output).token_ids == expected, "CPU fallback changed tokens");
        }
    }
    require(run_encoder(model, mel.data(), frames, model.mel_cfg.n_mels, output, parakeet::EncoderRoute::ggml(),
        -1, false) == 0 && !output.encoder_consumed && !output.encoder_out.empty(),
        "Host reuse retained consumed state");
}

void check_prompt_reuse(ParakeetCtcModel & model, const EncoderOutputs & encoded) {
    if (model.model_type != ParakeetModelType::NEMOTRON) return;
    std::vector<float> first, second, repeated;
    require(run_nemotron_prompt_projection(model, encoded.encoder_out.data(), encoded.n_enc_frames,
        encoded.d_model, 0, first) == 0, "First prompt failed");
    require(run_nemotron_prompt_projection(model, encoded.encoder_out.data(), encoded.n_enc_frames,
        encoded.d_model, 1, second) == 0, "Changed prompt failed");
    require(run_nemotron_prompt_projection(model, encoded.encoder_out.data(), encoded.n_enc_frames,
        encoded.d_model, 0, repeated) == 0, "Repeated prompt failed");
    require(first == repeated && first != second, "Prompt cache did not track locale changes");
}

void check_bypass_reuse(ParakeetCtcModel & model) {
    EncoderOutputs output;
    output.encoder_consumed = true;
    const int width = model.encoder_cfg.d_model;
    std::vector<float> features(static_cast<size_t>(width) * kBypassFrames, 0.0f);
    require(run_encoder_bypass_pre_encode(model, features.data(), kBypassFrames, width,
        output, 0) == 0, "Bypass encode failed");
    require(!output.encoder_consumed && !output.encoder_out.empty(), "Bypass retained consumed state");
}

std::vector<float> padded_mel(const ParakeetCtcModel & model, const std::vector<float> & mel) {
    const int width = model.mel_cfg.n_mels;
    const size_t valid_values = static_cast<size_t>(kValidMelFrames) * width;
    require(mel.size() >= valid_values, "Audio is too short for padded regression");
    std::vector<float> padded(static_cast<size_t>(kValidMelFrames + kPaddingMelFrames) * width, 0.0f);
    std::copy_n(mel.begin(), valid_values, padded.begin());
    require(std::any_of(padded.begin() + valid_values - width, padded.begin() + valid_values,
        [](float value) { return value != 0.0f; }), "Last valid mel frame is empty");
    return padded;
}

void check_padded_handoff(ParakeetCtcModel & model, TdtRuntimeWeights & runtime,
                          const std::vector<float> & mel) {
    if (model.model_type != ParakeetModelType::NEMOTRON) return;
    const auto padded = padded_mel(model, mel);
    const int frames = kValidMelFrames + kPaddingMelFrames;
    EncoderOutputs reference;
    require(run_encoder(model, padded.data(), frames, model.mel_cfg.n_mels,
        reference, parakeet::EncoderRoute::ggml(), -1, false) == 0, "Padded reference encode failed");
    require(reference.n_enc_frames == kValidEncoderFrames && reference.encoder_out.size() ==
        static_cast<size_t>(kPaddedEncoderFrames) * reference.d_model, "Expected padded encoder geometry");
    const auto expected = decode_host(model, runtime, reference);
    TdtDecodeResult decoded;
    double elapsed = 0;
    const auto decode = make_device_encoder_consumer(model, runtime, prompt_id(model),
        decode_options(model), decoded, elapsed);
    int calls = 0;
    const EncoderOutputConsumer consumer = [&](ggml_tensor * input, ggml_backend_t backend) {
        ++calls;
        return decode(input, backend);
    };
    EncoderOutputs output;
    require(run_encoder(model, padded.data(), frames, model.mel_cfg.n_mels, output, parakeet::EncoderRoute::ggml(),
        -1, false, false, consumer) == 0, "Padded consumer encode failed");
    require(calls == 0 && !output.encoder_consumed, "Consumer received padded encoder frames");
    require(output.n_enc_frames == kValidEncoderFrames && output.encoder_out == reference.encoder_out,
        "Padded fallback did not preserve host output");
    require(decode_host(model, runtime, output).token_ids == expected.token_ids,
        "Padded fallback tokens differ from host");
}

void run_test(const std::string & path, const std::string & wav, const std::string & backend) {
    ParakeetCtcModel model;
    require(load_from_gguf(path, model, kThreads, 0, false, backend) == 0, "Model load failed");
    TdtRuntimeWeights runtime;
    require(tdt_prepare_runtime(model, runtime) == 0, "Runtime preparation failed");
    std::vector<float> samples, mel;
    int sample_rate = 0, frames = 0;
    require(load_wav_mono_f32(wav, samples, sample_rate) == 0, "Audio load failed");
    require(compute_log_mel(samples.data(), (int) samples.size(), model.mel_cfg, mel, frames) == 0,
        "Mel computation failed");
    EncoderOutputs reference;
    require(run_encoder(model, mel.data(), frames, model.mel_cfg.n_mels, reference, parakeet::EncoderRoute::ggml(),
        -1, false) == 0, "Reference encoder failed");
    const auto expected = decode_host(model, runtime, reference);
    check_handoff(model, runtime, mel, frames, expected.token_ids);
    check_callbacks(model, mel, frames, reference);
    check_prompt_reuse(model, reference);
    check_bypass_reuse(model);
    check_padded_handoff(model, runtime, mel);

    EngineOptions options;
    options.model_gguf_path = path;
    options.backend = backend;
    options.n_threads = kThreads;
    options.language = model.model_type == ParakeetModelType::NEMOTRON ? "en-US" : "";
    Engine engine(options);
    const auto result = engine.transcribe_samples(samples.data(), (int) samples.size(), sample_rate);
    require(result.token_ids == expected.token_ids, "Engine tokens differ from host path");
    std::printf("Encoder handoff, fallback, cache reuse and Engine parity passed (%s).\n", backend.c_str());
}
} // namespace

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: test-encoder-device-decode model.gguf audio.wav backend\n");
        return 2;
    }
#ifdef _WIN32
    _putenv_s("PARAKEET_COREML_DISABLE", "1");
#else
    setenv("PARAKEET_COREML_DISABLE", "1", 1);
#endif
    try { run_test(argv[1], argv[2], argv[3]); }
    catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
