// Full-checkpoint GPU regression: two replies on the same engine must fit,
// whether it keeps the LM and decoder resident together or stages them per
// turn, and a long voice prompt with a short reply must release resident
// stages sized for the default voice.
#include "tts-cpp/moss/speech.h"
#include "chatterbox_tts_test_hooks.h"
#include "voice_features.h"
#include "dr_wav.h"
#include "../test_env_portable.h"
#include "../../../test/moss_gpu_arm.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static void write_reply(const std::filesystem::path & path,
                        const tts_cpp::moss::SpeechResult & result) {
    if (result.cancelled || result.reply_tokens == 0 || result.sample_rate <= 0 ||
        result.pcm.size() < (size_t) result.sample_rate / 4 ||
        !std::all_of(result.pcm.begin(), result.pcm.end(), [](float x) { return std::isfinite(x); }) ||
        !std::any_of(result.pcm.begin(), result.pcm.end(), [](float x) { return std::fabs(x) > 1e-5f; })) {
        throw std::runtime_error("reply must contain at least 250 ms of finite, non-silent audio");
    }
    drwav_data_format format{drwav_container_riff, DR_WAVE_FORMAT_PCM, 1,
                            (drwav_uint32) result.sample_rate, 16};
    drwav wav{};
    if (!drwav_init_file_write(&wav, path.string().c_str(), &format, nullptr)) {
        throw std::runtime_error("cannot open reply WAV");
    }
    std::vector<drwav_int16> pcm(result.pcm.size());
    drwav_f32_to_s16(pcm.data(), result.pcm.data(), pcm.size());
    const auto written = drwav_write_pcm_frames(&wav, pcm.size(), pcm.data());
    drwav_uninit(&wav);
    if (written != pcm.size()) throw std::runtime_error("cannot write reply WAV");
}

static std::vector<float> long_voice(const std::vector<float> & audio, int sample_rate) {
    const size_t target = (size_t) sample_rate * 59;
    std::vector<float> voice;
    while (!audio.empty() && voice.size() < target) {
        voice.insert(voice.end(), audio.begin(), audio.begin() + (std::ptrdiff_t) std::min(audio.size(), target - voice.size()));
    }
    return voice;
}

int main(int argc, char ** argv) {
    if (argc != 6) {
        std::fprintf(stderr, "usage: %s cuda|vulkan|hexagon|opencl|webgpu LM.gguf CODEC.gguf INPUT.wav OUTPUT_DIR\n", argv[0]);
        return 77;
    }
    try {
        const MossGpuArm arm = moss_gpu_arm(argv[1]);
        tts_cpp::moss::SpeechOptions options;
        if (arm.request == "hexagon") {
            options.backend = arm.request;
        } else {
            setenv("TTS_CPP_GPU_BACKEND", arm.request.c_str(), 1);
        }
        options.model_path = argv[2];
        options.codec_path = argv[3];
        options.use_gpu = true;
        options.n_threads = 4;
        tts_cpp::moss::SpeechEngine engine(options);
        const std::string backend = engine.backend_name();
        arm.require(backend.c_str());
        std::fprintf(stderr, "[moss-speech-e2e] backend: %s\n", backend.c_str());
        tts_cpp::moss::SpeechMessage message;
        if (!wav_load(argv[4], message.audio, message.sample_rate)) throw std::runtime_error("cannot read input WAV");
        tts_cpp::moss::SpeechRequest request;
        request.messages.push_back(std::move(message));
        request.max_new_tokens = 256;
        request.max_reply_seconds = 4;
        request.seed = 1234;
        for (int turn = 0; turn < 2; ++turn) {
            const auto result = engine.respond(request);
            if (backend != engine.backend_name()) throw std::runtime_error("backend changed after reply");
            write_reply(std::filesystem::path(argv[5]) / (turn == 0 ? "reply.wav" : "reply-repeat.wav"), result);
            std::fprintf(stderr, "reply %d: %d codes, %.2f seconds, prefill %.0f ms, generate %.0f ms, decode %.0f ms\n",
                         turn + 1, result.reply_tokens, (double) result.pcm.size() / result.sample_rate,
                         result.prefill_ms, result.generate_ms, result.decode_ms);
        }
        namespace hooks = tts_cpp::chatterbox::test_hooks;
        const bool resident = hooks::s3gen_cached_weights_bytes() > 0;
        // A 59 s voice and a 20 s reply outgrow the plan (the ~36 s default
        // voice and a 30 s reply) although the reply is under 30 s.
        request.voice = long_voice(request.messages.front().audio, request.messages.front().sample_rate);
        request.voice_sample_rate = request.messages.front().sample_rate;
        request.max_new_tokens = 512;
        request.max_reply_seconds = 20;
        const auto result = engine.respond(request);
        write_reply(std::filesystem::path(argv[5]) / "reply-long-voice.wav", result);
        std::fprintf(stderr, "long-voice reply: %d codes, decode %.0f ms, decoder %s\n", result.reply_tokens,
                     result.decode_ms, resident ? "was resident" : "was staged");
        if (result.reply_tokens < 200) {
            throw std::runtime_error("the long-voice reply is too short to outgrow the plan");
        }
        if (resident && hooks::s3gen_cached_weights_bytes() != 0) {
            throw std::runtime_error("a long voice prompt kept the decoder resident beyond its plan");
        }
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "moss speech e2e: %s\n", error.what());
        return 1;
    }
}
