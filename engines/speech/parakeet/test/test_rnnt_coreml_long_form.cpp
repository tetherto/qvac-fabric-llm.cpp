// Transducer fixed-capacity Core ML long-form regression gate.
//
// The JFK fixture is tiled past the sidecar's mel-frame capacity, then
// transcribed in separate child processes with Core ML enabled and disabled.
// An over-capacity result can report encoder_used_coreml only when the Engine
// selected the fixed-capacity long-form plan and every window used Core ML.

#include "coreml_test_options.h"
#include "parakeet/engine.h"
#include "test_utils.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

#ifndef _WIN32
namespace {

struct ChildResult {
    bool parsed = false;
    bool coreml_available = false;
    bool coreml_planned = false;
    bool coreml_used = false;
    int mel_frames = 0;
    int encoder_frames = 0;
    std::vector<int32_t> tokens;
};

std::vector<float> tile_to_seconds(const std::vector<float> & source,
                                   int sample_rate, int target_seconds) {
    const size_t target = static_cast<size_t>(sample_rate) *
                          static_cast<size_t>(target_seconds);
    std::vector<float> samples;
    samples.reserve(target);
    while (samples.size() < target) {
        const size_t remaining = target - samples.size();
        const size_t count = std::min(remaining, source.size());
        samples.insert(samples.end(), source.begin(), source.begin() + count);
    }
    return samples;
}

std::string run_child(const std::string & gguf, const std::string & wav,
                      int target_seconds, bool disable_coreml,
                      bool disable_windowing, bool deterministic_cpu = false) {
    int fds[2];
    if (pipe(fds) != 0) return {};

    const pid_t pid = fork();
    if (pid == 0) {
        close(fds[0]);
        if (disable_coreml) {
            setenv("PARAKEET_COREML_DISABLE", "1", 1);
        } else {
            unsetenv("PARAKEET_COREML_DISABLE");
        }

        std::vector<float> source;
        int sample_rate = 0;
        if (!parakeet_test::load_wav_pcm16le_mono(wav, source, sample_rate) ||
            source.empty()) {
            close(fds[1]);
            _exit(2);
        }
        const std::vector<float> samples =
            tile_to_seconds(source, sample_rate, target_seconds);

        try {
            parakeet::EngineOptions opts;
            opts.model_gguf_path = gguf;
            opts.n_gpu_layers = 999;
            if (deterministic_cpu) {
                // Exact token equality needs a stable reduction order in both
                // children; other modes exercise the GPU with a similarity gate.
                constexpr int kReferenceThreads = 2;
                opts.n_gpu_layers = 0;
                opts.n_threads = kReferenceThreads;
            }
            if (disable_windowing) {
                opts.long_form_window_frames = -1;
            }
            // Wait for every sidecar at construction: they load in the
            // background otherwise, and the call would run on ggml.
            if (!disable_coreml) opts.coreml = coreml_ready_options();
            parakeet::Engine engine(opts);
            const parakeet::EngineResult result = engine.transcribe_samples(
                samples.data(), static_cast<int>(samples.size()), sample_rate);

            std::ostringstream payload;
            payload << "coreml_available=" << (engine.encoder_on_coreml() ? 1 : 0)
                    << "\ncoreml_planned="
                    << (coreml_sidecar_planned(engine.coreml_status_json()) ? 1 : 0)
                    << "\ncoreml_used=" << (result.encoder_used_coreml ? 1 : 0)
                    << "\nmel_frames=" << result.mel_frames
                    << "\nencoder_frames=" << result.encoder_frames
                    << "\ntokens=";
            for (size_t i = 0; i < result.token_ids.size(); ++i) {
                if (i > 0) payload << ',';
                payload << result.token_ids[i];
            }
            payload << '\n';
            const std::string body = payload.str();
            for (size_t offset = 0; offset < body.size();) {
                const ssize_t written =
                    write(fds[1], body.data() + offset, body.size() - offset);
                if (written <= 0) break;
                offset += static_cast<size_t>(written);
            }
            close(fds[1]);
            _exit(0);
        } catch (...) {
            close(fds[1]);
            _exit(3);
        }
    }

    close(fds[1]);
    std::string output;
    char chunk[4096];
    ssize_t count = 0;
    while ((count = read(fds[0], chunk, sizeof(chunk))) > 0) {
        output.append(chunk, static_cast<size_t>(count));
    }
    close(fds[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) return {};
    return output;
}

bool parse_bool_line(const std::string & line, const char * key, bool & value) {
    const std::string prefix = std::string(key) + '=';
    if (line.rfind(prefix, 0) != 0) return false;
    value = line.substr(prefix.size()) == "1";
    return true;
}

ChildResult parse_result(const std::string & raw) {
    ChildResult result;
    std::istringstream input(raw);
    std::string line;
    if (!std::getline(input, line) ||
        !parse_bool_line(line, "coreml_available", result.coreml_available)) return result;
    if (!std::getline(input, line) ||
        !parse_bool_line(line, "coreml_planned", result.coreml_planned)) return result;
    if (!std::getline(input, line) ||
        !parse_bool_line(line, "coreml_used", result.coreml_used)) return result;
    if (!std::getline(input, line) || line.rfind("mel_frames=", 0) != 0) return result;
    result.mel_frames = std::atoi(line.c_str() + 11);
    if (!std::getline(input, line) || line.rfind("encoder_frames=", 0) != 0) return result;
    result.encoder_frames = std::atoi(line.c_str() + 15);
    if (!std::getline(input, line) || line.rfind("tokens=", 0) != 0) return result;

    std::istringstream tokens(line.substr(7));
    std::string token;
    while (std::getline(tokens, token, ',')) {
        if (!token.empty()) result.tokens.push_back(std::atoi(token.c_str()));
    }
    result.parsed = true;
    return result;
}

double token_similarity(const std::vector<int32_t> & lhs,
                        const std::vector<int32_t> & rhs) {
    if (lhs.empty() && rhs.empty()) return 1.0;
    std::vector<size_t> previous(rhs.size() + 1), current(rhs.size() + 1);
    for (size_t j = 0; j <= rhs.size(); ++j) previous[j] = j;
    for (size_t i = 1; i <= lhs.size(); ++i) {
        current[0] = i;
        for (size_t j = 1; j <= rhs.size(); ++j) {
            const size_t cost = lhs[i - 1] == rhs[j - 1] ? 0 : 1;
            current[j] = std::min({previous[j] + 1, current[j - 1] + 1,
                                   previous[j - 1] + cost});
        }
        std::swap(previous, current);
    }
    const size_t denominator = std::max<size_t>(1, std::max(lhs.size(), rhs.size()));
    return 1.0 - static_cast<double>(previous[rhs.size()]) /
                     static_cast<double>(denominator);
}

}  // namespace
#endif

int main(int argc, char ** argv) {
#ifdef _WIN32
    (void) argc;
    (void) argv;
    std::fprintf(stderr, "[rnnt-coreml-long-form] SKIP: Core ML is Apple-only.\n");
    return 0;
#else
    if (argc != 7) {
        std::fprintf(stderr,
            "usage: %s <rnnt.gguf> <wav> <fixed-mel-frames> "
            "<target-seconds> <min-token-similarity> "
            "<windowed-coreml|cache-aware|native-context>\n", argv[0]);
        return 2;
    }
    const std::string gguf = argv[1];
    const std::string wav = argv[2];
    const int fixed_mel_frames = std::atoi(argv[3]);
    const int target_seconds = std::atoi(argv[4]);
    const double minimum_similarity = std::atof(argv[5]);
    const std::string mode = argv[6];
    const bool cache_aware = mode == "cache-aware";
    const bool native_context = mode == "native-context";
    if (!cache_aware && !native_context && mode != "windowed-coreml") {
        std::fprintf(stderr, "invalid long-form test mode: %s\n", mode.c_str());
        return 2;
    }

    const ChildResult coreml = parse_result(
        run_child(gguf, wav, target_seconds,
                  /*disable_coreml=*/false,
                  /*disable_windowing=*/false, native_context));
    if (!coreml.parsed) {
        std::fprintf(stderr, "[rnnt-coreml-long-form] FAIL: Core ML child failed\n");
        return 1;
    }
    if (!coreml.coreml_available && coreml.coreml_planned) {
        std::fprintf(stderr,
            "[rnnt-coreml-long-form] FAIL: a Core ML sidecar is planned but was not active "
            "after waiting at construction\n");
        return 1;
    }
    if (!coreml.coreml_available) {
        std::fprintf(stderr,
            "[rnnt-coreml-long-form] SKIP: Core ML sidecar is not active.\n");
        return 0;
    }
    if (coreml.mel_frames <= fixed_mel_frames) {
        std::fprintf(stderr,
            "[rnnt-coreml-long-form] FAIL: fixture has %d mel frames, expected > %d\n",
            coreml.mel_frames, fixed_mel_frames);
        return 1;
    }
    if (!cache_aware && !native_context && !coreml.coreml_used) {
        std::fprintf(stderr,
            "[rnnt-coreml-long-form] FAIL: an over-capacity encoder window fell back to ggml\n");
        return 1;
    }
    if (cache_aware && coreml.coreml_used) {
        std::fprintf(stderr,
            "[rnnt-coreml-long-form] FAIL: oversized Nemotron input used independent Core ML windows\n");
        return 1;
    }
    if (native_context && coreml.coreml_used) {
        std::fprintf(stderr,
            "[rnnt-coreml-long-form] FAIL: oversized TDT input used unvalidated Core ML seams\n");
        return 1;
    }

    const ChildResult ggml = parse_result(
        run_child(gguf, wav, target_seconds,
                  /*disable_coreml=*/true,
                  /*disable_windowing=*/cache_aware, native_context));
    if (!ggml.parsed || ggml.coreml_available || ggml.coreml_planned || ggml.coreml_used) {
        std::fprintf(stderr,
            "[rnnt-coreml-long-form] FAIL: forced-ggml reference routing is invalid\n");
        return 1;
    }
    if (coreml.tokens.empty() || ggml.tokens.empty()) {
        std::fprintf(stderr,
            "[rnnt-coreml-long-form] FAIL: transcription returned no tokens\n");
        return 1;
    }
    if (native_context && (coreml.tokens != ggml.tokens ||
                           coreml.encoder_frames != ggml.encoder_frames)) {
        std::fprintf(stderr,
            "[rnnt-coreml-long-form] FAIL: TDT fallback differs from native reference "
            "(frames %d/%d, tokens %zu/%zu, similarity %.4f)\n",
            coreml.encoder_frames, ggml.encoder_frames, coreml.tokens.size(),
            ggml.tokens.size(), token_similarity(coreml.tokens, ggml.tokens));
        return 1;
    }
    if (cache_aware && coreml.encoder_frames != ggml.encoder_frames) {
        std::fprintf(stderr,
            "[rnnt-coreml-long-form] FAIL: cache-aware encoder frames %d "
            "do not match unwindowed reference %d\n",
            coreml.encoder_frames, ggml.encoder_frames);
        return 1;
    }

    const double similarity = token_similarity(coreml.tokens, ggml.tokens);
    std::printf(
        "[rnnt-coreml-long-form] mel=%d coreml_tokens=%zu ggml_tokens=%zu "
        "token_similarity=%.4f\n",
        coreml.mel_frames, coreml.tokens.size(), ggml.tokens.size(), similarity);
    if (similarity < minimum_similarity) {
        std::fprintf(stderr,
            "[rnnt-coreml-long-form] FAIL: token similarity %.4f is below %.4f\n",
            similarity, minimum_similarity);
        return 1;
    }

    std::printf("[rnnt-coreml-long-form] PASS\n");
    return 0;
#endif
}
