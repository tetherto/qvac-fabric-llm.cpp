// ggml GPU synthesis vs the Core ML sidecar on the same code sequences,
// parity-gated, markdown table to stdout. Skips (77) like the parity test and
// when no GPU ggml backend resolves.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
constexpr int SKIP_EXIT_CODE = 77;
}

#ifndef TTS_CPP_USE_COREML

int main() {
    std::fprintf(stderr, "[bench-audio8-coreml] built without TTS_CPP_COREML; skipping\n");
    return SKIP_EXIT_CODE;
}

#else

#include "backend_util.h"

#include "audio8/coreml_path.h"
#include "audio8/internal.h"
#include "test_env_portable.h"

#include <sys/stat.h>

using namespace tts_cpp::audio8::detail;

namespace {

bool path_exists(const std::string & path) {
    struct stat st {};
    return stat(path.c_str(), &st) == 0;
}

std::string find_decoder_gguf(const std::string & dir) {
    const char * candidates[] = {"audio8-codec-decoder-f32.gguf", "audio8-codec-decoder-f16.gguf",
                                 "audio8-codec-decoder-q8_0.gguf", "audio8-codec-decoder.gguf"};
    for (const char * name : candidates) {
        const std::string path = dir + "/" + name;
        if (path_exists(path)) return path;
    }
    return {};
}

// Deterministic codes inside every codebook, [num_codebooks, n_frames]
// row-major as decode_codes reads them.
std::vector<int32_t> make_codes(const codec_hparams & hp, int n_frames, uint32_t seed) {
    std::vector<int32_t> codes(static_cast<size_t>(hp.num_codebooks) * n_frames);
    uint32_t state = seed;
    for (int book = 0; book < hp.num_codebooks; ++book) {
        const int limit = book == 0 ? hp.semantic_codebook_size : hp.residual_codebook_size;
        for (int frame = 0; frame < n_frames; ++frame) {
            state = state * 1664525u + 1013904223u;
            codes[static_cast<size_t>(book) * n_frames + frame] =
                static_cast<int32_t>((state >> 8) % static_cast<uint32_t>(limit));
        }
    }
    return codes;
}

double cosine(const std::vector<float> & a, const std::vector<float> & b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += (double) a[i] * (double) b[i];
        na  += (double) a[i] * (double) a[i];
        nb  += (double) b[i] * (double) b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
}

double max_abs_err(const std::vector<float> & a, const std::vector<float> & b) {
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        worst = std::max(worst, (double) std::fabs(a[i] - b[i]));
    }
    return worst;
}

struct owned_codec {
    codec_model model;
    ~owned_codec() { free_codec(model); }
};

}  // namespace

namespace {

constexpr int    BENCH_AUDIO_SECONDS[] = {10, 24};
constexpr int    BENCH_REPS = 3;
constexpr int    N_THREADS  = 4;
constexpr int    GPU_LAYERS = 99;
constexpr double MIN_COSINE = 0.999;

struct leg {
    double synthesis_ms = 0.0;
    double total_ms     = 0.0;
    std::string backend;
    std::vector<float> pcm;
};

int frames_for(const codec_hparams & hp, int seconds) {
    return std::min(hp.max_frames, (seconds * hp.sample_rate + hp.frame_size - 1) / hp.frame_size);
}

bool time_leg(codec_model & model, const std::vector<int32_t> & codes, int n_frames, leg & out) {
    std::vector<double> synth, total;
    for (int rep = 0; rep < BENCH_REPS; ++rep) {
        std::vector<float> pcm;
        decode_timing timing;
        std::string error;
        const auto t0 = std::chrono::steady_clock::now();
        if (!decode_codes(model, codes.data(), n_frames, N_THREADS, nullptr, pcm, &error, nullptr,
                          &timing)) {
            std::fprintf(stderr, "[bench-audio8-coreml] decode failed: %s\n", error.c_str());
            return false;
        }
        const auto t1 = std::chrono::steady_clock::now();
        synth.push_back(timing.synthesis_ms);
        total.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        out.backend = timing.synthesis_backend;
        out.pcm = std::move(pcm);
    }
    std::sort(synth.begin(), synth.end());
    std::sort(total.begin(), total.end());
    out.synthesis_ms = synth[synth.size() / 2];
    out.total_ms = total[total.size() / 2];
    return true;
}

struct bench_row {
    int    audio_seconds;
    int    n_frames;
    leg    ggml;
    leg    coreml;
    double cos;
    std::string ggml_device;  // the ggml backend instance, e.g. "MTL0"
};

// false with *gpu_missing set when the ggml reference would be the CPU.
bool bench_one(const std::string & gguf, int audio_seconds, bench_row * row, bool * gpu_missing) {
    setenv("AUDIO8_COREML_DISABLE", "1", 1);
    owned_codec ggml;
    std::string error;
    const bool loaded = load_codec(gguf, GPU_LAYERS, /*backend=*/"", ggml.model, &error);
    unsetenv("AUDIO8_COREML_DISABLE");
    if (!loaded) {
        std::fprintf(stderr, "[bench-audio8-coreml] load: %s\n", error.c_str());
        return false;
    }
    if (tts_cpp::detail::backend_is_cpu(ggml.model.backend)) {
        *gpu_missing = true;
        return false;
    }
    const int n_frames = frames_for(ggml.model.hp, audio_seconds);
    const std::vector<int32_t> codes = make_codes(ggml.model.hp, n_frames, 0x2468ace1u);
    if (!time_leg(ggml.model, codes, n_frames, row->ggml)) return false;

    setenv("AUDIO8_COREML_STRICT", "1", 1);
    owned_codec coreml;
    bool ok = load_codec(gguf, GPU_LAYERS, /*backend=*/"", coreml.model, &error);
    if (ok) {
        std::vector<float> warmup;
        ok = decode_codes(coreml.model, codes.data(), n_frames, N_THREADS, nullptr, warmup, &error);
        if (!ok) std::fprintf(stderr, "[bench-audio8-coreml] warm-up: %s\n", error.c_str());
    } else {
        std::fprintf(stderr, "[bench-audio8-coreml] load: %s\n", error.c_str());
    }
    if (ok) ok = time_leg(coreml.model, codes, n_frames, row->coreml);
    unsetenv("AUDIO8_COREML_STRICT");
    if (!ok) return false;

    if (row->ggml.pcm.size() != row->coreml.pcm.size()) {
        std::fprintf(stderr, "[bench-audio8-coreml] FAIL: sizes ggml=%zu coreml=%zu\n",
                     row->ggml.pcm.size(), row->coreml.pcm.size());
        return false;
    }
    row->audio_seconds = audio_seconds;
    row->n_frames = n_frames;
    row->cos = cosine(row->ggml.pcm, row->coreml.pcm);
    row->ggml_device = ggml_backend_name(ggml.model.backend);
    std::fprintf(stderr,
                 "[bench-audio8-coreml] frames=%d synthesis ggml(%s)=%.0f ms coreml(%s)=%.0f ms; "
                 "whole decode %.0f vs %.0f ms (median of %d) cosine=%.7f\n",
                 n_frames, ggml_backend_name(ggml.model.backend), row->ggml.synthesis_ms,
                 row->coreml.backend.c_str(), row->coreml.synthesis_ms, row->ggml.total_ms,
                 row->coreml.total_ms, BENCH_REPS, row->cos);
    return row->cos >= MIN_COSINE;
}

void print_markdown(const std::vector<bench_row> & rows, const std::string & ggml_backend) {
    std::printf("### Audio8 codec synthesis: ggml %s vs Core ML sidecar\n\n", ggml_backend.c_str());
    std::printf("| frames | audio | ggml synthesis | Core ML synthesis | speedup | whole decode ggml / Core ML | cosine |\n");
    std::printf("|---|---|---|---|---|---|---|\n");
    for (const bench_row & r : rows) {
        std::printf("| %d | %d s | %.0f ms | %.0f ms (%s) | %.2fx | %.0f / %.0f ms | %.5f |\n",
                    r.n_frames, r.audio_seconds, r.ggml.synthesis_ms, r.coreml.synthesis_ms,
                    r.coreml.backend.c_str(), r.ggml.synthesis_ms / r.coreml.synthesis_ms,
                    r.ggml.total_ms, r.coreml.total_ms, r.cos);
    }
}

}  // namespace

int main() {
    const char * models_dir = std::getenv("AUDIO8_COREML_TEST_MODELS_DIR");
    if (models_dir == nullptr || *models_dir == '\0') {
        std::fprintf(stderr, "[bench-audio8-coreml] AUDIO8_COREML_TEST_MODELS_DIR not set; skipping\n");
        return SKIP_EXIT_CODE;
    }
    const std::string gguf = find_decoder_gguf(models_dir);
    if (gguf.empty()) {
        std::fprintf(stderr, "[bench-audio8-coreml] no codec decoder GGUF under %s; skipping\n", models_dir);
        return SKIP_EXIT_CODE;
    }
    const std::string sidecar = coreml_codec_sidecar_path(gguf);
    if (!path_exists(sidecar)) {
        std::fprintf(stderr, "[bench-audio8-coreml] no sidecar at %s; skipping\n", sidecar.c_str());
        return SKIP_EXIT_CODE;
    }

    std::vector<bench_row> rows;
    std::string ggml_backend = "GPU";
    for (int audio_seconds : BENCH_AUDIO_SECONDS) {
        bench_row row{};
        bool gpu_missing = false;
        if (!bench_one(gguf, audio_seconds, &row, &gpu_missing)) {
            if (gpu_missing) {
                std::fprintf(stderr, "[bench-audio8-coreml] no GPU ggml backend resolved; the reference "
                                     "would be CPU, skipping\n");
                return SKIP_EXIT_CODE;
            }
            return 1;
        }
        ggml_backend = row.ggml_device;
        rows.push_back(std::move(row));
    }
    print_markdown(rows, ggml_backend);
    return 0;
}

#endif  // TTS_CPP_USE_COREML
