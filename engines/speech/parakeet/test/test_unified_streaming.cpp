#include "parakeet/engine.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace parakeet;

namespace {

struct Opts {
    std::string model_path;
    std::string wav_path;
    int n_gpu_layers = 0;
    int n_threads = 0;
    bool verbose = false;
};

void print_usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model <parakeet-unified.gguf> --wav <input.wav> [--threads N] [--n-gpu-layers N] [--verbose]\n",
        argv0);
}

int parse_args(int argc, char ** argv, Opts & o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--model"        && i + 1 < argc) o.model_path = argv[++i];
        else if (a == "--wav"          && i + 1 < argc) o.wav_path = argv[++i];
        else if (a == "--n-gpu-layers" && i + 1 < argc) o.n_gpu_layers = std::atoi(argv[++i]);
        else if (a == "--threads"      && i + 1 < argc) o.n_threads = std::atoi(argv[++i]);
        else if (a == "--verbose" || a == "-v") o.verbose = true;
        else { print_usage(argv[0]); return 2; }
    }
    if (o.model_path.empty() || o.wav_path.empty()) {
        print_usage(argv[0]);
        return 2;
    }
    return 0;
}

bool load_wav_pcm(const std::string & path, std::vector<float> & out, int & sr) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char hdr[44];
    f.read(hdr, 44);
    if (!f || std::memcmp(hdr, "RIFF", 4) != 0 || std::memcmp(hdr + 8, "WAVE", 4) != 0) return false;
    sr = *reinterpret_cast<int32_t *>(hdr + 24);
    f.seekg(0, std::ios::end);
    const std::streamoff total = f.tellg();
    f.seekg(44, std::ios::beg);
    const size_t n_samples = static_cast<size_t>(total - 44) / sizeof(int16_t);
    std::vector<int16_t> pcm(n_samples);
    f.read(reinterpret_cast<char *>(pcm.data()), n_samples * sizeof(int16_t));
    if (!f) return false;
    out.resize(n_samples);
    constexpr float inv = 1.0f / 32768.0f;
    for (size_t i = 0; i < n_samples; ++i) out[i] = static_cast<float>(pcm[i]) * inv;
    return true;
}

StreamingOptions streaming_options(int sample_rate, int chunk_ms, int right_lookahead_ms) {
    StreamingOptions sopts;
    sopts.sample_rate = sample_rate;
    sopts.chunk_ms = chunk_ms;
    sopts.right_lookahead_ms = right_lookahead_ms;
    return sopts;
}

int report(const char * label, const std::string & actual, const std::string & expected, int chunk_ms, int right_ms) {
    if (actual == expected) {
        std::fprintf(stderr, "[test-unified-streaming] PASS %s chunk_ms=%d right_ms=%d: text byte-equal\n",
                     label, chunk_ms, right_ms);
        return 0;
    }
    std::fprintf(stderr, "[test-unified-streaming] FAIL %s chunk_ms=%d right_ms=%d\n  ref:  %s\n  got:  %s\n",
                 label, chunk_ms, right_ms, expected.c_str(), actual.c_str());
    return 1;
}

int check_mode2(Engine & engine, const Opts & opts, const EngineResult & ref, int chunk_ms, int right_ms) {
    std::string text;
    engine.transcribe_stream(opts.wav_path, streaming_options(ref.sample_rate, chunk_ms, right_ms),
        [&](const StreamingSegment & s) { text += s.text; });
    return report("Mode 2", text, ref.text, chunk_ms, right_ms);
}

void feed_in_blocks(StreamSession & session, const std::vector<float> & pcm, int block) {
    for (size_t offset = 0; offset < pcm.size(); offset += static_cast<size_t>(block)) {
        const int n = static_cast<int>(std::min<size_t>(block, pcm.size() - offset));
        session.feed_pcm_f32(pcm.data() + offset, n);
    }
}

int check_mode3(Engine & engine, const std::vector<float> & pcm, const EngineResult & ref, int chunk_ms, int right_ms) {
    std::string text;
    auto session = engine.stream_start(streaming_options(ref.sample_rate, chunk_ms, right_ms),
        [&](const StreamingSegment & s) { text += s.text; });
    feed_in_blocks(*session, pcm, 1024);
    session->finalize();
    return report("Mode 3", text, ref.text, chunk_ms, right_ms);
}

int check_lowest_latency_runs(Engine & engine, const std::vector<float> & pcm, const EngineResult & ref) {
    std::string text;
    auto session = engine.stream_start(streaming_options(ref.sample_rate, 80, 0),
        [&](const StreamingSegment & s) { text += s.text; });
    feed_in_blocks(*session, pcm, 1024);
    session->finalize();
    if (text.empty()) {
        std::fprintf(stderr, "[test-unified-streaming] FAIL chunk_ms=80 right_ms=0 produced no text\n");
        return 1;
    }
    std::fprintf(stderr, "[test-unified-streaming] PASS chunk_ms=80 right_ms=0 produced text: %s\n", text.c_str());
    return 0;
}

std::vector<int16_t> to_pcm_i16(const std::vector<float> & pcm) {
    std::vector<int16_t> out(pcm.size());
    for (size_t i = 0; i < pcm.size(); ++i) {
        const float scaled = std::max(-1.0f, std::min(1.0f, pcm[i])) * 32767.0f;
        out[i] = static_cast<int16_t>(scaled < 0.0f ? scaled - 0.5f : scaled + 0.5f);
    }
    return out;
}

void feed_i16_in_blocks(StreamSession & session, const std::vector<int16_t> & pcm, int block) {
    for (size_t offset = 0; offset < pcm.size(); offset += static_cast<size_t>(block)) {
        const int n = static_cast<int>(std::min<size_t>(block, pcm.size() - offset));
        session.feed_pcm_i16(pcm.data() + offset, n);
    }
}

int check_pcm_i16_matches_float(Engine & engine, const std::vector<float> & pcm, const EngineResult & ref) {
    std::string from_float;
    auto float_session = engine.stream_start(streaming_options(ref.sample_rate, 560, 560),
        [&](const StreamingSegment & s) { from_float += s.text; });
    feed_in_blocks(*float_session, pcm, 1024);
    float_session->finalize();

    std::string from_i16;
    auto i16_session = engine.stream_start(streaming_options(ref.sample_rate, 560, 560),
        [&](const StreamingSegment & s) { from_i16 += s.text; });
    feed_i16_in_blocks(*i16_session, to_pcm_i16(pcm), 1024);
    i16_session->finalize();

    return report("Mode 3 (feed_pcm_i16)", from_i16, from_float, 560, 560);
}

int check_cancel_stops_emission(Engine & engine, const std::vector<float> & pcm, const EngineResult & ref) {
    int segments_before = 0;
    int segments_after = 0;
    bool cancelled = false;
    auto session = engine.stream_start(streaming_options(ref.sample_rate, 560, 560),
        [&](const StreamingSegment &) { (cancelled ? segments_after : segments_before) += 1; });
    feed_in_blocks(*session, pcm, 1024);
    session->cancel();
    cancelled = true;
    try {
        feed_in_blocks(*session, pcm, 1024);
        session->finalize();
    } catch (const std::runtime_error &) {
    }
    if (segments_after != 0) {
        std::fprintf(stderr,
                     "[test-unified-streaming] FAIL cancel emitted %d segments after cancelling\n",
                     segments_after);
        return 1;
    }
    std::fprintf(stderr,
                 "[test-unified-streaming] PASS cancel: %d segments before, none after\n",
                 segments_before);
    return 0;
}

int check_odd_length_tail(Engine & engine, const std::vector<float> & pcm, const EngineResult & ref) {
    const std::vector<float> trimmed(pcm.begin(), pcm.end() - 37);
    const EngineResult offline = engine.transcribe_samples(trimmed.data(), static_cast<int>(trimmed.size()), ref.sample_rate);
    std::string text;
    auto session = engine.stream_start(streaming_options(ref.sample_rate, 560, 560),
        [&](const StreamingSegment & s) { text += s.text; });
    feed_in_blocks(*session, trimmed, 1024);
    session->finalize();
    return report("Mode 3 (odd sample count)", text, offline.text, 560, 560);
}

int check_untrained_chunk_snaps(Engine & engine, const std::vector<float> & pcm, const EngineResult & ref) {
    std::string text;
    double last_end = 0.0;
    auto session = engine.stream_start(streaming_options(ref.sample_rate, 1000, 2000),
        [&](const StreamingSegment & s) { text += s.text; last_end = s.end_s; });
    feed_in_blocks(*session, pcm, 1024);
    session->finalize();
    if (int rc = report("Mode 3 (snapped from 1000/2000)", text, ref.text, 1000, 2000); rc != 0) return rc;
    if (last_end <= 0.0) {
        std::fprintf(stderr, "[test-unified-streaming] FAIL snapped session emitted no timed segments\n");
        return 1;
    }
    return 0;
}

int run(const Opts & opts) {
    EngineOptions eopts;
    eopts.model_gguf_path = opts.model_path;
    eopts.n_gpu_layers = opts.n_gpu_layers;
    eopts.n_threads = opts.n_threads;
    eopts.verbose = opts.verbose;
    Engine engine(eopts);
    if (engine.model_type() != "rnnt") {
        std::fprintf(stderr, "[test-unified-streaming] FAIL: GGUF is not an RNN-T model\n");
        return 3;
    }
    const EngineResult ref = engine.transcribe(opts.wav_path);
    if (ref.text.empty()) {
        std::fprintf(stderr, "[test-unified-streaming] FAIL: empty reference transcript\n");
        return 4;
    }
    std::vector<float> pcm;
    int sr = 0;
    if (!load_wav_pcm(opts.wav_path, pcm, sr) || sr != ref.sample_rate) {
        std::fprintf(stderr, "[test-unified-streaming] FAIL: cannot load %s\n", opts.wav_path.c_str());
        return 5;
    }
    int failures = 0;
    failures += check_mode2(engine, opts, ref, 560, 560);
    failures += check_mode3(engine, pcm, ref, 160, 320);
    failures += check_mode3(engine, pcm, ref, 1040, 1040);
    failures += check_lowest_latency_runs(engine, pcm, ref);
    failures += check_untrained_chunk_snaps(engine, pcm, ref);
    failures += check_odd_length_tail(engine, pcm, ref);
    failures += check_pcm_i16_matches_float(engine, pcm, ref);
    failures += check_cancel_stops_emission(engine, pcm, ref);
    return failures == 0 ? 0 : 1;
}

}

int main(int argc, char ** argv) {
    Opts opts;
    if (int rc = parse_args(argc, argv, opts); rc != 0) return rc;
    try {
        return run(opts);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "[test-unified-streaming] FAIL: %s\n", error.what());
        return 6;
    }
}
