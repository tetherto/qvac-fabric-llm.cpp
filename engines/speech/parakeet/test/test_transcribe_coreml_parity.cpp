// End-to-end transcription parity: Engine::transcribe() driven by the Core ML
// (Apple Neural Engine) encoder must produce the same batch and Mode-2 streaming
// results as the ggml encoder, including tokens, segment boundaries, and EOU
// events.
//
// Each transcription runs in its own forked child so that every Engine is the
// only one in its process (a single process that constructs two GPU-backed
// Engines in sequence is not supported). Child A loads normally (the Core ML
// sidecar drives the encoder when present); child B loads with
// PARAKEET_COREML_DISABLE set (forcing the ggml encoder). The parent compares
// the two transcripts.
//
// The Core ML child waits for every sidecar at construction (sidecars load in
// the background otherwise, and the call would run on ggml). Skips (exit 0)
// only when no sidecar is planned (non-Apple build, PARAKEET_COREML off, or no
// `<model>-encoder.mlmodelc`); a planned sidecar that is not active fails.

#include "coreml_test_options.h"
#include "parakeet/engine.h"

#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <string>
#include <sstream>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

#ifndef _WIN32
namespace {

void append_tokens(std::ostringstream & out, const std::vector<int32_t> & tokens) {
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) out << ',';
        out << tokens[i];
    }
}

std::string transcribe_in_child(const std::string & gguf, const std::string & wav,
                                const std::string & language,
                                bool disable, bool streaming) {
    int fds[2];
    if (pipe(fds) != 0) {
        return std::string();
    }
    const pid_t pid = fork();
    if (pid == 0) {
        close(fds[0]);
        if (disable) {
            setenv("PARAKEET_COREML_DISABLE", "1", 1);
        }
        parakeet::EngineOptions opts;
        opts.model_gguf_path = gguf;
        opts.n_gpu_layers    = 999;
        opts.language        = language;
        if (!disable) opts.coreml = coreml_ready_options();
        parakeet::Engine engine(opts);
        std::vector<parakeet::StreamingSegment> segments;
        std::vector<parakeet::StreamEvent> events;
        parakeet::EngineResult result;
        if (streaming) {
            parakeet::StreamingOptions stream_opts;
            // Keep the fixture utterance in one decoder window. Approximate
            // Core ML/ggml encoder values can move a token across an arbitrary
            // short chunk seam even when the complete token stream is equal;
            // this still exercises and compares the streaming callback's
            // utterance boundary and EOU event.
            stream_opts.chunk_ms = 60000;
            stream_opts.on_event = [&](const parakeet::StreamEvent & event) {
                events.push_back(event);
            };
            result = engine.transcribe_stream(
                wav, stream_opts,
                [&](const parakeet::StreamingSegment & segment) {
                    segments.push_back(segment);
                });
        } else {
            result = engine.transcribe(wav);
        }

        std::ostringstream body;
        body << std::setprecision(17);
        body << "tokens=";
        append_tokens(body, result.token_ids);
        body << "\ntext=" << result.text << "\nsegments=" << segments.size();
        for (const auto & segment : segments) {
            body << "\nsegment=" << segment.start_s << ',' << segment.end_s << ','
                 << segment.chunk_index << ',' << segment.is_final << ','
                 << segment.starts_word << ',' << segment.is_eou_boundary << ','
                 << segment.eot_confidence << ',';
            append_tokens(body, segment.token_ids);
            body << ',' << segment.text;
        }
        body << "\nevents=" << events.size();
        for (const auto & event : events) {
            body << "\nevent=" << static_cast<int>(event.type) << ','
                 << event.timestamp_s << ',' << event.chunk_index << ','
                 << event.eot_confidence;
        }
        const std::string payload = std::string("coreml_available=") +
                                    (engine.encoder_on_coreml() ? "1" : "0") +
                                    "\ncoreml_planned=" +
                                    (coreml_sidecar_planned(engine.coreml_status_json()) ? "1" : "0") +
                                    "\ncoreml_used=" +
                                    (result.encoder_used_coreml ? "1" : "0") +
                                    "\n" + body.str();
        for (size_t off = 0; off < payload.size();) {
            const ssize_t n = write(fds[1], payload.data() + off, payload.size() - off);
            if (n <= 0) break;
            off += static_cast<size_t>(n);
        }
        close(fds[1]);
        _exit(0);
    }
    close(fds[1]);
    std::string out;
    char chunk[4096];
    ssize_t n;
    while ((n = read(fds[0], chunk, sizeof(chunk))) > 0) {
        out.append(chunk, static_cast<size_t>(n));
    }
    close(fds[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    return out;
}

// Reads one `key=0|1` header line at `pos` and advances past it.
bool parse_flag(const std::string & raw, size_t & pos, const char * key, bool & value) {
    const std::string prefix = std::string(key) + '=';
    if (raw.compare(pos, prefix.size(), prefix) != 0) return false;
    const size_t nl = raw.find('\n', pos);
    if (nl == std::string::npos) return false;
    value = raw.substr(pos + prefix.size(), nl - pos - prefix.size()) == "1";
    pos = nl + 1;
    return true;
}

bool parse_child(const std::string & raw, bool & coreml_available, bool & coreml_planned,
                 bool & coreml_used, std::string & body) {
    size_t pos = 0;
    if (!parse_flag(raw, pos, "coreml_available", coreml_available) ||
        !parse_flag(raw, pos, "coreml_planned", coreml_planned) ||
        !parse_flag(raw, pos, "coreml_used", coreml_used)) {
        return false;
    }
    body = raw.substr(pos);
    return true;
}

bool compare_mode(const std::string & gguf, const std::string & wav,
                  const std::string & language, bool streaming) {
    const char * mode = streaming ? "streaming" : "batch";
    bool coreml_available = false;
    bool coreml_planned = false;
    bool coreml_used = false;
    std::string coreml_body;
    if (!parse_child(transcribe_in_child(
                         gguf, wav, language, /*disable=*/false, streaming),
                     coreml_available, coreml_planned, coreml_used, coreml_body)) {
        std::fprintf(stderr, "[transcribe-coreml-parity] FAIL: no output from Core ML %s child\n", mode);
        return false;
    }
    if (!coreml_available && coreml_planned) {
        std::fprintf(stderr,
            "[transcribe-coreml-parity] FAIL: a Core ML sidecar is planned but was not "
            "active after waiting at construction (%s)\n", mode);
        return false;
    }
    if (!coreml_available) {
        std::fprintf(stderr,
            "[transcribe-coreml-parity] SKIP: Core ML encoder not active "
            "(non-Apple build, PARAKEET_COREML off, or no sidecar).\n");
        return true;
    }
    if (!coreml_used) {
        std::fprintf(stderr,
            "[transcribe-coreml-parity] FAIL: Core ML sidecar was available but "
            "the %s transcription fell back to ggml\n", mode);
        return false;
    }

    bool ggml_available = true;
    bool ggml_planned = true;
    bool ggml_used = true;
    std::string ggml_body;
    if (!parse_child(transcribe_in_child(
                         gguf, wav, language, /*disable=*/true, streaming),
                     ggml_available, ggml_planned, ggml_used, ggml_body)) {
        std::fprintf(stderr, "[transcribe-coreml-parity] FAIL: no output from ggml %s child\n", mode);
        return false;
    }
    if (ggml_available || ggml_planned || ggml_used) {
        std::fprintf(stderr, "[transcribe-coreml-parity] FAIL: PARAKEET_COREML_DISABLE ignored\n");
        return false;
    }
    if (coreml_body != ggml_body) {
        std::fprintf(stderr,
            "[transcribe-coreml-parity] FAIL: %s text, tokens, segments, or EOU events differ\n",
            mode);
        std::fprintf(stderr, "[transcribe-coreml-parity] Core ML:\n%s\n", coreml_body.c_str());
        std::fprintf(stderr, "[transcribe-coreml-parity] ggml:\n%s\n", ggml_body.c_str());
        return false;
    }
    std::fprintf(stderr, "[transcribe-coreml-parity] PASS (%s parity)\n", mode);
    return true;
}

}  // namespace
#endif

int main(int argc, char ** argv) {
#ifdef _WIN32
    (void) argc;
    (void) argv;
    std::fprintf(stderr, "[transcribe-coreml-parity] SKIP: Core ML is Apple-only.\n");
    return 0;
#else
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <gguf> <wav> [language]\n", argv[0]);
        return 2;
    }
    const std::string gguf = argv[1];
    const std::string wav  = argv[2];
    const std::string language = argc >= 4 ? argv[3] : "";

    if (!compare_mode(gguf, wav, language, /*streaming=*/false) ||
        !compare_mode(gguf, wav, language, /*streaming=*/true)) {
        return 1;
    }
    return 0;
#endif
}
