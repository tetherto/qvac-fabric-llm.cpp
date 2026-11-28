// Model-free: the sidecar's window plan (coreml_windows.h) and naming rule
// (coreml_path.h) on every platform.

#include "audio8/coreml_path.h"
#include "audio8/coreml_windows.h"

#include <cstdio>
#include <string>
#include <vector>

using tts_cpp::audio8::detail::coreml_codec_sidecar_path;
using tts_cpp::audio8::detail::coreml_window;
using tts_cpp::audio8::detail::plan_coreml_windows;

namespace {

int g_failures = 0;

void fail(const std::string & what) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
}

void expect(bool ok, const std::string & what) {
    if (!ok) fail(what);
}

std::string tag(int n_frames, int window, int context) {
    return "n=" + std::to_string(n_frames) + " window=" + std::to_string(window) +
           " context=" + std::to_string(context);
}

void check_plan(int n_frames, int window, int context) {
    const std::vector<coreml_window> plan = plan_coreml_windows(n_frames, window, context);
    const std::string where = tag(n_frames, window, context);
    expect(!plan.empty(), where + ": empty plan");
    if (plan.empty()) return;

    int next_core = 0;
    for (size_t i = 0; i < plan.size(); ++i) {
        const coreml_window & w = plan[i];
        const std::string at = where + " window " + std::to_string(i);
        expect(w.begin >= 0, at + ": begins before the utterance");
        expect(w.filled > 0 && w.filled <= window, at + ": filled outside (0, window]");
        expect(w.begin + w.filled <= n_frames, at + ": reads past the utterance");
        expect(w.core_begin == next_core, at + ": core does not continue the previous one");
        expect(w.core_end > w.core_begin, at + ": empty core");
        expect(w.core_begin >= w.begin, at + ": core starts before its window");
        expect(w.core_end <= w.begin + window, at + ": core ends past its window");
        if (i > 0) {
            expect(w.core_begin - w.begin >= context,
                   at + ": core starts inside the causal context");
        } else {
            expect(w.begin == 0 && w.core_begin == 0, at + ": first window not at zero");
        }
        if (n_frames > window) {
            expect(w.filled == window, at + ": padded although the utterance is longer");
        }
        next_core = w.core_end;
    }
    expect(next_core == n_frames, where + ": cores do not reach the end");
}

void check_short_is_padded() {
    const std::vector<coreml_window> plan = plan_coreml_windows(10, 64, 12);
    expect(plan.size() == 1, "short: expected a single window");
    if (plan.empty()) return;
    expect(plan[0].begin == 0 && plan[0].filled == 10 && plan[0].core_begin == 0 &&
               plan[0].core_end == 10,
           "short: expected {0, 10, 0, 10}");
}

void check_exact_fit() {
    const std::vector<coreml_window> plan = plan_coreml_windows(64, 64, 12);
    expect(plan.size() == 1 && plan[0].filled == 64 && plan[0].core_end == 64,
           "exact fit: expected one full window");
}

void check_window_count() {
    // Each window after the first advances by window - context frames until
    // the end-aligned last one: 64 + 3 * 52 = 220 covers 220 frames in four.
    expect(plan_coreml_windows(220, 64, 12).size() == 4, "220 frames: expected 4 windows");
    expect(plan_coreml_windows(221, 64, 12).size() == 5, "221 frames: expected 5 windows");
}

void check_rejections() {
    expect(plan_coreml_windows(0, 64, 12).empty(), "no frames must plan nothing");
    expect(plan_coreml_windows(100, 0, 0).empty(), "a zero window must plan nothing");
    expect(plan_coreml_windows(100, 12, 12).empty(),
           "a window no wider than the context must plan nothing");
    expect(plan_coreml_windows(100, 64, -1).empty(), "a negative context must plan nothing");
}

void check_path(const std::string & gguf, const std::string & want) {
    const std::string got = coreml_codec_sidecar_path(gguf);
    std::printf("[%s] %-46s -> %s\n", got == want ? "ok  " : "FAIL", gguf.c_str(), got.c_str());
    if (got != want) fail("expected " + want);
}

}  // namespace

int main() {
    const int contexts[] = {0, 1, 11, 12, 30, 63};
    const int windows[] = {16, 32, 64, 128};
    for (int window : windows) {
        for (int context : contexts) {
            if (context >= window) continue;
            for (int n_frames = 1; n_frames <= 3 * window + 5; ++n_frames) {
                check_plan(n_frames, window, context);
            }
            check_plan(512, window, context);
            check_plan(4096, window, context);
        }
    }
    check_short_is_padded();
    check_exact_fit();
    check_window_count();
    check_rejections();

    check_path("models/audio8-codec-decoder-q8_0.gguf", "models/audio8-codec-decoder.mlmodelc");
    check_path("models/audio8-codec-decoder-f32.gguf",  "models/audio8-codec-decoder.mlmodelc");
    check_path("models/audio8-codec-decoder-f16.gguf",  "models/audio8-codec-decoder.mlmodelc");
    check_path("models/audio8-codec-decoder.gguf",      "models/audio8-codec-decoder.mlmodelc");
    check_path("audio8-codec-decoder.Q8_0.gguf",        "audio8-codec-decoder.mlmodelc");
    check_path("/opt/m.odels/decoder.gguf",             "/opt/m.odels/decoder.mlmodelc");
    check_path("decoder",                               "decoder.mlmodelc");

    if (g_failures == 0) {
        std::printf("[coreml-windows] PASS\n");
        return 0;
    }
    std::printf("[coreml-windows] FAIL: %d case(s)\n", g_failures);
    return 1;
}
