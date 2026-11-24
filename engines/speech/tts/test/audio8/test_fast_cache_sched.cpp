// Equivalence and cache-integrity for the fast-AR graph cache against the
// dual-path dispatch in src/sched_dispatch.{h,cpp}.
//
// fast_pass keeps one built graph per position and replays it, which is only
// sound on the direct path: ggml_backend_sched_alloc_graph resets one shared
// arena and rewrites node->src[] in place, so a graph handed to the scheduler
// cannot be kept, and a kept graph cannot be handed to the scheduler. The same
// hazard is spelled out for T3 in test_t3_sched_equivalence.cpp.
//
// Two runs of the same model, because the branch that matters is only
// reachable when the scheduler is chosen while caching is still available.
//
//   forced first  S   TTS_CPP_FORCE_SCHED=1 before any frame, so the build
//                     itself goes through prepare_graph, comes back scheduler-
//                     backed, and takes the drop. Asserted directly: the entry
//                     is released and caching is off for the model's life.
//                 S'  a second frame with the hook cleared, which must stay on
//                     the per-call path and still match.
//
//   replayed      R   a second model with the hook clear, whose positions are
//                     all replayed, and whose logits must match S exactly
//
// Logits are compared byte for byte. Bit-exactness is the bar; do not relax it
// to a tolerance, because every phase runs the same graph on the same backend.

#include "audio8/internal.h"
#include "audio8/sampling.h"
#include "gpu_arm.h"
#include "test_env_portable.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace tts_cpp::audio8::detail;

namespace {

constexpr int GPU_LAYERS = 99;
constexpr int THREADS = 2;

int failures = 0;

void fail(const std::string & what) {
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
}

int requested_layers() {
    return audio8_test::is_gpu_test() ? GPU_LAYERS : 0;
}

// Every position's logits, in order, as the frame produced them.
using logit_trace = std::vector<std::vector<float>>;

// The picker records what it was handed and always answers 0, so the frame
// walks the same positions with the same inputs in every phase.
code_picker recording_picker(logit_trace & trace) {
    return [&trace](const std::vector<float> & logits, int) {
        trace.push_back(logits);
        return 0;
    };
}

bool run_frame(lm_model & model, logit_trace & trace) {
    std::vector<int32_t> codes;
    std::vector<float> prime(static_cast<size_t>(model.hp.fast_hidden), 0.0f);
    std::string error;
    if (!fast_step(model, prime, model.hp.semantic_begin, THREADS,
                   recording_picker(trace), codes, &error)) {
        fail("fast_step failed: " + error);
        return false;
    }
    return true;
}

bool same_trace(const logit_trace & left, const logit_trace & right) {
    if (left.size() != right.size()) return false;
    for (size_t index = 0; index < left.size(); ++index) {
        if (left[index].size() != right[index].size()) return false;
        if (std::memcmp(left[index].data(), right[index].data(),
                        left[index].size() * sizeof(float)) != 0) {
            return false;
        }
    }
    return true;
}

void expect_same(const logit_trace & got, const logit_trace & want, const char * what) {
    if (same_trace(got, want)) return;
    fail(std::string(what) + ": fast-AR logits differ");
}

bool nothing_retained(const lm_model & model) {
    for (const lm_model::fast_graph & cached : model.fast_graphs) {
        if (cached.ctx || cached.graph || cached.allocr) return false;
    }
    return true;
}

// The scheduler is selected inside prepare_graph, so the hook has to be on
// before the first frame for the build to meet it while caching is still live.
void check_dropped_on_first_scheduler_build(lm_model & model, logit_trace & forced) {
    setenv("TTS_CPP_FORCE_SCHED", "1", 1);
    const bool ok = run_frame(model, forced);
    unsetenv("TTS_CPP_FORCE_SCHED");
    if (!ok) return;
    if (!model.fast_cache_off) {
        fail("a scheduler-backed build left caching enabled");
    }
    if (!nothing_retained(model)) {
        fail("a scheduler-backed build kept a graph, context or allocator");
    }
}

void check_replayed_matches(lm_model & model, const logit_trace & scheduled) {
    logit_trace replayed, again;
    unsetenv("TTS_CPP_FORCE_SCHED");
    if (!run_frame(model, replayed)) return;
    if (model.fast_cache_off) fail("a supported backend disabled the cache");
    if (nothing_retained(model)) fail("a replayed frame kept no graph");
    expect_same(replayed, scheduled, "replayed against scheduler-backed");
    if (!run_frame(model, again)) return;
    expect_same(again, replayed, "a second replayed frame");
    if (replayed.empty()) fail("the frame produced no fast-AR positions");
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <lm.gguf>\n", argv[0]);
        return 1;
    }

    lm_model model;
    std::string error;
    if (!load_lm(argv[1], requested_layers(), /*backend=*/"", model, &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    if (!audio8_test::check_requested_gpu("fast cache sched", model.backend)) {
        free_lm(model);
        return 1;
    }

    logit_trace forced_first, after_forced_first;
    check_dropped_on_first_scheduler_build(model, forced_first);
    if (!forced_first.empty()) {
        run_frame(model, after_forced_first);
        expect_same(after_forced_first, forced_first, "per-call path after the drop");
    }
    free_lm(model);

    lm_model replayed;
    if (!load_lm(argv[1], requested_layers(), /*backend=*/"", replayed, &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    check_replayed_matches(replayed, forced_first);

    std::printf("backend: %s, %zu positions compared\n",
                ggml_backend_name(replayed.backend), forced_first.size());
    free_lm(replayed);

    if (failures == 0) {
        std::fprintf(stderr, "audio8 fast cache sched: PASS\n");
        return 0;
    }
    std::fprintf(stderr, "audio8 fast cache sched: %d failure(s)\n", failures);
    return 1;
}
