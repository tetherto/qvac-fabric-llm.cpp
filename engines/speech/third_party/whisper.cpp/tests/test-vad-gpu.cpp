// Regression coverage for VAD with use_gpu enabled.
//
// whisper_vad_init_with_params places the model weights via a buffer-type
// list that honors params.use_gpu, but whisper_vad_init_context used to
// force its compute backends to CPU. On any GPU build a use_gpu=true init
// therefore aborted inside ggml_backend_sched ("pre-allocated tensor in a
// buffer that cannot run the operation") before returning a context, and
// after that was fixed the LSTM matmul aborted in the CUDA mul-mat-vec
// kernel on the odd-stride transposed input (upstream issue
// ggml-org/whisper.cpp#3508). This test locks the contract:
//
//   1. Initializing a VAD context with use_gpu=true succeeds.
//   2. Detection with use_gpu=true completes and yields the same number
//      of per-chunk probabilities as the CPU run.
//   3. The GPU probabilities match the CPU probabilities within a small
//      tolerance, and both runs agree on the segment count.
//
// On machines without a GPU (or CPU-only builds), use_gpu=true falls back
// to the CPU backend, so the comparison degrades to CPU-vs-CPU and the
// test still guards the use_gpu=true init path.

#include "whisper.h"
#include "common-whisper.h"

#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

// Measured CPU-vs-CUDA drift on the jfk.wav fixture peaks at ~6e-3: the
// f16 weights and the 344-step LSTM recurrence accumulate small per-op
// backend differences. 1e-2 keeps headroom without masking real breakage
// (a wrong buffer or stride shows up as diffs of 0.1+).
static const float kProbTolerance = 1e-2f;

static std::vector<float> snapshot_probs(struct whisper_vad_context * vctx) {
    const int n = whisper_vad_n_probs(vctx);
    const float * p = whisper_vad_probs(vctx);
    return std::vector<float>(p, p + n);
}

static void assert_probs_near(const std::vector<float> & a,
                              const std::vector<float> & b,
                              float tol,
                              const char * label) {
    assert(a.size() == b.size());
    float worst = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        const float d = std::fabs(a[i] - b[i]);
        if (d > worst) worst = d;
    }
    printf("%s: max |diff| = %.6f over %zu probs (tol = %.6f)\n", label, worst, a.size(), tol);
    assert(worst <= tol);
}

static struct whisper_vad_context * init_vad_context(const std::string & model_path, bool use_gpu) {
    struct whisper_vad_context_params ctx_params = whisper_vad_default_context_params();
    constexpr int kVadTestThreads = 1;
    ctx_params.n_threads = kVadTestThreads;
    ctx_params.use_gpu = use_gpu;
    return whisper_vad_init_from_file_with_params(model_path.c_str(), ctx_params);
}

static std::vector<float> run_detection(struct whisper_vad_context * vctx,
                                        const std::vector<float> & pcmf32) {
    assert(whisper_vad_detect_speech(vctx, pcmf32.data(), (int)pcmf32.size()));
    return snapshot_probs(vctx);
}

static int count_segments(struct whisper_vad_context * vctx) {
    struct whisper_vad_params params = whisper_vad_default_params();
    struct whisper_vad_segments * segments = whisper_vad_segments_from_probs(vctx, params);
    assert(segments != nullptr);
    const int n = whisper_vad_segments_n_segments(segments);
    whisper_vad_free_segments(segments);
    return n;
}

int main() {
    const std::string vad_model_path = VAD_MODEL_PATH;
    const std::string sample_path    = SAMPLE_PATH;

    std::vector<float> pcmf32;
    std::vector<std::vector<float>> pcmf32s;
    assert(read_audio_data(sample_path.c_str(), pcmf32, pcmf32s, false));
    assert(pcmf32.size() > 0);

    struct whisper_vad_context * vctx_cpu = init_vad_context(vad_model_path, false);
    assert(vctx_cpu != nullptr);
    const auto probs_cpu = run_detection(vctx_cpu, pcmf32);
    assert(probs_cpu.size() > 0);

    struct whisper_vad_context * vctx_gpu = init_vad_context(vad_model_path, true);
    assert(vctx_gpu != nullptr);
    const auto probs_gpu = run_detection(vctx_gpu, pcmf32);

    assert_probs_near(probs_cpu, probs_gpu, kProbTolerance, "use_gpu probs vs CPU");

    const int n_segments_cpu = count_segments(vctx_cpu);
    const int n_segments_gpu = count_segments(vctx_gpu);
    printf("segments: cpu = %d, gpu = %d\n", n_segments_cpu, n_segments_gpu);
    assert(n_segments_cpu == n_segments_gpu);

    whisper_vad_free(vctx_gpu);
    whisper_vad_free(vctx_cpu);
    return 0;
}
