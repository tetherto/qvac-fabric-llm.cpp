// Regression coverage for the streaming VAD API added by upstream
// ggml-org/whisper.cpp PR #3677 (commit 166c20b4) and pulled into the
// tetherto fork. The upstream change did not ship a test,
// so we add one here to lock the contract:
//
//   1. whisper_vad_detect_speech is idempotent (reset is implicit).
//   2. whisper_vad_reset_state restores the LSTM state, so a no-reset
//      run after a reset produces the same probs as the very first run.
//   3. Two contiguous whisper_vad_detect_speech_no_reset calls on
//      adjacent halves of the input produce the same per-chunk probs
//      as a single whisper_vad_detect_speech(full_input) — i.e. the
//      LSTM state actually carries across the boundary (within the
//      tolerance of running the same graph through two scheduler
//      activations).
//
// Split is performed at a multiple of vctx->n_window so that the second
// half starts cleanly on a chunk boundary and no zero-padding is
// introduced mid-stream that would diverge from the single-shot run.

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

constexpr int kSileroWindow = 512;

static std::vector<float> test_idempotent(whisper_vad_context * vctx, const std::vector<float> & pcmf32) {
    // Detection resets state implicitly.
    assert(whisper_vad_detect_speech(vctx, pcmf32.data(), (int)pcmf32.size()));
    const auto probs_first = snapshot_probs(vctx);

    assert(whisper_vad_detect_speech(vctx, pcmf32.data(), (int)pcmf32.size()));
    const auto probs_second = snapshot_probs(vctx);

    assert_probs_near(probs_first, probs_second, 0.0f,
                      "Test1 detect_speech idempotent");

    return probs_first;
}

static void test_reset_state(whisper_vad_context * vctx, const std::vector<float> & pcmf32,
                             const std::vector<float> & probs_first) {
    // Resetting restores the initial LSTM state.
    whisper_vad_reset_state(vctx);
    assert(whisper_vad_detect_speech_no_reset(vctx, pcmf32.data(), (int)pcmf32.size()));
    const auto probs_no_reset_a = snapshot_probs(vctx);

    whisper_vad_reset_state(vctx);
    assert(whisper_vad_detect_speech_no_reset(vctx, pcmf32.data(), (int)pcmf32.size()));
    const auto probs_no_reset_b = snapshot_probs(vctx);

    assert_probs_near(probs_no_reset_a, probs_no_reset_b, 0.0f,
                      "Test2 reset_state restores LSTM");

    // detect_speech (which resets internally) must also match the
    // reset+no_reset sequence — proves they share identical semantics
    // when starting from a clean state.
    assert_probs_near(probs_first, probs_no_reset_a, 0.0f,
                      "Test2b detect_speech == reset+no_reset");
}

static void test_chunked(whisper_vad_context * vctx, const std::vector<float> & pcmf32,
                         const std::vector<float> & probs_first) {
    // Split exactly on a chunk boundary so we don't introduce mid-stream
    // zero padding. The Silero v6.2.0 VAD model fixture uses a fixed
    // 512-sample window at 16 kHz (32 ms); the chunk count is
    // ceil(n_samples / 512), with the last chunk zero-padded if short.
    const int total_chunks   = (int)probs_first.size();
    assert(total_chunks == ((int)pcmf32.size() + kSileroWindow - 1) / kSileroWindow);
    const int half_chunks    = total_chunks / 2;
    assert(half_chunks >= 1 && total_chunks - half_chunks >= 1);
    const int split_idx      = half_chunks * kSileroWindow;
    assert(split_idx > 0 && split_idx < (int)pcmf32.size());

    whisper_vad_reset_state(vctx);
    assert(whisper_vad_detect_speech_no_reset(vctx, pcmf32.data(), split_idx));
    const auto probs_part1 = snapshot_probs(vctx);
    assert((int)probs_part1.size() == half_chunks);

    assert(whisper_vad_detect_speech_no_reset(
            vctx, pcmf32.data() + split_idx, (int)pcmf32.size() - split_idx));
    const auto probs_part2 = snapshot_probs(vctx);
    assert((int)probs_part2.size() == total_chunks - half_chunks);

    // Concatenate part1 + part2 and compare against single-shot probs.
    std::vector<float> probs_stitched;
    probs_stitched.reserve(total_chunks);
    probs_stitched.insert(probs_stitched.end(), probs_part1.begin(), probs_part1.end());
    probs_stitched.insert(probs_stitched.end(), probs_part2.begin(), probs_part2.end());

    // Float-equality is the contract: the per-chunk graph is the same
    // and the LSTM state is preserved exactly across the no_reset call
    // boundary. If a future refactor introduces tiny numerical drift
    // across scheduler resets, bump the tolerance — but never silently.
    assert_probs_near(probs_first, probs_stitched, 0.0f,
                      "Test3 streaming == single-shot");
}

static void test_streaming(const std::vector<float> & pcmf32, int n_threads) {
    const std::string vad_model_path = VAD_MODEL_PATH;
    struct whisper_vad_context_params ctx_params = whisper_vad_default_context_params();
    ctx_params.n_threads = n_threads;
    struct whisper_vad_context * vctx = whisper_vad_init_from_file_with_params(
            vad_model_path.c_str(), ctx_params);
    assert(vctx != nullptr);

    const auto probs_first = test_idempotent(vctx, pcmf32);
    test_reset_state(vctx, pcmf32, probs_first);
    test_chunked(vctx, pcmf32, probs_first);
    whisper_vad_free(vctx);
}

int main() {
    std::vector<float> pcmf32;
    std::vector<std::vector<float>> pcmf32s;
    assert(read_audio_data(SAMPLE_PATH, pcmf32, pcmf32s, false));
    assert(!pcmf32.empty());

    // Keep every chunk and reset assertion without per-node worker synchronization.
    constexpr int kSerialThreads = 1;
    test_streaming(pcmf32, kSerialThreads);

    // Retain the default multi-threaded contract on a short, partially padded input.
    constexpr int kThreadedChunks = 4;
    constexpr int kThreadedSamples = kThreadedChunks * kSileroWindow + 1;
    assert(pcmf32.size() >= kThreadedSamples);
    const std::vector<float> threaded_samples(pcmf32.begin(), pcmf32.begin() + kThreadedSamples);
    test_streaming(threaded_samples, whisper_vad_default_context_params().n_threads);
    return 0;
}
