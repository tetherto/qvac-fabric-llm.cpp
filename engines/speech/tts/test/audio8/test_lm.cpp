// Audio8 DualAR parity against the PyTorch fixtures.
//
// The reference dump records, per generated frame, the semantic logits over
// the 4096 codebook rows plus EOS, the hidden state handed to the fast head,
// the fast head's own logits and the ten codebook values. Replaying the
// fixture's prompt through the engine with teacher-forced reference tokens checks
// the whole loop: embedding gate, KV cache,
// baked RoPE tables and both heads.

#include "audio8/internal.h"
#include "gpu_arm.h"
#include "npy.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace tts_cpp::audio8::detail;

namespace {

// Bars. The engine and the reference differ only in float reassociation, so
// the logit gap stays far below the spacing between the top candidates; the
// token trace has to match exactly.
constexpr double LOGIT_TOLERANCE = 5e-3;
constexpr double HIDDEN_TOLERANCE = 2e-3;
constexpr double GPU_LOGIT_TOLERANCE = 1.5e-2;
// Worst observed on an RTX 3090 with the teacher-forced trace: 3.7e-3 on
// CUDA and 3.1e-3 on Vulkan at step 0, where the softmax over the widest
// QK dynamic range amplifies ordinary cross-backend rounding.
constexpr double GPU_HIDDEN_TOLERANCE = 1e-2;
constexpr int GPU_LAYERS = 99;

double logit_tolerance() {
    return audio8_test::is_gpu_test() ? GPU_LOGIT_TOLERANCE : LOGIT_TOLERANCE;
}

double hidden_tolerance() {
    return audio8_test::is_gpu_test() ? GPU_HIDDEN_TOLERANCE : HIDDEN_TOLERANCE;
}

struct fixture {
    std::string dir;

    npy_array load(const std::string & name) const {
        return npy_load(dir + "/" + name + ".npy");
    }
};

std::vector<int32_t> to_i32(const npy_array & array) {
    std::vector<int32_t> out(array.n_elements());
    if (array.dtype == "<i8") {
        const int64_t * source = reinterpret_cast<const int64_t *>(array.data.data());
        for (size_t index = 0; index < out.size(); ++index) {
            out[index] = static_cast<int32_t>(source[index]);
        }
        return out;
    }
    const int32_t * source = reinterpret_cast<const int32_t *>(array.data.data());
    out.assign(source, source + out.size());
    return out;
}

const float * as_f32(const npy_array & array) {
    return reinterpret_cast<const float *>(array.data.data());
}

// The fixture stores the prompt row-major as [rows, width]; the engine reads
// it column-major, one frame per column.
std::vector<int32_t> transpose_prompt(const npy_array & array) {
    const int rows = static_cast<int>(array.shape[0]);
    const int width = static_cast<int>(array.shape[1]);
    const std::vector<int32_t> source = to_i32(array);
    std::vector<int32_t> frames(source.size());
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < width; ++column) {
            frames[static_cast<size_t>(column) * rows + row] =
                source[static_cast<size_t>(row) * width + column];
        }
    }
    return frames;
}

bool check(const char * tag, const std::vector<float> & got, const float * want, size_t count,
           double tolerance) {
    compare_stats stats = compare_f32(got.data(), want, count);
    print_compare(tag, stats);
    if (!std::isfinite(stats.max_abs_err)) {
        std::fprintf(stderr, "%s: FAIL non-finite values\n", tag);
        return false;
    }
    if (stats.max_abs_err > tolerance) {
        std::fprintf(stderr, "%s: FAIL max|delta| %.3e > %.1e\n", tag, stats.max_abs_err,
                     tolerance);
        return false;
    }
    return true;
}

// Greedy decoding over the semantic head: the reference masks everything but
// the codebook rows and EOS before sampling, which the baked head already
// does, and neither top-k, top-p nor temperature can move the argmax.
int decode_semantic(const lm_hparams & hp, const std::vector<float> & logits) {
    const int index = argmax_of(logits);
    return index < hp.codebook_size ? hp.semantic_begin + index : hp.eos;
}

struct step_report {
    int frames = 0;
    int agreeing_frames = 0;
    bool ok = true;
};

// The rollout is teacher-forced: every step feeds the reference tokens
// forward whatever the model picked, so the per-step numeric checks stay on
// the reference trajectory for the whole trace instead of only up to the
// first differing argmax. Token choice is then a separate agreement rate: an
// argmax can legitimately flip where two candidates sit within the numeric
// tolerance of each other, so full token identity is not a valid cross-
// backend bar, but a backend that computes the right numbers still picks the
// reference token in the overwhelming majority of frames. The bars are the
// worst observed per backend with headroom: exact on the CPU, Vulkan measured
// at 100 percent and CUDA at 96.9 (31 of 32 frames).
constexpr double CPU_TOKEN_AGREEMENT = 1.0;
constexpr double GPU_TOKEN_AGREEMENT = 0.9;

double token_agreement_bar() {
    return audio8_test::is_gpu_test() ? GPU_TOKEN_AGREEMENT : CPU_TOKEN_AGREEMENT;
}

// fast_frame claims to be fast_step's greedy equivalent in one graph. Both are
// already pinned to the reference codes elsewhere, but only comparing them
// directly states the claim, and it costs one extra frame to do it. Each frame
// rewrites the fast cache from position 0, so re-running the frame is safe.
bool check_chained_frame(lm_model & model, const std::vector<float> & carried, int semantic,
                         int n_threads, const std::vector<int32_t> & want) {
    if (!model.picks_codes) {
        std::printf("  [chained frame] skipped: %s does not pick codes\n",
                    ggml_backend_name(model.backend));
        return true;
    }
    std::vector<int32_t> codes;
    std::string error;
    if (!fast_frame(model, carried, semantic, n_threads, codes, &error)) {
        std::fprintf(stderr, "chained frame: FAIL %s\n", error.c_str());
        return false;
    }
    if (codes == want) {
        std::printf("  [chained frame] %zu codes match the per-position path\n", codes.size());
        return true;
    }
    std::fprintf(stderr, "chained frame: FAIL codes differ from the per-position path\n");
    return false;
}

bool run_steps(lm_model & model, const fixture & data, int n_threads, step_report & report) {
    const lm_hparams & hp = model.hp;
    const npy_array prompt = data.load("prompt");
    const npy_array hidden_ref = data.load("slow_hidden");
    const npy_array logits_ref = data.load("sem_logits");
    const npy_array semantic_ref = data.load("semantic");
    const npy_array fast_ref = data.load("fast_logits");
    const npy_array codes_ref = data.load("codes");

    const std::vector<int32_t> frames = transpose_prompt(prompt);
    const std::vector<int32_t> want_semantic = to_i32(semantic_ref);
    const std::vector<int32_t> want_codes = to_i32(codes_ref);
    const int steps = static_cast<int>(semantic_ref.shape[0]);
    const int rows = static_cast<int>(prompt.shape[0]);
    const int width = static_cast<int>(prompt.shape[1]);

    std::vector<float> logits;
    std::vector<float> carried;
    std::string error;
    if (!slow_step(model, frames.data(), width, 0, n_threads, logits, carried, &error)) {
        std::fprintf(stderr, "prefill: %s\n", error.c_str());
        return false;
    }

    std::vector<int32_t> codes;
    std::vector<float> fast_logits;
    for (int step = 0; step < steps; ++step) {
        char tag[64];
        std::snprintf(tag, sizeof(tag), "step %2d hidden", step);
        report.ok &= check(tag, carried, as_f32(hidden_ref) + static_cast<size_t>(step) * hp.hidden,
                           hp.hidden, hidden_tolerance());
        std::snprintf(tag, sizeof(tag), "step %2d sem_logits", step);
        report.ok &= check(tag, logits,
                           as_f32(logits_ref) + static_cast<size_t>(step) * logits.size(),
                           logits.size(), logit_tolerance());

        const int semantic = decode_semantic(hp, logits);
        bool frame_agrees = semantic == want_semantic[step];
        if (!frame_agrees) {
            std::fprintf(stderr, "step %d: semantic %d != %d\n", step, semantic,
                         want_semantic[step]);
        }
        const code_picker greedy = [&](const std::vector<float> & values, int position) {
            if (position == 1) fast_logits = values;
            return argmax_of(values);
        };
        if (!fast_step(model, carried, want_semantic[step], n_threads, greedy, codes, &error)) {
            std::fprintf(stderr, "step %d fast: %s\n", step, error.c_str());
            return false;
        }
        std::snprintf(tag, sizeof(tag), "step %2d fast_logits", step);
        report.ok &= check(tag, fast_logits,
                           as_f32(fast_ref) + static_cast<size_t>(step) * 9 * hp.codebook_size,
                           hp.codebook_size, logit_tolerance());
        for (int book = 0; book < hp.num_codebooks; ++book) {
            const int32_t want = want_codes[static_cast<size_t>(book) * steps + step];
            if (codes[book] == want) continue;
            std::fprintf(stderr, "step %d: codebook %d gave %d, want %d\n", step, book,
                         codes[book], want);
            frame_agrees = false;
        }
        if (step == 0) {
            report.ok &= check_chained_frame(model, carried, want_semantic[step], n_threads, codes);
        }
        report.frames = step + 1;
        if (frame_agrees) report.agreeing_frames++;
        if (want_semantic[step] == hp.eos || step + 1 == steps) break;

        std::vector<int32_t> column(rows);
        column[0] = want_semantic[step];
        for (int book = 0; book < hp.num_codebooks; ++book)
            column[book + 1] = want_codes[static_cast<size_t>(book) * steps + step];
        if (!slow_step(model, column.data(), 1, width + step, n_threads, logits, carried,
                       &error)) {
            std::fprintf(stderr, "step %d slow: %s\n", step, error.c_str());
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <audio8-lm.gguf> <ref-dir> [threads]\n", argv[0]);
        return 1;
    }
    const int n_threads = argc > 3 ? std::atoi(argv[3]) : 4;

    lm_model model;
    std::string error;
    const int n_gpu_layers = audio8_test::is_gpu_test() ? GPU_LAYERS : 0;
    if (!load_lm(argv[1], n_gpu_layers, /*backend=*/"", model, &error)) {
        std::fprintf(stderr, "load: %s\n", error.c_str());
        return 1;
    }
    std::printf("backend: %s\n", ggml_backend_name(model.backend));
    if (!audio8_test::check_requested_gpu("lm", model.backend)) {
        free_lm(model);
        return 1;
    }

    step_report report;
    const bool ran = run_steps(model, fixture{argv[2]}, n_threads, report);
    free_lm(model);
    if (!ran) return 1;
    const double agreement =
        report.frames > 0 ? (double) report.agreeing_frames / report.frames : 0.0;
    if (agreement < token_agreement_bar()) {
        std::fprintf(stderr, "token agreement %.3f below the %.3f bar\n",
                     agreement, token_agreement_bar());
        report.ok = false;
    }
    std::printf("\n%s: %d of %d teacher-forced frames picked the reference tokens (%.1f%%)\n",
                report.ok ? "PASS" : "FAIL", report.agreeing_frames, report.frames,
                100.0 * agreement);
    return report.ok ? 0 : 1;
}
