// A Qwen3.5 model's gated-delta-net layers on the NPU against the CPU
// backend, logit for logit.
//
//   test-xdna-gdn -m model.gguf
//
// A prompt of more than 144 tokens (the prefill recurrence's minimum) goes in
// as one batch with every token's logits, then 16 tokens one at a time, the
// CPU's greedy choices fed to both, so the fused decode carries the recurrent
// state on the device from token to token. Twice, the sequence removed in
// between: the second round's decode has to start from the reset state. For a layer with twice as many
// value heads as q/k groups (the 4B and 9B's; probes/gdn_widen_model.py makes
// one at the 2B's width) that is two runs of the prefill recurrence and two
// passes of the fused core a token.
//
// It fails on a prefill row's KL(cpu || npu) or a round's mean of them over
// its bound, a decode row's nmse over its bound, a later round's row off round
// 0's by more than a row's floor, any error the backend logs (a fused layer that
// gives up is an error) and when the fused decode built fewer sessions than the
// model has recurrent layers. Skips (exit 0) without a model or an XDNA device.
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Floors measured on this test, each the larger of the 0.8B's and the 2B's: a decode row's nmse
// (2B), a prefill row's KL and a round's mean prefill KL (0.8B). A value may exceed its floor 3x.
static constexpr double FLOOR_DECODE  = 0.00528;
static constexpr double FLOOR_KL_ROW  = 0.0448;
static constexpr double FLOOR_KL_MEAN = 0.00163;
static constexpr double FLOOR_MARGIN  = 3.0;
static constexpr int    N_PROMPT_MIN  = 160;
static constexpr int    N_DECODE      = 16;
static constexpr int    N_ROUNDS      = 2;

static ggml_backend_dev_t xdna_device(void) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "XDNA") == 0) {
            return dev;
        }
    }
    return nullptr;
}

struct log_state {
    int  errors   = 0;
    long sessions = -1;  // fused-layer sessions the backend released
};

static void on_log(ggml_log_level level, const char * text, void * ud) {
    log_state * st = (log_state *) ud;
    if (level == GGML_LOG_LEVEL_ERROR) {
        st->errors++;
    }
    const char * r = std::strstr(text, "released ");
    if (r && std::strstr(text, "fused-layer sessions")) {
        st->sessions = std::max(st->sessions, std::strtol(r + 9, nullptr, 10));
    }
    if (level >= GGML_LOG_LEVEL_WARN) {
        std::fputs(text, stderr);
    }
}

static int argmax(const float * l, int n) {
    int b = 0;
    for (int i = 1; i < n; i++) {
        b = l[i] > l[b] ? i : b;
    }
    return b;
}

static double nmse(const float * ref, const float * got, int n) {
    double err = 0, sq = 0;
    for (int i = 0; i < n; i++) {
        if (!std::isfinite(got[i])) {
            return INFINITY;
        }
        err += ((double) ref[i] - got[i]) * ((double) ref[i] - got[i]);
        sq += (double) ref[i] * ref[i];
    }
    return sq > 0 ? err / sq : INFINITY;  // a reference row of zeros fails too
}

static double log_sum_exp(const float * l, int n) {
    const double m   = *std::max_element(l, l + n);
    double       sum = 0;
    for (int i = 0; i < n; i++) {
        sum += std::exp(l[i] - m);
    }
    return m + std::log(sum);
}

// KL(softmax(ref) || softmax(got)): unlike nmse, blind to a shift of the whole row
static double kl_div(const float * ref, const float * got, int n) {
    const double lr = log_sum_exp(ref, n);
    const double lg = log_sum_exp(got, n);
    double       kl = 0;
    for (int i = 0; i < n; i++) {
        const double lp = ref[i] - lr;
        kl += std::exp(lp) * (lp - (got[i] - lg));
    }
    return std::isfinite(kl) ? kl : INFINITY;
}

// A round's prefill rows' KL: the largest, at which token, and the sum
struct round_kl {
    double max   = 0;
    size_t worst = 0;
    double sum   = 0;
};

// Prints each round's prefill KL; false when a round's mean is over its bound
static bool rounds_kl_ok(const std::vector<round_kl> & kls, size_t n_prompt) {
    bool ok = true;
    for (size_t r = 0; r < kls.size(); r++) {
        const double mean = kls[r].sum / (double) n_prompt;
        ok                = ok && mean < FLOOR_KL_MEAN * FLOOR_MARGIN;
        std::printf("round %zu prefill KL: max %.3g (token %zu), mean %.3g\n", r, kls[r].max, kls[r].worst, mean);
    }
    return ok;
}

// Each later round's rows against round 0's: a prefill row by KL, a decode row
// by nmse, each within its floor; false otherwise
static bool rounds_same(const std::vector<std::vector<float>> & got, size_t per_round, size_t n_prompt, int nv) {
    bool   ok      = true;
    int    differ  = 0;
    double kl_max  = 0;
    double dec_max = 0;
    for (size_t i = per_round; i < got.size(); i++) {
        const float * ref   = got[i % per_round].data();
        const bool    input = i % per_round < n_prompt;
        const double  e     = input ? kl_div(ref, got[i].data(), nv) : nmse(ref, got[i].data(), nv);
        differ += std::memcmp(ref, got[i].data(), (size_t) nv * sizeof(float)) != 0 ? 1 : 0;
        (input ? kl_max : dec_max) = std::fmax(input ? kl_max : dec_max, e);
        ok                         = ok && e < (input ? FLOOR_KL_ROW : FLOOR_DECODE);
    }
    std::printf("later rounds vs round 0: %d rows differ, prefill KL max %.3g, decode nmse max %.3g\n", differ, kl_max,
                dec_max);
    return ok;
}

// The recurrent layers: all but every full_attention_interval-th of the
// model's own blocks (the rest of block_count are its draft layers).
static int n_recurrent(const llama_model * model) {
    char      buf[64];
    const int n = llama_model_meta_val_str(model, "qwen35.block_count", buf, sizeof(buf)) > 0 ? atoi(buf) : 0;
    const int nextn =
        llama_model_meta_val_str(model, "qwen35.nextn_predict_layers", buf, sizeof(buf)) > 0 ? atoi(buf) : 0;
    const int every =
        llama_model_meta_val_str(model, "qwen35.full_attention_interval", buf, sizeof(buf)) > 0 ? atoi(buf) : 0;
    return every > 0 ? (n - nextn) - (n - nextn) / every : 0;
}

// The prompt with every token's logits, then N_DECODE tokens one at a time:
// `feed` when given (the CPU's choices), else the greedy ones. Each row of
// logits in `out`, in order.
static bool run_round(llama_context *                   ctx,
                      int                               nv,
                      const std::vector<llama_token> &  prompt,
                      const std::vector<llama_token> &  feed,
                      std::vector<llama_token> &        chosen,
                      std::vector<std::vector<float>> & out) {
    llama_batch b = llama_batch_init((int32_t) prompt.size(), 0, 1);
    for (size_t i = 0; i < prompt.size(); i++) {
        b.token[i]     = prompt[i];
        b.pos[i]       = (llama_pos) i;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = 0;
        b.logits[i]    = true;  // every row: no crop of the last layer
    }
    b.n_tokens = (int32_t) prompt.size();
    bool ok    = llama_decode(ctx, b) == 0;
    llama_batch_free(b);
    for (size_t i = 0; ok && i < prompt.size(); i++) {
        const float * l = llama_get_logits_ith(ctx, (int32_t) i);
        out.emplace_back(l, l + nv);
    }
    chosen.clear();
    for (int i = 0; ok && i < N_DECODE; i++) {
        llama_token t = feed.empty() ? argmax(out.back().data(), nv) : feed[(size_t) i];
        chosen.push_back(t);
        ok = llama_decode(ctx, llama_batch_get_one(&t, 1)) == 0;
        if (ok) {
            const float * l = llama_get_logits_ith(ctx, -1);
            out.emplace_back(l, l + nv);
        }
    }
    return ok;
}

// N_ROUNDS rounds in one context, the sequence removed between them: each
// round after the first starts from a zeroed recurrent state.
static bool run(llama_model *                     model,
                const std::vector<llama_token> &  prompt,
                const std::vector<llama_token> &  feed,
                std::vector<llama_token> &        chosen,
                std::vector<std::vector<float>> & out) {
    const int            nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx                = 1024;
    cp.n_batch              = 512;
    cp.n_ubatch             = 512;
    llama_context * ctx     = llama_init_from_model(model, cp);
    bool            ok      = ctx != nullptr;
    for (int r = 0; ok && r < N_ROUNDS; r++) {
        ok = run_round(ctx, nv, prompt, feed, chosen, out) && llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
    }
    llama_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    for (int i = 1; i + 1 < argc; i++) {
        if (std::strcmp(argv[i], "-m") == 0) {
            model_path = argv[i + 1];
        }
    }
    // the fused decode for every residual width it takes, not only 1024
    setenv("GGML_XDNA_FUSED_WIDE", "1", 0);
    // static: the backend may still log after main returns
    static log_state st;
    llama_log_set(on_log, &st);
    llama_backend_init();
    if (!model_path || !xdna_device()) {
        std::printf("test-xdna-gdn: no model or no XDNA device, skipping\n");
        llama_backend_free();
        return 0;
    }

    llama_model_params mp_cpu      = llama_model_default_params();
    ggml_backend_dev_t no_dev[1]   = { nullptr };
    mp_cpu.devices                 = no_dev;
    mp_cpu.n_gpu_layers            = 0;
    llama_model_params mp_npu      = llama_model_default_params();
    ggml_backend_dev_t npu_devs[2] = { xdna_device(), nullptr };
    mp_npu.devices                 = npu_devs;
    mp_npu.n_gpu_layers            = 99;

    llama_model * cpu = llama_model_load_from_file(model_path, mp_cpu);
    if (!cpu) {
        return 1;
    }
    const llama_vocab *      v = llama_model_get_vocab(cpu);
    std::string              text;
    std::vector<llama_token> prompt(2048);
    while ((int) prompt.size() < N_PROMPT_MIN || text.empty()) {
        text += "A gated delta net layer keeps a recurrent state that it updates with every token it reads. ";
        prompt.resize(2048);
        prompt.resize(
            llama_tokenize(v, text.c_str(), (int) text.size(), prompt.data(), (int) prompt.size(), true, false));
    }
    const int n_rec = n_recurrent(cpu);
    const int nv    = llama_vocab_n_tokens(v);  // v goes with the CPU model

    std::vector<llama_token>        fed, again;
    std::vector<std::vector<float>> want, got;
    if (!run(cpu, prompt, {}, fed, want)) {
        std::printf("test-xdna-gdn: the CPU run failed\n");
        return 1;
    }
    llama_model_free(cpu);
    const int errors_cpu = st.errors;

    llama_model * npu = llama_model_load_from_file(model_path, mp_npu);
    if (!npu || !run(npu, prompt, fed, again, got)) {
        std::printf("test-xdna-gdn: the NPU run failed\n");
        return 1;
    }
    llama_model_free(npu);

    const size_t per_round     = prompt.size() + N_DECODE;
    const bool   sized         = got.size() == want.size() && want.size() == per_round * N_ROUNDS;
    bool         passed        = sized;
    double       worst_prefill = 0, worst_decode = 0;
    size_t       worst_row    = 0;
    int          top1_prefill = 0, top1_decode = 0, over_prefill = 0;

    std::vector<round_kl> kls(N_ROUNDS);
    for (size_t i = 0; sized && i < want.size(); i++) {
        const double e                         = nmse(want[i].data(), got[i].data(), nv);
        const bool   same                      = argmax(want[i].data(), nv) == argmax(got[i].data(), nv);
        const size_t j                         = i % per_round;
        const bool   input                     = j < prompt.size();
        worst_row                              = input && e > worst_prefill ? i : worst_row;
        (input ? worst_prefill : worst_decode) = std::fmax(input ? worst_prefill : worst_decode, e);
        (input ? top1_prefill : top1_decode) += same ? 1 : 0;
        const double kl = input ? kl_div(want[i].data(), got[i].data(), nv) : 0;
        round_kl &   rk = kls[i / per_round];
        rk.worst        = kl > rk.max ? j : rk.worst;
        rk.max          = std::fmax(rk.max, kl);
        rk.sum += kl;
        const bool over = input ? kl >= FLOOR_KL_ROW * FLOOR_MARGIN : e >= FLOOR_DECODE * FLOOR_MARGIN;
        passed          = passed && !over;
        over_prefill += input && over ? 1 : 0;
        if (!input || over) {
            std::printf("round %zu %s %3zu: nmse %.3g KL %.3g top1 %s\n", i / per_round, input ? "prefill" : "decode",
                        input ? j : j - prompt.size(), e, kl, same ? "same" : "differs");
        }
    }
    const bool kl_ok   = sized && rounds_kl_ok(kls, prompt.size());
    const bool same_ok = sized && rounds_same(got, per_round, prompt.size(), nv);
    passed             = kl_ok && same_ok && passed;
    std::printf(
        "prefill %zu tokens x %d: worst nmse %.3g (round %zu token %zu), %d rows over the KL bound, top1 same %d\n",
        prompt.size(), N_ROUNDS, worst_prefill, worst_row / per_round, worst_row % per_round, over_prefill,
        top1_prefill);
    std::printf("decode %d tokens x %d: worst nmse %.3g, top1 same %d\n", N_DECODE, N_ROUNDS, worst_decode,
                top1_decode);
    llama_backend_free();
    const int errors_npu = st.errors - errors_cpu;
    std::printf("recurrent layers %d, fused sessions %ld, backend errors %d\n", n_rec, st.sessions, errors_npu);
    passed = passed && errors_npu == 0 && n_rec > 0 && st.sessions == n_rec;
    std::printf("test-xdna-gdn: %s\n", passed ? "passed" : "FAILED");
    return passed ? 0 : 1;
}
