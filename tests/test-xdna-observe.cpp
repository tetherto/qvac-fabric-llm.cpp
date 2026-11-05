// A decode watched through an eval callback (llama-eval-callback, imatrix):
// the scheduler then computes the graph in pieces that end at the watched
// tensors, and the fused decode has to give the logits it gives unwatched.
//
//   test-xdna-observe -m model.gguf
//
// The prompt as one batch, then 16 tokens one at a time, the unwatched run's
// greedy choices fed to both runs, each in a fresh context. The watched run
// watches l_out of the first recurrent layer and of the first attention layer
// on every other decode, so each watched decode follows one the fused layers
// took their inputs from in full. A split graph is not bit-exact, so every
// logit row has to stay within NMSE_MAX of the unwatched one, and its argmax
// at most PICK_GAP_MAX below the unwatched row's top logit.
//
// It fails on any error the backend logs, when the callback saw other than the
// two tensors a watched decode, and when the fused decode built fewer sessions
// than the model has recurrent layers. Skips (exit 0) without a model, an XDNA
// device or a Qwen3.5 layer layout.
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// a split that read the previous token's residual gave ~1; the repaired split worst 1.5e-3 (2B)
static constexpr double NMSE_MAX     = 1e-2;
// a split moves the top logit's margin over the runner-up candidates by up to 0.22 (0.8B), so near-tied rows swap
static constexpr float  PICK_GAP_MAX = 0.5f;

static constexpr int  N_DECODE    = 16;
static constexpr int  N_CTX       = 1024;
static constexpr int  WATCH_EVERY = 2;  // watch decode i when i % WATCH_EVERY == WATCH_EVERY - 1
static constexpr int  GDN_WATCHED = 0;  // the first layer is a recurrent one
static constexpr int  N_GPU_ALL   = 99;
static constexpr char L_OUT[]     = "l_out-";
static constexpr char PROMPT[]    = "A gated delta net layer keeps a recurrent state that it updates with every token.";

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

struct watch_state {
    bool armed   = false;
    int  gdn     = GDN_WATCHED;
    int  att     = -1;
    int  seen    = 0;  // tensors handed to the callback
    int  decodes = 0;  // watched decodes
};

static bool watched(const watch_state * w, const ggml_tensor * t) {
    if (!w->armed || std::strncmp(t->name, L_OUT, sizeof(L_OUT) - 1) != 0) {
        return false;
    }
    const int il = std::atoi(t->name + sizeof(L_OUT) - 1);
    return il == w->gdn || il == w->att;
}

static bool on_eval(ggml_tensor * t, bool ask, void * ud) {
    watch_state * w = (watch_state *) ud;
    if (ask) {
        return watched(w, t);
    }
    w->seen += watched(w, t) ? 1 : 0;
    return true;
}

static int meta_int(const llama_model * model, const char * key) {
    char buf[64];
    return llama_model_meta_val_str(model, key, buf, sizeof(buf)) > 0 ? std::atoi(buf) : 0;
}

// The recurrent layers: all but every full_attention_interval-th of the
// model's own blocks (the rest of block_count are its draft layers).
static int n_recurrent(const llama_model * model) {
    const int n     = meta_int(model, "qwen35.block_count") - meta_int(model, "qwen35.nextn_predict_layers");
    const int every = meta_int(model, "qwen35.full_attention_interval");
    return every > 0 ? n - n / every : 0;
}

static llama_token greedy(const float * l, int nv) {
    return (llama_token) (std::max_element(l, l + nv) - l);
}

static double nmse(const std::vector<float> & ref, const std::vector<float> & got) {
    double e = 0.0, r = 0.0;
    for (size_t i = 0; i < ref.size(); i++) {
        const double d = (double) got[i] - ref[i];
        e += d * d;
        r += (double) ref[i] * ref[i];
    }
    return r > 0.0 ? e / r : e;
}

static void push_row(llama_context * ctx, int nv, std::vector<std::vector<float>> & rows) {
    const float * l = llama_get_logits_ith(ctx, -1);
    rows.emplace_back(l, l + nv);
}

// The decode after the prompt: `feed`'s tokens when given, else the greedy
// ones, each into `fed`; watched (when `w` is) on every WATCH_EVERY-th.
static bool decode_tokens(llama_context *                   ctx,
                          int                               nv,
                          const std::vector<llama_token> &  feed,
                          watch_state *                     w,
                          std::vector<llama_token> &        fed,
                          std::vector<std::vector<float>> & rows) {
    for (int i = 0; i < N_DECODE; i++) {
        llama_token t = feed.empty() ? greedy(rows.back().data(), nv) : feed[i];
        fed.push_back(t);
        const bool watch = w && i % WATCH_EVERY == WATCH_EVERY - 1;
        if (watch) {
            w->armed = true;
            w->decodes++;
        }
        const bool ok = llama_decode(ctx, llama_batch_get_one(&t, 1)) == 0;
        if (w) {
            w->armed = false;
        }
        if (!ok) {
            return false;
        }
        push_row(ctx, nv, rows);
    }
    return true;
}

static bool run(llama_model *                     model,
                std::vector<llama_token>          prompt,
                const std::vector<llama_token> &  feed,
                watch_state *                     w,
                std::vector<llama_token> &        fed,
                std::vector<std::vector<float>> & rows) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx                = N_CTX;
    cp.cb_eval              = w ? on_eval : nullptr;
    cp.cb_eval_user_data    = w;
    llama_context * ctx     = llama_init_from_model(model, cp);
    if (!ctx) {
        return false;
    }
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    bool      ok = llama_decode(ctx, llama_batch_get_one(prompt.data(), (int) prompt.size())) == 0;
    if (ok) {
        push_row(ctx, nv, rows);
        ok = decode_tokens(ctx, nv, feed, w, fed, rows);
    }
    llama_free(ctx);
    return ok;
}

// How many rows are over NMSE_MAX or pick a token over PICK_GAP_MAX below the
// unwatched top, each one printed; `worst` gets the largest nmse.
static int count_off(const std::vector<std::vector<float>> & want,
                     const std::vector<std::vector<float>> & got,
                     double &                                worst) {
    int n = 0;
    worst = 0.0;
    for (size_t i = 0; i < want.size(); i++) {
        const double e     = nmse(want[i], got[i]);
        const int    nv    = (int) want[i].size();
        const float  below = want[i][greedy(want[i].data(), nv)] - want[i][greedy(got[i].data(), nv)];
        worst              = std::max(worst, e);
        if (e > NMSE_MAX || below > PICK_GAP_MAX) {
            std::printf("row %2zu (%s): nmse %.3g, argmax %.3g below the unwatched top\n", i,
                        i == 0 ? "prompt" : "decode", e, below);
            n++;
        }
    }
    return n;
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
        std::printf("test-xdna-observe: no model or no XDNA device, skipping\n");
        llama_backend_free();
        return 0;
    }
    llama_model_params mp      = llama_model_default_params();
    ggml_backend_dev_t devs[2] = { xdna_device(), nullptr };
    mp.devices                 = devs;
    mp.n_gpu_layers            = N_GPU_ALL;
    llama_model * model        = llama_model_load_from_file(model_path, mp);
    if (!model) {
        return 1;
    }
    const int every = meta_int(model, "qwen35.full_attention_interval");
    if (every <= GDN_WATCHED + 1) {
        std::printf("test-xdna-observe: not a Qwen3.5 layer layout, skipping\n");
        llama_model_free(model);
        llama_backend_free();
        return 0;
    }
    const llama_vocab *      v = llama_model_get_vocab(model);
    std::vector<llama_token> prompt(N_CTX);
    prompt.resize(
        llama_tokenize(v, PROMPT, (int) std::strlen(PROMPT), prompt.data(), (int) prompt.size(), true, false));

    watch_state w;
    w.att = every - 1;
    std::vector<llama_token>        fed, again;
    std::vector<std::vector<float>> want, got;
    const bool ok    = run(model, prompt, {}, nullptr, fed, want) && run(model, prompt, fed, &w, again, got);
    const int  n_rec = n_recurrent(model);
    llama_model_free(model);
    llama_backend_free();
    if (!ok) {
        std::printf("test-xdna-observe: a decode failed\n");
        return 1;
    }

    double    worst = 0.0;
    const int off   = want.size() == got.size() ? count_off(want, got, worst) : -1;
    std::printf(
        "rows %zu, off %d, worst nmse %.3g (bound %.3g); watched l_out-%d and l_out-%d on %d decodes, %d "
        "tensors seen\n",
        want.size(), off, worst, NMSE_MAX, w.gdn, w.att, w.decodes, w.seen);
    std::printf("recurrent layers %d, fused sessions %ld, backend errors %d\n", n_rec, st.sessions, st.errors);
    const bool passed =
        off == 0 && w.decodes > 0 && w.seen == 2 * w.decodes && st.errors == 0 && n_rec > 0 && st.sessions == n_rec;
    std::printf("test-xdna-observe: %s\n", passed ? "passed" : "FAILED");
    return passed ? 0 : 1;
}
