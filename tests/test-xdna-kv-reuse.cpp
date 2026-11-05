// A context made after another was freed, while a third stays alive, has to give the third's tokens; it
// cannot make the allocator hand the freed buffer's address back.
//
//   test-xdna-kv-reuse -m model.gguf [-c n_ctx] [rounds]
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

static constexpr int  N_DECODE  = 16;
static constexpr int  N_CTX     = 1024;
static constexpr int  ROUNDS    = 4;
static constexpr int  N_GPU_ALL = 99;
static constexpr char BIND[]    = "binds K ";
static constexpr char P1[]      = "A gated delta net layer keeps a recurrent state that it updates with every token.";
static constexpr char P2[]      = "The lighthouse keeper wrote down every ship that passed during the long winter.";

enum phase { PH_NONE, PH_B, PH_C };

// A run's binding as the backend logs it: the object, its K buffer and that buffer's generation.
struct bind_line {
    void *             obj = nullptr;
    void *             k   = nullptr;
    unsigned long long gen = 0;
};

struct log_state {
    int                      errors = 0;
    phase                    ph     = PH_NONE;
    std::map<void *, void *> b_k;  // object -> K buffer bound while B ran
    std::map<void *, void *> c_k;  // object -> K buffer bound while C ran
};

// "<who>: layer|pool <obj?> binds K <k> and V <v> (buffers <kgen> and <vgen>)"
static bool parse_bind(const char * text, bind_line & b) {
    const char * at = std::strstr(text, BIND);
    if (!at) {
        return false;
    }
    void *             v    = nullptr;
    unsigned long long vgen = 0;
    if (std::sscanf(at, "binds K %p and V %p (buffers %llu and %llu)", &b.k, &v, &b.gen, &vgen) != 4) {
        return false;
    }
    const char * layer = std::strstr(text, "layer ");
    const char * pool  = std::strstr(text, "pool ");
    if (layer && layer < at) {
        std::sscanf(layer, "layer %p", &b.obj);
    } else if (pool && pool < at) {
        std::sscanf(pool, "pool %p", &b.obj);
    }
    return true;
}

static void on_log(ggml_log_level level, const char * text, void * ud) {
    log_state * st = (log_state *) ud;
    if (level == GGML_LOG_LEVEL_ERROR) {
        st->errors++;
    }
    bind_line b;
    if (st->ph != PH_NONE && parse_bind(text, b)) {
        (st->ph == PH_B ? st->b_k : st->c_k)[b.obj] = b.k;
    }
    if (level >= GGML_LOG_LEVEL_WARN) {
        std::fputs(text, stderr);
    }
}

static ggml_backend_dev_t xdna_device(void) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "XDNA") == 0) {
            return dev;
        }
    }
    return nullptr;
}

static llama_token greedy(const float * l, int nv) {
    return (llama_token) (std::max_element(l, l + nv) - l);
}

static std::vector<llama_token> tokenize(const llama_model * model, const char * text) {
    const llama_vocab *      v = llama_model_get_vocab(model);
    std::vector<llama_token> t(N_CTX);
    t.resize(llama_tokenize(v, text, (int) std::strlen(text), t.data(), (int) t.size(), true, false));
    return t;
}

static llama_context * new_context(llama_model * model, int n_ctx) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx                = n_ctx;
    return llama_init_from_model(model, cp);
}

// The prompt as one batch, then N_DECODE greedy tokens one at a time into `out`.
static bool generate(llama_context * ctx, std::vector<llama_token> prompt, std::vector<llama_token> & out) {
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    if (llama_decode(ctx, llama_batch_get_one(prompt.data(), (int) prompt.size())) != 0) {
        return false;
    }
    for (int i = 0; i < N_DECODE; i++) {
        llama_token t = greedy(llama_get_logits_ith(ctx, -1), nv);
        out.push_back(t);
        if (llama_decode(ctx, llama_batch_get_one(&t, 1)) != 0) {
            return false;
        }
    }
    return true;
}

// Each object B bound has to bind again under C; `reused` counts those whose K buffer came back at B's address.
static int count_unbound(const log_state & st, int & reused) {
    int missing = 0;
    for (const auto & kv : st.b_k) {
        const auto it = st.c_k.find(kv.first);
        missing += it == st.c_k.end() ? 1 : 0;
        reused += it != st.c_k.end() && it->second == kv.second ? 1 : 0;
    }
    return missing;
}

// One round: B decodes P1 and is freed, C decodes P2 against `want`. Counts into the round's tallies.
static bool run_round(llama_model *                    model,
                      int                              n_ctx,
                      const std::vector<llama_token> & want,
                      log_state &                      st,
                      int &                            unbound,
                      int &                            reused) {
    std::vector<llama_token> b_out, c_out;
    st.b_k.clear();
    st.c_k.clear();
    st.ph              = PH_B;
    llama_context * b  = new_context(model, n_ctx);
    const bool      ok = b && generate(b, tokenize(model, P1), b_out);
    llama_free(b);
    if (!ok) {
        return false;
    }
    st.ph                = PH_C;
    llama_context * c    = new_context(model, n_ctx);
    const bool      c_ok = c && generate(c, tokenize(model, P2), c_out);
    llama_free(c);
    st.ph = PH_NONE;
    if (!c_ok) {
        return false;
    }
    int r = 0;
    unbound += count_unbound(st, r);
    reused += r;
    const bool same = c_out == want;
    std::printf("round: B bound %zu runs, C bound %zu, K buffer at B's address %d, tokens %s\n", st.b_k.size(),
                st.c_k.size(), r, same ? "same as A's" : "DIFFER from A's");
    // no binding seen: the attention did not run on the array, or its log did not reach the test
    return same && !st.b_k.empty();
}

struct args {
    const char * model_path = nullptr;
    int          n_ctx      = N_CTX;
    int          rounds     = ROUNDS;
};

static args parse_args(int argc, char ** argv) {
    args a;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            a.model_path = argv[++i];
        } else if (std::strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            a.n_ctx = std::atoi(argv[++i]);
        } else {
            a.rounds = std::max(1, std::atoi(argv[i]));
        }
    }
    return a;
}

// The rounds that gave A's tokens.
static int run_rounds(llama_model *                    model,
                      const args &                     a,
                      const std::vector<llama_token> & want,
                      log_state &                      st,
                      int &                            unbound,
                      int &                            reused) {
    int passed = 0;
    for (int r = 0; r < a.rounds; r++) {
        passed += run_round(model, a.n_ctx, want, st, unbound, reused) ? 1 : 0;
    }
    return passed;
}

int main(int argc, char ** argv) {
    const args       arg = parse_args(argc, argv);
    // static: the backend may still log after main returns
    static log_state st;
    llama_log_set(on_log, &st);
    llama_backend_init();
    if (!arg.model_path || !xdna_device()) {
        std::printf("test-xdna-kv-reuse: no model or no XDNA device, skipping\n");
        llama_backend_free();
        return 0;
    }
    llama_model_params mp      = llama_model_default_params();
    ggml_backend_dev_t devs[2] = { xdna_device(), nullptr };
    mp.devices                 = devs;
    mp.n_gpu_layers            = N_GPU_ALL;
    llama_model * model        = llama_model_load_from_file(arg.model_path, mp);
    if (!model) {
        return 1;
    }
    std::vector<llama_token> want;
    llama_context *          a       = new_context(model, arg.n_ctx);
    const bool               ok      = a && generate(a, tokenize(model, P2), want);
    int                      unbound = 0, reused = 0;
    const int                passed_rounds = ok ? run_rounds(model, arg, want, st, unbound, reused) : 0;
    llama_free(a);
    llama_model_free(model);
    llama_backend_free();
    if (!ok) {
        std::printf("test-xdna-kv-reuse: a decode failed\n");
        return 1;
    }
    std::printf(
        "n_ctx %d: %d of %d rounds gave A's tokens; runs not bound again under C %d; C's K buffer at B's "
        "address %d times; backend errors %d\n",
        arg.n_ctx, passed_rounds, arg.rounds, unbound, reused, st.errors);
    const bool passed = passed_rounds == arg.rounds && unbound == 0 && st.errors == 0;
    std::printf("test-xdna-kv-reuse: %s\n", passed ? "passed" : "FAILED");
    return passed ? 0 : 1;
}
