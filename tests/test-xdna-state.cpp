// A sequence's state saved, the sequence moved on, the state restored: what a
// server does when it reuses a cached prompt from a checkpoint.
//
//   test-xdna-state -m model.gguf
//
// Two runs of one prompt, both its last token decoded on its own and then 16
// greedy tokens. One saves the sequence (llama_state_seq_get_data) after the
// rest of the prompt, decodes two other tokens, and restores it
// (llama_state_seq_set_data) first. The fused decode keeps the recurrent state
// on the array between tokens, so a restore it did not follow went on from the
// detour; both runs have to give the same tokens. Skips (exit 0) without a
// model or an XDNA device.
//
// Then two things a prefill after a fused decode must get right on a context that goes on: a short continuation prompt after
// some decoded tokens, and a new prompt after the sequence was cleared (the fused sessions carry the state on the array and
// llama's cache only sees it when a graph that reads it follows it). The logits after every batch and every decoded token
// are compared with a run of the same steps with the chunk rule off (GGML_XDNA_HOST_CHUNK=0), fed the same tokens.
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static ggml_backend_dev_t xdna_device(void) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "XDNA") == 0) {
            // a listed NPU that does not open fails rather than skips, so a
            // broken NPU runner cannot pass
            auto ready = (bool (*)(void)) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev),
                                                                            "ggml_backend_xdna_device_ready");
            if (!ready || !ready()) {
                std::fprintf(stderr, "XDNA device is listed but did not open\n");
                std::exit(1);
            }
            return dev;
        }
    }
    return nullptr;
}

static llama_token greedy(llama_context * ctx, int nv) {
    const float * l = llama_get_logits_ith(ctx, -1);
    llama_token   b = 0;
    for (llama_token t = 1; t < nv; t++) {
        b = l[t] > l[b] ? t : b;
    }
    return b;
}

static std::vector<llama_token> gen(llama_context * ctx, int nv, int n) {
    std::vector<llama_token> out;
    for (int i = 0; i < n; i++) {
        llama_token t = greedy(ctx, nv);
        out.push_back(t);
        if (llama_decode(ctx, llama_batch_get_one(&t, 1)) != 0) {
            break;
        }
    }
    return out;
}


typedef int (*host_chunk_count_fn)(void);

// The backend's count of chunks it ran whole on the host; null on a backend without it.
static host_chunk_count_fn host_chunk_counter(ggml_backend_dev_t dev) {
    return dev ? (host_chunk_count_fn) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev),
                                                                         "ggml_backend_xdna_test_host_chunk_count") :
                 nullptr;
}

// What a scenario saw: the logits after every batch and decoded token (one vector a sequence), the tokens it decoded one at a
// time, and the host chunks of its prompts and of its decode steps.
struct trace {
    std::vector<llama_token>        tokens;
    std::vector<std::vector<float>> logits;
    int                             prompt_chunks = 0;
    int                             decode_chunks = 0;
};

constexpr int FOLLOW_STEPS = 8;
constexpr int N_SEQS       = 4;

static std::vector<llama_token> tokenize(const llama_vocab * v, const std::string & text, bool first) {
    std::vector<llama_token> t(256);
    t.resize(llama_tokenize(v, text.c_str(), (int) text.size(), t.data(), (int) t.size(), first, false));
    return t;
}

// The logits of the batch's tokens `at` (indices into the batch), joined; -1 is the last.
static void snapshot(llama_context * ctx, int nv, const std::vector<int> & at, trace & tr) {
    std::vector<float> all;
    for (int i : at) {
        const float * l = llama_get_logits_ith(ctx, i);
        all.insert(all.end(), l, l + nv);
    }
    tr.logits.push_back(std::move(all));
}

// One prompt on sequence 0, then its logits.
static bool decode_prompt(llama_context * ctx, std::vector<llama_token> & p, int nv, trace & tr, const host_chunk_count_fn chunks) {
    const int before = chunks ? chunks() : 0;
    const bool ok    = llama_decode(ctx, llama_batch_get_one(p.data(), (int) p.size())) == 0;
    tr.prompt_chunks += chunks ? chunks() - before : 0;
    snapshot(ctx, nv, { -1 }, tr);
    return ok;
}

// `n` single-token decodes of sequence 0: the greedy token, or the next of `forced`.
static bool decode_steps(llama_context * ctx, int nv, int n, trace & tr, const std::vector<llama_token> * forced, size_t & next) {
    for (int i = 0; i < n; i++) {
        llama_token t = forced ? (*forced)[next] : greedy(ctx, nv);
        next++;
        if (!forced) {
            tr.tokens.push_back(t);
        }
        if (llama_decode(ctx, llama_batch_get_one(&t, 1)) != 0) {
            return false;
        }
        snapshot(ctx, nv, { -1 }, tr);
    }
    return true;
}

static trace follow_up(llama_model * model, const llama_vocab * v, int nv, const std::vector<llama_token> * forced,
                       const host_chunk_count_fn chunks) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx                = 1024;
    llama_context *          ctx = llama_init_from_model(model, cp);
    std::vector<llama_token> first = tokenize(v, "A gated delta net layer keeps a recurrent state that it updates.", true);
    std::vector<llama_token> next  = tokenize(v, " The second request goes on with a short question about that state.", false);
    std::vector<llama_token> other = tokenize(v, "Thirteen horses crossed a quiet river at dawn.", true);
    trace                    tr;
    size_t                   k = 0;
    decode_prompt(ctx, first, nv, tr, chunks);
    decode_steps(ctx, nv, FOLLOW_STEPS, tr, forced, k);
    decode_prompt(ctx, next, nv, tr, chunks);
    decode_steps(ctx, nv, FOLLOW_STEPS, tr, forced, k);
    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
    decode_prompt(ctx, other, nv, tr, chunks);
    decode_steps(ctx, nv, FOLLOW_STEPS, tr, forced, k);
    llama_free(ctx);
    return tr;
}

// N_SEQS sequences prefilled in one batch, then decoded together a token a step: the step's graph has N_SEQS rows and one token
// a sequence, which is a decode.
static trace four_sequences(llama_model * model, const llama_vocab * v, int nv, const std::vector<llama_token> * forced,
                            const host_chunk_count_fn chunks) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx                = 1024;
    cp.n_seq_max            = N_SEQS;
    llama_context *          ctx = llama_init_from_model(model, cp);
    const char *             texts[N_SEQS] = { "The first sequence is about a quiet river.", "A second one speaks of the array and the host.",
                                               "Third, a short note on recurrent state.", "The fourth is a question about caches." };
    std::vector<llama_token> prompts[N_SEQS];
    llama_batch              prefill = llama_batch_init(256, 0, 1);
    for (int s = 0; s < N_SEQS; s++) {
        prompts[s] = tokenize(v, texts[s], true);
        for (size_t i = 0; i < prompts[s].size(); i++) {
            const int at           = prefill.n_tokens++;
            prefill.token[at]      = prompts[s][i];
            prefill.pos[at]        = (llama_pos) i;
            prefill.n_seq_id[at]   = 1;
            prefill.seq_id[at][0]  = s;
            prefill.logits[at]     = i + 1 == prompts[s].size();
        }
    }
    trace tr;
    llama_decode(ctx, prefill);
    std::vector<int> last_of;
    for (int i = 0; i < prefill.n_tokens; i++) {
        if (prefill.logits[i]) {
            last_of.push_back(i);
        }
    }
    snapshot(ctx, nv, last_of, tr);
    llama_batch_free(prefill);
    size_t next = 0;
    for (int step = 0; step < FOLLOW_STEPS; step++) {
        llama_batch b = llama_batch_init(N_SEQS, 0, 1);
        for (int s = 0; s < N_SEQS; s++) {
            const float * l = llama_get_logits_ith(ctx, last_of[s]);
            llama_token   t = 0;
            if (forced) {
                t = (*forced)[next];
            } else {
                for (llama_token c = 1; c < nv; c++) {
                    t = l[c] > l[t] ? c : t;
                }
                tr.tokens.push_back(t);
            }
            next++;
            const int at          = b.n_tokens++;
            b.token[at]           = t;
            b.pos[at]             = (llama_pos) (prompts[s].size() + step);
            b.n_seq_id[at]        = 1;
            b.seq_id[at][0]       = s;
            b.logits[at]          = true;
        }
        const int before = chunks ? chunks() : 0;
        llama_decode(ctx, b);
        tr.decode_chunks += chunks ? chunks() - before : 0;
        last_of.clear();
        for (int s = 0; s < N_SEQS; s++) {
            last_of.push_back(s);
        }
        snapshot(ctx, nv, last_of, tr);
        llama_batch_free(b);
    }
    llama_free(ctx);
    return tr;
}

static double nmse(const std::vector<float> & want, const std::vector<float> & got) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < want.size(); i++) {
        num += (double) (want[i] - got[i]) * (want[i] - got[i]);
        den += (double) want[i] * want[i];
    }
    return den > 0.0 ? num / den : num;
}

// The logits of the run against the reference, after every step: the array and the host differ by their roundings (1e-4 or so
// of the logits' energy), a sequence that went on from the wrong state by order one.
static bool same_logits(const char * what, const trace & want, const trace & got) {
    constexpr double TOL = 2e-2;
    double           worst = 0.0;
    size_t           at    = 0;
    if (want.logits.size() != got.logits.size()) {
        std::printf("%s: %zu snapshots against %zu\n", what, want.logits.size(), got.logits.size());
        return false;
    }
    for (size_t i = 0; i < want.logits.size(); i++) {
        const double e = nmse(want.logits[i], got.logits[i]);
        if (e > worst) {
            worst = e;
            at    = i;
        }
    }
    std::printf("%s: worst nmse %.3g at step %zu of %zu (limit %.0e)\n", what, worst, at, want.logits.size(), TOL);
    return worst < TOL;
}

static bool check_follow_up(llama_model * model, const llama_vocab * v, int nv, const host_chunk_count_fn chunks) {
    setenv("GGML_XDNA_HOST_CHUNK", "0", 1);
    const trace want = follow_up(model, v, nv, nullptr, chunks);
    unsetenv("GGML_XDNA_HOST_CHUNK");
    const trace got = follow_up(model, v, nv, &want.tokens, chunks);
    bool        ok  = same_logits("continuation and new prompt after a decode", want, got);
    // the prompts are short enough for the host, so the rule fires on the three of them
    const bool fired = !chunks || got.prompt_chunks == 3;
    std::printf("host chunks in the prompts: %d%s\n", got.prompt_chunks, chunks ? "" : " (no counter)");
    return ok && fired && want.prompt_chunks == 0;
}

static bool check_four_sequences(llama_model * model, const llama_vocab * v, int nv, const host_chunk_count_fn chunks) {
    setenv("GGML_XDNA_HOST_CHUNK", "0", 1);
    const trace want = four_sequences(model, v, nv, nullptr, chunks);
    unsetenv("GGML_XDNA_HOST_CHUNK");
    const trace got = four_sequences(model, v, nv, &want.tokens, chunks);
    const bool  ok  = same_logits("four sequences, a decode step each", want, got);
    std::printf("host chunks in the decode steps: %d\n", got.decode_chunks);
    return ok && got.decode_chunks == 0;
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    for (int i = 1; i + 1 < argc; i++) {
        if (std::strcmp(argv[i], "-m") == 0) {
            model_path = argv[i + 1];
        }
    }
    llama_backend_init();
    if (!model_path || (!getenv("ON_CPU") && !xdna_device())) {
        std::printf("test-xdna-state: no model or no XDNA device, skipping\n");
        llama_backend_free();
        return 0;
    }
    llama_model_params mp      = llama_model_default_params();
    ggml_backend_dev_t devs[2] = { xdna_device(), nullptr };
    if (getenv("ON_CPU") == nullptr) {
        mp.devices      = devs;
        mp.n_gpu_layers = 99;
    } else {
        mp.n_gpu_layers = 0;
    }
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        return 1;
    }
    const llama_vocab * v  = llama_model_get_vocab(model);
    const int           nv = llama_vocab_n_tokens(v);
    std::string         prompt;
    for (int r = 0; r < 8; r++) {
        prompt += "A gated delta net layer keeps a recurrent state that it updates with every token. ";
    }
    std::vector<llama_token> p(512);
    p.resize(llama_tokenize(v, prompt.c_str(), (int) prompt.size(), p.data(), (int) p.size(), true, false));

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx                = 1024;

    // the same steps either way: the prompt but its last token in one batch,
    // then the last token, then 16 greedy tokens. The restored run saves the
    // sequence after the first batch, decodes two detour tokens, and restores.
    const int n_head = (int) p.size() - 1;
    llama_token last = p.back();

    llama_context * ref = llama_init_from_model(model, cp);
    llama_decode(ref, llama_batch_get_one(p.data(), n_head));
    llama_decode(ref, llama_batch_get_one(&last, 1));
    const std::vector<llama_token> want = gen(ref, nv, 16);
    llama_free(ref);

    llama_context * ctx = llama_init_from_model(model, cp);
    llama_decode(ctx, llama_batch_get_one(p.data(), n_head));
    std::vector<uint8_t> st(llama_state_seq_get_size(ctx, 0));
    llama_state_seq_get_data(ctx, st.data(), st.size(), 0);
    llama_token detour[2] = { 13, 279 };
    llama_decode(ctx, llama_batch_get_one(&detour[0], 1));
    llama_decode(ctx, llama_batch_get_one(&detour[1], 1));
    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
    const size_t set = llama_state_seq_set_data(ctx, st.data(), st.size(), 0);
    llama_decode(ctx, llama_batch_get_one(&last, 1));
    const std::vector<llama_token> got = gen(ctx, nv, 16);
    llama_free(ctx);
    std::printf("state %zu bytes, restored %zu\n", st.size(), set);

    std::string sw, sg;
    for (llama_token t : want) sw += std::to_string(t) + " ";
    for (llama_token t : got) sg += std::to_string(t) + " ";
    std::printf("straight: %s\nrestored: %s\n", sw.c_str(), sg.c_str());
    const host_chunk_count_fn chunks = host_chunk_counter(xdna_device());
    bool                      ok     = want == got;
    if (getenv("ON_CPU") == nullptr) {
        ok = check_follow_up(model, v, nv, chunks) && ok;
        ok = check_four_sequences(model, v, nv, chunks) && ok;
    }
    std::printf("test-xdna-state: %s\n", ok ? "passed" : "FAILED");
    llama_model_free(model);
    llama_backend_free();
    return ok ? 0 : 1;
}
