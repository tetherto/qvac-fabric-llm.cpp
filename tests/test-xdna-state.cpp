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
//
// The straight run's logits after its last prompt token and after each greedy token are also compared with a CPU-only run
// of the same steps, fed its tokens: the runs above take the same routes on both sides, so a route that is wrong the same
// way every time passes them. So is a longer sequence's state restored after a shorter one was decoded: the restore writes
// cache rows below the ones that decode left on the device, and a test hook keeps them stale in memory until a flush.
//
// Then two sequences in one unified cache, as llama-server's default slots: the first from cell 0, whose decode attention
// reads the identity K/V row list the fused attention layer takes; the second in the cells after the first's; and the first
// again, past the second's cells. The fused layer has to leave those two lists to the host. Each step's logits are compared
// with a CPU-only run fed the same tokens.
#include "llama.h"

#include <algorithm>
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

static llama_token greedy_of(const float * l, int nv) {
    llama_token b = 0;
    for (llama_token t = 1; t < nv; t++) {
        b = l[t] > l[b] ? t : b;
    }
    return b;
}

static llama_token greedy(llama_context * ctx, int nv) {
    return greedy_of(llama_get_logits_ith(ctx, -1), nv);
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

typedef int (*stale_kv_fn)(int arm);

// The backend's hook that leaves the K/V rows the host next writes stale in memory until a flush, and counts the times it
// did; null on a backend without it.
static stale_kv_fn stale_kv_hook(ggml_backend_dev_t dev) {
    return dev ? (stale_kv_fn) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev),
                                                                 "ggml_backend_xdna_test_stale_kv") :
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

constexpr int STRAIGHT_TOKENS = 16;

// The prompt but its last token in one batch, then the last token, then STRAIGHT_TOKENS tokens, greedy or `forced`: the logits
// after the last token and after each one decoded.
static trace straight(llama_model * model, std::vector<llama_token> & p, int nv, const std::vector<llama_token> * forced) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx                = 1024;
    llama_context * ctx     = llama_init_from_model(model, cp);
    trace           tr;
    size_t          k = 0;
    llama_decode(ctx, llama_batch_get_one(p.data(), (int) p.size() - 1));
    if (llama_decode(ctx, llama_batch_get_one(&p.back(), 1)) == 0) {
        snapshot(ctx, nv, { -1 }, tr);
        decode_steps(ctx, nv, STRAIGHT_TOKENS, tr, forced, k);
    }
    llama_free(ctx);
    return tr;
}

constexpr int RESTORE_LONG_TOKENS  = 150;
constexpr int RESTORE_SHORT_TOKENS = 90;
constexpr int RESTORE_SHORT_STEPS =
    10;  // decoded after the short prompt: its rows on the device stop below the long one's

// the first `n` tokens of `text` repeated
static std::vector<llama_token> tokens_of(const llama_vocab * v, const std::string & text, int n) {
    std::string              s;
    std::vector<llama_token> t;
    while ((int) t.size() < n) {
        s += text;
        t = tokenize(v, s, true);
    }
    t.resize(n);
    return t;
}

// A long prompt's sequence saved and cleared, a short prompt prefilled and decoded RESTORE_SHORT_STEPS tokens and cleared,
// the long one restored (llama_state_seq_set_data, with `stale` armed when given), then one more of its tokens and
// STRAIGHT_TOKENS greedy or `forced`: the logits after each.
static trace restore_longer(llama_model *                    model,
                            const llama_vocab *              v,
                            int                              nv,
                            const std::vector<llama_token> * forced,
                            const stale_kv_fn                stale) {
    llama_context_params cp      = llama_context_default_params();
    cp.n_ctx                     = 1024;
    llama_context *          ctx = llama_init_from_model(model, cp);
    llama_memory_t           mem = llama_get_memory(ctx);
    std::vector<llama_token> longp =
        tokens_of(v, "Thirteen horses crossed a quiet river at dawn. ", RESTORE_LONG_TOKENS);
    std::vector<llama_token> shortp = tokens_of(v, "A short note on caches and the array. ", RESTORE_SHORT_TOKENS);
    llama_decode(ctx, llama_batch_get_one(longp.data(), (int) longp.size()));
    std::vector<uint8_t> st(llama_state_seq_get_size(ctx, 0));
    llama_state_seq_get_data(ctx, st.data(), st.size(), 0);
    llama_memory_seq_rm(mem, 0, -1, -1);
    llama_decode(ctx, llama_batch_get_one(shortp.data(), (int) shortp.size()));
    gen(ctx, nv, RESTORE_SHORT_STEPS);
    llama_memory_seq_rm(mem, 0, -1, -1);
    if (stale) {
        stale(1);
    }
    llama_state_seq_set_data(ctx, st.data(), st.size(), 0);
    trace  tr;
    size_t k = 0;
    if (llama_decode(ctx, llama_batch_get_one(&longp.front(), 1)) == 0) {
        snapshot(ctx, nv, { -1 }, tr);
        decode_steps(ctx, nv, STRAIGHT_TOKENS, tr, forced, k);
    }
    llama_free(ctx);
    return tr;
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
static bool same_logits(const char * what, const trace & want, const trace & got, double tol = 2e-2) {
    double worst = 0.0;
    size_t at    = 0;
    if (want.logits.size() != got.logits.size()) {
        std::printf("%s: %zu snapshots against %zu\n", what, want.logits.size(), got.logits.size());
        return false;
    }
    for (size_t i = 0; i < want.logits.size(); i++) {
        const double e = nmse(want.logits[i], got.logits[i]);
        if (!std::isfinite(e) || e > worst) {  // a NaN logit too
            worst = std::isfinite(e) ? e : INFINITY;
            at    = i;
        }
    }
    std::printf("%s: worst nmse %.3g at step %zu of %zu (limit %.0e)\n", what, worst, at, want.logits.size(), tol);
    return worst < tol;
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

// The weights stay as the XDNA backend's host ops read them: the CPU's repacked kernels round differently, which
// moved one 9B decode step 0.17 nmse away from both the unpacked CPU and the NPU.
static llama_model * load_on_cpu(const char * model_path) {
    llama_model_params mp      = llama_model_default_params();
    ggml_backend_dev_t none[1] = { nullptr };
    mp.devices                 = none;
    mp.n_gpu_layers            = 0;
    mp.use_extra_bufts         = false;
    llama_model * cpu          = llama_model_load_from_file(model_path, mp);
    if (!cpu) {
        std::printf("the CPU-only model did not load\n");
    }
    return cpu;
}

// The straight run's steps on the CPU alone, fed the straight run's tokens: its logits against the run's.
static bool check_cpu_reference(llama_model * cpu, std::vector<llama_token> & p, int nv, const trace & npu) {
    return cpu && same_logits("straight run against the CPU", straight(cpu, p, nv, &npu.tokens), npu);
}

// The restored rows left stale in memory once (the hook's count), so only a flush of every one of them reads them right.
static bool check_restore_longer(llama_model * model, llama_model * cpu, const llama_vocab * v, int nv) {
    const stale_kv_fn stale   = stale_kv_hook(xdna_device());
    const int         planted = stale ? stale(0) : 0;
    const trace       got     = restore_longer(model, v, nv, nullptr, stale);
    const int         left    = stale ? stale(0) - planted : 0;
    std::printf("restores left stale in memory: %d\n", left);
    const bool ok = cpu && same_logits("a longer state restored after a shorter decode, against the CPU",
                                       restore_longer(cpu, v, nv, &got.tokens, nullptr), got);
    return ok && left == 1;
}

constexpr int    UNIFIED_PROMPT_TOKENS = 40;
constexpr int    UNIFIED_FIRST_STEPS   = 8;   // sequence 0 from cell 0: its K/V row list is the identity
constexpr int    UNIFIED_SECOND_STEPS  = 16;  // sequence 1 in the cells after sequence 0's: a shifted list
constexpr int    UNIFIED_RESUME_STEPS  = 4;   // sequence 0 again, past sequence 1's cells: a list with a gap
constexpr int    UNIFIED_SEQS          = 2;
constexpr int    UNIFIED_CTX           = 1024;
constexpr double UNIFIED_NMSE_MAX      = 5e-2;  // the 4B reaches 0.028; a list the NPU ignores, 0.55 and more
constexpr char   KV_ROWS_ON[]          = "Flash Attention over K/V rows enabled";
constexpr char   LAYER_BINDS[]         = "xdna-att-layer: layer ";
constexpr char   PROMPT_A[] = "A gated delta net layer keeps a recurrent state that it updates with every token. ";
constexpr char   PROMPT_B[] = "The lighthouse keeper wrote down every ship that passed during the long winter. ";

// What llama and the backend log while the unified-cache run goes: the row-list line, the fused attention layer's
// binds while sequence 0 decodes from cell 0, and any error.
struct unified_log {
    bool rows_on     = false;
    bool in_first    = false;
    int  layer_binds = 0;
    int  errors      = 0;
};

static void on_unified_log(ggml_log_level level, const char * text, void * ud) {
    unified_log * st = (unified_log *) ud;
    st->rows_on      = st->rows_on || std::strstr(text, KV_ROWS_ON) != nullptr;
    st->layer_binds += st->in_first && std::strstr(text, LAYER_BINDS) != nullptr;
    st->errors += level == GGML_LOG_LEVEL_ERROR;
    if (level >= GGML_LOG_LEVEL_WARN) {
        std::fputs(text, stderr);
    }
}

// A sequence of the unified cache: its id, the position its next token takes and its last logits.
struct seq_cursor {
    llama_seq_id       id  = 0;
    llama_pos          pos = 0;
    std::vector<float> logits;
};

static void fill_batch(llama_batch & b, const seq_cursor & s, const llama_token * tokens, int n) {
    for (int i = 0; i < n; i++) {
        b.token[i]     = tokens[i];
        b.pos[i]       = s.pos + i;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = s.id;
        b.logits[i]    = i == n - 1;
    }
    b.n_tokens = n;
}

// `n` tokens of sequence `s` in one batch at its next positions, and its logits after the last.
static bool decode_seq(llama_context * ctx, int nv, seq_cursor & s, const llama_token * tokens, int n) {
    llama_batch b = llama_batch_init(n, 0, 1);
    fill_batch(b, s, tokens, n);
    const bool ok = llama_decode(ctx, b) == 0;
    llama_batch_free(b);
    if (ok) {
        const float * l = llama_get_logits_ith(ctx, -1);
        s.logits.assign(l, l + nv);
        s.pos += n;
    }
    return ok;
}

// `n` tokens of sequence `s`, the greedy pick of its last logits or the next of `forced`, each decoded on its own,
// with the logits after it.
static void seq_steps(llama_context *                  ctx,
                      int                              nv,
                      seq_cursor &                     s,
                      int                              n,
                      const std::vector<llama_token> * forced,
                      trace &                          tr) {
    const int m = forced ? std::min(n, (int) forced->size()) : n;
    for (int i = 0; i < m; i++) {
        const llama_token t = forced ? (*forced)[i] : greedy_of(s.logits.data(), (int) s.logits.size());
        tr.tokens.push_back(t);
        if (!decode_seq(ctx, nv, s, &t, 1)) {
            return;
        }
        tr.logits.push_back(s.logits);
    }
}

struct unified_traces {
    trace first;    // sequence 0 from cell 0
    trace second;   // sequence 1 behind sequence 0's cells
    trace resumed;  // sequence 0 past sequence 1's cells
};

// Two sequences in one unified cache, as llama-server's default slots: sequence 0's prompt and UNIFIED_FIRST_STEPS
// tokens from cell 0, sequence 1's prompt and UNIFIED_SECOND_STEPS tokens in the cells after them, then
// UNIFIED_RESUME_STEPS more tokens of sequence 0. Greedy, or each phase fed `forced`'s tokens; `log` watches the run.
static unified_traces unified_sequences(llama_model *          model,
                                        const llama_vocab *    v,
                                        int                    nv,
                                        const unified_traces * forced,
                                        unified_log *          log) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx                = UNIFIED_CTX;
    cp.n_seq_max            = UNIFIED_SEQS;
    cp.kv_unified           = true;
    if (log) {
        llama_log_set(on_unified_log, log);
    }
    llama_context *                ctx = llama_init_from_model(model, cp);
    const std::vector<llama_token> a   = tokens_of(v, PROMPT_A, UNIFIED_PROMPT_TOKENS);
    const std::vector<llama_token> b   = tokens_of(v, PROMPT_B, UNIFIED_PROMPT_TOKENS);
    seq_cursor                     s0, s1;
    s1.id = 1;
    unified_traces tr;
    if (ctx && decode_seq(ctx, nv, s0, a.data(), (int) a.size())) {
        if (log) {
            log->in_first = true;
        }
        seq_steps(ctx, nv, s0, UNIFIED_FIRST_STEPS, forced ? &forced->first.tokens : nullptr, tr.first);
        if (log) {
            log->in_first = false;
        }
        if (decode_seq(ctx, nv, s1, b.data(), (int) b.size())) {
            seq_steps(ctx, nv, s1, UNIFIED_SECOND_STEPS, forced ? &forced->second.tokens : nullptr, tr.second);
            seq_steps(ctx, nv, s0, UNIFIED_RESUME_STEPS, forced ? &forced->resumed.tokens : nullptr, tr.resumed);
        }
    }
    if (ctx) {
        llama_free(ctx);
    }
    if (log) {
        llama_log_set(nullptr, nullptr);
    }
    return tr;
}

// Whether GGML_XDNA_ATTN_LAYER leaves the fused attention layer on, read as the backend reads it.
static bool attn_layer_on(void) {
    const char * e = getenv("GGML_XDNA_ATTN_LAYER");
    return !e || std::strtol(e, nullptr, 10) != 0;
}

// The unified-cache run against a CPU-only one fed its tokens: every phase within UNIFIED_NMSE_MAX, with the row
// lists on, the fused attention layer run while sequence 0 held the cells from 0 (unless GGML_XDNA_ATTN_LAYER=0
// leaves the lists to the pool), and no error logged.
static bool check_unified_sequences(llama_model * model, llama_model * cpu, const llama_vocab * v, int nv) {
    unified_log          log;
    const unified_traces got      = unified_sequences(model, v, nv, nullptr, &log);
    const bool           layer_on = attn_layer_on();
    std::printf("unified cache: K/V row lists %s, %d fused attention layer binds from cell 0%s, %d errors\n",
                log.rows_on ? "on" : "off (the check did not run)", log.layer_binds,
                layer_on ? "" : " (the layer is off)", log.errors);
    bool ok = cpu != nullptr;
    if (cpu) {
        const unified_traces want = unified_sequences(cpu, v, nv, &got, nullptr);
        ok = same_logits("sequence 0 from cell 0, against the CPU", want.first, got.first, UNIFIED_NMSE_MAX);
        ok = same_logits("sequence 1 behind sequence 0's cells, against the CPU", want.second, got.second,
                         UNIFIED_NMSE_MAX) &&
             ok;
        ok = same_logits("sequence 0 past sequence 1's cells, against the CPU", want.resumed, got.resumed,
                         UNIFIED_NMSE_MAX) &&
             ok;
    }
    return ok && log.rows_on && (log.layer_binds > 0 || !layer_on) && log.errors == 0;
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

    const trace                    straight_run = straight(model, p, nv, nullptr);
    const std::vector<llama_token> & want       = straight_run.tokens;

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
    const std::vector<llama_token> got = gen(ctx, nv, STRAIGHT_TOKENS);
    llama_free(ctx);
    std::printf("state %zu bytes, restored %zu\n", st.size(), set);

    std::string sw, sg;
    for (llama_token t : want) sw += std::to_string(t) + " ";
    for (llama_token t : got) sg += std::to_string(t) + " ";
    std::printf("straight: %s\nrestored: %s\n", sw.c_str(), sg.c_str());
    const host_chunk_count_fn chunks = host_chunk_counter(xdna_device());
    bool                      ok     = want == got;
    if (getenv("ON_CPU") == nullptr) {
        llama_model * cpu = load_on_cpu(model_path);
        ok                = check_cpu_reference(cpu, p, nv, straight_run) && ok;
        ok                = check_restore_longer(model, cpu, v, nv) && ok;
        ok                = check_unified_sequences(model, cpu, v, nv) && ok;
        if (cpu) {
            llama_model_free(cpu);
        }
        ok = check_follow_up(model, v, nv, chunks) && ok;
        ok = check_four_sequences(model, v, nv, chunks) && ok;
    }
    std::printf("test-xdna-state: %s\n", ok ? "passed" : "FAILED");
    llama_model_free(model);
    llama_backend_free();
    return ok ? 0 : 1;
}
