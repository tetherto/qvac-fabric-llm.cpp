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
// Then a longer sequence's state restored after a shorter one was decoded: the
// restore writes cache rows below the ones that decode left on the device, and
// a test hook keeps them stale in memory until a flush. Its logits after every
// token are compared with a CPU-only run of the same steps, fed its tokens.
//
// Then two sequences in one unified cache, as llama-server's default slots:
// the first from cell 0, whose decode attention reads the identity K/V row
// list the fused attention layer takes; the second in the cells after the
// first's; and the first again, past the second's cells. The fused layer has
// to leave those two lists to the host. Each step's logits are compared with
// a CPU-only run fed the same tokens.
#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static ggml_backend_dev_t xdna_device(void) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "XDNA") == 0) {
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

constexpr int    RESTORE_LONG_TOKENS  = 150;
constexpr int    RESTORE_SHORT_TOKENS = 90;
// decoded after the short prompt: its rows on the device stop below the long one's
constexpr int    RESTORE_SHORT_STEPS  = 10;
constexpr int    RESTORE_STEPS        = 16;
constexpr double RESTORE_NMSE_MAX     = 2e-2;

typedef int (*stale_kv_fn)(int arm);

// The backend's hook that leaves the K/V rows the host next writes stale in memory until a flush, and counts the times it
// did; null on a backend without it.
static stale_kv_fn stale_kv_hook(ggml_backend_dev_t dev) {
    return dev ? (stale_kv_fn) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev),
                                                                 "ggml_backend_xdna_test_stale_kv") :
                 nullptr;
}

// The tokens a run decoded one at a time and the logits after each.
struct trace {
    std::vector<llama_token>        tokens;
    std::vector<std::vector<float>> logits;
};

static std::vector<llama_token> tokenize(const llama_vocab * v, const std::string & text) {
    std::vector<llama_token> t(text.size() + 2);
    t.resize(llama_tokenize(v, text.c_str(), (int) text.size(), t.data(), (int) t.size(), true, false));
    return t;
}

// the first `n` tokens of `text` repeated
static std::vector<llama_token> tokens_of(const llama_vocab * v, const std::string & text, int n) {
    std::string              s;
    std::vector<llama_token> t;
    while ((int) t.size() < n) {
        s += text;
        t = tokenize(v, s);
    }
    t.resize(n);
    return t;
}

static void snapshot(llama_context * ctx, int nv, trace & tr) {
    const float * l = llama_get_logits_ith(ctx, -1);
    tr.logits.emplace_back(l, l + nv);
}

// `n` tokens, greedy or the next of `forced`, each decoded on its own, with the logits after it.
static void decode_steps(llama_context * ctx, int nv, int n, const std::vector<llama_token> * forced, trace & tr) {
    const int m = forced ? std::min(n, (int) forced->size()) : n;
    for (int i = 0; i < m; i++) {
        llama_token t = forced ? (*forced)[i] : greedy(ctx, nv);
        tr.tokens.push_back(t);
        if (llama_decode(ctx, llama_batch_get_one(&t, 1)) != 0) {
            return;
        }
        snapshot(ctx, nv, tr);
    }
}

// A long prompt's sequence saved and cleared, a short prompt prefilled, decoded RESTORE_SHORT_STEPS tokens and cleared,
// the long one restored (llama_state_seq_set_data, with `stale` armed when given), then one more token and RESTORE_STEPS
// greedy or `forced` ones: the logits after each.
static trace restore_longer(llama_model *                    model,
                            const llama_vocab *              v,
                            int                              nv,
                            const std::vector<llama_token> * forced,
                            stale_kv_fn                      stale) {
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
    trace tr;
    if (llama_decode(ctx, llama_batch_get_one(&longp.front(), 1)) == 0) {
        snapshot(ctx, nv, tr);
        decode_steps(ctx, nv, RESTORE_STEPS, forced, tr);
    }
    llama_free(ctx);
    return tr;
}

// by the exponent bits, so a build with fast-math still sees a NaN or an infinity
static bool non_finite(float x) {
    uint32_t b;
    std::memcpy(&b, &x, sizeof(b));
    return (b & 0x7f800000u) == 0x7f800000u;
}

static size_t count_non_finite(const std::vector<float> & l) {
    size_t n = 0;
    for (float x : l) {
        n += non_finite(x);
    }
    return n;
}

static double nmse(const std::vector<float> & want, const std::vector<float> & got) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < want.size(); i++) {
        num += (double) (want[i] - got[i]) * (want[i] - got[i]);
        den += (double) want[i] * want[i];
    }
    return den > 0.0 ? num / den : num;
}

// Every snapshot finite and within `tol` of the reference's.
static bool same_logits(const char * what, const trace & want, const trace & got, double tol) {
    if (want.logits.size() != got.logits.size() || want.logits.empty()) {
        std::printf("%s: %zu snapshots against %zu\n", what, want.logits.size(), got.logits.size());
        return false;
    }
    double worst = 0.0;
    size_t at = 0, bad = 0;
    for (size_t i = 0; i < want.logits.size(); i++) {
        const double e = nmse(want.logits[i], got.logits[i]);
        bad += count_non_finite(got.logits[i]) + count_non_finite(want.logits[i]);
        if (e > worst) {
            worst = e;
            at    = i;
        }
    }
    std::printf("%s: worst nmse %.3g at step %zu of %zu, %zu non-finite logits (limit %.0e)\n", what, worst, at,
                want.logits.size(), bad, tol);
    return bad == 0 && worst < tol;
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

// The restored rows left stale in memory once (the hook's count), so only a flush of every one of them reads them right.
static bool check_restore_longer(llama_model * model, const char * model_path, const llama_vocab * v, int nv) {
    const stale_kv_fn stale   = stale_kv_hook(xdna_device());
    const int         planted = stale ? stale(0) : 0;
    const trace       got     = restore_longer(model, v, nv, nullptr, stale);
    const int         left    = stale ? stale(0) - planted : 0;
    std::printf("restores left stale in memory: %d\n", left);
    llama_model * cpu = load_on_cpu(model_path);
    const bool    ok  = cpu && same_logits("a longer state restored after a shorter decode, against the CPU",
                                           restore_longer(cpu, v, nv, &got.tokens, nullptr), got, RESTORE_NMSE_MAX);
    if (cpu) {
        llama_model_free(cpu);
    }
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
static bool check_unified_sequences(llama_model * model, const char * model_path, const llama_vocab * v, int nv) {
    unified_log          log;
    const unified_traces got      = unified_sequences(model, v, nv, nullptr, &log);
    const bool           layer_on = attn_layer_on();
    std::printf("unified cache: K/V row lists %s, %d fused attention layer binds from cell 0%s, %d errors\n",
                log.rows_on ? "on" : "off (the check did not run)", log.layer_binds,
                layer_on ? "" : " (the layer is off)", log.errors);
    llama_model * cpu = load_on_cpu(model_path);
    bool          ok  = cpu != nullptr;
    if (cpu) {
        const unified_traces want = unified_sequences(cpu, v, nv, &got, nullptr);
        ok = same_logits("sequence 0 from cell 0, against the CPU", want.first, got.first, UNIFIED_NMSE_MAX);
        ok = same_logits("sequence 1 behind sequence 0's cells, against the CPU", want.second, got.second,
                         UNIFIED_NMSE_MAX) &&
             ok;
        ok = same_logits("sequence 0 past sequence 1's cells, against the CPU", want.resumed, got.resumed,
                         UNIFIED_NMSE_MAX) &&
             ok;
        llama_model_free(cpu);
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
    bool ok = want == got;
    if (getenv("ON_CPU") == nullptr) {
        ok = check_restore_longer(model, model_path, v, nv) && ok;
        ok = check_unified_sequences(model, model_path, v, nv) && ok;
    }
    std::printf("test-xdna-state: %s\n", ok ? "passed" : "FAILED");
    llama_model_free(model);
    llama_backend_free();
    return ok ? 0 : 1;
}
