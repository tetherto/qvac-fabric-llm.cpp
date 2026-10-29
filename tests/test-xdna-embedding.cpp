// An embedding model on XDNA against the CPU backend: every layer's output and the pooled embeddings.
// test-xdna-embedding -m model.gguf; skips (exit 0) without a model or an XDNA device.
#include "ggml-backend.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

static const int    N_CTX          = 2048;
static const double MIN_COSINE     = 0.998;
static const double MAX_STAGE_NMSE = 5e-2;  // the NPU GEMMs' rounding reaches ~1.5e-2; a wrong op or buffer is O(0.1-1)
static const int    TOKEN_COUNTS[] = { 1, 8, 40, 70, 600, 1100 };
static const int    N_INPUTS       = sizeof(TOKEN_COUNTS) / sizeof(TOKEN_COUNTS[0]);
static const int    N_SEQ_MAX      = 8;  // divides N_CTX: the worst-case reserve rounds n_tokens up to a multiple of it
static const int    CHARS_A_TOKEN  = 8;  // at least this many characters of the corpus per wanted token
static const char   LAYER_OUT[]    = "l_out-";
static const char   POOLED_IN[]    = "result_embd";

static_assert(N_INPUTS <= N_SEQ_MAX && N_CTX % N_SEQ_MAX == 0, "the packed batch has to fit the context");

using stage_map = std::map<std::string, std::vector<float>>;
using embedding = std::vector<float>;
using tokens    = std::vector<llama_token>;

// one backend's model, a context on the production path and one that captures the stages
struct side {
    llama_model *   model = nullptr;
    llama_context * plain = nullptr;
    llama_context * trace = nullptr;
    stage_map       stages;
};

static ggml_backend_dev_t xdna_device(void) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "XDNA") == 0) {
            return dev;
        }
    }
    return nullptr;
}

static const char * model_arg(int argc, char ** argv) {
    for (int i = 1; i + 1 < argc; i++) {
        if (std::strcmp(argv[i], "-m") == 0) {
            return argv[i + 1];
        }
    }
    return nullptr;
}

static bool is_stage(const struct ggml_tensor * t) {
    const bool named = std::strncmp(t->name, LAYER_OUT, sizeof(LAYER_OUT) - 1) == 0 || std::strcmp(t->name, POOLED_IN) == 0;
    return named && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t);
}

static bool capture_stage(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) {
        return is_stage(t);
    }
    std::vector<float> & v = (*(stage_map *) user_data)[t->name];
    v.resize(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    return true;
}

static llama_context * make_context(llama_model * model, stage_map * stages) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx                = N_CTX;
    cp.n_batch              = N_CTX;
    cp.n_ubatch             = N_CTX;
    cp.n_seq_max            = N_SEQ_MAX;
    cp.embeddings           = true;
    cp.flash_attn_type      = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (stages) {
        cp.cb_eval           = capture_stage;
        cp.cb_eval_user_data = stages;
    }
    return llama_init_from_model(model, cp);
}

// no device: the CPU backend alone
static bool open_side(side & s, const char * path, ggml_backend_dev_t dev) {
    llama_model_params mp      = llama_model_default_params();
    ggml_backend_dev_t devs[2] = { dev, nullptr };
    mp.devices                 = devs;
    mp.n_gpu_layers            = dev ? 99 : 0;
    s.model                    = llama_model_load_from_file(path, mp);
    s.plain                    = s.model ? make_context(s.model, nullptr) : nullptr;
    s.trace                    = s.model ? make_context(s.model, &s.stages) : nullptr;
    return s.plain && s.trace;
}

static void close_side(side & s) {
    llama_free(s.trace);
    llama_free(s.plain);
    llama_model_free(s.model);
}

static tokens tokenize_corpus(const llama_vocab * vocab, int n_tokens) {
    std::string text;
    for (int i = 0; (int) text.size() < CHARS_A_TOKEN * n_tokens; i++) {
        text += "Passage " + std::to_string(i) + ": the neural processing unit computes matrix products while " +
                "the host runs normalisation, rotary position encoding and attention over every token. ";
    }
    tokens toks(text.size() + 2);
    toks.resize(llama_tokenize(vocab, text.c_str(), (int) text.size(), toks.data(), (int) toks.size(), true, false));
    return toks;
}

// prefixes of one corpus, one per entry of TOKEN_COUNTS; empty when the corpus is too short
static std::vector<tokens> make_inputs(const llama_vocab * vocab) {
    const tokens        corpus = tokenize_corpus(vocab, TOKEN_COUNTS[N_INPUTS - 1]);
    std::vector<tokens> inputs;
    for (int n : TOKEN_COUNTS) {
        if ((size_t) n > corpus.size()) {
            return {};
        }
        inputs.emplace_back(corpus.begin(), corpus.begin() + n);
    }
    return inputs;
}

static void add_sequence(llama_batch & b, const tokens & toks, llama_seq_id seq) {
    for (size_t i = 0; i < toks.size(); i++) {
        b.token[b.n_tokens]     = toks[i];
        b.pos[b.n_tokens]       = (llama_pos) i;
        b.n_seq_id[b.n_tokens]  = 1;
        b.seq_id[b.n_tokens][0] = seq;
        b.logits[b.n_tokens]    = true;
        b.n_tokens++;
    }
}

static std::vector<embedding> pooled(llama_context * ctx, int n_seqs) {
    const int              n_out = llama_model_n_embd_out(llama_get_model(ctx));
    std::vector<embedding> out;
    for (int s = 0; s < n_seqs; s++) {
        const float * e = llama_get_embeddings_seq(ctx, s);
        out.emplace_back(e, e + n_out);
    }
    return out;
}

// one sequence a slot, all in one batch; empty on a failed decode
static std::vector<embedding> embed(llama_context * ctx, const std::vector<tokens> & seqs) {
    llama_batch b = llama_batch_init(N_CTX, 0, (int) seqs.size());
    for (size_t s = 0; s < seqs.size(); s++) {
        add_sequence(b, seqs[s], (llama_seq_id) s);
    }
    llama_memory_clear(llama_get_memory(ctx), true);
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    return rc == 0 ? pooled(ctx, (int) seqs.size()) : std::vector<embedding>();
}

static double cosine(const embedding & a, const embedding & b) {
    double ab = 0, aa = 0, bb = 0;
    for (size_t i = 0; i < a.size(); i++) {
        ab += (double) a[i] * b[i];
        aa += (double) a[i] * a[i];
        bb += (double) b[i] * b[i];
    }
    return ab / std::sqrt(aa * bb);
}

static double nmse(const std::vector<float> & ref, const std::vector<float> & got) {
    double err = 0, sq = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        if (!std::isfinite(got[i])) {
            return INFINITY;
        }
        err += ((double) ref[i] - got[i]) * ((double) ref[i] - got[i]);
        sq += (double) ref[i] * ref[i];
    }
    return err / sq;
}

static std::vector<std::string> stage_names(const llama_model * model) {
    std::vector<std::string> names;
    for (int il = 0; il < llama_model_n_layer(model); il++) {
        names.push_back(LAYER_OUT + std::to_string(il));
    }
    names.push_back(POOLED_IN);
    return names;
}

// every stage's NMSE in graph order; names the first stage over the bound and returns the failures
static int compare_stages(const std::vector<std::string> & names, const stage_map & ref, const stage_map & got) {
    int         failures = 0;
    double      worst    = 0;
    std::string trace;
    for (const std::string & name : names) {
        const auto r = ref.find(name);
        const auto g = got.find(name);
        if (r == ref.end() || g == got.end() || r->second.size() != g->second.size()) {
            std::printf("    %s missing (cpu %d, xdna %d)\n", name.c_str(), r != ref.end(), g != got.end());
            failures++;
            continue;
        }
        const double e = nmse(r->second, g->second);
        char         buf[16];
        std::snprintf(buf, sizeof(buf), " %.1e", e);
        trace += buf;
        worst = std::fmax(worst, e);
        if (!(e <= MAX_STAGE_NMSE) && failures++ == 0) {
            std::printf("    first stage over %.0e: %s nmse %.3g\n", MAX_STAGE_NMSE, name.c_str(), e);
        }
    }
    std::printf("    stages %zu, worst nmse %.3g, in order:%s\n", names.size(), worst, trace.c_str());
    return failures;
}

static int check_stages(side & cpu, side & npu, const std::vector<std::string> & names, const tokens & toks) {
    cpu.stages.clear();
    npu.stages.clear();
    if (embed(cpu.trace, { toks }).empty() || embed(npu.trace, { toks }).empty()) {
        std::printf("    decode failed\n");
        return 1;
    }
    return compare_stages(names, cpu.stages, npu.stages);
}

// the production path (no callback): pooled cosine against the CPU, and a second XDNA run bit for bit
static int check_pooled(side & cpu, side & npu, const tokens & toks, embedding & cpu_out) {
    const std::vector<embedding> c  = embed(cpu.plain, { toks });
    const std::vector<embedding> n1 = embed(npu.plain, { toks });
    const std::vector<embedding> n2 = embed(npu.plain, { toks });
    if (c.empty() || n1.empty() || n2.empty()) {
        std::printf("    decode failed\n");
        return 1;
    }
    cpu_out              = c[0];
    const double cos     = cosine(c[0], n1[0]);
    const bool   repeats = n1[0] == n2[0];
    std::printf("    pooled cosine %.6f %s, repeat %s\n", cos, cos >= MIN_COSINE ? "ok" : "FAIL",
                repeats ? "identical" : "DIFFERS");
    return (cos >= MIN_COSINE ? 0 : 1) + (repeats ? 0 : 1);
}

// each input alone; fills the CPU's pooled embedding of each
static int check_inputs(side & cpu, side & npu, const std::vector<tokens> & inputs, std::vector<embedding> & cpu_single) {
    const std::vector<std::string> names    = stage_names(cpu.model);
    int                            failures = 0;
    for (size_t s = 0; s < inputs.size(); s++) {
        std::printf("input %zu: %zu tokens\n", s, inputs[s].size());
        failures += check_stages(cpu, npu, names, inputs[s]);
        failures += check_pooled(cpu, npu, inputs[s], cpu_single[s]);
    }
    return failures;
}

// all inputs as the sequences of one batch, against each one's CPU embedding alone
static int check_packed(side & npu, const std::vector<tokens> & inputs, const std::vector<embedding> & cpu_single) {
    const std::vector<embedding> packed = embed(npu.plain, inputs);
    if (packed.empty()) {
        std::printf("packed: decode failed\n");
        return 1;
    }
    int failures = 0;
    for (size_t s = 0; s < inputs.size(); s++) {
        const double cos = cosine(cpu_single[s], packed[s]);
        failures += cos >= MIN_COSINE ? 0 : 1;
        std::printf("packed seq %zu (%zu tokens): cosine %.6f %s\n", s, inputs[s].size(), cos,
                    cos >= MIN_COSINE ? "ok" : "FAIL");
    }
    return failures;
}

int main(int argc, char ** argv) {
    const char * model_path = model_arg(argc, argv);
    llama_backend_init();
    ggml_backend_dev_t dev = xdna_device();
    if (!model_path || !dev) {
        std::printf("test-xdna-embedding: no model or no XDNA device, skipping\n");
        llama_backend_free();
        return 0;
    }
    side cpu, npu;
    int  failures = open_side(cpu, model_path, nullptr) && open_side(npu, model_path, dev) ? 0 : 1;
    if (failures == 0) {
        const std::vector<tokens> inputs = make_inputs(llama_model_get_vocab(cpu.model));
        std::vector<embedding>    cpu_single(inputs.size());
        failures += inputs.empty() ? 1 : check_inputs(cpu, npu, inputs, cpu_single);
        failures += inputs.empty() ? 0 : check_packed(npu, inputs, cpu_single);
    }
    std::printf("test-xdna-embedding: %s\n", failures == 0 ? "passed" : "FAILED");
    close_side(npu);
    close_side(cpu);
    llama_backend_free();
    return failures == 0 ? 0 : 1;
}
