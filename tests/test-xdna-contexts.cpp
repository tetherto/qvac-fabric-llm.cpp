// Two llama contexts of one model on the XDNA NPU, computing at the same time.
//
//   test-xdna-contexts -m model.gguf [rounds]
//
// Each context first runs alone: its prompt, then 24 greedy tokens - the
// reference. Then both run together, each on its own thread, the threads
// released at a barrier before every llama_decode so that their graph
// computes overlap, the prefill of one against the prefill or the decode of
// the other. Each context has to produce its reference tokens again, round
// after round, and every decode has to succeed. The backend runs one graph
// compute at a time and keeps each context's fused-layer sessions apart;
// without either, the second context crashed or decoded on the first one's
// recurrent state.
//
// Skips (exit 0) without a model or an XDNA device.

#include "llama.h"

#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
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

struct barrier {
    std::mutex              m;
    std::condition_variable cv;
    int                     n, waiting = 0, gen = 0;
    explicit barrier(int n) : n(n) {}
    void wait() {
        std::unique_lock<std::mutex> lk(m);
        const int g = gen;
        if (++waiting == n) {
            waiting = 0;
            gen++;
            cv.notify_all();
        } else {
            cv.wait(lk, [&] { return gen != g; });
        }
    }
};

static const int N_GEN = 24;

// prompt, then N_GEN greedy tokens; `sync` (if any) is waited on before every decode
static bool generate(llama_model * model, const std::string & prompt, std::vector<llama_token> & out, barrier * sync) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx                = 1024;
    cp.n_batch              = 512;
    cp.n_ubatch             = 512;
    llama_context * ctx     = llama_init_from_model(model, cp);
    bool            ok      = ctx != nullptr;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(512);
    const int n = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), toks.data(), (int) toks.size(), true, false);
    toks.resize(n > 0 ? n : 0);
    ok = ok && !toks.empty();
    if (sync) {
        sync->wait();
    }
    ok = ok && llama_decode(ctx, llama_batch_get_one(toks.data(), (int) toks.size())) == 0;
    const int n_vocab = llama_vocab_n_tokens(vocab);
    for (int i = 0; i < N_GEN; i++) {
        llama_token best = 0;
        if (ok) {
            const float * logits = llama_get_logits_ith(ctx, -1);
            for (llama_token t = 1; t < n_vocab; t++) {
                best = logits[t] > logits[best] ? t : best;
            }
            out.push_back(best);
        }
        if (sync) {
            sync->wait();  // the other thread decodes at the same moment, even after a failure
        }
        ok = ok && llama_decode(ctx, llama_batch_get_one(&best, 1)) == 0;
    }
    if (ctx) {
        llama_free(ctx);
    }
    return ok;
}

static std::string str(const std::vector<llama_token> & v) {
    std::string s;
    for (llama_token t : v) {
        s += std::to_string(t) + " ";
    }
    return s;
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    int          rounds     = 3;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else {
            rounds = std::atoi(argv[i]);
        }
    }
    llama_backend_init();
    ggml_backend_dev_t dev = xdna_device();
    if (!model_path || !dev) {
        std::printf("test-xdna-contexts: no model or no XDNA device, skipping\n");
        llama_backend_free();
        return 0;
    }
    llama_model_params mp      = llama_model_default_params();
    ggml_backend_dev_t devs[2] = { dev, nullptr };
    mp.devices                 = devs;
    mp.n_gpu_layers            = 99;
    llama_model * model        = llama_model_load_from_file(model_path, mp);
    if (!model) {
        return 1;
    }

    // a short prompt and a ~200-token one, so a prefill overlaps a decode
    std::string pa = "The capital of France is";
    std::string pb;
    for (int r = 0; r < 12; r++) {
        pb += "The river that runs through the old town carries boats and stories downstream. ";
    }
    std::vector<llama_token> ref_a, ref_b;
    const bool ra = generate(model, pa, ref_a, nullptr);
    const bool rb = generate(model, pb, ref_b, nullptr);
    std::printf("alone A: %s| %s\nalone B: %s| %s\n", str(ref_a).c_str(), ra ? "ok" : "FAILED", str(ref_b).c_str(),
                rb ? "ok" : "FAILED");
    int failures = !ra + !rb;

    for (int r = 0; r < rounds; r++) {
        barrier                  sync(2);
        std::vector<llama_token> ta, tb;
        bool                     oa = false, ob = false;
        std::thread              a([&] { oa = generate(model, pa, ta, &sync); });
        std::thread              b([&] { ob = generate(model, pb, tb, &sync); });
        a.join();
        b.join();
        const bool same_a = ta == ref_a, same_b = tb == ref_b;
        std::printf("round %d: A %s%s, B %s%s\n", r, oa ? "decoded" : "FAILED to decode",
                    same_a ? ", as alone" : ", DIFFERS from alone", ob ? "decoded" : "FAILED to decode",
                    same_b ? ", as alone" : ", DIFFERS from alone");
        if (!same_a) {
            std::printf("  A: %s\n", str(ta).c_str());
        }
        if (!same_b) {
            std::printf("  B: %s\n", str(tb).c_str());
        }
        failures += !oa + !ob + !same_a + !same_b;
    }
    llama_model_free(model);
    llama_backend_free();
    std::printf("test-xdna-contexts: %s\n", failures ? "FAILED" : "passed");
    return failures ? 1 : 0;
}
