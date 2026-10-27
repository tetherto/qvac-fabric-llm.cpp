// Load, run and free a model on the XDNA NPU several times in one process
// (#293: repeated model loads).
//
//   test-xdna-reload -m model.gguf [cycles]
//
// Every cycle loads the model onto the NPU, generates 16 greedy tokens and
// frees the context and the model. The tokens have to be the same every
// cycle - state kept from an earlier model (fused-layer sessions, packed
// weights keyed by an address the new model may reuse) would change them -
// and the device buffers the process holds once a cycle is freed must not
// grow from one cycle to the next. Each XRT buffer object is a mapping of
// the accel device node, so /proc/self/maps counts them without any hook in
// the backend.
//
// Skips (exit 0) when no XDNA device is registered.

#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static int device_mappings(void) {
    std::ifstream f("/proc/self/maps");
    std::string   line;
    int           n = 0;
    while (std::getline(f, line)) {
        n += line.find("/dev/accel/") != std::string::npos;
    }
    return n;
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

static bool generate(llama_model * model, std::vector<llama_token> & out) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx                = 1024;
    cp.n_batch              = 512;
    cp.n_ubatch             = 512;
    cp.flash_attn_type      = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * ctx     = llama_init_from_model(model, cp);
    if (!ctx) {
        return false;
    }
    const llama_vocab * vocab  = llama_model_get_vocab(model);
    const char *        prompt = "The capital of France is";
    std::vector<llama_token> toks(64);
    const int n = llama_tokenize(vocab, prompt, (int) std::strlen(prompt), toks.data(), (int) toks.size(), true, false);
    toks.resize(n > 0 ? n : 0);

    bool ok = !toks.empty() && llama_decode(ctx, llama_batch_get_one(toks.data(), (int) toks.size())) == 0;
    const int n_vocab = llama_vocab_n_tokens(vocab);
    for (int i = 0; ok && i < 16; i++) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        llama_token   best   = 0;
        for (llama_token t = 1; t < n_vocab; t++) {
            best = logits[t] > logits[best] ? t : best;
        }
        out.push_back(best);
        ok = llama_decode(ctx, llama_batch_get_one(&best, 1)) == 0;
    }
    llama_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    int          cycles     = 3;
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else {
            cycles = std::atoi(argv[i]);
        }
    }
    if (!model_path) {
        model_path = common_get_model_or_exit(argc, argv);
    }

    llama_backend_init();
    // XDNA_RELOAD_CPU=1 runs the same cycles on the CPU, as the reference.
    ggml_backend_dev_t dev = std::getenv("XDNA_RELOAD_CPU") ? ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)
                                                              : xdna_device();
    if (!dev) {
        std::printf("test-xdna-reload: no XDNA device, skipping\n");
        llama_backend_free();
        return 0;
    }

    std::vector<llama_token> first;
    int                      failures = 0;
    int                      held[64] = {};
    cycles                            = cycles < 2 ? 2 : (cycles > 64 ? 64 : cycles);
    for (int c = 0; c < cycles; c++) {
        llama_model_params mp  = llama_model_default_params();
        ggml_backend_dev_t devs[2] = { dev, nullptr };
        mp.devices             = devs;
        mp.n_gpu_layers        = std::getenv("XDNA_RELOAD_CPU") ? 0 : 99;
        llama_model * model    = llama_model_load_from_file(model_path, mp);
        if (!model) {
            std::printf("cycle %d: model load failed\n", c);
            return 1;
        }
        std::vector<llama_token> toks;
        const bool               ok = generate(model, toks);
        llama_model_free(model);
        held[c] = device_mappings();

        std::string s;
        for (llama_token t : toks) {
            s += std::to_string(t) + " ";
        }
        std::printf("cycle %d: %s| %d device buffers held after free\n", c, s.c_str(), held[c]);
        if (!ok) {
            std::printf("FAIL: cycle %d did not decode\n", c);
            failures++;
        }
        if (c == 0) {
            first = toks;
        } else if (toks != first) {
            std::printf("FAIL: cycle %d generated different tokens than cycle 0\n", c);
            failures++;
        }
        // The first cycle leaves the process-lifetime state behind (the kernel
        // pool, the prefill runners). Everything built from a model goes with
        // it, so no later cycle may hold more than the first did.
        if (c >= 1 && held[c] > held[0]) {
            std::printf("FAIL: cycle %d holds %d device buffers after free, cycle 0 held %d\n", c, held[c],
                        held[0]);
            failures++;
        }
    }
    llama_backend_free();
    std::printf("test-xdna-reload: %s\n", failures ? "FAILED" : "passed");
    return failures ? 1 : 0;
}
