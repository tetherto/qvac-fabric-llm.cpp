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
#include "llama.h"

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
    std::printf("straight: %s\nrestored: %s\ntest-xdna-state: %s\n", sw.c_str(), sg.c_str(), want == got ? "passed" : "FAILED");
    llama_model_free(model);
    llama_backend_free();
    return want == got ? 0 : 1;
}
