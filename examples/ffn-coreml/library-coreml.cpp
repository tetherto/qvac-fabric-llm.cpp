#include "llama.h"
#include "llama-coreml-ffn.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s model.gguf cache-directory prompt\n", argv[0]);
        return 1;
    }
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { return 1; }
    auto cp = llama_context_default_params();
    cp.n_ctx = 4096;
    cp.n_batch = cp.n_ubatch = 512;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { llama_model_free(model); return 1; }
    auto params = llama_coreml_ffn_default_params();
    params.cache_dir = argv[2];
    char error[512];
    int result = 0;
    if (!llama_set_coreml_ffn(ctx, &params, error, sizeof(error))) {
        fprintf(stderr, "Core ML FFN: %s\n", error);
        result = 1;
    } else {
        const auto * vocab = llama_model_get_vocab(model);
        const int length = int(strlen(argv[3]));
        int count = -llama_tokenize(vocab, argv[3], length, nullptr, 0, true, false);
        std::vector<llama_token> tokens(count > 0 ? count : 0);
        if (count <= 0 || count > 4000 || llama_tokenize(vocab, argv[3], length, tokens.data(), count, true, false) != count) {
            fprintf(stderr, "invalid prompt length\n");
            result = 1;
        } else {
            for (int offset = 0; offset < count; offset += 512) {
                if (llama_decode(ctx, llama_batch_get_one(tokens.data() + offset, std::min(512, count - offset))) != 0) {
                    result = 1;
                    break;
                }
            }
            for (int i = 0; !result && i < 64; ++i) {
                const float * logits = llama_get_logits_ith(ctx, -1);
                const int size = llama_vocab_n_tokens(vocab);
                llama_token token = std::max_element(logits, logits + size) - logits;
                if (llama_vocab_is_eog(vocab, token)) { break; }
                char piece[256];
                const int n = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, true);
                if (n > 0) { fwrite(piece, 1, n, stdout); fflush(stdout); }
                if (llama_decode(ctx, llama_batch_get_one(&token, 1)) != 0) { result = 1; }
            }
            printf("\n");
        }
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return result;
}
