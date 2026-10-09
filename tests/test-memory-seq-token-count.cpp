#include "arg.h"
#include "common.h"
#include "llama.h"

#include <cstdio>
#include <vector>

static bool decode_prompt(llama_context * ctx, const std::vector<llama_token> & tokens) {
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        common_batch_add(batch, tokens[i], i, { 0 }, i == tokens.size() - 1);
    }

    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_ctx     = 128;
    params.n_batch   = 32;
    params.n_ubatch  = 32;
    params.n_predict = 1;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model *          model      = llama_init->model();
    llama_context *        ctx        = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "%s : failed to init\n", __func__);
        return 1;
    }

    llama_memory_t mem = llama_get_memory(ctx);
    if (mem == nullptr) {
        fprintf(stderr, "%s : failed to get memory\n", __func__);
        return 1;
    }

    GGML_ASSERT(llama_memory_seq_token_count(mem, 0) == 0);
    GGML_ASSERT(llama_memory_seq_token_count(mem, -1) == 0);

    std::vector<llama_token> tokens = common_tokenize(ctx, "The quick brown fox", true);
    if (tokens.size() < 2) {
        fprintf(stderr, "%s : prompt tokenized to fewer than two tokens\n", __func__);
        return 1;
    }

    if (!decode_prompt(ctx, tokens)) {
        fprintf(stderr, "%s : failed to decode prompt\n", __func__);
        return 1;
    }

    const uint32_t n_tokens = static_cast<uint32_t>(tokens.size());
    GGML_ASSERT(llama_memory_seq_token_count(mem, 0) == n_tokens);
    GGML_ASSERT(llama_memory_seq_token_count(mem, -1) == n_tokens);
    GGML_ASSERT(llama_memory_seq_pos_max(mem, 0) == static_cast<llama_pos>(n_tokens - 1));

    if (!llama_memory_seq_rm(mem, 0, static_cast<llama_pos>(n_tokens - 1), -1)) {
        fprintf(stderr, "%s : failed to remove tail token\n", __func__);
        return 1;
    }

    GGML_ASSERT(llama_memory_seq_token_count(mem, 0) == n_tokens - 1);
    GGML_ASSERT(llama_memory_seq_token_count(mem, -1) == n_tokens - 1);
    GGML_ASSERT(llama_memory_seq_pos_max(mem, 0) == static_cast<llama_pos>(n_tokens - 2));

    if (!llama_memory_seq_rm(mem, 0, -1, -1)) {
        fprintf(stderr, "%s : failed to clear sequence\n", __func__);
        return 1;
    }

    GGML_ASSERT(llama_memory_seq_token_count(mem, 0) == 0);
    GGML_ASSERT(llama_memory_seq_token_count(mem, -1) == 0);
    GGML_ASSERT(llama_memory_seq_pos_max(mem, 0) == -1);

    return 0;
}
