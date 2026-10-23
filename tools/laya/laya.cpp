// Typed decisions with Laya checkpoints (https://github.com/NandhaKishorM/laya)
//
// Reads a request {"state" | "states", "questions"} and prints the answers in the format of
// laya's Agent.predict / predict_batch. See common/laya.h.

#include "arg.h"
#include "common.h"
#include "laya.h"
#include "log.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <clocale>
#include <cstdio>
#include <stdexcept>

using json = nlohmann::ordered_json;

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m laya.gguf -f request.json\n", argv[0]);
    LOG("\n    %s -m laya.gguf -p '{\"state\": \"My payment failed twice\", \"questions\": {\"department\": {\"type\": \"choice\", \"instructions\": \"Route the ticket\", \"criteria\": [\"billing\", \"technical\"]}}}'\n", argv[0]);
    LOG("\nthe request is {\"state\": ..., \"questions\": {...}} or {\"states\": [...], \"questions\": {...}},\n");
    LOG("with optional \"max_len\" and \"head_max_len\" overrides\n");
    LOG("\n");
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    // tokens per forward pass; every sequence has to fit
    params.n_batch = 2048;

    // the request is JSON, keep its escapes
    params.escape = false;

    // the default warmup batch is not a laya sequence, see common_laya_warmup
    params.warmup = false;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_EMBEDDING, print_usage)) {
        return 1;
    }

    // as common_laya_context_params
    params.embedding    = true;
    params.pooling_type = LLAMA_POOLING_TYPE_RANK;
    params.n_ctx        = params.n_batch;
    params.n_ubatch     = params.n_batch;
    if (params.n_parallel == 1) {
        params.n_parallel = llama_max_parallel_sequences();
        params.kv_unified = true;
    }

    json request;
    try {
        request = json::parse(params.prompt);
    } catch (const std::exception & e) {
        LOG_ERR("%s: the request (-p or -f) is not valid JSON: %s\n", __func__, e.what());
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);

    llama_context * ctx = llama_init->context();
    if (ctx == nullptr) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }

    try {
        const common_laya_ptr laya = common_laya_init(ctx);

        common_laya_warmup(laya.get());

        const common_laya_result res = common_laya_predict(laya.get(), request);

        if (params.verbose_prompt) {
            for (const auto & seq : res.sequences) {
                const json row = {
                    { "state",    seq.state },
                    { "question", seq.question },
                    { "ids",      seq.tokens },
                    { "markers",  seq.markers },
                    { "act",      seq.act },
                    { "logits",   seq.logits },
                };
                fprintf(stderr, "laya_row: %s\n", row.dump().c_str());
            }
        }

        LOG_INF("%s: %zu sequences, %d tokens, %d forward passes in %.2f ms (%.1f sequences/s)\n",
                __func__, res.sequences.size(), res.n_tokens, res.n_passes, res.t_ms, 1000.0 * res.sequences.size() / res.t_ms);

        printf("%s\n", res.response.dump(2, ' ', false, json::error_handler_t::replace).c_str());
    } catch (const std::exception & e) {
        LOG_ERR("%s: %s\n", __func__, e.what());
        return 1;
    }

    llama_backend_free();

    return 0;
}
